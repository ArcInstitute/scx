//! GPU-accelerated FOR-BP (Frame of Reference + Bit Packing) decoder.
//!
//! Wraps the `forbp_decode_kernel` CUDA kernel with a CPU pre-parse stage
//! that extracts per-row metadata (frame_min, frame_bits, nnz, bit offsets)
//! from the encoded block headers. The GPU then decodes rows in parallel
//! (one thread per non-empty row).

use std::io::Cursor;

use byteorder::{LittleEndian, ReadBytesExt};
use cudarc::driver::safe::{CudaSlice, LaunchConfig};
use cudarc::driver::PushKernelArg;

use crate::device::GpuDevice;
use crate::error::GpuError;

/// Compiled PTX for the scalar (index_packing == 1) FOR-BP decode kernel.
const FORBP_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/forbp_decode.ptx"));

/// Compiled PTX for the BitPacker4x (index_packing == 2) FOR-BP decode kernel
/// (Task 4.4b) — decodes the SIMD layout the encoder uses for rows >= 128 nnz.
const FORBP_BP4X_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/forbp_decode_bp4x.ptx"));

/// Per-row metadata extracted during CPU pre-parse.
struct RowMeta {
    frame_min: u32,
    frame_bits: u8,
    nnz: u32,
    /// Bit offset within `data` where packed deltas start for this row.
    bit_offset: u32,
    /// Start index in the flat output array for this row's indices.
    output_offset: u32,
    /// Index packing layout: `1` = scalar LSB-first, `2` = BitPacker4x SIMD
    /// (`>= 128` nnz). Selects which GPU kernel decodes the row.
    index_packing: u8,
}

/// Pre-parse the FOR-BP encoded stream on CPU to extract per-row metadata.
///
/// Returns `(row_metas, all_row_lengths, total_nnz)` where:
/// - `row_metas` contains metadata for non-empty rows only (sent to GPU)
/// - `all_row_lengths` contains nnz for every row including empty ones (for CSR)
/// - `total_nnz` is the sum of all row nnz values
fn preparse_forbp(
    data: &[u8],
    n_rows: usize,
    index_dtype_u16: bool,
) -> Result<(Vec<RowMeta>, Vec<usize>, usize), GpuError> {
    let mut cursor = Cursor::new(data);
    let mut rows_remaining = n_rows;
    let mut metas = Vec::new();
    let mut all_row_lengths = Vec::with_capacity(n_rows);
    let mut output_offset: u32 = 0;

    while rows_remaining > 0 {
        // Block header: u32 LE block_nnz + u16 LE n_rows_in_block
        let _block_nnz = cursor
            .read_u32::<LittleEndian>()
            .map_err(|e| GpuError::InvalidShard(format!("FOR-BP prescan: block header: {e}")))?;
        let n_rows_in_block = cursor
            .read_u16::<LittleEndian>()
            .map_err(|e| GpuError::InvalidShard(format!("FOR-BP prescan: n_rows_in_block: {e}")))?
            as usize;

        // Read per-row nnz as LEB128 varints
        let pos = cursor.position() as usize;
        let mut varint_slice = &data[pos..];
        let mut row_nnzs = Vec::with_capacity(n_rows_in_block);
        for _ in 0..n_rows_in_block {
            let nnz = scx_codec::forbp::read_varint(&mut varint_slice)
                .map_err(|e| GpuError::InvalidShard(format!("FOR-BP prescan: varint: {e}")))?
                as usize;
            row_nnzs.push(nnz);
        }
        let varints_consumed = data[pos..].len() - varint_slice.len();
        cursor.set_position((pos + varints_consumed) as u64);

        // Parse per-row frame_min, frame_bits, and record bit offsets
        for &nnz in &row_nnzs {
            all_row_lengths.push(nnz);

            if nnz == 0 {
                continue;
            }

            // Read frame_min
            let frame_min = if index_dtype_u16 {
                cursor.read_u16::<LittleEndian>().map_err(|e| {
                    GpuError::InvalidShard(format!("FOR-BP prescan: frame_min: {e}"))
                })? as u32
            } else {
                cursor.read_u32::<LittleEndian>().map_err(|e| {
                    GpuError::InvalidShard(format!("FOR-BP prescan: frame_min: {e}"))
                })?
            };

            // Read frame_bits
            let frame_bits = cursor
                .read_u8()
                .map_err(|e| GpuError::InvalidShard(format!("FOR-BP prescan: frame_bits: {e}")))?;

            // Record the bit offset where packed deltas start
            let bit_offset = (cursor.position() as u32) * 8;

            // Skip past the packed deltas
            if frame_bits > 0 {
                let total_bits = frame_bits as usize * nnz;
                let total_bytes = total_bits.div_ceil(8);
                let new_pos = cursor.position() as usize + total_bytes;
                if new_pos > data.len() {
                    return Err(GpuError::InvalidShard(
                        "FOR-BP prescan: truncated packed deltas".into(),
                    ));
                }
                cursor.set_position(new_pos as u64);
            }

            // Mirror the encoder's choice (scx_codec::forbp::forbp_encode):
            // BitPacker4x SIMD layout iff frame_bits > 0 && nnz >= SIMD_THRESHOLD.
            let index_packing = if frame_bits > 0 && nnz >= scx_codec::forbp::SIMD_THRESHOLD {
                2
            } else {
                1
            };

            metas.push(RowMeta {
                frame_min,
                frame_bits,
                nnz: nnz as u32,
                bit_offset,
                output_offset,
                index_packing,
            });

            output_offset += nnz as u32;
        }

        if n_rows_in_block > rows_remaining {
            return Err(GpuError::InvalidShard(
                "FOR-BP prescan: n_rows_in_block exceeds remaining rows".into(),
            ));
        }
        rows_remaining -= n_rows_in_block;
    }

    Ok((metas, all_row_lengths, output_offset as usize))
}

/// Decode FOR-BP encoded column indices on GPU.
///
/// Returns `(CudaSlice<u32>, Vec<usize>, bool)` — GPU indices, CPU row_lengths,
/// and `took_host_fallback`. As of Task 4.4b both packing layouts decode on the
/// device (scalar via `forbp_decode_kernel`, BitPacker4x via
/// `forbp_decode_bp4x_kernel`), so `took_host_fallback` is always `false`; the
/// field is retained for the caller's profiler bucketing. Row lengths are
/// returned on CPU since they're needed for CSR construction.
///
/// Produces **bit-identical** output to `scx_codec::forbp::forbp_decode`.
pub fn forbp_decode_gpu(
    dev: &GpuDevice,
    data: &[u8],
    n_rows: usize,
    index_dtype_u16: bool,
) -> Result<(CudaSlice<u32>, Vec<usize>, bool), GpuError> {
    // CPU pre-parse all headers
    let (metas, all_row_lengths, total_nnz) = preparse_forbp(data, n_rows, index_dtype_u16)?;
    forbp_decode_gpu_core(dev, data, metas, all_row_lengths, total_nnz)
}

/// FOR-BP GPU decode driven by an encoder-emitted decode-metadata sidecar
/// ([`scx_codec::forbp::ForBpRowMetadata`]) instead of the CPU `preparse_forbp`
/// pass. The sidecar already stores exactly what the kernel needs (per-row
/// `indices_bit_offset`, `frame_min`, `frame_bits`, `nnz`, and `value_start` =
/// the output offset), so the full CPU scan of the bitstream is skipped.
///
/// `rows` must have one entry per CSR row (`rows.len() == n_rows`), including
/// empty rows. Output is **bit-identical** to [`forbp_decode_gpu`]; rows are
/// routed to the scalar or BitPacker4x kernel by their `index_packing` (Task
/// 4.4b), so the indices decode entirely on the device.
pub fn forbp_decode_gpu_with_metadata(
    dev: &GpuDevice,
    data: &[u8],
    rows: &[scx_codec::forbp::ForBpRowMetadata],
    n_rows: usize,
) -> Result<(CudaSlice<u32>, Vec<usize>, bool), GpuError> {
    if rows.len() != n_rows {
        return Err(GpuError::InvalidShard(format!(
            "FOR-BP sidecar: rows.len() {} != n_rows {n_rows}",
            rows.len()
        )));
    }
    let mut metas = Vec::with_capacity(rows.len());
    let mut all_row_lengths = Vec::with_capacity(rows.len());
    let mut total_nnz: usize = 0;
    for r in rows {
        let nnz = r.nnz as usize;
        all_row_lengths.push(nnz);
        total_nnz += nnz;
        if nnz == 0 {
            continue;
        }
        metas.push(RowMeta {
            frame_min: r.frame_min,
            frame_bits: r.frame_bits,
            nnz: r.nnz,
            bit_offset: u32::try_from(r.indices_bit_offset).map_err(|_| {
                GpuError::InvalidShard(format!(
                    "FOR-BP sidecar: indices_bit_offset {} exceeds u32",
                    r.indices_bit_offset
                ))
            })?,
            output_offset: u32::try_from(r.value_start).map_err(|_| {
                GpuError::InvalidShard(format!(
                    "FOR-BP sidecar: value_start {} exceeds u32",
                    r.value_start
                ))
            })?,
            index_packing: r.index_packing,
        });
    }
    forbp_decode_gpu_core(dev, data, metas, all_row_lengths, total_nnz)
}

/// Shared FOR-BP GPU decode core: takes the per-row metadata (from either the
/// CPU `preparse_forbp` pass or a decode sidecar), partitions rows by their
/// packing layout, and launches the matching kernel for each group — both
/// writing into the same device output buffer at each row's global
/// `output_offset`. No host fallback: the scalar (`index_packing == 1`) and
/// BitPacker4x (`index_packing == 2`, Task 4.4b) kernels together cover every
/// row, so the indices stay on the device (`took_host_fallback` is always
/// `false`). Output is **bit-identical** to `scx_codec::forbp::forbp_decode`.
fn forbp_decode_gpu_core(
    dev: &GpuDevice,
    data: &[u8],
    metas: Vec<RowMeta>,
    all_row_lengths: Vec<usize>,
    total_nnz: usize,
) -> Result<(CudaSlice<u32>, Vec<usize>, bool), GpuError> {
    if total_nnz == 0 {
        return Ok((dev.alloc_zeros::<u32>(0)?, all_row_lengths, false));
    }

    let bitstream_len = data.len() as u32;
    let d_bitstream = dev.htod_copy(data)?;
    let mut d_output = dev.alloc_zeros::<u32>(total_nnz)?;

    // Partition non-empty rows by packing layout. The scalar kernel decodes the
    // contiguous LSB-first stream (index_packing == 1); the BitPacker4x kernel
    // decodes the SIMD chunks (index_packing == 2, nnz >= 128). Both index
    // d_output by each row's global output_offset, so the two passes compose.
    let scalar: Vec<&RowMeta> = metas.iter().filter(|m| m.index_packing != 2).collect();
    let bp4x: Vec<&RowMeta> = metas.iter().filter(|m| m.index_packing == 2).collect();

    if !scalar.is_empty() {
        launch_forbp_group(
            dev,
            FORBP_PTX,
            "forbp_decode_kernel",
            &d_bitstream,
            &scalar,
            &mut d_output,
            bitstream_len,
        )?;
    }
    if !bp4x.is_empty() {
        launch_forbp_group(
            dev,
            FORBP_BP4X_PTX,
            "forbp_decode_bp4x_kernel",
            &d_bitstream,
            &bp4x,
            &mut d_output,
            bitstream_len,
        )?;
    }

    Ok((d_output, all_row_lengths, false))
}

/// Build the per-row struct-of-arrays for `metas` and launch `kernel_name` (from
/// `ptx`) over them, one thread per row, writing into the shared `d_output`.
/// Both FOR-BP kernels share this arg layout.
fn launch_forbp_group(
    dev: &GpuDevice,
    ptx: &'static str,
    kernel_name: &str,
    d_bitstream: &CudaSlice<u8>,
    metas: &[&RowMeta],
    d_output: &mut CudaSlice<u32>,
    bitstream_len: u32,
) -> Result<(), GpuError> {
    let module = dev.load_module_cached(ptx)?;
    let kernel = module
        .load_function(kernel_name)
        .map_err(|e| GpuError::KernelLaunchFailed(format!("load {kernel_name}: {e}")))?;

    let n = metas.len() as u32;
    let bit_offsets: Vec<u32> = metas.iter().map(|m| m.bit_offset).collect();
    let frame_mins: Vec<u32> = metas.iter().map(|m| m.frame_min).collect();
    let frame_bits: Vec<u8> = metas.iter().map(|m| m.frame_bits).collect();
    let nnzs: Vec<u32> = metas.iter().map(|m| m.nnz).collect();
    let out_offsets: Vec<u32> = metas.iter().map(|m| m.output_offset).collect();

    let d_bit_offsets = dev.htod_copy(&bit_offsets)?;
    let d_frame_mins = dev.htod_copy(&frame_mins)?;
    let d_frame_bits = dev.htod_copy(&frame_bits)?;
    let d_nnzs = dev.htod_copy(&nnzs)?;
    let d_out_offsets = dev.htod_copy(&out_offsets)?;

    // Launch: one thread per row, CUDA blocks of 256 threads.
    let threads_per_cuda_block = 256u32;
    let grid_dim = n.div_ceil(threads_per_cuda_block);
    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (threads_per_cuda_block, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        dev.stream()
            .launch_builder(&kernel)
            .arg(d_bitstream)
            .arg(&d_bit_offsets)
            .arg(&d_frame_mins)
            .arg(&d_frame_bits)
            .arg(&d_nnzs)
            .arg(&d_out_offsets)
            .arg(&mut *d_output)
            .arg(&n)
            .arg(&bitstream_len)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("{kernel_name}: {e}")))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_codec::forbp::{forbp_decode, forbp_encode};

    /// Helper to build flat indices and row_lengths from a Vec of Vec.
    fn flatten(rows: &[Vec<u32>]) -> (Vec<u32>, Vec<usize>) {
        let indices: Vec<u32> = rows.iter().flatten().copied().collect();
        let row_lengths: Vec<usize> = rows.iter().map(|r| r.len()).collect();
        (indices, row_lengths)
    }

    #[test]
    fn test_forbp_gpu_vs_cpu_basic() {
        let dev = require_gpu!();
        let rows = vec![vec![0u32, 5, 10, 20], vec![1, 3, 7], vec![100, 200]];
        let (indices, row_lengths) = flatten(&rows);
        let encoded = forbp_encode(&indices, &row_lengths, true).unwrap();
        let (cpu_indices, cpu_row_lengths) =
            forbp_decode(&encoded, row_lengths.len(), true).unwrap();

        let (d_output, gpu_row_lengths, _took_host_fallback) =
            forbp_decode_gpu(&dev, &encoded, row_lengths.len(), true).unwrap();
        let gpu_indices = dev.dtoh_copy(&d_output).unwrap();

        assert_eq!(gpu_indices, cpu_indices, "GPU indices must match CPU");
        assert_eq!(
            gpu_row_lengths, cpu_row_lengths,
            "GPU row_lengths must match CPU"
        );
    }

    #[test]
    fn test_forbp_gpu_vs_cpu_variable_nnz() {
        let dev = require_gpu!();
        // 140 rows spanning 2 FOR-BP blocks (128 + 12), mix of empty and non-empty
        let mut rows = Vec::new();
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        for i in 0..140 {
            if i % 5 == 0 {
                // Empty row
                rows.push(vec![]);
            } else {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let nnz = (state % 10 + 1) as usize;
                let mut row = Vec::with_capacity(nnz);
                let mut prev = 0u32;
                for _ in 0..nnz {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    prev += (state % 50 + 1) as u32;
                    if prev > 65535 {
                        prev = 65535;
                    }
                    row.push(prev);
                }
                rows.push(row);
            }
        }

        let (indices, row_lengths) = flatten(&rows);
        let encoded = forbp_encode(&indices, &row_lengths, true).unwrap();
        let (cpu_indices, cpu_row_lengths) =
            forbp_decode(&encoded, row_lengths.len(), true).unwrap();

        let (d_output, gpu_row_lengths, _took_host_fallback) =
            forbp_decode_gpu(&dev, &encoded, row_lengths.len(), true).unwrap();
        let gpu_indices = dev.dtoh_copy(&d_output).unwrap();

        assert_eq!(gpu_indices, cpu_indices);
        assert_eq!(gpu_row_lengths, cpu_row_lengths);
    }

    #[test]
    fn test_forbp_gpu_vs_cpu_u16_indices() {
        let dev = require_gpu!();
        // Rows with u16 indices including values near the u16 max (65535)
        let rows = vec![
            vec![0u32, 100, 65535],
            vec![1000, 32000, 65000],
            vec![0],
            vec![],
            vec![65534, 65535],
        ];

        let (indices, row_lengths) = flatten(&rows);
        let encoded = forbp_encode(&indices, &row_lengths, true).unwrap();
        let (cpu_indices, cpu_row_lengths) =
            forbp_decode(&encoded, row_lengths.len(), true).unwrap();

        let (d_output, gpu_row_lengths, _took_host_fallback) =
            forbp_decode_gpu(&dev, &encoded, row_lengths.len(), true).unwrap();
        let gpu_indices = dev.dtoh_copy(&d_output).unwrap();

        assert_eq!(gpu_indices, cpu_indices);
        assert_eq!(gpu_row_lengths, cpu_row_lengths);
    }

    // ---- Task 4.4b: BitPacker4x (>= 128-nnz) GPU decode parity ----

    /// Build a strictly-increasing dense row of length `nnz` with gaps in
    /// `1..=max_gap` (so `frame_bits ≈ bits_needed(max_gap)`).
    fn dense_row(nnz: usize, max_gap: u32, seed: u64) -> Vec<u32> {
        let mut state = seed | 1;
        let mut v = Vec::with_capacity(nnz);
        let mut col = 0u32;
        for _ in 0..nnz {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            col = col.saturating_add(1 + (state % max_gap.max(1) as u64) as u32);
            v.push(col);
        }
        v
    }

    /// Dense row whose second delta needs `bits` bits (the rest are +1), to
    /// exercise the `frame_bits ∈ {31, 32}` edges of the BitPacker4x kernel.
    fn dense_row_big_delta(nnz: usize, big: u32) -> Vec<u32> {
        let mut v = Vec::with_capacity(nnz);
        v.push(0);
        v.push(big);
        let mut col = big;
        for _ in 2..nnz {
            col += 1;
            v.push(col);
        }
        v
    }

    /// Encode `rows`, then assert GPU decode == host `forbp_decode` byte-for-byte
    /// and that the decode stayed on the device (no host fallback — Task 4.4b).
    fn assert_forbp_roundtrip(dev: &GpuDevice, rows: &[Vec<u32>], index_dtype_u16: bool) {
        let (indices, row_lengths) = flatten(rows);
        let encoded = forbp_encode(&indices, &row_lengths, index_dtype_u16).unwrap();
        let (cpu_indices, cpu_row_lengths) =
            forbp_decode(&encoded, row_lengths.len(), index_dtype_u16).unwrap();

        let (d_output, gpu_row_lengths, took_host_fallback) =
            forbp_decode_gpu(dev, &encoded, row_lengths.len(), index_dtype_u16).unwrap();
        let gpu_indices = dev.dtoh_copy(&d_output).unwrap();

        assert!(
            !took_host_fallback,
            "4.4b: FOR-BP indices must decode on the device, never host-fall-back"
        );
        assert_eq!(gpu_indices, cpu_indices, "GPU indices must match host");
        assert_eq!(gpu_row_lengths, cpu_row_lengths);
    }

    /// Dense rows (>= 128 nnz → BitPacker4x) at every chunk boundary + remainder,
    /// interleaved with empty and sparse (scalar-kernel) rows; u16 indices.
    #[test]
    fn test_forbp_gpu_bp4x_dense_u16() {
        let dev = require_gpu!();
        let rows = vec![
            dense_row(128, 200, 1), // exactly one chunk, no remainder
            vec![],
            dense_row(129, 50, 2),  // one chunk + 1 remainder
            dense_row(256, 150, 3), // two chunks
            dense_row(383, 64, 4),  // two chunks + 127 remainder
            vec![3, 9, 40],         // sparse → scalar kernel
            dense_row(200, 16, 5),
            dense_row(512, 7, 6),
        ];
        assert_forbp_roundtrip(&dev, &rows, true);
    }

    /// One dense (nnz = 256) row per target `frame_bits` (1, 7, 16, 31, 32),
    /// u32 indices — covers the no-span (fb = 32) and two-word-span paths.
    #[test]
    fn test_forbp_gpu_bp4x_frame_bits_sweep_u32() {
        let dev = require_gpu!();
        let rows = vec![
            (0..256u32).collect::<Vec<_>>(),   // fb = 1 (all deltas = 1)
            dense_row(256, 120, 11),           // fb ≈ 7
            dense_row(256, 60_000, 12),        // fb ≈ 16
            dense_row_big_delta(256, 1 << 30), // fb = 31
            dense_row_big_delta(256, 1 << 31), // fb = 32
        ];
        assert_forbp_roundtrip(&dev, &rows, false);
    }

    /// A dense (>= 128) row whose deltas are all zero (`frame_bits == 0`, all
    /// indices equal) — routed to the scalar kernel (index_packing == 1), which
    /// previously host-fell-back at >= 128 nnz. Plus a dense fb > 0 neighbour.
    #[test]
    fn test_forbp_gpu_bp4x_frame_bits_zero_dense() {
        let dev = require_gpu!();
        let rows = vec![
            vec![7u32; 200],        // all-equal → frame_bits == 0, scalar kernel
            dense_row(256, 32, 21), // fb > 0 → BitPacker4x kernel
        ];
        assert_forbp_roundtrip(&dev, &rows, true);
    }
}
