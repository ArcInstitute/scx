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

/// Compiled PTX for the FOR-BP decode kernel (produced by build.rs via nvcc --ptx).
const FORBP_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/forbp_decode.ptx"));

/// Per-row metadata extracted during CPU pre-parse.
struct RowMeta {
    frame_min: u32,
    frame_bits: u8,
    nnz: u32,
    /// Bit offset within `data` where packed deltas start for this row.
    bit_offset: u32,
    /// Start index in the flat output array for this row's indices.
    output_offset: u32,
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

            metas.push(RowMeta {
                frame_min,
                frame_bits,
                nnz: nnz as u32,
                bit_offset,
                output_offset,
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
/// and `took_host_fallback` (`true` when a SIMD-packed row forced the host
/// reference decoder + HtoD instead of the GPU kernel). Row lengths are returned
/// on CPU since they're needed for CSR construction. The `took_host_fallback`
/// flag lets the profiler avoid double-counting the host span (which this
/// function already records into `host_decode_scx1`/`htod_scx1`) inside the
/// caller's `gpu_decode` bucket.
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

    if total_nnz == 0 {
        return Ok((dev.alloc_zeros::<u32>(0)?, all_row_lengths, false));
    }

    // The encoder bit-packs any row with nnz >= SIMD_THRESHOLD using BitPacker4x's
    // SIMD layout (scx_codec::forbp::forbp_encode), which differs from the scalar
    // LSB-first packing the GPU kernel below assumes. The kernel only decodes the
    // scalar layout correctly, so when ANY row uses the SIMD path (ubiquitous in
    // real single-cell data — cells expressing >=128 genes) we decode the indices
    // on the host via the reference decoder (correct for both layouts) and upload.
    // The GPU kernel fast path is retained for all-sparse-row shards.
    if all_row_lengths
        .iter()
        .any(|&n| n >= scx_codec::forbp::SIMD_THRESHOLD)
    {
        let t_decode = crate::profile::start();
        let (indices, row_lengths) = scx_codec::forbp::forbp_decode(data, n_rows, index_dtype_u16)
            .map_err(|e| GpuError::InvalidShard(format!("FOR-BP host decode: {e:?}")))?;
        crate::profile::record_host_decode_since(crate::profile::CodecClass::Scx1, t_decode);

        let t_htod = crate::profile::start();
        let d_indices = dev.htod_copy(&indices)?;
        crate::profile::record_htod_since(
            crate::profile::CodecClass::Scx1,
            t_htod,
            indices.len() * 4,
        );
        return Ok((d_indices, row_lengths, true));
    }

    // Load PTX module (cached) and get kernel function
    let module = dev.load_module_cached(FORBP_PTX)?;
    let kernel = module
        .load_function("forbp_decode_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("load forbp_decode_kernel: {e}")))?;

    let n_nonempty = metas.len() as u32;
    let bitstream_len = data.len() as u32;

    // Build struct-of-arrays from metas
    let bit_offsets: Vec<u32> = metas.iter().map(|m| m.bit_offset).collect();
    let frame_mins: Vec<u32> = metas.iter().map(|m| m.frame_min).collect();
    let frame_bits: Vec<u8> = metas.iter().map(|m| m.frame_bits).collect();
    let nnzs: Vec<u32> = metas.iter().map(|m| m.nnz).collect();
    let out_offsets: Vec<u32> = metas.iter().map(|m| m.output_offset).collect();

    // Upload to GPU
    let d_bitstream = dev.htod_copy(data)?;
    let d_bit_offsets = dev.htod_copy(&bit_offsets)?;
    let d_frame_mins = dev.htod_copy(&frame_mins)?;
    let d_frame_bits = dev.htod_copy(&frame_bits)?;
    let d_nnzs = dev.htod_copy(&nnzs)?;
    let d_out_offsets = dev.htod_copy(&out_offsets)?;
    let mut d_output = dev.alloc_zeros::<u32>(total_nnz)?;

    // Launch: one thread per non-empty row, CUDA blocks of 256 threads
    let threads_per_cuda_block = 256u32;
    let grid_dim = n_nonempty.div_ceil(threads_per_cuda_block);
    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (threads_per_cuda_block, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        dev.stream()
            .launch_builder(&kernel)
            .arg(&d_bitstream)
            .arg(&d_bit_offsets)
            .arg(&d_frame_mins)
            .arg(&d_frame_bits)
            .arg(&d_nnzs)
            .arg(&d_out_offsets)
            .arg(&mut d_output)
            .arg(&n_nonempty)
            .arg(&bitstream_len)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("forbp_decode_kernel: {e}")))?;

    Ok((d_output, all_row_lengths, false))
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
}
