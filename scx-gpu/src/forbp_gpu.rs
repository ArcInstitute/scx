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

/// The largest FOR-BP sub-stream this decoder accepts, in bytes.
///
/// `RowMeta::bit_offset` is `cursor.position() * 8` in a `u32`, and both kernels
/// consume it — and every offset they derive from it (`cur_bit`, `base_byte`,
/// `chunk_base`) — as 32-bit. At `u32::MAX / 8` bytes the product stops fitting:
/// every row past that point would decode from a wrapped bit position while the
/// CPU decoder reads the same file correctly. Framed shards pass one row group
/// at a time and never approach it; the unframed `decode_scx1_gpu` path (legacy
/// v1 files, or `--row-group-rows 0`) passes the whole sub-stream, and an
/// ATAC-like modality reaches ~1 GB in a single shard.
///
/// Rejecting is deliberate rather than widening the kernels' whole bit domain to
/// 64-bit: the host decoder reads these files correctly, and `scx optimize`
/// re-frames the file permanently. The same ceiling makes `data.len() as u32`
/// (the kernels' `bitstream_len` bound) lossless.
///
/// The rejection is a [`GpuError::UnsupportedLayout`], which is what actually
/// routes `to_gpu_anndata` to host-assemble. It was `InvalidShard` first, and
/// that silently did **not** degrade — `alternate_route_may_succeed()` is false
/// for an input defect, so the call raised instead. Classifying it correctly is
/// the whole of the graceful part; the variant is not cosmetic.
pub(crate) const MAX_FORBP_SUBSTREAM_BYTES: usize = (u32::MAX / 8) as usize;

/// Reject a FOR-BP sub-stream whose byte length makes [`RowMeta::bit_offset`]
/// unrepresentable.
///
/// A free function over the *length* rather than an inline check on the slice,
/// so the boundary is testable without materialising half a gigabyte: the
/// guard a test exercises here is the one `preparse_forbp` actually calls.
pub(crate) fn check_forbp_substream_len(len: usize) -> Result<(), GpuError> {
    if len > MAX_FORBP_SUBSTREAM_BYTES {
        // `UnsupportedLayout`, not `InvalidShard`: the shard is **valid**, and the
        // CPU decoder reads it correctly — it is this route that cannot address
        // it. That is the distinction `GpuError::UnsupportedLayout` documents
        // ("a path that was never viable"), and it is what lets
        // `alternate_route_may_succeed` send `to_gpu_anndata` to host-assemble
        // rather than raising. Classifying a valid shard as corrupt would be both
        // wrong and user-hostile.
        return Err(GpuError::UnsupportedLayout(format!(
            "FOR-BP prescan: sub-stream is {len} bytes, above the \
             {MAX_FORBP_SUBSTREAM_BYTES}-byte GPU ceiling (the decode kernels address it \
             with 32-bit bit offsets); re-frame the file with \
             `scx optimize --row-group-rows 256`, or decode it on the CPU"
        )));
    }
    Ok(())
}

/// Pre-parse the FOR-BP encoded stream on CPU to extract per-row metadata.
///
/// Returns `(row_metas, all_row_lengths, total_nnz)` where:
/// - `row_metas` contains metadata for non-empty rows only (sent to GPU)
/// - `all_row_lengths` contains nnz for every row including empty ones (for CSR)
/// - `total_nnz` is the sum of all row nnz values
///
/// `nnz_hint` is the declared nnz the caller expects (`0` = unknown), and it is
/// the bound on a zero-payload run — see the `frame_bits == 0` arm below. It
/// mirrors `scx_codec::forbp::forbp_decode_with_hint`'s parameter of the same
/// name; this prescan is the third copy of that per-row parse.
fn preparse_forbp(
    data: &[u8],
    n_rows: usize,
    nnz_hint: usize,
    index_dtype_u16: bool,
) -> Result<(Vec<RowMeta>, Vec<usize>, usize), GpuError> {
    check_forbp_substream_len(data.len())?;
    // Both bounds are `forbp_decode_inner`'s (`scx-codec/src/forbp.rs`): every
    // row writes at least one varint byte even when empty, and an index needs at
    // least one bit. They bound `Vec::with_capacity` there and the cumulative
    // output here, before either can be driven by a hostile header.
    if n_rows > data.len() {
        return Err(GpuError::InvalidShard(format!(
            "FOR-BP prescan: n_rows={n_rows} exceeds the {} stream bytes that could encode them",
            data.len()
        )));
    }
    if data.len().checked_mul(8).is_some_and(|cap| nnz_hint > cap) {
        return Err(GpuError::InvalidShard(format!(
            "FOR-BP prescan: nnz_hint={nnz_hint} exceeds the {} bits the stream carries",
            data.len() * 8
        )));
    }
    // Ceiling on total decoded indices: exact when the caller declared the nnz
    // (both production call sites do), else the codec's absolute no-hint cap.
    let max_output = if nnz_hint > 0 {
        nnz_hint
    } else {
        scx_codec::forbp::FORBP_NO_HINT_MAX_NNZ
    };
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

            // `frame_bits` is a u8 straight off an untrusted stream, and it
            // reaches the CUDA kernel as a shift width. A delta is a `u32`, so
            // anything above 32 is unrepresentable and the stream is corrupt.
            // Both CPU decoders reject it (`forbp.rs`'s streaming decode, added
            // by the fuzz regression in #333); this prescan is the third copy of
            // the same per-row parse and was the one without the guard. Per-shard
            // checksum verification is opt-in on the read path, so "the catalog
            // would have caught it" does not hold here either.
            if frame_bits > 32 {
                return Err(GpuError::InvalidShard(format!(
                    "FOR-BP prescan: frame_bits={frame_bits} exceeds 32"
                )));
            }

            // Bound the *cumulative* decoded index count, on **every** non-empty
            // row. `forbp_decode_inner` applies this only to the zero-payload
            // (`frame_bits == 0`) case, and that is sound there because the CPU
            // pushes into a `Vec` as it decodes: its growth is bounded by the
            // pushes it actually makes, so a lying `nnz` on a positive-width row
            // costs nothing until the payload runs out.
            //
            // This decoder cannot borrow that reasoning. It sums the stream's own
            // per-row varints and hands the total to `alloc_zeros(total_nnz)`
            // *before* any decoding, so a `frame_bits == 1` row carrying real
            // payload bytes can pass the `need_bytes > remaining` check below
            // while claiming an `nnz` the shard header never declared — up to
            // `remaining * 8` of them. At the substream ceiling that is ~4.29e9
            // indices, a ~17 GB VRAM request, and `check_device_len` reports the
            // mismatch only afterwards. Gating this on `frame_bits == 0` left
            // exactly that hole open (found by codex).
            if (output_offset as usize).saturating_add(nnz) > max_output {
                return Err(GpuError::InvalidShard(format!(
                    "FOR-BP prescan: a row of {nnz} (frame_bits={frame_bits}) would take \
                     the decoded index count past the declared bound of {max_output}"
                )));
            }

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

            // Checked: `nnz` is an untrusted varint, and `output_offset` is the
            // kernel's per-row *write base* into a buffer sized from this same
            // running total. A wrap here is an out-of-bounds device write, not
            // merely a short allocation. No cheap test reaches it — the bounds
            // above cap the sum at `data.len() * 8`, so it takes a stream within
            // a factor of two of `MAX_FORBP_SUBSTREAM_BYTES` to cross `u32::MAX`
            // — but that stream is exactly the one the unframed path accepts.
            output_offset = u32::try_from(nnz)
                .ok()
                .and_then(|n| output_offset.checked_add(n))
                .ok_or_else(|| {
                    GpuError::InvalidShard(format!(
                        "FOR-BP prescan: decoded index count overflows u32 at a row of {nnz}"
                    ))
                })?;
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
/// Returns `(CudaSlice<u32>, Vec<usize>)` — GPU indices and CPU row_lengths. Both
/// packing layouts decode on the device (scalar via `forbp_decode_kernel`,
/// BitPacker4x via `forbp_decode_bp4x_kernel`, Task 4.4b). Row lengths are
/// returned on CPU since they're needed for CSR construction.
///
/// Produces **bit-identical** output to `scx_codec::forbp::forbp_decode`.
pub fn forbp_decode_gpu(
    dev: &GpuDevice,
    data: &[u8],
    n_rows: usize,
    index_dtype_u16: bool,
) -> Result<(CudaSlice<u32>, Vec<usize>), GpuError> {
    forbp_decode_gpu_with_hint(dev, data, n_rows, 0, index_dtype_u16)
}

/// Decode FOR-BP encoded column indices on GPU against a **declared** nnz.
///
/// The twin of [`forbp_decode_gpu`] that knows what the shard header says the
/// answer should be, mirroring `scx_codec::forbp::forbp_decode_with_hint`. The
/// hint is a bound, never an output size: the decode still derives its length
/// from the stream's own per-row varints, and the caller still compares the two
/// (`check_device_len` / `CombinedCsr::place`). What it buys is that a
/// zero-payload run — the one shape that carries no payload bytes to bound it —
/// is rejected during the prescan rather than after a multi-GB `alloc_zeros`.
///
/// Output is **bit-identical** to [`forbp_decode_gpu`] for any stream both accept.
pub fn forbp_decode_gpu_with_hint(
    dev: &GpuDevice,
    data: &[u8],
    n_rows: usize,
    nnz_hint: usize,
    index_dtype_u16: bool,
) -> Result<(CudaSlice<u32>, Vec<usize>), GpuError> {
    // CPU pre-parse all headers
    let (metas, all_row_lengths, total_nnz) =
        preparse_forbp(data, n_rows, nnz_hint, index_dtype_u16)?;
    forbp_decode_gpu_core(dev, data, metas, all_row_lengths, total_nnz)
}

/// Shared FOR-BP GPU decode core: takes the per-row metadata (from the CPU
/// `preparse_forbp` pass), partitions rows by their
/// packing layout, and launches the matching kernel for each group — both
/// writing into the same device output buffer at each row's global
/// `output_offset`. No host fallback: the scalar (`index_packing == 1`) and
/// BitPacker4x (`index_packing == 2`, Task 4.4b) kernels together cover every
/// row, so the indices stay on the device. Output is **bit-identical** to
/// `scx_codec::forbp::forbp_decode`.
fn forbp_decode_gpu_core(
    dev: &GpuDevice,
    data: &[u8],
    mut metas: Vec<RowMeta>,
    all_row_lengths: Vec<usize>,
    total_nnz: usize,
) -> Result<(CudaSlice<u32>, Vec<usize>), GpuError> {
    if total_nnz == 0 {
        return Ok((dev.alloc_zeros::<u32>(0)?, all_row_lengths));
    }

    // Lossless: `preparse_forbp` rejected anything above `MAX_FORBP_SUBSTREAM_BYTES`
    // (< 2^29), so this narrowing cannot truncate. Before that bound it could,
    // which would have made the kernels' only bounds check spuriously loose.
    let bitstream_len = data.len() as u32;
    let d_bitstream = dev.htod_copy(data)?;
    let mut d_output = dev.alloc_zeros::<u32>(total_nnz)?;

    // Partition non-empty rows by packing layout (in place — rows write disjoint
    // output_offset slots, so reordering is safe and avoids two temp Vecs). The
    // scalar kernel decodes the contiguous LSB-first stream (index_packing == 1);
    // the BitPacker4x kernel decodes the SIMD chunks (index_packing == 2,
    // nnz >= 128). Both index d_output by each row's global output_offset.
    metas.sort_by_key(|m| m.index_packing == 2); // scalar (false) first, bp4x (true) last
    let split = metas.partition_point(|m| m.index_packing != 2);
    let (scalar, bp4x) = metas.split_at(split);

    if !scalar.is_empty() {
        launch_forbp_group(
            dev,
            FORBP_PTX,
            "forbp_decode_kernel",
            &d_bitstream,
            scalar,
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
            bp4x,
            &mut d_output,
            bitstream_len,
        )?;
    }

    Ok((d_output, all_row_lengths))
}

/// Build the per-row struct-of-arrays for `metas` and launch `kernel_name` (from
/// `ptx`) over them, one thread per row, writing into the shared `d_output`.
/// Both FOR-BP kernels share this arg layout.
fn launch_forbp_group(
    dev: &GpuDevice,
    ptx: &'static str,
    kernel_name: &str,
    d_bitstream: &CudaSlice<u8>,
    metas: &[RowMeta],
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

    /// A `frame_bits` above 32 is unrepresentable for `u32` deltas and must be
    /// rejected here, not forwarded to the kernel as a shift width.
    ///
    /// Needs no GPU: the prescan is host-side. That is the point — the two CPU
    /// decoders in `scx-codec` have carried this guard since the #333 fuzz
    /// regression, and this third copy of the same per-row parse did not.
    #[test]
    fn preparse_rejects_frame_bits_above_32() {
        let rows = vec![vec![0u32, 5, 10, 20]];
        let (indices, row_lengths) = flatten(&rows);
        let mut encoded = forbp_encode(&indices, &row_lengths, true).unwrap();

        // Single-row u16-index block: u32 block_nnz, u16 n_rows_in_block, a
        // 1-byte nnz varint (nnz = 4), u16 frame_min, then frame_bits.
        const FRAME_BITS_OFF: usize = 4 + 2 + 1 + 2;

        // Premises, so a layout change fails here rather than making the
        // corruption below land on some unrelated byte and pass vacuously.
        assert!(
            (1..=32).contains(&encoded[FRAME_BITS_OFF]),
            "fixture drifted: byte {FRAME_BITS_OFF} is {}, not a plausible frame_bits",
            encoded[FRAME_BITS_OFF]
        );
        assert!(
            preparse_forbp(&encoded, 1, 0, true).is_ok(),
            "premise: the fixture parses before corruption"
        );

        encoded[FRAME_BITS_OFF] = 47;
        let Err(err) = preparse_forbp(&encoded, 1, 0, true) else {
            panic!("frame_bits=47 must be rejected, not handed to the kernel");
        };
        // Match the guard's own message, not merely `is_err()`. Without the
        // guard this input still errors — as "truncated packed deltas", because
        // 47 × 4 bits overruns the buffer — so an `is_err()` assertion would
        // pass against the unguarded code and prove nothing.
        assert!(
            matches!(&err, GpuError::InvalidShard(m) if m.contains("frame_bits=47")),
            "expected the frame_bits guard, got {err:?}"
        );

        // And the case that actually reaches the kernel: pad the stream so the
        // corrupted delta-skip stays in bounds and the truncation check cannot
        // fire at all. Unguarded, this parses "successfully" and forwards
        // frame_bits=47 to the GPU as a shift width.
        let mut padded = encoded.clone();
        padded.extend(std::iter::repeat_n(0u8, 64));
        let Err(err) = preparse_forbp(&padded, 1, 0, true) else {
            panic!("frame_bits=47 within a padded buffer must still be rejected");
        };
        assert!(
            matches!(&err, GpuError::InvalidShard(m) if m.contains("frame_bits=47")),
            "expected the frame_bits guard, got {err:?}"
        );
    }

    /// LEB128, the width `scx_codec::forbp::read_varint` consumes.
    fn varint(mut v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let byte = (v & 0x7F) as u8;
            v >>= 7;
            if v == 0 {
                out.push(byte);
                return out;
            }
            out.push(byte | 0x80);
        }
    }

    /// One u16-index block holding a single row whose nnz varint claims
    /// `claimed_nnz` and whose `frame_bits` is 0 — a zero-payload run.
    ///
    /// Hand-built rather than encoder-produced: `forbp_encode` will not emit a
    /// row that lies about its own length, and the lie is the whole fixture.
    fn zero_payload_row(claimed_nnz: u64) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&0u32.to_le_bytes()); // block_nnz (unread by the prescan)
        out.extend_from_slice(&1u16.to_le_bytes()); // n_rows_in_block
        out.extend_from_slice(&varint(claimed_nnz)); // per-row nnz
        out.extend_from_slice(&7u16.to_le_bytes()); // frame_min
        out.push(0); // frame_bits == 0 -> no payload follows
        out
    }

    /// A `frame_bits == 0` row carries no payload, so the `need_bytes > remaining`
    /// reject cannot see it: an nnz varint of half a billion parses out of ~14
    /// bytes and reaches `alloc_zeros(total_nnz)` as a multi-GB VRAM request.
    ///
    /// The CPU decoder has bounded the cumulative output since its own fuzz
    /// regression (`forbp_decode_inner`'s `max_output`); this prescan did not.
    /// Needs no GPU — the prescan is host-side.
    #[test]
    fn preparse_rejects_an_unbounded_zero_frame_bits_run() {
        // Premise: a *modest* zero-payload run still parses, so the rejection
        // below is about the bound and not about zero-payload rows as such.
        let small = zero_payload_row(4);
        let (_, row_lengths, total) =
            preparse_forbp(&small, 1, 0, true).expect("a bounded zero-payload run still parses");
        assert_eq!(
            row_lengths,
            vec![4],
            "premise: the fixture decodes as one row"
        );
        assert_eq!(total, 4, "premise: its nnz is the varint's value");

        let hostile = zero_payload_row(500_000_000);
        assert!(
            hostile.len() < 32,
            "premise: the lie costs {} bytes, so nothing bounds it by payload",
            hostile.len()
        );

        // No hint: the codec's absolute no-hint cap applies.
        let Err(err) = preparse_forbp(&hostile, 1, 0, true) else {
            panic!("an unbounded zero-payload run must be rejected before it sizes an allocation");
        };
        assert!(
            matches!(&err, GpuError::InvalidShard(m)
                if m.contains("frame_bits=0") && m.contains("bound of 16777216")),
            "expected the cumulative-output bound, got {err:?}"
        );

        // With a declared nnz the bound is exact rather than absolute, which is
        // what both production call sites pass.
        let Err(err) = preparse_forbp(&hostile, 1, 4, true) else {
            panic!("a declared nnz of 4 must reject a claimed 500_000_000");
        };
        assert!(
            matches!(&err, GpuError::InvalidShard(m) if m.contains("bound of 4")),
            "expected the declared bound in the message, got {err:?}"
        );
    }

    /// A `frame_bits > 0` row is bounded too, not just the zero-payload case.
    ///
    /// The CPU decoder bounds only the zero-payload arm, and that is sound there
    /// because it pushes into a `Vec` as it goes. This prescan sums the stream's
    /// varints and hands the total to `alloc_zeros` *before* decoding, so a row
    /// that carries real payload bytes while claiming a far larger `nnz` than the
    /// shard header declared reached the allocator unchecked. Found by codex on
    /// PR #549.
    #[test]
    fn preparse_bounds_a_positive_width_row_against_the_declared_nnz() {
        // One 1-bit-wide row claiming 4000 indices, with the 500 payload bytes
        // that claim genuinely requires — so the truncation check cannot catch
        // it. The header, however, declares 4.
        let claimed = 4_000usize;
        let mut hostile = Vec::new();
        hostile.extend_from_slice(&0u32.to_le_bytes()); // block_nnz
        hostile.extend_from_slice(&1u16.to_le_bytes()); // n_rows_in_block
        hostile.extend_from_slice(&varint(claimed as u64)); // per-row nnz
        hostile.extend_from_slice(&7u16.to_le_bytes()); // frame_min
        hostile.push(1); // frame_bits = 1  -> positive width
        hostile.resize(hostile.len() + claimed.div_ceil(8), 0); // real payload

        // Premise: the payload is present, so the truncation reject cannot fire
        // and an unbounded prescan accepts this row.
        assert!(
            preparse_forbp(&hostile, 1, 0, true).is_ok(),
            "premise: with no declared nnz the row is within the no-hint cap"
        );

        let Err(err) = preparse_forbp(&hostile, 1, 4, true) else {
            panic!("a positive-width row claiming {claimed} against a declared 4 must be rejected");
        };
        assert!(
            matches!(&err, GpuError::InvalidShard(m)
                if m.contains("frame_bits=1") && m.contains("bound of 4")),
            "expected the cumulative bound to name the positive-width row, got {err:?}"
        );
    }

    /// `RowMeta::bit_offset` is `cursor.position() * 8` in a `u32`. Past
    /// `u32::MAX / 8` bytes every row decodes from a wrapped bit position while
    /// the CPU reads the same file correctly — so the prescan refuses the
    /// sub-stream instead.
    ///
    /// Exercises the guard through the function `preparse_forbp` calls, over a
    /// length rather than a slice, so the boundary is checked without
    /// materialising half a gigabyte.
    #[test]
    fn preparse_rejects_a_substream_past_the_bit_offset_ceiling() {
        assert_eq!(
            MAX_FORBP_SUBSTREAM_BYTES, 536_870_911,
            "premise: the ceiling is u32::MAX / 8"
        );
        // The last accepted length must still multiply into a u32 bit offset.
        assert!(
            u32::try_from(MAX_FORBP_SUBSTREAM_BYTES as u64 * 8).is_ok(),
            "premise: the ceiling is the largest length whose bit offset fits"
        );
        assert!(
            check_forbp_substream_len(MAX_FORBP_SUBSTREAM_BYTES).is_ok(),
            "the boundary length itself is still decodable"
        );

        let Err(err) = check_forbp_substream_len(MAX_FORBP_SUBSTREAM_BYTES + 1) else {
            panic!("one byte past the ceiling must be rejected, not silently wrapped");
        };
        assert!(
            matches!(&err, GpuError::UnsupportedLayout(m) if m.contains("32-bit bit offsets")),
            "expected the bit-offset ceiling, got {err:?}"
        );
        // The variant is load-bearing, not cosmetic: it is what routes a valid
        // oversized shard to a host decode instead of raising at the caller.
        assert!(
            err.alternate_route_may_succeed(),
            "an oversized-but-valid shard must be answerable by another route"
        );
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_forbp_gpu_vs_cpu_basic() {
        let dev = require_gpu!();
        let rows = vec![vec![0u32, 5, 10, 20], vec![1, 3, 7], vec![100, 200]];
        let (indices, row_lengths) = flatten(&rows);
        let encoded = forbp_encode(&indices, &row_lengths, true).unwrap();
        let (cpu_indices, cpu_row_lengths) =
            forbp_decode(&encoded, row_lengths.len(), true).unwrap();

        let (d_output, gpu_row_lengths) =
            forbp_decode_gpu(&dev, &encoded, row_lengths.len(), true).unwrap();
        let gpu_indices = dev.dtoh_copy(&d_output).unwrap();

        assert_eq!(gpu_indices, cpu_indices, "GPU indices must match CPU");
        assert_eq!(
            gpu_row_lengths, cpu_row_lengths,
            "GPU row_lengths must match CPU"
        );
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
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

        let (d_output, gpu_row_lengths) =
            forbp_decode_gpu(&dev, &encoded, row_lengths.len(), true).unwrap();
        let gpu_indices = dev.dtoh_copy(&d_output).unwrap();

        assert_eq!(gpu_indices, cpu_indices);
        assert_eq!(gpu_row_lengths, cpu_row_lengths);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
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

        let (d_output, gpu_row_lengths) =
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
    /// (including the >= 128-nnz BitPacker4x rows decoded on-device — Task 4.4b).
    fn assert_forbp_roundtrip(dev: &GpuDevice, rows: &[Vec<u32>], index_dtype_u16: bool) {
        let (indices, row_lengths) = flatten(rows);
        let encoded = forbp_encode(&indices, &row_lengths, index_dtype_u16).unwrap();
        let (cpu_indices, cpu_row_lengths) =
            forbp_decode(&encoded, row_lengths.len(), index_dtype_u16).unwrap();

        let (d_output, gpu_row_lengths) =
            forbp_decode_gpu(dev, &encoded, row_lengths.len(), index_dtype_u16).unwrap();
        let gpu_indices = dev.dtoh_copy(&d_output).unwrap();

        assert_eq!(gpu_indices, cpu_indices, "GPU indices must match host");
        assert_eq!(gpu_row_lengths, cpu_row_lengths);
    }

    /// Dense rows (>= 128 nnz → BitPacker4x) at every chunk boundary + remainder,
    /// interleaved with empty and sparse (scalar-kernel) rows; u16 indices.
    #[test]
    #[ignore = "requires a CUDA GPU"]
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
    #[ignore = "requires a CUDA GPU"]
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
    #[ignore = "requires a CUDA GPU"]
    fn test_forbp_gpu_bp4x_frame_bits_zero_dense() {
        let dev = require_gpu!();
        let rows = vec![
            vec![7u32; 200],        // all-equal → frame_bits == 0, scalar kernel
            dense_row(256, 32, 21), // fb > 0 → BitPacker4x kernel
        ];
        assert_forbp_roundtrip(&dev, &rows, true);
    }
}
