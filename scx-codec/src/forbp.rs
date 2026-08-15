// FOR-BP encoder/decoder for indices (docs/codec.md (Indices))

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Cursor;

use bitpacking::{BitPacker, BitPacker4x};

use crate::bitstream::{BitStreamError, BitWriter};
use crate::dispatch::CodecError;

/// Block size for FOR-BP coding of indices: 128 rows per block.
pub const B_IDX: usize = 128;

/// Minimum NNZ per row to use BitPacker4x SIMD path.
/// BitPacker4x processes 128 values per call.
///
/// Public so consumers that re-implement the bit-unpacking (e.g. the scx-gpu
/// FOR-BP decoder) can detect rows packed with the SIMD layout — which differs
/// from the scalar LSB-first remainder packing — and route those correctly.
pub const SIMD_THRESHOLD: usize = BitPacker4x::BLOCK_LEN;

// ---------------------------------------------------------------------------
// LEB128 varint helpers (task 5.2)
// ---------------------------------------------------------------------------

/// Write unsigned LEB128 varint to a byte buffer.
pub fn write_varint(writer: &mut Vec<u8>, mut value: u32) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        writer.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Read unsigned LEB128 varint from a byte slice, advancing the slice.
pub fn read_varint(reader: &mut &[u8]) -> Result<u32, BitStreamError> {
    let mut result: u32 = 0;
    let mut shift: u32 = 0;
    loop {
        if reader.is_empty() {
            return Err(BitStreamError);
        }
        let byte = reader[0];
        *reader = &reader[1..];
        if shift == 28 {
            // 5th byte: only 4 value bits remain (bits 28..31). Any continuation
            // bit or value bit above bit 3 overflows u32 → malformed input.
            if byte > 0x0F {
                return Err(BitStreamError);
            }
            return Ok(result | ((byte as u32) << shift));
        }
        result |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute ceil(log2(max_val + 1)). Returns 0 for max_val == 0.
#[inline]
fn bits_needed(max_val: u32) -> u8 {
    if max_val == 0 {
        0
    } else {
        32 - max_val.leading_zeros() as u8
    }
}

// ---------------------------------------------------------------------------
// Encoder (task 5.3)
// ---------------------------------------------------------------------------

/// Encode CSR column indices using FOR-BP coding.
///
/// `indices` is the flat array of column indices (sorted within each row).
/// `row_lengths` gives the number of non-zero entries per row.
/// `index_dtype_u16` is true when the largest index `n_vars - 1` fits u16 —
/// i.e. up to 65_536 columns (frame_min written as u16).
pub fn forbp_encode(
    indices: &[u32],
    row_lengths: &[usize],
    index_dtype_u16: bool,
) -> Result<Vec<u8>, CodecError> {
    let mut output = Vec::new();
    let mut idx_offset: usize = 0;

    for block_rows in row_lengths.chunks(B_IDX) {
        let n_rows_in_block = block_rows.len();
        let block_nnz: usize = block_rows.iter().sum();

        // Validate block_nnz fits in u32
        if block_nnz > u32::MAX as usize {
            return Err(BitStreamError.into());
        }

        // Write block header
        output.write_u32::<LittleEndian>(block_nnz as u32).unwrap();
        output
            .write_u16::<LittleEndian>(n_rows_in_block as u16)
            .unwrap();

        // Write per-row nnz as LEB128 varints
        for &nnz in block_rows {
            write_varint(&mut output, nnz as u32);
        }

        // Encode each row's indices
        for &nnz in block_rows {
            if nnz > u32::MAX as usize {
                return Err(BitStreamError.into());
            }
            if nnz == 0 {
                continue;
            }

            let row_indices = &indices[idx_offset..idx_offset + nnz];

            // frame_min is the first (smallest) index in the sorted row
            let frame_min = row_indices[0];

            // Compute deltas: delta[0] = 0, delta[j] = indices[j] - indices[j-1].
            // Reject unsorted indices loudly — silent wrap on `u32 - u32`
            // produces unreadable shards in release.
            let mut deltas = Vec::with_capacity(nnz);
            deltas.push(0u32);
            for j in 1..nnz {
                if row_indices[j] < row_indices[j - 1] {
                    return Err(CodecError::MalformedInput(format!(
                        "FOR-BP requires sorted row indices; saw {} < {} at offset {j}",
                        row_indices[j],
                        row_indices[j - 1]
                    )));
                }
                deltas.push(row_indices[j] - row_indices[j - 1]);
            }

            let max_delta = deltas.iter().copied().max().unwrap_or(0);
            let frame_bits = bits_needed(max_delta);

            // Write frame_min
            if index_dtype_u16 {
                if frame_min > u16::MAX as u32 {
                    return Err(BitStreamError.into());
                }
                output.write_u16::<LittleEndian>(frame_min as u16).unwrap();
            } else {
                output.write_u32::<LittleEndian>(frame_min).unwrap();
            }

            // Write frame_bits
            output.push(frame_bits);

            // Bit-pack deltas if frame_bits > 0
            if frame_bits > 0 {
                if nnz >= SIMD_THRESHOLD {
                    // SIMD path: BitPacker4x for 128-value chunks
                    let packer = BitPacker4x::new();
                    let full_chunks = nnz / SIMD_THRESHOLD;
                    let remainder = nnz % SIMD_THRESHOLD;
                    let chunk_bytes = frame_bits as usize * SIMD_THRESHOLD / 8;
                    let mut compressed = vec![0u8; chunk_bytes];
                    for c in 0..full_chunks {
                        let start = c * SIMD_THRESHOLD;
                        packer.compress(
                            &deltas[start..start + SIMD_THRESHOLD],
                            &mut compressed,
                            frame_bits,
                        );
                        output.extend_from_slice(&compressed);
                    }
                    // Remainder: scalar BitWriter
                    if remainder > 0 {
                        let mut bw = BitWriter::new();
                        for &d in &deltas[full_chunks * SIMD_THRESHOLD..] {
                            bw.write_bits(d as u64, frame_bits);
                        }
                        output.extend_from_slice(&bw.flush());
                    }
                } else {
                    // Scalar path for small rows
                    let mut bw = BitWriter::new();
                    for &d in &deltas {
                        bw.write_bits(d as u64, frame_bits);
                    }
                    output.extend_from_slice(&bw.flush());
                }
            }

            idx_offset += nnz;
        }
    }

    if idx_offset != indices.len() {
        return Err(CodecError::MalformedInput(format!(
            "FOR-BP row lengths cover {idx_offset} indices, but input has {}",
            indices.len()
        )));
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Batch bit-unpacking
// ---------------------------------------------------------------------------

/// Bulk-extract `count` fixed-width integers from a packed byte stream.
///
/// Reads u64 words from `src` starting at `bit_offset` bits, extracting
/// `count` values of `bits` width each into `dst`. This replaces the
/// per-value `BitReader::read_bits()` loop with word-at-a-time extraction,
/// processing up to `floor(64 / bits)` values per u64 word — the exact
/// count per word is recomputed from the running bit position, since a
/// value straddling a word boundary leaves fewer than `floor(64 / bits)`
/// fully-contained values in the current word.
///
/// # Safety / correctness
/// - `dst.len()` must be `>= count`
/// - `src` must contain enough bytes for `bit_offset + count * bits` bits
/// - `bits` must be in 1..=32
#[inline]
fn unpack_fixed_width(src: &[u8], bit_offset: usize, count: usize, bits: u8, dst: &mut [u32]) {
    debug_assert!((1..=32).contains(&bits));
    debug_assert!(dst.len() >= count);

    let mask: u64 = (1u64 << bits) - 1;
    let mut bit_pos = bit_offset;
    let mut i = 0;

    // Main loop: process values by reading u64 words from the byte stream.
    // Each word can yield multiple values, reducing loop iterations.
    while i < count {
        let byte_idx = bit_pos >> 3; // bit_pos / 8
        let bit_idx = (bit_pos & 7) as u32; // bit_pos % 8

        // Read up to 8 bytes starting at byte_idx as a LE u64.
        // Handle the case where we're near the end of src.
        let available = src.len() - byte_idx;
        let word = if available >= 8 {
            // Fast path: read 8 bytes directly
            u64::from_le_bytes(src[byte_idx..byte_idx + 8].try_into().unwrap())
        } else {
            // Near end: read available bytes into a zero-padded u64
            let mut buf = [0u8; 8];
            buf[..available].copy_from_slice(&src[byte_idx..]);
            u64::from_le_bytes(buf)
        };

        // Shift down to align the first value in this word
        let mut w = word >> bit_idx;
        // How many bits are available in this word after the initial offset
        let bits_available = 64 - bit_idx;
        // How many complete values fit in the remaining bits
        let values_in_word = (bits_available / bits as u32) as usize;
        let to_extract = (count - i).min(values_in_word);

        for _ in 0..to_extract {
            dst[i] = (w & mask) as u32;
            w >>= bits;
            i += 1;
        }
        bit_pos += to_extract * bits as usize;
    }
}

// ---------------------------------------------------------------------------
// Decoder (task 5.4)
// ---------------------------------------------------------------------------

/// Decode FOR-BP encoded indices.
///
/// Returns `(indices, row_lengths)` where `indices` is the flat column index array
/// and `row_lengths` gives per-row nnz counts.
pub fn forbp_decode(
    data: &[u8],
    n_rows: usize,
    index_dtype_u16: bool,
) -> Result<(Vec<u32>, Vec<usize>), BitStreamError> {
    forbp_decode_inner(data, n_rows, 0, index_dtype_u16)
}

/// Decode FOR-BP encoded indices against a **declared** nnz.
///
/// When `nnz_hint > 0` it is both a pre-allocation size and a contract: the
/// stream's per-row nnz varints must sum to exactly `nnz_hint`, or decode
/// fails. That check is what keeps a corrupt shard from producing an `indices`
/// array shorter than its own `indptr.last()` — see `check_decoded_shape` in
/// `dispatch.rs`. Pass `0` (or use [`forbp_decode`]) when there is no external
/// count to check against; the length is then whatever the stream says.
pub fn forbp_decode_with_hint(
    data: &[u8],
    n_rows: usize,
    nnz_hint: usize,
    index_dtype_u16: bool,
) -> Result<(Vec<u32>, Vec<usize>), BitStreamError> {
    forbp_decode_inner(data, n_rows, nnz_hint, index_dtype_u16)
}

/// Cap on total decoded indices when the caller passes no `nnz_hint` (tests /
/// benches / fuzz; the production Scx1 path always passes the exact nnz via
/// [`forbp_decode_with_hint`]). A `frame_bits == 0` row is a zero-payload
/// constant run whose nnz the stream does not otherwise bound, so this caps the
/// allocation a hostile no-hint input can drive.
///
/// `1 << 24` (16.7M indices ≈ 64 MiB of i32 output) is chosen to sit far above
/// any realistic no-hint decode — the largest such caller is the codec bench,
/// well under a million indices — while keeping a hostile constant-run bounded
/// to tens of MiB rather than the multi-GiB OOM the old `nnz <= 1` guard blocked.
const FORBP_NO_HINT_MAX_NNZ: usize = 1 << 24;

fn forbp_decode_inner(
    data: &[u8],
    n_rows: usize,
    nnz_hint: usize,
    index_dtype_u16: bool,
) -> Result<(Vec<u32>, Vec<usize>), BitStreamError> {
    // F-f: bound both `with_capacity` args to what `data` could physically
    // encode before allocating, so a hostile `nnz_hint` or `n_rows` can't drive
    // an eager multi-GiB allocation. Guards direct callers of this primitive;
    // the Scx1 decode path also bounds these upstream with a `MalformedInput`
    // message. Indices need ≥1 bit each (`data.len() * 8`, via `checked_mul` so
    // a huge `data.len()` on 32-bit can't saturate the bound and be bypassed);
    // each row writes ≥1 varint byte even when empty (`forbp_encode`), so the
    // far tighter `n_rows ≤ data.len()` holds for any valid stream.
    if data.len().checked_mul(8).is_some_and(|cap| nnz_hint > cap) || n_rows > data.len() {
        return Err(BitStreamError);
    }
    // Ceiling on total decoded indices: exact when the caller knows the nnz
    // (production Scx1 path), else a generous absolute cap. Bounds the one
    // otherwise-unbounded allocation — a zero-payload `frame_bits == 0` run.
    let max_output = if nnz_hint > 0 {
        nnz_hint
    } else {
        FORBP_NO_HINT_MAX_NNZ
    };
    let mut all_indices = Vec::with_capacity(nnz_hint);
    let mut all_row_lengths = Vec::with_capacity(n_rows);
    let mut cursor = Cursor::new(data);
    let mut rows_remaining = n_rows;

    while rows_remaining > 0 {
        // Read block header
        let _block_nnz = cursor
            .read_u32::<LittleEndian>()
            .map_err(|_| BitStreamError)?;
        let n_rows_in_block = cursor
            .read_u16::<LittleEndian>()
            .map_err(|_| BitStreamError)? as usize;

        // Read per-row nnz varints from the remaining bytes
        let pos = cursor.position() as usize;
        let remaining_bytes = &data[pos..];
        let mut varint_slice = remaining_bytes;
        let mut row_nnzs = Vec::with_capacity(n_rows_in_block);
        for _ in 0..n_rows_in_block {
            row_nnzs.push(read_varint(&mut varint_slice)? as usize);
        }
        // Advance cursor past the varints we consumed
        let varints_consumed = remaining_bytes.len() - varint_slice.len();
        cursor.set_position((pos + varints_consumed) as u64);

        // Decode each row: batch-extract deltas, then prefix-sum
        for &nnz in &row_nnzs {
            all_row_lengths.push(nnz);
            if nnz == 0 {
                continue;
            }

            // Read frame_min
            let frame_min = if index_dtype_u16 {
                cursor
                    .read_u16::<LittleEndian>()
                    .map_err(|_| BitStreamError)? as u32
            } else {
                cursor
                    .read_u32::<LittleEndian>()
                    .map_err(|_| BitStreamError)?
            };

            // Read frame_bits
            let frame_bits = cursor.read_u8().map_err(|_| BitStreamError)?;

            // Reject an out-of-range frame width. Indices are u32, so a valid
            // frame packs at most 32 bits per delta; `unpack_fixed_width`
            // documents `bits in 1..=32` as a precondition (fuzz: frame_bits=47
            // tripped its debug_assert / produced a bad shift). The framed
            // metadata decode path applies the same guard.
            if frame_bits > 32 {
                return Err(BitStreamError);
            }

            // Bound this row's nnz *before* it drives an allocation
            // (`resize`/`push` below). A malformed nnz varint could otherwise
            // request gigabytes from a few-byte input (fuzz: a per-row nnz of
            // ~5.4e8 forced a 2 GiB allocation).
            if frame_bits == 0 {
                // A frame_bits == 0 row is a zero-payload run of equal indices:
                // the encoder only requires non-decreasing indices, so repeated
                // columns encode as all-zero deltas (valid for any nnz — real
                // CSR rows have distinct columns and never hit this, but the
                // contract permits it). It is decoded by the `else` branch below
                // as `frame_min` repeated `nnz` times. Since it carries no
                // payload, bound the *cumulative* output instead of payload
                // bytes so a hostile nnz varint can't drive an unbounded run.
                if all_indices.len().saturating_add(nnz) > max_output {
                    return Err(BitStreamError);
                }
            } else {
                // A frame_bits-packed row needs at least `nnz * frame_bits` bits
                // of payload; reject a length the remaining bytes can't hold.
                let need_bytes = (nnz as u64).saturating_mul(frame_bits as u64).div_ceil(8);
                let remaining = (data.len() as u64).saturating_sub(cursor.position());
                if need_bytes > remaining {
                    return Err(BitStreamError);
                }
            }

            if frame_bits > 0 {
                let start = all_indices.len();
                all_indices.resize(start + nnz, 0);

                if nnz >= SIMD_THRESHOLD {
                    // SIMD path: BitPacker4x for 128-value chunks
                    let packer = BitPacker4x::new();
                    let full_chunks = nnz / SIMD_THRESHOLD;
                    let remainder = nnz % SIMD_THRESHOLD;
                    let chunk_bytes = frame_bits as usize * SIMD_THRESHOLD / 8;
                    let pos = cursor.position() as usize;
                    let mut data_offset = pos;

                    for c in 0..full_chunks {
                        let dst_start = start + c * SIMD_THRESHOLD;
                        if data_offset + chunk_bytes > data.len() {
                            return Err(BitStreamError);
                        }
                        packer.decompress(
                            &data[data_offset..],
                            &mut all_indices[dst_start..dst_start + SIMD_THRESHOLD],
                            frame_bits,
                        );
                        data_offset += chunk_bytes;
                    }

                    // Remainder: scalar unpack
                    if remainder > 0 {
                        let rem_start = start + full_chunks * SIMD_THRESHOLD;
                        let rem_bits = remainder * frame_bits as usize;
                        let rem_bytes = rem_bits.div_ceil(8);
                        if data_offset + rem_bytes > data.len() {
                            return Err(BitStreamError);
                        }
                        unpack_fixed_width(
                            &data[data_offset..data_offset + rem_bytes],
                            0,
                            remainder,
                            frame_bits,
                            &mut all_indices[rem_start..],
                        );
                        data_offset += rem_bytes;
                    }

                    cursor.set_position(data_offset as u64);
                } else {
                    // Scalar path for small rows
                    let total_bits = frame_bits as usize * nnz;
                    let total_bytes = total_bits.div_ceil(8);
                    let pos = cursor.position() as usize;
                    if pos + total_bytes > data.len() {
                        return Err(BitStreamError);
                    }
                    unpack_fixed_width(
                        &data[pos..pos + total_bytes],
                        0,
                        nnz,
                        frame_bits,
                        &mut all_indices[start..],
                    );
                    cursor.set_position((pos + total_bytes) as u64);
                }

                // Prefix-sum to reconstruct absolute indices
                let mut prev = frame_min;
                for idx in &mut all_indices[start..start + nnz] {
                    prev = prev.checked_add(*idx).ok_or(BitStreamError)?;
                    *idx = prev;
                }
            } else {
                // frame_bits == 0: all indices equal frame_min
                for _ in 0..nnz {
                    all_indices.push(frame_min);
                }
            }
        }

        if n_rows_in_block > rows_remaining {
            return Err(BitStreamError);
        }
        rows_remaining -= n_rows_in_block;
    }

    // The output length is driven entirely by the stream's own per-row nnz
    // varints, so a corrupt stream decodes a different count than the caller
    // declared and the shard's three arrays silently disagree (the sibling
    // `delta_golomb_decode` / `rice_decode` both return exactly the requested
    // count). When the caller stated an nnz, it is a contract, not a hint.
    if nnz_hint > 0 && all_indices.len() != nnz_hint {
        return Err(BitStreamError);
    }

    Ok((all_indices, all_row_lengths))
}

// ---------------------------------------------------------------------------
// Tests (tasks 5.5–5.13)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build flat indices and row_lengths from a Vec of Vec.
    fn flatten(rows: &[Vec<u32>]) -> (Vec<u32>, Vec<usize>) {
        let indices: Vec<u32> = rows.iter().flatten().copied().collect();
        let row_lengths: Vec<usize> = rows.iter().map(|r| r.len()).collect();
        (indices, row_lengths)
    }

    fn round_trip(rows: &[Vec<u32>], index_dtype_u16: bool) {
        let (indices, row_lengths) = flatten(rows);
        let encoded = forbp_encode(&indices, &row_lengths, index_dtype_u16).unwrap();
        let (dec_indices, dec_row_lengths) =
            forbp_decode(&encoded, row_lengths.len(), index_dtype_u16).unwrap();
        assert_eq!(dec_indices, indices);
        assert_eq!(dec_row_lengths, row_lengths);
    }

    // Regression (fuzz_forbp OOM): a per-row nnz varint of ~5.4e8 previously
    // drove a multi-GB `resize`/`push` before any payload bounds check, so a
    // 20-byte input tripped libFuzzer's 2 GiB out-of-memory limit. Decode must
    // reject the malformed length loudly, never allocate gigabytes.
    #[test]
    fn decode_rejects_oversized_nnz_without_oom() {
        // block_nnz=0x0d04, n_rows_in_block=2, first row nnz varint
        // (0xfe,0xff,0xff,0xff,0x01) => ~5.37e8, frame_min=0, frame_bits=0.
        let data: [u8; 20] = [
            0x04, 0x0d, 0x00, 0x00, 0x02, 0x00, 0xfe, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        for &u16dt in &[true, false] {
            for &n_rows in &[1usize, 10, 128] {
                assert!(forbp_decode(&data, n_rows, u16dt).is_err());
            }
        }
    }

    // Regression (fuzz_forbp deadly signal): a frame_bits byte of 47 reached
    // `unpack_fixed_width`, whose `bits in 1..=32` precondition (u32 indices
    // pack <= 32 bits/delta) panicked under fuzz debug-assertions. Decode must
    // reject an out-of-range frame width instead of panicking.
    #[test]
    fn decode_rejects_oversized_frame_bits() {
        // block_nnz=1, n_rows_in_block=1, row nnz varint=1, then frame_min and
        // a frame_bits byte of 0x2f (47) / 0x2c (44) depending on frame_min width.
        let data: [u8; 17] = [
            0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x2f, 0x02, 0x2c, 0x2c, 0x2c,
            0x2c, 0x00, 0x00,
        ];
        for &u16dt in &[true, false] {
            for &n_rows in &[1usize, 10, 128] {
                assert!(forbp_decode(&data, n_rows, u16dt).is_err());
            }
        }
    }

    /// A >=128-nnz row of *repeated* column indices encodes as frame_bits == 0
    /// (zero deltas, no payload). The encoder allows non-decreasing (not only
    /// strictly-increasing) indices, so the host decoder must accept
    /// frame_bits == 0 for nnz > 1 too — regression for a guard that previously
    /// only allowed nnz <= 1 (the GPU test
    /// `test_forbp_gpu_bp4x_frame_bits_zero_dense` caught it only on H100 sbatch
    /// runs; this reproduces it in plain `cargo test`, no GPU required).
    #[test]
    fn frame_bits_zero_repeated_indices_round_trips() {
        round_trip(&[vec![7u32; 200]], true);
        round_trip(&[vec![7u32; 200]], false);
        // Constant run interleaved with a strictly-increasing dense (SIMD) row,
        // mirroring the GPU test's two-row fixture.
        round_trip(&[vec![7u32; 200], (0u32..256).collect()], true);
    }

    // 5.5: Single row, multiple rows, empty rows
    #[test]
    fn single_row() {
        round_trip(&[vec![0, 5, 10, 20]], true);
    }

    #[test]
    fn multiple_rows() {
        round_trip(&[vec![1, 3, 7], vec![0, 2, 4, 6, 8], vec![100, 200]], true);
    }

    #[test]
    fn empty_rows() {
        round_trip(&[vec![], vec![1, 2, 3], vec![], vec![], vec![5]], true);
    }

    #[test]
    fn all_empty_rows() {
        round_trip(&[vec![], vec![], vec![]], true);
    }

    // 5.6: Row with single index (frame_bits=0)
    #[test]
    fn single_index_per_row() {
        round_trip(&[vec![42]], true);
    }

    #[test]
    fn multiple_single_index_rows() {
        round_trip(&[vec![0], vec![100], vec![65535]], true);
    }

    // 5.7: Full block (128 rows), partial last block
    #[test]
    fn full_block_128_rows() {
        let rows: Vec<Vec<u32>> = (0..128).map(|i| vec![i as u32, i as u32 + 1]).collect();
        round_trip(&rows, true);
    }

    #[test]
    fn partial_last_block() {
        // 130 rows = one full block of 128 + partial block of 2
        let rows: Vec<Vec<u32>> = (0..130).map(|i| vec![i as u32 * 2]).collect();
        round_trip(&rows, true);
    }

    #[test]
    fn exact_two_blocks() {
        let rows: Vec<Vec<u32>> = (0..256).map(|i| vec![i as u32]).collect();
        round_trip(&rows, true);
    }

    // 5.8: u16 indices and u32 indices
    #[test]
    fn u16_indices() {
        round_trip(&[vec![0, 100, 65535]], true);
    }

    #[test]
    fn u32_indices() {
        round_trip(&[vec![0, 100_000, 200_000]], false);
    }

    #[test]
    fn u32_large_indices() {
        round_trip(&[vec![0, 1_000_000, 2_000_000, 3_000_000]], false);
    }

    // 5.9: Dense row with consecutive indices
    #[test]
    fn dense_consecutive_indices() {
        let row: Vec<u32> = (0..1000).collect();
        // gaps are all 1, so frame_bits should be 1
        round_trip(&[row], true);
    }

    // 5.10: Sparse row with large gaps
    #[test]
    fn sparse_large_gaps_u16() {
        round_trip(&[vec![0, 10000, 30000]], true);
    }

    #[test]
    fn sparse_large_gaps_u32() {
        round_trip(&[vec![0, 10000, 30000, 100000]], false);
    }

    // 5.11: Random round-trip
    #[test]
    fn random_round_trip() {
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let mut next = || -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let n_rows = 200;
        let mut rows = Vec::with_capacity(n_rows);
        for _ in 0..n_rows {
            let nnz = (next() % 20) as usize;
            let mut row = Vec::with_capacity(nnz);
            let mut prev = 0u32;
            for _ in 0..nnz {
                prev += (next() % 50 + 1) as u32;
                row.push(prev);
            }
            rows.push(row);
        }
        round_trip(&rows, true);
    }

    #[test]
    fn random_round_trip_u32() {
        let mut state: u64 = 0xCAFE_BABE_DEAD_BEEF;
        let mut next = || -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let n_rows = 150;
        let mut rows = Vec::with_capacity(n_rows);
        for _ in 0..n_rows {
            let nnz = (next() % 15) as usize;
            let mut row = Vec::with_capacity(nnz);
            let mut prev = 0u32;
            for _ in 0..nnz {
                prev += (next() % 1000 + 1) as u32;
                row.push(prev);
            }
            rows.push(row);
        }
        round_trip(&rows, false);
    }

    // Corrupt/hostile shard: prefix-sum overflow must return Err, not panic
    // (scx-codec pins overflow-checks=true even in release).
    #[test]
    fn corrupt_overflow_returns_err_not_panic() {
        let mut data = Vec::new();
        data.extend_from_slice(&2u32.to_le_bytes()); // block_nnz
        data.extend_from_slice(&1u16.to_le_bytes()); // n_rows_in_block
        data.push(2); // varint: nnz = 2
        data.extend_from_slice(&0xFFFF_FFF0u32.to_le_bytes()); // frame_min
        data.push(8); // frame_bits
        data.extend_from_slice(&[0x00, 0xFF]); // deltas [0, 255] → frame_min + 255 overflows
        let res = forbp_decode(&data, 1, false);
        assert!(res.is_err(), "overflow must return Err, not panic");
    }

    // 5.12: Verify block_nnz matches sum of row_nnz
    #[test]
    fn block_nnz_matches_row_nnz_sum() {
        let rows = vec![
            vec![1, 2, 3],
            vec![],
            vec![10, 20],
            vec![5],
            vec![],
            vec![0, 100, 200, 300],
        ];
        let (indices, row_lengths) = flatten(&rows);
        let encoded = forbp_encode(&indices, &row_lengths, true).unwrap();

        // Parse the first (and only) block header to verify
        let mut cursor = Cursor::new(&encoded);
        let block_nnz = cursor.read_u32::<LittleEndian>().unwrap();
        let n_rows_in_block = cursor.read_u16::<LittleEndian>().unwrap();

        assert_eq!(n_rows_in_block, 6);
        let expected_nnz: u32 = row_lengths.iter().sum::<usize>() as u32;
        assert_eq!(block_nnz, expected_nnz);
    }

    // 5.13: LEB128 edge cases
    #[test]
    fn leb128_edge_cases() {
        let test_values: Vec<u32> = vec![0, 127, 128, 16383, 16384, u32::MAX];
        for &val in &test_values {
            let mut buf = Vec::new();
            write_varint(&mut buf, val);
            let mut slice = buf.as_slice();
            let decoded = read_varint(&mut slice).unwrap();
            assert_eq!(decoded, val, "LEB128 mismatch for {val}");
            assert!(slice.is_empty(), "LEB128 leftover bytes for {val}");
        }
    }

    #[test]
    fn leb128_encoding_sizes() {
        // 0 and 127 should be 1 byte
        let mut buf = Vec::new();
        write_varint(&mut buf, 0);
        assert_eq!(buf.len(), 1);

        buf.clear();
        write_varint(&mut buf, 127);
        assert_eq!(buf.len(), 1);

        // 128 should be 2 bytes
        buf.clear();
        write_varint(&mut buf, 128);
        assert_eq!(buf.len(), 2);

        // 16383 should be 2 bytes
        buf.clear();
        write_varint(&mut buf, 16383);
        assert_eq!(buf.len(), 2);

        // 16384 should be 3 bytes
        buf.clear();
        write_varint(&mut buf, 16384);
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn leb128_truncated_data() {
        // A continuation byte with no follow-up
        let buf = vec![0x80];
        let mut slice = buf.as_slice();
        assert!(read_varint(&mut slice).is_err());
    }

    #[test]
    fn leb128_rejects_overflowing_fifth_byte() {
        // 5th byte carries value bits above bit 3 → value overflows u32.
        let buf = vec![0xFF, 0xFF, 0xFF, 0xFF, 0x10];
        let mut slice = buf.as_slice();
        assert!(read_varint(&mut slice).is_err());
    }

    #[test]
    fn leb128_rejects_overlong_with_continuation() {
        // Continuation bit set on the 5th byte demands a (nonexistent) 6th.
        let buf = vec![0x80, 0x80, 0x80, 0x80, 0x80, 0x00];
        let mut slice = buf.as_slice();
        assert!(read_varint(&mut slice).is_err());
    }

    #[test]
    fn decode_truncated_forbp() {
        // Just a partial block header — should error
        let data = vec![0x01, 0x00, 0x00, 0x00]; // block_nnz only, missing n_rows
        let result = forbp_decode(&data, 1, true);
        assert!(result.is_err());
    }

    // Mixed empty and non-empty across block boundary
    #[test]
    fn mixed_across_block_boundary() {
        let mut rows: Vec<Vec<u32>> = Vec::new();
        for i in 0..140 {
            if i % 3 == 0 {
                rows.push(vec![]);
            } else {
                rows.push(vec![i as u32, i as u32 + 10]);
            }
        }
        round_trip(&rows, true);
    }

    #[test]
    fn decode_malformed_n_rows_in_block() {
        // Craft a block header where n_rows_in_block (5) > expected n_rows (1)
        let mut data = Vec::new();
        data.write_u32::<LittleEndian>(0).unwrap(); // block_nnz = 0
        data.write_u16::<LittleEndian>(5).unwrap(); // n_rows_in_block = 5 (but we say n_rows=1)
                                                    // 5 varint zeros for the row nnz counts
        data.extend(std::iter::repeat_n(0x00u8, 5));
        let result = forbp_decode(&data, 1, true);
        assert!(result.is_err());
    }

    #[test]
    fn encode_rejects_u16_overflow() {
        // index 70000 > u16::MAX when index_dtype_u16=true
        let indices = vec![70000u32];
        let row_lengths = vec![1usize];
        let result = forbp_encode(&indices, &row_lengths, true);
        assert!(result.is_err());

        // Same index with u32 mode should succeed
        let result = forbp_encode(&indices, &row_lengths, false);
        assert!(result.is_ok());
    }

    // -----------------------------------------------------------------------
    // SIMD path tests (Phase 2E) — rows with NNZ >= 128 use BitPacker4x
    // -----------------------------------------------------------------------

    /// Helper to generate a sorted row of `n` indices with small gaps.
    fn make_sorted_row(n: usize, start: u32, max_gap: u32) -> Vec<u32> {
        let mut row = Vec::with_capacity(n);
        let mut prev = start;
        let mut state: u64 = 0xBEEF_CAFE_0000 + n as u64;
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let gap = (state % max_gap as u64 + 1) as u32;
            prev += gap;
            row.push(prev);
        }
        row
    }

    #[test]
    fn simd_roundtrip_large_row() {
        // 500 indices — exercises multiple SIMD chunks + remainder
        let row = make_sorted_row(500, 0, 50);
        round_trip(std::slice::from_ref(&row), true);
        round_trip(&[row], false);
    }

    #[test]
    fn simd_roundtrip_exact_128() {
        // Exactly 128 NNZ — one full SIMD chunk, no remainder
        let row = make_sorted_row(128, 0, 30);
        round_trip(&[row], true);
    }

    #[test]
    fn simd_roundtrip_129() {
        // 129 NNZ — one full chunk + 1 remainder
        let row = make_sorted_row(129, 0, 30);
        round_trip(&[row], true);
    }

    #[test]
    fn simd_roundtrip_256() {
        // 256 NNZ — two full SIMD chunks, no remainder
        let row = make_sorted_row(256, 0, 30);
        round_trip(&[row], true);
    }

    #[test]
    fn simd_mixed_small_large_rows() {
        // Mix of small (< 128 NNZ) and large (>= 128 NNZ) rows
        let rows = vec![
            vec![1, 5, 10],              // small
            vec![],                      // empty
            make_sorted_row(200, 0, 20), // large
            vec![42],                    // single
            make_sorted_row(128, 0, 15), // exact threshold
            vec![100, 200, 300],         // small
            make_sorted_row(300, 0, 10), // large
        ];
        round_trip(&rows, true);
    }

    #[test]
    fn simd_across_block_boundary() {
        // 130 rows, some with NNZ >= 128 — exercises block headers + SIMD
        let mut rows = Vec::new();
        for i in 0..130 {
            if i % 10 == 0 {
                rows.push(make_sorted_row(200, 0, 20));
            } else if i % 5 == 0 {
                rows.push(vec![]);
            } else {
                rows.push(vec![i as u32 * 3, i as u32 * 3 + 1]);
            }
        }
        round_trip(&rows, true);
    }

    #[test]
    fn simd_dense_row_1000() {
        // 1000 consecutive indices (dense row, typical scRNA-seq)
        // Gaps are all 1, so frame_bits = 1
        let row: Vec<u32> = (0..1000).collect();
        round_trip(&[row], true);
    }
}
