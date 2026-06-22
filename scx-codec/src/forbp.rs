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

/// Per-row decode metadata produced by the actual FOR-BP encoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForBpRowMetadata {
    pub nnz: u32,
    pub value_start: u64,
    pub frame_min: u32,
    pub frame_bits: u8,
    /// `0` empty, `1` scalar/no-payload layout, `2` BitPacker4x layout.
    pub index_packing: u8,
    /// Bit offset from the start of the encoded index stream to this row's
    /// packed delta payload.
    pub indices_bit_offset: u64,
}

/// Encoded FOR-BP bytes plus per-row metadata from the same encode pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForBpEncodeResult {
    pub bytes: Vec<u8>,
    pub rows: Vec<ForBpRowMetadata>,
}

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
        result |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
        if shift >= 35 {
            return Err(BitStreamError);
        }
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
/// `index_dtype_u16` is true when n_vars <= 65535 (frame_min written as u16).
pub fn forbp_encode(
    indices: &[u32],
    row_lengths: &[usize],
    index_dtype_u16: bool,
) -> Result<Vec<u8>, CodecError> {
    Ok(forbp_encode_with_metadata(indices, row_lengths, index_dtype_u16)?.bytes)
}

/// Encode CSR column indices and return the exact per-row decode metadata
/// observed during encoding.
pub fn forbp_encode_with_metadata(
    indices: &[u32],
    row_lengths: &[usize],
    index_dtype_u16: bool,
) -> Result<ForBpEncodeResult, CodecError> {
    let mut output = Vec::new();
    let mut rows = Vec::with_capacity(row_lengths.len());
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
            let value_start = idx_offset as u64;
            if nnz == 0 {
                rows.push(ForBpRowMetadata {
                    nnz: 0,
                    value_start,
                    frame_min: 0,
                    frame_bits: 0,
                    index_packing: 0,
                    indices_bit_offset: 0,
                });
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
            let indices_bit_offset = (output.len() as u64) * 8;
            let index_packing = if frame_bits > 0 && nnz >= SIMD_THRESHOLD {
                2
            } else {
                1
            };

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

            rows.push(ForBpRowMetadata {
                nnz: nnz as u32,
                value_start,
                frame_min,
                frame_bits,
                index_packing,
                indices_bit_offset,
            });
            idx_offset += nnz;
        }
    }

    if idx_offset != indices.len() {
        return Err(CodecError::MalformedInput(format!(
            "FOR-BP row lengths cover {idx_offset} indices, but input has {}",
            indices.len()
        )));
    }

    Ok(ForBpEncodeResult {
        bytes: output,
        rows,
    })
}

/// Decode FOR-BP indices using encoder-produced per-row metadata instead of
/// walking the block stream. This is used to verify decode sidecar offsets.
pub fn forbp_decode_with_metadata(
    data: &[u8],
    rows: &[ForBpRowMetadata],
) -> Result<Vec<u32>, BitStreamError> {
    let nnz_total: usize = rows
        .iter()
        .map(|row| row.nnz as usize)
        .try_fold(0usize, |acc, nnz| {
            acc.checked_add(nnz).ok_or(BitStreamError)
        })?;
    let mut all_indices = Vec::with_capacity(nnz_total);

    for row in rows {
        let nnz = row.nnz as usize;
        if nnz == 0 {
            continue;
        }
        if row.frame_bits > 32 {
            return Err(BitStreamError);
        }

        let start = all_indices.len();
        all_indices.resize(start + nnz, 0);

        if row.frame_bits > 0 {
            let bit_offset = usize::try_from(row.indices_bit_offset).map_err(|_| BitStreamError)?;
            let total_bits = nnz
                .checked_mul(row.frame_bits as usize)
                .ok_or(BitStreamError)?;
            if bit_offset.checked_add(total_bits).ok_or(BitStreamError)? > data.len() * 8 {
                return Err(BitStreamError);
            }

            if row.index_packing == 2 {
                if !bit_offset.is_multiple_of(8) {
                    return Err(BitStreamError);
                }
                let packer = BitPacker4x::new();
                let full_chunks = nnz / SIMD_THRESHOLD;
                let remainder = nnz % SIMD_THRESHOLD;
                let chunk_bytes = row.frame_bits as usize * SIMD_THRESHOLD / 8;
                let mut data_offset = bit_offset / 8;

                for c in 0..full_chunks {
                    let dst_start = start + c * SIMD_THRESHOLD;
                    if data_offset + chunk_bytes > data.len() {
                        return Err(BitStreamError);
                    }
                    packer.decompress(
                        &data[data_offset..],
                        &mut all_indices[dst_start..dst_start + SIMD_THRESHOLD],
                        row.frame_bits,
                    );
                    data_offset += chunk_bytes;
                }

                if remainder > 0 {
                    let rem_start = start + full_chunks * SIMD_THRESHOLD;
                    let rem_bits = remainder * row.frame_bits as usize;
                    let rem_bytes = rem_bits.div_ceil(8);
                    if data_offset + rem_bytes > data.len() {
                        return Err(BitStreamError);
                    }
                    unpack_fixed_width(
                        &data[data_offset..data_offset + rem_bytes],
                        0,
                        remainder,
                        row.frame_bits,
                        &mut all_indices[rem_start..],
                    );
                }
            } else if row.index_packing == 1 {
                unpack_fixed_width(
                    data,
                    bit_offset,
                    nnz,
                    row.frame_bits,
                    &mut all_indices[start..],
                );
            } else {
                return Err(BitStreamError);
            }
        } else if row.index_packing > 1 {
            return Err(BitStreamError);
        }

        let mut prev = row.frame_min;
        for delta in &mut all_indices[start..start + nnz] {
            prev = prev.checked_add(*delta).ok_or(BitStreamError)?;
            *delta = prev;
        }
    }

    Ok(all_indices)
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

/// Decode FOR-BP encoded indices with an optional nnz hint for pre-allocation.
///
/// When `nnz_hint > 0`, pre-allocates the output vectors for better performance.
pub fn forbp_decode_with_hint(
    data: &[u8],
    n_rows: usize,
    nnz_hint: usize,
    index_dtype_u16: bool,
) -> Result<(Vec<u32>, Vec<usize>), BitStreamError> {
    forbp_decode_inner(data, n_rows, nnz_hint, index_dtype_u16)
}

fn forbp_decode_inner(
    data: &[u8],
    n_rows: usize,
    nnz_hint: usize,
    index_dtype_u16: bool,
) -> Result<(Vec<u32>, Vec<usize>), BitStreamError> {
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
