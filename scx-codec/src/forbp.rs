// FOR-BP encoder/decoder for indices (SPEC §4.2)

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Cursor;

use crate::bitstream::{BitReader, BitStreamError, BitWriter};

/// Block size for FOR-BP coding of indices: 128 rows per block.
pub const B_IDX: usize = 128;

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
pub fn forbp_encode(indices: &[u32], row_lengths: &[usize], index_dtype_u16: bool) -> Vec<u8> {
    let mut output = Vec::new();
    let mut idx_offset: usize = 0;

    for block_rows in row_lengths.chunks(B_IDX) {
        let n_rows_in_block = block_rows.len();
        let block_nnz: usize = block_rows.iter().sum();

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
            if nnz == 0 {
                continue;
            }

            let row_indices = &indices[idx_offset..idx_offset + nnz];

            // frame_min is the first (smallest) index in the sorted row
            let frame_min = row_indices[0];

            // Compute deltas: delta[0] = 0, delta[j] = indices[j] - indices[j-1]
            let mut deltas = Vec::with_capacity(nnz);
            deltas.push(0u32);
            for j in 1..nnz {
                deltas.push(row_indices[j] - row_indices[j - 1]);
            }

            let max_delta = deltas.iter().copied().max().unwrap_or(0);
            let frame_bits = bits_needed(max_delta);

            // Write frame_min
            if index_dtype_u16 {
                output.write_u16::<LittleEndian>(frame_min as u16).unwrap();
            } else {
                output.write_u32::<LittleEndian>(frame_min).unwrap();
            }

            // Write frame_bits
            output.push(frame_bits);

            // Bit-pack deltas if frame_bits > 0
            if frame_bits > 0 {
                let mut bw = BitWriter::new();
                for &d in &deltas {
                    bw.write_bits(d as u64, frame_bits);
                }
                output.extend_from_slice(&bw.flush());
            }

            idx_offset += nnz;
        }
    }

    output
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
    let mut all_indices = Vec::new();
    let mut all_row_lengths = Vec::new();
    let mut cursor = Cursor::new(data);
    let mut rows_remaining = n_rows;

    while rows_remaining > 0 {
        // Read block header
        let _block_nnz = cursor.read_u32::<LittleEndian>().map_err(|_| BitStreamError)?;
        let n_rows_in_block =
            cursor.read_u16::<LittleEndian>().map_err(|_| BitStreamError)? as usize;

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

        // Decode each row
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

            // Read bit-packed deltas
            let mut deltas = Vec::with_capacity(nnz);
            if frame_bits > 0 {
                // Compute how many bytes the bit-packed deltas occupy
                let total_bits = frame_bits as usize * nnz;
                let total_bytes = total_bits.div_ceil(8);

                let pos = cursor.position() as usize;
                if pos + total_bytes > data.len() {
                    return Err(BitStreamError);
                }
                let bit_data = &data[pos..pos + total_bytes];
                let mut br = BitReader::new(bit_data);
                for _ in 0..nnz {
                    deltas.push(br.read_bits(frame_bits)? as u32);
                }
                cursor.set_position((pos + total_bytes) as u64);
            } else {
                // frame_bits == 0: all deltas are 0
                deltas.resize(nnz, 0);
            }

            // Reconstruct indices via prefix sum from frame_min + deltas
            let mut prev = frame_min;
            for &d in &deltas {
                let idx = prev + d;
                all_indices.push(idx);
                prev = idx;
            }
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
        let encoded = forbp_encode(&indices, &row_lengths, index_dtype_u16);
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
        round_trip(
            &[vec![1, 3, 7], vec![0, 2, 4, 6, 8], vec![100, 200]],
            true,
        );
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
        let encoded = forbp_encode(&indices, &row_lengths, true);

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
}
