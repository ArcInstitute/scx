// Rice-gap encoder/decoder for indices (codec_id = 5, "scx2").
//
// This is the scx2 indices codec. It mirrors the FOR-BP block structure
// (docs/codec.md §3 block header) so the shared per-row nnz table is identical,
// but replaces the per-row fixed-width bit-packing of column-gaps with adaptive
// Rice coding of those gaps — the same scheme the values stream already uses
// (§4), applied per row with a per-row `k`.
//
// Rationale: FOR-BP picks `frame_bits = ceil(log2(max_gap+1))` per *whole row*,
// so a single large gap widens every delta in that row. Adaptive Rice on the
// gaps spends ~entropy and degrades gracefully on the tail. Measured index-size
// reductions vs FOR-BP on real UMI matrices: 7–23% (see docs/codec.md §9).
//
// Per-row wire layout (within a B_IDX=128-row block, after the shared block
// header):
//   frame_min : u16 / u32     (first column index of the row, stored raw)
//   rice_k    : u8            (Rice parameter for this row's gaps, 0..=15)
//   gaps      : Rice(shifted) for j=1..nnz, where shifted = idx[j]-idx[j-1]-1
//               byte-padded to a byte boundary at the row end
// nnz==0 rows emit nothing; nnz==1 rows emit frame_min + rice_k only.
//
// Gaps are >= 1 for canonical (strictly-increasing, deduplicated) CSR rows, so
// `shifted = gap - 1 >= 0`. The encoder rejects non-increasing rows loudly,
// mirroring FOR-BP's sorted-input requirement.

use byteorder::{LittleEndian, WriteBytesExt};

use crate::bitstream::{BitReader, BitStreamError, BitWriter};
use crate::dispatch::CodecError;
use crate::forbp::{write_varint, ForBpEncodeResult, ForBpRowMetadata, B_IDX};
use crate::median::floor_median_u32_inplace;
use crate::rice::MAX_RICE_K;

/// `index_packing` marker stored in [`ForBpRowMetadata`] for scx2 Rice-gap rows.
/// Distinguishes scx2 rows from FOR-BP's `1` (scalar) / `2` (BitPacker4x) so the
/// shared decode-sidecar layout can carry either codec's per-row metadata.
pub const RICE_GAP_INDEX_PACKING: u8 = 3;

/// Compute the Rice parameter `k` from the median of shifted gaps.
/// Identical rule to the values codec (`k = clamp(floor(log2(0.6931*median)))`).
fn compute_k(median: u32) -> u8 {
    if median == 0 {
        return 0;
    }
    let raw = (std::f64::consts::LN_2 * median as f64).log2().floor() as i32;
    raw.clamp(0, MAX_RICE_K as i32) as u8
}

/// LEB128 varint read from a [`BitReader`] at a byte-aligned position.
/// Mirrors [`crate::forbp::read_varint`] but consumes from the bit reader so the
/// full-stream walk uses a single reader for byte- and bit-oriented fields.
fn read_varint_bits(reader: &mut BitReader) -> Result<u32, BitStreamError> {
    let mut result: u32 = 0;
    let mut shift: u32 = 0;
    loop {
        let byte = reader.read_bits(8)? as u8;
        if shift == 28 {
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
// Encoder
// ---------------------------------------------------------------------------

/// Encode CSR column indices using Rice-gap coding (scx2 indices).
///
/// `indices` is the flat column-index array (strictly increasing within each
/// row); `row_lengths` gives nnz per row; `index_dtype_u16` is true when
/// `n_vars <= 65535` (frame_min written as u16).
pub fn rice_gap_encode(
    indices: &[u32],
    row_lengths: &[usize],
    index_dtype_u16: bool,
) -> Result<Vec<u8>, CodecError> {
    Ok(rice_gap_encode_with_metadata(indices, row_lengths, index_dtype_u16)?.bytes)
}

/// Encode CSR column indices and return the per-row decode metadata observed
/// during encoding (for the random-access sidecar).
pub fn rice_gap_encode_with_metadata(
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
        if block_nnz > u32::MAX as usize {
            return Err(BitStreamError.into());
        }

        output.write_u32::<LittleEndian>(block_nnz as u32).unwrap();
        output
            .write_u16::<LittleEndian>(n_rows_in_block as u16)
            .unwrap();
        for &nnz in block_rows {
            write_varint(&mut output, nnz as u32);
        }

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

            let row = &indices[idx_offset..idx_offset + nnz];
            let frame_min = row[0];

            // shifted gaps: idx[j] - idx[j-1] - 1, requires strictly increasing.
            let mut shifted = Vec::with_capacity(nnz - 1);
            for j in 1..nnz {
                if row[j] <= row[j - 1] {
                    return Err(CodecError::MalformedInput(format!(
                        "scx2 requires strictly increasing row indices; saw {} <= {} at offset {j}",
                        row[j],
                        row[j - 1]
                    )));
                }
                shifted.push(row[j] - row[j - 1] - 1);
            }

            // Rice parameter from the median of shifted gaps. The median routine
            // reorders its scratch buffer, so use a throwaway clone.
            let mut scratch = shifted.clone();
            let median = floor_median_u32_inplace(&mut scratch);
            let k = compute_k(median);

            if index_dtype_u16 {
                if frame_min > u16::MAX as u32 {
                    return Err(BitStreamError.into());
                }
                output.write_u16::<LittleEndian>(frame_min as u16).unwrap();
            } else {
                output.write_u32::<LittleEndian>(frame_min).unwrap();
            }
            output.push(k);

            // Gaps start at the current (byte-aligned) position.
            let indices_bit_offset = (output.len() as u64) * 8;
            if !shifted.is_empty() {
                let mut bw = BitWriter::new();
                for &s in &shifted {
                    bw.write_unary((s >> k) as u64);
                    if k > 0 {
                        bw.write_bits((s & ((1u32 << k) - 1)) as u64, k);
                    }
                }
                output.extend_from_slice(&bw.flush());
            }

            rows.push(ForBpRowMetadata {
                nnz: nnz as u32,
                value_start,
                frame_min,
                frame_bits: k,
                index_packing: RICE_GAP_INDEX_PACKING,
                indices_bit_offset,
            });
            idx_offset += nnz;
        }
    }

    if idx_offset != indices.len() {
        return Err(CodecError::MalformedInput(format!(
            "scx2 row lengths cover {idx_offset} indices, but input has {}",
            indices.len()
        )));
    }

    Ok(ForBpEncodeResult {
        bytes: output,
        rows,
    })
}

// ---------------------------------------------------------------------------
// Decoder (full single-reader walk)
// ---------------------------------------------------------------------------

/// Decode scx2 Rice-gap indices.
///
/// Returns `(indices, row_lengths)` where `indices` is the flat column-index
/// array and `row_lengths` gives per-row nnz. A single [`BitReader`] walks the
/// whole stream; byte-oriented fields are read at byte-aligned positions and
/// each row's gap payload is `align_to_byte`-terminated.
pub fn rice_gap_decode(
    data: &[u8],
    n_rows: usize,
    index_dtype_u16: bool,
) -> Result<(Vec<u32>, Vec<usize>), BitStreamError> {
    rice_gap_decode_with_hint(data, n_rows, 0, index_dtype_u16)
}

/// Decode scx2 Rice-gap indices with an nnz hint for pre-allocation.
pub fn rice_gap_decode_with_hint(
    data: &[u8],
    n_rows: usize,
    nnz_hint: usize,
    index_dtype_u16: bool,
) -> Result<(Vec<u32>, Vec<usize>), BitStreamError> {
    let mut all_indices = Vec::with_capacity(nnz_hint);
    let mut all_row_lengths = Vec::with_capacity(n_rows);
    if n_rows == 0 {
        return Ok((all_indices, all_row_lengths));
    }
    let mut reader = BitReader::new(data);
    let mut rows_remaining = n_rows;

    while rows_remaining > 0 {
        // Block header (byte-aligned). 32/16 LSB-first bits over LE bytes == the
        // integer value.
        let _block_nnz = reader.read_bits(32)?;
        let n_rows_in_block = reader.read_bits(16)? as usize;
        if n_rows_in_block > rows_remaining {
            return Err(BitStreamError);
        }

        let mut row_nnzs = Vec::with_capacity(n_rows_in_block);
        for _ in 0..n_rows_in_block {
            row_nnzs.push(read_varint_bits(&mut reader)? as usize);
        }

        for &nnz in &row_nnzs {
            all_row_lengths.push(nnz);
            if nnz == 0 {
                continue;
            }
            let frame_min = if index_dtype_u16 {
                reader.read_bits(16)? as u32
            } else {
                reader.read_bits(32)? as u32
            };
            let k = reader.read_bits(8)? as u8;
            if k > MAX_RICE_K {
                return Err(BitStreamError);
            }

            all_indices.push(frame_min);
            let mut prev = frame_min;
            for _ in 1..nnz {
                let q = reader.read_unary()?;
                let r = if k > 0 { reader.read_bits(k)? } else { 0 };
                let shifted = q
                    .checked_shl(k as u32)
                    .map(|qk| qk | r)
                    .filter(|&v| v <= u32::MAX as u64)
                    .ok_or(BitStreamError)?;
                let gap = (shifted as u32).checked_add(1).ok_or(BitStreamError)?;
                prev = prev.checked_add(gap).ok_or(BitStreamError)?;
                all_indices.push(prev);
            }
            // Each row's gap payload is byte-padded by the encoder.
            reader.align_to_byte();
        }

        rows_remaining -= n_rows_in_block;
    }

    Ok((all_indices, all_row_lengths))
}

// ---------------------------------------------------------------------------
// Decoder (random access via encoder-produced metadata)
// ---------------------------------------------------------------------------

/// Decode scx2 indices for the given per-row metadata, seeking directly to each
/// row's `indices_bit_offset`. Used by the random-access / sidecar path; the
/// output matches the corresponding rows of [`rice_gap_decode`].
pub fn rice_gap_decode_with_metadata(
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
        if row.frame_bits > MAX_RICE_K {
            return Err(BitStreamError);
        }
        if row.index_packing != RICE_GAP_INDEX_PACKING {
            return Err(BitStreamError);
        }
        all_indices.push(row.frame_min);
        if nnz == 1 {
            continue;
        }
        let k = row.frame_bits;
        let bit_offset = usize::try_from(row.indices_bit_offset).map_err(|_| BitStreamError)?;
        let mut reader = BitReader::new_at(data, bit_offset)?;
        let mut prev = row.frame_min;
        for _ in 1..nnz {
            let q = reader.read_unary()?;
            let r = if k > 0 { reader.read_bits(k)? } else { 0 };
            let shifted = q
                .checked_shl(k as u32)
                .map(|qk| qk | r)
                .filter(|&v| v <= u32::MAX as u64)
                .ok_or(BitStreamError)?;
            let gap = (shifted as u32).checked_add(1).ok_or(BitStreamError)?;
            prev = prev.checked_add(gap).ok_or(BitStreamError)?;
            all_indices.push(prev);
        }
    }

    Ok(all_indices)
}

#[cfg(test)]
#[path = "rice_gap_tests.rs"]
mod tests;
