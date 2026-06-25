// Tests for the scx2 Rice-gap indices codec.

use super::*;

fn flatten(rows: &[Vec<u32>]) -> (Vec<u32>, Vec<usize>) {
    let indices: Vec<u32> = rows.iter().flatten().copied().collect();
    let row_lengths: Vec<usize> = rows.iter().map(|r| r.len()).collect();
    (indices, row_lengths)
}

/// Full round-trip through the single-reader walk decoder.
fn round_trip(rows: &[Vec<u32>], u16: bool) {
    let (indices, row_lengths) = flatten(rows);
    let encoded = rice_gap_encode(&indices, &row_lengths, u16).unwrap();
    let (dec_idx, dec_len) = rice_gap_decode(&encoded, row_lengths.len(), u16).unwrap();
    assert_eq!(dec_idx, indices, "indices mismatch");
    assert_eq!(dec_len, row_lengths, "row_lengths mismatch");

    // Metadata-driven decode must match the full-walk decode bit-for-bit.
    let with_meta = rice_gap_encode_with_metadata(&indices, &row_lengths, u16).unwrap();
    let meta_idx = rice_gap_decode_with_metadata(&with_meta.bytes, &with_meta.rows).unwrap();
    assert_eq!(meta_idx, indices, "metadata decode mismatch");
}

#[test]
fn single_row() {
    round_trip(&[vec![0, 5, 10, 20]], true);
}

#[test]
fn multiple_rows() {
    round_trip(&[vec![1, 3, 7], vec![0, 2, 4, 6, 8], vec![100, 200]], true);
}

#[test]
fn empty_and_singleton_rows() {
    round_trip(&[vec![], vec![1, 2, 3], vec![], vec![5], vec![]], true);
}

#[test]
fn all_empty_rows() {
    round_trip(&[vec![], vec![], vec![]], true);
}

#[test]
fn dense_consecutive_row_costs_like_forbp() {
    // Gaps all 1 → shifted 0 → k=0 → 1 bit/gap (must not regress to 2 bits).
    let row: Vec<u32> = (0..1000).collect();
    round_trip(std::slice::from_ref(&row), true);
    // Sanity: encoded gap payload ~= 1 bit/gap. 999 gaps -> ~125 bytes + headers.
    let (indices, row_lengths) = flatten(&[row]);
    let enc = rice_gap_encode(&indices, &row_lengths, true).unwrap();
    assert!(
        enc.len() < 200,
        "dense row should pack ~1 bit/gap, got {} bytes",
        enc.len()
    );
}

#[test]
fn sparse_large_gaps_u16() {
    round_trip(&[vec![0, 10000, 30000]], true);
}

#[test]
fn sparse_large_gaps_u32() {
    round_trip(&[vec![0, 10000, 30000, 100000]], false);
}

#[test]
fn heavy_tail_row() {
    // Bulk of small gaps + a few huge gaps: the case Rice-gap targets.
    let mut row = vec![0u32];
    let mut p = 0u32;
    for i in 0..300 {
        p += if i % 50 == 0 { 5000 } else { 1 + (i % 3) };
        row.push(p);
    }
    round_trip(&[row], false);
}

#[test]
fn full_block_and_partial() {
    let rows: Vec<Vec<u32>> = (0..130).map(|i| vec![i as u32, i as u32 + 1]).collect();
    round_trip(&rows, true);
}

#[test]
fn exact_two_blocks() {
    let rows: Vec<Vec<u32>> = (0..256).map(|i| vec![i as u32]).collect();
    round_trip(&rows, true);
}

#[test]
fn mixed_across_block_boundary() {
    let mut rows: Vec<Vec<u32>> = Vec::new();
    let mut p = 0u32;
    for i in 0..140 {
        if i % 3 == 0 {
            rows.push(vec![]);
        } else {
            let mut row = Vec::new();
            let mut q = i as u32;
            for _ in 0..(i % 7 + 1) {
                q += 1 + (i as u32 % 11);
                row.push(q);
            }
            p = p.wrapping_add(1);
            rows.push(row);
        }
    }
    let _ = p;
    round_trip(&rows, true);
}

#[test]
fn rejects_non_increasing_row() {
    // Equal adjacent indices (gap 0) is rejected.
    let res = rice_gap_encode(&[5, 5], &[2], false);
    assert!(matches!(res, Err(CodecError::MalformedInput(_))));
    // Decreasing is rejected.
    let res = rice_gap_encode(&[5, 3], &[2], false);
    assert!(matches!(res, Err(CodecError::MalformedInput(_))));
}

#[test]
fn decode_truncated_returns_err() {
    // Valid header claiming a non-empty row, but no body.
    let mut data = Vec::new();
    data.extend_from_slice(&2u32.to_le_bytes()); // block_nnz
    data.extend_from_slice(&1u16.to_le_bytes()); // n_rows_in_block
    data.push(2); // varint nnz = 2
                  // frame_min (u32) + k missing → error
    let res = rice_gap_decode(&data, 1, false);
    assert!(res.is_err());
}

#[test]
fn decode_rejects_corrupt_k() {
    // Build a header with frame_min then k=99 (> MAX_RICE_K).
    let mut data = Vec::new();
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes());
    data.push(2); // nnz = 2
    data.extend_from_slice(&0u32.to_le_bytes()); // frame_min
    data.push(99); // k out of range
    data.extend_from_slice(&[0u8; 4]); // some gap bytes
    let res = rice_gap_decode(&data, 1, false);
    assert!(res.is_err());
}

#[test]
fn decode_prefix_sum_overflow_returns_err() {
    // frame_min near u32::MAX with a large gap → index overflow must Err.
    let indices = vec![u32::MAX - 1];
    let _ = indices;
    // Hand-craft: frame_min = u32::MAX, nnz=2, k=0, one unary gap of huge q.
    let mut data = Vec::new();
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes());
    data.push(2);
    data.extend_from_slice(&u32::MAX.to_le_bytes()); // frame_min = u32::MAX
    data.push(0); // k = 0
                  // gap code: unary(1) = bits "10" then byte-pad. shifted=1 -> gap=2 -> overflow.
    let mut bw = BitWriter::new();
    bw.write_unary(1);
    data.extend_from_slice(&bw.flush());
    let res = rice_gap_decode(&data, 1, false);
    assert!(res.is_err(), "prefix-sum overflow must Err, not wrap/panic");
}

#[test]
fn random_round_trip() {
    let mut state: u64 = 0xA5A5_1234_DEAD_0001;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for u16 in [true, false] {
        let max_gap = if u16 { 50u32 } else { 5000 };
        let mut rows = Vec::new();
        for _ in 0..400 {
            let nnz = (next() % 25) as usize;
            let mut row = Vec::with_capacity(nnz);
            let mut prev = (next() % 100) as u32;
            for _ in 0..nnz {
                prev += 1 + (next() % max_gap as u64) as u32;
                row.push(prev);
            }
            rows.push(row);
        }
        round_trip(&rows, u16);
    }
}

#[test]
fn row_range_subsets_match_full() {
    // Build rows, encode-with-metadata, then decode arbitrary row windows via
    // the per-row metadata and confirm they equal the matching full-decode slice.
    let mut state: u64 = 0xFEED_FACE_0000_0007;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut rows = Vec::new();
    for _ in 0..200 {
        let nnz = (next() % 300) as usize; // exercises >128-nnz rows too
        let mut row = Vec::with_capacity(nnz);
        let mut prev = (next() % 50) as u32;
        for _ in 0..nnz {
            prev += 1 + (next() % 40) as u32;
            row.push(prev);
        }
        rows.push(row);
    }
    let (indices, row_lengths) = flatten(&rows);
    let enc = rice_gap_encode_with_metadata(&indices, &row_lengths, false).unwrap();
    let full = rice_gap_decode_with_metadata(&enc.bytes, &enc.rows).unwrap();
    assert_eq!(full, indices);

    // Decode each contiguous window [start, start+len) via its metadata slice.
    for &(start, len) in &[(0usize, 1usize), (5, 10), (50, 100), (199, 1), (0, 200)] {
        let slice = &enc.rows[start..start + len];
        let got = rice_gap_decode_with_metadata(&enc.bytes, slice).unwrap();
        let off0: usize = enc.rows[..start].iter().map(|r| r.nnz as usize).sum();
        let n: usize = slice.iter().map(|r| r.nnz as usize).sum();
        assert_eq!(got, &indices[off0..off0 + n], "window ({start},{len})");
    }
}
