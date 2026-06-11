// Delta-Golomb-Rice encoder/decoder for indptr (docs/codec.md (Indptr))
//
// Single-stream codec for monotonically non-decreasing u64 indptr arrays.
// Layout: [raw LE u64 first value] [1-byte k] [Rice-coded deltas] [byte-pad]

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Cursor;

use crate::bitstream::{BitReader, BitStreamError, BitWriter};
use crate::dispatch::CodecError;
use crate::median::floor_median_u64_inplace;
use crate::rice::MAX_RICE_K;

/// Compute the Rice parameter k from the median of delta values.
///
/// `k = clamp(floor(log2(0.6931 * median)), 0, MAX_RICE_K)`.
fn compute_k(median: u64) -> u8 {
    if median == 0 {
        return 0;
    }
    let raw = (std::f64::consts::LN_2 * median as f64).log2().floor() as i32;
    raw.clamp(0, MAX_RICE_K as i32) as u8
}

/// Encode an indptr array using Delta-Golomb-Rice coding.
///
/// The indptr array must be monotonically non-decreasing u64 values.
/// Returns the encoded byte vector. Returns `Err(CodecError::MalformedInput)`
/// if any window violates monotonicity — silent wrap on `w[1] - w[0]` would
/// otherwise produce unreadable shards in release.
pub fn delta_golomb_encode(indptr: &[u64]) -> Result<Vec<u8>, CodecError> {
    if indptr.is_empty() {
        return Ok(Vec::new());
    }

    let mut output = Vec::new();

    // Write first value as raw LE u64
    output
        .write_u64::<LittleEndian>(indptr[0])
        .expect("write to Vec cannot fail");

    if indptr.len() == 1 {
        return Ok(output);
    }

    // Compute deltas with monotonicity check. This buffer is a throwaway scratch
    // for the median only; the encode loop below recomputes deltas from `indptr`
    // in order, so the median selection may reorder it in place.
    let mut deltas: Vec<u64> = indptr
        .windows(2)
        .map(|w| {
            if w[1] < w[0] {
                return Err(CodecError::MalformedInput(format!(
                    "indptr must be monotone non-decreasing; saw {} < {}",
                    w[1], w[0]
                )));
            }
            Ok(w[1] - w[0])
        })
        .collect::<Result<_, _>>()?;

    // Compute Rice parameter k from floor median of deltas
    let median = floor_median_u64_inplace(&mut deltas);
    let k = compute_k(median);

    // Write k as 1 byte
    output.push(k);

    // Rice-encode all deltas in a single stream. Monotonicity was validated when
    // `deltas` was built above, so recomputing in original order is safe.
    let mut writer = BitWriter::new();
    for w in indptr.windows(2) {
        let d = w[1] - w[0];
        let q = d >> k;
        let r = d & ((1u64 << k) - 1);
        writer.write_unary(q);
        if k > 0 {
            writer.write_bits(r, k);
        }
    }

    output.extend_from_slice(&writer.flush());
    Ok(output)
}

/// Decode a Delta-Golomb-Rice encoded byte stream back to an indptr array.
///
/// `n_rows_plus_one` is the expected length of the output indptr array
/// (number of rows + 1).
pub fn delta_golomb_decode(
    data: &[u8],
    n_rows_plus_one: usize,
) -> Result<Vec<u64>, BitStreamError> {
    if n_rows_plus_one == 0 {
        return Ok(Vec::new());
    }

    if data.len() < 8 {
        return Err(BitStreamError);
    }

    // Read first value as LE u64
    let mut cursor = Cursor::new(&data[..8]);
    let first = cursor
        .read_u64::<LittleEndian>()
        .map_err(|_| BitStreamError)?;

    let mut indptr = Vec::with_capacity(n_rows_plus_one);
    indptr.push(first);

    if n_rows_plus_one == 1 {
        return Ok(indptr);
    }

    if data.len() < 9 {
        return Err(BitStreamError);
    }

    // Read k byte. `k` is encoded in a 4-bit field at write time, so any
    // value above MAX_RICE_K is a corrupt/hostile stream; reject it before
    // shifting (`q << k` with k >= 64 is UB-shaped — masks in release, panics
    // in debug). `read_bits` would also reject k > 64, but bound it here.
    let k = data[8];
    if k > MAX_RICE_K {
        return Err(BitStreamError);
    }

    // Decode deltas from bitstream
    let mut reader = BitReader::new(&data[9..]);
    let n_deltas = n_rows_plus_one - 1;

    for _ in 0..n_deltas {
        let q = reader.read_unary()?;
        let r = if k > 0 { reader.read_bits(k)? } else { 0 };
        // checked_shl/checked_add: a long unary run or a near-u64::MAX prefix
        // sum must surface as a decode error, not wrap silently in release.
        let delta = q
            .checked_shl(k as u32)
            .map(|qk| qk | r)
            .ok_or(BitStreamError)?;
        let prev = *indptr.last().unwrap();
        indptr.push(prev.checked_add(delta).ok_or(BitStreamError)?);
    }

    Ok(indptr)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Helper tests ---

    // floor_median_u64 tests have moved to crate::median::tests

    #[test]
    fn test_compute_k() {
        assert_eq!(compute_k(0), 0);
        assert_eq!(compute_k(1), 0);
        assert_eq!(compute_k(3), 1);
        assert_eq!(compute_k(100), 6);
    }

    // 4.4: Typical indptr
    #[test]
    fn round_trip_typical_indptr() {
        let indptr = vec![0u64, 150, 280, 500, 700];
        let encoded = delta_golomb_encode(&indptr).unwrap();
        let decoded = delta_golomb_decode(&encoded, indptr.len()).unwrap();
        assert_eq!(decoded, indptr);
    }

    // 4.5: Sparse rows (zero deltas)
    #[test]
    fn round_trip_sparse_rows() {
        let indptr = vec![0u64, 5, 5, 5, 10];
        let encoded = delta_golomb_encode(&indptr).unwrap();
        let decoded = delta_golomb_decode(&encoded, indptr.len()).unwrap();
        assert_eq!(decoded, indptr);
    }

    // 4.6: Single row (2-element indptr)
    #[test]
    fn round_trip_single_row() {
        let indptr = vec![0u64, 1000];
        let encoded = delta_golomb_encode(&indptr).unwrap();
        let decoded = delta_golomb_decode(&encoded, indptr.len()).unwrap();
        assert_eq!(decoded, indptr);
    }

    // 4.7: Large deltas
    #[test]
    fn round_trip_large_deltas() {
        let indptr = vec![0u64, 50_000, 100_000, 200_000];
        let encoded = delta_golomb_encode(&indptr).unwrap();
        let decoded = delta_golomb_decode(&encoded, indptr.len()).unwrap();
        assert_eq!(decoded, indptr);
    }

    // 4.8: Empty shard (1-element indptr)
    #[test]
    fn round_trip_empty_shard() {
        let indptr = vec![0u64];
        let encoded = delta_golomb_encode(&indptr).unwrap();
        assert_eq!(encoded.len(), 8); // just the raw u64
        let decoded = delta_golomb_decode(&encoded, 1).unwrap();
        assert_eq!(decoded, indptr);
    }

    // Edge case: truly empty slice
    #[test]
    fn round_trip_empty_slice() {
        let indptr: Vec<u64> = vec![];
        let encoded = delta_golomb_encode(&indptr).unwrap();
        assert!(encoded.is_empty());
        let decoded = delta_golomb_decode(&encoded, 0).unwrap();
        assert!(decoded.is_empty());
    }

    // 4.9: Monotonicity check
    #[test]
    fn decoded_is_monotonic() {
        let indptr = vec![0u64, 10, 10, 25, 100, 100, 100, 500];
        let encoded = delta_golomb_encode(&indptr).unwrap();
        let decoded = delta_golomb_decode(&encoded, indptr.len()).unwrap();
        for w in decoded.windows(2) {
            assert!(w[0] <= w[1], "indptr not monotonic: {} > {}", w[0], w[1]);
        }
        assert_eq!(decoded, indptr);
    }

    // 4.10: Random round-trip
    #[test]
    fn round_trip_random() {
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let mut indptr = vec![0u64];
        for _ in 0..1000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let delta = state % 500;
            indptr.push(indptr.last().unwrap() + delta);
        }

        let encoded = delta_golomb_encode(&indptr).unwrap();
        let decoded = delta_golomb_decode(&encoded, indptr.len()).unwrap();
        assert_eq!(decoded, indptr);
    }

    // Non-zero start
    #[test]
    fn round_trip_nonzero_start() {
        let indptr = vec![100u64, 200, 300, 400];
        let encoded = delta_golomb_encode(&indptr).unwrap();
        let decoded = delta_golomb_decode(&encoded, indptr.len()).unwrap();
        assert_eq!(decoded, indptr);
    }

    // All-zero deltas (all rows empty)
    #[test]
    fn round_trip_all_zero_deltas() {
        let indptr = vec![0u64; 100];
        let encoded = delta_golomb_encode(&indptr).unwrap();
        let decoded = delta_golomb_decode(&encoded, indptr.len()).unwrap();
        assert_eq!(decoded, indptr);
    }

    // Decode with truncated data returns error
    #[test]
    fn decode_truncated_data() {
        let result = delta_golomb_decode(&[0x00, 0x01, 0x02], 5);
        assert!(result.is_err());
    }

    // F2: a k byte above MAX_RICE_K is a corrupt stream — reject before
    // the `q << k` shift (which would be UB-shaped for k >= 64) rather
    // than panic/wrap.
    #[test]
    fn decode_rejects_oversized_k() {
        let mut data = vec![0u8; 8]; // first value = 0
        data.push(64); // k = 64 (invalid; > MAX_RICE_K)
        data.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
        assert!(delta_golomb_decode(&data, 2).is_err());
    }

    // F2: prefix-sum overflow on a hostile delta must surface as a decode
    // error, not wrap silently in release.
    #[test]
    fn decode_rejects_prefix_sum_overflow() {
        let mut data = Vec::new();
        data.extend_from_slice(&u64::MAX.to_le_bytes()); // first value = u64::MAX
        data.push(0); // k = 0 → delta == unary quotient
        data.push(0x01); // unary: one 1-bit then 0 → q = 1, so delta = 1
                         // prev (u64::MAX) + delta (1) overflows.
        assert!(delta_golomb_decode(&data, 2).is_err());
    }
}
