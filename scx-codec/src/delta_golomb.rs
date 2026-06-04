// Delta-Golomb-Rice encoder/decoder for indptr (docs/codec.md (Indptr))
//
// Single-stream codec for monotonically non-decreasing u64 indptr arrays.
// Layout: [raw LE u64 first value] [1-byte k] [Rice-coded deltas] [byte-pad]

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Cursor;

use crate::bitstream::{BitReader, BitStreamError, BitWriter};
use crate::dispatch::CodecError;
use crate::median::floor_median_u64;

/// Compute the Rice parameter k from the median of delta values.
///
/// `k = clamp(floor(log2(0.6931 * median)), 0, 15)`.
fn compute_k(median: u64) -> u8 {
    if median == 0 {
        return 0;
    }
    let raw = (std::f64::consts::LN_2 * median as f64).log2().floor() as i32;
    raw.clamp(0, 15) as u8
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

    // Compute deltas with monotonicity check.
    let deltas: Vec<u64> = indptr
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
    let median = floor_median_u64(&deltas);
    let k = compute_k(median);

    // Write k as 1 byte
    output.push(k);

    // Rice-encode all deltas in a single stream
    let mut writer = BitWriter::new();
    for &d in &deltas {
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

    // Read k byte
    let k = data[8];

    // Decode deltas from bitstream
    let mut reader = BitReader::new(&data[9..]);
    let n_deltas = n_rows_plus_one - 1;

    for _ in 0..n_deltas {
        let q = reader.read_unary()?;
        let r = if k > 0 { reader.read_bits(k)? } else { 0 };
        let delta = (q << k) | r;
        let prev = *indptr.last().unwrap();
        indptr.push(prev + delta);
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
}
