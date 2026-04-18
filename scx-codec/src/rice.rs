// Adaptive Rice encoder/decoder for values (docs/codec.md (Values))
//
// Implementation follows the adaptive Rice-coding scheme described in
// Malvar, H. S. "Adaptive run-length / Golomb-Rice encoding of quantised
// generalised Gaussian sources with unknown statistics." Data Compression
// Conference, 2006 (DCC'06), pp. 23–32. Block size B_VAL=256 matches the
// reference; the k-selection rule (k ≈ ⌊log2(0.6931 · median)⌋) is the
// standard Gaussian-geometric estimator used by JPEG-LS / HP-LOCO.

use crate::bitstream::{BitReader, BitStreamError, BitWriter};

/// Default block size for Rice coding of values.
pub const B_VAL: usize = 256;

/// Compute the floor median of a slice of u32 values.
///
/// For even length, returns the lower of the two middle values.
/// For odd length, returns the middle value.
fn floor_median(values: &[u32]) -> u32 {
    match values.len() {
        0 => 0,
        1 => values[0],
        n => {
            let mut sorted = values.to_vec();
            sorted.sort_unstable();
            if n % 2 == 0 {
                sorted[n / 2 - 1]
            } else {
                sorted[n / 2]
            }
        }
    }
}

/// Compute the Rice parameter k from the median of shifted values.
///
/// `k = max(0, floor(log2(0.6931 * median)))`, clamped to 0–15.
fn compute_k(median: u32) -> u8 {
    if median == 0 {
        return 0;
    }
    let raw = (std::f64::consts::LN_2 * median as f64).log2().floor() as i32;
    raw.clamp(0, 15) as u8
}

/// Encode non-zero count values using blocked Rice coding.
///
/// All values must be >= 1 (non-zero counts). They are shifted by -1 before encoding.
/// The output is a byte vector containing the encoded bitstream.
pub fn rice_encode(values: &[u32], block_size: usize) -> Result<Vec<u8>, BitStreamError> {
    if values.contains(&0) {
        return Err(BitStreamError);
    }

    let mut writer = BitWriter::new();

    for chunk in values.chunks(block_size) {
        // Shift: subtract 1 from each value
        let shifted: Vec<u32> = chunk.iter().map(|&v| v - 1).collect();

        // Compute Rice parameter k
        let median = floor_median(&shifted);
        let k = compute_k(median);

        // Write block header: 1 byte with k in low nibble
        writer.write_bits(k as u64, 8);

        // Encode each shifted value
        for &s in &shifted {
            let q = (s >> k) as u64;
            let r = (s & ((1u32 << k) - 1)) as u64;
            writer.write_unary(q);
            if k > 0 {
                writer.write_bits(r, k);
            }
        }

        // Pad to byte boundary after each block
        writer.pad_to_byte();
    }

    Ok(writer.flush())
}

/// Decode Rice-encoded values.
///
/// Returns `n_values` decoded values, each >= 1.
pub fn rice_decode(
    data: &[u8],
    n_values: usize,
    block_size: usize,
) -> Result<Vec<u32>, BitStreamError> {
    let mut output = Vec::with_capacity(n_values);
    let mut reader = BitReader::new(data);
    let mut remaining = n_values;

    while remaining > 0 {
        let block_len = remaining.min(block_size);

        // Read block header byte
        let k = reader.read_bits(8)? as u8 & 0x0F;

        for _ in 0..block_len {
            let q = reader.read_unary()? as u32;
            let r = if k > 0 {
                reader.read_bits(k)? as u32
            } else {
                0
            };
            let shifted = (q << k) | r;
            output.push(shifted + 1);
        }

        // Align to byte boundary
        reader.align_to_byte();

        remaining -= block_len;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Helper tests ---

    #[test]
    fn test_floor_median_empty() {
        assert_eq!(floor_median(&[]), 0);
    }

    #[test]
    fn test_floor_median_single() {
        assert_eq!(floor_median(&[42]), 42);
    }

    #[test]
    fn test_floor_median_even() {
        // [1, 2, 3, 4] → lower middle = sorted[1] = 2
        assert_eq!(floor_median(&[4, 1, 3, 2]), 2);
    }

    #[test]
    fn test_floor_median_odd() {
        // [1, 2, 3] → middle = sorted[1] = 2
        assert_eq!(floor_median(&[3, 1, 2]), 2);
    }

    // 3.10: k selection verification
    #[test]
    fn test_compute_k() {
        // median=0 → k=0
        assert_eq!(compute_k(0), 0);
        // median=1 → 0.6931*1=0.6931, log2(0.6931)≈-0.529, floor=-1, max(0,-1)=0
        assert_eq!(compute_k(1), 0);
        // median=2 → 0.6931*2=1.3862, log2≈0.471, floor=0
        assert_eq!(compute_k(2), 0);
        // median=3 → 0.6931*3=2.0793, log2≈1.056, floor=1
        assert_eq!(compute_k(3), 1);
        // median=7 → 0.6931*7=4.8517, log2≈2.278, floor=2
        assert_eq!(compute_k(7), 2);
        // median=100 → 0.6931*100=69.31, log2≈6.115, floor=6
        assert_eq!(compute_k(100), 6);
    }

    // 3.4: All-ones block
    #[test]
    fn round_trip_all_ones() {
        let values = vec![1u32; 256];
        let encoded = rice_encode(&values, B_VAL).unwrap();
        let decoded = rice_decode(&encoded, 256, B_VAL).unwrap();
        assert_eq!(decoded, values);
    }

    // 3.5: Typical UMI counts
    #[test]
    fn round_trip_typical_umi() {
        let values = vec![1, 1, 1, 2, 2, 3];
        let encoded = rice_encode(&values, B_VAL).unwrap();
        let decoded = rice_decode(&encoded, values.len(), B_VAL).unwrap();
        assert_eq!(decoded, values);
    }

    // 3.6: Uniform-ish values, verify k
    #[test]
    fn round_trip_uniform_verify_k() {
        let values: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        // shifted = [0, 1, 2, 3, 4, 5, 6, 7], median=3, k=1
        let shifted: Vec<u32> = values.iter().map(|&v| v - 1).collect();
        assert_eq!(floor_median(&shifted), 3);
        assert_eq!(compute_k(3), 1);

        let encoded = rice_encode(&values, B_VAL).unwrap();
        let decoded = rice_decode(&encoded, values.len(), B_VAL).unwrap();
        assert_eq!(decoded, values);
    }

    // 3.7: Outlier value
    #[test]
    fn round_trip_outlier() {
        let values = vec![1, 1, 1, 500];
        let encoded = rice_encode(&values, B_VAL).unwrap();
        let decoded = rice_decode(&encoded, values.len(), B_VAL).unwrap();
        assert_eq!(decoded, values);
    }

    // 3.8: Size edge cases
    #[test]
    fn round_trip_single_value() {
        let values = vec![42u32];
        let encoded = rice_encode(&values, B_VAL).unwrap();
        let decoded = rice_decode(&encoded, 1, B_VAL).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn round_trip_exact_block() {
        let values: Vec<u32> = (1..=256).collect();
        let encoded = rice_encode(&values, B_VAL).unwrap();
        let decoded = rice_decode(&encoded, 256, B_VAL).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn round_trip_two_blocks() {
        // 257 values → two blocks: 256 + 1
        let values: Vec<u32> = (1..=257).collect();
        let encoded = rice_encode(&values, B_VAL).unwrap();
        let decoded = rice_decode(&encoded, 257, B_VAL).unwrap();
        assert_eq!(decoded, values);
    }

    // 3.9: Random round-trip
    #[test]
    fn round_trip_random() {
        // Simple PRNG (xorshift)
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let values: Vec<u32> = (0..1000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 1000 + 1) as u32 // values in 1..=1000
            })
            .collect();

        let encoded = rice_encode(&values, B_VAL).unwrap();
        let decoded = rice_decode(&encoded, values.len(), B_VAL).unwrap();
        assert_eq!(decoded, values);
    }

    // 3.11: Block byte alignment
    #[test]
    fn block_byte_alignment() {
        // Encode multi-block input, verify we can decode correctly
        // (which implicitly verifies byte alignment between blocks)
        let values: Vec<u32> = vec![1; 300]; // 256 + 44 = two blocks
        let encoded = rice_encode(&values, B_VAL).unwrap();
        let decoded = rice_decode(&encoded, 300, B_VAL).unwrap();
        assert_eq!(decoded, values);
    }

    // 3.12: Reference test vector — hardcoded expected bytes for a known input
    #[test]
    fn reference_vector_all_ones_small() {
        // 4 values of [1,1,1,1] → shifted [0,0,0,0], median=0, k=0
        // Block header: 0x00 (k=0)
        // Each value: unary(0) = single 0-bit → 4 zero bits
        // Total: 1 header byte + ceil(4 bits / 8) = 1 body byte
        // Body bits: 0000 → padded to byte = 0x00
        let values = vec![1u32; 4];
        let encoded = rice_encode(&values, B_VAL).unwrap();
        assert_eq!(encoded, vec![0x00, 0x00]);

        let decoded = rice_decode(&encoded, 4, B_VAL).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn reference_vector_small_values() {
        // [1, 2] → shifted [0, 1], median=0, k=0
        // Header: 0x00
        // val 0: shifted=0, q=0, unary(0)=0-bit
        // val 1: shifted=1, q=1, unary(1)=1,0
        // Bits: 0 | 1 0 = 010 → padded to byte = 0b00000010 = 0x02
        let values = vec![1u32, 2];
        let encoded = rice_encode(&values, B_VAL).unwrap();
        assert_eq!(encoded, vec![0x00, 0x02]);

        let decoded = rice_decode(&encoded, 2, B_VAL).unwrap();
        assert_eq!(decoded, values);
    }

    // Verify decode error on truncated data
    #[test]
    fn decode_truncated_data() {
        let result = rice_decode(&[0x00], 100, B_VAL);
        assert!(result.is_err());
    }

    // Empty input
    #[test]
    fn round_trip_empty() {
        let values: Vec<u32> = vec![];
        let encoded = rice_encode(&values, B_VAL).unwrap();
        assert!(encoded.is_empty());
        let decoded = rice_decode(&encoded, 0, B_VAL).unwrap();
        assert!(decoded.is_empty());
    }

    // Large values
    #[test]
    fn round_trip_large_values() {
        let values = vec![1_000_000u32, 500_000, 100, 1];
        let encoded = rice_encode(&values, B_VAL).unwrap();
        let decoded = rice_decode(&encoded, values.len(), B_VAL).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn rice_encode_rejects_zero_values() {
        let values = vec![1, 0, 2, 3];
        let result = rice_encode(&values, B_VAL);
        assert!(result.is_err());
    }

    #[test]
    fn rice_encode_rejects_single_zero() {
        let result = rice_encode(&[0], B_VAL);
        assert!(result.is_err());
    }
}
