// Adaptive Rice encoder/decoder for values (docs/codec.md (Values))
//
// Implementation follows the adaptive Rice-coding scheme described in
// Malvar, H. S. "Adaptive run-length / Golomb-Rice encoding of quantised
// generalised Gaussian sources with unknown statistics." Data Compression
// Conference, 2006 (DCC'06), pp. 23–32. Block size B_VAL=256 matches the
// reference; the k-selection rule (k ≈ ⌊log2(0.6931 · median)⌋) is the
// standard Gaussian-geometric estimator used by JPEG-LS / HP-LOCO.

use crate::bitstream::{BitReader, BitStreamError, BitWriter};
use crate::dispatch::CodecError;
use crate::median::floor_median_u32_inplace;

/// Default block size for Rice coding of values.
pub const B_VAL: usize = 256;

/// Maximum Rice parameter `k`, single source of truth for the clamp in
/// `compute_k` and the decode-side parse-boundary rejection in both this
/// module and `delta_golomb`. `k` is encoded in a 4-bit field, so values
/// above 15 are unrepresentable and indicate a corrupt/hostile stream.
pub const MAX_RICE_K: u8 = 15;

/// Per-block decode metadata produced by the actual Rice encoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RiceBlockMetadata {
    pub value_start: u64,
    pub n_values: u16,
    pub bit_offset: u64,
    pub k: u8,
}

/// Encoded Rice bytes plus per-block metadata from the same encode pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RiceEncodeResult {
    pub bytes: Vec<u8>,
    pub blocks: Vec<RiceBlockMetadata>,
}

/// Compute the Rice parameter k from the median of shifted values.
///
/// `k = clamp(floor(log2(0.6931 * median)), 0, MAX_RICE_K)`.
fn compute_k(median: u32) -> u8 {
    if median == 0 {
        return 0;
    }
    let raw = (std::f64::consts::LN_2 * median as f64).log2().floor() as i32;
    raw.clamp(0, MAX_RICE_K as i32) as u8
}

/// Encode non-zero count values using blocked Rice coding.
///
/// All values must be >= 1 (non-zero counts). They are shifted by -1 before encoding.
/// The output is a byte vector containing the encoded bitstream.
pub fn rice_encode(values: &[u32], block_size: usize) -> Result<Vec<u8>, BitStreamError> {
    Ok(rice_encode_with_metadata(values, block_size)?.bytes)
}

/// Encode non-zero count values and return the exact per-block decode metadata
/// observed during encoding.
pub fn rice_encode_with_metadata(
    values: &[u32],
    block_size: usize,
) -> Result<RiceEncodeResult, BitStreamError> {
    if values.contains(&0) {
        return Err(BitStreamError);
    }

    let mut writer = BitWriter::new();
    let mut blocks = Vec::with_capacity(values.len().div_ceil(block_size));
    let mut value_start = 0u64;

    for chunk in values.chunks(block_size) {
        if chunk.len() > u16::MAX as usize {
            return Err(BitStreamError);
        }
        // Shift: subtract 1 from each value. This buffer is a throwaway scratch
        // used only to derive the Rice parameter, so the median selection may
        // reorder it in place (the encode loop below reads `chunk` directly).
        let mut shifted: Vec<u32> = chunk.iter().map(|&v| v - 1).collect();

        // Compute Rice parameter k
        let median = floor_median_u32_inplace(&mut shifted);
        let k = compute_k(median);
        let bit_offset = writer.position() as u64;

        blocks.push(RiceBlockMetadata {
            value_start,
            n_values: chunk.len() as u16,
            bit_offset,
            k,
        });

        // Write block header: 1 byte with k in low nibble
        writer.write_bits(k as u64, 8);

        // Encode each shifted value, recomputing the shift from `chunk` so the
        // values are emitted in original order (`shifted` was reordered above).
        for &v in chunk {
            let s = v - 1;
            let q = (s >> k) as u64;
            let r = (s & ((1u32 << k) - 1)) as u64;
            writer.write_unary(q);
            if k > 0 {
                writer.write_bits(r, k);
            }
        }

        // Pad to byte boundary after each block
        writer.pad_to_byte();
        value_start += chunk.len() as u64;
    }

    Ok(RiceEncodeResult {
        bytes: writer.flush(),
        blocks,
    })
}

/// Decode Rice-encoded values by seeking to encoder-produced block offsets.
pub fn rice_decode_with_metadata(
    data: &[u8],
    blocks: &[RiceBlockMetadata],
) -> Result<Vec<u32>, CodecError> {
    let n_values: usize = blocks.iter().map(|b| b.n_values as usize).sum();
    let mut output = Vec::with_capacity(n_values);

    for (block_idx, block) in blocks.iter().enumerate() {
        if block.n_values == 0 || block.k > MAX_RICE_K {
            return Err(CodecError::MalformedInput(format!(
                "invalid Rice metadata for block {block_idx}: n_values={}, k={}",
                block.n_values, block.k
            )));
        }
        if block.value_start != output.len() as u64 {
            return Err(CodecError::MalformedInput(format!(
                "Rice metadata block {block_idx} starts at value {}, expected {}",
                block.value_start,
                output.len()
            )));
        }
        let bit_offset = usize::try_from(block.bit_offset).map_err(|_| {
            CodecError::MalformedInput(format!(
                "Rice metadata block {block_idx} bit offset overflows usize"
            ))
        })?;
        let mut reader = BitReader::new_at(data, bit_offset)?;

        let block_header = reader.read_bits(8)? as u8;
        if block_header & 0xF0 != 0 {
            return Err(CodecError::MalformedInput(format!(
                "Rice block header has non-zero reserved high nibble: 0x{:02x}",
                block_header
            )));
        }
        let k = block_header & 0x0F;
        if k != block.k {
            return Err(CodecError::MalformedInput(format!(
                "Rice metadata block {block_idx} k {} != encoded k {}",
                block.k, k
            )));
        }

        for _ in 0..block.n_values {
            let q = reader.read_unary()?;
            let r = if k > 0 { reader.read_bits(k)? } else { 0 };
            let value = q
                .checked_shl(k as u32)
                .map(|qk| qk | r)
                .and_then(|shifted| shifted.checked_add(1))
                .filter(|&v| v <= u32::MAX as u64)
                .ok_or_else(|| {
                    CodecError::MalformedInput(
                        "Rice value overflows u32 (corrupt stream)".to_string(),
                    )
                })?;
            output.push(value as u32);
        }
    }

    Ok(output)
}

/// Decode Rice-encoded values.
///
/// Returns `n_values` decoded values, each >= 1.
pub fn rice_decode(
    data: &[u8],
    n_values: usize,
    block_size: usize,
) -> Result<Vec<u32>, CodecError> {
    let mut output = Vec::with_capacity(n_values);
    let mut reader = BitReader::new(data);
    let mut remaining = n_values;

    while remaining > 0 {
        let block_len = remaining.min(block_size);

        // Read block header byte. Spec §4 reserves the high nibble (must be
        // zero); a non-zero value indicates corruption — fail loudly rather
        // than silently masking it off.
        let block_header = reader.read_bits(8)? as u8;
        if block_header & 0xF0 != 0 {
            return Err(CodecError::MalformedInput(format!(
                "Rice block header has non-zero reserved high nibble: 0x{:02x}",
                block_header
            )));
        }
        let k = block_header & 0x0F;

        for _ in 0..block_len {
            // Keep the unary quotient as u64: casting to u32 first would
            // silently truncate a long (hostile) unary run before the shift.
            let q = reader.read_unary()?;
            let r = if k > 0 { reader.read_bits(k)? } else { 0 };
            // `(q << k) | r`, then `+1`, all range-checked against u32 so a
            // crafted stream returns MalformedInput rather than wrapping in
            // release (overflow-checks are off there without the profile flag).
            let value = q
                .checked_shl(k as u32)
                .map(|qk| qk | r)
                .and_then(|shifted| shifted.checked_add(1))
                .filter(|&v| v <= u32::MAX as u64)
                .ok_or_else(|| {
                    CodecError::MalformedInput(
                        "Rice value overflows u32 (corrupt stream)".to_string(),
                    )
                })?;
            output.push(value as u32);
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
    use crate::median::floor_median_u32;

    // --- Helper tests ---

    // floor_median tests have moved to crate::median::tests

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
        assert_eq!(floor_median_u32(&shifted), 3);
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

    // F2: a decoded value that overflows u32 (here `(q << k) + 1` with a
    // crafted quotient) must return MalformedInput rather than wrapping.
    #[test]
    fn rice_decode_rejects_u32_overflow() {
        let mut w = BitWriter::new();
        w.write_bits(MAX_RICE_K as u64, 8); // block header: k = 15
                                            // q = 2^17 → (q << 15) = 2^32 > u32::MAX, so value overflows u32.
        w.write_unary(1u64 << 17);
        w.write_bits(0, MAX_RICE_K); // r bits (k = 15)
        w.pad_to_byte();
        let data = w.flush();
        let result = rice_decode(&data, 1, B_VAL);
        assert!(
            matches!(result, Err(CodecError::MalformedInput(_))),
            "expected MalformedInput, got {result:?}"
        );
    }
}
