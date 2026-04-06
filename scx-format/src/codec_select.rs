// Auto-codec selection based on value distribution

use scx_codec::{CodecId, ValueEncoding};

/// Codec selection profile for user-facing codec choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecProfile {
    /// Automatic selection (Scx1 for small UMI integers, Zstd for larger/float).
    Auto,
    /// Optimizes for fast encode/decode: LZ4 with byte-shuffle.
    Fast,
    /// Optimizes for compression ratio: Zstd or Scx1 depending on data.
    Compact,
    /// Force the domain-specific Scx1 codec (integer only).
    Scx1,
}

/// Compute floor median of a u32 slice. Returns 0 for empty input.
fn floor_median_u32(values: &mut [u32]) -> u32 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    values[values.len() / 2]
}

/// Select codec using a profile hint.
///
/// - `Fast` → LZ4+shuffle for all data types.
/// - `Compact` → Scx1 for small UMI integers, Zstd for larger/float.
/// - `Scx1` → Force Scx1 (falls back to Zstd for float encodings).
/// - `Auto` → Same as `Compact` (backward-compatible default).
pub fn select_codec_with_profile(
    raw_values: &[u8],
    value_encoding: ValueEncoding,
    profile: CodecProfile,
) -> CodecId {
    match profile {
        CodecProfile::Fast => CodecId::Lz4Shuffle,
        CodecProfile::Scx1 => {
            if value_encoding.is_integer() {
                CodecId::Scx1
            } else {
                CodecId::Zstd
            }
        }
        CodecProfile::Auto | CodecProfile::Compact => select_codec(raw_values, value_encoding),
    }
}

/// Analyze raw value bytes and choose the best codec.
///
/// - Float/Float16 → always Zstd (Rice only handles integers)
/// - Integer → sample up to 10K raw value bytes, decode to u32, compute floor median
///   - median <= 8 → Scx1 (Rice optimal for small UMI counts)
///   - median > 8 → Zstd (LZ77 dictionary wins for larger values)
pub fn select_codec(raw_values: &[u8], value_encoding: ValueEncoding) -> CodecId {
    match value_encoding {
        ValueEncoding::Float32 | ValueEncoding::Float16 => CodecId::Zstd,

        ValueEncoding::Uint8 | ValueEncoding::Uint16 | ValueEncoding::Uint32 => {
            let bw = value_encoding.byte_width();
            if raw_values.is_empty() || bw == 0 {
                return CodecId::Zstd;
            }

            let n_values = raw_values.len() / bw;
            if n_values == 0 {
                return CodecId::Zstd;
            }

            // Sample up to 10K values
            let sample_count = n_values.min(10_000);
            let sample_bytes = sample_count * bw;

            let mut sample: Vec<u32> = match value_encoding {
                ValueEncoding::Uint8 => raw_values[..sample_bytes]
                    .iter()
                    .map(|&b| b as u32)
                    .collect(),
                ValueEncoding::Uint16 => raw_values[..sample_bytes]
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
                    .collect(),
                ValueEncoding::Uint32 => raw_values[..sample_bytes]
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                _ => unreachable!(),
            };

            let median = floor_median_u32(&mut sample);

            if median <= 8 {
                CodecId::Scx1
            } else {
                CodecId::Zstd
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_umi_values_select_scx1() {
        // Small UMI counts (median ~2) → Scx1
        let values: Vec<u8> = vec![1, 2, 1, 3, 2, 1, 1, 2, 4, 1];
        assert_eq!(select_codec(&values, ValueEncoding::Uint8), CodecId::Scx1);
    }

    #[test]
    fn test_large_values_select_zstd() {
        // Large values (uniform 0-1000 as u16) → Zstd
        let mut raw = Vec::new();
        for i in 0u16..1000 {
            raw.extend_from_slice(&i.to_le_bytes());
        }
        assert_eq!(select_codec(&raw, ValueEncoding::Uint16), CodecId::Zstd);
    }

    #[test]
    fn test_float_selects_zstd() {
        let raw = vec![0u8; 40]; // 10 float32 values
        assert_eq!(select_codec(&raw, ValueEncoding::Float32), CodecId::Zstd);
    }

    #[test]
    fn test_float16_selects_zstd() {
        let raw = vec![0u8; 20]; // 10 float16 values
        assert_eq!(select_codec(&raw, ValueEncoding::Float16), CodecId::Zstd);
    }

    #[test]
    fn test_empty_selects_zstd() {
        assert_eq!(select_codec(&[], ValueEncoding::Uint8), CodecId::Zstd);
    }

    #[test]
    fn test_floor_median() {
        let mut vals = vec![5, 1, 3, 2, 4];
        assert_eq!(floor_median_u32(&mut vals), 3);

        let mut vals2 = vec![10, 20, 30, 40];
        assert_eq!(floor_median_u32(&mut vals2), 30);

        let mut empty: Vec<u32> = vec![];
        assert_eq!(floor_median_u32(&mut empty), 0);
    }

    #[test]
    fn test_boundary_median_8_selects_scx1() {
        // All values are 8 → median = 8 → Scx1
        let values: Vec<u8> = vec![8; 100];
        assert_eq!(select_codec(&values, ValueEncoding::Uint8), CodecId::Scx1);
    }

    #[test]
    fn test_boundary_median_9_selects_zstd() {
        // All values are 9 → median = 9 → Zstd
        let values: Vec<u8> = vec![9; 100];
        assert_eq!(select_codec(&values, ValueEncoding::Uint8), CodecId::Zstd);
    }

    #[test]
    fn test_uint32_large_values_select_zstd() {
        let mut raw = Vec::new();
        for i in 100u32..200 {
            raw.extend_from_slice(&i.to_le_bytes());
        }
        assert_eq!(select_codec(&raw, ValueEncoding::Uint32), CodecId::Zstd);
    }
}
