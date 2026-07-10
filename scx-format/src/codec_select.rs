// Auto-codec selection based on value distribution

use crate::modality::ModalityType;
use scx_codec::floor_median_u32;
use scx_codec::{CodecId, ValueEncoding};

/// Codec selection profile for user-facing codec choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecProfile {
    /// Automatic selection (Scx1 for small UMI integers, Pcodec for floats, Zstd for larger integers).
    Auto,
    /// Optimizes for fast encode/decode: LZ4 with byte-shuffle.
    Fast,
    /// Optimizes for compression ratio: Zstd or Scx1 depending on data.
    Compact,
    /// Force the domain-specific Scx1 codec (integer only).
    Scx1,
}

/// Downstream decode target for the workload-aware `auto_v2` codec profile.
///
/// Biases the per-modality integer codec choice between the size/GPU-friendly
/// `ShufDeltaZstd` and the CPU-conservative heuristic winner (`Scx1`/`Zstd`).
/// Since Phase B brought ShufDeltaZstd CPU decode to ~parity with Scx1, this is
/// a transitional knob: for integers `Auto` already behaves like `compact-trial`
/// (adopt ShufDeltaZstd where it compresses at least as well). See [`pick_codec_v2`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeTarget {
    /// Balanced default: adopt ShufDeltaZstd for integers when strictly smaller.
    #[default]
    Auto,
    /// Conservative: keep the heuristic winner (never ShufDeltaZstd).
    Cpu,
    /// GPU training: prefer ShufDeltaZstd (GPU-decodable) on size ties.
    Gpu,
    /// Cloud/egress: prefer ShufDeltaZstd (smaller payload) on size ties.
    Storage,
}

/// Pick the final integer codec for the `auto_v2` profile, given the heuristic
/// winner and the measured framed sizes of the heuristic vs a ShufDeltaZstd
/// trial-encode.
///
/// `ShufDeltaZstd` only competes for **integer** data whose heuristic winner is
/// not already ShufDeltaZstd (float always stays Pcodec, so this returns the
/// heuristic unchanged there). Decision matrix:
///
/// - `Cpu` → `heuristic` (conservative; keeps Scx1 for low-median counts).
/// - `Auto` → ShufDeltaZstd iff `size_shufdelta < size_heuristic` (tie → heuristic;
///   matches the `compact-trial` "keep the smaller" rule).
/// - `Gpu` / `Storage` → ShufDeltaZstd iff `size_shufdelta <= size_heuristic`
///   (tie → ShufDeltaZstd: GPU-decodable / smaller egress).
pub fn pick_codec_v2(
    heuristic: CodecId,
    decode_target: DecodeTarget,
    size_heuristic: usize,
    size_shufdelta: usize,
    is_integer: bool,
) -> CodecId {
    if !is_integer || heuristic == CodecId::ShufDeltaZstd {
        return heuristic;
    }
    let adopt = match decode_target {
        DecodeTarget::Cpu => false,
        DecodeTarget::Auto => size_shufdelta < size_heuristic,
        DecodeTarget::Gpu | DecodeTarget::Storage => size_shufdelta <= size_heuristic,
    };
    if adopt {
        CodecId::ShufDeltaZstd
    } else {
        heuristic
    }
}

/// Select codec using a profile hint.
///
/// - `Fast` → LZ4+shuffle for all data types.
/// - `Compact` → Scx1 for small UMI integers, Pcodec for float, Zstd for larger integers.
/// - `Scx1` → Force Scx1 (falls back to Pcodec for float encodings).
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
                CodecId::Pcodec
            }
        }
        CodecProfile::Auto | CodecProfile::Compact => select_codec(raw_values, value_encoding),
    }
}

/// Analyze raw value bytes and choose the best codec.
///
/// - Float/Float16 → always Pcodec (optimal for float data)
/// - Integer → sample up to 10K raw value bytes, decode to u32, compute floor median
///   - median <= 8 → Scx1 (Rice optimal for small UMI counts)
///   - median > 8 → Zstd (LZ77 dictionary wins for larger values)
pub fn select_codec(raw_values: &[u8], value_encoding: ValueEncoding) -> CodecId {
    match value_encoding {
        ValueEncoding::Float32 | ValueEncoding::Float16 => CodecId::Pcodec,

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

            let sample: Vec<u32> = match value_encoding {
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

            let median = floor_median_u32(&sample);

            if median <= 8 {
                CodecId::Scx1
            } else {
                CodecId::Zstd
            }
        }
    }
}

/// Select codec using the modality's biological type as a hint.
///
/// Same shape as [`select_codec`] but routes per-modality:
///
/// - RNA / Custom / Methylation / Spatial → delegate to [`select_codec`]
///   (Scx1 for small UMI integer medians, Zstd otherwise; Pcodec for
///   floats — already correct for spatial float coordinates and
///   methylation per-CpG counts).
/// - Protein/ADT → Zstd for integers (Rice's UMI-distribution
///   assumption breaks for ADT counts), Pcodec for float CLR layers.
/// - ATAC → Zstd for binary peak presence (sample max ≤ 1, uint8 by
///   convention), Lz4Shuffle for integer peak counts, Pcodec for floats.
pub fn select_codec_for_modality(
    raw_values: &[u8],
    value_encoding: ValueEncoding,
    modality_type: ModalityType,
) -> CodecId {
    match modality_type {
        ModalityType::Rna
        | ModalityType::Custom
        | ModalityType::Spatial
        | ModalityType::Methylation => select_codec(raw_values, value_encoding),
        ModalityType::Protein => match value_encoding {
            ValueEncoding::Float32 | ValueEncoding::Float16 => CodecId::Pcodec,
            _ => CodecId::Zstd,
        },
        ModalityType::Atac => match value_encoding {
            ValueEncoding::Float32 | ValueEncoding::Float16 => CodecId::Pcodec,
            ValueEncoding::Uint8 if atac_sample_is_binary(raw_values) => CodecId::Zstd,
            _ => CodecId::Lz4Shuffle,
        },
    }
}

/// Heuristic: a uint8 ATAC payload is treated as binary peak-presence
/// when every sampled byte is in {0, 1}. Reuses the 10K-byte sampling
/// pattern from [`select_codec`].
fn atac_sample_is_binary(raw: &[u8]) -> bool {
    if raw.is_empty() {
        return false;
    }
    let n = raw.len().min(10_000);
    raw[..n].iter().all(|&b| b <= 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_v2_cpu_keeps_heuristic() {
        // Conservative: never adopt ShufDeltaZstd, even when strictly smaller.
        assert_eq!(
            pick_codec_v2(CodecId::Scx1, DecodeTarget::Cpu, 100, 50, true),
            CodecId::Scx1
        );
        assert_eq!(
            pick_codec_v2(CodecId::Zstd, DecodeTarget::Cpu, 100, 50, true),
            CodecId::Zstd
        );
    }

    #[test]
    fn pick_v2_auto_adopts_when_strictly_smaller() {
        assert_eq!(
            pick_codec_v2(CodecId::Scx1, DecodeTarget::Auto, 100, 99, true),
            CodecId::ShufDeltaZstd
        );
        // Tie → keep heuristic (compact-trial semantics).
        assert_eq!(
            pick_codec_v2(CodecId::Scx1, DecodeTarget::Auto, 100, 100, true),
            CodecId::Scx1
        );
        // Larger → keep heuristic.
        assert_eq!(
            pick_codec_v2(CodecId::Scx1, DecodeTarget::Auto, 100, 101, true),
            CodecId::Scx1
        );
    }

    #[test]
    fn pick_v2_gpu_storage_prefer_shufdelta_on_ties() {
        for dt in [DecodeTarget::Gpu, DecodeTarget::Storage] {
            // Tie → ShufDeltaZstd (GPU-decodable / smaller egress).
            assert_eq!(
                pick_codec_v2(CodecId::Scx1, dt, 100, 100, true),
                CodecId::ShufDeltaZstd
            );
            assert_eq!(
                pick_codec_v2(CodecId::Scx1, dt, 100, 90, true),
                CodecId::ShufDeltaZstd
            );
            // Strictly larger → keep heuristic.
            assert_eq!(
                pick_codec_v2(CodecId::Scx1, dt, 100, 101, true),
                CodecId::Scx1
            );
        }
    }

    #[test]
    fn pick_v2_float_never_adopts_shufdelta() {
        // Non-integer: heuristic (Pcodec) is returned regardless of target/size.
        for dt in [
            DecodeTarget::Auto,
            DecodeTarget::Cpu,
            DecodeTarget::Gpu,
            DecodeTarget::Storage,
        ] {
            assert_eq!(
                pick_codec_v2(CodecId::Pcodec, dt, 100, 1, false),
                CodecId::Pcodec
            );
        }
    }

    #[test]
    fn pick_v2_heuristic_already_shufdelta_is_noop() {
        assert_eq!(
            pick_codec_v2(CodecId::ShufDeltaZstd, DecodeTarget::Cpu, 100, 200, true),
            CodecId::ShufDeltaZstd
        );
    }

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
    fn test_float_selects_pcodec() {
        let raw = vec![0u8; 40]; // 10 float32 values
        assert_eq!(select_codec(&raw, ValueEncoding::Float32), CodecId::Pcodec);
    }

    #[test]
    fn test_float16_selects_pcodec() {
        let raw = vec![0u8; 20]; // 10 float16 values
        assert_eq!(select_codec(&raw, ValueEncoding::Float16), CodecId::Pcodec);
    }

    #[test]
    fn test_empty_selects_zstd() {
        assert_eq!(select_codec(&[], ValueEncoding::Uint8), CodecId::Zstd);
    }

    #[test]
    fn test_floor_median() {
        let vals = vec![5, 1, 3, 2, 4];
        assert_eq!(floor_median_u32(&vals), 3);

        let vals2 = vec![10, 20, 30, 40];
        // Standardized: even-length returns lower middle (20), not upper (30)
        assert_eq!(floor_median_u32(&vals2), 20);

        let empty: Vec<u32> = vec![];
        assert_eq!(floor_median_u32(&empty), 0);
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

    // --- select_codec_for_modality coverage --------------------------------

    #[test]
    fn protein_uint8_small_median_uses_zstd() {
        // Same fixture that picks Scx1 for RNA — Protein should override to Zstd.
        let values: Vec<u8> = vec![1, 2, 1, 3, 2, 1, 1, 2, 4, 1];
        assert_eq!(
            select_codec_for_modality(&values, ValueEncoding::Uint8, ModalityType::Protein),
            CodecId::Zstd
        );
    }

    #[test]
    fn protein_float_uses_pcodec() {
        let raw = vec![0u8; 40]; // 10 float32 values
        assert_eq!(
            select_codec_for_modality(&raw, ValueEncoding::Float32, ModalityType::Protein),
            CodecId::Pcodec
        );
    }

    #[test]
    fn atac_binary_uint8_uses_zstd() {
        let values: Vec<u8> = vec![0, 1, 0, 1, 1, 0, 1, 0, 0, 1];
        assert_eq!(
            select_codec_for_modality(&values, ValueEncoding::Uint8, ModalityType::Atac),
            CodecId::Zstd
        );
    }

    #[test]
    fn atac_count_uint8_uses_lz4shuffle() {
        let values: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 0, 1, 2, 3];
        assert_eq!(
            select_codec_for_modality(&values, ValueEncoding::Uint8, ModalityType::Atac),
            CodecId::Lz4Shuffle
        );
    }

    #[test]
    fn atac_uint16_counts_use_lz4shuffle() {
        let mut raw = Vec::new();
        for i in 0u16..20 {
            raw.extend_from_slice(&i.to_le_bytes());
        }
        assert_eq!(
            select_codec_for_modality(&raw, ValueEncoding::Uint16, ModalityType::Atac),
            CodecId::Lz4Shuffle
        );
    }

    #[test]
    fn atac_float_uses_pcodec() {
        let raw = vec![0u8; 40]; // 10 float32 values
        assert_eq!(
            select_codec_for_modality(&raw, ValueEncoding::Float32, ModalityType::Atac),
            CodecId::Pcodec
        );
    }

    #[test]
    fn rna_falls_through_to_select_codec() {
        let small: Vec<u8> = vec![1, 2, 1, 3, 2, 1, 1, 2, 4, 1];
        assert_eq!(
            select_codec_for_modality(&small, ValueEncoding::Uint8, ModalityType::Rna),
            CodecId::Scx1
        );
        let mut large = Vec::new();
        for i in 0u16..1000 {
            large.extend_from_slice(&i.to_le_bytes());
        }
        assert_eq!(
            select_codec_for_modality(&large, ValueEncoding::Uint16, ModalityType::Rna),
            CodecId::Zstd
        );
    }

    #[test]
    fn custom_falls_through_to_select_codec() {
        let small: Vec<u8> = vec![1, 2, 1, 3, 2, 1, 1, 2, 4, 1];
        assert_eq!(
            select_codec_for_modality(&small, ValueEncoding::Uint8, ModalityType::Custom),
            CodecId::Scx1
        );
    }

    #[test]
    fn spatial_float_uses_pcodec() {
        let raw = vec![0u8; 40];
        assert_eq!(
            select_codec_for_modality(&raw, ValueEncoding::Float32, ModalityType::Spatial),
            CodecId::Pcodec
        );
    }

    #[test]
    fn methylation_falls_through_to_select_codec() {
        let small: Vec<u8> = vec![1, 2, 1, 3, 2, 1, 1, 2, 4, 1];
        assert_eq!(
            select_codec_for_modality(&small, ValueEncoding::Uint8, ModalityType::Methylation),
            CodecId::Scx1
        );
    }

    #[test]
    fn atac_sample_is_binary_helper() {
        assert!(atac_sample_is_binary(&[0, 1, 0, 1, 1]));
        assert!(!atac_sample_is_binary(&[0, 1, 2]));
        assert!(!atac_sample_is_binary(&[]));
    }
}
