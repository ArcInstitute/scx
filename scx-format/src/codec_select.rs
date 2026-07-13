// Auto-codec selection based on value distribution

use crate::modality::ModalityType;
use scx_codec::floor_median_u32;
use scx_codec::{CodecId, ValueEncoding};

/// Internal mechanism backing the user-facing codec **intent axis**
/// (`codec="auto" | "fast" | "compact"`).
///
/// Biases the per-modality integer codec choice between the size/GPU-friendly
/// `ShufDeltaZstd` and the CPU-conservative heuristic winner (`Scx1`/`Zstd`).
/// Not user-settable: the writer entry points map the codec string to a variant
/// (`"auto"` → [`DecodeTarget::Auto`], `"fast"` → [`DecodeTarget::Cpu`],
/// `"compact"` → [`DecodeTarget::Storage`]). See [`pick_codec_v2`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodeTarget {
    /// Balanced default (`codec="auto"`): adopt ShufDeltaZstd for integers only
    /// when it is smaller by at least [`ADOPT_MARGIN`] (cost-aware — a marginal
    /// size win never pays the ShufDeltaZstd CPU-decode tax).
    #[default]
    Auto,
    /// Decode-speed-max (`codec="fast"`): keep the heuristic winner (never
    /// ShufDeltaZstd).
    Cpu,
    /// GPU training: prefer ShufDeltaZstd (GPU-decodable) on size ties.
    Gpu,
    /// Size-max (`codec="compact"`): prefer ShufDeltaZstd (smaller payload) on
    /// size ties.
    Storage,
}

/// Minimum fractional size win required before `codec="auto"` adopts
/// ShufDeltaZstd over the heuristic winner. Offsets the ~6–9% ShufDeltaZstd
/// CPU-decode tax so a marginal size gain doesn't erode decode speed by default;
/// the large (1.3–2×) real wins clear it comfortably. `codec="compact"` ignores
/// this (adopts on ties). Tunable.
pub const ADOPT_MARGIN: f64 = 0.05;

/// Pick the final integer codec for the adaptive profiles (`auto`/`compact`),
/// given the heuristic winner and the measured framed sizes of the heuristic vs
/// a ShufDeltaZstd trial-encode.
///
/// `ShufDeltaZstd` only competes for **integer** data whose heuristic winner is
/// not already ShufDeltaZstd (float always stays Pcodec, so this returns the
/// heuristic unchanged there). Decision matrix:
///
/// - `Cpu` (`fast`) → `heuristic` (conservative; keeps Scx1 for low-median counts).
/// - `Auto` (`auto`) → ShufDeltaZstd iff it is smaller by at least [`ADOPT_MARGIN`]
///   (`size_shufdelta < size_heuristic * (1 - ADOPT_MARGIN)`); a within-margin win
///   or tie keeps the heuristic (cost-aware — don't pay the decode tax for a marginal
///   size gain).
/// - `Gpu` / `Storage` (`compact`) → ShufDeltaZstd iff `size_shufdelta <= size_heuristic`
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
        // `size_heuristic == 0` (empty shard) → threshold is 0.0, `< 0.0` is
        // false, so we keep the heuristic — no division/overflow hazard.
        DecodeTarget::Auto => {
            (size_shufdelta as f64) < (size_heuristic as f64) * (1.0 - ADOPT_MARGIN)
        }
        DecodeTarget::Gpu | DecodeTarget::Storage => size_shufdelta <= size_heuristic,
    };
    if adopt {
        CodecId::ShufDeltaZstd
    } else {
        heuristic
    }
}

/// A user-facing `codec=` string resolved into the encoder's knobs.
///
/// The single source of truth for the codec **intent axis** shared by every
/// writer entry point (`scx convert`/`scx optimize`, `pyscx.from_anndata`/
/// `from_h5ad`/`from_10x`). Collapses what used to be four near-identical
/// string-match blocks. See [`resolve_codec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedCodec {
    /// Explicit codec force. `None` = adaptive/heuristic (decided per shard by
    /// the profile + [`pick_codec_v2`]).
    pub explicit_codec: Option<CodecId>,
    /// `compact-trial`: strict-`<` dual-encode (keep the smaller of
    /// {heuristic, ShufDeltaZstd} per shard).
    pub codec_trial: bool,
    /// Adaptive dual-encode bias for `auto`/`compact`. `None` = single-encode
    /// the heuristic (`fast`, or an explicit codec). Only takes effect on
    /// row-group-framed writes.
    pub decode_target: Option<DecodeTarget>,
    /// Whether this profile requires row-group framing (`row_group_rows > 0`).
    /// `auto` is deliberately `false`: it silently falls back to the heuristic
    /// single-encode when unframed rather than erroring (it is the default).
    pub requires_framing: bool,
    /// Profile name for the provenance `codec_selection` stamp.
    pub profile: &'static str,
}

/// Resolve a user-facing `codec=` string into [`ResolvedCodec`].
///
/// Intent axis: `auto` (default, cost-aware adaptive), `fast` (decode-max
/// heuristic), `compact` (size-max, adopt on ties). `compact-trial` and the
/// explicit codec forces (`none`/`scx1`/`zstd`/`lz4`/`pcodec`/`shufdelta`) are
/// retained. `auto_v2` was removed (pre-1.0 clean break) and errors with a
/// message naming its replacements.
pub fn resolve_codec(codec: Option<&str>) -> Result<ResolvedCodec, String> {
    let mk = |explicit_codec, codec_trial, decode_target, requires_framing, profile| {
        Ok(ResolvedCodec {
            explicit_codec,
            codec_trial,
            decode_target,
            requires_framing,
            profile,
        })
    };
    match codec {
        // Default: cost-aware adaptive. Framing not required — unframed writes
        // fall back to the heuristic single-encode (see the encoder).
        None | Some("auto") => mk(None, false, Some(DecodeTarget::Auto), false, "auto"),
        // Decode-speed-max: heuristic single-encode (the old `auto`).
        Some("fast") => mk(None, false, None, false, "fast"),
        // Size-max: adopt ShufDeltaZstd on ties. Framed only.
        Some("compact") => mk(None, false, Some(DecodeTarget::Storage), true, "compact"),
        // Compact-trial: strict-`<` dual-encode. Framed only.
        Some("compact-trial") => mk(None, true, None, true, "compact-trial"),
        Some("auto_v2") => Err(
            "codec='auto_v2' was removed. Use codec='auto' (cost-aware adaptive default, \
             adopts ShufDeltaZstd where it wins by a margin) or codec='compact' \
             (size-max, adopts on ties). The `decode_target` knob was removed with it."
                .to_string(),
        ),
        // Explicit codec force (`None`/`auto` and the profiles are handled above,
        // so `name` here is always a concrete codec string or unknown).
        Some(name) => match CodecId::parse_cli(name) {
            Ok(cid) => {
                let requires_framing = cid == Some(CodecId::ShufDeltaZstd);
                let profile = cid.map(|c| c.display_name()).unwrap_or("auto");
                mk(cid, false, None, requires_framing, profile)
            }
            Err(_) => Err(format!(
                "Unknown codec: '{name}'. Use 'auto' (default, adaptive), 'fast' (decode-max), \
                 'compact' (size-max), 'compact-trial', 'none', 'scx1', 'zstd', 'lz4', \
                 'pcodec', or 'shufdelta'."
            )),
        },
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
    fn pick_v2_auto_adopts_only_past_margin() {
        // ADOPT_MARGIN = 0.05 → threshold for size_heuristic=100 is 95.0.
        // Clears the margin (94 < 95) → adopt.
        assert_eq!(
            pick_codec_v2(CodecId::Scx1, DecodeTarget::Auto, 100, 94, true),
            CodecId::ShufDeltaZstd
        );
        // Exactly at the margin (95 is not < 95) → keep heuristic.
        assert_eq!(
            pick_codec_v2(CodecId::Scx1, DecodeTarget::Auto, 100, 95, true),
            CodecId::Scx1
        );
        // Within the margin (a marginal size win) → keep heuristic (cost-aware).
        assert_eq!(
            pick_codec_v2(CodecId::Scx1, DecodeTarget::Auto, 100, 99, true),
            CodecId::Scx1
        );
        // Tie → keep heuristic.
        assert_eq!(
            pick_codec_v2(CodecId::Scx1, DecodeTarget::Auto, 100, 100, true),
            CodecId::Scx1
        );
        // Larger → keep heuristic.
        assert_eq!(
            pick_codec_v2(CodecId::Scx1, DecodeTarget::Auto, 100, 101, true),
            CodecId::Scx1
        );
        // Large real win (1.5×) clears the margin comfortably.
        assert_eq!(
            pick_codec_v2(CodecId::Zstd, DecodeTarget::Auto, 150, 100, true),
            CodecId::ShufDeltaZstd
        );
    }

    #[test]
    fn pick_v2_auto_margin_invariant() {
        // Invariant: `auto` never adopts ShufDeltaZstd unless it is strictly
        // smaller than heuristic*(1-ADOPT_MARGIN). Sweep a grid.
        for size_heuristic in [1usize, 10, 100, 1000, 65536] {
            let threshold = (size_heuristic as f64) * (1.0 - ADOPT_MARGIN);
            for delta in 0..=size_heuristic + 5 {
                let size_shufdelta = delta;
                let picked = pick_codec_v2(
                    CodecId::Scx1,
                    DecodeTarget::Auto,
                    size_heuristic,
                    size_shufdelta,
                    true,
                );
                let adopted = picked == CodecId::ShufDeltaZstd;
                assert_eq!(
                    adopted,
                    (size_shufdelta as f64) < threshold,
                    "h={size_heuristic} s={size_shufdelta}: adopted={adopted}"
                );
            }
        }
        // Empty shard (heuristic size 0) never adopts.
        assert_eq!(
            pick_codec_v2(CodecId::Scx1, DecodeTarget::Auto, 0, 0, true),
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
    fn resolve_codec_intent_axis() {
        // Default / auto: adaptive, framing not required.
        for c in [None, Some("auto")] {
            let r = resolve_codec(c).unwrap();
            assert_eq!(r.explicit_codec, None);
            assert!(!r.codec_trial);
            assert_eq!(r.decode_target, Some(DecodeTarget::Auto));
            assert!(!r.requires_framing);
            assert_eq!(r.profile, "auto");
        }
        // fast: heuristic single-encode (old auto), framing not required.
        let r = resolve_codec(Some("fast")).unwrap();
        assert_eq!(r.explicit_codec, None);
        assert!(!r.codec_trial);
        assert_eq!(r.decode_target, None);
        assert!(!r.requires_framing);
        assert_eq!(r.profile, "fast");
        // compact: tie-adopt, framed only.
        let r = resolve_codec(Some("compact")).unwrap();
        assert_eq!(r.decode_target, Some(DecodeTarget::Storage));
        assert!(r.requires_framing);
        assert_eq!(r.profile, "compact");
        // compact-trial: strict trial, framed only.
        let r = resolve_codec(Some("compact-trial")).unwrap();
        assert!(r.codec_trial);
        assert_eq!(r.decode_target, None);
        assert!(r.requires_framing);
        // explicit forces.
        let r = resolve_codec(Some("scx1")).unwrap();
        assert_eq!(r.explicit_codec, Some(CodecId::Scx1));
        assert!(!r.requires_framing);
        assert_eq!(r.profile, "scx1");
        let r = resolve_codec(Some("shufdelta")).unwrap();
        assert_eq!(r.explicit_codec, Some(CodecId::ShufDeltaZstd));
        assert!(r.requires_framing);
    }

    #[test]
    fn resolve_codec_auto_v2_removed() {
        let err = resolve_codec(Some("auto_v2")).unwrap_err();
        assert!(err.contains("auto_v2"), "{err}");
        assert!(err.contains("auto"), "{err}");
        assert!(err.contains("compact"), "{err}");
    }

    #[test]
    fn resolve_codec_unknown_errors() {
        let err = resolve_codec(Some("gzip")).unwrap_err();
        assert!(err.contains("gzip"), "{err}");
        assert!(err.contains("fast") && err.contains("compact"), "{err}");
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
