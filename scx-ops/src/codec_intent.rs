//! The codec intent axis for derived-file ops (`compact`, `merge`, `sort`,
//! `subset`).
//!
//! # Why this module exists
//!
//! Every op that rewrites a file used to build `FramingConfig::default()`, whose
//! `decode_target` is `None` — which is exactly the `fast` profile. So each of
//! them silently ran `fast` while `subset`/`sort` advertised `--codec auto` and
//! `compact`/`merge` offered no choice at all. On a `shufdelta` input that flips
//! every integer shard to the `Scx1`/`Zstd` heuristic, roughly doubling bytes per
//! nnz: `scx compact` *grew* an 864 MiB file by 5.1% while claiming to reclaim
//! 562 MB of orphaned bytes, and `scx sort --by` turned it into 1.7 GiB.
//!
//! These two helpers are the single place a derived-file op turns a
//! [`ResolvedCodec`] into (a) the framing config the writer needs and (b) the
//! per-shard seed codec, so `auto` means the same adaptive thing here as it does
//! in `scx convert`.

use scx_codec::{CodecId, CodecSelection, ValueEncoding};
use scx_format_io::{
    codec_select::select_codec, encoder::DEFAULT_ROW_GROUP_ROWS, FramingConfig, ResolvedCodec,
};

use crate::error::{OpsError, Result};

/// Framing config for a file-rewriting op, carrying the caller's codec intent.
///
/// `output_framed` is the op's existing "is the output v4?" decision (derived
/// from the input's `format_version`), so an unframed input keeps producing
/// unframed output. The intent's `decode_target` / `trial` ride along, which is
/// what makes the adaptive dual-encode reachable — see
/// [`scx_format_io::encode_shard_adaptive`].
///
/// Errors when a *profile* requires framing (`compact`, `compact-trial`) but the
/// output will be unframed, rather than silently degrading to `fast` — which is
/// the failure mode this whole module exists to remove. `auto` deliberately does
/// not require framing (it degrades to the heuristic single-encode).
///
/// An **explicit** codec force is exempt from that guard even when
/// `resolve_codec` marks it `requires_framing` (it does for `shufdelta`). These
/// ops have always accepted `--codec shufdelta` on an unframed v3 input and
/// written an unframed shufdelta shard: the encoding is valid, it just forgoes
/// sub-shard random access. Erroring there would be a new restriction unrelated
/// to the codec-intent bug, so the guard is scoped to the profiles, where
/// framing is what makes the dual-encode reachable in the first place.
pub fn framing_for_rewrite(
    codec: ResolvedCodec,
    output_framed: bool,
    what: &str,
) -> Result<Option<FramingConfig>> {
    if codec.requires_framing && codec.explicit_codec.is_none() && !output_framed {
        return Err(OpsError::InvalidInput(format!(
            "--codec {} needs a row-group-framed (format_version {}) input, but {what} is \
             unframed. Run `scx optimize --codec {}` first (it frames), or re-convert the \
             source with `--row-group-rows {}`. `--codec auto` works on an unframed input \
             (it falls back to the single-encode heuristic).",
            codec.profile,
            scx_format_io::CURRENT_FORMAT_VERSION,
            codec.profile,
            DEFAULT_ROW_GROUP_ROWS,
        )));
    }
    Ok(output_framed.then_some(FramingConfig {
        row_group_rows: DEFAULT_ROW_GROUP_ROWS,
        target_nnz: None,
        trial: codec.codec_trial,
        decode_target: codec.decode_target,
    }))
}

/// The per-shard *candidate* codec for a rewrite: an explicit force when the
/// caller pinned one, else the value-distribution heuristic.
///
/// This is only a seed. When the framing config carries `decode_target` the
/// writer may re-select it (adopting `ShufDeltaZstd` on integer shards that win
/// by [`scx_format_io::codec_select::ADOPT_MARGIN`]); the codec that lands in the
/// shard header is whatever [`scx_format_io::encode_shard_adaptive`] returns.
///
/// Replaces the bare `select_codec(...)` calls that were scattered across
/// `compact`/`merge`/`sort` — each of which hardcoded the heuristic and so could
/// not honour an explicit `--codec`.
pub fn seed_codec(codec: ResolvedCodec, values: &[u8], enc: ValueEncoding) -> CodecId {
    match codec.explicit_codec {
        // Mirrors `encode_one_shard_from_bytes`: Scx1 is integer-only, so a
        // float shard forced to Scx1 would hit a hard CodecError. Downgrade
        // rather than fail, matching the convert path.
        Some(CodecId::Scx1) if !enc.is_integer() => CodecId::Zstd,
        Some(c) => c,
        None => select_codec(values, enc),
    }
}

/// Lift a legacy [`CodecSelection`] into the intent axis.
///
/// Transitional: `AppendOptions.codec` is still a `CodecSelection`, so `append`
/// needs this to call the shared helpers. `CodecSelection` cannot express
/// `fast` / `compact` / `compact-trial`, so `Auto` lifts to the adaptive `auto`
/// profile and an explicit codec lifts to a single-encode force — matching what
/// [`scx_format_io::resolve_codec`] produces for the same spellings. Delete this
/// once `AppendOptions` carries a `ResolvedCodec`.
pub fn intent_from_codec_selection(sel: CodecSelection) -> ResolvedCodec {
    match sel {
        CodecSelection::Auto => ResolvedCodec::AUTO,
        CodecSelection::Explicit(c) => ResolvedCodec {
            explicit_codec: Some(c),
            codec_trial: false,
            decode_target: None,
            requires_framing: c == CodecId::ShufDeltaZstd,
            profile: "explicit",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_format_io::resolve_codec;

    fn rc(s: &str) -> ResolvedCodec {
        resolve_codec(Some(s)).unwrap()
    }

    #[test]
    fn auto_threads_the_adaptive_decode_target() {
        let fc = framing_for_rewrite(rc("auto"), true, "input.scx")
            .unwrap()
            .expect("framed output");
        assert!(
            fc.decode_target.is_some(),
            "auto must carry decode_target or the writer cannot adopt ShufDeltaZstd"
        );
        assert!(!fc.trial);
    }

    #[test]
    fn fast_is_the_single_encode_heuristic() {
        let fc = framing_for_rewrite(rc("fast"), true, "input.scx")
            .unwrap()
            .expect("framed output");
        assert_eq!(fc.decode_target, None);
        assert!(!fc.trial);
    }

    #[test]
    fn auto_degrades_on_an_unframed_output_rather_than_erroring() {
        assert!(framing_for_rewrite(rc("auto"), false, "v3.scx")
            .unwrap()
            .is_none());
    }

    #[test]
    fn compact_profile_refuses_an_unframed_output() {
        let err = framing_for_rewrite(rc("compact"), false, "v3.scx").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("scx optimize"), "actionable remedy: {msg}");
        assert!(msg.contains("auto"), "names the working alternative: {msg}");
        framing_for_rewrite(rc("compact-trial"), false, "v3.scx").unwrap_err();
    }

    #[test]
    fn an_explicit_shufdelta_is_allowed_on_an_unframed_output() {
        // `resolve_codec` marks explicit shufdelta `requires_framing`, but
        // sort/subset have always written unframed shufdelta on a v3 input.
        // Pinned so the profile guard is never widened to explicit forces.
        assert!(rc("shufdelta").requires_framing);
        assert!(framing_for_rewrite(rc("shufdelta"), false, "v3.scx")
            .unwrap()
            .is_none());
    }

    #[test]
    fn seed_honours_an_explicit_force_and_downgrades_scx1_on_floats() {
        assert_eq!(
            seed_codec(rc("zstd"), &[], ValueEncoding::Uint8),
            CodecId::Zstd
        );
        assert_eq!(
            seed_codec(rc("scx1"), &[], ValueEncoding::Uint8),
            CodecId::Scx1
        );
        // Scx1 is integer-only; a float shard must not be forced into it.
        assert_eq!(
            seed_codec(rc("scx1"), &[], ValueEncoding::Float32),
            CodecId::Zstd
        );
    }
}
