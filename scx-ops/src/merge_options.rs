//! Options and policy types for [`crate::merge::merge_with_options`].
//!
//! Splits out the policy surface from `merge.rs` so the merge function
//! signature stays readable as we add var-identity validation, uns
//! conflict handling, and shard-target overrides.

use scx_engine::ConversionPredicateIndexOptions;

/// Top-level merge configuration. Default reproduces today's behaviour
/// (no predicate-index work, accept var identity on count alone, keep
/// first input's uns) so callers that didn't opt in stay byte-identical
/// across the Phase 2 refactor.
#[derive(Debug, Clone)]
pub struct MergeOptions {
    /// Predicate-index build configuration. Threaded down to
    /// [`scx_engine::build_and_write_conversion_predicate_indexes`] /
    /// the streaming builder. Defaults to "do nothing" via
    /// `index_auto_threshold = 0` (the existing sentinel that disables
    /// auto-detect and forced/preset columns).
    pub index_options: ConversionPredicateIndexOptions,

    /// When `true`, skip the var identity check (column names, types,
    /// row order) and trust callers' assertion that every input shares
    /// an identical var table. Equivalent to today's silent
    /// behaviour. Default `false` — Phase 2 strictness, which would
    /// otherwise silently corrupt the column axis when inputs disagree
    /// on gene order (see Issue 9 in MERGE-OBS-OFFSET-OVERFLOW.md).
    pub assume_identical_var: bool,

    /// Policy for combining `uns` (unstructured metadata) across
    /// inputs. Default [`UnsPolicy::First`] = today's behaviour.
    pub uns_policy: UnsPolicy,

    /// Override `shard_target_rows` for the output's obs metadata
    /// shards. `None` = derive from the first input's
    /// `header.shard_target_rows`. Each obs shard contains at most
    /// this many rows.
    pub shard_target_rows: Option<u32>,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            index_options: ConversionPredicateIndexOptions {
                index_obs: Vec::new(),
                index_var: Vec::new(),
                index_preset: None,
                index_auto_threshold: 0,
            },
            assume_identical_var: false,
            uns_policy: UnsPolicy::default(),
            shard_target_rows: None,
        }
    }
}

impl MergeOptions {
    /// Construct a `MergeOptions` that reproduces the legacy
    /// `merge_with_index_options(inputs, output, &index_options)` call
    /// — used by the back-compat wrapper so existing callers don't see
    /// a behaviour change.
    pub fn legacy_with_index_options(index_options: ConversionPredicateIndexOptions) -> Self {
        Self {
            index_options,
            assume_identical_var: false,
            uns_policy: UnsPolicy::First,
            shard_target_rows: None,
        }
    }
}

/// Policy for combining `uns` (unstructured) sections across merge
/// inputs. See Issue 9 in MERGE-OBS-OFFSET-OVERFLOW.md.
///
/// `uns` is typically a free-form JSON blob carrying preprocessing
/// parameters, model metadata, colour palettes, and source-specific
/// annotations. There is no universal "correct" merge strategy, so we
/// expose the choice as a CLI / pyscx flag rather than picking one
/// silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnsPolicy {
    /// Keep the first input's `uns` verbatim; drop the rest. Matches
    /// today's pre-refactor behaviour. Warnings are emitted via
    /// `log::warn!` when a dropped input had a non-equal `uns`.
    #[default]
    First,
    /// Error if any input's `uns` differs from the first input's
    /// `uns`. Strictest policy; useful for pipelines that pre-validate
    /// input consistency.
    RequireEqual,
    /// Write a JSON object that namespaces each input's `uns` under a
    /// per-input key (`input_0`, `input_1`, …). Lossless but
    /// non-canonical — downstream tools that consume specific `uns`
    /// keys (e.g. `colors`) need an extra unwrap step.
    Namespace,
    /// Keep the first input's `uns` as the canonical body, but record
    /// a `_scx_uns_conflicts` array listing per-key disagreements. A
    /// best-effort compromise between [`Self::First`] (silent data
    /// loss) and [`Self::Namespace`] (downstream churn).
    Summary,
}

impl UnsPolicy {
    /// Parse a CLI/pyscx string value into a policy. Accepts
    /// `"first"`, `"require-equal"` (also `"require_equal"`),
    /// `"namespace"`, `"summary"`. Case-insensitive.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "first" => Some(Self::First),
            "require-equal" | "require_equal" => Some(Self::RequireEqual),
            "namespace" => Some(Self::Namespace),
            "summary" => Some(Self::Summary),
            _ => None,
        }
    }

    /// Inverse of [`Self::parse`] — produces the canonical CLI spelling
    /// for provenance and diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::First => "first",
            Self::RequireEqual => "require-equal",
            Self::Namespace => "namespace",
            Self::Summary => "summary",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uns_policy_parse_round_trip() {
        for p in [
            UnsPolicy::First,
            UnsPolicy::RequireEqual,
            UnsPolicy::Namespace,
            UnsPolicy::Summary,
        ] {
            assert_eq!(UnsPolicy::parse(p.as_str()), Some(p));
        }
    }

    #[test]
    fn uns_policy_parse_underscored() {
        assert_eq!(
            UnsPolicy::parse("require_equal"),
            Some(UnsPolicy::RequireEqual)
        );
        assert_eq!(UnsPolicy::parse("FIRST"), Some(UnsPolicy::First));
        assert_eq!(UnsPolicy::parse("bogus"), None);
    }

    #[test]
    fn merge_options_default_matches_legacy() {
        let opts = MergeOptions::default();
        assert!(!opts.assume_identical_var);
        assert_eq!(opts.uns_policy, UnsPolicy::First);
        assert!(opts.shard_target_rows.is_none());
        assert_eq!(opts.index_options.index_auto_threshold, 0);
    }
}
