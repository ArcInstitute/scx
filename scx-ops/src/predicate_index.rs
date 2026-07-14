//! Shared scaffolding for wiring predicate-index sections into the
//! `merge`, `append`, and `compact` rewrites.
//!
//! These ops historically dropped `ObsPredicateIndex` / `VarPredicateIndex`
//! sections from their output, so query-time pushdown silently regressed
//! to full obs scans after every rewrite. The new `*_with_index_options`
//! entry points accept the same `ConversionPredicateIndexOptions` that
//! `scx convert` / `pyscx.from_anndata` already expose and produce a
//! [`PredicateIndexBuildSummary`] the caller can map onto its preferred
//! warning channel.
//!
//! Cross-crate warning routing is intentionally left to the caller:
//! `scx-ops` has no warning type of its own. It returns the raw outcomes
//! from [`scx_engine::build_and_write_conversion_predicate_indexes`] plus a
//! `multimodal_skip` slot on [`PredicateIndexBuildSummary`], and each caller
//! maps those onto its own channel — e.g. `scx-cli` / `pyscx` translate them
//! into `scx-convert`'s `ConvertWarning` variants. This keeps `scx-ops` free
//! of any dep on `scx-convert`.

use arrow::datatypes::Schema;
use scx_engine::{ConversionPredicateIndexOptions, ConversionPredicateIndexResult};

use crate::error::{OpsError, Result};

/// Outcome of attempting to build predicate indexes during a `merge`,
/// `append`, or `compact` rewrite. The caller decides how to surface
/// the per-axis outcomes to its user — e.g. `scx-convert`'s typed
/// `ConvertWarning`s, Python `warnings.warn(...)`, or CLI stderr.
#[derive(Debug, Default)]
pub struct PredicateIndexBuildSummary {
    /// Populated when the op actually invoked the engine builder. The
    /// `obs_outcomes` / `var_outcomes` fields carry per-column results
    /// that the caller maps to user-facing warnings (mirroring
    /// `scx-convert::pipeline::process_predicate_index_outcomes`).
    pub result: Option<ConversionPredicateIndexResult>,
    /// Populated when the op was multimodal and the caller requested an
    /// index. Predicate-index sections are unimodal-only today (the
    /// `scx-format::writer` emitters take no `modality_id` and the
    /// `scx-engine` read-side ignores it). Multimodal merge / compact /
    /// append skip the write and hand back this column list so the caller
    /// can surface it as a single warning (e.g. `scx-convert`'s
    /// `PredicateIndexSkippedMultimodal { columns }`).
    pub multimodal_skip: Option<Vec<String>>,
}

impl PredicateIndexBuildSummary {
    /// Empty summary — the rewrite path was invoked without any
    /// `--index-*` kwarg and produced no predicate-index sections.
    pub fn skipped() -> Self {
        Self::default()
    }

    /// True when the request was non-trivial but no index was written
    /// because the output is multimodal. Callers should emit a single
    /// warning naming the skipped columns in that case (e.g.
    /// `scx-convert`'s `PredicateIndexSkippedMultimodal { columns }`).
    pub fn was_multimodal_skip(&self) -> bool {
        self.multimodal_skip.is_some()
    }
}

/// True when the user explicitly requested any predicate-index work —
/// a forced obs / var column, a preset, OR a non-zero auto-threshold.
/// The legacy `merge` / `compact` / `append` / `append_from_reader`
/// wrappers delegate with `index_auto_threshold = 0` as a sentinel that
/// disables both auto-detect and any multimodal-skip warning (preserves
/// pre-fix behaviour where the legacy entry points emit no predicate
/// index and no warning).
///
/// The single-modality rewrite path calls the engine unconditionally;
/// this helper only gates the multimodal-skip surface so callers that
/// passed the legacy `Default::default()` (auto_threshold = 1000) but
/// hit a multimodal target don't silently see a skip warning they
/// didn't ask for.
pub fn user_wants_index(options: &ConversionPredicateIndexOptions) -> bool {
    !options.index_obs.is_empty()
        || !options.index_var.is_empty()
        || options.index_preset.is_some()
        || options.index_auto_threshold > 0
}

/// Flatten the requested columns / preset into a single `Vec<String>`
/// for the multimodal-skip warning. Mirrors
/// `scx_convert::h5mu::pipeline::emit_multimodal_index_skip_warning`
/// so the user sees the same payload regardless of the rewrite op that
/// triggered the skip.
pub fn requested_columns(options: &ConversionPredicateIndexOptions) -> Vec<String> {
    let mut columns: Vec<String> = Vec::new();
    columns.extend(options.index_obs.iter().cloned());
    columns.extend(options.index_var.iter().cloned());
    if let Some(name) = options.index_preset.as_deref() {
        columns.push(format!("preset:{name}"));
    }
    columns
}

/// Validate that every `index_obs` / `index_var` column the caller forced
/// is present in the corresponding output schema. Called by each
/// `*_with_index_options` rewrite op BEFORE any committing I/O so that
/// missing-forced-column errors fail loudly without leaving a half-built
/// output on disk.
///
/// Engine outcome flow (after writes) still emits
/// `BuildOutcome::ForcedColumnError` in defensive paths — the rewrite ops
/// rely on this upfront check to make that path unreachable for
/// rewrites. See `pyscx::convert::build_and_write_predicate_indexes_inline`
/// and `scx-convert::pipeline::process_predicate_index_outcomes` for the
/// equivalent fail-late paths that this duplicates as fail-fast.
pub fn validate_forced_columns(
    options: &ConversionPredicateIndexOptions,
    obs_schema: &Schema,
    var_schema: &Schema,
) -> Result<()> {
    let obs_missing: Vec<String> = options
        .index_obs
        .iter()
        .filter(|c| obs_schema.column_with_name(c).is_none())
        .cloned()
        .collect();
    let var_missing: Vec<String> = options
        .index_var
        .iter()
        .filter(|c| var_schema.column_with_name(c).is_none())
        .cloned()
        .collect();
    if obs_missing.is_empty() && var_missing.is_empty() {
        return Ok(());
    }
    let mut parts: Vec<String> = Vec::new();
    if !obs_missing.is_empty() {
        let available: Vec<String> = obs_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        parts.push(scx_engine::index::forced_columns_missing_message(
            "obs",
            &obs_missing,
            &available,
        ));
    }
    if !var_missing.is_empty() {
        let available: Vec<String> = var_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        parts.push(scx_engine::index::forced_columns_missing_message(
            "var",
            &var_missing,
            &available,
        ));
    }
    Err(OpsError::InvalidInput(parts.join("\n")))
}
