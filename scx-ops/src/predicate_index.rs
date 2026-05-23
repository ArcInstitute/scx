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
//! `scx-ops` returns the raw outcomes from
//! [`scx_engine::build_and_write_conversion_predicate_indexes`] plus a
//! `multimodal_skip` slot so `scx-cli` / `pyscx` can emit
//! `ConvertWarning::PredicateIndexSkippedMultimodal` /
//! `ConvertWarning::MissingPresetIndexColumn` themselves without
//! `scx-ops` taking a dep on `scx-convert`.

use scx_engine::ConversionPredicateIndexResult;

/// Outcome of attempting to build predicate indexes during a `merge`,
/// `append`, or `compact` rewrite. The caller decides how to surface
/// the per-axis outcomes to its user (typed `ConvertWarning`s, Python
/// `warnings.warn(...)`, CLI stderr, etc.).
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
    /// append must skip the write and surface this list as a single
    /// `PredicateIndexSkippedMultimodal { columns }` warning.
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
    /// `PredicateIndexSkippedMultimodal { columns }` warning in that
    /// case.
    pub fn was_multimodal_skip(&self) -> bool {
        self.multimodal_skip.is_some()
    }
}

/// True when the caller passed at least one index column / preset —
/// i.e. they expect a predicate-index to land on the output.
pub fn index_requested(options: &scx_engine::ConversionPredicateIndexOptions) -> bool {
    !options.index_obs.is_empty() || !options.index_var.is_empty() || options.index_preset.is_some()
}

/// Flatten the requested columns / preset into a single `Vec<String>`
/// for the multimodal-skip warning. Mirrors
/// `scx_convert::mudata_pipeline::emit_multimodal_index_skip_warning`
/// so the user sees the same payload regardless of the rewrite op that
/// triggered the skip.
pub fn requested_columns(options: &scx_engine::ConversionPredicateIndexOptions) -> Vec<String> {
    let mut columns: Vec<String> = Vec::new();
    columns.extend(options.index_obs.iter().cloned());
    columns.extend(options.index_var.iter().cloned());
    if let Some(name) = options.index_preset.as_deref() {
        columns.push(format!("preset:{name}"));
    }
    columns
}
