//! CLI-facing rendering and conversion front-end defaults.
//!
//! Nothing here reads or writes a predicate index. It lives in `scx-engine`
//! for a dependency reason recorded on [`forced_column_missing_message`]:
//! `pyscx` depends on `scx-engine` unconditionally but on `scx-convert` only
//! under the `hdf5` feature, so a helper shared by the CLI and the Python
//! conversion entry points cannot live in `scx-convert`. Moving it *within*
//! this crate is fine; moving it out is not.

use std::borrow::Cow;

/// Render an actionable error message for a forced obs/var index
/// column that doesn't exist in the source DataFrame. Adds the
/// available column list and, when one is close enough, a single
/// `Did you mean '<col>'?` suggestion (Levenshtein-normalised
/// threshold ≥ 0.6). Shared by `scx-convert/src/pipeline/index.rs` (CLI
/// path) and `pyscx/src/convert/from_anndata.rs` (Python path) so both surfaces
/// emit the same message. — E2-2026-05-20.
///
/// Lives in `scx-engine` rather than `scx-convert` because pyscx
/// depends on `scx-engine` unconditionally but only pulls in
/// `scx-convert` under the `hdf5` feature; the helper must remain
/// callable from `pyscx::convert::build_and_write_predicate_indexes_inline`
/// (which is reachable from CPU-only `pyscx.from_anndata` paths).
pub fn forced_column_missing_message(axis: &str, column: &str, available: &[String]) -> String {
    format!(
        "forced {axis} index column '{column}': missing column. {}",
        column_suggestion_suffix(axis, column, available)
    )
}

/// Build the `"column not found. Available {axis} columns: [...]. Did you mean
/// '{...}'?"` reason for `EngineError::SchemaError` (the runtime predicate
/// path: `pyscx.open(...).query().filter_obs("totl_counts >= 500")`). The
/// suffix structure mirrors [`forced_column_missing_message`] so users see
/// the same "Available / Did you mean" treatment regardless of whether the
/// missing-column error originates from convert-time or query-time. —
/// F5-2026-05-20-Tier2.
pub fn column_not_found_message(axis: &str, column: &str, available: &[String]) -> String {
    format!(
        "column not found. {}",
        column_suggestion_suffix(axis, column, available)
    )
}

/// Find the closest match for `column` in `available` using normalized
/// Levenshtein distance with a 0.6 acceptance threshold. Returns `None`
/// when no candidate clears the threshold (i.e. the user's input isn't
/// a near-typo of any existing column).
fn best_match(column: &str, available: &[String]) -> Option<String> {
    available
        .iter()
        .map(|n| (n, strsim::normalized_levenshtein(column, n)))
        .filter(|(_, s)| *s >= 0.6)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(n, _)| n.clone())
}

/// Render the `"Available {axis} columns: [...]."` footer.
///
/// Lists every column when `available.len() <= 64`; the 64-column cap
/// keeps the worst-case message bounded for atlases with hundreds of
/// obs columns. Falls back to the first-8 preview with `", \u{2026}"`
/// (Unicode ellipsis — avoids the `, ....` 4-dot artifact) when there
/// are more than 64 columns.
///
/// N2-2026-05-21-Tier2: gating is purely on `available.len()` now.
/// PR #116's E1 fix originally gated show-all on "strsim has a winner",
/// but scanpy-vocab inputs (`total_counts`, `pct_counts_mt`, …) have
/// no near-match in the Census obs schema, so the user couldn't see
/// `raw_sum` in the truncated preview. Length is the only meaningful
/// concern; the strsim suggestion is rendered separately.
///
/// Empty `available` yields the empty-axis fallback that hints at no
/// obs/var metadata being present in the file.
fn render_available_columns(axis: &str, available: &[String]) -> String {
    if available.is_empty() {
        return format!("Available {axis} columns: [] (this h5ad has no {axis} metadata).");
    }
    if available.len() <= 64 {
        let items: Vec<&str> = available.iter().map(String::as_str).collect();
        return format!("Available {axis} columns: {items:?}.");
    }
    let preview_n = 8;
    let preview: Vec<&str> = available[..preview_n].iter().map(String::as_str).collect();
    format!("Available {axis} columns: {preview:?}, \u{2026}.")
}

/// Shared suffix builder for [`forced_column_missing_message`] and
/// [`column_not_found_message`]. Renders the available-columns footer
/// (capped at 64) and appends a strsim "Did you mean ...?" hint when
/// there's a near-match.
pub(crate) fn column_suggestion_suffix(axis: &str, column: &str, available: &[String]) -> String {
    let mut msg = render_available_columns(axis, available);
    if let Some(s) = best_match(column, available) {
        msg.push_str(&format!(" Did you mean '{s}'?"));
    }
    msg
}

/// Aggregate-form of [`forced_column_missing_message`] for callers that
/// processed multiple `BuildOutcome::ForcedColumnError` entries with
/// `SkipReason::MissingColumn`. Used by `scx convert --index-obs/--index-var`
/// (and the parallel `pyscx.from_anndata` path) so the user sees ALL
/// typos in a single error rather than fixing them one per run.
///
/// `missing` is the list of forced columns that came back missing, in
/// the order they were requested. Each entry gets its own line with a
/// strsim suggestion (if any); the available-columns preview is shared
/// across them via the same `column_suggestion_suffix` helper so the
/// rendered format stays consistent with the single-column message.
///
/// For a single missing column, prefer [`forced_column_missing_message`]
/// — the singular form keeps the existing single-line wording.
pub fn forced_columns_missing_message(
    axis: &str,
    missing: &[String],
    available: &[String],
) -> String {
    if missing.is_empty() {
        // Degenerate guard. Should not happen in practice — the caller
        // is supposed to gate on `!missing.is_empty()` before calling.
        return format!("0 forced {axis} index columns are missing.");
    }
    if missing.len() == 1 {
        return forced_column_missing_message(axis, &missing[0], available);
    }
    let mut msg = format!(
        "{n} forced {axis} index columns are missing:",
        n = missing.len()
    );
    for column in missing {
        match best_match(column, available) {
            Some(s) => msg.push_str(&format!("\n  - '{column}': did you mean '{s}'?")),
            None => msg.push_str(&format!("\n  - '{column}'")),
        }
    }
    msg.push('\n');
    msg.push_str(&render_available_columns(axis, available));
    msg
}

/// Named column preset (e.g. `cellxgene`, `perturbseq`, `training`).
#[derive(Debug, Clone, PartialEq)]
pub struct IndexPreset {
    pub obs_columns: Vec<&'static str>,
    pub var_columns: Vec<&'static str>,
}

/// Resolve a named preset to its column list. Returns `None` for
/// unknown names; the conversion pipeline surfaces that as a clean
/// `ConvertError`.
pub fn index_preset_columns(name: &str) -> Option<IndexPreset> {
    match name {
        "cellxgene" => Some(IndexPreset {
            obs_columns: vec![
                "cell_type",
                "cell_type_ontology_term_id",
                "tissue",
                "tissue_ontology_term_id",
                "disease",
                "assay",
                "donor_id",
                "development_stage",
                "sex",
                "suspension_type",
            ],
            var_columns: vec!["feature_name", "feature_type"],
        }),
        "perturbseq" => Some(IndexPreset {
            obs_columns: vec![
                "cell_type",
                "donor",
                "batch",
                "condition",
                "perturbation",
                "guide_id",
                "target_gene",
                "control",
                "split",
            ],
            var_columns: vec!["feature_name", "feature_type"],
        }),
        "training" => Some(IndexPreset {
            obs_columns: vec![
                "cell_type",
                "donor",
                "batch",
                "dataset_id",
                "split",
                "organism",
                "tissue",
            ],
            var_columns: vec!["feature_name", "feature_type"],
        }),
        _ => None,
    }
}

/// Whether a named index preset implies a CSC sidecar should be built
/// when the caller did not explicitly pass a `csc` policy.
///
/// The accel-ready presets — `training` and `perturbseq` — drive
/// column/DE-heavy workloads (pseudobulk, `pdex_ref`, `rank_genes_groups`)
/// whose primary substrate is the column-major CSC sidecar, so selecting
/// one of those presets upgrades an *unset* `csc` to `auto`. `cellxgene`
/// is query/browse-oriented and does not imply CSC. An explicit `--csc`
/// value (including `off`) always wins over this default.
pub fn preset_implies_csc_auto(name: &str) -> bool {
    matches!(name, "training" | "perturbseq")
}

/// Resolve the effective CSC policy string for a conversion entry point.
///
/// An explicit `csc` always wins; when unset (`None`), an accel-ready
/// `index_preset` (`training` / `perturbseq`) upgrades the default to
/// `"auto"`, otherwise the default is `"off"`. Shared by the `scx convert`
/// CLI and the pyscx conversion entry points so the two front-ends cannot
/// drift.
pub fn resolve_csc_policy<'a>(csc: Option<&'a str>, index_preset: Option<&str>) -> Cow<'a, str> {
    match csc {
        Some(v) => Cow::Borrowed(v),
        None if index_preset.is_some_and(preset_implies_csc_auto) => Cow::Borrowed("auto"),
        None => Cow::Borrowed("off"),
    }
}
