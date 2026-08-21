//! Convert-time predicate-index construction.
//!
//! `scx-engine` owns the build; this module owns the decision of *when* to run
//! it, and the translation of its per-column outcomes into `ConvertWarning`s.

use scx_format_io::writer::ScxWriter;

use super::error::ConvertError;
use crate::options::IngestOptions;
use crate::warnings::{ConvertWarning, WarningSink};
use arrow::record_batch::RecordBatch;
use scx_engine::{
    build_and_write_conversion_predicate_indexes, BuildOutcome, ConversionPredicateIndexOptions,
    SkipReason,
};

/// Build and write obs/var predicate indexes from the
/// currently configured conversion options, then return the list of
/// columns that ended up indexed so the caller can stamp provenance.
///
/// Three sources of column names are combined:
///   1. `opts.index_obs` / `opts.index_var` (forced; missing/unsupported
///      columns produce a hard `ConvertError`),
///   2. `opts.index_preset` (skipped + warned via the sink on
///      missing/unsupported columns),
///   3. auto-detection on cardinality `< opts.index_auto_threshold` when
///      neither forced nor preset columns are supplied.
///
/// `csr_row_ranges` must reflect the actual on-disk shard boundaries
/// produced by the writer (the engine uses local row indices within
/// each shard, so any drift between assumed and actual ranges produces
/// silently wrong pruning).
///
/// All of the build orchestration (preset resolution, encoding, writer
/// calls) lives in
/// [`scx_engine::build_and_write_conversion_predicate_indexes`]; this
/// wrapper only maps the engine's typed outcomes into `ConvertError` /
/// `ConvertWarning::{MissingPresetIndexColumn, UnsupportedIndexColumn}`.
/// `pyscx::convert::build_and_write_predicate_indexes_inline` is the
/// Python-side mirror — keep their outcome handling shapes in sync.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_and_write_predicate_indexes(
    writer: &mut ScxWriter,
    obs: &RecordBatch,
    var: &RecordBatch,
    csr_row_ranges: &[(u64, u64)],
    n_vars: usize,
    opts: &IngestOptions,
    // Sort-on-convert: extra obs columns to force-index (the sort key) so its
    // now-contiguous `shard_ranges` are emitted. Merged into `index_obs`,
    // deduped, order preserved.
    extra_index_obs: &[String],
    sink: &mut WarningSink,
) -> Result<(Vec<String>, Vec<String>), ConvertError> {
    let mut index_obs = opts.index_obs.clone();
    for col in extra_index_obs {
        if !index_obs.iter().any(|c| c == col) {
            index_obs.push(col.clone());
        }
    }
    let engine_opts = ConversionPredicateIndexOptions {
        index_obs,
        index_var: opts.index_var.clone(),
        index_preset: opts.index_preset.clone(),
        index_auto_threshold: opts.index_auto_threshold,
    };
    let result = build_and_write_conversion_predicate_indexes(
        writer,
        obs,
        var,
        csr_row_ranges,
        n_vars,
        &engine_opts,
    )
    .map_err(|e| ConvertError::Other(format!("build predicate index: {e}")))?;

    // Preset name + per-axis expected count flow into the outcome
    // processor so it can batch a fully-missing preset into a single
    // actionable warning. Unknown preset names resolve
    // to (0, 0); engine surfaces those as `ConvertError`.
    let (preset_obs_expected, preset_var_expected) = opts
        .index_preset
        .as_deref()
        .and_then(scx_engine::index::index_preset_columns)
        .map(|p| (p.obs_columns.len(), p.var_columns.len()))
        .unwrap_or((0, 0));
    // `__index_level_0__` (pyarrow's canonical
    // name for an unnamed pandas index) survives into the arrow schema
    // for round-trip purposes but should never be suggested as a
    // user-facing column name in a "did you mean" / "available columns"
    // preview. Drop any `__`-prefixed entries.
    let obs_available: Vec<String> = obs
        .schema()
        .fields()
        .iter()
        .map(|f| f.name())
        .filter(|n| !n.starts_with("__"))
        .cloned()
        .collect();
    let var_available: Vec<String> = var
        .schema()
        .fields()
        .iter()
        .map(|f| f.name())
        .filter(|n| !n.starts_with("__"))
        .cloned()
        .collect();
    process_predicate_index_outcomes(
        result.obs_outcomes,
        "obs",
        opts.index_preset.as_deref(),
        preset_obs_expected,
        &obs_available,
        sink,
    )?;
    process_predicate_index_outcomes(
        result.var_outcomes,
        "var",
        opts.index_preset.as_deref(),
        preset_var_expected,
        &var_available,
        sink,
    )?;

    Ok((result.obs_indexed_columns, result.var_indexed_columns))
}

/// Demote per-column outcomes from
/// `scx_engine::build_and_write_conversion_predicate_indexes` into the
/// convert layer's policy: forced errors abort the convert; preset
/// skips emit a typed warning whose variant is chosen by the
/// `SkipReason` discriminant.
///
/// When `preset` is `Some` and EVERY preset column on this axis came
/// back `SkipReason::MissingColumn` (preset/file format mismatch),
/// collapse the burst into a single `PresetNoColumnsMatched` warning
/// pointing the user at the fix instead of emitting one
/// `MissingPresetIndexColumn` per column. Partial mismatch keeps the
/// per-column shape — that's a real schema drift worth surfacing.
pub(crate) fn process_predicate_index_outcomes(
    outcomes: Vec<BuildOutcome>,
    axis: &str,
    preset: Option<&str>,
    preset_expected: usize,
    available_columns: &[String],
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let mut missing: Vec<String> = Vec::new();
    let mut deferred: Vec<ConvertWarning> = Vec::new();
    // Forced-column errors used to be fail-fast on
    // the first miss, so users had to iterate one typo per run. Collect
    // them all and surface as a single aggregated error after the loop,
    // matching the `missing` / `deferred` pattern below for PresetSkipped.
    let mut forced_missing: Vec<String> = Vec::new();
    // Non-missing forced errors (unsupported dtype, high cardinality)
    // can't reuse the strsim renderer — the column DOES exist; the
    // engine's text already describes the real reason. Preserve fail-
    // fast on those: they're per-column type/cardinality problems that
    // aren't related to typo'd column names.
    for outcome in outcomes {
        match outcome {
            BuildOutcome::ForcedColumnError { column, reason } => {
                if matches!(reason, SkipReason::MissingColumn) {
                    forced_missing.push(column);
                } else {
                    return Err(ConvertError::Other(format!(
                        "forced {axis} index column '{column}': {reason}"
                    )));
                }
            }
            BuildOutcome::PresetSkipped { column, reason } => match reason {
                SkipReason::MissingColumn => missing.push(column),
                other => deferred.push(ConvertWarning::UnsupportedIndexColumn {
                    column,
                    reason: other.to_string(),
                }),
            },
        }
    }

    if !forced_missing.is_empty() {
        let msg = scx_engine::index::forced_columns_missing_message(
            axis,
            &forced_missing,
            available_columns,
        );
        return Err(ConvertError::Other(msg));
    }

    let aggregate = preset.is_some_and(|_| {
        !missing.is_empty() && preset_expected > 0 && missing.len() == preset_expected
    });

    if aggregate {
        sink.emit(ConvertWarning::PresetNoColumnsMatched {
            preset: preset.expect("aggregate => preset.is_some()").to_string(),
            axis: axis.to_string(),
            missing,
        });
    } else {
        for column in missing {
            sink.emit(ConvertWarning::MissingPresetIndexColumn { column });
        }
    }
    for w in deferred {
        sink.emit(w);
    }
    Ok(())
}
