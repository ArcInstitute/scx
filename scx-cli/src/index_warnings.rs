//! Shared rendering for predicate-index outcomes from the `scx merge`,
//! `scx append`, and `scx compact` subcommands.
//!
//! Mirrors `scx-convert/src/pipeline/index.rs::process_predicate_index_outcomes`
//! semantics in plain `eprintln!`s: forced-column errors are reported
//! by the rewrite op (it returns an error before producing a summary),
//! preset skips become per-column stderr warnings, and a multimodal
//! skip becomes a single stderr line.

use scx_engine::{BuildOutcome, SkipReason};
use scx_ops::PredicateIndexBuildSummary;

pub fn emit_index_summary(op_name: &str, summary: &PredicateIndexBuildSummary) {
    if let Some(columns) = summary.multimodal_skip.as_ref() {
        eprintln!(
            "warning: {op_name}: predicate index skipped — output is multimodal \
             but the engine read-side is unimodal-only (requested columns: \
             {columns:?})"
        );
        return;
    }
    let Some(result) = summary.result.as_ref() else {
        return;
    };
    emit_outcomes(op_name, "obs", &result.obs_outcomes);
    emit_outcomes(op_name, "var", &result.var_outcomes);
}

fn emit_outcomes(op_name: &str, axis: &str, outcomes: &[BuildOutcome]) {
    for outcome in outcomes {
        match outcome {
            // `ForcedColumnError` should have aborted the rewrite
            // before we got here, but render defensively in case a
            // future engine path demotes it.
            BuildOutcome::ForcedColumnError { column, reason } => {
                eprintln!(
                    "warning: {op_name}: forced {axis} index column '{column}' skipped: {reason}"
                );
            }
            BuildOutcome::PresetSkipped { column, reason } => match reason {
                SkipReason::MissingColumn => {
                    eprintln!("warning: {op_name}: preset {axis} index column '{column}' missing");
                }
                other => {
                    eprintln!(
                        "warning: {op_name}: preset {axis} index column '{column}' skipped: {other}"
                    );
                }
            },
        }
    }
}
