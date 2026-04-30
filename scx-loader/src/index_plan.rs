//! Plan-driven paired-batch reads from an `.scx` file.
//!
//! Sibling to the sequential `TrainingPipeline`. The consumer supplies a stream
//! of `Vec<(u64, u64)>` plans (perturbed, control row index pairs); this module
//! gathers the rows via `BackedCsrReader::read_row_indices`, projects + normalizes
//! them through the existing `HvgProjection` / `fused_normalize_log1p_dense`
//! primitives, and yields paired dense `IndexPlanBatch` values.
//!
//! See `PER-CELL-CONTROL-PAIRING.md` at the workspace root for the full design.
//!
//! Phase 0 — scaffolding only. The types defined here are stubs; the
//! implementation lands in subsequent phases.

use std::collections::HashMap;

use crate::batch::ObsColumn;

/// One paired batch produced by `IndexPlanLoader`.
///
/// `x` and `x_paired` are row-major dense `[B * n_output_cols]` buffers in the
/// post-sort plan order; `pairs[i]` is the `(pert_idx, ctrl_idx)` whose
/// expression occupies row `i` of both `x` and `x_paired`.
pub struct IndexPlanBatch {
    /// Perturbed-side dense expression `[n_pairs * n_output_cols]`.
    pub x: Vec<f32>,
    /// Control-side dense expression `[n_pairs * n_output_cols]`.
    pub x_paired: Vec<f32>,
    /// `(pert_idx, ctrl_idx)` pairs, in the order rows appear in `x` / `x_paired`.
    pub pairs: Vec<(u64, u64)>,
    /// Obs columns gathered for the perturbed side.
    pub obs: HashMap<String, ObsColumn>,
    /// Obs columns gathered for the control side.
    pub obs_paired: HashMap<String, ObsColumn>,
}
