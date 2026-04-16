//! Evaluation metrics for perturbation prediction benchmarking.
//!
//! Implements metrics used by `cell-eval` and `arc-bench` for evaluating
//! single-cell perturbation predictions. These operate on pseudobulk means
//! (aggregated per-perturbation expression) and per-cell data.
//!
//! ## Modules
//!
//! - [`bulk_metrics`] — Per-perturbation metrics on pseudobulk means:
//!   pearson_delta, MSE, MAE, MSE_delta, MAE_delta.

pub mod bulk_metrics;

/// Distance metric for pairwise computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance.
    Euclidean,
    /// Manhattan (L1) distance.
    L1,
    /// Cosine distance (1 - cosine_similarity).
    Cosine,
}

pub use bulk_metrics::{compute_bulk_metrics, BulkMetric, BulkMetricsResult};
