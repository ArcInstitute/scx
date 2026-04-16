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
//! - [`discrimination`] — Discrimination score: per-perturbation ranking of
//!   predicted effects against real effects (L1/L2/cosine, gene exclusion).
//! - [`distances`] — Shared pairwise distance kernels (Euclidean, L1, cosine)
//!   with streaming mean computation (no N×N allocation).
//! - [`edistance`] — Energy distance metric: per-perturbation e-distances
//!   with Pearson correlation between real and predicted.

pub mod bulk_metrics;
pub mod discrimination;
pub mod distances;
pub mod edistance;

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
pub use discrimination::{compute_discrimination_score, DiscriminationResult};
pub use edistance::{compute_energy_distance, fused_edistance, EDistanceResult};
