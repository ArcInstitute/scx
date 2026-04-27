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
//! - [`clustering`] — Clustering agreement metrics: AMI, NMI, ARI scoring
//!   for comparing cluster label assignments.
//! - [`discrimination`] — Discrimination score: per-perturbation ranking of
//!   predicted effects against real effects (L1/L2/cosine, gene exclusion).
//! - [`distances`] — Shared pairwise distance kernels (Euclidean, L1, cosine)
//!   with streaming mean computation (no N×N allocation).
//! - [`edistance`] — Energy distance metric: per-perturbation e-distances
//!   with Pearson correlation between real and predicted.
//! - [`knockdown`] — Per-cell knockdown efficiency and log fold change from
//!   arc-bench, with efficient CSR single-column extraction.

pub mod bulk_metrics;
pub mod clustering;
pub mod discrimination;
pub mod distances;
pub mod edistance;
pub mod knockdown;

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
pub use clustering::{
    adjusted_mutual_info, adjusted_rand_index, adjusted_rand_index_rescaled,
    normalized_mutual_info, ClusteringMetric,
};
pub use discrimination::{compute_discrimination_score, DiscriminationResult};
pub use distances::{DistanceBackend, PairwiseFloat};
pub use edistance::{compute_energy_distance, fused_edistance, EDistanceResult};
pub use knockdown::{
    compute_control_baseline, compute_knockdown_efficiency, compute_log_deviation,
};
