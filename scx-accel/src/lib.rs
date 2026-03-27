//! Rust-native analysis accelerators for SCX files.
//!
//! Provides high-performance implementations of common single-cell analysis
//! operations (PCA, kNN, UMAP, DE) that stream data shard-by-shard from SCX's
//! backed mode, avoiding full matrix materialization.
//!
//! ## GPU Acceleration
//!
//! When built with the `gpu` feature, GPU-accelerated variants are available:
//! - [`randomized_pca_gpu`] — GPU PCA via cuSPARSE SpMM + cuSOLVER QR

pub mod diffexp;
pub mod error;
pub mod neighbors;
pub mod pca;
pub mod pseudobulk;
pub mod umap;

pub use diffexp::{
    merge_diff_exp_results, wilcoxon_rank_sum, wilcoxon_rank_sum_sparse,
    wilcoxon_rank_sum_streaming, DiffExpResult,
};
pub use error::{AccelError, Result};
pub use neighbors::{build_knn_graph, KnnResult};
pub use pca::{randomized_pca, randomized_pca_inmemory, PcaResult};
pub use pseudobulk::{
    pseudobulk_aggregate, pseudobulk_aggregate_inmemory, AggregationMethod, PseudobulkResult,
};
pub use umap::{compute_umap, UmapResult};

// GPU-accelerated variants (behind "gpu" feature)
#[cfg(feature = "gpu")]
pub use neighbors::{build_knn_graph_gpu, cuvs_available};
#[cfg(feature = "gpu")]
pub use pca::{gpu_available, randomized_pca_gpu};
