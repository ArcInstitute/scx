//! Rust-native analysis accelerators for SCX files.
//!
//! Provides high-performance implementations of common single-cell analysis
//! operations (PCA, kNN, UMAP, DE) that stream data shard-by-shard from SCX's
//! backed mode, avoiding full matrix materialization.

pub mod diffexp;
pub mod error;
pub mod neighbors;
pub mod pca;
pub mod pseudobulk;
pub mod umap;

pub use diffexp::{
    merge_diff_exp_results, wilcoxon_rank_sum, wilcoxon_rank_sum_streaming, DiffExpResult,
};
pub use error::{AccelError, Result};
pub use neighbors::{build_knn_graph, KnnResult};
pub use pca::{randomized_pca, randomized_pca_inmemory, PcaResult};
pub use pseudobulk::{
    pseudobulk_aggregate, pseudobulk_aggregate_inmemory, AggregationMethod, PseudobulkResult,
};
pub use umap::{compute_umap, UmapResult};
