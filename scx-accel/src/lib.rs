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

pub mod csc;
pub mod diffexp;
pub mod error;
pub mod eval_metrics;
pub(crate) mod finite;
#[cfg(feature = "gpu")]
pub mod fused;
pub mod gene_score;
pub mod harmony;
pub mod hvg;
pub mod leiden;
pub mod lisi;
pub mod mem_budget;
pub mod nb_glm;
pub mod neighbors;
pub mod pca;
pub mod pflog;
pub mod projected_source;
pub mod pseudobulk;
pub mod route;
pub mod umap;

pub use csc::{
    pdex_ref_streaming_csc, pseudobulk_aggregate_csc, require_csc, streaming_clip_square_sum_csc,
    streaming_mean_var_csc, wilcoxon_rank_sum_streaming_csc, PreferFormat,
};
#[cfg(feature = "gpu")]
pub use csc::{streaming_clip_square_sum_csc_with_device, streaming_mean_var_csc_with_device};
pub use diffexp::{
    finalize_pdex, merge_diff_exp_results, pdex_ref, pdex_ref_core, pdex_ref_sparse,
    pdex_ref_streaming, wilcoxon_rank_sum, wilcoxon_rank_sum_sparse, wilcoxon_rank_sum_streaming,
    DiffExpResult, PdexRefResult,
};
pub use error::{AccelError, Result};
#[cfg(feature = "gpu")]
pub use eval_metrics::edistance::compute_energy_distance_gpu;
pub use eval_metrics::{
    bulk_metrics::{compute_bulk_metrics, pearson_correlation, BulkMetric, BulkMetricsResult},
    clustering::{
        adjusted_mutual_info, adjusted_rand_index, adjusted_rand_index_rescaled,
        normalized_mutual_info, ClusteringMetric,
    },
    discrimination::{compute_discrimination_score, DiscriminationResult},
    distances::{DistanceBackend, PairwiseFloat},
    edistance::{compute_energy_distance, fused_edistance, EDistanceResult},
    knockdown::{compute_control_baseline, compute_knockdown_efficiency, compute_log_deviation},
    DistanceMetric,
};
pub use gene_score::{score_genes, ScoreMethod};
pub use harmony::{harmony_integrate, BatchCovariate, HarmonyConfig, HarmonyResult};
pub use hvg::{
    streaming_clip_square_sum, streaming_clip_square_sum_batched, streaming_mean_var,
    streaming_mean_var_batched, streaming_mean_var_expm1, BatchedHvgStats, HvgStats,
};
pub use leiden::{leiden, LeidenConfig, LeidenResult};
pub use lisi::{compute_lisi, LisiConfig, LisiResult};
pub use nb_glm::{
    pseudobulk_nb_glm, DispersionMethod, DispersionTrend, NbGlmContrast, NbGlmDiagnostics,
    NbGlmOptions, NbGlmResult,
};
pub use neighbors::{build_knn_graph, KnnResult};
pub use pca::{
    covariance_pca, covariance_pca_inmemory, pflog_pca, randomized_pca, randomized_pca_inmemory,
    PcaResult, COVARIANCE_PCA_THRESHOLD,
};
pub use pflog::{
    estimate_alpha, pflog_baseline_from_delta, pflog_baseline_from_raw, AlphaEstimate,
    AlphaOptions, PFlog,
};
pub use projected_source::ProjectedShardSource;
pub use pseudobulk::{
    build_group_mapping, pseudobulk_aggregate, pseudobulk_aggregate_dense,
    pseudobulk_aggregate_from_slices, pseudobulk_aggregate_inmemory, AggregationMethod,
    GeomMeanMode, PseudobulkResult,
};
#[cfg(feature = "gpu")]
pub use pseudobulk::{
    pseudobulk_means_gpu_backed, pseudobulk_means_gpu_dense, pseudobulk_means_gpu_from_slices,
};
#[cfg(feature = "gpu")]
pub use route::plan_de_route_from_source;
pub use route::{
    plan_de_route, plan_hvg_route, plan_nb_glm_route, plan_simple_gpu_route, AccelExecutionInfo,
    AccelRoute, DeviceRequest, FallbackReason, InputLayout,
};
pub use umap::{compute_umap, UmapResult};

// GPU-accelerated variants (behind "gpu" feature)
#[cfg(feature = "gpu")]
pub use diffexp::{
    pdex_ref_gpu, pdex_ref_gpu_dense, wilcoxon_rank_sum_gpu, wilcoxon_rank_sum_gpu_dense,
    GpuDeShardInput,
};
#[cfg(feature = "gpu")]
pub use fused::pca_then_knn_gpu;
#[cfg(feature = "gpu")]
pub use harmony::harmony_integrate_gpu;
#[cfg(feature = "gpu")]
pub use hvg::{
    streaming_clip_square_sum_batched_with_device, streaming_clip_square_sum_with_device,
    streaming_mean_var_batched_with_device, streaming_mean_var_with_device,
};
#[cfg(feature = "gpu")]
pub use nb_glm::{finalize_nb_glm, gpu_nb_glm_fit_states, gpu_pseudobulk_nb_glm, NbGlmFitData};
#[cfg(feature = "gpu")]
pub use neighbors::cuvs_available;
#[cfg(feature = "gpu")]
pub use pca::{gpu_available, gpu_info, randomized_pca_gpu, GpuInfo};
#[cfg(feature = "gpu")]
pub use scx_gpu::nvcomp::nvcomp_enabled;
#[cfg(feature = "gpu")]
pub use scx_gpu::profile;
#[cfg(feature = "gpu")]
pub use scx_gpu::{
    cusparse_modern_abi_available, gpu_log1p, gpu_preprocess_to_csr, GpuDevice, GpuMathMode,
    GpuPcaTuning, ProfileSnapshot, QrMethod, SpmmAlgPolicy, StageStat,
};
#[cfg(feature = "gpu")]
pub use scx_gpu::{
    decode_csr_shards_to_device, decode_csr_shards_to_device_with_stats, decode_shard_gpu,
    decode_shard_gpu_with_stats, DeviceDecodeStats, GpuCsr, GpuCsrPointers,
};
#[cfg(feature = "gpu")]
pub use scx_gpu::{GPU_NB_GLM_NSUB_MAX, GPU_NB_GLM_PMAX};
