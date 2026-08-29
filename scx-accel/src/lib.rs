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

// --- Test-only macro (must precede module declarations for textual scoping) ---

/// Acquire a GPU device for a test, or return early if CUDA is unavailable.
///
/// The twin of `scx-gpu`'s `require_gpu!()`, sharing its decision function so
/// the two crates cannot drift; only the `return` has to be a macro.
///
/// **Pair it with `#[ignore = "requires a CUDA GPU"]`.** Without the attribute
/// the test is selected on a CPU-only host, returns immediately and counts as
/// passed. `scx-gpu/tests/gpu_test_gating.rs` enforces the pairing across both
/// crates, and also rejects a bare `GpuDevice::new(0)` in test code — 11 of the
/// GPU tests here used to gate inside a helper function, where nothing that
/// inspects test bodies could see them.
#[cfg(all(test, feature = "gpu"))]
macro_rules! require_gpu_or_skip {
    () => {
        match scx_gpu::test_gate::device_or_skip(module_path!()) {
            Some(_) => {}
            None => return,
        }
    };
}

/// Gate a test on an optional CUDA library rather than on the device itself.
///
/// The twin of `scx-gpu`'s `require_gpu_cap!`. Use *below* `require_gpu_or_skip!()`,
/// never instead of it: a missing device on a GPU node is a broken job, a missing
/// optional library is a deployment state worth reporting.
#[cfg(all(test, feature = "gpu"))]
macro_rules! require_gpu_cap {
    (cuvs) => {
        if !scx_gpu::test_gate::capability_or_skip(
            "cuVS",
            scx_gpu::test_gate::REQUIRE_CUVS_ENV,
            $crate::neighbors::gpu::cuvs_available(),
            module_path!(),
        ) {
            return;
        }
    };
}

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

// ─── Pinned harmonypy LISI reference (§7.19, ORG-7.21-4) ──────────────
#[cfg(test)]
#[path = "lisi_reference_tests.rs"]
mod lisi_reference_tests;

#[cfg(test)]
#[path = "lisi_reference_values.rs"]
pub(crate) mod lisi_reference_values;
pub mod mem_budget;
pub mod nb_glm;
pub mod neighbors;
pub mod pca;
pub mod pflog;
pub mod prefetch;
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
    binned_dispersion_norm, streaming_clip_square_sum, streaming_clip_square_sum_batched,
    streaming_mean_var, streaming_mean_var_batched, streaming_mean_var_expm1, BatchedHvgStats,
    HvgStats,
};
pub use leiden::{leiden, LeidenConfig, LeidenResult};
pub use lisi::{compute_lisi, LisiConfig, LisiResult};
pub use nb_glm::{
    pseudobulk_nb_glm, DispersionMethod, DispersionTrend, NbGlmContrast, NbGlmDiagnostics,
    NbGlmOptions, NbGlmResult,
};
pub use neighbors::{build_knn_graph, KnnResult};
pub use pca::{
    covariance_pca, covariance_pca_inmemory, covariance_pca_with_depth, pca_prefetch_depth,
    pflog_pca, pflog_pca_with_depth, randomized_pca, randomized_pca_inmemory,
    randomized_pca_with_depth, PcaResult, COVARIANCE_PCA_THRESHOLD,
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
    pseudobulk_means_gpu_dense, pseudobulk_means_gpu_from_slices, pseudobulk_means_gpu_streaming,
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
/// CPU per-stage timing profiler (io / decode / reduction / marshalling),
/// enabled by `SCX_CPU_PROFILE=1`. The CPU-path twin of [`scx_gpu::profile`];
/// the ranking oracle for the Phase-2 performance tasks. See
/// [`scx_format_io::profile`].
pub use scx_format_io::profile as cpu_profile;
#[cfg(feature = "gpu")]
pub use scx_gpu::nvcomp::nvcomp_enabled;
#[cfg(feature = "gpu")]
pub use scx_gpu::profile;
#[cfg(feature = "gpu")]
pub use scx_gpu::{
    cusparse_modern_abi_available, gpu_log1p, gpu_preprocess_to_csr, GpuDevice, GpuError,
    GpuMathMode, GpuPcaTuning, ProfileSnapshot, QrMethod, SpmmAlgPolicy, StageStat,
};
#[cfg(feature = "gpu")]
pub use scx_gpu::{
    decode_csr_shards_to_device, decode_csr_shards_to_device_with_stats, decode_shard_gpu,
    decode_shard_gpu_with_stats, DeviceDecodeStats, GpuCsr, GpuCsrPointers,
};
#[cfg(feature = "gpu")]
pub use scx_gpu::{GPU_NB_GLM_NSUB_MAX, GPU_NB_GLM_PMAX};
