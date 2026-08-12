//! GPU-accelerated codec decoding, cuSPARSE interop, cuSOLVER dense
//! operations, cuVS CAGRA kNN, GPU preprocessing, and GPUDirect Storage for SCX.
//!
//! This crate provides CUDA-based decoding of SCX Scx1 codecs (Rice values,
//! FOR-BP indices), cuSPARSE CSR matrix interop, GPU sparse-to-dense
//! conversion with HVG projection, SpMM (sparse × dense matrix multiply)
//! for GPU PCA, cuSOLVER QR decomposition, cuRAND random matrix generation,
//! and optional GPUDirect Storage (GDS) for direct NVMe-to-GPU data transfer.
//!
//! ## Architecture (cudarc 0.19)
//!
//! cudarc 0.19 replaced `CudaDevice` with two separate abstractions:
//! - [`cudarc::driver::CudaContext`] — device handle (context management, module loading)
//! - [`cudarc::driver::CudaStream`] — work scheduling (memory ops, kernel launches)
//!
//! The [`GpuDevice`] struct wraps both into a single ergonomic handle.

// --- Test-only macro (must precede module declarations for textual scoping) ---

/// Acquire a GPU device for a test, or return early if CUDA is unavailable.
///
/// Usage in `#[cfg(test)]` modules:
/// ```ignore
/// #[test]
/// #[ignore = "requires a CUDA GPU"]
/// fn my_gpu_test() {
///     let dev = require_gpu!();
/// }
/// ```
///
/// **The `#[ignore]` is not optional.** Without it the test is *selected* on a
/// CPU-only host, returns immediately, and is counted as passed — which is how
/// 170 tests in this crate came to be guaranteed no-ops in CI. With it, a
/// default run reports them as ignored, and the harness opts back in with
/// `--include-ignored` plus `SCX_REQUIRE_GPU=1` so a skip there is a failure.
/// `tests/gpu_test_gating.rs` enforces the pairing in both directions.
///
/// The decision itself lives in [`crate::test_gate`] so that `scx-accel`'s GPU
/// tests share it; only the `return` has to be a macro.
#[cfg(test)]
macro_rules! require_gpu {
    () => {
        match $crate::test_gate::device_or_skip(module_path!()) {
            Some(dev) => dev,
            None => return,
        }
    };
}

/// Gate a test on an optional CUDA library rather than on the device itself.
///
/// A GPU node legitimately may not carry nvcomp or cuVS, so these skip by
/// default and report via [`crate::test_gate::SKIP_MARKER`]; `SCX_REQUIRE_NVCOMP=1`
/// / `SCX_REQUIRE_CUVS=1` make them hard requirements for a run that means to
/// cover them. Use *below* `require_gpu!()`, never instead of it.
#[cfg(test)]
macro_rules! require_gpu_cap {
    (nvcomp) => {
        if !$crate::test_gate::capability_or_skip(
            "nvcomp",
            $crate::test_gate::REQUIRE_NVCOMP_ENV,
            $crate::nvcomp::nvcomp_available(),
            module_path!(),
        ) {
            return;
        }
    };
    (cuvs) => {
        if !$crate::test_gate::capability_or_skip(
            "cuVS",
            $crate::test_gate::REQUIRE_CUVS_ENV,
            $crate::gpu_knn::cuvs_available(),
            module_path!(),
        ) {
            return;
        }
    };
}

// --- Module declarations ---

#[cfg(any(test, feature = "bench"))]
pub mod test_utils;

pub mod backed_gpu_matrix_source;
pub mod cast_gpu;
pub mod cublas;
pub mod curand;
pub mod cusolver;
pub mod cusparse;
pub mod device;
pub mod device_resident;
pub mod error;
pub mod forbp_gpu;
pub mod gpu_csc_shard_source;
pub mod gpu_csr_assemble;
pub mod gpu_diffexp;
pub mod gpu_graph;
pub mod gpu_harmony;
pub mod gpu_hvg;
pub mod gpu_knn;
pub mod gpu_matrix_source;
pub mod gpu_nb_glm;
pub mod gpu_pairwise;
pub mod gpu_pca;
pub mod gpu_pca_resident;
pub mod gpu_preprocess;
pub mod gpu_pseudobulk;
pub mod gpu_shard_source;
pub mod linear_operator;
pub mod math_policy;
pub mod nvcomp;
pub mod preprocessed_gpu_matrix_source;
pub mod profile;
pub mod resident_gpu_csr_source;
pub mod rice_gpu;
pub mod shard_decode;
pub mod shufdelta_gpu;
pub mod sparse_dense;
pub mod staging;
pub mod test_gate;

// Re-export primary types for convenience.
pub use backed_gpu_matrix_source::BackedGpuMatrixSource;
pub use cublas::{gpu_sgemm, gpu_sgemv, gpu_sger, gpu_strsm, gpu_transpose_f32, CublasHandle};
pub use curand::random_gaussian_gpu;
pub use cusolver::{gpu_cholesky_qr2, gpu_eigh_sym, gpu_qr_q, CusolverHandle, QrMethod};
pub use cusparse::{
    cusparse_modern_abi_available, spmm_csr, spmm_csr_transpose, CusparseHandle,
    CusparseSpMatDescr, DnMatDescr, GpuCsrPointers,
};
pub use device::GpuDevice;
pub use device_resident::{DeviceEmbedding, DeviceKnnGraph};
pub use error::{GpuError, Result};
pub use forbp_gpu::forbp_decode_gpu;
pub use gpu_csc_shard_source::{GpuCscShardSource, GpuCscShardView, RawGpuCscShardSource};
pub use gpu_csr_assemble::{decode_csr_shards_to_device, decode_csr_shards_to_device_with_stats};
pub use gpu_diffexp::{
    build_cell_to_group_dev, build_cell_to_pool_dev, build_cell_to_pos_dev,
    default_gpu_de_gene_chunk_size, gpu_de_aux_elems, gpu_de_block_sort, gpu_de_budget_gene_chunk,
    gpu_de_combined_tie_term, gpu_de_per_gene_scratch_bytes, gpu_de_pseudobulk_all_groups,
    gpu_de_pseudobulk_csc_direct, gpu_de_pseudobulk_csr_direct, gpu_de_pvalues,
    gpu_de_scatter_csc_to_gene_major, gpu_de_scatter_csr_to_gene_major_filtered,
    gpu_de_scatter_gene_major, gpu_de_scatter_gene_major_dev, gpu_de_scatter_shard_to_dense,
    gpu_de_scatter_shard_to_gene_major, gpu_de_searchsorted_ranksum, gpu_de_searchsorted_u_stat,
    gpu_de_tie_term, gpu_de_upload_chunk, GpuDeChunkScratch, GPU_DE_BLOCK_SORT_CAPACITY,
    GPU_DE_MIN_GENE_CHUNK,
};
pub use gpu_graph::{
    capture_graph, cuda_graphs_enabled, exec_kernel_node_set_params, graph_kernel_nodes,
    read_kernel_node_params, set_cuda_graphs_enabled_override, GpuGraphCache, GpuGraphMetrics,
    GraphKey,
};
pub use gpu_harmony::{
    gpu_harmony_block_oe_update, gpu_harmony_block_softmax_penalty, gpu_harmony_compute_o_e_full,
    gpu_harmony_correction, gpu_harmony_correction_grouped, gpu_harmony_distances,
    gpu_harmony_distances_gemm, gpu_harmony_l2_normalize_cols, gpu_harmony_memory_bytes,
    gpu_harmony_obj_cross, gpu_harmony_obj_kmeans_entropy, gpu_harmony_reduce_objective,
    gpu_harmony_softmax, gpu_harmony_softmax_penalty, gpu_harmony_z_sum,
};
pub use gpu_hvg::{
    gpu_streaming_clip_square_sum, gpu_streaming_clip_square_sum_batched,
    gpu_streaming_clip_square_sum_csc, gpu_streaming_mean_var, gpu_streaming_mean_var_batched,
    gpu_streaming_mean_var_csc,
};
pub use gpu_knn::{cuvs_available, gpu_knn_cagra_device, GpuKnnResult};
pub use gpu_matrix_source::{GpuMatrixSource, GpuTransformSpec, LayoutSet, SourceRouteMetadata};
pub use gpu_nb_glm::{
    gpu_nb_glm_fit, GpuNbGlmFit, GpuNbGlmOpts, GpuNbGlmPass, GPU_NB_GLM_METHOD_CR_MLE,
    GPU_NB_GLM_METHOD_CR_SHRUNK, GPU_NB_GLM_METHOD_MOMENTS, GPU_NB_GLM_NSUB_MAX, GPU_NB_GLM_PMAX,
};
pub use gpu_pairwise::gpu_mean_pairwise_distance;
pub use gpu_pca::{
    gpu_randomized_pca, gpu_randomized_pca_device, mean_correct_gpu, GpuPcaDeviceResult,
    GpuPcaResult,
};
pub use gpu_preprocess::{
    gpu_apply_fused_ops, gpu_log1p, gpu_normalize, gpu_normalize_log1p, gpu_preprocess_to_csr,
};
pub use gpu_pseudobulk::{gpu_pseudobulk_means_csr, gpu_pseudobulk_means_dense};
pub use gpu_shard_source::{GpuPreprocessedShardSource, GpuShardSource, RawGpuShardSource};
pub use linear_operator::CenteredSparseOperator;
pub use math_policy::{GpuMathMode, GpuPcaTuning, SpmmAlgPolicy};
pub use preprocessed_gpu_matrix_source::PreprocessedGpuMatrixSource;
pub use profile::{ProfileSnapshot, StageStat};
pub use resident_gpu_csr_source::{
    try_build_resident, ResidentGpuCsrSource, DEFAULT_RESIDENT_MAX_FRAC,
};
pub use rice_gpu::rice_decode_gpu;
pub use shard_decode::{decode_shard_gpu, decode_shard_gpu_with_stats, DeviceDecodeStats, GpuCsr};
pub use sparse_dense::{
    sparse_to_dense_gpu, sparse_to_dense_gpu_into, sparse_to_dense_gpu_into_view, upload_hvg_map,
    validate_hvg_map, HVG_MAP_SKIP,
};
pub use staging::{GpuCsrShardView, GpuCsrSlot, InMemoryCsrShardSource, PinnedCsrSlot};

// Re-export cudarc types used in public API signatures.
pub use cudarc::driver::safe::{CudaGraph, CudaModule, CudaSlice, CudaStream};
pub use cudarc::nvrtc::Ptx;
