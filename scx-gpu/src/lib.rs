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

/// Try to acquire a GPU device, skipping the test if CUDA is unavailable.
///
/// Usage in `#[cfg(test)]` modules:
/// ```ignore
/// let dev = require_gpu!();
/// ```
///
/// Defined once here; used across all test modules in the crate.
#[cfg(test)]
macro_rules! require_gpu {
    () => {
        match $crate::device::GpuDevice::new(0) {
            Ok(dev) => dev,
            Err(_) => {
                eprintln!("CUDA not available — skipping GPU test");
                return;
            }
        }
    };
}

// --- Module declarations ---

#[cfg(any(test, feature = "bench"))]
pub mod test_utils;

pub mod cast_gpu;
pub mod cublas;
pub mod curand;
pub mod cusolver;
pub mod cusparse;
pub mod device;
pub mod error;
pub mod forbp_gpu;
pub mod gpu_diffexp;
pub mod gpu_harmony;
pub mod gpu_hvg;
pub mod gpu_knn;
pub mod gpu_pca;
pub mod gpu_pca_covariance;
pub mod gpu_preprocess;
pub mod gpu_umap;
pub mod linear_operator;
pub mod rice_gpu;
pub mod shard_decode;
pub mod shard_pipeline;
pub mod sparse_dense;

// Re-export primary types for convenience.
pub use cublas::{gpu_sgemm, gpu_sgemv, gpu_sger, gpu_strsm, CublasHandle};
pub use curand::random_gaussian_gpu;
pub use cusolver::{gpu_cholesky_qr2, gpu_eigh_sym, gpu_qr_q, CusolverHandle, QrMethod};
pub use cusparse::{
    cusparse_modern_abi_available, spmm_csr, spmm_csr_transpose, CusparseHandle,
    CusparseSpMatDescr, DnMatDescr, GpuCsrPointers,
};
pub use device::GpuDevice;
pub use error::{GpuError, Result};
pub use forbp_gpu::forbp_decode_gpu;
pub use gpu_diffexp::{
    default_gpu_de_gene_chunk_size, gpu_de_block_sort, gpu_de_combined_tie_term, gpu_de_pvalues,
    gpu_de_scatter_gene_major, gpu_de_searchsorted_ranksum, gpu_de_searchsorted_u_stat,
    gpu_de_tie_term, gpu_de_upload_chunk, GpuDeChunkScratch, GPU_DE_BLOCK_SORT_CAPACITY,
};
pub use gpu_harmony::{
    gpu_harmony_block_oe_update, gpu_harmony_block_softmax_penalty, gpu_harmony_compute_o_e_full,
    gpu_harmony_correction, gpu_harmony_correction_grouped, gpu_harmony_distances,
    gpu_harmony_distances_gemm, gpu_harmony_l2_normalize_cols, gpu_harmony_memory_bytes,
    gpu_harmony_obj_cross, gpu_harmony_obj_kmeans_entropy, gpu_harmony_softmax,
    gpu_harmony_softmax_penalty, gpu_harmony_z_sum,
};
pub use gpu_hvg::{
    gpu_streaming_clip_square_sum, gpu_streaming_clip_square_sum_batched, gpu_streaming_mean_var,
    gpu_streaming_mean_var_batched,
};
pub use gpu_knn::{cuvs_available, gpu_knn_cagra, GpuKnnResult};
pub use gpu_pca::{gpu_randomized_pca, mean_correct_gpu, GpuPcaResult};
pub use gpu_pca_covariance::gpu_covariance_pca;
pub use gpu_preprocess::{
    gpu_apply_fused_ops, gpu_log1p, gpu_normalize, gpu_normalize_log1p, gpu_preprocess_to_csr,
};
pub use gpu_umap::{gpu_umap_native, GpuUmapResult};
pub use linear_operator::CenteredSparseOperator;
pub use rice_gpu::rice_decode_gpu;
pub use shard_decode::{decode_shard_gpu, GpuCsr};
pub use shard_pipeline::DoubleBufferedShardLoader;
pub use sparse_dense::{sparse_to_dense_gpu, sparse_to_dense_gpu_into};

// Re-export cudarc types used in public API signatures.
pub use cudarc::driver::safe::{CudaModule, CudaSlice, CudaStream};
pub use cudarc::nvrtc::Ptx;
