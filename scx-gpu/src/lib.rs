//! GPU-accelerated codec decoding, cuSPARSE interop, cuSOLVER dense
//! operations, cuVS CAGRA kNN, and GPUDirect Storage for SCX.
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

#[cfg(any(test, feature = "bench"))]
pub mod test_utils;

pub mod cast_gpu;
pub mod curand;
pub mod cusolver;
pub mod cusparse;
pub mod device;
pub mod error;
pub mod forbp_gpu;
pub mod gpu_knn;
pub mod gpu_pca;
pub mod rice_gpu;
pub mod shard_decode;
pub mod sparse_dense;

// Re-export primary types for convenience.
pub use curand::random_gaussian_gpu;
pub use cusolver::{gpu_qr_q, CusolverHandle};
pub use cusparse::{
    spmm_csr, spmm_csr_transpose, CusparseHandle, CusparseSpMatDescr, DnMatDescr, GpuCsrPointers,
};
pub use device::GpuDevice;
pub use error::{GpuError, Result};
pub use forbp_gpu::forbp_decode_gpu;
pub use gpu_knn::{cuvs_available, gpu_knn_cagra, GpuKnnResult};
pub use gpu_pca::{gpu_randomized_pca, mean_correct_gpu, GpuPcaResult};
pub use rice_gpu::rice_decode_gpu;
pub use shard_decode::{decode_shard_gpu, GpuCsr};
pub use sparse_dense::sparse_to_dense_gpu;

// Re-export cudarc types used in public API signatures.
pub use cudarc::driver::safe::{CudaModule, CudaSlice, CudaStream};
pub use cudarc::nvrtc::Ptx;
