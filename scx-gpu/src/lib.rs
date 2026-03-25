//! GPU-accelerated codec decoding, cuSPARSE interop, and GPUDirect Storage
//! for SCX.
//!
//! This crate provides CUDA-based decoding of SCX Scx1 codecs (Rice values,
//! FOR-BP indices), cuSPARSE CSR matrix interop, GPU sparse-to-dense
//! conversion with HVG projection, and optional GPUDirect Storage (GDS) for
//! direct NVMe-to-GPU data transfer.
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
pub mod cusparse;
pub mod device;
pub mod error;
pub mod forbp_gpu;
pub mod rice_gpu;
pub mod shard_decode;
pub mod sparse_dense;

// Re-export primary types for convenience.
pub use cusparse::{CusparseHandle, CusparseSpMatDescr, GpuCsrPointers};
pub use device::GpuDevice;
pub use error::{GpuError, Result};
pub use forbp_gpu::forbp_decode_gpu;
pub use rice_gpu::rice_decode_gpu;
pub use shard_decode::{decode_shard_gpu, GpuCsr};
pub use sparse_dense::sparse_to_dense_gpu;

// Re-export cudarc types used in public API signatures.
pub use cudarc::driver::safe::{CudaModule, CudaSlice, CudaStream};
pub use cudarc::nvrtc::Ptx;
