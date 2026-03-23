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

pub mod device;
pub mod error;

// Re-export primary types for convenience.
pub use device::GpuDevice;
pub use error::{GpuError, Result};

// Re-export cudarc types used in public API signatures.
pub use cudarc::driver::safe::{CudaModule, CudaSlice, CudaStream};
pub use cudarc::nvrtc::Ptx;
