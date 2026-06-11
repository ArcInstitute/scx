//! Device-resident fused GPU pipelines.
//!
//! GPU-only; [`gpu`] runs multiple accelerators back-to-back while keeping
//! intermediate results resident on the device (e.g. [`gpu::pca_then_knn_gpu`]).

pub mod gpu;

pub use gpu::pca_then_knn_gpu;
