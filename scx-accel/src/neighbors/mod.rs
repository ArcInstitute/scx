//! k-nearest-neighbor graph construction.
//!
//! [`cpu`] holds the HNSW / exact-gemm host implementation; [`gpu`] (behind the
//! `gpu` feature) exposes the cuVS availability probe (device-resident CAGRA
//! kNN runs through the fused PCA→kNN pipeline in `crate::fused`).

pub mod cpu;
#[cfg(feature = "gpu")]
pub mod gpu;

pub use cpu::{build_knn_graph, KnnResult};
#[cfg(feature = "gpu")]
pub use gpu::cuvs_available;
