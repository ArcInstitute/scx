//! Principal component analysis.
//!
//! [`cpu`] holds the covariance and randomized host paths and the
//! `COVARIANCE_PCA_THRESHOLD` routing ceiling; [`gpu`] (behind the `gpu`
//! feature) is thin orchestration over the `scx-gpu` streaming PCA pipeline.

pub mod cpu;
#[cfg(feature = "gpu")]
pub mod gpu;

pub use cpu::{
    covariance_pca, covariance_pca_inmemory, pca_prefetch_depth, pflog_pca, randomized_pca,
    randomized_pca_inmemory, PcaResult, COVARIANCE_PCA_THRESHOLD,
};
#[cfg(feature = "gpu")]
pub use gpu::{gpu_available, gpu_info, randomized_pca_gpu, GpuInfo};
