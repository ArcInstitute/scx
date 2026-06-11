//! Harmony2 batch integration.
//!
//! [`cpu`] holds the host implementation, the shared [`cpu::HarmonyState`]
//! clustering state, and (behind the `gpu` feature) the GPU orchestration in
//! `cpu::gpu`, which drives the `scx-gpu` device kernels.

pub mod cpu;

#[cfg(feature = "gpu")]
pub use cpu::gpu::harmony_integrate_gpu;
pub use cpu::{harmony_integrate, BatchCovariate, HarmonyConfig, HarmonyResult};
