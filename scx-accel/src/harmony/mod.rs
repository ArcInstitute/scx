//! Harmony2 batch integration.
//!
//! [`cpu`] holds the host implementation, the shared [`cpu::HarmonyState`]
//! clustering state, and (behind the `gpu` feature) the GPU orchestration in
//! `cpu::gpu`, which drives the `scx-gpu` device kernels.
//!
//! On disk this op is three files — `cpu.rs`, `gpu.rs`, `tests.rs` — but `gpu`
//! and `tests` are declared as children of `cpu` (via `#[path]`, see `cpu.rs`),
//! not siblings, so their `use super::*` can reach the private `HarmonyState`
//! internals the GPU path and tests share with the host code.

pub mod cpu;

#[cfg(feature = "gpu")]
pub use cpu::gpu::harmony_integrate_gpu;
pub use cpu::{harmony_integrate, BatchCovariate, HarmonyConfig, HarmonyResult};
