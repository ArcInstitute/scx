//! GPU-accelerated PCA dispatch (behind the `gpu` feature).
//!
//! Thin orchestration over `scx_gpu::gpu_randomized_pca`; returns the same
//! [`PcaResult`](super::cpu::PcaResult) the CPU path produces.

use scx_format_io::ShardSource;

use super::cpu::PcaResult;
use crate::error::{AccelError, Result};

/// GPU-accelerated randomized PCA from any `ShardSource`.
///
/// Wraps [`scx_gpu::gpu_randomized_pca`] to stream data shard-by-shard on GPU
/// (cuSPARSE SpMM, cuSOLVER QR) and returns a [`PcaResult`] with host-side
/// data matching the CPU path's output format.
///
/// Requires the `gpu` feature to be enabled.
///
/// # Arguments
///
/// * `device_id` — CUDA device ordinal (0 for first GPU)
/// * `source` — any `ShardSource + Sync` (backed reader, lazy transform, or
///   in-memory CSR wrapper)
/// * `n_components` — Number of principal components to compute
/// * `n_oversamples` — Extra dimensions for accuracy (default: 10)
/// * `n_power_iterations` — Power iterations for spectral accuracy (default: 2)
/// * `zero_center` — Whether to mean-center the data (default: true)
/// * `seed` — Random seed for reproducibility
/// * `qr_method` — Householder (default, always-stable) or CholeskyQR2 (opt-in,
///   faster but fails with `CuSolverError` on non-SPD Gram matrices)
#[allow(clippy::too_many_arguments)]
pub fn randomized_pca_gpu<S: ShardSource + Sync>(
    device_id: usize,
    source: &S,
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    seed: u64,
    qr_method: scx_gpu::QrMethod,
    tuning: scx_gpu::GpuPcaTuning,
) -> Result<PcaResult> {
    let dev = scx_gpu::GpuDevice::new(device_id)
        .map_err(|e| AccelError::LinAlg(format!("GPU init failed: {e}")))?;

    let gpu_result = scx_gpu::gpu_randomized_pca(
        &dev,
        source,
        n_components,
        n_oversamples,
        n_power_iterations,
        zero_center,
        seed,
        qr_method,
        tuning,
    )
    .map_err(|e| AccelError::LinAlg(format!("GPU PCA failed: {e}")))?;

    // Convert GpuPcaResult → PcaResult
    // embeddings: f32 row-major → f64 row-major
    let embeddings: Vec<f64> = gpu_result.embeddings.iter().map(|&v| v as f64).collect();
    // components: f32 row-major → f64 row-major
    let components: Vec<f64> = gpu_result.components.iter().map(|&v| v as f64).collect();

    Ok(PcaResult {
        embeddings,
        components,
        variance_explained: gpu_result.variance_explained,
        variance_ratio: gpu_result.variance_ratio,
        mean: gpu_result.mean,
        n_components: gpu_result.n_components,
        n_obs: gpu_result.n_obs,
        n_vars: gpu_result.n_vars,
        resident_csr: Some(gpu_result.resident_csr),
    })
}

/// Check whether a GPU is available for GPU-accelerated PCA.
///
/// Returns `true` if at least one CUDA device is found.
pub fn gpu_available() -> bool {
    scx_gpu::GpuDevice::count().is_ok_and(|n| n > 0)
}

/// GPU device information returned by [`gpu_info`].
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// Human-readable device name (e.g. "NVIDIA A100-SXM4-80GB").
    pub device_name: String,
    /// Total VRAM in bytes.
    pub total_vram_bytes: usize,
    /// Free VRAM in bytes.
    pub free_vram_bytes: usize,
}

/// Query GPU device information for device 0.
///
/// Returns `None` if no GPU is available or CUDA initialization fails.
pub fn gpu_info() -> Option<GpuInfo> {
    let count = scx_gpu::GpuDevice::count().ok()?;
    if count == 0 {
        return None;
    }
    let dev = scx_gpu::GpuDevice::new(0).ok()?;
    let device_name = dev.name().unwrap_or_else(|_| "unknown".to_string());
    let (free, total) = dev.free_memory().ok()?;
    Some(GpuInfo {
        device_name,
        total_vram_bytes: total,
        free_vram_bytes: free,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
