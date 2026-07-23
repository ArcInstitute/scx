//! HVG device-dispatched wrappers (behind the `gpu` feature).
//!
//! Thin dispatchers: forward to the CPU path for `device = "cpu"` and to the
//! `scx-gpu` streaming kernels for `device = "gpu"`. A requested GPU route that
//! fails to initialize is a hard [`crate::error::AccelError::GpuInitFailed`], never a silent
//! CPU fallback (§4.1) — the binding resolves `"auto"` to `"cpu"` up front when
//! no GPU is available, so only an explicit/available GPU request reaches here.

use scx_format_io::ShardSource;

use super::cpu::{
    streaming_clip_square_sum, streaming_clip_square_sum_batched, streaming_mean_var,
    streaming_mean_var_batched, BatchedHvgStats, HvgStats,
};
use crate::error::Result;

/// Device-dispatched wrapper for [`streaming_mean_var`].
///
/// Forwards to the CPU implementation for `device = "cpu"` and to
/// [`scx_gpu::gpu_streaming_mean_var`] for `device = "gpu"`. The GPU path
/// accumulates per-column `Σ x` and `Σ x²` directly on-device via atomicAdd,
/// producing results that agree with the CPU path to ~1e-5 relative error on
/// typical scRNA-seq densities.
///
/// **Accuracy caveat (near-constant genes):** the ~1e-5 relative tolerance
/// holds only for genes with non-negligible variance. For near-constant genes
/// the relative error is effectively unbounded: the catastrophic-cancellation
/// `Σx² − n·mean²` form (see [`streaming_mean_var`] § Numerical stability)
/// combined with the GPU CSR path's nondeterministic-order f64 `atomicAdd`
/// reduction means the computed variance — and therefore the outcome of the
/// `< 0 → 0` clamp — can differ between runs and between GPU and CPU. At an HVG
/// dispersion/variance cutoff this can flip HVG membership for such genes. Pin
/// `device = "cpu"` if deterministic near-constant-gene behaviour is required.
///
/// Only available with `feature = "gpu"`.
///
/// A GPU route was requested (`device = "gpu"`), so GPU init failure is a hard
/// error ([`crate::error::AccelError::GpuInitFailed`]) — it does NOT silently fall back to the
/// CPU kernel, which would run under a GPU route stamp (§4.1). Callers that
/// want CPU on absent hardware pass `device = "cpu"` (or `"auto"`, resolved to
/// CPU by the binding when [`crate::gpu_available`] is false).
pub fn streaming_mean_var_with_device<S: ShardSource + Sync>(
    source: &S,
    device: &str,
    device_id: usize,
) -> Result<HvgStats> {
    if device != "gpu" {
        return streaming_mean_var(source);
    }

    let dev = scx_gpu::GpuDevice::new(device_id)
        .map_err(|e| crate::error::AccelError::GpuInitFailed(format!("device {device_id}: {e}")))?;
    let (means, variances) = scx_gpu::gpu_streaming_mean_var(&dev, source).map_err(|e| {
        crate::error::AccelError::LinAlg(format!("gpu_streaming_mean_var failed: {e}"))
    })?;
    Ok(HvgStats { means, variances })
}

/// Device-dispatched wrapper for [`streaming_clip_square_sum`].
///
/// Only available with `feature = "gpu"`. See
/// [`streaming_mean_var_with_device`] for semantics.
pub fn streaming_clip_square_sum_with_device<S: ShardSource + Sync>(
    source: &S,
    clip_val: &[f64],
    device: &str,
    device_id: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if device != "gpu" {
        return streaming_clip_square_sum(source, clip_val);
    }

    let dev = scx_gpu::GpuDevice::new(device_id)
        .map_err(|e| crate::error::AccelError::GpuInitFailed(format!("device {device_id}: {e}")))?;
    scx_gpu::gpu_streaming_clip_square_sum(&dev, source, clip_val).map_err(|e| {
        crate::error::AccelError::LinAlg(format!("gpu_streaming_clip_square_sum failed: {e}"))
    })
}

/// Device-dispatched wrapper for [`streaming_mean_var_batched`].
///
/// Forwards to the CPU implementation for `device = "cpu"`; for `device = "gpu"`
/// drives [`scx_gpu::gpu_streaming_mean_var_batched`] and finalises the
/// per-batch and global Bessel-corrected statistics on the host (identical
/// formula to the CPU function — see [`streaming_mean_var_batched`](super::cpu::streaming_mean_var_batched)).
///
/// A requested GPU route that fails to initialize errors
/// ([`crate::error::AccelError::GpuInitFailed`]); it does not silently fall back
/// to CPU (§4.1).
pub fn streaming_mean_var_batched_with_device<S: ShardSource + Sync>(
    source: &S,
    cell_batch: &[i32],
    n_batches: usize,
    device: &str,
    device_id: usize,
) -> Result<BatchedHvgStats> {
    if device != "gpu" {
        return streaming_mean_var_batched(source, cell_batch, n_batches);
    }

    let dev = scx_gpu::GpuDevice::new(device_id)
        .map_err(|e| crate::error::AccelError::GpuInitFailed(format!("device {device_id}: {e}")))?;

    let n_vars = source.n_vars();
    let (batch_sum, batch_sum_sq, batch_counts) =
        scx_gpu::gpu_streaming_mean_var_batched(&dev, source, cell_batch, n_batches).map_err(
            |e| crate::error::AccelError::LinAlg(format!("gpu_streaming_mean_var_batched: {e}")),
        )?;

    // Per-batch means & variances (Bessel's correction).
    let mut per_batch = Vec::with_capacity(n_batches);
    for b in 0..n_batches {
        let n = batch_counts[b] as f64;
        let mut means = vec![0.0f64; n_vars];
        let mut variances = vec![0.0f64; n_vars];
        if batch_counts[b] > 0 {
            let denom = (n - 1.0).max(1.0);
            for j in 0..n_vars {
                let mean = batch_sum[b][j] / n;
                means[j] = mean;
                variances[j] = ((batch_sum_sq[b][j] - n * mean * mean) / denom).max(0.0);
            }
        }
        per_batch.push(HvgStats { means, variances });
    }

    // Derive global stats from per-batch accumulators — matches the CPU path.
    let total_n: usize = batch_counts.iter().sum();
    let total_f = total_n as f64;
    let mut global_means = vec![0.0f64; n_vars];
    let mut global_variances = vec![0.0f64; n_vars];
    if total_n > 0 {
        let denom = (total_f - 1.0).max(1.0);
        for j in 0..n_vars {
            let global_sum: f64 = batch_sum.iter().map(|bs| bs[j]).sum();
            let global_sum_sq: f64 = batch_sum_sq.iter().map(|bs| bs[j]).sum();
            let mean = global_sum / total_f;
            global_means[j] = mean;
            global_variances[j] = ((global_sum_sq - total_f * mean * mean) / denom).max(0.0);
        }
    }

    Ok(BatchedHvgStats {
        per_batch,
        global: HvgStats {
            means: global_means,
            variances: global_variances,
        },
        batch_counts,
    })
}

/// Device-dispatched wrapper for [`streaming_clip_square_sum_batched`].
///
/// Only available with `feature = "gpu"`. See
/// [`streaming_mean_var_batched_with_device`] for semantics.
pub fn streaming_clip_square_sum_batched_with_device<S: ShardSource + Sync>(
    source: &S,
    cell_batch: &[i32],
    n_batches: usize,
    clip_vals: &[Vec<f64>],
    device: &str,
    device_id: usize,
) -> Result<Vec<(Vec<f64>, Vec<f64>)>> {
    if device != "gpu" {
        return streaming_clip_square_sum_batched(source, cell_batch, n_batches, clip_vals);
    }

    let dev = scx_gpu::GpuDevice::new(device_id)
        .map_err(|e| crate::error::AccelError::GpuInitFailed(format!("device {device_id}: {e}")))?;
    scx_gpu::gpu_streaming_clip_square_sum_batched(&dev, source, cell_batch, n_batches, clip_vals)
        .map_err(|e| {
            crate::error::AccelError::LinAlg(format!("gpu_streaming_clip_square_sum_batched: {e}"))
        })
}
