//! HVG device-dispatched wrappers (behind the `gpu` feature).
//!
//! Thin dispatchers: forward to the CPU path for `device = "cpu"` and to the
//! `scx-gpu` streaming kernels for `device = "gpu"`. A requested GPU route that
//! fails to initialize is a hard [`crate::error::AccelError::GpuInitFailed`],
//! never a silent CPU fallback (§4.1).
//!
//! `device = "auto"` is resolved to CPU up front by the binding when **no GPU
//! is visible** (`gpu_available()` is false). A GPU that is *visible* but whose
//! context fails to initialize (compute-exclusive already claimed, ECC/XID,
//! context-time OOM, driver/runtime mismatch) surfaces `GpuInitFailed` even
//! under `auto` — a deliberate fail-loud choice: the op errors rather than
//! silently returning a CPU result under a GPU route stamp. (Graceful
//! auto-degradation on a broken context would be an untestable branch — it
//! cannot be exercised on a healthy GPU — so it is intentionally not done here.)

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
/// A GPU route was requested (`device = "gpu"`, or `"auto"` on a host with a
/// visible GPU), so GPU init failure is a hard error
/// ([`crate::error::AccelError::GpuInitFailed`]) — it does NOT silently fall
/// back to the CPU kernel, which would run under a GPU route stamp (§4.1).
/// Callers that want CPU on absent hardware pass `device = "cpu"` (or `"auto"`,
/// which the binding resolves to CPU when [`crate::gpu_available`] is false).
/// A *visible-but-broken-context* GPU under `auto` fails loud rather than
/// degrading (see the module docs).
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

    // Per-batch and global finalize, through the same
    // `scx_sparse::finalize_column_moments` the CPU batched path uses — so
    // "matches the CPU path" below is shared code rather than two copies of the
    // formula that happen to agree.
    let mut per_batch = Vec::with_capacity(n_batches);
    for b in 0..n_batches {
        // Unlike `streaming_mean_var_batched`, this route never sees the shard
        // values on the host, so it cannot call `ensure_finite_hvg_data`. It
        // checks the accumulated moments instead — equivalent per column, since
        // a non-finite input value leaves its column's sum non-finite, and
        // O(n_vars) rather than O(nnz). Before Phase 7a `.max(0.0)` absorbed a
        // NaN variance to 0.0 here, so the gene silently looked constant.
        if let Some(j) = scx_sparse::first_non_finite_column(&batch_sum[b], &batch_sum_sq[b]) {
            return Err(crate::error::AccelError::InvalidInput(format!(
                "streaming_mean_var_batched_with_device: batch {b}, column {j}                  accumulated a non-finite moment (sum = {}, sum_sq = {}) — the                  input contains NaN/Inf, or a finite input overflowed. HVG                  statistics would be meaningless.",
                batch_sum[b][j], batch_sum_sq[b][j]
            )));
        }
        let m =
            scx_sparse::finalize_column_moments(&batch_sum[b], &batch_sum_sq[b], batch_counts[b]);
        m.warn_if_unstable(&format!(
            "streaming_mean_var_batched_with_device[batch {b}]"
        ));
        per_batch.push(HvgStats {
            means: m.means,
            variances: m.variances,
        });
    }

    let total_n: usize = batch_counts.iter().sum();
    let mut global_sum = vec![0.0f64; n_vars];
    let mut global_sum_sq = vec![0.0f64; n_vars];
    for j in 0..n_vars {
        global_sum[j] = batch_sum.iter().map(|bs| bs[j]).sum();
        global_sum_sq[j] = batch_sum_sq.iter().map(|bs| bs[j]).sum();
    }
    // The global sums are derived from the per-batch ones, each already checked
    // above, so this cannot fire on non-finite *input* — it can still fire if
    // summing finite per-batch moments overflows to an infinity, which is
    // exactly the case an input scan would miss.
    if let Some(j) = scx_sparse::first_non_finite_column(&global_sum, &global_sum_sq) {
        return Err(crate::error::AccelError::InvalidInput(format!(
            "streaming_mean_var_batched_with_device: global column {j}              accumulated a non-finite moment (sum = {}, sum_sq = {}) after              summing {n_batches} finite per-batch moments — the totals overflowed.",
            global_sum[j], global_sum_sq[j]
        )));
    }
    let global = scx_sparse::finalize_column_moments(&global_sum, &global_sum_sq, total_n);
    global.warn_if_unstable("streaming_mean_var_batched_with_device[global]");

    Ok(BatchedHvgStats {
        per_batch,
        global: HvgStats {
            means: global.means,
            variances: global.variances,
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
