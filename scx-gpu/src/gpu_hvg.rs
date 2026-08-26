//! Streaming GPU HVG (highly variable genes) primitives.
//!
//! Mirrors the CPU streaming functions in `scx_accel::hvg`:
//!
//! * [`gpu_streaming_mean_var`] — per-gene mean and variance (Bessel's correction).
//! * [`gpu_streaming_clip_square_sum`] — clipped sum / clipped-squared sum per gene
//!   used by the seurat_v3 HVG algorithm.
//! * [`gpu_streaming_mean_var_batched`] — same as `gpu_streaming_mean_var` but
//!   grouped by a per-cell batch id (one i32 per visible cell, `-1` to skip).
//! * [`gpu_streaming_clip_square_sum_batched`] — per-batch variant of
//!   `gpu_streaming_clip_square_sum` with per-batch clip thresholds.
//!
//! All four iterate shards via [`BackedGpuMatrixSource`] (the unified
//! [`GpuMatrixSource`] CSR path with G3 staging) and accumulate per-gene
//! statistics directly on-device in f64 via `atomicAdd` (compute 6.x+ required,
//! which scx-gpu already targets via `compute_70` in `build.rs`).
//!
//! These GPU kernels are accessed via the `scx_accel::*_with_device` dispatch
//! wrappers (see `scx-accel/src/hvg/gpu.rs`) when `device = "gpu"`. The f64
//! accumulator path keeps numerical behaviour parity with the CPU fallback —
//! the two implementations agree to ~1e-5 relative error on typical scRNA-seq
//! densities.

use cudarc::driver::safe::CudaSlice;
use cudarc::driver::PushKernelArg;

use scx_format_io::{ColumnShardSource, ShardSource};

use crate::backed_gpu_matrix_source::BackedGpuMatrixSource;
use crate::device::{flat_launch_1d, GpuDevice};
use crate::error::GpuError;
use crate::gpu_csc_shard_source::{GpuCscShardSource, RawGpuCscShardSource};
use crate::gpu_matrix_source::GpuMatrixSource;
use crate::gpu_matrix_source::{ValidationChecks, ValidationPolicy};

/// PTX for the HVG atomicAdd kernels. Reuses the same `colmajor_ops` module
/// (shared with PCA helpers) to avoid a second PTX load.
const COLMAJOR_OPS_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/colmajor_ops.ptx"));

/// Per-batch column accumulator: outer = batch, inner = per-gene sums.
pub type PerBatchColVec = Vec<Vec<f64>>;
/// Per-batch `(batch_sum, sq_batch_sum)` pair from the clipped accumulator.
pub type PerBatchClipSums = Vec<(Vec<f64>, Vec<f64>)>;

/// GPU equivalent of `scx_accel::streaming_mean_var`.
///
/// Streams shards through [`BackedGpuMatrixSource`], atomically
/// accumulating per-column `Σ x` and `Σ x²` into f64 device buffers, then
/// computes `mean = Σ / n` and `var = (Σ² - n · mean²) / (n - 1)` on the host
/// (O(n_vars) work). Negative variances from numerical noise are clamped to 0.
///
/// Returns a pair `(means, variances)`, both f64 vectors of length `n_vars`.
///
/// # Determinism (finding ACC6)
///
/// This CSR path accumulates `Σx`/`Σx²` via cross-block f64 `atomicAdd`, so the
/// summation order is **not** deterministic: the per-column sums can differ at
/// the last bits between runs and from the CPU path. For a near-constant gene
/// that tiny difference — combined with the cancellation-prone `Σx² − n·mean²`
/// form — can flip the `var < 0 → 0` clamp and thus HVG membership at a cutoff.
/// The deterministic alternatives are [`gpu_streaming_mean_var_csc`] (one block
/// per column, no cross-block atomics) and the CPU
/// `scx_accel::streaming_mean_var` (sequential, fixed-order f64 accumulation).
/// Making *this* CSR path deterministic — a per-block-partials + tree-merge
/// reduction (optionally emitting Welford `(count, mean, M2)` moments) — is a
/// tracked follow-on; prefer the CSC route when a CSC sidecar is available and
/// reproducibility matters.
pub fn gpu_streaming_mean_var(
    dev: &GpuDevice,
    source: &(dyn ShardSource + Sync),
) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
    let n_vars = source.n_vars();
    let n_obs = source.n_obs();

    if n_obs == 0 {
        return Ok((vec![0.0; n_vars], vec![0.0; n_vars]));
    }

    let mut d_col_sum = dev.alloc_zeros::<f64>(n_vars)?;
    let mut d_col_sum_sq = dev.alloc_zeros::<f64>(n_vars)?;

    let module = dev.load_module_cached(COLMAJOR_OPS_PTX)?;
    let func = module
        .load_function("col_sum_sq_nonzeros_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("col_sum_sq_nonzeros: {e}")))?;

    let mut src = BackedGpuMatrixSource::new(dev, source)?.with_validation(ValidationPolicy::new(
        ValidationChecks::IN_RANGE,
        "highly_variable_genes",
    ));
    src.for_each_gpu_csr_shard(&mut |_idx, slot| {
        let view = slot.view();
        let nnz = view.data.len() as i64;
        if nnz == 0 {
            return Ok(());
        }
        let threads: u32 = 256;
        let cfg = flat_launch_1d(nnz as u64, threads)?;
        unsafe {
            dev.stream()
                .launch_builder(&func)
                .arg(&view.indices)
                .arg(&view.data)
                .arg(&nnz)
                .arg(&mut d_col_sum)
                .arg(&mut d_col_sum_sq)
                .launch(cfg)
        }
        .map_err(|e| GpuError::KernelLaunchFailed(format!("col_sum_sq_nonzeros: {e}")))?;
        Ok(())
    })?;

    dev.synchronize()?;
    let col_sum: Vec<f64> = dev.dtoh_copy(&d_col_sum)?;
    let col_sum_sq: Vec<f64> = dev.dtoh_copy(&d_col_sum_sq)?;

    // The finalize is host arithmetic and is shared with the two CPU HVG paths
    // (`scx_accel::hvg::cpu`, `scx_accel::csc::mean_var`) via `scx-sparse` —
    // which is where it has to live, since `scx-accel` depends on this crate and
    // not the reverse. Only the accumulation is on the device.
    // The GPU routes have no on-device input finiteness gate (the CPU routes use
    // `ensure_finite_hvg_data`). Check the accumulated moments instead: a
    // non-finite input value in column j necessarily leaves `col_sum[j]`
    // non-finite, so this is exactly as strong at O(n_vars) instead of O(nnz).
    // This file's two finalizes already used `if var < 0.0`, so Phase 7a did not
    // change their clamp — a NaN variance propagated here before it too. (The
    // `.max(0.0)` sites were the four *batched* finalizes; see
    // `scx_sparse::moments`' table.) The gate is therefore closing a pre-existing
    // hole rather than a regression, and it is here because these routes are
    // ungated, not because their clamp moved.
    if let Some(j) = scx_sparse::first_non_finite_column(&col_sum, &col_sum_sq) {
        return Err(GpuError::InvalidShard(format!(
            "gpu_streaming_mean_var: column {j} accumulated a non-finite moment \
             (sum = {}, sum_sq = {}) — the input contains NaN/Inf, or a finite \
             input overflowed. HVG statistics would be meaningless.",
            col_sum[j], col_sum_sq[j]
        )));
    }
    let m = scx_sparse::finalize_column_moments(&col_sum, &col_sum_sq, n_obs);
    m.warn_if_unstable("gpu_streaming_mean_var");
    let (means, variances) = (m.means, m.variances);

    Ok((means, variances))
}

/// GPU equivalent of `scx_accel::streaming_clip_square_sum`.
///
/// For each nonzero `x` at column `c`, accumulates `min(x, clip_val[c])` and
/// `min(x, clip_val[c])²` into per-column f64 buffers via atomicAdd.
///
/// Returns `(batch_count_sum, sq_batch_count_sum)` — both f64 vectors of
/// length `n_vars`.
pub fn gpu_streaming_clip_square_sum(
    dev: &GpuDevice,
    source: &(dyn ShardSource + Sync),
    clip_val: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
    let n_vars = source.n_vars();
    if clip_val.len() != n_vars {
        return Err(GpuError::ShapeMismatch {
            expected: format!("clip_val.len() == n_vars = {n_vars}"),
            got: format!("clip_val.len() = {}", clip_val.len()),
        });
    }

    let mut d_batch_sum = dev.alloc_zeros::<f64>(n_vars)?;
    let mut d_sq_batch_sum = dev.alloc_zeros::<f64>(n_vars)?;
    let d_clip: CudaSlice<f64> = dev.htod_copy(clip_val)?;

    let module = dev.load_module_cached(COLMAJOR_OPS_PTX)?;
    let func = module
        .load_function("col_clip_sq_nonzeros_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("col_clip_sq_nonzeros: {e}")))?;

    let mut src = BackedGpuMatrixSource::new(dev, source)?.with_validation(ValidationPolicy::new(
        ValidationChecks::FINITE,
        "highly_variable_genes",
    ));
    src.for_each_gpu_csr_shard(&mut |_idx, slot| {
        let view = slot.view();
        let nnz = view.data.len() as i64;
        if nnz == 0 {
            return Ok(());
        }
        let threads: u32 = 256;
        let cfg = flat_launch_1d(nnz as u64, threads)?;
        unsafe {
            dev.stream()
                .launch_builder(&func)
                .arg(&view.indices)
                .arg(&view.data)
                .arg(&nnz)
                .arg(&d_clip)
                .arg(&mut d_batch_sum)
                .arg(&mut d_sq_batch_sum)
                .launch(cfg)
        }
        .map_err(|e| GpuError::KernelLaunchFailed(format!("col_clip_sq_nonzeros: {e}")))?;
        Ok(())
    })?;

    dev.synchronize()?;
    let batch_sum: Vec<f64> = dev.dtoh_copy(&d_batch_sum)?;
    let sq_batch_sum: Vec<f64> = dev.dtoh_copy(&d_sq_batch_sum)?;
    Ok((batch_sum, sq_batch_sum))
}

/// CSC-reduce equivalent of [`gpu_streaming_mean_var`].
///
/// Reads the gene-major CSC sidecar instead of the row-major CSR shards.
/// Each gene's nonzeros are contiguous on a CSC column, so the kernel
/// assigns **one block per column**, reduces that column's nonzeros in
/// shared memory, and writes a single `Σ x` / `Σ x²` per gene — no
/// cross-column `atomicAdd`, eliminating the hot-gene contention of the
/// CSR atomic path. Single-batch only (rows are ignored); the host
/// finalisation (`mean = Σ / n`, Bessel-corrected variance) matches the
/// CSR path exactly.
///
/// Returns `(means, variances)`, both f64 vectors of length `n_vars`.
pub fn gpu_streaming_mean_var_csc(
    dev: &GpuDevice,
    source: &(dyn ColumnShardSource + Sync),
) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
    let n_vars = source.n_vars();
    let n_obs = source.n_obs();
    if n_obs == 0 || n_vars == 0 {
        return Ok((vec![0.0; n_vars], vec![0.0; n_vars]));
    }

    let mut d_col_sum = dev.alloc_zeros::<f64>(n_vars)?;
    let mut d_col_sum_sq = dev.alloc_zeros::<f64>(n_vars)?;

    let module = dev.load_module_cached(COLMAJOR_OPS_PTX)?;
    let func = module
        .load_function("csc_col_mean_sq_reduce_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("csc_col_mean_sq_reduce: {e}")))?;

    let n_vars_i32 = n_vars as i32;
    let mut gpu = RawGpuCscShardSource::new(dev, source)?.with_validation(ValidationPolicy::new(
        ValidationChecks::IN_RANGE,
        "highly_variable_genes",
    ));
    gpu.for_each_gpu_csc_shard_in_range(0..n_vars as u32, |_idx, view| {
        let n_cols = view.n_cols();
        if n_cols == 0 {
            return Ok(());
        }
        let n_cols_i32 = n_cols as i32;
        let col_start_i32 = view.col_start as i32;
        // One block per column: a flat `n_cols × 256` launch is exactly
        // `n_cols` blocks, with the grid-cap check for free.
        let cfg = flat_launch_1d(n_cols as u64 * 256, 256)?;
        unsafe {
            dev.stream()
                .launch_builder(&func)
                .arg(&view.col_indptr)
                .arg(&view.data)
                .arg(&n_cols_i32)
                .arg(&col_start_i32)
                .arg(&n_vars_i32)
                .arg(&mut d_col_sum)
                .arg(&mut d_col_sum_sq)
                .launch(cfg)
        }
        .map_err(|e| GpuError::KernelLaunchFailed(format!("csc_col_mean_sq_reduce: {e}")))?;
        Ok(())
    })?;

    dev.synchronize()?;
    let col_sum: Vec<f64> = dev.dtoh_copy(&d_col_sum)?;
    let col_sum_sq: Vec<f64> = dev.dtoh_copy(&d_col_sum_sq)?;

    // The finalize is host arithmetic and is shared with the two CPU HVG paths
    // (`scx_accel::hvg::cpu`, `scx_accel::csc::mean_var`) via `scx-sparse` —
    // which is where it has to live, since `scx-accel` depends on this crate and
    // not the reverse. Only the accumulation is on the device.
    // Same accumulated-moment gate as the CSR route above; see the comment there
    // for why checking the sums is equivalent to scanning the input values.
    if let Some(j) = scx_sparse::first_non_finite_column(&col_sum, &col_sum_sq) {
        return Err(GpuError::InvalidShard(format!(
            "gpu_streaming_mean_var_csc: column {j} accumulated a non-finite \
             moment (sum = {}, sum_sq = {}) — the input contains NaN/Inf, or a \
             finite input overflowed. HVG statistics would be meaningless.",
            col_sum[j], col_sum_sq[j]
        )));
    }
    let m = scx_sparse::finalize_column_moments(&col_sum, &col_sum_sq, n_obs);
    m.warn_if_unstable("gpu_streaming_mean_var_csc");
    let (means, variances) = (m.means, m.variances);
    Ok((means, variances))
}

/// CSC-reduce equivalent of [`gpu_streaming_clip_square_sum`].
///
/// One block per gene reduces `Σ min(x, clip_val[c])` and
/// `Σ min(x, clip_val[c])²` over the column's contiguous nonzeros, writing
/// a single value per accumulator (no `atomicAdd`). `clip_val.len()` must
/// equal `source.n_vars()`.
///
/// Returns `(clipped_sum, clipped_sum_sq)` — both f64 vectors of length
/// `n_vars`.
pub fn gpu_streaming_clip_square_sum_csc(
    dev: &GpuDevice,
    source: &(dyn ColumnShardSource + Sync),
    clip_val: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
    let n_vars = source.n_vars();
    if clip_val.len() != n_vars {
        return Err(GpuError::ShapeMismatch {
            expected: format!("clip_val.len() == n_vars = {n_vars}"),
            got: format!("clip_val.len() = {}", clip_val.len()),
        });
    }
    let n_obs = source.n_obs();
    if n_obs == 0 || n_vars == 0 {
        return Ok((vec![0.0; n_vars], vec![0.0; n_vars]));
    }

    let mut d_clipped_sum = dev.alloc_zeros::<f64>(n_vars)?;
    let mut d_clipped_sum_sq = dev.alloc_zeros::<f64>(n_vars)?;
    let d_clip: CudaSlice<f64> = dev.htod_copy(clip_val)?;

    let module = dev.load_module_cached(COLMAJOR_OPS_PTX)?;
    let func = module
        .load_function("csc_col_clip_sq_reduce_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("csc_col_clip_sq_reduce: {e}")))?;

    let n_vars_i32 = n_vars as i32;
    let mut gpu = RawGpuCscShardSource::new(dev, source)?.with_validation(ValidationPolicy::new(
        ValidationChecks::FINITE,
        "highly_variable_genes",
    ));
    gpu.for_each_gpu_csc_shard_in_range(0..n_vars as u32, |_idx, view| {
        let n_cols = view.n_cols();
        if n_cols == 0 {
            return Ok(());
        }
        let n_cols_i32 = n_cols as i32;
        let col_start_i32 = view.col_start as i32;
        // One block per column: a flat `n_cols × 256` launch is exactly
        // `n_cols` blocks, with the grid-cap check for free.
        let cfg = flat_launch_1d(n_cols as u64 * 256, 256)?;
        unsafe {
            dev.stream()
                .launch_builder(&func)
                .arg(&view.col_indptr)
                .arg(&view.data)
                .arg(&n_cols_i32)
                .arg(&col_start_i32)
                .arg(&n_vars_i32)
                .arg(&d_clip)
                .arg(&mut d_clipped_sum)
                .arg(&mut d_clipped_sum_sq)
                .launch(cfg)
        }
        .map_err(|e| GpuError::KernelLaunchFailed(format!("csc_col_clip_sq_reduce: {e}")))?;
        Ok(())
    })?;

    dev.synchronize()?;
    let clipped_sum: Vec<f64> = dev.dtoh_copy(&d_clipped_sum)?;
    let clipped_sum_sq: Vec<f64> = dev.dtoh_copy(&d_clipped_sum_sq)?;
    Ok((clipped_sum, clipped_sum_sq))
}

/// GPU equivalent of `scx_accel::streaming_mean_var_batched`.
///
/// Per-batch mean and variance from a single shard pass. `cell_batch` maps
/// every visible cell (in shard-iteration order — same convention as the CPU
/// function) to a batch id; `-1` skips the cell.
///
/// Returns `(col_sum_per_batch, col_sum_sq_per_batch, batch_counts)` so the
/// caller can compute means/variances and derive global stats without a second
/// pass. Per-batch vectors are length `n_vars`; the outer `Vec` has length
/// `n_batches`. The caller (the `scx_accel::streaming_mean_var_batched_with_device`
/// wrapper) finalises these into [`scx_accel::BatchedHvgStats`] with the same
/// Bessel-corrected formula as the CPU path.
pub fn gpu_streaming_mean_var_batched(
    dev: &GpuDevice,
    source: &(dyn ShardSource + Sync),
    cell_batch: &[i32],
    n_batches: usize,
) -> Result<(PerBatchColVec, PerBatchColVec, Vec<usize>), GpuError> {
    let n_vars = source.n_vars();
    let n_obs = source.n_obs();
    if cell_batch.len() != n_obs {
        return Err(GpuError::ShapeMismatch {
            expected: format!("cell_batch.len() == n_obs = {n_obs}"),
            got: format!("cell_batch.len() = {}", cell_batch.len()),
        });
    }

    let buffer_len = n_batches.saturating_mul(n_vars);
    if buffer_len == 0 || n_obs == 0 {
        return Ok((
            vec![vec![0.0; n_vars]; n_batches],
            vec![vec![0.0; n_vars]; n_batches],
            vec![0usize; n_batches],
        ));
    }

    let mut d_col_sum = dev.alloc_zeros::<f64>(buffer_len)?;
    let mut d_col_sum_sq = dev.alloc_zeros::<f64>(buffer_len)?;

    let module = dev.load_module_cached(COLMAJOR_OPS_PTX)?;
    let func = module
        .load_function("col_sum_sq_nonzeros_batched_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("col_sum_sq_nonzeros_batched: {e}")))?;

    // Host-side batch counts so the caller can finalise variances without a
    // second pass. We accumulate by inspecting `cell_batch` once per shard
    // window — cheaper than reducing the per-batch sum buffer on device.
    let mut batch_counts = vec![0usize; n_batches];
    let mut cell_offset: usize = 0;

    // Single H2D copy of the full cell→batch mapping. `cell_batch` is
    // small (~`n_obs × 4 B` bytes) and never changes during the pass —
    // copying it per shard was pure waste. Each shard now slices into
    // this device buffer instead.
    let d_cell_batch: CudaSlice<i32> = dev.htod_copy(cell_batch)?;

    let mut src = BackedGpuMatrixSource::new(dev, source)?.with_validation(ValidationPolicy::new(
        ValidationChecks::FINITE,
        "highly_variable_genes",
    ));
    src.for_each_gpu_csr_shard(&mut |_idx, slot| {
        let view = slot.view();
        let shard_n_rows = view.shape.0;
        if shard_n_rows == 0 {
            return Ok(());
        }
        let window = &cell_batch[cell_offset..cell_offset + shard_n_rows];
        for &b in window {
            if b >= 0 && (b as usize) < n_batches {
                batch_counts[b as usize] += 1;
            }
        }
        let d_row_to_batch = d_cell_batch.slice(cell_offset..cell_offset + shard_n_rows);

        let nnz_i64 = view.data.len() as i64;
        let n_rows_i32 = shard_n_rows as i32;
        let n_vars_i32 = n_vars as i32;
        let n_batches_i32 = n_batches as i32;

        // Skip empty shards (no nonzeros => no atomicAdd work).
        if nnz_i64 == 0 {
            cell_offset += shard_n_rows;
            return Ok(());
        }

        let threads: u32 = 256;
        let cfg = flat_launch_1d(shard_n_rows as u64, threads)?;
        unsafe {
            dev.stream()
                .launch_builder(&func)
                .arg(&view.indptr)
                .arg(&view.indices)
                .arg(&view.data)
                .arg(&d_row_to_batch)
                .arg(&n_rows_i32)
                .arg(&n_vars_i32)
                .arg(&n_batches_i32)
                .arg(&mut d_col_sum)
                .arg(&mut d_col_sum_sq)
                .launch(cfg)
        }
        .map_err(|e| GpuError::KernelLaunchFailed(format!("col_sum_sq_nonzeros_batched: {e}")))?;

        cell_offset += shard_n_rows;
        Ok(())
    })?;

    dev.synchronize()?;
    let flat_sum: Vec<f64> = dev.dtoh_copy(&d_col_sum)?;
    let flat_sum_sq: Vec<f64> = dev.dtoh_copy(&d_col_sum_sq)?;

    let mut col_sum_per_batch = Vec::with_capacity(n_batches);
    let mut col_sum_sq_per_batch = Vec::with_capacity(n_batches);
    for b in 0..n_batches {
        let start = b * n_vars;
        let end = start + n_vars;
        col_sum_per_batch.push(flat_sum[start..end].to_vec());
        col_sum_sq_per_batch.push(flat_sum_sq[start..end].to_vec());
    }
    Ok((col_sum_per_batch, col_sum_sq_per_batch, batch_counts))
}

/// GPU equivalent of `scx_accel::streaming_clip_square_sum_batched`.
///
/// For each nonzero `x` at column `c` in a row mapped to batch `b`, accumulates
/// `min(x, clip_vals[b][c])` and its square into per-batch f64 buffers.
///
/// `clip_vals` is `[n_batches][n_vars]` (outer is batch). Returns a `Vec` of
/// per-batch `(batch_count_sum, sq_batch_count_sum)` tuples in the same order.
pub fn gpu_streaming_clip_square_sum_batched(
    dev: &GpuDevice,
    source: &(dyn ShardSource + Sync),
    cell_batch: &[i32],
    n_batches: usize,
    clip_vals: &[Vec<f64>],
) -> Result<PerBatchClipSums, GpuError> {
    let n_vars = source.n_vars();
    let n_obs = source.n_obs();
    if cell_batch.len() != n_obs {
        return Err(GpuError::ShapeMismatch {
            expected: format!("cell_batch.len() == n_obs = {n_obs}"),
            got: format!("cell_batch.len() = {}", cell_batch.len()),
        });
    }
    if clip_vals.len() != n_batches {
        return Err(GpuError::ShapeMismatch {
            expected: format!("clip_vals.len() == n_batches = {n_batches}"),
            got: format!("clip_vals.len() = {}", clip_vals.len()),
        });
    }
    for (b, cv) in clip_vals.iter().enumerate() {
        if cv.len() != n_vars {
            return Err(GpuError::ShapeMismatch {
                expected: format!("clip_vals[{b}].len() == n_vars = {n_vars}"),
                got: format!("clip_vals[{b}].len() = {}", cv.len()),
            });
        }
    }

    let buffer_len = n_batches.saturating_mul(n_vars);
    if buffer_len == 0 || n_obs == 0 {
        return Ok(vec![(vec![0.0; n_vars], vec![0.0; n_vars]); n_batches]);
    }

    // Flatten clip_vals to one device buffer of length n_batches × n_vars.
    let mut flat_clip = Vec::with_capacity(buffer_len);
    for cv in clip_vals {
        flat_clip.extend_from_slice(cv);
    }
    let d_clip: CudaSlice<f64> = dev.htod_copy(&flat_clip)?;

    let mut d_batch_sum = dev.alloc_zeros::<f64>(buffer_len)?;
    let mut d_sq_batch_sum = dev.alloc_zeros::<f64>(buffer_len)?;

    let module = dev.load_module_cached(COLMAJOR_OPS_PTX)?;
    let func = module
        .load_function("col_clip_sq_nonzeros_batched_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("col_clip_sq_nonzeros_batched: {e}")))?;

    let mut cell_offset: usize = 0;
    // Single H2D copy of `cell_batch`; sliced per shard inside the loop.
    // See `gpu_streaming_mean_var_batched` for rationale.
    let d_cell_batch: CudaSlice<i32> = dev.htod_copy(cell_batch)?;

    let mut src = BackedGpuMatrixSource::new(dev, source)?.with_validation(ValidationPolicy::new(
        ValidationChecks::FINITE,
        "highly_variable_genes",
    ));
    src.for_each_gpu_csr_shard(&mut |_idx, slot| {
        let view = slot.view();
        let shard_n_rows = view.shape.0;
        if shard_n_rows == 0 {
            return Ok(());
        }
        let d_row_to_batch = d_cell_batch.slice(cell_offset..cell_offset + shard_n_rows);

        let nnz_i64 = view.data.len() as i64;
        let n_rows_i32 = shard_n_rows as i32;
        let n_vars_i32 = n_vars as i32;
        let n_batches_i32 = n_batches as i32;

        if nnz_i64 == 0 {
            cell_offset += shard_n_rows;
            return Ok(());
        }

        let threads: u32 = 256;
        let cfg = flat_launch_1d(shard_n_rows as u64, threads)?;
        unsafe {
            dev.stream()
                .launch_builder(&func)
                .arg(&view.indptr)
                .arg(&view.indices)
                .arg(&view.data)
                .arg(&d_row_to_batch)
                .arg(&n_rows_i32)
                .arg(&n_vars_i32)
                .arg(&n_batches_i32)
                .arg(&d_clip)
                .arg(&mut d_batch_sum)
                .arg(&mut d_sq_batch_sum)
                .launch(cfg)
        }
        .map_err(|e| GpuError::KernelLaunchFailed(format!("col_clip_sq_nonzeros_batched: {e}")))?;

        cell_offset += shard_n_rows;
        Ok(())
    })?;

    dev.synchronize()?;
    let flat_bs: Vec<f64> = dev.dtoh_copy(&d_batch_sum)?;
    let flat_sbs: Vec<f64> = dev.dtoh_copy(&d_sq_batch_sum)?;

    let mut out = Vec::with_capacity(n_batches);
    for b in 0..n_batches {
        let start = b * n_vars;
        let end = start + n_vars;
        out.push((flat_bs[start..end].to_vec(), flat_sbs[start..end].to_vec()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use scx_sparse::{ScxCsc, ScxCsr};

    struct InMemorySource {
        shards: Vec<ScxCsr>,
        n_obs: usize,
        n_vars: usize,
    }

    impl ShardSource for InMemorySource {
        fn n_shards(&self) -> usize {
            self.shards.len()
        }
        fn n_obs(&self) -> usize {
            self.n_obs
        }
        fn n_vars(&self) -> usize {
            self.n_vars
        }
        fn read_shard(&self, shard_idx: usize) -> scx_format_io::Result<ScxCsr> {
            Ok(self.shards[shard_idx].clone())
        }
    }

    fn random_pos_csr(n_rows: usize, n_cols: usize, density: f32, seed: u64) -> ScxCsr {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indptr: Vec<i64> = Vec::with_capacity(n_rows + 1);
        let mut indices: Vec<i32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        indptr.push(0);
        for _ in 0..n_rows {
            for c in 0..n_cols {
                if rng.gen_bool(density as f64) {
                    indices.push(c as i32);
                    data.push(rng.gen_range(0.5..20.5));
                }
            }
            indptr.push(indices.len() as i64);
        }
        ScxCsr::new_unchecked((n_rows, n_cols), indptr, indices, data)
    }

    fn split_into_shards(csr: &ScxCsr, n_shards: usize) -> Vec<ScxCsr> {
        let (n_rows, n_cols) = (csr.n_rows(), csr.n_cols());
        let rows_per = n_rows.div_ceil(n_shards);
        let mut out = Vec::new();
        let mut row_start = 0;
        while row_start < n_rows {
            let row_end = (row_start + rows_per).min(n_rows);
            let p0 = csr.indptr[row_start] as usize;
            let p1 = csr.indptr[row_end] as usize;
            let shard_indptr: Vec<i64> = csr.indptr[row_start..=row_end]
                .iter()
                .map(|&p| p - csr.indptr[row_start])
                .collect();
            let shard_indices = csr.indices[p0..p1].to_vec();
            let shard_data = csr.data[p0..p1].to_vec();
            out.push(ScxCsr::new_unchecked(
                (row_end - row_start, n_cols),
                shard_indptr,
                shard_indices,
                shard_data,
            ));
            row_start = row_end;
        }
        out
    }

    /// Exact per-column mean and Bessel-corrected variance of **integer-valued**
    /// input, computed in integer arithmetic.
    ///
    /// This is the third-party oracle the GPU arms are measured against, and it
    /// is deliberately *not* the shipped formula. It shares no line with
    /// [`scx_sparse::finalize_column_moments`]: the variance here is
    ///
    /// ```text
    ///     var = (n·Σx² − (Σx)²) / (n·(n−1))
    /// ```
    ///
    /// — a ratio of two **exact `i128` integers**, converted to `f64` once, so
    /// the answer is correctly rounded rather than the closed form
    /// `(Σx² − n·mean²)/(n−1)` re-derived a third time. The two private copies
    /// that used to live here (one for the CSR arm, one for the dense/CSC arm)
    /// *were* that re-derivation, down to a `.max(0.0)` clamp where production
    /// uses `if var < 0.0` — so a NaN variance was silently zeroed on the
    /// reference side and passed through on the production side, and the test
    /// compared the shared finalize against a copy of itself (review §8.17).
    ///
    /// # Why an integer fixture makes an `abs = 0` GPU bar legitimate
    ///
    /// [`gpu_streaming_mean_var`] accumulates via cross-block f64 `atomicAdd`
    /// and therefore sums in a nondeterministic order (finding ACC6, module
    /// header above). That would normally rule out an exact bar. It does not
    /// here: when every value and every partial sum is exactly representable in
    /// f64, addition is exact, and exact addition is associative — so the
    /// accumulated moments are bit-identical whatever order the blocks land in.
    /// Keep the fixture's magnitudes under 2⁵³ or the assertion below fires.
    ///
    /// # Panics
    ///
    /// If an intermediate exceeds 2⁵³, where `f64` stops representing integers
    /// exactly and the "exact" claim above quietly stops holding. A fixture
    /// that grows past this must shrink, not widen the tolerance.
    fn exact_moments(col_sum: &[i128], col_sum_sq: &[i128], n: usize) -> (Vec<f64>, Vec<f64>) {
        const EXACT_F64_MAX: i128 = 1i128 << 53;
        let ni = n as i128;
        let denom = ni * (ni - 1).max(1);
        assert!(
            denom < EXACT_F64_MAX,
            "n·(n−1) = {denom} exceeds 2^53; this oracle is only exact below it"
        );
        let mut means = Vec::with_capacity(col_sum.len());
        let mut vars = Vec::with_capacity(col_sum.len());
        for (&s, &sq) in col_sum.iter().zip(col_sum_sq.iter()) {
            assert!(
                s.checked_mul(s).is_some_and(|s2| s2 < EXACT_F64_MAX) && ni * sq < EXACT_F64_MAX,
                "column moments exceed 2^53 (Σx = {s}, Σx² = {sq}, n = {n}); \
                 shrink the fixture rather than loosening the bar"
            );
            means.push(s as f64 / n as f64);
            // One rounding, on a division of two exactly-represented integers.
            vars.push((ni * sq - s * s) as f64 / denom as f64);
        }
        (means, vars)
    }

    /// Exact integer moments of a CSR fixture. Panics if a value is not an
    /// integer — the caller's fixture, not the code under test, is wrong.
    fn integer_moments_csr(csr: &ScxCsr) -> (Vec<i128>, Vec<i128>) {
        let mut s = vec![0i128; csr.n_cols()];
        let mut sq = vec![0i128; csr.n_cols()];
        for (&c, &v) in csr.indices.iter().zip(csr.data.iter()) {
            let iv = v as i128;
            assert_eq!(iv as f32, v, "fixture value {v} is not an integer");
            s[c as usize] += iv;
            sq[c as usize] += iv * iv;
        }
        (s, sq)
    }

    /// Exact integer moments of a dense fixture, for the CSC arm.
    fn integer_moments_dense(dense: &[Vec<f64>], n_cols: usize) -> (Vec<i128>, Vec<i128>) {
        let mut s = vec![0i128; n_cols];
        let mut sq = vec![0i128; n_cols];
        for row in dense {
            for (c, &v) in row.iter().enumerate().take(n_cols) {
                let iv = v as i128;
                assert_eq!(iv as f64, v, "fixture value {v} is not an integer");
                s[c] += iv;
                sq[c] += iv * iv;
            }
        }
        (s, sq)
    }

    /// Variance bar for a well-conditioned fixture (`residual/Σx²` ≈ 1).
    /// Measured worst case 2.1e-16, ≈1 ULP; one decimal order of headroom.
    const VAR_REL_WELL_CONDITIONED: f64 = 1e-15;

    /// Variance bar for the f32-accumulator probe, whose fixture is
    /// deliberately less well conditioned (`residual/Σx²` = 1.95e-03) so that
    /// its column sums can clear 2²⁴. Measured f64 noise there reaches 4.4e-14;
    /// the f32-accumulator signal it must catch is 2.7e-07. 1e-10 sits ~3.4
    /// orders above the noise and ~3.4 below the signal — the separation is the
    /// point, not the round number.
    const VAR_REL_LARGE_MAGNITUDE: f64 = 1e-10;

    /// Compare an arm against the oracle. One routine for every call site, but
    /// the variance bar is a **parameter**, because it is a property of the
    /// fixture's conditioning rather than of the code under test.
    ///
    /// Means are always asserted **exactly**: `Σx / n` is a single division of
    /// an exactly-represented integer, so it neither cancels nor depends on
    /// summation order, whatever the magnitudes.
    ///
    /// Variances do not have one bar, and assuming they did cost a red hardware
    /// run. The closed form loses about `eps / (residual/Σx²)` relatively, so
    /// its accuracy is set by how near-constant the column is:
    ///
    /// | fixture | residual/Σx² | measured worst rel |
    /// |---|---|---|
    /// | [`integer_csr`], values `1..=20` | ~1 | 2.1e-16 (≈1 ULP) |
    /// | the f32 probe, values `60_000..=70_000` | 1.95e-03 | 4.4e-14 |
    ///
    /// [`VAR_REL_WELL_CONDITIONED`] was measured on the first and then applied
    /// to the second, which is three orders too tight for it — the probe failed
    /// on an H100 at 1.245e-14 against a 1e-15 bar. The kernel was right and the
    /// bar was wrong.
    fn assert_matches_exact(
        got_means: &[f64],
        got_vars: &[f64],
        want_means: &[f64],
        want_vars: &[f64],
        var_rel_bar: f64,
        what: &str,
    ) {
        assert_eq!(got_means.len(), want_means.len(), "{what}: mean length");
        assert_eq!(got_vars.len(), want_vars.len(), "{what}: var length");
        for j in 0..want_means.len() {
            assert_eq!(
                got_means[j], want_means[j],
                "{what}: mean col {j} is not exactly the integer oracle's value"
            );
            let d = (got_vars[j] - want_vars[j]).abs();
            let rel = if want_vars[j] == 0.0 {
                d
            } else {
                d / want_vars[j].abs()
            };
            assert!(
                rel < var_rel_bar,
                "{what}: var col {j} = {} vs exact {} (rel {rel:.3e} >= {var_rel_bar:.0e})",
                got_vars[j],
                want_vars[j]
            );
        }
    }

    /// Integer-valued CSR fixture: every value, every `Σx` and every `Σx²` is
    /// exactly representable in f64, which is what lets the GPU arms be held to
    /// an exact bar despite `atomicAdd` ordering. See [`exact_moments`].
    fn integer_csr(n_rows: usize, n_cols: usize, density: f32, seed: u64) -> ScxCsr {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indptr: Vec<i64> = Vec::with_capacity(n_rows + 1);
        let mut indices: Vec<i32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        indptr.push(0);
        for _ in 0..n_rows {
            for c in 0..n_cols {
                if rng.gen_bool(density as f64) {
                    indices.push(c as i32);
                    data.push(rng.gen_range(1..=20) as f32);
                }
            }
            indptr.push(indices.len() as i64);
        }
        ScxCsr::new_unchecked((n_rows, n_cols), indptr, indices, data)
    }

    /// Integer-valued dense fixture, for the CSC arm.
    fn integer_dense(n_rows: usize, n_cols: usize, density: f64, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut dense = vec![vec![0.0f64; n_cols]; n_rows];
        for row in dense.iter_mut() {
            for v in row.iter_mut() {
                if rng.gen_bool(density) {
                    *v = rng.gen_range(1..=20) as f64;
                }
            }
        }
        dense
    }
    fn cpu_clip_square_sum(csr: &ScxCsr, clip_val: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n_vars = csr.n_cols();
        let mut bcs = vec![0.0f64; n_vars];
        let mut sbcs = vec![0.0f64; n_vars];
        for (&c, &v) in csr.indices.iter().zip(csr.data.iter()) {
            let c = c as usize;
            let v = (v as f64).min(clip_val[c]);
            bcs[c] += v;
            sbcs[c] += v * v;
        }
        (bcs, sbcs)
    }

    /// The bar the GPU tests use, justified on the CPU where it can be watched
    /// red without a device.
    ///
    /// On a fixture whose moments are exactly representable, the shipped closed
    /// form and the exact-integer oracle agree to ~1 ULP. Measured worst case
    /// 2.1e-16, so the GPU bar of 1e-15 has one decimal order of headroom and
    /// is a statement about f64, not a guess. If this fails, the GPU tests'
    /// tolerance is wrong and no amount of GPU debugging will show it.
    #[test]
    fn the_exact_oracle_and_the_closed_form_agree_to_one_ulp() {
        let csr = integer_csr(500, 100, 0.1, 55);
        let (s, sq) = integer_moments_csr(&csr);
        let (want_means, want_vars) = exact_moments(&s, &sq, 500);

        // The moments a kernel would hand the shared finalize.
        let cs: Vec<f64> = s.iter().map(|&v| v as f64).collect();
        let csq: Vec<f64> = sq.iter().map(|&v| v as f64).collect();
        let m = scx_sparse::finalize_column_moments(&cs, &csq, 500);

        assert!(
            m.unstable.is_empty(),
            "a well-conditioned fixture must not report cancellation; got {:?}",
            m.unstable
        );
        assert_matches_exact(
            &m.means,
            &m.variances,
            &want_means,
            &want_vars,
            VAR_REL_WELL_CONDITIONED,
            "finalize_column_moments on exact integer moments",
        );
    }

    /// Where the closed form *does* lose, and what the old `rel < 1e-5` bar
    /// could not see.
    ///
    /// ⚠️ The shape matters and the review's own example has it wrong. §8.17
    /// proposes "a gene at constant 8192.0 in 1000 of 1 M cells". Measured,
    /// that column's mean is 8.192, so `n·mean² = 6.7e7` against `Sx² = 6.7e10`
    /// — the subtraction cancels nothing and the closed form is exact. A
    /// *sparse spike* has a tiny mean by construction; the sibling test below
    /// pins that, so this fixture cannot quietly drift into it.
    ///
    /// What cancels is a **dense, near-constant column at large magnitude**.
    /// Here every one of 5000 cells is 8192.0 except five at 8193.0:
    /// `residual / Sx² = 1.5e-11`, about eleven decimal digits gone.
    ///
    /// [`scx_sparse::finalize_column_moments`] **reports** that and does not
    /// repair it, so this asserts the documented behaviour rather than a wish.
    #[test]
    fn the_closed_form_reports_cancellation_on_a_dense_near_constant_column() {
        const N: usize = 5_000;
        // 4995 cells at 8192, five at 8193 — accumulated exactly, so the only
        // error under test is the finalize's own subtraction.
        let (a, b) = (8192i128, 8193i128);
        let s = 4_995 * a + 5 * b;
        let sq = 4_995 * a * a + 5 * b * b;
        let (want_means, want_vars) = exact_moments(&[s], &[sq], N);

        let m = scx_sparse::finalize_column_moments(&[s as f64], &[sq as f64], N);

        assert_eq!(
            m.unstable,
            vec![0u32],
            "a dense near-constant column at large magnitude must be reported \
             unstable — that report is the whole contract, since the closed \
             form does not repair the loss"
        );
        assert_eq!(m.means[0], want_means[0], "the mean does not cancel");

        let rel = (m.variances[0] - want_vars[0]).abs() / want_vars[0].abs();
        assert!(
            rel > 1e-7,
            "this fixture is supposed to lose precision; rel = {rel:.3e} means \
             it no longer does and the test has stopped testing cancellation"
        );
        assert!(
            rel < 1e-5,
            "the point of this test: the loss ({rel:.3e}) is INSIDE the \
             `rel < 1e-5` bar these GPU tests used to carry, so that bar could \
             not see an eleven-digit cancellation even on a fixture that has one"
        );
    }

    /// Premise for the test above: the review's sparse-spike shape does **not**
    /// cancel, so a fixture that drifts into it would pass vacuously.
    #[test]
    fn a_sparse_spike_column_does_not_cancel() {
        // 100 hits in 100k rows, not the review's 1000-in-1M: same mean
        // (8.192), same non-cancellation, and `n·Σx²` stays under 2⁵³ so the
        // oracle itself is exact. `exact_moments` refuses the larger shape
        // rather than silently rounding — which is how this size was chosen.
        const N: usize = 100_000;
        let (hits, v) = (100i128, 8_192i128);
        let (s, sq) = (hits * v, hits * v * v);
        let (want_means, want_vars) = exact_moments(&[s], &[sq], N);

        let m = scx_sparse::finalize_column_moments(&[s as f64], &[sq as f64], N);

        assert!(
            m.unstable.is_empty(),
            "a sparse spike at 8192 is not a cancellation fixture — the mean \
             is 8.192, so n*mean^2 is three orders below Sx^2 and nothing \
             cancels (review SS8.17's example)"
        );
        assert_eq!(m.means[0], want_means[0]);
        assert_eq!(
            m.variances[0], want_vars[0],
            "with nothing to cancel the closed form is exact"
        );
    }

    /// A fixture whose column sums clear 2²⁴, so a kernel that switched its
    /// accumulator from f64 back to f32 is detectable.
    ///
    /// The other integer fixtures here cannot see that regression, and it is
    /// worth being explicit about why: their values are `1..=20` over 400–500
    /// rows, so a column sum is ~500 and every partial sum is exact in **f32**
    /// as well as f64. Both accumulators would agree to the last bit and the
    /// test would pass. Raised in review by Cursor Agent as a residual risk.
    ///
    /// The magnitudes are measured, not chosen. Four properties have to hold at
    /// once, and most obvious fixtures fail one of them:
    ///
    /// - values f32-exact (integers ≤ 2²⁴), since `ScxCsr::data` is `f32`;
    /// - column sum **above** 2²⁴ ≈ 1.68e7, or an f32 accumulator stays exact
    ///   and this test proves nothing. Two candidate fixtures with wider value
    ///   ranges were rejected for exactly this — their sums came to 1.4e7 and
    ///   an f32 accumulator reproduced them with zero error;
    /// - value **spread** wide enough that the closed form stays
    ///   well-conditioned. A near-constant column at this magnitude gets
    ///   flagged `unstable` instead, which is a different test (see
    ///   `the_closed_form_reports_cancellation_on_a_dense_near_constant_column`);
    /// - `n·Σx²` and `(Σx)²` under 2⁵³, or [`exact_moments`] refuses the
    ///   fixture rather than silently rounding.
    ///
    /// 400 dense rows of `60_000..=70_000` satisfies all four: measured f32
    /// accumulation error 2.7e-07 — eight orders above this test's `1e-15` bar
    /// — at `residual/Σx² = 1.9e-3`, comfortably clear of the 1e-7
    /// cancellation threshold.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_streaming_mean_var_detects_an_f32_accumulator() {
        let dev = require_gpu!();
        let (n_rows, n_cols) = (400usize, 16usize);
        let mut rng = StdRng::seed_from_u64(7);
        let dense: Vec<Vec<f64>> = (0..n_rows)
            .map(|_| {
                (0..n_cols)
                    .map(|_| rng.gen_range(60_000..=70_000) as f64)
                    .collect()
            })
            .collect();
        let csr = dense_to_csr(&dense, n_cols);

        // Premise: the sums really do clear the f32 exact-integer boundary. A
        // fixture that drifts below it passes while testing nothing.
        let (s, sq) = integer_moments_dense(&dense, n_cols);
        assert!(
            s.iter().all(|&v| v > (1i128 << 24)),
            "every column sum must exceed 2^24 for an f32 accumulator to lose; \
             got min {:?}",
            s.iter().min()
        );

        let source = InMemorySource {
            shards: split_into_shards(&csr, 4),
            n_obs: n_rows,
            n_vars: n_cols,
        };
        let (gm, gv) = gpu_streaming_mean_var(&dev, &source).expect("gpu streaming mean/var");
        let (want_means, want_vars) = exact_moments(&s, &sq, n_rows);
        assert_matches_exact(
            &gm,
            &gv,
            &want_means,
            &want_vars,
            VAR_REL_LARGE_MAGNITUDE,
            "f32-accumulator probe",
        );
    }

    /// The GPU CSR mean/variance arm against the exact-integer oracle.
    ///
    /// Bars are `abs = 0` on means and `rel < 1e-15` on variances, not the
    /// `rel < 1e-5` this used to carry. The old bar was nine orders looser than
    /// the arms actually are, and — measured — it could not see an eleven-digit
    /// cancellation loss either (see
    /// `the_closed_form_reports_cancellation_on_a_dense_near_constant_column`).
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_streaming_mean_var_matches_exact() {
        let dev = require_gpu!();
        let n_rows = 500;
        let n_cols = 100;
        let csr = integer_csr(n_rows, n_cols, 0.1, 55);
        let shards = split_into_shards(&csr, 4);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let (gpu_means, gpu_vars) =
            gpu_streaming_mean_var(&dev, &source).expect("gpu streaming mean/var");
        let (s, sq) = integer_moments_csr(&csr);
        let (want_means, want_vars) = exact_moments(&s, &sq, n_rows);
        assert_matches_exact(
            &gpu_means,
            &gpu_vars,
            &want_means,
            &want_vars,
            VAR_REL_WELL_CONDITIONED,
            "gpu_streaming_mean_var",
        );
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_streaming_clip_square_sum_matches_cpu() {
        let dev = require_gpu!();
        let n_rows = 500;
        let n_cols = 100;
        let csr = random_pos_csr(n_rows, n_cols, 0.1, 66);
        let shards = split_into_shards(&csr, 4);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };
        // Deterministic clip threshold per column.
        let clip_val: Vec<f64> = (0..n_cols).map(|j| 1.0 + (j as f64) * 0.05).collect();

        let (gpu_bcs, gpu_sbcs) = gpu_streaming_clip_square_sum(&dev, &source, &clip_val)
            .expect("gpu streaming clip sum");
        let (cpu_bcs, cpu_sbcs) = cpu_clip_square_sum(&csr, &clip_val);

        for j in 0..n_cols {
            let d1 = (gpu_bcs[j] - cpu_bcs[j]).abs();
            let d2 = (gpu_sbcs[j] - cpu_sbcs[j]).abs();
            let den1 = cpu_bcs[j].abs().max(1e-10);
            let den2 = cpu_sbcs[j].abs().max(1e-10);
            assert!(
                d1 / den1 < 1e-5,
                "bcs col {j}: gpu={} cpu={}",
                gpu_bcs[j],
                cpu_bcs[j]
            );
            assert!(
                d2 / den2 < 1e-5,
                "sbcs col {j}: gpu={} cpu={}",
                gpu_sbcs[j],
                cpu_sbcs[j]
            );
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_streaming_mean_var_empty() {
        let dev = require_gpu!();
        let source = InMemorySource {
            shards: vec![ScxCsr::new_unchecked((0, 5), vec![0], vec![], vec![])],
            n_obs: 0,
            n_vars: 5,
        };
        let (means, vars) = gpu_streaming_mean_var(&dev, &source).unwrap();
        assert_eq!(means, vec![0.0; 5]);
        assert_eq!(vars, vec![0.0; 5]);
    }

    // CPU reference for batched per-(batch, gene) sum / sum_sq. Mirrors
    // `scx_accel::streaming_mean_var_batched`'s accumulation loop without the
    // means/variances finalisation, so the test can compare raw accumulators.
    fn cpu_batched_sums(
        csr: &ScxCsr,
        cell_batch: &[i32],
        n_batches: usize,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>) {
        let (n_rows, n_cols) = (csr.n_rows(), csr.n_cols());
        let mut sum = vec![vec![0.0f64; n_cols]; n_batches];
        let mut sum_sq = vec![vec![0.0f64; n_cols]; n_batches];
        let mut counts = vec![0usize; n_batches];
        for (row, &b) in cell_batch.iter().enumerate().take(n_rows) {
            if b < 0 || (b as usize) >= n_batches {
                continue;
            }
            let b = b as usize;
            counts[b] += 1;
            let start = csr.indptr[row] as usize;
            let end = csr.indptr[row + 1] as usize;
            for j in start..end {
                let c = csr.indices[j] as usize;
                let v = csr.data[j] as f64;
                sum[b][c] += v;
                sum_sq[b][c] += v * v;
            }
        }
        (sum, sum_sq, counts)
    }

    fn cpu_batched_clip_sums(
        csr: &ScxCsr,
        cell_batch: &[i32],
        n_batches: usize,
        clip_vals: &[Vec<f64>],
    ) -> Vec<(Vec<f64>, Vec<f64>)> {
        let (n_rows, n_cols) = (csr.n_rows(), csr.n_cols());
        let mut out: Vec<(Vec<f64>, Vec<f64>)> = (0..n_batches)
            .map(|_| (vec![0.0f64; n_cols], vec![0.0f64; n_cols]))
            .collect();
        for (row, &b) in cell_batch.iter().enumerate().take(n_rows) {
            if b < 0 || (b as usize) >= n_batches {
                continue;
            }
            let b = b as usize;
            let start = csr.indptr[row] as usize;
            let end = csr.indptr[row + 1] as usize;
            for j in start..end {
                let c = csr.indices[j] as usize;
                let v = (csr.data[j] as f64).min(clip_vals[b][c]);
                out[b].0[c] += v;
                out[b].1[c] += v * v;
            }
        }
        out
    }

    fn deterministic_cell_batch(n_rows: usize, n_batches: usize) -> Vec<i32> {
        // Deterministic round-robin with a few -1 (skipped) cells mixed in
        // so the kernel's `b < 0` guard is exercised.
        (0..n_rows)
            .map(|i| {
                if i % 17 == 5 {
                    -1
                } else {
                    (i % n_batches) as i32
                }
            })
            .collect()
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_streaming_mean_var_batched_matches_cpu() {
        let dev = require_gpu!();
        let n_rows = 500;
        let n_cols = 100;
        let n_batches = 4;
        let csr = random_pos_csr(n_rows, n_cols, 0.1, 77);
        let shards = split_into_shards(&csr, 4);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };
        let cell_batch = deterministic_cell_batch(n_rows, n_batches);

        let (gpu_sum, gpu_sum_sq, gpu_counts) =
            gpu_streaming_mean_var_batched(&dev, &source, &cell_batch, n_batches)
                .expect("gpu streaming mean/var batched");
        let (cpu_sum, cpu_sum_sq, cpu_counts) = cpu_batched_sums(&csr, &cell_batch, n_batches);

        assert_eq!(gpu_counts, cpu_counts, "batch_counts diverged");
        for b in 0..n_batches {
            for j in 0..n_cols {
                let denom_s = cpu_sum[b][j].abs().max(1e-10);
                let denom_sq = cpu_sum_sq[b][j].abs().max(1e-10);
                let ds = (gpu_sum[b][j] - cpu_sum[b][j]).abs();
                let dsq = (gpu_sum_sq[b][j] - cpu_sum_sq[b][j]).abs();
                assert!(
                    ds / denom_s < 1e-5,
                    "sum mismatch b={b} col={j}: gpu={} cpu={} rel={}",
                    gpu_sum[b][j],
                    cpu_sum[b][j],
                    ds / denom_s
                );
                assert!(
                    dsq / denom_sq < 1e-5,
                    "sum_sq mismatch b={b} col={j}: gpu={} cpu={} rel={}",
                    gpu_sum_sq[b][j],
                    cpu_sum_sq[b][j],
                    dsq / denom_sq
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_streaming_clip_square_sum_batched_matches_cpu() {
        let dev = require_gpu!();
        let n_rows = 500;
        let n_cols = 100;
        let n_batches = 3;
        let csr = random_pos_csr(n_rows, n_cols, 0.1, 88);
        let shards = split_into_shards(&csr, 4);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };
        let cell_batch = deterministic_cell_batch(n_rows, n_batches);
        // Per-batch clip thresholds vary across batches and columns so the
        // kernel must index clip_val_per_batch[b * n_vars + c] correctly.
        let clip_vals: Vec<Vec<f64>> = (0..n_batches)
            .map(|b| {
                (0..n_cols)
                    .map(|j| 0.5 + 0.1 * (b as f64) + 0.03 * (j as f64))
                    .collect()
            })
            .collect();

        let gpu_out = gpu_streaming_clip_square_sum_batched(
            &dev,
            &source,
            &cell_batch,
            n_batches,
            &clip_vals,
        )
        .expect("gpu streaming clip sum batched");
        let cpu_out = cpu_batched_clip_sums(&csr, &cell_batch, n_batches, &clip_vals);

        for b in 0..n_batches {
            for j in 0..n_cols {
                let d1 = (gpu_out[b].0[j] - cpu_out[b].0[j]).abs();
                let d2 = (gpu_out[b].1[j] - cpu_out[b].1[j]).abs();
                let den1 = cpu_out[b].0[j].abs().max(1e-10);
                let den2 = cpu_out[b].1[j].abs().max(1e-10);
                assert!(
                    d1 / den1 < 1e-5,
                    "bcs b={b} col={j}: gpu={} cpu={}",
                    gpu_out[b].0[j],
                    cpu_out[b].0[j]
                );
                assert!(
                    d2 / den2 < 1e-5,
                    "sbcs b={b} col={j}: gpu={} cpu={}",
                    gpu_out[b].1[j],
                    cpu_out[b].1[j]
                );
            }
        }
    }

    // ── CSC reduce tests (one-block-per-column, no atomics) ─────────────

    /// In-memory [`ColumnShardSource`]: one `ScxCsc` per shard, each covering a
    /// contiguous global column range. Mirrors the fixture in
    /// `gpu_csc_shard_source.rs` tests; `read_csc_columns` is unused because the
    /// HVG reduce path only calls `read_csc_shard` + `csc_shard_col_range`.
    struct InMemCscSource {
        shards: Vec<ScxCsc>,
        ranges: Vec<(u32, u32)>,
        n_obs: usize,
        n_vars: usize,
    }

    impl scx_format_io::ColumnShardSource for InMemCscSource {
        fn n_csc_shards(&self) -> usize {
            self.shards.len()
        }
        fn n_obs(&self) -> usize {
            self.n_obs
        }
        fn n_vars(&self) -> usize {
            self.n_vars
        }
        fn read_csc_shard(&self, i: usize) -> scx_format_io::Result<ScxCsc> {
            Ok(self.shards[i].clone())
        }
        fn read_csc_columns(&self, _r: std::ops::Range<u32>) -> scx_format_io::Result<ScxCsc> {
            unimplemented!("HVG CSC reduce only calls read_csc_shard")
        }
        fn csc_shard_col_range(&self, i: usize) -> Option<(u32, u32)> {
            self.ranges.get(i).copied()
        }
    }

    /// Dense `n_rows × n_cols` matrix (0.0 where absent) with positive values.
    fn build_dense(n_rows: usize, n_cols: usize, density: f64, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut dense = vec![vec![0.0f64; n_cols]; n_rows];
        for row in dense.iter_mut() {
            for v in row.iter_mut() {
                if rng.gen_bool(density) {
                    *v = rng.gen_range(0.5..20.5);
                }
            }
        }
        dense
    }

    /// Row-major CSR view of the same dense fixture, so one matrix can be fed
    /// to both the CSR and the CSC arm and the two compared bit for bit.
    fn dense_to_csr(dense: &[Vec<f64>], n_cols: usize) -> ScxCsr {
        let mut indptr = vec![0i64];
        let mut indices: Vec<i32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        for row in dense {
            for (c, &v) in row.iter().enumerate().take(n_cols) {
                if v != 0.0 {
                    indices.push(c as i32);
                    data.push(v as f32);
                }
            }
            indptr.push(indices.len() as i64);
        }
        ScxCsr::new_unchecked((dense.len(), n_cols), indptr, indices, data)
    }

    /// Column-shard the dense matrix into `n_shards` contiguous column ranges
    /// (CSC, global row ids), so shards after the first have a non-zero
    /// `col_start` — exercising the kernel's `col_start + local_col` mapping.
    fn dense_to_csc_shards(dense: &[Vec<f64>], n_cols: usize, n_shards: usize) -> InMemCscSource {
        let n_rows = dense.len();
        let cols_per = n_cols.div_ceil(n_shards.max(1));
        let mut shards = Vec::new();
        let mut ranges = Vec::new();
        let mut cs = 0usize;
        while cs < n_cols {
            let ce = (cs + cols_per).min(n_cols);
            let mut indptr = vec![0i64];
            let mut row_indices: Vec<i32> = Vec::new();
            let mut data: Vec<f32> = Vec::new();
            for c in cs..ce {
                for (r, drow) in dense.iter().enumerate() {
                    if drow[c] != 0.0 {
                        row_indices.push(r as i32);
                        data.push(drow[c] as f32);
                    }
                }
                indptr.push(row_indices.len() as i64);
            }
            shards.push(ScxCsc::new_unchecked(
                (n_rows, ce - cs),
                indptr,
                row_indices,
                data,
            ));
            ranges.push((cs as u32, ce as u32));
            cs = ce;
        }
        InMemCscSource {
            shards,
            ranges,
            n_obs: n_rows,
            n_vars: n_cols,
        }
    }

    fn cpu_clip_dense(dense: &[Vec<f64>], n_cols: usize, clip: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut s = vec![0.0; n_cols];
        let mut sq = vec![0.0; n_cols];
        for c in 0..n_cols {
            for drow in dense {
                if drow[c] != 0.0 {
                    let vc = drow[c].min(clip[c]);
                    s[c] += vc;
                    sq[c] += vc * vc;
                }
            }
        }
        (s, sq)
    }

    /// The GPU CSC reduce against the same exact-integer oracle, and against
    /// the CSR arm **bit for bit**.
    ///
    /// The cross-arm half is the one that would catch a staging refactor: the
    /// two paths share nothing but the finalize, so identical bits mean both
    /// accumulated the same values from the same shards. It is legitimate to
    /// demand identity here even though the CSR arm sums via `atomicAdd` in a
    /// nondeterministic order — on an integer fixture every partial sum is
    /// exact, and exact addition is associative.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_streaming_mean_var_csc_matches_exact() {
        let dev = require_gpu!();
        let (n_rows, n_cols) = (400usize, 90usize);
        let dense = integer_dense(n_rows, n_cols, 0.15, 123);
        let src = dense_to_csc_shards(&dense, n_cols, 4); // multi-shard
        let (gm, gv) = gpu_streaming_mean_var_csc(&dev, &src).expect("gpu csc mean/var");

        let (s, sq) = integer_moments_dense(&dense, n_cols);
        let (want_means, want_vars) = exact_moments(&s, &sq, n_rows);
        assert_matches_exact(
            &gm,
            &gv,
            &want_means,
            &want_vars,
            VAR_REL_WELL_CONDITIONED,
            "gpu_streaming_mean_var_csc",
        );

        // Same matrix through the CSR arm: bit-identical, not merely close.
        let csr = dense_to_csr(&dense, n_cols);
        let csr_src = InMemorySource {
            shards: split_into_shards(&csr, 4),
            n_obs: n_rows,
            n_vars: n_cols,
        };
        let (rm, rv) = gpu_streaming_mean_var(&dev, &csr_src).expect("gpu csr mean/var");
        assert_eq!(rm, gm, "CSR and CSC means must be bit-identical");
        assert_eq!(rv, gv, "CSR and CSC variances must be bit-identical");
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_streaming_clip_square_sum_csc_matches_cpu() {
        let dev = require_gpu!();
        let (n_rows, n_cols) = (400usize, 90usize);
        let dense = build_dense(n_rows, n_cols, 0.15, 321);
        let src = dense_to_csc_shards(&dense, n_cols, 4);
        let clip: Vec<f64> = (0..n_cols).map(|j| 1.0 + 0.05 * (j as f64)).collect();
        let (gs, gsq) = gpu_streaming_clip_square_sum_csc(&dev, &src, &clip).expect("gpu csc clip");
        let (cs, csq) = cpu_clip_dense(&dense, n_cols, &clip);
        for j in 0..n_cols {
            assert!(
                (gs[j] - cs[j]).abs() / cs[j].abs().max(1e-10) < 1e-5,
                "clip sum col {j}: gpu={} cpu={}",
                gs[j],
                cs[j]
            );
            assert!(
                (gsq[j] - csq[j]).abs() / csq[j].abs().max(1e-10) < 1e-5,
                "clip sumsq col {j}: gpu={} cpu={}",
                gsq[j],
                csq[j]
            );
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_streaming_mean_var_csc_empty() {
        let dev = require_gpu!();
        let src = InMemCscSource {
            shards: vec![ScxCsc::new_unchecked((0, 5), vec![0; 6], vec![], vec![])],
            ranges: vec![(0, 5)],
            n_obs: 0,
            n_vars: 5,
        };
        let (m, v) = gpu_streaming_mean_var_csc(&dev, &src).unwrap();
        assert_eq!(m, vec![0.0; 5]);
        assert_eq!(v, vec![0.0; 5]);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_streaming_csc_hot_gene_and_all_zero() {
        // A hot gene (col 0 nonzero in every row) and an all-zero gene (col 1
        // empty) in a single shard. Proves the one-block-per-column reduce
        // matches CPU with no atomic contention, and empty columns yield 0.
        let dev = require_gpu!();
        let n_rows = 1000usize;
        let n_cols = 4usize;
        let mut dense = vec![vec![0.0f64; n_cols]; n_rows];
        for (r, row) in dense.iter_mut().enumerate() {
            row[0] = (r % 7) as f64 + 1.0; // hot column, all nonzero
                                           // col 1 stays all-zero
            if r % 3 == 0 {
                row[2] = 2.0;
            }
            if r % 5 == 0 {
                row[3] = 5.0;
            }
        }
        let src = dense_to_csc_shards(&dense, n_cols, 1);
        let (gm, gv) = gpu_streaming_mean_var_csc(&dev, &src).unwrap();
        let (s, sq) = integer_moments_dense(&dense, n_cols);
        let (cm, cv) = exact_moments(&s, &sq, n_rows);
        assert_matches_exact(
            &gm,
            &gv,
            &cm,
            &cv,
            VAR_REL_WELL_CONDITIONED,
            "gpu_streaming_mean_var_csc hot gene",
        );
        assert_eq!(gm[1], 0.0, "all-zero gene mean");
        assert_eq!(gv[1], 0.0, "all-zero gene var");
    }
}
