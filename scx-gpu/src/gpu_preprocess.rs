//! GPU fused preprocessing (normalize + log1p + row_scale) on GPU-resident CSR.
//!
//! Provides GPU-accelerated versions of the CPU preprocessing operations
//! in `scx-engine/src/fused_ops.rs` and `scx-loader/src/normalize.rs`.
//!
//! The kernels operate in-place on [`GpuCsr`] data, modifying the `data`
//! array while leaving `indptr` and `indices` untouched. The `normalize`
//! (and fused `normalize+log1p`) row reduction is dispatched across three
//! granularity tiers selected by mean row density (see [`choose_row_tier`]):
//! one thread per row (sparse rows), one warp per row (warp-shuffle reduce),
//! or one block per row (shared-memory reduce, very dense rows). `log1p`
//! alone is elementwise and row-independent, so it runs one thread per
//! nonzero (grid-strided, `indptr`-free). `row_scale` (the GPU counterpart of
//! CPU `Transform::RowScale`) multiplies each row by an explicit per-row
//! factor on the thread-per-row kernel and is applied last, in the canonical
//! order `normalize → log1p → row_scale`.
//!
//! ## Usage
//!
//! ```rust,ignore
//! // Decode a shard on GPU
//! let mut gpu_csr = decode_shard_gpu(&dev, shard_bytes)?;
//!
//! // Apply fused normalize+log1p in-place
//! gpu_normalize_log1p(&dev, &gpu_csr.indptr, &mut gpu_csr.data, gpu_csr.shape.0, 1e4)?;
//! ```
//!
//! ## Numerical differences
//!
//! The GPU path uses f32 arithmetic throughout (row sums, scaling factor,
//! log1p). The CPU path in `scx-engine` accumulates row sums in f64 and
//! computes the scaling factor in f64 before casting back to f32. This
//! produces small rounding differences (~1e-6 relative error). For single-cell
//! RNA-seq data these differences are negligible.

use cudarc::driver::safe::{CudaSlice, CudaView, CudaViewMut};
use cudarc::driver::PushKernelArg;

use scx_format_io::ShardSource;
use scx_sparse::{concatenate_csr, ScxCsr};

use crate::device::{flat_launch_1d, GpuDevice};
use crate::error::GpuError;
use crate::gpu_matrix_source::GpuMatrixSource;
use crate::preprocessed_gpu_matrix_source::PreprocessedGpuMatrixSource;

/// PTX source for the normalize+log1p kernels, compiled at build time.
const NORMALIZE_LOG1P_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/normalize_log1p.ptx"));

/// Granularity for the `normalize` / `normalize+log1p` row reduction.
///
/// All three tiers compute the *same* per-row normalization — the choice is
/// purely a performance trade-off, so a coarse selector is correctness-safe.
/// `Thread` is the original one-thread-per-row kernel (best for very sparse
/// rows). `Warp` puts one warp on each row (32-wide shuffle reduction). `Block`
/// puts one block on each row (shared-memory reduction) for very dense rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowTier {
    Thread,
    Warp,
    Block,
}

/// Rows with mean nnz at or below this use one thread per row — a warp would
/// leave most of its 32 lanes idle.
const TIER_THREAD_MAX_MEAN_NNZ: usize = 8;
/// Rows with mean nnz at or above this use one block per row — a single warp's
/// 32 lanes would each loop too many times.
const TIER_BLOCK_MIN_MEAN_NNZ: usize = 512;

/// Pick a row-reduction tier from the *mean* row density (`nnz / n_rows`).
///
/// Uses the shard mean rather than per-row nnz: it needs no host `indptr` (only
/// `n_rows` and `data.len()`, both already in hand), and since the tier is
/// correctness-neutral the coarse mean is sufficient for the roughly uniform
/// per-cell nnz of scRNA-seq data. A future refinement could bucket rows
/// individually for shards with heavy intra-shard density skew.
pub(crate) fn choose_row_tier(n_rows: usize, nnz: usize) -> RowTier {
    if n_rows == 0 {
        return RowTier::Thread;
    }
    let mean = nnz / n_rows;
    if mean <= TIER_THREAD_MAX_MEAN_NNZ {
        RowTier::Thread
    } else if mean >= TIER_BLOCK_MIN_MEAN_NNZ {
        RowTier::Block
    } else {
        RowTier::Warp
    }
}

/// Dispatch fused operations on a GPU-resident CSR view (in-place).
///
/// Single source of truth for the `(normalize, log1p)` kernel dispatch,
/// followed by an orthogonal `row_scale` tail (applied last) — all the public
/// slice-based entry points below are thin wrappers, and
/// `GpuPreprocessedShardSource` calls this directly on the slot's
/// exact-sized views. Kernel launches go on `dev.stream()` (the compute
/// stream); callers that need a different stream must wrap accordingly.
///
/// The `normalize` reduction granularity is auto-selected from the shard's mean
/// row density via [`choose_row_tier`]; [`apply_fused_ops_with_tier`] takes an
/// explicit tier (used by tests to force a kernel on a small matrix).
pub(crate) fn apply_fused_ops_inner(
    dev: &GpuDevice,
    indptr: &CudaView<'_, i64>,
    data: &mut CudaViewMut<'_, f32>,
    n_rows: usize,
    normalize: Option<f32>,
    log1p: bool,
    row_scale: Option<(&CudaView<'_, f32>, usize)>,
) -> Result<(), GpuError> {
    let tier = choose_row_tier(n_rows, data.len());
    apply_fused_ops_with_tier(dev, indptr, data, n_rows, normalize, log1p, row_scale, tier)
}

/// [`apply_fused_ops_inner`] with an explicit normalize-reduction [`RowTier`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_fused_ops_with_tier(
    dev: &GpuDevice,
    indptr: &CudaView<'_, i64>,
    data: &mut CudaViewMut<'_, f32>,
    n_rows: usize,
    normalize: Option<f32>,
    log1p: bool,
    row_scale: Option<(&CudaView<'_, f32>, usize)>,
    tier: RowTier,
) -> Result<(), GpuError> {
    if n_rows == 0 {
        return Ok(());
    }

    let module = dev.load_module_cached(NORMALIZE_LOG1P_PTX)?;
    let n_rows_i32 = n_rows as i32;
    let nnz = data.len();
    let threads: u32 = 256;
    // Thread-per-row config — also used by the row_scale tail below.
    let cfg = flat_launch_1d(n_rows as u64, threads)?;

    match (normalize, log1p) {
        (Some(target_sum), do_log1p) => {
            // Select the kernel + launch geometry by tier. All three variants
            // share the same argument list (indptr, data, n_rows, target_sum).
            let (func_name, norm_cfg) = match tier {
                RowTier::Thread => (
                    if do_log1p {
                        "normalize_log1p_kernel"
                    } else {
                        "normalize_kernel"
                    },
                    cfg,
                ),
                RowTier::Warp => {
                    // One warp per row: 256 threads = 8 warps per block. Sized
                    // as a flat 32-threads-per-row launch — same block count,
                    // but counted in u64 and checked against the grid_dim.x cap.
                    (
                        if do_log1p {
                            "normalize_log1p_warp_kernel"
                        } else {
                            "normalize_warp_kernel"
                        },
                        flat_launch_1d(n_rows as u64 * 32, threads)?,
                    )
                }
                RowTier::Block => (
                    if do_log1p {
                        "normalize_log1p_block_kernel"
                    } else {
                        "normalize_block_kernel"
                    },
                    // One block per row: a flat `n_rows × threads` launch is
                    // exactly `n_rows` blocks, with the cap check for free.
                    flat_launch_1d(n_rows as u64 * threads as u64, threads)?,
                ),
            };
            let func = module
                .load_function(func_name)
                .map_err(|e| GpuError::KernelLaunchFailed(format!("{func_name}: {e}")))?;
            unsafe {
                dev.stream()
                    .launch_builder(&func)
                    .arg(indptr)
                    .arg(&mut *data)
                    .arg(&n_rows_i32)
                    .arg(&target_sum)
                    .launch(norm_cfg)
            }
            .map_err(|e| GpuError::KernelLaunchFailed(format!("{func_name}: {e}")))?;
        }
        (None, true) => {
            // log1p is elementwise and row-independent: one thread per nonzero,
            // grid-strided. No reduction, so the row tier doesn't apply.
            let nnz_i64 = nnz as i64;
            // `log1p_nnz_kernel` is 64-bit and grid-strided; only its launcher
            // ever truncated the nnz count. The `.max(1)` preserves the previous
            // behaviour for `nnz == 0` — one block that the grid-stride loop
            // exits immediately, rather than an empty (illegal) grid.
            let mut log1p_cfg = flat_launch_1d(nnz as u64, threads)?;
            log1p_cfg.grid_dim.0 = log1p_cfg.grid_dim.0.max(1);
            let func = module
                .load_function("log1p_nnz_kernel")
                .map_err(|e| GpuError::KernelLaunchFailed(format!("log1p_nnz_kernel: {e}")))?;
            unsafe {
                dev.stream()
                    .launch_builder(&func)
                    .arg(&mut *data)
                    .arg(&nnz_i64)
                    .launch(log1p_cfg)
            }
            .map_err(|e| GpuError::KernelLaunchFailed(format!("log1p_nnz_kernel: {e}")))?;
        }
        (None, false) => {}
    }

    // Row-scale runs LAST (canonical order normalize → log1p → row_scale),
    // after any normalize/log1p above have written `data`. The factor vector
    // is global (iteration row order); `row_offset` selects this shard's slice.
    if let Some((factors, row_offset)) = row_scale {
        // Self-contained bounds guard: the streaming source validates
        // `factors.len() == n_obs` at construction, but the direct entry
        // points (`gpu_row_scale` / `gpu_apply_fused_ops`) reach here with no
        // length check. `row_scale_kernel` does an unguarded device read of
        // `factors[row_offset + row]`, so a short slice would be an OOB read.
        if row_offset + n_rows > factors.len() {
            return Err(GpuError::InvalidShard(format!(
                "row_scale factors len {} < row_offset {row_offset} + n_rows {n_rows}",
                factors.len()
            )));
        }
        // `row_scale_kernel` takes `row_offset` as `long long`, so there is no
        // i32 cap to enforce here — the length guard above is the real bound.
        let row_offset_i64 = row_offset as i64;
        let func = module
            .load_function("row_scale_kernel")
            .map_err(|e| GpuError::KernelLaunchFailed(format!("row_scale_kernel: {e}")))?;
        unsafe {
            dev.stream()
                .launch_builder(&func)
                .arg(indptr)
                .arg(&mut *data)
                .arg(&n_rows_i32)
                .arg(&row_offset_i64)
                .arg(factors)
                .launch(cfg)
        }
        .map_err(|e| GpuError::KernelLaunchFailed(format!("row_scale_kernel: {e}")))?;
    }
    Ok(())
}

/// Fused normalize_total + log1p on a GPU-resident CSR matrix (in-place).
///
/// For each row: `data[i] = log1p(data[i] / row_sum * target_sum)`.
/// Equivalent to `scx_engine::fused_ops::fused_normalize_log1p` on GPU.
pub fn gpu_normalize_log1p(
    dev: &GpuDevice,
    indptr: &CudaSlice<i64>,
    data: &mut CudaSlice<f32>,
    n_rows: usize,
    target_sum: f32,
) -> Result<(), GpuError> {
    let indptr_view = indptr.slice(..);
    let mut data_view = data.slice_mut(..);
    apply_fused_ops_inner(
        dev,
        &indptr_view,
        &mut data_view,
        n_rows,
        Some(target_sum),
        true,
        None,
    )
}

/// Normalize-only on a GPU-resident CSR matrix (in-place).
pub fn gpu_normalize(
    dev: &GpuDevice,
    indptr: &CudaSlice<i64>,
    data: &mut CudaSlice<f32>,
    n_rows: usize,
    target_sum: f32,
) -> Result<(), GpuError> {
    let indptr_view = indptr.slice(..);
    let mut data_view = data.slice_mut(..);
    apply_fused_ops_inner(
        dev,
        &indptr_view,
        &mut data_view,
        n_rows,
        Some(target_sum),
        false,
        None,
    )
}

/// Log1p-only on a GPU-resident CSR matrix (in-place).
pub fn gpu_log1p(
    dev: &GpuDevice,
    indptr: &CudaSlice<i64>,
    data: &mut CudaSlice<f32>,
    n_rows: usize,
) -> Result<(), GpuError> {
    let indptr_view = indptr.slice(..);
    let mut data_view = data.slice_mut(..);
    apply_fused_ops_inner(dev, &indptr_view, &mut data_view, n_rows, None, true, None)
}

/// Per-row explicit-factor scaling on a GPU-resident CSR matrix (in-place):
/// `data[i] *= factors[row]`. `factors` must have length ≥ `n_rows` (indexed
/// from row 0). Mirrors the CPU `Transform::RowScale` per-row multiply.
pub fn gpu_row_scale(
    dev: &GpuDevice,
    indptr: &CudaSlice<i64>,
    data: &mut CudaSlice<f32>,
    n_rows: usize,
    factors: &CudaSlice<f32>,
) -> Result<(), GpuError> {
    let indptr_view = indptr.slice(..);
    let mut data_view = data.slice_mut(..);
    let factors_view = factors.slice(..);
    apply_fused_ops_inner(
        dev,
        &indptr_view,
        &mut data_view,
        n_rows,
        None,
        false,
        Some((&factors_view, 0)),
    )
}

/// Dispatch fused operations on a GPU-resident CSR matrix (in-place).
///
/// - `(Some(target_sum), true)` → fused normalize+log1p
/// - `(Some(target_sum), false)` → normalize only
/// - `(None, true)` → log1p only
/// - `(None, false)` → no-op
pub fn gpu_apply_fused_ops(
    dev: &GpuDevice,
    indptr: &CudaSlice<i64>,
    data: &mut CudaSlice<f32>,
    n_rows: usize,
    normalize: Option<f32>,
    log1p: bool,
    row_scale: Option<&CudaSlice<f32>>,
) -> Result<(), GpuError> {
    let indptr_view = indptr.slice(..);
    let mut data_view = data.slice_mut(..);
    let row_scale_view = row_scale.map(|s| s.slice(..));
    apply_fused_ops_inner(
        dev,
        &indptr_view,
        &mut data_view,
        n_rows,
        normalize,
        log1p,
        row_scale_view.as_ref().map(|v| (v, 0)),
    )
}

/// Stream a [`ShardSource`] through [`gpu_apply_fused_ops`] and return a
/// single concatenated [`ScxCsr`] on the host.
///
/// This is the eager GPU path used by `pyscx.accel.normalize_total(device="gpu")`
/// and `log1p(device="gpu")`. It runs the transforms on a
/// [`PreprocessedGpuMatrixSource`](crate::PreprocessedGpuMatrixSource)
/// (device-resident) and downloads each transformed shard's
/// `(indptr, indices, data)` triple at the end — a **terminal D→H copy**
/// sitting on top of the device-resident abstraction. New consumers that don't
/// need a host CSR should consume that source directly to avoid the download.
///
/// The implementation drops the legacy per-shard `dev.synchronize()`:
/// `memcpy_dtoh` for pageable host destinations already blocks on the
/// compute stream, and a single `dev.synchronize()` at the API boundary
/// provides the final defence-in-depth barrier.
///
/// # Arguments
///
/// * `dev` — GPU device handle.
/// * `source` — shard-wise CSR source; must be `Sync` (required by the loader).
/// * `normalize` — `Some(target_sum)` to apply per-row normalization; `None` to
///   skip normalization.
/// * `log1p` — apply `log1p` elementwise after any normalization.
/// * `row_scale` — optional per-row factor vector (in the source's iteration
///   row order, length `n_obs`); applied last (`data *= factor`), after any
///   normalize/log1p.
///
/// When `normalize`, `log1p`, and `row_scale` are all `None`/`false`, each
/// shard is simply downloaded and concatenated (no kernel launched). Callers
/// can avoid that overhead by not calling this function for the no-op case.
pub fn gpu_preprocess_to_csr(
    dev: &GpuDevice,
    source: &(dyn ShardSource + Sync),
    normalize: Option<f32>,
    log1p: bool,
    row_scale: Option<&[f32]>,
) -> Result<ScxCsr, GpuError> {
    let n_vars = source.n_vars();
    let n_shards = source.n_shards();

    // Empty-source fast path: return a (0 × n_vars) CSR without spinning up
    // the streaming loader.
    if n_shards == 0 || source.n_obs() == 0 {
        return Ok(ScxCsr::new_unchecked((0, n_vars), vec![0], vec![], vec![]));
    }

    // `for_each_gpu_shard` invokes the callback strictly sequentially on
    // the calling thread (worker thread only decodes), and
    // `RawGpuShardSource::run` iterates `0..n_shards` in order — so a plain
    // `Vec::push` here yields the shards in the correct order without
    // needing a Mutex or post-hoc sort.
    let mut shard_csrs = Vec::<ScxCsr>::with_capacity(n_shards);

    // `PreprocessedGpuMatrixSource` rather than `GpuPreprocessedShardSource`
    // directly: it is the `GpuMatrixSource` face of the same thing, and giving
    // it this consumer is what lets `GpuShardSource` be demoted to
    // `pub(crate)`. Until now it was a wrapper with no non-test users at all.
    let mut gpu_source =
        PreprocessedGpuMatrixSource::new(dev, source, normalize, log1p, row_scale)?;
    gpu_source.for_each_gpu_csr_shard(&mut |_shard_idx, slot| {
        let view = slot.view();
        let n_rows = view.shape.0;

        // D→H per shard. `memcpy_dtoh` to a pageable Vec implicitly
        // synchronises with the compute stream via the driver's staging
        // logic (it blocks the host thread until the copy completes),
        // so the per-shard `dev.synchronize()` of the legacy
        // implementation is no longer required here.
        let mut indptr = vec![0i64; n_rows + 1];
        let mut indices = vec![0i32; view.nnz()];
        let mut data = vec![0.0f32; view.nnz()];
        dev.stream()
            .memcpy_dtoh(&view.indptr, &mut indptr)
            .map_err(|e| GpuError::CudaError(format!("dtoh indptr: {e}")))?;
        dev.stream()
            .memcpy_dtoh(&view.indices, &mut indices)
            .map_err(|e| GpuError::CudaError(format!("dtoh indices: {e}")))?;
        dev.stream()
            .memcpy_dtoh(&view.data, &mut data)
            .map_err(|e| GpuError::CudaError(format!("dtoh data: {e}")))?;

        shard_csrs.push(ScxCsr::new_unchecked(
            (n_rows, n_vars),
            indptr,
            indices,
            data,
        ));
        Ok(())
    })?;

    // Boundary sync — host-returning API contract.
    dev.synchronize()?;

    concatenate_csr(&shard_csrs, n_vars)
        .map_err(|e| GpuError::InvalidShard(format!("concatenate_csr: {e}")))
}

#[cfg(test)]
#[path = "gpu_preprocess_tests.rs"]
mod tests;
