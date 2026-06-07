//! Device-resident randomized-PCA power loop with optional CUDA-graph capture
//! (V3 plan Task 2.5).
//!
//! The streaming randomized-PCA core (`gpu_pca::randomized_pca_core`) drives the
//! power loop through [`crate::linear_operator::CenteredSparseOperator`], which
//! re-runs the host-orchestrated [`crate::gpu_shard_source::RawGpuShardSource`]
//! streaming pipeline on **every** `matmat`/`rmatmat` — re-reading, re-decoding
//! and re-uploading the entire matrix once per call (≈7 full passes for the
//! default 2 power iterations). For inputs whose full CSR fits device memory,
//! this module instead:
//!
//! 1. Drains the [`ShardSource`] **once** into a single device-resident
//!    [`GpuCsr`] + cuSPARSE descriptor ([`try_build_resident_csr`]).
//! 2. Runs the power loop as two plain `cusparseSpMM` calls per iteration on
//!    that fixed descriptor — no per-iteration decode/upload.
//! 3. Optionally **captures** each SpMM segment (forward / transpose) into a
//!    CUDA graph and replays it across power iterations
//!    ([`run_resident_power_loop`]), amortising kernel-launch latency.
//!
//! ## Why the scratch buffers must be pointer-stable for capture
//!
//! A captured graph bakes in the device pointers of its kernel arguments.
//! `cusolver::gpu_qr_q` uses a swap-and-return that hands back a **new**
//! `CudaSlice` (the input is left as a zero-length dummy), so naively threading
//! it through `scratch.d_y` would change that buffer's pointer every iteration
//! and invalidate the captured forward/transpose graphs. We therefore keep
//! `scratch.d_y` / `scratch.d_z` pointer-stable and run QR through
//! [`qr_into_stable`], which QRs a scratch copy and copies the `Q` factor back
//! into the original buffer.
//!
//! ## Stream discipline
//!
//! Each path runs entirely on a single stream: the direct path on the device's
//! default stream, the capture path on the capturable per-thread stream (via a
//! `dev.with_stream(...)` clone). Replayed graphs launch on their capture
//! stream, and QR runs on the same stream, so segments and QR serialise without
//! a cross-stream wait. The per-thread stream (unlike `new_stream()`) does not
//! flip cudarc into multi-stream mode, so default-stream-allocated scratch is
//! capture-safe (no auto-inserted `cuStreamWaitEvent`).
//!
//! ## Math mode / SpMM algorithm
//!
//! The [`GpuPcaTuning`] math mode is applied to the shared cuBLAS handle by the
//! caller. The SpMM algorithm follows the [`SpmmAlgPolicy`]; capture forces the
//! deterministic CSR algorithm (`CUSPARSE_SPMM_CSR_ALG2`) regardless, since the
//! heuristic default may use atomics / size its workspace per shape.

use std::sync::Arc;

use cudarc::cublas::sys as cbs;
use cudarc::cusparse::sys as csp;
use cudarc::driver::safe::{CudaGraph, CudaSlice, CudaStream};

use scx_format::ShardSource;

use crate::cublas::{gpu_sgemv, CublasHandle};
use crate::cusolver::{gpu_cholesky_qr2, gpu_qr_q, CusolverHandle, QrMethod};
use crate::cusparse::{
    spmm_csr_transpose_view_with_alg, spmm_csr_view_with_alg, CuSparseWorkspacePool,
    CusparseHandle, CusparseSpMatDescr, DnMatView, DnMatViewMut,
};
use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::gpu_graph::{capture_graph, cuda_graphs_enabled};
use crate::gpu_pca::{
    gpu_column_sums_into, gpu_mean_correct_colmajor_strided, gpu_outer_sub, GpuPcaScratch,
};
use crate::math_policy::{GpuPcaTuning, SpmmAlgPolicy};
use crate::shard_decode::GpuCsr;

/// VRAM headroom factor applied to the resident-CSR + scratch estimate before
/// comparing against free device memory.
const RESIDENT_VRAM_HEADROOM: f64 = 1.2;

/// Whether to attempt SpMM-segment CUDA-graph capture (Task 2.5). Off unless
/// `SCX_ENABLE_PCA_SPMM_CAPTURE` is `1`/`true` — capturing `cusparseSpMM` is not
/// safe on the cuSPARSE versions tested (it poisons the CUDA context, see
/// [`run_resident_power_loop`]). Opt-in so the path can be re-enabled when a
/// capture-safe cuSPARSE is available, without changing the default behaviour.
fn pca_spmm_capture_opt_in() -> bool {
    matches!(
        std::env::var("SCX_ENABLE_PCA_SPMM_CAPTURE").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

/// Drain `source` into a single device-resident CSR, or return `None` when the
/// resident matrix plus PCA dense scratch would not fit device memory (the
/// caller then falls back to the streaming power loop).
///
/// `k` is the oversampled rank used to size the dense scratch in the VRAM
/// pre-flight (`2·n_obs·k + n_vars·k` f32 for `d_y` / `d_z` + small vectors).
///
/// cuSPARSE SpMM tolerates unsorted column indices, so — unlike the GPU-DE
/// staging path — this builder does not reject unsorted/duplicate columns; it
/// concatenates shards verbatim.
pub(crate) fn try_build_resident_csr(
    dev: &GpuDevice,
    source: &(dyn ShardSource + Sync),
    k: usize,
) -> Result<Option<GpuCsr>, GpuError> {
    let (n_obs, n_vars) = source.shape();
    let n_shards = source.n_shards();
    if n_obs == 0 || n_vars == 0 || n_shards == 0 {
        return Ok(None);
    }

    // Preflight the nnz budget BEFORE accumulating any shard on the host, so a
    // matrix that will not fit device memory cannot first OOM host RAM
    // (finding 1). The fixed (nnz-independent) cost is the dense PCA scratch
    // (`d_y`/`d_z` + a transient transpose/U buffer + small vectors) plus the
    // resident indptr (i64 + the i32 copy the cuSPARSE descriptor owns); each
    // nnz then costs 8 bytes (i32 index + f32 value). Compute in u64 with
    // saturating arithmetic so pathological shapes can't overflow.
    let free_budget = {
        let (free, _total) = dev.free_memory()?;
        (free as f64 / RESIDENT_VRAM_HEADROOM) as u64
    };
    let nk = (n_obs as u64).saturating_mul(k as u64);
    let scratch_bytes = nk
        .saturating_mul(2)
        .saturating_add((n_vars as u64).saturating_mul(k as u64))
        .saturating_add((k as u64).saturating_mul(4))
        .saturating_mul(4);
    let indptr_bytes = (n_obs as u64 + 1).saturating_mul(8 + 4);
    let fixed_bytes = scratch_bytes.saturating_add(indptr_bytes);
    if fixed_bytes >= free_budget {
        return Ok(None);
    }
    // Max resident nnz allowed by VRAM, capped at i32::MAX: the cuSPARSE
    // descriptor downcasts the (concatenated) row offsets to i32
    // (`GpuCsr::to_cusparse_csr`), which the per-shard streaming path never
    // exceeded but the full concatenation can — so total nnz > i32::MAX must
    // hard-fall back to streaming (finding 2).
    let max_nnz = ((free_budget - fixed_bytes) / 8).min(i32::MAX as u64) as usize;
    if max_nnz == 0 {
        return Ok(None);
    }

    let mut indptr: Vec<i64> = Vec::with_capacity(n_obs + 1);
    indptr.push(0);
    let mut indices: Vec<i32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    let mut row_acc: i64 = 0;

    for s in 0..n_shards {
        let csr = source
            .read_shard(s)
            .map_err(|e| GpuError::InvalidShard(format!("resident PCA read shard {s}: {e}")))?;
        let nr = csr.n_rows();
        if nr == 0 {
            continue;
        }
        // Bail to the streaming fallback the moment the running nnz exceeds the
        // VRAM/i32 budget — this bounds the host accumulation to ~one device's
        // worth of CSR rather than materialising an over-budget matrix first.
        if indices.len() + csr.indices.len() > max_nnz {
            return Ok(None);
        }
        indices.extend_from_slice(&csr.indices);
        data.extend_from_slice(&csr.data);
        // Shard indptr starts at 0; offset each row pointer by the running
        // global nnz so the concatenated indptr is globally monotonic.
        for r in 0..nr {
            indptr.push(row_acc + csr.indptr[r + 1]);
        }
        row_acc += *csr.indptr.last().unwrap_or(&0);
    }

    // Empty / all-zero matrix → let the streaming path handle it.
    let nnz = indices.len();
    if indptr.len() != n_obs + 1 || nnz == 0 {
        return Ok(None);
    }

    let d_indptr = dev.htod_copy(&indptr)?;
    let d_indices = dev.htod_copy(&indices)?;
    let d_data = dev.htod_copy(&data)?;
    Ok(Some(GpuCsr {
        indptr: d_indptr,
        indices: d_indices,
        data: d_data,
        shape: (n_obs, n_vars),
    }))
}

/// Forward segment: `d_out (n_obs × k) = (X − μ)·d_v`, single SpMM on the
/// resident descriptor + mean correction. No device allocations (capture-safe);
/// `d_mc` (length `k`) is a caller-provided scratch reused across replays.
#[allow(clippy::too_many_arguments)]
fn matmat_resident(
    dev: &GpuDevice,
    cusparse: &CusparseHandle,
    cublas: &CublasHandle,
    desc: &CusparseSpMatDescr,
    d_means: Option<&CudaSlice<f32>>,
    d_v: &CudaSlice<f32>,
    d_out: &mut CudaSlice<f32>,
    d_mc: &mut CudaSlice<f32>,
    pool: &mut CuSparseWorkspacePool,
    n_obs: usize,
    n_vars: usize,
    k: usize,
    alg: csp::cusparseSpMMAlg_t,
) -> Result<(), GpuError> {
    dev.stream()
        .memset_zeros(d_out)
        .map_err(|e| GpuError::KernelLaunchFailed(format!("resident matmat zero: {e}")))?;

    if let Some(d_mu) = d_means {
        // mc = Vᵀ·μ (length k), recomputed each call from the current d_v.
        gpu_sgemv(
            cublas,
            dev.stream(),
            d_v,
            d_mu,
            d_mc,
            n_vars,
            k,
            1.0,
            0.0,
            cbs::cublasOperation_t::CUBLAS_OP_T,
        )?;
    }

    let b_view = DnMatView::contiguous(d_v, n_vars as i64, k as i64);
    let c_view = DnMatViewMut::contiguous(d_out, n_obs as i64, k as i64);
    spmm_csr_view_with_alg(
        cusparse,
        dev.stream(),
        dev,
        Some(pool),
        desc,
        b_view,
        c_view,
        1.0,
        0.0,
        alg,
    )?;

    if d_means.is_some() {
        gpu_mean_correct_colmajor_strided(dev, d_out, d_mc, n_obs, k, 0, n_obs)?;
    }
    Ok(())
}

/// Transpose segment: `d_out (n_vars × k) = (X − μ)ᵀ·d_y`, single transposed
/// SpMM on the resident descriptor + centering correction. No device
/// allocations (capture-safe); `d_sum_q` (length `k`) is reused across replays.
#[allow(clippy::too_many_arguments)]
fn rmatmat_resident(
    dev: &GpuDevice,
    cusparse: &CusparseHandle,
    desc: &CusparseSpMatDescr,
    d_means: Option<&CudaSlice<f32>>,
    d_y: &CudaSlice<f32>,
    d_out: &mut CudaSlice<f32>,
    d_sum_q: &mut CudaSlice<f32>,
    pool: &mut CuSparseWorkspacePool,
    n_obs: usize,
    n_vars: usize,
    k: usize,
    alg: csp::cusparseSpMMAlg_t,
) -> Result<(), GpuError> {
    dev.stream()
        .memset_zeros(d_out)
        .map_err(|e| GpuError::KernelLaunchFailed(format!("resident rmatmat zero: {e}")))?;

    let b_view = DnMatView::contiguous(d_y, n_obs as i64, k as i64);
    let c_view = DnMatViewMut::contiguous(d_out, n_vars as i64, k as i64);
    // Single SpMM over the whole matrix → β = 0 (overwrite), unlike the
    // streaming path that accumulates β = 1 across shards.
    spmm_csr_transpose_view_with_alg(
        cusparse,
        dev.stream(),
        dev,
        Some(pool),
        desc,
        b_view,
        c_view,
        1.0,
        0.0,
        alg,
    )?;

    if let Some(d_mu) = d_means {
        gpu_column_sums_into(dev, d_y, d_sum_q, n_obs, k)?;
        gpu_outer_sub(dev, d_out, d_mu, d_sum_q, n_vars, k)?;
    }
    Ok(())
}

/// QR that preserves the pointer of `stable` (capture-safe). `gpu_qr_q` /
/// `gpu_cholesky_qr2` swap-and-return a fresh buffer, so we QR a copy and copy
/// the `Q` factor back into `stable`, leaving its device pointer unchanged.
fn qr_into_stable(
    dev: &GpuDevice,
    cusolver: &CusolverHandle,
    cublas: &CublasHandle,
    qr_method: QrMethod,
    stable: &mut CudaSlice<f32>,
    rows: usize,
    cols: usize,
) -> Result<(), GpuError> {
    let mut qr_in = dev.alloc_zeros::<f32>(rows * cols)?;
    dev.stream()
        .memcpy_dtod(stable, &mut qr_in)
        .map_err(|e| GpuError::KernelLaunchFailed(format!("qr copy-in: {e}")))?;
    let q = match qr_method {
        QrMethod::Householder => gpu_qr_q(cusolver, dev.stream(), dev, &mut qr_in, rows, cols)?,
        QrMethod::Cholesky => gpu_cholesky_qr2(cublas, cusolver, dev, &mut qr_in, rows, cols)?,
    };
    dev.stream()
        .memcpy_dtod(&q, stable)
        .map_err(|e| GpuError::KernelLaunchFailed(format!("qr copy-back: {e}")))?;
    Ok(())
}

/// One non-captured power iteration on `active`: transpose → QR → forward → QR.
#[allow(clippy::too_many_arguments)]
fn power_iter_direct(
    active: &GpuDevice,
    cusparse: &CusparseHandle,
    cublas: &CublasHandle,
    cusolver: &CusolverHandle,
    desc: &CusparseSpMatDescr,
    d_means: Option<&CudaSlice<f32>>,
    scratch: &mut GpuPcaScratch,
    d_mc: &mut CudaSlice<f32>,
    d_sum_q: &mut CudaSlice<f32>,
    pool: &mut CuSparseWorkspacePool,
    qr_method: QrMethod,
    n_obs: usize,
    n_vars: usize,
    k: usize,
    alg: csp::cusparseSpMMAlg_t,
) -> Result<(), GpuError> {
    rmatmat_resident(
        active,
        cusparse,
        desc,
        d_means,
        &scratch.d_y,
        &mut scratch.d_z,
        d_sum_q,
        pool,
        n_obs,
        n_vars,
        k,
        alg,
    )?;
    qr_into_stable(
        active,
        cusolver,
        cublas,
        qr_method,
        &mut scratch.d_z,
        n_vars,
        k,
    )?;
    matmat_resident(
        active,
        cusparse,
        cublas,
        desc,
        d_means,
        &scratch.d_z,
        &mut scratch.d_y,
        d_mc,
        pool,
        n_obs,
        n_vars,
        k,
        alg,
    )?;
    qr_into_stable(
        active,
        cusolver,
        cublas,
        qr_method,
        &mut scratch.d_y,
        n_obs,
        k,
    )?;
    Ok(())
}

/// Run the randomized-PCA power loop on a device-resident CSR, filling
/// `scratch.d_y` with the final `Q` (`n_obs × k`, col-major) and `scratch.d_z`
/// with the final `B = (X − μ)ᵀ·Q` (`n_vars × k`, col-major) — exactly the two
/// buffers the streaming core leaves for the downstream SVD.
///
/// Returns `true` when a captured CUDA graph was replayed for at least one
/// segment, `false` when the direct (non-captured) resident path ran. Capture
/// is attempted only when [`cuda_graphs_enabled`] and `n_power_iterations ≥ 2`
/// (fewer iterations have nothing to replay); any capture failure falls back to
/// the direct resident path for the rest of the call.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_resident_power_loop(
    dev: &GpuDevice,
    gpu_csr: &GpuCsr,
    scratch: &mut GpuPcaScratch,
    d_omega: &CudaSlice<f32>,
    d_means: Option<&CudaSlice<f32>>,
    cusparse: &CusparseHandle,
    cublas: &CublasHandle,
    cusolver: &CusolverHandle,
    qr_method: QrMethod,
    n_obs: usize,
    n_vars: usize,
    k: usize,
    n_power_iterations: usize,
    tuning: GpuPcaTuning,
) -> Result<bool, GpuError> {
    // SpMM-segment CUDA-graph capture is **opt-in** (`SCX_ENABLE_PCA_SPMM_CAPTURE=1`)
    // and off by default. On the cuSPARSE versions tested (CUDA 12.x on H100),
    // capturing `cusparseSpMM` does not merely fail to replay — it poisons the
    // CUDA context with a sticky `CUDA_ERROR_INVALID_VALUE`, so even a direct
    // fallback in the same context then fails. There is no safe in-context
    // recovery once capture is attempted, so we gate the attempt entirely rather
    // than rely on a fallback. The residency win (single upload + one SpMM per
    // segment, no per-iteration re-decode) is delivered on the direct path
    // regardless; the capture machinery stays in place for a future cuSPARSE
    // that is capture-safe. The general `SCX_DISABLE_CUDA_GRAPHS` kill switch
    // still applies (UMAP / Harmony k-means capture custom kernels, which are
    // capture-safe).
    let want_capture =
        cuda_graphs_enabled() && n_power_iterations >= 2 && pca_spmm_capture_opt_in();

    // Resident scratch + cuSPARSE descriptor are built on the DEFAULT stream so
    // they remain valid for the direct path and (after `dev.synchronize()`) for
    // the per-thread capture stream. The descriptor downcasts indptr i64→i32.
    let mut d_mc = dev.alloc_zeros::<f32>(k)?;
    let mut d_sum_q = dev.alloc_zeros::<f32>(k)?;
    let mut pool = CuSparseWorkspacePool::new();
    let desc = gpu_csr.to_cusparse_csr(dev, dev.stream())?;
    dev.synchronize()?;

    if want_capture {
        // Best-effort capture on the per-thread (capturable) stream. `cusparseSpMM`
        // is not capturable on every cuSPARSE version — a captured-then-replayed
        // SpMM can corrupt the stream (the failure surfaces on the next sync). So
        // ANY error from the capture attempt (capture, replay, or sync) falls
        // back to the clean direct redo below: capture never compromises
        // correctness, it only ever fails to *accelerate*.
        let pts: Arc<CudaStream> = dev.context().per_thread_stream();
        let dev_pts = dev.with_stream(pts.clone());
        let captured = attempt_captured_loop(
            &dev_pts,
            &pts,
            cusparse,
            cublas,
            cusolver,
            &desc,
            d_means,
            scratch,
            d_omega,
            &mut d_mc,
            &mut d_sum_q,
            &mut pool,
            qr_method,
            n_obs,
            n_vars,
            k,
            n_power_iterations,
        );
        if captured.is_ok() {
            return Ok(true);
        }
        // Quiesce the device before the direct redo (the failed capture may have
        // left work queued / the per-thread stream in an error state).
        let _ = dev.synchronize();
    }

    // Direct resident loop on the default stream — a clean recompute from Ω,
    // honouring the requested SpMM policy. Also the sole path when capture is
    // disabled or `n_power_iterations < 2`.
    run_direct_resident_loop(
        dev,
        cusparse,
        cublas,
        cusolver,
        &desc,
        d_means,
        scratch,
        d_omega,
        &mut d_mc,
        &mut d_sum_q,
        &mut pool,
        qr_method,
        n_obs,
        n_vars,
        k,
        n_power_iterations,
        tuning.spmm_policy.to_alg(),
    )?;
    Ok(false)
}

/// Direct (non-captured) resident power loop on `active`, filling `scratch.d_y`
/// with the final `Q` and `scratch.d_z` with the final `B`. A clean recompute
/// from `d_omega`, so it is safe to call after a partial captured attempt.
#[allow(clippy::too_many_arguments)]
fn run_direct_resident_loop(
    active: &GpuDevice,
    cusparse: &CusparseHandle,
    cublas: &CublasHandle,
    cusolver: &CusolverHandle,
    desc: &CusparseSpMatDescr,
    d_means: Option<&CudaSlice<f32>>,
    scratch: &mut GpuPcaScratch,
    d_omega: &CudaSlice<f32>,
    d_mc: &mut CudaSlice<f32>,
    d_sum_q: &mut CudaSlice<f32>,
    pool: &mut CuSparseWorkspacePool,
    qr_method: QrMethod,
    n_obs: usize,
    n_vars: usize,
    k: usize,
    n_power_iterations: usize,
    alg: csp::cusparseSpMMAlg_t,
) -> Result<(), GpuError> {
    // Step 3-4: Y = (X − μ)·Ω; Q = qr(Y).
    matmat_resident(
        active,
        cusparse,
        cublas,
        desc,
        d_means,
        d_omega,
        &mut scratch.d_y,
        d_mc,
        pool,
        n_obs,
        n_vars,
        k,
        alg,
    )?;
    qr_into_stable(
        active,
        cusolver,
        cublas,
        qr_method,
        &mut scratch.d_y,
        n_obs,
        k,
    )?;
    // Step 5: power iterations.
    for _ in 0..n_power_iterations {
        power_iter_direct(
            active, cusparse, cublas, cusolver, desc, d_means, scratch, d_mc, d_sum_q, pool,
            qr_method, n_obs, n_vars, k, alg,
        )?;
    }
    // Step 6: final B = (X − μ)ᵀ·Q.
    rmatmat_resident(
        active,
        cusparse,
        desc,
        d_means,
        &scratch.d_y,
        &mut scratch.d_z,
        d_sum_q,
        pool,
        n_obs,
        n_vars,
        k,
        alg,
    )?;
    active.synchronize()?;
    Ok(())
}

/// Attempt the capture-accelerated power loop on the per-thread stream `dev_pts`
/// (forces the deterministic SpMM algorithm). Returns `Err` on any capture /
/// replay / sync failure so the caller can fall back to the direct path. On
/// success, `scratch.d_y` holds the final `Q` and `scratch.d_z` the final `B`.
#[allow(clippy::too_many_arguments)]
fn attempt_captured_loop(
    dev_pts: &GpuDevice,
    pts: &Arc<CudaStream>,
    cusparse: &CusparseHandle,
    cublas: &CublasHandle,
    cusolver: &CusolverHandle,
    desc: &CusparseSpMatDescr,
    d_means: Option<&CudaSlice<f32>>,
    scratch: &mut GpuPcaScratch,
    d_omega: &CudaSlice<f32>,
    d_mc: &mut CudaSlice<f32>,
    d_sum_q: &mut CudaSlice<f32>,
    pool: &mut CuSparseWorkspacePool,
    qr_method: QrMethod,
    n_obs: usize,
    n_vars: usize,
    k: usize,
    n_power_iterations: usize,
) -> Result<(), GpuError> {
    let alg = SpmmAlgPolicy::Deterministic.to_alg();

    // Initial Y = (X − μ)·Ω + QR (forward warm-up), then iteration 0 directly
    // (transpose warm-up) — all on the capture stream. The captured graphs then
    // bake the pointer-stable scratch (transpose: d_y→d_z, forward: d_z→d_y).
    matmat_resident(
        dev_pts,
        cusparse,
        cublas,
        desc,
        d_means,
        d_omega,
        &mut scratch.d_y,
        d_mc,
        pool,
        n_obs,
        n_vars,
        k,
        alg,
    )?;
    qr_into_stable(
        dev_pts,
        cusolver,
        cublas,
        qr_method,
        &mut scratch.d_y,
        n_obs,
        k,
    )?;
    power_iter_direct(
        dev_pts, cusparse, cublas, cusolver, desc, d_means, scratch, d_mc, d_sum_q, pool,
        qr_method, n_obs, n_vars, k, alg,
    )?;

    let (fwd_graph, tr_graph) = capture_resident_segments(
        dev_pts, pts, cusparse, cublas, desc, d_means, scratch, d_mc, d_sum_q, pool, n_obs, n_vars,
        k, alg,
    )?;

    for _ in 1..n_power_iterations {
        tr_graph
            .launch()
            .map_err(|e| GpuError::CudaError(format!("resident transpose replay: {e}")))?;
        qr_into_stable(
            dev_pts,
            cusolver,
            cublas,
            qr_method,
            &mut scratch.d_z,
            n_vars,
            k,
        )?;
        fwd_graph
            .launch()
            .map_err(|e| GpuError::CudaError(format!("resident forward replay: {e}")))?;
        qr_into_stable(
            dev_pts,
            cusolver,
            cublas,
            qr_method,
            &mut scratch.d_y,
            n_obs,
            k,
        )?;
    }
    // Step 6: final B = (X − μ)ᵀ·Q via the transpose graph replay.
    tr_graph
        .launch()
        .map_err(|e| GpuError::CudaError(format!("resident final transpose replay: {e}")))?;
    dev_pts.synchronize()?;
    Ok(())
}

/// Capture the forward and transpose SpMM segments as CUDA graphs on the
/// per-thread stream. Both segments read/write the pointer-stable
/// `scratch.d_y` / `scratch.d_z` (transpose: d_y→d_z, forward: d_z→d_y), so the
/// captured graphs stay valid across replays. Returns
/// `(forward_graph, transpose_graph)`.
#[allow(clippy::too_many_arguments)]
fn capture_resident_segments(
    active: &GpuDevice,
    pts: &Arc<CudaStream>,
    cusparse: &CusparseHandle,
    cublas: &CublasHandle,
    desc: &CusparseSpMatDescr,
    d_means: Option<&CudaSlice<f32>>,
    scratch: &mut GpuPcaScratch,
    d_mc: &mut CudaSlice<f32>,
    d_sum_q: &mut CudaSlice<f32>,
    pool: &mut CuSparseWorkspacePool,
    n_obs: usize,
    n_vars: usize,
    k: usize,
    alg: csp::cusparseSpMMAlg_t,
) -> Result<(CudaGraph, CudaGraph), GpuError> {
    // Transpose graph borrows d_y (read) + d_z (write).
    let tr_graph = {
        let GpuPcaScratch {
            ref d_y,
            ref mut d_z,
            ..
        } = *scratch;
        capture_graph(pts, |_s| {
            rmatmat_resident(
                active, cusparse, desc, d_means, d_y, d_z, d_sum_q, pool, n_obs, n_vars, k, alg,
            )
        })?
        .ok_or_else(|| GpuError::CudaError("resident transpose capture produced no graph".into()))?
    };

    // Forward graph borrows d_z (read) + d_y (write).
    let fwd_graph = {
        let GpuPcaScratch {
            ref mut d_y,
            ref d_z,
            ..
        } = *scratch;
        capture_graph(pts, |_s| {
            matmat_resident(
                active, cusparse, cublas, desc, d_means, d_z, d_y, d_mc, pool, n_obs, n_vars, k,
                alg,
            )
        })?
        .ok_or_else(|| GpuError::CudaError("resident forward capture produced no graph".into()))?
    };

    Ok((fwd_graph, tr_graph))
}

#[cfg(test)]
mod tests {
    use crate::cusolver::QrMethod;
    use crate::gpu_graph::set_cuda_graphs_enabled_override;
    use crate::gpu_pca::gpu_randomized_pca;
    use crate::math_policy::GpuPcaTuning;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use scx_format::ShardSource;
    use scx_sparse::ScxCsr;

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
        fn read_shard(&self, shard_idx: usize) -> scx_format::Result<ScxCsr> {
            Ok(self.shards[shard_idx].clone())
        }
    }

    /// Random CSR (sorted columns per row) split into `n_shards` row-shards.
    fn random_source(n_rows: usize, n_cols: usize, n_shards: usize, seed: u64) -> InMemorySource {
        let mut rng = StdRng::seed_from_u64(seed);
        let rows_per = n_rows.div_ceil(n_shards);
        let mut shards = Vec::new();
        let mut r0 = 0;
        while r0 < n_rows {
            let r1 = (r0 + rows_per).min(n_rows);
            let mut indptr = vec![0i64];
            let mut indices: Vec<i32> = Vec::new();
            let mut data: Vec<f32> = Vec::new();
            for _ in r0..r1 {
                for c in 0..n_cols {
                    if rng.gen_bool(0.15) {
                        indices.push(c as i32); // ascending c → sorted
                        data.push(rng.gen_range(0.0..5.0));
                    }
                }
                indptr.push(indices.len() as i64);
            }
            shards.push(ScxCsr::new_unchecked(
                (r1 - r0, n_cols),
                indptr,
                indices,
                data,
            ));
            r0 = r1;
        }
        InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        }
    }

    /// |cosine| between column `c` of two row-major (n_obs × k) embeddings.
    fn abs_cosine(a: &[f32], b: &[f32], n_obs: usize, k: usize, c: usize) -> f32 {
        let mut dot = 0f64;
        let mut na = 0f64;
        let mut nb = 0f64;
        for i in 0..n_obs {
            let x = a[i * k + c] as f64;
            let y = b[i * k + c] as f64;
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        (dot.abs() / (na.sqrt() * nb.sqrt()).max(1e-12)) as f32
    }

    /// Task 2.5: PCA results must be identical in subspace whether CUDA graphs
    /// are enabled or disabled. SpMM-segment capture is opt-in
    /// (`SCX_ENABLE_PCA_SPMM_CAPTURE`) and off here, so both runs take the direct
    /// resident path; this guards that toggling the graph kill switch does not
    /// perturb the resident PCA, and that the small (fits-VRAM) input routes
    /// through the resident path on both. Compared by |cosine| (sign-free), not
    /// bitwise.
    #[test]
    fn test_resident_capture_vs_direct_subspace() {
        let dev = require_gpu!();
        let (n_rows, n_cols, k) = (400usize, 60usize, 8usize);
        let source = random_source(n_rows, n_cols, 4, 17);
        let tuning = GpuPcaTuning::default();

        // Graphs enabled (but PCA SpMM capture is opt-in and off → direct path).
        set_cuda_graphs_enabled_override(Some(true));
        let captured = gpu_randomized_pca(
            &dev,
            &source,
            k,
            10,
            3,
            true,
            7,
            QrMethod::Householder,
            tuning,
        )
        .unwrap();

        // Capture off: direct resident path (never replays).
        set_cuda_graphs_enabled_override(Some(false));
        let direct = gpu_randomized_pca(
            &dev,
            &source,
            k,
            10,
            3,
            true,
            7,
            QrMethod::Householder,
            tuning,
        )
        .unwrap();
        assert!(!direct.graph_replayed, "graphs disabled → no replay");

        // Restore env-controlled behaviour for sibling tests.
        set_cuda_graphs_enabled_override(None);

        for c in 0..k {
            let cos = abs_cosine(&captured.embeddings, &direct.embeddings, n_rows, k, c);
            assert!(
                cos > 0.99,
                "PC {c}: |cosine| {cos} between captured and direct resident PCA"
            );
        }
    }
}
