//! Device-resident randomized-PCA power loop (V3 plan Task 2.5).
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
//!    that fixed descriptor — no per-iteration decode/upload
//!    ([`run_resident_power_loop`]).
//!
//! SpMM-segment CUDA-graph capture (an opt-in `cusparseSpMM` graph replay) was
//! removed: it poisoned the CUDA context on the cuSPARSE versions tested
//! (CUDA 12.x on H100) and was never a measured win over the direct resident
//! path, which already delivers the residency benefit.
//!
//! ## Math mode / SpMM algorithm
//!
//! The [`GpuPcaTuning`] math mode is applied to the shared cuBLAS handle by the
//! caller. The SpMM algorithm follows the `SpmmAlgPolicy` math policy.

use cudarc::cublas::sys as cbs;
use cudarc::cusparse::sys as csp;
use cudarc::driver::safe::CudaSlice;

use scx_format_io::ShardSource;

use crate::cublas::{gpu_sgemv, CublasHandle};
use crate::cusolver::{gpu_cholesky_qr2, gpu_qr_q, CusolverHandle, QrMethod};
use crate::cusparse::{
    spmm_csr_transpose_view_with_alg, spmm_csr_view_with_alg, CuSparseWorkspacePool,
    CusparseHandle, CusparseSpMatDescr, DnMatView, DnMatViewMut,
};
use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::gpu_pca::{
    gpu_column_sums_into, gpu_mean_correct_colmajor_strided, gpu_outer_sub, GpuPcaScratch,
};
use crate::math_policy::GpuPcaTuning;
use crate::shard_decode::GpuCsr;

/// VRAM headroom factor applied to the resident-CSR + scratch estimate before
/// comparing against free device memory.
const RESIDENT_VRAM_HEADROOM: f64 = 1.2;

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
/// Always returns `false` (no CUDA-graph replay): SpMM-segment capture was
/// removed. The `Result<bool, _>` shape is kept so callers can keep recording
/// `graph_replayed` without churn.
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
    // SpMM-segment CUDA-graph capture was removed: capturing `cusparseSpMM`
    // poisons the CUDA context on the cuSPARSE versions tested (CUDA 12.x on
    // H100) and was never a measured win. The residency benefit (single upload +
    // one SpMM per segment, no per-iteration re-decode) is delivered by the
    // direct loop below regardless.
    //
    // Resident scratch + cuSPARSE descriptor are built on the default stream.
    // The descriptor downcasts indptr i64→i32.
    let mut d_mc = dev.alloc_zeros::<f32>(k)?;
    let mut d_sum_q = dev.alloc_zeros::<f32>(k)?;
    let mut pool = CuSparseWorkspacePool::new();
    let desc = gpu_csr.to_cusparse_csr(dev, dev.stream())?;
    dev.synchronize()?;

    // Direct resident loop on the default stream — streams the CSR once, then
    // runs the power loop as plain cusparseSpMM calls honouring the SpMM policy.
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

#[cfg(test)]
mod tests {
    use crate::cusolver::QrMethod;
    use crate::gpu_graph::set_cuda_graphs_enabled_override;
    use crate::gpu_pca::gpu_randomized_pca;
    use crate::math_policy::GpuPcaTuning;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use scx_format_io::ShardSource;
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
        fn read_shard(&self, shard_idx: usize) -> scx_format_io::Result<ScxCsr> {
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
    /// are enabled or disabled. SpMM-segment capture was removed in Phase 3.4, so
    /// both runs take the direct resident path; this guards that toggling the
    /// graph kill switch does not perturb the resident PCA, and that the small
    /// (fits-VRAM) input routes through the resident path on both. Compared by
    /// |cosine| (sign-free), not bitwise.
    #[test]
    fn test_resident_capture_vs_direct_subspace() {
        let dev = require_gpu!();
        let (n_rows, n_cols, k) = (400usize, 60usize, 8usize);
        let source = random_source(n_rows, n_cols, 4, 17);
        let tuning = GpuPcaTuning::default();

        // Graphs enabled (PCA SpMM capture was removed in Phase 3.4 → direct path).
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
