//! GPU-accelerated randomized PCA pipeline.
//!
//! Provides the mean-correction CUDA kernel and the full GPU PCA pipeline
//! that streams shards from any [`ShardSource`], performing SpMM on GPU via
//! cuSPARSE, QR via cuSOLVER, and the final SVD via CPU `faer`.
//!
//! ## Pipeline
//!
//! 1. Column means via streaming shard decode (CPU — means are small)
//! 2. Ω = `random_gaussian_gpu(n_vars, k)` (cuRAND on GPU)
//! 3. Y = streaming GPU SpMM with mean correction (cuSPARSE + CUDA kernel)
//! 4. Q = `gpu_qr_q(Y)` (cuSOLVER)
//! 5. Power iteration: B = X^T @ Q, Q_B = qr(B), Y = X @ Q_B, Q = qr(Y)
//! 6. B = X^T @ Q (final projection)
//! 7. SVD of B on CPU via `faer` (small matrix, f64 for accuracy)
//! 8. Embeddings = Q @ V × Σ (GPU dense matmul or CPU)
//!
//! Peak GPU memory: ~500 MB for 1M cells (Y, Q matrices + 1 decoded shard).

use cudarc::cublas::sys as cbs;
use cudarc::driver::safe::CudaSlice;
use cudarc::driver::safe::LaunchConfig;
use cudarc::driver::PushKernelArg;
use faer::Mat;

use scx_format_io::ShardSource;
use scx_sparse::total_variance_from_col_sq;

use crate::cublas::{gpu_sgemm, gpu_transpose_f32, CublasHandle};
use crate::curand::random_gaussian_gpu;
use crate::cusolver::{gpu_cholesky_qr2, gpu_qr_q, CusolverHandle, QrMethod};
use crate::cusparse::{CuSparseWorkspacePool, CusparseHandle};
use crate::device::{flat_launch_1d, GpuDevice};
use crate::device_resident::DeviceEmbedding;
use crate::error::GpuError;
use crate::linear_operator::CenteredSparseOperator;
use crate::math_policy::GpuPcaTuning;

/// PTX source for col-major scatter/gather/mean-correct/column-sum kernels.
const COLMAJOR_OPS_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/colmajor_ops.ptx"));

/// Reusable per-PCA-run device scratch buffers.
///
/// Hoists the `(n_obs × k)` forward-SpMM output `d_y` and the `(n_vars × k)`
/// transpose-SpMM output `d_z` out of the power-iteration inner loop. The
/// pre-G2 implementation allocated both fresh inside
/// `streaming_gpu_spmm_forward` / `_transpose` on every call (~6 × per power
/// iter × ~30 iters); this scratch lets `gpu_randomized_pca` allocate once
/// and reuse across the whole run.
///
/// Sized for a known `k` at construction time.
pub struct GpuPcaScratch {
    /// `(n_obs × k)` col-major — receives forward SpMM output and
    /// is then consumed in-place by `gpu_qr_q` / `gpu_cholesky_qr2`.
    pub d_y: CudaSlice<f32>,
    /// `(n_vars × k)` col-major — receives transpose SpMM output.
    pub d_z: CudaSlice<f32>,
}

impl GpuPcaScratch {
    /// Allocate scratch sized for `n_obs × k` (forward) and `n_vars × k`
    /// (transpose).
    pub fn new(dev: &GpuDevice, n_obs: usize, n_vars: usize, k: usize) -> Result<Self, GpuError> {
        let d_y = dev.alloc_zeros::<f32>(n_obs * k)?;
        let d_z = dev.alloc_zeros::<f32>(n_vars * k)?;
        Ok(Self { d_y, d_z })
    }
}

/// Result of GPU-accelerated randomized PCA.
pub struct GpuPcaResult {
    /// Cell embeddings: row-major `(n_obs × n_components)` on host.
    pub embeddings: Vec<f32>,
    /// Principal components (loadings): row-major `(n_components × n_vars)` on host.
    pub components: Vec<f32>,
    /// Variance explained by each component (f64 for precision).
    pub variance_explained: Vec<f64>,
    /// Ratio of variance explained (each / total).
    pub variance_ratio: Vec<f64>,
    /// Column means used for centering (None if `zero_center=false`).
    pub mean: Option<Vec<f64>>,
    /// Number of components.
    pub n_components: usize,
    /// Number of observations.
    pub n_obs: usize,
    /// Number of variables.
    pub n_vars: usize,
    /// Whether a captured CUDA graph was replayed in the power loop (Task 2.5):
    /// Whether the whole matrix was held **device-resident** across the power
    /// loop (`true`) or the streaming operator re-decoded and re-uploaded it on
    /// every multiply (`false`). Decided dynamically against free VRAM, so the
    /// same input can go either way run to run — which is why it is recorded.
    pub resident_csr: bool,
}

/// Result of GPU randomized PCA with the embedding kept **device-resident**
/// (V3 plan Phase 2.3).
///
/// Identical to [`GpuPcaResult`] except `embeddings` is a [`DeviceEmbedding`]
/// (row-major `(n_obs × n_components)` on the GPU) instead of a host `Vec<f32>`.
/// The small loadings/variance arrays stay on the host (they are already
/// computed there via the CPU SVD of `B`). The fused PCA → kNN path feeds this
/// embedding straight into CAGRA without a GPU→host→GPU round-trip.
pub struct GpuPcaDeviceResult {
    /// Cell embeddings, device-resident, row-major `(n_obs × n_components)`.
    pub embeddings: DeviceEmbedding,
    /// Principal components (loadings): row-major `(n_components × n_vars)` on host.
    pub components: Vec<f32>,
    /// Variance explained by each component (f64 for precision).
    pub variance_explained: Vec<f64>,
    /// Ratio of variance explained (each / total).
    pub variance_ratio: Vec<f64>,
    /// Column means used for centering (None if `zero_center=false`).
    pub mean: Option<Vec<f64>>,
    /// Number of components.
    pub n_components: usize,
    /// Number of observations.
    pub n_obs: usize,
    /// Number of variables.
    pub n_vars: usize,
    /// Whether the whole matrix was held device-resident across the power loop.
    /// See [`GpuPcaResult::resident_csr`].
    pub resident_csr: bool,
}

/// Internal output of the shared randomized-PCA core: the scaled embedding kept
/// **col-major** on the device (`scratch`-independent owned buffer) plus the
/// host-side loadings/variance. The two public entry points differ only in how
/// they finalize `d_u`: [`gpu_randomized_pca`] transposes it to a host
/// row-major `Vec<f32>`; [`gpu_randomized_pca_device`] transposes it to a
/// row-major [`DeviceEmbedding`] on the GPU.
struct RandomizedPcaCore {
    /// Embedding `U·Σ`, col-major `(n_obs × n_components)`, owned device buffer.
    d_u: CudaSlice<f32>,
    components: Vec<f32>,
    variance_explained: Vec<f64>,
    variance_ratio: Vec<f64>,
    mean: Option<Vec<f64>>,
    n_components: usize,
    n_obs: usize,
    n_vars: usize,
    /// Whether a captured CUDA graph was replayed in the power loop (Task 2.5).
    /// `true` only on the device-resident capture path; `false` for the
    /// streaming or direct-resident paths.
    /// Whether the whole matrix was held device-resident across the power loop.
    /// See [`GpuPcaResult::resident_csr`].
    resident_csr: bool,
}

/// Shared randomized-PCA core: runs the full streaming SpMM / QR / SVD pipeline
/// and returns the scaled embedding `U·Σ` **col-major on the device** plus the
/// host-side loadings/variance. Both public entry points
/// ([`gpu_randomized_pca`], [`gpu_randomized_pca_device`]) wrap this and differ
/// only in how they finalize `d_u`.
///
/// # Algorithm (matching scx-accel CPU version)
///
/// 1. Column means via streaming shard decode + CPU accumulation
/// 2. Ω = random_gaussian_gpu(n_vars, k) on GPU
/// 3. Y = streaming_gpu_spmm_forward(X, Ω) with mean correction
/// 4. Q = gpu_qr_q(Y)
/// 5. Power iteration: B = X^T @ Q, Q_B = qr(B), Y = X @ Q_B, Q = qr(Y)
/// 6. B = X^T @ Q (streaming GPU SpMM transpose)
/// 7. SVD of B (small matrix, CPU faer in f64) → Û, Σ, V^T
/// 8. Embeddings = Q @ V × Σ (GPU cuBLAS sgemm + column scaling) — kept on GPU
///
/// Steps 3-6 stream from any `ShardSource` without materializing full X.
/// The `Sync` bound is required so the streaming `CenteredSparseOperator`
/// (which iterates via `RawGpuShardSource` on the G3 staging path) can borrow
/// `source` across its scoped pre-decode worker thread.
/// Peak GPU memory: ~500 MB for 1M cells (dominated by Y and Q matrices).
#[allow(clippy::too_many_arguments)]
fn randomized_pca_core(
    dev: &GpuDevice,
    source: &(dyn ShardSource + Sync),
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    seed: u64,
    qr_method: QrMethod,
    tuning: GpuPcaTuning,
) -> Result<RandomizedPcaCore, GpuError> {
    let (n_obs, n_vars) = source.shape();

    // Validate inputs
    if n_components == 0 || n_obs == 0 || n_vars == 0 {
        return Err(GpuError::ShapeMismatch {
            expected: "n_components > 0, n_obs > 0, n_vars > 0".into(),
            got: format!("n_components={n_components}, n_obs={n_obs}, n_vars={n_vars}"),
        });
    }
    if n_components > n_obs.min(n_vars) {
        return Err(GpuError::ShapeMismatch {
            expected: format!("n_components <= min(n_obs, n_vars) = {}", n_obs.min(n_vars)),
            got: format!("n_components = {n_components}"),
        });
    }

    let k = (n_components + n_oversamples).min(n_vars).min(n_obs);

    // Pre-flight GPU memory check: estimate peak usage and compare to free memory.
    // Peak = Y(n_obs*k) + Q(n_obs*k) + Z(n_vars*k) + shard_buf + means, all f32.
    {
        let max_shard_rows = source.max_shard_rows().map_err(format_scx_error)?.max(1);
        let peak_bytes = (2 * n_obs * k + n_vars * k + max_shard_rows * k + n_vars) * 4;
        let peak_with_headroom = (peak_bytes as f64 * 1.1) as usize;
        let (free, _total) = dev.free_memory()?;
        if peak_with_headroom > free {
            return Err(GpuError::OutOfMemory(format!(
                "GPU PCA requires ~{} MB but only {} MB free on device",
                peak_with_headroom / (1 << 20),
                free / (1 << 20)
            )));
        }
    }

    // Create handles
    let cusparse_handle = CusparseHandle::new()?;
    let cusolver_handle = CusolverHandle::new()?;
    let cublas_handle = CublasHandle::new()?;
    // Task 2.5: apply the requested math mode to every cuBLAS GEMM/GEMV/GER in
    // this PCA run (sticky on the handle). StrictFp32 (default) preserves the
    // pre-2.5 fp32 numerics; AllowTf32 trades mantissa precision for speed.
    cublas_handle.set_math_mode(tuning.math_mode)?;

    // Step 1: Compute column means and sum-of-squares (CPU-side, 1 pass)
    let (means, col_sum_sq) = source
        .col_means_and_sum_sq(zero_center)
        .map_err(format_scx_error)?;

    // Upload means to GPU for mean correction (if centering)
    let d_means: Option<CudaSlice<f32>> = means
        .as_ref()
        .map(|mu| {
            let mu_f32: Vec<f32> = mu.iter().map(|&v| v as f32).collect();
            dev.htod_copy(&mu_f32)
        })
        .transpose()?;

    // Step 2: Generate random Gaussian Ω on GPU (n_vars × k, col-major)
    let d_omega = random_gaussian_gpu(dev, dev.stream(), n_vars, k, seed)?;

    // G2: hoist the (n_obs × k) and (n_vars × k) dense SpMM outputs into a
    // single reusable scratch struct + share one cuSPARSE workspace across
    // every SpMM call in the entire PCA run. Pre-G2 each call inside
    // streaming_gpu_spmm_forward / _transpose allocated d_y / d_z / d_y_shard
    // / d_sum_q fresh; the strided matmat_pooled / rmatmat_pooled path on
    // CenteredSparseOperator now writes directly into scratch.d_y / scratch.d_z
    // with cuSPARSE workspace served by `pool`.
    let mut scratch = GpuPcaScratch::new(dev, n_obs, n_vars, k)?;
    let mut pool = CuSparseWorkspacePool::new();
    // `tuning.spmm_policy` reaches the streaming operator here, not just the
    // resident loop below: both power-loop paths must run the algorithm the
    // caller asked for, because `uns["scx_accel"]["pca"]["spmm_policy"]` is
    // stamped from that request either way.
    let op = CenteredSparseOperator::new(
        dev,
        &cusparse_handle,
        &cublas_handle,
        source,
        d_means.as_ref(),
        tuning.spmm_policy,
    );

    // Task 2.5: when the full matrix fits device memory, run the power loop on
    // a single device-resident CSR (one upload, two SpMM/iter, optional CUDA-
    // graph capture) instead of the streaming operator that re-decodes and
    // re-uploads the whole matrix on every matmat/rmatmat. Falls back to the
    // streaming path (below) when it won't fit VRAM. `try_build_resident_csr`
    // drains `source` once; the streaming `op` is only used on the fallback.
    //
    // A device failure while *building* the resident CSR also falls back rather
    // than failing the PCA. The builder's own pre-flight declines cleanly on a
    // matrix it can see is too big, but its three closing `htod_copy` calls can
    // still fail if free VRAM moved underneath it — and the streaming power
    // loop in the `else` arm computes the same answer without them. Residency
    // is an optimisation; it must not be the reason the call errors. A
    // malformed shard still propagates (`is_runtime_failure` = false): the
    // streaming operator reads the same shards and would only re-derive it.
    // Re-driving `source` is safe — the builder only calls `read_shard(s)` by
    // index and keeps no state in it.
    //
    // No pool trim on the decline, unlike the DE residency path: that one trims
    // because the caller's very next act is to size its gene chunk against free
    // VRAM, and untrimmed pool memory would shrink it to pay for buffers
    // nothing holds. Here the free-VRAM pre-flight is already behind us and the
    // streaming operator allocates out of the same pool, so trimming would
    // hand the driver back memory we are about to ask for again.
    //
    // Ensure the means upload + Ω generation (issued on the default stream
    // above) are complete before the resident loop, which may run on the
    // per-thread capture stream (no auto cross-stream sync there).
    dev.synchronize()?;
    let resident = crate::error::decline_on_runtime_failure(
        crate::gpu_pca_resident::try_build_resident_csr(dev, source, k),
        "GPU PCA resident CSR",
    )?;
    let resident_csr = resident.is_some();
    if let Some(gpu_csr) = resident {
        crate::gpu_pca_resident::run_resident_power_loop(
            dev,
            &gpu_csr,
            &mut scratch,
            &d_omega,
            d_means.as_ref(),
            &cusparse_handle,
            &cublas_handle,
            &cusolver_handle,
            qr_method,
            n_obs,
            n_vars,
            k,
            n_power_iterations,
            tuning,
        )?
    } else {
        // Streaming fallback: the matrix does not fit device memory. The
        // pre-2.5 streaming operator re-decodes + re-uploads the whole matrix
        // on each matmat / rmatmat. No CUDA-graph capture on this path.
        op.matmat_pooled(&d_omega, &mut scratch.d_y, k, &mut pool)?;

        // QR dispatch — Householder (default) or CholeskyQR2 (Phase 4 opt-in).
        // Both backends use `gpu_qr_q`'s swap-and-return pattern: the input
        // buffer is left as a zero-length dummy and Q is returned in a new
        // CudaSlice. To keep `scratch.d_y` / `scratch.d_z` alive across the
        // power loop (avoiding a per-iter re-allocation of the n_obs × k
        // forward output and n_vars × k transpose output), this closure
        // swaps the returned Q back into the scratch field and returns the
        // empty dummy for drop. The caller then reads Q from the scratch
        // slot. Net cost: two `std::mem::swap`s per QR call (no allocations).
        let qr_into =
            |scratch_slot: &mut CudaSlice<f32>, rows: usize, cols: usize| -> Result<(), GpuError> {
                let mut q = match qr_method {
                    QrMethod::Householder => gpu_qr_q(
                        &cusolver_handle,
                        dev.stream(),
                        dev,
                        scratch_slot,
                        rows,
                        cols,
                    )?,
                    QrMethod::Cholesky => gpu_cholesky_qr2(
                        &cublas_handle,
                        &cusolver_handle,
                        dev,
                        scratch_slot,
                        rows,
                        cols,
                    )?,
                };
                // After gpu_qr_q's internal swap, `scratch_slot` holds the empty
                // dummy and `q` owns the n_obs * k buffer of Q. Swap so the
                // scratch field reclaims the Q buffer.
                debug_assert_eq!(
                    scratch_slot.len(),
                    0,
                    "QR backend must leave its input as a zero-length dummy via \
                 std::mem::swap; see cusolver::gpu_qr_q / gpu_cholesky_qr2 \
                 for the contract",
                );
                std::mem::swap(scratch_slot, &mut q);
                // `q` (now the empty dummy) drops here.
                Ok(())
            };

        // Step 4: scratch.d_y ← qr(scratch.d_y); Q now lives in scratch.d_y.
        qr_into(&mut scratch.d_y, n_obs, k)?;

        // Step 5: Power iterations. Each iter does:
        //   B = (X − μ)ᵀ · Q   →  scratch.d_z   (Q lives in scratch.d_y)
        //   Q_B = qr(scratch.d_z)                (Q_B now in scratch.d_z)
        //   Y = (X − μ) · Q_B  →  scratch.d_y   (overwrites Q)
        //   Q = qr(scratch.d_y)                  (new Q in scratch.d_y)
        for _ in 0..n_power_iterations {
            op.rmatmat_pooled(&scratch.d_y, &mut scratch.d_z, k, &mut pool)?;
            qr_into(&mut scratch.d_z, n_vars, k)?;

            op.matmat_pooled(&scratch.d_z, &mut scratch.d_y, k, &mut pool)?;
            qr_into(&mut scratch.d_y, n_obs, k)?;
        }

        // Step 6: B = (X − μ)ᵀ · Q  (final, n_vars × k) — written into scratch.d_z
        op.rmatmat_pooled(&scratch.d_y, &mut scratch.d_z, k, &mut pool)?;
    }
    let d_b_final = &scratch.d_z;

    // Step 7: Download B to host, SVD via faer (f64 for accuracy)
    dev.synchronize()?;
    let b_host_f32 = dev.dtoh_copy(d_b_final)?;

    // Convert B to f64 faer::Mat (col-major → Mat is also col-major, perfect)
    let mut b_mat = Mat::<f64>::zeros(n_vars, k);
    for j in 0..k {
        for i in 0..n_vars {
            b_mat[(i, j)] = b_host_f32[j * n_vars + i] as f64;
        }
    }

    let svd = b_mat
        .thin_svd()
        .map_err(|e| GpuError::CuSolverError(format!("CPU SVD failed: {e:?}")))?;

    let u_hat = svd.U().to_owned();
    let s_col = svd.S().column_vector();
    let sigma: Vec<f64> = (0..k.min(n_vars)).map(|i| s_col[i]).collect();
    let v = svd.V().to_owned();

    // Step 8: Embeddings = Q @ V × Σ — kept GPU-resident (Phase 3.2–3.4).
    //
    // Q is col-major (n_obs × k) on GPU. V is (k × k) on host from faer; we
    // slice V[:, 0..n_components] into a flat Vec<f32> (col-major, length
    // k × n_components) and upload once. Similarly upload sigma[0..n_components].
    // Compute `U = Q @ V` via cuBLAS sgemm on GPU, then broadcast-scale
    // columns by σ via `gpu_scale_columns`. Single D→H copy of the final
    // (n_obs × n_components) embedding replaces the previous triple-loop
    // over the full downloaded Q (≈240 MB at 1M × 60).
    let eff_k = k.min(sigma.len());
    let mut v_slice_f32: Vec<f32> = Vec::with_capacity(eff_k * n_components);
    for pc in 0..n_components {
        for j in 0..eff_k {
            v_slice_f32.push(v[(j, pc)] as f32);
        }
    }
    let d_v_top = dev.htod_copy(&v_slice_f32)?;
    let sigma_f32: Vec<f32> = sigma.iter().take(n_components).map(|&s| s as f32).collect();
    let d_sigma = dev.htod_copy(&sigma_f32)?;

    let mut d_u = dev.alloc_zeros::<f32>(n_obs * n_components)?;
    // U = Q @ V  →  sgemm with A=Q (n_obs × k col-major), B=V (k × n_components
    // col-major), C=U (n_obs × n_components col-major). Inner dim = eff_k.
    // Q is in `scratch.d_y` (the last `qr_into` after Step 6 wrote it there;
    // Step 6 then ran `rmatmat_pooled` reading from `scratch.d_y` into
    // `scratch.d_z`, leaving `scratch.d_y` untouched).
    gpu_sgemm(
        &cublas_handle,
        dev.stream(),
        &scratch.d_y,
        &d_v_top,
        &mut d_u,
        n_obs,
        n_components,
        eff_k,
        1.0,
        0.0,
        cbs::cublasOperation_t::CUBLAS_OP_N,
        cbs::cublasOperation_t::CUBLAS_OP_N,
    )?;
    // U[:, j] *= σ[j] — one kernel launch, broadcast column scaling. `d_u` now
    // holds the scaled embedding U·Σ, col-major (n_obs × n_components), and is
    // returned device-resident; the wrappers below decide host vs device
    // finalization.
    gpu_scale_columns(dev, &mut d_u, &d_sigma, n_obs, n_components)?;

    // Components: rows of U_hat^T → (n_components × n_vars).
    // U_hat is already on host (from the CPU SVD of B — B is small, so this
    // stays on CPU per the Phase 3 plan).
    let mut components = vec![0.0f32; n_components * n_vars];
    for pc in 0..n_components {
        for v in 0..n_vars {
            components[pc * n_vars + v] = u_hat[(v, pc)] as f32;
        }
    }

    // Variance explained = σ² / (n-1)
    let denom = (n_obs as f64 - 1.0).max(1.0);
    let variance_explained: Vec<f64> = sigma
        .iter()
        .take(n_components)
        .map(|&s| s * s / denom)
        .collect();

    // Total variance from pre-computed column sum-of-squares
    let total_var = total_variance_from_col_sq(&col_sum_sq, means.as_deref(), n_obs);

    let variance_ratio: Vec<f64> = if total_var > 0.0 {
        variance_explained.iter().map(|&v| v / total_var).collect()
    } else {
        vec![0.0; n_components]
    };

    Ok(RandomizedPcaCore {
        d_u,
        components,
        variance_explained,
        variance_ratio,
        mean: means,
        n_components,
        n_obs,
        n_vars,
        resident_csr,
    })
}

/// GPU-accelerated randomized PCA returning a host [`GpuPcaResult`].
///
/// Thin wrapper over [`randomized_pca_core`] that downloads the col-major
/// embedding once and transposes it to the scanpy-compatible row-major
/// `(n_obs × n_components)` layout. Behaviour and output are unchanged from
/// before the device-residency refactor.
#[allow(clippy::too_many_arguments)]
pub fn gpu_randomized_pca(
    dev: &GpuDevice,
    source: &(dyn ShardSource + Sync),
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    seed: u64,
    qr_method: QrMethod,
    tuning: GpuPcaTuning,
) -> Result<GpuPcaResult, GpuError> {
    let core = randomized_pca_core(
        dev,
        source,
        n_components,
        n_oversamples,
        n_power_iterations,
        zero_center,
        seed,
        qr_method,
        tuning,
    )?;
    let RandomizedPcaCore {
        d_u,
        components,
        variance_explained,
        variance_ratio,
        mean,
        n_components,
        n_obs,
        n_vars,
        resident_csr,
    } = core;

    // Single D→H copy of the final embedding, col-major (n_obs × n_components).
    dev.synchronize()?;
    let u_host_colmajor = dev.dtoh_copy(&d_u)?;

    // Transpose col-major → row-major for the scanpy-compatible layout.
    let mut embeddings = vec![0.0f32; n_obs * n_components];
    for pc in 0..n_components {
        for i in 0..n_obs {
            embeddings[i * n_components + pc] = u_host_colmajor[pc * n_obs + i];
        }
    }

    Ok(GpuPcaResult {
        embeddings,
        components,
        variance_explained,
        variance_ratio,
        mean,
        n_components,
        n_obs,
        n_vars,
        resident_csr,
    })
}

/// GPU-accelerated randomized PCA returning a **device-resident** embedding
/// ([`GpuPcaDeviceResult`]) — V3 plan Phase 2.3.
///
/// Identical to [`gpu_randomized_pca`] except the embedding never touches the
/// host: the col-major `U·Σ` is transposed in place on the GPU (via
/// [`gpu_transpose_f32`]) into a row-major [`DeviceEmbedding`], ready to feed
/// straight into `gpu_knn_cagra_device` as the CAGRA dataset. The small
/// loadings/variance arrays remain on the host (already produced by the CPU
/// SVD).
#[allow(clippy::too_many_arguments)]
pub fn gpu_randomized_pca_device(
    dev: &GpuDevice,
    source: &(dyn ShardSource + Sync),
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    seed: u64,
    qr_method: QrMethod,
    tuning: GpuPcaTuning,
) -> Result<GpuPcaDeviceResult, GpuError> {
    let core = randomized_pca_core(
        dev,
        source,
        n_components,
        n_oversamples,
        n_power_iterations,
        zero_center,
        seed,
        qr_method,
        tuning,
    )?;
    let RandomizedPcaCore {
        d_u,
        components,
        variance_explained,
        variance_ratio,
        mean,
        n_components,
        n_obs,
        n_vars,
        resident_csr,
    } = core;

    // Transpose col-major (n_obs × n_components) → row-major on the device.
    // A fresh cuBLAS handle is intentional: the core's handle was scoped to its
    // own stream usage and already dropped, so the transpose needs its own.
    let cublas_handle = CublasHandle::new()?;
    // `d_u` and `d_rowmajor` are both live across the transpose, so peak GPU
    // memory transiently holds ~2× the embedding (~480 MB at 1M × 60 f32).
    let mut d_rowmajor = dev.alloc_zeros::<f32>(n_obs * n_components)?;
    gpu_transpose_f32(
        &cublas_handle,
        dev.stream(),
        &d_u,
        &mut d_rowmajor,
        n_obs,
        n_components,
    )?;
    dev.synchronize()?;
    let embeddings = DeviceEmbedding::new(d_rowmajor, n_obs, n_components)?;

    Ok(GpuPcaDeviceResult {
        embeddings,
        components,
        variance_explained,
        variance_ratio,
        mean,
        n_components,
        n_obs,
        n_vars,
        resident_csr,
    })
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Strided mean-correct: apply `Y[global_row + r, c] -= mc[c]` over a
/// `(shard_rows × k)` sub-region of `Y` (which is laid out as a
/// `(ld × k)` col-major matrix). Untouched rows are not read or written.
///
/// Used by the strided PCA matmat path: SpMM writes directly into the
/// global `(n_obs × k)` output at row offset `global_row`, then this kernel
/// corrects the same slice without copying through a contiguous shard
/// temporary.
pub(crate) fn gpu_mean_correct_colmajor_strided(
    dev: &GpuDevice,
    y: &mut CudaSlice<f32>,
    mc: &CudaSlice<f32>,
    shard_rows: usize,
    k: usize,
    global_row: usize,
    ld: usize,
) -> Result<(), GpuError> {
    let total = (shard_rows as u64) * (k as u64);
    if total == 0 {
        return Ok(());
    }
    debug_assert!(
        global_row + shard_rows <= ld,
        "strided mean-correct: global_row + shard_rows ({}) exceeds ld ({})",
        global_row + shard_rows,
        ld,
    );

    let module = dev.load_module_cached(COLMAJOR_OPS_PTX)?;
    let func = module
        .load_function("mean_correct_colmajor_strided_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("mean_correct_strided: {e}")))?;

    let shard_rows_i64 = shard_rows as i64;
    let k_i64 = k as i64;
    let global_row_i64 = global_row as i64;
    let ld_i64 = ld as i64;

    let cfg = flat_launch_1d(total, 256)?;

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(y)
            .arg(mc)
            .arg(&shard_rows_i64)
            .arg(&k_i64)
            .arg(&global_row_i64)
            .arg(&ld_i64)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("mean_correct_strided: {e}")))?;

    Ok(())
}

/// Compute column sums of a col-major matrix on GPU.
///
/// Returns a vector of k column sums.
pub(crate) fn gpu_column_sums(
    dev: &GpuDevice,
    x: &CudaSlice<f32>, // (m × k) col-major
    m: usize,
    k: usize,
) -> Result<CudaSlice<f32>, GpuError> {
    let mut out = dev.alloc_zeros::<f32>(k)?;
    gpu_column_sums_into(dev, x, &mut out, m, k)?;
    Ok(out)
}

/// [`gpu_column_sums`] writing into a caller-provided `out` (length `k`)
/// instead of allocating. Required inside CUDA-graph capture (Task 2.5), where
/// device allocations are forbidden — the resident PCA transpose segment
/// pre-allocates the column-sum buffer once and reuses it across replays.
///
/// `column_sum_kernel` uses exactly one block per column and writes each
/// `out[col]` with a single plain store, so for `m > 0` it fully overwrites
/// `out` and a reused (non-fresh) buffer is safe across power iterations with
/// no pre-zero. The only path that needs zeroing is the `m == 0` / `k == 0`
/// short-circuit, where the kernel is skipped and a reused buffer would
/// otherwise retain stale sums — that path `memset`s `out`. The zero is a
/// `memset` (not a device allocation), so it stays CUDA-graph-capture legal;
/// this function is capture-safe (no device allocs on any path).
pub(crate) fn gpu_column_sums_into(
    dev: &GpuDevice,
    x: &CudaSlice<f32>, // (m × k) col-major
    out: &mut CudaSlice<f32>,
    m: usize,
    k: usize,
) -> Result<(), GpuError> {
    if m == 0 || k == 0 {
        // Kernel is skipped here, so zero `out` — a reused buffer must still
        // read back zero. (For m > 0 the kernel writes every out[col] exactly
        // once, so there is no redundant pre-zero on the hot path.)
        dev.stream()
            .memset_zeros(out)
            .map_err(|e| GpuError::KernelLaunchFailed(format!("column_sum zero: {e}")))?;
        return Ok(());
    }

    let module = dev.load_module_cached(COLMAJOR_OPS_PTX)?;
    let func = module
        .load_function("column_sum_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("column_sum: {e}")))?;

    // `column_sum_kernel` takes `m`/`k` as i32. Both fit for any real matrix —
    // the resident CSR path already caps n_obs so the cuSPARSE i32 indptr fits —
    // but assert the contract (defense-in-depth, mirroring the crate's i32-cast
    // guards; cf. review finding G4).
    debug_assert!(
        m <= i32::MAX as usize && k <= i32::MAX as usize,
        "column_sum_kernel dims exceed i32: m={m}, k={k}"
    );
    let m_i32 = m as i32;
    let k_i32 = k as i32;

    // Launch: grid = (1, k) — exactly one block per column; block = (256, 1).
    // Each block reduces all m rows of its column in f64 (finding G2: no
    // cross-block atomicAdd → deterministic + precise). Dynamic shared memory:
    // `column_sum_kernel` uses `extern __shared__ double warp_sums[]`, needing
    // (blockDim.x / 32) doubles = (256/32) * 8 = 64 bytes.
    let threads: u32 = 256;
    let n_warps = threads.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (1, k as u32, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: n_warps * std::mem::size_of::<f64>() as u32,
    };

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(x)
            .arg(out)
            .arg(&m_i32)
            .arg(&k_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("column_sum: {e}")))?;

    Ok(())
}

/// Outer product subtraction on GPU: Z[v, j] -= mu[v] * sum_q[j].
pub(crate) fn gpu_outer_sub(
    dev: &GpuDevice,
    z: &mut CudaSlice<f32>, // (n_vars × k) col-major
    mu: &CudaSlice<f32>,    // [n_vars]
    sum_q: &CudaSlice<f32>, // [k]
    n_vars: usize,
    k: usize,
) -> Result<(), GpuError> {
    let total = (n_vars as u64) * (k as u64);
    if total == 0 {
        return Ok(());
    }
    let module = dev.load_module_cached(COLMAJOR_OPS_PTX)?;
    let func = module
        .load_function("outer_sub_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("outer_sub: {e}")))?;

    let n_vars_i64 = n_vars as i64;
    let k_i64 = k as i64;

    let cfg = flat_launch_1d(total, 256)?;

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(z)
            .arg(mu)
            .arg(sum_q)
            .arg(&n_vars_i64)
            .arg(&k_i64)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("outer_sub: {e}")))?;

    Ok(())
}

/// Broadcast-scale the columns of a col-major matrix on GPU.
///
/// `U[r, c] *= sigma[c]` in place, for all `(r, c)`.
/// Used by the GPU-resident final-embedding step in `gpu_randomized_pca`
/// (Phase 3.3): after computing `U = Q @ V` via cuBLAS `sgemm`, scale each
/// column by the corresponding singular value in one kernel launch.
pub(crate) fn gpu_scale_columns(
    dev: &GpuDevice,
    u: &mut CudaSlice<f32>,
    sigma: &CudaSlice<f32>,
    m: usize,
    k: usize,
) -> Result<(), GpuError> {
    let total = (m as u64) * (k as u64);
    if total == 0 {
        return Ok(());
    }
    let module = dev.load_module_cached(COLMAJOR_OPS_PTX)?;
    let func = module
        .load_function("scale_columns_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("scale_columns: {e}")))?;

    let m_i64 = m as i64;
    let k_i64 = k as i64;

    let cfg = flat_launch_1d(total, 256)?;

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(u)
            .arg(sigma)
            .arg(&m_i64)
            .arg(&k_i64)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("scale_columns: {e}")))?;

    Ok(())
}

// NOTE: `compute_means_and_col_sq` has been replaced by
// `BackedCsrReader::col_means_and_sum_sq()` in scx-format-io.
// `compute_total_variance_from_col_sq` has been replaced by
// `scx_sparse::total_variance_from_col_sq()`.

/// Format ScxError as GpuError.
fn format_scx_error(e: scx_format_io::ScxError) -> GpuError {
    GpuError::InvalidShard(format!("SCX read error: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_scale_columns() {
        // Phase 3.3 — broadcast-scale columns of a col-major matrix.
        let dev = require_gpu!();

        // U (col-major 4×3):
        //   col 0 = [1, 2, 3, 4]
        //   col 1 = [5, 6, 7, 8]
        //   col 2 = [9, 10, 11, 12]
        let u_host: Vec<f32> = (1..=12).map(|v| v as f32).collect();
        let sigma_host: Vec<f32> = vec![2.0, -1.0, 0.5];

        let mut d_u = dev.htod_copy(&u_host).unwrap();
        let d_sigma = dev.htod_copy(&sigma_host).unwrap();

        gpu_scale_columns(&dev, &mut d_u, &d_sigma, 4, 3).unwrap();
        dev.synchronize().unwrap();
        let out = dev.dtoh_copy(&d_u).unwrap();

        // Expected: col 0 × 2, col 1 × -1, col 2 × 0.5
        let expected: Vec<f32> = vec![
            2.0, 4.0, 6.0, 8.0, // col 0
            -5.0, -6.0, -7.0, -8.0, // col 1
            4.5, 5.0, 5.5, 6.0, // col 2
        ];
        for i in 0..out.len() {
            assert!(
                (out[i] - expected[i]).abs() < 1e-5,
                "scale_columns mismatch at {i}: got {}, expected {}",
                out[i],
                expected[i]
            );
        }
    }

    /// The col-major kernels must still do their work when the matrix has more
    /// than 2³¹ elements (review §8.4).
    ///
    /// With a 32-bit `total = m * k` the product overflows a signed `int`,
    /// which is UB, so what the kernel then does is nvcc's choice: measured
    /// against the pre-fix kernels at this exact shape (nvcc 12.6, `-O3`,
    /// `compute_70`, H100) it is `CUDA_ERROR_ILLEGAL_ADDRESS` — threads whose
    /// `idx` wrapped negative survive the bounds check and write below the
    /// buffer. Past 2³² elements the host's `(total as u32)` block count is
    /// the silent variant: a grid too small to reach the tail, no error at all.
    ///
    /// Either way the work does not happen, on the two paths that matter at
    /// census scale (36M cells × k=60 = 2.16e9, which the PCA VRAM pre-flight
    /// admits on an 80 GB H100): `zero_center=True` yielding an uncentered PCA,
    /// and the final embedding leaving unscaled by its singular values. The
    /// assertions below cover the silent shape; the `synchronize` covers the
    /// illegal-address shape.
    ///
    /// Sized at the smallest matrix that reaches the boundary — 2³¹ + 256
    /// elements, 8 GiB of f32 — because below it the bug does not exist. The
    /// buffer is `alloc_zeros`, so nothing 8 GiB wide is ever allocated on the
    /// host, and only five single elements are read back (a `dtoh_copy` would
    /// pull the whole 8 GiB). Both kernels run on the one allocation: two 8 GiB
    /// tests under libtest's default thread pool would double the VRAM demand.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn colmajor_kernels_do_their_work_past_2_31_elements() {
        let dev = require_gpu!();

        const K: usize = 4;
        // 2^31 + 256 elements: the first size at which a 32-bit element count
        // wraps. m is the row count of the (m × K) col-major matrix.
        let m: usize = (1usize << 31) / K + 64;
        let total = m * K;
        assert!(total > (1usize << 31), "test must cross the 2^31 boundary");

        // 8 GiB for the matrix, plus headroom for the driver's own allocations.
        require_gpu_cap!(vram: total * 4 + (2 << 30), &dev);

        let mut d_y = dev.alloc_zeros::<f32>(total).unwrap();
        // mc[c] = c + 1 against an all-zero Y, so a corrected element reads
        // -(col + 1) and an untouched one reads exactly 0.0.
        let d_mc = dev.htod_copy(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let d_sigma = dev.htod_copy(&[2.0f32; K]).unwrap();

        gpu_mean_correct_colmajor_strided(&dev, &mut d_y, &d_mc, m, K, 0, m).unwrap();
        gpu_scale_columns(&dev, &mut d_y, &d_sigma, m, K).unwrap();
        // Launches are async, so a wrapped index surfaces here rather than
        // above — this `unwrap` is load-bearing, not ceremony. Against the
        // pre-fix kernels it is what fires (CUDA_ERROR_ILLEGAL_ADDRESS).
        dev.synchronize().unwrap();

        // Sentinels spanning all four columns and straddling the 2^31 index:
        // flat 1<<31 and the final element are the two a 32-bit `idx` could not
        // address even if the count were right.
        for flat in [0usize, m + 7, 2 * m, 1usize << 31, total - 1] {
            let col = flat / m;
            let expected = -2.0 * (col + 1) as f32;
            let got = dev.stream().clone_dtoh(&d_y.slice(flat..flat + 1)).unwrap()[0];
            assert_eq!(
                got, expected,
                "element {flat} (col {col}) reads {got}, expected {expected} — \
                 0.0 means the kernel returned without touching the matrix",
            );
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_randomized_pca_phase3_parity() {
        // After the GPU-resident-embedding refactor, verify that
        // variance_ratio is non-negative, monotone-descending by PC index, and
        // sums to ≤ 1 + ε — this catches a σ-scaling bug in the
        // sgemm + `gpu_scale_columns` tail.
        //
        // The earlier loadings-vs-`gpu_covariance_pca` cosine oracle was dropped
        // along with the GPU covariance path; randomized-PCA loadings are now
        // validated against scanpy/rapids by the Python-side `subspace_cos_min`
        // gate (cross-crate parity can't live here — scx-gpu cannot depend on
        // scx-accel).
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        use scx_format_io::ShardSource;
        use scx_sparse::ScxCsr;

        let dev = require_gpu!();

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
            fn read_shard(&self, i: usize) -> scx_format_io::Result<ScxCsr> {
                Ok(self.shards[i].clone())
            }
        }

        fn random_csr(n_rows: usize, n_cols: usize, density: f32, seed: u64) -> ScxCsr {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut indptr: Vec<i64> = Vec::with_capacity(n_rows + 1);
            let mut indices: Vec<i32> = Vec::new();
            let mut data: Vec<f32> = Vec::new();
            indptr.push(0);
            for _ in 0..n_rows {
                for c in 0..n_cols {
                    if rng.gen_bool(density as f64) {
                        indices.push(c as i32);
                        data.push(rng.gen_range(-1.0..1.0));
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

        let n_rows = 800;
        let n_cols = 120;
        let k = 15;
        let mut csr = random_csr(n_rows, n_cols, 0.08, 314);
        // Inflate column variances geometrically so the top-k eigenvalues are
        // well-separated. Without this, uniformly random sparse data has
        // near-Marchenko-Pastur eigenvalues (adjacent ratios ~0.99) and
        // randomized PCA at 4 power iterations cannot resolve per-PC bases
        // even though the top-k subspace is recovered correctly. With
        // decay=0.85, adjacent eigenvalue ratio is ~0.72 — plenty of headroom
        // for q=4 to converge to per-PC cosine ≥ 0.99.
        let decay = 0.85f32;
        for nz in 0..csr.indices.len() {
            let c = csr.indices[nz] as usize;
            csr.data[nz] *= decay.powi(c as i32);
        }
        let shards = split_into_shards(&csr, 4);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let gpu_rand = gpu_randomized_pca(
            &dev,
            &source,
            k,
            10,
            4,
            true,
            42,
            QrMethod::default(),
            crate::math_policy::GpuPcaTuning::default(),
        )
        .unwrap();

        // Variance ratios: non-negative, monotone-descending within tolerance,
        // and sum ≤ 1 + ε (a σ-scaling bug would easily violate this).
        let sum_ratio: f64 = gpu_rand.variance_ratio.iter().sum();
        assert!(
            (0.0..=1.0 + 1e-3).contains(&sum_ratio),
            "variance_ratio sum out of range: {sum_ratio}"
        );
        for j in 1..gpu_rand.variance_ratio.len() {
            let prev = gpu_rand.variance_ratio[j - 1];
            let curr = gpu_rand.variance_ratio[j];
            assert!(prev >= 0.0 && curr >= 0.0);
            // Allow small numerical wiggle between adjacent PCs.
            assert!(
                curr <= prev + 1e-6,
                "variance_ratio not monotone at PC {j}: prev={prev}, curr={curr}"
            );
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_randomized_pca_cholesky_vs_householder() {
        // Phase 4.5 — opt-in CholeskyQR2 must produce the same PCA as the
        // default Householder path on well-conditioned inputs. Both paths
        // share the same RNG (matched seed), so cosine ≥ 0.999 is the right
        // bar — tighter than the GPU-vs-CPU Phase-3 parity (0.99) because
        // the only algorithmic difference is the QR step.
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        use scx_format_io::ShardSource;
        use scx_sparse::ScxCsr;

        let dev = require_gpu!();

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
            fn read_shard(&self, i: usize) -> scx_format_io::Result<ScxCsr> {
                Ok(self.shards[i].clone())
            }
        }

        fn random_csr(n_rows: usize, n_cols: usize, density: f32, seed: u64) -> ScxCsr {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut indptr: Vec<i64> = Vec::with_capacity(n_rows + 1);
            let mut indices: Vec<i32> = Vec::new();
            let mut data: Vec<f32> = Vec::new();
            indptr.push(0);
            for _ in 0..n_rows {
                for c in 0..n_cols {
                    if rng.gen_bool(density as f64) {
                        indices.push(c as i32);
                        data.push(rng.gen_range(-1.0..1.0));
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

        fn row_abs_cosine(a: &[f32], b: &[f32], k: usize, d: usize) -> f32 {
            let mut total = 0.0f32;
            for i in 0..k {
                let ra = &a[i * d..(i + 1) * d];
                let rb = &b[i * d..(i + 1) * d];
                let dot: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
                let na: f32 = ra.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb: f32 = rb.iter().map(|x| x * x).sum::<f32>().sqrt();
                total += (dot / (na * nb).max(1e-12)).abs();
            }
            total / k as f32
        }

        let n_rows = 800;
        let n_cols = 120;
        let k = 15;
        let csr = random_csr(n_rows, n_cols, 0.08, 2025);
        let shards = split_into_shards(&csr, 4);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let hh = gpu_randomized_pca(
            &dev,
            &source,
            k,
            10,
            4,
            true,
            42,
            QrMethod::Householder,
            crate::math_policy::GpuPcaTuning::default(),
        )
        .unwrap();
        let ch = gpu_randomized_pca(
            &dev,
            &source,
            k,
            10,
            4,
            true,
            42,
            QrMethod::Cholesky,
            crate::math_policy::GpuPcaTuning::default(),
        )
        .unwrap();

        let cos = row_abs_cosine(&ch.components, &hh.components, k, n_cols);
        assert!(
            cos > 0.999,
            "CholeskyQR2 vs Householder: cosine = {cos} (want > 0.999)"
        );

        // Variance ratios should be close between the two QR methods.
        for j in 0..k {
            let diff = (ch.variance_ratio[j] - hh.variance_ratio[j]).abs();
            assert!(
                diff < 1e-3,
                "variance_ratio[{j}] diff = {diff} between cholesky and householder"
            );
        }
    }

    /// The device-resident PCA entry (`gpu_randomized_pca_device`) must produce
    /// the same result as the host entry (`gpu_randomized_pca`): both share the
    /// `randomized_pca_core` pipeline (same seed → same `d_u`) and differ only
    /// in finalizing the embedding (host transpose vs device `cublasSgeam`
    /// transpose). The downloaded device embedding must therefore match the
    /// host embedding element-for-element, and the loadings/variance are shared.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_gpu_randomized_pca_device_matches_host() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        use scx_format_io::ShardSource;
        use scx_sparse::ScxCsr;

        let dev = require_gpu!();

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
            fn read_shard(&self, i: usize) -> scx_format_io::Result<ScxCsr> {
                Ok(self.shards[i].clone())
            }
        }

        let n_rows = 400;
        let n_cols = 60;
        let k = 12;
        let mut rng = StdRng::seed_from_u64(99);
        let mut indptr: Vec<i64> = vec![0];
        let mut indices: Vec<i32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        for _ in 0..n_rows {
            for c in 0..n_cols {
                if rng.gen_bool(0.1) {
                    indices.push(c as i32);
                    data.push(rng.gen_range(-1.0..1.0));
                }
            }
            indptr.push(indices.len() as i64);
        }
        let csr = ScxCsr::new_unchecked((n_rows, n_cols), indptr, indices, data);
        // Two shards to exercise the streaming path.
        let mid = n_rows / 2;
        let p_mid = csr.indptr[mid] as usize;
        let shard0 = ScxCsr::new_unchecked(
            (mid, n_cols),
            csr.indptr[0..=mid].to_vec(),
            csr.indices[0..p_mid].to_vec(),
            csr.data[0..p_mid].to_vec(),
        );
        let shard1 = ScxCsr::new_unchecked(
            (n_rows - mid, n_cols),
            csr.indptr[mid..=n_rows]
                .iter()
                .map(|&p| p - csr.indptr[mid])
                .collect(),
            csr.indices[p_mid..].to_vec(),
            csr.data[p_mid..].to_vec(),
        );
        let source = InMemorySource {
            shards: vec![shard0, shard1],
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let host = gpu_randomized_pca(
            &dev,
            &source,
            k,
            10,
            2,
            true,
            7,
            QrMethod::Householder,
            crate::math_policy::GpuPcaTuning::default(),
        )
        .unwrap();
        let dev_res = gpu_randomized_pca_device(
            &dev,
            &source,
            k,
            10,
            2,
            true,
            7,
            QrMethod::Householder,
            crate::math_policy::GpuPcaTuning::default(),
        )
        .unwrap();

        // Shapes.
        assert_eq!(dev_res.n_obs, n_rows);
        assert_eq!(dev_res.n_components, k);
        assert_eq!(dev_res.embeddings.shape(), (n_rows, k));
        assert_eq!(dev_res.components.len(), host.components.len());

        // The host and device entries share `randomized_pca_core` but each call
        // runs it *separately* (there is no single-call hook returning both), so
        // GPU SpMM/QR run-to-run nondeterminism makes them differ at f32 noise
        // level (~1e-6). A tight tolerance still catches a genuinely wrong device
        // transpose (which would be O(1) off) while tolerating that noise.
        for (a, b) in dev_res.components.iter().zip(host.components.iter()) {
            assert!(
                (a - b).abs() <= 1e-3,
                "device components {a} vs host {b} differ beyond tol"
            );
        }
        for (a, b) in dev_res
            .variance_explained
            .iter()
            .zip(host.variance_explained.iter())
        {
            assert!(
                (a - b).abs() <= 1e-3 * (1.0 + b.abs()),
                "variance_explained {a} vs {b} differ beyond tol"
            );
        }
        for (a, b) in dev_res
            .variance_ratio
            .iter()
            .zip(host.variance_ratio.iter())
        {
            assert!(
                (a - b).abs() <= 1e-3 * (1.0 + b.abs()),
                "variance_ratio {a} vs {b} differ beyond tol"
            );
        }

        // Device embedding downloads to the same row-major values as the host
        // path: the device `cublasSgeam` transpose is value-preserving, so the
        // two agree up to the same run-to-run GPU noise.
        let dev_emb = dev_res.embeddings.to_host(&dev).unwrap();
        assert_eq!(dev_emb.len(), host.embeddings.len());
        for (a, b) in dev_emb.iter().zip(host.embeddings.iter()) {
            assert!(
                (a - b).abs() <= 1e-3 * (1.0 + b.abs()),
                "device embedding {a} vs host {b} differ beyond tol"
            );
        }
    }
}
