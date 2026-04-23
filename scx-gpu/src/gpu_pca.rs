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

use scx_format::total_variance_from_col_sq;
use scx_format::ShardSource;

use crate::cublas::{gpu_sgemm, gpu_sgemv, CublasHandle};
use crate::curand::random_gaussian_gpu;
use crate::cusolver::{gpu_qr_q, CusolverHandle};
use crate::cusparse::{spmm_csr, spmm_csr_transpose, CusparseHandle};
use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::shard_decode::GpuCsr;

/// PTX source for the row-major mean-correction kernel, compiled at build time.
const MEAN_CORRECT_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/spmm_mean_correct.ptx"));

/// PTX source for col-major scatter/gather/mean-correct/column-sum kernels.
const COLMAJOR_OPS_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/colmajor_ops.ptx"));

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
}

/// GPU-accelerated randomized PCA.
///
/// Streams data shard-by-shard from `source`, performing SpMM on GPU via
/// cuSPARSE, QR via cuSOLVER, and the final SVD on CPU via `faer`.
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
/// 8. Embeddings = Q @ V × Σ (CPU — Q downloaded, small multiply)
///
/// Steps 3-6 stream from any `ShardSource` without materializing full X.
/// The `Sync` bound is required so that a future refactor to
/// `DoubleBufferedShardLoader` (Phase 2+) works without signature churn.
/// Peak GPU memory: ~500 MB for 1M cells (dominated by Y and Q matrices).
#[allow(clippy::too_many_arguments)]
pub fn gpu_randomized_pca(
    dev: &GpuDevice,
    source: &(dyn ShardSource + Sync),
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    seed: u64,
) -> Result<GpuPcaResult, GpuError> {
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

    // Step 3: Y = streaming_gpu_spmm_forward(X, Ω) with mean correction
    // Y is col-major (n_obs × k)
    let d_y = streaming_gpu_spmm_forward(
        dev,
        &cusparse_handle,
        &cublas_handle,
        source,
        &d_omega,
        d_means.as_ref(),
        n_obs,
        n_vars,
        k,
    )?;

    // Step 4: Q = qr(Y)
    let mut d_y_mut = d_y;
    let mut d_q = gpu_qr_q(&cusolver_handle, dev.stream(), dev, &mut d_y_mut, n_obs, k)?;

    // Step 5: Power iterations
    for _ in 0..n_power_iterations {
        // B = X^T @ Q (n_vars × k, col-major)
        let d_b = streaming_gpu_spmm_transpose(
            dev,
            &cusparse_handle,
            source,
            &d_q,
            d_means.as_ref(),
            n_obs,
            n_vars,
            k,
        )?;

        // Q_B = qr(B)
        let mut d_b_mut = d_b;
        let d_q_b = gpu_qr_q(&cusolver_handle, dev.stream(), dev, &mut d_b_mut, n_vars, k)?;

        // Y = X @ Q_B
        let d_y2 = streaming_gpu_spmm_forward(
            dev,
            &cusparse_handle,
            &cublas_handle,
            source,
            &d_q_b,
            d_means.as_ref(),
            n_obs,
            n_vars,
            k,
        )?;

        // Q = qr(Y)
        let mut d_y2_mut = d_y2;
        d_q = gpu_qr_q(&cusolver_handle, dev.stream(), dev, &mut d_y2_mut, n_obs, k)?;
    }

    // Step 6: B = X^T @ Q (final, n_vars × k)
    let d_b_final = streaming_gpu_spmm_transpose(
        dev,
        &cusparse_handle,
        source,
        &d_q,
        d_means.as_ref(),
        n_obs,
        n_vars,
        k,
    )?;

    // Step 7: Download B to host, SVD via faer (f64 for accuracy)
    dev.synchronize()?;
    let b_host_f32 = dev.dtoh_copy(&d_b_final)?;

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
    gpu_sgemm(
        &cublas_handle,
        dev.stream(),
        &d_q,
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
    // U[:, j] *= σ[j] — one kernel launch, broadcast column scaling.
    gpu_scale_columns(dev, &mut d_u, &d_sigma, n_obs, n_components)?;

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

    Ok(GpuPcaResult {
        embeddings,
        components,
        variance_explained,
        variance_ratio,
        mean: means,
        n_components,
        n_obs,
        n_vars,
    })
}

// ---------------------------------------------------------------------------
// Streaming GPU SpMM helpers
// ---------------------------------------------------------------------------

/// Streaming forward SpMM on GPU: Y = (X - μ) @ M, shard-by-shard.
///
/// For each shard:
///   1. Read shard to host (via ShardSource)
///   2. Upload indptr/indices/data to GPU → GpuCsr
///   3. GpuCsr → CusparseSpMatDescr (zero-copy on GPU)
///   4. cuSPARSE SpMM: Y_slice = A_shard @ M (accumulated with beta=1.0)
///   5. Mean-correction kernel on Y_slice rows
///
/// Returns Y as col-major (n_obs × k) on GPU.
#[allow(clippy::too_many_arguments)]
fn streaming_gpu_spmm_forward(
    dev: &GpuDevice,
    cusparse: &CusparseHandle,
    cublas: &CublasHandle,
    source: &dyn ShardSource,
    d_m: &CudaSlice<f32>, // (n_vars × k) col-major on GPU
    d_means: Option<&CudaSlice<f32>>,
    n_obs: usize,
    n_vars: usize,
    k: usize,
) -> Result<CudaSlice<f32>, GpuError> {
    let mut d_y = dev.alloc_zeros::<f32>(n_obs * k)?;
    let n_shards = source.n_shards();

    // Pre-compute mean correction vector on GPU: mc = Mᵀ · μ   (length k).
    // M is col-major (n_vars × k). cuBLAS sgemv with op_A = T gives y = Aᵀ · x
    // where A has backing shape (m, n) = (n_vars, k) and x length n_vars,
    // producing y of length k. Replaces the D→H round-trip that previously
    // downloaded the full `d_m` (n_vars × k) to host once per power iteration.
    let d_mc: Option<CudaSlice<f32>> = if let Some(d_mu) = d_means {
        let mut mc = dev.alloc_zeros::<f32>(k)?;
        gpu_sgemv(
            cublas,
            dev.stream(),
            d_m,
            d_mu,
            &mut mc,
            n_vars,
            k,
            1.0,
            0.0,
            cbs::cublasOperation_t::CUBLAS_OP_T,
        )?;
        Some(mc)
    } else {
        None
    };

    let mut global_row = 0usize;

    for shard_idx in 0..n_shards {
        let csr = source.read_shard(shard_idx).map_err(format_scx_error)?;
        let shard_rows = csr.n_rows();

        if shard_rows == 0 {
            continue;
        }

        // Upload CSR to GPU
        let gpu_csr = upload_csr_to_gpu(dev, &csr)?;

        // Create cuSPARSE descriptor
        let a_desc = gpu_csr.to_cusparse_csr(dev.stream())?;

        // Y_shard is a slice of Y starting at row `global_row`.
        // cuSPARSE SpMM: C = α·A·B + β·C
        // A: shard CSR (shard_rows × n_vars)
        // B: M col-major (n_vars × k)
        // C: Y_shard col-major (shard_rows × k)
        //
        // We need to point C at the right offset in d_y.
        // col-major Y: Y[i, j] = d_y[j * n_obs + i]
        // Y_shard starts at row global_row: Y_shard[r, j] = d_y[j * n_obs + global_row + r]
        // This is NOT contiguous in memory for col-major layout (columns are n_obs apart).
        //
        // Option: allocate a temporary shard-sized output and then scatter into d_y.
        let mut d_y_shard = dev.alloc_zeros::<f32>(shard_rows * k)?;

        spmm_csr(
            cusparse,
            dev.stream(),
            dev,
            &a_desc,
            d_m,
            &mut d_y_shard,
            shard_rows,
            n_vars,
            k,
            1.0,
            0.0,
        )?;

        // Apply mean correction on GPU: Y_shard[r, j] -= mc[j]
        if let Some(ref mc) = d_mc {
            gpu_mean_correct_colmajor(dev, &mut d_y_shard, mc, shard_rows, k)?;
        }

        // Scatter shard result into global Y on GPU (col-major)
        gpu_scatter_colmajor(dev, &d_y_shard, &mut d_y, shard_rows, k, global_row, n_obs)?;

        global_row += shard_rows;
    }

    Ok(d_y)
}

/// Streaming transpose SpMM on GPU: Z = (X - μ)^T @ Q, shard-by-shard.
///
/// Returns Z as col-major (n_vars × k) on GPU.
#[allow(clippy::too_many_arguments)]
fn streaming_gpu_spmm_transpose(
    dev: &GpuDevice,
    cusparse: &CusparseHandle,
    source: &dyn ShardSource,
    d_q: &CudaSlice<f32>, // (n_obs × k) col-major on GPU
    d_means: Option<&CudaSlice<f32>>,
    n_obs: usize,
    n_vars: usize,
    k: usize,
) -> Result<CudaSlice<f32>, GpuError> {
    let mut d_z = dev.alloc_zeros::<f32>(n_vars * k)?;
    let n_shards = source.n_shards();
    let mut global_row = 0usize;

    for shard_idx in 0..n_shards {
        let csr = source.read_shard(shard_idx).map_err(format_scx_error)?;
        let shard_rows = csr.n_rows();

        if shard_rows == 0 {
            continue;
        }

        // Upload CSR to GPU
        let gpu_csr = upload_csr_to_gpu(dev, &csr)?;
        let a_desc = gpu_csr.to_cusparse_csr(dev.stream())?;

        // Extract Q_shard on GPU: Q[global_row..global_row+shard_rows, :]
        // col-major Q: Q[i, j] = d_q[j * n_obs + i]
        // Q_shard needs to be a contiguous (shard_rows × k) col-major matrix.
        let d_q_shard = gpu_gather_colmajor(dev, d_q, shard_rows, k, global_row, n_obs)?;

        // Z += A^T @ Q_shard
        // A: (shard_rows × n_vars), A^T: (n_vars × shard_rows)
        // Q_shard: (shard_rows × k)
        // Result: (n_vars × k) — accumulated into d_z
        spmm_csr_transpose(
            cusparse,
            dev.stream(),
            dev,
            &a_desc,
            &d_q_shard,
            &mut d_z,
            shard_rows,
            n_vars,
            k,
            1.0,
            1.0, // beta=1.0 to accumulate across shards
        )?;

        global_row += shard_rows;
    }

    // Mean centering correction on GPU: Z -= μ @ (1^T @ Q)
    // (1^T @ Q) = column sums of Q = (1 × k)
    if let Some(d_mu) = d_means {
        // Compute column sums of Q entirely on GPU
        let d_sum_q = gpu_column_sums(dev, d_q, n_obs, k)?;

        // Z[v, j] -= means[v] * sum_q[j] — outer product subtraction on GPU
        gpu_outer_sub(dev, &mut d_z, d_mu, &d_sum_q, n_vars, k)?;
    }

    Ok(d_z)
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Upload an ScxCsr to GPU as GpuCsr.
pub(crate) fn upload_csr_to_gpu(
    dev: &GpuDevice,
    csr: &scx_sparse::ScxCsr,
) -> Result<GpuCsr, GpuError> {
    let d_indptr = dev.htod_copy(&csr.indptr)?;
    let d_indices = dev.htod_copy(&csr.indices)?;
    let d_data = dev.htod_copy(&csr.data)?;

    Ok(GpuCsr {
        indptr: d_indptr,
        indices: d_indices,
        data: d_data,
        shape: (csr.n_rows(), csr.n_cols()),
    })
}

/// Scatter a shard's col-major result into the global matrix — GPU kernel.
///
/// Replaces the CPU round-trip version that downloaded the entire n_obs×k
/// matrix to host per shard (240 MB for 1M cells × 60 PCs).
pub(crate) fn gpu_scatter_colmajor(
    dev: &GpuDevice,
    src: &CudaSlice<f32>,     // (shard_rows × k) col-major
    dst: &mut CudaSlice<f32>, // (n_obs × k) col-major
    shard_rows: usize,
    k: usize,
    global_row: usize,
    n_obs: usize,
) -> Result<(), GpuError> {
    let total = shard_rows * k;
    if total == 0 {
        return Ok(());
    }
    let module = dev.load_module_cached(COLMAJOR_OPS_PTX)?;
    let func = module
        .load_function("scatter_colmajor_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("scatter_colmajor: {e}")))?;

    let shard_rows_i32 = shard_rows as i32;
    let n_obs_i32 = n_obs as i32;
    let k_i32 = k as i32;
    let global_row_i32 = global_row as i32;

    let threads: u32 = 256;
    let blocks = (total as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(src)
            .arg(dst)
            .arg(&shard_rows_i32)
            .arg(&n_obs_i32)
            .arg(&k_i32)
            .arg(&global_row_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("scatter_colmajor: {e}")))?;

    Ok(())
}

/// Gather shard rows from global col-major matrix — GPU kernel.
///
/// Replaces the CPU round-trip version that downloaded the entire n_obs×k
/// matrix to host per shard.
pub(crate) fn gpu_gather_colmajor(
    dev: &GpuDevice,
    src: &CudaSlice<f32>, // (n_obs × k) col-major
    shard_rows: usize,
    k: usize,
    global_row: usize,
    n_obs: usize,
) -> Result<CudaSlice<f32>, GpuError> {
    let total = shard_rows * k;
    if total == 0 {
        return dev.alloc_zeros::<f32>(0);
    }
    let mut dst = dev.alloc_zeros::<f32>(total)?;

    let module = dev.load_module_cached(COLMAJOR_OPS_PTX)?;
    let func = module
        .load_function("gather_colmajor_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("gather_colmajor: {e}")))?;

    let shard_rows_i32 = shard_rows as i32;
    let n_obs_i32 = n_obs as i32;
    let k_i32 = k as i32;
    let global_row_i32 = global_row as i32;

    let threads: u32 = 256;
    let blocks = (total as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(src)
            .arg(&mut dst)
            .arg(&shard_rows_i32)
            .arg(&n_obs_i32)
            .arg(&k_i32)
            .arg(&global_row_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("gather_colmajor: {e}")))?;

    Ok(dst)
}

/// Mean-correct a col-major matrix on GPU: Y[r, j] -= mc[j].
///
/// Replaces the CPU round-trip version that downloaded shard-sized data.
pub(crate) fn gpu_mean_correct_colmajor(
    dev: &GpuDevice,
    y: &mut CudaSlice<f32>,
    mc: &CudaSlice<f32>,
    m: usize, // rows
    k: usize, // cols
) -> Result<(), GpuError> {
    let total = m * k;
    if total == 0 {
        return Ok(());
    }
    let module = dev.load_module_cached(COLMAJOR_OPS_PTX)?;
    let func = module
        .load_function("mean_correct_colmajor_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("mean_correct_colmajor: {e}")))?;

    let m_i32 = m as i32;
    let k_i32 = k as i32;

    let threads: u32 = 256;
    let blocks = (total as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(y)
            .arg(mc)
            .arg(&m_i32)
            .arg(&k_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("mean_correct_colmajor: {e}")))?;

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
    if m == 0 || k == 0 {
        return Ok(out);
    }

    let module = dev.load_module_cached(COLMAJOR_OPS_PTX)?;
    let func = module
        .load_function("column_sum_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("column_sum: {e}")))?;

    let m_i32 = m as i32;
    let k_i32 = k as i32;

    // Launch: grid = (ceil(m/256), k), block = (256, 1)
    // Shared memory: column_sum_kernel uses extern __shared__ float warp_sums[]
    // which needs (blockDim.x / 32) floats = (256/32) * 4 = 32 bytes.
    let threads: u32 = 256;
    let blocks_x = (m as u32).div_ceil(threads);
    let n_warps = threads.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (blocks_x, k as u32, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: n_warps * std::mem::size_of::<f32>() as u32,
    };

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(x)
            .arg(&mut out)
            .arg(&m_i32)
            .arg(&k_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("column_sum: {e}")))?;

    Ok(out)
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
    let total = n_vars * k;
    if total == 0 {
        return Ok(());
    }
    let module = dev.load_module_cached(COLMAJOR_OPS_PTX)?;
    let func = module
        .load_function("outer_sub_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("outer_sub: {e}")))?;

    let n_vars_i32 = n_vars as i32;
    let k_i32 = k as i32;

    let threads: u32 = 256;
    let blocks = (total as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(z)
            .arg(mu)
            .arg(sum_q)
            .arg(&n_vars_i32)
            .arg(&k_i32)
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
    let total = m * k;
    if total == 0 {
        return Ok(());
    }
    let module = dev.load_module_cached(COLMAJOR_OPS_PTX)?;
    let func = module
        .load_function("scale_columns_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("scale_columns: {e}")))?;

    let m_i32 = m as i32;
    let k_i32 = k as i32;

    let threads: u32 = 256;
    let blocks = (total as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(u)
            .arg(sigma)
            .arg(&m_i32)
            .arg(&k_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("scale_columns: {e}")))?;

    Ok(())
}

// NOTE: `compute_means_and_col_sq` has been replaced by
// `BackedCsrReader::col_means_and_sum_sq()` in scx-format.
// `compute_total_variance_from_col_sq` has been replaced by
// `scx_format::total_variance_from_col_sq()`.

/// Format ScxError as GpuError.
fn format_scx_error(e: scx_format::ScxError) -> GpuError {
    GpuError::InvalidShard(format!("SCX read error: {e}"))
}

/// Subtract a per-column correction vector from each row of a matrix.
///
/// Computes `Y[row, col] -= mc[col]` for all `(row, col)`.
/// The matrix Y is in **row-major** layout.
///
/// - `y`: row-major `[n_obs × k]` on GPU, modified in-place
/// - `mc`: correction vector `[k]` on GPU
/// - `n_obs`: number of rows
/// - `k`: number of columns
///
/// The kernel uses 256 threads per block with one thread per element.
/// Total threads = `n_obs × k`.
pub fn mean_correct_gpu(
    dev: &GpuDevice,
    y: &mut CudaSlice<f32>,
    mc: &CudaSlice<f32>,
    n_obs: usize,
    k: usize,
) -> Result<(), GpuError> {
    let total = n_obs * k;
    if total == 0 {
        return Ok(());
    }

    let module = dev.load_module_cached(MEAN_CORRECT_PTX)?;
    let func = module
        .load_function("mean_correct_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("mean_correct_kernel: {e}")))?;

    let n_obs_i32 = n_obs as i32;
    let k_i32 = k as i32;

    let threads: u32 = 256;
    let blocks = (total as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(y)
            .arg(mc)
            .arg(&n_obs_i32)
            .arg(&k_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("mean_correct_kernel: {e}")))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_correct() {
        let dev = require_gpu!();

        let n_obs = 4;
        let k = 3;

        // Y (row-major 4×3): each row = [10, 20, 30]
        let y_host: Vec<f32> = vec![
            10.0, 20.0, 30.0, // row 0
            10.0, 20.0, 30.0, // row 1
            10.0, 20.0, 30.0, // row 2
            10.0, 20.0, 30.0, // row 3
        ];
        // mc = [1, 2, 3]
        let mc_host: Vec<f32> = vec![1.0, 2.0, 3.0];

        let mut d_y = dev.htod_copy(&y_host).unwrap();
        let d_mc = dev.htod_copy(&mc_host).unwrap();

        mean_correct_gpu(&dev, &mut d_y, &d_mc, n_obs, k).unwrap();
        dev.synchronize().unwrap();

        let result = dev.dtoh_copy(&d_y).unwrap();

        // Expected: each row = [10-1, 20-2, 30-3] = [9, 18, 27]
        let expected: Vec<f32> = vec![
            9.0, 18.0, 27.0, 9.0, 18.0, 27.0, 9.0, 18.0, 27.0, 9.0, 18.0, 27.0,
        ];

        assert_eq!(result.len(), expected.len());
        for i in 0..result.len() {
            assert!(
                (result[i] - expected[i]).abs() < 1e-5,
                "mean_correct mismatch at {i}: got {}, expected {}",
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_mean_correct_empty() {
        let dev = require_gpu!();

        // Zero-size should not panic
        let mut d_y = dev.alloc_zeros::<f32>(0).unwrap();
        let d_mc = dev.alloc_zeros::<f32>(0).unwrap();
        mean_correct_gpu(&dev, &mut d_y, &d_mc, 0, 0).unwrap();
    }

    #[test]
    fn test_upload_csr_to_gpu() {
        let dev = require_gpu!();

        let csr = scx_sparse::ScxCsr::new(
            (3, 4),
            vec![0, 2, 3, 5],
            vec![0, 2, 1, 0, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap();

        let gpu_csr = upload_csr_to_gpu(&dev, &csr).unwrap();
        assert_eq!(gpu_csr.shape, (3, 4));
        assert_eq!(gpu_csr.indices.len(), 5);
        assert_eq!(gpu_csr.indptr.len(), 4);
    }

    #[test]
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

    #[test]
    fn test_gpu_randomized_pca_phase3_parity() {
        // Phase 3.5 — after the GPU-resident-embedding refactor, verify:
        //
        //   * variance_ratio is non-negative, monotone-descending by PC index,
        //     and sums to ≤ 1 + ε — catches a σ-scaling bug in the new
        //     sgemm + `gpu_scale_columns` tail.
        //   * loadings match `gpu_covariance_pca` on a small fixture (cosine ≥
        //     0.99). This cross-checks the Halko path against the dense-eigh
        //     reference (Phase 2) on shared ground truth, substituting for the
        //     "pre/post refactor snapshot" called out in the spec — we don't
        //     have access to a pre-refactor binary, but Phase 2's GPU
        //     covariance PCA is independently verified against a CPU reference
        //     (see `gpu_pca_covariance::tests`), so it is a valid regression
        //     oracle here.
        //
        // scx-gpu cannot depend on scx-accel (cycle), so CPU-vs-GPU parity
        // against `scx_accel::randomized_pca` is deferred to the Python-side
        // test suite (Phase 7.1).
        use crate::gpu_pca_covariance::gpu_covariance_pca;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        use scx_format::ShardSource;
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
            fn read_shard(&self, i: usize) -> scx_format::Result<ScxCsr> {
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
            assert_eq!(a.len(), k * d);
            assert_eq!(b.len(), k * d);
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
        let csr = random_csr(n_rows, n_cols, 0.08, 314);
        let shards = split_into_shards(&csr, 4);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let gpu_rand = gpu_randomized_pca(&dev, &source, k, 10, 4, true, 42).unwrap();
        let gpu_cov = gpu_covariance_pca(&dev, &source, k, true).unwrap();

        // Loadings must agree (sign-agnostic) with the covariance reference.
        let cos = row_abs_cosine(&gpu_rand.components, &gpu_cov.components, k, n_cols);
        assert!(
            cos > 0.99,
            "Phase-3 gpu_randomized_pca vs gpu_covariance_pca: cosine = {cos}"
        );

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
}
