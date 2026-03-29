//! GPU-accelerated randomized PCA pipeline.
//!
//! Provides the mean-correction CUDA kernel and the full GPU PCA pipeline
//! that streams shards from [`BackedCsrReader`], performing SpMM on GPU via
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

use cudarc::driver::safe::CudaSlice;
use cudarc::driver::safe::LaunchConfig;
use cudarc::driver::PushKernelArg;
use faer::Mat;

use scx_format::backed::BackedCsrReader;

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
/// Streams data shard-by-shard from `reader`, performing SpMM on GPU via
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
/// Steps 3-6 stream from BackedCsrReader without materializing full X.
/// Peak GPU memory: ~500 MB for 1M cells (dominated by Y and Q matrices).
#[allow(clippy::too_many_arguments)]
pub fn gpu_randomized_pca(
    dev: &GpuDevice,
    reader: &BackedCsrReader,
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    seed: u64,
) -> Result<GpuPcaResult, GpuError> {
    let (n_obs, n_vars) = reader.shape();

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

    // Create handles
    let cusparse_handle = CusparseHandle::new()?;
    let cusolver_handle = CusolverHandle::new()?;

    // Step 1: Compute column means and sum-of-squares (CPU-side, 1 pass)
    let (means, col_sum_sq) =
        compute_means_and_col_sq(reader, zero_center).map_err(format_scx_error)?;

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
        reader,
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
            reader,
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
            reader,
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
        reader,
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

    // Step 8: Embeddings = Q @ V × Σ (download Q, compute on CPU)
    // Q is (n_obs × k) col-major on GPU
    let q_host_f32 = dev.dtoh_copy(&d_q)?;

    // Convert Q to f64 row-major for multiply
    // Q col-major: Q[i,j] = q_host_f32[j * n_obs + i]
    let mut embeddings = vec![0.0f32; n_obs * n_components];
    for i in 0..n_obs {
        for pc in 0..n_components {
            let mut val = 0.0f64;
            for j in 0..k.min(sigma.len()) {
                // Q[i, j] * V[j, pc] * sigma[pc]
                let q_ij = q_host_f32[j * n_obs + i] as f64;
                let v_jpc = v[(j, pc)];
                val += q_ij * v_jpc;
            }
            embeddings[i * n_components + pc] = (val * sigma[pc]) as f32;
        }
    }

    // Components: rows of U_hat^T → (n_components × n_vars)
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
    let total_var = compute_total_variance_from_col_sq(&col_sum_sq, means.as_deref(), n_obs);

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
///   1. Read shard to host (BackedCsrReader)
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
    reader: &BackedCsrReader,
    d_m: &CudaSlice<f32>, // (n_vars × k) col-major on GPU
    d_means: Option<&CudaSlice<f32>>,
    n_obs: usize,
    n_vars: usize,
    k: usize,
) -> Result<CudaSlice<f32>, GpuError> {
    let mut d_y = dev.alloc_zeros::<f32>(n_obs * k)?;
    let n_shards = reader.index().n_shards();

    // Pre-compute mean correction vector on GPU: mc = M^T @ means (k × 1)
    // Actually mc = means^T @ M = (1 × k), stored as (k,)
    let d_mc: Option<CudaSlice<f32>> = if let Some(d_mu) = d_means {
        // mc[j] = Σ_v means[v] * M[v, j] for v in 0..n_vars
        // M is col-major (n_vars × k): M[v, j] = d_m[j * n_vars + v]
        // This is a simple GEMV: mc = M^T @ means
        // For simplicity, compute on CPU (means is small)
        let mu_host = dev.dtoh_copy(d_mu)?;
        let m_host = dev.dtoh_copy(d_m)?;
        let mut mc = vec![0.0f32; k];
        for j in 0..k {
            for v in 0..n_vars {
                mc[j] += mu_host[v] * m_host[j * n_vars + v];
            }
        }
        Some(dev.htod_copy(&mc)?)
    } else {
        None
    };

    let mut global_row = 0usize;

    for shard_idx in 0..n_shards {
        let csr = reader
            .read_shard_cached(shard_idx)
            .map_err(format_scx_error)?;
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
    reader: &BackedCsrReader,
    d_q: &CudaSlice<f32>, // (n_obs × k) col-major on GPU
    d_means: Option<&CudaSlice<f32>>,
    n_obs: usize,
    n_vars: usize,
    k: usize,
) -> Result<CudaSlice<f32>, GpuError> {
    let mut d_z = dev.alloc_zeros::<f32>(n_vars * k)?;
    let n_shards = reader.index().n_shards();
    let mut global_row = 0usize;

    for shard_idx in 0..n_shards {
        let csr = reader
            .read_shard_cached(shard_idx)
            .map_err(format_scx_error)?;
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
fn upload_csr_to_gpu(dev: &GpuDevice, csr: &scx_sparse::ScxCsr) -> Result<GpuCsr, GpuError> {
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
fn gpu_scatter_colmajor(
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
fn gpu_gather_colmajor(
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
fn gpu_mean_correct_colmajor(
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
fn gpu_column_sums(
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
    let threads: u32 = 256;
    let blocks_x = (m as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks_x, k as u32, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
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
fn gpu_outer_sub(
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

/// Compute column means and column sum-of-squares in one pass over all shards.
///
/// Mirrors `scx_accel::pca::compute_means_and_col_sq` but returns f64 for
/// the variance computation.
fn compute_means_and_col_sq(
    reader: &BackedCsrReader,
    zero_center: bool,
) -> std::result::Result<(Option<Vec<f64>>, Vec<f64>), scx_format::ScxError> {
    let n_vars = reader.n_vars();
    let n_shards = reader.index().n_shards();
    let mut col_sums = vec![0.0f64; n_vars];
    let mut col_sum_sq = vec![0.0f64; n_vars];

    for shard_idx in 0..n_shards {
        let csr = reader.read_shard_cached(shard_idx)?;
        for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
            let v = val as f64;
            col_sums[col as usize] += v;
            col_sum_sq[col as usize] += v * v;
        }
    }

    let means = if zero_center {
        let n_obs = reader.n_obs() as f64;
        Some(col_sums.iter().map(|s| s / n_obs).collect())
    } else {
        None
    };

    Ok((means, col_sum_sq))
}

/// Total variance from pre-computed column sum-of-squares.
/// Var(X_j) = (Σ x²_j - n·μ_j²) / (n-1)
fn compute_total_variance_from_col_sq(
    col_sum_sq: &[f64],
    means: Option<&[f64]>,
    n_obs: usize,
) -> f64 {
    let total = if let Some(mu) = means {
        col_sum_sq
            .iter()
            .zip(mu.iter())
            .map(|(&sq, &m)| sq - n_obs as f64 * m * m)
            .sum::<f64>()
    } else {
        col_sum_sq.iter().sum::<f64>()
    };
    total / (n_obs as f64 - 1.0).max(1.0)
}

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
    use crate::device::GpuDevice;

    macro_rules! require_gpu {
        () => {
            match GpuDevice::new(0) {
                Ok(dev) => dev,
                Err(_) => {
                    eprintln!("CUDA not available — skipping GPU test");
                    return;
                }
            }
        };
    }

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
}
