//! Randomized PCA for sparse CSR matrices with streaming SpMM.
//!
//! Implements the randomized SVD algorithm (Halko, Martinsson, Tropp 2011)
//! with shard-by-shard streaming over [`BackedCsrReader`], enabling PCA on
//! datasets larger than RAM.
//!
//! # Algorithm
//!
//! 1. Compute column means (1 pass over data via `col_sums`)
//! 2. Generate random Gaussian matrix Ω of shape (n_vars, k)
//! 3. Streaming SpMM: Y = (X - μ) @ Ω, computed shard-by-shard
//! 4. QR decomposition: Q = qr(Y).Q
//! 5. Power iteration (configurable): improves accuracy for slowly decaying spectra
//! 6. Form B = (X - μ)^T @ Q (streaming)
//! 7. SVD of B (small matrix): B = Û Σ V^T
//! 8. Recover U = Q @ V, truncate to n_components
//!
//! Peak memory: O(n_obs × k + n_vars × k) where k = n_components + n_oversamples.
//! One decoded shard (~16K × n_vars × 4 bytes) is held at a time.

use faer::Mat;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use scx_format::backed::BackedCsrReader;
use scx_format::total_variance_from_col_sq;
use scx_sparse::ScxCsr;

use crate::error::{AccelError, Result};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of randomized PCA.
#[derive(Debug, Clone)]
pub struct PcaResult {
    /// Cell embeddings: row-major (n_obs × n_components).
    pub embeddings: Vec<f64>,
    /// Principal components (loadings): row-major (n_components × n_vars).
    pub components: Vec<f64>,
    /// Variance explained by each component.
    pub variance_explained: Vec<f64>,
    /// Ratio of variance explained (each / total).
    pub variance_ratio: Vec<f64>,
    /// Column means used for centering (None if zero_center=false).
    pub mean: Option<Vec<f64>>,
    /// Number of components.
    pub n_components: usize,
    /// Number of observations.
    pub n_obs: usize,
    /// Number of variables.
    pub n_vars: usize,
}

// ---------------------------------------------------------------------------
// faer::Mat helpers — convert between row-major Vec<f64> and column-major Mat
// ---------------------------------------------------------------------------

/// Build a `faer::Mat<f64>` from a row-major `Vec<f64>`.
fn dense_from_row_major(data: &[f64], rows: usize, cols: usize) -> Mat<f64> {
    debug_assert_eq!(data.len(), rows * cols);
    let mut mat = Mat::<f64>::zeros(rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            mat[(r, c)] = data[r * cols + c];
        }
    }
    mat
}

/// Extract a `faer::Mat<f64>` into a row-major `Vec<f64>`.
fn mat_to_row_major(mat: &Mat<f64>) -> Vec<f64> {
    let (rows, cols) = (mat.nrows(), mat.ncols());
    let mut data = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            data[r * cols + c] = mat[(r, c)];
        }
    }
    data
}

/// Economy QR: return Q with orthonormal columns.
fn qr_thin_q(mat: &Mat<f64>) -> Mat<f64> {
    let qr = mat.qr();
    qr.compute_thin_Q()
}

/// Thin SVD: A = U Σ V^T. Returns (U, σ, V^T).
fn thin_svd_decomp(mat: &Mat<f64>) -> Result<(Mat<f64>, Vec<f64>, Mat<f64>)> {
    let svd = mat
        .thin_svd()
        .map_err(|e| AccelError::LinAlg(format!("SVD failed: {e:?}")))?;
    let k = mat.nrows().min(mat.ncols());

    let u = svd.U().to_owned();

    let s_col = svd.S().column_vector();
    let sigma: Vec<f64> = (0..k).map(|i| s_col[i]).collect();

    let vt = svd.V().transpose().to_owned();

    Ok((u, sigma, vt))
}

// ---------------------------------------------------------------------------
// Public API — streaming from BackedCsrReader
// ---------------------------------------------------------------------------

/// Compute randomized PCA from a backed SCX reader.
///
/// Streams data shard-by-shard — peak memory is one decoded shard plus
/// the working matrices (n_obs × k and n_vars × k where k = n_components + n_oversamples).
///
/// # Arguments
///
/// * `reader` — Backed CSR reader (provides shard-by-shard access)
/// * `n_components` — Number of principal components to compute
/// * `n_oversamples` — Extra dimensions for accuracy (default: 10)
/// * `n_power_iterations` — Power iterations for spectral accuracy (default: 2)
/// * `zero_center` — Whether to mean-center the data (default: true)
/// * `seed` — Random seed for reproducibility
pub fn randomized_pca(
    reader: &BackedCsrReader,
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    seed: u64,
) -> Result<PcaResult> {
    let (n_obs, n_vars) = reader.shape();
    validate_inputs(n_obs, n_vars, n_components)?;

    let k = (n_components + n_oversamples).min(n_vars).min(n_obs);
    // Fused pass: compute column means and sum-of-squares together (1 shard pass)
    let (means, col_sum_sq) = reader.col_means_and_sum_sq(zero_center)?;
    let means_ref = means.as_deref();

    // Step 2: Random Gaussian Ω (n_vars × k)
    let omega = dense_from_row_major(&random_gaussian(n_vars, k, seed), n_vars, k);

    // Step 3 + 4 + 5: Streaming SpMM → QR → power iteration
    let y = streaming_spmm_forward(reader, &omega, means_ref)?;
    let mut q = qr_thin_q(&y);

    for _ in 0..n_power_iterations {
        let b = streaming_spmm_transpose(reader, &q, means_ref)?;
        let q_b = qr_thin_q(&b);
        let y = streaming_spmm_forward(reader, &q_b, means_ref)?;
        q = qr_thin_q(&y);
    }

    // Step 6: B = (X - μ)^T @ Q
    let b = streaming_spmm_transpose(reader, &q, means_ref)?;

    // Step 7 + 8: SVD of B, recover embeddings (uses pre-computed col_sum_sq — no extra pass)
    let total_var = total_variance_from_col_sq(&col_sum_sq, means_ref, n_obs);
    build_pca_result(&q, &b, &means, n_components, n_obs, n_vars, total_var)
}

/// Compute randomized PCA from an in-memory ScxCsr matrix.
///
/// Convenience wrapper for testing and small datasets.
pub fn randomized_pca_inmemory(
    csr: &ScxCsr,
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    seed: u64,
) -> Result<PcaResult> {
    let (n_obs, n_vars) = (csr.n_rows(), csr.n_cols());
    validate_inputs(n_obs, n_vars, n_components)?;

    let k = (n_components + n_oversamples).min(n_vars).min(n_obs);

    let means = if zero_center {
        let sums = csr.col_sums();
        Some(sums.iter().map(|&s| s / n_obs as f64).collect::<Vec<f64>>())
    } else {
        None
    };
    let means_ref = means.as_deref();

    let omega = dense_from_row_major(&random_gaussian(n_vars, k, seed), n_vars, k);
    let y = spmm_forward_csr(csr, &omega, means_ref);
    let mut q = qr_thin_q(&y);

    for _ in 0..n_power_iterations {
        let b = spmm_transpose_csr(csr, &q, means_ref);
        let q_b = qr_thin_q(&b);
        let y = spmm_forward_csr(csr, &q_b, means_ref);
        q = qr_thin_q(&y);
    }

    let b = spmm_transpose_csr(csr, &q, means_ref);
    let total_var = compute_total_variance_inmemory(csr, means_ref);
    build_pca_result(&q, &b, &means, n_components, n_obs, n_vars, total_var)
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn validate_inputs(n_obs: usize, n_vars: usize, n_components: usize) -> Result<()> {
    if n_components == 0 {
        return Err(AccelError::InvalidInput(
            "n_components must be > 0".to_string(),
        ));
    }
    if n_obs == 0 || n_vars == 0 {
        return Err(AccelError::InvalidInput(
            "matrix must be non-empty".to_string(),
        ));
    }
    if n_components > n_obs.min(n_vars) {
        return Err(AccelError::InvalidInput(format!(
            "n_components ({n_components}) exceeds matrix rank bound min(n_obs, n_vars) = {}",
            n_obs.min(n_vars)
        )));
    }
    Ok(())
}

// NOTE: `compute_means_and_col_sq` has been replaced by
// `BackedCsrReader::col_means_and_sum_sq()` in scx-format.
// `compute_total_variance_from_col_sq` has been replaced by
// `scx_format::total_variance_from_col_sq()`.

/// Generate a random Gaussian matrix (rows × cols), row-major.
fn random_gaussian(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = StandardNormal;
    (0..rows * cols).map(|_| normal.sample(&mut rng)).collect()
}

/// Streaming forward SpMM: Y = (X - μ) @ M, shard-by-shard.
///
/// X is (n_obs × n_vars) stored as sharded CSR.
/// M is Mat<f64> (n_vars × k).
/// Returns Mat<f64> (n_obs × k).
fn streaming_spmm_forward(
    reader: &BackedCsrReader,
    m: &Mat<f64>,
    means: Option<&[f64]>,
) -> Result<Mat<f64>> {
    let (n_obs, n_vars) = reader.shape();
    let k = m.ncols();
    debug_assert_eq!(m.nrows(), n_vars);

    // Extract M to row-major for cache-friendly SpMM kernel access
    let m_data = mat_to_row_major(m);

    let mut y = vec![0.0f64; n_obs * k];

    // Pre-compute means^T @ M = (1 × k) to subtract from each row of Y
    let mean_correction: Option<Vec<f64>> = means.map(|mu| {
        let mut mc = vec![0.0f64; k];
        for v in 0..n_vars {
            for j in 0..k {
                mc[j] += mu[v] * m_data[v * k + j];
            }
        }
        mc
    });

    let n_shards = reader.index().n_shards();
    let mut global_row = 0usize;

    for shard_idx in 0..n_shards {
        let csr = reader.read_shard_cached(shard_idx)?;
        let shard_rows = csr.n_rows();

        spmm_forward_into(
            &csr,
            &m_data,
            k,
            &mut y,
            global_row,
            mean_correction.as_deref(),
        );
        global_row += shard_rows;
    }

    Ok(dense_from_row_major(&y, n_obs, k))
}

/// Streaming transpose SpMM: Z = (X - μ)^T @ Q, shard-by-shard.
///
/// Q is Mat<f64> (n_obs × k).
/// Returns Mat<f64> (n_vars × k).
fn streaming_spmm_transpose(
    reader: &BackedCsrReader,
    q: &Mat<f64>,
    means: Option<&[f64]>,
) -> Result<Mat<f64>> {
    let (n_obs, n_vars) = reader.shape();
    let k = q.ncols();
    debug_assert_eq!(q.nrows(), n_obs);

    // Extract Q to row-major for cache-friendly access
    let q_data = mat_to_row_major(q);

    let mut z = vec![0.0f64; n_vars * k];

    let n_shards = reader.index().n_shards();
    let mut global_row = 0usize;

    for shard_idx in 0..n_shards {
        let csr = reader.read_shard_cached(shard_idx)?;
        let shard_rows = csr.n_rows();

        // Z[col, :] += X[row, col] * Q[row, :]
        #[allow(clippy::needless_range_loop)]
        for r in 0..shard_rows {
            let start = csr.indptr[r] as usize;
            let end = csr.indptr[r + 1] as usize;
            let q_offset = (global_row + r) * k;

            for idx in start..end {
                let col = csr.indices[idx] as usize;
                let val = csr.data[idx] as f64;
                let z_offset = col * k;
                for j in 0..k {
                    z[z_offset + j] += val * q_data[q_offset + j];
                }
            }
        }

        global_row += shard_rows;
    }

    // Mean centering correction: Z -= μ @ (1^T @ Q)
    if let Some(mu) = means {
        let mut sum_q = vec![0.0f64; k];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n_obs {
            for j in 0..k {
                sum_q[j] += q_data[i * k + j];
            }
        }
        #[allow(clippy::needless_range_loop)]
        for v in 0..n_vars {
            for j in 0..k {
                z[v * k + j] -= mu[v] * sum_q[j];
            }
        }
    }

    Ok(dense_from_row_major(&z, n_vars, k))
}

/// In-memory forward SpMM: Y = (X - μ) @ M using a single CSR.
fn spmm_forward_csr(csr: &ScxCsr, m: &Mat<f64>, means: Option<&[f64]>) -> Mat<f64> {
    let n_obs = csr.n_rows();
    let n_vars = csr.n_cols();
    let k = m.ncols();

    let m_data = mat_to_row_major(m);

    let mean_correction: Option<Vec<f64>> = means.map(|mu| {
        let mut mc = vec![0.0f64; k];
        for v in 0..n_vars {
            for j in 0..k {
                mc[j] += mu[v] * m_data[v * k + j];
            }
        }
        mc
    });

    let mut y = vec![0.0f64; n_obs * k];
    spmm_forward_into(csr, &m_data, k, &mut y, 0, mean_correction.as_deref());
    dense_from_row_major(&y, n_obs, k)
}

/// In-memory transpose SpMM: Z = (X - μ)^T @ Q using a single CSR.
fn spmm_transpose_csr(csr: &ScxCsr, q: &Mat<f64>, means: Option<&[f64]>) -> Mat<f64> {
    let n_obs = csr.n_rows();
    let n_vars = csr.n_cols();
    let k = q.ncols();
    let q_data = mat_to_row_major(q);
    let mut z = vec![0.0f64; n_vars * k];

    #[allow(clippy::needless_range_loop)]
    for r in 0..n_obs {
        let start = csr.indptr[r] as usize;
        let end = csr.indptr[r + 1] as usize;
        let q_offset = r * k;

        for idx in start..end {
            let col = csr.indices[idx] as usize;
            let val = csr.data[idx] as f64;
            let z_offset = col * k;
            for j in 0..k {
                z[z_offset + j] += val * q_data[q_offset + j];
            }
        }
    }

    if let Some(mu) = means {
        let mut sum_q = vec![0.0f64; k];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n_obs {
            for j in 0..k {
                sum_q[j] += q_data[i * k + j];
            }
        }
        #[allow(clippy::needless_range_loop)]
        for v in 0..n_vars {
            for j in 0..k {
                z[v * k + j] -= mu[v] * sum_q[j];
            }
        }
    }

    dense_from_row_major(&z, n_vars, k)
}

/// Shared SpMM-forward kernel: accumulates X_shard @ M into y[global_row*k..].
///
/// Uses rayon intra-shard parallelism when the workload is large enough
/// (shard_rows × k > 10,000) since each output row is disjoint.
fn spmm_forward_into(
    csr: &ScxCsr,
    m_data: &[f64], // n_vars × k, row-major
    k: usize,
    y: &mut [f64], // n_obs × k, row-major
    global_row: usize,
    mean_correction: Option<&[f64]>, // length k
) {
    let shard_rows = csr.n_rows();
    let y_chunk = &mut y[global_row * k..(global_row + shard_rows) * k];

    // Threshold: only use rayon when work per shard is substantial
    if shard_rows * k > 10_000 {
        y_chunk
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(r, y_row)| {
                spmm_forward_row(csr, m_data, k, y_row, r, mean_correction);
            });
    } else {
        for (r, y_row) in y_chunk.chunks_mut(k).enumerate() {
            spmm_forward_row(csr, m_data, k, y_row, r, mean_correction);
        }
    }
}

/// Process one CSR row: y_row[j] += Σ val × M[col, j] − mc[j].
#[inline]
#[allow(clippy::needless_range_loop)]
fn spmm_forward_row(
    csr: &ScxCsr,
    m_data: &[f64],
    k: usize,
    y_row: &mut [f64],
    r: usize,
    mean_correction: Option<&[f64]>,
) {
    let start = csr.indptr[r] as usize;
    let end = csr.indptr[r + 1] as usize;
    for idx in start..end {
        let col = csr.indices[idx] as usize;
        let val = csr.data[idx] as f64;
        let m_offset = col * k;
        for j in 0..k {
            y_row[j] += val * m_data[m_offset + j];
        }
    }
    if let Some(mc) = mean_correction {
        for j in 0..k {
            y_row[j] -= mc[j];
        }
    }
}

// NOTE: `compute_total_variance_from_col_sq` has been moved to
// `scx_format::backed::total_variance_from_col_sq()` (re-exported
// from `scx_format::total_variance_from_col_sq`).

/// Total variance (in-memory).
fn compute_total_variance_inmemory(csr: &ScxCsr, means: Option<&[f64]>) -> f64 {
    let n_obs = csr.n_rows();
    let n_vars = csr.n_cols();

    if let Some(mu) = means {
        let mut total = 0.0f64;
        for r in 0..n_obs {
            let start = csr.indptr[r] as usize;
            let end = csr.indptr[r + 1] as usize;
            for j in start..end {
                let v = csr.data[j] as f64 - mu[csr.indices[j] as usize];
                total += v * v;
            }
        }
        // Add zero contributions per column
        let col_nnz = csr.col_nnz();
        for c in 0..n_vars {
            let n_zeros = n_obs as i64 - col_nnz[c];
            total += n_zeros as f64 * mu[c] * mu[c];
        }
        total / (n_obs as f64 - 1.0).max(1.0)
    } else {
        let total: f64 = csr.data.iter().map(|&v| (v as f64) * (v as f64)).sum();
        total / (n_obs as f64 - 1.0).max(1.0)
    }
}

/// Build PcaResult from Q, B, and SVD.
#[allow(clippy::too_many_arguments)]
fn build_pca_result(
    q: &Mat<f64>,
    b: &Mat<f64>,
    means: &Option<Vec<f64>>,
    n_components: usize,
    n_obs: usize,
    n_vars: usize,
    total_var: f64,
) -> Result<PcaResult> {
    let (u_hat, sigma, vt) = thin_svd_decomp(b)?;

    // Embeddings = Q @ V * Σ (take first n_components columns)
    let v = vt.transpose().to_owned();
    let embeddings_full: Mat<f64> = q * &v; // faer's optimized GEMM

    let mut scaled_embeddings = vec![0.0f64; n_obs * n_components];
    for i in 0..n_obs {
        for j in 0..n_components {
            scaled_embeddings[i * n_components + j] = embeddings_full[(i, j)] * sigma[j];
        }
    }

    // Components: rows of U_hat^T → (n_components × n_vars)
    let mut components = vec![0.0f64; n_components * n_vars];
    for pc in 0..n_components {
        for v in 0..n_vars {
            components[pc * n_vars + v] = u_hat[(v, pc)];
        }
    }

    // Variance explained = σ² / (n-1)
    let variance_explained: Vec<f64> = sigma
        .iter()
        .take(n_components)
        .map(|&s| s * s / (n_obs as f64 - 1.0).max(1.0))
        .collect();

    let variance_ratio: Vec<f64> = if total_var > 0.0 {
        variance_explained.iter().map(|&v| v / total_var).collect()
    } else {
        vec![0.0; n_components]
    };

    Ok(PcaResult {
        embeddings: scaled_embeddings,
        components,
        variance_explained,
        variance_ratio,
        mean: means.clone(),
        n_components,
        n_obs,
        n_vars,
    })
}

// ---------------------------------------------------------------------------
// GPU PCA dispatch (behind "gpu" feature)
// ---------------------------------------------------------------------------

/// GPU-accelerated randomized PCA from a backed SCX reader.
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
/// * `reader` — Backed CSR reader (provides shard-by-shard access)
/// * `n_components` — Number of principal components to compute
/// * `n_oversamples` — Extra dimensions for accuracy (default: 10)
/// * `n_power_iterations` — Power iterations for spectral accuracy (default: 2)
/// * `zero_center` — Whether to mean-center the data (default: true)
/// * `seed` — Random seed for reproducibility
#[cfg(feature = "gpu")]
pub fn randomized_pca_gpu(
    device_id: usize,
    reader: &BackedCsrReader,
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    seed: u64,
) -> Result<PcaResult> {
    let dev = scx_gpu::GpuDevice::new(device_id)
        .map_err(|e| AccelError::LinAlg(format!("GPU init failed: {e}")))?;

    let gpu_result = scx_gpu::gpu_randomized_pca(
        &dev,
        reader,
        n_components,
        n_oversamples,
        n_power_iterations,
        zero_center,
        seed,
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
    })
}

/// Check whether a GPU is available for GPU-accelerated PCA.
///
/// Returns `true` if at least one CUDA device is found.
#[cfg(feature = "gpu")]
pub fn gpu_available() -> bool {
    scx_gpu::GpuDevice::count().map_or(false, |n| n > 0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 10×5 sparse matrix for PCA tests.
    fn test_csr_10x5() -> ScxCsr {
        ScxCsr::new(
            (10, 5),
            vec![0, 2, 4, 7, 9, 12, 14, 17, 19, 22, 24],
            vec![
                0, 2, 1, 3, 0, 2, 4, 1, 4, 0, 2, 3, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3,
            ],
            vec![
                1.0, 3.0, 2.0, 4.0, 5.0, 1.0, 2.0, 3.0, 6.0, 2.0, 4.0, 1.0, 1.0, 3.0, 3.0, 2.0,
                5.0, 4.0, 2.0, 1.0, 5.0, 3.0, 2.0, 1.0,
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_pca_basic() {
        let csr = test_csr_10x5();
        let result = randomized_pca_inmemory(&csr, 3, 5, 2, true, 42).unwrap();

        assert_eq!(result.n_components, 3);
        assert_eq!(result.n_obs, 10);
        assert_eq!(result.n_vars, 5);
        assert_eq!(result.embeddings.len(), 30);
        assert_eq!(result.components.len(), 15);
        assert_eq!(result.variance_explained.len(), 3);
        assert_eq!(result.variance_ratio.len(), 3);
        assert!(result.mean.is_some());

        // Variance explained: positive and non-increasing
        for &ve in &result.variance_explained {
            assert!(ve > 0.0, "variance_explained should be positive");
        }
        for w in result.variance_explained.windows(2) {
            assert!(w[0] >= w[1], "variance_explained should be non-increasing");
        }

        // Variance ratio sum ∈ (0, 1]
        let ratio_sum: f64 = result.variance_ratio.iter().sum();
        assert!(
            ratio_sum > 0.0 && ratio_sum <= 1.0 + 1e-10,
            "ratio sum = {ratio_sum}"
        );
    }

    #[test]
    fn test_pca_no_center() {
        let csr = test_csr_10x5();
        let result = randomized_pca_inmemory(&csr, 2, 5, 2, false, 42).unwrap();

        assert_eq!(result.n_components, 2);
        assert!(result.mean.is_none());
        for &ve in &result.variance_explained {
            assert!(ve > 0.0);
        }
    }

    #[test]
    fn test_pca_components_orthogonal() {
        let csr = test_csr_10x5();
        let result = randomized_pca_inmemory(&csr, 3, 5, 2, true, 42).unwrap();

        for i in 0..result.n_components {
            for j in (i + 1)..result.n_components {
                let dot: f64 = (0..result.n_vars)
                    .map(|v| {
                        result.components[i * result.n_vars + v]
                            * result.components[j * result.n_vars + v]
                    })
                    .sum();
                assert!(dot.abs() < 0.1, "PC{i} · PC{j} = {dot}, expected ~0");
            }
        }
    }

    #[test]
    fn test_pca_error_zero_components() {
        let csr = test_csr_10x5();
        assert!(randomized_pca_inmemory(&csr, 0, 5, 2, true, 42).is_err());
    }

    #[test]
    fn test_pca_error_too_many_components() {
        let csr = test_csr_10x5();
        // 10×5 matrix → max rank = 5, requesting 6
        assert!(randomized_pca_inmemory(&csr, 6, 5, 2, true, 42).is_err());
    }

    #[test]
    fn test_spmm_matches_dense() {
        let csr = test_csr_10x5();
        let dense = csr.to_dense().unwrap();
        let n_obs = 10;
        let n_vars = 5;
        let k = 3;

        let m = dense_from_row_major(&random_gaussian(n_vars, k, 42), n_vars, k);
        let y_sparse = spmm_forward_csr(&csr, &m, None);
        let y_sparse_data = mat_to_row_major(&y_sparse);
        let m_data = mat_to_row_major(&m);

        // Dense matmul reference
        let mut y_dense = vec![0.0f64; n_obs * k];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n_obs {
            for j in 0..k {
                for v in 0..n_vars {
                    y_dense[i * k + j] += dense[i * n_vars + v] as f64 * m_data[v * k + j];
                }
            }
        }

        for i in 0..n_obs * k {
            assert!(
                (y_sparse_data[i] - y_dense[i]).abs() < 1e-10,
                "mismatch at {i}: {} vs {}",
                y_sparse_data[i],
                y_dense[i]
            );
        }
    }

    #[test]
    fn test_qr_orthonormal() {
        let data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let mat = dense_from_row_major(&data, 4, 2);
        let q = qr_thin_q(&mat);

        assert_eq!(q.nrows(), 4);
        assert_eq!(q.ncols(), 2);

        for c in 0..2 {
            let norm: f64 = (0..4).map(|r| q[(r, c)].powi(2)).sum();
            assert!((norm - 1.0).abs() < 1e-10, "col {c} norm = {norm}");
        }

        let dot: f64 = (0..4).map(|r| q[(r, 0)] * q[(r, 1)]).sum();
        assert!(dot.abs() < 1e-10, "cols should be orthogonal, dot={dot}");
    }

    #[test]
    fn test_svd_basic() {
        let data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mat = dense_from_row_major(&data, 3, 2);
        let (u, sigma, vt) = thin_svd_decomp(&mat).unwrap();

        assert_eq!(u.nrows(), 3);
        assert_eq!(u.ncols(), 2);
        assert_eq!(sigma.len(), 2);
        assert_eq!(vt.nrows(), 2);
        assert_eq!(vt.ncols(), 2);
        assert!(sigma[0] > 0.0);
        assert!(sigma[1] > 0.0);
        assert!(sigma[0] >= sigma[1]);
    }
}
