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

#![allow(clippy::needless_range_loop)]

use faer::Mat;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use scx_format::backed::BackedCsrReader;
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
// Simple row-major dense matrix with shape tracking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct DenseMat {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl DenseMat {
    fn from_data(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        Self { data, rows, cols }
    }

    fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /// Economy QR: return Q with orthonormal columns.
    fn qr_q(&self) -> Result<DenseMat> {
        let mut mat = Mat::<f64>::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                mat[(r, c)] = self.data[r * self.cols + c];
            }
        }

        let qr = mat.qr();
        let q_thin = qr.compute_thin_Q();

        let mut result = vec![0.0f64; self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                result[r * self.cols + c] = q_thin[(r, c)];
            }
        }

        Ok(DenseMat::from_data(result, self.rows, self.cols))
    }

    /// Thin SVD: A = U Σ V^T. Returns (U, σ, V^T).
    fn thin_svd(&self) -> Result<(DenseMat, Vec<f64>, DenseMat)> {
        let mut mat = Mat::<f64>::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                mat[(r, c)] = self.data[r * self.cols + c];
            }
        }

        let svd = mat
            .thin_svd()
            .map_err(|e| AccelError::LinAlg(format!("SVD failed: {e:?}")))?;
        let k = self.rows.min(self.cols);

        // U: rows × k
        let u_mat = svd.U();
        let mut u = vec![0.0f64; self.rows * k];
        for r in 0..self.rows {
            for c in 0..k {
                u[r * k + c] = u_mat[(r, c)];
            }
        }

        // Singular values
        let s_col = svd.S().column_vector();
        let sigma: Vec<f64> = (0..k).map(|i| s_col[i]).collect();

        // V^T: k × cols
        let v_mat = svd.V();
        let vt_mat = v_mat.transpose();
        let mut vt = vec![0.0f64; k * self.cols];
        for r in 0..k {
            for c in 0..self.cols {
                vt[r * self.cols + c] = vt_mat[(r, c)];
            }
        }

        Ok((
            DenseMat::from_data(u, self.rows, k),
            sigma,
            DenseMat::from_data(vt, k, self.cols),
        ))
    }

    fn transpose(&self) -> DenseMat {
        let mut t = vec![0.0f64; self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                t[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        DenseMat::from_data(t, self.cols, self.rows)
    }

    fn mul(&self, other: &DenseMat) -> DenseMat {
        debug_assert_eq!(self.cols, other.rows);
        let m = self.rows;
        let p = self.cols;
        let n = other.cols;
        let mut c = vec![0.0f64; m * n];
        for i in 0..m {
            for k in 0..p {
                let a_ik = self.data[i * p + k];
                if a_ik == 0.0 {
                    continue;
                }
                for j in 0..n {
                    c[i * n + j] += a_ik * other.data[k * n + j];
                }
            }
        }
        DenseMat::from_data(c, m, n)
    }
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
    validate_inputs(n_obs, n_vars, n_components, n_oversamples)?;

    let k = (n_components + n_oversamples).min(n_vars).min(n_obs);
    let means = compute_means_if_needed(reader, zero_center)?;
    let means_ref = means.as_deref();

    // Step 2: Random Gaussian Ω (n_vars × k)
    let omega = DenseMat::from_data(random_gaussian(n_vars, k, seed), n_vars, k);

    // Step 3 + 4 + 5: Streaming SpMM → QR → power iteration
    let y = streaming_spmm_forward(reader, &omega, means_ref)?;
    let mut q = y.qr_q()?;

    for _ in 0..n_power_iterations {
        let b = streaming_spmm_transpose(reader, &q, means_ref)?;
        let q_b = b.qr_q()?;
        let y = streaming_spmm_forward(reader, &q_b, means_ref)?;
        q = y.qr_q()?;
    }

    // Step 6: B = (X - μ)^T @ Q
    let b = streaming_spmm_transpose(reader, &q, means_ref)?;

    // Step 7 + 8: SVD of B, recover embeddings
    let total_var = compute_total_variance_streaming(reader, means_ref)?;
    build_pca_result(&q, &b, &means, n_components, k, n_obs, n_vars, total_var)
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
    validate_inputs(n_obs, n_vars, n_components, n_oversamples)?;

    let k = (n_components + n_oversamples).min(n_vars).min(n_obs);

    let means = if zero_center {
        let sums = csr.col_sums();
        Some(sums.iter().map(|&s| s / n_obs as f64).collect::<Vec<f64>>())
    } else {
        None
    };
    let means_ref = means.as_deref();

    let omega = DenseMat::from_data(random_gaussian(n_vars, k, seed), n_vars, k);
    let y = spmm_forward_csr(csr, &omega, means_ref);
    let mut q = y.qr_q()?;

    for _ in 0..n_power_iterations {
        let b = spmm_transpose_csr(csr, &q, means_ref);
        let q_b = b.qr_q()?;
        let y = spmm_forward_csr(csr, &q_b, means_ref);
        q = y.qr_q()?;
    }

    let b = spmm_transpose_csr(csr, &q, means_ref);
    let total_var = compute_total_variance_inmemory(csr, means_ref);
    build_pca_result(&q, &b, &means, n_components, k, n_obs, n_vars, total_var)
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn validate_inputs(
    n_obs: usize,
    n_vars: usize,
    n_components: usize,
    _n_oversamples: usize,
) -> Result<()> {
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

fn compute_means_if_needed(
    reader: &BackedCsrReader,
    zero_center: bool,
) -> Result<Option<Vec<f64>>> {
    if !zero_center {
        return Ok(None);
    }
    let n_obs = reader.n_obs();
    let col_sums = reader.col_sums()?;
    Ok(Some(col_sums.iter().map(|&s| s / n_obs as f64).collect()))
}

/// Generate a random Gaussian matrix (rows × cols), row-major.
fn random_gaussian(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = StandardNormal;
    (0..rows * cols).map(|_| normal.sample(&mut rng)).collect()
}

/// Streaming forward SpMM: Y = (X - μ) @ M, shard-by-shard.
///
/// X is (n_obs × n_vars) stored as sharded CSR.
/// M is DenseMat (n_vars × k).
/// Returns DenseMat (n_obs × k).
fn streaming_spmm_forward(
    reader: &BackedCsrReader,
    m: &DenseMat,
    means: Option<&[f64]>,
) -> Result<DenseMat> {
    let (n_obs, n_vars) = reader.shape();
    let k = m.cols;
    debug_assert_eq!(m.rows, n_vars);

    let mut y = vec![0.0f64; n_obs * k];

    // Pre-compute means^T @ M = (1 × k) to subtract from each row of Y
    let mean_correction: Option<Vec<f64>> = means.map(|mu| {
        let mut mc = vec![0.0f64; k];
        for v in 0..n_vars {
            for j in 0..k {
                mc[j] += mu[v] * m.data[v * k + j];
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
            &m.data,
            k,
            &mut y,
            global_row,
            mean_correction.as_deref(),
        );
        global_row += shard_rows;
    }

    Ok(DenseMat::from_data(y, n_obs, k))
}

/// Streaming transpose SpMM: Z = (X - μ)^T @ Q, shard-by-shard.
///
/// Q is DenseMat (n_obs × k).
/// Returns DenseMat (n_vars × k).
fn streaming_spmm_transpose(
    reader: &BackedCsrReader,
    q: &DenseMat,
    means: Option<&[f64]>,
) -> Result<DenseMat> {
    let (n_obs, n_vars) = reader.shape();
    let k = q.cols;
    debug_assert_eq!(q.rows, n_obs);

    let mut z = vec![0.0f64; n_vars * k];

    let n_shards = reader.index().n_shards();
    let mut global_row = 0usize;

    for shard_idx in 0..n_shards {
        let csr = reader.read_shard_cached(shard_idx)?;
        let shard_rows = csr.n_rows();

        // Z[col, :] += X[row, col] * Q[row, :]
        for r in 0..shard_rows {
            let start = csr.indptr[r] as usize;
            let end = csr.indptr[r + 1] as usize;
            let q_offset = (global_row + r) * k;

            for idx in start..end {
                let col = csr.indices[idx] as usize;
                let val = csr.data[idx] as f64;
                let z_offset = col * k;
                for j in 0..k {
                    z[z_offset + j] += val * q.data[q_offset + j];
                }
            }
        }

        global_row += shard_rows;
    }

    // Mean centering correction: Z -= μ @ (1^T @ Q)
    if let Some(mu) = means {
        let mut sum_q = vec![0.0f64; k];
        for i in 0..n_obs {
            for j in 0..k {
                sum_q[j] += q.data[i * k + j];
            }
        }
        for v in 0..n_vars {
            for j in 0..k {
                z[v * k + j] -= mu[v] * sum_q[j];
            }
        }
    }

    Ok(DenseMat::from_data(z, n_vars, k))
}

/// In-memory forward SpMM: Y = (X - μ) @ M using a single CSR.
fn spmm_forward_csr(csr: &ScxCsr, m: &DenseMat, means: Option<&[f64]>) -> DenseMat {
    let n_obs = csr.n_rows();
    let n_vars = csr.n_cols();
    let k = m.cols;

    let mean_correction: Option<Vec<f64>> = means.map(|mu| {
        let mut mc = vec![0.0f64; k];
        for v in 0..n_vars {
            for j in 0..k {
                mc[j] += mu[v] * m.data[v * k + j];
            }
        }
        mc
    });

    let mut y = vec![0.0f64; n_obs * k];
    spmm_forward_into(csr, &m.data, k, &mut y, 0, mean_correction.as_deref());
    DenseMat::from_data(y, n_obs, k)
}

/// In-memory transpose SpMM: Z = (X - μ)^T @ Q using a single CSR.
fn spmm_transpose_csr(csr: &ScxCsr, q: &DenseMat, means: Option<&[f64]>) -> DenseMat {
    let n_obs = csr.n_rows();
    let n_vars = csr.n_cols();
    let k = q.cols;
    let mut z = vec![0.0f64; n_vars * k];

    for r in 0..n_obs {
        let start = csr.indptr[r] as usize;
        let end = csr.indptr[r + 1] as usize;
        let q_offset = r * k;

        for idx in start..end {
            let col = csr.indices[idx] as usize;
            let val = csr.data[idx] as f64;
            let z_offset = col * k;
            for j in 0..k {
                z[z_offset + j] += val * q.data[q_offset + j];
            }
        }
    }

    if let Some(mu) = means {
        let mut sum_q = vec![0.0f64; k];
        for i in 0..n_obs {
            for j in 0..k {
                sum_q[j] += q.data[i * k + j];
            }
        }
        for v in 0..n_vars {
            for j in 0..k {
                z[v * k + j] -= mu[v] * sum_q[j];
            }
        }
    }

    DenseMat::from_data(z, n_vars, k)
}

/// Shared SpMM-forward kernel: accumulates X_shard @ M into y[global_row*k..].
fn spmm_forward_into(
    csr: &ScxCsr,
    m_data: &[f64], // n_vars × k, row-major
    k: usize,
    y: &mut [f64], // n_obs × k, row-major
    global_row: usize,
    mean_correction: Option<&[f64]>, // length k
) {
    let shard_rows = csr.n_rows();
    for r in 0..shard_rows {
        let start = csr.indptr[r] as usize;
        let end = csr.indptr[r + 1] as usize;
        let y_offset = (global_row + r) * k;

        for idx in start..end {
            let col = csr.indices[idx] as usize;
            let val = csr.data[idx] as f64;
            let m_offset = col * k;
            for j in 0..k {
                y[y_offset + j] += val * m_data[m_offset + j];
            }
        }

        if let Some(mc) = mean_correction {
            for j in 0..k {
                y[y_offset + j] -= mc[j];
            }
        }
    }
}

/// Streaming column sum-of-squares: Σ x_{ij}^2 per column.
fn streaming_col_sum_of_squares(reader: &BackedCsrReader) -> Result<Vec<f64>> {
    let n_vars = reader.n_vars();
    let n_shards = reader.index().n_shards();
    let mut sums = vec![0.0f64; n_vars];

    for shard_idx in 0..n_shards {
        let csr = reader.read_shard_cached(shard_idx)?;
        for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
            let v = val as f64;
            sums[col as usize] += v * v;
        }
    }

    Ok(sums)
}

/// Total variance (streaming, backed).
fn compute_total_variance_streaming(
    reader: &BackedCsrReader,
    means: Option<&[f64]>,
) -> Result<f64> {
    let n_obs = reader.n_obs();
    let _n_vars = reader.n_vars();
    let col_sum_sq = streaming_col_sum_of_squares(reader)?;

    let total = if let Some(mu) = means {
        col_sum_sq
            .iter()
            .zip(mu.iter())
            .map(|(&sq, &m)| sq - n_obs as f64 * m * m)
            .sum::<f64>()
    } else {
        col_sum_sq.iter().sum::<f64>()
    };

    Ok(total / (n_obs as f64 - 1.0).max(1.0))
}

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
    q: &DenseMat,
    b: &DenseMat,
    means: &Option<Vec<f64>>,
    n_components: usize,
    _k: usize,
    n_obs: usize,
    n_vars: usize,
    total_var: f64,
) -> Result<PcaResult> {
    let (u_hat, sigma, vt) = b.thin_svd()?;

    // Embeddings = Q @ V * Σ (take first n_components columns)
    let v = vt.transpose();
    let embeddings_full = q.mul(&v); // n_obs × k

    let mut scaled_embeddings = vec![0.0f64; n_obs * n_components];
    for i in 0..n_obs {
        for j in 0..n_components {
            scaled_embeddings[i * n_components + j] = embeddings_full.get(i, j) * sigma[j];
        }
    }

    // Components: rows of U_hat^T → (n_components × n_vars)
    let mut components = vec![0.0f64; n_components * n_vars];
    for pc in 0..n_components {
        for v in 0..n_vars {
            components[pc * n_vars + v] = u_hat.get(v, pc);
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

        let m = DenseMat::from_data(random_gaussian(n_vars, k, 42), n_vars, k);
        let y_sparse = spmm_forward_csr(&csr, &m, None);

        // Dense matmul reference
        let mut y_dense = vec![0.0f64; n_obs * k];
        for i in 0..n_obs {
            for j in 0..k {
                for v in 0..n_vars {
                    y_dense[i * k + j] += dense[i * n_vars + v] as f64 * m.data[v * k + j];
                }
            }
        }

        for i in 0..n_obs * k {
            assert!(
                (y_sparse.data[i] - y_dense[i]).abs() < 1e-10,
                "mismatch at {i}: {} vs {}",
                y_sparse.data[i],
                y_dense[i]
            );
        }
    }

    #[test]
    fn test_qr_orthonormal() {
        let data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let mat = DenseMat::from_data(data, 4, 2);
        let q = mat.qr_q().unwrap();

        assert_eq!(q.rows, 4);
        assert_eq!(q.cols, 2);

        for c in 0..2 {
            let norm: f64 = (0..4).map(|r| q.get(r, c).powi(2)).sum();
            assert!((norm - 1.0).abs() < 1e-10, "col {c} norm = {norm}");
        }

        let dot: f64 = (0..4).map(|r| q.get(r, 0) * q.get(r, 1)).sum();
        assert!(dot.abs() < 1e-10, "cols should be orthogonal, dot={dot}");
    }

    #[test]
    fn test_svd_basic() {
        let data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mat = DenseMat::from_data(data, 3, 2);
        let (u, sigma, vt) = mat.thin_svd().unwrap();

        assert_eq!(u.rows, 3);
        assert_eq!(u.cols, 2);
        assert_eq!(sigma.len(), 2);
        assert_eq!(vt.rows, 2);
        assert_eq!(vt.cols, 2);
        assert!(sigma[0] > 0.0);
        assert!(sigma[1] > 0.0);
        assert!(sigma[0] >= sigma[1]);
    }
}
