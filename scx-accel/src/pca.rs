//! Randomized PCA for sparse CSR matrices with streaming SpMM.
//!
//! Implements the randomized SVD algorithm (Halko, Martinsson, Tropp 2011)
//! with shard-by-shard streaming over [`ShardSource`], enabling PCA on
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

use std::cell::RefCell;

use faer::{Mat, MatRef};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use thread_local::ThreadLocal;

use scx_format::total_variance_from_col_sq;
use scx_format::ShardSource;

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
    /// Whether a captured CUDA graph was replayed in the GPU power loop
    /// (Task 2.5). `None` on CPU paths; `Some(true)` only when the device-
    /// resident capture path ran and replayed a graph.
    pub graph_replayed: Option<bool>,
}

// ---------------------------------------------------------------------------
// faer helpers
// ---------------------------------------------------------------------------

/// Economy QR on a row-major buffer: wraps as `MatRef`, computes QR, returns
/// the thin Q factor as an owned `Mat<f64>` (column-major).
fn qr_thin_q_row_major(data: &[f64], rows: usize, cols: usize) -> Mat<f64> {
    let view = MatRef::from_row_major_slice(data, rows, cols);
    let qr = view.qr();
    qr.compute_thin_Q()
}

/// Extract a column-major `Mat<f64>` to a row-major `Vec<f64>`.
///
/// Used only for small matrices (n_vars × k) during power iteration where
/// the forward SpMM kernel needs row-major input. For n_vars=2000, k=60
/// this is only 120K elements (< 1 MB).
fn mat_to_row_major_buf(mat: &Mat<f64>) -> Vec<f64> {
    let (rows, cols) = (mat.nrows(), mat.ncols());
    let mut data = vec![0.0f64; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            data[r * cols + c] = mat[(r, c)];
        }
    }
    data
}

/// Accumulate the sparse outer product X^T @ X directly from CSR nonzeros.
///
/// For each row, iterates pairs of nonzeros and accumulates `C[c1, c2] += v1 * v2`.
/// Exploits symmetry: only computes upper triangle and mirrors to lower.
/// Also accumulates `col_sums` in the same pass (caller provides the slice).
fn sparse_outer_product_accumulate(csr: &ScxCsr, col_sums: &mut [f64], cov: &mut Mat<f64>) {
    let n_rows = csr.n_rows();
    for r in 0..n_rows {
        let start = csr.indptr[r] as usize;
        let end = csr.indptr[r + 1] as usize;
        // Accumulate col_sums
        for idx in start..end {
            let c = csr.indices[idx] as usize;
            let v = csr.data[idx] as f64;
            col_sums[c] += v;
        }
        // Sparse outer product with symmetry exploitation
        for i in start..end {
            let c1 = csr.indices[i] as usize;
            let v1 = csr.data[i] as f64;
            cov[(c1, c1)] += v1 * v1; // diagonal
            for j in (i + 1)..end {
                let c2 = csr.indices[j] as usize;
                let v2 = csr.data[j] as f64;
                let prod = v1 * v2;
                cov[(c1, c2)] += prod;
                cov[(c2, c1)] += prod;
            }
        }
    }
}

/// Parallel sparse outer product accumulation using thread-local `Mat<f64>` accumulators.
///
/// Same algorithm as [`sparse_outer_product_accumulate`] but distributes rows across
/// rayon threads. Each thread gets its own n_vars × n_vars covariance matrix (~30 MB
/// for n_vars=2000) and col_sums vector, then results are reduced by element-wise addition.
fn sparse_outer_product_accumulate_par(csr: &ScxCsr, n_vars: usize) -> (Mat<f64>, Vec<f64>) {
    let n_rows = csr.n_rows();
    let chunk_size = (n_rows / rayon::current_num_threads()).max(256);

    (0..n_rows)
        .into_par_iter()
        .with_min_len(chunk_size)
        .fold(
            || (Mat::<f64>::zeros(n_vars, n_vars), vec![0.0f64; n_vars]),
            |(mut cov_local, mut sums_local), r| {
                let start = csr.indptr[r] as usize;
                let end = csr.indptr[r + 1] as usize;
                for idx in start..end {
                    let c = csr.indices[idx] as usize;
                    let v = csr.data[idx] as f64;
                    sums_local[c] += v;
                }
                for i in start..end {
                    let c1 = csr.indices[i] as usize;
                    let v1 = csr.data[i] as f64;
                    cov_local[(c1, c1)] += v1 * v1;
                    for j in (i + 1)..end {
                        let c2 = csr.indices[j] as usize;
                        let v2 = csr.data[j] as f64;
                        let prod = v1 * v2;
                        cov_local[(c1, c2)] += prod;
                        cov_local[(c2, c1)] += prod;
                    }
                }
                (cov_local, sums_local)
            },
        )
        .reduce(
            || (Mat::<f64>::zeros(n_vars, n_vars), vec![0.0f64; n_vars]),
            |(mut ca, mut sa), (cb, sb)| {
                ca += cb;
                for i in 0..n_vars {
                    sa[i] += sb[i];
                }
                (ca, sa)
            },
        )
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
// Public API — streaming from ShardSource
// ---------------------------------------------------------------------------

/// Compute randomized PCA from a backed SCX reader.
///
/// Streams data shard-by-shard — peak memory is one decoded shard plus
/// the working matrices (n_obs × k and n_vars × k where k = n_components + n_oversamples).
///
/// # CSR-only by design
///
/// PCA stays on the CSR `ShardSource` trait — there is no
/// `ColumnShardSource` overload. The randomized SpMM consumes one
/// shard's rows at a time (row-major access pattern); CSC's column-
/// major layout would force per-column gather scatter with no
/// measurable speed-up. The covariance variant is similar.
/// `pyscx.accel.pca` rejects `prefer_format="csc"` with a `ValueError`
/// for the same reason.
///
/// # Arguments
///
/// * `source` — Shard source (provides shard-by-shard access)
/// * `n_components` — Number of principal components to compute
/// * `n_oversamples` — Extra dimensions for accuracy (default: 10)
/// * `n_power_iterations` — Power iterations for spectral accuracy (default: 2)
/// * `zero_center` — Whether to mean-center the data (default: true)
/// * `seed` — Random seed for reproducibility
pub fn randomized_pca<S: ShardSource>(
    source: &S,
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    seed: u64,
) -> Result<PcaResult> {
    let (n_obs, n_vars) = source.shape();
    validate_inputs(n_obs, n_vars, n_components)?;
    warn_if_cache_undersized(source, "randomized_pca");

    let k = (n_components + n_oversamples).min(n_vars).min(n_obs);
    // Fused pass: compute column means and sum-of-squares together (1 shard pass)
    let (means, col_sum_sq) = source.col_means_and_sum_sq(zero_center)?;
    let means_ref = means.as_deref();

    // Step 2: Random Gaussian Ω (n_vars × k), row-major
    let omega = random_gaussian(n_vars, k, seed);

    // Step 3 + 4 + 5: Streaming SpMM → power iteration → QR
    // When n_power_iterations <= 2, skip intermediate QR on the transpose result
    // (matching sklearn's `_randomized_svd` `normalizer='auto'` threshold — see
    // sklearn.utils.extmath.randomized_svd). The transpose result (n_vars × k) is
    // fed directly back into the forward SpMM, saving one QR + one
    // mat_to_row_major_buf per iteration. QR is still applied to the forward result
    // (n_obs × k) each iteration to prevent basis collapse.
    //
    // For difficult spectra (slow singular-value decay — e.g. noisy unnormalized
    // counts or cold-start raw expression matrices) callers should pass
    // `n_power_iterations >= 4` to engage the full-QR path for better convergence.
    let y = streaming_spmm_forward(source, &omega, k, means_ref)?;
    let mut q = qr_thin_q_row_major(&y, n_obs, k);

    for _ in 0..n_power_iterations {
        let b = streaming_spmm_transpose(source, &q, means_ref)?;
        if n_power_iterations > 2 {
            // Full QR normalization on transpose result for numerical stability
            let q_b = qr_thin_q_row_major(&b, n_vars, k);
            let q_b_rm = mat_to_row_major_buf(&q_b);
            let y = streaming_spmm_forward(source, &q_b_rm, k, means_ref)?;
            q = qr_thin_q_row_major(&y, n_obs, k);
        } else {
            // Skip QR on b — feed row-major b directly into forward SpMM
            let y = streaming_spmm_forward(source, &b, k, means_ref)?;
            q = qr_thin_q_row_major(&y, n_obs, k);
        }
    }

    // Step 6: B = (X - μ)^T @ Q
    let b_rm = streaming_spmm_transpose(source, &q, means_ref)?;

    // Step 7 + 8: SVD of B, recover embeddings (uses pre-computed col_sum_sq — no extra pass)
    let total_var = total_variance_from_col_sq(&col_sum_sq, means_ref, n_obs);
    let b_view = MatRef::from_row_major_slice(&b_rm, n_vars, k);
    build_pca_result(&q, &b_view, &means, n_components, n_obs, n_vars, total_var)
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

    // All SpMM operations work with row-major Vec<f64>.
    // QR/SVD use MatRef::from_row_major_slice for zero-copy faer views.
    let omega = random_gaussian(n_vars, k, seed);
    let y = spmm_forward_csr(csr, &omega, k, means_ref);
    let mut q = qr_thin_q_row_major(&y, n_obs, k);

    for _ in 0..n_power_iterations {
        let b = spmm_transpose_csr(csr, &q, means_ref);
        if n_power_iterations > 2 {
            // Full QR normalization for numerical stability
            let q_b = qr_thin_q_row_major(&b, n_vars, k);
            let q_b_rm = mat_to_row_major_buf(&q_b);
            let y = spmm_forward_csr(csr, &q_b_rm, k, means_ref);
            q = qr_thin_q_row_major(&y, n_obs, k);
        } else {
            // Skip QR on transpose result — feed directly into forward SpMM
            let y = spmm_forward_csr(csr, &b, k, means_ref);
            q = qr_thin_q_row_major(&y, n_obs, k);
        }
    }

    let b_rm = spmm_transpose_csr(csr, &q, means_ref);
    let total_var = compute_total_variance_inmemory(csr, means_ref);
    let b_view = MatRef::from_row_major_slice(&b_rm, n_vars, k);
    build_pca_result(&q, &b_view, &means, n_components, n_obs, n_vars, total_var)
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Warn when a caching `ShardSource`'s decode cache is too small to hold
/// the shard working set across PCA's multiple passes. Out-of-core PCA
/// re-reads every shard ~6–7× (randomized) or 2× (covariance); if the LRU
/// can't hold `n_shards`, each pass evicts and re-decodes from disk — the
/// same silent perf cliff the DE streaming path warns about. No-op for
/// non-caching sources (`shard_cache_capacity() == None`).
fn warn_if_cache_undersized<S: ShardSource>(source: &S, op: &str) {
    if let Some(cap) = source.shard_cache_capacity() {
        let n_shards = source.n_shards();
        if n_shards > 1 && cap < n_shards {
            log::warn!(
                "{op}: shard cache capacity={cap} < n_shards={n_shards} — out-of-core \
                 PCA makes multiple passes over every shard, so the cached read path \
                 will evict and re-decode each shard on each pass. Raise the PCA \
                 `memory_budget` so the cache can hold all {n_shards} shards (or open \
                 with `cache_shards >= {n_shards}`) to realize the speedup."
            );
        }
    }
}

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
// `ShardSource::col_means_and_sum_sq()` in scx-format.
// `compute_total_variance_from_col_sq` has been replaced by
// `scx_format::total_variance_from_col_sq()`.

/// Generate a random Gaussian matrix (rows × cols), row-major.
fn random_gaussian(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = StandardNormal;
    (0..rows * cols).map(|_| normal.sample(&mut rng)).collect()
}

/// Streaming forward SpMM: Y = (X - μ) @ M, shard-by-shard.
///
/// X is (n_obs × n_vars) stored as sharded CSR.
/// `m_data` is row-major `&[f64]` of shape (n_vars × k).
/// Returns row-major `Vec<f64>` of shape (n_obs × k).
fn streaming_spmm_forward<S: ShardSource>(
    source: &S,
    m_data: &[f64],
    k: usize,
    means: Option<&[f64]>,
) -> Result<Vec<f64>> {
    let (n_obs, n_vars) = source.shape();
    debug_assert_eq!(m_data.len(), n_vars * k);

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

    let n_shards = source.n_shards();
    let mut global_row = 0usize;

    for shard_idx in 0..n_shards {
        let csr = source.read_shard_arc(shard_idx)?;
        let shard_rows = csr.n_rows();

        spmm_forward_into(
            &csr,
            m_data,
            k,
            &mut y,
            global_row,
            mean_correction.as_deref(),
        );
        global_row += shard_rows;
    }

    Ok(y)
}

/// Streaming transpose SpMM: Z = (X - μ)^T @ Q, shard-by-shard.
///
/// Q is `Mat<f64>` (n_obs × k), column-major. Each shard is parallelized
/// via rayon thread-local accumulators (same approach as `spmm_transpose_csr`).
/// Returns row-major `Vec<f64>` of shape (n_vars × k).
fn streaming_spmm_transpose<S: ShardSource>(
    source: &S,
    q: &Mat<f64>,
    means: Option<&[f64]>,
) -> Result<Vec<f64>> {
    let (n_obs, n_vars) = source.shape();
    let k = q.ncols();
    debug_assert_eq!(q.nrows(), n_obs);

    let mut z = vec![0.0f64; n_vars * k];
    let mut sum_q = if means.is_some() {
        vec![0.0f64; k]
    } else {
        vec![]
    };

    let n_shards = source.n_shards();
    let mut global_row = 0usize;

    // Per-thread scratch buffers hoisted across *all* shards: each thread's
    // buffer accumulates its contributions across every parallel shard, and
    // the final reduction into `z` / `sum_q` runs once after the shard loop.
    // Eliminates `n_shards - 1` zero-and-reduce passes over the `n_vars·k`
    // accumulator compared with per-shard reduction.
    let mut tls_z: ThreadLocal<RefCell<Vec<f64>>> = ThreadLocal::new();
    let mut tls_sq: ThreadLocal<RefCell<Vec<f64>>> = ThreadLocal::new();
    let tls_q_row: ThreadLocal<RefCell<Vec<f64>>> = ThreadLocal::new();
    let sq_len = if means.is_some() { k } else { 0 };

    for shard_idx in 0..n_shards {
        let csr = source.read_shard_arc(shard_idx)?;
        let shard_rows = csr.n_rows();
        let use_parallel = shard_rows * k > 10_000;

        if use_parallel {
            let gr_base = global_row;
            let chunk_size = (shard_rows / rayon::current_num_threads().max(1)).max(256);

            (0..shard_rows)
                .into_par_iter()
                .with_min_len(chunk_size)
                .for_each(|r| {
                    let mut zl = tls_z
                        .get_or(|| RefCell::new(vec![0.0f64; n_vars * k]))
                        .borrow_mut();
                    let mut sql = tls_sq
                        .get_or(|| RefCell::new(vec![0.0f64; sq_len]))
                        .borrow_mut();

                    let mut q_row = tls_q_row
                        .get_or(|| RefCell::new(vec![0.0f64; k]))
                        .borrow_mut();

                    let gr = gr_base + r;
                    for j in 0..k {
                        q_row[j] = q[(gr, j)];
                    }
                    if means.is_some() {
                        for j in 0..k {
                            sql[j] += q_row[j];
                        }
                    }
                    let start = csr.indptr[r] as usize;
                    let end = csr.indptr[r + 1] as usize;
                    for idx in start..end {
                        let col = csr.indices[idx] as usize;
                        let val = csr.data[idx] as f64;
                        let z_offset = col * k;
                        for j in 0..k {
                            zl[z_offset + j] += val * q_row[j];
                        }
                    }
                });
        } else {
            // Sequential path for small shards
            let mut q_row = vec![0.0f64; k];
            for r in 0..shard_rows {
                let gr = global_row + r;
                for j in 0..k {
                    q_row[j] = q[(gr, j)];
                }
                if means.is_some() {
                    for j in 0..k {
                        sum_q[j] += q_row[j];
                    }
                }
                let start = csr.indptr[r] as usize;
                let end = csr.indptr[r + 1] as usize;
                for idx in start..end {
                    let col = csr.indices[idx] as usize;
                    let val = csr.data[idx] as f64;
                    let z_offset = col * k;
                    for j in 0..k {
                        z[z_offset + j] += val * q_row[j];
                    }
                }
            }
        }

        global_row += shard_rows;
    }

    // Reduce thread-local parallel-path accumulators into the globals.
    // Sequential-path shards wrote directly to `z` / `sum_q`, so this step
    // folds in the parallel contributions only.
    for buf in tls_z.iter_mut() {
        let zl = buf.get_mut();
        for (zg, zv) in z.iter_mut().zip(zl.iter()) {
            *zg += *zv;
        }
    }
    for buf in tls_sq.iter_mut() {
        let sql = buf.get_mut();
        for (sg, sv) in sum_q.iter_mut().zip(sql.iter()) {
            *sg += *sv;
        }
    }

    // Mean centering correction: Z -= μ @ (1^T @ Q)
    if let Some(mu) = means {
        #[allow(clippy::needless_range_loop)]
        for v in 0..n_vars {
            for j in 0..k {
                z[v * k + j] -= mu[v] * sum_q[j];
            }
        }
    }

    Ok(z)
}

/// In-memory forward SpMM: Y = (X - μ) @ M using a single CSR.
///
/// `m_data` is row-major `&[f64]` of shape (n_vars × k).
/// Returns row-major `Vec<f64>` of shape (n_obs × k).
fn spmm_forward_csr(csr: &ScxCsr, m_data: &[f64], k: usize, means: Option<&[f64]>) -> Vec<f64> {
    let n_obs = csr.n_rows();
    let n_vars = csr.n_cols();

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
    spmm_forward_into(csr, m_data, k, &mut y, 0, mean_correction.as_deref());
    y
}

/// In-memory transpose SpMM: Z = (X - μ)^T @ Q using a single CSR.
///
/// Parallelized via rayon: each thread accumulates into a thread-local
/// `z_local` buffer (n_vars × k), then all buffers are reduced by summation.
/// Returns row-major `Vec<f64>` of shape (n_vars × k).
fn spmm_transpose_csr(csr: &ScxCsr, q: &Mat<f64>, means: Option<&[f64]>) -> Vec<f64> {
    let n_obs = csr.n_rows();
    let n_vars = csr.n_cols();
    let k = q.ncols();

    // Parallel threshold: use rayon when work is substantial
    let use_parallel = n_obs * k > 10_000;

    if use_parallel {
        // Each chunk of rows produces a thread-local (z_local, sum_q_local).
        // We partition rows into ~equal chunks for rayon.
        let chunk_size = (n_obs / rayon::current_num_threads().max(1)).max(256);

        let (z, sum_q) = (0..n_obs)
            .into_par_iter()
            .with_min_len(chunk_size)
            .fold(
                || {
                    (
                        vec![0.0f64; n_vars * k],
                        if means.is_some() {
                            vec![0.0f64; k]
                        } else {
                            vec![]
                        },
                    )
                },
                |(mut z_local, mut sq_local), r| {
                    // Copy one row from column-major Q into contiguous buffer
                    let mut q_row = vec![0.0f64; k];
                    for j in 0..k {
                        q_row[j] = q[(r, j)];
                    }
                    if means.is_some() {
                        for j in 0..k {
                            sq_local[j] += q_row[j];
                        }
                    }
                    let start = csr.indptr[r] as usize;
                    let end = csr.indptr[r + 1] as usize;
                    for idx in start..end {
                        let col = csr.indices[idx] as usize;
                        let val = csr.data[idx] as f64;
                        let z_offset = col * k;
                        for j in 0..k {
                            z_local[z_offset + j] += val * q_row[j];
                        }
                    }
                    (z_local, sq_local)
                },
            )
            .reduce(
                || {
                    (
                        vec![0.0f64; n_vars * k],
                        if means.is_some() {
                            vec![0.0f64; k]
                        } else {
                            vec![]
                        },
                    )
                },
                |(mut za, mut sqa), (zb, sqb)| {
                    for i in 0..za.len() {
                        za[i] += zb[i];
                    }
                    for i in 0..sqa.len() {
                        sqa[i] += sqb[i];
                    }
                    (za, sqa)
                },
            );

        // Apply mean correction
        if let Some(mu) = means {
            let mut z = z;
            #[allow(clippy::needless_range_loop)]
            for v in 0..n_vars {
                for j in 0..k {
                    z[v * k + j] -= mu[v] * sum_q[j];
                }
            }
            z
        } else {
            z
        }
    } else {
        // Sequential path for small matrices
        let mut z = vec![0.0f64; n_vars * k];
        let mut sum_q = if means.is_some() {
            vec![0.0f64; k]
        } else {
            vec![]
        };
        let mut q_row = vec![0.0f64; k];

        for r in 0..n_obs {
            for j in 0..k {
                q_row[j] = q[(r, j)];
            }
            if means.is_some() {
                for j in 0..k {
                    sum_q[j] += q_row[j];
                }
            }
            let start = csr.indptr[r] as usize;
            let end = csr.indptr[r + 1] as usize;
            for idx in start..end {
                let col = csr.indices[idx] as usize;
                let val = csr.data[idx] as f64;
                let z_offset = col * k;
                for j in 0..k {
                    z[z_offset + j] += val * q_row[j];
                }
            }
        }

        if let Some(mu) = means {
            #[allow(clippy::needless_range_loop)]
            for v in 0..n_vars {
                for j in 0..k {
                    z[v * k + j] -= mu[v] * sum_q[j];
                }
            }
        }

        z
    }
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
///
/// Uses the textbook two-pass variance formula: `Σ(x - μ)²` after a separate
/// mean pass. This is numerically safe for scRNA data because raw counts and
/// log1p-transformed values are small non-negative magnitudes where
/// catastrophic cancellation doesn't occur. If future callers feed data with
/// large offsets (e.g. non-centered embeddings), switch to Welford's
/// single-pass algorithm.
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
            let n_zeros = n_obs.saturating_sub(col_nnz[c] as usize);
            total += n_zeros as f64 * mu[c] * mu[c];
        }
        total / (n_obs as f64 - 1.0).max(1.0)
    } else {
        let total: f64 = csr.data.iter().map(|&v| (v as f64) * (v as f64)).sum();
        total / (n_obs as f64 - 1.0).max(1.0)
    }
}

/// Build PcaResult from Q (column-major Mat) and B (MatRef, possibly row-major view).
#[allow(clippy::too_many_arguments)]
fn build_pca_result(
    q: &Mat<f64>,
    b: &MatRef<'_, f64>,
    means: &Option<Vec<f64>>,
    n_components: usize,
    n_obs: usize,
    n_vars: usize,
    total_var: f64,
) -> Result<PcaResult> {
    let b_owned = b.to_owned();
    let (u_hat, sigma, vt) = thin_svd_decomp(&b_owned)?;

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
        for v_idx in 0..n_vars {
            components[pc * n_vars + v_idx] = u_hat[(v_idx, pc)];
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
        graph_replayed: None,
    })
}

// ---------------------------------------------------------------------------
// Covariance PCA — optimal when n_vars << n_obs (e.g. HVG-selected data)
// ---------------------------------------------------------------------------

/// Default threshold: use covariance method when n_vars <= this value.
pub const COVARIANCE_PCA_THRESHOLD: usize = 5_000;

/// Compute PCA via the covariance method, streaming from a [`ShardSource`].
///
/// Algorithm (2 passes over data):
/// 1. Accumulate covariance `C += X_shard^T @ X_shard` and column sums
/// 2. Mean-center: `C -= (col_sums^T @ col_sums) / n_obs`
/// 3. Eigendecompose C (self-adjoint, top k eigenvectors)
/// 4. Stream again to compute embeddings: `E += (X_shard - mean) @ V[:, top_k]`
///
/// Memory: O(n_vars²) for the covariance matrix. Only practical when n_vars ≤ ~5,000.
///
/// # CSR-only by design
///
/// Like [`randomized_pca`], the covariance build accumulates
/// `X_shard^T @ X_shard` from row-major nonzeros and offers no win on
/// CSC. CSC dispatch is rejected at the pyscx entry point —
/// `pyscx.accel.pca(prefer_format="csc")` raises `ValueError`.
pub fn covariance_pca<S: ShardSource>(
    source: &S,
    n_components: usize,
    zero_center: bool,
) -> Result<PcaResult> {
    let (n_obs, n_vars) = source.shape();
    validate_inputs(n_obs, n_vars, n_components)?;
    warn_if_cache_undersized(source, "covariance_pca");

    // --- Pass 1: Accumulate covariance matrix and column sums ---
    // Sparse outer product: accumulate C[c1,c2] += v1*v2 directly from CSR nonzeros.
    // No densification — touches only nonzero entries (~2% for typical HVG-selected data).
    let mut cov = Mat::<f64>::zeros(n_vars, n_vars);
    let mut col_sums = vec![0.0f64; n_vars];

    let n_shards = source.n_shards();
    for shard_idx in 0..n_shards {
        let csr = source.read_shard_arc(shard_idx)?;
        sparse_outer_product_accumulate(&csr, &mut col_sums, &mut cov);
    }

    // --- Mean centering ---
    let means = if zero_center {
        Some(
            col_sums
                .iter()
                .map(|&s| s / n_obs as f64)
                .collect::<Vec<f64>>(),
        )
    } else {
        None
    };

    if let Some(ref mu) = means {
        // C -= n_obs * (mu^T @ mu)  (rank-1 correction for mean centering)
        for i in 0..n_vars {
            for j in 0..n_vars {
                cov[(i, j)] -= n_obs as f64 * mu[i] * mu[j];
            }
        }
    }

    // Convert to sample covariance: C /= (n-1)
    let denom = (n_obs as f64 - 1.0).max(1.0);
    for i in 0..n_vars {
        for j in 0..n_vars {
            cov[(i, j)] /= denom;
        }
    }

    // --- Eigendecomposition ---
    let evd = cov
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|e| AccelError::LinAlg(format!("Eigendecomposition failed: {e:?}")))?;

    // faer returns eigenvalues in nondecreasing order; we want the largest k.
    let all_eigenvalues = evd.S().column_vector();
    let eigvecs = evd.U(); // columns are eigenvectors

    // Total variance = sum of all eigenvalues (they ARE the variances since C is sample cov)
    let total_var: f64 = (0..n_vars).map(|i| all_eigenvalues[i]).sum();

    // Select top k eigenvectors (last k columns, reversed for descending order)
    let n_components = n_components.min(n_vars);
    let mut variance_explained = Vec::with_capacity(n_components);
    // Build V: (n_vars × n_components) row-major — columns are the top eigenvectors
    let mut v_rm = vec![0.0f64; n_vars * n_components];
    for pc in 0..n_components {
        let eig_idx = n_vars - 1 - pc; // descending: largest first
        let eigenvalue = all_eigenvalues[eig_idx];
        variance_explained.push(eigenvalue.max(0.0));
        for v in 0..n_vars {
            v_rm[v * n_components + pc] = eigvecs[(v, eig_idx)];
        }
    }

    let variance_ratio: Vec<f64> = if total_var > 0.0 {
        variance_explained
            .iter()
            .map(|&ve| ve / total_var)
            .collect()
    } else {
        vec![0.0; n_components]
    };

    // Components: each PC is a row (n_components × n_vars)
    let mut components = vec![0.0f64; n_components * n_vars];
    for pc in 0..n_components {
        for v in 0..n_vars {
            components[pc * n_vars + v] = v_rm[v * n_components + pc];
        }
    }

    // --- Pass 2: Compute embeddings E = (X - μ) @ V ---
    let mut embeddings = vec![0.0f64; n_obs * n_components];
    let means_ref = means.as_deref();

    // Pre-compute mean correction: mu^T @ V (1 × n_components)
    let mean_correction: Option<Vec<f64>> = means_ref.map(|mu| {
        let mut mc = vec![0.0f64; n_components];
        for v in 0..n_vars {
            for pc in 0..n_components {
                mc[pc] += mu[v] * v_rm[v * n_components + pc];
            }
        }
        mc
    });

    let mut global_row = 0usize;
    for shard_idx in 0..n_shards {
        let csr = source.read_shard_arc(shard_idx)?;
        let shard_rows = csr.n_rows();

        // E[row, :] = X[row, :] @ V - mc
        for r in 0..shard_rows {
            let start = csr.indptr[r] as usize;
            let end = csr.indptr[r + 1] as usize;
            let e_offset = (global_row + r) * n_components;
            for idx in start..end {
                let c = csr.indices[idx] as usize;
                let val = csr.data[idx] as f64;
                let v_offset = c * n_components;
                for pc in 0..n_components {
                    embeddings[e_offset + pc] += val * v_rm[v_offset + pc];
                }
            }
            if let Some(ref mc) = mean_correction {
                for pc in 0..n_components {
                    embeddings[e_offset + pc] -= mc[pc];
                }
            }
        }
        global_row += shard_rows;
    }

    Ok(PcaResult {
        embeddings,
        components,
        variance_explained,
        variance_ratio,
        mean: means,
        n_components,
        n_obs,
        n_vars,
        graph_replayed: None,
    })
}

/// Compute PCA via the covariance method from an in-memory `ScxCsr`.
///
/// Single-shard special case of [`covariance_pca`].
pub fn covariance_pca_inmemory(
    csr: &ScxCsr,
    n_components: usize,
    zero_center: bool,
) -> Result<PcaResult> {
    let (n_obs, n_vars) = (csr.n_rows(), csr.n_cols());
    validate_inputs(n_obs, n_vars, n_components)?;

    // --- Build covariance matrix via sparse outer products ---
    // Parallel accumulation: each rayon thread gets a thread-local n_vars × n_vars
    // covariance matrix (~30 MB for n_vars=2000) and col_sums vector.
    let (mut cov, col_sums) = sparse_outer_product_accumulate_par(csr, n_vars);

    // Mean centering
    let means = if zero_center {
        Some(
            col_sums
                .iter()
                .map(|&s| s / n_obs as f64)
                .collect::<Vec<f64>>(),
        )
    } else {
        None
    };

    if let Some(ref mu) = means {
        for i in 0..n_vars {
            for j in 0..n_vars {
                cov[(i, j)] -= n_obs as f64 * mu[i] * mu[j];
            }
        }
    }

    // Sample covariance
    let denom = (n_obs as f64 - 1.0).max(1.0);
    for i in 0..n_vars {
        for j in 0..n_vars {
            cov[(i, j)] /= denom;
        }
    }

    // Eigendecomposition
    let evd = cov
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|e| AccelError::LinAlg(format!("Eigendecomposition failed: {e:?}")))?;

    let all_eigenvalues = evd.S().column_vector();
    let eigvecs = evd.U();
    let total_var: f64 = (0..n_vars).map(|i| all_eigenvalues[i]).sum();

    let n_components = n_components.min(n_vars);
    let mut variance_explained = Vec::with_capacity(n_components);
    let mut v_rm = vec![0.0f64; n_vars * n_components];
    for pc in 0..n_components {
        let eig_idx = n_vars - 1 - pc;
        let eigenvalue = all_eigenvalues[eig_idx];
        variance_explained.push(eigenvalue.max(0.0));
        for v in 0..n_vars {
            v_rm[v * n_components + pc] = eigvecs[(v, eig_idx)];
        }
    }

    let variance_ratio: Vec<f64> = if total_var > 0.0 {
        variance_explained
            .iter()
            .map(|&ve| ve / total_var)
            .collect()
    } else {
        vec![0.0; n_components]
    };

    let mut components = vec![0.0f64; n_components * n_vars];
    for pc in 0..n_components {
        for v in 0..n_vars {
            components[pc * n_vars + v] = v_rm[v * n_components + pc];
        }
    }

    // Embeddings: E = (X - μ) @ V
    let mut embeddings = vec![0.0f64; n_obs * n_components];
    let mean_correction: Option<Vec<f64>> = means.as_deref().map(|mu| {
        let mut mc = vec![0.0f64; n_components];
        for v in 0..n_vars {
            for pc in 0..n_components {
                mc[pc] += mu[v] * v_rm[v * n_components + pc];
            }
        }
        mc
    });

    for r in 0..n_obs {
        let start = csr.indptr[r] as usize;
        let end = csr.indptr[r + 1] as usize;
        let e_offset = r * n_components;
        for idx in start..end {
            let c = csr.indices[idx] as usize;
            let val = csr.data[idx] as f64;
            let v_offset = c * n_components;
            for pc in 0..n_components {
                embeddings[e_offset + pc] += val * v_rm[v_offset + pc];
            }
        }
        if let Some(ref mc) = mean_correction {
            for pc in 0..n_components {
                embeddings[e_offset + pc] -= mc[pc];
            }
        }
    }

    Ok(PcaResult {
        embeddings,
        components,
        variance_explained,
        variance_ratio,
        mean: means,
        n_components,
        n_obs,
        n_vars,
        graph_replayed: None,
    })
}

// ---------------------------------------------------------------------------
// GPU PCA dispatch (behind "gpu" feature)
// ---------------------------------------------------------------------------

/// GPU-accelerated randomized PCA from any `ShardSource`.
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
/// * `source` — any `ShardSource + Sync` (backed reader, lazy transform, or
///   in-memory CSR wrapper)
/// * `n_components` — Number of principal components to compute
/// * `n_oversamples` — Extra dimensions for accuracy (default: 10)
/// * `n_power_iterations` — Power iterations for spectral accuracy (default: 2)
/// * `zero_center` — Whether to mean-center the data (default: true)
/// * `seed` — Random seed for reproducibility
/// * `qr_method` — Householder (default, always-stable) or CholeskyQR2 (opt-in,
///   faster but fails with `CuSolverError` on non-SPD Gram matrices)
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn randomized_pca_gpu<S: ShardSource + Sync>(
    device_id: usize,
    source: &S,
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    seed: u64,
    qr_method: scx_gpu::QrMethod,
    tuning: scx_gpu::GpuPcaTuning,
) -> Result<PcaResult> {
    let dev = scx_gpu::GpuDevice::new(device_id)
        .map_err(|e| AccelError::LinAlg(format!("GPU init failed: {e}")))?;

    let gpu_result = scx_gpu::gpu_randomized_pca(
        &dev,
        source,
        n_components,
        n_oversamples,
        n_power_iterations,
        zero_center,
        seed,
        qr_method,
        tuning,
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
        graph_replayed: Some(gpu_result.graph_replayed),
    })
}

/// Check whether a GPU is available for GPU-accelerated PCA.
///
/// Returns `true` if at least one CUDA device is found.
#[cfg(feature = "gpu")]
pub fn gpu_available() -> bool {
    scx_gpu::GpuDevice::count().is_ok_and(|n| n > 0)
}

/// GPU device information returned by [`gpu_info`].
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// Human-readable device name (e.g. "NVIDIA A100-SXM4-80GB").
    pub device_name: String,
    /// Total VRAM in bytes.
    pub total_vram_bytes: usize,
    /// Free VRAM in bytes.
    pub free_vram_bytes: usize,
}

/// Query GPU device information for device 0.
///
/// Returns `None` if no GPU is available or CUDA initialization fails.
#[cfg(feature = "gpu")]
pub fn gpu_info() -> Option<GpuInfo> {
    let count = scx_gpu::GpuDevice::count().ok()?;
    if count == 0 {
        return None;
    }
    let dev = scx_gpu::GpuDevice::new(0).ok()?;
    let device_name = dev.name().unwrap_or_else(|_| "unknown".to_string());
    let (free, total) = dev.free_memory().ok()?;
    Some(GpuInfo {
        device_name,
        total_vram_bytes: total,
        free_vram_bytes: free,
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

        let m_data = random_gaussian(n_vars, k, 42);
        let y_sparse = spmm_forward_csr(&csr, &m_data, k, None);

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
                (y_sparse[i] - y_dense[i]).abs() < 1e-10,
                "mismatch at {i}: {} vs {}",
                y_sparse[i],
                y_dense[i]
            );
        }
    }

    #[test]
    fn test_qr_orthonormal() {
        let data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let q = qr_thin_q_row_major(&data, 4, 2);

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
        let view = MatRef::from_row_major_slice(&data, 3, 2);
        let mat = view.to_owned();
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

    #[test]
    fn test_qr_and_svd_on_row_major_slice() {
        // Verify QR and SVD produce correct results when input is a
        // MatRef::from_row_major_slice (transposed column-major view).
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let view = MatRef::from_row_major_slice(&data, 4, 3);

        // QR: Q should have orthonormal columns
        let q = qr_thin_q_row_major(&data, 4, 3);
        assert_eq!(q.nrows(), 4);
        assert_eq!(q.ncols(), 3);
        for c in 0..3 {
            let norm: f64 = (0..4).map(|r| q[(r, c)].powi(2)).sum();
            assert!(
                (norm - 1.0).abs() < 1e-10,
                "QR col {c} norm = {norm}, expected 1.0"
            );
        }

        // SVD: reconstruct A ≈ U @ diag(σ) @ V^T
        let mat = view.to_owned();
        let (u, sigma, vt) = thin_svd_decomp(&mat).unwrap();
        for i in 0..4 {
            for j in 0..3 {
                let reconstructed: f64 = (0..3).map(|s| u[(i, s)] * sigma[s] * vt[(s, j)]).sum();
                let original = view[(i, j)];
                assert!(
                    (reconstructed - original).abs() < 1e-10,
                    "SVD reconstruction mismatch at ({i},{j}): {reconstructed} vs {original}"
                );
            }
        }
    }

    #[test]
    fn test_covariance_pca_basic() {
        let csr = test_csr_10x5();
        let result = covariance_pca_inmemory(&csr, 3, true).unwrap();

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
    fn test_covariance_pca_cosine_similarity_vs_randomized() {
        // Both methods should produce similar top PCs (cosine similarity > 0.99)
        let csr = test_csr_10x5();
        let cov_result = covariance_pca_inmemory(&csr, 3, true).unwrap();
        let rand_result = randomized_pca_inmemory(&csr, 3, 5, 2, true, 42).unwrap();

        // Compare top-3 PC embeddings via cosine similarity per component
        for pc in 0..3 {
            let mut cov_col = vec![0.0f64; 10];
            let mut rand_col = vec![0.0f64; 10];
            for i in 0..10 {
                cov_col[i] = cov_result.embeddings[i * 3 + pc];
                rand_col[i] = rand_result.embeddings[i * 3 + pc];
            }

            let dot: f64 = cov_col.iter().zip(&rand_col).map(|(a, b)| a * b).sum();
            let norm_a: f64 = cov_col.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_b: f64 = rand_col.iter().map(|x| x * x).sum::<f64>().sqrt();
            let cosine = if norm_a > 0.0 && norm_b > 0.0 {
                (dot / (norm_a * norm_b)).abs()
            } else {
                0.0
            };

            assert!(
                cosine > 0.99,
                "PC{pc} cosine similarity = {cosine:.4} (expected > 0.99)"
            );
        }
    }

    #[test]
    fn test_covariance_pca_variance_ratio_matches_randomized() {
        let csr = test_csr_10x5();
        let cov_result = covariance_pca_inmemory(&csr, 3, true).unwrap();
        let rand_result = randomized_pca_inmemory(&csr, 3, 5, 2, true, 42).unwrap();

        // Variance ratios should be close (within 5% relative)
        for pc in 0..3 {
            let cov_vr = cov_result.variance_ratio[pc];
            let rand_vr = rand_result.variance_ratio[pc];
            let rel_diff = ((cov_vr - rand_vr) / rand_vr.max(1e-12)).abs();
            assert!(
                rel_diff < 0.05,
                "PC{pc} variance ratio: cov={cov_vr:.6}, rand={rand_vr:.6}, rel_diff={rel_diff:.4}"
            );
        }
    }

    #[test]
    fn test_covariance_pca_no_center() {
        let csr = test_csr_10x5();
        let result = covariance_pca_inmemory(&csr, 2, false).unwrap();

        assert_eq!(result.n_components, 2);
        assert!(result.mean.is_none());
        for &ve in &result.variance_explained {
            assert!(ve > 0.0);
        }
    }

    #[test]
    fn test_covariance_pca_components_orthogonal() {
        let csr = test_csr_10x5();
        let result = covariance_pca_inmemory(&csr, 3, true).unwrap();

        for i in 0..result.n_components {
            for j in (i + 1)..result.n_components {
                let dot: f64 = (0..result.n_vars)
                    .map(|v| {
                        result.components[i * result.n_vars + v]
                            * result.components[j * result.n_vars + v]
                    })
                    .sum();
                assert!(dot.abs() < 0.01, "PC{i} · PC{j} = {dot}, expected ~0");
            }
        }
    }

    #[test]
    fn test_sparse_covariance_matches_dense() {
        // Verify that sparse outer product accumulation produces the same
        // covariance matrix as the dense GEMM approach (X^T @ X).
        let csr = test_csr_10x5();
        let n_vars = csr.n_cols();
        let n_obs = csr.n_rows();

        // --- Dense GEMM reference ---
        let mut dense = vec![0.0f64; n_obs * n_vars];
        let mut col_sums_dense = vec![0.0f64; n_vars];
        for r in 0..n_obs {
            let start = csr.indptr[r] as usize;
            let end = csr.indptr[r + 1] as usize;
            for idx in start..end {
                let c = csr.indices[idx] as usize;
                let v = csr.data[idx] as f64;
                dense[r * n_vars + c] = v;
                col_sums_dense[c] += v;
            }
        }
        let mut cov_dense = Mat::<f64>::zeros(n_vars, n_vars);
        let x_ref = MatRef::from_row_major_slice(&dense, n_obs, n_vars);
        faer::linalg::matmul::matmul(
            cov_dense.as_mut(),
            faer::Accum::Add,
            x_ref.transpose(),
            x_ref,
            1.0,
            faer::Par::rayon(0),
        );

        // --- Sparse sequential ---
        let mut cov_sparse = Mat::<f64>::zeros(n_vars, n_vars);
        let mut col_sums_sparse = vec![0.0f64; n_vars];
        sparse_outer_product_accumulate(&csr, &mut col_sums_sparse, &mut cov_sparse);

        // --- Sparse parallel ---
        let (cov_par, col_sums_par) = sparse_outer_product_accumulate_par(&csr, n_vars);

        // Check col_sums match
        for c in 0..n_vars {
            assert!(
                (col_sums_dense[c] - col_sums_sparse[c]).abs() < 1e-10,
                "col_sums mismatch at {c}: dense={}, sparse={}",
                col_sums_dense[c],
                col_sums_sparse[c]
            );
            assert!(
                (col_sums_dense[c] - col_sums_par[c]).abs() < 1e-10,
                "col_sums mismatch at {c}: dense={}, par={}",
                col_sums_dense[c],
                col_sums_par[c]
            );
        }

        // Check covariance matrices match
        for i in 0..n_vars {
            for j in 0..n_vars {
                assert!(
                    (cov_dense[(i, j)] - cov_sparse[(i, j)]).abs() < 1e-10,
                    "cov mismatch at ({i},{j}): dense={}, sparse={}",
                    cov_dense[(i, j)],
                    cov_sparse[(i, j)]
                );
                assert!(
                    (cov_dense[(i, j)] - cov_par[(i, j)]).abs() < 1e-10,
                    "cov mismatch at ({i},{j}): dense={}, par={}",
                    cov_dense[(i, j)],
                    cov_par[(i, j)]
                );
            }
        }
    }
}
