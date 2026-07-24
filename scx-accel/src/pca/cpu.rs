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

use scx_format_io::ShardSource;
use scx_sparse::total_variance_from_col_sq;

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
/// Exploits symmetry: writes **only the lower triangle** (entries `(r, c)` with
/// `r >= c`). The eigensolver reads the lower triangle via
/// `self_adjoint_eigen(Side::Lower)`, so the upper triangle is never populated —
/// this halves the inner-loop stores and avoids a cache-unfriendly second scatter.
/// Also accumulates `col_sums` in the same pass (caller provides the slice).
///
/// Retained as the serial reference for `test_sparse_covariance_matches_dense`; the
/// production paths (streaming + in-memory) use the parallel variant below.
#[allow(dead_code)]
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
        // Sparse outer product — lower triangle only
        for i in start..end {
            let c1 = csr.indices[i] as usize;
            let v1 = csr.data[i] as f64;
            cov[(c1, c1)] += v1 * v1; // diagonal
            for j in (i + 1)..end {
                let c2 = csr.indices[j] as usize;
                let v2 = csr.data[j] as f64;
                let prod = v1 * v2;
                // Write the lower-triangle entry only (row >= col).
                let (lo, hi) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
                cov[(hi, lo)] += prod;
            }
        }
    }
}

/// Parallel sparse outer product accumulation using thread-local `Mat<f64>` accumulators.
///
/// Same algorithm as [`sparse_outer_product_accumulate`] (lower-triangle only) but
/// distributes rows across rayon threads. Each thread gets its own n_vars × n_vars
/// covariance matrix (~30 MB for n_vars=2000) and col_sums vector, then results are
/// reduced by element-wise addition.
fn sparse_outer_product_accumulate_par(csr: &ScxCsr, n_vars: usize) -> (Mat<f64>, Vec<f64>) {
    let n_rows = csr.n_rows();
    // Bound peak memory: each parallel fold segment allocates a full
    // n_vars×n_vars f64 accumulator (n_vars²·8 bytes). rayon `fold` produces one
    // accumulator per segment, and `with_min_len(L)` caps segments at n_rows/L,
    // so sizing the segment length by the budget-derived worker count limits the
    // number of concurrent accumulators — the same guarantee the streaming
    // covariance path gets from `cov_accumulator_workers` (previously this
    // in-memory path used the full thread count, so peak scaled with cores).
    let max_threads = rayon::current_num_threads().max(1);
    // `.max(1)` is defensive: cov_accumulator_workers already floors at 1, but
    // guard the `n_rows / workers` divisor against any future 0-return.
    let workers = cov_accumulator_workers(n_vars, cov_memory_budget(), max_threads).max(1);
    let chunk_size = (n_rows / workers).max(256);

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
                        // Write the lower-triangle entry only (row >= col).
                        let (lo, hi) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
                        cov_local[(hi, lo)] += prod;
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

/// Default per-call memory budget (bytes) for the streaming covariance-PCA
/// accumulators. Caps how many `n_vars × n_vars` f64 matrices the parallel build
/// holds concurrently. Override with `SCX_PCA_COV_MEMORY_BUDGET` (bytes). At the
/// n_vars ≤ [`COVARIANCE_PCA_THRESHOLD`] route ceiling a single accumulator is
/// ~200 MB, so this admits ~10 workers there and the full core count for smaller
/// n_vars.
const DEFAULT_COV_MEMORY_BUDGET: u64 = 2 * 1024 * 1024 * 1024; // 2 GiB

/// Read the covariance-accumulator memory budget (bytes) from the environment,
/// falling back to [`DEFAULT_COV_MEMORY_BUDGET`] when unset or unparseable.
fn cov_memory_budget() -> u64 {
    match std::env::var("SCX_PCA_COV_MEMORY_BUDGET") {
        Ok(s) => s.trim().parse::<u64>().unwrap_or(DEFAULT_COV_MEMORY_BUDGET),
        Err(_) => DEFAULT_COV_MEMORY_BUDGET,
    }
}

/// Number of concurrent covariance accumulators that fit in `budget` bytes,
/// clamped to `[1, max_threads]`. Each accumulator is one `n_vars × n_vars` f64
/// matrix (`n_vars² × 8` bytes). Mirrors the worker-deration arithmetic used by
/// scx-convert's parallel-streaming coordinator: `(budget / per_worker).max(1)`
/// then clamped to the requested thread count.
fn cov_accumulator_workers(n_vars: usize, budget: u64, max_threads: usize) -> usize {
    let mat_bytes = (n_vars as u64)
        .saturating_mul(n_vars as u64)
        .saturating_mul(8)
        .max(1);
    let by_budget = (budget / mat_bytes).max(1) as usize;
    by_budget.min(max_threads.max(1))
}

/// Stream shards and accumulate the covariance **lower triangle** + column sums,
/// reusing a bounded set of thread-local accumulators across **all** shards.
///
/// Peak memory is bounded to ≈ `(workers + 1) × n_vars² × 8` bytes regardless of
/// shard count or host core count, where `workers` is derived from
/// [`cov_memory_budget`]. This preserves the out-of-core property of the streaming
/// path — peak RAM scales with `n_vars` (≤ [`COVARIANCE_PCA_THRESHOLD`]), not
/// `n_obs` — while avoiding the per-shard accumulator reallocation of a naive
/// `fold`-per-shard loop. Like the accumulators above, only the lower triangle is
/// written (the eigensolver reads `Side::Lower`).
fn accumulate_covariance_streaming<S: ShardSource>(
    source: &S,
    n_vars: usize,
) -> Result<(Mat<f64>, Vec<f64>)> {
    let n_shards = source.n_shards();
    let max_threads = rayon::current_num_threads();
    let budget = cov_memory_budget();
    // Memory-derived worker cap, lowered further by the shared
    // `SCX_ACCEL_NUM_THREADS` policy when set (unset → no change).
    let workers = cov_accumulator_workers(n_vars, budget, max_threads)
        .min(crate::mem_budget::accel_num_threads().unwrap_or(usize::MAX));
    if workers < max_threads {
        log::debug!(
            "covariance_pca: capping accumulator workers to {workers} of {max_threads} \
             (n_vars={n_vars}, budget={budget} B, ~{} MB/accumulator) to bound peak memory",
            (n_vars as u64)
                .saturating_mul(n_vars as u64)
                .saturating_mul(8)
                / (1024 * 1024)
        );
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| AccelError::InvalidInput(format!("rayon thread pool: {e}")))?;

    // Per-thread accumulators, reused across every shard (≤ `workers` of them).
    // Shards are read sequentially on the calling thread; only the per-row
    // accumulation runs in the bounded pool, so `S` need not be `Sync`.
    let mut tls: ThreadLocal<RefCell<(Mat<f64>, Vec<f64>)>> = ThreadLocal::new();

    for shard_idx in 0..n_shards {
        // Cached read (T4.4): each pass decodes a shard once when the budget allows.
        let csr = source.read_shard_arc(shard_idx)?;
        let n_rows = csr.n_rows();
        if n_rows == 0 {
            continue;
        }
        // Bind the reduction guard only for shards that actually accumulate, so
        // empty shards don't inflate the reduction call count.
        let _r = scx_format_io::reduction_guard();
        let chunk_size = (n_rows / workers.max(1)).max(256);
        let csr_ref = &csr;
        let tls_ref = &tls;
        pool.install(move || {
            (0..n_rows)
                .into_par_iter()
                .with_min_len(chunk_size)
                .for_each(|r| {
                    let cell = tls_ref.get_or(|| {
                        RefCell::new((Mat::<f64>::zeros(n_vars, n_vars), vec![0.0f64; n_vars]))
                    });
                    let (cov_local, sums_local) = &mut *cell.borrow_mut();
                    let start = csr_ref.indptr[r] as usize;
                    let end = csr_ref.indptr[r + 1] as usize;
                    for idx in start..end {
                        let c = csr_ref.indices[idx] as usize;
                        let v = csr_ref.data[idx] as f64;
                        sums_local[c] += v;
                    }
                    for i in start..end {
                        let c1 = csr_ref.indices[i] as usize;
                        let v1 = csr_ref.data[i] as f64;
                        cov_local[(c1, c1)] += v1 * v1;
                        for j in (i + 1)..end {
                            let c2 = csr_ref.indices[j] as usize;
                            let v2 = csr_ref.data[j] as f64;
                            let prod = v1 * v2;
                            // Write the lower-triangle entry only (row >= col).
                            let (lo, hi) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
                            cov_local[(hi, lo)] += prod;
                        }
                    }
                });
        });
    }

    // Reduce the thread-local lower triangles + column sums into the result.
    let mut cov = Mat::<f64>::zeros(n_vars, n_vars);
    let mut col_sums = vec![0.0f64; n_vars];
    for cell in tls.iter_mut() {
        let (cov_local, sums_local) = cell.get_mut();
        for i in 0..n_vars {
            for j in 0..=i {
                cov[(i, j)] += cov_local[(i, j)];
            }
        }
        for (acc, &s) in col_sums.iter_mut().zip(sums_local.iter()) {
            *acc += s;
        }
    }
    Ok((cov, col_sums))
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

    // Step 7 + 8: SVD of B, recover embeddings (uses pre-computed col_sum_sq — no extra pass).
    // For the centered case, guard against catastrophic cancellation in the closed
    // form; on the rare unstable input re-stream the shards once for a stable
    // centered recompute (read_shard_arc is cached — T4.4).
    let mut total_var = total_variance_from_col_sq(&col_sum_sq, means_ref, n_obs);
    if let Some(mu) = means_ref {
        if closed_form_variance_unstable(&col_sum_sq, mu, n_obs) {
            log::debug!(
                "randomized_pca: closed-form total variance lost precision to \
                 cancellation; re-streaming shards for a stable centered recompute"
            );
            let mut total = 0.0f64;
            let mut col_nnz = vec![0u64; n_vars];
            for shard_idx in 0..source.n_shards() {
                let csr = source.read_shard_arc(shard_idx)?;
                accumulate_centered_ss(&csr, mu, &mut total, &mut col_nnz);
            }
            total_var = finalize_centered_variance(total, &col_nnz, mu, n_obs);
        }
    }
    let b_view = MatRef::from_row_major_slice(&b_rm, n_vars, k);
    build_pca_result(&q, &b_view, &means, n_components, n_obs, n_vars, total_var)
}

// ---------------------------------------------------------------------------
// PFlog (v4) — baseline-aware randomized PCA
// ---------------------------------------------------------------------------

/// Streaming forward SpMM with a per-row baseline offset:
/// `Y = (delta + baseline·1ᵀ − 1·μᵀ) @ M`.
///
/// Reuses [`streaming_spmm_forward`] for the `delta@M − 1⊗(μᵀM)` part (the
/// column-centering rank-1 term, when `means` is `Some`), then adds the
/// row-baseline rank-1 term `baseline_i · colsum(M)_j`. `delta` is whatever
/// `source` streams (the lazy `Scale{4α}→Log1p` source for PFlog v4).
fn streaming_spmm_forward_offset<S: ShardSource>(
    source: &S,
    m_data: &[f64],
    k: usize,
    means: Option<&[f64]>,
    baseline: &[f64],
) -> Result<Vec<f64>> {
    let (n_obs, n_vars) = source.shape();
    debug_assert_eq!(m_data.len(), n_vars * k);
    debug_assert_eq!(baseline.len(), n_obs);

    let mut y = streaming_spmm_forward(source, m_data, k, means)?;

    // colsum(M)_j = Σ_v M[v, j]
    let mut colsum_m = vec![0.0f64; k];
    for v in 0..n_vars {
        let row = &m_data[v * k..(v + 1) * k];
        for (acc, &mv) in colsum_m.iter_mut().zip(row.iter()) {
            *acc += mv;
        }
    }
    // Y[i, j] += baseline_i · colsum(M)_j
    for (i, &b) in baseline.iter().enumerate() {
        let y_row = &mut y[i * k..(i + 1) * k];
        for (yv, &cm) in y_row.iter_mut().zip(colsum_m.iter()) {
            *yv += b * cm;
        }
    }
    Ok(y)
}

/// Streaming transpose SpMM with a per-row baseline offset:
/// `B = (delta + baseline·1ᵀ − 1·μᵀ)ᵀ @ Q`.
///
/// Reuses [`streaming_spmm_transpose`] for the `deltaᵀQ − μ·(1ᵀQ)` part, then
/// adds the row-baseline rank-1 term `1 ⊗ (baselineᵀ Q)` to every variable row.
fn streaming_spmm_transpose_offset<S: ShardSource>(
    source: &S,
    q: &Mat<f64>,
    means: Option<&[f64]>,
    baseline: &[f64],
) -> Result<Vec<f64>> {
    let (n_obs, n_vars) = source.shape();
    let k = q.ncols();
    debug_assert_eq!(q.nrows(), n_obs);
    debug_assert_eq!(baseline.len(), n_obs);

    let mut z = streaming_spmm_transpose(source, q, means)?;

    // bq_j = Σ_i baseline_i · Q[i, j]
    let mut bq = vec![0.0f64; k];
    for (i, &b) in baseline.iter().enumerate() {
        if b == 0.0 {
            continue;
        }
        for (j, slot) in bq.iter_mut().enumerate() {
            *slot += b * q[(i, j)];
        }
    }
    // B[v, j] += bq_j for every variable row v
    for v in 0..n_vars {
        let z_row = &mut z[v * k..(v + 1) * k];
        for (zv, &bqj) in z_row.iter_mut().zip(bq.iter()) {
            *zv += bqj;
        }
    }
    Ok(z)
}

/// Total variance of the exact PFlog matrix `Z = delta + baseline·1ᵀ`,
/// computed in closed form from the `delta` column statistics + baseline.
///
/// `colmean_delta` is `Some` for the column-centered case (the variance of the
/// centered `Z`) and `None` for the uncentered second-moment sum. Both divide
/// by `n_obs − 1` to match [`total_variance_from_col_sq`]'s convention.
///
/// Centered: with `μ_j = colmean_delta_j + b̄`, `Σ_j Σ_i (delta_ij + b_i − μ_j)²`
/// expands (using `Σ_j delta_ij = −D·b_i`) to
/// `S_dd − D·B2 + n_obs·Σμ² − 2·Σ_j μ_j·C_j − 2·n_obs·b̄·Σ_j μ_j` with
/// `C_j = n_obs·colmean_delta_j`. Since `colmean_delta_j = μ_j − b̄`, the two
/// cross terms collapse to `2·n_obs·Σμ²`, leaving the closed form
/// `S_dd − D·B2 − n_obs·Σμ²`.
fn pflog_total_variance(
    col_sum_sq_delta: &[f64],
    colmean_delta: Option<&[f64]>,
    baseline: &[f64],
    n_obs: usize,
    n_vars: usize,
) -> f64 {
    let s_dd: f64 = col_sum_sq_delta.iter().sum();
    let b2: f64 = baseline.iter().map(|&b| b * b).sum();
    let denom = (n_obs as f64 - 1.0).max(1.0);
    let d = n_vars as f64;
    match colmean_delta {
        // Uncentered: Σ Z² = S_dd − D·B2 (the cross term 2Σδb = −2D·B2 cancels D·B2).
        None => ((s_dd - d * b2) / denom).max(0.0),
        Some(cd) => {
            let baseline_mean = baseline.iter().sum::<f64>() / (n_obs as f64).max(1.0);
            let n = n_obs as f64;
            let mut m2 = 0.0f64; // Σ_j μ_j²
            for &cdj in cd {
                let mu = cdj + baseline_mean;
                m2 += mu * mu;
            }
            // Cross terms cancel to −n·Σμ² (see doc comment): TSS = S_dd − D·B2 − n·Σμ².
            let tss = s_dd - d * b2 - n * m2;
            (tss / denom).max(0.0)
        }
    }
}

/// Exact PFlog (v4) randomized PCA, streaming the `delta` source out-of-core.
///
/// `delta_source` streams the sparse `delta` shards (the lazy `Scale{4α} → Log1p`
/// source, i.e. `delta_ij = log1p(4α·x_ij)`); `baseline` is the per-cell offset
/// from [`crate::pflog::pflog_baseline_from_raw`]. The randomized SVD is
/// driven exactly as [`randomized_pca`] but over the implicit dense
/// `Z = delta + baseline·1ᵀ`: each SpMM pass folds the row-baseline rank-1 term
/// alongside the existing column-centering rank-1 term, so `Z` is never
/// materialized.
///
/// When `zero_center` is set the column mean is `μ_j = colmean(delta)_j + mean(baseline)`
/// — it **must** include the baseline, or centering is wrong.
///
/// # Pass budget
///
/// Out-of-core cost is 1 column-stats pass + 2 passes per power iteration
/// (forward + transpose) + 1 final transpose. Each pass re-streams `delta_source`,
/// which recomputes `normalize→log1p` per pass; with a non-caching source this is
/// decode-bound (acceptable for randomized PCA).
#[allow(clippy::too_many_arguments)]
pub fn pflog_pca<S: ShardSource>(
    delta_source: &S,
    baseline: &[f64],
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    seed: u64,
) -> Result<PcaResult> {
    let (n_obs, n_vars) = delta_source.shape();
    validate_inputs(n_obs, n_vars, n_components)?;
    if baseline.len() != n_obs {
        return Err(AccelError::ShapeError(format!(
            "baseline has length {} but delta_source reports n_obs={n_obs}",
            baseline.len()
        )));
    }
    warn_if_cache_undersized(delta_source, "pflog_pca");

    let k = (n_components + n_oversamples).min(n_vars).min(n_obs);

    // Column means + sum-of-squares of `delta` (one pass). The column mean of Z
    // adds mean(baseline) uniformly to colmean(delta).
    let (colmean_delta, col_sum_sq_delta) = delta_source.col_means_and_sum_sq(zero_center)?;
    let baseline_mean = baseline.iter().sum::<f64>() / (n_obs as f64).max(1.0);
    let means: Option<Vec<f64>> = colmean_delta
        .as_ref()
        .map(|cd| cd.iter().map(|&m| m + baseline_mean).collect());
    let means_ref = means.as_deref();

    let omega = random_gaussian(n_vars, k, seed);
    let y = streaming_spmm_forward_offset(delta_source, &omega, k, means_ref, baseline)?;
    let mut q = qr_thin_q_row_major(&y, n_obs, k);

    for _ in 0..n_power_iterations {
        let b = streaming_spmm_transpose_offset(delta_source, &q, means_ref, baseline)?;
        if n_power_iterations > 2 {
            let q_b = qr_thin_q_row_major(&b, n_vars, k);
            let q_b_rm = mat_to_row_major_buf(&q_b);
            let y = streaming_spmm_forward_offset(delta_source, &q_b_rm, k, means_ref, baseline)?;
            q = qr_thin_q_row_major(&y, n_obs, k);
        } else {
            let y = streaming_spmm_forward_offset(delta_source, &b, k, means_ref, baseline)?;
            q = qr_thin_q_row_major(&y, n_obs, k);
        }
    }

    let b_rm = streaming_spmm_transpose_offset(delta_source, &q, means_ref, baseline)?;
    let total_var = pflog_total_variance(
        &col_sum_sq_delta,
        colmean_delta.as_deref(),
        baseline,
        n_obs,
        n_vars,
    );
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

    // Fused pass: column sums + sum-of-squares in one sweep over the nonzeros,
    // reused for both mean-centering and total variance (no separate variance
    // pass). Mirrors the streaming path's `col_means_and_sum_sq`.
    let (col_sums, col_sum_sq) = csr.col_sums_and_sum_sq();
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
    // Reuse the fused col_sum_sq (no extra full-matrix pass). For the centered
    // case, guard against catastrophic cancellation in the closed form
    // (`Σx² − nμ²`); on the rare unstable input recompute the stable centered
    // variance over the resident CSR.
    let mut total_var = total_variance_from_col_sq(&col_sum_sq, means_ref, n_obs);
    if let Some(mu) = means_ref {
        if closed_form_variance_unstable(&col_sum_sq, mu, n_obs) {
            total_var = stable_centered_total_variance_inmemory(csr, mu, n_obs);
        }
    }
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
// `ShardSource::col_means_and_sum_sq()` in scx-format-io.
// `compute_total_variance_from_col_sq` has been replaced by
// `scx_sparse::total_variance_from_col_sq()`.

/// Relative-cancellation threshold for the closed-form total variance.
///
/// The closed form `Σ col_sum_sq − n·Σ μ²` (via `total_variance_from_col_sq`)
/// loses precision to catastrophic cancellation when the true variance is tiny
/// relative to the magnitude being subtracted (e.g. near-constant large-offset
/// columns). For real scRNA data (counts / log1p — small non-negative values)
/// the ratio is O(1), far above this threshold, so the guard never fires on the
/// hot path; it only engages for pathological large-offset inputs.
const CLOSED_FORM_VAR_REL_EPS: f64 = 1e-7;

/// Returns `true` when the closed-form centered total variance has lost too much
/// precision to cancellation and the caller should recompute via the stable
/// centered formula. Only meaningful for the centered (`zero_center`) case — the
/// uncentered path sums `col_sum_sq` directly with no subtraction.
fn closed_form_variance_unstable(col_sum_sq: &[f64], means: &[f64], n_obs: usize) -> bool {
    let sum_sq: f64 = col_sum_sq.iter().sum();
    let mean_sq: f64 = means.iter().map(|&m| m * m).sum::<f64>() * n_obs as f64;
    // Catches both `≤ 0` (sign-flipped garbage) and tiny-positive garbage.
    (sum_sq - mean_sq) <= CLOSED_FORM_VAR_REL_EPS * sum_sq
}

/// Accumulate the centered sum-of-squares `Σ (x − μ_c)²` over one CSR shard's
/// stored entries, and tally per-column nnz (for the later zero-fill).
///
/// Numerically stable: it forms `(x − μ_c)` directly rather than the
/// cancellation-prone `Σx² − nμ²`. Used by the stable fallback in both PCA paths.
fn accumulate_centered_ss(csr: &ScxCsr, means: &[f64], total: &mut f64, col_nnz: &mut [u64]) {
    for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
        let c = col as usize;
        let d = val as f64 - means[c];
        *total += d * d;
        col_nnz[c] += 1;
    }
}

/// Finalize the stable centered total variance: fold the implicit-zero
/// contribution `(n_obs − nnz_c)·μ_c²` per column and divide by `n_obs − 1`.
fn finalize_centered_variance(mut total: f64, col_nnz: &[u64], means: &[f64], n_obs: usize) -> f64 {
    for (c, &nnz) in col_nnz.iter().enumerate() {
        let n_zeros = (n_obs as u64).saturating_sub(nnz) as f64;
        total += n_zeros * means[c] * means[c];
    }
    total / (n_obs as f64 - 1.0).max(1.0)
}

/// Stable centered total variance for an in-memory CSR (single shard).
fn stable_centered_total_variance_inmemory(csr: &ScxCsr, means: &[f64], n_obs: usize) -> f64 {
    let mut total = 0.0f64;
    let mut col_nnz = vec![0u64; means.len()];
    accumulate_centered_ss(csr, means, &mut total, &mut col_nnz);
    finalize_centered_variance(total, &col_nnz, means, n_obs)
}

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
        let _r = scx_format_io::reduction_guard();
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
        let _r = scx_format_io::reduction_guard();
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

        // The fold accumulator carries a reusable `q_row` scratch buffer as its
        // third element so it is allocated once per rayon task (per chunk)
        // rather than once per row. Values are identical to the per-row alloc.
        let (z, sum_q, _) = (0..n_obs)
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
                        vec![0.0f64; k],
                    )
                },
                |(mut z_local, mut sq_local, mut q_row), r| {
                    // Copy one row from column-major Q into the reused buffer
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
                    (z_local, sq_local, q_row)
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
                        Vec::new(),
                    )
                },
                |(mut za, mut sqa, qa), (zb, sqb, _qb)| {
                    for i in 0..za.len() {
                        za[i] += zb[i];
                    }
                    for i in 0..sqa.len() {
                        sqa[i] += sqb[i];
                    }
                    (za, sqa, qa)
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
// `scx_sparse::total_variance_from_col_sq()`.

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
    // No densification — touches only nonzero entries (~2% for typical HVG-selected
    // data). Streams shard-by-shard with a memory-bounded set of thread-local
    // accumulators (see `accumulate_covariance_streaming`); only the lower triangle
    // is populated. Shard reads go through the cached `read_shard_arc` (T4.4).
    let (mut cov, col_sums) = accumulate_covariance_streaming(source, n_vars)?;

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

    // Post-processing touches only the lower triangle (`j <= i`): the upper triangle
    // is intentionally left unpopulated since the eigensolver reads `Side::Lower`.
    if let Some(ref mu) = means {
        // C -= n_obs * (mu^T @ mu)  (rank-1 correction for mean centering)
        for i in 0..n_vars {
            for j in 0..=i {
                cov[(i, j)] -= n_obs as f64 * mu[i] * mu[j];
            }
        }
    }

    // Convert to sample covariance: C /= (n-1)
    let denom = (n_obs as f64 - 1.0).max(1.0);
    for i in 0..n_vars {
        for j in 0..=i {
            cov[(i, j)] /= denom;
        }
    }

    // --- Eigendecomposition ---
    // Reads the lower triangle only; the upper triangle of `cov` is unpopulated.
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
    for shard_idx in 0..source.n_shards() {
        let csr = source.read_shard_arc(shard_idx)?;
        let _r = scx_format_io::reduction_guard();
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

    // Post-processing touches only the lower triangle (`j <= i`): the accumulator
    // populates only that triangle and the eigensolver reads `Side::Lower`.
    if let Some(ref mu) = means {
        for i in 0..n_vars {
            for j in 0..=i {
                cov[(i, j)] -= n_obs as f64 * mu[i] * mu[j];
            }
        }
    }

    // Sample covariance
    let denom = (n_obs as f64 - 1.0).max(1.0);
    for i in 0..n_vars {
        for j in 0..=i {
            cov[(i, j)] /= denom;
        }
    }

    // Eigendecomposition — reads the lower triangle only.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cov_accumulator_workers_budget_cap() {
        // One n_vars=5000 accumulator is 5000² × 8 = 200 MB.
        let mat_bytes = 5000u64 * 5000 * 8;
        // A budget that fits ~3 accumulators caps workers to 3 regardless of cores.
        assert_eq!(cov_accumulator_workers(5000, 3 * mat_bytes, 32), 3);
        // A tiny budget collapses to a single worker (serial-equivalent), never 0.
        assert_eq!(cov_accumulator_workers(5000, 1, 32), 1);
        assert_eq!(cov_accumulator_workers(5000, 0, 32), 1);
        // A generous budget is clamped to the available thread count, not exceeded.
        assert_eq!(cov_accumulator_workers(2000, u64::MAX, 8), 8);
        // Small n_vars under the default budget uses all cores.
        assert_eq!(
            cov_accumulator_workers(2000, DEFAULT_COV_MEMORY_BUDGET, 16),
            16
        );
        // max_threads is floored at 1 even if a caller passes 0.
        assert_eq!(
            cov_accumulator_workers(100, DEFAULT_COV_MEMORY_BUDGET, 0),
            1
        );
    }

    /// A simple in-memory multi-shard `ShardSource` for streaming-PCA tests.
    struct VecShardSource {
        shards: Vec<ScxCsr>,
        n_obs: usize,
        n_vars: usize,
    }

    impl ShardSource for VecShardSource {
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

    /// Build an `n_rows × 3` CSR with one near-constant large-offset column
    /// (1e6 ± 1, present in every row) plus a small-variance column. The
    /// large-offset column makes the variance minuscule relative to the
    /// magnitude being subtracted in the closed form `Σx² − nμ²`, which the
    /// relative-cancellation guard detects. `n_rows` must be a multiple of 10.
    ///
    /// Analytic centered variance (for `n_rows` a multiple of 10):
    /// - col 0: mean 1e6, dev ±1 every row → `Σ(x−μ)² = n_rows`.
    /// - col 1: values `r % 5` (mean 2, per-5-cycle Σdev² = 10) → `2·n_rows`.
    /// - total numerator `= 3·n_rows`, so variance `= 3·n_rows / (n_rows − 1)`.
    fn cancellation_csr(n_rows: usize) -> ScxCsr {
        assert!(n_rows % 10 == 0, "n_rows must be a multiple of 10");
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        for r in 0..n_rows {
            indices.push(0i32);
            data.push(1.0e6_f32 + if r % 2 == 0 { 1.0 } else { -1.0 });
            indices.push(1i32);
            data.push((r % 5) as f32);
            indptr.push(indices.len() as i64);
        }
        ScxCsr::new((n_rows, 3), indptr, indices, data).unwrap()
    }

    #[test]
    fn test_stable_centered_variance_matches_closed_form() {
        // On benign small data, the stable centered formula and the closed form
        // agree to tight tolerance (locks the equivalence into the unit suite).
        let csr = test_csr_10x5();
        let n_obs = csr.n_rows();
        let (sums, col_sum_sq) = csr.col_sums_and_sum_sq();
        let means: Vec<f64> = sums.iter().map(|&s| s / n_obs as f64).collect();

        let closed = total_variance_from_col_sq(&col_sum_sq, Some(&means), n_obs);
        let stable = stable_centered_total_variance_inmemory(&csr, &means, n_obs);
        assert!(
            (closed - stable).abs() <= 1e-9 * closed.abs().max(1.0),
            "closed-form {closed} vs stable {stable} diverged on benign data"
        );
        // Guard must NOT fire for benign data.
        assert!(!closed_form_variance_unstable(&col_sum_sq, &means, n_obs));
    }

    #[test]
    fn test_cancellation_guard_detects_degenerate_closed_form() {
        let n_obs = 100usize;
        let means = vec![1.0e6_f64];
        // Σx² == n·μ² → closed-form numerator is 0 (the degenerate boundary
        // rounding pushes ≤0 in practice); the closed form yields a non-positive
        // total variance that would zero out variance_ratio. The guard must flag it.
        let degenerate = vec![n_obs as f64 * means[0] * means[0]];
        assert!(closed_form_variance_unstable(&degenerate, &means, n_obs));
        assert!(total_variance_from_col_sq(&degenerate, Some(&means), n_obs) <= 0.0);

        // A typical scRNA-scale column (small mean, variance comparable to the
        // magnitude) is NOT flagged: mean 2, per-element variance 1 →
        // numerator/Σx² = 1/5, far above the threshold.
        let healthy_means = vec![2.0_f64];
        let healthy = vec![n_obs as f64 * (healthy_means[0] * healthy_means[0] + 1.0)];
        assert!(!closed_form_variance_unstable(
            &healthy,
            &healthy_means,
            n_obs
        ));
    }

    #[test]
    fn test_inmemory_pca_variance_ratio_stable_under_cancellation() {
        // Large-offset near-constant column drives the variance far below the
        // magnitude being subtracted in the closed form. The relative guard must
        // detect this and route through the stable centered fallback.
        let n = 200usize;
        let csr = cancellation_csr(n);
        let (sums, col_sum_sq) = csr.col_sums_and_sum_sq();
        let means: Vec<f64> = sums.iter().map(|&s| s / n as f64).collect();

        // Guard fires for this regime.
        assert!(
            closed_form_variance_unstable(&col_sum_sq, &means, n),
            "fixture should trigger the cancellation guard"
        );
        // The stable fallback recovers the analytic variance 3·n / (n−1).
        let stable = stable_centered_total_variance_inmemory(&csr, &means, n);
        let expected = 3.0 * n as f64 / (n as f64 - 1.0);
        assert!(
            (stable - expected).abs() <= 1e-6 * expected,
            "stable variance {stable} != analytic {expected}"
        );

        // The PCA path uses the fallback: non-degenerate variance_ratio, and the
        // implied total variance (variance_explained / variance_ratio) matches the
        // stable value rather than a cancellation-corrupted one.
        let result = randomized_pca_inmemory(&csr, 2, 5, 2, true, 42).unwrap();
        assert!(
            result.variance_ratio.iter().any(|&r| r > 0.0),
            "variance_ratio should be non-degenerate, got {:?}",
            result.variance_ratio
        );
        let implied_total = result.variance_explained[0] / result.variance_ratio[0];
        assert!(
            (implied_total - stable).abs() <= 1e-3 * stable,
            "PCA implied total variance {implied_total} != stable {stable}"
        );
    }

    #[test]
    fn test_streaming_pca_variance_ratio_stable_under_cancellation() {
        // Same pathological data, split across 2 shards, through the streaming path.
        let full = cancellation_csr(200);
        let s0 = full.row_slice(0, 100).unwrap();
        let s1 = full.row_slice(100, 200).unwrap();
        let source = VecShardSource {
            shards: vec![s0, s1],
            n_obs: 200,
            n_vars: full.n_cols(),
        };
        let result = randomized_pca(&source, 2, 5, 2, true, 42).unwrap();
        let ratio_sum: f64 = result.variance_ratio.iter().sum();
        assert!(
            ratio_sum > 0.0 && result.variance_ratio.iter().any(|&r| r > 0.0),
            "streaming variance_ratio should be non-degenerate, got {:?}",
            result.variance_ratio
        );
    }

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

        // Check covariance matrices match. The sparse accumulators populate only
        // the lower triangle (row >= col) — the eigensolver reads `Side::Lower` —
        // so compare against the dense GEMM (which is fully symmetric) on the
        // lower triangle only.
        for i in 0..n_vars {
            for j in 0..=i {
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

    /// `pflog_total_variance` (both branches) must match a brute-force dense
    /// computation of `Σ Z² / (n−1)` (uncentered) and `Σ (Z − colmean)² / (n−1)`
    /// (centered) for the exact `Z = delta + baseline·1ᵀ`. Guards the simplified
    /// centered closed form `S_dd − D·B2 − n·Σμ²`. The closed form is
    /// transform-agnostic (it holds for any `delta` with `b_i = −Σδ/D`), so an
    /// arbitrary `delta` exercises it regardless of the v4 formula.
    #[test]
    fn pflog_total_variance_matches_dense() {
        // Arbitrary dense `delta`; baseline must be the true PFlog baseline
        // (`b_i = −(1/D) Σ_j delta_ij`) — the closed form relies on that identity.
        let delta = [
            [0.10f64, -0.30, 0.50, 0.20],
            [-0.40, 0.10, 0.00, 0.60],
            [0.25, 0.25, -0.15, -0.05],
            [0.70, -0.20, 0.10, -0.30],
            [-0.10, 0.40, -0.50, 0.30],
        ];
        let n_obs = delta.len();
        let n_vars = delta[0].len();
        let d = n_vars as f64;
        let baseline: Vec<f64> = delta
            .iter()
            .map(|row| -row.iter().sum::<f64>() / d)
            .collect();

        // Column stats fed to the kernel.
        let mut col_sum_sq = vec![0.0f64; n_vars];
        let mut col_mean = vec![0.0f64; n_vars];
        for row in &delta {
            for (j, &v) in row.iter().enumerate() {
                col_sum_sq[j] += v * v;
                col_mean[j] += v;
            }
        }
        for m in &mut col_mean {
            *m /= n_obs as f64;
        }
        let denom = (n_obs as f64 - 1.0).max(1.0);

        // Brute-force Z = delta + baseline.
        let z: Vec<Vec<f64>> = delta
            .iter()
            .enumerate()
            .map(|(i, row)| row.iter().map(|&v| v + baseline[i]).collect())
            .collect();

        // Uncentered: Σ Z² / (n−1).
        let ss_uncentered: f64 = z.iter().flat_map(|r| r.iter().map(|&v| v * v)).sum();
        let got_uncentered = pflog_total_variance(&col_sum_sq, None, &baseline, n_obs, n_vars);
        assert!(
            (got_uncentered - ss_uncentered / denom).abs() < 1e-9,
            "uncentered {got_uncentered} != {}",
            ss_uncentered / denom
        );

        // Centered: subtract per-column mean of Z, then Σ²/(n−1).
        let mut zmu = vec![0.0f64; n_vars];
        for row in &z {
            for (j, &v) in row.iter().enumerate() {
                zmu[j] += v;
            }
        }
        for m in &mut zmu {
            *m /= n_obs as f64;
        }
        let ss_centered: f64 = z
            .iter()
            .flat_map(|r| r.iter().enumerate().map(|(j, &v)| (v - zmu[j]).powi(2)))
            .sum();
        let got_centered =
            pflog_total_variance(&col_sum_sq, Some(&col_mean), &baseline, n_obs, n_vars);
        assert!(
            (got_centered - ss_centered / denom).abs() < 1e-9,
            "centered {got_centered} != {}",
            ss_centered / denom
        );
    }
}
