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
//!
//! # Determinism
//!
//! **The same input gives the same bits on every run.** That was not true before
//! this module's reductions were rewritten: the covariance build and the
//! transpose SpMM accumulated into per-worker buffers whose row→worker assignment
//! was decided by rayon work-stealing and whose merge order was
//! `ThreadLocal::iter_mut()`, so five consecutive runs of the same call produced
//! five different results.
//!
//! Both now **partition their output** instead of their input ([`colblocks`]):
//! each worker owns a disjoint slice of the covariance triangle (or of `Z`),
//! scans every row, and takes only the nonzeros landing in its slice. Nothing is
//! merged, so the schedule cannot reach the result — entry `(i, j)` sums rows in
//! ascending order at any block count, on any thread count, at any prefetch
//! depth. Shards arrive in order from [`crate::prefetch::for_each_shard_ordered`],
//! which is what extends that from one shard to the whole matrix. The forward
//! SpMM was already row-disjoint and needed no change.
//!
//! Two consequences worth stating plainly:
//!
//! - Peak memory fell with it. There is one `n_vars × n_vars` covariance
//!   accumulator and one `n_vars × k` transpose accumulator, not one of each per
//!   worker — which is why `SCX_PCA_COV_MEMORY_BUDGET` no longer has anything to
//!   cap (see [`warn_if_cov_memory_budget_set`]).
//! - The reductions are only *half* the op. faer's dense QR and self-adjoint
//!   eigendecomposition block their work by the ambient rayon width, so their low
//!   bits move when it does. They are stable run to run at a fixed width, which
//!   is all the defect above required — and it is the same contract numpy/scipy
//!   give, where LAPACK's bits likewise move with `OMP_NUM_THREADS`. Callers who
//!   need identity *across* thread counts set `SCX_ACCEL_DETERMINISTIC_LINALG=1`;
//!   see [`pin_linalg_if_requested`] for what that costs and why it is opt-in.
//!
//! The partitioned kernels require **canonical** CSR rows (strictly increasing
//! column indices). SCX writers guarantee it and `scx_engine::project_csr`
//! already documents the same precondition; a non-canonical CSR falls back to a
//! serial accumulation that is correct — including the coalesced value for
//! duplicate coordinates — and deterministic, but single-threaded. It is *not*
//! bit-identical to the partitioned kernel: that kernel cannot run on such input
//! at all, and summing a row in stored rather than ascending order lands on
//! different low bits.

use faer::{Mat, MatRef};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use scx_format_io::ShardSource;
use scx_sparse::total_variance_from_col_sq;

use scx_sparse::ScxCsr;

use super::colblocks;
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
    /// Whether the GPU power loop held the whole matrix **device-resident**
    /// (`Some(true)`) or fell back to the streaming operator (`Some(false)`).
    /// `None` on CPU paths, which have no residency decision to make.
    ///
    /// Recorded because the decision is dynamic — taken against *free* VRAM at
    /// call time, so the same script on the same data can take either path
    /// depending on what else is on the card. `SCX_GPU_PCA_RESIDENT=0` pins the
    /// streaming arm.
    pub resident_csr: Option<bool>,
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

/// Accumulate the sparse outer product `Xᵀ X` (lower triangle) and column sums
/// from CSR nonzeros, **serially, in row order**.
///
/// `cov` is `n_vars × n_vars` column-major: entry `(hi, lo)` lives at
/// `lo * n_vars + hi`. Only the lower triangle (`hi >= lo`) is written — the
/// eigensolver reads `Side::Lower`, so populating the upper triangle would
/// double the inner-loop stores for nothing.
///
/// Makes no assumption about index order within a row, which is what lets it be
/// both the bit-exact oracle for [`accumulate_covariance_into`] and its fallback
/// on a non-canonical CSR.
///
/// # Duplicate coordinates
///
/// A row may legally list the same column twice (scipy's `has_canonical_format
/// == False`), in which case the cell's value is the *sum* of the entries. The
/// pair loop handles that without materialising the sum: for `c1 == c2` the
/// coalesced diagonal is `(v1 + v2)² = v1² + v2² + 2·v1·v2`, so the cross term
/// counts twice — once for each ordering — where a genuine off-diagonal pair
/// counts once. Off-diagonal entries need no special case: every duplicate
/// contributes its own product, and those already sum to the coalesced value.
///
/// Missing that doubling is a wrong *answer*, not a rounding difference: a row
/// holding column 0 twice with values 1 and 2 must give `(XᵀX)[0,0] = 9`, and
/// counting the cross term once gives 7.
fn accumulate_covariance_serial(
    csr: &ScxCsr,
    cov: &mut [f64],
    col_sums: &mut [f64],
    n_vars: usize,
) {
    for r in 0..csr.n_rows() {
        let start = csr.indptr[r] as usize;
        let end = csr.indptr[r + 1] as usize;
        for idx in start..end {
            col_sums[csr.indices[idx] as usize] += csr.data[idx] as f64;
        }
        for i in start..end {
            let c1 = csr.indices[i] as usize;
            let v1 = csr.data[i] as f64;
            cov[c1 * n_vars + c1] += v1 * v1; // diagonal
            for j in (i + 1)..end {
                let c2 = csr.indices[j] as usize;
                let v2 = csr.data[j] as f64;
                let prod = v1 * v2;
                if c1 == c2 {
                    // Duplicate coordinate: both orderings land on the diagonal.
                    cov[c1 * n_vars + c1] += prod + prod;
                    continue;
                }
                // Write the lower-triangle entry only (row >= col).
                let (lo, hi) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
                cov[lo * n_vars + hi] += prod;
            }
        }
    }
}

/// Accumulate the covariance columns `[a, b)` and their column sums from a
/// **strictly increasing** CSR.
///
/// `cov_part` is the `(b − a)`-column, `n_vars`-row column-major slice the block
/// owns; `sum_part` is `col_sums[a..b]`. Because the row's indices ascend, every
/// pair `(c_i, c_j)` with `j >= i` already has `c_j >= c_i`, so the pair's
/// lower-triangle home is `(hi, lo) = (c_j, c_i)` with no comparison, and a pair
/// belongs to this block exactly when `c_i ∈ [a, b)` — a contiguous window found
/// by two binary searches. `j` starts at `i`, so the diagonal falls out of the
/// same loop.
#[inline]
fn accumulate_covariance_block(
    csr: &ScxCsr,
    cov_part: &mut [f64],
    sum_part: &mut [f64],
    n_vars: usize,
    a: usize,
    b: usize,
) {
    if a >= b {
        return;
    }
    for r in 0..csr.n_rows() {
        let start = csr.indptr[r] as usize;
        let end = csr.indptr[r + 1] as usize;
        let row_idx = &csr.indices[start..end];
        let row_dat = &csr.data[start..end];
        let (lo, hi) = colblocks::sorted_subrange(row_idx, a, b);
        for i in lo..hi {
            let c1 = row_idx[i] as usize;
            let v1 = row_dat[i] as f64;
            sum_part[c1 - a] += v1;
            let cov_col = &mut cov_part[(c1 - a) * n_vars..(c1 - a + 1) * n_vars];
            for (&c2, &v2) in row_idx[i..].iter().zip(&row_dat[i..]) {
                cov_col[c2 as usize] += v1 * v2 as f64;
            }
        }
    }
}

/// Fold one shard's contribution into the shared covariance lower triangle and
/// column sums, partitioning the **output** across rayon workers.
///
/// Each block owns a disjoint contiguous slice of `cov` (and of `col_sums`), so
/// nothing is merged and the split cannot reach the result: entry `(hi, lo)`
/// accumulates rows in ascending order for any block count on any thread count.
/// Callers get bit-identical output from a 1-thread and a 64-thread run, and the
/// accumulator is one `n_vars × n_vars` matrix rather than one per worker.
///
/// A non-canonical CSR (unsorted or duplicated column indices) breaks the
/// `c_j >= c_i` step the partition rests on; that falls back to
/// [`accumulate_covariance_serial`], which computes the correct value —
/// duplicates coalesced — deterministically, but single-threaded. Not
/// "bit-identical": there is nothing to be identical *to*, since the partitioned
/// kernel cannot run on this input, and a row summed in stored rather than
/// ascending order lands on different low bits.
fn accumulate_covariance_into(csr: &ScxCsr, cov: &mut [f64], col_sums: &mut [f64], n_vars: usize) {
    if csr.n_rows() == 0 || csr.indices.is_empty() {
        return;
    }
    if !colblocks::rows_strictly_increasing(csr) {
        warn_non_canonical_csr_once();
        accumulate_covariance_serial(csr, cov, col_sums, n_vars);
        return;
    }

    let n_blocks = colblocks::block_count(n_vars);
    if n_blocks == 1 {
        accumulate_covariance_block(csr, cov, col_sums, n_vars, 0, n_vars);
        return;
    }

    // Exact per-column op count: the nonzero at position `i` of a row drives
    // `end − i` updates, all of them in column `c_i`'s block. `O(nnz)` against
    // the `O(Σ mᵣ²)` main loop, and integer, so the plan is a pure function of
    // the data — it is a load-balancing hint only, never part of the result.
    let mut weights = vec![0u64; n_vars];
    for r in 0..csr.n_rows() {
        let start = csr.indptr[r] as usize;
        let end = csr.indptr[r + 1] as usize;
        for i in start..end {
            weights[csr.indices[i] as usize] += (end - i) as u64;
        }
    }
    let blocks = colblocks::plan_blocks(&weights, n_blocks);
    let cov_parts = colblocks::split_by_blocks(cov, &blocks, n_vars);
    let sum_parts = colblocks::split_by_blocks(col_sums, &blocks, 1);

    blocks
        .par_iter()
        .zip(cov_parts)
        .zip(sum_parts)
        .for_each(|((block, cov_part), sum_part)| {
            accumulate_covariance_block(csr, cov_part, sum_part, n_vars, block.start, block.end);
        });
}

/// Turn the accumulated cross-product `Xᵀ X` into the sample covariance, in
/// place: apply the rank-1 mean-centering correction (when centering) and divide
/// by `n_obs − 1`.
///
/// Touches only the lower triangle (`hi >= lo`), which is all the accumulator
/// populated and all `self_adjoint_eigen(Side::Lower)` reads. Iterating `hi`
/// inside `lo` walks the column-major buffer contiguously; the arithmetic per
/// entry — subtract, then divide — is unchanged.
fn finalize_covariance(cov: &mut [f64], means: Option<&[f64]>, n_obs: usize, n_vars: usize) {
    let n_obs_f = n_obs as f64;
    let denom = (n_obs_f - 1.0).max(1.0);
    for lo in 0..n_vars {
        let col = &mut cov[lo * n_vars..(lo + 1) * n_vars];
        match means {
            Some(mu) => {
                let m_lo = mu[lo];
                for (hi, slot) in col.iter_mut().enumerate().skip(lo) {
                    *slot = (*slot - n_obs_f * mu[hi] * m_lo) / denom;
                }
            }
            None => {
                for slot in col.iter_mut().skip(lo) {
                    *slot /= denom;
                }
            }
        }
    }
}

/// One-shot notice that a CSR reached the covariance build without canonical
/// rows. Not an error — the answer is identical — but the caller loses the
/// parallel path and should know why.
fn warn_non_canonical_csr_once() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        log::warn!(
            "covariance PCA: a CSR row's column indices are not strictly increasing; \
             falling back to a serial accumulation. The values are correct and still \
             deterministic, but this path does not parallelize. To restore it, canonicalize \
             the matrix: scipy `sort_indices()` for out-of-order columns and \
             `sum_duplicates()` for repeated ones — sorting alone leaves a duplicate \
             coordinate on this same serial path"
        );
    });
}

/// Pure parse of `SCX_ACCEL_DETERMINISTIC_LINALG`: `1` / `true` / `yes` / `on`
/// (case-insensitive) opt in, everything else — including unset — does not.
fn parse_deterministic_linalg(raw: Option<String>) -> bool {
    matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

/// Pin faer's dense decompositions to sequential execution, if asked.
///
/// # Why this is opt-in
///
/// SCX's own PCA reductions are bit-identical at any thread count — they
/// partition their output, so the schedule cannot reach the result. faer's dense
/// QR and self-adjoint eigendecomposition are different: they block their work by
/// the **ambient rayon width**, so their low bits move when the pool width does.
/// Measured on this tree: a thin QR's Q differs in 14 818 of 48 000 elements
/// between a 1- and a 2-thread pool, and a 400-variable EVD in essentially every
/// element; below roughly 64 variables the EVD does not parallelize and agrees.
///
/// Both are perfectly stable **run to run at a fixed width**, which is why the
/// default leaves them alone — that is the same contract numpy/scipy give, where
/// LAPACK's bits likewise move with `OMP_NUM_THREADS`. It is also enough to fix
/// the defect this all started from: repeated runs of one script on one machine
/// now agree exactly.
///
/// Callers who need identity *across* thread counts — comparing a laptop run
/// against a cluster run, say — set `SCX_ACCEL_DETERMINISTIC_LINALG=1` and pay
/// for it: the covariance route's `n_vars × n_vars` EVD is ~2.3× slower
/// sequential (measured at n_vars = 2000), while the randomized route's thin QR
/// is actually ~1.65× *faster*, sequential having less overhead at k ≈ 60.
///
/// # Why it is set once and never restored
///
/// faer's parallelism lives in a process-wide `AtomicUsize`, so setting it around
/// each call and restoring it afterwards would race any other faer user in the
/// process. Reading the knob once and applying it once is the only form of this
/// that is not a data race.
///
/// # What it guarantees, and what it merely touches
///
/// These are different sets, and conflating them is how the first version of
/// this comment got it wrong.
///
/// **Guaranteed:** CPU PCA and PFlog only. It is initialised from those five
/// entry points, so nothing is pinned until one of them runs, and only their
/// determinism is tested.
///
/// **Touched:** every *implicit* faer decomposition in the process, because the
/// setting is one global. After the first pinned PCA call, Harmony's LU fallback
/// (`harmony/cpu.rs`), NB-GLM's LLT/LU/QR (`nb_glm/{irls,dispersion,wald,validate}.rs`)
/// and the native-GPU PCA's host SVD (`scx-gpu`) all run sequentially too —
/// where before that call they did not. That is a real call-order-dependent
/// performance side effect, and it does *not* buy those ops determinism.
///
/// **Not touched:** call sites passing an **explicit** `Par`. The exact-kNN gemm
/// (`neighbors/cpu.rs`) and the eval-metrics distance gemm
/// (`eval_metrics/distances.rs`) hand faer `Par::rayon(0)` directly and ignore
/// the global entirely, so pinning cannot make them sequential or reproducible.
fn pin_linalg_if_requested() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if parse_deterministic_linalg(std::env::var("SCX_ACCEL_DETERMINISTIC_LINALG").ok()) {
            faer::set_global_parallelism(faer::Par::Seq);
            log::info!(
                "SCX_ACCEL_DETERMINISTIC_LINALG is set: pinning faer's dense decompositions \
                 to sequential execution, so CPU PCA / PFlog results are bit-identical \
                 across thread counts. The setting is a faer process-global, so every \
                 implicit faer decomposition (Harmony, NB-GLM, native-GPU PCA's host SVD) \
                 also runs sequentially from here on; call sites passing an explicit Par \
                 are unaffected. Expect a slower eigendecomposition on the covariance route"
            );
        }
    });
}

/// One-shot notice that `SCX_PCA_COV_MEMORY_BUDGET` no longer does anything.
///
/// It used to cap how many concurrent `n_vars × n_vars` f64 accumulators the
/// covariance build held, back when there was one per worker. The build now
/// partitions the output and holds exactly one regardless of thread count, so
/// there is nothing left to cap — but a user who set the knob deserves to be
/// told that rather than have it silently ignored.
fn warn_if_cov_memory_budget_set() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var_os("SCX_PCA_COV_MEMORY_BUDGET").is_some() {
            log::warn!(
                "SCX_PCA_COV_MEMORY_BUDGET is set but no longer has any effect: covariance \
                 PCA accumulates into a single shared n_vars × n_vars matrix instead of one \
                 per worker, so peak memory no longer scales with the thread count"
            );
        }
    });
}

/// Decode-prefetch depth for the streaming PCA passes.
///
/// The shared [`prefetch_depth`](crate::prefetch::prefetch_depth) —
/// `SCX_ACCEL_PREFETCH_DEPTH`, default 4, already capped by the rayon pool
/// width — additionally lowered by `SCX_ACCEL_NUM_THREADS` when that is set.
///
/// PCA's reductions honour `SCX_ACCEL_NUM_THREADS` through
/// [`colblocks::block_count`]; leaving decode-prefetch outside it would quietly
/// stop that knob from bounding PCA's concurrency at all.
///
/// `pyscx` calls this too, to size the slice of `pca(memory_budget=…)` it must
/// reserve for decoded-but-unconsumed shards — the two have to agree, so there
/// is exactly one definition.
pub fn pca_prefetch_depth() -> usize {
    crate::prefetch::prefetch_depth()
        .min(crate::mem_budget::accel_num_threads().unwrap_or(usize::MAX))
        .max(1)
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
pub fn randomized_pca<S: ShardSource + Sync + ?Sized>(
    source: &S,
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    seed: u64,
) -> Result<PcaResult> {
    // Resolved once for the whole run: every pass below shares one depth, and
    // `pyscx` reserves the matching slice of `memory_budget` from the same fn.
    randomized_pca_with_depth(
        source,
        n_components,
        n_oversamples,
        n_power_iterations,
        zero_center,
        seed,
        pca_prefetch_depth(),
    )
}

/// [`randomized_pca`] with an explicit decode-prefetch depth.
///
/// **The resource-explicit entry point.** `depth` is how many decoded shards the
/// pipeline may hold at once, so it is a memory knob: a binding that has a byte
/// budget (pyscx's `pca(memory_budget=…)`) resolves the depth against it and
/// passes the result here, instead of letting the process-wide default apply to
/// a path whose footprint it never sized. [`randomized_pca`] is the convenience
/// wrapper that takes [`pca_prefetch_depth`].
///
/// Also what lets the equivalence tests A/B depth 1 against depth 4 **in one
/// process** — [`prefetch_depth`](crate::prefetch::prefetch_depth) is a
/// `OnceLock`, so the environment knob cannot be flipped after the first read.
#[allow(clippy::too_many_arguments)]
pub fn randomized_pca_with_depth<S: ShardSource + Sync + ?Sized>(
    source: &S,
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    seed: u64,
    depth: usize,
) -> Result<PcaResult> {
    let (n_obs, n_vars) = source.shape();
    validate_inputs(n_obs, n_vars, n_components)?;
    warn_if_cache_undersized(source, "randomized_pca");
    pin_linalg_if_requested();

    let k = (n_components + n_oversamples).min(n_vars).min(n_obs);
    // Fused pass: compute column means and sum-of-squares together (1 shard pass)
    let (means, col_sum_sq) =
        scx_format_io::col_means_and_sum_sq_prefetched(source, zero_center, depth)?;
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
    let y = streaming_spmm_forward(source, &omega, k, means_ref, depth)?;
    let mut q = qr_thin_q_row_major(&y, n_obs, k);

    for _ in 0..n_power_iterations {
        let b = streaming_spmm_transpose(source, &q, means_ref, depth)?;
        if n_power_iterations > 2 {
            // Full QR normalization on transpose result for numerical stability
            let q_b = qr_thin_q_row_major(&b, n_vars, k);
            let q_b_rm = mat_to_row_major_buf(&q_b);
            let y = streaming_spmm_forward(source, &q_b_rm, k, means_ref, depth)?;
            q = qr_thin_q_row_major(&y, n_obs, k);
        } else {
            // Skip QR on b — feed row-major b directly into forward SpMM
            let y = streaming_spmm_forward(source, &b, k, means_ref, depth)?;
            q = qr_thin_q_row_major(&y, n_obs, k);
        }
    }

    // Step 6: B = (X - μ)^T @ Q
    let b_rm = streaming_spmm_transpose(source, &q, means_ref, depth)?;

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
            crate::prefetch::for_each_shard_ordered(source, depth, |_idx, csr| {
                accumulate_centered_ss(&csr, mu, &mut total, &mut col_nnz);
                Ok(())
            })?;
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
fn streaming_spmm_forward_offset<S: ShardSource + Sync + ?Sized>(
    source: &S,
    m_data: &[f64],
    k: usize,
    means: Option<&[f64]>,
    baseline: &[f64],
    depth: usize,
) -> Result<Vec<f64>> {
    let (n_obs, n_vars) = source.shape();
    debug_assert_eq!(m_data.len(), n_vars * k);
    debug_assert_eq!(baseline.len(), n_obs);

    let mut y = streaming_spmm_forward(source, m_data, k, means, depth)?;

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
fn streaming_spmm_transpose_offset<S: ShardSource + Sync + ?Sized>(
    source: &S,
    q: &Mat<f64>,
    means: Option<&[f64]>,
    baseline: &[f64],
    depth: usize,
) -> Result<Vec<f64>> {
    let (n_obs, n_vars) = source.shape();
    let k = q.ncols();
    debug_assert_eq!(q.nrows(), n_obs);
    debug_assert_eq!(baseline.len(), n_obs);

    let mut z = streaming_spmm_transpose(source, q, means, depth)?;

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
pub fn pflog_pca<S: ShardSource + Sync + ?Sized>(
    delta_source: &S,
    baseline: &[f64],
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    seed: u64,
) -> Result<PcaResult> {
    pflog_pca_with_depth(
        delta_source,
        baseline,
        n_components,
        n_oversamples,
        n_power_iterations,
        zero_center,
        seed,
        pca_prefetch_depth(),
    )
}

/// [`pflog_pca`] with an explicit decode-prefetch depth — see
/// [`randomized_pca_with_depth`] for why this seam exists.
#[allow(clippy::too_many_arguments)]
pub fn pflog_pca_with_depth<S: ShardSource + Sync + ?Sized>(
    delta_source: &S,
    baseline: &[f64],
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    seed: u64,
    depth: usize,
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
    pin_linalg_if_requested();

    let k = (n_components + n_oversamples).min(n_vars).min(n_obs);

    // Column means + sum-of-squares of `delta` (one pass). The column mean of Z
    // adds mean(baseline) uniformly to colmean(delta).
    let (colmean_delta, col_sum_sq_delta) =
        scx_format_io::col_means_and_sum_sq_prefetched(delta_source, zero_center, depth)?;
    let baseline_mean = baseline.iter().sum::<f64>() / (n_obs as f64).max(1.0);
    let means: Option<Vec<f64>> = colmean_delta
        .as_ref()
        .map(|cd| cd.iter().map(|&m| m + baseline_mean).collect());
    let means_ref = means.as_deref();

    let omega = random_gaussian(n_vars, k, seed);
    let y = streaming_spmm_forward_offset(delta_source, &omega, k, means_ref, baseline, depth)?;
    let mut q = qr_thin_q_row_major(&y, n_obs, k);

    for _ in 0..n_power_iterations {
        let b = streaming_spmm_transpose_offset(delta_source, &q, means_ref, baseline, depth)?;
        if n_power_iterations > 2 {
            let q_b = qr_thin_q_row_major(&b, n_vars, k);
            let q_b_rm = mat_to_row_major_buf(&q_b);
            let y = streaming_spmm_forward_offset(
                delta_source,
                &q_b_rm,
                k,
                means_ref,
                baseline,
                depth,
            )?;
            q = qr_thin_q_row_major(&y, n_obs, k);
        } else {
            let y = streaming_spmm_forward_offset(delta_source, &b, k, means_ref, baseline, depth)?;
            q = qr_thin_q_row_major(&y, n_obs, k);
        }
    }

    let b_rm = streaming_spmm_transpose_offset(delta_source, &q, means_ref, baseline, depth)?;
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
    pin_linalg_if_requested();

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
fn warn_if_cache_undersized<S: ShardSource + ?Sized>(source: &S, op: &str) {
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
///
/// ⚠️ This is the one implicit-zero count in the tree that still **absorbs** a
/// non-canonical shard instead of reporting it. Everywhere else — the six
/// `BackedCsrReader` statistics, their projected and masked twins in pyscx, and
/// the five `ScxCsr` methods — routes through
/// `scx_sparse::implicit_zero_count`, which returns
/// `CsrError::NonCanonicalAxis` when a column claims more stored entries than
/// there are rows. Here `saturating_sub` clamps it to zero and the PCA
/// continues.
///
/// That is deliberate, not an oversight: this function returns a bare `f64`,
/// and so does `stable_centered_total_variance_inmemory` below it, so giving it
/// an error channel means threading `Result` through the PCA result path. The
/// clamp is at least bounded — it never produced the ~1.8e19 wrap that the
/// unguarded subtractions did. Worth revisiting whenever that path grows a
/// `Result` for another reason.
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
fn streaming_spmm_forward<S: ShardSource + Sync + ?Sized>(
    source: &S,
    m_data: &[f64],
    k: usize,
    means: Option<&[f64]>,
    depth: usize,
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

    // Ordered delivery keeps `global_row` advancing in shard order, and each
    // output row is written by exactly one thread over that row's nonzeros in
    // index order — so this stays **bit-identical** to the sequential loop.
    let mut global_row = 0usize;
    crate::prefetch::for_each_shard_ordered(source, depth, |_shard_idx, csr| {
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
        Ok(())
    })?;

    Ok(y)
}

/// Column sums of `Q`: `sum_q[j] = Σ_r Q[r, j]`.
///
/// The column-centering rank-1 term needs this, and it does not depend on `X` at
/// all — so it is computed once per call rather than folded into the shard loop,
/// where it used to ride the same work-stealing reduction the rest of the pass
/// did. Parallel over `j`: each output element sums rows in ascending order on a
/// single worker, and `Q` is column-major so each `j` walks contiguous memory.
fn column_sums_of_q(q: &Mat<f64>) -> Vec<f64> {
    let (n_obs, k) = (q.nrows(), q.ncols());
    let mut sum_q = vec![0.0f64; k];
    sum_q.par_iter_mut().enumerate().for_each(|(j, slot)| {
        let mut acc = 0.0f64;
        for r in 0..n_obs {
            acc += q[(r, j)];
        }
        *slot = acc;
    });
    sum_q
}

/// `Z -= μ ⊗ (1ᵀ Q)` — the column-centering rank-1 term, applied once after all
/// shards have been folded in.
fn apply_transpose_mean_correction(z: &mut [f64], means: &[f64], sum_q: &[f64], k: usize) {
    debug_assert_eq!(z.len(), means.len() * k);
    // `chunks_mut(0)` panics, so record why it cannot happen rather than adding an
    // early return: an early return would turn a genuinely broken caller into a
    // silent no-op. Every PCA entry point calls `validate_inputs`, which rejects
    // `n_components == 0` and empty axes, and `k` is at least `n_components`
    // clamped to two non-zero axes — so `k >= 1` for every reachable call.
    debug_assert!(k > 0, "k must be non-zero — validate_inputs guarantees it");
    for (v, row) in z.chunks_mut(k).enumerate() {
        let mu = means[v];
        for (slot, &sq) in row.iter_mut().zip(sum_q) {
            *slot -= mu * sq;
        }
    }
}

/// Accumulate the variable rows `[a, b)` of `Z = Xᵀ Q` from one CSR.
///
/// `z_part` is the `(b − a)`-row, `k`-column row-major slice this block owns, and
/// `q_row` is caller-provided scratch of length `k` so the row gather allocates
/// once per block rather than once per row.
///
/// Each nonzero contributes to exactly one variable row, so the blocks' index
/// windows *tile* each CSR row: unlike the covariance kernel there is no read
/// amplification, and total work is independent of the block count. Element
/// `(v, j)` accumulates rows in ascending order at any width, which is what makes
/// the split invisible in the result.
#[allow(clippy::too_many_arguments)]
#[inline]
fn spmm_transpose_block(
    csr: &ScxCsr,
    q: &Mat<f64>,
    row_base: usize,
    z_part: &mut [f64],
    k: usize,
    a: usize,
    b: usize,
    sorted: bool,
    q_row: &mut [f64],
) {
    if a >= b {
        return;
    }
    for r in 0..csr.n_rows() {
        let start = csr.indptr[r] as usize;
        let end = csr.indptr[r + 1] as usize;
        let row_idx = &csr.indices[start..end];
        let row_dat = &csr.data[start..end];
        // A canonical row's columns in `[a, b)` form a contiguous window. A
        // non-canonical one does not, so scan the whole row and test membership
        // — slower, same bits, and still deterministic.
        let (lo, hi) = if sorted {
            colblocks::sorted_subrange(row_idx, a, b)
        } else {
            (0, row_idx.len())
        };
        if lo == hi {
            continue;
        }
        let gr = row_base + r;
        for (j, slot) in q_row.iter_mut().enumerate() {
            *slot = q[(gr, j)];
        }
        for (&col, &val) in row_idx[lo..hi].iter().zip(&row_dat[lo..hi]) {
            let col = col as usize;
            if !sorted && !(a..b).contains(&col) {
                continue;
            }
            let val = val as f64;
            let z_off = (col - a) * k;
            for (j, &qj) in q_row.iter().enumerate() {
                z_part[z_off + j] += val * qj;
            }
        }
    }
}

/// Fold one CSR's contribution into the shared `Z = Xᵀ Q` accumulator
/// (`n_vars × k`, row-major), partitioning the **variable axis** across rayon
/// workers.
///
/// `row_base` is the CSR's first global observation row, so the same kernel
/// serves the streaming loop (one call per shard, `row_base` advancing) and the
/// in-memory entry (one call at `row_base = 0`).
///
/// Blocks own disjoint slices of `z`, so nothing is merged and the accumulator is
/// one `n_vars × k` buffer instead of one per worker.
fn spmm_transpose_into(
    csr: &ScxCsr,
    q: &Mat<f64>,
    row_base: usize,
    z: &mut [f64],
    k: usize,
    n_vars: usize,
) {
    if csr.n_rows() == 0 || csr.indices.is_empty() {
        return;
    }
    let sorted = colblocks::rows_strictly_increasing(csr);
    // Below this the rayon overhead outweighs the work; one block is the same
    // arithmetic in the same order, so the threshold is invisible in the result.
    let n_blocks = if csr.n_rows() * k > 10_000 {
        colblocks::block_count(n_vars)
    } else {
        1
    };

    if n_blocks == 1 {
        let mut q_row = vec![0.0f64; k];
        spmm_transpose_block(csr, q, row_base, z, k, 0, n_vars, sorted, &mut q_row);
        return;
    }

    // Exact per-column op count: each nonzero in column `c` drives `k` updates,
    // all in variable row `c`. Load balancing only — see `colblocks`.
    let mut weights = vec![0u64; n_vars];
    for &c in &csr.indices {
        weights[c as usize] += 1;
    }
    let blocks = colblocks::plan_blocks(&weights, n_blocks);
    let z_parts = colblocks::split_by_blocks(z, &blocks, k);

    blocks.par_iter().zip(z_parts).for_each(|(block, z_part)| {
        let mut q_row = vec![0.0f64; k];
        spmm_transpose_block(
            csr,
            q,
            row_base,
            z_part,
            k,
            block.start,
            block.end,
            sorted,
            &mut q_row,
        );
    });
}

/// Streaming transpose SpMM: Z = (X - μ)^T @ Q, shard-by-shard.
///
/// Q is `Mat<f64>` (n_obs × k), column-major. Shards arrive in order through the
/// decode-prefetch pipeline and each folds into a single shared accumulator whose
/// variable rows are partitioned across workers (see [`spmm_transpose_into`]).
/// Returns row-major `Vec<f64>` of shape (n_vars × k).
fn streaming_spmm_transpose<S: ShardSource + Sync + ?Sized>(
    source: &S,
    q: &Mat<f64>,
    means: Option<&[f64]>,
    depth: usize,
) -> Result<Vec<f64>> {
    let (n_obs, n_vars) = source.shape();
    let k = q.ncols();
    debug_assert_eq!(q.nrows(), n_obs);

    let mut z = vec![0.0f64; n_vars * k];
    let mut global_row = 0usize;

    crate::prefetch::for_each_shard_ordered(source, depth, |_shard_idx, csr| {
        let _r = scx_format_io::reduction_guard();
        spmm_transpose_into(&csr, q, global_row, &mut z, k, n_vars);
        global_row += csr.n_rows();
        Ok(())
    })?;

    // Mean centering correction: Z -= μ @ (1^T @ Q)
    if let Some(mu) = means {
        apply_transpose_mean_correction(&mut z, mu, &column_sums_of_q(q), k);
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
/// Single-shard special case of [`streaming_spmm_transpose`], sharing its kernel
/// so the two routes cannot drift.
/// Returns row-major `Vec<f64>` of shape (n_vars × k).
fn spmm_transpose_csr(csr: &ScxCsr, q: &Mat<f64>, means: Option<&[f64]>) -> Vec<f64> {
    let n_vars = csr.n_cols();
    let k = q.ncols();

    let mut z = vec![0.0f64; n_vars * k];
    spmm_transpose_into(csr, q, 0, &mut z, k, n_vars);

    if let Some(mu) = means {
        apply_transpose_mean_correction(&mut z, mu, &column_sums_of_q(q), k);
    }

    z
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
        resident_csr: None,
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
pub fn covariance_pca<S: ShardSource + Sync + ?Sized>(
    source: &S,
    n_components: usize,
    zero_center: bool,
) -> Result<PcaResult> {
    covariance_pca_with_depth(source, n_components, zero_center, pca_prefetch_depth())
}

/// [`covariance_pca`] with an explicit decode-prefetch depth — see
/// [`randomized_pca_with_depth`] for why this seam exists.
pub fn covariance_pca_with_depth<S: ShardSource + Sync + ?Sized>(
    source: &S,
    n_components: usize,
    zero_center: bool,
    depth: usize,
) -> Result<PcaResult> {
    let (n_obs, n_vars) = source.shape();
    validate_inputs(n_obs, n_vars, n_components)?;
    warn_if_cache_undersized(source, "covariance_pca");
    pin_linalg_if_requested();

    warn_if_cov_memory_budget_set();

    // --- Pass 1: Accumulate covariance matrix and column sums ---
    // Sparse outer product: accumulate C[c1,c2] += v1*v2 directly from CSR nonzeros.
    // No densification — touches only nonzero entries (~2% for typical HVG-selected
    // data). Shards arrive in order through the decode-prefetch pipeline and each
    // one folds into a *single* shared accumulator whose columns are partitioned
    // across workers, so peak RAM scales with `n_vars` (≤ COVARIANCE_PCA_THRESHOLD)
    // and not with the thread count, and the result is bit-identical at any width.
    // Only the lower triangle is populated. Shard reads go through the cached
    // `read_shard_arc` (T4.4).
    let mut cov = vec![0.0f64; n_vars * n_vars];
    let mut col_sums = vec![0.0f64; n_vars];
    crate::prefetch::for_each_shard_ordered(source, depth, |_shard_idx, csr| {
        if csr.n_rows() > 0 {
            // Bind the reduction guard only for shards that actually accumulate, so
            // empty shards don't inflate the reduction call count.
            let _r = scx_format_io::reduction_guard();
            accumulate_covariance_into(&csr, &mut cov, &mut col_sums, n_vars);
        }
        Ok(())
    })?;

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

    finalize_covariance(&mut cov, means.as_deref(), n_obs, n_vars);

    // --- Eigendecomposition ---
    // Reads the lower triangle only; the upper triangle of `cov` is unpopulated.
    let evd = MatRef::from_column_major_slice(&cov, n_vars, n_vars)
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
    crate::prefetch::for_each_shard_ordered(source, depth, |_shard_idx, csr| {
        let _r = scx_format_io::reduction_guard();
        let shard_rows = csr.n_rows();

        // E[row, :] = X[row, :] @ V - mc — the same `Y = (X - μ) @ M` the
        // randomized route runs, so it uses the same kernel rather than a
        // hand-inlined copy of it. Output rows are disjoint and each row's
        // nonzeros are still consumed in index order, so this is bit-identical
        // to the serial loop it replaces; the difference is that
        // `spmm_forward_into` parallelises across rows above its work
        // threshold, and this pass was the covariance route's one entirely
        // single-threaded scan.
        spmm_forward_into(
            &csr,
            &v_rm,
            n_components,
            &mut embeddings,
            global_row,
            mean_correction.as_deref(),
        );
        global_row += shard_rows;
        Ok(())
    })?;

    Ok(PcaResult {
        embeddings,
        components,
        variance_explained,
        variance_ratio,
        mean: means,
        n_components,
        n_obs,
        n_vars,
        resident_csr: None,
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
    pin_linalg_if_requested();

    warn_if_cov_memory_budget_set();

    // --- Build covariance matrix via sparse outer products ---
    // One shared n_vars × n_vars accumulator with its columns partitioned across
    // rayon workers — the same kernel the streaming entry point folds each shard
    // through, so the two routes cannot drift.
    let mut cov = vec![0.0f64; n_vars * n_vars];
    let mut col_sums = vec![0.0f64; n_vars];
    accumulate_covariance_into(csr, &mut cov, &mut col_sums, n_vars);

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

    finalize_covariance(&mut cov, means.as_deref(), n_obs, n_vars);

    // Eigendecomposition — reads the lower triangle only.
    let evd = MatRef::from_column_major_slice(&cov, n_vars, n_vars)
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

    // Same `Y = (X - μ) @ M` the streaming route runs, through the same kernel
    // rather than a hand-inlined copy of it — output rows are disjoint and each
    // row's nonzeros are still consumed in index order, so this is bit-identical
    // to the loop it replaces (`covariance_embeddings_kernel_matches_the_inlined_scatter_bitwise`
    // is that oracle). The difference is that `spmm_forward_into` parallelises
    // across rows above its work threshold, where this pass was single-threaded.
    spmm_forward_into(
        csr,
        &v_rm,
        n_components,
        &mut embeddings,
        0,
        mean_correction.as_deref(),
    );

    Ok(PcaResult {
        embeddings,
        components,
        variance_explained,
        variance_ratio,
        mean: means,
        n_components,
        n_obs,
        n_vars,
        resident_csr: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Accumulate `csr` into a fresh lower triangle using exactly `n_blocks`
    /// column blocks — the seam that lets the tests below pin the block count
    /// instead of hoping the ambient pool width varies.
    fn covariance_with_blocks(
        csr: &ScxCsr,
        n_vars: usize,
        n_blocks: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut cov = vec![0.0f64; n_vars * n_vars];
        let mut col_sums = vec![0.0f64; n_vars];
        let blocks = colblocks::plan_blocks(&vec![1u64; n_vars], n_blocks);
        let cov_parts = colblocks::split_by_blocks(&mut cov, &blocks, n_vars);
        let sum_parts = colblocks::split_by_blocks(&mut col_sums, &blocks, 1);
        for ((block, cov_part), sum_part) in blocks.iter().zip(cov_parts).zip(sum_parts) {
            accumulate_covariance_block(csr, cov_part, sum_part, n_vars, block.start, block.end);
        }
        (cov, col_sums)
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
        assert!(n_rows.is_multiple_of(10), "n_rows must be a multiple of 10");
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
            let mut cov_col = [0.0f64; 10];
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
        let mut cov_sparse = vec![0.0f64; n_vars * n_vars];
        let mut col_sums_sparse = vec![0.0f64; n_vars];
        accumulate_covariance_serial(&csr, &mut cov_sparse, &mut col_sums_sparse, n_vars);

        // --- Sparse partitioned ---
        let mut cov_par = vec![0.0f64; n_vars * n_vars];
        let mut col_sums_par = vec![0.0f64; n_vars];
        accumulate_covariance_into(&csr, &mut cov_par, &mut col_sums_par, n_vars);

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
                    (cov_dense[(i, j)] - cov_sparse[j * n_vars + i]).abs() < 1e-10,
                    "cov mismatch at ({i},{j}): dense={}, sparse={}",
                    cov_dense[(i, j)],
                    cov_sparse[j * n_vars + i]
                );
                assert!(
                    (cov_dense[(i, j)] - cov_par[j * n_vars + i]).abs() < 1e-10,
                    "cov mismatch at ({i},{j}): dense={}, par={}",
                    cov_dense[(i, j)],
                    cov_par[j * n_vars + i]
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

    // -----------------------------------------------------------------------
    // Decode-prefetch: engagement, and equivalence at depth 1 vs depth 4
    // -----------------------------------------------------------------------

    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::thread::ThreadId;

    /// A `ShardSource` that records **which thread** decoded each shard.
    ///
    /// This is the anti-trap instrument, and the reason it exists is specific:
    /// `for_each_ordered` **silently** falls back to a sequential loop when the
    /// caller is a rayon worker, when the pool has one thread, or when the depth
    /// is 1. PCA owns two inner rayon pools, so a mis-nested wiring here would
    /// produce no speedup and no error — the whole task could land, pass every
    /// correctness test below, and do nothing at all.
    ///
    /// **Thread identity, not observed overlap.** An earlier version asserted a
    /// maximum-in-flight count of >= 2, which is a *timing* property: it held on
    /// a 24-core box 20 runs out of 20 and failed on a 2-core CI runner, where
    /// libtest's own parallelism saturates the pool and the spawned decodes run
    /// one at a time. "Did the pipeline engage" is structural — when it does,
    /// `read_shard` runs on a rayon worker; when it declines, on the calling
    /// thread — so that is what these tests assert. `max_live` is still
    /// recorded, but only to make a failure message informative.
    struct GaugedSource {
        shards: Vec<ScxCsr>,
        n_obs: usize,
        n_vars: usize,
        live: AtomicUsize,
        max_live: AtomicUsize,
        decode_threads: Mutex<HashSet<ThreadId>>,
    }

    impl GaugedSource {
        fn new(shards: Vec<ScxCsr>, n_obs: usize, n_vars: usize) -> Self {
            Self {
                shards,
                n_obs,
                n_vars,
                live: AtomicUsize::new(0),
                max_live: AtomicUsize::new(0),
                decode_threads: Mutex::new(HashSet::new()),
            }
        }

        /// True when at least one shard decoded somewhere other than `caller`.
        fn decoded_off_thread(&self, caller: ThreadId) -> bool {
            self.decode_threads
                .lock()
                .unwrap()
                .iter()
                .any(|t| *t != caller)
        }

        fn decode_thread_count(&self) -> usize {
            self.decode_threads.lock().unwrap().len()
        }

        fn max_concurrent_decodes(&self) -> usize {
            self.max_live.load(Ordering::SeqCst)
        }

        fn reset(&self) {
            self.max_live.store(0, Ordering::SeqCst);
            self.decode_threads.lock().unwrap().clear();
        }
    }

    /// Assert the pipeline engaged: some shard decoded off the calling thread.
    fn assert_prefetch_engaged(src: &GaugedSource, what: &str) {
        let me = std::thread::current().id();
        assert!(
            src.decoded_off_thread(me),
            "{what}: every shard decoded on the calling thread — the prefetch \
             pipeline declined to engage (threads seen: {}, max in flight: {})",
            src.decode_thread_count(),
            src.max_concurrent_decodes()
        );
    }

    impl ShardSource for GaugedSource {
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
            self.decode_threads
                .lock()
                .unwrap()
                .insert(std::thread::current().id());
            let now = self.live.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_live.fetch_max(now, Ordering::SeqCst);
            // Kept short: nothing asserts on overlap now, so this only widens the
            // window in which `max_live` can observe some.
            std::thread::sleep(std::time::Duration::from_millis(1));
            let out = self.shards[shard_idx].clone();
            self.live.fetch_sub(1, Ordering::SeqCst);
            Ok(out)
        }
    }

    /// A multi-shard fixture. `rows_per_shard × k` stays under
    /// `spmm_*`'s 10 000-element parallel threshold, so every reduction takes
    /// its **sequential** branch and the whole op is deterministic — which is
    /// what lets the equivalence tests below assert exact bits instead of a
    /// tolerance. `parallel_branch_fixture` is the deliberate counterpart.
    fn gauged_fixture(n_shards: usize, rows_per_shard: usize, n_vars: usize) -> GaugedSource {
        let shards: Vec<ScxCsr> = (0..n_shards)
            .map(|s| {
                let mut indptr = vec![0i64];
                let mut indices = Vec::new();
                let mut data = Vec::new();
                for r in 0..rows_per_shard {
                    let global = s * rows_per_shard + r;
                    for c in 0..n_vars {
                        if !(global + c).is_multiple_of(3) {
                            indices.push(c as i32);
                            data.push(((global % 7) + c + 1) as f32 * 0.5);
                        }
                    }
                    indptr.push(indices.len() as i64);
                }
                ScxCsr::new_unchecked((rows_per_shard, n_vars), indptr, indices, data)
            })
            .collect();
        GaugedSource::new(shards, n_shards * rows_per_shard, n_vars)
    }

    /// Skip rather than fail where the pipeline is *designed* not to engage: a
    /// single-thread pool takes the sequential fallback by construction.
    fn pool_can_prefetch() -> bool {
        rayon::current_num_threads() > 1
    }

    #[test]
    fn covariance_build_decodes_shards_concurrently() {
        if !pool_can_prefetch() {
            return;
        }
        let src = gauged_fixture(8, 12, 6);
        // The load-bearing case: the consume closure runs its own `par_iter` over
        // column blocks on the global pool while decode overlaps on the same one.
        covariance_pca_with_depth(&src, 2, true, 4).unwrap();
        assert_prefetch_engaged(&src, "covariance build");
    }

    #[test]
    fn streaming_spmm_passes_decode_shards_concurrently() {
        if !pool_can_prefetch() {
            return;
        }
        let src = gauged_fixture(8, 12, 6);
        let k = 4;
        let m = random_gaussian(6, k, 1);

        streaming_spmm_forward(&src, &m, k, None, 4).unwrap();
        assert_prefetch_engaged(&src, "forward SpMM");

        src.reset();
        let q = Mat::<f64>::from_fn(src.n_obs(), k, |i, j| (i + j) as f64 * 0.25);
        streaming_spmm_transpose(&src, &q, None, 4).unwrap();
        assert_prefetch_engaged(&src, "transpose SpMM");
    }

    #[test]
    fn whole_pca_ops_decode_shards_concurrently() {
        if !pool_can_prefetch() {
            return;
        }
        let src = gauged_fixture(8, 12, 6);
        randomized_pca_with_depth(&src, 2, 2, 2, true, 42, 4).unwrap();
        assert_prefetch_engaged(&src, "randomized_pca");

        src.reset();
        covariance_pca_with_depth(&src, 2, true, 4).unwrap();
        assert_prefetch_engaged(&src, "covariance_pca");
    }

    /// Depth 1 must take the sequential fallback — this is what makes
    /// `SCX_ACCEL_PREFETCH_DEPTH=1` a *genuine* baseline for the capture rather
    /// than merely an "off" arm. #373's GPU staging A/B got this wrong: there,
    /// depth 1 had zero decode threads where `main` had one.
    #[test]
    fn depth_one_decodes_on_the_calling_thread() {
        let src = gauged_fixture(6, 8, 5);
        randomized_pca_with_depth(&src, 2, 2, 2, true, 42, 1).unwrap();
        let me = std::thread::current().id();
        assert!(
            !src.decoded_off_thread(me),
            "depth 1 must take the sequential fallback and decode inline on the \
             calling thread — that is what makes SCX_ACCEL_PREFETCH_DEPTH=1 a \
             genuine baseline rather than merely an 'off' arm"
        );
        assert_eq!(src.max_concurrent_decodes(), 1);
    }

    fn assert_bits_eq(got: &[f64], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{what}[{i}]: {a} != {b}");
        }
    }

    #[test]
    fn forward_spmm_is_bit_identical_across_depths() {
        let src = gauged_fixture(6, 8, 5);
        let k = 4;
        let m = random_gaussian(5, k, 7);
        let means: Vec<f64> = (0..5).map(|v| 0.1 * v as f64).collect();
        for mu in [None, Some(means.as_slice())] {
            let seq = streaming_spmm_forward(&src, &m, k, mu, 1).unwrap();
            let pre = streaming_spmm_forward(&src, &m, k, mu, 4).unwrap();
            assert_bits_eq(&pre, &seq, "forward SpMM");
        }
    }

    /// Randomized PCA is *structurally* deterministic on this branch — the
    /// forward SpMM writes each output row once, the transpose takes its
    /// sequential path, and `col_means_and_sum_sq_prefetched` accumulates in
    /// shard order — so exact bits is the honest bar. (The covariance route is
    /// not, and gets its own test below.)
    #[test]
    fn randomized_pca_is_bit_identical_across_depths_on_the_sequential_branch() {
        let (n_shards, rows_per_shard, n_vars) = (6usize, 8usize, 5usize);
        let (n_components, n_oversamples) = (2usize, 2usize);
        let src = gauged_fixture(n_shards, rows_per_shard, n_vars);
        // Premise: the reductions really are on their *deterministic* branch,
        // so exact bits is the right bar here. If a future edit grew this
        // fixture past the threshold the assertion would start flaking on
        // f64 reassociation rather than on a real regression.
        let k = (n_components + n_oversamples).min(n_vars).min(src.n_obs());
        assert!(
            rows_per_shard * k <= 10_000,
            "fixture must stay under the parallel-reduction threshold"
        );

        let seq = randomized_pca_with_depth(&src, 2, 2, 2, true, 42, 1).unwrap();
        let pre = randomized_pca_with_depth(&src, 2, 2, 2, true, 42, 4).unwrap();
        assert_bits_eq(&pre.embeddings, &seq.embeddings, "randomized embeddings");
        assert_bits_eq(&pre.components, &seq.components, "randomized components");
        assert_bits_eq(
            &pre.variance_ratio,
            &seq.variance_ratio,
            "randomized variance_ratio",
        );
    }

    /// The covariance route gets a **relative** bar, not exact bits, and the
    /// difference is not a convenience.
    ///
    /// `accumulate_covariance_streaming` folds into `ThreadLocal` accumulators
    /// whose row→thread assignment is decided by work-stealing and whose merge
    /// order is `ThreadLocal::iter_mut()`. Measured on a cancelling-pair
    /// fixture through `pyscx.accel.pca`, five consecutive runs of
    /// `method="covariance"` produced five different results while
    /// `method="randomized"` produced one — and the diff against `main` shows
    /// both the accumulator declaration and the merge loop unchanged, so this
    /// predates decode-prefetch entirely.
    ///
    /// A well-conditioned fixture is the right one here, and this is the
    /// converse of the cancelling-fixture rule the column-sum kernels need. In
    /// a *reduction*, a lost ordering shows up only in rounding, so the fixture
    /// has to amplify it. In this pass a wiring bug is not subtle — a dropped
    /// or reordered `global_row` puts whole rows in the wrong place — so 1e-9
    /// relative separates "f64 reassociation" from "broken" with room to spare,
    /// and `col_means_and_sum_sq_prefetched` carries the cancelling-pair test
    /// for the reduction that genuinely needs one.
    #[test]
    fn covariance_pca_agrees_across_depths_within_reassociation_noise() {
        let src = gauged_fixture(6, 8, 5);
        let seq = covariance_pca_with_depth(&src, 2, true, 1).unwrap();
        let pre = covariance_pca_with_depth(&src, 2, true, 4).unwrap();
        let scale = seq
            .embeddings
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()))
            .max(1e-12);
        assert!(scale > 1e-6, "premise: embeddings must not be all-zero");
        for (i, (a, b)) in pre.embeddings.iter().zip(seq.embeddings.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-9 * scale,
                "covariance embedding[{i}] {a} vs {b} exceeds the reassociation bar"
            );
        }
    }

    /// The parallel reduction branches (`shard_rows × k > 10 000`) accumulate
    /// into `ThreadLocal` buffers whose row→thread assignment is decided by
    /// rayon work-stealing and whose merge order is `ThreadLocal::iter_mut()`.
    /// They are therefore **already** run-to-run non-bit-identical, before
    /// prefetch touches anything — so the bar here is a tolerance, and saying so
    /// is more honest than picking a fixture that hides it.
    #[test]
    fn pca_agrees_across_depths_on_the_parallel_branch() {
        let (rows_per_shard, n_vars) = (1200usize, 12usize);
        let (n_components, n_oversamples) = (2usize, 8usize);
        let src = gauged_fixture(4, rows_per_shard, n_vars);
        let k = (n_components + n_oversamples).min(n_vars).min(src.n_obs());
        assert!(
            rows_per_shard * k > 10_000,
            "premise: this fixture must reach the parallel reduction branch \
             (rows_per_shard {rows_per_shard} x k {k})"
        );

        let seq = randomized_pca_with_depth(&src, 2, 8, 2, true, 11, 1).unwrap();
        let pre = randomized_pca_with_depth(&src, 2, 8, 2, true, 11, 4).unwrap();
        let scale = seq
            .embeddings
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()))
            .max(1e-12);
        for (i, (a, b)) in pre.embeddings.iter().zip(seq.embeddings.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-9 * scale,
                "embedding[{i}] {a} vs {b} exceeds the f64-reassociation bar"
            );
        }
    }

    /// A fixture whose sums are **order-sensitive**: values span ~10 decades, so
    /// the pairwise products the covariance accumulates span ~20, and a running
    /// f64 sum loses different low bits depending on the order the rows arrive
    /// in. Without that span a bit-identity assertion is vacuous — a well-
    /// conditioned fixture is bit-identical under *every* order, including a
    /// broken one. `the_reassociating_fixture_can_detect_a_reordering` is the
    /// premise test that proves this one is not.
    ///
    /// Rows are dense so every column pair is exercised, and the LCG keeps the
    /// mantissas non-trivial (a period-4 pattern of exact powers of ten sums
    /// exactly and would defeat the point).
    fn reassociating_shard(row_base: usize, n_rows: usize, n_vars: usize) -> ScxCsr {
        let mut state = 0x2545_F491_4F6C_DD1Du64 ^ (row_base as u64).wrapping_mul(0x9E37_79B9);
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        for r in 0..n_rows {
            for c in 0..n_vars {
                let bits = next();
                // Mantissa in [1, 2), exponent cycling over 10^{-5..5}, random sign.
                let mant = 1.0 + (bits >> 40) as f32 / 16_777_216.0;
                let exp = 10f32.powi((((row_base + r + c) % 11) as i32) - 5);
                let sign = if bits & 1 == 0 { 1.0 } else { -1.0 };
                indices.push(c as i32);
                data.push(sign * mant * exp);
            }
            indptr.push(indices.len() as i64);
        }
        ScxCsr::new_unchecked((n_rows, n_vars), indptr, indices, data)
    }

    fn reassociating_fixture(
        n_shards: usize,
        rows_per_shard: usize,
        n_vars: usize,
    ) -> VecShardSource {
        VecShardSource {
            shards: (0..n_shards)
                .map(|s| reassociating_shard(s * rows_per_shard, rows_per_shard, n_vars))
                .collect(),
            n_obs: n_shards * rows_per_shard,
            n_vars,
        }
    }

    /// The premise every bit-identity assertion below rests on: this fixture's
    /// sums really do depend on accumulation order, so "the bits matched" is
    /// evidence of a fixed order rather than of an order that never mattered.
    #[test]
    fn the_reassociating_fixture_can_detect_a_reordering() {
        let csr = reassociating_shard(0, 64, 6);
        let n_vars = 6;

        let mut fwd_cov = vec![0.0f64; n_vars * n_vars];
        let mut fwd_sums = vec![0.0f64; n_vars];
        accumulate_covariance_serial(&csr, &mut fwd_cov, &mut fwd_sums, n_vars);

        // Same rows, visited last-to-first. Any correct implementation returns
        // the same *value*; only the low bits move.
        let mut rev_indptr = vec![0i64];
        let mut rev_indices = Vec::new();
        let mut rev_data = Vec::new();
        for r in (0..csr.n_rows()).rev() {
            let (s, e) = (csr.indptr[r] as usize, csr.indptr[r + 1] as usize);
            rev_indices.extend_from_slice(&csr.indices[s..e]);
            rev_data.extend_from_slice(&csr.data[s..e]);
            rev_indptr.push(rev_indices.len() as i64);
        }
        let rev = ScxCsr::new_unchecked((csr.n_rows(), n_vars), rev_indptr, rev_indices, rev_data);
        let mut rev_cov = vec![0.0f64; n_vars * n_vars];
        let mut rev_sums = vec![0.0f64; n_vars];
        accumulate_covariance_serial(&rev, &mut rev_cov, &mut rev_sums, n_vars);

        let moved = fwd_cov
            .iter()
            .zip(&rev_cov)
            .any(|(a, b)| a.to_bits() != b.to_bits());
        assert!(
            moved,
            "premise failed: reversing the row order left every covariance entry \
             bit-identical, so this fixture cannot tell a fixed accumulation order \
             from an arbitrary one and every bit-identity test using it is vacuous"
        );
    }

    /// The core claim, at the kernel rather than through a whole PCA: splitting
    /// the covariance output `n` ways must reproduce the serial accumulation
    /// **exactly**, for every `n`. Because the blocks are disjoint and each one
    /// walks rows in order, the split is invisible in the result — which is what
    /// makes the thread count a speed knob rather than a numeric one.
    ///
    /// `the_reassociating_fixture_can_detect_a_reordering` is the premise: this
    /// fixture's sums genuinely move under a different order, so matching bits
    /// here is evidence and not an accident of a well-conditioned input.
    ///
    /// This test and `covariance_pca_is_bit_identical_across_thread_counts` are
    /// not redundant, and neither subsumes the other. Making the block kernel
    /// walk rows backwards reddens *this* one (it disagrees with the serial
    /// oracle) and leaves the thread-count one green (backwards is still the
    /// same order at every width). Correctness and determinism are separate
    /// claims and need separate tests.
    #[test]
    fn covariance_accumulate_is_bit_identical_for_every_block_count() {
        let n_vars = 17usize;
        let csr = reassociating_shard(0, 200, n_vars);

        let mut want_cov = vec![0.0f64; n_vars * n_vars];
        let mut want_sums = vec![0.0f64; n_vars];
        accumulate_covariance_serial(&csr, &mut want_cov, &mut want_sums, n_vars);

        for n_blocks in [1usize, 2, 3, 5, 17, 32] {
            let (cov, sums) = covariance_with_blocks(&csr, n_vars, n_blocks);
            assert_bits_eq(&cov, &want_cov, &format!("covariance at {n_blocks} blocks"));
            assert_bits_eq(&sums, &want_sums, &format!("col_sums at {n_blocks} blocks"));
        }
    }

    /// A non-canonical CSR cannot take the partitioned path (the `c_j >= c_i`
    /// step is what makes a pair's block a function of `c_i` alone), so it falls
    /// back to the serial accumulation — which this pins against that same
    /// reference. Note the bar is the serial oracle, not the partitioned kernel:
    /// summing a row in stored rather than ascending order genuinely moves the
    /// low bits, so there is no bit-identity to assert across the two.
    #[test]
    fn an_unsorted_csr_matches_the_serial_oracle() {
        // Row 0's indices descend; row 1's are canonical.
        let csr = ScxCsr::new_unchecked(
            (2, 4),
            vec![0, 3, 6],
            vec![3, 1, 0, 0, 2, 3],
            vec![2.5, -1.5, 4.0, 0.5, -3.0, 7.0],
        );
        assert!(
            !colblocks::rows_strictly_increasing(&csr),
            "premise: the fixture must actually be non-canonical"
        );

        let mut want_cov = vec![0.0f64; 16];
        let mut want_sums = vec![0.0f64; 4];
        accumulate_covariance_serial(&csr, &mut want_cov, &mut want_sums, 4);

        let mut got_cov = vec![0.0f64; 16];
        let mut got_sums = vec![0.0f64; 4];
        accumulate_covariance_into(&csr, &mut got_cov, &mut got_sums, 4);

        assert_bits_eq(&got_cov, &want_cov, "unsorted covariance");
        assert_bits_eq(&got_sums, &want_sums, "unsorted col_sums");
    }

    /// A duplicate coordinate is a *value* question, not an ordering one, so it
    /// needs a value oracle: densify the row with duplicates summed — scipy's
    /// own semantics — and compare `XᵀX` against that.
    ///
    /// The predicate test elsewhere only asserts duplicates are *rejected* by
    /// `rows_strictly_increasing`, which says nothing about what the fallback
    /// they are routed to then computes. Before the `c1 == c2` doubling this
    /// returned 7 where the answer is 9.
    #[test]
    fn duplicate_coordinates_accumulate_to_their_coalesced_value() {
        let n_vars = 3usize;
        // Row 0: column 0 twice (1 + 2 = 3) and column 2 once.
        // Row 1: column 1 three times (0.5 + 0.25 + 1.25 = 2.0).
        let csr = ScxCsr::new_unchecked(
            (2, n_vars),
            vec![0, 3, 6],
            vec![0, 0, 2, 1, 1, 1],
            vec![1.0, 2.0, 4.0, 0.5, 0.25, 1.25],
        );
        assert!(
            !colblocks::rows_strictly_increasing(&csr),
            "premise: the fixture must carry duplicate coordinates"
        );

        // Oracle: coalesce to dense, then a plain Xᵀ X.
        let mut dense = vec![0.0f64; 2 * n_vars];
        for r in 0..2 {
            let (s, e) = (csr.indptr[r] as usize, csr.indptr[r + 1] as usize);
            for idx in s..e {
                dense[r * n_vars + csr.indices[idx] as usize] += csr.data[idx] as f64;
            }
        }
        let mut want = vec![0.0f64; n_vars * n_vars];
        let mut want_sums = vec![0.0f64; n_vars];
        for r in 0..2 {
            for c in 0..n_vars {
                want_sums[c] += dense[r * n_vars + c];
            }
            for lo in 0..n_vars {
                for hi in lo..n_vars {
                    want[lo * n_vars + hi] += dense[r * n_vars + lo] * dense[r * n_vars + hi];
                }
            }
        }

        let mut got = vec![0.0f64; n_vars * n_vars];
        let mut got_sums = vec![0.0f64; n_vars];
        accumulate_covariance_into(&csr, &mut got, &mut got_sums, n_vars);

        for lo in 0..n_vars {
            for hi in lo..n_vars {
                let (g, w) = (got[lo * n_vars + hi], want[lo * n_vars + hi]);
                assert!(
                    (g - w).abs() <= 1e-12 * w.abs().max(1.0),
                    "cov[{hi},{lo}]: got {g}, coalesced oracle says {w}"
                );
            }
        }
        for c in 0..n_vars {
            assert!((got_sums[c] - want_sums[c]).abs() <= 1e-12);
        }
    }

    /// Run `f` inside a private rayon pool of exactly `threads` threads.
    fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(f)
    }

    /// The covariance **reduction** is bit-identical at any thread count.
    ///
    /// The fixture has to be big enough that the build genuinely splits work.
    /// 300 rows/shard does **not**: the pre-fix
    /// `with_min_len((n_rows / workers).max(256))` cannot halve 300 into two
    /// ≥256-row chunks, so it ran as a single task on a single thread-local
    /// accumulator and passed this test for the wrong reason. 2048 rows split
    /// four ways at any thread count, and `n_vars = 6` gives the *new* kernel 6
    /// column blocks at width 12 against 1 at width 1.
    ///
    /// `n_vars` is deliberately small for a second reason: faer's self-adjoint
    /// eigendecomposition blocks by the ambient rayon width and its bits move
    /// with it above roughly 400 variables (measured). Keeping the fixture below
    /// that isolates *our* reduction, which is what this test is about. The
    /// whole-op claim across thread counts needs
    /// `SCX_ACCEL_DETERMINISTIC_LINALG=1` and lives in
    /// `tests/pca_linalg_parallelism.rs`; without it the guarantee is
    /// per-thread-count,
    /// which is `covariance_pca_is_bit_identical_run_to_run` below.
    #[test]
    fn covariance_reduction_is_bit_identical_across_thread_counts() {
        let (rows_per_shard, n_vars) = (2048usize, 6usize);
        assert!(
            rows_per_shard >= 512 && n_vars >= 2,
            "premise: the fixture must be splittable — rows for a row-partitioned \
             build, columns for a column-partitioned one"
        );
        let src = reassociating_fixture(4, rows_per_shard, n_vars);
        let one = in_pool(1, || covariance_pca_with_depth(&src, 3, true, 4).unwrap());
        for threads in [2usize, 7, 12] {
            let got = in_pool(threads, || {
                covariance_pca_with_depth(&src, 3, true, 4).unwrap()
            });
            assert_bits_eq(
                &got.embeddings,
                &one.embeddings,
                &format!("covariance embeddings at {threads} threads vs 1"),
            );
            assert_bits_eq(
                &got.components,
                &one.components,
                &format!("covariance components at {threads} threads vs 1"),
            );
        }
    }

    /// The transpose SpMM is bit-identical at any thread count, on a fixture that
    /// reaches its parallel branch. `Q` is fixed across the arms on purpose: the
    /// whole randomized route cannot make this claim without pinned linalg,
    /// because faer's QR reblocks with the pool width — so feeding a fixed `Q`
    /// is what isolates the reduction this change actually fixed.
    #[test]
    fn transpose_spmm_is_bit_identical_across_thread_counts() {
        let (rows_per_shard, n_vars, k) = (1200usize, 12usize, 10usize);
        let src = reassociating_fixture(4, rows_per_shard, n_vars);
        assert!(
            rows_per_shard * k > 10_000,
            "premise: this fixture must reach the parallel branch \
             (rows_per_shard {rows_per_shard} x k {k})"
        );
        let q = Mat::<f64>::from_fn(src.n_obs(), k, |i, j| {
            // Wide dynamic range, same reason as `reassociating_shard`.
            let e = 10f64.powi(((i + j) % 9) as i32 - 4);
            if (i + j) % 2 == 0 {
                e
            } else {
                -e * 1.5
            }
        });
        let means: Vec<f64> = (0..n_vars).map(|v| 0.25 * (v + 1) as f64).collect();

        for mu in [None, Some(means.as_slice())] {
            let one = in_pool(1, || streaming_spmm_transpose(&src, &q, mu, 4).unwrap());
            for threads in [2usize, 7, 12] {
                let got = in_pool(threads, || {
                    streaming_spmm_transpose(&src, &q, mu, 4).unwrap()
                });
                assert_bits_eq(
                    &got,
                    &one,
                    &format!(
                        "transpose SpMM at {threads} threads vs 1 (centered: {})",
                        mu.is_some()
                    ),
                );
            }
        }
    }

    /// Splitting the transpose output `n` ways must reproduce the single-block
    /// accumulation exactly, for every `n` — the transpose twin of
    /// `covariance_accumulate_is_bit_identical_for_every_block_count`.
    #[test]
    fn spmm_transpose_is_bit_identical_for_every_block_count() {
        let (n_vars, k) = (17usize, 6usize);
        let csr = reassociating_shard(0, 400, n_vars);
        let q = Mat::<f64>::from_fn(400, k, |i, j| {
            let e = 10f64.powi(((i + j) % 9) as i32 - 4);
            if (i + j) % 2 == 0 {
                e
            } else {
                -e * 1.5
            }
        });

        let mut want = vec![0.0f64; n_vars * k];
        let mut q_row = vec![0.0f64; k];
        spmm_transpose_block(&csr, &q, 0, &mut want, k, 0, n_vars, true, &mut q_row);

        for n_blocks in [1usize, 2, 3, 5, 17, 32] {
            let blocks = colblocks::plan_blocks(&vec![1u64; n_vars], n_blocks);
            let mut got = vec![0.0f64; n_vars * k];
            let parts = colblocks::split_by_blocks(&mut got, &blocks, k);
            for (block, part) in blocks.iter().zip(parts) {
                let mut scratch = vec![0.0f64; k];
                spmm_transpose_block(
                    &csr,
                    &q,
                    0,
                    part,
                    k,
                    block.start,
                    block.end,
                    true,
                    &mut scratch,
                );
            }
            assert_bits_eq(&got, &want, &format!("transpose at {n_blocks} blocks"));
        }
    }

    /// The premise for the transpose bit-identity tests: this fixture's sums are
    /// order-sensitive, so matching bits is evidence rather than an artefact of
    /// well-conditioned data.
    #[test]
    fn the_transpose_fixture_can_detect_a_reordering() {
        let (n_vars, k) = (17usize, 6usize);
        let csr = reassociating_shard(0, 400, n_vars);
        let q = Mat::<f64>::from_fn(400, k, |i, j| {
            let e = 10f64.powi(((i + j) % 9) as i32 - 4);
            if (i + j) % 2 == 0 {
                e
            } else {
                -e * 1.5
            }
        });

        let mut fwd = vec![0.0f64; n_vars * k];
        let mut scratch = vec![0.0f64; k];
        spmm_transpose_block(&csr, &q, 0, &mut fwd, k, 0, n_vars, true, &mut scratch);

        // The same products, gathered last row first.
        let mut rev = vec![0.0f64; n_vars * k];
        for r in (0..csr.n_rows()).rev() {
            let (s, e) = (csr.indptr[r] as usize, csr.indptr[r + 1] as usize);
            for idx in s..e {
                let c = csr.indices[idx] as usize;
                let v = csr.data[idx] as f64;
                for j in 0..k {
                    rev[c * k + j] += v * q[(r, j)];
                }
            }
        }

        assert!(
            fwd.iter()
                .zip(&rev)
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "premise failed: reversing the row order left every Z entry bit-identical, \
             so this fixture cannot tell a fixed accumulation order from an arbitrary \
             one and the bit-identity tests using it are vacuous"
        );
    }

    /// The defect this change exists for: repeated runs of the same call on the
    /// same machine used to disagree, because row→worker assignment was decided
    /// by work-stealing. This is the guarantee the default configuration makes —
    /// faer's dense decompositions are stable at a fixed width, so nothing else
    /// has to be pinned for it to hold.
    #[test]
    fn covariance_pca_is_bit_identical_run_to_run() {
        let src = reassociating_fixture(4, 2048, 6);
        let first = covariance_pca_with_depth(&src, 3, true, 4).unwrap();
        for run in 1..5 {
            let again = covariance_pca_with_depth(&src, 3, true, 4).unwrap();
            assert_bits_eq(
                &again.embeddings,
                &first.embeddings,
                &format!("covariance embeddings, run {run} vs run 0"),
            );
        }
    }

    #[test]
    fn randomized_pca_is_bit_identical_run_to_run() {
        let src = reassociating_fixture(4, 1200, 12);
        let first = randomized_pca_with_depth(&src, 2, 8, 2, true, 11, 4).unwrap();
        for run in 1..5 {
            let again = randomized_pca_with_depth(&src, 2, 8, 2, true, 11, 4).unwrap();
            assert_bits_eq(
                &again.embeddings,
                &first.embeddings,
                &format!("randomized embeddings, run {run} vs run 0"),
            );
        }
    }

    #[test]
    fn deterministic_linalg_opts_in_only_on_an_affirmative_value() {
        for on in ["1", "true", "TRUE", "yes", "On", " 1 "] {
            assert!(
                parse_deterministic_linalg(Some(on.into())),
                "{on:?} should opt in"
            );
        }
        for off in ["0", "false", "no", "", "2", "maybe"] {
            assert!(
                !parse_deterministic_linalg(Some(off.into())),
                "{off:?} should not opt in"
            );
        }
        assert!(!parse_deterministic_linalg(None));
    }

    #[test]
    fn pca_prefetch_depth_is_at_least_one() {
        // A `OnceLock` on both inputs, so this cannot A/B the env here — the
        // point is the floor: a zero depth would make `for_each_ordered`'s
        // `depth.max(1)` the only thing standing between us and a stall.
        assert!(pca_prefetch_depth() >= 1);
        assert!(pca_prefetch_depth() <= crate::prefetch::prefetch_depth());
    }

    /// The embeddings scatter exactly as `covariance_pca` used to inline it,
    /// kept as the oracle for the `spmm_forward_into` swap.
    fn inlined_embeddings_scatter_reference(
        csr: &ScxCsr,
        v_rm: &[f64],
        n_components: usize,
        embeddings: &mut [f64],
        global_row: usize,
        mean_correction: Option<&[f64]>,
    ) {
        for r in 0..csr.n_rows() {
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
            if let Some(mc) = mean_correction {
                for pc in 0..n_components {
                    embeddings[e_offset + pc] -= mc[pc];
                }
            }
        }
    }

    /// `covariance_pca`'s embeddings pass now calls the shared kernel instead of
    /// its own copy. The claim is bit-identity, not "close": each output row is
    /// written by exactly one thread and still walks that row's nonzeros in
    /// index order, so parallelising across rows reassociates nothing.
    ///
    /// Both `spmm_forward_into` branches are covered — the small fixture takes
    /// its serial path and the large one crosses the 10 000-element threshold
    /// into `par_chunks_mut`, which is the branch the claim is actually about.
    #[test]
    fn covariance_embeddings_kernel_matches_the_inlined_scatter_bitwise() {
        for (rows_per_shard, n_vars, n_components, expect_parallel) in
            [(8usize, 5usize, 4usize, false), (600, 12, 30, true)]
        {
            assert_eq!(
                rows_per_shard * n_components > 10_000,
                expect_parallel,
                "premise: fixture ({rows_per_shard} rows x {n_components} comps) must land \
                 on the {} branch",
                if expect_parallel {
                    "parallel"
                } else {
                    "serial"
                }
            );
            let src = gauged_fixture(3, rows_per_shard, n_vars);
            let n_obs = src.n_obs();
            let v_rm = random_gaussian(n_vars, n_components, 5);
            let mc: Vec<f64> = (0..n_components).map(|j| 0.03 * (j + 1) as f64).collect();

            for mean_correction in [None, Some(mc.as_slice())] {
                let mut got = vec![0.0f64; n_obs * n_components];
                let mut want = vec![0.0f64; n_obs * n_components];
                let mut global_row = 0usize;
                for shard_idx in 0..src.n_shards() {
                    let csr = src.read_shard(shard_idx).unwrap();
                    spmm_forward_into(
                        &csr,
                        &v_rm,
                        n_components,
                        &mut got,
                        global_row,
                        mean_correction,
                    );
                    inlined_embeddings_scatter_reference(
                        &csr,
                        &v_rm,
                        n_components,
                        &mut want,
                        global_row,
                        mean_correction,
                    );
                    global_row += csr.n_rows();
                }
                assert_bits_eq(&got, &want, "covariance embeddings scatter");
            }
        }
    }
}
