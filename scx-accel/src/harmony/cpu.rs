//! Harmony2 batch integration algorithm.
//!
//! Operates on dense PCA embeddings and corrects batch effects via iterative
//! soft k-means clustering + ridge regression. Reference: Korsunsky et al.
//!
//! This module is a clean-room implementation derived solely from the
//! published Harmony2 algorithm description (Korsunsky et al. 2019) — no
//! code is ported from harmonypy (GPL-3.0) or the R harmony package.

// The algorithm is index-heavy (cluster × cell × batch × PC loops), where
// iterator adapters hurt rather than help readability. Allow the pattern.
#![allow(clippy::needless_range_loop)]

use faer::linalg::solvers::DenseSolveCore;
use faer::Mat;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::error::{AccelError, Result};

// ─── Public Types ─────────────────────────────────────────────────────

/// Configuration for Harmony2 integration.
#[derive(Debug, Clone)]
pub struct HarmonyConfig {
    /// Number of clusters (default: min(N/30, 100), clamped to [2, N/2]).
    pub n_clusters: Option<usize>,
    /// Diversity penalty per covariate (default: 2.0 for each).
    pub theta: Option<Vec<f64>>,
    /// Initial soft assignment bandwidth (default: 0.1). Scalar input is
    /// expanded to length K internally.
    pub sigma: f64,
    /// Ridge penalty. None = dynamic estimation (recommended).
    pub lambda: Option<Vec<f64>>,
    /// Alpha for dynamic lambda estimation (default: 0.2).
    pub alpha: f64,
    /// Maximum Harmony iterations (default: 10).
    pub max_iter: usize,
    /// Maximum k-means sub-iterations per Harmony iteration (default: 4).
    pub max_iter_kmeans: usize,
    /// Harmony convergence tolerance (default: 1e-2).
    pub epsilon_harmony: f64,
    /// K-means convergence tolerance (default: 1e-3).
    pub epsilon_kmeans: f64,
    /// Convergence window size (default: 3).
    pub window_size: usize,
    /// Block size as fraction of N for stochastic updates (default: 0.05).
    pub block_size: f64,
    /// Minimum batch proportion per cluster to apply correction (default: 1e-5).
    pub batch_prop_cutoff: f64,
    /// Protection against overcorrection of small batches (default: 0.0).
    pub tau: f64,
    /// Random seed for reproducibility.
    pub random_state: u64,
    /// Number of threads (default: available parallelism).
    pub n_threads: Option<usize>,
}

impl Default for HarmonyConfig {
    fn default() -> Self {
        Self {
            n_clusters: None,
            theta: None,
            sigma: 0.1,
            lambda: None,
            alpha: 0.2,
            max_iter: 10,
            // INVARIANT: max_iter_kmeans >= 2 * window_size, otherwise
            // `check_convergence_kmeans` (which needs 2*window objectives) can
            // never fire and the k-means sub-loop always burns the full cap
            // (the pre-1.4 `4 < 2*3` convergence deadlock). At 6 the check
            // becomes reachable and k-means gets two extra refinement
            // sub-iterations (closer to the R reference). Raising this further
            // above 2*window_size would additionally enable early-exit before
            // the cap.
            max_iter_kmeans: 6,
            epsilon_harmony: 1e-2,
            epsilon_kmeans: 1e-3,
            window_size: 3,
            block_size: 0.05,
            batch_prop_cutoff: 1e-5,
            tau: 0.0,
            random_state: 0,
            n_threads: None,
        }
    }
}

/// Result of Harmony2 integration.
#[derive(Debug, Clone)]
pub struct HarmonyResult {
    /// Corrected embeddings, N x d, row-major f64.
    pub z_corrected: Vec<f64>,
    /// Number of cells.
    pub n_obs: usize,
    /// Number of PCs.
    pub n_pcs: usize,
    /// Number of clusters.
    pub n_clusters: usize,
    /// Per-iteration Harmony objective values.
    pub objective_harmony: Vec<f64>,
    /// Number of iterations until convergence (or max_iter).
    pub n_iterations: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Whether the GPU k-means sub-iter loop replayed a captured CUDA graph.
    ///
    /// `None` on the CPU path, which has no capture decision to make.
    /// `Some(true)` when capture succeeded and every later sub-iter replayed
    /// the graph. `Some(false)` means simply "no graph was replayed", which has
    /// **four** causes, not three: capture failed, capture produced no graph,
    /// `SCX_DISABLE_CUDA_GRAPHS=1` turned it off, or the run never reached a
    /// capture attempt at all — the first k-means sub-iteration is always a
    /// direct warm-up, so a configuration with only one sub-iteration in total
    /// (e.g. `max_iter = 1, max_iter_kmeans = 1`, or `max_iter = 0`) has no
    /// second sub-iter on which to capture. Read `Some(false)` as "the kernels
    /// ran directly", not as "capture was tried and failed"; the WARN log
    /// distinguishes the failure cases. The numbers are the same either way;
    /// the throughput is not, which is why a failure is recorded rather than
    /// swallowed.
    pub graph_replay: Option<bool>,
}

/// Batch covariate specification.
#[derive(Debug, Clone)]
pub struct BatchCovariate {
    /// Cell-to-level mapping (length N).
    pub labels: Vec<u32>,
    /// Number of levels in this covariate.
    pub n_levels: usize,
    /// Optional name (e.g., "donor", "technology").
    pub name: Option<String>,
}

// ─── Internal State ───────────────────────────────────────────────────

/// Global batch level index helper.
///
/// Global levels are numbered contiguously across covariates: covariate 0
/// levels come first, then covariate 1, etc. `cov_offset[c]` is the global
/// index of covariate c's first level.
struct BatchLayout {
    /// Per-covariate level offsets into the global batch axis.
    cov_offset: Vec<usize>,
    /// Total number of global batch levels (sum over covariates).
    b: usize,
    /// Number of covariates.
    c: usize,
}

impl BatchLayout {
    fn new(covariates: &[BatchCovariate]) -> Self {
        let mut cov_offset = Vec::with_capacity(covariates.len());
        let mut running = 0usize;
        for cov in covariates {
            cov_offset.push(running);
            running += cov.n_levels;
        }
        Self {
            cov_offset,
            b: running,
            c: covariates.len(),
        }
    }
}

struct HarmonyState {
    // Embeddings (d x N, column-major f64). `z_cos` (the L2-normalized
    // view of `z_corr` used for cosine-distance clustering) is NOT stored:
    // normalization is done per column on demand inside `compute_distances`
    // / `kmeans_plus_plus` / `kmeans_refine`. Saves d·N f64s (≈ 400 MB at
    // N=1M, d=50) at the cost of one extra O(d) norm per column access in
    // the k-means and distance paths.
    z_orig: Vec<f64>,
    z_corr: Vec<f64>,
    // Clustering. `y` stays f64 (d·K is small, accuracy-critical for the
    // ridge solve / centroid normalization). `r` and `dist_mat` are stored
    // f32 (K x N is the dominant CPU memory peak; values are bounded —
    // `r ∈ [0, 1]`, `dist ∈ [0, 4]` — so f32's ~7-digit mantissa is safe);
    // every reduction promotes to f64 on read.
    y: Vec<f64>,        // d x K, column-major (centroids as columns)
    r: Vec<f32>,        // K x N, row-major (cluster x cell)
    dist_mat: Vec<f32>, // K x N, row-major
    o: Vec<f64>,        // K x B, row-major
    e: Vec<f64>,        // K x B, row-major
    // Batch structure
    covariates: Vec<BatchCovariate>,
    layout: BatchLayout,
    /// For each covariate c, for each level l, the list of cell indices.
    batch_index: Vec<Vec<Vec<usize>>>,
    /// For each covariate c, for each cell i, the global batch index
    /// `cov_offset[c] + labels[c][i]`. Length C × N. Precomputed once so
    /// the parallel `compute_o_e` cluster pass can scan `r[ku, ·]`
    /// linearly without recomputing the offset per cell.
    cell_to_gb: Vec<Vec<usize>>,
    /// Global batch sizes, length B.
    n_b: Vec<f64>,
    /// Global batch proportions, length B.
    pr_b: Vec<f64>,
    // Parameters (expanded)
    theta: Vec<f64>, // length B (tau-scaled)
    sigma: Vec<f64>, // length K
    /// User-supplied lambda (length B+1, with intercept at index 0 = 0)
    /// or None for dynamic estimation.
    lambda_fixed: Option<Vec<f64>>,
    // Dimensions
    n: usize,
    d: usize,
    k: usize,
    // Convergence tracking
    objective_kmeans: Vec<f64>,
    objective_harmony: Vec<f64>,
    // Config
    config: HarmonyConfig,
    rng: ChaCha8Rng,
}

// ─── Public entry point ──────────────────────────────────────────────

/// Run Harmony2 batch integration on PCA embeddings.
///
/// # Arguments
/// * `embeddings` — PCA embeddings, N x d, row-major f32
/// * `n_obs` — number of cells (N)
/// * `n_pcs` — number of PCs (d)
/// * `covariates` — one or more batch covariates
/// * `config` — algorithm configuration
pub fn harmony_integrate(
    embeddings: &[f32],
    n_obs: usize,
    n_pcs: usize,
    covariates: &[BatchCovariate],
    config: &HarmonyConfig,
) -> Result<HarmonyResult> {
    // An explicit `config.n_threads` wins; otherwise the shared
    // `SCX_ACCEL_NUM_THREADS` policy sizes the private integration pool.
    let resolved_threads = config
        .n_threads
        .or_else(crate::mem_budget::accel_num_threads);
    if let Some(n) = resolved_threads {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| AccelError::InvalidInput(format!("rayon thread pool: {e}")))?;
        pool.install(|| run_inner(embeddings, n_obs, n_pcs, covariates, config))
    } else {
        run_inner(embeddings, n_obs, n_pcs, covariates, config)
    }
}

fn run_inner(
    embeddings: &[f32],
    n_obs: usize,
    n_pcs: usize,
    covariates: &[BatchCovariate],
    config: &HarmonyConfig,
) -> Result<HarmonyResult> {
    let mut state = HarmonyState::new(embeddings, n_obs, n_pcs, covariates, config)?;
    state.run()
}

// ─── HarmonyState::new (Initialization) ──────────────────────────────

impl HarmonyState {
    fn new(
        embeddings: &[f32],
        n_obs: usize,
        n_pcs: usize,
        covariates: &[BatchCovariate],
        config: &HarmonyConfig,
    ) -> Result<Self> {
        // --- Validation ---
        if n_obs == 0 || n_pcs == 0 {
            return Err(AccelError::InvalidInput(
                "n_obs and n_pcs must be > 0".to_string(),
            ));
        }
        if embeddings.len() != n_obs * n_pcs {
            return Err(AccelError::InvalidInput(format!(
                "embeddings length {} != n_obs * n_pcs = {}",
                embeddings.len(),
                n_obs * n_pcs
            )));
        }
        if covariates.is_empty() {
            return Err(AccelError::InvalidInput(
                "at least one batch covariate is required".to_string(),
            ));
        }
        for (ci, cov) in covariates.iter().enumerate() {
            if cov.labels.len() != n_obs {
                return Err(AccelError::InvalidInput(format!(
                    "covariate {} labels length {} != n_obs {}",
                    ci,
                    cov.labels.len(),
                    n_obs
                )));
            }
            if cov.n_levels == 0 {
                return Err(AccelError::InvalidInput(format!(
                    "covariate {} has zero levels",
                    ci
                )));
            }
            if let Some(&m) = cov.labels.iter().max() {
                if (m as usize) >= cov.n_levels {
                    return Err(AccelError::InvalidInput(format!(
                        "covariate {} has label {} >= n_levels {}",
                        ci, m, cov.n_levels
                    )));
                }
            }
        }
        if !embeddings.iter().all(|v| v.is_finite()) {
            return Err(AccelError::InvalidInput(
                "embeddings contain NaN or Inf".to_string(),
            ));
        }
        // §7.18. `sigma` is the softmax bandwidth: `update_r` computes
        // `exp(-dist / sigma)`. At `sigma = 0` that is `-inf` / `NaN`, the
        // `sum_sd > 0.0` test fails, and the "degenerate: assign uniform"
        // fallback fires for *every* cell — Harmony returns an essentially
        // uncorrected embedding and reports `converged = true`. A negative
        // `sigma` inverts the softmax silently, assigning each cell to the
        // cluster it is furthest from.
        //
        // Checked here rather than at the pyscx boundary on purpose: pyscx,
        // rscx and any future binding all reach `HarmonyState::new`, and an
        // invariant enforced in one binding is not enforced (the §9.5 lesson).
        if !config.sigma.is_finite() || config.sigma <= 0.0 {
            return Err(AccelError::InvalidInput(format!(
                "sigma must be finite and > 0 (got {}); it is the softmax \
                 bandwidth in exp(-dist / sigma)",
                config.sigma
            )));
        }

        // --- Dimensions ---
        let n = n_obs;
        let d = n_pcs;
        let k = resolve_n_clusters(config.n_clusters, n);
        if k < 2 {
            return Err(AccelError::InvalidInput(
                "n_clusters after clamping is < 2".to_string(),
            ));
        }

        // --- Transpose row-major f32 → column-major f64 (d x N) ---
        // Core math runs in f64 for stable covariance / LU / ridge solves;
        // the corrected embedding is cast back to f32 at the Python boundary.
        let mut z_orig = vec![0f64; d * n];
        for i in 0..n {
            for j in 0..d {
                z_orig[j + i * d] = embeddings[i * d + j] as f64;
            }
        }

        // --- Batch structure ---
        let layout = BatchLayout::new(covariates);
        let b = layout.b;

        let mut batch_index: Vec<Vec<Vec<usize>>> = covariates
            .iter()
            .map(|cov| vec![Vec::new(); cov.n_levels])
            .collect();
        for (ci, cov) in covariates.iter().enumerate() {
            for (i, &lab) in cov.labels.iter().enumerate() {
                batch_index[ci][lab as usize].push(i);
            }
        }

        // Precompute per-cell global-batch lookup (C × N) so the parallel
        // `compute_o_e` cluster pass can do a single linear scan of the
        // R row without recomputing `cov_offset[ci] + labels[ci][i]` per cell.
        let mut cell_to_gb: Vec<Vec<usize>> = Vec::with_capacity(layout.c);
        for (ci, cov) in covariates.iter().enumerate() {
            let off = layout.cov_offset[ci];
            let v: Vec<usize> = cov.labels.iter().map(|&l| off + l as usize).collect();
            cell_to_gb.push(v);
        }

        let mut n_b = vec![0f64; b];
        let mut pr_b = vec![0f64; b];
        let n_f = n as f64;
        for ci in 0..layout.c {
            for lvl in 0..covariates[ci].n_levels {
                let gb = layout.cov_offset[ci] + lvl;
                n_b[gb] = batch_index[ci][lvl].len() as f64;
                pr_b[gb] = n_b[gb] / n_f;
            }
        }

        // --- Theta expansion (per-covariate → per-level) + tau scaling ---
        let theta_cov = match &config.theta {
            Some(v) => {
                if v.len() != layout.c {
                    return Err(AccelError::InvalidInput(format!(
                        "theta length {} != n_covariates {}",
                        v.len(),
                        layout.c
                    )));
                }
                v.clone()
            }
            None => vec![2.0f64; layout.c],
        };
        let mut theta = vec![0f64; b];
        for ci in 0..layout.c {
            for lvl in 0..covariates[ci].n_levels {
                let gb = layout.cov_offset[ci] + lvl;
                theta[gb] = theta_cov[ci];
            }
        }
        if config.tau > 0.0 {
            let kf = k as f64;
            let t = config.tau;
            for gb in 0..b {
                let x = n_b[gb] / (kf * t);
                theta[gb] *= 1.0 - (-(x * x)).exp();
            }
        }

        // --- Sigma expansion ---
        let sigma = vec![config.sigma; k];

        // --- Lambda (fixed) ---
        let lambda_fixed = match &config.lambda {
            Some(v) => {
                // Accept either a single scalar (replicated across all levels)
                // or length B (one per level), or length == C (per-covariate,
                // replicated across its levels). Intercept is always 0.
                let mut lam = vec![0f64; b + 1];
                if v.len() == 1 {
                    for gb in 0..b {
                        lam[gb + 1] = v[0];
                    }
                } else if v.len() == layout.c {
                    for ci in 0..layout.c {
                        for lvl in 0..covariates[ci].n_levels {
                            let gb = layout.cov_offset[ci] + lvl;
                            lam[gb + 1] = v[ci];
                        }
                    }
                } else if v.len() == b {
                    lam[1..=b].copy_from_slice(&v[..b]);
                } else {
                    return Err(AccelError::InvalidInput(format!(
                        "lambda length {} must be 1, n_covariates ({}), or total levels ({})",
                        v.len(),
                        layout.c,
                        b
                    )));
                }
                Some(lam)
            }
            None => None,
        };

        // --- RNG ---
        let mut rng = ChaCha8Rng::seed_from_u64(config.random_state);

        // --- K-means++ seeding + Lloyd refinement ---
        let mut y = kmeans_plus_plus(&z_orig, d, n, k, &mut rng);
        kmeans_refine(&z_orig, &mut y, d, n, k, 10);
        l2_normalize_columns(&mut y, d, k);

        // --- Initial distances, R, O, E ---
        let dist_mat = compute_distances(&y, &z_orig, d, k, n);
        let r = softmax_r_from_dist(&dist_mat, &sigma, k, n);
        let (o, e) = compute_o_e(&r, &cell_to_gb, &layout, &pr_b, k, n);

        let z_corr = z_orig.clone();

        Ok(Self {
            z_orig,
            z_corr,
            y,
            r,
            dist_mat,
            o,
            e,
            covariates: covariates.to_vec(),
            layout,
            batch_index,
            cell_to_gb,
            n_b,
            pr_b,
            theta,
            sigma,
            lambda_fixed,
            n,
            d,
            k,
            objective_kmeans: Vec::new(),
            objective_harmony: Vec::new(),
            config: config.clone(),
            rng,
        })
    }
}

fn resolve_n_clusters(user: Option<usize>, n: usize) -> usize {
    let k = user.unwrap_or_else(|| (n / 30).clamp(2, 100));
    let upper = (n / 2).max(2);
    k.clamp(2, upper)
}

// ─── Primitive kernels ────────────────────────────────────────────────

/// L2-normalize each column of a (rows x cols) column-major matrix in place.
/// Zero-norm columns are left as zero.
fn l2_normalize_columns(m: &mut [f64], rows: usize, _cols: usize) {
    m.par_chunks_mut(rows).for_each(|col| {
        let n2: f64 = col.iter().map(|v| v * v).sum();
        if n2 > 0.0 {
            let inv = 1.0 / n2.sqrt();
            for v in col.iter_mut() {
                *v *= inv;
            }
        }
    });
}

/// Per-column L2 inverse-norms for an un-normalized d·N embedding
/// (column-major). Zero-norm columns get `inv = 0` (so scaled dots are
/// 0 and the 2·(1 - 0) distance is the "max" value, matching how the
/// earlier explicit `z_cos` path treated zero columns).
///
/// Length-N buffer (≈ 8 MB at N=1M) reused in `compute_distances` /
/// `kmeans_*` instead of recomputing the norm once per (cluster, cell).
fn compute_inv_norms(z: &[f64], d: usize, n: usize) -> Vec<f64> {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let col = &z[i * d..(i + 1) * d];
            let n2: f64 = col.iter().map(|v| v * v).sum();
            if n2 > 0.0 {
                1.0 / n2.sqrt()
            } else {
                0.0
            }
        })
        .collect()
}

/// dist[k, i] = 2 * (1 - Y[:, k] · normalize(z[:, i])).
///
/// `z` is the un-normalized d·N embedding (column-major). Per-cell
/// inverse L2 norms are precomputed once (length N) before the per-cluster
/// loop so each norm is computed once rather than K times. Output is
/// row-major (K x N) f32 — the dot product accumulates in f64 and is cast
/// only on store; values are bounded in `[0, 4]` so f32 storage carries no
/// meaningful precision loss. Rows are disjoint slices of length N, so we
/// parallelize across clusters.
///
/// PERF NOTE — do not "optimize" this into a faer/BLAS GEMM (`Yᵀ·Z`). This was
/// tried (perf-review item OPT-3.2, 2026-06) and measured as a *regression* at
/// every realistic Harmony scale: 0.29× at d=20, 0.45× at d=30, 0.80× at d=50
/// (K=100, N=300k), and 0.56× at N=1M/d=30 — i.e. 1.25–3.4× *slower*. A GEMM's
/// contraction dimension here is the PC count `d` (≈20–50), which is far too
/// short for the blocked microkernel to amortize its overhead, while this
/// scalar inner loop already auto-vectorizes and the per-cluster `par_chunks_mut`
/// gives K-way (100–200) core parallelism. The op is O(K·N·d) FLOPs either way —
/// there is no algorithmic headroom, only constant factors, and the scalar form
/// already wins them. (A GEMM only reached break-even at d=50 *and* K=200.)
fn compute_distances(y: &[f64], z: &[f64], d: usize, k: usize, n: usize) -> Vec<f32> {
    let inv_norms = compute_inv_norms(z, d, n);
    let mut out = vec![0f32; k * n];
    out.par_chunks_mut(n).enumerate().for_each(|(ku, row)| {
        let y_col = &y[ku * d..(ku + 1) * d];
        for i in 0..n {
            let z_col = &z[i * d..(i + 1) * d];
            let mut dot = 0f64;
            for j in 0..d {
                dot += y_col[j] * z_col[j];
            }
            row[i] = (2.0 * (1.0 - dot * inv_norms[i])) as f32;
        }
    });
    out
}

/// Tile size (cells per chunk) for the parallel softmax pass. Chosen so
/// the per-thread `tile_size × K` f64 scratch (≈ tile_size · K · 8 bytes)
/// stays well under typical per-core L2/L3 budgets — at K=200 this is
/// ≈ 6 MB per thread; at K=100 ≈ 3 MB.
const SOFTMAX_TILE: usize = 4096;

/// R = softmax(-dist / sigma[k]) per column, numerically stabilized.
///
/// `dist` is row-major K x N f32; sigma has length K; returns row-major
/// K x N f32. Per-cell softmax computation accumulates in f64 (max-stabilize,
/// exp, L1-normalize) and casts to f32 only on store.
///
/// Memory: replaces the prior two-pass `K×N` f64 col-major scratch with a
/// `SOFTMAX_TILE × K` f64 per-thread tile. At N=1M, K=100 that drops the
/// softmax peak from 800 MB f64 to ~3 MB per thread. Tiles are processed
/// in-order in fixed-size chunks (`SOFTMAX_TILE`); within each tile the
/// per-cell scan is sequential and the per-cluster scatter writes a
/// contiguous slice of the row-major output, so the result is bit-exact
/// regardless of how rayon schedules tiles across threads (the test
/// `test_determinism_same_seed` is the regression gate).
fn softmax_r_from_dist(dist: &[f32], sigma: &[f64], k: usize, n: usize) -> Vec<f32> {
    let mut r = vec![0f32; k * n];

    let n_tiles = n.div_ceil(SOFTMAX_TILE);

    // Each tile owns disjoint cell-index ranges per row of `r`. We pass
    // the output base pointer through a `Send + Sync` newtype so each
    // tile can scatter its `tile_size × K` softmax block into the
    // row-major output without going through the borrow checker (which
    // can't see the tile-disjointness invariant).
    //
    // Safety: each tile writes only to `r[ku*n + i_start..ku*n + i_end]`
    // for ku ∈ 0..k, and tiles have non-overlapping `[i_start, i_end)`
    // ranges, so no two threads ever target the same byte. The base
    // pointer remains valid for the entire `for_each` because `r` is
    // borrowed mutably here and not dropped.
    #[derive(Clone, Copy)]
    struct OutPtr(*mut f32);
    unsafe impl Send for OutPtr {}
    unsafe impl Sync for OutPtr {}
    let out_ptr = OutPtr(r.as_mut_ptr());

    (0..n_tiles).into_par_iter().for_each(|tile_idx| {
        let base = out_ptr; // copy the Send+Sync wrapper into the closure
        let i_start = tile_idx * SOFTMAX_TILE;
        let i_end = (i_start + SOFTMAX_TILE).min(n);
        let tile_len = i_end - i_start;

        // Per-tile cell-major scratch: [ cell_0_K, cell_1_K, ..., cell_{tile_len-1}_K ].
        let mut scratch = vec![0f64; tile_len * k];

        for (li, i) in (i_start..i_end).enumerate() {
            let cell = &mut scratch[li * k..(li + 1) * k];

            let mut max_neg = f64::NEG_INFINITY;
            for ku in 0..k {
                let v = -(dist[ku * n + i] as f64) / sigma[ku];
                cell[ku] = v;
                if v > max_neg {
                    max_neg = v;
                }
            }
            let mut sum = 0f64;
            for ku in 0..k {
                let e = (cell[ku] - max_neg).exp();
                cell[ku] = e;
                sum += e;
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for ku in 0..k {
                    cell[ku] *= inv;
                }
            } else {
                // Degenerate: assign uniform.
                let u = 1.0 / k as f64;
                for ku in 0..k {
                    cell[ku] = u;
                }
            }
        }

        // Scatter cluster-by-cluster into the row-major output. Each
        // cluster's destination slice `r[ku*n + i_start..ku*n + i_end]`
        // is contiguous and disjoint across (ku, tile_idx) pairs.
        for ku in 0..k {
            let dst_off = ku * n + i_start;
            // Safety: see top-of-function doc. `dst_off + tile_len ≤ k*n`
            // because ku < k and i_end ≤ n.
            let dst = unsafe { std::slice::from_raw_parts_mut(base.0.add(dst_off), tile_len) };
            for li in 0..tile_len {
                dst[li] = scratch[li * k + ku] as f32;
            }
        }
    });

    r
}

/// Compute O (K x B) and E (K x B) from R (K x N), batch_index, pr_b.
///
/// O[k, b] = sum_{i in batch b} R[k, i]
/// E[k, b] = pr_b[b] * sum_i R[k, i]
///
/// `r` is K x N row-major f32; reductions promote each entry to f64. The
/// outer loop is parallelised across clusters: each cluster owns disjoint
/// rows of `o` and `e`, so no shared accumulators are needed. Within a
/// cluster, `row_sum` and the per-batch O accumulators are computed in a
/// single linear scan of `r[ku, ·]` using the precomputed `cell_to_gb`
/// lookup. K up to ~200 ≥ thread count on Chimera, so cluster-level
/// parallelism saturates available cores.
///
/// Sequential summation within a cluster keeps results bit-exact w.r.t.
/// cell order, preserving the `test_determinism_same_seed` contract.
fn compute_o_e(
    r: &[f32],
    cell_to_gb: &[Vec<usize>],
    layout: &BatchLayout,
    pr_b: &[f64],
    k: usize,
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let b = layout.b;
    let mut o = vec![0f64; k * b];
    let mut e = vec![0f64; k * b];

    // Process clusters in parallel. Each thread owns one row of `o` and `e`.
    o.par_chunks_mut(b)
        .zip(e.par_chunks_mut(b))
        .enumerate()
        .for_each(|(ku, (o_row, e_row))| {
            let row_off = ku * n;
            let mut row_sum = 0f64;
            // Single pass: accumulate row_sum and per-batch O. Fused so
            // each entry of `r[ku, ·]` is read exactly once.
            for i in 0..n {
                let r_ki = r[row_off + i] as f64;
                row_sum += r_ki;
                for cov in cell_to_gb {
                    o_row[cov[i]] += r_ki;
                }
            }
            for gb in 0..b {
                e_row[gb] = pr_b[gb] * row_sum;
            }
        });

    (o, e)
}

// ─── K-means++ seeding (Harmony variant) ─────────────────────────────

/// D²-weighted k-means++ seeding (Harmony variant). Each candidate is scored by
/// its **minimum distance to ALL previously chosen centroids** — standard
/// k-means++ — not the distance to the last centroid only (the pre-1.4 bug,
/// which biased seeds and diverged from k-means++). Candidates are then sampled
/// proportionally to that distance via the exponential-race (Gumbel-min) trick.
///
/// The distance is `2·(1 − cosθ)`, which for unit vectors equals the squared
/// chord distance `‖a−b‖²`, so weighting by it is genuinely D²-proportional.
///
/// Returns a (d x K) column-major matrix of centroids (already L2-normalized
/// — equal to the normalized embedding column at each chosen cell index).
///
/// `z` is the un-normalized embedding. Per-cell inverse L2 norms are
/// precomputed once (H9 + follow-up) and reused across the K-1
/// centroid-selection passes instead of recomputing the norm ~K·N times.
fn kmeans_plus_plus(z: &[f64], d: usize, n: usize, k: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
    let mut y = vec![0f64; d * k];
    let mut chosen: Vec<usize> = Vec::with_capacity(k);
    let inv_norms = compute_inv_norms(z, d, n);

    // First centroid: uniform random cell (normalized).
    let i0 = rng.gen_range(0..n);
    chosen.push(i0);
    let inv0 = inv_norms[i0];
    let z0 = &z[i0 * d..(i0 + 1) * d];
    for t in 0..d {
        y[t] = z0[t] * inv0;
    }

    for ci in 1..k {
        // Try up to a few resamples on duplicate collisions.
        let mut picked = usize::MAX;
        'outer: for _attempt in 0..10 {
            let mut best_j = 0usize;
            let mut best_val = f64::INFINITY;
            for j in 0..n {
                let inv = inv_norms[j];
                if inv == 0.0 {
                    continue;
                }
                let z_col = &z[j * d..(j + 1) * d];
                // D² k-means++: minimum squared chord distance to ALL chosen
                // centroids (y[0..ci]), not just the last one. `2(1 - cosθ)` is
                // already the squared distance for unit vectors.
                let mut min_dist = f64::INFINITY;
                for cc in 0..ci {
                    let cent = &y[cc * d..(cc + 1) * d];
                    let mut dot = 0f64;
                    for t in 0..d {
                        dot += cent[t] * z_col[t];
                    }
                    let dist_c = (2.0 * (1.0 - dot * inv)).abs();
                    if dist_c < min_dist {
                        min_dist = dist_c;
                    }
                }
                if min_dist == 0.0 {
                    continue;
                }
                let u: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
                let v = -u.ln() / min_dist;
                if v < best_val {
                    best_val = v;
                    best_j = j;
                }
            }
            if !chosen.contains(&best_j) {
                picked = best_j;
                break 'outer;
            }
        }
        if picked == usize::MAX {
            // Fallback: pick any unused cell.
            for j in 0..n {
                if !chosen.contains(&j) {
                    picked = j;
                    break;
                }
            }
            if picked == usize::MAX {
                picked = ci % n;
            }
        }
        chosen.push(picked);
        let inv_p = inv_norms[picked];
        let z_p = &z[picked * d..(picked + 1) * d];
        let y_off = ci * d;
        for t in 0..d {
            y[y_off + t] = z_p[t] * inv_p;
        }
    }

    y
}

/// Lloyd's algorithm refinement on cosine distance with L2-normalized
/// centroids (keep_existing init). Runs `n_iter` iterations in place on `y`.
///
/// `z` is un-normalized; per-column inverse L2 norms are precomputed once
/// (H9 + follow-up) and scaled into the dot products so we avoid
/// normalizing twice per cell per iteration.
fn kmeans_refine(z: &[f64], y: &mut [f64], d: usize, n: usize, k: usize, n_iter: usize) {
    let inv_norms = compute_inv_norms(z, d, n);
    for _ in 0..n_iter {
        let mut assign = vec![0usize; n];
        for i in 0..n {
            let inv = inv_norms[i];
            if inv == 0.0 {
                assign[i] = 0;
                continue;
            }
            let z_col = &z[i * d..(i + 1) * d];
            let mut best_k = 0usize;
            let mut best_dot = f64::NEG_INFINITY;
            for ku in 0..k {
                let y_col = &y[ku * d..(ku + 1) * d];
                let mut dot = 0f64;
                for t in 0..d {
                    dot += y_col[t] * z_col[t];
                }
                // dot with the normalized column = dot * inv; the argmax is
                // unchanged by the positive-scale factor, so we only need
                // it for the centroid-update pass below.
                if dot > best_dot {
                    best_dot = dot;
                    best_k = ku;
                }
            }
            assign[i] = best_k;
        }

        // Update: new centroid = mean of assigned (normalized) cells, re-normalized.
        let mut sums = vec![0f64; d * k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let inv = inv_norms[i];
            if inv == 0.0 {
                continue;
            }
            let z_col = &z[i * d..(i + 1) * d];
            let ku = assign[i];
            let off = ku * d;
            for t in 0..d {
                sums[off + t] += z_col[t] * inv;
            }
            counts[ku] += 1;
        }
        for ku in 0..k {
            if counts[ku] > 0 {
                let off = ku * d;
                y[off..off + d].copy_from_slice(&sums[off..off + d]);
                let mut n2 = 0f64;
                for t in 0..d {
                    n2 += y[off + t] * y[off + t];
                }
                if n2 > 0.0 {
                    let inv = 1.0 / n2.sqrt();
                    for t in 0..d {
                        y[off + t] *= inv;
                    }
                }
            }
        }
    }
}

// ─── M-step: centroids from the current soft assignments ─────────────

impl HarmonyState {
    /// `Y[:, k] = normalize( Σ_i Z_cos[:, i] · R[k, i] )` — each centroid is
    /// the R-weighted mean of the L2-normalized embeddings, renormalized onto
    /// the unit sphere the cosine distance lives on.
    ///
    /// This is the M-step of the soft k-means sub-loop. Without it `y` and
    /// `dist_mat` are frozen for the whole sub-loop and the E-step can only
    /// trade `Σ R·dist` against the entropy penalty at fixed centroids —
    /// never move a centroid off a batch-driven mode. Pinned against
    /// harmonypy 0.2.0 `Harmony.cluster()`, which computes
    /// `Y = Z_cos @ R.T` followed by a per-column L2 normalization at the top
    /// of every sub-iteration; see `harmony_reference_tests.rs`.
    ///
    /// `z` is the un-normalized d·N embedding (column-major), cosine-normalized
    /// per cell through `compute_inv_norms` exactly as `compute_distances`
    /// does, so no `z_cos` buffer is materialized (H9). Clusters own disjoint
    /// d-length slices of `y`, so we parallelize across clusters and keep the
    /// inner cell loop sequential — the summation order is then independent of
    /// how rayon schedules, which `test_determinism_same_seed` gates.
    fn update_y(&mut self) {
        let d = self.d;
        let n = self.n;
        let inv_norms = compute_inv_norms(&self.z_corr, d, n);
        let z = &self.z_corr;
        let r = &self.r;
        self.y
            .par_chunks_mut(d)
            .enumerate()
            .for_each(|(ku, y_col)| {
                y_col.fill(0.0);
                for i in 0..n {
                    let w = r[ku * n + i] as f64 * inv_norms[i];
                    if w == 0.0 {
                        continue;
                    }
                    let z_col = &z[i * d..(i + 1) * d];
                    for t in 0..d {
                        y_col[t] += z_col[t] * w;
                    }
                }
            });
        l2_normalize_columns(&mut self.y, d, self.k);
    }
}

// ─── Cold-start R re-estimation (iterations 2+) ──────────────────────

impl HarmonyState {
    fn cold_start_r(&mut self) {
        // Cosine normalization is done column-by-column inside
        // `compute_distances`; no `z_cos` buffer is materialized (H9).
        self.dist_mat = compute_distances(&self.y, &self.z_corr, self.d, self.k, self.n);
        self.r = softmax_r_from_dist(&self.dist_mat, &self.sigma, self.k, self.n);
        let (o, e) = compute_o_e(
            &self.r,
            &self.cell_to_gb,
            &self.layout,
            &self.pr_b,
            self.k,
            self.n,
        );
        self.o = o;
        self.e = e;
    }
}

// ─── update_R: block-wise stochastic soft-assignment update ──────────

impl HarmonyState {
    fn update_r(&mut self) {
        let k = self.k;
        let n = self.n;
        let b = self.layout.b;
        let block_size = self.config.block_size.max(1.0 / n as f64);
        let n_blocks = (1.0 / block_size).ceil() as usize;

        // Shuffle cell order.
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut self.rng);

        let block_len = n.div_ceil(n_blocks);

        for blk in 0..n_blocks {
            let start = blk * block_len;
            if start >= n {
                break;
            }
            let end = (start + block_len).min(n);
            let block = &order[start..end];

            // (a) Decrement O and E by the block's current R contributions.
            for &i in block {
                for ci in 0..self.layout.c {
                    let gb = self.layout.cov_offset[ci] + self.covariates[ci].labels[i] as usize;
                    for ku in 0..k {
                        let r_ki = self.r[ku * n + i] as f64;
                        self.o[ku * b + gb] -= r_ki;
                        self.e[ku * b + gb] -= self.pr_b[gb] * r_ki;
                    }
                }
            }

            // (b,c,d) Compute new R per cell in parallel using snapshot O/E.
            // Each cell's new R row is independent given the current
            // (post-decrement) O/E state; the full rayon-parallel result is
            // written back serially to self.r below.
            //
            // Result layout: Vec of (cell_index, new_r_values[k]).
            let dist_mat = &self.dist_mat;
            let sigma = &self.sigma;
            let o = &self.o;
            let e_mat = &self.e;
            let theta = &self.theta;
            let layout = &self.layout;
            let covariates = &self.covariates;
            let new_r: Vec<(usize, Vec<f64>)> = block
                .par_iter()
                .map(|&i| {
                    // scale_dist = softmax(-dist/sigma), stabilized.
                    let mut vals = vec![0f64; k];
                    let mut max_neg = f64::NEG_INFINITY;
                    for ku in 0..k {
                        let v = -(dist_mat[ku * n + i] as f64) / sigma[ku];
                        vals[ku] = v;
                        if v > max_neg {
                            max_neg = v;
                        }
                    }
                    let mut sum_sd = 0f64;
                    for ku in 0..k {
                        let ev = (vals[ku] - max_neg).exp();
                        vals[ku] = ev;
                        sum_sd += ev;
                    }
                    if sum_sd > 0.0 {
                        let inv = 1.0 / sum_sd;
                        for v in vals.iter_mut() {
                            *v *= inv;
                        }
                    } else {
                        let u = 1.0 / k as f64;
                        for v in vals.iter_mut() {
                            *v = u;
                        }
                    }

                    // Diversity penalty — product over covariates of
                    // ((2E+1)/(O+E+1))^theta_b.
                    let mut pen = vec![1f64; k];
                    for ci in 0..layout.c {
                        let gb = layout.cov_offset[ci] + covariates[ci].labels[i] as usize;
                        let theta_b = theta[gb];
                        for ku in 0..k {
                            let o_kb = o[ku * b + gb];
                            let e_kb = e_mat[ku * b + gb];
                            let numer = 2.0 * e_kb + 1.0;
                            let denom = o_kb + e_kb + 1.0;
                            let ratio = if denom > 0.0 { numer / denom } else { 1.0 };
                            pen[ku] *= ratio.powf(theta_b);
                        }
                    }

                    // New R_i = vals * pen, L1-normalize.
                    let mut new_sum = 0f64;
                    for ku in 0..k {
                        let v = vals[ku] * pen[ku];
                        vals[ku] = v;
                        new_sum += v;
                    }
                    if new_sum > 0.0 && new_sum.is_finite() {
                        let inv = 1.0 / new_sum;
                        for v in vals.iter_mut() {
                            *v *= inv;
                        }
                    } else {
                        // Numerical fallback: uniform.
                        let u = 1.0 / k as f64;
                        for v in vals.iter_mut() {
                            *v = u;
                        }
                    }
                    (i, vals)
                })
                .collect();

            // Serial scatter into self.r (different cells → different columns,
            // so writes don't conflict, but rayon can't prove disjointness
            // without unsafe; the scatter is O(block_len * K) which is cheap).
            // Cast f64 vals to f32 on store to match the K×N storage layout.
            for (i, vals) in &new_r {
                for ku in 0..k {
                    self.r[ku * n + i] = vals[ku] as f32;
                }
            }

            // (e) Re-increment O/E with the new R contributions.
            for &i in block {
                for ci in 0..self.layout.c {
                    let gb = self.layout.cov_offset[ci] + self.covariates[ci].labels[i] as usize;
                    for ku in 0..k {
                        let r_ki = self.r[ku * n + i] as f64;
                        self.o[ku * b + gb] += r_ki;
                        self.e[ku * b + gb] += self.pr_b[gb] * r_ki;
                    }
                }
            }
        }
    }
}

// ─── Objective and convergence ───────────────────────────────────────

impl HarmonyState {
    /// Full Harmony objective (kmeans_err + entropy + cross_entropy) * 2000/N.
    fn compute_objective(&self) -> f64 {
        let k = self.k;
        let n = self.n;
        let b = self.layout.b;

        let norm_const = 2000.0 / n as f64;

        // kmeans_error = sum(R * dist). Both buffers are f32; promote
        // each factor to f64 before multiplying so the K·N reduction
        // accumulates in full f64 precision.
        let mut kmeans_err = 0f64;
        for ku in 0..k {
            let row_off = ku * n;
            for i in 0..n {
                kmeans_err += (self.r[row_off + i] as f64) * (self.dist_mat[row_off + i] as f64);
            }
        }

        // entropy = sum(xlogy(R, R) * sigma_k)
        // sigma broadcasts over K dimension (one sigma per k).
        let mut entropy = 0f64;
        for ku in 0..k {
            let s = self.sigma[ku];
            let row_off = ku * n;
            let mut row_e = 0f64;
            for i in 0..n {
                let v = self.r[row_off + i] as f64;
                if v > 0.0 {
                    row_e += v * v.ln();
                }
            }
            entropy += row_e * s;
        }

        // cross_entropy = sum(sigma_k * O[k,b] * theta_b * log((O+E+1)/(2E+1)))
        let mut cross = 0f64;
        for ku in 0..k {
            let s = self.sigma[ku];
            for gb in 0..b {
                let o_kb = self.o[ku * b + gb];
                let e_kb = self.e[ku * b + gb];
                let num = o_kb + e_kb + 1.0;
                let den = 2.0 * e_kb + 1.0;
                if num > 0.0 && den > 0.0 {
                    cross += s * o_kb * self.theta[gb] * (num / den).ln();
                }
            }
        }

        (kmeans_err + entropy + cross) * norm_const
    }
}

/// K-means windowed convergence check (absolute numerator).
fn check_convergence_kmeans(objectives: &[f64], window: usize, epsilon: f64) -> bool {
    if objectives.len() < 2 * window {
        return false;
    }
    let len = objectives.len();
    let old_sum: f64 = objectives[len - 2 * window..len - window].iter().sum();
    let new_sum: f64 = objectives[len - window..].iter().sum();
    if old_sum.abs() == 0.0 {
        return false;
    }
    ((old_sum - new_sum).abs() / old_sum.abs()) < epsilon
}

/// Harmony convergence check — signed numerator. Convergence fires only
/// when the objective is *decreasing*: if the objective increases, the
/// ratio is negative and convergence is not triggered. That constraint
/// implies the ratio must be non-negative AND below epsilon, i.e. small
/// improvement.
fn check_convergence_harmony(objectives: &[f64], epsilon: f64) -> bool {
    if objectives.len() < 2 {
        return false;
    }
    let len = objectives.len();
    let obj_old = objectives[len - 2];
    let obj_new = objectives[len - 1];
    if obj_old.abs() == 0.0 {
        return false;
    }
    let ratio = (obj_old - obj_new) / obj_old.abs();
    ratio >= 0.0 && ratio < epsilon
}

// ─── Clustering sub-loop ─────────────────────────────────────────────

impl HarmonyState {
    /// One Harmony iteration's clustering step. Returns true if k-means
    /// sub-loop converged early.
    fn cluster_iteration(&mut self, is_first: bool) -> bool {
        if !is_first {
            self.cold_start_r();
        }

        // Reset per-Harmony-iteration k-means objective window.
        let kmeans_objectives_start = self.objective_kmeans.len();
        let _ = kmeans_objectives_start;

        let mut local_objectives: Vec<f64> = Vec::new();

        for _sub in 0..self.config.max_iter_kmeans {
            // M-step, then the distances it invalidates, then the E-step —
            // harmonypy 0.2.0 `cluster()` orders them the same way.
            self.update_y();
            self.dist_mat = compute_distances(&self.y, &self.z_corr, self.d, self.k, self.n);
            self.update_r();
            let obj = self.compute_objective();
            local_objectives.push(obj);
            self.objective_kmeans.push(obj);
            if check_convergence_kmeans(
                &local_objectives,
                self.config.window_size,
                self.config.epsilon_kmeans,
            ) {
                return true;
            }
        }
        false
    }
}

// ─── Batch pruning / lambda / matrix inversion ───────────────────────

/// For cluster k, return (kept global batch indices, active covariates count).
/// A covariate is active if ≥2 of its levels survive the batch_prop_cutoff.
fn prune_batches_for_cluster(state: &HarmonyState, k_idx: usize) -> (Vec<usize>, usize) {
    let b = state.layout.b;
    let mut kept: Vec<usize> = Vec::new();
    let mut active_cov = 0usize;
    for ci in 0..state.layout.c {
        let n_levels = state.covariates[ci].n_levels;
        let mut local_kept: Vec<usize> = Vec::new();
        for lvl in 0..n_levels {
            let gb = state.layout.cov_offset[ci] + lvl;
            let n_b = state.n_b[gb];
            if n_b == 0.0 {
                continue;
            }
            let avg_r = state.o[k_idx * b + gb] / n_b;
            if avg_r > state.config.batch_prop_cutoff {
                local_kept.push(gb);
            }
        }
        if local_kept.len() >= 2 {
            kept.extend(local_kept);
            active_cov += 1;
        }
    }
    (kept, active_cov)
}

/// Build dynamic lambda for kept batches: [0, alpha*E[k,b_0], alpha*E[k,b_1], ...].
fn build_dynamic_lambda(state: &HarmonyState, k_idx: usize, kept: &[usize]) -> Vec<f64> {
    let b = state.layout.b;
    let alpha = state.config.alpha;
    let mut lam = vec![0f64; kept.len() + 1];
    for (j, &gb) in kept.iter().enumerate() {
        lam[j + 1] = alpha * state.e[k_idx * b + gb];
    }
    lam
}

/// Arrowhead inverse for a matrix of the form
///     [a   c^T]
///     [c    D ]
/// where D is diagonal. Matrix is passed as row-major (size x size).
///
/// Returns the row-major inverse in a new Vec. Errors with
/// `AccelError::NumericalInstability` when any `D[j]` is non-normal or
/// below `1e-15` in magnitude (typically an empty batch-level/cluster
/// combination) so the caller can fall back to the full LU inverse.
fn arrowhead_inverse(mat: &[f64], size: usize) -> Result<Vec<f64>> {
    debug_assert!(size >= 1);
    let a = mat[0];
    // Extract c (column) and D diagonal.
    let m = size - 1;
    let mut c = vec![0f64; m];
    let mut d = vec![0f64; m];
    for j in 0..m {
        c[j] = mat[(j + 1) * size]; // column 0, rows 1..=m
        d[j] = mat[(j + 1) * size + (j + 1)];
    }

    // Guard against near-zero diagonals that would blow up `c[j]*c[j]/d[j]`
    // or `1/d[j]` below. Pathological inputs (empty batch/cluster) reach
    // this path when `alpha=0` and all rows in a level are filtered out.
    for (j, &dj) in d.iter().enumerate() {
        if !dj.is_normal() || dj.abs() < 1e-15 {
            return Err(AccelError::NumericalInstability(format!(
                "arrowhead_inverse: near-zero diagonal d[{j}]={dj}; \
                 likely an empty batch-level/cluster combination"
            )));
        }
    }

    // u = a - c^T D^{-1} c
    let mut u = a;
    for j in 0..m {
        u -= c[j] * c[j] / d[j];
    }
    if !u.is_normal() || u.abs() < 1e-15 {
        return Err(AccelError::NumericalInstability(format!(
            "arrowhead_inverse: schur complement u={u} is non-normal"
        )));
    }

    let mut inv = vec![0f64; size * size];
    inv[0] = 1.0 / u;
    for j in 0..m {
        let val = -c[j] / (d[j] * u);
        inv[j + 1] = val; // inv[0, j+1]
        inv[(j + 1) * size] = val; // inv[j+1, 0]
    }
    for i in 0..m {
        for j in 0..m {
            let off_diag = c[i] * c[j] / (d[i] * d[j] * u);
            let diag = if i == j { 1.0 / d[i] } else { 0.0 };
            inv[(i + 1) * size + (j + 1)] = diag + off_diag;
        }
    }
    Ok(inv)
}

/// Full matrix inverse via faer partial-pivot LU. Input row-major, size x size.
fn full_matrix_inverse(mat: &[f64], size: usize) -> Result<Vec<f64>> {
    // faer uses column-major `Mat`. Transpose on the way in; since the
    // covariance matrix is symmetric, we can also just copy directly.
    let mut m = Mat::<f64>::zeros(size, size);
    for r in 0..size {
        for c in 0..size {
            m[(r, c)] = mat[r * size + c];
        }
    }
    let lu = m.as_ref().partial_piv_lu();
    let inv = lu.inverse();
    let mut out = vec![0f64; size * size];
    for r in 0..size {
        for c in 0..size {
            out[r * size + c] = inv[(r, c)];
        }
    }
    // Sanity: no NaN/Inf
    for v in &out {
        if !v.is_finite() {
            return Err(AccelError::LinAlg(
                "covariance matrix inversion produced non-finite values".to_string(),
            ));
        }
    }
    Ok(out)
}

// ─── Ridge correction (moe_correct_ridge) ────────────────────────────

impl HarmonyState {
    fn correct(&mut self) -> Result<()> {
        let n = self.n;
        let d = self.d;
        let k = self.k;
        let b = self.layout.b;
        let c_count = self.layout.c;

        // Reset Z_corr = Z_orig. Corrections are computed from the original.
        self.z_corr.copy_from_slice(&self.z_orig);

        for ku in 0..k {
            let (kept, active_cov) = prune_batches_for_cluster(self, ku);
            if active_cov == 0 || kept.is_empty() {
                continue;
            }
            let b_prime = kept.len();
            let size = b_prime + 1;

            // Build lambda for this cluster.
            let lambda = match &self.lambda_fixed {
                Some(lam) => {
                    let mut local = vec![0f64; size];
                    for (j, &gb) in kept.iter().enumerate() {
                        local[j + 1] = lam[gb + 1];
                    }
                    local
                }
                None => build_dynamic_lambda(self, ku, &kept),
            };

            // Build covariance matrix (size x size), row-major.
            // cov[0,0] = sum(O[k, kept])
            // cov[0, j+1] = cov[j+1, 0] = O[k, kept[j]]
            // cov[j+1, j+1] = O[k, kept[j]]
            let mut cov = vec![0f64; size * size];
            let mut sum_o = 0f64;
            for (j, &gb) in kept.iter().enumerate() {
                let o_kb = self.o[ku * b + gb];
                cov[j + 1] = o_kb; // row 0, col j+1
                cov[(j + 1) * size] = o_kb; // row j+1, col 0
                cov[(j + 1) * size + (j + 1)] = o_kb; // diagonal
                sum_o += o_kb;
            }
            cov[0] = sum_o;
            // Add diag(lambda).
            for j in 0..size {
                cov[j * size + j] += lambda[j];
            }

            // Invert: arrowhead for single-covariate, LU otherwise. Fall back
            // to the full LU inverse if the arrowhead guard trips on a
            // near-zero diagonal (H8).
            let inv_cov = if c_count == 1 {
                match arrowhead_inverse(&cov, size) {
                    Ok(inv) => inv,
                    Err(AccelError::NumericalInstability(msg)) => {
                        log::warn!(
                            "harmony: arrowhead_inverse unstable ({msg}); \
                             falling back to full LU inverse"
                        );
                        full_matrix_inverse(&cov, size)?
                    }
                    Err(e) => return Err(e),
                }
            } else {
                full_matrix_inverse(&cov, size)?
            };

            // Build z_sum per kept batch: z_sum[j] = sum_{i in batch_gb} Z_orig[:,i] * R[k,i]
            // (Since we reset Z_corr = Z_orig at the start, we use z_orig here.)
            //
            // Parallel across kept batches `j` — each writes to its own
            // disjoint d-row of `z_sum` and reads disjoint cell sets, so
            // there is no shared state across threads. Inner cell loop
            // stays sequential to preserve bit-exact summation order.
            // R is f32; promote to f64 on read.
            let mut z_sum = vec![0f64; b_prime * d]; // (B' x d), row-major
            let r_full = &self.r;
            let z_orig_full = &self.z_orig;
            let batch_index = &self.batch_index;
            z_sum
                .par_chunks_mut(d)
                .enumerate()
                .for_each(|(j, z_sum_row)| {
                    let gb = kept[j];
                    let (ci, lvl) = self.gb_to_cov_level(gb);
                    let cells = &batch_index[ci][lvl];
                    for &i in cells {
                        let r_ki = r_full[ku * n + i] as f64;
                        if r_ki == 0.0 {
                            continue;
                        }
                        let z_col = &z_orig_full[i * d..(i + 1) * d];
                        for t in 0..d {
                            z_sum_row[t] += z_col[t] * r_ki;
                        }
                    }
                });
            // z_sum_all = sum_j z_sum[j]
            let mut z_sum_all = vec![0f64; d];
            for j in 0..b_prime {
                let row_off = j * d;
                for t in 0..d {
                    z_sum_all[t] += z_sum[row_off + t];
                }
            }

            // W (size x d), row-major.
            // W[r, :] = inv_cov[r, 0] * z_sum_all + sum_j inv_cov[r, j+1] * z_sum[j]
            let mut w = vec![0f64; size * d];
            for r in 0..size {
                let ic0 = inv_cov[r * size];
                let row_off = r * d;
                for t in 0..d {
                    w[row_off + t] = ic0 * z_sum_all[t];
                }
                for j in 0..b_prime {
                    let icj = inv_cov[r * size + (j + 1)];
                    if icj == 0.0 {
                        continue;
                    }
                    let zj_off = j * d;
                    for t in 0..d {
                        w[row_off + t] += icj * z_sum[zj_off + t];
                    }
                }
            }

            // Zero out W[0, :] — the intercept is not a batch effect, so it
            // is not subtracted. harmonypy `moe_correct_ridge`: `W[0, :] = 0`.
            //
            // This row used to be copied into `Y[:, k]` first. That was the
            // stand-in for the missing M-step: a centroid written once per
            // *outer* iteration as a by-product of the correction solve.
            // `update_y` now owns `y`, and leaving the copy here would clobber
            // it once per outer iteration.
            for t in 0..d {
                w[t] = 0.0;
            }

            // Apply correction per batch:
            // Z_corr[:, cells_in_b] -= W[j+1, :].T * R[k, cells_in_b]
            // R is f32; promote to f64 on read for the multiply with
            // f64 W and z_corr.
            for (j, &gb) in kept.iter().enumerate() {
                let (ci, lvl) = self.gb_to_cov_level(gb);
                let cells = &self.batch_index[ci][lvl];
                let w_off = (j + 1) * d;
                for &i in cells {
                    let r_ki = self.r[ku * n + i] as f64;
                    if r_ki == 0.0 {
                        continue;
                    }
                    let z_off = i * d;
                    for t in 0..d {
                        self.z_corr[z_off + t] -= w[w_off + t] * r_ki;
                    }
                }
            }
        }

        Ok(())
    }

    #[inline]
    fn gb_to_cov_level(&self, gb: usize) -> (usize, usize) {
        for ci in 0..self.layout.c {
            let start = self.layout.cov_offset[ci];
            let end = start + self.covariates[ci].n_levels;
            if gb >= start && gb < end {
                return (ci, gb - start);
            }
        }
        // Caller invariant: `gb` ∈ `[0, layout.total_cov_levels)`. Every
        // call site synthesises `gb` from the partitioned batch-level loop,
        // so falling off the end here means the invariant was violated by
        // a coding bug. Panic in debug; in release, log at error level and
        // return a safe default so downstream indexing doesn't corrupt.
        let total_cov_levels = self.layout.cov_offset.last().copied().unwrap_or(0)
            + self.covariates.last().map(|c| c.n_levels).unwrap_or(0);
        debug_assert!(
            false,
            "gb_to_cov_level: gb={gb} is outside [0, {}), layout.c={}",
            total_cov_levels, self.layout.c,
        );
        log::error!(
            "gb_to_cov_level: gb={gb} outside [0, {}), layout.c={} — \
             caller invariant violated; returning safe default",
            total_cov_levels,
            self.layout.c,
        );
        (self.layout.c.saturating_sub(1), 0)
    }
}

// ─── Top-level driver ────────────────────────────────────────────────

impl HarmonyState {
    fn run(&mut self) -> Result<HarmonyResult> {
        let mut converged = false;
        let mut iters_used = 0usize;

        for iter in 0..self.config.max_iter {
            iters_used = iter + 1;

            self.cluster_iteration(iter == 0);
            self.correct()?;

            // Compute per-Harmony objective after correction (recompute R
            // from corrected embeddings first to keep the objective in sync
            // with the corrected state — the published Harmony2 pseudocode
            // computes the objective at the clustering step; we track the
            // last k-means sub-iteration objective, which reflects current R).
            if let Some(&last) = self.objective_kmeans.last() {
                self.objective_harmony.push(last);
            }

            if check_convergence_harmony(&self.objective_harmony, self.config.epsilon_harmony) {
                converged = true;
                break;
            }
        }

        // Transpose `Z_corr` from Harmony's internal (d × N) column-major
        // layout to AnnData's (N × d) row-major layout. Matches the layout
        // scanpy writes to `adata.obsm['X_pca_harmony']` and the R harmony
        // reference's `harmony::HarmonyMatrix$Z_corr` when cast to matrix.
        // Callers consuming `HarmonyResult.z_corr` should treat it as
        // row-major (`z_corr[i * d + j]` = cell `i`, PC `j`).
        let d = self.d;
        let n = self.n;
        let mut z_out = vec![0f64; n * d];
        for i in 0..n {
            for j in 0..d {
                z_out[i * d + j] = self.z_corr[j + i * d];
            }
        }

        Ok(HarmonyResult {
            z_corrected: z_out,
            n_obs: n,
            n_pcs: d,
            n_clusters: self.k,
            objective_harmony: std::mem::take(&mut self.objective_harmony),
            n_iterations: iters_used,
            converged,
            graph_replay: None,
        })
    }
}

// ─── GPU path (behind `gpu` feature) ─────────────────────────────────
// Declared as a submodule here (not a sibling) so the GPU orchestration can
// access the private `HarmonyState` clustering state it shares with the CPU
// path; the source file lives alongside at `harmony/gpu.rs`.
#[cfg(feature = "gpu")]
#[path = "gpu.rs"]
pub mod gpu;

// ─── Tests ────────────────────────────────────────────────────────────
#[cfg(test)]
#[path = "tests.rs"]
mod tests;

// ─── Pinned harmonypy reference (§7.4, ORG-7.21-4) ────────────────────
// Mounted here rather than beside `mod.rs` for the same reason `tests.rs` is:
// the arms drive `HarmonyState`'s private clustering buffers directly.
#[cfg(test)]
#[path = "harmony_reference_tests.rs"]
mod harmony_reference_tests;

#[cfg(test)]
#[path = "harmony_reference_values.rs"]
pub(crate) mod harmony_reference_values;
