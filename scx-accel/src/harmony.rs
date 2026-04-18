//! Harmony2 batch integration algorithm.
//!
//! Operates on dense PCA embeddings and corrects batch effects via iterative
//! soft k-means clustering + ridge regression. Reference: Korsunsky et al.
//!
//! This module is a clean-room implementation derived solely from the
//! published algorithm description in HARMONY2.md — no code is ported from
//! harmonypy (GPL-3.0) or the R harmony package.

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
            max_iter_kmeans: 4,
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
    /// Final soft cluster assignments, K x N, row-major f64.
    pub r_matrix: Vec<f64>,
    /// Number of clusters.
    pub n_clusters: usize,
    /// Per-iteration Harmony objective values.
    pub objective_harmony: Vec<f64>,
    /// Number of iterations until convergence (or max_iter).
    pub n_iterations: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
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
    // Clustering (f64)
    y: Vec<f64>,        // d x K, column-major (centroids as columns)
    r: Vec<f64>,        // K x N, row-major (cluster x cell)
    dist_mat: Vec<f64>, // K x N, row-major
    o: Vec<f64>,        // K x B, row-major
    e: Vec<f64>,        // K x B, row-major
    // Batch structure
    covariates: Vec<BatchCovariate>,
    layout: BatchLayout,
    /// For each covariate c, for each level l, the list of cell indices.
    batch_index: Vec<Vec<Vec<usize>>>,
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
    if let Some(n) = config.n_threads {
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
        let (o, e) = compute_o_e(&r, &batch_index, &layout, &pr_b, k, n);

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

/// dist[k, i] = 2 * (1 - Y[:, k] · normalize(z[:, i])).
///
/// `z` is the un-normalized d·N embedding (column-major); columns are
/// L2-normalized on the fly so we do not need to materialize a separate
/// `z_cos`. Output is row-major (K x N); rows are disjoint slices of
/// length N, so we parallelize across clusters.
fn compute_distances(y: &[f64], z: &[f64], d: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0f64; k * n];
    out.par_chunks_mut(n).enumerate().for_each(|(ku, row)| {
        let y_col = &y[ku * d..(ku + 1) * d];
        for i in 0..n {
            let z_col = &z[i * d..(i + 1) * d];
            let n2: f64 = z_col.iter().map(|v| v * v).sum();
            let inv = if n2 > 0.0 { 1.0 / n2.sqrt() } else { 0.0 };
            let mut dot = 0f64;
            for j in 0..d {
                dot += y_col[j] * z_col[j];
            }
            row[i] = 2.0 * (1.0 - dot * inv);
        }
    });
    out
}

/// R = softmax(-dist / sigma[k]) per column, numerically stabilized.
///
/// `dist` is row-major K x N; sigma has length K; returns row-major K x N.
///
/// Addresses L16: an earlier review flagged this as a "no-op rayon stub",
/// reflecting an intermediate refactor state. Both phases (per-cell softmax
/// and the K×N transpose) are parallelised via `par_chunks_mut`; confirmed
/// by the cell-count scaling in the T1 Harmony benchmark.
fn softmax_r_from_dist(dist: &[f64], sigma: &[f64], k: usize, n: usize) -> Vec<f64> {
    // Two-pass layout: compute each cell's softmax into a col-major
    // temporary (N contiguous K-blocks) so the hot loop writes are
    // contiguous and can run per-cell in parallel; then transpose
    // row-by-row in parallel to the final row-major K x N output.
    let mut col_major = vec![0f64; k * n];
    col_major
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(i, cell)| {
            let mut max_neg = f64::NEG_INFINITY;
            for ku in 0..k {
                let v = -dist[ku * n + i] / sigma[ku];
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
        });

    let mut r = vec![0f64; k * n];
    r.par_chunks_mut(n).enumerate().for_each(|(ku, row)| {
        for i in 0..n {
            row[i] = col_major[i * k + ku];
        }
    });
    r
}

/// Compute O (K x B) and E (K x B) from R (K x N), batch_index, pr_b.
///
/// O[k, b] = sum_{i in batch b} R[k, i]
/// E[k, b] = pr_b[b] * sum_i R[k, i]
fn compute_o_e(
    r: &[f64],
    batch_index: &[Vec<Vec<usize>>],
    layout: &BatchLayout,
    pr_b: &[f64],
    k: usize,
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let b = layout.b;
    let mut o = vec![0f64; k * b];
    let mut e = vec![0f64; k * b];

    // Row-sums of R (per cluster).
    let mut row_sum = vec![0f64; k];
    for ku in 0..k {
        let mut s = 0f64;
        for i in 0..n {
            s += r[ku * n + i];
        }
        row_sum[ku] = s;
    }

    for ci in 0..layout.c {
        for lvl in 0..batch_index[ci].len() {
            let gb = layout.cov_offset[ci] + lvl;
            let cells = &batch_index[ci][lvl];
            for ku in 0..k {
                let mut s = 0f64;
                let row_off = ku * n;
                for &i in cells {
                    s += r[row_off + i];
                }
                o[ku * b + gb] = s;
                e[ku * b + gb] = pr_b[gb] * row_sum[ku];
            }
        }
    }

    (o, e)
}

// ─── K-means++ seeding (Harmony variant) ─────────────────────────────

/// L2-normalize `z[i*d..(i+1)*d]` into `out`. Returns the column's L2 norm
/// (zero-norm columns are written as all-zero).
#[inline]
fn normalize_col_into(z: &[f64], d: usize, i: usize, out: &mut [f64]) -> f64 {
    let col = &z[i * d..(i + 1) * d];
    let n2: f64 = col.iter().map(|v| v * v).sum();
    if n2 > 0.0 {
        let inv = 1.0 / n2.sqrt();
        for (o, &c) in out.iter_mut().zip(col.iter()) {
            *o = c * inv;
        }
        n2.sqrt()
    } else {
        out.fill(0.0);
        0.0
    }
}

/// Gumbel-max weighted sampling from the most recently chosen centroid.
/// Returns a (d x K) column-major matrix of centroids (already L2-normalized
/// — equal to the normalized embedding column at each chosen cell index).
///
/// `z` is the un-normalized embedding; columns are L2-normalized on the fly
/// to avoid materializing a separate `z_cos` copy (H9).
fn kmeans_plus_plus(z: &[f64], d: usize, n: usize, k: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
    let mut y = vec![0f64; d * k];
    let mut chosen: Vec<usize> = Vec::with_capacity(k);
    let mut scratch = vec![0f64; d];

    // First centroid: uniform random cell (normalized).
    let i0 = rng.gen_range(0..n);
    chosen.push(i0);
    normalize_col_into(z, d, i0, &mut scratch);
    y[..d].copy_from_slice(&scratch);

    for ci in 1..k {
        // Compute cosine distance from the most recently chosen centroid.
        let last = &y[(ci - 1) * d..ci * d];

        // Try up to a few resamples on duplicate collisions.
        let mut picked = usize::MAX;
        'outer: for _attempt in 0..10 {
            let mut best_j = 0usize;
            let mut best_val = f64::INFINITY;
            for j in 0..n {
                let z_col = &z[j * d..(j + 1) * d];
                let n2: f64 = z_col.iter().map(|v| v * v).sum();
                if n2 == 0.0 {
                    continue;
                }
                let inv = 1.0 / n2.sqrt();
                let mut dot = 0f64;
                for t in 0..d {
                    dot += last[t] * z_col[t];
                }
                let dist = (2.0 * (1.0 - dot * inv)).abs();
                if dist == 0.0 {
                    continue;
                }
                let u: f64 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
                let v = -u.ln() / dist;
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
        normalize_col_into(z, d, picked, &mut scratch);
        y[ci * d..(ci + 1) * d].copy_from_slice(&scratch);
    }

    y
}

/// Lloyd's algorithm refinement on cosine distance with L2-normalized
/// centroids (keep_existing init). Runs `n_iter` iterations in place on `y`.
/// `z` is un-normalized; each column is L2-normalized inline (H9).
fn kmeans_refine(z: &[f64], y: &mut [f64], d: usize, n: usize, k: usize, n_iter: usize) {
    let mut scratch = vec![0f64; d];
    for _ in 0..n_iter {
        let mut assign = vec![0usize; n];
        for i in 0..n {
            let norm = normalize_col_into(z, d, i, &mut scratch);
            if norm == 0.0 {
                assign[i] = 0;
                continue;
            }
            let mut best_k = 0usize;
            let mut best_dot = f64::NEG_INFINITY;
            for ku in 0..k {
                let y_col = &y[ku * d..(ku + 1) * d];
                let mut dot = 0f64;
                for t in 0..d {
                    dot += y_col[t] * scratch[t];
                }
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
            let norm = normalize_col_into(z, d, i, &mut scratch);
            if norm == 0.0 {
                continue;
            }
            let ku = assign[i];
            let off = ku * d;
            for t in 0..d {
                sums[off + t] += scratch[t];
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

// ─── Cold-start R re-estimation (iterations 2+) ──────────────────────

impl HarmonyState {
    fn cold_start_r(&mut self) {
        // Cosine normalization is done column-by-column inside
        // `compute_distances`; no `z_cos` buffer is materialized (H9).
        self.dist_mat = compute_distances(&self.y, &self.z_corr, self.d, self.k, self.n);
        self.r = softmax_r_from_dist(&self.dist_mat, &self.sigma, self.k, self.n);
        let (o, e) = compute_o_e(
            &self.r,
            &self.batch_index,
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
                        let r_ki = self.r[ku * n + i];
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
                        let v = -dist_mat[ku * n + i] / sigma[ku];
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
            for (i, vals) in &new_r {
                for ku in 0..k {
                    self.r[ku * n + i] = vals[ku];
                }
            }

            // (e) Re-increment O/E with the new R contributions.
            for &i in block {
                for ci in 0..self.layout.c {
                    let gb = self.layout.cov_offset[ci] + self.covariates[ci].labels[i] as usize;
                    for ku in 0..k {
                        let r_ki = self.r[ku * n + i];
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

        // kmeans_error = sum(R * dist)
        let mut kmeans_err = 0f64;
        for ku in 0..k {
            let row_off = ku * n;
            for i in 0..n {
                kmeans_err += self.r[row_off + i] * self.dist_mat[row_off + i];
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
                let v = self.r[row_off + i];
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
/// when the objective is *decreasing* (HARMONY2.md:191-195: "If the
/// objective increases, the ratio is negative and convergence is not
/// triggered."). That constraint implies the ratio must be non-negative
/// AND below epsilon, i.e. small improvement.
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
            let mut z_sum = vec![0f64; b_prime * d]; // (B' x d), row-major
            for (j, &gb) in kept.iter().enumerate() {
                // Recover (covariate, level) from global batch index.
                let (ci, lvl) = self.gb_to_cov_level(gb);
                let cells = &self.batch_index[ci][lvl];
                for &i in cells {
                    let r_ki = self.r[ku * n + i];
                    if r_ki == 0.0 {
                        continue;
                    }
                    let z_col = &self.z_orig[i * d..(i + 1) * d];
                    let row_off = j * d;
                    for t in 0..d {
                        z_sum[row_off + t] += z_col[t] * r_ki;
                    }
                }
            }
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

            // Centroid: Y[:, k] = W[0, :]. Zero out W[0, :].
            let y_off = ku * d;
            for t in 0..d {
                self.y[y_off + t] = w[t];
                w[t] = 0.0;
            }

            // Apply correction per batch:
            // Z_corr[:, cells_in_b] -= W[j+1, :].T * R[k, cells_in_b]
            for (j, &gb) in kept.iter().enumerate() {
                let (ci, lvl) = self.gb_to_cov_level(gb);
                let cells = &self.batch_index[ci][lvl];
                let w_off = (j + 1) * d;
                for &i in cells {
                    let r_ki = self.r[ku * n + i];
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

        // L2-normalize centroid columns.
        l2_normalize_columns(&mut self.y, self.d, self.k);
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
            // with the corrected state — HARMONY2.md pseudocode computes
            // the objective at the clustering step; we track the last
            // k-means sub-iteration objective, which reflects current R).
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
            r_matrix: std::mem::take(&mut self.r),
            n_clusters: self.k,
            objective_harmony: std::mem::take(&mut self.objective_harmony),
            n_iterations: iters_used,
            converged,
        })
    }
}

// ─── GPU path (behind `gpu` feature) ─────────────────────────────────

#[cfg(feature = "gpu")]
mod gpu_impl {
    use super::*;
    // `gpu_harmony_softmax_penalty` is exposed by scx-gpu for future fusion
    // (one launch for softmax + diversity penalty). This orchestrator keeps
    // the block-wise R update on CPU for now, so we don't import it here.
    use scx_gpu::{
        gpu_harmony_correction, gpu_harmony_distances, gpu_harmony_l2_normalize_cols, GpuDevice,
    };

    /// Helper: convert a `Vec<f64>` slice to f32 (GPU kernels use f32).
    fn f64_to_f32(v: &[f64]) -> Vec<f32> {
        v.iter().map(|&x| x as f32).collect()
    }

    /// GPU-accelerated Harmony2 integration.
    ///
    /// Reuses the CPU `HarmonyState` for orchestration, substituting GPU
    /// kernels for the three compute-hot operations: cosine distance
    /// computation, column L2 normalization, and per-batch correction
    /// scatter-subtract. K-means++ seeding, soft assignment block updates,
    /// objective/convergence tracking, and covariance inversion remain on
    /// CPU — each is either inexpensive or requires branching that fits
    /// poorly on GPU.
    ///
    /// Output is bit-compatible in structure (same `HarmonyResult` shape)
    /// with the CPU path, but values differ slightly due to f32 rounding
    /// on GPU vs. f64 on CPU. Per-PC Pearson correlation with the CPU
    /// reference should be >= 0.99.
    pub fn harmony_integrate_gpu(
        embeddings: &[f32],
        n_obs: usize,
        n_pcs: usize,
        covariates: &[BatchCovariate],
        config: &HarmonyConfig,
    ) -> Result<HarmonyResult> {
        // Initialise CPU state (validates inputs, runs kmeans++/Lloyd, sets up
        // initial R/O/E). Reusing this keeps the two paths algorithmically in
        // step for the first iteration.
        let mut state = HarmonyState::new(embeddings, n_obs, n_pcs, covariates, config)?;
        let n = state.n;
        let d = state.d;
        let k = state.k;
        let b = state.layout.b;
        let c_count = state.layout.c;

        let dev =
            GpuDevice::new(0).map_err(|e| AccelError::LinAlg(format!("GPU init failed: {e}")))?;

        // Upload Z_orig once — it never changes.
        let d_z_orig = dev
            .htod_copy(&f64_to_f32(&state.z_orig))
            .map_err(|e| AccelError::LinAlg(format!("upload Z_orig: {e}")))?;

        // Persistent device buffers (reused across iterations).
        let mut d_z_corr = dev
            .htod_copy(&f64_to_f32(&state.z_orig))
            .map_err(|e| AccelError::LinAlg(format!("alloc Z_corr: {e}")))?;

        // Flatten per-covariate labels to (C x N) row-major i32 and upload.
        // Labels are u32 category indices; the GPU kernel uses i32, so
        // assert they fit. Batch count per covariate is bounded in practice
        // by `max_batches` (default 1024), well under i32::MAX.
        let mut labels_flat = vec![0i32; c_count * n];
        for (ci, cov) in covariates.iter().enumerate() {
            for (i, &lab) in cov.labels.iter().enumerate() {
                debug_assert!(lab <= i32::MAX as u32, "batch label {lab} exceeds i32::MAX");
                labels_flat[ci * n + i] = lab as i32;
            }
        }
        let _d_labels = dev
            .htod_copy(&labels_flat)
            .map_err(|e| AccelError::LinAlg(format!("upload batch labels: {e}")))?;

        let mut converged = false;
        let mut iters_used = 0usize;

        for iter in 0..state.config.max_iter {
            iters_used = iter + 1;

            if iter > 0 {
                // Cold-start R on GPU: Z_cos = l2_normalize(Z_corr), then
                // dist = 2*(1 - Y^T Z_cos). We keep R/dist on CPU once
                // downloaded so the k-means sub-loop (CPU) can proceed.
                let mut d_z_cos = dev
                    .htod_copy(&f64_to_f32(&state.z_corr))
                    .map_err(|e| AccelError::LinAlg(format!("upload Z_corr->cos: {e}")))?;
                gpu_harmony_l2_normalize_cols(&dev, &mut d_z_cos, d, n)
                    .map_err(|e| AccelError::LinAlg(format!("GPU L2 normalize: {e}")))?;

                let d_y = dev
                    .htod_copy(&f64_to_f32(&state.y))
                    .map_err(|e| AccelError::LinAlg(format!("upload Y: {e}")))?;
                let mut d_dist = dev
                    .alloc_zeros::<f32>(k * n)
                    .map_err(|e| AccelError::LinAlg(format!("alloc dist: {e}")))?;
                gpu_harmony_distances(&dev, &d_y, &d_z_cos, &mut d_dist, d, k, n)
                    .map_err(|e| AccelError::LinAlg(format!("GPU distances: {e}")))?;
                dev.synchronize()
                    .map_err(|e| AccelError::LinAlg(format!("sync: {e}")))?;

                let dist_f32 = dev
                    .dtoh_copy(&d_dist)
                    .map_err(|e| AccelError::LinAlg(format!("download dist: {e}")))?;
                // No host-side `z_cos` mirror is kept (H9) — cosine
                // normalization lives on-device for this iteration and is
                // recomputed on-the-fly by CPU paths on later iterations.
                state.dist_mat = dist_f32.iter().map(|&v| v as f64).collect();
                state.r = softmax_r_from_dist(&state.dist_mat, &state.sigma, k, n);
                let (o, e) = compute_o_e(
                    &state.r,
                    &state.batch_index,
                    &state.layout,
                    &state.pr_b,
                    k,
                    n,
                );
                state.o = o;
                state.e = e;
            }

            // K-means sub-loop stays on CPU — block-wise R update is branchy
            // and relies on interleaved O/E increments that aren't a natural
            // GPU fit at this workload size.
            let mut local_obj: Vec<f64> = Vec::new();
            for _sub in 0..state.config.max_iter_kmeans {
                state.update_r();
                let obj = state.compute_objective();
                local_obj.push(obj);
                state.objective_kmeans.push(obj);
                if check_convergence_kmeans(
                    &local_obj,
                    state.config.window_size,
                    state.config.epsilon_kmeans,
                ) {
                    break;
                }
            }

            // --- Correction step (GPU scatter-subtract) ---
            //
            // Reset Z_corr = Z_orig on device.
            dev.stream()
                .memcpy_dtod(&d_z_orig, &mut d_z_corr)
                .map_err(|e| AccelError::LinAlg(format!("reset Z_corr: {e}")))?;

            // Mirror CPU correction logic, with the per-batch scatter done on
            // GPU. Centroid rows W[0, :] still flow through CPU state.y.
            for ku in 0..k {
                let (kept, active_cov) = prune_batches_for_cluster(&state, ku);
                if active_cov == 0 || kept.is_empty() {
                    continue;
                }
                let b_prime = kept.len();
                let size = b_prime + 1;

                let lambda = match &state.lambda_fixed {
                    Some(lam) => {
                        let mut local = vec![0f64; size];
                        for (j, &gb) in kept.iter().enumerate() {
                            local[j + 1] = lam[gb + 1];
                        }
                        local
                    }
                    None => build_dynamic_lambda(&state, ku, &kept),
                };

                let mut cov = vec![0f64; size * size];
                let mut sum_o = 0f64;
                for (j, &gb) in kept.iter().enumerate() {
                    let o_kb = state.o[ku * b + gb];
                    cov[j + 1] = o_kb;
                    cov[(j + 1) * size] = o_kb;
                    cov[(j + 1) * size + (j + 1)] = o_kb;
                    sum_o += o_kb;
                }
                cov[0] = sum_o;
                for j in 0..size {
                    cov[j * size + j] += lambda[j];
                }

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

                // z_sum_all + z_sum[j] from Z_orig (CPU-resident).
                let mut z_sum = vec![0f64; b_prime * d];
                for (j, &gb) in kept.iter().enumerate() {
                    let (ci, lvl) = state.gb_to_cov_level(gb);
                    let cells = &state.batch_index[ci][lvl];
                    for &i in cells {
                        let r_ki = state.r[ku * n + i];
                        if r_ki == 0.0 {
                            continue;
                        }
                        let z_col = &state.z_orig[i * d..(i + 1) * d];
                        let row_off = j * d;
                        for t in 0..d {
                            z_sum[row_off + t] += z_col[t] * r_ki;
                        }
                    }
                }
                let mut z_sum_all = vec![0f64; d];
                for j in 0..b_prime {
                    let row_off = j * d;
                    for t in 0..d {
                        z_sum_all[t] += z_sum[row_off + t];
                    }
                }

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

                // Extract centroid into CPU state.y; zero W[0, :].
                let y_off = ku * d;
                for t in 0..d {
                    state.y[y_off + t] = w[t];
                    w[t] = 0.0;
                }

                // Upload R[k, :] once for this cluster.
                let r_row_f32: Vec<f32> = state.r[ku * n..(ku + 1) * n]
                    .iter()
                    .map(|&v| v as f32)
                    .collect();
                let d_r_row = dev
                    .htod_copy(&r_row_f32)
                    .map_err(|e| AccelError::LinAlg(format!("upload R row: {e}")))?;

                // Apply each kept batch on the GPU.
                for (j, &gb) in kept.iter().enumerate() {
                    let (ci, lvl) = state.gb_to_cov_level(gb);
                    let cells_i32: Vec<i32> = state.batch_index[ci][lvl]
                        .iter()
                        .map(|&i| i as i32)
                        .collect();
                    if cells_i32.is_empty() {
                        continue;
                    }
                    let d_cells = dev
                        .htod_copy(&cells_i32)
                        .map_err(|e| AccelError::LinAlg(format!("upload cells: {e}")))?;
                    let w_row_f32: Vec<f32> = w[(j + 1) * d..(j + 2) * d]
                        .iter()
                        .map(|&v| v as f32)
                        .collect();
                    let d_w = dev
                        .htod_copy(&w_row_f32)
                        .map_err(|e| AccelError::LinAlg(format!("upload W row: {e}")))?;
                    gpu_harmony_correction(&dev, &mut d_z_corr, d, n, &d_cells, &d_r_row, &d_w)
                        .map_err(|e| AccelError::LinAlg(format!("GPU correction: {e}")))?;
                }
            }

            // L2-normalize Y columns on GPU (d x K).
            let mut d_y = dev
                .htod_copy(&f64_to_f32(&state.y))
                .map_err(|e| AccelError::LinAlg(format!("upload Y for norm: {e}")))?;
            gpu_harmony_l2_normalize_cols(&dev, &mut d_y, d, k)
                .map_err(|e| AccelError::LinAlg(format!("GPU L2 normalize Y: {e}")))?;
            dev.synchronize()
                .map_err(|e| AccelError::LinAlg(format!("sync: {e}")))?;
            let y_back = dev
                .dtoh_copy(&d_y)
                .map_err(|e| AccelError::LinAlg(format!("download Y: {e}")))?;
            for (dst, src) in state.y.iter_mut().zip(y_back.iter()) {
                *dst = *src as f64;
            }

            // Sync Z_corr back to CPU state so cold_start_r can consume it
            // on the next iteration.
            let z_corr_back = dev
                .dtoh_copy(&d_z_corr)
                .map_err(|e| AccelError::LinAlg(format!("download Z_corr: {e}")))?;
            for (dst, src) in state.z_corr.iter_mut().zip(z_corr_back.iter()) {
                *dst = *src as f64;
            }

            if let Some(&last) = state.objective_kmeans.last() {
                state.objective_harmony.push(last);
            }
            if check_convergence_harmony(&state.objective_harmony, state.config.epsilon_harmony) {
                converged = true;
                break;
            }
        }

        // Build output as (N x d) row-major f64.
        let mut z_out = vec![0f64; n * d];
        for i in 0..n {
            for j in 0..d {
                z_out[i * d + j] = state.z_corr[j + i * d];
            }
        }

        Ok(HarmonyResult {
            z_corrected: z_out,
            n_obs: n,
            n_pcs: d,
            r_matrix: std::mem::take(&mut state.r),
            n_clusters: state.k,
            objective_harmony: std::mem::take(&mut state.objective_harmony),
            n_iterations: iters_used,
            converged,
        })
    }
}

#[cfg(feature = "gpu")]
pub use gpu_impl::harmony_integrate_gpu;

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- helpers ----------------------------------------------------------

    /// Random f32 row-major embeddings (N x d).
    fn random_embeddings(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n * d).map(|_| rng.gen::<f32>() - 0.5).collect()
    }

    /// Two-Gaussian-cluster embeddings with a batch variable that shifts one
    /// of the clusters; used to verify that a Harmony iteration reduces the
    /// objective.
    fn batched_gaussian(n_per: usize, d: usize, seed: u64) -> (Vec<f32>, Vec<u32>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut emb = Vec::with_capacity(4 * n_per * d);
        let mut batch = Vec::with_capacity(4 * n_per);
        // Cluster A, batch 0
        for _ in 0..n_per {
            for j in 0..d {
                let mean = if j == 0 { 2.0 } else { 0.0 };
                emb.push(mean + 0.3 * (rng.gen::<f32>() - 0.5));
            }
            batch.push(0);
        }
        // Cluster A, batch 1 (shifted)
        for _ in 0..n_per {
            for j in 0..d {
                let mean = if j == 0 {
                    2.0
                } else if j == 1 {
                    1.0
                } else {
                    0.0
                };
                emb.push(mean + 0.3 * (rng.gen::<f32>() - 0.5));
            }
            batch.push(1);
        }
        // Cluster B, batch 0
        for _ in 0..n_per {
            for j in 0..d {
                let mean = if j == 0 { -2.0 } else { 0.0 };
                emb.push(mean + 0.3 * (rng.gen::<f32>() - 0.5));
            }
            batch.push(0);
        }
        // Cluster B, batch 1 (shifted)
        for _ in 0..n_per {
            for j in 0..d {
                let mean = if j == 0 {
                    -2.0
                } else if j == 1 {
                    1.0
                } else {
                    0.0
                };
                emb.push(mean + 0.3 * (rng.gen::<f32>() - 0.5));
            }
            batch.push(1);
        }
        (emb, batch)
    }

    fn build_state(
        emb: &[f32],
        n: usize,
        d: usize,
        labels: Vec<u32>,
        n_levels: usize,
    ) -> HarmonyState {
        let cov = BatchCovariate {
            labels,
            n_levels,
            name: None,
        };
        let config = HarmonyConfig {
            n_clusters: Some(4),
            random_state: 42,
            ..Default::default()
        };
        HarmonyState::new(emb, n, d, &[cov], &config).unwrap()
    }

    // --- tests ------------------------------------------------------------

    #[test]
    fn test_kmeans_pp_distinct() {
        let n = 200;
        let d = 5;
        let k = 8;
        let emb = random_embeddings(n, d, 1);
        // Transpose + normalize, mimicking HarmonyState::new prelude.
        let mut z = vec![0f64; d * n];
        for i in 0..n {
            for j in 0..d {
                z[j + i * d] = emb[i * d + j] as f64;
            }
        }
        l2_normalize_columns(&mut z, d, n);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let y = kmeans_plus_plus(&z, d, n, k, &mut rng);
        // All K centroids should be distinct (d-dim vectors).
        for a in 0..k {
            for b in (a + 1)..k {
                let sa = &y[a * d..(a + 1) * d];
                let sb = &y[b * d..(b + 1) * d];
                let eq = sa.iter().zip(sb).all(|(x, y)| (x - y).abs() < 1e-12);
                assert!(!eq, "centroids {} and {} are identical", a, b);
            }
        }
    }

    #[test]
    fn test_soft_assignments_sum_to_one() {
        let n = 150;
        let d = 4;
        let emb = random_embeddings(n, d, 2);
        let labels = (0..n as u32).map(|i| i % 3).collect::<Vec<_>>();
        let s = build_state(&emb, n, d, labels, 3);
        let k = s.k;
        for i in 0..n {
            let mut sum = 0f64;
            for ku in 0..k {
                sum += s.r[ku * n + i];
            }
            assert!((sum - 1.0).abs() < 1e-9, "column {} sum={}", i, sum);
        }
    }

    #[test]
    fn test_o_e_consistency() {
        let n = 120;
        let d = 4;
        let emb = random_embeddings(n, d, 3);
        let labels = (0..n as u32).map(|i| i % 4).collect::<Vec<_>>();
        let s = build_state(&emb, n, d, labels, 4);
        let k = s.k;
        let b = s.layout.b;

        // For single covariate: sum_b O[k,b] == sum_i R[k,i]
        for ku in 0..k {
            let mut row_sum = 0f64;
            for i in 0..n {
                row_sum += s.r[ku * n + i];
            }
            let mut o_sum = 0f64;
            for gb in 0..b {
                o_sum += s.o[ku * b + gb];
            }
            assert!(
                (row_sum - o_sum).abs() < 1e-8,
                "cluster {}: row_sum={} o_sum={}",
                ku,
                row_sum,
                o_sum
            );
        }
        // E[k,b] ≈ pr_b[b] * row_sum(R).
        for ku in 0..k {
            let mut row_sum = 0f64;
            for i in 0..n {
                row_sum += s.r[ku * n + i];
            }
            for gb in 0..b {
                let expected = s.pr_b[gb] * row_sum;
                let got = s.e[ku * b + gb];
                assert!(
                    (expected - got).abs() < 1e-8,
                    "E[{},{}] expected {} got {}",
                    ku,
                    gb,
                    expected,
                    got
                );
            }
        }
    }

    #[test]
    fn test_arrowhead_inverse_matches_lu() {
        // Build a random arrowhead matrix.
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let size = 6; // B' = 5
        let mut mat = vec![0f64; size * size];
        let a: f64 = 5.0 + rng.gen::<f64>();
        mat[0] = a;
        for j in 1..size {
            let c: f64 = rng.gen::<f64>() + 0.1;
            let d: f64 = rng.gen::<f64>() + 2.0;
            mat[j] = c; // row 0
            mat[j * size] = c; // col 0
            mat[j * size + j] = d; // diagonal
        }
        // Ensure symmetric positive-definiteness-ish by bumping diagonal.
        // (Not strictly required for LU.)

        let inv_ah = arrowhead_inverse(&mat, size).unwrap();
        let inv_lu = full_matrix_inverse(&mat, size).unwrap();
        for r in 0..size {
            for c in 0..size {
                let d = inv_ah[r * size + c] - inv_lu[r * size + c];
                assert!(
                    d.abs() < 1e-8,
                    "mismatch at ({},{}): ah={} lu={}",
                    r,
                    c,
                    inv_ah[r * size + c],
                    inv_lu[r * size + c]
                );
            }
        }
    }

    #[test]
    fn test_convergence_detection() {
        // Asymptotic decreasing objective: obj → 10 from above, so the
        // relative decrease shrinks over time and eventually crosses epsilon.
        let objs: Vec<f64> = (0..30).map(|i| 10.0 + 1.0 / (i as f64 + 1.0)).collect();
        let mut fired_at = None;
        for i in 1..objs.len() {
            if check_convergence_harmony(&objs[..=i], 1e-2) {
                fired_at = Some(i);
                break;
            }
        }
        assert!(fired_at.is_some(), "Harmony convergence never fired");

        // Increasing objective must NOT fire Harmony convergence (signed).
        let rising: Vec<f64> = (0..10).map(|i| 10.0 + i as f64).collect();
        assert!(!check_convergence_harmony(&rising, 1e-2));

        // k-means (window=3): needs ≥6 samples; an asymptotic sequence
        // eventually has window sums nearly equal, triggering convergence.
        assert!(check_convergence_kmeans(&objs, 3, 1e-3));
        assert!(!check_convergence_kmeans(&objs[..5], 3, 1e-3));
    }

    #[test]
    fn test_batch_pruning() {
        // Construct a state with one tiny batch that should be excluded.
        let n = 100;
        let d = 4;
        let emb = random_embeddings(n, d, 5);
        // 3 levels: level 0 has 1 cell (tiny), levels 1 and 2 split the rest.
        let mut labels = vec![0u32; n];
        labels[0] = 0;
        for (i, lab) in labels.iter_mut().enumerate().take(n).skip(1) {
            *lab = if i % 2 == 0 { 1 } else { 2 };
        }
        let cov = BatchCovariate {
            labels,
            n_levels: 3,
            name: None,
        };
        let config = HarmonyConfig {
            n_clusters: Some(4),
            batch_prop_cutoff: 0.01, // tiny-batch level has avg_R ≈ 1/N which may be > cutoff
            ..Default::default()
        };
        let state = HarmonyState::new(&emb, n, d, &[cov], &config).unwrap();

        // With n_levels=3 and level 0 having 1 cell, the pruning may or may
        // not drop level 0 depending on avg_R. What we can guarantee: the
        // function returns a sensible (kept, active_cov) pair.
        let (kept, active) = prune_batches_for_cluster(&state, 0);
        assert!(active <= 1);
        if active == 1 {
            assert!(kept.len() >= 2); // covariate survival rule
        }

        // Explicit 2-level case where one level fails cutoff.
        let n2 = 100;
        let mut labels2 = vec![1u32; n2];
        labels2[0] = 0; // single cell in level 0
        let cov2 = BatchCovariate {
            labels: labels2,
            n_levels: 2,
            name: None,
        };
        let config2 = HarmonyConfig {
            n_clusters: Some(3),
            batch_prop_cutoff: 0.5, // force level 0 to fail
            ..Default::default()
        };
        let emb2 = random_embeddings(n2, d, 6);
        let state2 = HarmonyState::new(&emb2, n2, d, &[cov2], &config2).unwrap();
        let (kept2, active2) = prune_batches_for_cluster(&state2, 0);
        // Only 1 level survives, covariate drops out.
        assert_eq!(active2, 0);
        assert!(kept2.is_empty());
    }

    #[test]
    fn test_dynamic_lambda() {
        let n = 80;
        let d = 4;
        let emb = random_embeddings(n, d, 7);
        let labels = (0..n as u32).map(|i| i % 3).collect::<Vec<_>>();
        let s = build_state(&emb, n, d, labels, 3);
        let kept: Vec<usize> = (0..s.layout.b).collect();
        let lam = build_dynamic_lambda(&s, 0, &kept);
        assert_eq!(lam.len(), kept.len() + 1);
        assert_eq!(lam[0], 0.0);
        for (j, &gb) in kept.iter().enumerate() {
            let expected = s.config.alpha * s.e[gb];
            assert!((lam[j + 1] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_single_iteration_decreases_objective() {
        let n_per = 80;
        let d = 5;
        let (emb, labels) = batched_gaussian(n_per, d, 13);
        let n = emb.len() / d;
        let cov = BatchCovariate {
            labels,
            n_levels: 2,
            name: None,
        };
        let config = HarmonyConfig {
            n_clusters: Some(4),
            max_iter: 1,
            max_iter_kmeans: 4,
            random_state: 13,
            ..Default::default()
        };
        let mut state = HarmonyState::new(&emb, n, d, &[cov], &config).unwrap();
        // Record objective before any update and after one k-means sub-loop.
        let obj0 = state.compute_objective();
        state.update_r();
        let obj1 = state.compute_objective();
        assert!(
            obj1 <= obj0 + 1e-6,
            "objective did not decrease: obj0={} obj1={}",
            obj0,
            obj1
        );
    }

    #[test]
    fn test_determinism_same_seed() {
        let n = 200;
        let d = 5;
        let emb = random_embeddings(n, d, 21);
        let labels: Vec<u32> = (0..n as u32).map(|i| i % 3).collect();
        let cov = BatchCovariate {
            labels: labels.clone(),
            n_levels: 3,
            name: None,
        };
        let config = HarmonyConfig {
            n_clusters: Some(5),
            max_iter: 2,
            random_state: 99,
            ..Default::default()
        };
        let r1 = harmony_integrate(&emb, n, d, std::slice::from_ref(&cov), &config).unwrap();
        let r2 = harmony_integrate(&emb, n, d, std::slice::from_ref(&cov), &config).unwrap();
        assert_eq!(r1.z_corrected.len(), r2.z_corrected.len());
        for (a, b) in r1.z_corrected.iter().zip(r2.z_corrected.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "z_corrected diverged");
        }
    }

    #[test]
    fn test_multi_covariate_runs() {
        let n = 300;
        let d = 6;
        let emb = random_embeddings(n, d, 33);
        let labels_a: Vec<u32> = (0..n as u32).map(|i| i % 3).collect();
        let labels_b: Vec<u32> = (0..n as u32).map(|i| (i / 3) % 4).collect();
        let cov_a = BatchCovariate {
            labels: labels_a,
            n_levels: 3,
            name: Some("donor".into()),
        };
        let cov_b = BatchCovariate {
            labels: labels_b,
            n_levels: 4,
            name: Some("tech".into()),
        };
        let config = HarmonyConfig {
            n_clusters: Some(6),
            max_iter: 2,
            random_state: 55,
            ..Default::default()
        };
        let result = harmony_integrate(&emb, n, d, &[cov_a, cov_b], &config).unwrap();
        assert_eq!(result.n_obs, n);
        assert_eq!(result.n_pcs, d);
        assert_eq!(result.z_corrected.len(), n * d);
        assert!(result.z_corrected.iter().all(|v| v.is_finite()));
        assert_eq!(result.r_matrix.len(), result.n_clusters * n);
    }

    // ── GPU tests ─────────────────────────────────────────────────────
    // Skip silently on machines without a CUDA driver; run otherwise.

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_harmony_shape_matches_cpu() {
        // Skip if no GPU available.
        if scx_gpu::GpuDevice::new(0).is_err() {
            eprintln!("CUDA not available — skipping GPU harmony test");
            return;
        }
        let n = 200;
        let d = 6;
        let emb = random_embeddings(n, d, 100);
        let labels: Vec<u32> = (0..n as u32).map(|i| i % 3).collect();
        let cov = BatchCovariate {
            labels,
            n_levels: 3,
            name: None,
        };
        let config = HarmonyConfig {
            n_clusters: Some(5),
            max_iter: 2,
            random_state: 7,
            ..Default::default()
        };
        let result =
            harmony_integrate_gpu(&emb, n, d, std::slice::from_ref(&cov), &config).unwrap();
        assert_eq!(result.n_obs, n);
        assert_eq!(result.n_pcs, d);
        assert_eq!(result.z_corrected.len(), n * d);
        assert!(result.z_corrected.iter().all(|v| v.is_finite()));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_vs_cpu_per_pc_correlation() {
        // Skip on machines without CUDA.
        if scx_gpu::GpuDevice::new(0).is_err() {
            eprintln!("CUDA not available — skipping GPU harmony correlation test");
            return;
        }
        let n_per = 80;
        let d = 5;
        let (emb, labels) = batched_gaussian(n_per, d, 123);
        let n = emb.len() / d;
        let cov = BatchCovariate {
            labels,
            n_levels: 2,
            name: None,
        };
        let config = HarmonyConfig {
            n_clusters: Some(4),
            max_iter: 3,
            random_state: 11,
            ..Default::default()
        };
        let cpu = harmony_integrate(&emb, n, d, std::slice::from_ref(&cov), &config).unwrap();
        let gpu = harmony_integrate_gpu(&emb, n, d, std::slice::from_ref(&cov), &config).unwrap();

        // Compare per-PC Pearson correlation (f32 rounding → expect ~0.99+).
        for pc in 0..d {
            let mut x: Vec<f64> = Vec::with_capacity(n);
            let mut y: Vec<f64> = Vec::with_capacity(n);
            for i in 0..n {
                x.push(cpu.z_corrected[i * d + pc]);
                y.push(gpu.z_corrected[i * d + pc]);
            }
            let mx: f64 = x.iter().sum::<f64>() / n as f64;
            let my: f64 = y.iter().sum::<f64>() / n as f64;
            let mut num = 0f64;
            let mut dx = 0f64;
            let mut dy = 0f64;
            for i in 0..n {
                let a = x[i] - mx;
                let b = y[i] - my;
                num += a * b;
                dx += a * a;
                dy += b * b;
            }
            let r = num / (dx.sqrt() * dy.sqrt() + 1e-30);
            // Either strong correlation OR both PCs are near-constant (dx or dy ~ 0).
            if dx > 1e-8 && dy > 1e-8 {
                assert!(r > 0.95, "PC {pc}: r={r}");
            }
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_memory_estimate_reasonable() {
        use scx_gpu::gpu_harmony_memory_bytes;
        let bytes = gpu_harmony_memory_bytes(10_000, 30, 50, 3, 1);
        // Order-of-magnitude: ~few MB, well under 1 GB.
        assert!(bytes > 1_000_000);
        assert!(bytes < 1_000_000_000);
    }
}
