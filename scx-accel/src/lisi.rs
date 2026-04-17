//! Local Inverse Simpson Index (LISI) — local batch / label diversity.
//!
//! LISI = 1 / Σ_c p_c², where p_c is the Gaussian-kernel-weighted local
//! probability mass of category c in each cell's neighbourhood. Values
//! approach 1 when a cell's neighbours share a single category (poor
//! mixing) and approach the number of categories when neighbours are
//! uniformly distributed across categories (good mixing).
//!
//! Matches the R `lisi` reference package (Korsunsky et al., 2019):
//! * exact kNN by Euclidean distance (NOT HNSW, to stay numerically in
//!   lockstep with `FNN::get.knn`),
//! * per-cell beta (Gaussian bandwidth) calibrated via binary search to
//!   hit a target `perplexity` (identical to t-SNE's H-target routine),
//! * Simpson index over kernel-weighted neighbour probabilities.

use rayon::prelude::*;

use crate::error::{AccelError, Result};

// ─── Public types ─────────────────────────────────────────────────────

/// Configuration for LISI computation.
#[derive(Debug, Clone)]
pub struct LisiConfig {
    /// Target perplexity for the Gaussian kernel (default 30.0).
    pub perplexity: f64,
    /// Number of neighbours to retrieve (default: `3 * perplexity`).
    pub n_neighbors: usize,
    /// Binary-search tolerance on log-perplexity (default 1e-5).
    pub tol: f64,
    /// Binary-search iteration cap (default 200).
    pub max_iter: usize,
}

impl Default for LisiConfig {
    fn default() -> Self {
        Self {
            perplexity: 30.0,
            n_neighbors: 90, // 3 * perplexity
            tol: 1e-5,
            max_iter: 200,
        }
    }
}

/// Result of LISI computation.
#[derive(Debug, Clone)]
pub struct LisiResult {
    /// LISI value per cell, length N.
    pub lisi: Vec<f64>,
    /// Mean LISI across all cells.
    pub mean_lisi: f64,
    /// Median LISI across all cells.
    pub median_lisi: f64,
}

// ─── Public entry point ──────────────────────────────────────────────

/// Compute LISI per cell for a single categorical label vector.
///
/// * `embeddings` — `N x d` row-major f32 (typically PCA output).
/// * `labels` — per-cell category labels, contiguous integers
///   `0..n_categories`.
pub fn compute_lisi(
    embeddings: &[f32],
    n_obs: usize,
    n_dims: usize,
    labels: &[u32],
    config: &LisiConfig,
) -> Result<LisiResult> {
    // --- Validation ---
    if n_obs == 0 || n_dims == 0 {
        return Err(AccelError::InvalidInput(
            "n_obs and n_dims must be > 0".into(),
        ));
    }
    if embeddings.len() != n_obs * n_dims {
        return Err(AccelError::InvalidInput(format!(
            "embeddings length {} != n_obs * n_dims = {}",
            embeddings.len(),
            n_obs * n_dims
        )));
    }
    if labels.len() != n_obs {
        return Err(AccelError::InvalidInput(format!(
            "labels length {} != n_obs {}",
            labels.len(),
            n_obs
        )));
    }
    if config.perplexity <= 0.0 {
        return Err(AccelError::InvalidInput("perplexity must be > 0".into()));
    }
    let k = config.n_neighbors.max(1);
    if k >= n_obs {
        return Err(AccelError::InvalidInput(format!(
            "n_neighbors ({k}) must be < n_obs ({n_obs})"
        )));
    }

    // Widen to f64 once; the per-cell O(N * d) distance sweep dominates
    // runtime, and f64 avoids a second pass for the perplexity search.
    let emb64: Vec<f64> = embeddings.iter().map(|&v| v as f64).collect();

    // --- Exact kNN ---
    let (knn_idx, knn_dist) = exact_knn(&emb64, n_obs, n_dims, k);

    // --- Per-cell LISI (parallel) ---
    let target_logu = config.perplexity.ln();
    let lisi: Vec<f64> = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            let neigh_idx = &knn_idx[i * k..(i + 1) * k];
            let neigh_dist = &knn_dist[i * k..(i + 1) * k];
            let weights = hbeta_weights(neigh_dist, target_logu, config.tol, config.max_iter);
            simpson_inverse(&weights, neigh_idx, labels)
        })
        .collect();

    let (mean, median) = summarise(&lisi);
    Ok(LisiResult {
        lisi,
        mean_lisi: mean,
        median_lisi: median,
    })
}

// ─── Exact brute-force kNN ───────────────────────────────────────────

/// Brute-force exact kNN by squared Euclidean distance.
///
/// Returns `(idx, dist)` both of length `N * k` in row-major order: row i
/// holds the k nearest neighbour indices (excluding i itself) and their
/// non-squared Euclidean distances, sorted ascending.
///
/// `emb` is row-major `N x d` f64.
fn exact_knn(emb: &[f64], n: usize, d: usize, k: usize) -> (Vec<usize>, Vec<f64>) {
    // Precompute per-row squared norms so the distance reduces to
    //   ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b
    let norms: Vec<f64> = (0..n)
        .map(|i| {
            let row = &emb[i * d..(i + 1) * d];
            row.iter().map(|v| v * v).sum()
        })
        .collect();

    let mut knn_idx = vec![0usize; n * k];
    let mut knn_dist = vec![0f64; n * k];

    // Parallelise over query cells. Each thread keeps its own top-k heap.
    knn_idx
        .par_chunks_mut(k)
        .zip(knn_dist.par_chunks_mut(k))
        .enumerate()
        .for_each(|(i, (idx_row, dist_row))| {
            let row_i = &emb[i * d..(i + 1) * d];
            let norm_i = norms[i];

            // Max-heap keyed by squared distance so we can evict the
            // worst candidate in O(log k).
            use std::cmp::Ordering;
            use std::collections::BinaryHeap;

            #[derive(PartialEq)]
            struct Entry {
                dist_sq: f64,
                idx: usize,
            }
            impl Eq for Entry {}
            impl PartialOrd for Entry {
                fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                    Some(self.cmp(other))
                }
            }
            impl Ord for Entry {
                fn cmp(&self, other: &Self) -> Ordering {
                    // BinaryHeap is max-heap; we want smallest distances
                    // at the top for easy pop-worst. Partial-order by
                    // dist_sq ascending → flip compare so greatest sits
                    // at top of heap.
                    self.dist_sq
                        .partial_cmp(&other.dist_sq)
                        .unwrap_or(Ordering::Equal)
                }
            }

            let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(k + 1);

            for j in 0..n {
                if j == i {
                    continue;
                }
                let row_j = &emb[j * d..(j + 1) * d];
                let mut dot = 0f64;
                for t in 0..d {
                    dot += row_i[t] * row_j[t];
                }
                // Guard numerical negatives (|a-b|^2 should be ≥ 0 but
                // catastrophic cancellation can push slightly below 0).
                let dist_sq = (norm_i + norms[j] - 2.0 * dot).max(0.0);

                if heap.len() < k {
                    heap.push(Entry { dist_sq, idx: j });
                } else if let Some(top) = heap.peek() {
                    if dist_sq < top.dist_sq {
                        heap.pop();
                        heap.push(Entry { dist_sq, idx: j });
                    }
                }
            }

            // Drain heap in descending distance, then reverse.
            let mut pairs: Vec<(f64, usize)> =
                heap.into_iter().map(|e| (e.dist_sq, e.idx)).collect();
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

            for (slot, (dsq, jdx)) in pairs.into_iter().enumerate() {
                idx_row[slot] = jdx;
                dist_row[slot] = dsq.sqrt();
            }
        });

    (knn_idx, knn_dist)
}

// ─── Per-cell beta calibration (H-target binary search) ──────────────

/// Calibrate a Gaussian bandwidth (beta = 1/(2 σ²)) via binary search so
/// that the entropy of the kernel-weighted neighbour distribution equals
/// `log(perplexity)`. Returns the normalised probabilities.
///
/// This is the standard t-SNE `Hbeta` routine reused by the R `lisi`
/// reference package. Distances are non-squared Euclidean; we square
/// them once on entry so the inner loop only has to multiply by beta.
fn hbeta_weights(dists: &[f64], target_logu: f64, tol: f64, max_iter: usize) -> Vec<f64> {
    let k = dists.len();
    let d2: Vec<f64> = dists.iter().map(|&d| d * d).collect();

    let mut beta = 1.0f64;
    let mut beta_min = f64::NEG_INFINITY;
    let mut beta_max = f64::INFINITY;

    // Scratch buffer for exponentiated (and normalised) probabilities.
    let mut p = vec![0f64; k];

    // `hbeta_at` evaluates P and H for a given beta; both the R lisi
    // reference and t-SNE compute H = log(sum_p) + beta * sum(D² * p_un)
    // / sum_p_un, which is the entropy of the *normalised* distribution
    // before dividing out.
    fn hbeta_at(d2: &[f64], beta: f64, p: &mut [f64]) -> f64 {
        let mut sum_p = 0f64;
        for (i, &d) in d2.iter().enumerate() {
            let v = (-d * beta).exp();
            p[i] = v;
            sum_p += v;
        }
        if sum_p <= 0.0 {
            // Degenerate: assign uniform so downstream Simpson behaves.
            let u = 1.0 / p.len() as f64;
            for v in p.iter_mut() {
                *v = u;
            }
            return 0.0;
        }
        let mut sum_dp = 0f64;
        for (i, &d) in d2.iter().enumerate() {
            sum_dp += d * p[i];
        }
        let h = sum_p.ln() + beta * sum_dp / sum_p;
        // Normalise in place.
        let inv = 1.0 / sum_p;
        for v in p.iter_mut() {
            *v *= inv;
        }
        h
    }

    let mut h = hbeta_at(&d2, beta, &mut p);
    let mut h_diff = h - target_logu;

    let mut iter = 0usize;
    while h_diff.abs() > tol && iter < max_iter {
        if h_diff > 0.0 {
            // Entropy too high → increase beta (tighter kernel).
            beta_min = beta;
            beta = if beta_max.is_infinite() {
                beta * 2.0
            } else {
                (beta + beta_max) / 2.0
            };
        } else {
            beta_max = beta;
            beta = if beta_min.is_infinite() {
                beta / 2.0
            } else {
                (beta + beta_min) / 2.0
            };
        }
        h = hbeta_at(&d2, beta, &mut p);
        h_diff = h - target_logu;
        iter += 1;
    }

    p
}

// ─── Simpson index ───────────────────────────────────────────────────

/// Compute LISI = 1 / Σ_c p_c², where p_c is the kernel-weighted
/// probability mass of category c across the cell's neighbours.
///
/// `weights` sum to 1 (post-hbeta). `neigh_idx` indexes into `labels`.
fn simpson_inverse(weights: &[f64], neigh_idx: &[usize], labels: &[u32]) -> f64 {
    // Accumulate per-category probabilities using a small hashmap; since
    // labels are contiguous u32 we could use a Vec, but k is typically
    // ≤ 200 so the extra branching + zeroing would cost more than a
    // tiny hashmap for common inputs. Use a Vec when the dense path
    // obviously wins (# distinct labels ≈ k).
    use std::collections::HashMap;
    let mut probs: HashMap<u32, f64> = HashMap::with_capacity(weights.len());
    for (w, &j) in weights.iter().zip(neigh_idx.iter()) {
        *probs.entry(labels[j]).or_insert(0.0) += *w;
    }
    let mut s = 0f64;
    for &p in probs.values() {
        s += p * p;
    }
    if s > 0.0 {
        1.0 / s
    } else {
        1.0
    }
}

// ─── Summary stats ───────────────────────────────────────────────────

fn summarise(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        let mid = sorted.len() / 2;
        (sorted[mid - 1] + sorted[mid]) / 2.0
    };
    (mean, median)
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    /// Random (N x d) f32 embeddings, standard normal-ish.
    fn random_emb(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n * d).map(|_| rng.gen::<f32>() - 0.5).collect()
    }

    #[test]
    fn test_lisi_single_label_is_one() {
        let n = 100;
        let d = 5;
        let emb = random_emb(n, d, 7);
        let labels = vec![0u32; n];
        let config = LisiConfig {
            perplexity: 10.0,
            n_neighbors: 30,
            ..Default::default()
        };
        let result = compute_lisi(&emb, n, d, &labels, &config).unwrap();
        for &v in &result.lisi {
            assert!((v - 1.0).abs() < 1e-6, "expected LISI=1, got {v}");
        }
    }

    #[test]
    fn test_lisi_uniform_mix_approaches_n_categories() {
        // Random embeddings with labels drawn iid uniform from C
        // categories. Neighbours should be roughly uniformly labelled,
        // so mean LISI should approach C.
        let n = 400;
        let d = 5;
        let c = 3;
        let emb = random_emb(n, d, 11);
        let mut rng = ChaCha8Rng::seed_from_u64(12);
        let labels: Vec<u32> = (0..n).map(|_| rng.gen_range(0..c)).collect();
        let config = LisiConfig {
            perplexity: 30.0,
            n_neighbors: 90,
            ..Default::default()
        };
        let result = compute_lisi(&emb, n, d, &labels, &config).unwrap();
        // Mean LISI should be within ~0.5 of the number of categories.
        // Finite-sample noise keeps it from hitting exactly C.
        assert!(
            (result.mean_lisi - c as f64).abs() < 0.5,
            "mean LISI = {} (expected near {c})",
            result.mean_lisi
        );
        // All values should lie in [1, C + epsilon].
        for &v in &result.lisi {
            assert!(v >= 1.0 - 1e-6 && v <= c as f64 + 1e-3, "value {v}");
        }
    }

    #[test]
    fn test_hbeta_calibration_converges() {
        // Synthetic distance vector; verify the search reaches the target
        // perplexity within the requested tolerance.
        let dists: Vec<f64> = (0..50).map(|i| (i as f64 + 1.0).sqrt()).collect();
        let target = 30.0_f64.ln();
        let p = hbeta_weights(&dists, target, 1e-6, 200);
        // Probabilities must sum to 1 and be non-negative.
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-10, "sum {s}");
        assert!(p.iter().all(|&v| v >= 0.0));
        // Effective entropy ≈ target.
        let h: f64 = -p
            .iter()
            .filter(|&&v| v > 0.0)
            .map(|&v| v * v.ln())
            .sum::<f64>();
        assert!((h - target).abs() < 1e-3, "H={h} target={target}");
    }

    #[test]
    fn test_exact_knn_basic() {
        // 4 cells in 2D: (0,0), (1,0), (3,0), (10,0).
        // For cell 0, nearest in order should be cell 1, then 2, then 3.
        let emb: Vec<f64> = vec![0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 10.0, 0.0];
        let (idx, dist) = exact_knn(&emb, 4, 2, 3);
        assert_eq!(&idx[0..3], &[1, 2, 3]);
        assert!((dist[0] - 1.0).abs() < 1e-10);
        assert!((dist[1] - 3.0).abs() < 1e-10);
        assert!((dist[2] - 10.0).abs() < 1e-10);
    }
}
