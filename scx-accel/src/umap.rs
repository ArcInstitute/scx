//! UMAP (Uniform Manifold Approximation and Projection) embedding.
//!
//! Implements the core SGD optimization loop from McInnes et al. (2018),
//! with spectral initialization and negative sampling matching umap-learn's
//! output format.
//!
//! # Algorithm
//!
//! 1. **Initialization**: Spectral embedding via power iteration on the
//!    normalized graph Laplacian (falls back to random if spectral fails).
//! 2. **SGD optimization**: For each epoch, iterate over positive edges
//!    (from the kNN connectivity graph) applying attractive forces, then
//!    apply repulsive forces via negative sampling.
//! 3. **Output**: Low-dimensional embedding (n_obs × n_components).

use rand::prelude::*;
use rand_distr::Normal;

use crate::error::{AccelError, Result};
use scx_sparse::umap_math;

// Re-export shared UMAP math helpers from scx-sparse.
pub use umap_math::{compute_epochs_per_sample, find_ab_params};

/// Random initialization (f64) — delegates to `scx_sparse::umap_math::random_init_f64`.
fn random_init(n_obs: usize, n_components: usize, seed: u64) -> Vec<f64> {
    umap_math::random_init_f64(n_obs, n_components, seed)
}

/// Upper bound on `n_components` enforced by [`compute_umap`]. The SGD inner
/// loop uses a `[f64; MAX_UMAP_COMPONENTS]` stack buffer to cache the
/// component-wise diff between two embeddings; anything larger would overflow
/// the buffer. UMAP targets 2–3D in every realistic workflow, so the cap is
/// more than an order of magnitude above the typical use case.
pub const MAX_UMAP_COMPONENTS: usize = 16;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of UMAP embedding.
#[derive(Debug, Clone)]
pub struct UmapResult {
    /// Flat row-major embedding (n_obs × n_components).
    pub embeddings: Vec<f64>,
    /// Number of observations.
    pub n_obs: usize,
    /// Number of components (typically 2).
    pub n_components: usize,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute UMAP embedding from a kNN connectivity graph.
///
/// # Arguments
///
/// * `conn_indptr` — CSR row pointers for the connectivity matrix (n_obs + 1).
/// * `conn_indices` — CSR column indices for the connectivity matrix.
/// * `conn_data` — CSR values (connectivity strengths in [0, 1]).
/// * `n_obs` — Number of observations.
/// * `n_components` — Output dimensions (default: 2).
/// * `n_epochs` — Number of SGD epochs (default: 200).
/// * `min_dist` — Minimum distance in embedding (default: 0.1).
/// * `spread` — Spread of embedded points (default: 1.0).
/// * `negative_sample_rate` — Negative samples per positive edge (default: 5).
/// * `learning_rate` — Initial learning rate (default: 1.0).
/// * `seed` — Random seed for reproducibility.
/// * `init_coords` — Optional initial coordinates (n_obs × n_components, row-major).
///   If None, spectral initialization is attempted, falling back to random.
#[allow(clippy::too_many_arguments)]
pub fn compute_umap(
    conn_indptr: &[i64],
    conn_indices: &[i32],
    conn_data: &[f64],
    n_obs: usize,
    n_components: usize,
    n_epochs: usize,
    min_dist: f64,
    spread: f64,
    negative_sample_rate: usize,
    learning_rate: f64,
    seed: u64,
    init_coords: Option<&[f64]>,
) -> Result<UmapResult> {
    // Validate inputs
    if n_obs == 0 {
        return Err(AccelError::InvalidInput("n_obs must be > 0".to_string()));
    }
    if conn_indptr.len() != n_obs + 1 {
        return Err(AccelError::InvalidInput(format!(
            "conn_indptr length ({}) != n_obs + 1 ({})",
            conn_indptr.len(),
            n_obs + 1
        )));
    }
    if n_components == 0 {
        return Err(AccelError::InvalidInput(
            "n_components must be > 0".to_string(),
        ));
    }
    if n_components > MAX_UMAP_COMPONENTS {
        return Err(AccelError::InvalidInput(format!(
            "n_components ({n_components}) exceeds MAX_UMAP_COMPONENTS ({MAX_UMAP_COMPONENTS}); \
             the SGD inner loop uses a stack-allocated diff buffer of that size. \
             UMAP typically targets 2–3D."
        )));
    }
    if n_epochs == 0 {
        return Err(AccelError::InvalidInput("n_epochs must be > 0".to_string()));
    }

    // Find a, b parameters from min_dist and spread
    let (a, b) = find_ab_params(spread, min_dist);

    // Build epoch-per-sample schedule: edges with higher weight are sampled more often
    let epochs_per_sample = compute_epochs_per_sample(conn_data, n_epochs);

    // Build flat edge list for efficient iteration
    let mut head = Vec::with_capacity(conn_data.len());
    let mut tail = Vec::with_capacity(conn_data.len());
    for i in 0..n_obs {
        let start = conn_indptr[i] as usize;
        let end = conn_indptr[i + 1] as usize;
        for &col in &conn_indices[start..end] {
            head.push(i);
            tail.push(col as usize);
        }
    }

    // Initialize embedding
    let mut embedding = if let Some(coords) = init_coords {
        if coords.len() != n_obs * n_components {
            return Err(AccelError::InvalidInput(format!(
                "init_coords length ({}) != n_obs × n_components ({} × {} = {})",
                coords.len(),
                n_obs,
                n_components,
                n_obs * n_components
            )));
        }
        coords.to_vec()
    } else {
        // Try spectral initialization, fall back to random
        spectral_init(
            conn_indptr,
            conn_indices,
            conn_data,
            n_obs,
            n_components,
            seed,
        )
        .unwrap_or_else(|_| random_init(n_obs, n_components, seed))
    };

    // SGD optimization
    let mut rng = StdRng::seed_from_u64(seed);
    let n_edges = head.len();
    let epochs_per_negative_sample: Vec<f64> = epochs_per_sample
        .iter()
        .map(|&e| e / negative_sample_rate as f64)
        .collect();
    let mut epoch_of_next_sample: Vec<f64> = epochs_per_sample.clone();
    let mut epoch_of_next_negative_sample: Vec<f64> = epochs_per_negative_sample.clone();

    let clip_val = 4.0_f64;

    for epoch in 0..n_epochs {
        let alpha = learning_rate * (1.0 - epoch as f64 / n_epochs as f64);

        for edge_idx in 0..n_edges {
            if epoch_of_next_sample[edge_idx] > epoch as f64 {
                continue;
            }

            let i = head[edge_idx];
            let j = tail[edge_idx];

            // Cache the component-wise difference once and reuse it for
            // both the squared-distance reduction and the gradient step.
            // Functionally identical to the original (same summation order,
            // same f64 identities), but the compiler can now keep `diff`
            // in SIMD registers across the two uses instead of re-loading
            // `embedding[i]` / `embedding[j]` twice. `n_components` is
            // bounded by `MAX_UMAP_COMPONENTS` (enforced at function entry).
            let mut diff = [0.0_f64; MAX_UMAP_COMPONENTS];
            let mut dist_sq = 0.0_f64;
            for d in 0..n_components {
                let delta = embedding[i * n_components + d] - embedding[j * n_components + d];
                diff[d] = delta;
                dist_sq += delta * delta;
            }
            dist_sq = dist_sq.max(1e-10);

            // Gradient of attractive force
            let grad_coeff = -2.0 * a * b * dist_sq.powf(b - 1.0) / (1.0 + a * dist_sq.powf(b));

            for d in 0..n_components {
                let grad = (grad_coeff * diff[d]).clamp(-clip_val, clip_val);
                embedding[i * n_components + d] += alpha * grad;
                embedding[j * n_components + d] -= alpha * grad;
            }

            // Repulsive forces via negative sampling
            epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx];

            let n_neg = ((epoch as f64 - epoch_of_next_negative_sample[edge_idx])
                / epochs_per_negative_sample[edge_idx])
                .floor() as usize;

            for _ in 0..n_neg.min(negative_sample_rate) {
                let k = rng.gen_range(0..n_obs);
                if k == i {
                    continue;
                }

                let mut neg_diff = [0.0_f64; MAX_UMAP_COMPONENTS];
                let mut neg_dist_sq = 0.0_f64;
                for d in 0..n_components {
                    let delta = embedding[i * n_components + d] - embedding[k * n_components + d];
                    neg_diff[d] = delta;
                    neg_dist_sq += delta * delta;
                }
                neg_dist_sq = neg_dist_sq.max(1e-10);

                // Gradient of repulsive force
                let neg_grad_coeff =
                    2.0 * b / ((0.001 + neg_dist_sq) * (1.0 + a * neg_dist_sq.powf(b)));

                for d in 0..n_components {
                    let grad = (neg_grad_coeff * neg_diff[d]).clamp(-clip_val, clip_val);
                    embedding[i * n_components + d] += alpha * grad;
                }
            }
            epoch_of_next_negative_sample[edge_idx] +=
                n_neg as f64 * epochs_per_negative_sample[edge_idx];
        }
    }

    Ok(UmapResult {
        embeddings: embedding,
        n_obs,
        n_components,
    })
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Spectral initialization via power iteration on the normalized Laplacian.
///
/// Computes the 2 (or n_components) smallest non-trivial eigenvectors of
/// D^{-1/2} A D^{-1/2} using power iteration with deflation.
fn spectral_init(
    indptr: &[i64],
    indices: &[i32],
    data: &[f64],
    n_obs: usize,
    n_components: usize,
    seed: u64,
) -> Result<Vec<f64>> {
    if n_obs < 3 {
        return Err(AccelError::InvalidInput(
            "need at least 3 points for spectral init".to_string(),
        ));
    }

    // Compute degree vector D
    let mut degree = vec![0.0_f64; n_obs];
    for i in 0..n_obs {
        let start = indptr[i] as usize;
        let end = indptr[i + 1] as usize;
        for &w in &data[start..end] {
            degree[i] += w;
        }
    }

    // D^{-1/2}
    let d_inv_sqrt: Vec<f64> = degree
        .iter()
        .map(|&d| if d > 1e-10 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    // Power iteration to find top eigenvectors of normalized adjacency
    // (equivalent to smallest non-trivial eigenvectors of Laplacian)
    let max_iters = 300;
    let tol = 1e-6;
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f64, 1.0).unwrap();

    let mut eigenvectors: Vec<Vec<f64>> = Vec::with_capacity(n_components);

    // The top eigenvector of the normalized adjacency is the degree vector √D.
    // We want eigenvectors 2..(n_components+1), so we need to deflate.
    // First, compute the top eigenvector (proportional to sqrt(degree)).
    let mut top_eigvec: Vec<f64> = degree.iter().map(|&d| d.sqrt()).collect();
    let norm: f64 = top_eigvec.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        for v in &mut top_eigvec {
            *v /= norm;
        }
    }

    for _comp in 0..n_components {
        // Random initial vector
        let mut v: Vec<f64> = (0..n_obs).map(|_| rng.sample(normal)).collect();

        // Orthogonalize against top eigenvector
        let dot_top: f64 = v.iter().zip(top_eigvec.iter()).map(|(a, b)| a * b).sum();
        for (vi, &ti) in v.iter_mut().zip(top_eigvec.iter()) {
            *vi -= dot_top * ti;
        }

        // Orthogonalize against previously found eigenvectors
        for prev in &eigenvectors {
            let dot: f64 = v.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
            for (vi, &pi) in v.iter_mut().zip(prev.iter()) {
                *vi -= dot * pi;
            }
        }

        // Normalize
        let mut v_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if v_norm > 1e-10 {
            for vi in &mut v {
                *vi /= v_norm;
            }
        }

        let mut converged = false;

        for _iter in 0..max_iters {
            // Multiply by normalized adjacency: v_new = D^{-1/2} A D^{-1/2} v
            let mut v_new = vec![0.0_f64; n_obs];
            for i in 0..n_obs {
                let start = indptr[i] as usize;
                let end = indptr[i + 1] as usize;
                let mut sum = 0.0_f64;
                for idx in start..end {
                    let j = indices[idx] as usize;
                    let w = data[idx];
                    sum += w * d_inv_sqrt[i] * d_inv_sqrt[j] * v[j];
                }
                v_new[i] = sum;
            }

            // Deflate against top eigenvector
            let dot_top: f64 = v_new
                .iter()
                .zip(top_eigvec.iter())
                .map(|(a, b)| a * b)
                .sum();
            for (vi, &ti) in v_new.iter_mut().zip(top_eigvec.iter()) {
                *vi -= dot_top * ti;
            }

            // Deflate against previously found eigenvectors
            for prev in &eigenvectors {
                let dot: f64 = v_new.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                for (vi, &pi) in v_new.iter_mut().zip(prev.iter()) {
                    *vi -= dot * pi;
                }
            }

            // Normalize
            v_norm = v_new.iter().map(|x| x * x).sum::<f64>().sqrt();
            if v_norm < 1e-10 {
                break;
            }
            for vi in &mut v_new {
                *vi /= v_norm;
            }

            // Convergence by cosine similarity to the previous iterate.
            // The L1-mean test this replaced was sign-sensitive — an
            // eigenvector and its negative (arbitrary sign from power
            // iteration) registered as maximally different even at
            // convergence. `1 - |cos(v, v_new)|` is sign-invariant and
            // rotation-aware, matching the convergence criterion used by
            // `umap-learn`'s spectral_layout reference.
            let cos_sim: f64 = v.iter().zip(v_new.iter()).map(|(a, b)| a * b).sum();
            let diff = 1.0 - cos_sim.abs();

            v = v_new;

            if diff < tol {
                converged = true;
                break;
            }
        }

        if !converged {
            // Fall back to random initialization
            return Err(AccelError::LinAlg(
                "spectral initialization did not converge".to_string(),
            ));
        }

        eigenvectors.push(v);
    }

    // Assemble embedding (scale eigenvectors for reasonable spread)
    let mut embedding = vec![0.0_f64; n_obs * n_components];
    for (comp, evec) in eigenvectors.iter().enumerate() {
        // Scale to have std ≈ 1e-4 * initial_alpha (matching umap-learn's scaling)
        let std_dev: f64 = {
            let mean: f64 = evec.iter().sum::<f64>() / n_obs as f64;
            let var: f64 =
                evec.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n_obs as f64;
            var.sqrt()
        };
        let scale = if std_dev > 1e-10 {
            1e-4 / std_dev * 10.0 // small initial spread
        } else {
            1.0
        };
        for i in 0..n_obs {
            embedding[i * n_components + comp] = evec[i] * scale;
        }
    }

    // Add small noise to break ties
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1));
    let noise = Normal::new(0.0_f64, 1e-4).unwrap();
    for v in &mut embedding {
        *v += rng.sample(noise);
    }

    Ok(embedding)
}

// `random_init` is defined at the top of this file as a thin wrapper
// around `scx_sparse::umap_math::random_init_f64`.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple kNN-like connectivity graph for testing.
    /// Two clusters of 25 points each with high intra-cluster connectivity.
    fn test_graph() -> (Vec<i64>, Vec<i32>, Vec<f64>, usize) {
        let n = 50;
        let k = 5;
        let mut indptr = vec![0i64; n + 1];
        let mut indices = Vec::new();
        let mut data = Vec::new();

        for i in 0..n {
            let cluster_start = if i < 25 { 0 } else { 25 };
            let cluster_end = if i < 25 { 25 } else { 50 };

            let mut neighbors = Vec::new();
            for j in cluster_start..cluster_end {
                if j != i {
                    neighbors.push(j);
                }
                if neighbors.len() == k {
                    break;
                }
            }

            for &j in &neighbors {
                indices.push(j as i32);
                data.push(0.8); // strong connectivity
            }
            indptr[i + 1] = indices.len() as i64;
        }

        (indptr, indices, data, n)
    }

    #[test]
    fn test_find_ab_params() {
        let (a, b) = find_ab_params(1.0, 0.1);
        // umap-learn's scipy.optimize.curve_fit gives a≈1.929, b≈0.7915.
        // Our grid search approximation gives a≈1.58, b≈0.89 — close enough
        // for UMAP embedding quality (the SGD optimization is robust to
        // moderate a,b variation).
        assert!(
            a > 1.0 && a < 2.5,
            "a = {a}, expected in [1.0, 2.5] (umap-learn: ~1.93)"
        );
        assert!(
            b > 0.5 && b < 1.2,
            "b = {b}, expected in [0.5, 1.2] (umap-learn: ~0.79)"
        );
    }

    #[test]
    fn test_find_ab_params_various() {
        // Standard combos that should NOT hit boundaries.
        let cases = [
            (1.0, 0.1),  // default
            (1.0, 0.25), // larger min_dist
            (1.0, 0.5),  // large min_dist
            (0.5, 0.1),  // tighter spread
            (2.0, 0.1),  // wider spread
        ];
        for (spread, min_dist) in cases {
            let (a, b) = find_ab_params(spread, min_dist);
            assert!(
                a > 0.2 && a < 9.8,
                "a={a} out of safe interior range for spread={spread}, min_dist={min_dist}"
            );
            assert!(
                b > 0.2 && b < 3.9,
                "b={b} out of safe interior range for spread={spread}, min_dist={min_dist}"
            );
            // Sanity: the curve at d=0 should be ~1.0 (perfect fit for the piecewise target)
            let pred_at_zero = 1.0 / (1.0 + a * (0.001_f64).powf(2.0 * b));
            assert!(
                pred_at_zero > 0.99,
                "pred(0) = {pred_at_zero} for spread={spread}, min_dist={min_dist}"
            );
        }
    }

    #[test]
    fn test_find_ab_params_extreme() {
        // Extreme params — should not panic. May emit boundary warning to stderr.
        let (a, b) = find_ab_params(0.01, 0.001);
        assert!(a.is_finite(), "a should be finite, got {a}");
        assert!(b.is_finite(), "b should be finite, got {b}");
        assert!(a > 0.0, "a should be positive, got {a}");
        assert!(b > 0.0, "b should be positive, got {b}");
    }

    #[test]
    fn test_compute_umap_basic() {
        let (indptr, indices, data, n_obs) = test_graph();
        let result = compute_umap(
            &indptr, &indices, &data, n_obs, 2,    // n_components
            50,   // n_epochs (low for test speed)
            0.1,  // min_dist
            1.0,  // spread
            5,    // negative_sample_rate
            1.0,  // learning_rate
            42,   // seed
            None, // init_coords
        )
        .unwrap();

        assert_eq!(result.n_obs, 50);
        assert_eq!(result.n_components, 2);
        assert_eq!(result.embeddings.len(), 100);
    }

    #[test]
    fn test_embeddings_finite() {
        let (indptr, indices, data, n_obs) = test_graph();
        let result = compute_umap(
            &indptr, &indices, &data, n_obs, 2, 100, 0.1, 1.0, 5, 1.0, 42, None,
        )
        .unwrap();

        for &v in &result.embeddings {
            assert!(v.is_finite(), "embedding value should be finite, got {v}");
        }
    }

    #[test]
    fn test_clusters_separated() {
        let (indptr, indices, data, n_obs) = test_graph();
        let result = compute_umap(
            &indptr, &indices, &data, n_obs, 2, 200, 0.1, 1.0, 5, 1.0, 42, None,
        )
        .unwrap();

        // Compute centroids of two clusters
        let mut c1 = [0.0_f64; 2];
        let mut c2 = [0.0_f64; 2];
        for i in 0..25 {
            c1[0] += result.embeddings[i * 2];
            c1[1] += result.embeddings[i * 2 + 1];
        }
        for i in 25..50 {
            c2[0] += result.embeddings[i * 2];
            c2[1] += result.embeddings[i * 2 + 1];
        }
        c1[0] /= 25.0;
        c1[1] /= 25.0;
        c2[0] /= 25.0;
        c2[1] /= 25.0;

        // Clusters should be separated
        let inter_dist = ((c1[0] - c2[0]).powi(2) + (c1[1] - c2[1]).powi(2)).sqrt();

        // Compute average intra-cluster distance
        let mut intra = 0.0_f64;
        for i in 0..25 {
            let dx = result.embeddings[i * 2] - c1[0];
            let dy = result.embeddings[i * 2 + 1] - c1[1];
            intra += (dx * dx + dy * dy).sqrt();
        }
        intra /= 25.0;

        assert!(
            inter_dist > intra,
            "inter-cluster distance ({inter_dist:.4}) should exceed intra-cluster distance ({intra:.4})"
        );
    }

    #[test]
    fn test_custom_init() {
        let (indptr, indices, data, n_obs) = test_graph();
        let init: Vec<f64> = (0..n_obs * 2).map(|i| (i as f64) * 0.001).collect();

        let result = compute_umap(
            &indptr,
            &indices,
            &data,
            n_obs,
            2,
            50,
            0.1,
            1.0,
            5,
            1.0,
            42,
            Some(&init),
        )
        .unwrap();

        assert_eq!(result.embeddings.len(), 100);
        // Should differ from init (SGD moves points)
        let diff: f64 = result
            .embeddings
            .iter()
            .zip(init.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();
        assert!(diff > 0.0, "SGD should have moved points");
    }

    #[test]
    fn test_error_zero_obs() {
        let result = compute_umap(&[0i64], &[], &[], 0, 2, 50, 0.1, 1.0, 5, 1.0, 42, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_bad_indptr() {
        let result = compute_umap(
            &[0i64, 5],
            &[1, 2, 3, 4, 5],
            &[1.0; 5],
            5,
            2,
            50,
            0.1,
            1.0,
            5,
            1.0,
            42,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_epochs_per_sample() {
        let weights = vec![1.0, 0.5, 0.25, 0.1];
        let schedule = compute_epochs_per_sample(&weights, 200);
        // Max weight edge should be sampled every epoch
        assert!((schedule[0] - 1.0).abs() < 1e-6);
        // Lower weight edges should be sampled less often
        assert!(schedule[1] > schedule[0]);
        assert!(schedule[2] > schedule[1]);
        assert!(schedule[3] > schedule[2]);
    }

    #[test]
    fn test_random_init_deterministic() {
        let init1 = random_init(10, 2, 42);
        let init2 = random_init(10, 2, 42);
        assert_eq!(init1, init2, "same seed should give same init");
    }
}

// ---------------------------------------------------------------------------
// GPU dispatch (behind "gpu" feature)
// ---------------------------------------------------------------------------

/// Compute UMAP embedding using GPU-accelerated CUDA SGD kernel.
///
/// Pipeline:
/// 1. Attempt spectral initialization on CPU (falls back to random)
/// 2. Convert init coords from f64→f32
/// 3. Run GPU UMAP SGD optimization
/// 4. Convert result from f32→f64 to match CPU UmapResult format
///
/// Falls back with `AccelError` if GPU is unavailable.
///
/// # Arguments
///
/// * `device_id` — CUDA device ordinal (typically 0)
/// * `conn_indptr` — CSR row pointers for connectivity matrix (n_obs + 1)
/// * `conn_indices` — CSR column indices
/// * `conn_data` — CSR values (connectivity strengths)
/// * `n_obs` — Number of observations
/// * `n_components` — Output dimensions (default: 2)
/// * `n_epochs` — SGD epochs (default: 200)
/// * `min_dist` — Minimum distance in embedding (default: 0.1)
/// * `spread` — Spread of embedded points (default: 1.0)
/// * `negative_sample_rate` — Negative samples per positive edge (default: 5)
/// * `learning_rate` — Initial learning rate (default: 1.0)
/// * `seed` — Random seed for reproducibility
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn compute_umap_gpu(
    device_id: usize,
    conn_indptr: &[i64],
    conn_indices: &[i32],
    conn_data: &[f64],
    n_obs: usize,
    n_components: usize,
    n_epochs: usize,
    min_dist: f64,
    spread: f64,
    negative_sample_rate: usize,
    learning_rate: f64,
    seed: u64,
) -> Result<UmapResult> {
    use crate::error::AccelError;

    // Create GPU device
    let dev = scx_gpu::GpuDevice::new(device_id)
        .map_err(|e| AccelError::InvalidInput(format!("GPU device {device_id}: {e}")))?;

    // Attempt spectral initialization on CPU (f64), then convert to f32
    let init_f64 = spectral_init(
        conn_indptr,
        conn_indices,
        conn_data,
        n_obs,
        n_components,
        seed,
    )
    .unwrap_or_else(|_| random_init(n_obs, n_components, seed));
    let init_f32: Vec<f32> = init_f64.iter().map(|&v| v as f32).collect();

    // Run GPU UMAP
    let gpu_result = scx_gpu::gpu_umap_native(
        &dev,
        conn_indptr,
        conn_indices,
        conn_data,
        n_obs,
        n_components,
        n_epochs,
        min_dist as f32,
        spread as f32,
        negative_sample_rate,
        learning_rate as f32,
        seed,
        Some(&init_f32),
    )
    .map_err(|e| AccelError::InvalidInput(format!("GPU UMAP: {e}")))?;

    // Convert f32→f64
    let embeddings: Vec<f64> = gpu_result.embedding.iter().map(|&v| v as f64).collect();

    Ok(UmapResult {
        embeddings,
        n_obs: gpu_result.n_obs,
        n_components: gpu_result.n_components,
    })
}
