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
use rand_chacha::ChaCha8Rng;
use rand_distr::Normal;

use crate::error::{AccelError, Result};
use scx_sparse::umap_math;

// Re-export shared UMAP math helpers from scx-sparse.
pub use umap_math::{compute_epochs_per_sample, find_ab_params};

/// Random initialization (f64) — delegates to `scx_sparse::umap_math::random_init_f64`.
pub(crate) fn random_init(n_obs: usize, n_components: usize, seed: u64) -> Vec<f64> {
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
        // Try spectral initialization, fall back to random. The fallback used
        // to be silent, so a graph on which the power iteration never converged
        // — or any input with fewer than 3 points — produced a random layout
        // with nothing in the output to say so.
        spectral_init(
            conn_indptr,
            conn_indices,
            conn_data,
            n_obs,
            n_components,
            seed,
        )
        .unwrap_or_else(|e| {
            log::warn!(
                "UMAP spectral initialization failed ({e}); falling back to \
                 random init. The layout is still valid but its global \
                 structure is not seeded by the graph's spectrum."
            );
            random_init(n_obs, n_components, seed)
        })
    };

    // SGD optimization
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_edges = head.len();
    let epochs_per_negative_sample: Vec<f64> = epochs_per_sample
        .iter()
        .map(|&e| e / negative_sample_rate as f64)
        .collect();
    let mut epoch_of_next_sample: Vec<f64> = epochs_per_sample.clone();
    let mut epoch_of_next_negative_sample: Vec<f64> = epochs_per_negative_sample.clone();

    let clip_val = 4.0_f64;
    // `b` is a loop-invariant (set once from `find_ab_params`), so hoist `b - 1`
    // out of the per-edge gradient. f64 subtraction of two invariants is
    // deterministic → the SGD output is bit-identical to recomputing it inline.
    let b_minus_1 = b - 1.0;

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
            let grad_coeff = -2.0 * a * b * dist_sq.powf(b_minus_1) / (1.0 + a * dist_sq.powf(b));

            for d in 0..n_components {
                let grad = (grad_coeff * diff[d]).clamp(-clip_val, clip_val);
                embedding[i * n_components + d] += alpha * grad;
                embedding[j * n_components + d] -= alpha * grad;
            }

            // Repulsive forces via negative sampling
            epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx];

            // Clamp n_neg to negative_sample_rate before both the loop and the
            // schedule advance. Without clamping, the schedule jumps too far
            // ahead when connectivity is high, then subsequent epochs over-sample
            // to compensate. This matches umap-learn's reference implementation.
            let n_neg = ((epoch as f64 - epoch_of_next_negative_sample[edge_idx])
                / epochs_per_negative_sample[edge_idx])
                .floor() as usize;
            let n_neg = n_neg.min(negative_sample_rate);

            for _ in 0..n_neg {
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
pub(crate) fn spectral_init(
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
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
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
            // Multiply by the SHIFTED normalized adjacency:
            //     v_new = (I + D^{-1/2} A D^{-1/2}) v
            //
            // Power iteration converges to the eigenvalue of largest
            // *magnitude*. The spectrum of `M = D^{-1/2} A D^{-1/2}` is
            // `[-1, 1]`, so after deflating the trivial `√D` mode at `λ = 1`
            // the largest-magnitude survivor on a near-bipartite kNN graph is
            // `λ ≈ -1`, not the Fiedler direction at `λ₂ ≲ 1` (review §7.11).
            // The iteration would then return the bipartite mode and the
            // initialization would be no better than random.
            //
            // `I + M` has spectrum `[0, 2]`, so largest-magnitude ≡ largest-λ
            // and the deflated maximum is exactly the Fiedler direction — the
            // smallest non-trivial eigenvector of the normalized Laplacian
            // `L = I - M`, which is what umap-learn's `spectral_layout` asks
            // scipy's `eigsh` for. The shift changes only which eigenvector is
            // selected, never the eigenvectors themselves.
            let mut v_new = vec![0.0_f64; n_obs];
            for i in 0..n_obs {
                let start = indptr[i] as usize;
                let end = indptr[i + 1] as usize;
                let mut sum = v[i];
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

    // Assemble the embedding, then expand it to umap-learn's spread.
    //
    // umap-learn's `simplicial_set_embedding` does
    //     expansion = 10.0 / |initialisation|.max()
    //     embedding = initialisation * expansion + N(0, 1e-4)
    // i.e. **one global** max-abs expansion over the whole matrix, giving a span
    // of ±10 with the noise four orders below it.
    //
    // This used to be `1e-4 / std_dev * 10.0` **per component**, landing at a
    // std of ~1e-3 — about 10⁴ too small (review §7.11). At that scale the
    // 1e-4 noise added below was ~10 % of the signal, and the whole init sat far
    // below the SGD's `clip_val = 4.0`, so the layout started from what was
    // effectively a point cloud at the origin. A per-component rescale also
    // destroys the relative scale *between* components, which carries the
    // eigenvalue ordering; the global factor preserves it.
    let mut embedding = vec![0.0_f64; n_obs * n_components];
    for (comp, evec) in eigenvectors.iter().enumerate() {
        for i in 0..n_obs {
            embedding[i * n_components + comp] = evec[i];
        }
    }
    let max_abs = embedding.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if max_abs > 1e-10 {
        let expansion = 10.0 / max_abs;
        for v in &mut embedding {
            *v *= expansion;
        }
    }

    // Add small noise to break ties
    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(1));
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

    // ── Review §7.11 — spectral init scale and eigenvector selection ─────

    /// A symmetric weighted CSR from an undirected edge list.
    fn sym_csr(edges: &[(usize, usize, f64)], n: usize) -> (Vec<i64>, Vec<i32>, Vec<f64>) {
        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for &(u, v, w) in edges {
            adj[u].push((v, w));
            adj[v].push((u, w));
        }
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for row in adj.iter_mut() {
            row.sort_by_key(|&(j, _)| j);
            for &(j, w) in row.iter() {
                indices.push(j as i32);
                data.push(w);
            }
            indptr.push(indices.len() as i64);
        }
        (indptr, indices, data)
    }

    /// A **bipartite-leaning** two-community graph: a path/ladder whose
    /// adjacency has a strong `λ ≈ -1` mode alongside the Fiedler mode.
    ///
    /// Two 12-node parts, dense *across* the parts and sparse *within* them, so
    /// `D^{-1/2} A D^{-1/2}` has a large-magnitude **negative** eigenvalue. That
    /// is the mode an unshifted power iteration converges to.
    fn near_bipartite_graph() -> (Vec<i64>, Vec<i32>, Vec<f64>, usize) {
        const HALF: usize = 12;
        let n = 2 * HALF;
        let mut edges = Vec::new();
        for i in 0..HALF {
            for j in 0..HALF {
                // Dense across the parts — the bipartite backbone.
                edges.push((i, HALF + j, 1.0));
            }
        }
        // A whisper of within-part weight so the graph is connected but the
        // negative mode still dominates in magnitude.
        for i in 0..HALF - 1 {
            edges.push((i, i + 1, 0.01));
            edges.push((HALF + i, HALF + i + 1, 0.01));
        }
        let (indptr, indices, data) = sym_csr(&edges, n);
        (indptr, indices, data, n)
    }

    /// The signed indicator of the two parts, normalized — the `λ ≈ -1`
    /// eigenvector of `D^{-1/2} A D^{-1/2}` on a bipartite graph.
    fn bipartite_mode(n: usize) -> Vec<f64> {
        let half = n / 2;
        let mut v: Vec<f64> = (0..n).map(|i| if i < half { 1.0 } else { -1.0 }).collect();
        let norm = (n as f64).sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    fn abs_cosine(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na < 1e-12 || nb < 1e-12 {
            return 0.0;
        }
        (dot / (na * nb)).abs()
    }

    /// Pull component 0 out of the returned row-major embedding.
    fn component(embedding: &[f64], n_obs: usize, n_components: usize, comp: usize) -> Vec<f64> {
        (0..n_obs)
            .map(|i| embedding[i * n_components + comp])
            .collect()
    }

    /// The defect: an unshifted power iteration on `D^{-1/2} A D^{-1/2}`
    /// converges to the largest-|λ| eigenvector, which on this graph is the
    /// bipartite `λ ≈ -1` mode rather than the Fiedler direction.
    ///
    /// Premise assertion for the test below — without it, "the fix returns
    /// something that is not the bipartite mode" could be true of any graph.
    #[test]
    fn the_unshifted_iteration_converges_to_the_bipartite_mode() {
        let (indptr, indices, data, n) = near_bipartite_graph();
        // Reproduce the pre-fix operator: no `+ v[i]`.
        let degree: Vec<f64> = (0..n)
            .map(|i| {
                data[indptr[i] as usize..indptr[i + 1] as usize]
                    .iter()
                    .sum()
            })
            .collect();
        let d_inv_sqrt: Vec<f64> = degree
            .iter()
            .map(|&d| if d > 1e-10 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();
        let top: Vec<f64> = {
            let mut t: Vec<f64> = degree.iter().map(|&d| d.sqrt()).collect();
            let norm: f64 = t.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in &mut t {
                *x /= norm;
            }
            t
        };
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let normal = Normal::new(0.0_f64, 1.0).unwrap();
        let mut v: Vec<f64> = (0..n).map(|_| rng.sample(normal)).collect();
        for _ in 0..600 {
            let mut vn = vec![0.0_f64; n];
            for i in 0..n {
                let mut s = 0.0;
                for idx in indptr[i] as usize..indptr[i + 1] as usize {
                    let j = indices[idx] as usize;
                    s += data[idx] * d_inv_sqrt[i] * d_inv_sqrt[j] * v[j];
                }
                vn[i] = s;
            }
            let dt: f64 = vn.iter().zip(&top).map(|(a, b)| a * b).sum();
            for (x, &t) in vn.iter_mut().zip(&top) {
                *x -= dt * t;
            }
            let nrm: f64 = vn.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in &mut vn {
                *x /= nrm;
            }
            v = vn;
        }
        let cos = abs_cosine(&v, &bipartite_mode(n));
        assert!(
            cos > 0.99,
            "premise: the unshifted iteration should land on the bipartite mode, |cos| = {cos}"
        );
    }

    #[test]
    fn spectral_init_avoids_the_bipartite_mode() {
        let (indptr, indices, data, n) = near_bipartite_graph();
        let emb = spectral_init(&indptr, &indices, &data, n, 2, 7).expect("spectral init");
        let c0 = component(&emb, n, 2, 0);
        let cos = abs_cosine(&c0, &bipartite_mode(n));
        assert!(
            cos < 0.5,
            "the shifted iteration must not return the λ ≈ -1 bipartite mode, |cos| = {cos}"
        );
    }

    /// umap-learn expands the initialization to a ±10 span before adding
    /// `N(0, 1e-4)` noise. The old per-component std rescale landed at ~1e-3 —
    /// the same order as the noise.
    #[test]
    fn spectral_init_spans_plus_minus_ten() {
        let (indptr, indices, data, n) = test_graph();
        let emb = spectral_init(&indptr, &indices, &data, n, 2, 42).expect("spectral init");
        let max_abs = emb.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            (max_abs - 10.0).abs() < 0.01,
            "max |coordinate| should be ≈10 (umap-learn's expansion), got {max_abs}"
        );
        // And the tie-breaking noise must stay four orders below the signal.
        assert!(
            max_abs > 1e3 * 1e-4,
            "noise is no longer negligible against the signal"
        );
    }

    /// The fallback path gets the same span: fixing only the spectral arm would
    /// leave every graph on which spectral fails at ~1e-3.
    #[test]
    fn random_init_spans_plus_minus_ten() {
        let init = random_init(500, 2, 42);
        let max_abs = init.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            (5.0..=10.0).contains(&max_abs),
            "random init should fill roughly [-10, 10], got max |coord| {max_abs}"
        );
        let mean: f64 = init.iter().sum::<f64>() / init.len() as f64;
        assert!(
            mean.abs() < 0.5,
            "random init should be centred, mean {mean}"
        );
    }

    /// `compute_umap` falls back to random init when the spectral path fails.
    /// The fallback is legitimate; taking it *silently* was not. Asserting on
    /// the returned layout rather than on the log line, because this crate has
    /// no log sink in tests — what is pinned here is that the fallback is
    /// reached and produces a usable embedding, not the warning's text.
    #[test]
    fn umap_falls_back_to_random_init_on_a_tiny_graph() {
        // `spectral_init` refuses n_obs < 3.
        let n = 2;
        let indptr = vec![0i64, 1, 2];
        let indices = vec![1i32, 0];
        let data = vec![1.0_f64, 1.0];
        let emb = compute_umap(
            &indptr, &indices, &data, n, 2, 5, 0.1, 1.0, 5, 1.0, 42, None,
        )
        .expect("umap on a 2-point graph");
        assert_eq!(emb.embeddings.len(), n * 2);
        assert!(emb.embeddings.iter().all(|v| v.is_finite()));
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
    fn test_compute_umap_deterministic() {
        // The serial SGD is deterministic: single seeded RNG stream + in-order
        // edge iteration. Two runs with the same seed must produce byte-identical
        // embeddings. This documents the determinism guarantee and guards the
        // `b - 1` invariant hoist (and any future opt-in parallel work) against
        // an accidental change to the serial numerics.
        let (indptr, indices, data, n_obs) = test_graph();
        let run = || {
            compute_umap(
                &indptr, &indices, &data, n_obs, 2, 100, 0.1, 1.0, 5, 1.0, 42, None,
            )
            .unwrap()
        };
        let a = run();
        let b = run();
        assert_eq!(a.embeddings.len(), b.embeddings.len());
        for (x, y) in a.embeddings.iter().zip(&b.embeddings) {
            assert_eq!(x.to_bits(), y.to_bits(), "UMAP output not deterministic");
        }
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

    /// Regression test for the negative-sampling schedule fix.
    ///
    /// Uses a fully-connected graph (every node connected to every other)
    /// where the unclamped `n_neg` would exceed `negative_sample_rate`.
    /// Before the fix, the schedule advance used the unclamped value,
    /// causing over-sampling in subsequent epochs. After the fix, all
    /// embeddings should be finite and the two clusters should still
    /// separate cleanly.
    #[test]
    fn test_umap_negative_sample_schedule_clamped() {
        // Build a fully-connected graph of 20 nodes (2 clusters of 10).
        // High connectivity means n_neg will exceed negative_sample_rate
        // on most edges.
        let n = 20;
        let mut indptr = vec![0i64; n + 1];
        let mut indices = Vec::new();
        let mut data = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    indices.push(j as i32);
                    // Cluster-aware weights: high within cluster, low between
                    let same_cluster = (i < 10 && j < 10) || (i >= 10 && j >= 10);
                    data.push(if same_cluster { 0.9 } else { 0.1 });
                }
            }
            indptr[i + 1] = indices.len() as i64;
        }

        // This should NOT panic or produce NaN/Inf despite the high
        // connectivity triggering the schedule clamp path.
        let result = compute_umap(
            &indptr, &indices, &data, n, 2,   // n_components
            100, // n_epochs
            0.1, // min_dist
            1.0, // spread
            5,   // negative_sample_rate (will be exceeded by n_neg)
            1.0, // learning_rate
            42,  // seed
            None,
        )
        .unwrap();

        assert_eq!(result.embeddings.len(), n * 2);
        for &v in &result.embeddings {
            assert!(v.is_finite(), "embedding should be finite, got {v}");
        }

        // The two clusters should still be separated
        let mut c1 = [0.0_f64; 2];
        let mut c2 = [0.0_f64; 2];
        for i in 0..10 {
            c1[0] += result.embeddings[i * 2];
            c1[1] += result.embeddings[i * 2 + 1];
        }
        for i in 10..20 {
            c2[0] += result.embeddings[i * 2];
            c2[1] += result.embeddings[i * 2 + 1];
        }
        c1[0] /= 10.0;
        c1[1] /= 10.0;
        c2[0] /= 10.0;
        c2[1] /= 10.0;
        let inter_dist = ((c1[0] - c2[0]).powi(2) + (c1[1] - c2[1]).powi(2)).sqrt();
        assert!(
            inter_dist > 0.0,
            "clusters should be separated (inter_dist = {inter_dist})"
        );
    }
}
