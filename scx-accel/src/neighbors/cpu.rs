//! kNN graph construction via HNSW (Hierarchical Navigable Small World).
//!
//! Uses [`instant_distance`] for approximate nearest neighbor search,
//! then computes UMAP-style fuzzy set connectivities matching scanpy's
//! `sc.pp.neighbors()` output format.
//!
//! # Algorithm
//!
//! 1. Build HNSW index from dense PCA embeddings (n_obs × n_pcs)
//! 2. Query k nearest neighbors for each point
//! 3. Compute fuzzy set connectivities: for each point, find σ such that
//!    `Σ exp(-max(d - d_nearest, 0) / σ) = log2(k)`, then symmetrize
//! 4. Return distances and connectivities as sparse CSR triplets

use faer::linalg::matmul::matmul;
use faer::{Mat, MatRef};
use instant_distance::{Hnsw, Point, Search};
use rayon::prelude::*;

use crate::error::{AccelError, Result};

/// kNN backend dispatch threshold: at or below this `n_obs`, `build_knn_graph`
/// runs an exact-kNN gemm path (faer matmul + per-row partial top-k sort)
/// instead of HNSW. The same matmul-vs-tree trade-off as `edistance`: at
/// small `n_obs` the matmul wins because (a) HNSW build is
/// O(n × ef_construction × n_dims) with scalar inner products in
/// `instant-distance`, while (b) the gemm runs at AVX2/AVX-512 GEMM
/// throughput. Above ~5K rows HNSW's asymptotic edge starts to dominate.
/// Tuned against the Replogle clustering_agreement regression
/// (n_obs ≈ 2.4K, n_dims ≈ 18K).
const EXACT_KNN_NOBS_THRESHOLD: usize = 5_000;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of kNN graph construction.
#[derive(Debug, Clone)]
pub struct KnnResult {
    /// Neighbor indices: flat row-major (n_obs × n_neighbors).
    pub indices: Vec<usize>,
    /// Neighbor distances: flat row-major (n_obs × n_neighbors).
    pub distances: Vec<f64>,
    /// Connectivities CSR: row pointers (n_obs + 1).
    pub conn_indptr: Vec<i64>,
    /// Connectivities CSR: column indices.
    pub conn_indices: Vec<i32>,
    /// Connectivities CSR: values.
    pub conn_data: Vec<f64>,
    /// Distance CSR: row pointers (n_obs + 1).
    pub dist_indptr: Vec<i64>,
    /// Distance CSR: column indices.
    pub dist_indices: Vec<i32>,
    /// Distance CSR: values.
    pub dist_data: Vec<f64>,
    /// Number of neighbors.
    pub n_neighbors: usize,
    /// Number of observations.
    pub n_obs: usize,
}

// ---------------------------------------------------------------------------
// Point type for instant-distance
// ---------------------------------------------------------------------------

/// A dense point in Euclidean space for HNSW indexing.
#[derive(Clone, Debug)]
struct EuclideanPoint(Vec<f32>);

impl Point for EuclideanPoint {
    fn distance(&self, other: &Self) -> f32 {
        // Squared Euclidean distance (instant-distance uses this as the metric;
        // the ordering is preserved by the monotonic sqrt transformation).
        debug_assert_eq!(self.0.len(), other.0.len());
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(&a, &b)| {
                let d = a - b;
                d * d
            })
            .sum::<f32>()
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build a kNN graph from dense row-major data.
///
/// # Arguments
///
/// * `data` — Row-major dense matrix (n_obs × n_vars), f32.
/// * `n_obs` — Number of observations (rows).
/// * `n_vars` — Number of variables (columns / PCA dimensions).
/// * `n_neighbors` — Number of nearest neighbors (k).
/// * `ef_construction` — HNSW construction parameter (higher = more accurate, slower build).
/// * `ef_search` — HNSW search parameter (higher = more accurate, slower search).
/// * `seed` — Random seed for reproducibility.
pub fn build_knn_graph(
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    n_neighbors: usize,
    ef_construction: usize,
    ef_search: usize,
    seed: u64,
) -> Result<KnnResult> {
    // Validate inputs
    if n_obs == 0 || n_vars == 0 {
        return Err(AccelError::InvalidInput(
            "data must be non-empty".to_string(),
        ));
    }
    if data.len() != n_obs * n_vars {
        return Err(AccelError::InvalidInput(format!(
            "data length ({}) does not match n_obs × n_vars ({} × {} = {})",
            data.len(),
            n_obs,
            n_vars,
            n_obs * n_vars
        )));
    }
    if n_neighbors == 0 {
        return Err(AccelError::InvalidInput(
            "n_neighbors must be > 0".to_string(),
        ));
    }
    if n_neighbors > n_obs {
        return Err(AccelError::InvalidInput(format!(
            "n_neighbors ({n_neighbors}) exceeds n_obs ({n_obs})"
        )));
    }

    // Dispatch: exact gemm-based kNN at small `n_obs`, HNSW above the
    // threshold (see `EXACT_KNN_NOBS_THRESHOLD`). Both branches produce
    // the same `(indices, distances)` flat arrays of length
    // `n_obs * n_neighbors`, so the downstream CSR + connectivity
    // assembly is shared.
    let (indices, distances) = if n_obs <= EXACT_KNN_NOBS_THRESHOLD {
        log::debug!("build_knn_graph: exact path (n_obs={n_obs} <= {EXACT_KNN_NOBS_THRESHOLD})");
        build_knn_exact(data, n_obs, n_vars, n_neighbors)
    } else {
        log::debug!("build_knn_graph: HNSW path (n_obs={n_obs} > {EXACT_KNN_NOBS_THRESHOLD})");
        build_knn_hnsw(
            data,
            n_obs,
            n_vars,
            n_neighbors,
            ef_construction,
            ef_search,
            seed,
        )
    };

    // Build distance CSR matrix
    let (dist_indptr, dist_indices, dist_data) =
        build_knn_csr(&indices, &distances, n_obs, n_neighbors);

    // Compute UMAP-style connectivities
    let (conn_indptr, conn_indices, conn_data) =
        compute_connectivities(&indices, &distances, n_obs, n_neighbors);

    Ok(KnnResult {
        indices,
        distances,
        conn_indptr,
        conn_indices,
        conn_data,
        dist_indptr,
        dist_indices,
        dist_data,
        n_neighbors,
        n_obs,
    })
}

/// HNSW kNN — the original path, factored out behind the dispatch in
/// `build_knn_graph`. Returns flat `(indices, distances)` arrays of length
/// `n_obs * n_neighbors` (row-major).
fn build_knn_hnsw(
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    n_neighbors: usize,
    ef_construction: usize,
    ef_search: usize,
    seed: u64,
) -> (Vec<usize>, Vec<f64>) {
    // Build a single Vec of points, hand it to HNSW (which consumes it),
    // then regenerate per-query points from the caller's `data` slice
    // during the parallel search. Peak memory is ~1× the embedding, down
    // from 2× (the previous `query_points.clone()` held a second copy
    // resident for the entire search phase).
    let build_points: Vec<EuclideanPoint> = (0..n_obs)
        .map(|i| {
            let start = i * n_vars;
            EuclideanPoint(data[start..start + n_vars].to_vec())
        })
        .collect();

    // Build HNSW index
    // build_hnsw() returns (Hnsw, Vec<PointId>) where the Vec maps
    // original_index -> internal PointId (points are shuffled internally)
    let (hnsw, point_ids) = Hnsw::<EuclideanPoint>::builder()
        .ef_construction(ef_construction)
        .ef_search(ef_search)
        .seed(seed)
        .build_hnsw(build_points);

    // Build reverse mapping: internal PointId.0 -> original index
    let mut internal_to_original = vec![0usize; n_obs];
    for (original_idx, pid) in point_ids.iter().enumerate() {
        internal_to_original[pid.into_inner() as usize] = original_idx;
    }

    // Query k-nearest neighbors for each point in parallel. Each thread
    // synthesizes its query point from `data` on the fly — no duplicated
    // Vec<EuclideanPoint> needs to stay resident.
    let all_results: Vec<Vec<(usize, f64)>> = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            let start = i * n_vars;
            let query = EuclideanPoint(data[start..start + n_vars].to_vec());
            let mut search = Search::default();

            // Query k+1 neighbors since the point itself will be in the results
            hnsw.search(&query, &mut search)
                .take(n_neighbors + 1)
                .filter_map(|item| {
                    let original_idx = internal_to_original[item.pid.into_inner() as usize];
                    if original_idx == i {
                        None // skip self
                    } else {
                        // instant-distance returns squared Euclidean distance;
                        // take sqrt to get actual Euclidean distance
                        let dist = (item.distance as f64).sqrt();
                        Some((original_idx, dist))
                    }
                })
                .take(n_neighbors)
                .collect()
        })
        .collect();

    // Flatten into flat arrays
    let mut indices = vec![0usize; n_obs * n_neighbors];
    let mut distances = vec![0.0f64; n_obs * n_neighbors];

    for (i, neighbors) in all_results.iter().enumerate() {
        for (j, &(idx, dist)) in neighbors.iter().enumerate() {
            if j < n_neighbors {
                indices[i * n_neighbors + j] = idx;
                distances[i * n_neighbors + j] = dist;
            }
        }
        // If fewer neighbors found (shouldn't happen for reasonable ef),
        // fill with self-references at distance 0
        for j in neighbors.len()..n_neighbors {
            indices[i * n_neighbors + j] = i;
            distances[i * n_neighbors + j] = 0.0;
        }
    }

    (indices, distances)
}

/// Exact kNN via faer gemm + per-row partial top-k sort.
///
/// Builds the full `n_obs × n_obs` Gram matrix `G = X · Xᵀ` in one matmul,
/// then expands each row to squared distances `‖xᵢ‖² + ‖xⱼ‖² − 2·G[i,j]`,
/// and uses `select_nth_unstable_by` to extract the `n_neighbors` smallest
/// (i ≠ j) per row. Distances are returned in ascending order to match the
/// HNSW path's ordering convention (which `compute_connectivities` relies
/// on for `rho` = nearest-neighbor distance).
///
/// Used for `n_obs ≤ EXACT_KNN_NOBS_THRESHOLD`. At small `n_obs` the matmul
/// wins because the inner work runs at AVX-GEMM throughput while
/// `instant-distance`'s HNSW inner loops are scalar.
///
/// Returns flat `(indices, distances)` arrays of length `n_obs * n_neighbors`
/// (row-major), matching `build_knn_hnsw` exactly.
fn build_knn_exact(
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    n_neighbors: usize,
) -> (Vec<usize>, Vec<f64>) {
    // Gram matrix in f32 (matches the input precision; final distance
    // expansion widens to f64 to match the HNSW path's f64 output).
    let x_ref = MatRef::<f32>::from_row_major_slice(data, n_obs, n_vars);
    let mut gram = Mat::<f32>::zeros(n_obs, n_obs);
    matmul(
        gram.as_mut(),
        faer::Accum::Replace,
        x_ref,
        x_ref.transpose(),
        1.0_f32,
        faer::Par::rayon(0),
    );

    // Squared row norms in f64 (widened from f32 to keep precision tight on
    // long rows — the same Phase 1 pattern used in `eval_metrics::distances`).
    let row_norm_sq: Vec<f64> = (0..n_obs)
        .into_par_iter()
        .with_min_len(64)
        .map(|i| {
            let row = &data[i * n_vars..(i + 1) * n_vars];
            row.iter()
                .map(|&x| {
                    let xf = x as f64;
                    xf * xf
                })
                .sum::<f64>()
        })
        .collect();

    // Per-row top-k extraction. For each row `i`, build a `Vec<(j, d²_ij)>`
    // for `j ≠ i`, partition with `select_nth_unstable_by` so the smallest
    // `n_neighbors` distances land in the prefix, then sort the prefix
    // ascending so `compute_connectivities` sees the same ordering as the
    // HNSW path (which iterates the search results in distance-ascending
    // order via `take(n_neighbors + 1)`).
    //
    // Memory access: `gram` is column-major (faer default) and `G = X·Xᵀ`
    // is symmetric, so reading `gram[(j, i)]` for fixed `i` and varying `j`
    // walks down column `i` — contiguous in memory — instead of striding
    // across rows. At n_obs ≈ 5K the gram is ~100 MB (>> L2), so contiguous
    // access is a measurable win over the symmetric `gram[(i, j)]`.
    let per_row: Vec<Vec<(usize, f64)>> = (0..n_obs)
        .into_par_iter()
        .with_min_len(8)
        .map(|i| {
            let mut candidates: Vec<(usize, f64)> = Vec::with_capacity(n_obs - 1);
            let ai_sq = row_norm_sq[i];
            for j in 0..n_obs {
                if j == i {
                    continue;
                }
                // gram[(j, i)] == gram[(i, j)] (symmetric); the (j, i) form
                // walks contiguously down column `i` in faer's column-major
                // layout.
                let g = gram[(j, i)] as f64;
                // max(0, ·) clamps tiny negatives from FP cancellation on
                // near-identical rows. Same guard as `pairwise_gemm_row_sums`.
                let d_sq = (ai_sq + row_norm_sq[j] - 2.0 * g).max(0.0);
                candidates.push((j, d_sq));
            }
            // Partial sort: smallest `n_neighbors` distances land in
            // `candidates[..n_neighbors]` (unordered), the rest are
            // discarded. This is O(n) vs O(n log n) for a full sort.
            if n_neighbors < candidates.len() {
                candidates.select_nth_unstable_by(n_neighbors, |a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                candidates.truncate(n_neighbors);
            }
            // Order the prefix ascending by distance so downstream code
            // (compute_connectivities, sigma binary search) sees the same
            // ordering convention as the HNSW path.
            candidates.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            // Take sqrt at the very end to produce Euclidean distances.
            for slot in candidates.iter_mut() {
                slot.1 = slot.1.sqrt();
            }
            candidates
        })
        .collect();

    // Flatten into `(indices, distances)` row-major arrays.
    let mut indices = vec![0usize; n_obs * n_neighbors];
    let mut distances = vec![0.0f64; n_obs * n_neighbors];
    for (i, row) in per_row.iter().enumerate() {
        for (j, &(idx, dist)) in row.iter().enumerate() {
            indices[i * n_neighbors + j] = idx;
            distances[i * n_neighbors + j] = dist;
        }
    }

    (indices, distances)
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Build a CSR sparse matrix from kNN indices and distances.
///
/// Returns (indptr, indices, data) for an (n_obs × n_obs) sparse matrix
/// where entry (i, j) = distance from point i to neighbor j.
pub(crate) fn build_knn_csr(
    knn_indices: &[usize],
    knn_distances: &[f64],
    n_obs: usize,
    n_neighbors: usize,
) -> (Vec<i64>, Vec<i32>, Vec<f64>) {
    let nnz = n_obs * n_neighbors;
    let mut indptr = Vec::with_capacity(n_obs + 1);
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);

    indptr.push(0i64);
    for i in 0..n_obs {
        for j in 0..n_neighbors {
            let idx = knn_indices[i * n_neighbors + j];
            let dist = knn_distances[i * n_neighbors + j];
            indices.push(idx as i32);
            data.push(dist);
        }
        indptr.push((indices.len()) as i64);
    }

    (indptr, indices, data)
}

/// Compute UMAP-style fuzzy set connectivities from kNN distances.
///
/// For each point i, find bandwidth σ_i such that:
///     Σ_j exp(-max(d(i,j) - d(i,nearest), 0) / σ_i) = log2(n_neighbors)
///
/// Then compute membership strengths and symmetrize via fuzzy set union:
///     conn(i,j) = μ(i,j) + μ(j,i) - μ(i,j) * μ(j,i)
///
/// Returns CSR triplets (indptr, indices, data) for the symmetrized connectivity matrix.
pub(crate) fn compute_connectivities(
    knn_indices: &[usize],
    knn_distances: &[f64],
    n_obs: usize,
    n_neighbors: usize,
) -> (Vec<i64>, Vec<i32>, Vec<f64>) {
    let target = (n_neighbors as f64).ln() / std::f64::consts::LN_2; // log2(k)

    // Compute per-point bandwidths (σ). Each point is independent (its own
    // distance slice + ρ) and `find_sigma` is a deterministic fixed-iteration
    // binary search, so the parallel map is order-preserving and bit-identical
    // to the serial loop.
    let sigmas: Vec<f64> = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            let offset = i * n_neighbors;
            let rho = knn_distances[offset]; // nearest neighbor distance
            find_sigma(&knn_distances[offset..offset + n_neighbors], rho, target)
        })
        .collect();

    // Build the directed membership matrix A in CSR form (row i → μ(i,j)), with
    // each row's columns sorted ascending (the kNN list is distance-ordered, not
    // column-ordered, so sort per row). This replaces the previous
    // HashMap<(i,j)> + HashSet symmetrization with an allocation-light,
    // deterministic CSR transpose + sorted-merge.
    let mut a_indptr = Vec::with_capacity(n_obs + 1);
    a_indptr.push(0usize);
    let mut a_cols: Vec<usize> = Vec::with_capacity(n_obs * n_neighbors);
    let mut a_vals: Vec<f64> = Vec::with_capacity(n_obs * n_neighbors);
    let mut row_buf: Vec<(usize, f64)> = Vec::with_capacity(n_neighbors);
    #[allow(clippy::needless_range_loop)] // i indexes sigmas, knn_indices, knn_distances
    for i in 0..n_obs {
        let offset = i * n_neighbors;
        let rho = knn_distances[offset];
        let sigma = sigmas[i];
        row_buf.clear();
        for j_idx in 0..n_neighbors {
            let j = knn_indices[offset + j_idx];
            let d = knn_distances[offset + j_idx];
            let strength = if d <= rho || sigma <= 1e-10 {
                1.0
            } else {
                (-(d - rho) / sigma).exp()
            };
            row_buf.push((j, strength));
        }
        // Ascending by column; on a duplicate neighbor id the later entry wins,
        // matching the prior HashMap insert (last-write-wins).
        row_buf.sort_by_key(|&(col, _)| col);
        let mut prev: Option<usize> = None;
        for &(col, val) in &row_buf {
            if prev == Some(col) {
                *a_vals.last_mut().unwrap() = val;
            } else {
                a_cols.push(col);
                a_vals.push(val);
                prev = Some(col);
            }
        }
        a_indptr.push(a_cols.len());
    }

    // Transpose A → Aᵀ via a counting-scatter (Aᵀ[j] lists, for each row i that
    // has column j, the value μ(i,j) = A[i][j]); scattering rows 0..n_obs in
    // order leaves each Aᵀ row sorted ascending by original row index.
    let nnz = a_cols.len();
    let mut at_indptr = vec![0usize; n_obs + 1];
    for &c in &a_cols {
        at_indptr[c + 1] += 1;
    }
    for c in 0..n_obs {
        at_indptr[c + 1] += at_indptr[c];
    }
    let mut at_rows = vec![0usize; nnz]; // original row index i
    let mut at_vals = vec![0.0f64; nnz];
    let mut cursor = at_indptr[..n_obs].to_vec();
    for i in 0..n_obs {
        for idx in a_indptr[i]..a_indptr[i + 1] {
            let c = a_cols[idx];
            let dst = cursor[c];
            at_rows[dst] = i;
            at_vals[dst] = a_vals[idx];
            cursor[c] += 1;
        }
    }

    // Symmetrize row by row: conn(i,k) = μ(i,k) + μ(k,i) - μ(i,k)·μ(k,i), merging
    // the two column-sorted lists A[i] (μ(i,·)) and Aᵀ[i] (μ(·,i)). Emitting k in
    // ascending order gives a per-row column-sorted CSR; keep conn > 0. This is a
    // fixed 3-term expression per pair, so the result is byte-identical to the
    // previous map-based path (addition/multiplication of the same two operands).
    let mut indptr = Vec::with_capacity(n_obs + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0i64);
    for i in 0..n_obs {
        let (mut ap, ae) = (a_indptr[i], a_indptr[i + 1]);
        let (mut tp, te) = (at_indptr[i], at_indptr[i + 1]);
        while ap < ae || tp < te {
            let a_col = if ap < ae { Some(a_cols[ap]) } else { None };
            let t_col = if tp < te { Some(at_rows[tp]) } else { None };
            let (col, mu_ik, mu_ki) = match (a_col, t_col) {
                (Some(ac), Some(tc)) if ac == tc => {
                    let v = (ac, a_vals[ap], at_vals[tp]);
                    ap += 1;
                    tp += 1;
                    v
                }
                (Some(ac), Some(tc)) if ac < tc => {
                    let v = (ac, a_vals[ap], 0.0);
                    ap += 1;
                    v
                }
                (Some(_), Some(tc)) => {
                    let v = (tc, 0.0, at_vals[tp]);
                    tp += 1;
                    v
                }
                (Some(ac), None) => {
                    let v = (ac, a_vals[ap], 0.0);
                    ap += 1;
                    v
                }
                (None, Some(tc)) => {
                    let v = (tc, 0.0, at_vals[tp]);
                    tp += 1;
                    v
                }
                (None, None) => unreachable!("loop guard guarantees one side is non-empty"),
            };
            let conn = mu_ik + mu_ki - mu_ik * mu_ki;
            if conn > 0.0 {
                indices.push(col as i32);
                data.push(conn);
            }
        }
        indptr.push(indices.len() as i64);
    }

    (indptr, indices, data)
}

/// Binary search for σ such that Σ exp(-max(d - ρ, 0) / σ) = target.
///
/// 50 iterations is enough: starting bounds [1e-10, 1000], the interval
/// width drops below 1e-12 by iteration 50 — tighter than f64's ~52-bit
/// mantissa for typical σ magnitudes. Additional iterations just spin on
/// rounding noise.
fn find_sigma(distances: &[f64], rho: f64, target: f64) -> f64 {
    let mut lo = 1e-10_f64;
    let mut hi = 1000.0_f64;
    let mut mid;

    for _ in 0..50 {
        mid = (lo + hi) / 2.0;

        let sum: f64 = distances
            .iter()
            .map(|&d| {
                let adjusted = (d - rho).max(0.0);
                (-adjusted / mid).exp()
            })
            .sum();

        if sum > target {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    (lo + hi) / 2.0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate simple clustered test data: two clusters in 2D.
    fn two_cluster_data() -> (Vec<f32>, usize, usize) {
        let mut data = Vec::new();
        let n_per_cluster = 25;
        let n_obs = n_per_cluster * 2;
        let n_vars = 3;

        // Cluster 1: centered at (0, 0, 0)
        for i in 0..n_per_cluster {
            data.push(0.1 * (i as f32));
            data.push(0.1 * ((i % 5) as f32));
            data.push(0.05 * (i as f32));
        }

        // Cluster 2: centered at (10, 10, 10)
        for i in 0..n_per_cluster {
            data.push(10.0 + 0.1 * (i as f32));
            data.push(10.0 + 0.1 * ((i % 5) as f32));
            data.push(10.0 + 0.05 * (i as f32));
        }

        (data, n_obs, n_vars)
    }

    #[test]
    fn test_build_knn_basic() {
        let (data, n_obs, n_vars) = two_cluster_data();
        let result = build_knn_graph(&data, n_obs, n_vars, 5, 100, 50, 42).unwrap();

        assert_eq!(result.n_obs, n_obs);
        assert_eq!(result.n_neighbors, 5);
        assert_eq!(result.indices.len(), n_obs * 5);
        assert_eq!(result.distances.len(), n_obs * 5);
    }

    #[test]
    fn test_knn_distances_nonnegative() {
        let (data, n_obs, n_vars) = two_cluster_data();
        let result = build_knn_graph(&data, n_obs, n_vars, 5, 100, 50, 42).unwrap();

        for &d in &result.distances {
            assert!(d >= 0.0, "distance should be non-negative, got {d}");
        }
    }

    #[test]
    fn test_knn_cluster_separation() {
        let (data, n_obs, n_vars) = two_cluster_data();
        let result = build_knn_graph(&data, n_obs, n_vars, 5, 100, 50, 42).unwrap();

        // Points 0..25 are cluster 1, points 25..50 are cluster 2.
        // Each point's neighbors should mostly be from the same cluster.
        for i in 0..25 {
            let same_cluster_count = (0..5).filter(|&j| result.indices[i * 5 + j] < 25).count();
            assert!(
                same_cluster_count >= 4,
                "point {i} has only {same_cluster_count}/5 neighbors in same cluster"
            );
        }
        for i in 25..n_obs {
            let same_cluster_count = (0..5).filter(|&j| result.indices[i * 5 + j] >= 25).count();
            assert!(
                same_cluster_count >= 4,
                "point {i} has only {same_cluster_count}/5 neighbors in same cluster"
            );
        }
    }

    #[test]
    fn test_distance_csr_valid() {
        let (data, n_obs, n_vars) = two_cluster_data();
        let result = build_knn_graph(&data, n_obs, n_vars, 5, 100, 50, 42).unwrap();

        // CSR indptr
        assert_eq!(result.dist_indptr.len(), n_obs + 1);
        assert_eq!(result.dist_indptr[0], 0);
        assert_eq!(
            *result.dist_indptr.last().unwrap(),
            result.dist_data.len() as i64
        );

        // Each row has exactly k entries
        for i in 0..n_obs {
            let nnz = result.dist_indptr[i + 1] - result.dist_indptr[i];
            assert_eq!(nnz, 5, "row {i} has {nnz} entries, expected 5");
        }
    }

    #[test]
    fn test_connectivities_csr_valid() {
        let (data, n_obs, n_vars) = two_cluster_data();
        let result = build_knn_graph(&data, n_obs, n_vars, 5, 100, 50, 42).unwrap();

        // CSR indptr
        assert_eq!(result.conn_indptr.len(), n_obs + 1);
        assert_eq!(result.conn_indptr[0], 0);
        assert_eq!(
            *result.conn_indptr.last().unwrap(),
            result.conn_data.len() as i64
        );

        // All connectivity values in [0, 1]
        for &v in &result.conn_data {
            assert!(
                (0.0..=1.0 + 1e-10).contains(&v),
                "connectivity {v} not in [0,1]"
            );
        }

        // Connectivities have more entries than distances (symmetrized)
        assert!(
            result.conn_data.len() >= result.dist_data.len(),
            "connectivities should have at least as many entries as distances"
        );
    }

    /// Independent dense O(n²) reference for the fuzzy-simplicial-set union.
    /// Reuses `find_sigma` (its own correctness is covered by `test_find_sigma`)
    /// but computes the symmetrization independently of `compute_connectivities`,
    /// so it is a genuine oracle for the CSR-merge refactor (task 2.5a). Produces
    /// the same CSR triplet layout: per-row column-sorted, `conn > 0` filtered.
    fn dense_conn_reference(
        knn_indices: &[usize],
        knn_distances: &[f64],
        n_obs: usize,
        k: usize,
    ) -> (Vec<i64>, Vec<i32>, Vec<f64>) {
        let target = (k as f64).ln() / std::f64::consts::LN_2;
        let sigmas: Vec<f64> = (0..n_obs)
            .map(|i| {
                let off = i * k;
                find_sigma(&knn_distances[off..off + k], knn_distances[off], target)
            })
            .collect();
        // Dense directed membership μ(i,j); last-write-wins per (i,j) like the map.
        let mut mu = vec![0.0f64; n_obs * n_obs];
        for i in 0..n_obs {
            let off = i * k;
            let rho = knn_distances[off];
            let sigma = sigmas[i];
            for jn in 0..k {
                let j = knn_indices[off + jn];
                let d = knn_distances[off + jn];
                let s = if d <= rho || sigma <= 1e-10 {
                    1.0
                } else {
                    (-(d - rho) / sigma).exp()
                };
                mu[i * n_obs + j] = s;
            }
        }
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for i in 0..n_obs {
            for kk in 0..n_obs {
                let a = mu[i * n_obs + kk];
                let b = mu[kk * n_obs + i];
                if a > 0.0 || b > 0.0 {
                    let conn = a + b - a * b;
                    if conn > 0.0 {
                        indices.push(kk as i32);
                        data.push(conn);
                    }
                }
            }
            indptr.push(indices.len() as i64);
        }
        (indptr, indices, data)
    }

    /// Hand-built kNN fixture with self-neighbors (exercises the i==j path) and
    /// an asymmetric edge (2→3 present, 3→2 absent). Distances ascending per row.
    fn asymmetric_knn_fixture() -> (Vec<usize>, Vec<f64>, usize, usize) {
        let n_obs = 6;
        let k = 3;
        let knn_indices = vec![
            0, 1, 2, // 0
            1, 0, 2, // 1
            2, 1, 3, // 2 → 3 (asymmetric)
            3, 4, 5, // 3
            4, 3, 5, // 4
            5, 4, 3, // 5
        ];
        let knn_distances = vec![
            0.0, 1.0, 2.0, // 0
            0.0, 1.0, 3.0, // 1
            0.0, 1.5, 2.0, // 2
            0.0, 1.0, 2.0, // 3
            0.0, 1.0, 1.5, // 4
            0.0, 1.0, 2.0, // 5
        ];
        (knn_indices, knn_distances, n_obs, k)
    }

    #[test]
    fn connectivities_match_dense_reference_fixture() {
        let (idx, dist, n_obs, k) = asymmetric_knn_fixture();
        let (indptr, indices, data) = compute_connectivities(&idx, &dist, n_obs, k);
        let (r_indptr, r_indices, r_data) = dense_conn_reference(&idx, &dist, n_obs, k);
        assert_eq!(indptr, r_indptr, "indptr mismatch vs dense reference");
        assert_eq!(indices, r_indices, "indices mismatch vs dense reference");
        // Byte-identical f64: the symmetrization is a fixed per-pair expression.
        assert_eq!(
            data.len(),
            r_data.len(),
            "data length mismatch vs dense reference"
        );
        for (a, b) in data.iter().zip(&r_data) {
            assert_eq!(a.to_bits(), b.to_bits(), "conn weight {a} != reference {b}");
        }
    }

    #[test]
    fn connectivities_match_dense_reference_realistic() {
        // A realistic kNN from build_knn_graph, to cover typical (asymmetric,
        // no-self) neighbor structure at k=5.
        let (data, n_obs, n_vars) = two_cluster_data();
        let result = build_knn_graph(&data, n_obs, n_vars, 5, 100, 50, 42).unwrap();
        let knn_idx: Vec<usize> = result.indices.iter().map(|&i| i as usize).collect();
        let (indptr, indices, cdata) =
            compute_connectivities(&knn_idx, &result.distances, n_obs, 5);
        let (r_indptr, r_indices, r_data) =
            dense_conn_reference(&knn_idx, &result.distances, n_obs, 5);
        assert_eq!(indptr, r_indptr);
        assert_eq!(indices, r_indices);
        for (a, b) in cdata.iter().zip(&r_data) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn test_error_zero_neighbors() {
        let data = vec![1.0f32; 10];
        assert!(build_knn_graph(&data, 2, 5, 0, 100, 50, 42).is_err());
    }

    #[test]
    fn test_error_too_many_neighbors() {
        let data = vec![1.0f32; 10];
        assert!(build_knn_graph(&data, 2, 5, 5, 100, 50, 42).is_err());
    }

    #[test]
    fn test_error_wrong_data_length() {
        let data = vec![1.0f32; 9]; // should be 10 for 2×5
        assert!(build_knn_graph(&data, 2, 5, 1, 100, 50, 42).is_err());
    }

    #[test]
    fn test_find_sigma() {
        let distances = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let rho = 0.0;
        let target = (5.0_f64).ln() / std::f64::consts::LN_2;

        let sigma = find_sigma(&distances, rho, target);
        assert!(sigma > 0.0, "sigma should be positive");

        // Verify the sum is close to target
        let sum: f64 = distances
            .iter()
            .map(|&d| (-(d - rho).max(0.0) / sigma).exp())
            .sum();
        assert!(
            (sum - target).abs() < 0.01,
            "sum ({sum}) should be close to target ({target})"
        );
    }

    // ── Phase 6 — exact-kNN dispatch tests ───────────────────────────────

    /// LCG-driven deterministic synthetic matrix in [-1, 1].
    fn synth_data(n_obs: usize, n_vars: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..n_obs * n_vars)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 32) as f64 / u32::MAX as f64) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    /// Brute-force exact kNN reference for parity checks: for each row, sort
    /// every `j ≠ i` by Euclidean distance ascending and return the top-k.
    fn brute_force_knn(
        data: &[f32],
        n_obs: usize,
        n_vars: usize,
        k: usize,
    ) -> (Vec<usize>, Vec<f64>) {
        let mut indices = vec![0usize; n_obs * k];
        let mut distances = vec![0.0f64; n_obs * k];
        for i in 0..n_obs {
            let row_i = &data[i * n_vars..(i + 1) * n_vars];
            let mut all: Vec<(usize, f64)> = (0..n_obs)
                .filter(|&j| j != i)
                .map(|j| {
                    let row_j = &data[j * n_vars..(j + 1) * n_vars];
                    let d_sq: f64 = row_i
                        .iter()
                        .zip(row_j.iter())
                        .map(|(&a, &b)| {
                            let d = a as f64 - b as f64;
                            d * d
                        })
                        .sum();
                    (j, d_sq.sqrt())
                })
                .collect();
            all.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for (slot, &(idx, dist)) in all.iter().take(k).enumerate() {
                indices[i * k + slot] = idx;
                distances[i * k + slot] = dist;
            }
        }
        (indices, distances)
    }

    #[test]
    fn test_exact_knn_matches_brute_force() {
        // 1k × 100 synthetic — well below the dispatch threshold so we hit
        // the exact path, and small enough that brute force is also cheap.
        let n_obs = 1_000;
        let n_vars = 100;
        let k = 15;
        let data = synth_data(n_obs, n_vars, 0xA1A2_A3A4);

        let result = build_knn_graph(&data, n_obs, n_vars, k, 200, 50, 42).unwrap();
        let (ref_idx, ref_dist) = brute_force_knn(&data, n_obs, n_vars, k);

        // Every neighbor index should match brute force exactly (the exact
        // path uses select_nth + sort, which is the same partition rule).
        // Tie-break ordering for equidistant neighbors might differ between
        // sort_unstable_by implementations on the same key, so allow a
        // small index-permutation tolerance.
        let mut mismatches = 0;
        for i in 0..n_obs {
            for slot in 0..k {
                let a = result.indices[i * k + slot];
                let b = ref_idx[i * k + slot];
                if a != b {
                    // OK if distances match within FP tolerance — that's a
                    // tie-break disagreement, not a wrong answer.
                    let da = result.distances[i * k + slot];
                    let db = ref_dist[i * k + slot];
                    if (da - db).abs() > 1e-4 {
                        mismatches += 1;
                    }
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "exact path disagreed with brute force on {mismatches} non-tie entries"
        );

        // Distances themselves should match brute force within FP tolerance.
        for i in 0..n_obs * k {
            let diff = (result.distances[i] - ref_dist[i]).abs();
            assert!(
                diff < 1e-4,
                "distance mismatch at flat-idx {i}: exact={} ref={}",
                result.distances[i],
                ref_dist[i],
            );
        }
    }

    #[test]
    fn test_exact_vs_hnsw_recall_on_clusters() {
        // 1k × 100 synthetic with 3 well-separated clusters: HNSW with
        // ef_construction=200 should recover ≥ 95 % of exact kNN's top-k
        // (open question #5 in the spec asks for this). We construct two
        // KnnResults — one via the exact path (n_obs ≤ threshold) and one
        // via HNSW (forced by n_obs > threshold).
        //
        // Since the dispatch is internal to build_knn_graph, force the
        // HNSW path by calling `build_knn_hnsw` directly. Same for the
        // exact path.
        let n_obs = 1_000;
        let n_vars = 100;
        let k = 15;
        let mut data = Vec::with_capacity(n_obs * n_vars);
        let mut state: u64 = 0x1234_5678;
        for cluster in 0..3 {
            for _ in 0..(n_obs / 3) {
                for _ in 0..n_vars {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let noise = ((state >> 32) as f32 / u32::MAX as f32) * 0.5;
                    data.push(cluster as f32 * 10.0 + noise);
                }
            }
        }
        // Pad to exactly n_obs rows
        while data.len() < n_obs * n_vars {
            data.push(0.0);
        }

        let (exact_idx, _) = build_knn_exact(&data, n_obs, n_vars, k);
        let (hnsw_idx, _) = build_knn_hnsw(&data, n_obs, n_vars, k, 200, 50, 42);

        // Recall: per row, what fraction of HNSW's top-k are in exact's top-k?
        let mut total_hits = 0usize;
        for i in 0..n_obs {
            use std::collections::HashSet;
            let exact_set: HashSet<usize> = exact_idx[i * k..(i + 1) * k].iter().copied().collect();
            for j in 0..k {
                if exact_set.contains(&hnsw_idx[i * k + j]) {
                    total_hits += 1;
                }
            }
        }
        let recall = total_hits as f64 / (n_obs * k) as f64;
        assert!(
            recall >= 0.95,
            "HNSW recall vs exact kNN was {:.3}, expected >= 0.95",
            recall,
        );
    }

    #[test]
    fn test_dispatch_boundary() {
        // Both sides of EXACT_KNN_NOBS_THRESHOLD must produce a valid
        // KnnResult with the documented invariants.
        let n_vars = 32;
        let k = 5;

        // Just under the threshold → exact path.
        let n_below = EXACT_KNN_NOBS_THRESHOLD - 1;
        let data = synth_data(n_below, n_vars, 0xBEEF_0001);
        let r = build_knn_graph(&data, n_below, n_vars, k, 200, 50, 0).unwrap();
        assert_eq!(r.n_obs, n_below);
        assert_eq!(r.indices.len(), n_below * k);
        assert_eq!(r.conn_indptr.len(), n_below + 1);
        assert_eq!(r.conn_indptr[0], 0);
        assert_eq!(*r.conn_indptr.last().unwrap(), r.conn_data.len() as i64);

        // Just over the threshold → HNSW path. Trim n_vars so this stays
        // fast in CI but still has enough dimensionality for HNSW to be
        // meaningful.
        let n_above = EXACT_KNN_NOBS_THRESHOLD + 1;
        let small_n_vars = 8;
        let data = synth_data(n_above, small_n_vars, 0xBEEF_0002);
        let r = build_knn_graph(&data, n_above, small_n_vars, k, 200, 50, 0).unwrap();
        assert_eq!(r.n_obs, n_above);
        assert_eq!(r.indices.len(), n_above * k);
        assert_eq!(r.conn_indptr.len(), n_above + 1);
        assert_eq!(r.conn_indptr[0], 0);
        assert_eq!(*r.conn_indptr.last().unwrap(), r.conn_data.len() as i64);
    }
}
