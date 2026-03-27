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

use instant_distance::{Hnsw, Point, Search};
use rayon::prelude::*;

use crate::error::{AccelError, Result};

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

    // Build query points (used for search; separate from HNSW build points
    // because build_hnsw() consumes its input)
    let query_points: Vec<EuclideanPoint> = (0..n_obs)
        .map(|i| {
            let start = i * n_vars;
            EuclideanPoint(data[start..start + n_vars].to_vec())
        })
        .collect();

    // Build HNSW index
    // build_hnsw() returns (Hnsw, Vec<PointId>) where the Vec maps
    // original_index -> internal PointId (points are shuffled internally)
    let build_points: Vec<EuclideanPoint> = query_points.clone();
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

    // Query k-nearest neighbors for each point in parallel.
    // Hnsw::search takes &self (shared ref) and Hnsw<P> is Sync when P: Sync,
    // so concurrent searches are safe. Each thread creates its own Search struct
    // which holds per-query mutable state.
    let all_results: Vec<Vec<(usize, f64)>> = query_points
        .par_iter()
        .enumerate()
        .map(|(i, query)| {
            let mut search = Search::default();

            // Query k+1 neighbors since the point itself will be in the results
            hnsw.search(query, &mut search)
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

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Build a CSR sparse matrix from kNN indices and distances.
///
/// Returns (indptr, indices, data) for an (n_obs × n_obs) sparse matrix
/// where entry (i, j) = distance from point i to neighbor j.
fn build_knn_csr(
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
fn compute_connectivities(
    knn_indices: &[usize],
    knn_distances: &[f64],
    n_obs: usize,
    n_neighbors: usize,
) -> (Vec<i64>, Vec<i32>, Vec<f64>) {
    let target = (n_neighbors as f64).ln() / std::f64::consts::LN_2; // log2(k)

    // Compute per-point bandwidths (σ) and membership strengths
    let sigmas: Vec<f64> = (0..n_obs)
        .map(|i| {
            let offset = i * n_neighbors;
            let rho = knn_distances[offset]; // nearest neighbor distance

            // Binary search for σ
            find_sigma(&knn_distances[offset..offset + n_neighbors], rho, target)
        })
        .collect();

    // Compute asymmetric membership strengths and build O(1) lookup map.
    // HashMap keyed by (i, j) → μ(i,j) replaces the previous O(k) linear scan
    // per edge, reducing total symmetrization cost from O(n × k²) to O(n × k).
    use std::collections::HashMap;

    let mut mu_map: HashMap<(usize, usize), f64> = HashMap::with_capacity(n_obs * n_neighbors);

    #[allow(clippy::needless_range_loop)] // i indexes sigmas, knn_indices, and knn_distances
    for i in 0..n_obs {
        let offset = i * n_neighbors;
        let rho = knn_distances[offset];
        let sigma = sigmas[i];

        for j_idx in 0..n_neighbors {
            let j = knn_indices[offset + j_idx];
            let d = knn_distances[offset + j_idx];
            let strength = if d <= rho || sigma <= 1e-10 {
                1.0
            } else {
                (-(d - rho) / sigma).exp()
            };
            mu_map.insert((i, j), strength);
        }
    }

    // Symmetrize: conn(i,j) = μ(i,j) + μ(j,i) - μ(i,j) * μ(j,i)
    // Use a HashSet to track processed edges so each (i,j)/(j,i) pair is
    // computed exactly once — eliminates the previous dedup_by_key call that
    // could silently drop values with different floating-point rounding.
    let mut sym: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_obs];
    let mut seen = std::collections::HashSet::<(usize, usize)>::new();

    for &(i, j) in mu_map.keys() {
        // Skip if we already processed this edge from the (j,i) direction
        if !seen.insert((i, j)) {
            continue;
        }
        seen.insert((j, i));

        let mu_ij = mu_map.get(&(i, j)).copied().unwrap_or(0.0);
        let mu_ji = mu_map.get(&(j, i)).copied().unwrap_or(0.0);
        let conn = mu_ij + mu_ji - mu_ij * mu_ji;

        if conn > 0.0 {
            sym[i].push((j, conn));
            if i != j {
                sym[j].push((i, conn));
            }
        }
    }

    // Sort each row by column index (no dedup needed — seen set prevents duplicates)
    for row in &mut sym {
        row.sort_by_key(|&(col, _)| col);
    }

    // Convert to CSR
    let mut indptr = Vec::with_capacity(n_obs + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();

    indptr.push(0i64);
    for row in &sym {
        for &(col, val) in row {
            indices.push(col as i32);
            data.push(val);
        }
        indptr.push(indices.len() as i64);
    }

    (indptr, indices, data)
}

/// Binary search for σ such that Σ exp(-max(d - ρ, 0) / σ) = target.
fn find_sigma(distances: &[f64], rho: f64, target: f64) -> f64 {
    let mut lo = 1e-10_f64;
    let mut hi = 1000.0_f64;
    let mut mid;

    for _ in 0..64 {
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
}
