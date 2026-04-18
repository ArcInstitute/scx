//! Shared pairwise distance kernels for evaluation metrics.
//!
//! Provides point-to-point distance functions (Euclidean, L1, cosine) and
//! streaming mean pairwise distance computation that avoids materializing
//! the full `[N, N]` distance matrix — computing the running sum instead.

use rayon::prelude::*;

use super::DistanceMetric;

/// Euclidean distance between two row vectors of length `n_dims`.
///
/// `a` and `b` should point to contiguous slices of at least `n_dims` elements.
#[inline]
pub fn euclidean_distance(a: &[f64], b: &[f64], n_dims: usize) -> f64 {
    debug_assert!(a.len() >= n_dims && b.len() >= n_dims);
    let mut sum_sq = 0.0f64;
    for k in 0..n_dims {
        let diff = a[k] - b[k];
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

/// Manhattan (L1) distance between two row vectors of length `n_dims`.
#[inline]
pub fn l1_distance(a: &[f64], b: &[f64], n_dims: usize) -> f64 {
    debug_assert!(a.len() >= n_dims && b.len() >= n_dims);
    let mut sum_abs = 0.0f64;
    for k in 0..n_dims {
        sum_abs += (a[k] - b[k]).abs();
    }
    sum_abs
}

/// Cosine distance between two row vectors: `1 - cosine_similarity(a, b)`.
///
/// Returns 1.0 (maximum distance) if either vector has zero norm.
#[inline]
pub fn cosine_distance(a: &[f64], b: &[f64], n_dims: usize) -> f64 {
    debug_assert!(a.len() >= n_dims && b.len() >= n_dims);
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for k in 0..n_dims {
        dot += a[k] * b[k];
        norm_a += a[k] * a[k];
        norm_b += b[k] * b[k];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 1.0;
    }
    // Clamp to [0, 2] to handle floating-point imprecision.
    (1.0 - dot / denom).clamp(0.0, 2.0)
}

/// Dispatch a point-to-point distance by metric enum.
#[inline]
pub fn point_distance(a: &[f64], b: &[f64], n_dims: usize, metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => euclidean_distance(a, b, n_dims),
        DistanceMetric::L1 => l1_distance(a, b, n_dims),
        DistanceMetric::Cosine => cosine_distance(a, b, n_dims),
    }
}

/// Compute the mean pairwise distance between rows of two matrices without
/// materializing the full `[N_A, N_B]` distance matrix.
///
/// # Arguments
/// * `a` — `[N_A × D]` row-major matrix.
/// * `b` — `[N_B × D]` row-major matrix.
/// * `n_a` — Number of rows in `a`.
/// * `n_b` — Number of rows in `b`.
/// * `n_dims` — Dimensionality `D`.
/// * `metric` — Distance metric to use.
///
/// # Returns
/// `sum(d(a_i, b_j)) / (n_a * n_b)` for all pairs `(i, j)`.
///
/// Returns 0.0 if either matrix is empty.
pub fn mean_pairwise_distance(
    a: &[f64],
    b: &[f64],
    n_a: usize,
    n_b: usize,
    n_dims: usize,
    metric: DistanceMetric,
) -> f64 {
    if n_a == 0 || n_b == 0 {
        return 0.0;
    }
    debug_assert!(a.len() >= n_a * n_dims);
    debug_assert!(b.len() >= n_b * n_dims);

    // Parallelize the outer loop. `with_min_len(n_b * 16)` prevents
    // oversubscription when this runs inside `fused_edistance`, where the
    // caller already holds a rayon scope — tiny chunks would thrash.
    // Reduction is associative; float sum order is stable given fixed chunk
    // boundaries (rayon's split_at is deterministic for slices).
    let total: f64 = (0..n_a)
        .into_par_iter()
        .with_min_len((n_b * 16).max(1))
        .map(|i| {
            let row_a = &a[i * n_dims..(i + 1) * n_dims];
            let mut row_sum = 0.0f64;
            for j in 0..n_b {
                let row_b = &b[j * n_dims..(j + 1) * n_dims];
                row_sum += point_distance(row_a, row_b, n_dims, metric);
            }
            row_sum
        })
        .sum();
    total / (n_a as f64 * n_b as f64)
}

/// Compute the mean self-pairwise distance (all distinct pairs) for a single
/// matrix, using the upper-triangle optimization.
///
/// # Arguments
/// * `a` — `[N × D]` row-major matrix.
/// * `n` — Number of rows.
/// * `n_dims` — Dimensionality.
/// * `metric` — Distance metric to use.
///
/// # Returns
/// The mean distance over all `n*(n-1)/2` distinct unordered pairs,
/// equivalent to `pairwise_distances(a, a).mean()` but including the
/// zero diagonal (matching sklearn's convention for `pairwise_distances`).
///
/// Specifically: `sum(d(a_i, a_j) for i < j) * 2 / (n * n)`.
/// This matches `sklearn.metrics.pairwise_distances(a, a).mean()` because
/// the diagonal entries are 0 and the matrix is symmetric.
///
/// Returns 0.0 if `n < 2`.
pub fn mean_pairwise_distance_self(
    a: &[f64],
    n: usize,
    n_dims: usize,
    metric: DistanceMetric,
) -> f64 {
    if n < 2 {
        return 0.0;
    }
    debug_assert!(a.len() >= n * n_dims);

    let mut total = 0.0f64;
    for i in 0..n {
        let row_i = &a[i * n_dims..(i + 1) * n_dims];
        for j in (i + 1)..n {
            let row_j = &a[j * n_dims..(j + 1) * n_dims];
            total += point_distance(row_i, row_j, n_dims, metric);
        }
    }
    // Upper triangle has n*(n-1)/2 pairs. The full n×n matrix has these
    // pairs twice plus n zero-diagonal entries, so:
    //   full_sum = 2 * total
    //   mean = full_sum / (n * n)
    2.0 * total / (n as f64 * n as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Point distance tests ──────────────────────────────────────────

    #[test]
    fn test_euclidean_known() {
        let a = [3.0, 0.0];
        let b = [0.0, 4.0];
        let d = euclidean_distance(&a, &b, 2);
        assert!((d - 5.0).abs() < 1e-12, "3-4-5 triangle: got {d}");
    }

    #[test]
    fn test_euclidean_identical() {
        let a = [1.0, 2.0, 3.0];
        assert!(euclidean_distance(&a, &a, 3) < 1e-15);
    }

    #[test]
    fn test_l1_known() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 0.0, -1.0];
        // |3| + |2| + |4| = 9
        let d = l1_distance(&a, &b, 3);
        assert!((d - 9.0).abs() < 1e-12, "expected 9, got {d}");
    }

    #[test]
    fn test_l1_identical() {
        let a = [1.0, 2.0, 3.0];
        assert!(l1_distance(&a, &a, 3) < 1e-15);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let d = cosine_distance(&a, &b, 2);
        assert!(
            (d - 1.0).abs() < 1e-12,
            "orthogonal → distance 1.0, got {d}"
        );
    }

    #[test]
    fn test_cosine_identical_direction() {
        let a = [1.0, 2.0, 3.0];
        let b = [2.0, 4.0, 6.0]; // same direction, different magnitude
        let d = cosine_distance(&a, &b, 3);
        assert!(d.abs() < 1e-12, "same direction → distance 0, got {d}");
    }

    #[test]
    fn test_cosine_opposite() {
        let a = [1.0, 0.0];
        let b = [-1.0, 0.0];
        let d = cosine_distance(&a, &b, 2);
        assert!((d - 2.0).abs() < 1e-12, "opposite → distance 2.0, got {d}");
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = [0.0, 0.0];
        let b = [1.0, 2.0];
        let d = cosine_distance(&a, &b, 2);
        assert!(
            (d - 1.0).abs() < 1e-12,
            "zero vector → distance 1.0, got {d}"
        );
    }

    #[test]
    fn test_point_distance_dispatch() {
        let a = [3.0, 0.0];
        let b = [0.0, 4.0];
        let de = point_distance(&a, &b, 2, DistanceMetric::Euclidean);
        let dl = point_distance(&a, &b, 2, DistanceMetric::L1);
        let dc = point_distance(&a, &b, 2, DistanceMetric::Cosine);
        assert!((de - 5.0).abs() < 1e-12);
        assert!((dl - 7.0).abs() < 1e-12);
        assert!(dc > 0.0 && dc < 2.0);
    }

    // ── Mean pairwise distance tests ──────────────────────────────────

    #[test]
    fn test_mean_pairwise_2x2_euclidean() {
        // A = [[0, 0], [3, 4]]   B = [[0, 0], [3, 4]]
        // Full matrix:
        //   d(A0,B0)=0, d(A0,B1)=5, d(A1,B0)=5, d(A1,B1)=0
        // Mean = 10 / 4 = 2.5
        let a = vec![0.0, 0.0, 3.0, 4.0];
        let b = vec![0.0, 0.0, 3.0, 4.0];
        let m = mean_pairwise_distance(&a, &b, 2, 2, 2, DistanceMetric::Euclidean);
        assert!((m - 2.5).abs() < 1e-12, "expected 2.5, got {m}");
    }

    #[test]
    fn test_mean_pairwise_cross_euclidean() {
        // A = [[0, 0]]  B = [[1, 0], [0, 1]]
        // d(A0,B0)=1, d(A0,B1)=1
        // Mean = 2 / 2 = 1.0
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let m = mean_pairwise_distance(&a, &b, 1, 2, 2, DistanceMetric::Euclidean);
        assert!((m - 1.0).abs() < 1e-12, "expected 1.0, got {m}");
    }

    #[test]
    fn test_mean_pairwise_empty() {
        let a: Vec<f64> = vec![];
        let b = vec![1.0, 2.0];
        assert_eq!(
            mean_pairwise_distance(&a, &b, 0, 1, 2, DistanceMetric::Euclidean),
            0.0
        );
    }

    #[test]
    fn test_mean_pairwise_distance_matches_brute_force() {
        // 3×2 and 4×2 matrices, verify streaming matches explicit matrix
        let a = vec![0.0, 0.0, 1.0, 1.0, 2.0, 0.0];
        let b = vec![0.0, 1.0, 1.0, 0.0, 3.0, 3.0, -1.0, 2.0];

        let n_a = 3;
        let n_b = 4;
        let n_d = 2;

        // Brute force: compute all distances, sum, divide
        let mut brute_sum = 0.0;
        for i in 0..n_a {
            for j in 0..n_b {
                let di =
                    euclidean_distance(&a[i * n_d..(i + 1) * n_d], &b[j * n_d..(j + 1) * n_d], n_d);
                brute_sum += di;
            }
        }
        let brute_mean = brute_sum / (n_a as f64 * n_b as f64);

        let streaming = mean_pairwise_distance(&a, &b, n_a, n_b, n_d, DistanceMetric::Euclidean);
        assert!(
            (streaming - brute_mean).abs() < 1e-12,
            "streaming {streaming} != brute {brute_mean}"
        );
    }

    // ── Self-distance tests ───────────────────────────────────────────

    #[test]
    fn test_mean_self_distance_matches_full() {
        // A = [[0, 0], [3, 4], [1, 1]]
        // Full pairwise matrix (3×3 symmetric, diagonal=0):
        //   d(0,1)=5, d(0,2)=sqrt(2), d(1,2)=sqrt(4+9)=sqrt(13)
        //   full_sum = 2*(5 + sqrt(2) + sqrt(13))
        //   mean = full_sum / 9
        let a = vec![0.0, 0.0, 3.0, 4.0, 1.0, 1.0];
        let n = 3;
        let d = 2;

        let d01 = 5.0;
        let d02 = 2.0f64.sqrt();
        let d12 = 13.0f64.sqrt();
        let expected = 2.0 * (d01 + d02 + d12) / 9.0;

        let result = mean_pairwise_distance_self(&a, n, d, DistanceMetric::Euclidean);
        assert!(
            (result - expected).abs() < 1e-12,
            "self dist {result} != expected {expected}"
        );
    }

    #[test]
    fn test_mean_self_distance_vs_cross() {
        // Self-distance should equal cross-distance when a == b
        let a = vec![0.0, 0.0, 3.0, 4.0, 1.0, 1.0];
        let n = 3;
        let d = 2;

        let self_d = mean_pairwise_distance_self(&a, n, d, DistanceMetric::Euclidean);
        let cross_d = mean_pairwise_distance(&a, &a, n, n, d, DistanceMetric::Euclidean);
        assert!(
            (self_d - cross_d).abs() < 1e-12,
            "self {self_d} != cross {cross_d}"
        );
    }

    #[test]
    fn test_mean_self_distance_single_row() {
        let a = vec![1.0, 2.0, 3.0];
        assert_eq!(
            mean_pairwise_distance_self(&a, 1, 3, DistanceMetric::Euclidean),
            0.0
        );
    }

    #[test]
    fn test_mean_self_distance_identical_rows() {
        // All rows identical → distance = 0
        let a = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let d = mean_pairwise_distance_self(&a, 3, 2, DistanceMetric::Euclidean);
        assert!(d < 1e-15, "identical rows → 0, got {d}");
    }

    #[test]
    fn test_1d_distances() {
        // 1-dimensional case
        let a = vec![0.0, 3.0, 7.0];
        let n = 3;
        let d = 1;
        // pairs: |0-3|=3, |0-7|=7, |3-7|=4
        // full_sum = 2*(3+7+4) = 28, mean = 28/9
        let expected = 28.0 / 9.0;
        let result = mean_pairwise_distance_self(&a, n, d, DistanceMetric::Euclidean);
        assert!(
            (result - expected).abs() < 1e-12,
            "1d self: {result} != {expected}"
        );
    }

    #[test]
    fn test_high_dim() {
        // Quick smoke test with 100 dimensions
        let n = 5;
        let d = 100;
        let a: Vec<f64> = (0..n * d).map(|i| i as f64 * 0.01).collect();
        let result = mean_pairwise_distance_self(&a, n, d, DistanceMetric::Euclidean);
        assert!(result > 0.0, "high-dim distance should be positive");
        // Also test L1 / cosine
        let r_l1 = mean_pairwise_distance_self(&a, n, d, DistanceMetric::L1);
        assert!(r_l1 > 0.0);
        let r_cos = mean_pairwise_distance_self(&a, n, d, DistanceMetric::Cosine);
        assert!(r_cos > 0.0);
    }

    #[test]
    fn test_mean_pairwise_l1() {
        // A = [[0]], B = [[3], [7]]
        // distances: 3, 7 → mean = 5
        let a = vec![0.0];
        let b = vec![3.0, 7.0];
        let m = mean_pairwise_distance(&a, &b, 1, 2, 1, DistanceMetric::L1);
        assert!((m - 5.0).abs() < 1e-12);
    }
}
