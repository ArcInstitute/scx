//! Shared pairwise distance kernels for evaluation metrics.
//!
//! Provides point-to-point distance functions (Euclidean, L1, cosine) and
//! streaming mean pairwise distance computation that avoids materializing
//! the full `[N, N]` distance matrix — computing the running sum instead.
//!
//! For Euclidean and cosine metrics, a faer-backed gemm path is available
//! that materialises the `[N_A, N_B]` Gram matrix in one BLAS-level call
//! and expands it to distances. This is much faster than the row-by-row
//! scalar path on dense, low-dimensional embeddings (e.g. PCA outputs).
//!
//! Both kernels are generic over [`PairwiseFloat`] (`f32` or `f64`).
//! Reductions always accumulate in `f64` regardless of the input precision,
//! so the parity tolerances stay tight even when callers feed `f32` inputs.

use faer::linalg::matmul::matmul;
use faer::{Mat, MatRef};
use num_traits::{Float, One};
use rayon::prelude::*;

use super::DistanceMetric;
use crate::AccelError;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Sealed marker bundle for floating-point element types usable in the
/// generic pairwise-distance kernels. Implemented for `f32` and `f64`.
///
/// Sealed because the implementation relies on faer's `ComplexField` impls
/// which only exist for `f32`/`f64` here, and external impls would silently
/// route through the wrong matmul kernel.
pub trait PairwiseFloat:
    sealed::Sealed + Float + One + Send + Sync + Copy + faer_traits::ComplexField + 'static
{
    /// Widen to `f64` for reduction. The `f64` impl is a no-op.
    fn as_f64(self) -> f64;
    /// Narrow `f64` → Self. Used to apply an `f64` scaling factor (e.g. an
    /// inverse norm) back into an `F`-typed buffer.
    fn from_f64(x: f64) -> Self;
}

impl PairwiseFloat for f32 {
    #[inline]
    fn as_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
}

impl PairwiseFloat for f64 {
    #[inline]
    fn as_f64(self) -> f64 {
        self
    }
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
}

/// Backend selection for `mean_pairwise_distance` and friends.
///
/// `Auto` picks the gemm path for Euclidean/cosine and the scalar path for
/// L1 (which has no gemm formulation). `Gemm` requires a metric that admits
/// the `‖a‖² + ‖b‖² − 2·a·b` decomposition; requesting `Gemm` with `L1`
/// returns an error rather than silently falling back, so misconfigurations
/// are loud.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceBackend {
    /// Choose automatically based on metric.
    #[default]
    Auto,
    /// faer-based gemm path. Only valid for Euclidean and cosine.
    Gemm,
    /// Row-by-row scalar path. Always valid.
    Scalar,
}

/// Resolve `Auto` to a concrete backend for the given metric.
#[inline]
fn resolve_backend(
    backend: DistanceBackend,
    metric: DistanceMetric,
) -> crate::Result<DistanceBackend> {
    match (backend, metric) {
        (DistanceBackend::Auto, DistanceMetric::L1) => Ok(DistanceBackend::Scalar),
        (DistanceBackend::Auto, _) => Ok(DistanceBackend::Gemm),
        (DistanceBackend::Gemm, DistanceMetric::L1) => Err(AccelError::InvalidInput(
            "gemm backend not supported for L1 metric — use Scalar or Auto".to_string(),
        )),
        (b, _) => Ok(b),
    }
}

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

// ── Generic scalar distance helpers ─────────────────────────────────────
//
// These widen each element to `f64` before accumulation so f32 inputs do not
// lose precision in the inner reduction. For `F = f64` the `as_f64` calls
// monomorphise to no-ops, so these are equivalent to the existing f64 fns.

#[inline]
fn euclidean_distance_generic<F: PairwiseFloat>(a: &[F], b: &[F], n_dims: usize) -> f64 {
    debug_assert!(a.len() >= n_dims && b.len() >= n_dims);
    let mut sum_sq = 0.0f64;
    for k in 0..n_dims {
        let diff = a[k].as_f64() - b[k].as_f64();
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

#[inline]
fn l1_distance_generic<F: PairwiseFloat>(a: &[F], b: &[F], n_dims: usize) -> f64 {
    debug_assert!(a.len() >= n_dims && b.len() >= n_dims);
    let mut sum_abs = 0.0f64;
    for k in 0..n_dims {
        sum_abs += (a[k].as_f64() - b[k].as_f64()).abs();
    }
    sum_abs
}

#[inline]
fn cosine_distance_generic<F: PairwiseFloat>(a: &[F], b: &[F], n_dims: usize) -> f64 {
    debug_assert!(a.len() >= n_dims && b.len() >= n_dims);
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for k in 0..n_dims {
        let ak = a[k].as_f64();
        let bk = b[k].as_f64();
        dot += ak * bk;
        norm_a += ak * ak;
        norm_b += bk * bk;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 1.0;
    }
    (1.0 - dot / denom).clamp(0.0, 2.0)
}

#[inline]
fn point_distance_generic<F: PairwiseFloat>(
    a: &[F],
    b: &[F],
    n_dims: usize,
    metric: DistanceMetric,
) -> f64 {
    match metric {
        DistanceMetric::Euclidean => euclidean_distance_generic(a, b, n_dims),
        DistanceMetric::L1 => l1_distance_generic(a, b, n_dims),
        DistanceMetric::Cosine => cosine_distance_generic(a, b, n_dims),
    }
}

/// Compute per-row distance sums for `(a, b)` via faer's gemm.
///
/// Returns a `Vec<f64>` of length `n_a` where `row_sums[i] = sum_j d(a_i, b_j)`.
/// The reduction is deterministic in row order (sequential per-row sum), but
/// the matmul itself uses faer's parallel `Par::rayon(0)` and is not strictly
/// bit-deterministic across thread counts (results agree within floating-point
/// rounding, well below the `atol=1e-4` parity tolerance).
///
/// `metric` must be `Euclidean` or `Cosine`; `L1` has no gemm formulation.
///
/// The matmul runs at the input precision `F` (so f32 inputs use a single-precision
/// gemm — the throughput win that motivates the f32 path) but the row-norm² and
/// final distance expansion always run in `f64`, so the reduction stays tight
/// against the `atol=1e-4` parity bound regardless of input precision.
fn pairwise_gemm_row_sums<F: PairwiseFloat>(
    a: &[F],
    b: &[F],
    n_a: usize,
    n_b: usize,
    n_dims: usize,
    metric: DistanceMetric,
) -> Vec<f64> {
    debug_assert!(matches!(
        metric,
        DistanceMetric::Euclidean | DistanceMetric::Cosine
    ));

    // For cosine, work on row-normalized copies. Then cosine distance is
    // simply `1 - (Ã · B̃ᵀ)[i,j]`. Zero-norm rows are left as zeros; their
    // dot product with anything is 0, giving distance 1.0 (matches the
    // scalar `cosine_distance` semantics).
    let (a_buf, b_buf, a_ref, b_ref);
    let (a_view, b_view) = match metric {
        DistanceMetric::Cosine => {
            a_buf = normalize_rows(a, n_a, n_dims);
            b_buf = normalize_rows(b, n_b, n_dims);
            a_ref = MatRef::<F>::from_row_major_slice(&a_buf, n_a, n_dims);
            b_ref = MatRef::<F>::from_row_major_slice(&b_buf, n_b, n_dims);
            (a_ref, b_ref)
        }
        DistanceMetric::Euclidean => {
            let a_ref = MatRef::<F>::from_row_major_slice(a, n_a, n_dims);
            let b_ref = MatRef::<F>::from_row_major_slice(b, n_b, n_dims);
            (a_ref, b_ref)
        }
        DistanceMetric::L1 => unreachable!("L1 is not supported for gemm path"),
    };

    let mut gram = Mat::<F>::zeros(n_a, n_b);
    matmul(
        gram.as_mut(),
        faer::Accum::Replace,
        a_view,
        b_view.transpose(),
        F::one(),
        faer::Par::rayon(0),
    );

    match metric {
        DistanceMetric::Euclidean => {
            // Precompute squared row norms in f64 (widening from F as needed).
            let a_sq: Vec<f64> = (0..n_a)
                .into_par_iter()
                .with_min_len(64)
                .map(|i| {
                    let row = &a[i * n_dims..(i + 1) * n_dims];
                    row.iter()
                        .map(|&x| {
                            let xf = x.as_f64();
                            xf * xf
                        })
                        .sum::<f64>()
                })
                .collect();
            let b_sq: Vec<f64> = (0..n_b)
                .into_par_iter()
                .with_min_len(64)
                .map(|j| {
                    let row = &b[j * n_dims..(j + 1) * n_dims];
                    row.iter()
                        .map(|&x| {
                            let xf = x.as_f64();
                            xf * xf
                        })
                        .sum::<f64>()
                })
                .collect();

            (0..n_a)
                .into_par_iter()
                .with_min_len((n_b * 16).max(1))
                .map(|i| {
                    let ai_sq = a_sq[i];
                    let mut row_sum = 0.0f64;
                    for j in 0..n_b {
                        // max(0, ·) clamps tiny negatives from FP cancellation
                        // for near-identical rows.
                        let g = gram[(i, j)].as_f64();
                        let d_sq = (ai_sq + b_sq[j] - 2.0 * g).max(0.0);
                        row_sum += d_sq.sqrt();
                    }
                    row_sum
                })
                .collect()
        }
        DistanceMetric::Cosine => (0..n_a)
            .into_par_iter()
            .with_min_len((n_b * 16).max(1))
            .map(|i| {
                let mut row_sum = 0.0f64;
                for j in 0..n_b {
                    // Clamp to [0, 2] to match scalar cosine_distance behavior.
                    let d = (1.0 - gram[(i, j)].as_f64()).clamp(0.0, 2.0);
                    row_sum += d;
                }
                row_sum
            })
            .collect(),
        DistanceMetric::L1 => unreachable!(),
    }
}

/// Allocate a row-normalized copy of `[n_rows × n_dims]` row-major matrix.
/// Zero-norm rows are left as zeros — `cosine_distance(zero, x) = 1.0` falls
/// out naturally from the dot product being zero.
///
/// The norm² accumulator runs in `f64` to avoid f32 overflow on rows with
/// many large entries; the inverse-norm scaling is then narrowed back to `F`
/// before applying to each element.
fn normalize_rows<F: PairwiseFloat>(data: &[F], n_rows: usize, n_dims: usize) -> Vec<F> {
    let mut out = vec![F::from_f64(0.0); n_rows * n_dims];
    out.par_chunks_mut(n_dims)
        .enumerate()
        .with_min_len(64)
        .for_each(|(i, dst)| {
            let src = &data[i * n_dims..(i + 1) * n_dims];
            let norm_sq: f64 = src
                .iter()
                .map(|&x| {
                    let xf = x.as_f64();
                    xf * xf
                })
                .sum();
            if norm_sq > 0.0 {
                let inv = F::from_f64(norm_sq.sqrt().recip());
                for k in 0..n_dims {
                    dst[k] = src[k] * inv;
                }
            }
            // else leave as zeros
        });
    out
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
/// * `backend` — Backend dispatch (`Auto`, `Gemm`, or `Scalar`).
///
/// # Returns
/// `sum(d(a_i, b_j)) / (n_a * n_b)` for all pairs `(i, j)`.
///
/// Returns 0.0 if either matrix is empty.
///
/// # Errors
/// Returns `AccelError::InvalidInput` if `backend=Gemm` is combined with
/// `metric=L1` (L1 has no gemm formulation).
pub fn mean_pairwise_distance<F: PairwiseFloat>(
    a: &[F],
    b: &[F],
    n_a: usize,
    n_b: usize,
    n_dims: usize,
    metric: DistanceMetric,
    backend: DistanceBackend,
) -> crate::Result<f64> {
    if n_a == 0 || n_b == 0 {
        return Ok(0.0);
    }
    debug_assert!(a.len() >= n_a * n_dims);
    debug_assert!(b.len() >= n_b * n_dims);

    let backend = resolve_backend(backend, metric)?;

    // Parallelize per-row work but reduce sequentially in row order for
    // bit-stable output across thread counts and environments. Float
    // addition is not associative and rayon's `.sum()` combines partial
    // results in work-stealing order, so we can't use it here. The
    // `Vec<f64>` row-sum buffer costs O(n_a) transient memory — trivial
    // vs. the O(n_a * n_b * n_dims) inner work.
    //
    // `with_min_len(n_b * 16)` prevents oversubscription when this runs
    // inside `fused_edistance`, where the caller already holds a rayon
    // scope — tiny chunks would thrash.
    let row_sums: Vec<f64> = match backend {
        DistanceBackend::Gemm => pairwise_gemm_row_sums(a, b, n_a, n_b, n_dims, metric),
        DistanceBackend::Scalar => (0..n_a)
            .into_par_iter()
            .with_min_len((n_b * 16).max(1))
            .map(|i| {
                let row_a = &a[i * n_dims..(i + 1) * n_dims];
                let mut row_sum = 0.0f64;
                for j in 0..n_b {
                    let row_b = &b[j * n_dims..(j + 1) * n_dims];
                    row_sum += point_distance_generic(row_a, row_b, n_dims, metric);
                }
                row_sum
            })
            .collect(),
        DistanceBackend::Auto => unreachable!("resolve_backend collapses Auto"),
    };
    let total: f64 = row_sums.iter().sum();
    Ok(total / (n_a as f64 * n_b as f64))
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
pub fn mean_pairwise_distance_self<F: PairwiseFloat>(
    a: &[F],
    n: usize,
    n_dims: usize,
    metric: DistanceMetric,
    backend: DistanceBackend,
) -> crate::Result<f64> {
    if n < 2 {
        return Ok(0.0);
    }
    debug_assert!(a.len() >= n * n_dims);

    let backend = resolve_backend(backend, metric)?;

    match backend {
        DistanceBackend::Gemm => {
            // Re-use the (a, a) cross path; diagonal is exactly zero for
            // Euclidean / cosine on identical rows (clamping handles FP
            // cancellation), so dividing the full-matrix sum by n*n matches
            // the upper-triangle convention used by the scalar path.
            mean_pairwise_distance(a, a, n, n, n_dims, metric, DistanceBackend::Gemm)
        }
        DistanceBackend::Scalar => {
            // Parallelise the upper-triangle sum across rows. Each row `i`
            // has `n - i - 1` pairs to compute, so the work is uneven; the
            // `with_min_len((n * 16).max(1))` chunking matches the cross
            // path's heuristic and prevents oversubscription when this runs
            // inside `compute_energy_distance`'s pert-level par_iter.
            //
            // Reduction is sequential over the per-row sums for bit-stable
            // output across thread counts.
            let row_sums: Vec<f64> = (0..n)
                .into_par_iter()
                .with_min_len((n * 16).max(1))
                .map(|i| {
                    let row_i = &a[i * n_dims..(i + 1) * n_dims];
                    let mut row_sum = 0.0f64;
                    for j in (i + 1)..n {
                        let row_j = &a[j * n_dims..(j + 1) * n_dims];
                        row_sum += point_distance_generic(row_i, row_j, n_dims, metric);
                    }
                    row_sum
                })
                .collect();
            let total: f64 = row_sums.iter().sum();
            // Upper triangle has n*(n-1)/2 pairs. The full n×n matrix has these
            // pairs twice plus n zero-diagonal entries, so:
            //   full_sum = 2 * total
            //   mean = full_sum / (n * n)
            Ok(2.0 * total / (n as f64 * n as f64))
        }
        DistanceBackend::Auto => unreachable!("resolve_backend collapses Auto"),
    }
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
        let m = mean_pairwise_distance(
            &a,
            &b,
            2,
            2,
            2,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();
        assert!((m - 2.5).abs() < 1e-12, "expected 2.5, got {m}");
    }

    #[test]
    fn test_mean_pairwise_cross_euclidean() {
        // A = [[0, 0]]  B = [[1, 0], [0, 1]]
        // d(A0,B0)=1, d(A0,B1)=1
        // Mean = 2 / 2 = 1.0
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let m = mean_pairwise_distance(
            &a,
            &b,
            1,
            2,
            2,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();
        assert!((m - 1.0).abs() < 1e-12, "expected 1.0, got {m}");
    }

    #[test]
    fn test_mean_pairwise_empty() {
        let a: Vec<f64> = vec![];
        let b = vec![1.0, 2.0];
        assert_eq!(
            mean_pairwise_distance(
                &a,
                &b,
                0,
                1,
                2,
                DistanceMetric::Euclidean,
                DistanceBackend::Auto,
            )
            .unwrap(),
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

        let streaming = mean_pairwise_distance(
            &a,
            &b,
            n_a,
            n_b,
            n_d,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();
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

        let result = mean_pairwise_distance_self(
            &a,
            n,
            d,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();
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

        let self_d = mean_pairwise_distance_self(
            &a,
            n,
            d,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();
        let cross_d = mean_pairwise_distance(
            &a,
            &a,
            n,
            n,
            d,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();
        assert!(
            (self_d - cross_d).abs() < 1e-12,
            "self {self_d} != cross {cross_d}"
        );
    }

    #[test]
    fn test_mean_self_distance_single_row() {
        let a = vec![1.0, 2.0, 3.0];
        assert_eq!(
            mean_pairwise_distance_self(
                &a,
                1,
                3,
                DistanceMetric::Euclidean,
                DistanceBackend::Auto,
            )
            .unwrap(),
            0.0
        );
    }

    #[test]
    fn test_mean_self_distance_identical_rows() {
        // All rows identical → distance = 0
        let a = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let d = mean_pairwise_distance_self(
            &a,
            3,
            2,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();
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
        let result = mean_pairwise_distance_self(
            &a,
            n,
            d,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();
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
        let result = mean_pairwise_distance_self(
            &a,
            n,
            d,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();
        assert!(result > 0.0, "high-dim distance should be positive");
        // Also test L1 / cosine
        let r_l1 =
            mean_pairwise_distance_self(&a, n, d, DistanceMetric::L1, DistanceBackend::Scalar)
                .unwrap();
        assert!(r_l1 > 0.0);
        let r_cos =
            mean_pairwise_distance_self(&a, n, d, DistanceMetric::Cosine, DistanceBackend::Scalar)
                .unwrap();
        assert!(r_cos > 0.0);
    }

    #[test]
    fn test_mean_pairwise_l1() {
        // A = [[0]], B = [[3], [7]]
        // distances: 3, 7 → mean = 5
        let a = vec![0.0];
        let b = vec![3.0, 7.0];
        let m =
            mean_pairwise_distance(&a, &b, 1, 2, 1, DistanceMetric::L1, DistanceBackend::Scalar)
                .unwrap();
        assert!((m - 5.0).abs() < 1e-12);
    }

    // ── Backend dispatch tests ────────────────────────────────────────

    #[test]
    fn test_gemm_l1_rejected() {
        let a = vec![0.0, 1.0];
        let b = vec![1.0, 2.0];
        let err =
            mean_pairwise_distance(&a, &b, 1, 1, 2, DistanceMetric::L1, DistanceBackend::Gemm)
                .unwrap_err();
        assert!(matches!(err, AccelError::InvalidInput(_)));

        let err = mean_pairwise_distance_self(&a, 2, 1, DistanceMetric::L1, DistanceBackend::Gemm)
            .unwrap_err();
        assert!(matches!(err, AccelError::InvalidInput(_)));
    }

    #[test]
    fn test_gemm_matches_scalar_euclidean() {
        // Larger random-ish matrices to exercise gemm vs scalar.
        let n_a = 17;
        let n_b = 23;
        let n_d = 8;
        let a: Vec<f64> = (0..n_a * n_d).map(|i| (i as f64 * 0.137).sin()).collect();
        let b: Vec<f64> = (0..n_b * n_d).map(|i| (i as f64 * 0.241).cos()).collect();

        let scalar = mean_pairwise_distance(
            &a,
            &b,
            n_a,
            n_b,
            n_d,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();
        let gemm = mean_pairwise_distance(
            &a,
            &b,
            n_a,
            n_b,
            n_d,
            DistanceMetric::Euclidean,
            DistanceBackend::Gemm,
        )
        .unwrap();
        assert!(
            (scalar - gemm).abs() < 1e-10,
            "scalar={scalar} vs gemm={gemm}"
        );
    }

    #[test]
    fn test_gemm_matches_scalar_cosine() {
        let n_a = 13;
        let n_b = 19;
        let n_d = 7;
        let a: Vec<f64> = (0..n_a * n_d)
            .map(|i| ((i as f64 + 1.0) * 0.13).sin())
            .collect();
        let b: Vec<f64> = (0..n_b * n_d)
            .map(|i| ((i as f64 + 1.0) * 0.27).cos())
            .collect();

        let scalar = mean_pairwise_distance(
            &a,
            &b,
            n_a,
            n_b,
            n_d,
            DistanceMetric::Cosine,
            DistanceBackend::Scalar,
        )
        .unwrap();
        let gemm = mean_pairwise_distance(
            &a,
            &b,
            n_a,
            n_b,
            n_d,
            DistanceMetric::Cosine,
            DistanceBackend::Gemm,
        )
        .unwrap();
        assert!(
            (scalar - gemm).abs() < 1e-10,
            "scalar={scalar} vs gemm={gemm}"
        );
    }

    #[test]
    fn test_gemm_self_matches_scalar() {
        // Self-distance via gemm should match scalar within FP tolerance.
        let n = 11;
        let d = 5;
        let a: Vec<f64> = (0..n * d).map(|i| (i as f64 * 0.171).sin()).collect();

        let scalar = mean_pairwise_distance_self(
            &a,
            n,
            d,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();
        let gemm =
            mean_pairwise_distance_self(&a, n, d, DistanceMetric::Euclidean, DistanceBackend::Gemm)
                .unwrap();
        // Reduction order differs between gemm and scalar paths; FP rounding
        // produces sub-eps drift well within the parity tolerances.
        assert!(
            (scalar - gemm).abs() < 1e-8,
            "scalar={scalar} vs gemm={gemm}"
        );
    }

    #[test]
    fn test_gemm_zero_norm_cosine() {
        // Zero-norm rows give cosine distance 1.0 in the scalar path; the
        // gemm path should produce the same answer because normalize_rows
        // leaves zero rows as zeros (dot product = 0, distance = 1).
        let a = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]; // row 0 = zero
        let b = vec![1.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let scalar = mean_pairwise_distance(
            &a,
            &b,
            2,
            2,
            3,
            DistanceMetric::Cosine,
            DistanceBackend::Scalar,
        )
        .unwrap();
        let gemm = mean_pairwise_distance(
            &a,
            &b,
            2,
            2,
            3,
            DistanceMetric::Cosine,
            DistanceBackend::Gemm,
        )
        .unwrap();
        assert!(
            (scalar - gemm).abs() < 1e-12,
            "scalar={scalar} vs gemm={gemm}"
        );
    }

    #[test]
    fn test_auto_backend_routes_correctly() {
        // Auto + L1 should not error (routes to Scalar).
        let a = vec![0.0, 1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0];
        let _ = mean_pairwise_distance(&a, &b, 3, 3, 1, DistanceMetric::L1, DistanceBackend::Auto)
            .unwrap();
        // Auto + Euclidean should not error (routes to Gemm).
        let _ = mean_pairwise_distance(
            &a,
            &b,
            3,
            3,
            1,
            DistanceMetric::Euclidean,
            DistanceBackend::Auto,
        )
        .unwrap();
    }
}
