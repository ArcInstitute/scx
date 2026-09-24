//! Shared pairwise distance kernels for evaluation metrics.
//!
//! Provides point-to-point distance functions (Euclidean, L1, cosine) and mean
//! pairwise distance computation that never materialises the full `[N, N]`
//! distance matrix — the reduction keeps one `f64` per row of `a` and nothing
//! else.
//!
//! For Euclidean and cosine metrics a faer-backed gemm path is available, which
//! is much faster than the row-by-row scalar path on dense, low-dimensional
//! embeddings (e.g. PCA outputs). It does need a Gram, `a·bᵀ`, and that is the
//! allocation this module's budget governs.
//!
//! # Working set
//!
//! The Gram is built **one row block of `a` at a time**, sized by
//! [`crate::mem_budget::plan_gram_row_block`] against
//! `SCX_ACCEL_PAIRWISE_MEMORY_BUDGET` (default 256 MiB), and reduced before the
//! next block overwrites it — rather than as `n_a · n_b · size_of::<F>()`, which
//! at the 100 K control cells `energy_distance` is documented for, on its
//! default `dtype="f32"`, is 40 GB. An input whose whole Gram already fits the
//! budget is a single block, i.e. exactly the unblocked computation.
//!
//! Stated exactly, because the knob is a tuning surface:
//!
//! - The Gram block is `max(budget, one Gram row)`. The planner never returns a
//!   zero-row block, so a single row wider than the budget still gets one — the
//!   floor that guarantees progress rather than an error.
//! - **Cosine allocates outside the budget.** `normalize_rows` materialises a
//!   full `n_a · n_dims` copy, plus `n_b · n_dims` when `a` and `b` differ — at
//!   100 K × 2000 f32 that is ~800 MB for a self-distance and ~1.6 GB for a
//!   cross. Lowering the budget does not touch it. Euclidean keeps only the
//!   `O(n)` f64 row-norm buffers.
//! - `compute_energy_distance` evaluates perturbations on a rayon `par_iter`, so
//!   up to `rayon::current_num_threads()` of everything above is live at once,
//!   and its `extract_group_rows_indexed` copies each group's rows per task
//!   independently of anything here.
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
    /// Unit roundoff of the type the **gemm accumulates in**, which is `Self`.
    ///
    /// Used only by [`gemm_expansion_rel_floor`]: the expansion
    /// `‖a‖² + ‖b‖² − 2·a·b` cancels, and how much it can cancel before the
    /// answer is gone is set by the accumulator's precision, not by `f64`'s.
    fn accum_eps() -> f64;
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
    #[inline]
    fn accum_eps() -> f64 {
        f32::EPSILON as f64
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
    #[inline]
    fn accum_eps() -> f64 {
        f64::EPSILON
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
///
/// A thin `f64` alias for the generic body below; for `F = f64` the
/// widening in the generic body monomorphises away, so this is the same machine
/// code it always was.
#[inline]
pub fn euclidean_distance(a: &[f64], b: &[f64], n_dims: usize) -> f64 {
    euclidean_distance_generic(a, b, n_dims, None)
}

/// Manhattan (L1) distance between two row vectors of length `n_dims`.
#[inline]
pub fn l1_distance(a: &[f64], b: &[f64], n_dims: usize) -> f64 {
    l1_distance_generic(a, b, n_dims, None)
}

/// Cosine distance between two row vectors: `1 - cosine_similarity(a, b)`.
///
/// Returns 1.0 (maximum distance) if either vector has zero norm, and clamps the
/// result to `[0, 2]`. Both conventions match sklearn's `cosine_distances`, which
/// is the reference every `eval_metrics` parity claim is measured against.
#[inline]
pub fn cosine_distance(a: &[f64], b: &[f64], n_dims: usize) -> f64 {
    cosine_distance_generic(a, b, n_dims, None)
}

/// Dispatch a point-to-point distance by metric enum.
#[inline]
pub fn point_distance(a: &[f64], b: &[f64], n_dims: usize, metric: DistanceMetric) -> f64 {
    point_distance_generic(a, b, n_dims, None, metric)
}

/// Dispatch a point-to-point distance over a subset of the dimensions.
///
/// `keep`, when `Some`, is a length-`n_dims` mask: dimension `k` participates iff
/// `keep[k]`. `None` means every dimension participates and is exactly
/// [`point_distance`].
///
/// This exists so a column-excluding caller does not need its own copy of the
/// three metric bodies. `discrimination.rs` had one, and it disagreed with this
/// module on two of cosine's edge conventions — a zero denominator gave `0.0`
/// (identical) instead of `1.0` (maximally distant), and the `[0, 2]` clamp was
/// absent. Those are the kind of divergence that only shows up on a fixture
/// nobody wrote.
///
/// A masked call is not slower per active dimension than an unmasked one: the
/// mask is resolved once, outside the accumulation loop, by monomorphising the
/// same body over two different index iterators.
#[inline]
pub fn point_distance_masked(
    a: &[f64],
    b: &[f64],
    n_dims: usize,
    keep: Option<&[bool]>,
    metric: DistanceMetric,
) -> f64 {
    point_distance_generic(a, b, n_dims, keep, metric)
}

// --- the gemm expansion, and where it stops being trustworthy ----------------

/// Safety factor on [`gemm_expansion_rel_floor`].
///
/// The error bound below is `√n_dims · ε` up to a small constant that depends on
/// the summation order the BLAS chose, which is not knowable here. `1.0` is kept
/// deliberately: the measured error on the reference case (32 dims, `‖x‖² ≈
/// 6.4e8`) is 2.75e+02 against a floor of 4.4e+02, so the bound holds with
/// margin, and raising this only widens the band of pairs that pay for an exact
/// recomputation they did not need.
const GEMM_EXPANSION_SAFETY: f64 = 1.0;

/// The relative floor below which `‖a‖² + ‖b‖² − 2·a·b` has lost the answer, in
/// units of `(‖a‖² + ‖b‖²)`.
///
/// The dot product is accumulated at the scale of the *norms*, so the absolute
/// error surviving into the difference is bounded by the norms' rounding —
/// roughly `√n_dims · ε · (‖a‖² + ‖b‖²)` — and not by `d²`'s own magnitude. Once
/// `d²` falls below that, every digit of it is noise, and the `.max(0.0)` clamp
/// that follows makes the resulting bias one-sided (§7.12).
///
/// Measured, on 24 rows of 32 f32 dims with `‖x‖² ≈ 6.4e8` and two pairs planted
/// at `d² = 2⁻¹⁰` and `2⁻⁸`: the expansion returns **exactly 0.0 for both**, so
/// two pairs an order of magnitude apart in distance report as identical. With
/// `f64` accumulation the same floor is ~2e-8 of the norms and never fires on
/// data like this, which is the point of keying it to the accumulator.
#[inline]
pub(crate) fn gemm_expansion_rel_floor(n_dims: usize, accum_eps: f64) -> f64 {
    GEMM_EXPANSION_SAFETY * (n_dims as f64).sqrt() * accum_eps
}

/// Squared Euclidean distance between two rows, accumulated in `f64` from the
/// stored values — no norms, no cancellation.
///
/// The fallback for [`gemm_expansion_rel_floor`]. `O(n_dims)` per pair, taken
/// only for pairs the expansion cannot resolve, which on real embeddings is the
/// near-duplicates and nothing else.
#[inline]
pub(crate) fn exact_sq_distance<F: PairwiseFloat>(a: &[F], b: &[F]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x.as_f64() - y.as_f64();
            d * d
        })
        .sum()
}

/// Expand a Gram entry into `d²`, recomputing exactly when the expansion has
/// fallen inside its own error floor.
///
/// `rel_floor` comes from [`gemm_expansion_rel_floor`] and is hoisted out of the
/// caller's inner loop; `exact` is called only on the rare pair that needs it.
#[inline]
pub(crate) fn expand_sq_distance(
    a_sq: f64,
    b_sq: f64,
    gram: f64,
    rel_floor: f64,
    exact: impl FnOnce() -> f64,
) -> f64 {
    let d_sq = a_sq + b_sq - 2.0 * gram;
    if d_sq < rel_floor * (a_sq + b_sq) {
        return exact().max(0.0);
    }
    d_sq
}

// ── Generic scalar distance helpers ─────────────────────────────────────
//
// These widen each element to `f64` before accumulation so f32 inputs do not
// lose precision in the inner reduction. For `F = f64` the `as_f64` calls
// monomorphise to no-ops, so these are equivalent to the existing f64 fns.

// Each metric's accumulation loop is written once, generic over the *index
// source*. `dims_unmasked` yields `0..n_dims`; `dims_masked` yields only the
// dimensions a `keep` mask selects. Because the accumulators are generic over
// `impl Iterator<Item = usize>`, the compiler emits a separate specialised loop
// for each — so the unmasked path carries no per-element mask check, and the
// masked path carries no duplicated source. This is the reason a caller that
// needs to exclude a column does not need its own copy of the three metrics.

#[inline]
fn dims_unmasked(n_dims: usize) -> impl Iterator<Item = usize> {
    0..n_dims
}

#[inline]
fn dims_masked<'m>(n_dims: usize, keep: &'m [bool]) -> impl Iterator<Item = usize> + 'm {
    (0..n_dims).filter(move |&k| keep[k])
}

#[inline]
fn euclidean_acc<F: PairwiseFloat, I: Iterator<Item = usize>>(a: &[F], b: &[F], dims: I) -> f64 {
    let mut sum_sq = 0.0f64;
    for k in dims {
        let diff = a[k].as_f64() - b[k].as_f64();
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

#[inline]
fn l1_acc<F: PairwiseFloat, I: Iterator<Item = usize>>(a: &[F], b: &[F], dims: I) -> f64 {
    let mut sum_abs = 0.0f64;
    for k in dims {
        sum_abs += (a[k].as_f64() - b[k].as_f64()).abs();
    }
    sum_abs
}

/// Accumulate the three cosine sums, then apply the two conventions that make
/// this module agree with sklearn's `cosine_distances`: a zero denominator is
/// maximal distance (`1.0`), and the result is clamped to `[0, 2]`.
#[inline]
fn cosine_acc<F: PairwiseFloat, I: Iterator<Item = usize>>(a: &[F], b: &[F], dims: I) -> f64 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for k in dims {
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
fn euclidean_distance_generic<F: PairwiseFloat>(
    a: &[F],
    b: &[F],
    n_dims: usize,
    keep: Option<&[bool]>,
) -> f64 {
    debug_assert!(a.len() >= n_dims && b.len() >= n_dims);
    debug_assert!(keep.is_none_or(|m| m.len() >= n_dims));
    match keep {
        None => euclidean_acc(a, b, dims_unmasked(n_dims)),
        Some(m) => euclidean_acc(a, b, dims_masked(n_dims, m)),
    }
}

#[inline]
fn l1_distance_generic<F: PairwiseFloat>(
    a: &[F],
    b: &[F],
    n_dims: usize,
    keep: Option<&[bool]>,
) -> f64 {
    debug_assert!(a.len() >= n_dims && b.len() >= n_dims);
    debug_assert!(keep.is_none_or(|m| m.len() >= n_dims));
    match keep {
        None => l1_acc(a, b, dims_unmasked(n_dims)),
        Some(m) => l1_acc(a, b, dims_masked(n_dims, m)),
    }
}

#[inline]
fn cosine_distance_generic<F: PairwiseFloat>(
    a: &[F],
    b: &[F],
    n_dims: usize,
    keep: Option<&[bool]>,
) -> f64 {
    debug_assert!(a.len() >= n_dims && b.len() >= n_dims);
    debug_assert!(keep.is_none_or(|m| m.len() >= n_dims));
    match keep {
        None => cosine_acc(a, b, dims_unmasked(n_dims)),
        Some(m) => cosine_acc(a, b, dims_masked(n_dims, m)),
    }
}

#[inline]
fn point_distance_generic<F: PairwiseFloat>(
    a: &[F],
    b: &[F],
    n_dims: usize,
    keep: Option<&[bool]>,
    metric: DistanceMetric,
) -> f64 {
    match metric {
        DistanceMetric::Euclidean => euclidean_distance_generic(a, b, n_dims, keep),
        DistanceMetric::L1 => l1_distance_generic(a, b, n_dims, keep),
        DistanceMetric::Cosine => cosine_distance_generic(a, b, n_dims, keep),
    }
}

/// Which pairs the gemm driver reduces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GemmShape {
    /// Every row of `a` against every row of `b`. `row_sums[i] = Σ_j d(a_i, b_j)`.
    Cross,
    /// `a` and `b` are the same buffer and only the strict upper triangle is
    /// summed: `row_sums[i] = Σ_{j>i} d(a_i, a_j)`, using the same convention
    /// the scalar self path uses. Always half the expansion of `Cross`; half the
    /// *gemm* only once blocked (see `mean_pairwise_distance_self`).
    UpperTriangle,
}

/// Squared row norms of a `[n × n_dims]` row-major matrix, in `f64`.
///
/// Widened from `F` so an f32 input does not lose the norm to accumulation
/// error on long rows.
///
/// This one keeps its `with_min_len`, unlike the four the distance loops used
/// to carry: each item here is only `O(n_dims)` work, so a floor genuinely damps
/// per-task overhead — and 64 sits *below* the iterator length at any size worth
/// parallelising, so it never forbids splitting outright.
fn row_sq_norms<F: PairwiseFloat>(data: &[F], n: usize, n_dims: usize) -> Vec<f64> {
    // `par_chunks_exact` panics on a zero chunk size, and `n_dims == 0` reaches
    // here from the public entry points (only `n_a == 0 || n_b == 0` short-
    // circuits). A zero-width row has a zero norm — which is what the previous
    // `(0..n).map(|i| data[i*0..(i+1)*0].sum())` produced — so answer that
    // directly rather than letting the chunker reject it.
    if n_dims == 0 {
        return vec![0.0; n];
    }
    data[..n * n_dims]
        .par_chunks_exact(n_dims)
        .with_min_len(64)
        .map(|row| {
            row.iter()
                .map(|&x| {
                    let xf = x.as_f64();
                    xf * xf
                })
                .sum::<f64>()
        })
        .collect()
}

/// Compute per-row distance sums for `(a, b)` via faer's gemm, **blocking the
/// rows of `a`** so the Gram never exists in full.
///
/// Returns a `Vec<f64>` of length `n_a`. The Gram is only ever read by a
/// per-row reduction and nothing downstream needs a second row, so it is
/// materialised one row block at a time: `block_rows × n_b` elements sized to
/// `budget` by [`crate::mem_budget::plan_gram_row_block`], reduced, then
/// overwritten by the next block. An input whose whole Gram fits the budget is
/// one block, i.e. exactly the unblocked computation.
///
/// `metric` must be `Euclidean` or `Cosine`; `L1` has no gemm formulation.
///
/// The matmul runs at the input precision `F` (so f32 inputs use a single-precision
/// gemm — the throughput win that motivates the f32 path) but the row-norm² and
/// final distance expansion always run in `f64`, so the reduction stays tight
/// against the `atol=1e-4` parity bound regardless of input precision.
///
/// The *reduction* order is unchanged by the blocking: each `row_sums[i]` is an
/// independent serial sum over ascending `j`, and the caller reduces the vector
/// in ascending `i`. What does move is the Gram's own last bits — the matmul
/// still runs under `Par::rayon(0)`, which was never bit-stable across thread
/// counts, and a block changes its `m` dimension, so faer blocks it differently.
/// Both are the same class of drift and both stay well below the `atol=1e-4`
/// parity tolerance; an input inside the budget is one block and so is
/// bit-identical to the unblocked kernel.
#[allow(clippy::too_many_arguments)]
fn pairwise_gemm_row_sums<F: PairwiseFloat>(
    a: &[F],
    b: &[F],
    n_a: usize,
    n_b: usize,
    n_dims: usize,
    metric: DistanceMetric,
    shape: GemmShape,
    budget: u64,
) -> Vec<f64> {
    debug_assert!(matches!(
        metric,
        DistanceMetric::Euclidean | DistanceMetric::Cosine
    ));
    debug_assert!(
        shape == GemmShape::Cross || (std::ptr::eq(a, b) && n_a == n_b),
        "UpperTriangle requires `a` and `b` to be the same buffer"
    );

    // For cosine, work on row-normalized copies. Then cosine distance is
    // simply `1 - (Ã · B̃ᵀ)[i,j]`. Zero-norm rows are left as zeros; their
    // dot product with anything is 0, giving distance 1.0 (matches the
    // scalar `cosine_distance` semantics). A self-distance passes the same
    // slice twice, so normalize it once and alias — the same `ptr::eq` check
    // `scx_gpu::gpu_mean_pairwise_distance` makes, and at 100 K × 2000 dims it
    // is an 800 MB copy not made.
    //
    // `n_a == n_b` is part of the test, not redundant with `ptr::eq`: a caller
    // may legally pass the same buffer with different row counts (`(a, a, 3,
    // 5)` = the first 3 rows against the first 5), and aliasing there would
    // index the shorter side's norms out of bounds.
    let is_self = std::ptr::eq(a, b) && n_a == n_b;
    let cosine = metric == DistanceMetric::Cosine;
    let a_norm: Option<Vec<F>> = cosine.then(|| normalize_rows(a, n_a, n_dims));
    let b_norm: Option<Vec<F>> = (cosine && !is_self).then(|| normalize_rows(b, n_b, n_dims));
    let a_data: &[F] = a_norm.as_deref().unwrap_or(a);
    let b_data: &[F] = b_norm.as_deref().or(a_norm.as_deref()).unwrap_or(b);

    // Squared row norms (Euclidean only). O(n) next to the O(n_a·n_b·n_dims)
    // main work, so they are computed once over the whole input rather than
    // per block. Aliased for a self-distance.
    let (a_sq, b_sq_owned) = match metric {
        DistanceMetric::Euclidean => (
            row_sq_norms(a_data, n_a, n_dims),
            if is_self {
                None
            } else {
                Some(row_sq_norms(b_data, n_b, n_dims))
            },
        ),
        _ => (Vec::new(), None),
    };
    let b_sq: &[f64] = b_sq_owned.as_deref().unwrap_or(&a_sq);

    // Hoisted: depends only on `n_dims` and the accumulator's precision, so it
    // is one multiply per pair inside the loop rather than a sqrt.
    let rel_floor = gemm_expansion_rel_floor(n_dims, F::accum_eps());

    let block_rows =
        crate::mem_budget::plan_gram_row_block(n_a, n_b, std::mem::size_of::<F>(), budget);

    // One buffer for every block, reused. `Accum::Replace` overwrites it, so it
    // never needs clearing — and allocating per block would memset the entire
    // `n_a × n_b` product over the run, which is the cost the blocking exists to
    // avoid. Sized for the widest block either shape asks for: `n_b` rows for
    // `Cross`, and for `UpperTriangle` the first block's extent, also `n_b`.
    let mut gram = Mat::<F>::zeros(n_b, block_rows);
    let mut row_sums = vec![0.0f64; n_a];

    let mut a0 = 0usize;
    while a0 < n_a {
        let a1 = (a0 + block_rows).min(n_a);
        let rows = a1 - a0;
        // Which rows of `b` this block needs. `Cross` touches all of them;
        // `UpperTriangle` needs only `j >= a0` — the narrowing that removes the
        // symmetric half of the work as the blocks advance.
        let (b0, extent) = match shape {
            GemmShape::Cross => (0usize, n_b),
            GemmShape::UpperTriangle => (a0, n_b - a0),
        };

        // Compute the Gram block as `B · A_blockᵀ` of shape `(extent, rows)`
        // rather than `A_block · Bᵀ`. faer's `Mat` is column-major, so the
        // inner reduction loop (fixed row, varying `j`) walks down a single
        // column — contiguous access. The transposed layout gives the same
        // values: `(B·Aᵀ)[j, i] = b_j · a_i = (A·Bᵀ)[i, j]`.
        let a_block =
            MatRef::<F>::from_row_major_slice(&a_data[a0 * n_dims..a1 * n_dims], rows, n_dims);
        let b_block = MatRef::<F>::from_row_major_slice(
            &b_data[b0 * n_dims..(b0 + extent) * n_dims],
            extent,
            n_dims,
        );
        matmul(
            gram.as_mut().submatrix_mut(0, 0, extent, rows),
            faer::Accum::Replace,
            b_block,
            a_block.transpose(),
            F::one(),
            faer::Par::rayon(0),
        );

        let g = gram.as_ref().submatrix(0, 0, extent, rows);
        row_sums[a0..a1]
            .par_iter_mut()
            .enumerate()
            .for_each(|(t, slot)| {
                // Local column `t` of the block is global row `a0 + t`; local
                // Gram row `jl` is global `b` row `b0 + jl`. For the triangle,
                // `b0 == a0`, so `j > i` is exactly `jl > t`.
                let j_start = match shape {
                    GemmShape::Cross => 0,
                    GemmShape::UpperTriangle => t + 1,
                };
                let mut row_sum = 0.0f64;
                match metric {
                    DistanceMetric::Euclidean => {
                        let ai_sq = a_sq[a0 + t];
                        let ai = &a_data[(a0 + t) * n_dims..(a0 + t + 1) * n_dims];
                        for jl in j_start..extent {
                            // Below `rel_floor` the expansion has cancelled away
                            // the answer, so recompute that pair exactly; the
                            // old `.max(0.0)` silently reported 0 instead
                            // (§7.12).
                            let gv = g[(jl, t)].as_f64();
                            let bj = b0 + jl;
                            let d_sq = expand_sq_distance(ai_sq, b_sq[bj], gv, rel_floor, || {
                                exact_sq_distance(ai, &b_data[bj * n_dims..(bj + 1) * n_dims])
                            });
                            row_sum += d_sq.sqrt();
                        }
                    }
                    DistanceMetric::Cosine => {
                        for jl in j_start..extent {
                            // Clamp to [0, 2] to match scalar cosine_distance.
                            row_sum += (1.0 - g[(jl, t)].as_f64()).clamp(0.0, 2.0);
                        }
                    }
                    DistanceMetric::L1 => unreachable!(),
                }
                *slot = row_sum;
            });

        a0 = a1;
    }

    row_sums
}

/// Allocate a row-normalized copy of `[n_rows × n_dims]` row-major matrix.
/// Zero-norm rows are left as zeros — `cosine_distance(zero, x) = 1.0` falls
/// out naturally from the dot product being zero.
///
/// The norm² accumulator runs in `f64` to avoid f32 overflow on rows with
/// many large entries; the inverse-norm scaling is then narrowed back to `F`
/// before applying to each element.
fn normalize_rows<F: PairwiseFloat>(data: &[F], n_rows: usize, n_dims: usize) -> Vec<F> {
    // `par_chunks_mut` panics on a zero chunk size. Zero-width rows have nothing
    // to normalize and the output is empty either way. (This one predates the
    // blocking work — `row_sq_norms` grew the same edge in the round-1 cleanup,
    // and the test that covers both found this on its way past.)
    if n_dims == 0 {
        return Vec::new();
    }
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
    mean_pairwise_distance_with_budget(
        a,
        b,
        n_a,
        n_b,
        n_dims,
        metric,
        backend,
        crate::mem_budget::pairwise_memory_budget(),
    )
}

/// [`mean_pairwise_distance`] with the Gram block budget injected rather than
/// read from the env.
///
/// The public entry reads `SCX_ACCEL_PAIRWISE_MEMORY_BUDGET` through a
/// `OnceLock`, which a test setting the env var would race into whichever test
/// ran first. Taking the budget as an argument lets the tests below force a
/// block count directly. Same shim shape as `mem_budget`'s own tests.
#[allow(clippy::too_many_arguments)]
fn mean_pairwise_distance_with_budget<F: PairwiseFloat>(
    a: &[F],
    b: &[F],
    n_a: usize,
    n_b: usize,
    n_dims: usize,
    metric: DistanceMetric,
    backend: DistanceBackend,
    budget: u64,
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
    // No `with_min_len` floor: every item here is Ω(n_b · n_dims) work, which
    // dwarfs rayon's split overhead. The floor this used to carry —
    // `(n_b * 16).max(1)`, justified as preventing oversubscription under
    // `fused_edistance` — exceeded the iterator length whenever `n_a < 32·n_b`,
    // which for a perturbation-vs-control call is always, so it did not damp
    // splitting, it forbade it. Nesting inside another `par_iter` on the same
    // global pool does not oversubscribe; it work-steals.
    let row_sums: Vec<f64> = match backend {
        DistanceBackend::Gemm => {
            pairwise_gemm_row_sums(a, b, n_a, n_b, n_dims, metric, GemmShape::Cross, budget)
        }
        DistanceBackend::Scalar => (0..n_a)
            .into_par_iter()
            .map(|i| {
                let row_a = &a[i * n_dims..(i + 1) * n_dims];
                let mut row_sum = 0.0f64;
                for j in 0..n_b {
                    let row_b = &b[j * n_dims..(j + 1) * n_dims];
                    row_sum += point_distance_generic(row_a, row_b, n_dims, None, metric);
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
/// The mean distance over all `n*(n-1)/2` distinct unordered pairs, taken over
/// the full `n × n` matrix's cell count: `sum(d(a_i, a_j) for i < j) * 2 / (n * n)`.
///
/// This matches `sklearn.metrics.pairwise_distances(a, a).mean()`, which forces
/// the self diagonal to 0 when `X is Y` rather than evaluating it. Note that
/// *evaluating* it would not always give 0 — under cosine, a zero-norm row
/// normalizes to zeros and its self-similarity is `1 - 0 = 1` — which is why
/// both backends skip the pair instead of adding it and trusting it to vanish.
///
/// Returns 0.0 if `n < 2`.
pub fn mean_pairwise_distance_self<F: PairwiseFloat>(
    a: &[F],
    n: usize,
    n_dims: usize,
    metric: DistanceMetric,
    backend: DistanceBackend,
) -> crate::Result<f64> {
    mean_pairwise_distance_self_with_budget(
        a,
        n,
        n_dims,
        metric,
        backend,
        crate::mem_budget::pairwise_memory_budget(),
    )
}

/// [`mean_pairwise_distance_self`] with the Gram block budget injected rather
/// than read from the env. See [`mean_pairwise_distance_with_budget`].
fn mean_pairwise_distance_self_with_budget<F: PairwiseFloat>(
    a: &[F],
    n: usize,
    n_dims: usize,
    metric: DistanceMetric,
    backend: DistanceBackend,
    budget: u64,
) -> crate::Result<f64> {
    if n < 2 {
        return Ok(0.0);
    }
    debug_assert!(a.len() >= n * n_dims);

    let backend = resolve_backend(backend, metric)?;

    // Both backends sum the strict upper triangle. The full n×n matrix holds
    // each of those pairs twice, plus n diagonal cells that sklearn forces to 0
    // (they are *not* evaluated — see the note above about zero-norm rows), so:
    //   full_sum = 2 * total
    //   mean     = full_sum / (n * n)
    // which is what `sklearn.metrics.pairwise_distances(a, a).mean()` returns.
    let row_sums: Vec<f64> = match backend {
        DistanceBackend::Gemm => {
            // This used to re-enter the (a, a) cross path, which computed the
            // whole n×n Gram — the mirror half for nothing, plus a diagonal that
            // is only zero on well-behaved input (a cosine zero-norm row's is
            // 1.0). `UpperTriangle` narrows each block's columns operand to
            // `a[a0..]`. That always halves the expansion loop; it reduces the
            // *gemm* only once there is more than one block, since the first
            // block's operand is still all of `a` — total gemm work is
            // `≈ (k+1)/2k` of the square at `k` blocks. Measured 1.03× at one
            // block and 1.38× at eight (docs/performance/perturbation-metrics.md).
            pairwise_gemm_row_sums(a, a, n, n, n_dims, metric, GemmShape::UpperTriangle, budget)
        }
        DistanceBackend::Scalar => {
            // Row `i` has `n - i - 1` pairs, so the work is triangular and
            // rayon's adaptive splitting is exactly what balances it. The
            // floor this used to carry, `with_min_len((n * 16).max(1))`, was
            // never satisfiable — `16n > n` for every `n` — so it did not damp
            // splitting, it disabled it, and this ran single-threaded at every
            // pool size. Reduction stays sequential over the per-row sums for
            // bit-stable output across thread counts.
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let row_i = &a[i * n_dims..(i + 1) * n_dims];
                    let mut row_sum = 0.0f64;
                    for j in (i + 1)..n {
                        let row_j = &a[j * n_dims..(j + 1) * n_dims];
                        row_sum += point_distance_generic(row_i, row_j, n_dims, None, metric);
                    }
                    row_sum
                })
                .collect()
        }
        DistanceBackend::Auto => unreachable!("resolve_backend collapses Auto"),
    };
    let total: f64 = row_sums.iter().sum();
    Ok(2.0 * total / (n as f64 * n as f64))
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

    // ── Gram blocking ─────────────────────────────────────────────────

    /// Deterministic points in `[-1, 1)`, no `rand` dependency.
    fn pts<F: PairwiseFloat>(n: usize, d: usize, seed: u64) -> Vec<F> {
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n * d)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                F::from_f64(((s >> 40) as f64 / (1u64 << 24) as f64) * 2.0 - 1.0)
            })
            .collect()
    }

    /// Budget that admits exactly `rows` rows of `a` per Gram block.
    fn budget_for<F>(rows: usize, n_b: usize) -> u64 {
        (rows as u64) * (n_b as u64) * (std::mem::size_of::<F>() as u64)
    }

    /// A masked call must equal the unmasked call on the physically compacted
    /// vectors — for every metric, and for an all-true mask.
    ///
    /// This is the property that lets a column-excluding caller drop its own copy
    /// of the three metric bodies. Asserting it against a compacted reference
    /// rather than against constants is what makes it catch a divergence in
    /// *either* direction: a masked path that forgets the cosine clamp, and an
    /// unmasked path that drifts from it.
    #[test]
    fn masking_equals_compacting_the_vectors() {
        let a = [1.0f64, -2.0, 0.5, 7.0, -0.25];
        let b = [0.0f64, 3.0, 0.5, -1.0, 2.0];
        let n = a.len();

        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::L1,
            DistanceMetric::Cosine,
        ] {
            // All-true mask is exactly the unmasked call.
            let all = vec![true; n];
            let masked_all = point_distance_masked(&a, &b, n, Some(&all), metric);
            let unmasked = point_distance(&a, &b, n, metric);
            assert!(
                (masked_all - unmasked).abs() < 1e-15,
                "{metric:?}: all-true mask gave {masked_all}, unmasked gives {unmasked}"
            );
            // `None` is the same thing spelled without a mask.
            let none = point_distance_masked(&a, &b, n, None, metric);
            assert!(
                (none - unmasked).abs() < 1e-15,
                "{metric:?}: None mask gave {none}, unmasked gives {unmasked}"
            );

            // Every proper subset: mask == compact-then-unmasked.
            for drop_bits in 1u32..(1 << n) {
                let keep: Vec<bool> = (0..n).map(|k| drop_bits & (1 << k) == 0).collect();
                let ca: Vec<f64> = (0..n).filter(|&k| keep[k]).map(|k| a[k]).collect();
                let cb: Vec<f64> = (0..n).filter(|&k| keep[k]).map(|k| b[k]).collect();
                let got = point_distance_masked(&a, &b, n, Some(&keep), metric);
                let want = point_distance(&ca, &cb, ca.len(), metric);
                assert!(
                    (got - want).abs() < 1e-12,
                    "{metric:?} keep={keep:?}: masked {got} != compacted {want}"
                );
            }
        }
    }

    /// The zero-norm and clamp conventions, stated once so a future edit to
    /// either kernel has something to break.
    ///
    /// `1.0` for a zero denominator and a `[0, 2]` range are sklearn's
    /// `cosine_distances` conventions, and every `eval_metrics` cell-eval parity
    /// claim rests on matching them. Checked through the masked entry point too,
    /// because that is the path that used to disagree.
    #[test]
    fn cosine_zero_norm_and_range_hold_on_both_entry_points() {
        let zeros = [0.0f64, 0.0, 0.0];
        let v = [1.0f64, 2.0, 3.0];
        let w = [0.0f64, 5.0, 0.0];
        for (label, d) in [
            ("unmasked zero/zero", cosine_distance(&zeros, &zeros, 3)),
            ("unmasked zero/v", cosine_distance(&zeros, &v, 3)),
            (
                "masked zero/v",
                // Mask nothing out; the zero norm is intrinsic.
                point_distance_masked(
                    &zeros,
                    &v,
                    3,
                    Some(&[true, true, true]),
                    DistanceMetric::Cosine,
                ),
            ),
            (
                "masked-to-zero",
                // `w` is nonzero ONLY at index 1, and index 1 is the one the mask
                // drops -- so the kept view of `w` is [0.0, 0.0]. Masking to a
                // column that still holds a nonzero (the obvious mistake, and the
                // one the first draft of this test made) proves nothing.
                point_distance_masked(
                    &w,
                    &w,
                    3,
                    Some(&[true, false, true]),
                    DistanceMetric::Cosine,
                ),
            ),
        ] {
            assert!(
                (d - 1.0).abs() < 1e-15,
                "{label}: got {d}, want 1.0 (a zero denominator is MAXIMAL distance)"
            );
        }

        // Anti-parallel is the top of the range, and nothing exceeds it.
        let anti = point_distance(&[1.0, 0.0], &[-1.0, 0.0], 2, DistanceMetric::Cosine);
        assert!(
            (0.0..=2.0).contains(&anti) && (anti - 2.0).abs() < 1e-15,
            "anti-parallel cosine distance is {anti}, want exactly 2.0 within [0, 2]"
        );
    }

    #[test]
    fn blocking_does_not_change_the_cross_answer() {
        // Shapes chosen so the last block is full, one row short, and one row
        // over — the three ways a block loop gets its boundary wrong.
        for (n_a, n_b, n_d) in [(64usize, 40usize, 9usize), (65, 40, 9), (63, 40, 9)] {
            let a: Vec<f64> = pts(n_a, n_d, 3);
            let b: Vec<f64> = pts(n_b, n_d, 11);
            for metric in [DistanceMetric::Euclidean, DistanceMetric::Cosine] {
                let whole = mean_pairwise_distance_with_budget(
                    &a,
                    &b,
                    n_a,
                    n_b,
                    n_d,
                    metric,
                    DistanceBackend::Gemm,
                    u64::MAX,
                )
                .unwrap();
                for rows in [1usize, 2, 7, 16, n_a - 1, n_a] {
                    let blocked = mean_pairwise_distance_with_budget(
                        &a,
                        &b,
                        n_a,
                        n_b,
                        n_d,
                        metric,
                        DistanceBackend::Gemm,
                        budget_for::<f64>(rows, n_b),
                    )
                    .unwrap();
                    assert!(
                        (whole - blocked).abs() < 1e-12,
                        "{metric:?} {n_a}x{n_b}: {rows}-row blocks gave {blocked}, \
                         one block gives {whole}"
                    );
                }
            }
        }
    }

    #[test]
    fn blocked_cross_matches_the_scalar_backend() {
        // The oracle is the row-by-row formulation, not another run of the same
        // gemm — a blocking bug that also corrupted the unblocked path would
        // survive `blocking_does_not_change_the_cross_answer`.
        let (n_a, n_b, n_d) = (37usize, 23usize, 8usize);
        let a: Vec<f64> = pts(n_a, n_d, 5);
        let b: Vec<f64> = pts(n_b, n_d, 17);
        for metric in [DistanceMetric::Euclidean, DistanceMetric::Cosine] {
            let scalar =
                mean_pairwise_distance(&a, &b, n_a, n_b, n_d, metric, DistanceBackend::Scalar)
                    .unwrap();
            for rows in [1usize, 5, 36] {
                let blocked = mean_pairwise_distance_with_budget(
                    &a,
                    &b,
                    n_a,
                    n_b,
                    n_d,
                    metric,
                    DistanceBackend::Gemm,
                    budget_for::<f64>(rows, n_b),
                )
                .unwrap();
                assert!(
                    (scalar - blocked).abs() < 1e-10,
                    "{metric:?}: {rows}-row blocks gave {blocked}, scalar gives {scalar}"
                );
            }
        }
    }

    #[test]
    fn blocked_cross_matches_the_scalar_backend_f32() {
        // f32 is the documented pyscx default, and it is the precision the
        // blocking has to hold at — a wider block sums the same pairs, but the
        // gemm runs at F.
        let (n_a, n_b, n_d) = (48usize, 31usize, 12usize);
        let a: Vec<f32> = pts(n_a, n_d, 21);
        let b: Vec<f32> = pts(n_b, n_d, 29);
        for metric in [DistanceMetric::Euclidean, DistanceMetric::Cosine] {
            let scalar =
                mean_pairwise_distance(&a, &b, n_a, n_b, n_d, metric, DistanceBackend::Scalar)
                    .unwrap();
            for rows in [1usize, 7, n_a] {
                let blocked = mean_pairwise_distance_with_budget(
                    &a,
                    &b,
                    n_a,
                    n_b,
                    n_d,
                    metric,
                    DistanceBackend::Gemm,
                    budget_for::<f32>(rows, n_b),
                )
                .unwrap();
                assert!(
                    (scalar - blocked).abs() < 1e-6,
                    "{metric:?} f32: {rows}-row blocks gave {blocked}, scalar gives {scalar}"
                );
            }
        }
    }

    #[test]
    fn gemm_self_triangle_matches_the_scalar_backend() {
        // The self path narrows each block's columns operand to `a[a0..]` and
        // sums only `j > i`. Its diagonal block is the part that gets written
        // wrong: `n` is swept across block boundaries so that the diagonal
        // lands at the start, middle and end of a block.
        for n in [2usize, 3, 17, 64, 65] {
            let n_d = 7;
            let a: Vec<f64> = pts(n, n_d, 41);
            for metric in [DistanceMetric::Euclidean, DistanceMetric::Cosine] {
                let scalar =
                    mean_pairwise_distance_self(&a, n, n_d, metric, DistanceBackend::Scalar)
                        .unwrap();
                for rows in [1usize, 2, 5, 16, n] {
                    let gemm = mean_pairwise_distance_self_with_budget(
                        &a,
                        n,
                        n_d,
                        metric,
                        DistanceBackend::Gemm,
                        budget_for::<f64>(rows, n),
                    )
                    .unwrap();
                    assert!(
                        (scalar - gemm).abs() < 1e-10,
                        "{metric:?} n={n}: {rows}-row blocks gave {gemm}, scalar gives {scalar}"
                    );
                }
            }
        }
    }

    #[test]
    fn gemm_self_triangle_counts_every_pair_exactly_once() {
        // Premise for the test above: if the triangle dropped or double-counted
        // pairs, agreement with the scalar path would still be possible for a
        // fixture whose pairwise distances are all equal. These are not — the
        // three pairs of a 3-point fixture are 5, sqrt(2) and sqrt(13), so any
        // miscount moves the mean.
        //
        // The `j_start = t` vs `t + 1` boundary is *not* covered by this
        // fixture — every row here has a nonzero norm, so the diagonal it would
        // add is exactly 0 under both gemm metrics. It is covered by
        // `gemm_self_skips_the_diagonal_of_a_zero_norm_row` below, where a
        // zero-norm row makes the cosine diagonal 1.0 and the two spellings
        // differ by a whole unit.
        let a = vec![0.0, 0.0, 3.0, 4.0, 1.0, 1.0];
        let expected = 2.0 * (5.0 + 2.0f64.sqrt() + 13.0f64.sqrt()) / 9.0;
        for rows in [1usize, 2, 3] {
            let gemm = mean_pairwise_distance_self_with_budget(
                &a,
                3,
                2,
                DistanceMetric::Euclidean,
                DistanceBackend::Gemm,
                budget_for::<f64>(rows, 3),
            )
            .unwrap();
            assert!(
                (gemm - expected).abs() < 1e-12,
                "{rows}-row blocks gave {gemm}, hand-computed value is {expected}"
            );
        }
    }

    #[test]
    fn a_zero_width_input_does_not_panic() {
        // Regression: swapping `row_sq_norms` to `par_chunks_exact(n_dims)`
        // turned `n_dims == 0` from "every row has norm 0" into a panic, on a
        // shape the public entry points accept (only `n_a == 0 || n_b == 0`
        // short-circuits). Zero-width points are all identical, so every
        // distance is 0.
        let a: Vec<f64> = vec![];
        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::Cosine,
            DistanceMetric::L1,
        ] {
            for backend in [DistanceBackend::Gemm, DistanceBackend::Scalar] {
                if metric == DistanceMetric::L1 && backend == DistanceBackend::Gemm {
                    continue; // rejected by resolve_backend, tested elsewhere
                }
                let cross = mean_pairwise_distance(&a, &a, 2, 2, 0, metric, backend).unwrap();
                let selfd = mean_pairwise_distance_self(&a, 2, 0, metric, backend).unwrap();
                // Cosine's zero-norm rule gives 1.0 per pair; euclidean/L1 give 0.
                let expect_cross = if metric == DistanceMetric::Cosine {
                    1.0
                } else {
                    0.0
                };
                assert_eq!(cross, expect_cross, "{metric:?}/{backend:?} cross");
                assert_eq!(
                    selfd,
                    expect_cross / 2.0,
                    "{metric:?}/{backend:?} self (triangle over n^2)"
                );
            }
        }
    }

    #[test]
    fn gemm_self_skips_the_diagonal_of_a_zero_norm_row() {
        // A zero-norm row is where "the self diagonal is exactly 0" stops being
        // true. `normalize_rows` leaves such a row as zeros, so its cosine Gram
        // diagonal is `1 - 0 = 1`, not 1 - 1 = 0. Two all-zero rows:
        //
        //   strict upper triangle : 2 * 1 / 4     = 0.5   (scalar, and gemm now)
        //   full n*n square       : 4 * 1 / 4     = 1.0   (gemm before this change)
        //
        // The triangle is the right answer: it is what the scalar backend has
        // always returned, and what `sklearn.metrics.pairwise.cosine_distances`
        // returns, since that forces the self diagonal to 0 when `X is Y`.
        //
        // This is also the fixture that makes `j_start = t` vs `t + 1`
        // detectable — with nonzero rows the diagonal contributes 0 and the two
        // spellings are indistinguishable.
        for (label, a, n) in [
            ("two zero rows", vec![0.0f64; 6], 2usize),
            (
                "one zero row among unit rows",
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                3,
            ),
        ] {
            let scalar = mean_pairwise_distance_self(
                &a,
                n,
                3,
                DistanceMetric::Cosine,
                DistanceBackend::Scalar,
            )
            .unwrap();
            let gemm = mean_pairwise_distance_self(
                &a,
                n,
                3,
                DistanceMetric::Cosine,
                DistanceBackend::Gemm,
            )
            .unwrap();
            assert!(
                (scalar - gemm).abs() < 1e-12,
                "{label}: gemm self {gemm} != scalar self {scalar}"
            );

            // Premise: the full-square convention really does differ here, so
            // the equality above is not something every implementation passes.
            let full_square = mean_pairwise_distance(
                &a,
                &a,
                n,
                n,
                3,
                DistanceMetric::Cosine,
                DistanceBackend::Gemm,
            )
            .unwrap();
            assert!(
                (full_square - gemm).abs() > 1e-3,
                "{label}: the full n*n square gave {full_square}, same as the triangle's \
                 {gemm} — this fixture has no zero-norm row and proves nothing"
            );
        }
    }

    #[test]
    fn the_same_buffer_with_different_row_counts_is_not_a_self_distance() {
        // `(a, a, n_a, n_b)` with `n_a != n_b` means "the first n_a rows against
        // the first n_b rows" — legal, and *not* a self-distance. Keying the
        // norm/normalized-copy aliasing on `ptr::eq` alone would reuse the
        // shorter side's buffers for the longer one and index out of bounds.
        let (n_a, n_b, n_d) = (3usize, 7usize, 5usize);
        let a: Vec<f64> = pts(n_b, n_d, 71);
        for metric in [DistanceMetric::Euclidean, DistanceMetric::Cosine] {
            let expect =
                mean_pairwise_distance(&a, &a, n_a, n_b, n_d, metric, DistanceBackend::Scalar)
                    .unwrap();
            let got = mean_pairwise_distance(&a, &a, n_a, n_b, n_d, metric, DistanceBackend::Gemm)
                .unwrap();
            assert!(
                (expect - got).abs() < 1e-12,
                "{metric:?}: {got} vs {expect}"
            );
        }
    }

    #[test]
    fn a_one_row_block_is_the_floor_not_a_hang() {
        // `plan_gram_row_block` never returns 0, so a budget smaller than one
        // row's Gram still terminates — with one row per block.
        let (n, n_d) = (9usize, 4usize);
        let a: Vec<f64> = pts(n, n_d, 61);
        let scalar = mean_pairwise_distance_self(
            &a,
            n,
            n_d,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();
        let gemm = mean_pairwise_distance_self_with_budget(
            &a,
            n,
            n_d,
            DistanceMetric::Euclidean,
            DistanceBackend::Gemm,
            1, // one byte
        )
        .unwrap();
        assert!((scalar - gemm).abs() < 1e-12, "{gemm} vs {scalar}");
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
