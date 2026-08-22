//! Per-column mean/variance from raw moments — the one Bessel-corrected
//! finalize in the tree.
//!
//! # Why this lives in `scx-sparse`
//!
//! Before Phase 7a the `(Σx² − n·mean²)/(n−1)` finalize was written out seven
//! times: four in `scx-accel`'s `hvg/cpu.rs`, one in its `csc/mean_var.rs`, and
//! two in `scx-gpu`'s `gpu_hvg.rs`. `scx-accel` depends on `scx-gpu` (optional
//! `gpu` feature) and never the reverse, so no primitive in either of them can
//! be reached from the other. `scx-sparse` is the only crate below both — and it
//! already hosts [`crate::total_variance_from_col_sq`], the whole-matrix
//! analogue of this function, which `scx-gpu`'s PCA already calls.
//!
//! # Two conventions this settles, which used to be per-site
//!
//! **The clamp.** Five of the seven sites clamped with `if v < 0.0 { 0.0 }` and
//! two (the batched HVG arms) with `.max(0.0)`. Those are *not* equivalent:
//! Rust's `f64::max` returns the non-NaN operand, so `.max(0.0)` maps a NaN
//! variance silently to `0.0`, while `if v < 0.0` is false for NaN and lets it
//! propagate. This function takes the second behaviour — a NaN is a bug worth
//! seeing, not worth rounding to a plausible zero — and
//! `nan_variance_is_not_clamped_to_zero` pins it. Both forms were measured to be
//! bit-identical on real data, because every caller sits behind a finiteness
//! guard on its input; the divergence is reachable only by calling this
//! function directly.
//!
//! **The cancellation report.** `Σx² − n·mean²` is the cancellation-prone form.
//! It is used anyway because the stable two-pass alternative needs a second
//! pass over the data, and for a sparse column that pass cannot be skipped the
//! way the first one can (Welford is not an option either: it needs every
//! element in sequence *including the implicit zeros*, which is `O(n_obs·n_vars)`
//! rather than `O(nnz)`). What changes here is that the loss is **reported**
//! rather than absorbed: [`ColumnMoments::unstable`] names the columns whose
//! subtraction left fewer than ~7 significant digits, so a caller can warn
//! instead of returning a confidently wrong small number.
//!
//! The stable-form counterpart for callers that *can* afford the second pass is
//! [`crate::finalize_implicit_zero_variance`] — note it is `ddof = 0`
//! (population), where everything here is `ddof = 1` (sample), matching scanpy's
//! `mean_var(correction=1)`.

/// Relative floor below which `Σx² − n·mean²` has lost too much precision to
/// cancellation for the result to be trusted.
///
/// `1e-7` of the second moment: roughly "fewer than seven significant digits
/// survived the subtraction". Shared with the PCA total-variance guard
/// ([`closed_form_variance_unstable`]) so the two cannot drift — one threshold,
/// one definition.
pub const CLOSED_FORM_VAR_REL_EPS: f64 = 1e-7;

/// Per-column mean and variance, plus the columns whose variance is suspect.
#[derive(Debug, Clone, Default)]
pub struct ColumnMoments {
    /// Per-column mean, `Σx / n` (length = the number of columns).
    pub means: Vec<f64>,
    /// Per-column sample variance, `(Σx² − n·mean²)/(n−1)`, negatives clamped
    /// to zero. Same length as `means`.
    pub variances: Vec<f64>,
    /// Indices of columns where the closed form lost more than
    /// [`CLOSED_FORM_VAR_REL_EPS`] of its precision to cancellation. Empty in
    /// the overwhelmingly common case, and empty is not an allocation.
    ///
    /// A column whose second moment is exactly zero is **not** listed: its
    /// variance is exactly zero, not a cancelled non-zero.
    pub unstable: Vec<u32>,
}

/// Whether a `Σx² − n·mean²`-style residual has lost too much of its input's
/// magnitude to cancellation to be trusted.
///
/// `residual` is the difference; `second_moment` is the positive quantity it was
/// subtracted from. Returns `false` when `second_moment` is zero or negative:
/// there is nothing to cancel, so an exactly-zero residual is an exact answer,
/// not a lost one. That exclusion is load-bearing — without it every all-zero
/// column in a sparse matrix reports as unstable.
///
/// The single predicate behind all three cancellation guards in the tree: the
/// per-column one in [`finalize_column_moments`], the whole-matrix one in
/// [`closed_form_variance_unstable`], and `scx-accel`'s `pflog_total_variance`.
#[inline]
pub fn residual_lost_to_cancellation(residual: f64, second_moment: f64) -> bool {
    second_moment > 0.0 && residual <= CLOSED_FORM_VAR_REL_EPS * second_moment
}

impl ColumnMoments {
    /// Log the columns whose variance lost too much precision to cancellation.
    ///
    /// `Σx² − n·mean²` is used rather than a stable two-pass form because the
    /// stable form needs a second pass over the data, which for an out-of-core
    /// HVG is a full extra I/O pass on every call, to guard against an input
    /// almost nobody has. The compromise: keep the one-pass form and **say so**
    /// when it fails, instead of returning a confidently wrong small variance. A
    /// column listed here is near-constant at a large magnitude; its reported
    /// variance may carry only a handful of significant digits.
    ///
    /// Deliberately a log line and not an error. One such gene among 30k is not
    /// grounds for refusing an HVG call, and every caller of these kernels treats
    /// them as infallible but for input validation. It lives here rather than in
    /// `scx-accel` so the two `scx-gpu` callers get it too — the reason the
    /// primitive is in this crate at all.
    pub fn warn_if_unstable(&self, op: &str) {
        if self.unstable.is_empty() {
            return;
        }
        let shown: Vec<u32> = self.unstable.iter().copied().take(8).collect();
        let truncated = if self.unstable.len() > shown.len() {
            " (truncated)"
        } else {
            ""
        };
        log::warn!(
            "{op}: {} of the per-column variances lost precision to cancellation in \
             `sum(x^2) - n*mean^2` (near-constant columns at large magnitude); first \
             affected column indices: {shown:?}{truncated}",
            self.unstable.len(),
        );
    }
}

/// Bessel-corrected (`ddof = 1`) per-column mean and variance from raw moments.
///
/// `col_sum[j] = Σ x_ij` and `col_sum_sq[j] = Σ x_ij²` over **all** `n` rows —
/// which for a sparse column means the implicit zeros are already accounted for,
/// since they contribute nothing to either sum. `n` is the row count the moments
/// were accumulated over: the full `n_obs` for a whole-matrix reduction, or the
/// group's cell count for a per-batch one.
///
/// Returns zeros (and no `unstable` entries) when `n == 0`. Divides by
/// `max(n − 1, 1)`, so `n == 1` yields the zero-variance answer rather than an
/// infinity.
///
/// # Panics
///
/// If `col_sum` and `col_sum_sq` have different lengths. This is a caller bug,
/// not malformed data: every caller allocates both as `vec![0.0; n_vars]` in the
/// same breath, so a mismatch cannot arise from input. Panicking rather than
/// zipping is deliberate — a `zip` would truncate to the shorter slice and
/// return a short `Vec` whose entries a caller indexing by column would read as
/// a *different* column's variance. Same reasoning as
/// [`crate::finalize_implicit_zero_variance`]'s length check, which reports it
/// as an error because that function already has an error channel; this one is
/// called from three crates with three unrelated error types.
pub fn finalize_column_moments(col_sum: &[f64], col_sum_sq: &[f64], n: usize) -> ColumnMoments {
    assert_eq!(
        col_sum.len(),
        col_sum_sq.len(),
        "finalize_column_moments: col_sum and col_sum_sq must describe the same columns"
    );
    let n_cols = col_sum.len();
    if n == 0 {
        return ColumnMoments {
            means: vec![0.0; n_cols],
            variances: vec![0.0; n_cols],
            unstable: Vec::new(),
        };
    }

    let nf = n as f64;
    let denom = (nf - 1.0).max(1.0);
    let mut means = Vec::with_capacity(n_cols);
    let mut variances = Vec::with_capacity(n_cols);
    let mut unstable = Vec::new();

    for j in 0..n_cols {
        let mean = col_sum[j] / nf;
        means.push(mean);
        // Kept in exactly this form — `col_sum_sq[j] - n * mean * mean` — and not
        // fused into a `mul_add`. The two are algebraically identical and
        // numerically are not: on the golden fixture in
        // `scx-accel/src/hvg/moments_golden_tests.rs` the FMA form moves a
        // variance from 0.11111111110949423 to 0.11111111111040017. That test
        // pins bits precisely so this line cannot be "optimised" silently.
        let residual = col_sum_sq[j] - nf * mean * mean;
        // A zero second moment means an all-zero column: exactly zero variance,
        // nothing cancelled. Checking it first keeps such columns out of
        // `unstable`, where a bare relative test would put every one of them.
        if residual_lost_to_cancellation(residual, col_sum_sq[j]) {
            unstable.push(j as u32);
        }
        let var = residual / denom;
        // NaN falls through deliberately; see the module header.
        variances.push(if var < 0.0 { 0.0 } else { var });
    }

    ColumnMoments {
        means,
        variances,
        unstable,
    }
}

/// Whether the whole-matrix closed-form centered variance has lost too much
/// precision to cancellation for the caller to use it.
///
/// The matrix-wide counterpart of [`finalize_column_moments`]'s per-column
/// `unstable` test, and the guard on
/// [`crate::total_variance_from_col_sq`]'s centered result. Only meaningful when
/// centering: the uncentered path sums `col_sum_sq` with no subtraction and has
/// nothing to cancel.
///
/// Catches both `≤ 0` (a sign-flipped total) and tiny-positive garbage. Callers
/// that can afford a second pass over the data should recompute via the stable
/// centered form when this returns `true`.
pub fn closed_form_variance_unstable(col_sum_sq: &[f64], means: &[f64], n_obs: usize) -> bool {
    let sum_sq: f64 = col_sum_sq.iter().sum();
    let mean_sq: f64 = means.iter().map(|&m| m * m).sum::<f64>() * n_obs as f64;
    // Note the asymmetry with the per-column test: here a zero `sum_sq` *is*
    // reported unstable, because a whole matrix with no second moment has no
    // usable total variance for `variance_ratio` to divide by either way.
    sum_sq <= 0.0 || residual_lost_to_cancellation(sum_sq - mean_sq, sum_sq)
}

#[cfg(test)]
#[path = "moments_tests.rs"]
mod tests;
