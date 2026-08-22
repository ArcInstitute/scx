//! Unit tests for the one column-moments finalize.
//!
//! These cover what the `scx-accel` golden fixture *cannot*: the golden test
//! pins end-to-end output through five kernels, but every one of them sits
//! behind a finiteness guard on its input, so the behaviours this module
//! exercises — a NaN variance, a mismatched slice pair, a genuinely cancelled
//! column — are reachable only by calling the primitive directly.

use super::*;

#[test]
fn matches_the_hand_computed_sample_variance() {
    // col0 = [1, 2, 3, 4]: mean 2.5, var = (1.5² + 0.5² + 0.5² + 1.5²)/3 = 5/3.
    // col1 = [0, 0, 0, 0]: mean 0, var 0.
    let m = finalize_column_moments(&[10.0, 0.0], &[30.0, 0.0], 4);
    assert_eq!(m.means, vec![2.5, 0.0]);
    assert!((m.variances[0] - 5.0 / 3.0).abs() < 1e-15);
    assert_eq!(m.variances[1], 0.0);
}

#[test]
fn zero_rows_gives_zeros_not_a_division_by_zero() {
    let m = finalize_column_moments(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], 0);
    assert_eq!(m.means, vec![0.0; 3]);
    assert_eq!(m.variances, vec![0.0; 3]);
    assert!(m.unstable.is_empty());
}

#[test]
fn one_row_gives_zero_variance_not_infinity() {
    // ddof=1 with n=1 divides by zero; the `.max(1.0)` floor makes it 0/1.
    let m = finalize_column_moments(&[3.0], &[9.0], 1);
    assert_eq!(m.means, vec![3.0]);
    assert_eq!(m.variances, vec![0.0]);
    assert!(m.variances[0].is_finite());
}

#[test]
fn a_negative_residual_from_round_off_is_clamped_to_zero() {
    // A constant column: Σx² = n·mean² exactly in theory, and a hair either way
    // in practice. Feed a residual that is negative by construction.
    let n = 4;
    let mean = 7.0;
    let m = finalize_column_moments(&[mean * n as f64], &[n as f64 * mean * mean - 1e-9], n);
    assert_eq!(
        m.variances,
        vec![0.0],
        "a negative residual must not surface"
    );
}

/// The clamp divergence Phase 7a settles.
///
/// The two batched HVG sites used `.max(0.0)`, which returns the non-NaN operand
/// and therefore maps a NaN variance to a plausible-looking `0.0`. The other five
/// used `if v < 0.0`, which is false for NaN and lets it through. This function
/// takes the second behaviour on purpose: a NaN here means the accumulators are
/// already wrong, and reporting zero variance for a broken gene is worse than
/// reporting NaN.
///
/// **Not reachable from data.** Every caller runs a finiteness guard over the
/// shard values first (`ensure_finite_hvg_data` on the CSR path,
/// `ensure_finite_values` on the CSC path), and no finite f32 input can drive the
/// f64 accumulators to NaN. So this is a unit test on the primitive by necessity,
/// not by preference — do not go looking for the end-to-end case, and do not
/// "fix" an input guard to create one.
#[test]
fn nan_variance_is_not_clamped_to_zero() {
    let m = finalize_column_moments(&[f64::NAN], &[f64::NAN], 4);
    assert!(m.means[0].is_nan(), "a NaN sum must give a NaN mean");
    assert!(
        m.variances[0].is_nan(),
        "a NaN variance must propagate, not become 0.0 — got {}",
        m.variances[0]
    );
    // And the mirror image, so the assertion above is not passing because
    // *everything* comes back NaN: `.max(0.0)`'s behaviour is what we rejected.
    assert_eq!(f64::NAN.max(0.0), 0.0, "premise: f64::max swallows NaN");
}

#[test]
#[should_panic(expected = "must describe the same columns")]
fn mismatched_slice_lengths_panic_rather_than_truncate() {
    // A `zip` would return a 1-element Vec here, and a caller indexing by column
    // would read column 1's variance out of column 0's slot.
    finalize_column_moments(&[1.0, 2.0], &[1.0], 4);
}

/// A column whose variance survives the subtraction is not flagged, and one
/// whose variance is destroyed by it is.
#[test]
fn unstable_flags_only_the_cancelled_columns() {
    let n = 9usize;
    let nf = n as f64;

    // Column A: ordinary sparse gene, one nonzero of 7 in nine cells. The
    // residual is ~89% of the second moment — nothing lost.
    let a_sum = 7.0;
    let a_sq = 49.0;

    // Column B: constant 1e8 in every cell. mean² · n and Σx² agree to within
    // f64's last bits, so the residual is pure noise.
    let b_sum = 1e8 * nf;
    let b_sq = 1e16 * nf;

    // Column C: all implicit zeros. Exactly zero variance, nothing cancelled —
    // must NOT be flagged, which a bare relative test would get wrong (0 <= 0).
    let m = finalize_column_moments(&[a_sum, b_sum, 0.0], &[a_sq, b_sq, 0.0], n);
    assert_eq!(
        m.unstable,
        vec![1u32],
        "expected only the constant-1e8 column flagged, got {:?} (variances {:?})",
        m.unstable,
        m.variances
    );
    assert_eq!(m.variances[2], 0.0);
}

/// The whole-matrix guard agrees with itself on the two cases it exists for.
#[test]
fn closed_form_guard_catches_sign_flip_and_tiny_positive() {
    // A well-conditioned matrix: Σx² far above n·Σμ².
    assert!(!closed_form_variance_unstable(
        &[100.0, 100.0],
        &[1.0, 1.0],
        4
    ));
    // A degenerate one: the subtraction lands at or below zero.
    assert!(closed_form_variance_unstable(&[4.0, 4.0], &[1.0, 1.0], 4));
    // And tiny-positive garbage, which a `<= 0.0` test alone would miss.
    let sum_sq: f64 = 1.0e6;
    let mean = (sum_sq / 4.0 * (1.0 - 1e-12) / 2.0).sqrt();
    assert!(closed_form_variance_unstable(
        &[sum_sq / 2.0, sum_sq / 2.0],
        &[mean, mean],
        4
    ));
}
