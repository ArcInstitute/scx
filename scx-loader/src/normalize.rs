//! Fused normalize + log1p operations for dense rows.
//!
//! Port of `scx-engine/src/fused_ops.rs` adapted for the dense row format
//! used in the training pipeline. In the loader, normalization operates on
//! the dense output row (not sparse CSR) after the scatter step.
//!
//! The math is identical to scanpy's `normalize_total` + `log1p` but the
//! data layout differs: these functions operate on `&mut [f32]` dense rows
//! where zero values are explicitly stored.

/// Normalize a dense row to a target sum using an **explicit depth** as the
/// denominator (in-place).
///
/// `row[i] = row[i] / depth * target_sum`
///
/// `depth` is the cell's total count over the **full transcriptome** — NOT
/// necessarily `row.iter().sum()`. When `row` is an HVG-projected panel, its
/// own sum is the panel-local depth, which is a different statistic from
/// scanpy's `normalize_total` (computed over full depth, then subset to HVGs)
/// and from the pflog sibling ([`pflog_baseline_row`], full `D`).
/// Passing the full-row depth here keeps the panel path consistent with both.
///
/// Zero values remain zero (0.0 × factor = 0.0). If `depth <= 0.0` the row is
/// left unchanged (no division by zero).
pub fn normalize_dense_row_with_depth(row: &mut [f32], target_sum: f64, depth: f64) {
    if depth > 0.0 {
        let factor = target_sum / depth;
        for v in row.iter_mut() {
            *v = (*v as f64 * factor) as f32;
        }
    }
}

/// Normalize a dense row to a target sum (in-place), deriving the depth from
/// the row's own sum.
///
/// Equivalent to `scanpy.pp.normalize_total` for one row:
///   `row[i] = row[i] / row_sum * target_sum`
///
/// Convenience wrapper over [`normalize_dense_row_with_depth`] for callers
/// whose `row` **is** the full transcriptome (no HVG projection). Projected
/// callers must use [`normalize_dense_row_with_depth`] with the full-row depth.
///
/// # Arguments
/// - `row`: Dense row of f32 values to normalize in-place.
/// - `target_sum`: Target sum for normalization (e.g., 1e4).
pub fn normalize_dense_row(row: &mut [f32], target_sum: f64) {
    let row_sum: f64 = row.iter().map(|&v| v as f64).sum();
    normalize_dense_row_with_depth(row, target_sum, row_sum);
}

/// Apply `ln(x + 1)` to each element of a dense row (in-place), clipping
/// negatives to zero first.
///
/// Equivalent to `numpy.log1p` on non-negative input. Zeros map to
/// `ln(1.0) = 0.0`, preserving sparsity in the dense representation.
///
/// # Why the clip
///
/// `ln` is undefined below `-1` and singular at it, so an unguarded
/// `(v + 1.0).ln()` puts `NaN` (or `-inf`) into the batch for any stored value
/// `<= -1` and nothing downstream can attribute it. The sparse loader has
/// clipped unconditionally since it and the collate kernel were found to
/// disagree about what a row contained (see
/// [`crate::downsample::clip_negatives`]); this is the same clip, applied where
/// the dense loaders would otherwise produce `NaN`.
///
/// Scoped deliberately to the paths that take a log. `apply_dense_transforms`'
/// no-log arms are untouched, so a caller feeding a centered or scaled matrix
/// through with no transforms still gets its signed values back.
///
/// `f32::max` returns the non-`NaN` operand, so a stored `NaN` maps to `0` —
/// matching the collate kernel's `raw.max(0.0)` reads exactly.
///
/// # Arguments
/// - `row`: Dense row of f32 values to transform in-place.
pub fn log1p_dense_row(row: &mut [f32]) {
    for v in row.iter_mut() {
        *v = (v.max(0.0) + 1.0).ln();
    }
}

/// Fused normalize + log1p for a dense row using an **explicit depth** as the
/// normalization denominator (single pass, in-place).
///
/// Computes `row[i] = ln(row[i] * target_sum / depth + 1.0)` in a single pass.
/// See [`normalize_dense_row_with_depth`] for why `depth` is the full-row
/// depth rather than `row.iter().sum()` on the HVG-projected path.
///
/// Zeros produce `ln(0.0 * factor + 1.0) = ln(1.0) = 0.0`, staying zero.
/// Negatives (and `NaN`) are clipped to zero before the log, for the reason
/// [`log1p_dense_row`] documents.
///
/// `depth <= 0.0` skips the **scale** — there is nothing to divide by — but the
/// log still runs. Previously the whole body was skipped, so a row whose values
/// sum to zero or below came back untransformed even though the caller asked
/// for log1p; an all-zero row is unaffected either way, since `log1p(0) == 0`.
pub fn fused_normalize_log1p_dense_with_depth(row: &mut [f32], target_sum: f64, depth: f64) {
    let factor = if depth > 0.0 { target_sum / depth } else { 1.0 };
    for v in row.iter_mut() {
        // Scale in f64 for precision, cast to f32, then f32 ln for speed.
        // Numerically matches sequential normalize→log1p path.
        *v = (((*v as f64 * factor) as f32).max(0.0) + 1.0).ln();
    }
}

/// Fused normalize + log1p for a dense row (single pass, in-place), deriving
/// the depth from the row itself.
///
/// Computes `row[i] = ln(max(row[i], 0) * target_sum / depth + 1.0)` in a
/// single pass, avoiding two separate traversals of the dense row. `depth` is
/// [`transform_depth`] over this row with `log1p = true`, i.e. the sum of the
/// **clipped** values — not `row.iter().sum()`, which would let a negative
/// that is about to be discarded inflate every gene that survives it, and would
/// let one `NaN` skip normalization for the whole row.
///
/// Convenience wrapper over [`fused_normalize_log1p_dense_with_depth`] for
/// callers whose `row` **is** the full transcriptome (no HVG projection).
///
/// **Numerical note**: The fused version may produce slightly different f32
/// results compared to sequential normalize-then-log1p due to intermediate
/// rounding. Tests allow f32 epsilon tolerance (~1e-6 relative).
///
/// # Arguments
/// - `row`: Dense row of f32 values to transform in-place.
/// - `target_sum`: Target sum for normalization (e.g., 1e4).
pub fn fused_normalize_log1p_dense(row: &mut [f32], target_sum: f64) {
    let depth = transform_depth(row, /*log1p=*/ true);
    fused_normalize_log1p_dense_with_depth(row, target_sum, depth);
}

/// The normalization denominator for a row, over the values the transform will
/// actually see.
///
/// `csr_data` is the cell's **full pre-projection** stored nonzeros — see
/// [`normalize_dense_row_with_depth`] for why the panel-local sum is the wrong
/// statistic under an HVG projection.
///
/// # Why `log1p` changes the answer
///
/// On the log paths the values are clipped to zero before the log
/// ([`log1p_dense_row`]), so a denominator that still counts the pre-clip
/// negatives does not describe the row being normalized. Concretely, a row
/// `[-1, 3]` at `target_sum = 1e4` has raw depth `2`, giving a factor of `5000`
/// and `log1p(15000)` for the surviving gene — a **positive** gene silently
/// scaled by 1.5× because of a value that was then discarded. Clipping the
/// denominator too gives depth `3` and `log1p(10000)`, which is what
/// clip-then-normalize means and what the sparse gather already does (it clips
/// in `transform_row` before deriving its depth).
///
/// A stored `NaN` is the sharper case: it makes the raw sum `NaN`, `depth > 0.0`
/// is then false, and normalization is skipped for **every** otherwise-valid
/// value in that cell. Clipping maps it to `0` and the rest of the row
/// normalizes as it should.
///
/// When `log1p` is false the raw, signed sum is returned unchanged. That path
/// scales without taking a log, nothing is undefined, and a caller streaming a
/// pre-centered matrix through `normalize=True, log1p=False` still gets the
/// denominator its data implies.
#[inline]
pub fn transform_depth(csr_data: &[f32], log1p: bool) -> f64 {
    if log1p {
        csr_data.iter().map(|&v| v.max(0.0) as f64).sum()
    } else {
        csr_data.iter().map(|&v| v as f64).sum()
    }
}

/// Apply the configured dense-row transforms in-place.
///
/// Single dispatch point shared by `TrainingDataset` (`decode_stage.rs`)
/// and `IndexPlanDataset` (`index_plan.rs`) so both honour all four
/// `(normalize, log1p)` combinations identically.
///
/// `LoaderConfig::validate()` rejects non-positive `target_sum` at the
/// pipeline boundary, but the helper itself does not consult `target_sum`
/// when `normalize == false` — callers that disable normalization may
/// pass any value (e.g. `0.0`) without affecting the output.
///
/// # Arguments
/// - `row`: Dense row of f32 values to transform in-place.
/// - `normalize`: Whether to apply total-count normalization.
/// - `log1p`: Whether to apply `ln(x + 1)` element-wise.
/// - `target_sum`: Target sum for normalization (ignored if `normalize` is false).
/// - `depth`: The normalization denominator — the cell's total count over the
///   **full transcriptome**. Callers whose `row` is HVG-projected MUST pass the
///   full pre-projection row sum here, not the panel-local sum, so normalization
///   matches scanpy's normalize-then-subset and the pflog path. Ignored when
///   `normalize == false`.
///
///   **Derive it with [`transform_depth`], not `row.iter().sum()`.** When
///   `log1p` is set the values are clipped before the log, so a raw sum
///   describes a row that will not exist by the time it is divided: a discarded
///   negative inflates every surviving gene, and a single `NaN` makes the whole
///   sum `NaN` and skips normalization for the entire cell. `transform_depth`
///   clips exactly when a log follows and returns the signed sum otherwise, so
///   the normalize-only path is unaffected.
#[inline]
pub fn apply_dense_transforms(
    row: &mut [f32],
    normalize: bool,
    log1p: bool,
    target_sum: f64,
    depth: f64,
) {
    match (normalize, log1p) {
        (true, true) => fused_normalize_log1p_dense_with_depth(row, target_sum, depth),
        (true, false) => normalize_dense_row_with_depth(row, target_sum, depth),
        (false, true) => log1p_dense_row(row),
        (false, false) => {}
    }
}

/// Per-cell PFlog (v4) baseline from a sparse CSR row's stored values.
///
/// PFlog (v4, Booeshaghi et al.) is the shifted-log transform on **raw counts**
/// `z_ij = log1p(4α·x_ij) − (1/D)·Σ_k log1p(4α·x_ik)`, where `4α = four_alpha`
/// and the matrix-wide Anscombe pseudocount is `1/(4α)`. The `delta =
/// log1p(4α·x)` part is sparse (zeros stay zero) and the per-cell `baseline =
/// −(1/D)·Σ delta` is the value every original zero collapses to. Unlike v2
/// there is **no per-cell depth** (it cancels under the Anscombe scale).
///
/// **Critical**: `csr_data` MUST be the **full pre-projection** row and `n_vars`
/// the **full** feature count `D`. Computing `D` over an HVG-projected subset is
/// a different statistic that silently diverges from the analysis path
/// (`scx_accel::pflog_*`, which use full `D`).
///
/// Returns `None` only for `n_vars == 0` (degenerate). An empty cell yields
/// `Some(0.0)` (v4 is representable at zero depth). Accumulates in `f64`.
///
/// Values are clipped to zero before the `ln_1p`, as on every other log path
/// (see [`log1p_dense_row`]). It matters more here than anywhere else: this is
/// a **sum**, so one stored value below `-1/(4α)` would make the baseline `NaN`
/// and every gene in that cell `NaN` with it.
#[inline]
pub fn pflog_baseline_row(csr_data: &[f32], four_alpha: f64, n_vars: usize) -> Option<f64> {
    if n_vars == 0 {
        return None;
    }
    let sum_delta: f64 = csr_data
        .iter()
        .map(|&v| (four_alpha * v.max(0.0) as f64).ln_1p())
        .sum();
    Some(-sum_delta / n_vars as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- normalize_dense_row tests ----

    #[test]
    fn test_normalize_matches_scanpy() {
        // Row: [0, 5, 0, 10, 0], row_sum = 15, target = 10000
        // After normalize: [0, 5/15*10000, 0, 10/15*10000, 0]
        //                = [0, 3333.333, 0, 6666.667, 0]
        let mut row = vec![0.0, 5.0, 0.0, 10.0, 0.0];
        let target_sum = 10_000.0;

        normalize_dense_row(&mut row, target_sum);

        let expected_1 = (5.0_f64 / 15.0 * target_sum) as f32;
        let expected_3 = (10.0_f64 / 15.0 * target_sum) as f32;

        assert_eq!(row[0], 0.0);
        assert!((row[1] - expected_1).abs() < 1e-3);
        assert_eq!(row[2], 0.0);
        assert!((row[3] - expected_3).abs() < 1e-3);
        assert_eq!(row[4], 0.0);
    }

    #[test]
    fn test_normalize_row_sums_to_target() {
        let mut row = vec![1.0, 3.0, 0.0, 7.0, 2.0];
        let target = 1.0;

        normalize_dense_row(&mut row, target);

        let row_sum: f64 = row.iter().map(|&v| v as f64).sum();
        assert!((row_sum - 1.0).abs() < 1e-6, "row sum = {row_sum}");
    }

    // ---- log1p_dense_row tests ----

    #[test]
    fn test_log1p_matches_numpy() {
        // numpy.log1p([0, 5, 0, 10, 0]) = [0, ln(6), 0, ln(11), 0]
        let mut row = vec![0.0, 5.0, 0.0, 10.0, 0.0];

        log1p_dense_row(&mut row);

        assert!((row[0] - 0.0).abs() < 1e-7); // ln(1) = 0
        assert!((row[1] - (6.0_f32).ln()).abs() < 1e-6);
        assert!((row[2] - 0.0).abs() < 1e-7);
        assert!((row[3] - (11.0_f32).ln()).abs() < 1e-6);
        assert!((row[4] - 0.0).abs() < 1e-7);
    }

    // ---- fused_normalize_log1p_dense tests ----

    #[test]
    fn test_fused_matches_sequential() {
        let target = 10_000.0;

        // Sequential path
        let mut sequential = vec![0.0, 5.0, 0.0, 10.0, 0.0, 1.0, 3.0, 7.0, 2.0];
        normalize_dense_row(&mut sequential, target);
        log1p_dense_row(&mut sequential);

        // Fused path
        let mut fused = vec![0.0, 5.0, 0.0, 10.0, 0.0, 1.0, 3.0, 7.0, 2.0];
        fused_normalize_log1p_dense(&mut fused, target);

        for i in 0..fused.len() {
            assert!(
                (fused[i] - sequential[i]).abs() < 1e-5,
                "mismatch at [{}]: fused={}, seq={}",
                i,
                fused[i],
                sequential[i]
            );
        }
    }

    #[test]
    fn test_all_zero_row_unchanged() {
        let mut row = vec![0.0, 0.0, 0.0, 0.0];

        // normalize: row_sum=0 → no change
        normalize_dense_row(&mut row, 10_000.0);
        assert!(
            row.iter().all(|&v| v == 0.0),
            "normalize should leave zeros unchanged"
        );

        // log1p: ln(0+1)=0
        log1p_dense_row(&mut row);
        assert!(row.iter().all(|&v| v == 0.0), "log1p(0) should be 0");

        // fused: row_sum=0 → no change
        let mut row2 = vec![0.0, 0.0, 0.0];
        fused_normalize_log1p_dense(&mut row2, 10_000.0);
        assert!(
            row2.iter().all(|&v| v == 0.0),
            "fused should leave zeros unchanged"
        );
    }

    // -----------------------------------------------------------------
    // Negatives and NaN on the log paths
    //
    // `ln` is undefined below -1, so an unguarded `(v + 1.0).ln()` put NaN in
    // the batch for any stored value <= -1 — silently, and only on the dense
    // loaders, since the sparse gather has clipped unconditionally for a while.
    // -----------------------------------------------------------------

    #[test]
    fn log1p_clips_negatives_instead_of_emitting_nan() {
        let mut row = vec![-5.0f32, -1.0, -0.5, 0.0, 3.0];
        log1p_dense_row(&mut row);

        assert!(
            row.iter().all(|v| v.is_finite()),
            "no NaN or inf may reach the batch: {row:?}"
        );
        assert_eq!(
            &row[..4],
            &[0.0, 0.0, 0.0, 0.0],
            "negatives clip to log1p(0)"
        );
        assert!(
            (row[4] - 4.0f32.ln()).abs() < 1e-6,
            "positives are untouched"
        );
    }

    /// `f32::max` returns the non-NaN operand, so NaN maps to 0 — the same
    /// semantics `downsample::clip_negatives` documents and the collate
    /// kernel's `raw.max(0.0)` reads use.
    #[test]
    fn log1p_maps_nan_to_zero_like_the_sparse_clip() {
        let mut row = vec![f32::NAN, f32::NEG_INFINITY];
        log1p_dense_row(&mut row);
        assert_eq!(row, vec![0.0, 0.0]);

        let mut fused = vec![f32::NAN, f32::NEG_INFINITY, 4.0];
        fused_normalize_log1p_dense_with_depth(&mut fused, 10.0, 4.0);
        assert!(fused.iter().all(|v| v.is_finite()), "{fused:?}");
        assert_eq!(&fused[..2], &[0.0, 0.0]);
    }

    #[test]
    fn fused_clips_negatives_instead_of_emitting_nan() {
        let mut row = vec![-8.0f32, 0.0, 2.0];
        fused_normalize_log1p_dense_with_depth(&mut row, 10.0, 2.0);
        assert!(row.iter().all(|v| v.is_finite()), "{row:?}");
        assert_eq!(&row[..2], &[0.0, 0.0]);
    }

    /// A row whose values sum to zero or below used to skip the transform
    /// entirely — `depth <= 0.0` returned before the loop — so a caller who
    /// asked for log1p got raw values back. Now only the *scale* is skipped.
    #[test]
    fn fused_still_logs_when_the_depth_is_non_positive() {
        let mut row = vec![-9.0f32, 3.0];
        fused_normalize_log1p_dense_with_depth(&mut row, 10.0, -6.0);
        assert_eq!(row[0], 0.0, "the negative clips");
        assert!(
            (row[1] - 4.0f32.ln()).abs() < 1e-6,
            "the positive is logged unscaled, not passed through raw: {row:?}"
        );
    }

    /// The clip is scoped to the log paths on purpose. Someone feeding a
    /// centered or scaled matrix through with no transforms still gets their
    /// signed values back — pinning that this fix did not quietly become
    /// "the loader zeroes negatives".
    #[test]
    fn no_transform_still_passes_negatives_through() {
        let mut row = vec![-2.5f32, 0.0, 1.5];
        apply_dense_transforms(&mut row, false, false, 0.0, 0.0);
        assert_eq!(row, vec![-2.5, 0.0, 1.5]);

        // normalize-only likewise: it scales, it does not log.
        let mut scaled = vec![-2.0f32, 4.0];
        apply_dense_transforms(&mut scaled, true, false, 2.0, 2.0);
        assert_eq!(scaled, vec![-2.0, 4.0]);
    }

    /// The clip has to reach the **denominator**, not only the values.
    ///
    /// A row `[-1, 3]` has raw depth 2 but clipped depth 3. Normalizing by 2
    /// scales the surviving positive gene to 15000 instead of 10000 — a valid
    /// gene mis-scaled 1.5x by a value that is then discarded. Found by
    /// codex - gpt-5.6-sol.
    #[test]
    fn the_log_paths_normalize_by_the_clipped_depth() {
        let row = [-1.0f32, 3.0];
        assert_eq!(transform_depth(&row, /*log1p=*/ true), 3.0);
        assert_eq!(transform_depth(&row, /*log1p=*/ false), 2.0);

        let mut got = row;
        apply_dense_transforms(
            &mut got,
            /*normalize=*/ true,
            /*log1p=*/ true,
            10_000.0,
            transform_depth(&row, true),
        );
        assert_eq!(got[0], 0.0);
        assert!(
            (got[1] - 10_000.0f32.ln_1p()).abs() < 1e-3,
            "expected log1p(10000)={}, got {}",
            10_000.0f32.ln_1p(),
            got[1]
        );
    }

    /// A single stored `NaN` made the raw sum `NaN`, so `depth > 0.0` was false
    /// and normalization was skipped for **every** valid value in that cell.
    /// Found by codex - gpt-5.6-sol.
    #[test]
    fn a_nan_does_not_disable_normalization_for_the_whole_cell() {
        let row = [f32::NAN, 4.0];
        let depth = transform_depth(&row, /*log1p=*/ true);
        assert_eq!(depth, 4.0, "NaN must contribute 0, not poison the sum");

        let mut got = row;
        apply_dense_transforms(&mut got, true, true, 10_000.0, depth);
        assert_eq!(got[0], 0.0);
        assert!(
            (got[1] - 10_000.0f32.ln_1p()).abs() < 1e-3,
            "the valid gene must still be normalized: {got:?}"
        );
    }

    /// Normalize-only keeps the signed denominator — the deliberately
    /// unchanged path. Without this the fix above could quietly become
    /// "the loader always clips".
    #[test]
    fn normalize_only_keeps_the_signed_depth() {
        let row = [-1.0f32, 3.0];
        let mut got = row;
        apply_dense_transforms(
            &mut got,
            /*normalize=*/ true,
            /*log1p=*/ false,
            10_000.0,
            transform_depth(&row, false),
        );
        assert_eq!(got, [-5000.0, 15000.0]);
    }

    #[test]
    fn pflog_baseline_is_finite_on_a_row_containing_a_negative() {
        let baseline = pflog_baseline_row(&[-3.0, 1.0, 2.0], 0.4, 8).unwrap();
        assert!(
            baseline.is_finite(),
            "one bad value must not NaN the whole cell's baseline"
        );
        // The negative contributes ln1p(0) == 0, i.e. exactly as if absent.
        let without = pflog_baseline_row(&[1.0, 2.0], 0.4, 8).unwrap();
        assert!((baseline - without).abs() < 1e-12);
    }

    #[test]
    fn test_single_nonzero_normalize() {
        // Single non-zero value: after normalize, should equal target_sum
        let mut row = vec![0.0, 0.0, 42.0, 0.0];
        normalize_dense_row(&mut row, 10_000.0);

        assert!(
            (row[2] - 10_000.0).abs() < 1e-3,
            "single value should become target_sum"
        );
        assert_eq!(row[0], 0.0);
        assert_eq!(row[1], 0.0);
        assert_eq!(row[3], 0.0);
    }

    #[test]
    fn test_single_nonzero_fused() {
        let mut row = vec![0.0, 42.0, 0.0];
        fused_normalize_log1p_dense(&mut row, 10_000.0);

        // ln(42/42 * 10000 + 1) = ln(10001)
        let expected = (10_001.0_f64).ln() as f32;
        assert!((row[1] - expected).abs() < 1e-3);
        // Zero values → ln(0 * factor + 1) = ln(1) = 0
        assert!((row[0] - 0.0).abs() < 1e-7);
        assert!((row[2] - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_fused_zeros_stay_zero() {
        // Verify that zero values produce exactly 0.0 after fused operation
        let mut row = vec![0.0, 5.0, 0.0, 0.0, 10.0, 0.0];
        fused_normalize_log1p_dense(&mut row, 1e4);

        assert!((row[0] - 0.0).abs() < 1e-7);
        assert!((row[2] - 0.0).abs() < 1e-7);
        assert!((row[3] - 0.0).abs() < 1e-7);
        assert!((row[5] - 0.0).abs() < 1e-7);
        // Non-zeros should be positive
        assert!(row[1] > 0.0);
        assert!(row[4] > 0.0);
    }

    #[test]
    fn test_empty_row() {
        let mut row: Vec<f32> = vec![];
        normalize_dense_row(&mut row, 10_000.0);
        log1p_dense_row(&mut row);
        fused_normalize_log1p_dense(&mut row, 10_000.0);
        assert!(row.is_empty());
    }

    // ---- apply_dense_transforms (4-way dispatch) ----

    fn sample_row() -> Vec<f32> {
        vec![0.0, 5.0, 0.0, 10.0, 0.0, 1.0, 3.0, 7.0, 2.0]
    }

    #[test]
    fn test_apply_normalize_and_log1p_matches_fused() {
        let target = 1e4_f64;
        let mut a = sample_row();
        let depth: f64 = a.iter().map(|&v| v as f64).sum();
        apply_dense_transforms(&mut a, true, true, target, depth);

        let mut b = sample_row();
        fused_normalize_log1p_dense(&mut b, target);

        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-7);
        }
    }

    #[test]
    fn test_apply_normalize_only_matches_normalize_dense_row() {
        let target = 1e4_f64;
        let mut a = sample_row();
        let depth: f64 = a.iter().map(|&v| v as f64).sum();
        apply_dense_transforms(&mut a, true, false, target, depth);

        let mut b = sample_row();
        normalize_dense_row(&mut b, target);

        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-7);
        }
        // Sanity: row sums to target_sum (no log1p applied).
        let s: f64 = a.iter().map(|&v| v as f64).sum();
        assert!((s - target).abs() < 1e-2);
    }

    #[test]
    fn test_apply_log1p_only_matches_log1p_dense_row() {
        let mut a = sample_row();
        apply_dense_transforms(
            &mut a, false, true, /*ignored*/ 1.0, /*depth ignored*/ 0.0,
        );

        let mut b = sample_row();
        log1p_dense_row(&mut b);

        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-7);
        }
    }

    #[test]
    fn test_apply_no_transform_is_identity() {
        let mut a = sample_row();
        let original = a.clone();
        apply_dense_transforms(&mut a, false, false, 0.0, 0.0);
        assert_eq!(a, original);
    }

    // ---- explicit-depth normalization (L2 fix) ----

    /// The normalization denominator is the supplied `depth`, NOT the row's own
    /// sum. This is what makes the HVG-projected loader path normalize by full
    /// transcriptome depth (scanpy's normalize-then-subset), not the panel sum.
    #[test]
    fn test_normalize_with_depth_uses_explicit_depth() {
        // Panel row (a 2-gene HVG subset). Its own sum is 15, but the cell's
        // full-transcriptome depth is 100 (expression outside the panel).
        let mut row = vec![5.0f32, 10.0];
        let full_depth = 100.0_f64;
        let target = 1e4_f64;
        normalize_dense_row_with_depth(&mut row, target, full_depth);

        // Scaled by target/full_depth = 100, NOT target/15.
        assert!((row[0] - 500.0).abs() < 1e-2, "got {}", row[0]);
        assert!((row[1] - 1000.0).abs() < 1e-2, "got {}", row[1]);

        // And it differs from the panel-local (self-sum) result.
        let mut panel_local = vec![5.0f32, 10.0];
        normalize_dense_row(&mut panel_local, target);
        assert!(
            (panel_local[0] - row[0]).abs() > 1.0,
            "explicit full-depth must diverge from panel-local normalize"
        );
    }

    /// `apply_dense_transforms` threads the explicit depth through to the fused
    /// path identically to a direct `*_with_depth` call.
    #[test]
    fn test_apply_dense_transforms_honors_depth() {
        let target = 1e4_f64;
        let full_depth = 250.0_f64;

        let mut a = vec![0.0f32, 5.0, 0.0, 10.0];
        apply_dense_transforms(&mut a, true, true, target, full_depth);

        let mut b = vec![0.0f32, 5.0, 0.0, 10.0];
        fused_normalize_log1p_dense_with_depth(&mut b, target, full_depth);

        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-7);
        }
    }
}
