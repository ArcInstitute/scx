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
/// and from the pflog1ppf sibling ([`pflog1ppf_depth_baseline`], full `s_i`).
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

/// Apply `ln(x + 1)` to each element of a dense row (in-place).
///
/// Equivalent to `numpy.log1p`. Zeros map to `ln(1.0) = 0.0`, preserving
/// sparsity in the dense representation.
///
/// # Arguments
/// - `row`: Dense row of f32 values to transform in-place.
pub fn log1p_dense_row(row: &mut [f32]) {
    for v in row.iter_mut() {
        *v = (*v + 1.0).ln();
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
/// `depth <= 0.0` leaves the row unchanged (all-zero → still zero).
pub fn fused_normalize_log1p_dense_with_depth(row: &mut [f32], target_sum: f64, depth: f64) {
    if depth > 0.0 {
        let factor = target_sum / depth;
        for v in row.iter_mut() {
            // Scale in f64 for precision, cast to f32, then f32 ln for speed.
            // Numerically matches sequential normalize→log1p path.
            *v = ((*v as f64 * factor) as f32 + 1.0).ln();
        }
    }
}

/// Fused normalize + log1p for a dense row (single pass, in-place), deriving
/// the depth from the row's own sum.
///
/// Computes `row[i] = ln(row[i] * target_sum / row_sum + 1.0)` in a single
/// pass, avoiding two separate traversals of the dense row.
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
    let row_sum: f64 = row.iter().map(|&v| v as f64).sum();
    fused_normalize_log1p_dense_with_depth(row, target_sum, row_sum);
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
///   matches scanpy's normalize-then-subset and the pflog1ppf path. For a
///   full-transcriptome `row` this equals `row.iter().sum()`. Ignored when
///   `normalize == false`.
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

/// Per-cell PFlog1pPF depth and baseline from a sparse CSR row's stored values.
///
/// PFlog1pPF (Booeshaghi et al. 2026) is the shifted centered-log-ratio
/// transform `z_ij = log1p(x_ij/(c·s_i)) − (1/D)·Σ_k log1p(x_ik/(c·s_i))`. The
/// `delta = log1p(x/(c·s))` part is sparse (zeros stay zero) and the per-cell
/// `baseline = −(1/D)·Σ delta` is the value every original zero collapses to.
///
/// **Critical**: `csr_data` MUST be the **full pre-projection** row and `n_vars`
/// the **full** feature count `D`. Computing depth/`D` over an HVG-projected
/// subset is a different statistic that silently diverges from the analysis
/// path (`scx_accel::pflog_*`, which use full `s_i` and full `D`).
///
/// Returns `None` for a non-positive depth (empty cell) or `n_vars == 0` —
/// callers leave the output row zeroed (mirrors [`normalize_dense_row`]'s
/// `if row_sum > 0.0` guard). Accumulates in `f64`.
#[inline]
pub fn pflog1ppf_depth_baseline(csr_data: &[f32], c: f64, n_vars: usize) -> Option<(f64, f64)> {
    let depth: f64 = csr_data.iter().map(|&v| v as f64).sum();
    if depth <= 0.0 || n_vars == 0 {
        return None;
    }
    let inv = 1.0 / (c * depth);
    let sum_delta: f64 = csr_data.iter().map(|&v| (v as f64 * inv).ln_1p()).sum();
    Some((depth, -sum_delta / n_vars as f64))
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
