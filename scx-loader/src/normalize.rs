//! Fused normalize + log1p operations for dense rows.
//!
//! Port of `scx-engine/src/fused_ops.rs` adapted for the dense row format
//! used in the training pipeline. In the loader, normalization operates on
//! the dense output row (not sparse CSR) after the scatter step.
//!
//! The math is identical to scanpy's `normalize_total` + `log1p` but the
//! data layout differs: these functions operate on `&mut [f32]` dense rows
//! where zero values are explicitly stored.

/// Normalize a dense row to a target sum (in-place).
///
/// Equivalent to `scanpy.pp.normalize_total` for one row:
///   `row[i] = row[i] / row_sum * target_sum`
///
/// Zero values remain zero (0.0 × factor = 0.0). If the row sum is zero,
/// the row is left unchanged (no division by zero).
///
/// # Arguments
/// - `row`: Dense row of f32 values to normalize in-place.
/// - `target_sum`: Target sum for normalization (e.g., 1e4).
pub fn normalize_dense_row(row: &mut [f32], target_sum: f64) {
    let row_sum: f64 = row.iter().map(|&v| v as f64).sum();
    if row_sum > 0.0 {
        let factor = target_sum / row_sum;
        for v in row.iter_mut() {
            *v = (*v as f64 * factor) as f32;
        }
    }
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

/// Fused normalize + log1p for a dense row (single pass, in-place).
///
/// Computes `row[i] = ln(row[i] * target_sum / row_sum + 1.0)` in a single
/// pass, avoiding two separate traversals of the dense row.
///
/// Zeros produce `ln(0.0 * factor + 1.0) = ln(1.0) = 0.0`, staying zero.
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
    if row_sum > 0.0 {
        let factor = target_sum / row_sum;
        for v in row.iter_mut() {
            // Scale in f64 for precision, cast to f32, then f32 ln for speed.
            // Numerically matches sequential normalize→log1p path.
            *v = ((*v as f64 * factor) as f32 + 1.0).ln();
        }
    }
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
}
