//! Fused normalize + log1p operations for CSR data.
//!
//! Applies both transformations in a single pass over each CSR row, avoiding
//! a separate traversal for each operation. See SPEC.md §7.2 "Operation fusion"
//! and ROADMAP.md §2.6.

use scx_sparse::ScxCsr;

/// Normalize a single CSR row to a target sum.
///
/// Equivalent to `scanpy.pp.normalize_total` for one row:
///   `data[i] = data[i] / row_sum * target_sum`
///
/// If the row sum is zero (or the row is empty), the data is left unchanged.
pub fn normalize_row(data: &mut [f32], indptr: &[i64], row_idx: usize, target_sum: f64) {
    let start = indptr[row_idx] as usize;
    let end = indptr[row_idx + 1] as usize;
    let row_sum: f64 = data[start..end].iter().map(|&v| v as f64).sum();
    if row_sum > 0.0 {
        let factor = target_sum / row_sum;
        for v in &mut data[start..end] {
            *v = (*v as f64 * factor) as f32;
        }
    }
}

/// Apply `ln(x + 1)` to a single CSR row.
///
/// Equivalent to `numpy.log1p` for one row's non-zero values.
pub fn log1p_row(data: &mut [f32], indptr: &[i64], row_idx: usize) {
    let start = indptr[row_idx] as usize;
    let end = indptr[row_idx + 1] as usize;
    for v in &mut data[start..end] {
        *v = v.ln_1p();
    }
}

/// Fused normalize + log1p for a single CSR row.
///
/// Computes `ln(x / row_sum * target_sum + 1)` in a single pass, avoiding
/// an intermediate materialized array. Numerically equivalent to calling
/// `normalize_row` then `log1p_row` (within f32 rounding).
pub fn fused_normalize_log1p(
    data: &mut [f32],
    indptr: &[i64],
    row_idx: usize,
    target_sum: f64,
) {
    let start = indptr[row_idx] as usize;
    let end = indptr[row_idx + 1] as usize;
    let row_sum: f64 = data[start..end].iter().map(|&v| v as f64).sum();
    if row_sum > 0.0 {
        let factor = target_sum / row_sum;
        for v in &mut data[start..end] {
            // Scale in f64 for precision, cast to f32, then f32 ln for speed.
            // Numerically matches sequential normalize→log1p path.
            *v = ((*v as f64 * factor) as f32).ln_1p();
        }
    }
}

/// Apply fused operations to an entire CSR matrix.
///
/// Dispatches to the optimal path based on which operations are requested:
/// - `(Some(target_sum), true)` → fused normalize+log1p (single pass)
/// - `(Some(target_sum), false)` → normalize only
/// - `(None, true)` → log1p only
/// - `(None, false)` → no-op
pub fn apply_fused_ops(csr: &mut ScxCsr, normalize: Option<f64>, log1p: bool) {
    let n_rows = csr.shape.0;
    for row in 0..n_rows {
        match (normalize, log1p) {
            (Some(target_sum), true) => {
                fused_normalize_log1p(&mut csr.data, &csr.indptr, row, target_sum)
            }
            (Some(target_sum), false) => {
                normalize_row(&mut csr.data, &csr.indptr, row, target_sum)
            }
            (None, true) => log1p_row(&mut csr.data, &csr.indptr, row),
            (None, false) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a test CSR matrix.
    ///
    /// 3×5 matrix:
    /// row 0: [0, 5, 0, 10, 0]   → nnz at cols 1,3; values 5,10; row_sum=15
    /// row 1: [1, 0, 3, 0, 7]    → nnz at cols 0,2,4; values 1,3,7; row_sum=11
    /// row 2: [0, 0, 2, 0, 0]    → nnz at col 2; value 2; row_sum=2
    fn sample_csr() -> ScxCsr {
        ScxCsr::new(
            (3, 5),
            vec![0, 2, 5, 6],
            vec![1, 3, 0, 2, 4, 2],
            vec![5.0, 10.0, 1.0, 3.0, 7.0, 2.0],
        )
        .unwrap()
    }

    // ---- normalize_row tests ----

    #[test]
    fn normalize_row_matches_scanpy() {
        // scanpy.pp.normalize_total with target_sum=10000:
        // row 0 sum = 15 → factor = 10000/15 = 666.667
        //   5.0 → 5*666.667 = 3333.333
        //   10.0 → 10*666.667 = 6666.667
        let mut csr = sample_csr();
        let target_sum = 10_000.0;

        normalize_row(&mut csr.data, &csr.indptr, 0, target_sum);

        let expected_0 = (5.0_f64 / 15.0 * target_sum) as f32;
        let expected_1 = (10.0_f64 / 15.0 * target_sum) as f32;
        assert!((csr.data[0] - expected_0).abs() < 1e-3);
        assert!((csr.data[1] - expected_1).abs() < 1e-3);

        // Other rows remain unchanged
        assert_eq!(csr.data[2], 1.0);
        assert_eq!(csr.data[3], 3.0);
        assert_eq!(csr.data[4], 7.0);
        assert_eq!(csr.data[5], 2.0);
    }

    #[test]
    fn normalize_row_all_rows() {
        let mut csr = sample_csr();
        let target = 1.0;
        for row in 0..3 {
            normalize_row(&mut csr.data, &csr.indptr, row, target);
        }
        // After normalizing each row to sum=1, check row sums
        for row in 0..3 {
            let start = csr.indptr[row] as usize;
            let end = csr.indptr[row + 1] as usize;
            let sum: f64 = csr.data[start..end].iter().map(|&v| v as f64).sum();
            assert!((sum - 1.0).abs() < 1e-6, "row {} sum = {}", row, sum);
        }
    }

    // ---- log1p_row tests ----

    #[test]
    fn log1p_row_matches_numpy() {
        // numpy.log1p(5.0) = ln(6) ≈ 1.7918
        // numpy.log1p(10.0) = ln(11) ≈ 2.3979
        let mut csr = sample_csr();

        log1p_row(&mut csr.data, &csr.indptr, 0);

        let expected_0 = 5.0_f32.ln_1p();
        let expected_1 = 10.0_f32.ln_1p();
        assert!((csr.data[0] - expected_0).abs() < 1e-6);
        assert!((csr.data[1] - expected_1).abs() < 1e-6);

        // Other rows remain unchanged
        assert_eq!(csr.data[2], 1.0);
    }

    #[test]
    fn log1p_row_all_values() {
        let mut csr = sample_csr();
        for row in 0..3 {
            log1p_row(&mut csr.data, &csr.indptr, row);
        }
        // All values should be ln(original + 1)
        let original = vec![5.0_f32, 10.0, 1.0, 3.0, 7.0, 2.0];
        for (i, &orig) in original.iter().enumerate() {
            let expected = orig.ln_1p();
            assert!(
                (csr.data[i] - expected).abs() < 1e-6,
                "data[{}] = {}, expected {}",
                i,
                csr.data[i],
                expected
            );
        }
    }

    // ---- fused_normalize_log1p tests ----

    #[test]
    fn fused_matches_sequential() {
        // Fused should produce identical results to normalize-then-log1p
        let target = 10_000.0;

        // Sequential path
        let mut sequential = sample_csr();
        for row in 0..3 {
            normalize_row(&mut sequential.data, &sequential.indptr, row, target);
        }
        for row in 0..3 {
            log1p_row(&mut sequential.data, &sequential.indptr, row);
        }

        // Fused path
        let mut fused = sample_csr();
        for row in 0..3 {
            fused_normalize_log1p(&mut fused.data, &fused.indptr, row, target);
        }

        // Compare bit-level: both paths do f64 intermediate, cast to f32
        for i in 0..fused.data.len() {
            assert!(
                (fused.data[i] - sequential.data[i]).abs() < 1e-6,
                "mismatch at [{}]: fused={}, seq={}",
                i,
                fused.data[i],
                sequential.data[i]
            );
        }
    }

    // ---- zero-sum row test ----

    #[test]
    fn zero_sum_row_unchanged() {
        // Edge case: row with very small values that sum to effectively zero.
        // In CSR, rows with truly zero values shouldn't have entries, but
        // an empty row (nnz=0) should be handled gracefully.
        let mut csr = ScxCsr::new(
            (2, 3),
            vec![0, 0, 2], // row 0 is empty, row 1 has 2 entries
            vec![0, 2],
            vec![3.0, 7.0],
        )
        .unwrap();

        // Normalize empty row → no-op (no entries to modify)
        normalize_row(&mut csr.data, &csr.indptr, 0, 10_000.0);
        assert_eq!(csr.data, vec![3.0, 7.0]); // unchanged

        // Fused on empty row → no-op
        fused_normalize_log1p(&mut csr.data, &csr.indptr, 0, 10_000.0);
        assert_eq!(csr.data, vec![3.0, 7.0]); // unchanged

        // log1p on empty row → no-op
        log1p_row(&mut csr.data, &csr.indptr, 0);
        assert_eq!(csr.data, vec![3.0, 7.0]); // unchanged
    }

    // ---- single non-zero value per row ----

    #[test]
    fn single_nonzero_normalize() {
        // Row with a single non-zero value: after normalize, it should equal target_sum.
        let mut csr = ScxCsr::new(
            (1, 5),
            vec![0, 1],
            vec![2],
            vec![42.0],
        )
        .unwrap();

        normalize_row(&mut csr.data, &csr.indptr, 0, 10_000.0);
        // 42 / 42 * 10000 = 10000
        assert!((csr.data[0] - 10_000.0).abs() < 1e-3);
    }

    #[test]
    fn single_nonzero_fused() {
        let mut csr = ScxCsr::new(
            (1, 5),
            vec![0, 1],
            vec![2],
            vec![42.0],
        )
        .unwrap();

        fused_normalize_log1p(&mut csr.data, &csr.indptr, 0, 10_000.0);
        // ln(42/42 * 10000 + 1) = ln(10001)
        let expected = (10_001.0_f64).ln() as f32;
        assert!((csr.data[0] - expected).abs() < 1e-3);
    }

    // ---- apply_fused_ops tests ----

    #[test]
    fn apply_fused_ops_normalize_only() {
        let mut csr = sample_csr();
        apply_fused_ops(&mut csr, Some(1.0), false);
        for row in 0..3 {
            let start = csr.indptr[row] as usize;
            let end = csr.indptr[row + 1] as usize;
            let sum: f64 = csr.data[start..end].iter().map(|&v| v as f64).sum();
            assert!((sum - 1.0).abs() < 1e-6, "row {} sum = {}", row, sum);
        }
    }

    #[test]
    fn apply_fused_ops_log1p_only() {
        let mut csr = sample_csr();
        let original_data = csr.data.clone();
        apply_fused_ops(&mut csr, None, true);
        for (i, &orig) in original_data.iter().enumerate() {
            let expected = orig.ln_1p();
            assert!(
                (csr.data[i] - expected).abs() < 1e-6,
                "data[{}] = {}, expected {}",
                i,
                csr.data[i],
                expected
            );
        }
    }

    #[test]
    fn apply_fused_ops_both() {
        let target = 10_000.0;

        // Sequential path for reference
        let mut expected = sample_csr();
        apply_fused_ops(&mut expected, Some(target), false);
        apply_fused_ops(&mut expected, None, true);

        // Fused path
        let mut actual = sample_csr();
        apply_fused_ops(&mut actual, Some(target), true);

        for i in 0..actual.data.len() {
            assert!(
                (actual.data[i] - expected.data[i]).abs() < 1e-6,
                "mismatch at [{}]: fused={}, seq={}",
                i,
                actual.data[i],
                expected.data[i]
            );
        }
    }

    #[test]
    fn apply_fused_ops_noop() {
        let mut csr = sample_csr();
        let original = csr.data.clone();
        apply_fused_ops(&mut csr, None, false);
        assert_eq!(csr.data, original);
    }
}
