// Dense <-> CSR conversion

use crate::csr::{CsrError, ScxCsr};

/// Convert a dense row-major matrix to CSR format.
///
/// `dense` must have exactly `n_rows * n_cols` elements.
pub fn dense_to_csr(dense: &[f32], n_rows: usize, n_cols: usize) -> Result<ScxCsr, CsrError> {
    if n_cols > i32::MAX as usize {
        return Err(CsrError::ColumnOverflow(n_cols));
    }
    let expected_len = n_rows.checked_mul(n_cols).ok_or(CsrError::DimensionOverflow {
        rows: n_rows,
        cols: n_cols,
    })?;
    if dense.len() != expected_len {
        return Err(CsrError::DenseLengthMismatch {
            got: dense.len(),
            expected: expected_len,
        });
    }

    let mut indptr = Vec::with_capacity(n_rows + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();

    indptr.push(0i64);
    for row in 0..n_rows {
        let row_start = row * n_cols;
        for col in 0..n_cols {
            let val = dense[row_start + col];
            if val != 0.0 {
                indices.push(col as i32);
                data.push(val);
            }
        }
        indptr.push(data.len() as i64);
    }

    ScxCsr::new((n_rows, n_cols), indptr, indices, data)
}

/// Convert a CSR matrix to dense row-major format.
pub fn csr_to_dense(csr: &ScxCsr) -> Result<Vec<f32>, CsrError> {
    csr.to_dense()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_to_csr_round_trip() {
        #[rustfmt::skip]
        let dense = vec![
            0.0, 5.0, 0.0, 10.0, 0.0,
            1.0, 0.0, 3.0,  0.0, 7.0,
            0.0, 0.0, 2.0,  0.0, 0.0,
        ];
        let csr = dense_to_csr(&dense, 3, 5).unwrap();
        assert_eq!(csr.shape, (3, 5));
        assert_eq!(csr.nnz(), 6);
        assert_eq!(csr.indptr, vec![0, 2, 5, 6]);
        assert_eq!(csr.indices, vec![1, 3, 0, 2, 4, 2]);
        assert_eq!(csr.data, vec![5.0, 10.0, 1.0, 3.0, 7.0, 2.0]);

        // Round-trip back
        let back = csr_to_dense(&csr).unwrap();
        assert_eq!(back, dense);
    }

    #[test]
    fn dense_to_csr_all_zeros() {
        let dense = vec![0.0f32; 20];
        let csr = dense_to_csr(&dense, 4, 5).unwrap();
        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.indptr, vec![0, 0, 0, 0, 0]);
        assert_eq!(csr_to_dense(&csr).unwrap(), dense);
    }

    #[test]
    fn dense_to_csr_all_nonzero() {
        let dense = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let csr = dense_to_csr(&dense, 2, 3).unwrap();
        assert_eq!(csr.nnz(), 6);
        assert_eq!(csr_to_dense(&csr).unwrap(), dense);
    }

    #[test]
    fn csr_to_dense_delegates() {
        let csr = ScxCsr::new((2, 3), vec![0, 1, 3], vec![2, 0, 1], vec![5.0, 1.0, 2.0]).unwrap();
        assert_eq!(csr_to_dense(&csr).unwrap(), csr.to_dense().unwrap());
    }

    #[test]
    fn test_dense_to_csr_dimension_overflow() {
        let dense = vec![1.0f32; 4];
        let huge = usize::MAX / 2 + 1;
        let err = dense_to_csr(&dense, huge, 2).unwrap_err();
        assert!(matches!(err, CsrError::DimensionOverflow { .. }));
    }

    #[test]
    fn test_dense_to_csr_length_mismatch() {
        let dense = vec![1.0f32; 4];
        let err = dense_to_csr(&dense, 2, 3).unwrap_err();
        assert!(matches!(
            err,
            CsrError::DenseLengthMismatch {
                got: 4,
                expected: 6
            }
        ));
    }

    #[test]
    fn test_dense_to_csr_rejects_large_ncols() {
        let dense = vec![1.0f32; 1];
        let big_cols = i32::MAX as usize + 1;
        let err = dense_to_csr(&dense, 1, big_cols).unwrap_err();
        assert!(matches!(err, CsrError::ColumnOverflow(_)));
    }
}
