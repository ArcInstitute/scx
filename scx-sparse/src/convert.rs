// Dense <-> CSR conversion

use crate::csr::{CsrError, ScxCsr};

/// Convert a dense row-major matrix to CSR format.
///
/// `dense` must have exactly `n_rows * n_cols` elements.
pub fn dense_to_csr(dense: &[f32], n_rows: usize, n_cols: usize) -> Result<ScxCsr, CsrError> {
    assert_eq!(dense.len(), n_rows * n_cols);

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
pub fn csr_to_dense(csr: &ScxCsr) -> Vec<f32> {
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
        let back = csr_to_dense(&csr);
        assert_eq!(back, dense);
    }

    #[test]
    fn dense_to_csr_all_zeros() {
        let dense = vec![0.0f32; 20];
        let csr = dense_to_csr(&dense, 4, 5).unwrap();
        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.indptr, vec![0, 0, 0, 0, 0]);
        assert_eq!(csr_to_dense(&csr), dense);
    }

    #[test]
    fn dense_to_csr_all_nonzero() {
        let dense = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let csr = dense_to_csr(&dense, 2, 3).unwrap();
        assert_eq!(csr.nnz(), 6);
        assert_eq!(csr_to_dense(&csr), dense);
    }

    #[test]
    fn csr_to_dense_delegates() {
        let csr = ScxCsr::new(
            (2, 3),
            vec![0, 1, 3],
            vec![2, 0, 1],
            vec![5.0, 1.0, 2.0],
        )
        .unwrap();
        assert_eq!(csr_to_dense(&csr), csr.to_dense());
    }
}
