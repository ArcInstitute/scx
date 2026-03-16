// ScxCsr struct + operations (SPEC §6.1)

/// A CSR sparse matrix with scipy-compatible dtypes.
///
/// Uses `i64` indptr, `i32` indices, and `f32` data to match scipy's
/// CSR layout for zero-copy interop via numpy/PyO3.
#[derive(Debug, Clone)]
pub struct ScxCsr {
    /// (n_rows, n_cols)
    pub shape: (usize, usize),
    /// Row pointer array (length = n_rows + 1). scipy-compatible i64.
    pub indptr: Vec<i64>,
    /// Column indices (length = nnz). scipy-compatible i32.
    pub indices: Vec<i32>,
    /// Non-zero values (length = nnz). scipy-compatible f32.
    pub data: Vec<f32>,
}

impl ScxCsr {
    /// Create a new ScxCsr. No validation (Task 12 will add that).
    pub fn new(shape: (usize, usize), indptr: Vec<i64>, indices: Vec<i32>, data: Vec<f32>) -> Self {
        Self {
            shape,
            indptr,
            indices,
            data,
        }
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Number of rows.
    pub fn n_rows(&self) -> usize {
        self.shape.0
    }

    /// Number of columns.
    pub fn n_cols(&self) -> usize {
        self.shape.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_new_and_accessors() {
        let csr = ScxCsr::new(
            (3, 5),
            vec![0, 2, 5, 6],
            vec![1, 3, 0, 2, 4, 2],
            vec![5.0, 10.0, 1.0, 3.0, 7.0, 2.0],
        );
        assert_eq!(csr.n_rows(), 3);
        assert_eq!(csr.n_cols(), 5);
        assert_eq!(csr.nnz(), 6);
        assert_eq!(csr.shape, (3, 5));
    }

    #[test]
    fn csr_empty() {
        let csr = ScxCsr::new((0, 0), vec![0], vec![], vec![]);
        assert_eq!(csr.n_rows(), 0);
        assert_eq!(csr.n_cols(), 0);
        assert_eq!(csr.nnz(), 0);
    }
}
