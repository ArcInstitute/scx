// ScxCsr struct + operations (SPEC §6.1)

/// Errors from CSR construction and validation.
#[derive(Debug, thiserror::Error)]
pub enum CsrError {
    #[error("indptr length {got} != shape.0 + 1 ({expected})")]
    IndptrLength { got: usize, expected: usize },

    #[error("indptr is not monotonically non-decreasing at index {index}")]
    IndptrNotMonotonic { index: usize },

    #[error("indptr[0] must be 0, got {0}")]
    IndptrNonZeroStart(i64),

    #[error("indices.len() ({indices}) != data.len() ({data})")]
    IndicesDataMismatch { indices: usize, data: usize },

    #[error("nnz mismatch: indptr[last]={indptr_nnz}, indices.len()={actual_nnz}")]
    NnzMismatch { indptr_nnz: i64, actual_nnz: usize },

    #[error("index {index} out of range [0, {n_cols}) at position {position}")]
    IndexOutOfRange {
        index: i32,
        n_cols: usize,
        position: usize,
    },

    #[error("dimension overflow: {rows} * {cols} exceeds usize")]
    DimensionOverflow { rows: usize, cols: usize },

    #[error("dense array length {got} != expected {expected} (n_rows * n_cols)")]
    DenseLengthMismatch { got: usize, expected: usize },

    #[error("n_cols {0} exceeds i32::MAX, cannot represent as i32 column indices")]
    ColumnOverflow(usize),

    #[error("row_slice bounds invalid: start={start}, end={end}, n_rows={n_rows}")]
    RowSliceOutOfBounds {
        start: usize,
        end: usize,
        n_rows: usize,
    },
}

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
    /// Create a new ScxCsr with full validation.
    pub fn new(
        shape: (usize, usize),
        indptr: Vec<i64>,
        indices: Vec<i32>,
        data: Vec<f32>,
    ) -> Result<Self, CsrError> {
        // 1. indptr length
        let expected_len = shape.0 + 1;
        if indptr.len() != expected_len {
            return Err(CsrError::IndptrLength {
                got: indptr.len(),
                expected: expected_len,
            });
        }

        // 2. indptr[0] == 0 (scipy convention)
        if indptr[0] != 0 {
            return Err(CsrError::IndptrNonZeroStart(indptr[0]));
        }

        // 3. monotonically non-decreasing
        for i in 1..indptr.len() {
            if indptr[i] < indptr[i - 1] {
                return Err(CsrError::IndptrNotMonotonic { index: i });
            }
        }

        // 4. indices.len() == data.len()
        if indices.len() != data.len() {
            return Err(CsrError::IndicesDataMismatch {
                indices: indices.len(),
                data: data.len(),
            });
        }

        // 5. nnz match
        let indptr_nnz = *indptr.last().unwrap(); // safe: len >= 1
        if indptr_nnz as usize != indices.len() {
            return Err(CsrError::NnzMismatch {
                indptr_nnz,
                actual_nnz: indices.len(),
            });
        }

        // 6. all indices in [0, n_cols)
        for (pos, &idx) in indices.iter().enumerate() {
            if idx < 0 || idx as usize >= shape.1 {
                return Err(CsrError::IndexOutOfRange {
                    index: idx,
                    n_cols: shape.1,
                    position: pos,
                });
            }
        }

        Ok(Self {
            shape,
            indptr,
            indices,
            data,
        })
    }

    /// Create a new ScxCsr without validation.
    ///
    /// Use when the data is known to be valid (e.g., decoded from checksummed shards).
    pub fn new_unchecked(
        shape: (usize, usize),
        indptr: Vec<i64>,
        indices: Vec<i32>,
        data: Vec<f32>,
    ) -> Self {
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

    /// Extract a contiguous slice of rows `[start..end)` as a new ScxCsr.
    pub fn row_slice(&self, start: usize, end: usize) -> Result<ScxCsr, CsrError> {
        if start > end || end > self.n_rows() {
            return Err(CsrError::RowSliceOutOfBounds {
                start,
                end,
                n_rows: self.n_rows(),
            });
        }

        let nnz_start = self.indptr[start] as usize;
        let nnz_end = self.indptr[end] as usize;

        // Rebase indptr to start from 0
        let base = self.indptr[start];
        let indptr: Vec<i64> = self.indptr[start..=end].iter().map(|&v| v - base).collect();
        let indices = self.indices[nnz_start..nnz_end].to_vec();
        let data = self.data[nnz_start..nnz_end].to_vec();

        Ok(ScxCsr::new_unchecked((end - start, self.shape.1), indptr, indices, data))
    }

    /// Convert to a dense row-major matrix.
    pub fn to_dense(&self) -> Result<Vec<f32>, CsrError> {
        let (n_rows, n_cols) = self.shape;
        let total = n_rows.checked_mul(n_cols).ok_or(CsrError::DimensionOverflow {
            rows: n_rows,
            cols: n_cols,
        })?;
        let mut dense = vec![0.0f32; total];
        for row in 0..n_rows {
            let start = self.indptr[row] as usize;
            let end = self.indptr[row + 1] as usize;
            for j in start..end {
                let col = self.indices[j] as usize;
                dense[row * n_cols + col] = self.data[j];
            }
        }
        Ok(dense)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_csr() -> ScxCsr {
        // 3x5 matrix:
        // row 0: [0, 5, 0, 10, 0]
        // row 1: [1, 0, 3, 0, 7]
        // row 2: [0, 0, 2, 0, 0]
        ScxCsr::new(
            (3, 5),
            vec![0, 2, 5, 6],
            vec![1, 3, 0, 2, 4, 2],
            vec![5.0, 10.0, 1.0, 3.0, 7.0, 2.0],
        )
        .unwrap()
    }

    #[test]
    fn csr_new_and_accessors() {
        let csr = sample_csr();
        assert_eq!(csr.n_rows(), 3);
        assert_eq!(csr.n_cols(), 5);
        assert_eq!(csr.nnz(), 6);
        assert_eq!(csr.shape, (3, 5));
    }

    #[test]
    fn csr_empty() {
        let csr = ScxCsr::new((0, 0), vec![0], vec![], vec![]).unwrap();
        assert_eq!(csr.n_rows(), 0);
        assert_eq!(csr.n_cols(), 0);
        assert_eq!(csr.nnz(), 0);
    }

    // 12.7: row_slice
    #[test]
    fn row_slice_middle_rows() {
        let csr = sample_csr();
        let sliced = csr.row_slice(1, 3).unwrap();
        assert_eq!(sliced.shape, (2, 5));
        assert_eq!(sliced.indptr, vec![0, 3, 4]);
        assert_eq!(sliced.indices, vec![0, 2, 4, 2]);
        assert_eq!(sliced.data, vec![1.0, 3.0, 7.0, 2.0]);
    }

    #[test]
    fn row_slice_single_row() {
        let csr = sample_csr();
        let sliced = csr.row_slice(0, 1).unwrap();
        assert_eq!(sliced.shape, (1, 5));
        assert_eq!(sliced.indptr, vec![0, 2]);
        assert_eq!(sliced.indices, vec![1, 3]);
        assert_eq!(sliced.data, vec![5.0, 10.0]);
    }

    #[test]
    fn row_slice_empty() {
        let csr = sample_csr();
        let sliced = csr.row_slice(1, 1).unwrap();
        assert_eq!(sliced.shape, (0, 5));
        assert_eq!(sliced.indptr, vec![0]);
        assert_eq!(sliced.nnz(), 0);
    }

    // 12.8: to_dense
    #[test]
    fn to_dense_known() {
        let csr = sample_csr();
        let dense = csr.to_dense().unwrap();
        #[rustfmt::skip]
        let expected = vec![
            0.0, 5.0, 0.0, 10.0, 0.0,
            1.0, 0.0, 3.0,  0.0, 7.0,
            0.0, 0.0, 2.0,  0.0, 0.0,
        ];
        assert_eq!(dense, expected);
    }

    // 12.9: empty matrix
    #[test]
    fn empty_matrix_with_shape() {
        let csr = ScxCsr::new((5, 10), vec![0, 0, 0, 0, 0, 0], vec![], vec![]).unwrap();
        assert_eq!(csr.n_rows(), 5);
        assert_eq!(csr.n_cols(), 10);
        assert_eq!(csr.nnz(), 0);
        let dense = csr.to_dense().unwrap();
        assert_eq!(dense.len(), 50);
        assert!(dense.iter().all(|&v| v == 0.0));
    }

    // 12.10: single-row and single-column
    #[test]
    fn single_row_matrix() {
        let csr = ScxCsr::new((1, 4), vec![0, 2], vec![1, 3], vec![5.0, 9.0]).unwrap();
        assert_eq!(csr.to_dense().unwrap(), vec![0.0, 5.0, 0.0, 9.0]);
    }

    #[test]
    fn single_column_matrix() {
        let csr = ScxCsr::new((3, 1), vec![0, 1, 1, 1], vec![0], vec![7.0]).unwrap();
        assert_eq!(csr.to_dense().unwrap(), vec![7.0, 0.0, 0.0]);
    }

    // 12.11: mismatched lengths
    #[test]
    fn error_indptr_length() {
        let err = ScxCsr::new((3, 5), vec![0, 2, 5], vec![], vec![]).unwrap_err();
        assert!(matches!(
            err,
            CsrError::IndptrLength {
                got: 3,
                expected: 4
            }
        ));
    }

    #[test]
    fn error_indices_data_mismatch() {
        let err = ScxCsr::new(
            (1, 5),
            vec![0, 2],
            vec![1, 3],
            vec![5.0], // only 1, but indices has 2
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CsrError::IndicesDataMismatch {
                indices: 2,
                data: 1
            }
        ));
    }

    #[test]
    fn error_nnz_mismatch() {
        let err = ScxCsr::new(
            (1, 5),
            vec![0, 3], // claims 3 nnz
            vec![1, 2], // but only 2
            vec![1.0, 2.0],
        )
        .unwrap_err();
        assert!(matches!(err, CsrError::NnzMismatch { .. }));
    }

    // 12.12: non-monotonic indptr
    #[test]
    fn error_indptr_not_monotonic() {
        let err = ScxCsr::new(
            (2, 5),
            vec![0, 3, 1], // decreases at index 2
            vec![0, 1, 2],
            vec![1.0, 2.0, 3.0],
        )
        .unwrap_err();
        assert!(matches!(err, CsrError::IndptrNotMonotonic { index: 2 }));
    }

    #[test]
    fn error_indptr_negative_start() {
        let err = ScxCsr::new((1, 5), vec![-1, 0], vec![], vec![]).unwrap_err();
        assert!(matches!(err, CsrError::IndptrNonZeroStart(-1)));
    }

    #[test]
    fn error_indptr_nonzero_start() {
        let err = ScxCsr::new((1, 5), vec![5, 7], vec![1, 3], vec![1.0, 2.0]).unwrap_err();
        assert!(matches!(err, CsrError::IndptrNonZeroStart(5)));
    }

    // 12.13: out-of-range index
    #[test]
    fn error_index_out_of_range() {
        let err = ScxCsr::new(
            (1, 5),
            vec![0, 2],
            vec![1, 5], // 5 >= n_cols(5)
            vec![1.0, 2.0],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CsrError::IndexOutOfRange {
                index: 5,
                n_cols: 5,
                position: 1
            }
        ));
    }

    #[test]
    fn error_negative_index() {
        let err = ScxCsr::new((1, 5), vec![0, 1], vec![-1], vec![1.0]).unwrap_err();
        assert!(matches!(err, CsrError::IndexOutOfRange { index: -1, .. }));
    }

    #[test]
    fn error_row_slice_start_greater_than_end() {
        let csr = sample_csr();
        let err = csr.row_slice(2, 1).unwrap_err();
        assert!(matches!(
            err,
            CsrError::RowSliceOutOfBounds {
                start: 2,
                end: 1,
                n_rows: 3
            }
        ));
    }

    #[test]
    fn error_row_slice_end_exceeds_n_rows() {
        let csr = sample_csr();
        let err = csr.row_slice(0, 4).unwrap_err();
        assert!(matches!(
            err,
            CsrError::RowSliceOutOfBounds {
                start: 0,
                end: 4,
                n_rows: 3
            }
        ));
    }

    #[test]
    fn test_to_dense_dimension_overflow() {
        // Create a CSR with dimensions that overflow usize when multiplied
        let huge = usize::MAX / 2 + 1;
        let csr = ScxCsr::new_unchecked((huge, 2), vec![], vec![], vec![]);
        let err = csr.to_dense().unwrap_err();
        assert!(matches!(err, CsrError::DimensionOverflow { .. }));
    }
}
