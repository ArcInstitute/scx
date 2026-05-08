// ScxCsc struct + operations (column-major mirror of ScxCsr).
//
// CSC layout: indptr[c]..indptr[c+1] gives the row indices and values
// of column c. Row indices within a column are in arbitrary order.

use crate::csr::{CsrError, ScxCsr};
use crate::transpose::CscArrays;

/// Errors from CSC construction and validation.
#[derive(Debug, thiserror::Error)]
pub enum CscError {
    #[error("indptr length {got} != shape.1 + 1 ({expected})")]
    IndptrLength { got: usize, expected: usize },

    #[error("indptr is not monotonically non-decreasing at index {index}")]
    IndptrNotMonotonic { index: usize },

    #[error("indptr[0] must be 0, got {0}")]
    IndptrNonZeroStart(i64),

    #[error("indices.len() ({indices}) != data.len() ({data})")]
    IndicesDataMismatch { indices: usize, data: usize },

    #[error("nnz mismatch: indptr[last]={indptr_nnz}, indices.len()={actual_nnz}")]
    NnzMismatch { indptr_nnz: i64, actual_nnz: usize },

    #[error("row index {index} out of range [0, {n_rows}) at position {position}")]
    IndexOutOfRange {
        index: i32,
        n_rows: usize,
        position: usize,
    },

    #[error("duplicate row index {index} in column {col}")]
    DuplicateIndex { index: i32, col: usize },

    #[error("dimension overflow: {rows} * {cols} exceeds usize")]
    DimensionOverflow { rows: usize, cols: usize },

    #[error("col_slice bounds invalid: start={start}, end={end}, n_cols={n_cols}")]
    ColSliceOutOfBounds {
        start: usize,
        end: usize,
        n_cols: usize,
    },
}

/// A CSC sparse matrix with scipy-compatible dtypes.
///
/// Column-major mirror of [`ScxCsr`]. Uses `i64` indptr, `i32` indices
/// (row indices), and `f32` data. Within a column, row indices are in
/// arbitrary (unsorted) order.
#[derive(Debug, Clone)]
pub struct ScxCsc {
    /// (n_rows, n_cols) — same convention as `ScxCsr`.
    pub shape: (usize, usize),
    /// Column pointer array (length = n_cols + 1).
    pub indptr: Vec<i64>,
    /// Row indices (length = nnz).
    pub indices: Vec<i32>,
    /// Non-zero values (length = nnz).
    pub data: Vec<f32>,
}

impl ScxCsc {
    /// Create a new ScxCsc with full validation (including duplicate-row check
    /// per column).
    pub fn new(
        shape: (usize, usize),
        indptr: Vec<i64>,
        indices: Vec<i32>,
        data: Vec<f32>,
    ) -> Result<Self, CscError> {
        let (n_rows, n_cols) = shape;

        let expected_len = n_cols + 1;
        if indptr.len() != expected_len {
            return Err(CscError::IndptrLength {
                got: indptr.len(),
                expected: expected_len,
            });
        }

        if indptr[0] != 0 {
            return Err(CscError::IndptrNonZeroStart(indptr[0]));
        }

        for i in 1..indptr.len() {
            if indptr[i] < indptr[i - 1] {
                return Err(CscError::IndptrNotMonotonic { index: i });
            }
        }

        if indices.len() != data.len() {
            return Err(CscError::IndicesDataMismatch {
                indices: indices.len(),
                data: data.len(),
            });
        }

        let indptr_nnz = *indptr.last().unwrap();
        if indptr_nnz as usize != indices.len() {
            return Err(CscError::NnzMismatch {
                indptr_nnz,
                actual_nnz: indices.len(),
            });
        }

        // Indices in [0, n_rows) and no duplicates within a column.
        for col in 0..n_cols {
            let start = indptr[col] as usize;
            let end = indptr[col + 1] as usize;
            let mut seen = std::collections::HashSet::with_capacity(end - start);
            for (offset, &idx) in indices[start..end].iter().enumerate() {
                if idx < 0 || idx as usize >= n_rows {
                    return Err(CscError::IndexOutOfRange {
                        index: idx,
                        n_rows,
                        position: start + offset,
                    });
                }
                if !seen.insert(idx) {
                    return Err(CscError::DuplicateIndex { index: idx, col });
                }
            }
        }

        Ok(Self {
            shape,
            indptr,
            indices,
            data,
        })
    }

    /// Create a new ScxCsc without validation.
    ///
    /// # Invariants the caller must uphold
    ///
    /// 1. `indptr.len() == shape.1 + 1`
    /// 2. `indptr[0] == 0`
    /// 3. `indptr` is monotone non-decreasing
    /// 4. `indices.len() == data.len() == *indptr.last().unwrap() as usize`
    /// 5. Every `indices[k]` is in `[0, shape.0)`
    /// 6. No duplicate row index within a column
    ///
    /// In debug builds, invariants 1–4 are checked via `debug_assert!`.
    pub fn new_unchecked(
        shape: (usize, usize),
        indptr: Vec<i64>,
        indices: Vec<i32>,
        data: Vec<f32>,
    ) -> Self {
        debug_assert_eq!(
            indptr.len(),
            shape.1 + 1,
            "ScxCsc::new_unchecked: indptr.len() must equal shape.1 + 1"
        );
        debug_assert!(
            indptr.first().copied() == Some(0),
            "ScxCsc::new_unchecked: indptr[0] must be 0"
        );
        debug_assert!(
            indptr.windows(2).all(|w| w[0] <= w[1]),
            "ScxCsc::new_unchecked: indptr must be monotone non-decreasing"
        );
        debug_assert_eq!(
            indices.len(),
            data.len(),
            "ScxCsc::new_unchecked: indices.len() must equal data.len()"
        );
        debug_assert_eq!(
            indices.len() as i64,
            *indptr.last().unwrap_or(&0),
            "ScxCsc::new_unchecked: indices.len() must equal indptr.last()"
        );
        Self {
            shape,
            indptr,
            indices,
            data,
        }
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn n_rows(&self) -> usize {
        self.shape.0
    }

    pub fn n_cols(&self) -> usize {
        self.shape.1
    }

    /// Convert to a dense row-major matrix (n_rows × n_cols).
    pub fn to_dense(&self) -> Result<Vec<f32>, CscError> {
        let (n_rows, n_cols) = self.shape;
        let total = n_rows
            .checked_mul(n_cols)
            .ok_or(CscError::DimensionOverflow {
                rows: n_rows,
                cols: n_cols,
            })?;
        let mut dense = vec![0.0f32; total];
        for col in 0..n_cols {
            let start = self.indptr[col] as usize;
            let end = self.indptr[col + 1] as usize;
            for j in start..end {
                let row = self.indices[j] as usize;
                dense[row * n_cols + col] = self.data[j];
            }
        }
        Ok(dense)
    }

    /// Extract a contiguous slice of columns `[start..end)` as a new ScxCsc.
    pub fn col_slice(&self, start: usize, end: usize) -> Result<ScxCsc, CscError> {
        if start > end || end > self.n_cols() {
            return Err(CscError::ColSliceOutOfBounds {
                start,
                end,
                n_cols: self.n_cols(),
            });
        }

        let nnz_start = self.indptr[start] as usize;
        let nnz_end = self.indptr[end] as usize;

        let base = self.indptr[start];
        let indptr: Vec<i64> = self.indptr[start..=end].iter().map(|&v| v - base).collect();
        let indices = self.indices[nnz_start..nnz_end].to_vec();
        let data = self.data[nnz_start..nnz_end].to_vec();

        Ok(ScxCsc::new_unchecked(
            (self.shape.0, end - start),
            indptr,
            indices,
            data,
        ))
    }

    /// Convert to CSR by transposing the column-major arrays. Reuses the
    /// same two-pass scatter machinery as the CSR→CSC path: counting
    /// per-row nnz, prefix-sum, then scatter.
    pub fn to_csr(&self) -> Result<ScxCsr, CsrError> {
        let (n_rows, n_cols) = self.shape;
        let nnz = self.nnz();

        if nnz == 0 {
            return ScxCsr::new(
                (n_rows, n_cols),
                vec![0i64; n_rows + 1],
                Vec::new(),
                Vec::new(),
            );
        }

        // Pass 1: count per-row nnz.
        let mut row_counts = vec![0usize; n_rows];
        for &row_idx in &self.indices {
            row_counts[row_idx as usize] += 1;
        }

        // Pass 2: prefix-sum into CSR indptr.
        let mut indptr = Vec::with_capacity(n_rows + 1);
        indptr.push(0i64);
        let mut cumsum = 0i64;
        for &count in &row_counts {
            cumsum += count as i64;
            indptr.push(cumsum);
        }

        // Pass 3: scatter (col, val) pairs into per-row positions.
        let mut indices = vec![0i32; nnz];
        let mut data = vec![0.0f32; nnz];
        let mut workspace = vec![0usize; n_rows];

        for col in 0..n_cols {
            let col_start = self.indptr[col] as usize;
            let col_end = self.indptr[col + 1] as usize;
            for j in col_start..col_end {
                let row = self.indices[j] as usize;
                let dest = indptr[row] as usize + workspace[row];
                indices[dest] = col as i32;
                data[dest] = self.data[j];
                workspace[row] += 1;
            }
        }

        ScxCsr::new((n_rows, n_cols), indptr, indices, data)
    }
}

impl From<CscArrays> for ScxCsc {
    fn from(c: CscArrays) -> Self {
        ScxCsc::new_unchecked(c.shape, c.indptr, c.indices, c.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_csc() -> ScxCsc {
        // 3×5 matrix (same as csr.rs sample_csr):
        //   row 0: [0, 5, 0, 10, 0]
        //   row 1: [1, 0, 3,  0, 7]
        //   row 2: [0, 0, 2,  0, 0]
        // CSC layout (column-by-column):
        //   col 0: row 1 -> 1.0
        //   col 1: row 0 -> 5.0
        //   col 2: row 1 -> 3.0, row 2 -> 2.0
        //   col 3: row 0 -> 10.0
        //   col 4: row 1 -> 7.0
        ScxCsc::new(
            (3, 5),
            vec![0, 1, 2, 4, 5, 6],
            vec![1, 0, 1, 2, 0, 1],
            vec![1.0, 5.0, 3.0, 2.0, 10.0, 7.0],
        )
        .unwrap()
    }

    #[test]
    fn csc_new_and_accessors() {
        let csc = sample_csc();
        assert_eq!(csc.n_rows(), 3);
        assert_eq!(csc.n_cols(), 5);
        assert_eq!(csc.nnz(), 6);
    }

    #[test]
    fn csc_to_dense_round_trip() {
        let csc = sample_csc();
        let dense = csc.to_dense().unwrap();
        #[rustfmt::skip]
        let expected = vec![
            0.0, 5.0, 0.0, 10.0, 0.0,
            1.0, 0.0, 3.0,  0.0, 7.0,
            0.0, 0.0, 2.0,  0.0, 0.0,
        ];
        assert_eq!(dense, expected);
    }

    #[test]
    fn csc_col_slice_middle() {
        let csc = sample_csc();
        let sliced = csc.col_slice(1, 4).unwrap();
        assert_eq!(sliced.shape, (3, 3));
        // cols 1, 2, 3:
        //   col 0 (was 1): row 0 -> 5.0
        //   col 1 (was 2): row 1 -> 3.0, row 2 -> 2.0
        //   col 2 (was 3): row 0 -> 10.0
        assert_eq!(sliced.indptr, vec![0, 1, 3, 4]);
        assert_eq!(sliced.indices, vec![0, 1, 2, 0]);
        assert_eq!(sliced.data, vec![5.0, 3.0, 2.0, 10.0]);
    }

    #[test]
    fn csc_col_slice_empty_range() {
        let csc = sample_csc();
        let sliced = csc.col_slice(2, 2).unwrap();
        assert_eq!(sliced.shape, (3, 0));
        assert_eq!(sliced.indptr, vec![0]);
        assert_eq!(sliced.nnz(), 0);
    }

    #[test]
    fn csc_col_slice_full_range() {
        let csc = sample_csc();
        let sliced = csc.col_slice(0, csc.n_cols()).unwrap();
        assert_eq!(sliced.shape, csc.shape);
        assert_eq!(sliced.indptr, csc.indptr);
        assert_eq!(sliced.indices, csc.indices);
        assert_eq!(sliced.data, csc.data);
    }

    #[test]
    fn csc_col_slice_single_column() {
        let csc = sample_csc();
        let sliced = csc.col_slice(2, 3).unwrap();
        assert_eq!(sliced.shape, (3, 1));
        assert_eq!(sliced.indptr, vec![0, 2]);
        assert_eq!(sliced.indices, vec![1, 2]);
        assert_eq!(sliced.data, vec![3.0, 2.0]);
    }

    #[test]
    fn csc_col_slice_out_of_bounds() {
        let csc = sample_csc();
        let err = csc.col_slice(3, 6).unwrap_err();
        assert!(matches!(
            err,
            CscError::ColSliceOutOfBounds {
                start: 3,
                end: 6,
                n_cols: 5
            }
        ));
    }

    #[test]
    fn csc_to_csr_round_trip() {
        let csc = sample_csc();
        let csr = csc.to_csr().unwrap();
        assert_eq!(csr.shape, (3, 5));
        let dense = csr.to_dense().unwrap();
        #[rustfmt::skip]
        let expected = vec![
            0.0, 5.0, 0.0, 10.0, 0.0,
            1.0, 0.0, 3.0,  0.0, 7.0,
            0.0, 0.0, 2.0,  0.0, 0.0,
        ];
        assert_eq!(dense, expected);
    }

    #[test]
    fn csc_to_csr_empty() {
        let csc = ScxCsc::new((4, 3), vec![0, 0, 0, 0], vec![], vec![]).unwrap();
        let csr = csc.to_csr().unwrap();
        assert_eq!(csr.shape, (4, 3));
        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.indptr, vec![0; 5]);
    }

    #[test]
    fn csc_validation_indptr_length() {
        let err = ScxCsc::new((3, 5), vec![0, 1, 2], vec![], vec![]).unwrap_err();
        assert!(matches!(
            err,
            CscError::IndptrLength {
                got: 3,
                expected: 6
            }
        ));
    }

    #[test]
    fn csc_validation_non_monotonic() {
        let err =
            ScxCsc::new((3, 2), vec![0, 3, 1], vec![0, 1, 2], vec![1.0, 2.0, 3.0]).unwrap_err();
        assert!(matches!(err, CscError::IndptrNotMonotonic { index: 2 }));
    }

    #[test]
    fn csc_validation_out_of_range_index() {
        let err = ScxCsc::new(
            (2, 1),
            vec![0, 1],
            vec![5], // 5 >= n_rows=2
            vec![1.0],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CscError::IndexOutOfRange {
                index: 5,
                n_rows: 2,
                ..
            }
        ));
    }

    #[test]
    fn csc_validation_duplicate_index_per_column() {
        let err = ScxCsc::new((3, 1), vec![0, 2], vec![0, 0], vec![1.0, 2.0]).unwrap_err();
        assert!(matches!(err, CscError::DuplicateIndex { index: 0, col: 0 }));
    }

    #[test]
    fn csc_from_csc_arrays() {
        let arrays = CscArrays {
            shape: (2, 1),
            indptr: vec![0, 1],
            indices: vec![1],
            data: vec![3.5],
        };
        let csc: ScxCsc = arrays.into();
        assert_eq!(csc.shape, (2, 1));
        assert_eq!(csc.indices, vec![1]);
        assert_eq!(csc.data, vec![3.5]);
    }
}
