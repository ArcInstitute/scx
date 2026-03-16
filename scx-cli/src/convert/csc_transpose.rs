// CSC → CSR streaming scatter transpose

/// Transpose a CSC sparse matrix to CSR format.
///
/// CSC stores data column-by-column; CSR stores data row-by-row.
/// Uses the streaming scatter algorithm:
/// 1. Count per-row nnz from csc_indices
/// 2. Prefix-sum to build csr_indptr
/// 3. Scatter CSC entries into CSR positions
pub fn csc_to_csr(
    csc_indptr: &[i64],
    csc_indices: &[i32],
    csc_data: &[f32],
    n_rows: usize,
    n_cols: usize,
) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
    let nnz = csc_data.len();
    assert_eq!(csc_indices.len(), nnz);
    assert_eq!(csc_indptr.len(), n_cols + 1);

    // Step 1: Count per-row nnz
    let mut row_counts = vec![0i64; n_rows];
    for &row in csc_indices {
        row_counts[row as usize] += 1;
    }

    // Step 2: Prefix sum → csr_indptr
    let mut csr_indptr = Vec::with_capacity(n_rows + 1);
    csr_indptr.push(0i64);
    let mut cumulative = 0i64;
    for &count in &row_counts {
        cumulative += count;
        csr_indptr.push(cumulative);
    }

    // Step 3: Scatter
    let mut csr_indices = vec![0i32; nnz];
    let mut csr_data = vec![0.0f32; nnz];
    let mut cursor: Vec<i64> = csr_indptr[..n_rows].to_vec();

    for col in 0..n_cols {
        let col_start = csc_indptr[col] as usize;
        let col_end = csc_indptr[col + 1] as usize;
        for idx in col_start..col_end {
            let row = csc_indices[idx] as usize;
            let pos = cursor[row] as usize;
            csr_indices[pos] = col as i32;
            csr_data[pos] = csc_data[idx];
            cursor[row] += 1;
        }
    }

    (csr_indptr, csr_indices, csr_data)
}
