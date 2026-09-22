// Streaming CSR → CSC transpose (docs/architecture.md (CLI))

use crate::csr::{CsrError, ScxCsr};

/// CSC (Compressed Sparse Column) arrays.
///
/// Same dtypes as ScxCsr (i64 indptr, i32 indices, f32 data) but in
/// column-major order: indptr[j] gives the start of column j's entries.
#[derive(Debug, Clone)]
pub struct CscArrays {
    pub shape: (usize, usize), // (n_rows, n_cols) — same as input
    pub indptr: Vec<i64>,      // length = n_cols + 1
    pub indices: Vec<i32>,     // row indices (length = nnz)
    pub data: Vec<f32>,        // values (length = nnz)
}

#[derive(Debug, thiserror::Error)]
pub enum TransposeError {
    #[error("memory limit too small: need at least {min_bytes} bytes for 1 column chunk")]
    MemoryLimitTooSmall { min_bytes: usize },
    #[error("CSR data error: {0}")]
    CsrError(#[from] CsrError),
    #[error("shape mismatch: shard n_cols {shard_cols} != expected {expected_cols}")]
    ShapeMismatch {
        shard_cols: usize,
        expected_cols: usize,
    },
}

/// Transpose a CSR matrix to CSC in-memory.
///
/// Uses the standard two-pass scatter algorithm:
///   1. Count per-column nnz from CSR col indices
///   2. Prefix-sum → CSC indptr
///   3. Scatter CSR entries into CSC positions
///
/// This is the inverse of `csc_to_csr()` in scx-convert/src/h5ad/csc_transpose.rs.
pub fn csr_to_csc(csr: &ScxCsr) -> CscArrays {
    let (n_rows, n_cols) = csr.shape;
    let nnz = csr.nnz();

    if nnz == 0 {
        return CscArrays {
            shape: (n_rows, n_cols),
            indptr: vec![0i64; n_cols + 1],
            indices: Vec::new(),
            data: Vec::new(),
        };
    }

    // Pass 1: count per-column nnz
    let mut col_counts = vec![0usize; n_cols];
    for &col_idx in &csr.indices {
        col_counts[col_idx as usize] += 1;
    }

    // Build CSC indptr via prefix sum
    let mut indptr = Vec::with_capacity(n_cols + 1);
    indptr.push(0i64);
    let mut cumsum = 0i64;
    for &count in &col_counts {
        cumsum += count as i64;
        indptr.push(cumsum);
    }

    // Pass 2: scatter CSR entries into CSC arrays
    let mut indices = vec![0i32; nnz];
    let mut data = vec![0.0f32; nnz];
    // workspace tracks current write position within each column
    let mut workspace = vec![0usize; n_cols];

    for row in 0..n_rows {
        let row_start = csr.indptr[row] as usize;
        let row_end = csr.indptr[row + 1] as usize;
        for j in row_start..row_end {
            let col = csr.indices[j] as usize;
            let dest = indptr[col] as usize + workspace[col];
            indices[dest] = row as i32;
            data[dest] = csr.data[j];
            workspace[col] += 1;
        }
    }

    CscArrays {
        shape: (n_rows, n_cols),
        indptr,
        indices,
        data,
    }
}

// `streaming_csr_to_csc`, `streaming_csr_to_csc_iter`,
// `streaming_csr_to_csc_iter_with_cap` and `CscShardIterator` lived here.
//
// They answered one column chunk at a time by rescanning every nonzero of
// every shard twice, per chunk — `2 * nnz * n_chunks`, which is 4.8e11 index
// comparisons on census_1m — and they borrowed `&[ScxCsr]` for the iterator's
// whole lifetime, which forced every caller to hold the entire decoded CSR.
// `csc_builder::CscBuilder` replaces them: pushed shards, one pass, and a
// resident set the caller's budget bounds.
//
// Deleted rather than deprecated. A `#[deprecated]` `pub fn` still compiles,
// and what is left here is an algorithm that is asymptotically worse in the
// one dimension that matters, sitting in a published crate for the next
// caller to reach for. `compute_chunk_cols_with_cap` below stays, and is the
// one piece that did not go: it is still the shard-WIDTH rule, and keeping it
// byte for byte is why the emitted layout is unchanged.
//
// `transpose_column_chunk` survives as `pub(crate)` for tests only — it is the
// oracle `csc_builder`'s proptests compare against element for element, which
// is the executable form of the claim that the replacement emits the same
// bytes.

/// Compute chunk size as `min(compute_chunk_cols(...), max_cols)`.
///
/// `max_cols == 0` is treated as "no cap" (returns just the memory
/// bound) for ergonomic API symmetry with `usize::MAX` — both indicate
/// "don't constrain me on the column axis." Callers that genuinely
/// want zero-column chunks should not call this function at all.
pub fn compute_chunk_cols_with_cap(
    n_rows_total: usize,
    max_memory_bytes: usize,
    max_cols: usize,
) -> Result<usize, TransposeError> {
    let mem_bound = compute_chunk_cols(n_rows_total, max_memory_bytes)?;
    if max_cols == 0 {
        return Ok(mem_bound);
    }
    Ok(mem_bound.min(max_cols))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the number of columns per chunk, given a memory budget.
///
/// Budgets ~12 bytes per potential entry and sizes the chunk so the worst case
/// (every row has an entry in every column of the chunk) fits in
/// `max_memory_bytes`. Since the counting-scatter transpose (OPT-1.4) dropped
/// the old 12-byte `(usize, i32, f32)` tuple buffer, the actual per-entry peak
/// is now 8 bytes (i32 index + f32 value) plus small `chunk_n_cols`-sized
/// `col_counts`/`cursor` workspaces — so the 12 is a conservative over-estimate
/// that keeps chunking on the safe (smaller-chunk) side.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn compute_chunk_cols(
    n_rows_total: usize,
    max_memory_bytes: usize,
) -> Result<usize, TransposeError> {
    const BYTES_PER_ENTRY: usize = 12;

    if n_rows_total == 0 {
        // No rows → one pass handles everything (no memory needed)
        return Ok(usize::MAX);
    }

    let bytes_per_col = n_rows_total.saturating_mul(BYTES_PER_ENTRY);
    if bytes_per_col == 0 {
        return Ok(usize::MAX);
    }

    let chunk_cols = max_memory_bytes / bytes_per_col;
    if chunk_cols == 0 {
        return Err(TransposeError::MemoryLimitTooSmall {
            min_bytes: bytes_per_col,
        });
    }

    Ok(chunk_cols)
}

/// Transpose a range of columns `[col_start..col_end)` from multiple CSR shards
/// into CSC arrays covering those columns.
///
/// Returns a `CscArrays` where `shape = (n_rows_total, col_end - col_start)`,
/// and `indptr` has length `chunk_n_cols + 1`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn transpose_column_chunk(
    shards: &[ScxCsr],
    n_rows_total: usize,
    col_start: usize,
    col_end: usize,
) -> CscArrays {
    let chunk_n_cols = col_end - col_start;

    // O(nnz) counting scatter over the in-range entries — same algorithm as the
    // in-memory `csr_to_csc` (count → prefix-sum → scatter), restricted to
    // columns in `[col_start..col_end)`. Replaces the old collect-into-tuples +
    // `O(nnz log nnz)` sort (and its 12-byte-per-entry buffer). Output is
    // byte-identical: shards are scanned in order and rows in order, so
    // `global_row` is non-decreasing in scan order and CSR rows have unique
    // sorted column indices, so the per-column write cursor lays each column's
    // rows down in strictly increasing `global_row` order — exactly what the
    // stable `sort_by_key((col, row))` produced.

    // Pass 1: count per-(local) column nnz in the chunk's column range.
    let mut col_counts = vec![0usize; chunk_n_cols];
    for shard in shards {
        for &col in &shard.indices {
            let col = col as usize;
            if col >= col_start && col < col_end {
                col_counts[col - col_start] += 1;
            }
        }
    }

    // Prefix sum → CSC indptr (length chunk_n_cols + 1).
    let mut indptr = Vec::with_capacity(chunk_n_cols + 1);
    indptr.push(0i64);
    let mut cumsum = 0i64;
    for &count in &col_counts {
        cumsum += count as i64;
        indptr.push(cumsum);
    }
    let nnz = cumsum as usize;

    // Pass 2: scatter entries into their column positions. `cursor` tracks the
    // current write offset within each column.
    let mut indices = vec![0i32; nnz];
    let mut data = vec![0.0f32; nnz];
    let mut cursor = vec![0usize; chunk_n_cols];
    let mut row_offset: usize = 0;
    for shard in shards {
        let shard_n_rows = shard.n_rows();
        for row in 0..shard_n_rows {
            let start = shard.indptr[row] as usize;
            let end = shard.indptr[row + 1] as usize;
            for j in start..end {
                let col = shard.indices[j] as usize;
                if col >= col_start && col < col_end {
                    let local_col = col - col_start;
                    let dest = indptr[local_col] as usize + cursor[local_col];
                    indices[dest] = (row_offset + row) as i32;
                    data[dest] = shard.data[j];
                    cursor[local_col] += 1;
                }
            }
        }
        row_offset += shard_n_rows;
    }

    CscArrays {
        shape: (n_rows_total, chunk_n_cols),
        indptr,
        indices,
        data,
    }
}

/// Drive [`transpose_column_chunk`] over every chunk, the way the deleted
/// `streaming_csr_to_csc_iter_with_cap` did, and concatenate the result.
///
/// Test-only, and the **oracle**: `csc_builder`'s proptests compare the
/// builder's output against this chunk for chunk, which is the executable form
/// of the claim that the replacement emits the same bytes. It is kept as a
/// reference implementation rather than production code precisely because it
/// is the `2 * nnz * n_chunks` algorithm — correct, and the wrong thing to
/// call.
#[cfg(test)]
pub(crate) fn reference_chunks(
    shards: &[ScxCsr],
    n_rows_total: usize,
    n_cols: usize,
    max_memory_bytes: usize,
    max_cols: usize,
) -> Result<Vec<(usize, CscArrays)>, TransposeError> {
    for shard in shards {
        if shard.shape.1 != n_cols {
            return Err(TransposeError::ShapeMismatch {
                shard_cols: shard.shape.1,
                expected_cols: n_cols,
            });
        }
    }
    let chunk_cols = compute_chunk_cols_with_cap(n_rows_total, max_memory_bytes, max_cols)?;
    let mut out = Vec::new();
    let mut col_start = 0usize;
    while col_start < n_cols {
        let col_end = col_start.saturating_add(chunk_cols).min(n_cols);
        out.push((
            col_start,
            transpose_column_chunk(shards, n_rows_total, col_start, col_end),
        ));
        col_start = col_end;
    }
    Ok(out)
}

/// The same, concatenated into one whole-matrix `CscArrays`.
#[cfg(test)]
pub(crate) fn reference_whole(
    shards: &[ScxCsr],
    n_rows_total: usize,
    n_cols: usize,
    max_memory_bytes: usize,
) -> Result<CscArrays, TransposeError> {
    let chunks = reference_chunks(shards, n_rows_total, n_cols, max_memory_bytes, 0)?;
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for (_, chunk) in chunks {
        let base = *indptr.last().expect("seeded");
        indptr.extend(chunk.indptr[1..].iter().map(|v| v + base));
        indices.extend_from_slice(&chunk.indices);
        data.extend_from_slice(&chunk.data);
    }
    Ok(CscArrays {
        shape: (n_rows_total, n_cols),
        indptr,
        indices,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::ScxCsr;

    /// Helper: build a CSR from dense row-major data.
    fn dense_to_csr(dense: &[f32], n_rows: usize, n_cols: usize) -> ScxCsr {
        crate::convert::dense_to_csr(dense, n_rows, n_cols).unwrap()
    }

    /// Helper: convert CscArrays to dense column-major, then to row-major.
    fn csc_to_dense(csc: &CscArrays) -> Vec<f32> {
        let (n_rows, n_cols) = csc.shape;
        let mut dense = vec![0.0f32; n_rows * n_cols];
        for col in 0..n_cols {
            let start = csc.indptr[col] as usize;
            let end = csc.indptr[col + 1] as usize;
            for j in start..end {
                let row = csc.indices[j] as usize;
                dense[row * n_cols + col] = csc.data[j];
            }
        }
        dense
    }

    #[test]
    fn test_transpose_roundtrip() {
        // 3x5 matrix:
        // row 0: [0, 5, 0, 10, 0]
        // row 1: [1, 0, 3,  0, 7]
        // row 2: [0, 0, 2,  0, 0]
        #[rustfmt::skip]
        let dense = vec![
            0.0, 5.0, 0.0, 10.0, 0.0,
            1.0, 0.0, 3.0,  0.0, 7.0,
            0.0, 0.0, 2.0,  0.0, 0.0,
        ];
        let csr = dense_to_csr(&dense, 3, 5);
        let csc = csr_to_csc(&csr);

        assert_eq!(csc.shape, (3, 5));
        assert_eq!(csc.indptr.len(), 6); // n_cols + 1
        assert_eq!(csc.indices.len(), csr.nnz());
        assert_eq!(csc.data.len(), csr.nnz());

        // Verify round-trip via dense
        let recovered = csc_to_dense(&csc);
        assert_eq!(recovered, dense);
    }

    #[test]
    fn test_transpose_empty() {
        // Truly empty: 0x0
        let csr = ScxCsr::new((0, 0), vec![0], vec![], vec![]).unwrap();
        let csc = csr_to_csc(&csr);
        assert_eq!(csc.shape, (0, 0));
        assert_eq!(csc.indptr, vec![0]);
        assert_eq!(csc.indices.len(), 0);

        // Shaped empty: 5x10 with no nnz
        let csr2 = ScxCsr::new((5, 10), vec![0, 0, 0, 0, 0, 0], vec![], vec![]).unwrap();
        let csc2 = csr_to_csc(&csr2);
        assert_eq!(csc2.shape, (5, 10));
        assert_eq!(csc2.indptr.len(), 11);
        assert!(csc2.indptr.iter().all(|&v| v == 0));
        assert_eq!(csc2.indices.len(), 0);
    }

    #[test]
    fn test_transpose_single_row() {
        // 1x4 matrix: [0, 5, 0, 9]
        let csr = dense_to_csr(&[0.0, 5.0, 0.0, 9.0], 1, 4);
        let csc = csr_to_csc(&csr);

        assert_eq!(csc.shape, (1, 4));
        assert_eq!(csc.indptr, vec![0, 0, 1, 1, 2]); // cols 0,2 empty; cols 1,3 have 1 entry
        assert_eq!(csc.indices, vec![0, 0]); // both in row 0
        assert_eq!(csc.data, vec![5.0, 9.0]);
    }

    #[test]
    fn test_transpose_chunked() {
        // Create a 4x6 matrix with known data
        #[rustfmt::skip]
        let dense = vec![
            1.0, 0.0, 2.0, 0.0, 0.0, 3.0,
            0.0, 4.0, 0.0, 5.0, 0.0, 0.0,
            6.0, 0.0, 0.0, 0.0, 7.0, 0.0,
            0.0, 0.0, 8.0, 9.0, 0.0, 0.0,
        ];
        let csr = dense_to_csr(&dense, 4, 6);

        // In-memory reference
        let csc_ref = csr_to_csc(&csr);
        let dense_ref = csc_to_dense(&csc_ref);
        assert_eq!(dense_ref, dense);

        // Streaming with small memory limit — force multiple passes
        // 4 rows × 12 bytes = 48 bytes/col. Use 100 bytes → ~2 cols per chunk → 3 passes
        let csc_streamed = reference_whole(std::slice::from_ref(&csr), 4, 6, 100).unwrap();
        let dense_streamed = csc_to_dense(&csc_streamed);
        assert_eq!(dense_streamed, dense);

        // Verify indptr/indices/data match in-memory result
        assert_eq!(csc_streamed.indptr, csc_ref.indptr);
        assert_eq!(csc_streamed.indices, csc_ref.indices);
        assert_eq!(csc_streamed.data, csc_ref.data);
    }

    #[test]
    fn test_transpose_chunked_multi_shard() {
        // Two shards covering rows 0-1 and 2-3 of a 4x4 matrix
        #[rustfmt::skip]
        let dense = vec![
            1.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 3.0, 4.0,
            5.0, 0.0, 6.0, 0.0,
            0.0, 7.0, 0.0, 8.0,
        ];
        let full_csr = dense_to_csr(&dense, 4, 4);

        let shard0 = full_csr.row_slice(0, 2).unwrap();
        let shard1 = full_csr.row_slice(2, 4).unwrap();

        // Reference: in-memory transpose of full matrix
        let csc_ref = csr_to_csc(&full_csr);
        let dense_ref = csc_to_dense(&csc_ref);
        assert_eq!(dense_ref, dense);

        // Streaming multi-shard (generous memory)
        let csc_streamed =
            reference_whole(&[shard0.clone(), shard1.clone()], 4, 4, 1_000_000).unwrap();
        assert_eq!(csc_to_dense(&csc_streamed), dense);

        // Streaming multi-shard (tiny memory → 1 col per pass)
        let csc_tiny = reference_whole(&[shard0, shard1], 4, 4, 48).unwrap();
        assert_eq!(csc_to_dense(&csc_tiny), dense);
    }

    #[test]
    fn test_transpose_memory_limit_too_small() {
        let csr = dense_to_csr(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        // 2 rows × 12 = 24 bytes per col. Budget of 10 → not enough for 1 col.
        let err = reference_whole(&[csr], 2, 2, 10).unwrap_err();
        assert!(matches!(err, TransposeError::MemoryLimitTooSmall { .. }));
    }

    #[test]
    fn test_transpose_shape_mismatch() {
        let shard_a = dense_to_csr(&[1.0, 2.0], 1, 2);
        let shard_b = dense_to_csr(&[1.0, 2.0, 3.0], 1, 3);
        let err = reference_whole(&[shard_a, shard_b], 2, 2, 1_000_000).unwrap_err();
        assert!(matches!(err, TransposeError::ShapeMismatch { .. }));
    }

    // CLI9: an empty matrix (0 rows) makes compute_chunk_cols return the
    // usize::MAX "all columns" sentinel; the col_end computation must not
    // overflow-panic. Transposing a 0×3 matrix yields an empty CSC.
    #[test]
    fn test_transpose_empty_matrix_no_overflow() {
        let csc = reference_whole(&[], 0, 3, 1_000_000).unwrap();
        assert_eq!(csc.indptr, vec![0i64; 4]);
        assert!(csc.indices.is_empty());
        assert!(csc.data.is_empty());
    }

    // CLI9: a memory budget that admits exactly one column per pass still
    // transposes correctly (multiple single-column passes).
    #[test]
    fn test_transpose_single_column_budget() {
        // 2×3 matrix; 2 rows × 12 = 24 bytes/col → budget 24 → 1 col/pass.
        let shard = dense_to_csr(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0], 2, 3);
        let csc = reference_whole(&[shard], 2, 3, 24).unwrap();
        let reference = reference_whole(
            &[dense_to_csr(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0], 2, 3)],
            2,
            3,
            1_000_000,
        )
        .unwrap();
        assert_eq!(csc.indptr, reference.indptr);
        assert_eq!(csc.indices, reference.indices);
        assert_eq!(csc.data, reference.data);
    }

    #[test]
    fn test_compute_chunk_cols_with_cap_caps_when_smaller() {
        // 100 rows × 12 = 1200 bytes/col. Budget 12_000 → 10 cols.
        // Cap at 3 → expect 3.
        let n = compute_chunk_cols_with_cap(100, 12_000, 3).unwrap();
        assert_eq!(n, 3);
    }

    #[test]
    fn test_compute_chunk_cols_with_cap_uses_memory_when_smaller() {
        // 100 rows × 12 = 1200 bytes/col. Budget 12_000 → 10 cols.
        // Cap at 50 → memory wins, expect 10.
        let n = compute_chunk_cols_with_cap(100, 12_000, 50).unwrap();
        assert_eq!(n, 10);
    }

    #[test]
    fn test_compute_chunk_cols_with_cap_zero_means_no_cap() {
        let n_no_cap = compute_chunk_cols_with_cap(100, 12_000, 0).unwrap();
        let n_max_cap = compute_chunk_cols_with_cap(100, 12_000, usize::MAX).unwrap();
        assert_eq!(n_no_cap, n_max_cap);
    }

    #[test]
    fn test_streaming_iter_with_cap_emits_capped_chunks() {
        // 4×6 matrix; cap at 3 cols/chunk → ceil(6 / 3) = 2 chunks.
        #[rustfmt::skip]
        let dense = vec![
            1.0, 0.0, 2.0, 0.0, 0.0, 3.0,
            0.0, 4.0, 0.0, 5.0, 0.0, 0.0,
            6.0, 0.0, 0.0, 0.0, 7.0, 0.0,
            0.0, 0.0, 8.0, 9.0, 0.0, 0.0,
        ];
        let csr = dense_to_csr(&dense, 4, 6);
        let shards = [csr];

        // Generous memory bound — cap should drive the chunking.
        let chunks = reference_chunks(&shards, 4, 6, 1_000_000, 3).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].0, 0);
        assert_eq!(chunks[0].1.shape, (4, 3));
        assert_eq!(chunks[1].0, 3);
        assert_eq!(chunks[1].1.shape, (4, 3));
        let (chunk0, chunk1) = (&chunks[0].1, &chunks[1].1);

        // Reconstruct and verify.
        let mut reconstructed = vec![0.0f32; 24]; // 4×6
        for (chunk_idx, chunk) in [chunk0, chunk1].iter().enumerate() {
            let col_offset = chunk_idx * 3;
            for col in 0..chunk.shape.1 {
                let s = chunk.indptr[col] as usize;
                let e = chunk.indptr[col + 1] as usize;
                for j in s..e {
                    let row = chunk.indices[j] as usize;
                    reconstructed[row * 6 + col + col_offset] = chunk.data[j];
                }
            }
        }
        assert_eq!(reconstructed, dense);
    }

    #[test]
    fn test_streaming_iter_with_cap_uneven_last_chunk() {
        // 2×7 with cap=3 → chunks [0..3), [3..6), [6..7) (last is 1 col)
        let dense = vec![
            1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0,
        ];
        let csr = dense_to_csr(&dense, 2, 7);
        let shards = [csr];

        let chunks = reference_chunks(&shards, 2, 7, 1_000_000, 3).unwrap();
        // Three chunks of sizes 3, 3, 1 starting at 0, 3, 6 — the short last
        // chunk is the boundary an off-by-one lands on.
        assert_eq!(
            chunks.iter().map(|(_, c)| c.shape.1).collect::<Vec<_>>(),
            vec![3, 3, 1]
        );
        assert_eq!(
            chunks.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
            vec![0, 3, 6]
        );
    }

    #[test]
    fn test_streaming_iterator() {
        // 3x4 matrix split into chunks
        #[rustfmt::skip]
        let dense = vec![
            1.0, 0.0, 2.0, 0.0,
            0.0, 3.0, 0.0, 4.0,
            5.0, 0.0, 6.0, 0.0,
        ];
        let csr = dense_to_csr(&dense, 3, 4);

        // Force 2 cols per chunk → 2 iterations
        // 3 rows × 12 = 36 bytes/col → budget 72 → 2 cols per chunk
        let shards = [csr];
        let chunks: Vec<CscArrays> = reference_chunks(&shards, 3, 4, 72, 0)
            .unwrap()
            .into_iter()
            .map(|(_, c)| c)
            .collect();

        assert_eq!(chunks.len(), 2);

        // First chunk: cols 0-1
        assert_eq!(chunks[0].shape, (3, 2));
        // Second chunk: cols 2-3
        assert_eq!(chunks[1].shape, (3, 2));

        // Reconstruct full dense from chunks and verify
        let mut reconstructed = vec![0.0f32; 12]; // 3×4
        for (chunk_idx, chunk) in chunks.iter().enumerate() {
            let col_offset = chunk_idx * 2;
            for col in 0..chunk.shape.1 {
                let start = chunk.indptr[col] as usize;
                let end = chunk.indptr[col + 1] as usize;
                for j in start..end {
                    let row = chunk.indices[j] as usize;
                    let global_col = col + col_offset;
                    reconstructed[row * 4 + global_col] = chunk.data[j];
                }
            }
        }
        assert_eq!(reconstructed, dense);
    }
}
