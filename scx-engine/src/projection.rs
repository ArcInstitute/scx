// Gene/column projection during CSR decode.
//
// Reduces the number of columns extracted from each shard by only keeping
// entries whose column index appears in the requested gene set.

use arrow::array::{RecordBatch, UInt32Array};
use arrow::compute;
use scx_format_io::catalog::FullCatalogEntry;
use scx_sparse::{ScxCsc, ScxCsr};

use crate::error::Result;
use crate::reader::SectionReader;

/// Project a single CSR row, keeping only entries whose column index
/// appears in `gene_set`.
///
/// # Precondition
///
/// Both `indices` (column indices for one row) and `gene_set` MUST be
/// sorted ascending. The merge scan uses a monotonic `gi` pointer that
/// never decreases; on unsorted `indices` it silently drops every
/// column that follows a larger predecessor in the row, producing a
/// wrong-but-not-noisy result. Callers passing a scipy `csr_matrix`
/// must either check `has_sorted_indices` or call `.sort_indices()` /
/// `.sorted_indices()` before extracting `indices`. The canonical pyscx
/// boundary helper is `pyscx::convert::ensure_csr`, which already
/// enforces this. Debug builds catch violations via `debug_assert!`;
/// release builds skip the check, so the boundary fix is what holds
/// correctness in production.
///
/// Uses a sorted merge scan — O(nnz_row + n_genes).
///
/// Returns `(projected_indices, projected_data)` where indices are
/// remapped to `0..gene_set.len()` (position in gene_set).
pub fn project_csr_row(indices: &[i32], data: &[f32], gene_set: &[u32]) -> (Vec<i32>, Vec<f32>) {
    debug_assert!(
        indices.windows(2).all(|w| w[0] <= w[1]),
        "project_csr_row: row indices are not sorted ascending — unsorted input \
         silently produces wrong results because the merge scan uses a monotonic \
         pointer. Sort indices at the caller (e.g. via scipy `sort_indices()` or \
         `pyscx::convert::ensure_csr`) before invoking."
    );
    debug_assert!(
        gene_set.windows(2).all(|w| w[0] <= w[1]),
        "project_csr_row: gene_set is not sorted ascending — the merge scan \
         requires it. `project_csr` sorts internally; direct callers must too."
    );

    let mut out_indices = Vec::new();
    let mut out_data = Vec::new();

    let mut gi = 0; // pointer into gene_set

    for (pos, (&col_idx, &val)) in indices.iter().zip(data.iter()).enumerate() {
        let col = col_idx as u32;
        // Advance gene_set pointer past values smaller than current column
        while gi < gene_set.len() && gene_set[gi] < col {
            gi += 1;
        }
        if gi >= gene_set.len() {
            break; // remaining row indices are all beyond gene_set
        }
        if gene_set[gi] == col {
            // Remap to position within gene_set
            out_indices.push(gi as i32);
            out_data.push(val);
        }
        // If gene_set[gi] > col, this column is not in the gene set — skip it
        let _ = pos; // suppress unused variable warning
    }

    (out_indices, out_data)
}

/// Project an entire `ScxCsr` matrix to keep only the specified gene columns.
///
/// `gene_indices` are the original column indices to retain. They are sorted
/// internally if not already sorted. The output CSR has
/// `n_cols = gene_indices.len()` with column indices remapped to `0..n_cols`.
pub fn project_csr(csr: &ScxCsr, gene_indices: &[u32]) -> ScxCsr {
    // Sort gene indices (stable sort to preserve order of equal elements)
    let mut sorted_genes: Vec<u32> = gene_indices.to_vec();
    sorted_genes.sort_unstable();
    sorted_genes.dedup();

    let n_rows = csr.n_rows();
    let n_cols = sorted_genes.len();

    let mut new_indptr = Vec::with_capacity(n_rows + 1);
    new_indptr.push(0i64);
    let mut new_indices = Vec::new();
    let mut new_data = Vec::new();

    for row in 0..n_rows {
        let start = csr.indptr[row] as usize;
        let end = csr.indptr[row + 1] as usize;

        let (row_indices, row_data) = project_csr_row(
            &csr.indices[start..end],
            &csr.data[start..end],
            &sorted_genes,
        );

        new_indices.extend_from_slice(&row_indices);
        new_data.extend_from_slice(&row_data);
        let prev = *new_indptr.last().unwrap();
        new_indptr.push(prev + row_indices.len() as i64);
    }

    ScxCsr::new_unchecked((n_rows, n_cols), new_indptr, new_indices, new_data)
}

/// Project the var (gene) RecordBatch to keep only the specified gene rows.
///
/// Uses `arrow::compute::take()` for efficient row selection.
pub fn project_var(var: &RecordBatch, gene_indices: &[u32]) -> Result<RecordBatch> {
    let mut sorted_indices: Vec<u32> = gene_indices.to_vec();
    sorted_indices.sort_unstable();
    sorted_indices.dedup();
    let take_indices = UInt32Array::from(sorted_indices);
    let columns: Vec<_> = var
        .columns()
        .iter()
        .map(|col| compute::take(col.as_ref(), &take_indices, None))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    Ok(RecordBatch::try_new(var.schema(), columns)?)
}

/// Decode a shard and apply gene projection in one step.
///
/// Decodes the full shard via `reader.read_shard_from_entry()`, then applies
/// per-row projection to extract only the requested gene columns. Returns
/// scipy-compatible `(indptr, indices, data)` with remapped column indices.
///
/// **Note**: True decode-time projection (filtering during FOR-BP decode)
/// would be more efficient but requires modifying scx-codec. Start with
/// post-decode projection; optimize into the codec layer if benchmarks
/// show it matters.
pub fn decode_shard_projected(
    reader: &dyn SectionReader,
    entry: &FullCatalogEntry,
    gene_indices: &[u32],
) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
    let (indptr, indices, data) = reader.read_shard_from_entry(entry)?;

    // Sort gene indices for merge scan
    let mut sorted_genes: Vec<u32> = gene_indices.to_vec();
    sorted_genes.sort_unstable();
    sorted_genes.dedup();

    let n_rows = indptr.len() - 1;
    let mut new_indptr = Vec::with_capacity(n_rows + 1);
    new_indptr.push(0i64);
    let mut new_indices = Vec::new();
    let mut new_data = Vec::new();

    for row in 0..n_rows {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;

        let (row_indices, row_data) =
            project_csr_row(&indices[start..end], &data[start..end], &sorted_genes);

        new_indices.extend_from_slice(&row_indices);
        new_data.extend_from_slice(&row_data);
        let prev = *new_indptr.last().unwrap();
        new_indptr.push(prev + row_indices.len() as i64);
    }

    Ok((new_indptr, new_indices, new_data))
}

// ---------------------------------------------------------------------------
// CSC column projection — column-major counterpart to project_csr
// ---------------------------------------------------------------------------

/// Borrowed-slice view of a single CSC column.
///
/// Returns `(indices, data)` for column `col` — the row indices and
/// values stored under that column. Zero-copy hot path; useful when a
/// caller iterates one column at a time without materializing a new
/// `ScxCsc`.
///
/// Panics if `col >= indptr.len() - 1`. Callers are expected to bound
/// `col < n_cols` from a known-correct `ScxCsc`.
pub fn project_csc_column<'a>(
    indptr: &[i64],
    indices: &'a [i32],
    data: &'a [f32],
    col: usize,
) -> (&'a [i32], &'a [f32]) {
    let start = indptr[col] as usize;
    let end = indptr[col + 1] as usize;
    (&indices[start..end], &data[start..end])
}

/// Project an entire `ScxCsc` matrix to keep only the specified gene
/// (column) indices.
///
/// Mirrors [`project_csr`] for the column axis. `gene_indices` are the
/// original column indices to retain; sorted internally if not already
/// sorted, deduplicated. Output column indices are remapped to
/// `0..gene_indices.len()` (position in the deduplicated/sorted set).
/// Row indices in the output are unchanged from the input (CSC indices
/// are already global row IDs).
pub fn project_csc(csc: &ScxCsc, gene_indices: &[u32]) -> ScxCsc {
    let mut sorted_genes: Vec<u32> = gene_indices.to_vec();
    sorted_genes.sort_unstable();
    sorted_genes.dedup();

    let n_rows = csc.n_rows();
    let n_cols_out = sorted_genes.len();

    // Pre-compute total nnz to size buffers without growth churn.
    let mut total_nnz: usize = 0;
    for &g in &sorted_genes {
        let g = g as usize;
        if g >= csc.n_cols() {
            // Out-of-range gene index contributes nothing; skip.
            continue;
        }
        total_nnz += (csc.indptr[g + 1] - csc.indptr[g]) as usize;
    }

    let mut new_indptr = Vec::with_capacity(n_cols_out + 1);
    new_indptr.push(0i64);
    let mut new_indices = Vec::with_capacity(total_nnz);
    let mut new_data = Vec::with_capacity(total_nnz);

    for &g in &sorted_genes {
        let g = g as usize;
        if g < csc.n_cols() {
            let (col_idx, col_data) = project_csc_column(&csc.indptr, &csc.indices, &csc.data, g);
            new_indices.extend_from_slice(col_idx);
            new_data.extend_from_slice(col_data);
        }
        new_indptr.push(new_data.len() as i64);
    }

    ScxCsc::new_unchecked((n_rows, n_cols_out), new_indptr, new_indices, new_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use scx_codec::{CodecId, ValueEncoding};
    use scx_format_io::header::FileHeader;
    use scx_format_io::writer::ScxWriter;
    use scx_format_io::ScxReader;
    use std::sync::Arc;

    // -----------------------------------------------------------------------
    // D1: project_csr_row
    // -----------------------------------------------------------------------

    #[test]
    fn project_row_basic() {
        // Row with columns [0, 3, 5, 7], select genes {3, 5}
        let indices = vec![0, 3, 5, 7];
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let gene_set = vec![3, 5];

        let (out_idx, out_data) = project_csr_row(&indices, &data, &gene_set);
        assert_eq!(out_idx, vec![0, 1]); // remapped: 3→0, 5→1
        assert_eq!(out_data, vec![2.0, 3.0]);
    }

    #[test]
    fn project_row_no_match() {
        // Row has columns [0, 1, 2], gene_set is {5, 6}
        let indices = vec![0, 1, 2];
        let data = vec![1.0, 2.0, 3.0];
        let gene_set = vec![5, 6];

        let (out_idx, out_data) = project_csr_row(&indices, &data, &gene_set);
        assert!(out_idx.is_empty());
        assert!(out_data.is_empty());
    }

    #[test]
    fn project_row_all_match() {
        let indices = vec![2, 4, 6];
        let data = vec![10.0, 20.0, 30.0];
        let gene_set = vec![2, 4, 6];

        let (out_idx, out_data) = project_csr_row(&indices, &data, &gene_set);
        assert_eq!(out_idx, vec![0, 1, 2]);
        assert_eq!(out_data, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn project_row_empty_gene_set() {
        let indices = vec![0, 1, 2];
        let data = vec![1.0, 2.0, 3.0];
        let gene_set: Vec<u32> = vec![];

        let (out_idx, out_data) = project_csr_row(&indices, &data, &gene_set);
        assert!(out_idx.is_empty());
        assert!(out_data.is_empty());
    }

    #[test]
    fn project_row_empty_row() {
        let indices: Vec<i32> = vec![];
        let data: Vec<f32> = vec![];
        let gene_set = vec![0, 1, 2];

        let (out_idx, out_data) = project_csr_row(&indices, &data, &gene_set);
        assert!(out_idx.is_empty());
        assert!(out_data.is_empty());
    }

    // -----------------------------------------------------------------------
    // D1: project_csr
    // -----------------------------------------------------------------------

    fn sample_csr() -> ScxCsr {
        // 3×10 matrix:
        // row 0: cols [1, 3] = [5.0, 10.0]
        // row 1: cols [0, 2, 4] = [1.0, 3.0, 7.0]
        // row 2: cols [2] = [2.0]
        ScxCsr::new(
            (3, 10),
            vec![0, 2, 5, 6],
            vec![1, 3, 0, 2, 4, 2],
            vec![5.0, 10.0, 1.0, 3.0, 7.0, 2.0],
        )
        .unwrap()
    }

    #[test]
    fn project_3_genes_from_10() {
        let csr = sample_csr();
        // Select genes {0, 2, 4} → 3 columns
        let projected = project_csr(&csr, &[0, 2, 4]);
        assert_eq!(projected.shape, (3, 3));
        assert_eq!(projected.n_rows(), 3);
        assert_eq!(projected.n_cols(), 3);

        // row 0: original [1, 3] → none of {0,2,4} match 1 or 3 → empty
        assert_eq!(projected.indptr[0], 0);
        assert_eq!(projected.indptr[1], 0);

        // row 1: original [0, 2, 4] → all match → remapped [0, 1, 2]
        assert_eq!(projected.indptr[2], 3);
        assert_eq!(&projected.indices[0..3], &[0, 1, 2]);
        assert_eq!(&projected.data[0..3], &[1.0, 3.0, 7.0]);

        // row 2: original [2] → matches gene_set[1]=2 → remapped [1]
        assert_eq!(projected.indptr[3], 4);
        assert_eq!(projected.indices[3], 1);
        assert_eq!(projected.data[3], 2.0);
    }

    #[test]
    fn project_gene_not_present_in_any_row() {
        let csr = sample_csr();
        // Gene index 9 doesn't appear in any row
        let projected = project_csr(&csr, &[9]);
        assert_eq!(projected.shape, (3, 1));
        assert_eq!(projected.nnz(), 0);
        assert_eq!(projected.indptr, vec![0, 0, 0, 0]);
    }

    #[test]
    fn project_preserves_values_exactly() {
        let csr = sample_csr();
        let projected = project_csr(&csr, &[2]);
        // Only gene 2: row 0 has nothing, row 1 has 3.0, row 2 has 2.0
        assert_eq!(projected.shape, (3, 1));
        assert_eq!(projected.nnz(), 2);
        assert_eq!(projected.data, vec![3.0, 2.0]);
    }

    #[test]
    fn project_column_indices_remapped() {
        let csr = sample_csr();
        // Select genes {1, 4} → remapped to {0, 1}
        let projected = project_csr(&csr, &[1, 4]);
        assert_eq!(projected.shape, (3, 2));
        // row 0: col 1 → remapped 0
        assert_eq!(projected.indices[0], 0);
        // row 1: col 4 → remapped 1
        assert_eq!(projected.indices[1], 1);
    }

    #[test]
    fn project_empty_gene_indices() {
        let csr = sample_csr();
        let projected = project_csr(&csr, &[]);
        assert_eq!(projected.shape, (3, 0));
        assert_eq!(projected.nnz(), 0);
    }

    #[test]
    fn project_unsorted_gene_indices() {
        let csr = sample_csr();
        // Unsorted input: {4, 0, 2} should be handled correctly
        let projected = project_csr(&csr, &[4, 0, 2]);
        // Should produce same result as sorted {0, 2, 4}
        let projected_sorted = project_csr(&csr, &[0, 2, 4]);
        assert_eq!(projected.shape, projected_sorted.shape);
        assert_eq!(projected.indptr, projected_sorted.indptr);
        assert_eq!(projected.indices, projected_sorted.indices);
        assert_eq!(projected.data, projected_sorted.data);
    }

    #[test]
    fn project_large_gene_count_to_small() {
        // Simulate "3 genes from 30K" scenario
        let n_vars = 30_000;
        let n_rows = 5;
        // Build a CSR where each row has indices [0, 100, 200] with values [1.0, 2.0, 3.0]
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for _ in 0..n_rows {
            indices.extend_from_slice(&[0, 100, 200]);
            data.extend_from_slice(&[1.0, 2.0, 3.0]);
            indptr.push(*indptr.last().unwrap() + 3);
        }
        let csr = ScxCsr::new_unchecked((n_rows, n_vars), indptr, indices, data);

        // Project to 3 specific genes
        let projected = project_csr(&csr, &[0, 100, 200]);
        assert_eq!(projected.shape, (n_rows, 3));
        assert_eq!(projected.nnz(), n_rows * 3);
        // All projected indices should be 0, 1, 2 (remapped)
        for row in 0..n_rows {
            let start = projected.indptr[row] as usize;
            let end = projected.indptr[row + 1] as usize;
            assert_eq!(&projected.indices[start..end], &[0, 1, 2]);
            assert_eq!(&projected.data[start..end], &[1.0, 2.0, 3.0]);
        }
    }

    // -----------------------------------------------------------------------
    // D1: project_var
    // -----------------------------------------------------------------------

    fn sample_var(n: usize) -> RecordBatch {
        let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
        let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    #[test]
    fn project_var_selects_correct_rows() {
        let var = sample_var(10);
        let projected = project_var(&var, &[0, 3, 7]).unwrap();
        assert_eq!(projected.num_rows(), 3);

        let gene_ids = projected
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(gene_ids.value(0), "gene_0");
        assert_eq!(gene_ids.value(1), "gene_3");
        assert_eq!(gene_ids.value(2), "gene_7");
    }

    #[test]
    fn project_var_empty_indices() {
        let var = sample_var(10);
        let projected = project_var(&var, &[]).unwrap();
        assert_eq!(projected.num_rows(), 0);
    }

    #[test]
    fn project_var_single_gene() {
        let var = sample_var(5);
        let projected = project_var(&var, &[2]).unwrap();
        assert_eq!(projected.num_rows(), 1);

        let gene_ids = projected
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(gene_ids.value(0), "gene_2");
    }

    // -----------------------------------------------------------------------
    // D2: decode_shard_projected
    // -----------------------------------------------------------------------

    fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
        FileHeader::new_single_modality(n_obs, n_vars, nnz, 16384, 0, 0)
    }

    fn sample_obs(n: usize) -> RecordBatch {
        let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
        let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    fn write_test_file(dir: &tempfile::TempDir, n_obs: usize, n_vars: usize) -> std::path::PathBuf {
        let path = dir.path().join("test.scx");
        let header = sample_header(n_obs as u64, n_vars as u64, (n_obs * 2) as u64);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();

        // Build shard data: each row has 2 entries at deterministic column positions
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for row in 0..n_obs {
            let col0 = (row * 2) % n_vars;
            let col1 = (row * 2 + 1) % n_vars;
            // Ensure sorted column order within row
            let (c0, c1) = if col0 < col1 {
                (col0, col1)
            } else {
                (col1, col0)
            };
            indices.push(c0 as u32);
            indices.push(c1 as u32);
            values.push(((row + 1) % 256) as u8);
            values.push(((row + 2) % 256) as u8);
            indptr.push(indptr.last().unwrap() + 2);
        }

        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        writer.finish().unwrap();
        path
    }

    #[test]
    fn decode_projected_matches_decode_then_project() {
        let dir = tempfile::tempdir().unwrap();
        let n_obs = 10;
        let n_vars = 20;
        let path = write_test_file(&dir, n_obs, n_vars);
        let reader = ScxReader::open(&path).unwrap();

        let gene_indices = vec![0u32, 2, 5, 10, 15];

        // Method 1: decode full shard, then project
        let shards = reader.catalog().shards_sorted();
        let entry = shards[0];
        let (indptr1, indices1, data1) = reader.read_shard_from_entry(entry).unwrap();
        let full_csr = ScxCsr::new_unchecked((n_obs, n_vars), indptr1, indices1, data1);
        let projected = project_csr(&full_csr, &gene_indices);

        // Method 2: decode_shard_projected
        let (indptr2, indices2, data2) =
            decode_shard_projected(&reader, entry, &gene_indices).unwrap();

        // They must produce the same result
        assert_eq!(projected.indptr, indptr2);
        assert_eq!(projected.indices, indices2);
        assert_eq!(projected.data, data2);
    }

    // -----------------------------------------------------------------------
    // project_csc / project_csc_column
    // -----------------------------------------------------------------------

    fn sample_csc() -> ScxCsc {
        // 3×5 matrix (transpose of sample_csr's 3x10 — different fixture
        // for clarity here):
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
    fn project_csc_column_borrowed_slice() {
        let csc = sample_csc();
        let (col2_idx, col2_data) = project_csc_column(&csc.indptr, &csc.indices, &csc.data, 2);
        assert_eq!(col2_idx, &[1, 2]);
        assert_eq!(col2_data, &[3.0, 2.0]);

        let (col0_idx, col0_data) = project_csc_column(&csc.indptr, &csc.indices, &csc.data, 0);
        assert_eq!(col0_idx, &[1]);
        assert_eq!(col0_data, &[1.0]);
    }

    #[test]
    fn project_csc_basic() {
        let csc = sample_csc();
        // Select columns {0, 2, 4} → 3 columns of output, in remapped
        // 0..3 order.
        let projected = project_csc(&csc, &[0, 2, 4]);
        assert_eq!(projected.shape, (3, 3));
        assert_eq!(projected.nnz(), 4);
        // col 0 (orig 0): row 1 -> 1.0
        assert_eq!(&projected.indptr[0..2], &[0, 1]);
        assert_eq!(&projected.indices[0..1], &[1]);
        assert_eq!(&projected.data[0..1], &[1.0]);
        // col 1 (orig 2): row 1 -> 3.0, row 2 -> 2.0
        assert_eq!(projected.indptr[2], 3);
        assert_eq!(&projected.indices[1..3], &[1, 2]);
        assert_eq!(&projected.data[1..3], &[3.0, 2.0]);
        // col 2 (orig 4): row 1 -> 7.0
        assert_eq!(projected.indptr[3], 4);
        assert_eq!(projected.indices[3], 1);
        assert_eq!(projected.data[3], 7.0);
    }

    #[test]
    fn project_csc_unsorted_input() {
        let csc = sample_csc();
        let a = project_csc(&csc, &[4, 0, 2]);
        let b = project_csc(&csc, &[0, 2, 4]);
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.indptr, b.indptr);
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn project_csc_dedup() {
        let csc = sample_csc();
        let projected = project_csc(&csc, &[2, 2, 2]);
        // Three duplicates of column 2 should collapse to one.
        assert_eq!(projected.shape, (3, 1));
        assert_eq!(projected.nnz(), 2);
    }

    #[test]
    fn project_csc_empty_indices() {
        let csc = sample_csc();
        let projected = project_csc(&csc, &[]);
        assert_eq!(projected.shape, (3, 0));
        assert_eq!(projected.nnz(), 0);
    }

    #[test]
    fn project_csc_out_of_range_gene_skipped() {
        let csc = sample_csc(); // 5 columns
                                // Gene 99 is past the end; should be skipped, leaving an empty
                                // projected column.
        let projected = project_csc(&csc, &[2, 99]);
        assert_eq!(projected.shape, (3, 2));
        // Output col 0 (orig 2) has 2 nnz; col 1 (orig 99) has 0.
        assert_eq!(projected.indptr, vec![0, 2, 2]);
    }

    /// Densified equivalence: project then to_dense == to_dense then
    /// column-slice.
    #[test]
    fn project_csc_dense_equivalence() {
        let csc = sample_csc();
        let dense_full = csc.to_dense().unwrap(); // row-major, 3×5
        let projected = project_csc(&csc, &[1, 3]);
        let dense_proj = projected.to_dense().unwrap();
        // Reference: dense matrix's columns 1 and 3 in column-major sense
        let mut expected = vec![0.0f32; 3 * 2];
        for r in 0..3 {
            expected[r * 2] = dense_full[r * 5 + 1];
            expected[r * 2 + 1] = dense_full[r * 5 + 3];
        }
        assert_eq!(dense_proj, expected);
    }

    #[test]
    fn decode_projected_empty_genes() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 5, 10);
        let reader = ScxReader::open(&path).unwrap();

        let shards = reader.catalog().shards_sorted();
        let entry = shards[0];
        let (indptr, indices, data) = decode_shard_projected(&reader, entry, &[]).unwrap();

        assert_eq!(indptr.len(), 6); // 5 rows + 1
        assert!(indices.is_empty());
        assert!(data.is_empty());
        // All indptr values should be 0 (no nnz per row)
        assert!(indptr.iter().all(|&v| v == 0));
    }
}
