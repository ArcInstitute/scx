// Gene/column projection during CSR decode.
//
// Reduces the number of columns extracted from each shard by only keeping
// entries whose column index appears in the requested gene set. See
// SPEC.md §8.4 for the projection design.

use arrow::array::{RecordBatch, UInt32Array};
use arrow::compute;
use scx_format::catalog::FullCatalogEntry;
use scx_format::ScxReader;
use scx_sparse::ScxCsr;

use crate::error::Result;

/// Project a single CSR row, keeping only entries whose column index
/// appears in `gene_set`.
///
/// Both `indices` (sorted column indices for one row) and `gene_set`
/// must be sorted. Uses a sorted merge scan — O(nnz_row + n_genes).
///
/// Returns `(projected_indices, projected_data)` where indices are
/// remapped to `0..gene_set.len()` (position in gene_set).
pub fn project_csr_row(
    indices: &[i32],
    data: &[f32],
    gene_set: &[u32],
) -> (Vec<i32>, Vec<f32>) {
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

        let (row_indices, row_data) =
            project_csr_row(&csr.indices[start..end], &csr.data[start..end], &sorted_genes);

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
    let take_indices = UInt32Array::from(gene_indices.to_vec());
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
    reader: &ScxReader,
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use scx_codec::{CodecId, ValueEncoding};
    use scx_format::header::FileHeader;
    use scx_format::writer::ScxWriter;
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
        FileHeader {
            magic: scx_format::MAGIC,
            format_version: 1,
            header_length: 256,
            flags: 0,
            n_obs,
            n_vars,
            nnz,
            n_csr_shards: 0,
            n_csc_shards: 0,
            shard_target_rows: 16384,
            codec_id: 0,
            index_dtype: 0,
            endian: 0,
            reserved_padding: 0,
            root_catalog_offset: 0,
            root_catalog_length: 0,
            full_catalog_offset: 0,
            full_catalog_length: 0,
            manifest_sequence: 1,
            prev_catalog_offset: 0,
            file_checksum: 0,
            front_catalog_offset: 0,
            front_catalog_length: 0,
            reserved: [0u8; 132],
        }
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

    fn write_test_file(
        dir: &tempfile::TempDir,
        n_obs: usize,
        n_vars: usize,
    ) -> std::path::PathBuf {
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
        let full_csr = ScxCsr::new_unchecked(
            (n_obs, n_vars),
            indptr1,
            indices1,
            data1,
        );
        let projected = project_csr(&full_csr, &gene_indices);

        // Method 2: decode_shard_projected
        let (indptr2, indices2, data2) =
            decode_shard_projected(&reader, entry, &gene_indices).unwrap();

        // They must produce the same result
        assert_eq!(projected.indptr, indptr2);
        assert_eq!(projected.indices, indices2);
        assert_eq!(projected.data, data2);
    }

    #[test]
    fn decode_projected_empty_genes() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 5, 10);
        let reader = ScxReader::open(&path).unwrap();

        let shards = reader.catalog().shards_sorted();
        let entry = shards[0];
        let (indptr, indices, data) =
            decode_shard_projected(&reader, entry, &[]).unwrap();

        assert_eq!(indptr.len(), 6); // 5 rows + 1
        assert!(indices.is_empty());
        assert!(data.is_empty());
        // All indptr values should be 0 (no nnz per row)
        assert!(indptr.iter().all(|&v| v == 0));
    }
}
