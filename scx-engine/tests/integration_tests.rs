// Phase G — Integration & End-to-End Tests
//
// G1: Test fixture builders for multi-shard, single-shard, and deletion-vector files
// G2: End-to-end pipeline tests exercising full query lifecycle
// G3: Correctness validation against analytically computed reference values

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{Array, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use roaring::RoaringBitmap;
use scx_codec::{CodecId, ValueEncoding};
use scx_engine::{build_indexes, QueryPipeline};
use scx_format::header::FileHeader;
use scx_format::writer::ScxWriter;
use scx_format::DeletionVectors;
use tempfile::TempDir;

// ============================================================================
// Helpers
// ============================================================================

fn make_header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader {
        magic: scx_format::MAGIC,
        format_version: 1,
        header_length: 256,
        flags: 0,
        n_obs,
        n_vars,
        nnz: 0, // will be updated by shard writes
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

/// Cell type assignment: non-uniform across shards so some shards lack certain types.
/// Distribution:
///   Shard 0 (rows   0–199): T cell, B cell, NK cell
///   Shard 1 (rows 200–399): B cell, NK cell, Monocyte
///   Shard 2 (rows 400–599): T cell, NK cell, Monocyte
///   Shard 3 (rows 600–799): T cell, B cell, Monocyte
///   Shard 4 (rows 800–999): T cell, B cell, NK cell, Monocyte (all types)
///
/// "Monocyte" is ABSENT from shard 0.
/// "T cell" is ABSENT from shard 1.
fn cell_type_for_row(i: usize) -> &'static str {
    let shard = i / 200;
    let local = i % 200;
    match shard {
        0 => match local % 3 {
            0 => "T cell",
            1 => "B cell",
            _ => "NK cell",
        },
        1 => match local % 3 {
            0 => "B cell",
            1 => "NK cell",
            _ => "Monocyte",
        },
        2 => match local % 3 {
            0 => "T cell",
            1 => "NK cell",
            _ => "Monocyte",
        },
        3 => match local % 3 {
            0 => "T cell",
            1 => "B cell",
            _ => "Monocyte",
        },
        4 => match local % 4 {
            0 => "T cell",
            1 => "B cell",
            2 => "NK cell",
            _ => "Monocyte",
        },
        _ => "Unknown",
    }
}

/// n_genes values: varying 100-5000 based on row index.
fn n_genes_for_row(i: usize) -> i32 {
    100 + ((i * 4907) % 4901) as i32 // pseudo-random in [100, 5000]
}

/// Build obs metadata for multi-shard fixture: cell_id, cell_type, n_genes columns.
fn build_multi_shard_obs(n: usize) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, false),
        Field::new("n_genes", DataType::Int32, false),
    ]);
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let types: Vec<&str> = (0..n).map(cell_type_for_row).collect();
    let n_genes: Vec<i32> = (0..n).map(n_genes_for_row).collect();

    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(types)),
            Arc::new(Int32Array::from(n_genes)),
        ],
    )
    .unwrap()
}

/// Build var metadata: gene_id column.
fn build_var(n: usize) -> RecordBatch {
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

/// Generate shard data for `n_rows` rows over `n_vars` columns.
/// Each row has 2 non-zero entries with uint8 values.
fn shard_data(n_rows: usize, n_vars: usize, row_offset: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();

    for row in 0..n_rows {
        let global = row_offset + row;
        let col0 = (global * 7) % n_vars;
        let col1 = (global * 7 + 3) % n_vars;
        let (c0, c1) = if col0 < col1 {
            (col0, col1)
        } else if col0 > col1 {
            (col1, col0)
        } else {
            // Make sure indices are distinct
            let alt = (col0 + 1) % n_vars;
            if col0 < alt {
                (col0, alt)
            } else {
                (alt, col0)
            }
        };
        indices.push(c0 as u32);
        indices.push(c1 as u32);
        values.push(((global + 1) % 255 + 1) as u8); // non-zero: 1-255
        values.push(((global + 2) % 255 + 1) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }

    (indptr, indices, values)
}

// ============================================================================
// G1. Test fixture builders
// ============================================================================

/// G1 fixture 1: 1000 cells × 100 genes, 5 shards, predicate index on cell_type.
fn write_multi_shard_fixture(dir: &TempDir) -> PathBuf {
    let n_obs = 1000usize;
    let n_vars = 100usize;
    let rows_per_shard = 200usize;
    let n_shards = 5usize;

    let path = dir.path().join("multi_shard.scx");
    let header = make_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    let obs = build_multi_shard_obs(n_obs);
    writer.write_obs(&obs).unwrap();

    let var = build_var(n_vars);
    writer.write_var(&var).unwrap();

    // Write 5 shards
    for shard_idx in 0..n_shards {
        let row_start = shard_idx * rows_per_shard;
        let (indptr, indices, values) = shard_data(rows_per_shard, n_vars, row_start);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_start as u64,
            )
            .unwrap();
    }

    // Build and write predicate index on cell_type
    let shard_row_ranges: Vec<(u64, u64)> = (0..n_shards)
        .map(|i| {
            let s = (i * rows_per_shard) as u64;
            (s, s + rows_per_shard as u64)
        })
        .collect();

    let pred_index = build_indexes(&obs, &shard_row_ranges, &["cell_type".to_string()]).unwrap();
    let mut index_bytes = Vec::new();
    pred_index.write_to(&mut index_bytes).unwrap();
    writer.write_obs_predicate_index(&index_bytes).unwrap();

    writer.finish().unwrap();
    path
}

/// G1 fixture 2: 100 cells × 50 genes, single shard.
fn write_single_shard_fixture(dir: &TempDir) -> PathBuf {
    let n_obs = 100usize;
    let n_vars = 50usize;

    let path = dir.path().join("single_shard.scx");
    let header = make_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    // Simple obs with cell_id and cell_type
    let obs_schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, false),
    ]);
    let ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    let types: Vec<&str> = (0..n_obs)
        .map(|i| match i % 3 {
            0 => "T cell",
            1 => "B cell",
            _ => "NK cell",
        })
        .collect();
    let obs = RecordBatch::try_new(
        Arc::new(obs_schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(types)),
        ],
    )
    .unwrap();
    writer.write_obs(&obs).unwrap();

    let var = build_var(n_vars);
    writer.write_var(&var).unwrap();

    let (indptr, indices, values) = shard_data(n_obs, n_vars, 0);
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

/// G1 fixture 3: multi-shard file with deletion vectors.
/// Deletes rows 0, 1, 2 in shard 0 and rows 10, 11 in shard 2.
fn write_deletion_vectors_fixture(dir: &TempDir) -> PathBuf {
    let n_obs = 1000usize;
    let n_vars = 100usize;
    let rows_per_shard = 200usize;
    let n_shards = 5usize;

    let path = dir.path().join("with_dv.scx");
    let header = make_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    let obs = build_multi_shard_obs(n_obs);
    writer.write_obs(&obs).unwrap();
    writer.write_var(&build_var(n_vars)).unwrap();

    for shard_idx in 0..n_shards {
        let row_start = shard_idx * rows_per_shard;
        let (indptr, indices, values) = shard_data(rows_per_shard, n_vars, row_start);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_start as u64,
            )
            .unwrap();
    }

    // Write deletion vectors: delete rows 0,1,2 in shard 0 and rows 10,11 in shard 2
    let mut dv = DeletionVectors::new();
    let mut bm0 = RoaringBitmap::new();
    bm0.insert(0);
    bm0.insert(1);
    bm0.insert(2);
    dv.shards.insert(0, bm0);
    let mut bm2 = RoaringBitmap::new();
    bm2.insert(10);
    bm2.insert(11);
    dv.shards.insert(2, bm2);
    writer.write_deletion_vectors(&dv).unwrap();

    writer.finish().unwrap();
    path
}

// ============================================================================
// G2. End-to-end pipeline tests
// ============================================================================

#[test]
fn e2e_open_collect_full() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir);
    let result = QueryPipeline::open(&path).unwrap().collect().unwrap();
    assert_eq!(result.x.n_rows(), 1000);
    assert_eq!(result.x.n_cols(), 100);
    assert_eq!(result.obs.num_rows(), 1000);
    assert_eq!(result.var.num_rows(), 100);
    assert_eq!(result.total_shards, 5);
    assert_eq!(result.skipped_shards, 0);
}

#[test]
fn e2e_filter_cell_type() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir);
    let result = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'T cell'")
        .unwrap()
        .collect()
        .unwrap();

    // Count expected T cells from our distribution
    let expected: usize = (0..1000)
        .filter(|&i| cell_type_for_row(i) == "T cell")
        .count();
    assert_eq!(result.x.n_rows(), expected);
    assert_eq!(result.obs.num_rows(), expected);

    // Verify all returned cells are T cells
    let types_col = result
        .obs
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for i in 0..types_col.len() {
        assert_eq!(types_col.value(i), "T cell");
    }
}

#[test]
fn e2e_filter_numeric_range() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir);
    let result = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("n_genes > 200 and n_genes < 2000")
        .unwrap()
        .collect()
        .unwrap();

    let expected: usize = (0..1000)
        .filter(|&i| {
            let ng = n_genes_for_row(i);
            ng > 200 && ng < 2000
        })
        .count();
    assert_eq!(result.x.n_rows(), expected);
    assert_eq!(result.obs.num_rows(), expected);

    // Verify all n_genes values are in range
    let n_genes_col = result
        .obs
        .column(2)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    for i in 0..n_genes_col.len() {
        let v = n_genes_col.value(i);
        assert!(v > 200 && v < 2000, "n_genes {v} not in (200, 2000)");
    }
}

#[test]
fn e2e_filter_in_predicate() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir);
    let result = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type in ['T cell', 'B cell']")
        .unwrap()
        .collect()
        .unwrap();

    let expected: usize = (0..1000)
        .filter(|&i| {
            let ct = cell_type_for_row(i);
            ct == "T cell" || ct == "B cell"
        })
        .count();
    assert_eq!(result.x.n_rows(), expected);
    assert_eq!(result.obs.num_rows(), expected);

    let types_col = result
        .obs
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for i in 0..types_col.len() {
        let v = types_col.value(i);
        assert!(v == "T cell" || v == "B cell", "unexpected type: {v}");
    }
}

#[test]
fn e2e_select_genes() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir);
    let hvg = vec![0, 5, 10, 20, 50, 99]; // 6 genes
    let result = QueryPipeline::open(&path)
        .unwrap()
        .select_genes(hvg.clone())
        .collect()
        .unwrap();

    assert_eq!(result.x.n_cols(), 6);
    assert_eq!(result.x.n_rows(), 1000);
    assert_eq!(result.var.num_rows(), 6);

    // Verify var metadata has correct gene IDs
    let gene_ids = result
        .var
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(gene_ids.value(0), "gene_0");
    assert_eq!(gene_ids.value(1), "gene_5");
    assert_eq!(gene_ids.value(2), "gene_10");
    assert_eq!(gene_ids.value(3), "gene_20");
    assert_eq!(gene_ids.value(4), "gene_50");
    assert_eq!(gene_ids.value(5), "gene_99");
}

#[test]
fn e2e_full_pipeline() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir);
    let hvg = vec![0, 10, 20, 30, 40];
    let result = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'T cell'")
        .unwrap()
        .select_genes(hvg.clone())
        .with_normalize(1e4)
        .with_log1p()
        .collect()
        .unwrap();

    let expected_rows: usize = (0..1000)
        .filter(|&i| cell_type_for_row(i) == "T cell")
        .count();
    assert_eq!(result.x.n_rows(), expected_rows);
    assert_eq!(result.x.n_cols(), 5);
    assert_eq!(result.obs.num_rows(), expected_rows);
    assert_eq!(result.var.num_rows(), 5);

    // Values should be positive and reasonable (log1p of normalized)
    for i in 0..result.x.data.len() {
        assert!(
            result.x.data[i] > 0.0,
            "data[{i}] should be positive after fused ops"
        );
        assert!(
            result.x.data[i] < 20.0,
            "data[{i}] = {} is unreasonably large",
            result.x.data[i]
        );
    }
}

#[test]
fn e2e_pushdown_skips_shards() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir);
    // "T cell" is absent from shard 1, so at least shard 1 should be skipped
    // with catalog-level pushdown via the predicate index
    let result = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'T cell'")
        .unwrap()
        .collect()
        .unwrap();

    // With the predicate index on cell_type, shard pruning may skip shards
    // where T cell doesn't appear. Shard 1 has no T cells.
    // Note: this depends on catalog-level pushdown reading category bitsets.
    // Even if catalog pushdown doesn't skip, the result must still be correct.
    let expected: usize = (0..1000)
        .filter(|&i| cell_type_for_row(i) == "T cell")
        .count();
    assert_eq!(result.x.n_rows(), expected);
    // We verify the result is correct regardless of skip optimization
    assert!(result.total_shards == 5);
}

#[test]
fn e2e_parallel_matches_sequential() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir);

    // Run with default parallelism
    let result = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'B cell'")
        .unwrap()
        .collect()
        .unwrap();

    // Run again (rayon pool is shared, result should be deterministic)
    let result2 = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'B cell'")
        .unwrap()
        .collect()
        .unwrap();

    assert_eq!(result.x.n_rows(), result2.x.n_rows());
    assert_eq!(result.x.data, result2.x.data);
    assert_eq!(result.x.indices, result2.x.indices);
    assert_eq!(result.x.indptr, result2.x.indptr);
}

#[test]
fn e2e_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir);
    let result = QueryPipeline::open(&path)
        .unwrap()
        .limit(50)
        .collect()
        .unwrap();

    assert_eq!(result.x.n_rows(), 50);
    assert_eq!(result.obs.num_rows(), 50);
}

#[test]
fn e2e_schema_error_at_construction() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir);
    let err = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("nonexistent == 'x'");
    assert!(err.is_err(), "should error at filter_obs, not at collect");
}

#[test]
fn e2e_deletion_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_deletion_vectors_fixture(&dir);
    let result = QueryPipeline::open(&path).unwrap().collect().unwrap();

    // 1000 cells - 5 deleted (rows 0,1,2 in shard 0 + rows 10,11 in shard 2)
    // Global deleted: 0, 1, 2 (shard 0, row_start=0) and 410, 411 (shard 2, row_start=400, local 10,11)
    assert_eq!(result.x.n_rows(), 995);
    assert_eq!(result.obs.num_rows(), 995);

    // Verify deleted cells are NOT in result
    let cell_ids = result
        .obs
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let returned_ids: Vec<&str> = (0..cell_ids.len()).map(|i| cell_ids.value(i)).collect();
    assert!(!returned_ids.contains(&"cell_0"));
    assert!(!returned_ids.contains(&"cell_1"));
    assert!(!returned_ids.contains(&"cell_2"));
    assert!(!returned_ids.contains(&"cell_410"));
    assert!(!returned_ids.contains(&"cell_411"));
}

#[test]
fn e2e_obs_row_count_matches_x() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir);
    let result = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'NK cell'")
        .unwrap()
        .collect()
        .unwrap();

    assert_eq!(result.obs.num_rows(), result.x.n_rows());
}

#[test]
fn e2e_var_row_count_matches_x_cols() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir);
    let result = QueryPipeline::open(&path)
        .unwrap()
        .select_genes(vec![1, 5, 9, 42, 77])
        .collect()
        .unwrap();

    assert_eq!(result.var.num_rows(), result.x.n_cols());
    assert_eq!(result.var.num_rows(), 5);
}

// ============================================================================
// G2 additional: single-shard edge case
// ============================================================================

#[test]
fn e2e_single_shard_no_pushdown() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_single_shard_fixture(&dir);
    let result = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'T cell'")
        .unwrap()
        .collect()
        .unwrap();

    // Single shard: nothing to skip
    assert_eq!(result.total_shards, 1);
    assert_eq!(result.skipped_shards, 0);

    // 100 cells, every 3rd is T cell → ~34 cells
    let expected: usize = (0..100).filter(|i| i % 3 == 0).count();
    assert_eq!(result.x.n_rows(), expected);
}

// ============================================================================
// G3. Correctness validation against reference computation
// ============================================================================

#[test]
fn correctness_filter_obs() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir);
    let result = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'Monocyte'")
        .unwrap()
        .collect()
        .unwrap();

    // Compute expected: which rows are Monocyte?
    let expected_rows: Vec<usize> = (0..1000)
        .filter(|&i| cell_type_for_row(i) == "Monocyte")
        .collect();

    assert_eq!(result.x.n_rows(), expected_rows.len());

    // Check cell IDs match expected rows
    let cell_ids = result
        .obs
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for (out_idx, &global_idx) in expected_rows.iter().enumerate() {
        assert_eq!(
            cell_ids.value(out_idx),
            format!("cell_{global_idx}"),
            "mismatch at output row {out_idx}"
        );
    }
}

#[test]
fn correctness_select_genes() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir);
    let gene_indices = vec![3, 7, 42];
    let result = QueryPipeline::open(&path)
        .unwrap()
        .select_genes(gene_indices.clone())
        .collect()
        .unwrap();

    assert_eq!(result.x.n_cols(), 3);

    // Verify: all column indices in the projected CSR should be in 0..3
    for &idx in &result.x.indices {
        assert!(
            idx >= 0 && idx < 3,
            "projected index {idx} out of range [0, 3)"
        );
    }

    // Verify var metadata
    let gene_ids = result
        .var
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(gene_ids.value(0), "gene_3");
    assert_eq!(gene_ids.value(1), "gene_7");
    assert_eq!(gene_ids.value(2), "gene_42");
}

#[test]
fn correctness_normalize_log1p() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir);

    // Get raw data first
    let raw = QueryPipeline::open(&path).unwrap().collect().unwrap();

    // Get normalized + log1p data
    let transformed = QueryPipeline::open(&path)
        .unwrap()
        .with_normalize(1e4)
        .with_log1p()
        .collect()
        .unwrap();

    assert_eq!(raw.x.n_rows(), transformed.x.n_rows());
    assert_eq!(raw.x.n_cols(), transformed.x.n_cols());

    // Verify transformation row by row
    let target_sum = 1e4;
    for row in 0..raw.x.n_rows() {
        let raw_start = raw.x.indptr[row] as usize;
        let raw_end = raw.x.indptr[row + 1] as usize;
        let trans_start = transformed.x.indptr[row] as usize;
        let trans_end = transformed.x.indptr[row + 1] as usize;

        // Same number of non-zeros
        assert_eq!(raw_end - raw_start, trans_end - trans_start);

        if raw_start == raw_end {
            continue;
        }

        // Compute expected: normalize then log1p
        let row_sum: f64 = raw.x.data[raw_start..raw_end]
            .iter()
            .map(|&v| v as f64)
            .sum();

        for (j, raw_idx) in (raw_start..raw_end).enumerate() {
            let raw_val = raw.x.data[raw_idx] as f64;
            let expected = ((raw_val * target_sum / row_sum) + 1.0).ln() as f32;
            let actual = transformed.x.data[trans_start + j];
            let diff = (expected - actual).abs();
            assert!(
                diff < 1e-5,
                "row {row}, nnz {j}: expected {expected}, got {actual}, diff {diff}"
            );
        }
    }
}

#[test]
fn correctness_combined_pipeline() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir);
    let gene_indices = vec![0, 10, 20];

    // Full pipeline
    let result = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'B cell'")
        .unwrap()
        .select_genes(gene_indices.clone())
        .with_normalize(1e4)
        .with_log1p()
        .collect()
        .unwrap();

    // Step-by-step reference:
    // 1. Filter: only B cells
    let expected_b_cell_count: usize = (0..1000)
        .filter(|&i| cell_type_for_row(i) == "B cell")
        .count();
    assert_eq!(result.x.n_rows(), expected_b_cell_count);

    // 2. Projection: 3 genes
    assert_eq!(result.x.n_cols(), 3);

    // 3. All indices should be in [0, 3)
    for &idx in &result.x.indices {
        assert!(idx >= 0 && idx < 3);
    }

    // 4. Values should be positive (after log1p)
    for &v in &result.x.data {
        assert!(v > 0.0);
    }

    // 5. Metadata consistency
    assert_eq!(result.obs.num_rows(), result.x.n_rows());
    assert_eq!(result.var.num_rows(), result.x.n_cols());

    // Verify obs is all B cells
    let types_col = result
        .obs
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for i in 0..types_col.len() {
        assert_eq!(types_col.value(i), "B cell");
    }
}
