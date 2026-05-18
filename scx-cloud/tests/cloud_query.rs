//! End-to-end Phase 7 cloud-query tests.
//!
//! Verify that `QueryPipeline::from_reader(CloudSectionReader)` returns
//! the same `QueryResult` as the local `QueryPipeline::open(path)` for
//! each of the three on-disk layouts that `scx_cloud::open_cloud`
//! resolves:
//!   - exploded `.scxd/` directory (one object per section)
//!   - cloud-optimized packed `.scx` (front catalog)
//!   - non-cloud-optimized packed `.scx` (EOF catalog)
//!
//! Uses `object_store::LocalFileSystem` (resolved automatically when
//! `scx_cloud::open_cloud` is handed a filesystem path) so the tests
//! run without any cloud credentials.

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_cloud::{cloud_optimize, explode, CloudSectionReader};
use scx_codec::{CodecId, ValueEncoding};
use scx_engine::QueryPipeline;
use scx_format::header::{FileHeader, MAGIC};
use scx_format::writer::ScxWriter;

fn test_header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs,
        n_vars,
        nnz: 0,
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
        n_modalities: 0,
        modality_table_offset: 0,
        modality_table_length: 0,
        reserved: [0u8; 112],
    }
}

fn sample_obs(n: usize) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, false),
    ]);
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let types: Vec<&str> = (0..n)
        .map(|i| match i % 3 {
            0 => "T_cell",
            1 => "B_cell",
            _ => "NK_cell",
        })
        .collect();
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(types)),
        ],
    )
    .unwrap()
}

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

fn sample_shard_data(n_rows: usize, n_vars: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..n_rows {
        let col0 = (row * 2) % n_vars;
        let col1 = (row * 2 + 1) % n_vars;
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
    (indptr, indices, values)
}

/// Write a multi-shard test SCX file in the given tempdir. Returns its path.
fn write_test_scx(dir: &tempfile::TempDir, n_obs: usize, n_vars: usize) -> PathBuf {
    let path = dir.path().join("test.scx");
    let header = test_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    let rows_per_shard = 50;
    let mut row_offset = 0;
    while row_offset < n_obs {
        let shard_rows = std::cmp::min(rows_per_shard, n_obs - row_offset);
        let (indptr, indices, values) = sample_shard_data(shard_rows, n_vars);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_offset as u64,
            )
            .unwrap();
        row_offset += shard_rows;
    }
    writer.finish().unwrap();
    path
}

fn build_cloud_pipeline(rt: &Arc<tokio::runtime::Runtime>, url: &str) -> QueryPipeline {
    let reader = rt.block_on(scx_cloud::open_cloud(url)).unwrap();
    let adapter = CloudSectionReader::new(Arc::new(reader), Arc::clone(rt));
    QueryPipeline::from_reader(Box::new(adapter)).unwrap()
}

fn assert_results_match(local: &scx_engine::QueryResult, cloud: &scx_engine::QueryResult) {
    assert_eq!(local.x.n_rows(), cloud.x.n_rows(), "x.n_rows mismatch");
    assert_eq!(local.x.n_cols(), cloud.x.n_cols(), "x.n_cols mismatch");
    assert_eq!(local.x.indptr, cloud.x.indptr, "indptr mismatch");
    assert_eq!(local.x.indices, cloud.x.indices, "indices mismatch");
    assert_eq!(local.x.data, cloud.x.data, "data mismatch");
    assert_eq!(
        local.obs.num_rows(),
        cloud.obs.num_rows(),
        "obs n_rows mismatch"
    );
    assert_eq!(
        local.var.num_rows(),
        cloud.var.num_rows(),
        "var n_rows mismatch"
    );
}

/// Exploded `.scxd/` selective query returns the same result as the
/// local in-process query.
#[test]
fn cloud_query_exploded_matches_local_filter_obs() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = write_test_scx(&dir, 100, 50);

    let exploded_path = dir.path().join("test.scxd");
    explode(&scx_path, &exploded_path).unwrap();

    let local = QueryPipeline::open(&scx_path)
        .unwrap()
        .filter_obs("cell_type == 'T_cell'")
        .unwrap()
        .collect()
        .unwrap();

    let rt = Arc::new(tokio::runtime::Runtime::new().unwrap());
    let cloud = build_cloud_pipeline(&rt, &exploded_path.to_string_lossy())
        .filter_obs("cell_type == 'T_cell'")
        .unwrap()
        .collect()
        .unwrap();

    assert_results_match(&local, &cloud);
    assert!(local.x.n_rows() > 0, "filter should keep at least one cell");
}

/// Cloud-optimized packed `.scx` (front catalog) returns the same result
/// as the local in-process query for the same predicate.
#[test]
fn cloud_query_cloud_optimized_packed_matches_local() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = write_test_scx(&dir, 100, 50);

    let optimized_path = dir.path().join("optimized.scx");
    cloud_optimize(&scx_path, &optimized_path).unwrap();

    let local = QueryPipeline::open(&scx_path)
        .unwrap()
        .filter_obs("cell_type == 'B_cell'")
        .unwrap()
        .collect()
        .unwrap();

    let rt = Arc::new(tokio::runtime::Runtime::new().unwrap());
    let cloud = build_cloud_pipeline(&rt, &optimized_path.to_string_lossy())
        .filter_obs("cell_type == 'B_cell'")
        .unwrap()
        .collect()
        .unwrap();

    assert_results_match(&local, &cloud);
}

/// Non-cloud-optimized packed `.scx` (EOF catalog) returns the same
/// result. Slower in practice (extra HEAD + EOF range read) but
/// functionally identical.
#[test]
fn cloud_query_plain_packed_matches_local() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = write_test_scx(&dir, 100, 50);

    let local = QueryPipeline::open(&scx_path)
        .unwrap()
        .filter_obs("cell_type == 'NK_cell'")
        .unwrap()
        .collect()
        .unwrap();

    let rt = Arc::new(tokio::runtime::Runtime::new().unwrap());
    let cloud = build_cloud_pipeline(&rt, &scx_path.to_string_lossy())
        .filter_obs("cell_type == 'NK_cell'")
        .unwrap()
        .collect()
        .unwrap();

    assert_results_match(&local, &cloud);
}

/// Gene projection via `select_genes` produces the same column subset
/// in both code paths.
#[test]
fn cloud_query_select_genes_matches_local() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = write_test_scx(&dir, 80, 30);
    let exploded_path = dir.path().join("test.scxd");
    explode(&scx_path, &exploded_path).unwrap();

    let gene_indices: Vec<u32> = vec![0, 5, 7, 12, 20];

    let local = QueryPipeline::open(&scx_path)
        .unwrap()
        .filter_obs("cell_type == 'T_cell'")
        .unwrap()
        .select_genes(gene_indices.clone())
        .collect()
        .unwrap();

    let rt = Arc::new(tokio::runtime::Runtime::new().unwrap());
    let cloud = build_cloud_pipeline(&rt, &exploded_path.to_string_lossy())
        .filter_obs("cell_type == 'T_cell'")
        .unwrap()
        .select_genes(gene_indices)
        .collect()
        .unwrap();

    assert_results_match(&local, &cloud);
    assert_eq!(cloud.x.n_cols(), 5);
}

/// `total_shards` and `skipped_shards` accounting works on the cloud
/// path. Predicate indexes are absent on these test fixtures, so
/// skipped_shards is 0 — the assertion is that the COUNT matches
/// (i.e. shard pushdown gives the same answer regardless of backend).
#[test]
fn cloud_query_reports_total_shards() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = write_test_scx(&dir, 200, 20);
    let exploded_path = dir.path().join("test.scxd");
    explode(&scx_path, &exploded_path).unwrap();

    let rt = Arc::new(tokio::runtime::Runtime::new().unwrap());
    let cloud = build_cloud_pipeline(&rt, &exploded_path.to_string_lossy())
        .filter_obs("cell_type == 'T_cell'")
        .unwrap()
        .collect()
        .unwrap();

    let local = QueryPipeline::open(&scx_path)
        .unwrap()
        .filter_obs("cell_type == 'T_cell'")
        .unwrap()
        .collect()
        .unwrap();

    assert_eq!(cloud.total_shards, local.total_shards);
    assert_eq!(cloud.skipped_shards, local.skipped_shards);
    assert_eq!(cloud.total_shards, 4); // 200 cells / 50 per shard
}
