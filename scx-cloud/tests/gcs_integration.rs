//! GCS Integration Tests for scx-cloud.
//!
//! These tests require actual GCS access and are marked `#[ignore]` by default.
//! Run them with:
//!
//!   cargo test -p scx-cloud --test gcs_integration -- --ignored
//!
//! Prerequisites:
//!   - `gcloud auth application-default login`
//!   - Write access to `gs://arc-ctc-nextflow/scx-test/`
//!
//! The `setup_gcs_test_data` test should be run first to upload test data.
//! Other tests read/write to unique per-run paths to avoid conflicts.

use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_cloud::{pull, pull_filtered, push, PullOptions, PushOptions};
use scx_codec::{CodecId, ValueEncoding};
use scx_format::header::{FileHeader, MAGIC};
use scx_format::reader::ScxReader;
use scx_format::writer::ScxWriter;

const GCS_TEST_PREFIX: &str = "gs://arc-ctc-nextflow/scx-test";

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

fn create_test_scx(path: &Path, n_obs: usize, n_vars: usize) {
    let header = test_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(path, header).unwrap();

    // obs with cell_id and cell_type
    let ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    let types: Vec<String> = (0..n_obs)
        .map(|i| match i % 3 {
            0 => "T_cell".to_string(),
            1 => "B_cell".to_string(),
            _ => "Monocyte".to_string(),
        })
        .collect();
    let obs_schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, false),
    ]);
    let obs = RecordBatch::try_new(
        Arc::new(obs_schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                types.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    writer.write_obs(&obs).unwrap();

    let gene_ids: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    let var_schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    let var = RecordBatch::try_new(
        Arc::new(var_schema),
        vec![Arc::new(StringArray::from(
            gene_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    writer.write_var(&var).unwrap();

    let rows_per_shard = 500;
    let mut offset = 0;
    while offset < n_obs {
        let chunk = rows_per_shard.min(n_obs - offset);
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for r in 0..chunk {
            let c0 = (r * 2) % n_vars;
            let c1 = (r * 2 + 1) % n_vars;
            indices.push(c0 as u32);
            indices.push(c1 as u32);
            values.push(((r + 1) % 256) as u8);
            values.push(((r + 2) % 256) as u8);
            indptr.push(indptr.last().unwrap() + 2);
        }
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                offset as u64,
            )
            .unwrap();
        offset += chunk;
    }
    writer.finish().unwrap();
}

/// Generate a unique GCS path for this test run to avoid conflicts.
fn gcs_test_path(suffix: &str) -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("{GCS_TEST_PREFIX}/ci-{ts}-{suffix}")
}

// ============================================================
// Setup: upload test data to GCS
// ============================================================

/// Upload synthetic test data to GCS.
/// Run this first: `cargo test ... -- --ignored setup_gcs_test_data`
#[tokio::test]
#[ignore]
async fn setup_gcs_test_data() {
    let dir = tempfile::tempdir().unwrap();

    // Create pbmc3k-like test file (3000 obs, 500 vars)
    let pbmc_path = dir.path().join("pbmc3k.scx");
    create_test_scx(&pbmc_path, 3000, 500);

    // Push to GCS
    let gcs_dest = format!("{GCS_TEST_PREFIX}/pbmc3k.scxd/");
    let stats = push(&pbmc_path, &gcs_dest, PushOptions::default())
        .await
        .unwrap();

    eprintln!(
        "Uploaded pbmc3k: {} objects, {:.1} MB",
        stats.sections_uploaded,
        stats.bytes_uploaded as f64 / 1_000_000.0
    );

    // Verify by pulling back
    let pulled = dir.path().join("pbmc3k_verify.scx");
    pull(&gcs_dest, &pulled, PullOptions::default())
        .await
        .unwrap();
    let reader = ScxReader::open(&pulled).unwrap();
    assert_eq!(reader.n_obs(), 3000);
    assert_eq!(reader.n_vars(), 500);
    eprintln!(
        "Verified pbmc3k pull: {} obs, {} vars",
        reader.n_obs(),
        reader.n_vars()
    );
}

// ============================================================
// Test: pull from GCS
// ============================================================

#[tokio::test]
#[ignore]
async fn test_gcs_pull_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let gcs_source = format!("{GCS_TEST_PREFIX}/pbmc3k.scxd/");
    let output = dir.path().join("pulled.scx");

    let stats = pull(&gcs_source, &output, PullOptions::default())
        .await
        .unwrap();

    assert!(stats.bytes_downloaded > 0);
    assert!(stats.sections_downloaded > 0);

    let reader = ScxReader::open(&output).unwrap();
    assert_eq!(reader.n_obs(), 3000);
    assert_eq!(reader.n_vars(), 500);

    // Read obs, var, and CSR data
    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 3000);
    let var = reader.read_var().unwrap();
    assert_eq!(var.num_rows(), 500);
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 3000);
}

// ============================================================
// Test: push to GCS
// ============================================================

#[tokio::test]
#[ignore]
async fn test_gcs_push_succeeds() {
    let dir = tempfile::tempdir().unwrap();

    // Create local test file
    let input = dir.path().join("test_push.scx");
    create_test_scx(&input, 1000, 200);

    // Push to a unique GCS path
    let gcs_dest = gcs_test_path("push.scxd/");
    let stats = push(&input, &gcs_dest, PushOptions::default())
        .await
        .unwrap();

    assert!(stats.bytes_uploaded > 0);
    assert!(stats.sections_uploaded > 0);

    // Verify by pulling back
    let pulled = dir.path().join("pulled_back.scx");
    pull(&gcs_dest, &pulled, PullOptions::default())
        .await
        .unwrap();

    let reader_orig = ScxReader::open(&input).unwrap();
    let reader_pulled = ScxReader::open(&pulled).unwrap();
    assert_eq!(reader_orig.n_obs(), reader_pulled.n_obs());
    assert_eq!(reader_orig.n_vars(), reader_pulled.n_vars());

    let csr_orig = reader_orig.read_all_csr_shards().unwrap();
    let csr_pulled = reader_pulled.read_all_csr_shards().unwrap();
    assert_eq!(csr_orig.indptr, csr_pulled.indptr);
    assert_eq!(csr_orig.indices, csr_pulled.indices);
    assert_eq!(csr_orig.data, csr_pulled.data);
}

// ============================================================
// Test: pull → validate data integrity
// ============================================================

#[tokio::test]
#[ignore]
async fn test_gcs_pull_validate_integrity() {
    let dir = tempfile::tempdir().unwrap();
    let gcs_source = format!("{GCS_TEST_PREFIX}/pbmc3k.scxd/");
    let output = dir.path().join("pulled_validate.scx");

    pull(&gcs_source, &output, PullOptions::default())
        .await
        .unwrap();

    let reader = ScxReader::open(&output).unwrap();

    // Validate checksums
    let results = reader.validate().unwrap();
    for (name, passed) in &results {
        assert!(passed, "checksum validation failed for section: {name}");
    }
}

// ============================================================
// Test: selective pull from GCS with predicate
// ============================================================

#[tokio::test]
#[ignore]
async fn test_gcs_selective_pull_with_predicate() {
    let dir = tempfile::tempdir().unwrap();
    let gcs_source = format!("{GCS_TEST_PREFIX}/pbmc3k.scxd/");
    let output = dir.path().join("filtered.scx");

    // Filter for T_cell only (1/3 of 3000 = 1000 cells)
    let stats = pull_filtered(
        &gcs_source,
        &output,
        "cell_type == 'T_cell'",
        PullOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(stats.matching_cells, 1000);
    assert!(
        stats.skipped_shards > 0
            || stats.downloaded_shards < stats.total_shards
            || stats.downloaded_shards == stats.total_shards
    );

    let reader = ScxReader::open(&output).unwrap();
    assert_eq!(reader.n_obs(), 1000);

    // Verify all cells are T_cell
    let obs = reader.read_obs().unwrap();
    let cell_type_col = obs
        .column_by_name("cell_type")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for i in 0..arrow::array::Array::len(cell_type_col) {
        assert_eq!(cell_type_col.value(i), "T_cell");
    }
}

// ============================================================
// Test: scx info on pulled file shows correct stats
// ============================================================

#[tokio::test]
#[ignore]
async fn test_gcs_pulled_file_info_correct() {
    let dir = tempfile::tempdir().unwrap();
    let gcs_source = format!("{GCS_TEST_PREFIX}/pbmc3k.scxd/");
    let output = dir.path().join("pulled_info.scx");

    pull(&gcs_source, &output, PullOptions::default())
        .await
        .unwrap();

    let reader = ScxReader::open(&output).unwrap();

    // Basic info checks
    assert_eq!(reader.n_obs(), 3000);
    assert_eq!(reader.n_vars(), 500);

    // Verify header is valid
    let data = std::fs::read(&output).unwrap();
    let hdr =
        FileHeader::read_from(&mut Cursor::new(&data[..scx_format::header::HEADER_SIZE])).unwrap();
    assert_eq!(hdr.magic, MAGIC);
    assert_eq!(hdr.n_obs, 3000);
    assert_eq!(hdr.n_vars, 500);
    assert!(hdr.full_catalog_offset > 0);
    assert!(hdr.full_catalog_length > 0);
}

// ============================================================
// Test: push → pull round-trip preserves data
// ============================================================

#[tokio::test]
#[ignore]
async fn test_gcs_push_pull_roundtrip_data_integrity() {
    let dir = tempfile::tempdir().unwrap();

    // Create test file with known data
    let input = dir.path().join("roundtrip_src.scx");
    create_test_scx(&input, 2000, 300);

    // Push to GCS
    let gcs_dest = gcs_test_path("roundtrip.scxd/");
    push(&input, &gcs_dest, PushOptions::default())
        .await
        .unwrap();

    // Pull back
    let pulled = dir.path().join("roundtrip_pulled.scx");
    pull(&gcs_dest, &pulled, PullOptions::default())
        .await
        .unwrap();

    // Validate full data roundtrip
    let reader_orig = ScxReader::open(&input).unwrap();
    let reader_rt = ScxReader::open(&pulled).unwrap();

    assert_eq!(reader_orig.n_obs(), reader_rt.n_obs());
    assert_eq!(reader_orig.n_vars(), reader_rt.n_vars());

    // CSR data must be identical
    let csr_orig = reader_orig.read_all_csr_shards().unwrap();
    let csr_rt = reader_rt.read_all_csr_shards().unwrap();
    assert_eq!(csr_orig.indptr, csr_rt.indptr);
    assert_eq!(csr_orig.indices, csr_rt.indices);
    assert_eq!(csr_orig.data, csr_rt.data);

    // obs metadata must be identical
    let obs_orig = reader_orig.read_obs().unwrap();
    let obs_rt = reader_rt.read_obs().unwrap();
    assert_eq!(obs_orig.num_rows(), obs_rt.num_rows());
    assert_eq!(obs_orig.num_columns(), obs_rt.num_columns());
}
