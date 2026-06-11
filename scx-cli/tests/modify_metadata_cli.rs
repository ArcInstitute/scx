//! Integration tests for `scx set-uns` and `scx modify-metadata`, driven
//! through the real CLI binary.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::{FileHeader, MAGIC};
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;

fn header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader {
        magic: MAGIC,
        format_version: scx_format_io::CURRENT_FORMAT_VERSION,
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

fn obs_batch(n: usize, donor: &str) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let donors: Vec<String> = std::iter::repeat_n(donor.to_string(), n).collect();
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("donor", DataType::Utf8, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(donors)),
        ],
    )
    .unwrap()
}

fn var_batch(n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("g{i}")).collect();
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(StringArray::from(ids))]).unwrap()
}

fn write_fixture(dir: &tempfile::TempDir, name: &str, n_obs: usize, n_vars: usize) -> PathBuf {
    let path = dir.path().join(name);
    let mut writer = ScxWriter::new(&path, header(n_obs as u64, n_vars as u64)).unwrap();
    writer.write_obs(&obs_batch(n_obs, "donor_A")).unwrap();
    writer.write_var(&var_batch(n_vars)).unwrap();
    let indptr: Vec<u64> = vec![0u64; n_obs + 1];
    writer
        .write_csr_shard(&indptr, &[], &[], CodecId::None, ValueEncoding::Uint8, 0)
        .unwrap();
    writer
        .write_uns(&serde_json::json!({"state": "v0"}))
        .unwrap();
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1_710_000_000,
            action: "create".to_string(),
            tool: "test".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();
    writer.finish().unwrap();
    path
}

fn write_parquet(path: &Path, batch: &RecordBatch) {
    use parquet::arrow::ArrowWriter;
    let file = std::fs::File::create(path).unwrap();
    let mut w = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    w.write(batch).unwrap();
    w.close().unwrap();
}

fn scx_cli() -> std::process::Command {
    std::process::Command::new(env!("CARGO_BIN_EXE_scx"))
}

#[test]
fn set_uns_cli_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let scx = write_fixture(&dir, "f.scx", 20, 5);
    let uns_json = dir.path().join("uns.json");
    std::fs::write(&uns_json, r#"{"method": "cli", "k": 3}"#).unwrap();

    let out = scx_cli()
        .args([
            "set-uns",
            scx.to_str().unwrap(),
            "--uns",
            uns_json.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "set-uns failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let uns = ScxReader::open(&scx).unwrap().read_uns().unwrap();
    assert_eq!(uns["method"], "cli");
    assert_eq!(uns["k"], 3);
}

#[test]
fn modify_metadata_cli_obs_from_parquet() {
    let dir = tempfile::tempdir().unwrap();
    let scx = write_fixture(&dir, "f.scx", 20, 5);
    let pq = dir.path().join("obs.parquet");
    write_parquet(&pq, &obs_batch(20, "donor_Z"));

    let out = scx_cli()
        .args([
            "modify-metadata",
            scx.to_str().unwrap(),
            "--obs",
            pq.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "modify-metadata failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let obs = ScxReader::open(&scx).unwrap().read_obs().unwrap();
    assert_eq!(obs.num_rows(), 20);
    let donor = obs
        .column_by_name("donor")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(donor.value(0), "donor_Z");
}

#[test]
fn modify_metadata_cli_wrong_shape_fails() {
    let dir = tempfile::tempdir().unwrap();
    let scx = write_fixture(&dir, "f.scx", 20, 5);
    let pq = dir.path().join("obs_bad.parquet");
    write_parquet(&pq, &obs_batch(19, "donor_Z")); // one row short

    let out = scx_cli()
        .args([
            "modify-metadata",
            scx.to_str().unwrap(),
            "--obs",
            pq.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "expected failure on wrong obs row count"
    );
}
