//! Integration tests for scx-cli info and validate commands.

use std::sync::Arc;

use arrow::array::{Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::shard::SHARD_HEADER_SIZE;
use scx_format_io::writer::ScxWriter;

// ---------------------------------------------------------------------------
// Test helpers (mirrored from scx-format reader tests)
// ---------------------------------------------------------------------------

fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, n_vars, nnz, 16384, 0, 0)
}

fn sample_obs(n: usize) -> arrow::array::RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
    arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn sample_var(n: usize) -> arrow::array::RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    arrow::array::RecordBatch::try_new(
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
        indices.push(col0 as u32);
        indices.push(col1 as u32);
        values.push(((row + 1) % 256) as u8);
        values.push(((row + 2) % 256) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    (indptr, indices, values)
}

fn write_test_file(
    dir: &tempfile::TempDir,
    filename: &str,
    n_obs: usize,
    n_vars: usize,
    n_shards: usize,
    include_extras: bool,
) -> std::path::PathBuf {
    let path = dir.path().join(filename);
    let total_nnz = n_obs * 2;
    let header = sample_header(n_obs as u64, n_vars as u64, total_nnz as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    let rows_per_shard = n_obs / n_shards;
    for s in 0..n_shards {
        let shard_rows = if s == n_shards - 1 {
            n_obs - rows_per_shard * s
        } else {
            rows_per_shard
        };
        let (indptr, indices, values) = sample_shard_data(shard_rows, n_vars);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                (s * rows_per_shard) as u64,
            )
            .unwrap();
    }

    if include_extras {
        let obsm_schema = Schema::new(vec![
            Field::new("pc1", DataType::Float32, false),
            Field::new("pc2", DataType::Float32, false),
        ]);
        let obsm_batch = arrow::array::RecordBatch::try_new(
            Arc::new(obsm_schema),
            vec![
                Arc::new(Float32Array::from(
                    (0..n_obs).map(|i| i as f32).collect::<Vec<_>>(),
                )),
                Arc::new(Float32Array::from(
                    (0..n_obs).map(|i| (i as f32) * 2.0).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        writer.write_obsm("X_pca", &obsm_batch).unwrap();

        writer
            .write_uns(&serde_json::json!({"species": "human", "version": 2}))
            .unwrap();

        writer
            .write_provenance(vec![ProvenanceEntry {
                timestamp: 1710000000,
                action: "convert".to_string(),
                tool: "scx-cli 0.1.0".to_string(),
                params_json: "{}".to_string(),
                input_checksums: vec![],
            }])
            .unwrap();
    }

    writer.finish().unwrap();
    path
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// 14.7: `scx info` runs without error on a valid SCX file.
#[test]
fn test_info_runs_on_valid_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "info_test.scx", 6, 10, 2, true);

    let reader = scx_format_io::reader::ScxReader::open(&path).unwrap();
    let header = reader.header();

    // Basic assertions on what info would display
    assert_eq!(header.n_obs, 6);
    assert_eq!(header.n_vars, 10);
    assert_eq!(header.nnz, 12);
    assert_eq!(header.format_version, scx_format_io::CURRENT_FORMAT_VERSION);
    assert_eq!(header.n_csr_shards, 2);

    let catalog = reader.catalog();
    assert!(!catalog.entries.is_empty());

    // Verify file size is readable
    let meta = std::fs::metadata(&path).unwrap();
    assert!(meta.len() > 0);
}

/// 14.6: `scx validate` passes on a clean file.
#[test]
fn test_validate_passes_clean_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "valid.scx", 6, 10, 2, true);

    let reader = scx_format_io::reader::ScxReader::open(&path).unwrap();
    let catalog = reader.catalog();

    // Manually validate all checksums (same logic as validate command)
    for entry in &catalog.entries {
        let bytes = reader.section_bytes(entry).unwrap();
        let computed = scx_format_io::blake3_hash(bytes);
        assert_eq!(
            computed, entry.checksum,
            "checksum mismatch for section '{}'",
            entry.name
        );
    }
}

/// 14.6: `scx validate` detects corruption.
#[test]
fn test_validate_detects_corruption() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "corrupt.scx", 6, 10, 2, false);

    // Corrupt a byte in a CSR shard
    let reader = scx_format_io::reader::ScxReader::open(&path).unwrap();
    let shards = reader.catalog().shards_sorted();
    let shard_offset = shards[0].offset as usize;
    let corrupt_pos = shard_offset + SHARD_HEADER_SIZE + 1;
    drop(reader);

    let mut data = std::fs::read(&path).unwrap();
    data[corrupt_pos] ^= 0xFF;
    std::fs::write(&path, &data).unwrap();

    // Re-open and check: at least one section should fail
    let reader = scx_format_io::reader::ScxReader::open(&path).unwrap();
    let catalog = reader.catalog();

    let mut any_failed = false;
    for entry in &catalog.entries {
        let bytes = reader.section_bytes(entry).unwrap();
        let computed = scx_format_io::blake3_hash(bytes);
        if computed != entry.checksum {
            any_failed = true;
        }
    }
    assert!(any_failed, "corruption should have been detected");
}

/// Test that info works on a minimal file without extras.
#[test]
fn test_info_minimal_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "minimal.scx", 4, 8, 1, false);

    let reader = scx_format_io::reader::ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), 4);
    assert_eq!(reader.n_vars(), 8);
    assert_eq!(reader.header().n_csr_shards, 1);
}

/// Test the binary runs and shows help.
#[test]
fn test_cli_help() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_scx"))
        .arg("--help")
        .output()
        .expect("failed to run scx-cli");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("SCX file format tool"));
    assert!(stdout.contains("convert"));
    assert!(stdout.contains("info"));
    assert!(stdout.contains("validate"));
    assert!(stdout.contains("append"));
    assert!(stdout.contains("delete"));
    assert!(stdout.contains("compact"));
    assert!(stdout.contains("rollback"));
    assert!(stdout.contains("merge"));
    assert!(stdout.contains("query"));
    assert!(stdout.contains("benchmark"));
}

/// Test the info subcommand runs end-to-end via the binary.
#[test]
fn test_cli_info_subcommand() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "cli_info.scx", 6, 10, 2, true);

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_scx"))
        .args(["info", path.to_str().unwrap()])
        .output()
        .expect("failed to run scx-cli info");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains(&format!("SCX v{}", scx_format_io::CURRENT_FORMAT_VERSION)));
    assert!(stdout.contains("6 cells"));
    assert!(stdout.contains("10 genes"));
}

/// Test the validate subcommand runs end-to-end via the binary.
#[test]
fn test_cli_validate_subcommand() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "cli_valid.scx", 6, 10, 2, true);

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_scx"))
        .args(["validate", path.to_str().unwrap()])
        .output()
        .expect("failed to run scx-cli validate");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("All"));
    assert!(stdout.contains("passed"));
}

/// Test validate subcommand exits with code 1 on corruption.
#[test]
fn test_cli_validate_fails_on_corruption() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "cli_corrupt.scx", 6, 10, 2, false);

    // Corrupt
    let reader = scx_format_io::reader::ScxReader::open(&path).unwrap();
    let shards = reader.catalog().shards_sorted();
    let corrupt_pos = shards[0].offset as usize + SHARD_HEADER_SIZE + 1;
    drop(reader);

    let mut data = std::fs::read(&path).unwrap();
    data[corrupt_pos] ^= 0xFF;
    std::fs::write(&path, &data).unwrap();

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_scx"))
        .args(["validate", path.to_str().unwrap()])
        .output()
        .expect("failed to run scx-cli validate");
    assert!(
        !output.status.success(),
        "should exit with non-zero on corruption"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("FAIL"));
}

/// Test validate --verbose prints checksums.
#[test]
fn test_cli_validate_verbose() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "cli_verbose.scx", 4, 8, 1, false);

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_scx"))
        .args(["validate", "--verbose", path.to_str().unwrap()])
        .output()
        .expect("failed to run scx-cli validate");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("expected:"));
    assert!(stdout.contains("computed:"));
}
