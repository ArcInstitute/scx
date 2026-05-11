//! Integration tests for Phase 2 CLI extensions.
//!
//! Tests the actual CLI binary end-to-end via `std::process::Command`.

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format::header::{FileHeader, MAGIC};
use scx_format::provenance::ProvenanceEntry;
use scx_format::section::SectionType;
use scx_format::shard::{ShardHeader, SHARD_HEADER_SIZE};
use scx_format::writer::ScxWriter;
use scx_format::ScxReader;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
    FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
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
        n_modalities: 0,
        modality_table_offset: 0,
        modality_table_length: 0,
        reserved: [0u8; 112],
    }
}

fn sample_obs(n: usize) -> arrow::array::RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let types: Vec<&str> = (0..n)
        .map(|i| match i % 3 {
            0 => "T cell",
            1 => "B cell",
            _ => "NK cell",
        })
        .collect();
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, true),
    ]);
    arrow::array::RecordBatch::try_new(
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
) -> PathBuf {
    let path = dir.path().join(filename);
    let total_nnz = n_obs * 2;
    let header = sample_header(n_obs as u64, n_vars as u64, total_nnz as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    let (indptr, indices, values) = sample_shard_data(n_obs, n_vars);
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

    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1710000000,
            action: "create".to_string(),
            tool: "scx-cli 0.1.0".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();

    writer.finish().unwrap();
    path
}

fn scx_cli() -> std::process::Command {
    std::process::Command::new(env!("CARGO_BIN_EXE_scx-cli"))
}

// ---------------------------------------------------------------------------
// Full lifecycle test: convert → info → append → info → delete → info →
//                      compact → info → validate
// ---------------------------------------------------------------------------

#[test]
fn test_lifecycle_append_delete_compact() {
    let dir = tempfile::tempdir().unwrap();
    let target = write_test_file(&dir, "target.scx", 12, 10);
    let source = write_test_file(&dir, "source.scx", 6, 10);

    // Initial info
    let output = scx_cli()
        .args(["info", target.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("12 cells"));

    // Append
    let output = scx_cli()
        .args([
            "append",
            target.to_str().unwrap(),
            "--input",
            source.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "append failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Appended 6 cells"));

    // Info after append
    let output = scx_cli()
        .args(["info", target.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("18 cells"));

    // Delete T cells (every 3rd cell → 6 cells with cell_type == "T cell")
    let output = scx_cli()
        .args([
            "delete",
            target.to_str().unwrap(),
            "--filter",
            "cell_type == 'T cell'",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "delete failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Deleted"));
    assert!(stdout.contains("cells matching"));

    // Info after delete — should show deletion vectors
    let output = scx_cli()
        .args(["info", target.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("deletion_vectors"));

    // Compact
    let compacted = dir.path().join("compacted.scx");
    let output = scx_cli()
        .args([
            "compact",
            target.to_str().unwrap(),
            "--output",
            compacted.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "compact failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Compacted"));

    // Validate compacted file
    let output = scx_cli()
        .args(["validate", compacted.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "validate failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("passed"));
}

// ---------------------------------------------------------------------------
// Rollback lifecycle: append → rollback → info
// ---------------------------------------------------------------------------

#[test]
fn test_rollback_lifecycle() {
    let dir = tempfile::tempdir().unwrap();
    let target = write_test_file(&dir, "rollback_test.scx", 10, 8);
    let source = write_test_file(&dir, "rollback_source.scx", 4, 8);

    // Append
    let output = scx_cli()
        .args([
            "append",
            target.to_str().unwrap(),
            "--input",
            source.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(output.status.success());

    // Verify n_obs increased
    let reader = scx_format::reader::ScxReader::open(&target).unwrap();
    assert_eq!(reader.header().n_obs, 14);
    drop(reader);

    // Rollback
    let output = scx_cli()
        .args(["rollback", target.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "rollback failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Rolled back"));

    // Verify n_obs is back to original
    let reader = scx_format::reader::ScxReader::open(&target).unwrap();
    assert_eq!(reader.header().n_obs, 10);
}

// ---------------------------------------------------------------------------
// Merge lifecycle: create 2 files → merge → info
// ---------------------------------------------------------------------------

#[test]
fn test_merge_lifecycle() {
    let dir = tempfile::tempdir().unwrap();
    let file1 = write_test_file(&dir, "merge1.scx", 8, 10);
    let file2 = write_test_file(&dir, "merge2.scx", 6, 10);
    let merged = dir.path().join("merged.scx");

    let output = scx_cli()
        .args([
            "merge",
            file1.to_str().unwrap(),
            file2.to_str().unwrap(),
            "--output",
            merged.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "merge failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Merged 2 files"));
    assert!(stdout.contains("14 cells"));

    // Validate
    let output = scx_cli()
        .args(["validate", merged.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(output.status.success());
}

// ---------------------------------------------------------------------------
// Query --count
// ---------------------------------------------------------------------------

#[test]
fn test_query_count() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "query_test.scx", 12, 10);

    // Count T cells (every 3rd cell = 4 cells)
    let output = scx_cli()
        .args([
            "query",
            path.to_str().unwrap(),
            "cell_type == 'T cell'",
            "--count",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "query --count failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.trim(), "4");
}

// ---------------------------------------------------------------------------
// Query --count --json
// ---------------------------------------------------------------------------

#[test]
fn test_query_count_json() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "query_json_test.scx", 9, 10);

    let output = scx_cli()
        .args([
            "query",
            path.to_str().unwrap(),
            "cell_type == 'B cell'",
            "--count",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(parsed["count"], 3);
}

// ---------------------------------------------------------------------------
// Query --output writes valid SCX file
// ---------------------------------------------------------------------------

#[test]
fn test_query_output_writes_valid_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "query_output_test.scx", 12, 10);
    let output_path = dir.path().join("subset.scx");

    let output = scx_cli()
        .args([
            "query",
            path.to_str().unwrap(),
            "cell_type == 'T cell'",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "query --output failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Validate output file
    let output = scx_cli()
        .args(["validate", output_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "validate subset failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Check it has the right number of cells
    let reader = scx_format::reader::ScxReader::open(&output_path).unwrap();
    assert_eq!(reader.header().n_obs, 4); // 12/3 = 4 T cells
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

#[test]
fn test_append_mismatched_nvars() {
    let dir = tempfile::tempdir().unwrap();
    let target = write_test_file(&dir, "target_mismatch.scx", 8, 10);
    let source = write_test_file(&dir, "source_mismatch.scx", 4, 20);

    let output = scx_cli()
        .args([
            "append",
            target.to_str().unwrap(),
            "--input",
            source.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("n_vars mismatch"));
}

#[test]
fn test_cli_append_honors_codec_zstd() {
    // P0 #2: `scx append --codec zstd` must produce an appended shard
    // whose on-disk codec_id is Zstd. Before the fix, the codec arg
    // was silently ignored.
    let dir = tempfile::tempdir().unwrap();
    let target = write_test_file(&dir, "tgt_codec.scx", 8, 10);
    let source = write_test_file(&dir, "src_codec.scx", 4, 10);

    let output = scx_cli()
        .args([
            "append",
            target.to_str().unwrap(),
            "--input",
            source.to_str().unwrap(),
            "--codec",
            "zstd",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "append --codec zstd failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let reader = ScxReader::open(&target).unwrap();
    let shards = reader.catalog().shards(SectionType::CsrShard);
    let last = shards.last().expect("at least one CSR shard");
    let bytes = reader.section_bytes(last).unwrap();
    let sh =
        ShardHeader::read_from(&mut std::io::Cursor::new(&bytes[..SHARD_HEADER_SIZE])).unwrap();
    assert_eq!(
        sh.codec_id,
        CodecId::Zstd as u8,
        "appended shard codec must be Zstd"
    );
}

#[test]
fn test_append_rejects_zero_shard_size() {
    // P0 #1: `--shard-size 0` must fail fast at the CLI boundary.
    // The Append clap field is typed `NonZeroU32`, so clap rejects "0"
    // during arg parsing without ever invoking run_append.
    let dir = tempfile::tempdir().unwrap();
    let target = write_test_file(&dir, "target_zero.scx", 8, 10);
    let source = write_test_file(&dir, "source_zero.scx", 4, 10);

    let output = scx_cli()
        .args([
            "append",
            target.to_str().unwrap(),
            "--input",
            source.to_str().unwrap(),
            "--shard-size",
            "0",
        ])
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "expected failure for --shard-size 0"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    // clap's NonZeroU32 parse error mentions "shard-size" and rejects the value.
    assert!(
        stderr.contains("shard-size") || stderr.contains("shard_size"),
        "stderr did not mention shard-size: {stderr}"
    );
}

#[test]
fn test_delete_invalid_predicate() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "delete_invalid.scx", 8, 10);

    let output = scx_cli()
        .args([
            "delete",
            path.to_str().unwrap(),
            "--filter",
            "nonexistent_col == 'x'",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
}

#[test]
fn test_merge_single_file_error() {
    let dir = tempfile::tempdir().unwrap();
    let file1 = write_test_file(&dir, "single.scx", 8, 10);
    let merged = dir.path().join("merged_fail.scx");

    let output = scx_cli()
        .args([
            "merge",
            file1.to_str().unwrap(),
            "--output",
            merged.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("at least 2 files"));
}

#[test]
fn test_compact_output_exists_no_force() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file(&dir, "compact_input.scx", 8, 10);
    let output_path = dir.path().join("compact_output.scx");
    // Create existing output file
    std::fs::write(&output_path, b"existing").unwrap();

    let output = scx_cli()
        .args([
            "compact",
            input.to_str().unwrap(),
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("already exists"));
}

#[test]
fn test_compact_output_exists_with_force() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file(&dir, "compact_force.scx", 8, 10);
    let output_path = dir.path().join("compact_force_out.scx");
    // Create existing output file
    std::fs::write(&output_path, b"existing").unwrap();

    let output = scx_cli()
        .args([
            "compact",
            input.to_str().unwrap(),
            "--output",
            output_path.to_str().unwrap(),
            "--force",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "compact --force failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ---------------------------------------------------------------------------
// Info extensions
// ---------------------------------------------------------------------------

#[test]
fn test_info_json() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "info_json.scx", 8, 10);

    let output = scx_cli()
        .args(["info", path.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(parsed["n_obs"], 8);
    assert_eq!(parsed["n_vars"], 10);
    assert!(parsed["sections"].is_array());
}

#[test]
fn test_info_history_fresh_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "info_history.scx", 8, 10);

    let output = scx_cli()
        .args(["info", path.to_str().unwrap(), "--history"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "info --history failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Manifest history:"));
    assert!(stdout.contains("seq 1"));
}

// ---------------------------------------------------------------------------
// Delete --dry-run
// ---------------------------------------------------------------------------

#[test]
fn test_delete_dry_run() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "dry_run.scx", 12, 10);

    let output = scx_cli()
        .args([
            "delete",
            path.to_str().unwrap(),
            "--filter",
            "cell_type == 'T cell'",
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Would delete"));
    assert!(stdout.contains("4 cells"));

    // Verify file wasn't modified
    let reader = scx_format::reader::ScxReader::open(&path).unwrap();
    assert!(!reader.header().has_deletion_vectors());
}

// ---------------------------------------------------------------------------
// Query --limit
// ---------------------------------------------------------------------------

#[test]
fn test_query_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "query_limit.scx", 12, 10);
    let output_path = dir.path().join("limit_subset.scx");

    let output = scx_cli()
        .args([
            "query",
            path.to_str().unwrap(),
            "cell_type == 'T cell'",
            "--output",
            output_path.to_str().unwrap(),
            "--limit",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "query --limit failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Wrote 2 cells"));
}
