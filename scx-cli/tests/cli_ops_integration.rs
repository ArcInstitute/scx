//! Integration tests for Phase 2 CLI extensions.
//!
//! Tests the actual CLI binary end-to-end via `std::process::Command`.

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::section::SectionType;
use scx_format_io::shard::{ShardHeader, SHARD_HEADER_SIZE};
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, n_vars, nnz, 16384, 0, 0)
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

/// Write a 2-modality test SCX file (`rna`: `rna_vars`, `adt`: `adt_vars`) over a
/// shared `n_obs` obs axis with the global `cell_type` column from [`sample_obs`].
/// Each modality has one CSR shard tiling `[0, n_obs)`; row `r` in modality m
/// expresses gene `r % m_vars`. Mirrors the crate-private `test_utils` helper,
/// which is `#![cfg(test)]`-private and unreachable from this external test crate
/// (scx-cli exposes no `[lib]` target).
fn write_multimodal_test_file(
    dir: &tempfile::TempDir,
    filename: &str,
    n_obs: usize,
    rna_vars: usize,
    adt_vars: usize,
) -> PathBuf {
    use scx_format_io::modality::ModalityType;
    let path = dir.path().join(filename);
    let header = sample_header(n_obs as u64, rna_vars.max(adt_vars) as u64, n_obs as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();

    let rna_id = writer
        .add_modality(
            "rna",
            ModalityType::Rna,
            CodecId::None,
            ValueEncoding::Uint8,
            false,
        )
        .unwrap();
    let adt_id = writer
        .add_modality(
            "adt",
            ModalityType::Protein,
            CodecId::None,
            ValueEncoding::Uint8,
            false,
        )
        .unwrap();
    writer.write_var_for(rna_id, &sample_var(rna_vars)).unwrap();
    writer.write_var_for(adt_id, &sample_var(adt_vars)).unwrap();
    writer.set_modality_n_vars(rna_id, rna_vars as u64).unwrap();
    writer.set_modality_n_vars(adt_id, adt_vars as u64).unwrap();

    for (id, m_vars) in [(rna_id, rna_vars), (adt_id, adt_vars)] {
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for r in 0..n_obs {
            indices.push((r % m_vars) as u32);
            values.push(((r + 1) % 256) as u8);
            indptr.push(indptr.last().unwrap() + 1);
        }
        let shard = scx_format_io::ShardBuffers::new(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
        );
        writer.write_csr_shard_for(id, 0, shard).unwrap();
    }

    writer.finish().unwrap();
    path
}

fn scx_cli() -> std::process::Command {
    std::process::Command::new(env!("CARGO_BIN_EXE_scx"))
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
        .args(["append", target.to_str().unwrap(), source.to_str().unwrap()])
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
// Multimodal delete lifecycle: build 2-modality file → delete --filter →
//                              info (shows deletion vectors) → compact → validate
// ---------------------------------------------------------------------------

/// A whole-cell delete on a multimodal file removes the cell from every
/// modality. This exercises the per-modality deletion-vector path end-to-end
/// through the `scx` binary: an obs predicate delete, the `info` deletion
/// summary, and a multimodal compact that physically reclaims the rows from
/// both modalities.
#[test]
fn test_lifecycle_multimodal_delete_compact() {
    let dir = tempfile::tempdir().unwrap();
    // 12 cells; rows 0,3,6,9 are "T cell" (cell_type cycles T/B/NK).
    let target = write_multimodal_test_file(&dir, "mm_target.scx", 12, 8, 4);

    // Initial info — should report a multimodal file with 12 cells.
    let output = scx_cli()
        .args(["info", target.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "info failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("12 cells"));

    // Delete the T cells (rows 0,3,6,9 → 4 cells) via an obs predicate.
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

    // Info after delete — should surface the deletion vectors flag.
    let output = scx_cli()
        .args(["info", target.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("deletion_vectors"));

    // Compact — physically reclaims the 4 deleted cells from both modalities.
    let compacted = dir.path().join("mm_compacted.scx");
    let output = scx_cli()
        .args([
            "compact",
            target.to_str().unwrap(),
            compacted.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "compact failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("Compacted"));

    // Compacted file: logical n_obs dropped to 8, both modalities aligned.
    let reader = ScxReader::open(&compacted).unwrap();
    assert_eq!(reader.header().n_obs, 8, "12 - 4 T cells");
    let rna_id = reader.modality_id("rna").unwrap();
    let adt_id = reader.modality_id("adt").unwrap();
    for m in [rna_id, adt_id] {
        let csr = reader.read_all_csr_shards_for(m).unwrap();
        assert_eq!(csr.shape.0, 8, "modality {m} row count after compact");
    }

    // Validate the compacted multimodal file.
    let output = scx_cli()
        .args(["validate", compacted.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "validate failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("passed"));
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
        .args(["append", target.to_str().unwrap(), source.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(output.status.success());

    // Verify n_obs increased
    let reader = scx_format_io::reader::ScxReader::open(&target).unwrap();
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
    let reader = scx_format_io::reader::ScxReader::open(&target).unwrap();
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

// F9: `scx query` accepts the obs predicate via `--filter` (consistent with
// `scx subset` / `scx delete`), in addition to the positional form. Both
// spellings must produce identical results.
#[test]
fn test_query_count_filter_flag() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "query_filter_flag.scx", 12, 10);

    let output = scx_cli()
        .args([
            "query",
            path.to_str().unwrap(),
            "--filter",
            "cell_type == 'T cell'",
            "--count",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "query --filter --count failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout.trim(),
        "4",
        "--filter must match the positional form"
    );
}

// F1 (2026-06-14): omitting the predicate entirely is now valid — the query
// runs over all cells, mirroring pyscx `query()` with no `filter_obs`. (This
// supersedes the earlier F9 behaviour, which rejected a missing predicate.)
#[test]
fn test_query_no_filter_counts_all() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "query_no_filter.scx", 12, 10);

    let output = scx_cli()
        .args(["query", path.to_str().unwrap(), "--count"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "query with no predicate must succeed (counts all cells): {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout.trim(),
        "12",
        "no predicate must count all cells, got: {stdout}"
    );
}

// F9: supplying the predicate both positionally and via `--filter` is rejected
// by clap (`conflicts_with`).
#[test]
fn test_query_both_filter_forms_conflict() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "query_both_filters.scx", 12, 10);

    let output = scx_cli()
        .args([
            "query",
            path.to_str().unwrap(),
            "cell_type == 'T cell'",
            "--filter",
            "cell_type == 'B cell'",
            "--count",
        ])
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "supplying both positional and --filter must fail"
    );
}

// CLI6 regression: `--count` reports the true match count and is NOT capped
// by `--limit` (which is output-only). Previously `--count --limit 1` printed
// `min(matched, 1) = 1`.
#[test]
fn test_query_count_ignores_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "query_count_limit.scx", 12, 10);

    let output = scx_cli()
        .args([
            "query",
            path.to_str().unwrap(),
            "cell_type == 'T cell'",
            "--count",
            "--limit",
            "1",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "query --count --limit failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout.trim(),
        "4",
        "--count must report the true match count regardless of --limit"
    );
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
    let reader = scx_format_io::reader::ScxReader::open(&output_path).unwrap();
    assert_eq!(reader.header().n_obs, 4); // 12/3 = 4 T cells
}

// ---------------------------------------------------------------------------
// Fix 1 regression: `scx query <local.scxd>/ ... --output` must preserve the
// source value encoding. Before the fix, `is_cloud_url` returned true for any
// local directory, so this path forced ValueEncoding::Float32 even though the
// shard files on disk were Uint8.
// ---------------------------------------------------------------------------

#[cfg(feature = "cloud")]
#[test]
fn test_query_local_exploded_preserves_value_encoding() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = write_test_file(&dir, "src.scx", 12, 10);
    let exploded_dir = dir.path().join("src.scxd");

    // Explode the .scx into an .scxd/ directory.
    let out = scx_cli()
        .args([
            "explode",
            scx_path.to_str().unwrap(),
            exploded_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "explode failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Run `scx query <local.scxd>/ ... --output out.scx`.
    let out_path = dir.path().join("subset.scx");
    let out = scx_cli()
        .args([
            "query",
            exploded_dir.to_str().unwrap(),
            "cell_type == 'T cell'",
            "--output",
            out_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "query failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Inspect the first CSR shard of the output and assert encoding is Uint8
    // (the source encoding), NOT Float32 (the pre-fix fallback).
    let reader = ScxReader::open(&out_path).unwrap();
    let shards = reader.catalog().shards(SectionType::CsrShard);
    let first = shards.first().expect("output must have at least one shard");
    let bytes = reader.section_bytes(first).unwrap();
    let sh =
        ShardHeader::read_from(&mut std::io::Cursor::new(&bytes[..SHARD_HEADER_SIZE])).unwrap();
    assert_eq!(
        ValueEncoding::from_u8(sh.value_encoding).unwrap(),
        ValueEncoding::Uint8,
        "output should preserve source's Uint8 encoding, got {:?}",
        ValueEncoding::from_u8(sh.value_encoding)
    );
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
        .args(["append", target.to_str().unwrap(), source.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("n_vars mismatch"));
}

#[test]
fn test_cli_append_positional_source() {
    // B2: `scx append <target> <source>` (positional source) must actually
    // append. Before the F1 fix the source was a `--input` flag, so a bare
    // positional was dropped and the append silently no-op'd.
    let dir = tempfile::tempdir().unwrap();
    let target = write_test_file(&dir, "pos_target.scx", 8, 10);
    let source = write_test_file(&dir, "pos_source.scx", 4, 10);

    let output = scx_cli()
        .args(["append", target.to_str().unwrap(), source.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "positional append failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Target grew from 8 → 12 cells.
    let reader = ScxReader::open(&target).unwrap();
    assert_eq!(reader.n_obs(), 12, "appended cell count must be 8 + 4");
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
fn test_merge_missing_output_names_flag() {
    // A user copying `scx convert`'s positional-output habit types
    // `scx merge a b out` — `out` is swallowed as a third input and --output
    // is missing. The error must name --output and explain it's a flag.
    let dir = tempfile::tempdir().unwrap();
    let file1 = write_test_file(&dir, "m1.scx", 8, 10);
    let file2 = write_test_file(&dir, "m2.scx", 6, 10);
    let out = dir.path().join("out.scx");

    let output = scx_cli()
        .args([
            "merge",
            file1.to_str().unwrap(),
            file2.to_str().unwrap(),
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--output"),
        "error must name --output: {stderr}"
    );
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

#[test]
fn test_info_history_after_append() {
    // `append` writes a new catalog whose `prev_catalog_offset` chains to the
    // original — so `--history` walks the chain via `try_read_catalog_at`,
    // which the fresh-file test never exercises (its loop never runs). This
    // covers the exact-size catalog read on a real prior catalog.
    let dir = tempfile::tempdir().unwrap();
    let target = write_test_file(&dir, "hist_target.scx", 8, 10);
    let source = write_test_file(&dir, "hist_source.scx", 6, 10);

    let output = scx_cli()
        .args(["append", target.to_str().unwrap(), source.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "append failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = scx_cli()
        .args(["info", target.to_str().unwrap(), "--history"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "info --history failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Manifest history:"));
    // Both the appended head (seq 2) and the chained prior catalog (seq 1).
    assert!(
        stdout.contains("seq 2"),
        "expected appended manifest seq 2 in history:\n{stdout}"
    );
    assert!(
        stdout.contains("seq 1"),
        "expected prior manifest seq 1 (walked via try_read_catalog_at):\n{stdout}"
    );
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
    let reader = scx_format_io::reader::ScxReader::open(&path).unwrap();
    assert!(!reader.header().has_deletion_vectors());
}

// ---------------------------------------------------------------------------
// `or` with a NULL operand — the destructive surfaces
// ---------------------------------------------------------------------------

/// 6 cells. `cell_type` is NULL on rows 0 and 3; `n_genes` is non-null and
/// exceeds 500 on rows 0, 1 and 4. Row 0 is the interesting one: NULL
/// `cell_type` **and** a matching `n_genes`.
fn write_null_bearing_file(dir: &tempfile::TempDir, filename: &str) -> PathBuf {
    use arrow::array::Int64Array;

    let n_obs = 6usize;
    let n_vars = 4usize;
    let path = dir.path().join(filename);
    let mut writer = ScxWriter::new(
        &path,
        sample_header(n_obs as u64, n_vars as u64, (n_obs * 2) as u64),
    )
    .unwrap();

    let ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    let cell_type = vec![
        None,
        Some("B cell"),
        Some("T cell"),
        None,
        Some("T cell"),
        Some("B cell"),
    ];
    let n_genes = vec![900i64, 700, 100, 200, 800, 300];
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, true),
        Field::new("n_genes", DataType::Int64, false),
    ]);
    let obs = arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(cell_type)),
            Arc::new(Int64Array::from(n_genes)),
        ],
    )
    .unwrap();

    writer.write_obs(&obs).unwrap();
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
    writer.finish().unwrap();
    path
}

/// `scx delete --filter` runs the same predicate evaluator as the query engine,
/// so the non-Kleene `or` under-deleted: a cell whose `cell_type` is NULL was
/// spared even when the *other* operand matched it. Under-deleting is the
/// dangerous direction — the operator believes those cells are gone.
///
/// `cell_type == 'B cell' or n_genes > 500` must match rows 0 (NULL, 900),
/// 1 ('B cell', 700), 4 (800) and 5 ('B cell'). Row 0 is what regressed.
#[test]
fn test_delete_or_matches_rows_with_a_null_operand() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_null_bearing_file(&dir, "delete_null_or.scx");

    let output = scx_cli()
        .args([
            "delete",
            path.to_str().unwrap(),
            "--filter",
            "cell_type == 'B cell' or n_genes > 500",
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "delete --dry-run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("4 cells"),
        "`or` with a NULL operand must match 4 cells (rows 0,1,4,5), not 3 — \
         row 0 has a NULL cell_type and n_genes=900. Got: {stdout}"
    );

    // Each operand alone, so the count above cannot be an accident of the
    // fixture: 'B cell' matches rows 1 and 5; n_genes > 500 matches 0, 1, 4.
    for (expr, want) in [
        ("cell_type == 'B cell'", "2 cells"),
        ("n_genes > 500", "3 cells"),
    ] {
        let out = scx_cli()
            .args([
                "delete",
                path.to_str().unwrap(),
                "--filter",
                expr,
                "--dry-run",
            ])
            .output()
            .unwrap();
        assert!(out.status.success());
        let s = String::from_utf8_lossy(&out.stdout);
        assert!(s.contains(want), "`{expr}` should match {want}; got: {s}");
    }
}

/// The copy-out sibling of the delete case: `scx subset --filter` writes a new
/// file containing the matching cells, so the non-Kleene `or` silently produced
/// a *smaller* subset than asked for — cells whose `cell_type` was NULL were
/// left out even when `n_genes` matched them. Unlike `--dry-run` delete, this
/// asserts against the bytes actually written.
#[test]
fn test_subset_or_keeps_rows_with_a_null_operand() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_null_bearing_file(&dir, "subset_null_or.scx");
    let out = dir.path().join("subset_null_or_out.scx");

    let res = scx_cli()
        .args([
            "subset",
            path.to_str().unwrap(),
            out.to_str().unwrap(),
            "--filter",
            "cell_type == 'B cell' or n_genes > 500",
        ])
        .output()
        .unwrap();
    assert!(
        res.status.success(),
        "subset failed: {}",
        String::from_utf8_lossy(&res.stderr)
    );

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(
        reader.header().n_obs,
        4,
        "`or` with a NULL operand must keep 4 cells (rows 0,1,4,5); row 0 has a \
         NULL cell_type and n_genes=900"
    );

    // Identity, not just cardinality: the NULL-cell_type row must be present.
    let obs = reader.read_obs().unwrap();
    let idx = obs.schema().index_of("cell_id").unwrap();
    let ids = arrow::compute::cast(obs.column(idx), &DataType::Utf8).unwrap();
    let ids = ids
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|s| s.unwrap().to_string())
        .collect::<Vec<_>>();
    assert_eq!(ids, vec!["cell_0", "cell_1", "cell_4", "cell_5"]);
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

// ---------------------------------------------------------------------------
// D3: a cloud subcommand on a non-cloud build hints at `--features cloud`
// rather than leaving the user with a bare "unrecognized subcommand".
// ---------------------------------------------------------------------------

#[cfg(not(feature = "cloud"))]
#[test]
fn test_cloud_subcommand_hint_on_non_cloud_build() {
    let output = scx_cli()
        .arg("pull")
        .arg("gs://b/x.scxd")
        .arg("out.scx")
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "pull should fail on a non-cloud build"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("cloud subcommand") && stderr.contains("--features cloud"),
        "expected a cloud-feature hint, got stderr: {stderr}"
    );
}

#[cfg(not(feature = "cloud"))]
#[test]
fn test_unknown_subcommand_has_no_cloud_hint() {
    // A genuine typo must NOT get the cloud hint (it isn't a cloud command).
    let output = scx_cli().arg("flibble").output().unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("cloud subcommand"),
        "a non-cloud typo should not get the cloud hint, got: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// optimize --shard-obs: migrate a legacy single-section obs to sharded layout
// ---------------------------------------------------------------------------

#[test]
fn test_optimize_shard_obs_always_shards_single_section() {
    let dir = tempfile::tempdir().unwrap();
    // write_test_file emits a single-section obs (write_obs), so this is a
    // legacy-layout fixture.
    let input = write_test_file(&dir, "legacy.scx", 8, 10);
    let output = dir.path().join("sharded.scx");

    assert_eq!(
        ScxReader::open(&input).unwrap().obs_metadata_shard_count(),
        0,
        "fixture is single-section obs"
    );

    let out = scx_cli()
        .args([
            "optimize",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--shard-obs",
            "always",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "optimize --shard-obs always failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let reader = ScxReader::open(&output).unwrap();
    assert!(
        reader.obs_metadata_shard_count() > 0,
        "single-section obs migrated to sharded layout"
    );
    assert!(
        !reader
            .catalog()
            .entries
            .iter()
            .any(|e| e.section_type == SectionType::ObsMetadata),
        "no single-section ObsMetadata remains"
    );
    // Rows round-trip.
    assert_eq!(reader.read_obs().unwrap().num_rows(), 8);
}

#[test]
fn test_optimize_shard_obs_auto_keeps_small_single_section() {
    let dir = tempfile::tempdir().unwrap();
    // write_test_file uses shard_target_rows=16384, so n_obs=8 is far below the
    // threshold → auto must keep the single section (the no-op-for-small-files
    // contract), completing the CLI-level off/auto/always matrix.
    let input = write_test_file(&dir, "legacy.scx", 8, 10);
    let output = dir.path().join("auto_small.scx");

    let out = scx_cli()
        .args([
            "optimize",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--shard-obs",
            "auto",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());

    assert_eq!(
        ScxReader::open(&output).unwrap().obs_metadata_shard_count(),
        0,
        "auto keeps a sub-threshold single-section obs as a single section"
    );
}

#[test]
fn test_optimize_shard_obs_off_keeps_single_section() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file(&dir, "legacy.scx", 8, 10);
    let output = dir.path().join("kept.scx");

    let out = scx_cli()
        .args([
            "optimize",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--shard-obs",
            "off",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());

    assert_eq!(
        ScxReader::open(&output).unwrap().obs_metadata_shard_count(),
        0,
        "off preserves the single-section obs layout"
    );
}

#[test]
fn test_optimize_rejects_invalid_shard_obs() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file(&dir, "legacy.scx", 8, 10);
    let output = dir.path().join("out.scx");

    let out = scx_cli()
        .args([
            "optimize",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--shard-obs",
            "sometimes",
        ])
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "clap value_parser must reject an invalid --shard-obs value"
    );
}

// ---------------------------------------------------------------------------
// `scx build-csc` — in-place form (dogfood F9) + framing preservation
// ---------------------------------------------------------------------------

/// Write a **framed** (v4 / shard-v2) test file, so the framing-preservation
/// assertions below have something to preserve. `write_test_file` produces an
/// unframed v3 file, against which "output is still v4" is vacuous.
fn write_framed_test_file(
    dir: &tempfile::TempDir,
    filename: &str,
    n_obs: usize,
    n_vars: usize,
) -> PathBuf {
    let path = dir.path().join(filename);
    let mut header = sample_header(n_obs as u64, n_vars as u64, (n_obs * 2) as u64);
    header.format_version = scx_format_io::header::CURRENT_FORMAT_VERSION;
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.set_framing(Some(scx_format_io::FramingConfig {
        row_group_rows: 4,
        ..Default::default()
    }));
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
    let (indptr, indices, values) = sample_shard_data(n_obs, n_vars);
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::Zstd,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer.finish().unwrap();
    path
}

/// F9: `scx build-csc <INPUT>` with no `<OUTPUT>` adds the sidecar to the input
/// in place. The CSC store is described everywhere as a sidecar *on* a file, and
/// its in-place neighbours (`obs-import` / `doublet-import` /
/// `cellbender-import`) all mutate; requiring an `<OUTPUT>` here read as a
/// missing argument rather than a design choice.
#[test]
fn test_build_csc_in_place_adds_sidecar_to_input() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file(&dir, "in_place.scx", 8, 6);

    // Anti-vacuous: the fixture must NOT already have a sidecar, or "has_csc"
    // afterwards would prove nothing.
    assert!(
        !ScxReader::open(&input).unwrap().header().has_csc(),
        "fixture precondition: input must start without a CSC sidecar"
    );

    let out = scx_cli()
        .args(["build-csc", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "build-csc with no <OUTPUT> must succeed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // The *same path* now carries the sidecar — that is what "in place" means.
    let reader = ScxReader::open(&input).unwrap();
    assert!(
        reader.header().has_csc(),
        "in-place build-csc must set has_csc on the input path"
    );
    assert!(
        reader.header().n_csc_shards > 0,
        "in-place build-csc must emit CSC shards"
    );
    assert_eq!(reader.header().n_obs, 8, "obs axis must be unchanged");
    assert_eq!(reader.header().n_vars, 6, "var axis must be unchanged");

    // No stray staging file: `rebuild_csc_inplace` writes `*.rebuild_csc.tmp`
    // beside the target and must always clean it up.
    let leftovers: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(".tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "in-place build-csc leaked staging files: {leftovers:?}"
    );
}

/// `build-csc` flattens every CSR shard against the single top-level shape, so a
/// multimodal input must be refused. `pyscx.build_csc` had guarded this since it
/// was written; the CLI reached `read_var()` and failed with an opaque
/// `section not found: var`. The guard now lives in `scx_ops::run_build_csc`, so
/// **both** CLI forms report it — and the in-place form must additionally leave
/// the target untouched and leak no staging file.
#[test]
fn test_build_csc_rejects_multimodal_on_both_forms() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_multimodal_test_file(&dir, "cite.scx", 8, 6, 3);
    let before = std::fs::read(&input).unwrap();

    for args in [
        vec!["build-csc", input.to_str().unwrap()],
        vec![
            "build-csc",
            input.to_str().unwrap(),
            dir.path().join("out.scx").to_str().unwrap(),
        ],
    ] {
        let out = scx_cli().args(&args).output().unwrap();
        assert!(
            !out.status.success(),
            "build-csc {args:?} must refuse a multimodal input"
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("multimodal"),
            "the error must name the actual problem, not `section not found: \
             var`; got: {stderr}"
        );
        assert!(
            stderr.contains("--modality"),
            "and must name the way out (`scx subset --modality NAME`); got: {stderr}"
        );
    }

    // The in-place attempt must be a true no-op.
    assert_eq!(
        std::fs::read(&input).unwrap(),
        before,
        "a refused in-place build-csc must leave the target byte-identical"
    );
    let leftovers: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(".tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "a refused in-place build-csc leaked staging files: {leftovers:?}"
    );
}

/// In place does **not** imply undoable, and `docs/operations.md` now says so —
/// pin it so the claim cannot silently go stale in either direction.
///
/// `rebuild_csc_inplace` stages a wholly new file via `run_build_csc` and renames
/// it over the target, so it carries no prior catalog and does not go through the
/// `prepare_in_place` / `commit_in_place` manifest chain the import ops use.
#[test]
fn test_build_csc_in_place_is_not_rollback_able() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file(&dir, "no_rollback.scx", 8, 6);

    let out = scx_cli()
        .args(["build-csc", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(ScxReader::open(&input).unwrap().header().has_csc());

    let rb = scx_cli()
        .args(["rollback", input.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        !rb.status.success(),
        "an in-place build-csc leaves no prior catalog, so rollback must fail \
         loudly rather than appear to succeed and change nothing"
    );
    let stderr = String::from_utf8_lossy(&rb.stderr);
    assert!(
        stderr.contains("no previous catalog"),
        "the failure must say why, got: {stderr}"
    );
    // And the file is unchanged by the refused rollback.
    assert!(
        ScxReader::open(&input).unwrap().header().has_csc(),
        "a refused rollback must not have modified the file"
    );
}

/// Both forms must preserve row-group framing. The copy-out arm used to pass
/// `framing = None`, which routes through
/// `rewrite_output_format_version(&[4], 1) == 3` and silently downgraded a
/// framed v4 input to unframed v3 — so adding the in-place form without fixing
/// it would have put two framing behaviours on one subcommand.
#[test]
fn test_build_csc_preserves_v4_framing_both_forms() {
    let dir = tempfile::tempdir().unwrap();
    let in_place = write_framed_test_file(&dir, "framed_in_place.scx", 12, 6);
    let copy_src = write_framed_test_file(&dir, "framed_src.scx", 12, 6);
    let copy_out = dir.path().join("framed_out.scx");

    for args in [
        vec!["build-csc", in_place.to_str().unwrap()],
        vec![
            "build-csc",
            copy_src.to_str().unwrap(),
            copy_out.to_str().unwrap(),
        ],
    ] {
        let out = scx_cli().args(&args).output().unwrap();
        assert!(
            out.status.success(),
            "build-csc {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    for (label, path) in [("in-place", &in_place), ("copy-out", &copy_out)] {
        let reader = ScxReader::open(path).unwrap();
        assert_eq!(
            reader.header().format_version,
            scx_format_io::header::CURRENT_FORMAT_VERSION,
            "{label} build-csc on a v4 input must stay v4, not downgrade to v3"
        );
        assert!(
            reader.header().has_csc(),
            "{label} build-csc must emit the sidecar"
        );
        for entry in &reader.catalog().shards(SectionType::CsrShard) {
            let sh = reader.read_shard_header(entry).unwrap();
            assert!(
                sh.shard_format_version > 1,
                "{label}: CSR shard '{}' must stay framed v2, got v{}",
                entry.name,
                sh.shard_format_version
            );
        }
    }
}

/// `--force` overwrites an `<OUTPUT>`; there is no output to overwrite in the
/// in-place form. Refuse rather than ignore — silently accepting it would imply
/// a guard that does not exist.
#[test]
fn test_build_csc_in_place_rejects_force() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file(&dir, "force_no_output.scx", 6, 4);

    let out = scx_cli()
        .args(["build-csc", input.to_str().unwrap(), "--force"])
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "--force without <OUTPUT> must be rejected, not silently ignored"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--force"),
        "the error must name the offending flag, got: {stderr}"
    );

    // And it must not have half-run: no sidecar, no staging file.
    assert!(
        !ScxReader::open(&input).unwrap().header().has_csc(),
        "a rejected invocation must not have modified the input"
    );
}

// ---------------------------------------------------------------------------
// `scx subset --index-*` — query-ready subsets (dogfood F7)
// ---------------------------------------------------------------------------

/// Write a multi-shard file whose `cell_type` is **clustered** (one value per
/// contiguous run) rather than cycled.
///
/// Clustering is what makes Level-1 pruning observable at all: the catalog's
/// per-shard category bitsets can only eliminate a shard when the queried value
/// is absent from it. With `sample_obs`'s T/B/NK cycle every shard holds every
/// value, so "0 shards eliminated" would be correct with or without an index and
/// the assertion below would be vacuous.
fn write_clustered_test_file(
    dir: &tempfile::TempDir,
    filename: &str,
    n_obs: usize,
    n_vars: usize,
    shard_rows: usize,
) -> PathBuf {
    let path = dir.path().join(filename);
    let mut header = sample_header(n_obs as u64, n_vars as u64, (n_obs * 2) as u64);
    header.shard_target_rows = shard_rows as u32;
    let mut writer = ScxWriter::new(&path, header).unwrap();

    // 4 contiguous cell_type runs; `disease` splits the file in half so a
    // filter-subset retains a contiguous prefix spanning 2 of the 4 runs.
    let quarter = n_obs / 4;
    let half = n_obs / 2;
    let cell_ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    let cell_types: Vec<&str> = (0..n_obs)
        .map(|i| match i / quarter {
            0 => "T cell",
            1 => "B cell",
            2 => "NK cell",
            _ => "monocyte",
        })
        .collect();
    let diseases: Vec<&str> = (0..n_obs)
        .map(|i| if i < half { "normal" } else { "covid" })
        .collect();
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, true),
        Field::new("disease", DataType::Utf8, true),
    ]);
    let obs = arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                cell_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(cell_types)),
            Arc::new(StringArray::from(diseases)),
        ],
    )
    .unwrap();
    writer.write_obs(&obs).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    // One CSR shard per `shard_rows` rows, so the output has several shards to
    // prune among.
    let mut row = 0usize;
    while row < n_obs {
        let rows = std::cmp::min(shard_rows, n_obs - row);
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for r in row..row + rows {
            indices.push((r % n_vars) as u32);
            values.push(((r % 200) + 1) as u8);
            indptr.push(indptr.last().unwrap() + 1);
        }
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row as u64,
            )
            .unwrap();
        row += rows;
    }
    writer.finish().unwrap();
    path
}

/// F7 end-to-end: `scx subset --index-obs` makes the subset query-ready.
///
/// Asserts the *user-visible* property, not merely that a section exists —
/// `--explain`'s Level-1 shard elimination, measured against the identical
/// subset built without the flag. Without an obs predicate index the engine has
/// no category dictionary, so a `Utf8` equality predicate cannot resolve to the
/// catalog's per-shard category bitsets and nothing prunes.
#[test]
fn test_subset_index_preset_makes_output_query_ready() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_clustered_test_file(&dir, "atlas.scx", 400, 20, 50);
    let no_index = dir.path().join("subset_plain.scx");
    let indexed = dir.path().join("subset_indexed.scx");

    for (out, extra) in [
        (&no_index, Vec::new()),
        (&indexed, vec!["--index-obs", "cell_type"]),
    ] {
        let mut args = vec![
            "subset",
            input.to_str().unwrap(),
            out.to_str().unwrap(),
            "--filter",
            "disease == 'normal'",
            "--shard-size",
            "25",
        ];
        args.extend(extra);
        let res = scx_cli().args(&args).output().unwrap();
        assert!(
            res.status.success(),
            "subset failed: {}",
            String::from_utf8_lossy(&res.stderr)
        );
    }

    // The section itself: absent without the flag, present with it.
    assert!(
        ScxReader::open(&no_index)
            .unwrap()
            .read_obs_predicate_index_bytes()
            .unwrap()
            .is_none(),
        "no --index-* flag must leave the subset unindexed"
    );
    assert!(
        ScxReader::open(&indexed)
            .unwrap()
            .read_obs_predicate_index_bytes()
            .unwrap()
            .is_some(),
        "--index-obs must write an obs predicate index"
    );

    // And what it buys: pushdown.
    let explain = |path: &PathBuf| -> String {
        let out = scx_cli()
            .args([
                "query",
                path.to_str().unwrap(),
                "--filter",
                "cell_type == 'T cell'",
                "--count",
                "--explain",
            ])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "query failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        // `--explain` goes to stderr, the count to stdout; return both.
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stderr),
            String::from_utf8_lossy(&out.stdout)
        )
    };

    let plain = explain(&no_index);
    let with_index = explain(&indexed);

    // Correctness first: both must return the same rows. A pruning win that
    // changed the answer would be a catastrophe, not an optimisation.
    let matched = |s: &str| {
        s.lines()
            .find_map(|l| l.trim().strip_prefix("matched rows: "))
            .map(|v| v.trim().to_string())
            .unwrap_or_else(|| panic!("no `matched rows:` line in explain output:\n{s}"))
    };
    assert_eq!(
        matched(&plain),
        matched(&with_index),
        "indexing must not change which rows match"
    );

    let eliminated = |s: &str| -> (u32, u32) {
        let line = s
            .lines()
            .find_map(|l| {
                l.trim()
                    .strip_prefix("Level 1 (catalog-stats) eliminated: ")
            })
            .unwrap_or_else(|| panic!("no Level-1 line in explain output:\n{s}"));
        let (num, den) = line.trim().split_once('/').expect("N/M form");
        (num.parse().unwrap(), den.parse().unwrap())
    };
    let (plain_elim, plain_total) = eliminated(&plain);
    let (idx_elim, idx_total) = eliminated(&with_index);

    assert_eq!(
        plain_total, idx_total,
        "both subsets must have the same shard count, or the comparison is unfair"
    );
    assert_eq!(
        plain_elim, 0,
        "without an obs predicate index there is no category dictionary, so a \
         Utf8 equality predicate cannot prune any shard (got {plain_elim}/{plain_total})"
    );
    assert!(
        idx_elim > 0,
        "the whole point of --index-obs on a subset: the rebuilt index must let \
         Level 1 eliminate shards (got {idx_elim}/{idx_total})"
    );
}

/// The drop warning must name the remedy. Before F7 it said only "predicate
/// indices dropped (invalid after subsetting)", which left the user with no
/// one-step way to get a query-ready subset — and the docs pointed at
/// `scx convert --index-obs`, which cannot take an `.scx` input at all.
#[test]
fn test_subset_index_drop_warning_names_the_flags() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_clustered_test_file(&dir, "atlas_warn.scx", 200, 10, 50);

    // Give the input an index to drop, via a first indexed subset.
    let indexed_input = dir.path().join("indexed_input.scx");
    let res = scx_cli()
        .args([
            "subset",
            input.to_str().unwrap(),
            indexed_input.to_str().unwrap(),
            "--filter",
            "disease == 'normal'",
            "--index-obs",
            "cell_type",
        ])
        .output()
        .unwrap();
    assert!(res.status.success());

    // Now subset THAT without any --index-* flag: the warning must fire and
    // must name the flags.
    let out = dir.path().join("dropped.scx");
    let res = scx_cli()
        .args([
            "subset",
            indexed_input.to_str().unwrap(),
            out.to_str().unwrap(),
            "--filter",
            "cell_type == 'T cell'",
        ])
        .output()
        .unwrap();
    assert!(res.status.success());
    let stderr = String::from_utf8_lossy(&res.stderr);
    assert!(
        stderr.contains("predicate indices dropped"),
        "the drop must still be reported, got: {stderr}"
    );
    assert!(
        stderr.contains("--index-obs") && stderr.contains("--index-preset"),
        "the warning must name the flags that rebuild the index, got: {stderr}"
    );
    assert!(
        !stderr.contains("scx convert"),
        "must not point at `scx convert`, which cannot read an .scx input: {stderr}"
    );

    // Control: with a flag, no drop warning — the outcome is reported instead.
    let out2 = dir.path().join("rebuilt.scx");
    let res = scx_cli()
        .args([
            "subset",
            indexed_input.to_str().unwrap(),
            out2.to_str().unwrap(),
            "--filter",
            "cell_type == 'T cell'",
            "--index-obs",
            "cell_type",
        ])
        .output()
        .unwrap();
    assert!(res.status.success());
    let stderr2 = String::from_utf8_lossy(&res.stderr);
    assert!(
        !stderr2.contains("predicate indices dropped"),
        "a requested rebuild must not also warn about dropping, got: {stderr2}"
    );
    assert!(
        String::from_utf8_lossy(&res.stdout).contains("indexed obs columns: cell_type"),
        "the rebuild must report what it indexed, got: {}",
        String::from_utf8_lossy(&res.stdout)
    );
}

/// `--index-preset cellxgene` on a file lacking the preset's columns must warn
/// per column through the shared renderer, not fail — matching `merge` /
/// `compact` / `append`, whose wording comes from the same `emit_index_summary`.
#[test]
fn test_subset_index_preset_missing_columns_warns_like_siblings() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_clustered_test_file(&dir, "atlas_preset.scx", 200, 10, 50);
    let out = dir.path().join("preset.scx");

    let res = scx_cli()
        .args([
            "subset",
            input.to_str().unwrap(),
            out.to_str().unwrap(),
            "--filter",
            "disease == 'normal'",
            "--index-preset",
            "cellxgene",
        ])
        .output()
        .unwrap();
    assert!(
        res.status.success(),
        "a preset with missing columns must warn, not fail: {}",
        String::from_utf8_lossy(&res.stderr)
    );
    let stderr = String::from_utf8_lossy(&res.stderr);
    assert!(
        stderr.contains("subset: preset obs index column"),
        "preset skips must be attributed to `subset` via the shared renderer, \
         got: {stderr}"
    );
    // The preset's `cell_type` / `disease` DO exist here, so they must land.
    assert!(
        ScxReader::open(&out)
            .unwrap()
            .read_obs_predicate_index_bytes()
            .unwrap()
            .is_some(),
        "the columns the preset did find must still be indexed"
    );
}

// ---------------------------------------------------------------------------
// `scx query` on an empty result — category-miss note (dogfood E1)
// ---------------------------------------------------------------------------

/// Write a file whose `cell_type` is a **dictionary-encoded** categorical
/// declaring a value no row uses.
///
/// That unused value is the load-bearing half of E1: querying it is a *genuine*
/// zero and must stay silent, while a value outside the vocabulary is a typo and
/// must be called out. A fixture with only used categories cannot tell a
/// correct implementation from one that flags every empty result.
fn write_categorical_test_file(dir: &tempfile::TempDir, filename: &str) -> PathBuf {
    use arrow::array::{ArrayRef, DictionaryArray, Int32Array};
    use arrow::datatypes::Int32Type;

    let path = dir.path().join(filename);
    let n_obs = 6usize;
    let n_vars = 4usize;
    let mut header = sample_header(n_obs as u64, n_vars as u64, (n_obs * 2) as u64);
    header.shard_target_rows = 16384;
    let mut writer = ScxWriter::new(&path, header).unwrap();

    // Vocabulary: 3 values; keys only ever reference 0 and 1, so "monocyte" is
    // declared-but-unused.
    let keys = Int32Array::from(vec![0, 1, 0, 1, 0, 1]);
    let values = StringArray::from(vec!["B cell", "NK cell", "monocyte"]);
    let cell_type: ArrayRef =
        Arc::new(DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).unwrap());
    let ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    let obs = arrow::array::RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new(
                "cell_type",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            ),
        ])),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            cell_type,
        ],
    )
    .unwrap();
    writer.write_obs(&obs).unwrap();
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
    writer.finish().unwrap();
    path
}

/// Run `scx query --count` and return (stderr, stdout).
fn query_count(path: &std::path::Path, filter: &str) -> (String, String) {
    let out = scx_cli()
        .args([
            "query",
            path.to_str().unwrap(),
            "--filter",
            filter,
            "--count",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "query {filter:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    (
        String::from_utf8_lossy(&out.stderr).into_owned(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

/// E1: a `0` from a mistyped category and a `0` from an empty slice used to look
/// identical, which is the single most likely way to silently mis-slice an atlas.
#[test]
fn test_query_empty_match_names_the_missing_category() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_categorical_test_file(&dir, "cats.scx");

    // (a) Outside the vocabulary → a note naming the value, the column, the
    // vocabulary size, and the near-match.
    let (stderr, stdout) = query_count(&path, "cell_type == 'B cel'");
    assert_eq!(stdout.trim(), "0");
    assert!(
        stderr.contains("no category"),
        "a mistyped category must be called out, got: {stderr}"
    );
    for needle in ["B cel", "cell_type", "3 known categories", "B cell"] {
        assert!(
            stderr.contains(needle),
            "the note must mention {needle:?}, got: {stderr}"
        );
    }

    // (b) Declared but unused → a GENUINE zero. Silence here is the whole point:
    // the note must mean "you mistyped", not "your query returned nothing".
    let (stderr, stdout) = query_count(&path, "cell_type == 'monocyte'");
    assert_eq!(stdout.trim(), "0");
    assert!(
        !stderr.contains("no category"),
        "'monocyte' IS a declared category with zero rows — a genuine empty \
         result, which must not be reported as a typo: {stderr}"
    );

    // (c) A non-empty result must never carry the note.
    let (stderr, stdout) = query_count(&path, "cell_type == 'B cell'");
    assert_eq!(stdout.trim(), "3");
    assert!(!stderr.contains("no category"), "got: {stderr}");
}

/// The note is not gated behind `--explain`: a bare `--count` prints the same
/// misleading `0`, and a user with no reason to suspect a typo has no reason to
/// reach for a flag. It must also survive alongside `--explain` and `--json`.
#[test]
fn test_query_category_note_is_not_gated_behind_explain() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_categorical_test_file(&dir, "cats_flags.scx");

    for extra in [
        vec![],
        vec!["--explain"],
        vec!["--json"],
        vec!["--explain", "--json"],
    ] {
        let mut args = vec![
            "query",
            path.to_str().unwrap(),
            "--filter",
            "cell_type == 'B cel'",
            "--count",
        ];
        args.extend(extra.iter().copied());
        let out = scx_cli().args(&args).output().unwrap();
        assert!(out.status.success());
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("no category"),
            "the note must appear with flags {extra:?}, got: {stderr}"
        );
    }
}

/// The collect path (no `--count`) reports it too — `collect()` consumes the
/// pipeline, so this arm re-opens the source, and that plumbing is easy to get
/// wrong in only one of the two paths.
#[test]
fn test_query_category_note_on_the_collect_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_categorical_test_file(&dir, "cats_collect.scx");
    let out_path = dir.path().join("empty_out.scx");

    let out = scx_cli()
        .args([
            "query",
            path.to_str().unwrap(),
            "--filter",
            "cell_type == 'B cel'",
            "--output",
            out_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "an empty query must still succeed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no category") && stderr.contains("B cell"),
        "the collect path must carry the same note, got: {stderr}"
    );
}

/// No predicate at all: an empty file is not a mis-slice, and there is no
/// literal to have mistyped. Must stay silent rather than emit a bare note.
#[test]
fn test_query_no_filter_gets_no_category_note() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_categorical_test_file(&dir, "cats_nofilter.scx");

    let out = scx_cli()
        .args(["query", path.to_str().unwrap(), "--count"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("no category"), "got: {stderr}");
}

/// `build-csc` keeps the predicate index **sections** *and* the **pushdown** they
/// enable. Both halves are pinned, because a doc asserting either one alone is
/// wrong — and the operations matrix asserted one of them until this test existed.
///
/// **History, because the shape of the mistake is the reusable part.** The matrix
/// first said "Dropped", which contradicted the code: `copy_auxiliary_sections` →
/// `copy_predicate_indices` copies both sections verbatim and the copies stay
/// *valid*, since shard boundaries and row ranges are unchanged. "Preserved" was
/// just as wrong from the user's side, though: `run_build_csc` re-encodes every
/// shard through `write_csr_shard`, `compute_shard_stats` emits no `column_stats`,
/// and Level-1 pruning resolves a categorical predicate against those stats'
/// `CategoryBitset`. So the section was there and the pruning was gone — a full
/// scan returning the right rows, which is why nothing caught it.
///
/// That gap was pinned here rather than fixed, with a note saying the fix was to
/// wire the stats into the re-emit as `merge` does. Phase 5c did that, and found
/// `optimize` — the only other op declaring `Carry::Verbatim` for this family —
/// had the identical defect, unpinned and invisible because its own test has a
/// single CSR shard. See
/// `scx-ops::optimize::tests::optimize_preserves_level1_shard_pruning`.
///
/// The generalisable rule: **`Carry::Verbatim` on a predicate index is satisfied
/// by copying bytes, and the carry audit cannot see that the statistics those
/// bytes need went missing.** Any future op declaring it owes this assertion too.
#[test]
fn build_csc_carries_predicate_index_and_pushdown() {
    let dir = tempfile::tempdir().unwrap();
    // `write_clustered_test_file` gives obs a `cell_type` / `disease` to index.
    let input = write_clustered_test_file(&dir, "indexed_for_csc.scx", 200, 10, 50);

    // Build an indexed source by subsetting with --index-preset (F7's own path).
    let indexed = dir.path().join("indexed.scx");
    let out = scx_cli()
        .args([
            "subset",
            input.to_str().unwrap(),
            indexed.to_str().unwrap(),
            "--filter",
            "disease == 'normal'",
            // Small enough to force several output shards: Level-1 pruning is
            // unobservable on a single shard, so a default shard size would make
            // the pushdown assertion below vacuous.
            "--shard-size",
            "25",
            "--index-obs",
            "cell_type",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "fixture setup failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let before = ScxReader::open(&indexed)
        .unwrap()
        .read_obs_predicate_index_bytes()
        .unwrap()
        .map(<[u8]>::to_vec);
    assert!(
        before.is_some(),
        "fixture precondition: the input must carry an obs predicate index"
    );

    // Copy-out form.
    let copied = dir.path().join("csc_copy.scx");
    let out = scx_cli()
        .args([
            "build-csc",
            indexed.to_str().unwrap(),
            copied.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // In-place form.
    let in_place = dir.path().join("csc_in_place.scx");
    std::fs::copy(&indexed, &in_place).unwrap();
    let out = scx_cli()
        .args(["build-csc", in_place.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    for (label, path) in [("copy-out", &copied), ("in-place", &in_place)] {
        let reader = ScxReader::open(path).unwrap();
        assert!(
            reader.header().has_csc(),
            "{label}: the sidecar must have been built"
        );
        let after = reader
            .read_obs_predicate_index_bytes()
            .unwrap()
            .map(<[u8]>::to_vec);
        assert_eq!(
            after, before,
            "{label} build-csc must carry the obs predicate index through \
             byte-for-byte; docs/operations.md's build-csc row states this"
        );
    }

    // The other half: pruning. Section bytes surviving is necessary but not
    // sufficient — and here it is not sufficient, which is the pre-existing gap.
    let pruned = |path: &std::path::Path| -> (u32, String) {
        let out = scx_cli()
            .args([
                "query",
                path.to_str().unwrap(),
                "--filter",
                "cell_type == 'T cell'",
                "--count",
                "--explain",
            ])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        let n: u32 = stderr
            .lines()
            .find_map(|l| {
                l.trim()
                    .strip_prefix("Level 1 (catalog-stats) eliminated: ")
            })
            .and_then(|v| v.split_once('/'))
            .map(|(n, _)| n.parse().unwrap())
            .unwrap_or_else(|| panic!("no Level-1 line:\n{stderr}"));
        let matched = stderr
            .lines()
            .find_map(|l| l.trim().strip_prefix("matched rows: "))
            .unwrap_or("?")
            .to_string();
        (n, matched)
    };

    let (before_elim, before_matched) = pruned(&indexed);
    assert!(
        before_elim > 0,
        "fixture precondition: the indexed input must prune, or the comparison \
         below proves nothing (got {before_elim})"
    );
    let (after_elim, after_matched) = pruned(&copied);
    assert_eq!(
        after_elim, before_elim,
        "build-csc re-encodes every CSR shard, so it must re-derive the per-shard \
         column stats Level-1 pruning reads from the index it carried. Carrying \
         the section bytes alone leaves pruning off."
    );
    assert_eq!(
        before_matched, after_matched,
        "which rows match must not change either way"
    );

    // The in-place form never touches the CSR shards, so its stats were never
    // lost — assert that rather than assuming it, since the two forms take
    // different paths to the same catalog.
    let (in_place_elim, in_place_matched) = pruned(&in_place);
    assert_eq!(
        in_place_elim, before_elim,
        "in-place build-csc leaves the CSR shards alone, so their column stats \
         must survive untouched"
    );
    assert_eq!(in_place_matched, before_matched);
}

/// §11.12: `scx query --count --output out.scx` used to exit 0, print the
/// count, and silently discard `--output` — the count branch returns before
/// the write, and nothing declared the two mutually exclusive.
///
/// The reject side plus, below, the two accept sides. Without those the guard
/// would pass equally well if it rejected `--output` outright, and the flag
/// would be broken in the other direction with nothing to say so.
#[test]
fn query_count_and_output_are_mutually_exclusive() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "conflict.scx", 12, 10);
    let out = dir.path().join("never_written.scx");

    let output = scx_cli()
        .args([
            "query",
            path.to_str().unwrap(),
            "cell_type == 'T cell'",
            "--count",
            "--output",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "--count --output must be refused, not silently ignored"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("cannot be used with"),
        "expected a clap conflict message naming both flags, got: {stderr}"
    );
    assert!(
        !out.exists(),
        "the refused invocation created an output file"
    );
}

/// Accept side 1: `--output` alone still writes.
#[test]
fn query_output_without_count_still_writes() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "out_ok.scx", 12, 10);
    let out = dir.path().join("written.scx");

    let output = scx_cli()
        .args([
            "query",
            path.to_str().unwrap(),
            "cell_type == 'T cell'",
            "--output",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "query --output failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(out.exists(), "--output alone must still write the file");
}

// ---------------------------------------------------------------------------
// Destination overwrite protection
//
// `ScxWriter::finish` persists with `rename(2)`, which replaces
// unconditionally, so a subcommand with no existence check does not "overwrite
// on request" — it destroys the file silently. `optimize` / `compact` / `sort`
// / `build-csc` demanded `--force`; `convert`, `merge`, `subset`,
// `query --output`, `upgrade` and the cloud ops did not.
// ---------------------------------------------------------------------------

const SENTINEL: &[u8] = b"do not clobber me";

/// Assert that `args` refuses to write over `dest` without `--force`, leaves it
/// byte-identical, and succeeds once `--force` is added.
///
/// The byte-identity half is the point. An exit-code-only assertion passes just
/// as happily against a guard that fires *after* truncating the destination,
/// which is the worse failure of the two.
fn assert_force_protects(args: &[&str], dest: &std::path::Path) {
    let before = std::fs::read(dest).expect("destination must exist before the refusal");

    let out = scx_cli().args(args).output().unwrap();
    assert!(
        !out.status.success(),
        "expected a refusal for {args:?}, got success"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("already exists") && stderr.contains("--force"),
        "the refusal must name the collision and the flag: {stderr}"
    );
    assert_eq!(
        std::fs::read(dest).unwrap(),
        before,
        "a refused invocation must leave {} untouched",
        dest.display()
    );

    let mut forced: Vec<&str> = args.to_vec();
    forced.push("--force");
    let out = scx_cli().args(&forced).output().unwrap();
    assert!(
        out.status.success(),
        "--force should have been accepted for {forced:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn merge_refuses_to_clobber_its_output() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_test_file(&dir, "merge_a.scx", 6, 10);
    let b = write_test_file(&dir, "merge_b.scx", 5, 10);
    let dest = dir.path().join("merged.scx");
    std::fs::write(&dest, SENTINEL).unwrap();

    assert_force_protects(
        &[
            "merge",
            a.to_str().unwrap(),
            b.to_str().unwrap(),
            "--output",
            dest.to_str().unwrap(),
        ],
        &dest,
    );
    ScxReader::open(&dest).expect("the forced merge must have written a readable file");
}

/// `merge` is the one command with several inputs, and writing onto *any* of
/// them is the same data loss as writing onto the single input `compact` has.
#[test]
fn merge_refuses_to_write_onto_one_of_its_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_test_file(&dir, "merge_in_a.scx", 6, 10);
    let b = write_test_file(&dir, "merge_in_b.scx", 5, 10);
    let before = std::fs::read(&b).unwrap();

    let out = scx_cli()
        .args([
            "merge",
            a.to_str().unwrap(),
            b.to_str().unwrap(),
            "--output",
            b.to_str().unwrap(),
            "--force",
        ])
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "writing onto an input must be refused"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("must be different files"),
        "the refusal must say why: {stderr}"
    );
    assert_eq!(std::fs::read(&b).unwrap(), before, "input b must survive");
}

#[test]
fn subset_refuses_to_clobber_its_output() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file(&dir, "subset_src.scx", 9, 10);
    let dest = dir.path().join("subset_out.scx");
    std::fs::write(&dest, SENTINEL).unwrap();

    assert_force_protects(
        &[
            "subset",
            input.to_str().unwrap(),
            dest.to_str().unwrap(),
            "--filter",
            "cell_type == 'T cell'",
        ],
        &dest,
    );
    ScxReader::open(&dest).expect("the forced subset must have written a readable file");
}

/// `--dry-run` writes nothing, so an existing output is not a collision there
/// and must not be turned into one.
#[test]
fn subset_dry_run_ignores_an_existing_output() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file(&dir, "subset_dry_src.scx", 9, 10);
    let dest = dir.path().join("subset_dry_out.scx");
    std::fs::write(&dest, SENTINEL).unwrap();

    let out = scx_cli()
        .args([
            "subset",
            input.to_str().unwrap(),
            dest.to_str().unwrap(),
            "--filter",
            "cell_type == 'T cell'",
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "--dry-run must not trip the overwrite guard: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(std::fs::read(&dest).unwrap(), SENTINEL);
}

#[test]
fn query_output_refuses_to_clobber() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file(&dir, "query_src.scx", 9, 10);
    let dest = dir.path().join("query_out.scx");
    std::fs::write(&dest, SENTINEL).unwrap();

    assert_force_protects(
        &[
            "query",
            input.to_str().unwrap(),
            "--filter",
            "cell_type == 'T cell'",
            "--output",
            dest.to_str().unwrap(),
        ],
        &dest,
    );
    ScxReader::open(&dest).expect("the forced query must have written a readable file");
}

#[test]
fn upgrade_refuses_to_clobber_its_output() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file(&dir, "upgrade_src.scx", 6, 10);
    let dest = dir.path().join("upgrade_out.scx");
    std::fs::write(&dest, SENTINEL).unwrap();

    assert_force_protects(
        &["upgrade", input.to_str().unwrap(), dest.to_str().unwrap()],
        &dest,
    );
}

/// The unlisted data-loss defect this PR closes.
///
/// `compact` unlinked the output when `--force` was set and *then* read
/// `metadata(input)`. With `input == output` that deleted the input and failed
/// — the file was gone, and the command reported an error about it being
/// missing. `sort` and `build-csc` already carried an explicit same-path guard
/// with a comment naming exactly this hazard; `compact` was the third instance.
#[test]
fn compact_refuses_to_write_onto_its_own_input() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file(&dir, "compact_self.scx", 8, 10);
    let before = std::fs::read(&input).unwrap();

    let out = scx_cli()
        .args([
            "compact",
            input.to_str().unwrap(),
            input.to_str().unwrap(),
            "--force",
        ])
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "compacting a file onto itself must be refused, not attempted"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("must be different files"),
        "the refusal must say why: {stderr}"
    );
    assert_eq!(
        std::fs::read(&input).unwrap(),
        before,
        "the input must survive the refusal — the defect was that it did not"
    );
    ScxReader::open(&input).expect("and must still be readable");
}

/// `--force` is a question about a destination, so a form that writes none
/// must refuse it rather than accept it as a no-op.
///
/// A silently inert flag is the reason this is worth a test: `scx
/// modify-metadata --index-*` without `--obs`/`--var` used to exit 0, print
/// "Updated metadata…", and build nothing. `scx build-csc`'s in-place arm
/// already stated the rule; these are the other forms it applies to.
#[test]
fn force_is_refused_where_nothing_is_written() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file(&dir, "inert_force.scx", 6, 10);
    let i = input.to_str().unwrap();
    let out = dir.path().join("never_written.scx");

    let cases: Vec<Vec<&str>> = vec![
        // A query that prints rather than writes.
        vec!["query", i, "--filter", "cell_type == 'T cell'", "--force"],
        vec!["query", i, "--count", "--force"],
        // A dry run.
        vec![
            "subset",
            i,
            out.to_str().unwrap(),
            "--filter",
            "cell_type == 'T cell'",
            "--dry-run",
            "--force",
        ],
        // An in-place upgrade.
        vec!["upgrade", i, "--in-place", "--force"],
    ];

    for args in cases {
        let res = scx_cli().args(&args).output().unwrap();
        assert!(
            !res.status.success(),
            "expected a refusal for {args:?}, got success"
        );
        let stderr = String::from_utf8_lossy(&res.stderr);
        assert!(
            stderr.contains("--force applies only when writing to an output"),
            "{args:?}: {stderr}"
        );
    }
    assert!(!out.exists(), "no refused invocation may create a file");
}

/// `--in-place` writes a temp beside `<INPUT>` and renames over it, so a
/// positional `<OUTPUT>` passed alongside was silently ignored — the file the
/// user named was never written. Worse once `--force` existed: the guard would
/// demand `--force` for an output the command was never going to touch.
#[test]
fn upgrade_refuses_in_place_together_with_an_output_path() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_test_file(&dir, "upgrade_both.scx", 6, 10);
    let out = dir.path().join("ignored.scx");

    let res = scx_cli()
        .args([
            "upgrade",
            input.to_str().unwrap(),
            out.to_str().unwrap(),
            "--in-place",
        ])
        .output()
        .unwrap();
    assert!(!res.status.success(), "the combination must be refused");
    let stderr = String::from_utf8_lossy(&res.stderr);
    assert!(
        stderr.contains("--in-place") && stderr.contains("<OUTPUT>"),
        "the refusal must name both: {stderr}"
    );
    assert!(!out.exists(), "nothing should have been written");
}

/// `scx explode --force` must *replace* a non-empty `.scxd`, not overlay it.
///
/// `scx_cloud::explode` writes the sections the current catalog names and
/// leaves every other file alone, so a previous explode's extra shards would
/// ride along. That is invisible to `pack` / `info` / `query`, which read the
/// catalog, but not to a directory sync of the `.scxd`.
#[cfg(feature = "cloud")]
#[test]
fn explode_force_replaces_a_non_empty_directory() {
    let dir = tempfile::tempdir().unwrap();
    let big = write_test_file(&dir, "explode_big.scx", 12, 10);
    let out = dir.path().join("out.scxd");

    assert!(scx_cli()
        .args(["explode", big.to_str().unwrap(), out.to_str().unwrap()])
        .output()
        .unwrap()
        .status
        .success());
    // A file from a previous explode that the next one will not name.
    let orphan = out.join("X_shard_999.bin");
    std::fs::write(&orphan, b"left over from a previous explode").unwrap();

    // Without --force the non-empty directory is refused outright.
    let refused = scx_cli()
        .args(["explode", big.to_str().unwrap(), out.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!refused.status.success());
    assert!(orphan.exists(), "a refused explode must change nothing");

    let forced = scx_cli()
        .args([
            "explode",
            big.to_str().unwrap(),
            out.to_str().unwrap(),
            "--force",
        ])
        .output()
        .unwrap();
    assert!(
        forced.status.success(),
        "explode --force: {}",
        String::from_utf8_lossy(&forced.stderr)
    );
    assert!(
        !orphan.exists(),
        "--force must replace the directory, not overlay it"
    );
    assert!(
        out.join("header.bin").exists() || std::fs::read_dir(&out).unwrap().count() > 0,
        "the replacement explode must have populated the directory"
    );
}

/// `explode --force` replaces the destination tree wholesale, so an input
/// living inside it would be deleted along with everything else. The guard's
/// same-path check compares a file against a directory and cannot see this.
#[cfg(feature = "cloud")]
#[test]
fn explode_force_refuses_an_input_inside_the_destination() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("dest.scxd");
    std::fs::create_dir_all(&dest).unwrap();
    // Make it non-empty so `--force` takes the replacing path.
    std::fs::write(dest.join("stale.bin"), b"previous explode").unwrap();
    // …and put the source inside it.
    let input = write_test_file(&dir, "inside.scx", 6, 10);
    let inside = dest.join("inside.scx");
    std::fs::rename(&input, &inside).unwrap();
    let original = std::fs::read(&inside).unwrap();

    let out = scx_cli()
        .args([
            "explode",
            inside.to_str().unwrap(),
            dest.to_str().unwrap(),
            "--force",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success(), "the containment must be refused");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("is inside the output"),
        "the refusal must say why: {stderr}"
    );
    assert_eq!(
        std::fs::read(&inside).unwrap(),
        original,
        "the input must survive the refusal"
    );
}

/// Equality is not the whole "never overwrite an input" invariant.
///
/// A *directory* input and a file destination inside it compare unequal, so
/// the same-path check missed them entirely: `scx convert --from mtx dir
/// dir/matrix.mtx.gz --force` exited 0 and replaced the source matrix with an
/// SCX file. The mirror case — an input inside a directory destination — is
/// what the explode swap would delete.
#[test]
fn a_destination_inside_a_directory_input_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("mtx_in");
    std::fs::create_dir_all(&mtx_dir).unwrap();
    std::fs::write(
        mtx_dir.join("matrix.mtx"),
        "%%MatrixMarket matrix coordinate integer general\n2 2 1\n1 1 3\n",
    )
    .unwrap();
    std::fs::write(mtx_dir.join("barcodes.tsv"), "AAAC-1\nBBBC-1\n").unwrap();
    std::fs::write(
        mtx_dir.join("features.tsv"),
        "G1\tA\tGene Expression\nG2\tB\tGene Expression\n",
    )
    .unwrap();

    let dest = mtx_dir.join("matrix.mtx.gz");
    std::fs::write(&dest, b"a member of the source directory").unwrap();
    let before = std::fs::read(&dest).unwrap();

    let out = scx_cli()
        .args([
            "convert",
            "--from",
            "mtx",
            mtx_dir.to_str().unwrap(),
            dest.to_str().unwrap(),
            "--force",
        ])
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "writing into the source directory must be refused"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("is inside the input"),
        "the refusal must say why: {stderr}"
    );
    assert_eq!(
        std::fs::read(&dest).unwrap(),
        before,
        "the source member must survive the refusal"
    );
}
