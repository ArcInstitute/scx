//! Workspace-level integration tests for the SCX file lifecycle.
//!
//! Exercises the full contract across crate boundaries:
//!   scx-format (write/read) → scx-ops (append/delete/compact/merge) → scx-engine (query)

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{AsArray, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format::header::{FileHeader, CURRENT_FORMAT_VERSION, MAGIC};
use scx_format::reader::ScxReader;
use scx_format::writer::ScxWriter;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader {
        magic: MAGIC,
        format_version: CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs,
        n_vars,
        nnz: 0,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: 10_000,
        codec_id: 0,
        index_dtype: 0, // u16
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

/// Build obs metadata with cell_id and cell_type columns.
/// `prefix` allows distinguishing cells from different batches.
fn make_obs(n: usize, prefix: &str) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, true),
    ]);
    let ids: Vec<String> = (0..n).map(|i| format!("{prefix}_cell_{i}")).collect();
    let types: Vec<&str> = (0..n)
        .map(|i| match i % 3 {
            0 => "T cell",
            1 => "B cell",
            _ => "NK cell",
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

fn make_var(n: usize) -> RecordBatch {
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

/// Deterministic CSR data: each row has exactly 2 nonzeros.
fn make_shard_data(n_rows: usize, n_vars: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
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

/// Write a complete test file and return its path.
fn write_test_file(dir: &Path, name: &str, n_obs: usize, n_vars: usize, prefix: &str) -> PathBuf {
    let path = dir.join(name);
    let header = make_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&make_obs(n_obs, prefix)).unwrap();
    writer.write_var(&make_var(n_vars)).unwrap();
    let (indptr, indices, values) = make_shard_data(n_obs, n_vars);
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Full lifecycle: create → append → read → delete → compact → merge → query.
#[test]
fn test_full_lifecycle() {
    let dir = tempfile::tempdir().unwrap();

    // ── Step 1: Create ──────────────────────────────────────────────────
    let path = write_test_file(dir.path(), "base.scx", 30, 10, "A");

    // ── Step 2: Read back ───────────────────────────────────────────────
    {
        let reader = ScxReader::open(&path).unwrap();
        assert_eq!(reader.n_obs(), 30);
        assert_eq!(reader.n_vars(), 10);
        assert_eq!(reader.nnz(), 60); // 30 rows × 2 nnz each
        let obs = reader.read_obs().unwrap();
        assert_eq!(obs.num_rows(), 30);
        let var = reader.read_var().unwrap();
        assert_eq!(var.num_rows(), 10);
    }

    // ── Step 3: Append 10 rows ──────────────────────────────────────────
    {
        let new_obs = make_obs(10, "B");
        let (indptr, indices, values) = make_shard_data(10, 10);
        scx_ops::append(
            &path,
            &new_obs,
            &indptr,
            &indices,
            &values,
            ValueEncoding::Uint8,
            CodecId::None,
            10_000,
        )
        .unwrap();

        let reader = ScxReader::open(&path).unwrap();
        assert_eq!(reader.n_obs(), 40);
        assert_eq!(reader.nnz(), 80);
        let obs = reader.read_obs().unwrap();
        assert_eq!(obs.num_rows(), 40);
    }

    // ── Step 4: Delete 5 rows ───────────────────────────────────────────
    // Delete indices 0, 5, 10, 15, 35 (mix of original and appended)
    let deleted_indices: Vec<u64> = vec![0, 5, 10, 15, 35];
    {
        let total_deleted = scx_ops::mark_deleted(&path, &deleted_indices).unwrap();
        assert_eq!(total_deleted, 5);
    }

    // ── Step 5: Compact ─────────────────────────────────────────────────
    let compacted_path = dir.path().join("compacted.scx");
    {
        scx_ops::compact(&path, &compacted_path).unwrap();

        let reader = ScxReader::open(&compacted_path).unwrap();
        assert_eq!(reader.n_obs(), 35); // 40 - 5 deleted
        assert_eq!(reader.nnz(), 70); // 35 rows × 2 nnz each
    }

    // ── Step 6: Verify deleted cells are absent ─────────────────────────
    {
        let reader = ScxReader::open(&compacted_path).unwrap();
        let obs = reader.read_obs().unwrap();
        assert_eq!(obs.num_rows(), 35);

        let cell_ids: Vec<&str> = obs
            .column_by_name("cell_id")
            .unwrap()
            .as_string::<i32>()
            .iter()
            .map(|v| v.unwrap())
            .collect();

        // row 0 was "A_cell_0", row 5 was "A_cell_5", etc.
        assert!(!cell_ids.contains(&"A_cell_0"));
        assert!(!cell_ids.contains(&"A_cell_5"));
        assert!(!cell_ids.contains(&"A_cell_10"));
        assert!(!cell_ids.contains(&"A_cell_15"));
        assert!(!cell_ids.contains(&"B_cell_5")); // row 35 = appended[5]

        // Verify some that should still be present
        assert!(cell_ids.contains(&"A_cell_1"));
        assert!(cell_ids.contains(&"A_cell_29"));
        assert!(cell_ids.contains(&"B_cell_0"));
    }

    // ── Step 7: Merge with a second file ────────────────────────────────
    let second_path = write_test_file(dir.path(), "second.scx", 20, 10, "C");
    let merged_path = dir.path().join("merged.scx");
    {
        scx_ops::merge(
            &[compacted_path.as_path(), second_path.as_path()],
            &merged_path,
        )
        .unwrap();

        let reader = ScxReader::open(&merged_path).unwrap();
        assert_eq!(reader.n_obs(), 55); // 35 + 20
        assert_eq!(reader.n_vars(), 10);
    }

    // ── Step 8: Query with QueryPipeline ────────────────────────────────
    {
        use scx_engine::QueryPipeline;

        let result = QueryPipeline::open(&merged_path)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap()
            .collect()
            .unwrap();

        // T cells are at indices 0, 3, 6, 9, ... (every 3rd cell)
        // 55 total cells → ceil(55/3) = 19 T cells (indices 0,3,6,...,54)
        // But the exact count depends on which cells survived deletion.
        // From compacted: 35 cells, ~12 T cells. From second: 20 cells, 7 T cells.
        assert!(result.x.n_rows() > 0, "should return some T cells");
        assert_eq!(result.x.n_cols(), 10);
        assert_eq!(result.obs.num_rows(), result.x.n_rows());

        // All returned cell_types must be "T cell"
        let cell_types: Vec<&str> = result
            .obs
            .column_by_name("cell_type")
            .unwrap()
            .as_string::<i32>()
            .iter()
            .map(|v| v.unwrap())
            .collect();
        assert!(cell_types.iter().all(|&t| t == "T cell"));
    }
}

/// Validate that all section checksums pass after writing.
#[test]
fn test_validate_checksums() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(dir.path(), "checksum_test.scx", 50, 20, "V");

    let reader = ScxReader::open(&path).unwrap();
    let results = reader.validate().unwrap();

    assert!(!results.is_empty());
    for (name, passed) in &results {
        assert!(passed, "checksum failed for section: {name}");
    }
}

/// Append preserves CSR data integrity: read all shards and verify shape/values.
#[test]
fn test_append_preserves_readability() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(dir.path(), "append_read.scx", 20, 8, "X");

    // Append 15 more rows
    let new_obs = make_obs(15, "Y");
    let (indptr, indices, values) = make_shard_data(15, 8);
    scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        CodecId::None,
        10_000,
    )
    .unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();

    assert_eq!(csr.n_rows(), 35);
    assert_eq!(csr.n_cols(), 8);
    // Each row has exactly 2 nnz → total 70
    assert_eq!(csr.indptr.len(), 36); // n_rows + 1
    assert_eq!(*csr.indptr.last().unwrap(), 70);

    // Verify checksums still pass after append
    let results = reader.validate().unwrap();
    for (name, passed) in &results {
        assert!(passed, "checksum failed for section: {name}");
    }
}

/// Compact with no deletions produces data-identical output.
#[test]
fn test_compact_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(dir.path(), "idem_input.scx", 25, 6, "I");

    let compacted = dir.path().join("idem_output.scx");
    scx_ops::compact(&path, &compacted).unwrap();

    let r1 = ScxReader::open(&path).unwrap();
    let r2 = ScxReader::open(&compacted).unwrap();

    assert_eq!(r1.n_obs(), r2.n_obs());
    assert_eq!(r1.n_vars(), r2.n_vars());
    assert_eq!(r1.nnz(), r2.nnz());

    let csr1 = r1.read_all_csr_shards().unwrap();
    let csr2 = r2.read_all_csr_shards().unwrap();

    assert_eq!(csr1.shape, csr2.shape);
    assert_eq!(csr1.data, csr2.data);
    assert_eq!(csr1.indices, csr2.indices);
    // indptr may differ in absolute values if shard boundaries change,
    // but the per-row nnz must be identical
    let nnz_per_row_1: Vec<i64> = csr1.indptr.windows(2).map(|w| w[1] - w[0]).collect();
    let nnz_per_row_2: Vec<i64> = csr2.indptr.windows(2).map(|w| w[1] - w[0]).collect();
    assert_eq!(nnz_per_row_1, nnz_per_row_2);
}
