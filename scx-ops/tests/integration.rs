use arrow::array::{Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format::header::{FileHeader, MAGIC};
use scx_format::provenance::ProvenanceEntry;
use scx_format::writer::ScxWriter;
use scx_format::{ScxReader, ShardHeader, SHARD_HEADER_SIZE};
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn sample_header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader {
        magic: MAGIC,
        format_version: 1,
        header_length: 256,
        flags: 0,
        n_obs,
        n_vars,
        nnz: 0,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: 16384,
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
        reserved: [0u8; 132],
    }
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
    dir: &TempDir,
    filename: &str,
    n_obs: usize,
    n_vars: usize,
    n_shards: usize,
) -> PathBuf {
    let path = dir.path().join(filename);
    let header = sample_header(n_obs as u64, n_vars as u64);
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

    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1710000000,
            action: "convert".to_string(),
            tool: "test".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();

    writer.finish().unwrap();
    path
}

// ---------------------------------------------------------------------------
// Append tests
// ---------------------------------------------------------------------------

#[test]
fn test_append_read_back_all_cells() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "append.scx", 6, 10, 1);

    // Append 4 more rows
    let new_obs = sample_obs(4);
    let (indptr, indices, values) = sample_shard_data(4, 10);

    scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        CodecId::None,
        16384,
    )
    .unwrap();

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), 10);
    assert_eq!(reader.header().n_csr_shards, 2);

    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 10);

    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 10);
    assert_eq!(csr.nnz(), 20); // 10 rows * 2 nnz each
}

#[test]
fn test_append_manifest_sequence_increments() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "seq.scx", 6, 10, 1);

    let reader = ScxReader::open(&path).unwrap();
    let initial_seq = reader.header().manifest_sequence;
    drop(reader);

    let new_obs = sample_obs(2);
    let (indptr, indices, values) = sample_shard_data(2, 10);
    scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        CodecId::None,
        16384,
    )
    .unwrap();

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.header().manifest_sequence, initial_seq + 1);
}

// ---------------------------------------------------------------------------
// Delete tests
// ---------------------------------------------------------------------------

#[test]
fn test_delete_marks_cells() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "delete.scx", 6, 10, 1);

    let total_deleted = scx_ops::mark_deleted(&path, &[0, 3, 5]).unwrap();
    assert_eq!(total_deleted, 3);

    let reader = ScxReader::open(&path).unwrap();
    assert!(reader.header().has_deletion_vectors());

    let dv = reader.read_deletion_vectors().unwrap().unwrap();
    assert_eq!(dv.total_deleted(), 3);
    assert!(dv.is_deleted(0, 0));
    assert!(dv.is_deleted(0, 3));
    assert!(dv.is_deleted(0, 5));
    assert!(!dv.is_deleted(0, 1));
}

// ---------------------------------------------------------------------------
// Delete + Compact tests
// ---------------------------------------------------------------------------

#[test]
fn test_delete_then_compact() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "dc.scx", 6, 10, 1);

    // Delete 2 cells
    scx_ops::mark_deleted(&path, &[1, 4]).unwrap();

    // Compact
    let compact_path = dir.path().join("dc_compact.scx");
    scx_ops::compact(&path, &compact_path).unwrap();

    let reader = ScxReader::open(&compact_path).unwrap();
    assert_eq!(reader.n_obs(), 4); // 6 - 2
    assert!(!reader.header().has_deletion_vectors());

    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 4);

    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 4);
    assert_eq!(csr.nnz(), 8); // 4 rows * 2 nnz
}

// ---------------------------------------------------------------------------
// Compact without deletions
// ---------------------------------------------------------------------------

#[test]
fn test_compact_after_appends() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "multi.scx", 4, 10, 1);

    // Append 3 times
    for _ in 0..3 {
        let new_obs = sample_obs(2);
        let (indptr, indices, values) = sample_shard_data(2, 10);
        scx_ops::append(
            &path,
            &new_obs,
            &indptr,
            &indices,
            &values,
            ValueEncoding::Uint8,
            CodecId::None,
            16384,
        )
        .unwrap();
    }

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), 10);
    let original_size = std::fs::metadata(&path).unwrap().len();
    drop(reader);

    // Compact
    let compact_path = dir.path().join("compacted.scx");
    scx_ops::compact(&path, &compact_path).unwrap();

    let reader = ScxReader::open(&compact_path).unwrap();
    assert_eq!(reader.n_obs(), 10);
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.nnz(), 20);

    // Compacted file should be smaller (no stale catalogs/obs sections)
    let compact_size = std::fs::metadata(&compact_path).unwrap().len();
    assert!(
        compact_size < original_size,
        "compact ({compact_size}) should be < original ({original_size})"
    );
}

// ---------------------------------------------------------------------------
// Rollback tests
// ---------------------------------------------------------------------------

#[test]
fn test_rollback_after_append() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "rollback.scx", 6, 10, 1);

    // Record original state
    let reader = ScxReader::open(&path).unwrap();
    let original_n_obs = reader.n_obs();
    let original_seq = reader.header().manifest_sequence;
    drop(reader);

    // Append
    let new_obs = sample_obs(4);
    let (indptr, indices, values) = sample_shard_data(4, 10);
    scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        CodecId::None,
        16384,
    )
    .unwrap();

    // Verify append worked
    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), 10);
    drop(reader);

    // Rollback
    scx_ops::rollback(&path).unwrap();

    // Verify original state restored
    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), original_n_obs);
    assert_eq!(reader.header().manifest_sequence, original_seq);

    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 6);
}

// ---------------------------------------------------------------------------
// Merge tests
// ---------------------------------------------------------------------------

#[test]
fn test_merge_three_files() {
    let dir = tempfile::tempdir().unwrap();
    let path1 = write_test_file(&dir, "m1.scx", 4, 10, 1);
    let path2 = write_test_file(&dir, "m2.scx", 6, 10, 1);
    let path3 = write_test_file(&dir, "m3.scx", 8, 10, 1);

    let output = dir.path().join("merged.scx");
    scx_ops::merge(
        &[path1.as_path(), path2.as_path(), path3.as_path()],
        &output,
    )
    .unwrap();

    let reader = ScxReader::open(&output).unwrap();
    assert_eq!(reader.n_obs(), 18); // 4 + 6 + 8

    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 18);

    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 18);
    assert_eq!(csr.nnz(), 36); // 18 * 2

    // Provenance should have merge entry
    let prov = reader.read_provenance().unwrap();
    let last = prov.operations.last().unwrap();
    assert_eq!(last.action, "merge");
    assert_eq!(last.input_checksums.len(), 3);
}

// ---------------------------------------------------------------------------
// Flock test
// ---------------------------------------------------------------------------

#[test]
fn test_flock_serializes_appends() {
    use std::sync::{Arc, Barrier};
    use std::thread;

    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "flock.scx", 4, 10, 1);

    let path = Arc::new(path);
    let barrier = Arc::new(Barrier::new(2));

    let handles: Vec<_> = (0..2)
        .map(|_| {
            let path = Arc::clone(&path);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let new_obs = sample_obs(2);
                let (indptr, indices, values) = sample_shard_data(2, 10);
                scx_ops::append(
                    &path,
                    &new_obs,
                    &indptr,
                    &indices,
                    &values,
                    ValueEncoding::Uint8,
                    CodecId::None,
                    16384,
                )
                .unwrap();
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let reader = ScxReader::open(&*path).unwrap();
    assert_eq!(reader.n_obs(), 8); // 4 original + 2 + 2
}

// ---------------------------------------------------------------------------
// New test suite: comprehensive coverage
// ---------------------------------------------------------------------------

/// Append should record provenance entry
#[test]
fn test_append_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "prov_a.scx", 6, 10, 1);

    let new_obs = sample_obs(4);
    let (indptr, indices, values) = sample_shard_data(4, 10);
    scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        CodecId::None,
        16384,
    )
    .unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let prov = reader.read_provenance().unwrap();
    // Should have original "convert" + new "append"
    assert_eq!(prov.operations.len(), 2);
    assert_eq!(prov.operations[0].action, "convert");
    assert_eq!(prov.operations[1].action, "append");
    assert!(prov.operations[1].params_json.contains("n_new_rows"));
}

/// Delete should record provenance, and filtered read should exclude deleted rows
#[test]
fn test_delete_filtered_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "filt.scx", 6, 10, 1);

    scx_ops::mark_deleted(&path, &[0, 3, 5]).unwrap();

    let reader = ScxReader::open(&path).unwrap();

    // Unfiltered read should still have all rows
    let csr_all = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr_all.shape.0, 6);

    // Filtered read should exclude deleted rows
    let csr_filtered = reader.read_all_csr_shards_filtered().unwrap();
    assert_eq!(csr_filtered.shape.0, 3); // 6 - 3 deleted

    // Provenance should have delete entry
    let prov = reader.read_provenance().unwrap();
    let last = prov.operations.last().unwrap();
    assert_eq!(last.action, "delete");
}

/// rollback_to specific sequence
#[test]
fn test_rollback_to_specific_sequence() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "rb_to.scx", 4, 10, 1);

    // Record seq 1
    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.header().manifest_sequence, 1);
    drop(reader);

    // Append twice → seq 2, 3
    for _ in 0..2 {
        let new_obs = sample_obs(2);
        let (indptr, indices, values) = sample_shard_data(2, 10);
        scx_ops::append(
            &path,
            &new_obs,
            &indptr,
            &indices,
            &values,
            ValueEncoding::Uint8,
            CodecId::None,
            16384,
        )
        .unwrap();
    }

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.header().manifest_sequence, 3);
    assert_eq!(reader.n_obs(), 8); // 4 + 2 + 2
    drop(reader);

    // Rollback to seq 1
    scx_ops::rollback_to(&path, 1).unwrap();

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.header().manifest_sequence, 1);
    assert_eq!(reader.n_obs(), 4); // back to original
}

/// Merge with incompatible n_vars should error
#[test]
fn test_merge_incompatible_nvars() {
    let dir = tempfile::tempdir().unwrap();
    let path1 = write_test_file(&dir, "n1.scx", 4, 10, 1);
    let path2 = write_test_file(&dir, "n2.scx", 4, 20, 1);

    let output = dir.path().join("bad_merge.scx");
    let result = scx_ops::merge(&[path1.as_path(), path2.as_path()], &output);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        format!("{err}").contains("incompatible n_vars"),
        "expected IncompatibleVars error, got: {err}"
    );
}

/// Append data larger than shard_target_rows should create multiple shards
#[test]
fn test_append_multi_shard() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "multi.scx", 4, 10, 1);

    // Append 10 rows with shard_target_rows=3 → should create ~4 shards
    let new_obs = sample_obs(10);
    let (indptr, indices, values) = sample_shard_data(10, 10);
    scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        CodecId::None,
        3, // very small target
    )
    .unwrap();

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), 14);
    // Original 1 shard + at least 3 new shards (10/3 = 3.33 → 4 shards)
    assert!(
        reader.header().n_csr_shards >= 4,
        "expected >= 4 shards, got {}",
        reader.header().n_csr_shards
    );

    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 14);
    assert_eq!(csr.nnz(), 28); // 14 rows * 2 nnz
}

/// Delete same cell twice should be idempotent
#[test]
fn test_delete_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "idemp.scx", 6, 10, 1);

    let total1 = scx_ops::mark_deleted(&path, &[0, 3]).unwrap();
    assert_eq!(total1, 2);

    // Delete cell 0 again + new cell 4
    let total2 = scx_ops::mark_deleted(&path, &[0, 4]).unwrap();
    assert_eq!(total2, 3); // 0, 3, 4 — not 4 because 0 was already deleted

    let reader = ScxReader::open(&path).unwrap();
    let csr = reader.read_all_csr_shards_filtered().unwrap();
    assert_eq!(csr.shape.0, 3); // 6 - 3 = 3
}

/// Compact should produce correct nnz in header
#[test]
fn test_compact_correct_nnz() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "nnz.scx", 6, 10, 1);

    // Append → compact → verify nnz is correct
    let new_obs = sample_obs(4);
    let (indptr, indices, values) = sample_shard_data(4, 10);
    scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        CodecId::None,
        16384,
    )
    .unwrap();

    let compact_path = dir.path().join("nnz_compact.scx");
    scx_ops::compact(&path, &compact_path).unwrap();

    let reader = ScxReader::open(&compact_path).unwrap();
    assert_eq!(reader.n_obs(), 10);

    // Verify header nnz matches actual data
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(reader.header().nnz, csr.nnz() as u64);
    assert_eq!(csr.nnz(), 20); // 10 rows * 2 nnz each
}

/// Data integrity: known values survive compact
#[test]
fn test_data_integrity_after_compact() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "integ.scx", 6, 10, 1);

    // Read original values
    let reader = ScxReader::open(&path).unwrap();
    let original_csr = reader.read_all_csr_shards().unwrap();
    drop(reader);

    // Delete row 2 and compact
    scx_ops::mark_deleted(&path, &[2]).unwrap();
    let compact_path = dir.path().join("integ_compact.scx");
    scx_ops::compact(&path, &compact_path).unwrap();

    let reader = ScxReader::open(&compact_path).unwrap();
    let compacted_csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(compacted_csr.shape.0, 5); // 6 - 1

    // Verify remaining rows have correct values
    // Row 0 of compacted = row 0 of original
    let orig_row0_start = original_csr.indptr[0] as usize;
    let orig_row0_end = original_csr.indptr[1] as usize;
    let comp_row0_start = compacted_csr.indptr[0] as usize;
    let comp_row0_end = compacted_csr.indptr[1] as usize;
    assert_eq!(
        &original_csr.data[orig_row0_start..orig_row0_end],
        &compacted_csr.data[comp_row0_start..comp_row0_end]
    );
}

/// Merge preserves obs metadata in correct order
#[test]
fn test_merge_preserves_obs_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path1 = write_test_file(&dir, "mo1.scx", 3, 10, 1);
    let path2 = write_test_file(&dir, "mo2.scx", 4, 10, 1);

    let output = dir.path().join("merged_obs.scx");
    scx_ops::merge(&[path1.as_path(), path2.as_path()], &output).unwrap();

    let reader = ScxReader::open(&output).unwrap();
    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 7); // 3 + 4

    // Verify cell_ids are from both files in order
    use arrow::array::StringArray;
    let cell_ids = obs
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(cell_ids.value(0), "cell_0"); // from file 1
    assert_eq!(cell_ids.value(2), "cell_2"); // from file 1
    assert_eq!(cell_ids.value(3), "cell_0"); // from file 2 (starts over)
    assert_eq!(cell_ids.value(6), "cell_3"); // from file 2
}

// ---------------------------------------------------------------------------
// Phase 2 Step 3 — New tests
// ---------------------------------------------------------------------------

/// Task 6: Crash safety — partial append (no header update) leaves original data intact
#[test]
fn test_append_crash_safety() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "crash.scx", 6, 10, 1);

    // Record original state
    let reader = ScxReader::open(&path).unwrap();
    let original_n_obs = reader.n_obs();
    let original_csr = reader.read_all_csr_shards().unwrap();
    drop(reader);

    // Simulate partial append: write garbage at EOF but DON'T update the header.
    // This simulates a crash after writing new shard data but before the header
    // pwrite commit point (SPEC §3.6.2).
    {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        file.write_all(&[0xDE; 1024]).unwrap();
        file.sync_all().unwrap();
    }

    // Re-open: original data should be intact (reader uses header.full_catalog_offset)
    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), original_n_obs);
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape, original_csr.shape);
    assert_eq!(csr.nnz(), original_csr.nnz());
}

/// Bug 2.1 verification: shard header global_offset should be row index, not byte offset
#[test]
fn test_shard_header_global_offset_is_row_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "goff.scx", 6, 10, 1);

    // Append 4 rows
    let new_obs = sample_obs(4);
    let (indptr, indices, values) = sample_shard_data(4, 10);
    scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        CodecId::None,
        16384,
    )
    .unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let shards = reader.catalog().shards_sorted();
    assert_eq!(shards.len(), 2);

    // First shard: global_offset should be 0 (starts at row 0)
    let s0_bytes = reader.section_bytes(shards[0]).unwrap();
    let sh0 =
        ShardHeader::read_from(&mut std::io::Cursor::new(&s0_bytes[..SHARD_HEADER_SIZE])).unwrap();
    assert_eq!(sh0.global_offset, 0, "first shard should start at row 0");

    // Second shard (appended): global_offset should be 6 (original n_obs)
    let s1_bytes = reader.section_bytes(shards[1]).unwrap();
    let sh1 =
        ShardHeader::read_from(&mut std::io::Cursor::new(&s1_bytes[..SHARD_HEADER_SIZE])).unwrap();
    assert_eq!(sh1.global_offset, 6, "appended shard should start at row 6");
}

/// Bug 2.2 verification: compact produces manifest_sequence=0
#[test]
fn test_compact_manifest_sequence_zero() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "cmseq.scx", 6, 10, 1);

    let compact_path = dir.path().join("cmseq_compact.scx");
    scx_ops::compact(&path, &compact_path).unwrap();

    let reader = ScxReader::open(&compact_path).unwrap();
    assert_eq!(reader.header().manifest_sequence, 0);
}

/// Bug 2.3 verification: merge produces manifest_sequence=0
#[test]
fn test_merge_manifest_sequence_zero() {
    let dir = tempfile::tempdir().unwrap();
    let path1 = write_test_file(&dir, "ms1.scx", 4, 10, 1);
    let path2 = write_test_file(&dir, "ms2.scx", 4, 10, 1);

    let output = dir.path().join("ms_merged.scx");
    scx_ops::merge(&[path1.as_path(), path2.as_path()], &output).unwrap();

    let reader = ScxReader::open(&output).unwrap();
    assert_eq!(reader.header().manifest_sequence, 0);
}

// Helper to write a test file with layers and obsm
fn write_test_file_with_layers_obsm(
    dir: &TempDir,
    filename: &str,
    n_obs: usize,
    n_vars: usize,
) -> PathBuf {
    let path = dir.path().join(filename);
    let header = sample_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    // Write X shard
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

    // Write a layer with Float32 encoding
    let mut layer_values = Vec::new();
    for &v in &values {
        layer_values.extend_from_slice(&(v as f32 * 1.5).to_le_bytes());
    }
    writer
        .write_layer_csr_shard(
            &indptr,
            &indices,
            &layer_values,
            CodecId::None,
            ValueEncoding::Float32,
            0,
            "normalized",
            0,
        )
        .unwrap();

    // Write obsm
    let embedding_data: Vec<f64> = (0..n_obs * 2).map(|i| i as f64 * 0.1).collect();
    let schema = Schema::new(vec![
        Field::new("PC1", DataType::Float64, false),
        Field::new("PC2", DataType::Float64, false),
    ]);
    let batch = arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float64Array::from(embedding_data[..n_obs].to_vec())),
            Arc::new(Float64Array::from(embedding_data[n_obs..].to_vec())),
        ],
    )
    .unwrap();
    writer.write_obsm("X_pca", &batch).unwrap();

    // Write uns
    writer
        .write_uns(&serde_json::json!({"method": "test"}))
        .unwrap();

    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1710000000,
            action: "convert".to_string(),
            tool: "test".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();

    writer.finish().unwrap();
    path
}

/// Task 7: compact preserves layers, obsm, and uns through deletion
#[test]
fn test_compact_preserves_layers_obsm_uns() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file_with_layers_obsm(&dir, "laycomp.scx", 6, 10);

    // Delete 2 cells and compact
    scx_ops::mark_deleted(&path, &[1, 4]).unwrap();
    let compact_path = dir.path().join("laycomp_out.scx");
    scx_ops::compact(&path, &compact_path).unwrap();

    let reader = ScxReader::open(&compact_path).unwrap();
    assert_eq!(reader.n_obs(), 4); // 6 - 2

    // Layer should be present and row-filtered
    let layer = reader.read_layer("normalized").unwrap();
    assert_eq!(layer.shape.0, 4);

    // obsm should be present and row-filtered
    let obsm = reader.read_obsm("X_pca").unwrap();
    assert_eq!(obsm.num_rows(), 4);

    // uns should be preserved
    let uns = reader.read_uns().unwrap();
    assert_eq!(uns["method"], "test");
}

/// Task 7: append with Scx1 codec
#[test]
fn test_append_with_codec_scx1() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "scx1app.scx", 6, 10, 1);

    let new_obs = sample_obs(4);
    let (indptr, indices, values) = sample_shard_data(4, 10);
    scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        CodecId::Scx1,
        16384,
    )
    .unwrap();

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), 10);
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 10);
    assert_eq!(csr.nnz(), 20);
}

/// Task 7: compact round-trip preserves Zstd-encoded data
#[test]
fn test_compact_with_codec_zstd() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zstdcomp.scx");

    // Write file with Zstd codec
    let header = sample_header(6, 10);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(6)).unwrap();
    writer.write_var(&sample_var(10)).unwrap();
    let (indptr, indices, values) = sample_shard_data(6, 10);
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
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1710000000,
            action: "convert".to_string(),
            tool: "test".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();
    writer.finish().unwrap();

    // Delete and compact
    scx_ops::mark_deleted(&path, &[2]).unwrap();
    let compact_path = dir.path().join("zstdcomp_out.scx");
    scx_ops::compact(&path, &compact_path).unwrap();

    let reader = ScxReader::open(&compact_path).unwrap();
    assert_eq!(reader.n_obs(), 5);
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 5);
    assert_eq!(csr.nnz(), 10); // 5 rows * 2 nnz
}

/// Task 7: merge files that have active deletion vectors — DVs should be resolved
#[test]
fn test_merge_with_deletion_vectors() {
    let dir = tempfile::tempdir().unwrap();
    let path1 = write_test_file(&dir, "dvm1.scx", 6, 10, 1);
    let path2 = write_test_file(&dir, "dvm2.scx", 4, 10, 1);

    // Delete some cells from path1
    scx_ops::mark_deleted(&path1, &[0, 3]).unwrap();

    // Compact path1 first (merge doesn't handle DVs directly — compact resolves them)
    let compacted1 = dir.path().join("dvm1_compact.scx");
    scx_ops::compact(&path1, &compacted1).unwrap();

    let output = dir.path().join("dvm_merged.scx");
    scx_ops::merge(&[compacted1.as_path(), path2.as_path()], &output).unwrap();

    let reader = ScxReader::open(&output).unwrap();
    assert_eq!(reader.n_obs(), 8); // (6-2) + 4
    assert!(!reader.header().has_deletion_vectors());

    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 8);
    assert_eq!(csr.nnz(), 16); // 8 rows * 2 nnz
}

/// Task 7: rollback on fresh file with no previous catalog should error
#[test]
fn test_rollback_no_previous_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "norb.scx", 6, 10, 1);

    let result = scx_ops::rollback(&path);
    assert!(result.is_err(), "rollback on fresh file should error");
}

/// Task 7: merge single file produces equivalent output
#[test]
fn test_merge_single_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "single.scx", 6, 10, 1);

    let output = dir.path().join("single_merged.scx");
    scx_ops::merge(&[path.as_path()], &output).unwrap();

    let reader = ScxReader::open(&output).unwrap();
    assert_eq!(reader.n_obs(), 6);
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 6);
    assert_eq!(csr.nnz(), 12); // 6 rows * 2 nnz
}

/// Task 7: merge preserves per-layer value encoding (Bug 2.4 verification)
#[test]
fn test_merge_preserves_layer_encoding() {
    let dir = tempfile::tempdir().unwrap();
    let path1 = write_test_file_with_layers_obsm(&dir, "le1.scx", 4, 10);
    let path2 = write_test_file_with_layers_obsm(&dir, "le2.scx", 4, 10);

    let output = dir.path().join("le_merged.scx");
    scx_ops::merge(&[path1.as_path(), path2.as_path()], &output).unwrap();

    let reader = ScxReader::open(&output).unwrap();
    assert_eq!(reader.n_obs(), 8);

    // Layer should exist with all 8 rows
    let layer = reader.read_layer("normalized").unwrap();
    assert_eq!(layer.shape.0, 8);

    // Verify Float32 values survived — first cell's first value should be 1.0 * 1.5 = 1.5
    assert!(
        (layer.data[0] - 1.5).abs() < 0.01,
        "expected ~1.5, got {}",
        layer.data[0]
    );
}

/// Task 7: append with 0 rows is a no-op
#[test]
fn test_append_empty_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "empty_app.scx", 6, 10, 1);

    let original_size = std::fs::metadata(&path).unwrap().len();

    let new_obs = sample_obs(0);
    let indptr = vec![0u64];
    scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &[],
        &[],
        ValueEncoding::Uint8,
        CodecId::None,
        16384,
    )
    .unwrap();

    // File should be unchanged
    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), 6);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), original_size);
}

/// Issue 2.4: append should reject indices >= n_vars
#[test]
fn test_append_rejects_oob_indices() {
    let dir = tempfile::tempdir().unwrap();
    let n_vars = 10;
    let path = write_test_file(&dir, "oob.scx", 4, n_vars, 1);

    let new_obs = sample_obs(2);
    // Create indices with one value == n_vars (out of bounds)
    let indptr = vec![0u64, 2, 4];
    let indices = vec![0u32, n_vars as u32, 1, 3]; // index 10 is OOB for n_vars=10
    let values = vec![1u8, 2, 3, 4];

    let result = scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        CodecId::None,
        16384,
    );

    assert!(result.is_err(), "append should reject OOB indices");
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("out of bounds"),
        "expected IndexOutOfBounds error, got: {err}"
    );

    // File should be unchanged (error before any writes)
    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), 4);
}
