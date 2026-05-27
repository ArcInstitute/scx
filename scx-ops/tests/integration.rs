use arrow::array::{DictionaryArray, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Int8Type, Schema};
use scx_codec::{CodecId, CodecSelection, ValueEncoding};
use scx_format::header::{FileHeader, MAGIC};
use scx_format::provenance::ProvenanceEntry;
use scx_format::writer::ScxWriter;
use scx_format::{ScxReader, ShardHeader, SHARD_HEADER_SIZE};
use scx_ops::AppendOptions;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn sample_header(n_obs: u64, n_vars: u64) -> FileHeader {
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

/// Build CSC arrays for a single column range over a u8 dense matrix.
fn dense_to_csc_range(
    dense: &[u8],
    n_obs: usize,
    n_vars: usize,
    col_start: usize,
    col_end: usize,
) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr: Vec<u64> = vec![0];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for c in col_start..col_end {
        for r in 0..n_obs {
            let v = dense[r * n_vars + c];
            if v != 0 {
                indices.push(r as u32);
                values.push(v);
            }
        }
        indptr.push(indices.len() as u64);
    }
    (indptr, indices, values)
}

/// Write an SCX test file equipped with both CSR and a CSC sidecar.
/// Returns the file path. Phase H mutating-op tests use this to
/// verify that CSC drop + clear flag fires on the output.
fn write_csc_test_file(
    dir: &TempDir,
    filename: &str,
    n_obs: usize,
    n_vars: usize,
    cols_per_csc_shard: usize,
) -> PathBuf {
    let path = dir.path().join(filename);
    let header = sample_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    // CSR shard built from a deterministic dense pattern.
    let mut dense = vec![0u8; n_obs * n_vars];
    for r in 0..n_obs {
        for c in 0..n_vars {
            if (r + c) % 3 == 0 {
                dense[r * n_vars + c] = ((r * 7 + c * 11) % 200 + 1) as u8;
            }
        }
    }
    let mut indptr_csr = vec![0u64];
    let mut indices_csr = Vec::new();
    let mut values_csr = Vec::new();
    for r in 0..n_obs {
        for c in 0..n_vars {
            let v = dense[r * n_vars + c];
            if v != 0 {
                indices_csr.push(c as u32);
                values_csr.push(v);
            }
        }
        indptr_csr.push(indices_csr.len() as u64);
    }
    writer
        .write_csr_shard(
            &indptr_csr,
            &indices_csr,
            &values_csr,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();

    // CSC sidecar split into shards by column.
    let mut col_start = 0usize;
    while col_start < n_vars {
        let col_end = (col_start + cols_per_csc_shard).min(n_vars);
        let (ip, ix, vb) = dense_to_csc_range(&dense, n_obs, n_vars, col_start, col_end);
        writer
            .write_csc_shard(
                &ip,
                &ix,
                &vb,
                CodecId::None,
                ValueEncoding::Uint8,
                col_start as u64,
            )
            .unwrap();
        col_start = col_end;
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
        &AppendOptions::default(),
    )
    .unwrap();

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), 10);
    assert_eq!(reader.header().n_csr_shards, 2);

    // Phase 2d: append produces ObsMetadataShard sections via the
    // convert-on-append path. `read_obs()` transparently reassembles
    // them — exercise both the streaming view and the assembled view.
    assert!(reader.obs_metadata_shard_count() > 0);
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
        &AppendOptions::default(),
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

#[test]
fn test_mark_deleted_rejects_oob_indices() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "del_oob.scx", 6, 10, 1);

    let err = scx_ops::mark_deleted(&path, &[0, 999_999_999]).unwrap_err();
    assert!(
        matches!(
            err,
            scx_ops::OpsError::CellIndexOutOfBounds { index, n_obs }
                if index == 999_999_999 && n_obs == 6
        ),
        "expected CellIndexOutOfBounds, got: {err}"
    );

    // File untouched: no deletion vectors written.
    let reader = ScxReader::open(&path).unwrap();
    assert!(!reader.header().has_deletion_vectors());
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
            &AppendOptions::default(),
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
        &AppendOptions::default(),
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

    // Phase 2a: merge now emits ObsMetadataShard sections; `read_obs`
    // transparently reassembles them across shards.
    assert!(reader.obs_metadata_shard_count() > 0);
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
                    &AppendOptions::default(),
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
        &AppendOptions::default(),
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
            &AppendOptions::default(),
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
        &AppendOptions {
            shard_target_rows: NonZeroU32::new(3).unwrap(),
            ..AppendOptions::default()
        },
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
        &AppendOptions::default(),
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
    // pwrite commit point (docs/format.md (Append)).
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
        &AppendOptions::default(),
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

/// Read the codec_id from the most recently appended CSR shard.
fn last_appended_shard_codec(path: &std::path::Path) -> CodecId {
    use scx_format::section::SectionType;
    let reader = ScxReader::open(path).unwrap();
    let shards = reader.catalog().shards(SectionType::CsrShard);
    let last = shards
        .last()
        .expect("file must have at least one CSR shard");
    let bytes = reader.section_bytes(last).unwrap();
    let sh =
        ShardHeader::read_from(&mut std::io::Cursor::new(&bytes[..SHARD_HEADER_SIZE])).unwrap();
    CodecId::from_u8(sh.codec_id).expect("shard codec_id should be a valid CodecId")
}

/// P0 #2: append must honour `CodecSelection::Explicit(c)` — every legal
/// codec, when forced, must end up on disk in the appended shard's
/// ShardHeader. The previous test only checked shape/nnz, which is why
/// the `_codec_id` parameter-ignored bug was undetectable.
fn append_with_explicit_codec(
    fixture_name: &str,
    codec: CodecId,
    value_encoding: ValueEncoding,
    raw_values: Vec<u8>,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, fixture_name, 6, 10, 1);

    let new_obs = sample_obs(4);
    // 4 rows × 2 nnz each, indices in [0, 10).
    let indptr: Vec<u64> = vec![0, 2, 4, 6, 8];
    let indices: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7];
    assert_eq!(
        raw_values.len(),
        8 * value_encoding.byte_width(),
        "raw_values length must match 8 nnz × byte_width"
    );

    scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &raw_values,
        value_encoding,
        &AppendOptions {
            codec: CodecSelection::Explicit(codec),
            ..AppendOptions::default()
        },
    )
    .unwrap();

    assert_eq!(
        last_appended_shard_codec(&path),
        codec,
        "appended shard codec_id must equal the explicitly requested codec"
    );

    // Sanity round-trip: decoded CSR must still have the expected shape.
    let reader = ScxReader::open(&path).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 10);
    assert_eq!(csr.nnz(), 20);
}

fn u8_values(vs: &[u8]) -> Vec<u8> {
    vs.to_vec()
}

fn f32_values(vs: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vs.len() * 4);
    for v in vs {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[test]
fn test_append_with_explicit_codec_none() {
    append_with_explicit_codec(
        "explicit_none.scx",
        CodecId::None,
        ValueEncoding::Uint8,
        u8_values(&[1, 2, 3, 4, 5, 6, 7, 8]),
    );
}

#[test]
fn test_append_with_explicit_codec_scx1() {
    // Scx1 requires integer encoding; uint8 with small UMI values.
    append_with_explicit_codec(
        "explicit_scx1.scx",
        CodecId::Scx1,
        ValueEncoding::Uint8,
        u8_values(&[1, 2, 3, 4, 5, 6, 7, 8]),
    );
}

#[test]
fn test_append_with_explicit_codec_zstd() {
    append_with_explicit_codec(
        "explicit_zstd.scx",
        CodecId::Zstd,
        ValueEncoding::Uint8,
        u8_values(&[10, 20, 30, 40, 50, 60, 70, 80]),
    );
}

#[test]
fn test_append_with_explicit_codec_lz4() {
    append_with_explicit_codec(
        "explicit_lz4.scx",
        CodecId::Lz4Shuffle,
        ValueEncoding::Uint8,
        u8_values(&[1, 2, 3, 4, 5, 6, 7, 8]),
    );
}

#[test]
fn test_append_with_explicit_codec_pcodec() {
    // Pcodec's natural input is float data.
    append_with_explicit_codec(
        "explicit_pcodec.scx",
        CodecId::Pcodec,
        ValueEncoding::Float32,
        f32_values(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    );
}

#[test]
fn test_append_with_codec_auto_matches_select_codec_for_modality() {
    // CodecSelection::Auto should call select_codec_for_modality per
    // shard. With modality_id == 0 the resolved modality_type is RNA,
    // which delegates to select_codec — so the on-disk codec must match
    // what select_codec would have chosen for the shard data.
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "auto_codec.scx", 6, 10, 1);

    let new_obs = sample_obs(4);
    let (indptr, indices, values) = sample_shard_data(4, 10);
    let expected = scx_format::select_codec_for_modality(
        &values,
        ValueEncoding::Uint8,
        scx_format::ModalityType::Rna,
    );
    scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        &AppendOptions::default(),
    )
    .unwrap();

    assert_eq!(
        last_appended_shard_codec(&path),
        expected,
        "Auto must defer to select_codec_for_modality"
    );
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
        &AppendOptions::default(),
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
        &AppendOptions::default(),
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

/// P1 #11: append must reject `target_n_vars > u32::MAX` early, mirroring
/// the writer-side `NVarsOverflow` guard at scx-format::writer:496. Build
/// a synthetic target with `header.n_vars = u32::MAX + 1` (no CSR shards
/// — the writer-side check would otherwise reject the file at creation
/// time) and assert that append fails *before* writing anything.
#[test]
fn test_append_rejects_n_vars_overflow() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nvars_overflow.scx");
    let n_obs: u64 = 4;
    let n_vars_overflow: u64 = u32::MAX as u64 + 1;

    // Hand-rolled fixture: valid SCX file with no CSR shards but
    // header.n_vars > u32::MAX. ScxWriter::new doesn't validate the
    // header; we just skip write_csr_shard to bypass the writer-side
    // NVarsOverflow guard.
    let mut header = sample_header(n_obs, n_vars_overflow);
    header.index_dtype = 1; // u32, since n_vars > u16::MAX
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs as usize)).unwrap();
    // Skip write_var with the absurdly large n_vars — var is optional.
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
    let pre_append_size = std::fs::metadata(&path).unwrap().len();

    let new_obs = sample_obs(2);
    let indptr = vec![0u64, 1, 2];
    let indices = vec![0u32, 1];
    let values = vec![1u8, 2];

    let result = scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        &AppendOptions::default(),
    );

    assert!(result.is_err(), "append should reject n_vars > u32::MAX");
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("exceeds u32::MAX"),
        "expected NVarsOverflow error, got: {err}"
    );

    // File must be byte-identical post-failure — the guard fires before
    // any disk mutation.
    let post_append_size = std::fs::metadata(&path).unwrap().len();
    assert_eq!(
        post_append_size, pre_append_size,
        "append must not touch the file on n_vars overflow"
    );
}

// ---------------------------------------------------------------------------
// CSC drop on mutating ops (`append`, `compact`, `merge`)
// ---------------------------------------------------------------------------

/// Append on a CSC-equipped file drops the CSC sidecar:
///   - `header.has_csc()` returns false on the post-append file
///   - `header.n_csc_shards == 0`
///   - The full catalog contains zero CscShard entries
///   - CSR readback works as expected
#[test]
fn test_append_drops_csc_from_input() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_csc_test_file(&dir, "csc_input.scx", 6, 8, 4);

    // Sanity: pre-append, the file has CSC shards.
    {
        let r = ScxReader::open(&path).unwrap();
        assert!(r.header().has_csc());
        assert!(r.header().n_csc_shards >= 1);
    }

    let new_obs = sample_obs(4);
    let (indptr, indices, values) = sample_shard_data(4, 8);
    scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        &AppendOptions::default(),
    )
    .unwrap();

    let r = ScxReader::open(&path).unwrap();
    assert_eq!(r.n_obs(), 10);
    assert!(
        !r.header().has_csc(),
        "has_csc flag should be cleared after append"
    );
    assert_eq!(r.header().n_csc_shards, 0);

    // Catalog should contain no CscShard entries.
    let csc_count = r
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == scx_format::section::SectionType::CscShard)
        .count();
    assert_eq!(csc_count, 0);

    // CSR readback unchanged.
    let csr = r.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 10);
}

#[test]
fn test_compact_drops_csc_from_input() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_csc_test_file(&dir, "csc_compact_in.scx", 6, 8, 4);
    let output = dir.path().join("csc_compact_out.scx");

    scx_ops::compact(&input, &output).unwrap();

    let r = ScxReader::open(&output).unwrap();
    assert!(
        !r.header().has_csc(),
        "compact output should not advertise CSC"
    );
    assert_eq!(r.header().n_csc_shards, 0);
    let csc_count = r
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == scx_format::section::SectionType::CscShard)
        .count();
    assert_eq!(csc_count, 0);
}

#[test]
fn test_merge_drops_csc_from_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let p1 = write_csc_test_file(&dir, "csc_m1.scx", 4, 6, 3);
    let p2 = write_csc_test_file(&dir, "csc_m2.scx", 5, 6, 3);
    let output = dir.path().join("csc_merged.scx");

    scx_ops::merge(&[p1.as_path(), p2.as_path()], &output).unwrap();

    let r = ScxReader::open(&output).unwrap();
    assert!(
        !r.header().has_csc(),
        "merge output should not advertise CSC"
    );
    assert_eq!(r.header().n_csc_shards, 0);
    assert_eq!(r.n_obs(), 9);
    let csc_count = r
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == scx_format::section::SectionType::CscShard)
        .count();
    assert_eq!(csc_count, 0);
}

/// Pure-CSR file is untouched by the new CSC-drop logic — `has_csc`
/// stays false and the file remains structurally identical to the
/// pre-Phase-H behavior.
#[test]
fn test_append_pure_csr_unaffected() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "pure_csr.scx", 4, 6, 1);

    let new_obs = sample_obs(2);
    let (indptr, indices, values) = sample_shard_data(2, 6);
    scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        &AppendOptions::default(),
    )
    .unwrap();

    let r = ScxReader::open(&path).unwrap();
    assert!(!r.header().has_csc());
    assert_eq!(r.header().n_csc_shards, 0);
    assert_eq!(r.n_obs(), 6);
}

// ---------------------------------------------------------------------------
// LargeUtf8 round-trip tests (regression for Arrow IPC 2 GB offset overflow)
//
// On disk we now store string/binary obs columns as `LargeUtf8` /
// `LargeBinary` (64-bit offsets) so files with > ~2.8 M cells do not
// overflow Arrow IPC's 32-bit offset limit. Callers should still see
// the canonical narrow types in memory after read.
// ---------------------------------------------------------------------------

/// Build an obs RecordBatch with both a plain `Utf8` `cell_id` and a
/// `Dictionary(Int8, Utf8)` `cluster` column. Used to exercise both
/// the upcast/downcast path and the dictionary handling.
fn sample_obs_with_cluster(n: usize) -> arrow::array::RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let clusters: Vec<&str> = (0..n)
        .map(|i| match i % 3 {
            0 => "A",
            1 => "B",
            _ => "C",
        })
        .collect();

    let cluster_dt = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cluster", cluster_dt, true),
    ]);

    let cluster_arr: DictionaryArray<Int8Type> = clusters.into_iter().collect();

    arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(cluster_arr),
        ],
    )
    .unwrap()
}

#[test]
fn test_append_preserves_utf8_schema_via_largeutf8_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("append_utf8.scx");

    // Write baseline file with a richer obs (Utf8 cell_id + Dictionary cluster).
    {
        let header = sample_header(4, 10);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs_with_cluster(4)).unwrap();
        writer.write_var(&sample_var(10)).unwrap();
        let (indptr, indices, values) = sample_shard_data(4, 10);
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
    }

    // Append more rows.
    let new_obs = sample_obs_with_cluster(3);
    let (indptr, indices, values) = sample_shard_data(3, 10);
    scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        &AppendOptions::default(),
    )
    .unwrap();

    // Reopen and verify schema reports the canonical narrow `Utf8` —
    // not `LargeUtf8` — even though the data is stored as LargeUtf8 on
    // disk after the append (which exercises the inline-IPC bypass
    // path with explicit upcast/downcast).
    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), 7);

    // Phase 2d: append produces ObsMetadataShard sections; use the
    // assembled reader to materialise the full obs.
    let schema = reader.read_obs_schema().unwrap();
    assert_eq!(schema.field(0).name(), "cell_id");
    assert_eq!(schema.field(0).data_type(), &DataType::Utf8);

    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 7);
    assert_eq!(obs.schema().field(0).data_type(), &DataType::Utf8);
    let cell_ids = obs
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(cell_ids.value(0), "cell_0");
    assert_eq!(cell_ids.value(3), "cell_3");
    assert_eq!(cell_ids.value(4), "cell_0"); // start of appended batch
    assert_eq!(cell_ids.value(6), "cell_2");

    // The cluster column went through `unify_dict_columns` during
    // append, so it lands as a flat `Utf8` column on the merged side.
    // The key invariant for this test is that it is *not* `LargeUtf8`.
    let cluster_dt = obs.schema().field(1).data_type().clone();
    assert!(
        matches!(cluster_dt, DataType::Utf8 | DataType::Dictionary(_, _)),
        "cluster column dtype should be Utf8 or Dictionary, got {cluster_dt:?}",
    );
    assert!(
        !matches!(cluster_dt, DataType::LargeUtf8),
        "cluster column must not surface as LargeUtf8",
    );
}

#[test]
fn test_merge_preserves_utf8_schema_via_largeutf8_round_trip() {
    let dir = tempfile::tempdir().unwrap();

    // Build two SCX inputs with rich obs.
    let make_input = |name: &str, n: usize| -> PathBuf {
        let path = dir.path().join(name);
        let header = sample_header(n as u64, 10);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs_with_cluster(n)).unwrap();
        writer.write_var(&sample_var(10)).unwrap();
        let (indptr, indices, values) = sample_shard_data(n, 10);
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
                action: "convert".to_string(),
                tool: "test".to_string(),
                params_json: "{}".to_string(),
                input_checksums: vec![],
            }])
            .unwrap();
        writer.finish().unwrap();
        path
    };

    let path1 = make_input("merge1.scx", 5);
    let path2 = make_input("merge2.scx", 7);
    let output = dir.path().join("merged_utf8.scx");

    scx_ops::merge(&[path1.as_path(), path2.as_path()], &output).unwrap();

    let reader = ScxReader::open(&output).unwrap();
    assert_eq!(reader.n_obs(), 12);

    // Phase 2a: merge emits ObsMetadataShard sections; use the
    // assembled reader path for tests that want the merged batch.
    // Schema-only path (read_obs_schema) and full read must both
    // report the canonical narrow `Utf8` after the LargeUtf8 → Utf8
    // downcast on each shard.
    let schema = reader.read_obs_schema().unwrap();
    assert_eq!(schema.field(0).data_type(), &DataType::Utf8);

    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 12);
    assert_eq!(obs.schema().field(0).data_type(), &DataType::Utf8);
    // Compare dtypes column-by-column; `read_obs_schema` returns the
    // first shard's footer schema (with stamped shard metadata),
    // while the assembled batch's schema strips per-shard fields
    // (`shard_idx` / `row_start` / `n_shard_rows`). The field
    // payload structure is identical.
    for (i, f) in schema.fields().iter().enumerate() {
        assert_eq!(obs.schema().field(i).name(), f.name());
        assert_eq!(obs.schema().field(i).data_type(), f.data_type());
    }

    let cell_ids = obs
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(cell_ids.value(0), "cell_0"); // file 1
    assert_eq!(cell_ids.value(4), "cell_4");
    assert_eq!(cell_ids.value(5), "cell_0"); // file 2
    assert_eq!(cell_ids.value(11), "cell_6");

    let cluster_dt = obs.schema().field(1).data_type().clone();
    assert!(
        !matches!(cluster_dt, DataType::LargeUtf8),
        "cluster column must not surface as LargeUtf8 after merge",
    );
}

// ---------------------------------------------------------------------------
// Multimodal append tests (PR #68 regression)
// ---------------------------------------------------------------------------

/// `append` (with `AppendOptions.modality_id` set) must stamp the appended CSR shard's
/// `ShardHeader.n_minor` with the target modality's `n_vars`, not the
/// file-wide `header.n_vars`.  This was one of the original validation
/// sites in `scx-ops::append::append`.
#[test]
fn test_append_for_modality_uses_per_modality_n_vars() {
    use scx_format::modality::ModalityType;
    use scx_format::section::SectionType;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multimodal_append.scx");

    // Two modalities with DISTINCT n_vars; "rna" is the max so the
    // file-wide header.n_vars = 30 == rna.n_vars. We then append into
    // "adt" (n_vars = 10) — if the fix is missing, the appended shard
    // would stamp n_minor = 30 instead of 10.
    let rna_n_vars: u64 = 30;
    let adt_n_vars: u64 = 10;
    let header = sample_header(4, rna_n_vars);

    {
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(4)).unwrap();

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
        assert_eq!(rna_id, 1);
        assert_eq!(adt_id, 2);

        writer.write_var_for(rna_id, &sample_var(2)).unwrap();
        writer.write_var_for(adt_id, &sample_var(2)).unwrap();
        writer.set_modality_n_vars(rna_id, rna_n_vars).unwrap();
        writer.set_modality_n_vars(adt_id, adt_n_vars).unwrap();

        // Seed each modality with one CSR shard so the file is well-
        // formed before the append.
        let (rna_indptr, rna_indices, rna_values) = sample_shard_data(4, rna_n_vars as usize);
        writer
            .write_csr_shard_for(
                rna_id,
                &rna_indptr,
                &rna_indices,
                &rna_values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        let (adt_indptr, adt_indices, adt_values) = sample_shard_data(4, adt_n_vars as usize);
        writer
            .write_csr_shard_for(
                adt_id,
                &adt_indptr,
                &adt_indices,
                &adt_values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();

        writer.finish().unwrap();
    }

    // Append two new rows into the "adt" modality.
    let new_obs = sample_obs(2);
    let (new_indptr, new_indices, new_values) = sample_shard_data(2, adt_n_vars as usize);
    scx_ops::append(
        &path,
        &new_obs,
        &new_indptr,
        &new_indices,
        &new_values,
        ValueEncoding::Uint8,
        &AppendOptions {
            modality_id: 2,
            ..AppendOptions::default()
        },
    )
    .unwrap();

    // The appended shard should carry adt.n_vars in both stats and
    // on-disk header — not rna.n_vars (which is the file-wide max).
    let reader = ScxReader::open(&path).unwrap();
    let adt_shards: Vec<&scx_format::FullCatalogEntry> = reader
        .catalog()
        .shards(SectionType::CsrShard)
        .into_iter()
        .filter(|e| e.modality_id == 2)
        .collect();
    assert_eq!(
        adt_shards.len(),
        2,
        "adt should have 2 CSR shards after append (1 seed + 1 appended)"
    );

    for entry in &adt_shards {
        let stats = entry
            .stats
            .as_ref()
            .expect("v2 catalog must carry shard stats");
        assert_eq!(
            stats.col_end, adt_n_vars,
            "appended shard col_end should be adt.n_vars ({adt_n_vars}), got {}",
            stats.col_end
        );
        assert_eq!(stats.col_start, 0);

        let bytes = reader.section_bytes(entry).unwrap();
        let sh =
            ShardHeader::read_from(&mut std::io::Cursor::new(&bytes[..SHARD_HEADER_SIZE])).unwrap();
        assert_eq!(
            sh.n_minor as u64, adt_n_vars,
            "appended shard ShardHeader.n_minor should be adt.n_vars ({adt_n_vars}), got {}",
            sh.n_minor
        );
    }
}

// ---------------------------------------------------------------------------
// Streaming append (append_from_reader) tests — P1 #15 (review §8.3)
// ---------------------------------------------------------------------------

/// Streaming append must produce a result indistinguishable from the
/// bulk `append` path: same n_obs, same CSR contents, same obs.
#[test]
fn test_streaming_append_matches_bulk_append() {
    let dir = tempfile::tempdir().unwrap();

    // Two identical targets; two identical sources (single-shard).
    let target_a = write_test_file(&dir, "stream_a_target.scx", 6, 10, 1);
    let target_b = write_test_file(&dir, "stream_b_target.scx", 6, 10, 1);
    let source = write_test_file(&dir, "stream_source.scx", 4, 10, 1);

    // Bulk path: read source via read_all_csr_shards and call legacy append.
    let src_reader = ScxReader::open(&source).unwrap();
    let csr = src_reader.read_all_csr_shards().unwrap();
    let bulk_indptr: Vec<u64> = csr.indptr.iter().map(|&v| v as u64).collect();
    let bulk_indices: Vec<u32> = csr.indices.iter().map(|&v| v as u32).collect();
    let bulk_values: Vec<u8> = csr.data.iter().map(|&v| v as u8).collect();
    let src_obs = src_reader.read_obs().unwrap();
    drop(src_reader);

    scx_ops::append(
        &target_a,
        &src_obs,
        &bulk_indptr,
        &bulk_indices,
        &bulk_values,
        ValueEncoding::Uint8,
        &AppendOptions::default(),
    )
    .unwrap();

    // Streaming path: pass the source reader directly.
    let src_reader = ScxReader::open(&source).unwrap();
    scx_ops::append_from_reader(&target_b, &src_reader, &AppendOptions::default(), 0).unwrap();
    drop(src_reader);

    let ra = ScxReader::open(&target_a).unwrap();
    let rb = ScxReader::open(&target_b).unwrap();
    assert_eq!(ra.n_obs(), 10);
    assert_eq!(rb.n_obs(), 10);
    let csr_a = ra.read_all_csr_shards().unwrap();
    let csr_b = rb.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.shape, csr_b.shape);
    assert_eq!(csr_a.indptr, csr_b.indptr);
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data);
    let obs_a = ra.read_obs().unwrap();
    let obs_b = rb.read_obs().unwrap();
    assert_eq!(obs_a.num_rows(), obs_b.num_rows());
}

/// Multi-shard source must round-trip into the target with correct global
/// row offsets and shard count.
#[test]
fn test_streaming_append_multi_shard_source() {
    let dir = tempfile::tempdir().unwrap();
    let target = write_test_file(&dir, "stream_multi_target.scx", 4, 10, 1);
    // Source has 4 shards of 25 rows each (100 total).
    let source = write_test_file(&dir, "stream_multi_source.scx", 100, 10, 4);

    let target_pre_shards = {
        let r = ScxReader::open(&target).unwrap();
        r.header().n_csr_shards
    };

    let src_reader = ScxReader::open(&source).unwrap();
    scx_ops::append_from_reader(&target, &src_reader, &AppendOptions::default(), 0).unwrap();
    drop(src_reader);

    let reader = ScxReader::open(&target).unwrap();
    assert_eq!(reader.n_obs(), 104);
    // Each source shard becomes one target shard (source rows ≤
    // shard_target_rows), so we add 4 shards on top of the pre-existing.
    assert_eq!(reader.header().n_csr_shards, target_pre_shards + 4);

    // CSR readback over the whole file must reconstruct the appended rows
    // in source order (global_offset monotonic).
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 104);
}

/// Streaming append must honour an explicit codec selection on every
/// re-encoded shard, mirroring the bulk path's codec-respect test.
#[test]
fn test_streaming_append_respects_explicit_codec() {
    let dir = tempfile::tempdir().unwrap();
    // Target written with codec None; source likewise.
    let target = write_test_file(&dir, "stream_codec_target.scx", 4, 10, 1);
    let source = write_test_file(&dir, "stream_codec_source.scx", 8, 10, 2);

    let src_reader = ScxReader::open(&source).unwrap();
    scx_ops::append_from_reader(
        &target,
        &src_reader,
        &AppendOptions {
            codec: CodecSelection::Explicit(CodecId::Zstd),
            ..AppendOptions::default()
        },
        0,
    )
    .unwrap();
    drop(src_reader);

    // Every appended shard (source had 2) must carry codec_id == Zstd.
    let reader = ScxReader::open(&target).unwrap();
    let appended: Vec<&scx_format::FullCatalogEntry> = reader
        .catalog()
        .shards(scx_format::section::SectionType::CsrShard)
        .into_iter()
        .filter(|e| e.stats.as_ref().map(|s| s.row_start >= 4).unwrap_or(false))
        .collect();
    assert!(!appended.is_empty(), "expected appended shards in catalog");
    for entry in appended {
        let bytes = reader.section_bytes(entry).unwrap();
        let sh =
            ShardHeader::read_from(&mut std::io::Cursor::new(&bytes[..SHARD_HEADER_SIZE])).unwrap();
        assert_eq!(
            sh.codec_id,
            CodecId::Zstd as u8,
            "appended shard '{}' should be Zstd-encoded",
            entry.name
        );
    }
}

/// Raw-copy fast path: when codec selection is Auto and source and
/// target both use codec None / matching encoding / matching index dtype,
/// the appended shard's payload bytes must equal the source payload
/// (header diff only).
#[test]
fn test_streaming_append_raw_copy_fast_path() {
    let dir = tempfile::tempdir().unwrap();
    let target = write_test_file(&dir, "stream_raw_target.scx", 4, 10, 1);
    let source = write_test_file(&dir, "stream_raw_source.scx", 8, 10, 1);

    // Snapshot source payload bytes (header-stripped) before the append.
    let source_payload: Vec<u8> = {
        let r = ScxReader::open(&source).unwrap();
        let shards = r
            .catalog()
            .shards(scx_format::section::SectionType::CsrShard);
        assert_eq!(shards.len(), 1);
        let bytes = r.section_bytes(shards[0]).unwrap();
        bytes[SHARD_HEADER_SIZE..].to_vec()
    };

    let src_reader = ScxReader::open(&source).unwrap();
    scx_ops::append_from_reader(&target, &src_reader, &AppendOptions::default(), 0).unwrap();
    drop(src_reader);

    let reader = ScxReader::open(&target).unwrap();
    // Locate the appended shard (row_start == 4).
    let appended = reader
        .catalog()
        .shards(scx_format::section::SectionType::CsrShard)
        .into_iter()
        .find(|e| e.stats.as_ref().map(|s| s.row_start == 4).unwrap_or(false))
        .expect("appended shard not found");
    let bytes = reader.section_bytes(appended).unwrap();
    assert_eq!(
        &bytes[SHARD_HEADER_SIZE..],
        source_payload.as_slice(),
        "raw-copy fast path must leave payload bytes verbatim"
    );

    // Header must be patched: global_offset bumped, codec_id matches source's None.
    let sh =
        ShardHeader::read_from(&mut std::io::Cursor::new(&bytes[..SHARD_HEADER_SIZE])).unwrap();
    assert_eq!(sh.global_offset, 4);
    assert_eq!(sh.codec_id, CodecId::None as u8);
}

/// Raw-copy fast path must reuse the source `FullCatalogEntry.stats` for
/// invariant fields (nnz, value_min/max/sum) and patch only `row_start` /
/// `row_end`. Exercises a compressed (Zstd) source so the previous code
/// would have decoded — the new code must skip decode and clone stats.
#[test]
fn test_streaming_append_raw_copy_reuses_stats() {
    let dir = tempfile::tempdir().unwrap();

    // Build a Zstd-compressed source file directly so we can inspect its
    // pre-append stats.
    let source = dir.path().join("stream_raw_stats_source.scx");
    {
        let header = sample_header(8, 10);
        let mut writer = ScxWriter::new(&source, header).unwrap();
        writer.write_obs(&sample_obs(8)).unwrap();
        writer.write_var(&sample_var(10)).unwrap();
        let (indptr, indices, values) = sample_shard_data(8, 10);
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
    }

    // Snapshot the source's stats so we can compare after raw-copy.
    let src_stats = {
        let r = ScxReader::open(&source).unwrap();
        let shards = r
            .catalog()
            .shards(scx_format::section::SectionType::CsrShard);
        assert_eq!(shards.len(), 1);
        shards[0].stats.clone().expect("source shard has stats")
    };

    // Target uses the same codec so raw-copy is eligible.
    let target = dir.path().join("stream_raw_stats_target.scx");
    {
        let header = sample_header(4, 10);
        let mut writer = ScxWriter::new(&target, header).unwrap();
        writer.write_obs(&sample_obs(4)).unwrap();
        writer.write_var(&sample_var(10)).unwrap();
        let (indptr, indices, values) = sample_shard_data(4, 10);
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
    }

    let src_reader = ScxReader::open(&source).unwrap();
    scx_ops::append_from_reader(
        &target,
        &src_reader,
        &AppendOptions {
            codec: CodecSelection::Explicit(CodecId::Zstd),
            ..AppendOptions::default()
        },
        0,
    )
    .unwrap();
    drop(src_reader);

    let reader = ScxReader::open(&target).unwrap();
    let appended = reader
        .catalog()
        .shards(scx_format::section::SectionType::CsrShard)
        .into_iter()
        .find(|e| e.stats.as_ref().map(|s| s.row_start == 4).unwrap_or(false))
        .expect("appended shard not found");
    let appended_stats = appended.stats.as_ref().expect("appended shard has stats");

    // Row range must reflect the new global position.
    assert_eq!(appended_stats.row_start, 4);
    assert_eq!(appended_stats.row_end, 4 + 8);

    // All invariant fields must match the source exactly — this is what
    // pins the "reuse stats, don't recompute" contract.
    assert_eq!(appended_stats.nnz, src_stats.nnz);
    assert_eq!(appended_stats.value_min, src_stats.value_min);
    assert_eq!(appended_stats.value_max, src_stats.value_max);
    assert_eq!(appended_stats.value_sum, src_stats.value_sum);
    assert_eq!(appended_stats.col_start, src_stats.col_start);
    assert_eq!(appended_stats.col_end, src_stats.col_end);
}

/// Streaming append must drop CSC sidecars from the target, same as the
/// bulk path.
#[test]
fn test_streaming_append_drops_csc_from_input() {
    let dir = tempfile::tempdir().unwrap();
    let target = write_csc_test_file(&dir, "stream_csc_target.scx", 6, 8, 4);
    let source = write_test_file(&dir, "stream_csc_source.scx", 4, 8, 1);

    {
        let r = ScxReader::open(&target).unwrap();
        assert!(r.header().has_csc());
    }

    let src_reader = ScxReader::open(&source).unwrap();
    scx_ops::append_from_reader(&target, &src_reader, &AppendOptions::default(), 0).unwrap();
    drop(src_reader);

    let r = ScxReader::open(&target).unwrap();
    assert_eq!(r.n_obs(), 10);
    assert!(
        !r.header().has_csc(),
        "has_csc flag should be cleared after streaming append"
    );
    assert_eq!(r.header().n_csc_shards, 0);
    let csc_count = r
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == scx_format::section::SectionType::CscShard)
        .count();
    assert_eq!(csc_count, 0);
}

/// Streaming append into a per-modality target stamps the appended
/// shard's `ShardHeader.n_minor` with the target modality's `n_vars`.
#[test]
fn test_streaming_append_multimodal() {
    use scx_format::modality::ModalityType;
    use scx_format::section::SectionType;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stream_multimodal.scx");

    let rna_n_vars: u64 = 30;
    let adt_n_vars: u64 = 10;
    let header = sample_header(4, rna_n_vars);

    {
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(4)).unwrap();

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
        writer.write_var_for(rna_id, &sample_var(2)).unwrap();
        writer.write_var_for(adt_id, &sample_var(2)).unwrap();
        writer.set_modality_n_vars(rna_id, rna_n_vars).unwrap();
        writer.set_modality_n_vars(adt_id, adt_n_vars).unwrap();

        let (rna_indptr, rna_indices, rna_values) = sample_shard_data(4, rna_n_vars as usize);
        writer
            .write_csr_shard_for(
                rna_id,
                &rna_indptr,
                &rna_indices,
                &rna_values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        let (adt_indptr, adt_indices, adt_values) = sample_shard_data(4, adt_n_vars as usize);
        writer
            .write_csr_shard_for(
                adt_id,
                &adt_indptr,
                &adt_indices,
                &adt_values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();

        writer.finish().unwrap();
    }

    // Source: a single-modality file with adt-compatible n_vars = 10.
    let source = write_test_file(
        &dir,
        "stream_multimodal_source.scx",
        2,
        adt_n_vars as usize,
        1,
    );
    let src_reader = ScxReader::open(&source).unwrap();
    scx_ops::append_from_reader(
        &path,
        &src_reader,
        &AppendOptions {
            modality_id: 2,
            ..AppendOptions::default()
        },
        0, // source is single-modality
    )
    .unwrap();
    drop(src_reader);

    let reader = ScxReader::open(&path).unwrap();
    let adt_shards: Vec<&scx_format::FullCatalogEntry> = reader
        .catalog()
        .shards(SectionType::CsrShard)
        .into_iter()
        .filter(|e| e.modality_id == 2)
        .collect();
    assert_eq!(
        adt_shards.len(),
        2,
        "adt should have 2 CSR shards after streaming append"
    );
    // The newly appended shard should have n_minor == adt_n_vars.
    let appended = adt_shards
        .iter()
        .find(|e| e.stats.as_ref().map(|s| s.row_start == 4).unwrap_or(false))
        .expect("appended adt shard not found");
    let bytes = reader.section_bytes(appended).unwrap();
    let sh =
        ShardHeader::read_from(&mut std::io::Cursor::new(&bytes[..SHARD_HEADER_SIZE])).unwrap();
    assert_eq!(
        sh.n_minor as u64, adt_n_vars,
        "streaming-appended shard ShardHeader.n_minor should equal adt.n_vars"
    );
}

// ---------------------------------------------------------------------------
// Phase 6: multimodal lifecycle parity
// ---------------------------------------------------------------------------

/// Build a multimodal file with `rna` and `adt` modalities, returning
/// the path. Each modality has a CSR shard with a single row of nnz=2.
fn write_multimodal_with_csc(
    dir: &TempDir,
    filename: &str,
    n_obs: usize,
    rna_n_vars: u64,
    adt_n_vars: u64,
    with_adt_csc: bool,
) -> PathBuf {
    use scx_format::modality::ModalityType;
    let path = dir.path().join(filename);
    let header = sample_header(n_obs as u64, rna_n_vars);
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
    writer
        .write_var_for(rna_id, &sample_var(rna_n_vars as usize))
        .unwrap();
    writer
        .write_var_for(adt_id, &sample_var(adt_n_vars as usize))
        .unwrap();
    writer.set_modality_n_vars(rna_id, rna_n_vars).unwrap();
    writer.set_modality_n_vars(adt_id, adt_n_vars).unwrap();

    let (rna_indptr, rna_indices, rna_values) = sample_shard_data(n_obs, rna_n_vars as usize);
    writer
        .write_csr_shard_for(
            rna_id,
            &rna_indptr,
            &rna_indices,
            &rna_values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    let (adt_indptr, adt_indices, adt_values) = sample_shard_data(n_obs, adt_n_vars as usize);
    writer
        .write_csr_shard_for(
            adt_id,
            &adt_indptr,
            &adt_indices,
            &adt_values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();

    if with_adt_csc {
        // Build a single CSC shard for adt covering all its columns.
        let mut csc_indptr = vec![0u64];
        let mut csc_indices: Vec<u32> = Vec::new();
        let mut csc_values: Vec<u8> = Vec::new();
        // Reconstruct dense per-row from adt CSR data and build CSC.
        let mut dense = vec![0u8; n_obs * adt_n_vars as usize];
        for r in 0..n_obs {
            let s = adt_indptr[r] as usize;
            let e = adt_indptr[r + 1] as usize;
            for k in s..e {
                dense[r * adt_n_vars as usize + adt_indices[k] as usize] = adt_values[k];
            }
        }
        for c in 0..adt_n_vars as usize {
            for r in 0..n_obs {
                let v = dense[r * adt_n_vars as usize + c];
                if v != 0 {
                    csc_indices.push(r as u32);
                    csc_values.push(v);
                }
            }
            csc_indptr.push(csc_indices.len() as u64);
        }
        writer
            .write_csc_shard_for(
                adt_id,
                &csc_indptr,
                &csc_indices,
                &csc_values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
    }

    writer.finish().unwrap();
    path
}

/// Appending into RNA on a multimodal file drops ADT's CSC sidecar too.
/// Per-modality CSC preservation is a Phase F+ follow-on (see
/// docs/multimodal.md § append) — CSC `n_minor` is stamped from the
/// file-wide `header.n_obs` so any preserved sidecar would become stale
/// when global n_obs bumps. Regression guard against re-introducing the
/// premature preservation.
#[test]
fn test_append_into_rna_drops_adt_csc() {
    use scx_format::section::SectionType;

    let dir = tempfile::tempdir().unwrap();
    let path = write_multimodal_with_csc(&dir, "partial_csc.scx", 4, 30, 10, true);

    // Verify the seed file has CSC on adt only.
    {
        let reader = ScxReader::open(&path).unwrap();
        assert!(
            reader.header().has_csc(),
            "seed file should advertise has_csc"
        );
        let adt_info = reader.modality_info(2).unwrap();
        let rna_info = reader.modality_info(1).unwrap();
        assert!(adt_info.flags.has_csc(), "adt should have CSC");
        assert!(!rna_info.flags.has_csc(), "rna should not have CSC");
    }

    // Append two new rows into rna.
    let new_obs = sample_obs(2);
    let (new_indptr, new_indices, new_values) = sample_shard_data(2, 30);
    scx_ops::append(
        &path,
        &new_obs,
        &new_indptr,
        &new_indices,
        &new_values,
        ValueEncoding::Uint8,
        &AppendOptions {
            modality_id: 1, // rna
            ..AppendOptions::default()
        },
    )
    .unwrap();

    // After append, every modality's CSC is dropped and the file-wide
    // has_csc flag clears.
    let reader = ScxReader::open(&path).unwrap();
    assert!(
        !reader.header().has_csc(),
        "header.has_csc should clear when every modality's CSC is dropped"
    );
    let adt_info = reader.modality_info(2).unwrap();
    let rna_info = reader.modality_info(1).unwrap();
    assert!(
        !adt_info.flags.has_csc(),
        "append must clear adt's HAS_CSC flag even when appending to rna"
    );
    assert!(!rna_info.flags.has_csc(), "rna must still have no CSC");

    let csc_shards: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CscShard)
        .collect();
    assert!(
        csc_shards.is_empty(),
        "all CSC catalog entries must be dropped after append, found {}",
        csc_shards.len()
    );
}

/// Appending into a modality that owns the only CSC drops it and
/// clears the file-wide HAS_CSC flag.
#[test]
fn test_append_into_adt_drops_csc() {
    use scx_format::section::SectionType;
    let dir = tempfile::tempdir().unwrap();
    let path = write_multimodal_with_csc(&dir, "partial_csc_target.scx", 4, 30, 10, true);

    // Append two rows into adt — drops adt CSC (the only CSC present).
    let new_obs = sample_obs(2);
    let (new_indptr, new_indices, new_values) = sample_shard_data(2, 10);
    scx_ops::append(
        &path,
        &new_obs,
        &new_indptr,
        &new_indices,
        &new_values,
        ValueEncoding::Uint8,
        &AppendOptions {
            modality_id: 2, // adt
            ..AppendOptions::default()
        },
    )
    .unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let adt_info = reader.modality_info(2).unwrap();
    assert!(
        !adt_info.flags.has_csc(),
        "appending to adt should clear adt's HAS_CSC"
    );
    let adt_csc_shards: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CscShard && e.modality_id == 2)
        .collect();
    assert!(
        adt_csc_shards.is_empty(),
        "appending to adt should drop adt's CSC catalog entries"
    );

    // No other modality owned CSC, so the header should be cleared.
    assert!(
        !reader.header().has_csc(),
        "no remaining CSC anywhere => header should clear has_csc"
    );
}

/// Phase 6: multimodal compact applies the global delete mask to all
/// modalities and preserves the modality table.
#[test]
fn test_compact_multimodal_applies_to_all_modalities() {
    use scx_format::section::SectionType;
    let dir = tempfile::tempdir().unwrap();
    let n_obs = 6;
    let path = write_multimodal_with_csc(&dir, "compact_mm.scx", n_obs, 30, 10, false);

    // Delete the first two cells globally.
    scx_ops::mark_deleted(&path, &[0, 1]).unwrap();

    let output = dir.path().join("compact_mm_out.scx");
    scx_ops::compact(&path, &output).unwrap();

    let reader = ScxReader::open(&output).unwrap();
    assert_eq!(reader.header().n_obs, (n_obs - 2) as u64);
    let table = reader.modality_table().expect("modality table preserved");
    assert_eq!(table.entries.len(), 2);
    assert_eq!(table.entries[0].name, "rna");
    assert_eq!(table.entries[1].name, "adt");

    // Each modality should still have at least one CSR shard.
    for modality_id in 1u8..=2u8 {
        let shards: Vec<_> = reader
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CsrShard && e.modality_id == modality_id)
            .collect();
        assert!(
            !shards.is_empty(),
            "modality {modality_id} should have CSR shards after compact"
        );
    }

    // Compacted CSR should have (n_obs - 2) rows per modality.
    for modality_id in 1u8..=2u8 {
        let csr = reader.read_all_csr_shards_for(modality_id).unwrap();
        assert_eq!(
            csr.shape.0,
            n_obs - 2,
            "modality {modality_id} row count after compact"
        );
    }
}

/// Phase 6: multimodal merge concatenates rows from both inputs for
/// every shared modality. Final obs is the row union; per-modality
/// CSR row count matches the new global n_obs.
#[test]
fn test_merge_multimodal_concatenates_per_modality() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_multimodal_with_csc(&dir, "merge_a.scx", 4, 30, 10, false);
    let b = write_multimodal_with_csc(&dir, "merge_b.scx", 3, 30, 10, false);
    let out = dir.path().join("merge_out.scx");

    scx_ops::merge(&[&a, &b], &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.header().n_obs, 7, "merged obs is sum of inputs");
    let table = reader.modality_table().expect("modality table preserved");
    assert_eq!(table.entries.len(), 2);
    for modality_id in 1u8..=2u8 {
        let csr = reader.read_all_csr_shards_for(modality_id).unwrap();
        assert_eq!(
            csr.shape.0, 7,
            "modality {modality_id} should span the merged obs"
        );
    }
}

/// Phase 6: merge rejects inputs with mismatched modality structure.
#[test]
fn test_merge_multimodal_rejects_var_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_multimodal_with_csc(&dir, "mm_a.scx", 4, 30, 10, false);
    // b has a different adt n_vars — should error.
    let b = write_multimodal_with_csc(&dir, "mm_b.scx", 4, 30, 12, false);
    let out = dir.path().join("merge_mismatch.scx");

    let result = scx_ops::merge(&[&a, &b], &out);
    assert!(result.is_err(), "merge should reject modality mismatch");
}

/// Fixture for the merge/compact tests that exercise global obsm and
/// per-modality layers: same shape as `write_multimodal_with_csc` plus
/// a `counts` layer on rna, a `centered` layer on adt, and an `X_pca`
/// global obsm.
fn write_multimodal_with_layers_and_obsm(
    dir: &TempDir,
    filename: &str,
    n_obs: usize,
    rna_n_vars: u64,
    adt_n_vars: u64,
) -> PathBuf {
    use scx_format::modality::ModalityType;
    let path = dir.path().join(filename);
    let header = sample_header(n_obs as u64, rna_n_vars);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();

    // Global obsm.
    let pca: Vec<f64> = (0..n_obs * 2).map(|i| i as f64 * 0.25).collect();
    let pca_schema = Schema::new(vec![
        Field::new("PC1", DataType::Float64, false),
        Field::new("PC2", DataType::Float64, false),
    ]);
    let pca_batch = arrow::array::RecordBatch::try_new(
        Arc::new(pca_schema),
        vec![
            Arc::new(Float64Array::from(pca[..n_obs].to_vec())),
            Arc::new(Float64Array::from(pca[n_obs..].to_vec())),
        ],
    )
    .unwrap();
    writer.write_obsm("X_pca", &pca_batch).unwrap();

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
    writer
        .write_var_for(rna_id, &sample_var(rna_n_vars as usize))
        .unwrap();
    writer
        .write_var_for(adt_id, &sample_var(adt_n_vars as usize))
        .unwrap();
    writer.set_modality_n_vars(rna_id, rna_n_vars).unwrap();
    writer.set_modality_n_vars(adt_id, adt_n_vars).unwrap();

    let (rna_indptr, rna_indices, rna_values) = sample_shard_data(n_obs, rna_n_vars as usize);
    writer
        .write_csr_shard_for(
            rna_id,
            &rna_indptr,
            &rna_indices,
            &rna_values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    let (adt_indptr, adt_indices, adt_values) = sample_shard_data(n_obs, adt_n_vars as usize);
    writer
        .write_csr_shard_for(
            adt_id,
            &adt_indptr,
            &adt_indices,
            &adt_values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();

    // RNA `counts` layer: Float32, value = csr_value * 1.5.
    let mut rna_layer_values = Vec::new();
    for &v in &rna_values {
        rna_layer_values.extend_from_slice(&(v as f32 * 1.5_f32).to_le_bytes());
    }
    writer
        .write_layer_csr_shard_for(
            rna_id,
            "counts",
            0,
            &rna_indptr,
            &rna_indices,
            &rna_layer_values,
            CodecId::None,
            ValueEncoding::Float32,
            0,
        )
        .unwrap();

    // ADT `centered` layer: Float32, value = csr_value - 1.0.
    let mut adt_layer_values = Vec::new();
    for &v in &adt_values {
        adt_layer_values.extend_from_slice(&(v as f32 - 1.0_f32).to_le_bytes());
    }
    writer
        .write_layer_csr_shard_for(
            adt_id,
            "centered",
            0,
            &adt_indptr,
            &adt_indices,
            &adt_layer_values,
            CodecId::None,
            ValueEncoding::Float32,
            0,
        )
        .unwrap();

    writer.finish().unwrap();
    path
}

/// Phase 6 follow-up: multimodal merge must concatenate the global obsm
/// row-wise across inputs (single-modality merge already does this; the
/// initial multimodal path silently dropped it).
#[test]
fn test_merge_multimodal_preserves_global_obsm() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_multimodal_with_layers_and_obsm(&dir, "obsm_a.scx", 4, 30, 10);
    let b = write_multimodal_with_layers_and_obsm(&dir, "obsm_b.scx", 3, 30, 10);
    let out = dir.path().join("merge_obsm_out.scx");

    scx_ops::merge(&[&a, &b], &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    let pca = reader
        .read_obsm("X_pca")
        .expect("merged file should preserve global X_pca obsm");
    assert_eq!(pca.num_rows(), 7, "obsm rows = n_obs_a + n_obs_b");
    assert_eq!(pca.num_columns(), 2);
    let schema = pca.schema();
    assert_eq!(schema.field(0).name(), "PC1");
    assert_eq!(schema.field(1).name(), "PC2");
}

/// Phase 6 follow-up: multimodal merge must copy per-modality layers
/// across inputs. The initial multimodal path skipped layers entirely.
#[test]
fn test_merge_multimodal_preserves_per_modality_layers() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_multimodal_with_layers_and_obsm(&dir, "lay_a.scx", 4, 30, 10);
    let b = write_multimodal_with_layers_and_obsm(&dir, "lay_b.scx", 3, 30, 10);
    let out = dir.path().join("merge_layers_out.scx");

    scx_ops::merge(&[&a, &b], &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    let rna_counts = reader
        .read_layer_for(1, "counts")
        .expect("rna 'counts' layer must survive merge");
    assert_eq!(rna_counts.shape.0, 7, "rna counts row count = merged n_obs");
    let adt_centered = reader
        .read_layer_for(2, "centered")
        .expect("adt 'centered' layer must survive merge");
    assert_eq!(
        adt_centered.shape.0, 7,
        "adt centered row count = merged n_obs"
    );
}

/// Phase 6 follow-up: per-shard streaming refactor of `compact_multimodal`
/// must preserve correctness on per-modality layers.
#[test]
fn test_compact_multimodal_layers_streaming() {
    let dir = tempfile::tempdir().unwrap();
    let n_obs = 6;
    let path = write_multimodal_with_layers_and_obsm(&dir, "compact_lay.scx", n_obs, 30, 10);

    // Drop two cells globally.
    scx_ops::mark_deleted(&path, &[0, 3]).unwrap();
    let out = dir.path().join("compact_lay_out.scx");
    scx_ops::compact(&path, &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.header().n_obs, (n_obs - 2) as u64);
    let rna_counts = reader.read_layer_for(1, "counts").unwrap();
    assert_eq!(rna_counts.shape.0, n_obs - 2);
    let adt_centered = reader.read_layer_for(2, "centered").unwrap();
    assert_eq!(adt_centered.shape.0, n_obs - 2);
}
