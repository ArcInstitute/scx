use arrow::array::{DictionaryArray, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Int8Type, Schema};
use scx_codec::{CodecId, CodecSelection, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::writer::ScxWriter;
use scx_format_io::{ScxReader, ShardHeader, SHARD_HEADER_SIZE};
use scx_ops::AppendOptions;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn sample_header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, n_vars, 0, 16384, 0, 0)
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
    assert!(dv.is_deleted_global(0));
    assert!(dv.is_deleted_global(3));
    assert!(dv.is_deleted_global(5));
    assert!(!dv.is_deleted_global(1));
}

/// Appending a delete onto a file that still carries a legacy **v1** deletion
/// vector must fold the existing v1 (per-shard) deletions to the v2 global
/// representation and preserve them — not drop them. Uses the committed v1
/// golden fixture (n_obs=40, existing deletes {2,5,11,13,28}).
#[test]
fn test_delete_appends_onto_v1_deletion_vectors() {
    let golden = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/reference_files/v1_deletion_vectors.scx");
    if !golden.exists() {
        return; // fixture absent (mirrors the conformance anchor's skip)
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v1_append.scx");
    std::fs::copy(&golden, &path).unwrap();

    // Delete two more rows, disjoint from the existing v1 set.
    let total = scx_ops::mark_deleted(&path, &[0, 39]).unwrap();
    assert_eq!(total, 7, "5 pre-existing v1 deletes + 2 new");

    let reader = ScxReader::open(&path).unwrap();
    let dv = reader.read_deletion_vectors().unwrap().unwrap();
    assert_eq!(dv.total_deleted(), 7);
    // The pre-existing v1 deletions were folded and preserved, plus the new two.
    for row in [0u32, 2, 5, 11, 13, 28, 39] {
        assert!(dv.is_deleted_global(row), "row {row} must be deleted");
    }
    assert!(!dv.is_deleted_global(1), "survivor row 1 must remain");
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

    // Verify append worked, and the whole-file checksum verifies post-append.
    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), 10);
    assert!(
        reader.verify_file_checksum().unwrap(),
        "file_checksum must verify after append"
    );
    drop(reader);

    // Rollback
    scx_ops::rollback(&path).unwrap();

    // Verify original state restored
    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), original_n_obs);
    assert_eq!(reader.header().manifest_sequence, original_seq);

    // The whole-file checksum must still verify after rollback. Rollback
    // repoints the header at the older catalog while the superseded
    // catalog/sections remain trailing past the active catalog, so the checksum
    // extent must run to EOF (not to full_catalog_offset+length) to match the
    // value `finalize_header_with_checksum` stored.
    assert!(
        reader.verify_file_checksum().unwrap(),
        "file_checksum must verify after rollback (extent must reach EOF)"
    );

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

// OE2: a corrupt uns section must propagate as an error through merge, not be
// silently swallowed and conflated with "no uns section".
#[test]
fn test_merge_propagates_corrupt_uns() {
    use std::io::{Seek, SeekFrom, Write};

    let dir = tempfile::tempdir().unwrap();

    // Input with a valid uns section we will corrupt on disk.
    let path1 = dir.path().join("corrupt_uns.scx");
    {
        let header = sample_header(4, 10);
        let mut w = ScxWriter::new(&path1, header).unwrap();
        w.write_obs(&sample_obs(4)).unwrap();
        w.write_var(&sample_var(10)).unwrap();
        let (ip, ix, v) = sample_shard_data(4, 10);
        w.write_csr_shard(&ip, &ix, &v, CodecId::None, ValueEncoding::Uint8, 0)
            .unwrap();
        w.write_uns(&serde_json::json!({"method": "test"})).unwrap();
        w.finish().unwrap();
    }
    let path2 = write_test_file(&dir, "valid.scx", 4, 10, 1);

    // Locate the uns section payload and overwrite it (same length) with
    // bytes that are neither valid JSON nor a valid compressed frame, so
    // read_uns fails to parse rather than reporting absence.
    let (off, len) = {
        let reader = ScxReader::open(&path1).unwrap();
        let e = reader
            .catalog()
            .entries
            .iter()
            .find(|e| e.name == "uns")
            .expect("uns entry present");
        (e.offset, e.length as usize)
    };
    {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(&path1)
            .unwrap();
        f.seek(SeekFrom::Start(off)).unwrap();
        f.write_all(&vec![b'{'; len]).unwrap();
        f.flush().unwrap();
    }

    let output = dir.path().join("merged_corrupt.scx");
    let result = scx_ops::merge(&[path1.as_path(), path2.as_path()], &output);
    assert!(
        result.is_err(),
        "merge should propagate corrupt uns, not swallow it as 'no uns'"
    );
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

/// Regression (MED — compact.rs same bug class as merge layer encoding):
/// compact sampled the X / layer `value_encoding` from the first shard only,
/// then re-encoded every row with it. A file with mixed per-shard encodings
/// (as `scx merge` concat and `scx append` legitimately produce) aborted with
/// `ValueOutOfRange` when a later shard was wider. The encoding is now widened
/// across all shards, so compact succeeds and the wide value round-trips.
#[test]
fn test_compact_widens_value_encoding_across_shards() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed_enc.scx");
    let n_vars = 10usize;
    let header = sample_header(4, n_vars as u64);
    let mut w = ScxWriter::new(&path, header).unwrap();
    w.write_obs(&sample_obs(4)).unwrap();
    w.write_var(&sample_var(n_vars)).unwrap();

    let ip = vec![0u64, 1, 2];
    let ix = vec![0u32, 1];
    // Shard 0 (rows 0-1): Uint8.
    let mut v0 = Vec::new();
    scx_ops::helpers::encode_value(&mut v0, 5.0, ValueEncoding::Uint8).unwrap();
    scx_ops::helpers::encode_value(&mut v0, 7.0, ValueEncoding::Uint8).unwrap();
    w.write_csr_shard(&ip, &ix, &v0, CodecId::None, ValueEncoding::Uint8, 0)
        .unwrap();
    // Shard 1 (rows 2-3): Uint16 with a value > u8::MAX.
    let mut v1 = Vec::new();
    scx_ops::helpers::encode_value(&mut v1, 300.0, ValueEncoding::Uint16).unwrap();
    scx_ops::helpers::encode_value(&mut v1, 9.0, ValueEncoding::Uint16).unwrap();
    w.write_csr_shard(&ip, &ix, &v1, CodecId::None, ValueEncoding::Uint16, 2)
        .unwrap();
    w.write_provenance(vec![ProvenanceEntry {
        timestamp: 1710000000,
        action: "convert".to_string(),
        tool: "test".to_string(),
        params_json: "{}".to_string(),
        input_checksums: vec![],
    }])
    .unwrap();
    w.finish().unwrap();

    let out = dir.path().join("mixed_enc_compact.scx");
    scx_ops::compact(&path, &out)
        .expect("compact must widen X encoding instead of failing with ValueOutOfRange");

    let reader = ScxReader::open(&out).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 4);
    // Row 2's first value (300) must round-trip — it would have wrapped/aborted
    // under the first shard's Uint8 encoding.
    let r2 = csr.indptr[2] as usize;
    assert!(
        (csr.data[r2] - 300.0).abs() < 0.01,
        "expected wide value 300 to round-trip after compact, got {}",
        csr.data[r2]
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

// --- Patch 3: compact preservation of varm / obsp / varp ------------------

/// Build a COO pairwise `RecordBatch` (`row: Int32`, `col: Int32`,
/// `data: Float32`) with `n_rows`/`n_cols` schema metadata, matching the
/// wire format expected by `write_obsp` / `write_varp`.
fn coo_batch(
    rows: Vec<i32>,
    cols: Vec<i32>,
    data: Vec<f32>,
    n: usize,
) -> arrow::array::RecordBatch {
    use std::collections::HashMap;
    let schema = Schema::new_with_metadata(
        vec![
            Field::new("row", DataType::Int32, false),
            Field::new("col", DataType::Int32, false),
            Field::new("data", DataType::Float32, false),
        ],
        HashMap::from([
            ("n_rows".to_string(), n.to_string()),
            ("n_cols".to_string(), n.to_string()),
        ]),
    );
    arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(arrow::array::Int32Array::from(rows)),
            Arc::new(arrow::array::Int32Array::from(cols)),
            Arc::new(arrow::array::Float32Array::from(data)),
        ],
    )
    .unwrap()
}

/// Single-modality file with X plus a global `obsm`, `varm`, `obsp`, and
/// `varp` — used to verify compact preserves every mapping family.
///
/// The `obsp` entries are chosen so that deleting obs 1 and 4 leaves exactly
/// one survivor edge, `(2, 3)`, which remaps to `(1, 2)` in the compacted
/// index space `{0→0, 2→1, 3→2, 5→3}`.
fn write_test_file_with_all_mappings(
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

    // Global obsm: n_obs × 2.
    let emb: Vec<f64> = (0..n_obs * 2).map(|i| i as f64 * 0.1).collect();
    let obsm = arrow::array::RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("PC1", DataType::Float64, false),
            Field::new("PC2", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Float64Array::from(emb[..n_obs].to_vec())),
            Arc::new(Float64Array::from(emb[n_obs..].to_vec())),
        ],
    )
    .unwrap();
    writer.write_obsm("X_pca", &obsm).unwrap();

    // Global varm: n_vars × 2.
    let vemb: Vec<f64> = (0..n_vars * 2).map(|i| i as f64 * 0.5).collect();
    let varm = arrow::array::RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("L1", DataType::Float64, false),
            Field::new("L2", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Float64Array::from(vemb[..n_vars].to_vec())),
            Arc::new(Float64Array::from(vemb[n_vars..].to_vec())),
        ],
    )
    .unwrap();
    writer.write_varm("PCs", &varm).unwrap();

    // obsp (obs×obs COO).
    writer
        .write_obsp(
            "connectivities",
            &coo_batch(
                vec![0, 1, 2, 3, 4, 1],
                vec![1, 2, 3, 4, 5, 4],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 9.0],
                n_obs,
            ),
        )
        .unwrap();

    // varp (var×var COO) — never filtered by compact.
    writer
        .write_varp(
            "corr",
            &coo_batch(vec![0, 3], vec![1, 7], vec![1.5, 2.5], n_vars),
        )
        .unwrap();

    writer.finish().unwrap();
    path
}

/// Compact with no deletions preserves global varm, varp, and obsp.
#[test]
fn test_compact_preserves_varm_varp_obsp_no_deletions() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file_with_all_mappings(&dir, "maps.scx", 6, 10);
    let out = dir.path().join("maps_out.scx");
    scx_ops::compact(&path, &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();

    let varm = reader.read_all_varm().unwrap();
    assert_eq!(varm.get("PCs").expect("varm PCs preserved").num_rows(), 10);

    let varp = reader.read_all_varp().unwrap();
    assert_eq!(varp.get("corr").expect("varp corr preserved").num_rows(), 2);

    let obsp = reader.read_all_obsp().unwrap();
    assert_eq!(
        obsp.get("connectivities")
            .expect("obsp preserved")
            .num_rows(),
        6
    );
}

/// Compact remaps obsp COO through the obs keep-mask under deletions.
#[test]
fn test_compact_filters_obsp_with_deletions() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file_with_all_mappings(&dir, "maps_del.scx", 6, 10);
    scx_ops::mark_deleted(&path, &[1, 4]).unwrap();
    let out = dir.path().join("maps_del_out.scx");
    scx_ops::compact(&path, &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.n_obs(), 4);

    let obsp = reader.read_all_obsp().unwrap();
    let conn = obsp.get("connectivities").expect("obsp preserved");
    // Only the (2,3) edge survives; both endpoints kept, remapped to (1,2).
    assert_eq!(conn.num_rows(), 1);
    let rows = conn
        .column_by_name("row")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Int32Array>()
        .unwrap();
    let cols = conn
        .column_by_name("col")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Int32Array>()
        .unwrap();
    let data = conn
        .column_by_name("data")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Float32Array>()
        .unwrap();
    assert_eq!(rows.value(0), 1);
    assert_eq!(cols.value(0), 2);
    assert_eq!(data.value(0), 3.0);

    // n_rows / n_cols metadata reflect the compacted obs count.
    let md = conn.schema().metadata().clone();
    assert_eq!(md.get("n_rows").map(String::as_str), Some("4"));
    assert_eq!(md.get("n_cols").map(String::as_str), Some("4"));

    // var-axis mappings are unaffected by obs deletions.
    assert_eq!(reader.read_all_varm().unwrap()["PCs"].num_rows(), 10);
    assert_eq!(reader.read_all_varp().unwrap()["corr"].num_rows(), 2);
}

/// Compact remaps a v2 (`Int64` coordinate) obsp under deletions. The v2 wire
/// format self-describes the coordinate width; this pins that compaction
/// accepts Int64 coordinates rather than rejecting them as "not Int32". The
/// surviving axis is small so the output narrows back to Int32.
#[test]
fn test_compact_filters_obsp_int64_with_deletions() {
    use std::collections::HashMap;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("maps_i64.scx");
    let n_obs = 6usize;
    let n_vars = 10usize;
    let header = sample_header(n_obs as u64, n_vars as u64);
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

    // Same edge layout as `write_test_file_with_all_mappings`, but with
    // Int64 coordinate columns (the v2 width).
    let schema = Schema::new_with_metadata(
        vec![
            Field::new("row", DataType::Int64, false),
            Field::new("col", DataType::Int64, false),
            Field::new("data", DataType::Float32, false),
        ],
        HashMap::from([
            ("n_rows".to_string(), n_obs.to_string()),
            ("n_cols".to_string(), n_obs.to_string()),
        ]),
    );
    let obsp = arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(arrow::array::Int64Array::from(vec![0i64, 1, 2, 3, 4, 1])),
            Arc::new(arrow::array::Int64Array::from(vec![1i64, 2, 3, 4, 5, 4])),
            Arc::new(arrow::array::Float32Array::from(vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 9.0,
            ])),
        ],
    )
    .unwrap();
    writer.write_obsp("connectivities", &obsp).unwrap();
    writer.finish().unwrap();

    scx_ops::mark_deleted(&path, &[1, 4]).unwrap();
    let out = dir.path().join("maps_i64_out.scx");
    scx_ops::compact(&path, &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    let obsp = reader.read_all_obsp().unwrap();
    let conn = obsp.get("connectivities").expect("obsp preserved");
    // Only the (2,3) edge survives, remapped to (1,2). Small axis → Int32 out.
    assert_eq!(conn.num_rows(), 1);
    let rows = conn
        .column_by_name("row")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Int32Array>()
        .expect("output narrowed to Int32 for a small compacted axis");
    let cols = conn
        .column_by_name("col")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Int32Array>()
        .unwrap();
    assert_eq!(rows.value(0), 1);
    assert_eq!(cols.value(0), 2);
    let md = conn.schema().metadata().clone();
    assert_eq!(md.get("n_rows").map(String::as_str), Some("4"));
    assert_eq!(md.get("n_cols").map(String::as_str), Some("4"));
}

/// Multimodal file whose RNA modality carries a *sharded* per-modality obsm
/// (`ObsmEmbeddingShard`, type 20) and a per-modality varm.
fn write_multimodal_with_per_modality_mappings(
    dir: &TempDir,
    filename: &str,
    n_obs: usize,
    rna_n_vars: u64,
) -> PathBuf {
    use scx_format_io::modality::ModalityType;
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
    writer
        .write_var_for(rna_id, &sample_var(rna_n_vars as usize))
        .unwrap();
    writer.set_modality_n_vars(rna_id, rna_n_vars).unwrap();
    let (indptr, indices, values) = sample_shard_data(n_obs, rna_n_vars as usize);
    writer
        .write_csr_shard_for(
            rna_id,
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();

    // Per-modality obsm as a SHARDED section (forces ObsmEmbeddingShard).
    let emb: Vec<f64> = (0..n_obs * 2).map(|i| i as f64 * 0.1).collect();
    let obsm = arrow::array::RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("UMAP1", DataType::Float64, false),
            Field::new("UMAP2", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Float64Array::from(emb[..n_obs].to_vec())),
            Arc::new(Float64Array::from(emb[n_obs..].to_vec())),
        ],
    )
    .unwrap();
    writer
        .write_obsm_shard_for(rna_id, "X_umap", 0, 0, n_obs as u64, n_obs as u64, &obsm)
        .unwrap();

    // Per-modality varm.
    let nv = rna_n_vars as usize;
    let vemb: Vec<f64> = (0..nv * 2).map(|i| i as f64 * 0.3).collect();
    let varm = arrow::array::RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("L1", DataType::Float64, false),
            Field::new("L2", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Float64Array::from(vemb[..nv].to_vec())),
            Arc::new(Float64Array::from(vemb[nv..].to_vec())),
        ],
    )
    .unwrap();
    writer
        .write_varm_shard_for(rna_id, "PCs", 0, 0, rna_n_vars, rna_n_vars, &varm)
        .unwrap();

    writer.finish().unwrap();
    path
}

/// Compact preserves a sharded per-modality obsm and a per-modality varm.
#[test]
fn test_compact_preserves_per_modality_sharded_obsm_and_varm() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multimodal_with_per_modality_mappings(&dir, "mm_maps.scx", 6, 8);
    let out = dir.path().join("mm_maps_out.scx");
    scx_ops::compact(&path, &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert!(reader.is_multimodal());
    // Sharded per-modality obsm survives the type-20 discovery fix.
    assert_eq!(reader.read_obsm_for(1, "X_umap").unwrap().num_rows(), 6);
    // Per-modality varm survives compaction.
    assert_eq!(reader.read_varm_for(1, "PCs").unwrap().num_rows(), 8);
}

/// Compact row-filters a per-modality obsm under deletions; varm is
/// var-axis and stays full.
#[test]
fn test_compact_filters_per_modality_obsm_with_deletions() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multimodal_with_per_modality_mappings(&dir, "mm_del.scx", 6, 8);
    scx_ops::mark_deleted(&path, &[1, 4]).unwrap();
    let out = dir.path().join("mm_del_out.scx");
    scx_ops::compact(&path, &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.read_obsm_for(1, "X_umap").unwrap().num_rows(), 4);
    assert_eq!(reader.read_varm_for(1, "PCs").unwrap().num_rows(), 8);
}

/// Multimodal file carrying BOTH per-modality mappings (sharded obsm + varm on
/// the RNA modality) AND file-level global `obsm`/`varm`/`varp`/`obsp`
/// (`modality_id == 0`). Used to verify multimodal compact preserves the
/// globals without re-emitting the per-modality entries as spurious globals.
///
/// The global `obsp` edge layout matches the single-modality fixture: deleting
/// obs 1 and 4 leaves exactly one survivor edge, `(2, 3)` → `(1, 2)`.
fn write_multimodal_with_global_and_per_modality_mappings(
    dir: &TempDir,
    filename: &str,
    n_obs: usize,
    rna_n_vars: u64,
) -> PathBuf {
    use scx_format_io::modality::ModalityType;
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
    writer
        .write_var_for(rna_id, &sample_var(rna_n_vars as usize))
        .unwrap();
    writer.set_modality_n_vars(rna_id, rna_n_vars).unwrap();
    let (indptr, indices, values) = sample_shard_data(n_obs, rna_n_vars as usize);
    writer
        .write_csr_shard_for(
            rna_id,
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();

    // Per-modality sharded obsm + varm on the RNA modality (modality_id 1).
    let emb: Vec<f64> = (0..n_obs * 2).map(|i| i as f64 * 0.1).collect();
    let pm_obsm = arrow::array::RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("UMAP1", DataType::Float64, false),
            Field::new("UMAP2", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Float64Array::from(emb[..n_obs].to_vec())),
            Arc::new(Float64Array::from(emb[n_obs..].to_vec())),
        ],
    )
    .unwrap();
    writer
        .write_obsm_shard_for(rna_id, "X_umap", 0, 0, n_obs as u64, n_obs as u64, &pm_obsm)
        .unwrap();
    let nv = rna_n_vars as usize;
    let pm_vemb: Vec<f64> = (0..nv * 2).map(|i| i as f64 * 0.3).collect();
    let pm_varm = arrow::array::RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("L1", DataType::Float64, false),
            Field::new("L2", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Float64Array::from(pm_vemb[..nv].to_vec())),
            Arc::new(Float64Array::from(pm_vemb[nv..].to_vec())),
        ],
    )
    .unwrap();
    writer
        .write_varm_shard_for(rna_id, "PCs", 0, 0, rna_n_vars, rna_n_vars, &pm_varm)
        .unwrap();

    // Global mappings (modality_id 0) — written at top level.
    let gemb: Vec<f64> = (0..n_obs * 2).map(|i| i as f64 * 0.2).collect();
    let g_obsm = arrow::array::RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("PC1", DataType::Float64, false),
            Field::new("PC2", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Float64Array::from(gemb[..n_obs].to_vec())),
            Arc::new(Float64Array::from(gemb[n_obs..].to_vec())),
        ],
    )
    .unwrap();
    writer.write_obsm("X_pca_g", &g_obsm).unwrap();

    let gvemb: Vec<f64> = (0..nv * 2).map(|i| i as f64 * 0.7).collect();
    let g_varm = arrow::array::RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("G1", DataType::Float64, false),
            Field::new("G2", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Float64Array::from(gvemb[..nv].to_vec())),
            Arc::new(Float64Array::from(gvemb[nv..].to_vec())),
        ],
    )
    .unwrap();
    writer.write_varm("PCs_g", &g_varm).unwrap();

    writer
        .write_obsp(
            "conn_g",
            &coo_batch(
                vec![0, 1, 2, 3, 4, 1],
                vec![1, 2, 3, 4, 5, 4],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 9.0],
                n_obs,
            ),
        )
        .unwrap();
    writer
        .write_varp(
            "corr_g",
            &coo_batch(vec![0, 3], vec![1, 7], vec![1.5, 2.5], nv),
        )
        .unwrap();

    writer.finish().unwrap();
    path
}

/// Count catalog entries that are global (`modality_id == 0`) and live under a
/// per-modality `{prefix}/{modality}/` path — i.e. spurious globals re-emitted
/// from per-modality sections. Should always be zero after compact.
fn spurious_global_under(reader: &ScxReader, prefix: &str) -> usize {
    reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.modality_id == 0 && e.name.starts_with(prefix))
        .count()
}

/// Multimodal compact (no deletions) preserves global obsm/varm/varp/obsp and
/// leaves per-modality mappings intact, without re-emitting per-modality
/// sections as spurious globals.
#[test]
fn test_compact_multimodal_preserves_global_varm_varp_obsp() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multimodal_with_global_and_per_modality_mappings(&dir, "mm_glob.scx", 6, 8);
    let out = dir.path().join("mm_glob_out.scx");
    scx_ops::compact(&path, &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert!(reader.is_multimodal());

    // Globals preserved.
    assert_eq!(
        reader.read_all_obsm().unwrap()["X_pca_g"].num_rows(),
        6,
        "global obsm preserved"
    );
    assert_eq!(
        reader.read_all_varm().unwrap()["PCs_g"].num_rows(),
        8,
        "global varm preserved"
    );
    assert_eq!(
        reader.read_all_varp().unwrap()["corr_g"].num_rows(),
        2,
        "global varp preserved"
    );
    assert_eq!(
        reader.read_all_obsp().unwrap()["conn_g"].num_rows(),
        6,
        "global obsp preserved"
    );

    // Per-modality mappings intact.
    assert_eq!(reader.read_obsm_for(1, "X_umap").unwrap().num_rows(), 6);
    assert_eq!(reader.read_varm_for(1, "PCs").unwrap().num_rows(), 8);

    // No per-modality embedding re-emitted as a global (modality_id 0) section.
    assert_eq!(spurious_global_under(&reader, "obsm/rna/"), 0);
    assert_eq!(spurious_global_under(&reader, "varm/rna/"), 0);
}

/// Multimodal compact under deletions remaps the global obsp COO and
/// row-filters obs-axis mappings; var-axis globals stay full.
#[test]
fn test_compact_multimodal_filters_global_obsp_with_deletions() {
    let dir = tempfile::tempdir().unwrap();
    let path =
        write_multimodal_with_global_and_per_modality_mappings(&dir, "mm_glob_del.scx", 6, 8);
    scx_ops::mark_deleted(&path, &[1, 4]).unwrap();
    let out = dir.path().join("mm_glob_del_out.scx");
    scx_ops::compact(&path, &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.n_obs(), 4);

    // Global obsp: only the (2,3) edge survives, remapped to (1,2).
    let obsp = reader.read_all_obsp().unwrap();
    let conn = obsp.get("conn_g").expect("global obsp preserved");
    assert_eq!(conn.num_rows(), 1);
    let rows = conn
        .column_by_name("row")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Int32Array>()
        .unwrap();
    let cols = conn
        .column_by_name("col")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Int32Array>()
        .unwrap();
    assert_eq!(rows.value(0), 1);
    assert_eq!(cols.value(0), 2);
    let md = conn.schema().metadata().clone();
    assert_eq!(md.get("n_rows").map(String::as_str), Some("4"));
    assert_eq!(md.get("n_cols").map(String::as_str), Some("4"));

    // Obs-axis globals + per-modality obsm row-filtered; var-axis stays full.
    assert_eq!(reader.read_all_obsm().unwrap()["X_pca_g"].num_rows(), 4);
    assert_eq!(reader.read_obsm_for(1, "X_umap").unwrap().num_rows(), 4);
    assert_eq!(reader.read_all_varm().unwrap()["PCs_g"].num_rows(), 8);
    assert_eq!(reader.read_all_varp().unwrap()["corr_g"].num_rows(), 2);
    assert_eq!(reader.read_varm_for(1, "PCs").unwrap().num_rows(), 8);
}

/// Read the codec_id from the most recently appended CSR shard.
fn last_appended_shard_codec(path: &std::path::Path) -> CodecId {
    use scx_format_io::section::SectionType;
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
    let expected = scx_format_io::select_codec_for_modality(
        &values,
        ValueEncoding::Uint8,
        scx_format_io::ModalityType::Rna,
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

/// Write a file whose `"normalized"` layer is stored with `layer_enc`, the
/// first layer value set to `peak` and the rest to 1.0. The X shard is always
/// `Uint8`. Used to exercise the merge layer value-encoding widening path.
fn write_test_file_with_layer_enc(
    dir: &TempDir,
    filename: &str,
    n_obs: usize,
    n_vars: usize,
    layer_enc: ValueEncoding,
    peak: f32,
) -> PathBuf {
    let path = dir.path().join(filename);
    let header = sample_header(n_obs as u64, n_vars as u64);
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

    let mut layer_bytes = Vec::new();
    for j in 0..values.len() {
        let v = if j == 0 { peak } else { 1.0 };
        scx_ops::helpers::encode_value(&mut layer_bytes, v, layer_enc).unwrap();
    }
    writer
        .write_layer_csr_shard(
            &indptr,
            &indices,
            &layer_bytes,
            CodecId::None,
            layer_enc,
            0,
            "normalized",
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
}

/// Regression (MED — merge.rs non-sorted layer merge): the concat layer merge
/// previously sampled the layer value encoding from `readers[0]`'s first shard
/// only, so merging a `Uint8`-layer file with a `Uint16`-layer file holding a
/// value > 255 aborted with `ValueOutOfRange`. The encoding is now widened
/// across all inputs, so the merge succeeds and the wide value round-trips.
#[test]
fn test_merge_widens_layer_value_encoding_across_inputs() {
    let dir = tempfile::tempdir().unwrap();
    // readers[0]: narrow Uint8 layer. readers[1]: Uint16 layer with 300 > u8::MAX.
    let path1 = write_test_file_with_layer_enc(&dir, "lw1.scx", 4, 10, ValueEncoding::Uint8, 1.0);
    let path2 =
        write_test_file_with_layer_enc(&dir, "lw2.scx", 4, 10, ValueEncoding::Uint16, 300.0);

    let output = dir.path().join("lw_merged.scx");
    scx_ops::merge(&[path1.as_path(), path2.as_path()], &output)
        .expect("merge must widen layer encoding instead of failing with ValueOutOfRange");

    let reader = ScxReader::open(&output).unwrap();
    assert_eq!(reader.n_obs(), 8);

    let layer = reader.read_layer("normalized").unwrap();
    assert_eq!(layer.shape.0, 8);
    // readers[1]'s first value (300) lands at the first nnz of row 4 (rows 0-3
    // came from readers[0]). Its nnz offset = total nnz of readers[0] = 8.
    assert!(
        (layer.data[8] - 300.0).abs() < 0.01,
        "expected the wide value 300 to round-trip, got {}",
        layer.data[8]
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
        .filter(|e| e.section_type == scx_format_io::section::SectionType::CscShard)
        .count();
    assert_eq!(csc_count, 0);

    // CSR readback unchanged.
    let csr = r.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 10);
}

/// End-to-end CSC freshness lifecycle: a fresh CSC file is fresh
/// (`data_generation == csc_build_generation`); append bumps the data
/// generation and drops CSC; `build-csc` preserves that bumped generation
/// and rebuilds a matching sidecar, so the rebuilt file opens fresh via
/// `BackedCscReader`.
#[test]
fn test_csc_generation_lifecycle_append_then_rebuild() {
    use scx_format_io::backed::BackedCscReader;

    let dir = tempfile::tempdir().unwrap();
    let path = write_csc_test_file(&dir, "csc_gen.scx", 6, 8, 4);

    // Fresh file: matched generations, backed CSC reader opens.
    {
        let r = ScxReader::open(&path).unwrap();
        let cat = r.catalog();
        assert_eq!(cat.data_generation, cat.csc_build_generation);
        let gen0 = cat.data_generation;
        assert!(BackedCscReader::new(r, 0).is_ok());

        // Append bumps the data generation and drops CSC.
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

        let r2 = ScxReader::open(&path).unwrap();
        assert!(!r2.header().has_csc());
        assert_eq!(
            r2.catalog().data_generation,
            gen0 + 1,
            "append must bump data_generation"
        );
        assert_eq!(r2.catalog().csc_build_generation, 0);
    }

    // build-csc preserves the (bumped) data generation and rebuilds a
    // fresh sidecar matching it.
    let rebuilt = dir.path().join("csc_gen_rebuilt.scx");
    scx_ops::run_build_csc(&path, &rebuilt, "4G", false, 4, None).unwrap();

    let r3 = ScxReader::open(&rebuilt).unwrap();
    assert!(r3.header().has_csc());
    let cat3 = r3.catalog();
    assert_eq!(
        cat3.data_generation, cat3.csc_build_generation,
        "rebuilt sidecar must match the current data generation"
    );
    // The rebuilt file opens via the backed CSC reader (not stale).
    assert!(BackedCscReader::new(r3, 0).is_ok());
}

// OE1: rolling back an append that dropped the CSC sidecar must resync the
// CSC shard count and HAS_CSC flag from the restored catalog — not leave the
// header advertising absent shards or clear while shards are present.
#[test]
fn test_rollback_restores_csc_count_and_flag() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_csc_test_file(&dir, "csc_rollback.scx", 6, 8, 4);

    // Fresh file: CSC sidecar present (2 shards: 8 vars / 4 cols-per-shard).
    let reader = ScxReader::open(&path).unwrap();
    assert!(reader.header().has_csc());
    let original_n_csc = reader.header().n_csc_shards;
    assert_eq!(original_n_csc, 2);
    drop(reader);

    // Append drops the CSC sidecar and clears the flag.
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
    let reader = ScxReader::open(&path).unwrap();
    assert!(!reader.header().has_csc());
    assert_eq!(reader.header().n_csc_shards, 0);
    drop(reader);

    // Rollback must restore the CSC count and flag from the prior catalog.
    scx_ops::rollback(&path).unwrap();
    let reader = ScxReader::open(&path).unwrap();
    assert!(
        reader.header().has_csc(),
        "rollback should restore the HAS_CSC flag"
    );
    assert_eq!(
        reader.header().n_csc_shards,
        original_n_csc,
        "rollback should restore the CSC shard count"
    );
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
        .filter(|e| e.section_type == scx_format_io::section::SectionType::CscShard)
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
        .filter(|e| e.section_type == scx_format_io::section::SectionType::CscShard)
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

/// Multimodal append is deferred (review finding #3): appending into one
/// modality bumps the global `n_obs` while writing shards for only that
/// modality, leaving siblings under-covering the obs axis → an unreadable
/// file. `append` must reject any multimodal target with
/// `OpsError::MultimodalUnsupported` before touching the file.
#[test]
fn test_append_for_modality_rejected() {
    use scx_format_io::modality::ModalityType;
    use scx_format_io::section::SectionType;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multimodal_append.scx");

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
        // formed before the append attempt.
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

    // Attempt to append two new rows into the "adt" modality — must be rejected.
    let new_obs = sample_obs(2);
    let (new_indptr, new_indices, new_values) = sample_shard_data(2, adt_n_vars as usize);
    let err = scx_ops::append(
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
    .expect_err("multimodal append must be rejected");
    assert!(
        matches!(
            err,
            scx_ops::OpsError::MultimodalUnsupported { op: "append" }
        ),
        "expected MultimodalUnsupported, got {err:?}"
    );

    // The file must be left untouched: n_obs unchanged, adt still has its
    // single seed shard (the guard fires before any write).
    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(
        reader.n_obs(),
        4,
        "n_obs must be unchanged after a rejected append"
    );
    let adt_shards = reader
        .catalog()
        .shards(SectionType::CsrShard)
        .into_iter()
        .filter(|e| e.modality_id == 2)
        .count();
    assert_eq!(
        adt_shards, 1,
        "adt shard count must be unchanged after a rejected append"
    );
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

/// Regression: a source file whose shards carry *different* value encodings
/// (legal — the auto-codec selects `value_encoding` per shard from that
/// shard's value distribution, so multi-shard census files routinely mix
/// e.g. Scx1 and Zstd) must append cleanly. `append_from_reader` previously
/// derived one encoding from the first shard and rejected any shard that
/// differed with a `ShapeMismatch` ("value_encoding N differs from first
/// shard's M"), which broke `pyscx.append` / fragment_ops on census-scale
/// inputs. Each shard's own encoding must be preserved instead.
#[test]
fn test_streaming_append_mixed_value_encoding_source() {
    let dir = tempfile::tempdir().unwrap();
    let target = write_test_file(&dir, "mixed_enc_target.scx", 4, 4, 1);

    // Build a 2-shard source by hand: shard 0 Uint8, shard 1 Uint16 (values
    // > 255 to justify the wider encoding), both over n_vars = 4.
    let src_path = dir.path().join("mixed_enc_source.scx");
    let header = sample_header(4, 4);
    let mut w = ScxWriter::new(&src_path, header).unwrap();
    w.write_obs(&sample_obs(4)).unwrap();
    w.write_var(&sample_var(4)).unwrap();
    // shard 0 — rows 0..2, Uint8.
    w.write_csr_shard(
        &[0u64, 1, 2],
        &[0u32, 1],
        &[7u8, 9u8],
        CodecId::None,
        ValueEncoding::Uint8,
        0,
    )
    .unwrap();
    // shard 1 — rows 2..4, Uint16.
    let mut v1 = Vec::new();
    v1.extend_from_slice(&300u16.to_le_bytes());
    v1.extend_from_slice(&1000u16.to_le_bytes());
    w.write_csr_shard(
        &[0u64, 1, 2],
        &[2u32, 3],
        &v1,
        CodecId::None,
        ValueEncoding::Uint16,
        2,
    )
    .unwrap();
    w.write_provenance(vec![ProvenanceEntry {
        timestamp: 1710000000,
        action: "convert".to_string(),
        tool: "test".to_string(),
        params_json: "{}".to_string(),
        input_checksums: vec![],
    }])
    .unwrap();
    w.finish().unwrap();

    let src_reader = ScxReader::open(&src_path).unwrap();
    // Must NOT error on the heterogeneous encodings.
    scx_ops::append_from_reader(&target, &src_reader, &AppendOptions::default(), 0).unwrap();
    drop(src_reader);

    let reader = ScxReader::open(&target).unwrap();
    assert_eq!(reader.n_obs(), 8); // 4 target + 4 source
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 8);
    // The Uint16-shard values must survive the append (round-trip fidelity).
    let data: Vec<i64> = csr.data.iter().map(|&v| v as i64).collect();
    assert!(
        data.contains(&300) && data.contains(&1000),
        "Uint16-shard values must survive append: {data:?}"
    );
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
    let appended: Vec<&scx_format_io::FullCatalogEntry> = reader
        .catalog()
        .shards(scx_format_io::section::SectionType::CsrShard)
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
            .shards(scx_format_io::section::SectionType::CsrShard);
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
        .shards(scx_format_io::section::SectionType::CsrShard)
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
            .shards(scx_format_io::section::SectionType::CsrShard);
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
        .shards(scx_format_io::section::SectionType::CsrShard)
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
        .filter(|e| e.section_type == scx_format_io::section::SectionType::CscShard)
        .count();
    assert_eq!(csc_count, 0);
}

/// Streaming append (`append_from_reader`) into a multimodal target must be
/// rejected for the same reason as the bulk path (review finding #3): it would
/// leave sibling modalities under-covering the global obs axis.
#[test]
fn test_streaming_append_multimodal_rejected() {
    use scx_format_io::modality::ModalityType;
    use scx_format_io::section::SectionType;

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
    let err = scx_ops::append_from_reader(
        &path,
        &src_reader,
        &AppendOptions {
            modality_id: 2,
            ..AppendOptions::default()
        },
        0, // source is single-modality
    )
    .expect_err("multimodal streaming append must be rejected");
    assert!(
        matches!(
            err,
            scx_ops::OpsError::MultimodalUnsupported { op: "append" }
        ),
        "expected MultimodalUnsupported, got {err:?}"
    );
    drop(src_reader);

    // File left untouched: n_obs unchanged, adt still has its single seed shard.
    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(
        reader.n_obs(),
        4,
        "n_obs must be unchanged after a rejected append"
    );
    let adt_shards = reader
        .catalog()
        .shards(SectionType::CsrShard)
        .into_iter()
        .filter(|e| e.modality_id == 2)
        .count();
    assert_eq!(
        adt_shards, 1,
        "adt shard count must be unchanged after a rejected append"
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
    use scx_format_io::modality::ModalityType;
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

// Note: per-modality CSC-drop-on-append tests were removed when multimodal
// append was deferred (review finding #3 — it corrupted sibling obs coverage).
// Single-modality CSC drop on append stays covered by
// `test_append_drops_csc_from_input`. Multimodal CSC preservation across
// compact is covered by the multimodal-compact tests below.

/// Phase 6: multimodal compact applies the global delete mask to all
/// modalities and preserves the modality table.
#[test]
fn test_compact_multimodal_applies_to_all_modalities() {
    use scx_format_io::section::SectionType;
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
    use scx_format_io::modality::ModalityType;
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

/// Multimodal fixture with explicit per-modality var batches, optional
/// global uns, and optional per-modality uns. Used by the MergeOptions
/// regression tests below to construct inputs that diverge in only the
/// targeted axis.
#[allow(clippy::too_many_arguments)]
fn write_multimodal_with_options_fixture(
    dir: &TempDir,
    filename: &str,
    n_obs: usize,
    rna_var: &arrow::array::RecordBatch,
    adt_var: &arrow::array::RecordBatch,
    global_uns: Option<&serde_json::Value>,
    rna_uns: Option<&serde_json::Value>,
    adt_uns: Option<&serde_json::Value>,
) -> PathBuf {
    use scx_format_io::modality::ModalityType;
    let path = dir.path().join(filename);
    let header = sample_header(n_obs as u64, rna_var.num_rows() as u64);
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
    writer.write_var_for(rna_id, rna_var).unwrap();
    writer.write_var_for(adt_id, adt_var).unwrap();
    writer
        .set_modality_n_vars(rna_id, rna_var.num_rows() as u64)
        .unwrap();
    writer
        .set_modality_n_vars(adt_id, adt_var.num_rows() as u64)
        .unwrap();
    let (rna_indptr, rna_indices, rna_values) = sample_shard_data(n_obs, rna_var.num_rows());
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
    let (adt_indptr, adt_indices, adt_values) = sample_shard_data(n_obs, adt_var.num_rows());
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
    if let Some(g) = global_uns {
        writer.write_uns(g).unwrap();
    }
    if let Some(u) = rna_uns {
        writer.write_uns_for(rna_id, u).unwrap();
    }
    if let Some(u) = adt_uns {
        writer.write_uns_for(adt_id, u).unwrap();
    }
    writer.finish().unwrap();
    path
}

/// Build an alternative var batch with the same n_vars but a reordered
/// gene_id column — used to trigger per-modality `VarMismatch`.
fn sample_var_reordered(n: usize) -> arrow::array::RecordBatch {
    let ids: Vec<String> = (0..n).rev().map(|i| format!("gene_{i}")).collect();
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

#[test]
fn test_merge_multimodal_per_modality_var_mismatch_errors_by_default() {
    // Per-modality var identity check fires when adt's gene order
    // differs between inputs, even though n_vars matches. Before the
    // MergeOptions threading landed, multimodal silently took input 0's
    // var and produced a column-axis-corrupted file.
    let dir = tempfile::tempdir().unwrap();
    let rna = sample_var(8);
    let adt_a = sample_var(4);
    let adt_b = sample_var_reordered(4);
    let a = write_multimodal_with_options_fixture(
        &dir,
        "mm_var_a.scx",
        3,
        &rna,
        &adt_a,
        None,
        None,
        None,
    );
    let b = write_multimodal_with_options_fixture(
        &dir,
        "mm_var_b.scx",
        3,
        &rna,
        &adt_b,
        None,
        None,
        None,
    );
    let out = dir.path().join("mm_var_out.scx");
    let err = scx_ops::merge(&[&a, &b], &out).unwrap_err();
    assert!(
        matches!(err, scx_ops::OpsError::VarMismatch { .. }),
        "expected VarMismatch on per-modality var disagreement, got: {err:?}"
    );
    // With assume_identical_var the merge proceeds and uses input 0's var.
    let opts = scx_ops::MergeOptions {
        assume_identical_var: true,
        ..Default::default()
    };
    scx_ops::merge_with_options(&[a.as_path(), b.as_path()], &out, &opts).unwrap();
    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.header().n_obs, 6);
}

#[test]
fn test_merge_multimodal_uns_policy_namespace_applies_at_both_levels() {
    // UnsPolicy::Namespace wraps every input's payload under
    // `input_N`. Multimodal applies the policy independently to global
    // uns and to each modality's uns; both levels must show the wrap.
    let dir = tempfile::tempdir().unwrap();
    let rna = sample_var(6);
    let adt = sample_var(10);
    let g_a = serde_json::json!({"global": "A"});
    let g_b = serde_json::json!({"global": "B"});
    let rna_a = serde_json::json!({"layer": "rna_A"});
    let rna_b = serde_json::json!({"layer": "rna_B"});
    let a = write_multimodal_with_options_fixture(
        &dir,
        "mm_uns_a.scx",
        3,
        &rna,
        &adt,
        Some(&g_a),
        Some(&rna_a),
        None,
    );
    let b = write_multimodal_with_options_fixture(
        &dir,
        "mm_uns_b.scx",
        3,
        &rna,
        &adt,
        Some(&g_b),
        Some(&rna_b),
        None,
    );
    let out = dir.path().join("mm_uns_out.scx");
    let opts = scx_ops::MergeOptions {
        uns_policy: scx_ops::UnsPolicy::Namespace,
        ..Default::default()
    };
    scx_ops::merge_with_options(&[a.as_path(), b.as_path()], &out, &opts).unwrap();
    let reader = ScxReader::open(&out).unwrap();
    let g = reader.read_uns().unwrap();
    assert_eq!(g.get("input_0"), Some(&g_a));
    assert_eq!(g.get("input_1"), Some(&g_b));
    let rna_combined = reader.read_uns_for(1).unwrap();
    assert_eq!(rna_combined.get("input_0"), Some(&rna_a));
    assert_eq!(rna_combined.get("input_1"), Some(&rna_b));
    // adt had no uns in either input — no section should land.
    assert!(reader.read_uns_for(2).is_err());
    // Provenance records the policy + the conflict counter.
    let prov = reader.read_provenance().unwrap();
    let merge_op = prov.operations.last().unwrap();
    assert_eq!(merge_op.action, "merge");
    assert!(
        merge_op
            .params_json
            .contains("\"uns_policy\":\"namespace\""),
        "params_json = {}",
        merge_op.params_json
    );
    assert!(
        merge_op
            .params_json
            .contains("\"assume_identical_obs\":false"),
        "params_json = {}",
        merge_op.params_json
    );
}

#[test]
fn test_merge_multimodal_shard_target_override() {
    // `shard_target_rows` override on multimodal merge splits the
    // global obs into one shard per ceil(n_obs / target) rows.
    let dir = tempfile::tempdir().unwrap();
    let rna = sample_var(6);
    let adt = sample_var(10);
    let a = write_multimodal_with_options_fixture(
        &dir,
        "mm_shard_a.scx",
        10,
        &rna,
        &adt,
        None,
        None,
        None,
    );
    let b = write_multimodal_with_options_fixture(
        &dir,
        "mm_shard_b.scx",
        10,
        &rna,
        &adt,
        None,
        None,
        None,
    );
    let out = dir.path().join("mm_shard_out.scx");
    let opts = scx_ops::MergeOptions {
        shard_target_rows: Some(4),
        ..Default::default()
    };
    scx_ops::merge_with_options(&[a.as_path(), b.as_path()], &out, &opts).unwrap();
    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.header().n_obs, 20);
    // Each 10-row input gets sliced into ceil(10/4) = 3 shards
    // (4 + 4 + 2), so 2 inputs → 6 obs shards total.
    assert_eq!(reader.obs_metadata_shard_count(), 6);
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

// ---------------------------------------------------------------------------
// compact --reshape-obs (task 6c): legacy single-section obs → sharded
// ---------------------------------------------------------------------------

/// Write a single-section-obs SCX file with a caller-chosen
/// `shard_target_rows` so `compact --reshape-obs` produces multiple obs
/// shards from a small fixture.
fn write_single_section_obs_file(
    dir: &TempDir,
    filename: &str,
    n_obs: usize,
    n_vars: usize,
    shard_target_rows: u32,
) -> PathBuf {
    let path = dir.path().join(filename);
    let mut header = sample_header(n_obs as u64, n_vars as u64);
    header.shard_target_rows = shard_target_rows;
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
    writer.finish().unwrap();
    path
}

fn no_index_options() -> scx_engine::ConversionPredicateIndexOptions {
    scx_engine::ConversionPredicateIndexOptions {
        index_obs: Vec::new(),
        index_var: Vec::new(),
        index_preset: None,
        index_auto_threshold: 0,
    }
}

#[test]
fn compact_reshape_obs_converts_legacy_single_section_to_shards() {
    let dir = tempfile::tempdir().unwrap();
    // 10 rows, shard_target_rows = 4 → ceil(10/4) = 3 obs shards.
    let input = write_single_section_obs_file(&dir, "legacy.scx", 10, 10, 4);

    // Precondition: input is legacy single-section obs.
    let reader = ScxReader::open(&input).unwrap();
    assert_eq!(reader.obs_metadata_shard_count(), 0);
    drop(reader);

    let out = dir.path().join("reshaped.scx");
    scx_ops::compact_with_index_options(&input, &out, &no_index_options(), true).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.obs_metadata_shard_count(), 3);
    // Legacy single `obs` section must be absent (ObsVarLayout forbids mixing).
    assert!(reader.catalog().get("obs").is_none());

    // obs round-trips intact through the assembled read path.
    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 10);
    let ids = obs
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for i in 0..10 {
        assert_eq!(ids.value(i), format!("cell_{i}"));
    }

    // Provenance records the migration.
    let prov = reader.read_provenance().unwrap();
    let last = prov.operations.last().unwrap();
    assert_eq!(last.action, "compact");
    assert!(last.params_json.contains("\"reshape_obs\":true"));
}

#[test]
fn compact_without_reshape_keeps_single_section_obs() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_single_section_obs_file(&dir, "legacy.scx", 10, 10, 4);

    let out = dir.path().join("compacted.scx");
    scx_ops::compact_with_index_options(&input, &out, &no_index_options(), false).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.obs_metadata_shard_count(), 0);
    assert_eq!(reader.read_obs().unwrap().num_rows(), 10);
}

#[test]
fn compact_reshape_obs_is_idempotent_on_sharded_input() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_single_section_obs_file(&dir, "legacy.scx", 10, 10, 4);

    let once = dir.path().join("once.scx");
    scx_ops::compact_with_index_options(&input, &once, &no_index_options(), true).unwrap();
    let twice = dir.path().join("twice.scx");
    scx_ops::compact_with_index_options(&once, &twice, &no_index_options(), true).unwrap();

    let reader = ScxReader::open(&twice).unwrap();
    assert_eq!(reader.obs_metadata_shard_count(), 3);
    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 10);
}

/// Write a two-modality (rna + adt) SCX file with single-section obs and a
/// caller-chosen `shard_target_rows`, so `compact --reshape-obs` exercises
/// the multimodal write path (`compact_multimodal`) and produces multiple
/// obs shards from a small fixture.
fn write_multimodal_single_section_obs_file(
    dir: &TempDir,
    filename: &str,
    n_obs: usize,
    rna_n_vars: u64,
    adt_n_vars: u64,
    shard_target_rows: u32,
) -> PathBuf {
    use scx_format_io::modality::ModalityType;
    let path = dir.path().join(filename);
    let mut header = sample_header(n_obs as u64, rna_n_vars);
    header.shard_target_rows = shard_target_rows;
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
    writer.finish().unwrap();
    path
}

#[test]
fn compact_reshape_obs_converts_multimodal_legacy_to_shards() {
    let dir = tempfile::tempdir().unwrap();
    // 10 rows, shard_target_rows = 4 → ceil(10/4) = 3 obs shards.
    let input = write_multimodal_single_section_obs_file(&dir, "legacy_mm.scx", 10, 30, 10, 4);

    // Precondition: input is legacy single-section obs.
    let reader = ScxReader::open(&input).unwrap();
    assert_eq!(reader.obs_metadata_shard_count(), 0);
    drop(reader);

    let out = dir.path().join("reshaped_mm.scx");
    scx_ops::compact_with_index_options(&input, &out, &no_index_options(), true).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    // Obs migrated to sharded layout; legacy single section is gone.
    assert_eq!(reader.obs_metadata_shard_count(), 3);
    assert!(reader.catalog().get("obs").is_none());

    // Obs round-trips through the assembled multimodal read path.
    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 10);
    let ids = obs
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for i in 0..10 {
        assert_eq!(ids.value(i), format!("cell_{i}"));
    }

    // Both modalities and their CSR data are preserved.
    let table = reader.modality_table().expect("modality table preserved");
    assert_eq!(table.entries.len(), 2);
    for modality_id in 1u8..=2u8 {
        let csr = reader.read_all_csr_shards_for(modality_id).unwrap();
        assert_eq!(csr.shape.0, 10, "modality {modality_id} row count");
    }

    // Provenance records the migration.
    let prov = reader.read_provenance().unwrap();
    let last = prov.operations.last().unwrap();
    assert_eq!(last.action, "compact");
    assert!(last.params_json.contains("\"reshape_obs\":true"));
}

// ---------------------------------------------------------------------------
// compact --reshape-obs (Patch 4): streaming obs path on already-sharded input
// ---------------------------------------------------------------------------

/// Write an SCX file whose obs is **already** stored as multiple
/// `ObsMetadataShard` sections (via `write_obs_shard`), so
/// `compact --reshape-obs` takes the streaming obs path
/// (`write_obs_shards_streaming`) instead of the eager re-slice. X is a single
/// CSR shard. When `with_cluster` is set, obs carries a low-cardinality
/// `cluster` dictionary column for predicate-index tests.
fn write_sharded_obs_file(
    dir: &TempDir,
    filename: &str,
    n_obs: usize,
    n_vars: usize,
    obs_shard_rows: usize,
    with_cluster: bool,
) -> PathBuf {
    let path = dir.path().join(filename);
    let mut header = sample_header(n_obs as u64, n_vars as u64);
    header.shard_target_rows = obs_shard_rows as u32;
    let mut writer = ScxWriter::new(&path, header).unwrap();

    let obs = if with_cluster {
        sample_obs_with_cluster(n_obs)
    } else {
        sample_obs(n_obs)
    };
    let mut shard_idx = 0u32;
    let mut row_start = 0usize;
    while row_start < n_obs {
        let take = obs_shard_rows.min(n_obs - row_start);
        writer
            .write_obs_shard(
                shard_idx,
                row_start as u64,
                take as u64,
                n_obs as u64,
                &obs.slice(row_start, take),
            )
            .unwrap();
        shard_idx += 1;
        row_start += take;
    }

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

/// Multimodal variant of [`write_sharded_obs_file`]: global obs is pre-sharded;
/// two modalities (rna + adt) each carry a single CSR shard.
fn write_multimodal_sharded_obs_file(
    dir: &TempDir,
    filename: &str,
    n_obs: usize,
    rna_n_vars: u64,
    adt_n_vars: u64,
    obs_shard_rows: usize,
) -> PathBuf {
    use scx_format_io::modality::ModalityType;
    let path = dir.path().join(filename);
    let mut header = sample_header(n_obs as u64, rna_n_vars);
    header.shard_target_rows = obs_shard_rows as u32;
    let mut writer = ScxWriter::new(&path, header).unwrap();

    let obs = sample_obs(n_obs);
    let mut shard_idx = 0u32;
    let mut row_start = 0usize;
    while row_start < n_obs {
        let take = obs_shard_rows.min(n_obs - row_start);
        writer
            .write_obs_shard(
                shard_idx,
                row_start as u64,
                take as u64,
                n_obs as u64,
                &obs.slice(row_start, take),
            )
            .unwrap();
        shard_idx += 1;
        row_start += take;
    }

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

    let (i, j, v) = sample_shard_data(n_obs, rna_n_vars as usize);
    writer
        .write_csr_shard_for(rna_id, &i, &j, &v, CodecId::None, ValueEncoding::Uint8, 0)
        .unwrap();
    let (i, j, v) = sample_shard_data(n_obs, adt_n_vars as usize);
    writer
        .write_csr_shard_for(adt_id, &i, &j, &v, CodecId::None, ValueEncoding::Uint8, 0)
        .unwrap();
    writer.finish().unwrap();
    path
}

#[test]
fn compact_reshape_streams_sharded_obs_no_deletions() {
    let dir = tempfile::tempdir().unwrap();
    // 10 rows, obs shards of 4 → 3 input obs shards.
    let input = write_sharded_obs_file(&dir, "sharded.scx", 10, 8, 4, false);

    let reader = ScxReader::open(&input).unwrap();
    assert_eq!(reader.obs_metadata_shard_count(), 3);
    drop(reader);

    let out = dir.path().join("out.scx");
    scx_ops::compact_with_index_options(&input, &out, &no_index_options(), true).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    // Per-input-shard streaming: 3 shards in → 3 shards out (no deletions).
    assert_eq!(reader.obs_metadata_shard_count(), 3);
    assert!(reader.catalog().get("obs").is_none());

    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 10);
    let ids = obs
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for i in 0..10 {
        assert_eq!(ids.value(i), format!("cell_{i}"));
    }
}

#[test]
fn compact_reshape_streams_sharded_obs_with_deletions() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_sharded_obs_file(&dir, "sharded.scx", 10, 8, 4, false);
    // Delete global obs rows 1 and 4.
    scx_ops::mark_deleted(&input, &[1, 4]).unwrap();

    let out = dir.path().join("out.scx");
    scx_ops::compact_with_index_options(&input, &out, &no_index_options(), true).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert!(reader.obs_metadata_shard_count() > 0);
    assert!(reader.catalog().get("obs").is_none());
    assert_eq!(reader.header().n_obs, 8);

    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 8);
    let ids = obs
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    // Kept ids are the originals minus cell_1 / cell_4, in order.
    let expected: Vec<String> = (0..10)
        .filter(|i| *i != 1 && *i != 4)
        .map(|i| format!("cell_{i}"))
        .collect();
    let got: Vec<String> = (0..obs.num_rows())
        .map(|i| ids.value(i).to_string())
        .collect();
    assert_eq!(got, expected);
}

#[test]
fn compact_reshape_sharded_obs_builds_streaming_predicate_index() {
    let dir = tempfile::tempdir().unwrap();
    // 12 rows, obs shards of 4 → 3 input obs shards, with a `cluster` column.
    let input = write_sharded_obs_file(&dir, "sharded.scx", 12, 8, 4, true);

    let opts = scx_engine::ConversionPredicateIndexOptions {
        index_obs: vec!["cluster".to_string()],
        index_var: Vec::new(),
        index_preset: None,
        index_auto_threshold: 0,
    };
    let out = dir.path().join("out.scx");
    let summary = scx_ops::compact_with_index_options(&input, &out, &opts, true).unwrap();

    // The streaming index pass indexed the forced column.
    let result = summary.result.expect("predicate index built");
    assert!(result.obs_indexed_columns.iter().any(|c| c == "cluster"));

    let reader = ScxReader::open(&out).unwrap();
    assert!(reader.obs_metadata_shard_count() > 0);
    assert!(reader.read_obs_predicate_index_bytes().unwrap().is_some());

    // obs still round-trips with the cluster column intact.
    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 12);
    assert!(obs.schema().column_with_name("cluster").is_some());
}

#[test]
fn compact_reshape_all_obs_deleted_writes_single_empty_section() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_sharded_obs_file(&dir, "sharded.scx", 6, 8, 3, false);
    scx_ops::mark_deleted(&input, &[0, 1, 2, 3, 4, 5]).unwrap();

    let out = dir.path().join("out.scx");
    scx_ops::compact_with_index_options(&input, &out, &no_index_options(), true).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.header().n_obs, 0);
    // No obs shards; one empty single section keeps the file well-formed.
    assert_eq!(reader.obs_metadata_shard_count(), 0);
    assert_eq!(reader.read_obs().unwrap().num_rows(), 0);
}

#[test]
fn compact_reshape_multimodal_streams_sharded_obs() {
    let dir = tempfile::tempdir().unwrap();
    // 10 rows, obs shards of 4 → 3 input obs shards.
    let input = write_multimodal_sharded_obs_file(&dir, "mm_sharded.scx", 10, 30, 10, 4);

    let reader = ScxReader::open(&input).unwrap();
    assert_eq!(reader.obs_metadata_shard_count(), 3);
    drop(reader);

    let out = dir.path().join("out.scx");
    scx_ops::compact_with_index_options(&input, &out, &no_index_options(), true).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.obs_metadata_shard_count(), 3);
    assert!(reader.catalog().get("obs").is_none());

    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 10);
    let ids = obs
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for i in 0..10 {
        assert_eq!(ids.value(i), format!("cell_{i}"));
    }

    let table = reader.modality_table().expect("modality table preserved");
    assert_eq!(table.entries.len(), 2);
    for modality_id in 1u8..=2u8 {
        let csr = reader.read_all_csr_shards_for(modality_id).unwrap();
        assert_eq!(csr.shape.0, 10, "modality {modality_id} row count");
    }
}
