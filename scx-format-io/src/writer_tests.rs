use super::*;
use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

fn sample_header() -> FileHeader {
    FileHeader::new_single_modality(100, 50, 500, crate::DEFAULT_SHARD_TARGET_ROWS, 0, 0)
}

fn sample_obs() -> RecordBatch {
    let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(vec![
            "cell_0", "cell_1", "cell_2",
        ]))],
    )
    .unwrap()
}

fn sample_var() -> RecordBatch {
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(vec!["gene_0", "gene_1"]))],
    )
    .unwrap()
}

fn sample_shard_data() -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    // 3 rows, nnz=6
    let indptr = vec![0u64, 2, 5, 6];
    let indices = vec![1u32, 3, 0, 2, 4, 2];
    let values: Vec<u8> = vec![5, 10, 1, 3, 7, 2]; // u8 encoding
    (indptr, indices, values)
}

/// `set_csr_shard_column_stats_bulk` assigns each `Vec<ColumnStat>` to the
/// CSR shard at the matching `csr_shards_sorted` position (by row_start),
/// regardless of the order shards were written, and the stats round-trip
/// through the catalog.
#[test]
fn bulk_csr_shard_column_stats_assigns_by_sorted_position() {
    use crate::catalog::{column_name_hash, ColumnStat};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bulk.scx");
    let mut writer = ScxWriter::new(&path, sample_header()).unwrap();
    writer.write_obs(&sample_obs()).unwrap();
    writer.write_var(&sample_var()).unwrap();
    let (indptr, indices, values) = sample_shard_data();
    // Write shard covering rows 3..6 first, then 0..3 — so write order
    // differs from sorted (row_start) order.
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            3,
        )
        .unwrap();
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

    let hash = column_name_hash("cell_type");
    // per_shard[0] is for the row_start==0 shard; per_shard[1] for row_start==3.
    writer
        .set_csr_shard_column_stats_bulk(vec![
            vec![ColumnStat::CategoryBitset {
                column_name_hash: hash,
                bitset: vec![0b0000_0001],
            }],
            vec![ColumnStat::CategoryBitset {
                column_name_hash: hash,
                bitset: vec![0b0000_0010],
            }],
        ])
        .unwrap();
    writer.finish().unwrap();

    let reader = crate::ScxReader::open(&path).unwrap();
    let sorted = reader.catalog().csr_shards_sorted();
    assert_eq!(sorted.len(), 2);
    // Sorted by row_start: index 0 == rows 0..3, index 1 == rows 3..6.
    let stats0 = sorted[0].stats.as_ref().unwrap();
    let stats1 = sorted[1].stats.as_ref().unwrap();
    assert_eq!(stats0.row_start, 0);
    assert_eq!(stats1.row_start, 3);
    match &stats0.column_stats[0] {
        ColumnStat::CategoryBitset { bitset, .. } => assert_eq!(bitset, &vec![0b0000_0001]),
        other => panic!("unexpected stat: {other:?}"),
    }
    match &stats1.column_stats[0] {
        ColumnStat::CategoryBitset { bitset, .. } => assert_eq!(bitset, &vec![0b0000_0010]),
        other => panic!("unexpected stat: {other:?}"),
    }
}

/// Wrong per-shard length is rejected (guards against shard_id/range
/// misalignment — the bug class that left pushdown non-functional).
#[test]
fn bulk_csr_shard_column_stats_rejects_wrong_length() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bulk_bad.scx");
    let mut writer = ScxWriter::new(&path, sample_header()).unwrap();
    writer.write_obs(&sample_obs()).unwrap();
    writer.write_var(&sample_var()).unwrap();
    let (indptr, indices, values) = sample_shard_data();
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
    // One CSR shard but two stat vecs.
    let err = writer
        .set_csr_shard_column_stats_bulk(vec![vec![], vec![]])
        .unwrap_err();
    assert!(matches!(
        err,
        ScxError::ColumnStatsShardCountMismatch {
            got: 2,
            expected: 1
        }
    ));
}

/// Round-trip the `adopt_in_place` / `into_in_place_parts` pair: start
/// from a partial file (zero header + one section), hand the file to
/// `ScxWriter`, write an `obs_predicate_index` section, take the
/// pieces back, and verify the offset advanced and a new catalog
/// entry was appended. Used by `scx-ops::append::finalize_append`.
#[test]
fn adopt_in_place_writes_section_and_returns_state() {
    use std::io::Write;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("adopt.scx");

    // Lay down a placeholder header + one pre-existing fake section so
    // we exercise the non-empty-entries handoff path.
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .unwrap();
    let initial_bytes = vec![0u8; SECTIONS_START_OFFSET as usize + 16];
    file.write_all(&initial_bytes).unwrap();
    let initial_offset = initial_bytes.len() as u64;

    let prior_entry = FullCatalogEntry {
        name: "obs".to_string(),
        offset: SECTIONS_START_OFFSET,
        length: 16,
        section_type: SectionType::ObsMetadata,
        checksum: [0u8; 32],
        modality_id: 0,
        stats: None,
    };

    let mut writer =
        ScxWriter::adopt_in_place(file, sample_header(), initial_offset, vec![prior_entry])
            .unwrap();

    // Section payload — small placeholder bytes, not a real
    // PredicateIndex; `write_obs_predicate_index` does not validate
    // content, it only emits the section.
    let payload = b"predicate_index_payload";
    writer.write_obs_predicate_index(payload).unwrap();

    let (file, end_offset, entries) = writer.into_in_place_parts().unwrap();

    // Section bytes were written + alignment padding was applied.
    assert!(end_offset >= initial_offset + payload.len() as u64);
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].name, "obs");
    assert_eq!(entries[1].name, "obs_predicate_index");
    assert_eq!(entries[1].section_type, SectionType::ObsPredicateIndex);
    assert_eq!(entries[1].length, payload.len() as u64);
    // The new section's offset must equal the (aligned) caller-
    // supplied current_offset — i.e. it lands immediately after the
    // prior section.
    assert!(entries[1].offset >= initial_offset);

    // File on disk matches the returned end_offset.
    drop(file);
    let on_disk = std::fs::metadata(&path).unwrap().len();
    assert_eq!(on_disk, end_offset);
}

/// 10.14: Write minimal file → verify header fields and 8-byte alignment
#[test]
fn test_minimal_file_header_and_alignment() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.scx");
    let header = sample_header();

    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs()).unwrap();
    writer.write_var(&sample_var()).unwrap();

    let (indptr, indices, values) = sample_shard_data();
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

    let final_path = writer.finish().unwrap();
    assert!(final_path.exists());

    // Read back header
    let data = std::fs::read(&final_path).unwrap();
    let mut cursor = std::io::Cursor::new(&data);
    let hdr = FileHeader::read_from(&mut cursor).unwrap();

    assert_eq!(hdr.magic, crate::header::MAGIC);
    // Unframed writes stamp the default (v3), not the max-readable CURRENT (v4).
    assert_eq!(
        hdr.format_version,
        crate::header::DEFAULT_WRITE_FORMAT_VERSION
    );
    assert_eq!(hdr.n_csr_shards, 1);
    assert_eq!(hdr.root_catalog_offset, HEADER_SIZE as u64);
    assert!(hdr.full_catalog_offset >= SECTIONS_START_OFFSET);
    assert_eq!(hdr.full_catalog_offset % 8, 0);
    assert!(hdr.file_checksum != 0);

    // Verify full catalog is readable
    let fc_start = hdr.full_catalog_offset as usize;
    let fc_end = fc_start + hdr.full_catalog_length as usize;
    let mut fc_cursor = std::io::Cursor::new(&data[fc_start..fc_end]);
    let catalog =
        FullCatalog::read_from(&mut fc_cursor, hdr.full_catalog_length as usize, true).unwrap();
    assert_eq!(catalog.entries.len(), 3); // obs + var + 1 shard

    // Verify all section offsets are 8-byte aligned
    for entry in &catalog.entries {
        assert_eq!(
            entry.offset % 8,
            0,
            "section '{}' offset {} not 8-byte aligned",
            entry.name,
            entry.offset
        );
    }
}

/// 10.15: Temp file lifecycle — tmp exists before finish, final after.
/// With randomized temp file names we can't predict the exact path,
/// so we scan the directory for `.tmp` files instead.
#[test]
fn test_temp_file_lifecycle() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lifecycle.scx");

    let header = sample_header();
    let mut writer = ScxWriter::new(&path, header).unwrap();

    // A temp file should exist in the directory; final path should not.
    let tmp_files_before: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|s| s.contains(".tmp"))
                .unwrap_or(false)
        })
        .collect();
    assert!(
        !tmp_files_before.is_empty(),
        "temp file should exist after new()"
    );
    assert!(
        !path.exists(),
        "final path should not exist before finish()"
    );

    writer.write_obs(&sample_obs()).unwrap();
    writer.write_var(&sample_var()).unwrap();

    let (indptr, indices, values) = sample_shard_data();
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

    // Final file should exist; no temp files should remain.
    assert!(path.exists(), "final path should exist after finish()");
    let tmp_files_after: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|s| s.contains(".tmp"))
                .unwrap_or(false)
        })
        .collect();
    assert!(
        tmp_files_after.is_empty(),
        "no temp files should remain after finish()"
    );
}

/// 10.16: Write obs + var + 4 CSR shards → verify catalog entries
#[test]
fn test_four_shards_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("four_shards.scx");
    let header = sample_header();

    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs()).unwrap();
    writer.write_var(&sample_var()).unwrap();

    let (indptr, indices, values) = sample_shard_data();
    for i in 0..4u64 {
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                i * 3, // each shard has 3 rows
            )
            .unwrap();
    }

    let final_path = writer.finish().unwrap();

    // Read back catalog
    let data = std::fs::read(&final_path).unwrap();
    let mut cursor = std::io::Cursor::new(&data);
    let hdr = FileHeader::read_from(&mut cursor).unwrap();
    assert_eq!(hdr.n_csr_shards, 4);

    let fc_start = hdr.full_catalog_offset as usize;
    let fc_end = fc_start + hdr.full_catalog_length as usize;
    let mut fc_cursor = std::io::Cursor::new(&data[fc_start..fc_end]);
    let catalog =
        FullCatalog::read_from(&mut fc_cursor, hdr.full_catalog_length as usize, true).unwrap();

    assert_eq!(catalog.entries.len(), 6); // obs + var + 4 shards

    let csr_entries: Vec<_> = catalog
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CsrShard)
        .collect();
    assert_eq!(csr_entries.len(), 4);

    // Verify distinct row_starts
    let row_starts: Vec<u64> = csr_entries
        .iter()
        .map(|e| e.stats.as_ref().unwrap().row_start)
        .collect();
    assert_eq!(row_starts, vec![0, 3, 6, 9]);
}

/// 10.17: Write all section types → verify in catalog
#[test]
fn test_all_section_types() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("all_types.scx");
    let header = sample_header();

    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs()).unwrap();
    writer.write_var(&sample_var()).unwrap();

    let (indptr, indices, values) = sample_shard_data();
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

    // Layer shard
    writer
        .write_layer_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
            "raw",
            0,
        )
        .unwrap();

    // obsm
    let obsm_schema = Schema::new(vec![
        Field::new("x", DataType::Int32, false),
        Field::new("y", DataType::Int32, false),
    ]);
    let obsm_batch = RecordBatch::try_new(
        Arc::new(obsm_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int32Array::from(vec![4, 5, 6])),
        ],
    )
    .unwrap();
    writer.write_obsm("X_pca", &obsm_batch).unwrap();

    // uns
    writer
        .write_uns(&serde_json::json!({"key": "value"}))
        .unwrap();

    // provenance
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1710000000,
            action: "convert".to_string(),
            tool: "scx-cli 0.1.0".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();

    let final_path = writer.finish().unwrap();

    let data = std::fs::read(&final_path).unwrap();
    let mut cursor = std::io::Cursor::new(&data);
    let hdr = FileHeader::read_from(&mut cursor).unwrap();

    let fc_start = hdr.full_catalog_offset as usize;
    let fc_end = fc_start + hdr.full_catalog_length as usize;
    let mut fc_cursor = std::io::Cursor::new(&data[fc_start..fc_end]);
    let catalog =
        FullCatalog::read_from(&mut fc_cursor, hdr.full_catalog_length as usize, true).unwrap();

    // obs, var, csr_shard, layer_csr_shard, obsm, uns, provenance = 7
    assert_eq!(catalog.entries.len(), 7);

    let types: Vec<SectionType> = catalog.entries.iter().map(|e| e.section_type).collect();
    assert!(types.contains(&SectionType::ObsMetadata));
    assert!(types.contains(&SectionType::VarMetadata));
    assert!(types.contains(&SectionType::CsrShard));
    assert!(types.contains(&SectionType::LayerCsrShard));
    assert!(types.contains(&SectionType::ObsmEmbedding));
    assert!(types.contains(&SectionType::UnsBlob));
    assert!(types.contains(&SectionType::Provenance));
}

/// 10.18: Root catalog at offset 256 is <= 4096 bytes and readable
#[test]
fn test_root_catalog_structure() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("root_cat.scx");
    let header = sample_header();

    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs()).unwrap();
    writer.write_var(&sample_var()).unwrap();

    let (indptr, indices, values) = sample_shard_data();
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

    let final_path = writer.finish().unwrap();

    let data = std::fs::read(&final_path).unwrap();
    let mut cursor = std::io::Cursor::new(&data);
    let hdr = FileHeader::read_from(&mut cursor).unwrap();

    assert_eq!(hdr.root_catalog_offset, HEADER_SIZE as u64);
    assert!(hdr.root_catalog_length <= 4096);

    // Read root catalog from offset 256
    let rc_start = HEADER_SIZE;
    let mut rc_cursor = std::io::Cursor::new(&data[rc_start..rc_start + 4096]);
    let root_catalog = RootCatalog::read_from(&mut rc_cursor).unwrap();

    // Should have groups for ObsMetadata, VarMetadata, CsrShard
    assert!(root_catalog.n_section_groups >= 3);
    assert_eq!(
        root_catalog.entries.len(),
        root_catalog.n_section_groups as usize
    );

    // Verify each group has valid fields
    for entry in &root_catalog.entries {
        assert!(entry.first_section_offset >= SECTIONS_START_OFFSET);
        assert!(entry.n_sections > 0);
        assert!(entry.total_group_length > 0);
    }
}

/// Test Drop cleans up temp file when finish() is not called.
/// TempPath auto-deletes the temp file on drop.
#[test]
fn test_drop_cleans_up_temp() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dropped.scx");

    {
        let _writer = ScxWriter::new(&path, sample_header()).unwrap();
        // A temp file should exist somewhere in the directory.
        let tmp_count = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .map(|s| s.contains(".tmp"))
                    .unwrap_or(false)
            })
            .count();
        assert!(tmp_count > 0, "temp file should exist before drop");
    }
    // After drop: no temp files, no final file.
    let tmp_count_after = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|s| s.contains(".tmp"))
                .unwrap_or(false)
        })
        .count();
    assert_eq!(
        tmp_count_after, 0,
        "temp file should be cleaned up after drop"
    );
    assert!(
        !path.exists(),
        "final path should not exist after drop without finish"
    );
}

/// Two concurrent ScxWriters targeting the same final path should use
/// different temp files and not trample each other.
#[test]
fn test_concurrent_writers_no_collision() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("concurrent.scx");

    let writer1 = ScxWriter::new(&path, sample_header()).unwrap();
    let writer2 = ScxWriter::new(&path, sample_header()).unwrap();

    // Both writers should have created separate temp files.
    let tmp_files: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|s| s.contains(".tmp"))
                .unwrap_or(false)
        })
        .collect();
    assert_eq!(
        tmp_files.len(),
        2,
        "two concurrent writers should create two distinct temp files"
    );

    // Dropping both should clean up both temp files.
    drop(writer1);
    drop(writer2);

    let tmp_remaining: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|s| s.contains(".tmp"))
                .unwrap_or(false)
        })
        .collect();
    assert!(
        tmp_remaining.is_empty(),
        "all temp files should be cleaned up after dropping both writers"
    );
}

/// `make_sibling_tempfile` creates a temp file in the same directory
/// as the final path, with the `.{stem}_<rand>.tmp` naming convention.
#[test]
fn test_make_sibling_tempfile_creates_in_parent() {
    let dir = tempfile::tempdir().unwrap();
    let final_path = dir.path().join("dest.scx");
    let (file, tmp_path) = make_sibling_tempfile(&final_path).unwrap();
    // File handle should be valid (write 1 byte).
    let mut f = file;
    use std::io::Write;
    f.write_all(b"x").unwrap();
    // Temp path lives in the same parent directory.
    assert_eq!(tmp_path.parent(), Some(dir.path()));
    // Filename matches `.dest.scx_*.tmp`.
    let name = tmp_path.file_name().and_then(|n| n.to_str()).unwrap();
    assert!(
        name.starts_with(".dest.scx_"),
        "name {name:?} should start with `.dest.scx_`"
    );
    assert!(
        name.ends_with(".tmp"),
        "name {name:?} should end with `.tmp`"
    );
    // Dropping `tmp_path` cleans up.
    let path_clone = tmp_path.to_path_buf();
    drop(tmp_path);
    assert!(!path_clone.exists(), "temp file should be deleted on drop");
}

/// `finish()` should restore umask-respecting permissions on the
/// persisted file. `tempfile::NamedTempFile` creates `0600`; after the
/// post-persist `chmod_to_umask` call we expect `0o666 & !umask`.
///
/// This test is Unix-only and inherently single-threaded because it
/// reads (and briefly clears) the process umask. The umask cache in
/// `current_umask()` reads on first call — to make this test
/// deterministic regardless of test ordering we force a known umask
/// before any `chmod_to_umask` call in this binary may have run.
#[cfg(unix)]
#[test]
fn test_finish_sets_umask_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("perm.scx");

    let mut writer = ScxWriter::new(&path, sample_header()).unwrap();
    writer.write_obs(&sample_obs()).unwrap();
    writer.write_var(&sample_var()).unwrap();
    let (indptr, indices, values) = sample_shard_data();
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

    // Read back current umask the same way `chmod_to_umask` does.
    // SAFETY: same umask dance as `current_umask`.
    let umask = unsafe {
        let saved = libc::umask(0o022);
        libc::umask(saved);
        saved as u32
    };
    let expected_mode = 0o666 & !umask;
    let actual_mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
    assert_eq!(
        actual_mode, expected_mode,
        "persisted file mode {actual_mode:o} should equal 0o666 & !umask ({expected_mode:o})"
    );
}

/// Test compute_shard_stats (row-major branch)
#[test]
fn test_compute_shard_stats() {
    let values: Vec<u8> = vec![5, 10, 1, 3, 7, 2];
    // Row-major shard: rows [0, 3), col_end = n_minor (= n_vars).
    let stats = compute_shard_stats(&values, ValueEncoding::Uint8, MajorAxis::Row, 0, 3, 50, 6);
    assert_eq!(stats.row_start, 0);
    assert_eq!(stats.row_end, 3);
    assert_eq!(stats.col_start, 0);
    assert_eq!(stats.col_end, 50);
    assert_eq!(stats.nnz, 6);
    assert_eq!(stats.value_min, 1);
    assert_eq!(stats.value_max, 10);
    assert_eq!(stats.value_sum, 28); // 5+10+1+3+7+2
}

/// Test compute_shard_stats column-major branch.
#[test]
fn test_compute_shard_stats_col_major() {
    let values: Vec<u8> = vec![5, 10, 1];
    // Column-major shard: cols [100, 102), row pair = full [0, n_obs).
    let stats = compute_shard_stats(
        &values,
        ValueEncoding::Uint8,
        MajorAxis::Col,
        100,
        2,
        1000,
        3,
    );
    assert_eq!(stats.col_start, 100);
    assert_eq!(stats.col_end, 102);
    assert_eq!(stats.row_start, 0);
    assert_eq!(stats.row_end, 1000);
}

/// Test compute_shard_stats for Float32 returns zero stats
#[test]
fn test_compute_shard_stats_float32() {
    // Float32: 3 values as LE bytes (1.0f32, 2.5f32, 0.5f32)
    let mut values = Vec::new();
    values.extend_from_slice(&1.0f32.to_le_bytes());
    values.extend_from_slice(&2.5f32.to_le_bytes());
    values.extend_from_slice(&0.5f32.to_le_bytes());
    let stats = compute_shard_stats(&values, ValueEncoding::Float32, MajorAxis::Row, 0, 2, 50, 3);
    assert_eq!(stats.value_min, 0, "float32 value_min must be zero");
    assert_eq!(stats.value_max, 0, "float32 value_max must be zero");
    assert_eq!(stats.value_sum, 0, "float32 value_sum must be zero");
    assert_eq!(stats.row_start, 0);
    assert_eq!(stats.row_end, 2);
    assert_eq!(stats.nnz, 3);
}

/// Test compute_shard_stats for Float16 returns zero stats
#[test]
fn test_compute_shard_stats_float16() {
    let values = vec![0u8; 6]; // 3 × 2-byte float16 values
    let stats = compute_shard_stats(
        &values,
        ValueEncoding::Float16,
        MajorAxis::Row,
        10,
        5,
        50,
        3,
    );
    assert_eq!(stats.value_min, 0, "float16 value_min must be zero");
    assert_eq!(stats.value_max, 0, "float16 value_max must be zero");
    assert_eq!(stats.value_sum, 0, "float16 value_sum must be zero");
    assert_eq!(stats.row_start, 10);
    assert_eq!(stats.row_end, 15);
}

/// P3: Write CSR + CSC shards → verify CSC section in catalog and correct data
#[test]
fn test_csc_shard_write_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("with_csc.scx");
    let header = sample_header();

    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs()).unwrap();
    writer.write_var(&sample_var()).unwrap();

    let (indptr, indices, values) = sample_shard_data();

    // Write a CSR shard (row-major)
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

    // Write a CSC shard (column-major) with the same data
    // In a real scenario the indptr/indices represent column pointers/row indices
    writer
        .write_csc_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0, // col_start
        )
        .unwrap();

    let final_path = writer.finish().unwrap();

    // Read back and verify
    let data = std::fs::read(&final_path).unwrap();
    let mut cursor = std::io::Cursor::new(&data);
    let hdr = FileHeader::read_from(&mut cursor).unwrap();

    // Verify shard counts
    assert_eq!(hdr.n_csr_shards, 1);
    assert_eq!(hdr.n_csc_shards, 1);

    // Verify catalog has both shard types
    let fc_start = hdr.full_catalog_offset as usize;
    let fc_end = fc_start + hdr.full_catalog_length as usize;
    let mut fc_cursor = std::io::Cursor::new(&data[fc_start..fc_end]);
    let catalog =
        FullCatalog::read_from(&mut fc_cursor, hdr.full_catalog_length as usize, true).unwrap();

    // obs + var + 1 CSR shard + 1 CSC shard = 4 entries
    assert_eq!(catalog.entries.len(), 4);

    let csr_entries: Vec<_> = catalog
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CsrShard)
        .collect();
    assert_eq!(csr_entries.len(), 1);
    assert_eq!(csr_entries[0].name, "X_shard_0");

    let csc_entries: Vec<_> = catalog
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CscShard)
        .collect();
    assert_eq!(csc_entries.len(), 1);
    assert_eq!(csc_entries[0].name, "X_csc_shard_0");

    // CSC shard should have stats (computed from values)
    let csc_stats = csc_entries[0].stats.as_ref().unwrap();
    assert_eq!(csc_stats.nnz, 6);

    // CSC shard's on-disk header byte must be `1` (Phase A.1).
    let csc_section = &data[csc_entries[0].offset as usize..][..csc_entries[0].length as usize];
    let csc_sh =
        ShardHeader::read_from(&mut std::io::Cursor::new(&csc_section[..SHARD_HEADER_SIZE]))
            .unwrap();
    assert_eq!(csc_sh.shard_type, 1, "CSC shard_type byte must be 1");
    assert!(csc_sh.is_csc(SectionType::CscShard));

    // CSR shard byte must remain `0`.
    let csr_section = &data[csr_entries[0].offset as usize..][..csr_entries[0].length as usize];
    let csr_sh =
        ShardHeader::read_from(&mut std::io::Cursor::new(&csr_section[..SHARD_HEADER_SIZE]))
            .unwrap();
    assert_eq!(csr_sh.shard_type, 0, "CSR shard_type byte must be 0");
    assert!(!csr_sh.is_csc(SectionType::CsrShard));

    // All sections should be 8-byte aligned
    for entry in &catalog.entries {
        assert_eq!(
            entry.offset % 8,
            0,
            "section '{}' offset {} not 8-byte aligned",
            entry.name,
            entry.offset
        );
    }
}

/// A CSC sidecar written with framing on is emitted
/// row-group-framed (shard v2) and full-decodes byte-identically to its input.
#[test]
fn test_csc_shard_framed_round_trip() {
    use crate::reader::ScxReader;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("framed_csc.scx");
    let mut header = sample_header();
    header.format_version = crate::header::CURRENT_FORMAT_VERSION; // v4 (framed)
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.set_framing(Some(crate::encoder::FramingConfig {
        row_group_rows: 1, // ≥2 gene-groups over the fixture → framing exercised
        target_nnz: None,
        trial: false,
        decode_target: None,
    }));
    writer.write_obs(&sample_obs()).unwrap();
    writer.write_var(&sample_var()).unwrap();

    let (indptr, indices, values) = sample_shard_data();
    writer
        .write_csc_shard(
            &indptr,
            &indices,
            &values,
            CodecId::ShufDeltaZstd,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    let final_path = writer.finish().unwrap();

    let reader = ScxReader::open(&final_path).unwrap();
    // The CSC shard must carry the framed (v2) layout.
    let entry = reader.catalog().csc_shards_sorted()[0];
    let sh = reader.read_shard_header(entry).unwrap();
    assert_eq!(
        sh.shard_format_version,
        crate::shard::CURRENT_SHARD_FORMAT_VERSION,
        "framed CSC shard must be v2",
    );
    // …and full-decode byte-identically to the input.
    let csc = reader.read_csc_shard(0).unwrap();
    let exp_indptr: Vec<i64> = indptr.iter().map(|&v| v as i64).collect();
    let exp_indices: Vec<i32> = indices.iter().map(|&v| v as i32).collect();
    let exp_data: Vec<f32> = values.iter().map(|&v| v as f32).collect();
    assert_eq!(csc.indptr, exp_indptr);
    assert_eq!(csc.indices, exp_indices);
    assert_eq!(csc.data, exp_data);
}

/// Item 2 (scattered CSC reader): a gene-subset `read_csc_columns` over a
/// row-group-framed CSC file decodes only the touched column-groups via the
/// block index and is byte-identical to a full-shard decode + `col_slice`.
#[test]
fn read_csc_columns_scattered_matches_full_decode() {
    use crate::reader::ScxReader;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("framed_csc_scatter.scx");
    let mut header = sample_header();
    header.format_version = crate::header::CURRENT_FORMAT_VERSION;
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.set_framing(Some(crate::encoder::FramingConfig {
        row_group_rows: 1, // one column-group per gene → framing fully exercised
        target_nnz: None,
        trial: false,
        decode_target: None,
    }));
    writer.write_obs(&sample_obs()).unwrap();
    writer.write_var(&sample_var()).unwrap();
    let (indptr, indices, values) = sample_shard_data(); // 3 columns
    writer
        .write_csc_shard(
            &indptr,
            &indices,
            &values,
            CodecId::ShufDeltaZstd,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    let final_path = writer.finish().unwrap();

    let reader = ScxReader::open(&final_path).unwrap();
    // Ground truth: whole-shard decode (no scatter), then column-slice.
    let full = reader.read_csc_shard(0).unwrap();
    let n_cols = full.n_cols();
    for lo in 0..n_cols {
        for hi in (lo + 1)..=n_cols {
            let sub = reader.read_csc_columns(lo as u32..hi as u32).unwrap(); // framed → scattered
            let expected = full.col_slice(lo, hi).unwrap();
            assert_eq!(sub.indptr, expected.indptr, "indptr {lo}..{hi}");
            assert_eq!(sub.indices, expected.indices, "indices {lo}..{hi}");
            assert_eq!(sub.data, expected.data, "data {lo}..{hi}");
        }
    }
}

/// `decode_block_index_row_runs` must return an error (not panic) when a
/// requested run exceeds the shard's `n_major`. Callers build runs from
/// `ShardStats` ranges that aren't otherwise validated against the header, and
/// an out-of-range run would index past the decoded group CSR.
#[test]
fn decode_block_index_row_runs_rejects_out_of_range_run() {
    use crate::reader::ScxReader;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("framed_oob.scx");
    let mut header = sample_header();
    header.format_version = crate::header::CURRENT_FORMAT_VERSION;
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.set_framing(Some(crate::encoder::FramingConfig {
        row_group_rows: 1,
        target_nnz: None,
        trial: false,
        decode_target: None,
    }));
    writer.write_obs(&sample_obs()).unwrap();
    writer.write_var(&sample_var()).unwrap();
    let (indptr, indices, values) = sample_shard_data();
    let n_major = indptr.len() - 1;
    writer
        .write_csc_shard(
            &indptr,
            &indices,
            &values,
            CodecId::ShufDeltaZstd,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    let final_path = writer.finish().unwrap();

    let reader = ScxReader::open(&final_path).unwrap();
    let entry = reader.catalog().csc_shards_sorted()[0].clone();
    // A run that starts in range but extends past n_major must Err, not panic.
    let err = reader
        .decode_block_index_row_runs(&entry, &[(0, n_major + 5)])
        .unwrap_err();
    assert!(
        matches!(err, ScxError::InvalidCatalog(_)),
        "out-of-range run must yield InvalidCatalog, got {err:?}"
    );
    // A valid full-cover run still succeeds.
    assert!(reader
        .decode_block_index_row_runs(&entry, &[(0, n_major)])
        .unwrap()
        .is_some());
}

/// A tiny canonical CSR fixture (2 rows, nnz=3, 3 vars) for the T3.1 guard tests.
fn tiny_csr() -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    (vec![0u64, 2, 3], vec![0u32, 2, 1], vec![1.0f32, 2.0, 3.0])
}

/// T3.1 guard (§4.3 framed model): a v4 file requires framed (shard v2) shards.
/// Any unframed v1 CSR shard is rejected by `guard_no_legacy_shard_in_v4`; a
/// framed shard is accepted, and a v3 file still accepts plain unframed shards.
#[test]
fn v4_guard_rejects_unframed_v1_shard() {
    let dir = tempfile::tempdir().unwrap();
    let v4 = || {
        let mut h = sample_header();
        h.format_version = crate::header::CURRENT_FORMAT_VERSION;
        h
    };
    let write = |name: &str, header: FileHeader, pre: PreEncodedSection| {
        let mut w = ScxWriter::new(dir.path().join(name), header).unwrap();
        w.write_obs(&sample_obs()).unwrap();
        w.write_var(&sample_var()).unwrap();
        let r = w.write_preencoded_shard(pre);
        (w, r)
    };
    let (indptr, indices, values) = tiny_csr();

    // (1) Unframed (explicit Zstd) v1 shard → rejected in v4.
    let zstd_unframed = crate::encoder::encode_one_shard(
        &indptr,
        &indices,
        &values,
        Some(CodecId::Zstd),
        0,
        3,
        0,
        SectionType::CsrShard,
        ModalityType::Rna,
        "X_shard_0".to_string(),
        None,
    )
    .unwrap();
    let (_w, res) = write("v4_reject.scx", v4(), zstd_unframed);
    let err = res.unwrap_err();
    assert!(
        matches!(&err, ScxError::Io(e) if e.to_string().contains("random access")),
        "unframed v1 shard in v4 must be rejected, got {err:?}",
    );

    // (2) Framed shard → allowed in v4.
    let framed = crate::encoder::encode_one_shard(
        &indptr,
        &indices,
        &values,
        None,
        0,
        3,
        0,
        SectionType::CsrShard,
        ModalityType::Rna,
        "X_shard_0".to_string(),
        Some(crate::encoder::FramingConfig {
            row_group_rows: 1,
            target_nnz: None,
            trial: false,
            decode_target: None,
        }),
    )
    .unwrap();
    let (w, res) = write("v4_framed.scx", v4(), framed);
    res.unwrap();
    w.finish().unwrap();

    // (3) Unframed shard into a v3 file → accepted (the ordinary path).
    let plain = crate::encoder::encode_one_shard(
        &indptr,
        &indices,
        &values,
        Some(CodecId::Zstd),
        0,
        3,
        0,
        SectionType::CsrShard,
        ModalityType::Rna,
        "X_shard_0".to_string(),
        None,
    )
    .unwrap();
    let (w, res) = write("v3.scx", sample_header(), plain);
    res.unwrap();
    w.finish().unwrap();
}

/// T3.1 guard: `copy_section_verbatim` refuses to raw-copy a legacy (v1) CSR
/// shard into a v4 file. Builds an ordinary v3 file, then attempts to
/// verbatim-copy its shard into a v4 writer.
#[test]
fn copy_section_verbatim_rejects_legacy_shard_in_v4_file() {
    use crate::reader::ScxReader;
    let dir = tempfile::tempdir().unwrap();
    let (indptr, indices, values) = tiny_csr();

    let src_path = dir.path().join("legacy_v3.scx");
    let mut w = ScxWriter::new(&src_path, sample_header()).unwrap();
    w.write_obs(&sample_obs()).unwrap();
    w.write_var(&sample_var()).unwrap();
    w.write_preencoded_shard(
        crate::encoder::encode_one_shard(
            &indptr,
            &indices,
            &values,
            None,
            0,
            3,
            0,
            SectionType::CsrShard,
            ModalityType::Rna,
            "X_shard_0".to_string(),
            None,
        )
        .unwrap(),
    )
    .unwrap();
    let src_final = w.finish().unwrap();

    let reader = ScxReader::open(&src_final).unwrap();
    let entry = reader.catalog().csr_shards_sorted()[0].clone();
    let raw = reader.read_raw_shard_bytes(&entry).unwrap();

    let mut v4_header = sample_header();
    v4_header.format_version = crate::header::CURRENT_FORMAT_VERSION;
    let mut writer = ScxWriter::new(dir.path().join("v4_copy.scx"), v4_header).unwrap();
    writer.write_obs(&sample_obs()).unwrap();
    writer.write_var(&sample_var()).unwrap();
    let err = writer.copy_section_verbatim(&entry, raw).unwrap_err();
    assert!(
        matches!(&err, ScxError::Io(e) if e.to_string().contains("framed")),
        "verbatim-copy of a v1 shard into v4 must be rejected, got {err:?}",
    );
}

/// v2 strict shard_type validation: a CSC shard whose
/// `shard_type` byte is corrupted to 0 must be rejected by the
/// reader. This is the new behavior on the v2 catalog read path
/// (catalog-wins tolerance survives only on v1 reads).
#[test]
fn test_strict_shard_type_v2_rejects_corrupted_csc() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("strict_csc.scx");
    let header = sample_header();

    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs()).unwrap();
    writer.write_var(&sample_var()).unwrap();

    let (indptr, indices, values) = sample_shard_data();
    writer
        .write_csc_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();

    let final_path = writer.finish().unwrap();

    // Find the CSC shard's on-disk shard_type byte (offset 5 in
    // the shard header) and corrupt it from 1 → 0.
    let mut data = std::fs::read(&final_path).unwrap();
    let hdr = FileHeader::read_from(&mut std::io::Cursor::new(&data)).unwrap();
    let fc_start = hdr.full_catalog_offset as usize;
    let fc_end = fc_start + hdr.full_catalog_length as usize;
    let catalog = FullCatalog::read_from(
        &mut std::io::Cursor::new(&data[fc_start..fc_end]),
        hdr.full_catalog_length as usize,
        true,
    )
    .unwrap();
    let csc_entry = catalog
        .entries
        .iter()
        .find(|e| e.section_type == SectionType::CscShard)
        .unwrap();
    // shard_type is the 6th byte of the shard header (after the
    // 4-byte magic and 1-byte shard_format_version).
    let shard_type_offset = csc_entry.offset as usize + 4 + 1;
    assert_eq!(data[shard_type_offset], 1, "writer must emit shard_type=1");
    data[shard_type_offset] = 0;

    // Need to rewrite to a new path to preserve the original mmap
    // semantics; the file_checksum will not match either, so open
    // with verify_catalog/header disabled.
    let corrupt_path = dir.path().join("strict_csc_corrupt.scx");
    std::fs::write(&corrupt_path, &data).unwrap();

    // Open and try to read the CSC shard. The strict v2 validator
    // fires inside `read_shard_from_entry_inner` and returns
    // `InvalidShardType`.
    let reader = crate::reader::ScxReader::open_unchecked(&corrupt_path).unwrap();
    let err = reader.read_csc_shard(0).unwrap_err();
    match err {
        ScxError::InvalidShardType {
            expected,
            got,
            section_type,
        } => {
            assert_eq!(expected, 1);
            assert_eq!(got, 0);
            assert_eq!(section_type, SectionType::CscShard as u8);
        }
        other => panic!("expected InvalidShardType, got {other:?}"),
    }
}

/// P3: has_csc flag (bit 0) is auto-set when CSC shards are written
#[test]
fn test_has_csc_flag_set() {
    let dir = tempfile::tempdir().unwrap();

    // File WITHOUT CSC shards: flag should NOT be set
    {
        let path = dir.path().join("no_csc.scx");
        let header = sample_header();
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs()).unwrap();
        writer.write_var(&sample_var()).unwrap();
        let (indptr, indices, values) = sample_shard_data();
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

        let data = std::fs::read(&path).unwrap();
        let mut cursor = std::io::Cursor::new(&data);
        let hdr = FileHeader::read_from(&mut cursor).unwrap();
        assert!(
            !hdr.has_csc(),
            "has_csc should be false when no CSC shards written"
        );
        assert_eq!(hdr.n_csc_shards, 0);
    }

    // File WITH CSC shards: flag SHOULD be set
    {
        let path = dir.path().join("with_csc.scx");
        let header = sample_header();
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs()).unwrap();
        writer.write_var(&sample_var()).unwrap();
        let (indptr, indices, values) = sample_shard_data();
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
            .write_csc_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        writer.finish().unwrap();

        let data = std::fs::read(&path).unwrap();
        let mut cursor = std::io::Cursor::new(&data);
        let hdr = FileHeader::read_from(&mut cursor).unwrap();
        assert!(
            hdr.has_csc(),
            "has_csc should be true when CSC shards written"
        );
        assert_eq!(hdr.n_csc_shards, 1);
    }
}

// -----------------------------------------------------------------------
// Phase A.4 — CSC round-trip and codec sweep
// -----------------------------------------------------------------------

/// Build a 4-row, 6-column dense reference matrix with known entries.
///
/// Returns `(dense_row_major, n_rows, n_cols)`. Used by the
/// multi-shard CSC round-trip test below to sanity-check
/// densification.
fn dense_4x6() -> (Vec<f32>, usize, usize) {
    // Hand-picked sparse pattern across 6 columns; row indices in
    // [0, 4), unsorted within each column to exercise the col_slice
    // / concatenation paths without assuming sorted input.
    let n_rows = 4usize;
    let n_cols = 6usize;
    #[rustfmt::skip]
        let dense: Vec<f32> = vec![
            // col: 0    1    2    3    4    5
                   1.0, 0.0, 0.0, 4.0, 0.0, 7.0,
                   0.0, 2.0, 5.0, 0.0, 0.0, 8.0,
                   0.0, 0.0, 0.0, 0.0, 6.0, 0.0,
                   3.0, 0.0, 0.0, 0.0, 0.0, 9.0,
        ];
    (dense, n_rows, n_cols)
}

/// Build CSC arrays for `cols` (a contiguous range of column
/// indices) over a dense row-major matrix. Returns the on-disk
/// layout: `(indptr_u64, indices_u32, values_le_bytes)`.
fn csc_arrays_for_col_range(
    dense: &[f32],
    n_rows: usize,
    n_cols: usize,
    col_start: usize,
    col_end: usize,
    encoding: ValueEncoding,
) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr: Vec<u64> = Vec::with_capacity(col_end - col_start + 1);
    indptr.push(0);
    let mut indices: Vec<u32> = Vec::new();
    let mut values_f32: Vec<f32> = Vec::new();

    for col in col_start..col_end {
        for row in 0..n_rows {
            let v = dense[row * n_cols + col];
            if v != 0.0 {
                indices.push(row as u32);
                values_f32.push(v);
            }
        }
        indptr.push(indices.len() as u64);
    }

    // Encode values to LE bytes per the requested encoding.
    let mut values_bytes = Vec::with_capacity(values_f32.len() * encoding.byte_width());
    for &v in &values_f32 {
        encoding.encode_f32(&mut values_bytes, v).unwrap();
    }

    (indptr, indices, values_bytes)
}

/// Build a header for a CSC round-trip test fixture.
fn csc_test_header(n_obs: u64, n_vars: u64) -> FileHeader {
    // u32 indices on disk (Phase A test fixtures use n_vars=6
    // which fits in u16, but we want index_dtype to track
    // arrays we hand the writer; the writer reads it from the
    // header). u16 index_dtype byte = 0; u32 = 1.
    FileHeader::new_single_modality(n_obs, n_vars, 0, crate::DEFAULT_SHARD_TARGET_ROWS, 0, 0)
}

/// Round-trip a 2-shard CSC file and verify that
/// `read_all_csc_shards` densifies back to the source matrix.
#[test]
fn test_csc_two_shard_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two_shard_csc.scx");

    let (dense, n_rows, n_cols) = dense_4x6();
    let header = csc_test_header(n_rows as u64, n_cols as u64);

    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs()).unwrap();
    writer.write_var(&sample_var()).unwrap();

    // Need a CSR shard so the file passes basic invariants
    // (`n_obs > 0` requires at least one row-shard for downstream
    // tools); use a tiny 4-row CSR shard with all zeros.
    let csr_indptr = vec![0u64; n_rows + 1];
    writer
        .write_csr_shard(
            &csr_indptr,
            &[],
            &[],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();

    // Shard 1: cols [0..3); shard 2: cols [3..6).
    let (ip1, ix1, vb1) =
        csc_arrays_for_col_range(&dense, n_rows, n_cols, 0, 3, ValueEncoding::Uint8);
    let (ip2, ix2, vb2) =
        csc_arrays_for_col_range(&dense, n_rows, n_cols, 3, 6, ValueEncoding::Uint8);

    writer
        .write_csc_shard(&ip1, &ix1, &vb1, CodecId::None, ValueEncoding::Uint8, 0)
        .unwrap();
    writer
        .write_csc_shard(&ip2, &ix2, &vb2, CodecId::None, ValueEncoding::Uint8, 3)
        .unwrap();

    writer.finish().unwrap();

    // Read back via the high-level CSC API.
    let reader = crate::reader::ScxReader::open(&path).unwrap();
    assert_eq!(reader.csc_shard_count(), 2);

    let csc = reader.read_all_csc_shards().unwrap();
    assert_eq!(csc.shape, (n_rows, n_cols));
    let densified = csc.to_dense().unwrap();
    assert_eq!(densified, dense);

    // Per-shard reads also work.
    let s0 = reader.read_csc_shard(0).unwrap();
    assert_eq!(s0.n_cols(), 3);
    let s1 = reader.read_csc_shard(1).unwrap();
    assert_eq!(s1.n_cols(), 3);

    // CSC entries in the catalog have correct col_start/col_end.
    let csc_entries = reader.catalog().csc_shards_sorted();
    assert_eq!(csc_entries.len(), 2);
    let r0 = csc_entries[0].stats.as_ref().unwrap().col_range();
    let r1 = csc_entries[1].stats.as_ref().unwrap().col_range();
    assert_eq!(r0, 0..3);
    assert_eq!(r1, 3..6);
}

/// Regression test for the per-shard `index_dtype` writer fix.
///
/// Before the fix, `write_csc_shard` stamped every shard with
/// `header.index_dtype` (set from `n_vars` at file creation). Files
/// with `n_obs > 65535` and `n_vars ≤ 65535` would attempt to encode
/// CSC row indices as u16 and fail with
/// `codec error: I/O error: index 65546 exceeds u16 range`.
///
/// The fix derives `index_dtype` per shard from the actual minor-axis
/// bound: `n_obs` for CSC, `n_vars` for CSR. This test pins that
/// behavior with a synthetic file at `n_obs = 70_000, n_vars = 20`
/// (the file header still says `index_dtype = 0`/u16 — that's
/// correct for the CSR shards — but the CSC shard auto-widens to u32).
///
/// Without the writer fix in commit 0bb556e, this test fails at
/// `write_csc_shard` with the u16 overflow error.
#[test]
fn test_csc_shard_large_n_obs_roundtrip() {
    // n_obs > 65535 so CSC row indices need u32 even though n_vars
    // (=20) would fit in u16 if indices were column-style.
    let n_rows: usize = 66_000;
    let n_cols: usize = 20;
    // Per-CSR-shard cap is u16 (BlockIndexEntry::new). Split into 2
    // empty CSR shards of ~33000 rows each.
    let csr_rows_per_shard: usize = 33_000;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("large_n_obs_csc.scx");

    // File-level header says index_dtype = 0 (u16). CSR indices would
    // fit (n_vars=20 < 65535); CSC row indices would NOT.
    let header = csc_test_header(n_rows as u64, n_cols as u64);
    assert_eq!(header.index_dtype, 0);

    // Build obs/var batches sized to the dimensions.
    let obs_ids: Vec<String> = (0..n_rows).map(|i| format!("c{i}")).collect();
    let obs_schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
    let obs = RecordBatch::try_new(
        Arc::new(obs_schema),
        vec![Arc::new(StringArray::from(
            obs_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    let var_ids: Vec<String> = (0..n_cols).map(|i| format!("g{i}")).collect();
    let var_schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    let var = RecordBatch::try_new(
        Arc::new(var_schema),
        vec![Arc::new(StringArray::from(
            var_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();

    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&obs).unwrap();
    writer.write_var(&var).unwrap();

    // Two empty CSR shards (`n_rows + 1` zeros each) to stay under
    // the 65535 rows-per-block limit.
    for shard_start in (0..n_rows).step_by(csr_rows_per_shard) {
        let shard_end = (shard_start + csr_rows_per_shard).min(n_rows);
        let shard_rows = shard_end - shard_start;
        let csr_indptr = vec![0u64; shard_rows + 1];
        writer
            .write_csr_shard(
                &csr_indptr,
                &[],
                &[],
                CodecId::None,
                ValueEncoding::Uint8,
                shard_start as u64,
            )
            .unwrap();
    }

    // CSC shard: every gene gets ONE nonzero at row `65535 + col_idx`.
    // Row indices range from 65535 (just at the u16 boundary) to 65555
    // (clearly past it). Without the writer fix, this trips the u16
    // overflow at write time.
    let mut csc_indptr: Vec<u64> = Vec::with_capacity(n_cols + 1);
    csc_indptr.push(0);
    let mut csc_indices: Vec<u32> = Vec::with_capacity(n_cols);
    let mut csc_values: Vec<u8> = Vec::with_capacity(n_cols);
    for col in 0..n_cols {
        let row_idx = (65_535 + col) as u32; // 65535, 65536, …, 65554
        csc_indices.push(row_idx);
        csc_values.push(((col % 200) as u8).saturating_add(1));
        csc_indptr.push(csc_indices.len() as u64);
    }

    writer
        .write_csc_shard(
            &csc_indptr,
            &csc_indices,
            &csc_values,
            CodecId::None,
            ValueEncoding::Uint8,
            0, // covers cols [0, n_cols)
        )
        .unwrap();
    writer.finish().unwrap();

    // Read back via the high-level CSC API.
    let reader = crate::reader::ScxReader::open(&path).unwrap();
    assert_eq!(reader.csc_shard_count(), 1);
    // File-level index_dtype unchanged from the header (u16); the per-
    // shard index_dtype is what was widened to u32 internally.
    assert_eq!(reader.header().index_dtype, 0);

    let csc = reader.read_csc_shard(0).unwrap();
    assert_eq!(csc.shape, (n_rows, n_cols));
    // Spot-check every column's single nonzero.
    for col in 0..n_cols {
        let start = csc.indptr[col] as usize;
        let end = csc.indptr[col + 1] as usize;
        assert_eq!(end - start, 1, "col {col} should have exactly 1 nonzero");
        let expected_row = (65_535 + col) as i32;
        assert_eq!(
            csc.indices[start], expected_row,
            "col {col}: row index did not roundtrip (n_obs > 65535)"
        );
        let expected_val = ((col % 200) as u8).saturating_add(1) as f32;
        assert_eq!(csc.data[start], expected_val, "col {col}: value mismatch");
    }
}

/// Codec sweep: write a single CSC shard under every supported
/// codec × value-encoding combination and confirm round-trip
/// equality. Pcodec exercises a different decode path than
/// None/Zstd/Lz4Shuffle and is included.
///
/// Scx1 is integer-only; combinations with Float32/Float16 are
/// skipped (they would error at encode time).
#[test]
fn test_csc_codec_sweep() {
    let dir = tempfile::tempdir().unwrap();
    let (dense, n_rows, n_cols) = dense_4x6();

    let codecs = [
        CodecId::None,
        CodecId::Scx1,
        CodecId::Zstd,
        CodecId::Lz4Shuffle,
        CodecId::Pcodec,
    ];
    let encodings = [
        ValueEncoding::Uint8,
        ValueEncoding::Uint16,
        ValueEncoding::Uint32,
        ValueEncoding::Float32,
        ValueEncoding::Float16,
    ];

    for &codec in &codecs {
        for &enc in &encodings {
            if codec == CodecId::Scx1 && !enc.is_integer() {
                continue;
            }
            let label = format!("codec={codec:?}/enc={enc:?}");
            let path = dir
                .path()
                .join(format!("csc_sweep_{}_{}.scx", codec as u8, enc as u8));
            let header = csc_test_header(n_rows as u64, n_cols as u64);

            let mut writer = ScxWriter::new(&path, header).unwrap();
            writer.write_obs(&sample_obs()).unwrap();
            writer.write_var(&sample_var()).unwrap();

            // Empty CSR shard for the file invariant.
            let csr_indptr = vec![0u64; n_rows + 1];
            writer
                .write_csr_shard(
                    &csr_indptr,
                    &[],
                    &[],
                    CodecId::None,
                    ValueEncoding::Uint8,
                    0,
                )
                .unwrap();

            let (ip, ix, vb) = csc_arrays_for_col_range(&dense, n_rows, n_cols, 0, n_cols, enc);
            writer
                .write_csc_shard(&ip, &ix, &vb, codec, enc, 0)
                .unwrap();
            writer.finish().unwrap();

            let reader = crate::reader::ScxReader::open(&path).unwrap();
            let csc = reader.read_all_csc_shards().unwrap();
            let densified = csc.to_dense().unwrap();
            assert_eq!(densified, dense, "round-trip mismatch for {label}");
        }
    }
}

/// `read_csc_columns(range)` — verify that arbitrary contiguous
/// column slices across a multi-shard layout match the
/// densify-then-slice reference. Phase A asserts correctness only;
/// shard-skip count assertions are deferred to Phase E.5 once
/// `BackedCscReader::enable_metrics()` lands.
#[test]
fn test_read_csc_columns_range_correctness() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("csc_range.scx");

    let (dense, n_rows, n_cols) = dense_4x6();
    let header = csc_test_header(n_rows as u64, n_cols as u64);

    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs()).unwrap();
    writer.write_var(&sample_var()).unwrap();
    let csr_indptr = vec![0u64; n_rows + 1];
    writer
        .write_csr_shard(
            &csr_indptr,
            &[],
            &[],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();

    // Three CSC shards: cols [0..2), [2..4), [4..6).
    for (col_start, col_end) in [(0usize, 2usize), (2, 4), (4, 6)] {
        let (ip, ix, vb) = csc_arrays_for_col_range(
            &dense,
            n_rows,
            n_cols,
            col_start,
            col_end,
            ValueEncoding::Uint8,
        );
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
    }
    writer.finish().unwrap();

    let reader = crate::reader::ScxReader::open(&path).unwrap();
    assert_eq!(reader.csc_shard_count(), 3);

    // Reference: dense slice for the same column range.
    let dense_slice = |c_lo: usize, c_hi: usize| -> Vec<f32> {
        let cols = c_hi - c_lo;
        let mut out = vec![0.0f32; n_rows * cols];
        for r in 0..n_rows {
            for (out_c, src_c) in (c_lo..c_hi).enumerate() {
                out[r * cols + out_c] = dense[r * n_cols + src_c];
            }
        }
        out
    };

    let cases = [
        (0u32, 6u32),
        (1, 3), // partial-overlap on shards 0 and 1
        (2, 5), // partial-overlap on shards 1 and 2
        (3, 4), // single shard, partial slice
        (0, 0), // empty range
        (4, 6), // exact shard boundary
    ];
    for (c_lo, c_hi) in cases {
        let csc = reader.read_csc_columns(c_lo..c_hi).unwrap();
        assert_eq!(
            csc.shape,
            (n_rows, (c_hi - c_lo) as usize),
            "shape mismatch for cols [{c_lo}..{c_hi})"
        );
        let got = csc.to_dense().unwrap();
        let want = dense_slice(c_lo as usize, c_hi as usize);
        assert_eq!(got, want, "values mismatch for cols [{c_lo}..{c_hi})");
    }

    // read_csc_columns_subset over a sorted, non-contiguous selection.
    let subset = [0u32, 2, 3, 5];
    let csc = reader.read_csc_columns_subset(&subset).unwrap();
    assert_eq!(csc.shape, (n_rows, subset.len()));
    let got = csc.to_dense().unwrap();
    for (out_c, &src_c) in subset.iter().enumerate() {
        for r in 0..n_rows {
            assert_eq!(
                got[r * subset.len() + out_c],
                dense[r * n_cols + src_c as usize],
                "subset mismatch at row {r} col {src_c}"
            );
        }
    }
}

// -----------------------------------------------------------------------
// Phase B integration tests
// -----------------------------------------------------------------------

/// Full multimodal round-trip: register 3 modalities, write
/// distinct var batches per modality, then read everything back
/// through the per-modality reader API.
#[test]
fn test_phase_b_three_modality_round_trip() {
    use crate::modality::{ModalityFlags, ModalityType};
    use crate::reader::ScxReader;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multimodal.scx");
    let header = sample_header();

    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs()).unwrap();

    // Register three modalities. Order matters — modality_id is
    // 1-based and equals position+1.
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
    let atac_id = writer
        .add_modality(
            "atac",
            ModalityType::Atac,
            CodecId::None,
            ValueEncoding::Uint8,
            false,
        )
        .unwrap();
    assert_eq!(rna_id, 1);
    assert_eq!(adt_id, 2);
    assert_eq!(atac_id, 3);

    // Distinct per-modality var batches. The writer doesn't
    // enforce a relationship between var.num_rows and the
    // modality's n_vars; we set that explicitly.
    let var_rna = sample_var();
    writer.write_var_for(rna_id, &var_rna).unwrap();
    writer.write_var_for(adt_id, &var_rna).unwrap();
    writer.write_var_for(atac_id, &var_rna).unwrap();
    writer.set_modality_n_vars(rna_id, 50).unwrap();
    writer.set_modality_n_vars(adt_id, 50).unwrap();
    writer.set_modality_n_vars(atac_id, 50).unwrap();

    // One CSR shard per modality.
    let (indptr, indices, values) = sample_shard_data();
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
    writer
        .write_csr_shard_for(
            adt_id,
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer
        .write_csr_shard_for(
            atac_id,
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();

    let final_path = writer.finish().unwrap();

    // Read back through ScxReader.
    let reader = ScxReader::open(&final_path).unwrap();
    assert!(reader.is_multimodal());
    assert_eq!(reader.n_modalities(), 3);
    assert_eq!(reader.modality_names(), vec!["rna", "adt", "atac"]);

    assert_eq!(reader.modality_id("rna"), Some(1));
    assert_eq!(reader.modality_id("adt"), Some(2));
    assert_eq!(reader.modality_id("atac"), Some(3));
    assert_eq!(reader.modality_id("missing"), None);

    let info = reader.modality_info(2).unwrap();
    assert_eq!(info.name, "adt");
    assert_eq!(info.modality_type, ModalityType::Protein);
    assert_eq!(info.n_vars, 50);
    assert_eq!(info.n_csr_shards, 1);
    assert_eq!(info.n_csc_shards, 0);
    assert_eq!(info.flags, ModalityFlags::empty());

    // header.has_modalities flag is set.
    assert!(reader.header().has_modalities());

    // Per-modality var read.
    let var_back = reader.read_var_for(rna_id).unwrap();
    assert_eq!(var_back.num_rows(), var_rna.num_rows());

    // Per-modality CSR shard count + read.
    for id in [rna_id, adt_id, atac_id] {
        assert_eq!(reader.csr_shard_count_for(id), 1);
        let (ip, ix, dv) = reader.read_csr_shard_for(id, 0).unwrap();
        assert_eq!(ip.len(), indptr.len());
        assert_eq!(ix.len(), indices.len());
        assert_eq!(dv.len(), values.len());
    }
}

/// Regression test for PR #68: every CSR shard written via
/// `write_csr_shard_for` must stamp `ShardHeader.n_minor` and
/// `ShardStats.col_end` with the modality's own `n_vars` rather
/// than the file-wide `header.n_vars` (which is the max across
/// modalities). The existing 3-modality test uses uniform n_vars
/// so it can't catch the bug.
#[test]
fn test_multimodal_shard_stats_use_per_modality_n_vars() {
    use crate::modality::ModalityType;
    use crate::reader::ScxReader;
    use crate::section::SectionType;
    use crate::shard::ShardHeader;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("per_modality_nvars.scx");

    // header.n_vars is the file-wide max across modalities.
    let mut header = sample_header();
    header.n_vars = 200;

    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs()).unwrap();

    // Three modalities with DISTINCT n_vars; "atac" matches the
    // header max, "rna" / "adt" do not. This ensures any path that
    // accidentally falls back to header.n_vars (= 200) gets caught
    // for the latter two.
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
    let atac_id = writer
        .add_modality(
            "atac",
            ModalityType::Atac,
            CodecId::None,
            ValueEncoding::Uint8,
            false,
        )
        .unwrap();

    let expected: [(u8, u64); 3] = [(rna_id, 30), (adt_id, 12), (atac_id, 200)];

    writer.write_var_for(rna_id, &sample_var()).unwrap();
    writer.write_var_for(adt_id, &sample_var()).unwrap();
    writer.write_var_for(atac_id, &sample_var()).unwrap();
    for (id, n_vars) in expected {
        writer.set_modality_n_vars(id, n_vars).unwrap();
    }

    let (indptr, indices, values) = sample_shard_data();
    for (id, _) in expected {
        writer
            .write_csr_shard_for(
                id,
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
    }

    let final_path = writer.finish().unwrap();

    let reader = ScxReader::open(&final_path).unwrap();
    for (id, n_vars) in expected {
        let shards: Vec<&FullCatalogEntry> = reader
            .catalog()
            .shards(SectionType::CsrShard)
            .into_iter()
            .filter(|e| e.modality_id == id)
            .collect();
        assert_eq!(
            shards.len(),
            1,
            "modality_id {id} should have exactly 1 CSR shard"
        );
        let entry = shards[0];

        // Catalog stats: row-major shards stamp col_end = n_minor.
        let stats = entry
            .stats
            .as_ref()
            .expect("v2 catalog must carry shard stats");
        assert_eq!(
            stats.col_end, n_vars,
            "ShardStats.col_end for modality {id} should equal that \
                 modality's n_vars ({n_vars}), got {} (header.n_vars=200)",
            stats.col_end
        );
        assert_eq!(stats.col_start, 0);

        // On-disk shard header: n_minor field must also match.
        let bytes = reader.section_bytes(entry).unwrap();
        let sh = ShardHeader::read_from(&mut std::io::Cursor::new(
            &bytes[..crate::shard::SHARD_HEADER_SIZE],
        ))
        .unwrap();
        assert_eq!(
            sh.n_minor as u64, n_vars,
            "ShardHeader.n_minor for modality {id} should equal {n_vars}, \
                 got {}",
            sh.n_minor
        );
    }
}

/// Single-modality v2 file: no `add_modality` calls means no
/// `ModalityTable` section is emitted. The on-disk shape and the
/// reader-visible accessors match a v1 file.
#[test]
fn test_phase_b_single_modality_no_modality_table() {
    use crate::reader::ScxReader;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("single_modality.scx");
    let header = sample_header();

    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs()).unwrap();
    writer.write_var(&sample_var()).unwrap();
    let (indptr, indices, values) = sample_shard_data();
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
    let final_path = writer.finish().unwrap();

    let reader = ScxReader::open(&final_path).unwrap();
    assert!(!reader.is_multimodal());
    assert_eq!(reader.n_modalities(), 0);
    assert!(reader.modality_names().is_empty());
    assert!(!reader.header().has_modalities());
    assert_eq!(reader.header().modality_table_offset, 0);
    assert_eq!(reader.header().modality_table_length, 0);
    assert_eq!(reader.modality_info(0), None);
    assert_eq!(reader.modality_info(1), None);
    // Global accessors continue to work.
    assert_eq!(reader.read_var_for(0).unwrap().num_rows(), 2);
}

/// Per-modality CSC sidecars produce shard counts on the right
/// modality's `ModalityInfo`, and the `BackedCscReader::for_modality`
/// constructor scopes shard reads to that modality.
#[test]
fn test_phase_b_per_modality_csc() {
    use crate::backed::BackedCscReader;
    use crate::modality::ModalityType;
    use crate::reader::ScxReader;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multimodal_csc.scx");
    let header = sample_header();

    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs()).unwrap();
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
    writer.write_var_for(rna_id, &sample_var()).unwrap();
    writer.write_var_for(adt_id, &sample_var()).unwrap();
    writer.set_modality_n_vars(rna_id, 50).unwrap();
    writer.set_modality_n_vars(adt_id, 50).unwrap();

    let (indptr, indices, values) = sample_shard_data();
    // RNA gets a CSR shard only.
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
    // ADT gets BOTH CSR and CSC.
    writer
        .write_csr_shard_for(
            adt_id,
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer
        .write_csc_shard_for(
            adt_id,
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();

    let final_path = writer.finish().unwrap();
    let reader = ScxReader::open(&final_path).unwrap();

    // Per-modality counts reflect the writes.
    assert_eq!(reader.csc_shard_count_for(rna_id), 0);
    assert_eq!(reader.csc_shard_count_for(adt_id), 1);
    let adt_info = reader.modality_info(adt_id).unwrap();
    assert!(adt_info.flags.has_csc());
    let rna_info = reader.modality_info(rna_id).unwrap();
    assert!(!rna_info.flags.has_csc());

    // BackedCscReader scoped to RNA sees zero shards; scoped to
    // ADT sees the one shard. This is the cache-isolation
    // guarantee from B.5.
    let rna_csc =
        BackedCscReader::for_modality(ScxReader::open(&final_path).unwrap(), rna_id, 4).unwrap();
    assert_eq!(rna_csc.n_shards(), 0);
    let adt_csc =
        BackedCscReader::for_modality(ScxReader::open(&final_path).unwrap(), adt_id, 4).unwrap();
    assert_eq!(adt_csc.n_shards(), 1);
}

/// Phase B.6: `add_modality(..., build_csc=true)` triggers an
/// auto-emit transpose pass at finish() time. After finish(), the
/// modality's `n_csc_shards >= 1` and `flags.has_csc() == true`,
/// even though the caller never invoked `write_csc_shard_for` —
/// the writer read the CSR shards back from its temp file and
/// streamed them through `streaming_csr_to_csc_iter_with_cap`.
#[test]
fn test_phase_b3_auto_emit_csc() {
    use crate::modality::ModalityType;
    use crate::reader::ScxReader;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("auto_emit_csc.scx");
    let header = sample_header();
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs()).unwrap();

    let rna_id = writer
        .add_modality(
            "rna",
            ModalityType::Rna,
            CodecId::None,
            ValueEncoding::Uint8,
            true, // build_csc — Phase B.3 auto-emit
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
    writer.set_modality_n_vars(rna_id, 50).unwrap();
    writer.set_modality_n_vars(adt_id, 50).unwrap();
    writer.write_var_for(rna_id, &sample_var()).unwrap();
    writer.write_var_for(adt_id, &sample_var()).unwrap();

    // Both modalities get one CSR shard. Only `rna`'s
    // `build_csc=true`, so only its CSC sidecar should
    // auto-emit.
    let (indptr, indices, values) = sample_shard_data();
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
    writer
        .write_csr_shard_for(
            adt_id,
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();

    let final_path = writer.finish().unwrap();
    let reader = ScxReader::open(&final_path).unwrap();

    // RNA picked up the auto-emit; ADT did not.
    assert!(
        reader.csc_shard_count_for(rna_id) >= 1,
        "rna should have at least one auto-emitted CSC shard"
    );
    assert_eq!(
        reader.csc_shard_count_for(adt_id),
        0,
        "adt build_csc=false → no CSC sidecar"
    );
    assert!(reader.modality_info(rna_id).unwrap().flags.has_csc());
    assert!(!reader.modality_info(adt_id).unwrap().flags.has_csc());

    // The auto-emitted CSC stores the same nnz as the CSR. We
    // compare nnz rather than densifying because the CSR shape is
    // (n_shard_rows, n_modality_vars) while the CSC shape uses
    // file-wide n_obs (the column-axis slice covers the full obs
    // range, with zero rows for cells absent from the CSR shard).
    let csr = reader.read_all_csr_shards_for(rna_id).unwrap();
    let csc = reader.read_all_csc_shards_for(rna_id).unwrap();
    assert_eq!(
        *csr.indptr.last().unwrap_or(&0),
        *csc.indptr.last().unwrap_or(&0),
        "CSR and CSC nnz must agree after auto-emit"
    );
}

/// Phase B.4: per-modality CSC column-range reads return columns
/// from the right modality only.
#[test]
fn test_phase_b4_read_csc_columns_for() {
    use crate::modality::ModalityType;
    use crate::reader::ScxReader;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("b4_csc_columns_for.scx");
    let header = sample_header();
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs()).unwrap();

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
    writer.set_modality_n_vars(rna_id, 50).unwrap();
    writer.set_modality_n_vars(adt_id, 50).unwrap();
    writer.write_var_for(rna_id, &sample_var()).unwrap();
    writer.write_var_for(adt_id, &sample_var()).unwrap();

    let (indptr, indices, values) = sample_shard_data();
    // Both modalities get one CSC shard at col_start=0.
    writer
        .write_csc_shard_for(
            rna_id,
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer
        .write_csc_shard_for(
            adt_id,
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();

    let final_path = writer.finish().unwrap();
    let reader = ScxReader::open(&final_path).unwrap();

    // Per-modality CSC counters reflect what was written.
    assert_eq!(reader.csc_shard_count_for(rna_id), 1);
    assert_eq!(reader.csc_shard_count_for(adt_id), 1);

    // Per-modality CSC range read returns the modality's
    // contribution. We assert the call succeeds and returns a
    // non-empty result (exact column-slice semantics are
    // covered by the single-modality `read_csc_columns` tests).
    let rna_cols = reader.read_csc_columns_for(rna_id, 0..3).unwrap();
    assert!(
        rna_cols.shape.1 >= 1,
        "rna CSC range read should return ≥ 1 col"
    );
    let rna_subset = reader
        .read_csc_columns_subset_for(rna_id, &[0u32, 2])
        .unwrap();
    assert!(rna_subset.shape.1 >= 1);
}

// -----------------------------------------------------------------------
// Defensive tests (Patch 9): write_preencoded_shard CSC counting
// -----------------------------------------------------------------------

#[test]
fn preencoded_csc_shard_increments_csc_count() {
    use crate::shard::{BlockIndex, BlockIndexEntry, ShardHeader, SHARD_HEADER_SIZE, SHARD_MAGIC};
    use scx_codec::{CodecId, ValueEncoding};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("csc_preencoded.scx");

    let mut header = sample_header();
    header.n_obs = 3;
    header.n_vars = 2;
    header.nnz = 0;

    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs()).unwrap();
    writer.write_var(&sample_var()).unwrap();

    // First, write a normal CSR shard
    let (indptr, indices, values) = sample_shard_data();
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

    // Now craft a PreEncodedSection with section_type = CscShard
    let csc_indptr = vec![0u64, 1, 3]; // 2 columns
    let csc_indices = vec![0u32, 1, 2]; // 3 entries
    let csc_values: Vec<u8> = vec![10, 20, 30];

    let encoded = scx_codec::encode_shard(
        &csc_indptr,
        &csc_indices,
        &csc_values,
        CodecId::None,
        ValueEncoding::Uint8,
        true,
    )
    .unwrap();

    let block_index = BlockIndex {
        entries: vec![BlockIndexEntry::new(0, 2, 0, 0, 0, 3).unwrap()],
    };
    let mut bi_buf = Vec::new();
    block_index.write_to(&mut bi_buf).unwrap();

    let sh = ShardHeader {
        magic: SHARD_MAGIC,
        shard_format_version: 1,
        shard_type: 1, // CSC
        codec_id: CodecId::None as u8,
        value_encoding: ValueEncoding::Uint8 as u8,
        index_dtype: 0,
        reserved_flags: [0; 3],
        n_major: 2,
        n_minor: 3,
        nnz: 3,
        global_offset: 0,
        indptr_rel_offset: SHARD_HEADER_SIZE as u32,
        indptr_length: encoded.indptr_bytes.len() as u32,
        indices_rel_offset: SHARD_HEADER_SIZE as u32 + encoded.indptr_bytes.len() as u32,
        indices_length: encoded.indices_bytes.len() as u32,
        values_rel_offset: SHARD_HEADER_SIZE as u32
            + encoded.indptr_bytes.len() as u32
            + encoded.indices_bytes.len() as u32,
        values_length: encoded.values_bytes.len() as u32,
        block_index_rel_offset: SHARD_HEADER_SIZE as u32
            + encoded.indptr_bytes.len() as u32
            + encoded.indices_bytes.len() as u32
            + encoded.values_bytes.len() as u32,
        block_index_length: bi_buf.len() as u32,
        checksum: [0; 8], // dummy, we'll compute the real one
    };
    let mut hdr_buf = Vec::new();
    sh.write_to(&mut hdr_buf).unwrap();

    // Compute checksum from payload
    let mut payload = Vec::new();
    payload.extend_from_slice(&encoded.indptr_bytes);
    payload.extend_from_slice(&encoded.indices_bytes);
    payload.extend_from_slice(&encoded.values_bytes);
    payload.extend_from_slice(&bi_buf);
    let shard_checksum = crate::checksum::blake3_truncated_64(&payload);

    // Rewrite header with correct checksum
    let sh_corrected = ShardHeader {
        checksum: shard_checksum,
        ..sh
    };
    hdr_buf.clear();
    sh_corrected.write_to(&mut hdr_buf).unwrap();

    // Build full section for checksum
    let mut full_section = Vec::new();
    full_section.extend_from_slice(&hdr_buf);
    full_section.extend_from_slice(&payload);
    let section_checksum = crate::checksum::blake3_hash(&full_section);
    let section_length = full_section.len() as u64;

    let stats = compute_shard_stats(
        &csc_values,
        ValueEncoding::Uint8,
        MajorAxis::Col,
        0,
        2,
        3,
        3,
    );

    let pre = PreEncodedSection {
        encoded,
        block_index_bytes: bi_buf,
        header_buf: hdr_buf,
        section_checksum,
        section_length,
        stats,
        name: "X_csc_shard_0".to_string(),
        section_type: SectionType::CscShard,
        nnz: 3,
    };

    writer.write_preencoded_shard(pre).unwrap();
    let final_path = writer.finish().unwrap();

    // Verify the header now reports 1 CSC shard
    let reader = crate::reader::ScxReader::open(&final_path).unwrap();
    assert_eq!(
        reader.csc_shard_count(),
        1,
        "write_preencoded_shard should count CSC shards"
    );
    assert_eq!(reader.header().n_csr_shards, 1);
    assert_eq!(
        reader.header().n_csc_shards,
        1,
        "header n_csc_shards should reflect the preencoded CSC shard"
    );
    assert!(
        reader.header().has_csc(),
        "header has_csc flag should be set after writing a CSC shard"
    );
}

/// B4 regression: per-modality `nnz` / `n_csr_shards` must accumulate on the
/// modality table when CSR shards are written via the parallel
/// (`write_preencoded_shard`) and byte-passthrough (`copy_section_verbatim`)
/// paths inside a `with_modality` scope — these are the paths the streaming
/// h5mu convert / SCX→SCX rewrite use. Before the fix the modality table
/// reported `nnz 0` / `csr 0` for every modality even though the file totals
/// were correct.
#[test]
fn preencoded_and_verbatim_shards_accumulate_per_modality_stats() {
    use crate::reader::ScxReader;

    let dir = tempfile::tempdir().unwrap();

    // --- Build a source single-modality file to copy a shard from verbatim ---
    let src_path = dir.path().join("source.scx");
    let mut src_header = sample_header();
    src_header.n_obs = 3;
    src_header.n_vars = 50;
    src_header.nnz = 0;
    let mut src_writer = ScxWriter::new(&src_path, src_header).unwrap();
    src_writer.write_obs(&sample_obs()).unwrap();
    src_writer.write_var(&sample_var()).unwrap();
    let (indptr, indices, values) = sample_shard_data();
    let src_nnz = *indptr.last().unwrap();
    src_writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    let src_final = src_writer.finish().unwrap();
    let src_reader = ScxReader::open(&src_final).unwrap();
    let src_entry = src_reader.catalog().shards_sorted()[0].clone();
    let src_bytes = src_reader
        .read_raw_shard_bytes(&src_entry)
        .unwrap()
        .to_vec();

    // --- Build a target multimodal file ---
    let target_path = dir.path().join("multimodal.scx");
    let mut writer = ScxWriter::new(&target_path, sample_header()).unwrap();
    writer.write_obs(&sample_obs()).unwrap();

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
    writer.write_var_for(rna_id, &sample_var()).unwrap();
    writer.write_var_for(adt_id, &sample_var()).unwrap();
    writer.set_modality_n_vars(rna_id, 50).unwrap();
    writer.set_modality_n_vars(adt_id, 50).unwrap();

    // rna: parallel/pre-encoded path.
    let f32_vals: Vec<f32> = values.iter().map(|&v| v as f32).collect();
    let pre = crate::encoder::encode_one_shard(
        &indptr,
        &indices,
        &f32_vals,
        Some(CodecId::None),
        0,
        50,
        0,
        SectionType::CsrShard,
        ModalityType::Rna,
        "X/rna/shard_0".to_string(),
        None,
    )
    .unwrap();
    let pre_nnz = pre.nnz;
    writer
        .with_modality::<_, _, ScxError>(rna_id, |w| w.write_preencoded_shard(pre))
        .unwrap();

    // adt: byte-passthrough / verbatim copy path.
    writer
        .with_modality::<_, _, ScxError>(adt_id, |w| {
            w.copy_section_verbatim(&src_entry, &src_bytes)
        })
        .unwrap();

    let final_path = writer.finish().unwrap();

    // --- Verify per-modality stats are populated (not zero) ---
    let reader = ScxReader::open(&final_path).unwrap();
    let rna_info = reader.modality_info(rna_id).unwrap();
    assert_eq!(rna_info.name, "rna");
    assert_eq!(
        rna_info.n_csr_shards, 1,
        "rna preencoded CSR shard must count toward n_csr_shards"
    );
    assert_eq!(
        rna_info.nnz, pre_nnz,
        "rna per-modality nnz must equal the preencoded shard nnz"
    );

    let adt_info = reader.modality_info(adt_id).unwrap();
    assert_eq!(adt_info.name, "adt");
    assert_eq!(
        adt_info.n_csr_shards, 1,
        "adt verbatim-copied CSR shard must count toward n_csr_shards"
    );
    assert_eq!(
        adt_info.nnz, src_nnz,
        "adt per-modality nnz must equal the copied shard nnz"
    );

    // Per-modality totals reconcile with the file-level total.
    assert_eq!(rna_info.nnz + adt_info.nnz, reader.header().nnz);
    assert_eq!(pre_nnz, src_nnz, "both shards carry the same sample nnz");
}

/// Write a minimal valid file and return its path (plus the owning tempdir,
/// which must stay alive for the file to exist).
fn write_minimal_file() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("checksum_test.scx");
    let mut writer = ScxWriter::new(&path, sample_header()).unwrap();
    writer.write_obs(&sample_obs()).unwrap();
    writer.write_var(&sample_var()).unwrap();
    let (indptr, indices, values) = sample_shard_data();
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
    let final_path = writer.finish().unwrap();
    (dir, final_path)
}

/// A freshly written file's stored file_checksum matches a recomputation, and
/// `validate()` reports the `file_checksum` entry as passing.
#[test]
fn verify_file_checksum_passes_on_clean_file() {
    let (_dir, path) = write_minimal_file();
    let reader = crate::reader::ScxReader::open(&path).unwrap();

    assert!(
        reader.verify_file_checksum().unwrap(),
        "clean file must match its stored file_checksum"
    );

    let results = reader.validate().unwrap();
    let (name, passed) = &results[0];
    assert_eq!(name, "file_checksum");
    assert!(passed, "file_checksum entry must pass on a clean file");
}

/// Flipping a byte in the 256-byte header (here `n_obs`) leaves the file
/// openable and every per-section catalog checksum intact, but the whole-file
/// `file_checksum` no longer matches — so `validate()` now fails. This is the
/// corruption class that previously passed `scx validate` clean.
#[test]
fn validate_detects_header_corruption() {
    let (dir, path) = write_minimal_file();

    let mut data = std::fs::read(&path).unwrap();
    // `n_obs` is at byte offset 12: magic(4) + format_version(2) +
    // header_length(2) + flags(4). It is not cross-checked at open and is
    // covered by no per-section catalog checksum.
    data[12] ^= 0xFF;
    let corrupt_path = dir.path().join("header_corrupt.scx");
    std::fs::write(&corrupt_path, &data).unwrap();

    // The file still opens (the corrupt byte is not in the catalog) ...
    let reader = crate::reader::ScxReader::open(&corrupt_path).unwrap();
    // ... but the whole-file checksum no longer matches.
    assert!(
        !reader.verify_file_checksum().unwrap(),
        "header corruption must be caught by verify_file_checksum"
    );
    // validate() still returns Ok (file_checksum is non-essential — see its
    // doc note) but flags the file_checksum entry as failed. Every per-section
    // checksum is intact, which is exactly why this slipped through before.
    let results = reader.validate().unwrap();
    let file_ok = results
        .iter()
        .find(|(name, _)| name == "file_checksum")
        .map(|(_, p)| *p)
        .expect("file_checksum entry must be present");
    assert!(!file_ok, "validate() must flag header corruption");
    assert!(
        results
            .iter()
            .filter(|(name, _)| name != "file_checksum")
            .all(|(_, p)| *p),
        "per-section checksums stay intact under header corruption"
    );
}

/// Flipping a byte in the 4096-byte root catalog region (`[256..4352]`) is also
/// caught by the whole-file checksum.
#[test]
fn validate_detects_root_catalog_corruption() {
    let (dir, path) = write_minimal_file();

    let mut data = std::fs::read(&path).unwrap();
    // Pick a byte inside the root-catalog region but past the live entries
    // (the trailing padding) so the file still opens.
    data[HEADER_SIZE + 2048] ^= 0xFF;
    let corrupt_path = dir.path().join("root_corrupt.scx");
    std::fs::write(&corrupt_path, &data).unwrap();

    let reader = crate::reader::ScxReader::open_unchecked(&corrupt_path).unwrap();
    assert!(
        !reader.verify_file_checksum().unwrap(),
        "root-catalog corruption must be caught by verify_file_checksum"
    );
}

/// F1 oversized-group fix: a CSR shard with more than `MAX_BLOCK_ROWS`
/// (65,535) rows must write (splitting into multiple blocks) and read back
/// byte-identically. Before the fix the writer emitted a single block and
/// `BlockIndexEntry::new` failed with `BlockRowsOverflow`. This mirrors a
/// grouped shard holding one large group (the never-split-a-group invariant
/// disables the per-shard row cap).
#[test]
fn csr_shard_over_u16_rows_round_trips_as_multiple_blocks() {
    use crate::shard::{BlockIndex, ShardHeader, MAX_BLOCK_ROWS, SHARD_HEADER_SIZE};
    use std::io::Cursor;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("oversized_csr.scx");

    let n_rows: usize = 70_000; // > 65_535 → must split into 2 blocks
    let n_cols: u32 = 10;
    let mut header = FileHeader::new_single_modality(
        n_rows as u64,
        n_cols as u64,
        n_rows as u64, // nnz: one per row
        crate::DEFAULT_SHARD_TARGET_ROWS,
        0,
        0,
    );
    header.codec_id = CodecId::None as u8;
    header.index_dtype = 0;

    let mut writer = ScxWriter::new(&path, header).unwrap();
    // Minimal obs/var (row/col labels are irrelevant to the block-index path).
    let obs = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "cell_id",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(
            (0..n_rows).map(|i| format!("c{i}")).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    let var = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "gene_id",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(
            (0..n_cols).map(|i| format!("g{i}")).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    writer.write_obs(&obs).unwrap();
    writer.write_var(&var).unwrap();

    // One nonzero per row at column (row % n_cols), value 1.
    let indptr: Vec<u64> = (0..=n_rows as u64).collect();
    let indices: Vec<u32> = (0..n_rows).map(|r| (r as u32) % n_cols).collect();
    let values: Vec<u8> = vec![1u8; n_rows];

    // Must NOT error (pre-fix: BlockRowsOverflow at write time).
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
    let final_path = writer.finish().unwrap();

    let reader = crate::reader::ScxReader::open(&final_path).unwrap();
    let entry = reader.catalog().shards_sorted()[0];

    // The writer emitted multiple ≤MAX_BLOCK_ROWS blocks.
    let raw = reader.section_bytes(entry).unwrap();
    let sh = ShardHeader::read_from(&mut Cursor::new(&raw[..SHARD_HEADER_SIZE])).unwrap();
    let bi_start = sh.block_index_rel_offset as usize;
    let bi_end = bi_start + sh.block_index_length as usize;
    let bi =
        BlockIndex::read_from(&mut Cursor::new(&raw[bi_start..bi_end]), bi_end - bi_start).unwrap();
    assert_eq!(
        bi.entries.len(),
        2,
        "70k-row shard must split into 2 blocks"
    );
    assert_eq!(bi.entries[0].n_rows, u16::MAX);
    assert_eq!(
        bi.entries[1].n_rows as usize,
        n_rows - MAX_BLOCK_ROWS as usize
    );
    assert_eq!(bi.entries[1].row_start, MAX_BLOCK_ROWS);

    // Whole-shard decode round-trips exactly (the read path is block-agnostic).
    let (rt_indptr, rt_indices, rt_data) = reader.read_shard_from_entry(entry).unwrap();
    assert_eq!(rt_indptr.len(), n_rows + 1);
    assert_eq!(*rt_indptr.last().unwrap(), n_rows as i64);
    assert_eq!(rt_indices.len(), n_rows);
    assert!(rt_indices
        .iter()
        .enumerate()
        .all(|(r, &c)| c == (r as i32) % n_cols as i32));
    assert!(rt_data.iter().all(|&v| v == 1.0));
}

#[test]
fn duplicate_section_name_is_rejected() {
    // SCX-015: writing the same logical section twice produces a file whose
    // second section is silently unreachable (readers resolve to the first
    // match). The writer must reject it at the single write choke-point.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dup.scx");
    let mut writer = ScxWriter::new(&path, sample_header()).unwrap();
    writer.write_obs(&sample_obs()).unwrap();
    let err = writer.write_obs(&sample_obs()).unwrap_err();
    assert!(
        matches!(err, ScxError::DuplicateSection { .. }),
        "expected DuplicateSection, got {err:?}"
    );
}
