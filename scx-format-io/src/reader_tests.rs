use super::*;
use crate::header::CURRENT_FORMAT_VERSION;
use crate::provenance::ProvenanceEntry;
use crate::shard::SHARD_HEADER_SIZE;
use crate::writer::ScxWriter;
use arrow::array::{Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

#[test]
fn open_missing_file_error_includes_path() {
    let missing = "/nonexistent/scx-user/does_not_exist_xyz.scx";
    let msg = match ScxReader::open_unchecked(missing) {
        Ok(_) => panic!("expected open of a nonexistent path to fail"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains(missing),
        "open error should echo the offending path, got: {msg}"
    );
}

fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, n_vars, nnz, 16384, 0, 0)
}

fn sample_obs(n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn sample_var(n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

/// Build a small shard: n_rows rows, each with some nonzeros in n_vars columns.
fn sample_shard_data(n_rows: usize, n_vars: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();

    for row in 0..n_rows {
        // Each row has 2 nonzeros
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

/// Write a complete test file and return the path.
fn write_test_file(
    dir: &tempfile::TempDir,
    filename: &str,
    n_obs: usize,
    n_vars: usize,
    n_shards: usize,
    include_extras: bool,
) -> std::path::PathBuf {
    let path = dir.path().join(filename);
    let total_nnz = n_obs * 2; // 2 nnz per row
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
        // obsm
        let obsm_schema = Schema::new(vec![
            Field::new("pc1", DataType::Float32, false),
            Field::new("pc2", DataType::Float32, false),
        ]);
        let obsm_batch = RecordBatch::try_new(
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

        // uns
        writer
            .write_uns(&serde_json::json!({"species": "human", "version": 2}))
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
    }

    writer.finish().unwrap();
    path
}

// -----------------------------------------------------------------------
// 11.15: Full round-trip test
// -----------------------------------------------------------------------

#[test]
fn test_reader_full_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "full.scx", 6, 10, 2, true);

    let reader = ScxReader::open(&path).unwrap();

    // Summary accessors
    assert_eq!(reader.n_obs(), 6);
    assert_eq!(reader.n_vars(), 10);
    assert_eq!(reader.nnz(), 12);
    assert_eq!(reader.header().format_version, CURRENT_FORMAT_VERSION);

    // read_obs
    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 6);
    assert_eq!(obs.num_columns(), 1);

    // read_var
    let var = reader.read_var().unwrap();
    assert_eq!(var.num_rows(), 10);
    assert_eq!(var.num_columns(), 1);

    // read_csr_shard(0) — first shard
    let (indptr, indices, data) = reader.read_csr_shard(0).unwrap();
    assert_eq!(indptr.len(), 4); // 3 rows + 1
    assert_eq!(indices.len(), 6); // 3 rows * 2 nnz
    assert_eq!(data.len(), 6);

    // read_all_csr_shards
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape, (6, 10));
    assert_eq!(csr.indptr.len(), 7); // 6 rows + 1
    assert_eq!(csr.nnz(), 12); // 6 rows * 2

    // read_obsm
    let obsm = reader.read_obsm("X_pca").unwrap();
    assert_eq!(obsm.num_rows(), 6);
    assert_eq!(obsm.num_columns(), 2);

    // read_all_obsm
    let all_obsm = reader.read_all_obsm().unwrap();
    assert_eq!(all_obsm.len(), 1);
    assert!(all_obsm.contains_key("X_pca"));

    // read_uns
    let uns = reader.read_uns().unwrap();
    assert_eq!(uns["species"], "human");
    assert_eq!(uns["version"], 2);

    // read_provenance
    let prov = reader.read_provenance().unwrap();
    assert_eq!(prov.operations.len(), 1);
    assert_eq!(prov.operations[0].action, "convert");
}

/// Sharded obsm round-trip: three row-shards of the same logical
/// matrix should reassemble byte-equal to the unsharded equivalent.
#[test]
fn test_sharded_obsm_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sharded_obsm.scx");

    let n_obs: usize = 9;
    let n_vars: usize = 4;
    let header = FileHeader::new_single_modality(n_obs as u64, n_vars as u64, 0, 3, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    // Single CSR shard so the file passes its catalog invariants.
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

    // 3 row-shards of (3 × 2) obsm, values pc1 = row index, pc2 = 2*row.
    let obsm_schema = Arc::new(Schema::new(vec![
        Field::new("pc1", DataType::Float32, false),
        Field::new("pc2", DataType::Float32, false),
    ]));
    for shard_idx in 0u32..3 {
        let row_start = shard_idx as usize * 3;
        let rows: Vec<f32> = (row_start..row_start + 3).map(|r| r as f32).collect();
        let rows2: Vec<f32> = rows.iter().map(|r| r * 2.0).collect();
        let batch = RecordBatch::try_new(
            obsm_schema.clone(),
            vec![
                Arc::new(Float32Array::from(rows)),
                Arc::new(Float32Array::from(rows2)),
            ],
        )
        .unwrap();
        writer
            .write_obsm_shard(
                "X_pca",
                shard_idx,
                row_start as u64,
                batch.num_rows() as u64,
                n_obs as u64,
                &batch,
            )
            .unwrap();
    }

    writer.finish().unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let pca = reader.read_obsm("X_pca").unwrap();
    assert_eq!(pca.num_rows(), n_obs);
    assert_eq!(pca.num_columns(), 2);
    let pc1 = pca
        .column(0)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    for (i, v) in pc1.values().iter().enumerate() {
        assert_eq!(*v, i as f32);
    }
    let all = reader.read_all_obsm().unwrap();
    assert_eq!(all.len(), 1);
    assert!(all.contains_key("X_pca"));
    let names = reader.list_obsm();
    assert_eq!(names, vec!["X_pca".to_string()]);
}

/// A file written with the legacy single-section obsm path must
/// keep reading correctly after the sharded-aware reader changes.
/// This is the backward-compatibility guarantee for files written
/// before sharding was introduced.
#[test]
fn test_legacy_single_section_obsm_still_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "legacy_obsm.scx", 6, 10, 2, true);
    let reader = ScxReader::open(&path).unwrap();

    let obsm = reader.read_obsm("X_pca").unwrap();
    assert_eq!(obsm.num_rows(), 6);
    assert_eq!(obsm.num_columns(), 2);

    let all = reader.read_all_obsm().unwrap();
    assert_eq!(all.len(), 1);
    assert!(all.contains_key("X_pca"));
}

/// Helper for the sharded-reader regression tests: writes a minimal
/// SCX file with one CSR shard and `obsm/X_pca` split into
/// `n_obs / shard_rows` dense shards, then hands the caller the
/// `ScxWriter` mid-flight so it can override the obsm shard layout
/// (skip a shard, duplicate a `shard_idx`, etc.) before `finish()`.
fn build_obsm_test_writer(
    path: &std::path::Path,
    n_obs: usize,
    n_vars: usize,
    shard_rows: u32,
) -> ScxWriter {
    let header = FileHeader::new_single_modality(n_obs as u64, n_vars as u64, 0, shard_rows, 0, 0);
    let mut writer = ScxWriter::new(path, header).unwrap();
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
}

fn dense_obsm_shard_batch(row_start: usize, n_rows: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("pc1", DataType::Float32, false),
        Field::new("pc2", DataType::Float32, false),
    ]));
    let rows: Vec<f32> = (row_start..row_start + n_rows).map(|r| r as f32).collect();
    let rows2: Vec<f32> = rows.iter().map(|r| r * 2.0).collect();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float32Array::from(rows)),
            Arc::new(Float32Array::from(rows2)),
        ],
    )
    .unwrap()
}

/// A sharded `obsm/X_pca` whose middle shard is missing must fail
/// reads with `InvalidCatalog`, not silently return a truncated
/// matrix. Regression guard for the contiguity-check fix.
#[test]
fn test_sharded_read_rejects_missing_shard() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("missing_shard.scx");
    let n_obs = 9usize;
    let mut writer = build_obsm_test_writer(&path, n_obs, 4, 3);

    // Write shard 0 and shard 2 only — shard 1 is missing.
    for &shard_idx in &[0u32, 2u32] {
        let row_start = shard_idx as usize * 3;
        let batch = dense_obsm_shard_batch(row_start, 3);
        writer
            .write_obsm_shard(
                "X_pca",
                shard_idx,
                row_start as u64,
                batch.num_rows() as u64,
                n_obs as u64,
                &batch,
            )
            .unwrap();
    }
    writer.finish().unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let err = reader.read_obsm("X_pca").unwrap_err();
    assert!(
        matches!(err, ScxError::InvalidCatalog(_)),
        "expected InvalidCatalog, got {err:?}"
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("X_pca"),
        "error should name the logical section: {msg}"
    );
}

/// Two shards with the same `shard_idx` must be rejected as
/// `InvalidCatalog` — the second shard's stamped `shard_idx`
/// won't match its position after sorting.
#[test]
fn test_sharded_read_rejects_duplicate_shard_idx() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dup_shard.scx");
    let n_obs = 6usize;
    let mut writer = build_obsm_test_writer(&path, n_obs, 4, 3);

    // Two physical shards both stamped with shard_idx = 0. The
    // second one's catalog name is `..._shard_1` (so the catalog
    // walk picks both up), but its schema-stamped `shard_idx` is
    // still 0 — the reader should reject the position/stamp
    // mismatch.
    let batch0 = dense_obsm_shard_batch(0, 3);
    writer
        .write_obsm_shard(
            "X_pca",
            0,
            0,
            batch0.num_rows() as u64,
            n_obs as u64,
            &batch0,
        )
        .unwrap();
    let batch1 = dense_obsm_shard_batch(3, 3);
    // Stamp shard_idx = 0 on the second shard by re-using the
    // first shard's logical position in the metadata; section name
    // still carries `_shard_1` so it lands in the catalog.
    writer
        .write_obsm_shard(
            "X_pca",
            1, // section-name index — chosen so the catalog has _shard_1
            0, // row_start = 0 deliberately duplicates the first shard
            batch1.num_rows() as u64,
            n_obs as u64,
            &batch1,
        )
        .unwrap();
    writer.finish().unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let err = reader.read_obsm("X_pca").unwrap_err();
    assert!(
        matches!(err, ScxError::InvalidCatalog(_)),
        "expected InvalidCatalog, got {err:?}"
    );
}

/// A zero-row `obsm` shard must round-trip — `n_rows == 0` is the
/// edge case that disappeared from the disk-streaming path before
/// this fix landed. The writer-side override path and the
/// scx-convert disk-streaming branch both emit a single zero-row
/// shard; this test asserts the reader reassembles it as a
/// zero-row batch (not a `SectionNotFound`).
#[test]
fn test_sharded_read_zero_row_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zero_row.scx");
    let mut writer = build_obsm_test_writer(&path, 6, 4, 3);
    let empty = dense_obsm_shard_batch(0, 0);
    writer
        .write_obsm_shard("X_empty", 0, 0, 0, 0, &empty)
        .unwrap();
    writer.finish().unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let names = reader.list_obsm();
    assert_eq!(names, vec!["X_empty".to_string()]);
    let batch = reader.read_obsm("X_empty").unwrap();
    assert_eq!(batch.num_rows(), 0);
    assert_eq!(batch.num_columns(), 2);
}

/// The merged `RecordBatch` returned by `read_obsm` must NOT carry
/// per-shard metadata (`shard_idx`, `row_start`, `n_shard_rows`) —
/// those describe a single shard, not the reassembled matrix.
/// Stripping them prevents downstream consumers from being misled.
#[test]
fn test_sharded_read_strips_shard_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("strip_meta.scx");
    let n_obs = 6usize;
    let mut writer = build_obsm_test_writer(&path, n_obs, 4, 3);
    for shard_idx in 0u32..2 {
        let row_start = shard_idx as usize * 3;
        let batch = dense_obsm_shard_batch(row_start, 3);
        writer
            .write_obsm_shard(
                "X_pca",
                shard_idx,
                row_start as u64,
                batch.num_rows() as u64,
                n_obs as u64,
                &batch,
            )
            .unwrap();
    }
    writer.finish().unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let pca = reader.read_obsm("X_pca").unwrap();
    let md = pca.schema_ref().metadata();
    assert!(
        !md.contains_key("shard_idx"),
        "merged batch should not carry shard_idx (got metadata: {md:?})"
    );
    assert!(
        !md.contains_key("row_start"),
        "merged batch should not carry row_start (got metadata: {md:?})"
    );
    assert!(
        !md.contains_key("n_shard_rows"),
        "merged batch should not carry n_shard_rows (got metadata: {md:?})"
    );
    // n_rows_total describes the logical matrix and is preserved.
    assert_eq!(
        md.get("n_rows_total").map(String::as_str),
        Some("6"),
        "n_rows_total should survive (got metadata: {md:?})"
    );
}

/// Regression: when every shard carries the *same* categorical value,
/// Arrow's `concat` appends each shard's one-element dictionary, yielding
/// `["batch1", "batch1", "batch1", "batch1"]`. `to_pandas()` then raises
/// `ValueError: Categorical categories must be unique`. `assemble_sharded_metadata`
/// must collapse dictionary columns to a unified dictionary with distinct
/// values.
#[test]
fn test_assemble_unifies_duplicate_dictionary_categories() {
    use arrow::array::{Array, DictionaryArray};
    use arrow::datatypes::Int8Type;

    let dict_dt = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
    let n_shards = 4u32;

    let raw_batches: Vec<(u32, RecordBatch)> = (0..n_shards)
        .map(|shard_idx| {
            // One row per shard, all the same category — the duplicate
            // dictionary the writer's per-shard categoricals produce.
            let strs = StringArray::from(vec!["batch1"]);
            let dict = arrow::compute::cast(&(Arc::new(strs) as arrow::array::ArrayRef), &dict_dt)
                .unwrap();
            let metadata = std::collections::HashMap::from([
                ("shard_idx".to_string(), shard_idx.to_string()),
                ("row_start".to_string(), shard_idx.to_string()),
                ("n_shard_rows".to_string(), "1".to_string()),
                ("n_rows_total".to_string(), n_shards.to_string()),
            ]);
            let schema = Arc::new(
                Schema::new(vec![Field::new("gem_group", dict_dt.clone(), false)])
                    .with_metadata(metadata),
            );
            let batch = RecordBatch::try_new(schema, vec![dict]).unwrap();
            (shard_idx, batch)
        })
        .collect();

    let merged = assemble_sharded_metadata("obs", raw_batches).unwrap();
    assert_eq!(merged.num_rows(), n_shards as usize);

    let col = merged.column(0);
    // After unification the single surviving category fits an Int8 key —
    // `unify_dictionary_columns` narrows to the minimal key type.
    assert_eq!(
        col.data_type(),
        &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
        "1 distinct category should narrow to an Int8 key"
    );
    let dict = col
        .as_any()
        .downcast_ref::<DictionaryArray<Int8Type>>()
        .expect("gem_group should remain dictionary-encoded");
    let values = dict
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("dictionary values should be Utf8");
    assert_eq!(
        values.len(),
        1,
        "duplicate categories must be collapsed (got {:?})",
        (0..values.len())
            .map(|i| values.value(i))
            .collect::<Vec<_>>()
    );
    assert_eq!(values.value(0), "batch1");
    // Every row still resolves to the single surviving category.
    for i in 0..dict.len() {
        assert_eq!(values.value(dict.keys().value(i) as usize), "batch1");
    }
}

/// Regression: a categorical column whose per-shard vocabularies are
/// disjoint and sum to more than a narrow per-shard key can address.
/// Each shard here holds 50 unique categories encoded with an `Int8` key
/// (50 ≤ 127, the per-shard write is valid), but the union across 3 shards
/// is 150 distinct. Before the fix, `concat_batches` / the unify re-encode
/// overflowed the `Int8` key with `Dictionary key bigger than the key
/// type`; now the keys are widened to `Int32` before concat and narrowed
/// to the minimal fit (`Int16` for 150 distinct) after deduplication.
#[test]
fn test_assemble_high_cardinality_dictionary_widens_key() {
    use arrow::array::{Array, DictionaryArray};
    use arrow::datatypes::Int16Type;

    let per_shard = 50usize;
    let n_shards = 3u32;
    let total = per_shard * n_shards as usize;
    let narrow_dt = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));

    let raw_batches: Vec<(u32, RecordBatch)> = (0..n_shards)
        .map(|shard_idx| {
            let cats: Vec<String> = (0..per_shard)
                .map(|j| format!("s{shard_idx}_c{j}"))
                .collect();
            let strs = StringArray::from(cats.iter().map(|s| s.as_str()).collect::<Vec<_>>());
            // Per-shard categorical with a narrow Int8 key — valid locally
            // (50 ≤ 127) but disjoint across shards.
            let dict =
                arrow::compute::cast(&(Arc::new(strs) as arrow::array::ArrayRef), &narrow_dt)
                    .unwrap();
            let row_start = shard_idx as usize * per_shard;
            let metadata = std::collections::HashMap::from([
                ("shard_idx".to_string(), shard_idx.to_string()),
                ("row_start".to_string(), row_start.to_string()),
                ("n_shard_rows".to_string(), per_shard.to_string()),
                ("n_rows_total".to_string(), total.to_string()),
            ]);
            let schema = Arc::new(
                Schema::new(vec![Field::new("cell_type", narrow_dt.clone(), false)])
                    .with_metadata(metadata),
            );
            let batch = RecordBatch::try_new(schema, vec![dict]).unwrap();
            (shard_idx, batch)
        })
        .collect();

    // Pre-fix this returned `Err(Arrow("Dictionary key bigger than the key type"))`.
    let merged = assemble_sharded_metadata("obs", raw_batches).unwrap();
    assert_eq!(merged.num_rows(), total);

    let col = merged.column(0);
    assert_eq!(
        col.data_type(),
        &DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8)),
        "150 distinct categories should narrow to an Int16 key"
    );
    let dict = col
        .as_any()
        .downcast_ref::<DictionaryArray<Int16Type>>()
        .expect("cell_type should remain dictionary-encoded");
    let values = dict
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("dictionary values should be Utf8");
    assert_eq!(values.len(), total, "all 150 categories must survive");
    // Every row resolves to its original "s{shard}_c{j}" category.
    for shard in 0..n_shards as usize {
        for j in 0..per_shard {
            let row = shard * per_shard + j;
            let got = values.value(dict.keys().value(row) as usize);
            assert_eq!(got, format!("s{shard}_c{j}"));
        }
    }
}

/// Regression: an append writes obs categoricals as plain `Utf8` (its
/// `unify_dict_columns` decodes them) while `from_anndata` writes the same
/// column as a `Dictionary`. After an append, a sharded obs axis therefore
/// carries the column as `Dictionary` in the base shards and plain `Utf8` in
/// the appended shards. Before the fix, `concat_batches` rejected the mix with
/// *"It is not possible to concatenate arrays of different data types
/// (Dictionary(Int32, LargeUtf8), LargeUtf8)"* and the file's obs became
/// unreadable via `to_anndata()`. `reconcile_dictionary_representations` now
/// encodes the plain shard to a dictionary before concat.
#[test]
fn test_assemble_reconciles_mixed_dictionary_and_plain_shards() {
    use arrow::array::{Array, DictionaryArray};

    let dict_dt = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));

    // Shard 0: Dictionary-encoded (the `from_anndata` base layout).
    let s0_vals = StringArray::from(vec!["fibroblast", "epithelial"]);
    let s0_col =
        arrow::compute::cast(&(Arc::new(s0_vals) as arrow::array::ArrayRef), &dict_dt).unwrap();
    let s0_schema = Arc::new(
        Schema::new(vec![Field::new("cell_type", dict_dt.clone(), false)]).with_metadata(
            std::collections::HashMap::from([
                ("shard_idx".to_string(), "0".to_string()),
                ("row_start".to_string(), "0".to_string()),
                ("n_shard_rows".to_string(), "2".to_string()),
                ("n_rows_total".to_string(), "4".to_string()),
            ]),
        ),
    );
    let s0 = RecordBatch::try_new(s0_schema, vec![s0_col]).unwrap();

    // Shard 1: plain Utf8 (the appended layout), with a disjoint vocabulary.
    let s1_col = Arc::new(StringArray::from(vec!["neuron", "astrocyte"])) as arrow::array::ArrayRef;
    let s1_schema = Arc::new(
        Schema::new(vec![Field::new("cell_type", DataType::Utf8, false)]).with_metadata(
            std::collections::HashMap::from([
                ("shard_idx".to_string(), "1".to_string()),
                ("row_start".to_string(), "2".to_string()),
                ("n_shard_rows".to_string(), "2".to_string()),
                ("n_rows_total".to_string(), "4".to_string()),
            ]),
        ),
    );
    let s1 = RecordBatch::try_new(s1_schema, vec![s1_col]).unwrap();

    // Pre-fix: `Err(Arrow("...concatenate arrays of different data types..."))`.
    let merged = assemble_sharded_metadata("obs", vec![(0, s0), (1, s1)]).unwrap();
    assert_eq!(merged.num_rows(), 4);

    let col = merged.column(0);
    // 4 distinct categories survive, dictionary-encoded with a narrow key.
    assert_eq!(
        col.data_type(),
        &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
        "the reconciled column must remain dictionary-encoded"
    );
    let dict = col
        .as_any()
        .downcast_ref::<DictionaryArray<arrow::datatypes::Int8Type>>()
        .expect("cell_type should be dictionary-encoded after reconcile");
    let values = dict
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("dictionary values should be Utf8");
    let resolved: Vec<&str> = (0..dict.len())
        .map(|i| values.value(dict.keys().value(i) as usize))
        .collect();
    assert_eq!(
        resolved,
        vec!["fibroblast", "epithelial", "neuron", "astrocyte"],
        "every row must resolve to its original category across the mixed shards"
    );
}

/// Offset-overflow protection: the assembler upcasts to `LargeUtf8` before
/// concat (so a combined string column above `i32::MAX` can't overflow Arrow's
/// 32-bit offsets) and opportunistically narrows back afterwards. Drive a
/// shard set whose `cell_id` is already `LargeUtf8` on input and assert the
/// assembler concats without error and round-trips every value — a cheap proxy
/// for the >2 GB path that proves the consolidation did not drop the wide
/// handling the hand-rolled copies relied on.
#[test]
fn test_assemble_preserves_wide_offset_string_column() {
    use arrow::array::{Array, LargeStringArray};

    let n_shards = 3u32;
    let per_shard = 2usize;
    let total = per_shard * n_shards as usize;
    let raw_batches: Vec<(u32, RecordBatch)> = (0..n_shards)
        .map(|shard_idx| {
            let row_start = shard_idx as usize * per_shard;
            let ids: Vec<String> = (0..per_shard)
                .map(|j| format!("cell_{}", row_start + j))
                .collect();
            // Force the wide encoding on input.
            let col = LargeStringArray::from(ids.iter().map(|s| s.as_str()).collect::<Vec<_>>());
            let metadata = std::collections::HashMap::from([
                ("shard_idx".to_string(), shard_idx.to_string()),
                ("row_start".to_string(), row_start.to_string()),
                ("n_shard_rows".to_string(), per_shard.to_string()),
                ("n_rows_total".to_string(), total.to_string()),
            ]);
            let schema = Arc::new(
                Schema::new(vec![Field::new("cell_id", DataType::LargeUtf8, false)])
                    .with_metadata(metadata),
            );
            let batch = RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap();
            (shard_idx, batch)
        })
        .collect();

    let merged = assemble_sharded_metadata("obs", raw_batches).unwrap();
    assert_eq!(merged.num_rows(), total);
    // Small payload → opportunistically narrowed back to Utf8.
    assert_eq!(merged.column(0).data_type(), &DataType::Utf8);
    let arr = merged
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("cell_id narrows to Utf8 when offsets fit");
    let got: Vec<&str> = (0..arr.len()).map(|i| arr.value(i)).collect();
    assert_eq!(
        got,
        (0..total).map(|i| format!("cell_{i}")).collect::<Vec<_>>()
    );
}

#[test]
fn test_read_obs_schema_matches_full() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "schema.scx", 6, 10, 2, false);
    let reader = ScxReader::open(&path).unwrap();

    let obs_schema = reader.read_obs_schema().unwrap();
    let obs_batch = reader.read_obs().unwrap();
    assert_eq!(&obs_schema, obs_batch.schema().as_ref());

    let var_schema = reader.read_var_schema().unwrap();
    let var_batch = reader.read_var().unwrap();
    assert_eq!(&var_schema, var_batch.schema().as_ref());
}

// -----------------------------------------------------------------------
// 11.16: Checksum corruption detection
// -----------------------------------------------------------------------

#[test]
fn test_validate_detects_corruption() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "corrupt.scx", 6, 10, 2, false);

    // Read the file, corrupt a byte in a shard section, write back
    let mut data = std::fs::read(&path).unwrap();

    // Find a CSR shard section offset from the catalog
    let reader = ScxReader::open(&path).unwrap();
    let shards = reader.catalog().shards_sorted();
    assert!(!shards.is_empty());
    let shard_offset = shards[0].offset as usize;
    // Corrupt a byte in the shard payload (after the 76-byte header)
    let corrupt_pos = shard_offset + SHARD_HEADER_SIZE + 1;
    drop(reader);

    data[corrupt_pos] ^= 0xFF;
    std::fs::write(&path, &data).unwrap();

    // Re-open — open should still succeed (catalog checksum is intact)
    let reader = ScxReader::open(&path).unwrap();

    // validate() should detect the corruption and error (essential section)
    let result = reader.validate();
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ScxError::ChecksumMismatch { .. }
    ));
}

/// Regression guard for Phase 2A: verified path catches corruption,
/// unchecked path (default) does not error on corrupted shard payload.
#[test]
fn test_verified_vs_unchecked_shard_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "verify_guard.scx", 6, 10, 2, false);

    // Read the file, corrupt a byte in a shard payload, write back
    let mut data = std::fs::read(&path).unwrap();
    let reader = ScxReader::open(&path).unwrap();
    let shards = reader.catalog().shards_sorted();
    assert!(!shards.is_empty());
    let shard_entry = shards[0].clone();
    let shard_offset = shard_entry.offset as usize;
    let corrupt_pos = shard_offset + SHARD_HEADER_SIZE + 1;
    drop(reader);

    data[corrupt_pos] ^= 0xFF;
    std::fs::write(&path, &data).unwrap();

    // Re-open (catalog checksum is still intact since we only corrupted
    // shard payload bytes, not the catalog region)
    let reader = ScxReader::open(&path).unwrap();

    // Verified path should detect the corruption
    let verified_result = reader.read_shard_from_entry_verified(&shard_entry);
    assert!(verified_result.is_err());
    assert!(matches!(
        verified_result.unwrap_err(),
        ScxError::ChecksumMismatch { .. }
    ));

    // Unchecked path (default) should not error — it skips the checksum
    let unchecked_result = reader.read_shard_from_entry(&shard_entry);
    assert!(unchecked_result.is_ok());
}

// -----------------------------------------------------------------------
// 11.17: Unknown section types handled gracefully
// -----------------------------------------------------------------------

#[test]
fn test_known_sections_read_correctly() {
    // Validates that when all sections are known types, the reader works.
    // The catalog modification to skip unknown types is tested by the
    // fact that FullCatalog::read_from no longer panics on unknown types.
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "known.scx", 4, 8, 1, true);

    let reader = ScxReader::open(&path).unwrap();
    let catalog = reader.catalog();

    // All entries should have known section types
    for entry in &catalog.entries {
        assert!(SectionType::from_u8(entry.section_type as u8).is_some());
    }

    // All section reads should succeed
    assert!(reader.read_obs().is_ok());
    assert!(reader.read_var().is_ok());
    assert!(reader.read_csr_shard(0).is_ok());
    assert!(reader.read_uns().is_ok());
    assert!(reader.read_provenance().is_ok());
}

// -----------------------------------------------------------------------
// 11.18: Multi-shard assembly matches individual reads
// -----------------------------------------------------------------------

#[test]
fn test_four_shards_individual_vs_assembled() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "four.scx", 12, 10, 4, false);

    let reader = ScxReader::open(&path).unwrap();

    // Read individually
    let mut individual_indptr: Vec<i64> = Vec::new();
    let mut individual_indices: Vec<i32> = Vec::new();
    let mut individual_data: Vec<f32> = Vec::new();
    let mut cumulative_nnz: i64 = 0;

    for i in 0..4 {
        let (indptr, indices, data) = reader.read_csr_shard(i).unwrap();
        if i == 0 {
            individual_indptr.extend_from_slice(&indptr);
        } else {
            for &v in &indptr[1..] {
                individual_indptr.push(v + cumulative_nnz);
            }
        }
        cumulative_nnz += *indptr.last().unwrap_or(&0);
        individual_indices.extend_from_slice(&indices);
        individual_data.extend_from_slice(&data);
    }

    // Read assembled
    let csr = reader.read_all_csr_shards().unwrap();

    assert_eq!(csr.indptr, individual_indptr);
    assert_eq!(csr.indices, individual_indices);
    assert_eq!(csr.data, individual_data);
    assert_eq!(csr.shape, (12, 10));
}

// -----------------------------------------------------------------------
// 2B.7: Single-allocation assembly matches individual shard merge
// -----------------------------------------------------------------------

#[test]
fn test_single_alloc_assembly_1_shard() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "one_shard.scx", 8, 10, 1, false);
    let reader = ScxReader::open(&path).unwrap();

    let csr = reader.read_all_csr_shards().unwrap();
    // 1 shard: indptr should start at 0 and be monotonic
    assert_eq!(csr.indptr[0], 0);
    assert_eq!(csr.shape, (8, 10));
    for w in csr.indptr.windows(2) {
        assert!(w[1] >= w[0], "indptr not monotonic: {} > {}", w[0], w[1]);
    }
    for &idx in &csr.indices {
        assert!((0..10).contains(&idx), "index {} out of bounds", idx);
    }
}

#[test]
fn test_single_alloc_assembly_2_shards() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "two_shards.scx", 8, 10, 2, false);
    let reader = ScxReader::open(&path).unwrap();

    // Read individually (old merge pattern)
    let mut individual_indptr: Vec<i64> = Vec::new();
    let mut individual_indices: Vec<i32> = Vec::new();
    let mut individual_data: Vec<f32> = Vec::new();
    let mut cumulative_nnz: i64 = 0;

    for i in 0..2 {
        let (indptr, indices, data) = reader.read_csr_shard(i).unwrap();
        if i == 0 {
            individual_indptr.extend_from_slice(&indptr);
        } else {
            for &v in &indptr[1..] {
                individual_indptr.push(v + cumulative_nnz);
            }
        }
        cumulative_nnz += *indptr.last().unwrap_or(&0);
        individual_indices.extend_from_slice(&indices);
        individual_data.extend_from_slice(&data);
    }

    // Read assembled (single-allocation path)
    let csr = reader.read_all_csr_shards().unwrap();

    assert_eq!(csr.indptr, individual_indptr);
    assert_eq!(csr.indices, individual_indices);
    assert_eq!(csr.data, individual_data);
    assert_eq!(csr.shape, (8, 10));
}

#[test]
fn test_single_alloc_assembly_many_shards() {
    let dir = tempfile::tempdir().unwrap();
    // 100 rows, 20 vars, 10 shards = 10 rows per shard
    let path = write_test_file(&dir, "many_shards.scx", 100, 20, 10, false);
    let reader = ScxReader::open(&path).unwrap();

    // Read individually (old merge pattern)
    let mut individual_indptr: Vec<i64> = Vec::new();
    let mut individual_indices: Vec<i32> = Vec::new();
    let mut individual_data: Vec<f32> = Vec::new();
    let mut cumulative_nnz: i64 = 0;

    for i in 0..10 {
        let (indptr, indices, data) = reader.read_csr_shard(i).unwrap();
        if i == 0 {
            individual_indptr.extend_from_slice(&indptr);
        } else {
            for &v in &indptr[1..] {
                individual_indptr.push(v + cumulative_nnz);
            }
        }
        cumulative_nnz += *indptr.last().unwrap_or(&0);
        individual_indices.extend_from_slice(&indices);
        individual_data.extend_from_slice(&data);
    }

    // Read assembled (single-allocation path)
    let csr = reader.read_all_csr_shards().unwrap();

    assert_eq!(csr.indptr, individual_indptr);
    assert_eq!(csr.indices, individual_indices);
    assert_eq!(csr.data, individual_data);
    assert_eq!(csr.shape, (100, 20));

    // Verify invariants
    for w in csr.indptr.windows(2) {
        assert!(w[1] >= w[0], "indptr not monotonic");
    }
    for &idx in &csr.indices {
        assert!((0..20).contains(&idx), "index {} out of bounds", idx);
    }
}

// -----------------------------------------------------------------------
// 11.19: Section byte ranges match catalog entries
// -----------------------------------------------------------------------

#[test]
fn test_section_byte_ranges_match_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "ranges.scx", 6, 10, 2, true);

    let reader = ScxReader::open(&path).unwrap();
    let catalog = reader.catalog();

    // Verify obs and var sections can be sliced at their catalog offsets
    let obs_entry = catalog.get("obs").unwrap();
    let obs_bytes = reader.section_bytes(obs_entry).unwrap();
    assert_eq!(obs_bytes.len(), obs_entry.length as usize);

    let var_entry = catalog.get("var").unwrap();
    let var_bytes = reader.section_bytes(var_entry).unwrap();
    assert_eq!(var_bytes.len(), var_entry.length as usize);

    // Verify checksum of raw bytes matches catalog checksum
    assert_eq!(blake3_hash(obs_bytes), obs_entry.checksum);
    assert_eq!(blake3_hash(var_bytes), var_entry.checksum);

    // All entries should be 8-byte aligned
    for entry in &catalog.entries {
        assert_eq!(
            entry.offset % 8,
            0,
            "section '{}' not 8-byte aligned",
            entry.name
        );
    }
}

// -----------------------------------------------------------------------
// Additional edge case tests
// -----------------------------------------------------------------------

#[test]
fn test_shard_index_out_of_bounds() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "bounds.scx", 4, 8, 1, false);

    let reader = ScxReader::open(&path).unwrap();
    let result = reader.read_csr_shard(5);
    assert!(matches!(
        result.unwrap_err(),
        ScxError::ShardIndexOutOfBounds { index: 5, count: 1 }
    ));
}

#[test]
fn test_section_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "nofind.scx", 4, 8, 1, false);

    let reader = ScxReader::open(&path).unwrap();
    assert!(matches!(
        reader.read_uns().unwrap_err(),
        ScxError::SectionNotFound(_)
    ));
    assert!(matches!(
        reader.read_obsm("X_pca").unwrap_err(),
        ScxError::SectionNotFound(_)
    ));
}

#[test]
fn test_layer_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("layers.scx");
    let header = sample_header(6, 10, 12);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    writer.write_obs(&sample_obs(6)).unwrap();
    writer.write_var(&sample_var(10)).unwrap();

    // Write X shards
    let (indptr, indices, values) = sample_shard_data(6, 10);
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

    // Write "raw" layer shard
    let (indptr, indices, values) = sample_shard_data(6, 10);
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

    writer.finish().unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let names = reader.layer_names();
    assert_eq!(names, vec!["raw"]);

    let layer = reader.read_layer("raw").unwrap();
    assert_eq!(layer.shape, (6, 10));
    assert_eq!(layer.nnz(), 12);
}

#[test]
fn test_validate_passes_for_clean_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "clean.scx", 6, 10, 2, true);

    let reader = ScxReader::open(&path).unwrap();
    let results = reader.validate().unwrap();

    // All sections should pass
    for (name, passed) in &results {
        assert!(passed, "section '{}' failed checksum", name);
    }
}

/// Phase J.2: validate() re-checks BLAKE3 for CSC shards via the
/// generic catalog walk. Build a CSR + CSC test file, verify all
/// CSC sections appear in the results and pass.
#[test]
fn test_validate_csc_shards_pass_on_clean_file() {
    use crate::section::SectionType;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("csc_clean.scx");

    // Build a small CSR + CSC file directly.
    let n_obs = 8usize;
    let n_vars = 6usize;
    let header = sample_header(n_obs as u64, n_vars as u64, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    // CSR (single shard)
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

    // CSC: two shards of 3 cols each.
    for chunk_start in (0..n_vars).step_by(3) {
        let chunk_end = (chunk_start + 3).min(n_vars);
        let mut ip = vec![0u64];
        let mut ix: Vec<u32> = Vec::new();
        let mut vb: Vec<u8> = Vec::new();
        for c in chunk_start..chunk_end {
            ix.push((c % n_obs) as u32);
            vb.push((c as u8) + 1);
            ip.push(ix.len() as u64);
        }
        writer
            .write_csc_shard(
                &ip,
                &ix,
                &vb,
                CodecId::None,
                ValueEncoding::Uint8,
                chunk_start as u64,
            )
            .unwrap();
    }
    writer.finish().unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let results = reader.validate().unwrap();

    // Confirm the CSC shards are in the results AND all pass.
    let csc_results: Vec<_> = results
        .iter()
        .filter(|(name, _)| name.starts_with("X_csc_shard_"))
        .collect();
    assert_eq!(csc_results.len(), 2, "expected 2 CSC shards in validate()");
    for (name, passed) in &csc_results {
        assert!(*passed, "CSC section '{}' failed checksum", name);
    }

    // Also confirm `validate()` walks every catalog entry: total
    // results count >= number of catalog entries with CscShard
    // type, so no CSC entry was silently skipped.
    let n_csc_entries = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CscShard)
        .count();
    let n_csc_in_results = results
        .iter()
        .filter(|(name, _)| name.starts_with("X_csc_shard_"))
        .count();
    assert_eq!(n_csc_entries, n_csc_in_results);
}

/// Phase J.2: validate() flags corruption inside a CSC shard.
/// CSC is not in the "essential" set (corrupting only CSC
/// shouldn't fail the full file), so we expect Ok with a
/// `passed=false` row for the corrupted CSC section.
#[test]
fn test_validate_detects_csc_corruption() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("csc_corrupt.scx");

    let n_obs = 6usize;
    let n_vars = 4usize;
    let header = sample_header(n_obs as u64, n_vars as u64, 0);
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
    // One CSC shard.
    let ip = vec![0u64, 1, 2, 3, 4];
    let ix: Vec<u32> = vec![0, 1, 2, 3];
    let vb: Vec<u8> = vec![10, 20, 30, 40];
    writer
        .write_csc_shard(&ip, &ix, &vb, CodecId::None, ValueEncoding::Uint8, 0)
        .unwrap();
    writer.finish().unwrap();

    // Find the CSC shard offset in the catalog.
    let reader = ScxReader::open(&path).unwrap();
    let csc_entry = reader
        .catalog()
        .entries
        .iter()
        .find(|e| e.section_type == crate::section::SectionType::CscShard)
        .unwrap()
        .clone();
    let csc_offset = csc_entry.offset as usize;
    drop(reader);

    // Flip a byte in the CSC payload (after the 76-byte header).
    let mut data = std::fs::read(&path).unwrap();
    let corrupt_pos = csc_offset + SHARD_HEADER_SIZE + 1;
    data[corrupt_pos] ^= 0xFF;
    std::fs::write(&path, &data).unwrap();

    // validate() should NOT return Err (CSC is not "essential"),
    // but should report the CSC section as failed.
    let reader = ScxReader::open(&path).unwrap();
    let results = reader.validate().expect(
        "validate() should not error on CSC corruption since CSC is not \
             in the essential-section set",
    );
    let csc_passed: bool = results
        .iter()
        .find(|(name, _)| name == &csc_entry.name)
        .map(|(_, p)| *p)
        .expect("CSC entry should appear in validate() results");
    assert!(!csc_passed, "validate() should flag corrupted CSC section");
}

// -----------------------------------------------------------------------
// 16.8: Multi-operation provenance chain
// -----------------------------------------------------------------------

#[test]
fn test_multi_operation_provenance_chain() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prov_chain.scx");
    let header = sample_header(4, 8, 8);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    writer.write_obs(&sample_obs(4)).unwrap();
    writer.write_var(&sample_var(8)).unwrap();

    let (indptr, indices, values) = sample_shard_data(4, 8);
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

    // Write provenance with 3 chained operations
    let entries = vec![
        ProvenanceEntry {
            timestamp: 1710000000,
            action: "convert".to_string(),
            tool: "scx-cli 0.1.0".to_string(),
            params_json: r#"{"input":"raw.h5ad"}"#.to_string(),
            input_checksums: vec![[0xAA; 32]],
        },
        ProvenanceEntry {
            timestamp: 1710001000,
            action: "subset".to_string(),
            tool: "pyscx 0.1.0".to_string(),
            params_json: r#"{"n_cells":1000}"#.to_string(),
            input_checksums: vec![[0xBB; 32]],
        },
        ProvenanceEntry {
            timestamp: 1710002000,
            action: "normalize".to_string(),
            tool: "pyscx 0.1.0".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![[0xCC; 32], [0xDD; 32]],
        },
    ];

    writer.write_provenance(entries.clone()).unwrap();
    writer.finish().unwrap();

    // Read back and verify
    let reader = ScxReader::open(&path).unwrap();
    let prov = reader.read_provenance().unwrap();

    assert_eq!(prov.version, 1);
    assert_eq!(prov.operations.len(), 3);

    // Verify ordering and content
    assert_eq!(prov.operations[0].action, "convert");
    assert_eq!(prov.operations[0].timestamp, 1710000000);
    assert_eq!(prov.operations[0].input_checksums.len(), 1);

    assert_eq!(prov.operations[1].action, "subset");
    assert_eq!(prov.operations[1].timestamp, 1710001000);
    assert_eq!(prov.operations[1].params_json, r#"{"n_cells":1000}"#);

    assert_eq!(prov.operations[2].action, "normalize");
    assert_eq!(prov.operations[2].timestamp, 1710002000);
    assert_eq!(prov.operations[2].input_checksums.len(), 2);
    assert_eq!(prov.operations[2].input_checksums[0], [0xCC; 32]);
    assert_eq!(prov.operations[2].input_checksums[1], [0xDD; 32]);
}

// -----------------------------------------------------------------------
// Parallel shard decode tests
// -----------------------------------------------------------------------

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_matches_sequential() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "par_seq.scx", 12, 10, 4, false);

    let reader = ScxReader::open(&path).unwrap();
    let shards = reader.catalog().shards_sorted();

    let sequential = reader.assemble_shards(&shards).unwrap();
    let parallel = reader.assemble_shards_parallel(&shards).unwrap();

    assert_eq!(sequential.shape, parallel.shape);
    assert_eq!(sequential.indptr, parallel.indptr);
    assert_eq!(sequential.indices, parallel.indices);
    assert_eq!(sequential.data, parallel.data);
}

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_single_shard() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "par_single.scx", 6, 10, 1, false);

    let reader = ScxReader::open(&path).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();

    assert_eq!(csr.shape, (6, 10));
    assert_eq!(csr.nnz(), 12);
    assert_eq!(csr.indptr.len(), 7);
}

#[test]
#[cfg(feature = "parallel")]
fn test_parallel_thread_pool() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "par_pool.scx", 12, 10, 4, false);

    let reader = ScxReader::open(&path).unwrap();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();

    let csr = pool.install(|| reader.read_all_csr_shards()).unwrap();

    assert_eq!(csr.shape, (12, 10));
    assert_eq!(csr.nnz(), 24);
    assert_eq!(csr.indptr.len(), 13);
}

#[test]
fn test_values_to_f32_uint8() {
    let raw = vec![1u8, 2, 255];
    let result = values_to_f32(&raw, ValueEncoding::Uint8);
    assert_eq!(result, vec![1.0, 2.0, 255.0]);
}

#[test]
fn test_values_to_f32_uint16() {
    let raw: Vec<u8> = vec![0x01, 0x00, 0xFF, 0x00]; // 1, 255 as u16 LE
    let result = values_to_f32(&raw, ValueEncoding::Uint16);
    assert_eq!(result, vec![1.0, 255.0]);
}

#[test]
fn test_values_to_f32_float32() {
    let val: f32 = 1.23456;
    let raw = val.to_le_bytes().to_vec();
    let result = values_to_f32(&raw, ValueEncoding::Float32);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 1.23456).abs() < 1e-6);
}

// -----------------------------------------------------------------------
// Shared-catalog open (open_with_shared_catalog)
// -----------------------------------------------------------------------

/// `open_with_shared_catalog` must produce a reader whose
/// metadata (header, root catalog, full catalog, shard reads) is
/// indistinguishable from a fresh `open()` against the same file.
/// This is the load-bearing correctness check for the
/// `to_anndata_backed` catalog-sharing path.
#[test]
fn test_open_with_shared_catalog_matches_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "shared_catalog.scx", 8, 12, 2, false);

    let primary = ScxReader::open(&path).unwrap();
    let primary_catalog = primary.catalog_arc();

    let shared = ScxReader::open_with_shared_catalog(&path, primary_catalog).unwrap();

    // Header / root catalog must be identical (parsed fresh from
    // the secondary mmap, but the file is the same).
    assert_eq!(shared.header().n_obs, primary.header().n_obs);
    assert_eq!(shared.header().n_vars, primary.header().n_vars);
    assert_eq!(
        shared.header().full_catalog_offset,
        primary.header().full_catalog_offset
    );

    // Catalog entries must match field-for-field — the shared
    // path didn't re-parse, so this verifies the Arc shared
    // through.
    let primary_entries = &primary.catalog().entries;
    let shared_entries = &shared.catalog().entries;
    assert_eq!(primary_entries.len(), shared_entries.len());
    for (a, b) in primary_entries.iter().zip(shared_entries.iter()) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.offset, b.offset);
        assert_eq!(a.length, b.length);
        assert_eq!(a.section_type, b.section_type);
        assert_eq!(a.modality_id, b.modality_id);
        assert_eq!(a.checksum, b.checksum);
    }

    // Shard reads against the shared reader must produce the same
    // bytes as against the primary — confirms the mmap path is
    // independent and the cached catalog still drives correct
    // section addressing.
    let csr_shards = primary.catalog().shards_sorted();
    for entry in &csr_shards {
        let (ip_a, ix_a, dv_a) = primary.read_shard_from_entry(entry).unwrap();
        let (ip_b, ix_b, dv_b) = shared.read_shard_from_entry(entry).unwrap();
        assert_eq!(ip_a, ip_b);
        assert_eq!(ix_a, ix_b);
        assert_eq!(dv_a, dv_b);
    }
}

/// Smoke test: many readers can share a single `Arc<FullCatalog>`
/// without contention. Mirrors the `to_anndata_backed` shape (one
/// primary reader + several secondary readers sharing its catalog).
#[test]
fn test_shared_catalog_n_plus_3_pattern() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "n_plus_3.scx", 6, 8, 2, false);

    let primary = ScxReader::open(&path).unwrap();
    let shared = primary.catalog_arc();

    // Strong refcount before the secondaries: 1 (held by primary).
    assert_eq!(Arc::strong_count(&shared), 2); // primary + this binding

    let secondaries: Vec<ScxReader> = (0..5)
        .map(|_| ScxReader::open_with_shared_catalog(&path, Arc::clone(&shared)).unwrap())
        .collect();

    // Each secondary holds a refcount; primary + binding + 5 = 7.
    assert_eq!(Arc::strong_count(&shared), 7);

    // All secondaries see the same catalog content.
    for s in &secondaries {
        assert_eq!(s.catalog().entries.len(), primary.catalog().entries.len());
    }

    // Dropping a secondary decrements the refcount.
    drop(secondaries);
    assert_eq!(Arc::strong_count(&shared), 2);
}

/// `open_with_shared_catalog` must reject a catalog whose
/// `manifest_sequence` disagrees with the freshly-read file
/// header. This is the guardrail against silently combining a
/// stale catalog with a mutated file (append / compact / rollback
/// in `scx-ops` bumps the sequence on every mutation).
#[test]
fn test_open_with_shared_catalog_rejects_manifest_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "manifest_mismatch.scx", 6, 8, 2, false);

    let primary = ScxReader::open(&path).unwrap();
    let mut tweaked = (*primary.catalog_arc()).clone();
    tweaked.manifest_sequence = primary.header().manifest_sequence.wrapping_add(1);

    match ScxReader::open_with_shared_catalog(&path, Arc::new(tweaked)) {
        Err(ScxError::InvalidCatalog(msg)) => {
            assert!(
                msg.contains("manifest_sequence"),
                "error must name the mismatched field, got: {msg}",
            );
        }
        Err(other) => panic!("expected InvalidCatalog, got {other:?}"),
        Ok(_) => panic!("manifest_sequence mismatch must surface as an error"),
    }
}
