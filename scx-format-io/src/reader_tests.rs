use super::*;
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
    // Unframed writes stamp the default (v3), not the max-readable CURRENT (v4).
    assert_eq!(
        reader.header().format_version,
        crate::header::DEFAULT_WRITE_FORMAT_VERSION
    );

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

/// Regression: a boolean-valued pandas `Categorical`
/// (`pd.Categorical([True, False])` → `Dictionary(_, Boolean)`) made a
/// **row-sharded** file's `read_obs()` fail outright with *"Unsupported
/// output type for dictionary packing: Boolean"* — arrow cannot re-encode a
/// `Boolean` array into a dictionary, and both the key-widening and the
/// dedup step went through decode→re-encode.
///
/// The write side never complained and the same column in an *unsharded*
/// file read back fine, so this only appeared at the scale where obs is
/// sharded — the regime the sharded layout exists for.
#[test]
fn test_assemble_handles_a_boolean_valued_categorical() {
    use arrow::array::{Array, AsArray, BooleanArray, DictionaryArray, Int8Array};
    use arrow::datatypes::Int8Type;

    let dict_dt = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Boolean));
    // Two shards, each with its own two-entry [true, false] dictionary —
    // which is exactly the duplicate-categories shape the dedup step exists
    // to collapse, so this exercises both halves.
    let raw_batches: Vec<(u32, RecordBatch)> = (0..2u32)
        .map(|shard_idx| {
            let keys = Int8Array::from(vec![Some(0), Some(1), None]);
            let values: arrow::array::ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
            let dict: arrow::array::ArrayRef =
                Arc::new(DictionaryArray::<Int8Type>::try_new(keys, values).unwrap());
            let metadata = std::collections::HashMap::from([
                ("shard_idx".to_string(), shard_idx.to_string()),
                ("row_start".to_string(), (shard_idx * 3).to_string()),
                ("n_shard_rows".to_string(), "3".to_string()),
                ("n_rows_total".to_string(), "6".to_string()),
            ]);
            let schema = Arc::new(
                Schema::new(vec![Field::new("flag", dict_dt.clone(), true)])
                    .with_metadata(metadata),
            );
            (shard_idx, RecordBatch::try_new(schema, vec![dict]).unwrap())
        })
        .collect();

    let merged = assemble_sharded_metadata("obs", raw_batches).unwrap();
    assert_eq!(merged.num_rows(), 6);

    // Decode whatever representation survived (dictionary or plain) and
    // assert the values, so the test pins the data rather than the encoding.
    let col = merged.column(0);
    let plain = arrow::compute::cast(col, &DataType::Boolean).unwrap();
    let plain = plain.as_any().downcast_ref::<BooleanArray>().unwrap();
    let got: Vec<Option<bool>> = (0..plain.len())
        .map(|i| (!plain.is_null(i)).then(|| plain.value(i)))
        .collect();
    assert_eq!(
        got,
        vec![Some(true), Some(false), None, Some(true), Some(false), None]
    );

    // If it stayed a dictionary, its categories must be unique — otherwise
    // `to_pandas()` raises "Categorical categories must be unique".
    if let DataType::Dictionary(_, _) = col.data_type() {
        let dict = col.as_any_dictionary_opt().unwrap();
        let values = dict
            .values()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert!(
            values.len() <= 2,
            "duplicate boolean categories survived: {} entries",
            values.len()
        );
    }
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

/// Part 3 (memory-bounded dict unify): assembling many shards that all share the
/// same small category pool — with distinct per-shard local vocab orders and
/// null keys — must deduplicate to the unique set (not pool×n_shards), narrow
/// the key to the minimal fit, and decode every row (incl. nulls) correctly.
/// This exercises the values-remap dedup path that replaced the full-column
/// Utf8 round-trip.
#[test]
fn test_assemble_dictionary_dedup_many_shards() {
    use arrow::array::{Array, DictionaryArray};
    use arrow::datatypes::Int8Type;

    let pool = ["alpha", "beta", "gamma"];
    let per_shard = 4usize;
    let n_shards = 50u32;
    let total = per_shard * n_shards as usize;
    let dict_dt = DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));

    let mut expected: Vec<Option<String>> = Vec::with_capacity(total);
    let raw_batches: Vec<(u32, RecordBatch)> = (0..n_shards)
        .map(|shard_idx| {
            // Rotate the assignment per shard so each shard's *local* dictionary
            // has a different vocab order; null every 7th shard's row 1.
            let vals: Vec<Option<&str>> = (0..per_shard)
                .map(|r| {
                    if shard_idx % 7 == 0 && r == 1 {
                        None
                    } else {
                        Some(pool[(shard_idx as usize + r) % pool.len()])
                    }
                })
                .collect();
            for v in &vals {
                expected.push(v.map(|s| s.to_string()));
            }
            let strs = StringArray::from(vals);
            let dict = arrow::compute::cast(&(Arc::new(strs) as arrow::array::ArrayRef), &dict_dt)
                .unwrap();
            let row_start = shard_idx as usize * per_shard;
            let metadata = std::collections::HashMap::from([
                ("shard_idx".to_string(), shard_idx.to_string()),
                ("row_start".to_string(), row_start.to_string()),
                ("n_shard_rows".to_string(), per_shard.to_string()),
                ("n_rows_total".to_string(), total.to_string()),
            ]);
            let schema = Arc::new(
                Schema::new(vec![Field::new("cell_type", dict_dt.clone(), true)])
                    .with_metadata(metadata),
            );
            let batch = RecordBatch::try_new(schema, vec![dict]).unwrap();
            (shard_idx, batch)
        })
        .collect();

    let merged = assemble_sharded_metadata("obs", raw_batches).unwrap();
    assert_eq!(merged.num_rows(), total);

    let col = merged.column(0);
    assert_eq!(
        col.data_type(),
        &DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
        "3 distinct categories across 50 shards must dedup to an Int8 key"
    );
    let dict = col
        .as_any()
        .downcast_ref::<DictionaryArray<Int8Type>>()
        .expect("cell_type should remain dictionary-encoded");
    let values = dict
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("dictionary values should be Utf8");
    assert_eq!(
        values.len(),
        pool.len(),
        "duplicate categories across shards must collapse to the unique set"
    );
    let got_set: std::collections::HashSet<&str> =
        (0..values.len()).map(|i| values.value(i)).collect();
    let want_set: std::collections::HashSet<&str> = pool.iter().copied().collect();
    assert_eq!(
        got_set, want_set,
        "unified vocab must equal the category pool"
    );
    for (row, exp) in expected.iter().enumerate() {
        if dict.is_null(row) {
            assert_eq!(*exp, None, "row {row} should decode to null");
        } else {
            let got = values.value(dict.keys().value(row) as usize);
            assert_eq!(Some(got.to_string()), *exp, "row {row} mismatch");
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

/// Regression guard for Phase 2A: the verified path checks the shard checksum,
/// the unchecked path (default) does not.
///
/// The tamper here is the **stored checksum itself**, not the payload. That is
/// deliberate, and it is what isolates the property under test: with the payload
/// untouched, the shard still decodes to a structurally valid CSR, so the only
/// thing that can distinguish the two paths is whether they compare the
/// checksum. This test used to flip a payload byte and assert the unchecked read
/// returned `Ok` — which stopped being true once the codec gained a post-decode
/// CSR shape gate, because the flipped byte was in the *indptr* sub-stream and
/// the decoded indptr no longer ended at the declared `nnz`. That was the shape
/// gate working, but it left this test asserting the wrong thing: it would have
/// passed just as well if the unchecked path had started verifying checksums.
/// The payload-corruption case is covered below by its own test.
#[test]
fn test_verified_vs_unchecked_shard_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "verify_guard.scx", 6, 10, 2, false);

    let mut data = std::fs::read(&path).unwrap();
    let reader = ScxReader::open(&path).unwrap();
    let shards = reader.catalog().shards_sorted();
    assert!(!shards.is_empty());
    let shard_entry = shards[0].clone();
    let shard_offset = shard_entry.offset as usize;
    drop(reader);

    // The checksum is the last field of the shard header, and it covers only
    // the payload that follows — so corrupting it leaves every decode input
    // byte-identical.
    let checksum_pos = shard_offset + SHARD_HEADER_SIZE - 8;
    data[checksum_pos] ^= 0xFF;
    std::fs::write(&path, &data).unwrap();

    let reader = ScxReader::open(&path).unwrap();

    let verified_result = reader.read_shard_from_entry_verified(&shard_entry);
    assert!(verified_result.is_err());
    assert!(matches!(
        verified_result.unwrap_err(),
        ScxError::ChecksumMismatch { .. }
    ));

    // Unchecked path skips the comparison, so identical bytes read fine.
    let unchecked_result = reader.read_shard_from_entry(&shard_entry);
    assert!(
        unchecked_result.is_ok(),
        "unchecked read failed on an intact payload: {:?}",
        unchecked_result.err()
    );
}

/// Corruption inside the indptr sub-stream is caught even with checksums off.
///
/// The unchecked read path is the default one, and it is what the ML loader,
/// the query engine and the GPU host bounce all use. Before the codec enforced
/// the CSR shape after decode, a flipped indptr byte produced a structurally
/// invalid CSR that was returned as `Ok` — `indptr.last()` no longer equalled
/// the declared `nnz`, and `ScxCsr::new_unchecked` only `debug_assert`s that,
/// so a release build would index past the end of `indices` downstream.
#[test]
fn test_unchecked_read_still_rejects_a_structurally_broken_shard() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "shape_guard.scx", 6, 10, 2, false);

    let mut data = std::fs::read(&path).unwrap();
    let reader = ScxReader::open(&path).unwrap();
    let shard_entry = reader.catalog().shards_sorted()[0].clone();
    let shard_offset = shard_entry.offset as usize;
    drop(reader);

    // Second byte of the indptr sub-stream, which starts right after the header.
    data[shard_offset + SHARD_HEADER_SIZE + 1] ^= 0xFF;
    std::fs::write(&path, &data).unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let err = reader
        .read_shard_from_entry(&shard_entry)
        .expect_err("a corrupt indptr must not decode to an Ok CSR");
    assert!(
        !matches!(err, ScxError::ChecksumMismatch { .. }),
        "the unchecked path must not be verifying checksums; got {err:?}"
    );
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

/// Part 1 of the `scx sort` OOM fix: `read_obs_keys` must return the
/// requested obs column(s) byte-identical to projecting a full `read_obs()`,
/// while only decoding those columns from each shard. Exercises the sharded
/// path (cross-shard dictionary unify, where each shard carries a *local*
/// vocabulary) plus a numeric column, and verifies an unknown column errors
/// cleanly.
#[test]
fn test_read_obs_keys_matches_full_read_obs() {
    use arrow::array::{Array, DictionaryArray, Int64Array};
    use arrow::datatypes::Int8Type;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sharded_obs_keys.scx");

    let n_obs: usize = 9;
    let n_vars: usize = 4;
    let header = FileHeader::new_single_modality(n_obs as u64, n_vars as u64, 0, 3, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();

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
    writer.write_var(&sample_var(n_vars)).unwrap();

    // 3 obs shards, each with a LOCAL dictionary vocabulary so the assembler
    // must unify across shards. cell_type sequence: A B A | B C A | C C B.
    let cell_types = [["A", "B", "A"], ["B", "C", "A"], ["C", "C", "B"]];
    let obs_schema = Arc::new(Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new(
            "cell_type",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("n_genes", DataType::Int64, false),
    ]));
    for (shard_idx, types) in cell_types.iter().enumerate() {
        let row_start = shard_idx * 3;
        let ids: Vec<String> = (row_start..row_start + 3)
            .map(|i| format!("cell_{i}"))
            .collect();
        let dict: DictionaryArray<Int8Type> = types.iter().copied().map(Some).collect();
        let n_genes = Int64Array::from(
            (row_start..row_start + 3)
                .map(|i| (i as i64) * 10)
                .collect::<Vec<_>>(),
        );
        let batch = RecordBatch::try_new(
            obs_schema.clone(),
            vec![
                Arc::new(StringArray::from(
                    ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                )),
                Arc::new(dict),
                Arc::new(n_genes),
            ],
        )
        .unwrap();
        writer
            .write_obs_shard(shard_idx as u32, row_start as u64, 3, n_obs as u64, &batch)
            .unwrap();
    }
    writer.finish().unwrap();

    let reader = ScxReader::open(&path).unwrap();
    assert!(reader.obs_metadata_shard_count() >= 3);
    let full = reader.read_obs().unwrap();

    let to_utf8 = |a: &dyn Array| -> StringArray {
        arrow::compute::cast(a, &DataType::Utf8)
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .clone()
    };

    // Dictionary key column: projected assembly must match the full read,
    // same dtype (including the narrowed dictionary key) and same values.
    let keys = reader.read_obs_keys(&["cell_type".to_string()]).unwrap();
    assert_eq!(
        keys.num_columns(),
        1,
        "read_obs_keys must project to just the requested column"
    );
    let full_ct = full.column_by_name("cell_type").unwrap();
    let key_ct = keys.column_by_name("cell_type").unwrap();
    assert_eq!(
        full_ct.data_type(),
        key_ct.data_type(),
        "projected key dtype must match full read_obs"
    );
    assert_eq!(to_utf8(full_ct), to_utf8(key_ct));
    assert_eq!(
        to_utf8(key_ct),
        StringArray::from(vec!["A", "B", "A", "B", "C", "A", "C", "C", "B"])
    );

    // Numeric key column: projection works for non-dictionary columns too.
    let nkeys = reader.read_obs_keys(&["n_genes".to_string()]).unwrap();
    assert_eq!(nkeys.num_columns(), 1);
    let full_ng = full
        .column_by_name("n_genes")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let key_ng = nkeys
        .column_by_name("n_genes")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(full_ng, key_ng);

    // Plain `Utf8` key column: `read_obs_keys` dictionary-encodes string columns
    // during compaction (to drop the IPC-body alias), so the returned dtype is
    // `Dictionary` even though `read_obs` keeps `cell_id` plain. The *values*
    // must still match per row — that is the contract the sort relies on (the
    // RowConverter keys off decoded values, not dtype).
    let id_keys = reader.read_obs_keys(&["cell_id".to_string()]).unwrap();
    let id_col = id_keys.column_by_name("cell_id").unwrap();
    assert!(
        matches!(id_col.data_type(), DataType::Dictionary(_, _)),
        "plain Utf8 key should come back dictionary-encoded, got {:?}",
        id_col.data_type()
    );
    assert_eq!(
        to_utf8(id_col),
        to_utf8(full.column_by_name("cell_id").unwrap()),
        "decoded cell_id values must match the full read_obs",
    );

    // Unknown column is a clean error, not a panic.
    assert!(reader
        .read_obs_keys(&["does_not_exist".to_string()])
        .is_err());
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

    let sequential = assemble_x(&reader, &shards, RowMajorStrategy::Sequential).unwrap();
    let parallel = assemble_x(&reader, &shards, RowMajorStrategy::Parallel).unwrap();

    assert_eq!(sequential.shape, parallel.shape);
    assert_eq!(sequential.indptr, parallel.indptr);
    assert_eq!(sequential.indices, parallel.indices);
    assert_eq!(sequential.data, parallel.data);
}

// -----------------------------------------------------------------------
// Multi-shard concatenation (the assemblers' running row/nnz offsets)
// -----------------------------------------------------------------------

/// Test shim: assemble `X`-family shards with an explicit strategy.
fn assemble_x(
    reader: &ScxReader,
    shards: &[&FullCatalogEntry],
    strategy: RowMajorStrategy,
) -> Result<ScxCsr> {
    let n_vars = reader.header().n_vars as usize;
    reader.assemble_row_major(shards, n_vars, (0, n_vars), X_LABELS, strategy)
}

/// One row's nonzeros. `(global_row % 4) + 1` entries at columns walking by 3
/// from `global_row % n_vars`, values derived from the global row index.
///
/// Two properties are load-bearing: the per-row nnz **varies**, so a shard's
/// nnz cannot be recovered from its row count; and every value is distinct
/// enough that a nonzero landing in the wrong row shows up as a value
/// mismatch rather than cancelling out. Values are never 0 — a 0 would be
/// indistinguishable from an untouched slot in a freshly zeroed buffer, which
/// is exactly the failure a mis-computed offset produces.
fn irregular_row(global_row: usize, n_vars: usize) -> (Vec<u32>, Vec<u8>) {
    let nnz = (global_row % 4) + 1;
    // n_vars must exceed 3 * 3 for these to stay distinct without a dedup.
    assert!(
        n_vars > 9,
        "irregular_row needs n_vars > 9 to avoid collisions"
    );
    let cols: Vec<u32> = (0..nnz)
        .map(|k| ((global_row + k * 3) % n_vars) as u32)
        .collect();
    let mut sorted = cols.clone();
    sorted.sort_unstable();
    let values: Vec<u8> = sorted
        .iter()
        .enumerate()
        .map(|(k, _)| ((global_row * 7 + k * 3) % 200 + 1) as u8)
        .collect();
    (sorted, values)
}

/// Write `shard_rows.len()` CSR shards with the given per-shard row counts,
/// and return the assembled `(indptr, indices, data)` the reader must produce
/// — computed here by walking the rows linearly, independently of any
/// assembler.
fn write_irregular_shards(
    writer: &mut ScxWriter,
    shard_rows: &[usize],
    n_vars: usize,
    raw: bool,
) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
    let mut exp_indptr = vec![0i64];
    let mut exp_indices: Vec<i32> = Vec::new();
    let mut exp_data: Vec<f32> = Vec::new();

    let mut row_start = 0usize;
    for &rows in shard_rows {
        let mut indptr = vec![0u64];
        let mut indices: Vec<u32> = Vec::new();
        let mut values: Vec<u8> = Vec::new();
        for local in 0..rows {
            let (cols, vals) = irregular_row(row_start + local, n_vars);
            indptr.push(indptr.last().unwrap() + cols.len() as u64);
            exp_indptr.push(exp_indptr.last().unwrap() + cols.len() as i64);
            for (c, v) in cols.iter().zip(vals.iter()) {
                indices.push(*c);
                values.push(*v);
                exp_indices.push(*c as i32);
                exp_data.push(*v as f32);
            }
        }
        if raw {
            writer
                .write_raw_csr_shard(
                    &indptr,
                    &indices,
                    &values,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    row_start as u64,
                )
                .unwrap();
        } else {
            writer
                .write_csr_shard(
                    &indptr,
                    &indices,
                    &values,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    row_start as u64,
                )
                .unwrap();
        }
        row_start += rows;
    }
    (exp_indptr, exp_indices, exp_data)
}

/// Multi-shard concatenation is pinned by nothing else in the tree. Every one
/// of the golden files under `tests/reference_files/` is written with a
/// *single* `write_csr_shard` call (`golden_files.rs`), so the goldens
/// exercise decode and never exercise the running row / nnz offsets that
/// stitch shards together — and the backed reader reads shards one at a time,
/// so it does not cover them either.
///
/// That is the arithmetic a unification of the assemblers most easily breaks,
/// and it breaks into *shifted output*, not into an error: a wrong offset
/// writes a valid-looking CSR whose rows are off by one. Hence exact
/// `(indptr, indices, data)` assertions against a linearly computed
/// expectation, on every assembler that concatenates.
///
/// The geometry is deliberately irregular — shard row counts `[3, 7, 1, 5]`,
/// per-row nnz cycling `1..=4`, and a one-row shard in the middle — because
/// uniform shards make an off-by-one in a prefix sum invisible.
#[test]
fn multi_shard_assembly_is_pinned() {
    const SHARD_ROWS: &[usize] = &[3, 7, 1, 5];
    const N_VARS: usize = 11;
    const RAW_N_VARS: usize = 13;
    let n_obs: usize = SHARD_ROWS.iter().sum();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("irregular_multi_shard.scx");

    // `has_raw` is re-derived from the catalog at finalize, so it needs no
    // explicit set here.
    let header = sample_header(n_obs as u64, N_VARS as u64, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(N_VARS)).unwrap();
    let (exp_indptr, exp_indices, exp_data) =
        write_irregular_shards(&mut writer, SHARD_ROWS, N_VARS, false);

    // The raw matrix has its OWN, different column count. Assembling it
    // against `header.n_vars` instead of `raw_n_vars` is a distinct way to
    // get this wrong, so pin it in the same fixture.
    writer.set_raw_n_vars(RAW_N_VARS as u64);
    writer.write_raw_var(&sample_var(RAW_N_VARS)).unwrap();
    let (raw_indptr, raw_indices, raw_data) =
        write_irregular_shards(&mut writer, SHARD_ROWS, RAW_N_VARS, true);
    writer.finish().unwrap();

    // Premise: the fixture really is multi-shard with irregular geometry, and
    // the per-shard nnz really do differ. Without this the assertions below
    // could pass against a single-shard file and prove nothing.
    let reader = ScxReader::open(&path).unwrap();
    let shards = reader.catalog().shards_sorted();
    assert_eq!(
        shards.len(),
        SHARD_ROWS.len(),
        "fixture must be multi-shard"
    );
    let per_shard_nnz: Vec<u64> = shards
        .iter()
        .map(|e| e.stats.as_ref().unwrap().nnz)
        .collect();
    assert!(
        per_shard_nnz
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1,
        "shards must differ in nnz or a wrong offset stays invisible: {per_shard_nnz:?}"
    );
    assert!(exp_indptr.windows(2).any(|w| w[1] - w[0] != 1));

    let expect = |label: &str, csr: &ScxCsr, n_cols: usize, ip: &[i64], ix: &[i32], d: &[f32]| {
        assert_eq!(csr.shape, (n_obs, n_cols), "{label}: shape");
        assert_eq!(csr.indptr, ip, "{label}: indptr");
        assert_eq!(csr.indices, ix, "{label}: indices");
        assert_eq!(csr.data, d, "{label}: data");
    };

    expect(
        "read_all_csr_shards",
        &reader.read_all_csr_shards().unwrap(),
        N_VARS,
        &exp_indptr,
        &exp_indices,
        &exp_data,
    );
    expect(
        "assemble_row_major/Sequential",
        &assemble_x(&reader, &shards, RowMajorStrategy::Sequential).unwrap(),
        N_VARS,
        &exp_indptr,
        &exp_indices,
        &exp_data,
    );
    #[cfg(feature = "parallel")]
    expect(
        "assemble_row_major/Parallel",
        &assemble_x(&reader, &shards, RowMajorStrategy::Parallel).unwrap(),
        N_VARS,
        &exp_indptr,
        &exp_indices,
        &exp_data,
    );
    expect(
        "read_all_raw_csr_shards",
        &reader.read_all_raw_csr_shards().unwrap(),
        RAW_N_VARS,
        &raw_indptr,
        &raw_indices,
        &raw_data,
    );
}

/// The assembler's `madvise` hint is computed from raw catalog offsets, before
/// `section_bytes` gets a chance to reject the entry. `e.offset + e.length` was
/// a bare `u64` add and `max_end - min_offset` a bare `usize` subtract, so a
/// catalog with an absurd offset panicked the reader in debug (overflow) or
/// wrapped in release and then underflowed the subtraction.
///
/// A readahead hint is advisory, so the right answer is to skip it rather than
/// to fail: the entry is still rejected a moment later by `section_bytes`,
/// which is where a bad offset should surface. What must not happen is a panic
/// — `docs/conventions.md`: readers return errors on malformed input.
///
/// `read_all_raw_csr_shards` newly reaches this block: before the assemblers
/// were unified its hand-rolled body issued no hint at all.
#[test]
fn a_hostile_catalog_offset_does_not_panic_the_madvise_hint() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "hostile_offset.scx", 6, 4, 2, false);
    let reader = ScxReader::open(&path).unwrap();
    let shards = reader.catalog().shards_sorted();

    let mut doctored = shards[0].clone();
    doctored.offset = u64::MAX - 10;
    doctored.length = 100; // offset + length overflows u64

    for strategy in [
        RowMajorStrategy::Sequential,
        #[cfg(feature = "parallel")]
        RowMajorStrategy::Parallel,
    ] {
        let r = assemble_x(&reader, &[&doctored, shards[1]], strategy);
        assert!(
            r.is_err(),
            "{strategy:?}: an out-of-bounds shard offset must be an error"
        );
    }
}

/// A catalog whose `stats.nnz` disagrees with the decoded shard length must
/// return `Err` from both assemble paths, not panic. Before the fix the
/// decoded-vs-catalog length checks were `debug_assert_eq!` (compiled out in
/// release), so a mismatch panicked in `copy_from_slice` instead of erroring.
#[test]
fn test_assemble_shards_rejects_stat_drift() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "stat_drift.scx", 12, 10, 4, false);

    let reader = ScxReader::open(&path).unwrap();
    let shards = reader.catalog().shards_sorted();

    // Clone the first shard entry and inflate its catalog nnz so it no longer
    // matches what the shard actually decodes to.
    let mut doctored = shards[0].clone();
    doctored.stats.as_mut().unwrap().nnz += 1;

    assert!(
        assemble_x(&reader, &[&doctored], RowMajorStrategy::Sequential).is_err(),
        "sequential assemble must reject decoded-vs-catalog nnz drift"
    );

    #[cfg(feature = "parallel")]
    assert!(
        assemble_x(&reader, &[&doctored], RowMajorStrategy::Parallel).is_err(),
        "parallel assemble must reject decoded-vs-catalog nnz drift"
    );
}

/// A catalog whose `stats.row_end < stats.row_start` must return `Err` from
/// both assemble paths, not underflow-panic (debug) / wrap to a huge `usize`
/// (release) in the `(row_end - row_start)` shard-size precompute.
#[test]
fn test_assemble_shards_rejects_inverted_row_range() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "inverted_rows.scx", 12, 10, 4, false);

    let reader = ScxReader::open(&path).unwrap();
    let shards = reader.catalog().shards_sorted();

    // Clone the first shard entry and invert its row range.
    let mut doctored = shards[0].clone();
    {
        let s = doctored.stats.as_mut().unwrap();
        s.row_start = s.row_end + 1;
    }

    assert!(
        assemble_x(&reader, &[&doctored], RowMajorStrategy::Sequential).is_err(),
        "sequential assemble must reject inverted row range"
    );

    #[cfg(feature = "parallel")]
    assert!(
        assemble_x(&reader, &[&doctored], RowMajorStrategy::Parallel).is_err(),
        "parallel assemble must reject inverted row range"
    );
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

/// The modality table is parsed by *both* constructors, and until it was
/// extracted into one helper each carried its own byte-identical copy. Only
/// `open_inner`'s copy was ever exercised: every shared-catalog test uses a
/// single-modality file, where the branch is skipped entirely.
///
/// So: open a two-modality file through the shared path and check the table
/// actually came through, not just that the call returned `Ok`.
#[test]
fn open_with_shared_catalog_parses_the_modality_table() {
    use crate::modality::ModalityType;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shared_multimodal.scx");

    let header = sample_header(2, 4, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(2)).unwrap();
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
    writer.write_var_for(rna_id, &sample_var(4)).unwrap();
    writer.write_var_for(adt_id, &sample_var(2)).unwrap();
    writer.set_modality_n_vars(rna_id, 4).unwrap();
    writer.set_modality_n_vars(adt_id, 2).unwrap();
    writer
        .write_csr_shard_for(
            rna_id,
            &[0u64, 1, 2],
            &[0u32, 1],
            &[1u8, 2],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer.finish().unwrap();

    let primary = ScxReader::open(&path).unwrap();
    assert!(primary.is_multimodal(), "fixture premise");
    let shared = ScxReader::open_with_shared_catalog(&path, primary.catalog_arc()).unwrap();

    assert!(shared.is_multimodal());
    assert_eq!(shared.n_modalities(), primary.n_modalities());
    assert_eq!(shared.modality_names(), primary.modality_names());
    assert_eq!(shared.modality_id("adt"), primary.modality_id("adt"));
    assert_eq!(
        shared.modality_info(rna_id).map(|i| i.n_vars),
        Some(4),
        "per-modality n_vars must survive the shared-catalog open"
    );
}

/// `open_with_shared_catalog` must echo the offending path when the file is
/// missing, exactly as `open` / `open_unchecked` do
/// (`open_missing_file_error_includes_path`). It used bare `?` on `File::open`
/// and `Mmap::map`, so its error was a context-free "No such file or
/// directory (os error 2)" -- on the one path a DataLoader worker opens
/// thousands of times, where knowing *which* file is the whole diagnosis.
#[test]
fn open_with_shared_catalog_echoes_the_path() {
    let dir = tempfile::tempdir().unwrap();
    let real = write_test_file(&dir, "donor.scx", 4, 4, 1, false);
    let catalog = ScxReader::open(&real).unwrap().catalog_arc();

    let missing = dir.path().join("no_such_file_xyz.scx");
    let msg = match ScxReader::open_with_shared_catalog(&missing, catalog) {
        Ok(_) => panic!("expected open of a nonexistent path to fail"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains(&missing.display().to_string()),
        "shared-catalog open error should echo the offending path, got: {msg}"
    );
}

/// A v1 catalog that has NOT been reconciled must not reach
/// `open_with_shared_catalog` — and one that HAS been must still be accepted.
///
/// `open_inner` calls `reconcile_v1_csr_col_range(header.n_vars)` to backfill
/// `col_start` / `col_end` on v1 row-major entries. The shared path cannot: it
/// holds an `Arc<FullCatalog>` and the method takes `&mut self`.
///
/// The first version of this refused every `catalog_version < 2` catalog, which
/// was wrong in the direction that matters. `reconcile_v1_csr_col_range` does
/// not bump `catalog_version` — a reconciled v1 catalog still reports 1 — so
/// blanket rejection also refused the donors that come straight from
/// `ScxReader::open()`, and `pyscx.to_anndata(backed=True)` opens X through
/// this constructor unconditionally. A genuine v1 file that `open()` still
/// accepts would have failed on that Python surface.
///
/// So: verify the invariant instead of rejecting the version. An unreconciled
/// v1 entry carries `col_end == 0` (v1 stats have no column pair on disk and
/// `read_from` leaves it zeroed); a reconciled one carries
/// `col_end == header.n_vars`.
#[test]
fn open_with_shared_catalog_checks_v1_reconciliation_not_the_version() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "v1_donor.scx", 4, 4, 1, false);
    let reader = ScxReader::open(&path).unwrap();
    let n_vars = reader.header().n_vars;

    // (a) A reconciled v1 catalog — what `open()` actually hands out for a v1
    //     file — must be accepted.
    let mut reconciled = (*reader.catalog_arc()).clone();
    reconciled.catalog_version = 1;
    for e in &mut reconciled.entries {
        if let Some(st) = e.stats.as_mut() {
            if matches!(
                e.section_type,
                SectionType::CsrShard | SectionType::LayerCsrShard | SectionType::ObspCsrShard
            ) {
                st.col_start = 0;
                st.col_end = n_vars;
            }
        }
    }
    let shared = ScxReader::open_with_shared_catalog(&path, Arc::new(reconciled))
        .expect("a reconciled v1 catalog is what open() hands out; it must be accepted");
    assert_eq!(shared.catalog().catalog_version, 1);
    assert_eq!(shared.read_all_csr_shards().unwrap().shape, (4, 4));

    // (b) An UNRECONCILED v1 catalog — the state the shared path genuinely
    //     cannot fix up — must still be refused, with a message that says what
    //     to call instead.
    let mut unreconciled = (*reader.catalog_arc()).clone();
    unreconciled.catalog_version = 1;
    for e in &mut unreconciled.entries {
        if let Some(st) = e.stats.as_mut() {
            if matches!(
                e.section_type,
                SectionType::CsrShard | SectionType::LayerCsrShard | SectionType::ObspCsrShard
            ) {
                st.col_start = 0;
                st.col_end = 0;
            }
        }
    }
    let msg = match ScxReader::open_with_shared_catalog(&path, Arc::new(unreconciled)) {
        Ok(_) => panic!("an unreconciled v1 catalog must be refused"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("reconcile") && msg.contains("open()"),
        "the error must name the missing reconciliation and what to call instead, got: {msg}"
    );

    // (c) A v2+ catalog is untouched by any of this.
    assert!(
        ScxReader::open_with_shared_catalog(&path, reader.catalog_arc()).is_ok(),
        "the ordinary v2+ donor path must be unaffected"
    );
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

// ---------------------------------------------------------------------------
// 1C — obs_categorical / obs_categorical_many
// ---------------------------------------------------------------------------

/// Write a file whose obs is sharded with **disjoint per-shard categorical
/// vocabularies**, optionally writing `plain_shards` as plain `Utf8` rather than
/// `Dictionary` (the shape `append` produces).
///
/// `cell_type` sequence is A B A | B C A | C C B — every shard has a local
/// vocabulary that differs from its siblings, so a broken local→global remap
/// produces a correctly *shaped* result with wrong values.
fn write_categorical_obs_fixture(
    path: &std::path::Path,
    plain_shards: &[usize],
    with_nulls: bool,
) -> usize {
    use arrow::array::{DictionaryArray, Int64Array};
    use arrow::datatypes::Int8Type;

    let n_obs: usize = 9;
    let n_vars: usize = 4;
    let header = FileHeader::new_single_modality(n_obs as u64, n_vars as u64, 0, 3, 0, 0);
    let mut writer = ScxWriter::new(path, header).unwrap();

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
    writer.write_var(&sample_var(n_vars)).unwrap();

    let cell_types = [["A", "B", "A"], ["B", "C", "A"], ["C", "C", "B"]];
    for (shard_idx, types) in cell_types.iter().enumerate() {
        let row_start = shard_idx * 3;
        // With nulls, blank the middle row of shard 1 so a null sits mid-file
        // rather than at a boundary the loop might special-case.
        let rows: Vec<Option<&str>> = types
            .iter()
            .enumerate()
            .map(|(i, t)| {
                if with_nulls && shard_idx == 1 && i == 1 {
                    None
                } else {
                    Some(*t)
                }
            })
            .collect();

        let plain = plain_shards.contains(&shard_idx);
        let ct_field = Field::new(
            "cell_type",
            if plain {
                DataType::Utf8
            } else {
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8))
            },
            true,
        );
        let obs_schema = Arc::new(Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            ct_field,
            Field::new("n_genes", DataType::Int64, false),
        ]));

        let ct: Arc<dyn arrow::array::Array> = if plain {
            Arc::new(StringArray::from(rows.clone()))
        } else {
            Arc::new(rows.iter().copied().collect::<DictionaryArray<Int8Type>>())
        };
        let ids: Vec<String> = (row_start..row_start + 3)
            .map(|i| format!("cell_{i}"))
            .collect();
        let batch = RecordBatch::try_new(
            obs_schema,
            vec![
                Arc::new(StringArray::from(
                    ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                )),
                ct,
                Arc::new(Int64Array::from(
                    (row_start..row_start + 3)
                        .map(|i| (i as i64) * 10)
                        .collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        writer
            .write_obs_shard(shard_idx as u32, row_start as u64, 3, n_obs as u64, &batch)
            .unwrap();
    }
    writer.finish().unwrap();
    n_obs
}

/// Decode `(codes, categories)` back to strings so assertions read as data.
fn decode_codes(codes: &[i32], cats: &[String]) -> Vec<Option<String>> {
    codes
        .iter()
        .map(|&c| {
            if c < 0 {
                None
            } else {
                Some(cats[c as usize].clone())
            }
        })
        .collect()
}

/// `obs_categorical` must agree with the assembled `read_obs()` on every row,
/// across disjoint per-shard vocabularies.
#[test]
fn test_obs_categorical_matches_read_obs_on_sharded_dictionary() {
    use arrow::array::Array;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cat_dict.scx");
    write_categorical_obs_fixture(&path, &[], false);

    let reader = ScxReader::open(&path).unwrap();
    let (codes, cats) = reader.obs_categorical("cell_type").unwrap();

    // Ground truth from the materialising path.
    let full = reader.read_obs().unwrap();
    let expected = arrow::compute::cast(full.column_by_name("cell_type").unwrap(), &DataType::Utf8)
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .clone();
    let expected: Vec<Option<String>> = (0..expected.len())
        .map(|i| {
            if expected.is_null(i) {
                None
            } else {
                Some(expected.value(i).to_string())
            }
        })
        .collect();

    assert_eq!(decode_codes(&codes, &cats), expected);
    assert_eq!(codes.len(), 9, "one code per obs row");
    let mut sorted = cats.clone();
    sorted.sort();
    assert_eq!(sorted, vec!["A", "B", "C"], "one code per distinct value");
}

/// A file grown by `append` mixes `Dictionary` and plain `Utf8` shards on one
/// column. Both must fold into one vocabulary — the read side's
/// `reconcile_dictionary_representations` case, reached without a cast.
#[test]
fn test_obs_categorical_handles_mixed_dictionary_and_plain_shards() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cat_mixed.scx");
    // Shard 1 plain (as `append` writes), 0 and 2 dictionary-encoded.
    write_categorical_obs_fixture(&path, &[1], false);

    let reader = ScxReader::open(&path).unwrap();

    // Premise: the mix must actually exist on disk. Without this the fixture
    // could silently write every shard as Dictionary and the test below would
    // still pass while exercising only one representation.
    let ct_col = |shard: u32| -> DataType {
        let schema = reader.read_obs_schema_physical().unwrap();
        let idx = schema.index_of("cell_type").unwrap();
        reader
            .read_obs_shard_projected(shard, &[idx])
            .unwrap()
            .column(0)
            .data_type()
            .clone()
    };
    assert!(
        matches!(ct_col(0), DataType::Dictionary(_, _)),
        "shard 0 must be dictionary-encoded, got {:?}",
        ct_col(0)
    );
    // Plain string shards land as `LargeUtf8`, not `Utf8`: `write_arrow_ipc`
    // runs `upcast_to_large_types` on every batch. The projected read is the
    // *raw* on-disk view (no `downcast_large_types`), so this is what the
    // accumulator's plain arm actually receives.
    assert_eq!(
        ct_col(1),
        DataType::LargeUtf8,
        "shard 1 must be a plain string column (the shape `append` writes)"
    );

    let (codes, cats) = reader.obs_categorical("cell_type").unwrap();

    assert_eq!(
        decode_codes(&codes, &cats)
            .into_iter()
            .map(|v| v.unwrap())
            .collect::<Vec<_>>(),
        vec!["A", "B", "A", "B", "C", "A", "C", "C", "B"]
    );
    let mut sorted = cats.clone();
    sorted.sort();
    assert_eq!(
        sorted,
        vec!["A", "B", "C"],
        "mixed encodings must not duplicate categories"
    );
}

/// Nulls are pandas-coded as `-1`, not as a synthetic trailing level.
#[test]
fn test_obs_categorical_codes_nulls_as_minus_one() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cat_null.scx");
    write_categorical_obs_fixture(&path, &[], true);

    let reader = ScxReader::open(&path).unwrap();
    let (codes, cats) = reader.obs_categorical("cell_type").unwrap();
    assert_eq!(codes[4], -1, "the blanked row must code as -1");
    assert!(
        !cats.iter().any(|c| c == "NaN" || c.is_empty()),
        "null must not create a category: {cats:?}"
    );
    assert_eq!(codes.len(), 9);
}

/// The legacy single-section obs layout must go through the same fold.
#[test]
fn test_obs_categorical_on_legacy_single_section_obs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cat_legacy.scx");
    // `write_test_file` writes a legacy single-section obs via `write_obs`.
    write_test_file(&dir, "cat_legacy.scx", 6, 4, 2, false);

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(
        reader.obs_metadata_shard_count(),
        0,
        "premise: this fixture must be the legacy layout"
    );
    let (codes, cats) = reader.obs_categorical("cell_id").unwrap();
    assert_eq!(codes.len(), 6);
    // `cell_id` is unique per row, so every row gets its own category.
    assert_eq!(cats.len(), 6);
    assert_eq!(codes, vec![0, 1, 2, 3, 4, 5]);
}

/// `obs_categorical_many` must return one result per requested column, in order,
/// and take **one** projected read per shard rather than one per column.
#[test]
fn test_obs_categorical_many_is_one_pass() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cat_many.scx");
    write_categorical_obs_fixture(&path, &[], false);

    let reader = ScxReader::open(&path).unwrap();
    let before = reader
        .debug_counts()
        .read_obs_shard_projected
        .load(std::sync::atomic::Ordering::Relaxed);

    let cols = vec!["cell_type".to_string(), "cell_id".to_string()];
    let out = reader.obs_categorical_many(&cols).unwrap();
    assert_eq!(out.len(), 2, "one result per requested column, in order");

    let after = reader
        .debug_counts()
        .read_obs_shard_projected
        .load(std::sync::atomic::Ordering::Relaxed);
    // Debug-only counters: the `fetch_add` sites are `cfg(debug_assertions)`-gated,
    // so only assert when they are compiled in.
    if cfg!(debug_assertions) {
        assert_eq!(
            after - before,
            3,
            "3 obs shards ⇒ 3 projected reads for 2 columns, not 6"
        );
    }

    // Results must match the single-column accessor.
    let (ct_codes, ct_cats) = reader.obs_categorical("cell_type").unwrap();
    assert_eq!(out[0].0, ct_codes);
    assert_eq!(out[0].1, ct_cats);
    assert_eq!(out[1].1.len(), 9, "cell_id is unique per row");

    assert!(reader.obs_categorical_many(&[]).unwrap().is_empty());
}

/// The projected path must be *taken*, and the materialising ones avoided.
///
/// `read_obs == 0` alone is near-vacuous — `read_obs_keys` satisfies it too, and
/// so does anything else column-scoped — so the positive counter is what makes
/// this test mean something.
#[test]
fn test_obs_categorical_never_materialises_full_obs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cat_counts.scx");
    write_categorical_obs_fixture(&path, &[], false);

    let reader = ScxReader::open(&path).unwrap();
    reader.obs_categorical("cell_type").unwrap();

    let c = reader.debug_counts();
    let load = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(load(&c.read_obs), 0, "full obs must never be assembled");
    assert_eq!(load(&c.read_obs_shard), 0, "no whole-shard obs read either");
    if cfg!(debug_assertions) {
        assert_eq!(
            load(&c.read_obs_shard_projected),
            3,
            "the projected path must actually be taken, once per shard"
        );
    }
    // X must be untouched.
    assert_eq!(load(&c.read_shard_from_entry), 0);
}

/// Arrow IPC projection must return columns in the **requested** order, not in
/// ascending schema-index order — otherwise `obs_categorical_many` assigns each
/// accumulator the wrong column and silently swaps two columns' codes.
///
/// Covered on the sharded path by the Python `obs_categorical_many` test; this
/// pins the **legacy single-section** path, which goes through a different
/// projected reader (`FileReaderBuilder::with_projection` over the whole `obs`
/// section) and was otherwise only exercised with a single column.
#[test]
fn test_obs_categorical_many_preserves_request_order_on_legacy_obs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy_order.scx");

    let n_obs: usize = 4;
    let n_vars: usize = 4;
    let header = FileHeader::new_single_modality(n_obs as u64, n_vars as u64, 0, 4, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
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
    writer.write_var(&sample_var(n_vars)).unwrap();

    // Legacy single-section obs (`write_obs`) with TWO string columns whose
    // values are disjoint, so a projection swap cannot go unnoticed.
    let obs_schema = Arc::new(Schema::new(vec![
        Field::new("cell_type", DataType::Utf8, false),
        Field::new("donor", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        obs_schema,
        vec![
            Arc::new(StringArray::from(vec!["ct_a", "ct_b", "ct_a", "ct_c"])),
            Arc::new(StringArray::from(vec!["dn_x", "dn_y", "dn_y", "dn_x"])),
        ],
    )
    .unwrap();
    writer.write_obs(&batch).unwrap();
    writer.finish().unwrap();

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(
        reader.obs_metadata_shard_count(),
        0,
        "premise: this fixture must be the legacy single-section layout"
    );

    let solo_ct = reader.obs_categorical("cell_type").unwrap();
    let solo_dn = reader.obs_categorical("donor").unwrap();
    assert_eq!(solo_ct.1, vec!["ct_a", "ct_b", "ct_c"]);
    assert_eq!(solo_dn.1, vec!["dn_x", "dn_y"]);

    // `donor` is schema index 1 and `cell_type` index 0, so requesting
    // [donor, cell_type] is the descending-index case that would break if the
    // reader normalised the projection to ascending order.
    let rev = reader
        .obs_categorical_many(&["donor".to_string(), "cell_type".to_string()])
        .unwrap();
    assert_eq!(rev[0], solo_dn, "donor must land at request position 0");
    assert_eq!(rev[1], solo_ct, "cell_type must land at request position 1");

    let fwd = reader
        .obs_categorical_many(&["cell_type".to_string(), "donor".to_string()])
        .unwrap();
    assert_eq!(fwd[0], solo_ct);
    assert_eq!(fwd[1], solo_dn);
}

#[test]
fn test_obs_categorical_rejects_numeric_and_unknown_columns() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cat_reject.scx");
    write_categorical_obs_fixture(&path, &[], false);
    let reader = ScxReader::open(&path).unwrap();

    let err = reader.obs_categorical("n_genes").unwrap_err().to_string();
    assert!(err.contains("n_genes"), "must name the column: {err}");
    assert!(
        err.contains("string/categorical"),
        "must say what is supported: {err}"
    );

    let err = reader.obs_categorical("nope").unwrap_err().to_string();
    assert!(err.contains("nope"), "must name the column: {err}");
}

// ---------------------------------------------------------------------------
// Deletion-vector row filter: the keep mask must match the CSR it filters
// ---------------------------------------------------------------------------

/// Write a file whose header claims `n_obs` rows while the CSR shards cover
/// only `csr_rows`, with one deleted row so the deletion filter engages.
///
/// This is what a truncated file looks like from the reader's side: the keep
/// mask is built from `header.n_obs` (`build_keep_mask`), while the assembled
/// CSR's row count comes from summing the catalog's per-shard row ranges.
/// Nothing cross-checked them.
#[cfg(feature = "deletion-vectors")]
fn write_mask_csr_mismatch_file(
    dir: &tempfile::TempDir,
    filename: &str,
    n_obs: usize,
    csr_rows: usize,
) -> std::path::PathBuf {
    use crate::deletion_vectors::DeletionVectors;

    let n_vars = 8usize;
    let path = dir.path().join(filename);
    let (indptr, indices, values) = sample_shard_data(csr_rows, n_vars);
    let header = sample_header(n_obs as u64, n_vars as u64, *indptr.last().unwrap());
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
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
    let mut dv = DeletionVectors::new();
    dv.insert_global([0u32]);
    writer.write_deletion_vectors(&dv).unwrap();
    writer.finish().unwrap();
    path
}

/// A keep mask longer than the CSR indexed straight off the end of `indptr`
/// and panicked *inside the reader*. The convention is that a reader returns an
/// error on malformed input, so this must be `InvalidCatalog`.
#[cfg(feature = "deletion-vectors")]
#[test]
fn deletion_mask_longer_than_csr_errors() {
    let dir = tempfile::tempdir().unwrap();
    // 10 obs declared, 4 rows of CSR actually present.
    let path = write_mask_csr_mismatch_file(&dir, "mask_long.scx", 10, 4);
    let reader = ScxReader::open(&path).unwrap();

    let err = reader.read_all_csr_shards_filtered().unwrap_err();
    assert!(
        matches!(err, ScxError::InvalidCatalog(_)),
        "expected InvalidCatalog, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("10") && msg.contains('4'),
        "the error must report both counts so the file can be diagnosed: {msg}"
    );
}

/// The other direction, and the reason this enforces equality rather than only
/// the panicking bound: a mask *shorter* than the CSR never panics. It silently
/// drops every row past the end of the mask and returns a quietly truncated
/// matrix — the same corruption, delivered as an answer instead of a crash.
///
/// The pre-fix failure mode is therefore `Ok` with a wrong row count, which is
/// why this asserts the error rather than merely "does not panic".
#[cfg(feature = "deletion-vectors")]
#[test]
fn deletion_mask_shorter_than_csr_errors() {
    let dir = tempfile::tempdir().unwrap();
    // 4 obs declared, 10 rows of CSR actually present.
    let path = write_mask_csr_mismatch_file(&dir, "mask_short.scx", 4, 10);
    let reader = ScxReader::open(&path).unwrap();

    let err = reader.read_all_csr_shards_filtered().unwrap_err();
    assert!(
        matches!(err, ScxError::InvalidCatalog(_)),
        "expected InvalidCatalog, got {err:?}"
    );
}

/// The control both tests above need: when the mask and the CSR agree, the
/// filter still works and still deletes. Without this arm a fix that rejected
/// *every* file would pass the two tests above.
#[cfg(feature = "deletion-vectors")]
#[test]
fn deletion_mask_matching_the_csr_still_filters() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_mask_csr_mismatch_file(&dir, "mask_ok.scx", 6, 6);
    let reader = ScxReader::open(&path).unwrap();

    let csr = reader.read_all_csr_shards_filtered().unwrap();
    assert_eq!(csr.shape.0, 5, "one of six rows was deleted");
}

// -----------------------------------------------------------------------
// Zero-row shards, at file level
// -----------------------------------------------------------------------

/// The review asked for a zero-row *framed* shard round trip. Since §4.6 that
/// is impossible by construction — `encode_shard_framed` returns
/// `ZeroRowFramedShard` at write (`encoder.rs`), which is the fix. What §4.6
/// left uncovered is the arm it deliberately kept legal: an **unframed**
/// zero-row shard. `encoder.rs::zero_row_unframed_shard_is_still_accepted`
/// proves the encoder accepts one; nothing writes it to a real file and reads
/// it back.
///
/// Both shapes matter to the assemblers. A lone zero-row shard is the
/// empty-matrix case an `optimize` / `subset` output can produce; a zero-row
/// shard *between* two populated ones is what a delete-everything-in-a-shard
/// rewrite produces, and it is the one that walks an assembler's running row
/// and nnz offsets past a contributor of size 0.
#[test]
fn zero_row_unframed_shards_round_trip_through_a_file() {
    // (a) a file whose only shard is zero-row.
    let dir = tempfile::tempdir().unwrap();
    let only = dir.path().join("only_zero_row.scx");
    let mut header =
        FileHeader::new_single_modality(0, 2, 0, crate::DEFAULT_SHARD_TARGET_ROWS, 0, 0);
    header.format_version = 3;
    let mut writer = ScxWriter::new(&only, header).unwrap();
    writer.write_var(&sample_var(2)).unwrap();
    writer
        .write_csr_shard(&[0u64], &[], &[], CodecId::None, ValueEncoding::Uint8, 0)
        .unwrap();
    writer.finish().unwrap();

    let reader = ScxReader::open(&only).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape, (0, 2));
    assert_eq!(csr.indptr, vec![0]);
    assert!(csr.indices.is_empty());
    assert!(csr.data.is_empty());

    // (b) a zero-row shard sandwiched between two populated ones. Rows
    //     0..2 from shard 0, nothing from shard 1, rows 2..5 from shard 2.
    let sandwich = dir.path().join("zero_row_sandwich.scx");
    let mut header =
        FileHeader::new_single_modality(5, 2, 5, crate::DEFAULT_SHARD_TARGET_ROWS, 0, 0);
    header.format_version = 3;
    let mut writer = ScxWriter::new(&sandwich, header).unwrap();
    writer.write_var(&sample_var(2)).unwrap();
    writer
        .write_csr_shard(
            &[0u64, 1, 2],
            &[0u32, 1],
            &[7u8, 8],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer
        .write_csr_shard(&[0u64], &[], &[], CodecId::None, ValueEncoding::Uint8, 2)
        .unwrap();
    writer
        .write_csr_shard(
            &[0u64, 1, 2, 3],
            &[1u32, 0, 1],
            &[9u8, 10, 11],
            CodecId::None,
            ValueEncoding::Uint8,
            2,
        )
        .unwrap();
    writer.finish().unwrap();

    let reader = ScxReader::open(&sandwich).unwrap();
    // Premise: the middle shard really is zero-row on disk. Without this the
    // assertions below would also pass on a two-shard file.
    let shards = reader.catalog().shards_sorted();
    assert_eq!(shards.len(), 3, "fixture must keep the empty shard");
    assert!(
        shards.iter().any(|e| {
            let s = e.stats.as_ref().unwrap();
            s.row_end == s.row_start && s.nnz == 0
        }),
        "one shard must be genuinely zero-row"
    );

    let check = |label: &str, csr: &ScxCsr| {
        assert_eq!(csr.shape, (5, 2), "{label}: shape");
        assert_eq!(csr.indptr, vec![0, 1, 2, 3, 4, 5], "{label}: indptr");
        assert_eq!(csr.indices, vec![0, 1, 1, 0, 1], "{label}: indices");
        assert_eq!(csr.data, vec![7.0, 8.0, 9.0, 10.0, 11.0], "{label}: data");
    };
    check(
        "read_all_csr_shards",
        &reader.read_all_csr_shards().unwrap(),
    );
    check(
        "assemble_row_major/Sequential",
        &assemble_x(&reader, &shards, RowMajorStrategy::Sequential).unwrap(),
    );
    #[cfg(feature = "parallel")]
    check(
        "assemble_row_major/Parallel",
        &assemble_x(&reader, &shards, RowMajorStrategy::Parallel).unwrap(),
    );
}

// -----------------------------------------------------------------------
// CSC sidecar freshness on the ScxReader paths (review 4.7)
// -----------------------------------------------------------------------

/// Write a small file carrying both a CSR shard and a CSC sidecar.
///
/// `framed` selects the CSC encoding, and it decides which arm of
/// `read_csc_columns` the file exercises: a row-group-framed (v2) CSC shard
/// takes the `decode_block_index_row_runs` scatter path, an unframed one takes
/// the full-decode + `col_slice` path. The freshness fixtures below run both,
/// because the first version of them ran only the unframed arm and so passed
/// while the framed arm served a stale sidecar.
fn write_reader_csc_fixture(
    dir: &tempfile::TempDir,
    name: &str,
    framed: bool,
) -> std::path::PathBuf {
    let n_obs = 6usize;
    let n_vars = 4usize;
    let path = dir.path().join(name);
    let mut header = sample_header(n_obs as u64, n_vars as u64, 0);
    if framed {
        header.format_version = crate::header::CURRENT_FORMAT_VERSION;
    }
    let mut writer = ScxWriter::new(&path, header).unwrap();
    if framed {
        writer.set_framing(Some(crate::encoder::FramingConfig {
            row_group_rows: 1,
            target_nnz: None,
            trial: false,
            decode_target: None,
        }));
    }
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    // Dense reference, then the same values as CSR and as CSC.
    let mut dense = vec![0u8; n_obs * n_vars];
    for (r, row) in dense.chunks_mut(n_vars).enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            if (r + c) % 2 == 0 {
                *cell = ((r * 5 + c * 3) % 200 + 1) as u8;
            }
        }
    }

    let mut ip = vec![0u64];
    let mut ix: Vec<u32> = Vec::new();
    let mut vals: Vec<u8> = Vec::new();
    for r in 0..n_obs {
        for c in 0..n_vars {
            let v = dense[r * n_vars + c];
            if v != 0 {
                ix.push(c as u32);
                vals.push(v);
            }
        }
        ip.push(ix.len() as u64);
    }
    writer
        .write_csr_shard(&ip, &ix, &vals, CodecId::None, ValueEncoding::Uint8, 0)
        .unwrap();

    // Two CSC shards, two columns each.
    let mut col_start = 0usize;
    while col_start < n_vars {
        let col_end = (col_start + 2).min(n_vars);
        let mut cip = vec![0u64];
        let mut cix: Vec<u32> = Vec::new();
        let mut cvals: Vec<u8> = Vec::new();
        for c in col_start..col_end {
            for r in 0..n_obs {
                let v = dense[r * n_vars + c];
                if v != 0 {
                    cix.push(r as u32);
                    cvals.push(v);
                }
            }
            cip.push(cix.len() as u64);
        }
        writer
            .write_csc_shard(
                &cip,
                &cix,
                &cvals,
                CodecId::None,
                ValueEncoding::Uint8,
                col_start as u64,
            )
            .unwrap();
        col_start = col_end;
    }
    writer.finish().unwrap();
    path
}

/// Rewrite the file's catalog with `data_generation` bumped, leaving the CSC
/// sidecar's `csc_build_generation` behind — exactly the state a mutating op
/// leaves when it rewrites X and carries an old sidecar across.
///
/// Only a `u64` in the v4 trailer changes, so the re-serialized catalog is the
/// same length and can be spliced back in place; the assertion below makes
/// that a checked assumption rather than a hope. `FullCatalog::write_to`
/// recomputes the catalog's own BLAKE3, so the file still opens.
fn bump_data_generation_on_disk(path: &std::path::Path) {
    let mut bytes = std::fs::read(path).unwrap();
    let hdr = FileHeader::read_from(&mut std::io::Cursor::new(&bytes)).unwrap();
    let fc_start = hdr.full_catalog_offset as usize;
    let fc_len = hdr.full_catalog_length as usize;

    let mut catalog = FullCatalog::read_from(
        &mut std::io::Cursor::new(&bytes[fc_start..fc_start + fc_len]),
        fc_len,
        true,
    )
    .unwrap();
    assert_eq!(
        catalog.csc_build_generation, catalog.data_generation,
        "fixture premise: the written file starts fresh"
    );
    catalog.data_generation += 1;

    let mut buf = Vec::new();
    catalog.write_to(&mut buf).unwrap();
    assert_eq!(
        buf.len(),
        fc_len,
        "bumping a u64 must not change the catalog's serialized length"
    );
    bytes[fc_start..fc_start + fc_len].copy_from_slice(&buf);
    std::fs::write(path, &bytes).unwrap();
}

/// Review §4.7: `check_csc_sidecar_fresh` was called from `BackedCscReader`'s
/// constructors and nowhere else, so every `ScxReader` CSC read served a stale
/// sidecar without complaint — the silent wrong answer the v4 generation
/// counters exist to prevent.
///
/// The check does **not** live on `read_csc_from_entry`, which is where this
/// test first put it on the reasoning that every CSC read funnels through it.
/// Two funnel around it: the framed arm of `read_csc_columns` decodes via
/// `decode_block_index_row_runs` and `continue`s, and `scx upgrade` copies a
/// sidecar with `read_shard_from_entry`, then re-stamps it at the output's
/// generation — turning a detectable stale sidecar into an undetectable one.
/// So the guard sits at the four decode entry points instead, and this test
/// runs both arms and the raw decoders. See
/// `ScxReader::guard_csc_sidecar_fresh`.
#[test]
fn stale_csc_sidecar_is_refused_on_every_reader_csc_path() {
    for framed in [false, true] {
        stale_csc_sidecar_is_refused_impl(framed);
    }
}

/// `framed` picks which arm of `read_csc_columns` runs: the framed (v2) shard
/// takes the `decode_block_index_row_runs` scatter path, the unframed one the
/// full-decode + `col_slice` path. The first version of this fixture ran only
/// the unframed arm, so it passed while the scatter arm served a stale sidecar.
fn stale_csc_sidecar_is_refused_impl(framed: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = write_reader_csc_fixture(&dir, "stale_csc.scx", framed);
    bump_data_generation_on_disk(&path);

    let reader = ScxReader::open(&path).unwrap();
    // Premise: the file really is stale now, and really does have a sidecar.
    assert_ne!(
        reader.catalog().csc_build_generation,
        reader.catalog().data_generation
    );
    assert_eq!(reader.csc_shard_count(), 2);

    let expect_stale = |label: &str, r: Result<ScxCsc>| match r {
        Ok(_) => panic!("{label}: a stale CSC sidecar must not be served"),
        Err(e) => assert!(
            matches!(e, ScxError::StaleCscSidecar { .. }),
            "{label}: expected StaleCscSidecar, got {e}"
        ),
    };

    expect_stale("read_csc_shard", reader.read_csc_shard(0));
    expect_stale("read_csc_columns", reader.read_csc_columns(0..2));
    expect_stale(
        "read_csc_columns_subset",
        reader.read_csc_columns_subset(&[0, 3]),
    );
    expect_stale("read_all_csc_shards", reader.read_all_csc_shards());
    expect_stale("read_csc_shard_for", reader.read_csc_shard_for(0, 0));
    expect_stale("read_csc_columns_for", reader.read_csc_columns_for(0, 0..2));
    expect_stale(
        "read_csc_columns_subset_for",
        reader.read_csc_columns_subset_for(0, &[0, 3]),
    );

    // The raw decode entry points, not just the CSC-shaped wrappers. `scx
    // upgrade` copies a sidecar forward with `read_shard_from_entry` and then
    // re-stamps it at the output's generation, which turns a *detectable*
    // stale sidecar into an undetectable one — strictly worse than not
    // checking. Guarding only the wrappers leaves that path open.
    let csc_entry = reader.catalog().csc_shards_sorted()[0].clone();
    let raw = |label: &str, r: Result<(Vec<i64>, Vec<i32>, Vec<f32>)>| match r {
        Ok(_) => panic!("{label}: a stale CSC sidecar must not decode"),
        Err(e) => assert!(
            matches!(e, ScxError::StaleCscSidecar { .. }),
            "{label}: expected StaleCscSidecar, got {e}"
        ),
    };
    raw(
        "read_shard_from_entry",
        reader.read_shard_from_entry(&csc_entry),
    );
    raw(
        "read_shard_from_entry_verified",
        reader.read_shard_from_entry_verified(&csc_entry),
    );
    match reader.read_shard_indptr_from_entry(&csc_entry) {
        Ok(_) => panic!("read_shard_indptr_from_entry: stale sidecar must not decode"),
        Err(e) => assert!(matches!(e, ScxError::StaleCscSidecar { .. }), "got {e}"),
    }

    // A CSR entry in the same file must be unaffected — the guard keys on the
    // entry's section type, not on the file merely having a stale sidecar.
    let csr_entry = reader.catalog().shards_sorted()[0].clone();
    assert!(
        reader.read_shard_from_entry(&csr_entry).is_ok(),
        "the CSR side of a file with a stale CSC sidecar must still read"
    );
    assert!(reader.read_all_csr_shards().is_ok());
}

/// The accept side. A guard watched only in the red direction is proven to
/// fire, not proven to be aimed correctly — #436 shipped one that also broke
/// reading real f32 counts. So: a freshly written sidecar must still be served
/// by every path above, and a file with no sidecar at all must be unaffected.
#[test]
fn a_fresh_csc_sidecar_is_still_served_on_every_reader_csc_path() {
    for framed in [false, true] {
        a_fresh_csc_sidecar_is_still_served_impl(framed);
    }
}

fn a_fresh_csc_sidecar_is_still_served_impl(framed: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = write_reader_csc_fixture(&dir, "fresh_csc.scx", framed);

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(
        reader.catalog().csc_build_generation,
        reader.catalog().data_generation
    );

    assert_eq!(reader.read_csc_shard(0).unwrap().shape.1, 2);
    assert_eq!(reader.read_csc_columns(0..2).unwrap().shape.1, 2);
    assert_eq!(reader.read_csc_columns_subset(&[0, 3]).unwrap().shape.1, 2);
    assert_eq!(reader.read_all_csc_shards().unwrap().shape.1, 4);
    assert_eq!(reader.read_csc_shard_for(0, 0).unwrap().shape.1, 2);
    assert_eq!(reader.read_csc_columns_for(0, 0..2).unwrap().shape.1, 2);
    assert_eq!(
        reader
            .read_csc_columns_subset_for(0, &[0, 3])
            .unwrap()
            .shape
            .1,
        2
    );

    // A file with no CSC sidecar must not be dragged into the new guard: its
    // counters are whatever the writer left, and there is nothing to validate.
    let plain = write_test_file(&dir, "no_csc.scx", 6, 4, 2, false);
    let plain_reader = ScxReader::open(&plain).unwrap();
    assert_eq!(plain_reader.csc_shard_count(), 0);
    assert_eq!(plain_reader.read_all_csr_shards().unwrap().shape, (6, 4));
}
