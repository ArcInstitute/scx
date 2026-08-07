//! Tests for [`attach_external_obs`].
//!
//! Priorities, roughly by how badly a regression would hurt:
//!
//! 1. **The join is by key.** A permuted source must land each row on the right
//!    cell. A doublet caller run per-library returns rows in its own order, and
//!    a positional assumption silently puts every score on the wrong cell while
//!    still producing a correctly-shaped column.
//! 2. **Overwrite replaces, it does not merge.** Importing N per-batch tables
//!    one after another keeps only the last. The op must fail loudly without
//!    `overwrite`, and the replace-not-merge semantics must be pinned so nobody
//!    "helpfully" softens them later.
//! 3. **Nothing else is lost.** var, X, the CSC sidecar, deletion vectors and
//!    the provenance chain all survive; `scx rollback` undoes the lot; and the
//!    obs predicate index survives an *add* but is dropped on an *overwrite* of
//!    a column it covers.
//! 4. **A failed key names the fix.** On a real atlas there is often no barcode
//!    column and the index is duplicated, so the error has to say which columns
//!    would work.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, Float32Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;

use super::*;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn obs_batch(n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let schema = Schema::new(vec![Field::new("barcode", DataType::Utf8, false)]);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(StringArray::from(ids))]).unwrap()
}

fn var_batch(n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("g{i}")).collect();
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(StringArray::from(ids))]).unwrap()
}

/// An SCX file with `n_shards` X shards tiling `[0, n_obs)`. X itself is empty
/// (the op never reads it), but its shards and `var` must survive untouched.
fn write_fixture_with_obs(
    dir: &Path,
    name: &str,
    obs: RecordBatch,
    n_vars: usize,
    n_shards: usize,
) -> PathBuf {
    let n_obs = obs.num_rows();
    let path = dir.join(name);
    let rows_per = n_obs.div_ceil(n_shards);
    let header =
        FileHeader::new_single_modality(n_obs as u64, n_vars as u64, 0, rows_per as u32, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&obs).unwrap();
    writer.write_var(&var_batch(n_vars)).unwrap();

    let mut start = 0usize;
    while start < n_obs {
        let take = rows_per.min(n_obs - start);
        let indptr: Vec<u64> = vec![0u64; take + 1];
        writer
            .write_csr_shard(
                &indptr,
                &[],
                &[],
                CodecId::None,
                ValueEncoding::Uint8,
                start as u64,
            )
            .unwrap();
        start += take;
    }
    writer
        .write_uns(&serde_json::json!({"state": "v0"}))
        .unwrap();
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1_710_000_000,
            action: "create".to_string(),
            tool: "test".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();
    writer.finish().unwrap();
    path
}

fn write_fixture(dir: &Path, name: &str, n_obs: usize, n_vars: usize, n_shards: usize) -> PathBuf {
    write_fixture_with_obs(dir, name, obs_batch(n_obs), n_vars, n_shards)
}

/// Score/call annotations keyed by `row_keys`. `score(i)` varies per row so a
/// mis-join shows up in the values, not just in the row count.
fn score_data(row_keys: Vec<String>, score: impl Fn(usize) -> f32) -> ExternalObsData {
    let n = row_keys.len();
    let scores: Vec<f32> = (0..n).map(&score).collect();
    let calls: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
    let schema = Schema::new(vec![
        Field::new("dbl_score", DataType::Float32, true),
        Field::new("dbl_call", DataType::Boolean, true),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float32Array::from(scores)),
            Arc::new(BooleanArray::from(calls)),
        ],
    )
    .unwrap();
    ExternalObsData {
        row_keys,
        row_annotations: batch,
        row_embeddings: Vec::new(),
        uns: None,
        source_checksum: None,
        source_name: Some("calls.csv".to_string()),
    }
}

fn opts() -> AttachObsOptions {
    AttachObsOptions {
        status_column: Some("dbl_status".to_string()),
        provenance_action: "test_obs_import".to_string(),
        ..Default::default()
    }
}

fn keys(prefix: &str, n: usize) -> Vec<String> {
    (0..n).map(|i| format!("{prefix}{i}")).collect()
}

fn f32_col(batch: &RecordBatch, name: &str) -> Float32Array {
    batch
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap()
        .clone()
}

fn str_col(batch: &RecordBatch, name: &str) -> StringArray {
    let col = batch.column_by_name(name).unwrap();
    arrow::compute::cast(col, &DataType::Utf8)
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .clone()
}

// ---------------------------------------------------------------------------
// Happy path
// ---------------------------------------------------------------------------

#[test]
fn attaches_columns_covering_the_full_obs_axis() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 6, 3, 2);
    let data = score_data(keys("cell_", 6), |i| i as f32 + 0.5);

    let s = attach_external_obs(&path, &data, &opts()).unwrap();
    assert_eq!(s.n_obs, 6);
    assert_eq!(s.n_matched, 6);
    assert_eq!(s.n_target_rows_absent, 0);
    assert_eq!(s.n_source_rows_absent, 0);
    assert_eq!(s.obs_key_column, "barcode");
    assert!(!s.obs_index_dropped);

    let obs = ScxReader::open(&path).unwrap().read_obs().unwrap();
    assert_eq!(obs.num_rows(), 6);
    let scores = f32_col(&obs, "dbl_score");
    for i in 0..6 {
        assert_eq!(scores.value(i), i as f32 + 0.5);
    }
    // The pre-existing column survives alongside the new ones.
    assert!(obs.column_by_name("barcode").is_some());
    assert!(obs.column_by_name("dbl_call").is_some());
    assert_eq!(str_col(&obs, "dbl_status").value(0), "present");
}

#[test]
fn permuted_source_rows_join_by_key_not_position() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 6, 3, 2);

    // Reverse the source order and give each key a value derived from the key
    // itself, so a positional join produces visibly wrong numbers.
    let mut reversed = keys("cell_", 6);
    reversed.reverse();
    let scores: Vec<f32> = reversed
        .iter()
        .map(|k| k.trim_start_matches("cell_").parse::<f32>().unwrap() * 10.0)
        .collect();
    let schema = Schema::new(vec![Field::new("dbl_score", DataType::Float32, true)]);
    let batch =
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(Float32Array::from(scores))]).unwrap();
    let data = ExternalObsData {
        row_keys: reversed,
        row_annotations: batch,
        row_embeddings: Vec::new(),
        uns: None,
        source_checksum: None,
        source_name: None,
    };

    attach_external_obs(&path, &data, &opts()).unwrap();

    let obs = ScxReader::open(&path).unwrap().read_obs().unwrap();
    let scores = f32_col(&obs, "dbl_score");
    for i in 0..6 {
        assert_eq!(
            scores.value(i),
            i as f32 * 10.0,
            "row {i} got a value from the wrong source row — the join went positional"
        );
    }
}

#[test]
fn unmatched_target_rows_get_null_and_absent_status() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 6, 3, 1);
    // The tool only analysed the first three cells.
    let data = score_data(keys("cell_", 3), |i| i as f32 + 1.0);

    let s = attach_external_obs(&path, &data, &opts()).unwrap();
    assert_eq!(s.n_matched, 3);
    assert_eq!(s.n_target_rows_absent, 3);

    let obs = ScxReader::open(&path).unwrap().read_obs().unwrap();
    let scores = f32_col(&obs, "dbl_score");
    for i in 0..3 {
        assert!(scores.is_valid(i));
    }
    for i in 3..6 {
        assert!(
            scores.is_null(i),
            "row {i} must be null, not 0.0 — a score of zero is a claim the tool never made"
        );
    }
    let status = str_col(&obs, "dbl_status");
    assert_eq!(status.value(0), "present");
    assert_eq!(status.value(5), "absent");
}

#[test]
fn extra_source_rows_are_skipped_and_counted() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 3, 3, 1);
    // Ran on the raw all-droplet file; importing onto filtered cells.
    let data = score_data(keys("cell_", 8), |i| i as f32);

    let s = attach_external_obs(&path, &data, &opts()).unwrap();
    assert_eq!(s.n_matched, 3);
    assert_eq!(s.n_source_rows_absent, 5);
    assert_eq!(s.n_target_rows_absent, 0);
}

// ---------------------------------------------------------------------------
// Join failures
// ---------------------------------------------------------------------------

#[test]
fn zero_overlap_is_a_hard_error_naming_examples_from_both_sides() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 2, 1);
    let data = score_data(keys("OTHER_", 4), |i| i as f32);

    let err = attach_external_obs(&path, &data, &opts()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("cell_0"), "{msg}");
    assert!(msg.contains("OTHER_0"), "{msg}");
    assert!(msg.contains("suffix") || msg.contains("prefix"), "{msg}");
}

#[test]
fn duplicate_target_keys_error() {
    let dir = tempfile::tempdir().unwrap();
    // Two cells share a barcode — the merged-atlas shape.
    let obs = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "barcode",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(vec!["a", "b", "a", "c"]))],
    )
    .unwrap();
    let path = write_fixture_with_obs(dir.path(), "a.scx", obs, 2, 1);
    let data = score_data(vec!["a".into(), "b".into(), "c".into()], |i| i as f32);

    let err = attach_external_obs(&path, &data, &opts()).unwrap_err();
    assert!(matches!(err, OpsError::DuplicateJoinKey { .. }), "{err}");
}

#[test]
fn duplicate_source_keys_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 3, 2, 1);
    let data = score_data(
        vec!["cell_0".into(), "cell_1".into(), "cell_1".into()],
        |i| i as f32,
    );

    let err = attach_external_obs(&path, &data, &opts()).unwrap_err();
    assert!(matches!(err, OpsError::DuplicateJoinKey { .. }), "{err}");
}

#[test]
fn missing_and_extra_row_error_policies_are_honoured() {
    let dir = tempfile::tempdir().unwrap();

    let path = write_fixture(dir.path(), "m.scx", 5, 2, 1);
    let data = score_data(keys("cell_", 3), |i| i as f32);
    let err = attach_external_obs(
        &path,
        &data,
        &AttachObsOptions {
            missing_row_policy: MissingRowPolicy::Error,
            ..opts()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("missing_row_policy"), "{err}");

    let path = write_fixture(dir.path(), "e.scx", 3, 2, 1);
    let data = score_data(keys("cell_", 5), |i| i as f32);
    let err = attach_external_obs(
        &path,
        &data,
        &AttachObsOptions {
            extra_row_policy: ExtraRowPolicy::Error,
            ..opts()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("extra_row_policy"), "{err}");
}

#[test]
fn an_unknown_explicit_key_column_is_rejected_with_candidates_listed() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 3, 2, 1);
    let data = score_data(keys("cell_", 3), |i| i as f32);

    let err = attach_external_obs(
        &path,
        &data,
        &AttachObsOptions {
            join_key: ObsJoinKey::Column("nope".into()),
            ..opts()
        },
    )
    .unwrap_err();
    assert!(matches!(err, OpsError::KeyColumnUnresolved { .. }), "{err}");
    assert!(err.to_string().contains("barcode"), "{err}");
}

// ---------------------------------------------------------------------------
// Composite keys
// ---------------------------------------------------------------------------

/// obs where the barcode repeats across two libraries, so only `sample+barcode`
/// is unique. This is the genuine multi-library 10x-merge shape.
fn obs_two_libraries() -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("sample_id", DataType::Utf8, false),
        Field::new("barcode", DataType::Utf8, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(vec!["A", "A", "B", "B"])),
            Arc::new(StringArray::from(vec![
                "AAAC-1", "AAAG-1", "AAAC-1", "AAAG-1",
            ])),
        ],
    )
    .unwrap()
}

#[test]
fn composite_key_disambiguates_duplicate_barcodes_across_batches() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture_with_obs(dir.path(), "a.scx", obs_two_libraries(), 2, 1);

    // A bare barcode key cannot work here.
    let flat = score_data(vec!["AAAC-1".into(), "AAAG-1".into()], |i| i as f32);
    let err = attach_external_obs(
        &path,
        &flat,
        &AttachObsOptions {
            join_key: ObsJoinKey::Column("barcode".into()),
            ..opts()
        },
    )
    .unwrap_err();
    assert!(matches!(err, OpsError::DuplicateJoinKey { .. }), "{err}");

    // The composite does, and lands each library's score on its own rows.
    let source_obs = obs_two_libraries();
    let cols = vec!["sample_id".to_string(), "barcode".to_string()];
    let row_keys = build_composite_key(&source_obs, &cols).unwrap();
    let data = score_data(row_keys, |i| i as f32 * 100.0);

    let s = attach_external_obs(
        &path,
        &data,
        &AttachObsOptions {
            join_key: ObsJoinKey::Composite { columns: cols },
            ..opts()
        },
    )
    .unwrap();
    assert_eq!(s.n_matched, 4);
    assert_eq!(s.obs_key_column, "sample_id,barcode");

    let obs = ScxReader::open(&path).unwrap().read_obs().unwrap();
    let scores = f32_col(&obs, "dbl_score");
    for i in 0..4 {
        assert_eq!(scores.value(i), i as f32 * 100.0);
    }
}

#[test]
fn composite_key_component_containing_the_separator_is_rejected() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Utf8, false),
        Field::new("b", DataType::Utf8, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(vec![format!(
                "x{COMPOSITE_KEY_SEPARATOR}y"
            )])),
            Arc::new(StringArray::from(vec!["z"])),
        ],
    )
    .unwrap();

    let err = build_composite_key(&batch, &["a".to_string(), "b".to_string()]).unwrap_err();
    assert!(err.to_string().contains("U+001F"), "{err}");
}

#[test]
fn composite_key_missing_column_is_rejected_listing_present_columns() {
    let batch = obs_two_libraries();
    let err = build_composite_key(&batch, &["sample_id".into(), "nope".into()]).unwrap_err();
    assert!(matches!(err, OpsError::KeyColumnUnresolved { .. }), "{err}");
    assert!(err.to_string().contains("barcode"), "{err}");
}

#[test]
fn a_single_element_composite_is_just_that_column() {
    let batch = obs_two_libraries();
    let one = build_composite_key(&batch, &["barcode".to_string()]).unwrap();
    assert_eq!(one, vec!["AAAC-1", "AAAG-1", "AAAC-1", "AAAG-1"]);
}

// ---------------------------------------------------------------------------
// Key diagnosis
// ---------------------------------------------------------------------------

/// The measured `census_1m.scx` shape in miniature: no barcode column, an index
/// that repeats, and exactly one unique column that no fallback list would guess.
fn obs_census_shaped() -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("index", DataType::Utf8, false),
        Field::new("soma_joinid", DataType::Int64, false),
        Field::new("dataset_id", DataType::Utf8, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            // 0,1,0,1 — a stringified RangeIndex reused across chunks.
            Arc::new(StringArray::from(vec!["0", "1", "0", "1"])),
            Arc::new(Int64Array::from(vec![5718, 5719, 5720, 5721])),
            Arc::new(StringArray::from(vec!["d1", "d1", "d2", "d2"])),
        ],
    )
    .unwrap()
}

#[test]
fn duplicate_key_error_names_the_columns_that_are_unique() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture_with_obs(dir.path(), "a.scx", obs_census_shaped(), 2, 1);
    let data = score_data(vec!["0".into(), "1".into()], |i| i as f32);

    let err = attach_external_obs(&path, &data, &opts()).unwrap_err();
    let msg = err.to_string();
    assert!(matches!(err, OpsError::DuplicateJoinKey { .. }), "{msg}");
    assert!(
        msg.contains("soma_joinid"),
        "the error must name the column that WOULD work: {msg}"
    );
    assert!(msg.contains("Try key ="), "{msg}");
}

#[test]
fn diagnose_obs_key_reports_unique_columns_and_a_suggestion() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture_with_obs(dir.path(), "a.scx", obs_census_shaped(), 2, 1);

    let d = diagnose_obs_key(&path, Some(&ObsJoinKey::Column("index".into()))).unwrap();
    assert_eq!(d.n_obs, 4);
    assert_eq!(d.resolved_key.as_deref(), Some("index"));
    assert_eq!(d.resolved_cardinality, Some(2));
    assert_eq!(d.unique_columns, vec!["soma_joinid".to_string()]);
    assert_eq!(d.suggestion.as_deref(), Some("soma_joinid"));
}

// ---------------------------------------------------------------------------
// The obs index is reported as `obs_names`, and accepted back under that name
// ---------------------------------------------------------------------------

/// A pyarrow-written obs: the index is a physical `__index_level_0__` field,
/// which is what every `from_anndata` / h5ad-converted file actually carries.
fn obs_with_pyarrow_index() -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("cell_uid", DataType::Utf8, false),
        Field::new("donor", DataType::Utf8, false),
        // Appended last, exactly as `Table.from_pandas` emits it — which is why
        // schema order alone used to rank it behind every data column.
        Field::new("__index_level_0__", DataType::Utf8, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(vec!["u0", "u1", "u2", "u3"])),
            Arc::new(StringArray::from(vec!["d1", "d1", "d2", "d2"])),
            Arc::new(StringArray::from(vec!["AAAC", "AAAG", "AAAT", "AAAA"])),
        ],
    )
    .unwrap()
}

#[test]
fn the_diagnosis_never_leaks_the_physical_index_field_name() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture_with_obs(dir.path(), "a.scx", obs_with_pyarrow_index(), 2, 1);

    // `Some(Auto)`, not `None` — that is what the pyscx wrapper passes for a
    // bare `diagnose_obs_key(path)`, and it is the only form that reports what
    // auto-resolution would land on.
    let d = diagnose_obs_key(&path, Some(&ObsJoinKey::Auto)).unwrap();
    let rendered = format!("{d:?} {}", d.describe());
    assert!(
        !rendered.contains("__index_level_0__"),
        "`__index_level_0__` is pyarrow's serialization name; `read_obs()` hands \
         that field back as the frame's UNNAMED index, so following the \
         diagnosis literally is a KeyError: {rendered}"
    );
    assert_eq!(d.resolved_key.as_deref(), Some("obs_names"));
    assert_eq!(d.suggestion.as_deref(), Some("obs_names"));
    // The index outranks the other unique column, but does not hide it.
    assert_eq!(
        d.unique_columns,
        vec!["obs_names".to_string(), "cell_uid".to_string()],
        "the obs index ranks first: it is the identity a tool hands back"
    );
}

#[test]
fn obs_names_joins_and_records_the_physical_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture_with_obs(dir.path(), "a.scx", obs_with_pyarrow_index(), 2, 1);
    let data = score_data(vec!["AAAC".into(), "AAAT".into()], |i| i as f32 + 1.0);

    let s = attach_external_obs(
        &path,
        &data,
        &AttachObsOptions {
            join_key: ObsJoinKey::Column("obs_names".into()),
            ..opts()
        },
    )
    .unwrap();

    // Anything the diagnosis names must be a name the join accepts — the same
    // invariant that made an Int64 `soma_joinid` suggestion usable.
    assert_eq!(s.n_matched, 2);
    // ...but the summary keeps the PHYSICAL name, because it is what the
    // provenance entry records about the file on disk.
    assert_eq!(s.obs_key_column, "__index_level_0__");
    assert_eq!(display_key_name("obs", &s.obs_key_column), "obs_names");
}

#[test]
fn the_physical_index_name_is_still_accepted_as_a_key() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture_with_obs(dir.path(), "a.scx", obs_with_pyarrow_index(), 2, 1);
    let data = score_data(vec!["AAAC".into()], |_| 1.0);

    let s = attach_external_obs(
        &path,
        &data,
        &AttachObsOptions {
            join_key: ObsJoinKey::Column("__index_level_0__".into()),
            ..opts()
        },
    )
    .unwrap();
    assert_eq!(s.n_matched, 1, "the old spelling must keep working");
}

#[test]
fn a_failed_join_names_the_index_as_obs_names() {
    // The report called out "every join-failure message", not just the
    // diagnosis: a zero-overlap error quotes the key it joined on, and quoting
    // `__index_level_0__` there sends the reader to a column they cannot look at.
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture_with_obs(dir.path(), "a.scx", obs_with_pyarrow_index(), 2, 1);
    let data = score_data(vec!["nope-1".into(), "nope-2".into()], |i| i as f32);

    let err = attach_external_obs(
        &path,
        &data,
        &AttachObsOptions {
            join_key: ObsJoinKey::Column("obs_names".into()),
            ..opts()
        },
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(matches!(err, OpsError::AxisMismatch { .. }), "{msg}");
    assert!(msg.contains("on 'obs_names'"), "{msg}");
    assert!(!msg.contains("__index_level_0__"), "{msg}");
}

#[test]
fn a_duplicate_key_error_names_the_index_as_obs_names() {
    let dir = tempfile::tempdir().unwrap();
    // Same shape as a merged atlas: the pyarrow index repeats.
    let obs = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("cell_uid", DataType::Utf8, false),
            Field::new("__index_level_0__", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["u0", "u1", "u2", "u3"])),
            Arc::new(StringArray::from(vec!["DUP", "DUP", "DUP", "DUP"])),
        ],
    )
    .unwrap();
    let path = write_fixture_with_obs(dir.path(), "a.scx", obs, 2, 1);
    let data = score_data(vec!["DUP".into()], |_| 1.0);

    let err = attach_external_obs(
        &path,
        &data,
        &AttachObsOptions {
            join_key: ObsJoinKey::Column("obs_names".into()),
            ..opts()
        },
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(matches!(err, OpsError::DuplicateJoinKey { .. }), "{msg}");
    assert!(msg.contains("'obs_names'"), "{msg}");
    assert!(!msg.contains("__index_level_0__"), "{msg}");
    assert!(
        msg.contains("cell_uid"),
        "and it still names the column that would work: {msg}"
    );
}

#[test]
fn a_real_column_named_index_wins_over_the_alias() {
    // `obs_census_shaped` has a literal `index` column holding "0","1","0","1".
    // Treating `index` as an alias for the axis index would silently change
    // which column a caller keyed on.
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture_with_obs(dir.path(), "a.scx", obs_census_shaped(), 2, 1);

    let d = diagnose_obs_key(&path, Some(&ObsJoinKey::Column("index".into()))).unwrap();
    assert_eq!(d.resolved_key.as_deref(), Some("index"));
    assert_eq!(
        d.resolved_cardinality,
        Some(2),
        "the literal column, whose values repeat — not some resolved index"
    );
}

#[test]
fn a_composite_component_can_be_the_obs_index_alias() {
    let batch = obs_with_pyarrow_index();
    let aliased = build_composite_key(&batch, &["donor".into(), "obs_names".into()]).unwrap();
    let physical =
        build_composite_key(&batch, &["donor".into(), "__index_level_0__".into()]).unwrap();
    assert_eq!(
        aliased, physical,
        "a composite component resolves the alias exactly as a single key does"
    );
}

// ---------------------------------------------------------------------------
// A suggested key must be one the join accepts (F2)
// ---------------------------------------------------------------------------

/// The shape a file takes after two `doublet_import`s: the obs index repeats
/// (a merged atlas), and the only per-row-unique columns are float scores.
fn obs_with_unique_float_scores() -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("scrublet_score", DataType::Float32, false),
        Field::new("scdblfinder_score", DataType::Float32, false),
        Field::new("__index_level_0__", DataType::Utf8, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float32Array::from(vec![0.11, 0.22, 0.33, 0.44])),
            Arc::new(Float32Array::from(vec![0.51, 0.62, 0.73, 0.84])),
            Arc::new(StringArray::from(vec!["DUP", "DUP", "DUP", "DUP"])),
        ],
    )
    .unwrap()
}

#[test]
fn a_unique_float_column_is_never_suggested_as_a_key() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture_with_obs(dir.path(), "a.scx", obs_with_unique_float_scores(), 2, 1);

    let d = diagnose_obs_key(&path, None).unwrap();
    assert!(
        d.unique_columns.is_empty(),
        "every unique column here is a float, and `resolve_key_column` refuses \
         floats — offering one is a guaranteed dead end: {:?}",
        d.unique_columns
    );
    assert_ne!(d.suggestion.as_deref(), Some("scrublet_score"));
    assert!(
        d.unique_pairs.is_empty(),
        "a composite of two refused columns is equally unusable: {:?}",
        d.unique_pairs
    );
    // Set aside, not silently dropped: "nothing is unique" would be false here.
    assert_eq!(
        d.unusable_unique_columns,
        vec![
            "scrublet_score".to_string(),
            "scdblfinder_score".to_string()
        ]
    );
    let summary = d.describe();
    assert!(
        summary.contains("scrublet_score") && summary.contains("floats are refused"),
        "the summary must name what was set aside and why: {summary}"
    );
}

#[test]
fn diagnose_obs_key_finds_a_unique_pair_when_no_single_column_is_unique() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture_with_obs(dir.path(), "a.scx", obs_two_libraries(), 2, 1);

    let d = diagnose_obs_key(&path, None).unwrap();
    assert!(
        d.unique_columns.is_empty(),
        "neither column alone is unique: {:?}",
        d.unique_columns
    );
    assert!(
        d.unique_pairs
            .contains(&("barcode".to_string(), "sample_id".to_string()))
            || d.unique_pairs
                .contains(&("sample_id".to_string(), "barcode".to_string())),
        "{:?}",
        d.unique_pairs
    );
    let sug = d.suggestion.unwrap();
    assert!(
        sug.contains("sample_id") && sug.contains("barcode"),
        "{sug}"
    );
}

// ---------------------------------------------------------------------------
// Overwrite semantics (Delta 3)
// ---------------------------------------------------------------------------

#[test]
fn reimport_without_overwrite_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 2, 1);
    let data = score_data(keys("cell_", 4), |i| i as f32);

    attach_external_obs(&path, &data, &opts()).unwrap();
    let err = attach_external_obs(&path, &data, &opts()).unwrap_err();
    assert!(err.to_string().contains("already exists"), "{err}");
    assert!(
        err.to_string().contains("REPLACES rather than merges"),
        "the error must warn that overwrite is not a merge: {err}"
    );
}

#[test]
fn reimport_with_overwrite_replaces_and_does_not_merge() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 2, 1);

    // Batch A covers rows 0-1.
    let a = score_data(vec!["cell_0".into(), "cell_1".into()], |i| i as f32 + 1.0);
    attach_external_obs(&path, &a, &opts()).unwrap();

    // Batch B covers rows 2-3, imported with overwrite.
    let b = score_data(vec!["cell_2".into(), "cell_3".into()], |i| i as f32 + 10.0);
    attach_external_obs(
        &path,
        &b,
        &AttachObsOptions {
            overwrite: true,
            ..opts()
        },
    )
    .unwrap();

    let obs = ScxReader::open(&path).unwrap().read_obs().unwrap();
    let scores = f32_col(&obs, "dbl_score");
    // Batch A's values are GONE. This is the documented, deliberate behaviour:
    // concatenate the per-batch tables and import once instead.
    assert!(
        scores.is_null(0) && scores.is_null(1),
        "overwrite must replace, not merge — if this starts passing values \
         through, the concatenate-first contract has silently changed"
    );
    assert_eq!(scores.value(2), 10.0);
    assert_eq!(scores.value(3), 11.0);

    // Exactly one column of each name, not two.
    let schema = obs.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names.iter().filter(|n| **n == "dbl_score").count(), 1);
    assert_eq!(names.iter().filter(|n| **n == "dbl_status").count(), 1);
}

// ---------------------------------------------------------------------------
// In-place invariants
// ---------------------------------------------------------------------------

#[test]
fn deletion_vectors_and_the_provenance_chain_survive() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 6, 2, 2);
    crate::mark_deleted(&path, &[1, 3]).unwrap();

    let data = score_data(keys("cell_", 6), |i| i as f32);
    attach_external_obs(&path, &data, &opts()).unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let keep = reader.deletion_keep_mask().unwrap().expect("DV survives");
    assert!(!keep[1] && !keep[3] && keep[0]);

    let prov = reader.read_provenance().unwrap();
    let actions: Vec<&str> = prov.operations.iter().map(|o| o.action.as_str()).collect();
    assert_eq!(
        actions,
        vec!["create", "delete", "test_obs_import"],
        "the chain must be appended, not replaced"
    );
}

#[test]
fn var_and_x_sections_are_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 3, 2);

    let before: Vec<(String, u64, u64)> = ScxReader::open(&path)
        .unwrap()
        .catalog()
        .entries
        .iter()
        .filter(|e| {
            matches!(
                e.section_type,
                SectionType::VarMetadata | SectionType::VarMetadataShard | SectionType::CsrShard
            )
        })
        .map(|e| (e.name.clone(), e.offset, e.length))
        .collect();
    assert!(!before.is_empty());

    let data = score_data(keys("cell_", 4), |i| i as f32);
    attach_external_obs(&path, &data, &opts()).unwrap();

    let after: Vec<(String, u64, u64)> = ScxReader::open(&path)
        .unwrap()
        .catalog()
        .entries
        .iter()
        .filter(|e| {
            matches!(
                e.section_type,
                SectionType::VarMetadata | SectionType::VarMetadataShard | SectionType::CsrShard
            )
        })
        .map(|e| (e.name.clone(), e.offset, e.length))
        .collect();
    assert_eq!(
        before, after,
        "an obs-only attach must not rewrite or move var / X sections"
    );
}

#[test]
fn csc_sidecar_and_data_generation_are_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 2, 1);
    let cat = ScxReader::open(&path).unwrap().catalog().clone();
    let (before_data, before_csc) = (cat.data_generation, cat.csc_build_generation);

    let data = score_data(keys("cell_", 4), |i| i as f32);
    attach_external_obs(&path, &data, &opts()).unwrap();

    let cat = ScxReader::open(&path).unwrap().catalog().clone();
    assert_eq!(
        (cat.data_generation, cat.csc_build_generation),
        (before_data, before_csc),
        "bumping either would silently invalidate the CSC sidecar"
    );
}

#[test]
fn rollback_restores_the_pre_import_state() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 2, 1);
    let data = score_data(keys("cell_", 4), |i| i as f32);

    attach_external_obs(&path, &data, &opts()).unwrap();
    assert!(ScxReader::open(&path)
        .unwrap()
        .read_obs()
        .unwrap()
        .column_by_name("dbl_score")
        .is_some());

    crate::rollback(&path).unwrap();

    let obs = ScxReader::open(&path).unwrap().read_obs().unwrap();
    assert!(obs.column_by_name("dbl_score").is_none());
    assert!(obs.column_by_name("dbl_status").is_none());
    assert!(obs.column_by_name("barcode").is_some());
}

#[test]
fn provenance_records_the_source_checksum_and_match_counts() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 3, 2, 1);
    let mut data = score_data(keys("cell_", 3), |i| i as f32);
    data.source_checksum = Some([9u8; 32]);

    attach_external_obs(&path, &data, &opts()).unwrap();

    let prov = ScxReader::open(&path).unwrap().read_provenance().unwrap();
    let last = prov.operations.last().unwrap();
    assert_eq!(last.input_checksums, vec![[9u8; 32]]);
    assert!(
        last.params_json.contains("\"n_matched\":3"),
        "{}",
        last.params_json
    );
    assert!(
        last.params_json.contains("\"source_file\":\"calls.csv\""),
        "{}",
        last.params_json
    );
}

#[test]
fn a_rejected_import_leaves_the_file_byte_identical() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 2, 1);
    let before = std::fs::read(&path).unwrap();

    // Zero overlap: rejected after the join, before any write.
    let data = score_data(keys("nope_", 4), |i| i as f32);
    assert!(attach_external_obs(&path, &data, &opts()).is_err());

    assert_eq!(
        before,
        std::fs::read(&path).unwrap(),
        "a rejected import must not write a byte"
    );
}

#[test]
fn dry_run_writes_nothing_and_reports_the_real_match_count() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 6, 2, 1);
    let before = std::fs::read(&path).unwrap();

    let data = score_data(keys("cell_", 4), |i| i as f32);
    let s = attach_external_obs(
        &path,
        &data,
        &AttachObsOptions {
            dry_run: true,
            ..opts()
        },
    )
    .unwrap();

    assert_eq!(s.n_matched, 4);
    assert_eq!(s.n_target_rows_absent, 2);
    assert_eq!(
        s.obs_columns_added,
        vec!["dbl_status", "dbl_score", "dbl_call"]
    );
    assert_eq!(before, std::fs::read(&path).unwrap(), "dry run wrote bytes");
}

#[test]
fn nothing_to_attach_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 3, 2, 1);
    let empty = RecordBatch::new_empty(Arc::new(Schema::empty()));
    let data = ExternalObsData {
        row_keys: Vec::new(),
        row_annotations: empty,
        row_embeddings: Vec::new(),
        uns: None,
        source_checksum: None,
        source_name: None,
    };
    let err = attach_external_obs(
        &path,
        &data,
        &AttachObsOptions {
            status_column: None,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("nothing to attach"), "{err}");
}

#[test]
fn a_row_count_mismatch_is_caught_before_any_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 3, 2, 1);
    let before = std::fs::read(&path).unwrap();

    let mut data = score_data(keys("cell_", 3), |i| i as f32);
    data.row_keys.push("cell_extra".to_string()); // now 4 keys, 3 annotation rows

    let err = attach_external_obs(&path, &data, &opts()).unwrap_err();
    assert!(matches!(err, OpsError::ShapeMismatch { .. }), "{err}");
    assert_eq!(before, std::fs::read(&path).unwrap());
}

// ---------------------------------------------------------------------------
// Predicate index
// ---------------------------------------------------------------------------

/// A file whose obs carries a real predicate index over `cell_type` **and**
/// `n_counts`, plus the per-shard catalog `column_stats` that index implies.
///
/// Built at creation time rather than bolted on afterwards, so the index is a
/// genuine `build_obs_predicate_index_bytes` product rather than a stub.
///
/// The `apply_obs_shard_column_stats` call is load-bearing. Without it the
/// fixture has an index section but no column stats, and every assertion below
/// about pushdown surviving or being invalidated would be about the section
/// alone — which is exactly how a whole class of staleness bug stayed invisible:
/// Level-1 pruning reads the *stats*, and for the numeric `MinMax` arm never
/// consults the index at all.
///
/// Two CSR shards, so pruning is observable: rows 0..2 hold `n_counts` 10/20,
/// rows 2..4 hold 30/40.
fn fixture_with_obs_index(dir: &Path, name: &str) -> PathBuf {
    fixture_with_obs_stats(dir, name, true)
}

/// `write_index = false` yields a file with per-shard column stats and **no**
/// `ObsPredicateIndex` section. Not a contrived state: it is exactly what a
/// `modify_metadata` obs replace used to leave behind, and the reason the stats
/// clear cannot be gated on `obs_index_would_go_stale` (which short-circuits to
/// `false` when there is no index to read).
fn fixture_with_obs_stats(dir: &Path, name: &str, write_index: bool) -> PathBuf {
    let n_obs = 4usize;
    let n_vars = 2usize;
    let schema = Schema::new(vec![
        Field::new("barcode", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, false),
        Field::new("n_counts", DataType::Int64, false),
    ]);
    let obs = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                (0..n_obs).map(|i| format!("cell_{i}")).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(vec!["T", "B", "T", "B"])),
            Arc::new(Int64Array::from(vec![10i64, 20, 30, 40])),
        ],
    )
    .unwrap();

    let path = dir.join(name);
    let header = FileHeader::new_single_modality(n_obs as u64, n_vars as u64, 0, 2, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&obs).unwrap();
    writer.write_var(&var_batch(n_vars)).unwrap();
    let ranges: Vec<(u64, u64)> = vec![(0, 2), (2, 4)];
    for (start, _) in &ranges {
        let indptr: Vec<u64> = vec![0u64; 3];
        writer
            .write_csr_shard(
                &indptr,
                &[],
                &[],
                CodecId::None,
                ValueEncoding::Uint8,
                *start,
            )
            .unwrap();
    }

    let opts = scx_engine::PredicateIndexBuildOptions {
        forced_columns: vec!["cell_type".to_string(), "n_counts".to_string()],
        preset_columns: Vec::new(),
        auto_threshold: 1000,
        high_cardinality_threshold: 100_000,
    };
    let mut outcomes = Vec::new();
    let mut named = Vec::new();
    let bytes = scx_engine::build_obs_predicate_index_bytes(
        &obs,
        &ranges,
        &opts,
        &mut outcomes,
        &mut named,
    )
    .unwrap()
    .expect("cell_type must be indexable");
    assert_eq!(named, vec!["cell_type".to_string(), "n_counts".to_string()]);
    if write_index {
        writer.write_obs_predicate_index(&bytes).unwrap();
    }
    scx_engine::apply_obs_shard_column_stats(&mut writer, &bytes, ranges.len()).unwrap();

    writer
        .write_uns(&serde_json::json!({"state": "v0"}))
        .unwrap();
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1_710_000_000,
            action: "create".to_string(),
            tool: "test".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();
    writer.finish().unwrap();
    path
}

#[test]
fn predicate_index_survives_a_pure_add() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture_with_obs_index(dir.path(), "a.scx");
    assert!(has_obs_index(&path), "fixture must start with an index");

    let data = score_data(keys("cell_", 4), |i| i as f32);
    let s = attach_external_obs(&path, &data, &opts()).unwrap();

    assert!(!s.obs_index_dropped);
    assert!(
        has_obs_index(&path),
        "adding new columns cannot invalidate an index keyed on other columns — \
         dropping it would silently kill query pushdown"
    );
}

#[test]
fn predicate_index_is_dropped_when_an_indexed_column_is_overwritten() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture_with_obs_index(dir.path(), "a.scx");
    assert!(has_obs_index(&path));

    // Overwrite `cell_type` itself — the column the index covers.
    let schema = Schema::new(vec![Field::new("cell_type", DataType::Utf8, true)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(vec!["NK", "NK", "NK", "NK"]))],
    )
    .unwrap();
    let data = ExternalObsData {
        row_keys: keys("cell_", 4),
        row_annotations: batch,
        row_embeddings: Vec::new(),
        uns: None,
        source_checksum: None,
        source_name: None,
    };

    let s = attach_external_obs(
        &path,
        &data,
        &AttachObsOptions {
            overwrite: true,
            status_column: None,
            ..opts()
        },
    )
    .unwrap();

    assert!(s.obs_index_dropped);
    assert!(
        !has_obs_index(&path),
        "keeping the index here would leave pushdown matching 'T'/'B' rows that \
         no longer exist"
    );
}

fn has_obs_index(path: &Path) -> bool {
    ScxReader::open(path)
        .unwrap()
        .catalog()
        .entries
        .iter()
        .any(|e| e.section_type == SectionType::ObsPredicateIndex)
}

use crate::test_utils::columns_with_shard_stats;

/// Overwriting a column the catalog has stats for must drop **that column's**
/// stats, whether or not an index section happens to still be present.
///
/// Level-1 pruning reads the stats directly — for `MinMax` it never looks at the
/// index — so keeping them would leave shards excluded on bounds describing
/// values the import just replaced.
#[test]
fn overwriting_an_indexed_column_clears_its_shard_column_stats() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture_with_obs_index(dir.path(), "a.scx");
    assert_eq!(
        columns_with_shard_stats(&path, &["cell_type", "n_counts"]),
        vec!["cell_type".to_string(), "n_counts".to_string()],
        "fixture must start with stats for both indexed columns"
    );

    // Replace `n_counts` with values an order of magnitude larger.
    let schema = Schema::new(vec![Field::new("n_counts", DataType::Int64, true)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(Int64Array::from(vec![100i64, 200, 300, 400]))],
    )
    .unwrap();
    let data = ExternalObsData {
        row_keys: keys("cell_", 4),
        row_annotations: batch,
        row_embeddings: Vec::new(),
        uns: None,
        source_checksum: None,
        source_name: None,
    };
    crate::attach_external_obs(
        &path,
        &data,
        &AttachObsOptions {
            overwrite: true,
            status_column: None,
            ..opts()
        },
    )
    .unwrap();

    assert_eq!(
        columns_with_shard_stats(&path, &["cell_type", "n_counts"]),
        vec!["cell_type".to_string()],
        "the overwritten column's stats must go — and ONLY that column's. \
         Clearing wholesale would silently disable Level-1 pruning on a \
         cell_type index every time someone lands doublet calls."
    );

    // The rows are actually reachable again.
    let n = scx_engine::QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("n_counts > 150")
        .unwrap()
        .count()
        .unwrap()
        .matched_rows;
    assert_eq!(
        n, 3,
        "200/300/400 match; stale bounds would exclude all four"
    );
}

/// A pure *add* invalidates nothing, so every column keeps its stats. This is
/// the case that makes the scoped clear worth having.
#[test]
fn adding_a_new_column_keeps_every_shard_column_stat() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture_with_obs_index(dir.path(), "a.scx");

    let data = score_data(keys("cell_", 4), |i| i as f32);
    crate::attach_external_obs(&path, &data, &opts()).unwrap();

    assert_eq!(
        columns_with_shard_stats(&path, &["cell_type", "n_counts"]),
        vec!["cell_type".to_string(), "n_counts".to_string()],
        "adding a column cannot invalidate stats for columns it did not touch"
    );
}

/// The compounding case. `obs_index_would_go_stale` returns `false` early when
/// there is no index section — but a file can carry stats with no index (that is
/// precisely what a `modify_metadata` obs replace used to leave behind). Gating
/// the stats clear on that flag would let the stale bounds survive a second
/// rewrite, so the clear is unconditional.
#[test]
fn overwriting_clears_stats_even_when_the_index_section_is_already_gone() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture_with_obs_stats(dir.path(), "a.scx", false);
    assert!(!has_obs_index(&path), "precondition: no index section");
    assert!(
        columns_with_shard_stats(&path, &["n_counts"]).contains(&"n_counts".to_string()),
        "precondition: stats outlived the index section"
    );

    let schema = Schema::new(vec![Field::new("n_counts", DataType::Int64, true)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(Int64Array::from(vec![100i64, 200, 300, 400]))],
    )
    .unwrap();
    let data = ExternalObsData {
        row_keys: keys("cell_", 4),
        row_annotations: batch,
        row_embeddings: Vec::new(),
        uns: None,
        source_checksum: None,
        source_name: None,
    };
    let s = crate::attach_external_obs(
        &path,
        &data,
        &AttachObsOptions {
            overwrite: true,
            status_column: None,
            ..opts()
        },
    )
    .unwrap();
    assert!(
        !s.obs_index_dropped,
        "there was no index to drop — the stats clear must not depend on this"
    );

    assert!(
        columns_with_shard_stats(&path, &["n_counts"]).is_empty(),
        "stats that outlived their index must still be cleared on overwrite"
    );
}

// ---------------------------------------------------------------------------
// obsm and uns
// ---------------------------------------------------------------------------

#[test]
fn obsm_embedding_is_written_and_scattered_by_the_join() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 2, 1);

    let emb_schema = Schema::new(vec![
        Field::new("z0", DataType::Float32, true),
        Field::new("z1", DataType::Float32, true),
    ]);
    let emb = RecordBatch::try_new(
        Arc::new(emb_schema),
        vec![
            Arc::new(Float32Array::from(vec![1.0f32, 2.0])),
            Arc::new(Float32Array::from(vec![10.0f32, 20.0])),
        ],
    )
    .unwrap();

    let mut data = score_data(vec!["cell_0".into(), "cell_1".into()], |i| i as f32);
    data.row_embeddings = vec![("X_dbl".to_string(), emb)];

    let s = attach_external_obs(&path, &data, &opts()).unwrap();
    assert_eq!(s.obsm_keys_added, vec!["X_dbl".to_string()]);

    let reader = ScxReader::open(&path).unwrap();
    assert!(
        reader.header().has_obsm(),
        "the obsm header flag must be set"
    );
    let all = reader.read_all_obsm().unwrap();
    let (_, batch) = all.iter().find(|(k, _)| k.as_str() == "X_dbl").unwrap();
    assert_eq!(batch.num_rows(), 4);
    let z0 = f32_col(batch, "z0");
    assert_eq!(z0.value(0), 1.0);
    assert_eq!(z0.value(1), 2.0);
    assert!(z0.is_null(2) && z0.is_null(3));
}

#[test]
fn uns_is_merged_only_when_a_key_is_given_and_is_otherwise_left_alone() {
    let dir = tempfile::tempdir().unwrap();

    // No uns_key: the existing uns section is not even rewritten.
    let path = write_fixture(dir.path(), "a.scx", 3, 2, 1);
    let data = score_data(keys("cell_", 3), |i| i as f32);
    attach_external_obs(&path, &data, &opts()).unwrap();
    let uns = ScxReader::open(&path).unwrap().read_uns().unwrap();
    assert_eq!(uns["state"], "v0");

    // With a key: merged alongside the existing content.
    let path = write_fixture(dir.path(), "b.scx", 3, 2, 1);
    let mut data = score_data(keys("cell_", 3), |i| i as f32);
    data.uns = Some(serde_json::json!({"tool": "scDblFinder"}));
    attach_external_obs(
        &path,
        &data,
        &AttachObsOptions {
            uns_key: Some("doublet".into()),
            ..opts()
        },
    )
    .unwrap();
    let uns = ScxReader::open(&path).unwrap().read_uns().unwrap();
    assert_eq!(uns["state"], "v0", "the pre-existing uns must survive");
    assert_eq!(uns["doublet"]["tool"], "scDblFinder");
}

// ---------------------------------------------------------------------------
// Multimodal
// ---------------------------------------------------------------------------

#[test]
fn multimodal_file_at_modality_zero_succeeds_and_nonzero_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = crate::test_utils::fixture_multimodal(&dir);

    let n_obs = ScxReader::open(&path).unwrap().n_obs() as usize;
    let obs = ScxReader::open(&path).unwrap().read_obs().unwrap();
    let key_col = super::resolve_key_column("obs", &obs, None).unwrap();
    let row_keys = super::string_column(&obs, &key_col).unwrap();
    assert_eq!(row_keys.len(), n_obs);

    // obs is the shared global axis, so modality 0 is the right place to write.
    let data = score_data(row_keys, |i| i as f32);
    let s = attach_external_obs(&path, &data, &opts()).unwrap();
    assert_eq!(s.n_matched, n_obs as u64);

    // Both modalities still read back.
    let reader = ScxReader::open(&path).unwrap();
    assert!(reader.modality_table().is_some());
    assert!(reader
        .read_obs()
        .unwrap()
        .column_by_name("dbl_score")
        .is_some());

    // Per-modality obs annotation is a different feature and is refused.
    let err = attach_external_obs(
        &path,
        &score_data(keys("cell_", 1), |i| i as f32),
        &AttachObsOptions {
            modality_id: 1,
            overwrite: true,
            ..opts()
        },
    )
    .unwrap_err();
    assert!(
        matches!(err, OpsError::MultimodalUnsupported { .. }),
        "{err}"
    );
}

// ---------------------------------------------------------------------------
// Real-atlas acceptance (opt-in)
// ---------------------------------------------------------------------------

/// The Phase-0 failure mode, against a real merged atlas rather than a fixture.
///
/// Ignored by default and pointed at a file via `SCX_TEST_ATLAS`, because it
/// needs a multi-hundred-MB CELLxGENE-derived file that cannot live in the repo:
///
/// ```text
/// SCX_TEST_ATLAS=/path/to/census_1m.scx \
///   cargo test -p scx-ops --lib diagnoses_a_real_atlas -- --ignored --nocapture
/// ```
///
/// What it pins: on such a file there is no barcode column, the obs index is a
/// stringified `RangeIndex` reused across chunks, and the only unique column is
/// one no fallback list would guess. If `diagnose_obs_key` cannot find it, the
/// importer is a dead end on exactly the files this feature exists for.
#[test]
#[ignore]
fn diagnoses_a_real_atlas_whose_index_is_not_a_key() {
    let Ok(path) = std::env::var("SCX_TEST_ATLAS") else {
        eprintln!("set SCX_TEST_ATLAS to run this");
        return;
    };
    let path = PathBuf::from(path);

    let d = diagnose_obs_key(&path, Some(&ObsJoinKey::Auto)).unwrap();
    eprintln!("n_obs                 = {}", d.n_obs);
    eprintln!("resolved_key          = {:?}", d.resolved_key);
    eprintln!("resolved_cardinality  = {:?}", d.resolved_cardinality);
    eprintln!("unique_columns        = {:?}", d.unique_columns);
    eprintln!("unique_pairs          = {:?}", d.unique_pairs);
    eprintln!("suggestion            = {:?}", d.suggestion);
    eprintln!("describe()            = {}", d.describe());

    assert!(d.n_obs > 0);
    assert!(
        !d.unique_columns.is_empty() || !d.unique_pairs.is_empty(),
        "no usable join key found on {} — the importer would be a dead end here",
        path.display()
    );
    assert!(d.suggestion.is_some());
}

// ---------------------------------------------------------------------------
// E2 — a partial-coverage import must not read like a failed join
// ---------------------------------------------------------------------------

/// Panics if the closure is called, so a test can prove the coverage branch
/// never pays for (or prints) the key examples.
fn examples_must_not_be_needed() -> (Vec<String>, Vec<String>) {
    panic!("the coverage branch must not extract key examples");
}

fn some_examples() -> (Vec<String>, Vec<String>) {
    (
        vec!["0".to_string(), "1".to_string()],
        vec!["374584".to_string(), "374585".to_string()],
    )
}

/// The report's exact case: the documented per-batch workflow imported 6 of 116
/// `dataset_id` batches — 186,649 of 500,000 target rows — with every source row
/// matched. That is a perfect join over a deliberate subset, and it was reported
/// with the word "only" plus a target/source example pair, which is the standard
/// shape of a key-format-mismatch diagnostic.
#[test]
fn partial_coverage_reports_coverage_at_info_without_examples() {
    let (level, msg) = obs_join_coverage_report(
        186_649,
        500_000,
        313_351,
        0, // every source row matched
        "obs_names",
        examples_must_not_be_needed,
    )
    .expect("below half, so something must be reported");

    assert_eq!(
        level,
        log::Level::Info,
        "a deliberate partial import is information, not a warning"
    );
    assert!(
        !msg.contains("only"),
        "\"only\" frames a correct join as a shortfall: {msg}"
    );
    assert!(
        !msg.contains("examples"),
        "both example lists are valid keys from the same space; showing them \
         side by side reads as evidence of divergence: {msg}"
    );
    // What it must say instead: the arithmetic, and that nothing is wrong.
    for needle in ["coverage", "186649", "500000", "313351", "left null"] {
        assert!(msg.contains(needle), "must mention {needle:?}: {msg}");
    }
    assert!(
        msg.contains("not a key mismatch"),
        "must say outright that this is not the failure it resembles: {msg}"
    );
}

/// A genuine mismatch keeps every diagnostic it had: the "only" framing, the
/// example pair, and the hard-won prefix / `-1`-suffix hint.
#[test]
fn unmatched_source_rows_still_warn_with_examples() {
    let (level, msg) = obs_join_coverage_report(10, 100, 90, 40, "barcode", some_examples)
        .expect("below half, so something must be reported");

    assert_eq!(level, log::Level::Warn);
    assert!(msg.contains("only"), "{msg}");
    assert!(
        msg.contains("40 source rows matched no target row"),
        "{msg}"
    );
    assert!(
        msg.contains("-1") && msg.contains("prefix"),
        "the domain hint must survive: {msg}"
    );
    assert!(msg.contains("374584"), "source examples must appear: {msg}");
    assert!(msg.contains("barcode"), "the key name must appear: {msg}");
}

/// Above half, nothing is reported at all — the pre-existing threshold, kept so
/// this change alters the *shape* of the message and not how often it appears.
#[test]
fn high_coverage_reports_nothing() {
    assert!(
        obs_join_coverage_report(50, 100, 50, 0, "obs_names", examples_must_not_be_needed)
            .is_none(),
        "exactly half is not below half"
    );
    assert!(
        obs_join_coverage_report(99, 100, 1, 20, "obs_names", examples_must_not_be_needed)
            .is_none(),
        "high coverage stays silent even with unmatched source rows"
    );
}

/// The discriminator is `n_source_absent`, not the match fraction: two joins with
/// identical coverage must be reported differently based only on whether source
/// rows failed to land.
#[test]
fn identical_coverage_splits_on_unmatched_source_rows_alone() {
    let coverage =
        obs_join_coverage_report(10, 100, 90, 0, "k", examples_must_not_be_needed).unwrap();
    let mismatch = obs_join_coverage_report(10, 100, 90, 5, "k", some_examples).unwrap();
    assert_eq!(coverage.0, log::Level::Info);
    assert_eq!(mismatch.0, log::Level::Warn);
    assert_ne!(
        coverage.1, mismatch.1,
        "same coverage, different cause — the messages must differ"
    );
}
