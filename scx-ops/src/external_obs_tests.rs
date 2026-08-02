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

/// A file whose obs carries a real predicate index over `cell_type`.
///
/// Built at creation time rather than bolted on afterwards, so the index is a
/// genuine `build_obs_predicate_index_bytes` product rather than a stub.
fn fixture_with_obs_index(dir: &Path, name: &str) -> PathBuf {
    let n_obs = 4usize;
    let n_vars = 2usize;
    let schema = Schema::new(vec![
        Field::new("barcode", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, false),
    ]);
    let obs = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                (0..n_obs).map(|i| format!("cell_{i}")).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(vec!["T", "B", "T", "B"])),
        ],
    )
    .unwrap();

    let path = dir.join(name);
    let header =
        FileHeader::new_single_modality(n_obs as u64, n_vars as u64, 0, n_obs as u32, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&obs).unwrap();
    writer.write_var(&var_batch(n_vars)).unwrap();
    let indptr: Vec<u64> = vec![0u64; n_obs + 1];
    writer
        .write_csr_shard(&indptr, &[], &[], CodecId::None, ValueEncoding::Uint8, 0)
        .unwrap();

    let opts = scx_engine::PredicateIndexBuildOptions {
        forced_columns: vec!["cell_type".to_string()],
        preset_columns: Vec::new(),
        auto_threshold: 1000,
        high_cardinality_threshold: 100_000,
    };
    let mut outcomes = Vec::new();
    let mut named = Vec::new();
    let bytes = scx_engine::build_obs_predicate_index_bytes(
        &obs,
        &[(0u64, n_obs as u64)],
        &opts,
        &mut outcomes,
        &mut named,
    )
    .unwrap()
    .expect("cell_type must be indexable");
    assert_eq!(named, vec!["cell_type".to_string()]);
    writer.write_obs_predicate_index(&bytes).unwrap();

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
