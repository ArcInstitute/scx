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

/// How a fixture lays out its obs axis on disk.
///
/// The two layouts are not interchangeable at read time: a `Single` section is
/// one Arrow IPC batch with no per-shard reader, so the op has nothing to stream
/// and falls back to materialising. Every fixture in this file was `Single`
/// before the streaming rewrite, which meant the streaming path had no coverage
/// at all — so the core behaviours are now run over both.
#[derive(Clone, Copy, Debug)]
enum ObsLayout {
    /// One legacy `ObsMetadata` section.
    Single,
    /// `ObsMetadataShard` sections with exactly these per-shard row counts.
    ///
    /// Spelled out rather than derived from a shard size on purpose: a test
    /// needs to be able to make the obs shard boundaries *disagree* with
    /// `header.shard_target_rows`, which is what tells a rewrite that preserves
    /// the input layout apart from one that re-derives it. Unequal counts are
    /// also what catch a driver that computes each shard's global row start as
    /// `shard_idx * shard_target_rows` instead of a running cursor.
    Shards(&'static [usize]),
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
    write_fixture_with_layout(dir, name, obs, n_vars, n_shards, ObsLayout::Single)
}

fn write_fixture_with_layout(
    dir: &Path,
    name: &str,
    obs: RecordBatch,
    n_vars: usize,
    n_shards: usize,
    obs_layout: ObsLayout,
) -> PathBuf {
    let n_obs = obs.num_rows();
    let path = dir.join(name);
    let rows_per = n_obs.div_ceil(n_shards);
    let header =
        FileHeader::new_single_modality(n_obs as u64, n_vars as u64, 0, rows_per as u32, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    match obs_layout {
        ObsLayout::Single => writer.write_obs(&obs).unwrap(),
        ObsLayout::Shards(rows) => {
            assert_eq!(
                rows.iter().sum::<usize>(),
                n_obs,
                "obs shard row counts must tile the obs axis"
            );
            let mut start = 0usize;
            for (idx, take) in rows.iter().enumerate() {
                writer
                    .write_obs_shard(
                        idx as u32,
                        start as u64,
                        *take as u64,
                        n_obs as u64,
                        &obs.slice(start, *take),
                    )
                    .unwrap();
                start += take;
            }
        }
    }
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
        uns: serde_json::Map::new(),
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
        uns: serde_json::Map::new(),
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

/// A composite naming a column the target does not have must say so in the
/// caller's own spelling, and must say it on both obs layouts.
///
/// Single-column keys get this from `resolve_key_column`; a composite has no
/// equivalent, and on the streaming path the projected key read would otherwise
/// fail first with a bare "obs column not found" that neither lists the columns
/// that *are* present nor quotes what the user typed.
#[test]
fn a_composite_naming_a_missing_column_names_it_and_lists_the_alternatives() {
    let dir = tempfile::tempdir().unwrap();
    for (name, layout) in [
        ("single.scx", ObsLayout::Single),
        ("sharded.scx", ObsLayout::Shards(&[2, 2])),
    ] {
        let path = write_fixture_with_layout(dir.path(), name, obs_two_libraries(), 2, 1, layout);
        let data = score_data(vec!["x".into(), "y".into()], |i| i as f32);
        let err = attach_external_obs(
            &path,
            &data,
            &AttachObsOptions {
                join_key: ObsJoinKey::Composite {
                    columns: vec!["sample_id".into(), "no_such_column".into()],
                },
                ..opts()
            },
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("no_such_column") && msg.contains("barcode"),
            "on {name}, the error must quote the missing column and list what is \
             present, got: {msg}"
        );
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
    assert_eq!(d.n_rows, 4);
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
        uns: serde_json::Map::new(),
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
        uns: serde_json::Map::new(),
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
        uns: serde_json::Map::new(),
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
        uns: serde_json::Map::new(),
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
fn uns_is_left_alone_without_a_payload_and_merged_with_one() {
    let dir = tempfile::tempdir().unwrap();

    // Empty payload: the existing uns section is not even rewritten.
    let path = write_fixture(dir.path(), "a.scx", 3, 2, 1);
    let data = score_data(keys("cell_", 3), |i| i as f32);
    attach_external_obs(&path, &data, &opts()).unwrap();
    let uns = ScxReader::open(&path).unwrap().read_uns().unwrap();
    assert_eq!(uns["state"], "v0");

    // One entry: merged alongside the existing content.
    let path = write_fixture(dir.path(), "b.scx", 3, 2, 1);
    let mut data = score_data(keys("cell_", 3), |i| i as f32);
    data.uns.insert(
        "doublet".to_string(),
        serde_json::json!({"tool": "scDblFinder"}),
    );
    attach_external_obs(&path, &data, &opts()).unwrap();
    let uns = ScxReader::open(&path).unwrap().read_uns().unwrap();
    assert_eq!(uns["state"], "v0", "the pre-existing uns must survive");
    assert_eq!(uns["doublet"]["tool"], "scDblFinder");
}

/// Several top-level keys land in the same commit as the columns, and one
/// rollback removes every one of them — the shape `run_structured_de` needs
/// (two obs columns + N uns records) without a whole-obs `modify_metadata`.
#[test]
fn several_uns_keys_merge_in_one_commit_and_one_rollback_removes_them_all() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 3, 2, 1);
    let seq0 = ScxReader::open(&path).unwrap().header().manifest_sequence;

    let mut data = score_data(keys("cell_", 3), |i| i as f32);
    data.uns.insert("a".to_string(), serde_json::json!(1));
    data.uns
        .insert("b".to_string(), serde_json::json!({"x": [1.5, 2.5]}));
    data.uns.insert("c".to_string(), serde_json::json!("s"));
    attach_external_obs(&path, &data, &opts()).unwrap();

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.header().manifest_sequence, seq0 + 1, "one commit");
    assert_eq!(
        reader.read_uns().unwrap(),
        serde_json::json!({"state": "v0", "a": 1, "b": {"x": [1.5, 2.5]}, "c": "s"}),
        "every key lands; the pre-existing key is untouched"
    );
    let prov = reader.read_provenance().unwrap();
    let last = prov.operations.last().unwrap();
    assert!(
        last.params_json
            .contains("\"uns_keys_merged\":[\"a\",\"b\",\"c\"]"),
        "{}",
        last.params_json
    );
    drop(reader);

    crate::rollback(&path).unwrap();
    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(
        reader.read_uns().unwrap(),
        serde_json::json!({"state": "v0"})
    );
    assert!(reader
        .read_obs()
        .unwrap()
        .column_by_name("dbl_score")
        .is_none());
}

#[test]
fn uns_collision_names_the_key_and_overwrite_replaces_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 3, 2, 1);
    let bytes0 = std::fs::read(&path).unwrap();

    let mut data = score_data(keys("cell_", 3), |i| i as f32);
    data.uns.insert("fresh".to_string(), serde_json::json!(1));
    data.uns
        .insert("state".to_string(), serde_json::json!("v1"));

    let err = attach_external_obs(&path, &data, &opts()).unwrap_err();
    assert!(matches!(err, OpsError::InvalidInput(_)), "{err}");
    assert!(err.to_string().contains("'state'"), "{err}");
    assert_eq!(
        std::fs::read(&path).unwrap(),
        bytes0,
        "refused before any write"
    );

    attach_external_obs(
        &path,
        &data,
        &AttachObsOptions {
            overwrite: true,
            ..opts()
        },
    )
    .unwrap();
    let uns = ScxReader::open(&path).unwrap().read_uns().unwrap();
    assert_eq!(uns["state"], "v1");
    assert_eq!(uns["fresh"], 1);
}

/// Merging into a `uns` that is not a JSON object has no defined meaning;
/// refuse before writing rather than quietly replace it with `{}` (which is
/// what the old single-key path did).
#[test]
fn non_object_existing_uns_is_refused_before_writing() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 3, 2, 1);
    crate::set_uns(&path, &serde_json::json!([1, 2, 3])).unwrap();
    let bytes0 = std::fs::read(&path).unwrap();

    let mut data = score_data(keys("cell_", 3), |i| i as f32);
    data.uns.insert("a".to_string(), serde_json::json!(1));
    let err = attach_external_obs(&path, &data, &opts()).unwrap_err();
    assert!(matches!(err, OpsError::InvalidInput(_)), "{err}");
    assert!(err.to_string().contains("object"), "{err}");
    assert_eq!(std::fs::read(&path).unwrap(), bytes0);
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
    let key_col = super::resolve_key_column("obs", &obs.schema(), None).unwrap();
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
    eprintln!("n_obs                 = {}", d.n_rows);
    eprintln!("resolved_key          = {:?}", d.resolved_key);
    eprintln!("resolved_cardinality  = {:?}", d.resolved_cardinality);
    eprintln!("unique_columns        = {:?}", d.unique_columns);
    eprintln!("unique_pairs          = {:?}", d.unique_pairs);
    eprintln!("suggestion            = {:?}", d.suggestion);
    eprintln!("describe()            = {}", d.describe());

    assert!(d.n_rows > 0);
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
    let (level, msg) = axis_join_coverage_report(
        "obs",
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
    let (level, msg) = axis_join_coverage_report("obs", 10, 100, 90, 40, "barcode", some_examples)
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
        axis_join_coverage_report(
            "obs",
            50,
            100,
            50,
            0,
            "obs_names",
            examples_must_not_be_needed
        )
        .is_none(),
        "exactly half is not below half"
    );
    assert!(
        axis_join_coverage_report(
            "obs",
            99,
            100,
            1,
            20,
            "obs_names",
            examples_must_not_be_needed
        )
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
        axis_join_coverage_report("obs", 10, 100, 90, 0, "k", examples_must_not_be_needed).unwrap();
    let mismatch = axis_join_coverage_report("obs", 10, 100, 90, 5, "k", some_examples).unwrap();
    assert_eq!(coverage.0, log::Level::Info);
    assert_eq!(mismatch.0, log::Level::Warn);
    assert_ne!(
        coverage.1, mismatch.1,
        "same coverage, different cause — the messages must differ"
    );
}

// ---------------------------------------------------------------------------
// Sharded obs — the streaming rewrite
//
// Every other fixture in this file writes a single legacy `ObsMetadata`
// section, which has no per-shard reader and therefore exercises only the
// materialising fallback. These pin the streaming path: that it is taken, that
// it lands each annotation on the right global row across unequal shards, that
// it preserves the input's shard boundaries, and that it still refuses a
// malformed obs axis rather than normalising one into a well-formed file.
// ---------------------------------------------------------------------------

/// 4/2/4 over 10 rows, deliberately not `header.shard_target_rows` (which the
/// fixture sets from `n_shards`). Cumulative starts are 0/4/6; a driver that
/// derived them as `shard_idx * shard_target_rows` would get 0/4/8 and shift
/// every annotation in the last shard.
const UNEVEN: &[usize] = &[4, 2, 4];

fn sharded_fixture(dir: &Path, name: &str) -> PathBuf {
    write_fixture_with_layout(dir, name, obs_batch(10), 3, 2, ObsLayout::Shards(UNEVEN))
}

/// Scores that encode their own target row index, so a row-range error in the
/// per-shard driver shows up as a wrong *value*, not a wrong row count.
fn positional_probe(n: usize) -> ExternalObsData {
    let mut row_keys = keys("cell_", n);
    row_keys.reverse(); // permuted: a positional join cannot accidentally pass
    let scores: Vec<f32> = row_keys
        .iter()
        .map(|k| k.trim_start_matches("cell_").parse::<f32>().unwrap() * 10.0)
        .collect();
    let schema = Schema::new(vec![Field::new("dbl_score", DataType::Float32, true)]);
    let batch =
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(Float32Array::from(scores))]).unwrap();
    ExternalObsData {
        row_keys,
        row_annotations: batch,
        row_embeddings: Vec::new(),
        uns: serde_json::Map::new(),
        source_checksum: None,
        source_name: None,
    }
}

#[test]
fn join_lands_on_the_right_cell_across_uneven_obs_shards() {
    let dir = tempfile::tempdir().unwrap();
    let path = sharded_fixture(dir.path(), "a.scx");

    attach_external_obs(&path, &positional_probe(10), &opts()).unwrap();

    let obs = ScxReader::open(&path).unwrap().read_obs().unwrap();
    assert_eq!(obs.num_rows(), 10);
    let scores = f32_col(&obs, "dbl_score");
    for i in 0..10 {
        assert_eq!(
            scores.value(i),
            i as f32 * 10.0,
            "row {i} got the value for row {} — the per-shard driver used the \
             wrong global row offset",
            scores.value(i) / 10.0
        );
    }
}

/// The op must not assemble the whole obs table on a sharded target. Asserted
/// on the reader the op actually used, via the injectable entry point.
///
/// `read_obs == 0` alone is close to vacuous — a path that did nothing at all
/// satisfies it — so the projected counter must also have moved.
#[test]
fn sharded_obs_import_never_materializes_obs() {
    let dir = tempfile::tempdir().unwrap();
    let path = sharded_fixture(dir.path(), "a.scx");
    let data = score_data(keys("cell_", 10), |i| i as f32 + 0.5);

    let reader = ScxReader::open(&path).unwrap();
    let s = attach_external_obs_with_reader(&path, &reader, &data, &opts()).unwrap();
    assert!(
        s.obs_streamed,
        "a sharded target must take the streaming path"
    );

    use std::sync::atomic::Ordering;
    assert_eq!(
        reader.debug_counts().read_obs.load(Ordering::Relaxed),
        0,
        "the streaming import must not assemble the full obs table"
    );
    #[cfg(debug_assertions)]
    {
        assert!(
            reader
                .debug_counts()
                .read_obs_shard_projected
                .load(Ordering::Relaxed)
                > 0,
            "the join must have gone through the projected key read — otherwise \
             `read_obs == 0` only proves nothing happened"
        );
        // Sanity: the counter this test relies on does move when read_obs is called.
        let _ = reader.read_obs().unwrap();
        assert_eq!(reader.debug_counts().read_obs.load(Ordering::Relaxed), 1);
    }
}

/// The rewrite emits one output shard per input shard, preserving boundaries the
/// header's `shard_target_rows` does not describe.
#[test]
fn obs_shard_boundaries_survive_an_import() {
    let dir = tempfile::tempdir().unwrap();
    let path = sharded_fixture(dir.path(), "a.scx");
    let data = score_data(keys("cell_", 10), |i| i as f32);

    attach_external_obs(&path, &data, &opts()).unwrap();

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.obs_metadata_shard_count(), UNEVEN.len());
    for (idx, expected) in UNEVEN.iter().enumerate() {
        assert_eq!(
            reader.read_obs_shard(idx as u32).unwrap().num_rows(),
            *expected,
            "obs shard {idx} was re-sharded"
        );
    }
}

/// A legacy single-section target has nothing to stream; it must say so rather
/// than silently reporting the streamed path.
#[test]
fn legacy_single_section_obs_is_reported_as_not_streamed() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 6, 3, 2);
    let data = score_data(keys("cell_", 6), |i| i as f32);

    let s = attach_external_obs(&path, &data, &opts()).unwrap();
    assert!(
        !s.obs_streamed,
        "single-section obs has no per-shard reader — it cannot be streamed"
    );
}

/// A sharded obs axis with a hole must be refused, not normalised.
///
/// End-to-end property, not a test of any one layer: the refusal comes from the
/// cover walk inside `read_obs_keys`, which is why this stays green even with
/// the driver's own `ShardCoverCheck` removed. The check that *is* load-bearing
/// in the driver is pinned by
/// `a_shard_stamped_longer_than_its_payload_is_rejected` below.
#[test]
fn a_gapped_obs_axis_is_rejected_not_normalized() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gap.scx");
    let obs = obs_batch(10);

    // Shards 0 and 1 stamped at row_start 0 and 6 — rows 4..6 are covered by
    // nothing, so the axis does not tile [0, 10).
    let header = FileHeader::new_single_modality(10, 3, 0, 5, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer
        .write_obs_shard(0, 0, 4, 10, &obs.slice(0, 4))
        .unwrap();
    writer
        .write_obs_shard(1, 6, 4, 10, &obs.slice(6, 4))
        .unwrap();
    writer.write_var(&var_batch(3)).unwrap();
    writer
        .write_csr_shard(
            &[0u64; 11],
            &[],
            &[],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer.finish().unwrap();

    let data = score_data(keys("cell_", 10), |i| i as f32);
    let err = attach_external_obs(&path, &data, &opts()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("row_start") || msg.contains("cover"),
        "a gapped obs axis must be refused by name, got: {msg}"
    );
}

/// A dry run must reach the same verdict the real import will.
///
/// Round-1 finding (codex): every `--dry-run` surface promises it "runs every
/// validation", and the payload-vs-stamp check used to live inside the write
/// loop — past the dry-run return. On the cancelling fixture below the dry run
/// reported a clean join of 10 rows and the real import then failed, which is
/// the one way a preview can be worse than useless.
#[test]
fn a_dry_run_rejects_what_the_real_import_would_reject() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mis-stamped.scx");
    let obs = obs_batch(10);

    // Stamps tile [0, 10) and payloads sum to 10; only the per-shard pairing is
    // wrong, so nothing but the payload-vs-stamp check can see it.
    let header = FileHeader::new_single_modality(10, 3, 0, 5, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer
        .write_obs_shard(0, 0, 6, 10, &obs.slice(0, 4))
        .unwrap();
    writer
        .write_obs_shard(1, 6, 4, 10, &obs.slice(4, 6))
        .unwrap();
    writer.write_var(&var_batch(3)).unwrap();
    writer
        .write_csr_shard(
            &[0u64; 11],
            &[],
            &[],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer.finish().unwrap();

    let before = std::fs::read(&path).unwrap();
    let data = score_data(keys("cell_", 10), |i| i as f32);
    let dry = AttachObsOptions {
        dry_run: true,
        ..opts()
    };
    let err = attach_external_obs(&path, &data, &dry).unwrap_err();
    assert!(
        err.to_string().contains("carries"),
        "a dry run must refuse what the real import refuses, got: {err}"
    );
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "a rejected dry run must leave the file byte-identical"
    );
}

/// The two paths are one implementation seen from two angles. Same source, same
/// obs content, one sharded target and one single-section twin: the resulting
/// obs must agree on values, schema and the reported join counts.
#[test]
fn sharded_and_single_section_imports_agree() {
    let dir = tempfile::tempdir().unwrap();
    let sharded = write_fixture_with_layout(
        dir.path(),
        "sharded.scx",
        obs_batch(10),
        3,
        2,
        ObsLayout::Shards(UNEVEN),
    );
    let single = write_fixture_with_obs(dir.path(), "single.scx", obs_batch(10), 3, 2);

    let data = score_data(keys("cell_", 10), |i| i as f32 + 0.25);
    let a = attach_external_obs(&sharded, &data, &opts()).unwrap();
    let b = attach_external_obs(&single, &data, &opts()).unwrap();

    assert_eq!(a.n_matched, b.n_matched);
    assert_eq!(a.n_target_rows_absent, b.n_target_rows_absent);
    assert_eq!(a.n_source_rows_absent, b.n_source_rows_absent);
    assert_eq!(a.obs_key_column, b.obs_key_column);
    assert_eq!(a.obs_columns_added, b.obs_columns_added);

    let oa = ScxReader::open(&sharded).unwrap().read_obs().unwrap();
    let ob = ScxReader::open(&single).unwrap().read_obs().unwrap();
    assert_eq!(oa.schema(), ob.schema(), "the two paths disagree on schema");
    assert_eq!(oa.num_rows(), ob.num_rows());
    for i in 0..oa.num_columns() {
        assert_eq!(
            oa.column(i),
            ob.column(i),
            "column '{}' differs between the streaming and materialising paths",
            oa.schema().field(i).name()
        );
    }
}

/// Stamps that tile `[0, n_obs)` while the payloads they travel with do not
/// match them. The total still comes to `n_obs`, so neither the header shape
/// check nor `read_obs_keys`' cover walk notices — only the per-shard
/// payload-vs-stamp check does.
///
/// Left unchecked, the rewrite silently *normalises* this: it derives each
/// output shard's range from the rows it can see, so the file comes out
/// well-formed with obs rows bound to different matrix rows than the input
/// claimed. Rejecting beats laundering.
#[test]
fn a_shard_stamped_longer_than_its_payload_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mis-stamped.scx");
    let obs = obs_batch(10);

    let header = FileHeader::new_single_modality(10, 3, 0, 5, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    // Stamped 6 rows, carries 4.
    writer
        .write_obs_shard(0, 0, 6, 10, &obs.slice(0, 4))
        .unwrap();
    // Stamped 4 rows, carries 6 — so the payloads still total 10.
    writer
        .write_obs_shard(1, 6, 4, 10, &obs.slice(4, 6))
        .unwrap();
    writer.write_var(&var_batch(3)).unwrap();
    writer
        .write_csr_shard(
            &[0u64; 11],
            &[],
            &[],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer.finish().unwrap();

    let data = score_data(keys("cell_", 10), |i| i as f32);
    let err = attach_external_obs(&path, &data, &opts()).unwrap_err();
    assert!(
        err.to_string().contains("carries"),
        "a shard whose payload contradicts its stamp must be refused, got: {err}"
    );
}

/// A composite key on a sharded target goes through a path a single-column key
/// does not: two columns are projected out of each shard and `compact_key_shard`
/// dictionary-encodes them, so the batch `build_composite_key` fuses is
/// `Dictionary(Int32, Utf8)` rather than the plain `Utf8` the whole-table read
/// produced. The fused key has to come out identical either way, or every row
/// misses.
#[test]
fn a_composite_key_joins_across_obs_shards() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture_with_layout(
        dir.path(),
        "a.scx",
        obs_two_libraries(),
        2,
        1,
        ObsLayout::Shards(&[3, 1]),
    );

    let cols = vec!["sample_id".to_string(), "barcode".to_string()];
    let row_keys = build_composite_key(&obs_two_libraries(), &cols).unwrap();
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
    assert!(s.obs_streamed);
    assert_eq!(s.obs_key_column, "sample_id,barcode");

    let obs = ScxReader::open(&path).unwrap().read_obs().unwrap();
    let scores = f32_col(&obs, "dbl_score");
    for i in 0..4 {
        assert_eq!(
            scores.value(i),
            i as f32 * 100.0,
            "row {i} got another library's score — the composite key did not \
             survive the projected, dictionary-encoded shard read"
        );
    }
}

// ---------------------------------------------------------------------------
// Positional mode
// ---------------------------------------------------------------------------
//
// `ObsJoinKey::Positional` exists for exactly one caller shape: columns
// computed in-process from this file's own `read_obs()` output, which is
// already in physical row order — `doublet_consensus` being the motivating
// case, on files whose obs index is fully duplicated so no key join is
// possible. The key-join tests above (and `positional_probe`'s deliberately
// reversed keys) are untouched: they remain the proof that a *key* join is
// really keyed.

/// Row-index-encoding scores with NO keys: `score(i) = i * 10`, so a driver
/// that lands shard rows at the wrong global offset produces wrong values,
/// not just a wrong row count.
fn positional_data(n: usize) -> ExternalObsData {
    let scores: Vec<f32> = (0..n).map(|i| i as f32 * 10.0).collect();
    let schema = Schema::new(vec![Field::new("dbl_score", DataType::Float32, true)]);
    let batch =
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(Float32Array::from(scores))]).unwrap();
    ExternalObsData {
        row_keys: Vec::new(),
        row_annotations: batch,
        row_embeddings: Vec::new(),
        uns: serde_json::Map::new(),
        source_checksum: None,
        source_name: Some("<DataFrame>".to_string()),
    }
}

fn positional_opts() -> AttachObsOptions {
    AttachObsOptions {
        join_key: ObsJoinKey::Positional,
        status_column: None,
        provenance_action: "test_positional_attach".to_string(),
        ..Default::default()
    }
}

#[test]
fn positional_attach_lands_values_in_physical_row_order() {
    let dir = tempfile::tempdir().unwrap();
    let path = sharded_fixture(dir.path(), "a.scx");

    let s = attach_external_obs(&path, &positional_data(10), &positional_opts()).unwrap();
    assert_eq!(s.n_matched, 10);
    assert_eq!(s.n_target_rows_absent, 0);
    assert_eq!(s.n_source_rows_absent, 0);
    assert_eq!(s.obs_key_column, "<positional>");
    assert!(s.obs_streamed);

    let obs = ScxReader::open(&path).unwrap().read_obs().unwrap();
    assert_eq!(obs.num_rows(), 10);
    let scores = f32_col(&obs, "dbl_score");
    for i in 0..10 {
        assert_eq!(
            scores.value(i),
            i as f32 * 10.0,
            "row {i} got the value for row {} — positional scatter used the \
             wrong global row offset",
            scores.value(i) / 10.0
        );
    }
}

/// The capability the mode exists for: a file whose obs index is fully
/// duplicated, where the key join is structurally impossible.
#[test]
fn positional_attach_survives_duplicate_obs_index() {
    let dir = tempfile::tempdir().unwrap();
    let ids: Vec<String> = (0..4).map(|_| "dup".to_string()).collect();
    let schema = Schema::new(vec![Field::new("barcode", DataType::Utf8, false)]);
    let obs =
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(StringArray::from(ids))]).unwrap();
    let path = write_fixture_with_obs(dir.path(), "a.scx", obs, 2, 1);

    // The key join is refused outright…
    let keyed = score_data(
        vec!["dup".into(), "dup".into(), "dup".into(), "dup".into()],
        |i| i as f32,
    );
    let err = attach_external_obs(&path, &keyed, &opts()).unwrap_err();
    assert!(matches!(err, OpsError::DuplicateJoinKey { .. }), "{err}");

    // …and positional lands each row on its own cell.
    let s = attach_external_obs(&path, &positional_data(4), &positional_opts()).unwrap();
    assert_eq!(s.n_matched, 4);
    let scores = f32_col(
        &ScxReader::open(&path).unwrap().read_obs().unwrap(),
        "dbl_score",
    );
    for i in 0..4 {
        assert_eq!(scores.value(i), i as f32 * 10.0);
    }
}

#[test]
fn positional_attach_rejects_row_count_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = sharded_fixture(dir.path(), "a.scx"); // n_obs = 10
    let before = std::fs::read(&path).unwrap();

    let err = attach_external_obs(&path, &positional_data(9), &positional_opts()).unwrap_err();
    assert!(matches!(err, OpsError::ShapeMismatch { .. }), "{err}");
    let msg = err.to_string();
    assert!(
        msg.contains("9 rows")
            && msg.contains("n_obs = 10")
            && msg.contains("no logical deletions"),
        "the error must name the frame's rows and the file's, got: {msg}"
    );
    assert_eq!(
        before,
        std::fs::read(&path).unwrap(),
        "a rejected positional attach must not write a byte"
    );
}

// On a file with deletions the positional attach accepts either row space, told
// apart by length. `read_obs()` is logical since pyscx 0.17, so the live-length
// frame is the one an in-process caller now holds; the physical-length frame is
// `read_obs(logical=False)` and must keep working unchanged.

/// The `i`-th live row gets source row `i`; a deleted row gets null; the
/// deletion vector and the file's live count are untouched.
#[test]
fn positional_attach_accepts_a_live_length_frame_on_a_deleted_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = sharded_fixture(dir.path(), "a.scx"); // n_obs = 10
    crate::mark_deleted(&path, &[1, 3]).unwrap();

    let s = attach_external_obs(&path, &positional_data(8), &positional_opts()).unwrap();
    assert_eq!(
        s.n_obs, 10,
        "the summary's n_obs is the physical axis the join was built over"
    );
    assert_eq!(s.n_matched, 8);
    assert_eq!(
        s.n_target_rows_absent, 2,
        "the deleted rows are the absent ones"
    );
    assert_eq!(s.n_source_rows_absent, 0);
    assert_eq!(s.obs_key_column, "<positional>");

    let reader = ScxReader::open(&path).unwrap();
    let physical = reader.read_obs().unwrap();
    assert_eq!(physical.num_rows(), 10);
    let scores = f32_col(&physical, "dbl_score");
    let mut live = 0usize;
    for row in 0..10 {
        if row == 1 || row == 3 {
            assert!(
                scores.is_null(row),
                "deleted row {row} must be null, not a value"
            );
        } else {
            assert_eq!(
                scores.value(row),
                live as f32 * 10.0,
                "physical row {row} must carry live row {live}'s value"
            );
            live += 1;
        }
    }
    let logical = reader.read_obs_filtered().unwrap();
    let scores = f32_col(&logical, "dbl_score");
    for i in 0..8 {
        assert_eq!(
            scores.value(i),
            i as f32 * 10.0,
            "the logical read gives the frame back"
        );
    }
    let keep = reader.deletion_keep_mask().unwrap().unwrap();
    assert!(
        !keep[1] && !keep[3] && keep[0] && keep[9],
        "deletion vector carried"
    );

    let prov = reader.read_provenance().unwrap();
    let last = prov.operations.last().unwrap();
    assert!(
        last.params_json.contains("\"row_space\":\"logical\""),
        "provenance must record which row space landed: {}",
        last.params_json
    );
}

#[test]
fn positional_attach_still_accepts_a_physical_length_frame_on_a_deleted_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = sharded_fixture(dir.path(), "a.scx");
    crate::mark_deleted(&path, &[1, 3]).unwrap();

    let s = attach_external_obs(&path, &positional_data(10), &positional_opts()).unwrap();
    assert_eq!(s.n_matched, 10);
    assert_eq!(s.n_target_rows_absent, 0);

    let reader = ScxReader::open(&path).unwrap();
    let scores = f32_col(&reader.read_obs().unwrap(), "dbl_score");
    for row in 0..10 {
        assert_eq!(
            scores.value(row),
            row as f32 * 10.0,
            "a physical frame writes every row, the deleted ones included"
        );
    }
    let prov = reader.read_provenance().unwrap();
    assert!(
        prov.operations
            .last()
            .unwrap()
            .params_json
            .contains("\"row_space\":\"physical\""),
        "{}",
        prov.operations.last().unwrap().params_json
    );
}

/// Neither length: the error names both counts and the read that produces each,
/// and the file is untouched.
#[test]
fn positional_attach_rejects_neither_length_naming_both_counts() {
    let dir = tempfile::tempdir().unwrap();
    let path = sharded_fixture(dir.path(), "a.scx");
    crate::mark_deleted(&path, &[1, 3]).unwrap();
    let before = std::fs::read(&path).unwrap();

    for n in [7usize, 9] {
        let err = attach_external_obs(&path, &positional_data(n), &positional_opts()).unwrap_err();
        assert!(matches!(err, OpsError::ShapeMismatch { .. }), "{err}");
        let msg = err.to_string();
        for needle in [
            &format!("{n} rows"),
            "n_obs = 8",
            "n_obs_physical = 10",
            "read_obs()",
            "read_obs(logical=False)",
        ] {
            assert!(msg.contains(needle), "missing {needle:?} in: {msg}");
        }
    }
    assert_eq!(before, std::fs::read(&path).unwrap());
}

/// More than half deleted: the key join's coverage report would warn about a sample-name
/// prefix here, which is nonsense for a positional attach. The live join is
/// built directly and must stay quiet and correct.
#[test]
fn positional_live_attach_on_a_mostly_deleted_file_lands_the_survivors() {
    let dir = tempfile::tempdir().unwrap();
    let path = sharded_fixture(dir.path(), "a.scx");
    crate::mark_deleted(&path, &[0, 1, 2, 4, 5, 6, 8]).unwrap(); // live: 3, 7, 9

    let s = attach_external_obs(&path, &positional_data(3), &positional_opts()).unwrap();
    assert_eq!(s.n_matched, 3);
    assert_eq!(s.n_target_rows_absent, 7);
    let scores = f32_col(
        &ScxReader::open(&path).unwrap().read_obs().unwrap(),
        "dbl_score",
    );
    assert_eq!(scores.value(3), 0.0);
    assert_eq!(scores.value(7), 10.0);
    assert_eq!(scores.value(9), 20.0);
    assert!(scores.is_null(0) && scores.is_null(8));
}

/// A dense mapping has no null to stand in for a deleted row, so a live-length
/// frame may not carry obsm embeddings.
#[test]
fn positional_attach_rejects_row_embeddings_on_a_live_length_frame() {
    let dir = tempfile::tempdir().unwrap();
    let path = sharded_fixture(dir.path(), "a.scx");
    crate::mark_deleted(&path, &[1, 3]).unwrap();
    let before = std::fs::read(&path).unwrap();

    let mut data = positional_data(8);
    let emb = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("c0", DataType::Float32, true)])),
        vec![Arc::new(Float32Array::from(vec![0.0f32; 8]))],
    )
    .unwrap();
    data.row_embeddings = vec![("X_pca".to_string(), emb)];

    let err = attach_external_obs(&path, &data, &positional_opts()).unwrap_err();
    assert!(matches!(err, OpsError::InvalidInput(_)), "{err}");
    assert!(err.to_string().contains("obsm embeddings"), "{err}");
    assert_eq!(before, std::fs::read(&path).unwrap());
}

/// A 10-row file whose obs carries a pandas-style index column
/// (`__index_level_0__` = cell_0..cell_9), as every `from_anndata` file does —
/// the alignment check below resolves the file's barcodes through it.
fn indexed_fixture(dir: &Path, name: &str) -> PathBuf {
    let ids: Vec<String> = keys("cell_", 10);
    let schema = Schema::new(vec![
        Field::new("__index_level_0__", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, true),
    ]);
    let obs = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(
                (0..10)
                    .map(|i| if i % 2 == 0 { "T" } else { "B" })
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    write_fixture_with_obs(dir, name, obs, 2, 2)
}

/// Under positional, supplied keys are never joined on — they are an alignment
/// check: the frame's labels must equal the file's barcodes row for row, so a
/// frame sorted or reindexed after `read_obs()` (right length, wrong order) is
/// refused by row instead of landing every value on the wrong cell.
#[test]
fn positional_attach_keys_are_an_alignment_check_not_a_join() {
    let dir = tempfile::tempdir().unwrap();
    let path = indexed_fixture(dir.path(), "a.scx");
    let before = std::fs::read(&path).unwrap();

    // Right length, permuted: refused, naming the first offending row.
    let mut data = positional_data(10);
    let mut permuted = keys("cell_", 10);
    permuted.swap(2, 7);
    data.row_keys = permuted;
    let err = attach_external_obs(&path, &data, &positional_opts()).unwrap_err();
    assert!(matches!(err, OpsError::InvalidInput(_)), "{err}");
    let msg = err.to_string();
    assert!(
        msg.contains("different order") && msg.contains("frame row 2") && msg.contains("cell_7"),
        "the refusal must name the row and both labels: {msg}"
    );
    assert_eq!(
        before,
        std::fs::read(&path).unwrap(),
        "refused → not a byte written"
    );

    // Wrong key count: refused as a shape error.
    let mut data = positional_data(10);
    data.row_keys = keys("cell_", 9);
    let err = attach_external_obs(&path, &data, &positional_opts()).unwrap_err();
    assert!(matches!(err, OpsError::ShapeMismatch { .. }), "{err}");
    assert!(err.to_string().contains("row_keys has 9"), "{err}");

    // Matching keys: the check passes and the attach is positional as ever.
    let mut data = positional_data(10);
    data.row_keys = keys("cell_", 10);
    let s = attach_external_obs(&path, &data, &positional_opts()).unwrap();
    assert_eq!(s.n_matched, 10);
    let reader = ScxReader::open(&path).unwrap();
    let scores = f32_col(&reader.read_obs().unwrap(), "dbl_score");
    for i in 0..10 {
        assert_eq!(scores.value(i), i as f32 * 10.0);
    }
    let prov = reader.read_provenance().unwrap();
    assert!(
        prov.operations
            .last()
            .unwrap()
            .params_json
            .contains("\"positional_index_checked\":true"),
        "{}",
        prov.operations.last().unwrap().params_json
    );

    // Labels that are not the file's barcodes at all: ignored, as the index
    // always was under positional (a frame from another source with its own
    // row labels is still "row i annotates row i").
    let path2 = indexed_fixture(dir.path(), "b.scx");
    let mut data = positional_data(10);
    data.row_keys = keys("other_", 10);
    let s = attach_external_obs(&path2, &data, &positional_opts()).unwrap();
    assert_eq!(s.n_matched, 10);
}

/// The alignment check follows the frame's row space: live keys against the
/// live barcodes on a deleted file.
#[test]
fn positional_live_attach_checks_keys_against_the_live_barcodes() {
    let dir = tempfile::tempdir().unwrap();
    let path = indexed_fixture(dir.path(), "a.scx");
    crate::mark_deleted(&path, &[1, 3]).unwrap();

    // The live barcodes in order: cell_0, cell_2, cell_4..cell_9.
    let live: Vec<String> = (0..10)
        .filter(|i| *i != 1 && *i != 3)
        .map(|i| format!("cell_{i}"))
        .collect();
    let mut data = positional_data(8);
    data.row_keys = live.clone();
    let s = attach_external_obs(&path, &data, &positional_opts()).unwrap();
    assert_eq!(s.n_matched, 8);

    // The same labels reversed: a `read_obs().sort_values(...)` accident.
    let mut data = positional_data(8);
    data.row_keys = live.into_iter().rev().collect();
    let err = attach_external_obs(&path, &data, &positional_opts()).unwrap_err();
    assert!(err.to_string().contains("different order"), "{err}");
}

/// A two-level obs index compares as the composite key the key join builds, on
/// both sides — so a reordered MultiIndex frame is refused like a single-level one.
#[test]
fn positional_attach_checks_a_multi_level_index_as_a_composite_key() {
    let dir = tempfile::tempdir().unwrap();
    let mut meta = std::collections::HashMap::new();
    meta.insert(
        "pandas".to_string(),
        "{\"index_columns\":[\"lvl_a\",\"lvl_b\"]}".to_string(),
    );
    let schema = Schema::new(vec![
        Field::new("lvl_a", DataType::Utf8, false),
        Field::new("lvl_b", DataType::Utf8, false),
    ])
    .with_metadata(meta);
    let obs = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(keys("a", 10))),
            Arc::new(StringArray::from(keys("b", 10))),
        ],
    )
    .unwrap();
    let path = write_fixture_with_obs(dir.path(), "multi.scx", obs, 2, 2);
    crate::mark_deleted(&path, &[1, 3]).unwrap();

    let composite = |i: usize| format!("a{i}{}b{i}", COMPOSITE_KEY_SEPARATOR);
    let live: Vec<String> = (0..10)
        .filter(|i| *i != 1 && *i != 3)
        .map(composite)
        .collect();

    // Reversed live composite keys: refused, naming the levels.
    let mut data = positional_data(8);
    data.row_keys = live.iter().rev().cloned().collect();
    let err = attach_external_obs(&path, &data, &positional_opts()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("different order") && msg.contains("lvl_a+lvl_b"),
        "{msg}"
    );

    // In order: lands.
    let mut data = positional_data(8);
    data.row_keys = live;
    assert_eq!(
        attach_external_obs(&path, &data, &positional_opts())
            .unwrap()
            .n_matched,
        8
    );
}

/// A file with no obs index column cannot be checked against, so keys are
/// refused by name rather than compared against a guessed column.
#[test]
fn positional_attach_keys_need_an_obs_index_column_to_check_against() {
    let dir = tempfile::tempdir().unwrap();
    let path = sharded_fixture(dir.path(), "a.scx"); // `cell_id` only, no index column
    let mut data = positional_data(10);
    data.row_keys = keys("cell_", 10);
    let err = attach_external_obs(&path, &data, &positional_opts()).unwrap_err();
    assert!(err.to_string().contains("no index column"), "{err}");
    // Without keys the same file attaches positionally as before.
    attach_external_obs(&path, &positional_data(10), &positional_opts()).unwrap();
}

/// Every row deleted: `read_obs()` is a 0-row frame and is exactly the live
/// frame — the attach must accept it (all-null column) rather than refuse it.
#[test]
fn positional_attach_accepts_an_empty_live_frame_when_every_row_is_deleted() {
    let dir = tempfile::tempdir().unwrap();
    let path = sharded_fixture(dir.path(), "a.scx");
    crate::mark_deleted(&path, &(0..10u64).collect::<Vec<_>>()).unwrap();

    let s = attach_external_obs(&path, &positional_data(0), &positional_opts()).unwrap();
    assert_eq!(s.n_matched, 0);
    assert_eq!(s.n_target_rows_absent, 10);
    let obs = ScxReader::open(&path).unwrap().read_obs().unwrap();
    assert_eq!(obs.num_rows(), 10);
    let scores = f32_col(&obs, "dbl_score");
    assert!(
        (0..10).all(|i| scores.is_null(i)),
        "every deleted row is null"
    );
}

#[test]
fn positional_attach_rejects_status_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 2, 1);

    let err = attach_external_obs(
        &path,
        &positional_data(4),
        &AttachObsOptions {
            status_column: Some("dbl_status".to_string()),
            ..positional_opts()
        },
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("status_column"),
        "a status column is a constant under positional and must be refused, got: {err}"
    );
}

/// A positional dry run must reach the same verdict the real attach would —
/// including on the cancelling-stamps fixture only the payload-vs-stamp check
/// can see. Skipping the preflight because "there is no key to read" is the
/// regression this pins.
#[test]
fn positional_dry_run_is_honest() {
    // Clean file: exact summary, no bytes written.
    let dir = tempfile::tempdir().unwrap();
    let path = sharded_fixture(dir.path(), "a.scx");
    let before = std::fs::read(&path).unwrap();
    let s = attach_external_obs(
        &path,
        &positional_data(10),
        &AttachObsOptions {
            dry_run: true,
            ..positional_opts()
        },
    )
    .unwrap();
    assert_eq!(s.n_matched, 10);
    assert_eq!(s.obs_columns_added, vec!["dbl_score"]);
    assert_eq!(before, std::fs::read(&path).unwrap(), "dry run wrote bytes");

    // Mis-stamped file (stamps tile [0, 10), payloads disagree, totals cancel):
    // the dry run must refuse it, exactly as the keyed dry run does.
    let path = dir.path().join("mis-stamped.scx");
    let obs = obs_batch(10);
    let header = FileHeader::new_single_modality(10, 3, 0, 5, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer
        .write_obs_shard(0, 0, 6, 10, &obs.slice(0, 4))
        .unwrap();
    writer
        .write_obs_shard(1, 6, 4, 10, &obs.slice(4, 6))
        .unwrap();
    writer.write_var(&var_batch(3)).unwrap();
    writer
        .write_csr_shard(
            &[0u64; 11],
            &[],
            &[],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer.finish().unwrap();

    let before = std::fs::read(&path).unwrap();
    let err = attach_external_obs(
        &path,
        &positional_data(10),
        &AttachObsOptions {
            dry_run: true,
            ..positional_opts()
        },
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("carries"),
        "a positional dry run must run the payload-vs-stamp preflight, got: {err}"
    );
    assert_eq!(before, std::fs::read(&path).unwrap());
}

/// §10.5 at the ops layer: a positional pure add is still a pure add, so the
/// obs predicate index (and every shard column stat) survives.
#[test]
fn positional_pure_add_keeps_obs_predicate_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture_with_obs_index(dir.path(), "a.scx");
    assert!(has_obs_index(&path));
    assert_eq!(
        columns_with_shard_stats(&path, &["cell_type", "n_counts"]),
        vec!["cell_type".to_string(), "n_counts".to_string()]
    );

    let s = attach_external_obs(&path, &positional_data(4), &positional_opts()).unwrap();

    assert!(!s.obs_index_dropped);
    assert!(
        has_obs_index(&path),
        "a positional pure add keeps the index"
    );
    assert_eq!(
        columns_with_shard_stats(&path, &["cell_type", "n_counts"]),
        vec!["cell_type".to_string(), "n_counts".to_string()],
        "a positional pure add keeps every shard column stat"
    );
}

/// The one-key uns merge under positional — the exact combination the
/// `doublet_consensus` repoint uses: columns + one uns key, one commit.
#[test]
fn positional_attach_merges_one_uns_key_and_one_rollback_undoes_both() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 2, 1);

    let mut data = positional_data(4);
    data.uns.insert(
        "dbl_consensus".to_string(),
        serde_json::json!({"method": "majority"}),
    );
    let s = attach_external_obs(&path, &data, &positional_opts()).unwrap();
    assert_eq!(s.obs_columns_added, vec!["dbl_score"]);

    let reader = ScxReader::open(&path).unwrap();
    let uns = reader.read_uns().unwrap();
    assert_eq!(uns["dbl_consensus"]["method"], "majority");
    assert_eq!(
        uns["state"], "v0",
        "pre-existing uns keys must survive the merge"
    );
    drop(reader);

    crate::rollback(&path).unwrap();
    let reader = ScxReader::open(&path).unwrap();
    let obs = reader.read_obs().unwrap();
    assert!(obs.column_by_name("dbl_score").is_none());
    let uns = reader.read_uns().unwrap();
    assert!(
        uns.get("dbl_consensus").is_none(),
        "one rollback must undo obs and uns together — the attach is one commit"
    );
}

#[test]
fn positional_attach_provenance_records_positional_key() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 2, 1);

    attach_external_obs(&path, &positional_data(4), &positional_opts()).unwrap();

    let prov = ScxReader::open(&path).unwrap().read_provenance().unwrap();
    let last = prov.operations.last().unwrap();
    assert_eq!(last.action, "test_positional_attach");
    assert!(
        last.params_json
            .contains("\"obs_key_column\":\"<positional>\""),
        "{}",
        last.params_json
    );
    assert!(
        last.params_json.contains("\"n_matched\":4"),
        "{}",
        last.params_json
    );
}

#[test]
fn positional_attach_leaves_var_x_and_uns_sections_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 3, 2);

    let section_spans = |path: &Path| -> Vec<(String, u64, u64)> {
        ScxReader::open(path)
            .unwrap()
            .catalog()
            .entries
            .iter()
            .filter(|e| {
                matches!(
                    e.section_type,
                    SectionType::VarMetadata
                        | SectionType::VarMetadataShard
                        | SectionType::CsrShard
                        | SectionType::UnsBlob
                )
            })
            .map(|e| (e.name.clone(), e.offset, e.length))
            .collect()
    };
    let before = section_spans(&path);
    assert!(!before.is_empty());

    attach_external_obs(&path, &positional_data(4), &positional_opts()).unwrap();

    assert_eq!(
        before,
        section_spans(&path),
        "a positional obs-only attach (no uns payload) must not rewrite or move \
         var / X / uns sections"
    );
}

/// A status column sharing a name with an annotation would write TWO obs
/// columns with that name — `check_collisions` only compares planned names
/// against the OLD schema. (Round-2 finding: codex.)
#[test]
fn a_status_column_matching_an_annotation_name_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 2, 1);
    let before = std::fs::read(&path).unwrap();

    let data = score_data(keys("cell_", 4), |i| i as f32);
    let err = attach_external_obs(
        &path,
        &data,
        &AttachObsOptions {
            status_column: Some("dbl_score".to_string()), // = an annotation name
            ..opts()
        },
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("dbl_score") && err.to_string().contains("same name"),
        "{err}"
    );
    assert_eq!(before, std::fs::read(&path).unwrap());
}

/// Overwrite the file's `uns` section bytes in place so `read_uns()` fails for a
/// reason other than "absent" (checksum / JSON), keeping the catalog intact.
fn corrupt_uns_section(path: &Path) {
    use std::io::{Seek, SeekFrom, Write};
    let (offset, length) = {
        let r = ScxReader::open(path).unwrap();
        let e = r
            .catalog()
            .entries
            .iter()
            .find(|e| e.section_type == SectionType::UnsBlob)
            .expect("fixture has a uns section");
        (e.offset, e.length as usize)
    };
    let mut f = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    f.seek(SeekFrom::Start(offset)).unwrap();
    f.write_all(&vec![b'{'; length]).unwrap();
    assert!(
        ScxReader::open(path).unwrap().read_uns().is_err(),
        "corruption must make the read fail"
    );
}

/// An unreadable `uns` is not an absent one. Treating every read error as `{}`
/// would let an unrelated attach commit only its own keys and orphan the
/// existing metadata. (Round 1: codex.)
#[test]
fn an_unreadable_uns_section_fails_the_attach_instead_of_being_replaced() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 3, 2, 1);
    corrupt_uns_section(&path);
    let bytes0 = std::fs::read(&path).unwrap();

    // With a payload: refused before any write.
    let mut data = score_data(keys("cell_", 3), |i| i as f32);
    data.uns.insert("a".to_string(), serde_json::json!(1));
    let err = attach_external_obs(&path, &data, &opts()).unwrap_err();
    assert!(
        !matches!(err, OpsError::InvalidInput(_)),
        "must surface the read error itself, not an input complaint: {err}"
    );
    assert_eq!(std::fs::read(&path).unwrap(), bytes0, "file byte-identical");

    // Without a payload the section is never read or touched, so a plain
    // column attach on such a file still works.
    let data = score_data(keys("cell_", 3), |i| i as f32);
    attach_external_obs(&path, &data, &opts()).unwrap();
}

// ---------------------------------------------------------------------------
// Categorical fidelity
//
// A pandas categorical reaches this op as `Dictionary(_, Utf8)` with the
// `scx.categorical.ordered` field stamp, and `from_anndata` writes it to disk
// that way. Every in-place obs writer used to run the rebuilt table through
// the shared `unify_dict_columns` cast, which decoded each dictionary column
// to plain strings — so
// after any attach the column came back from `read_obs()` as `object`, with its
// category list and `ordered` bit gone. These pin the contract that it does
// not: values unchanged, declared order and unused levels kept, `ordered` kept,
// on both rewrite paths and for a categorical the *source* brings in.
// ---------------------------------------------------------------------------

/// Obs with a `barcode` key and a `cell_type` categorical the way pandas lands
/// one: `Dictionary(Int8, Utf8)`, an `ordered` stamp, and a declared level no
/// row uses. The unused level is what tells "carried the dictionary" apart from
/// "rebuilt one from the data".
fn categorical_obs_batch(n: usize) -> RecordBatch {
    use arrow::array::{DictionaryArray, Int8Array};
    use arrow::datatypes::Int8Type;
    use std::collections::HashMap;

    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let keys = Int8Array::from((0..n).map(|i| (i % 2) as i8).collect::<Vec<_>>());
    let values = StringArray::from(vec!["T cell", "B cell", "unused"]);
    let dict = DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap();
    let mut md = HashMap::new();
    md.insert(
        scx_format_io::CATEGORICAL_ORDERED_KEY.to_string(),
        "true".to_string(),
    );
    let schema = Schema::new(vec![
        Field::new("barcode", DataType::Utf8, false),
        Field::new("cell_type", dict.data_type().clone(), true).with_metadata(md),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(ids)), Arc::new(dict)],
    )
    .unwrap()
}

/// The column's declared dictionary values in declared order, or `None` when
/// it is not dictionary-encoded at all.
fn dict_values(batch: &RecordBatch, name: &str) -> Option<Vec<String>> {
    use arrow::array::AsArray;
    let col = batch.column_by_name(name).unwrap();
    let dict = col.as_any_dictionary_opt()?;
    let values = arrow::compute::cast(dict.values(), &DataType::Utf8).unwrap();
    let values = values.as_string::<i32>();
    Some(
        (0..values.len())
            .map(|i| values.value(i).to_string())
            .collect(),
    )
}

fn ordered_flag(batch: &RecordBatch, name: &str) -> Option<String> {
    batch
        .schema()
        .field_with_name(name)
        .unwrap()
        .metadata()
        .get(scx_format_io::CATEGORICAL_ORDERED_KEY)
        .cloned()
}

/// Per-row values with nulls kept, whatever the on-disk representation.
fn opt_str_col(batch: &RecordBatch, name: &str) -> Vec<Option<String>> {
    let s = str_col(batch, name);
    (0..s.len())
        .map(|i| (!s.is_null(i)).then(|| s.value(i).to_string()))
        .collect()
}

#[test]
fn existing_categorical_columns_survive_the_rewrite_on_both_paths() {
    for layout in [ObsLayout::Single, ObsLayout::Shards(&[4, 2])] {
        let dir = tempfile::tempdir().unwrap();
        let path = write_fixture_with_layout(
            dir.path(),
            "cat.scx",
            categorical_obs_batch(6),
            3,
            2,
            layout,
        );
        let before = opt_str_col(
            &ScxReader::open(&path).unwrap().read_obs().unwrap(),
            "cell_type",
        );

        let data = score_data(keys("cell_", 6), |i| i as f32);
        attach_external_obs(&path, &data, &opts()).unwrap();

        let obs = ScxReader::open(&path).unwrap().read_obs().unwrap();
        assert_eq!(
            dict_values(&obs, "cell_type"),
            Some(vec![
                "T cell".to_string(),
                "B cell".to_string(),
                "unused".to_string()
            ]),
            "{layout:?}: cell_type must still be a dictionary carrying its declared \
             vocabulary, got {:?}",
            obs.column_by_name("cell_type").unwrap().data_type()
        );
        assert_eq!(
            opt_str_col(&obs, "cell_type"),
            before,
            "{layout:?}: values changed"
        );
        assert_eq!(
            ordered_flag(&obs, "cell_type").as_deref(),
            Some("true"),
            "{layout:?}: the ordered flag was lost"
        );
        assert_eq!(f32_col(&obs, "dbl_score").value(5), 5.0);
    }
}

/// A categorical the *source* brings in lands as one too: dictionary dtype, the
/// caller's declared order (neither alphabetical nor first-appearance), its
/// unused level, its `ordered` stamp — and null keys, not a fabricated level,
/// on the rows the source does not cover.
#[test]
fn a_categorical_annotation_lands_as_a_dictionary_with_its_flag_and_null_keys() {
    use arrow::array::{DictionaryArray, Int32Array};
    use arrow::datatypes::Int32Type;
    use std::collections::HashMap;

    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 6, 3, 2);

    let keys = Int32Array::from(vec![1, 0, 1, 0]);
    let values = StringArray::from(vec!["doublet", "singlet", "unsure"]);
    let dict = DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).unwrap();
    let mut md = HashMap::new();
    md.insert(
        scx_format_io::CATEGORICAL_ORDERED_KEY.to_string(),
        "true".to_string(),
    );
    let schema = Schema::new(vec![Field::new(
        "dbl_class",
        dict.data_type().clone(),
        true,
    )
    .with_metadata(md)]);
    let data = ExternalObsData {
        row_keys: ["cell_5", "cell_0", "cell_3", "cell_1"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        row_annotations: RecordBatch::try_new(Arc::new(schema), vec![Arc::new(dict)]).unwrap(),
        row_embeddings: Vec::new(),
        uns: serde_json::Map::new(),
        source_checksum: None,
        source_name: None,
    };
    attach_external_obs(&path, &data, &opts()).unwrap();

    let obs = ScxReader::open(&path).unwrap().read_obs().unwrap();
    assert_eq!(
        dict_values(&obs, "dbl_class"),
        Some(vec![
            "doublet".to_string(),
            "singlet".to_string(),
            "unsure".to_string()
        ]),
        "the attached column must be a dictionary carrying the caller's vocabulary, got {:?}",
        obs.column_by_name("dbl_class").unwrap().data_type()
    );
    assert_eq!(ordered_flag(&obs, "dbl_class").as_deref(), Some("true"));
    let s = |v: &str| Some(v.to_string());
    assert_eq!(
        opt_str_col(&obs, "dbl_class"),
        vec![
            s("doublet"),
            s("doublet"),
            None,
            s("singlet"),
            None,
            s("singlet")
        ],
        "rows are joined by key; uncovered rows are null"
    );
}

/// The bindings' half of the same contract: the shared column-drop helper must
/// hand the op the source's field metadata, or an ordered factor from pyscx /
/// rscx arrives already unordered.
#[test]
fn drop_batch_columns_keeps_field_metadata() {
    use arrow::array::{DictionaryArray, Int32Array};
    use arrow::datatypes::Int32Type;
    use std::collections::HashMap;

    let dict = DictionaryArray::<Int32Type>::try_new(
        Int32Array::from(vec![0, 1]),
        Arc::new(StringArray::from(vec!["a", "b"])),
    )
    .unwrap();
    let mut md = HashMap::new();
    md.insert(
        scx_format_io::CATEGORICAL_ORDERED_KEY.to_string(),
        "true".to_string(),
    );
    let schema = Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("cat", dict.data_type().clone(), false).with_metadata(md),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(vec!["k0", "k1"])),
            Arc::new(dict),
        ],
    )
    .unwrap();

    let out = drop_batch_columns(&batch, &["key".to_string()]).unwrap();
    assert_eq!(out.num_columns(), 1);
    assert!(matches!(
        out.column(0).data_type(),
        DataType::Dictionary(_, _)
    ));
    assert!(
        out.schema().field(0).is_nullable(),
        "nullability is still forced"
    );
    assert_eq!(ordered_flag(&out, "cat").as_deref(), Some("true"));
}

/// A file written before this fix can carry one column as a dictionary in some
/// shards and as plain strings in others (an old attach stringified only the
/// shards it rewrote; `append` still does). Such a file must still read, still
/// take an attach, and come out with the column reconciled to a dictionary —
/// while the shards' *existing* columns keep whatever representation they had,
/// because the op rewrites a shard's rows, it does not repair its encoding.
#[test]
fn a_legacy_file_mixing_dictionary_and_plain_obs_shards_still_takes_an_attach() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.scx");
    let obs = categorical_obs_batch(6);
    // Shard 1 the way an old attach wrote it: the dictionary decoded to its
    // value type, the field rebuilt plain.
    let plain = {
        let ct = arrow::compute::cast(obs.column(1), &DataType::Utf8).unwrap();
        let schema = Schema::new(vec![
            obs.schema().field(0).as_ref().clone(),
            Field::new("cell_type", DataType::Utf8, true),
        ]);
        RecordBatch::try_new(Arc::new(schema), vec![obs.column(0).clone(), ct]).unwrap()
    };
    let header = FileHeader::new_single_modality(6, 3, 0, 3, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer
        .write_obs_shard(0, 0, 3, 6, &obs.slice(0, 3))
        .unwrap();
    writer
        .write_obs_shard(1, 3, 3, 6, &plain.slice(3, 3))
        .unwrap();
    writer.write_var(&var_batch(3)).unwrap();
    for start in [0u64, 3] {
        writer
            .write_csr_shard(
                &[0u64; 4],
                &[],
                &[],
                CodecId::None,
                ValueEncoding::Uint8,
                start,
            )
            .unwrap();
    }
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

    // Premise: the mix is real on disk.
    let reader = ScxReader::open(&path).unwrap();
    let pre0 = reader.read_obs_shard(0).unwrap();
    let pre1 = reader.read_obs_shard(1).unwrap();
    assert!(
        matches!(
            pre0.column_by_name("cell_type").unwrap().data_type(),
            DataType::Dictionary(_, _)
        ),
        "premise: shard 0 is a dictionary"
    );
    assert!(
        matches!(
            pre1.column_by_name("cell_type").unwrap().data_type(),
            DataType::Utf8 | DataType::LargeUtf8
        ),
        "premise: shard 1 is plain, got {:?}",
        pre1.column_by_name("cell_type").unwrap().data_type()
    );
    let before = opt_str_col(&reader.read_obs().unwrap(), "cell_type");
    drop(reader);

    let s =
        attach_external_obs(&path, &score_data(keys("cell_", 6), |i| i as f32), &opts()).unwrap();
    assert!(s.obs_streamed);

    let reader = ScxReader::open(&path).unwrap();
    let obs = reader.read_obs().unwrap();
    assert!(
        matches!(
            obs.column_by_name("cell_type").unwrap().data_type(),
            DataType::Dictionary(_, _)
        ),
        "the assembled column is reconciled to a dictionary, got {:?}",
        obs.column_by_name("cell_type").unwrap().data_type()
    );
    assert_eq!(opt_str_col(&obs, "cell_type"), before);
    assert_eq!(ordered_flag(&obs, "cell_type").as_deref(), Some("true"));

    // Untouched columns keep their per-shard representation and values.
    for (idx, pre) in [(0u32, &pre0), (1u32, &pre1)] {
        let post = reader.read_obs_shard(idx).unwrap();
        assert_eq!(
            post.column_by_name("cell_type").unwrap().data_type(),
            pre.column_by_name("cell_type").unwrap().data_type(),
            "shard {idx}: the attach must not re-encode a column it did not touch"
        );
        assert_eq!(
            opt_str_col(&post, "cell_type"),
            opt_str_col(pre, "cell_type"),
            "shard {idx}: values"
        );
        assert!(
            post.column_by_name("dbl_score").is_some(),
            "shard {idx}: new column"
        );
    }
}
