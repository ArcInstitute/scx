//! Tests for [`attach_external_layer`].
//!
//! Priorities, roughly by how badly a regression would hurt:
//!
//! 1. **The join is by key.** A permuted source must land each row on the right
//!    cell. This is the CellBender-filtered-output case (descending-UMI order),
//!    where a positional assumption silently scrambles every cell.
//! 2. **Re-import yields one layer, not two.** The writer's duplicate-section
//!    guard is inert under an adopted writer, so old entries must be dropped by
//!    stem or the layer reads back at 2x `n_obs`.
//! 3. **Nothing else is lost.** Deletion vectors, the provenance chain, and
//!    `data_generation` all survive; `scx rollback` undoes the lot.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Array, Float32Array, RecordBatch, StringArray};
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
/// (the op never reads it) but the shard *ranges* are what the layer must copy.
fn write_fixture(dir: &Path, name: &str, n_obs: usize, n_vars: usize, n_shards: usize) -> PathBuf {
    let path = dir.join(name);
    let rows_per = n_obs.div_ceil(n_shards);
    let header =
        FileHeader::new_single_modality(n_obs as u64, n_vars as u64, 0, rows_per as u32, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&obs_batch(n_obs)).unwrap();
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

/// Diagonal source data: row `i` holds `value(i)` at column `i % n_cols`, so a
/// mis-join shows up in the values, not just in the row count.
fn diagonal_data(
    row_keys: Vec<String>,
    col_keys: Vec<String>,
    value: impl Fn(usize) -> f32,
) -> ExternalLayerData {
    let n_cols = col_keys.len();
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for i in 0..row_keys.len() {
        indices.push((i % n_cols) as u32);
        values.push(value(i));
        indptr.push(indices.len() as u64);
    }
    ExternalLayerData {
        row_keys,
        col_keys,
        indptr,
        indices,
        values,
        row_annotations: None,
        row_embeddings: Vec::new(),
        col_annotations: None,
        uns: None,
        source_checksum: None,
        source_name: Some("test.h5".to_string()),
    }
}

fn opts(layer: &str) -> AttachLayerOptions {
    AttachLayerOptions {
        layer_name: layer.to_string(),
        status_column: Some("cb_status".to_string()),
        provenance_action: "test_import".to_string(),
        ..Default::default()
    }
}

fn keys(prefix: &str, n: usize) -> Vec<String> {
    (0..n).map(|i| format!("{prefix}{i}")).collect()
}

// ---------------------------------------------------------------------------
// Happy path
// ---------------------------------------------------------------------------

#[test]
fn attaches_layer_covering_the_full_obs_axis() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 8, 4, 2);
    let data = diagonal_data(keys("cell_", 8), keys("g", 4), |i| (i + 1) as f32);

    let summary = attach_external_layer(&path, &data, &opts("cb")).unwrap();
    assert_eq!(summary.n_matched, 8);
    assert_eq!(summary.n_target_rows_absent, 0);
    assert_eq!(summary.obs_key_column, "barcode");
    assert_eq!(summary.var_key_column, "gene_id");
    assert_eq!(summary.column_axis_match, ColumnAxisMatch::Identical);
    assert_eq!(summary.shard_ranges_from, ShardRangeSource::XShards);

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.layer_names(), vec!["cb".to_string()]);
    let layer = reader.read_layer("cb").unwrap();
    assert_eq!(layer.shape, (8, 4));
    for i in 0..8 {
        let (s, e) = (layer.indptr[i] as usize, layer.indptr[i + 1] as usize);
        assert_eq!(&layer.indices[s..e], &[(i % 4) as i32]);
        assert_eq!(&layer.data[s..e], &[(i + 1) as f32]);
    }
}

#[test]
fn layer_shard_ranges_match_the_x_shards() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 10, 3, 3);
    let data = diagonal_data(keys("cell_", 10), keys("g", 3), |i| (i + 1) as f32);
    attach_external_layer(&path, &data, &opts("cb")).unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let ranges = |t: SectionType| -> Vec<(u64, u64)> {
        let mut v: Vec<(u64, u64)> = reader
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == t)
            .filter_map(|e| e.stats.as_ref().map(|s| (s.row_start, s.row_end)))
            .collect();
        v.sort();
        v
    };
    assert_eq!(
        ranges(SectionType::LayerCsrShard),
        ranges(SectionType::CsrShard)
    );
}

/// The headline regression. CellBender's filtered output is in descending-UMI
/// order, so if the join were positional every value would land on the wrong
/// cell — and the row count would still look right.
#[test]
fn permuted_source_rows_join_by_key_not_position() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 6, 3, 2);

    // Source rows in reverse order; each value encodes its *target* row index.
    let rev: Vec<String> = (0..6).rev().map(|i| format!("cell_{i}")).collect();
    let mut indptr = vec![0u64];
    let (mut indices, mut values) = (Vec::new(), Vec::new());
    for key in &rev {
        let target: usize = key.strip_prefix("cell_").unwrap().parse().unwrap();
        indices.push((target % 3) as u32);
        values.push(100.0 + target as f32);
        indptr.push(indices.len() as u64);
    }
    let data = ExternalLayerData {
        row_keys: rev,
        col_keys: keys("g", 3),
        indptr,
        indices,
        values,
        row_annotations: None,
        row_embeddings: Vec::new(),
        col_annotations: None,
        uns: None,
        source_checksum: None,
        source_name: None,
    };
    attach_external_layer(&path, &data, &opts("cb")).unwrap();

    let layer = ScxReader::open(&path).unwrap().read_layer("cb").unwrap();
    for target in 0..6usize {
        let (s, e) = (
            layer.indptr[target] as usize,
            layer.indptr[target + 1] as usize,
        );
        assert_eq!(
            &layer.data[s..e],
            &[100.0 + target as f32],
            "row {target} received another cell's value"
        );
    }
}

#[test]
fn unmatched_target_rows_zero_fill_with_absent_status_and_null_annotations() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 6, 3, 2);

    // Source covers only cells 0, 2, 4.
    let src_keys: Vec<String> = [0, 2, 4].iter().map(|i| format!("cell_{i}")).collect();
    let mut data = diagonal_data(src_keys, keys("g", 3), |i| (i + 1) as f32);
    data.row_annotations = Some(
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "cb_prob",
                DataType::Float32,
                true,
            )])),
            vec![Arc::new(Float32Array::from(vec![0.9f32, 0.8, 0.7]))],
        )
        .unwrap(),
    );

    let summary = attach_external_layer(&path, &data, &opts("cb")).unwrap();
    assert_eq!(summary.n_matched, 3);
    assert_eq!(summary.n_target_rows_absent, 3);

    let reader = ScxReader::open(&path).unwrap();
    let layer = reader.read_layer("cb").unwrap();
    assert_eq!(layer.shape, (6, 3));
    for odd in [1usize, 3, 5] {
        assert_eq!(
            layer.indptr[odd],
            layer.indptr[odd + 1],
            "row {odd} must be empty"
        );
    }

    let obs = reader.read_obs().unwrap();
    let status = obs.column_by_name("cb_status").unwrap();
    let status = status.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(status.value(0), "present");
    assert_eq!(status.value(1), "absent");

    // null, not 0.0: `cell_probability = 0.0` is a claim the tool never made.
    let prob = obs.column_by_name("cb_prob").unwrap();
    let prob = prob.as_any().downcast_ref::<Float32Array>().unwrap();
    assert!(!prob.is_null(0) && prob.value(0) == 0.9);
    assert!(prob.is_null(1), "an unmatched row must be null, not 0.0");
}

#[test]
fn extra_source_rows_are_skipped_and_counted_by_whether_they_carry_counts() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 3, 2, 1);

    // 3 matching + 2 extra: one with counts, one empty.
    let data = ExternalLayerData {
        row_keys: vec![
            "cell_0".into(),
            "cell_1".into(),
            "cell_2".into(),
            "ghost_a".into(),
            "ghost_b".into(),
        ],
        col_keys: keys("g", 2),
        indptr: vec![0, 1, 2, 3, 4, 4], // ghost_b contributes nothing
        indices: vec![0, 1, 0, 1],
        values: vec![1.0, 2.0, 3.0, 9.0],
        row_annotations: None,
        row_embeddings: Vec::new(),
        col_annotations: None,
        uns: None,
        source_checksum: None,
        source_name: None,
    };

    let summary = attach_external_layer(&path, &data, &opts("cb")).unwrap();
    assert_eq!(summary.n_source_rows_absent, 2);
    assert_eq!(
        summary.n_source_rows_absent_nonzero, 1,
        "only the ghost carrying counts represents discarded data"
    );
}

// ---------------------------------------------------------------------------
// Join failures
// ---------------------------------------------------------------------------

#[test]
fn zero_overlap_is_a_hard_error_naming_examples_from_both_sides() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 2, 1);
    let data = diagonal_data(keys("sampleA_cell_", 4), keys("g", 2), |i| i as f32 + 1.0);

    let err = attach_external_layer(&path, &data, &opts("cb")).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("no target row key matched"), "{msg}");
    assert!(msg.contains("cell_0"), "must show a target example: {msg}");
    assert!(
        msg.contains("sampleA_cell_0"),
        "must show a source example: {msg}"
    );
}

#[test]
fn duplicate_source_keys_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 2, 1);

    let mut data = diagonal_data(keys("cell_", 4), keys("g", 2), |i| i as f32 + 1.0);
    data.row_keys[3] = "cell_0".to_string();
    let err = attach_external_layer(&path, &data, &opts("cb")).unwrap_err();
    assert!(
        err.to_string()
            .contains("source row keys contain duplicates"),
        "{err}"
    );
}

#[test]
fn missing_and_extra_row_error_policies_are_honoured() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 2, 1);

    let partial = diagonal_data(vec!["cell_0".into(), "cell_1".into()], keys("g", 2), |i| {
        i as f32 + 1.0
    });
    let err = attach_external_layer(
        &path,
        &partial,
        &AttachLayerOptions {
            missing_row_policy: MissingRowPolicy::Error,
            ..opts("cb")
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("missing_row_policy"), "{err}");

    let extra = diagonal_data(
        vec![
            "cell_0".into(),
            "cell_1".into(),
            "cell_2".into(),
            "cell_3".into(),
            "ghost".into(),
        ],
        keys("g", 2),
        |i| i as f32 + 1.0,
    );
    let err = attach_external_layer(
        &path,
        &extra,
        &AttachLayerOptions {
            extra_row_policy: ExtraRowPolicy::Error,
            ..opts("cb")
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("extra_row_policy"), "{err}");
}

// ---------------------------------------------------------------------------
// Column axis
// ---------------------------------------------------------------------------

#[test]
fn reordered_gene_axis_is_rejected_by_default_and_remaps_when_allowed() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 3, 3, 1);

    // Source column order reversed: g2, g1, g0.
    let rev_cols: Vec<String> = (0..3).rev().map(|i| format!("g{i}")).collect();
    let data = diagonal_data(keys("cell_", 3), rev_cols, |i| (i + 1) as f32);

    let err = attach_external_layer(&path, &data, &opts("cb")).unwrap_err();
    assert!(err.to_string().contains("column_axis_policy"), "{err}");

    let path2 = write_fixture(dir.path(), "b.scx", 3, 3, 1);
    let summary = attach_external_layer(
        &path2,
        &data,
        &AttachLayerOptions {
            column_axis_policy: ColumnAxisPolicy::AllowReorder,
            ..opts("cb")
        },
    )
    .unwrap();
    assert_eq!(summary.column_axis_match, ColumnAxisMatch::Reordered);

    // Source row i sits at source column i, i.e. gene g(2-i) → target column 2-i.
    let layer = ScxReader::open(&path2).unwrap().read_layer("cb").unwrap();
    for i in 0..3usize {
        let (s, e) = (layer.indptr[i] as usize, layer.indptr[i + 1] as usize);
        assert_eq!(
            &layer.indices[s..e],
            &[(2 - i) as i32],
            "row {i} column remap"
        );
    }
}

#[test]
fn a_source_gene_absent_from_the_target_is_always_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 3, 2, 1);
    let data = diagonal_data(
        keys("cell_", 3),
        vec!["g0".into(), "g_unknown".into()],
        |i| i as f32 + 1.0,
    );

    for policy in [
        ColumnAxisPolicy::RequireIdentical,
        ColumnAxisPolicy::AllowReorder,
        ColumnAxisPolicy::AllowSubset,
    ] {
        let err = attach_external_layer(
            &path,
            &data,
            &AttachLayerOptions {
                column_axis_policy: policy,
                ..opts("cb")
            },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("absent from the target var axis"),
            "{policy:?}: {err}"
        );
    }
}

// ---------------------------------------------------------------------------
// Re-import / overwrite
// ---------------------------------------------------------------------------

#[test]
fn reimport_without_overwrite_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 2, 1);
    let data = diagonal_data(keys("cell_", 4), keys("g", 2), |i| (i + 1) as f32);

    attach_external_layer(&path, &data, &opts("cb")).unwrap();
    let err = attach_external_layer(&path, &data, &opts("cb")).unwrap_err();
    assert!(err.to_string().contains("already exists"), "{err}");
}

/// The writer's duplicate-section guard is inert under an adopted writer, so
/// without an explicit stem-based drop the catalog keeps both shard families
/// and `read_layer` silently returns a 2x-tall matrix.
#[test]
fn reimport_with_overwrite_leaves_exactly_one_shard_family() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 8, 2, 2);
    let first = diagonal_data(keys("cell_", 8), keys("g", 2), |i| (i + 1) as f32);
    let second = diagonal_data(keys("cell_", 8), keys("g", 2), |i| (i + 100) as f32);

    attach_external_layer(&path, &first, &opts("cb")).unwrap();
    let n_x_shards = ScxReader::open(&path)
        .unwrap()
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CsrShard)
        .count();

    attach_external_layer(
        &path,
        &second,
        &AttachLayerOptions {
            overwrite: true,
            ..opts("cb")
        },
    )
    .unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let layer_shards = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::LayerCsrShard)
        .count();
    assert_eq!(
        layer_shards, n_x_shards,
        "the stale shard family was not dropped"
    );

    let layer = reader.read_layer("cb").unwrap();
    assert_eq!(layer.shape, (8, 2), "layer must not double in height");
    assert_eq!(layer.data[0], 100.0, "the second import's values must win");
}

/// A layer literally named `pca_shard` owns `pca_shard_shard_0`; importing a
/// layer named `pca` must not claim it.
#[test]
fn attaching_a_layer_does_not_clobber_a_prefix_sibling() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 2, 1);
    let data = diagonal_data(keys("cell_", 4), keys("g", 2), |i| (i + 1) as f32);

    attach_external_layer(&path, &data, &opts("pca_shard")).unwrap();
    attach_external_layer(
        &path,
        &data,
        &AttachLayerOptions {
            status_column: Some("pca_status".to_string()),
            ..opts("pca")
        },
    )
    .unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let mut names = reader.layer_names();
    names.sort();
    assert_eq!(names, vec!["pca".to_string(), "pca_shard".to_string()]);
    assert_eq!(reader.read_layer("pca_shard").unwrap().shape, (4, 2));
}

// ---------------------------------------------------------------------------
// Preservation
// ---------------------------------------------------------------------------

#[test]
fn deletion_vectors_and_the_provenance_chain_survive() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 6, 2, 2);
    crate::mark_deleted(&path, &[1, 3]).unwrap();

    let data = diagonal_data(keys("cell_", 6), keys("g", 2), |i| (i + 1) as f32);
    attach_external_layer(&path, &data, &opts("cb")).unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let keep = reader.deletion_keep_mask().unwrap().expect("DV survives");
    assert!(!keep[1] && !keep[3] && keep[0]);

    // create → mark_deleted → test_import: every prior entry must survive.
    let prov = reader.read_provenance().unwrap();
    let actions: Vec<&str> = prov.operations.iter().map(|o| o.action.as_str()).collect();
    assert_eq!(
        actions,
        vec!["create", "delete", "test_import"],
        "the chain must be appended, not replaced"
    );
}

#[test]
fn provenance_records_the_source_checksum() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 3, 2, 1);
    let mut data = diagonal_data(keys("cell_", 3), keys("g", 2), |i| (i + 1) as f32);
    data.source_checksum = Some([7u8; 32]);

    attach_external_layer(&path, &data, &opts("cb")).unwrap();
    let prov = ScxReader::open(&path).unwrap().read_provenance().unwrap();
    let last = prov.operations.last().unwrap();
    assert_eq!(last.input_checksums, vec![[7u8; 32]]);
    assert!(
        last.params_json.contains("\"n_matched\":3"),
        "{}",
        last.params_json
    );
}

#[test]
fn rollback_restores_the_pre_import_state() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 2, 1);
    let data = diagonal_data(keys("cell_", 4), keys("g", 2), |i| (i + 1) as f32);

    attach_external_layer(&path, &data, &opts("cb")).unwrap();
    assert_eq!(
        ScxReader::open(&path).unwrap().layer_names(),
        vec!["cb".to_string()]
    );

    crate::rollback(&path).unwrap();

    let reader = ScxReader::open(&path).unwrap();
    assert!(reader.layer_names().is_empty(), "layer survived a rollback");
    assert!(reader
        .read_obs()
        .unwrap()
        .column_by_name("cb_status")
        .is_none());
}

#[test]
fn data_generation_is_untouched_so_a_csc_sidecar_stays_fresh() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 2, 1);
    let before = ScxReader::open(&path).unwrap().catalog().data_generation;

    let data = diagonal_data(keys("cell_", 4), keys("g", 2), |i| (i + 1) as f32);
    attach_external_layer(&path, &data, &opts("cb")).unwrap();

    let after = ScxReader::open(&path).unwrap().catalog().data_generation;
    assert_eq!(
        after, before,
        "bumping data_generation would silently invalidate the CSC sidecar"
    );
}

// ---------------------------------------------------------------------------
// Encoding + validation
// ---------------------------------------------------------------------------

#[test]
fn integer_values_pick_a_narrow_uint_and_floats_pick_float32() {
    let dir = tempfile::tempdir().unwrap();

    let path = write_fixture(dir.path(), "int.scx", 3, 2, 1);
    let ints = diagonal_data(keys("cell_", 3), keys("g", 2), |i| (i + 1) as f32);
    let s = attach_external_layer(&path, &ints, &opts("cb")).unwrap();
    assert_eq!(s.value_encoding, ValueEncoding::Uint8);

    let path = write_fixture(dir.path(), "flt.scx", 3, 2, 1);
    let floats = diagonal_data(keys("cell_", 3), keys("g", 2), |i| i as f32 + 0.5);
    let s = attach_external_layer(&path, &floats, &opts("cb")).unwrap();
    assert_eq!(s.value_encoding, ValueEncoding::Float32);
}

#[test]
fn negative_and_non_finite_values_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    for (i, bad) in [-1.0f32, f32::NAN, f32::INFINITY].into_iter().enumerate() {
        let path = write_fixture(dir.path(), &format!("x{i}.scx"), 3, 2, 1);
        let data = diagonal_data(keys("cell_", 3), keys("g", 2), move |r| {
            if r == 1 {
                bad
            } else {
                1.0
            }
        });
        let err = attach_external_layer(&path, &data, &opts("cb")).unwrap_err();
        assert!(
            err.to_string().contains("finite and non-negative"),
            "{bad}: {err}"
        );
    }
}

#[test]
fn reserved_and_slashed_layer_names_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 3, 2, 1);
    let data = diagonal_data(keys("cell_", 3), keys("g", 2), |i| (i + 1) as f32);

    for bad in ["X", "raw", "a/b", ""] {
        let err = attach_external_layer(&path, &data, &opts(bad)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("reserved") || msg.contains("must not contain") || msg.contains("empty"),
            "{bad}: {msg}"
        );
    }
}

#[test]
fn an_unknown_explicit_key_column_is_rejected_with_the_candidates_listed() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 3, 2, 1);
    let data = diagonal_data(keys("cell_", 3), keys("g", 2), |i| (i + 1) as f32);

    let err = attach_external_layer(
        &path,
        &data,
        &AttachLayerOptions {
            obs_key_column: Some("nope".to_string()),
            ..opts("cb")
        },
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("not found"), "{msg}");
    assert!(msg.contains("barcode"), "must list the candidates: {msg}");
}

#[test]
fn var_annotations_are_written_and_uns_is_merged() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 3, 2, 1);
    let mut data = diagonal_data(keys("cell_", 3), keys("g", 2), |i| (i + 1) as f32);
    data.col_annotations = Some(
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "cb_ambient",
                DataType::Float32,
                true,
            )])),
            vec![Arc::new(Float32Array::from(vec![0.1f32, 0.2]))],
        )
        .unwrap(),
    );
    data.uns = Some(serde_json::json!({"version": 1}));

    attach_external_layer(
        &path,
        &data,
        &AttachLayerOptions {
            uns_key: Some("cellbender".to_string()),
            ..opts("cb")
        },
    )
    .unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let var = reader.read_var().unwrap();
    let amb = var.column_by_name("cb_ambient").unwrap();
    let amb = amb.as_any().downcast_ref::<Float32Array>().unwrap();
    assert_eq!(amb.value(0), 0.1);
    assert_eq!(amb.value(1), 0.2);

    let uns = reader.read_uns().unwrap();
    assert_eq!(uns["cellbender"]["version"], 1);
    assert_eq!(uns["state"], "v0", "pre-existing uns keys must survive");
}

#[test]
fn shape_mismatches_are_caught_before_any_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 4, 2, 1);
    let before = std::fs::read(&path).unwrap();

    let mut data = diagonal_data(keys("cell_", 4), keys("g", 2), |i| (i + 1) as f32);
    data.indptr.pop(); // now inconsistent with row_keys

    assert!(attach_external_layer(&path, &data, &opts("cb")).is_err());
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "a rejected import must leave the file byte-identical"
    );
}
