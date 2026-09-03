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

use arrow::array::{Array, Float32Array, Int64Array, RecordBatch, StringArray};
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

/// How a fixture lays out its obs axis. See the twin in `external_obs_tests.rs`
/// — a `Single` section has no per-shard reader, so it exercises the
/// materialising fallback and nothing else.
#[derive(Clone, Copy, Debug)]
enum ObsLayout {
    Single,
    /// `ObsMetadataShard` sections with exactly these per-shard row counts.
    Shards(&'static [usize]),
}

/// An SCX file with `n_shards` X shards tiling `[0, n_obs)`. X itself is empty
/// (the op never reads it) but the shard *ranges* are what the layer must copy.
fn write_fixture(dir: &Path, name: &str, n_obs: usize, n_vars: usize, n_shards: usize) -> PathBuf {
    write_fixture_with_layout(dir, name, n_obs, n_vars, n_shards, ObsLayout::Single)
}

fn write_fixture_with_layout(
    dir: &Path,
    name: &str,
    n_obs: usize,
    n_vars: usize,
    n_shards: usize,
    obs_layout: ObsLayout,
) -> PathBuf {
    let path = dir.join(name);
    let rows_per = n_obs.div_ceil(n_shards);
    let header =
        FileHeader::new_single_modality(n_obs as u64, n_vars as u64, 0, rows_per as u32, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    let obs = obs_batch(n_obs);
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
        uns: serde_json::Map::new(),
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
        uns: serde_json::Map::new(),
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
        uns: serde_json::Map::new(),
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

/// Regression: an explicit zero followed by a duplicate of the same column.
///
/// A hand-rolled canonicalizer here skipped the zero without emitting it but
/// still advanced its "previous column" marker, so the duplicate accumulated
/// onto whatever was emitted last — a different column, or even the previous
/// row. Row `[(1, 1.0), (2, 0.0), (2, 3.0)]` came out as `[(1, 4.0)]` instead
/// of `[(1, 1.0), (2, 3.0)]`. Now delegated to `scx_sparse::canonicalize_csr`,
/// which dedup-sums before dropping zeros.
#[test]
fn explicit_zero_before_a_duplicate_column_does_not_corrupt_the_row() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 2, 3, 1);

    // Row 0: col 1 = 1.0, then col 2 = 0.0, then col 2 = 3.0 (out of order too).
    // Row 1: a plain single entry, to catch cross-row bleed.
    let data = ExternalLayerData {
        row_keys: vec!["cell_0".into(), "cell_1".into()],
        col_keys: keys("g", 3),
        indptr: vec![0, 3, 4],
        indices: vec![2, 1, 2, 0],
        values: vec![0.0, 1.0, 3.0, 7.0],
        row_annotations: None,
        row_embeddings: Vec::new(),
        col_annotations: None,
        uns: serde_json::Map::new(),
        source_checksum: None,
        source_name: None,
    };

    let summary = attach_external_layer(&path, &data, &opts("cb")).unwrap();
    let layer = ScxReader::open(&path).unwrap().read_layer("cb").unwrap();

    let row0 = |i: usize| (layer.indptr[i] as usize, layer.indptr[i + 1] as usize);
    let (s0, e0) = row0(0);
    assert_eq!(&layer.indices[s0..e0], &[1, 2], "row 0 columns");
    assert_eq!(&layer.data[s0..e0], &[1.0, 3.0], "row 0 values");

    let (s1, e1) = row0(1);
    assert_eq!(&layer.indices[s1..e1], &[0], "row 1 must be untouched");
    assert_eq!(&layer.data[s1..e1], &[7.0]);

    // And the reported nnz counts what was written, not the pre-dedup input.
    assert_eq!(summary.layer_nnz, 3, "explicit zero must not be counted");
}

/// When every candidate var key fails, the reported error must be the *first*
/// candidate's — the caller's preferred key — not the last fallback's.
///
/// Telling those apart needs the two candidates to fail with *different* error
/// kinds; a target where both merely fail to overlap produces the same message
/// either way and would pass even with last-error behaviour. So `gene_id` (the
/// preferred candidate) carries duplicates and fails with `DuplicateJoinKey`,
/// while `gene_name` is well-formed but has no overlap and fails with
/// `AxisMismatch`.
#[test]
fn var_key_retry_reports_the_first_candidate_failure() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two_var_keys.scx");
    let header = FileHeader::new_single_modality(3, 2, 0, 16384, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&obs_batch(3)).unwrap();
    let var = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("gene_id", DataType::Utf8, false),
            Field::new("gene_name", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["dup", "dup"])),
            Arc::new(StringArray::from(vec!["SYM0", "SYM1"])),
        ],
    )
    .unwrap();
    writer.write_var(&var).unwrap();
    writer
        .write_csr_shard(&[0u64; 4], &[], &[], CodecId::None, ValueEncoding::Uint8, 0)
        .unwrap();
    writer.finish().unwrap();

    // Overlaps neither candidate, so both are tried and both fail.
    let data = diagonal_data(
        keys("cell_", 3),
        vec!["nope_a".into(), "nope_b".into()],
        |i| i as f32 + 1.0,
    );

    let msg = attach_external_layer(&path, &data, &opts("cb"))
        .unwrap_err()
        .to_string();
    assert!(
        msg.contains("duplicates"),
        "expected the preferred key's (gene_id) duplicate error, got: {msg}"
    );
    assert!(
        !msg.contains("absent from the target var axis"),
        "reported the last candidate's error instead of the first: {msg}"
    );
}

/// A column index past the declared column-key count is rejected up front,
/// rather than panicking inside the gather's `col_map[...]` lookup.
#[test]
fn out_of_range_column_index_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 2, 3, 1);
    let before = std::fs::read(&path).unwrap();

    let mut data = diagonal_data(keys("cell_", 2), keys("g", 3), |i| (i + 1) as f32);
    data.indices[0] = 99; // only 3 column keys exist

    let err = attach_external_layer(&path, &data, &opts("cb")).unwrap_err();
    assert!(err.to_string().contains("out of range"), "{err}");
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// A dry run must predict the same nnz the real import writes, including when
/// the source carries explicit zeros or duplicate coordinates.
#[test]
fn dry_run_nnz_matches_the_real_import() {
    let dir = tempfile::tempdir().unwrap();
    let messy = |p: &std::path::Path| ExternalLayerData {
        row_keys: vec!["cell_0".into(), "cell_1".into()],
        col_keys: keys("g", 3),
        indptr: vec![0, 3, 4],
        indices: vec![2, 1, 2, 0],
        values: vec![0.0, 1.0, 3.0, 7.0],
        row_annotations: None,
        row_embeddings: Vec::new(),
        col_annotations: None,
        uns: serde_json::Map::new(),
        source_checksum: None,
        source_name: Some(p.display().to_string()),
    };

    let dry_path = write_fixture(dir.path(), "dry.scx", 2, 3, 1);
    let dry = attach_external_layer(
        &dry_path,
        &messy(&dry_path),
        &AttachLayerOptions {
            dry_run: true,
            ..opts("cb")
        },
    )
    .unwrap();

    let real_path = write_fixture(dir.path(), "real.scx", 2, 3, 1);
    let real = attach_external_layer(&real_path, &messy(&real_path), &opts("cb")).unwrap();

    assert_eq!(
        dry.layer_nnz, real.layer_nnz,
        "a dry run must not overstate nnz relative to the import it previews"
    );
    assert_eq!(real.layer_nnz, 3);
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
    data.uns
        .insert("cellbender".to_string(), serde_json::json!({"version": 1}));

    attach_external_layer(&path, &data, &opts("cb")).unwrap();

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

// ---------------------------------------------------------------------------
// Predicate index
// ---------------------------------------------------------------------------

/// A fixture whose obs carries a real predicate index over `cell_type`.
fn fixture_with_obs_index(dir: &Path, name: &str) -> PathBuf {
    let n_obs = 4usize;
    let n_vars = 2usize;
    let obs = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("barcode", DataType::Utf8, false),
            Field::new("cell_type", DataType::Utf8, false),
            Field::new("n_counts", DataType::Int64, false),
        ])),
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

    let build_opts = scx_engine::PredicateIndexBuildOptions {
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
        &build_opts,
        &mut outcomes,
        &mut named,
    )
    .unwrap()
    .expect("cell_type must be indexable");
    writer.write_obs_predicate_index(&bytes).unwrap();
    // The per-shard catalog stats the index implies. Level-1 pruning reads these,
    // not the index section, so a fixture without them cannot observe a stale
    // bound at all — see the matching note in `external_obs_tests.rs`.
    scx_engine::apply_obs_shard_column_stats(&mut writer, &bytes, ranges.len()).unwrap();
    writer
        .write_uns(&serde_json::json!({"state": "v0"}))
        .unwrap();
    writer.finish().unwrap();
    path
}

fn has_obs_index(path: &Path) -> bool {
    ScxReader::open(path)
        .unwrap()
        .catalog()
        .entries
        .iter()
        .any(|e| e.section_type == SectionType::ObsPredicateIndex)
}

/// A pure add cannot invalidate an index keyed on other columns, so pushdown
/// must survive — dropping it here would be a silent performance cliff on every
/// CellBender import.
#[test]
fn layer_import_keeps_the_obs_predicate_index_on_a_pure_add() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture_with_obs_index(dir.path(), "a.scx");
    assert!(has_obs_index(&path));

    let data = diagonal_data(keys("cell_", 4), keys("g", 2), |i| (i + 1) as f32);
    attach_external_layer(&path, &data, &opts("cb")).unwrap();

    assert!(has_obs_index(&path));
}

/// The converse: overwriting a column the index covers leaves its entries
/// describing values that no longer exist, so the index must go. Without this
/// the index would still map "T"/"B" to shard ranges whose rows now read "NK",
/// and `query().filter_obs("cell_type == 'T'")` would return them.
#[test]
fn layer_import_drops_a_stale_obs_predicate_index_on_overwrite() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture_with_obs_index(dir.path(), "a.scx");
    assert!(has_obs_index(&path));

    let mut data = diagonal_data(keys("cell_", 4), keys("g", 2), |i| (i + 1) as f32);
    data.row_annotations = Some(
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "cell_type",
                DataType::Utf8,
                true,
            )])),
            vec![Arc::new(StringArray::from(vec!["NK", "NK", "NK", "NK"]))],
        )
        .unwrap(),
    );

    attach_external_layer(
        &path,
        &data,
        &AttachLayerOptions {
            layer_name: "cb".to_string(),
            overwrite: true,
            provenance_action: "test_import".to_string(),
            ..Default::default()
        },
    )
    .unwrap();

    assert!(
        !has_obs_index(&path),
        "the index still covers 'cell_type', whose values were just replaced"
    );
}

use crate::test_utils::columns_with_shard_stats;

/// Dropping the stale index is only half of it: the catalog's per-shard
/// `ColumnStat`s are what Level-1 pruning reads, and the numeric `MinMax` arm
/// never consults the index at all. An overwrite that leaves them behind
/// excludes shards on bounds describing values CellBender just replaced.
///
/// Scoped to the overwritten column — `cell_type` keeps its stats, because a
/// CellBender import joins by barcode and does not touch it.
#[test]
fn layer_import_clears_shard_column_stats_for_the_overwritten_column_only() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture_with_obs_index(dir.path(), "a.scx");
    assert_eq!(
        columns_with_shard_stats(&path, &["cell_type", "n_counts"]),
        vec!["cell_type".to_string(), "n_counts".to_string()],
        "fixture must start with stats for both indexed columns"
    );

    let mut data = diagonal_data(keys("cell_", 4), keys("g", 2), |i| (i + 1) as f32);
    data.row_annotations = Some(
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "n_counts",
                DataType::Int64,
                true,
            )])),
            vec![Arc::new(Int64Array::from(vec![100i64, 200, 300, 400]))],
        )
        .unwrap(),
    );

    attach_external_layer(
        &path,
        &data,
        &AttachLayerOptions {
            layer_name: "cb".to_string(),
            overwrite: true,
            provenance_action: "test_import".to_string(),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(
        columns_with_shard_stats(&path, &["cell_type", "n_counts"]),
        vec!["cell_type".to_string()],
        "the overwritten column's stats must go, and only that column's"
    );
    let n = scx_engine::QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("n_counts > 150")
        .unwrap()
        .count()
        .unwrap()
        .matched_rows;
    assert_eq!(
        n, 3,
        "200/300/400 match; the pre-import bounds (10–40) would exclude every shard"
    );
}

// ---------------------------------------------------------------------------
// E2 (layer twin) — a cell-called subset of an all-droplet target is normal
// ---------------------------------------------------------------------------

/// CellBender's `_filtered.h5` holds only the droplets it called as cells, so a
/// partial match against the raw all-droplet target is the *expected* shape, not
/// a defect. It must not be reported with the example pair that signals a
/// barcode-format mismatch.
#[test]
fn layer_partial_coverage_reports_coverage_at_info() {
    let (level, msg) = super::layer_join_coverage_report(
        8_000,
        50_000,
        42_000,
        0, // every CellBender row matched a raw droplet
        0,
        || panic!("the coverage branch must not extract key examples"),
    )
    .expect("below half, so something must be reported");

    assert_eq!(level, log::Level::Info);
    assert!(!msg.contains("only"), "{msg}");
    assert!(!msg.contains("examples"), "{msg}");
    assert!(msg.contains("coverage") && msg.contains("8000"), "{msg}");
    assert!(msg.contains("not a key mismatch"), "{msg}");
}

/// Unmatched source rows are a real signal here — a CellBender row with no raw
/// droplet means the barcodes disagree — and the count that *carries counts* is
/// the load-bearing number, so it must survive.
#[test]
fn layer_unmatched_source_rows_warn_with_the_nonzero_count() {
    let (level, msg) = super::layer_join_coverage_report(100, 1000, 900, 50, 37, || {
        (vec!["AAACCC-1".to_string()], vec!["AAACCC".to_string()])
    })
    .expect("below half, so something must be reported");

    assert_eq!(level, log::Level::Warn);
    assert!(msg.contains("only"), "{msg}");
    assert!(
        msg.contains("37 of them carry counts"),
        "the nonzero-source count is the load-bearing detail: {msg}"
    );
    assert!(msg.contains("AAACCC"), "examples must appear: {msg}");
}

#[test]
fn layer_high_coverage_reports_nothing() {
    assert!(
        super::layer_join_coverage_report(500, 1000, 500, 0, 0, || panic!("must not be called"))
            .is_none()
    );
}

/// Examples must come from the keys that did **not** match.
///
/// The scenario is the report's: a source whose first N keys match and whose
/// remainder carry a stray `-1` suffix. Head examples print two identical
/// matching keys — "both lists are valid keys drawn from the same space", which
/// is what made the pair actively confusing. Unmatched examples put `"c100"`
/// beside `"c100-1"` and the suffix is self-evident.
#[test]
fn unmatched_examples_shows_the_keys_that_failed() {
    use super::{examples, unmatched_examples};

    let keys: Vec<String> = (0..6).map(|i| format!("c{i}")).collect();
    let matched = vec![true, true, true, false, false, false];

    assert_eq!(
        unmatched_examples(&keys, &matched),
        vec!["c3", "c4", "c5"],
        "must pick the failures, not the head"
    );
    // Anti-vacuous: the head examples this replaces are the *matching* keys, so
    // the two functions must genuinely differ on this input.
    assert_ne!(
        unmatched_examples(&keys, &matched),
        examples(&keys),
        "if these agree the change is a no-op on the case it targets"
    );
}

/// When everything matched there is nothing to show, so fall back to the head
/// rather than printing an empty list.
#[test]
fn unmatched_examples_falls_back_when_all_matched() {
    use super::{examples, unmatched_examples};

    let keys: Vec<String> = (0..4).map(|i| format!("c{i}")).collect();
    let all = vec![true; 4];
    assert_eq!(unmatched_examples(&keys, &all), examples(&keys));
    assert!(!unmatched_examples(&keys, &all).is_empty());
}

// ---------------------------------------------------------------------------
// Sharded obs — the streaming rewrite
//
// Every other fixture here writes a single legacy `ObsMetadata` section, which
// has no per-shard reader and so exercises only the materialising fallback.
// These pin the streaming path. See the twin block in `external_obs_tests.rs`.
// ---------------------------------------------------------------------------

/// 4/2/4 over 10 rows, deliberately not `header.shard_target_rows`. Cumulative
/// starts are 0/4/6; a driver deriving them as `shard_idx * shard_target_rows`
/// would get 0/4/8 and shift everything in the last shard.
const UNEVEN: &[usize] = &[4, 2, 4];

fn sharded_fixture(dir: &Path, name: &str) -> PathBuf {
    write_fixture_with_layout(dir, name, 10, 3, 2, ObsLayout::Shards(UNEVEN))
}

/// Reverse-ordered source whose per-row annotation and row-sum both encode the
/// row's own target index, so a row-range error in the per-shard driver shows up
/// as a wrong *value* rather than a wrong row count.
fn positional_probe(n: usize) -> ExternalLayerData {
    let rev: Vec<String> = (0..n).rev().map(|i| format!("cell_{i}")).collect();
    let mut indptr = vec![0u64];
    let (mut indices, mut values) = (Vec::new(), Vec::new());
    let mut probs = Vec::new();
    for key in &rev {
        let target: usize = key.strip_prefix("cell_").unwrap().parse().unwrap();
        indices.push((target % 3) as u32);
        values.push(100.0 + target as f32);
        probs.push(target as f32);
        indptr.push(indices.len() as u64);
    }
    let schema = Schema::new(vec![Field::new("cb_prob", DataType::Float32, true)]);
    let ann =
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(Float32Array::from(probs))]).unwrap();
    ExternalLayerData {
        row_keys: rev,
        col_keys: keys("g", 3),
        indptr,
        indices,
        values,
        row_annotations: Some(ann),
        row_embeddings: Vec::new(),
        col_annotations: None,
        uns: serde_json::Map::new(),
        source_checksum: None,
        source_name: None,
    }
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

#[test]
fn layer_obs_columns_land_on_the_right_cell_across_uneven_shards() {
    let dir = tempfile::tempdir().unwrap();
    let path = sharded_fixture(dir.path(), "a.scx");

    let mut o = opts("cb");
    o.row_sum_column = Some("cb_sum".to_string());
    attach_external_layer(&path, &positional_probe(10), &o).unwrap();

    let obs = ScxReader::open(&path).unwrap().read_obs().unwrap();
    assert_eq!(obs.num_rows(), 10);
    let probs = f32_col(&obs, "cb_prob");
    let sums = f32_col(&obs, "cb_sum");
    for i in 0..10 {
        assert_eq!(
            probs.value(i),
            i as f32,
            "row {i} got the annotation for row {}",
            probs.value(i)
        );
        assert_eq!(
            sums.value(i),
            100.0 + i as f32,
            "row {i} got the row sum for row {}",
            sums.value(i) - 100.0
        );
    }
}

/// The op must not assemble the whole obs table on a sharded target. Asserted on
/// the reader the op actually used.
#[test]
fn sharded_layer_import_never_materializes_obs() {
    let dir = tempfile::tempdir().unwrap();
    let path = sharded_fixture(dir.path(), "a.scx");
    let data = diagonal_data(keys("cell_", 10), keys("g", 3), |i| (i + 1) as f32);

    let reader = ScxReader::open(&path).unwrap();
    let s = attach_external_layer_with_reader(&path, &reader, &data, &opts("cb")).unwrap();
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
            "the join must have gone through the projected key read"
        );
        let _ = reader.read_obs().unwrap();
        assert_eq!(reader.debug_counts().read_obs.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn layer_import_preserves_obs_shard_boundaries() {
    let dir = tempfile::tempdir().unwrap();
    let path = sharded_fixture(dir.path(), "a.scx");
    let data = diagonal_data(keys("cell_", 10), keys("g", 3), |i| (i + 1) as f32);

    attach_external_layer(&path, &data, &opts("cb")).unwrap();

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

/// One implementation seen from two angles: same source, same obs content, one
/// sharded target and one single-section twin.
#[test]
fn sharded_and_single_section_layer_imports_agree() {
    let dir = tempfile::tempdir().unwrap();
    let sharded = write_fixture_with_layout(
        dir.path(),
        "sharded.scx",
        10,
        3,
        2,
        ObsLayout::Shards(UNEVEN),
    );
    let single = write_fixture(dir.path(), "single.scx", 10, 3, 2);

    let mut o = opts("cb");
    o.row_sum_column = Some("cb_sum".to_string());
    let data = positional_probe(10);
    let a = attach_external_layer(&sharded, &data, &o).unwrap();
    let b = attach_external_layer(&single, &data, &o).unwrap();

    assert_eq!(a.n_matched, b.n_matched);
    assert_eq!(a.layer_nnz, b.layer_nnz);
    assert_eq!(a.obs_columns_added, b.obs_columns_added);
    assert!(a.obs_streamed && !b.obs_streamed);

    let oa = ScxReader::open(&sharded).unwrap().read_obs().unwrap();
    let ob = ScxReader::open(&single).unwrap().read_obs().unwrap();
    assert_eq!(oa.schema(), ob.schema(), "the two paths disagree on schema");
    for i in 0..oa.num_columns() {
        assert_eq!(
            oa.column(i),
            ob.column(i),
            "column '{}' differs between the streaming and materialising paths",
            oa.schema().field(i).name()
        );
    }
}

// ---------------------------------------------------------------------------
// Value encoding vs. canonicalization
// ---------------------------------------------------------------------------

/// One cell, one gene, the same coordinate twice — canonicalization sums them.
fn duplicate_coordinate_data(a: f32, b: f32) -> ExternalLayerData {
    ExternalLayerData {
        row_keys: vec!["cell_0".to_string()],
        col_keys: vec!["g0".to_string()],
        indptr: vec![0, 2],
        indices: vec![0, 0],
        values: vec![a, b],
        row_annotations: None,
        row_embeddings: Vec::new(),
        col_annotations: None,
        uns: serde_json::Map::new(),
        source_checksum: None,
        source_name: None,
    }
}

/// The encoding is detected **once** over the whole layer, before the per-shard
/// gather; canonicalization then sums duplicate coordinates, so a shard's values
/// can outgrow it. Re-encoding under the stale encoding fails three different
/// ways and only this one is loud — `Uint8` refuses `200 + 200 = 400` outright,
/// aborting the import on exactly the input canonicalization exists to repair.
/// (`Float16` yields infinity and `Uint32` saturates at 2³², both silently.)
#[test]
fn attach_layer_widens_a_detected_encoding_the_sums_outgrow() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 1, 1, 1);

    let data = duplicate_coordinate_data(200.0, 200.0);
    let summary = attach_external_layer(&path, &data, &opts("cb"))
        .expect("a duplicate-coordinate sum past the detected encoding must widen, not abort");

    assert_eq!(
        summary.value_encoding,
        ValueEncoding::Uint16,
        "the summary must report the encoding actually written"
    );
    let layer = ScxReader::open(&path).unwrap().read_layer("cb").unwrap();
    assert_eq!(
        layer.data,
        vec![400.0],
        "the summed value must survive intact, not saturate or error"
    );
}

/// A caller-pinned encoding is authoritative: widening it silently would
/// override a deliberate choice. Report that it no longer holds instead — which
/// is also the only way the `Uint32` and `Float16` arms stop being silent, since
/// neither errors on its own.
#[test]
fn attach_layer_rejects_a_pinned_encoding_the_sums_outgrow() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 1, 1, 1);

    let data = duplicate_coordinate_data(200.0, 200.0);
    let pinned = AttachLayerOptions {
        value_encoding: Some(ValueEncoding::Uint8),
        ..opts("cb")
    };
    let err = attach_external_layer(&path, &data, &pinned)
        .expect_err("a pinned encoding the canonicalized values outgrow must be reported");
    let msg = err.to_string();
    assert!(
        msg.contains("cb") && msg.contains("Uint8") && msg.contains("400"),
        "the error must name the layer, the pinned encoding and the offending value, got: {msg}"
    );
}

/// The 2³² ambiguity is irreducible for the *rewrite* paths — there an f32 of
/// 2³² is equally a decoded on-disk `u32::MAX`, so `encoding_for_canonicalized`
/// deliberately keeps `Uint32` and lets `as u32` saturate it back. It is **not**
/// ambiguous here: `ExternalLayerData` carries fresh values that never passed
/// through a `u32` decode, so a sum of `2³¹ + 2³¹` is exactly 2³² and nothing
/// else. Encoding it as `Uint32` writes `u32::MAX`, one less than the sum, with
/// no error — the same silent corruption this PR exists to remove, on the one
/// path that has the evidence to avoid it.
///
/// Read back through `f32` the two are indistinguishable (`u32::MAX as f32` is
/// 2³²), so the encoding is what has to be asserted.
#[test]
fn attach_layer_uses_the_fresh_data_rule_at_the_u32_ceiling() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 1, 1, 1);

    let two_pow_31 = 2_147_483_648.0f32;
    let data = duplicate_coordinate_data(two_pow_31, two_pow_31);
    let summary = attach_external_layer(&path, &data, &opts("cb")).unwrap();
    assert_eq!(
        summary.value_encoding,
        ValueEncoding::Float32,
        "a fresh sum above u32::MAX must not be written as Uint32 and saturated"
    );

    // The written header, not just the summary: keeping the two in sync while
    // passing the layer-wide encoding to the encoder would re-saturate the shard
    // and still satisfy the assertion above.
    let reader = ScxReader::open(&path).unwrap();
    let entry = reader
        .catalog()
        .entries
        .iter()
        .find(|e| e.section_type == SectionType::LayerCsrShard)
        .expect("the layer shard must exist");
    let sh = reader.read_shard_header(entry).unwrap();
    assert_eq!(
        ValueEncoding::from_u8(sh.value_encoding),
        Some(ValueEncoding::Float32),
        "the shard on disk must carry the widened encoding, not just the summary"
    );
}

/// The preview has to predict the boundary case too — this is the third of the
/// trio (detected / pinned / dry-run), and the one whose absence would let a
/// future split between the two loops go unnoticed.
#[test]
fn dry_run_predicts_the_encoding_at_the_u32_ceiling() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 1, 1, 1);

    let two_pow_31 = 2_147_483_648.0f32;
    let data = duplicate_coordinate_data(two_pow_31, two_pow_31);
    let preview = AttachLayerOptions {
        dry_run: true,
        ..opts("cb")
    };
    let summary = attach_external_layer(&path, &data, &preview).unwrap();
    assert_eq!(
        summary.value_encoding,
        ValueEncoding::Float32,
        "the preview must report the encoding the write will actually use"
    );

    let pinned_preview = AttachLayerOptions {
        dry_run: true,
        value_encoding: Some(ValueEncoding::Uint32),
        ..opts("cb")
    };
    assert!(
        attach_external_layer(&path, &data, &pinned_preview).is_err(),
        "a preview must surface the pinned failure at the ceiling, not defer it"
    );
}

/// The pinned branch has to answer the same question the same way: at 2³² the
/// values genuinely do not fit `Uint32`, so a caller who pinned it must be told,
/// not handed a silently saturated shard.
#[test]
fn attach_layer_rejects_a_pinned_uint32_at_the_ceiling() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 1, 1, 1);

    let two_pow_31 = 2_147_483_648.0f32;
    let data = duplicate_coordinate_data(two_pow_31, two_pow_31);
    let pinned = AttachLayerOptions {
        value_encoding: Some(ValueEncoding::Uint32),
        ..opts("cb")
    };
    assert!(
        attach_external_layer(&path, &data, &pinned).is_err(),
        "a pinned Uint32 that the canonicalized sum outgrows must be reported"
    );
}

/// The overflow diagnostic is the only thing that makes a pinned `Float16`
/// overflow audible at all, so it has to name the value that caused it. Folding
/// the reported maximum from `0.0` with a *signed* max reports `0` for an
/// all-negative shard — the one case the magnitude guard was just added for.
#[test]
fn attach_layer_pinned_overflow_names_the_offending_value() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 1, 1, 1);

    let data = duplicate_coordinate_data(-40_000.0, -40_000.0);
    let pinned = AttachLayerOptions {
        value_encoding: Some(ValueEncoding::Float16),
        ..opts("cb")
    };
    let err = attach_external_layer(&path, &data, &pinned)
        .expect_err("a pinned Float16 the sums outgrow must be reported");
    let msg = err.to_string();
    assert!(
        msg.contains("-80000"),
        "the diagnostic must name the value that overflowed, got: {msg}"
    );
}

/// `--dry-run` predicts the import; it has to predict this too. The write loop
/// would refuse a pinned encoding the sums outgrow, so a preview that returns
/// `Ok` sends the user into a failure it was asked to find.
#[test]
fn dry_run_reports_the_encoding_the_write_would_use() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 1, 1, 1);
    let data = duplicate_coordinate_data(200.0, 200.0);

    let preview = AttachLayerOptions {
        dry_run: true,
        ..opts("cb")
    };
    let summary = attach_external_layer(&path, &data, &preview).unwrap();
    assert_eq!(
        summary.value_encoding,
        ValueEncoding::Uint16,
        "the preview must report the widened encoding the write will use"
    );

    let pinned_preview = AttachLayerOptions {
        dry_run: true,
        value_encoding: Some(ValueEncoding::Uint8),
        ..opts("cb")
    };
    assert!(
        attach_external_layer(&path, &data, &pinned_preview).is_err(),
        "a preview must surface the pinned-encoding failure, not defer it to the write"
    );
}

#[test]
fn several_uns_keys_merge_in_one_commit_on_the_layer_op() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), "a.scx", 3, 2, 1);
    let mut data = diagonal_data(keys("cell_", 3), keys("g", 2), |i| (i + 1) as f32);
    data.uns.insert("a".to_string(), serde_json::json!(1));
    data.uns
        .insert("b".to_string(), serde_json::json!({"x": 2}));

    attach_external_layer(&path, &data, &opts("cb")).unwrap();

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(
        reader.read_uns().unwrap(),
        serde_json::json!({"state": "v0", "a": 1, "b": {"x": 2}})
    );
    let prov = reader.read_provenance().unwrap();
    assert!(
        prov.operations
            .last()
            .unwrap()
            .params_json
            .contains("\"uns_keys_merged\":[\"a\",\"b\"]"),
        "{}",
        prov.operations.last().unwrap().params_json
    );
}
