//! Phase 2g — streaming merge & append regression suite for
//! `MERGE-OBS-OFFSET-OVERFLOW.md`.
//!
//! Covers:
//! - Merge of many inputs emitting `ObsMetadataShard` sections;
//! - Convert-on-append from legacy `ObsMetadata` to sharded layout;
//! - Var identity validation (strict default vs. `assume_identical_var`);
//! - Uns policy variants (`first`, `require-equal`, `namespace`, `summary`);
//! - Provenance stamping of policy params;
//! - Predicate-index parity between batch-mode and streaming builder.
//!
//! The `>2 GiB` overflow regressions are gated `#[ignore]` and run via:
//!
//! ```bash
//! cargo test --release -p scx-ops --test streaming_merge_append -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{DictionaryArray, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use scx_codec::{CodecId, ValueEncoding};
use scx_format::header::{FileHeader, MAGIC};
use scx_format::provenance::ProvenanceEntry;
use scx_format::section::SectionType;
use scx_format::writer::ScxWriter;
use scx_format::ScxReader;

fn header(n_obs: u64, n_vars: u64) -> FileHeader {
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
        shard_target_rows: 16_384,
        codec_id: 0,
        index_dtype: 0,
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

fn obs_batch(start_row: usize, n: usize, donor: &str) -> RecordBatch {
    let cell_ids: Vec<String> = (start_row..start_row + n)
        .map(|i| format!("cell_{i:07}"))
        .collect();
    let donors: Vec<String> = std::iter::repeat(donor.to_string()).take(n).collect();
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("donor", DataType::Utf8, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(cell_ids)),
            Arc::new(StringArray::from(donors)),
        ],
    )
    .unwrap()
}

/// Standard 4-gene var, identical across inputs unless a test overrides.
fn var_batch() -> RecordBatch {
    let gene_ids = StringArray::from(vec!["g0", "g1", "g2", "g3"]);
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(gene_ids)]).unwrap()
}

/// A var batch with a different gene order — used to trigger
/// `OpsError::VarMismatch`.
fn var_batch_reordered() -> RecordBatch {
    let gene_ids = StringArray::from(vec!["g0", "g2", "g1", "g3"]);
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(gene_ids)]).unwrap()
}

fn write_zero_csr_shard(writer: &mut ScxWriter, row_start: u64, n_rows: u64) {
    let indptr: Vec<u64> = vec![0u64; (n_rows + 1) as usize];
    let indices: Vec<u32> = Vec::new();
    let values: Vec<u8> = Vec::new();
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            row_start,
        )
        .unwrap();
}

/// Build a legacy single-section SCX file with `n_obs` rows, the given
/// obs payload (`donor`), and the supplied var.
fn write_legacy_input(
    path: &std::path::Path,
    n_obs: u64,
    donor: &str,
    var: &RecordBatch,
    uns: Option<&serde_json::Value>,
) {
    let mut writer = ScxWriter::new(path, header(n_obs, 4)).unwrap();
    writer
        .write_obs(&obs_batch(0, n_obs as usize, donor))
        .unwrap();
    writer.write_var(var).unwrap();
    write_zero_csr_shard(&mut writer, 0, n_obs);
    if let Some(uns) = uns {
        writer.write_uns(uns).unwrap();
    }
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1710000000,
            action: "convert".to_string(),
            tool: "streaming_merge_append test fixture".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();
    writer.finish().unwrap();
}

#[test]
fn merge_small_inputs_emits_shards() {
    // Phase 2a: even small merges produce ObsMetadataShard sections.
    // No `ObsMetadata` single-section entry should land in the output
    // catalog (legacy single-section is an input format only).
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("in0.scx");
    let p1 = dir.path().join("in1.scx");
    let p2 = dir.path().join("in2.scx");
    let var = var_batch();
    write_legacy_input(&p0, 100, "donor_A", &var, None);
    write_legacy_input(&p1, 100, "donor_B", &var, None);
    write_legacy_input(&p2, 100, "donor_C", &var, None);
    let out = dir.path().join("merged.scx");
    scx_ops::merge(&[p0.as_path(), p1.as_path(), p2.as_path()], &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.n_obs(), 300);
    assert_eq!(reader.obs_metadata_shard_count(), 3, "one shard per input");
    // No legacy ObsMetadata section in the output.
    assert!(
        !reader
            .catalog()
            .entries
            .iter()
            .any(|e| e.section_type == SectionType::ObsMetadata),
        "merge output must not contain a single-section ObsMetadata entry"
    );
    let obs = reader.read_obs_assembled().unwrap();
    assert_eq!(obs.num_rows(), 300);
    let donors = obs
        .column_by_name("donor")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(donors.value(0), "donor_A");
    assert_eq!(donors.value(100), "donor_B");
    assert_eq!(donors.value(200), "donor_C");
}

#[test]
fn merge_var_mismatch_errors_by_default() {
    // Phase 2e default (assume_identical_var = false): var identity
    // check rejects inputs whose var rows differ, even when n_vars
    // matches.
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    write_legacy_input(&p0, 50, "donor_A", &var_batch(), None);
    write_legacy_input(&p1, 50, "donor_B", &var_batch_reordered(), None);
    let out = dir.path().join("out.scx");
    let err = scx_ops::merge(&[p0.as_path(), p1.as_path()], &out).unwrap_err();
    assert!(
        matches!(err, scx_ops::OpsError::VarMismatch { .. }),
        "expected OpsError::VarMismatch, got: {err:?}"
    );
}

#[test]
fn merge_var_mismatch_assume_identical_var_proceeds() {
    // Phase 2e `--assume-identical-var`: trust the caller and warn
    // rather than error. The merged file uses input 0's var verbatim.
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    write_legacy_input(&p0, 50, "donor_A", &var_batch(), None);
    write_legacy_input(&p1, 50, "donor_B", &var_batch_reordered(), None);
    let out = dir.path().join("out.scx");
    let opts = scx_ops::MergeOptions {
        assume_identical_var: true,
        ..Default::default()
    };
    scx_ops::merge_with_options(&[p0.as_path(), p1.as_path()], &out, &opts).unwrap();
    let reader = ScxReader::open(&out).unwrap();
    let var = reader.read_var_assembled().unwrap();
    let gene_ids = var
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(
        gene_ids.value(1),
        "g1",
        "merged var must come from input 0 (canonical order)"
    );
}

#[test]
fn merge_uns_policy_first_keeps_input_zero() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let uns_a = serde_json::json!({"colors": ["red"], "source": "A"});
    let uns_b = serde_json::json!({"colors": ["blue"], "source": "B"});
    write_legacy_input(&p0, 10, "donor_A", &var_batch(), Some(&uns_a));
    write_legacy_input(&p1, 10, "donor_B", &var_batch(), Some(&uns_b));
    let out = dir.path().join("out.scx");
    scx_ops::merge(&[p0.as_path(), p1.as_path()], &out).unwrap();
    let merged_uns = ScxReader::open(&out).unwrap().read_uns().unwrap();
    assert_eq!(merged_uns, uns_a, "default uns_policy=first keeps input 0");
}

#[test]
fn merge_uns_policy_require_equal_errors_on_conflict() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let uns_a = serde_json::json!({"params": {"k": 1}});
    let uns_b = serde_json::json!({"params": {"k": 2}});
    write_legacy_input(&p0, 10, "donor_A", &var_batch(), Some(&uns_a));
    write_legacy_input(&p1, 10, "donor_B", &var_batch(), Some(&uns_b));
    let out = dir.path().join("out.scx");
    let opts = scx_ops::MergeOptions {
        uns_policy: scx_ops::UnsPolicy::RequireEqual,
        ..Default::default()
    };
    let err = scx_ops::merge_with_options(&[p0.as_path(), p1.as_path()], &out, &opts).unwrap_err();
    assert!(
        matches!(err, scx_ops::OpsError::UnsConflict { .. }),
        "expected OpsError::UnsConflict, got: {err:?}"
    );
}

#[test]
fn merge_uns_policy_require_equal_passes_when_identical() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let uns = serde_json::json!({"shared": true});
    write_legacy_input(&p0, 10, "donor_A", &var_batch(), Some(&uns));
    write_legacy_input(&p1, 10, "donor_B", &var_batch(), Some(&uns));
    let out = dir.path().join("out.scx");
    let opts = scx_ops::MergeOptions {
        uns_policy: scx_ops::UnsPolicy::RequireEqual,
        ..Default::default()
    };
    scx_ops::merge_with_options(&[p0.as_path(), p1.as_path()], &out, &opts).unwrap();
    let merged = ScxReader::open(&out).unwrap().read_uns().unwrap();
    assert_eq!(merged, uns);
}

#[test]
fn merge_uns_policy_namespace_wraps_each_input() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let uns_a = serde_json::json!({"colors": ["red"]});
    let uns_b = serde_json::json!({"colors": ["blue"]});
    write_legacy_input(&p0, 10, "donor_A", &var_batch(), Some(&uns_a));
    write_legacy_input(&p1, 10, "donor_B", &var_batch(), Some(&uns_b));
    let out = dir.path().join("out.scx");
    let opts = scx_ops::MergeOptions {
        uns_policy: scx_ops::UnsPolicy::Namespace,
        ..Default::default()
    };
    scx_ops::merge_with_options(&[p0.as_path(), p1.as_path()], &out, &opts).unwrap();
    let merged = ScxReader::open(&out).unwrap().read_uns().unwrap();
    let obj = merged.as_object().unwrap();
    assert_eq!(obj.get("input_0"), Some(&uns_a));
    assert_eq!(obj.get("input_1"), Some(&uns_b));
}

#[test]
fn merge_uns_policy_summary_records_conflicts() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let uns_a = serde_json::json!({"params": 1});
    let uns_b = serde_json::json!({"params": 2});
    write_legacy_input(&p0, 10, "donor_A", &var_batch(), Some(&uns_a));
    write_legacy_input(&p1, 10, "donor_B", &var_batch(), Some(&uns_b));
    let out = dir.path().join("out.scx");
    let opts = scx_ops::MergeOptions {
        uns_policy: scx_ops::UnsPolicy::Summary,
        ..Default::default()
    };
    scx_ops::merge_with_options(&[p0.as_path(), p1.as_path()], &out, &opts).unwrap();
    let merged = ScxReader::open(&out).unwrap().read_uns().unwrap();
    let obj = merged.as_object().unwrap();
    // Canonical body from input 0 is preserved.
    assert_eq!(obj.get("params"), Some(&serde_json::json!(1)));
    // Conflict marker is present.
    let conflicts = obj.get("_scx_uns_conflicts").unwrap().as_array().unwrap();
    assert_eq!(conflicts.len(), 1);
    assert_eq!(
        conflicts[0],
        serde_json::json!({"input": 1, "status": "differs"})
    );
}

#[test]
fn merge_provenance_records_policy_params() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    write_legacy_input(&p0, 10, "donor_A", &var_batch(), None);
    write_legacy_input(&p1, 10, "donor_B", &var_batch(), None);
    let out = dir.path().join("out.scx");
    let opts = scx_ops::MergeOptions {
        assume_identical_var: true,
        uns_policy: scx_ops::UnsPolicy::Namespace,
        ..Default::default()
    };
    scx_ops::merge_with_options(&[p0.as_path(), p1.as_path()], &out, &opts).unwrap();
    let prov = ScxReader::open(&out).unwrap().read_provenance().unwrap();
    let last = prov.operations.last().unwrap();
    assert_eq!(last.action, "merge");
    assert!(
        last.params_json.contains("\"assume_identical_var\":true"),
        "params_json = {}",
        last.params_json
    );
    assert!(
        last.params_json.contains("\"uns_policy\":\"namespace\""),
        "params_json = {}",
        last.params_json
    );
}

#[test]
fn append_legacy_to_sharded_promotes() {
    // Convert-on-append: starting from a legacy single-section
    // ObsMetadata file, the first append produces an output where
    // obs is row-sharded (shard 0 = old obs, shard 1 = new obs).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy.scx");
    write_legacy_input(&path, 6, "donor_A", &var_batch(), None);

    // Pre-append: legacy single-section.
    let pre = ScxReader::open(&path).unwrap();
    assert_eq!(pre.obs_metadata_shard_count(), 0);
    assert!(pre
        .catalog()
        .entries
        .iter()
        .any(|e| e.section_type == SectionType::ObsMetadata));
    drop(pre);

    // Append 4 rows.
    let new_obs = obs_batch(6, 4, "donor_B");
    let n_vars = 4usize;
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..4usize {
        let col0 = (row * 2) % n_vars;
        let col1 = (row * 2 + 1) % n_vars;
        indices.push(col0 as u32);
        indices.push(col1 as u32);
        values.push(((row + 1) % 256) as u8);
        values.push(((row + 2) % 256) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        &scx_ops::AppendOptions::default(),
    )
    .unwrap();

    // Post-append: convert-on-append landed; obs is now sharded.
    let post = ScxReader::open(&path).unwrap();
    assert_eq!(post.n_obs(), 10);
    assert_eq!(
        post.obs_metadata_shard_count(),
        2,
        "convert-on-append: shard 0 = old obs (6 rows), shard 1 = new obs (4 rows)"
    );
    // The legacy ObsMetadata entry must no longer be listed in the
    // catalog. Its bytes remain on disk (orphaned) until the next
    // `scx compact`.
    assert!(
        !post
            .catalog()
            .entries
            .iter()
            .any(|e| e.section_type == SectionType::ObsMetadata),
        "convert-on-append must drop the legacy ObsMetadata catalog entry"
    );
    // Per-shard row metadata verifies the cover.
    let s0 = post.read_obs_shard(0).unwrap();
    assert_eq!(s0.num_rows(), 6);
    assert_eq!(
        s0.schema().metadata().get("row_start").map(|s| s.as_str()),
        Some("0")
    );
    assert_eq!(
        s0.schema()
            .metadata()
            .get("n_rows_total")
            .map(|s| s.as_str()),
        Some("10")
    );
    let s1 = post.read_obs_shard(1).unwrap();
    assert_eq!(s1.num_rows(), 4);
    assert_eq!(
        s1.schema().metadata().get("row_start").map(|s| s.as_str()),
        Some("6")
    );
    // Assembled batch matches old + new row contents.
    let assembled = post.read_obs_assembled().unwrap();
    let cell_ids = assembled
        .column_by_name("cell_id")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(cell_ids.value(0), "cell_0000000");
    assert_eq!(cell_ids.value(5), "cell_0000005");
    assert_eq!(cell_ids.value(6), "cell_0000006");
    assert_eq!(cell_ids.value(9), "cell_0000009");
}

#[test]
fn append_then_append_extends_shards() {
    // Second append should add a new shard without rewriting the old
    // ones. After two appends, obs has three shards (one per write).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two_appends.scx");
    write_legacy_input(&path, 4, "donor_A", &var_batch(), None);

    let n_vars = 4usize;
    let mk_csr = |n: usize| -> (Vec<u64>, Vec<u32>, Vec<u8>) {
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for row in 0..n {
            indices.push(((row * 2) % n_vars) as u32);
            indices.push(((row * 2 + 1) % n_vars) as u32);
            values.push(((row + 1) % 256) as u8);
            values.push(((row + 2) % 256) as u8);
            indptr.push(indptr.last().unwrap() + 2);
        }
        (indptr, indices, values)
    };

    let new1 = obs_batch(4, 3, "donor_B");
    let (i1, idx1, v1) = mk_csr(3);
    scx_ops::append(
        &path,
        &new1,
        &i1,
        &idx1,
        &v1,
        ValueEncoding::Uint8,
        &scx_ops::AppendOptions::default(),
    )
    .unwrap();

    let new2 = obs_batch(7, 2, "donor_C");
    let (i2, idx2, v2) = mk_csr(2);
    scx_ops::append(
        &path,
        &new2,
        &i2,
        &idx2,
        &v2,
        ValueEncoding::Uint8,
        &scx_ops::AppendOptions::default(),
    )
    .unwrap();

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), 9);
    let assembled = reader.read_obs_assembled().unwrap();
    assert_eq!(assembled.num_rows(), 9);
    let donors = assembled
        .column_by_name("donor")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(donors.value(0), "donor_A");
    assert_eq!(donors.value(4), "donor_B");
    assert_eq!(donors.value(7), "donor_C");
}

#[test]
fn merge_dict_columns_round_trip() {
    // Dict-encoded categorical with different per-input dictionaries:
    // the merge must unify dict columns shard-by-shard (Phase 2a) so
    // the assembled batch preserves the original string values.
    fn build_dict_obs(start: usize, n: usize, values: &[&str]) -> RecordBatch {
        let cell_ids: Vec<String> = (start..start + n).map(|i| format!("cell_{i:04}")).collect();
        let dict_keys: Int32Array = (0..n).map(|i| (i % values.len()) as i32).collect();
        let dict_values = StringArray::from(values.to_vec());
        let dict_array = DictionaryArray::<arrow::datatypes::Int32Type>::try_new(
            dict_keys,
            Arc::new(dict_values),
        )
        .unwrap();
        let schema = Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new(
                "label",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                false,
            ),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(cell_ids)), Arc::new(dict_array)],
        )
        .unwrap()
    }

    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let var = var_batch();
    // input 0: label dictionary is {"alpha", "beta"}
    {
        let obs = build_dict_obs(0, 4, &["alpha", "beta"]);
        let mut writer = ScxWriter::new(&p0, header(4, 4)).unwrap();
        writer.write_obs(&obs).unwrap();
        writer.write_var(&var).unwrap();
        write_zero_csr_shard(&mut writer, 0, 4);
        writer.finish().unwrap();
    }
    // input 1: label dictionary is {"gamma", "delta"} — disjoint
    {
        let obs = build_dict_obs(4, 4, &["gamma", "delta"]);
        let mut writer = ScxWriter::new(&p1, header(4, 4)).unwrap();
        writer.write_obs(&obs).unwrap();
        writer.write_var(&var).unwrap();
        write_zero_csr_shard(&mut writer, 0, 4);
        writer.finish().unwrap();
    }

    let out = dir.path().join("merged.scx");
    scx_ops::merge(&[p0.as_path(), p1.as_path()], &out).unwrap();

    let assembled = ScxReader::open(&out).unwrap().read_obs_assembled().unwrap();
    assert_eq!(assembled.num_rows(), 8);
    // After unify_dict_columns the label column lands as Utf8 (the
    // dictionary's value type).
    let labels = assembled
        .column_by_name("label")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap_or_else(|| {
            panic!(
                "expected Utf8 label, got {:?}",
                assembled.column(1).data_type()
            )
        });
    let observed: Vec<&str> = (0..8).map(|i| labels.value(i)).collect();
    assert_eq!(
        observed,
        vec!["alpha", "beta", "alpha", "beta", "gamma", "delta", "gamma", "delta"]
    );
}

#[test]
fn merge_with_index_options_streaming_matches_batch() {
    // The streaming predicate-index builder must produce a
    // byte-identical PredicateIndex section to the (legacy) batch
    // builder for the same logical obs content. Compare a sharded
    // merge output's obs predicate index to the bytes a batch-mode
    // builder would have produced on the assembled obs.
    use scx_engine::{build_obs_predicate_index_bytes, PredicateIndexBuildOptions};

    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    write_legacy_input(&p0, 64, "donor_A", &var_batch(), None);
    write_legacy_input(&p1, 64, "donor_B", &var_batch(), None);
    let out = dir.path().join("merged.scx");
    let opts = scx_ops::MergeOptions {
        index_options: scx_engine::ConversionPredicateIndexOptions {
            index_obs: vec!["donor".to_string()],
            index_var: Vec::new(),
            index_preset: None,
            index_auto_threshold: 0,
        },
        ..Default::default()
    };
    scx_ops::merge_with_options(&[p0.as_path(), p1.as_path()], &out, &opts).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    let on_disk = reader
        .read_obs_predicate_index_bytes()
        .unwrap()
        .expect("merge with --index-obs donor must emit an obs predicate index");

    // Compute the batch-mode reference. obs has 128 rows split into
    // two shards of 64 each; each output shard had row_start 0 and 64
    // respectively (one shard per input).
    let assembled = reader.read_obs_assembled().unwrap();
    let shard_row_ranges: Vec<(u64, u64)> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CsrShard && e.modality_id == 0)
        .filter_map(|e| e.stats.as_ref().map(|s| (s.row_start, s.row_end)))
        .collect();
    let mut shard_row_ranges = shard_row_ranges;
    shard_row_ranges.sort_by_key(|(s, _)| *s);

    let build_opts = PredicateIndexBuildOptions {
        forced_columns: vec!["donor".to_string()],
        preset_columns: Vec::new(),
        auto_threshold: 0,
        high_cardinality_threshold: 100_000,
    };
    let mut outcomes = Vec::new();
    let mut indexed_names = Vec::new();
    let reference = build_obs_predicate_index_bytes(
        &assembled,
        &shard_row_ranges,
        &build_opts,
        &mut outcomes,
        &mut indexed_names,
    )
    .unwrap()
    .expect("batch-mode builder must also emit the donor index");

    assert_eq!(
        on_disk, reference,
        "streaming and batch predicate-index byte serialisations must match exactly"
    );
}

#[test]
#[ignore]
fn merge_inputs_exceed_i32_max_obs_string_payload() {
    // > 2 GiB cumulative obs string payload across many inputs.
    // Each individual input fits under i32::MAX (the original failure
    // mode was `concat_batches` on the merged narrow-offset Utf8
    // array). The streaming refactor writes one ObsMetadataShard per
    // input chunk, so each shard's IPC offsets stay narrow.
    //
    // Allocates ~3 GiB peak; run via:
    //
    // ```bash
    // cargo test --release -p scx-ops --test streaming_merge_append \
    //     -- --ignored --nocapture merge_inputs_exceed_i32_max_obs_string_payload
    // ```
    use std::fmt::Write as _;

    const N_INPUTS: usize = 4;
    const ROWS_PER_INPUT: usize = 900_000;
    const PAYLOAD_LEN: usize = 880;

    let dir = tempfile::tempdir().unwrap();
    let mut paths: Vec<PathBuf> = Vec::new();
    let payload: String = "x".repeat(PAYLOAD_LEN);
    for i in 0..N_INPUTS {
        let p = dir.path().join(format!("in_{i}.scx"));
        let mut writer = ScxWriter::new(&p, header(ROWS_PER_INPUT as u64, 4)).unwrap();
        let cell_ids: Vec<String> = (0..ROWS_PER_INPUT)
            .map(|r| {
                let mut s = String::with_capacity(PAYLOAD_LEN + 20);
                let global = i * ROWS_PER_INPUT + r;
                write!(&mut s, "cell_{global:09}{payload}").unwrap();
                s
            })
            .collect();
        let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(cell_ids))],
        )
        .unwrap();
        writer.write_obs(&batch).unwrap();
        writer.write_var(&var_batch()).unwrap();
        write_zero_csr_shard(&mut writer, 0, ROWS_PER_INPUT as u64);
        writer.finish().unwrap();
        paths.push(p);
    }

    let out = dir.path().join("merged.scx");
    let path_refs: Vec<&std::path::Path> = paths.iter().map(|p| p.as_path()).collect();
    scx_ops::merge(&path_refs, &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.n_obs() as usize, N_INPUTS * ROWS_PER_INPUT);
    // The streaming refactor emits one ObsMetadataShard per input
    // (each input is processed as a single chunk because its row
    // count fits in `shard_target_rows`).
    assert!(reader.obs_metadata_shard_count() >= N_INPUTS);
}
