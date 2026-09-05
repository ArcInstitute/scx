//! Integration tests for in-place metadata replacement
//! (`scx_ops::modify_metadata` / `set_uns`).
//!
//! Covers:
//! - `uns` round-trip with the matrix (CSR shards) proven byte-identical;
//! - generation invariants + CSC sidecar still validating (the key win vs append);
//! - `obs` replace with predicate-index carry-forward vs. explicit rebuild;
//! - shape rejection leaving the file untouched;
//! - rollback restoring the prior `uns`;
//! - empty-patch rejection.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;

use scx_ops::{modify_metadata, set_uns, update_uns, MetadataPatch, OpsError};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, n_vars, 0, 16_384, 0, 0)
}

fn obs_batch(n: usize, donor: &str) -> RecordBatch {
    let cell_ids: Vec<String> = (0..n).map(|i| format!("cell_{i:07}")).collect();
    let donors: Vec<String> = std::iter::repeat_n(donor.to_string(), n).collect();
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

fn var_batch(n: usize) -> RecordBatch {
    let gene_ids: Vec<String> = (0..n).map(|i| format!("g{i}")).collect();
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(gene_ids))],
    )
    .unwrap()
}

/// Dense `n_obs × n_vars` deterministic pattern.
fn dense(n_obs: usize, n_vars: usize) -> Vec<u8> {
    let mut d = vec![0u8; n_obs * n_vars];
    for r in 0..n_obs {
        for c in 0..n_vars {
            if (r + c) % 3 == 0 {
                d[r * n_vars + c] = ((r * 7 + c * 11) % 200 + 1) as u8;
            }
        }
    }
    d
}

fn dense_to_csr(d: &[u8], n_obs: usize, n_vars: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for r in 0..n_obs {
        for c in 0..n_vars {
            let v = d[r * n_vars + c];
            if v != 0 {
                indices.push(c as u32);
                values.push(v);
            }
        }
        indptr.push(indices.len() as u64);
    }
    (indptr, indices, values)
}

fn dense_to_csc_range(
    d: &[u8],
    n_obs: usize,
    n_vars: usize,
    col_start: usize,
    col_end: usize,
) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for c in col_start..col_end {
        for r in 0..n_obs {
            let v = d[r * n_vars + c];
            if v != 0 {
                indices.push(r as u32);
                values.push(v);
            }
        }
        indptr.push(indices.len() as u64);
    }
    (indptr, indices, values)
}

/// Base file: obs + var + one real CSR shard (+ optional uns). No CSC.
fn write_base(path: &Path, n_obs: usize, n_vars: usize, uns: Option<&serde_json::Value>) {
    let mut writer = ScxWriter::new(path, header(n_obs as u64, n_vars as u64)).unwrap();
    writer.write_obs(&obs_batch(n_obs, "donor_A")).unwrap();
    writer.write_var(&var_batch(n_vars)).unwrap();
    let (indptr, indices, values) = dense_to_csr(&dense(n_obs, n_vars), n_obs, n_vars);
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
    if let Some(u) = uns {
        writer.write_uns(u).unwrap();
    }
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1_710_000_000,
            action: "convert".to_string(),
            tool: "modify_metadata test fixture".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();
    writer.finish().unwrap();
}

/// Base file with a CSC sidecar (one shard per `cols_per` columns).
fn write_base_with_csc(path: &Path, n_obs: usize, n_vars: usize, cols_per: usize) {
    let d = dense(n_obs, n_vars);
    let mut writer = ScxWriter::new(path, header(n_obs as u64, n_vars as u64)).unwrap();
    writer.write_obs(&obs_batch(n_obs, "donor_A")).unwrap();
    writer.write_var(&var_batch(n_vars)).unwrap();
    let (indptr, indices, values) = dense_to_csr(&d, n_obs, n_vars);
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
    let mut col_start = 0usize;
    while col_start < n_vars {
        let col_end = (col_start + cols_per).min(n_vars);
        let (ip, ix, vb) = dense_to_csc_range(&d, n_obs, n_vars, col_start, col_end);
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
    writer.finish().unwrap();
}

/// Snapshot CSR shard catalog entries as (name, offset, length, checksum).
fn csr_shard_snapshot(path: &Path) -> Vec<(String, u64, u64, [u8; 32])> {
    let r = ScxReader::open(path).unwrap();
    let mut v: Vec<_> = r
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CsrShard)
        .map(|e| (e.name.clone(), e.offset, e.length, e.checksum))
        .collect();
    v.sort();
    v
}

fn has_section(path: &Path, ty: SectionType) -> bool {
    ScxReader::open(path)
        .unwrap()
        .catalog()
        .entries
        .iter()
        .any(|e| e.section_type == ty)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn set_uns_round_trip_leaves_matrix_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.scx");
    let uns0 = serde_json::json!({"method": "original", "k": 1});
    write_base(&path, 50, 6, Some(&uns0));

    let pre = csr_shard_snapshot(&path);
    let seq0 = ScxReader::open(&path).unwrap().header().manifest_sequence;

    let uns1 = serde_json::json!({"method": "updated", "k": 2, "extra": [1, 2, 3]});
    set_uns(&path, &uns1).unwrap();

    let r = ScxReader::open(&path).unwrap();
    assert_eq!(r.read_uns().unwrap(), uns1, "uns replaced");
    assert_eq!(r.n_obs(), 50, "n_obs unchanged");
    assert_eq!(
        r.header().manifest_sequence,
        seq0 + 1,
        "manifest_sequence bumped once"
    );
    // CSR shards must be byte-identical — proves no re-encode.
    assert_eq!(
        csr_shard_snapshot(&path),
        pre,
        "CSR shards must not move/change"
    );
    // obs/var read back unchanged.
    assert_eq!(r.read_obs().unwrap().num_rows(), 50);
    assert_eq!(r.read_var().unwrap().num_rows(), 6);
}

#[test]
fn set_uns_preserves_generations_and_csc_sidecar() {
    use scx_format_io::backed::BackedCscReader;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("csc.scx");
    write_base_with_csc(&path, 40, 8, 4);

    let (data_gen0, csc_gen0) = {
        let r = ScxReader::open(&path).unwrap();
        assert!(r.header().has_csc(), "fixture has a CSC sidecar");
        assert!(
            BackedCscReader::new(r, 0).is_ok(),
            "CSC reader opens pre-modify"
        );
        let r = ScxReader::open(&path).unwrap();
        (
            r.catalog().data_generation,
            r.catalog().csc_build_generation,
        )
    };

    set_uns(&path, &serde_json::json!({"note": "added after csc build"})).unwrap();

    let r = ScxReader::open(&path).unwrap();
    assert!(r.header().has_csc(), "HAS_CSC flag preserved");
    assert_eq!(
        r.catalog().data_generation,
        data_gen0,
        "data_generation unchanged"
    );
    assert_eq!(
        r.catalog().csc_build_generation,
        csc_gen0,
        "csc_build_generation unchanged"
    );
    // The whole point: the CSC sidecar still validates without --rebuild-csc.
    assert!(
        BackedCscReader::new(r, 0).is_ok(),
        "CSC sidecar still valid after set_uns"
    );
}

#[test]
fn obs_replace_changes_values_keeps_matrix() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("obs.scx");
    write_base(&path, 30, 5, None);
    let pre = csr_shard_snapshot(&path);

    let new_obs = obs_batch(30, "donor_Z");
    modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(new_obs),
            ..Default::default()
        },
    )
    .unwrap();

    let r = ScxReader::open(&path).unwrap();
    let obs = r.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 30);
    let donor = obs
        .column_by_name("donor")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(donor.value(0), "donor_Z", "obs values replaced");
    assert_eq!(
        csr_shard_snapshot(&path),
        pre,
        "matrix untouched by obs replace"
    );
    // No index requested → no ObsPredicateIndex written.
    assert!(!has_section(&path, SectionType::ObsPredicateIndex));
}

#[test]
fn obs_replace_index_rebuild_then_carry_forward() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("obs_idx.scx");
    write_base(&path, 30, 5, None);

    // Replace obs AND request an index over `donor`.
    let patch = MetadataPatch {
        obs: Some(obs_batch(30, "donor_A")),
        index: scx_engine::ConversionPredicateIndexOptions {
            index_obs: vec!["donor".to_string()],
            ..Default::default()
        },
        ..Default::default()
    };
    modify_metadata(&path, &patch).unwrap();
    assert!(
        has_section(&path, SectionType::ObsPredicateIndex),
        "predicate index rebuilt when requested"
    );
    assert_eq!(indexed_donor_values(&path), vec!["donor_A".to_string()]);

    // Replace obs again WITHOUT index flags. The old section cannot survive
    // verbatim — its ranges describe `donor_A`, which is gone — so the op
    // rebuilds over the same column rather than dropping the index and leaving
    // the file unprunable.
    let summary = modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(obs_batch(30, "donor_B")),
            ..Default::default()
        },
    )
    .unwrap();

    assert!(summary.obs_carried_forward);
    assert!(!summary.obs_predicate_index_dropped);
    assert!(
        has_section(&path, SectionType::ObsPredicateIndex),
        "the index the file had must be carried forward, not dropped"
    );
    // Carried, and not stale: it describes the values that are actually there.
    assert_eq!(
        indexed_donor_values(&path),
        vec!["donor_B".to_string()],
        "a carried index describing the replaced values would be worse than none"
    );
}

/// The `donor` categorical values the file's obs predicate index covers.
fn indexed_donor_values(path: &Path) -> Vec<String> {
    let reader = ScxReader::open(path).unwrap();
    let bytes = reader
        .read_obs_predicate_index_bytes()
        .unwrap()
        .expect("file must carry an obs predicate index");
    let index = scx_engine::PredicateIndex::read_from(&mut std::io::Cursor::new(bytes)).unwrap();
    index
        .columns
        .iter()
        .filter_map(|c| match c {
            scx_engine::index::IndexedColumn::Categorical(cat) if cat.column_name == "donor" => {
                Some(cat.entries.iter().map(|e| e.value.clone()))
            }
            _ => None,
        })
        .flatten()
        .collect()
}

#[test]
fn shape_mismatch_leaves_file_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.scx");
    let uns0 = serde_json::json!({"v": 0});
    write_base(&path, 20, 4, Some(&uns0));
    let seq0 = ScxReader::open(&path).unwrap().header().manifest_sequence;
    let pre = csr_shard_snapshot(&path);

    // Wrong obs row count → ShapeMismatch.
    let err = modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(obs_batch(19, "x")),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, OpsError::ShapeMismatch { .. }), "got {err:?}");

    // File byte-intact: catalog still points at the old state.
    let r = ScxReader::open(&path).unwrap();
    assert_eq!(
        r.header().manifest_sequence,
        seq0,
        "manifest unchanged on reject"
    );
    assert_eq!(r.read_uns().unwrap(), uns0, "uns unchanged on reject");
    assert_eq!(csr_shard_snapshot(&path), pre);
}

#[test]
fn rollback_restores_previous_uns() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rb.scx");
    let uns0 = serde_json::json!({"state": "v0"});
    write_base(&path, 16, 4, Some(&uns0));
    let seq0 = ScxReader::open(&path).unwrap().header().manifest_sequence;

    set_uns(&path, &serde_json::json!({"state": "v1"})).unwrap();
    assert_eq!(
        ScxReader::open(&path).unwrap().read_uns().unwrap(),
        serde_json::json!({"state": "v1"})
    );

    scx_ops::rollback(&path).unwrap();
    let r = ScxReader::open(&path).unwrap();
    assert_eq!(
        r.read_uns().unwrap(),
        uns0,
        "rollback restores original uns"
    );
    assert_eq!(
        r.header().manifest_sequence,
        seq0,
        "rollback restores sequence"
    );
}

#[test]
fn obsm_replace_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("obsm.scx");
    write_base(&path, 12, 4, None);

    // 12 × 2 embedding.
    let schema = Schema::new(vec![
        Field::new("c0", DataType::Float32, false),
        Field::new("c1", DataType::Float32, false),
    ]);
    let emb = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float32Array::from(
                (0..12).map(|i| i as f32).collect::<Vec<_>>(),
            )),
            Arc::new(Float32Array::from(
                (0..12).map(|i| -(i as f32)).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();

    modify_metadata(
        &path,
        &MetadataPatch {
            obsm: Some(vec![("X_pca".to_string(), emb)]),
            ..Default::default()
        },
    )
    .unwrap();

    assert!(
        has_section(&path, SectionType::ObsmEmbedding),
        "obsm section written"
    );
    let got = ScxReader::open(&path)
        .unwrap()
        .read_obsm_for(0, "X_pca")
        .unwrap();
    assert_eq!(got.num_rows(), 12);
}

/// A first-ever in-place `obsm` must survive the next `compact`.
///
/// `commit_in_place` deliberately skips `sync_from_catalog`, so an in-place op
/// has to stamp `has_obsm` by hand — and `compact` gates the whole obsm block on
/// that flag, not on the catalog. Reading the embedding back from the *compacted*
/// file is the assertion that matters; `has_obsm()` alone would pass against a
/// fix that sets the bit while the section still went missing.
#[test]
fn in_place_obsm_survives_a_later_compact() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("obsm_compact.scx");
    write_base(&path, 12, 4, None);
    assert!(
        !ScxReader::open(&path).unwrap().header().has_obsm(),
        "fixture must start with no obsm, or the flag is already set for us"
    );

    let schema = Schema::new(vec![Field::new("c0", DataType::Float32, false)]);
    let emb = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(Float32Array::from(
            (0..12).map(|i| i as f32).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    modify_metadata(
        &path,
        &MetadataPatch {
            obsm: Some(vec![("X_umap".to_string(), emb)]),
            ..Default::default()
        },
    )
    .unwrap();

    let out = dir.path().join("compacted.scx");
    scx_ops::compact(&path, &out).unwrap();

    let got = ScxReader::open(&out)
        .unwrap()
        .read_obsm_for(0, "X_umap")
        .expect("obsm written in place must survive compact");
    assert_eq!(got.num_rows(), 12);
}

#[test]
fn empty_patch_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.scx");
    write_base(&path, 8, 3, None);
    let err = modify_metadata(&path, &MetadataPatch::default()).unwrap_err();
    assert!(matches!(err, OpsError::InvalidInput(_)), "got {err:?}");
}

// ---------------------------------------------------------------------------
// §11.12 — an index option with no axis to apply to is refused, not ignored
// ---------------------------------------------------------------------------

/// `--index-*` without the matching `--obs` / `--var` used to be a complete
/// no-op: the op exited 0, reported success, and built nothing, because index
/// rebuilds are gated on the axis actually being replaced.
///
/// The reject side. Each case names a different knob, because the axis rule is
/// not uniform: `index_obs` / `index_var` are axis-scoped, while `index_preset`
/// and `index_auto_threshold` span both axes and are therefore only
/// unhonourable when *neither* axis is supplied. A single `user_wants_index`
/// check would get the second pair wrong in one direction or the other.
#[test]
fn index_request_without_a_matching_axis_is_refused() {
    let dir = tempfile::tempdir().unwrap();

    let cases: Vec<(&str, MetadataPatch)> = vec![
        (
            "index_obs with no obs",
            MetadataPatch {
                uns: Some(serde_json::json!({"k": 1})),
                index: scx_engine::ConversionPredicateIndexOptions {
                    index_obs: vec!["donor".to_string()],
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
        (
            "index_var with no var",
            MetadataPatch {
                uns: Some(serde_json::json!({"k": 1})),
                index: scx_engine::ConversionPredicateIndexOptions {
                    index_var: vec!["gene".to_string()],
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
        (
            "index_preset with neither axis",
            MetadataPatch {
                uns: Some(serde_json::json!({"k": 1})),
                index: scx_engine::ConversionPredicateIndexOptions {
                    index_preset: Some("cellxgene".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
        (
            "index_auto_threshold with neither axis",
            MetadataPatch {
                uns: Some(serde_json::json!({"k": 1})),
                index: scx_engine::ConversionPredicateIndexOptions {
                    index_auto_threshold: 1000,
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
    ];

    for (label, patch) in cases {
        let path = dir.path().join(format!("{}.scx", label.replace(' ', "_")));
        write_base(&path, 30, 5, None);
        let before = std::fs::metadata(&path).unwrap().len();

        let err = modify_metadata(&path, &patch)
            .expect_err(&format!("{label}: expected a refusal, got success"));
        assert!(
            matches!(err, OpsError::InvalidInput(_)),
            "{label}: expected InvalidInput, got {err:?}"
        );
        // These strings are user-facing (`pyscx.modify_metadata` /
        // `scx modify-metadata`), and a multiline Rust literal without `\`
        // continuations bakes the source indentation into the message. Asserting
        // only the variant let 14-18 space runs ship once already, so assert the
        // text too.
        let msg = err.to_string();
        assert!(
            !msg.contains("  "),
            "{label}: refusal message carries wrap padding — a multiline literal \
             is missing its `\\` continuations: {msg:?}"
        );
        // Refused before `prepare_in_place`, so the file is untouched — a
        // rejection that had already appended would be a worse outcome than
        // the no-op it replaces.
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            before,
            "{label}: the refusal wrote to the file"
        );
    }
}

/// The accept side. Without it the guard above would pass just as well if it
/// rejected *every* index request, and the op would be broken in the other
/// direction with nothing to say so.
#[test]
fn index_request_with_its_axis_supplied_is_honoured() {
    let dir = tempfile::tempdir().unwrap();

    // Axis-scoped: obs supplied, obs indexed.
    let path = dir.path().join("obs_ok.scx");
    write_base(&path, 30, 5, None);
    modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(obs_batch(30, "donor_A")),
            index: scx_engine::ConversionPredicateIndexOptions {
                index_obs: vec!["donor".to_string()],
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .expect("index_obs alongside obs must be accepted");
    assert!(
        has_section(&path, SectionType::ObsPredicateIndex),
        "the accepted request must actually build the index — otherwise this \
         asserts only that it did not error"
    );

    // Cross-axis: a preset needs only ONE axis present to have work to do.
    let path = dir.path().join("preset_ok.scx");
    write_base(&path, 30, 5, None);
    modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(obs_batch(30, "donor_A")),
            index: scx_engine::ConversionPredicateIndexOptions {
                index_preset: Some("cellxgene".to_string()),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .expect("index_preset with obs supplied must be accepted");
}

// ---------------------------------------------------------------------------
// update_uns — shallow top-level merge
// ---------------------------------------------------------------------------

#[test]
fn update_uns_overwrites_only_the_named_keys() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.scx");
    write_base(
        &path,
        16,
        4,
        Some(&serde_json::json!({"a": 1, "b": {"x": 1}, "keep": [1.5, "s"]})),
    );
    let pre = csr_shard_snapshot(&path);
    let seq0 = ScxReader::open(&path).unwrap().header().manifest_sequence;

    update_uns(&path, &serde_json::json!({"b": 2, "c": 3})).unwrap();

    let r = ScxReader::open(&path).unwrap();
    assert_eq!(
        r.read_uns().unwrap(),
        serde_json::json!({"a": 1, "b": 2, "c": 3, "keep": [1.5, "s"]}),
        "patch keys overwrite (shallow: `b` is replaced, not deep-merged); \
         untouched keys survive verbatim"
    );
    assert_eq!(r.header().manifest_sequence, seq0 + 1, "one commit");
    assert_eq!(csr_shard_snapshot(&path), pre, "no matrix re-encode");
}

#[test]
fn update_uns_on_a_file_without_uns_creates_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("none.scx");
    write_base(&path, 16, 4, None);
    assert!(
        ScxReader::open(&path).unwrap().read_uns().is_err(),
        "fixture has no uns section"
    );

    update_uns(&path, &serde_json::json!({"c": 3})).unwrap();

    assert_eq!(
        ScxReader::open(&path).unwrap().read_uns().unwrap(),
        serde_json::json!({"c": 3})
    );
}

#[test]
fn update_uns_rejects_a_non_object_patch_and_leaves_the_file_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.scx");
    let uns0 = serde_json::json!({"a": 1});
    write_base(&path, 16, 4, Some(&uns0));
    let bytes0 = std::fs::read(&path).unwrap();

    let err = update_uns(&path, &serde_json::json!([1, 2])).unwrap_err();
    assert!(matches!(err, OpsError::InvalidInput(_)), "{err}");
    assert!(err.to_string().contains("object"), "{err}");

    assert_eq!(std::fs::read(&path).unwrap(), bytes0, "file byte-identical");
    assert_eq!(ScxReader::open(&path).unwrap().read_uns().unwrap(), uns0);
}

#[test]
fn update_uns_refuses_a_non_object_existing_uns() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("arr.scx");
    write_base(&path, 16, 4, Some(&serde_json::json!([1, 2, 3])));
    let bytes0 = std::fs::read(&path).unwrap();

    let err = update_uns(&path, &serde_json::json!({"c": 3})).unwrap_err();
    assert!(matches!(err, OpsError::InvalidInput(_)), "{err}");
    assert!(err.to_string().contains("set_uns"), "{err}");
    assert_eq!(std::fs::read(&path).unwrap(), bytes0, "file byte-identical");
}

#[test]
fn update_uns_then_rollback_restores_previous_uns() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rbm.scx");
    let uns0 = serde_json::json!({"state": "v0"});
    write_base(&path, 16, 4, Some(&uns0));
    let seq0 = ScxReader::open(&path).unwrap().header().manifest_sequence;

    update_uns(&path, &serde_json::json!({"added": true})).unwrap();
    assert_eq!(
        ScxReader::open(&path).unwrap().read_uns().unwrap(),
        serde_json::json!({"state": "v0", "added": true})
    );

    scx_ops::rollback(&path).unwrap();
    let r = ScxReader::open(&path).unwrap();
    assert_eq!(r.read_uns().unwrap(), uns0);
    assert_eq!(r.header().manifest_sequence, seq0);
}

#[test]
fn update_uns_preserves_generations_and_csc_sidecar() {
    use scx_format_io::backed::BackedCscReader;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("csc_m.scx");
    write_base_with_csc(&path, 40, 8, 4);
    let (data_gen0, csc_gen0) = {
        let r = ScxReader::open(&path).unwrap();
        (
            r.catalog().data_generation,
            r.catalog().csc_build_generation,
        )
    };

    update_uns(&path, &serde_json::json!({"note": "merged"})).unwrap();

    let r = ScxReader::open(&path).unwrap();
    assert!(r.header().has_csc());
    assert_eq!(r.catalog().data_generation, data_gen0);
    assert_eq!(r.catalog().csc_build_generation, csc_gen0);
    assert!(BackedCscReader::new(r, 0).is_ok(), "CSC reader still opens");
    let prov = ScxReader::open(&path).unwrap().read_provenance().unwrap();
    let last = prov.operations.last().unwrap();
    assert!(
        last.params_json.contains("\"uns_merge\":true"),
        "provenance must distinguish a merge from a replace: {}",
        last.params_json
    );
}

/// A categorical obs column goes through `modify_metadata(obs=)` as the
/// dictionary it arrived as — dtype, the caller's declared order, an unused
/// level, the `ordered` stamp — and the predicate index can still be built over
/// it. Before, the obs batch was cast to plain strings on its way to the shards,
/// so `read_obs()` handed the column back as `object` after every replace.
#[test]
fn obs_replace_keeps_categorical_columns_as_dictionaries() {
    use arrow::array::{AsArray, DictionaryArray, Int8Array};
    use arrow::datatypes::Int8Type;
    use std::collections::HashMap;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cat.scx");
    write_base(&path, 30, 5, None);

    let n = 30;
    let cell_ids: Vec<String> = (0..n).map(|i| format!("cell_{i:07}")).collect();
    // Declared order is neither alphabetical nor first-appearance, and the
    // third level is used by no row.
    let keys = Int8Array::from((0..n).map(|i| (i % 2) as i8).collect::<Vec<_>>());
    let values = StringArray::from(vec!["donor_B", "donor_A", "donor_unused"]);
    let dict = DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap();
    let mut md = HashMap::new();
    md.insert(
        scx_format_io::CATEGORICAL_ORDERED_KEY.to_string(),
        "true".to_string(),
    );
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("donor", dict.data_type().clone(), true).with_metadata(md),
    ]);
    let obs = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(cell_ids)), Arc::new(dict)],
    )
    .unwrap();

    modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(obs),
            index: scx_engine::ConversionPredicateIndexOptions {
                index_obs: vec!["donor".to_string()],
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();

    let back = ScxReader::open(&path).unwrap().read_obs().unwrap();
    let col = back.column_by_name("donor").unwrap();
    let dict = col.as_any_dictionary_opt().unwrap_or_else(|| {
        panic!(
            "donor must come back as a dictionary, got {:?}",
            col.data_type()
        )
    });
    let declared = arrow::compute::cast(dict.values(), &DataType::Utf8).unwrap();
    let declared = declared.as_string::<i32>();
    assert_eq!(
        (0..declared.len())
            .map(|i| declared.value(i))
            .collect::<Vec<_>>(),
        ["donor_B", "donor_A", "donor_unused"],
        "declared order and the unused level survive"
    );
    let flat = arrow::compute::cast(col, &DataType::Utf8).unwrap();
    let flat = flat.as_string::<i32>();
    assert_eq!(flat.value(0), "donor_B");
    assert_eq!(flat.value(1), "donor_A");
    assert_eq!(
        back.schema()
            .field_with_name("donor")
            .unwrap()
            .metadata()
            .get(scx_format_io::CATEGORICAL_ORDERED_KEY)
            .map(String::as_str),
        Some("true")
    );

    // The index was built over the dictionary column and describes its values.
    assert!(has_section(&path, SectionType::ObsPredicateIndex));
    let mut indexed = indexed_donor_values(&path);
    indexed.sort();
    assert_eq!(indexed, ["donor_A", "donor_B"]);
}

// ---------------------------------------------------------------------------
// Either row space for `obs=` on a file with deletions (pyscx 0.17: `read_obs()`
// returns live rows, `read_obs(logical=False)` the physical ones)
// ---------------------------------------------------------------------------

const DELETED_ROWS: [u64; 3] = [2, 7, 11];

fn str_col<'a>(batch: &'a RecordBatch, name: &str) -> &'a StringArray {
    batch
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
}

/// A live-length obs lands on the live rows in order and leaves every deleted
/// row null; the deletion vector, the live count and the matrix are untouched.
#[test]
fn obs_replace_accepts_a_live_length_frame_on_a_deleted_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("live.scx");
    write_base(&path, 20, 4, None);
    scx_ops::mark_deleted(&path, &DELETED_ROWS).unwrap();
    let pre = csr_shard_snapshot(&path);
    assert_eq!(
        ScxReader::open(&path)
            .unwrap()
            .read_obs_filtered()
            .unwrap()
            .num_rows(),
        17,
        "premise: 17 live rows"
    );

    modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(obs_batch(17, "donor_Z")),
            ..Default::default()
        },
    )
    .unwrap();

    let r = ScxReader::open(&path).unwrap();
    let physical = r.read_obs().unwrap();
    assert_eq!(
        physical.num_rows(),
        20,
        "the obs axis stays physical on disk"
    );
    let donor = str_col(&physical, "donor");
    let cell_id = str_col(&physical, "cell_id");
    let mut live = 0usize;
    for row in 0..20 {
        if DELETED_ROWS.contains(&(row as u64)) {
            assert!(donor.is_null(row), "deleted row {row}: donor must be null");
            // No pandas index column in this fixture → nothing to preserve.
            assert!(
                cell_id.is_null(row),
                "deleted row {row}: cell_id must be null"
            );
        } else {
            assert_eq!(donor.value(row), "donor_Z");
            assert_eq!(cell_id.value(row), format!("cell_{live:07}"));
            live += 1;
        }
    }
    let logical = r.read_obs_filtered().unwrap();
    assert_eq!(logical.num_rows(), 17);
    assert!(str_col(&logical, "donor")
        .iter()
        .all(|v| v == Some("donor_Z")));

    assert!(r.header().has_deletion_vectors(), "deletion vector carried");
    let keep = r.deletion_keep_mask().unwrap().unwrap();
    for row in DELETED_ROWS {
        assert!(!keep[row as usize]);
    }
    assert_eq!(csr_shard_snapshot(&path), pre, "matrix untouched");
}

/// A physical-length frame still writes every row, deleted ones included.
#[test]
fn obs_replace_still_accepts_a_physical_length_frame_on_a_deleted_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("phys.scx");
    write_base(&path, 20, 4, None);
    scx_ops::mark_deleted(&path, &DELETED_ROWS).unwrap();

    modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(obs_batch(20, "donor_Z")),
            ..Default::default()
        },
    )
    .unwrap();

    let r = ScxReader::open(&path).unwrap();
    let physical = r.read_obs().unwrap();
    assert_eq!(physical.num_rows(), 20);
    let donor = str_col(&physical, "donor");
    assert!((0..20).all(|row| donor.value(row) == "donor_Z"));
    assert_eq!(r.read_obs_filtered().unwrap().num_rows(), 17);
}

/// Neither length names both counts and the read that yields each; the file is
/// untouched.
#[test]
fn obs_replace_wrong_length_on_a_deleted_file_names_both_counts() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.scx");
    write_base(&path, 20, 4, None);
    scx_ops::mark_deleted(&path, &DELETED_ROWS).unwrap();
    let seq0 = ScxReader::open(&path).unwrap().header().manifest_sequence;

    for n in [16usize, 18] {
        let err = modify_metadata(
            &path,
            &MetadataPatch {
                obs: Some(obs_batch(n, "x")),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, OpsError::ShapeMismatch { .. }), "got {err:?}");
        let msg = err.to_string();
        for needle in [
            &format!("{n} rows"),
            "n_obs = 17",
            "n_obs_physical = 20",
            "read_obs()",
            "read_obs(logical=False)",
        ] {
            assert!(msg.contains(needle), "missing {needle:?} in: {msg}");
        }
    }
    assert_eq!(
        ScxReader::open(&path).unwrap().header().manifest_sequence,
        seq0,
        "manifest unchanged on reject"
    );
}

/// `obsm=` stays physical-length: a dense mapping has no null for a deleted row.
#[test]
fn obsm_replace_on_a_deleted_file_stays_physical_length() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("obsm.scx");
    write_base(&path, 20, 4, None);
    scx_ops::mark_deleted(&path, &DELETED_ROWS).unwrap();

    let emb = |n: usize| {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "c0",
                DataType::Float32,
                false,
            )])),
            vec![Arc::new(Float32Array::from(
                (0..n).map(|i| i as f32).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    };
    let err = modify_metadata(
        &path,
        &MetadataPatch {
            obsm: Some(vec![("X_pca".to_string(), emb(17))]),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, OpsError::ShapeMismatch { .. }), "got {err:?}");
    let msg = err.to_string();
    assert!(
        msg.contains("17 rows")
            && msg.contains("n_obs_physical = 20")
            && msg.contains("dense mapping"),
        "{msg}"
    );

    modify_metadata(
        &path,
        &MetadataPatch {
            obsm: Some(vec![("X_pca".to_string(), emb(20))]),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        ScxReader::open(&path)
            .unwrap()
            .read_obsm_for(0, "X_pca")
            .unwrap()
            .num_rows(),
        20
    );
}

/// The predicate index requested alongside a live-length frame is built over the
/// scattered, physical-length obs — the same row space as the CSR shards.
#[test]
fn obs_replace_live_frame_rebuilds_the_index_in_physical_space() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("live_idx.scx");
    write_base(&path, 20, 4, None);
    scx_ops::mark_deleted(&path, &DELETED_ROWS).unwrap();

    modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(obs_batch(17, "donor_Z")),
            index: scx_engine::ConversionPredicateIndexOptions {
                index_obs: vec!["donor".to_string()],
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    assert!(has_section(&path, SectionType::ObsPredicateIndex));
    assert_eq!(indexed_donor_values(&path), vec!["donor_Z".to_string()]);

    // The index must describe the physical axis: a query through it returns
    // exactly the live rows (the deleted rows are null in `donor`, so they
    // match neither the predicate nor the keep mask).
    let pipeline = scx_engine::QueryPipeline::open(&path).unwrap();
    let result = pipeline
        .filter_obs("donor == 'donor_Z'")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(result.x.shape.0, 17);
    assert_eq!(result.obs.num_rows(), 17);
}

/// A live-length frame keeps the deleted rows' barcodes when the file declares
/// an obs index column: a null key would become `""` in every later keyed join
/// and, twice over, a duplicate — which would block `obs_import` on the file
/// until `compact`. Everything else about a deleted row is null.
#[test]
fn obs_replace_live_frame_keeps_the_deleted_rows_barcodes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("barcodes.scx");
    let n = 20usize;
    let indexed_obs = |ids: Vec<String>, donor: &str| {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("__index_level_0__", DataType::Utf8, false),
                Field::new("donor", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(
                    std::iter::repeat_n(donor.to_string(), 0)
                        .chain((0..0).map(|_| String::new()))
                        .collect::<Vec<_>>(),
                )),
            ],
        )
    };
    let _ = indexed_obs; // (shape helper kept simple below)
    let build = |ids: Vec<String>, donor: &str| {
        let k = ids.len();
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("__index_level_0__", DataType::Utf8, false),
                Field::new("donor", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(vec![donor.to_string(); k])),
            ],
        )
        .unwrap()
    };
    {
        let mut writer = ScxWriter::new(&path, header(n as u64, 4)).unwrap();
        writer
            .write_obs(&build(
                (0..n).map(|i| format!("bc_{i}")).collect(),
                "donor_A",
            ))
            .unwrap();
        writer.write_var(&var_batch(4)).unwrap();
        let (indptr, indices, values) = dense_to_csr(&dense(n, 4), n, 4);
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
    scx_ops::mark_deleted(&path, &DELETED_ROWS).unwrap();

    // The caller renames the live barcodes too — the deleted rows never saw it.
    modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(build(
                (0..17).map(|i| format!("new_{i}")).collect(),
                "donor_Z",
            )),
            ..Default::default()
        },
    )
    .unwrap();

    let r = ScxReader::open(&path).unwrap();
    let physical = r.read_obs().unwrap();
    let idx = str_col(&physical, "__index_level_0__");
    let donor = str_col(&physical, "donor");
    let mut live = 0usize;
    for row in 0..n {
        if DELETED_ROWS.contains(&(row as u64)) {
            assert_eq!(
                idx.value(row),
                format!("bc_{row}"),
                "deleted row {row} keeps its barcode"
            );
            assert!(donor.is_null(row), "…but every other column is null");
        } else {
            assert_eq!(idx.value(row), format!("new_{live}"));
            assert_eq!(donor.value(row), "donor_Z");
            live += 1;
        }
    }

    // And the file stays joinable by its index: a keyed attach over the live
    // barcodes matches every live row and trips no duplicate-key check.
    let scores = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("s", DataType::Float32, true)])),
        vec![Arc::new(Float32Array::from(
            (0..17).map(|i| i as f32).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    let summary = scx_ops::attach_external_obs(
        &path,
        &scx_ops::ExternalObsData {
            row_keys: (0..17).map(|i| format!("new_{i}")).collect(),
            row_annotations: scores,
            row_embeddings: Vec::new(),
            uns: Default::default(),
            source_checksum: None,
            source_name: None,
        },
        &scx_ops::AttachObsOptions::default(),
    )
    .unwrap();
    assert_eq!(summary.n_matched, 17);
    assert_eq!(
        summary.n_target_rows_absent, 3,
        "the deleted rows, whose barcodes no source row names"
    );
}

/// The row space the frame arrived in is recorded in provenance: the obs on
/// disk is physical-length either way and does not say which frame produced it.
#[test]
fn obs_replace_records_the_row_space_in_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prov.scx");
    write_base(&path, 20, 4, None);
    scx_ops::mark_deleted(&path, &DELETED_ROWS).unwrap();

    let last_params = |path: &Path| {
        ScxReader::open(path)
            .unwrap()
            .read_provenance()
            .unwrap()
            .operations
            .last()
            .unwrap()
            .params_json
            .clone()
    };
    modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(obs_batch(17, "live")),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        last_params(&path).contains("\"row_space\":\"logical\""),
        "{}",
        last_params(&path)
    );
    modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(obs_batch(20, "phys")),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        last_params(&path).contains("\"row_space\":\"physical\""),
        "{}",
        last_params(&path)
    );
}

/// Obs with a pandas envelope declaring `levels` as its index columns.
fn envelope_obs(levels: &[&str], values: Vec<Vec<String>>, donor: &str) -> RecordBatch {
    let n = values[0].len();
    let mut fields: Vec<Field> = levels
        .iter()
        .map(|l| Field::new(*l, DataType::Utf8, false))
        .collect();
    fields.push(Field::new("donor", DataType::Utf8, false));
    let mut columns: Vec<Arc<dyn Array>> = values
        .into_iter()
        .map(|v| Arc::new(StringArray::from(v)) as Arc<dyn Array>)
        .collect();
    columns.push(Arc::new(StringArray::from(vec![donor.to_string(); n])));
    let quoted: Vec<String> = levels.iter().map(|l| format!("\"{l}\"")).collect();
    let mut meta = std::collections::HashMap::new();
    meta.insert(
        "pandas".to_string(),
        format!("{{\"index_columns\":[{}]}}", quoted.join(",")),
    );
    RecordBatch::try_new(Arc::new(Schema::new(fields).with_metadata(meta)), columns).unwrap()
}

fn write_base_with_obs(path: &Path, obs: RecordBatch, n_vars: usize) {
    let n = obs.num_rows();
    let mut writer = ScxWriter::new(path, header(n as u64, n_vars as u64)).unwrap();
    writer.write_obs(&obs).unwrap();
    writer.write_var(&var_batch(n_vars)).unwrap();
    let (indptr, indices, values) = dense_to_csr(&dense(n, n_vars), n, n_vars);
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

/// A live frame whose index was renamed (`rename_axis`) still keeps the deleted
/// rows' barcodes: the two sides' index columns are resolved independently and
/// paired by position, not by name.
#[test]
fn obs_replace_live_frame_with_a_renamed_index_keeps_the_deleted_rows_barcodes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("renamed.scx");
    let n = 20usize;
    write_base_with_obs(
        &path,
        envelope_obs(
            &["__index_level_0__"],
            vec![(0..n).map(|i| format!("bc_{i}")).collect()],
            "donor_A",
        ),
        4,
    );
    scx_ops::mark_deleted(&path, &DELETED_ROWS).unwrap();

    modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(envelope_obs(
                &["cell_id"],
                vec![(0..17).map(|i| format!("new_{i}")).collect()],
                "donor_Z",
            )),
            ..Default::default()
        },
    )
    .unwrap();

    let physical = ScxReader::open(&path).unwrap().read_obs().unwrap();
    assert!(
        physical.column_by_name("cell_id").is_some(),
        "the frame's index name wins"
    );
    let idx = str_col(&physical, "cell_id");
    let mut live = 0usize;
    for row in 0..n {
        if DELETED_ROWS.contains(&(row as u64)) {
            assert_eq!(
                idx.value(row),
                format!("bc_{row}"),
                "deleted row {row} keeps its barcode"
            );
        } else {
            assert_eq!(idx.value(row), format!("new_{live}"));
            live += 1;
        }
    }
}

/// Every index level is paired and preserved, not only the first.
#[test]
fn obs_replace_live_frame_pairs_every_index_level() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi.scx");
    let n = 20usize;
    write_base_with_obs(
        &path,
        envelope_obs(
            &["lvl_a", "lvl_b"],
            vec![
                (0..n).map(|i| format!("a_{i}")).collect(),
                (0..n).map(|i| format!("b_{i}")).collect(),
            ],
            "donor_A",
        ),
        4,
    );
    scx_ops::mark_deleted(&path, &DELETED_ROWS).unwrap();

    modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(envelope_obs(
                &["lvl_a", "lvl_b"],
                vec![
                    (0..17).map(|i| format!("na_{i}")).collect(),
                    (0..17).map(|i| format!("nb_{i}")).collect(),
                ],
                "donor_Z",
            )),
            ..Default::default()
        },
    )
    .unwrap();

    let physical = ScxReader::open(&path).unwrap().read_obs().unwrap();
    let a = str_col(&physical, "lvl_a");
    let b = str_col(&physical, "lvl_b");
    for row in DELETED_ROWS {
        let row = row as usize;
        assert_eq!(
            a.value(row),
            format!("a_{row}"),
            "level a kept on deleted row {row}"
        );
        assert_eq!(
            b.value(row),
            format!("b_{row}"),
            "level b kept on deleted row {row}"
        );
    }
    assert_eq!(a.value(0), "na_0");
    assert_eq!(b.value(0), "nb_0");
}

/// A live frame holding the file's own live barcodes in a different order is a
/// `sort_values` / `reindex` accident, not a rename: refused by row. Different
/// values (a rename) pass — that is what a replace is for.
#[test]
fn obs_replace_refuses_a_reordered_live_frame() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reorder.scx");
    let n = 20usize;
    write_base_with_obs(
        &path,
        envelope_obs(
            &["__index_level_0__"],
            vec![(0..n).map(|i| format!("bc_{i}")).collect()],
            "donor_A",
        ),
        4,
    );
    scx_ops::mark_deleted(&path, &DELETED_ROWS).unwrap();
    let seq0 = ScxReader::open(&path).unwrap().header().manifest_sequence;

    let live: Vec<String> = (0..n)
        .filter(|i| !DELETED_ROWS.contains(&(*i as u64)))
        .map(|i| format!("bc_{i}"))
        .collect();
    let mut reversed = live.clone();
    reversed.reverse();
    let err = modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(envelope_obs(
                &["__index_level_0__"],
                vec![reversed],
                "donor_Z",
            )),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, OpsError::InvalidInput(_)), "got {err:?}");
    let msg = err.to_string();
    assert!(
        msg.contains("different order") && msg.contains("row 0 of the frame is 'bc_19'"),
        "{msg}"
    );
    assert_eq!(
        ScxReader::open(&path).unwrap().header().manifest_sequence,
        seq0,
        "refused → nothing written"
    );

    // Same order: accepted. Renamed barcodes: accepted (a replace may rename).
    modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(envelope_obs(&["__index_level_0__"], vec![live], "donor_Z")),
            ..Default::default()
        },
    )
    .unwrap();
    modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(envelope_obs(
                &["__index_level_0__"],
                vec![(0..17).map(|i| format!("renamed_{i}")).collect()],
                "donor_Y",
            )),
            ..Default::default()
        },
    )
    .unwrap();
    let live_now = ScxReader::open(&path).unwrap().read_obs_filtered().unwrap();
    assert_eq!(
        str_col(&live_now, "__index_level_0__").value(0),
        "renamed_0"
    );
}

/// A live-length frame cannot change the number of index levels: levels are
/// paired by position, so a mismatch would pair the wrong columns or leave a
/// new level null on the deleted rows. Restructuring the index is a
/// physical-length replace.
#[test]
fn obs_replace_live_frame_refuses_an_index_arity_change() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("arity.scx");
    let n = 20usize;
    write_base_with_obs(
        &path,
        envelope_obs(
            &["lvl_a", "lvl_b"],
            vec![
                (0..n).map(|i| format!("a_{i}")).collect(),
                (0..n).map(|i| format!("b_{i}")).collect(),
            ],
            "donor_A",
        ),
        4,
    );
    scx_ops::mark_deleted(&path, &DELETED_ROWS).unwrap();
    let seq0 = ScxReader::open(&path).unwrap().header().manifest_sequence;

    let err = modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(envelope_obs(
                &["barcode"],
                vec![(0..17).map(|i| format!("bc_{i}")).collect()],
                "donor_Z",
            )),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, OpsError::InvalidInput(_)), "got {err:?}");
    let msg = err.to_string();
    assert!(
        msg.contains("index level count") && msg.contains("2 level(s)") && msg.contains("has 1"),
        "{msg}"
    );
    assert_eq!(
        ScxReader::open(&path).unwrap().header().manifest_sequence,
        seq0
    );

    // The physical-length frame restructures freely.
    modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(envelope_obs(
                &["barcode"],
                vec![(0..n).map(|i| format!("bc_{i}")).collect()],
                "donor_Z",
            )),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        ScxReader::open(&path)
            .unwrap()
            .read_obs_filtered()
            .unwrap()
            .num_rows(),
        17
    );
}

/// The arity guard covers a side with no declared index too: dropping the index
/// (1→0) would erase the deleted rows' identity, adding one (0→1) would leave it
/// null on every deleted row.
#[test]
fn obs_replace_live_frame_refuses_adding_or_dropping_the_index() {
    let dir = tempfile::tempdir().unwrap();

    // 1 → 0: the file has an index, the live frame has none.
    let path = dir.path().join("drop_index.scx");
    write_base_with_obs(
        &path,
        envelope_obs(
            &["__index_level_0__"],
            vec![(0..20).map(|i| format!("bc_{i}")).collect()],
            "donor_A",
        ),
        4,
    );
    scx_ops::mark_deleted(&path, &DELETED_ROWS).unwrap();
    let err = modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(obs_batch(17, "donor_Z")), // `cell_id` + `donor`, no index column
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("index level count"), "{err}");

    // 0 → 1: the file has no index, the live frame declares one.
    let path = dir.path().join("add_index.scx");
    write_base(&path, 20, 4, None); // `cell_id` + `donor`, no index column
    scx_ops::mark_deleted(&path, &DELETED_ROWS).unwrap();
    let err = modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(envelope_obs(
                &["__index_level_0__"],
                vec![(0..17).map(|i| format!("bc_{i}")).collect()],
                "donor_Z",
            )),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("index level count"), "{err}");
    // Physical-length: both restructurings are fine.
    modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(envelope_obs(
                &["__index_level_0__"],
                vec![(0..20).map(|i| format!("bc_{i}")).collect()],
                "donor_Z",
            )),
            ..Default::default()
        },
    )
    .unwrap();
}
