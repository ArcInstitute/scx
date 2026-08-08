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

use arrow::array::{Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;

use scx_ops::{modify_metadata, set_uns, MetadataPatch, OpsError};

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
