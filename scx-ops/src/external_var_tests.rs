//! Tests for [`attach_external_var`].
//!
//! Priorities, roughly by how badly a regression would hurt:
//!
//! 1. **The join is by key.** A permuted source must land each row on the right
//!    gene. A permuted annotation table that still produces a correctly-shaped
//!    column is the failure this op exists to prevent, and it is invisible in
//!    every shape assertion.
//! 2. **The layout survives.** A sharded var must come back sharded with the
//!    same boundaries and a single section must stay single. `cellbender_import`
//!    got this wrong for as long as it has existed because nothing asserted it.
//! 3. **Nothing else is lost.** obs, X, layers, the CSC sidecar, `.raw`, `varm`
//!    and the *obs* predicate index all survive; `rollback` undoes the lot; the
//!    var predicate index survives an add and is rebuilt on an overwrite; and
//!    the per-shard obs column stats are **not** cleared, because there is no
//!    var equivalent and clearing them would disable pruning on obs.
//! 4. **Overwrite replaces, it does not merge**, exactly as on obs.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    Array, BooleanArray, DictionaryArray, Float32Array, Int32Array, Int8Array, RecordBatch,
    StringArray,
};
use arrow::datatypes::{DataType, Field, Int32Type, Int8Type, Schema};
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

/// A var table with the three shapes a key can take: a pandas index
/// (`__index_level_0__`, reached as `var_names`), an ordinary string column
/// (`gene_id`), and a low-cardinality string column (`feature_type`) that can
/// carry a predicate index and cannot key a join.
fn var_batch(n: usize) -> RecordBatch {
    let names: Vec<String> = (0..n).map(|i| format!("GENE{i}")).collect();
    let ids: Vec<String> = (0..n).map(|i| format!("ENSG{i:05}")).collect();
    let types: Vec<&str> = (0..n)
        .map(|i| {
            if i % 2 == 0 {
                "Gene Expression"
            } else {
                "Peaks"
            }
        })
        .collect();
    let schema = Schema::new(vec![
        Field::new("__index_level_0__", DataType::Utf8, false),
        Field::new("gene_id", DataType::Utf8, false),
        Field::new("feature_type", DataType::Utf8, true),
    ])
    .with_metadata(pandas_envelope());
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(names)),
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(types)),
        ],
    )
    .unwrap()
}

/// The `pandas` schema envelope that declares which field carries the index.
///
/// `resolve_key_column`'s automatic path reads **only** this envelope — the
/// literal-name fallback (`resolve_index_columns`) is for consumers that must
/// *identify* an index rather than honour a declared one. A `from_anndata`
/// output carries it; a CLI-converted file does not, and then `key=None`
/// resolves through the gene-id fallback list instead. Both shapes are tested.
fn pandas_envelope() -> std::collections::HashMap<String, String> {
    let mut md = std::collections::HashMap::new();
    md.insert(
        "pandas".to_string(),
        serde_json::json!({"index_columns": ["__index_level_0__"]}).to_string(),
    );
    md
}

fn obs_batch(n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let schema = Schema::new(vec![Field::new("barcode", DataType::Utf8, false)]);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(StringArray::from(ids))]).unwrap()
}

/// How a fixture lays out its var axis on disk.
///
/// The two are not interchangeable: a `Single` section is one Arrow IPC batch
/// with no per-shard reader, and `Shards` is what `from_anndata` writes once
/// `n_vars > shard_target_rows`. The core behaviours run over both, because a
/// rewrite that quietly normalises one into the other is the defect this op was
/// written to avoid inheriting.
#[derive(Clone, Copy, Debug)]
enum VarLayout {
    Single,
    /// Per-shard row counts, spelled out so a test can make the boundaries
    /// *disagree* with `header.shard_target_rows` — which is what separates a
    /// rewrite that preserves the input layout from one that re-derives it.
    Shards(&'static [usize]),
}

struct Fixture {
    path: PathBuf,
    n_obs: usize,
    n_vars: usize,
}

/// An SCX file carrying every section family a var attach must leave alone:
/// obs, X shards, a layer, a CSC sidecar, `raw/X` + `raw/var` on its own wider
/// gene axis, a `varm` embedding, `uns` and a provenance chain — plus a var
/// predicate index over `index_var`.
fn write_fixture(
    dir: &Path,
    name: &str,
    n_obs: usize,
    var: RecordBatch,
    layout: VarLayout,
    index_var: &[&str],
) -> Fixture {
    let n_vars = var.num_rows();
    let path = dir.join(name);
    // Deliberately not a divisor of any `VarLayout::Shards` split below, so a
    // rewrite that re-derives var shard boundaries from the header produces a
    // different split and the layout tests can see it.
    let header = FileHeader::new_single_modality(n_obs as u64, n_vars as u64, 0, 3, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&obs_batch(n_obs)).unwrap();
    match layout {
        VarLayout::Single => writer.write_var(&var).unwrap(),
        VarLayout::Shards(rows) => {
            assert_eq!(
                rows.iter().sum::<usize>(),
                n_vars,
                "var shard row counts must tile the var axis"
            );
            let mut start = 0usize;
            for (idx, take) in rows.iter().enumerate() {
                writer
                    .write_var_shard(
                        idx as u32,
                        start as u64,
                        *take as u64,
                        n_vars as u64,
                        &var.slice(start, *take),
                    )
                    .unwrap();
                start += take;
            }
        }
    }

    let indptr: Vec<u64> = vec![0u64; n_obs + 1];
    writer
        .write_csr_shard(&indptr, &[], &[], CodecId::None, ValueEncoding::Uint8, 0)
        .unwrap();
    writer
        .write_csc_shard(
            &vec![0u64; n_vars + 1],
            &[],
            &[],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer
        .write_layer_csr_shard(
            "spliced",
            0,
            0,
            scx_format_io::writer::ShardBuffers::new(
                &indptr,
                &[],
                &[],
                CodecId::None,
                ValueEncoding::Uint8,
            ),
        )
        .unwrap();
    // raw carries its own, wider gene axis — a main-var column add must leave
    // it exactly as it was.
    writer
        .write_raw_csr_shard(&indptr, &[], &[], CodecId::None, ValueEncoding::Uint8, 0)
        .unwrap();
    writer.write_raw_var(&var_batch(n_vars + 2)).unwrap();
    let varm = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "pc1",
            DataType::Float32,
            true,
        )])),
        vec![Arc::new(Float32Array::from(vec![0.5f32; n_vars]))],
    )
    .unwrap();
    writer.write_varm("PCs", &varm).unwrap();
    writer
        .write_uns(&serde_json::json!({"state": "v0"}))
        .unwrap();
    if !index_var.is_empty() {
        let cols: Vec<String> = index_var.iter().map(|s| s.to_string()).collect();
        let pass = ObsVarIndexPass::carried(&[], &cols);
        let mut result = scx_engine::ConversionPredicateIndexResult::default();
        pass.write_var(&var, n_vars as u64, &mut writer, &mut result)
            .unwrap();
        assert_eq!(
            result.var_indexed_columns, cols,
            "fixture premise: the var predicate index must actually cover {index_var:?}"
        );
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
    Fixture {
        path,
        n_obs,
        n_vars,
    }
}

fn plain(dir: &Path, n_vars: usize) -> Fixture {
    write_fixture(dir, "a.scx", 4, var_batch(n_vars), VarLayout::Single, &[])
}

/// Per-gene annotations keyed by `row_keys`. `score(i)` varies per row so a
/// mis-join shows up in the values, not just the row count.
fn annot(row_keys: Vec<String>, score: impl Fn(usize) -> f32) -> ExternalVarData {
    let n = row_keys.len();
    let scores: Vec<f32> = (0..n).map(&score).collect();
    let symbols: Vec<String> = row_keys.iter().map(|k| format!("sym::{k}")).collect();
    let schema = Schema::new(vec![
        Field::new("peak_score", DataType::Float32, true),
        Field::new("norm_symbol", DataType::Utf8, true),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float32Array::from(scores)),
            Arc::new(StringArray::from(symbols)),
        ],
    )
    .unwrap();
    ExternalVarData {
        row_keys,
        row_annotations: batch,
        uns: serde_json::Map::new(),
        source_checksum: None,
        source_name: Some("peaks.csv".to_string()),
    }
}

fn opts() -> AttachVarOptions {
    AttachVarOptions {
        status_column: Some("peak_status".to_string()),
        provenance_action: "test_var_import".to_string(),
        ..Default::default()
    }
}

fn keys(prefix: &str, n: usize) -> Vec<String> {
    (0..n).map(|i| format!("{prefix}{i}")).collect()
}

fn gene_names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("GENE{i}")).collect()
}

fn f32_col(batch: &RecordBatch, name: &str) -> Float32Array {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("no column {name}"))
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap()
        .clone()
}

fn str_col(batch: &RecordBatch, name: &str) -> StringArray {
    let col = batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("no column {name}"));
    arrow::compute::cast(col, &DataType::Utf8)
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .clone()
}

/// Every catalog entry a var attach must leave byte- and offset-identical.
fn untouched(path: &Path) -> Vec<(String, u64, u64, [u8; 32])> {
    ScxReader::open(path)
        .unwrap()
        .catalog()
        .entries
        .iter()
        .filter(|e| {
            !matches!(
                e.section_type,
                SectionType::VarMetadata
                    | SectionType::VarMetadataShard
                    | SectionType::UnsBlob
                    | SectionType::Provenance
                    | SectionType::VarPredicateIndex
            )
        })
        .map(|e| (e.name.clone(), e.offset, e.length, e.checksum))
        .collect()
}

// ---------------------------------------------------------------------------
// The join is by key
// ---------------------------------------------------------------------------

#[test]
fn a_permuted_source_lands_each_value_on_its_own_gene() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 5);

    // Reverse the source rows: a positional attach would put GENE0's score on
    // GENE4 while producing a perfectly shaped column.
    let mut ks = gene_names(f.n_vars);
    ks.reverse();
    let score_of: std::collections::HashMap<String, f32> = ks
        .iter()
        .enumerate()
        .map(|(i, k)| (k.clone(), i as f32 * 10.0))
        .collect();
    let data = annot(ks, |i| i as f32 * 10.0);
    let s = attach_external_var(&f.path, &data, &opts()).unwrap();
    assert_eq!(s.n_matched, f.n_vars as u64);
    assert_eq!(s.var_key_column, "__index_level_0__");

    let var = ScxReader::open(&f.path).unwrap().read_var().unwrap();
    let got = f32_col(&var, "peak_score");
    let names = str_col(&var, "__index_level_0__");
    for i in 0..f.n_vars {
        assert_eq!(
            got.value(i),
            score_of[names.value(i)],
            "gene {} got another gene's score",
            names.value(i)
        );
    }
    assert_eq!(str_col(&var, "norm_symbol").value(0), "sym::GENE0");
}

#[test]
fn an_explicit_key_column_joins_on_that_column() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 4);
    let ks: Vec<String> = (0..f.n_vars).map(|i| format!("ENSG{i:05}")).collect();
    let data = annot(ks, |i| i as f32);
    let o = AttachVarOptions {
        join_key: AxisJoinKey::Column("gene_id".into()),
        ..opts()
    };
    let s = attach_external_var(&f.path, &data, &o).unwrap();
    assert_eq!(s.var_key_column, "gene_id");
    assert_eq!(s.n_matched, f.n_vars as u64);
}

#[test]
fn var_names_resolves_the_var_index() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 4);
    let data = annot(gene_names(f.n_vars), |i| i as f32);
    let o = AttachVarOptions {
        join_key: AxisJoinKey::Column("var_names".into()),
        ..opts()
    };
    let s = attach_external_var(&f.path, &data, &o).unwrap();
    // The summary reports the PHYSICAL name; the display translation is the
    // caller's job (`display_key_name`).
    assert_eq!(s.var_key_column, "__index_level_0__");
    assert_eq!(
        crate::display_key_name("var", &s.var_key_column),
        "var_names"
    );
}

#[test]
fn without_a_pandas_envelope_key_none_falls_through_to_the_gene_id_fallbacks() {
    let dir = tempfile::tempdir().unwrap();
    // A CLI-converted file carries no `pandas` schema metadata, so there is no
    // *declared* index and `key=None` resolves through VAR_KEY_FALLBACKS —
    // where `gene_id` is first. Pinned because it is the shape half the files
    // in the wild have, and it is a different answer from the test above.
    let var = var_batch(4);
    let stripped = RecordBatch::try_new(
        Arc::new(Schema::new(var.schema().fields().to_vec())),
        var.columns().to_vec(),
    )
    .unwrap();
    let f = write_fixture(dir.path(), "noenv.scx", 4, stripped, VarLayout::Single, &[]);
    let ks: Vec<String> = (0..f.n_vars).map(|i| format!("ENSG{i:05}")).collect();
    let data = annot(ks, |i| i as f32);
    let s = attach_external_var(&f.path, &data, &opts()).unwrap();
    assert_eq!(s.var_key_column, "gene_id");
    assert_eq!(s.n_matched, f.n_vars as u64);
}

#[test]
fn a_composite_key_disambiguates_repeated_gene_names() {
    let dir = tempfile::tempdir().unwrap();
    // Two genes share a symbol; only (symbol, id) is unique.
    let schema = Schema::new(vec![
        Field::new("__index_level_0__", DataType::Utf8, false),
        Field::new("gene_id", DataType::Utf8, false),
    ])
    .with_metadata(pandas_envelope());
    let var = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(vec!["DUP", "DUP", "X"])),
            Arc::new(StringArray::from(vec!["e1", "e2", "e3"])),
        ],
    )
    .unwrap();
    let f = write_fixture(dir.path(), "c.scx", 3, var, VarLayout::Single, &[]);

    let cols = vec!["var_names".to_string(), "gene_id".to_string()];
    let src = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("var_names", DataType::Utf8, false),
            Field::new("gene_id", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["DUP", "X", "DUP"])),
            Arc::new(StringArray::from(vec!["e2", "e3", "e1"])),
        ],
    )
    .unwrap();
    let row_keys = crate::build_composite_key_for("var", &src, &cols).unwrap();
    let data = annot(row_keys, |i| i as f32 + 100.0);
    let o = AttachVarOptions {
        join_key: AxisJoinKey::Composite { columns: cols },
        ..opts()
    };
    let s = attach_external_var(&f.path, &data, &o).unwrap();
    assert_eq!(s.n_matched, 3);
    assert_eq!(s.var_key_column, "__index_level_0__,gene_id");

    let var = ScxReader::open(&f.path).unwrap().read_var().unwrap();
    let got = f32_col(&var, "peak_score");
    // Source row 2 is (DUP, e1) -> file row 0; row 0 is (DUP, e2) -> file row 1.
    assert_eq!(got.value(0), 102.0);
    assert_eq!(got.value(1), 100.0);
    assert_eq!(got.value(2), 101.0);
}

#[test]
fn duplicate_target_keys_are_refused_with_a_working_key_named() {
    let dir = tempfile::tempdir().unwrap();
    let schema = Schema::new(vec![
        Field::new("__index_level_0__", DataType::Utf8, false),
        Field::new("gene_id", DataType::Utf8, false),
    ])
    .with_metadata(pandas_envelope());
    let var = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(vec!["DUP", "DUP"])),
            Arc::new(StringArray::from(vec!["e1", "e2"])),
        ],
    )
    .unwrap();
    let f = write_fixture(dir.path(), "d.scx", 2, var, VarLayout::Single, &[]);
    let data = annot(vec!["DUP".into(), "OTHER".into()], |i| i as f32);
    let err = attach_external_var(&f.path, &data, &opts()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("duplicates"), "{msg}");
    assert!(
        msg.contains("gene_id"),
        "the diagnosis must name the column that WOULD key: {msg}"
    );
    assert!(
        msg.contains("var columns that ARE unique"),
        "the diagnosis must talk about var, not obs: {msg}"
    );
}

#[test]
fn zero_overlap_is_a_hard_error() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 3);
    let data = annot(keys("nothing_", 3), |i| i as f32);
    let err = attach_external_var(&f.path, &data, &opts()).unwrap_err();
    assert!(err.to_string().contains("no target row key matched"));
}

// ---------------------------------------------------------------------------
// Row policies
// ---------------------------------------------------------------------------

#[test]
fn uncovered_genes_get_null_and_an_absent_status() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 4);
    // Covers GENE0 and GENE2 only.
    let data = annot(vec!["GENE0".into(), "GENE2".into()], |i| i as f32 + 1.0);
    let s = attach_external_var(&f.path, &data, &opts()).unwrap();
    assert_eq!((s.n_matched, s.n_target_rows_absent), (2, 2));

    let var = ScxReader::open(&f.path).unwrap().read_var().unwrap();
    let got = f32_col(&var, "peak_score");
    assert!(got.is_valid(0) && got.is_valid(2));
    assert!(
        got.is_null(1) && got.is_null(3),
        "an uncovered gene must be null, never a fabricated 0.0"
    );
    let status = str_col(&var, "peak_status");
    assert_eq!(
        (status.value(0), status.value(1)),
        ("present", "absent"),
        "the status marker must distinguish the two"
    );
}

#[test]
fn the_two_row_error_policies_are_honoured() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 4);

    let partial = annot(vec!["GENE0".into()], |_| 1.0);
    let o = AttachVarOptions {
        missing_row_policy: MissingRowPolicy::Error,
        ..opts()
    };
    let err = attach_external_var(&f.path, &partial, &o).unwrap_err();
    assert!(err.to_string().contains("missing_row_policy = Error"));

    let mut extra = gene_names(f.n_vars);
    extra.push("GENE_NOT_IN_FILE".into());
    let data = annot(extra, |i| i as f32);
    let o = AttachVarOptions {
        extra_row_policy: ExtraRowPolicy::Error,
        ..opts()
    };
    let err = attach_external_var(&f.path, &data, &o).unwrap_err();
    assert!(err.to_string().contains("extra_row_policy = Error"));
}

#[test]
fn overwrite_replaces_and_does_not_merge() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 4);

    let first = annot(vec!["GENE0".into(), "GENE1".into()], |_| 1.0);
    attach_external_var(&f.path, &first, &opts()).unwrap();

    let second = annot(vec!["GENE2".into(), "GENE3".into()], |_| 2.0);
    let err = attach_external_var(&f.path, &second, &opts()).unwrap_err();
    assert!(
        err.to_string().contains("pass overwrite=true"),
        "a second import without overwrite must fail loudly"
    );

    let o = AttachVarOptions {
        overwrite: true,
        ..opts()
    };
    attach_external_var(&f.path, &second, &o).unwrap();
    let var = ScxReader::open(&f.path).unwrap().read_var().unwrap();
    let got = f32_col(&var, "peak_score");
    assert!(
        got.is_null(0) && got.is_null(1),
        "if this starts passing the first import's values through, the \
         concatenate-first contract has silently changed"
    );
    assert_eq!((got.value(2), got.value(3)), (2.0, 2.0));
}

// ---------------------------------------------------------------------------
// Positional
// ---------------------------------------------------------------------------

#[test]
fn a_positional_attach_lands_row_for_row() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 4);
    let mut data = annot(gene_names(f.n_vars), |i| i as f32 * 7.0);
    data.row_keys.clear();
    let o = AttachVarOptions {
        join_key: AxisJoinKey::Positional,
        status_column: None,
        ..opts()
    };
    let s = attach_external_var(&f.path, &data, &o).unwrap();
    assert_eq!(s.var_key_column, "<positional>");
    assert_eq!(s.n_matched, f.n_vars as u64);
    let var = ScxReader::open(&f.path).unwrap().read_var().unwrap();
    let got = f32_col(&var, "peak_score");
    for i in 0..f.n_vars {
        assert_eq!(got.value(i), i as f32 * 7.0);
    }
}

#[test]
fn a_positional_frame_of_the_wrong_length_is_refused_naming_n_vars() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 4);
    let mut data = annot(gene_names(3), |i| i as f32);
    data.row_keys.clear();
    let o = AttachVarOptions {
        join_key: AxisJoinKey::Positional,
        status_column: None,
        ..opts()
    };
    let err = attach_external_var(&f.path, &data, &o).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("n_vars = 4") && msg.contains("3 rows"),
        "{msg}"
    );
}

#[test]
fn a_reordered_positional_frame_is_refused_by_its_own_labels() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 4);
    // Right length, right labels, wrong order — every value on the wrong gene.
    let mut ks = gene_names(f.n_vars);
    ks.swap(0, 3);
    let data = annot(ks, |i| i as f32);
    let o = AttachVarOptions {
        join_key: AxisJoinKey::Positional,
        status_column: None,
        ..opts()
    };
    let err = attach_external_var(&f.path, &data, &o).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("different order"), "{msg}");
    assert!(msg.contains("GENE3") && msg.contains("GENE0"), "{msg}");
}

#[test]
fn positional_labels_from_another_source_are_ignored() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 4);
    // Labels that are not this file's gene names at all: positional has always
    // ignored the index, and a frame built elsewhere is still "row i -> row i".
    let data = annot(keys("other_", f.n_vars), |i| i as f32);
    let o = AttachVarOptions {
        join_key: AxisJoinKey::Positional,
        status_column: None,
        ..opts()
    };
    attach_external_var(&f.path, &data, &o).unwrap();
}

#[test]
fn a_status_column_is_refused_under_positional() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 4);
    let mut data = annot(gene_names(f.n_vars), |i| i as f32);
    data.row_keys.clear();
    let o = AttachVarOptions {
        join_key: AxisJoinKey::Positional,
        ..opts()
    };
    let err = attach_external_var(&f.path, &data, &o).unwrap_err();
    assert!(err.to_string().contains("status_column is meaningless"));
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

fn var_shard_ranges(path: &Path) -> Vec<(String, u64, u64)> {
    let reader = ScxReader::open(path).unwrap();
    let mut v: Vec<(String, u64, u64)> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::VarMetadataShard)
        .map(|e| {
            let s = e.stats.as_ref().expect("a var shard stamps its row range");
            (e.name.clone(), s.row_start, s.row_end)
        })
        .collect();
    v.sort();
    v
}

#[test]
fn a_sharded_var_keeps_its_shard_boundaries() {
    let dir = tempfile::tempdir().unwrap();
    // 2 + 3 + 1: unequal, and not what `shard_target_rows = 3` would produce,
    // so a rewrite that re-derives the split cannot pass by coincidence.
    let f = write_fixture(
        dir.path(),
        "s.scx",
        4,
        var_batch(6),
        VarLayout::Shards(&[2, 3, 1]),
        &[],
    );
    let before = var_shard_ranges(&f.path);
    assert_eq!(before.len(), 3, "fixture premise: three var shards");

    let data = annot(gene_names(f.n_vars), |i| i as f32);
    let s = attach_external_var(&f.path, &data, &opts()).unwrap();
    assert!(s.var_streamed);

    assert_eq!(
        var_shard_ranges(&f.path),
        before,
        "a sharded var must come back with the same shard count and ranges"
    );
    // And the values must still be on the right genes across the boundaries.
    let var = ScxReader::open(&f.path).unwrap().read_var().unwrap();
    let got = f32_col(&var, "peak_score");
    let names = str_col(&var, "__index_level_0__");
    for i in 0..f.n_vars {
        assert_eq!(names.value(i), format!("GENE{i}"));
        assert_eq!(got.value(i), i as f32);
    }
}

#[test]
fn a_single_section_var_stays_a_single_section() {
    let dir = tempfile::tempdir().unwrap();
    let f = write_fixture(
        dir.path(),
        "one.scx",
        4,
        var_batch(6),
        VarLayout::Single,
        &[],
    );
    let data = annot(gene_names(f.n_vars), |i| i as f32);
    let s = attach_external_var(&f.path, &data, &opts()).unwrap();
    assert!(!s.var_streamed);

    let reader = ScxReader::open(&f.path).unwrap();
    assert_eq!(
        reader.var_metadata_shard_count(),
        0,
        "an attach must not shard a var that arrived as one section"
    );
    assert_eq!(
        reader
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::VarMetadata)
            .count(),
        1,
        "exactly one var section, not the old one plus a new one"
    );
}

// ---------------------------------------------------------------------------
// The predicate index, and the stats that are NOT ours
// ---------------------------------------------------------------------------

fn var_index_bytes(path: &Path) -> Option<Vec<u8>> {
    let reader = ScxReader::open(path).unwrap();
    reader
        .read_var_predicate_index_bytes()
        .unwrap()
        .map(|b| b.to_vec())
}

#[test]
fn a_pure_add_carries_the_var_predicate_index_verbatim() {
    let dir = tempfile::tempdir().unwrap();
    let f = write_fixture(
        dir.path(),
        "idx.scx",
        4,
        var_batch(4),
        VarLayout::Single,
        &["feature_type"],
    );
    let before = var_index_bytes(&f.path).expect("fixture premise: an index exists");

    let data = annot(gene_names(f.n_vars), |i| i as f32);
    let s = attach_external_var(&f.path, &data, &opts()).unwrap();
    assert!(
        !s.var_index_rebuilt,
        "a pure add leaves the index valid — it keys on column name"
    );
    assert_eq!(
        var_index_bytes(&f.path).as_deref(),
        Some(before.as_slice()),
        "a pure add must not even rewrite the index section"
    );
}

#[test]
fn overwriting_an_indexed_var_column_rebuilds_the_index_over_the_new_values() {
    let dir = tempfile::tempdir().unwrap();
    let f = write_fixture(
        dir.path(),
        "idx2.scx",
        4,
        var_batch(4),
        VarLayout::Single,
        &["feature_type"],
    );
    let before = var_index_bytes(&f.path).unwrap();

    // Overwrite the indexed column with values it has never held.
    let schema = Schema::new(vec![Field::new("feature_type", DataType::Utf8, true)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(vec![
            "Antibody Capture",
            "Antibody Capture",
            "Antibody Capture",
            "Antibody Capture",
        ]))],
    )
    .unwrap();
    let data = ExternalVarData {
        row_keys: gene_names(f.n_vars),
        row_annotations: batch,
        uns: serde_json::Map::new(),
        source_checksum: None,
        source_name: None,
    };
    let o = AttachVarOptions {
        overwrite: true,
        status_column: None,
        ..opts()
    };
    let s = attach_external_var(&f.path, &data, &o).unwrap();
    assert!(s.var_index_rebuilt);
    assert!(
        s.var_columns_not_carried.is_empty(),
        "the column is still a string, so the rebuild covers it"
    );

    let after = var_index_bytes(&f.path).expect("the index must be rebuilt, not dropped");
    assert_ne!(after, before, "a rebuild over new values must differ");
    let idx =
        scx_engine::PredicateIndex::read_from(&mut std::io::Cursor::new(after.as_slice())).unwrap();
    let covered: Vec<String> = idx
        .columns
        .iter()
        .map(|c| match c {
            scx_engine::index::IndexedColumn::Categorical(cat) => cat.column_name.clone(),
            scx_engine::index::IndexedColumn::Numeric(num) => num.column_name.clone(),
        })
        .collect();
    assert_eq!(covered, vec!["feature_type".to_string()]);
    match &idx.columns[0] {
        scx_engine::index::IndexedColumn::Categorical(cat) => {
            let values: Vec<&str> = cat.entries.iter().map(|e| e.value.as_str()).collect();
            assert_eq!(
                values,
                vec!["Antibody Capture"],
                "the rebuilt index must describe the NEW values, not the old ones"
            );
        }
        other => panic!("expected a categorical index, got {other:?}"),
    }
}

/// A file whose CSR shards carry per-shard **obs** `ColumnStat`s — the input
/// obs-axis Level-1 pushdown prunes from. Nothing else in this file produces
/// them, so without this the "stats survive" test below would compare an empty
/// list to an empty list and pass however the op behaved.
fn fixture_with_obs_column_stats(dir: &Path, name: &str) -> Fixture {
    let n_obs = 4usize;
    let var = var_batch(4);
    let n_vars = var.num_rows();
    let obs = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("barcode", DataType::Utf8, false),
            Field::new("cell_type", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec![
                "cell_0", "cell_1", "cell_2", "cell_3",
            ])),
            Arc::new(StringArray::from(vec!["T", "B", "T", "B"])),
        ],
    )
    .unwrap();

    let path = dir.join(name);
    let header = FileHeader::new_single_modality(n_obs as u64, n_vars as u64, 0, 2, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&obs).unwrap();
    writer.write_var(&var).unwrap();
    let ranges: Vec<(u64, u64)> = vec![(0, 2), (2, 4)];
    for (start, _) in &ranges {
        writer
            .write_csr_shard(
                &[0u64; 3],
                &[],
                &[],
                CodecId::None,
                ValueEncoding::Uint8,
                *start,
            )
            .unwrap();
    }
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
        &ranges,
        &opts,
        &mut outcomes,
        &mut named,
    )
    .unwrap()
    .expect("cell_type must be indexable");
    writer.write_obs_predicate_index(&bytes).unwrap();
    scx_engine::apply_obs_shard_column_stats(&mut writer, &bytes, ranges.len()).unwrap();
    writer
        .write_uns(&serde_json::json!({"state": "v0"}))
        .unwrap();
    writer.finish().unwrap();
    Fixture {
        path,
        n_obs,
        n_vars,
    }
}

fn obs_shard_column_stats(path: &Path) -> Vec<(String, usize)> {
    ScxReader::open(path)
        .unwrap()
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CsrShard)
        .map(|e| {
            (
                e.name.clone(),
                e.stats.as_ref().map_or(0, |s| s.column_stats.len()),
            )
        })
        .collect()
}

#[test]
fn a_var_attach_does_not_clear_the_obs_shard_column_stats() {
    let dir = tempfile::tempdir().unwrap();
    let f = fixture_with_obs_column_stats(dir.path(), "stats.scx");
    let before = obs_shard_column_stats(&f.path);
    assert!(
        !before.is_empty() && before.iter().all(|(_, n)| *n > 0),
        "fixture premise: every CSR shard must carry obs column stats, got {before:?}"
    );

    let data = annot(gene_names(f.n_vars), |i| i as f32);
    attach_external_var(&f.path, &data, &opts()).unwrap();

    assert_eq!(
        obs_shard_column_stats(&f.path),
        before,
        "those stats are the obs-axis Level-1 pushdown's input; a var attach \
         must not touch them"
    );
    assert!(
        ScxReader::open(&f.path)
            .unwrap()
            .read_obs_predicate_index_bytes()
            .unwrap()
            .is_some(),
        "and the obs predicate index itself must survive"
    );
}

// ---------------------------------------------------------------------------
// Nothing else is lost
// ---------------------------------------------------------------------------

#[test]
fn obs_x_layers_csc_raw_and_varm_are_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let f = write_fixture(dir.path(), "u.scx", 4, var_batch(4), VarLayout::Single, &[]);
    let before = untouched(&f.path);
    let before_gens = {
        let cat = ScxReader::open(&f.path).unwrap().catalog().clone();
        (cat.data_generation, cat.csc_build_generation)
    };
    // Premise: the snapshot actually covers the families it claims to.
    for ty in [
        SectionType::ObsMetadata,
        SectionType::CsrShard,
        SectionType::CscShard,
        SectionType::LayerCsrShard,
        SectionType::RawCsrShard,
        SectionType::RawVarMetadata,
        SectionType::VarmEmbedding,
    ] {
        assert!(
            ScxReader::open(&f.path)
                .unwrap()
                .catalog()
                .entries
                .iter()
                .any(|e| e.section_type == ty),
            "fixture premise: no {ty:?} section to protect"
        );
    }

    let data = annot(gene_names(f.n_vars), |i| i as f32);
    attach_external_var(&f.path, &data, &opts()).unwrap();

    assert_eq!(
        untouched(&f.path),
        before,
        "a var-only attach must not rewrite or move anything else"
    );
    let cat = ScxReader::open(&f.path).unwrap().catalog().clone();
    assert_eq!(
        (cat.data_generation, cat.csc_build_generation),
        before_gens,
        "bumping either would silently invalidate the CSC sidecar"
    );
    assert_eq!(cat.n_obs, f.n_obs as u64);
}

#[test]
fn rollback_restores_the_pre_attach_state() {
    let dir = tempfile::tempdir().unwrap();
    let f = write_fixture(dir.path(), "r.scx", 4, var_batch(4), VarLayout::Single, &[]);
    let before = std::fs::read(&f.path).unwrap();

    let mut data = annot(gene_names(f.n_vars), |i| i as f32);
    data.uns.insert("peaks".into(), serde_json::json!({"n": 4}));
    attach_external_var(&f.path, &data, &opts()).unwrap();
    let var = ScxReader::open(&f.path).unwrap().read_var().unwrap();
    assert!(var.column_by_name("peak_score").is_some());

    crate::rollback(&f.path).unwrap();
    let var = ScxReader::open(&f.path).unwrap().read_var().unwrap();
    assert!(
        var.column_by_name("peak_score").is_none(),
        "one rollback must undo the columns"
    );
    let uns = ScxReader::open(&f.path).unwrap().read_uns().unwrap();
    assert!(
        uns.get("peaks").is_none(),
        "and the uns payload that rode in the same commit"
    );
    // The append-only body means the prefix is still the original file.
    let after = std::fs::read(&f.path).unwrap();
    assert!(after.len() >= before.len());
}

#[test]
fn uns_entries_ride_in_the_same_commit() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 4);
    let mut data = annot(gene_names(f.n_vars), |i| i as f32);
    data.uns.insert("a".into(), serde_json::json!(1));
    data.uns.insert("b".into(), serde_json::json!("two"));
    attach_external_var(&f.path, &data, &opts()).unwrap();

    let uns = ScxReader::open(&f.path).unwrap().read_uns().unwrap();
    assert_eq!(uns.get("a"), Some(&serde_json::json!(1)));
    assert_eq!(uns.get("b"), Some(&serde_json::json!("two")));
    assert_eq!(
        uns.get("state"),
        Some(&serde_json::json!("v0")),
        "pre-existing keys must survive the shallow merge"
    );
}

#[test]
fn a_colliding_uns_key_is_refused_without_overwrite() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 4);
    let mut data = annot(gene_names(f.n_vars), |i| i as f32);
    data.uns.insert("state".into(), serde_json::json!("v1"));
    let err = attach_external_var(&f.path, &data, &opts()).unwrap_err();
    assert!(err.to_string().contains("uns key 'state' already exists"));
}

// ---------------------------------------------------------------------------
// Categoricals
// ---------------------------------------------------------------------------

#[test]
fn a_categorical_annotation_lands_as_a_dictionary_with_its_flag() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 4);

    // Three declared levels, only two used, ordered.
    let values = StringArray::from(vec!["promoter", "enhancer", "intergenic"]);
    let dict: DictionaryArray<Int8Type> = DictionaryArray::try_new(
        Int8Array::from(vec![Some(0), Some(1), Some(0), Some(1)]),
        Arc::new(values),
    )
    .unwrap();
    let mut md = std::collections::HashMap::new();
    md.insert(
        scx_format_io::CATEGORICAL_ORDERED_KEY.to_string(),
        "true".to_string(),
    );
    let field = Field::new(
        "peak_class",
        DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
        true,
    )
    .with_metadata(md);
    let batch =
        RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![Arc::new(dict)]).unwrap();
    let data = ExternalVarData {
        row_keys: gene_names(f.n_vars),
        row_annotations: batch,
        uns: serde_json::Map::new(),
        source_checksum: None,
        source_name: None,
    };
    let o = AttachVarOptions {
        status_column: None,
        ..opts()
    };
    attach_external_var(&f.path, &data, &o).unwrap();

    let var = ScxReader::open(&f.path).unwrap().read_var().unwrap();
    let field = var.schema().field_with_name("peak_class").unwrap().clone();
    assert!(
        matches!(field.data_type(), DataType::Dictionary(_, _)),
        "a categorical must not be demoted to plain strings, got {:?}",
        field.data_type()
    );
    assert_eq!(
        field
            .metadata()
            .get(scx_format_io::CATEGORICAL_ORDERED_KEY)
            .map(String::as_str),
        Some("true"),
        "the ordered flag must ride along"
    );
    let col = var.column_by_name("peak_class").unwrap();
    let dict = col
        .as_any()
        .downcast_ref::<DictionaryArray<Int8Type>>()
        .unwrap();
    let vals = dict
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(
        (0..vals.len()).map(|i| vals.value(i)).collect::<Vec<_>>(),
        vec!["promoter", "enhancer", "intergenic"],
        "the declared vocabulary, unused level included, must survive"
    );
}

// ---------------------------------------------------------------------------
// Guards
// ---------------------------------------------------------------------------

#[test]
fn a_multimodal_target_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = crate::test_utils::fixture_multimodal(&dir);
    let data = annot(gene_names(2), |i| i as f32);
    let err = attach_external_var(&path, &data, &opts()).unwrap_err();
    assert!(
        matches!(err, OpsError::MultimodalUnsupported { op } if op == "attach_external_var"),
        "got {err}"
    );
}

#[test]
fn nothing_to_attach_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 4);
    let empty = RecordBatch::new_empty(Arc::new(Schema::empty()));
    let data = ExternalVarData {
        row_keys: Vec::new(),
        row_annotations: empty,
        uns: serde_json::Map::new(),
        source_checksum: None,
        source_name: None,
    };
    let o = AttachVarOptions {
        status_column: None,
        ..opts()
    };
    let err = attach_external_var(&f.path, &data, &o).unwrap_err();
    assert!(err.to_string().contains("nothing to attach"));
}

#[test]
fn a_status_column_matching_an_annotation_name_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 4);
    let data = annot(gene_names(f.n_vars), |i| i as f32);
    let o = AttachVarOptions {
        status_column: Some("norm_symbol".to_string()),
        ..opts()
    };
    let err = attach_external_var(&f.path, &data, &o).unwrap_err();
    assert!(err
        .to_string()
        .contains("is also an annotation column in the"));
}

#[test]
fn dry_run_writes_nothing_and_reports_the_real_match_count() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 4);
    let before = std::fs::read(&f.path).unwrap();

    let data = annot(vec!["GENE0".into(), "GENE1".into()], |i| i as f32);
    let o = AttachVarOptions {
        dry_run: true,
        ..opts()
    };
    let s = attach_external_var(&f.path, &data, &o).unwrap();
    assert_eq!((s.n_matched, s.n_target_rows_absent), (2, 2));
    assert_eq!(s.var_columns_added.len(), 3, "status + two annotations");

    assert_eq!(
        std::fs::read(&f.path).unwrap(),
        before,
        "a dry run must leave the file byte-identical"
    );
}

#[test]
fn a_row_count_mismatch_is_caught_before_any_write() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 4);
    let before = std::fs::read(&f.path).unwrap();
    let mut data = annot(gene_names(3), |i| i as f32);
    data.row_keys.push("GENE3".into()); // 4 keys, 3 annotation rows
    let err = attach_external_var(&f.path, &data, &opts()).unwrap_err();
    assert!(err.to_string().contains("row keys"));
    assert_eq!(std::fs::read(&f.path).unwrap(), before);
}

// ---------------------------------------------------------------------------
// diagnose_var_key
// ---------------------------------------------------------------------------

#[test]
fn diagnose_var_key_names_the_usable_key_and_sets_aside_the_unusable() {
    let dir = tempfile::tempdir().unwrap();
    // `var_names` duplicated, `gene_id` unique, a unique float that cannot key.
    let schema = Schema::new(vec![
        Field::new("__index_level_0__", DataType::Utf8, false),
        Field::new("gene_id", DataType::Utf8, false),
        Field::new("gc_content", DataType::Float32, true),
    ])
    .with_metadata(pandas_envelope());
    let var = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(vec!["DUP", "DUP", "X"])),
            Arc::new(StringArray::from(vec!["e1", "e2", "e3"])),
            Arc::new(Float32Array::from(vec![0.1f32, 0.2, 0.3])),
        ],
    )
    .unwrap();
    let f = write_fixture(dir.path(), "diag.scx", 3, var, VarLayout::Single, &[]);

    let d = diagnose_var_key(&f.path, None).unwrap();
    assert_eq!(d.axis, "var");
    assert_eq!(d.n_rows, 3);
    assert_eq!(d.unique_columns, vec!["gene_id".to_string()]);
    assert_eq!(d.suggestion.as_deref(), Some("gene_id"));
    assert_eq!(
        d.unusable_unique_columns,
        vec!["gc_content".to_string()],
        "a unique float must be reported apart, not offered as a key"
    );
    assert!(d.describe().contains("var columns that ARE unique"));
}

#[test]
fn a_boolean_annotation_survives_the_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 4);
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "highly_variable",
            DataType::Boolean,
            true,
        )])),
        vec![Arc::new(BooleanArray::from(vec![true, false, true, false]))],
    )
    .unwrap();
    let data = ExternalVarData {
        row_keys: gene_names(f.n_vars),
        row_annotations: batch,
        uns: serde_json::Map::new(),
        source_checksum: None,
        source_name: None,
    };
    let o = AttachVarOptions {
        status_column: None,
        ..opts()
    };
    attach_external_var(&f.path, &data, &o).unwrap();
    let var = ScxReader::open(&f.path).unwrap().read_var().unwrap();
    let col = var
        .column_by_name("highly_variable")
        .unwrap()
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap()
        .clone();
    assert_eq!(
        (0..4).map(|i| col.value(i)).collect::<Vec<_>>(),
        vec![true, false, true, false]
    );
}

#[test]
fn provenance_records_the_key_and_the_counts() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 4);
    let data = annot(vec!["GENE0".into(), "GENE1".into()], |i| i as f32);
    attach_external_var(&f.path, &data, &opts()).unwrap();

    let reader = ScxReader::open(&f.path).unwrap();
    let prov = reader.read_provenance().unwrap();
    let last = prov.operations.last().unwrap();
    assert_eq!(last.action, "test_var_import");
    let params: serde_json::Value = serde_json::from_str(&last.params_json).unwrap();
    assert_eq!(params["n_vars"], 4);
    assert_eq!(params["n_matched"], 2);
    assert_eq!(params["var_key_column"], "__index_level_0__");
    assert_eq!(params["var_streamed"], false);
    assert_eq!(params["source_file"], "peaks.csv");
}

// ---------------------------------------------------------------------------
// Round-1 review follow-ups
// ---------------------------------------------------------------------------

/// A streamed var write must rebuild from each shard **as read**, not slice an
/// assembled table.
///
/// `read_var()` reconciles a var whose shards disagree on a column's encoding —
/// one dictionary-encoded, one plain `Utf8`, which is what an in-place rewrite
/// by an older writer leaves behind — so a slice of the assembled batch writes
/// the *unified* encoding to every shard. Measured: slicing turns
/// `[Dictionary(Int32, Utf8), Utf8]` into
/// `[Dictionary(Int8, Utf8), Dictionary(Int8, Utf8)]`, rewriting bytes for a
/// column the attach was not asked to touch. Every other test in this file
/// passes under either strategy, so without this one the choice is unpinned.
#[test]
fn a_var_whose_shards_disagree_on_encoding_keeps_each_shards_own() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.scx");

    let dict: DictionaryArray<Int32Type> = DictionaryArray::try_new(
        Int32Array::from(vec![0, 1]),
        Arc::new(StringArray::from(vec!["a", "b"])),
    )
    .unwrap();
    let dict_type = dict.data_type().clone();
    let shard0 = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("gene_id", DataType::Utf8, false),
            Field::new("kind", dict_type, true),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["ENSG0", "ENSG1"])),
            Arc::new(dict),
        ],
    )
    .unwrap();
    let shard1 = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("gene_id", DataType::Utf8, false),
            Field::new("kind", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["ENSG2", "ENSG3"])),
            Arc::new(StringArray::from(vec!["a", "c"])),
        ],
    )
    .unwrap();

    let mut w = ScxWriter::new(&path, FileHeader::new_single_modality(2, 4, 0, 2, 0, 0)).unwrap();
    w.write_obs(&obs_batch(2)).unwrap();
    w.write_var_shard(0, 0, 2, 4, &shard0).unwrap();
    w.write_var_shard(1, 2, 2, 4, &shard1).unwrap();
    w.write_csr_shard(&[0u64; 3], &[], &[], CodecId::None, ValueEncoding::Uint8, 0)
        .unwrap();
    w.finish().unwrap();

    let per_shard_kinds = |p: &Path| -> Vec<String> {
        let r = ScxReader::open(p).unwrap();
        (0..2)
            .map(|i| {
                format!(
                    "{:?}",
                    r.read_var_shard(i)
                        .unwrap()
                        .schema()
                        .field_with_name("kind")
                        .unwrap()
                        .data_type()
                )
            })
            .collect()
    };
    let before = per_shard_kinds(&path);
    assert_eq!(
        before,
        vec!["Dictionary(Int32, Utf8)".to_string(), "Utf8".to_string()],
        "fixture premise: the two shards must actually disagree"
    );

    let data = annot(gene_names_ensg(4), |i| i as f32);
    let o = AttachVarOptions {
        join_key: AxisJoinKey::Column("gene_id".into()),
        status_column: None,
        ..opts()
    };
    attach_external_var(&path, &data, &o).unwrap();

    assert_eq!(
        per_shard_kinds(&path),
        before,
        "each shard must keep its own encoding for a column the attach did not touch"
    );
    // And the values still landed.
    let var = ScxReader::open(&path).unwrap().read_var().unwrap();
    assert_eq!(
        (0..4)
            .map(|i| f32_col(&var, "peak_score").value(i))
            .collect::<Vec<_>>(),
        vec![0.0, 1.0, 2.0, 3.0]
    );
}

fn gene_names_ensg(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("ENSG{i}")).collect()
}

/// A `dry_run` must report the index outcome the real import would produce.
///
/// `var_index_rebuilt` can be decided from the planning boolean, but
/// `var_columns_not_carried` cannot: whether a covered column survives depends
/// on the *new* values. Overwriting an indexed string column with a boolean is
/// the case — the preview used to say `[]` and the write then named the drop.
#[test]
fn dry_run_previews_the_index_outcome_the_write_would_produce() {
    let dir = tempfile::tempdir().unwrap();
    let f = write_fixture(
        dir.path(),
        "prev.scx",
        4,
        var_batch(4),
        VarLayout::Single,
        &["feature_type"],
    );
    let before = std::fs::read(&f.path).unwrap();

    // A boolean replacement for the indexed string column: nothing left to
    // index, so the stale section is retired with no replacement.
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "feature_type",
            DataType::Boolean,
            true,
        )])),
        vec![Arc::new(BooleanArray::from(vec![true, false, true, false]))],
    )
    .unwrap();
    let data = ExternalVarData {
        row_keys: gene_names(f.n_vars),
        row_annotations: batch,
        uns: serde_json::Map::new(),
        source_checksum: None,
        source_name: None,
    };
    let base = AttachVarOptions {
        overwrite: true,
        status_column: None,
        ..opts()
    };

    let preview = attach_external_var(
        &f.path,
        &data,
        &AttachVarOptions {
            dry_run: true,
            overwrite: true,
            status_column: None,
            ..opts()
        },
    )
    .unwrap();
    assert_eq!(
        std::fs::read(&f.path).unwrap(),
        before,
        "a dry run must still write nothing"
    );

    let real = attach_external_var(&f.path, &data, &base).unwrap();
    assert_eq!(
        (
            preview.var_index_rebuilt,
            preview.var_index_dropped,
            preview.var_columns_not_carried.clone()
        ),
        (
            real.var_index_rebuilt,
            real.var_index_dropped,
            real.var_columns_not_carried.clone()
        ),
        "the preview must agree with the write it previews"
    );
    // And the outcome is the honest one: retired, not "rebuilt".
    assert!(!real.var_index_rebuilt, "no replacement could be built");
    assert!(real.var_index_dropped);
    assert_eq!(
        real.var_columns_not_carried,
        vec!["feature_type".to_string()]
    );
    assert!(
        var_index_bytes(&f.path).is_none(),
        "and no VarPredicateIndex section is left behind"
    );
}

/// The failure hint names the identifiers a *gene* join actually mismatches on.
#[test]
fn a_zero_overlap_var_join_does_not_mention_barcode_suffixes() {
    let dir = tempfile::tempdir().unwrap();
    let f = plain(dir.path(), 3);
    let data = annot(keys("nothing_", 3), |i| i as f32);
    let msg = attach_external_var(&f.path, &data, &opts())
        .unwrap_err()
        .to_string();
    assert!(
        msg.contains("Ensembl accession") && msg.contains("versioned"),
        "a gene join that misses is not a barcode-suffix problem: {msg}"
    );
    assert!(
        !msg.contains("sample-name prefix") && !msg.contains("'-1' suffix"),
        "the obs wording must not reach a var caller: {msg}"
    );
}

/// The middle case: a two-column index where one column survives the overwrite
/// and one does not.
///
/// The two existing arms cover all-survive (`rebuilt`, empty not-carried) and
/// none-survive (`dropped`, one name). Neither pins the branch where the
/// replacement section exists *and* something was lost — which is the outcome
/// that most needs reporting, since a caller sees a live index and would not
/// think to check what it still covers.
#[test]
fn a_partial_index_loss_is_rebuilt_and_still_names_what_it_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let f = write_fixture(
        dir.path(),
        "partial.scx",
        4,
        var_batch(4),
        VarLayout::Single,
        &["feature_type", "gene_id"],
    );

    // Overwrite only `feature_type`, with booleans. `gene_id` is untouched and
    // still indexable, so a replacement section exists — but `feature_type` is
    // gone from it.
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "feature_type",
            DataType::Boolean,
            true,
        )])),
        vec![Arc::new(BooleanArray::from(vec![true, false, true, false]))],
    )
    .unwrap();
    let data = ExternalVarData {
        row_keys: gene_names(f.n_vars),
        row_annotations: batch,
        uns: serde_json::Map::new(),
        source_checksum: None,
        source_name: None,
    };
    let o = AttachVarOptions {
        overwrite: true,
        status_column: None,
        ..opts()
    };
    let s = attach_external_var(&f.path, &data, &o).unwrap();

    assert!(s.var_index_rebuilt, "gene_id is still indexable");
    assert!(!s.var_index_dropped, "so the section is not retired");
    assert_eq!(
        s.var_columns_not_carried,
        vec!["feature_type".to_string()],
        "and the column that fell out must still be named"
    );

    let bytes = var_index_bytes(&f.path).expect("a replacement section exists");
    let idx =
        scx_engine::PredicateIndex::read_from(&mut std::io::Cursor::new(bytes.as_slice())).unwrap();
    let covered: Vec<String> = idx
        .columns
        .iter()
        .map(|c| match c {
            scx_engine::index::IndexedColumn::Categorical(cat) => cat.column_name.clone(),
            scx_engine::index::IndexedColumn::Numeric(num) => num.column_name.clone(),
        })
        .collect();
    assert_eq!(
        covered,
        vec!["gene_id".to_string()],
        "the rebuilt index covers exactly what survived"
    );
}
