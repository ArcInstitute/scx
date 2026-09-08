//! Tests for the h5ad `/obs` reader.
//!
//! The fixtures are built the way anndata writes them, because that is the
//! whole point of this reader: a leading `_index` sentinel that has to become
//! `__index_level_0__`, a pandas Categorical stored as codes + categories, and
//! a boolean column that arrives as a real `BooleanArray`. A hand-built ideal
//! dataframe would prove nothing the CSV reader's tests do not already cover.
//!
//! The equivalence that matters — h5ad and CSV producing the same import from
//! the same values — is asserted end to end in
//! `pyscx/tests/test_doublet_import.py`, against a file `anndata` actually
//! wrote rather than one assembled here.

use std::path::{Path, PathBuf};

use arrow::array::{Array, BooleanArray, Float64Array};
use arrow::datatypes::DataType;
use hdf5::types::VarLenUnicode;

use super::*;
use crate::DoubletImportOptions;

fn vlu(s: &str) -> VarLenUnicode {
    s.parse().unwrap()
}

fn str_ds(group: &hdf5::Group, name: &str, values: &[&str]) {
    let vals: Vec<VarLenUnicode> = values.iter().map(|s| vlu(s)).collect();
    group
        .new_dataset::<VarLenUnicode>()
        .shape([vals.len()])
        .create(name)
        .unwrap()
        .write(&vals)
        .unwrap();
}

fn attr(group: &hdf5::Group, name: &str, value: &str) {
    group
        .new_attr::<VarLenUnicode>()
        .create(name)
        .unwrap()
        .write_scalar(&vlu(value))
        .unwrap();
}

/// A scrublet-shaped `/obs`: an unnamed pandas index, a float score, a real
/// boolean call, and a categorical class column.
///
/// `index_name = "_index"` is anndata's sentinel for an unnamed index, which
/// the reader must surface as `__index_level_0__` — the name the ops crate's
/// key-fallback list knows.
fn write_obs_fixture(dir: &Path, name: &str, index_name: &str, barcodes: &[&str]) -> PathBuf {
    let path = dir.join(name);
    let file = hdf5::File::create(&path).unwrap();
    let obs = file.create_group("obs").unwrap();

    attr(&obs, "encoding-type", "dataframe");
    attr(&obs, "encoding-version", "0.2.0");
    attr(&obs, "_index", index_name);

    let cols = ["doublet_score", "predicted_doublet", "scDblFinder.class"];
    let order: Vec<VarLenUnicode> = cols.iter().map(|s| vlu(s)).collect();
    obs.new_attr::<VarLenUnicode>()
        .shape([order.len()])
        .create("column-order")
        .unwrap()
        .write(&order)
        .unwrap();

    str_ds(&obs, index_name, barcodes);

    let scores: Vec<f64> = (0..barcodes.len())
        .map(|i| 0.1 * (i as f64 + 1.0))
        .collect();
    obs.new_dataset::<f64>()
        .shape([scores.len()])
        .create("doublet_score")
        .unwrap()
        .write(&scores)
        .unwrap();

    // numpy bool → h5py int8, which the reader turns back into a BooleanArray.
    let calls: Vec<i8> = (0..barcodes.len()).map(|i| (i % 2) as i8).collect();
    let ds = obs
        .new_dataset::<i8>()
        .shape([calls.len()])
        .create("predicted_doublet")
        .unwrap();
    ds.write(&calls).unwrap();
    ds.new_attr::<VarLenUnicode>()
        .create("encoding-type")
        .unwrap()
        .write_scalar(&vlu("bool"))
        .unwrap();

    // A pandas Categorical: attribute-form codes + categories, exactly the
    // layout every 10x-derived h5ad on disk uses. This is the shape that made
    // `coerce_call` need a Dictionary arm.
    let codes: Vec<i8> = (0..barcodes.len()).map(|i| (i % 2) as i8).collect();
    let cds = obs
        .new_dataset::<i8>()
        .shape([codes.len()])
        .create("scDblFinder.class")
        .unwrap();
    cds.write(&codes).unwrap();
    cds.new_attr::<VarLenUnicode>()
        .create("encoding-type")
        .unwrap()
        .write_scalar(&vlu("categorical"))
        .unwrap();
    let cats = vec![vlu("singlet"), vlu("doublet")];
    cds.new_attr::<VarLenUnicode>()
        .shape([2])
        .create("categories")
        .unwrap()
        .write(&cats)
        .unwrap();

    file.close().unwrap();
    path
}

fn opts() -> AnnotationTableOptions {
    AnnotationTableOptions::default()
}

fn names(data: &ExternalObsData) -> Vec<String> {
    data.row_annotations
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

// ---------------------------------------------------------------------------
// Reading /obs
// ---------------------------------------------------------------------------

#[test]
fn reads_a_scrublet_shaped_obs_and_resolves_the_unnamed_index() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_obs_fixture(
        dir.path(),
        "s.h5ad",
        "_index",
        &["AAAC-1", "AAAG-1", "AAAT-1"],
    );

    let (data, info) = read_h5ad_obs(&p, &opts(), &[]).unwrap();

    assert_eq!(info.format, ObsSourceFormat::H5ad);
    assert_eq!(info.n_rows, 3);
    // The sentinel became the canonical name, which is why auto-resolution
    // finds it without any h5ad-specific fallback list.
    assert_eq!(info.key_columns, ["__index_level_0__"]);
    assert_eq!(data.row_keys, ["AAAC-1", "AAAG-1", "AAAT-1"]);

    // No delimiter, and nothing was renamed — both are CSV-only concepts, and
    // reporting otherwise would misdescribe how the file was read.
    assert_eq!(info.delimiter, None);
    assert!(!info.renamed_index_column);

    // The key is consumed; every other column comes across.
    assert_eq!(
        names(&data),
        ["doublet_score", "predicted_doublet", "scDblFinder.class"]
    );
    assert!(data.source_checksum.is_some());
    assert_eq!(data.source_name.as_deref(), Some("s.h5ad"));
}

#[test]
fn a_named_pandas_index_resolves_under_its_own_name() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_obs_fixture(dir.path(), "named.h5ad", "barcode", &["AAAC-1", "AAAG-1"]);

    let (data, info) = read_h5ad_obs(&p, &opts(), &[]).unwrap();
    assert_eq!(info.key_columns, ["barcode"]);
    assert_eq!(data.row_keys, ["AAAC-1", "AAAG-1"]);
    assert!(!names(&data).contains(&"barcode".to_string()));
}

#[test]
fn an_int_backed_call_column_derives_correctly() {
    // h5py stores a numpy bool as int8, and this reader hands it back as an
    // integer rather than a `BooleanArray`. That is fine — `coerce_call`
    // dispatches on the runtime type and its numeric arm maps 0/1 — but it is
    // worth pinning, because a profile that assumed `Boolean` would break on
    // every h5ad a scanpy tool writes.
    let dir = tempfile::tempdir().unwrap();
    let p = write_obs_fixture(dir.path(), "s.h5ad", "_index", &["A", "B", "C", "D"]);

    let (data, _) = read_h5ad_obs(&p, &opts(), &[]).unwrap();
    let col = data
        .row_annotations
        .column_by_name("predicted_doublet")
        .unwrap();
    assert!(
        col.data_type().is_integer(),
        "expected an integer-backed bool, got {:?}",
        col.data_type()
    );

    let o = DoubletImportOptions {
        tool: "scrublet".to_string(),
        ..Default::default()
    };
    let (derived, _) = crate::doublet::read_doublet_table(&p, &o).unwrap();
    let b = derived
        .row_annotations
        .column_by_name("scrublet_predicted")
        .unwrap()
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("the canonical call is always boolean");
    assert_eq!(
        (0..b.len()).map(|i| b.value(i)).collect::<Vec<_>>(),
        [false, true, false, true]
    );
}

#[test]
fn a_categorical_column_arrives_dictionary_encoded() {
    // Pinning the premise of the `coerce_call` Dictionary arm: if the reader
    // ever stopped producing dictionaries, that arm would be dead code and
    // this test says so.
    let dir = tempfile::tempdir().unwrap();
    let p = write_obs_fixture(dir.path(), "s.h5ad", "_index", &["A", "B"]);

    let (data, _) = read_h5ad_obs(&p, &opts(), &[]).unwrap();
    let col = data
        .row_annotations
        .column_by_name("scDblFinder.class")
        .unwrap();
    assert!(
        matches!(col.data_type(), DataType::Dictionary(_, _)),
        "expected a dictionary, got {:?}",
        col.data_type()
    );
}

#[test]
fn column_selection_rename_and_prefix_behave_as_on_the_csv_path() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_obs_fixture(dir.path(), "s.h5ad", "_index", &["A", "B"]);

    let mut o = opts();
    o.columns = Some(vec!["doublet_score".to_string()]);
    o.rename = [("doublet_score".to_string(), "score".to_string())]
        .into_iter()
        .collect();
    o.prefix = "scr_".to_string();

    let (data, info) = read_h5ad_obs(&p, &o, &[]).unwrap();
    assert_eq!(names(&data), ["scr_score"]);
    assert_eq!(info.columns_imported, ["scr_score"]);
}

#[test]
fn keep_key_columns_imports_the_index_too() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_obs_fixture(dir.path(), "s.h5ad", "_index", &["A", "B"]);

    let mut o = opts();
    o.keep_key_columns = true;
    let (data, _) = read_h5ad_obs(&p, &o, &[]).unwrap();
    assert!(names(&data).contains(&"__index_level_0__".to_string()));
}

/// The index is data only when it is the key.
///
/// Keying on a named column left the frame's own index field in the import set,
/// so the attach tried to write a column literally called
/// `__index_level_0__` — which collides with the target's own index field and
/// fails with a message about a column the caller never mentioned. Reachable
/// on obs (an h5ad keyed on a sample column) and unavoidable on var, where the
/// index is symbols and the key is usually `gene_id`.
#[test]
fn a_non_key_index_field_is_not_imported_as_a_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("named_key.h5ad");
    let file = hdf5::File::create(&path).unwrap();
    let obs = file.create_group("obs").unwrap();
    attr(&obs, "encoding-type", "dataframe");
    attr(&obs, "encoding-version", "0.2.0");
    attr(&obs, "_index", "_index");
    let order: Vec<VarLenUnicode> = ["alt_id", "score"].iter().map(|s| vlu(s)).collect();
    obs.new_attr::<VarLenUnicode>()
        .shape([order.len()])
        .create("column-order")
        .unwrap()
        .write(&order)
        .unwrap();
    str_ds(&obs, "_index", &["A", "B"]);
    str_ds(&obs, "alt_id", &["x", "y"]);
    obs.new_dataset::<f64>()
        .shape([2])
        .create("score")
        .unwrap()
        .write(&[1.0f64, 2.0])
        .unwrap();
    file.close().unwrap();

    let o = AnnotationTableOptions {
        key_columns: vec!["alt_id".to_string()],
        ..opts()
    };
    let (data, info) = read_h5ad_obs(&path, &o, &[]).unwrap();
    assert_eq!(info.key_columns, vec!["alt_id".to_string()]);
    assert_eq!(
        names(&data),
        vec!["score".to_string()],
        "the index must not be imported as `__index_level_0__` alongside the data"
    );

    // An explicit request still wins: naming it is how you import an index as
    // an ordinary column.
    let o = AnnotationTableOptions {
        key_columns: vec!["alt_id".to_string()],
        columns: Some(vec!["__index_level_0__".to_string(), "score".to_string()]),
        ..opts()
    };
    let (data, _) = read_h5ad_obs(&path, &o, &[]).unwrap();
    assert!(names(&data).contains(&"__index_level_0__".to_string()));
}

#[test]
fn a_composite_key_fuses_from_obs_columns() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi.h5ad");
    let file = hdf5::File::create(&path).unwrap();
    let obs = file.create_group("obs").unwrap();
    attr(&obs, "encoding-type", "dataframe");
    attr(&obs, "encoding-version", "0.2.0");
    attr(&obs, "_index", "_index");
    let order: Vec<VarLenUnicode> = ["sample_id", "barcode", "doublet_score"]
        .iter()
        .map(|s| vlu(s))
        .collect();
    obs.new_attr::<VarLenUnicode>()
        .shape([order.len()])
        .create("column-order")
        .unwrap()
        .write(&order)
        .unwrap();
    str_ds(&obs, "_index", &["r0", "r1"]);
    str_ds(&obs, "sample_id", &["A", "B"]);
    // The same barcode in two samples: only the pair is unique.
    str_ds(&obs, "barcode", &["AAAC-1", "AAAC-1"]);
    obs.new_dataset::<f64>()
        .shape([2])
        .create("doublet_score")
        .unwrap()
        .write(&[0.1_f64, 0.9])
        .unwrap();
    file.close().unwrap();

    let mut o = opts();
    o.key_columns = vec!["sample_id".to_string(), "barcode".to_string()];
    let (data, info) = read_h5ad_obs(&path, &o, &[]).unwrap();

    assert_eq!(info.key_columns, ["sample_id", "barcode"]);
    assert_ne!(data.row_keys[0], data.row_keys[1]);
    assert!(data.row_keys[0].contains('A') && data.row_keys[0].contains("AAAC-1"));
}

#[test]
fn a_missing_key_column_errors_naming_present_columns() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_obs_fixture(dir.path(), "s.h5ad", "_index", &["A", "B"]);

    let mut o = opts();
    o.key_columns = vec!["nope".to_string()];
    let e = read_h5ad_obs(&p, &o, &[]).unwrap_err();
    let m = e.to_string();
    assert!(m.contains("nope"), "{m}");
    assert!(m.contains("doublet_score"), "{m}");
}

#[test]
fn a_file_that_is_not_hdf5_errors_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("fake.h5ad");
    std::fs::write(&p, "barcode,score\nA,0.5\n").unwrap();
    let e = read_h5ad_obs(&p, &opts(), &[]).unwrap_err();
    assert!(e.to_string().contains("as HDF5"), "{e}");
}

// ---------------------------------------------------------------------------
// uns
// ---------------------------------------------------------------------------

fn write_uns(path: &Path, entries: &[(&str, f64)]) {
    let file = hdf5::File::open_rw(path).unwrap();
    let uns = file.create_group("uns").unwrap();
    for (k, v) in entries {
        uns.new_dataset::<f64>()
            .shape([1])
            .create(*k)
            .unwrap()
            .write(&[*v])
            .unwrap();
    }
    file.close().unwrap();
}

#[test]
fn no_uns_keys_requested_imports_none() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_obs_fixture(dir.path(), "s.h5ad", "_index", &["A", "B"]);
    write_uns(&p, &[("scrublet", 0.4)]);

    let (data, info) = read_h5ad_obs(&p, &opts(), &[]).unwrap();
    // Opt-in: /uns routinely holds large arrays and types the reader skips, so
    // pulling it wholesale would be a surprise in both size and content.
    assert!(data.uns.is_empty());
    assert!(info.uns_keys_imported.is_empty());
}

#[test]
fn a_requested_uns_key_is_imported() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_obs_fixture(dir.path(), "s.h5ad", "_index", &["A", "B"]);
    write_uns(&p, &[("scrublet", 0.4), ("other", 1.0)]);

    let (data, info) = read_h5ad_obs(&p, &opts(), &["scrublet".to_string()]).unwrap();
    // The selected keys ARE the payload, at top level, keyed by their source
    // names; a caller that wants them under one key nests them itself.
    let uns = Value::Object(data.uns);
    assert!(uns.get("scrublet").is_some(), "{uns}");
    assert!(uns.get("other").is_none(), "only what was asked for: {uns}");
    assert_eq!(info.uns_keys_imported, ["scrublet"]);
}

#[test]
fn a_missing_uns_key_errors_naming_the_keys_present() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_obs_fixture(dir.path(), "s.h5ad", "_index", &["A", "B"]);
    write_uns(&p, &[("scrublet", 0.4)]);

    let e = read_h5ad_obs(&p, &opts(), &["scdblfinder".to_string()]).unwrap_err();
    let m = e.to_string();
    // Silently importing nothing would look like the tool wrote nothing.
    assert!(m.contains("scdblfinder"), "{m}");
    assert!(m.contains("scrublet"), "{m}");
}

// ---------------------------------------------------------------------------
// Through the doublet wrapper
// ---------------------------------------------------------------------------

fn f32_at(data: &ExternalObsData, name: &str, i: usize) -> f32 {
    let col = data.row_annotations.column_by_name(name).unwrap();
    let a = arrow::compute::cast(col, &DataType::Float64).unwrap();
    a.as_any().downcast_ref::<Float64Array>().unwrap().value(i) as f32
}

#[test]
fn doublet_import_reads_an_h5ad_without_a_csv_detour() {
    // The Phase-5 exit criterion in miniature: the shape `sc.pp.scrublet`
    // leaves behind, imported directly.
    let dir = tempfile::tempdir().unwrap();
    let p = write_obs_fixture(dir.path(), "scrublet_out.h5ad", "_index", &["A", "B", "C"]);

    let o = DoubletImportOptions {
        tool: "scrublet".to_string(),
        ..Default::default()
    };
    let (data, info) = crate::doublet::read_doublet_table(&p, &o).unwrap();

    assert_eq!(info.table.format, ObsSourceFormat::H5ad);
    assert_eq!(info.score_source_column, "doublet_score");
    assert_eq!(
        info.call_source_column.as_deref(),
        Some("predicted_doublet")
    );
    assert_eq!(
        info.canonical_columns,
        ["scrublet_score", "scrublet_predicted"]
    );
    assert!((f32_at(&data, "scrublet_score", 1) - 0.2).abs() < 1e-6);

    let uns = &data.uns["scrublet"];
    assert_eq!(uns["source_format"], "h5ad");
}

#[test]
fn the_categorical_class_column_derives_through_the_wrapper() {
    // The h5ad-only hazard, end to end: `scDblFinder.class` is a Categorical
    // here, so without the Dictionary arm this import fails on a file whose
    // CSV equivalent works.
    let dir = tempfile::tempdir().unwrap();
    let p = write_obs_fixture(dir.path(), "sce.h5ad", "_index", &["A", "B", "C", "D"]);

    let o = DoubletImportOptions {
        tool: "scdblfinder".to_string(),
        // The fixture's score column is scrublet-named; point at it explicitly
        // so this test is about the *call* derivation and nothing else.
        score_column: Some("doublet_score".to_string()),
        ..Default::default()
    };
    let (data, info) = crate::doublet::read_doublet_table(&p, &o).unwrap();
    assert_eq!(
        info.call_source_column.as_deref(),
        Some("scDblFinder.class")
    );

    let col = data
        .row_annotations
        .column_by_name("scdblfinder_predicted")
        .unwrap();
    let b = col.as_any().downcast_ref::<BooleanArray>().unwrap();
    assert_eq!(
        (0..b.len()).map(|i| b.value(i)).collect::<Vec<_>>(),
        [false, true, false, true],
        "codes 0/1 map to categories singlet/doublet"
    );
}

#[test]
fn a_requested_uns_key_nests_under_the_wrapper_record() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_obs_fixture(dir.path(), "scrublet_out.h5ad", "_index", &["A", "B"]);
    write_uns(&p, &[("scrublet", 0.4)]);

    let o = DoubletImportOptions {
        tool: "scrublet".to_string(),
        uns_keys: vec!["scrublet".to_string()],
        ..Default::default()
    };
    let (data, _) = crate::doublet::read_doublet_table(&p, &o).unwrap();
    // The wrapper keys its record by `key_added` (here the profile default).
    assert_eq!(
        data.uns.len(),
        1,
        "{:?}",
        data.uns.keys().collect::<Vec<_>>()
    );
    let uns = &data.uns["scrublet"];

    // Nested, so the tool's own key keeps its source name and cannot collide
    // with the wrapper's fields however `key_added` was spelled.
    assert!(uns["source_uns"]["scrublet"].is_array() || uns["source_uns"]["scrublet"].is_number());
    assert_eq!(uns["tool"], "scrublet");
    assert_eq!(uns["key_added"], "scrublet");
}

// ---------------------------------------------------------------------------
// The var axis
//
// `read_h5ad_axis("var", ..)` reads `/var` through the same code. Worth its own
// fixture rather than trusting the obs arm: the group name is a parameter, and
// nothing else in the tree would notice if it stopped being used.
// ---------------------------------------------------------------------------

/// A `/var` group in anndata's own layout: an unnamed index of gene ids, a
/// symbol column and a categorical `feature_type`.
fn write_var_fixture(dir: &Path, name: &str, gene_ids: &[&str]) -> PathBuf {
    let path = dir.join(name);
    let file = hdf5::File::create(&path).unwrap();
    let var = file.create_group("var").unwrap();

    attr(&var, "encoding-type", "dataframe");
    attr(&var, "encoding-version", "0.2.0");
    attr(&var, "_index", "_index");

    let cols = ["gene_symbol", "feature_type"];
    let order: Vec<VarLenUnicode> = cols.iter().map(|s| vlu(s)).collect();
    var.new_attr::<VarLenUnicode>()
        .shape([order.len()])
        .create("column-order")
        .unwrap()
        .write(&order)
        .unwrap();

    str_ds(&var, "_index", gene_ids);
    let symbols: Vec<String> = gene_ids.iter().map(|g| format!("sym_{g}")).collect();
    str_ds(
        &var,
        "gene_symbol",
        &symbols.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    );

    let codes: Vec<i8> = (0..gene_ids.len()).map(|i| (i % 2) as i8).collect();
    let cds = var
        .new_dataset::<i8>()
        .shape([codes.len()])
        .create("feature_type")
        .unwrap();
    cds.write(&codes).unwrap();
    cds.new_attr::<VarLenUnicode>()
        .create("encoding-type")
        .unwrap()
        .write_scalar(&vlu("categorical"))
        .unwrap();
    let cats = vec![vlu("Gene Expression"), vlu("Peaks")];
    cds.new_attr::<VarLenUnicode>()
        .shape([2])
        .create("categories")
        .unwrap()
        .write(&cats)
        .unwrap();

    file.close().unwrap();
    path
}

#[test]
fn reads_var_from_an_h5ad_with_the_index_as_the_key() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_var_fixture(dir.path(), "v.h5ad", &["ENSG1", "ENSG2", "ENSG3"]);

    let (src, info) = crate::h5ad_obs::read_h5ad_axis("var", &path, &opts(), &[]).unwrap();
    // The `_index` sentinel must surface as the canonical name, which is what
    // makes automatic key resolution work without special casing.
    assert_eq!(info.key_columns, vec!["__index_level_0__".to_string()]);
    assert_eq!(
        src.row_keys,
        vec![
            "ENSG1".to_string(),
            "ENSG2".to_string(),
            "ENSG3".to_string()
        ]
    );
    assert_eq!(info.n_rows, 3);
    assert_eq!(info.format, ObsSourceFormat::H5ad);
    assert!(info.delimiter.is_none());

    let cols: Vec<String> = src
        .row_annotations
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(
        cols,
        vec!["gene_symbol".to_string(), "feature_type".to_string()]
    );
    // A pandas Categorical must arrive as a dictionary, so it lands on the file
    // as a dictionary too.
    assert!(matches!(
        src.row_annotations
            .schema()
            .field_with_name("feature_type")
            .unwrap()
            .data_type(),
        DataType::Dictionary(_, _)
    ));
}

#[test]
fn an_h5ad_with_no_var_group_errors_naming_var() {
    let dir = tempfile::tempdir().unwrap();
    // An obs-only file: the reader must not silently fall back to /obs.
    let path = write_obs_fixture(dir.path(), "obs_only.h5ad", "_index", &["c1", "c2"]);
    let err = crate::h5ad_obs::read_h5ad_axis("var", &path, &opts(), &[]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("/var"), "{msg}");
}

#[test]
fn a_named_var_key_resolves_against_var_and_reports_the_axis_on_failure() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_var_fixture(dir.path(), "v2.h5ad", &["ENSG1", "ENSG2"]);

    let o = AnnotationTableOptions {
        key_columns: vec!["gene_symbol".to_string()],
        ..opts()
    };
    let (src, info) = crate::h5ad_obs::read_h5ad_axis("var", &path, &o, &[]).unwrap();
    assert_eq!(info.key_columns, vec!["gene_symbol".to_string()]);
    assert_eq!(src.row_keys, vec!["sym_ENSG1", "sym_ENSG2"]);

    let bad = AnnotationTableOptions {
        key_columns: vec!["nope".to_string()],
        ..opts()
    };
    let err = crate::h5ad_obs::read_h5ad_axis("var", &path, &bad, &[]).unwrap_err();
    assert!(
        matches!(&err, scx_ops::OpsError::KeyColumnUnresolved { axis, .. } if *axis == "var"),
        "got {err}"
    );
    assert!(err.to_string().contains("/var of"), "{err}");
}
