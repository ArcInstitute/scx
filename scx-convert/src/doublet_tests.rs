//! Tests for the doublet-caller profile table and value derivation.
//!
//! The fixtures are written as the tools actually write them: R's `write.csv`
//! quotes its header and emits a leading unnamed index column and bare `NA`;
//! pandas' `to_csv` emits the unnamed index and `True`/`False`. Anything that
//! only tested a hand-built ideal CSV would miss both of the shapes that
//! actually arrive.

use std::fs;
use std::path::PathBuf;

use arrow::array::{Array, BooleanArray, Float32Array, RecordBatch};
use arrow::datatypes::DataType;

use super::*;

fn write(dir: &tempfile::TempDir, name: &str, body: &str) -> PathBuf {
    let p = dir.path().join(name);
    fs::write(&p, body).unwrap();
    p
}

fn opts(tool: &str) -> DoubletImportOptions {
    DoubletImportOptions {
        tool: tool.to_string(),
        ..Default::default()
    }
}

fn col_names(batch: &RecordBatch) -> Vec<String> {
    batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

fn f32_values(batch: &RecordBatch, name: &str) -> Vec<Option<f32>> {
    let a = batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("no column {name} in {:?}", col_names(batch)))
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("canonical score must be f32");
    (0..a.len())
        .map(|i| (!a.is_null(i)).then(|| a.value(i)))
        .collect()
}

fn bool_values(batch: &RecordBatch, name: &str) -> Vec<Option<bool>> {
    let a = batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("no column {name} in {:?}", col_names(batch)))
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("canonical call must be bool");
    (0..a.len())
        .map(|i| (!a.is_null(i)).then(|| a.value(i)))
        .collect()
}

// ---------------------------------------------------------------------------
// One test per tool profile
// ---------------------------------------------------------------------------

#[test]
fn scdblfinder_profile_emits_the_canonical_pair() {
    let dir = tempfile::tempdir().unwrap();
    // Exactly what `write.csv(as.data.frame(colData(sce)[, ...]))` produces:
    // a quoted header, an unnamed index column, and `NA` for the origin of a
    // random-origin doublet.
    let p = write(
        &dir,
        "sce.csv",
        "\"\",\"scDblFinder.score\",\"scDblFinder.class\",\"scDblFinder.mostLikelyOrigin\"\n\
         \"AAAC-1\",0.02,\"singlet\",NA\n\
         \"AAAG-1\",0.98,\"doublet\",\"1+2\"\n",
    );

    let (data, info) = read_doublet_table(&p, &opts("scdblfinder")).unwrap();

    assert_eq!(info.key_added, "scdblfinder");
    assert_eq!(info.score_source_column, "scDblFinder.score");
    assert_eq!(
        info.call_source_column.as_deref(),
        Some("scDblFinder.class")
    );
    assert_eq!(
        info.canonical_columns,
        ["scdblfinder_score", "scdblfinder_predicted"]
    );
    assert_eq!(data.row_keys, ["AAAC-1", "AAAG-1"]);

    let b = &data.row_annotations;
    assert_eq!(f32_values(b, "scdblfinder_score"), [Some(0.02), Some(0.98)]);
    assert_eq!(
        bool_values(b, "scdblfinder_predicted"),
        [Some(false), Some(true)]
    );
    // The one column that is neither score nor call survives, prefixed.
    assert_eq!(
        info.native_columns,
        ["scdblfinder_scDblFinder.mostLikelyOrigin"]
    );
}

#[test]
fn scrublet_profile_reads_a_pandas_boolean() {
    let dir = tempfile::tempdir().unwrap();
    // `adata.obs[["doublet_score","predicted_doublet"]].to_csv(path)`.
    let p = write(
        &dir,
        "scrublet.csv",
        ",doublet_score,predicted_doublet\n\
         AAAC-1,0.03,False\n\
         AAAG-1,0.71,True\n",
    );

    let (data, info) = read_doublet_table(&p, &opts("scrublet")).unwrap();

    assert_eq!(info.key_added, "scrublet");
    assert!(info.table.renamed_index_column);
    let b = &data.row_annotations;
    assert_eq!(f32_values(b, "scrublet_score"), [Some(0.03), Some(0.71)]);
    assert_eq!(
        bool_values(b, "scrublet_predicted"),
        [Some(false), Some(true)]
    );
}

#[test]
fn doubletfinder_profile_matches_parameterised_columns_by_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        &dir,
        "seurat.csv",
        "barcode,pANN_0.25_0.09_87,DF.classifications_0.25_0.09_87\n\
         AAAC-1,0.11,Singlet\n\
         AAAG-1,0.87,Doublet\n",
    );

    let (data, info) = read_doublet_table(&p, &opts("doubletfinder")).unwrap();

    assert_eq!(info.score_source_column, "pANN_0.25_0.09_87");
    assert_eq!(
        info.call_source_column.as_deref(),
        Some("DF.classifications_0.25_0.09_87")
    );
    let b = &data.row_annotations;
    assert_eq!(
        f32_values(b, "doubletfinder_score"),
        [Some(0.11), Some(0.87)]
    );
    assert_eq!(
        bool_values(b, "doubletfinder_predicted"),
        [Some(false), Some(true)]
    );
}

#[test]
fn doubletdetection_profile_reads_a_numeric_label() {
    let dir = tempfile::tempdir().unwrap();
    // The classifier emits NaN for cells it never converged on; that must stay
    // a null, not become a singlet call.
    let p = write(
        &dir,
        "dd.csv",
        "barcode,doublet_score,doublet_label\n\
         AAAC-1,1.2,0\n\
         AAAG-1,8.4,1\n\
         AAAT-1,3.0,NaN\n",
    );

    let (data, _) = read_doublet_table(&p, &opts("doubletdetection")).unwrap();
    let b = &data.row_annotations;
    assert_eq!(
        bool_values(b, "doubletdetection_predicted"),
        [Some(false), Some(true), None]
    );
}

#[test]
fn solo_profile_prefers_softmax_score_then_falls_back() {
    let dir = tempfile::tempdir().unwrap();
    let both = write(
        &dir,
        "solo_both.csv",
        "barcode,softmax_score,score,prediction\n\
         AAAC-1,0.1,9.9,singlet\n",
    );
    let (data, info) = read_doublet_table(&both, &opts("solo")).unwrap();
    assert_eq!(info.score_source_column, "softmax_score");
    // `score` is a declared alternate spelling that lost the race, and its
    // native name would be the canonical one. Dropped, but reported — not a
    // silent discard and not a spurious collision error.
    assert_eq!(info.dropped_alias_columns, ["score"]);
    assert_eq!(
        col_names(&data.row_annotations),
        ["solo_score", "solo_predicted"]
    );

    let only_score = write(
        &dir,
        "solo_score.csv",
        "barcode,score,prediction\nAAAC-1,0.1,doublet\n",
    );
    let (data, info) = read_doublet_table(&only_score, &opts("solo")).unwrap();
    assert_eq!(info.score_source_column, "score");
    assert_eq!(
        bool_values(&data.row_annotations, "solo_predicted"),
        [Some(true)]
    );
}

#[test]
fn scds_profile_omits_the_call_column() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        &dir,
        "scds.csv",
        "\"\",\"cxds_score\",\"bcds_score\",\"hybrid_score\"\n\
         \"AAAC-1\",0.4,0.2,0.31\n",
    );

    let (data, info) = read_doublet_table(&p, &opts("scds")).unwrap();

    assert_eq!(info.score_source_column, "hybrid_score");
    assert_eq!(info.call_source_column, None);
    assert_eq!(info.canonical_columns, ["scds_score"]);
    // Never thresholded into existence: a call is a scientific decision the
    // importer does not own.
    assert!(!col_names(&data.row_annotations).contains(&"scds_predicted".to_string()));
    // The two unused score flavours are still carried through.
    assert_eq!(info.native_columns, ["scds_cxds_score", "scds_bcds_score"]);
}

#[test]
fn a_declared_call_column_absent_is_reported_not_silent() {
    // B3, the dogfood repro: a table using SCRUBLET's column names imported as
    // doubletdetection. The read must SUCCEED (score-only is valid) while
    // reporting exactly why there is no call, so the binding can warn instead of
    // letting the user discover it much later in `doublet_consensus`.
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        &dir,
        "dd.csv",
        "barcode,doublet_score,predicted_doublet\n         AAAC-1,0.12,True\n",
    );

    let (data, info) = read_doublet_table(&p, &opts("doubletdetection")).unwrap();

    assert_eq!(info.call_source_column, None);
    assert_eq!(info.canonical_columns, ["doubletdetection_score"]);
    let m = info
        .call_column_missing
        .as_ref()
        .expect("a declared-but-absent call column must be reported");
    assert_eq!(m.expected_columns, ["doublet_label"]);
    assert_eq!(m.expected_prefix, None);
    assert!(m.expected.contains("doublet_label"), "{}", m.expected);
    // The "but what did I have?" half — this is what makes the message useful.
    assert!(m.present_columns.contains(&"predicted_doublet".to_string()));
    // No data is lost: the unmatched column survives verbatim, so the user can
    // fix forward with `call_column=` and not re-run the caller.
    assert_eq!(info.native_columns, ["doubletdetection_predicted_doublet"]);
    assert!(!col_names(&data.row_annotations).contains(&"doubletdetection_predicted".to_string()));
}

#[test]
fn a_profile_with_no_call_column_reports_nothing_missing() {
    // The assertion whose absence IS B3: scds and doubletdetection both end up
    // with `call_source_column == None`, and only this field tells them apart.
    let dir = tempfile::tempdir().unwrap();
    let p = write(&dir, "scds.csv", "barcode,hybrid_score\nAAAC-1,0.31\n");
    let (_, info) = read_doublet_table(&p, &opts("scds")).unwrap();
    assert_eq!(info.call_source_column, None);
    assert!(
        info.call_column_missing.is_none(),
        "scds declares no call column, so nothing is *missing*"
    );
}

#[test]
fn a_prefix_declared_call_column_absent_reports_the_prefix() {
    // doubletfinder declares its call by PREFIX, not by alias, so the
    // diagnostic has to carry the prefix rather than an empty alias list.
    let dir = tempfile::tempdir().unwrap();
    let p = write(&dir, "df.csv", "barcode,pANN_0.25_0.03_100\nAAAC-1,0.4\n");
    let (_, info) = read_doublet_table(&p, &opts("doubletfinder")).unwrap();
    let m = info.call_column_missing.as_ref().expect("reported");
    assert_eq!(m.expected_prefix.as_deref(), Some("DF.classifications_"));
    assert!(m.expected_columns.is_empty());
    assert!(m.expected.contains("DF.classifications_"), "{}", m.expected);
}

#[test]
fn an_explicit_call_column_that_is_absent_still_errors() {
    // An explicit override must stay a hard error for both roles — the 3-state
    // refactor must not soften it into a warning.
    let dir = tempfile::tempdir().unwrap();
    let p = write(&dir, "x.csv", "barcode,doublet_score\nAAAC-1,0.1\n");
    let mut o = opts("scrublet");
    o.call_column = Some("nope".to_string());
    let e = read_doublet_table(&p, &o).unwrap_err();
    assert!(e.to_string().contains("nope"), "{e}");
}

#[test]
fn a_declared_call_column_absent_still_imports_the_score_and_records_why() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        &dir,
        "dd.csv",
        "barcode,doublet_score,predicted_doublet\nAAAC-1,0.12,True\n",
    );
    let (data, _) = read_doublet_table(&p, &opts("doubletdetection")).unwrap();
    let uns = data.uns.as_ref().expect("uns record");
    assert_eq!(uns["call_column_status"], "declared_but_absent");
    assert_eq!(uns["expected_call_columns"][0], "doublet_label");

    // And the other two statuses, so all three are pinned in one place.
    let p2 = write(&dir, "scds.csv", "barcode,hybrid_score\nAAAC-1,0.3\n");
    let (d2, _) = read_doublet_table(&p2, &opts("scds")).unwrap();
    assert_eq!(
        d2.uns.as_ref().unwrap()["call_column_status"],
        "not_declared"
    );

    let p3 = write(
        &dir,
        "ok.csv",
        "barcode,doublet_score,doublet_label\nAAAC-1,0.12,1\n",
    );
    let (d3, _) = read_doublet_table(&p3, &opts("doubletdetection")).unwrap();
    assert_eq!(d3.uns.as_ref().unwrap()["call_column_status"], "resolved");
}

#[test]
fn scds_gains_a_call_column_when_one_is_named() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        &dir,
        "scds.csv",
        "barcode,hybrid_score,my_call\nAAAC-1,0.9,1\nAAAG-1,0.1,0\n",
    );
    let mut o = opts("scds");
    o.call_column = Some("my_call".to_string());

    let (data, info) = read_doublet_table(&p, &o).unwrap();
    assert_eq!(info.call_source_column.as_deref(), Some("my_call"));
    assert_eq!(
        bool_values(&data.row_annotations, "scds_predicted"),
        [Some(true), Some(false)]
    );
}

#[test]
fn generic_profile_takes_both_columns_from_the_caller() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        &dir,
        "mine.csv",
        "barcode,my_score,my_call\nAAAC-1,0.5,yes\nAAAG-1,0.9,no\n",
    );
    let mut o = opts("generic");
    o.key_added = "dbl".to_string();
    o.score_column = Some("my_score".to_string());
    o.call_column = Some("my_call".to_string());
    o.call_true = Some("yes".to_string());
    o.call_false = Some("no".to_string());

    let (data, info) = read_doublet_table(&p, &o).unwrap();
    assert_eq!(info.canonical_columns, ["dbl_score", "dbl_predicted"]);
    assert_eq!(
        bool_values(&data.row_annotations, "dbl_predicted"),
        [Some(true), Some(false)]
    );
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn missing_expected_column_errors_naming_present_columns() {
    let dir = tempfile::tempdir().unwrap();
    // A scrublet-shaped file handed to the scDblFinder profile.
    let p = write(
        &dir,
        "wrong.csv",
        "barcode,doublet_score,predicted_doublet\nAAAC-1,0.03,False\n",
    );

    let e = read_doublet_table(&p, &opts("scdblfinder")).unwrap_err();
    let m = e.to_string();
    assert!(m.contains("scDblFinder.score"), "{m}");
    // The actionable half: what the file really has.
    assert!(m.contains("doublet_score"), "{m}");
    assert!(m.contains("predicted_doublet"), "{m}");
}

#[test]
fn ambiguous_doubletfinder_pann_columns_error() {
    let dir = tempfile::tempdir().unwrap();
    // DoubletFinder run twice with different pK leaves both behind.
    let p = write(
        &dir,
        "two.csv",
        "barcode,pANN_0.25_0.09_87,pANN_0.25_0.30_54,DF.classifications_0.25_0.09_87\n\
         AAAC-1,0.1,0.2,Singlet\n",
    );

    let e = read_doublet_table(&p, &opts("doubletfinder")).unwrap_err();
    let m = e.to_string();
    assert!(m.contains("pANN_0.25_0.09_87"), "{m}");
    assert!(m.contains("pANN_0.25_0.30_54"), "{m}");

    // ... and naming one resolves it, which is what the message tells you to do.
    let mut o = opts("doubletfinder");
    o.score_column = Some("pANN_0.25_0.30_54".to_string());
    let (data, _) = read_doublet_table(&p, &o).unwrap();
    assert_eq!(
        f32_values(&data.row_annotations, "doubletfinder_score"),
        [Some(0.2)]
    );
}

#[test]
fn ambiguous_doubletfinder_classification_columns_error() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        &dir,
        "two.csv",
        "barcode,pANN_0.25_0.09_87,DF.classifications_0.25_0.09_87,DF.classifications_0.25_0.30_54\n\
         AAAC-1,0.1,Singlet,Doublet\n",
    );

    let e = read_doublet_table(&p, &opts("doubletfinder")).unwrap_err();
    let m = e.to_string();
    assert!(m.contains("DF.classifications_0.25_0.09_87"), "{m}");
    assert!(m.contains("DF.classifications_0.25_0.30_54"), "{m}");
}

#[test]
fn unknown_call_token_errors_naming_the_value() {
    let dir = tempfile::tempdir().unwrap();
    // A hypothetical future scDblFinder class the profile does not declare.
    let p = write(
        &dir,
        "sce.csv",
        "barcode,scDblFinder.score,scDblFinder.class\n\
         AAAC-1,0.02,singlet\n\
         AAAG-1,0.51,ambiguous\n",
    );

    let e = read_doublet_table(&p, &opts("scdblfinder")).unwrap_err();
    let m = e.to_string();
    assert!(m.contains("ambiguous"), "{m}");
    assert!(m.contains("doublet") && m.contains("singlet"), "{m}");

    // The escape hatch the message points at.
    let mut o = opts("scdblfinder");
    o.call_true = Some("ambiguous".to_string());
    o.call_false = Some("singlet".to_string());
    let (data, _) = read_doublet_table(&p, &o).unwrap();
    assert_eq!(
        bool_values(&data.row_annotations, "scdblfinder_predicted"),
        [Some(false), Some(true)]
    );
}

#[test]
fn out_of_range_numeric_call_errors() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        &dir,
        "dd.csv",
        "barcode,doublet_score,doublet_label\nAAAC-1,1.2,2\n",
    );
    let e = read_doublet_table(&p, &opts("doubletdetection")).unwrap_err();
    assert!(e.to_string().contains('2'), "{e}");
}

#[test]
fn text_score_column_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        &dir,
        "bad.csv",
        "barcode,scDblFinder.score,scDblFinder.class\nAAAC-1,high,doublet\n",
    );
    let e = read_doublet_table(&p, &opts("scdblfinder")).unwrap_err();
    let m = e.to_string();
    assert!(m.contains("scDblFinder.score"), "{m}");
    assert!(m.contains("not numeric"), "{m}");
}

#[test]
fn generic_requires_a_score_column() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(&dir, "mine.csv", "barcode,my_score\nAAAC-1,0.5\n");
    let e = read_doublet_table(&p, &opts("generic")).unwrap_err();
    let m = e.to_string();
    assert!(m.contains("score"), "{m}");
    assert!(
        m.contains("my_score"),
        "the message must name what is available: {m}"
    );
}

#[test]
fn generic_text_call_needs_call_true() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        &dir,
        "mine.csv",
        "barcode,my_score,my_call\nAAAC-1,0.5,yes\nAAAG-1,0.1,no\n",
    );
    let mut o = opts("generic");
    o.score_column = Some("my_score".to_string());
    o.call_column = Some("my_call".to_string());

    let e = read_doublet_table(&p, &o).unwrap_err();
    assert!(e.to_string().contains("call_true"), "{e}");

    // With only the positive token, the caller has explicitly asked for a
    // partition — everything else is a singlet, because they said so.
    o.call_true = Some("yes".to_string());
    let (data, _) = read_doublet_table(&p, &o).unwrap();
    assert_eq!(
        bool_values(&data.row_annotations, "generic_predicted"),
        [Some(true), Some(false)]
    );
}

#[test]
fn unknown_tool_errors_listing_the_valid_set() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(&dir, "x.csv", "barcode,doublet_score\nAAAC-1,0.5\n");
    let e = read_doublet_table(&p, &opts("scDblFinderr")).unwrap_err();
    let m = e.to_string();
    assert!(m.contains("scDblFinderr"), "{m}");
    for name in DOUBLET_PROFILE_NAMES {
        assert!(m.contains(name), "{m} is missing {name}");
    }
}

#[cfg(not(feature = "hdf5"))]
#[test]
fn an_h5ad_source_without_the_feature_says_what_to_do_instead() {
    // Phase 5 made h5ad a real source, so the blanket "not implemented yet"
    // refusal is gone. In a build with no HDF5 the refusal remains, and still
    // has to name the route that does work here.
    let dir = tempfile::tempdir().unwrap();
    let p = write(&dir, "scrublet_out.h5ad", "not really hdf5");
    let e = read_doublet_table(&p, &opts("scrublet")).unwrap_err();
    let m = e.to_string();
    assert!(m.contains("to_csv"), "{m}");
    assert!(m.contains("no HDF5 support"), "{m}");
}

#[cfg(feature = "hdf5")]
#[test]
fn an_h5ad_source_is_read_rather_than_refused_on_its_extension() {
    // The extension no longer decides the outcome; the file's contents do.
    // This one is not HDF5 at all, so it fails as an unreadable file rather
    // than as an unsupported format.
    let dir = tempfile::tempdir().unwrap();
    let p = write(&dir, "scrublet_out.h5ad", "not really hdf5");
    let e = read_doublet_table(&p, &opts("scrublet")).unwrap_err();
    let m = e.to_string();
    assert!(m.contains("as HDF5"), "{m}");
    assert!(
        !m.contains("not implemented"),
        "the Phase-4 deferral message must be gone: {m}"
    );
}

#[test]
fn an_h5mu_source_is_refused_with_the_modality_route() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(&dir, "atlas.h5mu", "not really hdf5");
    let e = read_doublet_table(&p, &opts("scrublet")).unwrap_err();
    let m = e.to_string();
    assert!(m.contains("multimodal"), "{m}");
    assert!(m.contains("--modality"), "{m}");
}

// ---------------------------------------------------------------------------
// Column plumbing
// ---------------------------------------------------------------------------

#[test]
fn native_columns_are_prefixed_and_the_sources_consumed() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        &dir,
        "sce.csv",
        "barcode,scDblFinder.score,scDblFinder.class,scDblFinder.weighted,nCount\n\
         AAAC-1,0.02,singlet,0.5,1200\n",
    );

    let (data, info) = read_doublet_table(&p, &opts("scdblfinder")).unwrap();
    let names = col_names(&data.row_annotations);

    assert_eq!(
        names,
        [
            "scdblfinder_score",
            "scdblfinder_predicted",
            "scdblfinder_scDblFinder.weighted",
            "scdblfinder_nCount",
        ]
    );
    // The score and class sources are gone: the same numbers under two names
    // would only invite the two to drift.
    assert!(!names.contains(&"scdblfinder_scDblFinder.score".to_string()));
    assert!(!names.contains(&"scdblfinder_scDblFinder.class".to_string()));
    assert_eq!(info.native_columns.len(), 2);
}

#[test]
fn keep_native_columns_false_emits_only_the_canonical_pair() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        &dir,
        "sce.csv",
        "barcode,scDblFinder.score,scDblFinder.class,scDblFinder.weighted\n\
         AAAC-1,0.02,singlet,0.5\n",
    );
    let mut o = opts("scdblfinder");
    o.keep_native_columns = false;

    let (data, info) = read_doublet_table(&p, &o).unwrap();
    assert_eq!(
        col_names(&data.row_annotations),
        ["scdblfinder_score", "scdblfinder_predicted"]
    );
    assert!(info.native_columns.is_empty());
}

#[test]
fn a_native_column_colliding_with_a_canonical_name_errors() {
    let dir = tempfile::tempdir().unwrap();
    // A source column literally named `score`, under key_added `dbl`, would
    // become `dbl_score` — the canonical name.
    let p = write(
        &dir,
        "mine.csv",
        "barcode,my_score,score\nAAAC-1,0.5,junk\n",
    );
    let mut o = opts("generic");
    o.key_added = "dbl".to_string();
    o.score_column = Some("my_score".to_string());

    let e = read_doublet_table(&p, &o).unwrap_err();
    let m = e.to_string();
    assert!(m.contains("dbl_score"), "{m}");
    assert!(m.contains("keep_native_columns"), "{m}");
}

#[test]
fn key_added_renames_every_canonical_column() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        &dir,
        "sce.csv",
        "barcode,scDblFinder.score,scDblFinder.class\nAAAC-1,0.02,doublet\n",
    );
    let mut o = opts("scdblfinder");
    o.key_added = "run2".to_string();

    let (data, info) = read_doublet_table(&p, &o).unwrap();
    assert_eq!(info.key_added, "run2");
    assert_eq!(
        col_names(&data.row_annotations),
        ["run2_score", "run2_predicted"]
    );
}

#[test]
fn f64_score_narrows_to_f32() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        &dir,
        "sce.csv",
        "barcode,scDblFinder.score,scDblFinder.class\nAAAC-1,0.123456789,doublet\n",
    );
    let (data, _) = read_doublet_table(&p, &opts("scdblfinder")).unwrap();
    let f = data
        .row_annotations
        .schema()
        .field_with_name("scdblfinder_score")
        .unwrap()
        .clone();
    assert_eq!(f.data_type(), &DataType::Float32);
    assert!(
        f.is_nullable(),
        "unmatched target rows must be able to be null"
    );
}

#[test]
fn an_integer_score_column_still_narrows_to_f32() {
    // arrow infers Int64 for a whole-number score column; the canonical
    // contract is f32 regardless of what the file happened to look like.
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        &dir,
        "sce.csv",
        "barcode,scDblFinder.score,scDblFinder.class\nAAAC-1,1,doublet\nAAAG-1,0,singlet\n",
    );
    let (data, _) = read_doublet_table(&p, &opts("scdblfinder")).unwrap();
    assert_eq!(
        f32_values(&data.row_annotations, "scdblfinder_score"),
        [Some(1.0), Some(0.0)]
    );
}

#[test]
fn a_null_call_stays_null() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        &dir,
        "sce.csv",
        "barcode,scDblFinder.score,scDblFinder.class\n\
         AAAC-1,0.02,singlet\n\
         AAAG-1,0.51,NA\n",
    );
    let (data, _) = read_doublet_table(&p, &opts("scdblfinder")).unwrap();
    assert_eq!(
        bool_values(&data.row_annotations, "scdblfinder_predicted"),
        [Some(false), None],
        "a cell the tool did not call is not a singlet"
    );
}

#[test]
fn a_composite_key_fuses_both_sides() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        &dir,
        "sce.csv",
        "sample_id,barcode,scDblFinder.score,scDblFinder.class\n\
         A,AAAC-1,0.02,singlet\n\
         B,AAAC-1,0.98,doublet\n",
    );
    let mut o = opts("scdblfinder");
    o.key_columns = vec!["sample_id".to_string(), "barcode".to_string()];

    let (data, info) = read_doublet_table(&p, &o).unwrap();
    assert_eq!(info.table.key_columns, ["sample_id", "barcode"]);
    // Same barcode in two samples, disambiguated.
    assert_ne!(data.row_keys[0], data.row_keys[1]);
    assert!(data.row_keys[0].contains("A") && data.row_keys[0].contains("AAAC-1"));
}

#[test]
fn uns_records_the_tool_and_the_resolved_columns() {
    let dir = tempfile::tempdir().unwrap();
    let p = write(
        &dir,
        "sce.csv",
        "barcode,scDblFinder.score,scDblFinder.class\nAAAC-1,0.02,doublet\n",
    );
    let (data, _) = read_doublet_table(&p, &opts("scdblfinder")).unwrap();
    let uns = data
        .uns
        .expect("the wrapper always records provenance in uns");
    assert_eq!(uns["tool"], "scdblfinder");
    assert_eq!(uns["source_score_column"], "scDblFinder.score");
    assert_eq!(uns["source_call_column"], "scDblFinder.class");
    assert_eq!(uns["n_rows_in_source"], 1);
}

#[test]
fn profile_has_call_column_reflects_the_table() {
    assert!(profile_has_call_column(
        doublet_profile("scdblfinder").unwrap()
    ));
    assert!(profile_has_call_column(
        doublet_profile("doubletfinder").unwrap()
    ));
    assert!(!profile_has_call_column(doublet_profile("scds").unwrap()));
    assert!(!profile_has_call_column(
        doublet_profile("generic").unwrap()
    ));
}

// ---------------------------------------------------------------------------
// Categorical (dictionary-encoded) columns
//
// An h5ad's obs stores a pandas Categorical as `Dictionary(Int32, Utf8)`, and a
// class / prediction column is exactly the kind pandas keeps that way. These
// pin the derivation at the `coerce_*` level so the guard holds independently
// of whether the hdf5 feature — or any h5ad file — is in play.
// ---------------------------------------------------------------------------

fn dict_utf8(values: &[Option<&str>]) -> ArrayRef {
    let arr: arrow::array::DictionaryArray<arrow::datatypes::Int32Type> =
        values.iter().copied().collect();
    Arc::new(arr) as ArrayRef
}

#[test]
fn a_categorical_call_column_derives_like_a_plain_one() {
    let cat = dict_utf8(&[Some("singlet"), Some("doublet"), None]);
    assert!(
        matches!(cat.data_type(), DataType::Dictionary(_, _)),
        "fixture must actually be dictionary-encoded: {:?}",
        cat.data_type()
    );

    let tokens = Some(CallTokens {
        doublet: "doublet",
        singlet: "singlet",
    });
    let got = coerce_call(&cat, "scDblFinder.class", tokens, None, None).unwrap();
    let got = got.as_any().downcast_ref::<BooleanArray>().unwrap();

    assert_eq!(
        (0..got.len())
            .map(|i| (!got.is_null(i)).then(|| got.value(i)))
            .collect::<Vec<_>>(),
        [Some(false), Some(true), None]
    );
}

#[test]
fn an_unknown_token_in_a_categorical_call_still_errors() {
    // The dictionary arm decodes; it must not weaken the strictness that the
    // plain-text arm enforces.
    let cat = dict_utf8(&[Some("singlet"), Some("ambiguous")]);
    let tokens = Some(CallTokens {
        doublet: "doublet",
        singlet: "singlet",
    });
    let e = coerce_call(&cat, "scDblFinder.class", tokens, None, None).unwrap_err();
    assert!(e.to_string().contains("ambiguous"), "{e}");
}

#[test]
fn a_categorical_score_column_decodes_to_f32() {
    // Unusual but legal: a numeric column stored as a Categorical.
    let arr: arrow::array::DictionaryArray<arrow::datatypes::Int32Type> =
        vec![Some("0.25"), Some("0.75")].into_iter().collect();
    let dict: ArrayRef = Arc::new(arr);
    // Utf8 values are still not a numeric score -- the decode must not turn the
    // non-numeric rejection into a silent all-null column.
    let e = coerce_score(&dict, "score").unwrap_err();
    assert!(e.to_string().contains("not numeric"), "{e}");

    let keys = arrow::array::Int32Array::from(vec![0, 1, 0]);
    let values = arrow::array::Float64Array::from(vec![0.25_f64, 0.75]);
    let numeric: ArrayRef = Arc::new(
        arrow::array::DictionaryArray::<arrow::datatypes::Int32Type>::try_new(
            keys,
            Arc::new(values),
        )
        .unwrap(),
    );
    let got = coerce_score(&numeric, "score").unwrap();
    assert_eq!(got.data_type(), &DataType::Float32);
    let got = got.as_any().downcast_ref::<Float32Array>().unwrap();
    assert_eq!(
        (0..got.len()).map(|i| got.value(i)).collect::<Vec<_>>(),
        [0.25, 0.75, 0.25]
    );
}

#[test]
fn every_advertised_profile_name_resolves() {
    // The clap value list and the lookup table are separate declarations; this
    // is what stops them drifting.
    for name in DOUBLET_PROFILE_NAMES {
        assert_eq!(doublet_profile(name).unwrap().name, *name);
    }
}
