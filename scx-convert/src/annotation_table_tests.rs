//! Tests for [`read_annotation_table`].
//!
//! Priorities, roughly by how badly a regression would hurt:
//!
//! 1. **Real-world file shapes parse.** R's `write.csv` writes `NA`; pandas'
//!    `to_csv` writes an unnamed index column. Both are on the flagship path
//!    (scDblFinder and Scrublet respectively), and both silently mangle the
//!    result if unhandled — a string score column, or a key that resolves to
//!    nothing.
//! 2. **Key tokens survive verbatim.** A barcode of `0012` narrowed to `Int64`
//!    and cast back is `12`, which matches no target row. The join then fails
//!    with zero overlap and no hint that the reader was at fault.
//! 3. **Ambiguity is refused, not guessed.** Duplicate or blank column names
//!    make `--columns` meaningless.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use arrow::array::{Array, Float64Array, StringArray};
use arrow::datatypes::DataType;

use super::*;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn write_file(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(body.as_bytes()).unwrap();
    p
}

fn opts() -> AnnotationTableOptions {
    AnnotationTableOptions::default()
}

fn col_names(data: &ExternalObsData) -> Vec<String> {
    data.row_annotations
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

fn dtype_of(data: &ExternalObsData, name: &str) -> DataType {
    data.row_annotations
        .schema()
        .field_with_name(name)
        .unwrap()
        .data_type()
        .clone()
}

// ---------------------------------------------------------------------------
// Happy path
// ---------------------------------------------------------------------------

#[test]
fn reads_a_minimal_barcode_score_call_table() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(
        dir.path(),
        "calls.csv",
        "barcode,score,call\nAAAC-1,0.10,singlet\nAAAG-1,0.90,doublet\n",
    );

    let (data, info) = read_annotation_table(&p, &opts()).unwrap();
    assert_eq!(info.n_rows, 2);
    assert_eq!(info.delimiter, Some(b','));
    assert_eq!(info.key_columns, vec!["barcode".to_string()]);
    assert_eq!(data.row_keys, vec!["AAAC-1", "AAAG-1"]);
    // The key column is consumed, not duplicated into the annotations.
    assert_eq!(col_names(&data), vec!["score", "call"]);
    assert_eq!(dtype_of(&data, "score"), DataType::Float64);
    assert_eq!(data.source_name.as_deref(), Some("calls.csv"));
    assert!(data.source_checksum.is_some());
    assert!(!info.renamed_index_column);
}

#[test]
fn tsv_and_csv_both_parse() {
    let dir = tempfile::tempdir().unwrap();
    let csv = write_file(dir.path(), "a.csv", "barcode,score\nAAAC-1,0.5\n");
    let tsv = write_file(dir.path(), "a.tsv", "barcode\tscore\nAAAC-1\t0.5\n");

    let (c, ci) = read_annotation_table(&csv, &opts()).unwrap();
    let (t, ti) = read_annotation_table(&tsv, &opts()).unwrap();
    assert_eq!(ci.delimiter, Some(b','));
    assert_eq!(ti.delimiter, Some(b'\t'));
    assert_eq!(c.row_keys, t.row_keys);
    assert_eq!(col_names(&c), col_names(&t));
}

#[test]
fn txt_delimiter_is_sniffed_from_the_header() {
    let dir = tempfile::tempdir().unwrap();
    // `.txt` says nothing about its format, so the header decides.
    let tabbed = write_file(
        dir.path(),
        "a.txt",
        "barcode\tscore\tcall\nAAAC-1\t0.5\tsinglet\n",
    );
    let (_, info) = read_annotation_table(&tabbed, &opts()).unwrap();
    assert_eq!(info.delimiter, Some(b'\t'));

    let commaed = write_file(
        dir.path(),
        "b.txt",
        "barcode,score,call\nAAAC-1,0.5,singlet\n",
    );
    let (_, info) = read_annotation_table(&commaed, &opts()).unwrap();
    assert_eq!(info.delimiter, Some(b','));
}

#[test]
fn an_explicit_delimiter_overrides_the_extension() {
    let dir = tempfile::tempdir().unwrap();
    // Mislabelled: tab-separated content in a .csv.
    let p = write_file(dir.path(), "wrong.csv", "barcode\tscore\nAAAC-1\t0.5\n");
    let (data, info) = read_annotation_table(
        &p,
        &AnnotationTableOptions {
            delimiter: Some(b'\t'),
            ..opts()
        },
    )
    .unwrap();
    assert_eq!(info.delimiter, Some(b'\t'));
    assert_eq!(data.row_keys, vec!["AAAC-1"]);
}

// ---------------------------------------------------------------------------
// Real-world file shapes
// ---------------------------------------------------------------------------

/// R's `write.csv(colData(sce))` writes `NA` for missing values. arrow's
/// default null rule only recognises the empty string, so without an explicit
/// null-token regex a single `NA` turns the score column into `Utf8` and the
/// score silently lands as a string.
///
/// This is the flagship path — scDblFinder → `write.csv` → import — and
/// `scDblFinder.mostLikelyOrigin` is `NA` for random-origin doublets, so it
/// fires on the first real file.
#[test]
fn r_style_na_does_not_poison_numeric_inference() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(
        dir.path(),
        "sce.csv",
        "barcode,scDblFinder.score,scDblFinder.mostLikelyOrigin\n\
         AAAC-1,0.02,NA\n\
         AAAG-1,0.98,1+2\n\
         AAAT-1,NA,NA\n",
    );

    let (data, _) = read_annotation_table(&p, &opts()).unwrap();
    assert_eq!(
        dtype_of(&data, "scDblFinder.score"),
        DataType::Float64,
        "an NA must be a null, not a token that forces the column to Utf8"
    );

    let score = data
        .row_annotations
        .column_by_name("scDblFinder.score")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(score.value(0), 0.02);
    assert_eq!(score.value(1), 0.98);
    assert!(score.is_null(2), "NA must read back as null");

    let origin = data
        .row_annotations
        .column_by_name("scDblFinder.mostLikelyOrigin")
        .unwrap();
    assert!(origin.is_null(0) && origin.is_null(2));
}

/// A barcode containing the letters `NA` must not be nulled — the null tokens
/// are anchored for exactly this reason.
#[test]
fn a_barcode_containing_na_is_not_treated_as_missing() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(
        dir.path(),
        "a.csv",
        "barcode,score\nNANA-1,0.5\nSNAP-2,0.6\n",
    );
    let (data, _) = read_annotation_table(&p, &opts()).unwrap();
    assert_eq!(data.row_keys, vec!["NANA-1", "SNAP-2"]);
}

/// `adata.obs.to_csv()` emits a leading unnamed column holding the index.
/// arrow names it `""`, which no key fallback matches, so auto-resolution would
/// fail on the most likely file a user hands us.
#[test]
fn pandas_style_unnamed_index_column_becomes_the_key() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(
        dir.path(),
        "scrublet.csv",
        ",doublet_score,predicted_doublet\nAAAC-1,0.03,False\nAAAG-1,0.71,True\n",
    );

    let (data, info) = read_annotation_table(&p, &opts()).unwrap();
    assert!(info.renamed_index_column);
    assert_eq!(info.key_columns, vec!["_index".to_string()]);
    assert_eq!(data.row_keys, vec!["AAAC-1", "AAAG-1"]);
    assert_eq!(
        col_names(&data),
        vec!["doublet_score", "predicted_doublet"],
        "the renamed index must not also appear as an annotation"
    );
}

/// The hazard the `Utf8` pin exists for: inference would narrow `0012` to
/// `Int64`, and casting back gives `12`, matching no target row.
#[test]
fn numeric_looking_barcodes_stay_strings() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(dir.path(), "a.csv", "barcode,score\n0012,0.5\n0034,0.6\n");
    let (data, _) = read_annotation_table(&p, &opts()).unwrap();
    assert_eq!(
        data.row_keys,
        vec!["0012", "0034"],
        "leading zeros must survive; a narrowed key silently matches nothing"
    );
}

/// The Census shape: the unique key is a large integer column.
#[test]
fn an_integer_key_column_round_trips_as_its_text_form() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(
        dir.path(),
        "a.csv",
        "soma_joinid,score\n5718,0.1\n5719,0.2\n",
    );
    let (data, _) = read_annotation_table(
        &p,
        &AnnotationTableOptions {
            key_columns: vec!["soma_joinid".into()],
            ..opts()
        },
    )
    .unwrap();
    assert_eq!(data.row_keys, vec!["5718", "5719"]);
}

// ---------------------------------------------------------------------------
// Keys
// ---------------------------------------------------------------------------

#[test]
fn composite_key_columns_are_consumed_and_excluded_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(
        dir.path(),
        "a.csv",
        "sample_id,barcode,score\nA,AAAC-1,0.1\nB,AAAC-1,0.2\n",
    );
    let (data, info) = read_annotation_table(
        &p,
        &AnnotationTableOptions {
            key_columns: vec!["sample_id".into(), "barcode".into()],
            ..opts()
        },
    )
    .unwrap();

    assert_eq!(info.key_columns, vec!["sample_id", "barcode"]);
    assert_eq!(col_names(&data), vec!["score"]);
    // Fused with the ops crate's internal separator, so both sides of the join
    // are built by the same code.
    let sep = scx_ops::COMPOSITE_KEY_SEPARATOR;
    assert_eq!(
        data.row_keys,
        vec![format!("A{sep}AAAC-1"), format!("B{sep}AAAC-1")]
    );
}

#[test]
fn keep_key_columns_opts_them_back_in() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(dir.path(), "a.csv", "barcode,score\nAAAC-1,0.5\n");
    let (data, _) = read_annotation_table(
        &p,
        &AnnotationTableOptions {
            keep_key_columns: true,
            ..opts()
        },
    )
    .unwrap();
    assert_eq!(col_names(&data), vec!["barcode", "score"]);
}

#[test]
fn missing_key_column_errors_listing_present_columns() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(dir.path(), "a.csv", "barcode,score\nAAAC-1,0.5\n");
    let err = read_annotation_table(
        &p,
        &AnnotationTableOptions {
            key_columns: vec!["cell_id".into()],
            ..opts()
        },
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("cell_id"), "{msg}");
    assert!(msg.contains("barcode") && msg.contains("score"), "{msg}");
}

#[test]
fn a_table_with_no_resolvable_key_errors() {
    let dir = tempfile::tempdir().unwrap();
    // No fallback name, no pandas index.
    let p = write_file(dir.path(), "a.csv", "score,call\n0.5,singlet\n");
    let err = read_annotation_table(&p, &opts()).unwrap_err();
    assert!(matches!(err, OpsError::KeyColumnUnresolved { .. }), "{err}");
}

// ---------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------

#[test]
fn column_subset_and_rename_apply() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(
        dir.path(),
        "a.csv",
        "barcode,score,call,extra\nAAAC-1,0.5,singlet,9\n",
    );
    let mut rename = HashMap::new();
    rename.insert("score".to_string(), "doublet_score".to_string());

    let (data, info) = read_annotation_table(
        &p,
        &AnnotationTableOptions {
            columns: Some(vec!["score".into(), "call".into()]),
            rename,
            prefix: "scdbl_".into(),
            ..opts()
        },
    )
    .unwrap();

    assert_eq!(col_names(&data), vec!["scdbl_doublet_score", "scdbl_call"]);
    assert_eq!(
        info.columns_imported,
        vec!["scdbl_doublet_score", "scdbl_call"]
    );
}

#[test]
fn a_requested_column_that_is_absent_errors() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(dir.path(), "a.csv", "barcode,score\nAAAC-1,0.5\n");
    let err = read_annotation_table(
        &p,
        &AnnotationTableOptions {
            columns: Some(vec!["nope".into()]),
            ..opts()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("nope"), "{err}");
    assert!(err.to_string().contains("score"), "{err}");
}

#[test]
fn a_rename_of_an_absent_column_errors() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(dir.path(), "a.csv", "barcode,score\nAAAC-1,0.5\n");
    let mut rename = HashMap::new();
    rename.insert("nope".to_string(), "x".to_string());
    let err = read_annotation_table(&p, &AnnotationTableOptions { rename, ..opts() }).unwrap_err();
    assert!(err.to_string().contains("rename source"), "{err}");
}

#[test]
fn every_imported_field_is_nullable() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(
        dir.path(),
        "a.csv",
        "barcode,score,call\nAAAC-1,0.5,singlet\n",
    );
    let (data, _) = read_annotation_table(&p, &opts()).unwrap();
    for f in data.row_annotations.schema().fields() {
        assert!(
            f.is_nullable(),
            "'{}' must admit nulls — the attach op scatters null into every \
             target row this table does not cover",
            f.name()
        );
    }
}

#[test]
fn a_table_of_only_a_key_column_errors_rather_than_attaching_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(dir.path(), "a.csv", "barcode\nAAAC-1\n");
    let err = read_annotation_table(&p, &opts()).unwrap_err();
    assert!(err.to_string().contains("no columns left"), "{err}");
}

// ---------------------------------------------------------------------------
// Malformed input
// ---------------------------------------------------------------------------

#[test]
fn empty_table_errors() {
    let dir = tempfile::tempdir().unwrap();
    let header_only = write_file(dir.path(), "a.csv", "barcode,score\n");
    let err = read_annotation_table(&header_only, &opts()).unwrap_err();
    assert!(err.to_string().contains("no data rows"), "{err}");

    let empty = write_file(dir.path(), "b.csv", "");
    assert!(read_annotation_table(&empty, &opts()).is_err());
}

#[test]
fn duplicate_or_blank_column_names_are_rejected() {
    let dir = tempfile::tempdir().unwrap();

    let dup = write_file(dir.path(), "d.csv", "barcode,score,score\nAAAC-1,0.1,0.2\n");
    let err = read_annotation_table(&dup, &opts()).unwrap_err();
    assert!(err.to_string().contains("duplicate column name"), "{err}");

    // A blank name anywhere but position 0 is not the pandas-index shape.
    let blank = write_file(dir.path(), "b.csv", "barcode,,score\nAAAC-1,x,0.1\n");
    let err = read_annotation_table(&blank, &opts()).unwrap_err();
    assert!(err.to_string().contains("blank name"), "{err}");
}

#[test]
fn a_missing_file_errors_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    let err = read_annotation_table(&dir.path().join("nope.csv"), &opts()).unwrap_err();
    assert!(matches!(err, OpsError::Io(_)), "{err}");
}

// ---------------------------------------------------------------------------
// Integration with the attach op
// ---------------------------------------------------------------------------

/// The Phase-2 exit criterion: a CSV of `barcode,score,call` lands on a fixture
/// SCX file, joined by key rather than by position.
#[test]
fn round_trip_through_attach_external_obs() {
    use arrow::array::{Float32Array, RecordBatch as RB};
    use arrow::datatypes::{Field, Schema};
    use scx_codec::{CodecId, ValueEncoding};
    use scx_format_io::header::FileHeader;
    use scx_format_io::writer::ScxWriter;
    use scx_format_io::ScxReader;
    use std::sync::Arc;

    let dir = tempfile::tempdir().unwrap();

    // A 4-cell fixture.
    let n_obs = 4usize;
    let obs = RB::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "barcode",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(vec![
            "AAAC-1", "AAAG-1", "AAAT-1", "AAAA-1",
        ]))],
    )
    .unwrap();
    let var = RB::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "gene_id",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(vec!["g0", "g1"]))],
    )
    .unwrap();
    let scx = dir.path().join("t.scx");
    let mut w = ScxWriter::new(&scx, FileHeader::new_single_modality(4, 2, 0, 4, 0, 0)).unwrap();
    w.write_obs(&obs).unwrap();
    w.write_var(&var).unwrap();
    w.write_csr_shard(
        &vec![0u64; n_obs + 1],
        &[],
        &[],
        CodecId::None,
        ValueEncoding::Uint8,
        0,
    )
    .unwrap();
    w.finish().unwrap();

    // Deliberately out of order and covering only 3 of 4 cells, which is what a
    // real tool hands back.
    let csv = write_file(
        dir.path(),
        "calls.csv",
        "barcode,score,call\nAAAT-1,0.30,singlet\nAAAC-1,0.10,singlet\nAAAG-1,0.90,doublet\n",
    );

    let (data, info) = read_annotation_table(&csv, &opts()).unwrap();
    assert_eq!(info.n_rows, 3);

    let summary = scx_ops::attach_external_obs(
        &scx,
        &data,
        &scx_ops::AttachObsOptions {
            status_column: Some("dbl_status".into()),
            provenance_action: "test_table_import".into(),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(summary.n_matched, 3);
    assert_eq!(summary.n_target_rows_absent, 1);

    let back = ScxReader::open(&scx).unwrap().read_obs().unwrap();
    let score =
        arrow::compute::cast(back.column_by_name("score").unwrap(), &DataType::Float32).unwrap();
    let score = score.as_any().downcast_ref::<Float32Array>().unwrap();
    // Row order is the FILE's, not the CSV's — proof the join used the key.
    assert_eq!(score.value(0), 0.10); // AAAC-1
    assert_eq!(score.value(1), 0.90); // AAAG-1
    assert_eq!(score.value(2), 0.30); // AAAT-1
    assert!(score.is_null(3), "AAAA-1 was not in the table"); // AAAA-1

    let status = back.column_by_name("dbl_status").unwrap();
    let status = arrow::compute::cast(status, &DataType::Utf8).unwrap();
    let status = status.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(status.value(0), "present");
    assert_eq!(status.value(3), "absent");
}

/// The exact byte shape `write.csv(as.data.frame(colData(sce)))` produces in R:
/// every field quoted, the rownames column with an **empty quoted** header, and
/// `NA` for missing values. This is the flagship Phase-4 acceptance path, so it
/// gets a test of its own rather than being approximated by the two shapes
/// above.
#[test]
fn the_literal_r_write_csv_shape_parses() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(
        dir.path(),
        "sce_colData.csv",
        "\"\",\"scDblFinder.score\",\"scDblFinder.class\",\"scDblFinder.mostLikelyOrigin\"\n\
         \"AAACCTGAGAAACCAT-1\",0.0213,\"singlet\",NA\n\
         \"AAACCTGAGAAACCGC-1\",0.9871,\"doublet\",\"1+2\"\n",
    );

    let (data, info) = read_annotation_table(&p, &opts()).unwrap();

    assert!(info.renamed_index_column);
    assert_eq!(info.key_columns, vec!["_index".to_string()]);
    assert_eq!(
        data.row_keys,
        vec!["AAACCTGAGAAACCAT-1", "AAACCTGAGAAACCGC-1"],
        "quotes must be stripped from the key, or nothing joins"
    );
    assert_eq!(
        dtype_of(&data, "scDblFinder.score"),
        DataType::Float64,
        "the NA in a sibling column must not drag the score to Utf8"
    );
    assert_eq!(
        col_names(&data),
        vec![
            "scDblFinder.score",
            "scDblFinder.class",
            "scDblFinder.mostLikelyOrigin"
        ]
    );
    let origin = data
        .row_annotations
        .column_by_name("scDblFinder.mostLikelyOrigin")
        .unwrap();
    assert!(
        origin.is_null(0),
        "NA is missing, not the literal text 'NA'"
    );
    assert!(origin.is_valid(1));
}

// ---------------------------------------------------------------------------
// Source dispatch
// ---------------------------------------------------------------------------

#[test]
fn sniff_routes_by_extension() {
    use std::path::Path;
    assert_eq!(
        sniff_obs_source(Path::new("calls.csv")),
        ObsSourceFormat::Table
    );
    assert_eq!(
        sniff_obs_source(Path::new("calls.tsv")),
        ObsSourceFormat::Table
    );
    // No extension at all is still a table — that is the common case for a
    // pipeline writing to a bare filename, and the delimiter sniffer copes.
    assert_eq!(sniff_obs_source(Path::new("calls")), ObsSourceFormat::Table);
    assert_eq!(
        sniff_obs_source(Path::new("out.h5ad")),
        ObsSourceFormat::H5ad
    );
    // Case-insensitive: an uppercase extension is the same file.
    assert_eq!(
        sniff_obs_source(Path::new("OUT.H5AD")),
        ObsSourceFormat::H5ad
    );
}

#[test]
fn read_obs_source_reads_a_table_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(
        dir.path(),
        "calls.csv",
        "barcode,score\nAAAC-1,0.5\nAAAG-1,0.9\n",
    );

    let (data, info) = read_obs_source(&p, &AnnotationTableOptions::default(), &[]).unwrap();
    assert_eq!(info.format, ObsSourceFormat::Table);
    assert_eq!(info.delimiter, Some(b','));
    assert_eq!(data.row_keys, ["AAAC-1", "AAAG-1"]);
    assert!(info.uns_keys_imported.is_empty());
}

#[test]
fn uns_keys_on_a_table_source_error_rather_than_being_ignored() {
    // Silently dropping the request would leave the caller believing the
    // metadata came across.
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(dir.path(), "calls.csv", "barcode,score\nAAAC-1,0.5\n");

    let e = read_obs_source(
        &p,
        &AnnotationTableOptions::default(),
        &["scrublet".to_string()],
    )
    .unwrap_err();
    let m = e.to_string();
    assert!(m.contains("scrublet"), "{m}");
    assert!(m.contains("carries no uns"), "{m}");
}

#[test]
fn an_h5mu_source_is_refused_with_the_modality_route() {
    // obs lives at /mod/<name>/obs there, so there is no single table to read.
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(dir.path(), "atlas.h5mu", "not really hdf5");

    let e = read_obs_source(&p, &AnnotationTableOptions::default(), &[]).unwrap_err();
    let m = e.to_string();
    assert!(m.contains("multimodal"), "{m}");
    assert!(m.contains("--modality"), "{m}");
}

#[cfg(not(feature = "hdf5"))]
#[test]
fn an_h5ad_source_without_the_feature_says_what_to_do_instead() {
    let dir = tempfile::tempdir().unwrap();
    let p = write_file(dir.path(), "out.h5ad", "not really hdf5");

    let e = read_obs_source(&p, &AnnotationTableOptions::default(), &[]).unwrap_err();
    let m = e.to_string();
    assert!(m.contains("no HDF5 support"), "{m}");
    assert!(m.contains("to_csv"), "{m}");
}
