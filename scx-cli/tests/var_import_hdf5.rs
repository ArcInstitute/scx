//! The hdf5-gated half of `scx var-import`'s source handling, driven through
//! the real binary.
//!
//! File-level gated on purpose: hdf5-gated tests live in hdf5-gated test
//! binaries so CI's hdf5 lane selects them with `--test var_import_hdf5` (the
//! dedup-guard's "test-hdf5 lane selection lists" step pins exactly that) and
//! no per-name recovery invocation exists for a gated test hiding inside an
//! ungated binary. The ungated sibling `var_import_cli.rs` carries the
//! `cfg(not(feature = "hdf5"))` counterpart and everything libhdf5-free.
#![cfg(feature = "hdf5")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use arrow::array::{Array, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;

fn scx() -> Command {
    Command::new(env!("CARGO_BIN_EXE_scx"))
}

/// A three-gene file keyed by `gene_id`, symbols in a second column.
fn simple_fixture(dir: &Path, name: &str) -> PathBuf {
    let n_obs = 2usize;
    let n_vars = 3usize;
    let obs = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "barcode",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(vec!["c0", "c1"]))],
    )
    .unwrap();
    let var = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("gene_id", DataType::Utf8, false),
            Field::new("gene_symbol", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["ENSG0", "ENSG1", "ENSG2"])),
            Arc::new(StringArray::from(vec!["SYM0", "SYM1", "SYM2"])),
        ],
    )
    .unwrap();
    let path = dir.join(name);
    let mut w = ScxWriter::new(
        &path,
        FileHeader::new_single_modality(n_obs as u64, n_vars as u64, 0, 16384, 0, 0),
    )
    .unwrap();
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
    path
}

/// An h5ad whose `/var` holds the annotation, in anndata's own layout: an
/// unnamed index (the `_index` sentinel) plus two data columns.
fn write_var_h5ad(dir: &Path, name: &str) -> PathBuf {
    use hdf5::types::VarLenUnicode;
    let vlu = |s: &str| -> VarLenUnicode { s.parse().unwrap() };
    let path = dir.join(name);
    let file = hdf5::File::create(&path).unwrap();
    let var = file.create_group("var").unwrap();
    for (k, v) in [
        ("encoding-type", "dataframe"),
        ("encoding-version", "0.2.0"),
        ("_index", "_index"),
    ] {
        var.new_attr::<VarLenUnicode>()
            .create(k)
            .unwrap()
            .write_scalar(&vlu(v))
            .unwrap();
    }
    let order: Vec<VarLenUnicode> = ["ens", "score"].iter().map(|s| vlu(s)).collect();
    var.new_attr::<VarLenUnicode>()
        .shape([order.len()])
        .create("column-order")
        .unwrap()
        .write(&order)
        .unwrap();
    // Index holds symbols; the join keys on `ens`, so the index must NOT be
    // imported as a column of its own.
    for (ds, vals) in [
        ("_index", vec!["SYM2", "SYM0"]),
        ("ens", vec!["ENSG2", "ENSG0"]),
    ] {
        let v: Vec<VarLenUnicode> = vals.iter().map(|s| vlu(s)).collect();
        var.new_dataset::<VarLenUnicode>()
            .shape([v.len()])
            .create(ds)
            .unwrap()
            .write(&v)
            .unwrap();
    }
    var.new_dataset::<f64>()
        .shape([2])
        .create("score")
        .unwrap()
        .write(&[2.0f64, 0.0])
        .unwrap();
    file.close().unwrap();
    path
}

fn f64_col(batch: &RecordBatch, name: &str) -> Float64Array {
    let col = batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("column '{name}' missing"));
    arrow::compute::cast(col, &DataType::Float64)
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .clone()
}

#[test]
fn imports_var_columns_from_an_h5ads_var_group() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "a.scx");
    let h5 = write_var_h5ad(dir.path(), "src.h5ad");

    let out = scx()
        .args([
            "var-import",
            scx_path.to_str().unwrap(),
            h5.to_str().unwrap(),
            "--key",
            "gene_id",
            "--source-key",
            "ens",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("(h5ad, 2 rows"), "{stdout}");
    assert!(stdout.contains("2/3 genes matched"), "{stdout}");

    let var = ScxReader::open(&scx_path).unwrap().read_var().unwrap();
    let got = f64_col(&var, "score");
    // Source row order is reversed relative to the file: a positional import
    // would swap these.
    assert_eq!(got.value(0), 0.0);
    assert!(got.is_null(1));
    assert_eq!(got.value(2), 2.0);
    assert!(
        var.column_by_name("__index_level_0__").is_none(),
        "the source h5ad's own index must not land as a var column: {:?}",
        var.schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>()
    );
}

#[test]
fn an_h5mu_source_is_refused_naming_the_extraction_route() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "a.scx");
    // Only the extension is inspected before the refusal, so the contents do
    // not have to be a real h5mu.
    let fake = dir.path().join("x.h5mu");
    std::fs::write(&fake, b"not hdf5").unwrap();

    let out = scx()
        .args([
            "var-import",
            scx_path.to_str().unwrap(),
            fake.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("/mod/<modality>/var"), "{err}");
    assert!(err.contains("subset --modality"), "{err}");
}
