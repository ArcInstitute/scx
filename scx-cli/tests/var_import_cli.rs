//! End-to-end `scx var-import`, driven through the real binary.
//!
//! Deliberately **ungated**, like its obs twin: the whole point of the CSV
//! importer is that it works without libhdf5, so the fixture is built directly
//! with `ScxWriter` rather than converted from h5ad.
//!
//! What matters here is the join, not the write. A key that fails to line up
//! produces a plausible-looking but empty column, so the tests check where the
//! *values* landed, not just that the command exited 0.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use arrow::array::{Array, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;

fn scx() -> Command {
    Command::new(env!("CARGO_BIN_EXE_scx"))
}

fn obs_batch(n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "barcode",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(ids))],
    )
    .unwrap()
}

/// Three genes keyed by an Ensembl-style `gene_id`, with the symbol in a
/// second column so a test can key on either.
fn var_batch(n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("ENSG{i}")).collect();
    let syms: Vec<String> = (0..n).map(|i| format!("SYM{i}")).collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("gene_id", DataType::Utf8, false),
            Field::new("gene_symbol", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(syms)),
        ],
    )
    .unwrap()
}

/// `var_shard_rows = None` writes a single `VarMetadata` section; `Some(rows)`
/// writes that many `VarMetadataShard`s, which is what a `from_anndata` output
/// above `shard_target_rows` looks like.
fn write_fixture(
    dir: &Path,
    name: &str,
    n_vars: usize,
    var_shard_rows: Option<&[usize]>,
) -> PathBuf {
    let n_obs = 4usize;
    let path = dir.join(name);
    let var = var_batch(n_vars);
    let mut w = ScxWriter::new(
        &path,
        FileHeader::new_single_modality(n_obs as u64, n_vars as u64, 0, 16384, 0, 0),
    )
    .unwrap();
    w.write_obs(&obs_batch(n_obs)).unwrap();
    match var_shard_rows {
        None => w.write_var(&var).unwrap(),
        Some(rows) => {
            assert_eq!(rows.iter().sum::<usize>(), n_vars);
            let mut start = 0usize;
            for (idx, take) in rows.iter().enumerate() {
                w.write_var_shard(
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

fn simple_fixture(dir: &Path, name: &str) -> PathBuf {
    write_fixture(dir, name, 3, None)
}

fn write_text(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    p
}

fn read_var(path: &Path) -> RecordBatch {
    ScxReader::open(path).unwrap().read_var().unwrap()
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

fn str_col(batch: &RecordBatch, name: &str) -> StringArray {
    let col = batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("column '{name}' missing"));
    arrow::compute::cast(col, &DataType::Utf8)
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .clone()
}

#[test]
fn imports_a_csv_and_joins_by_key_not_position() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "a.scx");
    // Reverse order and only two of three genes: a positional import would put
    // both scores on the wrong genes and leave the wrong one null.
    let csv = write_text(
        dir.path(),
        "peaks.csv",
        "gene_id,peak_score\nENSG2,0.25\nENSG0,0.75\n",
    );

    let out = scx()
        .args([
            "var-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("2/3 genes matched"), "{stdout}");
    assert!(stdout.contains("Undo with: scx rollback"), "{stdout}");

    let var = read_var(&scx_path);
    let got = f64_col(&var, "peak_score");
    assert_eq!(got.value(0), 0.75);
    assert!(got.is_null(1), "the uncovered gene must be null, not 0.0");
    assert_eq!(got.value(2), 0.25);
    // The pre-existing column survives.
    assert_eq!(str_col(&var, "gene_symbol").value(1), "SYM1");
}

#[test]
fn dry_run_leaves_the_file_byte_identical_and_prints_a_diagnosis() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "a.scx");
    let csv = write_text(dir.path(), "p.csv", "gene_id,score\nENSG0,1.0\n");
    let before = std::fs::read(&scx_path).unwrap();

    let out = scx()
        .args([
            "var-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Dry run: nothing written."), "{stdout}");
    assert!(stdout.contains("Key diagnosis:"), "{stdout}");
    assert!(
        stdout.contains("var columns that ARE unique"),
        "the diagnosis must talk about var, not obs: {stdout}"
    );
    assert_eq!(std::fs::read(&scx_path).unwrap(), before);
}

#[test]
fn var_names_keys_on_the_var_index_and_is_reported_as_such() {
    let dir = tempfile::tempdir().unwrap();
    // A file whose var index is the physical `__index_level_0__` field.
    let path = dir.path().join("idx.scx");
    let var = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "__index_level_0__",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(vec!["GA", "GB"]))],
    )
    .unwrap();
    let mut w =
        ScxWriter::new(&path, FileHeader::new_single_modality(2, 2, 0, 16384, 0, 0)).unwrap();
    w.write_obs(&obs_batch(2)).unwrap();
    w.write_var(&var).unwrap();
    w.write_csr_shard(&[0u64; 3], &[], &[], CodecId::None, ValueEncoding::Uint8, 0)
        .unwrap();
    w.finish().unwrap();

    let csv = write_text(dir.path(), "p.csv", "var_names,score\nGB,2.0\nGA,1.0\n");
    let out = scx()
        .args([
            "var-import",
            path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--key",
            "var_names",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("matched on var_names"),
        "the physical field name must never be printed: {stdout}"
    );
    let got = f64_col(&read_var(&path), "score");
    assert_eq!((got.value(0), got.value(1)), (1.0, 2.0));
}

#[test]
fn a_composite_key_can_name_different_columns_per_side() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "a.scx");
    let csv = write_text(
        dir.path(),
        "p.csv",
        "ens,sym,score\nENSG1,SYM1,1.0\nENSG0,SYM0,0.0\n",
    );
    let out = scx()
        .args([
            "var-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--key",
            "gene_id,gene_symbol",
            "--source-key",
            "ens,sym",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let got = f64_col(&read_var(&scx_path), "score");
    assert_eq!(got.value(0), 0.0);
    assert_eq!(got.value(1), 1.0);
    assert!(got.is_null(2));
}

#[test]
fn source_key_without_key_is_refused_and_names_var_names() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "a.scx");
    let csv = write_text(dir.path(), "p.csv", "ens,score\nENSG0,1.0\n");
    let out = scx()
        .args([
            "var-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--source-key",
            "ens",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--source-key needs --key"), "{err}");
    assert!(
        err.contains("--key var_names"),
        "the remedy must name the var index, not obs_names: {err}"
    );
}

#[test]
fn an_unknown_key_column_exits_nonzero_and_names_the_present_columns() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "a.scx");
    let csv = write_text(dir.path(), "p.csv", "gene_id,score\nENSG0,1.0\n");
    let out = scx()
        .args([
            "var-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--key",
            "nope",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("nope"), "{err}");
    assert!(err.contains("gene_id"), "{err}");
}

#[test]
fn a_second_import_errors_and_overwrite_replaces_rather_than_merging() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "a.scx");
    let a = write_text(dir.path(), "a.csv", "gene_id,score\nENSG0,1.0\n");
    let b = write_text(dir.path(), "b.csv", "gene_id,score\nENSG1,2.0\n");

    let run = |table: &Path, extra: &[&str]| {
        let mut c = scx();
        c.args([
            "var-import",
            scx_path.to_str().unwrap(),
            table.to_str().unwrap(),
        ]);
        c.args(extra);
        c.output().unwrap()
    };

    assert!(run(&a, &[]).status.success());
    let second = run(&b, &[]);
    assert!(!second.status.success());
    assert!(String::from_utf8_lossy(&second.stderr).contains("overwrite=true"));

    assert!(run(&b, &["--overwrite"]).status.success());
    let got = f64_col(&read_var(&scx_path), "score");
    assert!(
        got.is_null(0),
        "overwrite REPLACES; the concatenate-first contract must hold"
    );
    assert_eq!(got.value(1), 2.0);
}

#[test]
fn rollback_undoes_the_import() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "a.scx");
    let csv = write_text(dir.path(), "p.csv", "gene_id,score\nENSG0,1.0\n");
    assert!(scx()
        .args([
            "var-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap()
        ])
        .output()
        .unwrap()
        .status
        .success());
    assert!(read_var(&scx_path).column_by_name("score").is_some());

    assert!(scx()
        .args(["rollback", scx_path.to_str().unwrap()])
        .output()
        .unwrap()
        .status
        .success());
    assert!(read_var(&scx_path).column_by_name("score").is_none());
}

#[test]
fn column_selection_rename_and_prefix_apply() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "a.scx");
    let csv = write_text(
        dir.path(),
        "p.csv",
        "gene_id,keep,drop_me\nENSG0,1.0,9.0\nENSG1,2.0,9.0\nENSG2,3.0,9.0\n",
    );
    let out = scx()
        .args([
            "var-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--columns",
            "keep",
            "--rename",
            "keep=kept",
            "--prefix",
            "pk_",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let var = read_var(&scx_path);
    assert!(var.column_by_name("pk_kept").is_some());
    assert!(var.column_by_name("drop_me").is_none());
}

/// A sharded var must keep its layout through the CLI too — the property the
/// layer importer had been silently losing.
#[test]
fn a_sharded_var_keeps_its_shard_layout() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = write_fixture(dir.path(), "s.scx", 3, Some(&[2, 1]));
    let shards = |p: &Path| -> usize {
        ScxReader::open(p)
            .unwrap()
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::VarMetadataShard)
            .count()
    };
    assert_eq!(shards(&scx_path), 2, "fixture premise: two var shards");

    let csv = write_text(dir.path(), "p.csv", "gene_id,score\nENSG0,1.0\nENSG2,3.0\n");
    let out = scx()
        .args([
            "var-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("streamed shard-by-shard"),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert_eq!(shards(&scx_path), 2);
    let got = f64_col(&read_var(&scx_path), "score");
    assert_eq!(got.value(0), 1.0);
    assert!(got.is_null(1));
    assert_eq!(got.value(2), 3.0);
}

/// The CLI must say which of the two index outcomes happened, on a dry run as
/// well as a real import — a caller who indexed a var column deliberately needs
/// to know before losing pushdown, not after.
#[test]
fn an_unindexable_overwrite_reports_the_dropped_index_on_both_paths() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("idx.scx");
    let var = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("gene_id", DataType::Utf8, false),
            Field::new("feature_type", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["ENSG0", "ENSG1"])),
            Arc::new(StringArray::from(vec!["Gene Expression", "Peaks"])),
        ],
    )
    .unwrap();
    let mut w =
        ScxWriter::new(&path, FileHeader::new_single_modality(4, 2, 0, 16384, 0, 0)).unwrap();
    w.write_obs(&obs_batch(4)).unwrap();
    w.write_var(&var).unwrap();
    w.write_csr_shard(&[0u64; 5], &[], &[], CodecId::None, ValueEncoding::Uint8, 0)
        .unwrap();
    let opts = scx_engine::PredicateIndexBuildOptions {
        forced_columns: vec!["feature_type".to_string()],
        preset_columns: Vec::new(),
        auto_threshold: 1000,
        high_cardinality_threshold: 100_000,
    };
    let (mut outcomes, mut named) = (Vec::new(), Vec::new());
    let bytes = scx_engine::build_var_predicate_index_bytes(
        &var,
        &[(0, 2)],
        &opts,
        &mut outcomes,
        &mut named,
    )
    .unwrap()
    .expect("fixture premise: feature_type must be indexable");
    w.write_var_predicate_index(&bytes).unwrap();
    w.finish().unwrap();

    // Overwrite the indexed string column with booleans: nothing left to index.
    let csv = write_text(
        dir.path(),
        "p.csv",
        "gene_id,feature_type
ENSG0,true
ENSG1,false
",
    );
    let run = |extra: &[&str]| {
        let mut c = scx();
        c.args([
            "var-import",
            path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--overwrite",
        ]);
        c.args(extra);
        c.output().unwrap()
    };

    let preview = run(&["--dry-run"]);
    assert!(
        preview.status.success(),
        "{}",
        String::from_utf8_lossy(&preview.stderr)
    );
    let out = String::from_utf8_lossy(&preview.stdout);
    assert!(
        out.contains("would be DROPPED"),
        "a dry run must report the drop it is previewing, in the conditional: {out}"
    );
    assert!(
        !out.contains("was DROPPED"),
        "and must not claim it already happened, one line above \
         'nothing written': {out}"
    );

    let real = run(&[]);
    assert!(real.status.success());
    let out = String::from_utf8_lossy(&real.stdout);
    assert!(out.contains("was DROPPED"), "{out}");
    assert!(out.contains("no longer covers"), "{out}");
    assert!(
        ScxReader::open(&path)
            .unwrap()
            .read_var_predicate_index_bytes()
            .unwrap()
            .is_none(),
        "the stale section must be gone rather than left describing dead values"
    );
}

#[test]
fn a_status_column_marks_present_and_absent_genes() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "a.scx");
    let csv = write_text(dir.path(), "p.csv", "gene_id,score\nENSG1,1.0\n");
    let out = scx()
        .args([
            "var-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--status-column",
            "ann_status",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = str_col(&read_var(&scx_path), "ann_status");
    assert_eq!(
        (0..3).map(|i| s.value(i)).collect::<Vec<_>>(),
        vec!["absent", "present", "absent"]
    );
}

/// The `cfg(not(hdf5))` counterpart to `var_import_hdf5.rs`: the *source
/// format* is unavailable without libhdf5, never the command, and the error
/// must name the way out. Gated on the test rather than inside it, so under
/// `hdf5` it does not exist at all instead of passing vacuously.
#[cfg(not(feature = "hdf5"))]
#[test]
fn an_h5ad_source_is_refused_with_the_csv_route() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "a.scx");
    let fake = write_text(dir.path(), "x.h5ad", "not hdf5");
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
    assert!(err.contains("no HDF5 support"), "{err}");
    assert!(err.contains("adata.var"), "{err}");
}
