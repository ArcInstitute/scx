//! End-to-end `scx obs-import`, driven through the real binary.
//!
//! Deliberately **ungated**: the whole point of the CSV importer is that it
//! works without libhdf5, so the fixture is built directly with `ScxWriter`
//! (the `modify_metadata_cli.rs` pattern) rather than converted from h5ad.
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
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;

fn scx() -> Command {
    Command::new(env!("CARGO_BIN_EXE_scx"))
}

fn var_batch(n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("g{i}")).collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "gene_id",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(ids))],
    )
    .unwrap()
}

fn write_fixture(dir: &Path, name: &str, obs: RecordBatch, n_vars: usize) -> PathBuf {
    let n_obs = obs.num_rows();
    let path = dir.join(name);
    let mut w = ScxWriter::new(
        &path,
        FileHeader::new_single_modality(n_obs as u64, n_vars as u64, 0, 16384, 0, 0),
    )
    .unwrap();
    w.write_obs(&obs).unwrap();
    w.write_var(&var_batch(n_vars)).unwrap();
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

/// Four cells with unique barcodes — the ordinary single-library shape.
fn simple_fixture(dir: &Path, name: &str) -> PathBuf {
    let obs = RecordBatch::try_new(
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
    write_fixture(dir, name, obs, 2)
}

fn write_text(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    p
}

fn read_obs(path: &Path) -> RecordBatch {
    ScxReader::open(path).unwrap().read_obs().unwrap()
}

/// Read a numeric column as f64. Casts rather than downcasts: arrow infers
/// `Int64` for a whole-number score column and `Float64` for a fractional one,
/// and the tests care about the values, not which the CSV happened to imply.
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
fn imports_a_csv_and_joins_by_key_not_position() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "t.scx");
    // Out of order, and covering only three of four cells — what a real tool
    // run per library hands back.
    let csv = write_text(
        dir.path(),
        "calls.csv",
        "barcode,score,call\nAAAT-1,0.30,singlet\nAAAC-1,0.10,singlet\nAAAG-1,0.90,doublet\n",
    );

    let out = scx()
        .args([
            "obs-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--status-column",
            "dbl_status",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("3/4 target rows matched"), "{stdout}");
    assert!(stdout.contains("Undo with: scx rollback"), "{stdout}");

    let obs = read_obs(&scx_path);
    let score = f64_col(&obs, "score");
    // File order, not CSV order.
    assert_eq!(score.value(0), 0.10);
    assert_eq!(score.value(1), 0.90);
    assert_eq!(score.value(2), 0.30);
    assert!(score.is_null(3), "the uncovered cell must be null, not 0.0");
}

#[test]
fn dry_run_leaves_the_file_byte_identical() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "t.scx");
    let csv = write_text(dir.path(), "calls.csv", "barcode,score\nAAAC-1,0.5\n");
    let before = std::fs::read(&scx_path).unwrap();

    let out = scx()
        .args([
            "obs-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Dry run: nothing written."), "{stdout}");
    assert!(stdout.contains("Key diagnosis:"), "{stdout}");

    assert_eq!(before, std::fs::read(&scx_path).unwrap());
}

#[test]
fn a_composite_key_disambiguates_repeated_barcodes() {
    let dir = tempfile::tempdir().unwrap();
    // Two libraries sharing barcodes: only sample_id+barcode is unique.
    let obs = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("sample_id", DataType::Utf8, false),
            Field::new("barcode", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["A", "A", "B", "B"])),
            Arc::new(StringArray::from(vec![
                "AAAC-1", "AAAG-1", "AAAC-1", "AAAG-1",
            ])),
        ],
    )
    .unwrap();
    let scx_path = write_fixture(dir.path(), "t.scx", obs, 2);
    let csv = write_text(
        dir.path(),
        "calls.csv",
        "sample_id,barcode,score\nA,AAAC-1,1\nA,AAAG-1,2\nB,AAAC-1,3\nB,AAAG-1,4\n",
    );

    let out = scx()
        .args([
            "obs-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--key",
            "sample_id,barcode",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let obs = read_obs(&scx_path);
    let score = f64_col(&obs, "score");
    for i in 0..4 {
        assert_eq!(score.value(i), (i + 1) as f64);
    }
}

#[test]
fn an_unknown_key_column_exits_nonzero_and_names_the_present_columns() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "t.scx");
    let csv = write_text(dir.path(), "calls.csv", "barcode,score\nAAAC-1,0.5\n");

    let out = scx()
        .args([
            "obs-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--key",
            "nope",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success(), "a bad --key must not exit 0");
    let msg = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(msg.contains("nope"), "{msg}");
    assert!(msg.contains("barcode"), "{msg}");
}

/// The Phase-0 failure mode in miniature: the auto-resolved key is duplicated,
/// and the user needs to be told which column would work instead.
#[test]
fn a_duplicated_key_is_refused_and_the_error_names_a_working_column() {
    let dir = tempfile::tempdir().unwrap();
    let obs = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("barcode", DataType::Utf8, false),
            Field::new("cell_uid", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["dup", "dup", "x", "y"])),
            Arc::new(StringArray::from(vec!["u0", "u1", "u2", "u3"])),
        ],
    )
    .unwrap();
    let scx_path = write_fixture(dir.path(), "t.scx", obs, 2);
    let csv = write_text(dir.path(), "calls.csv", "barcode,score\ndup,0.5\n");
    let before = std::fs::read(&scx_path).unwrap();

    let out = scx()
        .args([
            "obs-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--key",
            "barcode",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let msg = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        msg.contains("cell_uid"),
        "the error must name the column that WOULD work: {msg}"
    );
    assert_eq!(
        before,
        std::fs::read(&scx_path).unwrap(),
        "a refused import must not write a byte"
    );
}

#[test]
fn a_second_import_errors_and_overwrite_replaces_rather_than_merging() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "t.scx");
    let a = write_text(dir.path(), "a.csv", "barcode,score\nAAAC-1,1\nAAAG-1,2\n");
    let b = write_text(dir.path(), "b.csv", "barcode,score\nAAAT-1,3\nAAAA-1,4\n");

    let ok = scx()
        .args([
            "obs-import",
            scx_path.to_str().unwrap(),
            a.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(ok.status.success());

    let clash = scx()
        .args([
            "obs-import",
            scx_path.to_str().unwrap(),
            b.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        !clash.status.success(),
        "a colliding import must not exit 0"
    );
    let msg = String::from_utf8_lossy(&clash.stderr).to_string()
        + &String::from_utf8_lossy(&clash.stdout);
    assert!(msg.contains("already exists"), "{msg}");

    let forced = scx()
        .args([
            "obs-import",
            scx_path.to_str().unwrap(),
            b.to_str().unwrap(),
            "--overwrite",
        ])
        .output()
        .unwrap();
    assert!(forced.status.success());

    // Batch A's values are GONE — overwrite replaces, it does not merge. This
    // is the documented contract; if it ever starts merging, this fails.
    let obs = read_obs(&scx_path);
    let score = f64_col(&obs, "score");
    assert!(score.is_null(0) && score.is_null(1));
    assert_eq!(score.value(2), 3.0);
    assert_eq!(score.value(3), 4.0);
}

#[test]
fn rollback_undoes_the_import() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "t.scx");
    let csv = write_text(dir.path(), "calls.csv", "barcode,score\nAAAC-1,0.5\n");

    assert!(scx()
        .args([
            "obs-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap()
        ])
        .output()
        .unwrap()
        .status
        .success());
    assert!(read_obs(&scx_path).column_by_name("score").is_some());

    assert!(scx()
        .args(["rollback", scx_path.to_str().unwrap()])
        .output()
        .unwrap()
        .status
        .success());

    let obs = read_obs(&scx_path);
    assert!(obs.column_by_name("score").is_none());
    assert!(obs.column_by_name("barcode").is_some());
}

/// pandas `obs.to_csv()` and a tab-separated file both work without flags.
#[test]
fn pandas_and_tsv_shapes_import_without_extra_flags() {
    let dir = tempfile::tempdir().unwrap();

    let p1 = simple_fixture(dir.path(), "a.scx");
    let pandas = write_text(dir.path(), "pandas.csv", ",score\nAAAC-1,0.1\nAAAG-1,0.2\n");
    let out = scx()
        .args(["obs-import", p1.to_str().unwrap(), pandas.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("_index"));

    let p2 = simple_fixture(dir.path(), "b.scx");
    let tsv = write_text(dir.path(), "calls.tsv", "barcode\tscore\nAAAC-1\t0.1\n");
    let out = scx()
        .args(["obs-import", p2.to_str().unwrap(), tsv.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn column_selection_rename_and_prefix_apply() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "t.scx");
    let csv = write_text(
        dir.path(),
        "calls.csv",
        "barcode,score,call,extra\nAAAC-1,0.5,singlet,9\n",
    );

    let out = scx()
        .args([
            "obs-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--columns",
            "score,call",
            "--rename",
            "score=doublet_score",
            "--prefix",
            "scdbl_",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let obs = read_obs(&scx_path);
    assert!(obs.column_by_name("scdbl_doublet_score").is_some());
    assert!(obs.column_by_name("scdbl_call").is_some());
    assert!(
        obs.column_by_name("scdbl_extra").is_none(),
        "--columns must exclude unlisted columns"
    );
}
