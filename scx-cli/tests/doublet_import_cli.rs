//! End-to-end `scx doublet-import`, driven through the real binary.
//!
//! Ungated, like `obs_import_cli.rs`: the fixture is built directly with
//! `ScxWriter` rather than converted from h5ad, so the whole doublet path is
//! proven to work in a build with no libhdf5.
//!
//! What this file is for, over and above the unit tests in
//! `scx-convert/src/doublet_tests.rs`, is the *seam*: that `--tool` reaches the
//! profile table, that the canonical columns survive the write and read back
//! with the right values, and that the command refuses loudly rather than
//! writing something plausible and wrong.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, Float64Array, RecordBatch, StringArray};
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

/// Read a numeric column as f64. Casts rather than downcasts, so the assertion
/// is about values rather than about which width the CSV happened to imply.
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

fn bool_col(batch: &RecordBatch, name: &str) -> BooleanArray {
    batch
        .column_by_name(name)
        .unwrap_or_else(|| panic!("column '{name}' missing"))
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap_or_else(|| panic!("column '{name}' is not boolean"))
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

// ---------------------------------------------------------------------------
// Happy path
// ---------------------------------------------------------------------------

#[test]
fn scdblfinder_output_lands_as_canonical_columns() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "t.scx");
    // Exactly the shape `write.csv(as.data.frame(colData(sce)[, ...]))` writes:
    // quoted header, unnamed index column, bare NA. Out of order and covering
    // only three of four cells, so a positional import would be visibly wrong.
    let csv = write_text(
        dir.path(),
        "sce.csv",
        "\"\",\"scDblFinder.score\",\"scDblFinder.class\",\"scDblFinder.mostLikelyOrigin\"\n\
         \"AAAT-1\",0.30,\"singlet\",NA\n\
         \"AAAC-1\",0.10,\"singlet\",NA\n\
         \"AAAG-1\",0.90,\"doublet\",\"1+2\"\n",
    );

    let out = scx()
        .args([
            "doublet-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--tool",
            "scdblfinder",
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

    // File order, not CSV order.
    let score = f64_col(&obs, "scdblfinder_score");
    assert!((score.value(0) - 0.10).abs() < 1e-6);
    assert!((score.value(1) - 0.90).abs() < 1e-6);
    assert!((score.value(2) - 0.30).abs() < 1e-6);
    assert!(score.is_null(3), "the uncovered cell must be null, not 0.0");

    // The derivation, which is the whole reason this wrapper exists.
    let called = bool_col(&obs, "scdblfinder_predicted");
    assert!(!called.value(0));
    assert!(called.value(1));
    assert!(!called.value(2));
    assert!(called.is_null(3));

    assert_eq!(str_col(&obs, "scdblfinder_status").value(3), "absent");
    // Every other column survives under the tool prefix, unchanged.
    assert!(obs
        .column_by_name("scdblfinder_scDblFinder.mostLikelyOrigin")
        .is_some());
}

#[test]
fn scrublet_output_lands_as_canonical_columns() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "t.scx");
    // `adata.obs[["doublet_score","predicted_doublet"]].to_csv(path)`.
    let csv = write_text(
        dir.path(),
        "scrublet.csv",
        ",doublet_score,predicted_doublet\n\
         AAAC-1,0.03,False\n\
         AAAG-1,0.71,True\n\
         AAAT-1,0.05,False\n\
         AAAA-1,0.02,False\n",
    );

    let out = scx()
        .args([
            "doublet-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--tool",
            "scrublet",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let obs = read_obs(&scx_path);
    let called = bool_col(&obs, "scrublet_predicted");
    assert_eq!(
        (0..4).map(|i| called.value(i)).collect::<Vec<_>>(),
        [false, true, false, false]
    );
    assert!((f64_col(&obs, "scrublet_score").value(1) - 0.71).abs() < 1e-6);
}

#[test]
fn two_tools_coexist_under_different_prefixes() {
    // The reason `key_added` defaults to the tool name: importing a second
    // caller must not collide with the first, so consensus across tools is a
    // matter of reading two columns rather than re-running anything.
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "t.scx");
    let a = write_text(
        dir.path(),
        "a.csv",
        "barcode,scDblFinder.score,scDblFinder.class\n\
         AAAC-1,0.1,singlet\nAAAG-1,0.9,doublet\nAAAT-1,0.2,singlet\nAAAA-1,0.3,singlet\n",
    );
    let b = write_text(
        dir.path(),
        "b.csv",
        "barcode,doublet_score,predicted_doublet\n\
         AAAC-1,0.05,False\nAAAG-1,0.80,True\nAAAT-1,0.10,False\nAAAA-1,0.15,False\n",
    );

    for (table, tool) in [(&a, "scdblfinder"), (&b, "scrublet")] {
        let out = scx()
            .args([
                "doublet-import",
                scx_path.to_str().unwrap(),
                table.to_str().unwrap(),
                "--tool",
                tool,
            ])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{tool}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    let obs = read_obs(&scx_path);
    assert!(bool_col(&obs, "scdblfinder_predicted").value(1));
    assert!(bool_col(&obs, "scrublet_predicted").value(1));
    assert_eq!(str_col(&obs, "scdblfinder_status").value(0), "present");
    assert_eq!(str_col(&obs, "scrublet_status").value(0), "present");
}

#[test]
fn scds_import_writes_no_call_column_and_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "t.scx");
    let csv = write_text(
        dir.path(),
        "scds.csv",
        "barcode,cxds_score,bcds_score,hybrid_score\n\
         AAAC-1,0.4,0.2,0.31\nAAAG-1,0.9,0.8,0.85\nAAAT-1,0.1,0.1,0.10\nAAAA-1,0.2,0.3,0.25\n",
    );

    let out = scx()
        .args([
            "doublet-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--tool",
            "scds",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    // The absence is announced, not left to be discovered.
    assert!(stdout.contains("no call column"), "{stdout}");

    let obs = read_obs(&scx_path);
    assert!(obs.column_by_name("scds_score").is_some());
    assert!(
        obs.column_by_name("scds_predicted").is_none(),
        "a call must never be invented by thresholding"
    );
    // The two unused score flavours are still carried across.
    assert!(obs.column_by_name("scds_cxds_score").is_some());
}

// ---------------------------------------------------------------------------
// Refusals and safety
// ---------------------------------------------------------------------------

#[test]
fn dry_run_leaves_the_file_byte_identical_and_diagnoses_the_key() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "t.scx");
    let csv = write_text(
        dir.path(),
        "sce.csv",
        "barcode,scDblFinder.score,scDblFinder.class\nAAAC-1,0.5,doublet\n",
    );
    let before = std::fs::read(&scx_path).unwrap();

    let out = scx()
        .args([
            "doublet-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--tool",
            "scdblfinder",
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
fn uniqueness_preflight_suggests_a_working_key() {
    // The Phase-0 atlas shape in miniature: the auto-resolved key is
    // duplicated, and exactly one column would work. Without the diagnosis the
    // user has to guess; with it, the fix is in the error.
    let dir = tempfile::tempdir().unwrap();
    let obs = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("barcode", DataType::Utf8, false),
            Field::new("cell_uid", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec![
                "AAAC-1", "AAAC-1", "AAAG-1", "AAAG-1",
            ])),
            Arc::new(StringArray::from(vec!["u0", "u1", "u2", "u3"])),
        ],
    )
    .unwrap();
    let scx_path = write_fixture(dir.path(), "dup.scx", obs, 2);
    let csv = write_text(
        dir.path(),
        "sce.csv",
        "barcode,scDblFinder.score,scDblFinder.class\nAAAC-1,0.5,doublet\n",
    );

    let out = scx()
        .args([
            "doublet-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--tool",
            "scdblfinder",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success(), "a duplicated key must refuse");
    let msg = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(msg.contains("cell_uid"), "{msg}");

    // And the suggested key works.
    let out = scx()
        .args([
            "doublet-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--tool",
            "scdblfinder",
            "--key",
            "cell_uid",
        ])
        .output()
        .unwrap();
    // The source table has no `cell_uid` column, so this fails on the *source*
    // side — which is itself the right failure, and names the columns present.
    let msg = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(msg.contains("cell_uid") && msg.contains("barcode"), "{msg}");
}

#[test]
fn an_unknown_tool_exits_non_zero_listing_the_valid_set() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "t.scx");
    let csv = write_text(dir.path(), "x.csv", "barcode,doublet_score\nAAAC-1,0.5\n");

    let out = scx()
        .args([
            "doublet-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--tool",
            "scDblFinderr",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let msg = String::from_utf8_lossy(&out.stderr);
    assert!(msg.contains("scdblfinder"), "{msg}");
    assert!(msg.contains("doubletdetection"), "{msg}");
}

#[test]
fn a_wrong_profile_refuses_naming_the_columns_present() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "t.scx");
    // A scrublet file handed to the scDblFinder profile.
    let csv = write_text(
        dir.path(),
        "scrublet.csv",
        "barcode,doublet_score,predicted_doublet\nAAAC-1,0.03,False\n",
    );
    let before = std::fs::read(&scx_path).unwrap();

    let out = scx()
        .args([
            "doublet-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--tool",
            "scdblfinder",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let msg = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(msg.contains("scDblFinder.score"), "{msg}");
    assert!(msg.contains("doublet_score"), "{msg}");
    assert_eq!(
        before,
        std::fs::read(&scx_path).unwrap(),
        "a rejected import must leave the file byte-identical"
    );
}

/// Only meaningful in a build with no HDF5: there, an `.h5ad` source is refused
/// with the "write a CSV instead" hint. With `--features hdf5` the extension no
/// longer decides — the file's contents do — so this fixture (which is not HDF5
/// at all) fails as an unreadable file instead. The unit-test sibling in
/// `scx-convert/src/doublet_tests.rs` has always been gated this way; this test
/// was not, so `cargo test -p scx-cli --features hdf5` failed on it.
#[cfg(not(feature = "hdf5"))]
#[test]
fn an_h5ad_source_is_refused_with_the_convert_hint() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "t.scx");
    let h5 = write_text(dir.path(), "scrublet_out.h5ad", "not really hdf5");

    let out = scx()
        .args([
            "doublet-import",
            scx_path.to_str().unwrap(),
            h5.to_str().unwrap(),
            "--tool",
            "scrublet",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let msg = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(msg.contains("to_csv"), "{msg}");
}

/// The `--features hdf5` counterpart: a file named `.h5ad` that is not HDF5
/// fails as an unreadable file, naming the format it tried.
#[cfg(feature = "hdf5")]
#[test]
fn a_non_hdf5_h5ad_source_fails_as_an_unreadable_file() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "t.scx");
    let h5 = write_text(dir.path(), "scrublet_out.h5ad", "not really hdf5");

    let out = scx()
        .args([
            "doublet-import",
            scx_path.to_str().unwrap(),
            h5.to_str().unwrap(),
            "--tool",
            "scrublet",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let msg = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(msg.contains("as HDF5"), "{msg}");
}

#[test]
fn a_declared_call_column_absent_warns_and_still_imports() {
    // B3: a scrublet-shaped table imported as doubletdetection. Exit 0 (a
    // score-only import is valid), but the two-case note must say WHICH case.
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "t.scx");
    let table = write_text(
        dir.path(),
        "dd.csv",
        "barcode,doublet_score,predicted_doublet\n\
         AAAC-1,0.10,True\nAAAG-1,0.20,False\n\
         AAAT-1,0.30,False\nAAAA-1,0.40,False\n",
    );

    let out = scx()
        .args([
            "doublet-import",
            scx_path.to_str().unwrap(),
            table.to_str().unwrap(),
            "--tool",
            "doubletdetection",
        ])
        .output()
        .unwrap();
    assert!(out.status.success(), "score-only import must still succeed");
    let msg = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        msg.contains("doublet_label"),
        "names what it expected: {msg}"
    );
    assert!(msg.contains("--call-column"), "names the remedy: {msg}");
    assert!(
        msg.contains("predicted_doublet"),
        "names what the table had: {msg}"
    );
    // Must NOT claim the by-design wording that belongs to scds.
    assert!(
        !msg.contains("will not choose a cutoff"),
        "that note is for a profile with no call column: {msg}"
    );
}

#[test]
fn rollback_undoes_a_doublet_import() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = simple_fixture(dir.path(), "t.scx");
    let csv = write_text(
        dir.path(),
        "sce.csv",
        "barcode,scDblFinder.score,scDblFinder.class\nAAAC-1,0.5,doublet\n",
    );

    let out = scx()
        .args([
            "doublet-import",
            scx_path.to_str().unwrap(),
            csv.to_str().unwrap(),
            "--tool",
            "scdblfinder",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(read_obs(&scx_path)
        .column_by_name("scdblfinder_score")
        .is_some());

    let out = scx()
        .args(["rollback", scx_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let obs = read_obs(&scx_path);
    assert!(obs.column_by_name("scdblfinder_score").is_none());
    assert!(obs.column_by_name("scdblfinder_status").is_none());
}
