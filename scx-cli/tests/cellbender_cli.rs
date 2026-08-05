//! End-to-end `scx cellbender-import`, driven through the real binary.
//!
//! Covers the full round trip a user actually performs: convert a 10x matrix
//! to SCX, import a CellBender output onto it, confirm the layer is visible and
//! structurally valid, then roll the whole thing back.

#![cfg(feature = "hdf5")]

use std::path::{Path, PathBuf};
use std::process::Command;

use hdf5::types::VarLenUnicode;

fn vlu(s: &str) -> VarLenUnicode {
    s.parse().unwrap()
}

fn scx() -> Command {
    Command::new(env!("CARGO_BIN_EXE_scx"))
}

fn write_strings(group: &hdf5::Group, name: &str, values: &[String]) {
    let v: Vec<VarLenUnicode> = values.iter().map(|s| vlu(s)).collect();
    group
        .new_dataset::<VarLenUnicode>()
        .shape([v.len()])
        .create(name)
        .unwrap()
        .write(&v)
        .unwrap();
}

/// A 10x CellRanger-v3 matrix with `shape` as a **dataset**, matching what real
/// CellRanger writes.
fn create_tenx(path: &Path, n_cells: usize, n_genes: usize) {
    let file = hdf5::File::create(path).unwrap();
    let matrix = file.create_group("matrix").unwrap();

    let mut indptr = vec![0i64];
    let (mut indices, mut data) = (Vec::<i32>::new(), Vec::<f32>::new());
    for row in 0..n_cells {
        indices.push((row % n_genes) as i32);
        data.push((row + 1) as f32);
        indptr.push(indices.len() as i64);
    }
    matrix
        .new_dataset::<i32>()
        .shape([2])
        .create("shape")
        .unwrap()
        .write(&[n_genes as i32, n_cells as i32])
        .unwrap();
    matrix
        .new_dataset::<i64>()
        .shape([indptr.len()])
        .create("indptr")
        .unwrap()
        .write(&indptr)
        .unwrap();
    matrix
        .new_dataset::<i32>()
        .shape([indices.len()])
        .create("indices")
        .unwrap()
        .write(&indices)
        .unwrap();
    matrix
        .new_dataset::<f32>()
        .shape([data.len()])
        .create("data")
        .unwrap()
        .write(&data)
        .unwrap();

    let barcodes: Vec<String> = (0..n_cells).map(|i| format!("cell_{i}")).collect();
    write_strings(&matrix, "barcodes", &barcodes);
    let features = matrix.create_group("features").unwrap();
    write_strings(
        &features,
        "id",
        &(0..n_genes).map(|i| format!("g{i}")).collect::<Vec<_>>(),
    );
    write_strings(
        &features,
        "name",
        &(0..n_genes).map(|i| format!("GENE{i}")).collect::<Vec<_>>(),
    );
}

/// A CellBender `remove-background` output over the same barcodes, but with the
/// rows **reversed** — the descending-UMI order the real filtered output uses.
fn create_cellbender(path: &Path, n_cells: usize, n_genes: usize) {
    let file = hdf5::File::create(path).unwrap();
    let matrix = file.create_group("matrix").unwrap();

    let barcodes: Vec<String> = (0..n_cells).rev().map(|i| format!("cell_{i}")).collect();
    let mut indptr = vec![0i64];
    let (mut indices, mut data) = (Vec::<i32>::new(), Vec::<f32>::new());
    for bc in &barcodes {
        let ordinal: usize = bc.strip_prefix("cell_").unwrap().parse().unwrap();
        indices.push((ordinal % n_genes) as i32);
        data.push((ordinal + 1) as f32);
        indptr.push(indices.len() as i64);
    }

    matrix
        .new_dataset::<i32>()
        .shape([2])
        .create("shape")
        .unwrap()
        .write(&[n_genes as i32, n_cells as i32])
        .unwrap();
    matrix
        .new_dataset::<i64>()
        .shape([indptr.len()])
        .create("indptr")
        .unwrap()
        .write(&indptr)
        .unwrap();
    matrix
        .new_dataset::<i32>()
        .shape([indices.len()])
        .create("indices")
        .unwrap()
        .write(&indices)
        .unwrap();
    matrix
        .new_dataset::<f32>()
        .shape([data.len()])
        .create("data")
        .unwrap()
        .write(&data)
        .unwrap();
    write_strings(&matrix, "barcodes", &barcodes);

    let features = matrix.create_group("features").unwrap();
    write_strings(
        &features,
        "id",
        &(0..n_genes).map(|i| format!("g{i}")).collect::<Vec<_>>(),
    );
    write_strings(
        &features,
        "name",
        &(0..n_genes).map(|i| format!("GENE{i}")).collect::<Vec<_>>(),
    );

    let dl = file.create_group("droplet_latents").unwrap();
    let probs: Vec<f32> = (0..n_cells).map(|i| 0.5 + 0.01 * i as f32).collect();
    dl.new_dataset::<f32>()
        .shape([n_cells])
        .create("cell_probability")
        .unwrap()
        .write(&probs)
        .unwrap();
    let idx: Vec<i64> = (0..n_cells as i64).collect();
    dl.new_dataset::<i64>()
        .shape([n_cells])
        .create("barcode_indices_for_latents")
        .unwrap()
        .write(&idx)
        .unwrap();

    let gl = file.create_group("global_latents").unwrap();
    let amb: Vec<f32> = (0..n_genes).map(|i| 0.01 * i as f32).collect();
    gl.new_dataset::<f32>()
        .shape([n_genes])
        .create("ambient_expression")
        .unwrap()
        .write(&amb)
        .unwrap();
}

fn setup(dir: &Path, n_cells: usize, n_genes: usize) -> (PathBuf, PathBuf) {
    let tenx = dir.join("raw.h5");
    let scx_path = dir.join("raw.scx");
    let cb = dir.join("cb_out.h5");
    create_tenx(&tenx, n_cells, n_genes);
    create_cellbender(&cb, n_cells, n_genes);

    let status = scx()
        .args([
            "convert",
            "--from",
            "10x",
            tenx.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success(), "10x -> scx convert failed");
    (scx_path, cb)
}

#[test]
fn cellbender_import_round_trip_then_rollback() {
    let dir = tempfile::tempdir().unwrap();
    let (scx_path, cb) = setup(dir.path(), 6, 3);

    let out = scx()
        .args([
            "cellbender-import",
            scx_path.to_str().unwrap(),
            cb.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "import failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("6/6 target rows matched"), "{stdout}");

    // `scx info` sees the layer.
    let info = scx()
        .args(["info", scx_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(info.status.success());
    assert!(
        String::from_utf8_lossy(&info.stdout).contains("cellbender"),
        "layer missing from `scx info`"
    );

    // The canonical-CSR gate on layer shards runs here.
    let validate = scx()
        .args(["validate", scx_path.to_str().unwrap(), "--deep"])
        .output()
        .unwrap();
    assert!(
        validate.status.success(),
        "validate --deep failed: {}",
        String::from_utf8_lossy(&validate.stdout)
    );

    // The layer landed on the right cells despite the reversed source order.
    let reader = scx_format_io::ScxReader::open(&scx_path).unwrap();
    let layer = reader.read_layer("cellbender").unwrap();
    assert_eq!(layer.shape, (6, 3));
    for row in 0..6usize {
        let (s, e) = (layer.indptr[row] as usize, layer.indptr[row + 1] as usize);
        assert_eq!(
            &layer.data[s..e],
            &[(row + 1) as f32],
            "row {row} got another cell's counts"
        );
    }
    drop(reader);

    let rollback = scx()
        .args(["rollback", scx_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        rollback.status.success(),
        "rollback failed: {}",
        String::from_utf8_lossy(&rollback.stderr)
    );
    let reader = scx_format_io::ScxReader::open(&scx_path).unwrap();
    assert!(
        reader.layer_names().is_empty(),
        "the layer survived a rollback"
    );
}

#[test]
fn dry_run_reports_the_join_and_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (scx_path, cb) = setup(dir.path(), 5, 2);
    let before = std::fs::read(&scx_path).unwrap();

    let out = scx()
        .args([
            "cellbender-import",
            scx_path.to_str().unwrap(),
            cb.to_str().unwrap(),
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("5/5 target rows matched"), "{stdout}");
    assert!(stdout.contains("nothing written"), "{stdout}");

    assert_eq!(
        std::fs::read(&scx_path).unwrap(),
        before,
        "a dry run must leave the file byte-identical"
    );
}

#[test]
fn convert_redirects_a_cellbender_file_to_the_import_command() {
    let dir = tempfile::tempdir().unwrap();
    let (_scx_path, cb) = setup(dir.path(), 4, 2);
    let out_path = dir.path().join("nope.scx");

    let out = scx()
        .args([
            "convert",
            "--from",
            "10x",
            cb.to_str().unwrap(),
            out_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("CellBender"), "{stderr}");
    assert!(
        stderr.contains("cellbender-import"),
        "must redirect: {stderr}"
    );
}

#[test]
fn reimport_requires_overwrite() {
    let dir = tempfile::tempdir().unwrap();
    let (scx_path, cb) = setup(dir.path(), 4, 2);
    let args = [
        "cellbender-import",
        scx_path.to_str().unwrap(),
        cb.to_str().unwrap(),
    ];

    assert!(scx().args(args).status().unwrap().success());

    let out = scx().args(args).output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("already exists"));

    let out = scx().args(args).arg("--overwrite").output().unwrap();
    assert!(
        out.status.success(),
        "overwrite failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Exactly one layer, at the target's height — not a doubled shard family.
    let reader = scx_format_io::ScxReader::open(&scx_path).unwrap();
    assert_eq!(reader.layer_names(), vec!["cellbender".to_string()]);
    assert_eq!(reader.read_layer("cellbender").unwrap().shape, (4, 2));
}
