//! Integration tests for `scx convert --stream` and
//! `scx convert --stream --csc always`. Spawn the `scx` binary
//! against a synthetic h5ad fixture built with the `hdf5` crate.
//!
//! Lives in `scx-cli/tests/` because it exercises the CLI surface;
//! the underlying streaming pipeline is tested separately in
//! `scx-convert/src/tests.rs`.

#![cfg(feature = "hdf5")]

use std::path::Path;
use std::process::Command;

use hdf5::types::VarLenUnicode;
use scx_format::reader::ScxReader;
use scx_format::section::SectionType;

fn vlu(s: &str) -> VarLenUnicode {
    s.parse::<VarLenUnicode>().expect("valid VarLenUnicode")
}

/// Build a minimal CSR-on-disk h5ad fixture with `n_obs` cells and
/// `n_vars` genes. Each row gets two nonzeros at columns
/// `(row * 3) % n_vars` and `(row * 3 + 1) % n_vars`.
fn create_test_h5ad(path: &Path, n_obs: usize, n_vars: usize) {
    let file = hdf5::File::create(path).unwrap();

    // CSR triplet.
    let mut indptr = vec![0i64];
    let mut indices = Vec::<i32>::new();
    let mut data = Vec::<f32>::new();
    for row in 0..n_obs {
        let c0 = (row * 3) % n_vars;
        let c1 = (row * 3 + 1) % n_vars;
        let (a, b) = if c0 <= c1 { (c0, c1) } else { (c1, c0) };
        if a == b {
            indices.push(a as i32);
            data.push(((row + 1) * 7 % 200 + 1) as f32);
        } else {
            indices.push(a as i32);
            data.push(((row + 1) * 7 % 200 + 1) as f32);
            indices.push(b as i32);
            data.push(((row + 2) * 11 % 200 + 1) as f32);
        }
        indptr.push(data.len() as i64);
    }

    let x = file.create_group("X").unwrap();
    x.new_dataset::<i64>()
        .shape([indptr.len()])
        .create("indptr")
        .unwrap()
        .write(&indptr)
        .unwrap();
    x.new_dataset::<i32>()
        .shape([indices.len()])
        .create("indices")
        .unwrap()
        .write(&indices)
        .unwrap();
    x.new_dataset::<f32>()
        .shape([data.len()])
        .create("data")
        .unwrap()
        .write(&data)
        .unwrap();
    x.new_attr::<VarLenUnicode>()
        .create("encoding-type")
        .unwrap()
        .write_scalar(&vlu("csr_matrix"))
        .unwrap();
    x.new_attr::<i64>()
        .shape([2])
        .create("shape")
        .unwrap()
        .write(&[n_obs as i64, n_vars as i64])
        .unwrap();

    // Minimal obs / var with _index.
    let obs = file.create_group("obs").unwrap();
    let obs_index: Vec<VarLenUnicode> = (0..n_obs).map(|i| vlu(&format!("cell_{i}"))).collect();
    obs.new_dataset::<VarLenUnicode>()
        .shape([n_obs])
        .create("_index")
        .unwrap()
        .write(&obs_index)
        .unwrap();
    obs.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&vlu("_index"))
        .unwrap();

    let var = file.create_group("var").unwrap();
    let var_index: Vec<VarLenUnicode> = (0..n_vars).map(|i| vlu(&format!("gene_{i}"))).collect();
    var.new_dataset::<VarLenUnicode>()
        .shape([n_vars])
        .create("_index")
        .unwrap()
        .write(&var_index)
        .unwrap();
    var.new_attr::<VarLenUnicode>()
        .create("_index")
        .unwrap()
        .write_scalar(&vlu("_index"))
        .unwrap();
}

#[test]
fn convert_stream_h5ad_to_scx() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx_path = dir.path().join("out.scx");
    create_test_h5ad(&h5ad, 64, 12);

    let scx_bin = env!("CARGO_BIN_EXE_scx");
    let status = Command::new(scx_bin)
        .args([
            "convert",
            "--from",
            "h5ad",
            "--to",
            "scx",
            "--stream=true",
            h5ad.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .status()
        .expect("scx convert --stream failed to spawn");
    assert!(status.success(), "scx convert --stream exited {status}");

    let reader = ScxReader::open(&scx_path).unwrap();
    let hdr = reader.header();
    assert_eq!(hdr.n_obs, 64);
    assert_eq!(hdr.n_vars, 12);
    assert!(hdr.n_csr_shards >= 1, "expected at least one CSR shard");
    assert_eq!(hdr.n_csc_shards, 0, "no CSC sidecar without --csc always");
}

#[test]
fn convert_stream_csc_always_emits_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx_path = dir.path().join("out_csc.scx");
    create_test_h5ad(&h5ad, 64, 12);

    let scx_bin = env!("CARGO_BIN_EXE_scx");
    let status = Command::new(scx_bin)
        .args([
            "convert",
            "--from",
            "h5ad",
            "--to",
            "scx",
            "--stream",
            "--csc",
            "always",
            h5ad.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .status()
        .expect("scx convert --stream --csc always failed to spawn");
    assert!(
        status.success(),
        "scx convert --stream --csc always exited {status}"
    );

    let reader = ScxReader::open(&scx_path).unwrap();
    let hdr = reader.header();
    assert_eq!(hdr.n_obs, 64);
    assert_eq!(hdr.n_vars, 12);
    assert!(hdr.n_csr_shards >= 1, "expected at least one CSR shard");
    assert!(
        hdr.n_csc_shards >= 1,
        "expected at least one CSC shard after --csc always"
    );

    // Verify a CSC shard entry actually lives in the full catalog —
    // header counts alone could be stale.
    let csc_entries = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CscShard)
        .count();
    assert!(
        csc_entries >= 1,
        "expected CSC shard entries in catalog, got {csc_entries}"
    );
}

/// Attach a dense `/layers/{name}` group to an existing h5ad fixture.
/// `open_layer_streaming` rejects dense layers; this lets us exercise
/// the warn-and-skip path through the CLI surface.
fn attach_dense_layer(h5ad: &Path, layer_name: &str, n_obs: usize, n_vars: usize) {
    let file = hdf5::File::open_rw(h5ad).unwrap();
    let layers = match file.group("layers") {
        Ok(g) => g,
        Err(_) => file.create_group("layers").unwrap(),
    };
    let dense = layers.create_group(layer_name).unwrap();
    dense
        .new_attr::<VarLenUnicode>()
        .create("encoding-type")
        .unwrap()
        .write_scalar(&vlu("array"))
        .unwrap();
    dense
        .new_attr::<i64>()
        .shape([2])
        .create("shape")
        .unwrap()
        .write(&[n_obs as i64, n_vars as i64])
        .unwrap();
}

#[test]
fn convert_stream_skips_unreadable_layer() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("with_bad_layer.h5ad");
    let scx_path = dir.path().join("out.scx");
    create_test_h5ad(&h5ad, 32, 8);
    attach_dense_layer(&h5ad, "dense_bad", 32, 8);

    let scx_bin = env!("CARGO_BIN_EXE_scx");
    let status = Command::new(scx_bin)
        .args([
            "convert",
            "--from",
            "h5ad",
            "--to",
            "scx",
            "--stream=true",
            h5ad.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .status()
        .expect("scx convert --stream failed to spawn");
    assert!(
        status.success(),
        "scx convert --stream must skip the bad layer, not abort; exited {status}"
    );

    let reader = ScxReader::open(&scx_path).unwrap();
    let hdr = reader.header();
    assert_eq!(hdr.n_obs, 32);
    assert_eq!(hdr.n_vars, 8);
    assert!(hdr.n_csr_shards >= 1, "expected at least one CSR shard");
    let dense_count = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| {
            e.section_type == SectionType::LayerCsrShard && e.name.starts_with("dense_bad_shard_")
        })
        .count();
    assert_eq!(
        dense_count, 0,
        "the dense layer must be silently skipped, not emit shards"
    );
}

/// Attach a string `obs/cell_type` column (2-value palette) to an
/// existing h5ad fixture so predicate-index builds have something to
/// index. Mirrors the convert-layer `create_test_h5ad_with_cell_type`
/// fixture in `scx-convert/src/tests.rs`.
fn attach_cell_type_obs(h5ad: &Path, n_obs: usize) {
    let file = hdf5::File::open_rw(h5ad).unwrap();
    let obs = file.group("obs").unwrap();
    let labels = ["A", "B"];
    let col: Vec<VarLenUnicode> = (0..n_obs).map(|i| vlu(labels[i % labels.len()])).collect();
    obs.new_dataset::<VarLenUnicode>()
        .shape([n_obs])
        .create("cell_type")
        .unwrap()
        .write(&col)
        .unwrap();
}

#[test]
fn convert_index_obs_writes_predicate_index() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx_path = dir.path().join("out.scx");
    create_test_h5ad(&h5ad, 16, 8);
    attach_cell_type_obs(&h5ad, 16);

    let scx_bin = env!("CARGO_BIN_EXE_scx");
    let output = Command::new(scx_bin)
        .args([
            "convert",
            "--from",
            "h5ad",
            "--to",
            "scx",
            "--index-obs",
            "cell_type",
            h5ad.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .output()
        .expect("scx convert --index-obs failed to spawn");
    assert!(
        output.status.success(),
        "convert --index-obs exited non-zero: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let reader = ScxReader::open(&scx_path).unwrap();
    let bytes = reader.read_obs_predicate_index_bytes().unwrap();
    assert!(
        bytes.is_some(),
        "expected an obs predicate index section after --index-obs cell_type"
    );
}

#[test]
fn convert_index_obs_missing_column_fails() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx_path = dir.path().join("out.scx");
    create_test_h5ad(&h5ad, 16, 8);
    attach_cell_type_obs(&h5ad, 16);

    let scx_bin = env!("CARGO_BIN_EXE_scx");
    let output = Command::new(scx_bin)
        .args([
            "convert",
            "--from",
            "h5ad",
            "--to",
            "scx",
            "--index-obs",
            "nonexistent_column",
            h5ad.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .output()
        .expect("scx convert --index-obs failed to spawn");
    assert!(
        !output.status.success(),
        "a forced missing index column must fail the convert"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("nonexistent_column"),
        "expected the missing column name in the error; got: {stderr}"
    );
}

#[test]
fn convert_unknown_index_preset_fails() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx_path = dir.path().join("out.scx");
    create_test_h5ad(&h5ad, 16, 8);

    let scx_bin = env!("CARGO_BIN_EXE_scx");
    let output = Command::new(scx_bin)
        .args([
            "convert",
            "--from",
            "h5ad",
            "--to",
            "scx",
            "--index-preset",
            "bogus",
            h5ad.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .output()
        .expect("scx convert --index-preset failed to spawn");
    assert!(
        !output.status.success(),
        "an unknown --index-preset must fail the convert"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("preset"),
        "expected an unknown-preset error; got: {stderr}"
    );
}

#[test]
fn convert_index_obs_lenient_trailing_comma() {
    // The CSV parser drops empty tokens (see `parse_index_columns`), so
    // a trailing comma resolves to just `cell_type` and the convert
    // succeeds rather than erroring on an empty column name.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx_path = dir.path().join("out.scx");
    create_test_h5ad(&h5ad, 16, 8);
    attach_cell_type_obs(&h5ad, 16);

    let scx_bin = env!("CARGO_BIN_EXE_scx");
    let output = Command::new(scx_bin)
        .args([
            "convert",
            "--from",
            "h5ad",
            "--to",
            "scx",
            "--index-obs",
            "cell_type,",
            h5ad.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .output()
        .expect("scx convert --index-obs failed to spawn");
    assert!(
        output.status.success(),
        "a trailing comma should be dropped, not fail: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let reader = ScxReader::open(&scx_path).unwrap();
    assert!(
        reader.read_obs_predicate_index_bytes().unwrap().is_some(),
        "cell_type should still be indexed after dropping the empty token"
    );
}

#[test]
fn convert_index_flags_rejected_on_scx_to_h5ad() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx_path = dir.path().join("mid.scx");
    let out_h5ad = dir.path().join("out.h5ad");
    create_test_h5ad(&h5ad, 16, 8);
    attach_cell_type_obs(&h5ad, 16);

    let scx_bin = env!("CARGO_BIN_EXE_scx");
    // First produce an SCX file to export from.
    let status = Command::new(scx_bin)
        .args([
            "convert",
            "--from",
            "h5ad",
            "--to",
            "scx",
            h5ad.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .status()
        .expect("convert h5ad→scx failed to spawn");
    assert!(status.success(), "setup convert h5ad→scx exited {status}");

    // Index flags are meaningless on an export direction and must be
    // rejected up front rather than silently ignored.
    let output = Command::new(scx_bin)
        .args([
            "convert",
            "--to",
            "h5ad",
            "--index-obs",
            "cell_type",
            scx_path.to_str().unwrap(),
            out_h5ad.to_str().unwrap(),
        ])
        .output()
        .expect("scx convert failed to spawn");
    assert!(
        !output.status.success(),
        "index flags on scx→h5ad must be rejected"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--index-obs") && stderr.contains("scx_to_h5ad"),
        "expected an index-direction rejection mentioning the direction; got: {stderr}"
    );
}

#[test]
fn convert_index_flags_rejected_on_mtx_to_scx() {
    // The direction guard fires before any file I/O, so an empty input
    // directory is enough — the failure must be the index rejection,
    // not an MTX-parse error.
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("mtx_in");
    std::fs::create_dir(&mtx_dir).unwrap();
    let out = dir.path().join("out.scx");

    let scx_bin = env!("CARGO_BIN_EXE_scx");
    let output = Command::new(scx_bin)
        .args([
            "convert",
            "--from",
            "mtx",
            "--index-obs",
            "cell_type",
            mtx_dir.to_str().unwrap(),
            out.to_str().unwrap(),
        ])
        .output()
        .expect("scx convert failed to spawn");
    assert!(
        !output.status.success(),
        "index flags on mtx→scx must be rejected"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--index-obs") && stderr.contains("mtx_to_scx"),
        "expected an index-direction rejection mentioning the direction; got: {stderr}"
    );
}

#[test]
fn convert_stream_rejects_h5mu_direction() {
    // We don't need a valid h5mu file — `--from h5mu` selects the
    // direction before any input is opened, and the `--stream`
    // guard runs first.
    let dir = tempfile::tempdir().unwrap();
    let dummy = dir.path().join("in.h5mu");
    std::fs::write(&dummy, b"not actually h5mu").unwrap();
    let out = dir.path().join("out.scx");

    let scx_bin = env!("CARGO_BIN_EXE_scx");
    let output = Command::new(scx_bin)
        .args([
            "convert",
            "--from",
            "h5mu",
            "--to",
            "scx",
            "--stream",
            dummy.to_str().unwrap(),
            out.to_str().unwrap(),
        ])
        .output()
        .expect("scx convert --stream failed to spawn");
    assert!(
        !output.status.success(),
        "scx convert --from h5mu --stream should have failed"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--stream") && stderr.contains("h5mu"),
        "expected error mentioning --stream and h5mu, got: {stderr}"
    );
}
