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
            "--stream",
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
