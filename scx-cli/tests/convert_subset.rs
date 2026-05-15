//! Integration test for `scx subset --modality NAME` extracting a single
//! modality from a multimodal SCX file built via h5mu → SCX conversion.
//!
//! Lives here (not in scx-convert) because it crosses the boundary into
//! scx-cli's `subset` command, which has no library API. Spawns the scx
//! binary for both conversion and subset steps; the only thing this file
//! does directly is build the h5mu fixture via the `hdf5` crate.

#![cfg(feature = "hdf5")]

use std::path::Path;
use std::process::Command;

use hdf5::types::VarLenUnicode;
use scx_format::reader::ScxReader;

fn vlu(s: &str) -> VarLenUnicode {
    s.parse::<VarLenUnicode>().expect("valid VarLenUnicode")
}

fn create_test_h5mu(path: &Path, n_obs: usize, rna_n_vars: usize, adt_n_vars: usize) {
    let file = hdf5::File::create(path).unwrap();

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

    let mod_group = file.create_group("mod").unwrap();

    let write_modality = |group: &hdf5::Group, n_vars: usize| {
        let mut indptr = vec![0i64];
        let mut indices: Vec<i32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        for row in 0..n_obs {
            indices.push((row % n_vars) as i32);
            data.push((row + 1) as f32);
            indptr.push(data.len() as i64);
        }

        let x = group.create_group("X").unwrap();
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

        let var = group.create_group("var").unwrap();
        let var_index: Vec<VarLenUnicode> =
            (0..n_vars).map(|i| vlu(&format!("feat_{i}"))).collect();
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
    };

    let rna = mod_group.create_group("rna").unwrap();
    write_modality(&rna, rna_n_vars);
    let adt = mod_group.create_group("adt").unwrap();
    write_modality(&adt, adt_n_vars);
}

#[test]
fn subset_extract_modality_from_h5mu() {
    let dir = tempfile::tempdir().unwrap();
    let h5mu_in = dir.path().join("in.h5mu");
    let scx_in = dir.path().join("multi.scx");
    let scx_out = dir.path().join("rna_only.scx");
    create_test_h5mu(&h5mu_in, 10, 25, 6);

    let scx_bin = env!("CARGO_BIN_EXE_scx");

    let status = Command::new(scx_bin)
        .args([
            "convert",
            "--from",
            "h5mu",
            "--to",
            "scx",
            h5mu_in.to_str().unwrap(),
            scx_in.to_str().unwrap(),
        ])
        .status()
        .expect("scx convert failed to spawn");
    assert!(status.success(), "scx convert h5mu→scx exited {status}");

    let status = Command::new(scx_bin)
        .args([
            "subset",
            "--modality",
            "rna",
            "--output",
            scx_out.to_str().unwrap(),
            scx_in.to_str().unwrap(),
        ])
        .status()
        .expect("scx subset failed to spawn");
    assert!(
        status.success(),
        "scx subset --modality rna exited {status}"
    );

    let out = ScxReader::open(&scx_out).unwrap();
    assert!(
        !out.is_multimodal(),
        "extracted file should be single-modality"
    );
    assert_eq!(out.header().n_obs, 10);
    assert_eq!(
        out.header().n_vars,
        25,
        "rna's n_vars (not the file-wide max)"
    );
    let csr = out.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape, (10, 25));
}
