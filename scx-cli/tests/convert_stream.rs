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
use scx_format_io::reader::ScxReader;
use scx_format_io::section::SectionType;

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

/// Regression: the streaming converter must name the single-modality X
/// CSR section `X_shard_{idx}` (uppercase). A lowercase `x_shard_0` is
/// tolerated by the reader (it resolves shards by `SectionType`, not name)
/// but makes `scx explode` / `scx push` reject every converted file — the
/// entire cloud-publish path. Asserting the on-disk catalog name catches
/// the regression without needing the `cloud` feature.
#[test]
fn convert_stream_names_x_shard_uppercase() {
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
    let names: Vec<&str> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CsrShard)
        .map(|e| e.name.as_str())
        .collect();
    assert!(!names.is_empty(), "expected at least one CSR shard");
    for n in &names {
        assert!(
            n.starts_with("X_shard_"),
            "single-modality CSR shard must be named X_shard_N (uppercase), got {n:?}"
        );
    }
}

/// Regression (end-to-end): a streamed-convert output must `explode`
/// cleanly into an `.scxd` directory with an `X/NNNNNN.shard` payload.
/// Before the fix the converter emitted `x_shard_0` and explode/push
/// rejected it with "invalid CsrShard name". Gated on `cloud` because
/// `scx explode` only exists in a cloud build.
#[cfg(feature = "cloud")]
#[test]
fn convert_stream_output_explodes() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx_path = dir.path().join("out.scx");
    let scxd_path = dir.path().join("out.scxd");
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

    let status = Command::new(scx_bin)
        .args([
            "explode",
            scx_path.to_str().unwrap(),
            scxd_path.to_str().unwrap(),
        ])
        .status()
        .expect("scx explode failed to spawn");
    assert!(
        status.success(),
        "scx explode of a streamed-convert output exited {status}"
    );
    assert!(
        scxd_path.join("X/000000.shard").is_file(),
        "explode must emit X/000000.shard, dir contents: {:?}",
        std::fs::read_dir(scxd_path.join("X")).map(|rd| rd
            .filter_map(|e| e.ok().map(|e| e.file_name()))
            .collect::<Vec<_>>())
    );
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
/// A minimal 10x CellRanger-v3 matrix (`shape` as a dataset, as real
/// CellRanger writes it). Mirrors `cellbender_cli.rs::create_tenx`; kept local
/// so this file has no cross-test-binary dependency.
fn create_test_tenx(path: &Path, n_cells: usize, n_genes: usize) {
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

    let file = hdf5::File::create(path).unwrap();
    let matrix = file.create_group("matrix").unwrap();

    // CSC on disk: 10x stores genes × cells, one nonzero per cell.
    let mut indptr = vec![0i64];
    let (mut indices, mut data) = (Vec::<i32>::new(), Vec::<f32>::new());
    for cell in 0..n_cells {
        indices.push((cell % n_genes) as i32);
        data.push((cell + 1) as f32);
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
fn bare_stream_flag_does_not_swallow_the_input_path() {
    // Replaces a former `convert_stream_rejects_h5mu_direction`, whose premise
    // was false — `h5mu_to_scx` *is* an allowed streaming direction, so the
    // direction guard could never be what failed it. It passed only because
    // clap's value-swallow error happened to contain both "--stream" and
    // ".h5mu".
    //
    // That swallow is the real bug: `--stream` declared `num_args = 0..=1`
    // without `require_equals` consumed the following positional, so
    // `--stream <input>` died with "invalid value '<input>' for '--stream'".
    // It escaped notice because every other call site puts `--stream` last or
    // before another flag. Assert the natural flags-before-positionals order
    // now works.
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx_path = dir.path().join("out.scx");
    create_test_h5ad(&h5ad, 16, 4);

    let scx_bin = env!("CARGO_BIN_EXE_scx");
    let output = Command::new(scx_bin)
        .args([
            "convert",
            "--from",
            "h5ad",
            "--stream",
            h5ad.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .output()
        .expect("scx convert --stream failed to spawn");
    assert!(
        output.status.success(),
        "a bare --stream before the positionals must not consume the input path; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(ScxReader::open(&scx_path).unwrap().header().n_obs, 16);
}

#[test]
fn convert_tenx_to_scx_succeeds_with_no_flags() {
    // 10x → SCX is a headline ingest path and had no default-flag CLI test:
    // the only tests that reached it hard-coded the `--stream=false`
    // workaround for the flag's rejected default.
    let dir = tempfile::tempdir().unwrap();
    let tenx = dir.path().join("raw.h5");
    let scx_path = dir.path().join("out.scx");
    create_test_tenx(&tenx, 8, 3);

    let scx_bin = env!("CARGO_BIN_EXE_scx");
    let output = Command::new(scx_bin)
        .args([
            "convert",
            "--from",
            "10x",
            tenx.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .output()
        .expect("scx convert failed to spawn");
    assert!(
        output.status.success(),
        "10x → scx must work with no flags; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let reader = ScxReader::open(&scx_path).unwrap();
    let hdr = reader.header();
    assert_eq!(hdr.n_obs, 8);
    assert_eq!(hdr.n_vars, 3);
}

#[test]
fn convert_stream_true_rejected_on_tenx() {
    // Resolution precedes file I/O, so a nonexistent input is enough.
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("absent.h5");
    let out = dir.path().join("out.scx");

    let scx_bin = env!("CARGO_BIN_EXE_scx");
    let output = Command::new(scx_bin)
        .args([
            "convert",
            "--from",
            "10x",
            "--stream",
            missing.to_str().unwrap(),
            out.to_str().unwrap(),
        ])
        .output()
        .expect("scx convert failed to spawn");
    assert!(
        !output.status.success(),
        "an explicit --stream on 10x → scx must be rejected"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--stream") && stderr.contains("tenx_to_scx"),
        "expected a --stream rejection naming the direction; got: {stderr}"
    );
    assert!(
        stderr.contains("Re-run without"),
        "the message must name the remedy; got: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// `--min-counts`: streaming low-UMI pre-trim on the export direction.
// ---------------------------------------------------------------------------

/// Per-row totals of `create_test_h5ad`'s deterministic fixture, so the
/// expected kept count is computed rather than hard-coded.
fn fixture_row_sums(n_obs: usize, n_vars: usize) -> Vec<f64> {
    (0..n_obs)
        .map(|row| {
            let c0 = (row * 3) % n_vars;
            let c1 = (row * 3 + 1) % n_vars;
            let v0 = ((row + 1) * 7 % 200 + 1) as f64;
            if c0 == c1 {
                v0
            } else {
                v0 + ((row + 2) * 11 % 200 + 1) as f64
            }
        })
        .collect()
}

fn h5ad_to_scx_fixture(dir: &Path, n_obs: usize, n_vars: usize) -> std::path::PathBuf {
    let h5ad = dir.join("in.h5ad");
    let scx_path = dir.join("in.scx");
    create_test_h5ad(&h5ad, n_obs, n_vars);
    let status = Command::new(env!("CARGO_BIN_EXE_scx"))
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
        .unwrap();
    assert!(status.success());
    scx_path
}

#[test]
fn convert_min_counts_filters_rows_on_export() {
    let dir = tempfile::tempdir().unwrap();
    let (n_obs, n_vars) = (64usize, 12usize);
    let scx_path = h5ad_to_scx_fixture(dir.path(), n_obs, n_vars);
    let out = dir.path().join("out.h5ad");

    let threshold = 150.0;
    let expected = fixture_row_sums(n_obs, n_vars)
        .iter()
        .filter(|&&s| s >= threshold)
        .count();
    assert!(
        expected > 0 && expected < n_obs,
        "threshold must actually bite: kept {expected} of {n_obs}"
    );

    let status = Command::new(env!("CARGO_BIN_EXE_scx"))
        .args([
            "convert",
            "--to",
            "h5ad",
            "--min-counts",
            &threshold.to_string(),
            scx_path.to_str().unwrap(),
            out.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success(), "scx convert --min-counts exited {status}");

    let f = hdf5::File::open(&out).unwrap();
    let shape: Vec<i64> = f
        .group("X")
        .unwrap()
        .attr("shape")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(shape, vec![expected as i64, n_vars as i64]);
}

/// `--min-counts` on an ingest direction must fail loudly rather than being
/// silently ignored (the option lives on the shared `ConvertOptions`).
#[test]
fn convert_min_counts_rejected_on_import_direction() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx_path = dir.path().join("out.scx");
    create_test_h5ad(&h5ad, 16, 8);

    let out = Command::new(env!("CARGO_BIN_EXE_scx"))
        .args([
            "convert",
            "--from",
            "h5ad",
            "--to",
            "scx",
            "--min-counts",
            "5",
            h5ad.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("--min-counts"), "{stderr}");
    assert!(stderr.contains("h5ad_to_scx"), "{stderr}");
}

#[test]
fn convert_min_counts_rejects_stream_false() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = h5ad_to_scx_fixture(dir.path(), 16, 8);
    let out_path = dir.path().join("out.h5ad");

    let out = Command::new(env!("CARGO_BIN_EXE_scx"))
        .args([
            "convert",
            "--to",
            "h5ad",
            "--stream=false",
            "--min-counts",
            "5",
            scx_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("--stream"), "{stderr}");
}

#[test]
fn convert_min_counts_zero_rows_errors() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = h5ad_to_scx_fixture(dir.path(), 16, 8);
    let out_path = dir.path().join("out.h5ad");

    let out = Command::new(env!("CARGO_BIN_EXE_scx"))
        .args([
            "convert",
            "--to",
            "h5ad",
            "--min-counts",
            "1000000",
            scx_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("keeps zero"), "{stderr}");
}

#[test]
fn convert_min_counts_negative_errors() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = h5ad_to_scx_fixture(dir.path(), 16, 8);
    let out_path = dir.path().join("out.h5ad");

    let out = Command::new(env!("CARGO_BIN_EXE_scx"))
        .args([
            "convert",
            "--to",
            "h5ad",
            // `=` form: clap treats a bare `-1` token as a flag, so this is
            // the only spelling that reaches the value parser.
            "--min-counts=-1",
            scx_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("non-negative"), "{stderr}");
}

// ---------------------------------------------------------------------------
// `scx convert --shard-obs` (organization phase 6c, ORG-11.16-4)
// ---------------------------------------------------------------------------

/// Run `scx convert` on a fixture and return the output's obs shard count.
fn convert_and_count_obs_shards(extra: &[&str], n_obs: usize, shard_size: &str) -> usize {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx_path = dir.path().join("out.scx");
    create_test_h5ad(&h5ad, n_obs, 12);

    let mut args: Vec<&str> = vec![
        "convert",
        "--from",
        "h5ad",
        "--to",
        "scx",
        "--shard-size",
        shard_size,
    ];
    args.extend_from_slice(extra);
    let h5ad_s = h5ad.to_str().unwrap().to_string();
    let scx_s = scx_path.to_str().unwrap().to_string();
    args.push(&h5ad_s);
    args.push(&scx_s);

    let status = Command::new(env!("CARGO_BIN_EXE_scx"))
        .args(&args)
        .status()
        .expect("scx convert failed to spawn");
    assert!(status.success(), "scx convert exited {status} for {args:?}");

    let reader = ScxReader::open(&scx_path).unwrap();
    assert_eq!(reader.header().n_obs, n_obs as u64);
    reader.obs_metadata_shard_count()
}

/// `--shard-obs` defaults to `auto` on `convert`, exactly as it does on
/// `optimize` — a converted file above the threshold carries sharded obs
/// without the operator asking, which is what puts it on the streaming h5ad
/// export path.
#[test]
fn convert_shard_obs_defaults_to_auto() {
    assert_eq!(convert_and_count_obs_shards(&[], 64, "16"), 4);
}

#[test]
fn convert_shard_obs_off_keeps_a_single_section() {
    assert_eq!(
        convert_and_count_obs_shards(&["--shard-obs", "off"], 64, "16"),
        0
    );
}

/// Below the `auto` threshold `always` still shards — otherwise the flag
/// would be indistinguishable from `auto` on every small file.
#[test]
fn convert_shard_obs_always_shards_below_the_threshold() {
    assert_eq!(
        convert_and_count_obs_shards(&["--shard-obs", "always"], 8, "1024"),
        1
    );
    assert_eq!(convert_and_count_obs_shards(&[], 8, "1024"), 0);
}

/// clap rejects an unknown value before any file is touched, so a typo can
/// never be read as "the default".
#[test]
fn convert_rejects_an_unknown_shard_obs_value() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    create_test_h5ad(&h5ad, 8, 4);
    let out = dir.path().join("out.scx");

    let output = Command::new(env!("CARGO_BIN_EXE_scx"))
        .args([
            "convert",
            "--from",
            "h5ad",
            "--shard-obs",
            "sometimes",
            h5ad.to_str().unwrap(),
            out.to_str().unwrap(),
        ])
        .output()
        .expect("scx convert failed to spawn");
    assert!(!output.status.success(), "expected a rejection");
    assert!(
        !out.exists(),
        "a rejected convert must not leave an output file"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("shard-obs"),
        "error should name the flag, got: {stderr}"
    );
}
