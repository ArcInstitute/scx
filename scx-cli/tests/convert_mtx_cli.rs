//! Integration tests for the `scx convert` MTX directions through the real
//! CLI argument parser.
//!
//! These live here rather than in `convert_stream.rs` because that file is
//! `#![cfg(feature = "hdf5")]`-gated, while MTX ↔ SCX needs no libhdf5 — so
//! this file runs on the default feature set.
//!
//! The regression they guard: `--stream` used to carry a clap default of
//! `true`, and the direction guard rejected on that default, so
//! `mtx_to_scx` / `scx_to_mtx` / `tenx_to_scx` were unreachable without
//! discovering `--stream=false`. Every test below therefore passes **no**
//! flags beyond `--from` / `--to` unless it is specifically about `--stream`.

use std::process::Command;

/// A Cell Ranger–style MTX directory, written uncompressed.
///
/// The reader accepts both `matrix.mtx` and `matrix.mtx.gz` (see its
/// "none of [matrix.mtx.gz, matrix.mtx] found" error), and the plain form
/// keeps this fixture dependency-free.
///
/// Cell Ranger writes `matrix.mtx` as **features × barcodes**, so this is a
/// 4-genes × 3-cells layout that the reader transposes to 3 cells × 4 genes.
fn create_mtx_dir(dir: &std::path::Path) {
    std::fs::create_dir_all(dir).unwrap();
    let mtx = "\
%%MatrixMarket matrix coordinate integer general
4 3 5
1 2 3
2 1 1
2 3 4
3 3 5
4 1 2
";
    std::fs::write(dir.join("matrix.mtx"), mtx).unwrap();
    std::fs::write(
        dir.join("barcodes.tsv"),
        "AAACCCAA-1\nBBBDDDBB-1\nCCCEEECC-1\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("features.tsv"),
        "ENSG001\tGeneA\tGene Expression\n\
         ENSG002\tGeneB\tGene Expression\n\
         ENSG003\tGeneC\tGene Expression\n\
         ENSG004\tGeneD\tGene Expression\n",
    )
    .unwrap();
}

fn scx() -> Command {
    Command::new(env!("CARGO_BIN_EXE_scx"))
}

// ---------------------------------------------------------------------------
// The directions must work with no flags at all.
// ---------------------------------------------------------------------------

#[test]
fn convert_mtx_to_scx_succeeds_with_no_flags() {
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("mtx_in");
    create_mtx_dir(&mtx_dir);
    let out = dir.path().join("out.scx");

    let output = scx()
        .args([
            "convert",
            "--from",
            "mtx",
            mtx_dir.to_str().unwrap(),
            out.to_str().unwrap(),
        ])
        .output()
        .expect("scx convert failed to spawn");
    assert!(
        output.status.success(),
        "mtx → scx must work with no flags; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let reader = scx_format_io::reader::ScxReader::open(&out).unwrap();
    assert_eq!(reader.header().n_obs, 3, "3 barcodes → 3 obs rows");
    assert_eq!(reader.header().n_vars, 4, "4 features → 4 var columns");
}

#[test]
fn convert_scx_to_mtx_succeeds_with_no_flags() {
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("mtx_in");
    create_mtx_dir(&mtx_dir);
    let scx_path = dir.path().join("mid.scx");

    let status = scx()
        .args([
            "convert",
            "--from",
            "mtx",
            mtx_dir.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .status()
        .expect("scx convert failed to spawn");
    assert!(status.success(), "setup: mtx → scx");

    let out_dir = dir.path().join("mtx_out");
    let output = scx()
        .args([
            "convert",
            "--to",
            "mtx",
            scx_path.to_str().unwrap(),
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("scx convert failed to spawn");
    assert!(
        output.status.success(),
        "scx → mtx must work with no flags; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    for name in ["matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz"] {
        assert!(out_dir.join(name).is_file(), "missing {name} in the export");
    }
}

// ---------------------------------------------------------------------------
// `--stream` semantics on a direction that has no streaming path.
// ---------------------------------------------------------------------------

#[test]
fn convert_stream_true_rejected_on_mtx_directions() {
    // The resolution happens before any file I/O, so an empty input
    // directory is enough — the failure must be the `--stream` rejection,
    // not an MTX-parse error.
    let dir = tempfile::tempdir().unwrap();
    let empty = dir.path().join("empty_in");
    std::fs::create_dir(&empty).unwrap();

    for (flag_args, direction) in [
        (["--from", "mtx"], "mtx_to_scx"),
        (["--to", "mtx"], "scx_to_mtx"),
    ] {
        let out = dir.path().join(format!("out_{direction}"));
        let output = scx()
            .args(["convert"])
            .args(flag_args)
            .args(["--stream", empty.to_str().unwrap(), out.to_str().unwrap()])
            .output()
            .expect("scx convert failed to spawn");
        assert!(
            !output.status.success(),
            "an explicit --stream on {direction} must be rejected"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("--stream") && stderr.contains(direction),
            "expected a --stream rejection naming {direction}; got: {stderr}"
        );
        assert!(
            stderr.contains("Re-run without"),
            "the message must name the remedy; got: {stderr}"
        );
    }
}

#[test]
fn convert_stream_false_accepted_on_mtx() {
    // Deliberate backward compatibility: `--stream=false` on a direction that
    // never streams is a satisfied request, not an error. Scripts written
    // against the older CLI — which rejected the *default* `true` and so
    // forced users to pass `--stream=false` — keep working unchanged.
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("mtx_in");
    create_mtx_dir(&mtx_dir);
    let out = dir.path().join("out.scx");

    let output = scx()
        .args([
            "convert",
            "--from",
            "mtx",
            "--stream=false",
            mtx_dir.to_str().unwrap(),
            out.to_str().unwrap(),
        ])
        .output()
        .expect("scx convert failed to spawn");
    assert!(
        output.status.success(),
        "--stream=false must remain a no-op on mtx → scx; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ---------------------------------------------------------------------------
// The cross-cutting guard: no direction may be blocked by a flag's default.
// ---------------------------------------------------------------------------

#[test]
fn every_convert_direction_reachable_without_stream_flag() {
    // Asserts a *negative* about the failure mode rather than success, so it
    // needs no fixtures and passes on both feature sets: without `hdf5` the
    // four HDF5 directions fail with "requires the 'hdf5' feature", which
    // contains neither string we check for.
    //
    // This generalizes past `--stream` to the whole direction-guard family:
    // any future flag whose default rejects a direction fails here.
    let dir = tempfile::tempdir().unwrap();
    let bogus = dir.path().join("bogus_input");

    let directions: [(&str, &str); 7] = [
        ("--from", "mtx"),
        ("--from", "h5ad"),
        ("--from", "h5mu"),
        ("--from", "10x"),
        ("--to", "mtx"),
        ("--to", "h5ad"),
        ("--to", "h5mu"),
    ];

    for (flag, value) in directions {
        let out = dir.path().join(format!("out_{value}"));
        let output = scx()
            .args([
                "convert",
                flag,
                value,
                bogus.to_str().unwrap(),
                out.to_str().unwrap(),
            ])
            .output()
            .expect("scx convert failed to spawn");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("--stream"),
            "`convert {flag} {value}` with no flags must never mention --stream; got: {stderr}"
        );
        assert!(
            !stderr.contains("only supported for"),
            "`convert {flag} {value}` must not be blocked by a direction guard \
             it did not opt into; got: {stderr}"
        );
    }
}
