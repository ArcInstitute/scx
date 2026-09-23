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

// ---------------------------------------------------------------------------
// `--shard-obs` on the MTX ingest direction (phase 6c review round 1).
//
// The flag was accepted here and silently did nothing: `run_convert` returns
// through `dispatch_mtx_to_scx` before the policy is ever parsed, so
// `--shard-obs always` exited 0 having written one legacy `obs_metadata`
// section. A successful no-op on an explicit request is worse than an error,
// and it made `docs/api.md`'s "every scx convert ingest" claim false.
// ---------------------------------------------------------------------------

fn mtx_obs_shard_count(extra: &[&str]) -> usize {
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("mtx_in");
    create_mtx_dir(&mtx_dir);
    let out = dir.path().join("out.scx");

    let mut args: Vec<&str> = vec!["convert", "--from", "mtx"];
    args.extend_from_slice(extra);
    let mtx_s = mtx_dir.to_str().unwrap().to_string();
    let out_s = out.to_str().unwrap().to_string();
    args.push(&mtx_s);
    args.push(&out_s);

    let output = scx().args(&args).output().expect("scx convert spawn");
    assert!(
        output.status.success(),
        "mtx convert {args:?} failed; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let reader = scx_format_io::reader::ScxReader::open(&out).unwrap();
    assert_eq!(reader.header().n_obs, 3);
    reader.obs_metadata_shard_count()
}

#[test]
fn mtx_shard_obs_always_actually_shards() {
    // 3 obs rows at a target of 100: `auto` must not shard, `always` must —
    // so the pair distinguishes "honoured" from "ignored".
    assert_eq!(
        mtx_obs_shard_count(&["--shard-size", "100", "--shard-obs", "always"]),
        1,
    );
    assert_eq!(mtx_obs_shard_count(&["--shard-size", "100"]), 0);
}

#[test]
fn mtx_shard_obs_auto_shards_above_the_threshold() {
    assert_eq!(mtx_obs_shard_count(&["--shard-size", "2"]), 2);
    assert_eq!(
        mtx_obs_shard_count(&["--shard-size", "2", "--shard-obs", "off"]),
        0,
    );
}

// ---------------------------------------------------------------------------
// Destination overwrite protection on both MTX directions
// ---------------------------------------------------------------------------

/// The MTX destination is a *directory* that `write_scx_to_mtx` opens with
/// `create_dir_all`, so "does the path exist" is the wrong question: an
/// existing (or empty) directory is the ordinary case. The collision is a
/// member an export would replace.
#[test]
fn convert_to_mtx_writes_into_an_existing_empty_directory() {
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("mtx_in");
    create_mtx_dir(&mtx_dir);
    let scx_path = dir.path().join("mid.scx");
    assert!(scx()
        .args([
            "convert",
            "--from",
            "mtx",
            mtx_dir.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .output()
        .unwrap()
        .status
        .success());

    let out_dir = dir.path().join("empty_out");
    std::fs::create_dir_all(&out_dir).unwrap();
    let out = scx()
        .args([
            "convert",
            "--to",
            "mtx",
            scx_path.to_str().unwrap(),
            out_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "an empty destination directory is not a collision: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out_dir.join("matrix.mtx.gz").exists());
}

/// …and an existing export in that directory *is* one. `--force` then clears
/// the members it is about to replace, including the v2 `genes.tsv.gz`
/// spelling: left behind next to a fresh `features.tsv.gz`, the MTX *reader*
/// accepts either, so the directory would describe two different matrices.
#[test]
fn convert_to_mtx_refuses_an_existing_export_and_clears_it_on_force() {
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("mtx_in");
    create_mtx_dir(&mtx_dir);
    let scx_path = dir.path().join("mid.scx");
    assert!(scx()
        .args([
            "convert",
            "--from",
            "mtx",
            mtx_dir.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .output()
        .unwrap()
        .status
        .success());

    let out_dir = dir.path().join("stale_out");
    std::fs::create_dir_all(&out_dir).unwrap();
    std::fs::write(out_dir.join("matrix.mtx.gz"), b"stale").unwrap();
    std::fs::write(out_dir.join("genes.tsv.gz"), b"stale v2 features").unwrap();

    let args = [
        "convert",
        "--to",
        "mtx",
        scx_path.to_str().unwrap(),
        out_dir.to_str().unwrap(),
    ];
    let out = scx().args(args).output().unwrap();
    assert!(!out.status.success(), "a stale export must be refused");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("already contains an MTX export")
            && stderr.contains("matrix.mtx.gz")
            && stderr.contains("--force"),
        "the refusal must name the members and the flag: {stderr}"
    );
    assert_eq!(
        std::fs::read(out_dir.join("matrix.mtx.gz")).unwrap(),
        b"stale",
        "a refused invocation must not have touched the directory"
    );

    let mut forced = args.to_vec();
    forced.push("--force");
    let out = scx().args(&forced).output().unwrap();
    assert!(
        out.status.success(),
        "--force: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !out_dir.join("genes.tsv.gz").exists(),
        "the stale v2 features file must not survive beside the fresh features.tsv.gz"
    );
    assert!(out_dir.join("features.tsv.gz").exists());
}

#[test]
fn convert_from_mtx_refuses_to_clobber_its_output() {
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("mtx_in");
    create_mtx_dir(&mtx_dir);
    let dest = dir.path().join("out.scx");
    std::fs::write(&dest, b"do not clobber me").unwrap();

    let args = [
        "convert",
        "--from",
        "mtx",
        mtx_dir.to_str().unwrap(),
        dest.to_str().unwrap(),
    ];
    let out = scx().args(args).output().unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("already exists") && stderr.contains("--force"),
        "{stderr}"
    );
    assert_eq!(
        std::fs::read(&dest).unwrap(),
        b"do not clobber me",
        "a refused convert must leave the destination untouched"
    );

    let mut forced = args.to_vec();
    forced.push("--force");
    assert!(scx().args(&forced).output().unwrap().status.success());
}

// ---------------------------------------------------------------------------
// `integer` headers are taken at their word
// ---------------------------------------------------------------------------

fn write_big_count_mtx(dir: &std::path::Path, value: &str) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("matrix.mtx"),
        format!("%%MatrixMarket matrix coordinate integer general\n2 2 2\n1 1 1\n2 2 {value}\n"),
    )
    .unwrap();
    std::fs::write(dir.join("barcodes.tsv"), "AAAC-1\nBBBC-1\n").unwrap();
    std::fs::write(
        dir.join("features.tsv"),
        "ENSG001\tGeneA\tGene Expression\nENSG002\tGeneB\tGene Expression\n",
    )
    .unwrap();
}

#[test]
fn convert_from_mtx_refuses_a_count_past_2_24_and_allow_lossy_accepts_it() {
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("big_mtx");
    write_big_count_mtx(&mtx_dir, "16777217");
    let dest = dir.path().join("big.scx");

    let out = scx()
        .args([
            "convert",
            "--from",
            "mtx",
            mtx_dir.to_str().unwrap(),
            dest.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!out.status.success(), "the rounding must be refused");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("16777217") && stderr.contains("--allow-lossy"),
        "{stderr}"
    );
    assert!(!dest.exists(), "nothing should have been written");

    let out = scx()
        .args([
            "convert",
            "--from",
            "mtx",
            mtx_dir.to_str().unwrap(),
            dest.to_str().unwrap(),
            "--allow-lossy",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "--allow-lossy must accept it: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(dest.exists());
}

/// A flag that cannot apply must fail rather than be silently inert — the
/// lesson `scx modify-metadata --index-*` and `scx query --count --output`
/// both taught.
#[test]
fn allow_lossy_is_rejected_on_a_non_mtx_direction() {
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("mtx_in");
    create_mtx_dir(&mtx_dir);
    let scx_path = dir.path().join("mid.scx");
    assert!(scx()
        .args([
            "convert",
            "--from",
            "mtx",
            mtx_dir.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .output()
        .unwrap()
        .status
        .success());

    let out = scx()
        .args([
            "convert",
            "--to",
            "mtx",
            scx_path.to_str().unwrap(),
            dir.path().join("out_mtx").to_str().unwrap(),
            "--allow-lossy",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--allow-lossy only applies to mtx"),
        "{stderr}"
    );
}

/// The defect all three round-1 reviewers found, and the invariant the
/// round-1 tests stated for `File` destinations but never checked here.
///
/// `--force` used to unlink the MTX members inside the destination guard,
/// which runs straight after direction detection — before the remaining flag
/// validation and before the source is even opened. So a forced invocation
/// that then failed for *any* reason had already destroyed the previous
/// export. Reproduced on the pre-fix build: the directory came back empty.
#[test]
fn a_forced_export_that_fails_after_the_guard_leaves_the_old_one_intact() {
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("mtx_in");
    create_mtx_dir(&mtx_dir);
    let scx_path = dir.path().join("mid.scx");
    assert!(scx()
        .args([
            "convert",
            "--from",
            "mtx",
            mtx_dir.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .output()
        .unwrap()
        .status
        .success());

    // Seed a real export, then record every member byte for byte.
    let out_dir = dir.path().join("export");
    assert!(scx()
        .args([
            "convert",
            "--to",
            "mtx",
            scx_path.to_str().unwrap(),
            out_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap()
        .status
        .success());
    let before: Vec<(String, Vec<u8>)> = member_snapshot(&out_dir);
    assert_eq!(before.len(), 3, "the seeded export should have 3 members");

    // Each of these is forced, collides with that export, and fails *after*
    // the guard: a flag the direction rejects, and a source that cannot be
    // opened. Neither may touch the directory.
    let missing = dir.path().join("does_not_exist.scx");
    for args in [
        vec![
            "convert",
            "--to",
            "mtx",
            scx_path.to_str().unwrap(),
            out_dir.to_str().unwrap(),
            "--force",
            "--allow-lossy",
        ],
        vec![
            "convert",
            "--to",
            "mtx",
            missing.to_str().unwrap(),
            out_dir.to_str().unwrap(),
            "--force",
        ],
    ] {
        let out = scx().args(&args).output().unwrap();
        assert!(
            !out.status.success(),
            "expected a failure for {args:?}, got success"
        );
        assert_eq!(
            member_snapshot(&out_dir),
            before,
            "a forced invocation that failed must leave every member byte-identical: {args:?}"
        );
    }
}

/// …and the accept side: a forced export that *succeeds* still replaces the
/// members and clears the stale v2 spelling. Without this the test above
/// would pass against a `--force` that had simply stopped working.
#[test]
fn a_forced_export_that_succeeds_replaces_members_and_clears_stale_aliases() {
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("mtx_in");
    create_mtx_dir(&mtx_dir);
    let scx_path = dir.path().join("mid.scx");
    assert!(scx()
        .args([
            "convert",
            "--from",
            "mtx",
            mtx_dir.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .output()
        .unwrap()
        .status
        .success());

    let out_dir = dir.path().join("export");
    std::fs::create_dir_all(&out_dir).unwrap();
    std::fs::write(out_dir.join("matrix.mtx.gz"), b"stale").unwrap();
    std::fs::write(out_dir.join("genes.tsv.gz"), b"stale v2 features").unwrap();

    let out = scx()
        .args([
            "convert",
            "--to",
            "mtx",
            scx_path.to_str().unwrap(),
            out_dir.to_str().unwrap(),
            "--force",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "--force: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_ne!(
        std::fs::read(out_dir.join("matrix.mtx.gz")).unwrap(),
        b"stale".to_vec(),
        "the stale matrix must have been replaced"
    );
    assert!(
        !out_dir.join("genes.tsv.gz").exists(),
        "the stale v2 features file must not survive beside features.tsv.gz"
    );
    assert!(out_dir.join("features.tsv.gz").exists());
    // No staging debris left behind.
    let leftovers: Vec<String> = std::fs::read_dir(&out_dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with('.'))
        .collect();
    assert!(leftovers.is_empty(), "staging debris: {leftovers:?}");
}

/// A refused export must not even create the destination directory — a
/// command that errors should not have made anything.
#[test]
fn a_refused_export_does_not_create_the_output_directory() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("does_not_exist.scx");
    let out_dir = dir.path().join("never_made");

    let out = scx()
        .args([
            "convert",
            "--to",
            "mtx",
            missing.to_str().unwrap(),
            out_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(
        !out_dir.exists(),
        "a refused export created its output directory"
    );
}

/// Every member of an MTX directory, sorted, for byte-for-byte comparison.
fn member_snapshot(dir: &std::path::Path) -> Vec<(String, Vec<u8>)> {
    let mut out: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| e.path().is_file())
                .map(|e| {
                    (
                        e.file_name().to_string_lossy().into_owned(),
                        std::fs::read(e.path()).unwrap_or_default(),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// A stale alias that cannot be removed must be found *before* the export,
/// not after it has already replaced the three primary members.
///
/// `clear_stale_mtx_aliases` runs post-export, so without a preflight the
/// command exits non-zero with `matrix.mtx.gz` already swapped — breaking the
/// "a failed command changes nothing" boundary the rest of the guard keeps.
#[test]
fn an_unremovable_stale_alias_is_caught_before_the_export() {
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("mtx_in");
    create_mtx_dir(&mtx_dir);
    let scx_path = dir.path().join("mid.scx");
    assert!(scx()
        .args([
            "convert",
            "--from",
            "mtx",
            mtx_dir.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .output()
        .unwrap()
        .status
        .success());

    let out_dir = dir.path().join("export");
    assert!(scx()
        .args([
            "convert",
            "--to",
            "mtx",
            scx_path.to_str().unwrap(),
            out_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap()
        .status
        .success());
    let before = member_snapshot(&out_dir);

    // A stale v2 alias that `remove_file` cannot take.
    std::fs::create_dir(out_dir.join("genes.tsv.gz")).unwrap();

    let out = scx()
        .args([
            "convert",
            "--to",
            "mtx",
            scx_path.to_str().unwrap(),
            out_dir.to_str().unwrap(),
            "--force",
        ])
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "an unremovable alias must be refused"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cannot be cleared after the export"),
        "the refusal must say why: {stderr}"
    );
    let after: Vec<(String, Vec<u8>)> = member_snapshot(&out_dir)
        .into_iter()
        .filter(|(n, _)| n != "genes.tsv.gz")
        .collect();
    assert_eq!(
        after, before,
        "the primary members must not have been replaced"
    );
}

/// A destination spelled `.` is the ordinary "write the MTX files here"
/// invocation and must not be refused.
///
/// `normalize(".")` is the empty path, and `Path::starts_with(empty)` is true
/// for every path — so a lexical containment arm that does not special-case it
/// refuses every relative destination, whatever filesystem the source is on.
#[test]
fn a_dot_destination_is_not_treated_as_containing_everything() {
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("mtx_in");
    create_mtx_dir(&mtx_dir);
    let scx_path = dir.path().join("mid.scx");
    assert!(scx()
        .args([
            "convert",
            "--from",
            "mtx",
            mtx_dir.to_str().unwrap(),
            scx_path.to_str().unwrap(),
        ])
        .output()
        .unwrap()
        .status
        .success());

    let work = dir.path().join("cwd");
    std::fs::create_dir_all(&work).unwrap();
    let out = scx()
        .current_dir(&work)
        .args(["convert", "--to", "mtx", scx_path.to_str().unwrap(), "."])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "`.` as the MTX destination must work: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(work.join("matrix.mtx.gz").exists());
}

// ---------------------------------------------------------------------------
// The MTX direction follows the ingest CSC default, and honours a budget.
// ---------------------------------------------------------------------------

/// Convert the fixture with the `auto` thresholds lowered to 0 and `extra`
/// flags; returns (exit ok, has_csc).
fn mtx_csc(extra: &[&str]) -> (bool, bool) {
    let dir = tempfile::tempdir().unwrap();
    let mtx_dir = dir.path().join("mtx");
    create_mtx_dir(&mtx_dir);
    let out = dir.path().join("out.scx");
    let mut args = vec![
        "convert".to_string(),
        "--from".into(),
        "mtx".into(),
        mtx_dir.to_str().unwrap().into(),
        out.to_str().unwrap().into(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let status = scx()
        .args(&args)
        .env("SCX_CSC_AUTO_OBS_THRESHOLD", "0")
        .env("SCX_CSC_AUTO_VARS_THRESHOLD", "0")
        .status()
        .unwrap();
    let has_csc = status.success()
        && scx_format_io::ScxReader::open(&out)
            .unwrap()
            .header()
            .n_csc_shards
            > 0;
    (status.success(), has_csc)
}

#[test]
fn mtx_to_scx_builds_a_sidecar_by_default_and_not_under_csc_off() {
    assert_eq!(mtx_csc(&[]), (true, true), "unset --csc is auto");
    assert_eq!(mtx_csc(&["--csc", "off"]), (true, false));
}

/// `--memory-budget` reaches the MTX direction's sidecar build (it returned
/// before the budget was parsed, so a budget never bounded that build), and an
/// invalid one is refused rather than ignored.
#[test]
fn mtx_to_scx_honours_and_validates_memory_budget() {
    assert_eq!(mtx_csc(&["--memory-budget", "1M"]), (true, true));
    assert!(
        !mtx_csc(&["--memory-budget", "10MB"]).0,
        "decimal MB is rejected"
    );
}
