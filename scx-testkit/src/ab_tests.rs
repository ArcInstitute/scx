//! Premise tests for the A/B manifest layer.
//!
//! Each was watched failing first — against a deliberately weakened assertion,
//! not against a stub — because a harness whose own tests have only ever been
//! green is exactly what this crate exists to stop.
//!
//! The env-reading tests share a mutex: `SCX_TESTKIT_AB_DUMP` /
//! `SCX_TESTKIT_AB_BASE` are process-global and `cargo test` runs a binary's
//! tests on threads, so two of them racing would make either one pass for the
//! wrong reason.

use super::*;
use crate::fixtures::{
    mixed_codec_file, mixed_codec_file_with, perturb_catalog_data_generation, FixtureOpts,
};

use std::path::Path;
use std::sync::{Mutex, MutexGuard, OnceLock};

fn env_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

/// Clear both A/B env vars for the duration of the guard's life.
struct ClearedEnv(#[allow(dead_code)] MutexGuard<'static, ()>);

impl ClearedEnv {
    fn new() -> Self {
        let g = env_lock();
        std::env::remove_var(DUMP_ENV);
        std::env::remove_var(BASE_ENV);
        Self(g)
    }
}

impl Drop for ClearedEnv {
    fn drop(&mut self) {
        std::env::remove_var(DUMP_ENV);
        std::env::remove_var(BASE_ENV);
    }
}

fn manifest_of(dir: &tempfile::TempDir, labels: &[(&str, FixtureOpts)]) -> OpDigestManifest {
    let mut m = OpDigestManifest::new();
    for (label, opts) in labels {
        let p = mixed_codec_file_with(&dir.path().join(format!("{label}.scx")), opts).unwrap();
        m.record(label, &p, Strictness::Content).unwrap();
    }
    m
}

fn two_op_manifest(dir: &tempfile::TempDir) -> OpDigestManifest {
    manifest_of(
        dir,
        &[
            ("compact", FixtureOpts::default()),
            ("merge", FixtureOpts::default()),
        ],
    )
}

/// A bare relative path — `base.json`, no directory — is what a reader of the
/// `SCX_TESTKIT_AB_DUMP` docs will type, and `Path::new("base.json").parent()`
/// is `Some("")`, not `None`.
///
/// On Linux `create_dir_all("")` returns `Ok(())`, so this passed before the
/// guard too; it is here because that is an undocumented platform detail on a
/// path the env var invites, not because a failure was observed. Runs from a
/// temp cwd, which is why it takes the same lock as the env-reading tests.
#[test]
fn a_bare_relative_dump_path_works() {
    let _env = ClearedEnv::new();
    let dir = tempfile::tempdir().unwrap();
    let m = two_op_manifest(&dir);

    let prev = std::env::current_dir().unwrap();
    std::env::set_current_dir(dir.path()).unwrap();
    let wrote = m.write_json(Path::new("base.json"));
    let read_back = OpDigestManifest::read_json(Path::new("base.json"));
    std::env::set_current_dir(prev).unwrap();

    wrote.expect("a bare relative dump path must be writable");
    assert_eq!(read_back.expect("and readable back"), m);
}

#[test]
fn a_manifest_round_trips_through_json() {
    let dir = tempfile::tempdir().unwrap();
    let m = two_op_manifest(&dir);
    let json = dir.path().join("nested/m.json");
    m.write_json(&json).unwrap();
    assert_eq!(OpDigestManifest::read_json(&json).unwrap(), m);
}

/// The premise the whole module rests on: two labels really do carry distinct
/// digests when their files differ, so an equality assertion is not vacuous.
#[test]
fn distinct_files_produce_distinct_entries() {
    let dir = tempfile::tempdir().unwrap();
    let m = manifest_of(
        &dir,
        &[
            ("plain", FixtureOpts::default()),
            (
                "perturbed",
                FixtureOpts {
                    perturb_float_bit: true,
                    ..Default::default()
                },
            ),
        ],
    );
    assert_ne!(
        m.ops["plain"].sections, m.ops["perturbed"].sections,
        "premise: the two fixtures must differ, or every comparison below is vacuous"
    );
}

#[test]
fn equal_manifests_compare_equal() {
    let dir = tempfile::tempdir().unwrap();
    let a = two_op_manifest(&dir);
    let b = {
        let dir2 = tempfile::tempdir().unwrap();
        two_op_manifest(&dir2)
    };
    // Different temp directories on purpose: the manifest is keyed by label,
    // not by path, which is the property that makes a cross-worktree A/B work.
    assert_manifests_eq(&a, &b);
}

#[test]
fn a_changed_section_is_reported_with_its_op_and_its_section() {
    let dir = tempfile::tempdir().unwrap();
    let a = two_op_manifest(&dir);

    let dir2 = tempfile::tempdir().unwrap();
    let mut b = OpDigestManifest::new();
    let compact = mixed_codec_file(&dir2.path().join("compact.scx")).unwrap();
    b.record("compact", &compact, Strictness::Content).unwrap();
    let merge = mixed_codec_file_with(
        &dir2.path().join("merge.scx"),
        &FixtureOpts {
            perturb_float_bit: true,
            ..Default::default()
        },
    )
    .unwrap();
    b.record("merge", &merge, Strictness::Content).unwrap();

    let err = std::panic::catch_unwind(|| assert_manifests_eq(&a, &b))
        .expect_err("a changed section must fail the comparison");
    let msg = err.downcast_ref::<String>().unwrap();
    assert!(msg.contains("X_shard_2"), "got:\n{msg}");
}

/// A missing op must fail *before* the per-op comparison, not be skipped.
///
/// This is the shape that silently degrades: a harness that intersects the two
/// label sets reports "all shared ops agree" for a run that measured one op.
#[test]
fn a_missing_op_fails_rather_than_narrowing_the_comparison() {
    let dir = tempfile::tempdir().unwrap();
    let full = two_op_manifest(&dir);

    let dir2 = tempfile::tempdir().unwrap();
    let partial = manifest_of(&dir2, &[("compact", FixtureOpts::default())]);

    let err = std::panic::catch_unwind(|| assert_manifests_eq(&partial, &full))
        .expect_err("a manifest missing an op must not pass");
    let msg = err.downcast_ref::<String>().unwrap();
    assert!(msg.contains("missing [\"merge\"]"), "got:\n{msg}");

    let err = std::panic::catch_unwind(|| assert_manifests_eq(&full, &partial))
        .expect_err("an extra op must not pass either");
    let msg = err.downcast_ref::<String>().unwrap();
    assert!(msg.contains("unexpected [\"merge\"]"), "got:\n{msg}");
}

#[test]
fn a_catalog_only_difference_is_caught() {
    let dir = tempfile::tempdir().unwrap();
    let a = manifest_of(&dir, &[("build_csc", FixtureOpts::default())]);

    let dir2 = tempfile::tempdir().unwrap();
    let p = mixed_codec_file(&dir2.path().join("build_csc.scx")).unwrap();
    perturb_catalog_data_generation(&p);
    let mut b = OpDigestManifest::new();
    b.record("build_csc", &p, Strictness::Content).unwrap();

    // Every section payload is byte-identical; only `data_generation` moved.
    assert_eq!(
        a.ops["build_csc"].sections, b.ops["build_csc"].sections,
        "premise: the perturbation must leave every section byte-identical, \
         or this proves nothing about the catalog half of the digest"
    );
    let err = std::panic::catch_unwind(|| assert_manifests_eq(&a, &b))
        .expect_err("a catalog-only difference must fail");
    let msg = err.downcast_ref::<String>().unwrap();
    assert!(msg.contains("catalog"), "got:\n{msg}");
}

#[test]
#[should_panic(expected = "duplicate manifest label")]
fn a_duplicate_label_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let p = mixed_codec_file(&dir.path().join("f.scx")).unwrap();
    let mut m = OpDigestManifest::new();
    m.record("compact", &p, Strictness::Content).unwrap();
    m.record("compact", &p, Strictness::Content).unwrap();
}

#[test]
#[should_panic(expected = "different harness version")]
fn a_manifest_from_another_harness_version_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut m = two_op_manifest(&dir);
    m.schema = "scx-testkit/op-digest-manifest/0".to_string();
    let json = dir.path().join("m.json");
    m.write_json(&json).unwrap();
    OpDigestManifest::read_json(&json).unwrap();
}

#[test]
fn golden_round_trips_and_a_first_run_cannot_pass() {
    let _env = ClearedEnv::new();
    let dir = tempfile::tempdir().unwrap();
    let m = two_op_manifest(&dir);
    let golden = dir.path().join("nested/op_output_identity.json");

    let first = std::panic::catch_unwind(|| resolve_against_env(&m, &golden).unwrap())
        .expect_err("a missing golden must be written AND fail");
    let msg = first.downcast_ref::<String>().unwrap();
    assert!(msg.contains("did not exist"), "got:\n{msg}");
    assert!(golden.exists());

    assert!(resolve_against_env(&m, &golden).unwrap());
}

#[test]
#[should_panic(expected = "refusing to resolve an empty manifest")]
fn an_empty_manifest_cannot_report_agreement() {
    let _env = ClearedEnv::new();
    let dir = tempfile::tempdir().unwrap();
    let golden = dir.path().join("g.json");
    resolve_against_env(&OpDigestManifest::new(), &golden).unwrap();
}

#[test]
fn dump_writes_and_base_compares() {
    let _env = ClearedEnv::new();
    let dir = tempfile::tempdir().unwrap();
    let m = two_op_manifest(&dir);
    let dumped = dir.path().join("base.json");
    let golden = dir.path().join("unused_golden.json");

    std::env::set_var(DUMP_ENV, &dumped);
    assert!(
        !resolve_against_env(&m, &golden).unwrap(),
        "a dump run asserts nothing and must say so"
    );
    assert!(dumped.exists());
    assert!(
        !golden.exists(),
        "a dump run must not touch the golden — that is the whole point of the mode"
    );
    std::env::remove_var(DUMP_ENV);

    std::env::set_var(BASE_ENV, &dumped);
    assert!(resolve_against_env(&m, &golden).unwrap());

    // And it fails against a different tree's output.
    let dir2 = tempfile::tempdir().unwrap();
    let other = manifest_of(
        &dir2,
        &[
            ("compact", FixtureOpts::default()),
            (
                "merge",
                FixtureOpts {
                    perturb_float_bit: true,
                    ..Default::default()
                },
            ),
        ],
    );
    let err = std::panic::catch_unwind(|| resolve_against_env(&other, &golden).unwrap())
        .expect_err("a changed tree must fail against the base manifest");
    let msg = err.downcast_ref::<String>().unwrap();
    assert!(msg.contains("X_shard_2"), "got:\n{msg}");
}

#[test]
#[should_panic(expected = "are both set")]
fn setting_both_env_vars_is_refused() {
    let _env = ClearedEnv::new();
    std::env::set_var(DUMP_ENV, "/tmp/a.json");
    std::env::set_var(BASE_ENV, "/tmp/b.json");
    ab_mode_from_env();
}

/// `wait_for_next_second` really does cross a second boundary.
///
/// Without this the clock-independence helpers below could be no-ops and every
/// caller would still be green — the exact failure the original helper hit.
#[test]
fn the_wait_actually_crosses_a_second() {
    let secs = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    };
    let before = secs();
    wait_for_next_second();
    assert!(secs() > before);
}
