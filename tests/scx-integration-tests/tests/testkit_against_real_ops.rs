//! Does the digest harness survive a *real* op?
//!
//! `scx-testkit`'s own premise tests write fixtures with a fixed provenance
//! timestamp, which proves the exclusion works but not that it works against
//! the thing it exists for: an `scx-ops` mutation that stamps
//! `SystemTime::now()` and re-hashes the file header. This file runs each op
//! twice, deliberately straddling a wall-clock second, and asserts the digests
//! agree — which is the entire claim that makes a checked-in golden viable for
//! Phases 5, 6 and 8.
//!
//! If any of these ever fails, the harness is not usable for that op and the
//! phase consuming it needs to know before it starts, not after.
//!
//! **This is the weaker of the two op-level claims.** It compares two runs of
//! the *same* code, so it says the harness works on an op and nothing about
//! whether that op's behaviour changed. The stronger claim — every op still
//! writes what it wrote at some earlier commit — lives in
//! `op_output_identity.rs`, which pins a golden manifest and can also A/B
//! across two worktrees.
//!
//! The run-it-twice helpers these cases use were private to this file until
//! PR-01; they now live in `scx_testkit::ab` so any op's tests can reach them.

use scx_testkit::ab::{
    assert_op_is_clock_independent, assert_provenance_actually_differs, run_twice_across_a_second,
    wait_for_next_second,
};
use scx_testkit::digest::{assert_digests_eq, digest_file, Strictness};
use scx_testkit::fixtures::mixed_codec_file;
use std::path::Path;

#[test]
fn mark_deleted_digests_agree_across_a_second() {
    let dir = tempfile::tempdir().unwrap();
    let src = mixed_codec_file(&dir.path().join("src.scx")).unwrap();
    let (a, b) = run_twice_across_a_second(&src, dir.path(), "mark_deleted", |p| {
        scx_ops::mark_deleted(p, &[1u64, 5, 9]).unwrap();
    });
    assert_provenance_actually_differs(&a, &b);
    assert_digests_eq(
        &digest_file(&a, Strictness::Content).unwrap(),
        &digest_file(&b, Strictness::Content).unwrap(),
    );
}

#[test]
fn compact_digests_agree_across_a_second() {
    let dir = tempfile::tempdir().unwrap();
    let src = mixed_codec_file(&dir.path().join("src.scx")).unwrap();
    scx_ops::mark_deleted(&src, &[2u64, 7]).unwrap();

    let (a, b) = (dir.path().join("ca.scx"), dir.path().join("cb.scx"));
    scx_ops::compact(&src, &a).unwrap();
    wait_for_next_second();
    scx_ops::compact(&src, &b).unwrap();

    assert_provenance_actually_differs(&a, &b);
    assert_digests_eq(
        &digest_file(&a, Strictness::Content).unwrap(),
        &digest_file(&b, Strictness::Content).unwrap(),
    );
}

/// And the harness still *fails* on a real difference produced by a real op —
/// `compact` after deleting different rows. A digest that agreed here would be
/// agreeing about nothing.
#[test]
fn compact_digests_differ_when_the_input_differs() {
    let dir = tempfile::tempdir().unwrap();
    let mk = |name: &str, deleted: &[u64]| {
        let src = mixed_codec_file(&dir.path().join(format!("{name}_src.scx"))).unwrap();
        scx_ops::mark_deleted(&src, deleted).unwrap();
        let out = dir.path().join(format!("{name}.scx"));
        scx_ops::compact(&src, &out).unwrap();
        out
    };
    let a = mk("a", &[2u64, 7]);
    let b = mk("b", &[3u64, 8]);
    assert_ne!(
        digest_file(&a, Strictness::Content).unwrap().sections,
        digest_file(&b, Strictness::Content).unwrap().sections,
        "compacting away different rows must change the digest"
    );
}

// ---------------------------------------------------------------------------
// The ops Phase 5's rewrite-loop consolidation (ORG-6.14-1) will actually
// touch, over a file that carries every section family.
//
// The cases above run on `mixed_codec_file`, which is X shards and `uns`. That
// is enough to prove the provenance exclusion works, and not enough to tell a
// phase that lifts `CsrEmitter` out of `sort_engine` and pushes `compact` and
// `merge` rows through it whether the harness still holds when the file also
// has layers, obsm/varm, obsp/varp, `.raw`, bitmaps, a group index, predicate
// indexes and a deletion vector.
//
// `merge` and `sort` are the gap that matters: neither was covered before, and
// both are in that phase's blast radius. Better to learn the harness does not
// hold for one of them now than after the refactor is written.
//
// What these prove is that provenance's clock does not leak into the digest for
// these ops -- i.e. that the harness is usable on them. They compare two runs of
// the *same* code, so they cannot and do not say anything about whether that
// code's behaviour changed; that claim needs a comparison across two trees.
// ---------------------------------------------------------------------------

mod common;

use common::fixture_all_families;

/// `assert_op_is_clock_independent` over the all-families fixture.
///
/// The helper itself is `scx_testkit::ab`'s; this only supplies the fixture,
/// which lives in this crate's `tests/common`.
fn over_every_family(label: &str, op: impl Fn(&Path, &Path)) {
    let dir = tempfile::tempdir().unwrap();
    let src = fixture_all_families(dir.path(), "src.scx");
    assert_op_is_clock_independent(&src, dir.path(), label, op);
}

#[test]
fn compact_over_every_family_is_clock_independent() {
    over_every_family("compact", |src, out| {
        scx_ops::compact(src, out).unwrap();
    });
}

#[test]
fn sort_over_every_family_is_clock_independent() {
    let opts = scx_ops::SortOptions {
        by: vec!["cell_type".to_string()],
        ..Default::default()
    };
    over_every_family("sort", |src, out| {
        scx_ops::sort_engine::sort(src, out, &opts).unwrap();
    });
}

#[test]
fn optimize_over_every_family_is_clock_independent() {
    over_every_family("optimize", |src, out| {
        scx_ops::optimize::optimize(src, out, None, scx_format_io::ObsShardPolicy::Off).unwrap();
    });
}

#[test]
fn build_csc_over_every_family_is_clock_independent() {
    over_every_family("build_csc", |src, out| {
        scx_ops::run_build_csc(src, out, "1G", false, 1024, None, None).unwrap();
    });
}

/// `merge` takes two inputs, so it does not fit the helper — and it is the op
/// whose provenance carries the *inputs'* checksums alongside its own timestamp,
/// which is the case most likely to break the exclusion.
#[test]
fn merge_over_every_family_is_clock_independent() {
    let dir = tempfile::tempdir().unwrap();
    let x = fixture_all_families(dir.path(), "x.scx");
    let y = fixture_all_families(dir.path(), "y.scx");
    let mk = |tag: &str| {
        let out = dir.path().join(format!("{tag}_merged.scx"));
        scx_ops::merge::merge(&[&x, &y], &out).unwrap();
        out
    };
    let a = mk("a");
    wait_for_next_second();
    let b = mk("b");

    assert_provenance_actually_differs(&a, &b);
    assert_digests_eq(
        &digest_file(&a, Strictness::Content).unwrap(),
        &digest_file(&b, Strictness::Content).unwrap(),
    );
}
