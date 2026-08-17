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

use scx_testkit::digest::{assert_digests_eq, digest_file, digest_file_excluding, Strictness};
use scx_testkit::fixtures::mixed_codec_file;
use std::path::Path;

/// Block until the wall-clock second changes, so the second run's provenance
/// timestamp is guaranteed to differ from the first's.
///
/// Without this the test passes trivially whenever both runs land in the same
/// second — which is most of the time, and exactly when it proves nothing.
fn wait_for_next_second() {
    let now = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    };
    let start = now();
    while now() == start {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

/// Run `op` on two fresh copies of the same input, separated by a second, and
/// return the two files.
fn run_twice_across_a_second(
    dir: &Path,
    op: impl Fn(&Path),
) -> (std::path::PathBuf, std::path::PathBuf) {
    let src = mixed_codec_file(&dir.join("src.scx")).unwrap();
    let (a, b) = (dir.join("a.scx"), dir.join("b.scx"));
    std::fs::copy(&src, &a).unwrap();
    op(&a);
    wait_for_next_second();
    std::fs::copy(&src, &b).unwrap();
    op(&b);
    (a, b)
}

/// The premise every case below depends on: the two runs really did stamp
/// different timestamps, so "the digests agree" is a statement about the
/// exclusion rather than about two identical files.
fn assert_provenance_actually_differs(a: &Path, b: &Path) {
    let inc = |p: &Path| digest_file_excluding(p, Strictness::Content, &[]).unwrap();
    assert_ne!(
        inc(a).sections,
        inc(b).sections,
        "premise: the two runs must have stamped different provenance; \
         without that this test cannot fail"
    );
    assert_ne!(
        std::fs::read(a).unwrap(),
        std::fs::read(b).unwrap(),
        "premise: the two files must differ on disk"
    );
}

#[test]
fn mark_deleted_digests_agree_across_a_second() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = run_twice_across_a_second(dir.path(), |p| {
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

/// Run `op` twice over **one** fixture, a second apart, and assert the digests
/// agree while the raw bytes do not.
///
/// One source, not two. Building a second fixture would make the two outputs
/// differ for a reason that has nothing to do with the clock — each fixture ends
/// with `mark_deleted`, which stamps its own `SystemTime::now()`, so the two
/// inputs' `file_checksum`s differ and `merge` copies those into its provenance.
/// The premise assertion then passes with `wait_for_next_second()` deleted,
/// which is how this helper was first written and how it was caught: removing
/// the wait left every case green.
fn assert_op_is_clock_independent(label: &str, op: impl Fn(&Path, &Path)) {
    let dir = tempfile::tempdir().unwrap();
    let src = fixture_all_families(dir.path(), "src.scx");

    let a = dir.path().join(format!("{label}_a.scx"));
    op(&src, &a);

    wait_for_next_second();

    let b = dir.path().join(format!("{label}_b.scx"));
    op(&src, &b);

    assert_provenance_actually_differs(&a, &b);
    assert_digests_eq(
        &digest_file(&a, Strictness::Content).unwrap(),
        &digest_file(&b, Strictness::Content).unwrap(),
    );
}

#[test]
fn compact_over_every_family_is_clock_independent() {
    assert_op_is_clock_independent("compact", |src, out| {
        scx_ops::compact(src, out).unwrap();
    });
}

#[test]
fn sort_over_every_family_is_clock_independent() {
    let opts = scx_ops::SortOptions {
        by: vec!["cell_type".to_string()],
        ..Default::default()
    };
    assert_op_is_clock_independent("sort", |src, out| {
        scx_ops::sort_engine::sort(src, out, &opts).unwrap();
    });
}

#[test]
fn optimize_over_every_family_is_clock_independent() {
    assert_op_is_clock_independent("optimize", |src, out| {
        scx_ops::optimize::optimize(src, out, None, scx_format_io::ObsShardPolicy::Off).unwrap();
    });
}

#[test]
fn build_csc_over_every_family_is_clock_independent() {
    assert_op_is_clock_independent("build_csc", |src, out| {
        scx_ops::run_build_csc(src, out, "1G", false, 1024, None).unwrap();
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
