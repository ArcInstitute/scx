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
