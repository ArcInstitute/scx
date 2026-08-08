//! Unit tests for [`super::FreshnessGuard`].
//!
//! These drive the guard directly against hand-written header bytes, so each
//! change shape can be exercised alone — including the ones no real op
//! produces on demand: a same-size header rewrite, a pure mtime bump, and a
//! rewrite that leaves every stat field byte-identical. The end-to-end arms,
//! where real ops do the mutating, live in `pyscx/tests/test_stale_handle.py`.
//!
//! That an *unwatched* reader is unaffected is not asserted here; it is
//! asserted by the rest of the workspace suite staying green, since `scx-ops`
//! brackets every one of its own in-place mutations with readers that would
//! start failing against themselves if watching were ever the default.

#![cfg(unix)]

use std::io::{Read, Write};

use super::*;
use crate::header::FileHeader;

/// A file whose first 256 bytes are a valid `FileHeader`. Nothing here reads
/// past the header, so the body is padding.
fn write_file(path: &Path, manifest_sequence: u64, catalog_offset: u64, body_len: usize) {
    let mut header = FileHeader::new_single_modality(4, 3, 12, 1024, 1, 1);
    header.manifest_sequence = manifest_sequence;
    header.full_catalog_offset = catalog_offset;
    header.full_catalog_length = 64;
    let mut buf = Vec::new();
    header.write_to(&mut buf).unwrap();
    buf.resize(HEADER_SIZE + body_len, 0);
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&buf).unwrap();
    f.sync_all().unwrap();
}

fn stamp(path: &Path) -> FreshnessGuard {
    let mut buf = [0u8; HEADER_SIZE];
    let mut f = std::fs::File::open(path).unwrap();
    Read::read_exact(&mut f, &mut buf).unwrap();
    let header = FileHeader::read_from(&mut Cursor::new(&buf)).unwrap();
    FreshnessGuard::stamp(path, &header).unwrap()
}

fn err(guard: &FreshnessGuard) -> String {
    match guard.check() {
        Ok(()) => panic!("expected the guard to report a change"),
        Err(e) => e.to_string(),
    }
}

/// Move the file's mtime by a whole minute, without touching a byte.
/// Explicit rather than "write the same bytes again and hope the clock
/// ticked" — on a coarse-granularity filesystem that would silently take the
/// stat fast path and prove nothing.
fn bump_mtime(path: &Path) {
    let meta = std::fs::metadata(path).unwrap();
    let atime = std::os::unix::fs::MetadataExt::atime(&meta);
    let mtime = std::os::unix::fs::MetadataExt::mtime(&meta);
    set_times(path, atime, mtime + 60);
    let after = std::os::unix::fs::MetadataExt::mtime(&std::fs::metadata(path).unwrap());
    assert_ne!(after, mtime, "utimes must actually move mtime");
}

fn set_times(path: &Path, atime: i64, mtime: i64) {
    let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
    let times = [
        libc::timeval {
            tv_sec: atime as libc::time_t,
            tv_usec: 0,
        },
        libc::timeval {
            tv_sec: mtime as libc::time_t,
            tv_usec: 0,
        },
    ];
    assert_eq!(unsafe { libc::utimes(c.as_ptr(), times.as_ptr()) }, 0);
}

#[test]
fn an_untouched_file_is_fresh() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("a.scx");
    write_file(&p, 1, 4352, 128);
    let guard = stamp(&p);
    assert!(guard.check().is_ok());
    // The stamp is immutable, so a second call must agree with the first.
    assert!(guard.check().is_ok());
}

#[test]
fn a_pure_mtime_bump_is_not_a_change() {
    // `touch`, a metadata-preserving copy, or a backup pass moves mtime
    // without moving a byte. If mtime were treated as evidence of a change,
    // each of those would break every open handle for nothing.
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("a.scx");
    write_file(&p, 1, 4352, 128);
    let guard = stamp(&p);

    bump_mtime(&p);

    assert!(
        guard.check().is_ok(),
        "a changed mtime over unchanged bytes must not read as a changed file"
    );
}

#[test]
fn a_bumped_manifest_sequence_is_a_change() {
    // The in-place shape: same inode, header rewritten, sections appended.
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("a.scx");
    write_file(&p, 1, 4352, 128);
    let guard = stamp(&p);

    write_file(&p, 2, 4480, 256);
    let msg = err(&guard);
    assert!(msg.contains("changed on disk"), "{msg}");
    assert!(msg.contains("manifest_sequence 1 → 2"), "{msg}");
}

#[test]
fn a_moved_catalog_pointer_at_the_same_sequence_is_a_change() {
    // A same-size rewrite can leave both the length and the sequence alone
    // while pointing at a different catalog. Comparing the sequence on its own
    // would call this fresh.
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("a.scx");
    write_file(&p, 7, 4352, 128);
    let guard = stamp(&p);

    write_file(&p, 7, 9999, 128);
    let msg = err(&guard);
    assert!(msg.contains("changed on disk"), "{msg}");
}

#[test]
fn a_replaced_inode_is_a_change_even_at_an_identical_sequence() {
    // The copy-out shape. A `compact` of a `manifest_sequence == 1` file
    // yields another `manifest_sequence == 1` file, so the header comparison
    // alone reports fresh — only the inode separates them. The replacement
    // here is byte-identical on purpose: nothing but the inode differs.
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("a.scx");
    write_file(&p, 1, 4352, 128);
    let guard = stamp(&p);

    let tmp = dir.path().join("a.scx.tmp");
    write_file(&tmp, 1, 4352, 128);
    std::fs::rename(&tmp, &p).unwrap();

    let msg = err(&guard);
    assert!(msg.contains("was replaced on disk"), "{msg}");
    assert!(msg.contains("compact"), "{msg}");
}

#[test]
fn an_unlinked_file_is_a_change() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("a.scx");
    write_file(&p, 1, 4352, 128);
    let guard = stamp(&p);
    std::fs::remove_file(&p).unwrap();

    let msg = err(&guard);
    assert!(msg.contains("can no longer be read at that path"), "{msg}");
}

#[test]
fn the_message_names_the_file_and_the_way_out() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("atlas.scx");
    write_file(&p, 1, 4352, 128);
    let guard = stamp(&p);
    write_file(&p, 2, 4352, 128);

    let msg = err(&guard);
    assert!(msg.contains("atlas.scx"), "{msg}");
    assert!(msg.contains("reload()"), "{msg}");
}

#[test]
fn a_change_is_caught_with_every_stat_field_identical() {
    // The measured case, reproduced exactly. Linux refreshes inode timestamps
    // from a coarse clock and skips the store when the value has not moved, so
    // an `append` and the `rollback` that undoes it — microseconds apart —
    // landed with `st_size`, `st_ino` *and* `st_mtime_ns` all identical across
    // a change that took the file from 8 rows to 4.
    //
    // Restoring mtime by hand here is what makes this a test rather than a
    // race: with it, no stat field distinguishes the two states, and only the
    // header re-read can. Delete that re-read and this is the test that goes
    // red.
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("a.scx");
    write_file(&p, 2, 4352, 128);
    let guard = stamp(&p);

    let before = std::fs::metadata(&p).unwrap();
    let (ino, size) = (std::os::unix::fs::MetadataExt::ino(&before), before.len());
    let (atime, mtime) = (
        std::os::unix::fs::MetadataExt::atime(&before),
        std::os::unix::fs::MetadataExt::mtime(&before),
    );

    // Same length, same inode (File::create truncates in place), rewound
    // sequence — the shape `rollback` leaves behind.
    write_file(&p, 1, 4352, 128);
    set_times(&p, atime, mtime);

    let after = std::fs::metadata(&p).unwrap();
    assert_eq!(std::os::unix::fs::MetadataExt::ino(&after), ino);
    assert_eq!(after.len(), size);
    assert_eq!(std::os::unix::fs::MetadataExt::mtime(&after), mtime);

    let msg = err(&guard);
    assert!(msg.contains("manifest_sequence 2 → 1"), "{msg}");
}
