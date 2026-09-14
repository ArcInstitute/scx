//! Detecting that the file under an open reader has changed.
//!
//! A reader holds an `Mmap`, and the writer never edits bytes a reader is
//! looking at: an in-place op appends its new sections and rewrites the
//! 256-byte header, a copy-out op writes a temp file and renames it over the
//! target. Both leave the mapping intact and readable, describing a state that
//! is no longer on disk. Nothing in the mapping itself can reveal that — the
//! evidence is at the *path*, so detection means going back to the path.
//!
//! Three change shapes have to be caught, and each defeats an obvious detector:
//!
//! | op | inode | size | mtime | `manifest_sequence` |
//! |---|---|---|---|---|
//! | in-place (`obs_import`, `modify_metadata`, `mark_deleted`, …) | same | grows | see below | bumped |
//! | copy-out (`compact`, `sort`, `merge`) | **new** | any | new | need not move¹ |
//! | `rollback` | same | **unchanged** | see below | rewound |
//!
//! ¹ a `compact` of a `manifest_sequence == 1` file produces another one.
//!
//! # mtime is not a change detector
//!
//! The obvious cheap gate — "same size and mtime, therefore unchanged, skip
//! the rest" — is **wrong**, and wrong in exactly the case this module exists
//! for. Linux refreshes inode timestamps from a coarse clock updated once per
//! timer tick (~1–4 ms), and skips the store entirely when the new value
//! equals the old. A script that opens a handle and mutates the file a
//! microsecond later stays inside one tick, so the mutation lands with mtime
//! **identical to the nanosecond**. Measured, not reasoned: an `append`
//! followed by a `rollback` produced `st_size` 15927 → 15927, `st_ino`
//! 26758838 → 26758838, and `st_mtime_ns` 1786215502618971144 →
//! 1786215502618971144, across a change that took the file from 8 rows to 4.
//!
//! So the header is not a tie-breaker consulted when the stat looks
//! suspicious; the header **is** the check. What the stat contributes is the
//! one thing a header read cannot see: that the path now holds a *different
//! inode*, which is what every copy-out op leaves behind and which the
//! retained descriptor below would happily read straight past.
//!
//! # The check
//!
//! 1. `stat(path)` — gone, or `(dev, ino)` moved → changed.
//! 2. `pread` 256 bytes through a descriptor held open since the stamp, and
//!    compare the catalog pointer → changed if it moved.
//!
//! Two syscalls, no allocation, no interior mutability. Holding the descriptor
//! is what keeps step 2 down to one syscall instead of open/read/close, and it
//! pins the same inode the mapping already pins, so it costs nothing that was
//! not already being kept alive. Only readers that opted in via
//! `ScxReader::watching` hold one.

use std::fs::File;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use crate::error::{Result, ScxError};
use crate::header::{FileHeader, HEADER_SIZE};

/// Which file the path pointed at when the stamp was taken. Deliberately
/// *only* the identity: no size, no timestamps — see the module docs for why
/// those cannot carry the verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InodeIdentity {
    dev: u64,
    ino: u64,
}

impl InodeIdentity {
    fn of(meta: &std::fs::Metadata) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Self {
                dev: meta.dev(),
                ino: meta.ino(),
            }
        }
        // Without inode numbers a replacement is invisible here and step 2
        // decides on its own — correct except for a rename of a
        // byte-identical file, which is indistinguishable from no change.
        #[cfg(not(unix))]
        {
            let _ = meta;
            Self { dev: 0, ino: 0 }
        }
    }
}

/// The header fields every mutating op moves. Deliberately not the whole
/// header: `n_obs` and the shard counters are *derived* from whichever catalog
/// the pointer below selects, so comparing the pointer is both necessary and
/// sufficient, and stays right if the header grows fields later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CatalogIdentity {
    manifest_sequence: u64,
    full_catalog_offset: u64,
    full_catalog_length: u64,
}

impl CatalogIdentity {
    fn of(header: &FileHeader) -> Self {
        Self {
            manifest_sequence: header.manifest_sequence,
            full_catalog_offset: header.full_catalog_offset,
            full_catalog_length: header.full_catalog_length,
        }
    }

    /// Re-read the header through an already-open descriptor. One `pread`; no
    /// seek (so it is safe to call concurrently on `&self`), no mmap, and the
    /// catalog is never touched.
    fn reread(file: &File) -> Result<Self> {
        let mut buf = [0u8; HEADER_SIZE];
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            file.read_exact_at(&mut buf, 0)?;
        }
        #[cfg(not(unix))]
        {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = file.try_clone()?;
            f.seek(SeekFrom::Start(0))?;
            f.read_exact(&mut buf)?;
        }
        Ok(Self::of(&FileHeader::read_from(&mut Cursor::new(&buf))?))
    }
}

/// A file's identity at a point in time, stamped without retaining a descriptor.
///
/// [`FreshnessGuard`] answers "has the file under this *open* reader changed",
/// and keeps the descriptor open so its re-read is a single `pread`. A bounded
/// reader registry asks a narrower question — "is the file I am about to
/// *reopen* still the one I scanned?" — and wants to hold nothing at all
/// between the two opens.
///
/// Not because a descriptor is scarce: an unwatched `ScxReader` holds none, and
/// what a bounded registry reclaims is the parsed catalog, not an fd. The
/// reason is that `watching()` is a property of the reader for its whole life —
/// it would put a `stat` and a `pread` on every section read of the default
/// path, to answer a question only a reopen asks. Same two fields and the same
/// verdict as `check()`; only the descriptor differs, which is why this lives
/// here rather than being re-derived by the caller from `header()` and a
/// `stat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileIdentity {
    inode: InodeIdentity,
    catalog: CatalogIdentity,
}

impl FileIdentity {
    /// Stamp `path`'s identity, using the header a reader has already parsed
    /// from it. One `stat`; the header is never re-read.
    pub fn stamp(path: &Path, header: &FileHeader) -> Result<Self> {
        let meta = std::fs::metadata(path).map_err(|e| {
            ScxError::Io(std::io::Error::new(
                e.kind(),
                format!("cannot stat '{}': {}", path.display(), e),
            ))
        })?;
        Ok(Self {
            inode: InodeIdentity::of(&meta),
            catalog: CatalogIdentity::of(header),
        })
    }

    /// `Ok(())` when `now` names the same inode and the same catalog as `self`.
    ///
    /// The two clauses are not redundant: an in-place op (`append`, `rollback`)
    /// keeps the inode and moves the catalog pointer, while a copy-out op
    /// (`compact`, `sort`, `merge`) renames a new inode into place and may land
    /// on any pointer at all. Timestamps and size decide neither — see the
    /// module docs.
    pub fn ensure_same(&self, path: &Path, now: &Self) -> Result<()> {
        if now.inode != self.inode {
            return Err(ScxError::FileChangedOnDisk {
                path: path.display().to_string(),
                detail: "was replaced on disk since it was first opened (a copy-out op such \
                         as compact, sort or merge writes a new file and renames it into place)"
                    .to_string(),
            });
        }
        if now.catalog != self.catalog {
            // The sequence counter is the usual mover, but it is not the only
            // field compared: an op that rewrites in place without bumping it
            // still moves the catalog pointer, and reporting
            // "manifest_sequence 1 -> 1" for that sends the reader looking at
            // the wrong thing. **Review on #536 (Antigravity).**
            let detail = if self.catalog.manifest_sequence != now.catalog.manifest_sequence {
                format!(
                    "changed on disk since it was first opened (manifest_sequence {} \u{2192} {})",
                    self.catalog.manifest_sequence, now.catalog.manifest_sequence
                )
            } else {
                format!(
                    "changed on disk since it was first opened (manifest_sequence unchanged at \
                     {}, but the catalog moved: offset {} \u{2192} {}, length {} \u{2192} {})",
                    self.catalog.manifest_sequence,
                    self.catalog.full_catalog_offset,
                    now.catalog.full_catalog_offset,
                    self.catalog.full_catalog_length,
                    now.catalog.full_catalog_length
                )
            };
            return Err(ScxError::FileChangedOnDisk {
                path: path.display().to_string(),
                detail,
            });
        }
        Ok(())
    }
}

/// Watches the file a reader was opened from for changes made behind its back.
///
/// Attached to an [`crate::reader::ScxReader`] only when the caller opts in via
/// `ScxReader::watching`. Readers opened the ordinary way carry `None` and pay
/// nothing — which is what keeps `scx-ops` (whose readers bracket its *own*
/// mutations), the CLI, and the training loader unaffected.
#[derive(Debug)]
pub struct FreshnessGuard {
    path: PathBuf,
    /// Held open since the stamp so step 2 is a single `pread`. Pins the same
    /// inode the reader's `Mmap` already pins, so it keeps nothing alive that
    /// was not already alive.
    file: File,
    inode: InodeIdentity,
    catalog: CatalogIdentity,
}

impl FreshnessGuard {
    /// Stamp the file's identity as of now. `header` is the one the reader just
    /// parsed, so this costs one `open` and one `stat` and re-reads nothing.
    pub(crate) fn stamp(path: &Path, header: &FileHeader) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            ScxError::Io(std::io::Error::new(
                e.kind(),
                format!("cannot open '{}' to watch it: {}", path.display(), e),
            ))
        })?;
        let meta = file.metadata().map_err(|e| {
            ScxError::Io(std::io::Error::new(
                e.kind(),
                format!("cannot stat '{}': {}", path.display(), e),
            ))
        })?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            inode: InodeIdentity::of(&meta),
            catalog: CatalogIdentity::of(header),
        })
    }

    /// The path this guard watches.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// `Ok(())` if the file is still the one that was opened.
    pub fn check(&self) -> Result<()> {
        let meta = match std::fs::metadata(&self.path) {
            Ok(meta) => meta,
            // Readable at open, not now. Unlinked, renamed away, permissions
            // revoked — whatever the reason, the mapping is no longer backed
            // by anything at this path.
            Err(e) => return Err(self.changed(format!("can no longer be read at that path ({e})"))),
        };
        if InodeIdentity::of(&meta) != self.inode {
            return Err(self.changed(
                "was replaced on disk since it was opened (a copy-out op such as compact, \
                 sort or merge writes a new file and renames it into place)"
                    .to_string(),
            ));
        }

        let catalog = match CatalogIdentity::reread(&self.file) {
            Ok(catalog) => catalog,
            // It no longer even parses as a header. Certainly not the file
            // that was opened.
            Err(e) => {
                return Err(self.changed(format!("is no longer readable as an SCX file ({e})")))
            }
        };
        if catalog == self.catalog {
            return Ok(());
        }
        Err(self.changed(format!(
            "changed on disk since it was opened (manifest_sequence {} → {})",
            self.catalog.manifest_sequence, catalog.manifest_sequence
        )))
    }

    fn changed(&self, detail: String) -> ScxError {
        ScxError::FileChangedOnDisk {
            path: self.path.display().to_string(),
            detail,
        }
    }
}

#[cfg(test)]
#[path = "freshness_tests.rs"]
mod tests;
