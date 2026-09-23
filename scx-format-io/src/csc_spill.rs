// Disk backing for the CSC builder's column buckets.
//
// The builder hands over whole blocks (1 MiB by default) in stream order and
// reads each bucket back exactly once, sequentially, so this store keeps **one
// file handle open at a time** and creates a bucket's file lazily on its first
// append. Both matter:
//
//   * One handle is why there is no descriptor cap here and no `RLIMIT_NOFILE`
//     probe anywhere. `scx-convert/src/h5ad/csc_stream.rs` opens one
//     `BufWriter<File>` per bucket and holds them all for a whole pass, which
//     is 11,922 descriptors at 50M cells against a default `ulimit -n` of
//     1024; that shape simply is not reachable from a blocked append.
//   * Lazy creation is what makes an all-in-memory build touch no disk at all.
//     A bucket that never overflows must produce no file, so `reader` answers
//     `Ok(None)` and the emit reads its RAM tail alone.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use scx_sparse::SpillStore;

use crate::error::ScxError;

/// A spill session directory, removed when the store is dropped.
///
/// The default root is deliberately **not** `std::env::temp_dir()`, which is
/// what `scx sort` and `scx convert` use. It is the output file's own
/// directory, because that is the filesystem the sidecar itself lands on —
/// appended in place, or staged beside the output by the copy-out form — so it
/// must already hold about as much as the spill, which is at most ~8 B/nnz. Defaulting to `/tmp` would
/// fail a census-scale rebuild on any host with a small tmpfs, and on a host
/// where `/tmp` *is* tmpfs the spill would count against RAM and silently
/// defeat the memory bound the spill exists to provide.
pub struct TempDirSpillStore {
    dir: tempfile::TempDir,
    /// Bytes appended per bucket; `0` means no file was ever created.
    bytes: Vec<u64>,
}

impl TempDirSpillStore {
    /// `root = None` uses the platform temp directory; callers that have an
    /// output path should pass its parent (see the type's own doc).
    pub fn new(root: Option<&Path>) -> Result<Self, ScxError> {
        let root = root.map_or_else(std::env::temp_dir, Path::to_path_buf);
        std::fs::create_dir_all(&root).map_err(|e| {
            ScxError::CscTranspose(format!(
                "failed to create the CSC spill root {}: {e}",
                root.display()
            ))
        })?;
        let dir = tempfile::Builder::new()
            .prefix("scx-csc-")
            .tempdir_in(&root)
            .map_err(|e| {
                ScxError::CscTranspose(format!(
                    "failed to create a CSC spill directory under {}: {e}",
                    root.display()
                ))
            })?;
        Ok(Self {
            dir,
            bytes: Vec::new(),
        })
    }

    pub fn path(&self) -> &Path {
        self.dir.path()
    }

    fn bucket_path(&self, bucket: usize) -> PathBuf {
        self.dir.path().join(format!("bucket_{bucket}.bin"))
    }
}

impl SpillStore for TempDirSpillStore {
    fn append(&mut self, bucket: usize, block: &[u8]) -> std::io::Result<()> {
        if self.bytes.len() <= bucket {
            self.bytes.resize(bucket + 1, 0);
        }
        let path = self.bucket_path(bucket);
        // No `BufWriter`: `block` is already a whole staging block (1 MiB by
        // default), so buffering it adds an 8 KiB allocation and a copy hop
        // around a single `write_all` that is orders of magnitude larger than
        // the buffer.
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        file.write_all(block)?;
        self.bytes[bucket] += block.len() as u64;
        Ok(())
    }

    /// One open for the whole spill, still closed before returning, so the
    /// store keeps its one-descriptor-at-a-time property.
    fn append_all(&mut self, bucket: usize, blocks: &[Vec<u8>]) -> std::io::Result<()> {
        if blocks.is_empty() {
            return Ok(());
        }
        if self.bytes.len() <= bucket {
            self.bytes.resize(bucket + 1, 0);
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.bucket_path(bucket))?;
        for block in blocks {
            file.write_all(block)?;
            self.bytes[bucket] += block.len() as u64;
        }
        Ok(())
    }

    fn reader(&self, bucket: usize) -> std::io::Result<Option<Box<dyn Read + '_>>> {
        if self.bytes.get(bucket).copied().unwrap_or(0) == 0 {
            return Ok(None);
        }
        Ok(Some(Box::new(File::open(self.bucket_path(bucket))?)))
    }

    fn spilled_bytes(&self, bucket: usize) -> u64 {
        self.bytes.get(bucket).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_sparse::{CscBuilder, CscBuilderConfig, MemSpillStore, ScxCsr, SpillStore};

    /// A deterministic 40 x 24 CSR in three unequal shards.
    fn shards() -> Vec<ScxCsr> {
        let mut out = Vec::new();
        let mut row = 0usize;
        for rows in [7usize, 1, 32] {
            let mut indptr = vec![0i64];
            let mut indices = Vec::new();
            let mut data = Vec::new();
            for r in row..row + rows {
                for k in 0..5 {
                    let c = (r * 7 + k * 5) % 24;
                    indices.push(c as i32);
                    data.push((r * 24 + c + 1) as f32);
                }
                let at = indices.len() - 5;
                indices[at..].sort_unstable();
                indptr.push(indices.len() as i64);
            }
            out.push(ScxCsr::new_unchecked((rows, 24), indptr, indices, data));
            row += rows;
        }
        out
    }

    fn drain(
        store: Box<dyn SpillStore>,
        spill_after_bytes: usize,
    ) -> Vec<(u64, Vec<u32>, Vec<f32>)> {
        let cfg = CscBuilderConfig {
            cols_per_shard: 4,
            memory_bytes: 1 << 20,
            spill_after_bytes,
            block_bytes: 8,
            ..CscBuilderConfig::default()
        };
        let mut b = CscBuilder::new(40, 24, cfg, store).expect("builder");
        let mut row_start = 0u64;
        for s in shards() {
            b.push_shard(row_start, &s).expect("push");
            row_start += s.n_rows() as u64;
        }
        let mut em = b.finish().expect("finish");
        let (mut ip, mut ix, mut dt) = (Vec::new(), Vec::new(), Vec::new());
        let mut out = Vec::new();
        while let Some(cs) = em.next_shard_into(&mut ip, &mut ix, &mut dt).expect("emit") {
            out.push((cs, ix.clone(), dt.clone()));
        }
        out
    }

    /// A real file store and the in-RAM one must be indistinguishable. The
    /// proptest in `scx-sparse` proves the *parser* is source-independent; this
    /// proves this store is a faithful byte sink for it.
    #[test]
    fn spilling_to_disk_matches_spilling_to_memory() {
        let dir = tempfile::tempdir().expect("root");
        let on_disk = TempDirSpillStore::new(Some(dir.path())).expect("store");
        assert_eq!(
            drain(Box::new(on_disk), 0),
            drain(Box::new(MemSpillStore::new()), 0)
        );
        // ...and both match the arm that never spills at all.
        assert_eq!(
            drain(Box::new(MemSpillStore::new()), usize::MAX),
            drain(Box::new(MemSpillStore::new()), 0)
        );
    }

    /// A bucket that never overflows must leave no file. That is what keeps a
    /// build that fits in RAM off the disk entirely, and it is the reason
    /// `reader` answers `Ok(None)` rather than opening an empty file.
    #[test]
    fn a_build_that_fits_in_memory_creates_no_files() {
        let dir = tempfile::tempdir().expect("root");
        let store = TempDirSpillStore::new(Some(dir.path())).expect("store");
        let session = store.path().to_path_buf();
        let _ = drain(Box::new(store), usize::MAX);
        // The session directory is dropped with the store; assert on the root.
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .expect("read root")
            .filter_map(Result::ok)
            .map(|e| e.path())
            .collect();
        assert!(
            leftovers.is_empty(),
            "no spill should have happened, found {leftovers:?} (session was {session:?})"
        );
    }

    /// The session directory does not survive the store, on success or on
    /// failure — the same contract `scx sort`'s `SpillDir` gives via `Drop`.
    #[test]
    fn the_session_directory_is_removed_on_drop() {
        let dir = tempfile::tempdir().expect("root");
        let session = {
            let store = TempDirSpillStore::new(Some(dir.path())).expect("store");
            let path = store.path().to_path_buf();
            let _ = drain(Box::new(store), 0);
            path
        };
        assert!(!session.exists(), "{session:?} outlived its store");
    }

    /// An unusable spill root is an actionable error naming the path, not a
    /// panic and not a bare `io::Error`.
    #[test]
    fn an_unwritable_spill_root_is_reported_with_its_path() {
        let dir = tempfile::tempdir().expect("root");
        let blocked = dir.path().join("not-a-directory");
        std::fs::write(&blocked, b"x").expect("write file");
        let msg = match TempDirSpillStore::new(Some(&blocked)) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a regular file must not be accepted as a spill root"),
        };
        assert!(msg.contains("not-a-directory"), "{msg}");
        assert!(msg.contains("spill"), "{msg}");
    }
}
