// File locking wrapper around fs4.

use fs4::fs_std::FileExt;
use std::fs::File;
use std::path::Path;

use crate::error::{OpsError, Result};

/// Exclusive file lock. Releases on drop.
pub struct FileLock {
    file: Option<File>,
}

impl FileLock {
    /// Acquire an exclusive lock on the file at `path`.
    /// Opens the file for read+write.
    pub fn acquire_exclusive(path: &Path) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;
        file.lock_exclusive().map_err(OpsError::LockFailed)?;
        Ok(Self { file: Some(file) })
    }

    /// Return a reference to the inner file.
    pub fn file(&self) -> &File {
        self.file.as_ref().expect("FileLock already consumed")
    }

    /// Consume the lock and return the inner file (lock remains held
    /// until the returned File is dropped).
    pub fn into_file(mut self) -> File {
        self.file.take().expect("FileLock already consumed")
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        if let Some(ref file) = self.file {
            let _ = file.unlock();
        }
    }
}

impl std::ops::Deref for FileLock {
    type Target = File;
    fn deref(&self) -> &File {
        self.file.as_ref().expect("FileLock already consumed")
    }
}

impl std::ops::DerefMut for FileLock {
    fn deref_mut(&mut self) -> &mut File {
        self.file.as_mut().expect("FileLock already consumed")
    }
}

impl std::io::Read for FileLock {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.file.as_mut().expect("FileLock already consumed").read(buf)
    }
}

impl std::io::Write for FileLock {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.file.as_mut().expect("FileLock already consumed").write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.as_mut().expect("FileLock already consumed").flush()
    }
}

impl std::io::Seek for FileLock {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.file.as_mut().expect("FileLock already consumed").seek(pos)
    }
}
