// File locking wrapper around fs4.

use fs4::fs_std::FileExt;
use std::fs::File;
use std::path::Path;

use crate::error::{OpsError, Result};

/// Exclusive file lock. Releases on drop.
pub struct FileLock {
    file: File,
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
        Ok(Self { file })
    }

    /// Return a reference to the inner file.
    pub fn file(&self) -> &File {
        &self.file
    }

    /// Consume the lock and return the inner file (lock remains held).
    pub fn into_file(self) -> File {
        // Prevent Drop from running (which would unlock)
        let file = unsafe { std::ptr::read(&self.file) };
        std::mem::forget(self);
        file
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

impl std::ops::Deref for FileLock {
    type Target = File;
    fn deref(&self) -> &File {
        &self.file
    }
}

impl std::ops::DerefMut for FileLock {
    fn deref_mut(&mut self) -> &mut File {
        &mut self.file
    }
}

impl std::io::Read for FileLock {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.file.read(buf)
    }
}

impl std::io::Write for FileLock {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.file.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

impl std::io::Seek for FileLock {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.file.seek(pos)
    }
}
