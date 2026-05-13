// File locking wrapper around fs4.

use fs4::fs_std::FileExt;
use std::fs::File;
use std::path::Path;

use crate::error::{OpsError, Result};

/// Exclusive file lock. Releases on drop.
///
/// The inner `Option<File>` is a necessary quirk of the `into_file()`
/// consumer — it lets `Drop` run `unlock()` even after the file has been
/// moved out. The `file_ref()` / `file_mut()` helpers centralise the
/// `None` check so every trait impl routes through a single site. None
/// of these paths are reachable from safe user code: every public API
/// that accesses the file (Deref, DerefMut, Read, Write, Seek, `file()`)
/// takes `&self` or `&mut self`, and `into_file()` consumes `self` by
/// value — so observing a `None` requires an impossible borrow.
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
        self.file_ref()
    }

    /// Consume the lock and return the inner file (lock remains held
    /// until the returned File is dropped).
    pub fn into_file(mut self) -> File {
        self.file.take().expect("FileLock already consumed")
    }

    /// Internal shared reference to the underlying `File`.
    ///
    /// `None` is structurally unreachable from safe code because every
    /// method that calls this takes `&self` (which is impossible after
    /// `into_file()` has consumed `self`). `unreachable!()` is preferred
    /// over `expect()` to communicate intent: this is not a user-facing
    /// error, but an invariant violation that would indicate a soundness
    /// bug in the `FileLock` API itself.
    #[inline]
    fn file_ref(&self) -> &File {
        match self.file.as_ref() {
            Some(f) => f,
            None => unreachable!("FileLock::file is None with &self still alive"),
        }
    }

    /// Internal mutable reference to the underlying `File`.
    ///
    /// Same invariant as [`file_ref`] — see that method's doc comment.
    #[inline]
    fn file_mut(&mut self) -> &mut File {
        match self.file.as_mut() {
            Some(f) => f,
            None => unreachable!("FileLock::file is None with &mut self still alive"),
        }
    }
}

/// Shared (read) file lock. Releases on drop.
/// Prevents exclusive access by other processes while held.
pub struct SharedFileLock {
    file: File,
}

impl SharedFileLock {
    /// Acquire a shared lock on the file at `path`.
    /// Opens the file for read-only access.
    pub fn acquire(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        file.lock_shared().map_err(OpsError::LockFailed)?;
        Ok(Self { file })
    }
}

impl Drop for SharedFileLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
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
        self.file_ref()
    }
}

impl std::ops::DerefMut for FileLock {
    fn deref_mut(&mut self) -> &mut File {
        self.file_mut()
    }
}

impl std::io::Read for FileLock {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.file_mut().read(buf)
    }
}

impl std::io::Write for FileLock {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.file_mut().write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file_mut().flush()
    }
}

impl std::io::Seek for FileLock {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.file_mut().seek(pos)
    }
}
