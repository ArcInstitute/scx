//! Whole-file BLAKE3, shared by the external-annotation readers.
//!
//! Lives in its own **ungated** module because both callers need it but only
//! one of them is `hdf5`-gated: `cellbender.rs` is behind the feature,
//! `annotation_table.rs` must not be (a CSV reader that requires libhdf5 would
//! block `scx obs-import` in a no-hdf5 build).

use std::path::Path;

/// Stream `path` through BLAKE3.
///
/// Streamed rather than read-then-hash: an annotation table is small, but a
/// CellBender output is not, and materializing it only to hash it would be a
/// pointless spike in peak RSS.
pub(crate) fn blake3_of_file(path: &Path) -> std::io::Result<[u8; 32]> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(*hasher.finalize().as_bytes())
}
