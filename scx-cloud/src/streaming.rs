//! Shared bounded-memory streaming helpers for the write-side cloud tools
//! (`push`, `explode`, `cloud_optimize`).
//!
//! These tools convert a packed `.scx` into a cloud-friendly form via
//! byte-faithful section copies — no decode or re-encode. Loading the whole
//! source into RAM (`std::fs::read`) makes peak memory scale with the
//! source-file size, which OOMs on multi-hundred-GB / TB atlases. The helpers
//! here read only the header + catalog up front (KB–MB, independent of source
//! size) and copy each section's byte range in bounded chunks, so peak memory
//! is `O(chunk × concurrency)` regardless of source size.

use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;

use scx_format_io::catalog::FullCatalog;
use scx_format_io::header::{FileHeader, HEADER_SIZE};

use crate::error::{CloudError, Result};

/// Streaming copy buffer size. 8 MiB is past the sequential-read sweet spot
/// for both local disk and object-store multipart parts; larger buffers don't
/// improve throughput and just raise peak RSS.
pub(crate) const CHUNK_SIZE: usize = 8 * 1024 * 1024;

/// Sections below this size upload via a single `put` rather than a multipart
/// upload: most metadata sections (obs/var/uns/indexes/catalog) are KB-scale,
/// and multipart-per-tiny-object adds round-trips (some backends also reject
/// sub-threshold multipart parts). X / CSC shards are bounded above this.
pub(crate) const MULTIPART_THRESHOLD: u64 = 16 * 1024 * 1024;

/// Read only the file header and full catalog from a packed `.scx`, without
/// materialising the whole file. Both reads are bounded (256 B header, KB–MB
/// catalog) regardless of source size.
pub(crate) fn read_header_and_catalog(path: &Path) -> Result<(FileHeader, FullCatalog)> {
    let mut file = File::open(path)?;

    let mut header_buf = [0u8; HEADER_SIZE];
    file.read_exact(&mut header_buf).map_err(|e| {
        CloudError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("file too small to contain SCX header: {e}"),
        ))
    })?;
    let header = FileHeader::read_from(&mut Cursor::new(&header_buf[..]))?;

    let fc_offset = header.full_catalog_offset;
    let fc_length = header.full_catalog_length as usize;
    file.seek(SeekFrom::Start(fc_offset))?;
    let mut catalog_buf = vec![0u8; fc_length];
    file.read_exact(&mut catalog_buf)?;
    let full_catalog = FullCatalog::read_from(&mut Cursor::new(&catalog_buf), fc_length, true)?;

    Ok((header, full_catalog))
}

/// Read the raw 256-byte file header verbatim (for `_header.bin` passthrough),
/// without parsing.
pub(crate) fn read_raw_header(path: &Path) -> Result<[u8; HEADER_SIZE]> {
    let mut file = File::open(path)?;
    let mut buf = [0u8; HEADER_SIZE];
    file.read_exact(&mut buf)?;
    Ok(buf)
}

/// Copy `length` bytes starting at `offset` in `source` into `sink`, in
/// bounded chunks. Used by `explode` (sink = a section file) and
/// `cloud_optimize` (sink = the output BufWriter). The source file position is
/// moved; the caller should not rely on it afterwards.
pub(crate) fn copy_section<W: Write>(
    source: &mut File,
    offset: u64,
    length: u64,
    sink: &mut W,
) -> Result<()> {
    source.seek(SeekFrom::Start(offset))?;
    let mut remaining = length;
    let cap = (length.min(CHUNK_SIZE as u64).max(1)) as usize;
    let mut buf = vec![0u8; cap];
    while remaining > 0 {
        let want = remaining.min(buf.len() as u64) as usize;
        source.read_exact(&mut buf[..want])?;
        sink.write_all(&buf[..want])?;
        remaining -= want as u64;
    }
    Ok(())
}
