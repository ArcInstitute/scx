//! Shared header-finalization helper for mutation ops.
//!
//! Historically `append`, `delete`, and `rollback` each implemented the same
//! checksum-finalize dance:
//!
//! 1. Write the header with `file_checksum = 0` to disk.
//! 2. Flush + hash the entire file to compute the BLAKE3 checksum.
//! 3. Seek back to 0 and write the header again with the final checksum.
//!
//! That pattern exposes a tiny crash window: a power loss between steps 1 and
//! 3 leaves a zero-checksum header on disk (unreadable by a strict reader).
//! [`finalize_header_with_checksum`] closes that window by computing the
//! BLAKE3 digest from in-memory bytes + the existing file body, and writing
//! the final header exactly once at offset 0.
//!
//! Addresses review findings H5, H7, and M16.

use std::io::{Read, Seek, SeekFrom, Write};

use scx_format_io::header::{FileHeader, HEADER_SIZE};

use crate::error::Result;

/// Write `header` at offset 0 with the correct BLAKE3-truncated file checksum
/// in a single disk write.
///
/// The file body (everything after the 256-byte header) must already be in
/// its final state when this is called — shards, catalog, root catalog at
/// offset 256, etc. `header.file_checksum` is overwritten by this function;
/// the caller should leave it at 0 (or any sentinel — the helper zeroes it
/// before hashing anyway) to make intent obvious.
///
/// Also writes durability barriers:
/// - `flush()` + `sync_all()` before reading the body (ensures the root
///   catalog the caller just wrote is durable on disk before we hash it),
///   and
/// - `sync_all()` after writing the header (ensures the final checksum is
///   durable before returning).
///
/// # Invariant
///
/// The stored checksum is computed over the full file with the
/// `file_checksum` field zeroed. Readers that verify the file checksum MUST
/// zero the same 8-byte field in their copy before re-hashing. The writer
/// path in `scx-format` uses the same convention (see `writer.rs::finish`).
pub fn finalize_header_with_checksum<F: Read + Write + Seek + SyncAllIfApplicable>(
    file: &mut F,
    header: &mut FileHeader,
) -> Result<()> {
    // 1. Durability barrier for everything written before the header.
    //    Without this, a crash after the header write could find the root
    //    catalog or appended body only partly on disk while a new header
    //    points at it.
    file.flush()?;
    file.sync_all_if_applicable()?;

    // 2. Serialize the header with checksum=0 into an in-memory buffer.
    header.file_checksum = 0;
    let mut header_bytes = Vec::with_capacity(HEADER_SIZE);
    header.write_to(&mut header_bytes)?;
    debug_assert_eq!(header_bytes.len(), HEADER_SIZE);

    // 3. Hash [header_bytes_with_zero || file_body_bytes[256..]]. We stream
    //    the body from disk in 64 KiB chunks rather than loading it.
    let mut hasher = blake3::Hasher::new();
    hasher.update(&header_bytes);

    file.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
    let mut chunk = [0u8; 65536];
    loop {
        let n = file.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        hasher.update(&chunk[..n]);
    }
    let digest = hasher.finalize();
    let file_checksum = scx_format_io::checksum::truncate_hash_to_u64(&digest);

    // 4. Patch the checksum field of the in-memory header bytes. Keeps the
    //    struct in sync with what hits disk.
    header.file_checksum = file_checksum;
    let mut patched = Vec::with_capacity(HEADER_SIZE);
    header.write_to(&mut patched)?;
    debug_assert_eq!(patched.len(), HEADER_SIZE);

    // 5. Single durable write of the final header.
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&patched)?;
    file.sync_all_if_applicable()?;
    Ok(())
}

/// Shim so `finalize_header_with_checksum` can call `sync_all` on anything
/// `Read + Write + Seek`.
///
/// For a real `File` (and the `FileLock` newtype, which derefs to `File`) we
/// call `File::sync_all`. For in-memory `Cursor<Vec<u8>>` we no-op — tests
/// don't need an fsync.
pub trait SyncAllIfApplicable {
    fn sync_all_if_applicable(&mut self) -> std::io::Result<()>;
}

impl SyncAllIfApplicable for std::fs::File {
    fn sync_all_if_applicable(&mut self) -> std::io::Result<()> {
        self.sync_all()
    }
}

impl SyncAllIfApplicable for crate::flock::FileLock {
    fn sync_all_if_applicable(&mut self) -> std::io::Result<()> {
        // `FileLock` derefs to `File`, but the trait dispatch here goes
        // through this impl first — so route explicitly.
        (**self).sync_all()
    }
}

impl<T: AsRef<[u8]>> SyncAllIfApplicable for std::io::Cursor<T> {
    fn sync_all_if_applicable(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    use scx_format_io::header::{CURRENT_FORMAT_VERSION, MAGIC};

    fn scratch_header() -> FileHeader {
        FileHeader {
            magic: MAGIC,
            format_version: CURRENT_FORMAT_VERSION,
            header_length: HEADER_SIZE as u16,
            flags: 0,
            n_obs: 10,
            n_vars: 20,
            nnz: 30,
            n_csr_shards: 1,
            n_csc_shards: 0,
            shard_target_rows: 16384,
            codec_id: 0,
            index_dtype: 0,
            endian: 0,
            reserved_padding: 0,
            root_catalog_offset: HEADER_SIZE as u64,
            root_catalog_length: 0,
            full_catalog_offset: 4352,
            full_catalog_length: 0,
            manifest_sequence: 1,
            prev_catalog_offset: 0,
            file_checksum: 0xDEAD,
            front_catalog_offset: 0,
            front_catalog_length: 0,
            n_modalities: 0,
            modality_table_offset: 0,
            modality_table_length: 0,
            reserved: [0u8; 112],
        }
    }

    /// Final checksum equals BLAKE3 of [zero-checksum header || body], so the
    /// same reader pattern (zero field + rehash + compare) reproduces the
    /// stored value bit-for-bit.
    #[test]
    fn finalize_matches_zero_field_rehash() {
        let mut header = scratch_header();
        let body: Vec<u8> = (0u8..=255).cycle().take(4096 + 128).collect();

        // Build an in-memory "file": 256-byte stale header + body.
        let mut buf = vec![0u8; HEADER_SIZE];
        buf.extend_from_slice(&body);
        let mut cursor = Cursor::new(buf);

        finalize_header_with_checksum(&mut cursor, &mut header).unwrap();

        // Re-verify: read header, zero the checksum field, rehash, compare.
        let file_bytes = cursor.into_inner();
        let mut reread = Cursor::new(&file_bytes);
        let hdr = FileHeader::read_from(&mut reread).unwrap();
        assert_eq!(hdr.file_checksum, header.file_checksum);
        assert_ne!(hdr.file_checksum, 0, "checksum must not be zero");

        // Recompute to prove the invariant.
        let mut copy = file_bytes.clone();
        // file_checksum lives at offset 4 + 2 + 2 + 4 + 8 + 8 + 8 + 4 + 4 + 4
        //                              + 1 + 1 + 1 + 1 + 8 + 8 + 8 + 8 + 8 + 8 = 96
        // i.e. 24 + 8 (n_obs) + ... let us just overwrite the 8 bytes we
        // know land there by re-serializing the header with checksum=0.
        let mut zero_hdr = hdr.clone();
        zero_hdr.file_checksum = 0;
        let mut zhb = Vec::with_capacity(HEADER_SIZE);
        zero_hdr.write_to(&mut zhb).unwrap();
        copy[..HEADER_SIZE].copy_from_slice(&zhb);

        let digest = blake3::hash(&copy);
        let expected = scx_format_io::checksum::truncate_hash_to_u64(&digest);
        assert_eq!(expected, hdr.file_checksum);
    }

    /// Calling the helper again after appending more body data updates the
    /// checksum — proves it re-reads the current file state each time.
    #[test]
    fn finalize_idempotent_across_body_changes() {
        let mut header = scratch_header();
        let mut buf = vec![0u8; HEADER_SIZE];
        buf.extend_from_slice(b"initial body");
        let mut cursor = Cursor::new(buf);
        finalize_header_with_checksum(&mut cursor, &mut header).unwrap();
        let first = header.file_checksum;

        // Append more body data and re-finalize.
        let mut buf = cursor.into_inner();
        buf.extend_from_slice(b" more data");
        let mut cursor = Cursor::new(buf);
        finalize_header_with_checksum(&mut cursor, &mut header).unwrap();

        assert_ne!(first, header.file_checksum);
    }
}
