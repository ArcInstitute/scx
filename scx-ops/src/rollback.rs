// Rollback operation: revert to a previous catalog state.

use std::collections::BTreeMap;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};
use scx_format::catalog::{FullCatalog, RootCatalog, RootCatalogEntry};
use scx_format::header::{FileHeader, HEADER_SIZE};
use scx_format::section::SectionType;

use crate::error::{OpsError, Result};
use crate::flock::FileLock;

/// Rollback to the immediately previous catalog.
pub fn rollback(path: &Path) -> Result<()> {
    let mut lock = FileLock::acquire_exclusive(path)?;

    let header = read_header(&mut lock)?;
    if header.prev_catalog_offset == 0 {
        return Err(OpsError::NoPreviousCatalog);
    }

    let prev_offset = header.prev_catalog_offset;
    apply_catalog_at(&mut lock, header, prev_offset)?;
    lock.sync_all()?;
    Ok(())
}

/// Rollback to a specific manifest sequence number by following the
/// prev_catalog_offset chain.
pub fn rollback_to(path: &Path, target_sequence: u64) -> Result<()> {
    let mut lock = FileLock::acquire_exclusive(path)?;

    let header = read_header(&mut lock)?;
    if header.manifest_sequence == target_sequence {
        return Ok(()); // already at target
    }

    // Walk the chain
    let mut catalog_offset = header.prev_catalog_offset;
    loop {
        if catalog_offset == 0 {
            return Err(OpsError::RollbackTargetNotFound(target_sequence));
        }

        let (catalog, _catalog_len) = read_catalog_at(&mut lock, catalog_offset)?;
        if catalog.manifest_sequence == target_sequence {
            apply_catalog_at(&mut lock, header, catalog_offset)?;
            lock.sync_all()?;
            return Ok(());
        }

        // If we've gone past the target (older), give up
        if catalog.manifest_sequence < target_sequence {
            return Err(OpsError::RollbackTargetNotFound(target_sequence));
        }

        catalog_offset = catalog.prev_catalog_offset;
    }
}

fn read_header(file: &mut (impl Read + Seek)) -> Result<FileHeader> {
    file.seek(SeekFrom::Start(0))?;
    let mut buf = [0u8; HEADER_SIZE];
    file.read_exact(&mut buf)?;
    Ok(FileHeader::read_from(&mut Cursor::new(&buf))?)
}

/// Read a FullCatalog at a given offset. Returns (catalog, byte_length).
fn read_catalog_at(file: &mut (impl Read + Seek), offset: u64) -> Result<(FullCatalog, u64)> {
    // We need to figure out the catalog length. The catalog is self-delimiting
    // via its trailing checksum, but we need the length to parse. We can read
    // from offset to EOF.
    file.seek(SeekFrom::Start(offset))?;
    let file_len = file.seek(SeekFrom::End(0))?;
    let catalog_len = file_len - offset;
    file.seek(SeekFrom::Start(offset))?;

    let mut buf = vec![0u8; catalog_len as usize];
    file.read_exact(&mut buf)?;

    // The catalog might not extend to EOF if there are later catalogs.
    // Try progressively smaller sizes. The minimum catalog is the header fields
    // (2+8+8+8+4 = 30 bytes) + checksum (32) = 62 bytes.
    // But for the current catalog (at prev offset), it extends until the next
    // thing was written after it. We'll try the full remaining length first.
    // If that fails, we need a smarter approach.
    //
    // Actually, the full_catalog_length in the header only tracks the *current*
    // catalog. For previous catalogs, we can use the fact that the next catalog
    // or sections start right after. Let's try: if this catalog was at `offset`,
    // find the next known offset after it.
    //
    // Simpler approach: read the header fields to get n_entries, compute
    // expected size, then re-read exactly that.
    let mut cursor = Cursor::new(&buf);
    let _catalog_version = cursor.read_u16::<LittleEndian>()?;
    let _manifest_sequence = cursor.read_u64::<LittleEndian>()?;
    let _prev_catalog_offset = cursor.read_u64::<LittleEndian>()?;
    let _n_obs = cursor.read_u64::<LittleEndian>()?;
    let n_entries = cursor.read_u32::<LittleEndian>()? as usize;

    // Compute expected payload size by scanning through entries
    let header_size = 2 + 8 + 8 + 8 + 4; // catalog header fields
    let mut entry_cursor_pos = cursor.position() as usize;

    for _ in 0..n_entries {
        if entry_cursor_pos + 2 > buf.len() {
            return Err(OpsError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "catalog truncated",
            )));
        }
        let name_len =
            u16::from_le_bytes([buf[entry_cursor_pos], buf[entry_cursor_pos + 1]]) as usize;
        entry_cursor_pos += 2 + name_len; // name_len + name bytes
        entry_cursor_pos += 8 + 8 + 1 + 32; // offset + length + type + checksum
        let stats_len =
            u16::from_le_bytes([buf[entry_cursor_pos], buf[entry_cursor_pos + 1]]) as usize;
        entry_cursor_pos += 2 + stats_len;
    }
    let entry_bytes = entry_cursor_pos - header_size;

    let total_catalog_len = header_size + entry_bytes + 32; // +32 for trailing checksum
    let catalog_buf = &buf[..total_catalog_len];
    let catalog = FullCatalog::read_from(&mut Cursor::new(catalog_buf), total_catalog_len)?;

    Ok((catalog, total_catalog_len as u64))
}

/// Apply a catalog at `catalog_offset` as the active catalog.
fn apply_catalog_at(
    file: &mut (impl Read + Write + Seek),
    mut header: FileHeader,
    catalog_offset: u64,
) -> Result<()> {
    let (catalog, catalog_len) = read_catalog_at(file, catalog_offset)?;

    // Count CSR shards and compute stats from the target catalog
    let mut n_csr_shards = 0u32;
    let mut total_nnz = 0u64;
    let mut has_dv = false;

    for entry in &catalog.entries {
        match entry.section_type {
            SectionType::CsrShard => {
                n_csr_shards += 1;
                if let Some(ref stats) = entry.stats {
                    total_nnz += stats.nnz;
                }
            }
            SectionType::DeletionVectors => {
                has_dv = true;
            }
            _ => {}
        }
    }

    // Update header
    header.full_catalog_offset = catalog_offset;
    header.full_catalog_length = catalog_len;
    header.manifest_sequence = catalog.manifest_sequence;
    header.prev_catalog_offset = catalog.prev_catalog_offset;
    header.n_obs = catalog.n_obs;
    header.n_csr_shards = n_csr_shards;
    header.nnz = total_nnz;
    if has_dv {
        header.set_deletion_vectors();
    } else {
        header.flags &= !(1 << 5);
    }

    // Rebuild root catalog
    let root_catalog = build_root_catalog(&catalog);
    let mut root_buf = Vec::new();
    root_catalog.write_to(&mut root_buf)?;
    let root_catalog_length = root_buf.len() as u64;
    root_buf.resize(4096, 0);

    // pwrite root catalog at 256
    file.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
    file.write_all(&root_buf)?;

    header.root_catalog_offset = HEADER_SIZE as u64;
    header.root_catalog_length = root_catalog_length;

    // Recompute file checksum
    header.file_checksum = 0;
    file.seek(SeekFrom::Start(0))?;
    header.write_to(file)?;
    file.flush()?;

    let file_checksum = compute_file_checksum(file)?;
    header.file_checksum = file_checksum;

    file.seek(SeekFrom::Start(0))?;
    header.write_to(file)?;

    file.flush()?;
    Ok(())
}

pub(crate) fn build_root_catalog(catalog: &FullCatalog) -> RootCatalog {
    let mut groups: BTreeMap<u8, Vec<&scx_format::FullCatalogEntry>> = BTreeMap::new();
    for entry in &catalog.entries {
        groups
            .entry(entry.section_type as u8)
            .or_default()
            .push(entry);
    }

    let mut root_entries = Vec::new();
    for (&group_type, entries) in &groups {
        let first_offset = entries.iter().map(|e| e.offset).min().unwrap_or(0);
        let total_length: u64 = entries.iter().map(|e| e.length).sum();
        let n_sections = entries.len() as u32;
        root_entries.push(RootCatalogEntry {
            group_type,
            first_section_offset: first_offset,
            total_group_length: total_length,
            n_sections,
            summary: [0u8; 32],
        });
    }

    RootCatalog {
        n_section_groups: root_entries.len() as u16,
        entries: root_entries,
    }
}

pub(crate) fn build_root_catalog_from_full(catalog: &FullCatalog) -> RootCatalog {
    build_root_catalog(catalog)
}

pub(crate) fn compute_file_checksum(file: &mut (impl Read + Seek)) -> Result<u64> {
    file.seek(SeekFrom::Start(0))?;
    let mut hasher = blake3::Hasher::new();
    let mut chunk = [0u8; 65536];
    loop {
        let n = file.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        hasher.update(&chunk[..n]);
    }
    let hash = hasher.finalize();
    Ok(u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap()))
}
