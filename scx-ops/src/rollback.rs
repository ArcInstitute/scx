// Rollback operation: revert to a previous catalog state.

use std::collections::BTreeMap;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};
use scx_format::catalog::{FullCatalog, RootCatalog, RootCatalogEntry};
use scx_format::header::{FileHeader, HEADER_SIZE};
use scx_format::section::SectionType;

use crate::checksum::finalize_header_with_checksum;
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
///
/// Instead of reading from offset to EOF (which could be gigabytes for files
/// with many appends), we first read the fixed catalog header to learn n_entries,
/// then scan entry headers to compute the exact catalog size, reading only
/// what's needed.
fn read_catalog_at(file: &mut (impl Read + Seek), offset: u64) -> Result<(FullCatalog, u64)> {
    let file_len = file.seek(SeekFrom::End(0))?;
    let max_available = file_len.saturating_sub(offset);

    // Catalog header: version(2) + manifest_sequence(8) + prev_catalog_offset(8)
    //                 + n_obs(8) + n_entries(4) = 30 bytes
    // Minimum catalog = 30 + 32 (trailing checksum) = 62 bytes
    const CATALOG_HEADER_SIZE: usize = 2 + 8 + 8 + 8 + 4;
    const TRAILING_CHECKSUM: usize = 32;
    let min_size = CATALOG_HEADER_SIZE + TRAILING_CHECKSUM;

    if (max_available as usize) < min_size {
        return Err(OpsError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "catalog truncated: not enough bytes for catalog header",
        )));
    }

    // Step 1: Read just the fixed catalog header to learn n_entries
    file.seek(SeekFrom::Start(offset))?;
    let mut header_buf = [0u8; CATALOG_HEADER_SIZE];
    file.read_exact(&mut header_buf)?;

    let mut cursor = Cursor::new(&header_buf[..]);
    let _catalog_version = cursor.read_u16::<LittleEndian>()?;
    let _manifest_sequence = cursor.read_u64::<LittleEndian>()?;
    let _prev_catalog_offset = cursor.read_u64::<LittleEndian>()?;
    let _n_obs = cursor.read_u64::<LittleEndian>()?;
    let n_entries = cursor.read_u32::<LittleEndian>()? as usize;

    // Step 2: Scan entry headers to compute total catalog size.
    // Each entry: name_len(2) + name(variable) + offset(8) + length(8)
    //             + type(1) + checksum(32) + stats_len(2) + stats(variable)
    // We read entries incrementally, one at a time, to avoid loading to EOF.
    let mut entries_size: usize = 0;
    for i in 0..n_entries {
        let entry_start = offset + CATALOG_HEADER_SIZE as u64 + entries_size as u64;

        // Read name_len (2 bytes)
        if entry_start + 2 > file_len {
            return Err(OpsError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("catalog truncated at entry {i}: cannot read name_len"),
            )));
        }
        file.seek(SeekFrom::Start(entry_start))?;
        let mut name_len_buf = [0u8; 2];
        file.read_exact(&mut name_len_buf)?;
        let name_len = u16::from_le_bytes(name_len_buf) as usize;

        // Fixed fields after name: offset(8) + length(8) + type(1) + checksum(32) = 49
        let fixed_after_name = 8 + 8 + 1 + 32;
        let stats_len_pos = entry_start + 2 + name_len as u64 + fixed_after_name as u64;

        if stats_len_pos + 2 > file_len {
            return Err(OpsError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("catalog truncated at entry {i}: cannot read stats_len"),
            )));
        }
        file.seek(SeekFrom::Start(stats_len_pos))?;
        let mut stats_len_buf = [0u8; 2];
        file.read_exact(&mut stats_len_buf)?;
        let stats_len = u16::from_le_bytes(stats_len_buf) as usize;

        entries_size += 2 + name_len + fixed_after_name + 2 + stats_len;
    }

    // Step 3: Now read the entire catalog in one go (header + entries + checksum)
    let total_catalog_len = CATALOG_HEADER_SIZE + entries_size + TRAILING_CHECKSUM;
    if offset + total_catalog_len as u64 > file_len {
        return Err(OpsError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "catalog truncated: computed size exceeds file",
        )));
    }

    file.seek(SeekFrom::Start(offset))?;
    let mut buf = vec![0u8; total_catalog_len];
    file.read_exact(&mut buf)?;

    let catalog = FullCatalog::read_from(&mut Cursor::new(&buf), total_catalog_len, true)?;
    Ok((catalog, total_catalog_len as u64))
}

/// Apply a catalog at `catalog_offset` as the active catalog.
fn apply_catalog_at<F: Read + Write + Seek + crate::checksum::SyncAllIfApplicable>(
    file: &mut F,
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

    // Clear front catalog — it references offsets from a cloud-optimized layout
    // that may not match the catalog we're rolling back to. (Finding 4.8)
    header.front_catalog_offset = 0;
    header.front_catalog_length = 0;
    header.clear_front_catalog();

    // Rebuild root catalog
    let root_catalog = build_root_catalog(&catalog);
    let mut root_buf = Vec::new();
    root_catalog.write_to(&mut root_buf)?;
    let root_catalog_length = root_buf.len() as u64;
    root_buf.resize(4096, 0);

    // pwrite root catalog at 256
    file.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
    file.write_all(&root_buf)?;

    // Durability barrier between root catalog and header write (H7).
    file.flush()?;
    file.sync_all_if_applicable()?;

    header.root_catalog_offset = HEADER_SIZE as u64;
    header.root_catalog_length = root_catalog_length;

    // Single-write header finalization (H5 + M16).
    finalize_header_with_checksum(file, &mut header)?;
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
