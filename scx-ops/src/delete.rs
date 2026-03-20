// Delete operation: logical deletion via deletion vectors.

use std::io::{Cursor, Seek, SeekFrom, Write};
use std::path::Path;

use roaring::RoaringBitmap;
use scx_format::catalog::{FullCatalog, FullCatalogEntry};
use scx_format::checksum::blake3_hash;
use scx_format::header::{FileHeader, HEADER_SIZE};
use scx_format::provenance::{Provenance, ProvenanceEntry};
use scx_format::section::{align_to_8, SectionType};
use scx_format::{DeletionVectors, ShardDeletion};

use crate::error::Result;
use crate::flock::FileLock;
use crate::rollback::{build_root_catalog_from_full, compute_file_checksum};

/// Mark cells as logically deleted. Returns the total number of deleted cells
/// (including previously deleted ones).
pub fn mark_deleted(path: &Path, cell_indices: &[u64]) -> Result<u64> {
    let mut lock = FileLock::acquire_exclusive(path)?;

    // Read header
    let mut header = {
        lock.seek(SeekFrom::Start(0))?;
        let mut buf = [0u8; HEADER_SIZE];
        std::io::Read::read_exact(&mut lock, &mut buf)?;
        FileHeader::read_from(&mut Cursor::new(&buf))?
    };

    // Read current full catalog
    let catalog = {
        let fc_offset = header.full_catalog_offset;
        let fc_length = header.full_catalog_length;
        lock.seek(SeekFrom::Start(fc_offset))?;
        let mut buf = vec![0u8; fc_length as usize];
        std::io::Read::read_exact(&mut lock, &mut buf)?;
        FullCatalog::read_from(&mut Cursor::new(&buf), fc_length as usize)?
    };

    // Load existing deletion vectors if present
    let mut dv = if header.has_deletion_vectors() {
        if let Some(dv_entry) = catalog
            .entries
            .iter()
            .find(|e| e.section_type == SectionType::DeletionVectors)
        {
            lock.seek(SeekFrom::Start(dv_entry.offset))?;
            let mut buf = vec![0u8; dv_entry.length as usize];
            std::io::Read::read_exact(&mut lock, &mut buf)?;
            DeletionVectors::read_from(&mut Cursor::new(&buf))?
        } else {
            DeletionVectors::new()
        }
    } else {
        DeletionVectors::new()
    };

    // Get shard stats sorted by row_start to map global indices -> (shard_id, local_row)
    let shards = catalog.shards_sorted();
    let shard_ranges: Vec<(u32, u64, u64)> = shards
        .iter()
        .enumerate()
        .filter_map(|(i, e)| e.stats.as_ref().map(|s| (i as u32, s.row_start, s.row_end)))
        .collect();

    // Map cell indices to per-shard bitmaps
    let mut new_dv = DeletionVectors::new();
    for &global_idx in cell_indices {
        for &(shard_id, row_start, row_end) in &shard_ranges {
            if global_idx >= row_start && global_idx < row_end {
                let local_row = (global_idx - row_start) as u32;
                if let Some(sd) = new_dv.shards.iter_mut().find(|sd| sd.shard_id == shard_id) {
                    sd.bitmap.insert(local_row);
                } else {
                    let mut bm = RoaringBitmap::new();
                    bm.insert(local_row);
                    new_dv.shards.push(ShardDeletion {
                        shard_id,
                        bitmap: bm,
                    });
                }
                break;
            }
        }
    }

    // Merge new deletions into existing
    dv.merge(&new_dv);

    // Serialize DV section
    let mut dv_bytes = Vec::new();
    dv.write_to(&mut dv_bytes)?;

    // Write DV section at EOF
    let file_end = lock.seek(SeekFrom::End(0))?;
    let aligned_offset = align_to_8(file_end);
    let pad = (aligned_offset - file_end) as usize;
    if pad > 0 {
        lock.write_all(&vec![0u8; pad])?;
    }
    let dv_section_offset = aligned_offset;
    lock.write_all(&dv_bytes)?;
    let dv_section_length = dv_bytes.len() as u64;
    let dv_checksum = blake3_hash(&dv_bytes);

    // Build new catalog: replace existing DV entry or add new one
    let old_catalog_offset = header.full_catalog_offset;

    // Read existing provenance BEFORE consuming catalog entries
    let existing_prov_ops = if let Some(prov_entry) = catalog
        .entries
        .iter()
        .find(|e| e.section_type == SectionType::Provenance)
    {
        lock.seek(SeekFrom::Start(prov_entry.offset))?;
        let mut prov_buf = vec![0u8; prov_entry.length as usize];
        std::io::Read::read_exact(&mut lock, &mut prov_buf)?;
        let prov = Provenance::read_from(&mut Cursor::new(&prov_buf))?;
        prov.operations
    } else {
        Vec::new()
    };

    let mut new_entries: Vec<FullCatalogEntry> = catalog
        .entries
        .into_iter()
        .filter(|e| {
            e.section_type != SectionType::DeletionVectors
                && e.section_type != SectionType::Provenance
        })
        .collect();

    new_entries.push(FullCatalogEntry {
        name: "deletion_vectors".to_string(),
        offset: dv_section_offset,
        length: dv_section_length,
        section_type: SectionType::DeletionVectors,
        checksum: dv_checksum,
        stats: None,
    });

    // Write provenance section
    let mut prov_ops = existing_prov_ops;
    prov_ops.push(ProvenanceEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        action: "delete".to_string(),
        tool: "scx-ops 0.1.0".to_string(),
        params_json: format!("{{\"n_cells_deleted\":{}}}", cell_indices.len()),
        input_checksums: vec![],
    });
    let prov = Provenance {
        version: 1,
        operations: prov_ops,
    };
    let mut prov_bytes = Vec::new();
    prov.write_to(&mut prov_bytes)?;

    lock.seek(SeekFrom::End(0))?;
    let prov_write_offset = lock.stream_position()?;
    let prov_aligned = align_to_8(prov_write_offset);
    let prov_pad = (prov_aligned - prov_write_offset) as usize;
    if prov_pad > 0 {
        lock.write_all(&vec![0u8; prov_pad])?;
    }
    let prov_offset = prov_aligned;
    lock.write_all(&prov_bytes)?;
    let prov_length = prov_bytes.len() as u64;
    let prov_checksum = blake3_hash(&prov_bytes);

    new_entries.push(FullCatalogEntry {
        name: "provenance".to_string(),
        offset: prov_offset,
        length: prov_length,
        section_type: SectionType::Provenance,
        checksum: prov_checksum,
        stats: None,
    });

    let new_manifest_sequence = header.manifest_sequence + 1;
    let new_catalog = FullCatalog {
        catalog_version: 1,
        manifest_sequence: new_manifest_sequence,
        prev_catalog_offset: old_catalog_offset,
        n_obs: header.n_obs,
        entries: new_entries,
    };

    // Write new catalog at current EOF
    let catalog_start = align_to_8(lock.seek(SeekFrom::End(0))?);
    let pad2 = (catalog_start - lock.seek(SeekFrom::End(0))?) as usize;
    if pad2 > 0 {
        lock.write_all(&vec![0u8; pad2])?;
    }
    let mut catalog_buf = Vec::new();
    new_catalog.write_to(&mut catalog_buf)?;
    lock.write_all(&catalog_buf)?;
    let new_catalog_length = catalog_buf.len() as u64;

    // Rebuild root catalog
    let root_catalog = build_root_catalog_from_full(&new_catalog);
    let mut root_buf = Vec::new();
    root_catalog.write_to(&mut root_buf)?;
    let root_catalog_length = root_buf.len() as u64;
    root_buf.resize(4096, 0);

    // pwrite root catalog at 256
    lock.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
    lock.write_all(&root_buf)?;

    // Update header
    header.full_catalog_offset = catalog_start;
    header.full_catalog_length = new_catalog_length;
    header.manifest_sequence = new_manifest_sequence;
    header.prev_catalog_offset = old_catalog_offset;
    header.root_catalog_offset = HEADER_SIZE as u64;
    header.root_catalog_length = root_catalog_length;
    header.set_deletion_vectors();

    // Write header with zero checksum first
    header.file_checksum = 0;
    lock.seek(SeekFrom::Start(0))?;
    header.write_to(&mut lock)?;
    lock.flush()?;

    // Compute and write final checksum
    let file_checksum = compute_file_checksum(&mut lock)?;
    header.file_checksum = file_checksum;
    lock.seek(SeekFrom::Start(0))?;
    header.write_to(&mut lock)?;

    lock.sync_all()?;

    Ok(dv.total_deleted())
}
