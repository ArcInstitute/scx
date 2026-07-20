// Delete operation: logical deletion via deletion vectors.

use std::io::{Cursor, Seek, SeekFrom, Write};
use std::path::Path;

use scx_format_io::catalog::{FullCatalog, FullCatalogEntry};
use scx_format_io::checksum::blake3_hash;
use scx_format_io::header::{FileHeader, HEADER_SIZE};
use scx_format_io::provenance::{Provenance, ProvenanceEntry};
use scx_format_io::section::{write_alignment_padding, SectionType};
use scx_format_io::DeletionVectors;

use crate::checksum::finalize_header_with_checksum;
use crate::error::{OpsError, Result};
use crate::flock::FileLock;
use crate::rollback::build_root_catalog_from_full;

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

    // Reject any index >= n_obs before mutating the file. Without this guard
    // such indices silently fall outside the shard range search below and the
    // returned total_deleted omits them, making user mistakes look successful.
    for &idx in cell_indices {
        if idx >= header.n_obs {
            return Err(OpsError::CellIndexOutOfBounds {
                index: idx,
                n_obs: header.n_obs,
            });
        }
    }

    // Read current full catalog
    let catalog = {
        let fc_offset = header.full_catalog_offset;
        let fc_length = header.full_catalog_length;
        lock.seek(SeekFrom::Start(fc_offset))?;
        let mut buf = vec![0u8; fc_length as usize];
        std::io::Read::read_exact(&mut lock, &mut buf)?;
        FullCatalog::read_from(&mut Cursor::new(&buf), fc_length as usize, true)?
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
            DeletionVectors::read_from(&mut Cursor::new(&buf), buf.len())?
        } else {
            DeletionVectors::new()
        }
    } else {
        DeletionVectors::new()
    };

    // Get shard stats sorted by row_start to map global indices -> (shard_id, local_row).
    // Use the sort-order index as shard_id. This is safe because deletion vectors are
    // always stored in the same catalog as the shards they reference, and shards_sorted()
    // is deterministic (sorted by row_start). The compact operation reads DVs and shards
    // from the same catalog, so the mapping is consistent.
    let shards = catalog.shards_sorted();
    let shard_ranges: Vec<(u32, u64, u64)> = shards
        .iter()
        .enumerate()
        .filter_map(|(i, e)| e.stats.as_ref().map(|s| (i as u32, s.row_start, s.row_end)))
        .collect();

    // Map cell indices to per-shard bitmaps.
    // shard_ranges is sorted by row_start, so use binary search (O(n log m))
    // instead of linear scan (O(n × m)).
    let mut new_dv = DeletionVectors::new();
    for &global_idx in cell_indices {
        // Find the first shard whose row_end > global_idx
        let shard_idx = shard_ranges.partition_point(|&(_, _, end)| end <= global_idx);
        if shard_idx < shard_ranges.len() {
            let (shard_id, row_start, _) = shard_ranges[shard_idx];
            if global_idx >= row_start {
                let local_row = (global_idx - row_start) as u32;
                new_dv.shards.entry(shard_id).or_default().insert(local_row);
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
    let pad = write_alignment_padding(&mut *lock, file_end)?;
    let dv_section_offset = file_end + pad as u64;
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
        let prov = Provenance::read_from(&mut Cursor::new(&prov_buf), prov_buf.len())?;
        prov.operations
    } else {
        Vec::new()
    };

    // Delete masks rows via a deletion-vector bitmap; it does NOT rewrite
    // the CSR shards, and it carries any CSC sidecar forward unchanged. The
    // sidecar therefore still matches the CSR content it was built from, so
    // preserve both generations (no bump) to keep the freshness invariant
    // `csc_build_generation == data_generation` intact.
    let src_data_generation = catalog.data_generation;
    let src_csc_build_generation = catalog.csc_build_generation;

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
        modality_id: 0, // deletion vectors operate on the global obs axis
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
        tool: concat!("scx-ops ", env!("CARGO_PKG_VERSION")).to_string(),
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
    let prov_pad = write_alignment_padding(&mut *lock, prov_write_offset)?;
    let prov_offset = prov_write_offset + prov_pad as u64;
    lock.write_all(&prov_bytes)?;
    let prov_length = prov_bytes.len() as u64;
    let prov_checksum = blake3_hash(&prov_bytes);

    new_entries.push(FullCatalogEntry {
        name: "provenance".to_string(),
        offset: prov_offset,
        length: prov_length,
        section_type: SectionType::Provenance,
        checksum: prov_checksum,
        modality_id: 0, // provenance is global
        stats: None,
    });

    let new_manifest_sequence = header.manifest_sequence + 1;
    let new_catalog = FullCatalog {
        catalog_version: scx_format_io::CURRENT_CATALOG_VERSION,
        manifest_sequence: new_manifest_sequence,
        prev_catalog_offset: old_catalog_offset,
        n_obs: header.n_obs,
        entries: new_entries,
        data_generation: src_data_generation,
        csc_build_generation: src_csc_build_generation,
    };

    // Write new catalog at current EOF
    let eof = lock.seek(SeekFrom::End(0))?;
    let pad2 = write_alignment_padding(&mut *lock, eof)?;
    let catalog_start = eof + pad2 as u64;
    let mut catalog_buf = Vec::new();
    new_catalog.write_to(&mut catalog_buf)?;
    lock.write_all(&catalog_buf)?;
    let new_catalog_length = catalog_buf.len() as u64;

    // --- Crash safety barrier ---
    // Flush and fsync all appended data (DV section, provenance, catalog)
    // to durable storage BEFORE updating the header and root catalog.
    // If the process crashes after this point, the old header still points
    // to the old full catalog, so the file remains valid (new data at EOF
    // is harmless orphaned bytes recoverable via prev_catalog_offset chain).
    lock.flush()?;
    lock.sync_all()?;

    // Rebuild root catalog
    let root_catalog = build_root_catalog_from_full(&new_catalog);
    let mut root_buf = Vec::new();
    root_catalog.write_to(&mut root_buf)?;
    let root_catalog_length = root_buf.len() as u64;
    root_buf.resize(4096, 0);

    // pwrite root catalog at 256
    lock.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
    lock.write_all(&root_buf)?;

    // Durability barrier between root catalog and header write (H7).
    lock.flush()?;
    lock.sync_all()?;

    // Update header
    header.full_catalog_offset = catalog_start;
    header.full_catalog_length = new_catalog_length;
    header.manifest_sequence = new_manifest_sequence;
    header.prev_catalog_offset = old_catalog_offset;
    header.root_catalog_offset = HEADER_SIZE as u64;
    header.root_catalog_length = root_catalog_length;
    header.set_deletion_vectors();

    // Single-write header finalization (H5 + M16).
    finalize_header_with_checksum(&mut lock, &mut header)?;

    Ok(dv.total_deleted())
}

#[cfg(test)]
mod tests {
    //! Safety net for the per-modality deletion-vector rewrite (DV wire format
    //! v1 → v2: global-obs-indexed bitmaps keyed by modality, replacing the
    //! flattened positional-shard keying).
    //!
    //! Must-stay-green set the DV v1→v2 rewrite must not regress:
    //!   - `scx-format-io/src/deletion_vectors.rs` unit tests (round_trip,
    //!     merge_vectors, build_keep_mask_translates_local_rows_to_global,
    //!     read_rejects_unknown_version, check_version_rejects_future,
    //!     empty_round_trip, sorted_serialization_byte_stable,
    //!     dv_rejects_oversized_n_shards, dv_rejects_oversized_bitmap_len)
    //!   - `scx-ops/tests/integration.rs`: test_delete_marks_cells,
    //!     test_mark_deleted_rejects_oob_indices, test_delete_filtered_read,
    //!     test_delete_then_compact, test_delete_idempotent
    //!   - `pyscx/tests/test_ops.py`: test_mark_deleted_*,
    //!     test_n_obs_reflects_deletions, test_compact
    //!   - `rscx/tests/testthat/test-ops.R`: delete / compact / rollback
    //!   - conformance `v1_deletion_vectors` validators (Rust + pyscx) plus the
    //!     Phase 0 back-compat anchor
    //!     `conformance_vectors::test_v1_deletion_vectors_backcompat_read_baseline`
    use crate::test_utils::{fixture_multimodal, fixture_multimodal_multishard};
    use scx_format_io::ScxReader;

    /// GREEN lock: on the single-shard-per-modality fixture the common eager
    /// filtered read is already correct today (the v1 `local = idx - row_start`
    /// / `global = row_start + local` arithmetic cancels for single-shard
    /// modalities). The DV v2 rewrite must keep this path correct.
    #[test]
    fn delete_multimodal_eager_read_drops_rows_from_all_modalities() {
        let dir = tempfile::tempdir().unwrap();
        let path = fixture_multimodal(&dir);
        let deleted = [3u64, 7];
        let n = crate::mark_deleted(&path, &deleted).unwrap();
        assert_eq!(n, deleted.len() as u64);

        let reader = ScxReader::open(&path).unwrap();
        let rna_id = reader.modality_id("rna").unwrap();
        let adt_id = reader.modality_id("adt").unwrap();
        for m in [rna_id, adt_id] {
            let csr = reader.read_all_csr_shards_for_filtered(m).unwrap();
            assert_eq!(csr.shape.0, 10, "modality {m}: 12 - 2 deleted rows");
            assert_eq!(csr.nnz(), 20, "modality {m}: 10 rows x 2 nnz");
        }
    }

    /// GREEN lock (multi-shard-per-modality): deleting a global cell must drop
    /// that row from EVERY modality's *eager* filtered read and keep every
    /// modality cell-aligned. Notably this already holds under the v1
    /// positional-shard mapping — even though `shards_sorted()` +
    /// `partition_point` runs over overlapping per-modality ranges, the write
    /// side's `local = idx - row_start` and the read side's
    /// `global = row_start + local` cancel through the *same* flattened list, so
    /// the obs-length keep-mask is reconstructed correctly. The v1 bug is
    /// therefore confined to per-modality DV *resolution* (only one modality's
    /// bitmap is marked) and the engine's modality-scoped-query guard — not this
    /// eager path. The DV v2 rewrite must keep this lock green.
    #[test]
    fn delete_multimodal_multishard_marks_all_modalities() {
        let dir = tempfile::tempdir().unwrap();
        let path = fixture_multimodal_multishard(&dir);
        // Rows chosen to land in non-first shards of each modality.
        let deleted = [3u64, 8, 10];
        let n = crate::mark_deleted(&path, &deleted).unwrap();
        assert_eq!(n, deleted.len() as u64);

        let reader = ScxReader::open(&path).unwrap();
        let rna_id = reader.modality_id("rna").unwrap();
        let adt_id = reader.modality_id("adt").unwrap();
        let expect_rows = 12 - deleted.len();
        for m in [rna_id, adt_id] {
            let csr = reader.read_all_csr_shards_for_filtered(m).unwrap();
            assert_eq!(
                csr.shape.0, expect_rows,
                "modality {m} must drop every deleted row"
            );
            assert_eq!(csr.nnz(), expect_rows * 2, "modality {m} row-alignment");
        }
    }

    /// Current-limitation lock — the concrete behavior this feature removes. A
    /// modality-scoped engine query against a multimodal file that carries
    /// deletion vectors is rejected today with
    /// `EngineError::MultimodalDeletionVectorsUnsupported` (the v1 DV keys are
    /// recorded against the flattened all-modality shard order and cannot be
    /// remapped per modality). Phase 4 removes this guard; this test then flips
    /// to assert the modality query succeeds and returns deletion-filtered rows.
    #[test]
    fn modality_scoped_query_with_deletions_is_currently_guarded() {
        let dir = tempfile::tempdir().unwrap();
        let path = fixture_multimodal(&dir);
        crate::mark_deleted(&path, &[4]).unwrap();

        let rna_id = ScxReader::open(&path).unwrap().modality_id("rna").unwrap();
        let err = scx_engine::QueryPipeline::open_for_modality(&path, rna_id).unwrap_err();
        assert!(
            matches!(
                err,
                scx_engine::EngineError::MultimodalDeletionVectorsUnsupported
            ),
            "expected the modality-DV guard, got {err:?}"
        );
    }
}
