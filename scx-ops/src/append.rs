// Append operation: add new rows to an existing SCX file.

use std::io::{Cursor, Seek, SeekFrom, Write};
use std::path::Path;

use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use scx_codec::{CodecId, ValueEncoding};
use scx_format::catalog::{FullCatalog, FullCatalogEntry};
use scx_format::checksum::{blake3_hash, blake3_truncated_64};
use scx_format::compute_shard_stats;
use scx_format::header::{FileHeader, HEADER_SIZE};
use scx_format::modality::ModalityTable;
use scx_format::provenance::{Provenance, ProvenanceEntry};
use scx_format::section::{align_to_8, SectionType};
use scx_format::shard::{
    derive_shard_type, BlockIndex, BlockIndexEntry, ShardHeader, SHARD_HEADER_SIZE, SHARD_MAGIC,
};

use crate::checksum::finalize_header_with_checksum;
use crate::error::{OpsError, Result};
use crate::flock::FileLock;
use crate::rollback::build_root_catalog_from_full;

/// Append new rows to an existing SCX file.
///
/// Single-modality / global-axis convenience. Equivalent to
/// [`append_for_modality`] with `modality_id = 0`.
#[allow(clippy::too_many_arguments)]
pub fn append(
    target_path: &Path,
    new_obs: &RecordBatch,
    new_indptr: &[u64],
    new_indices: &[u32],
    new_values: &[u8],
    value_encoding: ValueEncoding,
    codec_id: CodecId,
    shard_target_rows: u32,
) -> Result<()> {
    append_for_modality(
        target_path,
        new_obs,
        new_indptr,
        new_indices,
        new_values,
        value_encoding,
        codec_id,
        shard_target_rows,
        0,
    )
}

/// Append new rows to an existing SCX file, stamping new shards with
/// the given `modality_id` (Phase F.3 of MULTIMODAL-SUPPORT.md).
///
/// `modality_id = 0` matches the legacy single-modality / global
/// behaviour. On multimodal files, the caller MUST pass a registered
/// modality id (1..=n_modalities) — the modality table is updated
/// in-place to reflect the new shards' nnz / shard count.
///
/// Note: cells (obs) are global across modalities, so even a
/// per-modality append still updates the file's `n_obs`. The append
/// callers are expected to feed in CSR data indexed against the
/// chosen modality's `n_vars`, not the global `header.n_vars`.
#[allow(clippy::too_many_arguments)]
pub fn append_for_modality(
    target_path: &Path,
    new_obs: &RecordBatch,
    new_indptr: &[u64],
    new_indices: &[u32],
    new_values: &[u8],
    value_encoding: ValueEncoding,
    _codec_id: CodecId,
    shard_target_rows: u32,
    modality_id: u8,
) -> Result<()> {
    let mut lock = FileLock::acquire_exclusive(target_path)?;

    // Read header
    let mut header = {
        lock.seek(SeekFrom::Start(0))?;
        let mut buf = [0u8; HEADER_SIZE];
        std::io::Read::read_exact(&mut lock, &mut buf)?;
        FileHeader::read_from(&mut Cursor::new(&buf))?
    };

    // Read current full catalog
    let old_catalog = {
        let fc_offset = header.full_catalog_offset;
        let fc_length = header.full_catalog_length;
        lock.seek(SeekFrom::Start(fc_offset))?;
        let mut buf = vec![0u8; fc_length as usize];
        std::io::Read::read_exact(&mut lock, &mut buf)?;
        FullCatalog::read_from(&mut Cursor::new(&buf), fc_length as usize, true)?
    };

    if new_indptr.is_empty() {
        return Ok(());
    }
    let n_new_rows = new_indptr.len() - 1;
    if n_new_rows == 0 {
        return Ok(());
    }

    // Validate CSR shape invariants before touching any on-disk state.
    // (These fire before the file has been modified, so the target file is
    // untouched if any validation fails.)
    let expected_nnz = *new_indptr.last().expect("indptr is non-empty") as usize;
    if new_indices.len() != expected_nnz {
        return Err(OpsError::ShapeMismatch {
            detail: format!(
                "indices length {} does not match final indptr entry {}",
                new_indices.len(),
                expected_nnz
            ),
        });
    }
    let value_byte_size = match value_encoding {
        ValueEncoding::Uint8 => 1,
        ValueEncoding::Uint16 | ValueEncoding::Float16 => 2,
        ValueEncoding::Uint32 | ValueEncoding::Float32 => 4,
    };
    if new_values.len() != expected_nnz * value_byte_size {
        return Err(OpsError::ShapeMismatch {
            detail: format!(
                "values length {} does not match nnz {} × byte_width {} = {}",
                new_values.len(),
                expected_nnz,
                value_byte_size,
                expected_nnz * value_byte_size
            ),
        });
    }
    for w in new_indptr.windows(2) {
        if w[0] > w[1] {
            return Err(OpsError::ShapeMismatch {
                detail: "indptr is not monotonically non-decreasing".to_string(),
            });
        }
    }

    // obs schema / length must match the existing obs on disk.
    if new_obs.num_rows() != n_new_rows {
        return Err(OpsError::VarLengthMismatch {
            expected: n_new_rows,
            found: new_obs.num_rows(),
        });
    }

    let old_n_obs = header.n_obs;
    let old_n_csr_shards = header.n_csr_shards;
    let old_catalog_offset = header.full_catalog_offset;

    // Phase F.3: load the modality table on disk (if any) so we can
    // (a) resolve the target modality's name for shard naming, and
    // (b) update its per-modality counts before re-emitting it.
    let mut modality_table = if header.n_modalities > 0
        && header.modality_table_offset != 0
        && header.modality_table_length != 0
    {
        let mt_off = header.modality_table_offset;
        let mt_len = header.modality_table_length as usize;
        lock.seek(SeekFrom::Start(mt_off))?;
        let mut buf = vec![0u8; mt_len];
        std::io::Read::read_exact(&mut lock, &mut buf)?;
        Some(ModalityTable::read_from(&mut Cursor::new(&buf), mt_len)?)
    } else {
        None
    };

    // Resolve the target modality. modality_id == 0 means "global" /
    // legacy single-modality behaviour. > 0 must reference an entry
    // in the modality table.
    if modality_id != 0 {
        let table = modality_table.as_ref().ok_or_else(|| {
            OpsError::Format(scx_format::ScxError::InvalidCatalog(format!(
                "append: target file has no modality table but modality_id={modality_id} \
                 was requested"
            )))
        })?;
        if (modality_id as usize) > table.len() {
            return Err(OpsError::Format(scx_format::ScxError::InvalidCatalog(
                format!(
                    "append: modality_id={modality_id} out of range (file has {} modalities)",
                    table.len()
                ),
            )));
        }
    }
    let modality_name: Option<String> = if modality_id == 0 {
        None
    } else {
        modality_table
            .as_ref()
            .and_then(|t| t.entries.get((modality_id - 1) as usize))
            .map(|info| info.name.clone())
    };

    // Per-modality CSR shard count for naming. Global (modality_id == 0)
    // continues to use the file-wide `header.n_csr_shards`; per-modality
    // appends count only existing shards with that modality_id.
    let old_per_modality_csr = if modality_id == 0 {
        old_n_csr_shards
    } else {
        old_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CsrShard && e.modality_id == modality_id)
            .count() as u32
    };

    // Validate indices are within [0, n_vars). For per-modality
    // append, `n_vars` is the chosen modality's `n_vars`, not the
    // header's (which may be the file-wide max across modalities).
    let target_n_vars: u64 = if modality_id == 0 {
        header.n_vars
    } else {
        modality_table
            .as_ref()
            .and_then(|t| t.entries.get((modality_id - 1) as usize))
            .map(|info| info.n_vars)
            .unwrap_or(header.n_vars)
    };
    if let Some(&max_idx) = new_indices.iter().max() {
        if max_idx as u64 >= target_n_vars {
            return Err(OpsError::IndexOutOfBounds {
                index: max_idx,
                n_vars: target_n_vars,
            });
        }
    }

    // Read existing obs section for concatenation. This bypasses
    // `ScxReader::read_obs`, so apply `downcast_large_types` explicitly
    // — files written after the LargeUtf8 fix store obs as LargeUtf8 on
    // disk, but downstream code (and the schema check below) expects
    // the canonical narrow `Utf8` type.
    let old_obs = {
        let obs_entry = old_catalog.get("obs").ok_or_else(|| {
            OpsError::Format(scx_format::ScxError::SectionNotFound("obs".to_string()))
        })?;
        lock.seek(SeekFrom::Start(obs_entry.offset))?;
        let mut buf = vec![0u8; obs_entry.length as usize];
        std::io::Read::read_exact(&mut lock, &mut buf)?;
        let cursor = Cursor::new(buf);
        let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)?;
        let mut batches = reader.into_iter();
        let batch = batches.next().ok_or_else(|| {
            OpsError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "obs section contains no batches",
            ))
        })??;
        scx_format::downcast_large_types(&batch).map_err(OpsError::Format)?
    };

    // Validate obs schema equivalence between the target file's existing obs
    // and the new batch. `concat_batches` below runs after `unify_dict_columns`
    // strips `Dictionary(_, V) → V` on both sides, so the relevant comparison
    // is over *effective* value types — Arrow IPC round-trips Dictionary columns
    // down to their value type (e.g. `Dictionary(Int8, Utf8)` → `Utf8`), so a
    // strict `a.data_type() == b.data_type()` check would reject a legitimate
    // append where one side came from disk and the other from a fresh
    // AnnData → Arrow conversion. Surfacing the check here gives a clearer
    // diagnostic than the downstream `concat_batches` error and avoids any
    // chance that a partial write precedes it.
    let old_schema = old_obs.schema();
    let new_schema = new_obs.schema();
    if old_schema.fields().len() != new_schema.fields().len() {
        return Err(OpsError::SchemaMismatch {
            detail: format!(
                "obs column count: existing has {}, new has {}",
                old_schema.fields().len(),
                new_schema.fields().len()
            ),
        });
    }
    for (a, b) in old_schema.fields().iter().zip(new_schema.fields().iter()) {
        if a.name() != b.name() || effective_type(a.data_type()) != effective_type(b.data_type()) {
            return Err(OpsError::SchemaMismatch {
                detail: format!(
                    "obs column '{}'({:?}) vs new '{}'({:?})",
                    a.name(),
                    a.data_type(),
                    b.name(),
                    b.data_type()
                ),
            });
        }
    }

    // Seek to EOF for appending
    let mut write_offset = lock.seek(SeekFrom::End(0))?;

    // Shard the new data and write each shard
    let index_dtype_u16 = header.index_dtype == 0;
    let mut new_shard_entries = Vec::new();
    let mut total_new_nnz = 0u64;
    let mut row_offset = 0usize;

    while row_offset < n_new_rows {
        let shard_rows = std::cmp::min(shard_target_rows as usize, n_new_rows - row_offset);
        let shard_indptr_start = new_indptr[row_offset];
        let shard_indptr_end = new_indptr[row_offset + shard_rows];
        let shard_nnz = shard_indptr_end - shard_indptr_start;

        // Extract shard-local indptr (rebased to 0)
        let shard_indptr: Vec<u64> = new_indptr[row_offset..=row_offset + shard_rows]
            .iter()
            .map(|&v| v - shard_indptr_start)
            .collect();

        // Extract shard-local indices
        let idx_start = shard_indptr_start as usize;
        let idx_end = shard_indptr_end as usize;
        let shard_indices = &new_indices[idx_start..idx_end];

        // Extract shard-local values
        let value_byte_size = match value_encoding {
            ValueEncoding::Uint8 => 1,
            ValueEncoding::Uint16 | ValueEncoding::Float16 => 2,
            ValueEncoding::Uint32 | ValueEncoding::Float32 => 4,
        };
        let val_start = idx_start * value_byte_size;
        let val_end = idx_end * value_byte_size;
        let shard_values = &new_values[val_start..val_end];

        // Per-shard codec selection (auto-select optimal codec for this shard's data)
        let shard_codec = scx_format::select_codec(shard_values, value_encoding);

        let shard_idx = old_per_modality_csr + new_shard_entries.len() as u32;
        // Per-modality shards follow the writer's `X/{name}/shard_{i}`
        // convention (writer.rs::write_csr_shard_for); global / legacy
        // entries keep the flat `X_shard_{i}` naming.
        let shard_name = match modality_name.as_deref() {
            Some(mname) => format!("X/{mname}/shard_{shard_idx}"),
            None => format!("X_shard_{shard_idx}"),
        };
        let global_row_start = old_n_obs + row_offset as u64;

        // Pad to 8-byte alignment
        let aligned = align_to_8(write_offset);
        let pad = (aligned - write_offset) as usize;
        if pad > 0 {
            lock.write_all(&vec![0u8; pad])?;
            write_offset = aligned;
        }

        let shard_global_offset = write_offset;

        // Encode
        let encoded = scx_codec::encode_shard(
            &shard_indptr,
            shard_indices,
            shard_values,
            shard_codec,
            value_encoding,
            index_dtype_u16,
        )?;

        // Block index (single block)
        let block_index = BlockIndex {
            entries: vec![BlockIndexEntry::new(
                0,
                shard_rows as u32,
                0,
                0,
                0,
                shard_nnz,
            )?],
        };
        let mut block_index_bytes = Vec::new();
        block_index.write_to(&mut block_index_bytes)?;

        // Shard checksum
        let mut payload = Vec::new();
        payload.extend_from_slice(&encoded.indptr_bytes);
        payload.extend_from_slice(&encoded.indices_bytes);
        payload.extend_from_slice(&encoded.values_bytes);
        payload.extend_from_slice(&block_index_bytes);
        let shard_checksum = blake3_truncated_64(&payload);

        // Build shard header
        let indptr_rel_offset = SHARD_HEADER_SIZE as u32;
        let indptr_length = encoded.indptr_bytes.len() as u32;
        let indices_rel_offset = indptr_rel_offset + indptr_length;
        let indices_length = encoded.indices_bytes.len() as u32;
        let values_rel_offset = indices_rel_offset + indices_length;
        let values_length = encoded.values_bytes.len() as u32;
        let block_index_rel_offset = values_rel_offset + values_length;
        let block_index_length = block_index_bytes.len() as u32;

        let sh = ShardHeader {
            magic: SHARD_MAGIC,
            shard_format_version: 1,
            shard_type: derive_shard_type(SectionType::CsrShard),
            codec_id: shard_codec as u8,
            value_encoding: value_encoding as u8,
            index_dtype: header.index_dtype,
            reserved_flags: [0; 3],
            n_major: shard_rows as u32,
            // PR #68: use the target modality's n_vars (resolved
            // above) rather than `header.n_vars` (which is the
            // file-wide max across modalities). Required for v2
            // multimodal files; identical to the legacy single-
            // modality path where `target_n_vars == header.n_vars`.
            n_minor: target_n_vars as u32,
            nnz: shard_nnz,
            global_offset: global_row_start,
            indptr_rel_offset,
            indptr_length,
            indices_rel_offset,
            indices_length,
            values_rel_offset,
            values_length,
            block_index_rel_offset,
            block_index_length,
            checksum: shard_checksum,
        };

        let mut header_buf = Vec::with_capacity(SHARD_HEADER_SIZE);
        sh.write_to(&mut header_buf)?;

        // Full section data
        let mut section_data = Vec::new();
        section_data.extend_from_slice(&header_buf);
        section_data.extend_from_slice(&payload);
        let section_checksum = blake3_hash(&section_data);
        let section_length = section_data.len() as u64;

        lock.write_all(&section_data)?;
        write_offset += section_length;

        // append.rs always emits row-major CSR shards. Use the
        // target modality's n_vars (PR #68) — see ShardHeader.n_minor
        // above for rationale.
        let stats = compute_shard_stats(
            shard_values,
            value_encoding,
            scx_format::MajorAxis::Row,
            global_row_start,
            shard_rows as u64,
            target_n_vars,
            shard_nnz,
        );

        new_shard_entries.push(FullCatalogEntry {
            name: shard_name,
            offset: shard_global_offset,
            length: section_length,
            section_type: SectionType::CsrShard,
            checksum: section_checksum,
            // Phase F.3: stamp shards with the chosen modality id (0 =
            // global / legacy single-modality).
            modality_id,
            stats: Some(stats),
        });

        total_new_nnz += shard_nnz;
        row_offset += shard_rows;
    }

    // Concatenate old + new obs and write as new obs section.
    // Both batches may have dictionary-encoded columns with overlapping
    // categories. Unify both to non-dictionary types first, then concat.
    let merged_obs = {
        let old_unified = unify_dict_columns(&old_obs)?;
        let new_unified = unify_dict_columns(new_obs)?;
        concat_batches(&old_unified.schema(), &[old_unified, new_unified])?
    };
    // Inline write bypasses `ScxWriter::write_arrow_ipc`, so upcast
    // explicitly: obs columns may exceed Arrow IPC's 32-bit offset
    // limit at multi-million-cell scale and need 64-bit `LargeUtf8` /
    // `LargeBinary` offsets on disk.
    let merged_obs = scx_format::upcast_to_large_types(&merged_obs).map_err(OpsError::Format)?;
    let obs_ipc_bytes = {
        let mut buf = Vec::new();
        let mut writer =
            arrow::ipc::writer::FileWriter::try_new(&mut buf, merged_obs.schema_ref())?;
        writer.write(&merged_obs)?;
        writer.finish()?;
        buf
    };

    // Write new obs section
    let aligned = align_to_8(write_offset);
    let pad = (aligned - write_offset) as usize;
    if pad > 0 {
        lock.write_all(&vec![0u8; pad])?;
        write_offset = aligned;
    }
    let new_obs_offset = write_offset;
    lock.write_all(&obs_ipc_bytes)?;
    let new_obs_length = obs_ipc_bytes.len() as u64;
    let new_obs_checksum = blake3_hash(&obs_ipc_bytes);

    // Write provenance section (read existing, append entry, write)
    let prov_entries = {
        let mut entries = if let Some(prov_entry) = old_catalog
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
        entries.push(ProvenanceEntry {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            action: "append".to_string(),
            tool: "scx-ops 0.1.0".to_string(),
            params_json: format!("{{\"n_new_rows\":{n_new_rows}}}"),
            input_checksums: vec![],
        });
        entries
    };
    let prov = Provenance {
        version: 1,
        operations: prov_entries,
    };
    let mut prov_bytes = Vec::new();
    prov.write_to(&mut prov_bytes)?;

    // Seek back to EOF to write provenance
    lock.seek(SeekFrom::End(0))?;
    write_offset = lock.stream_position()?;
    let prov_aligned = align_to_8(write_offset);
    let pad = (prov_aligned - write_offset) as usize;
    if pad > 0 {
        lock.write_all(&vec![0u8; pad])?;
        write_offset = prov_aligned;
    }
    let prov_offset = write_offset;
    lock.write_all(&prov_bytes)?;
    let prov_length = prov_bytes.len() as u64;
    let prov_checksum = blake3_hash(&prov_bytes);
    write_offset += prov_length;

    // Build new catalog: old entries (minus old obs, minus old provenance,
    // minus stale CSC shards) + new shards + new obs + new provenance.
    //
    // CSC sidecars index global rows: appending rows shifts the row space
    // but the on-disk CSC `indices` arrays still reference the old row
    // count, so they must be dropped. Caller can opt back in via
    // `--rebuild-csc` in the CLI.
    let had_csc = header.has_csc();
    let n_dropped_csc = old_catalog
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CscShard)
        .count();
    let mut new_entries: Vec<FullCatalogEntry> = old_catalog
        .entries
        .into_iter()
        .filter(|e| {
            e.section_type != SectionType::ObsMetadata
                && e.section_type != SectionType::Provenance
                && e.section_type != SectionType::CscShard
        })
        .collect();
    if had_csc {
        log::warn!(
            "append dropped {n_dropped_csc} CSC shards from {target}: \
             rerun `scx build-csc` (or pass --rebuild-csc) to restore the \
             column-major sidecar",
            target = target_path.display()
        );
    }

    let n_new_csr_shards = new_shard_entries.len() as u32;
    new_entries.extend(new_shard_entries);
    new_entries.push(FullCatalogEntry {
        name: "obs".to_string(),
        offset: new_obs_offset,
        length: new_obs_length,
        section_type: SectionType::ObsMetadata,
        checksum: new_obs_checksum,
        modality_id: 0, // obs is shared across modalities (global)
        stats: None,
    });
    new_entries.push(FullCatalogEntry {
        name: "provenance".to_string(),
        offset: prov_offset,
        length: prov_length,
        section_type: SectionType::Provenance,
        checksum: prov_checksum,
        modality_id: 0, // provenance is global
        stats: None,
    });

    let new_n_obs = old_n_obs + n_new_rows as u64;
    let new_manifest_sequence = header.manifest_sequence + 1;
    let new_catalog = FullCatalog {
        catalog_version: scx_format::CURRENT_CATALOG_VERSION,
        manifest_sequence: new_manifest_sequence,
        prev_catalog_offset: old_catalog_offset,
        n_obs: new_n_obs,
        entries: new_entries,
    };

    // Phase F.3: re-emit the modality table at a fresh EOF location
    // (alongside the catalog) before the catalog so the header can
    // be updated atomically. Per-modality counts:
    //   - the target modality gets `n_csr_shards += new shards` and
    //     `nnz += total_new_nnz`
    //   - every modality drops its CSC sidecar count + HAS_CSC flag
    //     because the file-wide CSC drop applies uniformly (Phase
    //     F.3 todo: per-modality CSC preservation when only one
    //     modality is being appended into requires per-modality
    //     row-axis decoupling — the writer already writes shared
    //     obs, so for now match the file-wide drop).
    let (modality_table_offset, modality_table_length) =
        if let Some(mut table) = modality_table.take() {
            if modality_id != 0 {
                if let Some(info) = table.entries.get_mut((modality_id - 1) as usize) {
                    info.n_csr_shards += n_new_csr_shards;
                    info.nnz += total_new_nnz;
                }
            }
            // CSC sidecars are dropped file-wide on append (matches the
            // global header.clear_csc() / n_csc_shards = 0 below). Clear
            // every modality's CSC marker to keep the modality table in
            // sync with reality.
            for info in table.entries.iter_mut() {
                info.n_csc_shards = 0;
                info.flags = scx_format::ModalityFlags::from_bits_truncate(
                    info.flags.bits() & !scx_format::ModalityFlags::HAS_CSC,
                );
            }

            let mt_aligned = align_to_8(write_offset);
            let pad = (mt_aligned - write_offset) as usize;
            if pad > 0 {
                lock.write_all(&vec![0u8; pad])?;
                write_offset = mt_aligned;
            }
            let mut mt_buf = Vec::new();
            table.write_to(&mut mt_buf)?;
            lock.write_all(&mt_buf)?;
            let mt_len = mt_buf.len() as u64;
            write_offset += mt_len;
            (mt_aligned, mt_len)
        } else {
            (header.modality_table_offset, header.modality_table_length)
        };

    // Write new catalog
    let catalog_aligned = align_to_8(write_offset);
    let pad = (catalog_aligned - write_offset) as usize;
    if pad > 0 {
        lock.write_all(&vec![0u8; pad])?;
    }
    let new_catalog_offset = catalog_aligned;
    let mut catalog_buf = Vec::new();
    new_catalog.write_to(&mut catalog_buf)?;
    lock.write_all(&catalog_buf)?;
    let new_catalog_length = catalog_buf.len() as u64;

    // --- Crash safety barrier ---
    // Flush and fsync all appended data (shards, obs, provenance, catalog)
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

    lock.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
    lock.write_all(&root_buf)?;

    // Durability barrier between root catalog and header. Without this,
    // a power loss between the two writes could leave the new root catalog
    // on disk while the header still references the previous one — not
    // fatal (old root catalog data is still at its previous offset), but
    // the file's on-disk root catalog would silently go stale relative to
    // the header. (Finding H7.)
    lock.flush()?;
    lock.sync_all()?;

    // Update header
    header.n_obs = new_n_obs;
    header.nnz += total_new_nnz;
    header.n_csr_shards = new_catalog
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CsrShard)
        .count() as u32;
    // CSC sidecars were filtered out of `new_entries` above; reflect
    // that in the header's count + flag bit so readers don't try to
    // load shards that are no longer in the catalog.
    header.n_csc_shards = 0;
    header.clear_csc();
    header.full_catalog_offset = new_catalog_offset;
    header.full_catalog_length = new_catalog_length;
    header.modality_table_offset = modality_table_offset;
    header.modality_table_length = modality_table_length;
    header.manifest_sequence = new_manifest_sequence;
    header.prev_catalog_offset = old_catalog_offset;
    header.root_catalog_offset = HEADER_SIZE as u64;
    header.root_catalog_length = root_catalog_length;

    // Clear front catalog flag (stale after append)
    header.clear_front_catalog();
    header.front_catalog_offset = 0;
    header.front_catalog_length = 0;

    // Single-write header finalization (H5 + M16) — avoids the crash window
    // where a zero-checksum header could be the durable on-disk state.
    finalize_header_with_checksum(&mut lock, &mut header)?;
    Ok(())
}

/// Return the effective (dictionary-stripped) value type of a `DataType`.
/// Mirrors the behaviour of [`unify_dict_columns`] so the append-time schema
/// check can compare across `Dictionary(_, V)` ↔ `V` asymmetries introduced
/// by Arrow IPC round-trips.
fn effective_type(dt: &arrow::datatypes::DataType) -> &arrow::datatypes::DataType {
    match dt {
        arrow::datatypes::DataType::Dictionary(_, value_type) => value_type,
        other => other,
    }
}

/// Cast dictionary-encoded columns to their value type to unify (deduplicate)
/// dictionary entries after `concat_batches`.
///
/// Arrow's `concat_batches` concatenates dictionaries without deduplication,
/// which produces invalid categoricals for pandas. Casting dictionary → value
/// type (e.g. Utf8) removes duplicates. Arrow IPC will re-encode them as
/// dictionaries on the next write.
pub(crate) fn unify_dict_columns(
    batch: &RecordBatch,
) -> std::result::Result<RecordBatch, arrow::error::ArrowError> {
    use arrow::datatypes::DataType;

    let schema = batch.schema();
    let mut needs_unify = false;

    // Check if any columns are dictionary-encoded
    for field in schema.fields() {
        if matches!(field.data_type(), DataType::Dictionary(_, _)) {
            needs_unify = true;
            break;
        }
    }

    if !needs_unify {
        return Ok(batch.clone());
    }

    // Build new columns, casting dictionaries to their value type
    let mut new_fields = Vec::with_capacity(schema.fields().len());
    let mut new_columns = Vec::with_capacity(batch.num_columns());

    for (i, field) in schema.fields().iter().enumerate() {
        let col = batch.column(i);
        if let DataType::Dictionary(_, value_type) = field.data_type() {
            let cast_col = arrow::compute::cast(col, value_type)?;
            new_fields.push(arrow::datatypes::Field::new(
                field.name(),
                value_type.as_ref().clone(),
                field.is_nullable(),
            ));
            new_columns.push(cast_col);
        } else {
            new_fields.push(field.as_ref().clone());
            new_columns.push(col.clone());
        }
    }

    let new_schema = arrow::datatypes::Schema::new(new_fields);
    RecordBatch::try_new(std::sync::Arc::new(new_schema), new_columns)
}
