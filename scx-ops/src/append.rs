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
use scx_format::provenance::{Provenance, ProvenanceEntry};
use scx_format::section::{align_to_8, SectionType};
use scx_format::shard::{BlockIndex, BlockIndexEntry, ShardHeader, SHARD_HEADER_SIZE, SHARD_MAGIC};

use crate::error::{OpsError, Result};
use crate::flock::FileLock;
use crate::rollback::{build_root_catalog_from_full, compute_file_checksum};

/// Append new rows to an existing SCX file.
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
        FullCatalog::read_from(&mut Cursor::new(&buf), fc_length as usize)?
    };

    let n_new_rows = new_indptr.len() - 1;
    if n_new_rows == 0 {
        return Ok(());
    }

    // Validate n_vars match
    let file_n_vars = header.n_vars;
    // We can't directly check n_vars from the data, but we trust the caller
    // provides consistent indices.

    let old_n_obs = header.n_obs;
    let old_n_csr_shards = header.n_csr_shards;
    let old_catalog_offset = header.full_catalog_offset;

    // Read existing obs section for concatenation
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
        batches.next().ok_or_else(|| {
            OpsError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "obs section contains no batches",
            ))
        })??
    };

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

        let shard_idx = old_n_csr_shards + new_shard_entries.len() as u32;
        let shard_name = format!("X_shard_{shard_idx}");
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
            codec_id,
            value_encoding,
            index_dtype_u16,
        )?;

        // Block index (single block)
        let block_index = BlockIndex {
            entries: vec![BlockIndexEntry::new(0, shard_rows as u32, 0, 0, 0, shard_nnz)?],
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
            shard_type: 0,
            codec_id: codec_id as u8,
            value_encoding: value_encoding as u8,
            index_dtype: header.index_dtype,
            reserved_flags: [0; 3],
            n_major: shard_rows as u32,
            n_minor: file_n_vars as u32,
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

        let stats = compute_shard_stats(
            shard_values,
            value_encoding,
            global_row_start,
            shard_rows as u64,
            shard_nnz,
        );

        new_shard_entries.push(FullCatalogEntry {
            name: shard_name,
            offset: shard_global_offset,
            length: section_length,
            section_type: SectionType::CsrShard,
            checksum: section_checksum,
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

    // Build new catalog: old entries (minus old obs, minus old provenance) + new shards + new obs + new provenance
    let mut new_entries: Vec<FullCatalogEntry> = old_catalog
        .entries
        .into_iter()
        .filter(|e| {
            e.section_type != SectionType::ObsMetadata && e.section_type != SectionType::Provenance
        })
        .collect();

    new_entries.extend(new_shard_entries);
    new_entries.push(FullCatalogEntry {
        name: "obs".to_string(),
        offset: new_obs_offset,
        length: new_obs_length,
        section_type: SectionType::ObsMetadata,
        checksum: new_obs_checksum,
        stats: None,
    });
    new_entries.push(FullCatalogEntry {
        name: "provenance".to_string(),
        offset: prov_offset,
        length: prov_length,
        section_type: SectionType::Provenance,
        checksum: prov_checksum,
        stats: None,
    });

    let new_n_obs = old_n_obs + n_new_rows as u64;
    let new_manifest_sequence = header.manifest_sequence + 1;
    let new_catalog = FullCatalog {
        catalog_version: 1,
        manifest_sequence: new_manifest_sequence,
        prev_catalog_offset: old_catalog_offset,
        n_obs: new_n_obs,
        entries: new_entries,
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

    // Update header
    header.n_obs = new_n_obs;
    header.nnz += total_new_nnz;
    header.n_csr_shards = new_catalog
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CsrShard)
        .count() as u32;
    header.full_catalog_offset = new_catalog_offset;
    header.full_catalog_length = new_catalog_length;
    header.manifest_sequence = new_manifest_sequence;
    header.prev_catalog_offset = old_catalog_offset;
    header.root_catalog_offset = HEADER_SIZE as u64;
    header.root_catalog_length = root_catalog_length;

    // Clear front catalog flag (stale after append)
    header.clear_front_catalog();
    header.front_catalog_offset = 0;
    header.front_catalog_length = 0;

    // Write header with zero checksum
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
    Ok(())
}

/// Cast dictionary-encoded columns to their value type to unify (deduplicate)
/// dictionary entries after `concat_batches`.
///
/// Arrow's `concat_batches` concatenates dictionaries without deduplication,
/// which produces invalid categoricals for pandas. Casting dictionary → value
/// type (e.g. Utf8) removes duplicates. Arrow IPC will re-encode them as
/// dictionaries on the next write.
pub(crate) fn unify_dict_columns(batch: &RecordBatch) -> std::result::Result<RecordBatch, arrow::error::ArrowError> {
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
