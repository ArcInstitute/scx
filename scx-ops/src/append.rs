// Append operation: add new rows to an existing SCX file.

use std::io::{Cursor, Seek, SeekFrom, Write};
use std::num::NonZeroU32;
use std::path::Path;

use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use scx_codec::{CodecId, CodecSelection, ValueEncoding};
use scx_format::catalog::{FullCatalog, FullCatalogEntry};
use scx_format::checksum::{blake3_hash, blake3_truncated_64};
use scx_format::compute_shard_stats;
use scx_format::header::{FileHeader, HEADER_SIZE};
use scx_format::modality::{ModalityTable, ModalityType};
use scx_format::provenance::{Provenance, ProvenanceEntry};
use scx_format::reader::ScxReader;
use scx_format::section::{write_alignment_padding, SectionType};
use scx_format::shard::{
    derive_shard_type, BlockIndex, BlockIndexEntry, ShardHeader, SHARD_HEADER_SIZE, SHARD_MAGIC,
};

use crate::checksum::finalize_header_with_checksum;
use crate::error::{OpsError, Result};
use crate::flock::FileLock;
use crate::rollback::build_root_catalog_from_full;

/// Options controlling how new rows are appended to an SCX file.
///
/// Bundles codec selection, shard sizing, and modality routing into a
/// single struct so callers cannot silently mis-order positional
/// arguments.
#[derive(Debug, Clone)]
pub struct AppendOptions {
    /// Codec selection: `Auto` for per-shard auto-selection, or
    /// `Explicit(codec)` to force a specific codec on every new shard.
    pub codec: CodecSelection,
    /// Target number of rows per CSR shard.  Must be > 0.
    pub shard_target_rows: NonZeroU32,
    /// Modality to append into.  `0` = single-modality / global axis
    /// (legacy behaviour).  On multimodal files, pass the registered
    /// modality id (1..=n_modalities).
    pub modality_id: u8,
}

impl Default for AppendOptions {
    fn default() -> Self {
        Self {
            codec: CodecSelection::Auto,
            // SAFETY: 16384 != 0
            shard_target_rows: NonZeroU32::new(16384).unwrap(),
            modality_id: 0,
        }
    }
}

/// Append new rows to an existing SCX file.
///
/// Uses [`AppendOptions`] to control codec selection, shard sizing,
/// and modality routing.  Set `options.modality_id` to target a
/// specific modality on multimodal files (default `0` = global /
/// single-modality).
pub fn append(
    target_path: &Path,
    new_obs: &RecordBatch,
    new_indptr: &[u64],
    new_indices: &[u32],
    new_values: &[u8],
    value_encoding: ValueEncoding,
    options: &AppendOptions,
) -> Result<()> {
    let (mut lock, prep) = prepare_append(target_path, options.modality_id)?;

    if new_indptr.is_empty() {
        return Ok(());
    }
    let n_new_rows = new_indptr.len() - 1;
    if n_new_rows == 0 {
        return Ok(());
    }

    // Validate CSR shape invariants before touching any on-disk state.
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
    let value_byte_size = value_encoding_byte_width(value_encoding);
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

    if new_obs.num_rows() != n_new_rows {
        return Err(OpsError::VarLengthMismatch {
            expected: n_new_rows,
            found: new_obs.num_rows(),
        });
    }

    if let Some(&max_idx) = new_indices.iter().max() {
        if max_idx as u64 >= prep.target_n_vars {
            return Err(OpsError::IndexOutOfBounds {
                index: max_idx,
                n_vars: prep.target_n_vars,
            });
        }
    }

    // Read existing obs and validate schema equivalence.
    let old_obs = read_existing_obs(&mut lock, &prep.old_catalog)?;
    validate_obs_schema(&old_obs, new_obs)?;

    // Seek to EOF for appending and run the per-chunk write loop.
    let mut write_offset = lock.seek(SeekFrom::End(0))?;
    let mut new_shard_entries: Vec<FullCatalogEntry> = Vec::new();
    let mut total_new_nnz: u64 = 0;
    let mut row_offset = 0usize;

    while row_offset < n_new_rows {
        let shard_rows = std::cmp::min(
            options.shard_target_rows.get() as usize,
            n_new_rows - row_offset,
        );
        let shard_indptr_start = new_indptr[row_offset];
        let shard_indptr_end = new_indptr[row_offset + shard_rows];

        // Extract shard-local indptr (rebased to 0)
        let shard_indptr: Vec<u64> = new_indptr[row_offset..=row_offset + shard_rows]
            .iter()
            .map(|&v| v - shard_indptr_start)
            .collect();

        let idx_start = shard_indptr_start as usize;
        let idx_end = shard_indptr_end as usize;
        let shard_indices = &new_indices[idx_start..idx_end];
        let val_start = idx_start * value_byte_size;
        let val_end = idx_end * value_byte_size;
        let shard_values = &new_values[val_start..val_end];

        let shard_idx = prep.old_per_modality_csr + new_shard_entries.len() as u32;
        let global_row_start = prep.old_n_obs + row_offset as u64;

        let entry = write_csr_chunk(
            &mut lock,
            &mut write_offset,
            &prep,
            &shard_indptr,
            shard_indices,
            shard_values,
            value_encoding,
            options.codec,
            shard_idx,
            global_row_start,
        )?;
        total_new_nnz += entry_nnz(&entry);
        new_shard_entries.push(entry);

        row_offset += shard_rows;
    }

    finalize_append(
        target_path,
        &mut lock,
        prep,
        &old_obs,
        new_obs,
        new_shard_entries,
        total_new_nnz,
        n_new_rows as u64,
        write_offset,
    )
}

/// Streaming SCX → SCX append.
///
/// Decodes one source CSR shard at a time (or copies its raw bytes verbatim
/// when the codec / value encoding / index dtype / per-modality `n_vars`
/// all match the target's expectations) instead of materialising the entire
/// source matrix in host memory.
///
/// `options.modality_id = 0` and `source_modality_id = 0` match the legacy
/// single-modality / global path. For multimodal targets, set
/// `options.modality_id` to the registered modality id of the destination;
/// for multimodal sources, pass the matching `source_modality_id`.
///
/// Semantics: obs is global on both files (rows are global across
/// modalities), so the source's obs is concatenated to the target's obs
/// regardless of the modality_id arguments. CSC sidecars are dropped from
/// the target on append (same as [`append`]).
pub fn append_from_reader(
    target_path: &Path,
    source: &ScxReader,
    options: &AppendOptions,
    source_modality_id: u8,
) -> Result<()> {
    let (mut lock, prep) = prepare_append(target_path, options.modality_id)?;

    // Enumerate the source's CSR shard catalog entries (already sorted by
    // row_start by `csr_shards_for_modality`).
    let source_csr_entries: Vec<FullCatalogEntry> = source
        .catalog()
        .csr_shards_for_modality(source_modality_id)
        .into_iter()
        .cloned()
        .collect();

    if source_csr_entries.is_empty() {
        // Nothing to append — match `append`'s empty-input behaviour.
        return Ok(());
    }

    // Detect global per-append invariants from the first source shard header.
    let first_sh = source.read_shard_header(&source_csr_entries[0])?;
    let value_encoding = ValueEncoding::from_u8(first_sh.value_encoding)
        .ok_or(OpsError::UnknownValueEncoding(first_sh.value_encoding))?;
    let target_index_dtype = prep.header_index_dtype;

    // Validate that every source shard agrees on value_encoding and that
    // n_minor matches target_n_vars (otherwise re-encoding may still work,
    // but the raw-copy fast path is unsafe — handled inline below).
    let mut total_source_rows: u64 = 0;
    for entry in &source_csr_entries {
        let sh = source.read_shard_header(entry)?;
        if sh.value_encoding != first_sh.value_encoding {
            return Err(OpsError::ShapeMismatch {
                detail: format!(
                    "source shard '{}' value_encoding {} differs from first shard's {}",
                    entry.name, sh.value_encoding, first_sh.value_encoding
                ),
            });
        }
        if (sh.n_minor as u64) > prep.target_n_vars {
            return Err(OpsError::IndexOutOfBounds {
                index: sh.n_minor.saturating_sub(1),
                n_vars: prep.target_n_vars,
            });
        }
        total_source_rows += sh.n_major as u64;
    }

    // Read obs from the source (cells are global across modalities).
    let new_obs = source.read_obs().map_err(OpsError::Format)?;
    if new_obs.num_rows() as u64 != total_source_rows {
        return Err(OpsError::VarLengthMismatch {
            expected: total_source_rows as usize,
            found: new_obs.num_rows(),
        });
    }

    // Read existing obs and validate schema equivalence (before writing).
    let old_obs = read_existing_obs(&mut lock, &prep.old_catalog)?;
    validate_obs_schema(&old_obs, &new_obs)?;

    // Per-source-shard streaming loop.
    let mut write_offset = lock.seek(SeekFrom::End(0))?;
    let mut new_shard_entries: Vec<FullCatalogEntry> = Vec::new();
    let mut total_new_nnz: u64 = 0;
    let mut cumulative_row_offset: u64 = 0;

    for entry in &source_csr_entries {
        let sh = source.read_shard_header(entry)?;
        let shard_rows = sh.n_major as usize;
        if shard_rows == 0 {
            continue;
        }
        let global_row_start = prep.old_n_obs + cumulative_row_offset;

        // Raw-copy fast path eligibility.
        let raw_copy_ok = sh.index_dtype == target_index_dtype
            && (sh.n_minor as u64) == prep.target_n_vars
            && sh.n_major <= options.shard_target_rows.get()
            && match options.codec {
                CodecSelection::Auto => true,
                CodecSelection::Explicit(c) => sh.codec_id == c as u8,
            };

        if raw_copy_ok {
            let shard_idx = prep.old_per_modality_csr + new_shard_entries.len() as u32;
            let new_entry = raw_copy_csr_shard(
                &mut lock,
                &mut write_offset,
                &prep,
                source,
                entry,
                &sh,
                value_encoding,
                shard_idx,
                global_row_start,
            )?;
            total_new_nnz += entry_nnz(&new_entry);
            new_shard_entries.push(new_entry);
            cumulative_row_offset += shard_rows as u64;
            continue;
        }

        // Decode the source shard and re-encode (possibly splitting into
        // smaller chunks bounded by `shard_target_rows`). The Vec allocations
        // here are bounded by the source shard, not the whole source file.
        let (ip_i64, ix_i32, val_f32) = source.read_shard_from_entry(entry)?;
        let shard_indptr: Vec<u64> = ip_i64
            .iter()
            .map(|&v| {
                if v < 0 {
                    Err(OpsError::ShapeMismatch {
                        detail: format!("source shard '{}': negative indptr value {v}", entry.name),
                    })
                } else {
                    Ok(v as u64)
                }
            })
            .collect::<Result<Vec<u64>>>()?;
        let shard_indices: Vec<u32> = ix_i32
            .iter()
            .map(|&v| {
                if v < 0 {
                    Err(OpsError::ShapeMismatch {
                        detail: format!("source shard '{}': negative CSR index {v}", entry.name),
                    })
                } else if (v as u64) >= prep.target_n_vars {
                    Err(OpsError::IndexOutOfBounds {
                        index: v as u32,
                        n_vars: prep.target_n_vars,
                    })
                } else {
                    Ok(v as u32)
                }
            })
            .collect::<Result<Vec<u32>>>()?;
        let shard_values = scx_codec::values_to_raw_bytes(&val_f32, value_encoding)?;
        drop(ip_i64);
        drop(ix_i32);
        drop(val_f32);

        let value_byte_size = value_encoding_byte_width(value_encoding);
        let mut row_offset = 0usize;
        while row_offset < shard_rows {
            let chunk_rows = std::cmp::min(
                options.shard_target_rows.get() as usize,
                shard_rows - row_offset,
            );
            let chunk_ip_start = shard_indptr[row_offset];
            let chunk_ip_end = shard_indptr[row_offset + chunk_rows];
            let chunk_indptr: Vec<u64> = shard_indptr[row_offset..=row_offset + chunk_rows]
                .iter()
                .map(|&v| v - chunk_ip_start)
                .collect();
            let chunk_indices = &shard_indices[chunk_ip_start as usize..chunk_ip_end as usize];
            let val_start = chunk_ip_start as usize * value_byte_size;
            let val_end = chunk_ip_end as usize * value_byte_size;
            let chunk_values = &shard_values[val_start..val_end];

            let chunk_global_row_start = global_row_start + row_offset as u64;
            let chunk_shard_idx = prep.old_per_modality_csr + new_shard_entries.len() as u32;
            let new_entry = write_csr_chunk(
                &mut lock,
                &mut write_offset,
                &prep,
                &chunk_indptr,
                chunk_indices,
                chunk_values,
                value_encoding,
                options.codec,
                chunk_shard_idx,
                chunk_global_row_start,
            )?;
            total_new_nnz += entry_nnz(&new_entry);
            new_shard_entries.push(new_entry);
            row_offset += chunk_rows;
        }

        cumulative_row_offset += shard_rows as u64;
    }

    finalize_append(
        target_path,
        &mut lock,
        prep,
        &old_obs,
        &new_obs,
        new_shard_entries,
        total_new_nnz,
        cumulative_row_offset,
        write_offset,
    )
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Shared state captured during the prelude of any append: header, catalog,
/// modality routing, and resolved per-modality `n_vars`. The exclusive
/// `FileLock` is returned alongside this struct so helper functions can
/// take `&mut FileLock` and `&AppendPrep` without aliasing.
struct AppendPrep {
    header: FileHeader,
    header_index_dtype: u8,
    old_catalog: FullCatalog,
    old_n_obs: u64,
    old_catalog_offset: u64,
    modality_table: Option<ModalityTable>,
    modality_id: u8,
    modality_name: Option<String>,
    modality_type: ModalityType,
    target_n_vars: u64,
    old_per_modality_csr: u32,
}

/// Acquire the exclusive lock, read the header + full catalog + modality
/// table, resolve the requested modality, and apply pre-write validation
/// that does not depend on the new data.
fn prepare_append(target_path: &Path, modality_id: u8) -> Result<(FileLock, AppendPrep)> {
    let mut lock = FileLock::acquire_exclusive(target_path)?;

    let header = {
        lock.seek(SeekFrom::Start(0))?;
        let mut buf = [0u8; HEADER_SIZE];
        std::io::Read::read_exact(&mut lock, &mut buf)?;
        FileHeader::read_from(&mut Cursor::new(&buf))?
    };

    let old_catalog = {
        let fc_offset = header.full_catalog_offset;
        let fc_length = header.full_catalog_length;
        lock.seek(SeekFrom::Start(fc_offset))?;
        let mut buf = vec![0u8; fc_length as usize];
        std::io::Read::read_exact(&mut lock, &mut buf)?;
        FullCatalog::read_from(&mut Cursor::new(&buf), fc_length as usize, true)?
    };

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

    let modality_info = modality_table.as_ref().and_then(|t| t.info_of(modality_id));
    let modality_name: Option<String> = modality_info.map(|info| info.name.clone());
    let modality_type: ModalityType = modality_info
        .map(|info| info.modality_type)
        .unwrap_or(ModalityType::Rna);
    let target_n_vars: u64 = modality_info
        .map(|info| info.n_vars)
        .unwrap_or(header.n_vars);

    let old_per_modality_csr = if modality_id == 0 {
        header.n_csr_shards
    } else {
        old_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CsrShard && e.modality_id == modality_id)
            .count() as u32
    };

    if target_n_vars > u32::MAX as u64 {
        return Err(OpsError::Format(scx_format::ScxError::NVarsOverflow(
            target_n_vars,
        )));
    }

    let old_n_obs = header.n_obs;
    let old_catalog_offset = header.full_catalog_offset;
    let header_index_dtype = header.index_dtype;

    // suppress unused mut warning when modality_table happens to be None
    let _ = &mut modality_table;

    Ok((
        lock,
        AppendPrep {
            header,
            header_index_dtype,
            old_catalog,
            old_n_obs,
            old_catalog_offset,
            modality_table,
            modality_id,
            modality_name,
            modality_type,
            target_n_vars,
            old_per_modality_csr,
        },
    ))
}

/// Read the target file's existing obs Arrow IPC section as a `RecordBatch`,
/// downcasting LargeUtf8/LargeBinary to their narrow forms so downstream
/// schema comparison works regardless of when the file was written.
fn read_existing_obs(lock: &mut FileLock, old_catalog: &FullCatalog) -> Result<RecordBatch> {
    let obs_entry = old_catalog.get("obs").ok_or_else(|| {
        OpsError::Format(scx_format::ScxError::SectionNotFound("obs".to_string()))
    })?;
    lock.seek(SeekFrom::Start(obs_entry.offset))?;
    let mut buf = vec![0u8; obs_entry.length as usize];
    std::io::Read::read_exact(&mut *lock, &mut buf)?;
    let cursor = Cursor::new(buf);
    let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)?;
    let mut batches = reader.into_iter();
    let batch = batches.next().ok_or_else(|| {
        OpsError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "obs section contains no batches",
        ))
    })??;
    scx_format::downcast_large_types(&batch).map_err(OpsError::Format)
}

/// Compare two obs batches by column count and per-column (name, effective
/// type). Effective type strips `Dictionary(_, V) → V` because Arrow IPC
/// round-trips dictionaries down to their value type.
fn validate_obs_schema(old_obs: &RecordBatch, new_obs: &RecordBatch) -> Result<()> {
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
    Ok(())
}

#[inline]
fn value_encoding_byte_width(value_encoding: ValueEncoding) -> usize {
    match value_encoding {
        ValueEncoding::Uint8 => 1,
        ValueEncoding::Uint16 | ValueEncoding::Float16 => 2,
        ValueEncoding::Uint32 | ValueEncoding::Float32 => 4,
    }
}

#[inline]
fn entry_nnz(entry: &FullCatalogEntry) -> u64 {
    entry.stats.as_ref().map(|s| s.nnz).unwrap_or(0)
}

/// Encode a single CSR sub-shard, write it (and its block index) to the
/// target file at the current `write_offset` (8-byte aligned), and return
/// the resulting `FullCatalogEntry`. Updates `write_offset` in place.
#[allow(clippy::too_many_arguments)]
fn write_csr_chunk(
    lock: &mut FileLock,
    write_offset: &mut u64,
    prep: &AppendPrep,
    shard_indptr: &[u64],
    shard_indices: &[u32],
    shard_values: &[u8],
    value_encoding: ValueEncoding,
    codec_selection: CodecSelection,
    shard_idx: u32,
    global_row_start: u64,
) -> Result<FullCatalogEntry> {
    let shard_rows = shard_indptr.len().saturating_sub(1);
    let shard_nnz = *shard_indptr.last().unwrap_or(&0);

    let shard_codec = match codec_selection {
        CodecSelection::Auto => {
            scx_format::select_codec_for_modality(shard_values, value_encoding, prep.modality_type)
        }
        CodecSelection::Explicit(c) => c,
    };

    let shard_name = match prep.modality_name.as_deref() {
        Some(mname) => format!("X/{mname}/shard_{shard_idx}"),
        None => format!("X_shard_{shard_idx}"),
    };

    // Pad to 8-byte alignment
    let pad = write_alignment_padding(&mut *lock, *write_offset)?;
    *write_offset += pad as u64;

    let shard_global_offset = *write_offset;
    let index_dtype_u16 = prep.header_index_dtype == 0;

    let encoded = scx_codec::encode_shard(
        shard_indptr,
        shard_indices,
        shard_values,
        shard_codec,
        value_encoding,
        index_dtype_u16,
    )?;

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

    let mut payload = Vec::new();
    payload.extend_from_slice(&encoded.indptr_bytes);
    payload.extend_from_slice(&encoded.indices_bytes);
    payload.extend_from_slice(&encoded.values_bytes);
    payload.extend_from_slice(&block_index_bytes);
    let shard_checksum = blake3_truncated_64(&payload);

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
        index_dtype: prep.header_index_dtype,
        reserved_flags: [0; 3],
        n_major: shard_rows as u32,
        n_minor: prep.target_n_vars as u32,
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

    let mut section_data = Vec::with_capacity(header_buf.len() + payload.len());
    section_data.extend_from_slice(&header_buf);
    section_data.extend_from_slice(&payload);
    let section_checksum = blake3_hash(&section_data);
    let section_length = section_data.len() as u64;

    lock.write_all(&section_data)?;
    *write_offset += section_length;

    let stats = compute_shard_stats(
        shard_values,
        value_encoding,
        scx_format::MajorAxis::Row,
        global_row_start,
        shard_rows as u64,
        prep.target_n_vars,
        shard_nnz,
    );

    Ok(FullCatalogEntry {
        name: shard_name,
        offset: shard_global_offset,
        length: section_length,
        section_type: SectionType::CsrShard,
        checksum: section_checksum,
        modality_id: prep.modality_id,
        stats: Some(stats),
    })
}

/// Raw-copy a source CSR shard's section bytes into the target, patching
/// only `ShardHeader.n_minor` and `ShardHeader.global_offset` (and
/// recomputing the section-level BLAKE3 hash). The payload BLAKE3
/// (`sh.checksum`) is left untouched because the indptr/indices/values/
/// block_index bytes are byte-identical.
#[allow(clippy::too_many_arguments)]
fn raw_copy_csr_shard(
    lock: &mut FileLock,
    write_offset: &mut u64,
    prep: &AppendPrep,
    source: &ScxReader,
    entry: &FullCatalogEntry,
    sh: &ShardHeader,
    value_encoding: ValueEncoding,
    shard_idx: u32,
    global_row_start: u64,
) -> Result<FullCatalogEntry> {
    // Copy the source section into an owned buffer so we can mutate the
    // header. `read_raw_shard_bytes` returns an mmap slice.
    let src_bytes: Vec<u8> = source.read_raw_shard_bytes(entry)?.to_vec();
    if src_bytes.len() < SHARD_HEADER_SIZE {
        return Err(OpsError::Format(scx_format::ScxError::InvalidCatalog(
            format!("source shard '{}' too small for header", entry.name),
        )));
    }

    // Build the new (patched) header with the same fields except n_minor /
    // global_offset.
    let new_sh = ShardHeader {
        magic: sh.magic,
        shard_format_version: sh.shard_format_version,
        shard_type: sh.shard_type,
        codec_id: sh.codec_id,
        value_encoding: sh.value_encoding,
        index_dtype: sh.index_dtype,
        reserved_flags: sh.reserved_flags,
        n_major: sh.n_major,
        n_minor: prep.target_n_vars as u32,
        nnz: sh.nnz,
        global_offset: global_row_start,
        indptr_rel_offset: sh.indptr_rel_offset,
        indptr_length: sh.indptr_length,
        indices_rel_offset: sh.indices_rel_offset,
        indices_length: sh.indices_length,
        values_rel_offset: sh.values_rel_offset,
        values_length: sh.values_length,
        block_index_rel_offset: sh.block_index_rel_offset,
        block_index_length: sh.block_index_length,
        checksum: sh.checksum,
    };

    let mut header_buf = Vec::with_capacity(SHARD_HEADER_SIZE);
    new_sh.write_to(&mut header_buf)?;

    let mut section_data = Vec::with_capacity(src_bytes.len());
    section_data.extend_from_slice(&header_buf);
    section_data.extend_from_slice(&src_bytes[SHARD_HEADER_SIZE..]);
    let section_checksum = blake3_hash(&section_data);
    let section_length = section_data.len() as u64;

    // Pad to 8-byte alignment.
    let pad = write_alignment_padding(&mut *lock, *write_offset)?;
    *write_offset += pad as u64;
    let shard_global_offset = *write_offset;
    lock.write_all(&section_data)?;
    *write_offset += section_length;

    // Reuse the source entry's stats; only the row range is position-dependent.
    // `nnz`, `value_min/max/sum`, `col_start/col_end`, and `column_stats` are
    // invariant under raw copy because `raw_copy_ok` already requires
    // `sh.n_minor == prep.target_n_vars` (so the column extent is preserved).
    // The fallback path decodes only when the source entry is missing stats
    // (not produced by the current writer, but format-permitted).
    let stats = match entry.stats.as_ref() {
        Some(src_stats) => {
            let mut s = src_stats.clone();
            s.row_start = global_row_start;
            s.row_end = global_row_start + sh.n_major as u64;
            s
        }
        None => {
            let codec_id =
                CodecId::from_u8(sh.codec_id).ok_or(OpsError::UnknownCodec(sh.codec_id))?;
            if codec_id == CodecId::None {
                let values_start = sh.values_rel_offset as usize;
                let values_end = values_start + sh.values_length as usize;
                compute_shard_stats(
                    &src_bytes[values_start..values_end],
                    value_encoding,
                    scx_format::MajorAxis::Row,
                    global_row_start,
                    sh.n_major as u64,
                    prep.target_n_vars,
                    sh.nnz,
                )
            } else {
                let (_, _, val_f32) = source.read_shard_from_entry(entry)?;
                let raw = scx_codec::values_to_raw_bytes(&val_f32, value_encoding)?;
                compute_shard_stats(
                    &raw,
                    value_encoding,
                    scx_format::MajorAxis::Row,
                    global_row_start,
                    sh.n_major as u64,
                    prep.target_n_vars,
                    sh.nnz,
                )
            }
        }
    };

    let shard_name = match prep.modality_name.as_deref() {
        Some(mname) => format!("X/{mname}/shard_{shard_idx}"),
        None => format!("X_shard_{shard_idx}"),
    };

    Ok(FullCatalogEntry {
        name: shard_name,
        offset: shard_global_offset,
        length: section_length,
        section_type: SectionType::CsrShard,
        checksum: section_checksum,
        modality_id: prep.modality_id,
        stats: Some(stats),
    })
}

/// Post-shard-loop tail: merge obs, write obs/provenance/(modality table)/
/// catalog, fsync, rebuild root catalog, finalize header with checksum.
#[allow(clippy::too_many_arguments)]
fn finalize_append(
    target_path: &Path,
    lock: &mut FileLock,
    mut prep: AppendPrep,
    old_obs: &RecordBatch,
    new_obs: &RecordBatch,
    new_shard_entries: Vec<FullCatalogEntry>,
    total_new_nnz: u64,
    n_new_rows: u64,
    mut write_offset: u64,
) -> Result<()> {
    let merged_obs = {
        let old_unified = unify_dict_columns(old_obs)?;
        let new_unified = unify_dict_columns(new_obs)?;
        concat_batches(&old_unified.schema(), &[old_unified, new_unified])?
    };
    let merged_obs = scx_format::upcast_to_large_types(&merged_obs).map_err(OpsError::Format)?;
    let obs_ipc_bytes = {
        let mut buf = Vec::new();
        let mut writer =
            arrow::ipc::writer::FileWriter::try_new(&mut buf, merged_obs.schema_ref())?;
        writer.write(&merged_obs)?;
        writer.finish()?;
        buf
    };

    let pad = write_alignment_padding(&mut *lock, write_offset)?;
    write_offset += pad as u64;
    let new_obs_offset = write_offset;
    lock.write_all(&obs_ipc_bytes)?;
    let new_obs_length = obs_ipc_bytes.len() as u64;
    let new_obs_checksum = blake3_hash(&obs_ipc_bytes);

    // Provenance
    let prov_entries = {
        let mut entries = if let Some(prov_entry) = prep
            .old_catalog
            .entries
            .iter()
            .find(|e| e.section_type == SectionType::Provenance)
        {
            lock.seek(SeekFrom::Start(prov_entry.offset))?;
            let mut prov_buf = vec![0u8; prov_entry.length as usize];
            std::io::Read::read_exact(&mut *lock, &mut prov_buf)?;
            let prov = Provenance::read_from(&mut Cursor::new(&prov_buf), prov_buf.len())?;
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

    lock.seek(SeekFrom::End(0))?;
    write_offset = lock.stream_position()?;
    let pad = write_alignment_padding(&mut *lock, write_offset)?;
    write_offset += pad as u64;
    let prov_offset = write_offset;
    lock.write_all(&prov_bytes)?;
    let prov_length = prov_bytes.len() as u64;
    let prov_checksum = blake3_hash(&prov_bytes);
    write_offset += prov_length;

    // Build new catalog. Append always drops every CSC sidecar
    // (single- and multi-modality alike): CSC shard headers stamp
    // `n_minor` from the file-wide `header.n_obs`, so any preserved
    // sidecar becomes stale the moment global `n_obs` bumps. Per-
    // modality CSC preservation is a Phase F+ follow-on (see
    // docs/multimodal.md § append).
    let n_dropped_csc = prep
        .old_catalog
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CscShard)
        .count();
    let had_csc = n_dropped_csc > 0;
    let n_new_csr_shards = new_shard_entries.len() as u32;
    let mut new_entries: Vec<FullCatalogEntry> = prep
        .old_catalog
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
    new_entries.extend(new_shard_entries);
    new_entries.push(FullCatalogEntry {
        name: "obs".to_string(),
        offset: new_obs_offset,
        length: new_obs_length,
        section_type: SectionType::ObsMetadata,
        checksum: new_obs_checksum,
        modality_id: 0,
        stats: None,
    });
    new_entries.push(FullCatalogEntry {
        name: "provenance".to_string(),
        offset: prov_offset,
        length: prov_length,
        section_type: SectionType::Provenance,
        checksum: prov_checksum,
        modality_id: 0,
        stats: None,
    });

    let new_n_obs = prep.old_n_obs + n_new_rows;
    let new_manifest_sequence = prep.header.manifest_sequence + 1;
    let new_catalog = FullCatalog {
        catalog_version: scx_format::CURRENT_CATALOG_VERSION,
        manifest_sequence: new_manifest_sequence,
        prev_catalog_offset: prep.old_catalog_offset,
        n_obs: new_n_obs,
        entries: new_entries,
    };

    let (modality_table_offset, modality_table_length) =
        if let Some(mut table) = prep.modality_table.take() {
            if prep.modality_id != 0 {
                if let Some(info) = table.entries.get_mut((prep.modality_id - 1) as usize) {
                    info.n_csr_shards += n_new_csr_shards;
                    info.nnz += total_new_nnz;
                }
            }
            // Clear HAS_CSC and n_csc_shards on every modality —
            // append always drops the file-wide sidecar (see catalog
            // comment above).
            for info in table.entries.iter_mut() {
                info.n_csc_shards = 0;
                info.flags = scx_format::ModalityFlags::from_bits_truncate(
                    info.flags.bits() & !scx_format::ModalityFlags::HAS_CSC,
                );
            }

            let pad = write_alignment_padding(&mut *lock, write_offset)?;
            write_offset += pad as u64;
            let mt_offset = write_offset;
            let mut mt_buf = Vec::new();
            table.write_to(&mut mt_buf)?;
            lock.write_all(&mt_buf)?;
            let mt_len = mt_buf.len() as u64;
            write_offset += mt_len;
            (mt_offset, mt_len)
        } else {
            (
                prep.header.modality_table_offset,
                prep.header.modality_table_length,
            )
        };

    // Write new catalog
    let pad = write_alignment_padding(&mut *lock, write_offset)?;
    write_offset += pad as u64;
    let new_catalog_offset = write_offset;
    let mut catalog_buf = Vec::new();
    new_catalog.write_to(&mut catalog_buf)?;
    lock.write_all(&catalog_buf)?;
    let new_catalog_length = catalog_buf.len() as u64;

    // Crash safety barrier 1: durable shard / obs / provenance / catalog
    // before any pointer update.
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

    // Crash safety barrier 2: durable root catalog before header update.
    lock.flush()?;
    lock.sync_all()?;

    // Finalize header
    prep.header.n_obs = new_n_obs;
    prep.header.nnz += total_new_nnz;
    prep.header.n_csr_shards = new_catalog
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CsrShard)
        .count() as u32;
    // Append drops every CSC sidecar, so n_csc_shards collapses to 0
    // and HAS_CSC clears. Kept as a count to defend against any future
    // partial-preservation logic re-introduction.
    let remaining_csc = new_catalog
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CscShard)
        .count() as u32;
    prep.header.n_csc_shards = remaining_csc;
    if remaining_csc > 0 {
        prep.header.set_csc();
    } else {
        prep.header.clear_csc();
    }
    prep.header.full_catalog_offset = new_catalog_offset;
    prep.header.full_catalog_length = new_catalog_length;
    prep.header.modality_table_offset = modality_table_offset;
    prep.header.modality_table_length = modality_table_length;
    prep.header.manifest_sequence = new_manifest_sequence;
    prep.header.prev_catalog_offset = prep.old_catalog_offset;
    prep.header.root_catalog_offset = HEADER_SIZE as u64;
    prep.header.root_catalog_length = root_catalog_length;

    prep.header.clear_front_catalog();
    prep.header.front_catalog_offset = 0;
    prep.header.front_catalog_length = 0;

    finalize_header_with_checksum(lock, &mut prep.header)?;
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

    for field in schema.fields() {
        if matches!(field.data_type(), DataType::Dictionary(_, _)) {
            needs_unify = true;
            break;
        }
    }

    if !needs_unify {
        return Ok(batch.clone());
    }

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
