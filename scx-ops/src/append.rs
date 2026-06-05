// Append operation: add new rows to an existing SCX file.

use std::io::{Cursor, Seek, SeekFrom, Write};
use std::num::NonZeroU32;
use std::path::Path;

use arrow::array::RecordBatch;
use scx_codec::{CodecId, CodecSelection, ValueEncoding};
use scx_engine::ConversionPredicateIndexOptions;
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
use scx_format::writer::ScxWriter;

use crate::checksum::finalize_header_with_checksum;
use crate::error::{OpsError, Result};
use crate::flock::FileLock;
use crate::predicate_index::{
    requested_columns, user_wants_index, validate_forced_columns, PredicateIndexBuildSummary,
};
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
///
/// Drops any existing predicate-index sections on the target whose
/// per-shard row ranges no longer cover the post-append obs (i.e. the
/// stale-index problem on appended rows is preserved as-is). Use
/// [`append_with_index_options`] to rebuild the predicate index in the
/// same pass.
pub fn append(
    target_path: &Path,
    new_obs: &RecordBatch,
    new_indptr: &[u64],
    new_indices: &[u32],
    new_values: &[u8],
    value_encoding: ValueEncoding,
    options: &AppendOptions,
) -> Result<()> {
    // See `merge` for the `index_auto_threshold = 0` sentinel rationale.
    append_with_index_options(
        target_path,
        new_obs,
        new_indptr,
        new_indices,
        new_values,
        value_encoding,
        options,
        &ConversionPredicateIndexOptions {
            index_obs: Vec::new(),
            index_var: Vec::new(),
            index_preset: None,
            index_auto_threshold: 0,
        },
    )
    .map(|_| ())
}

/// Append new rows and optionally rebuild predicate indexes on the
/// updated target. The rebuild walks the unified pre- + post-append
/// obs RecordBatch + the unified per-output-shard row ranges so the
/// resulting index covers every row. Old `ObsPredicateIndex` /
/// `VarPredicateIndex` sections are filtered out of the catalog when a
/// rebuild is requested; otherwise they are left in place (matching
/// pre-fix behaviour).
///
/// Multimodal targets skip the predicate-index write — the engine
/// read-side ignores `modality_id` on predicate-index sections — and
/// surface the request via `summary.multimodal_skip` so the caller can
/// emit `PredicateIndexSkippedMultimodal`.
#[allow(clippy::too_many_arguments)]
pub fn append_with_index_options(
    target_path: &Path,
    new_obs: &RecordBatch,
    new_indptr: &[u64],
    new_indices: &[u32],
    new_values: &[u8],
    value_encoding: ValueEncoding,
    options: &AppendOptions,
    index_options: &ConversionPredicateIndexOptions,
) -> Result<PredicateIndexBuildSummary> {
    let (mut lock, prep) = prepare_append(target_path, options.modality_id)?;

    if new_indptr.is_empty() {
        return Ok(PredicateIndexBuildSummary::skipped());
    }
    let n_new_rows = new_indptr.len() - 1;
    if n_new_rows == 0 {
        return Ok(PredicateIndexBuildSummary::skipped());
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

        let shard_idx = next_shard_idx(prep.old_per_modality_csr, new_shard_entries.len())?;
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
        index_options,
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
///
/// Drops any existing predicate-index sections as-is (no rebuild). Use
/// [`append_from_reader_with_index_options`] to rebuild the predicate
/// index against the post-append target.
pub fn append_from_reader(
    target_path: &Path,
    source: &ScxReader,
    options: &AppendOptions,
    source_modality_id: u8,
) -> Result<()> {
    // See `merge` for the `index_auto_threshold = 0` sentinel rationale.
    append_from_reader_with_index_options(
        target_path,
        source,
        options,
        source_modality_id,
        &ConversionPredicateIndexOptions {
            index_obs: Vec::new(),
            index_var: Vec::new(),
            index_preset: None,
            index_auto_threshold: 0,
        },
    )
    .map(|_| ())
}

/// `append_from_reader` with predicate-index rebuild knobs. See
/// [`append_with_index_options`] for the semantics shared with the
/// in-memory append path.
pub fn append_from_reader_with_index_options(
    target_path: &Path,
    source: &ScxReader,
    options: &AppendOptions,
    source_modality_id: u8,
    index_options: &ConversionPredicateIndexOptions,
) -> Result<PredicateIndexBuildSummary> {
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
        return Ok(PredicateIndexBuildSummary::skipped());
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

    // Read obs from the source (cells are global across modalities,
    // even in v2 multimodal files — every modality covers the global
    // obs row axis). Compare row count against the source's declared
    // global `n_obs`, not the per-modality CSR shard row sum: those
    // happen to coincide for any modality whose shards span the full
    // axis, but the global `n_obs` is the authoritative invariant the
    // append-time validation should enforce.
    let new_obs = source.read_obs().map_err(OpsError::Format)?;
    if new_obs.num_rows() as u64 != source.n_obs() {
        return Err(OpsError::VarLengthMismatch {
            expected: source.n_obs() as usize,
            found: new_obs.num_rows(),
        });
    }
    // The per-modality CSR shard sum still needs to equal global
    // `n_obs` for the convert-on-append row-bookkeeping to be sound;
    // surface a clear error if a malformed source violates this.
    if total_source_rows != source.n_obs() {
        return Err(OpsError::ShapeMismatch {
            detail: format!(
                "source modality {source_modality_id} CSR shards cover {total_source_rows} \
                 rows but source declares n_obs={}; refuse to append from an inconsistent file",
                source.n_obs()
            ),
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
            let shard_idx = next_shard_idx(prep.old_per_modality_csr, new_shard_entries.len())?;
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
            let chunk_shard_idx =
                next_shard_idx(prep.old_per_modality_csr, new_shard_entries.len())?;
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
        index_options,
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

/// Read an existing obs payload as a single `RecordBatch`, handling
/// both legacy single-section [`SectionType::ObsMetadata`] files and
/// Phase 2 row-sharded [`SectionType::ObsMetadataShard`] files. The
/// sharded path concatenates shards in `shard_idx` order, matching
/// what [`scx_format::ScxReader::read_obs`] returns at query time —
/// but goes through the lock-held file handle instead of the mmap
/// `ScxReader` because the append path already holds the write lock
/// and can't open a second reader concurrently. Used by `append`
/// (which needs the full pre-existing obs in memory for the schema
/// check + the convert-on-append rewrite to `ObsMetadataShard` shard
/// 0).
///
/// Memory cost: O(old obs size). Same as today's pre-Phase-2 path —
/// the streaming win shows up in `finalize_append` where new obs is
/// no longer concatenated with old obs.
fn read_existing_obs(lock: &mut FileLock, old_catalog: &FullCatalog) -> Result<RecordBatch> {
    let shard_entries: Vec<(u32, &FullCatalogEntry)> = old_catalog
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::ObsMetadataShard)
        .filter_map(|e| {
            let suffix = e.name.strip_prefix("obs_metadata/shard_")?;
            let idx: u32 = suffix.parse().ok()?;
            Some((idx, e))
        })
        .collect();
    if shard_entries.is_empty() {
        return read_existing_arrow_ipc_section(lock, old_catalog, "obs");
    }
    let mut shards: Vec<(u32, &FullCatalogEntry)> = shard_entries;
    shards.sort_by_key(|(idx, _)| *idx);
    // Decode each shard and force the wide encoding before concat —
    // mirrors `ScxReader::read_sharded_layout_by_prefix`. Concatenating
    // narrow-offset batches would re-trigger Arrow's `Offset overflow
    // error` once the cumulative per-column string payload exceeds
    // `i32::MAX`, which is exactly the failure mode the streaming
    // merge-write path eliminated.
    let mut batches: Vec<RecordBatch> = Vec::with_capacity(shards.len());
    for (_, entry) in &shards {
        lock.seek(SeekFrom::Start(entry.offset))?;
        let mut buf = vec![0u8; entry.length as usize];
        std::io::Read::read_exact(&mut *lock, &mut buf)?;
        let cursor = Cursor::new(buf);
        let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)?;
        let mut iter = reader.into_iter();
        let batch = iter.next().ok_or_else(|| {
            OpsError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("{} contains no batches", entry.name),
            ))
        })??;
        batches.push(scx_format::upcast_to_large_types(&batch).map_err(OpsError::Format)?);
    }
    let wide_schema = batches[0].schema();
    let concatenated =
        arrow::compute::concat_batches(&wide_schema, batches.iter()).map_err(OpsError::Arrow)?;
    // Narrow back to `Utf8` / `Binary` for columns whose combined
    // offsets fit; columns above `i32::MAX` stay wide so the >2 GB
    // append case still reads cleanly.
    let narrowed = scx_format::downcast_large_types(&concatenated).map_err(OpsError::Format)?;
    // Strip per-shard schema metadata so the schema matches what
    // `ScxReader::read_obs` returns at query time.
    let narrowed_schema = narrowed.schema();
    let mut clean_metadata = narrowed_schema.metadata().clone();
    clean_metadata.remove("shard_idx");
    clean_metadata.remove("row_start");
    clean_metadata.remove("n_shard_rows");
    let clean_schema = std::sync::Arc::new(arrow::datatypes::Schema::new_with_metadata(
        narrowed_schema.fields().clone(),
        clean_metadata,
    ));
    Ok(RecordBatch::try_new(
        clean_schema,
        narrowed.columns().to_vec(),
    )?)
}

/// Read an existing Arrow IPC metadata section (`"obs"` or `"var"`) back
/// into a `RecordBatch`, downcasting `LargeUtf8` / `LargeBinary` to their
/// narrow forms so downstream schema comparison works regardless of when
/// the file was written.
fn read_existing_arrow_ipc_section(
    lock: &mut FileLock,
    old_catalog: &FullCatalog,
    section_name: &str,
) -> Result<RecordBatch> {
    let entry = old_catalog.get(section_name).ok_or_else(|| {
        OpsError::Format(scx_format::ScxError::SectionNotFound(
            section_name.to_string(),
        ))
    })?;
    lock.seek(SeekFrom::Start(entry.offset))?;
    let mut buf = vec![0u8; entry.length as usize];
    std::io::Read::read_exact(&mut *lock, &mut buf)?;
    let cursor = Cursor::new(buf);
    let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)?;
    let mut batches = reader.into_iter();
    let batch = batches.next().ok_or_else(|| {
        OpsError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{section_name} section contains no batches"),
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

/// Post-shard-loop tail: merge obs, write obs/(predicate indexes)/provenance/
/// (modality table)/catalog, fsync, rebuild root catalog, finalize header
/// with checksum.
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
    index_options: &ConversionPredicateIndexOptions,
) -> Result<PredicateIndexBuildSummary> {
    // Phase 2d + post-review bug 2 fix.
    //
    // Two append flows depending on input layout:
    //
    // 1. **Convert-on-append (legacy single-section input)** — input
    //    has an `ObsMetadata` section but no `ObsMetadataShard`s. We
    //    rewrite old obs as `ObsMetadataShard` shard 0 (one-time
    //    O(old_n_obs) cost) and write new obs as shards 1+.
    //    Subsequent appends to the file fall through to flow 2.
    //
    // 2. **Raw-copy-extend (already-sharded input)** — input already
    //    has `ObsMetadataShard` entries. We keep those entries in
    //    place (their bytes don't move, their catalog entries
    //    pass through the filter below), compute `next_shard_idx`
    //    after the highest existing index, and write new obs as
    //    shards continuing from there. No re-write of historical
    //    obs — every append after the first costs O(n_new_rows)
    //    instead of O(total_obs_so_far).
    //
    // The reader's cover-verification accepts the resulting
    // monotonically-non-decreasing `n_rows_total` stamps across
    // shards (each writer stamps the file's total at write time;
    // older shards retain their original smaller stamps).
    let new_unified = unify_dict_columns(new_obs)?;

    // Multimodal-skip vs single-modality rebuild decision.
    // `user_wants_index` treats a non-zero `index_auto_threshold` as an
    // explicit request — fixes the pre-fix bug where
    // `--index-auto-threshold N` alone was a no-op.
    let target_is_multimodal = prep.modality_table.is_some();
    let want_index = user_wants_index(index_options);
    let multimodal_skip = if target_is_multimodal && want_index {
        Some(requested_columns(index_options))
    } else {
        None
    };
    let rebuild_index = want_index && !target_is_multimodal;

    let new_n_obs = prep.old_n_obs + n_new_rows;
    let shard_target_rows = prep.header.shard_target_rows.max(1) as usize;

    // Detect input layout. Sorted by shard_idx so we can both
    // determine `next_shard_idx` and iterate them in order when
    // feeding the predicate-index builder.
    let mut existing_obs_shards: Vec<(u32, &FullCatalogEntry)> = prep
        .old_catalog
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::ObsMetadataShard)
        .filter_map(|e| {
            let suffix = e.name.strip_prefix("obs_metadata/shard_")?;
            let idx: u32 = suffix.parse().ok()?;
            Some((idx, e))
        })
        .collect();
    existing_obs_shards.sort_by_key(|(idx, _)| *idx);
    let input_is_sharded = !existing_obs_shards.is_empty();

    // For the convert-on-append path, we need the old obs payload in
    // memory once (to write as shard 0). For the raw-copy path, we
    // skip the materialization entirely — the predicate-index builder
    // reads each existing shard incrementally from the lock-held file.
    let old_unified_for_convert: Option<RecordBatch> = if input_is_sharded {
        None
    } else {
        Some(unify_dict_columns(old_obs)?)
    };

    // Pick a representative obs batch for forced-column schema
    // validation: either the convert-path's unified old obs, or
    // (for the raw-copy path) the new obs itself — they share the
    // same schema after the `validate_obs_schema` check at the
    // call sites.
    let schema_for_validate: &RecordBatch =
        old_unified_for_convert.as_ref().unwrap_or(&new_unified);

    // Read var first (and validate forced index columns) so the
    // seek-around-the-lock-cursor dance happens before any obs writes
    // land. On any forced-column error the file still reads as
    // pre-append (header still points to old catalog).
    let var_for_index = if rebuild_index {
        let var = read_existing_arrow_ipc_section(lock, &prep.old_catalog, "var")?;
        validate_forced_columns(index_options, &schema_for_validate.schema(), &var.schema())?;
        lock.seek(SeekFrom::Start(write_offset))?;
        Some(var)
    } else {
        None
    };

    // For the raw-copy path, snapshot the bytes of each existing
    // shard before handing the file off to `ScxWriter` (the writer's
    // adopted cursor would conflict with concurrent reads through the
    // same lock). Cheap: only the shard payloads we'd already read
    // for predicate-index construction; one read each rather than two.
    let existing_shard_payloads: Vec<Vec<u8>> = if input_is_sharded && rebuild_index {
        let mut out = Vec::with_capacity(existing_obs_shards.len());
        for (_, entry) in &existing_obs_shards {
            lock.seek(SeekFrom::Start(entry.offset))?;
            let mut buf = vec![0u8; entry.length as usize];
            std::io::Read::read_exact(&mut *lock, &mut buf)?;
            out.push(buf);
        }
        lock.seek(SeekFrom::Start(write_offset))?;
        out
    } else {
        Vec::new()
    };

    // Hand the open file off to `ScxWriter` for the obs-shard and
    // predicate-index section emits. The `FileLock` retains the
    // OS-level lock through the duplicate FD (closed below), so
    // concurrent appenders stay blocked. The clone's cursor advances
    // independently of the lock's; we resync via `seek` after
    // `into_in_place_parts`.
    let cloned_file = lock.file().try_clone()?;
    let mut writer =
        ScxWriter::adopt_in_place(cloned_file, prep.header.clone(), write_offset, Vec::new())?;

    let mut obs_index_builder = if rebuild_index {
        Some(
            scx_engine::ObsPredicateIndexBuilder::new(
                schema_for_validate.schema(),
                &predicate_index_build_options_for_obs(index_options),
            )
            .map_err(OpsError::Engine)?,
        )
    } else {
        None
    };

    let mut out_shard_idx: u32;
    let mut cumulative_obs_rows: u64;

    if input_is_sharded {
        // Raw-copy path: existing shards stay in place. Feed each
        // into the predicate-index builder by decoding the snapshot
        // bytes we captured before handing the file to ScxWriter.
        // Compute the next shard_idx from the existing chain.
        out_shard_idx = existing_obs_shards
            .last()
            .map(|(idx, _)| idx.saturating_add(1))
            .unwrap_or(0);
        cumulative_obs_rows = prep.old_n_obs;
        if let Some(b) = obs_index_builder.as_mut() {
            let mut row_offset: u64 = 0;
            for buf in &existing_shard_payloads {
                let cursor = Cursor::new(buf);
                let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)?;
                let mut iter = reader.into_iter();
                let batch = iter.next().ok_or_else(|| {
                    OpsError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "ObsMetadataShard contains no batches",
                    ))
                })??;
                let batch = scx_format::downcast_large_types(&batch).map_err(OpsError::Format)?;
                let n = batch.num_rows() as u64;
                b.push_shard(&batch, row_offset).map_err(OpsError::Engine)?;
                row_offset += n;
            }
        }
    } else {
        // Convert-on-append path: write old obs as ObsMetadataShard
        // shard 0 (one-time O(old_n_obs) cost), then continue with
        // new obs as shards 1+.
        let old_unified = old_unified_for_convert
            .as_ref()
            .expect("legacy convert path populates old_unified_for_convert");
        out_shard_idx = 0;
        cumulative_obs_rows = 0;
        if let Some(b) = obs_index_builder.as_mut() {
            b.push_shard(old_unified, cumulative_obs_rows)
                .map_err(OpsError::Engine)?;
        }
        writer.write_obs_shard(
            out_shard_idx,
            cumulative_obs_rows,
            old_unified.num_rows() as u64,
            new_n_obs,
            old_unified,
        )?;
        out_shard_idx += 1;
        cumulative_obs_rows += old_unified.num_rows() as u64;
    }

    // New obs, split into `shard_target_rows`-sized chunks so each
    // shard stays well under Arrow IPC's narrow-offset ceiling.
    // `RecordBatch::slice` shares Arrow buffers — no per-chunk copy.
    let new_n = new_unified.num_rows();
    let mut cursor = 0;
    while cursor < new_n {
        let take = std::cmp::min(shard_target_rows, new_n - cursor);
        let chunk = new_unified.slice(cursor, take);
        if let Some(b) = obs_index_builder.as_mut() {
            b.push_shard(&chunk, cumulative_obs_rows)
                .map_err(OpsError::Engine)?;
        }
        writer.write_obs_shard(
            out_shard_idx,
            cumulative_obs_rows,
            take as u64,
            new_n_obs,
            &chunk,
        )?;
        out_shard_idx += 1;
        cumulative_obs_rows += take as u64;
        cursor += take;
    }

    let drop_old_predicate_indexes = rebuild_index;
    // Per-shard column stats derived from the rebuilt obs index, applied to the
    // assembled catalog entries (old + new CSR shards) further below — append
    // does not write CSR shards through a fresh `ScxWriter`, so the writer's
    // bulk setter can't reach them.
    let mut per_shard_obs_stats: Option<Vec<Vec<scx_format::catalog::ColumnStat>>> = None;
    let index_result = if let Some(builder) = obs_index_builder {
        // Per-output-shard `(row_start, row_end)` for the full obs:
        // existing CSR shards (modality_id == 0) sorted by row_start,
        // plus the freshly-appended shards in append order.
        let mut shard_row_ranges: Vec<(u64, u64)> = prep
            .old_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CsrShard && e.modality_id == 0)
            .filter_map(|e| e.stats.as_ref().map(|s| (s.row_start, s.row_end)))
            .collect();
        shard_row_ranges.sort_by_key(|(s, _)| *s);
        shard_row_ranges.extend(
            new_shard_entries
                .iter()
                .filter_map(|e| e.stats.as_ref().map(|s| (s.row_start, s.row_end))),
        );

        let var = var_for_index.expect("var_for_index populated when rebuild_index is true");
        let mut result = scx_engine::ConversionPredicateIndexResult::default();
        let obs_bytes = builder
            .finish(
                &shard_row_ranges,
                &mut result.obs_outcomes,
                &mut result.obs_indexed_columns,
            )
            .map_err(OpsError::Engine)?;
        if let Some(bytes) = obs_bytes {
            writer.write_obs_predicate_index(&bytes)?;
            // Derive the per-shard CategoryBitset / MinMax stats now (the index
            // shard_id space == `shard_row_ranges` order). They're applied to
            // the assembled catalog entries below.
            let index = scx_engine::PredicateIndex::read_from(&mut Cursor::new(&bytes))
                .map_err(OpsError::Engine)?;
            per_shard_obs_stats = Some(scx_engine::derive_shard_column_stats(
                &index,
                shard_row_ranges.len(),
            ));
        }
        let preset_var = match index_options.index_preset.as_deref() {
            Some(name) => scx_engine::index_preset_columns(name)
                .map(|p| p.var_columns.iter().map(|s| (*s).to_string()).collect())
                .unwrap_or_default(),
            None => Vec::new(),
        };
        let var_row_ranges: [(u64, u64); 1] = [(0, prep.target_n_vars)];
        let var_build_opts = scx_engine::PredicateIndexBuildOptions {
            forced_columns: index_options.index_var.clone(),
            preset_columns: preset_var,
            auto_threshold: index_options.index_auto_threshold,
            high_cardinality_threshold: 100_000,
        };
        let var_bytes = scx_engine::build_var_predicate_index_bytes(
            &var,
            &var_row_ranges,
            &var_build_opts,
            &mut result.var_outcomes,
            &mut result.var_indexed_columns,
        )?;
        if let Some(bytes) = var_bytes {
            writer.write_var_predicate_index(&bytes)?;
        }
        Some(result)
    } else {
        None
    };

    let (cloned_file, new_offset, writer_new_entries) = writer.into_in_place_parts()?;
    drop(cloned_file);
    // Resync the lock's cursor to the new EOF so subsequent
    // `lock.write_all(...)` calls land after the predicate-index
    // sections rather than overwriting them. `write_offset` is
    // refreshed via `lock.stream_position()` below the provenance
    // write, so we don't update it here.
    lock.seek(SeekFrom::Start(new_offset))?;
    let _ = new_offset; // mirrors the original code's silent drop
    let _ = write_offset; // mirrors the original code's silent drop

    // Split the writer's emitted entries by kind so the catalog
    // assembly below can wire them in the right slots.
    let mut obs_shard_section_entries: Vec<FullCatalogEntry> = Vec::new();
    let mut predicate_index_section_entries: Vec<FullCatalogEntry> = Vec::new();
    for entry in writer_new_entries {
        match entry.section_type {
            SectionType::ObsMetadataShard => obs_shard_section_entries.push(entry),
            SectionType::ObsPredicateIndex | SectionType::VarPredicateIndex => {
                predicate_index_section_entries.push(entry)
            }
            _ => {
                return Err(OpsError::Format(scx_format::ScxError::InvalidCatalog(
                    format!(
                        "finalize_append: unexpected section type {:?} written by adopted writer",
                        entry.section_type
                    ),
                )));
            }
        }
    }

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
        // Stamp the convert-on-append payload shape so downstream tools can
        // tell that obs is now sharded. Append doesn't expose the merge-side
        // policy switches (var identity, uns policy) — there's only one input,
        // so they have no meaning here. Record the actually-indexed columns
        // (mirrors the convert/merge/compact paths) for the audit trail.
        let mut params = serde_json::json!({
            "n_new_rows": n_new_rows,
            "obs_layout": "ObsMetadataShard",
        });
        if let Some(ref result) = index_result {
            params["predicate_index"] = serde_json::json!({
                "obs_columns": result.obs_indexed_columns,
                "var_columns": result.var_indexed_columns,
                "preset": index_options.index_preset,
            });
        }
        entries.push(ProvenanceEntry {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            action: "append".to_string(),
            tool: concat!("scx-ops ", env!("CARGO_PKG_VERSION")).to_string(),
            params_json: params.to_string(),
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
    let n_new_csr_shards =
        u32::try_from(new_shard_entries.len()).map_err(|_| OpsError::ShapeMismatch {
            detail: format!(
                "appended CSR shard count {} exceeds u32::MAX",
                new_shard_entries.len(),
            ),
        })?;
    let mut new_entries: Vec<FullCatalogEntry> = prep
        .old_catalog
        .entries
        .into_iter()
        .filter(|e| {
            // Always drop ObsMetadata / Provenance / CscShard — they
            // are rewritten or invalidated by the append. ObsMetadata
            // bytes from a legacy file become orphaned (no catalog
            // entry points at them); they are recoverable by `scx
            // compact`. The convert-on-append path rewrites old obs
            // as `ObsMetadataShard` shard 0 and new rows as shards
            // 1+. Drop ObsPredicateIndex / VarPredicateIndex only
            // when we are rebuilding them; otherwise stale entries
            // remain in place (matches pre-fix behaviour where the
            // index covers the original rows but not the freshly-
            // appended ones).
            //
            // Bug 2 fix: we keep existing `ObsMetadataShard` entries
            // in the new catalog. The raw-copy-extend path relies on
            // them surviving the filter; the convert-on-append path
            // produces a brand-new shard 0 with the same name
            // (`obs_metadata/shard_0`) which never collides because
            // the legacy file by definition had no `ObsMetadataShard`
            // entries.
            let base_filter = e.section_type != SectionType::ObsMetadata
                && e.section_type != SectionType::Provenance
                && e.section_type != SectionType::CscShard;
            if !drop_old_predicate_indexes {
                return base_filter;
            }
            base_filter
                && e.section_type != SectionType::ObsPredicateIndex
                && e.section_type != SectionType::VarPredicateIndex
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
    new_entries.extend(obs_shard_section_entries);
    new_entries.extend(predicate_index_section_entries);
    new_entries.push(FullCatalogEntry {
        name: "provenance".to_string(),
        offset: prov_offset,
        length: prov_length,
        section_type: SectionType::Provenance,
        checksum: prov_checksum,
        modality_id: 0,
        stats: None,
    });

    // Apply the per-shard column stats to the assembled catalog so query-time
    // shard skipping works across both pre-existing and freshly-appended CSR
    // shards. `assign_csr_shard_column_stats` addresses modality-0 CSR shards by
    // `row_start` order, matching the `shard_row_ranges` the index was built on.
    if let Some(per_shard) = per_shard_obs_stats {
        scx_format::assign_csr_shard_column_stats(&mut new_entries, per_shard)?;
    }

    let new_manifest_sequence = prep.header.manifest_sequence + 1;
    let new_catalog = FullCatalog {
        catalog_version: scx_format::CURRENT_CATALOG_VERSION,
        manifest_sequence: new_manifest_sequence,
        prev_catalog_offset: prep.old_catalog_offset,
        n_obs: new_n_obs,
        entries: new_entries,
        // Append mutates the CSR data (new rows), so bump the data
        // generation. The CSC sidecar is always dropped above, so the
        // build generation resets to 0; any surviving CSC entry would now
        // mismatch and be rejected by the reader.
        data_generation: prep.old_catalog.data_generation + 1,
        csc_build_generation: 0,
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
    Ok(PredicateIndexBuildSummary {
        result: index_result,
        multimodal_skip,
    })
}

/// Compute the next CSR shard index for an append: the modality's
/// existing shard count plus the number of new shards already
/// emitted in this append call. Returns an error if the cumulative
/// count would overflow `u32` (the wire-format shard-index width) —
/// a defensive check; SCX `ScxWriter` rejects shards above
/// `u32::MAX` upstream too.
fn next_shard_idx(old_per_modality_csr: u32, n_appended_so_far: usize) -> Result<u32> {
    let added = u32::try_from(n_appended_so_far).map_err(|_| OpsError::ShapeMismatch {
        detail: format!(
            "appended CSR shard count {n_appended_so_far} exceeds u32::MAX (1 per call)"
        ),
    })?;
    old_per_modality_csr
        .checked_add(added)
        .ok_or_else(|| OpsError::ShapeMismatch {
            detail: format!("next CSR shard index overflows u32: {old_per_modality_csr} + {added}"),
        })
}

/// Build the obs predicate-index options from a conversion-time
/// options struct. Mirrors the preset / forced / auto resolution
/// applied by [`scx_engine::build_and_write_conversion_predicate_indexes`]
/// so the Phase 2d streaming append produces byte-identical predicate
/// indexes to the legacy batch path.
fn predicate_index_build_options_for_obs(
    index_options: &ConversionPredicateIndexOptions,
) -> scx_engine::PredicateIndexBuildOptions {
    let preset_obs = match index_options.index_preset.as_deref() {
        Some(name) => scx_engine::index_preset_columns(name)
            .map(|p| p.obs_columns.iter().map(|s| (*s).to_string()).collect())
            .unwrap_or_default(),
        None => Vec::new(),
    };
    scx_engine::PredicateIndexBuildOptions {
        forced_columns: index_options.index_obs.clone(),
        preset_columns: preset_obs,
        auto_threshold: index_options.index_auto_threshold,
        high_cardinality_threshold: 100_000,
    }
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
        if matches!(field.data_type(), DataType::Dictionary(_, _)) {
            // OE8: single-source the Dictionary(_, V) → V extraction through
            // `effective_type` so this cast path and the append-time schema
            // check can't drift on how they strip the dictionary.
            let value_type = effective_type(field.data_type());
            let cast_col = arrow::compute::cast(col, value_type)?;
            new_fields.push(arrow::datatypes::Field::new(
                field.name(),
                value_type.clone(),
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
