// Shared in-place mutation harness.
//
// SCX in-place mutations (append, and — Phase 1 of PY-MOD-API — metadata
// replacement) follow the same shape: acquire the exclusive lock, read the
// header / full catalog / modality table, append fresh section bytes at EOF,
// then atomically repoint the catalog with a two-barrier commit. This module
// holds the two primitives shared by those ops:
//
//   * [`prepare_in_place`] — the lock + header/catalog/modality-table read
//     prelude, returning the open lock alongside an [`AppendPrep`] snapshot.
//   * [`commit_in_place`] — the catalog-write → fsync → root-catalog-rebuild →
//     fsync → header-finalize atomic commit sequence.
//
// `append` layers its data-specific validation and header bumps (n_obs, nnz,
// shard counts, data_generation) around these; a metadata replace reuses them
// unchanged, leaving the matrix-read invariants untouched.

use std::io::{Cursor, Seek, SeekFrom, Write};
use std::path::Path;

use scx_format::catalog::FullCatalog;
use scx_format::header::{FileHeader, HEADER_SIZE};
use scx_format::modality::{ModalityTable, ModalityType};
use scx_format::section::{write_alignment_padding, SectionType};

use crate::checksum::finalize_header_with_checksum;
use crate::error::{OpsError, Result};
use crate::flock::FileLock;
use crate::rollback::build_root_catalog_from_full;

/// Shared state captured during the prelude of any in-place mutation: header,
/// catalog, modality routing, and resolved per-modality `n_vars`. The exclusive
/// `FileLock` is returned alongside this struct so helper functions can take
/// `&mut FileLock` and `&AppendPrep` without aliasing.
pub(crate) struct AppendPrep {
    pub(crate) header: FileHeader,
    pub(crate) header_index_dtype: u8,
    pub(crate) old_catalog: FullCatalog,
    pub(crate) old_n_obs: u64,
    pub(crate) old_catalog_offset: u64,
    pub(crate) modality_table: Option<ModalityTable>,
    pub(crate) modality_id: u8,
    pub(crate) modality_name: Option<String>,
    pub(crate) modality_type: ModalityType,
    pub(crate) target_n_vars: u64,
    pub(crate) old_per_modality_csr: u32,
}

/// Acquire the exclusive lock, read the header + full catalog + modality
/// table, resolve the requested modality, and apply pre-write validation
/// that does not depend on the new data.
pub(crate) fn prepare_in_place(
    target_path: &Path,
    modality_id: u8,
) -> Result<(FileLock, AppendPrep)> {
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

/// Atomic two-barrier commit for an in-place mutation.
///
/// Assumes the caller has already appended all replacement/new section bytes
/// (and, if present, the modality table) at EOF, and has set the data-specific
/// header fields (`n_obs`, `nnz`, shard counts, …). This helper then:
///
/// 1. writes `new_catalog` at the aligned EOF,
/// 2. fsyncs (barrier 1: durable body + catalog before any pointer update),
/// 3. rebuilds + writes the 4096-byte root catalog at [`HEADER_SIZE`],
/// 4. fsyncs (barrier 2: durable root catalog before the header update),
/// 5. sets the catalog-pointer header fields (derived from `new_catalog` and
///    the modality-table offset/length the caller passes in),
/// 6. finalizes the header with its file checksum in a single durable write —
///    the atomic commit point.
pub(crate) fn commit_in_place(
    lock: &mut FileLock,
    header: &mut FileHeader,
    new_catalog: &FullCatalog,
    modality_table_offset: u64,
    modality_table_length: u64,
) -> Result<()> {
    // Write the new catalog at the aligned EOF. At commit time everything the
    // caller appended (shards, obs, provenance, modality table) is already on
    // disk, so seeking to End gives the same position the append path threaded
    // through `write_offset`.
    let mut write_offset = lock.seek(SeekFrom::End(0))?;
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
    let root_catalog = build_root_catalog_from_full(new_catalog);
    let mut root_buf = Vec::new();
    root_catalog.write_to(&mut root_buf)?;
    let root_catalog_length = root_buf.len() as u64;
    root_buf.resize(4096, 0);

    lock.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
    lock.write_all(&root_buf)?;

    // Crash safety barrier 2: durable root catalog before header update.
    lock.flush()?;
    lock.sync_all()?;

    // Catalog-pointer header fields. `manifest_sequence` and
    // `prev_catalog_offset` are taken from `new_catalog` (the caller stamped
    // them there), keeping the header and catalog in lockstep.
    header.full_catalog_offset = new_catalog_offset;
    header.full_catalog_length = new_catalog_length;
    header.modality_table_offset = modality_table_offset;
    header.modality_table_length = modality_table_length;
    header.manifest_sequence = new_catalog.manifest_sequence;
    header.prev_catalog_offset = new_catalog.prev_catalog_offset;
    header.root_catalog_offset = HEADER_SIZE as u64;
    header.root_catalog_length = root_catalog_length;

    header.clear_front_catalog();
    header.front_catalog_offset = 0;
    header.front_catalog_length = 0;

    finalize_header_with_checksum(lock, header)?;
    Ok(())
}
