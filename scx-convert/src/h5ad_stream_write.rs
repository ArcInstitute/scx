// Streaming SCX → h5ad writer.
//
// Symmetrical to `h5ad_stream.rs` on the read side: walks SCX CSR
// shards in row order and writes hyperslab slices into pre-allocated
// `/X/{indptr,indices,data}` (and `/layers/{name}/…`) HDF5 datasets.
// Peak RSS is bounded by one shard's worth of CSR plus encode buffers
// regardless of total file size.
//
// Layout decisions
// ---------------
// * Pre-allocate the `indptr` / `indices` / `data` triplet by walking
//   the catalog stats once (`stats.nnz` over CSR shards). This avoids
//   HDF5 extendable datasets and keeps the on-disk layout
//   deterministic.
// * Deletion vectors: when present, the pre-scan decodes each shard
//   once to count kept nnz, then the write loop decodes again. The
//   second decode is acceptable for export — deterministic on-disk
//   layout is worth one extra pass over the CSR shards.
// * Metadata writes (`obs`, `var`, `obsm`, `varm`, `obsp`, `varp`,
//   `uns`) reuse the non-streaming helpers in `h5ad_write.rs`. Only
//   `/X` and `/layers/{name}` change.

use std::path::Path;

use hdf5::types::VarLenUnicode;
use ndarray::ArrayView1;
use scx_format::catalog::{FullCatalogEntry, ShardStats};
use scx_format::reader::ScxReader;
use scx_format::section::SectionType;

use super::h5ad_write::{write_dataframe_group_at, write_obsm_entry_at, write_uns_entries_at};
use super::pipeline::{ConvertError, ConvertOptions};
use super::warnings::WarningSink;

fn vlu(s: &str) -> VarLenUnicode {
    s.parse::<VarLenUnicode>().unwrap_or_else(|_| {
        let cleaned: String = s.chars().filter(|&c| c != '\0').collect();
        cleaned
            .parse::<VarLenUnicode>()
            .expect("cleaned string should have no NUL bytes")
    })
}

/// Streaming SCX → h5ad entry point. Mirrors `write_scx_to_h5ad`
/// (`h5ad_write.rs:28`) but writes `/X` and `/layers/{name}` shard-
/// by-shard via pre-allocated hyperslab datasets. Single-modality
/// SCX files only; multimodal files must use `scx_to_h5mu_streaming`
/// or `scx_modality_to_h5ad_streaming`.
pub fn write_scx_to_h5ad_streaming(
    scx_path: &Path,
    h5ad_path: &Path,
    opts: &ConvertOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let reader = ScxReader::open(scx_path)?;
    if reader.is_multimodal() {
        return Err(ConvertError::Other(format!(
            "SCX file '{}' is multimodal ({} modalities); use scx_to_h5mu_streaming \
             or scx_modality_to_h5ad_streaming",
            scx_path.display(),
            reader.n_modalities()
        )));
    }

    let file = hdf5::File::create(h5ad_path)?;
    let root = file.as_group()?;

    let keep_mask = build_keep_mask(&reader)?;

    // /X.
    let n_vars = reader.n_vars() as usize;
    stream_csr_to_group_at(
        &root,
        "X",
        &reader,
        0,
        SectionType::CsrShard,
        None,
        n_vars,
        keep_mask.as_deref(),
        opts,
        sink,
    )?;

    // obs.
    if let Ok(obs) = reader.read_obs() {
        write_dataframe_group_at(&root, "obs", &obs)?;
    }

    // var.
    if let Ok(var) = reader.read_var() {
        write_dataframe_group_at(&root, "var", &var)?;
    }

    // obsm.
    if let Ok(obsm_map) = reader.read_all_obsm() {
        if !obsm_map.is_empty() {
            let obsm_group = root.create_group("obsm")?;
            for (name, batch) in &obsm_map {
                write_obsm_entry_at(&obsm_group, name, batch)?;
            }
        }
    }

    // varm.
    if let Ok(varm_map) = reader.read_all_varm() {
        if !varm_map.is_empty() {
            let varm_group = root.create_group("varm")?;
            for (name, batch) in &varm_map {
                write_obsm_entry_at(&varm_group, name, batch)?;
            }
        }
    }

    // uns.
    if let Ok(uns) = reader.read_uns() {
        let uns_group = root.create_group("uns")?;
        write_uns_entries_at(&uns_group, &uns)?;
    }

    // /layers/{name}.
    stream_layers_at(&root, &reader, 0, keep_mask.as_deref(), opts, sink)?;

    Ok(())
}

/// Drain one named CSR matrix (`/X` or `/layers/{layer}`) from `reader`
/// into the HDF5 group `parent` under `name`. Pre-allocates the CSR
/// triplet using catalog stats (or a per-shard decode if deletion
/// vectors are active), then walks shards in row order writing
/// hyperslab slices.
#[allow(clippy::too_many_arguments)]
pub(super) fn stream_csr_to_group_at(
    parent: &hdf5::Group,
    name: &str,
    reader: &ScxReader,
    modality_id: u8,
    section_type: SectionType,
    layer_name: Option<&str>,
    n_vars: usize,
    keep_mask_opt: Option<&[bool]>,
    _opts: &ConvertOptions,
    _sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let shards = collect_shards(reader, modality_id, section_type, layer_name);

    let n_obs_total = total_rows_in_shards(&shards);
    let n_obs_kept = match keep_mask_opt {
        Some(mask) => {
            // Mask length must cover the union of shard row ranges. We
            // index by global row, so any short mask is a catalog /
            // header drift — surface as a real error rather than a
            // panic (callers may be in Python).
            if mask.len() < n_obs_total as usize {
                return Err(ConvertError::Other(format!(
                    "keep_mask length {} < n_obs_total {} (catalog/header drift)",
                    mask.len(),
                    n_obs_total
                )));
            }
            count_kept_rows_in_shards(&shards, mask) as u64
        }
        None => n_obs_total,
    };
    let total_nnz = precompute_total_nnz(reader, &shards, keep_mask_opt, layer_name, modality_id)?;

    let group = parent.create_group(name)?;
    create_csr_triplet(&group, n_obs_kept as usize, n_vars, total_nnz as usize)?;

    // Re-open the three datasets just created so we can hyperslab into them.
    let indptr_ds = group.dataset("indptr")?;
    let indices_ds = group.dataset("indices")?;
    let data_ds = group.dataset("data")?;

    // indptr[0] = 0; written once before the loop.
    indptr_ds.write_slice(ArrayView1::from(&[0i64][..]), ndarray::s![0..1])?;

    let mut nnz_offset: u64 = 0;
    let mut row_offset_kept: u64 = 0;

    for (shard_idx, entry) in shards.iter().enumerate() {
        let stats = require_stats(entry)?;
        let shard_row_start = stats.row_start as usize;

        let (indptr_local_i64, indices_local_i32, data_local_f32) =
            read_shard_payload(reader, modality_id, section_type, layer_name, shard_idx)?;

        let (kept_indptr_tail, kept_indices_i32, kept_data_f32) = filter_shard(
            indptr_local_i64,
            indices_local_i32,
            data_local_f32,
            keep_mask_opt,
            shard_row_start,
            nnz_offset,
        );

        if !kept_indices_i32.is_empty() {
            indices_ds.write_slice(
                ArrayView1::from(kept_indices_i32.as_slice()),
                ndarray::s![nnz_offset as usize..nnz_offset as usize + kept_indices_i32.len()],
            )?;
            data_ds.write_slice(
                ArrayView1::from(kept_data_f32.as_slice()),
                ndarray::s![nnz_offset as usize..nnz_offset as usize + kept_data_f32.len()],
            )?;
        }
        if !kept_indptr_tail.is_empty() {
            let lo = (row_offset_kept + 1) as usize;
            let hi = lo + kept_indptr_tail.len();
            indptr_ds.write_slice(
                ArrayView1::from(kept_indptr_tail.as_slice()),
                ndarray::s![lo..hi],
            )?;
        }
        row_offset_kept += kept_indptr_tail.len() as u64;
        nnz_offset += kept_indices_i32.len() as u64;
    }

    debug_assert_eq!(nnz_offset, total_nnz);
    debug_assert_eq!(row_offset_kept, n_obs_kept);

    Ok(())
}

/// Iterate every layer attached to `modality_id` and stream-write
/// each one under `parent/layers/{layer_name}`.
pub(super) fn stream_layers_at(
    parent: &hdf5::Group,
    reader: &ScxReader,
    modality_id: u8,
    keep_mask_opt: Option<&[bool]>,
    opts: &ConvertOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let names = reader.layer_names_for(modality_id);
    if names.is_empty() {
        return Ok(());
    }
    let n_vars = match modality_id {
        0 => reader.n_vars() as usize,
        _ => reader
            .modality_info(modality_id)
            .map(|info| info.n_vars as usize)
            .unwrap_or_else(|| reader.n_vars() as usize),
    };
    let layers_group = parent.create_group("layers")?;
    for layer_name in names {
        stream_csr_to_group_at(
            &layers_group,
            &layer_name,
            reader,
            modality_id,
            SectionType::LayerCsrShard,
            Some(&layer_name),
            n_vars,
            keep_mask_opt,
            opts,
            sink,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn collect_shards<'a>(
    reader: &'a ScxReader,
    modality_id: u8,
    section_type: SectionType,
    layer_name: Option<&str>,
) -> Vec<&'a FullCatalogEntry> {
    match (section_type, layer_name) {
        (SectionType::CsrShard, None) => reader.catalog().csr_shards_for_modality(modality_id),
        (SectionType::LayerCsrShard, Some(name)) if modality_id == 0 => {
            // Legacy single-modality naming: `{layer}_shard_{idx}`.
            let prefix = format!("{name}_shard_");
            let mut shards: Vec<&FullCatalogEntry> = reader
                .catalog()
                .entries
                .iter()
                .filter(|e| {
                    e.section_type == SectionType::LayerCsrShard && e.name.starts_with(&prefix)
                })
                .collect();
            shards.sort_by_key(|e| {
                e.stats
                    .as_ref()
                    .map_or(u64::MAX, |s| s.major_start(SectionType::LayerCsrShard))
            });
            shards
        }
        (SectionType::LayerCsrShard, Some(name)) => reader
            .catalog()
            .layer_csr_shards_for_modality(modality_id, name),
        _ => Vec::new(),
    }
}

#[allow(clippy::type_complexity)] // (indptr, indices, data) matches every read_csr_shard_* return type.
fn read_shard_payload(
    reader: &ScxReader,
    modality_id: u8,
    section_type: SectionType,
    layer_name: Option<&str>,
    shard_idx: usize,
) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>), ConvertError> {
    let (ip, ix, dv) = match (section_type, layer_name) {
        (SectionType::CsrShard, _) => reader.read_csr_shard_for(modality_id, shard_idx)?,
        (SectionType::LayerCsrShard, Some(name)) if modality_id == 0 => {
            reader.read_layer_csr_shard(name, shard_idx)?
        }
        (SectionType::LayerCsrShard, Some(name)) => {
            reader.read_layer_csr_shard_for(modality_id, name, shard_idx)?
        }
        _ => {
            return Err(ConvertError::Other(format!(
                "unsupported shard read: section_type={section_type:?}, layer_name={layer_name:?}"
            )));
        }
    };
    Ok((ip, ix, dv))
}

/// Indptr-only counterpart of [`read_shard_payload`]. Decodes just the
/// row-pointer array, skipping indices/data — used by precompute paths
/// that only need per-row nnz counts.
fn read_shard_indptr(
    reader: &ScxReader,
    modality_id: u8,
    section_type: SectionType,
    layer_name: Option<&str>,
    shard_idx: usize,
) -> Result<Vec<i64>, ConvertError> {
    let ip = match (section_type, layer_name) {
        (SectionType::CsrShard, _) => reader.read_csr_shard_indptr_for(modality_id, shard_idx)?,
        (SectionType::LayerCsrShard, Some(name)) if modality_id == 0 => {
            reader.read_layer_csr_shard_indptr(name, shard_idx)?
        }
        (SectionType::LayerCsrShard, Some(name)) => {
            reader.read_layer_csr_shard_indptr_for(modality_id, name, shard_idx)?
        }
        _ => {
            return Err(ConvertError::Other(format!(
                "unsupported shard read: section_type={section_type:?}, layer_name={layer_name:?}"
            )));
        }
    };
    Ok(ip)
}

/// `entry.stats` must be present on every shard in v2 catalogs (the
/// writer auto-upgrades v1 → v2 on serialise). Missing stats here
/// means the file is corrupt or truncated.
fn require_stats(entry: &FullCatalogEntry) -> Result<&ShardStats, ConvertError> {
    entry.stats.as_ref().ok_or_else(|| {
        ConvertError::Other(format!(
            "shard '{}' has no catalog stats — file may be corrupt or truncated",
            entry.name
        ))
    })
}

fn total_rows_in_shards(shards: &[&FullCatalogEntry]) -> u64 {
    shards
        .iter()
        .filter_map(|e| e.stats.as_ref().map(|s| s.row_end - s.row_start))
        .sum()
}

fn count_kept_rows_in_shards(shards: &[&FullCatalogEntry], keep_mask: &[bool]) -> usize {
    let mut kept = 0usize;
    for entry in shards {
        if let Some(stats) = entry.stats.as_ref() {
            for r in stats.row_start as usize..stats.row_end as usize {
                if r < keep_mask.len() && keep_mask[r] {
                    kept += 1;
                }
            }
        }
    }
    kept
}

fn precompute_total_nnz(
    reader: &ScxReader,
    shards: &[&FullCatalogEntry],
    keep_mask_opt: Option<&[bool]>,
    layer_name: Option<&str>,
    modality_id: u8,
) -> Result<u64, ConvertError> {
    match keep_mask_opt {
        None => {
            // Fast path: sum stats.nnz across shards (zero decode).
            let mut nnz = 0u64;
            for entry in shards {
                let stats = require_stats(entry)?;
                nnz = nnz.saturating_add(stats.nnz);
            }
            Ok(nnz)
        }
        Some(mask) => {
            // DV active: decode each shard's indptr once to count
            // kept nnz. Indices/data stay encoded — they're not
            // touched until the main write loop's full-shard read.
            let section_type = if layer_name.is_some() {
                SectionType::LayerCsrShard
            } else {
                SectionType::CsrShard
            };
            let mut nnz = 0u64;
            for (shard_idx, entry) in shards.iter().enumerate() {
                let stats = require_stats(entry)?;
                let row_start = stats.row_start as usize;
                let indptr_local =
                    read_shard_indptr(reader, modality_id, section_type, layer_name, shard_idx)?;
                let n_rows = indptr_local.len().saturating_sub(1);
                for r in 0..n_rows {
                    let global = row_start + r;
                    if global < mask.len() && mask[global] {
                        let start = indptr_local[r] as usize;
                        let end = indptr_local[r + 1] as usize;
                        nnz = nnz.saturating_add((end - start) as u64);
                    }
                }
            }
            Ok(nnz)
        }
    }
}

pub(super) fn build_keep_mask(reader: &ScxReader) -> Result<Option<Vec<bool>>, ConvertError> {
    if !reader.header().has_deletion_vectors() {
        return Ok(None);
    }
    let dv = match reader.read_deletion_vectors()? {
        Some(dv) if dv.total_deleted() > 0 => dv,
        _ => return Ok(None),
    };

    // Mirror `filter_csr_rows_by_deletion_vectors` (`reader.rs:1429`):
    // shard order = `shards_sorted()` (CSR), local rows from the
    // deletion-vector bitmap are translated to global rows via
    // `stats.row_start`.
    let n_obs = reader.n_obs() as usize;
    let shards = reader.catalog().shards_sorted();
    let mut keep = vec![true; n_obs];
    for (shard_idx, entry) in shards.iter().enumerate() {
        if let Some(stats) = entry.stats.as_ref() {
            if let Some(bitmap) = dv.shards.get(&(shard_idx as u32)) {
                for local_row in bitmap.iter() {
                    let global_row = stats.row_start + local_row as u64;
                    if (global_row as usize) < n_obs {
                        keep[global_row as usize] = false;
                    }
                }
            }
        }
    }
    Ok(Some(keep))
}

fn create_csr_triplet(
    group: &hdf5::Group,
    n_obs_kept: usize,
    n_vars: usize,
    total_nnz: usize,
) -> Result<(), ConvertError> {
    group
        .new_dataset::<i64>()
        .shape([n_obs_kept + 1])
        .create("indptr")?;
    group
        .new_dataset::<i32>()
        .shape([total_nnz])
        .create("indices")?;
    group
        .new_dataset::<f32>()
        .shape([total_nnz])
        .create("data")?;

    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&vlu("csr_matrix"))?;
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&vlu("0.1.0"))?;
    let shape = [n_obs_kept as i64, n_vars as i64];
    group
        .new_attr::<i64>()
        .shape([2])
        .create("shape")?
        .write(&shape)?;
    Ok(())
}

/// Filter a single shard by `keep_mask_opt` and rebase its indptr to
/// the cumulative absolute nnz offset (the value scipy stores at
/// `indptr[i+1]`). Returns `(kept_indptr_tail, kept_indices,
/// kept_data)`.
///
/// Fast path: `keep_mask_opt.is_none()` — the input `indices` / `data`
/// vectors are returned unchanged (no allocation, no copy). Takes the
/// three Vecs by value so the fast path can transfer ownership
/// directly.
fn filter_shard(
    indptr_local: Vec<i64>,
    indices_local: Vec<i32>,
    data_local: Vec<f32>,
    keep_mask_opt: Option<&[bool]>,
    shard_row_start: usize,
    nnz_offset: u64,
) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
    match keep_mask_opt {
        None => {
            let kept_indptr_tail: Vec<i64> = indptr_local
                .iter()
                .skip(1)
                .map(|v| *v + nnz_offset as i64)
                .collect();
            (kept_indptr_tail, indices_local, data_local)
        }
        Some(mask) => {
            let n_rows = indptr_local.len().saturating_sub(1);
            let mut kept_indptr_tail: Vec<i64> = Vec::new();
            let mut kept_indices: Vec<i32> = Vec::new();
            let mut kept_data: Vec<f32> = Vec::new();
            let mut running_nnz: i64 = nnz_offset as i64;
            for r in 0..n_rows {
                let global = shard_row_start + r;
                if global >= mask.len() || !mask[global] {
                    continue;
                }
                let start = indptr_local[r] as usize;
                let end = indptr_local[r + 1] as usize;
                kept_indices.extend_from_slice(&indices_local[start..end]);
                kept_data.extend_from_slice(&data_local[start..end]);
                running_nnz += (end - start) as i64;
                kept_indptr_tail.push(running_nnz);
            }
            (kept_indptr_tail, kept_indices, kept_data)
        }
    }
}
