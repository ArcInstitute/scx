// SCX → h5mu (MuData) conversion (Phase D.2).
//
// Inverts the h5mu_to_scx pipeline. For a multi-modality SCX v2 file
// (`n_modalities > 0`):
//
//   /obs                          ← global obs (modality_id = 0)
//   /obsm/{key}                   ← global obsm (modality_id = 0)
//   /uns/...                      ← global uns
//   /mod/{modality_name}/X        ← per-modality CSR matrix
//   /mod/{modality_name}/var      ← per-modality var
//   /mod/{modality_name}/obs      ← shared obs (a copy — mudata
//                                   spec wants per-modality obs to
//                                   exist; we point each modality
//                                   at the global cell list)
//   /mod/{modality_name}/obsm/{k} ← per-modality embeddings
//
// Single-modality v2 files raise — callers should use
// `scx_to_h5ad` instead, optionally with `--modality NAME` for
// extracting one modality from a multimodal file.

use std::path::Path;

use hdf5::types::VarLenUnicode;
use scx_format_io::reader::ScxReader;
use scx_format_io::section::SectionType;

use crate::h5_write_util::vlu;
use crate::h5ad::column_stream::write_dataframe_group_at;
use crate::h5ad::stream_write::{
    stream_csr_to_group_at, stream_layers_at, write_obs_streaming_or_eager,
};
use crate::h5ad::uns::write_uns_entries_at;
use crate::h5ad::write::{write_obsm_entry_at, write_sparse_group_at};
use crate::pipeline::ConvertError;
use crate::warnings::WarningSink;

/// Write an SCX file to h5mu format. Requires `reader.is_multimodal()`.
pub fn scx_to_h5mu(
    scx_path: &Path,
    h5mu_path: &Path,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let reader = ScxReader::open(scx_path)?;
    if !reader.is_multimodal() {
        return Err(ConvertError::Other(format!(
            "SCX file '{}' is single-modality (n_modalities=0); use --to h5ad instead",
            scx_path.display()
        )));
    }

    let file = hdf5::File::create(h5mu_path)?;
    let root = file.as_group()?;
    // Build the keep mask once and apply it symmetrically across
    // /X (via `read_all_csr_shards_for_filtered`) and obs (via the
    // streaming-or-eager dispatcher). Pre-fix, this path silently
    // dropped DVs on both legs — the per-modality eager CSR reader
    // does *not* filter internally, contrary to what the prior
    // "preserves prior behavior" comment claimed.
    let keep_mask = crate::h5ad::stream_write::build_keep_mask(&reader)?;
    write_h5mu_root_attrs_and_global_blocks(&reader, &file, &root, keep_mask.as_deref(), sink)?;

    // Per-modality blocks under /mod/{name}.
    let mod_group = root.create_group("mod")?;
    for modality_id in 1..=reader.n_modalities() as u8 {
        let info = reader.modality_info(modality_id).ok_or_else(|| {
            ConvertError::Other(format!(
                "modality_id {modality_id} not found in modality table"
            ))
        })?;
        let mname = info.name.clone();
        let modality_root = create_modality_group_with_attrs(&mod_group, &mname)?;

        // Per-modality X matrix (materialising path, DV-filtered).
        let csr = reader.read_all_csr_shards_for_filtered(modality_id)?;
        write_sparse_group_at(
            &modality_root,
            "X",
            &csr.indptr,
            &csr.indices,
            &csr.data,
            csr.shape.0,
            csr.shape.1,
        )?;

        write_h5mu_per_modality_non_x_blocks(
            &reader,
            &modality_root,
            modality_id,
            &mname,
            keep_mask.as_deref(),
            sink,
        )?;
    }

    Ok(())
}

/// Streaming SCX → h5mu (Phase 8). Bounds peak RSS to one shard's
/// worth of CSR per matrix written. Iterates each modality, streams
/// `/mod/{name}/X` plus all `/mod/{name}/layers/{layer}` via the
/// shard-by-shard writer in [`crate::h5ad::stream_write`]. Reuses the
/// existing materialising metadata helpers verbatim.
pub fn scx_to_h5mu_streaming(
    scx_path: &Path,
    h5mu_path: &Path,
    opts: &crate::ExportOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let reader = ScxReader::open(scx_path)?;
    if !reader.is_multimodal() {
        return Err(ConvertError::Other(format!(
            "SCX file '{}' is single-modality (n_modalities=0); use --to h5ad instead",
            scx_path.display()
        )));
    }

    // `export_min_counts` sums one modality's X, which is ambiguous when every
    // modality is being written. An explicit mask is unambiguous (the obs axis
    // is shared across modalities), so it is honoured.
    if opts.export_min_counts.is_some() {
        return Err(ConvertError::Other(
            "min_counts is ambiguous for a multimodal h5mu export (which modality's X?); \
             pass an explicit obs mask, or export one modality with to_h5ad(modality=…)"
                .into(),
        ));
    }

    let file = hdf5::File::create(h5mu_path)?;
    let root = file.as_group()?;
    let keep_mask =
        crate::h5ad::stream_write::build_export_keep_mask(&reader, scx_path, 0, opts)?.mask;
    write_h5mu_root_attrs_and_global_blocks(&reader, &file, &root, keep_mask.as_deref(), sink)?;

    let mod_group = root.create_group("mod")?;
    for modality_id in 1..=reader.n_modalities() as u8 {
        let info = reader.modality_info(modality_id).ok_or_else(|| {
            ConvertError::Other(format!(
                "modality_id {modality_id} not found in modality table"
            ))
        })?;
        let mname = info.name.clone();
        let modality_root = create_modality_group_with_attrs(&mod_group, &mname)?;

        // Per-modality X matrix (streaming).
        let n_vars = info.n_vars as usize;
        stream_csr_to_group_at(
            &modality_root,
            "X",
            &reader,
            modality_id,
            SectionType::CsrShard,
            None,
            n_vars,
            keep_mask.as_deref(),
            opts,
            sink,
        )?;

        write_h5mu_per_modality_non_x_blocks(
            &reader,
            &modality_root,
            modality_id,
            &mname,
            keep_mask.as_deref(),
            sink,
        )?;

        // Per-modality layers (streaming).
        stream_layers_at(
            &modality_root,
            &reader,
            modality_id,
            keep_mask.as_deref(),
            opts,
            sink,
        )?;
    }

    Ok(())
}

fn write_h5mu_root_attrs_and_global_blocks(
    reader: &ScxReader,
    file: &hdf5::File,
    root: &hdf5::Group,
    keep_mask_opt: Option<&[bool]>,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    // Mark file as MuData (mudata HDF5 convention — readers may
    // look for these attributes).
    file.new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&vlu("MuData"))?;
    file.new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&vlu("0.1.0"))?;

    // Global obs. Sharded sources auto-stream via
    // `write_obs_streaming_or_eager` regardless of the caller's
    // `--stream` choice (the alternative is to materialise the full
    // assembled obs, which is exactly what task 6a eliminates).
    write_obs_streaming_or_eager(root, reader, keep_mask_opt, sink)?;

    // Global obsm — entries with modality_id == 0 and section name
    // starting with "obsm/" but with no modality slash.
    {
        let global_obsm: Vec<_> = reader
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::ObsmEmbedding && e.modality_id == 0)
            .filter(|e| e.name.starts_with("obsm/"))
            .filter(|e| e.name.matches('/').count() == 1)
            .collect();
        if !global_obsm.is_empty() {
            let obsm_group = root.create_group("obsm")?;
            for entry in global_obsm {
                let key = entry.name.strip_prefix("obsm/").unwrap_or(&entry.name);
                if let Ok(batch) = reader.read_obsm(key) {
                    let filtered = match keep_mask_opt {
                        Some(mask) => {
                            crate::h5ad::stream_write::filter_record_batch_by_mask(&batch, mask)?
                        }
                        None => batch,
                    };
                    write_obsm_entry_at(&obsm_group, key, &filtered)?;
                }
            }
        }
    }

    // Global uns.
    if let Ok(uns) = reader.read_uns() {
        let uns_group = root.create_group("uns")?;
        write_uns_entries_at(&uns_group, &uns)?;
    }

    Ok(())
}

fn create_modality_group_with_attrs(
    mod_group: &hdf5::Group,
    mname: &str,
) -> Result<hdf5::Group, ConvertError> {
    let modality_root = mod_group.create_group(mname)?;
    modality_root
        .new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&vlu("anndata"))?;
    modality_root
        .new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&vlu("0.1.0"))?;
    Ok(modality_root)
}

fn write_h5mu_per_modality_non_x_blocks(
    reader: &ScxReader,
    modality_root: &hdf5::Group,
    modality_id: u8,
    mname: &str,
    keep_mask_opt: Option<&[bool]>,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    // Per-modality var. Var is per-modality in v2 (small, never
    // sharded in the current format) — stays on the eager path.
    let var = reader.read_var_for(modality_id)?;
    write_dataframe_group_at(modality_root, "var", &var, sink)?;

    // Per-modality obs is the shared global obs (mudata spec
    // requires per-modality obs; we point each modality at the
    // same data so downstream loaders that read /mod/{m}/obs
    // see consistent cell metadata). Streams from sharded sources.
    write_obs_streaming_or_eager(modality_root, reader, keep_mask_opt, sink)?;

    // Per-modality obsm: entries named `obsm/{mname}/{key}`.
    let mod_obsm: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::ObsmEmbedding && e.modality_id == modality_id)
        .collect();
    if !mod_obsm.is_empty() {
        let obsm_group = modality_root.create_group("obsm")?;
        for entry in mod_obsm {
            let prefix = format!("obsm/{mname}/");
            let key = entry
                .name
                .strip_prefix(&prefix)
                .unwrap_or(&entry.name)
                .to_string();
            if let Ok(batch) = reader.read_obsm_for(modality_id, &key) {
                let filtered = match keep_mask_opt {
                    Some(mask) => {
                        crate::h5ad::stream_write::filter_record_batch_by_mask(&batch, mask)?
                    }
                    None => batch,
                };
                write_obsm_entry_at(&obsm_group, &key, &filtered)?;
            }
        }
    }
    Ok(())
}

/// Extract a single modality from a multi-modality SCX file as a
/// stand-alone h5ad file. Maps modality `m` to a root-level h5ad
/// layout: `/X` is `m`'s primary CSR, `/obs` is the shared global
/// obs, `/var` is `m`'s var, `/obsm/{key}` is `m`'s obsm.
pub fn scx_modality_to_h5ad(
    scx_path: &Path,
    h5ad_path: &Path,
    modality_name: &str,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let reader = ScxReader::open(scx_path)?;
    let modality_id = reader.modality_id(modality_name).ok_or_else(|| {
        ConvertError::Other(format!(
            "modality '{modality_name}' not found in SCX file '{}'",
            scx_path.display()
        ))
    })?;

    let file = hdf5::File::create(h5ad_path)?;
    let root = file.as_group()?;

    // Mirror the streaming entry point: honor DVs symmetrically on
    // /X and obs via `read_all_csr_shards_for_filtered` + the shared
    // streaming-or-eager obs dispatcher.
    let keep_mask = crate::h5ad::stream_write::build_keep_mask(&reader)?;

    let csr = reader.read_all_csr_shards_for_filtered(modality_id)?;
    write_sparse_group_at(
        &root,
        "X",
        &csr.indptr,
        &csr.indices,
        &csr.data,
        csr.shape.0,
        csr.shape.1,
    )?;

    write_modality_to_h5ad_non_x_blocks(
        &reader,
        &root,
        modality_id,
        modality_name,
        keep_mask.as_deref(),
        sink,
    )?;

    Ok(())
}

/// Streaming variant of [`scx_modality_to_h5ad`]: extracts one
/// modality from a multimodal SCX file as h5ad, streaming `/X` and
/// `/layers/{name}` shard-by-shard.
pub fn scx_modality_to_h5ad_streaming(
    scx_path: &Path,
    h5ad_path: &Path,
    modality_name: &str,
    opts: &crate::ExportOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let reader = ScxReader::open(scx_path)?;
    let modality_id = reader.modality_id(modality_name).ok_or_else(|| {
        ConvertError::Other(format!(
            "modality '{modality_name}' not found in SCX file '{}'",
            scx_path.display()
        ))
    })?;
    let info = reader.modality_info(modality_id).ok_or_else(|| {
        ConvertError::Other(format!(
            "modality_id {modality_id} not found in modality table"
        ))
    })?;
    let n_vars = info.n_vars as usize;

    let file = hdf5::File::create(h5ad_path)?;
    let root = file.as_group()?;

    // `modality_id` scopes a `min_counts` pre-pass to this modality's X.
    let filter =
        crate::h5ad::stream_write::build_export_keep_mask(&reader, scx_path, modality_id, opts)?;
    let export_note =
        crate::h5ad::stream_write::build_export_provenance(&reader, scx_path, opts, &filter);
    let keep_mask = filter.mask;

    stream_csr_to_group_at(
        &root,
        "X",
        &reader,
        modality_id,
        SectionType::CsrShard,
        None,
        n_vars,
        keep_mask.as_deref(),
        opts,
        sink,
    )?;

    write_modality_to_h5ad_non_x_blocks(
        &reader,
        &root,
        modality_id,
        modality_name,
        keep_mask.as_deref(),
        sink,
    )?;

    stream_layers_at(
        &root,
        &reader,
        modality_id,
        keep_mask.as_deref(),
        opts,
        sink,
    )?;

    // `write_modality_to_h5ad_non_x_blocks` writes no `/uns`, so a filtered
    // modality export would otherwise silently skip the "records what it
    // dropped" contract that the single-modality path upholds. Only
    // materialised when a caller filter was actually applied, so unfiltered
    // modality exports stay byte-identical.
    if let Some(note) = export_note {
        let uns = serde_json::json!({
            crate::h5ad::stream_write::EXPORT_PROVENANCE_KEY: note,
        });
        let uns_group = root.create_group("uns")?;
        write_uns_entries_at(&uns_group, &uns)?;
    }

    Ok(())
}

fn write_modality_to_h5ad_non_x_blocks(
    reader: &ScxReader,
    root: &hdf5::Group,
    modality_id: u8,
    modality_name: &str,
    keep_mask_opt: Option<&[bool]>,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    // Shared global obs (cell axis); streams from sharded sources.
    write_obs_streaming_or_eager(root, reader, keep_mask_opt, sink)?;
    let var = reader.read_var_for(modality_id)?;
    write_dataframe_group_at(root, "var", &var, sink)?;

    let mod_obsm: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::ObsmEmbedding && e.modality_id == modality_id)
        .collect();
    if !mod_obsm.is_empty() {
        let obsm_group = root.create_group("obsm")?;
        for entry in mod_obsm {
            let prefix = format!("obsm/{modality_name}/");
            let key = entry
                .name
                .strip_prefix(&prefix)
                .unwrap_or(&entry.name)
                .to_string();
            if let Ok(batch) = reader.read_obsm_for(modality_id, &key) {
                let filtered = match keep_mask_opt {
                    Some(mask) => {
                        crate::h5ad::stream_write::filter_record_batch_by_mask(&batch, mask)?
                    }
                    None => batch,
                };
                write_obsm_entry_at(&obsm_group, &key, &filtered)?;
            }
        }
    }
    Ok(())
}
