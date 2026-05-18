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
use scx_format::reader::ScxReader;
use scx_format::section::SectionType;

use super::h5ad_stream_write::{stream_csr_to_group_at, stream_layers_at};
use super::h5ad_write::{
    write_dataframe_group_at, write_obsm_entry_at, write_sparse_group_at, write_uns_entries_at,
};
use super::pipeline::{ConvertError, ConvertOptions};
use super::warnings::WarningSink;

fn vlu(s: &str) -> VarLenUnicode {
    s.parse().unwrap_or_else(|_| "".parse().unwrap())
}

/// Write an SCX file to h5mu format. Requires `reader.is_multimodal()`.
pub fn scx_to_h5mu(scx_path: &Path, h5mu_path: &Path) -> Result<(), ConvertError> {
    let reader = ScxReader::open(scx_path)?;
    if !reader.is_multimodal() {
        return Err(ConvertError::Other(format!(
            "SCX file '{}' is single-modality (n_modalities=0); use --to h5ad instead",
            scx_path.display()
        )));
    }

    let file = hdf5::File::create(h5mu_path)?;
    let root = file.as_group()?;
    write_h5mu_root_attrs_and_global_blocks(&reader, &file, &root)?;

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

        // Per-modality X matrix (materialising path).
        let csr = reader.read_all_csr_shards_for(modality_id)?;
        write_sparse_group_at(
            &modality_root,
            "X",
            &csr.indptr,
            &csr.indices,
            &csr.data,
            csr.shape.0,
            csr.shape.1,
        )?;

        write_h5mu_per_modality_non_x_blocks(&reader, &modality_root, modality_id, &mname)?;
    }

    Ok(())
}

/// Streaming SCX → h5mu (Phase 8). Bounds peak RSS to one shard's
/// worth of CSR per matrix written. Iterates each modality, streams
/// `/mod/{name}/X` plus all `/mod/{name}/layers/{layer}` via the
/// shard-by-shard writer in [`crate::h5ad_stream_write`]. Reuses the
/// existing materialising metadata helpers verbatim.
pub fn scx_to_h5mu_streaming(
    scx_path: &Path,
    h5mu_path: &Path,
    opts: &ConvertOptions,
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
    write_h5mu_root_attrs_and_global_blocks(&reader, &file, &root)?;

    let keep_mask = crate::h5ad_stream_write::build_keep_mask(&reader)?;
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

        write_h5mu_per_modality_non_x_blocks(&reader, &modality_root, modality_id, &mname)?;

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
) -> Result<(), ConvertError> {
    // Mark file as MuData (mudata HDF5 convention — readers may
    // look for these attributes).
    file.new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&vlu("MuData"))?;
    file.new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&vlu("0.1.0"))?;

    // Global obs.
    if let Ok(obs) = reader.read_obs() {
        write_dataframe_group_at(root, "obs", &obs)?;
    }

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
                    write_obsm_entry_at(&obsm_group, key, &batch)?;
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
) -> Result<(), ConvertError> {
    // Per-modality var.
    let var = reader.read_var_for(modality_id)?;
    write_dataframe_group_at(modality_root, "var", &var)?;

    // Per-modality obs is the shared global obs (mudata spec
    // requires per-modality obs; we point each modality at the
    // same data so downstream loaders that read /mod/{m}/obs
    // see consistent cell metadata).
    if let Ok(obs) = reader.read_obs() {
        write_dataframe_group_at(modality_root, "obs", &obs)?;
    }

    // Per-modality obsm: entries named `obsm/{mname}/{key}`.
    let mod_obsm: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| {
            e.section_type == SectionType::ObsmEmbedding && e.modality_id == modality_id
        })
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
                write_obsm_entry_at(&obsm_group, &key, &batch)?;
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

    let csr = reader.read_all_csr_shards_for(modality_id)?;
    write_sparse_group_at(
        &root,
        "X",
        &csr.indptr,
        &csr.indices,
        &csr.data,
        csr.shape.0,
        csr.shape.1,
    )?;

    write_modality_to_h5ad_non_x_blocks(&reader, &root, modality_id, modality_name)?;

    Ok(())
}

/// Streaming variant of [`scx_modality_to_h5ad`]: extracts one
/// modality from a multimodal SCX file as h5ad, streaming `/X` and
/// `/layers/{name}` shard-by-shard.
pub fn scx_modality_to_h5ad_streaming(
    scx_path: &Path,
    h5ad_path: &Path,
    modality_name: &str,
    opts: &ConvertOptions,
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

    let keep_mask = crate::h5ad_stream_write::build_keep_mask(&reader)?;

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

    write_modality_to_h5ad_non_x_blocks(&reader, &root, modality_id, modality_name)?;

    stream_layers_at(
        &root,
        &reader,
        modality_id,
        keep_mask.as_deref(),
        opts,
        sink,
    )?;

    Ok(())
}

fn write_modality_to_h5ad_non_x_blocks(
    reader: &ScxReader,
    root: &hdf5::Group,
    modality_id: u8,
    modality_name: &str,
) -> Result<(), ConvertError> {
    if let Ok(obs) = reader.read_obs() {
        write_dataframe_group_at(root, "obs", &obs)?;
    }
    let var = reader.read_var_for(modality_id)?;
    write_dataframe_group_at(root, "var", &var)?;

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
                write_obsm_entry_at(&obsm_group, &key, &batch)?;
            }
        }
    }
    Ok(())
}
