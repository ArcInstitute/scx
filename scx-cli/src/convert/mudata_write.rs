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

use super::h5ad_write::{
    write_dataframe_group_at, write_obsm_entry_at, write_sparse_group_at, write_uns_entries_at,
};
use super::pipeline::ConvertError;

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
        write_dataframe_group_at(&root, "obs", &obs)?;
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

    // Per-modality blocks under /mod/{name}.
    let mod_group = root.create_group("mod")?;
    for modality_id in 1..=reader.n_modalities() as u8 {
        let info = reader.modality_info(modality_id).ok_or_else(|| {
            ConvertError::Other(format!(
                "modality_id {modality_id} not found in modality table"
            ))
        })?;
        let mname = info.name.clone();
        let modality_root = mod_group.create_group(&mname)?;

        // anndata-style group attributes for tooling that walks h5mu.
        modality_root
            .new_attr::<VarLenUnicode>()
            .create("encoding-type")?
            .write_scalar(&vlu("anndata"))?;
        modality_root
            .new_attr::<VarLenUnicode>()
            .create("encoding-version")?
            .write_scalar(&vlu("0.1.0"))?;

        // Per-modality X matrix.
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

        // Per-modality var.
        let var = reader.read_var_for(modality_id)?;
        write_dataframe_group_at(&modality_root, "var", &var)?;

        // Per-modality obs is the shared global obs (mudata spec
        // requires per-modality obs; we point each modality at the
        // same data so downstream loaders that read /mod/{m}/obs
        // see consistent cell metadata).
        if let Ok(obs) = reader.read_obs() {
            write_dataframe_group_at(&modality_root, "obs", &obs)?;
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

    if let Ok(obs) = reader.read_obs() {
        write_dataframe_group_at(&root, "obs", &obs)?;
    }
    let var = reader.read_var_for(modality_id)?;
    write_dataframe_group_at(&root, "var", &var)?;

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
