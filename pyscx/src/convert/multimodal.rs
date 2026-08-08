// Per-modality backed AnnData assembly.
//
// Extracted from the former pyscx/src/anndata.rs (T5.7).

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;

use scx_format_io::section::SectionType;

use crate::to_pyerr;

use super::*;

/// Assemble a backed AnnData scoped to a single modality. `obs_df` is the
/// pre-built pandas DataFrame (shared across modalities); pass `None` to
/// construct without an obs attached (rare).
#[allow(clippy::too_many_arguments)]
pub fn build_backed_anndata_for_modality<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    shared_catalog: &Arc<scx_format_io::FullCatalog>,
    modality_id: u8,
    modality_name: &str,
    cache_shards: usize,
    obs_df: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    use crate::backed::ScxBackedSparseDataset;
    use scx_format_io::BackedCsrReader;

    let anndata_mod = py.import("anndata")?;

    // Meta reader for var / obsm / modality_info introspection.
    let meta =
        crate::open_handle_reader_shared(path, Arc::clone(shared_catalog)).map_err(to_pyerr)?;
    let info = meta.modality_info(modality_id).ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "modality_info({modality_id}) returned None — modality table is corrupt"
        ))
    })?;
    let has_csc = info.flags.has_csc();

    // Per-modality backed X (CSR).
    let csr_reader =
        crate::open_handle_reader_shared(path, Arc::clone(shared_catalog)).map_err(to_pyerr)?;
    let backed_csr = Arc::new(BackedCsrReader::for_modality(
        csr_reader,
        modality_id,
        cache_shards,
    ));

    // Per-modality backed CSC sidecar (optional).
    let backed_csc = if has_csc {
        let csc_reader =
            crate::open_handle_reader_shared(path, Arc::clone(shared_catalog)).map_err(to_pyerr)?;
        Some(Arc::new(
            scx_format_io::BackedCscReader::for_modality(csc_reader, modality_id, cache_shards)
                .map_err(to_pyerr)?,
        ))
    } else {
        None
    };

    let mut x_dataset = ScxBackedSparseDataset::from_reader(Arc::clone(&backed_csr), cache_shards);
    x_dataset.with_csc_reader(backed_csc);
    x_dataset.with_modality_id(modality_id);
    x_dataset.with_source_path(path);

    // Per-modality var.
    let var_batch = meta.read_var_for(modality_id).map_err(to_pyerr)?;
    let var_table = record_batch_to_pyarrow(py, &var_batch)?;
    let var_df = pyarrow_table_to_pandas(&var_table)?;

    // Per-modality obsm: catalog entries with this modality_id.
    let obsm_dict = pyo3::types::PyDict::new(py);
    let prefix = format!("obsm/{modality_name}/");
    for entry in &meta.catalog().entries {
        if entry.section_type != SectionType::ObsmEmbedding {
            continue;
        }
        if entry.modality_id != modality_id {
            continue;
        }
        let key = entry
            .name
            .strip_prefix(&prefix)
            .unwrap_or(&entry.name)
            .to_string();
        let batch = meta.read_obsm_for(modality_id, &key).map_err(to_pyerr)?;
        let np_arr = obsm_batch_to_numpy(py, &batch)?;
        obsm_dict.set_item(&key, np_arr)?;
    }

    // Assemble AnnData kwargs.
    let kwargs = pyo3::types::PyDict::new(py);
    let x_py = x_dataset.into_pyobject(py)?;
    kwargs.set_item("X", x_py)?;
    if let Some(obs) = obs_df {
        kwargs.set_item("obs", obs)?;
    }
    kwargs.set_item("var", var_df)?;
    if !obsm_dict.is_empty() {
        kwargs.set_item("obsm", obsm_dict)?;
    }

    anndata_mod.call_method("AnnData", (), Some(&kwargs))
}

/// Public entrypoint: open `path`, resolve `modality` → `modality_id`, build a
/// backed AnnData wrapping that modality's CSR (+ CSC sidecar if present),
/// with the global obs attached.
pub fn to_anndata_backed_for_modality<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    modality: &str,
    cache_shards: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let reader = crate::open_handle_reader(path).map_err(to_pyerr)?;
    let modality_id = reader.modality_id(modality).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!(
            "unknown modality '{modality}' (available: {:?})",
            reader.modality_names()
        ))
    })?;
    let modality_name = reader
        .modality_info(modality_id)
        .map(|i| i.name.clone())
        .unwrap_or_else(|| modality.to_string());

    let obs_df = match reader.read_obs() {
        Ok(batch) => {
            let table = record_batch_to_pyarrow(py, &batch)?;
            Some(pyarrow_table_to_pandas(&table)?)
        }
        Err(scx_format_io::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    let shared_catalog = reader.catalog_arc();
    build_backed_anndata_for_modality(
        py,
        path,
        &shared_catalog,
        modality_id,
        &modality_name,
        cache_shards,
        obs_df.as_ref(),
    )
}
