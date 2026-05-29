// Pure-Rust h5ad metadata reader.
//
// Reads obs / var / uns / shape from an h5ad file without going through
// `anndata.read_h5ad`. The point of this module is to dodge anndata's
// eager `obsm` materialisation: callers that only need obs / var / uns
// (e.g. to mutate them before `pyscx.from_h5ad(..., obs_override=, ...)`)
// can call this and pay metadata bytes only.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::anndata::{
    emit_python_warnings, pyarrow_table_to_pandas, record_batch_to_pyarrow, uns_json_to_py,
};

/// Metadata snapshot for an h5ad file returned by
/// [`read_h5ad_metadata`]. Reads obs / var / uns and the X shape +
/// storage format eagerly; never touches obsm / varm / obsp / varp /
/// layers / X data.
///
/// Use this when you need to read `obs` / `var` / `uns` cheaply
/// (e.g. to merge in caller-side annotations before writing an SCX file
/// with [`pyscx.from_h5ad`]) without paying the 50–100 GB obsm
/// allocation that `anndata.read_h5ad(path, backed="r")` triggers on
/// every call, including in backed mode.
#[pyclass(module = "pyscx", name = "H5adMetadata")]
pub struct PyH5adMetadata {
    #[pyo3(get)]
    obs: Py<PyAny>,
    #[pyo3(get)]
    var: Py<PyAny>,
    #[pyo3(get)]
    uns: Py<PyAny>,
    #[pyo3(get)]
    n_obs: usize,
    #[pyo3(get)]
    n_vars: usize,
    #[pyo3(get)]
    x_format: String,
}

#[pymethods]
impl PyH5adMetadata {
    fn __repr__(&self) -> String {
        format!(
            "H5adMetadata(n_obs={}, n_vars={}, x_format='{}')",
            self.n_obs, self.n_vars, self.x_format
        )
    }
}

/// Read obs, var, uns, and shape from an h5ad file via the scx-convert
/// pure-Rust HDF5 readers — no `anndata` import, no `obsm` allocation.
///
/// Args:
///     path: Path to the h5ad file.
///     strict_uns: If True, fail on the first unsupported `uns` key.
///         Default False matches the existing convert behaviour
///         (skip-and-warn).
///
/// Returns:
///     `H5adMetadata` with attributes `obs` (pandas.DataFrame),
///     `var` (pandas.DataFrame), `uns` (dict), `n_obs` (int),
///     `n_vars` (int), `x_format` (`"csr"` / `"csc"` / `"dense"`).
///
/// Example:
///     meta = pyscx.read_h5ad_metadata("big.h5ad")
///     meta.obs["condition_id"] = ...
///     meta.uns["pipeline_version"] = "1.2.3"
///     pyscx.from_h5ad(
///         "big.h5ad", "big.scx",
///         obs_override=meta.obs,
///         uns_override=meta.uns,
///     )
#[cfg(feature = "hdf5")]
#[pyfunction]
#[pyo3(signature = (path, strict_uns=false))]
pub fn read_h5ad_metadata(
    py: Python<'_>,
    path: &str,
    strict_uns: bool,
) -> PyResult<PyH5adMetadata> {
    let path_buf = std::path::PathBuf::from(path);
    let mut sink = scx_convert::WarningSink::log();
    let parts = py
        .detach(|| scx_convert::read_h5ad_metadata_from_path(&path_buf, strict_uns, &mut sink))
        .map_err(|e| PyRuntimeError::new_err(format!("read_h5ad_metadata('{path}'): {e}")))?;
    emit_python_warnings(py, &sink)?;

    if parts.obs.num_rows() != parts.n_obs {
        return Err(PyValueError::new_err(format!(
            "h5ad inconsistency: obs has {} rows but X has n_obs={}",
            parts.obs.num_rows(),
            parts.n_obs
        )));
    }
    if parts.var.num_rows() != parts.n_vars {
        return Err(PyValueError::new_err(format!(
            "h5ad inconsistency: var has {} rows but X has n_vars={}",
            parts.var.num_rows(),
            parts.n_vars
        )));
    }

    let obs_table = record_batch_to_pyarrow(py, &parts.obs)?;
    let obs_df = pyarrow_table_to_pandas(&obs_table)?;
    let var_table = record_batch_to_pyarrow(py, &parts.var)?;
    let var_df = pyarrow_table_to_pandas(&var_table)?;

    let uns_py = match parts.uns.as_ref() {
        Some(json) => uns_json_to_py(py, json)?,
        None => PyDict::new(py).into_any(),
    };

    Ok(PyH5adMetadata {
        obs: obs_df.unbind(),
        var: var_df.unbind(),
        uns: uns_py.unbind(),
        n_obs: parts.n_obs,
        n_vars: parts.n_vars,
        x_format: parts.x_format.to_string(),
    })
}
