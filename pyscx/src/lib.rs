mod anndata;
mod experiment;
mod ops;
mod query;

#[cfg(feature = "cloud")]
mod cloud;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use experiment::PyExperiment;
use query::{PyQueryPipeline, PyQueryResult};
use scx_format::ScxError;

/// Convert an ScxError into a Python RuntimeError.
fn to_pyerr(e: ScxError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Open an SCX file and return a PyExperiment handle.
///
/// Example:
///     exp = pyscx.open("data.scx")
///     adata = exp.to_anndata()
#[pyfunction]
fn open(path: &str) -> PyResult<PyExperiment> {
    let reader = scx_format::ScxReader::open(path).map_err(to_pyerr)?;
    Ok(PyExperiment::new(reader, std::path::PathBuf::from(path)))
}

/// Convert an AnnData object to an SCX file.
///
/// Codec defaults to "auto", which selects the best codec based on value
/// distribution (Scx1 for small UMI counts, Zstd for large values or floats).
/// Explicit options: "none", "scx1", "zstd".
///
/// Example:
///     pyscx.from_anndata(adata, "output.scx")
///     pyscx.from_anndata(adata, "output.scx", codec="scx1", shard_size=8192)
#[pyfunction]
#[pyo3(signature = (adata, path, codec=None, shard_size=None))]
fn from_anndata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    path: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
) -> PyResult<()> {
    anndata::from_anndata_impl(py, adata, path, codec, shard_size)
}

/// Convert a 10x HDF5 file to SCX via scanpy.
///
/// Reads the 10x file with scanpy.read_10x_h5(), then writes via from_anndata.
#[pyfunction]
#[pyo3(signature = (h5_path, scx_path, codec=None, shard_size=None))]
fn from_10x(
    py: Python<'_>,
    h5_path: &str,
    scx_path: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
) -> PyResult<()> {
    let scanpy = py.import("scanpy")?;
    let adata = scanpy.call_method1("read_10x_h5", (h5_path,))?;
    anndata::from_anndata_impl(py, &adata, scx_path, codec, shard_size)
}

#[pymodule]
fn pyscx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core I/O
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(from_anndata, m)?)?;
    m.add_function(wrap_pyfunction!(from_10x, m)?)?;

    // File operations (scx-ops)
    m.add_function(wrap_pyfunction!(ops::append, m)?)?;
    m.add_function(wrap_pyfunction!(ops::append_from_anndata, m)?)?;
    m.add_function(wrap_pyfunction!(ops::mark_deleted, m)?)?;
    m.add_function(wrap_pyfunction!(ops::compact, m)?)?;
    m.add_function(wrap_pyfunction!(ops::rollback, m)?)?;
    m.add_function(wrap_pyfunction!(ops::merge, m)?)?;

    // Cloud operations (optional, behind "cloud" feature)
    #[cfg(feature = "cloud")]
    {
        m.add_function(wrap_pyfunction!(cloud::pull, m)?)?;
        m.add_function(wrap_pyfunction!(cloud::push, m)?)?;
        m.add_function(wrap_pyfunction!(cloud::cloud_optimize, m)?)?;
        m.add_function(wrap_pyfunction!(cloud::explode, m)?)?;
        m.add_function(wrap_pyfunction!(cloud::pack, m)?)?;
        m.add_function(wrap_pyfunction!(cloud::open_cloud, m)?)?;
        m.add_class::<cloud::PyCloudExperiment>()?;
    }

    // Classes
    m.add_class::<PyExperiment>()?;
    m.add_class::<PyQueryPipeline>()?;
    m.add_class::<PyQueryResult>()?;
    m.add_class::<scx_loader::TrainingDataset>()?;
    Ok(())
}

