mod accel;
mod anndata;
pub(crate) mod backed;
mod experiment;
pub(crate) mod lazy_transform;
mod ops;
mod preprocess;
pub(crate) mod projected_agg;
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
/// Args:
///     path: Path to the SCX file.
///     verify: If True (default), verify the catalog BLAKE3 checksum on open.
///         Set to False for performance-sensitive paths where the file is
///         trusted (e.g., repeated reads of a file that was already validated).
///
/// Example:
///     exp = pyscx.open("data.scx")
///     adata = exp.to_anndata()
///     # Fast open for trusted files:
///     exp = pyscx.open("data.scx", verify=False)
#[pyfunction]
#[pyo3(signature = (path, verify=true))]
fn open(path: &str, verify: bool) -> PyResult<PyExperiment> {
    let reader = if verify {
        scx_format::ScxReader::open(path)
    } else {
        scx_format::ScxReader::open_unchecked(path)
    }
    .map_err(to_pyerr)?;
    Ok(PyExperiment::new(reader, std::path::PathBuf::from(path)))
}

/// Validate all section checksums in an SCX file.
///
/// Opens the file with full catalog verification, then checks every section's
/// BLAKE3 checksum against the catalog. Returns a list of (section_name, passed)
/// tuples. Raises RuntimeError if any essential section (obs, var, CsrShard) fails.
///
/// Example:
///     results = pyscx.validate("data.scx")
///     for name, passed in results:
///         print(f"{name}: {'OK' if passed else 'FAIL'}")
#[pyfunction]
fn validate(path: &str) -> PyResult<Vec<(String, bool)>> {
    let reader = scx_format::ScxReader::open(path).map_err(to_pyerr)?;
    reader.validate().map_err(to_pyerr)
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

/// Convert a Cell Ranger MTX directory to SCX.
///
/// Reads the MTX directory (matrix.mtx[.gz], barcodes.tsv[.gz], features.tsv[.gz])
/// and writes an SCX file.
///
/// Example:
///     pyscx.from_mtx("/path/to/filtered_feature_bc_matrix", "output.scx")
#[pyfunction]
#[pyo3(signature = (mtx_dir, scx_path, codec=None, shard_size=None))]
fn from_mtx(
    mtx_dir: &str,
    scx_path: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
) -> PyResult<()> {
    scx_mtx::mtx_to_scx(
        std::path::Path::new(mtx_dir),
        std::path::Path::new(scx_path),
        shard_size.unwrap_or(16384),
        codec.unwrap_or("auto"),
        "pyscx",
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Convert an SCX file to a Cell Ranger–style MTX directory.
///
/// Output directory will contain: matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz
///
/// Example:
///     pyscx.to_mtx("data.scx", "/path/to/output_dir")
#[pyfunction]
fn to_mtx(scx_path: &str, output_dir: &str) -> PyResult<()> {
    scx_mtx::write_scx_to_mtx(
        std::path::Path::new(scx_path),
        std::path::Path::new(output_dir),
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn pyscx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core I/O
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(validate, m)?)?;
    m.add_function(wrap_pyfunction!(from_anndata, m)?)?;
    m.add_function(wrap_pyfunction!(from_10x, m)?)?;
    m.add_function(wrap_pyfunction!(from_mtx, m)?)?;
    m.add_function(wrap_pyfunction!(to_mtx, m)?)?;

    // Preprocessing pipeline
    m.add_function(wrap_pyfunction!(preprocess::preprocess, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess::save_layer, m)?)?;

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
    m.add_class::<backed::ScxBackedSparseDataset>()?;
    m.add_class::<backed::ScxBackedLayerDataset>()?;
    m.add_class::<backed::ScxComparisonResult>()?;
    m.add_class::<lazy_transform::ScxLazyTransformedDataset>()?;

    // Accelerators submodule
    let accel_module = PyModule::new(m.py(), "accel")?;
    accel_module.add_function(wrap_pyfunction!(accel::gpu_info, &accel_module)?)?;
    accel_module.add_function(wrap_pyfunction!(accel::estimate_gpu_memory, &accel_module)?)?;
    accel_module.add_function(wrap_pyfunction!(accel::pca, &accel_module)?)?;
    accel_module.add_function(wrap_pyfunction!(accel::neighbors, &accel_module)?)?;
    accel_module.add_function(wrap_pyfunction!(accel::umap, &accel_module)?)?;
    accel_module.add_function(wrap_pyfunction!(accel::rank_genes_groups, &accel_module)?)?;
    accel_module.add_function(wrap_pyfunction!(accel::pseudobulk_dex, &accel_module)?)?;
    accel_module.add_function(wrap_pyfunction!(accel::leiden, &accel_module)?)?;
    accel_module.add_function(wrap_pyfunction!(accel::normalize_total, &accel_module)?)?;
    accel_module.add_function(wrap_pyfunction!(accel::log1p, &accel_module)?)?;
    accel_module.add_function(wrap_pyfunction!(
        accel::calculate_qc_metrics,
        &accel_module
    )?)?;
    accel_module.add_function(wrap_pyfunction!(accel::filter_cells, &accel_module)?)?;
    accel_module.add_function(wrap_pyfunction!(accel::filter_genes, &accel_module)?)?;
    accel_module.add_function(wrap_pyfunction!(accel::subset_obs, &accel_module)?)?;
    accel_module.add_function(wrap_pyfunction!(
        accel::highly_variable_genes,
        &accel_module
    )?)?;
    accel_module.add_function(wrap_pyfunction!(accel::pseudobulk_means, &accel_module)?)?;
    accel_module.add_function(wrap_pyfunction!(
        accel::perturbation_metrics,
        &accel_module
    )?)?;
    accel_module.add_function(wrap_pyfunction!(accel::energy_distance, &accel_module)?)?;
    accel_module.add_function(wrap_pyfunction!(
        accel::discrimination_score,
        &accel_module
    )?)?;
    accel_module.add_function(wrap_pyfunction!(
        accel::knockdown_efficiency,
        &accel_module
    )?)?;
    accel_module.add_function(wrap_pyfunction!(
        accel::clustering_agreement,
        &accel_module
    )?)?;
    accel_module.add_function(wrap_pyfunction!(
        accel::rank_genes_groups_df,
        &accel_module
    )?)?;
    m.add_submodule(&accel_module)?;

    // Register backed classes as virtual subclasses of anndata.abc.CSRDataset.
    // This makes isinstance(x, CSRDataset) return True so AnnData accepts them.
    // Best-effort: if anndata isn't installed, skip silently.
    let py = m.py();
    if let Ok(abc) = py.import("anndata.abc") {
        if let Ok(csr_dataset) = abc.getattr("CSRDataset") {
            let _ = csr_dataset.call_method1("register", (m.getattr("ScxBackedSparseDataset")?,));
            let _ = csr_dataset.call_method1("register", (m.getattr("ScxBackedLayerDataset")?,));
            let _ =
                csr_dataset.call_method1("register", (m.getattr("ScxLazyTransformedDataset")?,));
        }
    }

    Ok(())
}
