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
/// `adata.uns` is serialized as JSON. NumPy arrays/scalars, pandas
/// Index/Series/Categorical, lists, tuples, and dicts of these are converted
/// recursively. NumPy arrays become plain Python lists on readback (lossy:
/// dtype/shape are not preserved). Non-finite floats and `bytes` raise
/// `ValueError` rather than being silently coerced.
///
/// By default `from_anndata()` does not mutate `adata.X` or `adata.layers`.
/// CSR inputs with unsorted indices are copied via
/// `scipy.sparse.csr_matrix.sorted_indices()` so the caller's matrices are
/// untouched. Pass `in_place=True` to allow in-place index sorting on the
/// caller's CSR matrices (saves one allocation per matrix; matches the
/// historical behavior). Note that even with `in_place=False`, dtype
/// conversion of `indptr` / `indices` / `data` to SCX's on-disk types
/// (`int64` / `int32` / `float32`) may allocate fresh numpy arrays — those
/// allocations never alias or mutate the caller's data.
///
/// Example:
///     pyscx.from_anndata(adata, "output.scx")
///     pyscx.from_anndata(adata, "output.scx", codec="scx1", shard_size=8192)
///     pyscx.from_anndata(adata, "output.scx", in_place=True)
#[pyfunction]
#[pyo3(signature = (adata, path, codec=None, shard_size=None, in_place=false))]
fn from_anndata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    path: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
    in_place: bool,
) -> PyResult<()> {
    anndata::from_anndata_impl(py, adata, path, codec, shard_size, in_place)
}

/// Convert a 10x HDF5 file to SCX via scanpy.
///
/// Reads the 10x file with scanpy.read_10x_h5(), then writes via from_anndata.
/// `in_place` mirrors the `from_anndata()` parameter; it has no observable
/// effect because the AnnData object returned by scanpy is freshly
/// constructed and has no other reference.
#[pyfunction]
#[pyo3(signature = (h5_path, scx_path, codec=None, shard_size=None, in_place=false))]
fn from_10x(
    py: Python<'_>,
    h5_path: &str,
    scx_path: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
    in_place: bool,
) -> PyResult<()> {
    let scanpy = py.import("scanpy")?;
    let adata = scanpy.call_method1("read_10x_h5", (h5_path,))?;
    anndata::from_anndata_impl(py, &adata, scx_path, codec, shard_size, in_place)
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
    m.add_class::<scx_loader::IndexPlanDataset>()?;
    m.add_class::<backed::ScxBackedSparseDataset>()?;
    m.add_class::<backed::ScxBackedLayerDataset>()?;
    m.add_class::<backed::ScxComparisonResult>()?;
    m.add_class::<lazy_transform::ScxLazyTransformedDataset>()?;

    // Accelerators submodule.  Functions are grouped by domain into
    // `register_*` helpers so adding a new accelerator only touches one
    // helper — not this ~100-line registration block. Each helper adds
    // every function to the flat `accel_module` (so `pyscx.accel.pca`
    // continues to work — no Python-side API break).
    let accel_module = PyModule::new(m.py(), "accel")?;
    register_gpu(&accel_module)?;
    register_dim_reduction(&accel_module)?;
    register_neighbors(&accel_module)?;
    register_clustering(&accel_module)?;
    register_de(&accel_module)?;
    register_pseudobulk(&accel_module)?;
    register_batch_integration(&accel_module)?;
    register_preprocessing(&accel_module)?;
    register_filtering(&accel_module)?;
    register_hvg(&accel_module)?;
    register_eval_metrics(&accel_module)?;
    m.add_submodule(&accel_module)?;

    // Route Rust-side `log::*!` calls through Python's `logging` module so
    // Python users can configure severity/filtering/sinks via the standard
    // `logging.getLogger("pyscx")` API. Initialized once at module import;
    // a subsequent re-import is a no-op.
    let _ = pyo3_log::try_init();

    // Route Rust-side `tracing::*!` events to stderr when RUST_LOG is set
    // (DEADLOCK-ISSUE.md §2.7). Off by default — `try_init()` is a no-op
    // on subsequent imports and silent without RUST_LOG. The
    // `TrainingPipeline::{new, start_epoch, next_batch, drop, shutdown}`
    // spans + `decode` / `I/O` thread entry/exit traces from Phase 1.4
    // surface here. We use stderr rather than Python `logging` because
    // `tracing-subscriber → log → pyo3-log` would require an extra bridge
    // layer for negligible gain on a diagnostic path that is already
    // gated.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_target(true)
        .with_thread_names(true)
        .try_init();

    // Register backed classes as virtual subclasses of anndata.abc.CSRDataset.
    // This makes isinstance(x, CSRDataset) return True so AnnData accepts them.
    // Best-effort: if anndata isn't installed we skip silently (common on
    // stripped-down envs); but if the import succeeds and `register` raises
    // we surface the error via `log::warn!` so a user debugging why
    // `ad.AnnData(X=scx_backed)` rejects the object has actionable output.
    let py = m.py();
    match py.import("anndata.abc") {
        Ok(abc) => match abc.getattr("CSRDataset") {
            Ok(csr_dataset) => {
                for cls_name in [
                    "ScxBackedSparseDataset",
                    "ScxBackedLayerDataset",
                    "ScxLazyTransformedDataset",
                ] {
                    if let Err(err) = csr_dataset.call_method1("register", (m.getattr(cls_name)?,))
                    {
                        log::warn!(
                            "failed to register {cls_name} with anndata.abc.CSRDataset: {err}"
                        );
                    }
                }
            }
            Err(err) => log::warn!(
                "anndata.abc.CSRDataset lookup failed ({err}); backed datasets won't \
                 isinstance-check as CSRDataset."
            ),
        },
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyModuleNotFoundError>(py) => {
            // anndata not installed — scx is usable without it, no warning.
        }
        Err(err) => log::warn!(
            "unexpected error importing anndata.abc ({err}); backed datasets won't \
             isinstance-check as CSRDataset."
        ),
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Accelerator registration helpers
// ---------------------------------------------------------------------------
//
// Each `register_*` helper wires one domain of accelerator functions onto the
// flat `pyscx.accel` module. Grouping keeps the per-domain churn localized
// when a new function is added and documents intent at the call site in
// `pymodule`.  Every helper is a thin PyModule::add_function loop — no Python
// semantics change versus the pre-M7 flat registration.

fn register_gpu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::gpu::gpu_info, m)?)?;
    m.add_function(wrap_pyfunction!(accel::gpu::estimate_gpu_memory, m)?)?;
    Ok(())
}

fn register_dim_reduction(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::pca::pca, m)?)?;
    m.add_function(wrap_pyfunction!(accel::umap::umap, m)?)?;
    Ok(())
}

fn register_neighbors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::neighbors::neighbors, m)?)?;
    Ok(())
}

fn register_clustering(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::leiden::leiden, m)?)?;
    Ok(())
}

fn register_de(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::de::rank_genes_groups, m)?)?;
    m.add_function(wrap_pyfunction!(accel::de::rank_genes_groups_df, m)?)?;
    Ok(())
}

fn register_pseudobulk(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::pseudobulk::pseudobulk_dex, m)?)?;
    Ok(())
}

fn register_batch_integration(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::harmony::harmony_integrate, m)?)?;
    m.add_function(wrap_pyfunction!(accel::lisi::compute_lisi, m)?)?;
    Ok(())
}

fn register_preprocessing(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::preprocessing::normalize_total, m)?)?;
    m.add_function(wrap_pyfunction!(accel::preprocessing::log1p, m)?)?;
    m.add_function(wrap_pyfunction!(
        accel::preprocessing::calculate_qc_metrics,
        m
    )?)?;
    #[cfg(feature = "gpu")]
    m.add_class::<accel::preprocessing::ScxGpuNormalizeMarker>()?;
    Ok(())
}

fn register_filtering(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::filtering::filter_cells, m)?)?;
    m.add_function(wrap_pyfunction!(accel::filtering::filter_genes, m)?)?;
    m.add_function(wrap_pyfunction!(accel::filtering::subset_obs, m)?)?;
    Ok(())
}

fn register_hvg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::hvg::highly_variable_genes, m)?)?;
    Ok(())
}

fn register_eval_metrics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accel::eval_metrics::pseudobulk_means, m)?)?;
    m.add_function(wrap_pyfunction!(
        accel::eval_metrics::perturbation_metrics,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(accel::eval_metrics::energy_distance, m)?)?;
    m.add_function(wrap_pyfunction!(
        accel::eval_metrics::energy_distance_details,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        accel::eval_metrics::discrimination_score,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        accel::eval_metrics::knockdown_efficiency,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        accel::eval_metrics::clustering_agreement,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        accel::eval_metrics::adjusted_mutual_info,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        accel::eval_metrics::normalized_mutual_info,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        accel::eval_metrics::adjusted_rand_index,
        m
    )?)?;
    Ok(())
}
