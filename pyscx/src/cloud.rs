//! Python bindings for scx-cloud operations.
//!
//! Provides PyO3 wrappers for pull, push, cloud_optimize, explode, and pack.
//! All async operations create a tokio runtime internally.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Pull from cloud/local exploded .scxd into a local .scx file.
///
/// Args:
///     source: Source URL or path (e.g., "gs://bucket/data.scxd/")
///     dest: Output .scx file path
///     filter: Optional predicate expression for selective pull
///     parallelism: Number of parallel download tasks (default: 8)
///
/// Returns:
///     dict with keys: bytes_downloaded, sections_downloaded, elapsed_secs,
///     throughput_mbps. For filtered pulls: total_shards, downloaded_shards,
///     skipped_shards, matching_cells, bytes_saved.
#[pyfunction]
#[pyo3(signature = (source, dest, filter=None, parallelism=None))]
pub fn pull(
    py: Python<'_>,
    source: &str,
    dest: &str,
    filter: Option<&str>,
    parallelism: Option<usize>,
) -> PyResult<PyObject> {
    let opts = scx_cloud::PullOptions {
        parallelism: parallelism.unwrap_or(8),
        reorder_buffer: 4,
        cloud_ready: true,
    };

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to create runtime: {e}")))?;

    let dest_path = std::path::PathBuf::from(dest);

    // Release the GIL during blocking cloud I/O (finding 9.3).
    if let Some(filter_expr) = filter {
        let source = source.to_string();
        let filter_expr = filter_expr.to_string();
        let stats = py
            .allow_threads(|| {
                rt.block_on(scx_cloud::pull_filtered(
                    &source,
                    &dest_path,
                    &filter_expr,
                    opts,
                ))
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("total_shards", stats.total_shards)?;
        dict.set_item("downloaded_shards", stats.downloaded_shards)?;
        dict.set_item("skipped_shards", stats.skipped_shards)?;
        dict.set_item("matching_cells", stats.matching_cells)?;
        dict.set_item("bytes_downloaded", stats.bytes_downloaded)?;
        dict.set_item("bytes_saved", stats.bytes_saved)?;
        dict.set_item("elapsed_secs", stats.elapsed.as_secs_f64())?;
        Ok(dict.into())
    } else {
        let source = source.to_string();
        let stats = py
            .allow_threads(|| rt.block_on(scx_cloud::pull(&source, &dest_path, opts)))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("bytes_downloaded", stats.bytes_downloaded)?;
        dict.set_item("sections_downloaded", stats.sections_downloaded)?;
        dict.set_item("elapsed_secs", stats.elapsed.as_secs_f64())?;
        dict.set_item("throughput_mbps", stats.throughput_mbps)?;
        Ok(dict.into())
    }
}

/// Push a local .scx file to cloud/local as exploded .scxd directory.
///
/// Args:
///     source: Local .scx file path
///     dest: Destination URL or path (e.g., "gs://bucket/data.scxd/")
///     parallelism: Number of parallel upload tasks (default: 8)
///
/// Returns:
///     dict with keys: bytes_uploaded, sections_uploaded, elapsed_secs, throughput_mbps
#[pyfunction]
#[pyo3(signature = (source, dest, parallelism=None))]
pub fn push(
    py: Python<'_>,
    source: &str,
    dest: &str,
    parallelism: Option<usize>,
) -> PyResult<PyObject> {
    let opts = scx_cloud::PushOptions {
        parallelism: parallelism.unwrap_or(8),
        multipart_threshold: 8 * 1024 * 1024,
    };

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to create runtime: {e}")))?;

    let source_path = std::path::PathBuf::from(source);
    let dest = dest.to_string();
    // Release the GIL during blocking cloud I/O (finding 9.3).
    let stats = py
        .allow_threads(|| rt.block_on(scx_cloud::push(&source_path, &dest, opts)))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("bytes_uploaded", stats.bytes_uploaded)?;
    dict.set_item("sections_uploaded", stats.sections_uploaded)?;
    dict.set_item("elapsed_secs", stats.elapsed.as_secs_f64())?;
    dict.set_item("throughput_mbps", stats.throughput_mbps)?;
    Ok(dict.into())
}

/// Cloud-optimize an SCX file by adding a front-of-file catalog.
///
/// Args:
///     input: Input .scx file path
///     output: Optional output path (default: rewrite in-place via atomic rename)
#[pyfunction]
#[pyo3(signature = (input, output=None))]
pub fn cloud_optimize(input: &str, output: Option<&str>) -> PyResult<()> {
    let input_path = std::path::Path::new(input);
    let output_path = output
        .map(std::path::Path::new)
        .unwrap_or(input_path);
    scx_cloud::cloud_optimize(input_path, output_path)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Explode a packed .scx file into a cloud-deployable .scxd directory.
///
/// Args:
///     input: Input .scx file path
///     output: Output directory path (should end in .scxd/)
#[pyfunction]
pub fn explode(input: &str, output: &str) -> PyResult<()> {
    let input_path = std::path::Path::new(input);
    let output_path = std::path::Path::new(output);
    scx_cloud::explode(input_path, output_path)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Pack an exploded .scxd directory back into a single .scx file.
///
/// Args:
///     input: Input directory path (should end in .scxd/)
///     output: Output .scx file path
#[pyfunction]
pub fn pack(input: &str, output: &str) -> PyResult<()> {
    let input_path = std::path::Path::new(input);
    let output_path = std::path::Path::new(output);
    scx_cloud::pack(input_path, output_path)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// A cloud-backed experiment handle for reading SCX data
/// without downloading the entire file.
///
/// Provides metadata accessors (n_obs, n_vars, nnz, shard_count)
/// and read methods for obs/var metadata.
#[pyclass]
pub struct PyCloudExperiment {
    reader: scx_cloud::CloudReader,
    #[allow(dead_code)] // Kept alive to own the tokio runtime for the reader's lifetime.
    rt: tokio::runtime::Runtime,
}

#[pymethods]
impl PyCloudExperiment {
    #[getter]
    fn n_obs(&self) -> u64 {
        self.reader.n_obs()
    }

    #[getter]
    fn n_vars(&self) -> u64 {
        self.reader.n_vars()
    }

    #[getter]
    fn nnz(&self) -> u64 {
        self.reader.nnz()
    }

    #[getter]
    fn shard_count(&self) -> u32 {
        self.reader.n_shards()
    }

    #[getter]
    fn format_version(&self) -> u16 {
        self.reader.header().format_version
    }

    #[getter]
    fn codec_id(&self) -> u8 {
        self.reader.header().codec_id
    }

    fn __repr__(&self) -> String {
        format!(
            "PyCloudExperiment(n_obs={}, n_vars={}, nnz={}, shards={})",
            self.reader.n_obs(),
            self.reader.n_vars(),
            self.reader.nnz(),
            self.reader.n_shards(),
        )
    }
}

/// Open an SCX file or exploded directory from cloud/local storage.
///
/// Detects layout automatically: exploded .scxd directories,
/// cloud-ready packed .scx, or non-cloud-ready packed .scx.
///
/// Args:
///     url: Source URL or local path (e.g., "gs://bucket/data.scxd/")
///
/// Returns:
///     PyCloudExperiment handle with metadata accessors
#[pyfunction]
pub fn open_cloud(py: Python<'_>, url: &str) -> PyResult<PyCloudExperiment> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to create runtime: {e}")))?;

    // Release the GIL during blocking cloud I/O (finding 9.3).
    let url = url.to_string();
    let reader = py
        .allow_threads(|| rt.block_on(scx_cloud::open_cloud(&url)))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyCloudExperiment { reader, rt })
}
