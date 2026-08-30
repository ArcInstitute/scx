//! GPU device info, memory estimation, and device resolution.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Whether GPU support is available — `True` iff `pyscx` was built with
/// `--features gpu` **and** a CUDA device is detected at runtime.
///
/// Use this when you only need a boolean check; `gpu_info()` is the right
/// call when you need device name / VRAM / etc. (it returns `None` in the
/// same cases this function returns `False`).
///
/// Example:
///     if pyscx.accel.gpu_available():
///         pyscx.accel.pca(adata, device="gpu")
///     else:
///         pyscx.accel.pca(adata, device="cpu")
#[pyfunction]
pub fn gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        scx_accel::gpu_available()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Query GPU device information.
///
/// Returns a dict with keys `device` (name string), `total_vram_gb` (f64),
/// and `free_vram_gb` (f64). Returns `None` if no GPU is available or the
/// `gpu` feature is disabled — in those cases `gpu_available()` returns
/// `False`. Use `gpu_available()` for a yes/no check, this for device info.
///
/// Example:
///     info = pyscx.accel.gpu_info()
///     if info is not None:
///         print(f"GPU: {info['device']}, {info['free_vram_gb']:.1f} GB free")
#[pyfunction]
pub fn gpu_info(py: Python<'_>) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "gpu")]
    {
        match scx_accel::gpu_info() {
            Some(info) => {
                let dict = PyDict::new(py);
                dict.set_item("device", info.device_name)?;
                dict.set_item(
                    "total_vram_gb",
                    info.total_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                )?;
                dict.set_item(
                    "free_vram_gb",
                    info.free_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                )?;
                Ok(dict.into_any().unbind())
            }
            None => Ok(py.None()),
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        Ok(py.None())
    }
}

/// Estimate GPU memory required for a given operation on the dataset.
///
/// Accepts an AnnData object (for shape/nnz), an operation name, and optional
/// keyword arguments for operation-specific parameters. Returns a dict with
/// `required_gb` (f64) and `fits_in_vram` (bool).
///
/// Supported operations: `"pca"`, `"knn"`, `"umap"`, `"leiden"`.
///
/// Example:
///     est = pyscx.accel.estimate_gpu_memory(adata, "pca", n_components=50)
///     if est["fits_in_vram"]:
///         pyscx.accel.pca(adata, device="gpu")
#[pyfunction]
#[pyo3(signature = (adata, operation, **kwargs))]
pub fn estimate_gpu_memory<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    operation: &str,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Py<PyAny>> {
    // Helper to extract a kwarg with a default
    let get_kwarg_usize = |key: &str, default: usize| -> PyResult<usize> {
        if let Some(kw) = kwargs {
            if let Some(val) = kw.get_item(key)? {
                return val.extract::<usize>();
            }
        }
        Ok(default)
    };

    // Extract shape from adata
    let n_obs: usize = adata.getattr("n_obs")?.extract()?;
    let n_vars: usize = adata.getattr("n_vars")?.extract()?;

    // Extract nnz (best-effort): try X.nnz, then X.data.shape[0], then estimate
    let nnz: usize = {
        let x = adata.getattr("X")?;
        if let Ok(val) = x.getattr("nnz").and_then(|v| v.extract::<usize>()) {
            val
        } else if let Ok(data) = x.getattr("data") {
            if let Ok(shape) = data.getattr("shape") {
                let shape_tuple: Vec<usize> = shape.extract()?;
                shape_tuple.first().copied().unwrap_or(n_obs * n_vars / 10)
            } else {
                n_obs * n_vars / 10 // fallback: assume ~10% density
            }
        } else {
            n_obs * n_vars / 10
        }
    };

    // Compute estimated bytes per operation.
    // All GPU paths use f32 (4 bytes per element).
    let estimated_bytes: usize = match operation {
        "pca" => {
            let n_components = get_kwarg_usize("n_components", 50)?;
            let n_oversamples = get_kwarg_usize("n_oversamples", 10)?;
            let k = n_components + n_oversamples;
            // Default shard size for SCX files
            let shard_size = get_kwarg_usize("shard_size", 16384)?;
            let shard_rows = shard_size.min(n_obs);

            // The in-VRAM covariance PCA path was removed; the native GPU PCA
            // path is always randomized now, so the footprint is the randomized
            // one: Y/Q (n_obs × k), Ω + B (n_vars × k), one decoded shard,
            // cuSOLVER QR workspace.
            let y_bytes = n_obs * k * 4;
            let omega_b_bytes = 2 * n_vars * k * 4;
            let shard_dense_bytes = shard_rows * n_vars * 4;
            let qr_workspace = 2 * n_obs * k * 4;
            y_bytes + omega_b_bytes + shard_dense_bytes + qr_workspace
        }
        "knn" => {
            let n_neighbors = get_kwarg_usize("n_neighbors", 15)?;
            let n_dims = get_kwarg_usize("n_dims", 50)?; // typically PCA dimensions

            // Embeddings on GPU: n_obs × n_dims × 4 (f32)
            let embeddings_bytes = n_obs * n_dims * 4;
            // CAGRA index: ~n_obs × n_neighbors × 8 (neighbor id u32 + distance f32)
            let index_bytes = n_obs * n_neighbors * 8;

            embeddings_bytes + index_bytes
        }
        "umap" => {
            let n_components = get_kwarg_usize("n_components", 2)?;

            // Embedding: n_obs × n_components × 4 (f32)
            let embedding_bytes = n_obs * n_components * 4;
            // Edge array: ~nnz × 16 (src u32 + dst u32 + weight f32 + epoch f32)
            // Use nnz from the kNN graph (obsp["connectivities"]), approximate with
            // n_obs * n_neighbors * 2 if we don't know. Use nnz from X as upper bound.
            let n_edges = nnz.min(n_obs * 30); // cap at reasonable kNN graph size
            let edge_bytes = n_edges * 16;

            embedding_bytes + edge_bytes
        }
        "leiden" => {
            // CSR on GPU for the kNN graph
            // indptr: (n_obs + 1) × 8 (i64)
            let indptr_bytes = (n_obs + 1) * 8;
            // indices: nnz × 4 (i32) + data: nnz × 4 (f32)
            let n_edges = nnz.min(n_obs * 30); // cap at kNN graph size
            let csr_bytes = n_edges * (4 + 4);
            // Partition array: n_obs × 4 (i32)
            let partition_bytes = n_obs * 4;
            // Working buffers: ~2 × n_obs × 4
            let work_bytes = 2 * n_obs * 4;

            indptr_bytes + csr_bytes + partition_bytes + work_bytes
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown operation: '{}'. Supported: 'pca', 'knn', 'umap', 'leiden'",
                operation
            )));
        }
    };

    let required_gb = estimated_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    // Query available VRAM
    let fits_in_vram: bool;

    #[cfg(feature = "gpu")]
    {
        fits_in_vram = match scx_accel::gpu_info() {
            Some(info) => {
                let free_bytes = info.free_vram_bytes;
                estimated_bytes < free_bytes
            }
            None => false,
        };
    }
    #[cfg(not(feature = "gpu"))]
    {
        fits_in_vram = false;
    }

    let dict = PyDict::new(py);
    dict.set_item("required_gb", required_gb)?;
    dict.set_item("fits_in_vram", fits_in_vram)?;
    Ok(dict.into_any().unbind())
}

/// Snapshot the GPU per-stage timing profiler.
///
/// Returns a dict breaking GPU wall-time into decode / upload / compute
/// buckets, or `None` when the `gpu` feature is disabled. The profiler only
/// accumulates when the process was started with `SCX_GPU_PROFILE=1`; with the
/// env var unset every bucket reads zero and `enabled` is `False`.
///
/// Bucket keys (each a sub-dict with `ms`, `count`, and — for uploads —
/// `bytes`): `host_decode_scx1`, `host_decode_generic`, `htod_scx1`,
/// `htod_generic`, `gpu_decode`, `compute`. The `host_decode_*` + `htod_*`
/// buckets quantify how much of the in-VRAM gap vs rapids-singlecell is
/// decode/upload (the Phase 4 format-work signal) rather than compute.
///
/// Example:
///     pyscx.accel.gpu_profile_reset()
///     pyscx.accel.pca(adata, device="gpu")   # run with SCX_GPU_PROFILE=1
///     prof = pyscx.accel.gpu_profile_snapshot()
#[pyfunction]
pub fn gpu_profile_snapshot(py: Python<'_>) -> PyResult<Py<PyAny>> {
    #[cfg(feature = "gpu")]
    {
        let snap = scx_accel::profile::snapshot();
        let stage = |s: scx_accel::StageStat| -> PyResult<Py<PyAny>> {
            let d = PyDict::new(py);
            d.set_item("ms", s.ns as f64 / 1.0e6)?;
            d.set_item("count", s.count)?;
            d.set_item("bytes", s.bytes)?;
            Ok(d.into_any().unbind())
        };
        let dict = PyDict::new(py);
        dict.set_item("enabled", snap.enabled)?;
        dict.set_item("host_decode_scx1", stage(snap.host_decode_scx1)?)?;
        dict.set_item("host_decode_generic", stage(snap.host_decode_generic)?)?;
        dict.set_item("htod_scx1", stage(snap.htod_scx1)?)?;
        dict.set_item("htod_generic", stage(snap.htod_generic)?)?;
        dict.set_item("gpu_decode", stage(snap.gpu_decode)?)?;
        dict.set_item("compute", stage(snap.compute)?)?;
        Ok(dict.into_any().unbind())
    }
    #[cfg(not(feature = "gpu"))]
    {
        Ok(py.None())
    }
}

/// Reset the GPU per-stage timing profiler counters to zero.
///
/// Call between benchmark runs so each snapshot reflects only the most recent
/// operation. No-op when the `gpu` feature is disabled.
#[pyfunction]
pub fn gpu_profile_reset() -> PyResult<()> {
    #[cfg(feature = "gpu")]
    {
        scx_accel::profile::reset();
    }
    Ok(())
}

/// rapids-singlecell + core-dep (cuML, cuPy) availability and versions.
///
/// rapids is a **detected runtime dependency**: SCX probes for it at dispatch
/// and routes GPU analysis to it when present, else falls back to CPU.
/// `available` is `true` only when all three import cleanly.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)] // fields consumed by the op routers
pub(crate) struct RapidsInfo {
    pub available: bool,
    pub rapids_version: Option<String>,
    pub cuml_version: Option<String>,
    pub cupy_version: Option<String>,
}

#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
fn module_version(py: Python<'_>, module: &str) -> Option<String> {
    crate::pyimport::import_module(py, module)
        .ok()
        .and_then(|m| m.getattr("__version__").ok())
        .and_then(|v| v.extract::<String>().ok())
}

/// Probe rapids-singlecell once and cache the result process-wide. The import
/// (and the CUDA init it triggers) is paid once; a host without the stack
/// records `available = false` with `None` versions. Tolerant of
/// `ImportError`/CUDA-init failures (they surface as `Err`, mapped to `None`).
///
/// Consumed by [`super::rapids::decide`], which injects the probe result
/// into the shared `scx_accel::route::plan_rapids_decision`.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn rapids_singlecell_info(py: Python<'_>) -> RapidsInfo {
    static INFO: std::sync::OnceLock<RapidsInfo> = std::sync::OnceLock::new();
    INFO.get_or_init(|| {
        let rapids_version = module_version(py, "rapids_singlecell");
        let cuml_version = module_version(py, "cuml");
        let cupy_version = module_version(py, "cupy");
        RapidsInfo {
            available: rapids_version.is_some() && cuml_version.is_some() && cupy_version.is_some(),
            rapids_version,
            cuml_version,
            cupy_version,
        }
    })
    .clone()
}

/// Whether cuPy is importable. cuPy (not rapids) is the hard requirement for the
/// `to_gpu_anndata` device handoff — the returned `X` is a
/// `cupyx.scipy.sparse.csr_matrix`. Returns the version string when present.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn cupy_info(py: Python<'_>) -> Option<String> {
    module_version(py, "cupy")
}

// Device-string resolution is shared with rscx (ORG-10.16-3): the grammar,
// availability checks, and error text live in `scx_accel::device`.
pub(crate) use scx_accel::device::ResolvedDevice;

/// Resolve the device string to a [`ResolvedDevice`], mapping the shared
/// [`DeviceError`](scx_accel::device::DeviceError) taxonomy onto the Python
/// exception types the error messages were written for: grammar/vocabulary
/// errors → `ValueError`; well-formed-but-unsatisfiable requests (no CUDA
/// device, out-of-range ordinal) and the dedicated feature-disabled variant
/// (non-gpu build) → `RuntimeError`.
pub(crate) fn resolve_device(device: &str) -> PyResult<ResolvedDevice> {
    scx_accel::device::resolve_device(device).map_err(|e| match e {
        scx_accel::device::DeviceError::Invalid(msg) => PyValueError::new_err(msg),
        scx_accel::device::DeviceError::Unavailable(msg) => PyRuntimeError::new_err(msg),
        // The shared Display is binding-neutral; this binding names itself,
        // keeping the historical pyscx text byte-identical.
        scx_accel::device::DeviceError::GpuFeatureDisabled(dev) => PyRuntimeError::new_err(
            format!("device='{dev}' requested but pyscx was built without the 'gpu' feature"),
        ),
    })
}
