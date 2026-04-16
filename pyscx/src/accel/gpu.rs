//! GPU device info, memory estimation, and device resolution.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Query GPU device information.
///
/// Returns a dict with keys `device` (name string), `total_vram_gb` (f64),
/// and `free_vram_gb` (f64). Returns `None` if no GPU is available or the
/// `gpu` feature is disabled.
///
/// Example:
///     info = pyscx.accel.gpu_info()
///     if info is not None:
///         print(f"GPU: {info['device']}, {info['free_vram_gb']:.1f} GB free")
#[pyfunction]
pub fn gpu_info(py: Python<'_>) -> PyResult<PyObject> {
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
) -> PyResult<PyObject> {
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

            // Y matrix on GPU: n_obs × k × 4 (f32)
            let y_bytes = n_obs * k * 4;
            // Ω and B matrices: n_vars × k × 4 each
            let omega_b_bytes = 2 * n_vars * k * 4;
            // One decoded shard for cuSPARSE SpMM: shard_rows × n_vars × 4
            let shard_dense_bytes = shard_rows * n_vars * 4;
            // cuSOLVER QR workspace: ~2 × n_obs × k × 4
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

/// Resolve the device string to a boolean (true = GPU, false = CPU).
///
/// "auto" → GPU if available (feature enabled + device found), else CPU.
/// "cpu" → always CPU.
/// "gpu" / "gpu:N" → always GPU (errors if unavailable).
pub(super) fn resolve_device(device: &str) -> PyResult<bool> {
    match device {
        "cpu" => Ok(false),
        "auto" => {
            #[cfg(feature = "gpu")]
            {
                Ok(scx_accel::gpu_available())
            }
            #[cfg(not(feature = "gpu"))]
            {
                Ok(false)
            }
        }
        d if d.starts_with("gpu") => {
            #[cfg(feature = "gpu")]
            {
                if scx_accel::gpu_available() {
                    Ok(true)
                } else {
                    Err(PyRuntimeError::new_err(
                        "device='gpu' requested but no CUDA GPU found",
                    ))
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                Err(PyRuntimeError::new_err(
                    "device='gpu' requested but pyscx was built without the 'gpu' feature",
                ))
            }
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown device: '{}'. Use 'auto', 'cpu', or 'gpu'",
            device
        ))),
    }
}
