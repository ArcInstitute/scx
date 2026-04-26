//! UMAP embedding — CPU SGD + GPU CUDA kernel + cuML fallback.

use numpy::PyArray2;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::gpu::resolve_device;

/// Compute UMAP embedding from a kNN graph.
///
/// Reads `adata.obsp["connectivities"]` (from `pyscx.accel.neighbors()` or
/// `sc.pp.neighbors()`) and computes a 2D embedding via SGD optimization.
/// Results are written to `adata.obsm["X_umap"]`.
///
/// When `device="gpu"`, uses a native CUDA SGD kernel for 10–300× speedup
/// on large datasets. Falls back to cuML UMAP (if importable) or CPU.
///
/// Args:
///     adata: AnnData with obsp["connectivities"] (CSR, n_obs × n_obs)
///     n_components: Output dimensions (default: 2)
///     n_epochs: SGD epochs (default: 200)
///     min_dist: Minimum distance in embedding (default: 0.1)
///     spread: Spread of embedded points (default: 1.0)
///     negative_sample_rate: Negative samples per positive edge (default: 5)
///     learning_rate: Initial learning rate (default: 1.0)
///     random_state: Random seed (default: 0)
///     device: Device selection — "auto" (default), "cpu", "gpu", or
///         "gpu:N" to target CUDA device N on multi-GPU systems.
///
/// Note: GPU UMAP is non-deterministic due to intentional atomicAdd race
/// conditions on embedding updates (matches cuML). Embeddings will differ
/// from CPU UMAP but preserve equivalent cluster structure.
#[pyfunction]
#[pyo3(signature = (adata, n_components=2, n_epochs=200, min_dist=0.1, spread=1.0, negative_sample_rate=5, learning_rate=1.0, random_state=0, device="auto"))]
#[allow(clippy::too_many_arguments)]
pub fn umap(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    n_components: usize,
    n_epochs: usize,
    min_dist: f64,
    spread: f64,
    negative_sample_rate: usize,
    learning_rate: f64,
    random_state: u64,
    device: &str,
) -> PyResult<()> {
    let numpy = py.import("numpy")?;

    // Determine effective device
    let _device = resolve_device(device)?;
    #[cfg(feature = "gpu")]
    let _gpu_id = _device.gpu_id();

    // Extract connectivities CSR from adata.obsp["connectivities"]
    let obsp = adata.getattr("obsp")?;
    let conn = obsp.get_item("connectivities").map_err(|_| {
        PyRuntimeError::new_err(
            "'connectivities' not found in adata.obsp. Run neighbors first: \
             pyscx.accel.neighbors(adata) or sc.pp.neighbors(adata)",
        )
    })?;

    let shape: (usize, usize) = conn.getattr("shape")?.extract()?;
    let n_obs = shape.0;

    // Extract CSR components
    let indptr: Vec<i64> = numpy
        .call_method1("asarray", (conn.getattr("indptr")?,))?
        .call_method1("astype", ("int64",))?
        .extract::<Vec<i64>>()?;
    let indices: Vec<i32> = numpy
        .call_method1("asarray", (conn.getattr("indices")?,))?
        .call_method1("astype", ("int32",))?
        .extract::<Vec<i32>>()?;
    let data: Vec<f64> = numpy
        .call_method1("asarray", (conn.getattr("data")?,))?
        .call_method1("astype", ("float64",))?
        .extract::<Vec<f64>>()?;

    // GPU path
    #[cfg(feature = "gpu")]
    if let Some(device_id) = _gpu_id {
        // Try native CUDA SGD kernel first
        match scx_accel::compute_umap_gpu(
            device_id,
            &indptr,
            &indices,
            &data,
            n_obs,
            n_components,
            n_epochs,
            min_dist,
            spread,
            negative_sample_rate,
            learning_rate,
            random_state,
        ) {
            Ok(result) => {
                write_umap_to_adata(py, adata, &result)?;
                write_umap_backend(py, adata, "scx-gpu-cuda")?;
                return Ok(());
            }
            Err(e) => {
                // Native CUDA failed — try cuML fallback
                let cuml_ok = try_cuml_umap(
                    py,
                    adata,
                    n_components,
                    n_epochs,
                    min_dist,
                    spread,
                    negative_sample_rate,
                    learning_rate,
                    random_state,
                );
                if cuml_ok.is_ok() {
                    return Ok(());
                }
                // Both GPU paths failed — fall through to CPU with warning
                let warnings = py.import("warnings")?;
                warnings.call_method1(
                    "warn",
                    (format!(
                        "GPU UMAP failed (native: {e}; cuML not available) — falling back to CPU"
                    ),),
                )?;
            }
        }
    }

    // Suppress unused variable warning when gpu feature is not enabled
    let _ = _device;

    // CPU path (default or fallback)
    let result = scx_accel::compute_umap(
        &indptr,
        &indices,
        &data,
        n_obs,
        n_components,
        n_epochs,
        min_dist,
        spread,
        negative_sample_rate,
        learning_rate,
        random_state,
        None,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Write results to adata.obsm["X_umap"]
    write_umap_to_adata(py, adata, &result)?;
    write_umap_backend(py, adata, "scx-accel-cpu")?;

    Ok(())
}

/// Write UMAP backend metadata to adata.uns["umap"].
fn write_umap_backend(py: Python<'_>, adata: &Bound<'_, PyAny>, backend: &str) -> PyResult<()> {
    let uns = adata.getattr("uns")?;
    let umap_dict = PyDict::new(py);
    umap_dict.set_item("backend", backend)?;
    uns.set_item("umap", umap_dict)?;
    Ok(())
}

/// Try cuML UMAP as a fallback when native CUDA kernel is unavailable.
///
/// Attempts to import `cuml.manifold.UMAP` at runtime. If cuML is importable,
/// runs UMAP via cuML's Python API and writes results to adata.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn try_cuml_umap(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    n_components: usize,
    n_epochs: usize,
    min_dist: f64,
    spread: f64,
    _negative_sample_rate: usize,
    learning_rate: f64,
    random_state: u64,
) -> PyResult<()> {
    // Try importing cuML
    let cuml_umap = py
        .import("cuml.manifold")
        .map_err(|_| PyRuntimeError::new_err("cuML not available"))?;

    let umap_cls = cuml_umap.getattr("UMAP")?;

    // Create UMAP instance with parameters matching our API
    let kwargs = PyDict::new(py);
    kwargs.set_item("n_components", n_components)?;
    kwargs.set_item("n_epochs", n_epochs)?;
    kwargs.set_item("min_dist", min_dist)?;
    kwargs.set_item("spread", spread)?;
    kwargs.set_item("learning_rate", learning_rate)?;
    kwargs.set_item("random_state", random_state as i64)?;

    // cuML UMAP expects the precomputed kNN graph via adata
    // Use the "precomputed" metric with the connectivities matrix
    kwargs.set_item("metric", "precomputed")?;
    let model = umap_cls.call((), Some(&kwargs))?;

    // Fit on the connectivities matrix
    let obsp = adata.getattr("obsp")?;
    let conn = obsp.get_item("connectivities")?;
    let embedding = model.call_method1("fit_transform", (&conn,))?;

    // Write to adata.obsm["X_umap"]
    let obsm = adata.getattr("obsm")?;
    obsm.set_item("X_umap", &embedding)?;

    // Record backend
    write_umap_backend(py, adata, "cuml")?;

    Ok(())
}

/// Write UMAP results to adata.obsm["X_umap"].
fn write_umap_to_adata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    result: &scx_accel::UmapResult,
) -> PyResult<()> {
    // Convert f64→f32 in Rust to avoid intermediate f64 numpy allocation
    let embeddings_arr = PyArray2::<f32>::from_vec2(
        py,
        &(0..result.n_obs)
            .map(|i| {
                (0..result.n_components)
                    .map(|j| result.embeddings[i * result.n_components + j] as f32)
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>(),
    )?;
    let obsm = adata.getattr("obsm")?;
    obsm.set_item("X_umap", embeddings_arr)?;

    Ok(())
}
