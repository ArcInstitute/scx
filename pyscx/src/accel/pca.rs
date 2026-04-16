//! PCA bindings — randomized and covariance PCA, streaming + in-memory.

use scx_format::ShardSource;

use numpy::PyArray2;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::ScxLazyTransformedDataset;

use super::gpu::resolve_device;

/// Run randomized PCA on an AnnData whose X is backed by SCX.
///
/// Results are written to `adata.obsm["X_pca"]`, `adata.varm["PCs"]`,
/// and `adata.uns["pca"]`, matching scanpy's output format.
///
/// Args:
///     adata: AnnData object with X as ScxBackedSparseDataset (or materialized)
///     n_comps: Number of principal components (default: 50)
///     zero_center: Whether to mean-center data (default: True)
///     random_state: Random seed for reproducibility (default: 0)
///     n_oversamples: Extra dimensions for accuracy (default: 10)
///     n_power_iterations: Power iterations for spectral accuracy (default: 2)
///     device: Device selection — "auto" (default), "cpu", or "gpu"
///
/// Note: GPU mode uses f32 precision throughout (CPU uses f64 intermediates),
/// producing slightly different but equally valid results. See docs/scanpy.md.
#[pyfunction]
#[pyo3(signature = (adata, n_comps=50, zero_center=true, random_state=0, n_oversamples=10, n_power_iterations=2, device="auto"))]
#[allow(clippy::too_many_arguments)]
pub fn pca(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    n_comps: usize,
    zero_center: bool,
    random_state: u64,
    n_oversamples: usize,
    n_power_iterations: usize,
    device: &str,
) -> PyResult<()> {
    // Determine effective device
    let _use_gpu = resolve_device(device)?;
    let backend: &str;

    // Extract X from adata
    let x = adata.getattr("X")?;

    // Try GPU path first if requested
    #[cfg(feature = "gpu")]
    if _use_gpu {
        if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
            let reader = &backed.backed;
            let result = scx_accel::randomized_pca_gpu(
                0, // device_id = 0 (first GPU)
                reader,
                n_comps,
                n_oversamples,
                n_power_iterations,
                zero_center,
                random_state,
            )
            .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?;

            write_pca_to_adata(py, adata, &result, "scx-gpu-cusparse")?;
            return Ok(());
        }
    }

    // CPU path (default or fallback)
    // Auto-route: use covariance method when n_vars <= threshold (faster for HVG data)
    let cov_threshold = scx_accel::COVARIANCE_PCA_THRESHOLD;

    let result = if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        // Streaming PCA from backed mode
        backend = "scx-accel-cpu";
        let reader = &*backed.backed;
        let (_n_obs, n_vars) = reader.shape();
        if n_vars <= cov_threshold {
            scx_accel::covariance_pca(reader, n_comps, zero_center)
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
        } else {
            scx_accel::randomized_pca(
                reader,
                n_comps,
                n_oversamples,
                n_power_iterations,
                zero_center,
                random_state,
            )
            .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
        }
    } else if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
        backend = "scx-accel-cpu";
        // Streaming PCA through lazy transforms — no materialization
        let source = lazy.as_shard_source();
        let (_n_obs, n_vars) = source.shape();
        if n_vars <= cov_threshold {
            scx_accel::covariance_pca(&source, n_comps, zero_center)
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
        } else {
            scx_accel::randomized_pca(
                &source,
                n_comps,
                n_oversamples,
                n_power_iterations,
                zero_center,
                random_state,
            )
            .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
        }
    } else {
        // Materialized: extract scipy CSR → ScxCsr → in-memory PCA
        backend = "scx-accel-cpu";
        let scipy_sparse = py.import("scipy.sparse")?;
        let is_sparse = scipy_sparse
            .call_method1("issparse", (&x,))?
            .extract::<bool>()?;

        if is_sparse {
            let csr = scipy_sparse.call_method1("csr_matrix", (&x,))?;
            let shape: (usize, usize) = csr.getattr("shape")?.extract()?;
            let indptr_np = csr.getattr("indptr")?;
            let indices_np = csr.getattr("indices")?;
            let data_np = csr.getattr("data")?;

            // Convert to Vec
            let np = py.import("numpy")?;
            let indptr: Vec<i64> = np
                .call_method1("asarray", (&indptr_np,))?
                .call_method1("astype", ("int64",))?
                .extract::<Vec<i64>>()?;
            let indices: Vec<i32> = np
                .call_method1("asarray", (&indices_np,))?
                .call_method1("astype", ("int32",))?
                .extract::<Vec<i32>>()?;
            let data: Vec<f32> = np
                .call_method1("asarray", (&data_np,))?
                .call_method1("astype", ("float32",))?
                .extract::<Vec<f32>>()?;

            let csr = scx_sparse::ScxCsr::new_unchecked(shape, indptr, indices, data);
            if shape.1 <= cov_threshold {
                scx_accel::covariance_pca_inmemory(&csr, n_comps, zero_center)
                    .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
            } else {
                scx_accel::randomized_pca_inmemory(
                    &csr,
                    n_comps,
                    n_oversamples,
                    n_power_iterations,
                    zero_center,
                    random_state,
                )
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
            }
        } else {
            // Dense numpy array: convert to CSR first
            let csr = scipy_sparse.call_method1("csr_matrix", (&x,))?;
            let shape: (usize, usize) = csr.getattr("shape")?.extract()?;
            let indptr: Vec<i64> = csr
                .getattr("indptr")?
                .call_method1("astype", ("int64",))?
                .extract::<Vec<i64>>()?;
            let indices: Vec<i32> = csr
                .getattr("indices")?
                .call_method1("astype", ("int32",))?
                .extract::<Vec<i32>>()?;
            let data: Vec<f32> = csr
                .getattr("data")?
                .call_method1("astype", ("float32",))?
                .extract::<Vec<f32>>()?;

            let csr = scx_sparse::ScxCsr::new_unchecked(shape, indptr, indices, data);
            if shape.1 <= cov_threshold {
                scx_accel::covariance_pca_inmemory(&csr, n_comps, zero_center)
                    .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
            } else {
                scx_accel::randomized_pca_inmemory(
                    &csr,
                    n_comps,
                    n_oversamples,
                    n_power_iterations,
                    zero_center,
                    random_state,
                )
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
            }
        }
    };

    // Write results to AnnData slots
    write_pca_to_adata(py, adata, &result, backend)?;

    Ok(())
}

/// Write PCA results to AnnData slots matching scanpy's format.
fn write_pca_to_adata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    result: &scx_accel::PcaResult,
    backend: &str,
) -> PyResult<()> {
    let numpy = py.import("numpy")?;

    // adata.obsm["X_pca"] = embeddings (n_obs × n_components) as float32
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
    obsm.set_item("X_pca", embeddings_arr)?;

    // adata.varm["PCs"] = components transposed to (n_vars × n_components) as float32
    let pcs_arr = PyArray2::<f32>::from_vec2(
        py,
        &(0..result.n_vars)
            .map(|v| {
                (0..result.n_components)
                    .map(|pc| result.components[pc * result.n_vars + v] as f32)
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>(),
    )?;
    let varm = adata.getattr("varm")?;
    varm.set_item("PCs", pcs_arr)?;

    // adata.uns["pca"] = dict with variance info + backend
    let pca_dict = PyDict::new(py);

    let var_explained = numpy.call_method1("array", (result.variance_explained.clone(),))?;
    pca_dict.set_item("variance", var_explained)?;

    let var_ratio = numpy.call_method1("array", (result.variance_ratio.clone(),))?;
    pca_dict.set_item("variance_ratio", var_ratio)?;

    pca_dict.set_item("backend", backend)?;

    let uns = adata.getattr("uns")?;
    uns.set_item("pca", pca_dict)?;

    Ok(())
}
