//! Python bindings for SCX accelerators (PCA, kNN, etc.).
//!
//! Exposes `pyscx.accel.pca(adata, ...)` which runs randomized PCA
//! streaming from SCX's backed mode and writes results to standard
//! AnnData slots (obsm["X_pca"], varm["PCs"], uns["pca"]).

use numpy::PyArray2;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::ScxBackedSparseDataset;

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
#[pyfunction]
#[pyo3(signature = (adata, n_comps=50, zero_center=true, random_state=0, n_oversamples=10, n_power_iterations=2))]
pub fn pca(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    n_comps: usize,
    zero_center: bool,
    random_state: u64,
    n_oversamples: usize,
    n_power_iterations: usize,
) -> PyResult<()> {
    // Extract X from adata
    let x = adata.getattr("X")?;

    // Try to extract as ScxBackedSparseDataset for streaming PCA
    let result = if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        // Streaming PCA from backed mode
        let reader = &backed.backed;
        scx_accel::randomized_pca(
            reader,
            n_comps,
            n_oversamples,
            n_power_iterations,
            zero_center,
            random_state,
        )
        .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
    } else {
        // Materialized: extract scipy CSR → ScxCsr → in-memory PCA
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
            scx_accel::randomized_pca_inmemory(
                &csr,
                n_comps,
                n_oversamples,
                n_power_iterations,
                zero_center,
                random_state,
            )
            .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
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
    };

    // Write results to AnnData slots
    write_pca_to_adata(py, adata, &result)?;

    Ok(())
}

/// Write PCA results to AnnData slots matching scanpy's format.
fn write_pca_to_adata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    result: &scx_accel::PcaResult,
) -> PyResult<()> {
    let numpy = py.import("numpy")?;

    // adata.obsm["X_pca"] = embeddings (n_obs × n_components)
    let embeddings_arr = PyArray2::<f64>::from_vec2(
        py,
        &(0..result.n_obs)
            .map(|i| {
                (0..result.n_components)
                    .map(|j| result.embeddings[i * result.n_components + j])
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>(),
    )?;
    // Convert to float32 for consistency with scanpy
    let embeddings_f32 = embeddings_arr.call_method1("astype", ("float32",))?;
    let obsm = adata.getattr("obsm")?;
    obsm.set_item("X_pca", embeddings_f32)?;

    // adata.varm["PCs"] = components.T (n_vars × n_components)
    let components_arr = PyArray2::<f64>::from_vec2(
        py,
        &(0..result.n_components)
            .map(|pc| {
                (0..result.n_vars)
                    .map(|v| result.components[pc * result.n_vars + v])
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>(),
    )?;
    // Transpose: scanpy stores PCs as (n_vars × n_components)
    let pcs = components_arr
        .getattr("T")?
        .call_method1("astype", ("float32",))?
        .call_method0("copy")?;
    let varm = adata.getattr("varm")?;
    varm.set_item("PCs", pcs)?;

    // adata.uns["pca"] = dict with variance info
    let pca_dict = PyDict::new(py);

    let var_explained = numpy.call_method1("array", (result.variance_explained.clone(),))?;
    pca_dict.set_item("variance", var_explained)?;

    let var_ratio = numpy.call_method1("array", (result.variance_ratio.clone(),))?;
    pca_dict.set_item("variance_ratio", var_ratio)?;

    let uns = adata.getattr("uns")?;
    uns.set_item("pca", pca_dict)?;

    Ok(())
}
