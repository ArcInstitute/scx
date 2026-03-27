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

/// Build a kNN graph using approximate nearest neighbors (HNSW).
///
/// Reads `adata.obsm["X_pca"]` and computes a kNN graph plus UMAP-style
/// connectivities. Results are written to `adata.obsp["distances"]`,
/// `adata.obsp["connectivities"]`, and `adata.uns["neighbors"]`,
/// matching scanpy's `sc.pp.neighbors()` output format.
///
/// Args:
///     adata: AnnData object with obsm["X_pca"] (n_obs × n_pcs)
///     n_neighbors: Number of nearest neighbors (default: 15)
///     use_rep: Key in adata.obsm to use (default: "X_pca")
///     random_state: Random seed for reproducibility (default: 0)
///     ef_construction: HNSW construction parameter (default: 200)
///     ef_search: HNSW search parameter (default: 200)
#[pyfunction]
#[pyo3(signature = (adata, n_neighbors=15, use_rep="X_pca", random_state=0, ef_construction=200, ef_search=200))]
pub fn neighbors(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    n_neighbors: usize,
    use_rep: &str,
    random_state: u64,
    ef_construction: usize,
    ef_search: usize,
) -> PyResult<()> {
    let numpy = py.import("numpy")?;

    // Extract representation matrix from adata.obsm[use_rep]
    let obsm = adata.getattr("obsm")?;
    let rep_data = obsm.get_item(use_rep).map_err(|_| {
        PyRuntimeError::new_err(format!(
            "'{use_rep}' not found in adata.obsm. Run PCA first: pyscx.accel.pca(adata)"
        ))
    })?;

    // Convert to float32 numpy array and get shape
    let arr = numpy
        .call_method1("asarray", (&rep_data,))?
        .call_method1("astype", ("float32",))?;
    let shape: (usize, usize) = arr.getattr("shape")?.extract()?;
    let (n_obs, n_vars) = shape;

    // Flatten to Vec<f32>
    let flat = arr.call_method0("ravel")?;
    let data: Vec<f32> = flat.extract()?;

    // Build kNN graph
    let result = scx_accel::build_knn_graph(
        &data,
        n_obs,
        n_vars,
        n_neighbors,
        ef_construction,
        ef_search,
        random_state,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Write results to AnnData
    write_neighbors_to_adata(py, adata, &result, n_neighbors, use_rep)?;

    Ok(())
}

/// Write kNN results to AnnData slots matching scanpy's format.
fn write_neighbors_to_adata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    result: &scx_accel::KnnResult,
    n_neighbors: usize,
    use_rep: &str,
) -> PyResult<()> {
    let scipy_sparse = py.import("scipy.sparse")?;
    let numpy = py.import("numpy")?;
    let n_obs = result.n_obs;

    // Build distance CSR matrix (n_obs × n_obs)
    let dist_indptr = numpy.call_method1("array", (result.dist_indptr.clone(),))?;
    let dist_indices = numpy.call_method1("array", (result.dist_indices.clone(),))?;
    let dist_data = numpy.call_method1("array", (result.dist_data.clone(),))?;
    let dist_shape = (n_obs, n_obs);
    let distances_csr = scipy_sparse.call_method1(
        "csr_matrix",
        ((&dist_data, &dist_indices, &dist_indptr), dist_shape),
    )?;

    // Build connectivities CSR matrix (n_obs × n_obs)
    let conn_indptr = numpy.call_method1("array", (result.conn_indptr.clone(),))?;
    let conn_indices = numpy.call_method1("array", (result.conn_indices.clone(),))?;
    let conn_data = numpy.call_method1("array", (result.conn_data.clone(),))?;
    let conn_shape = (n_obs, n_obs);
    let connectivities_csr = scipy_sparse.call_method1(
        "csr_matrix",
        ((&conn_data, &conn_indices, &conn_indptr), conn_shape),
    )?;

    // Write to adata.obsp
    let obsp = adata.getattr("obsp")?;
    obsp.set_item("distances", &distances_csr)?;
    obsp.set_item("connectivities", &connectivities_csr)?;

    // Write to adata.uns["neighbors"]
    let neighbors_dict = PyDict::new(py);
    neighbors_dict.set_item("connectivities_key", "connectivities")?;
    neighbors_dict.set_item("distances_key", "distances")?;

    let params_dict = PyDict::new(py);
    params_dict.set_item("n_neighbors", n_neighbors)?;
    params_dict.set_item("method", "hnsw")?;
    params_dict.set_item("use_rep", use_rep)?;
    neighbors_dict.set_item("params", params_dict)?;

    let uns = adata.getattr("uns")?;
    uns.set_item("neighbors", neighbors_dict)?;

    Ok(())
}

/// Compute UMAP embedding from a kNN graph.
///
/// Reads `adata.obsp["connectivities"]` (from `pyscx.accel.neighbors()` or
/// `sc.pp.neighbors()`) and computes a 2D embedding via SGD optimization.
/// Results are written to `adata.obsm["X_umap"]`.
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
#[pyfunction]
#[pyo3(signature = (adata, n_components=2, n_epochs=200, min_dist=0.1, spread=1.0, negative_sample_rate=5, learning_rate=1.0, random_state=0))]
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
) -> PyResult<()> {
    let numpy = py.import("numpy")?;

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

    // Compute UMAP
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

    Ok(())
}

/// Write UMAP results to adata.obsm["X_umap"].
fn write_umap_to_adata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    result: &scx_accel::UmapResult,
) -> PyResult<()> {
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
    obsm.set_item("X_umap", embeddings_f32)?;

    Ok(())
}
