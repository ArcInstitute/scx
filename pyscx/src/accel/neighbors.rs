//! kNN graph construction — HNSW (CPU) + cuVS CAGRA (GPU).

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::gpu::resolve_device;

/// Build a kNN graph using approximate nearest neighbors.
///
/// Reads `adata.obsm["X_pca"]` and computes a kNN graph plus UMAP-style
/// connectivities. Results are written to `adata.obsp["distances"]`,
/// `adata.obsp["connectivities"]`, and `adata.uns["neighbors"]`,
/// matching scanpy's `sc.pp.neighbors()` output format.
///
/// When `device="gpu"`, uses cuVS CAGRA (GPU graph-based ANN) for 20-50×
/// speedup on large datasets. Falls back to CPU HNSW if cuVS is unavailable.
///
/// Args:
///     adata: AnnData object with obsm["X_pca"] (n_obs × n_pcs)
///     n_neighbors: Number of nearest neighbors (default: 15)
///     use_rep: Key in adata.obsm to use (default: "X_pca")
///     random_state: Random seed for reproducibility (default: 0)
///     ef_construction: HNSW construction parameter (default: 200, CPU only)
///     ef_search: HNSW search parameter (default: 200, CPU only)
///     device: Device selection — "auto" (default), "cpu", "gpu", or
///         "gpu:N" to target CUDA device N on multi-GPU systems.
///
/// Note: GPU mode uses cuVS CAGRA (graph-based ANN) instead of HNSW. Both are
/// approximate; neighbor sets may differ slightly. See docs/scanpy.md.
#[pyfunction]
#[pyo3(signature = (adata, n_neighbors=15, use_rep="X_pca", random_state=0, ef_construction=200, ef_search=200, device="auto"))]
#[allow(clippy::too_many_arguments)]
pub fn neighbors(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    n_neighbors: usize,
    use_rep: &str,
    random_state: u64,
    ef_construction: usize,
    ef_search: usize,
    device: &str,
) -> PyResult<()> {
    let numpy = py.import("numpy")?;

    // Determine effective device
    let _device = resolve_device(device)?;
    #[cfg(feature = "gpu")]
    let _gpu_id = _device.gpu_id();

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

    // GPU path
    #[cfg(feature = "gpu")]
    if let Some(device_id) = _gpu_id {
        // Check if cuVS CAGRA is available
        if scx_accel::cuvs_available() {
            let result =
                scx_accel::build_knn_graph_gpu(device_id, &data, n_obs, n_vars, n_neighbors)
                    .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?;

            write_neighbors_to_adata(py, adata, &result, n_neighbors, use_rep, "cagra")?;
            return Ok(());
        }
        // cuVS not available — fall through to CPU with warning
        let warnings = py.import("warnings")?;
        warnings.call_method1(
            "warn",
            ("cuVS library not found — falling back to CPU HNSW. \
              Install cuVS for GPU-accelerated kNN: \
              conda install -c rapidsai -c conda-forge libcuvs",),
        )?;
    }

    // Suppress unused variable warning when gpu feature is not enabled
    let _ = _device;

    // CPU path (default or fallback)
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
    write_neighbors_to_adata(py, adata, &result, n_neighbors, use_rep, "hnsw")?;

    Ok(())
}

/// Write kNN results to AnnData slots matching scanpy's format.
fn write_neighbors_to_adata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    result: &scx_accel::KnnResult,
    n_neighbors: usize,
    use_rep: &str,
    method: &str,
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
    params_dict.set_item("method", method)?;
    params_dict.set_item("use_rep", use_rep)?;
    neighbors_dict.set_item("params", params_dict)?;

    let uns = adata.getattr("uns")?;
    uns.set_item("neighbors", neighbors_dict)?;

    Ok(())
}
