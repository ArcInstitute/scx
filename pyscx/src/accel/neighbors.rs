//! kNN graph construction — HNSW (CPU) + cuVS CAGRA (GPU).

use numpy::PyArray1;
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
/// approximate; neighbor sets may differ slightly. See
/// docs/scanpy/accel-embedding-clustering.md.
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
    // Rebuild an AnnData view as actual before the first write below
    // (obsm / obs / uns), so a backed X is not gathered by anndata's
    // copy-on-write. No var-order guard: this op is gene-order agnostic.
    super::prepare_target_no_var_guard(py, adata, "neighbors")?;

    let numpy = crate::pyimport::import_module(py, "numpy")?;

    // Determine effective device
    let _device = resolve_device(device)?;

    // In-VRAM `device="gpu"` kNN hands off to rapids-singlecell
    // (`rsc.pp.neighbors`) when `X` is in memory. backed/lazy X stays on the
    // native cuVS streaming/device-resident path.
    #[cfg(feature = "gpu")]
    {
        use crate::backed::ScxBackedSparseDataset;
        use crate::lazy_transform::ScxLazyTransformedDataset;
        let xp = adata.getattr("X")?;
        let x_in_memory = xp.cast::<ScxBackedSparseDataset>().is_err()
            && xp.cast::<ScxLazyTransformedDataset>().is_err();
        if x_in_memory {
            match super::rapids::decide(py, _device, "neighbors") {
                super::rapids::RapidsDecision::Rapids(gid) => {
                    super::rapids::run(py, adata, "neighbors", gid, |py, adata| {
                        let kw = super::rapids::kwargs(py);
                        kw.set_item("n_neighbors", n_neighbors)?;
                        kw.set_item("use_rep", use_rep)?;
                        // ef_construction / ef_search are HNSW-specific knobs with
                        // no rapids (cuVS) analogue — intentionally not forwarded.
                        kw.set_item("random_state", random_state)?;
                        super::rapids::rsc_fn(py, "pp", "neighbors")?.call((adata,), Some(&kw))?;
                        Ok(())
                    })?;
                    return Ok(());
                }
                super::rapids::RapidsDecision::NoRapidsCpu => {
                    neighbors(
                        py,
                        adata,
                        n_neighbors,
                        use_rep,
                        random_state,
                        ef_construction,
                        ef_search,
                        "cpu",
                    )?;
                    return super::rapids::stamp_no_rapids(
                        py,
                        adata,
                        "neighbors",
                        scx_accel::AccelRoute::CpuCsr,
                    );
                }
                super::rapids::RapidsDecision::Native => {}
            }
        }
    }

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

    // Record the planned route on adata.uns["scx_accel"]["neighbors"].
    //
    // The in-VRAM native CAGRA dispatch was removed. In-VRAM `device="gpu"` kNN
    // routes to rapids-singlecell (intercepted above); the standalone entry now
    // runs CPU HNSW for backed/lazy `X` and under SCX_FORCE_NATIVE_GPU. The
    // device-resident CAGRA path survives only inside the fused `pca_neighbors`
    // pipeline (`scx_accel::pca_then_knn_gpu`), where the PCA embedding never
    // leaves the device. The no-rapids CPU fallback (route `cpu_*` +
    // `fallback_reason="no_rapids"`) is stamped in the rapids interception
    // above, so this stamp covers only the genuine CPU runs.
    let info = super::route::simple_exec_info(
        device,
        false,
        scx_accel::AccelRoute::GpuCsr,
        scx_accel::AccelRoute::CpuCsr,
    );
    super::route::announce_route(py, "neighbors", device, &info);
    // Rolled back if the kNN build below raises (e.g. n_neighbors > n_obs).
    let route = super::route::RouteStamp::write(py, adata, "neighbors", &info)?;

    // Suppress unused variable warning when gpu feature is not enabled
    let _ = _device;

    // CPU path (default or fallback) — release GIL for the computation
    let result = py
        .detach(|| {
            scx_accel::build_knn_graph(
                &data,
                n_obs,
                n_vars,
                n_neighbors,
                ef_construction,
                ef_search,
                random_state,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Write results to AnnData
    write_neighbors_to_adata(py, adata, result, n_neighbors, use_rep, "hnsw")?;

    route.commit();
    Ok(())
}

/// Write kNN results to AnnData slots matching scanpy's format.
///
/// Takes `result` **by value** so the six CSR buffers can be handed to numpy
/// with [`PyArray1::from_vec`], which moves them. The pre-4.3 spelling —
/// `numpy.array(vec.clone())` — cloned the Rust buffer, then pyo3 converted the
/// `Vec` into a Python `list` (one `PyLong`/`PyFloat` object per element), then
/// numpy re-parsed that list. At 1M cells × k=15 that is tens of millions of
/// transient Python objects per call, on a path shared by CPU kNN, GPU kNN,
/// `pca_neighbors` and `pca_neighbors_umap`.
///
/// Output is unchanged. The intermediate dtypes differ — `np.array(list[int])`
/// is int64 regardless of the Rust width, whereas `from_vec` preserves i64/i32 —
/// but `scipy.sparse.csr_matrix` re-derives the index dtype through
/// `get_index_dtype(..., check_contents=True)`, so the matrices it builds are
/// identical either way. `test_marshalling_dtypes.py` asserts that rather than
/// assuming it.
pub(crate) fn write_neighbors_to_adata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    result: scx_accel::KnnResult,
    n_neighbors: usize,
    use_rep: &str,
    method: &str,
) -> PyResult<()> {
    let t_marshal = scx_accel::cpu_profile::start();
    let scipy_sparse = crate::pyimport::import_module(py, "scipy.sparse")?;
    let n_obs = result.n_obs;
    let scx_accel::KnnResult {
        conn_indptr,
        conn_indices,
        conn_data,
        dist_indptr,
        dist_indices,
        dist_data,
        ..
    } = result;
    let marshalled_bytes = (conn_indptr.len() + dist_indptr.len()) * 8
        + (conn_indices.len() + dist_indices.len()) * 4
        + (conn_data.len() + dist_data.len()) * 8;

    // Build distance CSR matrix (n_obs × n_obs)
    let dist_indptr = PyArray1::from_vec(py, dist_indptr);
    let dist_indices = PyArray1::from_vec(py, dist_indices);
    let dist_data = PyArray1::from_vec(py, dist_data);
    let dist_shape = (n_obs, n_obs);
    let distances_csr = scipy_sparse.call_method1(
        "csr_matrix",
        ((&dist_data, &dist_indices, &dist_indptr), dist_shape),
    )?;

    // Build connectivities CSR matrix (n_obs × n_obs)
    let conn_indptr = PyArray1::from_vec(py, conn_indptr);
    let conn_indices = PyArray1::from_vec(py, conn_indices);
    let conn_data = PyArray1::from_vec(py, conn_data);
    let conn_shape = (n_obs, n_obs);
    let connectivities_csr = scipy_sparse.call_method1(
        "csr_matrix",
        ((&conn_data, &conn_indices, &conn_indptr), conn_shape),
    )?;
    scx_accel::cpu_profile::record_marshalling_since(t_marshal, marshalled_bytes);

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
