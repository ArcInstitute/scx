//! Python bindings for SCX accelerators (PCA, kNN, etc.).
//!
//! Exposes `pyscx.accel.pca(adata, ...)` which runs randomized PCA
//! streaming from SCX's backed mode and writes results to standard
//! AnnData slots (obsm["X_pca"], varm["PCs"], uns["pca"]).

use std::sync::Arc;

use numpy::PyArray2;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::{ScxBackedLayerDataset, ScxBackedSparseDataset};
use crate::lazy_transform::{ScxLazyTransformedDataset, Transform};
use crate::projected_agg;

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

/// A single stratum: the composite key values and a boolean mask over adata.obs.
struct Stratum {
    /// Key values for each stratify_by column.
    key: Vec<String>,
}

/// Extract and validate strata from adata.obs.
///
/// Returns (strata, boolean_masks_as_py_arrays, stratify_col_names).
/// Drops NaN rows with a logged warning. Filters by min_cells_per_stratum.
fn extract_strata<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    stratify_by: &[String],
    min_cells_per_stratum: usize,
    forbidden_cols: &[&str],
) -> PyResult<(Vec<Stratum>, Vec<Bound<'py, PyAny>>)> {
    let obs = adata.getattr("obs")?;
    let warnings = py.import("warnings")?;
    let pd = py.import("pandas")?;
    let np = py.import("numpy")?;

    // Validate each stratify_by column exists and doesn't collide.
    for col in stratify_by {
        if !obs
            .call_method1("__contains__", (col.as_str(),))?
            .extract::<bool>()?
        {
            return Err(PyValueError::new_err(format!(
                "stratify_by column '{}' not found in adata.obs",
                col
            )));
        }
        for forbidden in forbidden_cols {
            if col.as_str() == *forbidden {
                return Err(PyValueError::new_err(format!(
                    "stratify_by column '{}' collides with '{}'",
                    col, forbidden
                )));
            }
        }
    }

    // Extract columns as string arrays.
    let mut col_arrays: Vec<Vec<String>> = Vec::new();
    let n_obs: usize = adata.getattr("n_obs")?.extract()?;
    let mut nan_mask = vec![false; n_obs];

    for col_name in stratify_by {
        let col = obs.get_item(col_name.as_str())?;
        // Check for NaN: convert to str, NaN becomes "nan"
        let str_col = col.call_method1("astype", ("str",))?;
        let labels: Vec<String> = str_col.call_method0("tolist")?.extract()?;

        // Also check pandas isna
        let isna = pd.call_method1("isna", (&col,))?;
        let isna_list: Vec<bool> = isna.call_method0("tolist")?.extract()?;
        for (i, is_na) in isna_list.iter().enumerate() {
            if *is_na {
                nan_mask[i] = true;
            }
        }

        col_arrays.push(labels);
    }

    let nan_count = nan_mask.iter().filter(|&&x| x).count();
    if nan_count > 0 {
        let msg = format!(
            "Dropped {} cells with NaN in stratify_by column(s) {:?}",
            nan_count, stratify_by
        );
        warnings.call_method1("warn", (msg,))?;
    }

    // Build composite keys for each cell (excluding NaN rows).
    let mut key_to_indices: std::collections::BTreeMap<Vec<String>, Vec<usize>> =
        std::collections::BTreeMap::new();
    for i in 0..n_obs {
        if nan_mask[i] {
            continue;
        }
        let key: Vec<String> = col_arrays.iter().map(|c| c[i].clone()).collect();
        key_to_indices.entry(key).or_default().push(i);
    }

    // Filter by min_cells_per_stratum and build results.
    let mut strata = Vec::new();
    let mut masks = Vec::new();
    let mut skipped = 0usize;

    for (key, indices) in &key_to_indices {
        if indices.len() < min_cells_per_stratum {
            skipped += 1;
            continue;
        }
        strata.push(Stratum { key: key.clone() });

        // Build boolean mask. Use direct index setting (O(n_obs)) instead of
        // Vec::contains per cell (which would be O(n_obs × stratum_size)).
        let mut mask_vec = vec![false; n_obs];
        for &idx in indices {
            mask_vec[idx] = true;
        }
        let mask = np.call_method1("array", (mask_vec,))?;
        masks.push(mask);
    }

    if skipped > 0 {
        let msg = format!(
            "Skipped {} strata with fewer than {} cells",
            skipped, min_cells_per_stratum
        );
        warnings.call_method1("warn", (msg,))?;
    }

    if strata.is_empty() {
        return Err(PyValueError::new_err(format!(
            "all strata were filtered out (min_cells_per_stratum={}). \
             No strata had enough cells for DE analysis.",
            min_cells_per_stratum
        )));
    }

    Ok((strata, masks))
}

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
    let result = if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        // Streaming PCA from backed mode
        backend = "scx-accel-cpu";
        let reader = &*backed.backed;
        scx_accel::randomized_pca(
            reader,
            n_comps,
            n_oversamples,
            n_power_iterations,
            zero_center,
            random_state,
        )
        .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
    } else if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
        backend = "scx-accel-cpu";
        // Streaming PCA through lazy transforms — no materialization
        let source = lazy.as_shard_source();
        scx_accel::randomized_pca(
            &source,
            n_comps,
            n_oversamples,
            n_power_iterations,
            zero_center,
            random_state,
        )
        .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
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
    write_pca_to_adata(py, adata, &result, backend)?;

    Ok(())
}

/// Resolve the device string to a boolean (true = GPU, false = CPU).
///
/// "auto" → GPU if available (feature enabled + device found), else CPU.
/// "cpu" → always CPU.
/// "gpu" / "gpu:N" → always GPU (errors if unavailable).
fn resolve_device(device: &str) -> PyResult<bool> {
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
///     device: Device selection — "auto" (default), "cpu", or "gpu"
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
    let use_gpu = resolve_device(device)?;

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
    if use_gpu {
        // Check if cuVS CAGRA is available
        if scx_accel::cuvs_available() {
            let result = scx_accel::build_knn_graph_gpu(
                0, // device_id = 0 (first GPU)
                &data,
                n_obs,
                n_vars,
                n_neighbors,
            )
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
    #[cfg(not(feature = "gpu"))]
    let _ = use_gpu;

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
///     device: Device selection — "auto" (default), "cpu", or "gpu"
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
    let use_gpu = resolve_device(device)?;

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
    if use_gpu {
        // Try native CUDA SGD kernel first
        match scx_accel::compute_umap_gpu(
            0, // device_id = 0
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
    #[cfg(not(feature = "gpu"))]
    let _ = use_gpu;

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

/// Run Wilcoxon rank-sum differential expression analysis.
///
/// Compares each group against the rest (or a specific reference group)
/// using parallel Wilcoxon rank-sum tests. Results are written to
/// `adata.uns["rank_genes_groups"]` in the same format as scanpy's
/// `sc.tl.rank_genes_groups(method="wilcoxon")`.
///
/// Args:
///     adata: AnnData object with X and obs[groupby]
///     groupby: Column in adata.obs to group cells by
///     reference: Group name to compare against (default: "rest" = 1-vs-rest)
///     n_genes: Number of top genes to report per group (default: all genes)
///     method: Statistical method (currently only "wilcoxon")
///     gene_chunk_size: When set and X is backed by SCX, uses streaming
///         gene-chunked DE to avoid full matrix materialization. Specifies
///         the number of genes to process per chunk (default: None = full in-memory).
/// Run Wilcoxon rank-sum DE on a single adata (no stratification).
///
/// Returns the DiffExpResult from scx_accel.
fn run_rank_genes_groups_inner(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    groupby: &str,
    reference: &str,
    gene_chunk_size: Option<usize>,
) -> PyResult<(scx_accel::DiffExpResult, Vec<String>)> {
    let numpy = py.import("numpy")?;
    let scipy_sparse = py.import("scipy.sparse")?;

    // Extract group labels from adata.obs[groupby].
    let obs = adata.getattr("obs")?;
    let group_col = obs.get_item(groupby)?;
    let group_labels: Vec<String> = group_col
        .call_method1("astype", ("str",))?
        .call_method0("tolist")?
        .extract()?;

    // Determine unique group names (sorted, matching scanpy's default).
    let cat_attr = group_col.getattr("cat");
    let unique_groups: Vec<String> = if let Ok(cat) = cat_attr {
        cat.getattr("categories")?
            .call_method0("tolist")?
            .extract()?
    } else {
        let mut unique: Vec<String> = group_labels.to_vec();
        unique.sort();
        unique.dedup();
        unique
    };

    // Map labels → indices.
    let group_name_to_idx: std::collections::HashMap<&str, usize> = unique_groups
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();

    let groups: Vec<usize> = group_labels
        .iter()
        .map(|label| *group_name_to_idx.get(label.as_str()).unwrap_or(&0))
        .collect();

    // Resolve reference.
    let ref_idx: Option<usize> = if reference == "rest" {
        None
    } else {
        Some(*group_name_to_idx.get(reference).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "reference group '{reference}' not found in adata.obs['{groupby}']"
            ))
        })?)
    };

    // Get gene names.
    let var = adata.getattr("var")?;
    let var_names = var.getattr("index")?;
    let gene_names: Vec<String> = var_names.call_method0("tolist")?.extract()?;

    // Auto-detect whether data has been log-transformed (sc.pp.log1p sets
    // adata.uns["log1p"]). When true, logFC uses expm1 back-transform to
    // match scanpy's formula.
    let log_transformed = adata
        .getattr("uns")?
        .call_method1("get", ("log1p",))
        .map(|v| !v.is_none())
        .unwrap_or(false);

    // Check if X is a ScxBackedSparseDataset for streaming path.
    let x = adata.getattr("X")?;
    let result = if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        // Backed mode: stream shards with gene-chunked DE.
        let chunk_size = gene_chunk_size.unwrap_or(500);
        scx_accel::wilcoxon_rank_sum_streaming(
            &backed.backed,
            &gene_names,
            &groups,
            &unique_groups,
            ref_idx,
            chunk_size,
            log_transformed,
        )
        .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
    } else {
        let is_sparse = scipy_sparse
            .call_method1("issparse", (&x,))?
            .extract::<bool>()?;

        if is_sparse {
            // Sparse in-memory: extract CSR arrays and use gene-chunked path
            // to avoid O(n_obs × n_vars) dense materialization.
            let csr_obj = scipy_sparse.call_method1("csr_matrix", (&x,))?;
            let shape: (usize, usize) = csr_obj.getattr("shape")?.extract()?;
            let np = py.import("numpy")?;
            let indptr: Vec<i64> = np
                .call_method1("asarray", (csr_obj.getattr("indptr")?,))?
                .call_method1("astype", ("int64",))?
                .extract::<Vec<i64>>()?;
            let indices: Vec<i32> = np
                .call_method1("asarray", (csr_obj.getattr("indices")?,))?
                .call_method1("astype", ("int32",))?
                .extract::<Vec<i32>>()?;
            let data: Vec<f32> = np
                .call_method1("asarray", (csr_obj.getattr("data")?,))?
                .call_method1("astype", ("float32",))?
                .extract::<Vec<f32>>()?;

            let csr = scx_sparse::ScxCsr::new_unchecked(shape, indptr, indices, data);
            scx_accel::wilcoxon_rank_sum_sparse(
                &csr,
                &gene_names,
                &groups,
                &unique_groups,
                ref_idx,
                gene_chunk_size.unwrap_or(500),
                log_transformed,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        } else {
            // Dense numpy array: flatten and use direct wilcoxon_rank_sum
            let dense = numpy
                .call_method1("asarray", (&x,))?
                .call_method1("astype", ("float32",))?;
            let shape: (usize, usize) = dense.getattr("shape")?.extract()?;
            let (n_obs, n_vars) = shape;

            let flat = dense.call_method0("ravel")?;
            let data: Vec<f32> = flat.extract()?;

            scx_accel::wilcoxon_rank_sum(
                &data,
                n_obs,
                n_vars,
                &gene_names,
                &groups,
                &unique_groups,
                ref_idx,
                log_transformed,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        }
    };

    Ok((result, unique_groups))
}

/// Convert a DiffExpResult into a pandas DataFrame.
///
/// Each group's genes become rows with columns: gene, scores, pvals, pvals_adj,
/// logfoldchanges, group.
fn de_result_to_dataframe<'py>(
    py: Python<'py>,
    result: &scx_accel::DiffExpResult,
    n_genes: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let pd = py.import("pandas")?;
    let mut all_frames: Vec<Bound<'py, PyAny>> = Vec::new();

    for (i, group_name) in result.group_names.iter().enumerate() {
        let full_n_genes = result.names[i].len();
        let n = n_genes.unwrap_or(full_n_genes).min(full_n_genes);

        let names_slice = &result.names[i][..n];
        let scores_slice = &result.scores[i][..n];
        let pvals_slice = &result.pvals[i][..n];
        let pvals_adj_slice = &result.pvals_adj[i][..n];
        let logfc_slice = &result.logfoldchanges[i][..n];

        let dict = PyDict::new(py);
        dict.set_item("gene", names_slice.to_vec())?;
        dict.set_item("scores", scores_slice.to_vec())?;
        dict.set_item("pvals", pvals_slice.to_vec())?;
        dict.set_item("pvals_adj", pvals_adj_slice.to_vec())?;
        dict.set_item("logfoldchanges", logfc_slice.to_vec())?;
        dict.set_item("group", vec![group_name.clone(); n])?;

        let df = pd.call_method1("DataFrame", (dict,))?;
        all_frames.push(df);
    }

    if all_frames.is_empty() {
        // Return empty DataFrame with the right columns.
        let dict = PyDict::new(py);
        for col in [
            "gene",
            "scores",
            "pvals",
            "pvals_adj",
            "logfoldchanges",
            "group",
        ] {
            dict.set_item(col, pyo3::types::PyList::empty(py))?;
        }
        return pd.call_method1("DataFrame", (dict,));
    }

    let frame_list = pyo3::types::PyList::new(py, &all_frames)?;
    let combined = pd.call_method(
        "concat",
        (frame_list,),
        Some(&{
            let kw = PyDict::new(py);
            kw.set_item("ignore_index", true)?;
            kw
        }),
    )?;
    Ok(combined)
}

#[pyfunction]
#[pyo3(signature = (adata, groupby, reference="rest", n_genes=None, method="wilcoxon", gene_chunk_size=None, stratify_by=None, min_cells_per_stratum=50))]
#[allow(clippy::too_many_arguments)]
pub fn rank_genes_groups(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    groupby: &str,
    reference: &str,
    n_genes: Option<usize>,
    method: &str,
    gene_chunk_size: Option<usize>,
    stratify_by: Option<Vec<String>>,
    min_cells_per_stratum: usize,
) -> PyResult<PyObject> {
    if method != "wilcoxon" {
        return Err(PyRuntimeError::new_err(format!(
            "unsupported method '{method}': only 'wilcoxon' is currently supported"
        )));
    }

    // --- Stratified path ---
    if let Some(ref strat_cols) = stratify_by {
        let forbidden = vec![groupby];
        let (strata, masks) =
            extract_strata(py, adata, strat_cols, min_cells_per_stratum, &forbidden)?;

        let pd = py.import("pandas")?;
        let warnings = py.import("warnings")?;
        let mut all_frames: Vec<Bound<'_, PyAny>> = Vec::new();

        for (stratum, mask) in strata.iter().zip(masks.iter()) {
            // Subset adata by mask.
            let sub_adata = adata.get_item(mask)?;
            let sub_adata = sub_adata.call_method0("copy")?;

            // Run DE on the subset.
            match run_rank_genes_groups_inner(py, &sub_adata, groupby, reference, gene_chunk_size) {
                Ok((result, _unique)) => {
                    let df = de_result_to_dataframe(py, &result, n_genes)?;
                    // Add stratum columns.
                    for (j, col_name) in strat_cols.iter().enumerate() {
                        df.set_item(col_name.as_str(), stratum.key[j].as_str())?;
                    }
                    all_frames.push(df);
                }
                Err(e) => {
                    let key_str = stratum.key.join(", ");
                    let msg = format!("DE failed for stratum [{}]: {}", key_str, e);
                    warnings.call_method1("warn", (msg,))?;
                }
            }
        }

        if all_frames.is_empty() {
            return Err(PyValueError::new_err(
                "all strata failed during stratified DE analysis",
            ));
        }

        let frame_list = pyo3::types::PyList::new(py, &all_frames)?;
        let combined = pd.call_method(
            "concat",
            (frame_list,),
            Some(&{
                let kw = PyDict::new(py);
                kw.set_item("ignore_index", true)?;
                kw
            }),
        )?;

        return Ok(combined.unbind());
    }

    // --- Non-stratified path (original behavior) ---
    let (result, _unique_groups) =
        run_rank_genes_groups_inner(py, adata, groupby, reference, gene_chunk_size)?;

    // Write results to adata.uns["rank_genes_groups"] in scanpy format.
    write_de_to_adata(py, adata, &result, groupby, reference, n_genes)?;

    Ok(py.None())
}

/// Write DE results to adata.uns["rank_genes_groups"] matching scanpy's format.
///
/// Scanpy stores results as numpy structured arrays (rec.arrays) with one
/// field per group. Each field contains gene names/scores/p-values sorted
/// by the test statistic.
fn write_de_to_adata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    result: &scx_accel::DiffExpResult,
    groupby: &str,
    reference: &str,
    n_genes: Option<usize>,
) -> PyResult<()> {
    let numpy = py.import("numpy")?;
    let n_groups = result.group_names.len();
    let full_n_genes = if n_groups > 0 {
        result.names[0].len()
    } else {
        0
    };
    let n_genes = n_genes.unwrap_or(full_n_genes).min(full_n_genes);

    let rgg = PyDict::new(py);

    // params dict
    let params = PyDict::new(py);
    params.set_item("groupby", groupby)?;
    params.set_item("reference", reference)?;
    params.set_item("method", "wilcoxon")?;
    params.set_item("use_raw", false)?;

    // Helper to build structured array (like scanpy's recarray format).
    // Scanpy stores e.g. names as a structured array with dtype like:
    //   [('group_A', 'O'), ('group_B', 'O')]
    // Each row is one gene rank position.
    let build_structured =
        |field_data: &[Vec<String>], groups: &[String]| -> PyResult<Bound<'_, PyAny>> {
            // Build dtype: list of (group_name, 'U200') tuples.
            let dt_list = pyo3::types::PyList::empty(py);
            for gn in groups {
                let tup = pyo3::types::PyTuple::new(py, [gn.as_str(), "U200"])?;
                dt_list.append(tup)?;
            }
            let dtype = numpy.call_method1("dtype", (dt_list,))?;

            // Build empty structured array, then fill fields.
            let arr = numpy.call_method1("empty", (n_genes,))?;
            let arr = arr.call_method1("astype", (&dtype,))?;
            for (i, gn) in groups.iter().enumerate() {
                let vals = &field_data[i];
                let col = pyo3::types::PyList::new(py, &vals[..n_genes])?;
                arr.set_item(gn.as_str(), col)?;
            }
            Ok(arr.unbind().into_bound(py))
        };

    let build_structured_f64 =
        |field_data: &[Vec<f64>], groups: &[String]| -> PyResult<Bound<'_, PyAny>> {
            let dt_list = pyo3::types::PyList::empty(py);
            for gn in groups {
                let tup = pyo3::types::PyTuple::new(py, [gn.as_str(), "f8"])?;
                dt_list.append(tup)?;
            }
            let dtype = numpy.call_method1("dtype", (dt_list,))?;

            let arr = numpy.call_method1("empty", (n_genes,))?;
            let arr = arr.call_method1("astype", (&dtype,))?;
            for (i, gn) in groups.iter().enumerate() {
                let vals: Vec<f64> = field_data[i][..n_genes].to_vec();
                let np_vals = numpy.call_method1("array", (vals,))?;
                arr.set_item(gn.as_str(), np_vals)?;
            }
            Ok(arr.unbind().into_bound(py))
        };

    let names = build_structured(&result.names, &result.group_names)?;
    let scores = build_structured_f64(&result.scores, &result.group_names)?;
    let pvals = build_structured_f64(&result.pvals, &result.group_names)?;
    let pvals_adj = build_structured_f64(&result.pvals_adj, &result.group_names)?;
    let logfoldchanges = build_structured_f64(&result.logfoldchanges, &result.group_names)?;

    rgg.set_item("params", params)?;
    rgg.set_item("names", names)?;
    rgg.set_item("scores", scores)?;
    rgg.set_item("pvals", pvals)?;
    rgg.set_item("pvals_adj", pvals_adj)?;
    rgg.set_item("logfoldchanges", logfoldchanges)?;

    let uns = adata.getattr("uns")?;
    uns.set_item("rank_genes_groups", rgg)?;

    Ok(())
}

/// Pseudobulk differential expression via Rust aggregation + pydeseq2.
///
/// Aggregates single-cell counts into pseudobulk samples by grouping cells
/// according to metadata columns (e.g., `["perturbation", "donor"]`), then
/// uses `pydeseq2` for negative binomial GLM testing.
///
/// Args:
///     adata: AnnData object with X and obs columns for groupby
///     groupby: List of obs column names to group by (e.g., ["perturbation", "donor"])
///     test_col: Column in groupby that contains the condition to test
///     reference: Reference level in test_col (e.g., "control")
///     design: DESeq2 design formula (default: auto-generated as "~ test_col")
///     aggr_method: "sum" (default) or "mean"
///     min_cells_per_group: Skip groups with fewer cells (default: 10)
///
/// Returns:
///     pandas DataFrame with columns: gene, baseMean, log2FoldChange,
///     lfcSE, stat, pvalue, padj, target, reference
#[pyfunction]
#[pyo3(signature = (adata, groupby, test_col, reference, design=None, aggr_method="sum", min_cells_per_group=10, stratify_by=None, min_cells_per_stratum=50))]
#[allow(clippy::too_many_arguments)]
pub fn pseudobulk_dex(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    groupby: Vec<String>,
    test_col: &str,
    reference: &str,
    design: Option<&str>,
    aggr_method: &str,
    min_cells_per_group: usize,
    stratify_by: Option<Vec<String>>,
    min_cells_per_stratum: usize,
) -> PyResult<PyObject> {
    // --- Stratified path ---
    if let Some(ref strat_cols) = stratify_by {
        // Forbidden columns: test_col and all groupby columns.
        let mut forbidden: Vec<&str> = groupby.iter().map(|s| s.as_str()).collect();
        forbidden.push(test_col);
        let (strata, masks) =
            extract_strata(py, adata, strat_cols, min_cells_per_stratum, &forbidden)?;

        let pd = py.import("pandas")?;
        let warnings = py.import("warnings")?;
        let mut all_frames: Vec<Bound<'_, PyAny>> = Vec::new();

        for (stratum, mask) in strata.iter().zip(masks.iter()) {
            // Subset adata by mask.
            let sub_adata = adata.get_item(mask)?;
            let sub_adata = sub_adata.call_method0("copy")?;

            // Run pseudobulk_dex on the subset (recursive call without stratify).
            match pseudobulk_dex(
                py,
                &sub_adata,
                groupby.clone(),
                test_col,
                reference,
                design,
                aggr_method,
                min_cells_per_group,
                None, // no nested stratification
                50,   // unused since stratify_by=None
            ) {
                Ok(result_obj) => {
                    let result_df = result_obj.bind(py);
                    // Add stratum columns.
                    for (j, col_name) in strat_cols.iter().enumerate() {
                        result_df.set_item(col_name.as_str(), stratum.key[j].as_str())?;
                    }
                    all_frames.push(result_df.clone());
                }
                Err(e) => {
                    let key_str = stratum.key.join(", ");
                    let msg = format!("Pseudobulk DE failed for stratum [{}]: {}", key_str, e);
                    warnings.call_method1("warn", (msg,))?;
                }
            }
        }

        if all_frames.is_empty() {
            return Err(PyValueError::new_err(
                "all strata failed during stratified pseudobulk DE analysis",
            ));
        }

        let frame_list = pyo3::types::PyList::new(py, &all_frames)?;
        let combined = pd.call_method(
            "concat",
            (frame_list,),
            Some(&{
                let kw = PyDict::new(py);
                kw.set_item("ignore_index", true)?;
                kw
            }),
        )?;

        return Ok(combined.unbind());
    }

    // --- Non-stratified path (original behavior) ---
    // Validate test_col is in groupby.
    if !groupby.contains(&test_col.to_string()) {
        return Err(PyRuntimeError::new_err(format!(
            "test_col '{}' must be one of the groupby columns: {:?}",
            test_col, groupby
        )));
    }

    let method = match aggr_method {
        "sum" => scx_accel::AggregationMethod::Sum,
        "mean" => scx_accel::AggregationMethod::Mean,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "unsupported aggr_method '{}': use 'sum' or 'mean'",
                aggr_method
            )))
        }
    };

    // Extract groupby columns from adata.obs.
    let obs = adata.getattr("obs")?;
    let mut obs_groups: Vec<Vec<String>> = Vec::with_capacity(groupby.len());
    for col_name in &groupby {
        let col = obs.get_item(col_name.as_str())?;
        let labels: Vec<String> = col
            .call_method1("astype", ("str",))?
            .call_method0("tolist")?
            .extract()?;
        obs_groups.push(labels);
    }

    // Get gene names.
    let var = adata.getattr("var")?;
    let var_names = var.getattr("index")?;
    let gene_names: Vec<String> = var_names.call_method0("tolist")?.extract()?;

    // Perform aggregation: backed or in-memory.
    let x = adata.getattr("X")?;
    let result = if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        scx_accel::pseudobulk_aggregate(
            &backed.backed,
            &obs_groups,
            &groupby,
            &gene_names,
            method,
            min_cells_per_group,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    } else {
        // In-memory: extract scipy CSR → ScxCsr.
        let scipy_sparse = py.import("scipy.sparse")?;
        let is_sparse = scipy_sparse
            .call_method1("issparse", (&x,))?
            .extract::<bool>()?;

        let csr_obj = if is_sparse {
            scipy_sparse.call_method1("csr_matrix", (&x,))?
        } else if x.hasattr("toarray")? {
            // Backed dataset with toarray — materialize.
            let arr = x.call_method0("toarray")?;
            scipy_sparse.call_method1("csr_matrix", (&arr,))?
        } else {
            let np = py.import("numpy")?;
            let arr = np
                .call_method1("asarray", (&x,))?
                .call_method1("astype", ("float32",))?;
            scipy_sparse.call_method1("csr_matrix", (&arr,))?
        };

        let shape: (usize, usize) = csr_obj.getattr("shape")?.extract()?;
        let np = py.import("numpy")?;
        let indptr: Vec<i64> = np
            .call_method1("asarray", (csr_obj.getattr("indptr")?,))?
            .call_method1("astype", ("int64",))?
            .extract::<Vec<i64>>()?;
        let indices: Vec<i32> = np
            .call_method1("asarray", (csr_obj.getattr("indices")?,))?
            .call_method1("astype", ("int32",))?
            .extract::<Vec<i32>>()?;
        let data: Vec<f32> = np
            .call_method1("asarray", (csr_obj.getattr("data")?,))?
            .call_method1("astype", ("float32",))?
            .extract::<Vec<f32>>()?;

        let csr = scx_sparse::ScxCsr::new_unchecked(shape, indptr, indices, data);

        scx_accel::pseudobulk_aggregate_inmemory(
            &csr,
            &obs_groups,
            &groupby,
            &gene_names,
            method,
            min_cells_per_group,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    };

    if result.n_groups == 0 {
        return Err(PyRuntimeError::new_err(
            "no groups passed the min_cells_per_group filter",
        ));
    }

    // Build counts DataFrame and metadata DataFrame for pydeseq2.
    let pd = py.import("pandas")?;
    let np = py.import("numpy")?;

    // counts_df: rows = pseudobulk samples, columns = genes
    let counts_array = np.call_method1("array", (result.counts.clone(),))?;
    let counts_2d = counts_array.call_method1("reshape", ((result.n_groups, result.n_vars),))?;

    // Sample indices (row labels for pydeseq2 counts matrix).
    let sample_names: Vec<String> = (0..result.n_groups)
        .map(|i| format!("sample_{}", i))
        .collect();
    let sample_index = pd.call_method1("Index", (sample_names.clone(),))?;
    let gene_index = pd.call_method1("Index", (result.gene_names.clone(),))?;

    let counts_df = pd.call_method(
        "DataFrame",
        (counts_2d,),
        Some(&{
            let kw = PyDict::new(py);
            kw.set_item("index", &sample_index)?;
            kw.set_item("columns", &gene_index)?;
            kw
        }),
    )?;

    // metadata_df: rows = samples, columns = groupby columns + n_cells
    let meta_dict = PyDict::new(py);
    for (col_idx, col_name) in result.groupby_columns.iter().enumerate() {
        let vals: Vec<String> = result
            .group_labels
            .iter()
            .map(|l| l[col_idx].clone())
            .collect();
        meta_dict.set_item(col_name.as_str(), vals)?;
    }
    meta_dict.set_item("n_cells", result.cell_counts.clone())?;

    let metadata_df = pd.call_method(
        "DataFrame",
        (meta_dict,),
        Some(&{
            let kw = PyDict::new(py);
            kw.set_item("index", &sample_index)?;
            kw
        }),
    )?;

    // Import pydeseq2 at runtime.
    let pydeseq2 = py.import("pydeseq2.dds").map_err(|_| {
        PyRuntimeError::new_err(
            "pydeseq2 is required for pseudobulk DE but is not installed.\n\
             Install with: pip install pydeseq2\n\
             Or: uv pip install pydeseq2",
        )
    })?;
    let pydeseq2_stats = py.import("pydeseq2.ds").map_err(|_| {
        PyRuntimeError::new_err(
            "pydeseq2.ds module not found. Ensure pydeseq2 is properly installed.\n\
             Install with: pip install pydeseq2",
        )
    })?;

    // Design formula.
    let _design_str = design
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("~ {}", test_col));

    // Determine contrasts: all levels of test_col vs reference.
    let test_col_idx = groupby.iter().position(|c| c == test_col).unwrap();
    let mut test_levels: Vec<String> = result
        .group_labels
        .iter()
        .map(|l| l[test_col_idx].clone())
        .collect();
    test_levels.sort();
    test_levels.dedup();

    let target_levels: Vec<&String> = test_levels
        .iter()
        .filter(|l| l.as_str() != reference)
        .collect();

    if target_levels.is_empty() {
        return Err(PyRuntimeError::new_err(format!(
            "no target levels found: all groups have test_col='{}'. \
             Check that reference='{}' is correct.",
            reference, reference
        )));
    }

    // Run DESeq2 per contrast and collect results.
    let mut all_results: Vec<Bound<'_, PyAny>> = Vec::new();

    // Create DeseqDataSet.
    let dds = pydeseq2.call_method(
        "DeseqDataSet",
        (),
        Some(&{
            let kw = PyDict::new(py);
            kw.set_item("counts", &counts_df)?;
            kw.set_item("metadata", &metadata_df)?;
            kw.set_item("design", &_design_str)?;
            kw
        }),
    )?;

    // Run DESeq2 pipeline.
    dds.call_method0("deseq2")?;

    for target in &target_levels {
        // Create DeseqStats for this contrast.
        let stat = pydeseq2_stats.call_method(
            "DeseqStats",
            (&dds,),
            Some(&{
                let kw = PyDict::new(py);
                let contrast =
                    pyo3::types::PyList::new(py, [test_col, target.as_str(), reference])?;
                kw.set_item("contrast", contrast)?;
                kw
            }),
        )?;

        stat.call_method0("summary")?;

        // Extract results DataFrame.
        let results_df = stat.getattr("results_df")?;
        let results_df = results_df.call_method0("copy")?;

        // Add target and reference columns.
        results_df.set_item("target", target.as_str())?;
        results_df.set_item("reference", reference)?;

        // Move gene from index to column.
        let reset = results_df.call_method(
            "reset_index",
            (),
            Some(&{
                let kw = PyDict::new(py);
                kw.set_item("names", pyo3::types::PyList::new(py, ["gene"])?)?;
                kw
            }),
        )?;

        all_results.push(reset);
    }

    // Concatenate all contrast results.
    let result_list = pyo3::types::PyList::new(py, &all_results)?;
    let combined = pd.call_method(
        "concat",
        (result_list,),
        Some(&{
            let kw = PyDict::new(py);
            kw.set_item("ignore_index", true)?;
            kw
        }),
    )?;

    Ok(combined.unbind())
}

/// Run Leiden community detection on a kNN graph.
///
/// Reads `adata.obsp["connectivities"]` (from `pyscx.accel.neighbors()` or
/// `sc.pp.neighbors()`) and partitions the graph using the Leiden algorithm.
/// Results are written to `adata.obs[key_added]` (cluster labels as strings)
/// and `adata.uns["leiden"]` (parameters and backend metadata).
///
/// When `device="gpu"`, tries cuGraph Leiden (GPU-accelerated, up to 47×
/// faster than igraph on million-cell datasets). Falls back to CPU leidenalg
/// via igraph if cuGraph is unavailable.
///
/// Args:
///     adata: AnnData with obsp["connectivities"] (CSR, n_obs × n_obs)
///     resolution: Resolution parameter controlling cluster granularity (default: 1.0)
///     key_added: Column name in adata.obs for cluster labels (default: "leiden")
///     random_state: Random seed for reproducibility (default: 0)
///     n_iterations: Maximum optimization iterations; -1 for until convergence (default: -1)
///     device: Device selection — "auto" (default), "cpu", or "gpu"
///
/// Notes:
///     GPU and CPU Leiden may produce different partitions on the same graph
///     due to algorithmic differences (cuGraph uses a different refinement
///     strategy than leidenalg). Both produce valid, high-quality community
///     structures. Compare results via ARI or NMI when switching backends.
#[pyfunction]
#[pyo3(signature = (adata, resolution=1.0, key_added="leiden", random_state=0, n_iterations=-1, device="auto"))]
#[allow(clippy::too_many_arguments)]
pub fn leiden(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    resolution: f64,
    key_added: &str,
    random_state: u64,
    n_iterations: i64,
    device: &str,
) -> PyResult<()> {
    // Determine effective device
    let use_gpu = resolve_device(device)?;

    // Extract connectivities CSR from adata.obsp["connectivities"]
    let obsp = adata.getattr("obsp")?;
    let conn = obsp.get_item("connectivities").map_err(|_| {
        PyRuntimeError::new_err(
            "'connectivities' not found in adata.obsp. Run neighbors first: \
             pyscx.accel.neighbors(adata) or sc.pp.neighbors(adata)",
        )
    })?;

    // GPU path: try cuGraph Leiden
    if use_gpu {
        match try_cugraph_leiden(
            py,
            adata,
            &conn,
            resolution,
            key_added,
            random_state,
            n_iterations,
        ) {
            Ok(()) => return Ok(()),
            Err(e) => {
                // cuGraph not available or failed — fall through to CPU
                let warnings = py.import("warnings")?;
                warnings.call_method1(
                    "warn",
                    (format!(
                        "GPU Leiden failed ({e}) — falling back to CPU leidenalg. \
                         Install cuGraph for GPU acceleration: \
                         conda install -c rapidsai -c conda-forge cugraph"
                    ),),
                )?;
            }
        }
    }

    // CPU path: leidenalg via igraph
    run_cpu_leiden(
        py,
        adata,
        &conn,
        resolution,
        key_added,
        random_state,
        n_iterations,
    )
}

/// Try GPU Leiden via cuGraph Python import.
///
/// Converts the connectivities CSR matrix to a cuGraph Graph, runs
/// `cugraph.leiden()`, and writes results to adata.
fn try_cugraph_leiden(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    conn: &Bound<'_, PyAny>,
    resolution: f64,
    key_added: &str,
    random_state: u64,
    max_iter: i64,
) -> PyResult<()> {
    // Import cuGraph — if not installed, return error immediately
    let cugraph = py
        .import("cugraph")
        .map_err(|_| PyRuntimeError::new_err("cugraph not available"))?;

    let numpy = py.import("numpy")?;
    let pd = py.import("pandas")?;

    // Extract COO from connectivities CSR: cuGraph works with edge lists
    let scipy_sparse = py.import("scipy.sparse")?;
    let coo = scipy_sparse
        .call_method1("triu", (&conn,))?
        .call_method0("tocoo")?;

    let rows = numpy
        .call_method1("asarray", (coo.getattr("row")?,))?
        .call_method1("astype", ("int32",))?;
    let cols = numpy
        .call_method1("asarray", (coo.getattr("col")?,))?
        .call_method1("astype", ("int32",))?;
    let weights = numpy
        .call_method1("asarray", (coo.getattr("data")?,))?
        .call_method1("astype", ("float32",))?;

    // Build edge list DataFrame for cuGraph
    let edge_dict = PyDict::new(py);
    edge_dict.set_item("src", &rows)?;
    edge_dict.set_item("dst", &cols)?;
    edge_dict.set_item("weight", &weights)?;
    let edge_df = pd.call_method1("DataFrame", (edge_dict,))?;

    // Try cudf for GPU acceleration, fall back to pandas
    let cudf_available = py.import("cudf").is_ok();
    let edge_df = if cudf_available {
        let cudf = py.import("cudf")?;
        cudf.call_method1("DataFrame", (&edge_df,))?
    } else {
        edge_df
    };

    // Create cuGraph Graph
    let graph = cugraph.call_method0("Graph")?;
    let from_cudf_kwargs = PyDict::new(py);
    from_cudf_kwargs.set_item("source", "src")?;
    from_cudf_kwargs.set_item("destination", "dst")?;
    from_cudf_kwargs.set_item("edge_attr", "weight")?;
    from_cudf_kwargs.set_item("renumber", true)?;
    graph.call_method("from_cudf_edgelist", (&edge_df,), Some(&from_cudf_kwargs))?;

    // Run Leiden
    let leiden_kwargs = PyDict::new(py);
    leiden_kwargs.set_item("resolution", resolution)?;
    leiden_kwargs.set_item("random_state", random_state as i32)?;
    if max_iter > 0 {
        leiden_kwargs.set_item("max_iter", max_iter)?;
    }

    let leiden_result = cugraph.call_method("leiden", (&graph,), Some(&leiden_kwargs))?;

    // leiden returns (partition_df, modularity) tuple
    let parts_df = leiden_result.get_item(0)?;
    let modularity: f64 = leiden_result.get_item(1)?.extract()?;

    // Sort by vertex ID to align with adata.obs order
    let parts_sorted = parts_df.call_method1("sort_values", ("vertex",))?;
    let cluster_col = parts_sorted.get_item("partition")?;

    // Convert to pandas if needed (cudf → pandas)
    // Note: .values is a property (not a method) on both pandas and cudf Series
    let cluster_labels = if cudf_available {
        cluster_col.call_method0("to_pandas")?.getattr("values")?
    } else {
        cluster_col.getattr("values")?
    };

    // Convert to string labels (matching scanpy convention)
    let labels_str = cluster_labels.call_method1("astype", ("str",))?;

    // Write to adata.obs[key_added] as a Categorical
    let obs = adata.getattr("obs")?;
    let cat_labels = pd.call_method1("Categorical", (&labels_str,))?;
    obs.set_item(key_added, cat_labels)?;

    // Write metadata to adata.uns["leiden"]
    let leiden_dict = PyDict::new(py);
    let params_dict = PyDict::new(py);
    params_dict.set_item("resolution", resolution)?;
    params_dict.set_item("random_state", random_state)?;
    params_dict.set_item("n_iterations", max_iter)?;
    leiden_dict.set_item("params", params_dict)?;
    leiden_dict.set_item("backend", "cugraph")?;
    leiden_dict.set_item("modularity", modularity)?;

    let uns = adata.getattr("uns")?;
    uns.set_item(key_added, leiden_dict)?;

    Ok(())
}

/// Run CPU Leiden clustering via leidenalg + igraph.
///
/// This mirrors scanpy's `sc.tl.leiden()` implementation: converts the
/// connectivities CSR matrix to an igraph Graph and runs leidenalg's
/// `find_partition()` with `RBConfigurationVertexPartition`.
fn run_cpu_leiden(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    conn: &Bound<'_, PyAny>,
    resolution: f64,
    key_added: &str,
    random_state: u64,
    n_iterations: i64,
) -> PyResult<()> {
    // Import leidenalg + igraph
    let leidenalg = py.import("leidenalg").map_err(|_| {
        PyRuntimeError::new_err(
            "leidenalg is required for CPU Leiden but is not installed.\n\
             Install with: pip install leidenalg\n\
             Or: conda install -c conda-forge leidenalg",
        )
    })?;

    let igraph = py.import("igraph").map_err(|_| {
        PyRuntimeError::new_err(
            "igraph is required for CPU Leiden but is not installed.\n\
             Install with: pip install igraph\n\
             Or: conda install -c conda-forge python-igraph",
        )
    })?;

    let numpy = py.import("numpy")?;
    let pd = py.import("pandas")?;

    // Convert connectivities CSR to COO for igraph edge list
    let scipy_sparse = py.import("scipy.sparse")?;

    // Upper-triangular to avoid double-counting edges (undirected graph)
    let upper = scipy_sparse.call_method1("triu", (&conn,))?;
    let coo = upper.call_method0("tocoo")?;

    let shape: (usize, usize) = conn.getattr("shape")?.extract()?;
    let n_obs = shape.0;

    let rows = numpy
        .call_method1("asarray", (coo.getattr("row")?,))?
        .call_method0("tolist")?;
    let cols = numpy
        .call_method1("asarray", (coo.getattr("col")?,))?
        .call_method0("tolist")?;
    let weights = numpy
        .call_method1("asarray", (coo.getattr("data")?,))?
        .call_method0("tolist")?;

    // Build igraph Graph
    let graph = igraph.call_method1("Graph", (n_obs,))?;

    // Build edge list as tuples
    let rows_vec: Vec<i64> = rows.extract()?;
    let cols_vec: Vec<i64> = cols.extract()?;
    let edges: Vec<(i64, i64)> = rows_vec.into_iter().zip(cols_vec).collect();
    let edge_list = pyo3::types::PyList::new(py, &edges)?;

    graph.call_method1("add_edges", (&edge_list,))?;

    // Set edge weights
    let weights_list: Vec<f64> = weights.extract()?;
    let py_weights = pyo3::types::PyList::new(py, &weights_list)?;
    let es = graph.getattr("es")?;
    es.set_item("weight", py_weights)?;

    // Set random seed for reproducibility
    // leidenalg uses a seed parameter in find_partition
    let partition_type = leidenalg.getattr("RBConfigurationVertexPartition")?;

    let kwargs = PyDict::new(py);
    kwargs.set_item("resolution_parameter", resolution)?;
    kwargs.set_item("weights", "weight")?;
    kwargs.set_item("seed", random_state)?;
    if n_iterations > 0 {
        kwargs.set_item("n_iterations", n_iterations)?;
    }

    let partition =
        leidenalg.call_method("find_partition", (&graph, &partition_type), Some(&kwargs))?;

    // Extract cluster assignments
    let membership = partition.getattr("membership")?;
    let membership_arr = numpy.call_method1("array", (&membership,))?;
    let labels_str = membership_arr.call_method1("astype", ("str",))?;

    // Write to adata.obs[key_added] as a Categorical
    let obs = adata.getattr("obs")?;
    let cat_labels = pd.call_method1("Categorical", (&labels_str,))?;
    obs.set_item(key_added, cat_labels)?;

    // Compute modularity for metadata
    let modularity: f64 = partition.call_method0("quality")?.extract()?;

    // Write metadata to adata.uns[key_added]
    let leiden_dict = PyDict::new(py);
    let params_dict = PyDict::new(py);
    params_dict.set_item("resolution", resolution)?;
    params_dict.set_item("random_state", random_state)?;
    params_dict.set_item("n_iterations", n_iterations)?;
    leiden_dict.set_item("params", params_dict)?;
    leiden_dict.set_item("backend", "leidenalg")?;
    leiden_dict.set_item("modularity", modularity)?;

    let uns = adata.getattr("uns")?;
    uns.set_item(key_added, leiden_dict)?;

    Ok(())
}

/// Normalize total counts per cell without materialization.
///
/// Replaces `adata.X` with a lazy wrapper that applies row normalization
/// during `__getitem__`. The row sums are precomputed via streaming and
/// cached in the wrapper.
///
/// Three cases:
/// 1. X is `ScxBackedSparseDataset` → create new `ScxLazyTransformedDataset`
/// 2. X is `ScxLazyTransformedDataset` → append NormalizeTotal transform
/// 3. X is scipy sparse/dense → delegate to `sc.pp.normalize_total()`
///
/// Args:
///     adata: AnnData object
///     target_sum: Target total counts per cell (default: 1e4)
#[pyfunction]
#[pyo3(signature = (adata, target_sum=10000.0))]
pub fn normalize_total(py: Python<'_>, adata: &Bound<'_, PyAny>, target_sum: f64) -> PyResult<()> {
    let x = adata.getattr("X")?;

    // Case 1: X is ScxBackedSparseDataset — create new lazy wrapper
    if let Ok(backed) = x.downcast::<ScxBackedSparseDataset>() {
        let backed_ref = backed.borrow();

        // Compute row sums via streaming over ALL physical rows.
        // Must be in physical-row space because transforms are applied
        // per-shard before deletion vector filtering.
        let all_row_sums = backed_ref
            .backed
            .row_sums()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let lazy = ScxLazyTransformedDataset::new(
            Arc::clone(&backed_ref.backed),
            backed_ref.shape_val,
            backed_ref.kept_to_global.clone(),
            // Inherit col_projection: transforms operate on full columns,
            // projection is applied per-shard after transforms
            backed_ref.col_projection().map(|c| c.to_vec()),
            vec![Transform::NormalizeTotal {
                row_sums: Arc::new(all_row_sums),
                target_sum,
            }],
        );
        // Drop the borrow before setattr to avoid RefCell borrow conflict
        drop(backed_ref);
        adata.setattr("X", Bound::new(py, lazy)?)?;
        return Ok(());
    }

    // Case 2: X is already ScxLazyTransformedDataset — append transform
    if let Ok(lazy) = x.downcast::<ScxLazyTransformedDataset>() {
        let mut lazy_ref = lazy.borrow_mut();
        // Compute row sums through existing transforms (streaming).
        // streaming_row_sums() returns a global-length vector (n_obs_global),
        // which is what apply_transforms_to_csr expects (indexes by global row).
        // Do NOT filter through deletion vector — that would produce a
        // kept-length vector causing index-out-of-bounds on datasets with
        // active deletions.
        let sums = lazy_ref.streaming_row_sums()?;
        lazy_ref.transforms.push(Transform::NormalizeTotal {
            row_sums: Arc::new(sums),
            target_sum,
        });
        return Ok(());
    }

    // Case 3: X is a regular scipy sparse or dense — delegate to scanpy
    let sc = py.import("scanpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("target_sum", target_sum)?;
    sc.getattr("pp")?
        .call_method("normalize_total", (adata,), Some(&kwargs))?;
    Ok(())
}

/// Apply log1p (ln(x + 1)) element-wise without materialization.
///
/// Replaces `adata.X` with a lazy wrapper that applies log1p during
/// `__getitem__`. When chained after `normalize_total`, the fused
/// optimization in `ScxLazyTransformedDataset` computes
/// `ln(x * target_sum / row_sum + 1)` in a single pass.
///
/// Three cases:
/// 1. X is `ScxBackedSparseDataset` → create new `ScxLazyTransformedDataset`
/// 2. X is `ScxLazyTransformedDataset` → append Log1p transform
/// 3. X is scipy sparse/dense → delegate to `sc.pp.log1p()`
///
/// Args:
///     adata: AnnData object
#[pyfunction]
#[pyo3(signature = (adata,))]
pub fn log1p(py: Python<'_>, adata: &Bound<'_, PyAny>) -> PyResult<()> {
    let x = adata.getattr("X")?;

    // Case 1: X is ScxBackedSparseDataset — create new lazy wrapper with Log1p
    if let Ok(backed) = x.downcast::<ScxBackedSparseDataset>() {
        let backed_ref = backed.borrow();

        let lazy = ScxLazyTransformedDataset::new(
            Arc::clone(&backed_ref.backed),
            backed_ref.shape_val,
            backed_ref.kept_to_global.clone(),
            // col_projection is not inherited for log1p:
            // log1p applies element-wise over the full column set
            None,
            vec![Transform::Log1p],
        );
        // Drop the borrow before setattr to avoid RefCell borrow conflict
        drop(backed_ref);
        adata.setattr("X", Bound::new(py, lazy)?)?;
        return Ok(());
    }

    // Case 2: X is already ScxLazyTransformedDataset — append Log1p transform
    if let Ok(lazy) = x.downcast::<ScxLazyTransformedDataset>() {
        let mut lazy_ref = lazy.borrow_mut();
        lazy_ref.transforms.push(Transform::Log1p);
        return Ok(());
    }

    // Case 3: X is a regular scipy sparse or dense — delegate to scanpy
    let sc = py.import("scanpy")?;
    sc.getattr("pp")?.call_method1("log1p", (adata,))?;
    Ok(())
}

/// Calculate QC metrics natively using streaming aggregation.
///
/// Replacement for `sc.pp.calculate_qc_metrics()` that works on backed and
/// lazy-transformed SCX data without materialization.  Falls back to scanpy
/// for regular scipy/dense matrices.
///
/// Computes per-cell (obs) and per-gene (var) metrics and writes them to
/// `adata.obs` / `adata.var` columns, matching scanpy's naming convention.
///
/// Args:
///     adata: AnnData object
///     qc_vars: list of boolean column names in `adata.var` identifying gene
///              subsets (e.g. `["mt"]` for mitochondrial genes)
///     log1p: if True, also add log1p-transformed versions of count metrics
///     inplace: if True (default), write metrics to adata.obs/var;
///              if False, return (obs_df, var_df)
#[pyfunction]
#[pyo3(signature = (adata, qc_vars=None, log1p=true, inplace=true))]
pub fn calculate_qc_metrics<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    qc_vars: Option<Vec<String>>,
    log1p: bool,
    inplace: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let x = adata.getattr("X")?;
    let qc_vars = qc_vars.unwrap_or_default();

    // Detect backed or lazy-transformed SCX dataset
    let is_backed = x.downcast::<ScxBackedSparseDataset>().is_ok();
    let is_lazy = x.downcast::<ScxLazyTransformedDataset>().is_ok();

    if !is_backed && !is_lazy {
        // Delegate to scanpy for regular scipy/dense
        let sc = py.import("scanpy")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("inplace", inplace)?;
        kwargs.set_item("log1p", log1p)?;
        if !qc_vars.is_empty() {
            kwargs.set_item("qc_vars", &qc_vars)?;
        }
        return sc
            .getattr("pp")?
            .call_method("calculate_qc_metrics", (adata,), Some(&kwargs));
    }

    // --- Streaming path for SCX-backed / lazy data ---

    let np = py.import("numpy")?;
    let pd = py.import("pandas")?;

    // Compute per-cell total_counts and n_genes_by_counts
    let (total_counts, n_genes): (Vec<f64>, Vec<i64>) = if is_backed {
        let backed = x.extract::<PyRef<ScxBackedSparseDataset>>()?;
        let row_sums = backed
            .backed
            .row_sums()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let row_nnz = backed
            .backed
            .row_nnz()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        (
            backed.filter_row_results(&row_sums),
            backed.filter_row_results(&row_nnz),
        )
    } else {
        let lazy = x.extract::<PyRef<ScxLazyTransformedDataset>>()?;
        let row_sums = lazy.streaming_row_sums()?;
        let row_nnz = lazy
            .backed
            .row_nnz()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        (
            lazy.filter_row_results(&row_sums),
            lazy.filter_row_results(&row_nnz),
        )
    };

    // Compute per-gene total_counts and n_cells_by_counts
    let (gene_total_counts, n_cells): (Vec<f64>, Vec<i64>) = if is_backed {
        let backed = x.extract::<PyRef<ScxBackedSparseDataset>>()?;
        let col_sums = match &backed.kept_to_global {
            Some(kept) => backed
                .backed
                .col_sums_masked(kept)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            None => backed
                .backed
                .col_sums()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        };
        let col_nnz = match &backed.kept_to_global {
            Some(kept) => backed
                .backed
                .col_nnz_masked(kept)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .iter()
                .map(|&v| v as i64)
                .collect(),
            None => backed
                .backed
                .col_nnz()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        };
        (col_sums, col_nnz)
    } else {
        let lazy = x.extract::<PyRef<ScxLazyTransformedDataset>>()?;
        let col_sums = if lazy.kept_to_global.is_some() {
            lazy.streaming_col_sums_masked()?
        } else {
            lazy.streaming_col_sums()?
        };
        let col_nnz = if let Some(ref kept) = lazy.kept_to_global {
            lazy.backed
                .col_nnz_masked(kept)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .iter()
                .map(|&v| v as i64)
                .collect()
        } else {
            lazy.backed
                .col_nnz()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        };
        (col_sums, col_nnz)
    };

    // Build obs DataFrame
    let obs_dict = PyDict::new(py);
    obs_dict.set_item("n_genes_by_counts", numpy::PyArray::from_vec(py, n_genes))?;
    obs_dict.set_item(
        "total_counts",
        numpy::PyArray::from_vec(py, total_counts.clone()),
    )?;
    if log1p {
        let log_total: Vec<f64> = total_counts.iter().map(|&v| (v + 1.0).ln()).collect();
        obs_dict.set_item(
            "log1p_total_counts",
            numpy::PyArray::from_vec(py, log_total),
        )?;
    }

    // Build var DataFrame
    let var_dict = PyDict::new(py);
    var_dict.set_item("n_cells_by_counts", numpy::PyArray::from_vec(py, n_cells))?;
    var_dict.set_item(
        "total_counts",
        numpy::PyArray::from_vec(py, gene_total_counts.clone()),
    )?;
    if log1p {
        let log_gene_total: Vec<f64> = gene_total_counts.iter().map(|&v| (v + 1.0).ln()).collect();
        var_dict.set_item(
            "log1p_total_counts",
            numpy::PyArray::from_vec(py, log_gene_total),
        )?;
    }

    // Compute qc_var metrics (per-cell counts for gene subsets)
    for qc_var in &qc_vars {
        let var_df = adata.getattr("var")?;
        let mask_series = var_df.get_item(qc_var.as_str())?;
        let mask_values = mask_series.getattr("values")?;
        // Get column indices where mask is True
        let col_indices_py = np.call_method1("where", (&mask_values,))?;
        let col_indices_arr = col_indices_py.get_item(0)?;
        let col_indices: Vec<u32> = col_indices_arr
            .call_method1("astype", ("uint32",))?
            .extract()?;

        if col_indices.is_empty() {
            // No genes in this qc_var — fill with zeros
            let n_obs = total_counts.len();
            let zeros = vec![0.0f64; n_obs];
            obs_dict.set_item(
                format!("total_counts_{qc_var}"),
                numpy::PyArray::from_vec(py, zeros.clone()),
            )?;
            obs_dict.set_item(
                format!("pct_counts_{qc_var}"),
                numpy::PyArray::from_vec(py, zeros),
            )?;
            continue;
        }

        // Streaming projected row sums for the gene subset
        let subset_sums = if is_backed {
            let backed = x.extract::<PyRef<ScxBackedSparseDataset>>()?;
            let all_sums = projected_agg::row_sums_projected(&backed.backed, &col_indices)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            backed.filter_row_results(&all_sums)
        } else {
            let lazy = x.extract::<PyRef<ScxLazyTransformedDataset>>()?;
            // For lazy data, streaming through transforms with projection
            // is not yet supported. Fall back to the raw (pre-transform) sums
            // since QC metrics are typically computed before normalization.
            let all_sums = projected_agg::row_sums_projected(&lazy.backed, &col_indices)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            lazy.filter_row_results(&all_sums)
        };

        // pct_counts = subset_sum / total * 100
        let pct_counts: Vec<f64> = subset_sums
            .iter()
            .zip(total_counts.iter())
            .map(|(&s, &t)| if t > 0.0 { s / t * 100.0 } else { 0.0 })
            .collect();

        // Compute log1p before moving subset_sums
        let log1p_subset_sums: Option<Vec<f64>> = if log1p {
            Some(subset_sums.iter().map(|&v| (v + 1.0).ln()).collect())
        } else {
            None
        };

        obs_dict.set_item(
            format!("total_counts_{qc_var}"),
            numpy::PyArray::from_vec(py, subset_sums),
        )?;
        obs_dict.set_item(
            format!("pct_counts_{qc_var}"),
            numpy::PyArray::from_vec(py, pct_counts),
        )?;
        if let Some(log1p_sums) = log1p_subset_sums {
            obs_dict.set_item(
                format!("log1p_total_counts_{qc_var}"),
                numpy::PyArray::from_vec(py, log1p_sums),
            )?;
        }
    }

    // Create DataFrames
    let obs_index = adata.getattr("obs")?.getattr("index")?;
    let var_index = adata.getattr("var")?.getattr("index")?;
    let obs_df = pd.call_method1("DataFrame", (obs_dict,))?;
    obs_df.setattr("index", &obs_index)?;
    let var_df = pd.call_method1("DataFrame", (var_dict,))?;
    var_df.setattr("index", &var_index)?;

    if inplace {
        // Write columns to adata.obs / adata.var
        let adata_obs = adata.getattr("obs")?;
        let adata_var = adata.getattr("var")?;
        // Iterate obs_df columns
        let obs_columns: Vec<String> = obs_df
            .getattr("columns")?
            .call_method0("tolist")?
            .extract()?;
        for col in &obs_columns {
            let values = obs_df.get_item(col.as_str())?;
            adata_obs.set_item(col.as_str(), &values)?;
        }
        let var_columns: Vec<String> = var_df
            .getattr("columns")?
            .call_method0("tolist")?
            .extract()?;
        for col in &var_columns {
            let values = var_df.get_item(col.as_str())?;
            adata_var.set_item(col.as_str(), &values)?;
        }
        Ok(py.None().into_bound(py))
    } else {
        // Return (obs_df, var_df) tuple
        let tuple = pyo3::types::PyTuple::new(py, &[obs_df, var_df])?;
        Ok(tuple.into_any())
    }
}

// ---------------------------------------------------------------------------
// Cell / gene filtering without materialization
// ---------------------------------------------------------------------------

/// Helper: compute row NNZ for a backed dataset (respecting col_projection + deletions).
fn backed_row_nnz(backed: &ScxBackedSparseDataset) -> PyResult<Vec<i64>> {
    let all_nnz = if let Some(cols) = backed.col_projection() {
        projected_agg::row_nnz_projected(&backed.backed, cols)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    } else {
        backed
            .backed
            .row_nnz()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    };
    Ok(backed.filter_row_results(&all_nnz))
}

/// Helper: compute row sums for a backed dataset (respecting col_projection + deletions).
fn backed_row_sums(backed: &ScxBackedSparseDataset) -> PyResult<Vec<f64>> {
    let all_sums = if let Some(cols) = backed.col_projection() {
        projected_agg::row_sums_projected(&backed.backed, cols)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    } else {
        backed
            .backed
            .row_sums()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    };
    Ok(backed.filter_row_results(&all_sums))
}

/// Helper: compute col NNZ for a backed dataset (4-way dispatch).
fn backed_col_nnz(backed: &ScxBackedSparseDataset) -> PyResult<Vec<i64>> {
    let counts = match (backed.col_projection(), &backed.kept_to_global) {
        (Some(cols), Some(kept)) => {
            projected_agg::col_nnz_masked_projected(&backed.backed, kept, cols)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        }
        (Some(cols), None) => projected_agg::col_nnz_projected(&backed.backed, cols)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        (None, Some(kept)) => {
            let f_counts = backed
                .backed
                .col_nnz_masked(kept)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            f_counts.iter().map(|&v| v as i64).collect()
        }
        (None, None) => backed
            .backed
            .col_nnz()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    };
    Ok(counts)
}

/// Helper: compute col sums for a backed dataset (4-way dispatch).
fn backed_col_sums(backed: &ScxBackedSparseDataset) -> PyResult<Vec<f64>> {
    let sums = match (backed.col_projection(), &backed.kept_to_global) {
        (Some(cols), Some(kept)) => {
            projected_agg::col_sums_masked_projected(&backed.backed, kept, cols)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        }
        (Some(cols), None) => projected_agg::col_sums_projected(&backed.backed, cols)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        (None, Some(kept)) => backed
            .backed
            .col_sums_masked(kept)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        (None, None) => backed
            .backed
            .col_sums()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    };
    Ok(sums)
}

/// Build a boolean keep-mask from optional min/max thresholds on two metrics.
fn build_keep_mask(
    n: usize,
    nnz: Option<&[i64]>,
    sums: Option<&[f64]>,
    min_nnz: Option<i64>,
    max_nnz: Option<i64>,
    min_sum: Option<f64>,
    max_sum: Option<f64>,
) -> Vec<bool> {
    let mut keep = vec![true; n];
    if let Some(min_v) = min_nnz {
        if let Some(vals) = nnz {
            for (i, &v) in vals.iter().enumerate() {
                if v < min_v {
                    keep[i] = false;
                }
            }
        }
    }
    if let Some(max_v) = max_nnz {
        if let Some(vals) = nnz {
            for (i, &v) in vals.iter().enumerate() {
                if v > max_v {
                    keep[i] = false;
                }
            }
        }
    }
    if let Some(min_v) = min_sum {
        if let Some(vals) = sums {
            for (i, &v) in vals.iter().enumerate() {
                if v < min_v {
                    keep[i] = false;
                }
            }
        }
    }
    if let Some(max_v) = max_sum {
        if let Some(vals) = sums {
            for (i, &v) in vals.iter().enumerate() {
                if v > max_v {
                    keep[i] = false;
                }
            }
        }
    }
    keep
}

/// Compose a new deletion vector from a boolean mask and an existing kept_to_global.
fn compose_kept_to_global(keep: &[bool], existing: Option<&[u64]>) -> Vec<u64> {
    match existing {
        Some(existing) => keep
            .iter()
            .enumerate()
            .filter(|(_, &k)| k)
            .map(|(i, _)| existing[i])
            .collect(),
        None => keep
            .iter()
            .enumerate()
            .filter(|(_, &k)| k)
            .map(|(i, _)| i as u64)
            .collect(),
    }
}

/// Slice `adata.obs` and `adata.obsm` to match a boolean keep-mask.
fn slice_obs_and_obsm<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    keep: &[bool],
) -> PyResult<()> {
    let np = py.import("numpy")?;
    let mask_arr = numpy::PyArray::from_vec(py, keep.to_vec());

    // Slice obs — use _obs to bypass anndata's shape validation
    // (X.shape is already updated via set_kept_to_global before this call)
    let obs = adata.getattr("obs")?;
    let filtered_obs = obs.getattr("loc")?.get_item(&mask_arr)?;
    adata.setattr("_obs", filtered_obs)?;

    // Slice obsm entries
    let obsm = adata.getattr("obsm")?;
    // obsm may be empty or a dict-like; get keys safely
    let keys_result: PyResult<Vec<String>> = obsm
        .call_method0("keys")?
        .try_iter()?
        .map(|k| k.and_then(|k| k.extract()))
        .collect();
    if let Ok(keys) = keys_result {
        // Build numpy boolean array for indexing
        let np_mask = np.call_method1("array", (mask_arr,))?;
        for key in &keys {
            let arr = obsm.get_item(key)?;
            let sliced = arr.get_item(&np_mask)?;
            obsm.set_item(key, sliced)?;
        }
    }

    Ok(())
}

/// Update all backed layers with a new deletion vector.
fn update_layers_kept_to_global(adata: &Bound<'_, PyAny>, new_kept: &[u64]) -> PyResult<()> {
    let layers = adata.getattr("layers")?;
    let keys_result: PyResult<Vec<String>> = layers
        .call_method0("keys")?
        .try_iter()?
        .map(|k| k.and_then(|k| k.extract()))
        .collect();
    if let Ok(keys) = keys_result {
        for key in &keys {
            let layer_obj = layers.get_item(key)?;
            if let Ok(layer) = layer_obj.downcast::<ScxBackedLayerDataset>() {
                layer
                    .borrow_mut()
                    .inner
                    .set_kept_to_global(new_kept.to_vec());
            }
            // ScxLazyTransformedDataset layers are unlikely but handle them
            if let Ok(lazy_layer) = layer_obj.downcast::<ScxLazyTransformedDataset>() {
                lazy_layer
                    .borrow_mut()
                    .set_kept_to_global(new_kept.to_vec());
            }
        }
    }
    Ok(())
}

/// Update all backed layers with a new column projection.
fn update_layers_col_projection(adata: &Bound<'_, PyAny>, new_cols: &[u32]) -> PyResult<()> {
    let layers = adata.getattr("layers")?;
    let keys_result: PyResult<Vec<String>> = layers
        .call_method0("keys")?
        .try_iter()?
        .map(|k| k.and_then(|k| k.extract()))
        .collect();
    if let Ok(keys) = keys_result {
        for key in &keys {
            let layer_obj = layers.get_item(key)?;
            if let Ok(layer) = layer_obj.downcast::<ScxBackedLayerDataset>() {
                layer
                    .borrow_mut()
                    .inner
                    .set_col_projection(new_cols.to_vec());
            }
            if let Ok(lazy_layer) = layer_obj.downcast::<ScxLazyTransformedDataset>() {
                lazy_layer
                    .borrow_mut()
                    .set_col_projection(new_cols.to_vec());
            }
        }
    }
    Ok(())
}

/// Filter cells (rows) without materializing backed data.
///
/// Replacement for `sc.pp.filter_cells()` that works on backed and
/// lazy-transformed SCX data. Computes row metrics via streaming,
/// builds a boolean mask, and updates the deletion vector on X and
/// layers. Also slices `adata.obs` and `adata.obsm` to match.
///
/// Falls back to `sc.pp.filter_cells()` for regular scipy/dense matrices.
///
/// Args:
///     adata: AnnData object
///     min_genes: Minimum number of genes expressed (row NNZ >= threshold)
///     max_genes: Maximum number of genes expressed (row NNZ <= threshold)
///     min_counts: Minimum total counts per cell (row sum >= threshold)
///     max_counts: Maximum total counts per cell (row sum <= threshold)
#[pyfunction]
#[pyo3(signature = (adata, min_genes=None, max_genes=None, min_counts=None, max_counts=None))]
pub fn filter_cells(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    min_genes: Option<i64>,
    max_genes: Option<i64>,
    min_counts: Option<f64>,
    max_counts: Option<f64>,
) -> PyResult<()> {
    if min_genes.is_none() && max_genes.is_none() && min_counts.is_none() && max_counts.is_none() {
        return Ok(());
    }

    let x = adata.getattr("X")?;
    let need_nnz = min_genes.is_some() || max_genes.is_some();
    let need_sums = min_counts.is_some() || max_counts.is_some();

    // Case 1: X is ScxBackedSparseDataset
    if let Ok(backed) = x.downcast::<ScxBackedSparseDataset>() {
        let n_obs = backed.borrow().shape_val.0;
        let row_nnz = if need_nnz {
            Some(backed_row_nnz(&backed.borrow())?)
        } else {
            None
        };
        let row_sums = if need_sums {
            Some(backed_row_sums(&backed.borrow())?)
        } else {
            None
        };

        let keep = build_keep_mask(
            n_obs,
            row_nnz.as_deref(),
            row_sums.as_deref(),
            min_genes,
            max_genes,
            min_counts,
            max_counts,
        );

        let new_kept = compose_kept_to_global(&keep, backed.borrow().kept_to_global.as_deref());

        backed.borrow_mut().set_kept_to_global(new_kept.clone());
        slice_obs_and_obsm(py, adata, &keep)?;
        update_layers_kept_to_global(adata, &new_kept)?;
        return Ok(());
    }

    // Case 2: X is ScxLazyTransformedDataset
    if let Ok(lazy) = x.downcast::<ScxLazyTransformedDataset>() {
        let lazy_ref = lazy.borrow();
        let n_obs = lazy_ref.shape_val.0;

        // NNZ is transform-invariant: use underlying backed reader
        let row_nnz = if need_nnz {
            let all_nnz = lazy_ref
                .backed
                .row_nnz()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Some(lazy_ref.filter_row_results(&all_nnz))
        } else {
            None
        };

        let row_sums = if need_sums {
            let all_sums = lazy_ref.streaming_row_sums()?;
            Some(lazy_ref.filter_row_results(&all_sums))
        } else {
            None
        };

        let keep = build_keep_mask(
            n_obs,
            row_nnz.as_deref(),
            row_sums.as_deref(),
            min_genes,
            max_genes,
            min_counts,
            max_counts,
        );

        let new_kept = compose_kept_to_global(&keep, lazy_ref.kept_to_global.as_deref());

        drop(lazy_ref);
        lazy.borrow_mut().set_kept_to_global(new_kept.clone());
        slice_obs_and_obsm(py, adata, &keep)?;
        update_layers_kept_to_global(adata, &new_kept)?;
        return Ok(());
    }

    // Case 3: fallback to scanpy
    let sc = py.import("scanpy")?;
    let kwargs = PyDict::new(py);
    if let Some(v) = min_genes {
        kwargs.set_item("min_genes", v)?;
    }
    if let Some(v) = max_genes {
        kwargs.set_item("max_genes", v)?;
    }
    if let Some(v) = min_counts {
        kwargs.set_item("min_counts", v)?;
    }
    if let Some(v) = max_counts {
        kwargs.set_item("max_counts", v)?;
    }
    sc.getattr("pp")?
        .call_method("filter_cells", (adata,), Some(&kwargs))?;
    Ok(())
}

/// Filter genes (columns) without materializing backed data.
///
/// Replacement for `sc.pp.filter_genes()` that works on backed and
/// lazy-transformed SCX data. Computes column metrics via streaming,
/// builds a boolean mask, and sets `col_projection` on X and layers.
/// Also slices `adata.var` to match.
///
/// Falls back to `sc.pp.filter_genes()` for regular scipy/dense matrices.
///
/// **Note:** After `filter_genes`, column projection is active. Streaming
/// PCA via `as_shard_source()` applies the projection per-shard.
///
/// Args:
///     adata: AnnData object
///     min_cells: Minimum number of cells expressing gene (col NNZ >= threshold)
///     max_cells: Maximum number of cells expressing gene (col NNZ <= threshold)
///     min_counts: Minimum total counts per gene (col sum >= threshold)
///     max_counts: Maximum total counts per gene (col sum <= threshold)
#[pyfunction]
#[pyo3(signature = (adata, min_cells=None, max_cells=None, min_counts=None, max_counts=None))]
pub fn filter_genes(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    min_cells: Option<i64>,
    max_cells: Option<i64>,
    min_counts: Option<f64>,
    max_counts: Option<f64>,
) -> PyResult<()> {
    if min_cells.is_none() && max_cells.is_none() && min_counts.is_none() && max_counts.is_none() {
        return Ok(());
    }

    let x = adata.getattr("X")?;
    let need_nnz = min_cells.is_some() || max_cells.is_some();
    let need_sums = min_counts.is_some() || max_counts.is_some();

    // Case 1: X is ScxBackedSparseDataset
    if let Ok(backed) = x.downcast::<ScxBackedSparseDataset>() {
        let n_vars = backed.borrow().shape_val.1;
        let col_nnz = if need_nnz {
            Some(backed_col_nnz(&backed.borrow())?)
        } else {
            None
        };
        let col_sums = if need_sums {
            Some(backed_col_sums(&backed.borrow())?)
        } else {
            None
        };

        let keep = build_keep_mask(
            n_vars,
            col_nnz.as_deref(),
            col_sums.as_deref(),
            min_cells,
            max_cells,
            min_counts,
            max_counts,
        );

        // Compose with existing col_projection
        let new_col_indices: Vec<u32> = match backed.borrow().col_projection() {
            Some(existing) => keep
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| existing[i])
                .collect(),
            None => keep
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| i as u32)
                .collect(),
        };

        // Slice var BEFORE updating col_projection (AnnData validates
        // shape consistency on .var setter, so we use ._var to bypass).
        let mask_arr = numpy::PyArray::from_vec(py, keep);
        let var = adata.getattr("var")?;
        let filtered_var = var.getattr("loc")?.get_item(&mask_arr)?;
        // Use _var to skip shape validation (X.shape changes next)
        adata.setattr("_var", filtered_var)?;

        backed
            .borrow_mut()
            .set_col_projection(new_col_indices.clone());

        update_layers_col_projection(adata, &new_col_indices)?;
        return Ok(());
    }

    // Case 2: X is ScxLazyTransformedDataset
    if let Ok(lazy) = x.downcast::<ScxLazyTransformedDataset>() {
        let lazy_ref = lazy.borrow();
        let n_vars = lazy_ref.shape_val.1;

        // For lazy datasets, NNZ is transform-invariant: use underlying backed reader
        // with the lazy dataset's col_projection and kept_to_global
        let col_nnz = if need_nnz {
            let counts = match (lazy_ref.col_projection(), &lazy_ref.kept_to_global) {
                (Some(cols), Some(kept)) => {
                    projected_agg::col_nnz_masked_projected(&lazy_ref.backed, kept, cols)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                }
                (Some(cols), None) => projected_agg::col_nnz_projected(&lazy_ref.backed, cols)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                (None, Some(kept)) => {
                    let f_counts = lazy_ref
                        .backed
                        .col_nnz_masked(kept)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    f_counts.iter().map(|&v| v as i64).collect()
                }
                (None, None) => lazy_ref
                    .backed
                    .col_nnz()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            };
            Some(counts)
        } else {
            None
        };

        // Col sums through transforms (streaming).
        // streaming_col_sums returns full-width (all original columns).
        // When col_projection is active, extract only projected columns.
        let col_sums = if need_sums {
            let full_sums = if lazy_ref.kept_to_global.is_some() {
                lazy_ref.streaming_col_sums_masked()?
            } else {
                lazy_ref.streaming_col_sums()?
            };
            if let Some(cols) = lazy_ref.col_projection() {
                Some(
                    cols.iter()
                        .map(|&c| full_sums[c as usize])
                        .collect::<Vec<f64>>(),
                )
            } else {
                Some(full_sums)
            }
        } else {
            None
        };

        let keep = build_keep_mask(
            n_vars,
            col_nnz.as_deref(),
            col_sums.as_deref(),
            min_cells,
            max_cells,
            min_counts,
            max_counts,
        );

        let new_col_indices: Vec<u32> = match lazy_ref.col_projection() {
            Some(existing) => keep
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| existing[i])
                .collect(),
            None => keep
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| i as u32)
                .collect(),
        };

        drop(lazy_ref);

        // Slice var BEFORE updating col_projection (AnnData validates
        // shape consistency on .var setter, so we use ._var to bypass).
        let mask_arr = numpy::PyArray::from_vec(py, keep);
        let var = adata.getattr("var")?;
        let filtered_var = var.getattr("loc")?.get_item(&mask_arr)?;
        adata.setattr("_var", filtered_var)?;

        lazy.borrow_mut()
            .set_col_projection(new_col_indices.clone());

        update_layers_col_projection(adata, &new_col_indices)?;
        return Ok(());
    }

    // Case 3: fallback to scanpy
    let sc = py.import("scanpy")?;
    let kwargs = PyDict::new(py);
    if let Some(v) = min_cells {
        kwargs.set_item("min_cells", v)?;
    }
    if let Some(v) = max_cells {
        kwargs.set_item("max_cells", v)?;
    }
    if let Some(v) = min_counts {
        kwargs.set_item("min_counts", v)?;
    }
    if let Some(v) = max_counts {
        kwargs.set_item("max_counts", v)?;
    }
    sc.getattr("pp")?
        .call_method("filter_genes", (adata,), Some(&kwargs))?;
    Ok(())
}

/// Subset observations (rows) without materializing backed data.
///
/// Accepts a boolean mask (numpy array or list of bool) or integer index
/// array (numpy array or list of int) and updates the backed dataset's
/// deletion vector in-place. Also slices `adata.obs`, `adata.obsm`, and
/// updates all backed layers to match.
///
/// This is the general-purpose version of `filter_cells()` — use it for
/// custom QC logic, cluster-based filtering, or any arbitrary subsetting:
///
///     mask = adata.obs['doublet_score'] < 0.5
///     pyscx.accel.subset_obs(adata, mask)
///
///     # Or with integer indices (treated as set membership):
///     indices = [0, 5, 10, 15, 20]
///     pyscx.accel.subset_obs(adata, indices)
///
/// **Note:** Integer indices are converted to a boolean mask internally,
/// so they behave as set membership rather than ordered selection:
/// duplicate indices are silently collapsed, and the original order is
/// not preserved (`[5, 0, 10]` produces the same result as `[0, 5, 10]`).
/// This differs from NumPy fancy indexing. Use a boolean mask if you need
/// exact control over which rows are kept.
///
/// Falls back to standard numpy/pandas subsetting for non-SCX data
/// (materializes X via `adata._X = adata.X[mask]`).
///
/// Args:
///     adata: AnnData object
///     mask_or_indices: Boolean mask (length n_obs) or integer index array
#[pyfunction]
#[pyo3(signature = (adata, mask_or_indices))]
pub fn subset_obs(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    mask_or_indices: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let np = py.import("numpy")?;

    // Get n_obs from adata.shape[0]
    let shape = adata.getattr("shape")?;
    let n_obs: usize = shape.get_item(0)?.extract()?;

    // Convert input to a numpy array to determine dtype
    let arr = np.call_method1("asarray", (mask_or_indices,))?;
    let dtype_str: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;

    // Build boolean keep mask
    let keep: Vec<bool> = match dtype_str.as_str() {
        // Boolean mask
        "b" => {
            let bool_arr: numpy::PyReadonlyArray1<'_, bool> = arr.extract()?;
            let slice = bool_arr
                .as_slice()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            if slice.len() != n_obs {
                return Err(PyValueError::new_err(format!(
                    "Boolean mask length ({}) does not match n_obs ({})",
                    slice.len(),
                    n_obs
                )));
            }
            slice.to_vec()
        }
        // Integer indices
        "i" | "u" => {
            let idx_arr = arr.call_method1("astype", (np.getattr("int64")?,))?;
            let idx_ro: numpy::PyReadonlyArray1<'_, i64> = idx_arr.extract()?;
            let indices = idx_ro
                .as_slice()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            // Validate bounds
            for &idx in indices {
                if idx < 0 || (idx as usize) >= n_obs {
                    return Err(PyValueError::new_err(format!(
                        "Index {} is out of bounds for n_obs={}",
                        idx, n_obs
                    )));
                }
            }

            // Build boolean mask from indices
            let mut mask = vec![false; n_obs];
            for &idx in indices {
                mask[idx as usize] = true;
            }
            mask
        }
        _ => {
            return Err(PyValueError::new_err(
                "mask_or_indices must be a boolean mask or integer index array",
            ));
        }
    };

    // Check if any rows are kept
    let n_kept = keep.iter().filter(|&&k| k).count();
    if n_kept == n_obs {
        // All rows kept — nothing to do
        return Ok(());
    }

    let x = adata.getattr("X")?;

    // Case 1: X is ScxBackedSparseDataset
    if let Ok(backed) = x.downcast::<ScxBackedSparseDataset>() {
        let new_kept = compose_kept_to_global(&keep, backed.borrow().kept_to_global.as_deref());
        backed.borrow_mut().set_kept_to_global(new_kept.clone());
        slice_obs_and_obsm(py, adata, &keep)?;
        update_layers_kept_to_global(adata, &new_kept)?;
        return Ok(());
    }

    // Case 2: X is ScxLazyTransformedDataset
    if let Ok(lazy) = x.downcast::<ScxLazyTransformedDataset>() {
        let new_kept = compose_kept_to_global(&keep, lazy.borrow().kept_to_global.as_deref());
        lazy.borrow_mut().set_kept_to_global(new_kept.clone());
        slice_obs_and_obsm(py, adata, &keep)?;
        update_layers_kept_to_global(adata, &new_kept)?;
        return Ok(());
    }

    // Case 3: Fallback for non-SCX data — materialize subset
    let mask_arr = numpy::PyArray::from_vec(py, keep.clone());
    let np_mask = np.call_method1("array", (mask_arr,))?;

    // Slice X
    let x_sliced = x.get_item(&np_mask)?;
    adata.setattr("_X", x_sliced)?;

    // Slice obs and obsm
    slice_obs_and_obsm(py, adata, &keep)?;

    Ok(())
}
