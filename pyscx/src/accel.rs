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

use scx_format::ShardSource;

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
    rankby_abs: bool,
    tie_correct: bool,
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
            rankby_abs,
            tie_correct,
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
                rankby_abs,
                tie_correct,
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
                rankby_abs,
                tie_correct,
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
#[pyo3(signature = (adata, groupby, reference="rest", n_genes=None, method="wilcoxon", gene_chunk_size=None, stratify_by=None, min_cells_per_stratum=50, rankby_abs=false, tie_correct=false))]
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
    rankby_abs: bool,
    tie_correct: bool,
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
            match run_rank_genes_groups_inner(
                py,
                &sub_adata,
                groupby,
                reference,
                gene_chunk_size,
                rankby_abs,
                tie_correct,
            ) {
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
    let (result, _unique_groups) = run_rank_genes_groups_inner(
        py,
        adata,
        groupby,
        reference,
        gene_chunk_size,
        rankby_abs,
        tie_correct,
    )?;

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
///     n_iterations: Maximum optimization iterations; 2 runs two outer passes
///         (matching leidenalg package default), -1 for until convergence (default: 2)
///     device: Device selection — "auto" (default), "cpu", or "gpu"
///
/// Notes:
///     GPU and CPU Leiden may produce different partitions on the same graph
///     due to algorithmic differences (cuGraph uses a different refinement
///     strategy than leidenalg). Both produce valid, high-quality community
///     structures. Compare results via ARI or NMI when switching backends.
#[pyfunction]
#[pyo3(signature = (adata, resolution=1.0, key_added="leiden", random_state=0, n_iterations=2, device="auto", parallel=false))]
#[allow(clippy::too_many_arguments)]
pub fn leiden(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    resolution: f64,
    key_added: &str,
    random_state: u64,
    n_iterations: i64,
    device: &str,
    parallel: bool,
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

    // Priority 1: Rust-native Leiden (fastest, no Python dependencies)
    match run_rust_leiden(
        py,
        adata,
        &conn,
        resolution,
        key_added,
        random_state,
        n_iterations,
        parallel,
    ) {
        Ok(()) => return Ok(()),
        Err(e) => {
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                (format!(
                    "Rust-native Leiden failed ({e}) — falling back to GPU/Python path"
                ),),
            )?;
        }
    }

    // Priority 2: GPU cuGraph Leiden
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

    // Priority 3: Python leidenalg via igraph (fallback)
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

/// Run Leiden community detection via Rust-native implementation.
///
/// Extracts CSR from the connectivities sparse matrix, calls
/// `scx_accel::leiden()` directly (no Python igraph or leidenalg required),
/// and writes results to adata.
#[allow(clippy::too_many_arguments)]
fn run_rust_leiden(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    conn: &Bound<'_, PyAny>,
    resolution: f64,
    key_added: &str,
    random_state: u64,
    n_iterations: i64,
    parallel: bool,
) -> PyResult<()> {
    let numpy = py.import("numpy")?;
    let pd = py.import("pandas")?;

    // Extract CSR components from connectivities sparse matrix.
    let shape: (usize, usize) = conn.getattr("shape")?.extract()?;
    let n_obs = shape.0;

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

    let max_iter = if n_iterations > 0 {
        n_iterations as usize
    } else {
        0 // 0 means use default (run until convergence)
    };

    // Run Rust-native Leiden — releases the GIL for the compute-heavy part.
    let result = py
        .allow_threads(|| {
            scx_accel::leiden(
                &indptr,
                &indices,
                &data,
                n_obs,
                resolution,
                random_state,
                max_iter,
                parallel,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Rust Leiden error: {e}")))?;

    // Convert membership Vec<usize> to string labels (scanpy convention).
    let membership_strs: Vec<String> = result.membership.iter().map(|c| c.to_string()).collect();
    let labels = pyo3::types::PyList::new(py, &membership_strs)?;
    let cat_labels = pd.call_method1("Categorical", (&labels,))?;

    // Write to adata.obs[key_added]
    let obs = adata.getattr("obs")?;
    obs.set_item(key_added, cat_labels)?;

    // Write metadata to adata.uns[key_added]
    let leiden_dict = PyDict::new(py);
    let params_dict = PyDict::new(py);
    params_dict.set_item("resolution", resolution)?;
    params_dict.set_item("random_state", random_state)?;
    params_dict.set_item("n_iterations", n_iterations)?;
    leiden_dict.set_item("params", params_dict)?;
    leiden_dict.set_item("backend", "scx-accel")?;
    leiden_dict.set_item("modularity", result.modularity)?;
    leiden_dict.set_item("n_communities", result.n_communities)?;

    let uns = adata.getattr("uns")?;
    uns.set_item(key_added, leiden_dict)?;

    Ok(())
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

        // Scanpy compat: sum only over projected (user-visible) genes.
        // After filter_genes(), col_projection restricts to kept genes.
        // Without col_projection, this sums all columns (same as before).
        // Must be in physical-row space because transforms are applied
        // per-shard before deletion vector filtering.
        let all_row_sums = if let Some(cols) = backed_ref.col_projection() {
            projected_agg::row_sums_projected(&backed_ref.backed, cols)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        } else {
            backed_ref
                .backed
                .row_sums()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        };

        let non_negative = backed_ref.non_negative;
        let lazy = ScxLazyTransformedDataset::new(
            Arc::clone(&backed_ref.backed),
            backed_ref.shape_val,
            backed_ref.kept_to_global.clone(),
            // Inherit col_projection: transforms operate on full columns,
            // projection is applied per-shard after transforms
            backed_ref.col_projection_arc(),
            vec![Transform::NormalizeTotal {
                row_sums: Arc::new(all_row_sums),
                target_sum,
            }],
            non_negative,
        );
        // Drop the borrow before setattr to avoid RefCell borrow conflict
        drop(backed_ref);
        adata.setattr("X", Bound::new(py, lazy)?)?;
        return Ok(());
    }

    // Case 2: X is already ScxLazyTransformedDataset — append transform
    if let Ok(lazy) = x.downcast::<ScxLazyTransformedDataset>() {
        let mut lazy_ref = lazy.borrow_mut();
        // Scanpy compat: sum only over projected (user-visible) genes.
        // streaming_row_sums_projected() applies transforms to the full-width
        // shard first (so prior NormalizeTotal sees correct denominator),
        // then project_csr restricts to projected genes before summing.
        // Returns a global-length vector (n_obs_global), which is what
        // apply_transforms_to_csr expects (indexes by global row).
        let sums = lazy_ref.streaming_row_sums_projected()?;
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

        let non_negative = backed_ref.non_negative;
        let lazy = ScxLazyTransformedDataset::new(
            Arc::clone(&backed_ref.backed),
            backed_ref.shape_val,
            backed_ref.kept_to_global.clone(),
            // Inherit col_projection so user-visible shape matches adata.var
            backed_ref.col_projection_arc(),
            vec![Transform::Log1p],
            non_negative,
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

/// Helper: compute row NNZ and sums in a single shard scan (fused).
///
/// Avoids the double I/O of `backed_row_nnz()` + `backed_row_sums()` when
/// `filter_cells` needs both `min_genes` and `min_counts`.
fn backed_row_nnz_and_sums(backed: &ScxBackedSparseDataset) -> PyResult<(Vec<i64>, Vec<f64>)> {
    let (all_nnz, all_sums) = if let Some(cols) = backed.col_projection() {
        projected_agg::row_nnz_and_sums_projected(&backed.backed, cols)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    } else {
        backed
            .backed
            .row_nnz_and_sums()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    };
    Ok((
        backed.filter_row_results(&all_nnz),
        backed.filter_row_results(&all_sums),
    ))
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
        // Fused: compute both NNZ and sums in a single shard scan when both are needed
        let (row_nnz, row_sums) = if need_nnz && need_sums {
            let (nnz, sums) = backed_row_nnz_and_sums(&backed.borrow())?;
            (Some(nnz), Some(sums))
        } else {
            (
                if need_nnz {
                    Some(backed_row_nnz(&backed.borrow())?)
                } else {
                    None
                },
                if need_sums {
                    Some(backed_row_sums(&backed.borrow())?)
                } else {
                    None
                },
            )
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

        let new_kept = compose_kept_to_global(
            &keep,
            backed
                .borrow()
                .kept_to_global
                .as_ref()
                .map(|v| v.as_slice()),
        );

        backed.borrow_mut().set_kept_to_global(new_kept.clone());
        slice_obs_and_obsm(py, adata, &keep)?;
        update_layers_kept_to_global(adata, &new_kept)?;
        return Ok(());
    }

    // Case 2: X is ScxLazyTransformedDataset
    if let Ok(lazy) = x.downcast::<ScxLazyTransformedDataset>() {
        let lazy_ref = lazy.borrow();
        let n_obs = lazy_ref.shape_val.0;

        // Fused: when both NNZ and sums are needed, compute them in a single
        // shard scan. NNZ is transform-invariant but we compute it from the
        // same decoded shard to avoid double I/O.
        let (row_nnz, row_sums) = if need_nnz && need_sums {
            let (all_nnz, all_sums) = lazy_ref.streaming_row_nnz_and_sums()?;
            (
                Some(lazy_ref.filter_row_results(&all_nnz)),
                Some(lazy_ref.filter_row_results(&all_sums)),
            )
        } else {
            let nnz = if need_nnz {
                let all_nnz = lazy_ref
                    .backed
                    .row_nnz()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Some(lazy_ref.filter_row_results(&all_nnz))
            } else {
                None
            };
            let sums = if need_sums {
                let all_sums = lazy_ref.streaming_row_sums()?;
                Some(lazy_ref.filter_row_results(&all_sums))
            } else {
                None
            };
            (nnz, sums)
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

        let new_kept = compose_kept_to_global(
            &keep,
            lazy_ref.kept_to_global.as_ref().map(|v| v.as_slice()),
        );

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
        let new_kept = compose_kept_to_global(
            &keep,
            backed
                .borrow()
                .kept_to_global
                .as_ref()
                .map(|v| v.as_slice()),
        );
        backed.borrow_mut().set_kept_to_global(new_kept.clone());
        slice_obs_and_obsm(py, adata, &keep)?;
        update_layers_kept_to_global(adata, &new_kept)?;
        return Ok(());
    }

    // Case 2: X is ScxLazyTransformedDataset
    if let Ok(lazy) = x.downcast::<ScxLazyTransformedDataset>() {
        let new_kept = compose_kept_to_global(
            &keep,
            lazy.borrow().kept_to_global.as_ref().map(|v| v.as_slice()),
        );
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

// ──────────────────────────────────────────────────────────────────────────────
// Highly Variable Genes
// ──────────────────────────────────────────────────────────────────────────────

/// Streaming highly-variable gene selection without materialization.
///
/// Computes HVG statistics shard-by-shard via the `ShardSource` abstraction,
/// then selects the top `n_top_genes` by normalized variance (seurat_v3) or
/// normalized dispersion (seurat).
///
/// Args:
///     adata: AnnData with X as ScxBackedSparseDataset or ScxLazyTransformedDataset
///     n_top_genes: Number of highly variable genes to select (default: 2000)
///     flavor: "seurat_v3" (raw counts) or "seurat" (log-normalized) (default: "seurat_v3")
///     batch_key: Column in adata.obs for batch-aware HVG (default: None)
///     span: Loess span for seurat_v3 (default: 0.3)
///     subset: If True, subset adata to HVG via column projection (default: False)
///     n_bins: Number of bins for seurat flavor (default: 20)
#[pyfunction]
#[pyo3(signature = (adata, n_top_genes=2000, flavor="seurat_v3", batch_key=None, span=0.3, subset=false, n_bins=20))]
#[allow(clippy::too_many_arguments)]
pub fn highly_variable_genes<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    n_top_genes: usize,
    flavor: &str,
    batch_key: Option<&str>,
    span: f64,
    subset: bool,
    n_bins: usize,
) -> PyResult<()> {
    let x = adata.getattr("X")?;

    // ── Try SCX backed dataset ──────────────────────────────────────────
    if let Ok(backed) = x.downcast::<ScxBackedSparseDataset>() {
        let backed_ref = backed.borrow();
        let reader = Arc::clone(&backed_ref.backed);
        let n_vars = backed_ref.shape_val.1;
        let n_obs = backed_ref.shape_val.0;
        let kept = backed_ref.kept_to_global.clone();
        let col_proj = backed_ref.col_projection_arc();
        drop(backed_ref);

        return hvg_on_source(
            py,
            adata,
            &x,
            reader,
            vec![],
            kept,
            col_proj,
            n_obs,
            n_vars,
            n_top_genes,
            flavor,
            batch_key,
            span,
            subset,
            n_bins,
        );
    }

    // ── Try SCX lazy-transformed dataset ────────────────────────────────
    if let Ok(lazy) = x.downcast::<ScxLazyTransformedDataset>() {
        let lazy_ref = lazy.borrow();
        let reader = Arc::clone(&lazy_ref.backed);
        let transforms = lazy_ref.transforms.clone();
        let n_vars = lazy_ref.shape_val.1;
        let n_obs = lazy_ref.shape_val.0;
        let kept = lazy_ref.kept_to_global.clone();
        let col_proj = lazy_ref.col_projection.clone();
        drop(lazy_ref);

        return hvg_on_source(
            py,
            adata,
            &x,
            reader,
            transforms,
            kept,
            col_proj,
            n_obs,
            n_vars,
            n_top_genes,
            flavor,
            batch_key,
            span,
            subset,
            n_bins,
        );
    }

    // ── Fallback to scanpy ──────────────────────────────────────────────
    let sc = py.import("scanpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("n_top_genes", n_top_genes)?;
    kwargs.set_item("flavor", flavor)?;
    kwargs.set_item("span", span)?;
    kwargs.set_item("subset", subset)?;
    kwargs.set_item("n_bins", n_bins)?;
    if let Some(bk) = batch_key {
        kwargs.set_item("batch_key", bk)?;
    }
    sc.getattr("pp")?
        .call_method("highly_variable_genes", (adata,), Some(&kwargs))?;
    Ok(())
}

/// Core HVG logic for SCX-backed sources. Dispatches to seurat_v3 or seurat.
#[allow(clippy::too_many_arguments)]
fn hvg_on_source<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    x_obj: &Bound<'py, PyAny>,
    reader: Arc<scx_format::BackedCsrReader>,
    transforms: Vec<Transform>,
    kept_to_global: Option<Arc<Vec<u64>>>,
    col_projection: Option<Arc<Vec<u32>>>,
    n_obs: usize,
    n_vars: usize,
    n_top_genes: usize,
    flavor: &str,
    batch_key: Option<&str>,
    span: f64,
    subset: bool,
    n_bins: usize,
) -> PyResult<()> {
    match flavor {
        "seurat_v3" | "seurat_v3_paper" => hvg_seurat_v3(
            py,
            adata,
            x_obj,
            reader,
            transforms,
            kept_to_global,
            col_projection,
            n_obs,
            n_vars,
            n_top_genes,
            batch_key,
            span,
            subset,
            flavor,
        ),
        "seurat" => hvg_seurat(
            py,
            adata,
            x_obj,
            reader,
            transforms,
            kept_to_global,
            col_projection,
            n_obs,
            n_vars,
            n_top_genes,
            batch_key,
            subset,
            n_bins,
        ),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported HVG flavor '{flavor}'. Use 'seurat_v3' or 'seurat'."
        ))),
    }
}

/// Build a LazyShardSource, optionally filtered to a batch of cells.
fn build_shard_source(
    reader: &Arc<scx_format::BackedCsrReader>,
    transforms: &[Transform],
    kept_to_global: &Option<Arc<Vec<u64>>>,
    col_projection: &Option<Arc<Vec<u32>>>,
    n_vars: usize,
    batch_indices: Option<&[usize]>,
) -> crate::lazy_transform::LazyShardSource {
    use crate::lazy_transform::LazyShardSource;

    match batch_indices {
        Some(indices) => {
            // Compose batch indices with existing kept_to_global
            let global_rows: Vec<u64> = match kept_to_global {
                Some(existing) => indices.iter().map(|&i| existing[i]).collect(),
                None => indices.iter().map(|&i| i as u64).collect(),
            };
            LazyShardSource::with_kept_rows(
                Arc::clone(reader),
                transforms.to_vec(),
                global_rows,
                col_projection.clone(),
                n_vars,
            )
        }
        None => {
            // Full dataset (or existing kept_to_global).
            // Pass None when no filtering needed — avoids allocating a full
            // identity range and skips the deletion-vector path in read_shard.
            let n_obs = match kept_to_global {
                Some(ref k) => k.len(),
                None => reader.shape().0,
            };
            LazyShardSource::new(
                Arc::clone(reader),
                transforms.to_vec(),
                kept_to_global.as_ref().map(Arc::clone),
                col_projection.clone(),
                n_obs,
                n_vars,
            )
        }
    }
}

/// seurat_v3 flavor: raw count data, loess fit, clipped variance.
#[allow(clippy::too_many_arguments)]
fn hvg_seurat_v3<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    x_obj: &Bound<'py, PyAny>,
    reader: Arc<scx_format::BackedCsrReader>,
    transforms: Vec<Transform>,
    kept_to_global: Option<Arc<Vec<u64>>>,
    col_projection: Option<Arc<Vec<u32>>>,
    n_obs: usize,
    n_vars: usize,
    n_top_genes: usize,
    batch_key: Option<&str>,
    span: f64,
    subset: bool,
    flavor: &str,
) -> PyResult<()> {
    // ── 1. Determine batches ────────────────────────────────────────────
    let batches: Vec<Vec<usize>> = match batch_key {
        Some(bk) => {
            let obs = adata.getattr("obs")?;
            let batch_col = obs.get_item(bk)?;
            // Handle both categorical and non-categorical columns:
            // wrap in pd.Categorical() which is a no-op for already-categorical data.
            let pd = py.import("pandas")?;
            let np = py.import("numpy")?;
            let cat = pd.call_method1("Categorical", (&batch_col,))?;
            let codes = cat.getattr("codes")?;
            let cat_codes: Vec<i64> = np
                .call_method1("asarray", (&codes,))?
                .call_method1("astype", ("int64",))?
                .extract()?;
            let n_batches = *cat_codes.iter().max().unwrap_or(&0) as usize + 1;
            let mut groups = vec![vec![]; n_batches];
            for (i, &code) in cat_codes.iter().enumerate() {
                if code >= 0 {
                    groups[code as usize].push(i);
                }
            }
            groups.into_iter().filter(|g| !g.is_empty()).collect()
        }
        None => vec![(0..n_obs).collect()],
    };

    let n_batches_actual = batches.len();

    // Build cell-to-batch mapping for batched streaming
    let mut cell_batch = vec![-1i32; n_obs];
    for (batch_id, batch_cells) in batches.iter().enumerate() {
        for &cell_idx in batch_cells {
            cell_batch[cell_idx] = batch_id as i32;
        }
    }

    // ── 2. Batched streaming mean/var (single pass for ALL batches + global) ──
    let source = build_shard_source(
        &reader,
        &transforms,
        &kept_to_global,
        &col_projection,
        n_vars,
        None,
    );
    let batched_stats =
        scx_accel::streaming_mean_var_batched(&source, &cell_batch, n_batches_actual)
            .map_err(|e| PyRuntimeError::new_err(format!("streaming_mean_var_batched: {e}")))?;

    let global_stats = batched_stats.global.clone();

    // ── 3. Per-batch: loess fit → clip_val (in-memory, no I/O) ───────────
    let mut all_clip_vals: Vec<Vec<f64>> = Vec::new();
    let mut batch_estimat_vars: Vec<Vec<f64>> = Vec::new();

    for (b, batch_cells) in batches.iter().enumerate() {
        let batch_n = batch_cells.len();
        if batch_n < 2 {
            all_clip_vals.push(vec![0.0; n_vars]);
            batch_estimat_vars.push(vec![0.0; n_vars]);
            continue;
        }

        let batch_stats = &batched_stats.per_batch[b];

        // Loess fit via Python (on non-constant genes)
        let mut estimat_var = vec![0.0f64; n_vars];
        let not_const: Vec<bool> = batch_stats.variances.iter().map(|&v| v > 0.0).collect();
        let x_vals: Vec<f64> = batch_stats
            .means
            .iter()
            .zip(not_const.iter())
            .filter(|(_, &nc)| nc)
            .map(|(&m, _)| m.max(1e-300).log10())
            .collect();
        let y_vals: Vec<f64> = batch_stats
            .variances
            .iter()
            .zip(not_const.iter())
            .filter(|(_, &nc)| nc)
            .map(|(&v, _)| v.max(1e-300).log10())
            .collect();

        if x_vals.len() >= 3 {
            let x_arr = numpy::PyArray::from_vec(py, x_vals);
            let y_arr = numpy::PyArray::from_vec(py, y_vals);

            let loess_mod = py.import("skmisc.loess")?;
            let loess_cls = loess_mod.getattr("loess")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("span", span)?;
            kwargs.set_item("degree", 2)?;
            let model = loess_cls.call((x_arr, y_arr), Some(&kwargs))?;
            model.call_method0("fit")?;
            let fitted: Vec<f64> = model
                .getattr("outputs")?
                .getattr("fitted_values")?
                .extract()?;

            let mut fi = 0;
            for (j, &nc) in not_const.iter().enumerate() {
                if nc {
                    estimat_var[j] = fitted[fi];
                    fi += 1;
                }
            }
        }

        // reg_std and clip_val
        let mut clip_val = vec![0.0f64; n_vars];
        let batch_n_f = batch_n as f64;
        let sqrt_n = batch_n_f.sqrt();
        for j in 0..n_vars {
            let reg_std = 10.0f64.powf(estimat_var[j]).sqrt();
            clip_val[j] = reg_std * sqrt_n + batch_stats.means[j];
        }

        all_clip_vals.push(clip_val);
        batch_estimat_vars.push(estimat_var);
    }

    // ── 4. Batched streaming clipped sums (single pass for ALL batches) ──
    let all_clipped = scx_accel::streaming_clip_square_sum_batched(
        &source,
        &cell_batch,
        n_batches_actual,
        &all_clip_vals,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("streaming_clip_square_sum_batched: {e}")))?;

    // ── 5. Compute normalized variance per batch (in-memory) ─────────────
    let mut all_norm_vars: Vec<Vec<f64>> = Vec::new();
    for (b, batch_cells) in batches.iter().enumerate() {
        let batch_n = batch_cells.len();
        if batch_n < 2 {
            all_norm_vars.push(vec![0.0; n_vars]);
            continue;
        }

        let batch_stats = &batched_stats.per_batch[b];
        let (ref bcs, ref sbcs) = all_clipped[b];
        let estimat_var = &batch_estimat_vars[b];
        let batch_n_f = batch_n as f64;
        let denom_n = (batch_n_f - 1.0).max(1.0);

        let mut norm_gene_var = vec![0.0f64; n_vars];
        for j in 0..n_vars {
            let reg_std_sq = 10.0f64.powf(estimat_var[j]);
            if reg_std_sq > 0.0 {
                norm_gene_var[j] = (1.0 / (denom_n * reg_std_sq))
                    * (batch_n_f * batch_stats.means[j] * batch_stats.means[j] + sbcs[j]
                        - 2.0 * bcs[j] * batch_stats.means[j]);
            }
        }
        all_norm_vars.push(norm_gene_var);
    }

    // ── 4. Rank genes and select top N ──────────────────────────────────
    let n_batches = all_norm_vars.len();

    // Mean normalized variance across batches
    let mut mean_norm_var = vec![0.0f64; n_vars];
    for nv in &all_norm_vars {
        for (j, &v) in nv.iter().enumerate() {
            mean_norm_var[j] += v;
        }
    }
    for v in &mut mean_norm_var {
        *v /= n_batches as f64;
    }

    // For multi-batch: rank within each batch, then combine ranks
    let (hvg_mask, ranks) = if n_batches > 1 {
        // Per-batch ranks: for each batch, rank genes by normalized variance (descending)
        let mut batch_ranks: Vec<Vec<usize>> = Vec::new();
        for nv in &all_norm_vars {
            let mut indices: Vec<usize> = (0..n_vars).collect();
            indices.sort_by(|&a, &b| {
                nv[b]
                    .partial_cmp(&nv[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut rank = vec![0usize; n_vars];
            for (r, &idx) in indices.iter().enumerate() {
                rank[idx] = r;
            }
            batch_ranks.push(rank);
        }

        // Count in how many batches each gene is in top n_top_genes
        let mut nbatches_hv = vec![0usize; n_vars];
        let mut median_ranks = vec![f64::NAN; n_vars];
        for j in 0..n_vars {
            let ranks_j: Vec<usize> = batch_ranks.iter().map(|br| br[j]).collect();
            nbatches_hv[j] = ranks_j.iter().filter(|&&r| r < n_top_genes).count();
            // Median of ranks where gene is in top n_top_genes
            let mut valid: Vec<f64> = ranks_j
                .iter()
                .filter(|&&r| r < n_top_genes)
                .map(|&r| r as f64)
                .collect();
            if !valid.is_empty() {
                valid.sort_by(|a, b| a.partial_cmp(b).unwrap());
                median_ranks[j] = valid[valid.len() / 2];
            }
        }

        // Sort genes: by nbatches (desc), then median_rank (asc)
        let mut gene_order: Vec<usize> = (0..n_vars).collect();
        if flavor == "seurat_v3_paper" {
            gene_order.sort_by(|&a, &b| {
                nbatches_hv[b].cmp(&nbatches_hv[a]).then(
                    median_ranks[a]
                        .partial_cmp(&median_ranks[b])
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
            });
        } else {
            gene_order.sort_by(|&a, &b| {
                median_ranks[a]
                    .partial_cmp(&median_ranks[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(nbatches_hv[b].cmp(&nbatches_hv[a]))
            });
        }

        let mut mask = vec![false; n_vars];
        let mut rank_out = vec![f64::NAN; n_vars];
        for (r, &g) in gene_order.iter().enumerate().take(n_top_genes.min(n_vars)) {
            mask[g] = true;
            rank_out[g] = r as f64;
        }
        (mask, rank_out)
    } else {
        // Single batch: simple rank by normalized variance descending
        let mut indices: Vec<usize> = (0..n_vars).collect();
        indices.sort_by(|&a, &b| {
            mean_norm_var[b]
                .partial_cmp(&mean_norm_var[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut mask = vec![false; n_vars];
        let mut rank_out = vec![f64::NAN; n_vars];
        for (r, &g) in indices.iter().enumerate().take(n_top_genes.min(n_vars)) {
            mask[g] = true;
            rank_out[g] = r as f64;
        }
        (mask, rank_out)
    };

    // ── 5. Write results to adata.var ───────────────────────────────────
    let var = adata.getattr("var")?;
    var.set_item(
        "highly_variable",
        numpy::PyArray::from_vec(py, hvg_mask.clone()),
    )?;
    var.set_item("means", numpy::PyArray::from_vec(py, global_stats.means))?;
    var.set_item(
        "variances",
        numpy::PyArray::from_vec(py, global_stats.variances),
    )?;
    var.set_item(
        "variances_norm",
        numpy::PyArray::from_vec(py, mean_norm_var),
    )?;
    var.set_item("highly_variable_rank", numpy::PyArray::from_vec(py, ranks))?;

    // ── 6. Subset if requested ──────────────────────────────────────────
    if subset {
        apply_hvg_subset(py, adata, x_obj, &hvg_mask)?;
    }

    Ok(())
}

/// seurat flavor: log-normalized data, binned dispersion normalization.
#[allow(clippy::too_many_arguments)]
fn hvg_seurat<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    x_obj: &Bound<'py, PyAny>,
    reader: Arc<scx_format::BackedCsrReader>,
    transforms: Vec<Transform>,
    kept_to_global: Option<Arc<Vec<u64>>>,
    col_projection: Option<Arc<Vec<u32>>>,
    _n_obs: usize,
    n_vars: usize,
    n_top_genes: usize,
    batch_key: Option<&str>,
    subset: bool,
    n_bins: usize,
) -> PyResult<()> {
    // For batched seurat, fall back to scanpy (complex aggregation logic)
    if batch_key.is_some() {
        let sc = py.import("scanpy")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("n_top_genes", n_top_genes)?;
        kwargs.set_item("flavor", "seurat")?;
        kwargs.set_item("subset", subset)?;
        kwargs.set_item("n_bins", n_bins)?;
        kwargs.set_item("batch_key", batch_key)?;
        sc.getattr("pp")?
            .call_method("highly_variable_genes", (adata,), Some(&kwargs))?;
        return Ok(());
    }

    // ── 1. Streaming mean/var ───────────────────────────────────────────
    let source = build_shard_source(
        &reader,
        &transforms,
        &kept_to_global,
        &col_projection,
        n_vars,
        None,
    );
    let stats = scx_accel::streaming_mean_var(&source)
        .map_err(|e| PyRuntimeError::new_err(format!("streaming_mean_var: {e}")))?;

    // ── 2. Compute dispersion (matching scanpy's seurat flavor) ────────
    let mut dispersions = vec![0.0f64; n_vars];
    let mut log_dispersions = vec![f64::NAN; n_vars];
    let mut log_means = vec![0.0f64; n_vars];
    let mut means_for_disp = stats.means.clone();

    for j in 0..n_vars {
        // scanpy: mean[mean == 0] = 1e-12 (before dispersion computation)
        if means_for_disp[j] == 0.0 {
            means_for_disp[j] = 1e-12;
        }
        dispersions[j] = stats.variances[j] / means_for_disp[j];
        // scanpy: dispersion[dispersion == 0] = NaN, then log(dispersion)
        if dispersions[j] > 0.0 {
            log_dispersions[j] = dispersions[j].ln();
        } else {
            dispersions[j] = f64::NAN;
        }
        // scanpy: mean = log1p(mean) — overwrite mean with log1p for binning
        log_means[j] = (means_for_disp[j] + 1.0).ln();
    }

    // ── 3. Bin by mean, z-score dispersion within bins (via Python) ────
    let log_means_np = numpy::PyArray::from_vec(py, log_means);
    let log_disp_np = numpy::PyArray::from_vec(py, log_dispersions);

    let helpers = py.import("pyscx._hvg_helpers")?;
    let dispersions_norm: Vec<f64> = helpers
        .call_method1(
            "binned_dispersion_norm",
            (log_means_np, log_disp_np, n_bins),
        )?
        .extract()?;

    // ── 4. Select top genes by normalized dispersion ────────────────────
    let mut indices: Vec<usize> = (0..n_vars).collect();
    indices.sort_by(|&a, &b| {
        dispersions_norm[b]
            .partial_cmp(&dispersions_norm[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut mask = vec![false; n_vars];
    for &g in indices.iter().take(n_top_genes.min(n_vars)) {
        mask[g] = true;
    }

    // ── 5. Write results to adata.var ───────────────────────────────────
    let var = adata.getattr("var")?;
    var.set_item(
        "highly_variable",
        numpy::PyArray::from_vec(py, mask.clone()),
    )?;
    var.set_item("means", numpy::PyArray::from_vec(py, stats.means))?;
    var.set_item("dispersions", numpy::PyArray::from_vec(py, dispersions))?;
    var.set_item(
        "dispersions_norm",
        numpy::PyArray::from_vec(
            py,
            dispersions_norm
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<f32>>(),
        ),
    )?;

    // ── 6. Subset if requested ──────────────────────────────────────────
    if subset {
        apply_hvg_subset(py, adata, x_obj, &mask)?;
    }

    Ok(())
}

/// Apply HVG subset: set column projection on X (and layers), slice var.
///
/// Order: update X col_projection + layers FIRST so shapes match,
/// then set `_var` (bypassing AnnData shape validation).
fn apply_hvg_subset(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    x_obj: &Bound<'_, PyAny>,
    hvg_mask: &[bool],
) -> PyResult<()> {
    let mask_arr = numpy::PyArray::from_vec(py, hvg_mask.to_vec());

    if let Ok(backed) = x_obj.downcast::<ScxBackedSparseDataset>() {
        let new_col_indices: Vec<u32> = match backed.borrow().col_projection() {
            Some(existing) => hvg_mask
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| existing[i])
                .collect(),
            None => hvg_mask
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| i as u32)
                .collect(),
        };

        // Update X and layers FIRST so shapes are consistent
        backed
            .borrow_mut()
            .set_col_projection(new_col_indices.clone());
        update_layers_col_projection(adata, &new_col_indices)?;

        // Slice var via _var: AnnData's public var setter validates
        // len(value) == self.n_vars, where n_vars is derived from the current
        // _var DataFrame. Since we're changing the column count, the public
        // setter would reject the new (shorter) DataFrame. Setting _var
        // directly is the same approach used by anndata's own _inplace_subset_var.
        let var = adata.getattr("var")?;
        let filtered_var = var.getattr("loc")?.get_item(&mask_arr)?;
        adata.setattr("_var", filtered_var)?;
    } else if let Ok(lazy) = x_obj.downcast::<ScxLazyTransformedDataset>() {
        let new_col_indices: Vec<u32> = match lazy.borrow().col_projection() {
            Some(existing) => hvg_mask
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| existing[i])
                .collect(),
            None => hvg_mask
                .iter()
                .enumerate()
                .filter(|(_, &k)| k)
                .map(|(i, _)| i as u32)
                .collect(),
        };

        lazy.borrow_mut()
            .set_col_projection(new_col_indices.clone());
        update_layers_col_projection(adata, &new_col_indices)?;

        // See comment above in backed branch for why _var is used.
        let var = adata.getattr("var")?;
        let filtered_var = var.getattr("loc")?.get_item(&mask_arr)?;
        adata.setattr("_var", filtered_var)?;
    }

    Ok(())
}

/// Call `array.astype(dtype, copy=False)` — avoids a deep copy when the
/// source already has the target dtype. This mirrors numpy's behavior where
/// `copy=False` returns the same array object if no conversion is needed.
fn astype_no_copy<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("copy", false)?;
    arr.call_method("astype", (dtype,), Some(&kwargs))
}

/// Holds borrowed CSR array slices extracted from a scipy CSR matrix.
///
/// The `PyReadonlyArray1` borrows keep the underlying numpy arrays alive
/// for the lifetime `'py`.
struct CsrSlices<'py> {
    _indptr: numpy::PyReadonlyArray1<'py, i64>,
    _indices: numpy::PyReadonlyArray1<'py, i32>,
    _data: numpy::PyReadonlyArray1<'py, f32>,
}

impl<'py> CsrSlices<'py> {
    fn indptr(&self) -> &[i64] {
        // SAFETY: the readonly array is guaranteed contiguous by the
        // as_slice() check in extract_csr_slices.
        self._indptr.as_slice().unwrap()
    }
    fn indices(&self) -> &[i32] {
        self._indices.as_slice().unwrap()
    }
    fn data(&self) -> &[f32] {
        self._data.as_slice().unwrap()
    }
}

/// Extract CSR indptr/indices/data as borrowed Rust slices from a scipy
/// CSR matrix object.
///
/// This consolidates the repeated `getattr → asarray → astype_no_copy →
/// PyReadonlyArray1 → as_slice` pattern used by multiple bindings.
/// Emits a Python `warnings.warn()` if the CSR `nnz` exceeds `warn_nnz`
/// (set to 0 to suppress the warning).
fn extract_csr_slices<'py>(
    py: Python<'py>,
    np: &Bound<'py, PyModule>,
    csr: &Bound<'py, PyAny>,
    warn_label: &str,
    warn_nnz: usize,
) -> PyResult<CsrSlices<'py>> {
    // Optional large-data warning based on nnz.
    if warn_nnz > 0 {
        let nnz: usize = csr.getattr("nnz")?.extract()?;
        // Each nonzero costs 4 bytes (data) + 4 bytes (index) = 8 bytes.
        // indptr is small relative to nnz for large matrices.
        let estimated_bytes = nnz * 8;
        if estimated_bytes > 2_000_000_000 {
            let gb = estimated_bytes as f64 / 1e9;
            let shape: (usize, usize) = csr.getattr("shape")?.extract()?;
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                (format!(
                    "{warn_label}: materializing a {:.1} GB CSR matrix ({} × {}, nnz={nnz}). \
                     Consider subsetting the data for better performance.",
                    gb, shape.0, shape.1
                ),),
            )?;
        }
    }

    let indptr_obj = csr.getattr("indptr")?;
    let indptr_arr = np.call_method1("asarray", (&indptr_obj,))?;
    let indptr_arr = astype_no_copy(py, &indptr_arr, "int64")?;
    let indptr_ro: numpy::PyReadonlyArray1<'_, i64> = indptr_arr.extract()?;
    indptr_ro
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(format!("indptr not contiguous: {e}")))?;

    let indices_obj = csr.getattr("indices")?;
    let indices_arr = np.call_method1("asarray", (&indices_obj,))?;
    let indices_arr = astype_no_copy(py, &indices_arr, "int32")?;
    let indices_ro: numpy::PyReadonlyArray1<'_, i32> = indices_arr.extract()?;
    indices_ro
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(format!("indices not contiguous: {e}")))?;

    let data_obj = csr.getattr("data")?;
    let data_arr = np.call_method1("asarray", (&data_obj,))?;
    let data_arr = astype_no_copy(py, &data_arr, "float32")?;
    let data_ro: numpy::PyReadonlyArray1<'_, f32> = data_arr.extract()?;
    data_ro
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(format!("data not contiguous: {e}")))?;

    Ok(CsrSlices {
        _indptr: indptr_ro,
        _indices: indices_ro,
        _data: data_ro,
    })
}

/// Compute pseudobulk means (group-by mean on sparse X).
///
/// Aggregates single-cell expression into per-group means, returning the
/// dense means matrix and group names. This is the foundation for
/// perturbation evaluation metrics (pearson_delta, MSE, discrimination
/// score, etc.).
///
/// Supports backed SCX, lazy-transformed, scipy CSR, and dense numpy inputs.
///
/// Args:
///     adata: AnnData object with X and obs columns for groupby
///     groupby: Column name in adata.obs to group by (e.g., "perturbation")
///     min_cells_per_group: Skip groups with fewer cells (default: 1)
///
/// Returns:
///     Tuple of (means, group_names):
///     - means: numpy array of shape [P, G] (float64) — per-group mean expression
///     - group_names: list of str — group names in order
///
/// Example:
///     means, groups = pyscx.accel.pseudobulk_means(adata, "perturbation")
///     # means.shape == (n_perturbations, n_genes)
///     # groups == ["control", "drug_A", "drug_B", ...]
#[pyfunction]
#[pyo3(signature = (adata, groupby, min_cells_per_group=1))]
pub fn pseudobulk_means<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    groupby: &str,
    min_cells_per_group: usize,
) -> PyResult<PyObject> {
    let np = py.import("numpy")?;

    // Extract groupby column from adata.obs as Vec<String>.
    let obs = adata.getattr("obs")?;
    let col = obs.get_item(groupby).map_err(|_| {
        PyValueError::new_err(format!(
            "groupby column '{}' not found in adata.obs",
            groupby
        ))
    })?;
    let labels: Vec<String> = col
        .call_method1("astype", ("str",))?
        .call_method0("tolist")?
        .extract()?;
    let obs_groups = vec![labels];
    let groupby_columns = vec![groupby.to_string()];

    // Get gene names.
    let var = adata.getattr("var")?;
    let var_names = var.getattr("index")?;
    let gene_names: Vec<String> = var_names.call_method0("tolist")?.extract()?;

    // Perform aggregation with Mean method: backed, lazy-transformed, or in-memory.
    let x = adata.getattr("X")?;
    let result = if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        scx_accel::pseudobulk_aggregate(
            &backed.backed,
            &obs_groups,
            &groupby_columns,
            &gene_names,
            scx_accel::AggregationMethod::Mean,
            min_cells_per_group,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    } else if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
        // Lazy-transformed datasets: materialize through the transform pipeline
        // (normalize, log1p, etc.) to get a scipy CSR, then aggregate in-memory.
        let scipy_csr = lazy.to_memory_py(py)?;
        let shape: (usize, usize) = scipy_csr.getattr("shape")?.extract()?;

        // Use astype with copy=False to avoid redundant copies when dtypes match,
        // then borrow via PyReadonlyArray1 for zero-copy slice access.
        let indptr_obj = scipy_csr.getattr("indptr")?;
        let indptr_arr = np.call_method1("asarray", (&indptr_obj,))?;
        let indptr_arr = astype_no_copy(py, &indptr_arr, "int64")?;
        let indptr_ro: numpy::PyReadonlyArray1<'_, i64> = indptr_arr.extract()?;
        let indptr_slice = indptr_ro
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let indices_obj = scipy_csr.getattr("indices")?;
        let indices_arr = np.call_method1("asarray", (&indices_obj,))?;
        let indices_arr = astype_no_copy(py, &indices_arr, "int32")?;
        let indices_ro: numpy::PyReadonlyArray1<'_, i32> = indices_arr.extract()?;
        let indices_slice = indices_ro
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let data_obj = scipy_csr.getattr("data")?;
        let data_arr = np.call_method1("asarray", (&data_obj,))?;
        let data_arr = astype_no_copy(py, &data_arr, "float32")?;
        let data_ro: numpy::PyReadonlyArray1<'_, f32> = data_arr.extract()?;
        let data_slice = data_ro
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        scx_accel::pseudobulk_aggregate_from_slices(
            shape,
            indptr_slice,
            indices_slice,
            data_slice,
            &obs_groups,
            &groupby_columns,
            &gene_names,
            scx_accel::AggregationMethod::Mean,
            min_cells_per_group,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    } else {
        // In-memory: extract scipy CSR → zero-copy slices.
        let scipy_sparse = py.import("scipy.sparse")?;
        let is_sparse = scipy_sparse
            .call_method1("issparse", (&x,))?
            .extract::<bool>()?;

        let csr_obj = if is_sparse {
            scipy_sparse.call_method1("csr_matrix", (&x,))?
        } else if x.hasattr("toarray")? {
            let arr = x.call_method0("toarray")?;
            scipy_sparse.call_method1("csr_matrix", (&arr,))?
        } else {
            let arr = np
                .call_method1("asarray", (&x,))?
                .call_method1("astype", ("float32",))?;
            scipy_sparse.call_method1("csr_matrix", (&arr,))?
        };

        let shape: (usize, usize) = csr_obj.getattr("shape")?.extract()?;

        // Use astype with copy=False to avoid redundant copies when dtypes match,
        // then borrow via PyReadonlyArray1 for zero-copy slice access.
        let indptr_obj = csr_obj.getattr("indptr")?;
        let indptr_arr = np.call_method1("asarray", (&indptr_obj,))?;
        let indptr_arr = astype_no_copy(py, &indptr_arr, "int64")?;
        let indptr_ro: numpy::PyReadonlyArray1<'_, i64> = indptr_arr.extract()?;
        let indptr_slice = indptr_ro
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let indices_obj = csr_obj.getattr("indices")?;
        let indices_arr = np.call_method1("asarray", (&indices_obj,))?;
        let indices_arr = astype_no_copy(py, &indices_arr, "int32")?;
        let indices_ro: numpy::PyReadonlyArray1<'_, i32> = indices_arr.extract()?;
        let indices_slice = indices_ro
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let data_obj = csr_obj.getattr("data")?;
        let data_arr = np.call_method1("asarray", (&data_obj,))?;
        let data_arr = astype_no_copy(py, &data_arr, "float32")?;
        let data_ro: numpy::PyReadonlyArray1<'_, f32> = data_arr.extract()?;
        let data_slice = data_ro
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        scx_accel::pseudobulk_aggregate_from_slices(
            shape,
            indptr_slice,
            indices_slice,
            data_slice,
            &obs_groups,
            &groupby_columns,
            &gene_names,
            scx_accel::AggregationMethod::Mean,
            min_cells_per_group,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    };

    if result.n_groups == 0 {
        return Err(PyRuntimeError::new_err(
            "no groups passed the min_cells_per_group filter",
        ));
    }

    // Convert to numpy [P, G] float64 array.
    let counts_array = np.call_method1("array", (result.counts.clone(),))?;
    let means_2d = counts_array.call_method1("reshape", ((result.n_groups, result.n_vars),))?;

    // Extract group names (first column of group_labels, since we only have
    // one groupby column).
    let group_names: Vec<String> = result.group_labels.iter().map(|l| l[0].clone()).collect();

    // Return (means, group_names) tuple.
    let tuple = pyo3::types::PyTuple::new(
        py,
        &[
            means_2d.into_any(),
            pyo3::types::PyList::new(py, &group_names)?.into_any(),
        ],
    )?;

    Ok(tuple.into())
}

/// Compute bulk perturbation metrics between real and predicted AnnData objects.
///
/// First computes pseudobulk means for both inputs, then evaluates per-perturbation
/// metrics comparing real vs predicted expression profiles.
///
/// Available metrics:
/// - **pearson_delta**: Pearson correlation of perturbation effects (delta from control)
/// - **mse**: Mean squared error of pseudobulk means
/// - **mae**: Mean absolute error of pseudobulk means
/// - **mse_delta**: MSE of perturbation effects (delta from control)
/// - **mae_delta**: MAE of perturbation effects (delta from control)
///
/// Args:
///     adata_real: AnnData object with real (ground truth) data
///     adata_pred: AnnData object with predicted data
///     pert_col: Column name in obs for perturbation labels (default: "perturbation")
///     control: Label for control perturbation (default: "control")
///     metrics: List of metric names to compute (default: all five)
///     min_cells_per_group: Skip groups with fewer cells (default: 1)
///
/// Returns:
///     dict[str, dict[str, float]] — {metric_name: {perturbation: value}}
///
/// Example:
///     results = pyscx.accel.perturbation_metrics(adata_real, adata_pred)
///     # results["pearson_delta"]["drug_A"] == 0.95
///     # results["mse"]["drug_A"] == 0.12
#[pyfunction]
#[pyo3(signature = (adata_real, adata_pred, pert_col="perturbation", control="control", metrics=None, min_cells_per_group=1))]
#[allow(clippy::too_many_arguments)]
pub fn perturbation_metrics<'py>(
    py: Python<'py>,
    adata_real: &Bound<'py, PyAny>,
    adata_pred: &Bound<'py, PyAny>,
    pert_col: &str,
    control: &str,
    metrics: Option<Vec<String>>,
    min_cells_per_group: usize,
) -> PyResult<PyObject> {
    // Determine which metrics to compute.
    let default_metrics = vec![
        "pearson_delta".to_string(),
        "mse".to_string(),
        "mae".to_string(),
        "mse_delta".to_string(),
        "mae_delta".to_string(),
    ];
    let metric_names = metrics.unwrap_or(default_metrics);

    let bulk_metrics: Vec<scx_accel::BulkMetric> = metric_names
        .iter()
        .map(|name| {
            scx_accel::BulkMetric::parse(name).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown metric '{}'. Valid: pearson_delta, mse, mae, mse_delta, mae_delta",
                    name
                ))
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

    // Use shared helper for pseudobulk computation, alignment, and NaN validation.
    let (means_real_flat, means_pred_flat, common, n_genes, _gene_names) =
        compute_aligned_pseudobulk_means(
            py,
            adata_real,
            adata_pred,
            pert_col,
            control,
            None, // no embed_key for perturbation_metrics
            min_cells_per_group,
        )?;

    let n_perts = common.len();

    // Find control index.
    let ctrl_idx = common.iter().position(|s| s == control).ok_or_else(|| {
        PyValueError::new_err(format!(
            "control '{}' not found in perturbation groups. Ensure control exists and passes min_cells_per_group filter. Available: {:?}",
            control, common
        ))
    })?;

    // Call Rust bulk metrics computation.
    let result = scx_accel::compute_bulk_metrics(
        &means_real_flat,
        &means_pred_flat,
        ctrl_idx,
        n_perts,
        n_genes,
        &common,
        &bulk_metrics,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Convert to dict[str, dict[str, float]].
    let outer_dict = PyDict::new(py);
    for (metric_name, values) in &result.metrics {
        let inner_dict = PyDict::new(py);
        for (i, pert_name) in result.pert_names.iter().enumerate() {
            inner_dict.set_item(pert_name.as_str(), values[i])?;
        }
        outer_dict.set_item(metric_name.as_str(), inner_dict)?;
    }

    Ok(outer_dict.into_any().unbind())
}

/// Compute energy distance between real and predicted perturbation data.
///
/// For each perturbation, computes the energy distance (e-distance) between
/// perturbation cells and control cells on both real and predicted sides,
/// then returns the Pearson correlation of per-perturbation e-distances.
///
/// This is the most expensive cell-eval metric — O(N²) pairwise distances
/// per perturbation. The Rust implementation avoids materializing N×N
/// distance matrices (streaming accumulation), precomputes control
/// self-distances once, and parallelizes across perturbations with rayon.
///
/// Args:
///     adata_real: AnnData with real (ground truth) data
///     adata_pred: AnnData with predicted data
///     pert_col: Column name in obs for perturbation labels (default: "perturbation")
///     control: Label for control perturbation (default: "control")
///     metric: Distance metric — "euclidean" (default), "l1", or "cosine"
///     embed_key: If set, use adata.obsm[embed_key] instead of X (default: None)
///
/// Returns:
///     float — Pearson correlation of per-perturbation e-distances
///
/// Example:
///     corr = pyscx.accel.energy_distance(adata_real, adata_pred)
///     # corr ≈ 0.85 means real and predicted perturbation effects
///     # have similar relative magnitudes
#[pyfunction]
#[pyo3(signature = (adata_real, adata_pred, pert_col="perturbation", control="control", metric="euclidean", embed_key=None))]
#[allow(clippy::too_many_arguments)]
pub fn energy_distance<'py>(
    py: Python<'py>,
    adata_real: &Bound<'py, PyAny>,
    adata_pred: &Bound<'py, PyAny>,
    pert_col: &str,
    control: &str,
    metric: &str,
    embed_key: Option<&str>,
) -> PyResult<f64> {
    let np = py.import("numpy")?;

    // Parse distance metric.
    let dist_metric = match metric.to_lowercase().as_str() {
        "euclidean" | "l2" => scx_accel::DistanceMetric::Euclidean,
        "l1" | "manhattan" | "cityblock" => scx_accel::DistanceMetric::L1,
        "cosine" => scx_accel::DistanceMetric::Cosine,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown metric '{}'. Valid: euclidean, l1, cosine",
                metric
            )))
        }
    };

    // ── Extract dense matrices from both AnnData objects ────────────
    let (real_flat, n_real, n_dims_real) = extract_dense_matrix(py, &np, adata_real, embed_key)?;
    let (pred_flat, n_pred, n_dims_pred) = extract_dense_matrix(py, &np, adata_pred, embed_key)?;

    if n_dims_real != n_dims_pred {
        return Err(PyValueError::new_err(format!(
            "dimension mismatch: real has {} features but pred has {}",
            n_dims_real, n_dims_pred
        )));
    }
    let n_dims = n_dims_real;

    // ── Extract perturbation labels ─────────────────────────────────
    let real_labels = extract_obs_column(py, adata_real, pert_col)?;
    let pred_labels = extract_obs_column(py, adata_pred, pert_col)?;

    if real_labels.len() != n_real {
        return Err(PyValueError::new_err(format!(
            "real adata has {} obs but X has {} rows",
            real_labels.len(),
            n_real
        )));
    }
    if pred_labels.len() != n_pred {
        return Err(PyValueError::new_err(format!(
            "pred adata has {} obs but X has {} rows",
            pred_labels.len(),
            n_pred
        )));
    }

    // ── Build group → index mapping ─────────────────────────────────
    // Collect unique group names from both sides.
    let mut all_groups = std::collections::BTreeSet::new();
    for l in real_labels.iter().chain(pred_labels.iter()) {
        all_groups.insert(l.as_str());
    }
    let group_to_idx: std::collections::HashMap<&str, u32> = all_groups
        .iter()
        .enumerate()
        .map(|(i, &g)| (g, i as u32))
        .collect();

    // Map labels to group indices.
    let real_groups: Vec<u32> = real_labels
        .iter()
        .map(|l| group_to_idx[l.as_str()])
        .collect();
    let pred_groups: Vec<u32> = pred_labels
        .iter()
        .map(|l| group_to_idx[l.as_str()])
        .collect();

    // Identify control group index.
    let ctrl_group_idx = *group_to_idx.get(control).ok_or_else(|| {
        PyValueError::new_err(format!(
            "control '{}' not found in perturbation labels. Available: {:?}",
            control,
            all_groups.iter().collect::<Vec<_>>()
        ))
    })?;

    // Non-control perturbation names and their group indices.
    let mut pert_names = Vec::new();
    let mut pert_group_indices = Vec::new();
    for &g in &all_groups {
        if g != control {
            pert_names.push(g.to_string());
            pert_group_indices.push(group_to_idx[g]);
        }
    }

    if pert_names.is_empty() {
        return Err(PyValueError::new_err(
            "no non-control perturbation groups found",
        ));
    }

    // ── Call Rust energy distance ────────────────────────────────────
    let result = scx_accel::compute_energy_distance(
        &real_flat,
        &pred_flat,
        &real_groups,
        &pred_groups,
        ctrl_group_idx,
        &pert_names,
        &pert_group_indices,
        n_dims,
        dist_metric,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    Ok(result.correlation)
}

/// Compute discrimination score between real and predicted perturbation data.
///
/// For each perturbation, computes how well the predicted perturbation effect
/// ranks among all real perturbation effects by pairwise distance. A score of
/// 1.0 means the correct perturbation is the closest match; 0.0 means it is
/// the furthest.
///
/// This metric builds on pseudobulk means: effects are computed as
/// means[pert] - means[control] for each perturbation.
///
/// When `exclude_target_gene=True` (default) and not using embeddings, the
/// gene column matching each perturbation's name is excluded from the distance
/// computation, preventing trivially high scores from knockdown-gene dominance.
///
/// Args:
///     adata_real: AnnData with real (ground truth) data
///     adata_pred: AnnData with predicted data
///     pert_col: Column name in obs for perturbation labels (default: "perturbation")
///     control: Label for control perturbation (default: "control")
///     metric: Distance metric — "l1" (default), "l2"/"euclidean", or "cosine"
///     exclude_target_gene: Exclude gene named after perturbation (default: True)
///     embed_key: If set, use adata.obsm[embed_key] instead of X (default: None).
///         When set, exclude_target_gene is ignored (gene names don't apply to
///         embeddings). When metric is L1/manhattan/cityblock, embed_key is forced
///         to None (matching cell-eval behavior).
///     min_cells_per_group: Skip groups with fewer cells (default: 1)
///
/// Returns:
///     dict[str, float] — {perturbation_name: normalized_rank_score}
///
/// Example:
///     scores = pyscx.accel.discrimination_score(adata_real, adata_pred)
///     # scores["drug_A"] == 0.95  (high = good prediction)
///     # Three metric variants correspond to cell-eval's:
///     #   discrimination_score_l1 → metric="l1"
///     #   discrimination_score_l2 → metric="l2"
///     #   discrimination_score_cosine → metric="cosine"
#[pyfunction]
#[pyo3(signature = (adata_real, adata_pred, pert_col="perturbation", control="control", metric="l1", exclude_target_gene=true, embed_key=None, min_cells_per_group=1))]
#[allow(clippy::too_many_arguments)]
pub fn discrimination_score<'py>(
    py: Python<'py>,
    adata_real: &Bound<'py, PyAny>,
    adata_pred: &Bound<'py, PyAny>,
    pert_col: &str,
    control: &str,
    metric: &str,
    exclude_target_gene: bool,
    embed_key: Option<&str>,
    min_cells_per_group: usize,
) -> PyResult<PyObject> {
    // Parse distance metric.
    let dist_metric = match metric.to_lowercase().as_str() {
        "euclidean" | "l2" => scx_accel::DistanceMetric::Euclidean,
        "l1" | "manhattan" | "cityblock" => scx_accel::DistanceMetric::L1,
        "cosine" => scx_accel::DistanceMetric::Cosine,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown metric '{}'. Valid: l1, l2, euclidean, cosine",
                metric
            )))
        }
    };

    // Cell-eval behavior: L1/manhattan/cityblock forces embed_key=None
    let effective_embed_key = if matches!(
        metric.to_lowercase().as_str(),
        "l1" | "manhattan" | "cityblock"
    ) {
        None
    } else {
        embed_key
    };

    // Determine if we're using embeddings (affects exclude_target_gene behavior).
    let using_embeddings = effective_embed_key.is_some();

    // ── Compute pseudobulk means for both real and predicted ────────
    let (means_real_flat, means_pred_flat, common, n_genes, gene_names) =
        compute_aligned_pseudobulk_means(
            py,
            adata_real,
            adata_pred,
            pert_col,
            control,
            effective_embed_key,
            min_cells_per_group,
        )?;

    let n_perts = common.len();

    // Find control index.
    let ctrl_idx = common.iter().position(|s| s == control).ok_or_else(|| {
        PyValueError::new_err(format!(
            "control '{}' not found in perturbation groups. Available: {:?}",
            control, common
        ))
    })?;

    // ── Compute perturbation effects: means[p] - means[ctrl] ────────
    let ctrl_real = &means_real_flat[ctrl_idx * n_genes..(ctrl_idx + 1) * n_genes];
    let ctrl_pred = &means_pred_flat[ctrl_idx * n_genes..(ctrl_idx + 1) * n_genes];

    // Build effect matrices (excluding control row).
    let n_output = n_perts - 1;
    let mut real_effects = Vec::with_capacity(n_output * n_genes);
    let mut pred_effects = Vec::with_capacity(n_output * n_genes);
    let mut output_pert_names = Vec::with_capacity(n_output);

    for p in 0..n_perts {
        if p == ctrl_idx {
            continue;
        }
        output_pert_names.push(common[p].clone());
        let row_real = &means_real_flat[p * n_genes..(p + 1) * n_genes];
        let row_pred = &means_pred_flat[p * n_genes..(p + 1) * n_genes];
        for g in 0..n_genes {
            real_effects.push(row_real[g] - ctrl_real[g]);
        }
        for g in 0..n_genes {
            pred_effects.push(row_pred[g] - ctrl_pred[g]);
        }
    }

    // ── Gene exclusion setup ────────────────────────────────────────
    // exclude_target_gene only applies when not using embeddings.
    let effective_exclude = exclude_target_gene && !using_embeddings;
    let gene_names_ref = if effective_exclude {
        Some(gene_names.as_slice())
    } else {
        None
    };

    // ── Call Rust discrimination score ───────────────────────────────
    let result = scx_accel::compute_discrimination_score(
        &real_effects,
        &pred_effects,
        n_output,
        n_genes,
        &output_pert_names,
        gene_names_ref,
        dist_metric,
        effective_exclude,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // ── Convert to dict[str, float] ─────────────────────────────────
    let dict = PyDict::new(py);
    for (i, pert_name) in result.pert_names.iter().enumerate() {
        dict.set_item(pert_name.as_str(), result.scores[i])?;
    }

    Ok(dict.into_any().unbind())
}

/// Shared helper: compute pseudobulk means for both AnnData objects, align
/// them to a common set of perturbations (sorted), and return flat f64 arrays.
///
/// Returns: (means_real_flat, means_pred_flat, common_pert_names, n_genes, gene_names)
#[allow(clippy::type_complexity)]
fn compute_aligned_pseudobulk_means<'py>(
    py: Python<'py>,
    adata_real: &Bound<'py, PyAny>,
    adata_pred: &Bound<'py, PyAny>,
    pert_col: &str,
    control: &str,
    embed_key: Option<&str>,
    min_cells_per_group: usize,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<String>, usize, Vec<String>)> {
    let np = py.import("numpy")?;

    // Validate no NaN in perturbation labels (NaN → "nan" is silent and wrong).
    for (label, adata) in [("real", adata_real), ("pred", adata_pred)] {
        let obs = adata.getattr("obs")?;
        let series = obs.get_item(pert_col).map_err(|_| {
            PyValueError::new_err(format!("column '{}' not found in adata.obs", pert_col))
        })?;
        let pd = py.import("pandas")?;
        let isna = pd.call_method1("isna", (&series,))?;
        let any_na: bool = isna.call_method0("any")?.extract()?;
        if any_na {
            let n_na: usize = isna.call_method0("sum")?.extract()?;
            return Err(PyValueError::new_err(format!(
                "{label} adata.obs['{}'] has {} NaN values. Remove or fill NaN before calling.",
                pert_col, n_na
            )));
        }
    }

    // For embed_key: use obsm-based pseudobulk (compute manually).
    // For X: use the existing pseudobulk_means infrastructure.
    let (means_real_np, groups_real, means_pred_np, groups_pred, gene_names) =
        if let Some(key) = embed_key {
            // Extract obsm embeddings and compute group means manually
            let (means_r, groups_r) =
                compute_obsm_pseudobulk(py, &np, adata_real, pert_col, key, min_cells_per_group)?;
            let (means_p, groups_p) =
                compute_obsm_pseudobulk(py, &np, adata_pred, pert_col, key, min_cells_per_group)?;
            // No gene names when using embeddings
            let n_dims: usize = means_r.getattr("shape")?.extract::<(usize, usize)>()?.1;
            let empty_genes: Vec<String> = (0..n_dims).map(|i| format!("embed_{i}")).collect();
            (means_r, groups_r, means_p, groups_p, empty_genes)
        } else {
            // Use standard X-based pseudobulk means
            let means_real_obj = pseudobulk_means(py, adata_real, pert_col, min_cells_per_group)?;
            let means_pred_obj = pseudobulk_means(py, adata_pred, pert_col, min_cells_per_group)?;

            let real_tuple = means_real_obj.bind(py);
            let pred_tuple = means_pred_obj.bind(py);

            let means_r = real_tuple.get_item(0)?;
            let groups_r: Vec<String> = real_tuple.get_item(1)?.extract()?;
            let means_p = pred_tuple.get_item(0)?;
            let groups_p: Vec<String> = pred_tuple.get_item(1)?.extract()?;

            // Extract gene names from adata_real.var_names
            let var = adata_real.getattr("var")?;
            let var_names = var.getattr("index")?;
            let gene_names: Vec<String> = var_names.call_method0("tolist")?.extract()?;

            (means_r, groups_r, means_p, groups_p, gene_names)
        };

    // ── Align perturbation groups ───────────────────────────────────
    if groups_real.is_empty() || groups_pred.is_empty() {
        return Err(PyValueError::new_err(
            "no groups passed the min_cells_per_group filter",
        ));
    }

    let real_set: std::collections::HashSet<&str> =
        groups_real.iter().map(|s| s.as_str()).collect();
    let pred_set: std::collections::HashSet<&str> =
        groups_pred.iter().map(|s| s.as_str()).collect();

    let mut common: Vec<String> = real_set
        .intersection(&pred_set)
        .map(|s| s.to_string())
        .collect();
    common.sort();

    if common.is_empty() {
        return Err(PyValueError::new_err(
            "no common perturbation groups between real and predicted",
        ));
    }

    // Ensure control is in the common set
    if !common.contains(&control.to_string()) {
        return Err(PyValueError::new_err(format!(
            "control '{}' not found in common perturbation groups. Available: {:?}",
            control, common
        )));
    }

    // Reorder both matrices to common ordering
    let real_idx_map: std::collections::HashMap<&str, usize> = groups_real
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let pred_idx_map: std::collections::HashMap<&str, usize> = groups_pred
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    let real_indices: Vec<usize> = common.iter().map(|s| real_idx_map[s.as_str()]).collect();
    let pred_indices: Vec<usize> = common.iter().map(|s| pred_idx_map[s.as_str()]).collect();

    let real_idx_arr = np.call_method1("array", (real_indices,))?;
    let pred_idx_arr = np.call_method1("array", (pred_indices,))?;

    let means_real_ordered = means_real_np.get_item(&real_idx_arr)?;
    let means_pred_ordered = means_pred_np.get_item(&pred_idx_arr)?;

    let shape: (usize, usize) = means_real_ordered.getattr("shape")?.extract()?;
    let n_genes = shape.1;

    let means_real_flat: Vec<f64> = means_real_ordered
        .call_method0("ravel")?
        .call_method1("astype", ("float64",))?
        .extract()?;
    let means_pred_flat: Vec<f64> = means_pred_ordered
        .call_method0("ravel")?
        .call_method1("astype", ("float64",))?
        .extract()?;

    Ok((
        means_real_flat,
        means_pred_flat,
        common,
        n_genes,
        gene_names,
    ))
}

/// Compute pseudobulk means from adata.obsm[embed_key] using numpy group-by.
fn compute_obsm_pseudobulk<'py>(
    py: Python<'py>,
    np: &Bound<'py, PyModule>,
    adata: &Bound<'py, PyAny>,
    pert_col: &str,
    embed_key: &str,
    min_cells_per_group: usize,
) -> PyResult<(Bound<'py, PyAny>, Vec<String>)> {
    let obsm = adata.getattr("obsm")?;
    let embeddings = obsm.get_item(embed_key).map_err(|_| {
        PyValueError::new_err(format!("embed_key '{}' not found in adata.obsm", embed_key))
    })?;
    let matrix = np
        .call_method1("asarray", (&embeddings,))?
        .call_method1("astype", ("float64",))?;
    let shape: (usize, usize) = matrix.getattr("shape")?.extract()?;
    let n_obs = shape.0;
    let n_dims = shape.1;

    // Warn for large materializations (consistent with extract_dense_matrix).
    let n_bytes = n_obs * n_dims * 8; // f64 = 8 bytes
    if n_bytes > 500_000_000 {
        let mb = n_bytes / (1024 * 1024);
        eprintln!(
            "[pyscx] warning: materializing obsm['{embed_key}'] ({n_obs}×{n_dims}) \
             into {mb} MB of memory"
        );
    }

    let labels = extract_obs_column(py, adata, pert_col)?;
    if labels.len() != n_obs {
        return Err(PyValueError::new_err(format!(
            "obs has {} rows but obsm['{embed_key}'] has {n_obs} rows",
            labels.len()
        )));
    }

    // Group by label and compute mean.
    // BTreeMap guarantees sorted iteration over keys, producing deterministic
    // group ordering. The downstream sort in compute_aligned_pseudobulk_means
    // is a harmless no-op but kept for defensive correctness.
    let mut group_map: std::collections::BTreeMap<String, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, label) in labels.iter().enumerate() {
        group_map.entry(label.clone()).or_default().push(i);
    }

    // Filter by min_cells_per_group
    let groups: Vec<(String, Vec<usize>)> = group_map
        .into_iter()
        .filter(|(_, indices)| indices.len() >= min_cells_per_group)
        .collect();

    if groups.is_empty() {
        return Err(PyRuntimeError::new_err(
            "no groups passed the min_cells_per_group filter",
        ));
    }

    let n_groups = groups.len();
    let mut means_data = vec![0.0f64; n_groups * n_dims];
    let matrix_flat: Vec<f64> = matrix.call_method0("ravel")?.extract()?;

    for (g, (_, indices)) in groups.iter().enumerate() {
        let n = indices.len() as f64;
        for &i in indices {
            for d in 0..n_dims {
                means_data[g * n_dims + d] += matrix_flat[i * n_dims + d];
            }
        }
        for d in 0..n_dims {
            means_data[g * n_dims + d] /= n;
        }
    }

    let group_names: Vec<String> = groups.iter().map(|(name, _)| name.clone()).collect();

    let means_arr = np.call_method1("array", (means_data,))?;
    let means_2d = means_arr.call_method1("reshape", ((n_groups, n_dims),))?;

    Ok((means_2d, group_names))
}

/// Extract a dense `[N, D]` matrix from adata.X (or adata.obsm[embed_key])
/// as a flat `Vec<f64>`.
///
/// Handles scipy sparse (converts to dense), numpy arrays, and SCX backed types.
/// Emits a Python warning for large materializations (>2 GB estimated).
fn extract_dense_matrix<'py>(
    py: Python<'py>,
    np: &Bound<'py, PyModule>,
    adata: &Bound<'py, PyAny>,
    embed_key: Option<&str>,
) -> PyResult<(Vec<f64>, usize, usize)> {
    let matrix_obj = if let Some(key) = embed_key {
        let obsm = adata.getattr("obsm")?;
        obsm.get_item(key).map_err(|_| {
            PyValueError::new_err(format!("embed_key '{}' not found in adata.obsm", key))
        })?
    } else {
        adata.getattr("X")?
    };

    // Check if it's a SCX backed or lazy-transformed type — must materialize.
    let dense = if matrix_obj.is_instance_of::<ScxBackedSparseDataset>()
        || matrix_obj.is_instance_of::<ScxLazyTransformedDataset>()
    {
        let arr = matrix_obj.call_method0("toarray")?;
        arr.call_method1("astype", ("float64",))?
    } else {
        // Check for scipy sparse.
        let scipy_sparse = py.import("scipy.sparse")?;
        let is_sparse: bool = scipy_sparse
            .call_method1("issparse", (&matrix_obj,))?
            .extract()?;

        if is_sparse {
            let arr = matrix_obj.call_method0("toarray")?;
            arr.call_method1("astype", ("float64",))?
        } else {
            np.call_method1("asarray", (&matrix_obj,))?
                .call_method1("astype", ("float64",))?
        }
    };

    let shape: (usize, usize) = dense.getattr("shape")?.extract()?;

    // Warn if the dense matrix is very large (> 2 GB).
    let estimated_bytes = shape.0 * shape.1 * 8; // f64 = 8 bytes
    if estimated_bytes > 2_000_000_000 {
        let gb = estimated_bytes as f64 / 1e9;
        let warnings = py.import("warnings")?;
        warnings.call_method1(
            "warn",
            (format!(
                "energy_distance: materializing a {:.1} GB dense matrix ({} × {} × 8 bytes). \
                 Consider using embed_key='X_pca' or subsetting the data.",
                gb, shape.0, shape.1
            ),),
        )?;
    }

    let flat: Vec<f64> = dense.call_method0("ravel")?.extract()?;

    Ok((flat, shape.0, shape.1))
}

/// Extract a column from adata.obs as Vec<String>.
///
/// Raises ValueError if the column contains NaN values (which would
/// silently become the string `"nan"` after `.astype(str)`).
fn extract_obs_column<'py>(
    _py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    col: &str,
) -> PyResult<Vec<String>> {
    let obs = adata.getattr("obs")?;
    let series = obs
        .get_item(col)
        .map_err(|_| PyValueError::new_err(format!("column '{}' not found in adata.obs", col)))?;

    // Detect NaN values before string conversion (NaN → "nan" is silent and wrong).
    let pd = _py.import("pandas")?;
    let isna = pd.call_method1("isna", (&series,))?;
    let any_na: bool = isna.call_method0("any")?.extract()?;
    if any_na {
        let n_na: usize = isna.call_method0("sum")?.extract()?;
        return Err(PyValueError::new_err(format!(
            "column '{}' has {} NaN values. Remove or fill NaN before calling.",
            col, n_na
        )));
    }

    let labels: Vec<String> = series
        .call_method1("astype", ("str",))?
        .call_method0("tolist")?
        .extract()?;
    Ok(labels)
}

/// Compute per-cell knockdown efficiency and log deviation.
///
/// For each perturbed cell, this function measures how effectively the
/// perturbation knocked down its target gene. Perturbation names must match
/// gene names (var_names) for the lookup to work.
///
/// **Knockdown efficiency** (computed on normalized, NOT log-transformed data):
///     KD = 1 - x_target / (μ_control[target_gene] + eps)
///
/// **Log fold change** (computed on log1p-transformed data):
///     FC = x_log[target_gene] - log1p(μ_control[target_gene])
///
/// Control cells and cells whose perturbation name doesn't match any gene
/// will have NaN in both output columns.
///
/// This function operates in two passes matching arc-bench's pipeline order:
/// 1. Compute KD on raw normalized data (before log1p)
/// 2. Apply log1p, then compute log deviation
///
/// For data already in log-space (e.g., adata already log1p-transformed),
/// only the log deviation is meaningful. The efficiency column will still be
/// computed but may not be physically meaningful on log-space values.
///
/// Args:
///     adata: AnnData object with sparse or dense X matrix.
///         Must have obs[pert_col] with perturbation labels where perturbation
///         names match gene names (var_names).
///     pert_col: Column name in adata.obs for perturbation labels (default: "perturbation")
///     control: Label for control perturbation (default: "control")
///     eps: Small value for numerical stability (default: 1e-8)
///
/// Returns:
///     None — writes two columns to adata.obs:
///     - "KnockDownEfficiency": per-cell knockdown efficiency (float32)
///     - "KnockDownGeneFC": per-cell log fold change (float32)
///
/// Example:
///     pyscx.accel.knockdown_efficiency(adata, pert_col="perturbation")
///     adata.obs["KnockDownEfficiency"]  # per-cell KD scores
///     adata.obs["KnockDownGeneFC"]      # per-cell log FC
#[pyfunction]
#[pyo3(signature = (adata, pert_col="perturbation", control="control", eps=1e-8))]
pub fn knockdown_efficiency<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    pert_col: &str,
    control: &str,
    eps: f64,
) -> PyResult<()> {
    let np = py.import("numpy")?;

    // ── Extract perturbation labels ─────────────────────────────────
    let pert_labels = extract_obs_column(py, adata, pert_col)?;
    let n_obs = pert_labels.len();

    // ── Extract gene names ──────────────────────────────────────────
    let var = adata.getattr("var")?;
    let var_names = var.getattr("index")?;
    let gene_names: Vec<String> = var_names.call_method0("tolist")?.extract()?;
    let n_vars = gene_names.len();

    // ── Extract CSR data from adata.X ───────────────────────────────
    let x = adata.getattr("X")?;
    let scipy_sparse = py.import("scipy.sparse")?;

    // Get CSR matrix — handle backed, lazy-transformed, sparse, and dense inputs.
    // TODO: For backed SCX, consider streaming single-column extraction per shard
    // instead of materializing the full matrix — the knockdown metric only needs
    // one gene column per perturbation, making full materialization wasteful at scale.
    let csr_obj = if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        // Backed SCX: materialize to scipy CSR for column access
        drop(backed);
        let arr = x.call_method0("toarray")?;
        scipy_sparse.call_method1("csr_matrix", (&arr,))?
    } else if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
        // Lazy-transformed: materialize through transforms
        let scipy_csr = lazy.to_memory_py(py)?;
        drop(lazy);
        scipy_csr
    } else {
        let is_sparse: bool = scipy_sparse.call_method1("issparse", (&x,))?.extract()?;
        if is_sparse {
            scipy_sparse.call_method1("csr_matrix", (&x,))?
        } else {
            let arr = np
                .call_method1("asarray", (&x,))?
                .call_method1("astype", ("float32",))?;
            scipy_sparse.call_method1("csr_matrix", (&arr,))?
        }
    };

    let shape: (usize, usize) = csr_obj.getattr("shape")?.extract()?;
    if shape.0 != n_obs {
        return Err(PyValueError::new_err(format!(
            "X has {} rows but obs has {} rows",
            shape.0, n_obs
        )));
    }
    if shape.1 != n_vars {
        return Err(PyValueError::new_err(format!(
            "X has {} columns but var has {} genes",
            shape.1, n_vars
        )));
    }

    // ── Extract CSR arrays as zero-copy slices ──────────────────────
    let slices = extract_csr_slices(py, &np, &csr_obj, "knockdown_efficiency", 1)?;

    // ── Compute control baseline ────────────────────────────────────
    let baseline = scx_accel::compute_control_baseline(
        slices.indptr(),
        slices.indices(),
        slices.data(),
        &pert_labels,
        control,
        n_vars,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // ── Compute knockdown efficiency (on current data) ──────────────
    let efficiency = scx_accel::compute_knockdown_efficiency(
        slices.indptr(),
        slices.indices(),
        slices.data(),
        &pert_labels,
        control,
        &gene_names,
        &baseline,
        eps,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // ── Compute log deviation ───────────────────────────────────────
    // Arc-bench computes log deviation AFTER log1p on the data.
    // We apply log1p to the data values (on a copy) and log1p to baseline.
    let baseline_log: Vec<f64> = baseline.iter().map(|&v| v.ln_1p()).collect();

    // Apply log1p to the CSR data values for the log deviation pass.
    // f32 → f64 → ln_1p → f32: intentional double promotion for accuracy.
    let data_log: Vec<f32> = slices
        .data()
        .iter()
        .map(|&v| (v as f64).ln_1p() as f32)
        .collect();

    let log_fc = scx_accel::compute_log_deviation(
        slices.indptr(),
        slices.indices(),
        &data_log,
        &pert_labels,
        control,
        &gene_names,
        &baseline_log,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // ── Write results to adata.obs ──────────────────────────────────
    let obs = adata.getattr("obs")?;
    let eff_array = numpy::PyArray::from_vec(py, efficiency);
    obs.set_item("KnockDownEfficiency", eff_array)?;

    let fc_array = numpy::PyArray::from_vec(py, log_fc);
    obs.set_item("KnockDownGeneFC", fc_array)?;

    Ok(())
}

/// Compute clustering agreement between real and predicted perturbation centroids.
///
/// Builds centroid matrices (pseudobulk means per perturbation, excluding
/// control), constructs kNN graphs, clusters via Leiden at multiple resolutions,
/// and scores the agreement between real and predicted cluster assignments
/// using AMI, NMI, or ARI.
///
/// This metric evaluates whether predicted perturbation effects preserve the
/// cluster structure of real perturbation effects. It chains existing SCX
/// accelerators (pseudobulk means) with scanpy's neighbors/Leiden (for the
/// small centroid matrices, typically 50–200 rows) and Rust-native AMI/NMI/ARI
/// scoring.
///
/// Args:
///     adata_real: AnnData with real (ground truth) data
///     adata_pred: AnnData with predicted data
///     pert_col: Column name in obs for perturbation labels (default: "perturbation")
///     control: Label for control perturbation (default: "control")
///     metric: Agreement metric — "ami" (default), "nmi", or "ari"
///     real_resolution: Leiden resolution for real centroids (default: 1.0)
///     pred_resolutions: Tuple of Leiden resolutions to sweep for predicted
///         centroids (default: (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0))
///     n_neighbors: Number of neighbors for kNN graph (default: 15)
///     embed_key: If set, use adata.obsm[embed_key] instead of X (default: None)
///     min_cells_per_group: Skip groups with fewer cells (default: 1)
///
/// Returns:
///     float — Best clustering agreement score across predicted resolutions
///
/// Example:
///     score = pyscx.accel.clustering_agreement(adata_real, adata_pred)
///     # score ≈ 0.7 means good cluster structure preservation
///     #
///     # Metric variants correspond to cell-eval's ClusteringAgreement:
///     #   metric="ami" → adjusted_mutual_info_score
///     #   metric="nmi" → normalized_mutual_info_score
///     #   metric="ari" → (adjusted_rand_score + 1) / 2
#[pyfunction]
#[pyo3(signature = (adata_real, adata_pred, pert_col="perturbation", control="control", metric="ami", real_resolution=1.0, pred_resolutions=None, n_neighbors=15, embed_key=None, min_cells_per_group=1))]
#[allow(clippy::too_many_arguments)]
pub fn clustering_agreement<'py>(
    py: Python<'py>,
    adata_real: &Bound<'py, PyAny>,
    adata_pred: &Bound<'py, PyAny>,
    pert_col: &str,
    control: &str,
    metric: &str,
    real_resolution: f64,
    pred_resolutions: Option<Vec<f64>>,
    n_neighbors: usize,
    embed_key: Option<&str>,
    min_cells_per_group: usize,
) -> PyResult<f64> {
    let np = py.import("numpy")?;
    let sc = py.import("scanpy")?;
    let ad_mod = py.import("anndata")?;

    // Parse clustering metric.
    let clustering_metric = scx_accel::ClusteringMetric::parse(metric).ok_or_else(|| {
        PyValueError::new_err(format!("unknown metric '{}'. Valid: ami, nmi, ari", metric))
    })?;

    let default_resolutions = vec![0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0];
    let resolutions = pred_resolutions.unwrap_or(default_resolutions);

    if resolutions.is_empty() {
        return Err(PyValueError::new_err("pred_resolutions must not be empty"));
    }

    // ── Compute pseudobulk means (centroids) for both sides ─────────
    let (means_real_flat, means_pred_flat, common, n_genes, _gene_names) =
        compute_aligned_pseudobulk_means(
            py,
            adata_real,
            adata_pred,
            pert_col,
            control,
            embed_key,
            min_cells_per_group,
        )?;

    let n_perts = common.len();

    // Find control index and filter it out.
    let ctrl_idx = common.iter().position(|s| s == control);

    // Build non-control perturbation names and centroid matrices.
    let mut pert_names: Vec<String> = Vec::with_capacity(n_perts);
    let mut centroids_real: Vec<f64> = Vec::with_capacity(n_perts * n_genes);
    let mut centroids_pred: Vec<f64> = Vec::with_capacity(n_perts * n_genes);

    for p in 0..n_perts {
        if Some(p) == ctrl_idx {
            continue;
        }
        pert_names.push(common[p].clone());
        centroids_real.extend_from_slice(&means_real_flat[p * n_genes..(p + 1) * n_genes]);
        centroids_pred.extend_from_slice(&means_pred_flat[p * n_genes..(p + 1) * n_genes]);
    }

    let n_output = pert_names.len();
    if n_output < 2 {
        return Err(PyValueError::new_err(format!(
            "need at least 2 non-control perturbations for clustering agreement, got {}",
            n_output
        )));
    }

    // Sort centroids by perturbation name to align between real and pred.
    // We create sorted indices and reorder both centroid matrices.
    let mut sorted_indices: Vec<usize> = (0..n_output).collect();
    sorted_indices.sort_by(|&a, &b| pert_names[a].cmp(&pert_names[b]));

    let mut sorted_real = vec![0.0f64; n_output * n_genes];
    let mut sorted_pred = vec![0.0f64; n_output * n_genes];

    for (new_idx, &old_idx) in sorted_indices.iter().enumerate() {
        sorted_real[new_idx * n_genes..(new_idx + 1) * n_genes]
            .copy_from_slice(&centroids_real[old_idx * n_genes..(old_idx + 1) * n_genes]);
        sorted_pred[new_idx * n_genes..(new_idx + 1) * n_genes]
            .copy_from_slice(&centroids_pred[old_idx * n_genes..(old_idx + 1) * n_genes]);
    }

    // ── Build AnnData centroid objects for scanpy ────────────────────
    let real_arr = np.call_method1("array", (sorted_real,))?;
    let real_2d = real_arr.call_method1("reshape", ((n_output, n_genes),))?;
    let real_2d_f64 = real_2d.call_method1("astype", ("float64",))?;
    let ad_real_cent = ad_mod.call_method(
        "AnnData",
        (),
        Some(&{
            let kw = PyDict::new(py);
            kw.set_item("X", &real_2d_f64)?;
            kw
        }),
    )?;

    let pred_arr = np.call_method1("array", (sorted_pred,))?;
    let pred_2d = pred_arr.call_method1("reshape", ((n_output, n_genes),))?;
    let pred_2d_f64 = pred_2d.call_method1("astype", ("float64",))?;
    let ad_pred_cent = ad_mod.call_method(
        "AnnData",
        (),
        Some(&{
            let kw = PyDict::new(py);
            kw.set_item("X", &pred_2d_f64)?;
            kw
        }),
    )?;

    // ── Build kNN graphs and cluster ────────────────────────────────
    let effective_n_neighbors = n_neighbors.min(n_output - 1);

    // Build kNN graph + Leiden for real centroids.
    let sc_pp = sc.getattr("pp")?;
    let sc_tl = sc.getattr("tl")?;

    sc_pp.call_method(
        "neighbors",
        (&ad_real_cent,),
        Some(&{
            let kw = PyDict::new(py);
            kw.set_item("n_neighbors", effective_n_neighbors)?;
            kw.set_item("use_rep", "X")?;
            kw
        }),
    )?;

    sc_tl.call_method(
        "leiden",
        (&ad_real_cent,),
        Some(&{
            let kw = PyDict::new(py);
            kw.set_item("resolution", real_resolution)?;
            kw.set_item("key_added", "real_clusters")?;
            kw.set_item("flavor", "igraph")?;
            kw.set_item("n_iterations", 2)?;
            kw
        }),
    )?;

    // Extract real cluster labels as u32 array.
    let real_obs = ad_real_cent.getattr("obs")?;
    let real_labels_series = real_obs.get_item("real_clusters")?;
    let real_label_codes: Vec<i64> = real_labels_series
        .getattr("cat")?
        .getattr("codes")?
        .call_method0("tolist")?
        .extract()?;
    // Validate and convert label codes. Pandas categorical codes use -1 for
    // missing/NA values, which would silently become u32::MAX.
    if real_label_codes.iter().any(|&c| c < 0) {
        return Err(PyRuntimeError::new_err(
            "Leiden produced NA cluster labels for real centroids",
        ));
    }
    let real_labels_u32: Vec<u32> = real_label_codes.iter().map(|&c| c as u32).collect();

    // Build kNN graph for predicted centroids (reusable across resolutions).
    sc_pp.call_method(
        "neighbors",
        (&ad_pred_cent,),
        Some(&{
            let kw = PyDict::new(py);
            kw.set_item("n_neighbors", effective_n_neighbors)?;
            kw.set_item("use_rep", "X")?;
            kw
        }),
    )?;

    // ── Sweep predicted resolutions and compute best score ──────────
    // Initialize to NEG_INFINITY so that even if all scores are negative
    // (possible with AMI), we return an actual computed value.
    let mut best_score = f64::NEG_INFINITY;

    for &r in &resolutions {
        let pred_key = format!("pred_clusters_{}", r);

        sc_tl.call_method(
            "leiden",
            (&ad_pred_cent,),
            Some(&{
                let kw = PyDict::new(py);
                kw.set_item("resolution", r)?;
                kw.set_item("key_added", pred_key.as_str())?;
                kw.set_item("flavor", "igraph")?;
                kw.set_item("n_iterations", 2)?;
                kw
            }),
        )?;

        // Extract predicted cluster labels.
        let pred_obs = ad_pred_cent.getattr("obs")?;
        let pred_labels_series = pred_obs.get_item(pred_key.as_str())?;
        let pred_label_codes: Vec<i64> = pred_labels_series
            .getattr("cat")?
            .getattr("codes")?
            .call_method0("tolist")?
            .extract()?;
        // Validate and convert predicted label codes.
        if pred_label_codes.iter().any(|&c| c < 0) {
            return Err(PyRuntimeError::new_err(format!(
                "Leiden produced NA cluster labels for predicted centroids at resolution {}",
                r
            )));
        }
        let pred_labels_u32: Vec<u32> = pred_label_codes.iter().map(|&c| c as u32).collect();

        // Compute scoring metric in Rust.
        let score = clustering_metric.score(&real_labels_u32, &pred_labels_u32);
        if score > best_score {
            best_score = score;
        }
    }

    Ok(best_score)
}
