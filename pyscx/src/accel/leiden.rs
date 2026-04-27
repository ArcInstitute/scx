//! Leiden community detection — Rust-native (CPU) and cuGraph (GPU) as peers.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::gpu::resolve_device;

/// Run Leiden community detection on a kNN graph.
///
/// Reads `adata.obsp["connectivities"]` (from `pyscx.accel.neighbors()` or
/// `sc.pp.neighbors()`) and partitions the graph using the Leiden algorithm.
/// Results are written to `adata.obs[key_added]` (cluster labels as strings)
/// and `adata.uns[key_added]` (parameters and backend metadata).
///
/// ## Backends (selected by `device`)
///
/// Two backends are exposed as peers — `device` is authoritative; there is
/// **no** silent cross-backend fallback.
///
/// * `device="cpu"` → Rust-native (`scx_accel::leiden`). Honors `parallel`;
///   ignores `theta` (with `UserWarning` if non-default).
/// * `device="gpu"` (or `"gpu:N"`) → cuGraph (`cugraph.leiden`). Honors
///   `theta`; ignores `parallel` (with `UserWarning` if `True`). `gpu:N`
///   pins the call to CUDA device `N` via `cupy.cuda.Device(N)`. Bare
///   `device="gpu"` is equivalent to `device="gpu:0"`. A host without
///   cuGraph raises `RuntimeError` — no fallback.
/// * `device="auto"` (default) → cuGraph if a CUDA device is visible and
///   `cugraph` imports cleanly, else Rust-native. Matches the rest of
///   `pyscx.accel.*`. **Note**: on a host with cuGraph installed, the
///   default partition now comes from cuGraph (ARI ≈ 0.92 vs leidenalg)
///   instead of Rust-native (ARI ≈ 0.97). Pin `device="cpu"` to preserve
///   reproducibility of pre-spec cluster IDs.
///
/// Args:
///     adata: AnnData with obsp["connectivities"] (CSR, n_obs × n_obs)
///     resolution: Resolution parameter controlling cluster granularity (default: 1.0)
///     key_added: Column name in adata.obs for cluster labels (default: "leiden")
///     random_state: Random seed for reproducibility (default: 0)
///     n_iterations: Maximum optimization iterations; 2 runs two outer passes
///         (matching leidenalg package default), -1 for until convergence (default: 2).
///         On the cuGraph path this can safely be raised — rapids-singlecell
///         defaults to 100 and convergence is cheap; see docs/scanpy.md.
///     device: Device selection — "auto" (default), "cpu", "gpu", or "gpu:N".
///     parallel: Run the Rust-native Leiden in parallel mode (conflict-free
///         graph coloring). Default `False`. **Ignored on the cuGraph path**
///         (warns).
///     theta: Resolution scaling knob for cuGraph Leiden only (default 1.0).
///         **Ignored on the Rust-native path** (warns when non-default).
///
/// Notes:
///     The two backends produce different partitions on the same graph
///     (cuGraph ARI ≈ 0.92 vs leidenalg; Rust-native ARI ≈ 0.97). Both
///     produce valid, high-quality community structures. Pin `device="cpu"`
///     when downstream analysis (DE, annotation transfer, UMAP coloring) is
///     keyed on specific cluster IDs and reproducibility matters.
#[pyfunction]
#[pyo3(signature = (adata, resolution=1.0, key_added="leiden", random_state=0, n_iterations=2, device="auto", parallel=false, theta=1.0))]
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
    theta: f64,
) -> PyResult<()> {
    let resolved = resolve_device(device)?;
    let use_gpu = resolved.is_gpu();

    // Surface ignored kwargs so the user notices when a knob silently does
    // nothing on the chosen backend. Fires on the *resolved* backend, so
    // `device="auto"` that ends up on GPU still warns about `parallel=True`.
    warn_if_ignored_theta(py, theta, use_gpu)?;
    warn_if_ignored_parallel(py, parallel, use_gpu)?;

    // Extract connectivities CSR from adata.obsp["connectivities"]
    let obsp = adata.getattr("obsp")?;
    let conn = obsp.get_item("connectivities").map_err(|_| {
        PyRuntimeError::new_err(
            "'connectivities' not found in adata.obsp. Run neighbors first: \
             pyscx.accel.neighbors(adata) or sc.pp.neighbors(adata)",
        )
    })?;

    #[cfg(feature = "gpu")]
    if let Some(gpu_id) = resolved.gpu_id() {
        return try_cugraph_leiden(
            py,
            adata,
            &conn,
            resolution,
            key_added,
            random_state,
            n_iterations,
            theta,
            parallel,
            device,
            gpu_id,
        );
    }

    run_rust_leiden(
        py,
        adata,
        &conn,
        resolution,
        key_added,
        random_state,
        n_iterations,
        parallel,
        theta,
        device,
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
    theta: f64,
    device: &str,
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
    params_dict.set_item("device", device)?;
    params_dict.set_item("parallel", parallel)?;
    let ignored = if (theta - 1.0).abs() > f64::EPSILON {
        pyo3::types::PyList::new(py, ["theta"])?
    } else {
        pyo3::types::PyList::empty(py)
    };
    params_dict.set_item("ignored", ignored)?;
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
/// Pins all cuPy / cuDF / cuGraph allocations to CUDA device `gpu_id` for
/// the duration of the call via `cupy.cuda.Device(gpu_id)` (the canonical
/// RAPIDS pattern — cuGraph allocates on the active cuPy device). Converts
/// the connectivities CSR to a cuGraph Graph, runs `cugraph.leiden()`, and
/// writes results to adata. `theta` is forwarded as the cuGraph-specific
/// resolution scaling kwarg.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn try_cugraph_leiden(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    conn: &Bound<'_, PyAny>,
    resolution: f64,
    key_added: &str,
    random_state: u64,
    max_iter: i64,
    theta: f64,
    parallel: bool,
    device: &str,
    gpu_id: usize,
) -> PyResult<()> {
    // Import cuGraph — if not installed, return error immediately. No fallback.
    let cugraph = py.import("cugraph").map_err(|_| {
        PyRuntimeError::new_err(
            "cugraph not available. Install for GPU Leiden: \
             conda install -c rapidsai -c conda-forge cugraph",
        )
    })?;

    // Pin cuPy / cuDF / cuGraph allocations to GPU `gpu_id`. cuPy ships with
    // every RAPIDS install, so this never fails when cugraph imported.
    let cupy_cuda = py.import("cupy.cuda").map_err(|e| {
        PyRuntimeError::new_err(format!(
            "cupy.cuda not importable (required to honor device='{device}'): {e}"
        ))
    })?;
    let device_ctx = cupy_cuda.getattr("Device")?.call1((gpu_id,))?;
    device_ctx.call_method0("__enter__")?;

    // Body factored into a closure so __exit__ runs even when the body
    // errors (Rust has no try/finally; this matches Python `with` semantics).
    let body_result: PyResult<()> = (|| {
        let numpy = py.import("numpy")?;
        let pd = py.import("pandas")?;

        // Extract COO from connectivities CSR: cuGraph works with edge lists
        let scipy_sparse = py.import("scipy.sparse")?;
        let coo = scipy_sparse
            .call_method1("triu", (conn,))?
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
        leiden_kwargs.set_item("theta", theta)?;
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

        // Write metadata to adata.uns[key_added]
        let leiden_dict = PyDict::new(py);
        let params_dict = PyDict::new(py);
        params_dict.set_item("resolution", resolution)?;
        params_dict.set_item("random_state", random_state)?;
        params_dict.set_item("n_iterations", max_iter)?;
        params_dict.set_item("theta", theta)?;
        params_dict.set_item("device", device)?;
        params_dict.set_item("gpu_id", gpu_id)?;
        let ignored = if parallel {
            pyo3::types::PyList::new(py, ["parallel"])?
        } else {
            pyo3::types::PyList::empty(py)
        };
        params_dict.set_item("ignored", ignored)?;
        leiden_dict.set_item("params", params_dict)?;
        leiden_dict.set_item("backend", "cugraph")?;
        leiden_dict.set_item("modularity", modularity)?;

        let uns = adata.getattr("uns")?;
        uns.set_item(key_added, leiden_dict)?;

        Ok(())
    })();

    // Always run __exit__, regardless of body outcome. If both error, the
    // body's error wins (matches Python `with` semantics).
    let exit_result = device_ctx.call_method1("__exit__", (py.None(), py.None(), py.None()));
    body_result?;
    exit_result?;
    Ok(())
}

/// Emit a `UserWarning` when `theta` is non-default but the resolved backend
/// (Rust-native) ignores it.
fn warn_if_ignored_theta(py: Python<'_>, theta: f64, use_gpu: bool) -> PyResult<()> {
    if !use_gpu && (theta - 1.0).abs() > f64::EPSILON {
        let warnings = py.import("warnings")?;
        let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
        warnings.call_method1(
            "warn",
            (
                format!("`theta={theta}` is cuGraph-only and is ignored on the Rust-native path."),
                category,
            ),
        )?;
    }
    Ok(())
}

/// Emit a `UserWarning` when `parallel=True` but the resolved backend
/// (cuGraph) ignores it.
fn warn_if_ignored_parallel(py: Python<'_>, parallel: bool, use_gpu: bool) -> PyResult<()> {
    if use_gpu && parallel {
        let warnings = py.import("warnings")?;
        let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
        warnings.call_method1(
            "warn",
            (
                "`parallel=True` only affects the Rust-native path and is ignored on the cuGraph path.",
                category,
            ),
        )?;
    }
    Ok(())
}
