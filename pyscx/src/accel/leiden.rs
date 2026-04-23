//! Leiden community detection — Rust-native, cuGraph GPU, leidenalg CPU fallback.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::gpu::resolve_device;

/// Run Leiden community detection on a kNN graph.
///
/// Reads `adata.obsp["connectivities"]` (from `pyscx.accel.neighbors()` or
/// `sc.pp.neighbors()`) and partitions the graph using the Leiden algorithm.
/// Results are written to `adata.obs[key_added]` (cluster labels as strings)
/// and `adata.uns["leiden"]` (parameters and backend metadata).
///
/// ## Backend priority
///
/// Three implementations are tried in order; the first that succeeds wins:
///
/// 1. **Rust-native** (`scx_accel::leiden`) — always attempted first. Fastest
///    on single-node workloads and has no Python dependencies. Ignores
///    `device`; setting `device="gpu"` does **not** guarantee GPU execution
///    if the Rust-native path succeeds.
/// 2. **cuGraph** (`cugraph.leiden`) — attempted only when `device="gpu"` (or
///    `device="auto"` on a GPU-available host) AND the Rust-native path
///    raised an error. Requires `cugraph` installed.
/// 3. **leidenalg** — Python fallback via `igraph` + `leidenalg`.
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
///     device: Device selection — "auto" (default), "cpu", or "gpu". Only
///         gates the cuGraph attempt (see Backend priority above).
///     parallel: Run the Rust-native Leiden in parallel mode (conflict-free
///         graph coloring). Default `False`.
///     theta: Resolution scaling knob for cuGraph Leiden only (default 1.0).
///         **Ignored** by the Rust-native and leidenalg backends.
///
/// Notes:
///     GPU and CPU Leiden may produce different partitions on the same graph
///     due to algorithmic differences (cuGraph uses a different refinement
///     strategy than leidenalg). Both produce valid, high-quality community
///     structures. Compare results via ARI or NMI when switching backends.
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

    // Priority 1: Rust-native Leiden (fastest, no Python dependencies).
    // `theta` is cuGraph-only and is not forwarded here.
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
            theta,
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

    // Priority 3: Python leidenalg via igraph (fallback).
    // `theta` is cuGraph-only and is not forwarded here.
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
/// `cugraph.leiden()`, and writes results to adata. `theta` is forwarded as
/// the optional cuGraph-specific resolution scaling kwarg.
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

    // Write metadata to adata.uns["leiden"]
    let leiden_dict = PyDict::new(py);
    let params_dict = PyDict::new(py);
    params_dict.set_item("resolution", resolution)?;
    params_dict.set_item("random_state", random_state)?;
    params_dict.set_item("n_iterations", max_iter)?;
    params_dict.set_item("theta", theta)?;
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
