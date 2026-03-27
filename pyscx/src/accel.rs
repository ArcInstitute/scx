//! Python bindings for SCX accelerators (PCA, kNN, etc.).
//!
//! Exposes `pyscx.accel.pca(adata, ...)` which runs randomized PCA
//! streaming from SCX's backed mode and writes results to standard
//! AnnData slots (obsm["X_pca"], varm["PCs"], uns["pca"]).

use numpy::PyArray2;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::ScxBackedSparseDataset;

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

        // Build boolean mask.
        let mask_vec: Vec<bool> = (0..n_obs).map(|i| indices.contains(&i)).collect();
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
    let result = if let (Some(chunk_size), Ok(backed)) = (
        gene_chunk_size,
        x.extract::<PyRef<ScxBackedSparseDataset>>(),
    ) {
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

        let dense = if is_sparse || x.hasattr("toarray")? {
            x.call_method0("toarray")?
        } else {
            numpy
                .call_method1("asarray", (&x,))?
                .call_method1("astype", ("float32",))?
        };

        let shape: (usize, usize) = dense.getattr("shape")?.extract()?;
        let (n_obs, n_vars) = shape;

        let flat = dense
            .call_method1("astype", ("float32",))?
            .call_method0("ravel")?;
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
