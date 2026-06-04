//! Wilcoxon rank-sum differential expression, stratified DE, and cell-eval bridge.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::ScxBackedSparseDataset;

/// A single stratum: the composite key values and a boolean mask over adata.obs.
pub(super) struct Stratum {
    /// Key values for each stratify_by column.
    pub(super) key: Vec<String>,
}

/// Extract and validate strata from adata.obs.
///
/// Returns (strata, boolean_masks_as_py_arrays, stratify_col_names).
/// Drops NaN rows with a logged warning. Filters by min_cells_per_stratum.
pub(super) fn extract_strata<'py>(
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

/// Run Wilcoxon rank-sum DE on a single adata (no stratification).
///
/// Returns the DiffExpResult from scx_accel.
#[allow(clippy::too_many_arguments)]
fn run_rank_genes_groups_inner(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    groupby: &str,
    reference: &str,
    gene_chunk_size: Option<usize>,
    rankby_abs: bool,
    tie_correct: bool,
    prefer_format: &str,
    device: &str,
    gpu_device_id: Option<usize>,
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

    // Check if X is a ScxBackedSparseDataset / ScxLazyTransformedDataset
    // for streaming path. CSC dispatch routes through `as_column_source()`.
    let x = adata.getattr("X")?;

    if prefer_format == "csc" {
        if gpu_device_id.is_some() {
            return Err(PyRuntimeError::new_err(
                "device='gpu' with prefer_format='csc' is not supported in v1; \
                 use device='cpu' for CSC dispatch or prefer_format='csr' for GPU.",
            ));
        }
        // CSC dispatch: works on both backed and lazy datasets via
        // `as_column_source`. The kernel reads each gene chunk as a
        // CSC slab once, scatters into a row-major dense buffer, and
        // hands it to the existing `wilcoxon_rank_sum` kernel.
        let chunk_size = gene_chunk_size.unwrap_or(500);

        if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
            // Validate CSC availability under the GIL, then clone the Arc so
            // the Rust kernel can run without holding the GIL.
            if backed.kept_to_global.is_some() {
                return Err(PyRuntimeError::new_err(
                    "CSC requested but unavailable: a row deletion vector is active",
                ));
            }
            let csc_reader = backed
                .backed_csc
                .as_ref()
                .ok_or_else(|| {
                    PyRuntimeError::new_err(
                        "CSC requested but unavailable: file has no CSC sidecar",
                    )
                })?
                .clone();
            drop(backed);
            // Pass the concrete `BackedCscReader` (not `&dyn`) so the
            // closure is `Send` — `dyn ColumnShardSource` is not `Send`.
            let result = py
                .detach(|| {
                    scx_accel::wilcoxon_rank_sum_streaming_csc(
                        csc_reader.as_ref(),
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        chunk_size,
                        log_transformed,
                        rankby_abs,
                        tie_correct,
                    )
                })
                .map(|mut r| {
                    r.exec_info = super::route::cpu_exec_info(
                        device,
                        scx_accel::InputLayout::BackedCsc,
                        false, // no GPU CSC kernel
                        true,
                        Some(chunk_size),
                    );
                    r
                })
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?;
            return Ok((result, unique_groups));
        }
        if let Ok(lazy) = x.extract::<PyRef<crate::lazy_transform::ScxLazyTransformedDataset>>() {
            let lazy_src = lazy.as_column_source().ok_or_else(|| {
                PyRuntimeError::new_err(
                    "CSC requested but unavailable: file has no CSC sidecar, \
                     the transform chain contains a non-column-local op, or \
                     a row deletion vector is active",
                )
            })?;
            drop(lazy);
            let result = py
                .detach(|| {
                    scx_accel::wilcoxon_rank_sum_streaming_csc(
                        &lazy_src,
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        chunk_size,
                        log_transformed,
                        rankby_abs,
                        tie_correct,
                    )
                })
                .map(|mut r| {
                    r.exec_info = super::route::cpu_exec_info(
                        device,
                        scx_accel::InputLayout::BackedCsc,
                        false, // no GPU CSC kernel
                        true,
                        Some(chunk_size),
                    );
                    r
                })
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?;
            return Ok((result, unique_groups));
        }
        return Err(PyRuntimeError::new_err(
            "prefer_format='csc' requires adata.X to be a backed or lazy SCX \
             dataset; got a regular scipy/dense matrix",
        ));
    }

    let result = if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        // Backed mode: stream shards with gene-chunked DE. Clone the Arc
        // so the kernel runs without holding the GIL.
        let chunk_size = gene_chunk_size.unwrap_or(500);
        let reader = std::sync::Arc::clone(&backed.backed);
        // If a CSC sidecar reader exists on the dataset, hand it to the GPU
        // streaming path so the default v3 route can dispatch to the
        // CSC-direct Wilcoxon driver. None falls through to the v3 CSR-direct
        // fallback. Mirrors the pdex_ref backed path. Only bind under the gpu
        // feature — the CPU branch takes no CSC.
        #[cfg(feature = "gpu")]
        let csc_reader = backed.backed_csc.as_ref().map(std::sync::Arc::clone);
        drop(backed);
        match gpu_device_id {
            #[cfg(feature = "gpu")]
            Some(device_id) => py
                .detach(|| {
                    scx_accel::wilcoxon_rank_sum_gpu(
                        device_id,
                        scx_accel::GpuDeShardInput::Backed {
                            csr: &reader,
                            csc: csc_reader.as_deref(),
                        },
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        Some(chunk_size),
                        log_transformed,
                        rankby_abs,
                        tie_correct,
                    )
                })
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?,
            #[cfg(not(feature = "gpu"))]
            Some(_) => unreachable!("gpu_device_id is None when gpu feature is disabled"),
            None => py
                .detach(|| {
                    scx_accel::wilcoxon_rank_sum_streaming(
                        &reader,
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        chunk_size,
                        log_transformed,
                        rankby_abs,
                        tie_correct,
                    )
                })
                .map(|mut r| {
                    r.exec_info = super::route::cpu_exec_info(
                        device,
                        scx_accel::InputLayout::BackedCsr,
                        true,
                        false,
                        Some(chunk_size),
                    );
                    r
                })
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?,
        }
    } else {
        // ScxLazyTransformedDataset (non-CSC GPU path) — route
        // through `wilcoxon_rank_sum_gpu` with `GpuDeShardInput::Lazy`
        // (device-resident shard
        // pipeline) before falling through to the scipy/numpy paths.
        // CPU lazy without CSC keeps the scipy/numpy fallback.
        #[cfg(feature = "gpu")]
        {
            if let Some(device_id) = gpu_device_id {
                if let Ok(lazy) =
                    x.extract::<PyRef<crate::lazy_transform::ScxLazyTransformedDataset>>()
                {
                    let chunk_size = gene_chunk_size.unwrap_or(500);
                    let lazy_src = lazy.as_shard_source();
                    drop(lazy);
                    let result = py
                        .detach(|| {
                            scx_accel::wilcoxon_rank_sum_gpu(
                                device_id,
                                scx_accel::GpuDeShardInput::Lazy(&lazy_src),
                                &gene_names,
                                &groups,
                                &unique_groups,
                                ref_idx,
                                Some(chunk_size),
                                log_transformed,
                                rankby_abs,
                                tie_correct,
                            )
                        })
                        .map_err(|e: scx_accel::AccelError| {
                            PyRuntimeError::new_err(e.to_string())
                        })?;
                    return Ok((result, unique_groups));
                }
            }
        }

        let is_sparse = scipy_sparse
            .call_method1("issparse", (&x,))?
            .extract::<bool>()?;

        if is_sparse {
            // Sparse in-memory: extract CSR arrays and use gene-chunked path
            // to avoid O(n_obs × n_vars) dense materialization.
            //
            // `ensure_csr` short-circuits when the input is already CSR with
            // sorted indices (the common h5ad case) and copies-then-sorts
            // only when the caller's CSR has `has_sorted_indices == False`.
            // Sorted indices are a precondition of
            // `scx_engine::project_csr_row`; without this step the CPU
            // sparse path silently returns U = n_g·n_ref/2 for every gene
            // on inputs with unsorted CSRs (e.g. pbmc10k.h5ad).
            let (csr_obj, _) = crate::anndata::ensure_csr(py, &x, /* in_place */ false)?;
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
            let chunk_size = gene_chunk_size.unwrap_or(500);
            match gpu_device_id {
                #[cfg(feature = "gpu")]
                Some(device_id) => py
                    .detach(|| {
                        scx_accel::wilcoxon_rank_sum_gpu(
                            device_id,
                            scx_accel::GpuDeShardInput::Csr(&csr),
                            &gene_names,
                            &groups,
                            &unique_groups,
                            ref_idx,
                            Some(chunk_size),
                            log_transformed,
                            rankby_abs,
                            tie_correct,
                        )
                    })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                #[cfg(not(feature = "gpu"))]
                Some(_) => unreachable!("gpu_device_id is None when gpu feature is disabled"),
                None => py
                    .detach(|| {
                        scx_accel::wilcoxon_rank_sum_sparse(
                            &csr,
                            &gene_names,
                            &groups,
                            &unique_groups,
                            ref_idx,
                            chunk_size,
                            log_transformed,
                            rankby_abs,
                            tie_correct,
                        )
                    })
                    .map(|mut r| {
                        r.exec_info = super::route::cpu_exec_info(
                            device,
                            scx_accel::InputLayout::CsrHost,
                            true,
                            false,
                            Some(chunk_size),
                        );
                        r
                    })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            }
        } else {
            // Dense numpy array: flatten and use direct wilcoxon_rank_sum
            let dense = numpy
                .call_method1("asarray", (&x,))?
                .call_method1("astype", ("float32",))?;
            let shape: (usize, usize) = dense.getattr("shape")?.extract()?;
            let (n_obs, n_vars) = shape;

            let flat = dense.call_method0("ravel")?;
            let data: Vec<f32> = flat.extract()?;

            match gpu_device_id {
                #[cfg(feature = "gpu")]
                Some(device_id) => py
                    .detach(|| {
                        scx_accel::wilcoxon_rank_sum_gpu_dense(
                            device_id,
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
                    })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                #[cfg(not(feature = "gpu"))]
                Some(_) => unreachable!("gpu_device_id is None when gpu feature is disabled"),
                None => py
                    .detach(|| {
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
                            0,
                        )
                    })
                    .map(|mut r| {
                        r.exec_info = super::route::cpu_exec_info(
                            device,
                            scx_accel::InputLayout::DenseHost,
                            true,
                            false,
                            None,
                        );
                        r
                    })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            }
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

/// Wilcoxon rank-sum differential expression, scanpy-compatible.
///
/// Writes results to ``adata.uns["rank_genes_groups"]`` as scanpy-style
/// structured arrays. The chosen accelerator execution route is recorded both
/// as ``adata.uns["rank_genes_groups"]["scx_accel_route"]`` and under the
/// unified ``adata.uns["scx_accel"]["rank_genes_groups"]`` dict (keys:
/// ``route``, ``fallback_reason``, ``chunk_size``, ``csc_available``, ...).
/// Check ``route`` when comparing CPU vs GPU performance — GPU is fastest only
/// when the input layout matches the op. Returns ``None`` (results live on
/// ``adata.uns``); the stratified path (``stratify_by``) instead returns a
/// concatenated pandas DataFrame and does not write route metadata.
#[pyfunction]
#[pyo3(signature = (adata, groupby, reference="rest", n_genes=None, method="wilcoxon", gene_chunk_size=None, stratify_by=None, min_cells_per_stratum=50, rankby_abs=false, tie_correct=false, prefer_format="csr", device="auto"))]
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
    prefer_format: &str,
    device: &str,
) -> PyResult<Py<PyAny>> {
    if method != "wilcoxon" {
        return Err(PyRuntimeError::new_err(format!(
            "unsupported method '{method}': only 'wilcoxon' is currently supported"
        )));
    }
    if !matches!(prefer_format, "csr" | "csc") {
        return Err(PyValueError::new_err(format!(
            "Invalid prefer_format={prefer_format:?}; expected 'csr' or 'csc'"
        )));
    }
    let resolved = super::gpu::resolve_device(device)?;
    #[cfg(feature = "gpu")]
    let gpu_device_id = resolved.gpu_id();
    #[cfg(not(feature = "gpu"))]
    let gpu_device_id: Option<usize> = {
        let _ = resolved;
        None
    };
    // CSC has no GPU kernel in v1. Silently fall back to CPU when device
    // is "auto" (the default) so users passing only `prefer_format="csc"`
    // on a GPU host don't hit an error. Reject only when the user
    // explicitly asked for GPU.
    let gpu_device_id = if prefer_format == "csc" {
        if device.starts_with("gpu") {
            return Err(PyRuntimeError::new_err(
                "device='gpu' with prefer_format='csc' is not supported in v1; \
                 use device='cpu' or device='auto' for CSC dispatch.",
            ));
        }
        None
    } else {
        gpu_device_id
    };

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
                prefer_format,
                device,
                gpu_device_id,
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
        prefer_format,
        device,
        gpu_device_id,
    )?;

    // Write results to adata.uns["rank_genes_groups"] in scanpy format.
    write_de_to_adata(py, adata, &result, groupby, reference, n_genes)?;

    // Record the accelerator execution route: both inside the scanpy-style
    // rank_genes_groups dict (as `scx_accel_route`) and under the unified
    // adata.uns["scx_accel"]["rank_genes_groups"] lookup. `result.exec_info`
    // is already complete (route + reason) — set by the single planner on both
    // the CPU dispatch sites and inside scx-accel for GPU routes.
    if let Ok(rgg) = adata.getattr("uns")?.get_item("rank_genes_groups") {
        rgg.set_item("scx_accel_route", result.exec_info.route.as_str())?;
    }
    super::route::write_accel_route(py, adata, "rank_genes_groups", &result.exec_info)?;

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

/// Convert a DiffExpResult into a polars DataFrame matching cell-eval's
/// `DEResults` schema.
///
/// Output columns:
///   - `target` (Utf8): group/perturbation name
///   - `feature` (Utf8): gene name
///   - `fold_change` (Float64): 2^(log2_fold_change) — linear fold change
///   - `p_value` (Float64): raw p-value
///   - `fdr` (Float64): BH-adjusted p-value
///   - `log2_fold_change` (Float64): log2 fold change
///   - `abs_log2_fold_change` (Float64): |log2_fold_change|
fn de_result_to_cell_eval_dataframe<'py>(
    py: Python<'py>,
    result: &scx_accel::DiffExpResult,
    n_genes: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let pl = py.import("polars").map_err(|_| {
        PyRuntimeError::new_err(
            "polars is required for rank_genes_groups_df(). \
             Install it: pip install 'pyscx[eval]'  (or: pip install polars)",
        )
    })?;

    // Pre-compute total row count for capacity pre-allocation.
    let total_rows: usize = result
        .group_names
        .iter()
        .enumerate()
        .map(|(i, _)| {
            n_genes
                .unwrap_or(result.names[i].len())
                .min(result.names[i].len())
        })
        .sum();

    // Build flat column vectors from per-group arrays.
    let mut targets: Vec<String> = Vec::with_capacity(total_rows);
    let mut features: Vec<String> = Vec::with_capacity(total_rows);
    let mut fold_changes: Vec<f64> = Vec::with_capacity(total_rows);
    let mut p_values: Vec<f64> = Vec::with_capacity(total_rows);
    let mut fdrs: Vec<f64> = Vec::with_capacity(total_rows);
    let mut log2_fcs: Vec<f64> = Vec::with_capacity(total_rows);
    let mut abs_log2_fcs: Vec<f64> = Vec::with_capacity(total_rows);

    for (i, group_name) in result.group_names.iter().enumerate() {
        let full_n_genes = result.names[i].len();
        let n = n_genes.unwrap_or(full_n_genes).min(full_n_genes);

        // Batch-clone the group name once per group instead of per-gene.
        targets.extend(std::iter::repeat_n(group_name.clone(), n));
        features.extend(result.names[i][..n].iter().cloned());

        for j in 0..n {
            let lfc = result.logfoldchanges[i][j];
            log2_fcs.push(lfc);
            // Non-finite values (NaN, ±Inf) are passed through intentionally:
            // NaN.abs() → NaN, (-Inf).abs() → Inf.  Downstream polars consumers
            // can filter these via drop_nulls()/is_finite() as needed.
            abs_log2_fcs.push(lfc.abs());
            // Convert log2 fold change to linear fold change: 2^lfc.
            // Use f64::exp2 for precision. Non-finite values pass through.
            fold_changes.push(if lfc.is_finite() { lfc.exp2() } else { lfc });

            p_values.push(result.pvals[i][j]);
            fdrs.push(result.pvals_adj[i][j]);
        }
    }

    // Build polars DataFrame from column vectors with explicit column order
    // matching cell-eval's DEResults schema.
    let dict = PyDict::new(py);
    dict.set_item("target", targets)?;
    dict.set_item("feature", features)?;
    dict.set_item("fold_change", fold_changes)?;
    dict.set_item("p_value", p_values)?;
    dict.set_item("fdr", fdrs)?;
    dict.set_item("log2_fold_change", log2_fcs)?;
    dict.set_item("abs_log2_fold_change", abs_log2_fcs)?;

    let df = pl.call_method1("DataFrame", (dict,))?;
    // Enforce column order to match cell-eval's DEResults schema, regardless
    // of dict iteration order or polars constructor behavior.
    let column_order = pyo3::types::PyList::new(
        py,
        [
            "target",
            "feature",
            "fold_change",
            "p_value",
            "fdr",
            "log2_fold_change",
            "abs_log2_fold_change",
        ],
    )?;
    let df = df.call_method1("select", (column_order,))?;
    Ok(df)
}

/// Run Wilcoxon rank-sum DE and return results as a polars DataFrame in
/// cell-eval's `DEResults` format.
///
/// This is the format bridge between SCX's Wilcoxon DE and cell-eval's DE
/// metric pipeline. The returned DataFrame can be fed directly into
/// `cell_eval.data.DEResults` or `cell_eval.data.DEComparison`. The polars
/// DataFrame carries no metadata; the accelerator execution route is recorded
/// on `adata.uns["scx_accel"]["rank_genes_groups_df"]` instead.
///
/// Output columns:
///   - `target` (str): perturbation/group name
///   - `feature` (str): gene name
///   - `fold_change` (f64): linear fold change (2^log2FC)
///   - `p_value` (f64): raw p-value
///   - `fdr` (f64): BH-adjusted p-value
///   - `log2_fold_change` (f64): log2 fold change
///   - `abs_log2_fold_change` (f64): |log2FC|
///
/// Args:
///     adata: AnnData object with X and obs[groupby]
///     groupby: Column in adata.obs to group cells by
///     reference: Group name to compare against (default: "rest" = 1-vs-rest)
///     n_genes: Number of top genes to report per group (default: all genes)
///     gene_chunk_size: Genes per chunk for streaming DE (default: None, which
///         uses 500 internally for sparse/backed inputs)
///     rankby_abs: Sort genes by |score| instead of signed score (default: False)
///     tie_correct: Apply tie correction in the Wilcoxon test (default: False)
///
/// Example:
///     de_df = pyscx.accel.rank_genes_groups_df(adata, "perturbation")
///     # de_df is a polars DataFrame with cell-eval columns
#[pyfunction]
#[pyo3(signature = (adata, groupby, reference="rest", n_genes=None, gene_chunk_size=None, rankby_abs=false, tie_correct=false, device="auto"))]
#[allow(clippy::too_many_arguments)]
pub fn rank_genes_groups_df(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    groupby: &str,
    reference: &str,
    n_genes: Option<usize>,
    gene_chunk_size: Option<usize>,
    rankby_abs: bool,
    tie_correct: bool,
    device: &str,
) -> PyResult<Py<PyAny>> {
    let resolved = super::gpu::resolve_device(device)?;
    #[cfg(feature = "gpu")]
    let gpu_device_id = resolved.gpu_id();
    #[cfg(not(feature = "gpu"))]
    let gpu_device_id: Option<usize> = {
        let _ = resolved;
        None
    };
    // `rank_genes_groups_df` is the cell-eval-style entry; CSC dispatch
    // is reserved for the scanpy-style `rank_genes_groups`. Pin to CSR.
    let (result, _unique_groups) = run_rank_genes_groups_inner(
        py,
        adata,
        groupby,
        reference,
        gene_chunk_size,
        rankby_abs,
        tie_correct,
        "csr",
        device,
        gpu_device_id,
    )?;

    // Record the accelerator execution route on adata.uns; the returned
    // polars DataFrame carries no metadata of its own. `result.exec_info` is
    // already complete (route + reason) from the single planner.
    super::route::write_accel_route(py, adata, "rank_genes_groups_df", &result.exec_info)?;

    let df = de_result_to_cell_eval_dataframe(py, &result, n_genes)?;
    Ok(df.unbind())
}

// ---------------------------------------------------------------------------
// pdex `mode="ref"` accelerator binding
// ---------------------------------------------------------------------------

/// Auto-detect whether `adata.X` looks log1p-transformed.
///
/// Mirrors `pdex._utils._detect_is_log1p` and the existing
/// `adata.uns["log1p"]` probe used by `rank_genes_groups`.  Prefers the
/// explicit annotation when present.
fn detect_is_log1p(py: Python<'_>, adata: &Bound<'_, PyAny>) -> PyResult<bool> {
    if let Ok(uns) = adata.getattr("uns") {
        if let Ok(v) = uns.call_method1("get", ("log1p",)) {
            if !v.is_none() {
                return Ok(true);
            }
        }
    }
    // Fall back to a max-value heuristic on adata.X, matching pdex's default.
    // Skip the probe for backed datasets — touching X here would force a load.
    let x = adata.getattr("X")?;
    if x.extract::<PyRef<ScxBackedSparseDataset>>().is_ok() {
        return Ok(false);
    }
    let np = py.import("numpy")?;
    let scipy_sparse = py.import("scipy.sparse")?;
    let is_sparse = scipy_sparse
        .call_method1("issparse", (&x,))?
        .extract::<bool>()
        .unwrap_or(false);
    let max_val: f64 = if is_sparse {
        let data = x.getattr("data")?;
        let m = np.call_method1("max", (data,))?;
        m.extract::<f64>().unwrap_or(f64::NAN)
    } else {
        let m = np.call_method1("max", (x,))?;
        m.extract::<f64>().unwrap_or(f64::NAN)
    };
    // pdex's heuristic: log1p-transformed counts rarely exceed ~30.
    Ok(max_val.is_finite() && max_val < 30.0)
}

/// Resolve group encoding and reference index, mirroring
/// `run_rank_genes_groups_inner`.
fn resolve_groups_and_reference(
    adata: &Bound<'_, PyAny>,
    groupby: &str,
    reference: &str,
) -> PyResult<(Vec<usize>, Vec<String>, usize)> {
    let obs = adata.getattr("obs")?;
    let group_col = obs.get_item(groupby)?;
    let group_labels: Vec<String> = group_col
        .call_method1("astype", ("str",))?
        .call_method0("tolist")?
        .extract()?;

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

    let group_name_to_idx: std::collections::HashMap<&str, usize> = unique_groups
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();

    let ref_idx = *group_name_to_idx.get(reference).ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "reference group '{reference}' not found in adata.obs['{groupby}']"
        ))
    })?;

    // Unknown groups (NaN / empty strings after astype("str") become "nan" /
    // "") get mapped to a sentinel that exceeds n_groups, so pdex_ref drops
    // them. Use `unique_groups.len()` as the out-of-range marker.
    let oor = unique_groups.len();
    let groups: Vec<usize> = group_labels
        .iter()
        .map(|label| {
            if label.is_empty() || label == "nan" {
                oor
            } else {
                *group_name_to_idx.get(label.as_str()).unwrap_or(&oor)
            }
        })
        .collect();

    Ok((groups, unique_groups, ref_idx))
}

/// Run pdex `mode="ref"` against an AnnData, dispatching to the SCX-backed
/// streaming, in-memory-CSR, or dense kernel based on `adata.X`.
#[allow(clippy::too_many_arguments)]
fn run_pdex_ref_inner(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    groupby: &str,
    reference: &str,
    geometric_mean: bool,
    is_log1p: Option<bool>,
    epsilon: f64,
    gene_chunk_size: Option<usize>,
    prefer_format: &str,
    device: &str,
    gpu_device_id: Option<usize>,
) -> PyResult<scx_accel::PdexRefResult> {
    let numpy = py.import("numpy")?;
    let scipy_sparse = py.import("scipy.sparse")?;

    let (groups, unique_groups, ref_idx) = resolve_groups_and_reference(adata, groupby, reference)?;

    let var = adata.getattr("var")?;
    let var_names = var.getattr("index")?;
    let gene_names: Vec<String> = var_names.call_method0("tolist")?.extract()?;

    let resolved_log1p = match is_log1p {
        Some(v) => v,
        None => detect_is_log1p(py, adata)?,
    };
    let mode = scx_accel::GeomMeanMode::from_flags(geometric_mean, resolved_log1p);

    let x = adata.getattr("X")?;

    if prefer_format == "csc" {
        if gpu_device_id.is_some() {
            return Err(PyRuntimeError::new_err(
                "device='gpu' with prefer_format='csc' is not supported in v1; \
                 use device='cpu' for CSC dispatch or prefer_format='csr' for GPU.",
            ));
        }
        let chunk_size = gene_chunk_size.unwrap_or(500);

        if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
            if backed.kept_to_global.is_some() {
                return Err(PyRuntimeError::new_err(
                    "CSC requested but unavailable: a row deletion vector is active",
                ));
            }
            let csc_reader = backed
                .backed_csc
                .as_ref()
                .ok_or_else(|| {
                    PyRuntimeError::new_err(
                        "CSC requested but unavailable: file has no CSC sidecar",
                    )
                })?
                .clone();
            drop(backed);
            return py
                .detach(|| {
                    scx_accel::pdex_ref_streaming_csc(
                        csc_reader.as_ref(),
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        chunk_size,
                        mode,
                        epsilon,
                    )
                })
                .map(|mut r| {
                    r.exec_info = super::route::cpu_exec_info(
                        device,
                        scx_accel::InputLayout::BackedCsc,
                        false, // no GPU CSC kernel
                        true,
                        Some(chunk_size),
                    );
                    r
                })
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()));
        }
        if let Ok(lazy) = x.extract::<PyRef<crate::lazy_transform::ScxLazyTransformedDataset>>() {
            let lazy_src = lazy.as_column_source().ok_or_else(|| {
                PyRuntimeError::new_err(
                    "CSC requested but unavailable: file has no CSC sidecar, \
                     the transform chain contains a non-column-local op, or \
                     a row deletion vector is active",
                )
            })?;
            drop(lazy);
            return py
                .detach(|| {
                    scx_accel::pdex_ref_streaming_csc(
                        &lazy_src,
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        chunk_size,
                        mode,
                        epsilon,
                    )
                })
                .map(|mut r| {
                    r.exec_info = super::route::cpu_exec_info(
                        device,
                        scx_accel::InputLayout::BackedCsc,
                        false, // no GPU CSC kernel
                        true,
                        Some(chunk_size),
                    );
                    r
                })
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()));
        }
        return Err(PyRuntimeError::new_err(
            "prefer_format='csc' requires adata.X to be a backed or lazy SCX \
             dataset; got a regular scipy/dense matrix",
        ));
    }

    if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        let chunk_size = gene_chunk_size.unwrap_or(500);
        let reader = std::sync::Arc::clone(&backed.backed);
        // G4.3: if a CSC sidecar reader exists on the dataset, hand it to
        // the GPU streaming path so the default v3 route can dispatch to
        // the CSC-direct driver. None falls through to the v3 CSR-direct
        // fallback. Only bind under the gpu feature — the CPU branch doesn't
        // take a CSC reader.
        #[cfg(feature = "gpu")]
        let csc_reader = backed.backed_csc.as_ref().map(std::sync::Arc::clone);
        drop(backed);
        return match gpu_device_id {
            #[cfg(feature = "gpu")]
            Some(device_id) => py
                .detach(|| {
                    scx_accel::pdex_ref_gpu(
                        device_id,
                        scx_accel::GpuDeShardInput::Backed {
                            csr: &reader,
                            csc: csc_reader.as_deref(),
                        },
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        Some(chunk_size),
                        mode,
                        epsilon,
                    )
                })
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string())),
            #[cfg(not(feature = "gpu"))]
            Some(_) => unreachable!("gpu_device_id is None when gpu feature is disabled"),
            None => py
                .detach(|| {
                    scx_accel::pdex_ref_streaming(
                        &reader,
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        chunk_size,
                        mode,
                        epsilon,
                    )
                })
                .map(|mut r| {
                    r.exec_info = super::route::cpu_exec_info(
                        device,
                        scx_accel::InputLayout::BackedCsr,
                        true,
                        false,
                        Some(chunk_size),
                    );
                    r
                })
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string())),
        };
    }

    // G1.8: ScxLazyTransformedDataset (non-CSC GPU path). See the matching
    // branch in `run_rank_genes_groups_inner`. CPU lazy without CSC still
    // falls through to scipy-CSR / numpy materialisation below.
    #[cfg(feature = "gpu")]
    if let Some(device_id) = gpu_device_id {
        if let Ok(lazy) = x.extract::<PyRef<crate::lazy_transform::ScxLazyTransformedDataset>>() {
            let chunk_size = gene_chunk_size.unwrap_or(500);
            let lazy_src = lazy.as_shard_source();
            drop(lazy);
            return py
                .detach(|| {
                    scx_accel::pdex_ref_gpu(
                        device_id,
                        scx_accel::GpuDeShardInput::Lazy(&lazy_src),
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        Some(chunk_size),
                        mode,
                        epsilon,
                    )
                })
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()));
        }
    }

    let is_sparse = scipy_sparse
        .call_method1("issparse", (&x,))?
        .extract::<bool>()?;

    if is_sparse {
        // `ensure_csr` enforces sorted column indices — required by
        // `scx_engine::project_csr_row`. Without this, unsorted scipy
        // CSR inputs silently return U = n_g·n_ref/2 for every gene.
        let (csr_obj, _) = crate::anndata::ensure_csr(py, &x, /* in_place */ false)?;
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
        let chunk_size = gene_chunk_size.unwrap_or(500);
        match gpu_device_id {
            #[cfg(feature = "gpu")]
            Some(device_id) => py
                .detach(|| {
                    scx_accel::pdex_ref_gpu(
                        device_id,
                        scx_accel::GpuDeShardInput::Csr(&csr),
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        Some(chunk_size),
                        mode,
                        epsilon,
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string())),
            #[cfg(not(feature = "gpu"))]
            Some(_) => unreachable!("gpu_device_id is None when gpu feature is disabled"),
            None => py
                .detach(|| {
                    scx_accel::pdex_ref_sparse(
                        &csr,
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        chunk_size,
                        mode,
                        epsilon,
                    )
                })
                .map(|mut r| {
                    r.exec_info = super::route::cpu_exec_info(
                        device,
                        scx_accel::InputLayout::CsrHost,
                        true,
                        false,
                        Some(chunk_size),
                    );
                    r
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string())),
        }
    } else {
        let dense = numpy
            .call_method1("asarray", (&x,))?
            .call_method1("astype", ("float32",))?;
        let shape: (usize, usize) = dense.getattr("shape")?.extract()?;
        let (n_obs, n_vars) = shape;
        let flat = dense.call_method0("ravel")?;
        let data: Vec<f32> = flat.extract()?;

        match gpu_device_id {
            #[cfg(feature = "gpu")]
            Some(device_id) => py
                .detach(|| {
                    scx_accel::pdex_ref_gpu_dense(
                        device_id,
                        &data,
                        n_obs,
                        n_vars,
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        mode,
                        epsilon,
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string())),
            #[cfg(not(feature = "gpu"))]
            Some(_) => unreachable!("gpu_device_id is None when gpu feature is disabled"),
            None => py
                .detach(|| {
                    scx_accel::pdex_ref(
                        &data,
                        n_obs,
                        n_vars,
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        mode,
                        epsilon,
                    )
                })
                .map(|mut r| {
                    r.exec_info = super::route::cpu_exec_info(
                        device,
                        scx_accel::InputLayout::DenseHost,
                        true,
                        false,
                        None,
                    );
                    r
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string())),
        }
    }
}

/// Convert a `PdexRefResult` to a polars DataFrame matching pdex's row schema.
///
/// Columns: `target`, `feature`, `target_mean`, `ref_mean`,
/// `target_membership`, `ref_membership`, `fold_change` (=log2_fold_change,
/// deprecated alias for migration), `log2_fold_change`, `percent_change`,
/// `p_value`, `statistic`, `fdr`.
fn pdex_ref_result_to_dataframe<'py>(
    py: Python<'py>,
    result: &scx_accel::PdexRefResult,
) -> PyResult<Bound<'py, PyAny>> {
    let pl = py.import("polars").map_err(|_| {
        PyRuntimeError::new_err(
            "polars is required for pdex_ref(). Install it: \
             pip install 'pyscx[eval]'  (or: pip install polars)",
        )
    })?;

    let n_genes = result.feature_names.len();
    let n_test = result.group_names.len();
    let total_rows = n_genes * n_test;

    let mut targets: Vec<String> = Vec::with_capacity(total_rows);
    let mut features: Vec<String> = Vec::with_capacity(total_rows);
    let mut target_means: Vec<f64> = Vec::with_capacity(total_rows);
    let mut ref_means: Vec<f64> = Vec::with_capacity(total_rows);
    let mut target_memberships: Vec<u64> = Vec::with_capacity(total_rows);
    let mut ref_memberships: Vec<u64> = Vec::with_capacity(total_rows);
    let mut log2_fcs: Vec<f64> = Vec::with_capacity(total_rows);
    let mut percent_changes: Vec<f64> = Vec::with_capacity(total_rows);
    let mut p_values: Vec<f64> = Vec::with_capacity(total_rows);
    let mut statistics: Vec<f64> = Vec::with_capacity(total_rows);
    let mut fdrs: Vec<f64> = Vec::with_capacity(total_rows);

    for (tg, group_name) in result.group_names.iter().enumerate() {
        targets.extend(std::iter::repeat_n(group_name.clone(), n_genes));
        features.extend(result.feature_names.iter().cloned());
        target_means.extend(result.target_means[tg].iter().copied());
        ref_means.extend(result.ref_means.iter().copied());
        target_memberships.extend(std::iter::repeat_n(
            result.target_memberships[tg] as u64,
            n_genes,
        ));
        ref_memberships.extend(std::iter::repeat_n(result.ref_membership as u64, n_genes));
        log2_fcs.extend(result.log2_fold_changes[tg].iter().copied());
        percent_changes.extend(result.percent_changes[tg].iter().copied());
        p_values.extend(result.p_values[tg].iter().copied());
        statistics.extend(result.statistics[tg].iter().copied());
        fdrs.extend(result.fdrs[tg].iter().copied());
    }

    // pdex emits `fold_change` as a duplicate of `log2_fold_change` (a
    // deprecated alias retained for one release). Mirror that exactly so
    // downstream `DEResults` finds both columns.
    let fold_changes = log2_fcs.clone();

    let dict = PyDict::new(py);
    dict.set_item("target", targets)?;
    dict.set_item("feature", features)?;
    dict.set_item("target_mean", target_means)?;
    dict.set_item("ref_mean", ref_means)?;
    dict.set_item("target_membership", target_memberships)?;
    dict.set_item("ref_membership", ref_memberships)?;
    dict.set_item("fold_change", fold_changes)?;
    dict.set_item("log2_fold_change", log2_fcs)?;
    dict.set_item("percent_change", percent_changes)?;
    dict.set_item("p_value", p_values)?;
    dict.set_item("statistic", statistics)?;
    dict.set_item("fdr", fdrs)?;

    let df = pl.call_method1("DataFrame", (dict,))?;
    let column_order = pyo3::types::PyList::new(
        py,
        [
            "target",
            "feature",
            "target_mean",
            "ref_mean",
            "target_membership",
            "ref_membership",
            "fold_change",
            "log2_fold_change",
            "percent_change",
            "p_value",
            "statistic",
            "fdr",
        ],
    )?;
    let df = df.call_method1("select", (column_order,))?;
    Ok(df)
}

/// pdex `mode="ref"` differential expression on an SCX-backed or in-memory
/// AnnData, returned as a polars DataFrame matching pdex's row schema.
///
/// This is the SCX-native equivalent of `pdex.pdex(adata, groupby, mode="ref")`:
/// per (group, gene), reports pseudobulk means in natural (count) space, log2
/// fold change, percent change, Mann-Whitney U statistic, two-sided p-value,
/// and BH-adjusted FDR. The reference group is excluded from output.
///
/// Output columns: `target`, `feature`, `target_mean`, `ref_mean`,
/// `target_membership`, `ref_membership`, `fold_change` (=log2_fold_change,
/// deprecated alias retained for migration), `log2_fold_change`,
/// `percent_change`, `p_value`, `statistic`, `fdr`.
///
/// Args:
///     adata: AnnData object with X and obs[groupby]
///     groupby: Column in adata.obs to group cells by
///     reference: Reference group name (default: "non-targeting")
///     is_log1p: Whether adata.X contains log1p-transformed values. None
///         (default) auto-detects via adata.uns["log1p"] and a max-value
///         heuristic, matching pdex.
///     geometric_mean: If True (default), pseudobulk summary is the geometric
///         mean of expression values back-transformed to count space (matches
///         pdex's `geometric_mean=True`). If False, arithmetic mean is used.
///     epsilon: Pseudocount added to target_mean and ref_mean before computing
///         fold_change and percent_change. Default 0.0.
///     gene_chunk_size: Genes per chunk for sparse/backed streaming
///         (default: 500). Ignored for dense input.
///
/// The accelerator execution route is recorded on
/// ``adata.uns["scx_accel"]["pdex_ref"]`` (keys: ``route``,
/// ``fallback_reason``, ``chunk_size``, ``csc_available``, ...). The
/// CSC-direct GPU route (``route == "gpu_csc_v3"``) requires a backed SCX file
/// with a CSC sidecar (v3 is the default GPU DE route); in-memory CSR inputs
/// fall back to ``"gpu_csr_v3"`` with ``fallback_reason == "no_csc_sidecar"``. Check
/// ``route`` when comparing performance.
#[pyfunction]
#[pyo3(signature = (adata, groupby, *, reference="non-targeting", is_log1p=None, geometric_mean=true, epsilon=0.0, gene_chunk_size=None, prefer_format="csr", device="auto"))]
#[allow(clippy::too_many_arguments)]
pub fn pdex_ref(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    groupby: &str,
    reference: &str,
    is_log1p: Option<bool>,
    geometric_mean: bool,
    epsilon: f64,
    gene_chunk_size: Option<usize>,
    prefer_format: &str,
    device: &str,
) -> PyResult<Py<PyAny>> {
    if epsilon < 0.0 || !epsilon.is_finite() {
        return Err(PyValueError::new_err(format!(
            "epsilon must be non-negative and finite (got {epsilon})"
        )));
    }
    if !matches!(prefer_format, "csr" | "csc") {
        return Err(PyValueError::new_err(format!(
            "Invalid prefer_format={prefer_format:?}; expected 'csr' or 'csc'"
        )));
    }
    let resolved = super::gpu::resolve_device(device)?;
    #[cfg(feature = "gpu")]
    let gpu_device_id = resolved.gpu_id();
    #[cfg(not(feature = "gpu"))]
    let gpu_device_id: Option<usize> = {
        let _ = resolved;
        None
    };
    // CSC has no GPU kernel in v1. Silently fall back to CPU when device
    // is "auto" so `prefer_format="csc"` works on GPU hosts; reject only
    // when the user explicitly asked for GPU.
    let gpu_device_id = if prefer_format == "csc" {
        if device.starts_with("gpu") {
            return Err(PyRuntimeError::new_err(
                "device='gpu' with prefer_format='csc' is not supported in v1; \
                 use device='cpu' or device='auto' for CSC dispatch.",
            ));
        }
        None
    } else {
        gpu_device_id
    };
    let result = run_pdex_ref_inner(
        py,
        adata,
        groupby,
        reference,
        geometric_mean,
        is_log1p,
        epsilon,
        gene_chunk_size,
        prefer_format,
        device,
        gpu_device_id,
    )?;
    // Record the accelerator execution route on adata.uns["scx_accel"]["pdex_ref"].
    // `result.exec_info` is already complete (route + reason) from the single
    // planner — CPU sites via `cpu_exec_info`, GPU routes inside scx-accel.
    super::route::write_accel_route(py, adata, "pdex_ref", &result.exec_info)?;
    let df = pdex_ref_result_to_dataframe(py, &result)?;
    Ok(df.unbind())
}
