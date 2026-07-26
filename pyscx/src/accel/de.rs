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
/// Resolve scanpy's `use_raw` / `layer` selection contract.
///
/// `use_raw=None` (the scanpy default) resolves to `True` iff `adata.raw` is
/// present and no `layer` was requested; otherwise `False`. `use_raw=True` with
/// a `layer` is rejected (mutually exclusive, matching scanpy). Returns the
/// resolved boolean.
fn resolve_use_raw(
    adata: &Bound<'_, PyAny>,
    use_raw: Option<bool>,
    layer: Option<&str>,
) -> PyResult<bool> {
    if layer.is_some() && matches!(use_raw, Some(true)) {
        return Err(PyValueError::new_err(
            "Cannot specify both use_raw=True and layer=...; they are mutually exclusive.",
        ));
    }
    let has_raw = !adata.getattr("raw")?.is_none();
    Ok(match use_raw {
        Some(v) => v,
        None => has_raw && layer.is_none(),
    })
}

/// Select the DE input matrix and its gene names per the resolved `use_raw` /
/// `layer` contract. `use_raw` → `adata.raw.X` with `adata.raw.var.index`;
/// `layer` → `adata.layers[layer]` with `adata.var.index`; otherwise `adata.X`
/// with `adata.var.index`. The returned matrix flows through the same
/// backed/lazy/scipy/dense dispatch as before — only the source object changes.
fn select_de_matrix<'py>(
    adata: &Bound<'py, PyAny>,
    use_raw: bool,
    layer: Option<&str>,
) -> PyResult<(Bound<'py, PyAny>, Vec<String>)> {
    let var_names_of = |frame: &Bound<'py, PyAny>| -> PyResult<Vec<String>> {
        frame
            .getattr("index")?
            .call_method0("tolist")?
            .extract::<Vec<String>>()
    };
    if use_raw {
        let raw = adata.getattr("raw")?;
        if raw.is_none() {
            return Err(PyValueError::new_err("use_raw=True but adata.raw is None."));
        }
        let x = raw.getattr("X")?;
        let gene_names = var_names_of(&raw.getattr("var")?)?;
        Ok((x, gene_names))
    } else if let Some(name) = layer {
        let x = adata.getattr("layers")?.get_item(name).map_err(|_| {
            PyValueError::new_err(format!("layer '{name}' not found in adata.layers"))
        })?;
        let gene_names = var_names_of(&adata.getattr("var")?)?;
        Ok((x, gene_names))
    } else {
        let x = adata.getattr("X")?;
        let gene_names = var_names_of(&adata.getattr("var")?)?;
        Ok((x, gene_names))
    }
}

/// Runtime CSC-sidecar availability probe for the `prefer_format="auto"` policy.
///
/// Mirrors the single capability-detection point (`as_column_source`): a valid
/// CSC route needs a sidecar present, no active row-deletion vector, and — for a
/// Refuse an explicit `prefer_format="csc"` on a **subset** backed handle.
///
/// The gene-major sidecar is written against the full axis and has no
/// projection surface, so a subset handle reaches the CSC kernel with
/// visible-width `gene_names` (or a row count the sidecar cannot express).
/// The kernel does catch it, but as a bare
/// `gene_names length 15 != source.n_vars() 30` — say what actually happened.
///
/// Only the *explicit* CSC request lands here; `prefer_format="auto"` never
/// picks CSC for a subset handle (`csc_route_available` excludes a projected
/// one, and any `kept_to_global` makes `as_column_source()` return `None`).
fn reject_csc_on_subset(backed: &ScxBackedSparseDataset) -> PyResult<()> {
    if backed.kept_to_global.is_some() {
        return Err(PyRuntimeError::new_err(
            "CSC requested but unavailable: a row deletion vector is active \
             (this dataset has been subset along obs, e.g. by filter_cells). \
             Use prefer_format='csr'.",
        ));
    }
    if backed.col_projection_arc().is_some() {
        return Err(PyRuntimeError::new_err(
            "CSC requested but unavailable: a column projection is active \
             (this dataset has been subset along var, e.g. by filter_genes or \
             highly_variable_genes(subset=True)); the CSC sidecar is full-axis. \
             Use prefer_format='csr'.",
        ));
    }
    Ok(())
}

/// Runtime CSC-sidecar availability probe for the `prefer_format="auto"` policy.
///
/// Mirrors the single capability-detection point (`as_column_source`): a valid
/// CSC route needs a sidecar present, no active row-deletion vector, and — for a
/// lazy source — only column-local transforms. Never errors: a `false` result
/// just routes `auto` to the CSR streamer. A materialized matrix (numpy/scipy,
/// e.g. `use_raw`/`layer`) is not a backed/lazy SCX dataset → `false`.
fn csc_route_available(x: &Bound<'_, PyAny>) -> bool {
    if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        // `as_column_source` exposes the *full-axis* CSC reader and ignores an
        // active column projection (a gene subset, e.g. `adata[:, highly_variable]`
        // on a backed file that keeps its sidecar). Routing such a projected
        // dataset to the CSC kernel would trip its `n_vars` guard and raise,
        // where the CSR streamer read the projected columns fine. §5.2 lists
        // "filtering" among the `auto` gates — so exclude projected backed
        // datasets from CSC-direct (they fall back to CSR). The lazy path below
        // does not need this: its CSC reader honours the projection.
        return backed.col_projection_arc().is_none() && backed.as_column_source().is_some();
    }
    if let Ok(lazy) = x.extract::<PyRef<crate::lazy_transform::ScxLazyTransformedDataset>>() {
        // A materialized matrix (numpy/scipy from `use_raw`/`layer`) is neither
        // a backed nor a lazy SCX dataset, so it never reaches here → CSR.
        return lazy.as_column_source().is_some();
    }
    false
}

/// Resolve `prefer_format` to a concrete `"csr"` / `"csc"` route.
///
/// `"auto"` (the default since Phase-2 §5.2) picks the CSC-direct CPU route when
/// a valid CSC sidecar is available and the op runs on CPU; on GPU it stays
/// `"csr"` so the planner routes `gpu_csc_v3` from the CSR path when a sidecar
/// is present. Explicit `"csr"` / `"csc"` pass through unchanged.
fn resolve_de_format(
    prefer_format: &str,
    gpu_device_id: Option<usize>,
    x: &Bound<'_, PyAny>,
) -> &'static str {
    match prefer_format {
        "auto" => {
            if gpu_device_id.is_some() {
                "csr"
            } else if csc_route_available(x) {
                "csc"
            } else {
                "csr"
            }
        }
        "csc" => "csc",
        other => {
            // Callers validate `"auto"|"csr"|"csc"` upstream; a stray value here
            // means a new internal caller bypassed validation. Fail loud in debug,
            // fall back to the safe CSR streamer in release.
            debug_assert!(
                other == "csr",
                "resolve_de_format: unvalidated prefer_format {other:?}"
            );
            "csr"
        }
    }
}

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
    use_raw: bool,
    layer: Option<&str>,
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

    // Unknown groups (NaN / empty after astype("str") → "nan" / "") map to a
    // sentinel >= n_groups so the Wilcoxon kernels drop them, instead of
    // contaminating group 0. Mirrors resolve_groups_and_reference.
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

    // Select the input matrix + gene names per the use_raw/layer contract
    // (adata.X by default; adata.raw.X with raw var names for use_raw; a named
    // layer otherwise). The selected matrix flows through the same dispatch.
    let (x, gene_names) = select_de_matrix(adata, use_raw, layer)?;

    // Resolve the `"auto"` policy (§5.2) against the *selected* matrix: CSC-direct
    // on CPU when a valid sidecar is present, else CSR; CSR on GPU (the planner
    // routes gpu_csc_v3 from there). Explicit "csr"/"csc" pass through.
    let prefer_format = resolve_de_format(prefer_format, gpu_device_id, &x);

    // Auto-detect whether data has been log-transformed (sc.pp.log1p sets
    // adata.uns["log1p"]). When true, logFC uses expm1 back-transform to
    // match scanpy's formula.
    let log_transformed = adata
        .getattr("uns")?
        .call_method1("get", ("log1p",))
        .map(|v| !v.is_none())
        .unwrap_or(false);

    if prefer_format == "csc" {
        if gpu_device_id.is_some() {
            return Err(PyRuntimeError::new_err(
                "prefer_format='csc' selects the CPU column-major path and has no \
                 GPU kernel, so it cannot be combined with device='gpu'. For GPU \
                 CSC-direct DE (route gpu_csc_v3), keep prefer_format='csr' \
                 with device='gpu' (or 'auto'): when the file has a CSC sidecar the \
                 planner routes to gpu_csc_v3 automatically. For the CPU column-major \
                 path, use device='cpu'.",
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
            reject_csc_on_subset(&backed)?;
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
        // The handle's *view*, not the file: `groups` is one label per
        // *visible* cell and `gene_names` one per visible gene, so a subset
        // handle has to stream its window. `with_cached_reads` because the
        // kernel walks every shard once per gene chunk — the same LRU the raw
        // reader served from.
        let source = backed.as_shard_source().with_cached_reads();
        // Only the GPU arm still needs the concrete reader (for the
        // CSC-direct `Backed` input); the CPU arm runs entirely off `source`.
        #[cfg(feature = "gpu")]
        let reader = std::sync::Arc::clone(&backed.backed);
        #[cfg(feature = "gpu")]
        let has_view = backed.has_axis_view();
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
                    // Load-bearing, not bookkeeping. Neither of the CSC
                    // gates elsewhere protects this path: the `csc` below
                    // comes straight off `backed.backed_csc`, bypassing
                    // `as_column_source()`'s deletion check, and
                    // `csc_route_available` is never consulted on GPU
                    // (`resolve_de_format` short-circuits to "csr" whenever
                    // `gpu_device_id.is_some()`). So a subset handle with a
                    // sidecar would otherwise reach the CSC-direct kernel and
                    // read *on-disk* columns under visible-width
                    // `gene_names` — a silent wrong answer, since
                    // `Backed::shape()` takes `n_vars` from `gene_names.len()`
                    // and the widths agree. Route it to the generic `Lazy`
                    // input; only an unsubset handle keeps `Backed` and with
                    // it the CSC-direct `gpu_csc_v3` route.
                    let input = if has_view {
                        scx_accel::GpuDeShardInput::Lazy(&source)
                    } else {
                        scx_accel::GpuDeShardInput::Backed {
                            csr: &reader,
                            csc: csc_reader.as_deref(),
                        }
                    };
                    scx_accel::wilcoxon_rank_sum_gpu(
                        device_id,
                        input,
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
                        &source,
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
            let (csr_obj, _) = crate::convert::ensure_csr(py, &x, /* in_place */ false)?;
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
///
/// ``use_raw`` / ``layer`` select the analyzed matrix (scanpy semantics):
/// ``use_raw`` analyzes ``adata.raw.X`` (with ``adata.raw.var`` names),
/// ``layer`` analyzes ``adata.layers[layer]``, and they are mutually exclusive.
/// ``use_raw=None`` (default) resolves to ``True`` iff ``adata.raw`` exists and
/// ``layer`` is None, else ``False``. The resolved ``use_raw`` and ``layer`` are
/// written to ``adata.uns["rank_genes_groups"]["params"]``.
///
/// NOTE: the log-fold-change back-transform uses the ``adata.uns["log1p"]``
/// flag, which describes ``X``. For the conventional case (``.raw`` /
/// ``layer`` hold log-normalized data, like ``X``) this is correct; if ``.raw``
/// holds raw counts while ``X`` is log1p-transformed, the logFC is computed as
/// if the counts were log-space. Prefer ``pdex_ref`` (which exposes an explicit
/// ``is_log1p`` override) when analyzing a matrix whose transform state differs
/// from ``X``.
#[pyfunction]
#[pyo3(signature = (adata, groupby, reference="rest", n_genes=None, method="wilcoxon", gene_chunk_size=None, stratify_by=None, min_cells_per_stratum=50, rankby_abs=false, tie_correct=false, prefer_format="auto", device="auto", use_raw=None, layer=None))]
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
    use_raw: Option<bool>,
    layer: Option<&str>,
) -> PyResult<Py<PyAny>> {
    if method != "wilcoxon" {
        return Err(PyRuntimeError::new_err(format!(
            "unsupported method '{method}': only 'wilcoxon' is currently supported"
        )));
    }
    if !matches!(prefer_format, "csr" | "csc" | "auto") {
        return Err(PyValueError::new_err(format!(
            "Invalid prefer_format={prefer_format:?}; expected 'auto', 'csr', or 'csc'"
        )));
    }
    // A presentation-ordered backed `X` (`preserve_var_order=True`) has no
    // `ShardSource` spelling: the source emits columns in sorted on-disk order
    // while `adata.var` — and so `gene_names` — stays in request order. Now
    // that this op streams the handle's *view*, the two widths match, so the
    // mismatch would be a silent gene/column permutation instead of a shape
    // error. Refuse, as the other streaming accel ops do.
    super::reject_preserve_var_order(adata, "rank_genes_groups")?;

    // Resolve scanpy's use_raw/layer contract once (mutual-exclusion + default).
    let resolved_use_raw = resolve_use_raw(adata, use_raw, layer)?;
    let resolved = super::gpu::resolve_device(device)?;
    #[cfg(feature = "gpu")]
    let gpu_device_id = resolved.gpu_id();
    #[cfg(not(feature = "gpu"))]
    let gpu_device_id: Option<usize> = {
        let _ = resolved;
        None
    };
    // `prefer_format="csc"` selects the CPU column-major path (it is a CPU-only
    // knob — the GPU CSC-direct route `gpu_csc_v3` is reached via the *default*
    // prefer_format="csr", chosen by the planner when a CSC sidecar is present).
    // Reject when the user explicitly asked for GPU; for device="auto" fall back
    // to CPU, but nudge on a GPU host so the silent CPU pin is not surprising.
    let gpu_device_id = if prefer_format == "csc" {
        if device.starts_with("gpu") {
            return Err(PyRuntimeError::new_err(
                "prefer_format='csc' selects the CPU column-major path and has no \
                 GPU kernel, so it cannot be combined with device='gpu'. For GPU \
                 CSC-direct DE (route gpu_csc_v3), keep prefer_format='csr' \
                 with device='gpu' (or 'auto'): when the file has a CSC sidecar the \
                 planner routes to gpu_csc_v3 automatically. For the CPU column-major \
                 path, use device='cpu'.",
            ));
        }
        if device == "auto" && super::route::gpu_available() {
            py.import("warnings")?.call_method1(
                "warn",
                (
                    "rank_genes_groups(device=\"auto\", prefer_format=\"csc\") runs on the \
                     CPU: prefer_format=\"csc\" pins the CPU column-major path even on a GPU \
                     host. For GPU CSC-direct DE (route gpu_csc_v3), drop prefer_format \
                     (pass \"csr\" explicitly) with device=\"auto\"/\"gpu\" — the planner \
                     routes to gpu_csc_v3 automatically when a CSC sidecar is present.",
                    py.get_type::<pyo3::exceptions::PyUserWarning>(),
                ),
            )?;
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
                resolved_use_raw,
                layer,
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
        resolved_use_raw,
        layer,
    )?;

    // Write results to adata.uns["rank_genes_groups"] in scanpy format.
    write_de_to_adata(
        py,
        adata,
        &result,
        groupby,
        reference,
        n_genes,
        resolved_use_raw,
        layer,
    )?;

    // Record the accelerator execution route: both inside the scanpy-style
    // rank_genes_groups dict (as `scx_accel_route`) and under the unified
    // adata.uns["scx_accel"]["rank_genes_groups"] lookup. `result.exec_info`
    // is already complete (route + reason) — set by the single planner on both
    // the CPU dispatch sites and inside scx-accel for GPU routes.
    if let Ok(rgg) = adata.getattr("uns")?.get_item("rank_genes_groups") {
        rgg.set_item("scx_accel_route", result.exec_info.route.as_str())?;
    }
    super::route::announce_route(py, "rank_genes_groups", device, &result.exec_info);
    super::route::warn_materialized_csc_sidecar(
        py,
        "rank_genes_groups",
        device,
        adata,
        &result.exec_info,
    );
    super::route::write_accel_route(py, adata, "rank_genes_groups", &result.exec_info)?;

    Ok(py.None())
}

/// Write DE results to adata.uns["rank_genes_groups"] matching scanpy's format.
///
/// Scanpy stores results as numpy structured arrays (rec.arrays) with one
/// field per group. Each field contains gene names/scores/p-values sorted
/// by the test statistic.
#[allow(clippy::too_many_arguments)]
fn write_de_to_adata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    result: &scx_accel::DiffExpResult,
    groupby: &str,
    reference: &str,
    n_genes: Option<usize>,
    use_raw: bool,
    layer: Option<&str>,
) -> PyResult<()> {
    let numpy = py.import("numpy")?;
    // The builder closures below iterate `result.group_names` (length n_groups)
    // and index `field_data[i]`, so every outer field vector must have at least
    // n_groups rows or that indexing panics. Rectangular by construction, but
    // validate explicitly so a malformed DiffExpResult returns an error instead
    // of panicking (the `full_n_genes` zip only bounds the *inner* lengths).
    let n_groups = result.group_names.len();
    if result.names.len() < n_groups
        || result.scores.len() < n_groups
        || result.pvals.len() < n_groups
        || result.pvals_adj.len() < n_groups
        || result.logfoldchanges.len() < n_groups
    {
        return Err(PyRuntimeError::new_err(
            "DiffExpResult has fewer per-field rows than groups (malformed result)",
        ));
    }
    // Rank depth = the SHORTEST per-group vector across all emitted fields, not
    // just group 0's. Groups are rectangular by construction
    // (`[n_groups][n_genes]`), so this is a no-op on well-formed results, but
    // clamping to the min keeps the numpy structured array rectangular and every
    // `[..n_genes]` slice in the builder closures below in-bounds even on a
    // malformed/degenerate DiffExpResult — avoids a PanicException. The zip also
    // stops at the shortest outer vec.
    let full_n_genes = result
        .names
        .iter()
        .zip(&result.scores)
        .zip(&result.pvals)
        .zip(&result.pvals_adj)
        .zip(&result.logfoldchanges)
        .map(|((((n, s), p), pa), l)| n.len().min(s.len()).min(p.len()).min(pa.len()).min(l.len()))
        .min()
        .unwrap_or(0);
    let n_genes = n_genes.unwrap_or(full_n_genes).min(full_n_genes);

    let rgg = PyDict::new(py);

    // params dict
    let params = PyDict::new(py);
    params.set_item("groupby", groupby)?;
    params.set_item("reference", reference)?;
    params.set_item("method", "wilcoxon")?;
    params.set_item("use_raw", use_raw)?;
    params.set_item("layer", layer)?;

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

/// Build a DataFrame of the given `columns` in `column_order`, as either a
/// polars or a pandas DataFrame depending on `output`.
///
/// `output="polars"` (default) requires polars; `output="pandas"` builds a
/// pandas DataFrame directly and does **not** import polars, so pandas-only
/// callers can use the DE DataFrame helpers without installing polars. Column
/// schema is identical across both.
pub(super) fn build_de_dataframe<'py>(
    py: Python<'py>,
    columns: &Bound<'py, PyDict>,
    column_order: &[&str],
    output: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let order = pyo3::types::PyList::new(py, column_order)?;
    match output {
        "polars" => {
            let pl = py.import("polars").map_err(|_| {
                PyRuntimeError::new_err(
                    "output='polars' requires polars. Install it \
                     (pip install 'pyscx[eval]' or pip install polars), \
                     or pass output='pandas'.",
                )
            })?;
            let df = pl.call_method1("DataFrame", (columns,))?;
            // Enforce column order regardless of dict iteration / constructor.
            df.call_method1("select", (order,))
        }
        "pandas" => {
            let pd = py.import("pandas").map_err(|_| {
                PyRuntimeError::new_err("output='pandas' requires pandas (pip install pandas).")
            })?;
            let df = pd.call_method1("DataFrame", (columns,))?;
            // `df[[col, ...]]` selects and orders columns deterministically.
            df.get_item(order)
        }
        other => Err(PyValueError::new_err(format!(
            "Invalid output={other:?}; expected 'polars' or 'pandas'"
        ))),
    }
}

/// Convert a DiffExpResult into a DataFrame matching cell-eval's `DEResults`
/// schema, as polars or pandas per `output`.
fn de_result_to_cell_eval_dataframe<'py>(
    py: Python<'py>,
    result: &scx_accel::DiffExpResult,
    n_genes: Option<usize>,
    output: &str,
) -> PyResult<Bound<'py, PyAny>> {
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

    // Build column vectors into a dict; `build_de_dataframe` constructs the
    // polars/pandas frame and enforces the cell-eval DEResults column order.
    let dict = PyDict::new(py);
    dict.set_item("target", targets)?;
    dict.set_item("feature", features)?;
    dict.set_item("fold_change", fold_changes)?;
    dict.set_item("p_value", p_values)?;
    dict.set_item("fdr", fdrs)?;
    dict.set_item("log2_fold_change", log2_fcs)?;
    dict.set_item("abs_log2_fold_change", abs_log2_fcs)?;

    build_de_dataframe(
        py,
        &dict,
        &[
            "target",
            "feature",
            "fold_change",
            "p_value",
            "fdr",
            "log2_fold_change",
            "abs_log2_fold_change",
        ],
        output,
    )
}

/// Extract a scanpy-style DE DataFrame from a precomputed
/// `adata.uns[key]` (the `sc.get.rank_genes_groups_df` alias). Reads the
/// scanpy-format structured arrays written by `rank_genes_groups`; never
/// recomputes. Returns scanpy's columns (`names, scores, logfoldchanges,
/// pvals, pvals_adj`), with a leading `group` column when `group` is a list.
#[allow(clippy::too_many_arguments)]
fn extract_rank_genes_groups_df<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    group: &Bound<'py, PyAny>,
    key: &str,
    n_genes: Option<usize>,
    pval_cutoff: Option<f64>,
    log2fc_min: Option<f64>,
    log2fc_max: Option<f64>,
    output: &str,
) -> PyResult<Bound<'py, PyAny>> {
    // `group` is a single name (no `group` column, matching scanpy) or a list
    // of names (with a `group` column).
    let (groups, multi): (Vec<String>, bool) = if let Ok(s) = group.extract::<String>() {
        (vec![s], false)
    } else if let Ok(v) = group.extract::<Vec<String>>() {
        if v.is_empty() {
            return Err(PyValueError::new_err(
                "group must be a non-empty group name (str) or list of names",
            ));
        }
        (v, true)
    } else {
        return Err(PyValueError::new_err(
            "group must be a group name (str) or a list of group names",
        ));
    };

    let rgg = adata.getattr("uns")?.get_item(key).map_err(|_| {
        PyValueError::new_err(format!(
            "adata.uns[{key:?}] not found — run pyscx.accel.rank_genes_groups(adata, groupby=...) \
             to populate it, or pass groupby= to compute DE here"
        ))
    })?;

    // Available group names are the structured-array field names of `names`.
    let names_arr = rgg.get_item("names")?;
    let available: Vec<String> = names_arr
        .getattr("dtype")?
        .getattr("names")?
        .extract()
        .unwrap_or_default();
    for g in &groups {
        if !available.contains(g) {
            return Err(PyValueError::new_err(format!(
                "group {g:?} not in adata.uns[{key:?}]; available: {available:?}"
            )));
        }
    }

    // Read one structured-array field for one group → Vec, via `.tolist()`.
    let read_str = |field: &str, g: &str| -> PyResult<Vec<String>> {
        rgg.get_item(field)?
            .get_item(g)?
            .call_method0("tolist")?
            .extract()
    };
    let read_f64 = |field: &str, g: &str| -> PyResult<Vec<f64>> {
        rgg.get_item(field)?
            .get_item(g)?
            .call_method0("tolist")?
            .extract()
    };

    let mut col_group: Vec<String> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut scores: Vec<f64> = Vec::new();
    let mut logfoldchanges: Vec<f64> = Vec::new();
    let mut pvals: Vec<f64> = Vec::new();
    let mut pvals_adj: Vec<f64> = Vec::new();

    for g in &groups {
        let g_names = read_str("names", g)?;
        let g_scores = read_f64("scores", g)?;
        let g_lfc = read_f64("logfoldchanges", g)?;
        let g_pvals = read_f64("pvals", g)?;
        let g_padj = read_f64("pvals_adj", g)?;
        let full = g_names.len();
        // The five fields are read independently; a malformed / hand-edited
        // `uns` with mismatched lengths would otherwise index out of bounds
        // (a Rust panic that crashes the interpreter). Fail cleanly instead.
        if g_scores.len() != full
            || g_lfc.len() != full
            || g_pvals.len() != full
            || g_padj.len() != full
        {
            return Err(PyValueError::new_err(format!(
                "malformed adata.uns[{key:?}] for group {g:?}: field lengths differ \
                 (names={full}, scores={}, logfoldchanges={}, pvals={}, pvals_adj={})",
                g_scores.len(),
                g_lfc.len(),
                g_pvals.len(),
                g_padj.len()
            )));
        }
        let n = n_genes.unwrap_or(full).min(full);
        for i in 0..n {
            // scanpy-style row filters (only applied when set). Positive
            // comparisons mean NaN rows fail the predicate and are dropped,
            // matching scanpy's `df[df[col] < cutoff]` semantics
            // (scanpy/get/get.py uses strict `<` / `>` / `<`).
            let keep = pval_cutoff.is_none_or(|c| g_padj[i] < c)
                && log2fc_min.is_none_or(|m| g_lfc[i] > m)
                && log2fc_max.is_none_or(|m| g_lfc[i] < m);
            if !keep {
                continue;
            }
            if multi {
                col_group.push(g.clone());
            }
            names.push(g_names[i].clone());
            scores.push(g_scores[i]);
            logfoldchanges.push(g_lfc[i]);
            pvals.push(g_pvals[i]);
            pvals_adj.push(g_padj[i]);
        }
    }

    let dict = PyDict::new(py);
    if multi {
        dict.set_item("group", col_group)?;
    }
    dict.set_item("names", names)?;
    dict.set_item("scores", scores)?;
    dict.set_item("logfoldchanges", logfoldchanges)?;
    dict.set_item("pvals", pvals)?;
    dict.set_item("pvals_adj", pvals_adj)?;

    let column_order: &[&str] = if multi {
        &[
            "group",
            "names",
            "scores",
            "logfoldchanges",
            "pvals",
            "pvals_adj",
        ]
    } else {
        &["names", "scores", "logfoldchanges", "pvals", "pvals_adj"]
    };
    build_de_dataframe(py, &dict, column_order, output)
}

/// Differential-expression DataFrame — **two modes**, selected by which kwarg
/// you pass.
///
/// **Compute (`groupby=`)** — re-runs Wilcoxon rank-sum DE and returns a polars
/// (or pandas) DataFrame in cell-eval's `DEResults` format. This is the format
/// bridge between SCX's Wilcoxon DE and cell-eval's DE metric pipeline; the
/// frame can be fed directly into `cell_eval.data.DEResults` /
/// `cell_eval.data.DEComparison`. The accelerator execution route is recorded
/// on `adata.uns["scx_accel"]["rank_genes_groups_df"]`. Columns:
///   - `target` (str): perturbation/group name
///   - `feature` (str): gene name
///   - `fold_change` (f64): linear fold change (2^log2FC)
///   - `p_value` (f64): raw p-value
///   - `fdr` (f64): BH-adjusted p-value
///   - `log2_fold_change` (f64): log2 fold change
///   - `abs_log2_fold_change` (f64): |log2FC|
///
/// **Extract (`group=`)** — the scanpy `sc.get.rank_genes_groups_df` alias: does
/// **not** recompute; reads the precomputed `adata.uns[key]` (written by
/// `pyscx.accel.rank_genes_groups`) and returns scanpy's native columns
/// (`names, scores, logfoldchanges, pvals, pvals_adj`), with a leading `group`
/// column when `group` is a list. Optional scanpy filters `pval_cutoff` /
/// `log2fc_min` / `log2fc_max` apply. (`gene_symbols=` var-name remapping is not
/// supported yet.) Pass either `groupby=` or `group=`, not both. To extract
/// **all** groups, pass the list of names
/// (`group=list(adata.uns[key]["names"].dtype.names)`); `group=None` routes to
/// the compute path.
///
/// Args:
///     adata: AnnData object with X and obs[groupby]
///     groupby: obs column to group cells by (compute mode)
///     reference: Group to compare against (default: "rest" = 1-vs-rest)
///     n_genes: Number of top genes per group (default: all genes). In extract
///         mode this is a pyscx extension (scanpy's extractor has no `n_genes`):
///         it truncates to top-N *before* the `pval_cutoff` / `log2fc_*` filters.
///     gene_chunk_size: Genes per chunk for streaming DE (default: None → 500
///         internally for sparse/backed inputs)
///     rankby_abs: Sort genes by |score| instead of signed score (default: False)
///     tie_correct: Apply tie correction in the Wilcoxon test (default: False)
///     output: `"polars"` (default) or `"pandas"`. Identical columns either way;
///         `"pandas"` does not require polars.
///     device: compute-mode only; ignored in extract (`group=`) mode.
///     group: extraction mode — a group name (str) or list of names to pull from
///         `adata.uns[key]`.
///     key: uns key to extract from (default: `"rank_genes_groups"`).
///     pval_cutoff / log2fc_min / log2fc_max: scanpy-style row filters (extraction
///         mode only): keep rows with `pvals_adj < pval_cutoff`,
///         `logfoldchanges > log2fc_min`, `logfoldchanges < log2fc_max`.
///
/// Example:
///     de_df = pyscx.accel.rank_genes_groups_df(adata, "perturbation")  # compute
///     ex = pyscx.accel.rank_genes_groups_df(adata, group="0")          # extract (scanpy-style)
#[pyfunction]
#[pyo3(signature = (adata, groupby=None, reference="rest", n_genes=None, gene_chunk_size=None, rankby_abs=false, tie_correct=false, device="auto", output="polars", *, group=None, key="rank_genes_groups", pval_cutoff=None, log2fc_min=None, log2fc_max=None))]
#[allow(clippy::too_many_arguments)]
pub fn rank_genes_groups_df(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    groupby: Option<&str>,
    reference: &str,
    n_genes: Option<usize>,
    gene_chunk_size: Option<usize>,
    rankby_abs: bool,
    tie_correct: bool,
    device: &str,
    output: &str,
    group: Option<&Bound<'_, PyAny>>,
    key: &str,
    pval_cutoff: Option<f64>,
    log2fc_min: Option<f64>,
    log2fc_max: Option<f64>,
) -> PyResult<Py<PyAny>> {
    if !matches!(output, "polars" | "pandas") {
        return Err(PyValueError::new_err(format!(
            "Invalid output={output:?}; expected 'polars' or 'pandas'"
        )));
    }

    // Extraction mode (scanpy `sc.get.rank_genes_groups_df` alias): when
    // `group=` is given, read precomputed results from `adata.uns[key]` instead
    // of recomputing. Mutually exclusive with the compute path's `groupby=`.
    if let Some(group) = group {
        if groupby.is_some() {
            return Err(PyValueError::new_err(
                "pass either groupby= (compute DE) or group= (extract precomputed \
                 adata.uns[...]), not both",
            ));
        }
        let df = extract_rank_genes_groups_df(
            py,
            adata,
            group,
            key,
            n_genes,
            pval_cutoff,
            log2fc_min,
            log2fc_max,
            output,
        )?;
        return Ok(df.unbind());
    }

    let groupby = groupby.ok_or_else(|| {
        PyValueError::new_err(
            "rank_genes_groups_df needs groupby= (to compute DE) or group= (to extract \
             precomputed adata.uns[\"rank_genes_groups\"]); got neither",
        )
    })?;

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
        false, // use_raw: this cell-eval bridge is X-only
        None,  // layer
    )?;

    // Record the accelerator execution route on adata.uns; the returned
    // polars DataFrame carries no metadata of its own. `result.exec_info` is
    // already complete (route + reason) from the single planner.
    super::route::announce_route(py, "rank_genes_groups_df", device, &result.exec_info);
    super::route::warn_materialized_csc_sidecar(
        py,
        "rank_genes_groups_df",
        device,
        adata,
        &result.exec_info,
    );
    super::route::write_accel_route(py, adata, "rank_genes_groups_df", &result.exec_info)?;

    let df = de_result_to_cell_eval_dataframe(py, &result, n_genes, output)?;
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
    cpm_filter: Option<f64>,
    gene_chunk_size: Option<usize>,
    prefer_format: &str,
    device: &str,
    gpu_device_id: Option<usize>,
    use_raw: bool,
    layer: Option<&str>,
) -> PyResult<scx_accel::PdexRefResult> {
    let numpy = py.import("numpy")?;
    let scipy_sparse = py.import("scipy.sparse")?;

    let (groups, unique_groups, ref_idx) = resolve_groups_and_reference(adata, groupby, reference)?;

    // Select the input matrix + gene names per the use_raw/layer contract.
    let (x, gene_names) = select_de_matrix(adata, use_raw, layer)?;

    // Resolve the `"auto"` policy (§5.2): CSC-direct on CPU when a valid sidecar
    // is present, else CSR; CSR on GPU (planner routes gpu_csc_v3 from there).
    let prefer_format = resolve_de_format(prefer_format, gpu_device_id, &x);

    let resolved_log1p = match is_log1p {
        Some(v) => v,
        None => detect_is_log1p(py, adata)?,
    };
    let mode = scx_accel::GeomMeanMode::from_flags(geometric_mean, resolved_log1p);

    if prefer_format == "csc" {
        if gpu_device_id.is_some() {
            return Err(PyRuntimeError::new_err(
                "prefer_format='csc' selects the CPU column-major path and has no \
                 GPU kernel, so it cannot be combined with device='gpu'. For GPU \
                 CSC-direct DE (route gpu_csc_v3), keep prefer_format='csr' \
                 with device='gpu' (or 'auto'): when the file has a CSC sidecar the \
                 planner routes to gpu_csc_v3 automatically. For the CPU column-major \
                 path, use device='cpu'.",
            ));
        }
        let chunk_size = gene_chunk_size.unwrap_or(500);

        if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
            reject_csc_on_subset(&backed)?;
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
                        cpm_filter,
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
                        cpm_filter,
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
        // The handle's *view* — see the matching branch in
        // `run_rank_genes_groups_inner` for why the raw reader is wrong here.
        let source = backed.as_shard_source().with_cached_reads();
        // GPU-only: the concrete reader backs the CSC-direct `Backed` input.
        #[cfg(feature = "gpu")]
        let reader = std::sync::Arc::clone(&backed.backed);
        #[cfg(feature = "gpu")]
        let has_view = backed.has_axis_view();
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
                    // Subset handle → generic `Lazy`; unsubset → `Backed`,
                    // preserving the CSC-direct `gpu_csc_v3` route. See the
                    // matching branch in `run_rank_genes_groups_inner`: this
                    // switch is what stops a subset handle running the
                    // CSC-direct kernel against on-disk columns.
                    let input = if has_view {
                        scx_accel::GpuDeShardInput::Lazy(&source)
                    } else {
                        scx_accel::GpuDeShardInput::Backed {
                            csr: &reader,
                            csc: csc_reader.as_deref(),
                        }
                    };
                    scx_accel::pdex_ref_gpu(
                        device_id,
                        input,
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        Some(chunk_size),
                        mode,
                        epsilon,
                        cpm_filter,
                    )
                })
                .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string())),
            #[cfg(not(feature = "gpu"))]
            Some(_) => unreachable!("gpu_device_id is None when gpu feature is disabled"),
            None => py
                .detach(|| {
                    scx_accel::pdex_ref_streaming(
                        &source,
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        chunk_size,
                        mode,
                        epsilon,
                        cpm_filter,
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
                        cpm_filter,
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
        let (csr_obj, _) = crate::convert::ensure_csr(py, &x, /* in_place */ false)?;
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
                        cpm_filter,
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
                        cpm_filter,
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
                        cpm_filter,
                    )
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string())),
            #[cfg(not(feature = "gpu"))]
            Some(_) => unreachable!("gpu_device_id is None when gpu feature is disabled"),
            None => py
                .detach(|| {
                    let mut r = scx_accel::pdex_ref_core(
                        &data,
                        n_obs,
                        n_vars,
                        &gene_names,
                        &groups,
                        &unique_groups,
                        ref_idx,
                        mode,
                        epsilon,
                        cpm_filter.is_some(),
                    )?;
                    // The dense kernel is not chunked, so apply the CPM filter +
                    // survivor-scoped FDR here (no-op when cpm_filter is None,
                    // beyond the redundant BH recompute we skip by guarding).
                    if cpm_filter.is_some() {
                        scx_accel::finalize_pdex(&mut r, cpm_filter);
                    }
                    Ok::<_, scx_accel::AccelError>(r)
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

/// Best-effort check for negative entries in `adata.X`, used only to warn when
/// `cpm_filter` is set (CPM assumes non-negative counts). Cheap in-memory paths
/// only: scipy sparse (`.data.min()`) and numpy ndarray (`.min()`). Backed/lazy
/// SCX datasets are skipped — SCX stores non-negative counts by construction, so
/// matching pdex's backed-array sampling here is unnecessary.
fn pdex_x_has_negative(py: Python<'_>, adata: &Bound<'_, PyAny>) -> bool {
    let Ok(x) = adata.getattr("X") else {
        return false;
    };
    if let Ok(scipy_sparse) = py.import("scipy.sparse") {
        if let Ok(true) = scipy_sparse
            .call_method1("issparse", (&x,))
            .and_then(|r| r.extract::<bool>())
        {
            return x
                .getattr("data")
                .and_then(|d| d.call_method0("min"))
                .and_then(|m| m.extract::<f64>())
                .map(|m| m < 0.0)
                .unwrap_or(false);
        }
    }
    if let Ok(numpy) = py.import("numpy") {
        if let Ok(ndarray_ty) = numpy.getattr("ndarray") {
            if let Ok(true) = x.is_instance(&ndarray_ty) {
                return x
                    .call_method0("min")
                    .and_then(|m| m.extract::<f64>())
                    .map(|m| m < 0.0)
                    .unwrap_or(false);
            }
        }
    }
    false
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
    output: &str,
) -> PyResult<Bound<'py, PyAny>> {
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
        // When `cpm_filter` dropped rows, `kept_indices[tg]` maps each surviving
        // per-group row back to its gene in the shared `feature_names` /
        // `ref_means` axis; the per-group vectors are already compacted to the
        // survivors. Without filtering, every group spans the full gene axis.
        let n_rows = result.target_means[tg].len();
        targets.extend(std::iter::repeat_n(group_name.clone(), n_rows));
        match result.kept_indices.as_ref() {
            Some(kept) => {
                for &gi in &kept[tg] {
                    features.push(result.feature_names[gi].clone());
                    ref_means.push(result.ref_means[gi]);
                }
            }
            None => {
                features.extend(result.feature_names.iter().cloned());
                ref_means.extend(result.ref_means.iter().copied());
            }
        }
        target_means.extend(result.target_means[tg].iter().copied());
        target_memberships.extend(std::iter::repeat_n(
            result.target_memberships[tg] as u64,
            n_rows,
        ));
        ref_memberships.extend(std::iter::repeat_n(result.ref_membership as u64, n_rows));
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

    build_de_dataframe(
        py,
        &dict,
        &[
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
        output,
    )
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
///     epsilon: Pseudocount added to target_mean and ref_mean (count space)
///         before computing log2_fold_change and percent_change — NOT applied to
///         CPM or the Mann-Whitney U test. Default 1e-9 (a finite-guard matching
///         pdex >= 0.2.x): denominators stay strictly positive so no ±inf/NaN
///         arises. Pass epsilon=0.0 to recover the legacy behaviour where genes
///         undetected in the reference yield ±inf (0/0 collapses to 0.0 either
///         way).
///     cpm_filter: Optional per-gene expression floor T (counts-per-million).
///         When set, a gene is kept for a test group iff its pooled (arithmetic,
///         count-space) target CPM > T OR the reference CPM > T (strict >);
///         dropped rows are removed and FDR is recomputed over the surviving
///         genes only. The CPM view is mode-independent (always arithmetic) and
///         never reported. None (default) disables filtering.
///     gene_chunk_size: Genes per chunk for sparse/backed streaming
///         (default: 500). Ignored for dense input.
///     output: Return type — `"polars"` (default) or `"pandas"`. Columns are
///         identical either way; `"pandas"` builds a pandas DataFrame directly and
///         does not require polars. (A polars result also supports `.to_pandas()`.)
///     use_raw: Analyze `adata.raw.X` (with `adata.raw.var` names) instead of
///         `adata.X`. `None` (default) → `True` iff `adata.raw` is present and
///         `layer` is None (scanpy semantics), else `False`. Mutually exclusive
///         with `layer`. Recorded on `adata.uns["scx_accel"]["pdex_ref"]`.
///     layer: Analyze `adata.layers[layer]` instead of `adata.X`. Mutually
///         exclusive with `use_raw=True`.
///
/// The accelerator execution route is recorded on
/// ``adata.uns["scx_accel"]["pdex_ref"]`` (keys: ``route``,
/// ``fallback_reason``, ``chunk_size``, ``csc_available``, ...). The
/// CSC-direct GPU route (``route == "gpu_csc_v3"``) requires a backed SCX file
/// with a CSC sidecar (v3 is the default GPU DE route); in-memory CSR inputs
/// fall back to ``"gpu_csr_v3"`` with ``fallback_reason == "no_csc_sidecar"``. Check
/// ``route`` when comparing performance.
#[pyfunction]
#[pyo3(signature = (adata, groupby, *, reference="non-targeting", is_log1p=None, geometric_mean=true, epsilon=1e-9, cpm_filter=None, gene_chunk_size=None, prefer_format="auto", device="auto", output="polars", use_raw=None, layer=None))]
#[allow(clippy::too_many_arguments)]
pub fn pdex_ref(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    groupby: &str,
    reference: &str,
    is_log1p: Option<bool>,
    geometric_mean: bool,
    epsilon: f64,
    cpm_filter: Option<f64>,
    gene_chunk_size: Option<usize>,
    prefer_format: &str,
    device: &str,
    output: &str,
    use_raw: Option<bool>,
    layer: Option<&str>,
) -> PyResult<Py<PyAny>> {
    if epsilon < 0.0 || !epsilon.is_finite() {
        return Err(PyValueError::new_err(format!(
            "epsilon must be non-negative and finite (got {epsilon})"
        )));
    }
    if let Some(t) = cpm_filter {
        if !t.is_finite() {
            return Err(PyValueError::new_err(format!(
                "cpm_filter must be finite (got {t})"
            )));
        }
        // pdex warns (does not error) when counts contain negatives, since CPM
        // assumes non-negative expression. Only checked for cheap in-memory X.
        if pdex_x_has_negative(py, adata) {
            py.import("warnings")?.call_method1(
                "warn",
                (
                    "cpm_filter is set but adata.X contains negative values; \
                     counts-per-million assumes non-negative expression, so the \
                     filter may behave unexpectedly.",
                    py.get_type::<pyo3::exceptions::PyUserWarning>(),
                ),
            )?;
        }
    }
    if !matches!(prefer_format, "csr" | "csc" | "auto") {
        return Err(PyValueError::new_err(format!(
            "Invalid prefer_format={prefer_format:?}; expected 'auto', 'csr', or 'csc'"
        )));
    }
    // A presentation-ordered backed `X` (`preserve_var_order=True`) has no
    // `ShardSource` spelling: the source emits columns in sorted on-disk order
    // while `adata.var` — and so `gene_names` — stays in request order. Now
    // that this op streams the handle's *view*, the two widths match, so the
    // mismatch would be a silent gene/column permutation instead of a shape
    // error. Refuse, as the other streaming accel ops do.
    super::reject_preserve_var_order(adata, "pdex_ref")?;

    if !matches!(output, "polars" | "pandas") {
        return Err(PyValueError::new_err(format!(
            "Invalid output={output:?}; expected 'polars' or 'pandas'"
        )));
    }
    // Resolve scanpy's use_raw/layer contract once (mutual-exclusion + default).
    let resolved_use_raw = resolve_use_raw(adata, use_raw, layer)?;
    let resolved = super::gpu::resolve_device(device)?;
    #[cfg(feature = "gpu")]
    let gpu_device_id = resolved.gpu_id();
    #[cfg(not(feature = "gpu"))]
    let gpu_device_id: Option<usize> = {
        let _ = resolved;
        None
    };
    // `prefer_format="csc"` selects the CPU column-major path (it is a CPU-only
    // knob — the GPU CSC-direct route `gpu_csc_v3` is reached via the *default*
    // prefer_format="csr", chosen by the planner when a CSC sidecar is present).
    // Reject when the user explicitly asked for GPU; for device="auto" fall back
    // to CPU, but nudge on a GPU host so the silent CPU pin is not surprising.
    let gpu_device_id = if prefer_format == "csc" {
        if device.starts_with("gpu") {
            return Err(PyRuntimeError::new_err(
                "prefer_format='csc' selects the CPU column-major path and has no \
                 GPU kernel, so it cannot be combined with device='gpu'. For GPU \
                 CSC-direct DE (route gpu_csc_v3), keep prefer_format='csr' \
                 with device='gpu' (or 'auto'): when the file has a CSC sidecar the \
                 planner routes to gpu_csc_v3 automatically. For the CPU column-major \
                 path, use device='cpu'.",
            ));
        }
        if device == "auto" && super::route::gpu_available() {
            py.import("warnings")?.call_method1(
                "warn",
                (
                    "pdex_ref(device=\"auto\", prefer_format=\"csc\") runs on the CPU: \
                     prefer_format=\"csc\" pins the CPU column-major path even on a GPU \
                     host. For GPU CSC-direct DE (route gpu_csc_v3), drop prefer_format \
                     (pass \"csr\" explicitly) with device=\"auto\"/\"gpu\" — the planner \
                     routes to gpu_csc_v3 automatically when a CSC sidecar is present.",
                    py.get_type::<pyo3::exceptions::PyUserWarning>(),
                ),
            )?;
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
        cpm_filter,
        gene_chunk_size,
        prefer_format,
        device,
        gpu_device_id,
        resolved_use_raw,
        layer,
    )?;
    // Record the accelerator execution route on adata.uns["scx_accel"]["pdex_ref"].
    // `result.exec_info` is already complete (route + reason) from the single
    // planner — CPU sites via `cpu_exec_info`, GPU routes inside scx-accel.
    super::route::announce_route(py, "pdex_ref", device, &result.exec_info);
    super::route::warn_materialized_csc_sidecar(py, "pdex_ref", device, adata, &result.exec_info);
    super::route::write_accel_route(py, adata, "pdex_ref", &result.exec_info)?;
    // Record the resolved data-selection contract alongside the route so callers
    // can see which matrix was analyzed (X / raw.X / a layer).
    // Defensive: `get_item` with `?` would raise KeyError if either key is
    // absent. `write_accel_route` just created both, but check both with
    // `if let Ok(...)` so a missing route entry never crashes the op.
    if let Ok(scx_accel) = adata.getattr("uns")?.get_item("scx_accel") {
        if let Ok(entry) = scx_accel.get_item("pdex_ref") {
            entry.set_item("use_raw", resolved_use_raw)?;
            entry.set_item("layer", layer)?;
        }
    }
    let df = pdex_ref_result_to_dataframe(py, &result, output)?;
    Ok(df.unbind())
}
