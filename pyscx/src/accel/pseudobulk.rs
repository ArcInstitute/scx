//! Pseudobulk differential expression via Rust aggregation + pydeseq2.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::ScxBackedSparseDataset;

use super::de::extract_strata;

/// Upper bound for the env-derived / auto-probed worker count. An explicit
/// `n_cpus` argument bypasses this so callers can opt into larger pools.
const DESEQ_DEFAULT_MAX_CPUS: usize = 8;

/// Resolve a safe worker cap for pydeseq2's loky inference backend.
///
/// Precedence: explicit `n_cpus` (if `> 0`, used verbatim — the opt-in to a
/// larger pool) → `SLURM_CPUS_PER_TASK` → `OMP_NUM_THREADS` →
/// `available_parallelism`. Every non-explicit source is clamped to
/// `[1, DESEQ_DEFAULT_MAX_CPUS]`, so a large SLURM allocation does not silently
/// re-create the OOM below.
///
/// pydeseq2's `DefaultInference` defaults to `n_cpus=None`, which spawns one
/// loky worker process per core. Each worker is a full Python interpreter
/// (~300 MB RSS once numba/scipy are imported), so on a 192-core node the
/// pool alone needs ~58 GB and OOM-kills the job even for tiny inputs.
fn resolve_deseq_n_cpus(n_cpus: Option<usize>) -> usize {
    resolve_deseq_n_cpus_impl(
        n_cpus,
        std::env::var("SLURM_CPUS_PER_TASK").ok(),
        std::env::var("OMP_NUM_THREADS").ok(),
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
    )
}

/// Pure core of [`resolve_deseq_n_cpus`] — env/probe values are passed in so
/// the precedence logic is testable without mutating process state.
fn resolve_deseq_n_cpus_impl(
    n_cpus: Option<usize>,
    slurm_cpus: Option<String>,
    omp_threads: Option<String>,
    available_parallelism: usize,
) -> usize {
    if let Some(n) = n_cpus {
        if n > 0 {
            return n;
        }
    }
    for v in [slurm_cpus, omp_threads].into_iter().flatten() {
        if let Ok(n) = v.trim().parse::<usize>() {
            if n > 0 {
                return n.clamp(1, DESEQ_DEFAULT_MAX_CPUS);
            }
        }
    }
    available_parallelism.clamp(1, DESEQ_DEFAULT_MAX_CPUS)
}

#[cfg(test)]
mod tests {
    use super::resolve_deseq_n_cpus_impl;

    #[test]
    fn explicit_n_cpus_wins_over_env_and_probe() {
        assert_eq!(
            resolve_deseq_n_cpus_impl(Some(3), Some("64".into()), Some("64".into()), 192),
            3
        );
    }

    #[test]
    fn slurm_cpus_takes_precedence_over_omp() {
        assert_eq!(
            resolve_deseq_n_cpus_impl(None, Some("4".into()), Some("64".into()), 192),
            4
        );
    }

    #[test]
    fn omp_used_when_slurm_absent() {
        assert_eq!(
            resolve_deseq_n_cpus_impl(None, None, Some("6".into()), 192),
            6
        );
    }

    #[test]
    fn falls_back_to_probe_clamped_to_8() {
        // Explicit 0 and unparseable/zero env are all ignored.
        assert_eq!(
            resolve_deseq_n_cpus_impl(Some(0), Some("0".into()), None, 192),
            8
        );
        assert_eq!(
            resolve_deseq_n_cpus_impl(None, Some("x".into()), None, 4),
            4
        );
        assert_eq!(resolve_deseq_n_cpus_impl(None, None, None, 1), 1);
    }

    #[test]
    fn env_derived_values_clamp_to_max() {
        // A large SLURM/OMP allocation must not re-create the loky OOM.
        assert_eq!(
            resolve_deseq_n_cpus_impl(None, Some("192".into()), None, 4),
            8
        );
        assert_eq!(
            resolve_deseq_n_cpus_impl(None, None, Some("64".into()), 4),
            8
        );
    }

    #[test]
    fn explicit_n_cpus_is_not_clamped() {
        // Explicit value is the opt-in to a larger pool.
        assert_eq!(resolve_deseq_n_cpus_impl(Some(64), None, None, 4), 64);
    }
}

/// Aggregate single-cell counts into pseudobulk samples, dispatching across the
/// CSC-sidecar / backed-CSR / in-memory-scipy-CSR layouts. Shared by
/// [`pseudobulk_dex`] and the NB-GLM bindings (`super::nb_glm`).
pub(super) fn aggregate_pseudobulk(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    groupby: &[String],
    prefer_format: &str,
    gene_indices: Option<&[u32]>,
    method: scx_accel::AggregationMethod,
    min_cells_per_group: usize,
) -> PyResult<scx_accel::PseudobulkResult> {
    // Extract groupby columns from adata.obs.
    let obs = adata.getattr("obs")?;
    let mut obs_groups: Vec<Vec<String>> = Vec::with_capacity(groupby.len());
    for col_name in groupby {
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

    let result = if prefer_format == "csc" {
        // CSC dispatch requires a gene subset, since
        // full-gene CSC pseudobulk has no measurable speedup over CSR.
        // Resolve the gene subset: explicit `gene_indices` kwarg takes
        // precedence; otherwise fall back to the dataset's
        // `col_projection` if set.
        let resolved_indices: Vec<u32> = if let Some(gi) = gene_indices {
            if gi.is_empty() {
                return Err(PyRuntimeError::new_err(
                    "prefer_format='csc' requires non-empty gene_indices, \
                     or a column projection on adata.X (e.g. via X[:, var_mask])",
                ));
            }
            gi.to_vec()
        } else if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
            match backed.col_projection() {
                Some(cols) => cols.to_vec(),
                None => {
                    return Err(PyRuntimeError::new_err(
                        "prefer_format='csc' requires a gene subset; pass \
                         gene_indices=... or apply a column projection \
                         via adata[:, mask] / X[:, indices] before calling",
                    ));
                }
            }
        } else if let Ok(lazy) =
            x.extract::<PyRef<crate::lazy_transform::ScxLazyTransformedDataset>>()
        {
            match lazy.col_projection() {
                Some(cols) => cols.to_vec(),
                None => {
                    return Err(PyRuntimeError::new_err(
                        "prefer_format='csc' requires a gene subset; pass \
                         gene_indices=... or apply a column projection",
                    ));
                }
            }
        } else {
            return Err(PyRuntimeError::new_err(
                "prefer_format='csc' requires adata.X to be a backed or lazy \
                 SCX dataset; got a regular scipy/dense matrix",
            ));
        };

        let n_obs = obs_groups[0].len();
        let (cell_to_group, group_labels) = scx_accel::build_group_mapping(&obs_groups, n_obs);
        let n_groups = group_labels.len();
        let projected_gene_names: Vec<String> = resolved_indices
            .iter()
            .map(|&c| {
                gene_names.get(c as usize).cloned().ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "gene index {c} out of range for n_vars = {}",
                        gene_names.len()
                    ))
                })
            })
            .collect::<PyResult<Vec<_>>>()?;

        if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
            let source = backed.as_column_source().ok_or_else(|| {
                PyRuntimeError::new_err(
                    "CSC requested but unavailable: file has no CSC sidecar, \
                     or a row deletion vector is active",
                )
            })?;
            scx_accel::pseudobulk_aggregate_csc(
                source,
                &cell_to_group,
                n_groups,
                group_labels,
                groupby,
                &projected_gene_names,
                &resolved_indices,
                method,
                min_cells_per_group,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        } else if let Ok(lazy) =
            x.extract::<PyRef<crate::lazy_transform::ScxLazyTransformedDataset>>()
        {
            let lazy_src = lazy.as_column_source().ok_or_else(|| {
                PyRuntimeError::new_err(
                    "CSC requested but unavailable: file has no CSC sidecar, \
                     the transform chain contains a non-column-local op, or \
                     a row deletion vector is active",
                )
            })?;
            scx_accel::pseudobulk_aggregate_csc(
                &lazy_src,
                &cell_to_group,
                n_groups,
                group_labels,
                groupby,
                &projected_gene_names,
                &resolved_indices,
                method,
                min_cells_per_group,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        } else {
            unreachable!("type check above")
        }
    } else if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        scx_accel::pseudobulk_aggregate(
            &backed.backed,
            &obs_groups,
            groupby,
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
            groupby,
            &gene_names,
            method,
            min_cells_per_group,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    };

    Ok(result)
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
///     n_cpus: Worker cap for pydeseq2's parallel inference. `None` (default)
///         resolves a safe bound from the environment (`SLURM_CPUS_PER_TASK`,
///         then `OMP_NUM_THREADS`) and otherwise falls back to
///         `min(available_parallelism, 8)`. This prevents pydeseq2's loky
///         backend from forking one worker process per core on many-core
///         hosts (each a full Python interpreter), which OOMs the job.
///
///     backend: DE engine — "pydeseq2" (default) or "nb_glm" (Rust-native
///         negative-binomial GLM, no pydeseq2 dependency). Both emit the same
///         column schema.
///     nbglm_options: Optional dict of NB-GLM tuning knobs (only used when
///         backend="nb_glm"; see pyscx.accel.nb_glm). Ignored for pydeseq2.
///
/// Returns:
///     pandas DataFrame with columns: gene, baseMean, log2FoldChange,
///     lfcSE, stat, pvalue, padj, target, reference
#[pyfunction]
#[pyo3(signature = (adata, groupby, test_col, reference, design=None, aggr_method="sum", min_cells_per_group=10, stratify_by=None, min_cells_per_stratum=50, prefer_format="csr", gene_indices=None, n_cpus=None, backend="pydeseq2", nbglm_options=None))]
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
    prefer_format: &str,
    gene_indices: Option<Vec<u32>>,
    n_cpus: Option<usize>,
    backend: &str,
    nbglm_options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    if !matches!(prefer_format, "csr" | "csc") {
        return Err(PyValueError::new_err(format!(
            "Invalid prefer_format={prefer_format:?}; expected 'csr' or 'csc'"
        )));
    }
    if !matches!(backend, "pydeseq2" | "nb_glm") {
        return Err(PyValueError::new_err(format!(
            "Invalid backend={backend:?}; expected 'pydeseq2' or 'nb_glm'"
        )));
    }
    // The NB-GLM backend wants replicates as rows of ONE design (place the
    // replicate column in `groupby`), which is incompatible with the per-stratum
    // recursion below — each stratum would yield one sample per condition and the
    // fit would degenerate. Reject the combination with actionable guidance.
    if backend == "nb_glm" && stratify_by.is_some() {
        return Err(PyValueError::new_err(
            "pseudobulk_dex(backend=\"nb_glm\") does not support stratify_by: NB-GLM \
             treats replicates as rows of a single design, so put the replicate column \
             (batch/donor/well) directly in `groupby`, or use pyscx.accel.pdex_nb_glm \
             (which merges groupby + stratify_by for you).",
        ));
    }
    // §2.5: the NB count likelihood requires replicate-level *summed* counts.
    // Fractional aggregates (mean/median) are not valid inputs to the model, so
    // the count backend accepts only aggr_method="sum".
    if backend == "nb_glm" && aggr_method != "sum" {
        return Err(PyValueError::new_err(format!(
            "pseudobulk_dex(backend=\"nb_glm\") requires aggr_method=\"sum\" (got \
             {aggr_method:?}): the negative-binomial count model is defined on summed \
             replicate counts, not fractional {aggr_method} aggregates."
        )));
    }

    // Record the planned route on adata.uns["scx_accel"]["pseudobulk_dex"].
    // Pseudobulk aggregation + pydeseq2 is CPU-only; the only dispatch choice is
    // the gene-axis layout — cpu_csc when prefer_format="csc" (reads the
    // gene-major sidecar), cpu_csr otherwise. Stamped on the top-level adata
    // before the stratified branch (which recurses on discarded sub_adata
    // copies). This is the route the CSC dispatch gate asserts.
    let pb_route = if backend == "nb_glm" {
        scx_accel::AccelRoute::CpuNbGlm
    } else if prefer_format == "csc" {
        scx_accel::AccelRoute::CpuCsc
    } else {
        scx_accel::AccelRoute::CpuCsr
    };
    super::route::write_accel_route(
        py,
        adata,
        "pseudobulk_dex",
        &scx_accel::AccelExecutionInfo::new(pb_route, scx_accel::FallbackReason::None),
    )?;

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
                prefer_format,
                gene_indices.clone(),
                n_cpus,
                backend,
                nbglm_options,
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

    let result = aggregate_pseudobulk(
        py,
        adata,
        &groupby,
        prefer_format,
        gene_indices.as_deref(),
        method,
        min_cells_per_group,
    )?;

    if result.n_groups == 0 {
        return Err(PyRuntimeError::new_err(
            "no groups passed the min_cells_per_group filter",
        ));
    }

    // NB-GLM backend: fit the Rust-native negative-binomial GLM per contrast and
    // assemble the same PyDESeq2-style pandas schema the pydeseq2 path returns.
    if backend == "nb_glm" {
        let test_col_idx = groupby.iter().position(|c| c == test_col).unwrap();
        let nb_opts = super::nb_glm::nbglm_options_from_dict(py, nbglm_options)?;
        // An explicit `contrast` in nbglm_options also triggers the design-aware
        // path (it needs a named design to resolve the contrast against).
        let contrast_override = match nbglm_options {
            Some(d) => d.get_item("contrast")?,
            None => None,
        };
        // `pseudobulk_dex` has no `device=` kwarg; the NB-GLM backend runs on CPU
        // here (the GPU path is reached via `pyscx.accel.pdex_nb_glm(device=…)`).
        let df = if design.is_some() || contrast_override.is_some() {
            // §3.11: honor a caller-supplied formula design + optional contrast.
            // Default the formula to `~ test_col` (reproduces the intercept +
            // target-indicator model, but as a single shared-dispersion fit).
            let owned_default;
            let formula = match design {
                Some(s) => s,
                None => {
                    owned_default = format!("~ {test_col}");
                    owned_default.as_str()
                }
            };
            super::nb_glm::fit_targets_pandas_with_design(
                py,
                &result,
                test_col_idx,
                reference,
                formula,
                contrast_override.as_ref(),
                &nb_opts,
                None,
            )?
        } else {
            super::nb_glm::fit_targets_pandas(py, &result, test_col_idx, reference, &nb_opts, None)?
        };
        return Ok(df.unbind());
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

    // Bound every thread/process pool pydeseq2 can spin up. Three multipliers
    // stack on a many-core host (192 cores on a Chimera node) and OOM the job:
    //   1. loky/joblib worker processes (one per core) — capped via
    //      `DefaultInference(n_cpus=...)` below;
    //   2. numba parallel threads, inside each worker *and* the main process;
    //   3. the BLAS pool (OpenMP / MKL / OpenBLAS).
    // `n_cpus` only governs (1); (2) and (3) read their own env vars, so we
    // cap those too. The loky workers are spawned later (at `.deseq2()` /
    // `.summary()`) and inherit this env, so the cap reaches each worker's
    // numba/BLAS pools — that per-worker explosion was the residual OOM.
    let resolved_n_cpus = resolve_deseq_n_cpus(n_cpus);
    {
        let cap = resolved_n_cpus.to_string();
        // `n_cpus` (via DefaultInference below) only bounds the loky *process*
        // count; numba and the BLAS pool (OpenMP/MKL/OpenBLAS) read their own
        // env vars. Set them through Python's `os.environ` under the GIL — never
        // raw `std::env::set_var`, which races with `getenv` in background
        // BLAS/OpenMP threads (and is `unsafe` on edition 2024). `setdefault`
        // is a no-op when the operator already set a value, and CPython's
        // `os.environ` calls `putenv`, so freshly-spawned loky workers inherit
        // the cap.
        //
        // NB: this is process-sticky — the first `pseudobulk_dex` call's cap
        // persists for the process lifetime and cannot be raised later via this
        // API. Acceptable for OOM avoidance; pass a larger explicit `n_cpus`
        // (and pre-set these env vars yourself) if you need a bigger pool.
        let os_environ = py.import("os")?.getattr("environ")?;
        for var in [
            "NUMBA_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        ] {
            os_environ.call_method1("setdefault", (var, cap.as_str()))?;
        }
        // Best-effort: rein in numba in *this* process too. The env var only
        // binds freshly-spawned workers; the main interpreter may already have
        // imported numba (via scanpy / pydeseq2), so reduce its live pool. A
        // BLAS pool already sized in the parent is not shrunk, but the loky
        // workers (which inherit the env) are the actual OOM driver.
        if let Ok(numba) = py.import("numba") {
            let _ = numba.call_method1("set_num_threads", (resolved_n_cpus,));
        }
    }
    let inference = py
        .import("pydeseq2.default_inference")
        .map_err(|_| {
            PyRuntimeError::new_err(
                "pydeseq2.default_inference module not found. Ensure pydeseq2 is properly installed.\n\
                 Install with: pip install pydeseq2",
            )
        })?
        .call_method(
            "DefaultInference",
            (),
            Some(&{
                let kw = PyDict::new(py);
                kw.set_item("n_cpus", resolved_n_cpus)?;
                kw
            }),
        )?;

    // Create DeseqDataSet.
    let dds = pydeseq2.call_method(
        "DeseqDataSet",
        (),
        Some(&{
            let kw = PyDict::new(py);
            kw.set_item("counts", &counts_df)?;
            kw.set_item("metadata", &metadata_df)?;
            kw.set_item("design", &_design_str)?;
            kw.set_item("inference", &inference)?;
            kw
        }),
    )?;

    // Run DESeq2 pipeline.
    dds.call_method0("deseq2")?;

    for target in &target_levels {
        // Create DeseqStats for this contrast. `DeseqStats` builds its OWN
        // inference if none is passed — and that default is `n_cpus=None`,
        // which re-forks one loky worker per core for the Wald tests and
        // OOM-kills the job (the DeseqDataSet `inference=` above only covers
        // size-factor/dispersion/LFC fitting). Reuse the capped inference.
        let stat = pydeseq2_stats.call_method(
            "DeseqStats",
            (&dds,),
            Some(&{
                let kw = PyDict::new(py);
                let contrast =
                    pyo3::types::PyList::new(py, [test_col, target.as_str(), reference])?;
                kw.set_item("contrast", contrast)?;
                kw.set_item("inference", &inference)?;
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
