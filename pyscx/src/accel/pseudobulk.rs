//! Pseudobulk differential expression via Rust aggregation + pydeseq2.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::ScxBackedSparseDataset;

use super::de::extract_strata;

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
