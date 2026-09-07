//! DE-result DataFrame marshalling and the `rank_genes_groups_df`
//! entry point (cell-eval bridge).

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyDict;

use super::*;

/// Build a DataFrame of the given `columns` in `column_order`, as either a
/// polars or a pandas DataFrame depending on `output`.
///
/// `output="pandas"` (the default) builds a pandas DataFrame directly and does
/// **not** import polars — pandas is a hard dependency via anndata, so the
/// default path needs nothing beyond a base `pip install pyscx`. `output="polars"`
/// requires polars, which lives in the optional `eval` extra; it is the opt-in
/// for cell-eval, whose `DEResults.data` is typed `pl.DataFrame`. Column schema
/// is identical across both.
pub(crate) fn build_de_dataframe<'py>(
    py: Python<'py>,
    columns: &Bound<'py, PyDict>,
    column_order: &[&str],
    output: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let order = pyo3::types::PyList::new(py, column_order)?;
    match output {
        "polars" => {
            let pl = crate::optional_deps::import_optional_with_hint(
                py,
                "polars",
                crate::optional_deps::EXTRA_EVAL,
                "output=\"polars\"",
                "polars",
                Some("The default output=\"pandas\" needs no extra."),
            )?;
            let df = pl.call_method1("DataFrame", (columns,))?;
            // Enforce column order regardless of dict iteration / constructor.
            df.call_method1("select", (order,))
        }
        "pandas" => {
            let pd = crate::pyimport::import_module(py, "pandas").map_err(|_| {
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
/// schema, as pandas (the default) or polars per `output`.
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
    group: Option<&Bound<'py, PyAny>>,
    key: &str,
    n_genes: Option<usize>,
    pval_cutoff: Option<f64>,
    log2fc_min: Option<f64>,
    log2fc_max: Option<f64>,
    output: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let rgg = adata.getattr("uns")?.get_item(key).map_err(|_| {
        PyValueError::new_err(format!(
            "adata.uns[{key:?}] not found — run pyscx.accel.rank_genes_groups(adata, groupby=...) \
             to populate it, or pass groupby= to compute DE here"
        ))
    })?;

    // Available group names are the structured-array field names of `names`.
    // Resolved before the `group` decision below so `group=None` can expand to
    // all of them without a second enumeration site.
    let names_arr = rgg.get_item("names")?;
    let available: Vec<String> = names_arr
        .getattr("dtype")?
        .getattr("names")?
        .extract()
        .unwrap_or_default();

    // `group` is None (every group, with a `group` column — scanpy's
    // "All groups are returned if group is None"), a single name (no `group`
    // column, also matching scanpy), or a list of names (with a `group` column).
    let (groups, multi): (Vec<String>, bool) = match group {
        None => {
            if available.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "adata.uns[{key:?}][\"names\"] has no structured-array fields, so there \
                     are no groups to extract — run \
                     pyscx.accel.rank_genes_groups(adata, groupby=...) to populate it, or \
                     pass groupby= to compute DE here"
                )));
            }
            // NOT sorted: `dtype.names` is the order `rank_genes_groups` wrote
            // and the order scanpy's own all-groups path iterates. Sorting here
            // would silently diverge from scanpy's row order.
            (available.clone(), true)
        }
        Some(g) => {
            if let Ok(s) = g.extract::<String>() {
                (vec![s], false)
            } else if let Ok(v) = g.extract::<Vec<String>>() {
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
            }
        }
    };
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
    // scanpy adds `pct_nz_group` / `pct_nz_reference` only when the matching
    // `pts` / `pts_rest` table was written (`rank_genes_groups(pts=True)`), and
    // there is no fallback for a missing `pts_rest` — a pairwise run has a
    // `pct_nz_group` column and no `pct_nz_reference`.
    let pts_table = if rgg.contains("pts")? {
        Some(rgg.get_item("pts")?)
    } else {
        None
    };
    let pts_rest_table = match &pts_table {
        Some(_) if rgg.contains("pts_rest")? => Some(rgg.get_item("pts_rest")?),
        _ => None,
    };
    let has_pts = pts_table.is_some();
    let has_pts_rest = pts_rest_table.is_some();
    let mut pct_nz_group: Vec<f64> = Vec::new();
    let mut pct_nz_reference: Vec<f64> = Vec::new();
    // The `pts` frame is `genes × groups` in var order, not rank order and not
    // truncated by `n_genes`, so the per-row value is looked up by gene name —
    // the join scanpy's `melt` + `merge` performs. The index is read once (it is
    // shared by every column and by `pts_rest`) and must be unique: on
    // duplicated var names scanpy's merge multiplies rows, and any single-valued
    // lookup would hand one gene's fraction to the other, so refuse rather than
    // guess — `rank_genes_groups(pts=True)` refuses to write such a table too.
    let pts_row_of: std::collections::HashMap<String, usize> = match &pts_table {
        Some(table) => {
            let index = table.getattr("index")?;
            let names: Vec<String> = index
                .call_method1("astype", ("str",))?
                .call_method0("tolist")?
                .extract()?;
            let mut map = std::collections::HashMap::with_capacity(names.len());
            for (row, name) in names.into_iter().enumerate() {
                if map.insert(name.clone(), row).is_some() {
                    return Err(PyValueError::new_err(format!(
                        "adata.uns[{key:?}][\"pts\"] has a duplicated var name {name:?}, so \
                         pct_nz_group / pct_nz_reference cannot be joined by gene name; make \
                         var_names unique (adata.var_names_make_unique()) before \
                         rank_genes_groups(pts=True)"
                    )));
                }
            }
            if let Some(rest) = &pts_rest_table {
                let same: bool = rest
                    .getattr("index")?
                    .call_method1("equals", (&index,))?
                    .extract()?;
                if !same {
                    return Err(PyValueError::new_err(format!(
                        "adata.uns[{key:?}][\"pts_rest\"] is not indexed like [\"pts\"]; the two \
                         tables must share one var-name index"
                    )));
                }
            }
            map
        }
        None => std::collections::HashMap::new(),
    };
    let gather_pct = |table: &Bound<'py, PyAny>, g: &str, kept: &[String]| -> PyResult<Vec<f64>> {
        let values: Vec<f64> = table.get_item(g)?.call_method0("tolist")?.extract()?;
        Ok(kept
            .iter()
            .map(|name| {
                pts_row_of
                    .get(name)
                    .and_then(|&row| values.get(row).copied())
                    .unwrap_or(f64::NAN)
            })
            .collect())
    };

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
        // `group=None` now makes the all-groups build the DEFAULT, so this loop
        // runs n_groups x n_genes times on real shapes (49 groups x 61,497 genes
        // is ~3M rows). Record the row count before the inner loop and extend
        // `col_group` once per group afterwards, instead of cloning the group
        // name per row.
        let rows_before = names.len();
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
            names.push(g_names[i].clone());
            scores.push(g_scores[i]);
            logfoldchanges.push(g_lfc[i]);
            pvals.push(g_pvals[i]);
            pvals_adj.push(g_padj[i]);
        }
        if multi {
            col_group.extend(std::iter::repeat_n(g.clone(), names.len() - rows_before));
        }
        if let Some(table) = &pts_table {
            let kept = &names[rows_before..];
            pct_nz_group.extend(gather_pct(table, g, kept)?);
            if let Some(rest) = &pts_rest_table {
                pct_nz_reference.extend(gather_pct(rest, g, kept)?);
            }
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

    let mut column_order: Vec<&str> = Vec::with_capacity(8);
    if multi {
        column_order.push("group");
    }
    column_order.extend(["names", "scores", "logfoldchanges", "pvals", "pvals_adj"]);
    if has_pts {
        dict.set_item("pct_nz_group", pct_nz_group)?;
        column_order.push("pct_nz_group");
        if has_pts_rest {
            dict.set_item("pct_nz_reference", pct_nz_reference)?;
            column_order.push("pct_nz_reference");
        }
    }
    build_de_dataframe(py, &dict, &column_order, output)
}

/// Differential-expression DataFrame — **two modes**, selected by which kwarg
/// you pass.
///
/// **Compute (`groupby=`)** — re-runs Wilcoxon rank-sum DE and returns a
/// DataFrame in cell-eval's `DEResults` column format. This is the format bridge
/// between SCX's Wilcoxon DE and cell-eval's DE metric pipeline — but **pass
/// `output="polars"` to feed it to cell-eval**, whose `DEResults.data` is typed
/// `pl.DataFrame`; the default pandas frame is rejected there. The accelerator
/// execution route is recorded on
/// `adata.uns["scx_accel"]["rank_genes_groups_df"]`. Columns:
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
/// supported yet.) Pass either `groupby=` or `group=`, not both.
///
/// **`group=None` (or an omitted `group=`) with no `groupby=` extracts every
/// group** in `adata.uns[key]`, leading `group` column included — matching
/// `sc.get.rank_genes_groups_df`'s "All groups are returned if `group` is
/// `None`". With `groupby=` set, `group=None` still routes to the compute path.
/// With neither and no `uns[key]`, it is an error naming both remedies.
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
///     output: `"pandas"` (default) or `"polars"`. Identical columns either way.
///         `"pandas"` needs no optional dependency; `"polars"` requires the
///         `eval` extra and is what cell-eval consumes.
///     device: compute-mode only; ignored in extract (`group=`) mode.
///     group: extraction mode — a group name (str), a list of names, or `None` /
///         omitted to pull every group in `adata.uns[key]`.
///     key: uns key to extract from (default: `"rank_genes_groups"`).
///     pval_cutoff / log2fc_min / log2fc_max: scanpy-style row filters (extraction
///         mode only): keep rows with `pvals_adj < pval_cutoff`,
///         `logfoldchanges > log2fc_min`, `logfoldchanges < log2fc_max`.
///
/// Example:
///     de_df = pyscx.accel.rank_genes_groups_df(adata, "perturbation")  # compute
///     ex = pyscx.accel.rank_genes_groups_df(adata, group="0")          # extract (scanpy-style)
///     ce = pyscx.accel.rank_genes_groups_df(adata, "perturbation", output="polars")  # cell-eval
#[pyfunction]
#[pyo3(signature = (adata, groupby=None, reference="rest", n_genes=None, gene_chunk_size=None, rankby_abs=false, tie_correct=false, device="auto", output="pandas", *, group=None, key="rank_genes_groups", pval_cutoff=None, log2fc_min=None, log2fc_max=None))]
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

    // After the cheap arg check, before the first write to `adata` (the
    // compute mode below stamps `uns["scx_accel"]`): rebuild a view as actual
    // so a backed X is not gathered by anndata's copy-on-write.
    //
    // Deliberately the no-var-guard variant, and not because this op is
    // order-indifferent — its compute mode returns gene-labelled DE rows, and
    // on a presentation-ordered backed `X` it used to label the sorted on-disk
    // projection with request-ordered `adata.var` names (measured: `g2`'s
    // effect reported under `g0`). That is now refused, but by
    // `select_de_matrix`, which guards the matrix the kernel actually reads.
    // Guarding here instead would also reject the **extraction** mode below,
    // which reads `adata.uns[key]` and never touches the matrix — a
    // presentation-ordered handle is no obstacle to pulling out results
    // computed before the reorder.
    crate::accel::prepare_target_no_var_guard(py, adata, "rank_genes_groups_df")?;

    // Extraction mode (scanpy `sc.get.rank_genes_groups_df` alias): read
    // precomputed results from `adata.uns[key]` instead of recomputing.
    //
    // scanpy's extractor returns EVERY group for `group=None` ("All groups are
    // returned if group is None"). An explicit Python `group=None` and an
    // omitted `group=` both arrive here as Rust `None` — pyo3 cannot tell them
    // apart — so route on the object instead of the argument: a precomputed
    // `uns[key]` means extract-all, and its absence with no `groupby=` is the
    // "got neither" case the error below names.
    //
    // The `groupby.is_none()` conjunct is load-bearing: without it,
    // `rank_genes_groups_df(adata, groupby="batch")` on an adata that already
    // carries `uns["rank_genes_groups"]` — the normal pipeline shape — would
    // silently switch from compute to extract.
    let extract_all = group.is_none()
        && groupby.is_none()
        && adata
            .getattr("uns")
            .and_then(|u| u.contains(key))
            .unwrap_or(false);

    if group.is_some() || extract_all {
        if group.is_some() && groupby.is_some() {
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
        PyValueError::new_err(format!(
            "rank_genes_groups_df needs groupby= (to compute DE) or group= (to extract \
             precomputed adata.uns[{key:?}]); got neither, and adata.uns[{key:?}] does not \
             exist so there is nothing to extract. Run \
             pyscx.accel.rank_genes_groups(adata, groupby=...) first, then call this with no \
             arguments to get every group."
        ))
    })?;

    let resolved = crate::accel::gpu::resolve_device(device)?;
    #[cfg(feature = "gpu")]
    let gpu_device_id = resolved.gpu_id();
    #[cfg(not(feature = "gpu"))]
    let gpu_device_id: Option<usize> = {
        let _ = resolved;
        None
    };
    // `rank_genes_groups_df` is the cell-eval-style entry; CSC dispatch
    // is reserved for the scanpy-style `rank_genes_groups`. Pin to CSR.
    let run = run_rank_genes_groups_inner(
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
        false, // pts: cell-eval's DEResults schema has no fraction-expressing column
        None,  // groups: every group, as the schema consumers expect
        "rank_genes_groups_df",
    )?;
    let result = run.result;

    // Record the accelerator execution route on adata.uns; the returned
    // polars DataFrame carries no metadata of its own. `result.exec_info` is
    // already complete (route + reason) from the single planner.
    crate::accel::route::announce_route(py, "rank_genes_groups_df", device, &result.exec_info);
    crate::accel::route::warn_materialized_csc_sidecar(
        py,
        "rank_genes_groups_df",
        device,
        adata,
        &result.exec_info,
    );
    crate::accel::route::write_accel_route(py, adata, "rank_genes_groups_df", &result.exec_info)?;

    let df = de_result_to_cell_eval_dataframe(py, &result, n_genes, output)?;
    Ok(df.unbind())
}

// ---------------------------------------------------------------------------
// pdex `mode="ref"` accelerator binding
// ---------------------------------------------------------------------------
