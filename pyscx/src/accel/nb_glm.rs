//! Python bindings for the Rust-native pseudobulk NB-GLM (`scx_accel`).
//!
//! Two entry points (both CPU-only; no `device=` — spec §13):
//!   * [`nb_glm`] — direct DESeq2-replacement on already-pseudobulked matrices,
//!     returning a pandas DataFrame with PyDESeq2-style column names.
//!   * [`pdex_nb_glm`] — pseudobulk-from-AnnData with a replicate stratifier,
//!     emitting the cell-eval/pdex polars schema so `cell-eval-scx` consumes it
//!     with zero changes (spec §4.4, §10).
//!
//! `pseudobulk_dex(backend="nb_glm")` also routes through [`fit_targets_pandas`].

use std::collections::HashMap;
use std::f64::consts::LN_2;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use scx_accel::{DispersionMethod, NbGlmContrast, NbGlmOptions, NbGlmResult, PseudobulkResult};

use super::de::build_de_dataframe;
use super::pseudobulk::aggregate_pseudobulk;

/// Copy a 2-D array-like into a row-major `Vec<f64>` plus its `(rows, cols)`.
fn dense2d_f64(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, usize, usize)> {
    let np = py.import("numpy")?;
    let a = np.call_method1("asarray", (obj,))?;
    let a = super::util::astype_no_copy(py, &a, "float64")?; // no copy if already f64
    let a = np.call_method1("ascontiguousarray", (a,))?;
    let ro: PyReadonlyArray2<'_, f64> = a
        .extract()
        .map_err(|_| PyValueError::new_err("expected a 2-D numeric array"))?;
    let sh = ro.shape();
    let (r, c) = (sh[0], sh[1]);
    let v = ro
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        .to_vec();
    Ok((v, r, c))
}

/// Copy a 1-D array-like into a `Vec<f64>`.
fn dense1d_f64(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let np = py.import("numpy")?;
    let a = np.call_method1("asarray", (obj,))?;
    let a = super::util::astype_no_copy(py, &a, "float64")?; // no copy if already f64
    let a = np.call_method1("ascontiguousarray", (a,))?;
    let ro: PyReadonlyArray1<'_, f64> = a
        .extract()
        .map_err(|_| PyValueError::new_err("expected a 1-D numeric array"))?;
    Ok(ro
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        .to_vec())
}

/// Drop cells in under-populated strata before aggregation. A "stratum" is a
/// unique combination of the `stratify_by` columns; one with fewer than
/// `min_cells_per_stratum` total cells cannot form a reliable pseudobulk
/// replicate. Returns the original `adata` (no copy) when nothing is dropped,
/// else a subset copy.
fn filter_sparse_strata<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    strat: &[String],
    min_cells_per_stratum: usize,
) -> PyResult<Bound<'py, PyAny>> {
    if min_cells_per_stratum == 0 || strat.is_empty() {
        return Ok(adata.clone());
    }
    let obs = adata.getattr("obs")?;
    let mut cols: Vec<Vec<String>> = Vec::with_capacity(strat.len());
    for c in strat {
        let labels: Vec<String> = obs
            .get_item(c.as_str())?
            .call_method1("astype", ("str",))?
            .call_method0("tolist")?
            .extract()?;
        cols.push(labels);
    }
    let n = cols.first().map(|v| v.len()).unwrap_or(0);
    // Composite per-cell stratum key (\u{1} separator can't collide with labels).
    let keys: Vec<String> = (0..n)
        .map(|i| {
            cols.iter()
                .map(|c| c[i].as_str())
                .collect::<Vec<_>>()
                .join("\u{1}")
        })
        .collect();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for k in &keys {
        *counts.entry(k.as_str()).or_insert(0) += 1;
    }
    let keep: Vec<bool> = keys
        .iter()
        .map(|k| counts[k.as_str()] >= min_cells_per_stratum)
        .collect();
    if keep.iter().all(|&b| b) {
        return Ok(adata.clone());
    }
    let mask = py.import("numpy")?.call_method1("array", (keep,))?;
    adata.get_item(&mask)?.call_method0("copy")
}

/// Recognised keys for the options dict (mirror `NbGlmOptions` field names).
const NBGLM_OPTION_KEYS: &[&str] = &[
    "dispersion",
    "min_disp",
    "max_disp",
    "min_mu",
    "eta_min",
    "eta_max",
    "beta_ridge",
    "max_irls_iters",
    "irls_tol",
    "disp_newton_iters",
    "max_outer_iters",
    "outer_tol",
    "fit_dispersion_trend",
    "shrink_dispersion",
    "cooks_filtering",
    "cooks_cutoff",
    "independent_filtering",
    "independent_filter_alpha",
];

/// Parse an optional options dict onto `NbGlmOptions::default()`. Recognised keys
/// mirror the Rust field names; an **unrecognised key raises** `ValueError` so a
/// typo (`max_outer_iter` vs `max_outer_iters`) fails loudly instead of silently
/// no-opping.
pub(super) fn nbglm_options_from_dict(
    _py: Python<'_>,
    dict: Option<&Bound<'_, PyDict>>,
) -> PyResult<NbGlmOptions> {
    let mut o = NbGlmOptions::default();
    let Some(d) = dict else { return Ok(o) };

    // Reject unknown keys up front (loud misconfiguration).
    for key in d.keys() {
        let k: String = key.extract()?;
        if !NBGLM_OPTION_KEYS.contains(&k.as_str()) {
            return Err(PyValueError::new_err(format!(
                "unknown nb_glm option {k:?}; valid keys: {NBGLM_OPTION_KEYS:?}"
            )));
        }
    }

    if let Some(v) = d.get_item("dispersion")? {
        let s: String = v.extract()?;
        o.dispersion = match s.as_str() {
            "moments" => DispersionMethod::Moments,
            "cox_reid_mle" => DispersionMethod::CoxReidMle,
            "cox_reid_shrunk" => DispersionMethod::CoxReidShrunk,
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid dispersion={other:?}; expected 'moments', \
                     'cox_reid_mle', or 'cox_reid_shrunk'"
                )))
            }
        };
    }
    if let Some(v) = d.get_item("min_disp")? {
        o.min_disp = v.extract()?;
    }
    if let Some(v) = d.get_item("max_disp")? {
        o.max_disp = v.extract()?;
    }
    if let Some(v) = d.get_item("min_mu")? {
        o.min_mu = v.extract()?;
    }
    if let Some(v) = d.get_item("eta_min")? {
        o.eta_min = v.extract()?;
    }
    if let Some(v) = d.get_item("eta_max")? {
        o.eta_max = v.extract()?;
    }
    if let Some(v) = d.get_item("beta_ridge")? {
        o.beta_ridge = v.extract()?;
    }
    if let Some(v) = d.get_item("max_irls_iters")? {
        o.max_irls_iters = v.extract()?;
    }
    if let Some(v) = d.get_item("irls_tol")? {
        o.irls_tol = v.extract()?;
    }
    if let Some(v) = d.get_item("disp_newton_iters")? {
        o.disp_newton_iters = v.extract()?;
    }
    if let Some(v) = d.get_item("max_outer_iters")? {
        o.max_outer_iters = v.extract()?;
    }
    if let Some(v) = d.get_item("outer_tol")? {
        o.outer_tol = v.extract()?;
    }
    if let Some(v) = d.get_item("fit_dispersion_trend")? {
        o.fit_dispersion_trend = v.extract()?;
    }
    if let Some(v) = d.get_item("shrink_dispersion")? {
        o.shrink_dispersion = v.extract()?;
    }
    if let Some(v) = d.get_item("cooks_filtering")? {
        o.cooks_filtering = v.extract()?;
    }
    if let Some(v) = d.get_item("cooks_cutoff")? {
        // `None` keeps the DESeq2 default `qf(0.99, p, m−p)`; a float overrides it.
        o.cooks_cutoff = if v.is_none() {
            None
        } else {
            Some(v.extract()?)
        };
    }
    if let Some(v) = d.get_item("independent_filtering")? {
        o.independent_filtering = v.extract()?;
    }
    if let Some(v) = d.get_item("independent_filter_alpha")? {
        o.independent_filter_alpha = v.extract()?;
    }
    Ok(o)
}

/// Build an [`NbGlmContrast`] from a Python object: an integer coefficient index,
/// a weight vector, or `None` (defaults to the last coefficient — the DESeq2
/// "last coefficient" convention, since a numeric design has no column names).
fn contrast_from_pyany(
    obj: Option<&Bound<'_, PyAny>>,
    n_features: usize,
) -> PyResult<NbGlmContrast> {
    match obj {
        None => Ok(NbGlmContrast::Coefficient {
            index: n_features - 1,
        }),
        Some(o) => {
            if let Ok(idx) = o.extract::<usize>() {
                if idx >= n_features {
                    return Err(PyValueError::new_err(format!(
                        "contrast index {idx} out of range for n_features={n_features}"
                    )));
                }
                Ok(NbGlmContrast::Coefficient { index: idx })
            } else if let Ok(weights) = o.extract::<Vec<f64>>() {
                if weights.len() != n_features {
                    return Err(PyValueError::new_err(format!(
                        "contrast weight vector length {} != n_features {n_features}",
                        weights.len()
                    )));
                }
                Ok(NbGlmContrast::Vector { weights })
            } else {
                Err(PyValueError::new_err(
                    "contrast must be an int (coefficient index) or a sequence \
                     of floats (weight vector)",
                ))
            }
        }
    }
}

/// One fitted contrast (target vs reference).
pub(super) struct TargetFit {
    pub target: String,
    pub result: NbGlmResult,
}

/// Fit one NB-GLM per non-reference level of the condition column (at
/// `cond_col_idx` in the pseudobulk group labels), each on the `{target,
/// reference}` pseudobulk replicates with design `[intercept, is_target]`.
///
/// Strata enter only as replicates (extra rows), never as design covariates, so
/// the marginal effect aligns with `pdex_ref`'s pooled test (the parity bar).
/// Targets without ≥2 replicates per condition are skipped with a warning.
pub(super) fn fit_targets_nbglm(
    py: Python<'_>,
    result: &PseudobulkResult,
    cond_col_idx: usize,
    reference: &str,
    options: &NbGlmOptions,
) -> PyResult<Vec<TargetFit>> {
    let n_groups = result.n_groups;
    let n_vars = result.n_vars;
    let cond: Vec<&str> = (0..n_groups)
        .map(|g| result.group_labels[g][cond_col_idx].as_str())
        .collect();

    if !cond.contains(&reference) {
        return Err(PyRuntimeError::new_err(format!(
            "reference level '{reference}' not found in the grouping column"
        )));
    }
    let mut targets: Vec<String> = cond
        .iter()
        .filter(|c| **c != reference)
        .map(|c| c.to_string())
        .collect();
    targets.sort();
    targets.dedup();
    if targets.is_empty() {
        return Err(PyRuntimeError::new_err(format!(
            "no non-reference levels found (all groups are reference='{reference}')"
        )));
    }

    let warnings = py.import("warnings")?;
    let mut fits = Vec::with_capacity(targets.len());
    for target in &targets {
        let sub: Vec<usize> = (0..n_groups)
            .filter(|&g| cond[g] == target.as_str() || cond[g] == reference)
            .collect();
        let n_target = sub.iter().filter(|&&g| cond[g] == target.as_str()).count();
        let n_ref = sub.iter().filter(|&&g| cond[g] == reference).count();
        let n_sub = sub.len();
        // Replicate guard: need ≥2 pseudobulk samples per condition and more
        // samples than design columns (intercept + treatment).
        if n_target < 2 || n_ref < 2 || n_sub <= 2 {
            warnings.call_method1(
                "warn",
                (format!(
                    "NB-GLM: skipping target '{target}' — needs >=2 pseudobulk \
                     replicates per condition (target={n_target}, reference={n_ref}). \
                     Provide a stratify_by spanning more strata."
                ),),
            )?;
            continue;
        }

        // Gene-major counts [n_vars x n_sub] and design [n_sub x 2].
        let mut cg = vec![0.0_f64; n_vars * n_sub];
        let mut design = vec![0.0_f64; n_sub * 2];
        for (s, &g) in sub.iter().enumerate() {
            design[s * 2] = 1.0;
            design[s * 2 + 1] = if cond[g] == target.as_str() { 1.0 } else { 0.0 };
        }
        // Transpose group-major `counts` → gene-major `cg`; inner loop over `s`
        // writes `cg` contiguously (stride 1) for cache-friendly stores.
        for j in 0..n_vars {
            for (s, &g) in sub.iter().enumerate() {
                cg[j * n_sub + s] = result.counts[g * n_vars + j];
            }
        }

        match scx_accel::pseudobulk_nb_glm(
            &cg,
            n_vars,
            n_sub,
            &design,
            2,
            None,
            NbGlmContrast::Coefficient { index: 1 },
            options.clone(),
        ) {
            Ok(r) => fits.push(TargetFit {
                target: target.clone(),
                result: r,
            }),
            Err(e) => {
                warnings
                    .call_method1("warn", (format!("NB-GLM: target '{target}' failed: {e}"),))?;
            }
        }
    }
    Ok(fits)
}

/// Assemble per-target NB-GLM fits into the PyDESeq2-style pandas schema used by
/// `pseudobulk_dex` (`gene, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj,
/// target, reference`). `lfcSE` is rescaled to the log2 scale.
pub(super) fn fit_targets_pandas<'py>(
    py: Python<'py>,
    result: &PseudobulkResult,
    cond_col_idx: usize,
    reference: &str,
    options: &NbGlmOptions,
) -> PyResult<Bound<'py, PyAny>> {
    let fits = fit_targets_nbglm(py, result, cond_col_idx, reference, options)?;
    if fits.is_empty() {
        return Err(PyRuntimeError::new_err(
            "NB-GLM produced no fittable contrasts: every target had < 2 pseudobulk \
             replicates per condition. A pseudobulk NB-GLM needs replicates — include a \
             batch/donor/well column in `groupby`. For no-replicate layouts use \
             de_method=\"pdex_ref\" or \"wilcoxon\" (per-cell tests).",
        ));
    }
    let n_vars = result.n_vars;
    let cap = fits.len() * n_vars;
    let (mut gene, mut target_col, mut ref_col) = (
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
    );
    let (mut base_mean, mut log2fc, mut lfc_se, mut stat, mut pvalue, mut padj) = (
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
    );
    for tf in &fits {
        let r = &tf.result;
        for j in 0..n_vars {
            gene.push(result.gene_names[j].clone());
            target_col.push(tf.target.clone());
            ref_col.push(reference.to_string());
            base_mean.push(r.base_mean[j]);
            log2fc.push(r.log2_fold_change[j]);
            lfc_se.push(r.standard_error[j] / LN_2);
            stat.push(r.wald_stat[j]);
            pvalue.push(r.p_value[j]);
            padj.push(r.p_adj[j]);
        }
    }
    let dict = PyDict::new(py);
    dict.set_item("gene", gene)?;
    dict.set_item("baseMean", base_mean)?;
    dict.set_item("log2FoldChange", log2fc)?;
    dict.set_item("lfcSE", lfc_se)?;
    dict.set_item("stat", stat)?;
    dict.set_item("pvalue", pvalue)?;
    dict.set_item("padj", padj)?;
    dict.set_item("target", target_col)?;
    dict.set_item("reference", ref_col)?;

    let pd = py.import("pandas")?;
    let df = pd.call_method1("DataFrame", (&dict,))?;
    let order = PyList::new(
        py,
        [
            "gene",
            "baseMean",
            "log2FoldChange",
            "lfcSE",
            "stat",
            "pvalue",
            "padj",
            "target",
            "reference",
        ],
    )?;
    df.get_item(order)
}

/// Fit a Rust-native negative-binomial GLM on an already-pseudobulked count
/// matrix and test a contrast (a DESeq2 replacement). CPU-only, `f64` — there is
/// no `device=` argument.
///
/// `counts` is `[n_samples × n_genes]` (default) or `[n_genes × n_samples]` per
/// `counts_axis`; `design` is `[n_samples × n_features]` and full column rank.
/// `size_factors=None` computes DESeq2 median-ratio factors. `contrast` is an
/// integer coefficient index, a weight vector, or `None` (the last coefficient,
/// DESeq2 convention). `options` is an optional dict overriding `NbGlmOptions`
/// fields (`dispersion`, `min_disp`, `max_disp`, `max_irls_iters`, `irls_tol`,
/// `max_outer_iters`, `fit_dispersion_trend`, `shrink_dispersion`).
///
/// Returns a pandas DataFrame with PyDESeq2-style columns: `gene, baseMean,
/// log2FoldChange, lfcSE, stat, pvalue, padj, dispersion, converged, n_iter`
/// (`lfcSE` is on the log2 scale).
///
/// Results are DESeq2-*style*, not DESeq2-*identical*: it implements IRLS +
/// Cox–Reid dispersion + trend/shrinkage, plus DESeq2-default Cook's-distance
/// outlier filtering and base-mean independent filtering (both on by default;
/// the `cooks` column reports the max Cook's distance per gene). It still omits
/// apeglm LFC shrinkage. Use PyDESeq2 when exact DESeq2 numerics are required.
/// See docs/pseudobulk_nb_glm.md.
#[pyfunction]
#[pyo3(signature = (counts, design, size_factors=None, contrast=None, gene_names=None, sample_names=None, options=None, counts_axis="samples_by_genes"))]
#[allow(clippy::too_many_arguments)]
pub fn nb_glm(
    py: Python<'_>,
    counts: &Bound<'_, PyAny>,
    design: &Bound<'_, PyAny>,
    size_factors: Option<&Bound<'_, PyAny>>,
    contrast: Option<&Bound<'_, PyAny>>,
    gene_names: Option<Vec<String>>,
    sample_names: Option<Vec<String>>,
    options: Option<&Bound<'_, PyDict>>,
    counts_axis: &str,
) -> PyResult<Py<PyAny>> {
    let _ = sample_names; // accepted for API symmetry; output is per-gene
    let (counts_vec, r, c) = dense2d_f64(py, counts)?;
    let (n_genes, n_samples, counts_gene_major) = match counts_axis {
        "samples_by_genes" => {
            let (n_samples, n_genes) = (r, c);
            let mut cg = vec![0.0_f64; n_genes * n_samples];
            for s in 0..n_samples {
                for g in 0..n_genes {
                    cg[g * n_samples + s] = counts_vec[s * n_genes + g];
                }
            }
            (n_genes, n_samples, cg)
        }
        "genes_by_samples" => (r, c, counts_vec),
        other => {
            return Err(PyValueError::new_err(format!(
                "invalid counts_axis={other:?}; expected 'samples_by_genes' or \
                 'genes_by_samples'"
            )))
        }
    };

    let (design_vec, ds_rows, n_features) = dense2d_f64(py, design)?;
    if ds_rows != n_samples {
        return Err(PyValueError::new_err(format!(
            "design has {ds_rows} rows but counts imply {n_samples} samples"
        )));
    }

    let size_factors = match size_factors {
        Some(sf) => Some(dense1d_f64(py, sf)?),
        None => None,
    };
    let options = nbglm_options_from_dict(py, options)?;
    let contrast = contrast_from_pyany(contrast, n_features)?;

    let res = scx_accel::pseudobulk_nb_glm(
        &counts_gene_major,
        n_genes,
        n_samples,
        &design_vec,
        n_features,
        size_factors.as_deref(),
        contrast,
        options,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let gene_labels: Vec<String> =
        gene_names.unwrap_or_else(|| (0..n_genes).map(|i| format!("gene_{i}")).collect());
    if gene_labels.len() != n_genes {
        return Err(PyValueError::new_err(format!(
            "gene_names length {} != n_genes {n_genes}",
            gene_labels.len()
        )));
    }

    let lfc_se: Vec<f64> = res.standard_error.iter().map(|s| s / LN_2).collect();
    let dict = PyDict::new(py);
    dict.set_item("gene", gene_labels)?;
    dict.set_item("baseMean", res.base_mean)?;
    dict.set_item("log2FoldChange", res.log2_fold_change)?;
    dict.set_item("lfcSE", lfc_se)?;
    dict.set_item("stat", res.wald_stat)?;
    dict.set_item("pvalue", res.p_value)?;
    dict.set_item("padj", res.p_adj)?;
    dict.set_item("dispersion", res.dispersion)?;
    dict.set_item("cooks", res.cooks)?;
    dict.set_item("converged", res.converged)?;
    dict.set_item("n_iter", res.n_iter)?;

    let pd = py.import("pandas")?;
    let df = pd.call_method1("DataFrame", (&dict,))?;
    let order = PyList::new(
        py,
        [
            "gene",
            "baseMean",
            "log2FoldChange",
            "lfcSE",
            "stat",
            "pvalue",
            "padj",
            "dispersion",
            "cooks",
            "converged",
            "n_iter",
        ],
    )?;
    Ok(df.get_item(order)?.unbind())
}

/// Pseudobulk NB-GLM differential expression for the cell-eval/pdex consumer.
///
/// Aggregates `groupby × stratify_by` pseudobulk **replicates** from `adata`,
/// fits one Rust-native NB-GLM per non-reference perturbation vs `reference`, and
/// returns the cell-eval `DEResults` **polars** schema (`target, feature,
/// fold_change, p_value, fdr, log2_fold_change, abs_log2_fold_change`) — the same
/// schema as `rank_genes_groups_df` / `pdex_ref`, so it drops into `cell_eval`
/// unchanged. CPU-only (no `device=`); records `route="cpu_nb_glm"` on
/// `adata.uns["scx_accel"]["pdex_nb_glm"]`.
///
/// A pseudobulk NB-GLM needs ≥ 2 samples per condition to estimate dispersion, so
/// `stratify_by` (a batch/donor/well/replicate obs column spanning ≥ 2 strata) is
/// **required** — `pdex_nb_glm` raises a `ValueError` pointing to `pdex_ref` /
/// `rank_genes_groups` when it is absent. NB-GLM needs **raw counts**: log1p
/// input (auto-detected via `adata.uns["log1p"]`, or `is_log1p=True`) is
/// rejected. DESeq2-*style*, not DESeq2-*identical*. See
/// docs/pseudobulk_nb_glm.md.
#[pyfunction]
#[pyo3(signature = (adata, groupby, reference, stratify_by=None, min_cells_per_group=10, min_cells_per_stratum=50, is_log1p=None, nbglm_options=None, gene_chunk_size=None, prefer_format="csr"))]
#[allow(clippy::too_many_arguments)]
pub fn pdex_nb_glm(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    groupby: &str,
    reference: &str,
    stratify_by: Option<Vec<String>>,
    min_cells_per_group: usize,
    min_cells_per_stratum: usize,
    is_log1p: Option<bool>,
    nbglm_options: Option<&Bound<'_, PyDict>>,
    gene_chunk_size: Option<usize>,
    prefer_format: &str,
) -> PyResult<Py<PyAny>> {
    // Replicate guard (§4.4): a pseudobulk NB-GLM needs ≥2 samples per condition.
    let strat = match &stratify_by {
        Some(s) if !s.is_empty() => s.clone(),
        _ => {
            return Err(PyValueError::new_err(
                "pdex_nb_glm requires stratify_by (e.g. a batch/donor/well/replicate \
                 obs column) to form pseudobulk replicates: without it, dispersion is \
                 unidentifiable per perturbation. For no-replicate layouts use \
                 de_method=\"pdex_ref\" or \"wilcoxon\" (per-cell tests).",
            ))
        }
    };

    // `pdex_nb_glm` fits all genes (gene-parallel, no gene subset), so the CSC
    // sidecar path — which requires a gene projection — does not apply.
    if prefer_format != "csr" {
        return Err(PyValueError::new_err(format!(
            "pdex_nb_glm only supports prefer_format=\"csr\" (it fits every gene; there \
             is no gene-subset path for the column-major CSC sidecar). Got {prefer_format:?}."
        )));
    }

    // `gene_chunk_size` is reserved for a future out-of-core fit; v1 holds all
    // genes in memory. Warn rather than silently ignore a caller-set value.
    if gene_chunk_size.is_some() {
        py.import("warnings")?.call_method1(
            "warn",
            (
                "pdex_nb_glm: gene_chunk_size is not implemented in v1 (all genes are \
              fit in memory) and is ignored.",
            ),
        )?;
    }

    // NB-GLM needs raw counts; refuse log1p-normalized input.
    let is_log1p = match is_log1p {
        Some(b) => b,
        None => adata.getattr("uns")?.contains("log1p").unwrap_or(false),
    };
    if is_log1p {
        return Err(PyValueError::new_err(
            "pdex_nb_glm requires raw counts but the data looks log1p-normalized \
             (adata.uns['log1p'] present, or is_log1p=True). Pass raw counts, or use \
             de_method=\"pdex_ref\"/\"wilcoxon\".",
        ));
    }

    // Drop whole strata (unique stratify_by combinations) with too few total
    // cells before aggregation — a sparse donor/batch cannot form a reliable
    // pseudobulk replicate. (`min_cells_per_group` still filters each
    // condition×stratum combo afterward.)
    let working = filter_sparse_strata(py, adata, &strat, min_cells_per_stratum)?;

    // Aggregate by [groupby, *stratify_by]: each condition×stratum combo is one
    // pseudobulk replicate. groupby is column 0 of the resulting group labels.
    let mut combined = Vec::with_capacity(1 + strat.len());
    combined.push(groupby.to_string());
    combined.extend(strat.iter().cloned());

    let result = aggregate_pseudobulk(
        py,
        &working,
        &combined,
        prefer_format,
        None,
        scx_accel::AggregationMethod::Sum,
        min_cells_per_group,
    )?;
    if result.n_groups == 0 {
        return Err(PyRuntimeError::new_err(
            "no pseudobulk groups passed the min_cells_per_group filter",
        ));
    }

    let options = nbglm_options_from_dict(py, nbglm_options)?;
    let fits = fit_targets_nbglm(py, &result, 0, reference, &options)?;
    if fits.is_empty() {
        return Err(PyRuntimeError::new_err(
            "no perturbation had enough pseudobulk replicates to fit NB-GLM; \
             check stratify_by spans ≥2 strata per perturbation",
        ));
    }

    // Stamp the route on adata.uns["scx_accel"]["pdex_nb_glm"].
    super::route::write_accel_route(
        py,
        adata,
        "pdex_nb_glm",
        &scx_accel::AccelExecutionInfo::new(
            scx_accel::AccelRoute::CpuNbGlm,
            scx_accel::FallbackReason::None,
        ),
    )?;

    // Flatten to the cell-eval/pdex column vectors.
    let n_vars = result.n_vars;
    let cap = fits.len() * n_vars;
    let mut targets = Vec::with_capacity(cap);
    let mut features = Vec::with_capacity(cap);
    let mut fold_changes = Vec::with_capacity(cap);
    let mut p_values = Vec::with_capacity(cap);
    let mut fdrs = Vec::with_capacity(cap);
    let mut log2_fcs = Vec::with_capacity(cap);
    let mut abs_log2_fcs = Vec::with_capacity(cap);
    for tf in &fits {
        let r = &tf.result;
        for j in 0..n_vars {
            targets.push(tf.target.clone());
            features.push(result.gene_names[j].clone());
            let lfc = r.log2_fold_change[j];
            log2_fcs.push(lfc);
            abs_log2_fcs.push(lfc.abs());
            fold_changes.push(if lfc.is_finite() {
                lfc.exp2()
            } else {
                f64::NAN
            });
            p_values.push(r.p_value[j]);
            fdrs.push(r.p_adj[j]);
        }
    }

    let dict = PyDict::new(py);
    dict.set_item("target", targets)?;
    dict.set_item("feature", features)?;
    dict.set_item("fold_change", fold_changes)?;
    dict.set_item("p_value", p_values)?;
    dict.set_item("fdr", fdrs)?;
    dict.set_item("log2_fold_change", log2_fcs)?;
    dict.set_item("abs_log2_fold_change", abs_log2_fcs)?;

    let df = build_de_dataframe(
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
        "polars",
    )?;
    Ok(df.unbind())
}
