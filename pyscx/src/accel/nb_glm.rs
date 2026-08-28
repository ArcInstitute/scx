//! Python bindings for the Rust-native pseudobulk NB-GLM (`scx_accel`).
//!
//! Two entry points (both accept `device="auto"|"cpu"|"gpu"|"gpu:N"`; the GPU
//! path is the Stage-A host-fed dense fitter in `scx-gpu/kernels/nb_glm.cu`):
//!   * [`nb_glm`] — direct DESeq2-replacement on already-pseudobulked matrices,
//!     returning a pandas DataFrame with PyDESeq2-style column names.
//!   * [`pdex_nb_glm`] — pseudobulk-from-AnnData with a replicate stratifier,
//!     emitting the cell-eval/pdex *column* schema (spec §4.4, §10). The
//!     container is **not** drop-in for `cell_eval`: this defaults to pandas and
//!     `cell_eval`'s `DEResults.data` is typed `pl.DataFrame`, so a consumer must
//!     pass `output="polars"`. See docs/pseudobulk_nb_glm.md.
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

/// Whether the problem dims fit the GPU kernel's register bounds (p ≤ PMAX,
/// n_sub ≤ NSUB_MAX). Always `false` without the `gpu` feature.
fn gpu_dims_ok(n_features: usize, n_samples: usize) -> bool {
    #[cfg(feature = "gpu")]
    {
        n_features <= scx_accel::GPU_NB_GLM_PMAX && n_samples <= scx_accel::GPU_NB_GLM_NSUB_MAX
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = (n_features, n_samples);
        false
    }
}

/// Extract the GPU device id from a [`ResolvedDevice`] (always `None` without
/// the `gpu` feature, where a GPU device cannot be resolved).
fn resolve_gpu_id(resolved: super::gpu::ResolvedDevice) -> Option<usize> {
    #[cfg(feature = "gpu")]
    {
        resolved.gpu_id()
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = resolved;
        None
    }
}

/// A resolved GPU device for an NB-GLM op, held once and reused across a
/// many-target loop (creating a CUDA context per target would be ruinous).
/// Carries nothing without the `gpu` feature.
#[cfg(feature = "gpu")]
type NbGlmGpuDev = Option<scx_accel::GpuDevice>;
#[cfg(not(feature = "gpu"))]
type NbGlmGpuDev = Option<()>;

/// Create the reusable [`NbGlmGpuDev`] for a resolved GPU device id. `None`
/// device id (CPU) yields `None`. Errors only on CUDA context-creation failure.
fn make_gpu_dev(gpu_device_id: Option<usize>) -> PyResult<NbGlmGpuDev> {
    #[cfg(feature = "gpu")]
    {
        match gpu_device_id {
            Some(id) => Ok(Some(scx_accel::GpuDevice::new(id).map_err(|e| {
                PyRuntimeError::new_err(format!("GPU NB-GLM device init failed: {e}"))
            })?)),
            None => Ok(None),
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = gpu_device_id;
        Ok(None)
    }
}

/// Fit one pseudobulk NB-GLM on the GPU when `gpu_dev` is `Some` **and** the dims
/// fit the kernel, else on the CPU. The per-fit dim guard means a single
/// over-large target in a many-target sweep silently uses the CPU rather than
/// erroring — the dominant (in-bounds) targets still take the GPU route.
#[allow(clippy::too_many_arguments)]
fn fit_one(
    py: Python<'_>,
    gpu_dev: &NbGlmGpuDev,
    cg: &[f64],
    n_genes: usize,
    n_sub: usize,
    design: &[f64],
    n_features: usize,
    size_factors: Option<&[f64]>,
    contrast: NbGlmContrast,
    options: NbGlmOptions,
) -> Result<NbGlmResult, scx_accel::AccelError> {
    #[cfg(feature = "gpu")]
    {
        if let Some(dev) = gpu_dev {
            if gpu_dims_ok(n_features, n_sub) {
                // The `GpuDevice` handle is `!Send`, so we cannot release the GIL
                // around the device fit; it is short (the fit runs on-device).
                return scx_accel::gpu_pseudobulk_nb_glm(
                    dev,
                    cg,
                    n_genes,
                    n_sub,
                    design,
                    n_features,
                    size_factors,
                    contrast,
                    options,
                );
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = gpu_dev;
    }
    // The CPU fit is pure Rust (gene-parallel rayon) — release the GIL for its
    // duration so other Python threads run, matching the sibling accel ops.
    py.detach(|| {
        scx_accel::pseudobulk_nb_glm(
            cg,
            n_genes,
            n_sub,
            design,
            n_features,
            size_factors,
            contrast,
            options,
        )
    })
}

/// Copy a 2-D array-like into a row-major `Vec<f64>` plus its `(rows, cols)`.
fn dense2d_f64(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, usize, usize)> {
    let np = crate::pyimport::import_module(py, "numpy")?;
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
    let np = crate::pyimport::import_module(py, "numpy")?;
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
    let mask = numpy::PyArray1::from_vec(py, keep);
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

// `"contrast"` is intentionally NOT a recognised `NbGlmOptions` key: the low-level
// `nb_glm` pyfunction has its own `contrast=` argument and must still reject a
// stray `options={"contrast": …}` as an unknown key (fail-loud, per §2.4). The
// design-aware `pseudobulk_dex`/`pdex_nb_glm` wrappers read + strip `"contrast"`
// from `nbglm_options` themselves (see `take_contrast_override`) before parsing.
const NBGLM_CONTRAST_KEY: &str = "contrast";

/// `(explicit contrast object or None, nbglm_options dict with "contrast" removed)`.
type ContrastAndOptions<'py> = (Option<Bound<'py, PyAny>>, Option<Bound<'py, PyDict>>);

/// Split an optional explicit `contrast` out of an `nbglm_options` dict: returns
/// the (Python) contrast object (only when present **and not** `None`, so an
/// explicit `{"contrast": None}` means "no override") plus a copy of the dict with
/// `"contrast"` removed — the latter is safe to pass to [`nbglm_options_from_dict`]
/// (which rejects unknown keys). Returns `(None, original)` when no dict was given.
pub(super) fn take_contrast_override<'py>(
    nbglm_options: Option<&Bound<'py, PyDict>>,
) -> PyResult<ContrastAndOptions<'py>> {
    let Some(d) = nbglm_options else {
        return Ok((None, None));
    };
    let contrast = d.get_item(NBGLM_CONTRAST_KEY)?.filter(|o| !o.is_none());
    if contrast.is_none() {
        return Ok((None, Some(d.clone())));
    }
    let rest = d.copy()?;
    rest.del_item(NBGLM_CONTRAST_KEY)?;
    Ok((contrast, Some(rest)))
}

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
    gpu_device_id: Option<usize>,
) -> PyResult<Vec<TargetFit>> {
    let gpu_dev = make_gpu_dev(gpu_device_id)?;
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

    // Per-target loop is **serial** by design: each fit is already gene-parallel
    // (rayon) and the per-target host tail (MTC etc.) is parallelized internally.
    // Parallelizing this outer loop was measured to *lower* the GPU/CPU speedup —
    // it speeds the CPU baseline (whose fit overlaps across targets) far more than
    // the GPU path (whose fit is off-CPU), and nests rayon. The GPU device handle
    // is `!Sync` and must stay serial regardless.
    let warnings = crate::pyimport::import_module(py, "warnings")?;
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

        // Gene-major counts [n_vars × n_sub] and design [n_sub × 2].
        let mut cg = vec![0.0_f64; n_vars * n_sub];
        let mut design = vec![0.0_f64; n_sub * 2];
        for (s, &g) in sub.iter().enumerate() {
            design[s * 2] = 1.0;
            design[s * 2 + 1] = if cond[g] == target.as_str() { 1.0 } else { 0.0 };
        }
        {
            let _t =
                scx_accel::nb_glm::profile::start(scx_accel::nb_glm::profile::Phase::Transpose);
            for j in 0..n_vars {
                for (s, &g) in sub.iter().enumerate() {
                    cg[j * n_sub + s] = result.counts[g * n_vars + j];
                }
            }
        }

        match fit_one(
            py,
            &gpu_dev,
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
    gpu_device_id: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let fits = fit_targets_nbglm(py, result, cond_col_idx, reference, options, gpu_device_id)?;
    if fits.is_empty() {
        return Err(PyRuntimeError::new_err(
            "NB-GLM produced no fittable contrasts: every target had < 2 pseudobulk \
             replicates per condition. A pseudobulk NB-GLM needs replicates — include a \
             batch/donor/well column in `groupby`. For no-replicate layouts use \
             de_method=\"pdex_ref\" or \"wilcoxon\" (per-cell tests).",
        ));
    }
    fits_to_pandas(py, result, reference, &fits)
}

/// Assemble per-target NB-GLM fits into the PyDESeq2-style pandas schema
/// (`gene, baseMean, log2FoldChange, lfcSE, stat, pvalue, padj, target,
/// reference`). Shared by the default and design-aware `pseudobulk_dex` paths.
fn fits_to_pandas<'py>(
    py: Python<'py>,
    result: &PseudobulkResult,
    reference: &str,
    fits: &[TargetFit],
) -> PyResult<Bound<'py, PyAny>> {
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
    for tf in fits {
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

    let pd = crate::pyimport::import_module(py, "pandas")?;
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

/// Resolve the design-matrix column index for the `test_col` treatment coefficient
/// of `target`. Matches the bare `test_col[T.target]` name first, then any
/// treatment-coded wrapper (`C(test_col, …)[T.target]`) via a *unique*
/// contains-`test_col` + `[T.target]`-suffix match. Returns `None` if absent or
/// ambiguous (so the caller fails loud rather than picking an arbitrary column).
fn resolve_contrast_column(columns: &[String], test_col: &str, target: &str) -> Option<usize> {
    let exact = format!("{test_col}[T.{target}]");
    if let Some(i) = columns.iter().position(|c| c == &exact) {
        return Some(i);
    }
    let suffix = format!("[T.{target}]");
    let mut matches = columns
        .iter()
        .enumerate()
        .filter(|(_, c)| c.contains(test_col) && c.ends_with(&suffix))
        .map(|(i, _)| i);
    let first = matches.next()?;
    match matches.next() {
        Some(_) => None, // ambiguous (>1 match)
        None => Some(first),
    }
}

/// Design-aware sibling of [`fit_targets_nbglm`]: builds the pseudobulk-*sample*
/// design matrix from a formulaic `design_formula` over the groupby columns, fits
/// **one** full-design NB-GLM across all samples (shared dispersion,
/// covariate-adjusted), and extracts one contrast per non-reference `test_col`
/// level — or a single explicit `contrast_override`. DESeq2-*style*, not
/// -*identical*. The formula references the groupby columns; `reference` is coded
/// as the base level of `test_col` (ordered Categorical) so the default per-target
/// contrast is `test_col[T.<target>]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_targets_nbglm_with_design(
    py: Python<'_>,
    result: &PseudobulkResult,
    test_col_idx: usize,
    reference: &str,
    design_formula: &str,
    contrast_override: Option<&Bound<'_, PyAny>>,
    options: &NbGlmOptions,
    gpu_device_id: Option<usize>,
) -> PyResult<Vec<TargetFit>> {
    let gpu_dev = make_gpu_dev(gpu_device_id)?;
    let n_groups = result.n_groups;
    let n_vars = result.n_vars;
    let test_col = result.groupby_columns[test_col_idx].as_str();

    // test_col levels, `reference` first so treatment coding uses it as the base.
    let mut uniq: Vec<&str> = (0..n_groups)
        .map(|g| result.group_labels[g][test_col_idx].as_str())
        .collect();
    uniq.sort_unstable();
    uniq.dedup();
    if !uniq.contains(&reference) {
        return Err(PyValueError::new_err(format!(
            "reference level '{reference}' not found in test_col '{test_col}'"
        )));
    }
    let mut ordered_levels: Vec<&str> = vec![reference];
    ordered_levels.extend(uniq.into_iter().filter(|l| *l != reference));
    let targets: Vec<String> = ordered_levels[1..].iter().map(|s| s.to_string()).collect();

    // --- metadata_df: rows = pseudobulk samples, cols = groupby columns. ---
    let pd = crate::pyimport::import_module(py, "pandas")?;
    let meta_dict = PyDict::new(py);
    for (col_idx, col_name) in result.groupby_columns.iter().enumerate() {
        let vals: Vec<String> = (0..n_groups)
            .map(|g| result.group_labels[g][col_idx].clone())
            .collect();
        if col_idx == test_col_idx {
            // Ordered Categorical with `reference` first → its treatment-coding
            // base level, so `test_col[T.<target>]` is target-vs-reference.
            let kw = PyDict::new(py);
            kw.set_item("categories", &ordered_levels)?;
            kw.set_item("ordered", true)?;
            let cat = pd.getattr("Categorical")?.call((vals,), Some(&kw))?;
            meta_dict.set_item(col_name.as_str(), cat)?;
        } else {
            meta_dict.set_item(col_name.as_str(), vals)?;
        }
    }
    let sample_names: Vec<String> = (0..n_groups).map(|i| format!("sample_{i}")).collect();
    let sample_index = pd.call_method1("Index", (sample_names,))?;
    let df_kw = PyDict::new(py);
    df_kw.set_item("index", &sample_index)?;
    let metadata_df = pd.call_method("DataFrame", (meta_dict,), Some(&df_kw))?;

    // --- Build the numeric design matrix via formulaic (pydeseq2's parser). ---
    let formulaic = crate::optional_deps::import_optional_with_hint(
        py,
        "formulaic",
        crate::optional_deps::EXTRA_NBGLM,
        "a `design=` formula on the NB-GLM backend",
        "formulaic",
        Some("Omit `design` to use the fixed intercept + target design, which needs no extra."),
    )?;
    let x = formulaic.call_method1("model_matrix", (design_formula, &metadata_df))?;
    // A one-sided formula (`~ rhs`) yields a single ModelMatrix (DataFrame-like); a
    // two-sided / multi-part formula yields a `ModelMatrices` with no `.columns`.
    if !x.hasattr("columns")? {
        return Err(PyValueError::new_err(format!(
            "design formula {design_formula:?} must be one-sided (`~ terms`); a two-sided \
             or multi-part formula is not supported for the NB-GLM design."
        )));
    }
    let columns: Vec<String> = x.getattr("columns")?.call_method0("tolist")?.extract()?;
    let (design_vec, ds_rows, n_features) = dense2d_f64(py, &x)?;
    if ds_rows != n_groups {
        return Err(PyValueError::new_err(format!(
            "design formula {design_formula:?} produced {ds_rows} rows but there are \
             {n_groups} pseudobulk samples"
        )));
    }
    // Full column rank needs more samples than columns. The engine also guards this
    // (validate::validate_inputs), but reports raw dims — give an actionable message.
    if n_groups <= n_features {
        return Err(PyValueError::new_err(format!(
            "design formula {design_formula:?} has {n_features} columns but only \
             {n_groups} pseudobulk samples — the model is under-determined. Reduce the \
             design or add replicate samples (a batch/donor/well column in `groupby`)."
        )));
    }
    // Guard the per-contrast-refit scale cliff: each contrast re-runs the full
    // O(n_genes·n_features²·iters) IRLS fit (until the fit-once/test-many engine
    // entry lands — see docs/pseudobulk_nb_glm.md). Refuse pathological design
    // widths / level counts rather than silently launching an hours-long job.
    const MAX_DESIGN_FEATURES: usize = 100;
    if n_features > MAX_DESIGN_FEATURES {
        return Err(PyValueError::new_err(format!(
            "design formula {design_formula:?} has {n_features} columns (> \
             {MAX_DESIGN_FEATURES}); the NB-GLM design path re-fits per contrast and would \
             be prohibitively slow. Reduce the design width, or use pyscx.accel.nb_glm for \
             a single explicit contrast on a prebuilt design."
        )));
    }

    // Full-sample gene-major counts [n_vars × n_groups] (all samples, not a subset).
    let mut cg = vec![0.0_f64; n_vars * n_groups];
    for j in 0..n_vars {
        for s in 0..n_groups {
            cg[j * n_groups + s] = result.counts[s * n_vars + j];
        }
    }

    // --- Contrasts: explicit override, or one per non-reference target level. ---
    let contrast_specs: Vec<(String, NbGlmContrast)> = if let Some(obj) = contrast_override {
        let c = contrast_from_pyany(Some(obj), n_features)?;
        let label = match &c {
            NbGlmContrast::Coefficient { index } => columns
                .get(*index)
                .cloned()
                .unwrap_or_else(|| format!("coef[{index}]")),
            NbGlmContrast::Vector { .. } => "custom_contrast".to_string(),
        };
        vec![(label, c)]
    } else {
        if targets.len() > MAX_DESIGN_FEATURES {
            return Err(PyValueError::new_err(format!(
                "test_col '{test_col}' has {} non-reference levels (> {MAX_DESIGN_FEATURES}); \
                 the design path re-fits per level and would be prohibitively slow. Use \
                 fewer levels, or pyscx.accel.nb_glm per contrast.",
                targets.len()
            )));
        }
        let mut specs = Vec::with_capacity(targets.len());
        for target in &targets {
            let idx = resolve_contrast_column(&columns, test_col, target).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "design formula {design_formula:?} does not produce a unique treatment \
                     coefficient for target level '{target}' of test_col '{test_col}' \
                     (looked for a column `{test_col}…[T.{target}]`). The design must include \
                     `{test_col}` with treatment coding and an intercept (no-intercept `~ 0 + \
                     …` is not supported). Design columns: {columns:?}"
                ))
            })?;
            specs.push((target.clone(), NbGlmContrast::Coefficient { index: idx }));
        }
        specs
    };

    // Fit the full design once per contrast. NOTE(perf, 3.11 follow-up): the IRLS +
    // dispersion fit is contrast-independent, so every call after the first re-runs an
    // identical deterministic fit — a fit-once/test-many CPU seam (mirroring the GPU
    // `gpu_nb_glm_fit_states` + `finalize_nb_glm` split) would remove the O(k) waste.
    let warnings = crate::pyimport::import_module(py, "warnings")?;
    let mut fits = Vec::with_capacity(contrast_specs.len());
    for (label, contrast) in contrast_specs {
        match fit_one(
            py,
            &gpu_dev,
            &cg,
            n_vars,
            n_groups,
            &design_vec,
            n_features,
            None,
            contrast,
            options.clone(),
        ) {
            Ok(r) => fits.push(TargetFit {
                target: label,
                result: r,
            }),
            Err(e) => {
                warnings
                    .call_method1("warn", (format!("NB-GLM: contrast '{label}' failed: {e}"),))?;
            }
        }
    }
    Ok(fits)
}

/// Design-aware `pseudobulk_dex(backend="nb_glm")` assembly: fit with a formula
/// design + optional explicit contrast, then emit the standard pandas schema.
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_targets_pandas_with_design<'py>(
    py: Python<'py>,
    result: &PseudobulkResult,
    test_col_idx: usize,
    reference: &str,
    design_formula: &str,
    contrast_override: Option<&Bound<'_, PyAny>>,
    options: &NbGlmOptions,
    gpu_device_id: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let fits = fit_targets_nbglm_with_design(
        py,
        result,
        test_col_idx,
        reference,
        design_formula,
        contrast_override,
        options,
        gpu_device_id,
    )?;
    if fits.is_empty() {
        return Err(PyRuntimeError::new_err(
            "NB-GLM produced no fittable contrasts for the supplied design. Check that \
             the design is full column rank and there are more pseudobulk samples than \
             design columns (include a batch/donor/well replicate column in `groupby`).",
        ));
    }
    fits_to_pandas(py, result, reference, &fits)
}

/// Fit a Rust-native negative-binomial GLM on an already-pseudobulked count
/// matrix and test a contrast (a DESeq2 replacement). `f64` throughout;
/// `device="auto"|"cpu"|"gpu"|"gpu:N"` selects CPU vs the Stage-A GPU fitter
/// (the GPU path requires `n_features ≤ 8` and `n_samples ≤ 64`, else it falls
/// back to CPU).
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
#[pyo3(signature = (counts, design, size_factors=None, contrast=None, gene_names=None, sample_names=None, options=None, counts_axis="samples_by_genes", device="auto"))]
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
    device: &str,
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

    // Resolve device + plan the route. `nb_glm` operates on raw arrays (no
    // AnnData), so there is no `uns` to stamp — but we still announce a silent
    // GPU→CPU fallback so an explicit `device="gpu"` with over-large dims warns.
    let resolved = super::gpu::resolve_device(device)?;
    let gpu_device_id = resolve_gpu_id(resolved);
    let gpu_eligible = gpu_dims_ok(n_features, n_samples);
    let info = super::route::nb_glm_exec_info(device, gpu_eligible);
    super::route::announce_route(py, "nb_glm", device, &info);
    let gpu_dev = make_gpu_dev(if info.route.is_gpu() {
        gpu_device_id
    } else {
        None
    })?;

    let res = fit_one(
        py,
        &gpu_dev,
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

    let pd = crate::pyimport::import_module(py, "pandas")?;
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

/// Snapshot the NB-GLM per-phase wall-time profiler (Stage B0).
///
/// Returns a dict mapping each phase name (`transpose`, `cpu_mle_pass`,
/// `cpu_shrink_pass`, `gpu_mle_fit`, `gpu_trend_prior`, `gpu_shrink_fit`,
/// `wald_cooks`, `mtc`, `assembly`) to a `{ms, calls}` sub-dict. Accumulates
/// only when the process was started with `SCX_NBGLM_PROFILE=1`; otherwise every
/// bucket reads zero. Mirrors `gpu_profile_snapshot`. The `mtc` (multiple-testing
/// correction: independent filter + BH) and `wald_cooks` buckets quantify the
/// per-target host tail that bounds the GPU speedup (the GPU fit itself is a
/// small fraction of wall on the many-target sweep).
#[pyfunction]
pub fn nb_glm_profile_snapshot(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    for stat in scx_accel::nb_glm::profile::snapshot() {
        let d = PyDict::new(py);
        d.set_item("ms", stat.ms)?;
        d.set_item("calls", stat.calls)?;
        dict.set_item(stat.name, d)?;
    }
    Ok(dict.into_any().unbind())
}

/// Reset the NB-GLM per-phase profiler counters to zero (call between bench runs).
#[pyfunction]
pub fn nb_glm_profile_reset() -> PyResult<()> {
    scx_accel::nb_glm::profile::reset();
    Ok(())
}

/// Pseudobulk NB-GLM differential expression for the cell-eval/pdex consumer.
///
/// Aggregates `groupby × stratify_by` pseudobulk **replicates** from `adata`,
/// fits one Rust-native NB-GLM per non-reference perturbation vs `reference`, and
/// returns the cell-eval `DEResults` column schema (`target, feature,
/// fold_change, p_value, fdr, log2_fold_change, abs_log2_fold_change`) — the same
/// schema as `rank_genes_groups_df` / `pdex_ref`. `output="pandas"` (the default)
/// needs no optional dependency; **pass `output="polars"` to feed `cell_eval`**,
/// whose `DEResults.data` is typed `pl.DataFrame` and rejects a pandas frame.
/// `device="auto"|"cpu"|"gpu"|"gpu:N"` selects the fitter (GPU =
/// Stage-A host-fed dense kernel); records the planned route on
/// `adata.uns["scx_accel"]["pdex_nb_glm"]` (`gpu_nb_glm_csr` / `cpu_nb_glm`).
///
/// A pseudobulk NB-GLM needs ≥ 2 samples per condition to estimate dispersion, so
/// `stratify_by` (a **list** of batch/donor/well/replicate obs columns, e.g.
/// `["donor"]`, jointly spanning ≥ 2 strata) is **required** — `pdex_nb_glm`
/// raises a `ValueError` pointing to `pdex_ref` / `rank_genes_groups` when it is
/// absent. Pass a list even for a single column; a bare string is rejected with a
/// clear error. NB-GLM needs **raw counts**: log1p
/// input (auto-detected via `adata.uns["log1p"]`, or `is_log1p=True`) is
/// rejected. DESeq2-*style*, not DESeq2-*identical*. See
/// docs/pseudobulk_nb_glm.md.
///
/// `design` (optional, §3.11): a formulaic formula over `groupby` + `stratify_by`
/// (e.g. `"~ perturbation + donor"`) to fit a covariate-adjusted joint model
/// (shared dispersion) with one contrast per non-reference `groupby` level, instead
/// of the default fixed `[intercept, is_target]` per-target fit. Requires
/// `formulaic`. Custom designs run on **CPU** (the GPU kernel is `p ≤ 8`). An
/// explicit `nbglm_options["contrast"]` is rejected here (the cell-eval schema is
/// keyed by perturbation name — use `pseudobulk_dex`/`accel.nb_glm` for that).
#[pyfunction]
#[pyo3(signature = (adata, groupby, reference, stratify_by=None, min_cells_per_group=10, min_cells_per_stratum=50, is_log1p=None, nbglm_options=None, gene_chunk_size=None, prefer_format="csr", device="auto", design=None, output="pandas"))]
#[allow(clippy::too_many_arguments)]
pub fn pdex_nb_glm(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    groupby: &str,
    reference: &str,
    stratify_by: Option<&Bound<'_, PyAny>>,
    min_cells_per_group: usize,
    min_cells_per_stratum: usize,
    is_log1p: Option<bool>,
    nbglm_options: Option<&Bound<'_, PyDict>>,
    gene_chunk_size: Option<usize>,
    prefer_format: &str,
    device: &str,
    design: Option<&str>,
    output: &str,
) -> PyResult<Py<PyAny>> {
    // Validated up front, mirroring `rank_genes_groups_df` / `pdex_ref`: a typo
    // must not cost a full NB-GLM fit before it is reported.
    if !matches!(output, "polars" | "pandas") {
        return Err(PyValueError::new_err(format!(
            "Invalid output={output:?}; expected 'polars' or 'pandas'"
        )));
    }

    // A presentation-ordered backed `X` (`preserve_var_order=True`) has no
    // `ShardSource` spelling, and the shared pseudobulk aggregation now
    // streams the handle's view — so a request-ordered gene axis would be a
    // silent permutation against `adata.var` rather than a shape error.
    super::prepare_target(py, adata, "pdex_nb_glm")?;

    // `stratify_by` is list-only; turn the opaque PyO3 `Can't extract 'str' to
    // 'Vec'` into an actionable message when a bare string slips through.
    let stratify_by: Option<Vec<String>> = match stratify_by {
        None => None,
        Some(obj) => Some(obj.extract::<Vec<String>>().map_err(|_| {
            PyValueError::new_err(
                "stratify_by must be a list of obs column names \
                 (e.g. stratify_by=[\"donor\"]), not a bare string",
            )
        })?),
    };

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
        crate::pyimport::import_module(py, "warnings")?.call_method1(
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
        None => super::util::uns_log1p_present(adata),
    };
    if is_log1p {
        return Err(PyValueError::new_err(
            "pdex_nb_glm requires raw counts but the data looks log1p-normalized \
             (adata.uns['log1p'] present, or is_log1p=True). Pass raw counts, or use \
             de_method=\"pdex_ref\"/\"wilcoxon\".",
        ));
    }

    // §3.11: `pdex_nb_glm` honors a `design` FORMULA (covariate-adjusted joint fit,
    // per-perturbation contrasts). An explicit `nbglm_options["contrast"]` is NOT
    // accepted here: its single arbitrary contrast has no perturbation-name `target`,
    // and the cell-eval schema this returns is keyed by perturbation name — a
    // coefficient-name target would silently fail the downstream join.
    let (contrast_override, options_dict) = take_contrast_override(nbglm_options)?;
    if contrast_override.is_some() {
        return Err(PyValueError::new_err(
            "pdex_nb_glm does not accept an explicit nbglm_options[\"contrast\"]: it emits \
             the cell-eval schema keyed by perturbation name, so only the automatic \
             per-perturbation contrasts are valid. Pass a `design` formula, or use \
             pseudobulk_dex(backend=\"nb_glm\") / accel.nb_glm for a custom contrast.",
        ));
    }
    let design_aware = design.is_some();

    // Resolve device + plan the route **before** the expensive aggregation, so an
    // invalid `device=` string or `device="gpu"` on a CPU-only build / no-GPU host
    // fails fast (rather than after a full pseudobulk pass). The DEFAULT path's
    // per-target design is `[intercept, is_target]` (p=2 ≤ PMAX) so the GPU kernel
    // fits → `gpu_eligible = true` (the per-target `n_sub` guard in `fit_one` may
    // still route an over-large target to CPU). The design-aware path fits the FULL
    // formula width (routinely p > GPU_NB_GLM_PMAX = 8) across all samples, so it
    // runs on CPU — stamp it honestly as CPU (§4.1 route honesty) and skip the CUDA
    // context entirely.
    let resolved = super::gpu::resolve_device(device)?;
    let info = super::route::nb_glm_exec_info(device, !design_aware);
    super::route::announce_route(py, "pdex_nb_glm", device, &info);
    let gpu_device_id = if info.route.is_gpu() {
        resolve_gpu_id(resolved)
    } else {
        None
    };

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

    let result = {
        let _t = scx_accel::nb_glm::profile::start(scx_accel::nb_glm::profile::Phase::Aggregate);
        aggregate_pseudobulk(
            py,
            &working,
            &combined,
            prefer_format,
            None,
            scx_accel::AggregationMethod::Sum,
            min_cells_per_group,
        )?
    };
    if result.n_groups == 0 {
        return Err(PyRuntimeError::new_err(
            "no pseudobulk groups passed the min_cells_per_group filter",
        ));
    }

    let options = nbglm_options_from_dict(py, options_dict.as_ref())?;
    // §3.11: a `design` formula fits one full-design model (groupby=column 0,
    // stratifiers available as covariates) with per-perturbation contrasts;
    // otherwise the fixed [intercept, is_target] per-target path.
    let fits = if let Some(formula) = design {
        fit_targets_nbglm_with_design(
            py,
            &result,
            0,
            reference,
            formula,
            None,
            &options,
            gpu_device_id,
        )?
    } else {
        fit_targets_nbglm(py, &result, 0, reference, &options, gpu_device_id)?
    };
    if fits.is_empty() {
        return Err(PyRuntimeError::new_err(if design_aware {
            "NB-GLM produced no fittable contrasts for the supplied design: check the \
             design is full column rank and there are more pseudobulk samples than \
             design columns (add a batch/donor/well replicate column in `groupby`)."
        } else {
            "no perturbation had enough pseudobulk replicates to fit NB-GLM; \
             check stratify_by spans ≥2 strata per perturbation"
        }));
    }

    // Stamp the planned route on adata.uns["scx_accel"]["pdex_nb_glm"]. NOTE: a
    // stamped `gpu_nb_glm_csr` reflects the *intended* route; an individual target
    // whose pseudobulk has n_sub > GPU_NB_GLM_NSUB_MAX falls back to CPU inside
    // `fit_one` (rare — n_sub is 2·replicates), so a GPU-stamped sweep may have run
    // a few targets on CPU.
    super::route::write_accel_route(py, adata, "pdex_nb_glm", &info)?;

    // Flatten to the cell-eval/pdex column vectors.
    let _flatten_timer =
        scx_accel::nb_glm::profile::start(scx_accel::nb_glm::profile::Phase::Flatten);
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
    drop(_flatten_timer);

    let df = {
        let _t = scx_accel::nb_glm::profile::start(scx_accel::nb_glm::profile::Phase::Dataframe);
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
        )?
    };
    Ok(df.unbind())
}
