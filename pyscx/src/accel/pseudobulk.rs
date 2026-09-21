//! Pseudobulk differential expression via Rust aggregation + pydeseq2.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::ScxBackedSparseDataset;
use crate::optional_deps::{import_optional, EXTRA_PYDESEQ2};

use super::de::extract_strata;

/// Upper bound for the env-derived / auto-probed worker count. An explicit
/// `n_cpus` argument bypasses this so callers can opt into larger pools.
const DESEQ_DEFAULT_MAX_CPUS: usize = 8;

/// The DE engine `pseudobulk_dex` uses when the caller does not name one.
///
/// Was `"pydeseq2"` through v0.12. pydeseq2 is an *optional* dependency that no
/// extra installed, so the flagship pseudobulk surface raised on a base install
/// while the Rust-native NB-GLM — which needs nothing — sat behind an opt-in.
const DEFAULT_BACKEND: &str = "nb_glm";

/// How to refer to the backend in a guard message. A caller who never typed
/// `backend=` must not be told their kwarg is at fault.
fn backend_phrase(explicit: bool) -> &'static str {
    if explicit {
        "backend=\"nb_glm\""
    } else {
        "the default backend (\"nb_glm\", which replaced \"pydeseq2\" as the default)"
    }
}

/// The escape hatch, appended only when the caller did not choose the backend —
/// someone who explicitly asked for NB-GLM is not looking for a way back to
/// pydeseq2.
fn pydeseq2_way_back(explicit: bool) -> &'static str {
    if explicit {
        ""
    } else {
        " To keep the previous behaviour pass backend=\"pydeseq2\" \
         (pip install 'pyscx[pydeseq2]')."
    }
}

/// Set once the transition warning below has been emitted, so "once per
/// process" is a latch rather than a hope.
///
/// Python's own dedup keys on the *caller's* module and line, and `warnings`
/// filters are user-controlled (`simplefilter("always")` in notebooks and in
/// this project's own tests), so leaving it to the warnings machinery would make
/// a call in a loop warn on every iteration. Mirrors `NO_RAPIDS_WARNED` in
/// `rapids.rs`.
static DEFAULT_BACKEND_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Warn, once per process, that a default-backend call would previously have run
/// pydeseq2 and now runs NB-GLM.
///
/// Deliberately gated on pydeseq2 being **importable**: that is exactly the set
/// of callers whose numbers move under them. A base install has nothing to be
/// warned about — its previous behaviour was a `RuntimeError` — and telling a
/// new user that a default they never saw has changed is pure noise.
///
/// `find_spec` rather than a real import: importing pydeseq2 costs seconds and
/// pulls in statsmodels, which would be an absurd price for a warning check.
///
/// **Infallible by construction, and that is the contract, not an oversight.**
/// Returns `()`, so there is no `?` to add later. `warnings.warn` *raises* under
/// `-W error` / `simplefilter("error")` — a common pytest and CI setting — so a
/// `?` anywhere on this path makes a courtesy notice about a default change
/// abort the caller's DE run. Every step is therefore ignored on failure.
fn warn_default_backend_changed(py: Python<'_>) {
    if DEFAULT_BACKEND_WARNED.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    // `find_spec` can itself raise (a missing parent package, a module whose
    // `__spec__` is None, a custom meta-path finder). Treat "cannot tell" as
    // "not available" and stay quiet.
    let available = crate::pyimport::import_module(py, "importlib.util")
        .and_then(|m| m.call_method1("find_spec", ("pydeseq2",)))
        .map(|spec| !spec.is_none())
        .unwrap_or(false);
    if !available {
        // Deliberately not latched: a caller can install pydeseq2 and re-run in
        // the same process (a notebook), and they should still hear about it.
        return;
    }
    if DEFAULT_BACKEND_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    if let Ok(warnings) = crate::pyimport::import_module(py, "warnings") {
        let _ = warnings.call_method1(
            "warn",
            (
                "pseudobulk_dex() now defaults to backend=\"nb_glm\" (the Rust-native \
                 negative-binomial GLM); it defaulted to \"pydeseq2\" through v0.12. \
                 NB-GLM is DESeq2-*style*, not DESeq2-identical, and applies Cook's / \
                 independent filtering by default, so padj can be NaN for outlier and \
                 low-base-mean genes. Pass backend=\"pydeseq2\" to keep the previous \
                 numerics, or backend=\"nb_glm\" to silence this.",
                py.get_type::<pyo3::exceptions::PyUserWarning>(),
            ),
        );
    }
}

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
    let obs_groups = extract_obs_group_columns(adata, groupby)?;

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
            let source = backed
                .as_column_source()
                .ok_or_else(crate::accel::csc_unavailable)?;
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
            let lazy_src = lazy
                .as_column_source()
                .ok_or_else(crate::accel::csc_unavailable)?;
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
        // The handle's *view*: `obs_groups` came from `adata.obs`, so it is one
        // label per *visible* cell. Streaming the raw reader would walk every
        // on-disk row and trip the kernel's `obs_groups` length guard on any
        // subset handle (`filter_cells` → `pseudobulk` used to raise).
        let source = backed.as_shard_source();
        scx_accel::pseudobulk_aggregate(
            &source,
            &obs_groups,
            groupby,
            &gene_names,
            method,
            min_cells_per_group,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    } else {
        // In-memory: extract scipy CSR → ScxCsr.
        let scipy_sparse = crate::pyimport::import_module(py, "scipy.sparse")?;
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
            let np = crate::pyimport::import_module(py, "numpy")?;
            let arr = np
                .call_method1("asarray", (&x,))?
                .call_method1("astype", ("float32",))?;
            scipy_sparse.call_method1("csr_matrix", (&arr,))?
        };

        let shape: (usize, usize) = csr_obj.getattr("shape")?.extract()?;
        let np = crate::pyimport::import_module(py, "numpy")?;
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

/// Read the obs columns that define a pseudobulk group, one `Vec<String>` of
/// per-cell labels per column (`astype("str")`, so categoricals and integers
/// come out as their string form). The composite key over several columns is
/// built downstream by `scx_accel::build_group_mapping`; this is the one
/// extraction both `pseudobulk_dex` and `pseudobulk_means` go through.
pub(crate) fn extract_obs_group_columns(
    adata: &Bound<'_, PyAny>,
    groupby: &[String],
) -> PyResult<Vec<Vec<String>>> {
    let obs = adata.getattr("obs")?;
    let mut obs_groups: Vec<Vec<String>> = Vec::with_capacity(groupby.len());
    for col_name in groupby {
        let col = obs.get_item(col_name.as_str()).map_err(|_| {
            PyValueError::new_err(format!(
                "groupby column '{col_name}' not found in adata.obs"
            ))
        })?;
        let labels: Vec<String> = col
            .call_method1("astype", ("str",))?
            .call_method0("tolist")?
            .extract()?;
        obs_groups.push(labels);
    }
    Ok(obs_groups)
}

/// Coerce a `str | list[str]` column argument to a column list.
///
/// pyo3 refuses `str` → `Vec<String>` (correctly — it would otherwise
/// char-split), but the resulting `TypeError: argument 'sample_cols': Can't
/// extract 'str' to 'Vec'` names neither the fix nor the sibling kwarg, which
/// is precisely the unhelpful landing F8 exists to remove. So `groupby`, the
/// two aliases, and `pseudobulk_means`' `groupby` all go through here instead
/// of being typed `Vec<String>`.
pub(crate) fn coerce_column_list(name: &str, value: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(one) = value.extract::<String>() {
        return Ok(vec![one]);
    }
    value.extract::<Vec<String>>().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be an obs column name (str) or a list of obs column names; \
             got {}",
            value
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "an unknown type".to_string())
        ))
    })
}

/// Resolve `groupby` / `sample_cols` / `sample_key` down to the one list of obs
/// columns that defines a pseudobulk sample (dogfood F8).
///
/// These are three spellings of one parameter, so supplying more than one is a
/// user error rather than something to merge. All three accept a bare string
/// as well as a list — `sample_key="donor_id"` is the exact spelling a user
/// reaching for the replicate role types, and `groupby="donor_id"` is what
/// `pseudobulk_means(groupby="…")` had always taken, so none may raise on it.
fn resolve_sample_columns(
    groupby: Option<&Bound<'_, PyAny>>,
    sample_cols: Option<&Bound<'_, PyAny>>,
    sample_key: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<String>> {
    let mut supplied: Vec<&str> = Vec::new();
    if groupby.is_some() {
        supplied.push("groupby");
    }
    if sample_cols.is_some() {
        supplied.push("sample_cols");
    }
    if sample_key.is_some() {
        supplied.push("sample_key");
    }
    if supplied.len() > 1 {
        return Err(PyValueError::new_err(format!(
            "pass only one of groupby / sample_cols / sample_key (got {}); they are three \
             spellings of the same parameter — the obs columns that together define a \
             pseudobulk sample",
            supplied.join(", ")
        )));
    }

    if let Some(v) = groupby {
        return coerce_column_list("groupby", v);
    }
    if let Some(v) = sample_cols {
        return coerce_column_list("sample_cols", v);
    }
    if let Some(v) = sample_key {
        return coerce_column_list("sample_key", v);
    }

    // The role contrast belongs here, in the message a user hits when they
    // guessed wrong, not only in the docs they did not open.
    Err(PyValueError::new_err(
        "pseudobulk_dex requires the obs columns that define a pseudobulk sample — pass \
         groupby=, sample_cols=, or sample_key=. These are condition PLUS replicate, e.g. \
         groupby=[\"disease\", \"donor_id\"] with test_col=\"disease\". Note this is the \
         opposite of rank_genes_groups(groupby=...), where `groupby` IS the compared \
         column; here the compared column is `test_col`, and the replicate column \
         (batch/donor/well) is what supplies the replication.",
    ))
}

/// Pseudobulk differential expression via Rust aggregation + pydeseq2.
///
/// **`groupby` here does not mean what it means in `rank_genes_groups`.** In
/// `accel.rank_genes_groups` (and throughout scanpy) `groupby` names the column
/// whose levels are compared. In `pseudobulk_dex` it names the columns that
/// together define one pseudobulk *sample* — condition **plus** replicate, e.g.
/// `["disease", "donor_id"]` — and the column being compared is `test_col`.
/// Passing only the condition column yields one sample per condition and so no
/// replication, which is why the replicate-role spellings `sample_cols=` /
/// `sample_key=` are accepted as aliases for `groupby`.
///
/// Aggregates single-cell counts into pseudobulk samples by grouping cells
/// according to those columns, then uses `pydeseq2` for negative binomial GLM
/// testing.
///
/// Args:
///     adata: AnnData object with X and the obs columns named below
///     groupby: Obs columns defining a pseudobulk sample — condition + replicate
///         (e.g. ["disease", "donor_id"]). NOT the compared column; see above.
///     sample_cols: Alias for `groupby`, named for the role it plays. Pass one
///         of `groupby` / `sample_cols` / `sample_key`, not several.
///     sample_key: Single-column alias for `groupby`; accepts a bare string
///         (e.g. sample_key="donor_id") and wraps it in a one-element list.
///     test_col: Which of those columns holds the condition to compare
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
///     backend: DE engine — "nb_glm" (default; Rust-native negative-binomial
///         GLM, no optional dependency) or "pydeseq2" (exact DESeq2 numerics,
///         needs `pip install 'pyscx[pydeseq2]'`). Both emit the same column
///         schema. The default was "pydeseq2" through v0.12; it changed because
///         pydeseq2 is an optional dependency, so the default path did not run
///         on a base install. NB-GLM is DESeq2-*style*, not DESeq2-identical —
///         pin backend="pydeseq2" when you need the exact numerics, and note
///         that `stratify_by` and `aggr_method="mean"` are pydeseq2-only.
///     nbglm_options: Optional dict of NB-GLM tuning knobs (only used when
///         backend="nb_glm"; see pyscx.accel.nb_glm). Ignored for pydeseq2.
///
/// Returns:
///     pandas DataFrame with columns: gene, baseMean, log2FoldChange,
///     lfcSE, stat, pvalue, padj, target, reference
#[pyfunction]
#[pyo3(signature = (adata, groupby=None, test_col=None, reference=None, design=None, aggr_method="sum", min_cells_per_group=10, stratify_by=None, min_cells_per_stratum=50, prefer_format="csr", gene_indices=None, n_cpus=None, backend=None, nbglm_options=None, *, sample_cols=None, sample_key=None))]
#[allow(clippy::too_many_arguments)]
pub fn pseudobulk_dex(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    // `groupby` / `test_col` / `reference` are semantically required, but pyo3
    // cannot express a required positional *after* an optional one — and adding
    // the `sample_cols` / `sample_key` aliases makes `groupby` optional. So the
    // requirement is re-imposed below with messages that carry more than the
    // stock `TypeError: missing required argument` did.
    groupby: Option<&Bound<'_, PyAny>>,
    test_col: Option<&str>,
    reference: Option<&str>,
    design: Option<&str>,
    aggr_method: &str,
    min_cells_per_group: usize,
    stratify_by: Option<Vec<String>>,
    min_cells_per_stratum: usize,
    prefer_format: &str,
    gene_indices: Option<Vec<u32>>,
    n_cpus: Option<usize>,
    // `None` means "the caller did not choose", which is not the same as an
    // explicit `backend="nb_glm"`: the guards below and the transition warning
    // both need to tell those apart so they never blame a kwarg nobody typed.
    backend: Option<&str>,
    nbglm_options: Option<&Bound<'_, PyDict>>,
    sample_cols: Option<&Bound<'_, PyAny>>,
    sample_key: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // Resolve the three semantically-required arguments before any work, then
    // shadow them as non-Option for the rest of the body.
    let groupby = resolve_sample_columns(groupby, sample_cols, sample_key)?;
    let test_col = test_col.ok_or_else(|| {
        PyValueError::new_err(
            "pseudobulk_dex requires test_col: which of the `groupby` columns holds the \
             condition being compared (e.g. test_col=\"disease\" for \
             groupby=[\"disease\", \"donor_id\"])",
        )
    })?;
    let reference = reference.ok_or_else(|| {
        PyValueError::new_err(
            "pseudobulk_dex requires reference: the level of test_col to compare against \
             (e.g. reference=\"normal\")",
        )
    })?;

    if !matches!(prefer_format, "csr" | "csc") {
        return Err(PyValueError::new_err(format!(
            "Invalid prefer_format={prefer_format:?}; expected 'csr' or 'csc'"
        )));
    }
    // A presentation-ordered backed `X` (`preserve_var_order=True`) has no
    // `ShardSource` spelling: the source emits columns in sorted on-disk order
    // while `adata.var` — and so `gene_names` — stays in request order. Now
    // that this op streams the handle's *view*, the two widths match, so the
    // mismatch would be a silent gene/column permutation instead of a shape
    // error. Refuse, as the other streaming accel ops do.
    super::prepare_target(py, adata, "pseudobulk_dex")?;

    let backend_was_explicit = backend.is_some();
    let backend = backend.unwrap_or(DEFAULT_BACKEND);
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
        return Err(PyValueError::new_err(format!(
            "pseudobulk_dex(stratify_by=…) is not supported by {}: NB-GLM \
             treats replicates as rows of a single design, so put the replicate column \
             (batch/donor/well) directly in `groupby`, or use pyscx.accel.pdex_nb_glm \
             (which merges groupby + stratify_by for you).{}",
            backend_phrase(backend_was_explicit),
            pydeseq2_way_back(backend_was_explicit),
        )));
    }
    // Reject a value that is not an aggregation method at all *before* the
    // backend guard below. Otherwise `aggr_method="tpyo"` is answered with a
    // paragraph about the negative-binomial count model, which is true and
    // completely beside the point. Parsing to the enum here also makes an
    // unvalidated string unrepresentable at the dispatch below.
    let parsed_aggr_method = scx_accel::AggregationMethod::parse(aggr_method)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    // §2.5: the NB count likelihood requires replicate-level *summed* counts.
    // Fractional aggregates (mean/median) are not valid inputs to the model, so
    // the count backend accepts only aggr_method="sum".
    if backend == "nb_glm" && aggr_method != "sum" {
        return Err(PyValueError::new_err(format!(
            "pseudobulk_dex requires aggr_method=\"sum\" under {} (got \
             {aggr_method:?}): the negative-binomial count model is defined on summed \
             replicate counts, not fractional {aggr_method} aggregates.{}",
            backend_phrase(backend_was_explicit),
            pydeseq2_way_back(backend_was_explicit),
        )));
    }
    if backend == "nb_glm" && !backend_was_explicit {
        warn_default_backend_changed(py);
    }

    // Record the planned route on adata.uns["scx_accel"]["pseudobulk_dex"].
    // Pseudobulk DE is CPU-only; the only dispatch choice is the gene-axis
    // layout — cpu_csc when prefer_format="csc" (reads the gene-major sidecar),
    // cpu_csr otherwise. Stamped on the top-level adata before the stratified
    // branch (which recurses on discarded sub_adata copies). This is the route
    // the CSC dispatch gate asserts.
    //
    // The NB-GLM backend reports its engine (`cpu_nb_glm`) rather than the
    // layout, because the engine is the bigger fact about the run. Aggregation
    // still honours `prefer_format` on that route, so the layout is carried
    // separately in `csc_available` — without it, flipping the default would
    // have made CSC dispatch invisible on the path most callers now take.
    let mut info = scx_accel::AccelExecutionInfo::new(
        if backend == "nb_glm" {
            scx_accel::AccelRoute::CpuNbGlm
        } else if prefer_format == "csc" {
            scx_accel::AccelRoute::CpuCsc
        } else {
            scx_accel::AccelRoute::CpuCsr
        },
        scx_accel::FallbackReason::None,
    );
    info.csc_available = Some(prefer_format == "csc");
    // Rolled back if anything below raises — see RouteStamp.
    let route = super::route::RouteStamp::write(py, adata, "pseudobulk_dex", &info)?;

    // --- Stratified path ---
    if let Some(ref strat_cols) = stratify_by {
        // Forbidden columns: test_col and all groupby columns.
        let mut forbidden: Vec<&str> = groupby.iter().map(|s| s.as_str()).collect();
        forbidden.push(test_col);
        let (strata, masks) =
            extract_strata(py, adata, strat_cols, min_cells_per_stratum, &forbidden)?;

        let pd = crate::pyimport::import_module(py, "pandas")?;
        let warnings = crate::pyimport::import_module(py, "warnings")?;
        let mut all_frames: Vec<Bound<'_, PyAny>> = Vec::new();

        for (stratum, mask) in strata.iter().zip(masks.iter()) {
            // Subset adata by mask.
            let sub_adata = adata.get_item(mask)?;
            let sub_adata = sub_adata.call_method0("copy")?;

            // Run pseudobulk_dex on the subset (recursive call without stratify).
            // Pass the *resolved* column list, never the raw aliases — the
            // recursion must not re-run alias resolution (and `sample_cols` /
            // `sample_key` are already folded into `groupby` at this point).
            let groupby_list = pyo3::types::PyList::new(py, &groupby)?;
            match pseudobulk_dex(
                py,
                &sub_adata,
                Some(groupby_list.as_any()),
                Some(test_col),
                Some(reference),
                design,
                aggr_method,
                min_cells_per_group,
                None, // no nested stratification
                50,   // unused since stratify_by=None
                prefer_format,
                gene_indices.clone(),
                n_cpus,
                // The *resolved* backend, never the raw argument: the recursion
                // must not re-resolve a `None` (and re-warn once per stratum).
                // Only reachable with "pydeseq2" — nb_glm + stratify_by is
                // rejected above.
                Some(backend),
                nbglm_options,
                None, // sample_cols: already resolved into `groupby`
                None, // sample_key: ditto
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

        route.commit();
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

    // Parsed up front (before the backend guards) by the shared
    // `AggregationMethod::parse`, so an unrecognised value can no longer
    // reach this point as a silently-defaulted string.
    let method = parsed_aggr_method;

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
        // Split any explicit `contrast` out of nbglm_options (an explicit `None` is
        // treated as absent); the rest is parsed as NbGlmOptions. An explicit
        // `contrast` also triggers the design-aware path (it resolves against a
        // named design).
        let (contrast_override, opts_dict) = super::nb_glm::take_contrast_override(nbglm_options)?;
        let nb_opts = super::nb_glm::nbglm_options_from_dict(py, opts_dict.as_ref())?;
        // `pseudobulk_dex` has no `device=` kwarg; the NB-GLM backend runs on CPU
        // here (the GPU path is reached via `pyscx.accel.pdex_nb_glm(device=…)`).
        let df = if design.is_some() || contrast_override.is_some() {
            // §3.11: honor a caller-supplied formula design + optional contrast.
            // Default the formula to `~ `test_col`` (reproduces the intercept +
            // target-indicator model, but as a single shared-dispersion fit).
            // Backtick-quote so a non-identifier test_col (e.g. "cell type") is a
            // valid formulaic token.
            let owned_default;
            let formula = match design {
                Some(s) => s,
                None => {
                    owned_default = format!("~ `{test_col}`");
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
        route.commit();
        return Ok(df.unbind());
    }

    // Build counts DataFrame and metadata DataFrame for pydeseq2.
    let pd = crate::pyimport::import_module(py, "pandas")?;

    // counts_df: rows = pseudobulk samples, columns = genes
    let counts_array = numpy::PyArray1::from_slice(py, &result.counts);
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

    // Import pydeseq2 at runtime. Only reached on an explicit
    // `backend="pydeseq2"` since the default flipped to `"nb_glm"`, which needs
    // no optional dependency at all.
    let pydeseq2 = import_optional(
        py,
        "pydeseq2.dds",
        EXTRA_PYDESEQ2,
        "pyscx.accel.pseudobulk_dex(backend=\"pydeseq2\")",
        "pydeseq2",
    )?;
    let pydeseq2_stats = import_optional(
        py,
        "pydeseq2.ds",
        EXTRA_PYDESEQ2,
        "pyscx.accel.pseudobulk_dex(backend=\"pydeseq2\")",
        "pydeseq2",
    )?;

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

    // Bound the loky/joblib worker *process* count — one interpreter per core on
    // a 192-core node is ~58 GB of RSS and OOM-kills the job even for tiny
    // inputs. That is this cap's whole job; it is applied via
    // `DefaultInference(n_cpus=...)` just below.
    //
    // Deliberately NOT capped here: the per-worker numba and BLAS
    // (OpenMP/MKL/OpenBLAS) thread pools. pydeseq2 already wraps every one of
    // its `Parallel` calls in `parallel_backend(..., inner_max_num_threads=1)`,
    // and joblib's `_prepare_worker_env` applies an explicit
    // `inner_max_num_threads` to the *worker* environment while ignoring the
    // parent's — so setting `NUMBA_NUM_THREADS` & co. here has no effect on the
    // workers at all.
    //
    // It is also actively harmful. numba re-reads `NUMBA_NUM_THREADS` on every
    // fresh compilation and raises if it disagrees with the already-launched
    // pool, so mutating the parent's environment made every later numba compile
    // in the process fail with "Cannot set NUMBA_NUM_THREADS to a different
    // value once the threads have been launched" — one `pseudobulk_dex` call
    // poisoned the rest of the session (79 tests across 21 files, and any
    // scanpy step a user took afterwards). Same reasoning rules out
    // `numba.set_num_threads` on the parent: it silently shrinks the caller's
    // pool for the process lifetime. Leave process-global thread state alone.
    //
    // Regression coverage: `pyscx/tests/test_pseudobulk_thread_env.py`.
    let resolved_n_cpus = resolve_deseq_n_cpus(n_cpus);
    let inference = import_optional(
        py,
        "pydeseq2.default_inference",
        EXTRA_PYDESEQ2,
        "pyscx.accel.pseudobulk_dex(backend=\"pydeseq2\")",
        "pydeseq2",
    )?
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

    route.commit();
    Ok(combined.unbind())
}
