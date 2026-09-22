//! Wilcoxon rank-sum differential expression, stratified DE, and cell-eval bridge.
//!
//! Split into route-family submodules (ORG-10.16-6): `wilcoxon` (the
//! rank-sum kernel driver + entry point), `frames` (DataFrame marshalling +
//! `rank_genes_groups_df`), `pdex` (the pdex_ref parity kernel). This module
//! keeps the shared input plumbing (matrix selection, CSC routing, strata)
//! and the group-label cluster both kernels use, and re-exports every
//! `#[pyfunction]` so lib.rs's `accel::de::<fn>` registration paths are
//! unchanged.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::{ScxLazyTransformedDataset, Transform};

pub(crate) mod frames;
pub(crate) mod pdex;
pub(crate) mod wilcoxon;

pub(crate) use frames::*;
pub(crate) use pdex::*;
pub(crate) use wilcoxon::*;

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
    let warnings = crate::pyimport::import_module(py, "warnings")?;
    let pd = crate::pyimport::import_module(py, "pandas")?;

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
        let mask = numpy::PyArray1::from_vec(py, mask_vec).into_any();
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
///
/// The gene axis of the *selected* matrix is checked here, not by the caller's
/// `prepare_target` prologue: that one reads `adata.X`, so a presentation
/// order living on a named layer went unseen. `gene_names` comes from
/// `adata.var` in request order while the streaming kernels emit the sorted
/// on-disk projection, which is a silent gene permutation rather than an
/// error. Guarding at the point of selection keeps the check on whichever
/// matrix the op will actually read, for all three arms.
fn select_de_matrix<'py>(
    adata: &Bound<'py, PyAny>,
    use_raw: bool,
    layer: Option<&str>,
    op: &str,
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
        crate::accel::reject_presentation_ordered_source(&x, op)?;
        Ok((x, gene_names))
    } else if let Some(name) = layer {
        let x = adata.getattr("layers")?.get_item(name).map_err(|_| {
            PyValueError::new_err(format!("layer '{name}' not found in adata.layers"))
        })?;
        let gene_names = var_names_of(&adata.getattr("var")?)?;
        crate::accel::reject_presentation_ordered_source(&x, op)?;
        Ok((x, gene_names))
    } else {
        let x = adata.getattr("X")?;
        let gene_names = var_names_of(&adata.getattr("var")?)?;
        crate::accel::reject_presentation_ordered_source(&x, op)?;
        Ok((x, gene_names))
    }
}

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
/// Close to the single capability-detection point (`as_column_source`), with
/// one addition of its own: a valid CSC route needs a sidecar present, no
/// active row-deletion vector, and — on a **backed** handle only — no column
/// projection. That third conjunct is enforced here rather than in
/// `as_column_source`, because the full-axis sidecar it hands back cannot
/// serve a projected gene axis while a lazy source remaps one itself. A lazy
/// transform chain is no longer a disqualifier — every `Transform` is
/// CSC-applicable — which **widens what `auto` picks**: a
/// `normalize_total → log1p` chain used to resolve to `cpu_csr` here and now
/// resolves to `cpu_csc`. The two produce bit-identical output (pinned in
/// `pyscx/src/lazy_transform/shard_source_tests.rs`), so the visible difference
/// is the recorded route and the wall time.
///
/// Never errors: a `false` result just routes `auto` to the CSR streamer. A
/// materialized matrix (numpy/scipy, e.g. `use_raw`/`layer`) is not a
/// backed/lazy SCX dataset → `false`.
fn csc_route_available(x: &Bound<'_, PyAny>) -> bool {
    if let Some(handle) = crate::accel::backed_dataset_ref(x) {
        let backed = handle.get();
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

/// pdex's heuristic threshold: log1p-transformed counts rarely exceed ~30.
const LOG1P_MAX_VALUE_HEURISTIC: f64 = 30.0;

/// Name a lazy transform chain for an error message, e.g. `normalize_total → scale`.
fn describe_transform_chain(transforms: &[Transform]) -> String {
    transforms
        .iter()
        .map(|t| match t {
            Transform::NormalizeTotal { .. } => "normalize_total",
            Transform::Log1p => "log1p",
            Transform::RowScale { .. } => "row_scale",
            Transform::Scale { .. } => "scale",
        })
        .collect::<Vec<_>>()
        .join(" → ")
}

/// Auto-detect whether the DE input matrix looks log1p-transformed.
///
/// Mirrors `pdex._utils._detect_is_log1p`, preferring the explicit
/// `uns["log1p"]` annotation and falling back to a max-value heuristic. The
/// answer must not depend on how `X` happens to be stored: an in-memory matrix
/// and the backed handle onto the identical data have to agree, or `pdex_ref`
/// silently picks a different [`GeomMeanMode`] for each and their means differ
/// by orders of magnitude (`mean(expm1(x))` vs `mean(x)`).
///
/// `x` is the matrix `select_de_matrix` chose — not `adata.X`, which is a
/// different matrix under `use_raw=True` / `layer=`.
///
/// Resolution order:
/// 1. `uns["log1p"]` present → yes.
/// 2. Lazy dataset → the transform chain is the ground truth. A `Log1p` in it
///    means yes. Any other non-empty chain rescales the values away from what
///    the catalog recorded, so the probe refuses rather than read stale stats.
/// 3. Backed dataset (or an empty chain) → the catalog's integer `value_max`,
///    which needs no decode. Exact for integer-encoded shards.
/// 4. Float-encoded shards → refuse rather than guess (see below).
/// 5. In-memory scipy / dense → the max-value heuristic, unchanged.
fn detect_is_log1p(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    if super::util::uns_log1p_present(adata) {
        return Ok(true);
    }

    // Lazy: ask the chain, then fall through to its source file.
    let backed_arc = if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
        let transforms = lazy.transforms();
        if transforms.iter().any(|t| matches!(t, Transform::Log1p)) {
            return Ok(true);
        }
        if !transforms.is_empty() {
            // A rescaling chain (normalize_total, row_scale, scale) leaves the
            // data not-log1p, but it also detaches the values from the file's
            // catalog stats — `normalize_total(target_sum=1e4)` over small
            // counts lands nowhere near the counts the shards recorded. The
            // heuristic would then read the wrong numbers, and read them
            // differently from the in-memory arm. Refuse instead.
            return Err(PyValueError::new_err(format!(
                "pdex_ref: cannot auto-detect whether this matrix is log1p-transformed. \
                 adata.uns['log1p'] is absent and adata.X carries a lazy transform chain \
                 ({}) that rescales the stored values, so the SCX catalog's recorded value \
                 range no longer describes them. This chain contains no log1p, so \
                 is_log1p=False is almost certainly what you want; pass it explicitly \
                 (or is_log1p=True if the file itself already held log-space values).",
                describe_transform_chain(transforms)
            )));
        }
        Some(std::sync::Arc::clone(&lazy.backed))
    } else {
        crate::accel::backed_dataset_ref(x).map(|h| std::sync::Arc::clone(&h.get().backed))
    };

    if let Some(backed) = backed_arc {
        // Catalog-only: O(shards), no payload read, no materialization.
        // A superset bound (the file, not a row/column view onto it) is fine
        // here — the heuristic only asks whether values reach counts scale.
        match backed.catalog_int_value_max() {
            // An empty matrix has no values, so neither answer is evidence —
            // and `0 < 30` would silently claim "log1p". The in-memory arm
            // raises here too (numpy cannot reduce an empty array), so falling
            // through to the refusal below is what keeps the two layouts
            // agreeing. `Some(0)` can only mean empty: a positive max returns
            // above.
            Some(0) => {}
            Some(max_val) => return Ok((max_val as f64) < LOG1P_MAX_VALUE_HEURISTIC),
            None => {}
        }
        // No usable bound: float-encoded shards write `value_max = 0` by
        // design, a shard may carry no stats at all, or the matrix is empty.
        // Which one cannot be told apart here, so the message must not assert
        // any of them. Guessing is what made backed and in-memory disagree.
        return Err(PyValueError::new_err(
            "pdex_ref: cannot auto-detect whether this matrix is log1p-transformed. \
             adata.uns['log1p'] is absent and the SCX catalog cannot bound this file's \
             value range — its shards are float-encoded (the format records no range for \
             those), carry no statistics, or hold no values at all. The max-value \
             heuristic used for an in-memory matrix therefore has nothing to read, and \
             guessing would make the backed result disagree with the in-memory one. \
             Pass is_log1p=True for log-space data or is_log1p=False for raw counts. \
             To apply the heuristic yourself: `is_log1p=adata.X.max() < 30` \
             (a streaming max, no materialization).",
        ));
    }

    // In-memory scipy sparse / dense.
    let np = crate::pyimport::import_module(py, "numpy")?;
    let scipy_sparse = crate::pyimport::import_module(py, "scipy.sparse")?;
    let is_sparse = scipy_sparse
        .call_method1("issparse", (x,))?
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
    Ok(max_val.is_finite() && max_val < LOG1P_MAX_VALUE_HEURISTIC)
}

/// Resolve group encoding and reference index, mirroring
/// `run_rank_genes_groups_inner`.
///
/// `requested` (pdex_ref's `groups=`) restricts the *tested targets* before
/// the kernel: the returned universe is `requested ++ [reference]` in request
/// order and every other level's cells carry the unlabelled sentinel. That is
/// exact, not an approximation — each target is compared with the reference
/// alone (Mann-Whitney U, pseudobulk fold change, per-target CPM filter,
/// per-target BH), so cells of an unselected target never entered a selected
/// target's row in the first place. The kernels' `test_groups` become exactly
/// the request, which is what shrinks the CPU work and the GPU footprint.
/// The unlabelled-cell warning is computed against the *full* universe so it
/// still counts only cells with no label at all.
fn resolve_groups_and_reference(
    adata: &Bound<'_, PyAny>,
    groupby: &str,
    reference: &str,
    requested: Option<&[String]>,
) -> PyResult<(Vec<usize>, Vec<String>, usize)> {
    let obs = adata.getattr("obs")?;
    let group_col = single_obs_column(obs.get_item(groupby)?, groupby)?;
    let group_labels: Vec<String> = group_col
        .call_method1("astype", ("str",))?
        .call_method0("tolist")?
        .extract()?;

    let missing = missing_label_mask(adata.py(), &group_col, group_labels.len())?;
    let unique_groups = group_level_universe(&group_col, &group_labels, &missing)?;

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

    // Missing values, and any label outside the universe, get a sentinel that
    // exceeds n_groups, so pdex_ref drops them. `unique_groups.len()` is the
    // out-of-range marker.
    let groups = encode_group_labels(
        &group_labels,
        &group_name_to_idx,
        unique_groups.len(),
        &missing,
    );
    // `resolve_groups_and_reference` serves `pdex_ref` only, which is always
    // pairwise against a named reference.
    warn_unlabelled_cells(adata.py(), &groups, unique_groups.len(), groupby, false);

    let Some(req) = requested else {
        return Ok((groups, unique_groups, ref_idx));
    };
    validate_groups_request(req)?;
    let mut restricted: Vec<String> = Vec::with_capacity(req.len() + 1);
    for name in req {
        if !group_name_to_idx.contains_key(name.as_str()) {
            return Err(PyValueError::new_err(format!(
                "groups: {name:?} is not a level of adata.obs[{groupby:?}]; available: \
                 {unique_groups:?}"
            )));
        }
        if name == reference {
            return Err(PyValueError::new_err(format!(
                "groups: {name:?} is the reference group and cannot also be a tested target"
            )));
        }
        restricted.push(name.clone());
    }
    restricted.push(reference.to_string());
    let restricted_ref = restricted.len() - 1;
    // Full-universe code → restricted code; everything else (unselected levels
    // and the already-unlabelled) → the new sentinel `restricted.len()`.
    let oor = restricted.len();
    let mut remap = vec![oor; unique_groups.len() + 1];
    for (new_idx, name) in restricted.iter().enumerate() {
        remap[group_name_to_idx[name.as_str()]] = new_idx;
    }
    let groups: Vec<usize> = groups.iter().map(|&c| remap[c]).collect();
    Ok((groups, restricted, restricted_ref))
}

/// The shape checks a `groups=` request needs before any label is looked at:
/// non-empty and free of repeats. Membership is checked where the label
/// universe is known (`rank_genes_groups`' inner driver,
/// [`resolve_groups_and_reference`] for `pdex_ref`).
pub(crate) fn validate_groups_request(groups: &[String]) -> PyResult<()> {
    if groups.is_empty() {
        return Err(PyValueError::new_err(
            "groups must name at least one group (got an empty list); omit it to test \
             every group",
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(groups.len());
    for name in groups {
        if !seen.insert(name.as_str()) {
            return Err(PyValueError::new_err(format!(
                "groups lists {name:?} more than once"
            )));
        }
    }
    Ok(())
}

/// Reject an `obs[groupby]` lookup that came back as a frame rather than a
/// column.
///
/// A duplicated column name in `adata.obs` makes `obs[groupby]` a `DataFrame`,
/// and the very next call — `astype("str").tolist()` — then fails with
/// `AttributeError: 'DataFrame' object has no attribute 'tolist'`, which names
/// neither the column nor the duplication. Checked before either conversion so
/// the message can.
fn single_obs_column<'py>(
    group_col: Bound<'py, PyAny>,
    groupby: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let ndim: usize = match group_col.getattr("ndim") {
        Ok(n) => n.extract()?,
        // Not a pandas/numpy object at all; let the existing conversion produce
        // whatever error it would have.
        Err(_) => return Ok(group_col),
    };
    if ndim != 1 {
        return Err(PyValueError::new_err(format!(
            "adata.obs[{groupby:?}] is {ndim}-dimensional, not a single column — \
             the name is duplicated in adata.obs"
        )));
    }
    Ok(group_col)
}

/// Per-cell "this value is missing", asked of pandas rather than inferred from
/// how a value happens to print.
///
/// `astype("str")` renders `np.nan` as `"nan"`, `None` as `"None"` and `pd.NA`
/// (including a nullable `string` dtype's) as `"<NA>"`, so a spelling denylist
/// fails in **both** directions: it misses every NA kind but one, and it steals
/// real group names — a cluster genuinely called `"None"` is not a missing
/// value. `pandas.isna` answers the question that was actually being asked, and
/// it is the same call [`extract_strata`] already makes on its own columns.
///
/// Costs one more `tolist()` over the column, alongside the `astype("str")` one
/// every caller here already pays.
///
/// `n_obs` is the label count the mask has to line up with. Both consumers
/// `zip` the two, and `zip` truncates to the shorter, so a short mask would
/// silently shrink the label universe rather than fail. The multi-column case
/// is caught earlier and more precisely by [`single_obs_column`]; this is the
/// backstop for anything else that returns the wrong length.
fn missing_label_mask(
    py: Python<'_>,
    group_col: &Bound<'_, PyAny>,
    n_obs: usize,
) -> PyResult<Vec<bool>> {
    let pd = crate::pyimport::import_module(py, "pandas")?;
    let mask: Vec<bool> = pd
        .call_method1("isna", (group_col,))?
        .call_method0("tolist")?
        .extract()?;
    if mask.len() != n_obs {
        return Err(PyValueError::new_err(format!(
            "pandas.isna(obs[groupby]) returned {} values for {n_obs} cells; \
             is the column name duplicated in adata.obs?",
            mask.len()
        )));
    }
    Ok(mask)
}

/// The level universe for a `groupby` column: its declared categories when it
/// is categorical, else the sorted distinct labels of the **non-missing** rows.
///
/// The non-categorical arm is what scanpy reaches through `sanitize_anndata` →
/// `astype("category")`, where a missing value is never a category. Deriving it
/// from every row instead would mint a level no cell can be in (`"nan"`,
/// `"None"`, `"<NA>"`) — a phantom `pts` column, a phantom `pdex_ref` target,
/// and, since the singlet guard counts cells per level, a spurious "only
/// contain one sample" error on any object column with a missing value. Note
/// that `""` is **not** among them: `pandas.isna("")` is false, so an
/// empty-string label is an ordinary group.
fn group_level_universe(
    group_col: &Bound<'_, PyAny>,
    group_labels: &[String],
    missing: &[bool],
) -> PyResult<Vec<String>> {
    if let Ok(cat) = group_col.getattr("cat") {
        // `astype("str")` before `tolist`, because a categorical's categories
        // need not be strings — `pd.Categorical([0, 1, 0])` is an ordinary way
        // to spell cluster ids, and extracting `Vec<String>` from `[0, 1]`
        // raised `TypeError: 'int' object is not an instance of 'str'` where
        // scanpy simply stringifies. The column's own labels already come
        // through `astype("str")`, so this is what makes the two line up.
        return cat
            .getattr("categories")?
            .call_method1("astype", ("str",))?
            .call_method0("tolist")?
            .extract();
    }
    let mut unique: Vec<String> = group_labels
        .iter()
        .zip(missing)
        .filter(|(_, &is_na)| !is_na)
        .map(|(label, _)| label.clone())
        .collect();
    unique.sort();
    unique.dedup();
    Ok(unique)
}

/// Map string labels onto group indices, sending every missing value — and
/// anything unrecognised — to the out-of-range sentinel `oor = n_groups`.
///
/// `missing` comes from [`missing_label_mask`], never from the printed form: by
/// the time a label reaches here a `None` is already the string `"None"`, and a
/// category genuinely named `"nan"` is indistinguishable from `np.nan`. The
/// kernels give any index `>= n_groups` no group of its own; whether such a cell
/// still takes part in the comparison depends on the arm, and
/// [`warn_unlabelled_cells`] says which (see `scx_accel::diffexp::groups`).
fn encode_group_labels(
    group_labels: &[String],
    group_name_to_idx: &std::collections::HashMap<&str, usize>,
    oor: usize,
    missing: &[bool],
) -> Vec<usize> {
    group_labels
        .iter()
        .zip(missing)
        .map(|(label, &is_na)| {
            if is_na {
                oor
            } else {
                *group_name_to_idx.get(label.as_str()).unwrap_or(&oor)
            }
        })
        .collect()
}

/// Tell the caller that some cells carry no group label, and what that means
/// for the comparison they asked for.
///
/// An `obs` column with a handful of unannotated cells looks exactly like one
/// without, and the two arms treat them differently, so neither silence would
/// be safe:
///
/// * `one_vs_rest` — the cells rank alongside everyone else and count in every
///   group's "rest" (scanpy's rule, and SCX's since 0.17 / X9). Nothing is
///   dropped; what they do not get is a group, a row of results or a `pts`
///   column of their own.
/// * pairwise (`reference = <name>`, and every `pdex_ref` call) — the
///   comparison is `group ∪ reference`, so a cell in neither takes no part at
///   all.
fn warn_unlabelled_cells(
    py: Python<'_>,
    groups: &[usize],
    n_groups: usize,
    groupby: &str,
    one_vs_rest: bool,
) {
    let n_unlabelled = groups.iter().filter(|&&g| g >= n_groups).count();
    if n_unlabelled == 0 {
        return;
    }
    let short = if one_vs_rest {
        "are in every group's 'rest' but in no group of their own"
    } else {
        "take no part in the test"
    };
    // Also on the Rust log, so a batch pipeline running under
    // `-W ignore` / `warnings.simplefilter("ignore")` still leaves a record.
    log::warn!(
        "DE on obs['{groupby}']: {n_unlabelled} of {} cells have no group label and {short}",
        groups.len()
    );
    let detail = if one_vs_rest {
        "They are in no group of their own — no result row, no `pts` column — but they \
         are in the rank pool and in every group's 'rest', numerator and denominator \
         alike, as scanpy 1.12 does. Label them to give them a group of their own, or \
         drop them to leave them out of 'rest' as well."
    } else {
        "This test compares each group with the named reference group, so a cell in \
         neither takes no part in it at all. Drop or label them to silence this."
    };
    if let Ok(warnings) = crate::pyimport::import_module(py, "warnings") {
        let _ = warnings.call_method1(
            "warn",
            (
                format!(
                    "{n_unlabelled} of {} cells have no group label in obs['{groupby}'] \
                     (a pandas missing value, or a value outside the column's \
                     categories). {detail}",
                    groups.len()
                ),
                py.get_type::<pyo3::exceptions::PyUserWarning>(),
            ),
        );
    }
}
