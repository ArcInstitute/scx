//! Keeping AnnData's aligned members in step with an in-place axis subset.
//!
//! # Why this is not just `adata.layers[k] = adata.layers[k][:, keep]`
//!
//! anndata validates an aligned mapping **when the attribute is read**
//! (`AlignedMappingProperty.__get__` → `construct` → `_validate_value`). The
//! instant `X` / `_var` / `_obs` changes width, `adata.layers` and `adata.obsm`
//! can no longer be *read* — so repair code that goes through the public
//! property can never run, because reading is what raises. No ordering escapes
//! it: writing the members first fails against the old shape, writing `X` first
//! fails against the new one.
//!
//! The raw stores (`adata._layers`, `_obsm`, `_varm`, `_obsp`, `_varp`) are
//! plain dicts — or, for a backed AnnData, one of the [`crate::lazy_mapping`]
//! bridges. Neither validates, so [`subset_axis`] works there and ordering
//! stops mattering.
//!
//! Everything funnels through [`subset_axis`] deliberately: Phase 4.0b replaces
//! that one body with anndata's own `_inplace_subset_var` / `_inplace_subset_obs`
//! once the SCX handles register `as_view` / `_subset`, and no call site has to
//! change.

use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice};

use crate::backed::{ScxBackedLayerDataset, ScxBackedObsmDataset, ScxBackedSparseDataset};
use crate::lazy_mapping::{
    ScxLazyLayersMapping, ScxLazyObsmMapping, ScxLazyPairwiseMapping, ScxLazyVarmMapping,
};
use crate::lazy_transform::ScxLazyTransformedDataset;

/// The axis of the *parent* being subset.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Axis {
    Obs,
    Var,
}

/// Which axes of an aligned *value* a parent-axis subset touches.
///
/// `obsm` is obs-aligned on its rows; `varm` is var-aligned on its rows;
/// `layers` is obs-aligned on rows and var-aligned on columns; `obsp` / `varp`
/// are square, so their axis subsets both dimensions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ValueAxes {
    Rows,
    Cols,
    Both,
}

/// One axis subset, in the two currencies aligned values speak.
///
/// A plain numpy / scipy / pandas value is at the *visible* width, so it wants
/// positional indices. An SCX handle is a lazy window onto the whole file, so
/// it wants the composed absolute mapping — setting that costs no I/O, where
/// positional slicing would force a materialization.
#[derive(Clone)]
pub(crate) struct AxisSelection {
    /// Positional indices into the value's pre-subset axis.
    pub positional: Arc<Vec<i64>>,
    /// Visible row → global file row, for SCX handles on the obs axis.
    pub kept_to_global: Option<Arc<Vec<u64>>>,
    /// Visible column → on-disk column (and whether that order is a
    /// `preserve_var_order` presentation permutation), for the var axis.
    pub col_projection: Option<(Arc<Vec<u32>>, bool)>,
}

impl AxisSelection {
    pub(crate) fn rows(positional: Vec<i64>, kept_to_global: Arc<Vec<u64>>) -> Self {
        Self {
            positional: Arc::new(positional),
            kept_to_global: Some(kept_to_global),
            col_projection: None,
        }
    }

    pub(crate) fn cols(positional: Vec<i64>, cols: Arc<Vec<u32>>, preserve_order: bool) -> Self {
        Self {
            positional: Arc::new(positional),
            kept_to_global: None,
            col_projection: Some((cols, preserve_order)),
        }
    }
}

/// The five aligned stores, and how each parent axis reaches them.
///
/// Returned as `(raw attribute name, which axes of the value to subset)`. The
/// two omitted-by-construction pairings (`obsm` under a var subset, `varm`
/// under an obs subset) simply don't appear.
fn aligned_stores(axis: Axis) -> &'static [(&'static str, ValueAxes)] {
    match axis {
        Axis::Obs => &[
            ("_layers", ValueAxes::Rows),
            ("_obsm", ValueAxes::Rows),
            ("_obsp", ValueAxes::Both),
        ],
        Axis::Var => &[
            ("_layers", ValueAxes::Cols),
            ("_varm", ValueAxes::Rows),
            ("_varp", ValueAxes::Both),
        ],
    }
}

/// Subset the **obs** axis of `adata` in place against a visible-space mask.
///
/// The single entry point for every obs-axis mutation — `filter_cells`,
/// `subset_obs`, and anything added later. Composes the new deletion vector
/// from `X`'s current state, re-points `X` and every obs-aligned member, and
/// slices `_obs`. Nothing is materialized: SCX handles absorb the subset as a
/// `kept_to_global` update and lazy mapping entries defer it to decode time.
///
/// A non-SCX `X` hands the whole job to anndata's own `_inplace_subset_obs` —
/// the implementation scanpy's `filter_cells` calls. Phase 4.0b makes that the
/// only branch.
pub(crate) fn subset_obs_axis(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    keep: &[bool],
) -> PyResult<()> {
    let x = adata.getattr("X")?;
    let positional = positional_indices(keep);

    let existing: Option<Arc<Vec<u64>>> = if let Ok(b) = x.cast::<ScxBackedSparseDataset>() {
        b.borrow().kept_to_global.clone()
    } else if let Ok(l) = x.cast::<ScxLazyTransformedDataset>() {
        l.borrow().kept_to_global.clone()
    } else {
        return inplace_subset_via_anndata(py, adata, "_inplace_subset_obs", keep);
    };

    let new_kept = Arc::new(compose_kept_to_global(
        keep,
        existing.as_ref().map(|v| v.as_slice()),
    ));
    let sel = AxisSelection::rows(positional, new_kept);

    // X first — the raw stores don't validate, so order is free, but keeping
    // X's state authoritative before the members read it is easier to reason
    // about.
    if !subset_scx_handle(&x, ValueAxes::Rows, &sel)? {
        return Err(PyRuntimeError::new_err(
            "internal error: X changed type during an obs subset",
        ));
    }

    // `_obs` bypasses anndata's shape validation (X's row count already moved).
    let obs = adata.getattr("obs")?;
    let sliced = obs.getattr("iloc")?.get_item(idx_array(py, &sel)?)?;
    adata.setattr("_obs", sliced)?;

    subset_axis(py, adata, Axis::Obs, &sel)
}

/// Subset the **var** axis of `adata` in place against a visible-space mask.
///
/// The var-axis twin of [`subset_obs_axis`], used by `filter_genes` and by HVG
/// selection with `subset=True`. `keep` is indexed in the same order as
/// `adata.var`, which under `preserve_var_order` is presentation order, not
/// sorted on-disk order — `visible_ondisk_in_presentation_order` is what keeps
/// the composed projection in step with the var rows.
///
/// **Out of scope, as of 4.0a:** `adata.raw` carries its own var axis and is
/// left untouched, matching anndata (a var subset does not reach `raw`).
pub(crate) fn subset_var_axis(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    keep: &[bool],
) -> PyResult<()> {
    let x = adata.getattr("X")?;
    let positional = positional_indices(keep);

    let (new_cols, preserve_order) = if let Ok(b) = x.cast::<ScxBackedSparseDataset>() {
        let backed = b.borrow();
        let preserve = backed.col_presentation_arc().is_some();
        (
            compose_col_projection(
                keep,
                backed.visible_ondisk_in_presentation_order().as_deref(),
            ),
            preserve,
        )
    } else if let Ok(l) = x.cast::<ScxLazyTransformedDataset>() {
        (
            compose_col_projection(keep, l.borrow().col_projection()),
            false,
        )
    } else {
        return inplace_subset_via_anndata(py, adata, "_inplace_subset_var", keep);
    };

    let sel = AxisSelection::cols(positional, Arc::new(new_cols), preserve_order);

    if !subset_scx_handle(&x, ValueAxes::Cols, &sel)? {
        return Err(PyRuntimeError::new_err(
            "internal error: X changed type during a var subset",
        ));
    }

    // `_var` bypasses anndata's shape validation (X's width already moved).
    let var = adata.getattr("var")?;
    let sliced = var.getattr("iloc")?.get_item(idx_array(py, &sel)?)?;
    adata.setattr("_var", sliced)?;

    subset_axis(py, adata, Axis::Var, &sel)
}

/// Hand an in-memory AnnData to anndata's own in-place subsetter.
///
/// Not a fallback so much as the correct answer: `_inplace_subset_{obs,var}` is
/// what `sc.pp.filter_{cells,genes}` calls, and it already handles every
/// aligned member plus `raw`. SCX only reimplements it because a backed `X`
/// cannot be materialized.
fn inplace_subset_via_anndata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    method: &str,
    keep: &[bool],
) -> PyResult<()> {
    let mask = numpy::PyArray1::from_slice(py, keep);
    adata.call_method1(method, (mask,))?;
    Ok(())
}

fn positional_indices(keep: &[bool]) -> Vec<i64> {
    keep.iter()
        .enumerate()
        .filter(|(_, &k)| k)
        .map(|(i, _)| i as i64)
        .collect()
}

fn idx_array<'py>(
    py: Python<'py>,
    sel: &AxisSelection,
) -> PyResult<Bound<'py, numpy::PyArray1<i64>>> {
    Ok(numpy::PyArray1::from_slice(py, &sel.positional))
}

/// Compose a new deletion vector from a visible-space mask and the existing one.
pub(crate) fn compose_kept_to_global(keep: &[bool], existing: Option<&[u64]>) -> Vec<u64> {
    match existing {
        Some(existing) => keep
            .iter()
            .enumerate()
            .filter(|(_, &k)| k)
            .map(|(i, _)| existing[i])
            .collect(),
        None => keep
            .iter()
            .enumerate()
            .filter(|(_, &k)| k)
            .map(|(i, _)| i as u64)
            .collect(),
    }
}

/// Compose a new column projection from a visible-space mask and the existing
/// visible→on-disk map.
fn compose_col_projection(keep: &[bool], existing: Option<&[u32]>) -> Vec<u32> {
    match existing {
        Some(existing) => keep
            .iter()
            .enumerate()
            .filter(|(_, &k)| k)
            .map(|(i, _)| existing[i])
            .collect(),
        None => keep
            .iter()
            .enumerate()
            .filter(|(_, &k)| k)
            .map(|(i, _)| i as u32)
            .collect(),
    }
}

/// Bring every aligned member into line with a subset already applied to `X`.
///
/// `adata` must already carry the new `X` state and the sliced `_obs` / `_var`;
/// this handles the members. Reads and writes only the raw stores, so it is
/// immune to anndata's read-time validation and order-independent.
fn subset_axis(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    axis: Axis,
    sel: &AxisSelection,
) -> PyResult<()> {
    for (attr, axes) in aligned_stores(axis) {
        let Ok(store) = adata.getattr(*attr) else {
            // AnnData always defines all five; a caller passing something else
            // (a Raw, a mock) simply has no members to fix up.
            continue;
        };
        if store.is_none() {
            continue;
        }
        subset_store(py, &store, *axes, sel)?;
    }
    Ok(())
}

/// Apply a selection to one raw store, whatever kind of mapping it is.
fn subset_store(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    axes: ValueAxes,
    sel: &AxisSelection,
) -> PyResult<()> {
    // The lazy bridges own their own deferral: cached values are sliced now,
    // un-fetched keys record the selection and apply it on decode. Going
    // through the generic dict path instead would force every section on disk
    // to materialize just to be subset.
    macro_rules! try_lazy {
        ($ty:ty) => {
            if let Ok(m) = store.cast::<$ty>() {
                return m.borrow().apply_axis_subset(py, axes, sel.clone());
            }
        };
    }
    try_lazy!(ScxLazyPairwiseMapping);
    try_lazy!(ScxLazyVarmMapping);
    try_lazy!(ScxLazyObsmMapping);
    try_lazy!(ScxLazyLayersMapping);

    let dict = store.cast::<PyDict>().map_err(|_| {
        PyRuntimeError::new_err(format!(
            "cannot subset aligned store of type {}: expected a dict or an SCX \
             lazy mapping",
            store
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_default()
        ))
    })?;
    let keys: Vec<Py<PyAny>> = dict.keys().iter().map(|k| k.unbind()).collect();
    for key in keys {
        let key = key.bind(py);
        let Some(value) = dict.get_item(key)? else {
            continue;
        };
        if let Some(replacement) = subset_value(py, &value, axes, sel)? {
            dict.set_item(key, replacement)?;
        }
    }
    Ok(())
}

/// Subset a single aligned value.
///
/// Returns `Some(new_value)` when the value had to be replaced, `None` when an
/// SCX handle absorbed the subset in place (no copy, no decode).
pub(crate) fn subset_value(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axes: ValueAxes,
    sel: &AxisSelection,
) -> PyResult<Option<Py<PyAny>>> {
    // A square (`Both`) member is never an SCX handle — obsp/varp decode to
    // scipy — so don't try to compose a one-axis projection onto one.
    if axes != ValueAxes::Both && subset_scx_handle(value, axes, sel)? {
        return Ok(None);
    }
    Ok(Some(slice_positional(py, value, axes, &sel.positional)?))
}

/// Compose the subset into an SCX handle's own projection state. Returns
/// `false` when `value` is not one, so the caller falls through to slicing.
fn subset_scx_handle(
    value: &Bound<'_, PyAny>,
    axes: ValueAxes,
    sel: &AxisSelection,
) -> PyResult<bool> {
    if let Ok(layer) = value.cast::<ScxBackedLayerDataset>() {
        let mut borrowed = layer.borrow_mut();
        apply_to_backed(&mut borrowed.inner, axes, sel)?;
        return Ok(true);
    }
    if let Ok(backed) = value.cast::<ScxBackedSparseDataset>() {
        apply_to_backed(&mut backed.borrow_mut(), axes, sel)?;
        return Ok(true);
    }
    if let Ok(lazy) = value.cast::<ScxLazyTransformedDataset>() {
        let mut borrowed = lazy.borrow_mut();
        match axes {
            ValueAxes::Rows => {
                if let Some(kept) = &sel.kept_to_global {
                    borrowed.set_kept_to_global(kept.as_ref().clone());
                }
            }
            ValueAxes::Cols => {
                // A lazy dataset cannot carry a presentation reorder: lazy
                // arithmetic on a `preserve_var_order` backed X materializes
                // instead, so `preserve_order` is unreachable here.
                if let Some((cols, _)) = &sel.col_projection {
                    borrowed.set_col_projection(cols.as_ref().clone());
                }
            }
            ValueAxes::Both => unreachable!("guarded by subset_value"),
        }
        return Ok(true);
    }
    if let Ok(obsm) = value.cast::<ScxBackedObsmDataset>() {
        if axes == ValueAxes::Rows {
            if let Some(kept) = &sel.kept_to_global {
                obsm.borrow_mut().set_kept_to_global(Arc::clone(kept));
            }
            return Ok(true);
        }
        // A column subset of a backed dense embedding has no lazy
        // representation; fall through and gather it.
    }
    Ok(false)
}

fn apply_to_backed(
    backed: &mut ScxBackedSparseDataset,
    axes: ValueAxes,
    sel: &AxisSelection,
) -> PyResult<()> {
    match axes {
        ValueAxes::Rows => {
            if let Some(kept) = &sel.kept_to_global {
                backed.set_kept_to_global(kept.as_ref().clone());
            }
        }
        ValueAxes::Cols => {
            if let Some((cols, preserve_order)) = &sel.col_projection {
                if *preserve_order {
                    backed.set_col_projection_ordered(cols.as_ref().clone());
                } else {
                    backed.set_col_projection(cols.as_ref().clone());
                }
            }
        }
        ValueAxes::Both => unreachable!("guarded by subset_value"),
    }
    Ok(())
}

/// Positionally index a plain value (numpy / scipy / pandas / anything that
/// supports fancy indexing) along the requested axes.
///
/// This is what makes non-SCX members *correct* rather than merely
/// unreachable: skipping them, as the pre-4.0a code did, left an in-memory
/// layer at the old width so the AnnData broke later instead of now.
fn slice_positional(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    axes: ValueAxes,
    positional: &[i64],
) -> PyResult<Py<PyAny>> {
    let idx = numpy::PyArray1::from_slice(py, positional);
    let full = PySlice::full(py);

    // pandas indexes labels through `[]`; `.iloc` is the positional door.
    let sliced = if value.hasattr("iloc")? {
        let iloc = value.getattr("iloc")?;
        match axes {
            ValueAxes::Rows => iloc.get_item(&idx)?,
            ValueAxes::Cols => iloc.get_item((&full, &idx))?,
            ValueAxes::Both => iloc
                .get_item(&idx)?
                .getattr("iloc")?
                .get_item((&full, &idx))?,
        }
    } else {
        match axes {
            ValueAxes::Rows => value.get_item(&idx)?,
            ValueAxes::Cols => value.get_item((&full, &idx))?,
            ValueAxes::Both => value.get_item(&idx)?.get_item((&full, &idx))?,
        }
    };
    Ok(sliced.unbind())
}

/// Deferred subsets recorded on a lazy mapping.
///
/// Applied in insertion order at `fetch` time. Sequencing them rather than
/// composing them is deliberate: each selection's positional indices are
/// relative to the width left by the previous one, so replaying the sequence is
/// correct by construction and there is no composition arithmetic to get wrong.
#[derive(Default)]
pub(crate) struct PendingSubsets(std::sync::Mutex<Vec<(ValueAxes, AxisSelection)>>);

impl PendingSubsets {
    fn push(&self, axes: ValueAxes, sel: AxisSelection) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((axes, sel));
    }

    fn snapshot(&self) -> Vec<(ValueAxes, AxisSelection)> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Apply every recorded subset to a freshly decoded value.
    pub(crate) fn apply(&self, py: Python<'_>, obj: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let pending = self.snapshot();
        if pending.is_empty() {
            return Ok(obj);
        }
        let mut current = obj;
        for (axes, sel) in &pending {
            if let Some(next) = subset_value(py, current.bind(py), *axes, sel)? {
                current = next;
            }
        }
        Ok(current)
    }
}

/// Record a subset on a lazy mapping: slice what is already cached, defer the rest.
pub(crate) fn apply_subset_to_lazy_mapping(
    py: Python<'_>,
    state: &std::sync::Mutex<std::collections::HashMap<String, Option<Py<PyAny>>>>,
    pending: &PendingSubsets,
    axes: ValueAxes,
    sel: AxisSelection,
) -> PyResult<()> {
    // Cached entries have already been decoded (and already had any earlier
    // pending subsets applied), so they take this one now.
    let cached: Vec<(String, Py<PyAny>)> = {
        let state = state.lock().unwrap_or_else(|e| e.into_inner());
        state
            .iter()
            .filter_map(|(k, v)| v.as_ref().map(|obj| (k.clone(), obj.clone_ref(py))))
            .collect()
    };
    let mut updates: Vec<(String, Py<PyAny>)> = Vec::new();
    for (key, obj) in cached {
        if let Some(next) = subset_value(py, obj.bind(py), axes, &sel)? {
            updates.push((key, next));
        }
    }
    {
        let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
        for (key, obj) in updates {
            state.insert(key, Some(obj));
        }
    }
    // Un-fetched keys get it on decode — nothing reads from disk here.
    pending.push(axes, sel);
    Ok(())
}
