//! In-place axis subsetting — anndata's job, plus the one part it can't do.
//!
//! # anndata owns the subset
//!
//! `sc.pp.filter_genes` *is* `adata._inplace_subset_var(mask)`. SCX only ever
//! reimplemented that because a backed `X` could not be materialized; once the
//! handles register `as_view` / `_subset` (see [`crate::anndata_hooks`]) they
//! are subsettable, and anndata's own machinery — which builds a fresh
//! `AnnData` and swaps it in — handles `obs`, `var`, `uns`, unused categorical
//! levels, `raw`, and every aligned member, including ones nobody here
//! enumerated. A raise leaves the original untouched.
//!
//! The one substitution is `.copy()`: `AnnData.copy()` materializes an SCX
//! handle *by design* (`adata[mask].copy()` is the documented "subset, then
//! materialize" workflow), so [`rebuild_via_anndata`] goes through
//! `view._mutated_copy(X=view.X, …)` — the same object graph anndata would
//! build, minus the materialization.
//!
//! # The one part it can't do: the lazy bridges
//!
//! `_varm` / `_obsp` / `_varp` — and `_layers` / `_obsm` on some open paths —
//! are [`crate::lazy_mapping`] bridges that decode a section only on first key
//! access. anndata cannot see through that: `AnnData.copy()` reads
//! `self.varm` / `obsp` / `varp`, and `AlignedMappingProperty.__get__` →
//! `AlignedActual.__init__` runs `_validate_value` over *every* entry, which
//! forces a decode. Measured: one `adata[:, mask].copy()` takes `_obsp` from
//! `0 materialized` to `1 materialized`. On a census-scale file carrying a kNN
//! graph that is a full `n_obs × n_obs` read nobody asked for.
//!
//! So the bridges are detached for the duration of anndata's subset and
//! re-attached with the selection *recorded* on [`PendingSubsets`], which
//! applies it at decode time. The alternative — lazy per-key value handles, so
//! a bridge could satisfy `_validate_value` without decoding — would change
//! what `adata.obsp[k]` returns, and scanpy consumers expect scipy.
//!
//! The selection handed to a bridge is read back off `adata.X` *after* the
//! subset, so a bridge's window is composed from the same `kept_to_global` /
//! `col_projection` X ended up with rather than a parallel computation that
//! could drift from it.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, OnceLock};

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

    /// A selection that only plain values can consume.
    ///
    /// Used to seed a lazy bridge with the projection applied at *open* time:
    /// there is no SCX handle to compose into, the value simply has to be cut
    /// down to the visible axis when it decodes.
    pub(crate) fn positional_only(positional: Vec<i64>) -> Self {
        Self {
            positional: Arc::new(positional),
            kept_to_global: None,
            col_projection: None,
        }
    }
}

/// Every raw aligned store, regardless of axis. `_mutated_copy` reads all of
/// them on any subset, so all of them have to be shielded from it.
const ALL_ALIGNED_STORES: [&str; 5] = ["_layers", "_obsm", "_varm", "_obsp", "_varp"];

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
/// `subset_obs`, and anything added later.
pub(crate) fn subset_obs_axis(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    keep: &[bool],
) -> PyResult<()> {
    subset_axis(py, adata, Axis::Obs, keep)
}

/// Subset the **var** axis of `adata` in place against a visible-space mask.
///
/// The var-axis twin of [`subset_obs_axis`], used by `filter_genes` and by HVG
/// selection with `subset=True`. `keep` is indexed in the same order as
/// `adata.var`, which under `preserve_var_order` is presentation order rather
/// than sorted on-disk order; `_subset` composes through
/// `visible_ondisk_in_presentation_order`, so request order survives.
pub(crate) fn subset_var_axis(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    keep: &[bool],
) -> PyResult<()> {
    subset_axis(py, adata, Axis::Var, keep)
}

fn subset_axis(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    axis: Axis,
    keep: &[bool],
) -> PyResult<()> {
    // Subsetting a *view* in place is not expressible — `_init_as_actual`
    // would detach it from the parent it is supposed to track. Rebuild it as
    // an actual AnnData first, keeping `X` lazy. This one call covers
    // `filter_cells` / `filter_genes` / `subset_obs` / `subset_var` and HVG
    // `subset=True`, and must precede the marker clear below: `uns.pop` on a
    // view's transient `DictView` copy is silently lost.
    devirtualize_scx_view(py, adata, "subset")?;

    // Any axis subset invalidates a pending GPU normalize-fusion marker: its
    // `kept_to_global` / `n_obs` / `n_vars` no longer describe `adata`. Doing
    // it here rather than per-op is what covers HVG `subset=True`, which never
    // invalidated it and so could re-run the fused pass at the pre-subset width.
    crate::accel::preprocessing::clear_gpu_normalize_marker(adata)?;

    // A filter that keeps everything must change nothing. Subsetting anyway
    // would install an identity `kept_to_global` / `col_projection`, which no
    // longer closes the CSC capability gate but does make every CSC read pay a
    // row-compaction pass that cannot drop anything, on a file that was never
    // filtered.
    if keep.iter().all(|&k| k) {
        return Ok(());
    }

    let positional = positional_indices(keep);
    let x = adata.getattr("X")?;
    if !is_scx_handle(&x) {
        // A plain in-memory `X` is anndata's own routine, verbatim: it already
        // handles every aligned member plus `raw`, and there is nothing to keep
        // out of core.
        let mask = numpy::PyArray1::from_slice(py, keep);
        let method = match axis {
            Axis::Obs => "_inplace_subset_obs",
            Axis::Var => "_inplace_subset_var",
        };
        adata.call_method1(method, (mask,))?;
        return Ok(());
    }
    if !crate::anndata_hooks::hooks_registered() {
        return Err(crate::anndata_hooks::missing_hooks_error());
    }
    drop(x);

    let bridges = detach_lazy_bridges(py, adata, axis)?;
    let outcome = subset_detached(py, adata, axis, positional, &bridges);

    // Reattach unconditionally, whatever happened. A detached store is an empty
    // dict left behind by `detach_lazy_bridges`, so leaking one turns a
    // recoverable error into a silent `KeyError` from an unrelated call later
    // on — strictly worse than the failure that caused it.
    for (attr, _, bridge) in bridges {
        adata.setattr(attr, bridge)?;
    }
    outcome
}

/// The body of [`subset_axis`] that runs while the bridges are detached.
///
/// Split out so the caller can reattach on every path: an early `?` in here
/// cannot leak a detached store. On failure before the swap `adata` is
/// untouched (anndata builds the replacement first), and on failure after it
/// the bridges come back un-subset, which is loud — a shape mismatch on the
/// next read — rather than silent.
fn subset_detached(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    axis: Axis,
    positional: Vec<i64>,
    bridges: &[DetachedBridge],
) -> PyResult<()> {
    rebuild_via_anndata(py, adata, axis, &positional)?;
    let sel = selection_from_x(adata, axis, positional)?;
    for (_, axes, bridge) in bridges {
        if let Some(axes) = axes {
            apply_bridge_subset(py, bridge.bind(py), *axes, &sel)?;
        }
    }
    Ok(())
}

/// Replace `adata` with its own subset, the way anndata would.
///
/// This is `AnnData._inplace_subset_{obs,var}` with exactly one substitution:
/// where anndata writes `self[idx].copy()` we write
/// `view._mutated_copy(X=view.X, …)`. `AnnData.copy()` calls `.copy()` on the
/// matrix and on every aligned value, and an SCX handle's `.copy()`
/// materializes — deliberately, because `adata[mask].copy()` is the documented
/// "subset, then materialize" workflow. `view.X` is the *un-copied*
/// `_subset(ref.X, idx)`, so the accelerators get the same object graph
/// anndata would have built, minus the materialization.
///
/// Everything else is anndata's: `obs` / `var` sliced and re-categorized,
/// `uns` deep-copied, `raw` subset on the obs axis, and — because it builds a
/// whole new `AnnData` and `_init_as_actual` swaps it in — a raise part-way
/// through leaves the original untouched.
fn rebuild_via_anndata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    axis: Axis,
    positional: &[i64],
) -> PyResult<()> {
    let idx = numpy::PyArray1::from_slice(py, positional);
    let view = match axis {
        Axis::Obs => adata.get_item(&idx)?,
        Axis::Var => adata.get_item((PySlice::full(py), &idx))?,
    };

    reinit_as_actual_keeping_handles(py, adata, &view)
}

/// `view._mutated_copy(X=view.X, …)` → `target._init_as_actual(new)`.
///
/// The substitution over `AnnData.copy()` described on [`rebuild_via_anndata`]:
/// `X` and the aligned mappings go over **un-copied**, so an SCX handle stays
/// lazy instead of being gathered.
///
/// `target` and `view` are distinct when a caller builds the view itself
/// ([`rebuild_via_anndata`]) and the *same object* when de-viewing a view in
/// place ([`devirtualize_scx_view`]) — `_init_as_actual` mutates the target,
/// which is exactly what anndata's own copy-on-write does to a view.
fn reinit_as_actual_keeping_handles(
    py: Python<'_>,
    target: &Bound<'_, PyAny>,
    view: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("X", view.getattr("X")?)?;
    for name in ["layers", "obsm", "varm", "obsp", "varp"] {
        kwargs.set_item(name, subset_mapping_keeping_handles(py, view, name)?)?;
    }
    // `raw` for the same reason as `X`. Left to `_mutated_copy`'s fallback it
    // becomes `self.raw.copy()` → `Raw.copy()` → `.copy()` on the handle, which
    // materializes the whole raw counts matrix. `adata.raw = adata` before
    // `filter_genes` is the canonical scanpy pattern, so that would read an
    // entire second matrix off disk on every gene filter. Handing over the
    // view's own `Raw` keeps `raw._X` un-copied — `_init_as_actual` rebuilds it
    // as `Raw(self, raw._X, raw.var, raw.varm)`.
    let raw = view.getattr("raw")?;
    if !raw.is_none() {
        kwargs.set_item("raw", raw)?;
    }
    let new = view.call_method("_mutated_copy", (), Some(&kwargs))?;
    target.call_method1("_init_as_actual", (new,))?;
    Ok(())
}

/// One-shot-per-op registry for the de-view notice, so a per-gene / per-batch
/// loop over a view doesn't flood the user with duplicates.
static DEVIEW_WARNED: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();

/// Rebuild an anndata **view** over a backed/lazy `X` as an actual `AnnData`,
/// without materializing the matrix. Returns whether a rebuild happened.
///
/// # Why every write-back accelerator calls this first
///
/// An accelerator writes its result onto `adata` — `X`, `obs`, `var`, `obsm`,
/// or just the `uns["scx_accel"]` route stamp. On a view every one of those
/// goes through anndata's copy-on-write, and copy-on-write is
/// `adata.copy()` → `_subset(ref.X, idx).copy()` → **the whole matrix in RAM**.
/// That is the wrong answer for a handle whose entire purpose is to stay on
/// disk (a 500k × 3k gene subset went 1.8 GB → 10.3 GB), and on the one path
/// where copy-on-write does *not* fire — a nested `uns` write, when
/// `uns["scx_accel"]` already exists — the object stays a view and the
/// subsequent `adata.X = …` dies inside anndata with
/// `'ScxBackedSparseDataset' object does not support item assignment`.
///
/// So we reach the same end state anndata's copy-on-write would (an actual
/// `AnnData`, detached from its parent, results landing on it and not on the
/// parent) via [`reinit_as_actual_keeping_handles`], which keeps `X` lazy.
///
/// # When it declines
///
/// - `adata` is not a view → nothing to do.
/// - The parent's `X` is not an SCX handle → a plain scipy/dense `X` has no
///   lazy handle to protect, anndata's own copy-on-write is correct and cheap
///   there, and de-viewing anyway would suppress the
///   `ImplicitModificationWarning` scanpy users expect.
/// - `adata` is not an `AnnData` at all (a `MuData` modality, a duck type) →
///   every probe is `.ok()`-guarded, so this is a no-op rather than an
///   `AttributeError`.
///
/// The parent's `X` is probed via `_adata_ref`, never `adata.X`: on a view of
/// an *in-memory* parent, reading `adata.X` performs a real submatrix copy —
/// pure waste for a case we then decline.
///
/// # Known carve-out
///
/// A view whose index is not expressible as a window (a duplicated or
/// descending selection, e.g. `adata[[2, 2, 7]]` — see
/// [`crate::anndata_hooks`]) already holds a materialized `view.X`. The rebuild
/// then installs scipy, exactly as anndata's copy-on-write would. Nothing to
/// special-case; it is documented in `docs/api/python-accel.md`.
pub(crate) fn devirtualize_scx_view(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    op: &'static str,
) -> PyResult<bool> {
    let is_view = adata
        .getattr("is_view")
        .ok()
        .and_then(|v| v.is_truthy().ok())
        .unwrap_or(false);
    if !is_view {
        return Ok(false);
    }
    let Some(parent_x) = adata
        .getattr("_adata_ref")
        .ok()
        .and_then(|parent| parent.getattr("X").ok())
    else {
        return Ok(false);
    };
    if !is_scx_handle(&parent_x) {
        return Ok(false);
    }
    // `view.X` below goes through the `_subset` hook. Without the hooks
    // registered anndata would materialize it behind our back, which is the
    // one outcome this function exists to prevent.
    if !crate::anndata_hooks::hooks_registered() {
        return Err(crate::anndata_hooks::missing_hooks_error());
    }

    reinit_as_actual_keeping_handles(py, adata, adata)?;

    log::info!(
        target: "pyscx.accel",
        "{op}: rebuilt an AnnData view as actual in place; X stays lazy",
    );
    let first_time = {
        let set = DEVIEW_WARNED.get_or_init(|| Mutex::new(HashSet::new()));
        let mut guard = set.lock().unwrap_or_else(|e| e.into_inner());
        guard.insert(op)
    };
    if first_time {
        // anndata emits `ImplicitModificationWarning` for the same transition,
        // so staying silent would *lose* signal about the detachment. What the
        // message adds is the part anndata's does not say: no data was copied.
        let msg = format!(
            "pyscx.accel.{op}: the AnnData passed in was a view (e.g. adata[:, mask]); it has \
             been rebuilt in place as a regular AnnData so the result can be written to it. \
             No data was copied — X stays lazy — but the object no longer tracks the parent it \
             was sliced from, and the result lands on it, not on the parent. Use \
             pyscx.accel.subset_var / subset_obs to subset a backed AnnData in place and avoid \
             the transition."
        );
        let warned = crate::pyimport::import_module(py, "anndata")
            .and_then(|m| m.getattr("ImplicitModificationWarning"))
            .and_then(|cls| {
                crate::pyimport::import_module(py, "warnings")?
                    .call_method1("warn", (msg.as_str(), cls))
            })
            .is_ok();
        if !warned {
            if let Ok(warnings) = crate::pyimport::import_module(py, "warnings") {
                let _ = warnings.call_method1(
                    "warn",
                    (msg, py.get_type::<pyo3::exceptions::PyUserWarning>()),
                );
            }
        }
    }
    Ok(true)
}

/// `AlignedMapping.copy()`, except SCX handles are passed through un-copied.
///
/// Reading `view.<name>[key]` already yields `as_view(_subset(value, idx))`, so
/// the subset itself is anndata's. The only change is skipping the trailing
/// `.copy()` for a handle, which would gather it. Plain values are still
/// copied — anndata copies them for a reason: the sliced result is a *view*
/// into the parent's buffer, and storing that would pin the un-subset array
/// alive.
fn subset_mapping_keeping_handles<'py>(
    py: Python<'py>,
    view: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let mapping = view.getattr(name)?;
    let out = PyDict::new(py);
    for key in mapping.try_iter()? {
        let key = key?;
        let value = mapping.get_item(&key)?;
        // Exactly the four-class predicate, so use the one that names the
        // four. `is_scx_handle` stays at three for its X-only callers.
        let value = if crate::anndata_hooks::is_handle_class(&value) {
            value
        } else if value.hasattr("copy")? {
            value.call_method0("copy")?
        } else {
            // anndata reaches for `copy.copy` on the one value type without a
            // `.copy()` method (awkward arrays, whose buffers are immutable).
            crate::pyimport::import_module(py, "copy")?.call_method1("copy", (&value,))?
        };
        out.set_item(key, value)?;
    }
    Ok(out)
}

fn is_scx_handle(value: &Bound<'_, PyAny>) -> bool {
    value.cast::<ScxBackedSparseDataset>().is_ok()
        || value.cast::<ScxLazyTransformedDataset>().is_ok()
        || value.cast::<ScxBackedLayerDataset>().is_ok()
}

/// A bridge held out of `adata` for the duration of the subset: the raw-store
/// attribute it came from, how this axis subsets it (`None` = it doesn't), and
/// the bridge itself.
type DetachedBridge = (&'static str, Option<ValueAxes>, Py<PyAny>);

/// Take the lazy bridges out of `adata` so anndata's subset cannot decode them.
///
/// **Every** bridge is detached, not just the ones this axis subsets:
/// `_mutated_copy` reads all five aligned mappings whichever axis moved, so an
/// obs subset would otherwise decode `varm` and `varp` for nothing. Off-axis
/// bridges come back with no selection recorded — the returned `ValueAxes` is
/// `None` for those.
///
/// Plain-dict stores are left in place: anndata subsets those correctly and
/// cheaply, because their SCX-handle values go through the registered
/// `_subset` / `copy()` and stay lazy.
fn detach_lazy_bridges(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    axis: Axis,
) -> PyResult<Vec<DetachedBridge>> {
    let touched = aligned_stores(axis);
    let mut out = Vec::new();
    for attr in ALL_ALIGNED_STORES {
        let Ok(store) = adata.getattr(attr) else {
            continue;
        };
        if !is_lazy_bridge(&store) {
            continue;
        }
        adata.setattr(attr, PyDict::new(py))?;
        let axes = touched
            .iter()
            .find(|(name, _)| *name == attr)
            .map(|(_, axes)| *axes);
        out.push((attr, axes, store.unbind()));
    }
    Ok(out)
}

fn is_lazy_bridge(store: &Bound<'_, PyAny>) -> bool {
    store.cast::<ScxLazyPairwiseMapping>().is_ok()
        || store.cast::<ScxLazyVarmMapping>().is_ok()
        || store.cast::<ScxLazyObsmMapping>().is_ok()
        || store.cast::<ScxLazyLayersMapping>().is_ok()
}

fn apply_bridge_subset(
    py: Python<'_>,
    store: &Bound<'_, PyAny>,
    axes: ValueAxes,
    sel: &AxisSelection,
) -> PyResult<()> {
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
    Err(PyRuntimeError::new_err(
        "internal error: detached a store that is not an SCX lazy mapping",
    ))
}

/// Build the deferred selection from the window `X` actually ended up with.
///
/// A bridge value can itself be an SCX handle (`ScxLazyObsmMapping` yields
/// `ScxBackedObsmDataset` on the backed path), and those absorb the subset as a
/// `kept_to_global` / `col_projection` update rather than a gather — so the
/// selection has to carry the composed mapping, not just positions. Reading it
/// back off the post-subset `X` is what guarantees the two agree.
fn selection_from_x(
    adata: &Bound<'_, PyAny>,
    axis: Axis,
    positional: Vec<i64>,
) -> PyResult<AxisSelection> {
    let x = adata.getattr("X")?;
    Ok(match axis {
        Axis::Obs => {
            let kept: Option<Arc<Vec<u64>>> = if let Ok(b) = x.cast::<ScxBackedSparseDataset>() {
                b.borrow().kept_to_global.clone()
            } else if let Ok(l) = x.cast::<ScxLazyTransformedDataset>() {
                l.borrow().kept_to_global.clone()
            } else {
                None
            };
            match kept {
                Some(kept) => AxisSelection::rows(positional, kept),
                None => AxisSelection::positional_only(positional),
            }
        }
        Axis::Var => {
            let cols: Option<(Arc<Vec<u32>>, bool)> =
                if let Ok(b) = x.cast::<ScxBackedSparseDataset>() {
                    let backed = b.borrow();
                    let preserve = backed.col_presentation_arc().is_some();
                    backed
                        .visible_ondisk_in_presentation_order()
                        .map(|c| (Arc::new(c), preserve))
                } else if let Ok(l) = x.cast::<ScxLazyTransformedDataset>() {
                    l.borrow()
                        .col_projection()
                        .map(|c| (Arc::new(c.to_vec()), false))
                } else {
                    None
                };
            match cols {
                Some((cols, preserve)) => AxisSelection::cols(positional, cols, preserve),
                None => AxisSelection::positional_only(positional),
            }
        }
    })
}

fn positional_indices(keep: &[bool]) -> Vec<i64> {
    keep.iter()
        .enumerate()
        .filter(|(_, &k)| k)
        .map(|(i, _)| i as i64)
        .collect()
}

/// Compose a new deletion vector from *positional* visible-row indices.
///
/// The `_subset` twin of [`compose_kept_to_global`]: anndata hands us an index
/// array rather than a mask, and unlike a mask it may repeat or reorder rows.
/// Both are honoured — `kept_to_global` is a plain visible→global lookup, so
/// duplication and permutation cost nothing and need no special case.
pub(crate) fn compose_rows_positional(
    existing: Option<&[u64]>,
    rows: &[i64],
    n_visible: usize,
) -> PyResult<Vec<u64>> {
    // Bound by the map itself when there is one. Every handle keeps
    // `shape_val` equal to its visible length (`set_kept_to_global` /
    // `set_col_projection*` update both together), so the two agree — but
    // `n_visible` arrives as a separate argument, and if a future construction
    // path ever broke that invariant the indexing below would *panic* rather
    // than raise. Deriving the bound from `existing` makes the out-of-bounds
    // impossible instead of merely unlikely.
    let len = existing.map_or(n_visible, |e| e.len());
    rows.iter()
        .map(|&i| {
            let i = normalize_index(i, len, "row")?;
            Ok(match existing {
                Some(existing) => existing[i],
                None => i as u64,
            })
        })
        .collect()
}

/// Compose a new column projection from *positional* visible-column indices.
///
/// `existing` is the visible→on-disk map **in presentation order**, so the
/// composed result is also in presentation order and must be installed with
/// `set_col_projection_ordered`.
pub(crate) fn compose_cols_positional(
    existing: Option<&[u32]>,
    cols: &[i64],
    n_visible: usize,
) -> PyResult<Vec<u32>> {
    // See [`compose_rows_positional`] — bound by the map so an invariant
    // violation raises instead of panicking.
    let len = existing.map_or(n_visible, |e| e.len());
    cols.iter()
        .map(|&i| {
            let i = normalize_index(i, len, "column")?;
            Ok(match existing {
                Some(existing) => existing[i],
                None => i as u32,
            })
        })
        .collect()
}

/// Whether a composed row map selects every row of an already-unsubset handle.
///
/// Both conditions are load-bearing. `was_unsubset` because composing `0..n`
/// onto an *existing* map reproduces that map, which is a real window and must
/// be kept. And the **length** must match the handle's current row count: a
/// prefix like `0..25` of a 100-row handle also satisfies `composed[i] == i`,
/// but dropping it there would leave `X` at full height while `obs` shrank —
/// which is exactly what `iter_chunks`' `adata[start:end].copy()` does.
pub(crate) fn is_identity_rows(composed: &[u64], n_visible: usize, was_unsubset: bool) -> bool {
    was_unsubset
        && composed.len() == n_visible
        && composed.iter().enumerate().all(|(i, &g)| i as u64 == g)
}

/// Resolve a possibly-negative index against a visible axis length.
fn normalize_index(i: i64, len: usize, what: &str) -> PyResult<usize> {
    let normalized = if i < 0 { i + len as i64 } else { i };
    if normalized < 0 || normalized >= len as i64 {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "{what} index {i} out of range for {len} {what}s"
        )));
    }
    Ok(normalized as usize)
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
                borrowed.set_kept_to_global(require_rows(sel, "a lazy layer")?.as_ref().clone());
            }
            ValueAxes::Cols => {
                // A lazy dataset cannot carry a presentation reorder: lazy
                // arithmetic on a `preserve_var_order` backed X materializes
                // instead, so `preserve_order` is unreachable here.
                let (cols, _) = require_cols(sel, "a lazy layer")?;
                borrowed.set_col_projection(cols.as_ref().clone());
            }
            ValueAxes::Both => unreachable!("guarded by subset_value"),
        }
        return Ok(true);
    }
    if let Ok(obsm) = value.cast::<ScxBackedObsmDataset>() {
        if axes == ValueAxes::Rows && sel.kept_to_global.is_some() {
            let kept = require_rows(sel, "a backed embedding")?;
            obsm.borrow_mut().set_kept_to_global(Arc::clone(kept));
            return Ok(true);
        }
        // `ValueAxes::Rows` means *obs* for `_obsm` but *var* for `_varm`, and a
        // backed dense embedding can only absorb an obs subset. Falling through
        // gathers it rather than silently reporting an absorption that did not
        // happen — the failure mode this module exists to end.
    }
    Ok(false)
}

/// An obs-axis selection must carry the row mapping an SCX handle composes.
///
/// Returning `Ok` without applying anything would report the value as absorbed
/// while leaving it at the old width — a silent no-op of exactly the kind
/// §9.18 catalogues.
fn require_rows<'a>(sel: &'a AxisSelection, what: &str) -> PyResult<&'a Arc<Vec<u64>>> {
    sel.kept_to_global.as_ref().ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "internal error: row subset of {what} carries no kept_to_global mapping"
        ))
    })
}

/// The var-axis counterpart of [`require_rows`].
fn require_cols<'a>(sel: &'a AxisSelection, what: &str) -> PyResult<&'a (Arc<Vec<u32>>, bool)> {
    sel.col_projection.as_ref().ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "internal error: column subset of {what} carries no col_projection"
        ))
    })
}

fn apply_to_backed(
    backed: &mut ScxBackedSparseDataset,
    axes: ValueAxes,
    sel: &AxisSelection,
) -> PyResult<()> {
    match axes {
        ValueAxes::Rows => {
            backed.set_kept_to_global(require_rows(sel, "a backed matrix")?.as_ref().clone());
        }
        ValueAxes::Cols => {
            let (cols, preserve_order) = require_cols(sel, "a backed matrix")?;
            if *preserve_order {
                backed.set_col_projection_ordered(cols.as_ref().clone());
            } else {
                backed.set_col_projection(cols.as_ref().clone());
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
    /// Seed the list with the column projection a `to_anndata(var_names=[...])`
    /// open already applied to `X` / `var` / `layers`.
    ///
    /// The var-axis bridges decode at **physical** width — unlike the obs-axis
    /// ones, which receive the open-time filter directly — so without this the
    /// deferred selections, whose indices are visible-space, would index the
    /// wrong rows. Seeding rather than adding a separate filter keeps one
    /// invariant to hold: *a bridge's decode baseline is the visible width its
    /// selections were recorded against.*
    pub(crate) fn seed_open_projection(&self, axes: ValueAxes, indices: &[u32]) {
        self.push(
            axes,
            AxisSelection::positional_only(indices.iter().map(|&c| c as i64).collect()),
        );
    }

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
///
/// # Ordering is load-bearing
///
/// The push and the cache snapshot must happen **before** any Python call, and
/// in that order. The slicing loop below runs arbitrary Python (`__getitem__` on
/// numpy / scipy / pandas), and numpy releases the GIL for large copies — so
/// another thread can run [`PendingSubsets::apply`] via `fetch` partway through
/// this function. Given that:
///
/// * **Push after the loop** (the obvious order) loses the subset entirely: a
///   concurrent `fetch` of a not-yet-cached key applies a `pending` that does
///   not contain this subset, caches the result, and — because it is now cached
///   — never consults `pending` again. Silently un-subset, permanently.
/// * **Push before the snapshot**, with no Python call between them, closes it.
///   A concurrent `fetch` either has not inserted yet (so it will apply this
///   subset itself on decode) or inserted before the push (so the snapshot sees
///   it and slices it here). It cannot land in between, because `push` and the
///   snapshot are pure Rust — `clone_ref` is an incref, not a Python call — so
///   the GIL cannot be released between them.
///
/// The double-apply that the reverse order would risk is not recoverable
/// either: these selections are positional, so applying one twice is not a
/// no-op.
pub(crate) fn apply_subset_to_lazy_mapping(
    py: Python<'_>,
    state: &std::sync::Mutex<std::collections::HashMap<String, Option<Py<PyAny>>>>,
    pending: &PendingSubsets,
    axes: ValueAxes,
    sel: AxisSelection,
) -> PyResult<()> {
    // Un-fetched keys get it on decode — nothing reads from disk here. Must
    // precede the snapshot; see the ordering note above.
    pending.push(axes, sel.clone());

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
    Ok(())
}
