//! Teaching anndata that an SCX handle is a subsettable lazy array.
//!
//! anndata reaches a matrix through three `singledispatch` seams:
//!
//! | hook | module | SCX answer |
//! |---|---|---|
//! | `as_view` | `anndata._core.views` | identity — a handle already *is* a lazy view |
//! | `_subset` | `anndata._core.index` | a lazy clone with the window composed |
//! | `to_memory` | `anndata._core.file_backing` | `handle.to_memory()` |
//!
//! Before Phase 4.0b none were registered, so `adata[:, mask]` raised
//! `NotImplementedError: No view type has been registered` and
//! `AnnData._inplace_subset_var` with it. With them, `adata[:, mask]` builds a
//! lazy view, `adata[:, mask].copy()` materializes (as documented), and
//! `adata[mask].to_memory()` produces a fully in-memory AnnData. It is also
//! what lets [`crate::axis_align`] hand the axis bookkeeping to anndata instead
//! of reimplementing it.
//!
//! # `_subset` is deliberately not `__getitem__`
//!
//! Registering it as its own function leaves the public indexing semantics
//! exactly as they were — `X[mask]` still materializes to scipy, `X[:, cols]`
//! still returns a lazy projected clone. Only anndata's internal path is
//! always-lazy, and the in-place accelerators reach it through
//! `_mutated_copy(X=view.X, …)`, which never calls `.copy()` on the matrix.

use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::exceptions::{PyIndexError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};

use crate::backed::{ScxBackedLayerDataset, ScxBackedObsmDataset, ScxBackedSparseDataset};
use crate::lazy_transform::ScxLazyTransformedDataset;

/// Whether the `as_view` / `_subset` / `to_memory` registrations all landed.
///
/// Registration is best-effort at import (anndata is an optional dependency),
/// but [`crate::axis_align`] now *depends* on it, so it checks this flag and
/// raises something actionable rather than letting anndata's
/// `NotImplementedError` surface from inside a user's `filter_genes`.
static HOOKS_REGISTERED: AtomicBool = AtomicBool::new(false);

pub(crate) fn hooks_registered() -> bool {
    HOOKS_REGISTERED.load(Ordering::Relaxed)
}

/// The error a caller sees when the axis-subset path needs the hooks and they
/// are not there.
pub(crate) fn missing_hooks_error() -> PyErr {
    PyRuntimeError::new_err(
        "pyscx could not register its subset hooks with anndata, so axis \
         subsetting on a backed or lazy X is unavailable. This usually means \
         an unsupported anndata version — see docs/compatibility-matrix.md for \
         the tested range. Materialize first with `adata.X = adata.X.to_memory()` \
         to work around it.",
    )
}

// ---------------------------------------------------------------------------
// The hook implementations
// ---------------------------------------------------------------------------

/// `as_view(obj, view_args)` → `obj`.
///
/// anndata wraps in-memory values so that writing through a view raises
/// `ImplicitModificationWarning` and re-materializes the parent. An SCX handle
/// is already a read-through window onto a file; there is nothing to guard and
/// nothing to copy-on-write.
#[pyfunction]
#[pyo3(signature = (obj, view_args=None))]
fn scx_as_view(obj: Py<PyAny>, view_args: Option<Py<PyAny>>) -> Py<PyAny> {
    let _ = view_args;
    obj
}

/// `to_memory(x, *, copy=False)` → the materialized value.
///
/// Registered so `AnnData.to_memory()` reaches inside an SCX-backed object
/// instead of passing the handles through untouched — its `object` fallback
/// returns unrecognized values as-is, so `adata.to_memory().X` used to still be
/// a handle. `copy` is irrelevant here: `to_memory` always allocates.
/// `copy` is positional-or-keyword deliberately. anndata 0.12 calls this
/// `to_memory(attr, copy=copy)`, but 0.11 — inside the declared
/// `anndata>=0.11,<0.13` range — calls it **positionally**, and a keyword-only
/// parameter would make `adata.to_memory()` raise `TypeError` there. The
/// permissive signature satisfies both.
#[pyfunction]
#[pyo3(signature = (x, copy=false))]
fn scx_to_memory<'py>(x: &Bound<'py, PyAny>, copy: bool) -> PyResult<Bound<'py, PyAny>> {
    let _ = copy;
    x.call_method0("to_memory")
}

/// `_subset(value, subset_idx)` → a lazy clone with the window composed.
///
/// `subset_idx` is a 1- or 2-tuple of normalized indices
/// (`slice | ndarray[int] | ndarray[bool]`). Element 0 always addresses the
/// value's own axis 0 — for a `varm` value that is the var axis, for a layer
/// it is obs.
#[pyfunction]
fn scx_subset<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    subset_idx: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let (n_rows, n_cols) = shape_of(value)?;
    let (row_obj, col_obj) = split_subset_idx(subset_idx)?;
    let rows = match row_obj {
        Some(ref o) => positional_from_index(py, o, n_rows)?,
        None => None,
    };
    let cols = match col_obj {
        Some(ref o) => positional_from_index(py, o, n_cols)?,
        None => None,
    };
    let rows = rows.as_deref();
    let cols = cols.as_deref();

    // Not every selection can be expressed as a lazy window. When it cannot,
    // materialize — which is exactly what anndata's `object` fallback did
    // before these hooks existed, so this is the shipped behaviour rather than
    // a new limitation.
    if !expressible_as_window(value, rows, cols) {
        return value.get_item(subset_idx);
    }

    if let Ok(layer) = value.cast::<ScxBackedLayerDataset>() {
        let out = layer.borrow().subset_clone(rows, cols)?;
        return Ok(out.into_pyobject(py)?.into_any());
    }
    if let Ok(backed) = value.cast::<ScxBackedSparseDataset>() {
        let out = backed.borrow().subset_clone(rows, cols)?;
        return Ok(out.into_pyobject(py)?.into_any());
    }
    if let Ok(lazy) = value.cast::<ScxLazyTransformedDataset>() {
        let out = lazy.borrow().subset_clone(rows, cols)?;
        return Ok(out.into_pyobject(py)?.into_any());
    }
    if let Ok(obsm) = value.cast::<ScxBackedObsmDataset>() {
        // A dense embedding is aligned on one axis, so anndata only ever hands
        // it a row index. A column index would have to gather, so fall back to
        // the materializing `__getitem__` rather than pretend otherwise.
        if cols.is_none() {
            let out = match rows {
                Some(rows) => obsm.borrow().subset_clone(rows)?,
                None => obsm.borrow().clone_handle(),
            };
            return Ok(out.into_pyobject(py)?.into_any());
        }
        return value.get_item(subset_idx);
    }

    Err(PyRuntimeError::new_err(format!(
        "internal error: pyscx registered `_subset` for a type it cannot \
         subset: {}",
        value
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_default()
    )))
}

// ---------------------------------------------------------------------------
// Index handling
// ---------------------------------------------------------------------------

/// Whether this selection can be carried as a lazy projection update rather
/// than materialized.
///
/// # Rows must be strictly ascending
///
/// `kept_to_global` (visible row → global file row) is a **construction
/// invariant: strictly ascending**. Every masked column kernel
/// `partition_point`s it instead of scanning — `col_aggregate_masked` /
/// `col_sums_and_nnz_masked` in `scx-format-io`, `projected_agg`, the lazy
/// shard source, `shard_boundaries`. Handed a reordered or duplicated map they
/// compute a garbage `lo..hi` and index out of bounds, which surfaces as a
/// `PanicException` with a wrapped `u64` from `X.sum(axis=0)` / `getnnz` / HVG
/// / QC.
///
/// Critically this is invisible on a single-shard file, where `partition_point`
/// returns the whole slice and order-independent aggregates come out right — so
/// small fixtures pass and the failure only appears at real scale. Hence the
/// guard here rather than a test-shaped assumption.
///
/// # Columns must be unique
///
/// The column axis *can* express a reorder (`set_col_projection_ordered` keeps
/// a separate `col_presentation` permutation), but it dedups — so a repeated
/// column would silently narrow `X` while anndata expects the repeat, and the
/// shapes would diverge. A lazily-transformed `X` additionally cannot express a
/// reorder at all: it stores its projection sorted.
fn expressible_as_window(
    value: &Bound<'_, PyAny>,
    rows: Option<&[i64]>,
    cols: Option<&[i64]>,
) -> bool {
    if let Some(rows) = rows {
        if !is_strictly_ascending(rows) {
            return false;
        }
    }
    if let Some(cols) = cols {
        if !is_strictly_ascending(cols) {
            // Unique-but-reordered is fine for a backed handle, which carries a
            // presentation permutation; a lazy one has nowhere to put it.
            let unique = {
                let mut sorted = cols.to_vec();
                sorted.sort_unstable();
                sorted.windows(2).all(|w| w[0] < w[1])
            };
            if !unique || value.cast::<ScxLazyTransformedDataset>().is_ok() {
                return false;
            }
        }
    }
    true
}

fn is_strictly_ascending(v: &[i64]) -> bool {
    v.windows(2).all(|w| w[0] < w[1])
}

fn shape_of(value: &Bound<'_, PyAny>) -> PyResult<(usize, usize)> {
    value.getattr("shape")?.extract()
}

/// The per-axis index objects a `subset_idx` tuple carries, `None` where the
/// tuple did not address that axis at all.
type AxisIndices<'py> = (Option<Bound<'py, PyAny>>, Option<Bound<'py, PyAny>>);

/// Split anndata's `subset_idx` tuple into (axis-0 index, axis-1 index).
fn split_subset_idx<'py>(subset_idx: &Bound<'py, PyAny>) -> PyResult<AxisIndices<'py>> {
    let Ok(tuple) = subset_idx.cast::<PyTuple>() else {
        // Not a tuple: anndata's own `_subset` treats a bare index as axis 0.
        return Ok((Some(subset_idx.clone()), None));
    };
    match tuple.len() {
        0 => Ok((None, None)),
        1 => Ok((Some(tuple.get_item(0)?), None)),
        2 => Ok((Some(tuple.get_item(0)?), Some(tuple.get_item(1)?))),
        n => Err(PyIndexError::new_err(format!(
            "too many indices for 2-dimensional array: got {n}"
        ))),
    }
}

/// Normalize one index into positional indices, or `None` for "the whole axis".
///
/// Returning `None` rather than `Some(0..len)` matters: it is what lets a
/// one-axis subset leave the other axis's projection state untouched, instead
/// of installing an identity `col_projection` that would suppress the
/// no-projection fast paths.
fn positional_from_index(
    py: Python<'_>,
    idx: &Bound<'_, PyAny>,
    len: usize,
) -> PyResult<Option<Vec<i64>>> {
    if idx.is_none() {
        return Ok(None);
    }
    if let Ok(slice) = idx.cast::<PySlice>() {
        let ind = slice.indices(len as isize)?;
        if ind.start == 0 && ind.stop == len as isize && ind.step == 1 {
            return Ok(None);
        }
        let mut out = Vec::new();
        let mut i = ind.start;
        while (ind.step > 0 && i < ind.stop) || (ind.step < 0 && i > ind.stop) {
            out.push(i as i64);
            i += ind.step;
        }
        return Ok(Some(out));
    }
    if let Ok(i) = idx.extract::<i64>() {
        return Ok(Some(vec![i]));
    }

    let np = crate::pyimport::import_module(py, "numpy")?;
    let arr = np.call_method1("asarray", (idx,))?;
    // Detect boolean masks via the dtype `kind`, which is stable across numpy
    // versions (unlike `str(dtype)`, which can be "bool" / "bool_" / "bool8").
    let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    let arr = if kind == "b" {
        let mask_len: usize = arr.len()?;
        if mask_len != len {
            return Err(PyIndexError::new_err(format!(
                "boolean index did not match indexed array; size of axis is {len} \
                 but size of corresponding boolean axis is {mask_len}"
            )));
        }
        arr.call_method0("nonzero")?
            .cast::<PyTuple>()?
            .get_item(0)?
    } else {
        arr
    };
    let flat = arr.call_method1("astype", (np.getattr("int64")?,))?;
    let readonly: numpy::PyReadonlyArray1<'_, i64> = flat.extract()?;
    let slice = readonly
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    // Normalize negatives here rather than leaving it to the compose helpers:
    // `expressible_as_window` inspects these values, and `[-1, 0]` is ascending
    // as written but descending once resolved — taking the lazy path on it
    // would build exactly the non-monotone map that panics.
    slice
        .iter()
        .map(|&i| {
            let n = len as i64;
            let normalized = if i < 0 { i + n } else { i };
            if normalized < 0 || normalized >= n {
                return Err(PyIndexError::new_err(format!(
                    "index {i} is out of bounds for axis of length {len}"
                )));
            }
            Ok(normalized)
        })
        .collect::<PyResult<Vec<i64>>>()
        .map(Some)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Every SCX handle class that can appear as `X`, a layer, or an aligned value.
const HANDLE_CLASSES: [&str; 4] = [
    "ScxBackedSparseDataset",
    "ScxBackedLayerDataset",
    "ScxBackedObsmDataset",
    "ScxLazyTransformedDataset",
];

/// One `singledispatch` seam: where it lives and which implementation it takes.
struct Seam {
    module: &'static str,
    attr: &'static str,
}

const SEAMS: [Seam; 3] = [
    Seam {
        module: "anndata._core.views",
        attr: "as_view",
    },
    Seam {
        module: "anndata._core.index",
        attr: "_subset",
    },
    Seam {
        module: "anndata._core.file_backing",
        attr: "to_memory",
    },
];

/// Register all three `singledispatch` hooks for all four handle classes.
///
/// Best-effort, matching the neighbouring `anndata.abc.CSRDataset`
/// registration: pyscx is usable without anndata. Success is recorded in
/// [`HOOKS_REGISTERED`] so the axis-subset path can fail with a useful message
/// instead of anndata's.
pub(crate) fn register_anndata_subset_hooks(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    if crate::pyimport::import_module(py, "anndata").is_err() {
        // anndata not installed — scx is usable without it, no warning.
        return Ok(());
    }

    let impls = [
        pyo3::wrap_pyfunction!(scx_as_view, m)?,
        pyo3::wrap_pyfunction!(scx_subset, m)?,
        pyo3::wrap_pyfunction!(scx_to_memory, m)?,
    ];

    let mut all_ok = true;
    for (seam, hook_impl) in SEAMS.iter().zip(impls.iter()) {
        let dispatcher = match crate::pyimport::import_module(py, seam.module)
            .and_then(|module| module.getattr(seam.attr))
        {
            Ok(d) => d,
            Err(err) => {
                log::warn!(
                    "{}.{} lookup failed ({err}); backed axis subsetting will be \
                     unavailable.",
                    seam.module,
                    seam.attr
                );
                all_ok = false;
                continue;
            }
        };
        let register = match dispatcher.getattr("register") {
            Ok(r) => r,
            Err(err) => {
                log::warn!(
                    "{}.{} is not a singledispatch ({err}); backed axis subsetting \
                     will be unavailable.",
                    seam.module,
                    seam.attr
                );
                all_ok = false;
                continue;
            }
        };
        for cls_name in HANDLE_CLASSES {
            if let Err(err) = register.call1((m.getattr(cls_name)?, hook_impl)) {
                log::warn!(
                    "failed to register {cls_name} with {}.{}: {err}",
                    seam.module,
                    seam.attr
                );
                all_ok = false;
            }
        }
    }

    HOOKS_REGISTERED.store(all_ok, Ordering::Relaxed);
    Ok(())
}
