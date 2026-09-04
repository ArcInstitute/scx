//! One selector resolver for every sparse handle's rows **and** columns, and
//! for `Experiment.gather_rows_sparse`.
//!
//! `handle[rows]` and `handle[:, cols]` on `ScxBackedSparseDataset`,
//! `ScxBackedLayerDataset` and `ScxLazyTransformedDataset`, and
//! `gather_rows_sparse(rows)`, all accept the same array selectors — a boolean
//! mask or a 1-D integer array-like — and must agree on what is out of range.
//! The per-class `getitem_rows` bodies used to carry their own copies of the
//! row rules and disagreed on the one check numpy makes that neither did (a
//! boolean mask of the wrong length); the column arm accepted only an
//! ascending `ndarray` and silently decoded the **whole matrix** for an `int`,
//! a `list`, a `slice`, a `range` or a reordered array (S11 / REC-7).

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};

/// Which axis a selector addresses — only the words in the error messages
/// differ, and the row wording is pinned by tests (`"boolean row mask"`,
/// `"row index … out of range for … rows"`).
#[derive(Clone, Copy)]
pub(crate) enum Axis {
    Row,
    Col,
}

impl Axis {
    fn singular(self) -> &'static str {
        match self {
            Axis::Row => "row",
            Axis::Col => "column",
        }
    }

    fn plural(self) -> &'static str {
        match self {
            Axis::Row => "rows",
            Axis::Col => "columns",
        }
    }
}

/// Resolve an array selector to **visible** positions in `[0, n_visible)`.
///
/// - A boolean array must have exactly `n_visible` entries (`IndexError`
///   otherwise, numpy's rule); its `True` positions are returned in order.
/// - An integer array-like (ndarray, list, range, tuple) may be unsorted and
///   may repeat positions; negative entries of a *signed* array wrap once
///   (`-1` is the last row / column); anything still outside `[0, n_visible)`
///   is an `IndexError` naming the value. An unsigned array is bounds-checked
///   as `uint64` — it is never converted to a signed type, so `2**64 - 1` is
///   out of range, not the last position.
/// - An empty plain sequence (`[]`, `range(0)`, `()`) selects nothing, as in
///   numpy, even though `np.asarray` types it `float64`.
/// - Anything else — a float ndarray (empty or not), a 0-D or 2-D array — is
///   an `IndexError`, as in numpy ("arrays used as indices must be of integer
///   (or boolean) type").
///
/// Reads `dtype.kind`, never `str(dtype)`: numpy's C code imports
/// `numpy._core._dtype` on every dtype stringification via the frame-sensitive
/// `PyImport_Import`, which detonates when a handle is indexed from
/// restricted-exec globals (no `__import__`). `kind` is a plain C descriptor
/// char. Pinned by `test_sandbox_exec.py`.
pub(crate) fn resolve_axis_selector(
    py: Python<'_>,
    selector: &Bound<'_, PyAny>,
    n_visible: usize,
    axis: Axis,
) -> PyResult<Vec<usize>> {
    let what = axis.singular();
    let whats = axis.plural();
    let np = crate::pyimport::import_module(py, "numpy")?;
    let arr = np.call_method1("asarray", (selector,))?;
    let dtype_kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    let ndim: usize = arr.getattr("ndim")?.extract()?;
    if ndim != 1 {
        return Err(PyIndexError::new_err(format!(
            "{what} selector must be one-dimensional, got a {ndim}-D array"
        )));
    }
    // `np.asarray([])` / `np.asarray(range(0))` is an empty *float64* array,
    // so a plain empty sequence would fall into the dtype rejection below.
    // numpy treats `x[[]]` as an empty integer index; only an ndarray the
    // caller explicitly typed float is refused.
    if arr.len()? == 0 && !selector.is_instance(&np.getattr("ndarray")?)? {
        return Ok(Vec::new());
    }

    if dtype_kind == "b" {
        let len = arr.len()?;
        if len != n_visible {
            return Err(PyIndexError::new_err(format!(
                "boolean {what} mask has {len} entries but the matrix has {n_visible} {whats}"
            )));
        }
        // nonzero returns a tuple of arrays; for 1-D, it's (array_of_indices,)
        let nonzero = arr.call_method0("nonzero")?;
        let idx_arr = nonzero.cast::<PyTuple>()?.get_item(0)?;
        let flat = idx_arr.call_method1("astype", (np.getattr("int64")?,))?;
        let readonly: numpy::PyReadonlyArray1<'_, i64> = flat.extract()?;
        let slice = readonly
            .as_slice()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        return Ok(slice.iter().map(|&v| v as usize).collect());
    }

    if dtype_kind == "u" {
        // Unsigned stays unsigned: casting `uint64` to `int64` wraps
        // `2**64 - 1` to `-1`, which the negative-wrap rule below would then
        // turn into "the last position" instead of an `IndexError`.
        let flat = arr.call_method1("astype", (np.getattr("uint64")?,))?;
        let readonly: numpy::PyReadonlyArray1<'_, u64> = flat.extract()?;
        let slice = readonly
            .as_slice()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        return slice
            .iter()
            .map(|&v| {
                if v >= n_visible as u64 {
                    return Err(PyIndexError::new_err(format!(
                        "{what} index {v} out of range for {n_visible} {whats}"
                    )));
                }
                Ok(v as usize)
            })
            .collect();
    }
    if dtype_kind != "i" {
        return Err(PyIndexError::new_err(format!(
            "{what} selector must be an integer array or a boolean mask, got dtype kind '{dtype_kind}'"
        )));
    }
    let flat = arr.call_method1("astype", (np.getattr("int64")?,))?;
    let readonly: numpy::PyReadonlyArray1<'_, i64> = flat.extract()?;
    let slice = readonly
        .as_slice()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let n = n_visible as i64;
    slice
        .iter()
        .map(|&v| {
            let normalized = if v < 0 { n + v } else { v };
            if normalized < 0 || normalized >= n {
                return Err(PyIndexError::new_err(format!(
                    "{what} index {v} out of range for {n_visible} {whats}"
                )));
            }
            Ok(normalized as usize)
        })
        .collect()
}

/// [`resolve_axis_selector`] for the row axis — `handle[rows]` and
/// `gather_rows_sparse(rows)`.
pub(crate) fn resolve_row_selector(
    py: Python<'_>,
    selector: &Bound<'_, PyAny>,
    n_visible: usize,
) -> PyResult<Vec<usize>> {
    resolve_axis_selector(py, selector, n_visible, Axis::Row)
}

/// [`resolve_axis_selector`] for the column axis.
pub(crate) fn resolve_col_selector(
    py: Python<'_>,
    selector: &Bound<'_, PyAny>,
    n_visible: usize,
) -> PyResult<Vec<usize>> {
    resolve_axis_selector(py, selector, n_visible, Axis::Col)
}

/// Resolve the column half of `handle[:, cols]` to visible column positions in
/// request order, covering the forms an array selector does not: a `slice`
/// (expanded through `slice.indices`, so a negative step is a reorder) and a
/// scalar `int` (one column; negative wraps once). Everything else goes
/// through [`resolve_col_selector`].
///
/// Returns `Ok(None)` for the full `:` slice: scipy's `X[:, :]` is a copy of
/// the matrix, so the caller keeps returning the materialised row CSR there,
/// consistent with `X[:]`.
pub(crate) fn resolve_col_request(
    py: Python<'_>,
    col_idx: &Bound<'_, PyAny>,
    n_visible: usize,
) -> PyResult<Option<Vec<usize>>> {
    if let Ok(slice) = col_idx.cast::<PySlice>() {
        let ind = slice.indices(n_visible as isize)?;
        if ind.start == 0 && ind.stop == n_visible as isize && ind.step == 1 {
            return Ok(None);
        }
        let mut out = Vec::with_capacity(ind.slicelength);
        let mut i = ind.start;
        for _ in 0..ind.slicelength {
            out.push(i as usize);
            i += ind.step;
        }
        return Ok(Some(out));
    }
    // A Python `bool` is an `int` subclass; leave it to the array rules
    // (numpy rejects a 0-D mask) rather than reading `True` as column 1.
    if !col_idx.is_instance_of::<pyo3::types::PyBool>() {
        if let Ok(k) = col_idx.extract::<i64>() {
            let n = n_visible as i64;
            let normalized = if k < 0 { n + k } else { k };
            if normalized < 0 || normalized >= n {
                return Err(PyIndexError::new_err(format!(
                    "column index {k} out of range for {n_visible} columns"
                )));
            }
            return Ok(Some(vec![normalized as usize]));
        }
    }
    Ok(Some(resolve_col_selector(py, col_idx, n_visible)?))
}

/// `mat[:, remap]` for a scipy sparse matrix.
pub(crate) fn scipy_column_gather<'py>(
    py: Python<'py>,
    mat: &Bound<'py, PyAny>,
    remap: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    let builtins = crate::pyimport::import_module(py, "builtins")?;
    let slice_none = builtins.call_method1("slice", (py.None(),))?;
    let cols: Vec<i64> = remap.iter().map(|&c| c as i64).collect();
    let cols = numpy::PyArray1::from_vec(py, cols);
    let key = PyTuple::new(py, [slice_none, cols.into_any()])?;
    mat.get_item(key)
}

/// Whether `idx` is the full `:` over an axis of `len` (`slice(None)`, or any
/// slice that resolves to `0..len` with step 1). On the row axis it is the gate
/// for `handle[:, cols]` projecting instead of reading; on the column axis it
/// is what lets `handle[rows, :]` return the row read unchanged. Shared by the
/// backed and lazy handles.
pub(crate) fn is_full_slice(idx: &Bound<'_, PyAny>, len: usize) -> PyResult<bool> {
    if let Ok(slice) = idx.cast::<PySlice>() {
        let ind = slice.indices(len as isize)?;
        Ok(ind.start == 0 && ind.stop == len as isize && ind.step == 1)
    } else {
        Ok(false)
    }
}
