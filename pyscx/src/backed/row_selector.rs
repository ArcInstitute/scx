//! One row-selector resolver for every sparse handle and for
//! `Experiment.gather_rows_sparse`.
//!
//! `handle[rows]` on `ScxBackedSparseDataset`, `ScxBackedLayerDataset` and
//! `ScxLazyTransformedDataset`, and `gather_rows_sparse(rows)`, all accept the
//! same two array selectors — a boolean mask or a 1-D integer array-like — and
//! must agree on what is out of range. The two `getitem_rows` bodies used to
//! carry their own copies of this and disagreed on the one check numpy makes
//! that neither did: a boolean mask of the wrong length (a longer mask failed
//! late through `to_global_row`, a shorter one silently returned fewer rows).

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Resolve a row selector to **visible** row positions in `[0, n_visible)`.
///
/// - A boolean array must have exactly `n_visible` entries (`IndexError`
///   otherwise, numpy's rule); its `True` positions are returned in order.
/// - An integer array-like (ndarray, list, range) may be unsorted and may
///   repeat rows; negative entries of a *signed* array wrap once (`-1` is the
///   last row); anything still outside `[0, n_visible)` is an `IndexError`
///   naming the value. An unsigned array is bounds-checked as `uint64` — it is
///   never converted to a signed type, so `2**64 - 1` is out of range, not
///   the last row.
/// - Anything else — a float array, a 2-D array — is an `IndexError`, as in
///   numpy ("arrays used as indices must be of integer (or boolean) type").
///
/// Reads `dtype.kind`, never `str(dtype)`: numpy's C code imports
/// `numpy._core._dtype` on every dtype stringification via the frame-sensitive
/// `PyImport_Import`, which detonates when a handle is indexed from
/// restricted-exec globals (no `__import__`). `kind` is a plain C descriptor
/// char. Pinned by `test_sandbox_exec.py`.
pub(crate) fn resolve_row_selector(
    py: Python<'_>,
    selector: &Bound<'_, PyAny>,
    n_visible: usize,
) -> PyResult<Vec<usize>> {
    let np = crate::pyimport::import_module(py, "numpy")?;
    let arr = np.call_method1("asarray", (selector,))?;
    let dtype_kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    let ndim: usize = arr.getattr("ndim")?.extract()?;
    if ndim != 1 {
        return Err(PyIndexError::new_err(format!(
            "row selector must be one-dimensional, got a {ndim}-D array"
        )));
    }

    if dtype_kind == "b" {
        let len = arr.len()?;
        if len != n_visible {
            return Err(PyIndexError::new_err(format!(
                "boolean row mask has {len} entries but the matrix has {n_visible} rows"
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
        // turn into "the last row" instead of an `IndexError`.
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
                        "row index {v} out of range for {n_visible} rows"
                    )));
                }
                Ok(v as usize)
            })
            .collect();
    }
    if dtype_kind != "i" {
        return Err(PyIndexError::new_err(format!(
            "row selector must be an integer array or a boolean mask, got dtype kind '{dtype_kind}'"
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
                    "row index {v} out of range for {n_visible} rows"
                )));
            }
            Ok(normalized as usize)
        })
        .collect()
}
