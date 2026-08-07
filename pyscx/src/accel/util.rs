//! Shared utility helpers for CSR extraction and type conversion.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Has this AnnData been log1p-transformed, per its `uns` annotation?
///
/// The single definition of that question. It used to be spelled three
/// different ways — `uns.get("log1p") is not None` in `de.rs` (twice) and
/// `"log1p" in uns` in `nb_glm.rs`, which disagree on a `uns["log1p"] = None`
/// entry. Key presence is scanpy's own test (`"log1p" in adata.uns`) and is the
/// conservative direction for the NB-GLM raw-counts guard.
///
/// `hvg.rs`'s `log1p_base_scale` is deliberately not routed through here: it
/// asks a different question (*which* log base was used), not whether one was.
pub(crate) fn uns_log1p_present(adata: &Bound<'_, PyAny>) -> bool {
    adata
        .getattr("uns")
        .and_then(|uns| uns.contains("log1p"))
        .unwrap_or(false)
}

/// Stamp `uns["log1p"] = {"base": None}`, matching `sc.pp.log1p`.
///
/// Called by the backed and lazy arms of `pyscx.accel.log1p`, which produce a
/// transformed `X` without going through scanpy and so would otherwise leave
/// the file's own downstream consumers — `rank_genes_groups`' `expm1` logFC
/// branch, `pdex_ref`'s mean mode, `pdex_nb_glm`'s raw-counts guard — believing
/// the data is still counts.
pub(crate) fn stamp_uns_log1p(py: Python<'_>, adata: &Bound<'_, PyAny>) -> PyResult<()> {
    let entry = PyDict::new(py);
    // `{"base": None}` — natural log, exactly what `sc.pp.log1p` records.
    // `hvg.rs`'s `log1p_base_scale` reads this key back.
    entry.set_item("base", py.None())?;
    adata.getattr("uns")?.set_item("log1p", entry)?;
    Ok(())
}

/// Build a 2-D `float32` numpy array from a single flat row-major `Vec<f32>`
/// of length `rows * cols` (shape `[rows, cols]`).
///
/// Replaces the `Vec<Vec<f32>>` + `PyArray2::from_vec2` pattern, which
/// allocated one small inner `Vec` per row (N allocations at atlas scale).
/// `PyArray1::from_vec` hands the owned buffer to numpy (no copy) and
/// `reshape` returns a row-major view — so there is no `unsafe`, no extra
/// copy, and a length mismatch fails loud as a Python error at the FFI
/// boundary (reshape rejects `rows*cols != data.len()`) rather than aborting
/// the interpreter.
pub(super) fn flat_pyarray2<'py>(
    py: Python<'py>,
    data: Vec<f32>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    PyArray1::from_vec(py, data)
        .reshape([rows, cols])
        .map_err(|e| {
            PyValueError::new_err(format!("flat_pyarray2: cannot reshape {rows}×{cols}: {e}"))
        })
}

/// Call `array.astype(dtype, copy=False)` — avoids a deep copy when the
/// source already has the target dtype. This mirrors numpy's behavior where
/// `copy=False` returns the same array object if no conversion is needed.
pub(super) fn astype_no_copy<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("copy", false)?;
    arr.call_method("astype", (dtype,), Some(&kwargs))
}
