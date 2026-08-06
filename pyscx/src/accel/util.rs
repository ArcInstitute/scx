//! Shared utility helpers for CSR extraction and type conversion.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
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

/// Holds borrowed CSR array slices extracted from a scipy CSR matrix.
///
/// The `PyReadonlyArray1` borrows keep the underlying numpy arrays alive
/// for the lifetime `'py`.
pub(super) struct CsrSlices<'py> {
    _indptr: numpy::PyReadonlyArray1<'py, i64>,
    _indices: numpy::PyReadonlyArray1<'py, i32>,
    _data: numpy::PyReadonlyArray1<'py, f32>,
}

impl<'py> CsrSlices<'py> {
    pub(super) fn indptr(&self) -> &[i64] {
        // SAFETY: the readonly array is guaranteed contiguous by the
        // as_slice() check in extract_csr_slices.
        self._indptr.as_slice().unwrap()
    }
    pub(super) fn indices(&self) -> &[i32] {
        self._indices.as_slice().unwrap()
    }
    pub(super) fn data(&self) -> &[f32] {
        self._data.as_slice().unwrap()
    }
}

/// Extract CSR indptr/indices/data as borrowed Rust slices from a scipy
/// CSR matrix object.
///
/// This consolidates the repeated `getattr → asarray → astype_no_copy →
/// PyReadonlyArray1 → as_slice` pattern used by multiple bindings.
/// Emits a Python `warnings.warn()` if the CSR `nnz` exceeds `warn_nnz`
/// (set to 0 to suppress the warning).
pub(super) fn extract_csr_slices<'py>(
    py: Python<'py>,
    np: &Bound<'py, PyModule>,
    csr: &Bound<'py, PyAny>,
    warn_label: &str,
    warn_nnz: usize,
) -> PyResult<CsrSlices<'py>> {
    // Optional large-data warning based on nnz.
    if warn_nnz > 0 {
        let nnz: usize = csr.getattr("nnz")?.extract()?;
        // Each nonzero costs 4 bytes (data) + 4 bytes (index) = 8 bytes.
        // indptr is small relative to nnz for large matrices.
        let estimated_bytes = nnz * 8;
        if estimated_bytes > 2_000_000_000 {
            let gb = estimated_bytes as f64 / 1e9;
            let shape: (usize, usize) = csr.getattr("shape")?.extract()?;
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                (format!(
                    "{warn_label}: materializing a {:.1} GB CSR matrix ({} × {}, nnz={nnz}). \
                     Consider subsetting the data for better performance.",
                    gb, shape.0, shape.1
                ),),
            )?;
        }
    }

    let indptr_obj = csr.getattr("indptr")?;
    let indptr_arr = np.call_method1("asarray", (&indptr_obj,))?;
    let indptr_arr = astype_no_copy(py, &indptr_arr, "int64")?;
    let indptr_ro: numpy::PyReadonlyArray1<'_, i64> = indptr_arr.extract()?;
    indptr_ro.as_slice().map_err(|e| {
        PyValueError::new_err(format!(
            "CSR indptr is not C-contiguous ({e}); pass a canonical scipy CSR \
             (try `X = X.tocsr(); X.sort_indices()`) or re-run \
             `pyscx.from_anndata(...)` to rewrite it"
        ))
    })?;

    let indices_obj = csr.getattr("indices")?;
    let indices_arr = np.call_method1("asarray", (&indices_obj,))?;
    let indices_arr = astype_no_copy(py, &indices_arr, "int32")?;
    let indices_ro: numpy::PyReadonlyArray1<'_, i32> = indices_arr.extract()?;
    indices_ro.as_slice().map_err(|e| {
        PyValueError::new_err(format!(
            "CSR indices are not C-contiguous ({e}); pass a canonical scipy CSR \
             (try `X = X.tocsr(); X.sort_indices()`) or re-run \
             `pyscx.from_anndata(...)` to rewrite it"
        ))
    })?;

    let data_obj = csr.getattr("data")?;
    let data_arr = np.call_method1("asarray", (&data_obj,))?;
    let data_arr = astype_no_copy(py, &data_arr, "float32")?;
    let data_ro: numpy::PyReadonlyArray1<'_, f32> = data_arr.extract()?;
    data_ro
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(format!("data not contiguous: {e}")))?;

    Ok(CsrSlices {
        _indptr: indptr_ro,
        _indices: indices_ro,
        _data: data_ro,
    })
}

/// Extract a materialized scipy sparse or dense `X` into an in-memory
/// [`scx_sparse::ScxCsr`]. Handles both `scipy.sparse.*` and dense numpy
/// arrays. Shared by the in-memory CPU PCA path and the in-memory HVG path.
pub(super) fn extract_materialized_csr(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
) -> PyResult<scx_sparse::ScxCsr> {
    let scipy_sparse = py.import("scipy.sparse")?;
    let is_sparse = scipy_sparse
        .call_method1("issparse", (x,))?
        .extract::<bool>()?;

    let csr_py = scipy_sparse.call_method1("csr_matrix", (x,))?;
    let shape: (usize, usize) = csr_py.getattr("shape")?.extract()?;

    let (indptr, indices, data): (Vec<i64>, Vec<i32>, Vec<f32>) = if is_sparse {
        let np = py.import("numpy")?;
        let indptr_np = csr_py.getattr("indptr")?;
        let indices_np = csr_py.getattr("indices")?;
        let data_np = csr_py.getattr("data")?;
        let indptr = np
            .call_method1("asarray", (&indptr_np,))?
            .call_method1("astype", ("int64",))?
            .extract::<Vec<i64>>()?;
        let indices = np
            .call_method1("asarray", (&indices_np,))?
            .call_method1("astype", ("int32",))?
            .extract::<Vec<i32>>()?;
        let data = np
            .call_method1("asarray", (&data_np,))?
            .call_method1("astype", ("float32",))?
            .extract::<Vec<f32>>()?;
        (indptr, indices, data)
    } else {
        let indptr = csr_py
            .getattr("indptr")?
            .call_method1("astype", ("int64",))?
            .extract::<Vec<i64>>()?;
        let indices = csr_py
            .getattr("indices")?
            .call_method1("astype", ("int32",))?
            .extract::<Vec<i32>>()?;
        let data = csr_py
            .getattr("data")?
            .call_method1("astype", ("float32",))?
            .extract::<Vec<f32>>()?;
        (indptr, indices, data)
    };

    Ok(scx_sparse::ScxCsr::new_unchecked(
        shape, indptr, indices, data,
    ))
}
