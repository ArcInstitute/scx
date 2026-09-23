//! Lower-level column-axis aggregation entry points.
//!
//! `pyscx.accel.{col_sums,col_nnz,col_min,col_max,col_var}` expose the
//! streaming projected-aggregation kernels with explicit
//! `prefer_format` dispatch. Today the array-protocol dunders on
//! `ScxBackedSparseDataset` (`sum/mean/var/min/max`) cover the same
//! ground but can't accept custom kwargs without breaking numpy
//! interop, so these standalone functions are the path for callers
//! who want to opt into CSC.
//!
//! All five entries return 1-D `numpy.ndarray`s of length
//! `dataset.n_vars` (or `len(col_indices)` if a column projection is
//! active on the dataset).
//!
//! # The GIL is released for the scan
//!
//! Every entry point here does a **full-matrix streaming decode**. Until Phase
//! 4.3 it held the GIL for the whole of it — `run_csr_f64` even had a
//! `let _ = py;` where the release belonged — so `pyscx.accel.col_sums(...)`
//! blocked every other Python thread until it finished, and concurrent use from
//! a dataloader or a server thread pool serialised completely. Every sibling
//! heavy entry point (`filtering.rs`, `preprocessing.rs`, `de.rs`, `pca.rs`)
//! already released it.
//!
//! A `PyRef` cannot cross `py.detach(...)`, so each entry first snapshots what
//! the scan needs into an owned [`CsrHandle`] / a `LazyShardSource` (`Arc` clones and
//! owned index vectors — no matrix data is copied), drops the `PyRef`, runs the
//! kernel detached, and re-acquires only to build the numpy array.

use numpy::PyArray;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

use scx_format_io::BackedCsrReader;

use crate::backed::detached;
use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::LazyShardSource;
use crate::projected_agg;

/// Helper: validate `prefer_format` and translate to a discriminator.
fn validated_prefer(prefer_format: &str) -> PyResult<&'static str> {
    match prefer_format {
        "csr" => Ok("csr"),
        "csc" => Ok("csc"),
        other => Err(PyValueError::new_err(format!(
            "Invalid prefer_format={other:?}; expected 'csr' or 'csc'"
        ))),
    }
}

/// Owned snapshot of a backed handle's CSR state, taken under the GIL so the
/// streaming scan can run detached.
///
/// Cheap: three `Arc` clones.
struct CsrHandle {
    reader: Arc<BackedCsrReader>,
    /// Visible → global row map, when a deletion vector / row subset is active.
    kept: Option<Arc<Vec<u64>>>,
    /// Visible → on-disk column ids, when a projection is active.
    cols: Option<Arc<Vec<u32>>>,
    /// Visible row count (`shape_val.0`), i.e. post-subset.
    n_obs: usize,
}

impl CsrHandle {
    fn extract(dataset: &Bound<'_, PyAny>, op: &str) -> PyResult<Self> {
        let backed = dataset
            .extract::<PyRef<ScxBackedSparseDataset>>()
            .map_err(|_| {
                PyRuntimeError::new_err(format!(
                    "{op} requires adata.X / dataset to be ScxBackedSparseDataset \
                     (lazy datasets currently must materialize for these aggregations; \
                     use the dunder methods on the lazy wrapper which apply transforms)"
                ))
            })?;
        Ok(CsrHandle {
            reader: Arc::clone(&backed.backed),
            kept: backed.kept_to_global.clone(),
            cols: backed.col_projection_arc(),
            n_obs: backed.shape_val.0,
        })
    }

    fn kept_slice(&self) -> Option<&[u64]> {
        self.kept.as_ref().map(|k| k.as_slice())
    }

    /// Visible → on-disk column ids, or `None` when no projection is active.
    ///
    /// Every CSR entry point branches on this rather than materialising the
    /// identity range when it is `None`. `col_min` / `col_max` / `col_var` used
    /// to: the projected kernels run `project_csr` on every decoded shard, so an
    /// identity "projection" copied the whole matrix once per pass (twice for
    /// `col_var`) and made those three 8-15x slower than `col_sums` on the same
    /// handle — slow enough that the CSC sidecar looked like the faster layout
    /// for them when it was not.
    fn cols_slice(&self) -> Option<&[u32]> {
        self.cols.as_ref().map(|c| c.as_slice())
    }
}

/// `(source, columns, n_obs)` for the `prefer_format="csc"` path.
///
/// `columns` are **identity positions**, never `col_projection()`: the source
/// already exposes the projected axis (`n_vars()` is the projected width, and
/// `read_csc_columns` maps a requested range *through* the projection).
/// Passing global ids back in would apply the projection a second time —
/// silently reading the wrong genes, or indexing past a short slab and
/// panicking in `walk_csc_runs`. `n_obs` is likewise the visible count, which
/// is what the compacted slab's rows index into.
///
/// Owned, because the scan runs under `py.detach(...)` and a borrow from the
/// `PyRef` cannot cross it. There is no wrapper type: the kernels take
/// `&dyn ColumnShardSource` and `LazyShardSource` implements it, so one
/// existed only to call `f(&self.0)`.
fn csc_source(dataset: &Bound<'_, PyAny>) -> PyResult<(LazyShardSource, Vec<u32>, usize)> {
    if !crate::accel::is_scx_matrix_handle(dataset) {
        return Err(PyRuntimeError::new_err(
            "prefer_format='csc' requires dataset to be ScxBackedSparseDataset \
             or ScxLazyTransformedDataset",
        ));
    }
    let src = crate::accel::csc_source_for(dataset).ok_or_else(crate::accel::csc_unavailable)?;
    // Both axes off the source, which owns the authoritative visible shape.
    // This used to walk backed-vs-lazy by hand purely to read `shape_val.0`,
    // which `csc_source_for` then walked again — and it did it with the bare
    // `extract::<PyRef<ScxBackedSparseDataset>>()` form that misses a backed
    // *layer* handle, the exact shape `backed_dataset_ref` exists to catch.
    let n_obs = scx_format_io::ColumnShardSource::n_obs(&src);
    let cols: Vec<u32> = (0..scx_format_io::ShardSource::n_vars(&src) as u32).collect();
    Ok((src, cols, n_obs))
}

fn to_py_err(e: scx_format_io::ScxError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Streaming column sums (axis=0). Returns 1-D float64 array of
/// length n_vars.
#[pyfunction]
#[pyo3(signature = (dataset, prefer_format="csr"))]
pub fn col_sums<'py>(
    py: Python<'py>,
    dataset: &Bound<'py, PyAny>,
    prefer_format: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if validated_prefer(prefer_format)? == "csc" {
        let (src, cols, _n_obs) = csc_source(dataset)?;
        let sums = detached(py, || projected_agg::col_sums_projected_csc(&src, &cols))
            .map_err(to_py_err)?;
        return Ok(PyArray::from_vec(py, sums).into_any());
    }

    let h = CsrHandle::extract(dataset, "col_sums")?;
    let sums = detached(py, || match (h.cols_slice(), h.kept_slice()) {
        (Some(cols), Some(kept)) => projected_agg::col_sums_masked_projected(&h.reader, kept, cols),
        (Some(cols), None) => projected_agg::col_sums_projected(&h.reader, cols),
        (None, Some(kept)) => h.reader.col_sums_masked(kept),
        (None, None) => h.reader.col_sums(),
    })
    .map_err(to_py_err)?;
    Ok(PyArray::from_vec(py, sums).into_any())
}

/// Streaming column NNZ (axis=0). Returns 1-D uint32 array.
#[pyfunction]
#[pyo3(signature = (dataset, prefer_format="csr"))]
pub fn col_nnz<'py>(
    py: Python<'py>,
    dataset: &Bound<'py, PyAny>,
    prefer_format: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if validated_prefer(prefer_format)? == "csc" {
        let (src, cols, _n_obs) = csc_source(dataset)?;
        let counts = detached(py, || projected_agg::col_nnz_projected_csc(&src, &cols))
            .map_err(to_py_err)?;
        return Ok(PyArray::from_vec(py, counts).into_any());
    }

    let h = CsrHandle::extract(dataset, "col_nnz")?;
    let counts: Vec<u32> = detached(py, || match (h.cols_slice(), h.kept_slice()) {
        (Some(cols), Some(kept)) => projected_agg::col_nnz_masked_projected(&h.reader, kept, cols),
        (Some(cols), None) => projected_agg::col_nnz_projected(&h.reader, cols),
        // The masked kernel returns Vec<f64>; narrow to the uint32 the
        // unmasked path produces so the dtype is the same either way.
        (None, Some(kept)) => Ok(h
            .reader
            .col_nnz_masked(kept)?
            .iter()
            .map(|&v| v as u32)
            .collect()),
        (None, None) => h.reader.col_nnz(),
    })
    .map_err(to_py_err)?;
    Ok(PyArray::from_vec(py, counts).into_any())
}

/// Streaming column max (axis=0). Returns 1-D float64 array.
#[pyfunction]
#[pyo3(signature = (dataset, prefer_format="csr"))]
pub fn col_max<'py>(
    py: Python<'py>,
    dataset: &Bound<'py, PyAny>,
    prefer_format: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if validated_prefer(prefer_format)? == "csc" {
        let (src, cols, n_obs) = csc_source(dataset)?;
        let res = detached(py, || {
            projected_agg::col_max_projected_csc(&src, &cols, n_obs)
        })
        .map_err(to_py_err)?;
        return Ok(PyArray::from_vec(py, res).into_any());
    }

    let h = CsrHandle::extract(dataset, "col_max")?;
    let maxes = detached(py, || match (h.cols_slice(), h.kept_slice()) {
        // Implicit-zero correction counts against the *visible* row count.
        (Some(cols), Some(kept)) => {
            projected_agg::col_max_masked_projected(&h.reader, kept, cols, kept.len())
        }
        (Some(cols), None) => projected_agg::col_max_projected(&h.reader, cols, h.n_obs),
        (None, Some(kept)) => h.reader.col_max_masked(kept),
        (None, None) => h.reader.col_max(),
    })
    .map_err(to_py_err)?;
    Ok(PyArray::from_vec(py, maxes).into_any())
}

/// Streaming column min (axis=0).
#[pyfunction]
#[pyo3(signature = (dataset, prefer_format="csr"))]
pub fn col_min<'py>(
    py: Python<'py>,
    dataset: &Bound<'py, PyAny>,
    prefer_format: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if validated_prefer(prefer_format)? == "csc" {
        let (src, cols, n_obs) = csc_source(dataset)?;
        let res = detached(py, || {
            projected_agg::col_min_projected_csc(&src, &cols, n_obs)
        })
        .map_err(to_py_err)?;
        return Ok(PyArray::from_vec(py, res).into_any());
    }

    let h = CsrHandle::extract(dataset, "col_min")?;
    let mins = detached(py, || match (h.cols_slice(), h.kept_slice()) {
        (Some(cols), Some(kept)) => {
            projected_agg::col_min_masked_projected(&h.reader, kept, cols, kept.len())
        }
        (Some(cols), None) => projected_agg::col_min_projected(&h.reader, cols, h.n_obs),
        (None, Some(kept)) => h.reader.col_min_masked(kept),
        (None, None) => h.reader.col_min(),
    })
    .map_err(to_py_err)?;
    Ok(PyArray::from_vec(py, mins).into_any())
}

/// Streaming column variance (axis=0). Population variance (n_obs in
/// the denominator) over all rows or kept rows when a deletion vector
/// is active.
#[pyfunction]
#[pyo3(signature = (dataset, prefer_format="csr"))]
pub fn col_var<'py>(
    py: Python<'py>,
    dataset: &Bound<'py, PyAny>,
    prefer_format: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if validated_prefer(prefer_format)? == "csc" {
        let (src, cols, n_obs) = csc_source(dataset)?;
        let res = detached(py, || {
            projected_agg::col_var_projected_csc(&src, &cols, n_obs)
        })
        .map_err(to_py_err)?;
        return Ok(PyArray::from_vec(py, res).into_any());
    }

    let h = CsrHandle::extract(dataset, "col_var")?;
    let vars = detached(py, || match (h.cols_slice(), h.kept_slice()) {
        // The masked kernels derive their denominator from `kept`.
        (Some(cols), Some(kept)) => projected_agg::col_var_masked_projected(&h.reader, kept, cols),
        (Some(cols), None) => projected_agg::col_var_projected(&h.reader, cols, h.n_obs),
        (None, Some(kept)) => h.reader.col_var_masked(kept),
        (None, None) => h.reader.col_var(),
    })
    .map_err(to_py_err)?;
    Ok(PyArray::from_vec(py, vars).into_any())
}
