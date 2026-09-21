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
//! the scan needs into an owned [`CsrHandle`] / [`CscHandle`] (`Arc` clones and
//! owned index vectors — no matrix data is copied), drops the `PyRef`, runs the
//! kernel detached, and re-acquires only to build the numpy array.

use numpy::PyArray;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

use scx_format_io::{BackedCscReader, BackedCsrReader};

use crate::backed::detached;
use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::{LazyShardSource, ScxLazyTransformedDataset};
use crate::projected_agg;

type AggResult<T> = std::result::Result<T, scx_format_io::ScxError>;

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
/// Cheap: two `Arc` clones and (at most) one `Vec<u32>` of visible column ids.
struct CsrHandle {
    reader: Arc<BackedCsrReader>,
    /// Visible → global row map, when a deletion vector / row subset is active.
    kept: Option<Arc<Vec<u64>>>,
    /// Visible → on-disk column ids, when a projection is active.
    cols: Option<Arc<Vec<u32>>>,
    /// Visible row count (`shape_val.0`), i.e. post-subset.
    n_obs: usize,
    /// Visible column count (`shape_val.1`), i.e. post-projection.
    n_vars: usize,
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
            n_vars: backed.shape_val.1,
        })
    }

    fn kept_slice(&self) -> Option<&[u64]> {
        self.kept.as_ref().map(|k| k.as_slice())
    }

    fn cols_slice(&self) -> Option<&[u32]> {
        self.cols.as_ref().map(|c| c.as_slice())
    }

    /// Visible column ids, materialising the identity range when no projection
    /// is active (several kernels take a column list unconditionally).
    fn cols_or_identity(&self) -> Vec<u32> {
        match self.cols_slice() {
            Some(c) => c.to_vec(),
            None => (0..self.n_vars as u32).collect(),
        }
    }
}

/// Owned CSC source for the `prefer_format="csc"` path.
///
/// Both variants are `Send`, which is the point: `as_column_source()` hands
/// back a `&dyn ColumnShardSource` borrowed from the `PyRef` and so cannot be
/// moved into a detached closure.
enum CscHandle {
    Backed(Arc<BackedCscReader>),
    Lazy(Box<LazyShardSource>),
}

impl CscHandle {
    fn extract(dataset: &Bound<'_, PyAny>) -> PyResult<(Self, Vec<u32>, usize)> {
        if let Ok(backed) = dataset.extract::<PyRef<ScxBackedSparseDataset>>() {
            let csc = backed.as_column_source_owned().ok_or_else(|| {
                crate::accel::csc_unavailable(
                    backed.backed_csc.is_some(),
                    backed.kept_to_global.is_some(),
                )
            })?;
            let cols: Vec<u32> = match backed.col_projection() {
                Some(c) => c.to_vec(),
                None => (0..backed.shape_val.1 as u32).collect(),
            };
            return Ok((CscHandle::Backed(csc), cols, backed.shape_val.0));
        }
        if let Ok(lazy) = dataset.extract::<PyRef<ScxLazyTransformedDataset>>() {
            let src = lazy.as_column_source().ok_or_else(|| {
                crate::accel::csc_unavailable(
                    lazy.backed_csc.is_some(),
                    lazy.kept_to_global.is_some(),
                )
            })?;
            // Identity positions, NOT `lazy.col_projection()`: `src` already
            // exposes the projected axis (`n_vars()` is the projected width,
            // and `read_csc_columns` maps a requested range *through* the
            // projection). Passing the global ids back in would apply the
            // projection a second time — silently reading the wrong genes, or
            // indexing past a short slab and panicking in `walk_csc_runs`.
            let cols: Vec<u32> = (0..scx_format_io::ShardSource::n_vars(&src) as u32).collect();
            return Ok((CscHandle::Lazy(Box::new(src)), cols, lazy.shape_val.0));
        }
        Err(PyRuntimeError::new_err(
            "prefer_format='csc' requires dataset to be ScxBackedSparseDataset \
             or ScxLazyTransformedDataset",
        ))
    }

    /// Run `f` against whichever concrete column source this holds.
    ///
    /// A closure rather than `&dyn ColumnShardSource` because the CSC kernels
    /// are generic over `S: ColumnShardSource` and the two variants are
    /// different concrete types.
    fn with<T>(
        &self,
        f: impl Fn(&dyn scx_format_io::ColumnShardSource) -> AggResult<T>,
    ) -> AggResult<T> {
        match self {
            CscHandle::Backed(csc) => f(csc.as_ref() as &dyn scx_format_io::ColumnShardSource),
            CscHandle::Lazy(src) => f(src.as_ref() as &dyn scx_format_io::ColumnShardSource),
        }
    }
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
        let (src, cols, _n_obs) = CscHandle::extract(dataset)?;
        let sums = detached(py, || {
            src.with(|s| projected_agg::col_sums_projected_csc(s, &cols))
        })
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
        let (src, cols, _n_obs) = CscHandle::extract(dataset)?;
        let counts = detached(py, || {
            src.with(|s| projected_agg::col_nnz_projected_csc(s, &cols))
        })
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
        let (src, cols, n_obs) = CscHandle::extract(dataset)?;
        let res = detached(py, || {
            src.with(|s| projected_agg::col_max_projected_csc(s, &cols, n_obs))
        })
        .map_err(to_py_err)?;
        return Ok(PyArray::from_vec(py, res).into_any());
    }

    let h = CsrHandle::extract(dataset, "col_max")?;
    let cols_owned = h.cols_or_identity();
    let maxes = detached(py, || match h.kept_slice() {
        // Implicit-zero correction counts against the *visible* row count.
        Some(kept) => {
            projected_agg::col_max_masked_projected(&h.reader, kept, &cols_owned, kept.len())
        }
        None => projected_agg::col_max_projected(&h.reader, &cols_owned, h.n_obs),
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
        let (src, cols, n_obs) = CscHandle::extract(dataset)?;
        let res = detached(py, || {
            src.with(|s| projected_agg::col_min_projected_csc(s, &cols, n_obs))
        })
        .map_err(to_py_err)?;
        return Ok(PyArray::from_vec(py, res).into_any());
    }

    let h = CsrHandle::extract(dataset, "col_min")?;
    let cols_owned = h.cols_or_identity();
    let mins = detached(py, || match h.kept_slice() {
        Some(kept) => {
            projected_agg::col_min_masked_projected(&h.reader, kept, &cols_owned, kept.len())
        }
        None => projected_agg::col_min_projected(&h.reader, &cols_owned, h.n_obs),
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
        let (src, cols, n_obs) = CscHandle::extract(dataset)?;
        let res = detached(py, || {
            src.with(|s| projected_agg::col_var_projected_csc(s, &cols, n_obs))
        })
        .map_err(to_py_err)?;
        return Ok(PyArray::from_vec(py, res).into_any());
    }

    let h = CsrHandle::extract(dataset, "col_var")?;
    let cols_owned = h.cols_or_identity();
    let vars = detached(py, || match h.kept_slice() {
        // `col_var_masked_projected` derives its denominator from `kept`.
        Some(kept) => projected_agg::col_var_masked_projected(&h.reader, kept, &cols_owned),
        None => projected_agg::col_var_projected(&h.reader, &cols_owned, h.n_obs),
    })
    .map_err(to_py_err)?;
    Ok(PyArray::from_vec(py, vars).into_any())
}
