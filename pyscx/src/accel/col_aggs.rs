//! Lower-level column-axis aggregation entry points (Phase F.5).
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

use numpy::PyArray;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::ScxLazyTransformedDataset;
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

/// Generic CSR-path adapter: invokes the right `col_*_projected*`
/// helper based on (col_projection, kept_to_global) state. Returns the
/// column-axis result as `Vec<f64>`. Caller wraps in numpy.
fn run_csr_f64<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    full: impl FnOnce(&scx_format::BackedCsrReader) -> scx_format::Result<Vec<f64>>,
    proj: impl FnOnce(&scx_format::BackedCsrReader, &[u32]) -> scx_format::Result<Vec<f64>>,
    masked: impl FnOnce(&scx_format::BackedCsrReader, &[u64]) -> scx_format::Result<Vec<f64>>,
    masked_proj: impl FnOnce(
        &scx_format::BackedCsrReader,
        &[u64],
        &[u32],
    ) -> scx_format::Result<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    let _ = py;
    if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        let kept = backed.kept_to_global.as_ref().map(|a| a.as_slice());
        let cols = backed.col_projection();
        match (cols, kept) {
            (Some(cols), Some(kept)) => masked_proj(&backed.backed, kept, cols)
                .map_err(|e| PyRuntimeError::new_err(e.to_string())),
            (Some(cols), None) => {
                proj(&backed.backed, cols).map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }
            (None, Some(kept)) => {
                masked(&backed.backed, kept).map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }
            (None, None) => {
                full(&backed.backed).map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }
        }
    } else {
        Err(PyRuntimeError::new_err(
            "col_* aggregations require adata.X / dataset to be ScxBackedSparseDataset \
             (lazy datasets currently must materialize for these aggregations; \
             use the dunder methods on the lazy wrapper which apply transforms)",
        ))
    }
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
    let prefer = validated_prefer(prefer_format)?;

    if prefer == "csc" {
        return col_sums_csc_dispatch(py, dataset);
    }

    // CSR path.
    let sums = run_csr_f64(
        py,
        dataset,
        |r| r.col_sums(),
        projected_agg::col_sums_projected,
        |r, kept| r.col_sums_masked(kept),
        projected_agg::col_sums_masked_projected,
    )?;
    Ok(PyArray::from_vec(py, sums).into_any())
}

fn col_sums_csc_dispatch<'py>(
    py: Python<'py>,
    dataset: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(backed) = dataset.extract::<PyRef<ScxBackedSparseDataset>>() {
        let source = backed.as_column_source().ok_or_else(|| {
            PyRuntimeError::new_err(
                "CSC requested but unavailable: file has no CSC sidecar, \
                 or a row deletion vector is active",
            )
        })?;
        let cols = backed.col_projection();
        let cols_owned: Vec<u32> = match cols {
            Some(c) => c.to_vec(),
            None => (0..backed.shape_val.1 as u32).collect(),
        };
        let sums = projected_agg::col_sums_projected_csc(source, &cols_owned)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        return Ok(PyArray::from_vec(py, sums).into_any());
    }
    if let Ok(lazy) = dataset.extract::<PyRef<ScxLazyTransformedDataset>>() {
        let lazy_src = lazy.as_column_source().ok_or_else(|| {
            PyRuntimeError::new_err(
                "CSC requested but unavailable: file has no CSC sidecar, \
                 the transform chain contains a non-column-local op, or a \
                 row deletion vector is active",
            )
        })?;
        let cols = lazy.col_projection();
        let cols_owned: Vec<u32> = match cols {
            Some(c) => c.to_vec(),
            None => (0..lazy.shape_val.1 as u32).collect(),
        };
        let sums = projected_agg::col_sums_projected_csc(&lazy_src, &cols_owned)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        return Ok(PyArray::from_vec(py, sums).into_any());
    }
    Err(PyRuntimeError::new_err(
        "prefer_format='csc' requires dataset to be ScxBackedSparseDataset \
         or ScxLazyTransformedDataset",
    ))
}

/// Streaming column NNZ (axis=0). Returns 1-D int64 array.
#[pyfunction]
#[pyo3(signature = (dataset, prefer_format="csr"))]
pub fn col_nnz<'py>(
    py: Python<'py>,
    dataset: &Bound<'py, PyAny>,
    prefer_format: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let prefer = validated_prefer(prefer_format)?;

    if prefer == "csc" {
        return col_nnz_csc_dispatch(py, dataset);
    }

    // CSR path: returns Vec<i64>.
    if let Ok(backed) = dataset.extract::<PyRef<ScxBackedSparseDataset>>() {
        let kept = backed.kept_to_global.as_ref().map(|a| a.as_slice());
        let cols = backed.col_projection();
        let counts: Vec<i64> = match (cols, kept) {
            (Some(cols), Some(kept)) => {
                projected_agg::col_nnz_masked_projected(&backed.backed, kept, cols)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
            (Some(cols), None) => projected_agg::col_nnz_projected(&backed.backed, cols)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            (None, Some(kept)) => backed
                .backed
                .col_nnz_masked(kept)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .iter()
                .map(|&v| v as i64)
                .collect(),
            (None, None) => backed
                .backed
                .col_nnz()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        };
        return Ok(PyArray::from_vec(py, counts).into_any());
    }
    Err(PyRuntimeError::new_err(
        "col_nnz requires dataset to be ScxBackedSparseDataset",
    ))
}

fn col_nnz_csc_dispatch<'py>(
    py: Python<'py>,
    dataset: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(backed) = dataset.extract::<PyRef<ScxBackedSparseDataset>>() {
        let source = backed
            .as_column_source()
            .ok_or_else(|| PyRuntimeError::new_err("CSC requested but unavailable"))?;
        let cols = backed.col_projection();
        let cols_owned: Vec<u32> = match cols {
            Some(c) => c.to_vec(),
            None => (0..backed.shape_val.1 as u32).collect(),
        };
        let counts = projected_agg::col_nnz_projected_csc(source, &cols_owned)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        return Ok(PyArray::from_vec(py, counts).into_any());
    }
    if let Ok(lazy) = dataset.extract::<PyRef<ScxLazyTransformedDataset>>() {
        let lazy_src = lazy
            .as_column_source()
            .ok_or_else(|| PyRuntimeError::new_err("CSC requested but unavailable"))?;
        let cols = lazy.col_projection();
        let cols_owned: Vec<u32> = match cols {
            Some(c) => c.to_vec(),
            None => (0..lazy.shape_val.1 as u32).collect(),
        };
        let counts = projected_agg::col_nnz_projected_csc(&lazy_src, &cols_owned)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        return Ok(PyArray::from_vec(py, counts).into_any());
    }
    Err(PyRuntimeError::new_err(
        "prefer_format='csc' requires backed or lazy SCX dataset",
    ))
}

/// Streaming column max (axis=0). Returns 1-D float64 array.
#[pyfunction]
#[pyo3(signature = (dataset, prefer_format="csr"))]
pub fn col_max<'py>(
    py: Python<'py>,
    dataset: &Bound<'py, PyAny>,
    prefer_format: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let prefer = validated_prefer(prefer_format)?;
    if prefer == "csc" {
        return col_max_csc_dispatch(py, dataset);
    }
    if let Ok(backed) = dataset.extract::<PyRef<ScxBackedSparseDataset>>() {
        let n_obs = backed.shape_val.0;
        let kept = backed.kept_to_global.as_ref().map(|a| a.as_slice());
        let cols = backed.col_projection();
        let maxes: Vec<f64> = match (cols, kept) {
            (Some(cols), Some(kept)) => {
                projected_agg::col_max_masked_projected(&backed.backed, kept, cols, kept.len())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
            (Some(cols), None) => projected_agg::col_max_projected(&backed.backed, cols, n_obs)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            (None, Some(kept)) => projected_agg::col_max_masked_projected(
                &backed.backed,
                kept,
                &(0..backed.shape_val.1 as u32).collect::<Vec<_>>(),
                kept.len(),
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            (None, None) => projected_agg::col_max_projected(
                &backed.backed,
                &(0..backed.shape_val.1 as u32).collect::<Vec<_>>(),
                n_obs,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        };
        return Ok(PyArray::from_vec(py, maxes).into_any());
    }
    Err(PyRuntimeError::new_err(
        "col_max requires dataset to be ScxBackedSparseDataset",
    ))
}

fn col_max_csc_dispatch<'py>(
    py: Python<'py>,
    dataset: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(backed) = dataset.extract::<PyRef<ScxBackedSparseDataset>>() {
        let source = backed
            .as_column_source()
            .ok_or_else(|| PyRuntimeError::new_err("CSC requested but unavailable"))?;
        let n_obs = backed.shape_val.0;
        let cols_owned: Vec<u32> = match backed.col_projection() {
            Some(c) => c.to_vec(),
            None => (0..backed.shape_val.1 as u32).collect(),
        };
        let res = projected_agg::col_max_projected_csc(source, &cols_owned, n_obs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        return Ok(PyArray::from_vec(py, res).into_any());
    }
    if let Ok(lazy) = dataset.extract::<PyRef<ScxLazyTransformedDataset>>() {
        let lazy_src = lazy
            .as_column_source()
            .ok_or_else(|| PyRuntimeError::new_err("CSC requested but unavailable"))?;
        let n_obs = lazy.shape_val.0;
        let cols_owned: Vec<u32> = match lazy.col_projection() {
            Some(c) => c.to_vec(),
            None => (0..lazy.shape_val.1 as u32).collect(),
        };
        let res = projected_agg::col_max_projected_csc(&lazy_src, &cols_owned, n_obs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        return Ok(PyArray::from_vec(py, res).into_any());
    }
    Err(PyRuntimeError::new_err(
        "prefer_format='csc' requires backed or lazy SCX dataset",
    ))
}

/// Streaming column min (axis=0).
#[pyfunction]
#[pyo3(signature = (dataset, prefer_format="csr"))]
pub fn col_min<'py>(
    py: Python<'py>,
    dataset: &Bound<'py, PyAny>,
    prefer_format: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let prefer = validated_prefer(prefer_format)?;
    if prefer == "csc" {
        return col_min_csc_dispatch(py, dataset);
    }
    if let Ok(backed) = dataset.extract::<PyRef<ScxBackedSparseDataset>>() {
        let n_obs = backed.shape_val.0;
        let kept = backed.kept_to_global.as_ref().map(|a| a.as_slice());
        let cols = backed.col_projection();
        let mins: Vec<f64> = match (cols, kept) {
            (Some(cols), Some(kept)) => {
                projected_agg::col_min_masked_projected(&backed.backed, kept, cols, kept.len())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
            (Some(cols), None) => projected_agg::col_min_projected(&backed.backed, cols, n_obs)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            (None, Some(kept)) => projected_agg::col_min_masked_projected(
                &backed.backed,
                kept,
                &(0..backed.shape_val.1 as u32).collect::<Vec<_>>(),
                kept.len(),
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            (None, None) => projected_agg::col_min_projected(
                &backed.backed,
                &(0..backed.shape_val.1 as u32).collect::<Vec<_>>(),
                n_obs,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        };
        return Ok(PyArray::from_vec(py, mins).into_any());
    }
    Err(PyRuntimeError::new_err(
        "col_min requires dataset to be ScxBackedSparseDataset",
    ))
}

fn col_min_csc_dispatch<'py>(
    py: Python<'py>,
    dataset: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(backed) = dataset.extract::<PyRef<ScxBackedSparseDataset>>() {
        let source = backed
            .as_column_source()
            .ok_or_else(|| PyRuntimeError::new_err("CSC requested but unavailable"))?;
        let n_obs = backed.shape_val.0;
        let cols_owned: Vec<u32> = match backed.col_projection() {
            Some(c) => c.to_vec(),
            None => (0..backed.shape_val.1 as u32).collect(),
        };
        let res = projected_agg::col_min_projected_csc(source, &cols_owned, n_obs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        return Ok(PyArray::from_vec(py, res).into_any());
    }
    if let Ok(lazy) = dataset.extract::<PyRef<ScxLazyTransformedDataset>>() {
        let lazy_src = lazy
            .as_column_source()
            .ok_or_else(|| PyRuntimeError::new_err("CSC requested but unavailable"))?;
        let n_obs = lazy.shape_val.0;
        let cols_owned: Vec<u32> = match lazy.col_projection() {
            Some(c) => c.to_vec(),
            None => (0..lazy.shape_val.1 as u32).collect(),
        };
        let res = projected_agg::col_min_projected_csc(&lazy_src, &cols_owned, n_obs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        return Ok(PyArray::from_vec(py, res).into_any());
    }
    Err(PyRuntimeError::new_err(
        "prefer_format='csc' requires backed or lazy SCX dataset",
    ))
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
    let prefer = validated_prefer(prefer_format)?;
    if prefer == "csc" {
        return col_var_csc_dispatch(py, dataset);
    }
    if let Ok(backed) = dataset.extract::<PyRef<ScxBackedSparseDataset>>() {
        let n_obs = backed.shape_val.0;
        let kept = backed.kept_to_global.as_ref().map(|a| a.as_slice());
        let cols = backed.col_projection();
        let n_vars = backed.shape_val.1;
        let vars: Vec<f64> = match (cols, kept) {
            (Some(cols), Some(kept)) => {
                projected_agg::col_var_masked_projected(&backed.backed, kept, cols)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
            (Some(cols), None) => projected_agg::col_var_projected(&backed.backed, cols, n_obs)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            (None, Some(kept)) => {
                let cols_full: Vec<u32> = (0..n_vars as u32).collect();
                projected_agg::col_var_masked_projected(&backed.backed, kept, &cols_full)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
            (None, None) => {
                let cols_full: Vec<u32> = (0..n_vars as u32).collect();
                projected_agg::col_var_projected(&backed.backed, &cols_full, n_obs)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
        };
        return Ok(PyArray::from_vec(py, vars).into_any());
    }
    Err(PyRuntimeError::new_err(
        "col_var requires dataset to be ScxBackedSparseDataset",
    ))
}

fn col_var_csc_dispatch<'py>(
    py: Python<'py>,
    dataset: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(backed) = dataset.extract::<PyRef<ScxBackedSparseDataset>>() {
        let source = backed
            .as_column_source()
            .ok_or_else(|| PyRuntimeError::new_err("CSC requested but unavailable"))?;
        let n_obs = backed.shape_val.0;
        let cols_owned: Vec<u32> = match backed.col_projection() {
            Some(c) => c.to_vec(),
            None => (0..backed.shape_val.1 as u32).collect(),
        };
        let res = projected_agg::col_var_projected_csc(source, &cols_owned, n_obs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        return Ok(PyArray::from_vec(py, res).into_any());
    }
    if let Ok(lazy) = dataset.extract::<PyRef<ScxLazyTransformedDataset>>() {
        let lazy_src = lazy
            .as_column_source()
            .ok_or_else(|| PyRuntimeError::new_err("CSC requested but unavailable"))?;
        let n_obs = lazy.shape_val.0;
        let cols_owned: Vec<u32> = match lazy.col_projection() {
            Some(c) => c.to_vec(),
            None => (0..lazy.shape_val.1 as u32).collect(),
        };
        let res = projected_agg::col_var_projected_csc(&lazy_src, &cols_owned, n_obs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        return Ok(PyArray::from_vec(py, res).into_any());
    }
    Err(PyRuntimeError::new_err(
        "prefer_format='csc' requires backed or lazy SCX dataset",
    ))
}
