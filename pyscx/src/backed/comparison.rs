// ScxComparisonResult + row-factor scaling helpers.
//
// Extracted from the former pyscx/src/backed.rs (T5.7).

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use scx_format::BackedCsrReader;

use crate::convert::csr_to_scipy;
use crate::lazy_transform::Transform;

/// Lazy comparison result returned by __gt__, __lt__, etc.
///
/// Short-circuits common patterns:
/// - `(X > 0).sum(axis=1)` → `getnnz(axis=1)` (no materialization)
/// - `(X > 0).sum(axis=0)` → `getnnz(axis=0)` (no materialization)
///
/// Falls back to full materialization + scipy for any other operation.
#[pyclass(name = "_ComparisonResult")]
pub struct ScxComparisonResult {
    pub(crate) backed: Arc<BackedCsrReader>,
    pub(crate) shape_val: (usize, usize),
    pub(crate) op: String,
    pub(crate) threshold: f64,
    pub(crate) kept_to_global: Option<Arc<Vec<u64>>>,
    /// Inherited from the parent dataset — gates the getnnz short-circuit.
    pub(crate) non_negative: bool,
    /// When created from ScxLazyTransformedDataset, transforms to apply
    /// before comparison during materialization.
    pub(crate) transforms: Option<Vec<Transform>>,
}

impl ScxComparisonResult {
    /// Create a comparison result from a lazy-transformed source.
    ///
    /// The transforms will be applied during materialization,
    /// but the (X > 0).sum() → getnnz() short-circuit still works
    /// since NormalizeTotal/Log1p/RowScale preserve sparsity patterns.
    pub fn new_for_lazy(
        backed: Arc<BackedCsrReader>,
        shape_val: (usize, usize),
        op: String,
        threshold: f64,
        kept_to_global: Option<Arc<Vec<u64>>>,
        non_negative: bool,
        transforms: Vec<Transform>,
    ) -> Self {
        Self {
            backed,
            shape_val,
            op,
            threshold,
            kept_to_global,
            non_negative,
            transforms: Some(transforms),
        }
    }

    /// Materialize the comparison result as a scipy sparse matrix.
    fn materialize<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if let Some(ref transforms) = self.transforms {
            // Lazy-transform path: stream shards, apply transforms, materialize
            let lazy = crate::lazy_transform::ScxLazyTransformedDataset::new(
                Arc::clone(&self.backed),
                self.shape_val,
                self.kept_to_global.clone(),
                None,
                transforms.clone(),
                self.non_negative,
            );
            let mat = lazy.to_memory_py(py)?;
            let method = match self.op.as_str() {
                "gt" => "__gt__",
                "ge" => "__ge__",
                "lt" => "__lt__",
                "le" => "__le__",
                "eq" => "__eq__",
                "ne" => "__ne__",
                _ => "__gt__",
            };
            return mat.call_method1(method, (self.threshold,));
        }

        // Standard path: read raw data. Decode the full matrix off the GIL (P1).
        let csr = detached(py, || {
            if let Some(ref kept) = self.kept_to_global {
                self.backed
                    .read_row_indices(kept)
                    .map_err(|e| e.to_string())
            } else {
                self.backed.read_all().map_err(|e| e.to_string())
            }
        })
        .map_err(PyRuntimeError::new_err)?;
        let mat = csr_to_scipy(py, csr)?;
        let method = match self.op.as_str() {
            "gt" => "__gt__",
            "ge" => "__ge__",
            "lt" => "__lt__",
            "le" => "__le__",
            "eq" => "__eq__",
            "ne" => "__ne__",
            _ => "__gt__",
        };
        mat.call_method1(method, (self.threshold,))
    }

    /// Check if this comparison can be short-circuited for `.sum()`.
    ///
    /// `(X > 0).sum(axis)` == `getnnz(axis)` for non-negative data.
    /// This is the standard scanpy pattern for QC metrics.
    ///
    /// # Correctness assumption
    ///
    /// This short-circuit is only correct for **non-negative** data (e.g.,
    /// raw UMI counts, normalized counts, log1p-transformed data). For data
    /// that may contain negative values (e.g., after `sc.pp.scale()` which
    /// centers to zero-mean), `getnnz` overcounts because it includes
    /// stored negative entries.
    ///
    /// The `non_negative` flag is inherited from the parent
    /// `ScxBackedSparseDataset` and gates this optimization at runtime.
    fn can_shortcircuit_sum(&self) -> bool {
        self.non_negative && self.op == "gt" && self.threshold == 0.0
    }
}

#[pymethods]
impl ScxComparisonResult {
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.shape_val
    }

    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        np.call_method1("dtype", ("bool",))
    }

    #[getter]
    fn ndim(&self) -> usize {
        2
    }

    fn __repr__(&self) -> String {
        format!(
            "_ComparisonResult(op='{}', threshold={}, shape=({}, {}), shortcircuit={})",
            self.op,
            self.threshold,
            self.shape_val.0,
            self.shape_val.1,
            self.can_shortcircuit_sum()
        )
    }

    /// Sum of comparison result along an axis.
    ///
    /// **Optimization:** For `(X > 0).sum(axis=...)`, this returns
    /// `getnnz(axis=...)` without materializing the full matrix.
    /// This is the critical path for `sc.pp.calculate_qc_metrics`.
    ///
    /// **Note:** The `getnnz` short-circuit assumes non-negative data.
    /// See [`can_shortcircuit_sum`] for details on this assumption.
    /// Accept and ignore extra kwargs (`out`, `keepdims`, `initial`, `where`)
    /// that NumPy passes when dispatching `np.sum()` on this object.
    #[pyo3(signature = (axis=None, dtype=None, out=None, keepdims=false, initial=None, r#where=None))]
    #[allow(unused_variables, clippy::too_many_arguments)]
    fn sum<'py>(
        &self,
        py: Python<'py>,
        axis: Option<i32>,
        dtype: Option<&Bound<'py, PyAny>>,
        out: Option<&Bound<'py, PyAny>>,
        keepdims: bool,
        initial: Option<&Bound<'py, PyAny>>,
        r#where: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if self.can_shortcircuit_sum() {
            // Short-circuit: (X > 0).sum(axis) == getnnz(axis)
            // getnnz counts stored entries, which for non-negative data
            // (UMI counts) is exactly the number of entries > 0.
            match axis {
                Some(0) => {
                    // B5: emit int64 (was uint32) so both axes of this `.sum()`
                    // shortcut agree with each other and with the materialized
                    // `(X>0).sum(axis)` fallback (scipy sums bools to int64).
                    // B2: decode the column nnz off the GIL.
                    let counts: Vec<i64> = detached(py, || {
                        let raw = match &self.kept_to_global {
                            Some(kept) => self
                                .backed
                                .col_nnz_masked(kept)
                                .map(|v| v.into_iter().map(|c| c as i64).collect::<Vec<i64>>()),
                            None => self
                                .backed
                                .col_nnz()
                                .map(|v| v.into_iter().map(|c| c as i64).collect::<Vec<i64>>()),
                        };
                        raw.map_err(|e| e.to_string())
                    })
                    .map_err(PyRuntimeError::new_err)?;
                    let arr = numpy::PyArray::from_vec(py, counts);
                    arr.call_method1("reshape", ((1i32, self.shape_val.1),))
                }
                Some(1) => {
                    // B2: decode row nnz off the GIL; keep-mask filter is cheap.
                    let all_nnz = detached(py, || self.backed.row_nnz().map_err(|e| e.to_string()))
                        .map_err(PyRuntimeError::new_err)?;
                    let filtered = match &self.kept_to_global {
                        Some(mapping) => mapping.iter().map(|&g| all_nnz[g as usize]).collect(),
                        None => all_nnz,
                    };
                    let arr = numpy::PyArray::from_vec(py, filtered);
                    arr.call_method1("reshape", ((self.shape_val.0, 1i32),))
                }
                None => {
                    let total = detached(py, || self.backed.total_nnz().map_err(|e| e.to_string()))
                        .map_err(PyRuntimeError::new_err)?;
                    // Return a numpy int64 scalar (not a bare Python int) so this
                    // shortcut matches the materialized fallback's scalar type.
                    let np = py.import("numpy")?;
                    np.call_method1("int64", (total as i64,))
                }
                Some(_) => Err(PyValueError::new_err("axis must be 0, 1, or None")),
            }
        } else {
            // Fallback: materialize and sum
            let mat = self.materialize(py)?;
            match axis {
                Some(a) => mat.call_method1("sum", (a,)),
                None => mat.call_method0("sum"),
            }
        }
    }

    /// NNZ counts of comparison result.
    #[pyo3(signature = (axis=None))]
    fn getnnz<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.materialize(py)?;
        match axis {
            Some(a) => mat.call_method1("getnnz", (a,)),
            None => mat.call_method0("getnnz"),
        }
    }

    /// Materialize as dense numpy array.
    fn toarray<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.materialize(py)?;
        mat.call_method0("toarray")
    }

    /// Materialize as scipy CSR matrix.
    fn tocsr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.materialize(py)
    }

    /// Materialize as scipy CSC matrix.
    fn tocsc<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.materialize(py)?;
        mat.call_method0("tocsc")
    }

    /// Support indexing on the comparison result.
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.materialize(py)?;
        mat.get_item(index)
    }

    /// Multiply — materializes and delegates.
    fn multiply<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.materialize(py)?;
        mat.call_method1("multiply", (other,))
    }

    /// Mean — materializes and delegates.
    #[pyo3(signature = (axis=None))]
    fn mean<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.materialize(py)?;
        match axis {
            Some(a) => mat.call_method1("mean", (a,)),
            None => mat.call_method0("mean"),
        }
    }

    /// Max — materializes and delegates.
    #[pyo3(signature = (axis=None))]
    fn max<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.materialize(py)?;
        match axis {
            Some(a) => mat.call_method1("max", (a,)),
            None => mat.call_method0("max"),
        }
    }

    /// Min — materializes and delegates.
    #[pyo3(signature = (axis=None))]
    fn min<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.materialize(py)?;
        match axis {
            Some(a) => mat.call_method1("min", (a,)),
            None => mat.call_method0("min"),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Run a pure-Rust closure with the GIL released (B2). Thin standardizing
/// wrapper over [`Python::detach`] for the backed-data aggregation
/// convention: `let raw = detached(py, || reader_op())?; build_numpy(py, raw)`.
/// Every backed/lazy `#[pymethods]` entry point that does shard I/O + decode +
/// aggregation should run that pure-Rust work through here so a multi-GB
/// reduction never blocks other Python threads (e.g. a training-loader
/// consumer). The closure must not touch Python (no `py`, no `PyErr`); map the
/// Rust error to a `PyErr` after it returns.
pub(crate) fn detached<T, E, F>(py: Python<'_>, f: F) -> Result<T, E>
where
    F: FnOnce() -> Result<T, E> + Send,
    T: Send,
    E: Send,
{
    py.detach(f)
}

/// Build an `int32` numpy array of nnz counts. scipy's `getnnz` returns
/// `int32` for both axes, so this is the single source for every
/// nnz-producing path (axis 0/1, masked, projected, comparison) — keeping
/// the dtype uniform and matching scipy. B5.
pub(crate) fn nnz_to_numpy<'py, I: IntoIterator<Item = i64>>(
    py: Python<'py>,
    counts: I,
) -> Bound<'py, numpy::PyArray1<i32>> {
    let v: Vec<i32> = counts
        .into_iter()
        .map(|c| {
            // scipy's getnnz is itself int32, so this matches the contract; but a
            // per-row/col count above i32::MAX (only reachable on a pathological
            // ~2.1B-nnz axis) would wrap silently. Surface it in debug builds and
            // saturate in release rather than emit a negative count.
            debug_assert!(
                c <= i32::MAX as i64,
                "nnz count {c} exceeds i32::MAX; getnnz output would overflow"
            );
            c.min(i32::MAX as i64) as i32
        })
        .collect();
    numpy::PyArray::from_vec(py, v)
}

/// Try to extract a per-row (length `n_obs`) float64 scale vector from a numpy
/// array or scipy matrix, for the row-scaling fast path of `__mul__` /
/// `__truediv__`.
///
/// B3: interception keys off the **pre-ravel** shape and only accepts
/// *unambiguously* row-oriented operands:
///   - `(n_obs,)`     — 1D row factors (`np.ravel(row_sums)`)
///   - `(n_obs, 1)`   — explicit column vector
///
/// A `(1, n)` row vector is a per-*column* broadcast, not a row factor, so it
/// is no longer intercepted. And on a square matrix (`n_obs == n_vars`) a bare
/// 1-D operand of that length is ambiguous (could be a genuine per-gene
/// vector), so it returns `None` → the caller materializes (always correct).
///
/// Returns `Ok(Some(vec))` on an unambiguous row factor of length `n_obs`,
/// `Ok(None)` to fall through to materialization, `Err` only on real Python
/// errors.
pub(crate) fn try_extract_row_factors(
    py: Python<'_>,
    other: &Bound<'_, PyAny>,
    n_obs: usize,
    n_vars: usize,
) -> PyResult<Option<Vec<f64>>> {
    let np = py.import("numpy")?;

    // Convert to numpy array, handling scipy matrices, lists, scalars, etc.
    let arr = match np.call_method1("asarray", (other,)) {
        Ok(a) => a,
        Err(_) => return Ok(None),
    };

    // Inspect the pre-ravel shape to disambiguate orientation.
    let shape: Vec<usize> = match arr.getattr("shape").and_then(|s| s.extract()) {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };
    let is_row_oriented = match shape.as_slice() {
        // 1-D length n_obs: row factor, unless the matrix is square (then the
        // operand could equally be a per-gene vector — ambiguous, materialize).
        [len] => *len == n_obs && n_obs != n_vars,
        // Explicit column vector: unambiguously per-row regardless of squareness.
        [rows, 1] => *rows == n_obs,
        // (1, n) row vectors and everything else are not row factors.
        _ => false,
    };
    if !is_row_oriented {
        return Ok(None);
    }

    // Flatten to 1D
    let flat = match arr.call_method0("ravel") {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };

    // Cast to float64
    let flat_f64 = match flat.call_method1("astype", (np.getattr("float64")?,)) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };

    let readonly: numpy::PyReadonlyArray1<'_, f64> = match flat_f64.extract() {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };
    let slice = readonly
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    if slice.len() == n_obs {
        Ok(Some(slice.to_vec()))
    } else {
        Ok(None)
    }
}

/// Expand a kept-row-space vector to global-row-space.
///
/// When `kept_to_global` is `Some`, the input has length `n_kept` and the
/// output has length `n_obs_global`, with `default` at deleted-row positions.
/// When `kept_to_global` is `None`, returns the input unchanged.
pub(crate) fn expand_to_global(
    kept_values: Vec<f64>,
    kept_to_global: Option<&[u64]>,
    n_obs_global: usize,
    default: f64,
) -> Vec<f64> {
    match kept_to_global {
        Some(mapping) => {
            let mut global = vec![default; n_obs_global];
            for (kept_idx, &global_idx) in mapping.iter().enumerate() {
                global[global_idx as usize] = kept_values[kept_idx];
            }
            global
        }
        None => kept_values,
    }
}

// ─── Phase D.4: ScxBackedMuDataset ──────────────────────────────────────────
