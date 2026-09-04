// ScxBackedObsmDataset — backed access to obsm/varm embeddings.
//
// Extracted from the former pyscx/src/backed.rs (T5.7).

use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};

use scx_format_io::BackedDenseReader;

use crate::convert::obsm_batch_to_numpy;

use super::*;

/// Dense analog of [`ScxBackedLayerDataset`] for a single `obsm`
/// embedding. Wraps a [`BackedDenseReader`]; `m[idx]` gathers only the
/// requested rows from the touched `ObsmEmbeddingShard`s, so per-access
/// memory is `O(batch × n_cols)` and independent of `n_obs`.
///
/// Attached to AnnData's `_obsm[key]` (bypassing axis-length validation,
/// like the other backed datasets). Deletion vectors / `obs_filter`
/// compose via `kept_to_global` (user row → global file row).
#[pyclass(name = "ScxBackedObsmDataset")]
pub struct ScxBackedObsmDataset {
    backed: Arc<BackedDenseReader>,
    name: String,
    /// `(n_rows_visible, n_cols)` — visible rows reflect deletions / filter.
    shape_val: (usize, usize),
    cache_shards: usize,
    /// User-visible row i → global file row. `None` = identity.
    kept_to_global: Option<Arc<Vec<u64>>>,
}

impl ScxBackedObsmDataset {
    /// Build over the full mapping (no row remap).
    pub fn from_reader(backed: Arc<BackedDenseReader>, cache_shards: usize, name: String) -> Self {
        let shape_val = backed.shape();
        ScxBackedObsmDataset {
            backed,
            name,
            shape_val,
            cache_shards,
            kept_to_global: None,
        }
    }

    /// Build with a deletion-vector / obs_filter row remap. `shape_val.0`
    /// becomes `kept_to_global.len()`.
    pub fn from_reader_with_deletions(
        backed: Arc<BackedDenseReader>,
        cache_shards: usize,
        name: String,
        kept_to_global: Vec<u64>,
    ) -> Self {
        let (_, n_cols) = backed.shape();
        let n_kept = kept_to_global.len();
        ScxBackedObsmDataset {
            backed,
            name,
            shape_val: (n_kept, n_cols),
            cache_shards,
            kept_to_global: Some(Arc::new(kept_to_global)),
        }
    }

    /// Re-point the row window after an obs-axis subset of the parent AnnData.
    ///
    /// `kept_to_global` is absolute (visible row → global file row), not a
    /// composition against the current window, because the caller derives it
    /// from `X`, which is subset in lockstep. Costs no I/O: this is what lets
    /// an obs subset keep a backed embedding backed instead of gathering it
    /// into a dense array.
    pub(crate) fn set_kept_to_global(&mut self, kept_to_global: Arc<Vec<u64>>) {
        self.shape_val.0 = kept_to_global.len();
        self.kept_to_global = Some(kept_to_global);
    }

    /// A second handle onto the same embedding, same row window.
    pub(crate) fn clone_handle(&self) -> Self {
        ScxBackedObsmDataset {
            backed: Arc::clone(&self.backed),
            name: self.name.clone(),
            shape_val: self.shape_val,
            cache_shards: self.cache_shards,
            kept_to_global: self.kept_to_global.clone(),
        }
    }

    /// A handle onto a sub-window, composing rather than gathering.
    ///
    /// Rows only: an `obsm` / `varm` value is aligned on one axis, so anndata
    /// hands `_subset` a 1-tuple. A column index would have to gather, which
    /// the caller can do explicitly with `m[:, cols]`.
    pub(crate) fn subset_clone(&self, rows: &[i64]) -> PyResult<Self> {
        let mut out = self.clone_handle();
        let composed = crate::axis_align::compose_rows_positional(
            self.kept_to_global.as_ref().map(|v| v.as_slice()),
            rows,
            self.shape_val.0,
        )?;
        out.set_kept_to_global(Arc::new(composed));
        Ok(out)
    }

    fn to_global_row(&self, user_row: usize) -> PyResult<u64> {
        match &self.kept_to_global {
            Some(mapping) => mapping.get(user_row).copied().ok_or_else(|| {
                PyIndexError::new_err(format!(
                    "row index {} out of range for {} rows",
                    user_row,
                    mapping.len()
                ))
            }),
            None => {
                if user_row >= self.shape_val.0 {
                    Err(PyIndexError::new_err(format!(
                        "row index {} out of range for {} rows",
                        user_row, self.shape_val.0
                    )))
                } else {
                    Ok(user_row as u64)
                }
            }
        }
    }

    /// Resolve a row index (int / slice / int-array / bool-mask) into a
    /// list of global file rows, plus `single` = true when the index was
    /// a scalar int (numpy semantics: `m[i]` returns a 1-D row vector).
    fn resolve_rows(
        &self,
        py: Python<'_>,
        row_idx: &Bound<'_, PyAny>,
    ) -> PyResult<(Vec<u64>, bool)> {
        let n = self.shape_val.0 as i64;

        if let Ok(i) = row_idx.extract::<i64>() {
            let norm = if i < 0 { n + i } else { i };
            if norm < 0 || norm >= n {
                return Err(PyIndexError::new_err(format!(
                    "row index {} out of range for {} rows",
                    i, self.shape_val.0
                )));
            }
            return Ok((vec![self.to_global_row(norm as usize)?], true));
        }

        if let Ok(slice) = row_idx.cast::<PySlice>() {
            let n_isize = self.shape_val.0 as isize;
            let indices = slice.indices(n_isize)?;
            let step = indices.step;
            let mut rows = Vec::new();
            let mut i = indices.start;
            while (step > 0 && i < indices.stop) || (step < 0 && i > indices.stop) {
                if i >= 0 && i < n_isize {
                    rows.push(self.to_global_row(i as usize)?);
                }
                i += step;
            }
            return Ok((rows, false));
        }

        // numpy array or list — the one bool-mask / int-array resolver every
        // handle uses (numpy's mask-length rule, negative wrap for signed
        // arrays, unsigned bounds-checked as `uint64`), then the deletion map.
        let visible = super::resolve_row_selector(py, row_idx, self.shape_val.0)?;
        let rows = visible
            .iter()
            .map(|&v| self.to_global_row(v))
            .collect::<PyResult<Vec<u64>>>()?;
        Ok((rows, false))
    }

    /// Gather `rows` and return a 2-D numpy array `(rows.len(), n_cols)`.
    fn gather_rows_2d<'py>(&self, py: Python<'py>, rows: &[u64]) -> PyResult<Bound<'py, PyAny>> {
        // Decode off the GIL (P1); build the numpy array on-GIL.
        let batch = detached(py, || {
            self.backed
                .read_row_indices(rows)
                .map_err(|e| e.to_string())
        })
        .map_err(PyRuntimeError::new_err)?;
        obsm_batch_to_numpy(py, &batch)
    }
}

#[pymethods]
impl ScxBackedObsmDataset {
    /// Cached at construction, so it never reaches `section_bytes` and would
    /// otherwise keep reporting the row count the file had before an `append`
    /// or a `mark_deleted`.
    #[getter]
    fn shape(&self) -> PyResult<(usize, usize)> {
        self.backed.check_fresh().map_err(crate::to_pyerr)?;
        Ok(self.shape_val)
    }

    #[getter]
    fn ndim(&self) -> usize {
        2
    }

    #[getter]
    fn backend(&self) -> &str {
        "scx"
    }

    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use arrow::datatypes::DataType;
        let name = match self.backed.dtype() {
            DataType::Float32 => "float32",
            DataType::Float64 => "float64",
            DataType::Int32 => "int32",
            DataType::Int64 => "int64",
            DataType::Int16 => "int16",
            DataType::Int8 => "int8",
            DataType::UInt32 => "uint32",
            DataType::UInt64 => "uint64",
            // Fallback: let numpy infer from the gathered array's dtype.
            _ => "float32",
        };
        let np = crate::pyimport::import_module(py, "numpy")?;
        np.call_method1("dtype", (name,))
    }

    fn __len__(&self) -> PyResult<usize> {
        self.backed.check_fresh().map_err(crate::to_pyerr)?;
        Ok(self.shape_val.0)
    }

    fn __repr__(&self) -> String {
        format!(
            "ScxBackedObsmDataset(key='{}', shape=({}, {}), cache_shards={})",
            self.name, self.shape_val.0, self.shape_val.1, self.cache_shards
        )
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // 2-D tuple index (rows, cols): gather rows, then apply the column
        // index on axis 1 via numpy on the materialised 2-D array.
        if let Ok(tuple) = index.cast::<PyTuple>() {
            if tuple.len() == 2 {
                let row_idx = tuple.get_item(0)?;
                let col_idx = tuple.get_item(1)?;
                let (rows, single) = self.resolve_rows(py, &row_idx)?;
                let arr2d = self.gather_rows_2d(py, &rows)?;
                let builtins = crate::pyimport::import_module(py, "builtins")?;
                let slice_all = builtins.call_method1("slice", (py.None(),))?;
                // arr2d[:, col_idx]
                let col_indexed =
                    arr2d.get_item(PyTuple::new(py, [slice_all.as_any(), &col_idx])?)?;
                return if single {
                    // Reduce the single gathered row: result[0]
                    col_indexed.get_item(0)
                } else {
                    Ok(col_indexed)
                };
            }
            if tuple.len() == 1 {
                let row_idx = tuple.get_item(0)?;
                return self.__getitem__(py, &row_idx);
            }
            return Err(PyIndexError::new_err(
                "too many indices for 2-dimensional array",
            ));
        }

        let (rows, single) = self.resolve_rows(py, index)?;
        let arr2d = self.gather_rows_2d(py, &rows)?;
        if single {
            // numpy `m[i]` → 1-D row vector of length n_cols.
            arr2d.get_item(0)
        } else {
            Ok(arr2d)
        }
    }

    /// Materialise the full embedding as a dense 2-D numpy array,
    /// applying the row remap if present.
    fn to_memory<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.kept_to_global {
            Some(kept) => self.gather_rows_2d(py, kept),
            None => {
                // Decode off the GIL (P1); consistent with gather_rows_2d.
                let batch = detached(py, || self.backed.read_all().map_err(|e| e.to_string()))
                    .map_err(PyRuntimeError::new_err)?;
                obsm_batch_to_numpy(py, &batch)
            }
        }
    }

    /// Alias for [`Self::to_memory`] — obsm values are already dense.
    fn toarray<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.to_memory(py)
    }

    /// Materialize, matching the other handles' `copy()`.
    ///
    /// `AnnData._mutated_copy` calls `.copy()` on every aligned value it was
    /// not handed, so without this a backed embedding could not survive
    /// `adata[mask].copy()` at all — it raised `AttributeError` before Phase
    /// 4.0b. The in-place accelerators pass `obsm=` explicitly, so they keep
    /// the handle instead of gathering it.
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.to_memory(py)
    }

    /// numpy `__array__` protocol so `np.asarray(m)` materialises.
    ///
    /// numpy >= 2.0 passes a `copy` keyword (`True` / `False` / `None`).
    /// A backed dataset has no in-memory buffer to alias — `to_memory`
    /// always allocates a fresh array — so `copy=False` (which forbids
    /// copying) cannot be honoured and raises `ValueError`, matching
    /// numpy's array-protocol contract (and h5py's backed `__array__`).
    /// `True` / `None` (the `np.asarray` default) return the freshly
    /// materialised array. Accepting the kwarg keeps `np.asarray(m)` /
    /// `np.array(m)` from emitting numpy 2.x's missing-`copy` warning.
    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if copy == Some(false) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "ScxBackedObsmDataset cannot be converted to an array without \
                 copying (copy=False); it materialises a fresh array on access",
            ));
        }
        let arr = self.to_memory(py)?;
        match dtype {
            Some(dt) => arr.call_method1("astype", (dt,)),
            None => Ok(arr),
        }
    }
}
