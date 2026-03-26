// ScxBackedSparseDataset — PyO3 class for on-demand sparse access.
//
// Implements the anndata.abc.CSRDataset interface for backed mode.

use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};

use scx_format::BackedCsrReader;

use crate::anndata::csr_to_scipy;

/// PyO3 class wrapping BackedCsrReader for on-demand sparse access.
///
/// The inner BackedCsrReader uses interior mutability (Mutex) for its
/// cache, so this struct needs only a shared Arc — no outer Mutex required.
/// The GIL serializes Python-side access, but the Arc allows the
/// BackedCsrReader to outlive any single PyExperiment reference.
///
/// Note: `format` is not exposed as a PyO3 getter because it is a ClassVar
/// on the Python ABC side (inherited as "csr" from CSRDataset). It should
/// be set as a class attribute in the Python wrapper.
#[pyclass(name = "ScxBackedSparseDataset")]
pub struct ScxBackedSparseDataset {
    backed: Arc<BackedCsrReader>,
    shape_val: (usize, usize),
    n_shards: usize,
    cache_shards: usize,
    /// If deletions are present, maps user-visible row i → global row index.
    /// When None, no remapping is needed (no deletions).
    kept_to_global: Option<Vec<u64>>,
}

impl ScxBackedSparseDataset {
    /// Create a new ScxBackedSparseDataset from a BackedCsrReader.
    pub fn from_reader(backed: Arc<BackedCsrReader>, cache_shards: usize) -> Self {
        let shape_val = backed.shape();
        let n_shards = backed.index().n_shards();
        ScxBackedSparseDataset {
            backed,
            shape_val,
            n_shards,
            cache_shards,
            kept_to_global: None,
        }
    }

    /// Create a new ScxBackedSparseDataset with deletion vector remapping.
    ///
    /// `kept_to_global` maps user-visible row index → global (file-level) row index,
    /// excluding deleted rows. The shape is adjusted to `(kept.len(), n_vars)`.
    pub fn from_reader_with_deletions(
        backed: Arc<BackedCsrReader>,
        cache_shards: usize,
        kept_to_global: Vec<u64>,
    ) -> Self {
        let (_, n_vars) = backed.shape();
        let n_kept = kept_to_global.len();
        let n_shards = backed.index().n_shards();
        ScxBackedSparseDataset {
            backed,
            shape_val: (n_kept, n_vars),
            n_shards,
            cache_shards,
            kept_to_global: Some(kept_to_global),
        }
    }
}

#[pymethods]
impl ScxBackedSparseDataset {
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.shape_val
    }

    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        np.call_method1("dtype", ("float32",))
    }

    /// Returns "csr" to match the anndata ABC format ClassVar.
    #[getter]
    fn format(&self) -> &str {
        "csr"
    }

    #[getter]
    fn backend(&self) -> &str {
        "scx"
    }

    // ndim and __len__ are NOT part of the anndata ABC but are
    // needed for scipy/numpy interop in practice.
    #[getter]
    fn ndim(&self) -> usize {
        2
    }

    fn __len__(&self) -> usize {
        self.shape_val.0
    }

    fn __repr__(&self) -> String {
        format!(
            "ScxBackedSparseDataset(shape=({}, {}), n_shards={}, cache_shards={})",
            self.shape_val.0, self.shape_val.1, self.n_shards, self.cache_shards
        )
    }

    /// Load a slice from disk.
    ///
    /// Supports:
    /// - Row slicing:      X[100:200]       → csr_matrix
    /// - Row + col slice:  X[100:200, :500] → csr_matrix
    /// - Boolean mask:     X[mask]          → csr_matrix
    /// - Fancy indexing:   X[[0, 5, 10]]    → csr_matrix
    /// - Scalar indexing:  X[0, 5]          → float
    /// - Integer index:    X[5]             → csr_matrix (single row)
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Case 1: Tuple index (rows, cols)
        if let Ok(tuple) = index.downcast::<PyTuple>() {
            if tuple.len() == 2 {
                let row_idx = tuple.get_item(0)?;
                let col_idx = tuple.get_item(1)?;
                return self.getitem_2d(py, &row_idx, &col_idx);
            }
            if tuple.len() == 1 {
                let row_idx = tuple.get_item(0)?;
                return self.getitem_rows(py, &row_idx);
            }
            return Err(PyIndexError::new_err(
                "too many indices for 2-dimensional array",
            ));
        }

        // Case 2: Single index (rows only)
        self.getitem_rows(py, index)
    }

    /// Materialize the full matrix into memory.
    /// Required by anndata.abc.CSRDataset.
    fn to_memory<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // When deletion vectors are present, we need to filter the full matrix
        // using the kept_to_global mapping rather than returning all rows.
        if let Some(ref kept) = self.kept_to_global {
            let csr = self
                .backed
                .read_row_indices(kept)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            return csr_to_scipy(py, csr);
        }
        let csr = self
            .backed
            .read_all()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        csr_to_scipy(py, csr)
    }

    // --- Scipy compatibility (NOT part of anndata ABC) ---

    /// Materialize as dense numpy array.
    fn toarray<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method0("toarray")
    }

    /// Materialize as scipy CSR. Same as to_memory().
    fn tocsr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.to_memory(py)
    }

    /// Materialize and convert to CSC.
    fn tocsc<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method0("tocsc")
    }

    /// Dense array property (scipy compat).
    #[getter]
    #[allow(non_snake_case)]
    fn A<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.toarray(py)
    }

    /// Copy — materializes the full matrix. Required by AnnData .copy().
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method0("copy")
    }

    // --- Comparison operators (materialize + delegate to scipy) ---
    // These are used by scanpy's filter_cells (X > 0), filter_genes, etc.

    fn __gt__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method1("__gt__", (other,))
    }

    fn __ge__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method1("__ge__", (other,))
    }

    fn __lt__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method1("__lt__", (other,))
    }

    fn __le__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method1("__le__", (other,))
    }

    fn __eq__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method1("__eq__", (other,))
    }

    fn __ne__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method1("__ne__", (other,))
    }

    // --- Arithmetic operators ---
    // Used by scanpy's normalize_total (multiply), scale, etc.

    fn __add__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.add(other)
    }

    fn __sub__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.sub(other)
    }

    fn __mul__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.mul(other)
    }

    fn __truediv__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method1("__truediv__", (other,))
    }

    fn __matmul__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method1("__matmul__", (other,))
    }

    // --- Aggregation methods ---
    // Used by scanpy's filter_cells (sum per row), HVG (mean/var per column),
    // normalize_total (sum per row), etc.

    #[pyo3(signature = (axis=None))]
    fn sum<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        match axis {
            Some(a) => mat.call_method1("sum", (a,)),
            None => mat.call_method0("sum"),
        }
    }

    #[pyo3(signature = (axis=None))]
    fn mean<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        match axis {
            Some(a) => mat.call_method1("mean", (a,)),
            None => mat.call_method0("mean"),
        }
    }

    #[pyo3(signature = (axis=None))]
    fn getnnz<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        match axis {
            Some(a) => mat.call_method1("getnnz", (a,)),
            None => mat.call_method0("getnnz"),
        }
    }

    /// Element-wise multiply (Hadamard product). Used by normalize_total.
    fn multiply<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method1("multiply", (other,))
    }

    /// Element-wise power. Used by HVG variance computation.
    fn power<'py>(&self, py: Python<'py>, n: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method1("power", (n,))
    }

    /// Number of stored values (nonzeros).
    #[getter]
    fn nnz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.getattr("nnz")
    }

    /// Maximum element along an axis.
    #[pyo3(signature = (axis=None))]
    fn max<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        match axis {
            Some(a) => mat.call_method1("max", (a,)),
            None => mat.call_method0("max"),
        }
    }

    /// Minimum element along an axis.
    #[pyo3(signature = (axis=None))]
    fn min<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        match axis {
            Some(a) => mat.call_method1("min", (a,)),
            None => mat.call_method0("min"),
        }
    }
}

impl ScxBackedSparseDataset {
    /// Handle 1D row indexing (slice, int, bool mask, fancy index).
    fn getitem_rows<'py>(
        &self,
        py: Python<'py>,
        row_idx: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Integer index → single row
        if let Ok(i) = row_idx.extract::<i64>() {
            let row = self.normalize_row_index(i)?;
            let global_row = self.to_global_row(row);
            let csr = self
                .backed
                .read_rows(global_row as u64, global_row as u64 + 1)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            return csr_to_scipy(py, csr);
        }

        // Slice index
        if let Ok(slice) = row_idx.downcast::<PySlice>() {
            let indices = slice.indices(self.shape_val.0 as isize)?;
            let start = indices.start.max(0) as u64;
            let stop = indices.stop.max(0) as u64;
            let step = indices.step;

            if step == 1 && self.kept_to_global.is_none() {
                // Contiguous slice, no deletions — direct range read
                let csr = self
                    .backed
                    .read_rows(start, stop)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                return csr_to_scipy(py, csr);
            }

            // With deletions or non-unit step — expand to individual indices
            let mut rows = Vec::new();
            let mut i = indices.start;
            while (step > 0 && i < indices.stop) || (step < 0 && i > indices.stop) {
                if i >= 0 && (i as usize) < self.shape_val.0 {
                    rows.push(self.to_global_row(i as usize) as u64);
                }
                i += step;
            }

            // If contiguous after remapping (unit step, no deletions was
            // already handled above), use read_row_indices for correctness
            let csr = self
                .backed
                .read_row_indices(&rows)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            return csr_to_scipy(py, csr);
        }

        // Numpy array or list
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (row_idx,))?;
        let dtype_str: String = arr.getattr("dtype")?.call_method0("__str__")?.extract()?;

        if dtype_str == "bool" {
            // Boolean mask → extract True indices
            let nonzero = arr.call_method0("nonzero")?;
            // nonzero returns a tuple of arrays; for 1D, it's (array_of_indices,)
            let idx_tuple = nonzero.downcast::<PyTuple>()?;
            let idx_arr = idx_tuple.get_item(0)?;
            let flat = idx_arr.call_method1("astype", (np.getattr("int64")?,))?;
            let readonly: numpy::PyReadonlyArray1<'_, i64> = flat.extract()?;
            let slice = readonly
                .as_slice()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let rows: Vec<u64> = slice
                .iter()
                .map(|&v| self.to_global_row(v as usize) as u64)
                .collect();
            let csr = self
                .backed
                .read_row_indices(&rows)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            return csr_to_scipy(py, csr);
        }

        // Integer array / list → fancy indexing
        let flat = arr.call_method1("astype", (np.getattr("int64")?,))?;
        let readonly: numpy::PyReadonlyArray1<'_, i64> = flat.extract()?;
        let slice = readonly
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let rows: Vec<u64> = slice
            .iter()
            .map(|&v| {
                let normalized = if v < 0 {
                    self.shape_val.0 as i64 + v
                } else {
                    v
                };
                self.to_global_row(normalized as usize) as u64
            })
            .collect();
        let csr = self
            .backed
            .read_row_indices(&rows)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        csr_to_scipy(py, csr)
    }

    /// Handle 2D indexing (rows, cols).
    fn getitem_2d<'py>(
        &self,
        py: Python<'py>,
        row_idx: &Bound<'py, PyAny>,
        col_idx: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Check for scalar (int, int) → return float
        let row_is_int = row_idx.extract::<i64>().is_ok();
        let col_is_int = col_idx.extract::<i64>().is_ok();

        if row_is_int && col_is_int {
            let row = self.normalize_row_index(row_idx.extract::<i64>()?)?;
            let col = col_idx.extract::<i64>()?;
            let col = if col < 0 {
                (self.shape_val.1 as i64 + col) as usize
            } else {
                col as usize
            };
            if col >= self.shape_val.1 {
                return Err(PyIndexError::new_err(format!(
                    "column index {} out of range for {} columns",
                    col, self.shape_val.1
                )));
            }
            // Read single row, extract single column
            let csr = self
                .backed
                .read_rows(row as u64, row as u64 + 1)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            // Search for the column in the sparse row
            for (i, &idx) in csr.indices.iter().enumerate() {
                if idx as usize == col {
                    return Ok(csr.data[i].into_pyobject(py)?.into_any());
                }
            }
            return Ok(0.0f32.into_pyobject(py)?.into_any());
        }

        // Get the full row selection first
        let row_csr = self.getitem_rows(py, row_idx)?;

        // Check if col_idx is a full slice (`:`) — if so, return as-is
        if let Ok(slice) = col_idx.downcast::<PySlice>() {
            let indices = slice.indices(self.shape_val.1 as isize)?;
            if indices.start == 0 && indices.stop == self.shape_val.1 as isize && indices.step == 1
            {
                return Ok(row_csr);
            }
        }

        // Apply column selection: row_csr[:, col_idx]
        // scipy sparse needs slice(None) for "all rows", not Python None
        let builtins = py.import("builtins")?;
        let slice_none = builtins.call_method1("slice", (py.None(),))?;
        let col_tuple = PyTuple::new(py, &[slice_none.unbind(), col_idx.clone().unbind()])?;
        row_csr.get_item(col_tuple)
    }

    /// Normalize a (possibly negative) row index.
    fn normalize_row_index(&self, i: i64) -> PyResult<usize> {
        let row = if i < 0 {
            (self.shape_val.0 as i64 + i) as usize
        } else {
            i as usize
        };
        if row >= self.shape_val.0 {
            return Err(PyIndexError::new_err(format!(
                "row index {} out of range for {} rows",
                i, self.shape_val.0
            )));
        }
        Ok(row)
    }

    /// Map a user-visible row index to the global (file-level) row index.
    /// If no deletion vectors are present, this is the identity function.
    fn to_global_row(&self, user_row: usize) -> usize {
        match &self.kept_to_global {
            Some(mapping) => mapping[user_row] as usize,
            None => user_row,
        }
    }
}

/// Same as ScxBackedSparseDataset but reads layer shards by name.
///
/// Note: Layers in SCX are stored as separate CSR shard sets with different
/// section names. The BackedCsrReader for a layer is constructed from the
/// layer's catalog entries rather than the X entries.
#[pyclass(name = "ScxBackedLayerDataset")]
pub struct ScxBackedLayerDataset {
    inner: ScxBackedSparseDataset,
    layer_name: String,
}

impl ScxBackedLayerDataset {
    /// Create a new layer dataset wrapping a BackedCsrReader for a specific layer.
    pub fn from_reader(
        backed: Arc<BackedCsrReader>,
        cache_shards: usize,
        layer_name: String,
    ) -> Self {
        let inner = ScxBackedSparseDataset::from_reader(backed, cache_shards);
        ScxBackedLayerDataset { inner, layer_name }
    }

    /// Create a new layer dataset with deletion vector remapping.
    pub fn from_reader_with_deletions(
        backed: Arc<BackedCsrReader>,
        cache_shards: usize,
        layer_name: String,
        kept_to_global: Vec<u64>,
    ) -> Self {
        let inner = ScxBackedSparseDataset::from_reader_with_deletions(
            backed,
            cache_shards,
            kept_to_global,
        );
        ScxBackedLayerDataset { inner, layer_name }
    }
}

#[pymethods]
impl ScxBackedLayerDataset {
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape_val
    }

    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.dtype(py)
    }

    #[getter]
    fn format(&self) -> &str {
        "csr"
    }

    #[getter]
    fn backend(&self) -> &str {
        "scx"
    }

    #[getter]
    fn ndim(&self) -> usize {
        2
    }

    #[getter]
    fn layer_name(&self) -> &str {
        &self.layer_name
    }

    fn __len__(&self) -> usize {
        self.inner.shape_val.0
    }

    fn __repr__(&self) -> String {
        format!(
            "ScxBackedLayerDataset(layer='{}', shape=({}, {}), n_shards={}, cache_shards={})",
            self.layer_name,
            self.inner.shape_val.0,
            self.inner.shape_val.1,
            self.inner.n_shards,
            self.inner.cache_shards
        )
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.__getitem__(py, index)
    }

    fn to_memory<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.to_memory(py)
    }

    fn toarray<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.toarray(py)
    }

    fn tocsr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.tocsr(py)
    }

    fn tocsc<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.tocsc(py)
    }

    #[getter]
    #[allow(non_snake_case)]
    fn A<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.A(py)
    }

    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.copy(py)
    }

    // --- Comparison operators ---

    fn __gt__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.__gt__(py, other)
    }

    fn __ge__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.__ge__(py, other)
    }

    fn __lt__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.__lt__(py, other)
    }

    fn __le__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.__le__(py, other)
    }

    fn __eq__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.__eq__(py, other)
    }

    fn __ne__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.__ne__(py, other)
    }

    // --- Arithmetic operators ---

    fn __add__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.__add__(py, other)
    }

    fn __sub__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.__sub__(py, other)
    }

    fn __mul__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.__mul__(py, other)
    }

    fn __truediv__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.__truediv__(py, other)
    }

    fn __matmul__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.__matmul__(py, other)
    }

    // --- Aggregation methods ---

    #[pyo3(signature = (axis=None))]
    fn sum<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.sum(py, axis)
    }

    #[pyo3(signature = (axis=None))]
    fn mean<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.mean(py, axis)
    }

    #[pyo3(signature = (axis=None))]
    fn getnnz<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.getnnz(py, axis)
    }

    fn multiply<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.multiply(py, other)
    }

    fn power<'py>(&self, py: Python<'py>, n: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.power(py, n)
    }

    #[getter]
    fn nnz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.nnz(py)
    }

    #[pyo3(signature = (axis=None))]
    fn max<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.max(py, axis)
    }

    #[pyo3(signature = (axis=None))]
    fn min<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.min(py, axis)
    }
}
