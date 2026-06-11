// ScxBackedLayerDataset — backed access to an AnnData layer.
//
// Extracted from the former pyscx/src/backed.rs (T5.7).

use std::sync::Arc;

use pyo3::prelude::*;

use scx_format::BackedCsrReader;

use super::*;

/// Same as ScxBackedSparseDataset but reads layer shards by name.
///
/// Note: Layers in SCX are stored as separate CSR shard sets with different
/// section names. The BackedCsrReader for a layer is constructed from the
/// layer's catalog entries rather than the X entries.
#[pyclass(name = "ScxBackedLayerDataset")]
pub struct ScxBackedLayerDataset {
    pub(crate) inner: ScxBackedSparseDataset,
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

    fn __rmul__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.__rmul__(py, other)
    }

    fn __rtruediv__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.__rtruediv__(py, other)
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
    fn var<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.var(py, axis)
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
    fn nnz(&self) -> PyResult<usize> {
        self.inner.nnz()
    }

    #[getter]
    fn n_shards(&self) -> usize {
        self.inner.n_shards
    }

    fn shard_boundaries(&self) -> Vec<(usize, usize)> {
        self.inner.shard_boundaries()
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

// ---------------------------------------------------------------------------
// ScxComparisonResult — lazy comparison wrapper for fused optimization
// ---------------------------------------------------------------------------
