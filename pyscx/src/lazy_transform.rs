// ScxLazyTransformedDataset — PyO3 class for lazy per-row transforms.
//
// Wraps a BackedCsrReader with a chain of transforms (NormalizeTotal, Log1p,
// RowScale) applied lazily during __getitem__. Implements the same interface
// as ScxBackedSparseDataset so it can replace adata.X transparently.
//
// Peak memory = O(shard_size) — one decoded shard at a time, never the full matrix.

use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};

use scx_format::BackedCsrReader;
use scx_sparse::ScxCsr;

use crate::anndata::csr_to_scipy;
use crate::backed::ScxComparisonResult;

// ---------------------------------------------------------------------------
// Transform enum
// ---------------------------------------------------------------------------

/// Per-shard transform operations applied lazily during __getitem__.
#[derive(Clone, Debug)]
pub enum Transform {
    /// Divide each row by its precomputed sum, multiply by target_sum.
    /// Semantically equivalent to sc.pp.normalize_total().
    NormalizeTotal {
        row_sums: Arc<Vec<f64>>,
        target_sum: f64,
    },

    /// Element-wise ln(x + 1) on non-zero values.
    /// Semantically equivalent to sc.pp.log1p().
    Log1p,

    /// Per-row multiply by a scalar vector.
    /// Used by normalize_total(inplace=False) which returns X * (target_sum / row_sums).
    RowScale { factors: Arc<Vec<f64>> },
}

// ---------------------------------------------------------------------------
// ScxLazyTransformedDataset
// ---------------------------------------------------------------------------

/// PyO3 class wrapping BackedCsrReader with chained per-row transforms.
///
/// Implements the same anndata.abc.CSRDataset-compatible interface as
/// ScxBackedSparseDataset but applies transforms on-read. Transforms are
/// applied in-order to each decoded shard during __getitem__.
#[pyclass(name = "ScxLazyTransformedDataset")]
pub struct ScxLazyTransformedDataset {
    pub(crate) backed: Arc<BackedCsrReader>,
    pub(crate) shape_val: (usize, usize),
    pub(crate) transforms: Vec<Transform>,
    pub(crate) kept_to_global: Option<Vec<u64>>,
    pub(crate) col_projection: Option<Vec<u32>>,
    /// Whether the data is known to be non-negative after transforms.
    /// NormalizeTotal and Log1p preserve non-negativity.
    pub(crate) non_negative: bool,
}

impl ScxLazyTransformedDataset {
    /// Create a new lazy transformed dataset.
    pub fn new(
        backed: Arc<BackedCsrReader>,
        shape_val: (usize, usize),
        kept_to_global: Option<Vec<u64>>,
        col_projection: Option<Vec<u32>>,
        transforms: Vec<Transform>,
    ) -> Self {
        Self {
            backed,
            shape_val,
            transforms,
            kept_to_global,
            col_projection,
            non_negative: true,
        }
    }

    /// Create a `LazyShardSource` for streaming algorithms (PCA).
    ///
    /// Supports column projection: when active, `LazyShardSource` applies
    /// per-shard column filtering and reports `n_vars()` as the projected
    /// column count.
    pub(crate) fn as_shard_source(&self) -> LazyShardSource {
        LazyShardSource {
            backed: Arc::clone(&self.backed),
            transforms: self.transforms.clone(),
            kept_to_global: self.kept_to_global.clone(),
            col_projection: self.col_projection.clone(),
            shape_val: self.shape_val,
        }
    }

    /// Replace the deletion vector, adjusting shape.0.
    pub(crate) fn set_kept_to_global(&mut self, kept: Vec<u64>) {
        self.shape_val.0 = kept.len();
        self.kept_to_global = Some(kept);
    }

    /// Set column projection on this dataset.
    pub(crate) fn set_col_projection(&mut self, col_indices: Vec<u32>) {
        let mut sorted = col_indices;
        sorted.sort_unstable();
        sorted.dedup();
        self.shape_val.1 = sorted.len();
        self.col_projection = Some(sorted);
    }

    /// Read access to col_projection (for composition in filter_genes).
    pub(crate) fn col_projection(&self) -> Option<&[u32]> {
        self.col_projection.as_deref()
    }

    /// Apply all transforms in-place on a decoded CSR shard.
    ///
    /// `global_row_offset` is the starting global row index for this shard,
    /// used to look up per-row parameters (row_sums, factors).
    fn apply_transforms(&self, csr: &mut ScxCsr, global_row_offset: usize) {
        apply_transforms_to_csr(&self.transforms, csr, global_row_offset);
    }

    /// Apply column projection to a CSR matrix if projection is active.
    fn apply_col_projection(&self, csr: ScxCsr) -> ScxCsr {
        match &self.col_projection {
            Some(indices) => scx_engine::projection::project_csr(&csr, indices),
            None => csr,
        }
    }

    /// Map a user-visible row index to global (file-level) row index.
    fn to_global_row(&self, user_row: usize) -> PyResult<usize> {
        match &self.kept_to_global {
            Some(mapping) => mapping.get(user_row).map(|&g| g as usize).ok_or_else(|| {
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
                    Ok(user_row)
                }
            }
        }
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

    /// Filter row-level results through deletion vector remapping.
    pub(crate) fn filter_row_results<T: Copy>(&self, all_values: &[T]) -> Vec<T> {
        match &self.kept_to_global {
            Some(mapping) => mapping.iter().map(|&g| all_values[g as usize]).collect(),
            None => all_values.to_vec(),
        }
    }

    // --- Streaming aggregation through transforms ---

    /// Stream all shards, apply transforms, compute per-row sums.
    pub(crate) fn streaming_row_sums(&self) -> PyResult<Vec<f64>> {
        let n_obs_global = self.backed.shape().0;
        let mut sums = vec![0.0f64; n_obs_global];
        let mut global_row = 0usize;

        for shard_idx in 0..self.backed.index().n_shards() {
            let mut csr = self
                .backed
                .read_shard_uncached(shard_idx)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            self.apply_transforms(&mut csr, global_row);
            for row in 0..csr.n_rows() {
                let s = csr.indptr[row] as usize;
                let e = csr.indptr[row + 1] as usize;
                sums[global_row + row] = csr.data[s..e].iter().map(|&v| v as f64).sum();
            }
            global_row += csr.n_rows();
        }
        Ok(sums)
    }

    /// Stream all shards, apply transforms, compute per-column sums.
    pub(crate) fn streaming_col_sums(&self) -> PyResult<Vec<f64>> {
        let n_vars = self.backed.shape().1;
        let mut sums = vec![0.0f64; n_vars];
        let mut global_row = 0usize;

        for shard_idx in 0..self.backed.index().n_shards() {
            let mut csr = self
                .backed
                .read_shard_uncached(shard_idx)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            self.apply_transforms(&mut csr, global_row);
            for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                sums[col as usize] += val as f64;
            }
            global_row += csr.n_rows();
        }
        Ok(sums)
    }

    /// Stream all shards, apply transforms, compute per-column variance (pop, ddof=0).
    ///
    /// Two-pass: first compute means via col_sums, then accumulate (x-mean)².
    fn streaming_col_var(&self) -> PyResult<Vec<f64>> {
        let n_vars = self.backed.shape().1;
        let n_obs = self.shape_val.0;
        if n_obs == 0 {
            return Ok(vec![0.0f64; n_vars]);
        }

        // Pass 1: column means through transforms
        let col_sums_filtered = if self.kept_to_global.is_some() {
            self.streaming_col_sums_masked()?
        } else {
            self.streaming_col_sums()?
        };
        let col_means: Vec<f64> = col_sums_filtered
            .iter()
            .map(|&s| s / n_obs as f64)
            .collect();

        // Pass 2: accumulate (val - mean)² for stored entries
        let mut sq_devs = vec![0.0f64; n_vars];
        let mut col_nnz = vec![0usize; n_vars];
        let mut global_row = 0usize;

        for shard_idx in 0..self.backed.index().n_shards() {
            let mut csr = self
                .backed
                .read_shard_uncached(shard_idx)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            self.apply_transforms(&mut csr, global_row);

            if let Some(ref kept) = self.kept_to_global {
                let (s_start, s_end) = match self.backed.index().shard_range(shard_idx) {
                    Some(r) => r,
                    None => {
                        global_row += csr.n_rows();
                        continue;
                    }
                };
                let lo = kept.partition_point(|&r| r < s_start);
                let hi = kept.partition_point(|&r| r < s_end);
                for &g_row in &kept[lo..hi] {
                    let local = (g_row - s_start) as usize;
                    let s = csr.indptr[local] as usize;
                    let e = csr.indptr[local + 1] as usize;
                    for j in s..e {
                        let c = csr.indices[j] as usize;
                        let diff = csr.data[j] as f64 - col_means[c];
                        sq_devs[c] += diff * diff;
                        col_nnz[c] += 1;
                    }
                }
            } else {
                for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                    let c = col as usize;
                    let diff = val as f64 - col_means[c];
                    sq_devs[c] += diff * diff;
                    col_nnz[c] += 1;
                }
            }

            global_row += csr.n_rows();
        }

        // Add contribution from implicit zeros
        let mut variances = vec![0.0f64; n_vars];
        for c in 0..n_vars {
            let n_zeros = n_obs - col_nnz[c];
            let total = sq_devs[c] + n_zeros as f64 * col_means[c] * col_means[c];
            variances[c] = total / n_obs as f64;
        }
        Ok(variances)
    }

    /// Stream all shards, apply transforms, compute masked column sums (deletion-aware).
    pub(crate) fn streaming_col_sums_masked(&self) -> PyResult<Vec<f64>> {
        let n_vars = self.backed.shape().1;
        let mut sums = vec![0.0f64; n_vars];
        let mut global_row = 0usize;

        let kept = match &self.kept_to_global {
            Some(k) => k,
            None => return self.streaming_col_sums(),
        };

        for shard_idx in 0..self.backed.index().n_shards() {
            let mut csr = self
                .backed
                .read_shard_uncached(shard_idx)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            self.apply_transforms(&mut csr, global_row);

            let (s_start, s_end) = match self.backed.index().shard_range(shard_idx) {
                Some(r) => r,
                None => {
                    global_row += csr.n_rows();
                    continue;
                }
            };
            let lo = kept.partition_point(|&r| r < s_start);
            let hi = kept.partition_point(|&r| r < s_end);
            for &g_row in &kept[lo..hi] {
                let local = (g_row - s_start) as usize;
                let s = csr.indptr[local] as usize;
                let e = csr.indptr[local + 1] as usize;
                for j in s..e {
                    sums[csr.indices[j] as usize] += csr.data[j] as f64;
                }
            }

            global_row += csr.n_rows();
        }
        Ok(sums)
    }

    /// Materialize the full matrix with all transforms applied.
    ///
    /// Public within the crate so ScxComparisonResult can call it.
    pub(crate) fn materialize_csr(&self) -> PyResult<ScxCsr> {
        let mut global_row = 0usize;
        let mut all_slices = Vec::new();

        for shard_idx in 0..self.backed.index().n_shards() {
            let mut csr = self
                .backed
                .read_shard_uncached(shard_idx)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            self.apply_transforms(&mut csr, global_row);
            global_row += csr.n_rows();
            all_slices.push(csr);
        }

        // Concatenate all shards
        let full = concatenate_csr_vec(&all_slices, self.backed.n_vars());

        // Apply deletion vector filtering
        let full = if let Some(ref kept) = self.kept_to_global {
            extract_rows(&full, kept)
        } else {
            full
        };

        // Apply column projection
        Ok(self.apply_col_projection(full))
    }

    /// Materialize the full matrix as a scipy CSR, callable from other modules.
    pub(crate) fn to_memory_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let csr = self.materialize_csr()?;
        csr_to_scipy(py, csr)
    }
}

#[pymethods]
impl ScxLazyTransformedDataset {
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
        "scx-lazy"
    }

    #[getter]
    fn ndim(&self) -> usize {
        2
    }

    #[getter]
    fn non_negative(&self) -> bool {
        self.non_negative
    }

    fn __len__(&self) -> usize {
        self.shape_val.0
    }

    fn __repr__(&self) -> String {
        let transform_names: Vec<&str> = self
            .transforms
            .iter()
            .map(|t| match t {
                Transform::NormalizeTotal { .. } => "NormalizeTotal",
                Transform::Log1p => "Log1p",
                Transform::RowScale { .. } => "RowScale",
            })
            .collect();
        format!(
            "ScxLazyTransformedDataset(shape=({}, {}), transforms={:?})",
            self.shape_val.0, self.shape_val.1, transform_names
        )
    }

    /// Load a slice from disk, apply transforms, return scipy CSR.
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

    /// Materialize the full matrix into memory with all transforms applied.
    fn to_memory<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.to_memory_py(py)
    }

    /// Copy — materializes the full matrix. Required by AnnData .copy().
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method0("copy")
    }

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

    // --- Comparison operators (lazy, with fused optimization) ---

    fn __gt__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.make_comparison_result(py, "gt", other)
    }

    fn __ge__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.make_comparison_result(py, "ge", other)
    }

    fn __lt__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.make_comparison_result(py, "lt", other)
    }

    fn __le__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.make_comparison_result(py, "le", other)
    }

    fn __eq__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.make_comparison_result(py, "eq", other)
    }

    fn __ne__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.make_comparison_result(py, "ne", other)
    }

    // --- Arithmetic operators ---
    // These materialize since the lazy wrapper is read-only.

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
        // Try to express as a lazy RowScale (multiply by per-row factor).
        // This intercepts scanpy's axis_mul_or_truediv path.
        if let Some(row_factors) =
            crate::backed::try_extract_row_factors(py, other, self.shape_val.0)?
        {
            let global_factors = crate::backed::expand_to_global(
                row_factors,
                self.kept_to_global.as_deref(),
                self.backed.shape().0,
                1.0,
            );
            let mut new_transforms = self.transforms.clone();
            new_transforms.push(Transform::RowScale {
                factors: Arc::new(global_factors),
            });
            let lazy = ScxLazyTransformedDataset::new(
                Arc::clone(&self.backed),
                self.shape_val,
                self.kept_to_global.clone(),
                self.col_projection.clone(),
                new_transforms,
            );
            return Ok(Bound::new(py, lazy)?.into_any());
        }

        // Cannot be expressed as row scaling — fall back to materialization
        let mat = self.to_memory(py)?;
        mat.call_method1("__mul__", (other,))
    }

    fn __truediv__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Try to express as a lazy RowScale (multiply by 1/factor).
        // This intercepts scanpy's axis_mul_or_truediv path.
        if let Some(row_factors) =
            crate::backed::try_extract_row_factors(py, other, self.shape_val.0)?
        {
            let inv_factors: Vec<f64> = row_factors
                .iter()
                .map(|&f| if f != 0.0 { 1.0 / f } else { 0.0 })
                .collect();
            let global_inv = crate::backed::expand_to_global(
                inv_factors,
                self.kept_to_global.as_deref(),
                self.backed.shape().0,
                1.0,
            );
            let mut new_transforms = self.transforms.clone();
            new_transforms.push(Transform::RowScale {
                factors: Arc::new(global_inv),
            });
            let lazy = ScxLazyTransformedDataset::new(
                Arc::clone(&self.backed),
                self.shape_val,
                self.kept_to_global.clone(),
                self.col_projection.clone(),
                new_transforms,
            );
            return Ok(Bound::new(py, lazy)?.into_any());
        }

        // Cannot be expressed as row scaling — fall back to materialization
        let mat = self.to_memory(py)?;
        mat.call_method1("__truediv__", (other,))
    }

    fn __rmul__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Multiplication is commutative — delegate to __mul__.
        self.__mul__(py, other)
    }

    fn __rtruediv__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // other / self — not expressible as a lazy row scale, materialize
        // to dense (scipy sparse doesn't support scalar / sparse).
        let mat = self.to_memory(py)?;
        let dense = mat.call_method0("toarray")?;
        other.div(&dense)
    }

    fn __matmul__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method1("__matmul__", (other,))
    }

    /// Sum along an axis, streaming through transforms.
    ///
    /// Decodes each shard, applies transforms, then aggregates.
    /// Peak memory = O(shard_size), not the full matrix.
    #[pyo3(signature = (axis=None))]
    fn sum<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        match axis {
            Some(0) => {
                let sums = if self.kept_to_global.is_some() {
                    self.streaming_col_sums_masked()?
                } else {
                    self.streaming_col_sums()?
                };
                let arr = numpy::PyArray::from_vec(py, sums);
                arr.call_method1("reshape", ((1i32, self.shape_val.1),))
            }
            Some(1) => {
                let all_sums = self.streaming_row_sums()?;
                let filtered = self.filter_row_results(&all_sums);
                let arr = numpy::PyArray::from_vec(py, filtered);
                arr.call_method1("reshape", ((self.shape_val.0, 1i32),))
            }
            None => {
                let all_sums = self.streaming_row_sums()?;
                let filtered = self.filter_row_results(&all_sums);
                let total: f64 = filtered.iter().sum();
                Ok(total.into_pyobject(py)?.into_any())
            }
            Some(_) => Err(PyRuntimeError::new_err("axis must be 0, 1, or None")),
        }
    }

    /// Mean along an axis, streaming through transforms.
    #[pyo3(signature = (axis=None))]
    fn mean<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        match axis {
            Some(0) => {
                let sums = if self.kept_to_global.is_some() {
                    self.streaming_col_sums_masked()?
                } else {
                    self.streaming_col_sums()?
                };
                let n = self.shape_val.0 as f64;
                let means: Vec<f64> = sums.iter().map(|&s| s / n).collect();
                let arr = numpy::PyArray::from_vec(py, means);
                arr.call_method1("reshape", ((1i32, self.shape_val.1),))
            }
            Some(1) => {
                let all_sums = self.streaming_row_sums()?;
                let filtered = self.filter_row_results(&all_sums);
                let n = self.shape_val.1 as f64;
                let means: Vec<f64> = filtered.iter().map(|&s| s / n).collect();
                let arr = numpy::PyArray::from_vec(py, means);
                arr.call_method1("reshape", ((self.shape_val.0, 1i32),))
            }
            None => {
                let all_sums = self.streaming_row_sums()?;
                let filtered = self.filter_row_results(&all_sums);
                let total: f64 = filtered.iter().sum();
                let n = (self.shape_val.0 as f64) * (self.shape_val.1 as f64);
                Ok((total / n).into_pyobject(py)?.into_any())
            }
            Some(_) => Err(PyRuntimeError::new_err("axis must be 0, 1, or None")),
        }
    }

    /// Variance along an axis, streaming through transforms.
    ///
    /// Two-pass: first computes means, then accumulates (x - mean)².
    #[pyo3(signature = (axis=None))]
    fn var<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        match axis {
            Some(0) => {
                let var = self.streaming_col_var()?;
                let arr = numpy::PyArray::from_vec(py, var);
                arr.call_method1("reshape", ((1i32, self.shape_val.1),))
            }
            Some(1) | None => {
                // For row var and scalar var, materialize since per-row
                // variance through transforms requires careful col-count
                // handling per-row — fallback is simpler and correct.
                let mat = self.to_memory(py)?;
                match axis {
                    Some(1) => {
                        let np = py.import("numpy")?;
                        let mean = mat.call_method1("mean", (1i32,))?;
                        let mean_sq = mat
                            .call_method1("power", (2,))?
                            .call_method1("mean", (1i32,))?;
                        np.call_method1("subtract", (&mean_sq, &mean.call_method1("power", (2,))?))
                    }
                    None => {
                        let np = py.import("numpy")?;
                        let mean = mat.call_method0("mean")?;
                        let mean_sq = mat.call_method1("power", (2,))?.call_method0("mean")?;
                        np.call_method1("subtract", (&mean_sq, &mean.call_method1("power", (2,))?))
                    }
                    _ => unreachable!(),
                }
            }
            Some(_) => Err(PyRuntimeError::new_err("axis must be 0, 1, or None")),
        }
    }

    /// NNZ counts along an axis.
    ///
    /// NNZ is unchanged by NormalizeTotal, Log1p, and RowScale since these
    /// transforms preserve the sparsity pattern (non-zeros stay non-zero,
    /// zeros stay zero). So we delegate to the underlying BackedCsrReader.
    #[pyo3(signature = (axis=None))]
    fn getnnz<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        match axis {
            Some(0) => {
                let counts = match (&self.col_projection, &self.kept_to_global) {
                    (Some(cols), Some(kept)) => {
                        crate::projected_agg::col_nnz_masked_projected(&self.backed, kept, cols)
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    }
                    (Some(cols), None) => {
                        crate::projected_agg::col_nnz_projected(&self.backed, cols)
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                    }
                    (None, Some(kept)) => {
                        let f_counts = self
                            .backed
                            .col_nnz_masked(kept)
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                        f_counts.iter().map(|&v| v as i64).collect::<Vec<i64>>()
                    }
                    (None, None) => self
                        .backed
                        .col_nnz()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                };
                Ok(numpy::PyArray::from_vec(py, counts).into_any())
            }
            Some(1) => {
                let all_nnz = if let Some(ref cols) = self.col_projection {
                    crate::projected_agg::row_nnz_projected(&self.backed, cols)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                } else {
                    self.backed
                        .row_nnz()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                };
                let filtered = self.filter_row_results(&all_nnz);
                Ok(numpy::PyArray::from_vec(py, filtered).into_any())
            }
            None => match (&self.col_projection, &self.kept_to_global) {
                (Some(cols), Some(kept)) => {
                    let nnz =
                        crate::projected_agg::col_nnz_masked_projected(&self.backed, kept, cols)
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    let total: i64 = nnz.iter().sum();
                    Ok((total as usize).into_pyobject(py)?.into_any())
                }
                (Some(cols), None) => {
                    let nnz = crate::projected_agg::col_nnz_projected(&self.backed, cols)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    let total: i64 = nnz.iter().sum();
                    Ok((total as usize).into_pyobject(py)?.into_any())
                }
                (None, Some(kept)) => {
                    let all_nnz = self
                        .backed
                        .row_nnz()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    let total: i64 = kept.iter().map(|&g| all_nnz[g as usize]).sum();
                    Ok((total as usize).into_pyobject(py)?.into_any())
                }
                (None, None) => {
                    let total = self
                        .backed
                        .total_nnz()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    Ok(total.into_pyobject(py)?.into_any())
                }
            },
            Some(_) => Err(PyRuntimeError::new_err("axis must be 0, 1, or None")),
        }
    }

    /// Element-wise multiply (Hadamard product). Materializes.
    fn multiply<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method1("multiply", (other,))
    }

    /// Element-wise power. Materializes.
    fn power<'py>(&self, py: Python<'py>, n: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method1("power", (n,))
    }

    /// Number of stored values (nonzeros) — NNZ preserved by transforms.
    /// Respects deletion vectors and column projections.
    #[getter]
    fn nnz(&self) -> PyResult<usize> {
        match (&self.col_projection, &self.kept_to_global) {
            (Some(cols), Some(kept)) => {
                let nnz = crate::projected_agg::col_nnz_masked_projected(&self.backed, kept, cols)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok(nnz.iter().sum::<i64>() as usize)
            }
            (Some(cols), None) => {
                let nnz = crate::projected_agg::col_nnz_projected(&self.backed, cols)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok(nnz.iter().sum::<i64>() as usize)
            }
            (None, Some(kept)) => {
                let all_nnz = self
                    .backed
                    .row_nnz()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok(kept.iter().map(|&g| all_nnz[g as usize]).sum::<i64>() as usize)
            }
            (None, None) => self
                .backed
                .total_nnz()
                .map_err(|e| PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Number of CSR shards in the backing file.
    #[getter]
    fn n_shards(&self) -> usize {
        self.backed.index().n_shards()
    }

    /// Return shard boundaries as a list of (row_start, row_end) tuples.
    fn shard_boundaries(&self) -> Vec<(usize, usize)> {
        let n_shards = self.backed.index().n_shards();
        match &self.kept_to_global {
            None => (0..n_shards)
                .filter_map(|i| {
                    self.backed
                        .index()
                        .shard_range(i)
                        .map(|(s, e)| (s as usize, e as usize))
                })
                .collect(),
            Some(kept) => {
                let mut boundaries = Vec::with_capacity(n_shards);
                let mut user_row = 0usize;
                for shard_idx in 0..n_shards {
                    let (_s_start, s_end) = match self.backed.index().shard_range(shard_idx) {
                        Some(r) => r,
                        None => continue,
                    };
                    let chunk_start = user_row;
                    while user_row < kept.len() && kept[user_row] < s_end {
                        user_row += 1;
                    }
                    if user_row > chunk_start {
                        boundaries.push((chunk_start, user_row));
                    }
                }
                boundaries
            }
        }
    }

    /// Maximum element along an axis. Materializes (transforms change values).
    #[pyo3(signature = (axis=None))]
    fn max<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        match axis {
            Some(a) => mat.call_method1("max", (a,)),
            None => mat.call_method0("max"),
        }
    }

    /// Minimum element along an axis. Materializes (transforms change values).
    #[pyo3(signature = (axis=None))]
    fn min<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        match axis {
            Some(a) => mat.call_method1("min", (a,)),
            None => mat.call_method0("min"),
        }
    }
}

impl ScxLazyTransformedDataset {
    /// Handle 1D row indexing (slice, int, bool mask, fancy index).
    fn getitem_rows<'py>(
        &self,
        py: Python<'py>,
        row_idx: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Integer index → single row
        if let Ok(i) = row_idx.extract::<i64>() {
            let row = self.normalize_row_index(i)?;
            let global_row = self.to_global_row(row)?;
            let mut csr = self
                .backed
                .read_rows(global_row as u64, global_row as u64 + 1)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            self.apply_transforms(&mut csr, global_row);
            let csr = self.apply_col_projection(csr);
            return csr_to_scipy(py, csr);
        }

        // Slice index
        if let Ok(slice) = row_idx.downcast::<PySlice>() {
            let indices = slice.indices(self.shape_val.0 as isize)?;
            let start = indices.start.max(0) as u64;
            let stop = indices.stop.max(0) as u64;
            let step = indices.step;

            if step == 1 && self.kept_to_global.is_none() {
                // Contiguous slice, no deletions — direct range read + transform
                let mut csr = self
                    .backed
                    .read_rows(start, stop)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                self.apply_transforms(&mut csr, start as usize);
                let csr = self.apply_col_projection(csr);
                return csr_to_scipy(py, csr);
            }

            // With deletions or non-unit step — expand to individual global indices
            let mut rows = Vec::new();
            let mut i = indices.start;
            while (step > 0 && i < indices.stop) || (step < 0 && i > indices.stop) {
                if i >= 0 && (i as usize) < self.shape_val.0 {
                    rows.push(self.to_global_row(i as usize)? as u64);
                }
                i += step;
            }

            let mut csr = self
                .backed
                .read_row_indices(&rows)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            // We need to apply transforms row-by-row with correct global offsets.
            // For fancy indexing, apply transforms per-row.
            self.apply_transforms_per_row(&mut csr, &rows);
            let csr = self.apply_col_projection(csr);
            return csr_to_scipy(py, csr);
        }

        // Numpy array or list
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (row_idx,))?;
        let dtype_str: String = arr.getattr("dtype")?.call_method0("__str__")?.extract()?;

        if dtype_str == "bool" {
            // Boolean mask → extract True indices
            let nonzero = arr.call_method0("nonzero")?;
            let idx_tuple = nonzero.downcast::<PyTuple>()?;
            let idx_arr = idx_tuple.get_item(0)?;
            let flat = idx_arr.call_method1("astype", (np.getattr("int64")?,))?;
            let readonly: numpy::PyReadonlyArray1<'_, i64> = flat.extract()?;
            let slice = readonly
                .as_slice()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let rows: Vec<u64> = slice
                .iter()
                .map(|&v| self.to_global_row(v as usize).map(|g| g as u64))
                .collect::<PyResult<Vec<u64>>>()?;
            let mut csr = self
                .backed
                .read_row_indices(&rows)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            self.apply_transforms_per_row(&mut csr, &rows);
            let csr = self.apply_col_projection(csr);
            return csr_to_scipy(py, csr);
        }

        // Integer array / list → fancy indexing
        let flat = arr.call_method1("astype", (np.getattr("int64")?,))?;
        let readonly: numpy::PyReadonlyArray1<'_, i64> = flat.extract()?;
        let slice = readonly
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let n = self.shape_val.0 as i64;
        let rows: Vec<u64> = slice
            .iter()
            .map(|&v| {
                let normalized = if v < 0 { n + v } else { v };
                if normalized < 0 || normalized >= n {
                    return Err(PyIndexError::new_err(format!(
                        "row index {} out of range for {} rows",
                        v, self.shape_val.0
                    )));
                }
                self.to_global_row(normalized as usize).map(|g| g as u64)
            })
            .collect::<PyResult<Vec<u64>>>()?;
        let mut csr = self
            .backed
            .read_row_indices(&rows)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.apply_transforms_per_row(&mut csr, &rows);
        let csr = self.apply_col_projection(csr);
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
            let global_row = self.to_global_row(row)?;
            let mut csr = self
                .backed
                .read_rows(global_row as u64, global_row as u64 + 1)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            self.apply_transforms(&mut csr, global_row);

            // When col_projection is active, remap user-visible col to on-disk col
            let lookup_col = if let Some(ref proj) = self.col_projection {
                *proj.get(col).ok_or_else(|| {
                    PyIndexError::new_err(format!(
                        "column index {} out of range for {} projected columns",
                        col,
                        proj.len()
                    ))
                })? as usize
            } else {
                col
            };
            for (i, &idx) in csr.indices.iter().enumerate() {
                if idx as usize == lookup_col {
                    return Ok(csr.data[i].into_pyobject(py)?.into_any());
                }
            }
            return Ok(0.0f32.into_pyobject(py)?.into_any());
        }

        // Get the full row selection first
        let row_csr = self.getitem_rows(py, row_idx)?;

        // Check if col_idx is a full slice (`:`)
        if let Ok(slice) = col_idx.downcast::<PySlice>() {
            let indices = slice.indices(self.shape_val.1 as isize)?;
            if indices.start == 0 && indices.stop == self.shape_val.1 as isize && indices.step == 1
            {
                return Ok(row_csr);
            }
        }

        // Apply column selection
        let builtins = py.import("builtins")?;
        let slice_none = builtins.call_method1("slice", (py.None(),))?;
        let col_tuple = PyTuple::new(py, &[slice_none.unbind(), col_idx.clone().unbind()])?;
        row_csr.get_item(col_tuple)
    }

    /// Apply transforms row-by-row for non-contiguous access.
    ///
    /// When rows are fetched via `read_row_indices` (fancy indexing), the resulting
    /// CSR has rows at positions 0..N but they correspond to global rows in `global_rows`.
    /// We need to use the correct global row offset for each row's transform lookup.
    fn apply_transforms_per_row(&self, csr: &mut ScxCsr, global_rows: &[u64]) {
        for (local_row, &global_row) in global_rows.iter().enumerate() {
            let start = csr.indptr[local_row] as usize;
            let end = csr.indptr[local_row + 1] as usize;

            // Detect fused NormalizeTotal + Log1p
            if self.transforms.len() >= 2 {
                if let (
                    Transform::NormalizeTotal {
                        row_sums,
                        target_sum,
                    },
                    Transform::Log1p,
                ) = (&self.transforms[0], &self.transforms[1])
                {
                    let g = global_row as usize;
                    let sum = row_sums[g];
                    if sum > 0.0 {
                        let factor = *target_sum / sum;
                        for v in &mut csr.data[start..end] {
                            *v = ((*v as f64 * factor) as f32).ln_1p();
                        }
                    }
                    // Apply remaining transforms (index 2+)
                    for transform in &self.transforms[2..] {
                        self.apply_single_row_transform(csr, transform, local_row, g);
                    }
                    continue;
                }
            }

            // General path
            for transform in &self.transforms {
                let g = global_row as usize;
                self.apply_single_row_transform(csr, transform, local_row, g);
            }
        }
    }

    /// Apply a single transform to a single row.
    fn apply_single_row_transform(
        &self,
        csr: &mut ScxCsr,
        transform: &Transform,
        local_row: usize,
        global_row: usize,
    ) {
        let start = csr.indptr[local_row] as usize;
        let end = csr.indptr[local_row + 1] as usize;

        match transform {
            Transform::NormalizeTotal {
                row_sums,
                target_sum,
            } => {
                let sum = row_sums[global_row];
                if sum > 0.0 {
                    let factor = *target_sum / sum;
                    for v in &mut csr.data[start..end] {
                        *v = (*v as f64 * factor) as f32;
                    }
                }
            }
            Transform::Log1p => {
                for v in &mut csr.data[start..end] {
                    *v = v.ln_1p();
                }
            }
            Transform::RowScale { factors } => {
                let factor = factors[global_row] as f32;
                for v in &mut csr.data[start..end] {
                    *v *= factor;
                }
            }
        }
    }

    /// Create a lazy comparison result wrapper.
    fn make_comparison_result<'py>(
        &self,
        py: Python<'py>,
        op: &str,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // For lazy comparison, we need a materialized view since transforms
        // change values. However, for the (X > 0).sum() pattern, NNZ is
        // preserved so we can still short-circuit.
        if let Ok(threshold) = other.extract::<f64>() {
            let result = ScxComparisonResult::new_for_lazy(
                Arc::clone(&self.backed),
                self.shape_val,
                op.to_string(),
                threshold,
                self.kept_to_global.clone(),
                self.non_negative,
                self.transforms.clone(),
            );
            Ok(Bound::new(py, result)?.into_any())
        } else {
            // Non-numeric comparison — materialize immediately
            let mat = self.to_memory(py)?;
            let method = match op {
                "gt" => "__gt__",
                "ge" => "__ge__",
                "lt" => "__lt__",
                "le" => "__le__",
                "eq" => "__eq__",
                "ne" => "__ne__",
                _ => "__gt__",
            };
            mat.call_method1(method, (other,))
        }
    }
}

// ---------------------------------------------------------------------------
// Free-standing transform application
// ---------------------------------------------------------------------------

/// Apply all transforms in-place on a decoded CSR shard.
///
/// `global_row_offset` is the starting global row index for this shard,
/// used to look up per-row parameters (row_sums, factors).
///
/// Shared by both `ScxLazyTransformedDataset` and `LazyShardSource`.
fn apply_transforms_to_csr(transforms: &[Transform], csr: &mut ScxCsr, global_row_offset: usize) {
    // Detect fused NormalizeTotal + Log1p pattern for the first two transforms
    if transforms.len() >= 2 {
        if let (
            Transform::NormalizeTotal {
                row_sums,
                target_sum,
            },
            Transform::Log1p,
        ) = (&transforms[0], &transforms[1])
        {
            // Fused path: ln(x * target_sum / row_sum + 1) in one pass
            for row in 0..csr.n_rows() {
                let g = global_row_offset + row;
                let sum = row_sums[g];
                if sum > 0.0 {
                    let factor = *target_sum / sum;
                    let start = csr.indptr[row] as usize;
                    let end = csr.indptr[row + 1] as usize;
                    for v in &mut csr.data[start..end] {
                        *v = ((*v as f64 * factor) as f32).ln_1p();
                    }
                }
            }
            // Apply remaining transforms (index 2+)
            for transform in &transforms[2..] {
                apply_single_transform(csr, transform, global_row_offset);
            }
            return;
        }
    }

    // General path: apply each transform sequentially
    for transform in transforms {
        apply_single_transform(csr, transform, global_row_offset);
    }
}

/// Apply a single transform in-place.
fn apply_single_transform(csr: &mut ScxCsr, transform: &Transform, global_row_offset: usize) {
    match transform {
        Transform::NormalizeTotal {
            row_sums,
            target_sum,
        } => {
            for row in 0..csr.n_rows() {
                let g = global_row_offset + row;
                let sum = row_sums[g];
                if sum > 0.0 {
                    let factor = *target_sum / sum;
                    let start = csr.indptr[row] as usize;
                    let end = csr.indptr[row + 1] as usize;
                    for v in &mut csr.data[start..end] {
                        *v = (*v as f64 * factor) as f32;
                    }
                }
            }
        }
        Transform::Log1p => {
            for v in &mut csr.data {
                *v = v.ln_1p();
            }
        }
        Transform::RowScale { factors } => {
            for row in 0..csr.n_rows() {
                let g = global_row_offset + row;
                let factor = factors[g] as f32;
                let start = csr.indptr[row] as usize;
                let end = csr.indptr[row + 1] as usize;
                for v in &mut csr.data[start..end] {
                    *v *= factor;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LazyShardSource — ShardSource impl for streaming PCA through transforms
// ---------------------------------------------------------------------------

/// Shard source that applies lazy transforms per-shard.
///
/// Enables streaming PCA (and other shard-by-shard algorithms) on
/// lazy-transformed data without materializing the full matrix.
pub(crate) struct LazyShardSource {
    backed: Arc<BackedCsrReader>,
    transforms: Vec<Transform>,
    kept_to_global: Option<Vec<u64>>,
    col_projection: Option<Vec<u32>>,
    shape_val: (usize, usize),
}

impl scx_format::ShardSource for LazyShardSource {
    fn n_shards(&self) -> usize {
        self.backed.index().n_shards()
    }

    fn n_obs(&self) -> usize {
        self.shape_val.0
    }

    fn n_vars(&self) -> usize {
        match &self.col_projection {
            Some(cols) => cols.len(),
            None => self.shape_val.1,
        }
    }

    fn read_shard(&self, shard_idx: usize) -> scx_format::Result<ScxCsr> {
        let (s_start, _) = self.backed.index().shard_range(shard_idx).ok_or_else(|| {
            scx_format::ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: self.backed.index().n_shards(),
            }
        })?;
        let global_row = s_start as usize;

        let mut csr = self.backed.read_shard_uncached(shard_idx)?;
        apply_transforms_to_csr(&self.transforms, &mut csr, global_row);

        // Apply column projection (remap column indices to projected space)
        if let Some(ref cols) = self.col_projection {
            csr = scx_engine::projection::project_csr(&csr, cols);
        }

        // Apply deletion vector: filter to only kept rows within this shard
        if let Some(ref kept) = self.kept_to_global {
            let (s_start, s_end) = self.backed.index().shard_range(shard_idx).unwrap();
            let lo = kept.partition_point(|&r| r < s_start);
            let hi = kept.partition_point(|&r| r < s_end);
            if hi > lo {
                let local_rows: Vec<usize> = kept[lo..hi]
                    .iter()
                    .map(|&g| (g - s_start) as usize)
                    .collect();
                csr = extract_local_rows(&csr, &local_rows);
            } else {
                // No kept rows in this shard — return empty
                let n_projected = self
                    .col_projection
                    .as_ref()
                    .map_or(self.shape_val.1, |c| c.len());
                csr = ScxCsr::new_unchecked((0, n_projected), vec![0], vec![], vec![]);
            }
        }

        Ok(csr)
    }

    // col_means_and_sum_sq: use the default trait impl which iterates
    // read_shard() — transforms and col_projection are applied per-shard.
}

/// Extract specific rows from a CSR by local (within-shard) row indices.
fn extract_local_rows(csr: &ScxCsr, local_rows: &[usize]) -> ScxCsr {
    let n_cols = csr.n_cols();
    let mut indptr = Vec::with_capacity(local_rows.len() + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();

    indptr.push(0i64);
    for &local_row in local_rows {
        let s = csr.indptr[local_row] as usize;
        let e = csr.indptr[local_row + 1] as usize;
        indices.extend_from_slice(&csr.indices[s..e]);
        data.extend_from_slice(&csr.data[s..e]);
        indptr.push(indices.len() as i64);
    }

    ScxCsr::new_unchecked((local_rows.len(), n_cols), indptr, indices, data)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Concatenate multiple ScxCsr slices into one.
fn concatenate_csr_vec(slices: &[ScxCsr], n_cols: usize) -> ScxCsr {
    if slices.is_empty() {
        return ScxCsr::new_unchecked((0, n_cols), vec![0], vec![], vec![]);
    }
    if slices.len() == 1 {
        return slices[0].clone();
    }

    let total_rows: usize = slices.iter().map(|s| s.n_rows()).sum();
    let total_nnz: usize = slices.iter().map(|s| s.nnz()).sum();

    let mut indptr = Vec::with_capacity(total_rows + 1);
    let mut indices = Vec::with_capacity(total_nnz);
    let mut data = Vec::with_capacity(total_nnz);

    indptr.push(0i64);
    let mut offset = 0i64;

    for csr in slices {
        for row in 0..csr.n_rows() {
            let s = csr.indptr[row] as usize;
            let e = csr.indptr[row + 1] as usize;
            indices.extend_from_slice(&csr.indices[s..e]);
            data.extend_from_slice(&csr.data[s..e]);
            offset += (e - s) as i64;
            indptr.push(offset);
        }
    }

    ScxCsr::new_unchecked((total_rows, n_cols), indptr, indices, data)
}

/// Extract specific rows from a CSR by global row indices.
fn extract_rows(csr: &ScxCsr, indices: &[u64]) -> ScxCsr {
    let n_rows = indices.len();
    let n_cols = csr.shape.1;
    let mut indptr = Vec::with_capacity(n_rows + 1);
    let mut new_indices = Vec::new();
    let mut new_data = Vec::new();

    indptr.push(0i64);
    for &g_row in indices {
        let r = g_row as usize;
        let s = csr.indptr[r] as usize;
        let e = csr.indptr[r + 1] as usize;
        new_indices.extend_from_slice(&csr.indices[s..e]);
        new_data.extend_from_slice(&csr.data[s..e]);
        indptr.push(new_indices.len() as i64);
    }

    ScxCsr::new_unchecked((n_rows, n_cols), indptr, new_indices, new_data)
}
