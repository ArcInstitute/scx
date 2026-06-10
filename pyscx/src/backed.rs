// ScxBackedSparseDataset — PyO3 class for on-demand sparse access.
//
// Implements the anndata.abc.CSRDataset interface for backed mode.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};

use scx_format::{BackedCscReader, BackedCsrReader, BackedDenseReader};

use crate::anndata::{csr_to_scipy, obsm_batch_to_numpy};
use crate::lazy_transform::Transform;
use scx_engine::projection::project_csr;

use crate::projected_agg;

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
    pub(crate) backed: Arc<BackedCsrReader>,
    /// Optional CSC sidecar reader. Populated when the file has CSC
    /// shards AND the open path requested CSC capability. `None` ⇒
    /// `as_column_source()` always returns `None`.
    pub(crate) backed_csc: Option<Arc<BackedCscReader>>,
    pub(crate) shape_val: (usize, usize),
    pub(crate) n_shards: usize,
    pub(crate) cache_shards: usize,
    /// If deletions are present, maps user-visible row i → global row index.
    /// When None, no remapping is needed (no deletions).
    /// Arc-wrapped to avoid O(n) deep clones when creating lazy datasets.
    pub(crate) kept_to_global: Option<Arc<Vec<u64>>>,
    /// If column projection is active, sorted column indices to retain.
    /// CSR outputs are filtered through `project_csr()` before returning.
    /// Arc-wrapped to avoid O(n) deep clones when creating lazy datasets.
    col_projection: Option<Arc<Vec<u32>>>,
    /// Whether the data is known to be non-negative. Defaults to `true`
    /// (raw UMI counts, normalized, log1p). Set to `false` after operations
    /// that produce negative values (e.g., `sc.pp.scale()`), which disables
    /// the `(X > 0).sum() → getnnz()` short-circuit optimization.
    pub(crate) non_negative: bool,
    /// Phase B.5: optional `modality_id` tag for multimodal-aware
    /// callers. `None` = legacy single-modality / global path.
    /// `Some(id)` = the dataset's `backed` CSR reader and (if any)
    /// `backed_csc` reader were constructed with
    /// `BackedCsrReader::for_modality(...)` /
    /// `BackedCscReader::for_modality(...)`. Surfaced via the
    /// `modality_id` Python getter for introspection / tests; not
    /// used internally for routing (the readers are already pinned
    /// at construction time).
    pub(crate) modality_id: Option<u8>,
    /// On-disk source SCX file path, when the wrapper was constructed
    /// from `pyscx.open(path)`. Read by the Phase 8b SCX → SCX writer
    /// (`pyscx.from_anndata` dispatch) so it can open a fresh
    /// `ScxReader` for catalog introspection and byte-passthrough
    /// shard copies. `None` for wrappers built without a known path
    /// (e.g. ad-hoc readers in tests) — in that case the writer falls
    /// back to decode-encode.
    pub(crate) source_path: Option<PathBuf>,
}

impl ScxBackedSparseDataset {
    /// Create a new ScxBackedSparseDataset from a BackedCsrReader.
    pub fn from_reader(backed: Arc<BackedCsrReader>, cache_shards: usize) -> Self {
        let shape_val = backed.shape();
        let n_shards = backed.index().n_shards();
        ScxBackedSparseDataset {
            backed,
            backed_csc: None,
            shape_val,
            n_shards,
            cache_shards,
            kept_to_global: None,
            col_projection: None,
            non_negative: true,
            modality_id: None,
            source_path: None,
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
            backed_csc: None,
            shape_val: (n_kept, n_vars),
            n_shards,
            cache_shards,
            kept_to_global: Some(Arc::new(kept_to_global)),
            col_projection: None,
            non_negative: true,
            modality_id: None,
            source_path: None,
        }
    }

    /// Builder-style setter for the on-disk source path. Populated by
    /// `pyscx.open(path).to_anndata(backed=True)` (and the
    /// per-modality variant) so the Phase 8b `from_anndata` writer
    /// can recover the source path for byte-passthrough copies.
    pub fn with_source_path(&mut self, path: impl Into<PathBuf>) -> &mut Self {
        self.source_path = Some(path.into());
        self
    }

    /// Returns the on-disk source path that this wrapper was built
    /// from, if known. `None` when the wrapper was constructed without
    /// a path (e.g. ad-hoc readers in tests) — callers should fall
    /// back to decode-encode in that case.
    pub fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }

    /// Phase B.5: builder-style setter for the `modality_id` tag.
    /// Used by `ScxBackedMuDataset` to mark per-modality datasets
    /// after constructing them with the per-modality `BackedCsrReader`
    /// / `BackedCscReader` filters.
    pub fn with_modality_id(&mut self, modality_id: u8) -> &mut Self {
        self.modality_id = Some(modality_id);
        self
    }

    /// Attach an optional CSC sidecar reader. After this call,
    /// `as_column_source()` may return `Some` if the gate conditions
    /// are also satisfied. Returns `&mut Self` for builder-style use.
    pub fn with_csc_reader(&mut self, backed_csc: Option<Arc<BackedCscReader>>) -> &mut Self {
        self.backed_csc = backed_csc;
        self
    }

    /// Capability gate: returns `Some(&dyn ColumnShardSource)` iff this
    /// dataset can serve CSC reads. **Single capability-detection point
    /// in the codebase** for `prefer_format="csc"` dispatch.
    ///
    /// Returns `Some` iff:
    /// - `backed_csc` is set (file has a CSC sidecar AND the open path
    ///   requested CSC capability), AND
    /// - `kept_to_global` is `None` (no row deletion vector active —
    ///   CSC indices encode global rows; deletions would require a
    ///   per-shard index remap that the CSC reader doesn't perform).
    ///
    /// Note: column projection is intentionally NOT a hard
    /// disqualifier here. `BackedCscReader::read_csc_columns_subset`
    /// supports gather-style column projection, so consumers that
    /// honor `col_projection()` can still use the CSC path.
    /// However, this base wrapper does not transparently apply
    /// `col_projection` to CSC reads — that is the consumer's
    /// responsibility (or, more typically, lives on
    /// `ScxLazyTransformedDataset` which does apply it). Direct
    /// callers of `as_column_source` on a projected
    /// `ScxBackedSparseDataset` get the full-axis view; reach for
    /// `col_projection()` if you need projected reads.
    pub fn as_column_source(&self) -> Option<&dyn scx_format::ColumnShardSource> {
        if self.kept_to_global.is_some() {
            return None;
        }
        let backed_csc = self.backed_csc.as_ref()?;
        Some(backed_csc.as_ref() as &dyn scx_format::ColumnShardSource)
    }

    /// Set column projection on this dataset.
    /// `col_indices` are the original column indices to retain (will be sorted internally).
    /// Shape is adjusted: n_vars becomes col_indices.len().
    pub fn set_col_projection(&mut self, col_indices: Vec<u32>) {
        let mut sorted = col_indices;
        sorted.sort_unstable();
        sorted.dedup();
        self.shape_val.1 = sorted.len();
        self.col_projection = Some(Arc::new(sorted));
    }

    /// Replace the deletion vector, adjusting shape.0.
    pub(crate) fn set_kept_to_global(&mut self, kept: Vec<u64>) {
        self.shape_val.0 = kept.len();
        self.kept_to_global = Some(Arc::new(kept));
    }

    /// Read access to col_projection (for composition in filter_genes).
    pub(crate) fn col_projection(&self) -> Option<&[u32]> {
        self.col_projection.as_ref().map(|v| v.as_slice())
    }

    /// Clone the col_projection Arc (O(1) ref-count increment).
    pub(crate) fn col_projection_arc(&self) -> Option<Arc<Vec<u32>>> {
        self.col_projection.clone()
    }

    /// Apply column projection to a CSR matrix if projection is active.
    /// Returns the original CSR if no projection is set.
    fn apply_col_projection(&self, csr: scx_sparse::ScxCsr) -> scx_sparse::ScxCsr {
        match &self.col_projection {
            Some(indices) => project_csr(&csr, indices),
            None => csr,
        }
    }
}

#[pymethods]
impl ScxBackedSparseDataset {
    /// Set column projection from Python. Restricts aggregation and access to
    /// a subset of columns. `col_indices` are the original (0-based) column indices.
    /// Shape is adjusted: n_vars becomes len(col_indices).
    #[pyo3(name = "set_col_projection")]
    fn py_set_col_projection(&mut self, col_indices: Vec<u32>) {
        self.set_col_projection(col_indices);
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.shape_val
    }

    /// Phase B.5: optional modality_id tag. `None` for legacy
    /// single-modality / global datasets; `Some(id)` for datasets
    /// constructed via `ScxBackedMuDataset.mod[name]`.
    #[getter]
    fn modality_id(&self) -> Option<u8> {
        self.modality_id
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

    /// Whether the data is known to be non-negative.
    ///
    /// Raw count data is always non-negative. This flag is used by
    /// `ScxComparisonResult` to enable the `(X > 0).sum() → getnnz()`
    /// short-circuit optimization.
    #[getter]
    fn non_negative(&self) -> bool {
        self.non_negative
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
        if let Ok(tuple) = index.cast::<PyTuple>() {
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
        // B2: decode the full matrix off the GIL; build the scipy array after.
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
        let csr = self.apply_col_projection(csr);
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

    // --- Comparison operators (lazy, with fused optimization) ---
    // These return a _ComparisonResult wrapper that short-circuits
    // `.sum()` → `getnnz()` for the `(X > 0).sum(axis=1)` pattern
    // used by scanpy's calculate_qc_metrics, filter_cells, filter_genes.

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
        // Try to extract a per-row scaling vector.  If the multiplier is
        // a 1D or column array with length == n_obs, express as lazy
        // RowScale transform — mirroring __truediv__ (which inverts).
        if let Some(row_factors) =
            try_extract_row_factors(py, other, self.shape_val.0, self.shape_val.1)?
        {
            let global_factors = expand_to_global(
                row_factors,
                self.kept_to_global.as_ref().map(|v| v.as_slice()),
                self.backed.shape().0,
                1.0,
            );
            let lazy = crate::lazy_transform::ScxLazyTransformedDataset::new(
                Arc::clone(&self.backed),
                self.shape_val,
                self.kept_to_global.clone(),
                self.col_projection.clone(),
                vec![Transform::RowScale {
                    factors: Arc::new(global_factors),
                }],
                self.non_negative,
            )
            .with_csc_reader(self.backed_csc.clone())
            .with_source_path(self.source_path.clone());
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
        // Try to extract a per-row scaling vector. If the divisor is a 1D or
        // column array with length == n_obs, we can express the division as a
        // lazy RowScale transform (multiplying by 1/factor) — avoiding full
        // matrix materialization.  This is the code path scanpy's
        // `axis_mul_or_truediv(..., op=truediv)` takes for normalize_total
        // when the numba CSR path isn't available.
        if let Some(row_factors) =
            try_extract_row_factors(py, other, self.shape_val.0, self.shape_val.1)?
        {
            // A zero divisor leaves the row unscaled (1/0 → 0, i.e. multiply by
            // 0 drops the row to empty) — this matches scanpy `normalize_total`,
            // where empty rows stay empty, NOT raw scipy float division (which
            // would yield inf/nan for a nonzero numerator over a zero divisor).
            let inv_factors: Vec<f64> = row_factors
                .iter()
                .map(|&f| if f != 0.0 { 1.0 / f } else { 0.0 })
                .collect();
            let global_inv = expand_to_global(
                inv_factors,
                self.kept_to_global.as_ref().map(|v| v.as_slice()),
                self.backed.shape().0,
                1.0,
            );
            let lazy = crate::lazy_transform::ScxLazyTransformedDataset::new(
                Arc::clone(&self.backed),
                self.shape_val,
                self.kept_to_global.clone(),
                self.col_projection.clone(),
                vec![Transform::RowScale {
                    factors: Arc::new(global_inv),
                }],
                self.non_negative,
            )
            .with_csc_reader(self.backed_csc.clone())
            .with_source_path(self.source_path.clone());
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

    /// Sum along an axis without materializing the full matrix.
    ///
    /// When deletion vectors are present, `axis=0` uses masked column sums
    /// to exclude deleted rows' contributions.
    /// When column projection is active, uses streaming projected aggregation.
    #[pyo3(signature = (axis=None))]
    fn sum<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        match axis {
            Some(0) => {
                // B2: release the GIL for the heavy shard decode + reduction.
                let sums = detached(py, || self.col_sums_raw())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let arr = numpy::PyArray::from_vec(py, sums);
                // Return as (1, n_vars) matrix to match scipy convention
                arr.call_method1("reshape", ((1i32, self.shape_val.1),))
            }
            Some(1) => {
                let all_sums = detached(py, || self.row_sums_raw())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                // Apply deletion vector remapping if present
                let filtered = self.filter_row_results(&all_sums);
                let arr = numpy::PyArray::from_vec(py, filtered);
                // Return as (n_obs, 1) matrix to match scipy convention
                arr.call_method1("reshape", ((self.shape_val.0, 1i32),))
            }
            None => {
                // Total sum — use row sums + filter for correctness with deletions
                let all_sums = detached(py, || self.row_sums_raw())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let filtered = self.filter_row_results(&all_sums);
                let total: f64 = filtered.iter().sum();
                Ok(total.into_pyobject(py)?.into_any())
            }
            Some(_) => Err(PyValueError::new_err("axis must be 0, 1, or None")),
        }
    }

    /// Mean along an axis without materializing the full matrix.
    /// When column projection is active, uses streaming projected aggregation.
    #[pyo3(signature = (axis=None))]
    fn mean<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        match axis {
            Some(0) => {
                let sums = detached(py, || self.col_sums_raw())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let n = self.shape_val.0 as f64;
                let means: Vec<f64> = sums.iter().map(|&s| s / n).collect();
                let arr = numpy::PyArray::from_vec(py, means);
                arr.call_method1("reshape", ((1i32, self.shape_val.1),))
            }
            Some(1) => {
                let all_sums = detached(py, || self.row_sums_raw())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let filtered = self.filter_row_results(&all_sums);
                let n = self.shape_val.1 as f64;
                let means: Vec<f64> = filtered.iter().map(|&s| s / n).collect();
                let arr = numpy::PyArray::from_vec(py, means);
                arr.call_method1("reshape", ((self.shape_val.0, 1i32),))
            }
            None => {
                let all_sums = detached(py, || self.row_sums_raw())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let filtered = self.filter_row_results(&all_sums);
                let total: f64 = filtered.iter().sum();
                let n = (self.shape_val.0 as f64) * (self.shape_val.1 as f64);
                Ok((total / n).into_pyobject(py)?.into_any())
            }
            Some(_) => Err(PyValueError::new_err("axis must be 0, 1, or None")),
        }
    }

    /// Variance along an axis without materializing the full matrix.
    ///
    /// `axis=0`: per-column variance (native Rust, two-pass streaming).
    /// `axis=1`: per-row variance (native Rust, shard-by-shard).
    /// When column projection is active, uses streaming projected aggregation.
    #[pyo3(signature = (axis=None))]
    fn var<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        match axis {
            Some(0) => {
                // B2: heavy decode + per-column two-pass variance off the GIL.
                let var = detached(py, || self.col_var_raw()).map_err(PyRuntimeError::new_err)?;
                let arr = numpy::PyArray::from_vec(py, var);
                arr.call_method1("reshape", ((1i32, self.shape_val.1),))
            }
            Some(1) => {
                // Row var — each row's variance is independent.
                if let Some(ref cols) = self.col_projection {
                    // B6: stream per-row sum/sumsq over the projected columns in
                    // one pass and derive variance, instead of materializing the
                    // projected submatrix via to_memory().
                    let n_proj = cols.len() as f64;
                    let stats = detached(py, || {
                        projected_agg::row_stats_projected(&self.backed, cols)
                    })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    let sums = self.filter_row_results(&stats.sums);
                    let sumsq = self.filter_row_results(&stats.sumsq);
                    let var: Vec<f64> = sums
                        .iter()
                        .zip(&sumsq)
                        .map(|(&s, &sq)| {
                            if n_proj == 0.0 {
                                0.0
                            } else {
                                let mean = s / n_proj;
                                (sq / n_proj - mean * mean).max(0.0)
                            }
                        })
                        .collect();
                    let arr = numpy::PyArray::from_vec(py, var);
                    return arr.call_method1("reshape", ((self.shape_val.0, 1i32),));
                }
                let all_var = detached(py, || self.backed.row_var().map_err(|e| e.to_string()))
                    .map_err(PyRuntimeError::new_err)?;
                let filtered = self.filter_row_results(&all_var);
                let arr = numpy::PyArray::from_vec(py, filtered);
                arr.call_method1("reshape", ((self.shape_val.0, 1i32),))
            }
            None => {
                if let Some(ref cols) = self.col_projection {
                    // B6: scalar variance over projected columns from the same
                    // one-pass row stats (no materialization).
                    let n_proj = cols.len();
                    let stats = detached(py, || {
                        projected_agg::row_stats_projected(&self.backed, cols)
                    })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    let sums = self.filter_row_results(&stats.sums);
                    let sumsq = self.filter_row_results(&stats.sumsq);
                    let n_total = (sums.len() as f64) * (n_proj as f64);
                    if n_total == 0.0 {
                        return Ok(0.0f64.into_pyobject(py)?.into_any());
                    }
                    let mean = sums.iter().sum::<f64>() / n_total;
                    let variance = (sumsq.iter().sum::<f64>() / n_total - mean * mean).max(0.0);
                    return Ok(variance.into_pyobject(py)?.into_any());
                }
                // Total scalar variance via Var(X) = E[X²] - (E[X])².
                // Both reductions stream shard-by-shard, off the GIL.
                let n_obs = self.shape_val.0;
                let n_vars = self.shape_val.1;
                let n_total = (n_obs as f64) * (n_vars as f64);
                if n_total == 0.0 {
                    return Ok(0.0f64.into_pyobject(py)?.into_any());
                }
                let (all_sums, all_sq) = detached(py, || {
                    let s = self.backed.row_sums().map_err(|e| e.to_string())?;
                    let sq = self
                        .backed
                        .row_sum_of_squares()
                        .map_err(|e| e.to_string())?;
                    Ok::<_, String>((s, sq))
                })
                .map_err(PyRuntimeError::new_err)?;
                let mean = self.filter_row_results(&all_sums).iter().sum::<f64>() / n_total;
                let mean_sq = self.filter_row_results(&all_sq).iter().sum::<f64>() / n_total;
                let variance = mean_sq - mean * mean;
                Ok(variance.into_pyobject(py)?.into_any())
            }
            Some(_) => Err(PyValueError::new_err("axis must be 0, 1, or None")),
        }
    }

    /// NNZ counts along an axis without materializing the full matrix.
    /// When column projection is active, uses streaming projected aggregation.
    #[pyo3(signature = (axis=None))]
    fn getnnz<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        match axis {
            Some(0) => {
                let counts = detached(py, || self.col_nnz_raw())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                // B5: scipy `getnnz` returns int32 for both axes — emit int32
                // here too (was uint32) so dtype is uniform across axes.
                Ok(nnz_to_numpy(py, counts.into_iter().map(|c| c as i64)).into_any())
            }
            Some(1) => {
                let all_nnz = detached(py, || self.row_nnz_raw())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                let filtered = self.filter_row_results(&all_nnz);
                // B5: int32 to match axis=0 and scipy (was int64).
                Ok(nnz_to_numpy(py, filtered).into_any())
            }
            None => {
                // B2: scalar total nnz — run the (potentially heavy, for the
                // masked/projected arms) shard decode + reduction off the GIL.
                let total = detached(py, || self.total_nnz_raw())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok(total.into_pyobject(py)?.into_any())
            }
            Some(_) => Err(PyValueError::new_err("axis must be 0, 1, or None")),
        }
    }

    /// Element-wise multiply (Hadamard product). Used by normalize_total.
    fn multiply<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        // If other is also a backed/lazy dataset, materialize it first so scipy can handle it
        let other_mat = if other.is_instance_of::<ScxBackedSparseDataset>()
            || other.is_instance_of::<crate::lazy_transform::ScxLazyTransformedDataset>()
        {
            other.call_method0("to_memory")?
        } else {
            other.clone()
        };
        mat.call_method1("multiply", (other_mat,))
    }

    /// Element-wise power. Used by HVG variance computation.
    fn power<'py>(&self, py: Python<'py>, n: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method1("power", (n,))
    }

    /// Number of stored values (nonzeros) — without materializing.
    /// Respects deletion vectors and column projections.
    #[getter]
    fn nnz(&self) -> PyResult<usize> {
        match (&self.col_projection, &self.kept_to_global) {
            (Some(cols), Some(kept)) => {
                let nnz = projected_agg::col_nnz_masked_projected(&self.backed, kept, cols)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok(nnz.iter().sum::<u32>() as usize)
            }
            (Some(cols), None) => {
                let nnz = projected_agg::col_nnz_projected(&self.backed, cols)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok(nnz.iter().sum::<u32>() as usize)
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
        self.n_shards
    }

    /// Return shard boundaries as a list of (row_start, row_end) tuples.
    ///
    /// When deletion vectors are present, the boundaries are remapped to
    /// user-visible row space (i.e., deleted rows are excluded from counts).
    /// Each tuple represents a contiguous chunk of user-visible rows that
    /// came from one on-disk shard.
    fn shard_boundaries(&self) -> Vec<(usize, usize)> {
        let n_shards = self.n_shards;
        match &self.kept_to_global {
            None => {
                // No deletions — shard boundaries map directly
                (0..n_shards)
                    .filter_map(|i| {
                        self.backed
                            .index()
                            .shard_range(i)
                            .map(|(s, e)| (s as usize, e as usize))
                    })
                    .collect()
            }
            Some(kept) => {
                // With deletions: for each shard, find which user-visible
                // rows fall into that shard's global row range.
                let mut boundaries = Vec::with_capacity(n_shards);
                let mut user_row = 0usize;
                for shard_idx in 0..n_shards {
                    let (_s_start, s_end) = match self.backed.index().shard_range(shard_idx) {
                        Some(r) => r,
                        None => continue,
                    };
                    let chunk_start = user_row;
                    // Count how many kept rows fall in [s_start, s_end)
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

    /// Maximum element along an axis without materializing the full matrix.
    ///
    /// Uses native Rust shard-streaming max. Respects deletion vectors
    /// for both axis=0 (column max) and axis=1 (row max).
    /// When column projection is active, uses streaming projected aggregation.
    #[pyo3(signature = (axis=None))]
    fn max<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        match axis {
            Some(0) => {
                let maxes = detached(py, || self.col_max_raw()).map_err(PyRuntimeError::new_err)?;
                let arr = numpy::PyArray::from_vec(py, maxes);
                arr.call_method1("reshape", ((1i32, self.shape_val.1),))
            }
            Some(1) => {
                if self.col_projection.is_some() {
                    // Row max on projected subset — fall back to to_memory
                    // (whose decode is itself off the GIL).
                    let mat = self.to_memory(py)?;
                    return mat.call_method1("max", (1i32,));
                }
                let all_max = detached(py, || self.backed.row_max().map_err(|e| e.to_string()))
                    .map_err(PyRuntimeError::new_err)?;
                let filtered = self.filter_row_results(&all_max);
                let arr = numpy::PyArray::from_vec(py, filtered);
                arr.call_method1("reshape", ((self.shape_val.0, 1i32),))
            }
            None => {
                // Scalar max — compute from column maxes
                let maxes = detached(py, || self.col_max_raw()).map_err(PyRuntimeError::new_err)?;
                let total_max = maxes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                Ok(total_max.into_pyobject(py)?.into_any())
            }
            Some(_) => Err(PyValueError::new_err("axis must be 0, 1, or None")),
        }
    }

    /// Minimum element along an axis without materializing the full matrix.
    ///
    /// Uses native Rust shard-streaming min. Respects deletion vectors.
    /// When column projection is active, uses streaming projected aggregation.
    #[pyo3(signature = (axis=None))]
    fn min<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        match axis {
            Some(0) => {
                let mins = detached(py, || self.col_min_raw()).map_err(PyRuntimeError::new_err)?;
                let arr = numpy::PyArray::from_vec(py, mins);
                arr.call_method1("reshape", ((1i32, self.shape_val.1),))
            }
            Some(1) => {
                if self.col_projection.is_some() {
                    // Row min on projected subset — fall back to to_memory
                    // (whose decode is itself off the GIL).
                    let mat = self.to_memory(py)?;
                    return mat.call_method1("min", (1i32,));
                }
                let all_min = detached(py, || self.backed.row_min().map_err(|e| e.to_string()))
                    .map_err(PyRuntimeError::new_err)?;
                let filtered = self.filter_row_results(&all_min);
                let arr = numpy::PyArray::from_vec(py, filtered);
                arr.call_method1("reshape", ((self.shape_val.0, 1i32),))
            }
            None => {
                let mins = detached(py, || self.col_min_raw()).map_err(PyRuntimeError::new_err)?;
                let total_min = mins.iter().cloned().fold(f64::INFINITY, f64::min);
                Ok(total_min.into_pyobject(py)?.into_any())
            }
            Some(_) => Err(PyValueError::new_err("axis must be 0, 1, or None")),
        }
    }
}

impl ScxBackedSparseDataset {
    /// Filter row-level results through deletion vector remapping.
    ///
    /// When `kept_to_global` is present, extracts only the values at global
    /// indices corresponding to kept rows. Otherwise returns the input as-is.
    pub(crate) fn filter_row_results<T: Copy>(&self, all_values: &[T]) -> Vec<T> {
        match &self.kept_to_global {
            Some(mapping) => mapping.iter().map(|&g| all_values[g as usize]).collect(),
            None => all_values.to_vec(),
        }
    }

    // B2: pure-Rust aggregation kernels factored out of the `#[pymethods]`
    // entry points so the heavy shard decode can run under `detached(py, ..)`
    // (GIL released). Each routes the projection/mask combination to the right
    // backend call and unifies the (differing) backend error types to `String`
    // — the closure passed to `detached` must not construct a `PyErr`.

    /// Column sums (`axis=0`), honoring column projection and row keep-mask.
    fn col_sums_raw(&self) -> Result<Vec<f64>, String> {
        match (&self.col_projection, &self.kept_to_global) {
            (Some(cols), Some(kept)) => {
                projected_agg::col_sums_masked_projected(&self.backed, kept, cols)
                    .map_err(|e| e.to_string())
            }
            (Some(cols), None) => {
                projected_agg::col_sums_projected(&self.backed, cols).map_err(|e| e.to_string())
            }
            (None, Some(kept)) => self.backed.col_sums_masked(kept).map_err(|e| e.to_string()),
            (None, None) => self.backed.col_sums().map_err(|e| e.to_string()),
        }
    }

    /// Per-row sums (`axis=1` / total), honoring column projection. Deletion
    /// remapping is applied by the caller via `filter_row_results`.
    fn row_sums_raw(&self) -> Result<Vec<f64>, String> {
        if let Some(ref cols) = self.col_projection {
            projected_agg::row_sums_projected(&self.backed, cols).map_err(|e| e.to_string())
        } else {
            self.backed.row_sums().map_err(|e| e.to_string())
        }
    }

    /// Column nnz counts (`axis=0`), honoring column projection and keep-mask.
    fn col_nnz_raw(&self) -> Result<Vec<u32>, String> {
        match (&self.col_projection, &self.kept_to_global) {
            (Some(cols), Some(kept)) => {
                projected_agg::col_nnz_masked_projected(&self.backed, kept, cols)
                    .map_err(|e| e.to_string())
            }
            (Some(cols), None) => {
                projected_agg::col_nnz_projected(&self.backed, cols).map_err(|e| e.to_string())
            }
            (None, Some(kept)) => self
                .backed
                .col_nnz_masked(kept)
                .map(|f| f.iter().map(|&v| v as u32).collect())
                .map_err(|e| e.to_string()),
            (None, None) => self.backed.col_nnz().map_err(|e| e.to_string()),
        }
    }

    /// Per-row nnz counts (`axis=1`), honoring column projection.
    fn row_nnz_raw(&self) -> Result<Vec<i64>, String> {
        if let Some(ref cols) = self.col_projection {
            projected_agg::row_nnz_projected(&self.backed, cols).map_err(|e| e.to_string())
        } else {
            self.backed.row_nnz().map_err(|e| e.to_string())
        }
    }

    /// Scalar total nnz (`getnnz(axis=None)`), honoring column projection and
    /// keep-mask. The masked/projected arms decode shards, so this runs through
    /// `detached` at the call site like the per-axis kernels.
    fn total_nnz_raw(&self) -> Result<usize, String> {
        match (&self.col_projection, &self.kept_to_global) {
            (Some(cols), Some(kept)) => {
                let nnz = projected_agg::col_nnz_masked_projected(&self.backed, kept, cols)
                    .map_err(|e| e.to_string())?;
                Ok(nnz.iter().map(|&v| v as u64).sum::<u64>() as usize)
            }
            (Some(cols), None) => {
                let nnz = projected_agg::col_nnz_projected(&self.backed, cols)
                    .map_err(|e| e.to_string())?;
                Ok(nnz.iter().map(|&v| v as u64).sum::<u64>() as usize)
            }
            (None, Some(kept)) => {
                // Sum row NNZ for kept rows only.
                let all_nnz = self.backed.row_nnz().map_err(|e| e.to_string())?;
                let total: i64 = kept.iter().map(|&g| all_nnz[g as usize]).sum();
                Ok(total as usize)
            }
            (None, None) => self.backed.total_nnz().map_err(|e| e.to_string()),
        }
    }

    /// Column variance (`axis=0`), honoring column projection and keep-mask.
    fn col_var_raw(&self) -> Result<Vec<f64>, String> {
        match (&self.col_projection, &self.kept_to_global) {
            (Some(cols), Some(kept)) => {
                projected_agg::col_var_masked_projected(&self.backed, kept, cols)
                    .map_err(|e| e.to_string())
            }
            (Some(cols), None) => {
                let n_obs = self.backed.shape().0;
                projected_agg::col_var_projected(&self.backed, cols, n_obs)
                    .map_err(|e| e.to_string())
            }
            (None, Some(kept)) => self.backed.col_var_masked(kept).map_err(|e| e.to_string()),
            (None, None) => self.backed.col_var().map_err(|e| e.to_string()),
        }
    }

    /// Column maxima (`axis=0`), honoring column projection and keep-mask.
    fn col_max_raw(&self) -> Result<Vec<f64>, String> {
        match (&self.col_projection, &self.kept_to_global) {
            (Some(cols), Some(kept)) => {
                projected_agg::col_max_masked_projected(&self.backed, kept, cols, kept.len())
                    .map_err(|e| e.to_string())
            }
            (Some(cols), None) => {
                let n_obs = self.backed.shape().0;
                projected_agg::col_max_projected(&self.backed, cols, n_obs)
                    .map_err(|e| e.to_string())
            }
            (None, Some(kept)) => self.backed.col_max_masked(kept).map_err(|e| e.to_string()),
            (None, None) => self.backed.col_max().map_err(|e| e.to_string()),
        }
    }

    /// Column minima (`axis=0`), honoring column projection and keep-mask.
    fn col_min_raw(&self) -> Result<Vec<f64>, String> {
        match (&self.col_projection, &self.kept_to_global) {
            (Some(cols), Some(kept)) => {
                projected_agg::col_min_masked_projected(&self.backed, kept, cols, kept.len())
                    .map_err(|e| e.to_string())
            }
            (Some(cols), None) => {
                let n_obs = self.backed.shape().0;
                projected_agg::col_min_projected(&self.backed, cols, n_obs)
                    .map_err(|e| e.to_string())
            }
            (None, Some(kept)) => self.backed.col_min_masked(kept).map_err(|e| e.to_string()),
            (None, None) => self.backed.col_min().map_err(|e| e.to_string()),
        }
    }

    /// Create a lazy comparison result wrapper.
    ///
    /// If the threshold can be extracted as a numeric f64, returns a
    /// `ScxComparisonResult` that can short-circuit `.sum()` → `getnnz()`.
    /// Otherwise falls back to immediate materialization.
    fn make_comparison_result<'py>(
        &self,
        py: Python<'py>,
        op: &str,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Try to extract threshold as f64
        if let Ok(threshold) = other.extract::<f64>() {
            let result = ScxComparisonResult {
                backed: Arc::clone(&self.backed),
                shape_val: self.shape_val,
                op: op.to_string(),
                threshold,
                kept_to_global: self.kept_to_global.clone(),
                non_negative: self.non_negative,
                transforms: None,
            };
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
            let csr = self
                .backed
                .read_rows(global_row as u64, global_row as u64 + 1)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let csr = self.apply_col_projection(csr);
            return csr_to_scipy(py, csr);
        }

        // Slice index
        if let Ok(slice) = row_idx.cast::<PySlice>() {
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
                let csr = self.apply_col_projection(csr);
                return csr_to_scipy(py, csr);
            }

            // With deletions or non-unit step — expand to individual indices
            let mut rows = Vec::new();
            let mut i = indices.start;
            while (step > 0 && i < indices.stop) || (step < 0 && i > indices.stop) {
                if i >= 0 && (i as usize) < self.shape_val.0 {
                    rows.push(self.to_global_row(i as usize)? as u64);
                }
                i += step;
            }

            // If contiguous after remapping (unit step, no deletions was
            // already handled above), use read_row_indices for correctness
            let csr = self
                .backed
                .read_row_indices(&rows)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
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
            // nonzero returns a tuple of arrays; for 1D, it's (array_of_indices,)
            let idx_tuple = nonzero.cast::<PyTuple>()?;
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
            let csr = self
                .backed
                .read_row_indices(&rows)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
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
        let csr = self
            .backed
            .read_row_indices(&rows)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
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
            // Read single row, extract single column
            let global_row = self.to_global_row(row)?;
            let csr = self
                .backed
                .read_rows(global_row as u64, global_row as u64 + 1)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
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
            // Search for the column in the sparse row
            for (i, &idx) in csr.indices.iter().enumerate() {
                if idx as usize == lookup_col {
                    return Ok(csr.data[i].into_pyobject(py)?.into_any());
                }
            }
            return Ok(0.0f32.into_pyobject(py)?.into_any());
        }

        // ── Non-materializing column projection ────────────────────────
        // When row_idx selects ALL rows (`:` or `slice(None)`) and col_idx
        // is an array or boolean mask, return a new ScxBackedSparseDataset
        // with col_projection set instead of materializing to scipy.
        // This keeps subsequent aggregation (sum, var, etc.) on the f64
        // streaming path and avoids O(n_obs × n_vars) materialization.
        if self.is_all_rows_slice(py, row_idx)? {
            if let Some(col_indices) = self.extract_col_indices(py, col_idx)? {
                let composed = self.compose_col_projection(&col_indices);
                let new_ds = ScxBackedSparseDataset {
                    backed: Arc::clone(&self.backed),
                    backed_csc: self.backed_csc.clone(),
                    shape_val: (self.shape_val.0, composed.len()),
                    n_shards: self.n_shards,
                    cache_shards: self.cache_shards,
                    kept_to_global: self.kept_to_global.clone(),
                    col_projection: Some(Arc::new(composed)),
                    non_negative: self.non_negative,
                    modality_id: self.modality_id,
                    source_path: self.source_path.clone(),
                };
                return Ok(new_ds.into_pyobject(py)?.into_any().unbind().into_bound(py));
            }
        }

        // Get the full row selection first
        let row_csr = self.getitem_rows(py, row_idx)?;

        // Check if col_idx is a full slice (`:`) — if so, return as-is
        if let Ok(slice) = col_idx.cast::<PySlice>() {
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

    /// Check whether `row_idx` selects all rows (is `slice(None)` / `:`).
    fn is_all_rows_slice(&self, _py: Python<'_>, row_idx: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(slice) = row_idx.cast::<PySlice>() {
            let indices = slice.indices(self.shape_val.0 as isize)?;
            Ok(
                indices.start == 0
                    && indices.stop == self.shape_val.0 as isize
                    && indices.step == 1,
            )
        } else {
            Ok(false)
        }
    }

    /// Try to extract integer column indices from `col_idx`.
    /// Returns `Some(Vec<u32>)` for ndarray (int or bool), `None` if not an array
    /// (e.g. a slice or scalar — those fall through to the old path).
    fn extract_col_indices(
        &self,
        py: Python<'_>,
        col_idx: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Vec<u32>>> {
        let numpy = py.import("numpy")?;
        let is_ndarray = col_idx.is_instance(&numpy.getattr("ndarray")?)?;
        if !is_ndarray {
            return Ok(None);
        }

        let dtype_str: String = col_idx.getattr("dtype")?.getattr("kind")?.extract()?;

        match dtype_str.as_str() {
            // Boolean mask → convert to integer indices
            "b" => {
                let mask: Vec<bool> = col_idx.extract()?;
                if mask.len() != self.shape_val.1 {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "boolean index length {} doesn't match axis 1 size {}",
                        mask.len(),
                        self.shape_val.1,
                    )));
                }
                let indices: Vec<u32> = mask
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &b)| if b { Some(i as u32) } else { None })
                    .collect();
                Ok(Some(indices))
            }
            // Integer array (signed or unsigned).
            // Only use non-materializing projection for sorted, unique indices.
            // Unsorted or duplicate indices need materialization to preserve
            // user-specified column order/repetition (numpy __getitem__ semantics).
            "i" | "u" => {
                let indices: Vec<i64> = col_idx.extract()?;
                let n = self.shape_val.1 as i64;
                let resolved: Vec<u32> = indices
                    .iter()
                    .map(|&i| {
                        let i = if i < 0 { n + i } else { i };
                        if i < 0 || i >= n {
                            Err(pyo3::exceptions::PyIndexError::new_err(format!(
                                "column index {} out of range for axis of size {}",
                                i, n,
                            )))
                        } else {
                            Ok(i as u32)
                        }
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                if !resolved.windows(2).all(|w| w[0] < w[1]) {
                    return Ok(None); // unsorted or duplicates → materialize
                }
                Ok(Some(resolved))
            }
            _ => Ok(None),
        }
    }

    /// Compose new column indices with an existing col_projection.
    ///
    /// `new_indices` are in the user-visible column space (`0..shape_val.1`).
    /// Returns sorted, deduplicated indices in the original on-disk column space
    /// (matching the convention that `col_projection` is always sorted).
    ///
    /// The output is always in on-disk column order regardless of input order,
    /// and duplicate indices in `new_indices` are silently collapsed.
    fn compose_col_projection(&self, new_indices: &[u32]) -> Vec<u32> {
        let mut composed = match &self.col_projection {
            Some(existing) => {
                // new_indices are relative to visible columns; map through existing
                new_indices.iter().map(|&i| existing[i as usize]).collect()
            }
            None => new_indices.to_vec(),
        };
        composed.sort_unstable();
        composed.dedup();
        composed
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
    ///
    /// Returns `PyIndexError` if `user_row` is out of bounds.
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
}

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

/// Lazy comparison result returned by __gt__, __lt__, etc.
///
/// Short-circuits common patterns:
/// - `(X > 0).sum(axis=1)` → `getnnz(axis=1)` (no materialization)
/// - `(X > 0).sum(axis=0)` → `getnnz(axis=0)` (no materialization)
///
/// Falls back to full materialization + scipy for any other operation.
#[pyclass(name = "_ComparisonResult")]
pub struct ScxComparisonResult {
    backed: Arc<BackedCsrReader>,
    shape_val: (usize, usize),
    op: String,
    threshold: f64,
    kept_to_global: Option<Arc<Vec<u64>>>,
    /// Inherited from the parent dataset — gates the getnnz short-circuit.
    non_negative: bool,
    /// When created from ScxLazyTransformedDataset, transforms to apply
    /// before comparison during materialization.
    transforms: Option<Vec<Transform>>,
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

        // Standard path: read raw data
        let csr = if let Some(ref kept) = self.kept_to_global {
            self.backed
                .read_row_indices(kept)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        } else {
            self.backed
                .read_all()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        };
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

use scx_format::ScxReader;
use std::sync::Mutex as StdMutex;

/// Phase D.4: backed wrapper for multimodal SCX files. Holds an
/// `Arc<ScxReader>` and lazily exposes per-modality
/// `ScxBackedSparseDataset`s via the dict-like `.mod` attribute.
///
/// Construction is opt-in: `pyscx.open(path)` keeps returning
/// `PyExperiment` for backward compat. Users who want the lazy
/// per-modality surface call `pyscx.ScxBackedMuDataset(path)`
/// directly.
///
/// ## Surface
///
/// - `mu.is_multimodal` / `mu.n_modalities` / `mu.modality_names`
/// - `mu.modality_id(name) -> Optional[int]`
/// - `mu.modality_info(modality_id) -> Optional[dict]`
/// - `mu.obs` (cached pandas DataFrame; computed on first access)
/// - `mu.mod` (dict-like proxy; `mu.mod["rna"]` lazily yields a
///   `ScxBackedSparseDataset` pinned to that modality)
/// - `mu.to_mudata()` — eager round-trip to a real `mudata.MuData`
///   for users who want the materialised view.
#[pyclass(name = "ScxBackedMuDataset")]
pub struct ScxBackedMuDataset {
    /// Path to the SCX file. Re-opened per `.mod[name]` access since
    /// `BackedCsrReader::for_modality` / `BackedCscReader::for_modality`
    /// consume the `ScxReader`. Reader open is mmap-cheap.
    path: std::path::PathBuf,
    /// Cached metadata reader for `is_multimodal` / `modality_*` /
    /// `obs` accessors.
    meta_reader: Arc<ScxReader>,
    /// Cached global obs as a pandas DataFrame. Computed on first
    /// `obs` access; reused thereafter.
    cached_obs: StdMutex<Option<Py<PyAny>>>,
    /// CSR cache size to use when constructing per-modality
    /// `BackedCsrReader`s. Defaults to 4 (matches the experiment
    /// default).
    cache_shards: usize,
}

#[pymethods]
impl ScxBackedMuDataset {
    /// Open a multimodal SCX file as a backed wrapper. Raises if the
    /// file is single-modality (use `pyscx.open(path)` for those).
    #[new]
    #[pyo3(signature = (path, cache_shards=None))]
    fn new(path: &str, cache_shards: Option<usize>) -> PyResult<Self> {
        let path_buf = std::path::PathBuf::from(path);
        let reader = ScxReader::open(&path_buf)
            .map_err(|e| PyRuntimeError::new_err(format!("ScxReader::open: {e}")))?;
        if !reader.is_multimodal() {
            return Err(PyRuntimeError::new_err(format!(
                "ScxBackedMuDataset: file '{path}' is single-modality. \
                 Use pyscx.open(path) for those."
            )));
        }
        Ok(ScxBackedMuDataset {
            path: path_buf,
            meta_reader: Arc::new(reader),
            cached_obs: StdMutex::new(None),
            cache_shards: cache_shards.unwrap_or(4),
        })
    }

    #[getter]
    fn is_multimodal(&self) -> bool {
        true
    }

    #[getter]
    fn n_modalities(&self) -> u32 {
        self.meta_reader.n_modalities()
    }

    #[getter]
    fn modality_names(&self) -> Vec<String> {
        self.meta_reader
            .modality_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn modality_id(&self, name: &str) -> Option<u8> {
        self.meta_reader.modality_id(name)
    }

    fn modality_info<'py>(
        &self,
        py: Python<'py>,
        modality_id: u8,
    ) -> PyResult<Option<Bound<'py, pyo3::types::PyDict>>> {
        use pyo3::types::PyDict;
        let info = match self.meta_reader.modality_info(modality_id) {
            Some(i) => i,
            None => return Ok(None),
        };
        let d = PyDict::new(py);
        d.set_item("name", &info.name)?;
        d.set_item("modality_type", info.modality_type as u8)?;
        d.set_item("default_codec_id", info.default_codec_id)?;
        d.set_item("default_value_encoding", info.default_value_encoding)?;
        d.set_item("n_vars", info.n_vars)?;
        d.set_item("nnz", info.nnz)?;
        d.set_item("n_csr_shards", info.n_csr_shards)?;
        d.set_item("n_csc_shards", info.n_csc_shards)?;
        d.set_item("flags", info.flags.bits())?;
        Ok(Some(d))
    }

    /// Lazy global obs. First call materialises the obs as a pandas
    /// DataFrame; subsequent calls return the same Python object.
    #[getter]
    fn obs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Fast path: cached obs already constructed.
        {
            let guard = self.cached_obs.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(ref obj) = *guard {
                return Ok(obj.clone_ref(py));
            }
        }
        // Slow path: read obs, convert to pandas, cache.
        let batch = self
            .meta_reader
            .read_obs()
            .map_err(|e| PyRuntimeError::new_err(format!("read_obs: {e}")))?;
        let table = crate::anndata::record_batch_to_pyarrow(py, &batch)?;
        let df = crate::anndata::pyarrow_table_to_pandas(&table)?;
        let obj: Py<PyAny> = df.unbind();
        let mut guard = self.cached_obs.lock().unwrap_or_else(|e| e.into_inner());
        *guard = Some(obj.clone_ref(py));
        Ok(obj)
    }

    /// Dict-like `.mod` accessor. Returns a `ScxBackedMuModality`
    /// proxy whose `__getitem__(name)` lazily yields a fresh
    /// `ScxBackedSparseDataset` pinned to that modality.
    #[getter]
    fn r#mod(&self, py: Python<'_>) -> PyResult<Py<ScxBackedMuModality>> {
        Py::new(
            py,
            ScxBackedMuModality {
                path: self.path.clone(),
                modality_names: self.modality_names(),
                cache_shards: self.cache_shards,
            },
        )
    }

    /// Eagerly materialise as a real `mudata.MuData`. Convenience
    /// shortcut for users who want the materialised view.
    fn to_mudata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::mudata::to_mudata(py, &self.meta_reader)
    }

    fn __repr__(&self) -> String {
        let names: Vec<String> = self.modality_names();
        format!(
            "ScxBackedMuDataset(n_obs={}, modalities={:?})",
            self.meta_reader.n_obs(),
            names,
        )
    }
}

/// Phase D.4: dict-like proxy returned from
/// `ScxBackedMuDataset.mod`. Each `__getitem__(name)` constructs a
/// fresh `BackedCsrReader::for_modality` (and matching
/// `BackedCscReader::for_modality` if the modality has CSC sidecars)
/// against a freshly-opened `ScxReader`, then wraps both in a
/// `ScxBackedSparseDataset` tagged with `modality_id = id`.
///
/// Reader open is mmap-cheap so per-modality reopens are negligible
/// vs the alternative of cloning a single shared `ScxReader` (which
/// the existing constructors don't support).
#[pyclass(name = "ScxBackedMuModality")]
pub struct ScxBackedMuModality {
    path: std::path::PathBuf,
    modality_names: Vec<String>,
    cache_shards: usize,
}

#[pymethods]
impl ScxBackedMuModality {
    fn __getitem__(&self, py: Python<'_>, name: &str) -> PyResult<Py<ScxBackedSparseDataset>> {
        // Resolve name → modality_id via a cheap meta reader.
        let meta = ScxReader::open(&self.path)
            .map_err(|e| PyRuntimeError::new_err(format!("ScxReader::open: {e}")))?;
        let modality_id = meta.modality_id(name).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "modality '{name}' not found (available: {:?})",
                self.modality_names
            ))
        })?;
        let has_csc = meta
            .modality_info(modality_id)
            .map(|i| i.flags.has_csc())
            .unwrap_or(false);
        drop(meta);

        // Build per-modality CSR reader.
        let csr_reader = ScxReader::open(&self.path)
            .map_err(|e| PyRuntimeError::new_err(format!("ScxReader::open: {e}")))?;
        let backed_csr = Arc::new(BackedCsrReader::for_modality(
            csr_reader,
            modality_id,
            self.cache_shards,
        ));

        // Optional per-modality CSC reader.
        let backed_csc = if has_csc {
            let csc_reader = ScxReader::open(&self.path)
                .map_err(|e| PyRuntimeError::new_err(format!("ScxReader::open: {e}")))?;
            let r = BackedCscReader::for_modality(csc_reader, modality_id, self.cache_shards)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("BackedCscReader::for_modality: {e}"))
                })?;
            Some(Arc::new(r))
        } else {
            None
        };

        let mut ds = ScxBackedSparseDataset::from_reader(backed_csr, self.cache_shards);
        ds.with_csc_reader(backed_csc).with_modality_id(modality_id);
        ds.with_source_path(&self.path);
        Py::new(py, ds)
    }

    fn __contains__(&self, name: &str) -> bool {
        self.modality_names.iter().any(|n| n == name)
    }

    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Yield modality names. Easiest cross-version: convert to a
        // PyList and call its __iter__.
        let list = pyo3::types::PyList::new(py, &slf.modality_names)?;
        let iter = list.try_iter()?;
        Ok(iter.into_pyobject(py)?.into_any().unbind())
    }

    fn __len__(&self) -> usize {
        self.modality_names.len()
    }

    fn keys(&self) -> Vec<String> {
        self.modality_names.clone()
    }

    fn __repr__(&self) -> String {
        format!("ScxBackedMuModality(modalities={:?})", self.modality_names)
    }
}

// Suppress unused-import warning on `Path` when no other code in this
// file references it (we only need it transitively for path conversion).
#[allow(dead_code)]
fn _path_marker(_p: &Path) {}

// ---------------------------------------------------------------------------
// ScxBackedObsmDataset — backed dense row-gather for an obsm embedding
// ---------------------------------------------------------------------------

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

        // numpy array or list
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (row_idx,))?;
        // Detect boolean masks via the dtype `kind` ('b'), which is stable
        // across numpy versions/platforms (unlike the `str(dtype)` text,
        // which can be "bool" / "bool_" / "bool8").
        let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;

        if kind == "b" {
            // A boolean mask must be 1-D and exactly `shape[0]` long — match
            // numpy's semantics rather than silently mis-selecting rows
            // (a short mask would otherwise gather only its leading rows, and
            // a 2-D mask would collapse to its first nonzero coordinate).
            let ndim: usize = arr.getattr("ndim")?.extract()?;
            let len: usize = arr.len()?;
            if ndim != 1 || len != self.shape_val.0 {
                return Err(PyIndexError::new_err(format!(
                    "boolean index did not match indexed array along axis 0; \
                     size of axis is {} but size of corresponding boolean axis is {}",
                    self.shape_val.0, len
                )));
            }
            let nonzero = arr.call_method0("nonzero")?;
            let idx_tuple = nonzero.cast::<PyTuple>()?;
            let idx_arr = idx_tuple.get_item(0)?;
            let flat = idx_arr.call_method1("astype", (np.getattr("int64")?,))?;
            let readonly: numpy::PyReadonlyArray1<'_, i64> = flat.extract()?;
            let slice = readonly
                .as_slice()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let rows = slice
                .iter()
                .map(|&v| self.to_global_row(v as usize))
                .collect::<PyResult<Vec<u64>>>()?;
            return Ok((rows, false));
        }

        let flat = arr.call_method1("astype", (np.getattr("int64")?,))?;
        let readonly: numpy::PyReadonlyArray1<'_, i64> = flat.extract()?;
        let slice = readonly
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let rows = slice
            .iter()
            .map(|&v| {
                let norm = if v < 0 { n + v } else { v };
                if norm < 0 || norm >= n {
                    return Err(PyIndexError::new_err(format!(
                        "row index {} out of range for {} rows",
                        v, self.shape_val.0
                    )));
                }
                self.to_global_row(norm as usize)
            })
            .collect::<PyResult<Vec<u64>>>()?;
        Ok((rows, false))
    }

    /// Gather `rows` and return a 2-D numpy array `(rows.len(), n_cols)`.
    fn gather_rows_2d<'py>(&self, py: Python<'py>, rows: &[u64]) -> PyResult<Bound<'py, PyAny>> {
        let batch = self
            .backed
            .read_row_indices(rows)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        obsm_batch_to_numpy(py, &batch)
    }
}

#[pymethods]
impl ScxBackedObsmDataset {
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.shape_val
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
        let np = py.import("numpy")?;
        np.call_method1("dtype", (name,))
    }

    fn __len__(&self) -> usize {
        self.shape_val.0
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
                let builtins = py.import("builtins")?;
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
                let batch = self
                    .backed
                    .read_all()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                obsm_batch_to_numpy(py, &batch)
            }
        }
    }

    /// Alias for [`Self::to_memory`] — obsm values are already dense.
    fn toarray<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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
