// ScxLazyTransformedDataset — PyO3 class for lazy per-row transforms.
//
// Extracted from the former pyscx/src/lazy_transform.rs (T5.7).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use scx_format_io::{BackedCscReader, BackedCsrReader};
use scx_sparse::ScxCsr;

use crate::backed::detached;

use super::*;

/// PyO3 class wrapping BackedCsrReader with chained per-row transforms.
///
/// Implements the same anndata.abc.CSRDataset-compatible interface as
/// ScxBackedSparseDataset but applies transforms on-read. Transforms are
/// applied in-order to each decoded shard during __getitem__.

#[pyclass(name = "ScxLazyTransformedDataset")]
pub struct ScxLazyTransformedDataset {
    pub(crate) backed: Arc<BackedCsrReader>,
    /// Optional CSC sidecar reader, propagated from the originating
    /// `ScxBackedSparseDataset`. Carried forward through every
    /// transform-chain extension (`log1p`, `normalize_total`, etc.) so
    /// that `as_column_source()` can light up CSC dispatch — which every
    /// transform chain now qualifies for, since each `Transform` is
    /// CSC-applicable.
    pub(crate) backed_csc: Option<Arc<BackedCscReader>>,
    pub(crate) shape_val: (usize, usize),
    pub(crate) transforms: Vec<Transform>,
    /// Arc-wrapped to avoid O(n) deep clones when constructing new lazy
    /// datasets via __mul__ / __truediv__ / comparison operators.
    pub(crate) kept_to_global: Option<Arc<Vec<u64>>>,
    /// Arc-wrapped for the same reason as `kept_to_global`.
    pub(crate) col_projection: Option<Arc<Vec<u32>>>,
    /// Whether the data is known to be non-negative after transforms.
    /// NormalizeTotal and Log1p preserve non-negativity.
    pub(crate) non_negative: bool,
    /// On-disk source SCX file path, propagated from the originating
    /// `ScxBackedSparseDataset` through every transform-chain
    /// extension. Read by the Phase 8b SCX → SCX writer for
    /// catalog introspection (the lazy path never does byte
    /// passthrough — transforms always force a re-encode).
    pub(crate) source_path: Option<PathBuf>,
}

impl ScxLazyTransformedDataset {
    /// Create a new lazy transformed dataset.
    pub fn new(
        backed: Arc<BackedCsrReader>,
        shape_val: (usize, usize),
        kept_to_global: Option<Arc<Vec<u64>>>,
        col_projection: Option<Arc<Vec<u32>>>,
        transforms: Vec<Transform>,
        non_negative: bool,
    ) -> Self {
        Self {
            backed,
            backed_csc: None,
            shape_val,
            transforms,
            kept_to_global,
            col_projection,
            non_negative,
            source_path: None,
        }
    }

    /// Builder-style setter for the on-disk source path. Used when
    /// the lazy dataset is built from a backed source with a known
    /// path so the Phase 8b SCX → SCX writer can recover it.
    pub fn with_source_path(mut self, path: Option<PathBuf>) -> Self {
        self.source_path = path;
        self
    }

    /// Returns the on-disk source path that this wrapper was built
    /// from, if known. `None` when the originating
    /// `ScxBackedSparseDataset` had no source path.
    pub fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }

    /// Returns the ordered list of transforms stacked on this
    /// dataset. Read by the Phase 8b SCX → SCX writer for
    /// provenance JSON; full per-row factor / row-sum vectors are
    /// summarised by length in the provenance entry.
    pub fn transforms(&self) -> &[Transform] {
        &self.transforms
    }

    /// Builder method: attach a CSC sidecar reader. Mirrors
    /// `ScxBackedSparseDataset::with_csc_reader`. After this call,
    /// `as_column_source()` returns `Some`.
    pub fn with_csc_reader(mut self, backed_csc: Option<Arc<BackedCscReader>>) -> Self {
        self.backed_csc = backed_csc;
        self
    }

    /// Capability gate: returns `Some(LazyShardSource)` iff this lazy
    /// dataset can serve CSC reads — i.e. iff `backed_csc` is set. It is now
    /// the *same* function as `ScxBackedSparseDataset::as_column_source`,
    /// which used to return a borrowed full-axis `&dyn ColumnShardSource`
    /// and had to refuse a window; both return this owned view, which is
    /// materialized fresh per call and cheap (clones `Arc` handles only).
    ///
    /// Neither the transform chain nor a row filter is a condition. Every
    /// `Transform` has a column-major form, the row-indexed `NormalizeTotal`
    /// / `RowScale` included, read at the global row `ScxCsc::indices`
    /// already carries; and the row filter is applied by renumbering the
    /// slab onto the live row space after those transforms have run.
    ///
    /// Callers consume the `LazyShardSource` via the
    /// `ColumnShardSource` trait impl on `LazyShardSource`. Crate-private
    /// because `LazyShardSource` itself is `pub(crate)`.
    ///
    pub(crate) fn as_column_source(&self) -> Option<LazyShardSource> {
        let source = self.as_shard_source();
        if source.supports_csc() {
            Some(source)
        } else {
            None
        }
    }

    /// Create a `LazyShardSource` for streaming algorithms (PCA, HVG,
    /// CSC consumers).
    ///
    /// Supports column projection: when active, `LazyShardSource` applies
    /// per-shard column filtering and reports `n_vars()` as the projected
    /// column count.
    ///
    /// Threads `backed_csc` through, so consumers that route via
    /// `ColumnShardSource` (Phase F) get the CSC plumbing for free.
    /// Whether CSC is actually serviceable is gated separately by
    /// `LazyShardSource::supports_csc()` — a sidecar must be present, and
    /// that is the whole condition.
    pub(crate) fn as_shard_source(&self) -> LazyShardSource {
        LazyShardSource::new_with_csc(
            Arc::clone(&self.backed),
            self.backed_csc.clone(),
            self.transforms.clone(),
            self.kept_to_global.clone(),
            self.col_projection.clone(),
            self.shape_val.0,
            self.shape_val.1,
        )
    }

    /// Replace the deletion vector, adjusting shape.0.
    pub(crate) fn set_kept_to_global(&mut self, kept: Vec<u64>) {
        self.shape_val.0 = kept.len();
        self.kept_to_global = Some(Arc::new(kept));
    }

    /// Set column projection on this dataset.
    pub(crate) fn set_col_projection(&mut self, col_indices: Vec<u32>) {
        let mut sorted = col_indices;
        sorted.sort_unstable();
        sorted.dedup();
        self.shape_val.1 = sorted.len();
        self.col_projection = Some(Arc::new(sorted));
    }

    /// Read access to col_projection (for composition in filter_genes).
    pub(crate) fn col_projection(&self) -> Option<&[u32]> {
        self.col_projection.as_ref().map(|v| v.as_slice())
    }

    /// A second handle onto the same window with the same transform chain.
    ///
    /// The per-row transform parameters are indexed by *global* row, so a
    /// changed `kept_to_global` re-points the window without invalidating
    /// them — which is why a row subset needs no transform surgery.
    pub(crate) fn clone_handle(&self) -> Self {
        Self {
            backed: Arc::clone(&self.backed),
            backed_csc: self.backed_csc.clone(),
            shape_val: self.shape_val,
            transforms: self.transforms.clone(),
            kept_to_global: self.kept_to_global.clone(),
            col_projection: self.col_projection.clone(),
            non_negative: self.non_negative,
            source_path: self.source_path.clone(),
        }
    }

    /// The lazy twin of [`crate::backed::ScxBackedSparseDataset::subset_clone`].
    ///
    /// A lazy dataset has no `col_presentation`, so it cannot express a
    /// column *reorder* — `set_col_projection` sorts. A composed order that is
    /// not already ascending is therefore rejected rather than silently
    /// permuted: `adata.var` would follow the request order while `X` followed
    /// disk order, which is the silent-divergence class §9.18 exists to close.
    /// A mask-derived subset (every `filter_genes` / HVG path) always composes
    /// ascending and never hits this.
    pub(crate) fn subset_clone(
        &self,
        rows: Option<&[i64]>,
        cols: Option<&[i64]>,
    ) -> PyResult<Self> {
        let mut out = self.clone_handle();
        if let Some(rows) = rows {
            out.set_kept_to_global(crate::axis_align::compose_rows_positional(
                self.kept_to_global.as_ref().map(|v| v.as_slice()),
                rows,
                self.shape_val.0,
            )?);
        }
        if let Some(cols) = cols {
            let composed = crate::axis_align::compose_cols_positional(
                self.col_projection.as_ref().map(|v| v.as_slice()),
                cols,
                self.shape_val.1,
            )?;
            if composed.windows(2).any(|w| w[0] >= w[1]) {
                return Err(PyRuntimeError::new_err(
                    "cannot reorder or repeat the columns of a lazily transformed X \
                     (normalize_total / log1p / row_scale): the projection is stored \
                     sorted, so var and X would disagree. Materialize first with \
                     `adata.X = adata.X.to_memory()`.",
                ));
            }
            out.set_col_projection(composed);
        }
        Ok(out)
    }

    /// Apply all transforms in-place on a decoded CSR shard.
    ///
    /// `global_row_offset` is the starting global row index for this shard,
    /// used to look up per-row parameters (row_sums, factors).
    pub(crate) fn apply_transforms(&self, csr: &mut ScxCsr, global_row_offset: usize) {
        apply_transforms_to_csr(&self.transforms, csr, global_row_offset);
    }

    /// Apply column projection to a CSR matrix if projection is active.
    pub(crate) fn apply_col_projection(&self, csr: ScxCsr) -> ScxCsr {
        match &self.col_projection {
            Some(indices) => scx_engine::projection::project_csr(&csr, indices),
            None => csr,
        }
    }

    /// Map a user-visible row index to global (file-level) row index.
    pub(crate) fn to_global_row(&self, user_row: usize) -> PyResult<usize> {
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
    pub(crate) fn normalize_row_index(&self, i: i64) -> PyResult<usize> {
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

    /// Project a physical-column-width vector down to the user-visible column subset.
    ///
    /// # Column Projection Design ("Option A: post-filtering")
    ///
    /// Streaming column aggregation methods (`streaming_col_sums`,
    /// `streaming_col_sums_masked`, `streaming_col_var`) intentionally operate in
    /// **physical column space** — they return vectors of length `backed.shape().1`
    /// covering all on-disk columns. Callers then post-filter via this method to
    /// extract only the projected columns, producing a result of length
    /// `shape_val.1`.
    ///
    /// This design is correct because:
    /// - Column statistics (sum, mean, variance) are independent per column.
    ///   Computing them over all columns and subsetting is numerically equivalent
    ///   to computing only over the subset.
    /// - Transforms (especially `NormalizeTotal`) must see the **full-width** row
    ///   to compute correct per-row scaling factors. Projecting columns before
    ///   transforming would produce wrong normalization denominators.
    ///
    /// An alternative ("Option B") would integrate `project_csr` per-shard inside
    /// each streaming method, skipping non-projected entries during accumulation.
    /// This would reduce inner-loop iterations on heavily filtered datasets but
    /// adds per-shard allocation overhead from `project_csr`. Option A is simpler
    /// and sufficient for typical workloads.
    ///
    /// # Contract
    ///
    /// Every `sum(axis=0)`, `mean(axis=0)`, and `var(axis=0)` call site **must**
    /// apply this method before reshaping to `(1, shape_val.1)`, otherwise the
    /// numpy reshape will fail with a dimension mismatch when `col_projection`
    /// is active.
    pub(crate) fn apply_col_projection_to_vec(&self, values: Vec<f64>) -> Vec<f64> {
        match &self.col_projection {
            Some(cols) => cols.iter().map(|&c| values[c as usize]).collect(),
            None => values,
        }
    }
}

#[pymethods]
impl ScxLazyTransformedDataset {
    /// Guarded: the cached scalar never reaches `section_bytes`, so without
    /// this a lazy handle would keep reporting the row count the file had
    /// before an `append` or a `mark_deleted`.
    #[getter]
    fn shape(&self) -> PyResult<(usize, usize)> {
        self.backed.check_fresh().map_err(crate::to_pyerr)?;
        Ok(self.shape_val)
    }

    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = crate::pyimport::import_module(py, "numpy")?;
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

    fn __len__(&self) -> PyResult<usize> {
        self.backed.check_fresh().map_err(crate::to_pyerr)?;
        Ok(self.shape_val.0)
    }

    fn __repr__(&self) -> String {
        let transform_names: Vec<&str> = self
            .transforms
            .iter()
            .map(|t| match t {
                Transform::NormalizeTotal { .. } => "NormalizeTotal",
                Transform::Log1p => "Log1p",
                Transform::RowScale { .. } => "RowScale",
                Transform::Scale { .. } => "Scale",
            })
            .collect();
        format!(
            "ScxLazyTransformedDataset(shape=({}, {}), transforms={:?})",
            self.shape_val.0, self.shape_val.1, transform_names
        )
    }

    /// Returns the stacked transforms as a JSON-serialisable Python
    /// list of `{"name": str, "params": dict}` entries.
    ///
    /// Per-row factor / row-sum vectors are summarised by length
    /// (`{"factors_len": n}` / `{"row_sums_len": n}`) so provenance
    /// payloads stay small.
    fn transforms_repr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for t in &self.transforms {
            let entry = PyDict::new(py);
            let params = PyDict::new(py);
            let name = match t {
                Transform::NormalizeTotal {
                    row_sums,
                    target_sum,
                } => {
                    params.set_item("row_sums_len", row_sums.len())?;
                    params.set_item("target_sum", *target_sum)?;
                    "normalize_total"
                }
                Transform::Log1p => "log1p",
                Transform::RowScale { factors } => {
                    params.set_item("factors_len", factors.len())?;
                    "row_scale"
                }
                Transform::Scale { factor } => {
                    params.set_item("factor", *factor)?;
                    "scale"
                }
            };
            entry.set_item("name", name)?;
            entry.set_item("params", params)?;
            list.append(entry)?;
        }
        Ok(list)
    }

    /// Load a slice from disk, apply transforms, return scipy CSR.
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

    /// Materialize the full matrix into memory with all transforms applied.
    pub(crate) fn to_memory<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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
            crate::backed::try_extract_row_factors(py, other, self.shape_val.0, self.shape_val.1)?
        {
            let global_factors = crate::backed::expand_to_global(
                row_factors,
                self.kept_to_global.as_ref().map(|v| v.as_slice()),
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
        // Try to express as a lazy RowScale (multiply by 1/factor).
        // This intercepts scanpy's axis_mul_or_truediv path.
        if let Some(row_factors) =
            crate::backed::try_extract_row_factors(py, other, self.shape_val.0, self.shape_val.1)?
        {
            // A zero divisor leaves the row unscaled (1/0 → 0, i.e. multiply by
            // 0 drops the row to empty) — this matches scanpy `normalize_total`,
            // where empty rows stay empty, NOT raw scipy float division (which
            // would yield inf/nan for a nonzero numerator over a zero divisor).
            let inv_factors: Vec<f64> = row_factors
                .iter()
                .map(|&f| if f != 0.0 { 1.0 / f } else { 0.0 })
                .collect();
            let global_inv = crate::backed::expand_to_global(
                inv_factors,
                self.kept_to_global.as_ref().map(|v| v.as_slice()),
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

    /// Sum along an axis, streaming through transforms.
    ///
    /// Decodes each shard, applies transforms, then aggregates.
    /// Peak memory = O(shard_size), not the full matrix.
    ///
    /// `axis=0` (column sums): streams in physical column space, then
    /// post-filters to projected columns via `apply_col_projection_to_vec()`.
    /// `axis=1` (row sums): [`Self::row_sums_raw`] applies transforms to the
    /// full-width row (a prior `NormalizeTotal` needs the denominator it was
    /// configured with) and *then* restricts to the projected columns, so the
    /// result covers only genes the caller can see; `filter_row_results()`
    /// selects the kept rows.
    #[pyo3(signature = (axis=None))]
    fn sum<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        match axis {
            Some(0) => {
                // Streaming col sums return physical-width vector;
                // apply_col_projection_to_vec extracts projected subset.
                // Shard decode + transform + reduction runs off the GIL.
                let sums = detached(py, || {
                    if self.kept_to_global.is_some() {
                        self.streaming_col_sums_masked()
                    } else {
                        self.streaming_col_sums()
                    }
                })
                .map_err(PyRuntimeError::new_err)?;
                let sums = self.apply_col_projection_to_vec(sums);
                let arr = numpy::PyArray::from_vec(py, sums);
                arr.call_method1("reshape", ((1i32, self.shape_val.1),))
            }
            Some(1) => {
                let all_sums =
                    detached(py, || self.row_sums_raw()).map_err(PyRuntimeError::new_err)?;
                let filtered = self.filter_row_results(&all_sums);
                let arr = numpy::PyArray::from_vec(py, filtered);
                arr.call_method1("reshape", ((self.shape_val.0, 1i32),))
            }
            None => {
                let all_sums =
                    detached(py, || self.row_sums_raw()).map_err(PyRuntimeError::new_err)?;
                let filtered = self.filter_row_results(&all_sums);
                let total: f64 = filtered.iter().sum();
                Ok(total.into_pyobject(py)?.into_any())
            }
            Some(_) => Err(PyRuntimeError::new_err("axis must be 0, 1, or None")),
        }
    }

    /// Mean along an axis, streaming through transforms.
    ///
    /// Same projection strategy as `sum()`: axis=0 computes in physical column
    /// space then post-filters via `apply_col_projection_to_vec()`; axis=1
    /// sums only the visible columns via [`Self::row_sums_raw`], which is what
    /// makes the `shape_val.1` denominator right.
    #[pyo3(signature = (axis=None))]
    fn mean<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        match axis {
            Some(0) => {
                // Shard decode + transform + reduction runs off the GIL.
                let sums = detached(py, || {
                    if self.kept_to_global.is_some() {
                        self.streaming_col_sums_masked()
                    } else {
                        self.streaming_col_sums()
                    }
                })
                .map_err(PyRuntimeError::new_err)?;
                // Post-filter to projected columns, then compute means.
                let sums = self.apply_col_projection_to_vec(sums);
                let n = self.shape_val.0 as f64;
                let means: Vec<f64> = sums.iter().map(|&s| s / n).collect();
                let arr = numpy::PyArray::from_vec(py, means);
                arr.call_method1("reshape", ((1i32, self.shape_val.1),))
            }
            Some(1) => {
                let all_sums =
                    detached(py, || self.row_sums_raw()).map_err(PyRuntimeError::new_err)?;
                let filtered = self.filter_row_results(&all_sums);
                let n = self.shape_val.1 as f64;
                let means: Vec<f64> = filtered.iter().map(|&s| s / n).collect();
                let arr = numpy::PyArray::from_vec(py, means);
                arr.call_method1("reshape", ((self.shape_val.0, 1i32),))
            }
            None => {
                let all_sums =
                    detached(py, || self.row_sums_raw()).map_err(PyRuntimeError::new_err)?;
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
    /// `axis=0`: Two-pass streaming in physical column space, then post-filtered
    /// via `apply_col_projection_to_vec()`. Both passes (means and sq_devs) compute
    /// over all physical columns; since per-column variance is independent, the
    /// projected subset values are correct.
    ///
    /// `axis=1` / `None`: Falls back to materialization — per-row variance across
    /// a column subset requires tracking which projected columns have stored
    /// entries per row, which the current streaming architecture doesn't support.
    /// `to_memory()` is projection- and deletion-aware, so the fallback is
    /// correct on the visible axes.
    #[pyo3(signature = (axis=None))]
    fn var<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        match axis {
            Some(0) => {
                let var =
                    detached(py, || self.streaming_col_var()).map_err(PyRuntimeError::new_err)?;
                // Apply column projection: streaming_col_var returns physical-width
                let var = self.apply_col_projection_to_vec(var);
                let arr = numpy::PyArray::from_vec(py, var);
                arr.call_method1("reshape", ((1i32, self.shape_val.1),))
            }
            Some(1) | None => {
                // For row var and scalar var, materialize since per-row
                // variance through transforms requires careful col-count
                // handling per-row — fallback is simpler and correct.
                //
                // `Var(X) = E[X²] - E[X]²`. Squaring goes through `np.square`,
                // not `.power(2)`: `.power` is a *scipy sparse* method, and the
                // means here are an `np.matrix` (axis=1) or a Python float
                // (axis=None), neither of which has it — squaring the mean used
                // to raise `AttributeError` and made both arms unreachable.
                let np = crate::pyimport::import_module(py, "numpy")?;
                let mat = self.to_memory(py)?;
                let (mean, mean_sq) = match axis {
                    Some(1) => (
                        mat.call_method1("mean", (1i32,))?,
                        mat.call_method1("power", (2,))?
                            .call_method1("mean", (1i32,))?,
                    ),
                    None => (
                        mat.call_method0("mean")?,
                        mat.call_method1("power", (2,))?.call_method0("mean")?,
                    ),
                    _ => unreachable!(),
                };
                np.call_method1(
                    "subtract",
                    (&mean_sq, &np.call_method1("square", (&mean,))?),
                )
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
                        f_counts.iter().map(|&v| v as u32).collect::<Vec<u32>>()
                    }
                    (None, None) => self
                        .backed
                        .col_nnz()
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                };
                Ok(numpy::PyArray::from_vec(py, counts).into_any())
            }
            Some(1) => {
                let all_nnz =
                    detached(py, || self.row_nnz_raw()).map_err(PyRuntimeError::new_err)?;
                let filtered = self.filter_row_results(&all_nnz);
                Ok(numpy::PyArray::from_vec(py, filtered).into_any())
            }
            None => match (&self.col_projection, &self.kept_to_global) {
                (Some(cols), Some(kept)) => {
                    let nnz =
                        crate::projected_agg::col_nnz_masked_projected(&self.backed, kept, cols)
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    // Sum as u64: projected nnz can exceed u32::MAX at atlas scale.
                    let total: u64 = nnz.iter().map(|&v| v as u64).sum();
                    Ok((total as usize).into_pyobject(py)?.into_any())
                }
                (Some(cols), None) => {
                    let nnz = crate::projected_agg::col_nnz_projected(&self.backed, cols)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    // Sum as u64: projected nnz can exceed u32::MAX at atlas scale.
                    let total: u64 = nnz.iter().map(|&v| v as u64).sum();
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
                // Sum as u64: projected nnz can exceed u32::MAX at atlas scale.
                Ok(nnz.iter().map(|&v| v as u64).sum::<u64>() as usize)
            }
            (Some(cols), None) => {
                let nnz = crate::projected_agg::col_nnz_projected(&self.backed, cols)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                // Sum as u64: projected nnz can exceed u32::MAX at atlas scale.
                Ok(nnz.iter().map(|&v| v as u64).sum::<u64>() as usize)
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

    /// The on-disk encoding of the *source* matrix, before the transforms
    /// (`normalize_total` / `log1p` produce floats on read regardless) — see
    /// `ScxBackedSparseDataset.stored_dtype`.
    #[getter]
    fn stored_dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::backed::stored_dtype_object(py, &self.backed)
    }

    /// The decoded-shard LRU size of the reader this handle shares with the
    /// `ScxBackedSparseDataset` it was derived from; `0` = no cache.
    #[getter]
    fn cache_shards(&self) -> usize {
        self.backed.cache_shards()
    }

    /// Refused, as on `ScxBackedSparseDataset`: the transforms would have to
    /// run over the whole matrix to answer.
    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        _py: Python<'py>,
        dtype: Option<Bound<'py, PyAny>>,
        copy: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = (dtype, copy);
        Err(crate::backed::no_implicit_array_error(
            "ScxLazyTransformedDataset",
        ))
    }

    /// Return shard boundaries as a list of (row_start, row_end) tuples.
    ///
    /// Same tiling contract as `ScxBackedSparseDataset::shard_boundaries`:
    /// user-visible row space, first pair starts at 0, last ends at `n_obs`,
    /// each starts where the previous ended (an all-deleted shard is omitted).
    /// `pyscx.iter_chunks` uses it on a lazily transformed `X` too.
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

// ---------------------------------------------------------------------------
// Free-standing transform application
// ---------------------------------------------------------------------------
