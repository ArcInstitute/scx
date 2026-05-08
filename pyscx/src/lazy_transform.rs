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

use scx_format::{BackedCscReader, BackedCsrReader};
use scx_sparse::{ScxCsc, ScxCsr};

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

impl Transform {
    /// Returns `true` iff the transform's output for a given matrix
    /// element depends only on its own column (not on row sums or
    /// per-row factors).
    ///
    /// Used by `LazyShardSource`'s `ColumnShardSource` implementation
    /// and `ScxBackedSparseDataset::as_column_source()` as the
    /// transform-chain compatibility test for CSC dispatch.
    ///
    /// - `Log1p`: `ln(x + 1)` is element-wise, no row context. **true**
    /// - `NormalizeTotal`: divides by per-row sum. **false**
    /// - `RowScale`: multiplies each row by a per-row factor. **false**
    pub fn is_column_local(&self) -> bool {
        matches!(self, Transform::Log1p)
    }
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
    /// Optional CSC sidecar reader, propagated from the originating
    /// `ScxBackedSparseDataset`. Carried forward through every
    /// transform-chain extension (`log1p`, `normalize_total`, etc.) so
    /// that `as_column_source()` can light up CSC dispatch when the
    /// transform chain remains column-local.
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
        }
    }

    /// Builder method: attach a CSC sidecar reader. Mirrors
    /// `ScxBackedSparseDataset::with_csc_reader`. After this call,
    /// `as_column_source()` may return `Some` if the transform chain
    /// is column-local and no row deletion vector is active.
    pub fn with_csc_reader(mut self, backed_csc: Option<Arc<BackedCscReader>>) -> Self {
        self.backed_csc = backed_csc;
        self
    }

    /// Capability gate: returns `Some(LazyShardSource)` iff this
    /// lazy dataset can serve CSC reads. Mirrors
    /// `ScxBackedSparseDataset::as_column_source` but returns an
    /// owned `LazyShardSource` (rather than a borrowed `&dyn`) because
    /// the underlying `LazyShardSource` is materialized fresh per call;
    /// it's cheap (clones `Arc` handles only).
    ///
    /// Returns `Some` iff:
    /// - `backed_csc` is set (file has a CSC sidecar AND was opened
    ///   with CSC capability), AND
    /// - every transform in the chain returns `is_column_local() == true`
    ///   (NormalizeTotal / RowScale would corrupt CSC reads), AND
    /// - `kept_to_global` is `None` (row deletions break the global
    ///   row indices encoded in CSC `indices`).
    ///
    /// Callers consume the `LazyShardSource` via the
    /// `ColumnShardSource` trait impl on `LazyShardSource`. Crate-private
    /// because `LazyShardSource` itself is `pub(crate)`.
    ///
    /// `#[allow(dead_code)]` until consumers in `pyscx::accel` reach
    /// for it.
    #[allow(dead_code)]
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
    /// `LazyShardSource::supports_csc()` (transform-chain check).
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
    fn apply_col_projection_to_vec(&self, values: Vec<f64>) -> Vec<f64> {
        match &self.col_projection {
            Some(cols) => cols.iter().map(|&c| values[c as usize]).collect(),
            None => values,
        }
    }

    // --- Streaming aggregation through transforms ---
    //
    // These methods stream shard-by-shard, applying lazy transforms in-place,
    // then accumulating per-column or per-row statistics. All column-axis
    // methods return vectors in **physical column space** (length =
    // backed.shape().1). Callers must post-filter via apply_col_projection_to_vec()
    // when col_projection is active. See the doc comment on that method for
    // rationale.

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

    /// Stream all shards, apply transforms, compute per-row NNZ and sums in a single pass.
    ///
    /// Avoids the double I/O of calling `row_nnz()` + `streaming_row_sums()` separately.
    /// NNZ is computed pre-transform (from indptr, which transforms don't change) while
    /// sums are computed post-transform. Both use the same decoded shard.
    ///
    /// Used by `filter_cells` when both `min_genes` and `min_counts` are specified.
    /// Returns global-length vectors (NOT filtered through deletion vectors).
    pub(crate) fn streaming_row_nnz_and_sums(&self) -> PyResult<(Vec<i64>, Vec<f64>)> {
        let n_obs_global = self.backed.shape().0;
        let mut all_nnz = vec![0i64; n_obs_global];
        let mut all_sums = vec![0.0f64; n_obs_global];
        let mut global_row = 0usize;

        for shard_idx in 0..self.backed.index().n_shards() {
            let mut csr = self
                .backed
                .read_shard_uncached(shard_idx)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            // NNZ from indptr before transforms (transforms preserve sparsity pattern)
            for row in 0..csr.n_rows() {
                all_nnz[global_row + row] = csr.indptr[row + 1] - csr.indptr[row];
            }
            // Apply transforms then compute sums
            self.apply_transforms(&mut csr, global_row);
            for row in 0..csr.n_rows() {
                let s = csr.indptr[row] as usize;
                let e = csr.indptr[row + 1] as usize;
                all_sums[global_row + row] = csr.data[s..e].iter().map(|&v| v as f64).sum();
            }
            global_row += csr.n_rows();
        }
        Ok((all_nnz, all_sums))
    }

    /// Stream all shards, apply transforms, project to visible columns, compute per-row sums.
    ///
    /// **Scanpy compatibility:** After `filter_genes()`, scanpy's `normalize_total`
    /// sums only over the visible (kept) gene set because `adata.X` is already sliced.
    /// In SCX backed mode, `adata.X` is still the full-width matrix with a
    /// `col_projection` mask. This method applies transforms to the full-width shard
    /// first (so prior `NormalizeTotal` transforms divide by the correct whole-row
    /// denominator), then calls `project_csr` to restrict to the projected gene subset
    /// before summing each row.
    ///
    /// Falls back to `streaming_row_sums()` when no `col_projection` is active.
    ///
    /// Returns a global-length vector (`n_obs_global`), NOT filtered through
    /// deletion vectors.
    pub(crate) fn streaming_row_sums_projected(&self) -> PyResult<Vec<f64>> {
        let cols = match &self.col_projection {
            Some(c) => c,
            None => return self.streaming_row_sums(),
        };
        let n_obs_global = self.backed.shape().0;
        let mut sums = vec![0.0f64; n_obs_global];
        let mut global_row = 0usize;
        for shard_idx in 0..self.backed.index().n_shards() {
            let mut csr = self
                .backed
                .read_shard_uncached(shard_idx)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            self.apply_transforms(&mut csr, global_row);
            let projected = scx_engine::projection::project_csr(&csr, cols);
            for row in 0..projected.n_rows() {
                let s = projected.indptr[row] as usize;
                let e = projected.indptr[row + 1] as usize;
                sums[global_row + row] = projected.data[s..e].iter().map(|&v| v as f64).sum();
            }
            global_row += csr.n_rows();
        }
        Ok(sums)
    }

    /// Stream all shards, apply transforms, compute per-column sums.
    ///
    /// Returns a vector of length `backed.shape().1` (physical column count),
    /// NOT `shape_val.1` (projected). Callers must apply
    /// `apply_col_projection_to_vec()` before exposing to Python.
    pub(crate) fn streaming_col_sums(&self) -> PyResult<Vec<f64>> {
        let n_vars = self.backed.shape().1; // physical width, intentionally
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
    ///
    /// Returns a vector of length `backed.shape().1` (physical column count).
    /// Callers must apply `apply_col_projection_to_vec()` before exposing to
    /// Python. This is correct because per-column variance is independent —
    /// computing var for filtered-out columns is wasted work but doesn't affect
    /// the values for kept columns.
    fn streaming_col_var(&self) -> PyResult<Vec<f64>> {
        let n_vars = self.backed.shape().1; // physical width, intentionally
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
            debug_assert!(
                col_nnz[c] <= n_obs,
                "col_nnz[{}] = {} exceeds n_obs = {}",
                c,
                col_nnz[c],
                n_obs
            );
            let n_zeros = n_obs - col_nnz[c];
            let total = sq_devs[c] + n_zeros as f64 * col_means[c] * col_means[c];
            variances[c] = total / n_obs as f64;
        }
        Ok(variances)
    }

    /// Stream all shards, apply transforms, compute masked column sums (deletion-aware).
    ///
    /// Like `streaming_col_sums()`, returns a vector of length `backed.shape().1`
    /// (physical column count). Callers must apply `apply_col_projection_to_vec()`.
    pub(crate) fn streaming_col_sums_masked(&self) -> PyResult<Vec<f64>> {
        let n_vars = self.backed.shape().1; // physical width, intentionally
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
            .with_csc_reader(self.backed_csc.clone());
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
            .with_csc_reader(self.backed_csc.clone());
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
    /// `axis=1` (row sums): streams over full-width rows (transforms need all
    /// columns), then filters to kept rows via `filter_row_results()`.
    #[pyo3(signature = (axis=None))]
    fn sum<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        match axis {
            Some(0) => {
                // Streaming col sums return physical-width vector;
                // apply_col_projection_to_vec extracts projected subset.
                let sums = if self.kept_to_global.is_some() {
                    self.streaming_col_sums_masked()?
                } else {
                    self.streaming_col_sums()?
                };
                let sums = self.apply_col_projection_to_vec(sums);
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
    ///
    /// Same column projection strategy as `sum()`: axis=0 computes in physical
    /// column space, then post-filters via `apply_col_projection_to_vec()`.
    #[pyo3(signature = (axis=None))]
    fn mean<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        match axis {
            Some(0) => {
                let sums = if self.kept_to_global.is_some() {
                    self.streaming_col_sums_masked()?
                } else {
                    self.streaming_col_sums()?
                };
                // Post-filter to projected columns, then compute means.
                let sums = self.apply_col_projection_to_vec(sums);
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
    /// `axis=0`: Two-pass streaming in physical column space, then post-filtered
    /// via `apply_col_projection_to_vec()`. Both passes (means and sq_devs) compute
    /// over all physical columns; since per-column variance is independent, the
    /// projected subset values are correct.
    ///
    /// `axis=1` / `None`: Falls back to materialization — per-row variance across
    /// a column subset requires tracking which projected columns have stored
    /// entries per row, which the current streaming architecture doesn't support.
    #[pyo3(signature = (axis=None))]
    fn var<'py>(&self, py: Python<'py>, axis: Option<i32>) -> PyResult<Bound<'py, PyAny>> {
        match axis {
            Some(0) => {
                let var = self.streaming_col_var()?;
                // Apply column projection: streaming_col_var returns physical-width
                let var = self.apply_col_projection_to_vec(var);
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

        // ── Non-materializing column projection ────────────────────────
        // When row_idx selects ALL rows (`:` or `slice(None)`) and col_idx
        // is an array or boolean mask, return a new ScxLazyTransformedDataset
        // with col_projection set instead of materializing to scipy.
        if self.is_all_rows_slice(py, row_idx)? {
            if let Some(col_indices) = self.extract_col_indices(py, col_idx)? {
                let composed = self.compose_col_projection(&col_indices);
                let new_ds = ScxLazyTransformedDataset::new(
                    Arc::clone(&self.backed),
                    (self.shape_val.0, composed.len()),
                    self.kept_to_global.clone(),
                    Some(Arc::new(composed)),
                    self.transforms.clone(),
                    self.non_negative,
                )
                .with_csc_reader(self.backed_csc.clone());
                return Ok(new_ds.into_pyobject(py)?.into_any().unbind().into_bound(py));
            }
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

    /// Check whether `row_idx` selects all rows (is `slice(None)` / `:`).
    fn is_all_rows_slice(&self, _py: Python<'_>, row_idx: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(slice) = row_idx.downcast::<PySlice>() {
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
    /// Returns `Some(Vec<u32>)` for ndarray (int or bool), `None` otherwise.
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
            Some(existing) => new_indices.iter().map(|&i| existing[i as usize]).collect(),
            None => new_indices.to_vec(),
        };
        composed.sort_unstable();
        composed.dedup();
        composed
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
                    } else {
                        // Zero-sum row: all stored values must be zero for CSR
                        // from count data. In the unfused path, NormalizeTotal
                        // skips the row and Log1p applies ln(0+1)=0, so both
                        // paths produce identical results when this invariant
                        // holds. Assert to catch upstream data corruption.
                        debug_assert!(
                            csr.data[start..end].iter().all(|&v| v == 0.0),
                            "Fused NormalizeTotal+Log1p: zero-sum row {} has non-zero values",
                            g
                        );
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
                } else {
                    // Zero-sum row: all stored values must be zero for CSR
                    // from count data. In the unfused path, NormalizeTotal
                    // skips the row and Log1p applies ln(0+1)=0, so both
                    // paths produce identical results when this invariant
                    // holds. Assert to catch upstream data corruption.
                    let start = csr.indptr[row] as usize;
                    let end = csr.indptr[row + 1] as usize;
                    debug_assert!(
                        csr.data[start..end].iter().all(|&v| v == 0.0),
                        "Fused NormalizeTotal+Log1p: zero-sum row {} has non-zero values",
                        g
                    );
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
    /// Optional CSC sidecar reader. Populated when the underlying file
    /// has CSC shards AND the open path requests CSC capability.
    /// `None` ⇒ this `LazyShardSource` cannot serve `ColumnShardSource`
    /// methods (they will return an error).
    backed_csc: Option<Arc<BackedCscReader>>,
    transforms: Vec<Transform>,
    kept_to_global: Option<Arc<Vec<u64>>>,
    col_projection: Option<Arc<Vec<u32>>>,
    shape_val: (usize, usize),
}

impl LazyShardSource {
    /// Create a shard source with an optional pre-existing kept-to-global mapping.
    ///
    /// Pass `None` for `kept_to_global` to signal "all rows kept" — this skips
    /// the deletion-vector filtering path in `read_shard` and avoids allocating
    /// a full identity range vector.
    pub(crate) fn new(
        backed: Arc<BackedCsrReader>,
        transforms: Vec<Transform>,
        kept_to_global: Option<Arc<Vec<u64>>>,
        col_projection: Option<Arc<Vec<u32>>>,
        n_obs: usize,
        n_vars: usize,
    ) -> Self {
        LazyShardSource {
            backed,
            backed_csc: None,
            transforms,
            kept_to_global,
            col_projection,
            shape_val: (n_obs, n_vars),
        }
    }

    /// Create a shard source with both CSR and CSC backings.
    ///
    /// Used by callers that want CSC-capable streaming. The CSC reader
    /// must already be constructed (typically by the caller after
    /// inspecting `header.has_csc()`).
    pub(crate) fn new_with_csc(
        backed: Arc<BackedCsrReader>,
        backed_csc: Option<Arc<BackedCscReader>>,
        transforms: Vec<Transform>,
        kept_to_global: Option<Arc<Vec<u64>>>,
        col_projection: Option<Arc<Vec<u32>>>,
        n_obs: usize,
        n_vars: usize,
    ) -> Self {
        LazyShardSource {
            backed,
            backed_csc,
            transforms,
            kept_to_global,
            col_projection,
            shape_val: (n_obs, n_vars),
        }
    }

    /// Create a batch-filtered shard source for streaming HVG computation.
    ///
    /// `kept_to_global` contains the global row indices for cells in this batch.
    /// Reuses the existing deletion vector infrastructure in `read_shard()`.
    pub(crate) fn with_kept_rows(
        backed: Arc<BackedCsrReader>,
        transforms: Vec<Transform>,
        kept_to_global: Vec<u64>,
        col_projection: Option<Arc<Vec<u32>>>,
        n_vars: usize,
    ) -> Self {
        let n_obs = kept_to_global.len();
        LazyShardSource {
            backed,
            backed_csc: None,
            transforms,
            kept_to_global: Some(Arc::new(kept_to_global)),
            col_projection,
            shape_val: (n_obs, n_vars),
        }
    }

    /// Returns `true` if this lazy source can serve CSC reads:
    /// CSC sidecar present, all transforms column-local, no row
    /// deletion vector active.
    ///
    /// Predicate used by `ScxLazyTransformedDataset::as_column_source()`
    /// (the analog to `ScxBackedSparseDataset::as_column_source` for
    /// the lazy-transformed wrapper). `#[allow(dead_code)]` until the
    /// CSC consumers in `pyscx::accel` reach for it.
    #[allow(dead_code)]
    pub(crate) fn supports_csc(&self) -> bool {
        self.backed_csc.is_some()
            && self.transforms.iter().all(Transform::is_column_local)
            && self.kept_to_global.is_none()
    }
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

    /// Row counts are unchanged by transforms and column projection — delegate
    /// to the wrapped reader's O(1) implementation.
    fn max_shard_rows(&self) -> scx_format::Result<usize> {
        self.backed.max_shard_rows()
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

/// Apply column-local transforms in-place on a decoded CSC shard.
///
/// The capability gate at `ScxBackedSparseDataset::as_column_source`
/// usually filters out non-column-local transforms before this path
/// runs. As a defense in depth, this helper returns an error rather
/// than silently producing wrong results if a non-column-local
/// transform sneaks through (e.g., a future caller that bypasses the
/// gate).
fn apply_transforms_to_csc(transforms: &[Transform], csc: &mut ScxCsc) -> scx_format::Result<()> {
    for transform in transforms {
        match transform {
            Transform::Log1p => {
                for v in &mut csc.data {
                    *v = v.ln_1p();
                }
            }
            Transform::NormalizeTotal { .. } | Transform::RowScale { .. } => {
                return Err(scx_format::ScxError::Io(std::io::Error::other(
                    "CSC unavailable: chain contains a non-column-local transform \
                     (NormalizeTotal or RowScale). Use prefer_format='csr' or remove \
                     the transform.",
                )));
            }
        }
    }
    Ok(())
}

impl scx_format::ColumnShardSource for LazyShardSource {
    fn n_csc_shards(&self) -> usize {
        match &self.backed_csc {
            Some(b) => b.n_shards(),
            None => 0,
        }
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

    fn read_csc_shard(&self, shard_idx: usize) -> scx_format::Result<ScxCsc> {
        if self.kept_to_global.is_some() {
            return Err(scx_format::ScxError::Io(std::io::Error::other(
                "CSC unavailable: row deletion vector is active",
            )));
        }
        let backed = self.backed_csc.as_ref().ok_or_else(|| {
            scx_format::ScxError::Io(std::io::Error::other(
                "CSC unavailable: file has no CSC sidecar (open with CSC enabled)",
            ))
        })?;
        let mut csc = (*backed.read_shard_cached(shard_idx)?).clone();
        apply_transforms_to_csc(&self.transforms, &mut csc)?;
        if let Some(ref proj) = self.col_projection {
            // `proj` is sorted/dedup'd GLOBAL column IDs, but `csc` is a
            // shard slab whose own column space is `0..shard_n_cols`.
            // Filter `proj` to entries inside this shard's global range,
            // remap to shard-local, then project.
            let (g_lo, g_hi) =
                scx_format::ColumnShardSource::csc_shard_col_range(backed.as_ref(), shard_idx)
                    .ok_or_else(|| {
                        scx_format::ScxError::Io(std::io::Error::other(
                            "CSC unavailable: missing shard col range for projection remap",
                        ))
                    })?;
            let p_lo = proj.partition_point(|&g| g < g_lo);
            let p_hi = proj.partition_point(|&g| g < g_hi);
            let local: Vec<u32> = proj[p_lo..p_hi].iter().map(|&g| g - g_lo).collect();
            csc = scx_engine::projection::project_csc(&csc, &local);
        }
        Ok(csc)
    }

    fn read_csc_columns(&self, col_range: std::ops::Range<u32>) -> scx_format::Result<ScxCsc> {
        if self.kept_to_global.is_some() {
            return Err(scx_format::ScxError::Io(std::io::Error::other(
                "CSC unavailable: row deletion vector is active",
            )));
        }
        let backed = self.backed_csc.as_ref().ok_or_else(|| {
            scx_format::ScxError::Io(std::io::Error::other(
                "CSC unavailable: file has no CSC sidecar (open with CSC enabled)",
            ))
        })?;

        // When a column projection is active, the user-facing column
        // axis is the projected one. Translate the projected range into
        // the underlying global range, fetch via the inner reader, and
        // re-project the result to the projected axis.
        let mut csc = match &self.col_projection {
            Some(proj) => {
                let lo = col_range.start as usize;
                let hi = (col_range.end as usize).min(proj.len());
                if lo >= hi {
                    // Empty range — return an empty CSC sized to the
                    // projected n_vars window.
                    return Ok(ScxCsc::new_unchecked(
                        (self.shape_val.0, 0),
                        vec![0],
                        Vec::new(),
                        Vec::new(),
                    ));
                }
                let global_subset = &proj[lo..hi];
                backed.read_csc_columns_subset(global_subset)?
            }
            None => backed.read_csc_columns(col_range)?,
        };

        apply_transforms_to_csc(&self.transforms, &mut csc)?;
        Ok(csc)
    }

    fn csc_shard_col_range(&self, shard_idx: usize) -> Option<(u32, u32)> {
        let backed = self.backed_csc.as_ref()?;
        let (g_lo, g_hi) =
            scx_format::ColumnShardSource::csc_shard_col_range(backed.as_ref(), shard_idx)?;
        match &self.col_projection {
            Some(proj) => {
                // Map the inner shard's global range [g_lo, g_hi) onto
                // the projected axis. Consumers iterating shards see
                // ranges in the same axis as `n_vars()` (projected).
                let p_lo = proj.partition_point(|&g| g < g_lo) as u32;
                let p_hi = proj.partition_point(|&g| g < g_hi) as u32;
                Some((p_lo, p_hi))
            }
            None => Some((g_lo, g_hi)),
        }
    }
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
