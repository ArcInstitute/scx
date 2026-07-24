// ScxLazyTransformedDataset — PyO3 class for lazy per-row transforms.
//
// Extracted from the former pyscx/src/lazy_transform.rs (T5.7).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice, PyTuple};

use scx_format_io::{BackedCscReader, BackedCsrReader};
use scx_sparse::ScxCsr;

use crate::backed::detached;
use crate::backed::ScxComparisonResult;
use crate::convert::csr_to_scipy;

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
    ///
    /// Pure-Rust (returns `Result<_, String>`, no `PyErr`) so callers can run
    /// it through `detached` with the GIL released.
    pub(crate) fn streaming_row_sums(&self) -> Result<Vec<f64>, String> {
        let n_obs_global = self.backed.shape().0;
        let mut sums = vec![0.0f64; n_obs_global];
        let mut global_row = 0usize;

        for shard_idx in 0..self.backed.index().n_shards() {
            let mut csr = self
                .backed
                .read_shard_uncached(shard_idx)
                .map_err(|e| e.to_string())?;
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
    ///
    /// Pure-Rust (returns `Result<_, String>`, no `PyErr`) so callers can run
    /// it through `detached` with the GIL released.
    pub(crate) fn streaming_row_nnz_and_sums(&self) -> Result<(Vec<i64>, Vec<f64>), String> {
        let n_obs_global = self.backed.shape().0;
        let mut all_nnz = vec![0i64; n_obs_global];
        let mut all_sums = vec![0.0f64; n_obs_global];
        let mut global_row = 0usize;

        for shard_idx in 0..self.backed.index().n_shards() {
            let mut csr = self
                .backed
                .read_shard_uncached(shard_idx)
                .map_err(|e| e.to_string())?;
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
    ///
    /// Pure-Rust (returns `Result<_, String>`, no `PyErr`) so callers can run
    /// it through `detached` with the GIL released.
    pub(crate) fn streaming_row_sums_projected(&self) -> Result<Vec<f64>, String> {
        match self.col_projection.clone() {
            Some(cols) => self.streaming_row_sums_for_cols(&cols),
            None => self.streaming_row_sums(),
        }
    }

    /// Stream all shards, apply transforms to the **full-width** shard, then
    /// restrict to `cols` before summing each row.
    ///
    /// `cols` are indices into the underlying reader's column space (on-disk
    /// columns), NOT the visible axis — a caller holding visible-space indices
    /// must compose them through `col_projection` first.
    ///
    /// Transforms run before the projection so a prior `NormalizeTotal`
    /// divides by the correct whole-row denominator; see
    /// [`Self::streaming_row_sums_projected`].
    ///
    /// Used for `calculate_qc_metrics`' per-`qc_var` subset sums, which must be
    /// post-transform so `pct_counts_<v>` divides a transformed numerator by a
    /// transformed denominator.
    ///
    /// Returns a global-length vector (`n_obs_global`), NOT filtered through
    /// deletion vectors.
    pub(crate) fn streaming_row_sums_for_cols(&self, cols: &[u32]) -> Result<Vec<f64>, String> {
        let n_obs_global = self.backed.shape().0;
        let mut sums = vec![0.0f64; n_obs_global];
        let mut global_row = 0usize;
        for shard_idx in 0..self.backed.index().n_shards() {
            let mut csr = self
                .backed
                .read_shard_uncached(shard_idx)
                .map_err(|e| e.to_string())?;
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
    ///
    /// Pure-Rust (returns `Result<_, String>`, no `PyErr`) so callers can run
    /// it through `detached` with the GIL released.
    pub(crate) fn streaming_col_sums(&self) -> Result<Vec<f64>, String> {
        let n_vars = self.backed.shape().1; // physical width, intentionally
        let mut sums = vec![0.0f64; n_vars];
        let mut global_row = 0usize;

        for shard_idx in 0..self.backed.index().n_shards() {
            let mut csr = self
                .backed
                .read_shard_uncached(shard_idx)
                .map_err(|e| e.to_string())?;
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
    fn streaming_col_var(&self) -> Result<Vec<f64>, String> {
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
                .map_err(|e| e.to_string())?;
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
    ///
    /// Pure-Rust (returns `Result<_, String>`, no `PyErr`) so callers can run
    /// it through `detached` with the GIL released.
    pub(crate) fn streaming_col_sums_masked(&self) -> Result<Vec<f64>, String> {
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
                .map_err(|e| e.to_string())?;
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

    /// Fused per-cell QC pass through the transform chain: row nnz, row sums
    /// and per-`qc_var` subset sums over the visible columns, in one scan.
    ///
    /// Lazy twin of [`crate::projected_agg::qc_row_pass`]. Transforms are
    /// applied to the **full-width** shard before projection so a prior
    /// `NormalizeTotal` divides by the denominator it was configured with; see
    /// [`Self::streaming_row_sums_projected`].
    ///
    /// nnz is read from `indptr` *before* the transforms run, matching
    /// [`Self::streaming_row_nnz_and_sums`] — the supported transforms are
    /// value-wise and preserve the sparsity pattern.
    ///
    /// Returns global-length vectors (NOT filtered through deletion vectors).
    pub(crate) fn streaming_qc_row_pass(
        &self,
        qc_bits: &[u64],
        n_qc: usize,
    ) -> Result<crate::projected_agg::QcRowStats, String> {
        let cols = self.col_projection.clone();
        let mut out = crate::projected_agg::QcRowStats::zeroed(self.backed.shape().0, n_qc);
        let mut global_row = 0usize;
        for shard_idx in 0..self.backed.index().n_shards() {
            let mut csr = self
                .backed
                .read_shard_uncached(shard_idx)
                .map_err(|e| e.to_string())?;
            let n_rows = csr.n_rows();
            self.apply_transforms(&mut csr, global_row);
            match cols.as_deref() {
                Some(c) => crate::projected_agg::accumulate_qc_rows_into(
                    &scx_engine::projection::project_csr(&csr, c),
                    global_row,
                    qc_bits,
                    &mut out,
                ),
                None => crate::projected_agg::accumulate_qc_rows_into(
                    &csr, global_row, qc_bits, &mut out,
                ),
            }
            global_row += n_rows;
        }
        Ok(out)
    }

    /// Fused per-column sums + nnz through the transform chain, honoring
    /// column projection and keep-mask. Length = `shape_val.1`.
    ///
    /// One scan producing both statistics; the column-axis counterpart of
    /// [`Self::streaming_qc_row_pass`]. NNZ counts stored entries, which the
    /// value-wise transforms leave untouched, so it matches the raw reader's
    /// `col_nnz`.
    pub(crate) fn col_sums_and_nnz_raw(&self) -> Result<(Vec<f64>, Vec<u32>), String> {
        let n_vars = self.backed.shape().1; // physical width, projected below
        let mut sums = vec![0.0f64; n_vars];
        let mut counts = vec![0u32; n_vars];
        let mut global_row = 0usize;

        for shard_idx in 0..self.backed.index().n_shards() {
            let mut csr = self
                .backed
                .read_shard_uncached(shard_idx)
                .map_err(|e| e.to_string())?;
            let n_rows = csr.n_rows();
            self.apply_transforms(&mut csr, global_row);

            match &self.kept_to_global {
                None => {
                    for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                        let c = col as usize;
                        sums[c] += val as f64;
                        counts[c] += 1;
                    }
                }
                Some(kept) => {
                    let (s_start, s_end) = match self.backed.index().shard_range(shard_idx) {
                        Some(r) => r,
                        None => {
                            global_row += n_rows;
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
                            sums[c] += csr.data[j] as f64;
                            counts[c] += 1;
                        }
                    }
                }
            }
            global_row += n_rows;
        }

        let sums = self.apply_col_projection_to_vec(sums);
        let counts = match &self.col_projection {
            Some(cols) => cols.iter().map(|&c| counts[c as usize]).collect(),
            None => counts,
        };
        Ok((sums, counts))
    }

    // --- Visible-space aggregation wrappers ---
    //
    // The `streaming_*` kernels above deliberately return **physical**-width
    // column vectors and full-length row vectors. These `*_raw` wrappers are
    // the visible-space equivalents — they mirror the identically named
    // helpers on `ScxBackedSparseDataset` so a caller holding either type
    // routes the `(col_projection, kept_to_global)` combination the same way
    // and gets a vector whose length matches `shape_val`.

    /// Materialize the full matrix with all transforms applied.
    ///
    /// Public within the crate so ScxComparisonResult can call it.
    ///
    /// Pure-Rust (returns `Result<_, String>`, no `PyErr`) so callers can run
    /// it through `detached` with the GIL released.
    pub(crate) fn materialize_csr(&self) -> Result<ScxCsr, String> {
        let mut global_row = 0usize;
        let mut all_slices = Vec::new();

        for shard_idx in 0..self.backed.index().n_shards() {
            let mut csr = self
                .backed
                .read_shard_uncached(shard_idx)
                .map_err(|e| e.to_string())?;
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
    ///
    /// Shard decode + transform runs off the GIL (`detached`); only the scipy
    /// object is built on the GIL. This is the single hot path behind every
    /// `to_memory`-based method (`multiply`, `power`, `std`, row/scalar `var`, …).
    pub(crate) fn to_memory_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let csr = detached(py, || self.materialize_csr()).map_err(PyRuntimeError::new_err)?;
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
    /// `axis=1` (row sums): streams over full-width rows (transforms need all
    /// columns), then filters to kept rows via `filter_row_results()`.
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
                    detached(py, || self.streaming_row_sums()).map_err(PyRuntimeError::new_err)?;
                let filtered = self.filter_row_results(&all_sums);
                let arr = numpy::PyArray::from_vec(py, filtered);
                arr.call_method1("reshape", ((self.shape_val.0, 1i32),))
            }
            None => {
                let all_sums =
                    detached(py, || self.streaming_row_sums()).map_err(PyRuntimeError::new_err)?;
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
                    detached(py, || self.streaming_row_sums()).map_err(PyRuntimeError::new_err)?;
                let filtered = self.filter_row_results(&all_sums);
                let n = self.shape_val.1 as f64;
                let means: Vec<f64> = filtered.iter().map(|&s| s / n).collect();
                let arr = numpy::PyArray::from_vec(py, means);
                arr.call_method1("reshape", ((self.shape_val.0, 1i32),))
            }
            None => {
                let all_sums =
                    detached(py, || self.streaming_row_sums()).map_err(PyRuntimeError::new_err)?;
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
            // Decode + transform + project off the GIL; build scipy on-GIL.
            let csr = detached(py, || {
                self.backed
                    .read_rows(global_row as u64, global_row as u64 + 1)
                    .map(|mut csr| {
                        self.apply_transforms(&mut csr, global_row);
                        self.apply_col_projection(csr)
                    })
                    .map_err(|e| e.to_string())
            })
            .map_err(PyRuntimeError::new_err)?;
            return csr_to_scipy(py, csr);
        }

        // Slice index
        if let Ok(slice) = row_idx.cast::<PySlice>() {
            let indices = slice.indices(self.shape_val.0 as isize)?;
            let start = indices.start.max(0) as u64;
            let stop = indices.stop.max(0) as u64;
            let step = indices.step;

            if step == 1 && self.kept_to_global.is_none() {
                // Contiguous slice, no deletions — direct range read + transform.
                // Decode + transform + project off the GIL; build scipy on-GIL.
                let csr = detached(py, || {
                    self.backed
                        .read_rows(start, stop)
                        .map(|mut csr| {
                            self.apply_transforms(&mut csr, start as usize);
                            self.apply_col_projection(csr)
                        })
                        .map_err(|e| e.to_string())
                })
                .map_err(PyRuntimeError::new_err)?;
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

            // Decode + per-row transform + project off the GIL; scipy on-GIL.
            let csr = detached(py, || {
                self.backed
                    .read_row_indices(&rows)
                    .map(|mut csr| {
                        // Apply transforms row-by-row with correct global offsets.
                        self.apply_transforms_per_row(&mut csr, &rows);
                        self.apply_col_projection(csr)
                    })
                    .map_err(|e| e.to_string())
            })
            .map_err(PyRuntimeError::new_err)?;
            return csr_to_scipy(py, csr);
        }

        // Numpy array or list
        let np = py.import("numpy")?;
        let arr = np.call_method1("asarray", (row_idx,))?;
        let dtype_str: String = arr.getattr("dtype")?.call_method0("__str__")?.extract()?;

        if dtype_str == "bool" {
            // Boolean mask → extract True indices
            let nonzero = arr.call_method0("nonzero")?;
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
            // Decode + per-row transform + project off the GIL; scipy on-GIL.
            let csr = detached(py, || {
                self.backed
                    .read_row_indices(&rows)
                    .map(|mut csr| {
                        self.apply_transforms_per_row(&mut csr, &rows);
                        self.apply_col_projection(csr)
                    })
                    .map_err(|e| e.to_string())
            })
            .map_err(PyRuntimeError::new_err)?;
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
        // Decode + per-row transform + project off the GIL; scipy on-GIL.
        let csr = detached(py, || {
            self.backed
                .read_row_indices(&rows)
                .map(|mut csr| {
                    self.apply_transforms_per_row(&mut csr, &rows);
                    self.apply_col_projection(csr)
                })
                .map_err(|e| e.to_string())
        })
        .map_err(PyRuntimeError::new_err)?;
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
                .with_csc_reader(self.backed_csc.clone())
                .with_source_path(self.source_path.clone());
                return Ok(new_ds.into_pyobject(py)?.into_any().unbind().into_bound(py));
            }
        }

        // Get the full row selection first
        let row_csr = self.getitem_rows(py, row_idx)?;

        // Check if col_idx is a full slice (`:`)
        if let Ok(slice) = col_idx.cast::<PySlice>() {
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
            Transform::Scale { factor } => {
                for v in &mut csr.data[start..end] {
                    *v = (*v as f64 * *factor) as f32;
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
