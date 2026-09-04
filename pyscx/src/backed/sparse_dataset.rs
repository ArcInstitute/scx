// ScxBackedSparseDataset — PyO3 class for on-demand sparse access.
//
// Extracted from the former pyscx/src/backed.rs (T5.7).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};

use scx_format_io::{BackedCscReader, BackedCsrReader};

use crate::convert::csr_to_scipy;
use crate::lazy_transform::Transform;
use scx_engine::projection::project_csr;

use crate::projected_agg;

use super::*;

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
    /// If deletions are present, maps user-visible row i → global row index.
    /// When None, no remapping is needed (no deletions).
    /// Arc-wrapped to avoid O(n) deep clones when creating lazy datasets.
    pub(crate) kept_to_global: Option<Arc<Vec<u64>>>,
    /// If column projection is active, sorted column indices to retain.
    /// CSR outputs are filtered through `project_csr()` before returning.
    /// Arc-wrapped to avoid O(n) deep clones when creating lazy datasets.
    col_projection: Option<Arc<Vec<u32>>>,
    /// If the visible columns should be presented in a caller-requested
    /// order (`preserve_var_order`, `adata[:, [7, 2, 11]]`, `X[:, [7, 2, 11]]`),
    /// this is the presentation **permutation** over the (sorted)
    /// `col_projection`: output column `k` is taken from sorted-projection
    /// column `col_presentation[k]`. `None` ⇒ identity (the default sorted
    /// order). Always `None` unless `col_projection` is also `Some`; length
    /// equals `col_projection.len()`. A permutation cannot express a
    /// *repeated* column, so a selector with repeats never becomes a handle:
    /// `__getitem__` materialises the projected unique columns and gathers.
    col_presentation: Option<Arc<Vec<u32>>>,
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
    pub fn from_reader(backed: Arc<BackedCsrReader>) -> Self {
        let shape_val = backed.shape();
        let n_shards = backed.index().n_shards();
        ScxBackedSparseDataset {
            backed,
            backed_csc: None,
            shape_val,
            n_shards,
            kept_to_global: None,
            col_projection: None,
            col_presentation: None,
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
            kept_to_global: Some(Arc::new(kept_to_global)),
            col_projection: None,
            col_presentation: None,
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
    pub fn as_column_source(&self) -> Option<&dyn scx_format_io::ColumnShardSource> {
        if self.kept_to_global.is_some() {
            return None;
        }
        let backed_csc = self.backed_csc.as_ref()?;
        Some(backed_csc.as_ref() as &dyn scx_format_io::ColumnShardSource)
    }

    /// [`Self::as_column_source`] as an **owned** handle, under the identical
    /// gate.
    ///
    /// The borrowed form is tied to the `PyRef` it came from, so it cannot
    /// cross a `py.detach(...)` boundary. Callers that release the GIL for the
    /// streaming scan — `pyscx.accel.col_*` with `prefer_format="csc"` — take
    /// this instead and get an `Arc` they can move into the detached closure.
    ///
    /// Keep the two gates in step: an `Arc` handed out here bypasses nothing,
    /// but if the deletion-vector condition above ever grows a clause, this
    /// must grow it too.
    pub(crate) fn as_column_source_owned(&self) -> Option<Arc<BackedCscReader>> {
        if self.kept_to_global.is_some() {
            return None;
        }
        self.backed_csc.as_ref().map(Arc::clone)
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
        // A plain (sorted) projection has no presentation reorder.
        self.col_presentation = None;
    }

    /// Set column projection while preserving the caller's request order.
    ///
    /// `ordered` are original column indices in request order; duplicates
    /// are dropped keeping the first occurrence. The gather still runs over
    /// the sorted set (`col_projection`); the visible columns are then
    /// reordered via `col_presentation` so they follow `ordered`.
    pub fn set_col_projection_ordered(&mut self, ordered: Vec<u32>) {
        // Dedup preserving first occurrence -> request order R.
        let mut seen = std::collections::HashSet::new();
        let request: Vec<u32> = ordered.into_iter().filter(|&i| seen.insert(i)).collect();

        let mut sorted = request.clone();
        sorted.sort_unstable();
        // (already unique)

        // perm[k] = position of request[k] within sorted.
        let perm: Vec<u32> = request
            .iter()
            .map(|g| sorted.partition_point(|&s| s < *g) as u32)
            .collect();
        let is_identity = perm.iter().enumerate().all(|(k, &p)| k as u32 == p);

        self.shape_val.1 = sorted.len();
        self.col_projection = Some(Arc::new(sorted));
        self.col_presentation = if is_identity {
            None
        } else {
            Some(Arc::new(perm))
        };
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

    /// Clone the col_presentation Arc (O(1) ref-count increment).
    pub(crate) fn col_presentation_arc(&self) -> Option<Arc<Vec<u32>>> {
        self.col_presentation.clone()
    }

    /// On-disk column indices for every visible column, in presentation
    /// order. Used by gene-filtering composition so request order survives
    /// further `filter_genes` / HVG selection. Requires `col_projection`
    /// to be set.
    pub(crate) fn visible_ondisk_in_presentation_order(&self) -> Option<Vec<u32>> {
        let proj = self.col_projection.as_ref()?;
        Some(match &self.col_presentation {
            Some(perm) => perm.iter().map(|&p| proj[p as usize]).collect(),
            None => proj.as_ref().clone(),
        })
    }

    /// Apply column projection to a CSR matrix if projection is active.
    /// Returns the original CSR if no projection is set. When a
    /// presentation order is active, columns are reordered after the
    /// (sorted) gather to follow the caller's requested order.
    pub(crate) fn apply_col_projection(&self, csr: scx_sparse::ScxCsr) -> scx_sparse::ScxCsr {
        match &self.col_projection {
            Some(indices) => {
                let projected = project_csr(&csr, indices);
                match &self.col_presentation {
                    Some(perm) => scx_engine::projection::reorder_csr_columns(&projected, perm),
                    None => projected,
                }
            }
            None => csr,
        }
    }

    /// This handle's **view** as a streaming [`ShardSource`].
    ///
    /// `kept_to_global` and `col_projection` are folded in, and `n_obs` /
    /// `n_vars` report the *visible* widths — so a kernel consuming this sees
    /// exactly the matrix `adata.X` presents, and any per-cell / per-gene array
    /// it produces lines up with `adata.obs` / `adata.var`.
    ///
    /// Reach for this, never for `&*self.backed`: the raw reader is *the file*,
    /// not *the view*. It streams every on-disk row and column, silently
    /// ignoring both fields — which is how PCA came to attribute each cell's
    /// embedding to the wrong cell after a `filter_cells`.
    ///
    /// Mirrors [`crate::lazy_transform::ScxLazyTransformedDataset::as_shard_source`],
    /// with an empty transform chain. Multi-pass kernels (out-of-core PCA)
    /// should chain
    /// [`with_cached_reads`](crate::lazy_transform::LazyShardSource::with_cached_reads)
    /// to keep serving from the reader's decoded-shard LRU.
    ///
    /// `col_presentation` is **not** representable here (the source emits
    /// columns in sorted-projection order). Callers must first reject a
    /// presentation-ordered handle via
    /// [`crate::accel::reject_preserve_var_order`].
    pub(crate) fn as_shard_source(&self) -> crate::lazy_transform::LazyShardSource {
        crate::lazy_transform::LazyShardSource::new(
            Arc::clone(&self.backed),
            Vec::new(),
            self.kept_to_global.clone(),
            self.col_projection.clone(),
            self.shape_val.0,
            self.shape_val.1,
        )
    }

    /// Whether this handle is a strict *window* onto the file rather than the
    /// whole of it — i.e. some axis has been subset.
    ///
    /// Lets a call site keep passing the raw reader in the (overwhelmingly
    /// common) unsubset case, where the reader and the view are the same
    /// matrix, and reach for [`Self::as_shard_source`] only when they differ.
    /// That matters where the concrete reader type unlocks something a
    /// `ShardSource` cannot express — GPU DE's CSC-direct route, which takes
    /// `GpuDeShardInput::Backed { csr, csc }`.
    ///
    /// On that route the switch is **load-bearing**: the `csc` handed to
    /// `Backed` is read straight off `backed_csc`, so it never passes through
    /// `as_column_source()`'s deletion gate, and `csc_route_available` is not
    /// consulted on GPU at all. Without this predicate a subset handle with a
    /// sidecar runs the CSC-direct kernel against *on-disk* columns, and the
    /// widths agree, so nothing catches it.
    ///
    /// Only the GPU DE dispatch needs this today — the CPU kernels are all
    /// generic over `ShardSource` and take the view unconditionally.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub(crate) fn has_axis_view(&self) -> bool {
        self.kept_to_global.is_some() || self.col_projection.is_some()
    }

    /// A second handle onto the same file, same window. O(1) — every heavy
    /// field is behind an `Arc`.
    pub(crate) fn clone_handle(&self) -> Self {
        ScxBackedSparseDataset {
            backed: Arc::clone(&self.backed),
            backed_csc: self.backed_csc.clone(),
            shape_val: self.shape_val,
            n_shards: self.n_shards,
            kept_to_global: self.kept_to_global.clone(),
            col_projection: self.col_projection.clone(),
            col_presentation: self.col_presentation.clone(),
            non_negative: self.non_negative,
            modality_id: self.modality_id,
            source_path: self.source_path.clone(),
        }
    }

    /// A handle onto a sub-window of this one, composing rather than reading.
    ///
    /// Backs `anndata._core.index._subset`. `rows` / `cols` are positional
    /// indices into the **visible** axes; `None` means "the whole axis". The
    /// composed column map is in presentation order, so it goes in through
    /// [`Self::set_col_projection_ordered`] — that is what keeps
    /// `preserve_var_order` alive across `adata[:, idx]`.
    pub(crate) fn subset_clone(
        &self,
        rows: Option<&[i64]>,
        cols: Option<&[i64]>,
    ) -> PyResult<Self> {
        let mut out = self.clone_handle();
        if let Some(rows) = rows {
            let composed = crate::axis_align::compose_rows_positional(
                self.kept_to_global.as_ref().map(|v| v.as_slice()),
                rows,
                self.shape_val.0,
            )?;
            // An identity map is not a subset. Installing one anyway would set
            // `kept_to_global`, and *any* `kept_to_global` closes the CSC
            // capability gate (`as_column_source` returns `None`) — permanently
            // downgrading the `gpu_csc_v3` CSC-direct DE route on a file that
            // was never really subset. The mutating ops guard this with an
            // all-kept early return; this covers a caller that reaches
            // `_subset` directly, e.g. `adata[np.arange(n_obs)]`.
            if !crate::axis_align::is_identity_rows(
                &composed,
                self.shape_val.0,
                self.kept_to_global.is_none(),
            ) {
                out.set_kept_to_global(composed);
            }
        }
        if let Some(cols) = cols {
            let base = self.visible_ondisk_in_presentation_order();
            out.set_col_projection_ordered(crate::axis_align::compose_cols_positional(
                base.as_deref(),
                cols,
                self.shape_val.1,
            )?);
        }
        Ok(out)
    }

    /// Reorder a per-visible-column vector (in sorted-projection order) into
    /// presentation order. No-op when no presentation reorder is active.
    pub(crate) fn present_reorder<T: Clone>(&self, values: Vec<T>) -> Vec<T> {
        match &self.col_presentation {
            Some(perm) => perm.iter().map(|&p| values[p as usize].clone()).collect(),
            None => values,
        }
    }
}

#[pymethods]
impl ScxBackedSparseDataset {
    /// Set column projection from Python. Restricts aggregation and access to
    /// a subset of columns. `col_indices` are the original (0-based) column indices.
    /// Shape is adjusted: n_vars becomes len(col_indices).
    #[pyo3(name = "set_col_projection")]
    pub(crate) fn py_set_col_projection(&mut self, col_indices: Vec<u32>) {
        self.set_col_projection(col_indices);
    }

    /// Answered from a scalar cached at construction, so it never reaches
    /// `section_bytes` and would otherwise keep reporting the row count the
    /// file had before an `append` or a `mark_deleted`.
    #[getter]
    pub(crate) fn shape(&self) -> PyResult<(usize, usize)> {
        self.backed.check_fresh().map_err(crate::to_pyerr)?;
        Ok(self.shape_val)
    }

    /// Phase B.5: optional modality_id tag. `None` for legacy
    /// single-modality / global datasets; `Some(id)` for datasets
    /// constructed via `ScxBackedMuDataset.mod[name]`.
    #[getter]
    pub(crate) fn modality_id(&self) -> Option<u8> {
        self.modality_id
    }

    /// Always `float32`: the type every read decodes to (scipy CSR interop,
    /// anndata's `CSRDataset` expectations). The on-disk value encoding is
    /// [`Self::stored_dtype`].
    #[getter]
    pub(crate) fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = crate::pyimport::import_module(py, "numpy")?;
        np.call_method1("dtype", ("float32",))
    }

    /// The on-disk value encoding of this handle's shards as a `numpy.dtype`
    /// — `uint8` / `uint16` / `uint32` for integer counts, `float32` /
    /// `float16` for continuous data. When shards mix it is the widest (any
    /// float ⇒ `float32`, else the widest integer); a file with no shards
    /// reports `float32`. Reads one 76-byte header per shard, decodes nothing;
    /// `stored_dtype.kind in "ui"` answers "are these counts?" in O(shards).
    #[getter]
    pub(crate) fn stored_dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        stored_dtype_object(py, &self.backed)
    }

    /// The decoded-shard LRU size this handle's reader was built with
    /// (`to_anndata(backed=True, cache_shards=…)`); `0` means every read
    /// decodes afresh. Read-only: the count is fixed when the reader is built,
    /// and one `to_anndata` call builds `X` and each layer with the same count.
    #[getter]
    pub(crate) fn cache_shards(&self) -> usize {
        self.backed.cache_shards()
    }

    /// The numpy array protocol — **refused**. A handle is a window onto a
    /// file, and `np.asarray(handle)` would decode `n_obs × n_vars` at once;
    /// at atlas scale that is the worse failure, so it raises `TypeError`
    /// naming the explicit paths (`to_memory()`, `toarray()`, a slice).
    /// Before this it returned a 0-d object array that failed far away.
    #[pyo3(signature = (dtype=None, copy=None))]
    pub(crate) fn __array__<'py>(
        &self,
        _py: Python<'py>,
        dtype: Option<Bound<'py, PyAny>>,
        copy: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = (dtype, copy);
        Err(no_implicit_array_error("ScxBackedSparseDataset"))
    }

    /// Returns "csr" to match the anndata ABC format ClassVar.
    #[getter]
    pub(crate) fn format(&self) -> &str {
        "csr"
    }

    #[getter]
    pub(crate) fn backend(&self) -> &str {
        "scx"
    }

    // ndim and __len__ are NOT part of the anndata ABC but are
    // needed for scipy/numpy interop in practice.
    #[getter]
    pub(crate) fn ndim(&self) -> usize {
        2
    }

    /// Whether the data is known to be non-negative.
    ///
    /// Raw count data is always non-negative. This flag is used by
    /// `ScxComparisonResult` to enable the `(X > 0).sum() → getnnz()`
    /// short-circuit optimization.
    #[getter]
    pub(crate) fn non_negative(&self) -> bool {
        self.non_negative
    }

    pub(crate) fn __len__(&self) -> PyResult<usize> {
        self.backed.check_fresh().map_err(crate::to_pyerr)?;
        Ok(self.shape_val.0)
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "ScxBackedSparseDataset(shape=({}, {}), n_shards={}, cache_shards={})",
            self.shape_val.0,
            self.shape_val.1,
            self.n_shards,
            self.backed.cache_shards()
        )
    }

    /// Load a slice from disk, or project columns without loading.
    ///
    /// Supports:
    /// - Row slicing:      X[100:200]       → csr_matrix
    /// - Row + col slice:  X[100:200, :500] → csr_matrix
    /// - Boolean mask:     X[mask]          → csr_matrix
    /// - Fancy indexing:   X[[0, 5, 10]]    → csr_matrix
    /// - Scalar indexing:  X[0, 5]          → float
    /// - Integer index:    X[5]             → csr_matrix (single row)
    /// - Column selector:  X[:, 5] / X[:, [7, 2, 11]] / X[:, 10:20] / X[:, mask]
    ///                     → a projected `ScxBackedSparseDataset` (no decode);
    ///                     repeated columns (`X[:, [3, 1, 3]]`) → csr_matrix
    ///                     built from the projected unique columns
    pub(crate) fn __getitem__<'py>(
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
    pub(crate) fn to_memory<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // B2: decode off the GIL; build the scipy array after.
        let csr = detached(py, || {
            if self.col_projection.is_some() {
                // A projected handle assembles shard by shard: each shard is
                // decoded once and projected (and row-filtered) while it is
                // still one shard wide, then the narrow pieces are concatenated
                // — peak = 2× the *projected* result + one shard, never the
                // whole matrix. `as_shard_source` folds `kept_to_global` and
                // `col_projection`; it cannot carry `col_presentation`, so
                // `materialize_projected` applies the permutation to each piece
                // before concatenating (never to the concatenated result — that
                // would hold a third result-sized buffer).
                materialize_projected(self).map_err(|e| e.to_string())
            } else if let Some(ref kept) = self.kept_to_global {
                // Deletion vectors: gather the kept rows rather than every row.
                self.backed
                    .read_row_indices(kept)
                    .map_err(|e| e.to_string())
            } else {
                self.backed.read_all().map_err(|e| e.to_string())
            }
        })
        .map_err(PyRuntimeError::new_err)?;
        csr_to_scipy(py, csr)
    }

    // --- Scipy compatibility (NOT part of anndata ABC) ---

    /// Materialize as dense numpy array.
    pub(crate) fn toarray<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method0("toarray")
    }

    /// Materialize as scipy CSR. Same as to_memory().
    pub(crate) fn tocsr<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.to_memory(py)
    }

    /// Materialize and convert to CSC.
    pub(crate) fn tocsc<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method0("tocsc")
    }

    /// Dense array property (scipy compat).
    #[getter]
    #[allow(non_snake_case)]
    pub(crate) fn A<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.toarray(py)
    }

    /// Copy — materializes the full matrix. Required by AnnData .copy().
    ///
    /// Deliberately *not* a lazy clone. `AnnData.copy()` on a view runs
    /// `_subset(ref.X, idx).copy()`, so this is what makes the documented
    /// `adata[mask].copy()` → "subset, materialize, then run scanpy" workflow
    /// mean what it says. The in-place accelerators stay out-of-core by
    /// building their replacement through `_mutated_copy(X=view.X, …)`, which
    /// never calls `.copy()` on the matrix — see [`crate::axis_align`].
    pub(crate) fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method0("copy")
    }

    // --- Comparison operators (lazy, with fused optimization) ---
    // These return a _ComparisonResult wrapper that short-circuits
    // `.sum()` → `getnnz()` for the `(X > 0).sum(axis=1)` pattern
    // used by scanpy's calculate_qc_metrics, filter_cells, filter_genes.

    pub(crate) fn __gt__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.make_comparison_result(py, "gt", other)
    }

    pub(crate) fn __ge__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.make_comparison_result(py, "ge", other)
    }

    pub(crate) fn __lt__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.make_comparison_result(py, "lt", other)
    }

    pub(crate) fn __le__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.make_comparison_result(py, "le", other)
    }

    pub(crate) fn __eq__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.make_comparison_result(py, "eq", other)
    }

    pub(crate) fn __ne__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.make_comparison_result(py, "ne", other)
    }

    // --- Arithmetic operators ---
    // Used by scanpy's normalize_total (multiply), scale, etc.

    pub(crate) fn __add__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.add(other)
    }

    pub(crate) fn __sub__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.sub(other)
    }

    pub(crate) fn __mul__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Try to extract a per-row scaling vector.  If the multiplier is
        // a 1D or column array with length == n_obs, express as lazy
        // RowScale transform — mirroring __truediv__ (which inverts).
        // (Skipped when a presentation reorder is active: the lazy transform
        // can't carry it, so we fall back to materialization below.)
        if self.col_presentation.is_none() {
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
        }

        // Cannot be expressed as row scaling — fall back to materialization
        let mat = self.to_memory(py)?;
        mat.call_method1("__mul__", (other,))
    }

    pub(crate) fn __truediv__<'py>(
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
        if self.col_presentation.is_none() {
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
        }

        // Cannot be expressed as row scaling — fall back to materialization
        let mat = self.to_memory(py)?;
        mat.call_method1("__truediv__", (other,))
    }

    pub(crate) fn __rmul__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Multiplication is commutative — delegate to __mul__.
        self.__mul__(py, other)
    }

    pub(crate) fn __rtruediv__<'py>(
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

    pub(crate) fn __matmul__<'py>(
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
    pub(crate) fn sum<'py>(
        &self,
        py: Python<'py>,
        axis: Option<i32>,
    ) -> PyResult<Bound<'py, PyAny>> {
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
    pub(crate) fn mean<'py>(
        &self,
        py: Python<'py>,
        axis: Option<i32>,
    ) -> PyResult<Bound<'py, PyAny>> {
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
    pub(crate) fn var<'py>(
        &self,
        py: Python<'py>,
        axis: Option<i32>,
    ) -> PyResult<Bound<'py, PyAny>> {
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
                                // Conditional, not `.max(0.0)`: see the note on
                                // the unprojected scalar arm below. `f64::max`
                                // ignores NaN, which would turn a NaN variance
                                // into a real-looking 0.0.
                                let c = sq / n_proj - mean * mean;
                                if c < 0.0 {
                                    0.0
                                } else {
                                    c
                                }
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
                    // Conditional, not `.max(0.0)` — NaN must survive; see below.
                    let centered = sumsq.iter().sum::<f64>() / n_total - mean * mean;
                    let variance = if centered < 0.0 { 0.0 } else { centered };
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
                // E[X²] − E[X]² is exact only for canonical data: a duplicated
                // coordinate inflates both moments unevenly and can drive this
                // negative (measured: shape (1,2) storing [1,2,3] gives
                // 7 − 9 = −2.0). This form never subtracted a count so it never
                // wrapped, but a negative variance is not a defensible answer
                // either. It is a clamp, not a detector — see
                // `scx_sparse::implicit_zero_count` § "What this does NOT
                // detect".
                //
                // ⚠️ Use the conditional, NOT `.max(0.0)`. Rust's `f64::max`
                // *ignores* NaN and returns the other operand, so `.max(0.0)`
                // silently converts a NaN variance to `0.0`. NaN in `X` is
                // supported input (the dense h5ad streamer deliberately
                // preserves it), and a NaN there must stay NaN rather than be
                // reported as a real zero variance. `NaN < 0.0` is false, so
                // this form passes it through. Same shape as the CSC kernels.
                let centered = mean_sq - mean * mean;
                let variance = if centered < 0.0 { 0.0 } else { centered };
                Ok(variance.into_pyobject(py)?.into_any())
            }
            Some(_) => Err(PyValueError::new_err("axis must be 0, 1, or None")),
        }
    }

    /// NNZ counts along an axis without materializing the full matrix.
    /// When column projection is active, uses streaming projected aggregation.
    #[pyo3(signature = (axis=None))]
    pub(crate) fn getnnz<'py>(
        &self,
        py: Python<'py>,
        axis: Option<i32>,
    ) -> PyResult<Bound<'py, PyAny>> {
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
    pub(crate) fn multiply<'py>(
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
    pub(crate) fn power<'py>(
        &self,
        py: Python<'py>,
        n: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mat = self.to_memory(py)?;
        mat.call_method1("power", (n,))
    }

    /// Number of stored values (nonzeros) — without materializing.
    /// Respects deletion vectors and column projections.
    ///
    /// Delegates to `total_nnz_raw()` (the same routine `getnnz(axis=None)`
    /// uses) so the projected per-column counts are summed as `u64` — a plain
    /// `Vec<u32>` sum overflows once projected nnz exceeds `u32::MAX` (atlas
    /// scale).
    #[getter]
    pub(crate) fn nnz(&self) -> PyResult<usize> {
        self.total_nnz_raw().map_err(PyRuntimeError::new_err)
    }

    /// Number of CSR shards in the backing file.
    #[getter]
    pub(crate) fn n_shards(&self) -> usize {
        self.n_shards
    }

    /// Return shard boundaries as a list of (row_start, row_end) tuples.
    ///
    /// When deletion vectors are present, the boundaries are remapped to
    /// user-visible row space (i.e., deleted rows are excluded from counts).
    /// Each tuple represents a contiguous chunk of user-visible rows that
    /// came from one on-disk shard.
    ///
    /// **Tiling contract** — `pyscx.iter_chunks(chunk_size="shard")` relies on
    /// it, and `test_chunk_iterator.py` pins it: `b[0].0 == 0`,
    /// `b.last().1 == n_obs` (visible), and `b[i].0 == b[i-1].1`, i.e. the
    /// pairs tile `[0, n_obs)` exactly with no gaps or overlap. A shard whose
    /// rows are all deleted is omitted, so `len(b)` may be below `n_shards`.
    pub(crate) fn shard_boundaries(&self) -> Vec<(usize, usize)> {
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
    pub(crate) fn max<'py>(
        &self,
        py: Python<'py>,
        axis: Option<i32>,
    ) -> PyResult<Bound<'py, PyAny>> {
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
    pub(crate) fn min<'py>(
        &self,
        py: Python<'py>,
        axis: Option<i32>,
    ) -> PyResult<Bound<'py, PyAny>> {
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
    pub(crate) fn col_sums_raw(&self) -> Result<Vec<f64>, String> {
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
        .map(|v| self.present_reorder(v))
    }

    /// Per-row sums (`axis=1` / total), honoring column projection. Deletion
    /// remapping is applied by the caller via `filter_row_results`.
    pub(crate) fn row_sums_raw(&self) -> Result<Vec<f64>, String> {
        if let Some(ref cols) = self.col_projection {
            projected_agg::row_sums_projected(&self.backed, cols).map_err(|e| e.to_string())
        } else {
            self.backed.row_sums().map_err(|e| e.to_string())
        }
    }

    /// Column nnz counts (`axis=0`), honoring column projection and keep-mask.
    pub(crate) fn col_nnz_raw(&self) -> Result<Vec<u32>, String> {
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
        .map(|v| self.present_reorder(v))
    }

    /// Fused column sums + nnz (`axis=0`), honoring column projection and
    /// keep-mask. One shard scan in place of [`Self::col_sums_raw`] +
    /// [`Self::col_nnz_raw`]; bit-identical to calling both.
    pub(crate) fn col_sums_and_nnz_raw(&self) -> Result<(Vec<f64>, Vec<u32>), String> {
        match (&self.col_projection, &self.kept_to_global) {
            (Some(cols), Some(kept)) => {
                projected_agg::col_sums_and_nnz_masked_projected(&self.backed, kept, cols)
                    .map_err(|e| e.to_string())
            }
            (Some(cols), None) => projected_agg::col_sums_and_nnz_projected(&self.backed, cols)
                .map_err(|e| e.to_string()),
            (None, Some(kept)) => self
                .backed
                .col_sums_and_nnz_masked(kept)
                .map_err(|e| e.to_string()),
            (None, None) => self.backed.col_sums_and_nnz().map_err(|e| e.to_string()),
        }
        .map(|(sums, counts)| (self.present_reorder(sums), self.present_reorder(counts)))
    }

    /// Fused per-cell QC pass: row nnz, row sums and per-`qc_var` subset sums
    /// over the visible columns, in a single shard scan.
    ///
    /// `qc_bits` is indexed by visible column (bit *k* ⇒ member of subset *k*);
    /// see [`projected_agg::qc_row_pass`]. Row vectors are global-length —
    /// apply [`Self::filter_row_results`] for the visible rows.
    pub(crate) fn qc_row_pass_raw(
        &self,
        qc_bits: &[u64],
        n_qc: usize,
    ) -> Result<projected_agg::QcRowStats, String> {
        projected_agg::qc_row_pass(
            &self.backed,
            self.col_projection.as_ref().map(|c| c.as_slice()),
            qc_bits,
            n_qc,
        )
        .map_err(|e| e.to_string())
    }

    /// Per-row nnz counts (`axis=1`), honoring column projection.
    pub(crate) fn row_nnz_raw(&self) -> Result<Vec<i64>, String> {
        if let Some(ref cols) = self.col_projection {
            projected_agg::row_nnz_projected(&self.backed, cols).map_err(|e| e.to_string())
        } else {
            self.backed.row_nnz().map_err(|e| e.to_string())
        }
    }

    /// Scalar total nnz (`getnnz(axis=None)`), honoring column projection and
    /// keep-mask. The masked/projected arms decode shards, so this runs through
    /// `detached` at the call site like the per-axis kernels.
    pub(crate) fn total_nnz_raw(&self) -> Result<usize, String> {
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
    pub(crate) fn col_var_raw(&self) -> Result<Vec<f64>, String> {
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
        .map(|v| self.present_reorder(v))
    }

    /// Column maxima (`axis=0`), honoring column projection and keep-mask.
    pub(crate) fn col_max_raw(&self) -> Result<Vec<f64>, String> {
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
        .map(|v| self.present_reorder(v))
    }

    /// Column minima (`axis=0`), honoring column projection and keep-mask.
    pub(crate) fn col_min_raw(&self) -> Result<Vec<f64>, String> {
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
        .map(|v| self.present_reorder(v))
    }

    /// Create a lazy comparison result wrapper.
    ///
    /// If the threshold can be extracted as a numeric f64, returns a
    /// `ScxComparisonResult` that can short-circuit `.sum()` → `getnnz()`.
    /// Otherwise falls back to immediate materialization.
    pub(crate) fn make_comparison_result<'py>(
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
                col_projection: self.col_projection.clone(),
                col_presentation: self.col_presentation.clone(),
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
    pub(crate) fn getitem_rows<'py>(
        &self,
        py: Python<'py>,
        row_idx: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Integer index → single row
        if let Ok(i) = row_idx.extract::<i64>() {
            let row = self.normalize_row_index(i)?;
            let global_row = self.to_global_row(row)?;
            // Decode + project off the GIL (P1); build the scipy object on-GIL.
            let csr = detached(py, || {
                self.backed
                    .read_rows(global_row as u64, global_row as u64 + 1)
                    .map(|csr| self.apply_col_projection(csr))
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
                // Contiguous slice, no deletions — direct range read.
                // Decode + project off the GIL (P1).
                let csr = detached(py, || {
                    self.backed
                        .read_rows(start, stop)
                        .map(|csr| self.apply_col_projection(csr))
                        .map_err(|e| e.to_string())
                })
                .map_err(PyRuntimeError::new_err)?;
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
            // already handled above), use read_row_indices for correctness.
            // Decode + project off the GIL (P1).
            let csr = detached(py, || {
                self.backed
                    .read_row_indices(&rows)
                    .map(|csr| self.apply_col_projection(csr))
                    .map_err(|e| e.to_string())
            })
            .map_err(PyRuntimeError::new_err)?;
            return csr_to_scipy(py, csr);
        }

        // Numpy array or list — a boolean mask (length-checked) or an integer
        // array-like (negative wrap, `IndexError` on out-of-range): one shared
        // resolver, then the bounded gather. `read_row_indices` decodes each
        // touched shard once and assembles the result in place, in request
        // order, duplicates included.
        let visible = resolve_row_selector(py, row_idx, self.shape_val.0)?;
        let rows: Vec<u64> = visible
            .iter()
            .map(|&v| self.to_global_row(v).map(|g| g as u64))
            .collect::<PyResult<Vec<u64>>>()?;
        // Decode + project off the GIL (P1).
        let csr = detached(py, || {
            self.backed
                .read_row_indices(&rows)
                .map(|csr| self.apply_col_projection(csr))
                .map_err(|e| e.to_string())
        })
        .map_err(PyRuntimeError::new_err)?;
        csr_to_scipy(py, csr)
    }

    /// Handle 2D indexing (rows, cols).
    pub(crate) fn getitem_2d<'py>(
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
            // Read single row, extract single column. Decode off the GIL (P1)
            // — the cost is a full shard decode even for one element.
            let global_row = self.to_global_row(row)?;
            let csr = detached(py, || {
                self.backed
                    .read_rows(global_row as u64, global_row as u64 + 1)
                    .map_err(|e| e.to_string())
            })
            .map_err(PyRuntimeError::new_err)?;
            // When col_projection is active, remap user-visible col to on-disk col.
            // With a presentation reorder, the visible column maps through the
            // permutation first (visible → sorted-projection position → on-disk).
            let lookup_col = if let Some(ref proj) = self.col_projection {
                let sorted_pos = match &self.col_presentation {
                    Some(perm) => *perm.get(col).ok_or_else(|| {
                        PyIndexError::new_err(format!(
                            "column index {} out of range for {} projected columns",
                            col,
                            perm.len()
                        ))
                    })? as usize,
                    None => col,
                };
                *proj.get(sorted_pos).ok_or_else(|| {
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
        // `X[:, cols]`: every column selector form (int, list, range, slice,
        // int / bool ndarray, any order) is resolved once and composed through
        // the current window by `subset_clone`, so a projected handle comes
        // back with no decode — a reordered request rides on the presentation
        // permutation, which is also how `adata[:, [7, 2, 11]]` arrives here,
        // so composing on top of an existing presentation is exact. A request
        // with *repeated* columns cannot be a handle (the presentation is a
        // permutation); it materialises the projected unique columns and
        // gathers them with scipy — peak is the result, never the whole
        // matrix. `X[:, :]` resolves to `None` and falls through to return the
        // row CSR, like `X[:]`.
        if is_full_slice(row_idx, self.shape_val.0)? {
            if let Some(sel) = resolve_col_request(py, col_idx, self.shape_val.1)? {
                let sel_i64: Vec<i64> = sel.iter().map(|&c| c as i64).collect();
                let handle = self.subset_clone(None, Some(&sel_i64))?;
                // `set_col_projection_ordered` dedups, so the visible width is the
                // number of distinct columns: equal to the request length ⇔ no
                // repeats, and the handle is the answer with nothing else built.
                if handle.shape_val.1 == sel.len() {
                    return Ok(handle.into_pyobject(py)?.into_any());
                }
                // Repeats: the handle's visible order is the request's distinct
                // columns in order of first appearance, so `remap[k]` is the
                // position of `sel[k]` among those — `mat[:, remap]` rebuilds
                // the request from the projected read.
                let mut first_pos: std::collections::HashMap<usize, usize> =
                    std::collections::HashMap::with_capacity(handle.shape_val.1);
                let remap: Vec<usize> = sel
                    .iter()
                    .map(|&c| {
                        let next = first_pos.len();
                        *first_pos.entry(c).or_insert(next)
                    })
                    .collect();
                return scipy_column_gather(py, &handle.to_memory(py)?, &remap);
            }
        }

        // Get the full row selection first
        let row_csr = self.getitem_rows(py, row_idx)?;

        // Check if col_idx is a full slice (`:`) — if so, return as-is
        if is_full_slice(col_idx, self.shape_val.1)? {
            return Ok(row_csr);
        }

        // Apply column selection: row_csr[:, col_idx]
        // scipy sparse needs slice(None) for "all rows", not Python None
        let builtins = crate::pyimport::import_module(py, "builtins")?;
        let slice_none = builtins.call_method1("slice", (py.None(),))?;
        let col_tuple = PyTuple::new(py, &[slice_none.unbind(), col_idx.clone().unbind()])?;
        row_csr.get_item(col_tuple)
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

    /// Map a user-visible row index to the global (file-level) row index.
    /// If no deletion vectors are present, this is the identity function.
    ///
    /// Returns `PyIndexError` if `user_row` is out of bounds.
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
}

/// `numpy.dtype` for a reader's on-disk value encoding — see
/// `ScxBackedSparseDataset::stored_dtype`. Shared by the three sparse handles
/// (the lazy one reports its *source*'s encoding, before transforms).
pub(crate) fn stored_dtype_object<'py>(
    py: Python<'py>,
    backed: &BackedCsrReader,
) -> PyResult<Bound<'py, PyAny>> {
    // The first call folds every shard header; run it with the GIL released
    // (a memo hit is an atomic load and costs nothing either way).
    let name = detached(py, || backed.stored_value_encoding())
        .map_err(crate::to_pyerr)?
        .map_or("float32", |enc| enc.numpy_name());
    let np = crate::pyimport::import_module(py, "numpy")?;
    np.call_method1("dtype", (name,))
}

/// The `TypeError` every sparse handle's `__array__` raises.
pub(crate) fn no_implicit_array_error(class_name: &str) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(format!(
        "{class_name} is a lazy handle onto an SCX file and does not convert to a \
         numpy array implicitly: np.asarray(handle) would decode the whole \
         n_obs × n_vars matrix at once. Call handle.to_memory() for scipy CSR or \
         handle.toarray() for a dense array (both work on a column-projected \
         handle such as X[:, genes] too), take a row window with handle[rows] \
         (a scipy CSR), or open the file with pyscx.open(path).to_anndata() for \
         an in-memory AnnData."
    ))
}

/// `to_memory()` for a handle with a column projection: one decode per
/// shard, projected — and, under a presentation order, reordered — while one
/// shard wide, then concatenated. Peak = the projected pieces + the
/// concatenated result (2× the projected result) + one shard's transient.
/// The reorder happens per piece, not on the concatenated result: a column
/// permutation commutes with row concatenation, and reordering afterwards
/// would hold a third result-sized buffer while the other two are still live
/// (measured at ~2.9× a plain `to_memory()` for `X[:, ::-1]`). Sequential on
/// purpose — decoding shards in parallel would hold every decoded shard at
/// once, which is the peak this exists to avoid.
fn materialize_projected(ds: &ScxBackedSparseDataset) -> scx_format_io::Result<scx_sparse::ScxCsr> {
    use scx_format_io::ShardSource;
    let source = ds.as_shard_source();
    let n_vars = source.n_vars();
    let mut pieces = Vec::with_capacity(source.n_shards());
    for shard_idx in 0..source.n_shards() {
        let piece = source.read_shard(shard_idx)?;
        pieces.push(match &ds.col_presentation {
            Some(perm) => scx_engine::projection::reorder_csr_columns(&piece, perm),
            None => piece,
        });
    }
    Ok(scx_sparse::concatenate_csr(&pieces, n_vars)?)
}
