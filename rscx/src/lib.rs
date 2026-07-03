use std::path::PathBuf;

use extendr_api::prelude::*;
use scx_engine::pipeline::{QueryPipeline, QueryResult};
use scx_format_io::ScxReader;

use crate::query::RGroupShardHandle;

mod accel;
mod backed;
mod harmony;
mod interop;
mod lazy;
mod lisi;
mod ops;
mod query;
mod util;

pub use backed::RBackedSparse;
pub use lazy::RLazyTransformed;
pub use query::{RQueryPipeline, RQueryResult};

// ── Phase B: Core Reader ────────────────────────────────────────

/// R class wrapping an SCX file handle (mmap-backed, lazy).
#[extendr]
pub struct ScxExperiment {
    reader: ScxReader,
    path: PathBuf, // stored for query() re-open
}

impl std::fmt::Debug for ScxExperiment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScxExperiment")
            .field("path", &self.path)
            .finish()
    }
}

impl ScxExperiment {
    /// Fallible body of `new` (kept off the `#[extendr]` surface so the public
    /// constructor can return `Robj` + `throw_on_err`; see B3).
    fn open_impl(path: &str) -> Result<Self> {
        let path_buf = PathBuf::from(path);
        let reader = ScxReader::open(&path_buf)
            .map_err(|e| Error::Other(format!("failed to open SCX file '{}': {}", path, e)))?;
        Ok(Self {
            reader,
            path: path_buf,
        })
    }

    /// Fallible body of `obs` (see B3 / `throw_on_err`).
    fn obs_impl(&self) -> Result<Robj> {
        let batch = self
            .reader
            .read_obs()
            .map_err(|e| Error::Other(e.to_string()))?;
        interop::record_batch_to_dataframe(&batch)
    }

    /// Fallible body of `var` (see B3 / `throw_on_err`).
    fn var_impl(&self) -> Result<Robj> {
        let batch = self
            .reader
            .read_var()
            .map_err(|e| Error::Other(e.to_string()))?;
        interop::record_batch_to_dataframe(&batch)
    }

    /// Fallible body of `x_matrix` (see B3 / `throw_on_err`).
    fn x_matrix_impl(&self) -> Result<Robj> {
        let csr = self
            .reader
            .read_all_csr_shards()
            .map_err(|e| Error::Other(e.to_string()))?;
        interop::csr_to_dgcmatrix(&csr)
    }

    /// Fallible body of `layer` (see B3 / `throw_on_err`).
    fn layer_impl(&self, name: &str) -> Result<Robj> {
        let csr = self
            .reader
            .read_layer(name)
            .map_err(|e| Error::Other(e.to_string()))?;
        interop::csr_to_dgcmatrix(&csr)
    }

    /// Fallible body of `query` (see B3 / `throw_on_err`).
    fn query_impl(&self) -> Result<RQueryPipeline> {
        let path = self
            .path
            .to_str()
            .ok_or_else(|| Error::Other("file path is not valid UTF-8".into()))?;
        RQueryPipeline::from_path(path)
    }

    /// Open a fresh `QueryPipeline` over this file (re-open, like `query()`).
    fn open_pipeline(&self) -> Result<QueryPipeline> {
        QueryPipeline::open(&self.path).map_err(|e| Error::Other(e.to_string()))
    }

    /// Fallible body of `read_group` (F2 grouped reads; see B3 / `throw_on_err`).
    fn read_group_impl(&self, label: &str) -> Result<RQueryResult> {
        let qr = self
            .open_pipeline()?
            .read_group(label)
            .map_err(|e| Error::Other(e.to_string()))?;
        Ok(RQueryResult::from_result(qr))
    }

    /// Fallible body of `read_reference`. `Ok(NULL)` when the archive has no
    /// reference rows.
    fn read_reference_impl(&self) -> Result<Robj> {
        match self
            .open_pipeline()?
            .read_reference()
            .map_err(|e| Error::Other(e.to_string()))?
        {
            Some(qr) => Ok(RQueryResult::from_result(qr).into()),
            None => Ok(().into()),
        }
    }

    /// Fallible body of `group_labels` (character vector of distinct labels).
    fn group_labels_impl(&self) -> Result<Robj> {
        let labels = self
            .open_pipeline()?
            .group_labels()
            .map_err(|e| Error::Other(e.to_string()))?;
        Ok(labels.into())
    }

    /// Fallible body of `iter_group_shards` (R list of `RGroupShardHandle`).
    fn iter_group_shards_impl(&self) -> Result<Robj> {
        let handles = self
            .open_pipeline()?
            .iter_group_shards()
            .map_err(|e| Error::Other(e.to_string()))?;
        let items: Vec<Robj> = handles
            .into_iter()
            .map(|h| RGroupShardHandle::new(self.path.clone(), h).into())
            .collect();
        Ok(List::from_values(items).into())
    }

    /// Fallible body of `x_backed`: open a lazy, on-demand backed reader over
    /// this file's X. Re-opens the file (the backed reader owns its own
    /// `ScxReader`), exactly like `query_impl`.
    fn x_backed_impl(&self, cache_shards: usize) -> Result<RBackedSparse> {
        let path = self
            .path
            .to_str()
            .ok_or_else(|| Error::Other("file path is not valid UTF-8".into()))?;
        RBackedSparse::open_impl(path, cache_shards)
    }

    /// Fallible body of `x_lazy`: open a lazy-transform handle (empty chain)
    /// over this file's X, ready for `normalize_total`/`log1p`/`row_scale`.
    fn x_lazy_impl(&self, cache_shards: usize) -> Result<RLazyTransformed> {
        let path = self
            .path
            .to_str()
            .ok_or_else(|| Error::Other("file path is not valid UTF-8".into()))?;
        RLazyTransformed::open_impl(path, cache_shards)
    }

    /// Fallible body of `to_seurat` (kept off the `#[extendr]` surface so the
    /// public method can return `Robj` + `throw_on_err`; see B7).
    fn to_seurat_impl(&self) -> Result<Robj> {
        if self.reader.is_multimodal() {
            interop::to_seurat_multimodal(&self.reader)
        } else {
            // Reuse the existing single-modality path by
            // materialising a QueryResult-equivalent in-memory
            // structure. Read X / obs / var directly.
            let csr = self
                .reader
                .read_all_csr_shards()
                .map_err(|e| Error::Other(e.to_string()))?;
            let obs = self
                .reader
                .read_obs()
                .map_err(|e| Error::Other(e.to_string()))?;
            let var = self
                .reader
                .read_var()
                .map_err(|e| Error::Other(e.to_string()))?;
            let result = QueryResult {
                x: csr,
                obs,
                var,
                skipped_shards: 0,
                total_shards: 0,
                candidate_shard_rows: 0,
                matched_rows: 0,
                // TODO: wire the u32→f32 decode-loss guard for R reads. Hardcoded
                // 0 means R-side reads of >2²⁴ integer archives still round
                // silently (the guard is currently pyscx-only).
                max_value: 0,
            };
            interop::to_seurat_v5(&result)
        }
    }
}

#[extendr]
impl ScxExperiment {
    /// Open an SCX file. Returns a lazy handle (no data read yet).
    ///
    /// Returns `Robj` (not `Result`) and throws a clean R error via
    /// `throw_on_err` on failure: a fallible `#[extendr]` constructor would
    /// otherwise `unwrap()`-panic in extendr 0.8.0, masking the real message
    /// (e.g. an unsupported format version) behind "User function panicked".
    /// See B3.
    // Returns `Robj` (the externalptr wrapping `Self`) rather than `Self` so the
    // open error can be thrown cleanly via `throw_on_err`; the `new` name is the
    // extendr constructor convention the R wrapper depends on.
    #[allow(clippy::new_ret_no_self)]
    fn new(path: &str) -> Robj {
        crate::util::throw_on_err(Self::open_impl(path))
    }

    /// Number of observations (cells).
    /// Returns R numeric (f64) to handle >2B cells without i32 overflow.
    fn n_obs(&self) -> Robj {
        Robj::from(self.reader.n_obs() as f64)
    }

    /// Number of variables (genes).
    fn n_vars(&self) -> Robj {
        Robj::from(self.reader.n_vars() as f64)
    }

    /// Total non-zero entries.
    fn nnz(&self) -> Robj {
        Robj::from(self.reader.nnz() as f64)
    }

    /// Number of CSR shards.
    fn shard_count(&self) -> i32 {
        self.reader.header().n_csr_shards as i32
    }

    /// Read obs metadata as an R data.frame.
    ///
    /// Reads the Arrow IPC obs section, converts RecordBatch columns to R vectors.
    /// Categorical/dictionary columns → R factors.
    /// String columns → R character vectors.
    /// Numeric columns → R double vectors.
    /// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3).
    fn obs(&self) -> Robj {
        crate::util::throw_on_err(self.obs_impl())
    }

    /// Read var metadata as an R data.frame.
    ///
    /// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3).
    fn var(&self) -> Robj {
        crate::util::throw_on_err(self.var_impl())
    }

    /// Read the X matrix as a dgCMatrix (Matrix package sparse matrix).
    ///
    /// SCX stores CSR (row-major, cells × genes).
    /// R's dgCMatrix is CSC (column-major).
    /// Conversion steps:
    ///   1. Read all CSR shards via reader.read_all_csr_shards()
    ///   2. Transpose CSR → CSC (indptr_csc, indices_csc, data_csc)
    ///   3. Cast: indptr i64→i32 (safe for <2B nnz), data f32→f64 (lossless widening)
    ///   4. Construct dgCMatrix via new("dgCMatrix", i=, p=, x=, Dim=)
    ///
    /// The resulting dgCMatrix has dimensions (n_obs × n_vars),
    /// same as the original CSR orientation (cells as rows, genes as columns).
    ///
    /// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3).
    fn x_matrix(&self) -> Robj {
        crate::util::throw_on_err(self.x_matrix_impl())
    }

    /// Read a named layer as dgCMatrix.
    ///
    /// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3).
    fn layer(&self, name: &str) -> Robj {
        crate::util::throw_on_err(self.layer_impl(name))
    }

    /// List available layer names.
    fn layer_names(&self) -> Vec<String> {
        self.reader.layer_names()
    }

    /// Start a query pipeline. Returns an RQueryPipeline.
    /// Re-opens the file (QueryPipeline::open creates its own ScxReader).
    ///
    /// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3).
    fn query(&self) -> Robj {
        crate::util::throw_on_err(self.query_impl())
    }

    /// F2: read exactly the cells of one `group_by` label as an `RQueryResult`
    /// (call `$to_dgcmatrix()` / `$to_seurat()` / `$to_sce()` / `$obs()` on it).
    /// Grouped archive only (written with `group_by`). Unknown label or an
    /// ungrouped file → clean R `stop()` via `throw_on_err`.
    fn read_group(&self, label: &str) -> Robj {
        crate::util::throw_on_err(self.read_group_impl(label))
    }

    /// F2: read the reference cells (e.g. "non-targeting") as an `RQueryResult`,
    /// or `NULL` if the archive has no reference rows.
    fn read_reference(&self) -> Robj {
        crate::util::throw_on_err(self.read_reference_impl())
    }

    /// F2: distinct group labels present in the archive (character vector).
    fn group_labels(&self) -> Robj {
        crate::util::throw_on_err(self.group_labels_impl())
    }

    /// F2: one `RGroupShardHandle` per non-reference shard (a list), for
    /// streaming reads that keep ~one shard resident.
    fn iter_group_shards(&self) -> Robj {
        crate::util::throw_on_err(self.iter_group_shards_impl())
    }

    /// Open a lazy, on-demand backed view of X (no data read yet).
    ///
    /// Returns an `RBackedSparse` that reads rows from disk on demand with a
    /// shard-level LRU cache (`cache_shards` shards). Use it to slice
    /// atlas-scale files row-by-row instead of materialising the whole matrix
    /// via `x_matrix()`.
    ///
    /// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3).
    fn x_backed(&self, cache_shards: f64) -> Robj {
        util::throw_on_err(self.x_backed_impl(cache_shards as usize))
    }

    /// Open a lazy-transform view of X (no data read, no transforms yet).
    ///
    /// Chain `normalize_total` / `log1p` / `row_scale` onto the returned handle
    /// (`scx_normalize_total()` etc.); transforms are applied on read without
    /// materialising the matrix.
    ///
    /// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3).
    fn x_lazy(&self, cache_shards: f64) -> Robj {
        util::throw_on_err(self.x_lazy_impl(cache_shards as usize))
    }

    /// Phase I.1: True if this file has a registered modality table.
    fn is_multimodal(&self) -> bool {
        self.reader.is_multimodal()
    }

    /// Phase I.1: list modality names registered in the file (in
    /// insertion order). Empty for single-modality / v1 files.
    fn modality_names(&self) -> Vec<String> {
        self.reader
            .modality_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Phase I.1: build a Seurat v5 multi-assay object from this
    /// SCX file. On a single-modality file, returns a one-assay
    /// Seurat v5 object (matches the legacy
    /// `RQueryResult$to_seurat()` shape). On a multi-modality file,
    /// returns a Seurat v5 object with one assay per modality, all
    /// sharing the global obs as their meta.data.
    ///
    /// Returns `Robj` and throws a clean R error via `throw_on_err` on
    /// failure (e.g. missing `Seurat`) rather than `unwrap()`-panicking in
    /// extendr 0.8.0, which masks the message behind "User function
    /// panicked". See B7.
    fn to_seurat(&self) -> Robj {
        util::throw_on_err(self.to_seurat_impl())
    }

    /// Phase I.2: build a Bioconductor `MultiAssayExperiment` from
    /// this SCX file. Single-modality files raise — use
    /// `RQueryResult$to_sce()` for those.
    ///
    /// Returns `Robj` and throws a clean R error via `throw_on_err` (see
    /// `to_seurat` above and B7).
    fn to_mae(&self) -> Robj {
        util::throw_on_err(interop::to_mae(&self.reader))
    }
}

// ── Module Registration ─────────────────────────────────────────

extendr_module! {
    mod rscx;
    use query;
    use backed;
    use lazy;
    use ops;
    use interop;
    use harmony;
    use lisi;
    use accel;
    impl ScxExperiment;
}
