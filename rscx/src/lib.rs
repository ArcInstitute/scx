use std::path::PathBuf;

use extendr_api::prelude::*;
use scx_engine::pipeline::QueryResult;
use scx_format_io::ScxReader;

mod accel;
mod harmony;
mod interop;
mod lisi;
mod ops;
mod query;
mod util;

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
    use ops;
    use interop;
    use harmony;
    use lisi;
    use accel;
    impl ScxExperiment;
}
