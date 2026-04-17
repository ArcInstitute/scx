use std::path::PathBuf;

use extendr_api::prelude::*;
use scx_format::ScxReader;

mod harmony;
mod interop;
mod lisi;
mod ops;
mod query;

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

#[extendr]
impl ScxExperiment {
    /// Open an SCX file. Returns a lazy handle (no data read yet).
    fn new(path: &str) -> Result<Self> {
        let path_buf = PathBuf::from(path);
        let reader = ScxReader::open(&path_buf)
            .map_err(|e| Error::Other(format!("failed to open SCX file '{}': {}", path, e)))?;
        Ok(Self {
            reader,
            path: path_buf,
        })
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
    fn obs(&self) -> Result<Robj> {
        let batch = self
            .reader
            .read_obs()
            .map_err(|e| Error::Other(e.to_string()))?;
        interop::record_batch_to_dataframe(&batch)
    }

    /// Read var metadata as an R data.frame.
    fn var(&self) -> Result<Robj> {
        let batch = self
            .reader
            .read_var()
            .map_err(|e| Error::Other(e.to_string()))?;
        interop::record_batch_to_dataframe(&batch)
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
    fn x_matrix(&self) -> Result<Robj> {
        let csr = self
            .reader
            .read_all_csr_shards()
            .map_err(|e| Error::Other(e.to_string()))?;
        interop::csr_to_dgcmatrix(&csr)
    }

    /// Read a named layer as dgCMatrix.
    fn layer(&self, name: &str) -> Result<Robj> {
        let csr = self
            .reader
            .read_layer(name)
            .map_err(|e| Error::Other(e.to_string()))?;
        interop::csr_to_dgcmatrix(&csr)
    }

    /// List available layer names.
    fn layer_names(&self) -> Vec<String> {
        self.reader.layer_names()
    }

    /// Start a query pipeline. Returns an RQueryPipeline.
    /// Re-opens the file (QueryPipeline::open creates its own ScxReader).
    fn query(&self) -> Result<RQueryPipeline> {
        RQueryPipeline::from_path(self.path.to_str().unwrap_or(""))
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
    impl ScxExperiment;
}
