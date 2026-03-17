// PyExperiment — lazy handle for SCX files

use pyo3::prelude::*;
use scx_format::ScxReader;

use crate::anndata;
use crate::to_pyerr;

/// A handle to an open SCX file.
///
/// Provides metadata accessors and methods to convert to AnnData.
#[pyclass]
pub struct PyExperiment {
    reader: ScxReader,
}

impl PyExperiment {
    /// Construct from an already-opened ScxReader (Rust-only).
    pub fn from_reader(reader: ScxReader) -> Self {
        Self { reader }
    }
}

#[pymethods]
impl PyExperiment {
    /// Number of observations (cells).
    #[getter]
    fn n_obs(&self) -> u64 {
        self.reader.n_obs()
    }

    /// Number of variables (genes).
    #[getter]
    fn n_vars(&self) -> u64 {
        self.reader.n_vars()
    }

    /// Total number of non-zero entries.
    #[getter]
    fn nnz(&self) -> u64 {
        self.reader.nnz()
    }

    /// Number of CSR shards in the file.
    #[getter]
    fn shard_count(&self) -> u32 {
        self.reader.header().n_csr_shards
    }

    /// Format version (currently 1).
    #[getter]
    fn format_version(&self) -> u16 {
        self.reader.header().format_version
    }

    /// Codec ID (0=None, 1=Scx1, 2=Zstd).
    #[getter]
    fn codec_id(&self) -> u8 {
        self.reader.header().codec_id
    }

    /// List of layer names in the file.
    #[getter]
    fn layer_names(&self) -> Vec<String> {
        self.reader.layer_names()
    }

    /// Convert this SCX file to an AnnData object.
    ///
    /// Returns an anndata.AnnData with X, obs, var, and optionally
    /// obsm, uns, and layers populated from the file.
    fn to_anndata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        anndata::to_anndata(py, &self.reader)
    }

    /// Validate all section checksums.
    ///
    /// Returns a list of (section_name, passed) tuples.
    fn validate(&self) -> PyResult<Vec<(String, bool)>> {
        self.reader.validate().map_err(to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyExperiment(n_obs={}, n_vars={}, nnz={}, shards={}, codec={})",
            self.reader.n_obs(),
            self.reader.n_vars(),
            self.reader.nnz(),
            self.reader.header().n_csr_shards,
            self.reader.header().codec_id,
        )
    }
}
