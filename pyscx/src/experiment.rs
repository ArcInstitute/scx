// PyExperiment — lazy handle for SCX files

use std::path::PathBuf;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use scx_engine::QueryPipeline;
use scx_format::ScxReader;

use crate::anndata;
use crate::query::PyQueryPipeline;
use crate::to_pyerr;

/// A handle to an open SCX file.
///
/// Provides metadata accessors and methods to convert to AnnData.
#[pyclass]
pub struct PyExperiment {
    reader: ScxReader,
    pub(crate) path: PathBuf,
}

impl PyExperiment {
    /// Construct from an already-opened ScxReader and its path (Rust-only).
    pub fn new(reader: ScxReader, path: PathBuf) -> Self {
        Self { reader, path }
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

    /// Create a new query pipeline for this file.
    ///
    /// Returns a `PyQueryPipeline` builder — call `.filter_obs()`,
    /// `.select_genes()`, etc., then `.collect()` to execute.
    ///
    /// Example:
    ///     result = pyscx.open("data.scx").query().collect()
    fn query(&self) -> PyResult<PyQueryPipeline> {
        let pipeline = QueryPipeline::open(&self.path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyQueryPipeline::from_pipeline(pipeline))
    }

    /// Mark cells as logically deleted using a boolean mask.
    ///
    /// The mask should be a boolean numpy array whose length matches n_obs.
    /// Cells where the mask is True are marked as deleted.
    /// Returns the total number of deleted cells (including previously deleted).
    ///
    /// Example:
    ///     exp = pyscx.open("experiment.scx")
    ///     adata = exp.to_anndata()
    ///     total = exp.mark_deleted(adata.obs["is_doublet"] == True)
    fn mark_deleted(&mut self, mask: PyReadonlyArray1<'_, bool>) -> PyResult<u64> {
        let mask_slice = mask
            .as_slice()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Collect indices where mask is True
        let indices: Vec<u64> = mask_slice
            .iter()
            .enumerate()
            .filter(|(_, &v)| v)
            .map(|(i, _)| i as u64)
            .collect();

        let total = scx_ops::mark_deleted(&self.path, &indices)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Re-open reader so subsequent reads see the updated file (finding 9.6).
        self.reader = ScxReader::open(&self.path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(total)
    }

    /// Convert this SCX file to an AnnData object.
    ///
    /// Args:
    ///     backed: If True, X and layers are wrapped in ScxBackedSparseDataset
    ///             for on-demand shard decoding (read-only).
    ///     cache_shards: Number of decoded shards to LRU-cache (default 4).
    ///                   Only used when backed=True.
    ///     var_names: Optional list of gene names to project to at load time.
    ///                Only loads the specified genes. Not supported with backed=True.
    ///     obs_filter: Optional predicate expression (e.g., "cell_type == 'T cell'")
    ///                 to filter observations. Uses predicate pushdown for shard skipping.
    ///     layers: Optional list of layer names to load. If None, all layers are loaded.
    ///
    /// Returns an anndata.AnnData with X, obs, var, and optionally
    /// obsm, uns, and layers populated from the file.
    #[pyo3(signature = (backed=false, cache_shards=4, var_names=None, obs_filter=None, layers=None))]
    fn to_anndata<'py>(
        &self,
        py: Python<'py>,
        backed: bool,
        cache_shards: usize,
        var_names: Option<Vec<String>>,
        obs_filter: Option<&str>,
        layers: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if backed {
            if var_names.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "var_names is not supported with backed=True. \
                     Use the query pipeline instead: pyscx.open(path).query().select_genes([...]).collect()"
                ));
            }
            anndata::to_anndata_backed(py, &self.path, cache_shards, obs_filter, layers.as_deref())
        } else {
            anndata::to_anndata_filtered(
                py,
                &self.path,
                &self.reader,
                var_names.as_deref(),
                obs_filter,
                layers.as_deref(),
            )
        }
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
