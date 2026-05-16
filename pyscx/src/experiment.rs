// PyExperiment — lazy handle for SCX files

use std::path::PathBuf;

use arrow::array::Array;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use scx_engine::QueryPipeline;
use scx_format::backed::BackedCsrReader;
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

/// Phase 5b: open a fresh `BackedCsrReader` for the requested modality
/// from a file path. `modality = None` → modality_id 0 (the unimodal /
/// global X) on non-multimodal files; on multimodal files we require an
/// explicit modality unless there is exactly one.
fn open_backed_csr(path: &PathBuf, modality: Option<&str>) -> PyResult<BackedCsrReader> {
    let opened = ScxReader::open(path).map_err(to_pyerr)?;
    if !opened.is_multimodal() {
        return Ok(BackedCsrReader::new(opened, 4));
    }
    let modality_id = match modality {
        Some(name) => opened.modality_id(name).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown modality '{name}'"))
        })?,
        None => {
            let names = opened.modality_names();
            if names.len() == 1 {
                opened.modality_id(names[0]).unwrap_or(1)
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "file is multimodal with {} modalities; pass modality=... \
                     (one of {:?})",
                    names.len(),
                    names
                )));
            }
        }
    };
    Ok(BackedCsrReader::for_modality(opened, modality_id, 4))
}

/// Resolve a gene name against the appropriate modality's `var`.
fn resolve_gene_name(reader: &ScxReader, modality: Option<&str>, name: &str) -> PyResult<u32> {
    use scx_format::SectionType;
    let modality_id = if reader.is_multimodal() {
        match modality {
            Some(m) => reader.modality_id(m).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("unknown modality '{m}'"))
            })?,
            None => {
                let names = reader.modality_names();
                if names.len() == 1 {
                    reader.modality_id(names[0]).unwrap_or(1)
                } else {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "file is multimodal; pass modality=... to resolve gene name",
                    ));
                }
            }
        }
    } else {
        0
    };
    let var_section_name = if modality_id == 0 {
        "var".to_string()
    } else {
        match reader.modality_info(modality_id) {
            Some(info) => format!("var/{}", info.name),
            None => "var".to_string(),
        }
    };
    let entry = reader
        .catalog()
        .entries
        .iter()
        .find(|e| {
            e.section_type == SectionType::VarMetadata
                && (e.name == var_section_name || (modality_id == 0 && e.name == "var"))
        })
        .ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "var section '{var_section_name}' not found"
            ))
        })?;
    let bytes = reader.section_bytes(entry).map_err(to_pyerr)?;
    let cursor = std::io::Cursor::new(bytes);
    let mut arrow_reader = arrow::ipc::reader::FileReader::try_new(cursor, None)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let batch = arrow_reader
        .next()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("var record batch is empty"))?
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    // The pandas index column is conventionally named "_index" or
    // tagged via Arrow schema metadata; try the most common keys
    // before failing.
    let candidate_columns = ["_index", "gene_name", "feature_name", "gene_id", "name"];
    for col_name in candidate_columns {
        if let Some((idx, _)) = batch.schema().column_with_name(col_name) {
            let col = batch.column(idx);
            if let Some(arr) = col.as_any().downcast_ref::<arrow::array::StringArray>() {
                for i in 0..arr.len() {
                    if !arr.is_null(i) && arr.value(i) == name {
                        return Ok(i as u32);
                    }
                }
            }
            if let Some(arr) = col
                .as_any()
                .downcast_ref::<arrow::array::LargeStringArray>()
            {
                for i in 0..arr.len() {
                    if !arr.is_null(i) && arr.value(i) == name {
                        return Ok(i as u32);
                    }
                }
            }
        }
    }
    Err(pyo3::exceptions::PyKeyError::new_err(format!(
        "gene name '{name}' not found in var index"
    )))
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

    /// File-header codec ID
    /// (`0=None`, `1=Scx1`, `2=Zstd`, `3=Lz4Shuffle`, `4=Pcodec`).
    ///
    /// Per-shard codec overrides are stored in each shard header; readers
    /// must consult `ShardHeader.codec_id` rather than this value when
    /// decoding individual shards.
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

        let expected = self.reader.n_obs() as usize;
        if mask_slice.len() != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "mask length {} does not match n_obs {}",
                mask_slice.len(),
                expected
            )));
        }

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
    ///     preserve_slots: When True together with obs_filter in non-backed mode,
    ///                     materialize obsm and layers after filtering instead of
    ///                     dropping them. Skips query-engine predicate pushdown for X
    ///                     reads; the obs_filter expression is parsed by pandas.eval
    ///                     instead of the SCX predicate engine, so syntax must be
    ///                     pandas-compatible. No effect when obs_filter is None or
    ///                     when backed=True (backed mode already preserves slots).
    ///
    /// Returns an anndata.AnnData with X, obs, var, and optionally
    /// obsm, uns, and layers populated from the file.
    #[pyo3(signature = (backed=false, cache_shards=4, var_names=None, obs_filter=None, layers=None, preserve_slots=false))]
    #[allow(clippy::too_many_arguments)]
    fn to_anndata<'py>(
        &self,
        py: Python<'py>,
        backed: bool,
        cache_shards: usize,
        var_names: Option<Vec<String>>,
        obs_filter: Option<&str>,
        layers: Option<Vec<String>>,
        preserve_slots: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if backed {
            anndata::to_anndata_backed(
                py,
                &self.path,
                cache_shards,
                var_names.as_deref(),
                obs_filter,
                layers.as_deref(),
            )
        } else {
            anndata::to_anndata_filtered(
                py,
                &self.path,
                &self.reader,
                var_names.as_deref(),
                obs_filter,
                layers.as_deref(),
                preserve_slots,
            )
        }
    }

    /// Validate all section checksums.
    ///
    /// Returns a list of (section_name, passed) tuples.
    fn validate(&self) -> PyResult<Vec<(String, bool)>> {
        self.reader.validate().map_err(to_pyerr)
    }

    /// True if this file is multimodal (Phase B / v2 with
    /// `n_modalities > 0`). Mirrors `header.has_modalities()`.
    #[getter]
    fn is_multimodal(&self) -> bool {
        self.reader.is_multimodal()
    }

    /// Number of registered modalities (0 for v1 files and
    /// single-modality v2 files).
    #[getter]
    fn n_modalities(&self) -> u32 {
        self.reader.n_modalities()
    }

    /// Ordered list of modality names (empty for single-modality
    /// files). The position in the list maps 1:1 to the 1-based
    /// modality_id (`names[i] -> modality_id = i + 1`).
    #[getter]
    fn modality_names(&self) -> Vec<String> {
        self.reader
            .modality_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Resolve a modality name to its 1-based `modality_id`.
    /// Returns `None` for unknown names or single-modality files.
    fn modality_id(&self, name: &str) -> Option<u8> {
        self.reader.modality_id(name)
    }

    /// Per-modality information block for the given 1-based
    /// `modality_id`. Returns a dict with the on-disk fields of
    /// `ModalityInfo` (name, modality_type, default_codec_id,
    /// default_value_encoding, n_vars, nnz, n_csr_shards,
    /// n_csc_shards, flags). Useful for introspection (e.g. asserting
    /// per-modality codec routing in tests).
    fn modality_info<'py>(
        &self,
        py: Python<'py>,
        modality_id: u8,
    ) -> PyResult<Option<Bound<'py, pyo3::types::PyDict>>> {
        use pyo3::types::PyDict;
        let info = match self.reader.modality_info(modality_id) {
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

    /// Materialise this file as a `mudata.MuData` object.
    ///
    /// Iterates the registered modalities, builds an AnnData per
    /// modality via the existing zero-copy CSR path, and attaches
    /// them to a `MuData(...)` with the shared global obs.
    /// Single-modality files raise `RuntimeError` directing to
    /// `to_anndata()`.
    fn to_mudata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::mudata::to_mudata(py, &self.reader)
    }

    /// Phase 5b: per-gene detection counts (number of cells where
    /// each gene is expressed).
    ///
    /// Only `axis="var"` is supported in the first cut (per-cell
    /// detection counts would require a transpose). On multimodal
    /// files, pass `modality=` to select a specific modality;
    /// otherwise the global X (modality_id = 0) is used.
    ///
    /// Fast path: when bitmap sidecars are present for every CSR
    /// shard, this is O(roaring-cardinality). Otherwise the call
    /// falls back to a full CSR scan.
    ///
    /// Returns a numpy `int64` array of length `n_vars`.
    #[pyo3(signature = (axis = "var", modality = None))]
    fn detection_counts<'py>(
        &self,
        py: Python<'py>,
        axis: &str,
        modality: Option<&str>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        if axis != "var" {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "detection_counts: only axis='var' is supported (got '{axis}')"
            )));
        }
        let backed = open_backed_csr(&self.path, modality)?;
        let counts: Vec<i64> = py
            .allow_threads(|| backed.gene_detection_counts())
            .map_err(to_pyerr)?
            .into_iter()
            .map(|c| c as i64)
            .collect();
        Ok(PyArray1::from_vec(py, counts))
    }

    /// Phase 5b: global row indices of cells expressing the given gene.
    ///
    /// `gene` may be either an integer gene index (`0..n_vars`) or a
    /// string name (looked up against the modality's `var.index`).
    /// Returns a numpy `uint32` array of global row ids.
    #[pyo3(signature = (gene, modality = None))]
    fn cells_expressing<'py>(
        &self,
        py: Python<'py>,
        gene: &Bound<'_, PyAny>,
        modality: Option<&str>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let backed = open_backed_csr(&self.path, modality)?;
        // Resolve gene → gene_idx. Integer fast path; string falls
        // through to a var.index lookup.
        let gene_idx: u32 = if let Ok(idx) = gene.extract::<u32>() {
            idx
        } else {
            let name: String = gene.extract().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "gene must be an integer index or a string name",
                )
            })?;
            resolve_gene_name(&self.reader, modality, &name)?
        };
        let rows = py
            .allow_threads(|| backed.cells_expressing_gene(gene_idx))
            .map_err(to_pyerr)?;
        Ok(PyArray1::from_vec(py, rows))
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
