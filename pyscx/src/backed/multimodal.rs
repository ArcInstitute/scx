// ScxBackedMuDataset / ScxBackedMuModality — backed multimodal access.
//
// Extracted from the former pyscx/src/backed.rs (T5.7).

use std::path::Path;
use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use scx_format_io::{BackedCscReader, BackedCsrReader};

use super::*;
use scx_format_io::ScxReader;
use std::sync::Mutex as StdMutex;

/// Phase D.4: backed wrapper for multimodal SCX files. Holds an
/// `Arc<ScxReader>` and lazily exposes per-modality
/// `ScxBackedSparseDataset`s via the dict-like `.mod` attribute.
///
/// Construction is opt-in: `pyscx.open(path)` keeps returning
/// `PyExperiment` for backward compat. Users who want the lazy
/// per-modality surface call `pyscx.ScxBackedMuDataset(path)`
/// directly.
///
/// ## Surface
///
/// - `mu.is_multimodal` / `mu.n_modalities` / `mu.modality_names`
/// - `mu.modality_id(name) -> Optional[int]`
/// - `mu.modality_info(modality_id) -> Optional[dict]`
/// - `mu.obs` (cached pandas DataFrame; computed on first access)
/// - `mu.mod` (dict-like proxy; `mu.mod["rna"]` lazily yields a
///   `ScxBackedSparseDataset` pinned to that modality)
/// - `mu.to_mudata()` — eager round-trip to a real `mudata.MuData`
///   for users who want the materialised view.
#[pyclass(name = "ScxBackedMuDataset")]
pub struct ScxBackedMuDataset {
    /// Path to the SCX file. Re-opened per `.mod[name]` access since
    /// `BackedCsrReader::for_modality` / `BackedCscReader::for_modality`
    /// consume the `ScxReader`. Reader open is mmap-cheap.
    path: std::path::PathBuf,
    /// Cached metadata reader for `is_multimodal` / `modality_*` /
    /// `obs` accessors.
    meta_reader: Arc<ScxReader>,
    /// Cached global obs as a pandas DataFrame. Computed on first
    /// `obs` access; reused thereafter.
    cached_obs: StdMutex<Option<Py<PyAny>>>,
    /// CSR cache size to use when constructing per-modality
    /// `BackedCsrReader`s. Defaults to 4 (matches the experiment
    /// default).
    cache_shards: usize,
}

#[pymethods]
impl ScxBackedMuDataset {
    /// Open a multimodal SCX file as a backed wrapper. Raises if the
    /// file is single-modality (use `pyscx.open(path)` for those).
    #[new]
    #[pyo3(signature = (path, cache_shards=None))]
    fn new(path: &str, cache_shards: Option<usize>) -> PyResult<Self> {
        let path_buf = std::path::PathBuf::from(path);
        let reader = ScxReader::open(&path_buf)
            .map_err(|e| PyRuntimeError::new_err(format!("ScxReader::open: {e}")))?;
        if !reader.is_multimodal() {
            return Err(PyRuntimeError::new_err(format!(
                "ScxBackedMuDataset: file '{path}' is single-modality. \
                 Use pyscx.open(path) for those."
            )));
        }
        Ok(ScxBackedMuDataset {
            path: path_buf,
            meta_reader: Arc::new(reader),
            cached_obs: StdMutex::new(None),
            cache_shards: cache_shards.unwrap_or(4),
        })
    }

    #[getter]
    fn is_multimodal(&self) -> bool {
        true
    }

    #[getter]
    fn n_modalities(&self) -> u32 {
        self.meta_reader.n_modalities()
    }

    #[getter]
    fn modality_names(&self) -> Vec<String> {
        self.meta_reader
            .modality_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn modality_id(&self, name: &str) -> Option<u8> {
        self.meta_reader.modality_id(name)
    }

    fn modality_info<'py>(
        &self,
        py: Python<'py>,
        modality_id: u8,
    ) -> PyResult<Option<Bound<'py, pyo3::types::PyDict>>> {
        use pyo3::types::PyDict;
        let info = match self.meta_reader.modality_info(modality_id) {
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

    /// Lazy global obs. First call materialises the obs as a pandas
    /// DataFrame; subsequent calls return the same Python object.
    #[getter]
    fn obs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Fast path: cached obs already constructed.
        {
            let guard = self.cached_obs.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(ref obj) = *guard {
                return Ok(obj.clone_ref(py));
            }
        }
        // Slow path: read obs, convert to pandas, cache.
        let batch = self
            .meta_reader
            .read_obs()
            .map_err(|e| PyRuntimeError::new_err(format!("read_obs: {e}")))?;
        let table = crate::convert::record_batch_to_pyarrow(py, &batch)?;
        let df = crate::convert::pyarrow_table_to_pandas(&table)?;
        let obj: Py<PyAny> = df.unbind();
        let mut guard = self.cached_obs.lock().unwrap_or_else(|e| e.into_inner());
        *guard = Some(obj.clone_ref(py));
        Ok(obj)
    }

    /// Dict-like `.mod` accessor. Returns a `ScxBackedMuModality`
    /// proxy whose `__getitem__(name)` lazily yields a fresh
    /// `ScxBackedSparseDataset` pinned to that modality.
    #[getter]
    fn r#mod(&self, py: Python<'_>) -> PyResult<Py<ScxBackedMuModality>> {
        Py::new(
            py,
            ScxBackedMuModality {
                path: self.path.clone(),
                modality_names: self.modality_names(),
                cache_shards: self.cache_shards,
            },
        )
    }

    /// Eagerly materialise as a real `mudata.MuData`. Convenience
    /// shortcut for users who want the materialised view.
    fn to_mudata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // Convenience materialize on a backed handle: pass allow_lossy=true so the
        // decode-loss guard does not hard-error here, consistent with backed reads
        // being ungated (the caller already opted into a backed workflow).
        crate::mudata::to_mudata(py, &self.meta_reader, true)
    }

    fn __repr__(&self) -> String {
        let names: Vec<String> = self.modality_names();
        format!(
            "ScxBackedMuDataset(n_obs={}, modalities={:?})",
            self.meta_reader.n_obs(),
            names,
        )
    }
}

/// Phase D.4: dict-like proxy returned from
/// `ScxBackedMuDataset.mod`. Each `__getitem__(name)` constructs a
/// fresh `BackedCsrReader::for_modality` (and matching
/// `BackedCscReader::for_modality` if the modality has CSC sidecars)
/// against a freshly-opened `ScxReader`, then wraps both in a
/// `ScxBackedSparseDataset` tagged with `modality_id = id`.
///
/// Reader open is mmap-cheap so per-modality reopens are negligible
/// vs the alternative of cloning a single shared `ScxReader` (which
/// the existing constructors don't support).
#[pyclass(name = "ScxBackedMuModality")]
pub struct ScxBackedMuModality {
    path: std::path::PathBuf,
    modality_names: Vec<String>,
    cache_shards: usize,
}

#[pymethods]
impl ScxBackedMuModality {
    fn __getitem__(&self, py: Python<'_>, name: &str) -> PyResult<Py<ScxBackedSparseDataset>> {
        // Resolve name → modality_id via a cheap meta reader.
        let meta = ScxReader::open(&self.path)
            .map_err(|e| PyRuntimeError::new_err(format!("ScxReader::open: {e}")))?;
        let modality_id = meta.modality_id(name).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "modality '{name}' not found (available: {:?})",
                self.modality_names
            ))
        })?;
        let has_csc = meta
            .modality_info(modality_id)
            .map(|i| i.flags.has_csc())
            .unwrap_or(false);
        drop(meta);

        // Build per-modality CSR reader.
        let csr_reader = ScxReader::open(&self.path)
            .map_err(|e| PyRuntimeError::new_err(format!("ScxReader::open: {e}")))?;
        let backed_csr = Arc::new(BackedCsrReader::for_modality(
            csr_reader,
            modality_id,
            self.cache_shards,
        ));

        // Optional per-modality CSC reader.
        let backed_csc = if has_csc {
            let csc_reader = ScxReader::open(&self.path)
                .map_err(|e| PyRuntimeError::new_err(format!("ScxReader::open: {e}")))?;
            let r = BackedCscReader::for_modality(csc_reader, modality_id, self.cache_shards)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("BackedCscReader::for_modality: {e}"))
                })?;
            Some(Arc::new(r))
        } else {
            None
        };

        let mut ds = ScxBackedSparseDataset::from_reader(backed_csr, self.cache_shards);
        ds.with_csc_reader(backed_csc).with_modality_id(modality_id);
        ds.with_source_path(&self.path);
        Py::new(py, ds)
    }

    fn __contains__(&self, name: &str) -> bool {
        self.modality_names.iter().any(|n| n == name)
    }

    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Yield modality names. Easiest cross-version: convert to a
        // PyList and call its __iter__.
        let list = pyo3::types::PyList::new(py, &slf.modality_names)?;
        let iter = list.try_iter()?;
        Ok(iter.into_pyobject(py)?.into_any().unbind())
    }

    fn __len__(&self) -> usize {
        self.modality_names.len()
    }

    fn keys(&self) -> Vec<String> {
        self.modality_names.clone()
    }

    fn __repr__(&self) -> String {
        format!("ScxBackedMuModality(modalities={:?})", self.modality_names)
    }
}

// Suppress unused-import warning on `Path` when no other code in this
// file references it (we only need it transitively for path conversion).
#[allow(dead_code)]
fn _path_marker(_p: &Path) {}

// ---------------------------------------------------------------------------
// ScxBackedObsmDataset — backed dense row-gather for an obsm embedding
// ---------------------------------------------------------------------------
