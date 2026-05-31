// Lazy mappings for AnnData's obsp / varp / varm / layers slots.
//
// Each `to_anndata()` call returns an AnnData whose `_obsp`, `_varp`,
// `_varm`, `_layers` private attributes are set directly to one of the
// classes in this module, bypassing the public property setter (which
// would otherwise trigger `AlignedActual.__init__` and force eager
// materialization via `coerce_array`).
//
// The lazy mappings expose the full `MutableMapping` protocol:
//   * `__getitem__(key)` — read + decode the matching section on first
//     access; subsequent reads of the same key hit the local cache.
//   * `__contains__` / `__iter__` / `__len__` — answer from the
//     catalog-derived key set without any I/O.
//   * `__setitem__` / `__delitem__` — mutate the in-memory cache;
//     never write back to the on-disk SCX file.
//
// When `to_anndata()` is called with `eager=True`, the caller invokes
// `materialize_all(py)` and substitutes a plain `dict` for the lazy
// wrapper before handing it to AnnData. That path mirrors pre-fix
// behaviour for callers that want to fully detach the AnnData from
// the SCX file handle.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use scx_format::ScxReader;

use crate::anndata::{
    coo_record_batch_to_scipy, csr_to_scipy, filter_coo_obsp_by_kept_rows,
    filter_obs_by_deletion_vectors, obsm_batch_to_numpy,
};
use crate::to_pyerr;

/// Which pairwise axis the mapping is bound to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PairwiseAxis {
    Obsp,
    Varp,
}

impl PairwiseAxis {
    fn label(self) -> &'static str {
        match self {
            PairwiseAxis::Obsp => "obsp",
            PairwiseAxis::Varp => "varp",
        }
    }
}

/// Lazy mapping for `ad.obsp` / `ad.varp`. Constructed by `to_anndata()`
/// and assigned to the AnnData's `_obsp` / `_varp` slot directly. The
/// pairwise sparse matrix for each key is decoded the first time the
/// key is looked up.
#[pyclass(name = "ScxLazyPairwiseMapping", mapping)]
pub struct ScxLazyPairwiseMapping {
    reader: Arc<ScxReader>,
    axis: PairwiseAxis,
    /// Per-obsp deletion-vector remap. `Some(_)` only when the SCX file
    /// has a deletion vector AND `axis == Obsp` (varp's axis is `var`,
    /// which has no deletion vector).
    kept_to_global: Option<Arc<Vec<u64>>>,
    /// Materialization state. `None` = key exists on disk but has not
    /// yet been read; `Some(obj)` = cached value (either disk-loaded
    /// and validated, or user-inserted via `__setitem__`).
    state: Mutex<HashMap<String, Option<Py<PyAny>>>>,
}

impl ScxLazyPairwiseMapping {
    pub(crate) fn new(
        reader: Arc<ScxReader>,
        axis: PairwiseAxis,
        kept_to_global: Option<Arc<Vec<u64>>>,
    ) -> Self {
        let keys = match axis {
            PairwiseAxis::Obsp => reader.list_obsp(),
            PairwiseAxis::Varp => reader.list_varp(),
        };
        let state: HashMap<String, Option<Py<PyAny>>> =
            keys.into_iter().map(|k| (k, None)).collect();
        Self {
            reader,
            axis,
            kept_to_global,
            state: Mutex::new(state),
        }
    }

    /// Convert the lazy mapping into a plain `dict` by materializing
    /// every key. Used by the `eager=True` `to_anndata()` path.
    pub(crate) fn materialize_all<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let keys: Vec<String> = self.state.lock().unwrap().keys().cloned().collect();
        let dict = PyDict::new(py);
        for key in keys {
            let value = self.fetch(py, &key)?;
            dict.set_item(&key, value)?;
        }
        Ok(dict)
    }

    fn fetch(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        {
            let state = self.state.lock().unwrap();
            match state.get(key) {
                Some(Some(obj)) => return Ok(obj.clone_ref(py)),
                Some(None) => {}
                None => return Err(PyKeyError::new_err(key.to_string())),
            }
        }
        let batch = match self.axis {
            PairwiseAxis::Obsp => self.reader.read_obsp(key),
            PairwiseAxis::Varp => self.reader.read_varp(key),
        }
        .map_err(to_pyerr)?;
        let bound = match (&self.axis, &self.kept_to_global) {
            (PairwiseAxis::Obsp, Some(kept)) => {
                let filtered = filter_coo_obsp_by_kept_rows(&batch, kept)?;
                coo_record_batch_to_scipy(py, &filtered)?
            }
            _ => coo_record_batch_to_scipy(py, &batch)?,
        };
        let obj: Py<PyAny> = bound.unbind();
        let mut state = self.state.lock().unwrap();
        // Re-check: a concurrent fetch may have populated the slot while
        // we were decoding. Return whatever's already there so callers
        // converge on the same Py<PyAny> identity (the `a is b` invariant
        // asserted by `test_to_anndata_obsp_is_lazy`).
        if let Some(Some(cached)) = state.get(key) {
            return Ok(cached.clone_ref(py));
        }
        state.insert(key.to_string(), Some(obj.clone_ref(py)));
        Ok(obj)
    }
}

#[pymethods]
impl ScxLazyPairwiseMapping {
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        self.fetch(py, key)
    }

    fn __setitem__(&self, key: String, value: Py<PyAny>) {
        self.state.lock().unwrap().insert(key, Some(value));
    }

    fn __delitem__(&self, key: &str) -> PyResult<()> {
        let mut state = self.state.lock().unwrap();
        if state.remove(key).is_none() {
            return Err(PyKeyError::new_err(key.to_string()));
        }
        Ok(())
    }

    fn __contains__(&self, key: &str) -> bool {
        self.state.lock().unwrap().contains_key(key)
    }

    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let keys: Vec<String> = slf.state.lock().unwrap().keys().cloned().collect();
        let list = PyList::new(py, &keys)?;
        let iter = list.try_iter()?;
        Ok(iter.into_pyobject(py)?.into_any().unbind())
    }

    fn __len__(&self) -> usize {
        self.state.lock().unwrap().len()
    }

    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let keys: Vec<String> = self.state.lock().unwrap().keys().cloned().collect();
        PyList::new(py, &keys)
    }

    fn values(slf: PyRef<'_, Self>) -> PyResult<Py<ScxLazyValueIterator>> {
        let py = slf.py();
        let keys: Vec<String> = slf.state.lock().unwrap().keys().cloned().collect();
        let parent: Py<PyAny> = slf.into_pyobject(py)?.into_any().unbind();
        Py::new(
            py,
            ScxLazyValueIterator {
                parent,
                keys: keys.into_iter(),
            },
        )
    }

    fn items(slf: PyRef<'_, Self>) -> PyResult<Py<ScxLazyItemIterator>> {
        let py = slf.py();
        let keys: Vec<String> = slf.state.lock().unwrap().keys().cloned().collect();
        let parent: Py<PyAny> = slf.into_pyobject(py)?.into_any().unbind();
        Py::new(
            py,
            ScxLazyItemIterator {
                parent,
                keys: keys.into_iter(),
            },
        )
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        match self.fetch(py, key) {
            Ok(v) => Ok(v),
            Err(e) if e.is_instance_of::<PyKeyError>(py) => {
                Ok(default.unwrap_or_else(|| py.None()))
            }
            Err(e) => Err(e),
        }
    }

    fn __repr__(&self) -> String {
        let state = self.state.lock().unwrap();
        let n_keys = state.len();
        let n_cached = state.values().filter(|v| v.is_some()).count();
        format!(
            "ScxLazyPairwiseMapping({}, {} keys, {} materialized)",
            self.axis.label(),
            n_keys,
            n_cached,
        )
    }
}

/// Lazy mapping for `ad.varm`. Each value is decoded to a dense numpy
/// 2-D array on first access (mirrors the eager pre-fix behaviour at
/// `anndata.rs:obsm_batch_to_numpy`).
#[pyclass(name = "ScxLazyVarmMapping", mapping)]
pub struct ScxLazyVarmMapping {
    reader: Arc<ScxReader>,
    state: Mutex<HashMap<String, Option<Py<PyAny>>>>,
}

impl ScxLazyVarmMapping {
    pub(crate) fn new(reader: Arc<ScxReader>) -> Self {
        let state: HashMap<String, Option<Py<PyAny>>> =
            reader.list_varm().into_iter().map(|k| (k, None)).collect();
        Self {
            reader,
            state: Mutex::new(state),
        }
    }

    pub(crate) fn materialize_all<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let keys: Vec<String> = self.state.lock().unwrap().keys().cloned().collect();
        let dict = PyDict::new(py);
        for key in keys {
            let value = self.fetch(py, &key)?;
            dict.set_item(&key, value)?;
        }
        Ok(dict)
    }

    fn fetch(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        {
            let state = self.state.lock().unwrap();
            match state.get(key) {
                Some(Some(obj)) => return Ok(obj.clone_ref(py)),
                Some(None) => {}
                None => return Err(PyKeyError::new_err(key.to_string())),
            }
        }
        let batch = self.reader.read_varm(key).map_err(to_pyerr)?;
        let bound = obsm_batch_to_numpy(py, &batch)?;
        let obj: Py<PyAny> = bound.unbind();
        let mut state = self.state.lock().unwrap();
        // See `ScxLazyPairwiseMapping::fetch` for the double-check rationale.
        if let Some(Some(cached)) = state.get(key) {
            return Ok(cached.clone_ref(py));
        }
        state.insert(key.to_string(), Some(obj.clone_ref(py)));
        Ok(obj)
    }
}

#[pymethods]
impl ScxLazyVarmMapping {
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        self.fetch(py, key)
    }

    fn __setitem__(&self, key: String, value: Py<PyAny>) {
        self.state.lock().unwrap().insert(key, Some(value));
    }

    fn __delitem__(&self, key: &str) -> PyResult<()> {
        let mut state = self.state.lock().unwrap();
        if state.remove(key).is_none() {
            return Err(PyKeyError::new_err(key.to_string()));
        }
        Ok(())
    }

    fn __contains__(&self, key: &str) -> bool {
        self.state.lock().unwrap().contains_key(key)
    }

    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let keys: Vec<String> = slf.state.lock().unwrap().keys().cloned().collect();
        let list = PyList::new(py, &keys)?;
        let iter = list.try_iter()?;
        Ok(iter.into_pyobject(py)?.into_any().unbind())
    }

    fn __len__(&self) -> usize {
        self.state.lock().unwrap().len()
    }

    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let keys: Vec<String> = self.state.lock().unwrap().keys().cloned().collect();
        PyList::new(py, &keys)
    }

    fn values(slf: PyRef<'_, Self>) -> PyResult<Py<ScxLazyValueIterator>> {
        let py = slf.py();
        let keys: Vec<String> = slf.state.lock().unwrap().keys().cloned().collect();
        let parent: Py<PyAny> = slf.into_pyobject(py)?.into_any().unbind();
        Py::new(
            py,
            ScxLazyValueIterator {
                parent,
                keys: keys.into_iter(),
            },
        )
    }

    fn items(slf: PyRef<'_, Self>) -> PyResult<Py<ScxLazyItemIterator>> {
        let py = slf.py();
        let keys: Vec<String> = slf.state.lock().unwrap().keys().cloned().collect();
        let parent: Py<PyAny> = slf.into_pyobject(py)?.into_any().unbind();
        Py::new(
            py,
            ScxLazyItemIterator {
                parent,
                keys: keys.into_iter(),
            },
        )
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        match self.fetch(py, key) {
            Ok(v) => Ok(v),
            Err(e) if e.is_instance_of::<PyKeyError>(py) => {
                Ok(default.unwrap_or_else(|| py.None()))
            }
            Err(e) => Err(e),
        }
    }

    fn __repr__(&self) -> String {
        let state = self.state.lock().unwrap();
        let n_keys = state.len();
        let n_cached = state.values().filter(|v| v.is_some()).count();
        format!(
            "ScxLazyVarmMapping({} keys, {} materialized)",
            n_keys, n_cached
        )
    }
}

/// Lazy mapping for `ad.obsm`. Each value is decoded to a dense numpy
/// 2-D array on first access (mirrors the eager pre-fix behaviour at
/// `anndata.rs:obsm_batch_to_numpy`), with the file's deletion vectors
/// applied so rows line up with `obs` — unlike [`ScxLazyVarmMapping`],
/// which sits on the `var` axis and needs no row filtering.
///
/// Unlike `obsp` / `varp` / `varm` / `layers`, `obsm` is eager by
/// default in `to_anndata()`. This lazy bridge is only installed when
/// the caller has explicitly opted into selective loading
/// (`to_anndata(obsm=[...], eager=False)`), so default behaviour stays
/// byte-identical. The `obsm_filter` passed to [`Self::new`] is the same
/// key set used by the eager selective path.
#[pyclass(name = "ScxLazyObsmMapping", mapping)]
pub struct ScxLazyObsmMapping {
    reader: Arc<ScxReader>,
    /// `Some(_)` enables the Phase-3 backed dense row-gather mode: each
    /// key materialises to a [`ScxBackedObsmDataset`] (shard-aware,
    /// `O(batch)` memory) instead of a full dense numpy array.
    backed: Option<BackedObsmConfig>,
    state: Mutex<HashMap<String, Option<Py<PyAny>>>>,
}

/// Construction inputs for a backed (row-gather) obsm value. Each key
/// opens its own `BackedDenseReader` over a sibling `ScxReader` (sharing
/// the parsed catalog), so per-key LRU caches stay independent and
/// fork-safe — mirroring the per-layer `BackedCsrReader` opens.
pub(crate) struct BackedObsmConfig {
    pub(crate) path: std::path::PathBuf,
    pub(crate) cache_shards: usize,
    pub(crate) shared_catalog: Arc<scx_format::FullCatalog>,
    /// User-visible row i → global file row (deletion vectors). `None` =
    /// identity. `obs_filter` composition is intentionally excluded
    /// (the backed-obsm path falls back to eager under `obs_filter`).
    pub(crate) kept_to_global: Option<Arc<Vec<u64>>>,
}

impl ScxLazyObsmMapping {
    pub(crate) fn new(reader: Arc<ScxReader>, obsm_filter: Option<&[String]>) -> Self {
        Self::new_inner(reader, obsm_filter, None)
    }

    /// Phase 3: backed dense row-gather mode. `reader` is used only for
    /// key enumeration; each value is built from `config`.
    pub(crate) fn new_backed(
        reader: Arc<ScxReader>,
        obsm_filter: Option<&[String]>,
        config: BackedObsmConfig,
    ) -> Self {
        Self::new_inner(reader, obsm_filter, Some(config))
    }

    fn new_inner(
        reader: Arc<ScxReader>,
        obsm_filter: Option<&[String]>,
        backed: Option<BackedObsmConfig>,
    ) -> Self {
        let mut keys = reader.list_obsm();
        if let Some(filter) = obsm_filter {
            keys.retain(|n| filter.iter().any(|f| f == n));
        }
        let state: HashMap<String, Option<Py<PyAny>>> =
            keys.into_iter().map(|k| (k, None)).collect();
        Self {
            reader,
            backed,
            state: Mutex::new(state),
        }
    }

    fn fetch(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        {
            let state = self.state.lock().unwrap();
            match state.get(key) {
                Some(Some(obj)) => return Ok(obj.clone_ref(py)),
                Some(None) => {}
                None => return Err(PyKeyError::new_err(key.to_string())),
            }
        }
        let obj: Py<PyAny> = match &self.backed {
            Some(cfg) => {
                // Backed row-gather: open a sibling reader (shared catalog)
                // and wrap a per-key BackedDenseReader.
                let r =
                    ScxReader::open_with_shared_catalog(&cfg.path, Arc::clone(&cfg.shared_catalog))
                        .map_err(to_pyerr)?;
                let backed = Arc::new(
                    scx_format::BackedDenseReader::new_obsm(r, key, cfg.cache_shards)
                        .map_err(to_pyerr)?,
                );
                let ds = match &cfg.kept_to_global {
                    Some(k) => crate::backed::ScxBackedObsmDataset::from_reader_with_deletions(
                        backed,
                        cfg.cache_shards,
                        key.to_string(),
                        k.as_ref().clone(),
                    ),
                    None => crate::backed::ScxBackedObsmDataset::from_reader(
                        backed,
                        cfg.cache_shards,
                        key.to_string(),
                    ),
                };
                ds.into_pyobject(py)?.into_any().unbind()
            }
            None => {
                let batch = self.reader.read_obsm(key).map_err(to_pyerr)?;
                // Apply deletion vectors so the dense array's rows match
                // `obs` (the eager path does the same — see
                // `to_anndata_with_layers`).
                let filtered = filter_obs_by_deletion_vectors(&self.reader, batch)?;
                obsm_batch_to_numpy(py, &filtered)?.unbind()
            }
        };
        let mut state = self.state.lock().unwrap();
        // See `ScxLazyPairwiseMapping::fetch` for the double-check rationale.
        if let Some(Some(cached)) = state.get(key) {
            return Ok(cached.clone_ref(py));
        }
        state.insert(key.to_string(), Some(obj.clone_ref(py)));
        Ok(obj)
    }
}

#[pymethods]
impl ScxLazyObsmMapping {
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        self.fetch(py, key)
    }

    fn __setitem__(&self, key: String, value: Py<PyAny>) {
        self.state.lock().unwrap().insert(key, Some(value));
    }

    fn __delitem__(&self, key: &str) -> PyResult<()> {
        let mut state = self.state.lock().unwrap();
        if state.remove(key).is_none() {
            return Err(PyKeyError::new_err(key.to_string()));
        }
        Ok(())
    }

    fn __contains__(&self, key: &str) -> bool {
        self.state.lock().unwrap().contains_key(key)
    }

    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let keys: Vec<String> = slf.state.lock().unwrap().keys().cloned().collect();
        let list = PyList::new(py, &keys)?;
        let iter = list.try_iter()?;
        Ok(iter.into_pyobject(py)?.into_any().unbind())
    }

    fn __len__(&self) -> usize {
        self.state.lock().unwrap().len()
    }

    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let keys: Vec<String> = self.state.lock().unwrap().keys().cloned().collect();
        PyList::new(py, &keys)
    }

    fn values(slf: PyRef<'_, Self>) -> PyResult<Py<ScxLazyValueIterator>> {
        let py = slf.py();
        let keys: Vec<String> = slf.state.lock().unwrap().keys().cloned().collect();
        let parent: Py<PyAny> = slf.into_pyobject(py)?.into_any().unbind();
        Py::new(
            py,
            ScxLazyValueIterator {
                parent,
                keys: keys.into_iter(),
            },
        )
    }

    fn items(slf: PyRef<'_, Self>) -> PyResult<Py<ScxLazyItemIterator>> {
        let py = slf.py();
        let keys: Vec<String> = slf.state.lock().unwrap().keys().cloned().collect();
        let parent: Py<PyAny> = slf.into_pyobject(py)?.into_any().unbind();
        Py::new(
            py,
            ScxLazyItemIterator {
                parent,
                keys: keys.into_iter(),
            },
        )
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        match self.fetch(py, key) {
            Ok(v) => Ok(v),
            Err(e) if e.is_instance_of::<PyKeyError>(py) => {
                Ok(default.unwrap_or_else(|| py.None()))
            }
            Err(e) => Err(e),
        }
    }

    fn __repr__(&self) -> String {
        let state = self.state.lock().unwrap();
        let n_keys = state.len();
        let n_cached = state.values().filter(|v| v.is_some()).count();
        format!(
            "ScxLazyObsmMapping({} keys, {} materialized)",
            n_keys, n_cached
        )
    }
}

/// Lazy mapping for `ad.layers` in the **non-backed** `to_anndata()`
/// path. Each value is decoded to a scipy `csr_matrix` on first
/// access via `ScxReader::read_layer_filtered` (which applies the
/// file's deletion vector, matching the eager pre-fix code at
/// `anndata.rs:read_layer_filtered`).
///
/// The backed `to_anndata_backed()` path continues to use the
/// existing per-layer `ScxBackedLayerDataset` wrappers, which are
/// already lazy at the row level and cheap to construct.
#[pyclass(name = "ScxLazyLayersMapping", mapping)]
pub struct ScxLazyLayersMapping {
    reader: Arc<ScxReader>,
    state: Mutex<HashMap<String, Option<Py<PyAny>>>>,
}

impl ScxLazyLayersMapping {
    pub(crate) fn new(reader: Arc<ScxReader>, layer_filter: Option<&[String]>) -> Self {
        let mut keys = reader.layer_names();
        if let Some(filter) = layer_filter {
            keys.retain(|n| filter.iter().any(|f| f == n));
        }
        let state: HashMap<String, Option<Py<PyAny>>> =
            keys.into_iter().map(|k| (k, None)).collect();
        Self {
            reader,
            state: Mutex::new(state),
        }
    }

    pub(crate) fn materialize_all<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let keys: Vec<String> = self.state.lock().unwrap().keys().cloned().collect();
        let dict = PyDict::new(py);
        for key in keys {
            let value = self.fetch(py, &key)?;
            dict.set_item(&key, value)?;
        }
        Ok(dict)
    }

    fn fetch(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        {
            let state = self.state.lock().unwrap();
            match state.get(key) {
                Some(Some(obj)) => return Ok(obj.clone_ref(py)),
                Some(None) => {}
                None => return Err(PyKeyError::new_err(key.to_string())),
            }
        }
        let csr = self.reader.read_layer_filtered(key).map_err(to_pyerr)?;
        let bound = csr_to_scipy(py, csr)?;
        let obj: Py<PyAny> = bound.unbind();
        let mut state = self.state.lock().unwrap();
        // See `ScxLazyPairwiseMapping::fetch` for the double-check rationale.
        if let Some(Some(cached)) = state.get(key) {
            return Ok(cached.clone_ref(py));
        }
        state.insert(key.to_string(), Some(obj.clone_ref(py)));
        Ok(obj)
    }
}

#[pymethods]
impl ScxLazyLayersMapping {
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        self.fetch(py, key)
    }

    fn __setitem__(&self, key: String, value: Py<PyAny>) {
        self.state.lock().unwrap().insert(key, Some(value));
    }

    fn __delitem__(&self, key: &str) -> PyResult<()> {
        let mut state = self.state.lock().unwrap();
        if state.remove(key).is_none() {
            return Err(PyKeyError::new_err(key.to_string()));
        }
        Ok(())
    }

    fn __contains__(&self, key: &str) -> bool {
        self.state.lock().unwrap().contains_key(key)
    }

    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let keys: Vec<String> = slf.state.lock().unwrap().keys().cloned().collect();
        let list = PyList::new(py, &keys)?;
        let iter = list.try_iter()?;
        Ok(iter.into_pyobject(py)?.into_any().unbind())
    }

    fn __len__(&self) -> usize {
        self.state.lock().unwrap().len()
    }

    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let keys: Vec<String> = self.state.lock().unwrap().keys().cloned().collect();
        PyList::new(py, &keys)
    }

    fn values(slf: PyRef<'_, Self>) -> PyResult<Py<ScxLazyValueIterator>> {
        let py = slf.py();
        let keys: Vec<String> = slf.state.lock().unwrap().keys().cloned().collect();
        let parent: Py<PyAny> = slf.into_pyobject(py)?.into_any().unbind();
        Py::new(
            py,
            ScxLazyValueIterator {
                parent,
                keys: keys.into_iter(),
            },
        )
    }

    fn items(slf: PyRef<'_, Self>) -> PyResult<Py<ScxLazyItemIterator>> {
        let py = slf.py();
        let keys: Vec<String> = slf.state.lock().unwrap().keys().cloned().collect();
        let parent: Py<PyAny> = slf.into_pyobject(py)?.into_any().unbind();
        Py::new(
            py,
            ScxLazyItemIterator {
                parent,
                keys: keys.into_iter(),
            },
        )
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        match self.fetch(py, key) {
            Ok(v) => Ok(v),
            Err(e) if e.is_instance_of::<PyKeyError>(py) => {
                Ok(default.unwrap_or_else(|| py.None()))
            }
            Err(e) => Err(e),
        }
    }

    fn __repr__(&self) -> String {
        let state = self.state.lock().unwrap();
        let n_keys = state.len();
        let n_cached = state.values().filter(|v| v.is_some()).count();
        format!(
            "ScxLazyLayersMapping({} keys, {} materialized)",
            n_keys, n_cached
        )
    }
}

/// Lazy iterator over the values of an `ScxLazy*Mapping`. Each
/// `__next__` call delegates to the parent's `__getitem__` (which
/// routes through the parent's `fetch`, so it picks up the same caching
/// + double-checked-insert behaviour). Holds the parent as
/// `Py<PyAny>` rather than a typed handle so a single iterator class
/// works for all three mapping kinds.
#[pyclass(name = "ScxLazyValueIterator")]
pub struct ScxLazyValueIterator {
    parent: Py<PyAny>,
    keys: std::vec::IntoIter<String>,
}

#[pymethods]
impl ScxLazyValueIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let Some(k) = slf.keys.next() else {
            return Ok(None);
        };
        let v = slf.parent.bind(py).get_item(&k)?;
        Ok(Some(v.unbind()))
    }
}

/// Lazy iterator over `(key, value)` pairs of an `ScxLazy*Mapping`.
/// See [`ScxLazyValueIterator`].
#[pyclass(name = "ScxLazyItemIterator")]
pub struct ScxLazyItemIterator {
    parent: Py<PyAny>,
    keys: std::vec::IntoIter<String>,
}

#[pymethods]
impl ScxLazyItemIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let Some(k) = slf.keys.next() else {
            return Ok(None);
        };
        let v = slf.parent.bind(py).get_item(&k)?;
        let tup = PyTuple::new(py, &[k.into_pyobject(py)?.into_any(), v])?;
        Ok(Some(tup.into_any().unbind()))
    }
}
