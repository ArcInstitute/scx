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
    coo_record_batch_to_scipy, csr_to_scipy, filter_coo_obsp_by_kept_rows, obsm_batch_to_numpy,
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
    state: Mutex<HashMap<String, Option<PyObject>>>,
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
        let state: HashMap<String, Option<PyObject>> =
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

    fn fetch(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
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
        let obj: PyObject = bound.unbind();
        self.state
            .lock()
            .unwrap()
            .insert(key.to_string(), Some(obj.clone_ref(py)));
        Ok(obj)
    }
}

#[pymethods]
impl ScxLazyPairwiseMapping {
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        self.fetch(py, key)
    }

    fn __setitem__(&self, key: String, value: PyObject) {
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

    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<PyObject> {
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

    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let keys: Vec<String> = self.state.lock().unwrap().keys().cloned().collect();
        let list = PyList::empty(py);
        for k in keys {
            list.append(self.fetch(py, &k)?)?;
        }
        Ok(list)
    }

    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let keys: Vec<String> = self.state.lock().unwrap().keys().cloned().collect();
        let list = PyList::empty(py);
        for k in keys {
            let v = self.fetch(py, &k)?;
            let tuple = PyTuple::new(py, &[k.into_pyobject(py)?.into_any(), v.into_bound(py)])?;
            list.append(tuple)?;
        }
        Ok(list)
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<PyObject>) -> PyObject {
        match self.fetch(py, key) {
            Ok(v) => v,
            Err(_) => default.unwrap_or_else(|| py.None()),
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
    state: Mutex<HashMap<String, Option<PyObject>>>,
}

impl ScxLazyVarmMapping {
    pub(crate) fn new(reader: Arc<ScxReader>) -> Self {
        let state: HashMap<String, Option<PyObject>> =
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

    fn fetch(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
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
        let obj: PyObject = bound.unbind();
        self.state
            .lock()
            .unwrap()
            .insert(key.to_string(), Some(obj.clone_ref(py)));
        Ok(obj)
    }
}

#[pymethods]
impl ScxLazyVarmMapping {
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        self.fetch(py, key)
    }

    fn __setitem__(&self, key: String, value: PyObject) {
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

    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<PyObject> {
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

    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let keys: Vec<String> = self.state.lock().unwrap().keys().cloned().collect();
        let list = PyList::empty(py);
        for k in keys {
            list.append(self.fetch(py, &k)?)?;
        }
        Ok(list)
    }

    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let keys: Vec<String> = self.state.lock().unwrap().keys().cloned().collect();
        let list = PyList::empty(py);
        for k in keys {
            let v = self.fetch(py, &k)?;
            let tuple = PyTuple::new(py, &[k.into_pyobject(py)?.into_any(), v.into_bound(py)])?;
            list.append(tuple)?;
        }
        Ok(list)
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<PyObject>) -> PyObject {
        match self.fetch(py, key) {
            Ok(v) => v,
            Err(_) => default.unwrap_or_else(|| py.None()),
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
    state: Mutex<HashMap<String, Option<PyObject>>>,
}

impl ScxLazyLayersMapping {
    pub(crate) fn new(reader: Arc<ScxReader>, layer_filter: Option<&[String]>) -> Self {
        let mut keys = reader.layer_names();
        if let Some(filter) = layer_filter {
            keys.retain(|n| filter.iter().any(|f| f == n));
        }
        let state: HashMap<String, Option<PyObject>> =
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

    fn fetch(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
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
        let obj: PyObject = bound.unbind();
        self.state
            .lock()
            .unwrap()
            .insert(key.to_string(), Some(obj.clone_ref(py)));
        Ok(obj)
    }
}

#[pymethods]
impl ScxLazyLayersMapping {
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        self.fetch(py, key)
    }

    fn __setitem__(&self, key: String, value: PyObject) {
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

    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<PyObject> {
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

    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let keys: Vec<String> = self.state.lock().unwrap().keys().cloned().collect();
        let list = PyList::empty(py);
        for k in keys {
            list.append(self.fetch(py, &k)?)?;
        }
        Ok(list)
    }

    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let keys: Vec<String> = self.state.lock().unwrap().keys().cloned().collect();
        let list = PyList::empty(py);
        for k in keys {
            let v = self.fetch(py, &k)?;
            let tuple = PyTuple::new(py, &[k.into_pyobject(py)?.into_any(), v.into_bound(py)])?;
            list.append(tuple)?;
        }
        Ok(list)
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<PyObject>) -> PyObject {
        match self.fetch(py, key) {
            Ok(v) => v,
            Err(_) => default.unwrap_or_else(|| py.None()),
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
