// uns serialize/deserialize (Python <-> JSON, plain & tagged).
//
// Extracted from the former pyscx/src/anndata.rs (T5.7).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use std::collections::HashSet;

use scx_format::ScxReader;

use crate::to_pyerr;

use crate::convert::*;

/// Sentinel key marking a tagged envelope in the on-disk JSON.
const SCX_TYPE_KEY: &str = "__scx_type__";

/// Convert a `serde_json::Value` (uns JSON tree) into a Python dict
/// using the existing `__scx_type__` envelope decoder. Thin wrapper
/// around [`json_to_py`] for use from other pyscx modules that don't
/// want to construct [`UnsReadCtx`] directly. Gated on the `hdf5`
/// feature because its sole callers (`pyscx.read_h5ad_metadata`,
/// `pyscx.from_h5ad`) are h5ad-only.
#[cfg(feature = "hdf5")]
pub(crate) fn uns_json_to_py<'py>(
    py: Python<'py>,
    val: &serde_json::Value,
) -> PyResult<Bound<'py, PyAny>> {
    let mut ctx = UnsReadCtx::new(py)?;
    json_to_py(val, &mut ctx)
}

/// Convert a Python object (typically a dict supplied as `uns_override`)
/// into a `serde_json::Value` using the existing tagged-envelope writer.
/// Thin wrapper around [`normalize_uns_value`]. Reachable without the
/// `hdf5` feature so the in-place `set_uns` / `modify_metadata` bindings
/// (`crate::ops`) can serialize a `uns` dict on any build.
pub(crate) fn uns_py_to_json<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    uns_format: UnsFormat,
) -> PyResult<serde_json::Value> {
    let np = py.import("numpy")?;
    let np_generic = np.getattr("generic")?;
    let np_ndarray = np.getattr("ndarray")?;
    let mut ctx = UnsWriteCtx::new(uns_format, &np_generic, &np_ndarray);
    normalize_uns_value(obj, "uns", &mut ctx)
}

/// Mutable context threaded through the writer so per-value handlers share
/// the NumPy module handles and a lazy-imported pandas module without
/// repeating `py.import("pandas")` on every dispatch.
pub(crate) struct UnsWriteCtx<'a, 'py> {
    format: UnsFormat,
    np_generic: &'a Bound<'py, PyAny>,
    np_ndarray: &'a Bound<'py, PyAny>,
    pd_lazy: Option<Bound<'py, PyModule>>,
    visiting: HashSet<usize>,
}

impl<'a, 'py> UnsWriteCtx<'a, 'py> {
    pub(crate) fn new(
        format: UnsFormat,
        np_generic: &'a Bound<'py, PyAny>,
        np_ndarray: &'a Bound<'py, PyAny>,
    ) -> Self {
        Self {
            format,
            np_generic,
            np_ndarray,
            pd_lazy: None,
            visiting: HashSet::new(),
        }
    }

    /// Lazy pandas import. Cached per write so we pay at most one
    /// `import pandas` per `from_anndata()` call, and only when the input
    /// actually contains a pandas object under tagged mode.
    fn pandas(&mut self, py: Python<'py>) -> PyResult<&Bound<'py, PyModule>> {
        if self.pd_lazy.is_none() {
            self.pd_lazy = Some(py.import("pandas")?);
        }
        Ok(self.pd_lazy.as_ref().unwrap())
    }
}

/// True if a NumPy `dtype.kind` is a fixed-width numeric type whose buffer
/// can be stored verbatim as little-endian bytes (bool / int / uint / float).
/// Excludes complex (`c`), datetime (`M`), timedelta (`m`), object (`O`),
/// and string (`U`/`S`) kinds, which take separate write paths or error out.
pub(crate) fn is_numeric_kind(kind: &str) -> bool {
    matches!(kind, "b" | "i" | "u" | "f")
}

/// Recursively normalize a Python value into a `serde_json::Value` so it can
/// be written into the SCX `uns` section. Two modes:
///
/// - `Plain` — legacy lossy path. NumPy arrays/scalars and pandas
///   `Index`/`Series`/`Categorical` collapse to plain JSON lists. `.tolist()`
///   fallback covers duck-typed array-likes.
/// - `Tagged` — wraps NumPy arrays, NumPy scalars, tuples, structured
///   recarrays, and the three pandas container types in `__scx_type__`
///   envelopes so the read path can reconstruct the original Python type.
///   Numeric arrays/scalars store raw bytes as base64-LE; object/string
///   arrays store a JSON list of elements. See [`UnsFormat::Tagged`].
///
/// In both modes:
/// - `None` → `null`
/// - `bool` → `bool` (checked before `int`)
/// - `int` → JSON number (i64 / u64; out-of-range errors)
/// - `float` → JSON number (non-finite raw Python floats still error in
///   tagged mode — only ndarray-backed NaN/Inf round-trips, since the base64
///   envelope preserves raw bytes)
/// - `str` → string
/// - `dict` → object; non-string keys are stringified via `str(k)`
/// - `list` → JSON array
/// - `bytes` → error (no portable JSON representation)
///
/// `key_path` accumulates a Python-style accessor (e.g.
/// `uns['rank_genes_groups']['names'][0]`) for inclusion in error messages.
///
/// `ctx.visiting` tracks Py<PyAny> identities currently on the recursion stack
/// for container branches (dict / list / tuple / `.tolist()` fallback). A
/// repeat hit means the input contains a cycle (e.g. `d = {}; d["x"] = d`).
/// We raise `ValueError` instead of recursing into a Rust stack overflow.
pub(crate) fn normalize_uns_value<'py>(
    obj: &Bound<'py, PyAny>,
    key_path: &str,
    ctx: &mut UnsWriteCtx<'_, 'py>,
) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }

    // NumPy scalar / array first: in NumPy 1.x some scalars subclass Python
    // numeric types, so we must dispatch on np.generic before bool/int/float.
    if obj.is_instance(ctx.np_generic)? {
        match ctx.format {
            UnsFormat::Plain => {
                let item = obj.call_method0("item")?;
                return normalize_uns_value(&item, key_path, ctx);
            }
            UnsFormat::Tagged => {
                return encode_np_scalar_tagged(obj, key_path, ctx);
            }
        }
    }
    if obj.is_instance(ctx.np_ndarray)? {
        match ctx.format {
            UnsFormat::Plain => {
                let lst = obj.call_method0("tolist")?;
                return normalize_uns_value(&lst, key_path, ctx);
            }
            UnsFormat::Tagged => {
                return encode_ndarray_tagged(obj, key_path, ctx);
            }
        }
    }

    // bool before int: Python bool is a subclass of int.
    if obj.cast::<PyBool>().is_ok() {
        return Ok(serde_json::Value::Bool(obj.extract::<bool>()?));
    }

    if obj.cast::<PyInt>().is_ok() {
        if let Ok(i) = obj.extract::<i64>() {
            return Ok(serde_json::Value::Number(i.into()));
        }
        if let Ok(u) = obj.extract::<u64>() {
            return Ok(serde_json::Value::Number(u.into()));
        }
        return Err(PyValueError::new_err(format!(
            "uns at {key_path}: integer is too large for JSON (must fit in i64 or u64)"
        )));
    }

    if obj.cast::<PyFloat>().is_ok() {
        let f: f64 = obj.extract()?;
        if !f.is_finite() {
            return Err(PyValueError::new_err(format!(
                "uns at {key_path}: non-finite float ({f}) cannot be serialized to JSON"
            )));
        }
        return serde_json::Number::from_f64(f)
            .map(serde_json::Value::Number)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "uns at {key_path}: float {f} cannot be represented in JSON"
                ))
            });
    }

    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(serde_json::Value::String(s.extract()?));
    }

    if obj.cast::<PyBytes>().is_ok() {
        return Err(PyValueError::new_err(format!(
            "uns at {key_path}: bytes are not JSON-serializable"
        )));
    }

    let id = obj.as_ptr() as usize;
    if !ctx.visiting.insert(id) {
        return Err(PyValueError::new_err(format!(
            "uns at {key_path}: circular reference detected"
        )));
    }
    let result = normalize_container(obj, key_path, ctx);
    ctx.visiting.remove(&id);
    result
}

/// Container / fallback dispatch — split out so `normalize_uns_value` can
/// wrap it with cycle-tracking insert/remove.
pub(crate) fn normalize_container<'py>(
    obj: &Bound<'py, PyAny>,
    key_path: &str,
    ctx: &mut UnsWriteCtx<'_, 'py>,
) -> PyResult<serde_json::Value> {
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = serde_json::Map::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            let key_str: String = k.str()?.extract()?;
            let new_path = format!("{key_path}['{key_str}']");
            map.insert(key_str, normalize_uns_value(&v, &new_path, ctx)?);
        }
        return Ok(serde_json::Value::Object(map));
    }

    if let Ok(lst) = obj.cast::<PyList>() {
        let mut arr = Vec::with_capacity(lst.len());
        for (i, item) in lst.iter().enumerate() {
            let new_path = format!("{key_path}[{i}]");
            arr.push(normalize_uns_value(&item, &new_path, ctx)?);
        }
        return Ok(serde_json::Value::Array(arr));
    }

    if let Ok(tup) = obj.cast::<PyTuple>() {
        let mut arr = Vec::with_capacity(tup.len());
        for (i, item) in tup.iter().enumerate() {
            let new_path = format!("{key_path}[{i}]");
            arr.push(normalize_uns_value(&item, &new_path, ctx)?);
        }
        match ctx.format {
            UnsFormat::Plain => return Ok(serde_json::Value::Array(arr)),
            UnsFormat::Tagged => {
                let mut env = serde_json::Map::with_capacity(2);
                env.insert(
                    SCX_TYPE_KEY.to_string(),
                    serde_json::Value::String("tuple".to_string()),
                );
                env.insert("data".to_string(), serde_json::Value::Array(arr));
                return Ok(serde_json::Value::Object(env));
            }
        }
    }

    // Tagged mode: detect pandas Index / Series / Categorical before the
    // generic `.tolist()` fallback so the original container type round-trips
    // with its metadata (name, codes, categories, ordered).
    if ctx.format == UnsFormat::Tagged {
        if let Some(env) = encode_pandas_tagged(obj, key_path, ctx)? {
            return Ok(env);
        }
    }

    // Generic fallback: any object exposing a callable `.tolist()`. Covers
    // pandas Index / Series / Categorical (in `Plain` mode) and any
    // duck-typed array-like.
    if let Ok(method) = obj.getattr("tolist") {
        if method.is_callable() {
            let lst = method.call0()?;
            return normalize_uns_value(&lst, key_path, ctx);
        }
    }

    let type_name: String = obj.get_type().getattr("__name__")?.extract()?;
    Err(PyValueError::new_err(format!(
        "uns at {key_path}: cannot serialize {type_name} to JSON; supported types are None, bool, int, float, str, dict, list, tuple, NumPy arrays/scalars, and any object exposing a callable .tolist() (pandas Series/Index/Categorical)"
    )))
}

/// Wrap a NumPy scalar (`np.generic` instance) in a `scalar` envelope under
/// tagged mode. The 1-element raw byte buffer is base64-LE-encoded so the
/// scalar's dtype (e.g. `float32`, `int64`, `bool`) survives the round-trip.
/// Non-base64 dtypes (datetime, complex, …) fall back to `.item()` and the
/// usual plain-JSON path so the value still serializes.
pub(crate) fn encode_np_scalar_tagged<'py>(
    obj: &Bound<'py, PyAny>,
    key_path: &str,
    ctx: &mut UnsWriteCtx<'_, 'py>,
) -> PyResult<serde_json::Value> {
    let dtype = obj.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    if !is_numeric_kind(&kind) {
        let item = obj.call_method0("item")?;
        return normalize_uns_value(&item, key_path, ctx);
    }
    // `dtype.str` (e.g. "<f4", "|b1") carries explicit byte order, so the
    // round-trip is portable across endianness: the read side reconstructs
    // the dtype from this label rather than relying on native byte order.
    let dtype_str: String = dtype.getattr("str")?.extract()?;
    // Wrap the scalar in a 0-d array so we can reuse ndarray byte conversion.
    let np = ctx.np_generic.py().import("numpy")?;
    let arr = np.call_method1("asarray", (obj,))?;
    let bytes = ndarray_bytes_le(&arr, key_path)?;
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    let mut env = serde_json::Map::with_capacity(3);
    env.insert(
        SCX_TYPE_KEY.to_string(),
        serde_json::Value::String("scalar".to_string()),
    );
    env.insert("dtype".to_string(), serde_json::Value::String(dtype_str));
    env.insert("data".to_string(), serde_json::Value::String(b64));
    Ok(serde_json::Value::Object(env))
}

/// Encode an `np.ndarray` as a tagged JSON envelope.
///
/// Dispatch by `dtype.kind`:
/// - `b`/`i`/`u`/`f` (bool/int/uint/float): base64-LE raw bytes.
/// - `O`/`U`/`S` (object/unicode/bytes string): JSON list of strings.
/// - `V` (structured): `recarray` envelope with `dtype.descr` + base64-LE
///   raw bytes. Lets `rank_genes_groups["names"]`-style structured arrays
///   round-trip with their field names and per-field dtypes intact.
/// - `M`/`m`/`c` (datetime / timedelta / complex): error, not yet supported.
pub(crate) fn encode_ndarray_tagged<'py>(
    obj: &Bound<'py, PyAny>,
    key_path: &str,
    _ctx: &mut UnsWriteCtx<'_, 'py>,
) -> PyResult<serde_json::Value> {
    let dtype = obj.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    let shape: Vec<usize> = obj.getattr("shape")?.extract()?;
    let shape_json: Vec<serde_json::Value> = shape
        .iter()
        .map(|s| serde_json::Value::Number((*s as u64).into()))
        .collect();

    let mut env = serde_json::Map::new();
    env.insert(
        SCX_TYPE_KEY.to_string(),
        serde_json::Value::String("ndarray".to_string()),
    );

    match kind.as_str() {
        "b" | "i" | "u" | "f" => {
            // `dtype.str` (e.g. "<f4", "|b1") encodes byte order explicitly,
            // matching the `base64le` byte stream the read side decodes.
            let dtype_str: String = dtype.getattr("str")?.extract()?;
            let bytes = ndarray_bytes_le(obj, key_path)?;
            use base64::Engine;
            let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
            env.insert("dtype".to_string(), serde_json::Value::String(dtype_str));
            env.insert("shape".to_string(), serde_json::Value::Array(shape_json));
            env.insert(
                "encoding".to_string(),
                serde_json::Value::String("base64le".to_string()),
            );
            env.insert("data".to_string(), serde_json::Value::String(b64));
            Ok(serde_json::Value::Object(env))
        }
        "O" | "U" | "S" => {
            let lst = obj.call_method0("tolist")?;
            let data = pylist_to_string_json_array(&lst, key_path)?;
            // For O the payload is a JSON list of pickled-Python-strings, so
            // the byte-order prefix in `dtype.str` ("|O") would be misleading;
            // we keep the explicit "object" sentinel. For U/S the `dtype.str`
            // form (e.g. "<U10", "|S5") carries width and byte order without
            // assuming UCS-4 from `itemsize / 4`.
            let dtype_label: String = if kind == "O" {
                "object".to_string()
            } else {
                dtype.getattr("str")?.extract()?
            };
            env.insert(
                "dtype".to_string(),
                serde_json::Value::String(dtype_label),
            );
            env.insert("shape".to_string(), serde_json::Value::Array(shape_json));
            env.insert(
                "encoding".to_string(),
                serde_json::Value::String("json".to_string()),
            );
            env.insert("data".to_string(), data);
            Ok(serde_json::Value::Object(env))
        }
        "V" => {
            // Structured ndarray (recarray-like). Save descr + raw bytes.
            let descr_py = dtype.getattr("descr")?;
            let descr_json = pytuple_descr_to_json(&descr_py, key_path)?;
            let bytes = ndarray_bytes_le(obj, key_path)?;
            use base64::Engine;
            let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
            let mut env = serde_json::Map::new();
            env.insert(
                SCX_TYPE_KEY.to_string(),
                serde_json::Value::String("recarray".to_string()),
            );
            env.insert("descr".to_string(), descr_json);
            env.insert("shape".to_string(), serde_json::Value::Array(shape_json));
            env.insert(
                "encoding".to_string(),
                serde_json::Value::String("base64le".to_string()),
            );
            env.insert("data".to_string(), serde_json::Value::String(b64));
            Ok(serde_json::Value::Object(env))
        }
        other => Err(PyValueError::new_err(format!(
            "uns at {key_path}: ndarray dtype kind '{other}' is not supported in uns_format='tagged' (got dtype.kind={other:?}); supported kinds are b/i/u/f (numeric), O/U/S (object/string), V (structured)"
        ))),
    }
}

/// Walk a Python list-of-strings (possibly nested for multi-dim arrays) into
/// a JSON array, asserting that every leaf is a `str`. Used for object/string
/// dtype ndarrays in tagged mode.
pub(crate) fn pylist_to_string_json_array<'py>(
    obj: &Bound<'py, PyAny>,
    key_path: &str,
) -> PyResult<serde_json::Value> {
    if let Ok(lst) = obj.cast::<PyList>() {
        let mut arr = Vec::with_capacity(lst.len());
        for (i, item) in lst.iter().enumerate() {
            let new_path = format!("{key_path}[{i}]");
            arr.push(pylist_to_string_json_array(&item, &new_path)?);
        }
        return Ok(serde_json::Value::Array(arr));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(serde_json::Value::String(s.extract()?));
    }
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if obj.cast::<PyBytes>().is_ok() {
        // Decode UTF-8 bytes; reject otherwise.
        let b: &[u8] = obj.cast::<PyBytes>().unwrap().as_bytes();
        let s = std::str::from_utf8(b).map_err(|_| {
            PyValueError::new_err(format!(
                "uns at {key_path}: bytes element in object/string array is not valid UTF-8"
            ))
        })?;
        return Ok(serde_json::Value::String(s.to_string()));
    }
    let type_name: String = obj.get_type().getattr("__name__")?.extract()?;
    Err(PyValueError::new_err(format!(
        "uns at {key_path}: object/string ndarray element must be str or None, got {type_name}"
    )))
}

/// Convert a NumPy `dtype.descr` (a list of `(name, fmt)` or
/// `(name, fmt, shape)` tuples) into a JSON-friendly array of arrays.
pub(crate) fn pytuple_descr_to_json<'py>(
    descr: &Bound<'py, PyAny>,
    key_path: &str,
) -> PyResult<serde_json::Value> {
    let lst = descr.cast::<PyList>().map_err(|_| {
        PyValueError::new_err(format!(
            "uns at {key_path}: structured dtype.descr is not a list"
        ))
    })?;
    let mut out = Vec::with_capacity(lst.len());
    for (i, item) in lst.iter().enumerate() {
        let tup = item.cast::<PyTuple>().map_err(|_| {
            PyValueError::new_err(format!(
                "uns at {key_path}: dtype.descr[{i}] is not a tuple"
            ))
        })?;
        let mut row = Vec::with_capacity(tup.len());
        for el in tup.iter() {
            if let Ok(s) = el.cast::<PyString>() {
                row.push(serde_json::Value::String(s.extract()?));
            } else if let Ok(t) = el.cast::<PyTuple>() {
                // Nested shape tuple, e.g. ('a', '<i4', (3,)).
                let mut inner = Vec::with_capacity(t.len());
                for d in t.iter() {
                    let n: u64 = d.extract()?;
                    inner.push(serde_json::Value::Number(n.into()));
                }
                row.push(serde_json::Value::Array(inner));
            } else if let Ok(l) = el.cast::<PyList>() {
                // Nested descr for sub-record (recursive).
                let nested = pytuple_descr_to_json(l.as_any(), key_path)?;
                row.push(nested);
            } else {
                let type_name: String = el.get_type().getattr("__name__")?.extract()?;
                return Err(PyValueError::new_err(format!(
                    "uns at {key_path}: unsupported dtype.descr element type {type_name}"
                )));
            }
        }
        out.push(serde_json::Value::Array(row));
    }
    Ok(serde_json::Value::Array(out))
}

/// Get a `Vec<u8>` of an ndarray's raw little-endian bytes, copying as
/// needed to guarantee LE byte order and C-contiguous layout. The byte
/// order conversion is a no-op on typical LE platforms; on BE platforms it
/// produces the correct bytes for the on-disk envelope.
pub(crate) fn ndarray_bytes_le<'py>(arr: &Bound<'py, PyAny>, key_path: &str) -> PyResult<Vec<u8>> {
    let dtype = arr.getattr("dtype")?;
    let le_dtype = dtype.call_method1("newbyteorder", ("<",))?;
    let arr_le = arr.call_method1("astype", (le_dtype,))?;
    let np = arr.py().import("numpy")?;
    let arr_c = np.call_method1("ascontiguousarray", (arr_le,))?;
    let bytes_obj = arr_c.call_method0("tobytes")?;
    let pybytes = bytes_obj.cast::<PyBytes>().map_err(|_| {
        PyValueError::new_err(format!(
            "uns at {key_path}: ndarray.tobytes() did not return bytes"
        ))
    })?;
    Ok(pybytes.as_bytes().to_vec())
}

/// In tagged mode, recognize a pandas `Index` / `Series` / `Categorical`
/// and emit the corresponding envelope. Returns `Ok(None)` if `obj` is not
/// a pandas object (caller falls back to the generic `.tolist()` path).
pub(crate) fn encode_pandas_tagged<'py>(
    obj: &Bound<'py, PyAny>,
    key_path: &str,
    ctx: &mut UnsWriteCtx<'_, 'py>,
) -> PyResult<Option<serde_json::Value>> {
    let py = obj.py();
    let pd = ctx.pandas(py)?.clone();
    let cat_cls = pd.getattr("Categorical")?;
    let idx_cls = pd.getattr("Index")?;
    let series_cls = pd.getattr("Series")?;

    if obj.is_instance(&cat_cls)? {
        let categories = obj.getattr("categories")?;
        let codes = obj.getattr("codes")?;
        let ordered: bool = obj.getattr("ordered")?.extract()?;
        let cats_inner = encode_ndarray_tagged(
            &categories.call_method1("to_numpy", ())?,
            &format!("{key_path}.categories"),
            ctx,
        )?;
        let codes_inner = encode_ndarray_tagged(&codes, &format!("{key_path}.codes"), ctx)?;
        let mut env = serde_json::Map::new();
        env.insert(
            SCX_TYPE_KEY.to_string(),
            serde_json::Value::String("categorical".to_string()),
        );
        env.insert("categories".to_string(), cats_inner);
        env.insert("codes".to_string(), codes_inner);
        env.insert("ordered".to_string(), serde_json::Value::Bool(ordered));
        return Ok(Some(serde_json::Value::Object(env)));
    }

    if obj.is_instance(&idx_cls)? {
        let name = obj.getattr("name")?;
        let values = obj.call_method1("to_numpy", ())?;
        let inner = encode_ndarray_tagged(&values, key_path, ctx)?;
        let mut env = serde_json::Map::new();
        env.insert(
            SCX_TYPE_KEY.to_string(),
            serde_json::Value::String("pandas.Index".to_string()),
        );
        env.insert("name".to_string(), pyobj_to_simple_json(&name, key_path)?);
        env.insert("data".to_string(), inner);
        return Ok(Some(serde_json::Value::Object(env)));
    }

    if obj.is_instance(&series_cls)? {
        let name = obj.getattr("name")?;
        let values = obj.call_method1("to_numpy", ())?;
        let inner = encode_ndarray_tagged(&values, key_path, ctx)?;
        let mut env = serde_json::Map::new();
        env.insert(
            SCX_TYPE_KEY.to_string(),
            serde_json::Value::String("pandas.Series".to_string()),
        );
        env.insert("name".to_string(), pyobj_to_simple_json(&name, key_path)?);
        env.insert("data".to_string(), inner);
        return Ok(Some(serde_json::Value::Object(env)));
    }

    Ok(None)
}

/// Encode a small, scalar-like Python value (string, int, float, bool, None)
/// to JSON. Used for the `name` field of `pd.Index` / `pd.Series` envelopes,
/// which is conventionally a hashable scalar.
pub(crate) fn pyobj_to_simple_json<'py>(
    obj: &Bound<'py, PyAny>,
    key_path: &str,
) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(serde_json::Value::String(s.extract()?));
    }
    if obj.cast::<PyBool>().is_ok() {
        return Ok(serde_json::Value::Bool(obj.extract()?));
    }
    if obj.cast::<PyInt>().is_ok() {
        if let Ok(i) = obj.extract::<i64>() {
            return Ok(serde_json::Value::Number(i.into()));
        }
        if let Ok(u) = obj.extract::<u64>() {
            return Ok(serde_json::Value::Number(u.into()));
        }
    }
    if obj.cast::<PyFloat>().is_ok() {
        let f: f64 = obj.extract()?;
        if f.is_finite() {
            if let Some(n) = serde_json::Number::from_f64(f) {
                return Ok(serde_json::Value::Number(n));
            }
        }
    }
    // Fallback: stringify.
    let s: String = obj.str()?.extract()?;
    Err(PyValueError::new_err(format!(
        "uns at {key_path}: unsupported scalar name value {s}"
    )))
}

// ---------------------------------------------------------------------------
// uns deserialization (tagged-JSON envelopes → Python types)
// ---------------------------------------------------------------------------

/// Mutable context for the uns reader. Lazy-imports pandas so files that
/// only contain plain JSON never pay for the import.
pub(crate) struct UnsReadCtx<'py> {
    py: Python<'py>,
    np: Bound<'py, PyModule>,
    pd_lazy: Option<Bound<'py, PyModule>>,
}

impl<'py> UnsReadCtx<'py> {
    fn new(py: Python<'py>) -> PyResult<Self> {
        Ok(Self {
            py,
            np: py.import("numpy")?,
            pd_lazy: None,
        })
    }

    fn pandas(&mut self) -> PyResult<&Bound<'py, PyModule>> {
        if self.pd_lazy.is_none() {
            self.pd_lazy = Some(self.py.import("pandas")?);
        }
        Ok(self.pd_lazy.as_ref().unwrap())
    }
}

/// Single-pass `serde_json::Value` → Python conversion. Plain JSON
/// (`null` / `bool` / number / string / array / object) maps to the
/// corresponding Python type; objects with a string `__scx_type__` key are
/// inspected for envelope shape and reconstructed into the original
/// NumPy / pandas type if their required keys are all present. Otherwise
/// the object is built as a `dict` and recurses over its values. Replaces
/// a previous `serde_json::to_string` → `json.loads` → tree-walk pipeline
/// that paid for two intermediate traversals.
pub(crate) fn json_to_py<'py>(
    val: &serde_json::Value,
    ctx: &mut UnsReadCtx<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    match val {
        serde_json::Value::Null => Ok(py.None().into_bound(py)),
        serde_json::Value::Bool(b) => Ok(b.into_pyobject(py)?.to_owned().into_any()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any())
            } else if let Some(u) = n.as_u64() {
                Ok(u.into_pyobject(py)?.into_any())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any())
            } else {
                Err(PyValueError::new_err(format!(
                    "uns: serde_json::Number out of range: {n}"
                )))
            }
        }
        serde_json::Value::String(s) => Ok(PyString::new(py, s).into_any()),
        serde_json::Value::Array(items) => {
            let mut decoded: Vec<Bound<'py, PyAny>> = Vec::with_capacity(items.len());
            for v in items {
                decoded.push(json_to_py(v, ctx)?);
            }
            Ok(PyList::new(py, decoded)?.into_any())
        }
        serde_json::Value::Object(map) => json_object_to_py(map, ctx),
    }
}

/// Object dispatch for `json_to_py`: detect known envelopes by `__scx_type__`
/// plus required-key presence, otherwise build a plain dict and recurse.
pub(crate) fn json_object_to_py<'py>(
    map: &serde_json::Map<String, serde_json::Value>,
    ctx: &mut UnsReadCtx<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(serde_json::Value::String(tag)) = map.get(SCX_TYPE_KEY) {
        match envelope_required_keys(tag) {
            Some(keys) if json_map_has_all_keys(map, keys) => {
                return decode_tagged_envelope(map, tag, ctx);
            }
            None => {
                // Unknown tag — preserve the forward-compat warning so users
                // notice files written by a newer pyscx. The dict still
                // comes back verbatim for introspection.
                let py = ctx.py;
                let warnings = py.import("warnings")?;
                let msg = format!(
                    "uns: unknown __scx_type__ tag '{tag}' — returning the raw tagged dict; \
                     upgrade pyscx if you wrote this file with a newer version"
                );
                warnings.call_method1("warn", (msg,))?;
                return build_plain_dict(map, ctx);
            }
            Some(_) => {
                // Known tag but missing required keys — treat as a plain dict
                // that happens to use our sentinel key. Silent (the user did
                // nothing wrong: `__scx_type__` is just a string in their
                // metadata).
            }
        }
    }
    build_plain_dict(map, ctx)
}

pub(crate) fn build_plain_dict<'py>(
    map: &serde_json::Map<String, serde_json::Value>,
    ctx: &mut UnsReadCtx<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let out = PyDict::new(ctx.py);
    for (k, v) in map.iter() {
        let decoded = json_to_py(v, ctx)?;
        out.set_item(k, decoded)?;
    }
    Ok(out.into_any())
}

/// Required structural keys for each known envelope tag. Used to distinguish
/// "real envelope" from "user dict that happens to contain `__scx_type__`".
/// Returns `None` for unrecognised tags (forward-compat / warning path).
pub(crate) fn envelope_required_keys(tag: &str) -> Option<&'static [&'static str]> {
    match tag {
        "ndarray" => Some(&["dtype", "shape", "encoding", "data"]),
        "scalar" => Some(&["dtype", "data"]),
        "tuple" => Some(&["data"]),
        "recarray" => Some(&["descr", "shape", "data"]),
        "categorical" => Some(&["categories", "codes", "ordered"]),
        "pandas.Index" => Some(&["data", "name"]),
        "pandas.Series" => Some(&["data", "name"]),
        _ => None,
    }
}

pub(crate) fn json_map_has_all_keys(
    map: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> bool {
    keys.iter().all(|k| map.contains_key(*k))
}

pub(crate) fn decode_tagged_envelope<'py>(
    map: &serde_json::Map<String, serde_json::Value>,
    tag: &str,
    ctx: &mut UnsReadCtx<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    // `json_object_to_py` only dispatches here when the tag is known *and* all
    // required keys are present; the catch-all is a guard for the case where
    // a new tag is added to `envelope_required_keys` without a decoder.
    match tag {
        "ndarray" => decode_ndarray_envelope(map, ctx),
        "scalar" => decode_scalar_envelope(map, ctx),
        "tuple" => decode_tuple_envelope(map, ctx),
        "recarray" => decode_recarray_envelope(map, ctx),
        "categorical" => decode_categorical_envelope(map, ctx),
        "pandas.Index" => decode_pandas_index_envelope(map, ctx),
        "pandas.Series" => decode_pandas_series_envelope(map, ctx),
        other => Err(PyValueError::new_err(format!(
            "uns: envelope tag '{other}' has required keys registered but no decoder"
        ))),
    }
}

pub(crate) fn require_str_json<'a>(
    map: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> PyResult<&'a str> {
    match map.get(key) {
        Some(serde_json::Value::String(s)) => Ok(s.as_str()),
        Some(_) => Err(PyValueError::new_err(format!(
            "uns envelope '{key}' is not a string"
        ))),
        None => Err(PyValueError::new_err(format!(
            "uns envelope missing key '{key}'"
        ))),
    }
}

pub(crate) fn require_value_json<'a>(
    map: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> PyResult<&'a serde_json::Value> {
    map.get(key)
        .ok_or_else(|| PyValueError::new_err(format!("uns envelope missing key '{key}'")))
}

pub(crate) fn decode_base64_bytes_json<'py>(
    py: Python<'py>,
    map: &serde_json::Map<String, serde_json::Value>,
) -> PyResult<Bound<'py, PyBytes>> {
    use base64::Engine;
    let b64 = require_str_json(map, "data")?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.as_bytes())
        .map_err(|e| PyValueError::new_err(format!("uns: base64 decode failed: {e}")))?;
    Ok(PyBytes::new(py, &bytes))
}

pub(crate) fn extract_shape_json(
    map: &serde_json::Map<String, serde_json::Value>,
) -> PyResult<Vec<usize>> {
    let shape_v = require_value_json(map, "shape")?;
    let arr = match shape_v {
        serde_json::Value::Array(a) => a,
        _ => return Err(PyValueError::new_err("uns: envelope 'shape' is not a list")),
    };
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        let n = v
            .as_u64()
            .ok_or_else(|| PyValueError::new_err("uns: envelope 'shape' element is not a uint"))?;
        out.push(n as usize);
    }
    Ok(out)
}

pub(crate) fn decode_ndarray_envelope<'py>(
    map: &serde_json::Map<String, serde_json::Value>,
    ctx: &mut UnsReadCtx<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let dtype: String = require_str_json(map, "dtype")?.to_owned();
    let shape = extract_shape_json(map)?;
    let encoding: String = require_str_json(map, "encoding")?.to_owned();
    let py = ctx.py;
    let arr: Bound<'py, PyAny> = match encoding.as_str() {
        "base64le" => {
            let pybytes = decode_base64_bytes_json(py, map)?;
            let np = &ctx.np;
            let dtype_arg = np.call_method1("dtype", (dtype.as_str(),))?;
            np.call_method1("frombuffer", (pybytes, dtype_arg))?
        }
        "json" => {
            // Object or fixed-width string array. `data` is a JSON list of
            // strings; convert it to a Python list first so np.array sees
            // the per-element Python strings rather than serde values.
            // `json_to_py` borrows `ctx` mutably, so we do that conversion
            // before reborrowing `ctx.np` for the dtype/array calls.
            let data_val = require_value_json(map, "data")?;
            let data = json_to_py(data_val, ctx)?;
            let np = &ctx.np;
            // Accept both legacy "object" sentinel and dtype.str-form "|O".
            // Other string dtypes (e.g. "<U10", "|S5") go through unchanged.
            let np_dtype = if dtype == "object" {
                np.call_method1("dtype", ("object",))?
            } else {
                np.call_method1("dtype", (dtype.as_str(),))?
            };
            np.call_method1("array", (data, np_dtype))?
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "uns ndarray envelope: unknown encoding '{other}'"
            )))
        }
    };
    // Reshape and copy so the returned array owns its buffer and is writable
    // (np.frombuffer hands back a read-only view over the PyBytes).
    let shape_tup = pyo3::types::PyTuple::new(py, shape.iter().map(|s| *s as i64))?;
    let reshaped = arr.call_method1("reshape", (shape_tup,))?;
    reshaped.call_method0("copy")
}

pub(crate) fn decode_scalar_envelope<'py>(
    map: &serde_json::Map<String, serde_json::Value>,
    ctx: &mut UnsReadCtx<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let dtype = require_str_json(map, "dtype")?;
    let pybytes = decode_base64_bytes_json(ctx.py, map)?;
    let dtype_arg = ctx.np.call_method1("dtype", (dtype,))?;
    let arr = ctx.np.call_method1("frombuffer", (pybytes, dtype_arg))?;
    // Index 0 returns a NumPy scalar (np.generic).
    let zero: i64 = 0;
    arr.call_method1("__getitem__", (zero,))
}

pub(crate) fn decode_tuple_envelope<'py>(
    map: &serde_json::Map<String, serde_json::Value>,
    ctx: &mut UnsReadCtx<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let data_v = require_value_json(map, "data")?;
    let arr = match data_v {
        serde_json::Value::Array(a) => a,
        _ => {
            return Err(PyValueError::new_err(
                "uns tuple envelope: 'data' is not a list",
            ))
        }
    };
    let mut decoded: Vec<Bound<'py, PyAny>> = Vec::with_capacity(arr.len());
    for item in arr {
        decoded.push(json_to_py(item, ctx)?);
    }
    Ok(pyo3::types::PyTuple::new(ctx.py, decoded)?.into_any())
}

pub(crate) fn decode_recarray_envelope<'py>(
    map: &serde_json::Map<String, serde_json::Value>,
    ctx: &mut UnsReadCtx<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let descr = require_value_json(map, "descr")?;
    let shape = extract_shape_json(map)?;
    let pybytes = decode_base64_bytes_json(ctx.py, map)?;
    let np = &ctx.np;
    let dtype = build_structured_dtype_from_json(np, descr)?;
    let arr = np.call_method1("frombuffer", (pybytes, dtype))?;
    let shape_tup = pyo3::types::PyTuple::new(ctx.py, shape.iter().map(|s| *s as i64))?;
    let reshaped = arr.call_method1("reshape", (shape_tup,))?;
    reshaped.call_method0("copy")
}

/// Rebuild a structured `np.dtype` from a descr JSON tree (a list of
/// `[name, fmt]` or `[name, fmt, [shape...]]` entries; fmt may itself be a
/// nested descr list, in which case the sub-list's first element is also a
/// list — that's how we distinguish sub-descr from a shape tuple).
pub(crate) fn build_structured_dtype_from_json<'py>(
    np: &Bound<'py, PyModule>,
    descr: &serde_json::Value,
) -> PyResult<Bound<'py, PyAny>> {
    let entries = match descr {
        serde_json::Value::Array(a) => a,
        _ => return Err(PyValueError::new_err("uns recarray: descr is not a list")),
    };
    let py = np.py();
    let mut py_entries: Vec<Bound<'py, PyAny>> = Vec::with_capacity(entries.len());
    for entry in entries {
        let row = match entry {
            serde_json::Value::Array(r) => r,
            _ => {
                return Err(PyValueError::new_err(
                    "uns recarray: descr entry is not a list",
                ))
            }
        };
        let mut tup_items: Vec<Bound<'py, PyAny>> = Vec::with_capacity(row.len());
        for el in row {
            match el {
                serde_json::Value::String(s) => {
                    tup_items.push(PyString::new(py, s).into_any());
                }
                serde_json::Value::Array(inner) => {
                    // Sub-descr (list of lists) vs shape tuple (list of ints).
                    // Empty list falls into the shape branch and produces an
                    // empty shape tuple — matches the pre-refactor behavior.
                    let first_is_list = matches!(inner.first(), Some(serde_json::Value::Array(_)));
                    if first_is_list {
                        tup_items.push(build_structured_dtype_from_json(np, el)?);
                    } else {
                        let mut shape_items: Vec<i64> = Vec::with_capacity(inner.len());
                        for d in inner {
                            let n = d.as_i64().ok_or_else(|| {
                                PyValueError::new_err(
                                    "uns recarray: descr shape element is not an int",
                                )
                            })?;
                            shape_items.push(n);
                        }
                        let tup = pyo3::types::PyTuple::new(py, shape_items)?;
                        tup_items.push(tup.into_any());
                    }
                }
                _ => {
                    return Err(PyValueError::new_err(
                        "uns recarray: unsupported descr element type",
                    ))
                }
            }
        }
        let tup = pyo3::types::PyTuple::new(py, tup_items)?;
        py_entries.push(tup.into_any());
    }
    let descr_list = PyList::new(py, py_entries)?;
    np.call_method1("dtype", (descr_list,))
}

pub(crate) fn decode_categorical_envelope<'py>(
    map: &serde_json::Map<String, serde_json::Value>,
    ctx: &mut UnsReadCtx<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let categories_v = require_value_json(map, "categories")?;
    let codes_v = require_value_json(map, "codes")?;
    let ordered_v = require_value_json(map, "ordered")?;
    let categories = json_to_py(categories_v, ctx)?;
    let codes = json_to_py(codes_v, ctx)?;
    let ordered: bool = ordered_v
        .as_bool()
        .ok_or_else(|| PyValueError::new_err("uns categorical envelope: 'ordered' is not bool"))?;
    let pd = ctx.pandas()?.clone();
    let cat_cls = pd.getattr("Categorical")?;
    let kwargs = PyDict::new(ctx.py);
    kwargs.set_item("categories", categories)?;
    kwargs.set_item("ordered", ordered)?;
    cat_cls.call_method("from_codes", (codes,), Some(&kwargs))
}

pub(crate) fn decode_pandas_index_envelope<'py>(
    map: &serde_json::Map<String, serde_json::Value>,
    ctx: &mut UnsReadCtx<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let data_v = require_value_json(map, "data")?;
    let name_v = require_value_json(map, "name")?;
    let data = json_to_py(data_v, ctx)?;
    let name = json_to_py(name_v, ctx)?;
    let pd = ctx.pandas()?.clone();
    let idx_cls = pd.getattr("Index")?;
    let kwargs = PyDict::new(ctx.py);
    kwargs.set_item("name", name)?;
    idx_cls.call((data,), Some(&kwargs))
}

pub(crate) fn decode_pandas_series_envelope<'py>(
    map: &serde_json::Map<String, serde_json::Value>,
    ctx: &mut UnsReadCtx<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let data_v = require_value_json(map, "data")?;
    let name_v = require_value_json(map, "name")?;
    let data = json_to_py(data_v, ctx)?;
    let name = json_to_py(name_v, ctx)?;
    let pd = ctx.pandas()?.clone();
    let series_cls = pd.getattr("Series")?;
    let kwargs = PyDict::new(ctx.py);
    kwargs.set_item("name", name)?;
    series_cls.call((data,), Some(&kwargs))
}

/// Read the `uns` section from an SCX file and reconstruct any tagged
/// envelopes into Python types in a single recursive pass over the
/// `serde_json::Value` tree. Returns `None` if the file has no uns.
/// Shared by all three to_anndata entry points so the reconstruction is
/// applied consistently.
pub(crate) fn read_uns_as_pyobject<'py>(
    py: Python<'py>,
    reader: &ScxReader,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let json_val = match reader.read_uns() {
        Ok(v) => v,
        Err(scx_format::ScxError::SectionNotFound(_)) => return Ok(None),
        Err(e) => return Err(to_pyerr(e)),
    };
    let mut ctx = UnsReadCtx::new(py)?;
    Ok(Some(json_to_py(&json_val, &mut ctx)?))
}

// ---------------------------------------------------------------------------
// 1D: Parallel shard encoding helpers
// ---------------------------------------------------------------------------
