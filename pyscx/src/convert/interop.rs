// Arrow / pyarrow / pandas / scipy interchange helpers.
//
// Extracted from the former pyscx/src/anndata.rs (T5.7).

use arrow::array::RecordBatch;
use numpy::PyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::sync::Arc;

use scx_format_io::ScxReader;

use crate::to_pyerr;

use super::*;

/// Convert an Arrow RecordBatch to a pyarrow Table via IPC bytes.
///
/// Upcasts `Utf8 → LargeUtf8` so the in-memory IPC buffer doesn't
/// overflow Arrow's 32-bit offset limit on multi-million-cell obs
/// (see [`scx_format_io::arrow_compat`]). pyarrow handles `LargeUtf8`
/// natively and pandas conversion via `to_pandas()` produces the same
/// `object` dtype either way.
pub(crate) fn record_batch_to_pyarrow<'py>(
    py: Python<'py>,
    batch: &RecordBatch,
) -> PyResult<Bound<'py, PyAny>> {
    let batch = scx_format_io::upcast_to_large_types(batch)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    // Defensive: stamp pandas `index_columns` metadata if the schema
    // carries a literal `__index_level_0__` / `_index` column without
    // it. anndata 0.10+ hard-rejects `_index` as a regular DataFrame
    // column on `write_h5ad`, so this prevents `_index` from leaking
    // into `df.columns` regardless of how the underlying SCX was
    // written. See `scx_format_io::ensure_pandas_index_metadata` doc.
    let batch = scx_format_io::ensure_pandas_index_metadata(&batch);
    // Serialize to Arrow IPC file format
    let mut buf = Vec::new();
    {
        let mut writer = arrow::ipc::writer::FileWriter::try_new(&mut buf, batch.schema_ref())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        writer
            .write(&batch)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        writer
            .finish()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    }

    let py_bytes = PyBytes::new(py, &buf);
    let pa = py.import("pyarrow")?;
    let ipc = pa.getattr("ipc")?;
    let reader = ipc.call_method1("open_file", (py_bytes,))?;
    let table = reader.call_method0("read_all")?;
    Ok(table)
}

/// Convert a pyarrow Table to a pandas DataFrame.
///
/// Branches on the schema's `pandas` metadata envelope shape:
///
/// - **No envelope** → plain `Table.to_pandas()`.
/// - **Full envelope** (from `pyarrow.Table.from_pandas`, preserved
///   verbatim through the SCX on-disk round-trip) → plain
///   `Table.to_pandas()`. pyarrow natively restores the index name,
///   multi-level indexes, and pandas-extension dtypes (`Int64`,
///   `boolean`, `Categorical`, …) from the envelope.
/// - **Minimal envelope** `{"index_columns": [...]}` (stamped by
///   `scx-convert/src/h5ad_read.rs::read_dataframe_group` and
///   `scx_format_io::ensure_pandas_index_metadata`) → `Table.to_pandas()`
///   KeyErrors on the missing `columns` field, so strip the `pandas`
///   key first, then `set_index(drop=True, inplace=True)` manually.
///   The `__index_level_0__` sentinel becomes `df.index.name = None`
///   to match anndata semantics for an unnamed index.
pub(crate) fn pyarrow_table_to_pandas<'py>(
    table: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = table.py();

    // pyarrow's `to_pandas()` restores `pd.Categorical` for dictionary columns
    // but ignores Arrow FIELD metadata, so it always returns the factor
    // unordered. Collect the columns flagged ordered (`scx.categorical.ordered`,
    // stamped by the h5ad reader) BEFORE conversion — `self_destruct` frees the
    // Arrow buffers as columns convert — then restore the bit on the DataFrame.
    let ordered_cols = ordered_categorical_columns(table)?;

    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("self_destruct", true)?;

    let df = match extract_scx_envelope_info(table)? {
        // No envelope, or a full pyarrow envelope: `to_pandas()` restores
        // dtypes / index natively.
        None => table.call_method("to_pandas", (), Some(&kwargs))?,
        Some(info) if !info.is_minimal => table.call_method("to_pandas", (), Some(&kwargs))?,
        // Minimal envelope: strip + manual set_index.
        Some(info) => {
            let stripped = strip_pandas_metadata(table)?;
            let df = stripped.call_method("to_pandas", (), Some(&kwargs))?;
            let set_idx_kwargs = pyo3::types::PyDict::new(py);
            set_idx_kwargs.set_item("drop", true)?;
            set_idx_kwargs.set_item("inplace", true)?;
            df.call_method(
                "set_index",
                (info.index_col.as_str(),),
                Some(&set_idx_kwargs),
            )?;
            if info.index_col == "__index_level_0__" {
                df.getattr("index")?.setattr("name", py.None())?;
            }
            df
        }
    };

    apply_categorical_ordered(&df, &ordered_cols)?;
    Ok(df)
}

/// Names of dictionary columns whose Arrow `Field` metadata flags them as
/// ordered categoricals (`scx.categorical.ordered == "true"`). Read from the
/// pyarrow Table schema; must be called before a `self_destruct` `to_pandas()`.
pub(crate) fn ordered_categorical_columns(table: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let py = table.py();
    let schema = table.getattr("schema")?;
    let names: Vec<String> = schema.getattr("names")?.extract()?;
    let key = PyBytes::new(py, scx_convert::CATEGORICAL_ORDERED_KEY.as_bytes());
    let mut out = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let field = schema.call_method1("field", (i,))?;
        let md = field.getattr("metadata")?;
        let Ok(md) = md.cast::<PyDict>() else {
            continue; // None (no metadata) or unexpected type
        };
        if let Some(val) = md.get_item(&key)? {
            // pyarrow field metadata values are `bytes`, but tolerate `str`
            // and any unexpected type (→ treat as not-ordered) so a stray
            // metadata entry can never crash the whole `to_anndata`.
            let is_ordered = val
                .extract::<Vec<u8>>()
                .map(|b| b == b"true")
                .or_else(|_| val.extract::<String>().map(|s| s == "true"))
                .unwrap_or(false);
            if is_ordered {
                out.push(name.clone());
            }
        }
    }
    Ok(out)
}

/// Apply `.cat.as_ordered()` to each named categorical column present in `df`.
/// No-op for columns that were dropped (e.g. the index) or aren't categorical.
///
/// Replaces each flagged column with its `.cat.as_ordered()` form via
/// `DataFrame.isetitem(pos, value)` — positional column replacement. This both
/// allows the ordered→unordered dtype change (an in-place `.loc[:, col]` would
/// reject it) and avoids the `df[col] = ...` path, which under pandas
/// Copy-on-Write warn mode emits a spurious chained-assignment `FutureWarning`
/// when `__setitem__` sees the low refcount of a frame held only from Rust.
pub(crate) fn apply_categorical_ordered(
    df: &Bound<'_, PyAny>,
    ordered_cols: &[String],
) -> PyResult<()> {
    if ordered_cols.is_empty() {
        return Ok(());
    }
    let columns = df.getattr("columns")?;
    for col in ordered_cols {
        if !columns.contains(col)? {
            continue;
        }
        let series = df.get_item(col)?;
        let dtype_name: String = series.getattr("dtype")?.getattr("name")?.extract()?;
        if dtype_name != "category" {
            continue;
        }
        let ordered = series.getattr("cat")?.call_method0("as_ordered")?;
        let pos: usize = columns
            .call_method1("get_loc", (col.as_str(),))?
            .extract()?;
        df.call_method1("isetitem", (pos, ordered))?;
    }
    Ok(())
}

/// Shape of the pandas-metadata envelope on a pyarrow Table's schema,
/// as it concerns `pyarrow_table_to_pandas`.
pub(crate) struct EnvelopeInfo {
    /// First string entry from `index_columns` — the column to use as
    /// the DataFrame index on the minimal-envelope branch.
    index_col: String,
    /// `true` when the envelope is the bare `{"index_columns": [...]}`
    /// shape stamped by `scx-convert/src/h5ad_read.rs` and
    /// `scx_format_io::ensure_pandas_index_metadata`. `false` when the
    /// envelope is the full pyarrow shape stamped by
    /// `pyarrow.Table.from_pandas` (carries `columns`,
    /// `column_indexes`, `pandas_version`, `creator`).
    is_minimal: bool,
}

/// Parse the schema's `pandas` metadata envelope, if any. Distinguishes
/// the minimal envelope (KeyError-on-`to_pandas`, must be stripped) from
/// the full envelope (carries dtype hints, must be preserved so
/// pandas-extension dtypes and multi-level indexes round-trip).
pub(crate) fn extract_scx_envelope_info<'py>(
    table: &Bound<'py, PyAny>,
) -> PyResult<Option<EnvelopeInfo>> {
    let schema = table.getattr("schema")?;
    let metadata = schema.getattr("metadata")?;
    if metadata.is_none() {
        return Ok(None);
    }
    // metadata behaves like a dict {bytes: bytes}.
    let raw = match metadata.get_item("pandas") {
        Ok(v) => v,
        Err(_) => match metadata.get_item(pyo3::types::PyBytes::new(table.py(), b"pandas")) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        },
    };
    let raw_bytes: &[u8] = if let Ok(b) = raw.cast::<pyo3::types::PyBytes>() {
        b.as_bytes()
    } else if let Ok(s) = raw.extract::<&str>() {
        s.as_bytes()
    } else {
        return Ok(None);
    };
    let parsed: serde_json::Value = match serde_json::from_slice(raw_bytes) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    let idx = parsed
        .get("index_columns")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.iter().find_map(|v| v.as_str().map(String::from)));
    let Some(index_col) = idx else {
        return Ok(None);
    };
    // pyarrow's full envelope (from `Table.from_pandas`) always carries
    // a `columns` array alongside `index_columns`. The minimal envelope
    // stamped by `read_dataframe_group` /
    // `ensure_pandas_index_metadata` has only `index_columns`.
    let is_minimal = !parsed.get("columns").is_some_and(|v| v.is_array());
    Ok(Some(EnvelopeInfo {
        index_col,
        is_minimal,
    }))
}

/// Drop the `pandas` key from `table.schema.metadata`, returning a
/// new pyarrow Table with the rest of the schema metadata preserved.
pub(crate) fn strip_pandas_metadata<'py>(table: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = table.py();
    let schema = table.getattr("schema")?;
    let metadata = schema.getattr("metadata")?;
    if metadata.is_none() {
        return Ok(table.clone());
    }
    let dict = pyo3::types::PyDict::new(py);
    let items = metadata.call_method0("items")?;
    let iter = items.try_iter()?;
    for item in iter {
        let item = item?;
        let key = item.get_item(0)?;
        let value = item.get_item(1)?;
        let key_bytes: &[u8] = if let Ok(b) = key.cast::<pyo3::types::PyBytes>() {
            b.as_bytes()
        } else if let Ok(s) = key.extract::<&str>() {
            s.as_bytes()
        } else {
            // Pass unknown key shapes through unchanged.
            dict.set_item(key, value)?;
            continue;
        };
        if key_bytes == b"pandas" {
            continue;
        }
        dict.set_item(key, value)?;
    }
    table.call_method1("replace_schema_metadata", (dict,))
}

/// Convert an ScxCsr to a scipy.sparse.csr_matrix via zero-copy numpy arrays.
pub(crate) fn csr_to_scipy<'py>(
    py: Python<'py>,
    csr: scx_sparse::ScxCsr,
) -> PyResult<Bound<'py, PyAny>> {
    let shape = (csr.shape.0, csr.shape.1);

    // Zero-copy: moves Vec ownership to numpy
    let indptr = PyArray1::from_vec(py, csr.indptr);
    let indices = PyArray1::from_vec(py, csr.indices);
    let data = PyArray1::from_vec(py, csr.data);

    let scipy_sparse = py.import("scipy.sparse")?;
    let args = ((data, indices, indptr),);
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("shape", shape)?;
    kwargs.set_item("copy", false)?;
    scipy_sparse.call_method("csr_matrix", args, Some(&kwargs))
}

/// Build a numpy 2-D array directly from a RecordBatch of homogeneous numeric
/// float columns (B4). Returns `None` (→ caller falls back to the pandas path)
/// when the batch is empty, heterogeneous, non-float, or contains nulls.
///
/// This avoids the RecordBatch → pyarrow Table → pandas DataFrame → `.values`
/// round-trip for the common dense-float obsm/varm case, including the f32→f64
/// upcast that `.values` can introduce (breaking the f32 zero-copy contract).
pub(crate) fn record_batch_to_numpy2d<'py>(
    py: Python<'py>,
    batch: &RecordBatch,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    use arrow::array::{Array, Float32Array, Float64Array};
    use arrow::datatypes::DataType;

    let n_rows = batch.num_rows();
    let n_cols = batch.num_columns();
    if n_cols == 0 {
        return Ok(None);
    }
    let dt = batch.schema().field(0).data_type().clone();
    let homogeneous = batch.schema().fields().iter().all(|f| f.data_type() == &dt);
    if !homogeneous {
        return Ok(None);
    }
    // Nulls would read as a default value via `.value()`; defer those to pandas.
    if batch.columns().iter().any(|c| c.null_count() > 0) {
        return Ok(None);
    }

    match dt {
        DataType::Float32 => {
            let Some(cols) = (0..n_cols)
                .map(|c| batch.column(c).as_any().downcast_ref::<Float32Array>())
                .collect::<Option<Vec<_>>>()
            else {
                return Ok(None);
            };
            let mut flat: Vec<f32> = Vec::with_capacity(n_rows * n_cols);
            for r in 0..n_rows {
                for col in &cols {
                    flat.push(col.value(r));
                }
            }
            let arr = PyArray1::from_vec(py, flat)
                .into_any()
                .call_method1("reshape", ((n_rows, n_cols),))?;
            Ok(Some(arr))
        }
        DataType::Float64 => {
            let Some(cols) = (0..n_cols)
                .map(|c| batch.column(c).as_any().downcast_ref::<Float64Array>())
                .collect::<Option<Vec<_>>>()
            else {
                return Ok(None);
            };
            let mut flat: Vec<f64> = Vec::with_capacity(n_rows * n_cols);
            for r in 0..n_rows {
                for col in &cols {
                    flat.push(col.value(r));
                }
            }
            let arr = PyArray1::from_vec(py, flat)
                .into_any()
                .call_method1("reshape", ((n_rows, n_cols),))?;
            Ok(Some(arr))
        }
        _ => Ok(None),
    }
}

/// Convert an Arrow RecordBatch (obsm/varm) to a numpy 2D array.
pub(crate) fn obsm_batch_to_numpy<'py>(
    py: Python<'py>,
    batch: &RecordBatch,
) -> PyResult<Bound<'py, PyAny>> {
    // Fast path: dense homogeneous float columns → numpy 2-D directly.
    if let Some(arr) = record_batch_to_numpy2d(py, batch)? {
        return Ok(arr);
    }
    // Fallback: heterogeneous / non-numeric → pyarrow + pandas.
    let table = record_batch_to_pyarrow(py, batch)?;
    let df = pyarrow_table_to_pandas(&table)?;
    df.getattr("values")
}

/// Read obsm embeddings, optionally restricted to a caller-supplied set of
/// keys (`to_anndata(obsm=[...])`).
///
/// - `filter == None` → load every obsm key (byte-identical to the prior
///   unconditional `read_all_obsm()` behaviour, including the
///   `SectionNotFound` → empty-map fallback).
/// - `filter == Some(keys)` → load only the listed keys. Each key is
///   validated against [`ScxReader::list_obsm`]; an unknown key raises
///   `KeyError` (an empty list loads no obsm). This is the selective path
///   that drops the per-worker RAM of unused embeddings.
pub(crate) fn read_obsm_selected(
    reader: &ScxReader,
    filter: Option<&[String]>,
) -> PyResult<std::collections::HashMap<String, RecordBatch>> {
    match filter {
        None => match reader.read_all_obsm() {
            Ok(map) => Ok(map),
            Err(scx_format_io::ScxError::SectionNotFound(_)) => {
                Ok(std::collections::HashMap::new())
            }
            Err(e) => Err(to_pyerr(e)),
        },
        Some(keys) => {
            validate_obsm_keys(reader, filter)?;
            let mut map = std::collections::HashMap::with_capacity(keys.len());
            for key in keys {
                let batch = reader.read_obsm(key).map_err(to_pyerr)?;
                map.insert(key.clone(), batch);
            }
            Ok(map)
        }
    }
}

/// Validate that every key in `filter` exists in the file's obsm catalog
/// (a catalog-only scan via [`ScxReader::list_obsm`] — reads no shard
/// bytes). `None` validates nothing. An unknown key raises `KeyError`.
/// Used by the lazy obsm path to fail fast on a bad key without
/// defeating the per-key deferral.
pub(crate) fn validate_obsm_keys(reader: &ScxReader, filter: Option<&[String]>) -> PyResult<()> {
    if let Some(keys) = filter {
        let available = reader.list_obsm();
        for key in keys {
            if !available.iter().any(|a| a == key) {
                return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                    "obsm key {key:?} not found; available obsm keys: {available:?}"
                )));
            }
        }
    }
    Ok(())
}

/// Populate `obsm_dict` with eager dense numpy arrays for the selected
/// obsm keys, applying deletion vectors and (when `obs_filter` is set)
/// the obs_filter row slice. Extracted from the backed `to_anndata`
/// path so the backed dense row-gather branch can skip it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_eager_obsm_dict(
    py: Python<'_>,
    reader: &ScxReader,
    obsm_filter: Option<&[String]>,
    obs_filter: Option<&str>,
    apply_deletion_vectors: bool,
    kept_to_global: &Option<Vec<u64>>,
    dv_kept_to_global: &Option<Vec<u64>>,
    obsm_dict: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<()> {
    let obsm_map = read_obsm_selected(reader, obsm_filter)?;
    if obs_filter.is_some() {
        // When obs_filter is present, obsm must be sliced to match kept_to_global.
        if let Some(kept) = kept_to_global {
            // Pre-compute positions once for all obsm entries (loop-invariant —
            // they depend only on kept and deletion vectors).
            // `dv_mapping` (kept global rows) is built by `compute_kept_to_global`
            // as an ascending `(0..n_obs).filter(...)` scan, so it is sorted —
            // binary_search keeps this O(K log D) instead of O(K * D).
            let positions: Vec<i64> = match dv_kept_to_global {
                Some(dv_mapping) => kept
                    .iter()
                    .filter_map(|&g| dv_mapping.binary_search(&g).ok().map(|p| p as i64))
                    .collect(),
                None => kept.iter().map(|&g| g as i64).collect(),
            };
            for (name, batch) in &obsm_map {
                let filtered = if apply_deletion_vectors {
                    filter_obs_by_deletion_vectors(reader, batch.clone())?
                } else {
                    batch.clone()
                };
                let np_arr = obsm_batch_to_numpy(py, &filtered)?;
                let idx_arr = numpy::PyArray1::from_slice(py, &positions);
                let sliced = np_arr.call_method1("__getitem__", (idx_arr,))?;
                obsm_dict.set_item(name, sliced)?;
            }
        }
    } else {
        for (name, batch) in &obsm_map {
            let filtered = if apply_deletion_vectors {
                filter_obs_by_deletion_vectors(reader, batch.clone())?
            } else {
                batch.clone()
            };
            let np_arr = obsm_batch_to_numpy(py, &filtered)?;
            obsm_dict.set_item(name, np_arr)?;
        }
    }
    Ok(())
}

/// Returns true if either axis would overflow Int32 coordinates and the
/// COO batch must therefore use the Int64 row/col encoding ("v2 layout").
/// For all current workloads — even atlas-scale sub-billion-cell files —
/// both axes fit in Int32 and this returns false, keeping coordinates
/// at 4 bytes each on disk. Only axes ≥ 2^31 trip the Int64 path.
pub(crate) fn coo_needs_int64_coords(n_rows: usize, n_cols: usize) -> bool {
    n_rows > i32::MAX as usize || n_cols > i32::MAX as usize
}

/// Borrowed view of the row/col columns of a pairwise COO RecordBatch,
/// dispatched on whichever Int32 / Int64 dtype the on-disk batch uses.
/// Lets every consumer treat both wire-format widths uniformly.
pub(crate) enum CooCoordsRef<'a> {
    Int32(&'a arrow::array::Int32Array, &'a arrow::array::Int32Array),
    Int64(&'a arrow::array::Int64Array, &'a arrow::array::Int64Array),
}

impl CooCoordsRef<'_> {
    pub(crate) fn len(&self) -> usize {
        use arrow::array::Array;
        match self {
            CooCoordsRef::Int32(r, _) => r.len(),
            CooCoordsRef::Int64(r, _) => r.len(),
        }
    }

    pub(crate) fn row_i64(&self, i: usize) -> i64 {
        match self {
            CooCoordsRef::Int32(r, _) => r.value(i) as i64,
            CooCoordsRef::Int64(r, _) => r.value(i),
        }
    }

    pub(crate) fn col_i64(&self, i: usize) -> i64 {
        match self {
            CooCoordsRef::Int32(_, c) => c.value(i) as i64,
            CooCoordsRef::Int64(_, c) => c.value(i),
        }
    }
}

/// Extract the row/col columns of a pairwise COO RecordBatch, accepting
/// either Int32 (v1 layout) or Int64 (v2 layout). Returns an error if
/// the columns are not both Int32 or both Int64. The data column is
/// validated separately by each caller.
pub(crate) fn coo_coords_from_batch(batch: &RecordBatch) -> PyResult<CooCoordsRef<'_>> {
    use arrow::array::{Int32Array, Int64Array};
    use arrow::datatypes::DataType;
    let row_dt = batch.column(0).data_type().clone();
    let col_dt = batch.column(1).data_type().clone();
    match (&row_dt, &col_dt) {
        (DataType::Int32, DataType::Int32) => {
            let r = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| {
                    PyRuntimeError::new_err("COO row column claims Int32 but downcast failed")
                })?;
            let c = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| {
                    PyRuntimeError::new_err("COO col column claims Int32 but downcast failed")
                })?;
            Ok(CooCoordsRef::Int32(r, c))
        }
        (DataType::Int64, DataType::Int64) => {
            let r = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    PyRuntimeError::new_err("COO row column claims Int64 but downcast failed")
                })?;
            let c = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    PyRuntimeError::new_err("COO col column claims Int64 but downcast failed")
                })?;
            Ok(CooCoordsRef::Int64(r, c))
        }
        _ => Err(PyValueError::new_err(format!(
            "Pairwise COO RecordBatch has mismatched or unsupported coord dtypes \
             (row={row_dt:?}, col={col_dt:?}); expected matching Int32 (v1) or Int64 (v2)."
        ))),
    }
}

/// Convert a scipy sparse matrix to a COO Arrow RecordBatch.
///
/// The resulting batch has columns `row: Int32 | Int64`, `col: Int32 | Int64`,
/// `data: Float32` (nnz rows) and schema metadata `n_rows` and `n_cols`.
/// Coordinate width is chosen by [`coo_needs_int64_coords`] so files at
/// sub-2^31 axes stay byte-identical to the v1 layout. Data is cast to
/// float32; precision is reduced if the source uses float64.
pub(crate) fn sparse_to_coo_record_batch(
    py: Python<'_>,
    mat: &Bound<'_, PyAny>,
) -> PyResult<RecordBatch> {
    use arrow::array::{Float32Array, Int32Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::collections::HashMap;

    let scipy_sparse = py.import("scipy.sparse")?;
    let coo = scipy_sparse.call_method1("coo_matrix", (mat,))?;
    let shape: (usize, usize) = coo.getattr("shape")?.extract()?;
    let np = py.import("numpy")?;

    let need_i64 = coo_needs_int64_coords(shape.0, shape.1);

    let (row_col_dtype, row_array, col_array): (
        DataType,
        Arc<dyn arrow::array::Array>,
        Arc<dyn arrow::array::Array>,
    ) = if need_i64 {
        let row: Vec<i64> = np
            .call_method1("asarray", (coo.getattr("row")?,))?
            .call_method1("astype", ("int64",))?
            .extract()?;
        let col: Vec<i64> = np
            .call_method1("asarray", (coo.getattr("col")?,))?
            .call_method1("astype", ("int64",))?
            .extract()?;
        (
            DataType::Int64,
            Arc::new(Int64Array::from(row)),
            Arc::new(Int64Array::from(col)),
        )
    } else {
        let row: Vec<i32> = np
            .call_method1("asarray", (coo.getattr("row")?,))?
            .call_method1("astype", ("int32",))?
            .extract()?;
        let col: Vec<i32> = np
            .call_method1("asarray", (coo.getattr("col")?,))?
            .call_method1("astype", ("int32",))?
            .extract()?;
        (
            DataType::Int32,
            Arc::new(Int32Array::from(row)),
            Arc::new(Int32Array::from(col)),
        )
    };
    let data: Vec<f32> = np
        .call_method1("asarray", (coo.getattr("data")?,))?
        .call_method1("astype", ("float32",))?
        .extract()?;

    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("row", row_col_dtype.clone(), false),
            Field::new("col", row_col_dtype, false),
            Field::new("data", DataType::Float32, false),
        ],
        HashMap::from([
            ("n_rows".to_string(), shape.0.to_string()),
            ("n_cols".to_string(), shape.1.to_string()),
        ]),
    ));
    RecordBatch::try_new(
        schema,
        vec![row_array, col_array, Arc::new(Float32Array::from(data))],
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Convert a COO Arrow RecordBatch back to a scipy.sparse.csr_matrix.
///
/// Reads `row`, `col`, `data` columns and `n_rows`/`n_cols` schema metadata.
/// Accepts both v1 (`Int32` row/col) and v2 (`Int64` row/col) layouts.
pub(crate) fn coo_record_batch_to_scipy<'py>(
    py: Python<'py>,
    batch: &RecordBatch,
) -> PyResult<Bound<'py, PyAny>> {
    use arrow::array::Float32Array;
    use numpy::PyArray1;

    let meta = batch.schema().metadata().clone();
    let n_rows: usize = meta
        .get("n_rows")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| PyRuntimeError::new_err("Missing n_rows in sparse matrix metadata"))?;
    let n_cols: usize = meta
        .get("n_cols")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| PyRuntimeError::new_err("Missing n_cols in sparse matrix metadata"))?;

    let coords = coo_coords_from_batch(batch)?;
    let data_arr = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| PyRuntimeError::new_err("Invalid data column in sparse matrix batch"))?;

    let (row_np, col_np): (Bound<'_, PyAny>, Bound<'_, PyAny>) = match &coords {
        CooCoordsRef::Int32(r, c) => (
            PyArray1::from_slice(py, r.values()).into_any(),
            PyArray1::from_slice(py, c.values()).into_any(),
        ),
        CooCoordsRef::Int64(r, c) => (
            PyArray1::from_slice(py, r.values()).into_any(),
            PyArray1::from_slice(py, c.values()).into_any(),
        ),
    };
    let data_np = PyArray1::from_slice(py, data_arr.values());

    let scipy_sparse = py.import("scipy.sparse")?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("shape", (n_rows, n_cols))?;
    scipy_sparse.call_method("csr_matrix", ((data_np, (row_np, col_np)),), Some(&kwargs))
}

/// Subset a COO obsp RecordBatch by a kept-row set on both axes.
///
/// `obsp` is square (obs × obs); the same kept set applies to rows and cols.
/// Returns a new batch containing only the entries whose row AND col are kept,
/// with indices remapped to the user-visible 0..kept_rows.len() range and the
/// `n_rows` / `n_cols` schema metadata updated to `kept_rows.len()`.
///
/// Width-generic: accepts both v1 (`Int32`) and v2 (`Int64`) COO inputs.
/// Output width is chosen by [`coo_needs_int64_coords`] on `kept_rows.len()`
/// so the remap shrinks the on-disk footprint when an extreme axis is filtered
/// down. `kept_rows` MUST be sorted ascending — that is the construction
/// invariant of `compose_kept_to_global` and lets the remap run as a
/// `binary_search` instead of allocating a dense `Vec<i32>` of length `n_rows`
/// (≈ 2.24 GB at 561M cells).
pub(crate) fn filter_coo_obsp_by_kept_rows(
    batch: &RecordBatch,
    kept_rows: &[u64],
) -> PyResult<RecordBatch> {
    use arrow::array::{Float32Array, Int32Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::collections::HashMap;

    let meta = batch.schema().metadata().clone();
    let n_rows: usize = meta
        .get("n_rows")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| PyRuntimeError::new_err("Missing n_rows in sparse matrix metadata"))?;
    let n_cols: usize = meta
        .get("n_cols")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| PyRuntimeError::new_err("Missing n_cols in sparse matrix metadata"))?;

    let coords = coo_coords_from_batch(batch)?;
    let data_arr = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| PyRuntimeError::new_err("Invalid data column in sparse matrix batch"))?;

    // Binary-search correctness requires sorted input. `kept_rows` is
    // produced sorted by `compose_kept_to_global` today, but enforce in
    // release too so a future unsorted caller fails loudly rather than
    // silently dropping or misrouting entries.
    if kept_rows.windows(2).any(|w| w[0] > w[1]) {
        return Err(PyValueError::new_err(
            "filter_coo_obsp_by_kept_rows requires kept_rows sorted ascending",
        ));
    }
    let kept_len = kept_rows.len();
    let output_i64 = coo_needs_int64_coords(kept_len, kept_len);

    // Binary-search remap: O(nnz log kept_len) time, zero extra memory
    // beyond the kept_rows slice (which the caller already owns).
    let remap = |orig: i64, axis_len: usize| -> Option<i64> {
        if orig < 0 {
            return None;
        }
        let orig_u = orig as u64;
        if (orig as usize) >= axis_len {
            return None;
        }
        kept_rows.binary_search(&orig_u).ok().map(|i| i as i64)
    };

    let nnz = coords.len();
    let data_vals = data_arr.values();
    let mut new_row_i64: Vec<i64> = Vec::with_capacity(nnz);
    let mut new_col_i64: Vec<i64> = Vec::with_capacity(nnz);
    let mut new_data: Vec<f32> = Vec::with_capacity(nnz);
    for k in 0..nnz {
        let r = coords.row_i64(k);
        let c = coords.col_i64(k);
        let Some(nr) = remap(r, n_rows) else { continue };
        let Some(nc) = remap(c, n_cols) else { continue };
        new_row_i64.push(nr);
        new_col_i64.push(nc);
        new_data.push(data_vals[k]);
    }

    let (row_dtype, row_array, col_array): (
        DataType,
        Arc<dyn arrow::array::Array>,
        Arc<dyn arrow::array::Array>,
    ) = if output_i64 {
        (
            DataType::Int64,
            Arc::new(Int64Array::from(new_row_i64)),
            Arc::new(Int64Array::from(new_col_i64)),
        )
    } else {
        let new_row_i32: Vec<i32> = new_row_i64.into_iter().map(|v| v as i32).collect();
        let new_col_i32: Vec<i32> = new_col_i64.into_iter().map(|v| v as i32).collect();
        (
            DataType::Int32,
            Arc::new(Int32Array::from(new_row_i32)),
            Arc::new(Int32Array::from(new_col_i32)),
        )
    };

    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("row", row_dtype.clone(), false),
            Field::new("col", row_dtype, false),
            Field::new("data", DataType::Float32, false),
        ],
        HashMap::from([
            ("n_rows".to_string(), kept_len.to_string()),
            ("n_cols".to_string(), kept_len.to_string()),
        ]),
    ));
    RecordBatch::try_new(
        schema,
        vec![row_array, col_array, Arc::new(Float32Array::from(new_data))],
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}
