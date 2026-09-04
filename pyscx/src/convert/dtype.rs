// Codec / value-encoding / dtype conversion helpers.
//
// Extracted from the former pyscx/src/anndata.rs (T5.7).

use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatchOptions;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashSet;
use std::io::Cursor;
use std::sync::Arc;

use scx_codec::{CodecId, ValueEncoding};

/// Encode f32 values to raw LE bytes according to a value encoding.
///
/// Delegates to the canonical [`scx_codec::value_encoding::values_to_raw_bytes`]
/// so all three historical call-site copies (pyscx/anndata, scx-cli/dtype,
/// scx-mtx/convert) share one implementation. Returns
/// `Err(CodecError)` if any value falls outside the range representable
/// by the chosen integer encoding.
pub(crate) fn encode_values(
    data: &[f32],
    encoding: ValueEncoding,
) -> Result<Vec<u8>, scx_codec::CodecError> {
    scx_codec::value_encoding::values_to_raw_bytes(data, encoding)
}

// Detection lives in `scx_codec::value_encoding` — re-exported here under
// the historical name so call sites elsewhere in `pyscx/` don't have to
// change.
pub(crate) use scx_codec::value_encoding::detect_value_encoding;

/// Resolve a `codec=` string for writer paths that do **not** support the
/// row-group-framed / adaptive profiles (`from_h5mu`, `merge`, `append`,
/// `from_mtx`). Routes through the shared [`scx_format_io::resolve_codec`] so
/// the intent-axis vocabulary gets consistent errors — `auto_v2` reports as
/// removed, and the framing-only profiles (`compact` / `compact-trial`) report
/// a clear "use `scx convert` / `from_anndata`" message instead of a bare
/// "Unknown codec". `auto` and `fast` resolve to the heuristic single-encode
/// (`None`) here (the adaptive `auto` write is deferred on these paths); every
/// explicit codec — including `shufdelta` — passes through unchanged.
pub(crate) fn parse_codec_nonframed(codec: Option<&str>) -> PyResult<Option<CodecId>> {
    use scx_format_io::DecodeTarget;
    let resolved = scx_format_io::resolve_codec(codec).map_err(PyRuntimeError::new_err)?;
    if resolved.codec_trial || resolved.decode_target == Some(DecodeTarget::Storage) {
        return Err(PyRuntimeError::new_err(format!(
            "codec='{}' needs row-group-framed output and is not supported on this path; \
             use `scx convert` / `pyscx.from_anndata` for the framed/adaptive profiles, \
             or pass 'auto'/'fast'/an explicit codec here.",
            resolved.profile
        )));
    }
    Ok(resolved.explicit_codec)
}

/// Codec-selection `params_json` for the in-memory `from_anndata` provenance
/// stamp. Records the resolved profile intent (`auto`/`fast`/`compact`/
/// `compact-trial`/explicit codec name); the realized per-shard codecs are
/// reported read-side by `scx info`. Kept local to pyscx (not the hdf5-gated
/// `scx_convert` helper) so the in-memory path compiles without the `hdf5`
/// feature.
pub(crate) fn codec_selection_params_json(profile: &str) -> String {
    format!("{{\"codec_selection\":{{\"profile\":\"{profile}\"}}}}")
}

/// Convert a pandas DataFrame to an Arrow RecordBatch via pyarrow IPC.
///
/// A **0-row** frame takes a different route through pyarrow: `write_table`
/// emits zero IPC batches for a 0-row table (there is no batch to write), and
/// rebuilding an empty batch from the schema alone on the Rust side would
/// lose every categorical's declared dictionary — dictionary values live in
/// the array, not the schema. `RecordBatch.from_pandas` + `write_batch` emits
/// one real 0-row batch that carries the dictionaries and the `ordered` flag,
/// so a 0-row obs keeps `pd.Categorical([], categories=[...])`'s categories
/// exactly as a populated one does. The result is then run through
/// [`coerce_null_fields_for_empty_batch`]. Frames with rows are untouched.
pub(crate) fn pandas_to_record_batch(
    py: Python<'_>,
    df: &Bound<'_, PyAny>,
) -> PyResult<RecordBatch> {
    let pa = crate::pyimport::import_module(py, "pyarrow")?;
    let table_cls = pa.getattr("Table")?;
    let table = table_cls.call_method1("from_pandas", (df,))?;
    let n_rows: usize = table.getattr("num_rows")?.extract()?;

    // Serialize to IPC bytes
    let sink_cls = pa.getattr("BufferOutputStream")?;
    let sink = sink_cls.call0()?;
    let ipc = pa.getattr("ipc")?;
    if n_rows == 0 {
        let rb = pa
            .getattr("RecordBatch")?
            .call_method1("from_pandas", (df,))?;
        let schema = rb.getattr("schema")?;
        let writer = ipc.call_method1("new_file", (&sink, &schema))?;
        writer.call_method1("write_batch", (&rb,))?;
        writer.call_method0("close")?;
    } else {
        let schema = table.getattr("schema")?;
        let writer = ipc.call_method1("new_file", (&sink, &schema))?;
        writer.call_method1("write_table", (&table,))?;
        writer.call_method0("close")?;
    }
    let buf = sink.call_method0("getvalue")?;
    let py_bytes = buf.call_method0("to_pybytes")?;
    let bytes: &[u8] = py_bytes.extract()?;

    // Decode in Rust. Downcast `LargeUtf8 → Utf8` so the rest of the
    // Rust pipeline (and SCX writer) sees canonical narrow types
    // regardless of what pyarrow chose on its side.
    let cursor = Cursor::new(bytes.to_vec());
    let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let batch = reader
        .into_iter()
        .next()
        .ok_or_else(|| PyRuntimeError::new_err("Arrow IPC contains no batches"))?
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let batch = scx_format_io::downcast_large_types(&batch)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let batch = if batch.num_rows() == 0 {
        coerce_null_fields_for_empty_batch(&batch)?
    } else {
        batch
    };

    // `pyarrow.Table.from_pandas` drops the pandas `ordered` flag of categorical
    // columns. Re-stamp the `scx.categorical.ordered` field metadata the read
    // path (`interop::ordered_categorical_columns`) looks for, mirroring the
    // h5ad ingest path (`scx-convert::h5ad::read::read_categorical_group`).
    let ordered = ordered_categorical_columns_from_df(df)?;
    if ordered.is_empty() {
        return Ok(batch);
    }
    stamp_ordered_categorical_metadata(&batch, &ordered)
}

/// Store the columns pyarrow could not type in a 0-row frame as string.
///
/// With no values to look at, pyarrow infers Arrow `Null` for an empty
/// `object` column **and for the index** (`__index_level_0__`) of an empty
/// frame, and `Dictionary(_, Null)` for a categorical that anndata's row
/// subset pruned to zero categories. No populated frame ever produces those
/// types (an `object` string column is `string`), so leaving them would give
/// a 0-row file a schema its populated sibling never matches — `append`'s
/// obs-schema check, `merge`'s column unification and a forced `index_obs=`
/// (whose dtype gate rejects `Null`) all compare against it. String is what
/// pyarrow would have inferred from one row, and it is what `read_obs()`
/// returns for the column either way (`object`).
///
/// Zero rows only: a populated all-null column is genuinely typeless and
/// keeps round-tripping as `Null`. Field metadata (the `scx.categorical.*`
/// stamps) and the schema's `pandas` envelope are preserved.
fn coerce_null_fields_for_empty_batch(batch: &RecordBatch) -> PyResult<RecordBatch> {
    debug_assert_eq!(batch.num_rows(), 0);
    fn coerced(dt: &DataType) -> Option<DataType> {
        match dt {
            DataType::Null => Some(DataType::Utf8),
            DataType::Dictionary(key, value) if **value == DataType::Null => {
                Some(DataType::Dictionary(key.clone(), Box::new(DataType::Utf8)))
            }
            _ => None,
        }
    }
    let schema = batch.schema();
    if !schema
        .fields()
        .iter()
        .any(|f| coerced(f.data_type()).is_some())
    {
        return Ok(batch.clone());
    }
    let mut fields = Vec::with_capacity(schema.fields().len());
    let mut columns = Vec::with_capacity(schema.fields().len());
    for (field, column) in schema.fields().iter().zip(batch.columns()) {
        match coerced(field.data_type()) {
            Some(dt) => {
                columns.push(arrow::array::new_empty_array(&dt));
                fields.push(Arc::new(field.as_ref().clone().with_data_type(dt)));
            }
            None => {
                columns.push(column.clone());
                fields.push(field.clone());
            }
        }
    }
    let new_schema = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
    RecordBatch::try_new_with_options(
        new_schema,
        columns,
        &RecordBatchOptions::new().with_row_count(Some(0)),
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Names of columns in a pandas DataFrame that are ordered categoricals
/// (`dtype == "category"` and `series.cat.ordered`). Used to re-attach the
/// ordered flag that `pyarrow.Table.from_pandas` discards.
fn ordered_categorical_columns_from_df(df: &Bound<'_, PyAny>) -> PyResult<HashSet<String>> {
    let mut out = HashSet::new();
    let columns = df.getattr("columns")?;
    for col in columns.try_iter()? {
        let col = col?;
        let Ok(name) = col.extract::<String>() else {
            continue; // non-string column label — cannot be an obs/var field name
        };
        let series = df.get_item(&col)?;
        let dtype_name: String = crate::pyimport::dtype_name_of(&series)?;
        if dtype_name != "category" {
            continue;
        }
        let ordered: bool = series.getattr("cat")?.getattr("ordered")?.extract()?;
        if ordered {
            out.insert(name);
        }
    }
    Ok(out)
}

/// Return a copy of `batch` with `scx.categorical.ordered = "true"` stamped on
/// the field metadata of every column named in `ordered`.
fn stamp_ordered_categorical_metadata(
    batch: &RecordBatch,
    ordered: &HashSet<String>,
) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let new_fields: Vec<Arc<Field>> = schema
        .fields()
        .iter()
        .map(|f| {
            if ordered.contains(f.name()) {
                let mut md = f.metadata().clone();
                md.insert(
                    scx_format_io::CATEGORICAL_ORDERED_KEY.to_string(),
                    "true".to_string(),
                );
                Arc::new(f.as_ref().clone().with_metadata(md))
            } else {
                f.clone()
            }
        })
        .collect();
    let new_schema = Arc::new(Schema::new_with_metadata(
        new_fields,
        schema.metadata().clone(),
    ));
    RecordBatch::try_new(new_schema, batch.columns().to_vec())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Build an Arrow RecordBatch from an obsm/varm value, skipping the
/// `pd.DataFrame(arr) → pyarrow` round-trip when the input is a plain
/// 2D numpy ndarray of a supported numeric dtype (`float32`, `float64`,
/// `int32`, `int64`). Falls back to [`pandas_to_record_batch`] for
/// pandas DataFrames, structured arrays, or other dtypes.
///
/// The fast path emits one Arrow column per ndarray column with the
/// column name `"0".."N-1"`, matching the shape that
/// `scx-convert::pipeline::read_dense_mapping_shard` emits on the
/// streaming ingest side.
pub(crate) fn numpy_or_pandas_to_record_batch(
    py: Python<'_>,
    arr: &Bound<'_, PyAny>,
) -> PyResult<RecordBatch> {
    use arrow::array::{Float32Array, Float64Array, Int32Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use pyo3::types::PyBytes;

    let np = crate::pyimport::import_module(py, "numpy")?;
    let np_ndarray = np.getattr("ndarray")?;
    let is_ndarray: bool = arr.is_instance(&np_ndarray)?;
    if !is_ndarray {
        let pd = crate::pyimport::import_module(py, "pandas")?;
        let df = pd.call_method1("DataFrame", (arr,))?;
        return pandas_to_record_batch(py, &df);
    }
    let ndim: usize = arr.getattr("ndim")?.extract()?;
    if ndim != 2 {
        let pd = crate::pyimport::import_module(py, "pandas")?;
        let df = pd.call_method1("DataFrame", (arr,))?;
        return pandas_to_record_batch(py, &df);
    }
    let dtype = arr.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;

    let shape: (usize, usize) = arr.getattr("shape")?.extract()?;
    let (n_rows, n_cols) = shape;
    let total = n_rows.saturating_mul(n_cols);

    // Numeric dtypes route to the fast path; everything else falls
    // through to the pandas-backed builder so dtype handling stays in
    // one place.
    let target: &str = match kind.as_str() {
        "f" => {
            let dtype_str: String = dtype.getattr("str")?.extract()?;
            if dtype_str.ends_with("f4") {
                "float32"
            } else {
                "float64"
            }
        }
        "i" => {
            let dtype_str: String = dtype.getattr("str")?.extract()?;
            if dtype_str.ends_with("i4") {
                "int32"
            } else {
                "int64"
            }
        }
        _ => {
            let pd = crate::pyimport::import_module(py, "pandas")?;
            let df = pd.call_method1("DataFrame", (arr,))?;
            return pandas_to_record_batch(py, &df);
        }
    };

    // Transpose then `ascontiguousarray` so the flat buffer is
    // column-major relative to the input shape — column `c` lives in a
    // contiguous range `[c * n_rows, (c+1) * n_rows)`. Compared to the
    // legacy `.reshape(-1).tolist()` + strided indexing, this skips
    // both the per-element Python scalar allocation and the per-column
    // strided Rust loop. The `astype(..., copy=False)` enforces native
    // byte order (no copy when the dtype already matches).
    let arr_t = arr.getattr("T")?;
    let arr_cast = arr_t.call_method(
        "astype",
        (target,),
        Some(&{
            let kw = pyo3::types::PyDict::new(py);
            kw.set_item("copy", false)?;
            kw
        }),
    )?;
    let arr_c = np.call_method1("ascontiguousarray", (arr_cast,))?;
    let bytes_obj = arr_c.call_method0("tobytes")?;
    let bytes: &[u8] = bytes_obj.cast::<PyBytes>()?.as_bytes();

    let mut fields = Vec::with_capacity(n_cols);
    let mut columns: Vec<arrow::array::ArrayRef> = Vec::with_capacity(n_cols);

    match target {
        "float32" => {
            debug_assert_eq!(bytes.len(), total.saturating_mul(4));
            let flat: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            for c in 0..n_cols {
                let v = flat[c * n_rows..(c + 1) * n_rows].to_vec();
                fields.push(Field::new(c.to_string(), DataType::Float32, false));
                columns.push(Arc::new(Float32Array::from(v)));
            }
        }
        "float64" => {
            debug_assert_eq!(bytes.len(), total.saturating_mul(8));
            let flat: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            for c in 0..n_cols {
                let v = flat[c * n_rows..(c + 1) * n_rows].to_vec();
                fields.push(Field::new(c.to_string(), DataType::Float64, false));
                columns.push(Arc::new(Float64Array::from(v)));
            }
        }
        "int32" => {
            debug_assert_eq!(bytes.len(), total.saturating_mul(4));
            let flat: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|c| i32::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            for c in 0..n_cols {
                let v = flat[c * n_rows..(c + 1) * n_rows].to_vec();
                fields.push(Field::new(c.to_string(), DataType::Int32, false));
                columns.push(Arc::new(Int32Array::from(v)));
            }
        }
        "int64" => {
            debug_assert_eq!(bytes.len(), total.saturating_mul(8));
            let flat: Vec<i64> = bytes
                .chunks_exact(8)
                .map(|c| i64::from_ne_bytes(c.try_into().unwrap()))
                .collect();
            for c in 0..n_cols {
                let v = flat[c * n_rows..(c + 1) * n_rows].to_vec();
                fields.push(Field::new(c.to_string(), DataType::Int64, false));
                columns.push(Arc::new(Int64Array::from(v)));
            }
        }
        _ => unreachable!("target is one of f32/f64/i32/i64"),
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, columns).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Estimated in-memory footprint (bytes) of a dense obsm/varm batch.
/// Sums per-column buffer sizes from the Arrow array data; small
/// constant factor overhead (validity bitmaps, metadata) is ignored.
pub(crate) fn estimate_dense_bytes(batch: &RecordBatch) -> u64 {
    batch
        .columns()
        .iter()
        .map(|c| (c.len() as u64) * (data_type_byte_width(c.data_type()) as u64))
        .sum()
}

pub(crate) fn data_type_byte_width(dt: &arrow::datatypes::DataType) -> usize {
    use arrow::datatypes::DataType;
    match dt {
        DataType::Float32 | DataType::Int32 | DataType::UInt32 => 4,
        DataType::Float64 | DataType::Int64 | DataType::UInt64 => 8,
        DataType::Int16 | DataType::UInt16 => 2,
        DataType::Int8 | DataType::UInt8 | DataType::Boolean => 1,
        _ => 8,
    }
}

/// Estimated in-memory footprint (bytes) of a sparse COO obsp/varp
/// batch. `nnz × 12` (Int32 row + Int32 col + Float32 data); ignores
/// schema/metadata overhead.
pub(crate) fn estimate_coo_bytes(batch: &RecordBatch) -> u64 {
    (batch.num_rows() as u64) * 12
}

/// Ensure X is a CSR matrix; convert from dense or CSC if needed.
/// Extract a matrix as CSR, avoiding unnecessary copies when possible (1C.1/1C.5).
///
/// Returns `(csr, pre_validated)`:
/// - `csr`: A `scipy.sparse.csr_matrix` with sorted indices
/// - `pre_validated`: If true, the input was already CSR and the caller can
///   skip per-element validation in the shard loop (after calling
///   [`validate_csr_arrays`]).
///
/// `in_place` controls whether unsorted CSR inputs may be sorted in-place:
/// - `false` (default for write paths): use `.sorted_indices()`, which
///   returns a fresh CSR; the caller's matrix is never mutated.
/// - `true`: use `.sort_indices()`, which sorts the caller's CSR in place.
///   Avoids an allocation but mutates user input — only appropriate for
///   benchmark/conversion workflows that explicitly opt in.
///
/// Already-sorted CSR, dense, and CSC inputs are unaffected by `in_place`
/// — they take paths that either return the input unchanged or produce a
/// fresh allocation regardless.
pub(crate) fn ensure_csr<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    in_place: bool,
) -> PyResult<(Bound<'py, PyAny>, bool)> {
    let scipy_sparse = crate::pyimport::import_module(py, "scipy.sparse")?;
    let is_sparse = scipy_sparse
        .call_method1("issparse", (x,))?
        .extract::<bool>()?;

    if !is_sparse {
        // Dense → CSR (no bypass)
        let csr = scipy_sparse.call_method1("csr_matrix", (x,))?;
        return Ok((csr, false));
    }

    let format: String = x.getattr("format")?.extract()?;
    if format != "csr" {
        // CSC or other → CSR (no bypass)
        let csr = x.call_method0("tocsr")?;
        return Ok((csr, false));
    }

    // Already CSR — ensure sorted indices.
    // .sort_indices() sorts in-place (mutates caller's CSR; no allocation).
    // .sorted_indices() returns a fresh CSR with sorted indices and no
    // aliasing of the input's data/indices/indptr arrays.
    let has_sorted: bool = x.getattr("has_sorted_indices")?.extract()?;
    if has_sorted {
        return Ok((x.clone(), true));
    }
    if in_place {
        x.call_method0("sort_indices")?;
        Ok((x.clone(), true))
    } else {
        let csr = x.call_method0("sorted_indices")?;
        Ok((csr, true))
    }
}

/// Call `.astype(target_dtype)` only if the array's dtype doesn't already match.
/// Avoids Python call overhead when dtype is already correct (common for h5ad CSR).
pub(crate) fn astype_if_needed<'py>(
    arr: &Bound<'py, PyAny>,
    np: &Bound<'py, PyModule>,
    target_dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let dtype_name: String = crate::pyimport::dtype_name_of(arr)?;
    if dtype_name == target_dtype {
        Ok(arr.clone())
    } else {
        arr.call_method1("astype", (np.getattr(target_dtype)?,))
    }
}

// ---------------------------------------------------------------------------
// uns serialization
// ---------------------------------------------------------------------------

/// Encoding mode for `uns` serialization.
///
/// `Plain` (legacy) collapses NumPy arrays / pandas containers to plain JSON
/// lists, losing dtype, shape, and pandas metadata on read.
///
/// `Tagged` (default) wraps non-trivial values in a JSON envelope keyed by
/// `__scx_type__`. Numeric ndarray buffers are stored as base64-encoded
/// little-endian bytes so dtype, shape, and NaN/Inf round-trip bit-exact.
/// On-disk JSON is still valid plain JSON — the envelope adds metadata
/// alongside the data, so old readers see ugly dicts but not crashes.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) enum UnsFormat {
    Plain,
    Tagged,
}

pub(crate) fn parse_uns_format(s: &str) -> PyResult<UnsFormat> {
    match s {
        "plain" => Ok(UnsFormat::Plain),
        "tagged" => Ok(UnsFormat::Tagged),
        other => Err(PyValueError::new_err(format!(
            "invalid uns_format '{other}'; expected 'plain' or 'tagged'"
        ))),
    }
}

/// Accept `memory_budget` as either an int (raw bytes), a string
/// parsed via [`scx_convert::MemoryBudget::parse`] (`"2GiB"`,
/// `"512M"`), or `None` for "use default heuristics". Anything else
/// returns a `TypeError`.
pub(crate) fn parse_memory_budget(v: Option<&Bound<'_, PyAny>>) -> PyResult<Option<u64>> {
    let Some(obj) = v else { return Ok(None) };
    if obj.is_none() {
        return Ok(None);
    }
    if let Ok(n) = obj.extract::<u64>() {
        return Ok(Some(n));
    }
    if let Ok(s) = obj.extract::<String>() {
        let bytes = scx_convert::MemoryBudget::parse(&s).map_err(PyValueError::new_err)?;
        return Ok(Some(bytes));
    }
    Err(PyValueError::new_err(
        "memory_budget must be None, an int (bytes), or a string like '2GiB' / '512M'",
    ))
}
