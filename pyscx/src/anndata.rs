// to_anndata / from_anndata conversion

use arrow::array::RecordBatch;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use std::collections::HashSet;
use std::io::Cursor;
use std::sync::Arc;

use rayon::prelude::*;
use scx_codec::{CodecId, ValueEncoding};
use scx_format::header::MAGIC;
use scx_format::section::SectionType;
use scx_format::{
    select_codec_for_modality, FileHeader, ModalityType, PreEncodedSection, ProvenanceEntry,
    ScxReader, ScxWriter,
};
use scx_sparse::canonicalize_csr;

use crate::to_pyerr;

/// Memory budget for the convert-time streaming CSR→CSC transpose.
/// Matches scx-cli's convert pipeline. The full CSR matrix already
/// lives in RAM at this point, so this only bounds the per-chunk
/// transpose working set.
const PYSCX_CSC_MEMORY_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Default memory budget for the eager [`to_anndata`] full-assembly
/// path (Phase 4d). Estimated bytes above this threshold trigger a
/// `UserWarning` that recommends `to_anndata(backed=True)` or
/// `pyscx.open(path).query()`. Assembly still proceeds — the warning
/// is advisory.
const DEFAULT_EAGER_MEMORY_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Catalog-only estimate of the bytes required to assemble the full X
/// matrix plus obs / var metadata into an in-memory AnnData. Sums
/// `nnz × 16` for CSR shards (i32 indices + f32 data), `n_rows × 8`
/// for the assembled CSR indptr (i64), and the on-disk size of every
/// obs / var section (sharded or single). Walks `reader.catalog()`
/// only — no payload reads.
fn estimate_eager_assembly_bytes(reader: &ScxReader) -> u64 {
    let entries = &reader.catalog().entries;
    let mut nnz: u64 = 0;
    let mut x_rows: u64 = 0;
    for entry in entries {
        if entry.section_type != SectionType::CsrShard || entry.modality_id != 0 {
            continue;
        }
        if let Some(stats) = &entry.stats {
            nnz = nnz.saturating_add(stats.nnz);
            x_rows = x_rows.saturating_add(stats.row_end.saturating_sub(stats.row_start));
        }
    }
    let mut meta_bytes: u64 = 0;
    for entry in entries {
        match entry.section_type {
            SectionType::ObsMetadata
            | SectionType::VarMetadata
            | SectionType::ObsMetadataShard
            | SectionType::VarMetadataShard => {
                meta_bytes = meta_bytes.saturating_add(entry.length);
            }
            _ => {}
        }
    }
    nnz.saturating_mul(16)
        .saturating_add(x_rows.saturating_mul(8))
        .saturating_add(meta_bytes)
}

/// Phase 5b: build and (conditionally) write a detection-bitmap shard
/// for the in-memory `from_anndata` write path. Mirrors
/// `scx_convert::pipeline::build_and_write_bitmap_for_shard` but emits
/// a Python `UserWarning` instead of `ConvertWarning::BitmapSkipped`.
///
/// Only the unimodal RNA case is exercised here (multimodal MuData
/// converts go through `scx-convert::h5mu_to_scx[_streaming]`); the
/// auto policy is therefore conservative — no ATAC eagerness branch.
#[allow(clippy::too_many_arguments)]
fn build_and_write_bitmap_for_shard_python(
    py: Python<'_>,
    writer: &mut ScxWriter,
    indptr: &[u64],
    indices: &[u32],
    row_start: u64,
    n_rows: u32,
    n_vars: u32,
    encoded_csr_size: usize,
    policy: scx_format::BitmapPolicy,
) -> PyResult<()> {
    use scx_format::bitmap::BitmapShard;
    use scx_format::BitmapPolicy;
    const DENSITY_THRESHOLD: f32 = 0.30;
    const N_VARS_CAP: u32 = 1_000_000;
    const SIZE_PERCENT: usize = 15;

    let emit_warning = |msg: String| -> PyResult<()> {
        py.import("warnings")?.call_method1("warn", (msg,))?;
        Ok(())
    };

    if matches!(policy, BitmapPolicy::Off) {
        return Ok(());
    }
    if n_vars > N_VARS_CAP && !matches!(policy, BitmapPolicy::Always) {
        return emit_warning(format!(
            "bitmap skipped: n_vars {n_vars} exceeds auto cap {N_VARS_CAP}"
        ));
    }
    let nnz = *indptr.last().unwrap_or(&0);
    let cells = n_rows as u64;
    let density = if cells == 0 || n_vars == 0 {
        0.0_f32
    } else {
        nnz as f32 / (cells as f32 * n_vars as f32)
    };
    if matches!(policy, BitmapPolicy::Auto) && density > DENSITY_THRESHOLD {
        return emit_warning(format!(
            "bitmap skipped: density {density:.3} above auto threshold {DENSITY_THRESHOLD}"
        ));
    }
    let shard = BitmapShard::build_from_csr(row_start, n_rows, n_vars, indptr, indices);
    if matches!(policy, BitmapPolicy::Auto) {
        let est = shard.estimated_encoded_size();
        if encoded_csr_size > 0
            && est.saturating_mul(100) > encoded_csr_size.saturating_mul(SIZE_PERCENT)
        {
            return emit_warning(format!(
                "bitmap skipped: estimated {est} bytes > {SIZE_PERCENT}% of CSR shard ({encoded_csr_size})"
            ));
        }
    }
    writer.write_bitmap_shard(&shard).map_err(to_pyerr)?;
    Ok(())
}

/// Phase 5a: build and write obs / var predicate indexes from a Python
/// in-memory AnnData write path. Thin wrapper over
/// `scx_engine::build_and_write_conversion_predicate_indexes`; only the
/// outcome-to-Python mapping differs from the convert-side wrapper in
/// `scx_convert::pipeline::build_and_write_predicate_indexes`
/// (`PyValueError` for forced errors, `warnings.warn(...)` for preset
/// skips). Keep those two outcome maps in sync.
#[allow(clippy::too_many_arguments)]
fn build_and_write_predicate_indexes_inline(
    py: Python<'_>,
    writer: &mut ScxWriter,
    obs: &RecordBatch,
    var: &RecordBatch,
    csr_row_ranges: &[(u64, u64)],
    n_vars: usize,
    index_obs: &[String],
    index_var: &[String],
    index_preset: Option<&str>,
    index_auto_threshold: usize,
) -> PyResult<()> {
    use scx_engine::{
        build_and_write_conversion_predicate_indexes, BuildOutcome,
        ConversionPredicateIndexOptions, EngineError,
    };

    let engine_opts = ConversionPredicateIndexOptions {
        index_obs: index_obs.to_vec(),
        index_var: index_var.to_vec(),
        index_preset: index_preset.map(|s| s.to_string()),
        index_auto_threshold,
    };
    let result = build_and_write_conversion_predicate_indexes(
        writer,
        obs,
        var,
        csr_row_ranges,
        n_vars,
        &engine_opts,
    )
    .map_err(|e| match e {
        // Unknown preset is user-facing — surface as PyValueError so it
        // shows up as a clean `ValueError` in Python.
        EngineError::UnknownIndexPreset(_) => PyValueError::new_err(e.to_string()),
        other => PyRuntimeError::new_err(format!("build predicate index: {other}")),
    })?;

    let emit_warning = |msg: String| -> PyResult<()> {
        py.import("warnings")?.call_method1("warn", (msg,))?;
        Ok(())
    };
    // Drop pyarrow-internal `__*` columns
    // (notably `__index_level_0__`) before passing to the error
    // renderer — they live in the arrow schema for round-trip but
    // are never the right user-facing suggestion.
    let obs_available: Vec<String> = obs
        .schema()
        .fields()
        .iter()
        .map(|f| f.name())
        .filter(|n| !n.starts_with("__"))
        .cloned()
        .collect();
    let var_available: Vec<String> = var
        .schema()
        .fields()
        .iter()
        .map(|f| f.name())
        .filter(|n| !n.starts_with("__"))
        .cloned()
        .collect();
    let process = |outcomes: Vec<BuildOutcome>, axis: &str, available: &[String]| -> PyResult<()> {
        // Aggregate forced missing-column errors
        // into a single error so users see ALL typos in one shot,
        // matching the CLI's `process_predicate_index_outcomes` policy.
        // Non-missing forced errors (unsupported dtype, high
        // cardinality) stay fail-fast — the column exists, the message
        // is per-column.
        let mut forced_missing: Vec<String> = Vec::new();
        for outcome in outcomes {
            match outcome {
                BuildOutcome::ForcedColumnError { column, reason } => {
                    if matches!(reason, scx_engine::index::SkipReason::MissingColumn) {
                        forced_missing.push(column);
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "forced {axis} index column '{column}': {reason}"
                        )));
                    }
                }
                BuildOutcome::PresetSkipped { column, reason } => {
                    emit_warning(format!(
                        "predicate index skipped for {axis} column '{column}': {reason}"
                    ))?;
                }
            }
        }
        if !forced_missing.is_empty() {
            let msg =
                scx_engine::index::forced_columns_missing_message(axis, &forced_missing, available);
            return Err(PyValueError::new_err(msg));
        }
        Ok(())
    };
    process(result.obs_outcomes, "obs", &obs_available)?;
    process(result.var_outcomes, "var", &var_available)?;

    Ok(())
}

/// Streaming CSR → CSC transpose over the in-memory `(indptr, indices,
/// data)` arrays, writing each emitted chunk as one CSC shard.
///
/// Mirrors `scx-cli::convert::write_csc_shards_from_csr` so the two
/// import paths produce structurally identical CSC sidecars (same
/// `csc_cols_per_shard`, same encoder).
#[allow(clippy::too_many_arguments)]
fn write_csc_shards_from_csr(
    writer: &mut ScxWriter,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    value_encoding: ValueEncoding,
    codec_id: CodecId,
    csc_cols_per_shard: usize,
) -> Result<(), scx_format::ScxError> {
    let mut indptr_u64: Vec<u64> = indptr
        .iter()
        .map(|&v| {
            if v < 0 {
                Err(scx_format::ScxError::InvalidCatalog(format!(
                    "negative CSR indptr value {v} before CSC transpose"
                )))
            } else {
                Ok(v as u64)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut indices_u32: Vec<u32> = indices
        .iter()
        .map(|&v| {
            if v < 0 {
                Err(scx_format::ScxError::InvalidCatalog(format!(
                    "negative CSR index value {v} before CSC transpose"
                )))
            } else {
                Ok(v as u32)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut values = data.to_vec();
    canonicalize_csr(&mut indptr_u64, &mut indices_u32, &mut values);
    let csr = scx_sparse::ScxCsr::new_unchecked(
        (n_obs, n_vars),
        indptr_u64.iter().map(|&v| v as i64).collect(),
        indices_u32.iter().map(|&v| v as i32).collect(),
        values,
    );
    let shards = std::slice::from_ref(&csr);

    let mut iter = scx_sparse::streaming_csr_to_csc_iter_with_cap(
        shards,
        n_obs,
        n_vars,
        PYSCX_CSC_MEMORY_BYTES,
        csc_cols_per_shard,
    )
    .map_err(|e| scx_format::ScxError::Io(std::io::Error::other(format!("CSC transpose: {e}"))))?;

    loop {
        let col_start = iter.current_col_start() as u64;
        let chunk = match iter.next() {
            Some(c) => c.map_err(|e| {
                scx_format::ScxError::Io(std::io::Error::other(format!("CSC chunk: {e}")))
            })?,
            None => break,
        };
        let csc_indptr_u64: Vec<u64> = chunk.indptr.iter().map(|&v| v as u64).collect();
        let csc_indices_u32: Vec<u32> = chunk.indices.iter().map(|&i| i as u32).collect();
        let raw_values = encode_values(&chunk.data, value_encoding)?;
        writer.write_csc_shard(
            &csc_indptr_u64,
            &csc_indices_u32,
            &raw_values,
            codec_id,
            value_encoding,
            col_start,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// to_anndata: SCX → AnnData
// ---------------------------------------------------------------------------

/// Convert an Arrow RecordBatch to a pyarrow Table via IPC bytes.
///
/// Upcasts `Utf8 → LargeUtf8` so the in-memory IPC buffer doesn't
/// overflow Arrow's 32-bit offset limit on multi-million-cell obs
/// (see [`scx_format::arrow_compat`]). pyarrow handles `LargeUtf8`
/// natively and pandas conversion via `to_pandas()` produces the same
/// `object` dtype either way.
pub(crate) fn record_batch_to_pyarrow<'py>(
    py: Python<'py>,
    batch: &RecordBatch,
) -> PyResult<Bound<'py, PyAny>> {
    let batch = scx_format::upcast_to_large_types(batch)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    // Defensive: stamp pandas `index_columns` metadata if the schema
    // carries a literal `__index_level_0__` / `_index` column without
    // it. anndata 0.10+ hard-rejects `_index` as a regular DataFrame
    // column on `write_h5ad`, so this prevents `_index` from leaking
    // into `df.columns` regardless of how the underlying SCX was
    // written. See `scx_format::ensure_pandas_index_metadata` doc.
    let batch = scx_format::ensure_pandas_index_metadata(&batch);
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
///   `scx_format::ensure_pandas_index_metadata`) → `Table.to_pandas()`
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
fn ordered_categorical_columns(table: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
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
fn apply_categorical_ordered(df: &Bound<'_, PyAny>, ordered_cols: &[String]) -> PyResult<()> {
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
struct EnvelopeInfo {
    /// First string entry from `index_columns` — the column to use as
    /// the DataFrame index on the minimal-envelope branch.
    index_col: String,
    /// `true` when the envelope is the bare `{"index_columns": [...]}`
    /// shape stamped by `scx-convert/src/h5ad_read.rs` and
    /// `scx_format::ensure_pandas_index_metadata`. `false` when the
    /// envelope is the full pyarrow shape stamped by
    /// `pyarrow.Table.from_pandas` (carries `columns`,
    /// `column_indexes`, `pandas_version`, `creator`).
    is_minimal: bool,
}

/// Parse the schema's `pandas` metadata envelope, if any. Distinguishes
/// the minimal envelope (KeyError-on-`to_pandas`, must be stripped) from
/// the full envelope (carries dtype hints, must be preserved so
/// pandas-extension dtypes and multi-level indexes round-trip).
fn extract_scx_envelope_info<'py>(table: &Bound<'py, PyAny>) -> PyResult<Option<EnvelopeInfo>> {
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
fn strip_pandas_metadata<'py>(table: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
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
fn record_batch_to_numpy2d<'py>(
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
            Err(scx_format::ScxError::SectionNotFound(_)) => Ok(std::collections::HashMap::new()),
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
fn build_eager_obsm_dict(
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
fn coo_needs_int64_coords(n_rows: usize, n_cols: usize) -> bool {
    n_rows > i32::MAX as usize || n_cols > i32::MAX as usize
}

/// Borrowed view of the row/col columns of a pairwise COO RecordBatch,
/// dispatched on whichever Int32 / Int64 dtype the on-disk batch uses.
/// Lets every consumer treat both wire-format widths uniformly.
enum CooCoordsRef<'a> {
    Int32(&'a arrow::array::Int32Array, &'a arrow::array::Int32Array),
    Int64(&'a arrow::array::Int64Array, &'a arrow::array::Int64Array),
}

impl CooCoordsRef<'_> {
    fn len(&self) -> usize {
        use arrow::array::Array;
        match self {
            CooCoordsRef::Int32(r, _) => r.len(),
            CooCoordsRef::Int64(r, _) => r.len(),
        }
    }

    fn row_i64(&self, i: usize) -> i64 {
        match self {
            CooCoordsRef::Int32(r, _) => r.value(i) as i64,
            CooCoordsRef::Int64(r, _) => r.value(i),
        }
    }

    fn col_i64(&self, i: usize) -> i64 {
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
fn coo_coords_from_batch(batch: &RecordBatch) -> PyResult<CooCoordsRef<'_>> {
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

/// Build an AnnData object from an ScxReader with optional layer filtering.
///
/// `eager` controls how `obsp` / `varp` / `varm` / `layers` are
/// populated. When `false` (default for `pyscx.open(...).to_anndata()`),
/// these slots are wrapped in `ScxLazyPairwiseMapping` /
/// `ScxLazyVarmMapping` / `ScxLazyLayersMapping` and attached to the
/// AnnData's private `_obsp` / `_varp` / `_varm` / `_layers` storage,
/// deferring each section's decode until the consumer first accesses
/// `ad.obsp[…]` etc. Keeps peak RSS of `to_anndata()` bounded for
/// files that carry large kNN graphs / embeddings. When `true`, every
/// section is decoded up front and a plain `dict` is passed through
/// the AnnData constructor — matches pre-fix behaviour and detaches
/// the returned AnnData from the SCX file handle. See
/// [`crate::lazy_mapping`].
#[allow(clippy::too_many_arguments)]
fn to_anndata_with_layers<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    reader: &ScxReader,
    layer_filter: Option<&[String]>,
    obsm_filter: Option<&[String]>,
    eager: bool,
    memory_budget: Option<u64>,
    skip_x: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use crate::lazy_mapping::{
        PairwiseAxis, ScxLazyLayersMapping, ScxLazyObsmMapping, ScxLazyPairwiseMapping,
        ScxLazyVarmMapping,
    };
    use std::sync::Arc;

    let anndata_mod = py.import("anndata")?;

    // Phase 4d: catalog-only estimate of the full-assembly bytes. If
    // the estimate exceeds the budget (caller's `memory_budget` kwarg,
    // or `DEFAULT_EAGER_MEMORY_BUDGET_BYTES` = 8 GiB when unset) emit a
    // `UserWarning` recommending the backed / query alternatives.
    // Assembly proceeds regardless — the warning is advisory.
    let budget = memory_budget.unwrap_or(DEFAULT_EAGER_MEMORY_BUDGET_BYTES);
    let est_bytes = estimate_eager_assembly_bytes(reader);
    if est_bytes > budget {
        warn_python_convert(
            py,
            &scx_convert::ConvertWarning::EagerAssemblyMemoryHigh {
                estimated_bytes: est_bytes,
                budget_bytes: budget,
            },
        )?;
    }

    // X — assemble all CSR shards (with deletion vector filtering).
    //
    // `skip_x` builds an X-less skeleton: obs/var define the shape and the
    // caller assigns `adata.X` afterwards. Used by `to_gpu_anndata`'s
    // device-resident streamed path, which decodes X straight onto the GPU
    // instead of materialising a host scipy CSR here. obs/var/obsm/uns/layers
    // are assembled identically either way.
    let x = if skip_x {
        None
    } else {
        let csr = reader.read_all_csr_shards_filtered().map_err(to_pyerr)?;
        Some(csr_to_scipy(py, csr)?)
    };

    // obs metadata — filter by deletion vectors if present
    let obs = match reader.read_obs() {
        Ok(batch) => {
            let filtered_batch = filter_obs_by_deletion_vectors(reader, batch)?;
            let table = record_batch_to_pyarrow(py, &filtered_batch)?;
            Some(pyarrow_table_to_pandas(&table)?)
        }
        Err(scx_format::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    // var metadata
    let var = match reader.read_var() {
        Ok(batch) => {
            let table = record_batch_to_pyarrow(py, &batch)?;
            Some(pyarrow_table_to_pandas(&table)?)
        }
        Err(scx_format::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    // obsm embeddings.
    //
    // `obsm` is eager by default (it tends to be small relative to
    // obsp/varp/varm). It only becomes a lazy bridge when the caller has
    // explicitly opted into selective loading via `obsm=[...]` AND not
    // forced `eager=True` — keeping default semantics byte-identical
    // while letting the random-access dataloader path defer (and skip)
    // per-key materialisation. See `ScxLazyObsmMapping`.
    //
    // `obsm_filter` restricts the loaded keys in either mode
    // (`to_anndata(obsm=[...])`).
    let lazy_obsm_requested = !eager && obsm_filter.is_some();
    let obsm_dict = pyo3::types::PyDict::new(py);
    if lazy_obsm_requested {
        // Lazy path: validate the requested keys exist now via a
        // catalog-only scan (no shard bytes read — preserves the
        // deferral guarantee), then let the bridge read each on first
        // access.
        validate_obsm_keys(reader, obsm_filter)?;
    } else {
        let obsm_map = read_obsm_selected(reader, obsm_filter)?;
        for (name, batch) in &obsm_map {
            let filtered = filter_obs_by_deletion_vectors(reader, batch.clone())?;
            let np_arr = obsm_batch_to_numpy(py, &filtered)?;
            obsm_dict.set_item(name, np_arr)?;
        }
    }

    // uns — reconstruct any `__scx_type__` envelopes back into NumPy
    // ndarrays / scalars / tuples / pandas Index/Series/Categorical /
    // structured recarrays. Plain JSON passes through unchanged.
    let uns_dict = read_uns_as_pyobject(py, reader)?;

    // Sibling reader for the lazy bridges (`_obsp`/`_varp`/`_varm`/
    // `_layers`). Independent mmap so the returned AnnData stays valid
    // after the caller's `ScxReader` drops. Skipped when no lazy slot
    // is needed.
    let has_obsp = !reader.list_obsp().is_empty();
    let has_varp = !reader.list_varp().is_empty();
    let has_varm = !reader.list_varm().is_empty();
    let layer_names = reader.layer_names();
    let has_layers = if let Some(filter) = layer_filter {
        layer_names.iter().any(|n| filter.iter().any(|f| f == n))
    } else {
        !layer_names.is_empty()
    };
    let need_lazy = has_obsp || has_varp || has_varm || has_layers || lazy_obsm_requested;

    let obsp_kept = if has_obsp {
        compute_kept_to_global(reader)?.map(Arc::new)
    } else {
        None
    };

    let lazy_reader: Option<Arc<ScxReader>> = if need_lazy {
        Some(Arc::new(
            ScxReader::open_with_shared_catalog(path, reader.catalog_arc()).map_err(to_pyerr)?,
        ))
    } else {
        None
    };
    let lazy_obsp = lazy_reader
        .as_ref()
        .filter(|_| has_obsp)
        .map(|r| ScxLazyPairwiseMapping::new(Arc::clone(r), PairwiseAxis::Obsp, obsp_kept.clone()));
    let lazy_varp = lazy_reader
        .as_ref()
        .filter(|_| has_varp)
        .map(|r| ScxLazyPairwiseMapping::new(Arc::clone(r), PairwiseAxis::Varp, None));
    let lazy_varm = lazy_reader
        .as_ref()
        .filter(|_| has_varm)
        .map(|r| ScxLazyVarmMapping::new(Arc::clone(r)));
    let lazy_layers = lazy_reader
        .as_ref()
        .filter(|_| has_layers)
        .map(|r| ScxLazyLayersMapping::new(Arc::clone(r), layer_filter));
    let lazy_obsm = lazy_reader
        .as_ref()
        .filter(|_| lazy_obsm_requested)
        .map(|r| ScxLazyObsmMapping::new(Arc::clone(r), obsm_filter));

    // Build AnnData kwargs.
    let kwargs = pyo3::types::PyDict::new(py);
    if let Some(x) = x {
        kwargs.set_item("X", x)?;
    }
    if let Some(obs) = obs {
        kwargs.set_item("obs", obs)?;
    }
    if let Some(var) = var {
        kwargs.set_item("var", var)?;
    }
    if !obsm_dict.is_empty() {
        kwargs.set_item("obsm", obsm_dict)?;
    }
    if let Some(uns) = uns_dict {
        kwargs.set_item("uns", uns)?;
    }
    if eager {
        // Materialize each lazy bridge up front; AnnData's __init__
        // receives plain dicts (same shape as pre-fix). Returned AnnData
        // is fully detached from the SCX file handle.
        if let Some(m) = &lazy_obsp {
            kwargs.set_item("obsp", m.materialize_all(py)?)?;
        }
        if let Some(m) = &lazy_varp {
            kwargs.set_item("varp", m.materialize_all(py)?)?;
        }
        if let Some(m) = &lazy_varm {
            kwargs.set_item("varm", m.materialize_all(py)?)?;
        }
        if let Some(m) = &lazy_layers {
            kwargs.set_item("layers", m.materialize_all(py)?)?;
        }
    }

    let adata = anndata_mod.call_method("AnnData", (), Some(&kwargs))?;

    // Reconstruct `adata.raw` if the file carries a raw count matrix.
    // Raw shares X's obs axis; when deletion vectors are active the raw
    // rows would need the same filtering as X, which this path does not
    // yet apply — warn and drop rather than emit a misaligned raw.
    if reader.has_raw() {
        if reader.header().has_deletion_vectors() {
            warn_python_convert(
                py,
                &scx_convert::ConvertWarning::DroppedRaw {
                    raw_n_vars: reader.raw_n_vars().unwrap_or(0),
                },
            )?;
        } else {
            let raw_csr = reader.read_all_raw_csr_shards().map_err(to_pyerr)?;
            let raw_x = csr_to_scipy(py, raw_csr)?;
            let raw_var_batch = reader.read_raw_var().map_err(to_pyerr)?;
            let raw_var_table = record_batch_to_pyarrow(py, &raw_var_batch)?;
            let raw_var = pyarrow_table_to_pandas(&raw_var_table)?;
            let raw_kwargs = pyo3::types::PyDict::new(py);
            raw_kwargs.set_item("X", raw_x)?;
            raw_kwargs.set_item("var", raw_var)?;
            let raw_adata = anndata_mod.call_method("AnnData", (), Some(&raw_kwargs))?;
            // `adata.raw = AnnData(X=..., var=...)` stores it as a Raw —
            // the canonical scanpy idiom.
            adata.setattr("raw", raw_adata)?;
        }
    }

    if !eager {
        // Lazy mode: attach each bridge to AnnData's private `_obsp` /
        // `_varp` / `_varm` / `_layers` storage. AnnData's
        // `AlignedMappingProperty` descriptor reads from these on
        // every public `.obsp` (etc.) access — the first access drives
        // the bridge's per-key materialization through AnnData's
        // validation loop; subsequent accesses hit the bridge's cache.
        // We bypass the property setter (which would otherwise iterate
        // and validate every entry up front, defeating the lazy point).
        if let Some(m) = lazy_obsp {
            adata.setattr("_obsp", m.into_pyobject(py)?)?;
        }
        if let Some(m) = lazy_varp {
            adata.setattr("_varp", m.into_pyobject(py)?)?;
        }
        if let Some(m) = lazy_varm {
            adata.setattr("_varm", m.into_pyobject(py)?)?;
        }
        if let Some(m) = lazy_layers {
            adata.setattr("_layers", m.into_pyobject(py)?)?;
        }
        // Lazy obsm is only built when the caller opted into selective
        // loading (`obsm=[...]`, eager=False); default behaviour keeps
        // obsm eager. Attach via `_obsm` to bypass AnnData's axis-length
        // validation, same as the other bridges.
        if let Some(m) = lazy_obsm {
            adata.setattr("_obsm", m.into_pyobject(py)?)?;
        }
    }

    Ok(adata)
}

/// Build an AnnData with optional var_names projection, obs_filter, and layers selection.
///
/// For obs_filter: delegates to the QueryPipeline for predicate pushdown.
/// For var_names: resolves gene names to column indices and applies column slicing.
/// For layers: filters which layers are loaded.
///
/// When `preserve_slots=true` and `obs_filter` is set, the eager path
/// `to_anndata_with_layers()` is used (loading X / obs / var / obsm /
/// layers with deletion vectors applied) and then sliced by a pandas.eval
/// boolean mask. This preserves obsm and layers at the cost of the query
/// engine's predicate-pushdown shard skipping. When `preserve_slots=false`
/// (default), the query-engine path runs and emits a warning if obsm or
/// layers exist on disk (since they are dropped from the result).
#[allow(clippy::too_many_arguments)]
pub fn to_anndata_filtered<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    reader: &ScxReader,
    var_names: Option<&[String]>,
    obs_filter: Option<&str>,
    layer_filter: Option<&[String]>,
    obsm_filter: Option<&[String]>,
    preserve_slots: bool,
    eager: bool,
    memory_budget: Option<u64>,
    skip_x: bool,
) -> PyResult<Bound<'py, PyAny>> {
    // `skip_x` (the X-less skeleton for `to_gpu_anndata`) is only meaningful on
    // the no-filter fast path — the caller guarantees no var_names / obs_filter /
    // layer projection when it sets it (those paths reshape X and must build it).
    debug_assert!(
        !skip_x || (var_names.is_none() && obs_filter.is_none() && layer_filter.is_none()),
        "skip_x requires no var_names / obs_filter / layer_filter"
    );

    // Fast path: no filtering → use existing implementation
    if var_names.is_none() && obs_filter.is_none() && layer_filter.is_none() {
        return to_anndata_with_layers(
            py,
            path,
            reader,
            None,
            obsm_filter,
            eager,
            memory_budget,
            skip_x,
        );
    }

    // preserve_slots=true with obs_filter: load full AnnData, then filter
    // rows via pandas.eval. Keeps obsm / layers / uns intact at the cost
    // of skipping query-engine predicate pushdown. Force eager so the
    // pandas-side __getitem__ slicing operates on real arrays rather
    // than lazy bridges (which AnnData iterates / validates during
    // `.copy()` anyway).
    if let (Some(expr), true) = (obs_filter, preserve_slots) {
        let full = to_anndata_with_layers(
            py,
            path,
            reader,
            layer_filter,
            obsm_filter,
            true,
            memory_budget,
            false,
        )?;

        let obs_attr = full.getattr("obs")?;
        let mask = obs_attr.call_method1("eval", (expr,)).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "preserve_slots=True parses obs_filter via pandas.eval; \
                 failed to evaluate {expr:?}: {e}"
            ))
        })?;

        // Reject non-boolean results: AnnData treats numeric arrays as
        // positional indices, which would silently reorder rows instead
        // of failing on a malformed predicate.
        let dtype_kind: String = mask.getattr("dtype")?.getattr("kind")?.extract()?;
        if dtype_kind != "b" {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "preserve_slots=True requires obs_filter to evaluate to a \
                 boolean mask (e.g. \"cell_type == 'T cell'\"); expression \
                 {expr:?} produced dtype kind {dtype_kind:?}"
            )));
        }

        // Surface the grammar shift: this path evaluates obs_filter via
        // pandas.eval, which does not match the SCX predicate engine
        // (e.g. pandas accepts `&` / `|` / `~`; SCX accepts only
        // `and` / `or` / `not`). Users opted into preserve_slots=True, so
        // one warning per call is appropriate.
        py.import("warnings")?.call_method1(
            "warn",
            (format!(
                "preserve_slots=True evaluated obs_filter {expr:?} via pandas.eval; \
                 grammar differs from the SCX predicate engine used by \
                 preserve_slots=False (see docs/scanpy.md \"Filter Expression Compatibility\")."
            ),),
        )?;

        let builtins = py.import("builtins")?;
        let slice_all = builtins.call_method1("slice", (py.None(),))?;
        let col_idx = if let Some(names) = var_names {
            let indices = resolve_var_names_to_indices(reader, names)?;
            PyArray1::from_vec(py, indices).into_any().unbind()
        } else {
            slice_all.unbind()
        };
        let idx = pyo3::types::PyTuple::new(py, &[mask.unbind(), col_idx])?;
        return full.get_item(idx)?.call_method0("copy");
    }

    // If obs_filter is specified, use the query engine for predicate pushdown
    if let Some(expr) = obs_filter {
        use scx_engine::QueryPipeline;

        let mut pipeline =
            QueryPipeline::open(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        pipeline = pipeline
            .filter_obs(expr)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // If var_names is also specified, resolve to gene indices
        if let Some(names) = var_names {
            let gene_indices = resolve_var_names_to_indices(reader, names)?;
            pipeline = pipeline.select_genes(gene_indices);
        }

        let result = pipeline
            .collect()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let anndata_mod = py.import("anndata")?;
        let x = csr_to_scipy(py, result.x)?;
        let obs_table = record_batch_to_pyarrow(py, &result.obs)?;
        let obs_df = pyarrow_table_to_pandas(&obs_table)?;
        let var_table = record_batch_to_pyarrow(py, &result.var)?;
        let var_df = pyarrow_table_to_pandas(&var_table)?;

        // uns (still loaded from reader; see read_uns_as_pyobject for the
        // tagged-envelope reconstruction).
        let uns_dict = read_uns_as_pyobject(py, reader)?;

        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("X", x)?;
        kwargs.set_item("obs", obs_df)?;
        kwargs.set_item("var", var_df)?;
        if let Some(uns) = uns_dict {
            kwargs.set_item("uns", uns)?;
        }
        // obsm, varm, obsp, varp, and layers are not available via QueryResult.
        // Warn if the source file contains them so users know they're being dropped.
        let has_obsm = reader
            .read_all_obsm()
            .map(|m| !m.is_empty())
            .unwrap_or(false);
        let has_varm = reader
            .read_all_varm()
            .map(|m| !m.is_empty())
            .unwrap_or(false);
        let has_obsp = reader
            .read_all_obsp()
            .map(|m| !m.is_empty())
            .unwrap_or(false);
        let has_varp = reader
            .read_all_varp()
            .map(|m| !m.is_empty())
            .unwrap_or(false);
        let has_layers = !reader.layer_names().is_empty();
        if has_obsm || has_varm || has_obsp || has_varp || has_layers {
            let warnings = py.import("warnings")?;
            let mut parts = Vec::new();
            if has_obsm {
                parts.push("obsm");
            }
            if has_varm {
                parts.push("varm");
            }
            if has_obsp {
                parts.push("obsp");
            }
            if has_varp {
                parts.push("varp");
            }
            if has_layers {
                parts.push("layers");
            }
            warnings.call_method1(
                "warn",
                (format!(
                    "obs_filter with non-backed mode uses the query engine, which does not \
                     load {}. Pass preserve_slots=True to materialize them (skips predicate \
                     pushdown), use backed=True, or load the full dataset and filter in Python.",
                    parts.join(", ")
                ),),
            )?;
        }

        // The obs-filtered query path does not subset the raw matrix's
        // obs axis — warn + drop rather than emit a misaligned raw.
        if reader.has_raw() {
            warn_python_convert(
                py,
                &scx_convert::ConvertWarning::DroppedRaw {
                    raw_n_vars: reader.raw_n_vars().unwrap_or(0),
                },
            )?;
        }

        let adata = anndata_mod.call_method("AnnData", (), Some(&kwargs))?;
        return Ok(adata);
    }

    // No obs_filter but var_names and/or layers specified
    // Load normally, then apply var_names column projection. Force eager
    // because slicing the AnnData by var_names triggers AlignedMapping
    // validation across all aligned slots (obsp / varp / varm), which
    // would materialize through the lazy bridges anyway — doing it up
    // front avoids fragmenting the cost across implicit slicing.
    let adata = to_anndata_with_layers(
        py,
        path,
        reader,
        layer_filter,
        obsm_filter,
        true,
        memory_budget,
        false,
    )?;

    if let Some(names) = var_names {
        // Resolve via the same path as backed / query-engine: scans all string
        // columns (so gene symbols in non-index columns work) and returns
        // sorted positional indices. Slicing adata[:, np_indices] then projects
        // X, layers, var, varm, and varp consistently.
        let indices = resolve_var_names_to_indices(reader, names)?;
        let np_indices = PyArray1::from_vec(py, indices);

        let builtins = py.import("builtins")?;
        let slice_all = builtins.call_method1("slice", (py.None(),))?;
        let idx =
            pyo3::types::PyTuple::new(py, &[slice_all.unbind(), np_indices.into_any().unbind()])?;
        let sliced = adata.get_item(idx)?;
        let copied = sliced.call_method0("copy")?;
        return Ok(copied);
    }

    Ok(adata)
}

/// Resolve gene names to column indices using the var metadata.
fn resolve_var_names_to_indices(reader: &ScxReader, names: &[String]) -> PyResult<Vec<u32>> {
    let var_batch = match reader.read_var() {
        Ok(batch) => batch,
        Err(scx_format::ScxError::SectionNotFound(_)) => {
            return Err(PyRuntimeError::new_err(
                "Cannot resolve var_names: this SCX file has no var metadata. \
                 Open without var_names to load all genes."
                    .to_string(),
            ));
        }
        Err(e) => return Err(to_pyerr(e)),
    };

    // Try to find gene names in the var DataFrame index.
    // The index column is typically the first column (or named "gene_id").
    // We check all string columns.
    let mut name_to_idx: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();

    for col_idx in 0..var_batch.num_columns() {
        let col = var_batch.column(col_idx);
        if let Some(str_arr) = col.as_any().downcast_ref::<arrow::array::StringArray>() {
            for (row, val) in str_arr.iter().enumerate() {
                if let Some(v) = val {
                    name_to_idx.entry(v).or_insert(row as u32);
                }
            }
        }
    }

    let mut indices = Vec::with_capacity(names.len());
    let mut not_found = Vec::new();
    for name in names {
        match name_to_idx.get(name.as_str()) {
            Some(&idx) => indices.push(idx),
            None => not_found.push(name.as_str()),
        }
    }

    if indices.is_empty() {
        return Err(PyRuntimeError::new_err(format!(
            "None of the requested var_names were found in the var metadata: {:?}",
            not_found
        )));
    }

    // Sort + dedup so all callers produce var rows in sorted column-position
    // order, matching scx-engine::project_var(). Keeps eager / backed /
    // query-engine paths consistent under reordered or duplicated requests.
    indices.sort_unstable();
    indices.dedup();

    Ok(indices)
}

/// Build an AnnData object with backed (on-demand) X and layers.
///
/// Opens a new ScxReader (independent mmap) so the backed dataset can
/// outlive the PyExperiment that created it. obs/var/obsm/uns are loaded
/// eagerly (same as non-backed mode).
///
/// When deletion vectors are present, a `kept_to_global` mapping is
/// computed and passed to `ScxBackedSparseDataset` so that user-visible
/// row indices exclude deleted rows (matching non-backed behavior).
#[allow(clippy::too_many_arguments)]
pub fn to_anndata_backed<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    cache_shards: usize,
    var_names: Option<&[String]>,
    obs_filter: Option<&str>,
    layer_filter: Option<&[String]>,
    obsm_filter: Option<&[String]>,
    eager: bool,
) -> PyResult<Bound<'py, PyAny>> {
    to_anndata_backed_with_options(
        py,
        path,
        cache_shards,
        var_names,
        obs_filter,
        layer_filter,
        obsm_filter,
        true,
        eager,
    )
}

/// Internal entrypoint for the backed AnnData builder.
///
/// `apply_deletion_vectors`: when `true` (the default for the public
/// `to_anndata_backed`), the X / obs / obsm / obsp paths are filtered through
/// the file's global deletion vectors. When `false`, the function returns the
/// unfiltered axes — used by `mudata::to_mudata_backed` so that the inner
/// AnnData's `obs` row count matches the outer MuData's global `obs` (which
/// is also unfiltered, matching the eager `to_mudata` path's behaviour).
#[allow(clippy::too_many_arguments)]
pub(crate) fn to_anndata_backed_with_options<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    cache_shards: usize,
    var_names: Option<&[String]>,
    obs_filter: Option<&str>,
    layer_filter: Option<&[String]>,
    obsm_filter: Option<&[String]>,
    apply_deletion_vectors: bool,
    eager: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use crate::backed::{ScxBackedLayerDataset, ScxBackedSparseDataset};
    use crate::lazy_mapping::{
        PairwiseAxis, ScxLazyObsmMapping, ScxLazyPairwiseMapping, ScxLazyVarmMapping,
    };
    use scx_format::BackedCsrReader;
    use std::sync::Arc;

    let anndata_mod = py.import("anndata")?;
    let reader = ScxReader::open(path).map_err(to_pyerr)?;
    // Share one parsed `FullCatalog` across the N+3 `ScxReader`
    // instances this function constructs (main reader + X CSR + CSC
    // sidecar + one per backed layer). The catalog is bytes-identical
    // across all opens of the same file, so re-parsing it N+3 times
    // per worker is pure overhead — the worker-amplification path that
    // motivated the Arc-sharing change. The shard cache and
    // singleflight table stay per-instance; only the immutable
    // catalog is reused. See docs/multithreading.md for the
    // fork-safety contract.
    let shared_catalog = reader.catalog_arc();

    // --- Compute kept_to_global from deletion vectors (if present) ---
    // Cache the deletion-vector-only mapping; obs_filter may mutate kept_to_global
    // further, but obsm filtering needs the original DV-only version.
    // Skipped when `apply_deletion_vectors` is false (e.g. `to_mudata_backed`
    // single-modality wrap, where DVs are intentionally not applied to keep
    // inner-AnnData obs in lockstep with the unfiltered outer MuData obs).
    let dv_kept_to_global = if apply_deletion_vectors {
        compute_kept_to_global(&reader)?
    } else {
        None
    };
    let mut kept_to_global = dv_kept_to_global.clone();

    // --- obs (eager, optionally filtered by deletion vectors) ---
    let obs = match reader.read_obs() {
        Ok(batch) => {
            let filtered_batch = if apply_deletion_vectors {
                filter_obs_by_deletion_vectors(&reader, batch)?
            } else {
                batch
            };
            let table = record_batch_to_pyarrow(py, &filtered_batch)?;
            Some(pyarrow_table_to_pandas(&table)?)
        }
        Err(scx_format::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    // --- Apply obs_filter if specified ---
    // Evaluate on the pandas DataFrame rather than QueryPipeline. In backed
    // mode, shard-level pushdown has negligible benefit since X is lazy (only
    // accessed shards are decoded). Pandas .query() is simpler and supports
    // richer expressions.
    let obs = if let Some(expr) = obs_filter {
        if let Some(obs_df) = obs {
            // Use pandas query to filter
            let filtered = obs_df.call_method1("query", (expr,))?;
            let original_idx = obs_df.getattr("index")?;
            let filtered_idx = filtered.getattr("index")?;

            // Get positional indices of kept rows in the (already deletion-filtered) obs
            let np = py.import("numpy")?;
            let isin_mask = original_idx.call_method1("isin", (&filtered_idx,))?;
            let where_result = np.call_method1("where", (&isin_mask,))?;
            // np.where returns a tuple; first element is array of indices
            let pos_indices = where_result.get_item(0)?;
            let pos_arr: numpy::PyReadonlyArray1<'_, i64> = pos_indices
                .call_method1("astype", (np.getattr("int64")?,))?
                .extract()?;
            let pos_slice = pos_arr
                .as_slice()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            // Update kept_to_global to reflect the obs_filter
            match &kept_to_global {
                Some(existing) => {
                    // existing maps user-visible → global. Now further filter.
                    let new_kept: Vec<u64> =
                        pos_slice.iter().map(|&i| existing[i as usize]).collect();
                    kept_to_global = Some(new_kept);
                }
                None => {
                    // No prior deletions. pos_slice maps directly to global.
                    let new_kept: Vec<u64> = pos_slice.iter().map(|&i| i as u64).collect();
                    kept_to_global = Some(new_kept);
                }
            }

            Some(filtered)
        } else {
            None
        }
    } else {
        obs
    };

    // --- Resolve var_names to column indices ---
    let col_indices = if let Some(names) = var_names {
        Some(resolve_var_names_to_indices(&reader, names)?)
    } else {
        None
    };

    // --- X: backed ---
    let x_reader =
        ScxReader::open_with_shared_catalog(path, Arc::clone(&shared_catalog)).map_err(to_pyerr)?;
    let has_csc = x_reader.header().has_csc();
    let x_backed = Arc::new(BackedCsrReader::new(x_reader, cache_shards));
    let x_backed_csc: Option<Arc<scx_format::BackedCscReader>> = if has_csc {
        // Open a separate ScxReader for the CSC sidecar (BackedCscReader
        // takes ownership). Header check is cheap; the reader holds a
        // mmap and per-shard catalog, but no shards decode until we
        // actually call read_csc_shard(). Catalog parse is skipped via
        // the shared `Arc<FullCatalog>`.
        let csc_reader = ScxReader::open_with_shared_catalog(path, Arc::clone(&shared_catalog))
            .map_err(to_pyerr)?;
        Some(Arc::new(
            scx_format::BackedCscReader::new(csc_reader, cache_shards).map_err(to_pyerr)?,
        ))
    } else {
        None
    };
    let mut x_dataset = match &kept_to_global {
        Some(mapping) => ScxBackedSparseDataset::from_reader_with_deletions(
            Arc::clone(&x_backed),
            cache_shards,
            mapping.clone(),
        ),
        None => ScxBackedSparseDataset::from_reader(Arc::clone(&x_backed), cache_shards),
    };
    x_dataset.with_csc_reader(x_backed_csc);
    x_dataset.with_source_path(path);
    if let Some(ref indices) = col_indices {
        x_dataset.set_col_projection(indices.clone());
    }

    // --- var (eager, optionally filtered by var_names) ---
    let var = match reader.read_var() {
        Ok(batch) => {
            let table = record_batch_to_pyarrow(py, &batch)?;
            let df = pyarrow_table_to_pandas(&table)?;
            if let Some(ref indices) = col_indices {
                // Slice var positionally with the same indices used to project
                // X (set_col_projection above). Using df.iloc keeps var aligned
                // with X when names match a non-index column like gene_symbol;
                // the prior var.index.isin(names) approach produced an empty
                // var when symbols were resolved from non-index columns.
                let np_indices = PyArray1::from_slice(py, indices);
                let iloc = df.getattr("iloc")?;
                let filtered = iloc.get_item(np_indices)?;
                Some(filtered)
            } else {
                Some(df)
            }
        }
        Err(scx_format::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    // --- obsm ---
    //
    // Backed dense row-gather (`ScxBackedObsmDataset`) kicks in when the
    // caller selected obsm keys (`obsm=[...]`), did not force `eager`,
    // and did not pass `obs_filter`. Under `obs_filter` we fall back to
    // the eager path (composing a pandas-query row mask with shard
    // gather is deferred), and `obsm=None` keeps the historical
    // eager-all behaviour.
    let use_backed_obsm = obsm_filter.is_some() && !eager && obs_filter.is_none();
    // `obsm_filter` restricts the loaded keys; under backed mode we only
    // need to validate them (the bridge reads each lazily).
    let obsm_dict = pyo3::types::PyDict::new(py);
    if use_backed_obsm {
        validate_obsm_keys(&reader, obsm_filter)?;
    } else {
        build_eager_obsm_dict(
            py,
            &reader,
            obsm_filter,
            obs_filter,
            apply_deletion_vectors,
            &kept_to_global,
            &dv_kept_to_global,
            &obsm_dict,
        )?;
    }

    let lazy_obsm = if use_backed_obsm {
        let obsm_reader = Arc::new(
            ScxReader::open_with_shared_catalog(path, Arc::clone(&shared_catalog))
                .map_err(to_pyerr)?,
        );
        // obs_filter is None here, so kept_to_global == dv_kept_to_global
        // (deletion vectors only).
        let kept_arc = kept_to_global.as_ref().map(|k| Arc::new(k.clone()));
        let config = crate::lazy_mapping::BackedObsmConfig {
            path: path.to_path_buf(),
            cache_shards,
            shared_catalog: Arc::clone(&shared_catalog),
            kept_to_global: kept_arc,
        };
        Some(ScxLazyObsmMapping::new_backed(
            obsm_reader,
            obsm_filter,
            config,
        ))
    } else {
        None
    };

    // --- obsp / varp / varm — lazy bridges by default ---
    //
    // Same `Arc<ScxReader>` (sibling of the main one, sharing the
    // parsed catalog) backs all three bridges; refcount-only clones
    // when handing it to each `ScxLazyPairwiseMapping` /
    // `ScxLazyVarmMapping`. Each bridge decodes its sections on the
    // consumer's first `ad.obsp[…]` / `.varp[…]` / `.varm[…]` access.
    // See `to_anndata_with_layers` for the contract between
    // `eager=true/false` and AnnData's private `_obsp` / `_varp` /
    // `_varm` storage.
    let has_obsp = !reader.list_obsp().is_empty();
    let has_varp = !reader.list_varp().is_empty();
    let has_varm = !reader.list_varm().is_empty();
    let need_lazy_aligned = has_obsp || has_varp || has_varm;
    let lazy_reader: Option<Arc<ScxReader>> = if need_lazy_aligned {
        Some(Arc::new(
            ScxReader::open_with_shared_catalog(path, Arc::clone(&shared_catalog))
                .map_err(to_pyerr)?,
        ))
    } else {
        None
    };
    // Backed-path obsp filter mirrors the eager pre-fix logic at
    // anndata.rs (kept_to_global composes deletion vectors with
    // obs_filter); shared by Arc so the bridge holds its own ref.
    let kept_to_global_arc = kept_to_global.as_ref().map(|k| Arc::new(k.clone()));
    let lazy_obsp = lazy_reader.as_ref().filter(|_| has_obsp).map(|r| {
        ScxLazyPairwiseMapping::new(
            Arc::clone(r),
            PairwiseAxis::Obsp,
            kept_to_global_arc.clone(),
        )
    });
    let lazy_varp = lazy_reader
        .as_ref()
        .filter(|_| has_varp)
        .map(|r| ScxLazyPairwiseMapping::new(Arc::clone(r), PairwiseAxis::Varp, None));
    let lazy_varm = lazy_reader
        .as_ref()
        .filter(|_| has_varm)
        .map(|r| ScxLazyVarmMapping::new(Arc::clone(r)));

    // --- uns (eager; tagged envelopes reconstructed) ---
    let uns_dict = read_uns_as_pyobject(py, &reader)?;

    // --- layers (backed, with optional filtering) ---
    let all_layer_names = reader.layer_names();
    let layers_dict = pyo3::types::PyDict::new(py);
    for name in &all_layer_names {
        // Skip layers not in the filter list (if specified)
        if let Some(filter) = layer_filter {
            if !filter.iter().any(|f| f == name) {
                continue;
            }
        }
        let l_reader = ScxReader::open_with_shared_catalog(path, Arc::clone(&shared_catalog))
            .map_err(to_pyerr)?;
        let l_backed = Arc::new(BackedCsrReader::new_for_layer(l_reader, name, cache_shards));
        let mut l_dataset = match &kept_to_global {
            Some(mapping) => ScxBackedLayerDataset::from_reader_with_deletions(
                l_backed,
                cache_shards,
                name.clone(),
                mapping.clone(),
            ),
            None => ScxBackedLayerDataset::from_reader(l_backed, cache_shards, name.clone()),
        };
        if let Some(ref indices) = col_indices {
            l_dataset.inner.set_col_projection(indices.clone());
        }
        let l_py = l_dataset.into_pyobject(py)?;
        layers_dict.set_item(name, l_py)?;
    }

    // Build AnnData kwargs
    let kwargs = pyo3::types::PyDict::new(py);
    let x_py = x_dataset.into_pyobject(py)?;
    kwargs.set_item("X", x_py)?;
    if let Some(obs) = obs {
        kwargs.set_item("obs", obs)?;
    }
    if let Some(var) = var {
        kwargs.set_item("var", var)?;
    }
    if !obsm_dict.is_empty() {
        kwargs.set_item("obsm", obsm_dict)?;
    }
    if let Some(uns) = uns_dict {
        kwargs.set_item("uns", uns)?;
    }
    if !layers_dict.is_empty() {
        kwargs.set_item("layers", layers_dict)?;
    }
    if eager {
        // Eager mode: materialize lazy bridges up front and pass
        // through normal kwargs path. Caller receives an AnnData
        // detached from the SCX file handle.
        if let Some(m) = &lazy_obsp {
            kwargs.set_item("obsp", m.materialize_all(py)?)?;
        }
        if let Some(m) = &lazy_varp {
            kwargs.set_item("varp", m.materialize_all(py)?)?;
        }
        if let Some(m) = &lazy_varm {
            kwargs.set_item("varm", m.materialize_all(py)?)?;
        }
    }

    let adata = anndata_mod.call_method("AnnData", (), Some(&kwargs))?;

    // Backed mode does not reconstruct the raw matrix — warn + drop.
    if reader.has_raw() {
        warn_python_convert(
            py,
            &scx_convert::ConvertWarning::DroppedRaw {
                raw_n_vars: reader.raw_n_vars().unwrap_or(0),
            },
        )?;
    }

    if !eager {
        // Lazy mode: attach bridges directly to AnnData's private
        // storage to bypass `AlignedMappingProperty.__set__`'s eager
        // validation. See [`to_anndata_with_layers`].
        if let Some(m) = lazy_obsp {
            adata.setattr("_obsp", m.into_pyobject(py)?)?;
        }
        if let Some(m) = lazy_varp {
            adata.setattr("_varp", m.into_pyobject(py)?)?;
        }
        if let Some(m) = lazy_varm {
            adata.setattr("_varm", m.into_pyobject(py)?)?;
        }
        // Backed dense row-gather obsm (only built when obsm was
        // selected, not eager, no obs_filter). Attach via `_obsm` to
        // bypass AnnData's axis-length validation, like the other
        // bridges.
        if let Some(m) = lazy_obsm {
            adata.setattr("_obsm", m.into_pyobject(py)?)?;
        }
    }

    Ok(adata)
}

/// Compute `kept_to_global` mapping from deletion vectors.
///
/// Returns `None` if there are no deletions. Otherwise returns a Vec
/// where `kept_to_global[i]` is the global (file-level) row index for
/// user-visible row `i`.
fn compute_kept_to_global(reader: &ScxReader) -> PyResult<Option<Vec<u64>>> {
    let dv_opt = reader.read_deletion_vectors().map_err(to_pyerr)?;
    let dv = match dv_opt {
        Some(dv) if dv.total_deleted() > 0 => dv,
        _ => return Ok(None),
    };

    let n_obs = reader.n_obs() as usize;
    let shards = reader.catalog().shards_sorted();

    // Build a deleted-rows set
    let mut deleted = vec![false; n_obs];
    for (shard_idx, shard_entry) in shards.iter().enumerate() {
        if let Some(ref stats) = shard_entry.stats {
            if let Some(bitmap) = dv.shards.get(&(shard_idx as u32)) {
                for local_row in bitmap.iter() {
                    let global_row = stats.row_start + local_row as u64;
                    if (global_row as usize) < n_obs {
                        deleted[global_row as usize] = true;
                    }
                }
            }
        }
    }

    // Build mapping: user-visible row i → global row
    let kept: Vec<u64> = (0..n_obs)
        .filter(|&i| !deleted[i])
        .map(|i| i as u64)
        .collect();

    Ok(Some(kept))
}

/// Filter an obs RecordBatch to exclude deleted rows.
///
/// Builds a boolean keep-mask from the deletion vectors (same logic
/// as `read_all_csr_shards_filtered`) and applies
/// `arrow::compute::filter_record_batch`.
pub(crate) fn filter_obs_by_deletion_vectors(
    reader: &ScxReader,
    obs: arrow::array::RecordBatch,
) -> PyResult<arrow::array::RecordBatch> {
    let dv_opt = reader.read_deletion_vectors().map_err(to_pyerr)?;
    let dv = match dv_opt {
        Some(dv) if dv.total_deleted() > 0 => dv,
        _ => return Ok(obs), // No deletions — return as-is
    };

    let n_obs = obs.num_rows();
    let shards = reader.catalog().shards_sorted();

    // Build keep mask (same logic as reader.read_all_csr_shards_filtered)
    let mut keep = vec![true; n_obs];
    for (shard_idx, shard_entry) in shards.iter().enumerate() {
        if let Some(ref stats) = shard_entry.stats {
            if let Some(bitmap) = dv.shards.get(&(shard_idx as u32)) {
                for local_row in bitmap.iter() {
                    let global_row = stats.row_start + local_row as u64;
                    if (global_row as usize) < n_obs {
                        keep[global_row as usize] = false;
                    }
                }
            }
        }
    }

    let bool_array = arrow::array::BooleanArray::from(keep);
    arrow::compute::filter_record_batch(&obs, &bool_array)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to filter obs: {}", e)))
}

// ---------------------------------------------------------------------------
// from_anndata: AnnData → SCX
// ---------------------------------------------------------------------------

// Detection lives in `scx_codec::value_encoding` — re-exported here under
// the historical name so call sites elsewhere in `pyscx/` don't have to
// change.
pub(crate) use scx_codec::value_encoding::detect_value_encoding;

// ---------------------------------------------------------------------------
// Type conversion helpers (D2)
// ---------------------------------------------------------------------------

/// Convert i64 slice to Vec<u64> with overflow check.
///
/// Returns PyValueError if any element is negative.
#[allow(dead_code)]
pub(crate) fn i64_to_u64(v: &[i64]) -> PyResult<Vec<u64>> {
    v.iter()
        .map(|&val| {
            if val < 0 {
                Err(PyRuntimeError::new_err(format!(
                    "negative value {} cannot be converted to u64",
                    val
                )))
            } else {
                Ok(val as u64)
            }
        })
        .collect()
}

/// Convert i32 slice to Vec<u32> with overflow check.
///
/// Returns PyValueError if any element is negative.
#[allow(dead_code)]
pub(crate) fn i32_to_u32(v: &[i32]) -> PyResult<Vec<u32>> {
    v.iter()
        .map(|&val| {
            if val < 0 {
                Err(PyRuntimeError::new_err(format!(
                    "negative value {} cannot be converted to u32",
                    val
                )))
            } else {
                Ok(val as u32)
            }
        })
        .collect()
}

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

/// Parse codec name string to Option<CodecId>.
/// Returns None for auto mode (default), Some(id) for explicit codec.
pub(crate) fn parse_codec(codec: Option<&str>) -> PyResult<Option<CodecId>> {
    match codec {
        None | Some("auto") => Ok(None),
        Some("none") => Ok(Some(CodecId::None)),
        Some("scx1") => Ok(Some(CodecId::Scx1)),
        Some("zstd") => Ok(Some(CodecId::Zstd)),
        Some("lz4") => Ok(Some(CodecId::Lz4Shuffle)),
        Some("pcodec") => Ok(Some(CodecId::Pcodec)),
        Some(other) => Err(PyRuntimeError::new_err(format!(
            "Unknown codec: '{}'. Use 'auto', 'none', 'scx1', 'zstd', 'lz4', or 'pcodec'.",
            other
        ))),
    }
}

/// Convert a pandas DataFrame to an Arrow RecordBatch via pyarrow IPC.
pub(crate) fn pandas_to_record_batch(
    py: Python<'_>,
    df: &Bound<'_, PyAny>,
) -> PyResult<RecordBatch> {
    let pa = py.import("pyarrow")?;
    let table_cls = pa.getattr("Table")?;
    let table = table_cls.call_method1("from_pandas", (df,))?;

    // Serialize to IPC bytes
    let sink_cls = pa.getattr("BufferOutputStream")?;
    let sink = sink_cls.call0()?;
    let ipc = pa.getattr("ipc")?;
    let schema = table.getattr("schema")?;
    let writer = ipc.call_method1("new_file", (&sink, &schema))?;
    writer.call_method1("write_table", (&table,))?;
    writer.call_method0("close")?;
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
    scx_format::downcast_large_types(&batch).map_err(|e| PyRuntimeError::new_err(e.to_string()))
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

    let np = py.import("numpy")?;
    let np_ndarray = np.getattr("ndarray")?;
    let is_ndarray: bool = arr.is_instance(&np_ndarray)?;
    if !is_ndarray {
        let pd = py.import("pandas")?;
        let df = pd.call_method1("DataFrame", (arr,))?;
        return pandas_to_record_batch(py, &df);
    }
    let ndim: usize = arr.getattr("ndim")?.extract()?;
    if ndim != 2 {
        let pd = py.import("pandas")?;
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
            let pd = py.import("pandas")?;
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
fn estimate_dense_bytes(batch: &RecordBatch) -> u64 {
    batch
        .columns()
        .iter()
        .map(|c| (c.len() as u64) * (data_type_byte_width(c.data_type()) as u64))
        .sum()
}

fn data_type_byte_width(dt: &arrow::datatypes::DataType) -> usize {
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
fn estimate_coo_bytes(batch: &RecordBatch) -> u64 {
    (batch.num_rows() as u64) * 12
}

/// Forward a [`scx_convert::ConvertWarning`] to Python's `warnings.warn`
/// as a `UserWarning`, with the category name prefixed so consumers
/// can filter on it. Mirrors the per-key warning surface used by the
/// streaming pipeline; the in-memory `from_anndata` path doesn't wire
/// a [`WarningSink`] today so we emit live instead of summarising.
fn warn_python_convert(py: Python<'_>, w: &scx_convert::ConvertWarning) -> PyResult<()> {
    let warnings_mod = py.import("warnings")?;
    let user_warning = py.import("builtins")?.getattr("UserWarning")?;
    let msg = format!("{}: {}", w.category(), w);
    warnings_mod.getattr("warn")?.call1((msg, user_warning))?;
    Ok(())
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
    let scipy_sparse = py.import("scipy.sparse")?;
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
    let dtype_name: String = arr.getattr("dtype")?.getattr("name")?.extract()?;
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
struct UnsWriteCtx<'a, 'py> {
    format: UnsFormat,
    np_generic: &'a Bound<'py, PyAny>,
    np_ndarray: &'a Bound<'py, PyAny>,
    pd_lazy: Option<Bound<'py, PyModule>>,
    visiting: HashSet<usize>,
}

impl<'a, 'py> UnsWriteCtx<'a, 'py> {
    fn new(
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
fn is_numeric_kind(kind: &str) -> bool {
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
fn normalize_uns_value<'py>(
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
fn normalize_container<'py>(
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
fn encode_np_scalar_tagged<'py>(
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
fn encode_ndarray_tagged<'py>(
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
fn pylist_to_string_json_array<'py>(
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
fn pytuple_descr_to_json<'py>(
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
fn ndarray_bytes_le<'py>(arr: &Bound<'py, PyAny>, key_path: &str) -> PyResult<Vec<u8>> {
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
fn encode_pandas_tagged<'py>(
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
fn pyobj_to_simple_json<'py>(
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
struct UnsReadCtx<'py> {
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
fn json_to_py<'py>(
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
fn json_object_to_py<'py>(
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

fn build_plain_dict<'py>(
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
fn envelope_required_keys(tag: &str) -> Option<&'static [&'static str]> {
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

fn json_map_has_all_keys(map: &serde_json::Map<String, serde_json::Value>, keys: &[&str]) -> bool {
    keys.iter().all(|k| map.contains_key(*k))
}

fn decode_tagged_envelope<'py>(
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

fn require_str_json<'a>(
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

fn require_value_json<'a>(
    map: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> PyResult<&'a serde_json::Value> {
    map.get(key)
        .ok_or_else(|| PyValueError::new_err(format!("uns envelope missing key '{key}'")))
}

fn decode_base64_bytes_json<'py>(
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

fn extract_shape_json(map: &serde_json::Map<String, serde_json::Value>) -> PyResult<Vec<usize>> {
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

fn decode_ndarray_envelope<'py>(
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

fn decode_scalar_envelope<'py>(
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

fn decode_tuple_envelope<'py>(
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

fn decode_recarray_envelope<'py>(
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
fn build_structured_dtype_from_json<'py>(
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

fn decode_categorical_envelope<'py>(
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

fn decode_pandas_index_envelope<'py>(
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

fn decode_pandas_series_envelope<'py>(
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
fn read_uns_as_pyobject<'py>(
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

/// Shard boundary computed sequentially before parallel encoding.
struct ShardBoundary {
    row_start: usize,
    row_end: usize,
    nnz_start: usize,
    nnz_end: usize,
    indptr_base: i64,
    shard_idx: u32,
}

/// Parallel-encode CSR shards using rayon.
///
/// Clones the numpy-borrowed arrays into Rust-owned `Arc` slices for thread
/// safety, then encodes all shards in parallel under `py.detach()`.
/// Returns `PreEncodedSection`s in shard order, ready for sequential write.
#[allow(clippy::too_many_arguments)]
fn parallel_encode_csr_shards(
    py: Python<'_>,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    boundaries: &[ShardBoundary],
    csr_validated: bool,
    explicit_codec: Option<CodecId>,
    index_dtype: u8,
    n_vars: u32,
    section_type: SectionType,
    name_prefix: &str,
) -> PyResult<Vec<PreEncodedSection>> {
    if boundaries.is_empty() {
        return Ok(Vec::new());
    }

    // Clone into Rust-owned Arc slices for Send + Sync across rayon threads.
    let indptr_owned: Arc<[i64]> = indptr.to_vec().into();
    let indices_owned: Arc<[i32]> = indices.to_vec().into();
    let data_owned: Arc<[f32]> = data.to_vec().into();
    let name_prefix = name_prefix.to_string();

    let result: Result<Vec<PreEncodedSection>, String> = py.detach(|| {
        boundaries
            .par_iter()
            .map(|b| {
                // 1. Rebase indptr for this shard
                let mut shard_indptr: Vec<u64> = if csr_validated {
                    indptr_owned[b.row_start..=b.row_end]
                        .iter()
                        .map(|&v| (v - b.indptr_base) as u64)
                        .collect()
                } else {
                    indptr_owned[b.row_start..=b.row_end]
                        .iter()
                        .map(|&v| {
                            if v < b.indptr_base {
                                Err(format!(
                                    "indptr value {v} < base {} (non-monotonic)",
                                    b.indptr_base
                                ))
                            } else {
                                Ok((v - b.indptr_base) as u64)
                            }
                        })
                        .collect::<Result<Vec<u64>, String>>()?
                };

                // 2. Convert indices i32 → u32
                let mut shard_indices: Vec<u32> = if csr_validated {
                    indices_owned[b.nnz_start..b.nnz_end]
                        .iter()
                        .map(|&v| v as u32)
                        .collect()
                } else {
                    indices_owned[b.nnz_start..b.nnz_end]
                        .iter()
                        .map(|&v| {
                            if v < 0 {
                                Err(format!("negative CSR index {v}"))
                            } else {
                                Ok(v as u32)
                            }
                        })
                        .collect::<Result<Vec<u32>, String>>()?
                };

                let data_borrow = &data_owned[b.nnz_start..b.nnz_end];
                let name = format!("{name_prefix}_shard_{}", b.shard_idx);

                // Skip the per-shard f32 copy when the source is already
                // canonical (the common case); only materialize + canonicalize
                // a genuinely non-canonical shard.
                let encode = |indptr: &[u64], indices: &[u32], data: &[f32]| {
                    scx_format::encode_one_shard(
                        indptr,
                        indices,
                        data,
                        explicit_codec,
                        index_dtype,
                        n_vars,
                        b.row_start as u64,
                        section_type,
                        ModalityType::Rna,
                        name.clone(),
                    )
                    .map_err(|e| e.to_string())
                };
                if scx_sparse::is_canonical_csr(&shard_indptr, &shard_indices, data_borrow) {
                    encode(&shard_indptr, &shard_indices, data_borrow)
                } else {
                    let mut shard_data = data_borrow.to_vec();
                    canonicalize_csr(&mut shard_indptr, &mut shard_indices, &mut shard_data);
                    encode(&shard_indptr, &shard_indices, &shard_data)
                }
            })
            .collect()
    });

    result.map_err(PyRuntimeError::new_err)
}

/// Forward each non-empty category in `sink` as a single
/// `warnings.warn(..., UserWarning)` call on the Python side.
///
/// The conversion itself runs under `py.detach`, so emission
/// happens after the GIL is reacquired. One Python-side warning per
/// category (with its aggregate count) is enough for Phase 0; per-
/// emission forwarding would require holding the GIL across the
/// whole conversion.
#[cfg(feature = "hdf5")]
pub(crate) fn emit_python_warnings(
    py: Python<'_>,
    sink: &scx_convert::WarningSink,
) -> PyResult<()> {
    if sink.total() == 0 {
        return Ok(());
    }
    let warnings_mod = py.import("warnings")?;
    let warn = warnings_mod.getattr("warn")?;
    let user_warning = py.import("builtins")?.getattr("UserWarning")?;
    for (cat, count) in sink.counts() {
        let msg = format!("scx conversion: {count} warning(s) of type '{cat}'");
        warn.call1((msg, user_warning.clone()))?;
    }
    Ok(())
}

/// Route a backed AnnData object through the streaming converter.
/// Extracts in-memory `obs` / `var` / `uns` / `obsm` / `varm` /
/// `obsp` / `varp` into Rust types so any caller mutations are
/// preserved, then invokes `scx_convert::h5ad_to_scx_streaming` on
/// the backing h5ad file.
///
/// X and layers always come from disk via streaming — there's no
/// override hook for those (they're potentially too large to extract
/// from a backed AnnData into memory). Emits a `UserWarning` when
/// the backed AnnData has any layers, because the streaming reads
/// will overwrite any in-memory layer mutations.
#[cfg(feature = "hdf5")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn route_backed_anndata_to_streaming(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    path: &str,
    explicit_codec: Option<CodecId>,
    shard_target_rows: u32,
    csc_policy: scx_format::CscPolicy,
    csc_cols_per_shard: usize,
    uns_format_parsed: UnsFormat,
    stream: bool,
    strict_uns: bool,
    dense_zero_epsilon: f32,
    memory_budget: Option<u64>,
    temp_dir: Option<&str>,
    index_obs: Vec<String>,
    index_var: Vec<String>,
    index_preset: Option<String>,
    index_auto_threshold: usize,
    bitmap: &str,
    reader_threads: Option<usize>,
    writer_queue_depth: usize,
) -> PyResult<()> {
    let bitmap_policy = scx_format::BitmapPolicy::parse(bitmap)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    // Resolve the on-disk h5ad path. `anndata` 0.12 exposes both
    // `adata.filename` (preferred) and `adata.file.filename` (older
    // name); we try both. Recent anndata returns `pathlib.PosixPath`
    // rather than a bare `str`, so go through Python's `str(...)` —
    // it's a no-op on `str` and stringifies `Path` cleanly.
    fn fspath_str(v: &Bound<'_, PyAny>) -> Option<String> {
        if v.is_none() {
            return None;
        }
        v.str()
            .ok()
            .and_then(|s| s.extract::<String>().ok())
            .filter(|s| !s.is_empty())
    }
    let filename: String = adata
        .getattr("filename")
        .ok()
        .and_then(|v| fspath_str(&v))
        .or_else(|| {
            adata
                .getattr("file")
                .ok()
                .and_then(|f| f.getattr("filename").ok())
                .and_then(|v| fspath_str(&v))
        })
        .unwrap_or_default();
    if filename.is_empty() || !std::path::Path::new(&filename).exists() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "backed AnnData has no resolvable h5ad filename; use \
             pyscx.from_h5ad(path, out) or convert to a non-backed \
             AnnData first",
        ));
    }

    // Build the overrides from the in-memory AnnData. `obs` / `var`
    // are always extracted from Python — the pandas → Arrow conversion
    // carries the categorical and nullable-encoding metadata that
    // scx-convert's pure-Rust `read_dataframe_group` would lose.
    // `obsm` / `varm` / `obsp` / `varp` / `uns` are extracted only when
    // mutation detection sees a divergence from the on-disk h5ad —
    // otherwise the streaming pipeline reads them from disk one shard
    // at a time. This avoids the per-shard OOM that the wholesale
    // Python extraction causes for inputs with large embeddings
    // (e.g. Parse-PBMC obsm reaching tens of GB).
    let obs_override = pandas_to_record_batch(py, &adata.getattr("obs")?)?;
    let var_override = pandas_to_record_batch(py, &adata.getattr("var")?)?;

    // Open the source h5ad through h5py. We only ever read group
    // `.keys()` — never any dataset value — so anndata's lazy
    // `obsm[key]` materialisation path stays untriggered.
    let h5py = py.import("h5py")?;
    let h5_kwargs = pyo3::types::PyDict::new(py);
    h5_kwargs.set_item("mode", "r")?;
    let h5_file = h5py.call_method("File", (&filename,), Some(&h5_kwargs))?;

    let obsm_clean = section_keys_match(py, &h5_file, adata, "obsm")?;
    let varm_clean = section_keys_match(py, &h5_file, adata, "varm")?;
    let obsp_clean = section_keys_match(py, &h5_file, adata, "obsp")?;
    let varp_clean = section_keys_match(py, &h5_file, adata, "varp")?;
    let uns_clean = section_keys_match(py, &h5_file, adata, "uns")?;

    let obsm_override = if obsm_clean {
        None
    } else {
        Some(extract_dense_mapping(py, adata, "obsm")?)
    };
    let varm_override = if varm_clean {
        None
    } else {
        Some(extract_dense_mapping(py, adata, "varm")?)
    };
    let obsp_override = if obsp_clean {
        None
    } else {
        Some(extract_coo_mapping(py, adata, "obsp")?)
    };
    let varp_override = if varp_clean {
        None
    } else {
        Some(extract_coo_mapping(py, adata, "varp")?)
    };
    let uns_override = if uns_clean {
        None
    } else {
        extract_uns_value(py, adata, uns_format_parsed)?
    };

    // Close the h5py file handle before scx-convert opens the same path
    // via the Rust `hdf5` crate. Concurrent libhdf5 access from h5py
    // and the Rust crate on a single file isn't documented as safe.
    let _ = h5_file.call_method0("close");

    // Layer mutations on a backed AnnData are not propagated — the
    // streaming pipeline always reads layers from disk. Warn so the
    // user knows.
    if let Ok(layers) = adata.getattr("layers") {
        if let Ok(len_val) = layers.call_method0("__len__") {
            if let Ok(len) = len_val.extract::<usize>() {
                if len > 0 {
                    let msg = format!(
                        "backed AnnData has {len} layer(s); layer data will be read \
                         from the on-disk h5ad file. Any in-memory layer mutations \
                         will be lost. Use pyscx.from_h5ad(path, out) on a \
                         freshly-written h5ad if you need mutated layers preserved.",
                    );
                    let _ = py
                        .import("warnings")
                        .and_then(|w| w.call_method1("warn", (msg,)));
                }
            }
        }
    }

    // Breadcrumb for the corner case where a user replaced a value
    // under an existing key (the keys-only heuristic can't catch this).
    let routed_from_disk: Vec<&str> = [
        ("obsm", obsm_clean),
        ("varm", varm_clean),
        ("obsp", obsp_clean),
        ("varp", varp_clean),
        ("uns", uns_clean),
    ]
    .into_iter()
    .filter_map(|(name, clean)| if clean { Some(name) } else { None })
    .collect();
    if !routed_from_disk.is_empty() {
        let msg = format!(
            "pyscx: streaming {} from the on-disk h5ad (no Python-side mutation \
             detected via top-level key comparison). If you replaced a value \
             under an existing key in-place, the on-disk version wins; re-add \
             the key under a fresh name to force the Python value through.",
            routed_from_disk.join(", ")
        );
        let _ = py
            .import("warnings")
            .and_then(|w| w.call_method1("warn", (msg,)));
    }

    let overrides = scx_convert::StreamingOverrides {
        obs: Some(obs_override),
        var: Some(var_override),
        uns: uns_override,
        obsm: obsm_override,
        varm: varm_override,
        obsp: obsp_override,
        varp: varp_override,
    };

    let opts = scx_convert::ConvertOptions {
        shard_target_rows,
        codec: explicit_codec,
        csc: csc_policy,
        csc_cols_per_shard,
        tool: "pyscx".into(),
        memory_budget,
        stream,
        strict_uns,
        dense_zero_epsilon,
        temp_dir: temp_dir.map(std::path::PathBuf::from),
        modalities: None,
        modality_types: Vec::new(),
        index_obs,
        index_var,
        index_preset,
        index_auto_threshold,
        bitmap: bitmap_policy,
        reader_threads,
        writer_queue_depth,
    };
    let input = std::path::PathBuf::from(filename);
    let output = std::path::PathBuf::from(path);

    // Honour `stream=false` by routing to the non-streaming
    // `h5ad_to_scx` path. The backed-AnnData overrides for obs / var /
    // uns / obsm / varm / obsp / varp are dropped on this path —
    // `from_anndata` (non-streaming) is the canonical caller when
    // those mutations need preserving.
    let mut sink = scx_convert::WarningSink::log();
    if stream {
        py.detach(|| {
            scx_convert::h5ad_to_scx_streaming(&input, &output, &opts, &overrides, &mut sink)
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    } else {
        py.detach(|| scx_convert::h5ad_to_scx(&input, &output, &opts, &mut sink))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    }
    emit_python_warnings(py, &sink)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 8b: SCX → SCX streaming writer (backed and lazy `X`).
// ---------------------------------------------------------------------------

/// Extracted metadata + uns JSON for an AnnData object that wraps an
/// SCX-backed or lazy `X`. Returned by `extract_scx_overrides` so the
/// route functions below can interleave metadata writes with shard
/// reads.
struct ScxOverrides {
    obs: RecordBatch,
    var: RecordBatch,
    obsm: Vec<(String, RecordBatch)>,
    varm: Vec<(String, RecordBatch)>,
    obsp: Vec<(String, RecordBatch)>,
    varp: Vec<(String, RecordBatch)>,
    uns: Option<serde_json::Value>,
}

fn extract_scx_overrides(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    uns_format_parsed: UnsFormat,
) -> PyResult<ScxOverrides> {
    let obs = pandas_to_record_batch(py, &adata.getattr("obs")?)?;
    let var = pandas_to_record_batch(py, &adata.getattr("var")?)?;
    let obsm = extract_dense_mapping(py, adata, "obsm")?;
    let varm = extract_dense_mapping(py, adata, "varm")?;
    let obsp = extract_coo_mapping(py, adata, "obsp")?;
    let varp = extract_coo_mapping(py, adata, "varp")?;
    let uns = extract_uns_value(py, adata, uns_format_parsed)?;
    Ok(ScxOverrides {
        obs,
        var,
        obsm,
        varm,
        obsp,
        varp,
        uns,
    })
}

/// Warn (Python `UserWarning`) that a CSC sidecar on the source is
/// being dropped on rewrite. Matches the convention documented in
/// `AGENTS.md`'s "CSC storage" bullet — mutating ops drop the
/// sidecar by default; callers opt into a rebuild via `csc="always"`.
fn warn_csc_dropped(py: Python<'_>) {
    let msg = "source SCX has a CSC sidecar; the rewrite drops it. \
               Pass csc=\"always\" to rebuild a fresh CSC sidecar over the new CSR shards.";
    let _ = py
        .import("warnings")
        .and_then(|w| w.call_method1("warn", (msg,)));
}

/// Decompose a scipy CSR matrix, canonicalize it, and invoke `f` with
/// slices suitable for `encode_one_shard`. `indptr`, `indices`, and
/// `data` are owned because v3 writers must sort rows, sum duplicate
/// coordinates, and drop explicit zeros before encoding.
fn decompose_scipy_csr_with<F, R>(py: Python<'_>, csr: &Bound<'_, PyAny>, f: F) -> PyResult<R>
where
    F: FnOnce(&[u64], &[u32], &[f32]) -> PyResult<R>,
{
    let np = py.import("numpy")?;

    let indptr_obj = csr.getattr("indptr")?;
    let indptr_arr = astype_if_needed(&indptr_obj, &np, "int64")?;
    let indptr: PyReadonlyArray1<'_, i64> = indptr_arr.extract()?;
    let indptr_slice = indptr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let mut indptr_u64: Vec<u64> = indptr_slice
        .iter()
        .map(|&v| {
            if v < 0 {
                Err(PyRuntimeError::new_err(format!(
                    "negative indptr value {v} from wrapper.__getitem__"
                )))
            } else {
                Ok(v as u64)
            }
        })
        .collect::<PyResult<Vec<u64>>>()?;

    let indices_obj = csr.getattr("indices")?;
    let indices_arr = astype_if_needed(&indices_obj, &np, "int32")?;
    let indices: PyReadonlyArray1<'_, i32> = indices_arr.extract()?;
    let indices_slice = indices
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let mut indices_u32: Vec<u32> = indices_slice
        .iter()
        .map(|&v| {
            if v < 0 {
                Err(PyRuntimeError::new_err(format!(
                    "negative column index {v} from wrapper.__getitem__"
                )))
            } else {
                Ok(v as u32)
            }
        })
        .collect::<PyResult<Vec<u32>>>()?;

    let data_obj = csr.getattr("data")?;
    let data_arr = astype_if_needed(&data_obj, &np, "float32")?;
    let data: PyReadonlyArray1<'_, f32> = data_arr.extract()?;
    let data_slice = data
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Fast path: scipy CSR with `has_canonical_format == True` (the common
    // case) needs no sort/dedup/zero-drop, so pass the borrowed numpy buffer
    // through with no f32 copy. Only materialize + canonicalize when the input
    // is actually non-canonical.
    if scx_sparse::is_canonical_csr(&indptr_u64, &indices_u32, data_slice) {
        f(&indptr_u64, &indices_u32, data_slice)
    } else {
        let mut data_vec = data_slice.to_vec();
        canonicalize_csr(&mut indptr_u64, &mut indices_u32, &mut data_vec);
        f(&indptr_u64, &indices_u32, &data_vec)
    }
}

/// Build a fresh `FileHeader` template for an SCX → SCX rewrite.
/// Catalog offsets, shard counts, and `nnz` are written by
/// `ScxWriter::finish()`.
///
/// `source_format_version` is the source SCX's `format_version`. Because
/// passthrough / per-shard re-encode does **not** re-canonicalize, the v3
/// canonical-CSR invariant may only be claimed when the source already
/// guarantees it (is itself v3+). `rewrite_output_format_version` gates the
/// stamp so a pre-v3 source is never silently upgraded to a false v3 claim.
fn build_output_header(
    n_obs: u64,
    n_vars: u64,
    shard_target_rows: u32,
    codec: CodecId,
    index_dtype: u8,
    source_format_version: u16,
) -> FileHeader {
    FileHeader {
        magic: MAGIC,
        // Single-modality output → feature floor 1.
        format_version: scx_format::rewrite_output_format_version(&[source_format_version], 1),
        header_length: 256,
        flags: 0,
        n_obs,
        n_vars,
        nnz: 0,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows,
        codec_id: codec as u8,
        index_dtype,
        endian: 0,
        reserved_padding: 0,
        root_catalog_offset: 0,
        root_catalog_length: 0,
        full_catalog_offset: 0,
        full_catalog_length: 0,
        manifest_sequence: 1,
        prev_catalog_offset: 0,
        file_checksum: 0,
        front_catalog_offset: 0,
        front_catalog_length: 0,
        n_modalities: 0,
        modality_table_offset: 0,
        modality_table_length: 0,
        reserved: [0u8; 112],
    }
}

/// Route an AnnData with `adata.X = ScxBackedSparseDataset` through
/// an SCX → SCX streaming writer.
///
/// Two modes:
///
/// * **Byte-passthrough**: when the source and target shard layouts
///   agree (same `shard_target_rows`, same codec, no row deletions,
///   no column projection, source is built from a single modality
///   with `modality_id == None`), pre-encoded CSR shards are copied
///   from the source file into the target writer via
///   [`ScxWriter::copy_section_verbatim`]. No decode + re-encode.
/// * **Decode + encode**: otherwise the writer iterates the
///   wrapper's user-visible shard boundaries, calls
///   `wrapper[start:end]` to materialise each shard as a scipy CSR
///   (deletions and column projection already applied by the
///   wrapper), then hands it to [`scx_format::encode_one_shard`]
///   plus [`ScxWriter::write_preencoded_shard`].
///
/// `shard_size_overridden` reflects whether the caller passed
/// `shard_size=…`. When `explicit_codec` is `None` and `shard_size`
/// was not overridden *and* every other precondition holds, the
/// rewrite is byte-faithful; otherwise it falls back to
/// decode + encode.
#[allow(clippy::too_many_arguments)]
fn route_scx_backed_to_scx(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    backed: &crate::backed::ScxBackedSparseDataset,
    out_path: &str,
    explicit_codec: Option<CodecId>,
    shard_target_rows: u32,
    csc_policy: scx_format::CscPolicy,
    csc_cols_per_shard: usize,
    uns_format_parsed: UnsFormat,
    shard_size_overridden: bool,
) -> PyResult<()> {
    let src_path = backed.source_path().ok_or_else(|| {
        pyo3::exceptions::PyNotImplementedError::new_err(
            "ScxBackedSparseDataset has no known source path; the SCX → SCX writer needs a \
             path-backed wrapper (constructed via pyscx.open(path).to_anndata(backed=True)).",
        )
    })?;
    let src_path_owned = src_path.to_path_buf();

    let src_reader = ScxReader::open(&src_path_owned).map_err(to_pyerr)?;
    let src_header = src_reader.header().clone();
    let src_n_obs = src_header.n_obs;
    let src_n_vars = src_header.n_vars;
    let src_codec_id = src_header.codec_id;
    let src_shard_rows = src_header.shard_target_rows;
    let src_has_csc = src_header.has_csc();
    let modality_id = backed.modality_id;

    // User-visible dimensions after any row deletions / column
    // projection. Used by the decode-encode path's output header so
    // the new file's shape matches what the AnnData wrapper exposes.
    let (out_n_obs_visible, out_n_vars_visible) =
        (backed.shape_val.0 as u64, backed.shape_val.1 as u64);

    // Passthrough preconditions. Any false → fall through to
    // decode-encode.
    let target_codec_for_passthrough = match explicit_codec {
        Some(c) => c as u8 == src_codec_id,
        None => true,
    };
    let target_shard_rows_matches = if shard_size_overridden {
        shard_target_rows == src_shard_rows
    } else {
        true
    };
    let no_deletions = backed.kept_to_global.is_none();
    let no_projection = backed.col_projection().is_none();
    let single_modality_source = modality_id.is_none();
    let passthrough_ok = target_codec_for_passthrough
        && target_shard_rows_matches
        && no_deletions
        && no_projection
        && single_modality_source;

    // Output header / writer setup. For passthrough, mirror the
    // source's codec / shard_target_rows / index_dtype so the
    // catalog and per-shard headers stay self-consistent. On the
    // decode-encode fallback, default to the source's codec choice
    // (preserves Scx1/Pcodec/etc — only override when the caller
    // passes `codec=`).
    let out_codec_id: u8 = if passthrough_ok {
        src_codec_id
    } else {
        match explicit_codec {
            Some(c) => c as u8,
            None => src_codec_id,
        }
    };
    let out_shard_rows = if passthrough_ok {
        src_shard_rows
    } else {
        shard_target_rows
    };
    // Key index_dtype off the user-visible n_vars (after any column
    // projection), not the source's. Allows u16 when projecting a
    // large source down to a small gene subset, and matches the
    // in-memory path which sees only the visible shape.
    let out_index_dtype = if passthrough_ok {
        src_header.index_dtype
    } else if out_n_vars_visible <= 65535 {
        0
    } else {
        1
    };

    let codec_for_header = CodecId::from_u8(out_codec_id).ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "unknown codec id {out_codec_id} from source SCX header"
        ))
    })?;
    // Output dimensions: passthrough mirrors the source header
    // (preconditions guarantee no deletions / projection). The
    // decode-encode path uses the wrapper's user-visible shape so
    // the rewrite drops any deleted rows and respects column
    // projection.
    let (out_n_obs, out_n_vars) = if passthrough_ok {
        (src_n_obs, src_n_vars)
    } else {
        (out_n_obs_visible, out_n_vars_visible)
    };
    let header = build_output_header(
        out_n_obs,
        out_n_vars,
        out_shard_rows,
        codec_for_header,
        out_index_dtype,
        src_header.format_version,
    );

    let mut writer = ScxWriter::new(out_path, header).map_err(to_pyerr)?;

    // Extract metadata overrides up front so the writer can interleave
    // metadata writes with shard I/O in the canonical order.
    let ov = extract_scx_overrides(py, adata, uns_format_parsed)?;

    // CSC sidecar policy. Resolve `Auto` against the output shape so a
    // CSC sidecar is (re)built only when the rewritten dataset is large
    // enough to benefit.
    let csc_build = csc_policy.should_build_csc(out_n_obs, out_n_vars);
    let csc_dropped = src_has_csc && !csc_build;
    if csc_dropped {
        warn_csc_dropped(py);
    }

    // Write obs/var first.
    py.detach(|| -> Result<(), scx_format::ScxError> {
        writer.write_obs(&ov.obs)?;
        writer.write_var(&ov.var)?;
        Ok(())
    })
    .map_err(to_pyerr)?;

    let n_vars_u32 = u32::try_from(out_n_vars)
        .map_err(|_| PyRuntimeError::new_err(format!("n_vars {out_n_vars} exceeds u32::MAX")))?;
    // Pin the per-shard encode codec to the header's codec so the
    // recorded `codec_id` and the actual shard encodings stay
    // consistent. When the user passed an explicit codec we use it;
    // otherwise we use the source's codec (which `out_codec_id` now
    // mirrors).
    let codec_for_encode = Some(codec_for_header);

    if passthrough_ok {
        // Byte-passthrough. Iterate source CSR shards in row order;
        // copy each verbatim. `modality_id == None` is already
        // enforced above, so we can write at the global modality
        // (current_modality_id == 0).
        let csr_shards = src_reader.catalog().csr_shards_sorted();
        py.detach(|| -> Result<(), scx_format::ScxError> {
            for entry in csr_shards {
                let bytes = src_reader.read_raw_shard_bytes(entry)?;
                writer.copy_section_verbatim(entry, bytes)?;
            }
            Ok(())
        })
        .map_err(to_pyerr)?;
    } else {
        // Decode + encode. Drive iteration over user-visible shard
        // boundaries so deletions / column projection already apply
        // via the wrapper's `__getitem__`.
        let bounds = compute_wrapper_boundaries_backed(backed, out_shard_rows);
        let adata_x = adata.getattr("X")?;
        for (i, (start, end)) in bounds.iter().enumerate() {
            let py_slice = pyo3::types::PySlice::new(py, *start as isize, *end as isize, 1);
            // Use the Python-visible wrapper to honour deletion /
            // projection semantics. Calling through PyAny gives us
            // the wrapper's __getitem__ (returns scipy CSR).
            let shard_obj = adata_x.call_method1("__getitem__", (py_slice,))?;
            let pre = decompose_scipy_csr_with(py, &shard_obj, |indptr, indices, data| {
                py.detach(|| {
                    scx_format::encode_one_shard(
                        indptr,
                        indices,
                        data,
                        codec_for_encode,
                        out_index_dtype,
                        n_vars_u32,
                        *start as u64,
                        SectionType::CsrShard,
                        ModalityType::Rna,
                        format!("X_shard_{i}"),
                    )
                })
                .map_err(to_pyerr)
            })?;
            py.detach(|| writer.write_preencoded_shard(pre))
                .map_err(to_pyerr)?;
        }
    }

    // Layers (decode-encode, never passthrough — keeps the byte
    // path bounded to X). `adata.layers` from a backed AnnData
    // contains `ScxBackedLayerDataset` instances which slice-via-
    // `__getitem__` exactly like X.
    stream_write_layers(
        py,
        adata,
        &mut writer,
        out_n_obs,
        out_n_vars,
        out_shard_rows,
        codec_for_encode,
        out_index_dtype,
    )?;

    // Write remaining metadata (obsm / varm / obsp / varp / uns) after
    // the X shards, matching the canonical layout. obsm / varm / obsp /
    // varp are emitted as row-sharded sections so the on-disk layout
    // matches what the streaming pipeline produces (readers handle
    // both sharded and legacy single-section layouts transparently).
    py.detach(|| -> Result<(), scx_format::ScxError> {
        for (k, b) in &ov.obsm {
            for_each_dense_shard(
                b,
                out_shard_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_obsm_shard(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        for (k, b) in &ov.varm {
            for_each_dense_shard(
                b,
                out_shard_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_varm_shard(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        for (k, b) in &ov.obsp {
            for_each_coo_shard(
                b,
                out_shard_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_obsp_shard_coo(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        for (k, b) in &ov.varp {
            for_each_coo_shard(
                b,
                out_shard_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_varp_shard_coo(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        if let Some(ref uns_json) = ov.uns {
            writer.write_uns(uns_json)?;
        }
        Ok(())
    })
    .map_err(to_pyerr)?;

    // Provenance: x_source / passthrough / source_path / csc_dropped.
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let params_json = serde_json::json!({
        "x_source": "backed",
        "passthrough": passthrough_ok,
        "source_path": src_path_owned.display().to_string(),
        "csc_dropped": csc_dropped,
    });
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp,
            action: "from_anndata".to_string(),
            tool: format!("pyscx {}", env!("CARGO_PKG_VERSION")),
            params_json: params_json.to_string(),
            input_checksums: vec![],
        }])
        .map_err(to_pyerr)?;

    writer.finish().map_err(to_pyerr)?;

    // Optional CSC sidecar rebuild over the just-written file.
    if csc_build {
        py.detach(|| {
            scx_ops::rebuild_csc_inplace(std::path::Path::new(out_path), csc_cols_per_shard, "4G")
                .map_err(|e| e.to_string())
        })
        .map_err(|e| PyRuntimeError::new_err(format!("rebuild_csc_inplace failed: {e}")))?;
    }
    Ok(())
}

/// Route an AnnData with `adata.X = ScxLazyTransformedDataset`
/// through an SCX → SCX streaming writer.
///
/// Always decode + encode. The wrapper's `__getitem__` returns the
/// transformed scipy CSR for the requested row slice; the writer
/// hands it to `encode_one_shard` and `write_preencoded_shard`.
/// Any source CSC sidecar is invalidated by the transforms and is
/// dropped with a `UserWarning` unless `csc="always"` is passed (in
/// which case it is rebuilt post-finalise).
#[allow(clippy::too_many_arguments)]
fn route_scx_lazy_to_scx(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    lazy: &crate::lazy_transform::ScxLazyTransformedDataset,
    out_path: &str,
    explicit_codec: Option<CodecId>,
    shard_target_rows: u32,
    csc_policy: scx_format::CscPolicy,
    csc_cols_per_shard: usize,
    uns_format_parsed: UnsFormat,
) -> PyResult<()> {
    let (n_obs_usize, n_vars_usize) = lazy.shape_val;
    let n_obs = n_obs_usize as u64;
    let n_vars = n_vars_usize as u64;
    // Resolve `Auto` against the (lazy) source shape.
    let csc_build = csc_policy.should_build_csc(n_obs, n_vars);

    // Default codec to the source SCX's choice when known (preserves
    // Scx1 / Pcodec / etc through the rewrite). Falls back to Zstd
    // when no source path is recorded (the lazy wrapper can in
    // principle be built without one).
    // Open the source once for both codec and format_version. The lazy
    // per-shard re-encode applies value transforms only (no column reorder),
    // so it preserves a canonical source but cannot canonicalize a pre-v3
    // one — gate the v3 stamp on the source version.
    let src_header_meta: Option<(Option<CodecId>, u16)> = lazy.source_path().and_then(|p| {
        ScxReader::open(p).ok().map(|r| {
            (
                CodecId::from_u8(r.header().codec_id),
                r.header().format_version,
            )
        })
    });
    let src_codec: Option<CodecId> = src_header_meta.and_then(|(c, _)| c);
    let src_format_version: u16 = src_header_meta
        .map(|(_, v)| v)
        .unwrap_or(scx_format::CURRENT_FORMAT_VERSION);
    let out_codec = explicit_codec.or(src_codec).unwrap_or(CodecId::Zstd);
    let index_dtype: u8 = if n_vars <= 65535 { 0 } else { 1 };
    let n_vars_u32 = u32::try_from(n_vars)
        .map_err(|_| PyRuntimeError::new_err(format!("n_vars {n_vars} exceeds u32::MAX")))?;

    let header = build_output_header(
        n_obs,
        n_vars,
        shard_target_rows,
        out_codec,
        index_dtype,
        src_format_version,
    );
    let mut writer = ScxWriter::new(out_path, header).map_err(to_pyerr)?;

    // Source CSC sidecar (if any) is always invalidated by the
    // transform chain. Warn unless the user opted into a rebuild.
    let src_has_csc = lazy.backed_csc.is_some();
    let csc_dropped = src_has_csc && !csc_build;
    if csc_dropped {
        warn_csc_dropped(py);
    }

    let ov = extract_scx_overrides(py, adata, uns_format_parsed)?;

    py.detach(|| -> Result<(), scx_format::ScxError> {
        writer.write_obs(&ov.obs)?;
        writer.write_var(&ov.var)?;
        Ok(())
    })
    .map_err(to_pyerr)?;

    // Lazy transforms always force decode + encode. Iterate the
    // wrapper's user-visible shard boundaries so any deletion vector
    // already applies. Pin per-shard encode codec to the header
    // codec so the recorded `codec_id` and the actual encodings
    // stay consistent.
    let codec_for_encode = Some(out_codec);
    let bounds = compute_wrapper_boundaries_lazy(lazy, shard_target_rows);
    let adata_x = adata.getattr("X")?;
    for (i, (start, end)) in bounds.iter().enumerate() {
        let py_slice = pyo3::types::PySlice::new(py, *start as isize, *end as isize, 1);
        let shard_obj = adata_x.call_method1("__getitem__", (py_slice,))?;
        let pre = decompose_scipy_csr_with(py, &shard_obj, |indptr, indices, data| {
            py.detach(|| {
                scx_format::encode_one_shard(
                    indptr,
                    indices,
                    data,
                    codec_for_encode,
                    index_dtype,
                    n_vars_u32,
                    *start as u64,
                    SectionType::CsrShard,
                    ModalityType::Rna,
                    format!("X_shard_{i}"),
                )
            })
            .map_err(to_pyerr)
        })?;
        py.detach(|| writer.write_preencoded_shard(pre))
            .map_err(to_pyerr)?;
    }

    // Layers — never transformed by the lazy X chain, so we just
    // stream them through the same decode-encode pipeline as X.
    stream_write_layers(
        py,
        adata,
        &mut writer,
        n_obs,
        n_vars,
        shard_target_rows,
        codec_for_encode,
        index_dtype,
    )?;

    py.detach(|| -> Result<(), scx_format::ScxError> {
        for (k, b) in &ov.obsm {
            for_each_dense_shard(
                b,
                shard_target_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_obsm_shard(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        for (k, b) in &ov.varm {
            for_each_dense_shard(
                b,
                shard_target_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_varm_shard(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        for (k, b) in &ov.obsp {
            for_each_coo_shard(
                b,
                shard_target_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_obsp_shard_coo(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        for (k, b) in &ov.varp {
            for_each_coo_shard(
                b,
                shard_target_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_varp_shard_coo(k, idx, row_start, n_shard_rows, n_total, shard)
                },
            )?;
        }
        if let Some(ref uns_json) = ov.uns {
            writer.write_uns(uns_json)?;
        }
        Ok(())
    })
    .map_err(to_pyerr)?;

    // Provenance: lazy_transforms summary.
    let transforms_repr: Vec<serde_json::Value> = lazy
        .transforms()
        .iter()
        .map(|t| match t {
            crate::lazy_transform::Transform::NormalizeTotal {
                row_sums,
                target_sum,
            } => serde_json::json!({
                "name": "normalize_total",
                "params": { "row_sums_len": row_sums.len(), "target_sum": target_sum },
            }),
            crate::lazy_transform::Transform::Log1p => serde_json::json!({
                "name": "log1p",
                "params": {},
            }),
            crate::lazy_transform::Transform::RowScale { factors } => serde_json::json!({
                "name": "row_scale",
                "params": { "factors_len": factors.len() },
            }),
        })
        .collect();
    let source_path_json: serde_json::Value = lazy
        .source_path()
        .map(|p| serde_json::Value::String(p.display().to_string()))
        .unwrap_or(serde_json::Value::Null);
    let params_json = serde_json::json!({
        "x_source": "lazy",
        "passthrough": false,
        "source_path": source_path_json,
        "lazy_transforms": transforms_repr,
        "csc_dropped": csc_dropped,
    });
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp,
            action: "from_anndata".to_string(),
            tool: format!("pyscx {}", env!("CARGO_PKG_VERSION")),
            params_json: params_json.to_string(),
            input_checksums: vec![],
        }])
        .map_err(to_pyerr)?;

    writer.finish().map_err(to_pyerr)?;

    if csc_build {
        py.detach(|| {
            scx_ops::rebuild_csc_inplace(std::path::Path::new(out_path), csc_cols_per_shard, "4G")
                .map_err(|e| e.to_string())
        })
        .map_err(|e| PyRuntimeError::new_err(format!("rebuild_csc_inplace failed: {e}")))?;
    }
    Ok(())
}

/// Compute output shard boundaries for the backed decode-encode
/// path. Sized by the **target** `shard_target_rows` (not the
/// source's), since the user may have asked for a different chunk
/// size — and that's the only reason we're on the decode-encode path
/// instead of byte-passthrough.
fn compute_wrapper_boundaries_backed(
    backed: &crate::backed::ScxBackedSparseDataset,
    target_shard_rows: u32,
) -> Vec<(usize, usize)> {
    let n_obs = backed.shape_val.0;
    chunk_boundaries(n_obs, target_shard_rows as usize)
}

fn compute_wrapper_boundaries_lazy(
    lazy: &crate::lazy_transform::ScxLazyTransformedDataset,
    target_shard_rows: u32,
) -> Vec<(usize, usize)> {
    let n_obs = lazy.shape_val.0;
    chunk_boundaries(n_obs, target_shard_rows as usize)
}

fn chunk_boundaries(n_obs: usize, target_rows: usize) -> Vec<(usize, usize)> {
    if n_obs == 0 || target_rows == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n_obs.div_ceil(target_rows));
    let mut start = 0usize;
    while start < n_obs {
        let end = (start + target_rows).min(n_obs);
        out.push((start, end));
        start = end;
    }
    out
}

/// Stream `adata.layers` shard-by-shard into the writer using the
/// same decode-encode pattern as the X path. Shared by both the
/// backed and lazy SCX → SCX routes — neither transforms layers,
/// so the logic is identical.
///
/// Each layer wrapper (`ScxBackedLayerDataset` or a scipy CSR) must
/// support `__getitem__(slice)` and report `(n_obs, n_vars)` via
/// `.shape`. The shape must match the output X dims; otherwise we
/// raise a `ValueError` matching the in-memory path's contract.
#[allow(clippy::too_many_arguments)]
fn stream_write_layers(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    writer: &mut ScxWriter,
    out_n_obs: u64,
    out_n_vars: u64,
    out_shard_rows: u32,
    codec_for_encode: Option<CodecId>,
    index_dtype: u8,
) -> PyResult<()> {
    let layers = match adata.getattr("layers") {
        Ok(l) => l,
        Err(_) => return Ok(()),
    };
    let keys: Vec<String> = py
        .import("builtins")?
        .call_method1("list", (layers.call_method0("keys")?,))?
        .extract()?;
    if keys.is_empty() {
        return Ok(());
    }

    let n_vars_u32 = u32::try_from(out_n_vars)
        .map_err(|_| PyRuntimeError::new_err(format!("n_vars {out_n_vars} exceeds u32::MAX")))?;
    let bounds = chunk_boundaries(out_n_obs as usize, out_shard_rows as usize);

    for layer_name in &keys {
        let layer = layers.call_method1("__getitem__", (layer_name,))?;
        let l_shape: (u64, u64) = layer.getattr("shape")?.extract()?;
        if l_shape != (out_n_obs, out_n_vars) {
            return Err(PyValueError::new_err(format!(
                "Layer '{layer_name}' has shape ({}, {}), expected ({}, {})",
                l_shape.0, l_shape.1, out_n_obs, out_n_vars
            )));
        }

        for (i, (start, end)) in bounds.iter().enumerate() {
            let py_slice = pyo3::types::PySlice::new(py, *start as isize, *end as isize, 1);
            let shard_obj = layer.call_method1("__getitem__", (py_slice,))?;
            let pre = decompose_scipy_csr_with(py, &shard_obj, |indptr, indices, data| {
                py.detach(|| {
                    scx_format::encode_one_shard(
                        indptr,
                        indices,
                        data,
                        codec_for_encode,
                        index_dtype,
                        n_vars_u32,
                        *start as u64,
                        SectionType::LayerCsrShard,
                        ModalityType::Rna,
                        format!("{layer_name}_shard_{i}"),
                    )
                })
                .map_err(to_pyerr)
            })?;
            py.detach(|| writer.write_preencoded_shard(pre))
                .map_err(to_pyerr)?;
        }
    }
    Ok(())
}

/// Mutation detection for the backed-routing path. Compares the
/// top-level key set of `getattr(adata, attr)` against the on-disk
/// h5py group at `/<attr>`. Returns `true` when the two key sets match
/// exactly (sender hasn't mutated this section in Python, so the
/// pipeline can stream from the source h5ad instead of materialising
/// the Python copy).
///
/// Cheapness invariants:
/// 1. `adata.obsm.keys()` on a backed AnnData lists the underlying
///    h5py group members without loading any dataset.
/// 2. `h5_file[attr].keys()` is a pure metadata read on h5py.
///
/// We never touch `adata.obsm[key]` here — that would force the very
/// h5py-to-numpy read we're trying to avoid.
///
/// Limitation: same-key replacements ("user did `adata.obsm['X_pca'] =
/// new_array`" without renaming) aren't detected. Document this with a
/// `UserWarning` on the routing path so users have a breadcrumb.
///
/// Gated behind the `hdf5` feature because the sole caller
/// (`route_backed_anndata_to_streaming`) is. Without `hdf5` the
/// streaming backed-AnnData path falls through to the in-memory branch
/// in `from_anndata_impl` and this helper would be dead code.
#[cfg(feature = "hdf5")]
fn section_keys_match(
    py: Python<'_>,
    h5_file: &Bound<'_, PyAny>,
    adata: &Bound<'_, PyAny>,
    attr: &str,
) -> PyResult<bool> {
    let py_keys: Vec<String> = match adata.getattr(attr) {
        Ok(section) => match section.call_method0("keys") {
            Ok(keys_obj) => py
                .import("builtins")?
                .call_method1("list", (keys_obj,))?
                .extract()
                .unwrap_or_default(),
            Err(_) => Vec::new(),
        },
        Err(_) => Vec::new(),
    };
    let disk_keys: Vec<String> = match h5_file.get_item(attr) {
        Ok(group) => match group.call_method0("keys") {
            Ok(keys_obj) => py
                .import("builtins")?
                .call_method1("list", (keys_obj,))?
                .extract()
                .unwrap_or_default(),
            Err(_) => Vec::new(),
        },
        Err(_) => Vec::new(),
    };
    let py_set: std::collections::BTreeSet<&String> = py_keys.iter().collect();
    let disk_set: std::collections::BTreeSet<&String> = disk_keys.iter().collect();
    Ok(py_set == disk_set)
}

/// Slice a dense obsm/varm `RecordBatch` into row-aligned shards and
/// emit each via `f`. Used by the SCX-backed / lazy / in-memory
/// `from_anndata` paths so all pyscx-produced SCX files share the
/// sharded on-disk layout the streaming pipeline emits.
///
/// The callback signature is `(shard_idx, row_start, n_shard_rows,
/// n_rows_total, batch)`; `n_shard_rows == batch.num_rows()` for dense
/// but the parameter is passed explicitly so the writer's contiguity
/// metadata is sourced from one place.
fn for_each_dense_shard<F>(
    batch: &RecordBatch,
    shard_target_rows: u32,
    mut f: F,
) -> std::result::Result<(), scx_format::ScxError>
where
    F: FnMut(u32, u64, u64, u64, &RecordBatch) -> std::result::Result<(), scx_format::ScxError>,
{
    let n_rows = batch.num_rows();
    let n_total = n_rows as u64;
    if n_rows == 0 {
        return f(0, 0, 0, 0, batch);
    }
    let step = shard_target_rows.max(1) as usize;
    let mut shard_idx = 0u32;
    let mut row_start = 0usize;
    while row_start < n_rows {
        let n = (n_rows - row_start).min(step);
        let shard = batch.slice(row_start, n);
        f(shard_idx, row_start as u64, n as u64, n_total, &shard)?;
        row_start += n;
        shard_idx += 1;
    }
    Ok(())
}

/// Slice a COO obsp/varp `RecordBatch` into row-shards keyed by the
/// `row` column. Buckets non-zero triples by `row / shard_target_rows`
/// then emits one shard per non-empty bucket. Used by the in-Python
/// override paths to keep on-disk obsp/varp layout symmetric with the
/// streaming pipeline. Returns `ScxError` (not `PyResult`) so callers
/// can drive it from inside `py.detach(...)`.
fn for_each_coo_shard<F>(
    batch: &RecordBatch,
    shard_target_rows: u32,
    mut f: F,
) -> std::result::Result<(), scx_format::ScxError>
where
    F: FnMut(u32, u64, u64, u64, &RecordBatch) -> std::result::Result<(), scx_format::ScxError>,
{
    use arrow::array::{Array, Float32Array, Int32Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};

    let invalid = |msg: String| {
        scx_format::ScxError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, msg))
    };

    // Width-generic: accept both v1 (Int32) and v2 (Int64) row/col columns,
    // and emit shards with the same coord dtype as the input. Reuse
    // `coo_coords_from_batch` so the inner bucketing loop dispatches on
    // `CooCoordsRef` (static match) rather than `Box<dyn Fn>` (per-element
    // vtable call + heap alloc).
    let coords = coo_coords_from_batch(batch).map_err(|e| invalid(e.to_string()))?;
    let coord_dt = match &coords {
        CooCoordsRef::Int32(_, _) => DataType::Int32,
        CooCoordsRef::Int64(_, _) => DataType::Int64,
    };
    let data_arr = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| invalid("sparse override: column 2 must be Float32".into()))?;
    let nnz = coords.len();

    let metadata = batch.schema_ref().metadata().clone();
    let n_rows: usize = metadata
        .get("n_rows")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| invalid("sparse override: missing 'n_rows' metadata".into()))?;
    let n_cols: usize = metadata
        .get("n_cols")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| invalid("sparse override: missing 'n_cols' metadata".into()))?;

    let step = shard_target_rows.max(1) as usize;
    if n_rows == 0 {
        return f(0, 0, 0, 0, batch);
    }

    // Bucket count scales with the logical row axis. For atlas-scale v2
    // obsp (axis ≥ 2^31) this would allocate hundreds of thousands of
    // empty `Vec`s up-front — fail fast with a clear message rather than
    // silently OOM. Real callers either route through the streaming
    // converter (which writes shards incrementally without this bucket
    // table) or use Phase 6's dedicated huge-obsp path.
    let n_shards = n_rows.div_ceil(step);
    const MAX_BUCKETS: usize = 1_000_000;
    if n_shards > MAX_BUCKETS {
        return Err(invalid(format!(
            "for_each_coo_shard: logical n_rows={n_rows} would require {n_shards} shard buckets \
             (cap {MAX_BUCKETS}). Route this obsp / varp through the streaming converter \
             or pre-split it into row bands before re-shading."
        )));
    }
    let mut bucket_row: Vec<Vec<i64>> = (0..n_shards).map(|_| Vec::new()).collect();
    let mut bucket_col: Vec<Vec<i64>> = (0..n_shards).map(|_| Vec::new()).collect();
    let mut bucket_data: Vec<Vec<f32>> = (0..n_shards).map(|_| Vec::new()).collect();
    for i in 0..nnz {
        let r = coords.row_i64(i);
        if r < 0 {
            return Err(invalid(format!("sparse override: negative row index {r}")));
        }
        let r_us = r as usize;
        let shard = r_us / step;
        if shard >= n_shards {
            return Err(invalid(format!(
                "sparse override: row {r} exceeds n_rows={n_rows}"
            )));
        }
        bucket_row[shard].push(r);
        bucket_col[shard].push(coords.col_i64(i));
        bucket_data[shard].push(data_arr.value(i));
    }

    let n_total = n_rows as u64;
    for shard_idx in 0..n_shards {
        let row_start = shard_idx * step;
        let n_shard_rows = step.min(n_rows - row_start);
        let schema = Arc::new(Schema::new_with_metadata(
            vec![
                Field::new("row", coord_dt.clone(), false),
                Field::new("col", coord_dt.clone(), false),
                Field::new("data", DataType::Float32, false),
            ],
            std::collections::HashMap::from([
                ("n_rows".to_string(), n_rows.to_string()),
                ("n_cols".to_string(), n_cols.to_string()),
            ]),
        ));
        let row_i64_taken = std::mem::take(&mut bucket_row[shard_idx]);
        let col_i64_taken = std::mem::take(&mut bucket_col[shard_idx]);
        let (row_array, col_array): (Arc<dyn Array>, Arc<dyn Array>) = match &coord_dt {
            DataType::Int32 => (
                Arc::new(Int32Array::from(
                    row_i64_taken
                        .into_iter()
                        .map(|v| v as i32)
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Int32Array::from(
                    col_i64_taken
                        .into_iter()
                        .map(|v| v as i32)
                        .collect::<Vec<_>>(),
                )),
            ),
            DataType::Int64 => (
                Arc::new(Int64Array::from(row_i64_taken)),
                Arc::new(Int64Array::from(col_i64_taken)),
            ),
            _ => unreachable!(),
        };
        let shard_batch = RecordBatch::try_new(
            schema,
            vec![
                row_array,
                col_array,
                Arc::new(Float32Array::from(std::mem::take(
                    &mut bucket_data[shard_idx],
                ))),
            ],
        )
        .map_err(scx_format::ScxError::Arrow)?;
        f(
            shard_idx as u32,
            row_start as u64,
            n_shard_rows as u64,
            n_total,
            &shard_batch,
        )?;
    }
    Ok(())
}

/// Helper for the backed-routing path. Reads a dense mapping
/// (`obsm` / `varm`) from a Python AnnData and returns
/// `Vec<(name, RecordBatch)>`. Missing groups → empty Vec.
fn extract_dense_mapping(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    attr: &str,
) -> PyResult<Vec<(String, RecordBatch)>> {
    let group = match adata.getattr(attr) {
        Ok(g) => g,
        Err(_) => return Ok(Vec::new()),
    };
    let keys: Vec<String> = py
        .import("builtins")?
        .call_method1("list", (group.call_method0("keys")?,))?
        .extract()?;
    keys.iter()
        .map(|key| {
            let arr = group.call_method1("__getitem__", (key,))?;
            let pd = py.import("pandas")?;
            let df = pd.call_method1("DataFrame", (&arr,))?;
            let batch = pandas_to_record_batch(py, &df)?;
            Ok((key.clone(), batch))
        })
        .collect()
}

/// Helper for the backed-routing path. Reads a sparse pairwise
/// mapping (`obsp` / `varp`) as COO RecordBatches. Missing groups →
/// empty Vec.
fn extract_coo_mapping(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    attr: &str,
) -> PyResult<Vec<(String, RecordBatch)>> {
    let group = match adata.getattr(attr) {
        Ok(g) => g,
        Err(_) => return Ok(Vec::new()),
    };
    let keys: Vec<String> = py
        .import("builtins")?
        .call_method1("list", (group.call_method0("keys")?,))?
        .extract()?;
    keys.iter()
        .map(|key| {
            let mat = group.call_method1("__getitem__", (key,))?;
            let batch = sparse_to_coo_record_batch(py, &mat)?;
            Ok((key.clone(), batch))
        })
        .collect()
}

/// Helper for the backed-routing path. Extracts `uns` from a Python
/// AnnData into an optional `serde_json::Value`. Returns `None` if
/// `uns` is empty (no `__scx_uns__` section written), matching the
/// non-backed path.
fn extract_uns_value(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    uns_format_parsed: UnsFormat,
) -> PyResult<Option<serde_json::Value>> {
    let uns = adata.getattr("uns")?;
    let uns_len: usize = uns.call_method0("__len__")?.extract()?;
    if uns_len == 0 {
        return Ok(None);
    }
    let np = py.import("numpy")?;
    let np_generic = np.getattr("generic")?;
    let np_ndarray = np.getattr("ndarray")?;
    let mut ctx = UnsWriteCtx::new(uns_format_parsed, &np_generic, &np_ndarray);
    Ok(Some(normalize_uns_value(&uns, "uns", &mut ctx)?))
}

/// Implementation of from_anndata: extract data from AnnData and write SCX.
///
/// `in_place`: when true, allow [`ensure_csr`] to sort caller-owned CSR
/// indices in place (mutates `adata.X` / `adata.layers[*]`). When false
/// (default), unsorted CSR inputs are copied via `.sorted_indices()` so
/// the caller's matrices are untouched.
#[allow(clippy::too_many_arguments)]
pub fn from_anndata_impl(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    path: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
    in_place: bool,
    csc: &str,
    csc_cols_per_shard: usize,
    uns_format: &str,
    index_obs: Vec<String>,
    index_var: Vec<String>,
    index_preset: Option<String>,
    index_auto_threshold: usize,
    bitmap: &str,
    memory_budget: Option<u64>,
    force_legacy_metadata: bool,
) -> PyResult<()> {
    let explicit_codec = parse_codec(codec)?;
    let shard_target_rows = shard_size.unwrap_or(16384);
    let csc_policy =
        scx_format::CscPolicy::parse(csc).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let uns_format_parsed = parse_uns_format(uns_format)?;

    // Backed AnnData → route through the streaming converter
    // (`scx_convert::h5ad_to_scx_streaming`) instead of the in-memory
    // path, which would fail at the `ensure_csr` step (backed `X` is
    // an `_CSRDataset`, not a scipy sparse matrix). In-memory
    // mutations on `obs` / `var` / `uns` / `obsm` / `varm` / `obsp` /
    // `varp` are extracted to Rust and passed as `StreamingOverrides`
    // so user edits aren't silently overwritten by the on-disk
    // version. Available only when pyscx was built with the `hdf5`
    // feature; without it the call falls through to the in-memory
    // path which raises a clear error on the backed `_CSRDataset`.
    let is_backed: bool = adata
        .getattr("isbacked")
        .ok()
        .and_then(|v| v.extract::<bool>().ok())
        .unwrap_or(false);
    if is_backed {
        #[cfg(feature = "hdf5")]
        {
            // `from_anndata(adata)` doesn't take Phase 1 kwargs yet —
            // it always asks for streaming with default policy. Phase
            // 1 kwarg surface lives on `pyscx.from_h5ad(path, ...)`.
            return route_backed_anndata_to_streaming(
                py,
                adata,
                path,
                explicit_codec,
                shard_target_rows,
                csc_policy,
                csc_cols_per_shard,
                uns_format_parsed,
                true,  // stream
                false, // strict_uns
                0.0,   // dense_zero_epsilon
                memory_budget,
                None, // temp_dir
                index_obs,
                index_var,
                index_preset,
                index_auto_threshold,
                bitmap,
                None, // reader_threads (auto)
                4,    // writer_queue_depth (default)
            );
        }
        #[cfg(not(feature = "hdf5"))]
        {
            let _ = (
                explicit_codec,
                csc_policy,
                csc_cols_per_shard,
                uns_format_parsed,
                &index_obs,
                &index_var,
                &index_preset,
                index_auto_threshold,
                bitmap,
            );
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "pyscx was built without the `hdf5` feature; backed AnnData \
                 routing requires libhdf5. Rebuild with \
                 `maturin develop --features hdf5` or convert the AnnData \
                 to a non-backed form first.",
            ));
        }
    }

    // Extract X as CSR. By default we do not mutate caller-owned CSR
    // matrices; pass `in_place=true` to opt into the original in-place
    // sort behavior for speed/memory.
    let x = adata.getattr("X")?;

    // Phase 8b: SCX-backed or lazy `X` → stream from the source SCX
    // file without materialising X into a scipy CSR. Falls through
    // to the existing in-memory path for scipy / numpy input. The
    // `extract::<PyRef<…>>()` calls are no-ops on non-matching
    // types (fail-fast, no Python call overhead).
    if let Ok(backed) = x.extract::<PyRef<crate::backed::ScxBackedSparseDataset>>() {
        return route_scx_backed_to_scx(
            py,
            adata,
            &backed,
            path,
            explicit_codec,
            shard_target_rows,
            csc_policy,
            csc_cols_per_shard,
            uns_format_parsed,
            shard_size.is_some(),
        );
    }
    if let Ok(lazy) = x.extract::<PyRef<crate::lazy_transform::ScxLazyTransformedDataset>>() {
        return route_scx_lazy_to_scx(
            py,
            adata,
            &lazy,
            path,
            explicit_codec,
            shard_target_rows,
            csc_policy,
            csc_cols_per_shard,
            uns_format_parsed,
        );
    }

    let (x_csr, csr_validated) = ensure_csr(py, &x, in_place)?;

    // Get shape
    let shape: (u64, u64) = x_csr.getattr("shape")?.extract()?;
    let n_obs = shape.0;
    let n_vars = shape.1;

    // Resolve the CSC policy now that the in-memory shape is known
    // (`Auto` compares against the size thresholds).
    let csc_build = csc_policy.should_build_csc(n_obs, n_vars);

    if n_vars > u32::MAX as u64 {
        return Err(PyRuntimeError::new_err(format!(
            "n_vars ({n_vars}) exceeds u32::MAX; SCX format requires n_vars <= {}",
            u32::MAX
        )));
    }

    // Extract CSR arrays — skip .astype() when dtypes already match (1C.1)
    let np = py.import("numpy")?;

    let indptr_obj = x_csr.getattr("indptr")?;
    let indptr_arr = astype_if_needed(&indptr_obj, &np, "int64")?;
    let indptr: PyReadonlyArray1<'_, i64> = indptr_arr.extract()?;
    let indptr_slice = indptr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let expected_indptr_len = (n_obs as usize) + 1;
    if indptr_slice.len() != expected_indptr_len {
        return Err(PyValueError::new_err(format!(
            "X indptr has length {}, expected n_obs + 1 = {} (X.shape = ({}, {}))",
            indptr_slice.len(),
            expected_indptr_len,
            n_obs,
            n_vars
        )));
    }

    let indices_obj = x_csr.getattr("indices")?;
    let indices_arr = astype_if_needed(&indices_obj, &np, "int32")?;
    let indices: PyReadonlyArray1<'_, i32> = indices_arr.extract()?;
    let indices_slice = indices
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let data_obj = x_csr.getattr("data")?;
    // T3.7: warn on lossy float64 → float32 value downcast at the write
    // boundary so the precision loss recorded in the round-trip fidelity
    // table is also visible at runtime, not just in docs.
    if let Ok(name) = data_obj
        .getattr("dtype")
        .and_then(|d| d.getattr("name"))
        .and_then(|n| n.extract::<String>())
    {
        if name == "float64" || name == "float128" {
            // stacklevel=2 so `-W error` points at the user's
            // `write()` / `from_anndata()` call, not the PyO3 bridge frame.
            let warn_fn = py.import("warnings")?.getattr("warn")?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("stacklevel", 2)?;
            warn_fn.call(
                (format!(
                    "X values are stored as float32 in SCX; the source matrix is \
                     {name}, so values are downcast and precision is reduced. \
                     This is expected — see the round-trip fidelity table in the docs."
                ),),
                Some(&kwargs),
            )?;
        }
    }
    let data_arr = astype_if_needed(&data_obj, &np, "float32")?;
    let data: PyReadonlyArray1<'_, f32> = data_arr.extract()?;
    let data_slice = data
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let nnz = data_slice.len() as u64;

    // 1C.2: Fast upfront validation when CSR bypass is active.
    // After this, the shard loop can skip per-element checks.
    if csr_validated {
        scx_sparse::validate_csr_arrays(indptr_slice, indices_slice, n_vars)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    }

    // Determine index dtype.
    //
    // CSR shards encode column indices (bounded by `n_vars`); CSC
    // sidecars encode global row indices (bounded by `n_obs`). The
    // file header carries one shared `index_dtype` that drives the
    // u16/u32 encoding choice in `write_shard_inner`. When CSC is
    // requested, fall back to u32 if EITHER axis exceeds u16. This
    // costs CSR a few bytes per index when n_obs > 65535 but
    // unblocks CSC writes on large-cell datasets (`census_1m`+).
    let index_dtype: u8 = {
        let max_axis = if csc_build { n_obs.max(n_vars) } else { n_vars };
        if max_axis <= 65535 {
            0
        } else {
            1
        }
    };

    // Peek at first shard's data to set file header codec_id (informational only;
    // readers use the per-shard header). Per-shard encoding/codec selection
    // happens inside the shard loop below.
    let first_shard_nnz_end = if n_obs as usize > 0 {
        indptr_slice[(shard_target_rows as usize).min(n_obs as usize)] as usize
    } else {
        0
    };
    let first_shard_data = &data_slice[..first_shard_nnz_end];
    let first_encoding = detect_value_encoding(first_shard_data);
    let first_values = encode_values(first_shard_data, first_encoding)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let header_codec = match explicit_codec {
        Some(codec_id) => {
            if codec_id == CodecId::Scx1 && !first_encoding.is_integer() {
                CodecId::Zstd
            } else {
                codec_id
            }
        }
        None => select_codec_for_modality(&first_values, first_encoding, ModalityType::Rna),
    };

    // Build FileHeader
    let header = FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs,
        n_vars,
        nnz,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows,
        codec_id: header_codec as u8,
        index_dtype,
        endian: 0,
        reserved_padding: 0,
        root_catalog_offset: 0,
        root_catalog_length: 0,
        full_catalog_offset: 0,
        full_catalog_length: 0,
        manifest_sequence: 1,
        prev_catalog_offset: 0,
        file_checksum: 0,
        front_catalog_offset: 0,
        front_catalog_length: 0,
        n_modalities: 0,
        modality_table_offset: 0,
        modality_table_length: 0,
        reserved: [0u8; 112],
    };

    let mut writer = ScxWriter::new(path, header).map_err(to_pyerr)?;

    // Write obs — always write even for 0-cell datasets to preserve column schema (finding 9.7).
    let obs_df = adata.getattr("obs")?;
    let obs_batch = pandas_to_record_batch(py, &obs_df)?;

    // Write var — always write even for 0-gene datasets to preserve column schema (finding 9.7).
    let var_df = adata.getattr("var")?;
    let var_batch = pandas_to_record_batch(py, &var_df)?;

    // Phase 4c: shard obs/var when row counts exceed shard_target_rows so
    // `from_anndata` emits the same `ObsMetadataShard` / `VarMetadataShard`
    // layout that merge / append / streaming ingest produce at atlas scale.
    // `force_legacy_metadata=true` opts back into single-section writes.
    let step = shard_target_rows as usize;
    let obs_rows = obs_batch.num_rows();
    let var_rows = var_batch.num_rows();
    let shard_obs = !force_legacy_metadata && obs_rows > step;
    let shard_var = !force_legacy_metadata && var_rows > step;
    py.detach(|| -> std::result::Result<(), scx_format::ScxError> {
        if shard_obs {
            let n_total = obs_rows as u64;
            let mut shard_idx: u32 = 0;
            let mut row_start: usize = 0;
            while row_start < obs_rows {
                let lo = row_start;
                let hi = (lo + step).min(obs_rows);
                let shard = obs_batch.slice(lo, hi - lo);
                writer.write_obs_shard(shard_idx, lo as u64, (hi - lo) as u64, n_total, &shard)?;
                shard_idx += 1;
                row_start = hi;
            }
        } else {
            writer.write_obs(&obs_batch)?;
        }
        if shard_var {
            let n_total = var_rows as u64;
            let mut shard_idx: u32 = 0;
            let mut row_start: usize = 0;
            while row_start < var_rows {
                let lo = row_start;
                let hi = (lo + step).min(var_rows);
                let shard = var_batch.slice(lo, hi - lo);
                writer.write_var_shard(shard_idx, lo as u64, (hi - lo) as u64, n_total, &shard)?;
                shard_idx += 1;
                row_start = hi;
            }
        } else {
            writer.write_var(&var_batch)?;
        }
        Ok(())
    })
    .map_err(to_pyerr)?;

    // Write CSR shards (1D: parallel shard encoding)
    let n_obs_usize = n_obs as usize;
    let shard_rows = shard_target_rows as usize;

    // 1D.1: Compute shard boundaries sequentially
    let mut boundaries = Vec::new();
    {
        let mut row_start: usize = 0;
        let mut shard_idx: u32 = 0;
        while row_start < n_obs_usize {
            let row_end = (row_start + shard_rows).min(n_obs_usize);
            let base = indptr_slice[row_start];
            if !csr_validated && base < 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "negative indptr value {base} at row {row_start}"
                )));
            }
            boundaries.push(ShardBoundary {
                row_start,
                row_end,
                nnz_start: base as usize,
                nnz_end: indptr_slice[row_end] as usize,
                indptr_base: base,
                shard_idx,
            });
            row_start = row_end;
            shard_idx += 1;
        }
    }

    // 1D.2+1D.3: Parallel encode + sequential write
    let pre_encoded = parallel_encode_csr_shards(
        py,
        indptr_slice,
        indices_slice,
        data_slice,
        &boundaries,
        csr_validated,
        explicit_codec,
        index_dtype,
        n_vars as u32,
        SectionType::CsrShard,
        "X",
    )?;
    // Phase 5b: parse bitmap policy once.
    let bitmap_policy = scx_format::BitmapPolicy::parse(bitmap)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    for (boundary, section) in boundaries.iter().zip(pre_encoded) {
        let encoded_csr_size = section.section_length as usize;
        writer.write_preencoded_shard(section).map_err(to_pyerr)?;
        if !matches!(bitmap_policy, scx_format::BitmapPolicy::Off) {
            // Build the bitmap from the same canonical local CSR
            // representation used by the encoded shard.
            let lo = boundary.row_start;
            let hi = boundary.row_end;
            let nnz_lo = boundary.nnz_start;
            let nnz_hi = boundary.nnz_end;
            let mut local_indptr: Vec<u64> = indptr_slice[lo..=hi]
                .iter()
                .map(|&v| (v - boundary.indptr_base) as u64)
                .collect();
            let mut local_indices: Vec<u32> = indices_slice[nnz_lo..nnz_hi]
                .iter()
                .map(|&v| v as u32)
                .collect();
            let mut local_data = data_slice[nnz_lo..nnz_hi].to_vec();
            canonicalize_csr(&mut local_indptr, &mut local_indices, &mut local_data);
            let n_rows = (hi - lo) as u32;
            build_and_write_bitmap_for_shard_python(
                py,
                &mut writer,
                &local_indptr,
                &local_indices,
                lo as u64,
                n_rows,
                n_vars as u32,
                encoded_csr_size,
                bitmap_policy,
            )?;
        }
    }

    // Phase 4a/4b: stream obsm/varm/obsp/varp one key at a time.
    // Each iteration extracts a single key's value under the GIL,
    // builds one RecordBatch (numpy fast-path for plain numeric ndarrays
    // skips the `pd.DataFrame(arr)` roundtrip), optionally warns when
    // the estimated peak footprint exceeds `memory_budget`, then writes
    // shards in `py.detach` and drops the batch before moving on.
    // This bounds peak RSS to one key's payload at a time instead of
    // the full sum of all mappings.
    let obsm = adata.getattr("obsm")?;
    let obsm_keys: Vec<String> = py
        .import("builtins")?
        .call_method1("list", (obsm.call_method0("keys")?,))?
        .extract()?;
    for key in &obsm_keys {
        let arr = obsm.call_method1("__getitem__", (key,))?;
        let batch = numpy_or_pandas_to_record_batch(py, &arr)?;
        let est = estimate_dense_bytes(&batch);
        if let Some(budget) = memory_budget {
            if est > budget {
                warn_python_convert(
                    py,
                    &scx_convert::ConvertWarning::MappingPeakFootprintHigh {
                        key: key.clone(),
                        axis: "obsm",
                        estimated_bytes: est,
                        budget_bytes: budget,
                    },
                )?;
            }
        }
        py.detach(|| -> Result<(), scx_format::ScxError> {
            for_each_dense_shard(
                &batch,
                shard_target_rows,
                |idx, row_start, n_shard_rows, n_total, shard| {
                    writer.write_obsm_shard(key, idx, row_start, n_shard_rows, n_total, shard)
                },
            )
        })
        .map_err(to_pyerr)?;
    }

    // varm — same pattern. Duck-typed AnnData-likes may omit
    // `varm`/`obsp`/`varp` entirely; missing attrs are treated as empty.
    if let Ok(varm) = adata.getattr("varm") {
        let varm_keys: Vec<String> = py
            .import("builtins")?
            .call_method1("list", (varm.call_method0("keys")?,))?
            .extract()?;
        for key in &varm_keys {
            let arr = varm.call_method1("__getitem__", (key,))?;
            let batch = numpy_or_pandas_to_record_batch(py, &arr)?;
            let est = estimate_dense_bytes(&batch);
            if let Some(budget) = memory_budget {
                if est > budget {
                    warn_python_convert(
                        py,
                        &scx_convert::ConvertWarning::MappingPeakFootprintHigh {
                            key: key.clone(),
                            axis: "varm",
                            estimated_bytes: est,
                            budget_bytes: budget,
                        },
                    )?;
                }
            }
            py.detach(|| -> Result<(), scx_format::ScxError> {
                for_each_dense_shard(
                    &batch,
                    shard_target_rows,
                    |idx, row_start, n_shard_rows, n_total, shard| {
                        writer.write_varm_shard(key, idx, row_start, n_shard_rows, n_total, shard)
                    },
                )
            })
            .map_err(to_pyerr)?;
        }
    }

    // obsp — sparse COO, one key at a time.
    if let Ok(obsp) = adata.getattr("obsp") {
        let obsp_keys: Vec<String> = py
            .import("builtins")?
            .call_method1("list", (obsp.call_method0("keys")?,))?
            .extract()?;
        for key in &obsp_keys {
            let mat = obsp.call_method1("__getitem__", (key,))?;
            let batch = sparse_to_coo_record_batch(py, &mat)?;
            let est = estimate_coo_bytes(&batch);
            if let Some(budget) = memory_budget {
                if est > budget {
                    warn_python_convert(
                        py,
                        &scx_convert::ConvertWarning::MappingPeakFootprintHigh {
                            key: key.clone(),
                            axis: "obsp",
                            estimated_bytes: est,
                            budget_bytes: budget,
                        },
                    )?;
                }
            }
            py.detach(|| -> Result<(), scx_format::ScxError> {
                for_each_coo_shard(
                    &batch,
                    shard_target_rows,
                    |idx, row_start, n_shard_rows, n_total, shard| {
                        writer.write_obsp_shard_coo(
                            key,
                            idx,
                            row_start,
                            n_shard_rows,
                            n_total,
                            shard,
                        )
                    },
                )
            })
            .map_err(to_pyerr)?;
        }
    }

    // varp — sparse COO, one key at a time.
    if let Ok(varp) = adata.getattr("varp") {
        let varp_keys: Vec<String> = py
            .import("builtins")?
            .call_method1("list", (varp.call_method0("keys")?,))?
            .extract()?;
        for key in &varp_keys {
            let mat = varp.call_method1("__getitem__", (key,))?;
            let batch = sparse_to_coo_record_batch(py, &mat)?;
            let est = estimate_coo_bytes(&batch);
            if let Some(budget) = memory_budget {
                if est > budget {
                    warn_python_convert(
                        py,
                        &scx_convert::ConvertWarning::MappingPeakFootprintHigh {
                            key: key.clone(),
                            axis: "varp",
                            estimated_bytes: est,
                            budget_bytes: budget,
                        },
                    )?;
                }
            }
            py.detach(|| -> Result<(), scx_format::ScxError> {
                for_each_coo_shard(
                    &batch,
                    shard_target_rows,
                    |idx, row_start, n_shard_rows, n_total, shard| {
                        writer.write_varp_shard_coo(
                            key,
                            idx,
                            row_start,
                            n_shard_rows,
                            n_total,
                            shard,
                        )
                    },
                )
            })
            .map_err(to_pyerr)?;
        }
    }

    // 1E.2: Collect uns JSON under GIL.
    // Use a recursive Python-side normalizer so common AnnData payloads
    // (NumPy arrays/scalars, pandas Index/Series/Categorical) survive the
    // JSON boundary instead of erroring out of `json.dumps`. Under
    // `UnsFormat::Tagged` (default), payloads are wrapped in `__scx_type__`
    // envelopes so dtype/shape/NaN/Inf round-trip losslessly.
    let uns = adata.getattr("uns")?;
    let uns_len: usize = uns.call_method0("__len__")?.extract()?;
    let uns_json: Option<serde_json::Value> = if uns_len > 0 {
        let np_generic = np.getattr("generic")?;
        let np_ndarray = np.getattr("ndarray")?;
        let mut ctx = UnsWriteCtx::new(uns_format_parsed, &np_generic, &np_ndarray);
        Some(normalize_uns_value(&uns, "uns", &mut ctx)?)
    } else {
        None
    };

    // Write uns outside GIL.
    if let Some(ref json_val) = uns_json {
        py.detach(|| writer.write_uns(json_val)).map_err(to_pyerr)?;
    }

    // Write layers
    let layers = adata.getattr("layers")?;
    let layer_keys: Vec<String> = py
        .import("builtins")?
        .call_method1("list", (layers.call_method0("keys")?,))?
        .extract()?;
    for layer_name in &layer_keys {
        let layer_x = layers.call_method1("__getitem__", (layer_name,))?;
        let (layer_csr, l_csr_validated) = ensure_csr(py, &layer_x, in_place)?;

        let l_shape: (u64, u64) = layer_csr.getattr("shape")?.extract()?;
        if l_shape != (n_obs, n_vars) {
            return Err(PyValueError::new_err(format!(
                "Layer '{layer_name}' has shape ({}, {}), expected ({}, {})",
                l_shape.0, l_shape.1, n_obs, n_vars
            )));
        }

        let l_indptr_obj = layer_csr.getattr("indptr")?;
        let l_indptr_arr = astype_if_needed(&l_indptr_obj, &np, "int64")?;
        let l_indptr: PyReadonlyArray1<'_, i64> = l_indptr_arr.extract()?;
        let l_indptr_slice = l_indptr
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        if l_indptr_slice.len() != expected_indptr_len {
            return Err(PyValueError::new_err(format!(
                "Layer '{layer_name}' indptr has length {}, expected n_obs + 1 = {}",
                l_indptr_slice.len(),
                expected_indptr_len
            )));
        }

        let l_indices_obj = layer_csr.getattr("indices")?;
        let l_indices_arr = astype_if_needed(&l_indices_obj, &np, "int32")?;
        let l_indices: PyReadonlyArray1<'_, i32> = l_indices_arr.extract()?;
        let l_indices_slice = l_indices
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let l_data_obj = layer_csr.getattr("data")?;
        let l_data_arr = astype_if_needed(&l_data_obj, &np, "float32")?;
        let l_data: PyReadonlyArray1<'_, f32> = l_data_arr.extract()?;
        let l_data_slice = l_data
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // 1C.2: Upfront validation for layer bypass
        if l_csr_validated {
            scx_sparse::validate_csr_arrays(l_indptr_slice, l_indices_slice, n_vars)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }

        // 1D: Parallel shard encoding for layers
        let mut l_boundaries = Vec::new();
        {
            let mut l_row_start: usize = 0;
            let mut shard_idx: u32 = 0;
            while l_row_start < n_obs_usize {
                let l_row_end = (l_row_start + shard_rows).min(n_obs_usize);
                let l_base = l_indptr_slice[l_row_start];
                if !l_csr_validated && l_base < 0 {
                    return Err(PyRuntimeError::new_err(format!(
                        "layer '{layer_name}': negative indptr value {l_base} at row {l_row_start}"
                    )));
                }
                l_boundaries.push(ShardBoundary {
                    row_start: l_row_start,
                    row_end: l_row_end,
                    nnz_start: l_base as usize,
                    nnz_end: l_indptr_slice[l_row_end] as usize,
                    indptr_base: l_base,
                    shard_idx,
                });
                l_row_start = l_row_end;
                shard_idx += 1;
            }
        }

        let l_pre_encoded = parallel_encode_csr_shards(
            py,
            l_indptr_slice,
            l_indices_slice,
            l_data_slice,
            &l_boundaries,
            l_csr_validated,
            explicit_codec,
            index_dtype,
            n_vars as u32,
            SectionType::LayerCsrShard,
            layer_name,
        )?;
        for section in l_pre_encoded {
            writer.write_preencoded_shard(section).map_err(to_pyerr)?;
        }
    }

    // Optional CSC sidecar — streaming transpose over the in-memory
    // CSR view of X. Layers are CSR-only (no layer-CSC support yet —
    // a `LayerCscShard` section type would need to land first).
    if csc_build {
        py.detach(|| -> Result<(), scx_format::ScxError> {
            write_csc_shards_from_csr(
                &mut writer,
                indptr_slice,
                indices_slice,
                data_slice,
                n_obs as usize,
                n_vars as usize,
                first_encoding,
                header_codec,
                csc_cols_per_shard,
            )
        })
        .map_err(to_pyerr)?;
    }

    // Phase 5a: predicate indexes. Row ranges come from the boundaries
    // we already computed for the CSR shards — guaranteed to match
    // what's on disk because they drove the write itself.
    let csr_row_ranges: Vec<(u64, u64)> = boundaries
        .iter()
        .map(|b| (b.row_start as u64, b.row_end as u64))
        .collect();
    build_and_write_predicate_indexes_inline(
        py,
        &mut writer,
        &obs_batch,
        &var_batch,
        &csr_row_ranges,
        n_vars as usize,
        &index_obs,
        &index_var,
        index_preset.as_deref(),
        index_auto_threshold,
    )?;

    // Write provenance
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            action: "from_anndata".to_string(),
            tool: format!("pyscx {}", env!("CARGO_PKG_VERSION")),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .map_err(to_pyerr)?;

    writer.finish().map_err(to_pyerr)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-modality backed AnnData assembly.
//
// Builds a backed `AnnData` scoped to a single modality of a multimodal SCX
// file. X is wrapped in `ScxBackedSparseDataset` over a
// `BackedCsrReader::for_modality` (+ optional `BackedCscReader::for_modality`
// when the modality has a CSC sidecar). `obs` is the global obs DataFrame
// shared across modalities; `var` and `obsm` come from the per-modality
// reader helpers (`read_var_for`, `read_obsm_for`).
//
// Deletion vectors are NOT applied on this path. They are global (operate on
// the global obs axis) and would apply identically across modalities; lifting
// `compute_kept_to_global` / `filter_obs_by_deletion_vectors` into the
// per-modality helper is a follow-on. The existing `to_mudata` (eager) path
// also skips deletion vectors, so this matches today's eager behaviour.
// ---------------------------------------------------------------------------

/// Assemble a backed AnnData scoped to a single modality. `obs_df` is the
/// pre-built pandas DataFrame (shared across modalities); pass `None` to
/// construct without an obs attached (rare).
#[allow(clippy::too_many_arguments)]
pub fn build_backed_anndata_for_modality<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    shared_catalog: &Arc<scx_format::FullCatalog>,
    modality_id: u8,
    modality_name: &str,
    cache_shards: usize,
    obs_df: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    use crate::backed::ScxBackedSparseDataset;
    use scx_format::BackedCsrReader;

    let anndata_mod = py.import("anndata")?;

    // Meta reader for var / obsm / modality_info introspection.
    let meta =
        ScxReader::open_with_shared_catalog(path, Arc::clone(shared_catalog)).map_err(to_pyerr)?;
    let info = meta.modality_info(modality_id).ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "modality_info({modality_id}) returned None — modality table is corrupt"
        ))
    })?;
    let has_csc = info.flags.has_csc();

    // Per-modality backed X (CSR).
    let csr_reader =
        ScxReader::open_with_shared_catalog(path, Arc::clone(shared_catalog)).map_err(to_pyerr)?;
    let backed_csr = Arc::new(BackedCsrReader::for_modality(
        csr_reader,
        modality_id,
        cache_shards,
    ));

    // Per-modality backed CSC sidecar (optional).
    let backed_csc = if has_csc {
        let csc_reader = ScxReader::open_with_shared_catalog(path, Arc::clone(shared_catalog))
            .map_err(to_pyerr)?;
        Some(Arc::new(
            scx_format::BackedCscReader::for_modality(csc_reader, modality_id, cache_shards)
                .map_err(to_pyerr)?,
        ))
    } else {
        None
    };

    let mut x_dataset = ScxBackedSparseDataset::from_reader(Arc::clone(&backed_csr), cache_shards);
    x_dataset.with_csc_reader(backed_csc);
    x_dataset.with_modality_id(modality_id);
    x_dataset.with_source_path(path);

    // Per-modality var.
    let var_batch = meta.read_var_for(modality_id).map_err(to_pyerr)?;
    let var_table = record_batch_to_pyarrow(py, &var_batch)?;
    let var_df = pyarrow_table_to_pandas(&var_table)?;

    // Per-modality obsm: catalog entries with this modality_id.
    let obsm_dict = pyo3::types::PyDict::new(py);
    let prefix = format!("obsm/{modality_name}/");
    for entry in &meta.catalog().entries {
        if entry.section_type != SectionType::ObsmEmbedding {
            continue;
        }
        if entry.modality_id != modality_id {
            continue;
        }
        let key = entry
            .name
            .strip_prefix(&prefix)
            .unwrap_or(&entry.name)
            .to_string();
        let batch = meta.read_obsm_for(modality_id, &key).map_err(to_pyerr)?;
        let np_arr = obsm_batch_to_numpy(py, &batch)?;
        obsm_dict.set_item(&key, np_arr)?;
    }

    // Assemble AnnData kwargs.
    let kwargs = pyo3::types::PyDict::new(py);
    let x_py = x_dataset.into_pyobject(py)?;
    kwargs.set_item("X", x_py)?;
    if let Some(obs) = obs_df {
        kwargs.set_item("obs", obs)?;
    }
    kwargs.set_item("var", var_df)?;
    if !obsm_dict.is_empty() {
        kwargs.set_item("obsm", obsm_dict)?;
    }

    anndata_mod.call_method("AnnData", (), Some(&kwargs))
}

/// Public entrypoint: open `path`, resolve `modality` → `modality_id`, build a
/// backed AnnData wrapping that modality's CSR (+ CSC sidecar if present),
/// with the global obs attached.
pub fn to_anndata_backed_for_modality<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    modality: &str,
    cache_shards: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let reader = ScxReader::open(path).map_err(to_pyerr)?;
    let modality_id = reader.modality_id(modality).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!(
            "unknown modality '{modality}' (available: {:?})",
            reader.modality_names()
        ))
    })?;
    let modality_name = reader
        .modality_info(modality_id)
        .map(|i| i.name.clone())
        .unwrap_or_else(|| modality.to_string());

    let obs_df = match reader.read_obs() {
        Ok(batch) => {
            let table = record_batch_to_pyarrow(py, &batch)?;
            Some(pyarrow_table_to_pandas(&table)?)
        }
        Err(scx_format::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    let shared_catalog = reader.catalog_arc();
    build_backed_anndata_for_modality(
        py,
        path,
        &shared_catalog,
        modality_id,
        &modality_name,
        cache_shards,
        obs_df.as_ref(),
    )
}
