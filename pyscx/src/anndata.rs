// to_anndata / from_anndata conversion

use arrow::array::RecordBatch;
use byteorder::{LittleEndian, WriteBytesExt};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::io::Cursor;

use scx_codec::{CodecId, ValueEncoding};
use scx_format::header::MAGIC;
use scx_format::{select_codec, FileHeader, ProvenanceEntry, ScxReader, ScxWriter};

use crate::to_pyerr;

// ---------------------------------------------------------------------------
// to_anndata: SCX → AnnData
// ---------------------------------------------------------------------------

/// Convert an Arrow RecordBatch to a pyarrow Table via IPC bytes.
pub(crate) fn record_batch_to_pyarrow<'py>(
    py: Python<'py>,
    batch: &RecordBatch,
) -> PyResult<Bound<'py, PyAny>> {
    // Serialize to Arrow IPC file format
    let mut buf = Vec::new();
    {
        let mut writer = arrow::ipc::writer::FileWriter::try_new(&mut buf, batch.schema_ref())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        writer
            .write(batch)
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
pub(crate) fn pyarrow_table_to_pandas<'py>(
    table: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = pyo3::types::PyDict::new(table.py());
    kwargs.set_item("self_destruct", true)?;
    table.call_method("to_pandas", (), Some(&kwargs))
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

/// Convert an Arrow RecordBatch (obsm) to a numpy 2D array.
fn obsm_batch_to_numpy<'py>(py: Python<'py>, batch: &RecordBatch) -> PyResult<Bound<'py, PyAny>> {
    let table = record_batch_to_pyarrow(py, batch)?;
    let df = pyarrow_table_to_pandas(&table)?;
    df.getattr("values")
}

/// Build an AnnData object from an ScxReader.
///
/// When deletion vectors are present, deleted cells are excluded from
/// both the CSR matrix and the obs metadata.
pub fn to_anndata<'py>(py: Python<'py>, reader: &ScxReader) -> PyResult<Bound<'py, PyAny>> {
    let anndata_mod = py.import("anndata")?;

    // X — assemble all CSR shards (with deletion vector filtering)
    let csr = reader.read_all_csr_shards_filtered().map_err(to_pyerr)?;
    let x = csr_to_scipy(py, csr)?;

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

    // obsm embeddings
    let obsm_map = match reader.read_all_obsm() {
        Ok(map) => map,
        Err(scx_format::ScxError::SectionNotFound(_)) => std::collections::HashMap::new(),
        Err(e) => return Err(to_pyerr(e)),
    };
    let obsm_dict = pyo3::types::PyDict::new(py);
    for (name, batch) in &obsm_map {
        let np_arr = obsm_batch_to_numpy(py, batch)?;
        obsm_dict.set_item(name, np_arr)?;
    }

    // uns
    let uns_dict = match reader.read_uns() {
        Ok(json_val) => {
            let json_str = serde_json::to_string(&json_val)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let json_mod = py.import("json")?;
            Some(json_mod.call_method1("loads", (json_str,))?)
        }
        Err(scx_format::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    // layers
    let layer_names = reader.layer_names();
    let layers_dict = pyo3::types::PyDict::new(py);
    for name in &layer_names {
        match reader.read_layer(name) {
            Ok(layer_csr) => {
                let scipy_mat = csr_to_scipy(py, layer_csr)?;
                layers_dict.set_item(name, scipy_mat)?;
            }
            Err(scx_format::ScxError::SectionNotFound(_)) => {}
            Err(e) => return Err(to_pyerr(e)),
        }
    }

    // Build AnnData kwargs
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("X", x)?;
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

    let adata = anndata_mod.call_method("AnnData", (), Some(&kwargs))?;
    Ok(adata)
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
pub fn to_anndata_backed<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    cache_shards: usize,
) -> PyResult<Bound<'py, PyAny>> {
    use crate::backed::{ScxBackedLayerDataset, ScxBackedSparseDataset};
    use scx_format::BackedCsrReader;
    use std::sync::Arc;

    let anndata_mod = py.import("anndata")?;
    let reader = ScxReader::open(path).map_err(to_pyerr)?;

    // --- Compute kept_to_global from deletion vectors (if present) ---
    let kept_to_global = compute_kept_to_global(&reader)?;

    // --- X: backed ---
    let x_reader = ScxReader::open(path).map_err(to_pyerr)?;
    let x_backed = Arc::new(BackedCsrReader::new(x_reader, cache_shards));
    let x_dataset = match &kept_to_global {
        Some(mapping) => ScxBackedSparseDataset::from_reader_with_deletions(
            Arc::clone(&x_backed),
            cache_shards,
            mapping.clone(),
        ),
        None => ScxBackedSparseDataset::from_reader(Arc::clone(&x_backed), cache_shards),
    };

    // --- obs (eager, filtered by deletion vectors) ---
    let obs = match reader.read_obs() {
        Ok(batch) => {
            let filtered_batch = filter_obs_by_deletion_vectors(&reader, batch)?;
            let table = record_batch_to_pyarrow(py, &filtered_batch)?;
            Some(pyarrow_table_to_pandas(&table)?)
        }
        Err(scx_format::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    // --- var (eager) ---
    let var = match reader.read_var() {
        Ok(batch) => {
            let table = record_batch_to_pyarrow(py, &batch)?;
            Some(pyarrow_table_to_pandas(&table)?)
        }
        Err(scx_format::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    // --- obsm (eager) ---
    let obsm_map = match reader.read_all_obsm() {
        Ok(map) => map,
        Err(scx_format::ScxError::SectionNotFound(_)) => std::collections::HashMap::new(),
        Err(e) => return Err(to_pyerr(e)),
    };
    let obsm_dict = pyo3::types::PyDict::new(py);
    for (name, batch) in &obsm_map {
        let np_arr = obsm_batch_to_numpy(py, batch)?;
        obsm_dict.set_item(name, np_arr)?;
    }

    // --- uns (eager) ---
    let uns_dict = match reader.read_uns() {
        Ok(json_val) => {
            let json_str = serde_json::to_string(&json_val)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let json_mod = py.import("json")?;
            Some(json_mod.call_method1("loads", (json_str,))?)
        }
        Err(scx_format::ScxError::SectionNotFound(_)) => None,
        Err(e) => return Err(to_pyerr(e)),
    };

    // --- layers (backed) ---
    let layer_names = reader.layer_names();
    let layers_dict = pyo3::types::PyDict::new(py);
    for name in &layer_names {
        let l_reader = ScxReader::open(path).map_err(to_pyerr)?;
        let l_backed = Arc::new(BackedCsrReader::new_for_layer(l_reader, name, cache_shards));
        let l_dataset = match &kept_to_global {
            Some(mapping) => ScxBackedLayerDataset::from_reader_with_deletions(
                l_backed,
                cache_shards,
                name.clone(),
                mapping.clone(),
            ),
            None => ScxBackedLayerDataset::from_reader(l_backed, cache_shards, name.clone()),
        };
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

    let adata = anndata_mod.call_method("AnnData", (), Some(&kwargs))?;
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
            if let Some(sd) = dv.shards.iter().find(|sd| sd.shard_id == shard_idx as u32) {
                for local_row in sd.bitmap.iter() {
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
fn filter_obs_by_deletion_vectors(
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
            if let Some(sd) = dv.shards.iter().find(|sd| sd.shard_id == shard_idx as u32) {
                for local_row in sd.bitmap.iter() {
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

/// Detect the best value encoding for f32 data.
///
/// Checks `is_finite()` so that Infinity/NaN fall through to Float32
/// (finding 9.4). Compares max as f64 to avoid precision loss for
/// values > 2^24 (finding 9.1).
pub(crate) fn detect_value_encoding(data: &[f32]) -> ValueEncoding {
    let all_integer = data
        .iter()
        .all(|&v| v.is_finite() && v >= 0.0 && v == v.floor());

    if !all_integer {
        return ValueEncoding::Float32;
    }

    // Compare as f64 to avoid precision loss for values > 2^24 and
    // saturation for values > u32::MAX (finding 9.1).
    let max_val: f64 = data.iter().map(|&v| v as f64).fold(0.0f64, f64::max);

    if max_val <= 255.0 {
        ValueEncoding::Uint8
    } else if max_val <= 65535.0 {
        ValueEncoding::Uint16
    } else {
        ValueEncoding::Uint32
    }
}

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
pub(crate) fn encode_values(data: &[f32], encoding: ValueEncoding) -> Vec<u8> {
    match encoding {
        ValueEncoding::Uint8 => data.iter().map(|&v| v as u8).collect(),
        ValueEncoding::Uint16 => {
            let mut buf = Vec::with_capacity(data.len() * 2);
            for &v in data {
                buf.write_u16::<LittleEndian>(v as u16).unwrap();
            }
            buf
        }
        ValueEncoding::Uint32 => {
            let mut buf = Vec::with_capacity(data.len() * 4);
            for &v in data {
                buf.write_u32::<LittleEndian>(v as u32).unwrap();
            }
            buf
        }
        ValueEncoding::Float32 => {
            let mut buf = Vec::with_capacity(data.len() * 4);
            for &v in data {
                buf.write_f32::<LittleEndian>(v).unwrap();
            }
            buf
        }
        ValueEncoding::Float16 => {
            // Float16 not yet supported — fall back to Float32 (finding 9.1).
            eprintln!("warning: Float16 encoding not supported, falling back to Float32");
            let mut buf = Vec::with_capacity(data.len() * 4);
            for &v in data {
                buf.write_f32::<LittleEndian>(v).unwrap();
            }
            buf
        }
    }
}

/// Parse codec name string to Option<CodecId>.
/// Returns None for auto mode (default), Some(id) for explicit codec.
pub(crate) fn parse_codec(codec: Option<&str>) -> PyResult<Option<CodecId>> {
    match codec {
        None | Some("auto") => Ok(None),
        Some("none") => Ok(Some(CodecId::None)),
        Some("scx1") => Ok(Some(CodecId::Scx1)),
        Some("zstd") => Ok(Some(CodecId::Zstd)),
        Some(other) => Err(PyRuntimeError::new_err(format!(
            "Unknown codec: '{}'. Use 'auto', 'none', 'scx1', or 'zstd'.",
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

    // Decode in Rust
    let cursor = Cursor::new(bytes.to_vec());
    let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let batch = reader
        .into_iter()
        .next()
        .ok_or_else(|| PyRuntimeError::new_err("Arrow IPC contains no batches"))?
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(batch)
}

/// Ensure X is a CSR matrix; convert from dense or CSC if needed.
pub(crate) fn ensure_csr<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let scipy_sparse = py.import("scipy.sparse")?;
    let is_sparse = scipy_sparse
        .call_method1("issparse", (x,))?
        .extract::<bool>()?;

    if !is_sparse {
        // Dense → CSR
        return scipy_sparse.call_method1("csr_matrix", (x,));
    }

    let format: String = x.getattr("format")?.extract()?;
    if format == "csr" {
        // Already CSR, ensure canonical form
        let result = x.call_method0("sorted_indices")?;
        Ok(result)
    } else {
        // CSC or other → CSR
        x.call_method0("tocsr")
    }
}

/// Implementation of from_anndata: extract data from AnnData and write SCX.
pub fn from_anndata_impl(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    path: &str,
    codec: Option<&str>,
    shard_size: Option<u32>,
) -> PyResult<()> {
    let explicit_codec = parse_codec(codec)?;
    let shard_target_rows = shard_size.unwrap_or(16384);

    // Extract X as CSR
    let x = adata.getattr("X")?;
    let x_csr = ensure_csr(py, &x)?;

    // Get shape
    let shape: (u64, u64) = x_csr.getattr("shape")?.extract()?;
    let n_obs = shape.0;
    let n_vars = shape.1;

    // Extract CSR arrays — ensure proper dtypes
    let np = py.import("numpy")?;

    let indptr_obj = x_csr.getattr("indptr")?;
    let indptr_arr = indptr_obj.call_method1("astype", (np.getattr("int64")?,))?;
    let indptr: PyReadonlyArray1<'_, i64> = indptr_arr.extract()?;
    let indptr_slice = indptr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let indices_obj = x_csr.getattr("indices")?;
    let indices_arr = indices_obj.call_method1("astype", (np.getattr("int32")?,))?;
    let indices: PyReadonlyArray1<'_, i32> = indices_arr.extract()?;
    let indices_slice = indices
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let data_obj = x_csr.getattr("data")?;
    let data_arr = data_obj.call_method1("astype", (np.getattr("float32")?,))?;
    let data: PyReadonlyArray1<'_, f32> = data_arr.extract()?;
    let data_slice = data
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let nnz = data_slice.len() as u64;

    // Detect value encoding
    let value_encoding = detect_value_encoding(data_slice);

    // Encode values to raw bytes
    let values_bytes = encode_values(data_slice, value_encoding);

    // Determine effective codec: auto-select or use explicit
    let effective_codec = match explicit_codec {
        Some(codec_id) => {
            // Explicit codec — fall back to Zstd if Scx1 on floats
            if codec_id == CodecId::Scx1 && !value_encoding.is_integer() {
                CodecId::Zstd
            } else {
                codec_id
            }
        }
        None => select_codec(&values_bytes, value_encoding),
    };

    // Determine index dtype
    let index_dtype: u8 = if n_vars <= 65535 { 0 } else { 1 };

    // Build FileHeader
    let header = FileHeader {
        magic: MAGIC,
        format_version: 1,
        header_length: 256,
        flags: 0,
        n_obs,
        n_vars,
        nnz,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows,
        codec_id: effective_codec as u8,
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
        reserved: [0u8; 132],
    };

    let mut writer = ScxWriter::new(path, header).map_err(to_pyerr)?;

    // Write obs — always write even for 0-cell datasets to preserve column schema (finding 9.7).
    let obs_df = adata.getattr("obs")?;
    let obs_batch = pandas_to_record_batch(py, &obs_df)?;
    writer.write_obs(&obs_batch).map_err(to_pyerr)?;

    // Write var — always write even for 0-gene datasets to preserve column schema (finding 9.7).
    let var_df = adata.getattr("var")?;
    let var_batch = pandas_to_record_batch(py, &var_df)?;
    writer.write_var(&var_batch).map_err(to_pyerr)?;

    // Write CSR shards
    let n_obs_usize = n_obs as usize;
    let shard_rows = shard_target_rows as usize;

    let mut row_start: usize = 0;
    while row_start < n_obs_usize {
        let row_end = (row_start + shard_rows).min(n_obs_usize);

        // Rebase indptr for this shard (finding 9.2: validate non-negative).
        let base = indptr_slice[row_start];
        if base < 0 {
            return Err(PyRuntimeError::new_err(format!(
                "negative indptr value {base} at row {row_start}"
            )));
        }
        let shard_indptr: Vec<u64> = indptr_slice[row_start..=row_end]
            .iter()
            .map(|&v| {
                if v < base {
                    Err(PyRuntimeError::new_err(format!(
                        "indptr value {v} < base {base} (non-monotonic)"
                    )))
                } else {
                    Ok((v - base) as u64)
                }
            })
            .collect::<PyResult<Vec<u64>>>()?;

        let nnz_start = base as usize;
        let nnz_end = indptr_slice[row_end] as usize;

        // Convert indices from i32 to u32 (finding 9.2: validate non-negative).
        let shard_indices: Vec<u32> = indices_slice[nnz_start..nnz_end]
            .iter()
            .map(|&v| {
                if v < 0 {
                    Err(PyRuntimeError::new_err(format!("negative CSR index {v}")))
                } else {
                    Ok(v as u32)
                }
            })
            .collect::<PyResult<Vec<u32>>>()?;

        // Slice values bytes
        let bw = value_encoding.byte_width();
        let shard_values = &values_bytes[nnz_start * bw..nnz_end * bw];

        writer
            .write_csr_shard(
                &shard_indptr,
                &shard_indices,
                shard_values,
                effective_codec,
                value_encoding,
                row_start as u64,
            )
            .map_err(to_pyerr)?;

        row_start = row_end;
    }

    // Write obsm
    let obsm = adata.getattr("obsm")?;
    let obsm_keys: Vec<String> = py
        .import("builtins")?
        .call_method1("list", (obsm.call_method0("keys")?,))?
        .extract()?;
    for key in &obsm_keys {
        let arr = obsm.call_method1("__getitem__", (key,))?;
        // Wrap numpy array in a pandas DataFrame, then convert to RecordBatch
        let pd = py.import("pandas")?;
        let df = pd.call_method1("DataFrame", (&arr,))?;
        let batch = pandas_to_record_batch(py, &df)?;
        writer.write_obsm(key, &batch).map_err(to_pyerr)?;
    }

    // Write uns
    let uns = adata.getattr("uns")?;
    let uns_len: usize = uns.call_method0("__len__")?.extract()?;
    if uns_len > 0 {
        let json_mod = py.import("json")?;
        let json_str: String = json_mod.call_method1("dumps", (&uns,))?.extract()?;
        let json_val: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse uns JSON: {}", e)))?;
        writer.write_uns(&json_val).map_err(to_pyerr)?;
    }

    // Write layers
    let layers = adata.getattr("layers")?;
    let layer_keys: Vec<String> = py
        .import("builtins")?
        .call_method1("list", (layers.call_method0("keys")?,))?
        .extract()?;
    for layer_name in &layer_keys {
        let layer_x = layers.call_method1("__getitem__", (layer_name,))?;
        let layer_csr = ensure_csr(py, &layer_x)?;

        let l_indptr_obj = layer_csr.getattr("indptr")?;
        let l_indptr_arr = l_indptr_obj.call_method1("astype", (np.getattr("int64")?,))?;
        let l_indptr: PyReadonlyArray1<'_, i64> = l_indptr_arr.extract()?;
        let l_indptr_slice = l_indptr
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let l_indices_obj = layer_csr.getattr("indices")?;
        let l_indices_arr = l_indices_obj.call_method1("astype", (np.getattr("int32")?,))?;
        let l_indices: PyReadonlyArray1<'_, i32> = l_indices_arr.extract()?;
        let l_indices_slice = l_indices
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let l_data_obj = layer_csr.getattr("data")?;
        let l_data_arr = l_data_obj.call_method1("astype", (np.getattr("float32")?,))?;
        let l_data: PyReadonlyArray1<'_, f32> = l_data_arr.extract()?;
        let l_data_slice = l_data
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let l_value_encoding = detect_value_encoding(l_data_slice);
        let l_values_bytes = encode_values(l_data_slice, l_value_encoding);

        // Auto-select codec per layer independently, or use explicit
        let l_effective_codec = match explicit_codec {
            Some(codec_id) => {
                if codec_id == CodecId::Scx1 && !l_value_encoding.is_integer() {
                    CodecId::Zstd
                } else {
                    codec_id
                }
            }
            None => select_codec(&l_values_bytes, l_value_encoding),
        };

        let mut l_row_start: usize = 0;
        let mut shard_idx: u32 = 0;
        while l_row_start < n_obs_usize {
            let l_row_end = (l_row_start + shard_rows).min(n_obs_usize);

            let l_base = l_indptr_slice[l_row_start];
            if l_base < 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "layer '{layer_name}': negative indptr value {l_base} at row {l_row_start}"
                )));
            }
            let l_shard_indptr: Vec<u64> = l_indptr_slice[l_row_start..=l_row_end]
                .iter()
                .map(|&v| {
                    if v < l_base {
                        Err(PyRuntimeError::new_err(format!(
                            "layer '{layer_name}': indptr value {v} < base {l_base} (non-monotonic)"
                        )))
                    } else {
                        Ok((v - l_base) as u64)
                    }
                })
                .collect::<PyResult<Vec<u64>>>()?;

            let l_nnz_start = l_base as usize;
            let l_nnz_end = l_indptr_slice[l_row_end] as usize;

            let l_shard_indices: Vec<u32> = l_indices_slice[l_nnz_start..l_nnz_end]
                .iter()
                .map(|&v| {
                    if v < 0 {
                        Err(PyRuntimeError::new_err(format!(
                            "layer '{layer_name}': negative CSR index {v}"
                        )))
                    } else {
                        Ok(v as u32)
                    }
                })
                .collect::<PyResult<Vec<u32>>>()?;

            let l_bw = l_value_encoding.byte_width();
            let l_shard_values = &l_values_bytes[l_nnz_start * l_bw..l_nnz_end * l_bw];

            writer
                .write_layer_csr_shard(
                    &l_shard_indptr,
                    &l_shard_indices,
                    l_shard_values,
                    l_effective_codec,
                    l_value_encoding,
                    l_row_start as u64,
                    layer_name,
                    shard_idx,
                )
                .map_err(to_pyerr)?;

            l_row_start = l_row_end;
            shard_idx += 1;
        }
    }

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
