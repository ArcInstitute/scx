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

use crate::to_pyerr;

/// Memory budget for the convert-time streaming CSR→CSC transpose.
/// Matches scx-cli's convert pipeline. The full CSR matrix already
/// lives in RAM at this point, so this only bounds the per-chunk
/// transpose working set.
const PYSCX_CSC_MEMORY_BYTES: usize = 4 * 1024 * 1024 * 1024;

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
    let csr = scx_sparse::ScxCsr::new_unchecked(
        (n_obs, n_vars),
        indptr.to_vec(),
        indices.to_vec(),
        data.to_vec(),
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

/// Convert an Arrow RecordBatch (obsm/varm) to a numpy 2D array.
pub(crate) fn obsm_batch_to_numpy<'py>(
    py: Python<'py>,
    batch: &RecordBatch,
) -> PyResult<Bound<'py, PyAny>> {
    let table = record_batch_to_pyarrow(py, batch)?;
    let df = pyarrow_table_to_pandas(&table)?;
    df.getattr("values")
}

/// Convert a scipy sparse matrix to a COO Arrow RecordBatch.
///
/// The resulting batch has columns `row: Int32`, `col: Int32`, `data: Float32`
/// (nnz rows) and schema metadata `n_rows` and `n_cols`.  Data is cast to
/// float32; precision is reduced if the source uses float64.
pub(crate) fn sparse_to_coo_record_batch(
    py: Python<'_>,
    mat: &Bound<'_, PyAny>,
) -> PyResult<RecordBatch> {
    use arrow::array::{Float32Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::collections::HashMap;

    let scipy_sparse = py.import("scipy.sparse")?;
    let coo = scipy_sparse.call_method1("coo_matrix", (mat,))?;
    let shape: (usize, usize) = coo.getattr("shape")?.extract()?;
    let np = py.import("numpy")?;

    let row: Vec<i32> = np
        .call_method1("asarray", (coo.getattr("row")?,))?
        .call_method1("astype", ("int32",))?
        .extract()?;
    let col: Vec<i32> = np
        .call_method1("asarray", (coo.getattr("col")?,))?
        .call_method1("astype", ("int32",))?
        .extract()?;
    let data: Vec<f32> = np
        .call_method1("asarray", (coo.getattr("data")?,))?
        .call_method1("astype", ("float32",))?
        .extract()?;

    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("row", DataType::Int32, false),
            Field::new("col", DataType::Int32, false),
            Field::new("data", DataType::Float32, false),
        ],
        HashMap::from([
            ("n_rows".to_string(), shape.0.to_string()),
            ("n_cols".to_string(), shape.1.to_string()),
        ]),
    ));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(row)),
            Arc::new(Int32Array::from(col)),
            Arc::new(Float32Array::from(data)),
        ],
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Convert a COO Arrow RecordBatch back to a scipy.sparse.csr_matrix.
///
/// Reads `row`, `col`, `data` columns and `n_rows`/`n_cols` schema metadata.
pub(crate) fn coo_record_batch_to_scipy<'py>(
    py: Python<'py>,
    batch: &RecordBatch,
) -> PyResult<Bound<'py, PyAny>> {
    use arrow::array::{Float32Array, Int32Array};
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

    let row_arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| PyRuntimeError::new_err("Invalid row column in sparse matrix batch"))?;
    let col_arr = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| PyRuntimeError::new_err("Invalid col column in sparse matrix batch"))?;
    let data_arr = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| PyRuntimeError::new_err("Invalid data column in sparse matrix batch"))?;

    let row_np = PyArray1::from_slice(py, row_arr.values());
    let col_np = PyArray1::from_slice(py, col_arr.values());
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
pub(crate) fn filter_coo_obsp_by_kept_rows(
    batch: &RecordBatch,
    kept_rows: &[u64],
) -> PyResult<RecordBatch> {
    use arrow::array::{Float32Array, Int32Array};
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

    let row_arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| PyRuntimeError::new_err("Invalid row column in sparse matrix batch"))?;
    let col_arr = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| PyRuntimeError::new_err("Invalid col column in sparse matrix batch"))?;
    let data_arr = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| PyRuntimeError::new_err("Invalid data column in sparse matrix batch"))?;

    // Build original→user-visible remaps. -1 means dropped.
    let mut row_remap = vec![-1i32; n_rows];
    for (new_idx, &orig) in kept_rows.iter().enumerate() {
        let orig_usize = orig as usize;
        if orig_usize < n_rows {
            row_remap[orig_usize] = new_idx as i32;
        }
    }
    // For obsp the axes are identical, but n_cols may legitimately differ from
    // n_rows on a malformed file — build the col remap independently.
    let col_remap: Vec<i32> = if n_cols == n_rows {
        row_remap.clone()
    } else {
        let mut r = vec![-1i32; n_cols];
        for (new_idx, &orig) in kept_rows.iter().enumerate() {
            let orig_usize = orig as usize;
            if orig_usize < n_cols {
                r[orig_usize] = new_idx as i32;
            }
        }
        r
    };

    let nnz = row_arr.len();
    let mut new_row: Vec<i32> = Vec::with_capacity(nnz);
    let mut new_col: Vec<i32> = Vec::with_capacity(nnz);
    let mut new_data: Vec<f32> = Vec::with_capacity(nnz);
    let row_vals = row_arr.values();
    let col_vals = col_arr.values();
    let data_vals = data_arr.values();
    for k in 0..nnz {
        let r = row_vals[k];
        let c = col_vals[k];
        if r < 0 || c < 0 {
            continue;
        }
        let r_us = r as usize;
        let c_us = c as usize;
        if r_us >= n_rows || c_us >= n_cols {
            continue;
        }
        let nr = row_remap[r_us];
        let nc = col_remap[c_us];
        if nr >= 0 && nc >= 0 {
            new_row.push(nr);
            new_col.push(nc);
            new_data.push(data_vals[k]);
        }
    }

    let kept_len = kept_rows.len();
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("row", DataType::Int32, false),
            Field::new("col", DataType::Int32, false),
            Field::new("data", DataType::Float32, false),
        ],
        HashMap::from([
            ("n_rows".to_string(), kept_len.to_string()),
            ("n_cols".to_string(), kept_len.to_string()),
        ]),
    ));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(new_row)),
            Arc::new(Int32Array::from(new_col)),
            Arc::new(Float32Array::from(new_data)),
        ],
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Build an AnnData object from an ScxReader.
///
/// When deletion vectors are present, deleted cells are excluded from
/// both the CSR matrix and the obs metadata.
pub fn to_anndata<'py>(py: Python<'py>, reader: &ScxReader) -> PyResult<Bound<'py, PyAny>> {
    to_anndata_with_layers(py, reader, None)
}

/// Build an AnnData object from an ScxReader with optional layer filtering.
fn to_anndata_with_layers<'py>(
    py: Python<'py>,
    reader: &ScxReader,
    layer_filter: Option<&[String]>,
) -> PyResult<Bound<'py, PyAny>> {
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
        let filtered = filter_obs_by_deletion_vectors(reader, batch.clone())?;
        let np_arr = obsm_batch_to_numpy(py, &filtered)?;
        obsm_dict.set_item(name, np_arr)?;
    }

    // varm (dense, var × components — no deletion vector filtering needed)
    let varm_map = match reader.read_all_varm() {
        Ok(map) => map,
        Err(scx_format::ScxError::SectionNotFound(_)) => std::collections::HashMap::new(),
        Err(e) => return Err(to_pyerr(e)),
    };
    let varm_dict = pyo3::types::PyDict::new(py);
    for (name, batch) in &varm_map {
        let np_arr = obsm_batch_to_numpy(py, batch)?;
        varm_dict.set_item(name, np_arr)?;
    }

    // obsp (obs × obs sparse — subset to kept rows when a deletion vector is active)
    let obsp_kept = compute_kept_to_global(reader)?;
    let obsp_map = match reader.read_all_obsp() {
        Ok(map) => map,
        Err(scx_format::ScxError::SectionNotFound(_)) => std::collections::HashMap::new(),
        Err(e) => return Err(to_pyerr(e)),
    };
    let obsp_dict = pyo3::types::PyDict::new(py);
    for (name, batch) in &obsp_map {
        let scipy_mat = if let Some(ref kept) = obsp_kept {
            let filtered = filter_coo_obsp_by_kept_rows(batch, kept)?;
            coo_record_batch_to_scipy(py, &filtered)?
        } else {
            coo_record_batch_to_scipy(py, batch)?
        };
        obsp_dict.set_item(name, scipy_mat)?;
    }

    // varp (var × var sparse — no deletion vector applies to the var axis)
    let varp_map = match reader.read_all_varp() {
        Ok(map) => map,
        Err(scx_format::ScxError::SectionNotFound(_)) => std::collections::HashMap::new(),
        Err(e) => return Err(to_pyerr(e)),
    };
    let varp_dict = pyo3::types::PyDict::new(py);
    for (name, batch) in &varp_map {
        let scipy_mat = coo_record_batch_to_scipy(py, batch)?;
        varp_dict.set_item(name, scipy_mat)?;
    }

    // uns — reconstruct any `__scx_type__` envelopes back into NumPy
    // ndarrays / scalars / tuples / pandas Index/Series/Categorical /
    // structured recarrays. Plain JSON passes through unchanged.
    let uns_dict = read_uns_as_pyobject(py, reader)?;

    // layers (with optional filtering)
    let all_layer_names = reader.layer_names();
    let layers_dict = pyo3::types::PyDict::new(py);
    for name in &all_layer_names {
        // Skip layers not in the filter list (if specified)
        if let Some(filter) = layer_filter {
            if !filter.iter().any(|f| f == name) {
                continue;
            }
        }
        match reader.read_layer_filtered(name) {
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
    if !varm_dict.is_empty() {
        kwargs.set_item("varm", varm_dict)?;
    }
    if !obsp_dict.is_empty() {
        kwargs.set_item("obsp", obsp_dict)?;
    }
    if !varp_dict.is_empty() {
        kwargs.set_item("varp", varp_dict)?;
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
pub fn to_anndata_filtered<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    reader: &ScxReader,
    var_names: Option<&[String]>,
    obs_filter: Option<&str>,
    layer_filter: Option<&[String]>,
    preserve_slots: bool,
) -> PyResult<Bound<'py, PyAny>> {
    // Fast path: no filtering → use existing implementation
    if var_names.is_none() && obs_filter.is_none() && layer_filter.is_none() {
        return to_anndata(py, reader);
    }

    // preserve_slots=true with obs_filter: load full AnnData, then filter
    // rows via pandas.eval. Keeps obsm / layers / uns intact at the cost
    // of skipping query-engine predicate pushdown.
    if let (Some(expr), true) = (obs_filter, preserve_slots) {
        let full = to_anndata_with_layers(py, reader, layer_filter)?;

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

        let adata = anndata_mod.call_method("AnnData", (), Some(&kwargs))?;
        return Ok(adata);
    }

    // No obs_filter but var_names and/or layers specified
    // Load normally, then apply var_names column projection
    let adata = to_anndata_with_layers(py, reader, layer_filter)?;

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
pub fn to_anndata_backed<'py>(
    py: Python<'py>,
    path: &std::path::Path,
    cache_shards: usize,
    var_names: Option<&[String]>,
    obs_filter: Option<&str>,
    layer_filter: Option<&[String]>,
) -> PyResult<Bound<'py, PyAny>> {
    use crate::backed::{ScxBackedLayerDataset, ScxBackedSparseDataset};
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
    let dv_kept_to_global = compute_kept_to_global(&reader)?;
    let mut kept_to_global = dv_kept_to_global.clone();

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

    // --- obsm (eager, filtered by deletion vectors + obs_filter) ---
    // When obs_filter is used, obsm must also be filtered to match obs rows.
    let obsm_map = match reader.read_all_obsm() {
        Ok(map) => map,
        Err(scx_format::ScxError::SectionNotFound(_)) => std::collections::HashMap::new(),
        Err(e) => return Err(to_pyerr(e)),
    };
    let obsm_dict = pyo3::types::PyDict::new(py);
    if obs_filter.is_some() {
        // When obs_filter is present, obsm must be sliced to match kept_to_global
        if let Some(ref kept) = kept_to_global {
            // Pre-compute dv_kept and positions once for all obsm entries
            // (these are loop-invariant — they depend only on kept and deletion vectors)
            let dv_kept = &dv_kept_to_global;
            let positions: Vec<i64> = match &dv_kept {
                Some(dv_mapping) => {
                    // Find position of each kept global row in dv_mapping
                    kept.iter()
                        .filter_map(|&g| {
                            dv_mapping.iter().position(|&dv| dv == g).map(|p| p as i64)
                        })
                        .collect()
                }
                None => kept.iter().map(|&g| g as i64).collect(),
            };

            for (name, batch) in &obsm_map {
                // First filter by deletion vectors
                let filtered = filter_obs_by_deletion_vectors(&reader, batch.clone())?;
                let np_arr = obsm_batch_to_numpy(py, &filtered)?;
                // Slice to the obs_filter rows using pre-computed positions
                let idx_arr = numpy::PyArray1::from_slice(py, &positions);
                let sliced = np_arr.call_method1("__getitem__", (idx_arr,))?;
                obsm_dict.set_item(name, sliced)?;
            }
        }
    } else {
        for (name, batch) in &obsm_map {
            let filtered = filter_obs_by_deletion_vectors(&reader, batch.clone())?;
            let np_arr = obsm_batch_to_numpy(py, &filtered)?;
            obsm_dict.set_item(name, np_arr)?;
        }
    }

    // --- varm (eager, dense — no deletion vector filtering needed) ---
    let varm_map = match reader.read_all_varm() {
        Ok(map) => map,
        Err(scx_format::ScxError::SectionNotFound(_)) => std::collections::HashMap::new(),
        Err(e) => return Err(to_pyerr(e)),
    };
    let varm_dict = pyo3::types::PyDict::new(py);
    for (name, batch) in &varm_map {
        let np_arr = obsm_batch_to_numpy(py, batch)?;
        varm_dict.set_item(name, np_arr)?;
    }

    // --- obsp (eager, obs × obs sparse — subset by the same kept_to_global ---
    // --- mapping that was applied to X and obs above, composing deletion ---
    // --- vector + obs_filter) ---
    let obsp_map = match reader.read_all_obsp() {
        Ok(map) => map,
        Err(scx_format::ScxError::SectionNotFound(_)) => std::collections::HashMap::new(),
        Err(e) => return Err(to_pyerr(e)),
    };
    let obsp_dict = pyo3::types::PyDict::new(py);
    for (name, batch) in &obsp_map {
        let scipy_mat = if let Some(ref kept) = kept_to_global {
            let filtered = filter_coo_obsp_by_kept_rows(batch, kept)?;
            coo_record_batch_to_scipy(py, &filtered)?
        } else {
            coo_record_batch_to_scipy(py, batch)?
        };
        obsp_dict.set_item(name, scipy_mat)?;
    }

    // --- varp (eager, var × var sparse) ---
    let varp_map = match reader.read_all_varp() {
        Ok(map) => map,
        Err(scx_format::ScxError::SectionNotFound(_)) => std::collections::HashMap::new(),
        Err(e) => return Err(to_pyerr(e)),
    };
    let varp_dict = pyo3::types::PyDict::new(py);
    for (name, batch) in &varp_map {
        let scipy_mat = coo_record_batch_to_scipy(py, batch)?;
        varp_dict.set_item(name, scipy_mat)?;
    }

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
    if !varm_dict.is_empty() {
        kwargs.set_item("varm", varm_dict)?;
    }
    if !obsp_dict.is_empty() {
        kwargs.set_item("obsp", obsp_dict)?;
    }
    if !varp_dict.is_empty() {
        kwargs.set_item("varp", varp_dict)?;
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

/// Sentinel key marking a tagged envelope in the on-disk JSON.
const SCX_TYPE_KEY: &str = "__scx_type__";

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
/// `ctx.visiting` tracks PyObject identities currently on the recursion stack
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
    if obj.downcast::<PyBool>().is_ok() {
        return Ok(serde_json::Value::Bool(obj.extract::<bool>()?));
    }

    if obj.downcast::<PyInt>().is_ok() {
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

    if obj.downcast::<PyFloat>().is_ok() {
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

    if let Ok(s) = obj.downcast::<PyString>() {
        return Ok(serde_json::Value::String(s.extract()?));
    }

    if obj.downcast::<PyBytes>().is_ok() {
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
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            let key_str: String = k.str()?.extract()?;
            let new_path = format!("{key_path}['{key_str}']");
            map.insert(key_str, normalize_uns_value(&v, &new_path, ctx)?);
        }
        return Ok(serde_json::Value::Object(map));
    }

    if let Ok(lst) = obj.downcast::<PyList>() {
        let mut arr = Vec::with_capacity(lst.len());
        for (i, item) in lst.iter().enumerate() {
            let new_path = format!("{key_path}[{i}]");
            arr.push(normalize_uns_value(&item, &new_path, ctx)?);
        }
        return Ok(serde_json::Value::Array(arr));
    }

    if let Ok(tup) = obj.downcast::<PyTuple>() {
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
    if let Ok(lst) = obj.downcast::<PyList>() {
        let mut arr = Vec::with_capacity(lst.len());
        for (i, item) in lst.iter().enumerate() {
            let new_path = format!("{key_path}[{i}]");
            arr.push(pylist_to_string_json_array(&item, &new_path)?);
        }
        return Ok(serde_json::Value::Array(arr));
    }
    if let Ok(s) = obj.downcast::<PyString>() {
        return Ok(serde_json::Value::String(s.extract()?));
    }
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if obj.downcast::<PyBytes>().is_ok() {
        // Decode UTF-8 bytes; reject otherwise.
        let b: &[u8] = obj.downcast::<PyBytes>().unwrap().as_bytes();
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
    let lst = descr.downcast::<PyList>().map_err(|_| {
        PyValueError::new_err(format!(
            "uns at {key_path}: structured dtype.descr is not a list"
        ))
    })?;
    let mut out = Vec::with_capacity(lst.len());
    for (i, item) in lst.iter().enumerate() {
        let tup = item.downcast::<PyTuple>().map_err(|_| {
            PyValueError::new_err(format!(
                "uns at {key_path}: dtype.descr[{i}] is not a tuple"
            ))
        })?;
        let mut row = Vec::with_capacity(tup.len());
        for el in tup.iter() {
            if let Ok(s) = el.downcast::<PyString>() {
                row.push(serde_json::Value::String(s.extract()?));
            } else if let Ok(t) = el.downcast::<PyTuple>() {
                // Nested shape tuple, e.g. ('a', '<i4', (3,)).
                let mut inner = Vec::with_capacity(t.len());
                for d in t.iter() {
                    let n: u64 = d.extract()?;
                    inner.push(serde_json::Value::Number(n.into()));
                }
                row.push(serde_json::Value::Array(inner));
            } else if let Ok(l) = el.downcast::<PyList>() {
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
    let pybytes = bytes_obj.downcast::<PyBytes>().map_err(|_| {
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
    if let Ok(s) = obj.downcast::<PyString>() {
        return Ok(serde_json::Value::String(s.extract()?));
    }
    if obj.downcast::<PyBool>().is_ok() {
        return Ok(serde_json::Value::Bool(obj.extract()?));
    }
    if obj.downcast::<PyInt>().is_ok() {
        if let Ok(i) = obj.extract::<i64>() {
            return Ok(serde_json::Value::Number(i.into()));
        }
        if let Ok(u) = obj.extract::<u64>() {
            return Ok(serde_json::Value::Number(u.into()));
        }
    }
    if obj.downcast::<PyFloat>().is_ok() {
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
/// safety, then encodes all shards in parallel under `py.allow_threads()`.
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

    let result: Result<Vec<PreEncodedSection>, String> = py.allow_threads(|| {
        boundaries
            .par_iter()
            .map(|b| {
                // 1. Rebase indptr for this shard
                let shard_indptr: Vec<u64> = if csr_validated {
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
                let shard_indices: Vec<u32> = if csr_validated {
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

                let shard_data = &data_owned[b.nnz_start..b.nnz_end];
                let name = format!("{name_prefix}_shard_{}", b.shard_idx);
                scx_format::encode_one_shard(
                    &shard_indptr,
                    &shard_indices,
                    shard_data,
                    explicit_codec,
                    index_dtype,
                    n_vars,
                    b.row_start as u64,
                    section_type,
                    ModalityType::Rna,
                    name,
                )
                .map_err(|e| e.to_string())
            })
            .collect()
    });

    result.map_err(PyRuntimeError::new_err)
}

/// Forward each non-empty category in `sink` as a single
/// `warnings.warn(..., UserWarning)` call on the Python side.
///
/// The conversion itself runs under `py.allow_threads`, so emission
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
    csc_always: bool,
    csc_cols_per_shard: usize,
    uns_format_parsed: UnsFormat,
) -> PyResult<()> {
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

    // Build the overrides from the in-memory AnnData. Each extraction
    // mirrors the inline logic used by the non-backed path
    // (`pandas_to_record_batch`, `sparse_to_coo_record_batch`,
    // `normalize_uns_value`) so the on-disk SCX output matches what
    // the user sees in Python.
    let obs_override = pandas_to_record_batch(py, &adata.getattr("obs")?)?;
    let var_override = pandas_to_record_batch(py, &adata.getattr("var")?)?;

    let obsm_override = extract_dense_mapping(py, adata, "obsm")?;
    let varm_override = extract_dense_mapping(py, adata, "varm")?;
    let obsp_override = extract_coo_mapping(py, adata, "obsp")?;
    let varp_override = extract_coo_mapping(py, adata, "varp")?;
    let uns_override = extract_uns_value(py, adata, uns_format_parsed)?;

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

    let overrides = scx_convert::StreamingOverrides {
        obs: Some(obs_override),
        var: Some(var_override),
        uns: uns_override,
        obsm: Some(obsm_override),
        varm: Some(varm_override),
        obsp: Some(obsp_override),
        varp: Some(varp_override),
    };

    let opts = scx_convert::ConvertOptions {
        shard_target_rows,
        codec: explicit_codec,
        csc: csc_always,
        csc_cols_per_shard,
        tool: "pyscx".into(),
        ..scx_convert::ConvertOptions::default()
    };
    let input = std::path::PathBuf::from(filename);
    let output = std::path::PathBuf::from(path);

    let mut sink = scx_convert::WarningSink::log();
    py.allow_threads(|| {
        scx_convert::h5ad_to_scx_streaming(&input, &output, &opts, &overrides, &mut sink)
    })
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    emit_python_warnings(py, &sink)?;
    Ok(())
}

/// Helper for the backed-routing path. Reads a dense mapping
/// (`obsm` / `varm`) from a Python AnnData and returns
/// `Vec<(name, RecordBatch)>`. Missing groups → empty Vec.
#[cfg(feature = "hdf5")]
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
#[cfg(feature = "hdf5")]
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
#[cfg(feature = "hdf5")]
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
) -> PyResult<()> {
    let explicit_codec = parse_codec(codec)?;
    let shard_target_rows = shard_size.unwrap_or(16384);
    let csc_always = match csc {
        "off" => false,
        "always" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "invalid csc value '{other}'; expected 'off' or 'always'"
            )))
        }
    };
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
            return route_backed_anndata_to_streaming(
                py,
                adata,
                path,
                explicit_codec,
                shard_target_rows,
                csc_always,
                csc_cols_per_shard,
                uns_format_parsed,
            );
        }
        #[cfg(not(feature = "hdf5"))]
        {
            let _ = (
                explicit_codec,
                csc_always,
                csc_cols_per_shard,
                uns_format_parsed,
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
    let (x_csr, csr_validated) = ensure_csr(py, &x, in_place)?;

    // Get shape
    let shape: (u64, u64) = x_csr.getattr("shape")?.extract()?;
    let n_obs = shape.0;
    let n_vars = shape.1;

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
        let max_axis = if csc_always {
            n_obs.max(n_vars)
        } else {
            n_vars
        };
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

    // 1E.2: Write obs/var outside GIL (pure Rust Arrow IPC serialization + I/O)
    py.allow_threads(|| {
        writer.write_obs(&obs_batch)?;
        writer.write_var(&var_batch)?;
        Ok::<(), scx_format::ScxError>(())
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
    for section in pre_encoded {
        writer.write_preencoded_shard(section).map_err(to_pyerr)?;
    }

    // 1E.2: Collect obsm RecordBatches under GIL
    let obsm = adata.getattr("obsm")?;
    let obsm_keys: Vec<String> = py
        .import("builtins")?
        .call_method1("list", (obsm.call_method0("keys")?,))?
        .extract()?;
    let obsm_batches: Vec<(String, RecordBatch)> = obsm_keys
        .iter()
        .map(|key| {
            let arr = obsm.call_method1("__getitem__", (key,))?;
            let pd = py.import("pandas")?;
            let df = pd.call_method1("DataFrame", (&arr,))?;
            let batch = pandas_to_record_batch(py, &df)?;
            Ok((key.clone(), batch))
        })
        .collect::<PyResult<Vec<_>>>()?;

    // 1E.2: Collect varm RecordBatches under GIL (dense, like obsm).
    // Duck-typed AnnData-likes may omit `varm`/`obsp`/`varp` entirely —
    // missing attrs are treated as empty, matching the obsm contract.
    let varm_batches: Vec<(String, RecordBatch)> = match adata.getattr("varm") {
        Ok(varm) => {
            let varm_keys: Vec<String> = py
                .import("builtins")?
                .call_method1("list", (varm.call_method0("keys")?,))?
                .extract()?;
            varm_keys
                .iter()
                .map(|key| {
                    let arr = varm.call_method1("__getitem__", (key,))?;
                    let pd = py.import("pandas")?;
                    let df = pd.call_method1("DataFrame", (&arr,))?;
                    let batch = pandas_to_record_batch(py, &df)?;
                    Ok((key.clone(), batch))
                })
                .collect::<PyResult<Vec<_>>>()?
        }
        Err(_) => Vec::new(),
    };

    // 1E.2: Collect obsp COO RecordBatches under GIL (obs × obs sparse).
    let obsp_batches: Vec<(String, RecordBatch)> = match adata.getattr("obsp") {
        Ok(obsp) => {
            let obsp_keys: Vec<String> = py
                .import("builtins")?
                .call_method1("list", (obsp.call_method0("keys")?,))?
                .extract()?;
            obsp_keys
                .iter()
                .map(|key| {
                    let mat = obsp.call_method1("__getitem__", (key,))?;
                    let batch = sparse_to_coo_record_batch(py, &mat)?;
                    Ok((key.clone(), batch))
                })
                .collect::<PyResult<Vec<_>>>()?
        }
        Err(_) => Vec::new(),
    };

    // 1E.2: Collect varp COO RecordBatches under GIL (var × var sparse).
    let varp_batches: Vec<(String, RecordBatch)> = match adata.getattr("varp") {
        Ok(varp) => {
            let varp_keys: Vec<String> = py
                .import("builtins")?
                .call_method1("list", (varp.call_method0("keys")?,))?
                .extract()?;
            varp_keys
                .iter()
                .map(|key| {
                    let mat = varp.call_method1("__getitem__", (key,))?;
                    let batch = sparse_to_coo_record_batch(py, &mat)?;
                    Ok((key.clone(), batch))
                })
                .collect::<PyResult<Vec<_>>>()?
        }
        Err(_) => Vec::new(),
    };

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

    // 1E.2: Write obsm, varm, obsp, varp, and uns outside GIL (pure Rust)
    py.allow_threads(|| {
        for (key, batch) in &obsm_batches {
            writer.write_obsm(key, batch)?;
        }
        for (key, batch) in &varm_batches {
            writer.write_varm(key, batch)?;
        }
        for (key, batch) in &obsp_batches {
            writer.write_obsp(key, batch)?;
        }
        for (key, batch) in &varp_batches {
            writer.write_varp(key, batch)?;
        }
        if let Some(ref json_val) = uns_json {
            writer.write_uns(json_val)?;
        }
        Ok::<(), scx_format::ScxError>(())
    })
    .map_err(to_pyerr)?;

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
    if csc_always {
        py.allow_threads(|| -> Result<(), scx_format::ScxError> {
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
