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
use scx_format::shard::{BlockIndex, BlockIndexEntry, ShardHeader, SHARD_HEADER_SIZE, SHARD_MAGIC};
use scx_format::{
    compute_shard_stats, select_codec_for_modality, FileHeader, ModalityType, PreEncodedSection,
    ProvenanceEntry, ScxReader, ScxWriter,
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

/// Convert an Arrow RecordBatch (obsm) to a numpy 2D array.
pub(crate) fn obsm_batch_to_numpy<'py>(
    py: Python<'py>,
    batch: &RecordBatch,
) -> PyResult<Bound<'py, PyAny>> {
    let table = record_batch_to_pyarrow(py, batch)?;
    let df = pyarrow_table_to_pandas(&table)?;
    df.getattr("values")
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

        // uns (still loaded from reader)
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

        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("X", x)?;
        kwargs.set_item("obs", obs_df)?;
        kwargs.set_item("var", var_df)?;
        if let Some(uns) = uns_dict {
            kwargs.set_item("uns", uns)?;
        }
        // obsm and layers are not available via QueryResult. Warn if the
        // source file contains them so users know they're being dropped.
        let has_obsm = reader
            .read_all_obsm()
            .map(|m| !m.is_empty())
            .unwrap_or(false);
        let has_layers = !reader.layer_names().is_empty();
        if has_obsm || has_layers {
            let warnings = py.import("warnings")?;
            let mut parts = Vec::new();
            if has_obsm {
                parts.push("obsm");
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
                    parts.join(" or ")
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
    let x_reader = ScxReader::open(path).map_err(to_pyerr)?;
    let has_csc = x_reader.header().has_csc();
    let x_backed = Arc::new(BackedCsrReader::new(x_reader, cache_shards));
    let x_backed_csc: Option<Arc<scx_format::BackedCscReader>> = if has_csc {
        // Open a separate ScxReader for the CSC sidecar (BackedCscReader
        // takes ownership). Header check is cheap; the reader holds a
        // mmap and per-shard catalog, but no shards decode until we
        // actually call read_csc_shard().
        let csc_reader = ScxReader::open(path).map_err(to_pyerr)?;
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
        let l_reader = ScxReader::open(path).map_err(to_pyerr)?;
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

/// Fast upfront validation of CSR arrays (1C.2).
///
/// Single O(nnz) pass checking:
/// - indptr is monotonically non-decreasing with indptr\[0\] >= 0
/// - All indices are non-negative and < n_vars
///
/// When this passes, the shard loop can skip per-element validation.
fn validate_csr_arrays(indptr: &[i64], indices: &[i32], n_vars: u64) -> PyResult<()> {
    if !indptr.is_empty() && indptr[0] < 0 {
        return Err(PyRuntimeError::new_err(format!(
            "negative indptr value {} at position 0",
            indptr[0]
        )));
    }
    for i in 1..indptr.len() {
        if indptr[i] < indptr[i - 1] {
            return Err(PyRuntimeError::new_err(format!(
                "non-monotonic indptr: value {} at position {} < {} at position {}",
                indptr[i],
                i,
                indptr[i - 1],
                i - 1
            )));
        }
    }

    for (i, &idx) in indices.iter().enumerate() {
        if idx < 0 || (idx as u64) >= n_vars {
            return Err(PyRuntimeError::new_err(format!(
                "CSR index {} out of valid range [0, {}) at position {}",
                idx, n_vars, i
            )));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// uns serialization
// ---------------------------------------------------------------------------

/// Recursively normalize a Python value into a `serde_json::Value` so it can
/// be written into the SCX `uns` section. Replaces a previous `json.dumps`
/// call that errored on common AnnData payloads (NumPy arrays/scalars,
/// pandas Index/Series).
///
/// Conversion rules:
/// - `None` → `null`
/// - `bool` → `bool` (checked before `int`, since Python `bool` ⊂ `int`)
/// - `int` → JSON number (i64 or u64; out-of-range integers error)
/// - `float` → JSON number (non-finite values error rather than become null,
///   to avoid silent data loss in scientific metadata)
/// - `str` → string
/// - `dict` → object; non-string keys are stringified via `str(k)`
/// - `list` / `tuple` → array
/// - `bytes` → error (no portable JSON representation)
/// - NumPy scalar (`np.generic`) → recurse on `.item()`
/// - NumPy array (`np.ndarray`) → recurse on `.tolist()` (multi-dim arrays
///   produce nested lists; object arrays are recursed element-wise)
/// - Any other object exposing a callable `.tolist()` → recurse on its
///   result. This covers pandas `Index`, `Series`, and `Categorical`, but
///   also any duck-typed array-like (e.g. third-party tensors). The
///   fallback is intentionally broad — narrowing it would reject
///   legitimate user payloads with no compensating safety win.
/// - Anything else → error naming the offending type and key path
///
/// `key_path` accumulates a Python-style accessor (e.g.
/// `uns['rank_genes_groups']['names'][0]`) for inclusion in error messages.
///
/// `visiting` tracks PyObject identities currently on the recursion stack
/// for container branches (dict / list / tuple / `.tolist()` fallback). A
/// repeat hit means the input contains a cycle (e.g. `d = {}; d["x"] = d`,
/// or a class whose `.tolist()` returns `self`). We raise `ValueError`
/// instead of recursing into a Rust stack overflow — the latter would
/// abort the Python process. NumPy `ndarray.tolist()` always returns a
/// fresh list, so that branch doesn't need tracking. Scalar leaves
/// (int/float/str) aren't tracked either: Python's small-int / interned-
/// string caches share PyObject identity across uses and would produce
/// false positives.
///
/// Note: this is one-way. Round-tripping through SCX converts NumPy arrays
/// to plain Python lists on readback, since the `uns` section stores JSON.
fn normalize_uns_value<'py>(
    obj: &Bound<'py, PyAny>,
    key_path: &str,
    np_generic: &Bound<'py, PyAny>,
    np_ndarray: &Bound<'py, PyAny>,
    visiting: &mut HashSet<usize>,
) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }

    // NumPy scalar / array first: in NumPy 1.x some scalars subclass Python
    // numeric types, so we must dispatch on np.generic before bool/int/float.
    if obj.is_instance(np_generic)? {
        let item = obj.call_method0("item")?;
        return normalize_uns_value(&item, key_path, np_generic, np_ndarray, visiting);
    }
    if obj.is_instance(np_ndarray)? {
        let lst = obj.call_method0("tolist")?;
        return normalize_uns_value(&lst, key_path, np_generic, np_ndarray, visiting);
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
    if !visiting.insert(id) {
        return Err(PyValueError::new_err(format!(
            "uns at {key_path}: circular reference detected"
        )));
    }
    let result = normalize_container(obj, key_path, np_generic, np_ndarray, visiting);
    visiting.remove(&id);
    result
}

/// Container / fallback dispatch — split out so `normalize_uns_value` can
/// wrap it with cycle-tracking insert/remove.
fn normalize_container<'py>(
    obj: &Bound<'py, PyAny>,
    key_path: &str,
    np_generic: &Bound<'py, PyAny>,
    np_ndarray: &Bound<'py, PyAny>,
    visiting: &mut HashSet<usize>,
) -> PyResult<serde_json::Value> {
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            let key_str: String = k.str()?.extract()?;
            let new_path = format!("{key_path}['{key_str}']");
            map.insert(
                key_str,
                normalize_uns_value(&v, &new_path, np_generic, np_ndarray, visiting)?,
            );
        }
        return Ok(serde_json::Value::Object(map));
    }

    if let Ok(lst) = obj.downcast::<PyList>() {
        let mut arr = Vec::with_capacity(lst.len());
        for (i, item) in lst.iter().enumerate() {
            let new_path = format!("{key_path}[{i}]");
            arr.push(normalize_uns_value(
                &item, &new_path, np_generic, np_ndarray, visiting,
            )?);
        }
        return Ok(serde_json::Value::Array(arr));
    }

    if let Ok(tup) = obj.downcast::<PyTuple>() {
        let mut arr = Vec::with_capacity(tup.len());
        for (i, item) in tup.iter().enumerate() {
            let new_path = format!("{key_path}[{i}]");
            arr.push(normalize_uns_value(
                &item, &new_path, np_generic, np_ndarray, visiting,
            )?);
        }
        return Ok(serde_json::Value::Array(arr));
    }

    // Generic fallback: any object exposing a callable `.tolist()`. Covers
    // pandas Index / Series / Categorical and most array-like duck types.
    if let Ok(method) = obj.getattr("tolist") {
        if method.is_callable() {
            let lst = method.call0()?;
            return normalize_uns_value(&lst, key_path, np_generic, np_ndarray, visiting);
        }
    }

    let type_name: String = obj.get_type().getattr("__name__")?.extract()?;
    Err(PyValueError::new_err(format!(
        "uns at {key_path}: cannot serialize {type_name} to JSON; supported types are None, bool, int, float, str, dict, list, tuple, NumPy arrays/scalars, and any object exposing a callable .tolist() (pandas Series/Index/Categorical)"
    )))
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
    let index_dtype_u16 = index_dtype == 0;

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

                // 3. Detect value encoding and encode values
                let shard_data = &data_owned[b.nnz_start..b.nnz_end];
                let shard_value_encoding = detect_value_encoding(shard_data);
                let shard_values_bytes =
                    encode_values(shard_data, shard_value_encoding).map_err(|e| e.to_string())?;

                // 4. Select codec
                let shard_codec = match explicit_codec {
                    Some(codec_id) => {
                        if codec_id == CodecId::Scx1 && !shard_value_encoding.is_integer() {
                            CodecId::Zstd
                        } else {
                            codec_id
                        }
                    }
                    None => select_codec_for_modality(
                        &shard_values_bytes,
                        shard_value_encoding,
                        ModalityType::Rna,
                    ),
                };

                // 5. Encode shard
                let encoded = scx_codec::encode_shard(
                    &shard_indptr,
                    &shard_indices,
                    &shard_values_bytes,
                    shard_codec,
                    shard_value_encoding,
                    index_dtype_u16,
                )
                .map_err(|e| format!("encode_shard failed: {e}"))?;

                // 6. Build block index (Phase 1: single entry covering entire shard)
                let n_major = (shard_indptr.len() - 1) as u32;
                let nnz = *shard_indptr.last().unwrap_or(&0);
                let block_index = BlockIndex {
                    entries: vec![BlockIndexEntry::new(0, n_major, 0, 0, 0, nnz)
                        .map_err(|e| format!("{e}"))?],
                };
                let mut block_index_bytes = Vec::new();
                block_index
                    .write_to(&mut block_index_bytes)
                    .map_err(|e| format!("{e}"))?;

                // 7. Shard-level checksum (8-byte truncated BLAKE3)
                let mut shard_hasher = blake3::Hasher::new();
                shard_hasher.update(&encoded.indptr_bytes);
                shard_hasher.update(&encoded.indices_bytes);
                shard_hasher.update(&encoded.values_bytes);
                shard_hasher.update(&block_index_bytes);
                let shard_hash = shard_hasher.finalize();
                let mut shard_checksum = [0u8; 8];
                shard_checksum.copy_from_slice(&shard_hash.as_bytes()[..8]);

                // 8. Build ShardHeader with relative offsets
                let indptr_rel_offset = SHARD_HEADER_SIZE as u32;
                let indptr_length = encoded.indptr_bytes.len() as u32;
                let indices_rel_offset = indptr_rel_offset + indptr_length;
                let indices_length = encoded.indices_bytes.len() as u32;
                let values_rel_offset = indices_rel_offset + indices_length;
                let values_length = encoded.values_bytes.len() as u32;
                let block_index_rel_offset = values_rel_offset + values_length;
                let block_index_length = block_index_bytes.len() as u32;

                let shard_header = ShardHeader {
                    magic: SHARD_MAGIC,
                    shard_format_version: 1,
                    shard_type: 0, // CSR
                    codec_id: shard_codec as u8,
                    value_encoding: shard_value_encoding as u8,
                    index_dtype,
                    reserved_flags: [0; 3],
                    n_major,
                    n_minor: n_vars,
                    nnz,
                    global_offset: b.row_start as u64,
                    indptr_rel_offset,
                    indptr_length,
                    indices_rel_offset,
                    indices_length,
                    values_rel_offset,
                    values_length,
                    block_index_rel_offset,
                    block_index_length,
                    checksum: shard_checksum,
                };

                let mut header_buf = Vec::with_capacity(SHARD_HEADER_SIZE);
                shard_header
                    .write_to(&mut header_buf)
                    .map_err(|e| format!("{e}"))?;

                // 9. Section-level checksum (full 32-byte BLAKE3)
                let mut section_hasher = blake3::Hasher::new();
                section_hasher.update(&header_buf);
                section_hasher.update(&encoded.indptr_bytes);
                section_hasher.update(&encoded.indices_bytes);
                section_hasher.update(&encoded.values_bytes);
                section_hasher.update(&block_index_bytes);
                let section_checksum = *section_hasher.finalize().as_bytes();

                let section_length = (header_buf.len()
                    + encoded.indptr_bytes.len()
                    + encoded.indices_bytes.len()
                    + encoded.values_bytes.len()
                    + block_index_bytes.len()) as u64;

                // 10. Compute shard stats. pyscx writes row-major CSR
                // shards exclusively (CSC sidecars are emitted via a
                // separate path, see scx-cli/src/build_csc.rs).
                let stats = compute_shard_stats(
                    &shard_values_bytes,
                    shard_value_encoding,
                    scx_format::MajorAxis::Row,
                    b.row_start as u64,
                    n_major as u64,
                    n_vars as u64,
                    nnz,
                );

                let name = format!("{name_prefix}_shard_{}", b.shard_idx);

                Ok(PreEncodedSection {
                    encoded,
                    block_index_bytes,
                    header_buf,
                    section_checksum,
                    section_length,
                    stats,
                    name,
                    section_type,
                    nnz,
                })
            })
            .collect()
    });

    result.map_err(PyRuntimeError::new_err)
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
        validate_csr_arrays(indptr_slice, indices_slice, n_vars)?;
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

    // 1E.2: Collect uns JSON under GIL.
    // Use a recursive Python-side normalizer so common AnnData payloads
    // (NumPy arrays/scalars, pandas Index/Series/Categorical) survive the
    // JSON boundary instead of erroring out of `json.dumps`.
    let uns = adata.getattr("uns")?;
    let uns_len: usize = uns.call_method0("__len__")?.extract()?;
    let uns_json: Option<serde_json::Value> = if uns_len > 0 {
        let np_generic = np.getattr("generic")?;
        let np_ndarray = np.getattr("ndarray")?;
        let mut visiting = HashSet::new();
        Some(normalize_uns_value(
            &uns,
            "uns",
            &np_generic,
            &np_ndarray,
            &mut visiting,
        )?)
    } else {
        None
    };

    // 1E.2: Write obsm and uns outside GIL (pure Rust)
    py.allow_threads(|| {
        for (key, batch) in &obsm_batches {
            writer.write_obsm(key, batch)?;
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
            validate_csr_arrays(l_indptr_slice, l_indices_slice, n_vars)?;
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
