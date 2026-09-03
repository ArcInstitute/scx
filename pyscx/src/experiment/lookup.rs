//! Gene / column resolution and repr plumbing for `Experiment` —
//! shared with the cloud reader (`crate::cloud`) and the query pipeline.

// PyExperiment — lazy handle for SCX files

use std::path::PathBuf;

use arrow::array::Array;
use numpy::PyArray1;

use scx_format_io::backed::BackedCsrReader;
use scx_format_io::ScxReader;

use super::*;
use crate::to_pyerr;

/// One obs column's `(codes, categories)` as handed to Python: an `int32` numpy
/// array plus the category strings. Shared by the local and cloud accessors.
pub(crate) type PyCategorical<'py> = (Bound<'py, PyArray1<i32>>, Vec<String>);

/// Project a decoded batch down to `cols`, keeping the pandas index column(s).
///
/// The index columns are retained for the same reason `read_obs`'s pushed-down
/// projection retains them: dropping them loses the frame's index, and the
/// schema's pandas envelope still advertises an `index_columns` entry that
/// pyarrow would then fail to resolve.
///
/// An unknown column name is a `KeyError` naming what is available, rather
/// than a silently narrower frame.
pub(crate) fn project_batch_columns(
    batch: &arrow::record_batch::RecordBatch,
    cols: &[String],
) -> PyResult<arrow::record_batch::RecordBatch> {
    let schema = batch.schema();
    let mut indices: Vec<usize> = Vec::new();
    for idx_col in scx_format_io::resolve_index_columns(&schema) {
        if let Ok(i) = schema.index_of(&idx_col) {
            if !cols.contains(&idx_col) {
                indices.push(i);
            }
        }
    }
    for name in cols {
        let i = schema.index_of(name).map_err(|_| {
            // List only the columns a caller could meaningfully ask for. The
            // pandas index column is retained unconditionally and is often an
            // internal name (`__index_level_0__`), so offering it as a
            // suggestion is noise.
            // Same resolver as the retention loop above, so a file with no
            // envelope does not offer `__index_level_0__` as a suggestion
            // here while silently retaining it there.
            let index_cols = scx_format_io::resolve_index_columns(&schema);
            let available = schema
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .filter(|n| !index_cols.iter().any(|ic| ic == n))
                .collect::<Vec<_>>()
                .join(", ");
            pyo3::exceptions::PyKeyError::new_err(format!(
                "column '{name}' not found; available columns: {available}"
            ))
        })?;
        if !indices.contains(&i) {
            indices.push(i);
        }
    }
    batch.project(&indices).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("column projection failed: {e}"))
    })
}

/// Phase 5b: open a fresh `BackedCsrReader` for the requested modality (and,
/// optionally, layer) from a file path. `modality = None` → modality_id 0 (the
/// unimodal / global X) on non-multimodal files; on multimodal files we require
/// an explicit modality unless there is exactly one. `layer = Some(name)`
/// opens that layer's shard family instead of X (`ValueError` when the file
/// has no such layer); layers are not modality-scoped in the backed reader, so
/// `layer=` on a multimodal file is refused rather than guessed.
pub(crate) fn open_backed_csr(
    path: &PathBuf,
    modality: Option<&str>,
    layer: Option<&str>,
    cache_shards: usize,
) -> PyResult<BackedCsrReader> {
    let opened = crate::open_handle_reader(path).map_err(to_pyerr)?;
    if let Some(name) = layer {
        if opened.is_multimodal() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "layer= is not supported on a multimodal file; open the modality with \
                 to_mudata() and index its layer handle instead",
            ));
        }
        let names = opened.layer_names();
        if !names.iter().any(|n| n == name) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "layer '{name}' not found (available: {names:?})"
            )));
        }
        return Ok(BackedCsrReader::new_for_layer(opened, name, cache_shards));
    }
    if !opened.is_multimodal() {
        return Ok(BackedCsrReader::new(opened, cache_shards));
    }
    let modality_id = match modality {
        Some(name) => opened.modality_id(name).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown modality '{name}'"))
        })?,
        None => {
            let names = opened.modality_names();
            if names.len() == 1 {
                opened.modality_id(names[0]).unwrap_or(1)
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "file is multimodal with {} modalities; pass modality=... \
                     (one of {:?})",
                    names.len(),
                    names
                )));
            }
        }
    };
    Ok(BackedCsrReader::for_modality(
        opened,
        modality_id,
        cache_shards,
    ))
}

/// Resolve a gene name against the appropriate modality's `var`.
pub(crate) fn resolve_gene_name(
    reader: &ScxReader,
    modality: Option<&str>,
    name: &str,
) -> PyResult<u32> {
    use scx_format_io::SectionType;
    let modality_id = if reader.is_multimodal() {
        match modality {
            Some(m) => reader.modality_id(m).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("unknown modality '{m}'"))
            })?,
            None => {
                let names = reader.modality_names();
                if names.len() == 1 {
                    reader.modality_id(names[0]).unwrap_or(1)
                } else {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "file is multimodal; pass modality=... to resolve gene name",
                    ));
                }
            }
        }
    } else {
        0
    };
    let var_section_name = if modality_id == 0 {
        "var".to_string()
    } else {
        match reader.modality_info(modality_id) {
            Some(info) => format!("var/{}", info.name),
            None => "var".to_string(),
        }
    };
    let entry = reader
        .catalog()
        .entries
        .iter()
        .find(|e| e.section_type == SectionType::VarMetadata && e.name == var_section_name)
        .ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "var section '{var_section_name}' not found"
            ))
        })?;
    let bytes = reader.section_bytes(entry).map_err(to_pyerr)?;
    let cursor = std::io::Cursor::new(bytes);
    let mut arrow_reader = arrow::ipc::reader::FileReader::try_new(cursor, None)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let batch = arrow_reader
        .next()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("var record batch is empty"))?
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    // Probe column order:
    //   1. Columns named in the Arrow IPC `pandas` schema metadata's
    //      `index_columns` array (the authoritative source from
    //      `Table.from_pandas`, including named indexes like
    //      `var.index.name = "gene_symbols"`).
    //   2. `__index_level_0__` — the canonical pyarrow name for an
    //      unnamed pandas index.
    //   3. The original heuristic list, kept so any files that pre-date
    //      pandas-metadata-aware writes still resolve.
    lookup_gene_in_batch(&batch, name).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err(format!("gene name '{name}' not found in var index"))
    })
}

/// Resolve a single gene name to its row index within a `var`
/// `RecordBatch`, probing the pandas index column(s) first and then a
/// fallback list of conventional gene-id column names. Shared by the
/// local `resolve_gene_name` and the cloud `read_cloud` paths.
pub(crate) fn lookup_gene_in_batch(
    batch: &arrow::record_batch::RecordBatch,
    name: &str,
) -> Option<u32> {
    let pandas_index_cols = scx_format_io::pandas_index_columns(batch.schema().as_ref());
    let fallback_columns = [
        "__index_level_0__",
        "_index",
        "gene_name",
        "feature_name",
        "gene_id",
        "name",
    ];
    let probe = pandas_index_cols
        .iter()
        .map(String::as_str)
        .chain(fallback_columns.iter().copied());
    for col_name in probe {
        if let Some(idx) = lookup_string_in_column(batch, col_name, name) {
            return Some(idx);
        }
    }
    None
}

/// Scan a single string column of a `RecordBatch` for an exact match
/// and return the row index. Handles both `Utf8` (`StringArray`) and
/// `LargeUtf8` (`LargeStringArray`). Returns `None` when the column
/// is absent, has a non-string dtype, or contains no match.
pub(crate) fn lookup_string_in_column(
    batch: &arrow::record_batch::RecordBatch,
    col_name: &str,
    target: &str,
) -> Option<u32> {
    let (idx, _) = batch.schema().column_with_name(col_name)?;
    let col = batch.column(idx);
    if let Some(arr) = col.as_any().downcast_ref::<arrow::array::StringArray>() {
        for i in 0..arr.len() {
            if !arr.is_null(i) && arr.value(i) == target {
                return Some(i as u32);
            }
        }
    }
    if let Some(arr) = col
        .as_any()
        .downcast_ref::<arrow::array::LargeStringArray>()
    {
        for i in 0..arr.len() {
            if !arr.is_null(i) && arr.value(i) == target {
                return Some(i as u32);
            }
        }
    }
    None
}

/// Field names of an Arrow schema, dropping the pandas index column(s)
/// so the result mirrors `adata.obs.columns` / `adata.var.columns`
/// rather than including the `_index` / `__index_level_0__` field.
pub(crate) fn schema_data_columns(schema: Option<arrow::datatypes::Schema>) -> Vec<String> {
    let Some(schema) = schema else {
        return Vec::new();
    };
    let index_cols = scx_format_io::pandas_index_columns(&schema);
    schema
        .fields()
        .iter()
        .map(|f| f.name().to_string())
        .filter(|name| {
            !index_cols.contains(name) && name != "__index_level_0__" && name != "_index"
        })
        .collect()
}

/// Render the AnnData-style `repr` lines shared by `Experiment` and
/// `CloudExperiment`: a header line plus one indented line per non-empty
/// metadata group (`obs: 'a', 'b'`). Mirrors `anndata.AnnData.__repr__`.
pub(crate) fn format_anndata_repr(
    kind: &str,
    n_obs: u64,
    n_vars: u64,
    groups: &[(&str, Vec<String>)],
) -> String {
    let mut out = format!("{kind} object with n_obs × n_vars = {n_obs} × {n_vars}");
    for (label, keys) in groups {
        if keys.is_empty() {
            continue;
        }
        let joined = keys
            .iter()
            .map(|k| format!("'{k}'"))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!("\n    {label}: {joined}"));
    }
    out
}
