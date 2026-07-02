// PyExperiment — lazy handle for SCX files

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::Array;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use scx_engine::QueryPipeline;
use scx_format_io::backed::BackedCsrReader;
use scx_format_io::ScxReader;

use crate::convert;
use crate::query::PyQueryPipeline;
use crate::to_pyerr;

/// A handle to an open SCX file.
///
/// Provides metadata accessors and methods to convert to AnnData.
#[pyclass(name = "Experiment")]
pub struct PyExperiment {
    reader: ScxReader,
    pub(crate) path: PathBuf,
    /// Memoized deletion-vector popcount (live-count complement). Computed once
    /// on the first `n_obs`/repr access for deletion-bearing files so repeated
    /// access doesn't re-decode the deletion section. `OnceLock` keeps the
    /// pyclass `Send + Sync`.
    n_deleted: std::sync::OnceLock<u64>,
    /// Shared, lazily-opened query pipeline for the grouped-read API
    /// (`read_group` / `read_reference` / `group_labels` / `iter_group_shards`).
    /// Opened once per `Experiment` so a streaming loop parses the catalog and
    /// `group_index` sidecar a single time (combined with the engine-side
    /// `GroupIndex` cache). Shared into each `GroupShard` via `Arc`.
    grouped_pipeline: std::sync::OnceLock<Arc<QueryPipeline>>,
}

impl PyExperiment {
    /// Construct from an already-opened ScxReader and its path (Rust-only).
    pub fn new(reader: ScxReader, path: PathBuf) -> Self {
        Self {
            reader,
            path,
            n_deleted: std::sync::OnceLock::new(),
            grouped_pipeline: std::sync::OnceLock::new(),
        }
    }

    /// The shared grouped-read pipeline, opening it once on first use.
    fn grouped_pipeline(&self) -> PyResult<Arc<QueryPipeline>> {
        if let Some(p) = self.grouped_pipeline.get() {
            return Ok(Arc::clone(p));
        }
        let p = Arc::new(QueryPipeline::open(&self.path).map_err(engine_to_pyerr)?);
        let _ = self.grouped_pipeline.set(p);
        Ok(Arc::clone(
            self.grouped_pipeline
                .get()
                .expect("grouped_pipeline populated above"),
        ))
    }

    /// Resolve the optional `modality` kwarg shared by `uns_keys` and
    /// `read_uns` to a `modality_id`. `None` → 0 (global uns). A name
    /// that does not appear in the modality table raises `KeyError`,
    /// which also covers the "named modality on a non-multimodal file"
    /// case (the modality table is empty there).
    fn resolve_uns_modality(&self, modality: Option<&str>) -> PyResult<u8> {
        match modality {
            None => Ok(0),
            Some(name) => self.reader.modality_id(name).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("unknown modality '{name}'"))
            }),
        }
    }
}

/// Phase 5b: open a fresh `BackedCsrReader` for the requested modality
/// from a file path. `modality = None` → modality_id 0 (the unimodal /
/// global X) on non-multimodal files; on multimodal files we require an
/// explicit modality unless there is exactly one.
fn open_backed_csr(
    path: &PathBuf,
    modality: Option<&str>,
    cache_shards: usize,
) -> PyResult<BackedCsrReader> {
    let opened = ScxReader::open(path).map_err(to_pyerr)?;
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
fn resolve_gene_name(reader: &ScxReader, modality: Option<&str>, name: &str) -> PyResult<u32> {
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
fn lookup_string_in_column(
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

impl PyExperiment {
    /// Live (non-deleted) row count: physical `n_obs` minus the deletion-vector
    /// popcount. Best-effort — falls back to the physical count when the file
    /// has no deletion vectors or the deletion section can't be read, so it
    /// never panics (the `n_obs` getter and repr must always render).
    fn logical_n_obs(&self) -> u64 {
        let physical = self.reader.n_obs();
        if !self.reader.header().has_deletion_vectors() {
            return physical;
        }
        // Decode the deletion section at most once per Experiment (best-effort:
        // a failed read memoizes 0, matching the pre-cache physical fallback).
        let deleted = *self.n_deleted.get_or_init(|| {
            self.reader
                .read_deletion_vectors()
                .ok()
                .flatten()
                .map(|dv| dv.total_deleted())
                .unwrap_or(0)
        });
        physical.saturating_sub(deleted)
    }
}

#[pymethods]
impl PyExperiment {
    /// Number of observations (cells), reflecting **live** rows — i.e. the
    /// physical row count minus any deletion-vector entries. Matches
    /// `to_anndata().n_obs` and `query().count()` after `mark_deleted`. See
    /// [`Self::n_obs_physical`] for the raw, pre-deletion header count.
    #[getter]
    fn n_obs(&self) -> u64 {
        self.logical_n_obs()
    }

    /// Physical (pre-deletion) row count straight from the file header. Equals
    /// [`Self::n_obs`] when the file has no deletion vectors; larger when rows
    /// have been logically deleted via `mark_deleted` (until `compact`).
    #[getter]
    fn n_obs_physical(&self) -> u64 {
        self.reader.n_obs()
    }

    /// Number of variables (genes).
    #[getter]
    fn n_vars(&self) -> u64 {
        self.reader.n_vars()
    }

    /// `(n_obs, n_vars)` — mirrors `anndata.AnnData.shape`. `n_obs` is the
    /// logical (post-deletion) row count.
    #[getter]
    fn shape(&self) -> (u64, u64) {
        (self.logical_n_obs(), self.reader.n_vars())
    }

    /// Total number of non-zero entries.
    #[getter]
    fn nnz(&self) -> u64 {
        self.reader.nnz()
    }

    /// Number of CSR shards in the file.
    #[getter]
    fn shard_count(&self) -> u32 {
        self.reader.header().n_csr_shards
    }

    /// Number of `ObsMetadataShard` sections in the catalog. Zero on
    /// legacy single-section files (`obs` is one `ObsMetadata` section);
    /// `>= 1` on Phase 2 / 4 sharded files written by merge, append, or
    /// `from_anndata` when `n_obs > shard_target_rows`.
    #[getter]
    fn obs_metadata_shard_count(&self) -> usize {
        self.reader.obs_metadata_shard_count()
    }

    /// Number of `VarMetadataShard` sections. Mirror of
    /// [`Self::obs_metadata_shard_count`].
    #[getter]
    fn var_metadata_shard_count(&self) -> usize {
        self.reader.var_metadata_shard_count()
    }

    /// Format version (currently 1).
    #[getter]
    fn format_version(&self) -> u16 {
        self.reader.header().format_version
    }

    /// Filesystem path the experiment was opened from. Returned as a
    /// plain string so it can be passed straight back to converters
    /// like `pyscx.to_h5ad(exp, out)` (the Python wrapper picks this
    /// up via `getattr(source, "path", None)`).
    #[getter]
    fn path(&self) -> String {
        self.path.display().to_string()
    }

    /// File-header codec ID
    /// (`0=None`, `1=Scx1`, `2=Zstd`, `3=Lz4Shuffle`, `4=Pcodec`).
    ///
    /// Per-shard codec overrides are stored in each shard header; readers
    /// must consult `ShardHeader.codec_id` rather than this value when
    /// decoding individual shards.
    #[getter]
    fn codec_id(&self) -> u8 {
        self.reader.header().codec_id
    }

    /// File-header index dtype (`0=u16`, `1=u32`). Used for testing the
    /// SCX → SCX writer's projection-aware index-dtype selection.
    #[getter]
    fn index_dtype(&self) -> u8 {
        self.reader.header().index_dtype
    }

    /// Names of the layers in the file.
    ///
    /// Callable method (`exp.layer_names()`), consistent with the
    /// `obs_keys()` / `var_keys()` / `obsm_keys()` / `varm_keys()` /
    /// `uns_keys()` accessor family (F7).
    fn layer_names(&self) -> Vec<String> {
        self.reader.layer_names()
    }

    /// Column names in `obs` (the cell metadata), excluding the pandas
    /// index column. Pure Arrow IPC footer read — no batch decode.
    /// Raises if the obs section cannot be read (e.g. corrupt file).
    ///
    /// Callable method (e.g. `exp.obs_keys()`) to match AnnData's
    /// `adata.obs_keys()`, not a property.
    fn obs_keys(&self) -> PyResult<Vec<String>> {
        let schema = self.reader.read_obs_schema_physical().map_err(to_pyerr)?;
        Ok(schema_data_columns(Some(schema)))
    }

    /// Column names in `var` (the gene metadata), excluding the pandas
    /// index column. Pure Arrow IPC footer read — no batch decode.
    /// Raises if the var section cannot be read (e.g. corrupt file).
    ///
    /// Callable method (e.g. `exp.var_keys()`) to match AnnData's
    /// `adata.var_keys()`, not a property.
    fn var_keys(&self) -> PyResult<Vec<String>> {
        let schema = self.reader.read_var_schema_physical().map_err(to_pyerr)?;
        Ok(schema_data_columns(Some(schema)))
    }

    /// Keys of the `obsm` cell-embedding mappings. Pure catalog scan.
    /// Callable method (`exp.obsm_keys()`) to match AnnData.
    fn obsm_keys(&self) -> Vec<String> {
        self.reader.list_obsm()
    }

    /// Keys of the `varm` gene-embedding mappings. Pure catalog scan.
    /// Callable method (`exp.varm_keys()`) to match AnnData.
    fn varm_keys(&self) -> Vec<String> {
        self.reader.list_varm()
    }

    /// Top-level keys of the unstructured `uns` mapping. Reads the small
    /// `uns` JSON section but not any matrix payload.
    /// Callable method (`exp.uns_keys()`) to match AnnData.
    ///
    /// On multimodal files, pass `modality=<name>` to read keys from
    /// that modality's `uns/<name>` section instead of the global one.
    /// Unknown modality names raise `KeyError`.
    #[pyo3(signature = (modality=None))]
    fn uns_keys(&self, modality: Option<&str>) -> PyResult<Vec<String>> {
        let modality_id = self.resolve_uns_modality(modality)?;
        match self.reader.read_uns_for(modality_id) {
            Ok(serde_json::Value::Object(map)) => Ok(map.keys().cloned().collect()),
            Ok(_) => Ok(Vec::new()),
            Err(scx_format_io::ScxError::SectionNotFound(_)) => Ok(Vec::new()),
            Err(e) => Err(to_pyerr(e)),
        }
    }

    /// The full unstructured `uns` mapping as a Python dict, with tagged
    /// envelopes reconstructed into NumPy arrays / pandas types (same
    /// reconstruction `to_anndata` applies to `adata.uns`). Returns
    /// `None` when the file has no `uns` section.
    ///
    /// Reads only the small `uns` JSON section (stored in the root
    /// catalog area) — does NOT touch obs, var, obsm, or X. Safe to call
    /// on multi-hundred-GB atlases where `to_anndata()` would OOM on
    /// obs materialization.
    ///
    /// On multimodal files, pass `modality=<name>` to read that
    /// modality's `uns/<name>` section instead of the global one.
    /// Unknown modality names raise `KeyError`.
    #[pyo3(signature = (modality=None))]
    fn read_uns<'py>(
        &self,
        py: Python<'py>,
        modality: Option<&str>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let modality_id = self.resolve_uns_modality(modality)?;
        convert::uns::read_uns_as_pyobject(py, &self.reader, modality_id)
    }

    /// Read the `obs` (cell metadata) table as a pandas DataFrame **without
    /// touching X**. Routes through `ScxReader::read_obs` (full obs) or
    /// `read_obs_keys` (when `columns` is given), so the cost is
    /// `O(obs_metadata_bytes)`, not `O(X_bytes)`.
    ///
    /// `columns` selects a subset by **physical** column name (matching
    /// `obs_keys()`); projecting avoids materialising unselected columns. The
    /// pandas index column (cell barcodes) is always retained regardless of
    /// `columns`, so a projected frame keeps the same index as the unprojected
    /// `read_obs()`.
    ///
    /// Note: on atlas-scale files with many obs shards this still assembles
    /// the full obs table across shards. For enumerating the distinct values
    /// of a single categorical column, prefer `distinct_values()`, which scans
    /// per-shard dictionaries and never assembles the whole table.
    ///
    /// dtype caveat: for plain (non-categorical) string columns the projected
    /// path may return pandas `category` dtype (the projection dictionary-
    /// encodes key columns), whereas the cloud `CloudExperiment.read_obs` and
    /// the unprojected `read_obs()` return `object`. Compare values, not dtype.
    #[pyo3(signature = (columns=None))]
    fn read_obs<'py>(
        &self,
        py: Python<'py>,
        columns: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Decode off the GIL; only the pyarrow/pandas conversion needs Python.
        let batch = py
            .detach(|| match columns {
                Some(cols) => {
                    // Retain the pandas index column(s) so the projected frame
                    // keeps its barcode index (parity with unprojected
                    // read_obs). Including the index column also prevents a
                    // pyarrow KeyError when the schema's pandas envelope still
                    // advertises an `index_columns` entry the projection would
                    // otherwise drop (e.g. `scx convert`-produced files).
                    let schema = self.reader.read_obs_schema_physical()?;
                    let mut proj: Vec<String> = Vec::new();
                    for idx_col in scx_format_io::pandas_index_columns(&schema) {
                        if schema.index_of(&idx_col).is_ok() && !cols.contains(&idx_col) {
                            proj.push(idx_col);
                        }
                    }
                    proj.extend(cols);
                    self.reader.read_obs_keys(&proj)
                }
                None => self.reader.read_obs(),
            })
            .map_err(to_pyerr)?;
        let table = convert::record_batch_to_pyarrow(py, &batch)?;
        convert::pyarrow_table_to_pandas(&table)
    }

    /// Distinct values of a single **string/categorical** `obs` column,
    /// returned as `(values, has_more)`. Pushes the computation into Rust and
    /// never touches X.
    ///
    /// Fast path: for dictionary-encoded (categorical) columns only the
    /// per-shard dictionary catalogs are scanned — rows are never decoded, so
    /// low-cardinality enumeration is nearly free and the expensive full-obs
    /// assembly is skipped entirely. Plain `Utf8`/`LargeUtf8` columns
    /// (e.g. produced by `append`) take a streaming union-of-distincts across
    /// shards.
    ///
    /// Semantics:
    /// - **Nulls are excluded** from the result.
    /// - **Dictionary values are a superset:** a non-compact dictionary may
    ///   carry categories that no row references; those are still surfaced.
    /// - `limit` (without `sort`) returns the **first N encountered** distinct
    ///   values; `has_more` is `True` when more exist. With `sort=True`, all
    ///   distinct values are collected, sorted, then truncated to `limit`.
    ///
    /// Raises `ValueError` for non-string columns and (a corrupt-file-class)
    /// error for an unknown column name.
    #[pyo3(signature = (col, *, limit=None, sort=false))]
    fn distinct_values(
        &self,
        py: Python<'_>,
        col: &str,
        limit: Option<usize>,
        sort: bool,
    ) -> PyResult<(Vec<String>, bool)> {
        // The Utf8-streaming path scans every row; release the GIL for it.
        py.detach(|| self.reader.distinct_obs_values(col, limit, sort))
            .map_err(to_pyerr)
    }

    /// Codec / shard / format-version internals as a one-line string.
    ///
    /// The AnnData-style `repr` lists the obs/var/obsm/uns keys a scanpy
    /// user expects; the on-disk encoding details live here instead.
    fn info(&self) -> String {
        let h = self.reader.header();
        format!(
            "SCX file: format_version={}, codec_id={}, index_dtype={}, \
             csr_shards={}, nnz={}, has_csc={}, path={}",
            h.format_version,
            h.codec_id,
            h.index_dtype,
            h.n_csr_shards,
            self.reader.nnz(),
            h.has_csc(),
            self.path.display(),
        )
    }

    /// `True` when the file has a CSC sidecar (gene-major shards).
    #[getter]
    fn has_csc(&self) -> bool {
        self.reader.header().has_csc()
    }

    /// `True` when the file carries logical deletion vectors — i.e. some
    /// rows are marked deleted and will be dropped (row count shrinks) on
    /// `to_anndata` / `to_h5ad` export.
    #[getter]
    fn has_deletions(&self) -> bool {
        self.reader.header().has_deletion_vectors()
    }

    /// Read the provenance chain as a list of dicts:
    /// `[{"timestamp": int, "action": str, "tool": str,
    ///   "params_json": str, "input_checksums": list[bytes]}]`.
    ///
    /// `params_json` is the raw JSON string written by the producer.
    /// Use `json.loads(entry["params_json"])` to decode. Returns an
    /// empty list when the file has no `provenance` section.
    fn provenance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        use pyo3::types::{PyDict, PyList};
        let list = PyList::empty(py);
        let prov = match self.reader.read_provenance() {
            Ok(p) => p,
            Err(scx_format_io::ScxError::SectionNotFound(_)) => return Ok(list),
            Err(e) => return Err(to_pyerr(e)),
        };
        for entry in prov.operations {
            let d = PyDict::new(py);
            d.set_item("timestamp", entry.timestamp)?;
            d.set_item("action", entry.action)?;
            d.set_item("tool", entry.tool)?;
            d.set_item("params_json", entry.params_json)?;
            let checksums = PyList::empty(py);
            for c in &entry.input_checksums {
                checksums.append(pyo3::types::PyBytes::new(py, c))?;
            }
            d.set_item("input_checksums", checksums)?;
            list.append(d)?;
        }
        Ok(list)
    }

    /// Create a new query pipeline for this file.
    ///
    /// Returns a `PyQueryPipeline` builder — call `.filter_obs()`,
    /// `.select_genes()`, etc., then `.collect()` to execute.
    ///
    /// Example:
    ///     result = pyscx.open("data.scx").query().collect()
    fn query(&self) -> PyResult<PyQueryPipeline> {
        let pipeline = QueryPipeline::open(&self.path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyQueryPipeline::from_pipeline(pipeline))
    }

    /// F2: read exactly the cells of one `group_by` label as an AnnData.
    ///
    /// Grouped archive only (written with `scx sort --group-by`). Raises
    /// `KeyError` (carrying close matches) for an unknown label, `ValueError`
    /// if the archive is not grouped. Output matches
    /// `query().collect().to_anndata()`.
    ///
    /// Example:
    ///     adata = pyscx.open("screen.scx").read_group("MYC")
    fn read_group<'py>(&self, py: Python<'py>, label: &str) -> PyResult<Bound<'py, PyAny>> {
        let pipeline = self.grouped_pipeline()?;
        let result = py
            .detach(|| pipeline.read_group(label))
            .map_err(engine_to_pyerr)?;
        crate::query::query_result_to_anndata(py, result)
    }

    /// F2: read the reference cells (e.g. "non-targeting") as an AnnData, or
    /// `None` if the archive has no reference rows. Returns the full reference
    /// region (all leading reference shards).
    fn read_reference<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let pipeline = self.grouped_pipeline()?;
        let result = py
            .detach(|| pipeline.read_reference())
            .map_err(engine_to_pyerr)?;
        match result {
            Some(qr) => Ok(Some(crate::query::query_result_to_anndata(py, qr)?)),
            None => Ok(None),
        }
    }

    /// F2: distinct group labels present in the archive.
    fn group_labels(&self) -> PyResult<Vec<String>> {
        let pipeline = self.grouped_pipeline()?;
        pipeline.group_labels().map_err(engine_to_pyerr)
    }

    /// F2: one `GroupShard` per non-reference shard, for streaming reads that
    /// keep ~one shard resident.
    ///
    /// Example:
    ///     for gs in pyscx.open("screen.scx").iter_group_shards():
    ///         adata = gs.to_anndata()
    fn iter_group_shards(&self) -> PyResult<Vec<PyGroupShard>> {
        let pipeline = self.grouped_pipeline()?;
        let handles = pipeline.iter_group_shards().map_err(engine_to_pyerr)?;
        Ok(handles
            .into_iter()
            .map(|handle| PyGroupShard::new(Arc::clone(&pipeline), handle))
            .collect())
    }

    /// Mark cells as logically deleted using a boolean mask.
    ///
    /// The mask should be a boolean numpy array whose length matches n_obs.
    /// Cells where the mask is True are marked as deleted.
    /// Returns the total number of deleted cells (including previously deleted).
    ///
    /// Example:
    ///     exp = pyscx.open("experiment.scx")
    ///     adata = exp.to_anndata()
    ///     total = exp.mark_deleted(adata.obs["is_doublet"] == True)
    fn mark_deleted(&mut self, mask: PyReadonlyArray1<'_, bool>) -> PyResult<u64> {
        let mask_slice = mask
            .as_slice()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let expected = self.reader.n_obs() as usize;
        if mask_slice.len() != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "mask length {} does not match n_obs {}",
                mask_slice.len(),
                expected
            )));
        }

        // Collect indices where mask is True
        let indices: Vec<u64> = mask_slice
            .iter()
            .enumerate()
            .filter(|(_, &v)| v)
            .map(|(i, _)| i as u64)
            .collect();

        let total = scx_ops::mark_deleted(&self.path, &indices)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Re-open reader so subsequent reads see the updated file (finding 9.6).
        self.reader = ScxReader::open(&self.path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Invalidate caches that snapshot the pre-deletion file. The grouped
        // pipeline holds its own ScxReader + deletion-vector snapshot, so a
        // grouped read populated before this mutation would otherwise still
        // return just-deleted rows; the cached deleted-count is likewise stale.
        self.grouped_pipeline = std::sync::OnceLock::new();
        self.n_deleted = std::sync::OnceLock::new();

        Ok(total)
    }

    /// Convert this SCX file to an AnnData object.
    ///
    /// Args:
    ///     backed: If True, X and layers are wrapped in ScxBackedSparseDataset
    ///             for on-demand shard decoding (read-only).
    ///     cache_shards: Number of decoded shards to LRU-cache (default 4).
    ///                   Only used when backed=True.
    ///     var_names: Optional list of gene names to project to at load time.
    ///                Only loads the specified genes (a set selector by default;
    ///                see preserve_var_order). By default the returned gene axis
    ///                is in sorted original-column order; pass
    ///                preserve_var_order=True to follow the request order instead.
    ///     preserve_var_order: When True, the var/X gene axis follows the order
    ///                of `var_names` (duplicates dropped, first occurrence wins)
    ///                rather than sorted column order. Default False. Works on the
    ///                eager and backed paths; not supported by highly_variable_genes
    ///                on the resulting backed dataset.
    ///     strict_var_names: When True (default), any name in `var_names` that is
    ///                absent from the var metadata raises KeyError. Pass False to
    ///                silently drop unknown names (the pre-0.8.6 behaviour).
    ///     obs_filter: Optional predicate expression (e.g., "cell_type == 'T cell'")
    ///                 to filter observations. Uses predicate pushdown for shard skipping.
    ///     layers: Optional list of layer names to load. If None, all layers are loaded.
    ///     obsm: Optional list of obsm keys to load. If None (default), all
    ///           obsm embeddings are loaded (byte-identical to prior behaviour).
    ///           When set, only the listed keys are read — dropping the
    ///           per-process RAM of unused embeddings on the random-access
    ///           dataloader path. An unknown key raises KeyError.
    ///     preserve_slots: When True together with obs_filter in non-backed mode,
    ///                     materialize obsm and layers after filtering instead of
    ///                     dropping them. Skips query-engine predicate pushdown for X
    ///                     reads; the obs_filter expression is parsed by pandas.eval
    ///                     instead of the SCX predicate engine, so syntax must be
    ///                     pandas-compatible. No effect when obs_filter is None or
    ///                     when backed=True (backed mode already preserves slots).
    ///     modality: Select a single modality of a multimodal file
    ///                     and return a backed AnnData scoped to that modality
    ///                     (per-modality X, var, and obsm; the global obs is
    ///                     shared). Currently requires `backed=True`. The
    ///                     filter kwargs (`var_names`, `obs_filter`, `layers`)
    ///                     are not supported in this mode — use
    ///                     `scx subset --modality NAME --filter ...` to
    ///                     materialise a filtered single-modality file first.
    ///     eager: When False (default), `obsp`, `varp`, `varm`, and (in the
    ///            non-backed path) `layers` are returned as lazy bridges
    ///            that decode each entry on first access. Keeps peak
    ///            RSS of `to_anndata()` itself bounded for files that
    ///            carry large kNN graphs / embeddings; downstream code
    ///            that touches these slots pays the same one-time
    ///            decode cost it would otherwise pay at construction
    ///            time. Pass `eager=True` to materialize everything up
    ///            front and detach the returned AnnData from the SCX
    ///            file handle (e.g., before closing the experiment or
    ///            handing the AnnData to a subprocess).
    ///     memory_budget: Optional budget for the eager non-backed
    ///            assembly path. Accepts `None` (default 8 GiB), an int
    ///            byte count, or a string like `"4G"` / `"512MiB"`.
    ///            When the catalog-only estimate of the assembled
    ///            `X` + indptr + obs/var bytes exceeds the budget,
    ///            `to_anndata()` emits a `UserWarning` recommending
    ///            `backed=True` or `pyscx.open(path).query()`.
    ///            Assembly still proceeds — the warning is advisory.
    ///            Has no effect when `backed=True` (backed mode is
    ///            already memory-bounded) or in the query-engine path
    ///            (`obs_filter` without `preserve_slots`).
    ///
    /// Returns an anndata.AnnData with X, obs, var, and optionally
    /// obsm, uns, and layers populated from the file.
    #[pyo3(signature = (backed=false, cache_shards=4, var_names=None, obs_filter=None, layers=None, preserve_slots=false, modality=None, eager=false, memory_budget=None, obsm=None, preserve_var_order=false, strict_var_names=true))]
    #[allow(clippy::too_many_arguments)]
    fn to_anndata<'py>(
        &self,
        py: Python<'py>,
        backed: bool,
        cache_shards: usize,
        var_names: Option<Vec<String>>,
        obs_filter: Option<&str>,
        layers: Option<Vec<String>>,
        preserve_slots: bool,
        modality: Option<String>,
        eager: bool,
        memory_budget: Option<Bound<'_, PyAny>>,
        obsm: Option<Vec<String>>,
        preserve_var_order: bool,
        strict_var_names: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let memory_budget_bytes = convert::parse_memory_budget(memory_budget.as_ref())?;
        if let Some(name) = modality.as_deref() {
            if !backed {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "to_anndata(modality=...) currently requires backed=True; \
                     use to_mudata() for eager multimodal extraction",
                ));
            }
            if var_names.is_some() || obs_filter.is_some() || layers.is_some() || obsm.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "to_anndata(modality=..., backed=True) does not support \
                     var_names / obs_filter / layers / obsm; use \
                     `scx subset --modality NAME --filter ...` to materialise \
                     a filtered single-modality file first",
                ));
            }
            return convert::to_anndata_backed_for_modality(py, &self.path, name, cache_shards);
        }
        if backed {
            convert::to_anndata_backed(
                py,
                &self.path,
                cache_shards,
                var_names.as_deref(),
                obs_filter,
                layers.as_deref(),
                obsm.as_deref(),
                eager,
                preserve_var_order,
                strict_var_names,
            )
        } else {
            let adata = convert::to_anndata_filtered(
                py,
                &self.path,
                &self.reader,
                var_names.as_deref(),
                obs_filter,
                layers.as_deref(),
                obsm.as_deref(),
                preserve_slots,
                eager,
                memory_budget_bytes,
                false,
                preserve_var_order,
                strict_var_names,
            )?;
            // F10: a materialized (in-memory CSR) AnnData drops the on-disk CSC
            // sidecar, so a later GPU DE call silently falls back to the slower
            // gpu_csr_v3 route. Stamp a hint when the source file has a sidecar so
            // the DE op can point the user at `to_anndata(backed=True)` (which
            // preserves the sidecar and engages gpu_csc_v3). See
            // `accel::route::warn_materialized_csc_sidecar`.
            //
            // Authoritative for the file just opened: set the hint when this file
            // has a sidecar, and *remove* any stale flag inherited from a prior
            // round-trip when it does not — so a sidecar-less file can never carry
            // a leftover `True` that would trigger a misleading warning.
            let uns = adata.getattr("uns")?;
            if self.has_csc() {
                uns.set_item("scx_source_has_csc_sidecar", true)?;
            } else if uns.contains("scx_source_has_csc_sidecar").unwrap_or(false) {
                let _ = uns.del_item("scx_source_has_csc_sidecar");
            }
            Ok(adata)
        }
    }

    /// Return a **GPU-resident** AnnData whose `X` is a
    /// `cupyx.scipy.sparse.csr_matrix` decoded onto the device (ACC-RUST-OPT-V4
    /// Phase 1.2). This is the cheapest path from SCX-on-disk to a GPU matrix
    /// rapids-singlecell operates on — `rsc.get.anndata_to_GPU(adata)` is a no-op
    /// on the returned object (no host re-upload).
    ///
    /// **≤VRAM only.** If the matrix would not fit in free device memory this
    /// raises (it does **not** silently OOM or fall back) — use a backed /
    /// streaming workflow (`open(...).to_anndata(backed=True)` + `pyscx.accel.*`)
    /// for the >VRAM regime.
    ///
    /// Accepts the same shaping options as `to_anndata` (`var_names`,
    /// `obs_filter`, `layers`, `obsm`, `preserve_var_order`, `strict_var_names`);
    /// obs/var/obsm/uns/layers are host-resident and `X` is the GPU-resident
    /// matrix. Requires cuPy.
    ///
    /// Ownership: the device buffers are owned by a single SCX-side holder that
    /// the returned AnnData keeps alive (via `X`'s `.base` chain); they are freed
    /// when the AnnData / its `X` is garbage-collected. Chained `rsc.*` ops
    /// allocate their own outputs from cuPy's pool.
    ///
    /// Example:
    ///     adata = pyscx.open("atlas.scx").to_gpu_anndata()
    ///     import rapids_singlecell as rsc
    ///     rsc.pp.pca(adata)            # runs in-VRAM; no host bounce
    #[pyo3(signature = (var_names=None, obs_filter=None, layers=None, obsm=None, device="gpu", memory_budget=None, preserve_var_order=false, strict_var_names=true))]
    #[allow(clippy::too_many_arguments)]
    fn to_gpu_anndata<'py>(
        &self,
        py: Python<'py>,
        var_names: Option<Vec<String>>,
        obs_filter: Option<&str>,
        layers: Option<Vec<String>>,
        obsm: Option<Vec<String>>,
        device: &str,
        memory_budget: Option<Bound<'_, PyAny>>,
        preserve_var_order: bool,
        strict_var_names: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        #[cfg(feature = "gpu")]
        {
            use pyo3::exceptions::{PyRuntimeError, PyValueError};

            // cuPy is the hard requirement — the returned X is cupyx-sparse.
            let cupy_version = crate::accel::gpu::cupy_info(py).ok_or_else(|| {
                PyRuntimeError::new_err(
                    "to_gpu_anndata requires cuPy (cupyx.scipy.sparse). Install the rapids \
                     analysis backend — see docs/gpu-setup.md.",
                )
            })?;
            // Resolve to a concrete GPU ordinal; reject device="cpu".
            let resolved = crate::accel::gpu::resolve_device(device)?;
            let gpu_id = resolved.gpu_id().ok_or_else(|| {
                PyValueError::new_err(
                    "to_gpu_anndata requires a GPU device ('gpu', 'gpu:N', or 'auto' on a GPU host)",
                )
            })?;

            let dev = scx_accel::GpuDevice::new(gpu_id)
                .map_err(|e| PyRuntimeError::new_err(format!("GPU device {gpu_id}: {e}")))?;
            const HEADROOM: f64 = 1.2;
            let memory_budget_bytes = convert::parse_memory_budget(memory_budget.as_ref())?;

            // Fast path: a full-matrix handoff with no row/column reshaping decodes
            // X straight onto the device (no host scipy CSR, no re-upload — the
            // decode→host→re-upload trip Phase 0.2 measured as ~92% of census_1m
            // PCA wall). Any var_names / obs_filter / layer projection, a deletion
            // vector, or a multimodal source falls back to the host-assemble path
            // below — on-device filtered decode is Phase 4 format work.
            let n_csr_shards = self.reader.csr_shard_count_for(0) as usize;
            let fast_path = var_names.is_none()
                && obs_filter.is_none()
                && layers.is_none()
                && !self.reader.is_multimodal()
                && !self.reader.header().has_deletion_vectors()
                && n_csr_shards > 0;

            let (adata, holder, n_rows, n_cols, bytes_uploaded, transfer_mode): (
                Bound<'py, PyAny>,
                crate::accel::gpu_handoff::GpuCsrMatrix,
                usize,
                usize,
                u64,
                &'static str,
            ) = if fast_path {
                // X-less skeleton (obs / var / obsm / uns / layers assembled eagerly;
                // X is assigned after the device decode below).
                let adata = convert::to_anndata_filtered(
                    py,
                    &self.path,
                    &self.reader,
                    None,
                    None,
                    None,
                    obsm.as_deref(),
                    false, // preserve_slots
                    true,  // eager
                    memory_budget_bytes,
                    true,  // skip_x
                    false, // preserve_var_order (fast path: var_names is None)
                    false, // strict_var_names (no names to check)
                )?;

                // Raw shard bytes (borrow the reader's mmap) + a cheap header
                // pre-scan for the VRAM gate and the honest HtoD byte count.
                let mut shard_refs: Vec<&[u8]> = Vec::with_capacity(n_csr_shards);
                let mut shard_metadata: Vec<Option<scx_codec::Scx1DecodeMetadata>> =
                    Vec::with_capacity(n_csr_shards);
                let mut total_rows: usize = 0;
                let mut total_nnz: usize = 0;
                for i in 0..n_csr_shards {
                    let bytes = self
                        .reader
                        .read_raw_csr_shard_bytes_for(0, i)
                        .map_err(|e| PyRuntimeError::new_err(format!("read CSR shard {i}: {e}")))?;
                    let header = scx_format_io::shard::ShardHeader::read_from(
                        &mut std::io::Cursor::new(bytes),
                    )
                    .map_err(|e| PyRuntimeError::new_err(format!("shard {i} header: {e}")))?;
                    total_rows += header.n_major as usize;
                    total_nnz += header.nnz as usize;
                    // Resolve the decode sidecar so Scx1 indices/values decode on the
                    // device from the encoder-emitted offsets (no CPU prescan, no host
                    // bounce). `None` for non-Scx1 / stale / absent — those fall back
                    // to host decode + HtoD inside the assembler, whose returned
                    // DeviceDecodeStats report the real uploaded byte count below.
                    let meta = self.reader.scx1_metadata_for_csr_shard(0, i).map_err(|e| {
                        PyRuntimeError::new_err(format!("shard {i} decode sidecar: {e}"))
                    })?;
                    shard_metadata.push(meta);
                    shard_refs.push(bytes);
                }

                let n_cols: usize = adata.getattr("n_vars")?.extract()?;
                if n_cols > i32::MAX as usize {
                    return Err(PyValueError::new_err(format!(
                        "to_gpu_anndata: n_cols ({n_cols}) exceeds the i32 column-index range; \
                         the cupyx CSR handoff requires i32 indices."
                    )));
                }

                // ≤VRAM pre-flight on the device-resident CSR size (HEADROOM covers
                // the transient single-shard decode buffer during concat).
                let device_bytes = (total_nnz as u64) * 8 + (total_rows as u64 + 1) * 8;
                let (free, total) = dev
                    .free_memory()
                    .map_err(|e| PyRuntimeError::new_err(format!("query free VRAM: {e}")))?;
                if (device_bytes as f64) * HEADROOM > free as f64 {
                    return Err(PyValueError::new_err(format!(
                        "to_gpu_anndata needs ~{:.1} GB device memory for X ({} nnz) but only \
                         {:.1} GB of {:.1} GB is free on GPU {}. This is the >VRAM regime: use a \
                         backed/streaming workflow (open(...).to_anndata(backed=True) + \
                         pyscx.accel.*), not to_gpu_anndata.",
                        device_bytes as f64 / 1e9,
                        total_nnz,
                        free as f64 / 1e9,
                        total as f64 / 1e9,
                        gpu_id,
                    )));
                }

                let (gpu_csr, decode_stats) = scx_accel::decode_csr_shards_to_device_with_metadata(
                    &dev,
                    &shard_refs,
                    &shard_metadata,
                )
                .map_err(|e| PyRuntimeError::new_err(format!("GPU shard assembly failed: {e}")))?;
                let n_rows = gpu_csr.shape.0;
                let holder = crate::accel::gpu_handoff::adopt_device_csr(dev, gpu_csr)?;
                // Honest transfer mode: a genuine fully-in-VRAM Scx1 decode (only the
                // tiny indptr uploaded — dense >=128-nnz FOR-BP rows now decode on
                // device via the BitPacker4x kernel, Task 4.4b) vs a path where some
                // shard still bounced through the host because it is not an Scx1
                // sidecar shard — a non-Scx1 codec or a sidecar-less Scx1 shard.
                // `bytes_uploaded` is the real HtoD total from the decode, not a
                // header estimate.
                let transfer_mode = if decode_stats.fully_device_decoded {
                    "scx_device_decode_gpu"
                } else {
                    "scx_device_handoff_streamed"
                };
                (
                    adata,
                    holder,
                    n_rows,
                    n_cols,
                    decode_stats.host_uploaded_bytes,
                    transfer_mode,
                )
            } else {
                // Host-assemble fallback (filtered / projected / multimodal inputs):
                // the full option surface via the eager path, then a single HtoD.
                let adata = convert::to_anndata_filtered(
                    py,
                    &self.path,
                    &self.reader,
                    var_names.as_deref(),
                    obs_filter,
                    layers.as_deref(),
                    obsm.as_deref(),
                    false, // preserve_slots
                    true,  // eager
                    memory_budget_bytes,
                    false, // skip_x
                    preserve_var_order,
                    strict_var_names,
                )?;

                // Pull X's CSR arrays. scipy may store indptr/indices as int32 when
                // they fit, so coerce to the GpuCsr layout (f32 data / i32 indices /
                // i64 indptr) via astype(copy=False) — a no-op when already correct.
                let np = py.import("numpy")?;
                let f32_ty = np.getattr("float32")?;
                let i32_ty = np.getattr("int32")?;
                let i64_ty = np.getattr("int64")?;
                let x = adata.getattr("X")?;
                let (n_rows, n_cols): (usize, usize) = x.getattr("shape")?.extract()?;
                if n_cols > i32::MAX as usize {
                    return Err(PyValueError::new_err(format!(
                        "to_gpu_anndata: n_cols ({n_cols}) exceeds the i32 column-index range; \
                         the cupyx CSR handoff requires i32 indices."
                    )));
                }
                let astype = |arr: Bound<'py, PyAny>,
                              ty: &Bound<'py, PyAny>|
                 -> PyResult<Bound<'py, PyAny>> {
                    let kw = pyo3::types::PyDict::new(py);
                    kw.set_item("copy", false)?;
                    arr.call_method("astype", (ty,), Some(&kw))
                };
                let data_arr = astype(x.getattr("data")?, &f32_ty)?;
                let indices_arr = astype(x.getattr("indices")?, &i32_ty)?;
                let indptr_arr = astype(x.getattr("indptr")?, &i64_ty)?;
                let data: Vec<f32> = data_arr
                    .extract::<PyReadonlyArray1<f32>>()?
                    .as_slice()?
                    .to_vec();
                let indices: Vec<i32> = indices_arr
                    .extract::<PyReadonlyArray1<i32>>()?
                    .as_slice()?
                    .to_vec();
                let indptr: Vec<i64> = indptr_arr
                    .extract::<PyReadonlyArray1<i64>>()?
                    .as_slice()?
                    .to_vec();

                let bytes_uploaded = (data.len() as u64) * 4
                    + (indices.len() as u64) * 4
                    + (indptr.len() as u64) * 8;
                let (free, total) = dev
                    .free_memory()
                    .map_err(|e| PyRuntimeError::new_err(format!("query free VRAM: {e}")))?;
                if (bytes_uploaded as f64) * HEADROOM > free as f64 {
                    return Err(PyValueError::new_err(format!(
                        "to_gpu_anndata needs ~{:.1} GB device memory for X ({} nnz) but only \
                         {:.1} GB of {:.1} GB is free on GPU {}. This is the >VRAM regime: use a \
                         backed/streaming workflow (open(...).to_anndata(backed=True) + \
                         pyscx.accel.*), not to_gpu_anndata.",
                        bytes_uploaded as f64 / 1e9,
                        indices.len(),
                        free as f64 / 1e9,
                        total as f64 / 1e9,
                        gpu_id,
                    )));
                }

                let holder = crate::accel::gpu_handoff::upload_host_csr(
                    dev, &indptr, &indices, &data, n_rows, n_cols,
                )?;
                (
                    adata,
                    holder,
                    n_rows,
                    n_cols,
                    bytes_uploaded,
                    "scx_device_handoff",
                )
            };

            // Adopt the device buffers into a cupyx CSR and assign as X.
            let holder = Bound::new(py, holder)?;
            let cupy = py.import("cupy")?;
            let cupyx_sparse = py.import("cupyx.scipy.sparse")?;
            let adopt = |method: &str| -> PyResult<Bound<'py, PyAny>> {
                // cupy.asarray adopts the CAI view without copy and sets .base to
                // it, transitively keeping `holder` (and its device memory) alive.
                cupy.call_method1("asarray", (holder.call_method0(method)?,))
            };
            let data_cp = adopt("data")?;
            let indices_cp = adopt("indices")?;
            let indptr_cp = adopt("indptr")?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("shape", (n_rows, n_cols))?;
            kwargs.set_item("copy", false)?;
            let gpu_x = cupyx_sparse.call_method(
                "csr_matrix",
                ((data_cp, indices_cp, indptr_cp),),
                Some(&kwargs),
            )?;
            adata.setattr("X", gpu_x)?;

            // Honest device-handoff metadata (ACC-RUST-OPT-V4 §4.4).
            let mut info = scx_accel::route::AccelExecutionInfo::new(
                scx_accel::route::AccelRoute::GpuCsr,
                scx_accel::route::FallbackReason::None,
            );
            info.transfer_mode = Some(transfer_mode);
            info.device_id = Some(gpu_id);
            info.bytes_uploaded = Some(bytes_uploaded);
            info.cupy_version = Some(cupy_version);
            crate::accel::route::write_accel_route(py, &adata, "to_gpu_anndata", &info)?;

            Ok(adata)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (
                py,
                var_names,
                obs_filter,
                layers,
                obsm,
                device,
                memory_budget,
                preserve_var_order,
                strict_var_names,
            );
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "to_gpu_anndata requires pyscx built with the 'gpu' feature",
            ))
        }
    }

    /// Validate section checksums (and, with `deep`, decode-level integrity).
    ///
    /// With `deep=True`, additionally decodes every sparse shard to verify the
    /// v3 canonical CSR invariant and verifies every decode sidecar; results
    /// are appended with `canonical-csr `/`decode-sidecar ` prefixed names.
    /// Mirrors `scx validate --deep`.
    ///
    /// Returns a list of (section_name, passed) tuples.
    #[pyo3(signature = (deep=false))]
    fn validate(&self, py: Python<'_>, deep: bool) -> PyResult<Vec<(String, bool)>> {
        let mut results = self.reader.validate().map_err(to_pyerr)?;
        if deep {
            // Deep validation re-decodes every shard (CPU-bound, pure Rust) —
            // run it off the GIL so other Python threads aren't blocked.
            py.detach(|| crate::deep_validate_into(&self.reader, &mut results));
        }
        Ok(results)
    }

    /// True if this file is multimodal (Phase B / v2 with
    /// `n_modalities > 0`). Mirrors `header.has_modalities()`.
    #[getter]
    fn is_multimodal(&self) -> bool {
        self.reader.is_multimodal()
    }

    /// Number of registered modalities (0 for v1 files and
    /// single-modality v2 files).
    #[getter]
    fn n_modalities(&self) -> u32 {
        self.reader.n_modalities()
    }

    /// Ordered list of modality names (empty for single-modality
    /// files). The position in the list maps 1:1 to the 1-based
    /// modality_id (`names[i] -> modality_id = i + 1`).
    #[getter]
    fn modality_names(&self) -> Vec<String> {
        self.reader
            .modality_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Resolve a modality name to its 1-based `modality_id`.
    /// Returns `None` for unknown names or single-modality files.
    fn modality_id(&self, name: &str) -> Option<u8> {
        self.reader.modality_id(name)
    }

    /// Per-modality information block for the given 1-based
    /// `modality_id`. Returns a dict with the on-disk fields of
    /// `ModalityInfo` (name, modality_type, default_codec_id,
    /// default_value_encoding, n_vars, nnz, n_csr_shards,
    /// n_csc_shards, flags). Useful for introspection (e.g. asserting
    /// per-modality codec routing in tests).
    fn modality_info<'py>(
        &self,
        py: Python<'py>,
        modality_id: u8,
    ) -> PyResult<Option<Bound<'py, pyo3::types::PyDict>>> {
        use pyo3::types::PyDict;
        let info = match self.reader.modality_info(modality_id) {
            Some(i) => i,
            None => return Ok(None),
        };
        let d = PyDict::new(py);
        d.set_item("name", &info.name)?;
        d.set_item("modality_type", info.modality_type as u8)?;
        d.set_item("default_codec_id", info.default_codec_id)?;
        d.set_item("default_value_encoding", info.default_value_encoding)?;
        d.set_item("n_vars", info.n_vars)?;
        d.set_item("nnz", info.nnz)?;
        d.set_item("n_csr_shards", info.n_csr_shards)?;
        d.set_item("n_csc_shards", info.n_csc_shards)?;
        d.set_item("flags", info.flags.bits())?;
        Ok(Some(d))
    }

    /// Materialise this file as a `mudata.MuData` object.
    ///
    /// Eager mode (default): iterates registered modalities, builds a
    /// scipy-CSR-backed AnnData per modality, and attaches them to a
    /// `MuData(...)` with the shared global obs. Single-modality files
    /// raise `RuntimeError` directing to `to_anndata()`.
    ///
    /// Backed mode (`backed=True`, Phase 6b): each modality's X is wrapped
    /// in `ScxBackedSparseDataset` (per-modality `BackedCsrReader` + CSC
    /// sidecar if present). Single-modality files are wrapped in a
    /// one-modality `MuData` rather than raising.
    #[pyo3(signature = (backed=false, cache_shards=4))]
    fn to_mudata<'py>(
        &self,
        py: Python<'py>,
        backed: bool,
        cache_shards: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        if backed {
            crate::mudata::to_mudata_backed(py, &self.path, &self.reader, cache_shards)
        } else {
            crate::mudata::to_mudata(py, &self.reader)
        }
    }

    /// Per-gene detection counts for **all** genes — `axis="var"` only.
    ///
    /// The first argument is `axis` (a string), **not** a gene list: this
    /// returns the full per-gene array (one count per `var`). For the cells
    /// expressing a *specific* gene, use the companion
    /// `cells_expressing(gene)` instead.
    ///
    /// Only `axis="var"` is supported (per-cell detection counts would require
    /// a transpose). On multimodal files, pass `modality=` to select a
    /// specific modality; otherwise the global X (modality_id = 0) is used.
    ///
    /// Fast path: when bitmap sidecars are present for every CSR
    /// shard, this is O(roaring-cardinality). Otherwise the call
    /// falls back to a full CSR scan.
    ///
    /// Returns a numpy `int64` array of length `n_vars`.
    #[pyo3(signature = (axis = "var", modality = None))]
    fn detection_counts<'py>(
        &self,
        py: Python<'py>,
        axis: &str,
        modality: Option<&str>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        if axis != "var" {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "detection_counts: only axis='var' is supported (got '{axis}')"
            )));
        }
        let backed = open_backed_csr(&self.path, modality, 4)?;
        let counts: Vec<i64> = py
            .detach(|| backed.gene_detection_counts())
            .map_err(to_pyerr)?
            .into_iter()
            .map(|c| c as i64)
            .collect();
        Ok(PyArray1::from_vec(py, counts))
    }

    /// Phase 5b: global row indices of cells expressing the given gene.
    ///
    /// `gene` may be either an integer gene index (`0..n_vars`) or a
    /// string name (looked up against the modality's `var.index`).
    /// Returns a numpy `uint32` array of global row ids.
    #[pyo3(signature = (gene, modality = None))]
    fn cells_expressing<'py>(
        &self,
        py: Python<'py>,
        gene: &Bound<'_, PyAny>,
        modality: Option<&str>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let backed = open_backed_csr(&self.path, modality, 4)?;
        // Resolve gene → gene_idx. Integer fast path; string falls
        // through to a var.index lookup.
        let gene_idx: u32 = if let Ok(idx) = gene.extract::<u32>() {
            idx
        } else {
            let name: String = gene.extract().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "gene must be an integer index or a string name",
                )
            })?;
            resolve_gene_name(&self.reader, modality, &name)?
        };
        let rows = py
            .detach(|| backed.cells_expressing_gene(gene_idx))
            .map_err(to_pyerr)?;
        Ok(PyArray1::from_vec(py, rows))
    }

    /// Gather specific rows as a sparse `scipy.sparse.csr_matrix`, in the
    /// requested order.
    ///
    /// A synchronous sparse gather over the backed reader's `read_rows_with`:
    /// each touched shard is decoded once, and per-row `(indices, data)` slices
    /// are scattered into a request-order CSR. It allocates no intermediate
    /// `ScxCsr` (the data is zero-copy only at the final numpy handoff — each
    /// nnz is copied into the request-order buffers). `rows` may contain
    /// duplicates and need not be sorted; output rows follow `rows` order.
    /// Returns raw-local gene indices (no global-vocab remap). On multimodal
    /// files, pass `modality=`.
    ///
    /// `cache_shards` bounds peak decoded-shard memory for this gather (it is
    /// not a speedup knob: within a single call each shard is decoded exactly
    /// once, so the LRU never serves a repeat hit). A fresh reader is opened
    /// per call — intentional, so the method is fork-safe and stateless; this
    /// is the eval / random-access utility, not the training hot path (use
    /// `SparseCellSetDataset` for that).
    ///
    /// Out-of-range row ids raise `IndexError`. This is a drop-in for the
    /// backed `adata.X[rows]` analysis path and the random-access utility an
    /// `IterableDataset` cannot serve.
    #[pyo3(signature = (rows, modality = None, cache_shards = 4))]
    fn gather_rows_sparse<'py>(
        &self,
        py: Python<'py>,
        rows: PyReadonlyArray1<'_, u64>,
        modality: Option<&str>,
        cache_shards: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let rows = rows.as_slice()?;
        let n_rows = rows.len();
        let backed = open_backed_csr(&self.path, modality, cache_shards)?;
        let n_vars = backed.n_vars();
        let n_obs = backed.n_obs() as u64;

        // Bounds-check up front so out-of-range ids surface as a clean
        // IndexError rather than a generic read error from the gather loop.
        if let Some(&bad) = rows.iter().find(|&&r| r >= n_obs) {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "row index {bad} out of range for {n_obs} cells"
            )));
        }

        // Scatter each row's CSR into its request position (GIL released).
        let mut per_row: Vec<Option<(Vec<i32>, Vec<f32>)>> = (0..n_rows).map(|_| None).collect();
        py.detach(|| {
            backed.read_rows_with(rows, |orig_pos, indices, data| {
                per_row[orig_pos] = Some((indices.to_vec(), data.to_vec()));
                Ok(())
            })
        })
        .map_err(to_pyerr)?;

        // Assemble the CSR in request order.
        let nnz: usize = per_row
            .iter()
            .map(|r| r.as_ref().map_or(0, |(idx, _)| idx.len()))
            .sum();
        let mut indptr: Vec<i64> = Vec::with_capacity(n_rows + 1);
        indptr.push(0);
        let mut indices: Vec<i32> = Vec::with_capacity(nnz);
        let mut data: Vec<f32> = Vec::with_capacity(nnz);
        for row in &per_row {
            if let Some((idx, val)) = row {
                indices.extend_from_slice(idx);
                data.extend_from_slice(val);
            }
            indptr.push(indices.len() as i64);
        }

        let csr = scx_sparse::ScxCsr::new_unchecked((n_rows, n_vars), indptr, indices, data);
        convert::csr_to_scipy(py, csr)
    }

    fn __repr__(&self) -> String {
        // Best-effort: the repr must always render, so a failed schema read
        // degrades to an empty key list here (the public `obs_keys` /
        // `var_keys` getters surface the error loudly instead).
        format_anndata_repr(
            "Experiment",
            self.logical_n_obs(),
            self.reader.n_vars(),
            &[
                (
                    "obs",
                    schema_data_columns(self.reader.read_obs_schema_physical().ok()),
                ),
                (
                    "var",
                    schema_data_columns(self.reader.read_var_schema_physical().ok()),
                ),
                ("uns", self.uns_keys(None).unwrap_or_default()),
                ("obsm", self.obsm_keys()),
                ("varm", self.varm_keys()),
                ("layers", self.layer_names()),
            ],
        )
    }
}

/// Map an `scx_engine::EngineError` to the right Python exception for the
/// grouped-read API: unknown label → `KeyError` (with close matches in the
/// message), not-grouped → `ValueError`, everything else → `RuntimeError`.
pub(crate) fn engine_to_pyerr(e: scx_engine::EngineError) -> PyErr {
    use scx_engine::EngineError as E;
    match e {
        E::UnknownGroupLabel { .. } => pyo3::exceptions::PyKeyError::new_err(e.to_string()),
        E::NotGrouped => pyo3::exceptions::PyValueError::new_err(e.to_string()),
        other => pyo3::exceptions::PyRuntimeError::new_err(other.to_string()),
    }
}

/// F2: a non-reference shard's grouped contents, with deferred I/O.
///
/// Holds a shared `Arc<QueryPipeline>` (cloned from the parent `Experiment`), so
/// streaming over `iter_group_shards()` opens/parses the file once rather than
/// re-opening per shard.
#[pyclass(name = "GroupShard")]
pub struct PyGroupShard {
    pipeline: Arc<QueryPipeline>,
    handle: scx_engine::GroupShardHandle,
}

impl PyGroupShard {
    /// Construct from a shared pipeline + engine handle (Rust-only). Shared by
    /// the local and cloud `iter_group_shards` surfaces.
    pub(crate) fn new(pipeline: Arc<QueryPipeline>, handle: scx_engine::GroupShardHandle) -> Self {
        Self { pipeline, handle }
    }
}

#[pymethods]
impl PyGroupShard {
    /// This shard's index in the grouped layout.
    #[getter]
    fn shard_index(&self) -> u32 {
        self.handle.shard_index
    }

    /// First global output row in this shard (inclusive).
    #[getter]
    fn global_start(&self) -> u64 {
        self.handle.global_start
    }

    /// One past the last global output row in this shard (exclusive).
    #[getter]
    fn global_stop(&self) -> u64 {
        self.handle.global_stop
    }

    /// Labels present in this shard.
    #[getter]
    fn labels(&self) -> Vec<String> {
        self.handle
            .groups
            .iter()
            .map(|(l, _, _)| l.clone())
            .collect()
    }

    /// Per-label **shard-local** `(start, stop)` row ranges as a dict
    /// `{label: (start, stop)}` (offsets relative to this shard's start).
    #[getter]
    fn groups(&self) -> std::collections::HashMap<String, (u64, u64)> {
        self.handle
            .groups
            .iter()
            .map(|(l, ls, le)| (l.clone(), (*ls, *le)))
            .collect()
    }

    /// Read this shard's rows as an AnnData (deferred I/O — decodes only this
    /// shard's range).
    fn to_anndata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let pipeline = Arc::clone(&self.pipeline);
        let (start, stop) = (self.handle.global_start, self.handle.global_stop);
        let result = py
            .detach(|| pipeline.read_row_range(start, stop))
            .map_err(engine_to_pyerr)?;
        crate::query::query_result_to_anndata(py, result)
    }

    /// Read just the cells of `label` within this shard as an AnnData (deferred
    /// I/O — decodes only the label's sub-range). Raises `KeyError` if the label
    /// is not resident in this shard.
    fn read_group<'py>(&self, py: Python<'py>, label: &str) -> PyResult<Bound<'py, PyAny>> {
        let (start, stop) = self.handle.range(label).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!(
                "label '{label}' is not in shard {}",
                self.handle.shard_index
            ))
        })?;
        let pipeline = Arc::clone(&self.pipeline);
        let result = py
            .detach(|| pipeline.read_row_range(start, stop))
            .map_err(engine_to_pyerr)?;
        crate::query::query_result_to_anndata(py, result)
    }

    fn __repr__(&self) -> String {
        format!(
            "GroupShard(shard_index={}, rows={}..{}, labels={})",
            self.handle.shard_index,
            self.handle.global_start,
            self.handle.global_stop,
            self.handle.groups.len()
        )
    }
}
