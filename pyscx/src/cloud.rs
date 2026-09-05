//! Python bindings for scx-cloud operations.
//!
//! Provides PyO3 wrappers for pull, push, cloud_optimize, explode, and pack.
//! All async operations create a tokio runtime internally.

use std::sync::Arc;

use numpy::PyArray1;
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use scx_engine::QueryPipeline;

use crate::experiment::{engine_to_pyerr, PyGroupShard};
use crate::query::PyQueryPipeline;

/// Map a `scx_cloud::CloudError` to the most appropriate Python exception.
///
/// A missing/wrong path (`CatalogNotFound`) becomes `FileNotFoundError`
/// with the actionable ".scxd/ vs .scx" message instead of a raw,
/// percent-encoded object_store 404. Everything else (auth, network,
/// timeouts) stays a verbose `RuntimeError` — those messages are worth
/// surfacing in full.
fn cloud_to_pyerr(e: scx_cloud::CloudError) -> PyErr {
    match e {
        scx_cloud::CloudError::CatalogNotFound(_) => PyFileNotFoundError::new_err(e.to_string()),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// Build the tokio runtime that backs a long-lived `CloudReader`.
///
/// The runtime services `block_on` calls from rayon worker threads
/// inside `scx_engine::collect`'s parallel shard decode (each cloud
/// shard fetch ends up on a different rayon worker, each blocking on
/// async I/O). Sizing tokio's worker pool to match what rayon would
/// use prevents a starvation case where all rayon workers are blocked
/// on tokio tasks that have no executor.
fn build_cloud_runtime() -> std::io::Result<tokio::runtime::Runtime> {
    let worker_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(4);
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .enable_all()
        .thread_name("scx-cloud-io")
        .build()
}

/// Pull from cloud/local exploded .scxd into a local .scx file.
///
/// Args:
///     source: Source URL or path (e.g., "gs://bucket/data.scxd/")
///     dest: Output .scx file path
///     filter: Optional predicate expression for selective pull
///         (e.g., ``"n_counts > 1000"``). When ``None``, issues a full,
///         unfiltered pull — the function does NOT raise on missing
///         filter. Callers that want an error on missing predicates
///         must guard at the callsite. Returned dict shape differs
///         between the filtered and unfiltered branches (see below).
///     parallelism: Number of parallel download tasks (default: 8)
///     filter_mode: ``"shard"`` (default) or ``"exact"``.  ``"shard"``
///         downloads complete shards containing any matching cell —
///         the output may include non-matching cells from partially
///         matching shards.  ``"exact"`` is reserved for a future
///         release and currently raises ``RuntimeError``.
///
/// Returns:
///     When ``filter=None`` (full pull): dict with keys
///     ``bytes_downloaded``, ``sections_downloaded``, ``elapsed_secs``,
///     ``throughput_mbps``.
///
///     When ``filter`` is set (selective pull): dict with keys
///     ``total_shards``, ``downloaded_shards``, ``skipped_shards``,
///     ``matching_cells``, ``bytes_downloaded``, ``bytes_saved``,
///     ``elapsed_secs``, ``filter_mode``, ``omitted_section_types``.
///     No ``throughput_mbps`` key — selective pulls pair byte counts
///     with ``bytes_saved`` for shard-skip accounting.
///
/// Note:
///     Benchmark callers (``cost_model``, ``cloud_reader_vs_pull``)
///     rely on the ``filter=None`` → full-pull behavior as a
///     pass-through, and detect "no predicate synthesized" upstream
///     of this call rather than inside it. See
///     ``benchmarks/comprehensive/benchmarks/cloud_reader_vs_pull.py``
///     for the skip-when-None pattern.
#[pyfunction]
#[pyo3(signature = (source, dest, filter=None, parallelism=None, filter_mode=None))]
pub fn pull(
    py: Python<'_>,
    source: &str,
    dest: &str,
    filter: Option<&str>,
    parallelism: Option<usize>,
    filter_mode: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let mode = match filter_mode.unwrap_or("shard") {
        "shard" => scx_cloud::FilterMode::Shard,
        "exact" => scx_cloud::FilterMode::Exact,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid filter_mode: '{other}'; expected 'shard' or 'exact'"
            )));
        }
    };

    let opts = scx_cloud::PullOptions {
        parallelism: parallelism.unwrap_or(8),
        cloud_ready: true,
        filter_mode: mode,
        retry_config: scx_cloud::RetryConfig::default(),
    };

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to create runtime: {e}")))?;

    let dest_path = std::path::PathBuf::from(dest);

    // Release the GIL during blocking cloud I/O (finding 9.3).
    if let Some(filter_expr) = filter {
        let source = source.to_string();
        let filter_expr = filter_expr.to_string();
        let stats = py
            .detach(|| {
                rt.block_on(scx_cloud::pull_filtered(
                    &source,
                    &dest_path,
                    &filter_expr,
                    opts,
                ))
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("total_shards", stats.total_shards)?;
        dict.set_item("downloaded_shards", stats.downloaded_shards)?;
        dict.set_item("skipped_shards", stats.skipped_shards)?;
        dict.set_item("matching_cells", stats.matching_cells)?;
        dict.set_item("bytes_downloaded", stats.bytes_downloaded)?;
        dict.set_item("bytes_saved", stats.bytes_saved)?;
        dict.set_item("elapsed_secs", stats.elapsed.as_secs_f64())?;
        dict.set_item(
            "filter_mode",
            match stats.filter_mode {
                scx_cloud::FilterMode::Shard => "shard",
                scx_cloud::FilterMode::Exact => "exact",
            },
        )?;
        let omitted: Vec<String> = stats
            .omitted_section_types
            .iter()
            .map(|st| format!("{st:?}"))
            .collect();
        dict.set_item("omitted_section_types", omitted)?;
        Ok(dict.into())
    } else {
        let source = source.to_string();
        let stats = py
            .detach(|| rt.block_on(scx_cloud::pull(&source, &dest_path, opts)))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let dict = PyDict::new(py);
        dict.set_item("bytes_downloaded", stats.bytes_downloaded)?;
        dict.set_item("sections_downloaded", stats.sections_downloaded)?;
        dict.set_item("elapsed_secs", stats.elapsed.as_secs_f64())?;
        dict.set_item("throughput_mbps", stats.throughput_mbps)?;
        Ok(dict.into())
    }
}

/// Push a local .scx file to cloud/local as exploded .scxd directory.
///
/// Args:
///     source: Local .scx file path
///     dest: Destination URL or path (e.g., "gs://bucket/data.scxd/")
///     parallelism: Number of parallel upload tasks (default: 8)
///
/// Returns:
///     dict with keys: bytes_uploaded, sections_uploaded, elapsed_secs, throughput_mbps
#[pyfunction]
#[pyo3(signature = (source, dest, parallelism=None))]
pub fn push(
    py: Python<'_>,
    source: &str,
    dest: &str,
    parallelism: Option<usize>,
) -> PyResult<Py<PyAny>> {
    let opts = scx_cloud::PushOptions {
        parallelism: parallelism.unwrap_or(8),
    };

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to create runtime: {e}")))?;

    let source_path = std::path::PathBuf::from(source);
    let dest = dest.to_string();
    // Release the GIL during blocking cloud I/O (finding 9.3).
    let stats = py
        .detach(|| rt.block_on(scx_cloud::push(&source_path, &dest, opts)))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let dict = PyDict::new(py);
    dict.set_item("bytes_uploaded", stats.bytes_uploaded)?;
    dict.set_item("sections_uploaded", stats.sections_uploaded)?;
    dict.set_item("elapsed_secs", stats.elapsed.as_secs_f64())?;
    dict.set_item("throughput_mbps", stats.throughput_mbps)?;
    Ok(dict.into())
}

/// Cloud-optimize an SCX file by adding a front-of-file catalog.
///
/// Args:
///     input: Input .scx file path
///     output: Optional output path (default: rewrite in-place via atomic rename)
#[pyfunction]
#[pyo3(signature = (input, output=None))]
pub fn cloud_optimize(input: &str, output: Option<&str>) -> PyResult<()> {
    let input_path = std::path::Path::new(input);
    let output_path = output.map(std::path::Path::new).unwrap_or(input_path);
    scx_cloud::cloud_optimize(input_path, output_path)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Explode a packed .scx file into a cloud-deployable .scxd directory.
///
/// Args:
///     input: Input .scx file path
///     output: Output directory path (should end in .scxd/)
#[pyfunction]
pub fn explode(input: &str, output: &str) -> PyResult<()> {
    let input_path = std::path::Path::new(input);
    let output_path = std::path::Path::new(output);
    scx_cloud::explode(input_path, output_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Pack an exploded .scxd directory back into a single .scx file.
///
/// Args:
///     input: Input directory path (should end in .scxd/)
///     output: Output .scx file path
#[pyfunction]
pub fn pack(input: &str, output: &str) -> PyResult<()> {
    let input_path = std::path::Path::new(input);
    let output_path = std::path::Path::new(output);
    scx_cloud::pack(input_path, output_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// A cloud-backed experiment handle for reading SCX data
/// without downloading the entire file.
///
/// Provides metadata accessors (n_obs, n_vars, nnz, shard_count)
/// and read methods for obs/var metadata.
#[pyclass(name = "CloudExperiment")]
pub struct PyCloudExperiment {
    reader: Arc<scx_cloud::CloudReader>,
    // Kept alive so the runtime handle stored inside any
    // `CloudSectionReader` constructed from `.query()` stays valid for
    // as long as that pipeline lives.
    rt: Arc<tokio::runtime::Runtime>,
    // Parsed modality table, fetched once at open (a small section).
    // `None` for single-modality / v1 files. Mirrors how the local
    // `Experiment` caches its modality table so the per-modality
    // accessors don't re-range-read on every call (B5).
    modalities: Option<scx_format_io::modality::ModalityTable>,
    // F2 grouped reads (7.3): a shared cloud-backed `QueryPipeline`, opened
    // once so repeated grouped calls don't re-fetch the catalog / schemas /
    // `group_index` over the network (combines with the engine-side
    // `GroupIndex` cache).
    grouped_pipeline: std::sync::OnceLock<Arc<QueryPipeline>>,
    /// The `gs://` / `s3://` / local URL this handle was opened from.
    /// `CloudReader` does not retain it (it keeps a parsed `ReaderLayout`),
    /// so we keep it here to back `path` / `info`. Mirrors
    /// `PyExperiment::path` semantically.
    url: String,
}

impl PyCloudExperiment {
    /// Live (non-deleted) row count: the header `n_obs` minus the deleted
    /// rows of the reader's keep mask — the cloud twin of
    /// `PyExperiment::logical_n_obs_of`. Header-only when the file flags no
    /// deletions; otherwise the mask `CloudReader` decodes once (success-only)
    /// and shares with every logical read, so `n_obs` and `read_obs()` cannot
    /// disagree: a fetch failure is raised here exactly as it would be there.
    /// `__repr__`, which must not raise, uses the best-effort twin.
    fn logical_n_obs(&self, py: Python<'_>) -> PyResult<u64> {
        let physical = self.reader.n_obs();
        if !self.reader.header().has_deletion_vectors() {
            return Ok(physical);
        }
        // A section fetch on first access: release the GIL around it like
        // every other cloud read, so a slow range request cannot stall
        // unrelated Python threads.
        let keep = py
            .detach(|| self.rt.block_on(self.reader.deletion_keep_mask()))
            .map_err(cloud_to_pyerr)?;
        Ok(keep.map_or(physical, |k| k.iter().filter(|b| **b).count() as u64))
    }

    fn logical_n_obs_best_effort(&self, py: Python<'_>) -> u64 {
        self.logical_n_obs(py)
            .unwrap_or_else(|_| self.reader.n_obs())
    }

    /// Resolve the optional `modality` kwarg shared by `uns_keys` and
    /// `read_uns` to a `modality_id`. `None` → 0 (global uns). A name that
    /// does not appear in the cached modality table raises `KeyError`, which
    /// also covers the "named modality on a non-multimodal file" case (the
    /// table is `None` there). Shared by `uns_keys` / `read_uns` / `read_var`.
    /// Mirrors `PyExperiment::resolve_uns_modality`.
    fn resolve_uns_modality(&self, modality: Option<&str>) -> PyResult<u8> {
        match modality {
            None => Ok(0),
            Some(name) => self
                .modalities
                .as_ref()
                .and_then(|t| t.id_of(name))
                .ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(format!("unknown modality '{name}'"))
                }),
        }
    }
}

#[pymethods]
impl PyCloudExperiment {
    /// Number of observations (cells), reflecting **live** rows — the header
    /// count minus any deletion-vector entries, like the local
    /// `Experiment.n_obs`, and what `query().count()` / `read_obs()` on this
    /// handle agree with. See `n_obs_physical` for the header count.
    #[getter]
    fn n_obs(&self, py: Python<'_>) -> PyResult<u64> {
        self.logical_n_obs(py)
    }

    #[getter]
    fn n_vars(&self) -> u64 {
        self.reader.n_vars()
    }

    /// `(n_obs, n_vars)` — mirrors `anndata.AnnData.shape` and the local
    /// `Experiment.shape`; `n_obs` is the live count (see `n_obs`).
    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<(u64, u64)> {
        Ok((self.logical_n_obs(py)?, self.reader.n_vars()))
    }

    /// Column names in `obs` (cell metadata), excluding the pandas index —
    /// the vocabulary accepted by `query().filter_obs(...)`. Mirrors the
    /// local `Experiment.obs_keys()`.
    ///
    /// I/O cost: for a sharded obs this reads only the FIRST shard (all
    /// shards share one schema), avoiding an atlas-scale assemble of every
    /// shard; for a single-section obs it reads that one section's Arrow IPC
    /// footer. Neither path assembles or caches the full obs table.
    fn obs_keys(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let schema = py
            .detach(|| {
                self.rt.block_on(async {
                    if self.reader.obs_metadata_shard_count() > 0 {
                        self.reader
                            .read_obs_shard(0)
                            .await
                            .map(|b| b.schema().as_ref().clone())
                    } else {
                        self.reader.read_obs_schema().await
                    }
                })
            })
            .map_err(cloud_to_pyerr)?;
        Ok(crate::experiment::schema_data_columns(Some(schema)))
    }

    /// Column names in `var` (gene metadata), excluding the pandas index.
    /// Mirror of [`Self::obs_keys`]. `var` is gene-axis (small — tens of
    /// thousands of rows) and rarely sharded, so this reads only the first
    /// var section's Arrow IPC footer (no assembly, no caching).
    fn var_keys(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let schema = py
            .detach(|| self.rt.block_on(self.reader.read_var_schema()))
            .map_err(cloud_to_pyerr)?;
        Ok(crate::experiment::schema_data_columns(Some(schema)))
    }

    /// Keys of the `obsm` cell-embedding mappings. Pure catalog scan (the
    /// catalog is held in memory after open, so no network I/O). Mirrors the
    /// local `Experiment.obsm_keys`.
    fn obsm_keys(&self) -> Vec<String> {
        self.reader.list_obsm()
    }

    /// Keys of the `varm` gene-embedding mappings. Pure catalog scan.
    /// Mirrors the local `Experiment.varm_keys`.
    fn varm_keys(&self) -> Vec<String> {
        self.reader.list_varm()
    }

    /// Names of the layers in the file. Pure catalog scan. Mirrors the local
    /// `Experiment.layer_names`.
    fn layer_names(&self) -> Vec<String> {
        self.reader.layer_names()
    }

    #[getter]
    fn nnz(&self) -> u64 {
        self.reader.nnz()
    }

    #[getter]
    fn shard_count(&self) -> u32 {
        self.reader.n_shards()
    }

    /// Number of `ObsMetadataShard` sections (0 on legacy single-section
    /// obs files). Lets operators distinguish sharded-obs vs legacy cloud
    /// files, which `shard_count` (CSR shards) conflates.
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

    #[getter]
    fn format_version(&self) -> u16 {
        self.reader.header().format_version
    }

    #[getter]
    fn codec_id(&self) -> u8 {
        self.reader.header().codec_id
    }

    /// `True` when the file has a CSC sidecar (gene-major shards).
    /// Mirrors the local `Experiment.has_csc`. Header-only, no I/O.
    #[getter]
    fn has_csc(&self) -> bool {
        self.reader.header().has_csc()
    }

    /// `True` when the file carries logical deletion vectors — some rows
    /// are marked deleted and drop on export. Mirrors the local
    /// `Experiment.has_deletions`. Header-only, no I/O.
    #[getter]
    fn has_deletions(&self) -> bool {
        self.reader.header().has_deletion_vectors()
    }

    /// File-header index dtype (`0=u16`, `1=u32`). Mirrors the local
    /// `Experiment.index_dtype`.
    #[getter]
    fn index_dtype(&self) -> u8 {
        self.reader.header().index_dtype
    }

    /// Physical row count straight from the file header — the count before
    /// any deletion-vector masking. Mirrors the local
    /// `Experiment.n_obs_physical`: equals `n_obs` when the file has no
    /// deletions, larger when rows were `mark_deleted` (until `compact`).
    /// Header-only, no I/O. (Before 0.17 the cloud `n_obs` was this same
    /// header count; it is now the live count, like the local handle.)
    #[getter]
    fn n_obs_physical(&self) -> u64 {
        self.reader.header().n_obs
    }

    /// The URL this handle was opened from (`gs://` / `s3://` / local
    /// path). Mirrors `PyExperiment.path`, which returns the local path.
    #[getter]
    fn path(&self) -> String {
        self.url.clone()
    }

    /// Codec / shard / format-version internals as a one-line string.
    /// Mirrors `PyExperiment.info` token for token (pinned by
    /// `test_cloud.py`); the AnnData-style repr lists keys, the on-disk
    /// encoding details live here. The `value_encoding` / `is_integer` tokens
    /// cost one 76-byte range read per CSR shard (in parallel, as `scx info`
    /// does on a cloud URL); everything else is header / catalog only.
    fn info(&self, py: Python<'_>) -> PyResult<String> {
        let h = self.reader.header();
        let (_codecs, encodings) = py
            .detach(|| self.rt.block_on(self.reader.csr_shard_field_summaries()))
            .map_err(cloud_to_pyerr)?;
        let (value_encoding, is_integer) = crate::experiment::render_value_encodings(&encodings);
        Ok(format!(
            "SCX file: format_version={}, codec_id={}, index_dtype={}, \
             csr_shards={}, nnz={}, has_csc={}, value_encoding={}, is_integer={}, \
             max_value={}, path={}",
            h.format_version,
            h.codec_id,
            h.index_dtype,
            h.n_csr_shards,
            self.reader.nnz(),
            h.has_csc(),
            value_encoding,
            is_integer,
            self.reader.catalog().csr_max_value(None),
            self.url,
        ))
    }

    /// True if this file is multimodal (v2 with `n_modalities > 0`).
    /// Mirrors the local `Experiment.is_multimodal`. The modality table
    /// is fetched once at open, so this is free.
    #[getter]
    fn is_multimodal(&self) -> bool {
        self.modalities.is_some()
    }

    /// Number of registered modalities (0 for v1 / single-modality files).
    #[getter]
    fn n_modalities(&self) -> u32 {
        self.modalities
            .as_ref()
            .map(|t| t.entries.len() as u32)
            .unwrap_or(0)
    }

    /// Ordered list of modality names (empty for single-modality files).
    /// Position in the list maps 1:1 to the 1-based modality_id
    /// (`names[i] -> modality_id = i + 1`). Mirrors the local
    /// `Experiment.modality_names`.
    #[getter]
    fn modality_names(&self) -> Vec<String> {
        self.modalities
            .as_ref()
            .map(|t| t.entries.iter().map(|e| e.name.clone()).collect())
            .unwrap_or_default()
    }

    /// Resolve a modality name to its 1-based `modality_id`. Returns
    /// `None` for unknown names or single-modality files.
    fn modality_id(&self, name: &str) -> Option<u8> {
        self.modalities.as_ref().and_then(|t| t.id_of(name))
    }

    /// Per-modality information block for the given 1-based `modality_id`.
    /// Returns a dict with the on-disk fields of `ModalityInfo` (same
    /// shape as the local `Experiment.modality_info`).
    fn modality_info<'py>(
        &self,
        py: Python<'py>,
        modality_id: u8,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let table = match &self.modalities {
            Some(t) => t,
            None => return Ok(None),
        };
        let info = match modality_id
            .checked_sub(1)
            .and_then(|idx| table.entries.get(idx as usize))
        {
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

    /// Keys of the `uns` (unstructured metadata) section, or `[]` when the
    /// file has no `uns`. `modality=None` reads the global `uns`; a modality
    /// name reads that modality's `uns/<name>` section (raises `KeyError` for
    /// an unknown name). Mirrors the local `Experiment.uns_keys`.
    #[pyo3(signature = (modality=None))]
    fn uns_keys(&self, py: Python<'_>, modality: Option<&str>) -> PyResult<Vec<String>> {
        let modality_id = self.resolve_uns_modality(modality)?;
        let val = py
            .detach(|| self.rt.block_on(self.reader.read_uns_for(modality_id)))
            .map_err(cloud_to_pyerr)?;
        match val {
            Some(serde_json::Value::Object(map)) => Ok(map.keys().cloned().collect()),
            _ => Ok(Vec::new()),
        }
    }

    /// Read the `uns` (unstructured metadata) section as a Python object,
    /// reconstructing NumPy/pandas envelopes. Returns `None` when the file
    /// has no `uns`. `modality=None` reads the global `uns`; a modality name
    /// reads that modality's `uns/<name>` section (raises `KeyError` for an
    /// unknown name). Mirrors the local `Experiment.read_uns`.
    #[pyo3(signature = (modality=None))]
    fn read_uns<'py>(
        &self,
        py: Python<'py>,
        modality: Option<&str>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let modality_id = self.resolve_uns_modality(modality)?;
        let val = py
            .detach(|| self.rt.block_on(self.reader.read_uns_for(modality_id)))
            .map_err(cloud_to_pyerr)?;
        match val {
            Some(v) => Ok(Some(crate::convert::uns::json_value_to_pyobject(py, &v)?)),
            None => Ok(None),
        }
    }

    /// Read the `obs` (cell metadata) table as a pandas DataFrame over the
    /// cloud path, without touching X. Mirrors the local
    /// `Experiment.read_obs`, row space included: `logical=True` (default)
    /// returns the live rows (deletion vectors applied, `len == n_obs`, the
    /// frame `query().collect()` on this handle agrees with); `logical=False`
    /// the physical table (`n_obs_physical` rows). Changed in 0.17, with the
    /// local handle. `columns` projects a subset by physical name.
    ///
    /// `columns` is a genuine **pushdown**: each obs shard is fetched as a
    /// projected range read, so the network cost is the requested columns' bytes
    /// rather than the whole obs body. (Before this it projected the fully
    /// assembled batch, which fetched everything regardless.) The pandas index
    /// column (cell barcodes) is always retained, so a projected frame keeps the
    /// same index as the unprojected `read_obs()`.
    ///
    /// Note the projected path does not populate the assembled-obs cache that
    /// unprojected `read_obs()` fills and reuses. For a single categorical
    /// column's distinct values prefer `distinct_values()`; for its codes prefer
    /// `obs_categorical()`.
    #[pyo3(signature = (columns=None, *, logical=true))]
    fn read_obs<'py>(
        &self,
        py: Python<'py>,
        columns: Option<Vec<String>>,
        logical: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let batch = py
            .detach(|| match columns {
                Some(cols) => self.rt.block_on(async {
                    // Retain the pandas index column(s) so the projected frame
                    // keeps its barcode index and pyarrow can restore it (parity
                    // with unprojected read_obs; avoids dropping the index the
                    // pandas envelope still advertises).
                    let schema = self.reader.read_obs_schema().await?;
                    let mut names: Vec<String> = Vec::new();
                    for idx_col in scx_format_io::resolve_index_columns(&schema) {
                        if schema.index_of(&idx_col).is_ok() && !cols.contains(&idx_col) {
                            names.push(idx_col);
                        }
                    }
                    names.extend(cols);
                    if logical {
                        self.reader.read_obs_keys_filtered(&names).await
                    } else {
                        self.reader.read_obs_keys(&names).await
                    }
                }),
                None if logical => self.rt.block_on(self.reader.read_obs_filtered()),
                None => self.rt.block_on(self.reader.read_obs()),
            })
            .map_err(cloud_to_pyerr)?;
        let table = crate::convert::record_batch_to_pyarrow(py, &batch)?;
        crate::convert::pyarrow_table_to_pandas(&table)
    }

    /// Read `var` (gene metadata) as a pandas DataFrame over the cloud path.
    ///
    /// Mirror of the local [`Experiment::read_var`]. Without it, a cloud
    /// caller who wants gene symbols has to `to_anndata()` the whole matrix
    /// over the network — the same asymmetry that motivated adding `read_var`
    /// locally, one layer over.
    ///
    /// As locally, `columns` is a convenience projection applied **after** the
    /// fetch, not a pushdown: `var` is one section sized by `n_vars` (a few MB
    /// even on an atlas), so there is no per-column range read to save. This
    /// differs from `CloudExperiment.read_obs`, where `columns` *is* a genuine
    /// network pushdown because `obs` scales with `n_obs`. The pandas index
    /// column (gene names) is always retained.
    ///
    /// `modality=<name>` selects one modality's gene axis on a multimodal
    /// file; unknown name → `KeyError`.
    #[pyo3(signature = (columns=None, *, modality=None))]
    fn read_var<'py>(
        &self,
        py: Python<'py>,
        columns: Option<Vec<String>>,
        modality: Option<&str>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let modality_id = self.resolve_uns_modality(modality)?;
        let batch = py
            .detach(|| self.rt.block_on(self.reader.read_var_for(modality_id)))
            .map_err(cloud_to_pyerr)?;
        let batch = match columns {
            None => batch,
            Some(cols) => crate::experiment::project_batch_columns(&batch, &cols)?,
        };
        let table = crate::convert::record_batch_to_pyarrow(py, &batch)?;
        crate::convert::pyarrow_table_to_pandas(&table)
    }

    /// Distinct values of a single string/categorical `obs` column over the
    /// cloud path, returned as `(values, has_more)`. Mirrors the local
    /// `Experiment.distinct_values` (same dictionary-superset / null / `limit`
    /// / `sort` semantics); never touches X or assembles the full obs table.
    #[pyo3(signature = (col, *, limit=None, sort=false))]
    fn distinct_values(
        &self,
        py: Python<'_>,
        col: &str,
        limit: Option<usize>,
        sort: bool,
    ) -> PyResult<(Vec<String>, bool)> {
        py.detach(|| {
            self.rt
                .block_on(self.reader.distinct_obs_values(col, limit, sort))
        })
        .map_err(cloud_to_pyerr)
    }

    /// `(codes, categories)` for a single string/categorical `obs` column over
    /// the cloud path. Mirrors the local `Experiment.obs_categorical` exactly —
    /// same `-1`-for-null, first-seen-order and unreferenced-level semantics,
    /// and the same row space: `logical=True` (default) one code per live row,
    /// `logical=False` one per physical row.
    ///
    /// Each shard is a projected range read folded immediately, so neither the
    /// full obs body nor the assembled table is ever fetched.
    #[pyo3(signature = (col, *, logical=true))]
    fn obs_categorical<'py>(
        &self,
        py: Python<'py>,
        col: &str,
        logical: bool,
    ) -> PyResult<crate::experiment::PyCategorical<'py>> {
        let (codes, categories) = py
            .detach(|| {
                self.rt.block_on(async {
                    if logical {
                        self.reader.obs_categorical_filtered(col).await
                    } else {
                        self.reader.obs_categorical(col).await
                    }
                })
            })
            .map_err(cloud_to_pyerr)?;
        Ok((PyArray1::from_vec(py, codes), categories))
    }

    /// `obs_categorical` for several columns in one pass over the obs shards.
    /// Mirrors the local `Experiment.obs_categorical_many`, `logical=` included.
    #[pyo3(signature = (cols, *, logical=true))]
    fn obs_categorical_many<'py>(
        &self,
        py: Python<'py>,
        cols: Vec<String>,
        logical: bool,
    ) -> PyResult<Vec<crate::experiment::PyCategorical<'py>>> {
        let out = py
            .detach(|| {
                self.rt.block_on(async {
                    if logical {
                        self.reader.obs_categorical_many_filtered(&cols).await
                    } else {
                        self.reader.obs_categorical_many(&cols).await
                    }
                })
            })
            .map_err(cloud_to_pyerr)?;
        Ok(out
            .into_iter()
            .map(|(codes, cats)| (PyArray1::from_vec(py, codes), cats))
            .collect())
    }

    /// Materialise a multimodal file as `mudata.MuData`.
    ///
    /// Not yet supported over the cloud path: cloud multimodal reads
    /// would need per-modality section routing that the cloud query
    /// pipeline does not implement. Pull the file locally first (B5).
    #[pyo3(signature = (backed=false, cache_shards=4))]
    fn to_mudata(&self, backed: bool, cache_shards: usize) -> PyResult<()> {
        let _ = (backed, cache_shards);
        Err(PyRuntimeError::new_err(
            "to_mudata() over the cloud path is not yet supported; \
             pull the file locally (`scx pull <url> <dir>`) and open it \
             with `pyscx.open(...).to_mudata()`",
        ))
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        // No key lines: listing obs/var/obsm/uns keys would require
        // network range reads, so the cloud repr stays to the cheap
        // header line. Use `.query()` / `read_cloud()` to materialise.
        let mut repr = crate::experiment::format_anndata_repr(
            "CloudExperiment",
            self.logical_n_obs_best_effort(py),
            self.reader.n_vars(),
            &[],
        );
        // Surface multimodality so the other modalities aren't silently
        // invisible — `shape`/`n_vars` reflect the primary modality (B5).
        if let Some(table) = &self.modalities {
            let names: Vec<&str> = table.entries.iter().map(|e| e.name.as_str()).collect();
            repr.push_str(&format!(
                "\n    multimodal: {} modalities [{}]",
                names.len(),
                names.join(", ")
            ));
        }
        repr
    }

    /// Open a lazy query pipeline backed by this cloud experiment.
    ///
    /// The returned pipeline supports the same chain as the local
    /// `pyscx.open(path).query()` pipeline (`filter_obs`,
    /// `filter_var`, `select_genes`, `with_normalize`, `with_log1p`,
    /// `limit`, `collect`). I/O happens lazily on `.collect()` and
    /// downloads only the catalog sections, obs metadata, var
    /// metadata, predicate indexes (if present), and CSR shards that
    /// match the predicate / projection — no `scx pull` to local disk
    /// is required first.
    ///
    /// `modality` scopes the query to one modality of a multimodal file
    /// (X / `select_genes` / `filter_var` resolve against that modality's var;
    /// `filter_obs` stays on the shared global obs axis). On a multimodal file
    /// `modality` is required (omitting → `ValueError`); an unknown name →
    /// `KeyError`. Omit it on single-modality files.
    #[pyo3(signature = (modality=None))]
    fn query(&self, py: Python<'_>, modality: Option<&str>) -> PyResult<PyQueryPipeline> {
        // Resolve name → 1-based id against the cached modality table (0 =
        // global). Mirrors the local `Experiment.query`.
        let modality_id: u8 = match modality {
            None => {
                if self.modalities.is_some() {
                    return Err(PyValueError::new_err(format!(
                        "file is multimodal; pass modality=... (one of {:?})",
                        self.modality_names()
                    )));
                }
                0
            }
            Some(name) => self.modality_id(name).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "unknown modality '{name}'; available: {:?}",
                    self.modality_names()
                ))
            })?,
        };
        let reader = Arc::clone(&self.reader);
        let rt = Arc::clone(&self.rt);
        let pipeline = py
            .detach(|| {
                let adapter = scx_cloud::CloudSectionReader::new(reader, rt);
                QueryPipeline::from_reader_for_modality(Box::new(adapter), modality_id)
            })
            .map_err(crate::query::engine_to_pyerr)?;
        Ok(PyQueryPipeline::from_pipeline(pipeline))
    }

    /// F2: read exactly the cells of one `group_by` label as an AnnData, over
    /// the network. Grouped archive only (written with `scx sort --group-by` /
    /// `pyscx.sort(group_by=...)`). `KeyError` (with close matches) for an
    /// unknown label, `ValueError` if the archive is not grouped.
    fn read_group<'py>(&self, py: Python<'py>, label: &str) -> PyResult<Bound<'py, PyAny>> {
        let pipeline = self.grouped_pipeline(py)?;
        let result = py
            .detach(|| pipeline.read_group(label))
            .map_err(engine_to_pyerr)?;
        crate::query::query_result_to_anndata(py, result)
    }

    /// F2: read the reference cells (e.g. "non-targeting") as an AnnData, or
    /// `None` if the archive has no reference rows.
    fn read_reference<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let pipeline = self.grouped_pipeline(py)?;
        let result = py
            .detach(|| pipeline.read_reference())
            .map_err(engine_to_pyerr)?;
        match result {
            Some(qr) => Ok(Some(crate::query::query_result_to_anndata(py, qr)?)),
            None => Ok(None),
        }
    }

    /// F2: distinct group labels present in the archive.
    fn group_labels(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let pipeline = self.grouped_pipeline(py)?;
        py.detach(|| pipeline.group_labels())
            .map_err(engine_to_pyerr)
    }

    /// F2: one `GroupShard` per non-reference shard, for streaming reads that
    /// keep ~one shard resident. Each shard reuses the shared cloud pipeline.
    fn iter_group_shards(&self, py: Python<'_>) -> PyResult<Vec<PyGroupShard>> {
        let pipeline = self.grouped_pipeline(py)?;
        let handles = py
            .detach(|| pipeline.iter_group_shards())
            .map_err(engine_to_pyerr)?;
        Ok(handles
            .into_iter()
            .map(|handle| PyGroupShard::new(Arc::clone(&pipeline), handle))
            .collect())
    }
}

impl PyCloudExperiment {
    /// The shared cloud-backed grouped-read pipeline, opened once on first use
    /// (catalog + schemas fetched a single time over the network).
    fn grouped_pipeline(&self, py: Python<'_>) -> PyResult<Arc<QueryPipeline>> {
        if let Some(p) = self.grouped_pipeline.get() {
            return Ok(Arc::clone(p));
        }
        let reader = Arc::clone(&self.reader);
        let rt = Arc::clone(&self.rt);
        let pipeline = py
            .detach(|| {
                let adapter = scx_cloud::CloudSectionReader::new(reader, rt);
                QueryPipeline::from_reader(Box::new(adapter))
            })
            .map_err(engine_to_pyerr)?;
        let arc = Arc::new(pipeline);
        // If another thread won the race, `set` fails; return the winner from the
        // cell so all callers share one pipeline (matches PyExperiment).
        let _ = self.grouped_pipeline.set(arc);
        Ok(Arc::clone(
            self.grouped_pipeline
                .get()
                .expect("grouped_pipeline populated above"),
        ))
    }
}

/// Open an SCX file or exploded directory from cloud/local storage.
///
/// Detects layout automatically: exploded .scxd directories,
/// cloud-ready packed .scx, or non-cloud-ready packed .scx.
///
/// Args:
///     url: Source URL or local path (e.g., "gs://bucket/data.scxd/")
///
/// Returns:
///     PyCloudExperiment handle with metadata accessors
#[pyfunction]
pub fn open_cloud(py: Python<'_>, url: &str) -> PyResult<PyCloudExperiment> {
    let rt = build_cloud_runtime()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to create runtime: {e}")))?;

    // Release the GIL during blocking cloud I/O (finding 9.3).
    let url = url.to_string();
    let (reader, modalities) = py
        .detach(|| {
            rt.block_on(async {
                let reader = scx_cloud::open_cloud(&url).await?;
                // Fetch the modality table up front (small section, works
                // for packed + exploded) so multimodal files are
                // discoverable instead of silently projecting to the
                // primary modality (B5).
                let modalities = reader.modality_table().await?;
                Ok::<_, scx_cloud::CloudError>((reader, modalities))
            })
        })
        .map_err(cloud_to_pyerr)?;

    Ok(PyCloudExperiment {
        reader: Arc::new(reader),
        rt: Arc::new(rt),
        modalities,
        grouped_pipeline: std::sync::OnceLock::new(),
        url,
    })
}

/// Read an SCX file from cloud/local object storage into an AnnData in
/// one call — the flat helper mirroring `scanpy.read_h5ad` for cloud
/// sources. Equivalent to
/// `open_cloud(url).query().filter_obs(obs_filter).select_genes(...).collect().to_anndata()`
/// but without the explicit chain.
///
/// Args:
///     url: Source URL or local path (e.g., "gs://bucket/data.scxd/"
///          or "file:///path/data.scxd").
///     obs_filter: Optional obs predicate (e.g. "cell_type == 'T cell'")
///                 pushed down so only matching shards are fetched.
///     var_names: Optional list of gene names to project to. Resolved
///                against the file's `var` index; an unknown name raises
///                KeyError.
///
/// Returns an `anndata.AnnData`. Normalization / log1p transforms are
/// not exposed here — build the explicit `open_cloud(url).query()` chain
/// when you need them.
#[pyfunction]
#[pyo3(signature = (url, *, obs_filter=None, var_names=None, modality=None))]
pub fn read_cloud<'py>(
    py: Python<'py>,
    url: &str,
    obs_filter: Option<&str>,
    var_names: Option<Vec<String>>,
    modality: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let rt = build_cloud_runtime()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to create runtime: {e}")))?;
    let url_s = url.to_string();
    let reader = py
        .detach(|| rt.block_on(scx_cloud::open_cloud(&url_s)))
        .map_err(cloud_to_pyerr)?;
    let reader = Arc::new(reader);
    let rt = Arc::new(rt);

    // Resolve `modality` name → 1-based id (0 = global) against the modality
    // table (fetched once). Multimodal + None → ValueError; unknown → KeyError.
    let modality_table = {
        let reader_for_table = Arc::clone(&reader);
        py.detach(|| rt.block_on(reader_for_table.modality_table()))
            .map_err(cloud_to_pyerr)?
    };
    let modality_id: u8 = match modality {
        None => {
            if modality_table.is_some() {
                let names: Vec<String> = modality_table
                    .as_ref()
                    .map(|t| t.entries.iter().map(|m| m.name.clone()).collect())
                    .unwrap_or_default();
                return Err(PyValueError::new_err(format!(
                    "file is multimodal; pass modality=... (one of {names:?})"
                )));
            }
            0
        }
        Some(name) => modality_table
            .as_ref()
            .and_then(|t| t.id_of(name))
            .ok_or_else(|| {
                let names: Vec<String> = modality_table
                    .as_ref()
                    .map(|t| t.entries.iter().map(|m| m.name.clone()).collect())
                    .unwrap_or_default();
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "unknown modality '{name}'; available: {names:?}"
                ))
            })?,
    };

    // Resolve gene names → indices against the (modality-scoped) var index
    // before the reader is moved into the section-reader adapter. An explicitly
    // empty `var_names=[]` is honored as "project to zero genes" (it must
    // NOT fall through to None / all-genes, which would silently download
    // the full matrix for a caller-computed empty marker list).
    let gene_indices: Option<Vec<u32>> = match var_names {
        // Empty list → project to zero genes without a wasted var read.
        Some(names) if names.is_empty() => Some(Vec::new()),
        Some(names) => {
            let reader_for_var = Arc::clone(&reader);
            let var_batch = py
                .detach(|| rt.block_on(reader_for_var.read_var_for(modality_id)))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let mut indices = Vec::with_capacity(names.len());
            for name in &names {
                match crate::experiment::lookup_gene_in_batch(&var_batch, name) {
                    Some(idx) => indices.push(idx),
                    None => {
                        return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                            "gene name '{name}' not found in var index"
                        )))
                    }
                }
            }
            Some(indices)
        }
        _ => None,
    };

    let reader_for_adapter = Arc::clone(&reader);
    let rt_for_adapter = Arc::clone(&rt);
    let mut pipeline = py
        .detach(|| {
            let adapter = scx_cloud::CloudSectionReader::new(reader_for_adapter, rt_for_adapter);
            QueryPipeline::from_reader_for_modality(Box::new(adapter), modality_id)
        })
        .map_err(crate::query::engine_to_pyerr)?;

    if let Some(expr) = obs_filter {
        pipeline = pipeline
            .filter_obs(expr)
            .map_err(crate::query::engine_to_pyerr)?;
    }
    if let Some(indices) = gene_indices {
        pipeline = pipeline.select_genes(indices);
    }

    let result = py
        .detach(|| pipeline.collect())
        .map_err(crate::query::engine_to_pyerr)?;
    crate::query::query_result_to_anndata(py, result)
}
