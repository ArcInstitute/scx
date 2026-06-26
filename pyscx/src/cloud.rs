//! Python bindings for scx-cloud operations.
//!
//! Provides PyO3 wrappers for pull, push, cloud_optimize, explode, and pack.
//! All async operations create a tokio runtime internally.

use std::sync::Arc;

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
}

#[pymethods]
impl PyCloudExperiment {
    #[getter]
    fn n_obs(&self) -> u64 {
        self.reader.n_obs()
    }

    #[getter]
    fn n_vars(&self) -> u64 {
        self.reader.n_vars()
    }

    /// `(n_obs, n_vars)` — mirrors `anndata.AnnData.shape` and the local
    /// `Experiment.shape`. Header-only, no network I/O.
    #[getter]
    fn shape(&self) -> (u64, u64) {
        (self.reader.n_obs(), self.reader.n_vars())
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

    fn __repr__(&self) -> String {
        // No key lines: listing obs/var/obsm/uns keys would require
        // network range reads, so the cloud repr stays to the cheap
        // header line. Use `.query()` / `read_cloud()` to materialise.
        let mut repr = crate::experiment::format_anndata_repr(
            "CloudExperiment",
            self.reader.n_obs(),
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
    fn query(&self, py: Python<'_>) -> PyResult<PyQueryPipeline> {
        let reader = Arc::clone(&self.reader);
        let rt = Arc::clone(&self.rt);
        let pipeline = py
            .detach(|| {
                let adapter = scx_cloud::CloudSectionReader::new(reader, rt);
                QueryPipeline::from_reader(Box::new(adapter))
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
        let _ = self.grouped_pipeline.set(Arc::clone(&arc));
        Ok(arc)
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
#[pyo3(signature = (url, *, obs_filter=None, var_names=None))]
pub fn read_cloud<'py>(
    py: Python<'py>,
    url: &str,
    obs_filter: Option<&str>,
    var_names: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyAny>> {
    let rt = build_cloud_runtime()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to create runtime: {e}")))?;
    let url_s = url.to_string();
    let reader = py
        .detach(|| rt.block_on(scx_cloud::open_cloud(&url_s)))
        .map_err(cloud_to_pyerr)?;
    let reader = Arc::new(reader);
    let rt = Arc::new(rt);

    // Resolve gene names → indices against the file's var index before
    // the reader is moved into the section-reader adapter. An explicitly
    // empty `var_names=[]` is honored as "project to zero genes" (it must
    // NOT fall through to None / all-genes, which would silently download
    // the full matrix for a caller-computed empty marker list).
    let gene_indices: Option<Vec<u32>> = match var_names {
        // Empty list → project to zero genes without a wasted var read.
        Some(names) if names.is_empty() => Some(Vec::new()),
        Some(names) => {
            let reader_for_var = Arc::clone(&reader);
            let var_batch = py
                .detach(|| rt.block_on(reader_for_var.read_var()))
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
            QueryPipeline::from_reader(Box::new(adapter))
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

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
