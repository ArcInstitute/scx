// PyExperiment — lazy handle for SCX files
//
// Split into submodules (ORG-10.16-6): `lookup` (gene/column resolution and
// repr plumbing shared with the cloud reader), `gpu_anndata` (the cfg(gpu)
// to_gpu_anndata device path), `group_shard` (PyGroupShard + the shared
// error adapters). This module keeps the pyclass, its single #[pymethods]
// block (pyo3 without `multiple-pymethods` allows exactly one), and the
// inherent impls; the submodule items are re-exported so existing
// `crate::experiment::*` paths are unchanged.

// PyExperiment — lazy handle for SCX files

use std::path::PathBuf;
use std::sync::Arc;

use numpy::{PyArray1, PyReadonlyArray1};

use pyo3::prelude::*;
use scx_engine::QueryPipeline;
use scx_format_io::ScxReader;

use crate::convert;
use crate::query::PyQueryPipeline;
use crate::to_pyerr;

pub(crate) mod gpu_anndata;
pub(crate) mod group_shard;
pub(crate) mod lookup;
pub(crate) mod materialize;

pub(crate) use group_shard::*;
pub(crate) use lookup::*;

/// A handle to an open SCX file.
///
/// Provides metadata accessors and methods to convert to AnnData.
#[pyclass(name = "Experiment")]
pub struct PyExperiment {
    /// `None` after `close()`. Private and only reachable through
    /// [`Self::reader`], which is what makes the closed / stale checks
    /// structural: a read method added later cannot get at the mapping
    /// without going past them.
    reader: Option<ScxReader>,
    pub(crate) path: PathBuf,
    /// Memoized deletion-vector popcount (live-count complement). Computed once
    /// on the first `n_obs`/repr access for deletion-bearing files so repeated
    /// access doesn't re-decode the deletion section. `OnceLock` keeps the
    /// pyclass `Send + Sync`.
    n_deleted: std::sync::OnceLock<u64>,
    /// Memoized `(value_encoding rendering, is_integer)` — one shard-header read
    /// per CSR shard on the first `value_encoding` / `is_integer` / `info()`
    /// access, an atomic load afterwards. Reset with the reader on `reload` /
    /// `close`; every read goes through `reader()?` first, so a stale handle
    /// refuses before it can answer from this.
    value_encoding_memo: std::sync::OnceLock<(String, bool)>,
    /// Shared, lazily-opened query pipeline for the grouped-read API
    /// (`read_group` / `read_reference` / `group_labels` / `iter_group_shards`).
    /// Opened once per `Experiment` so a streaming loop parses the catalog and
    /// `group_index` sidecar a single time (combined with the engine-side
    /// `GroupIndex` cache). Shared into each `GroupShard` via `Arc`.
    grouped_pipeline: std::sync::OnceLock<Arc<QueryPipeline>>,
}

impl PyExperiment {
    /// Construct from an already-opened ScxReader and its path (Rust-only).
    ///
    /// The reader should have been opened with `ScxReader::watching()` — see
    /// [`Self::reader`]. Passing an unwatched one is not an error; it just
    /// means this handle will not notice the file changing underneath it.
    pub fn new(reader: ScxReader, path: PathBuf) -> Self {
        Self {
            reader: Some(reader),
            path,
            n_deleted: std::sync::OnceLock::new(),
            value_encoding_memo: std::sync::OnceLock::new(),
            grouped_pipeline: std::sync::OnceLock::new(),
        }
    }

    /// The open reader, refusing if the handle is closed or if the file has
    /// changed since it was opened.
    ///
    /// Every read on this class goes through here. The freshness half is
    /// redundant for section reads — `ScxReader::section_bytes` checks too —
    /// but not for the many answers that come straight off the parsed header
    /// or catalog (`n_obs` after an `append` is the obvious one), and paying
    /// for one extra `stat` on the section paths is worth not having to
    /// remember which is which.
    fn reader(&self) -> PyResult<&ScxReader> {
        let reader = self.reader.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Experiment for '{}' is closed; re-open it with pyscx.open()",
                self.path.display()
            ))
        })?;
        reader.check_fresh().map_err(to_pyerr)?;
        Ok(reader)
    }

    /// Re-open the file and drop every cache that snapshots it.
    ///
    /// Shared by the public `reload()` and by `mark_deleted`, which mutates
    /// through this same handle — two copies of this would be two chances for
    /// one of them to forget a cache.
    fn reopen(&mut self) -> PyResult<()> {
        self.reader = Some(
            ScxReader::open(&self.path)
                .and_then(|r| r.watching())
                .map_err(to_pyerr)?,
        );
        // Both snapshot the pre-mutation file: the grouped pipeline holds its
        // own reader and deletion-vector view, and the deleted count is a
        // memoized popcount.
        self.grouped_pipeline = std::sync::OnceLock::new();
        self.n_deleted = std::sync::OnceLock::new();
        self.value_encoding_memo = std::sync::OnceLock::new();
        Ok(())
    }

    /// The shared grouped-read pipeline, opening it once on first use.
    fn grouped_pipeline(&self) -> PyResult<Arc<QueryPipeline>> {
        // Opened lazily, so a first use *after* a mutation would stamp the new
        // file and quietly serve it while `read_obs` on the same handle
        // refuses. Gate on this handle's own view before either branch.
        self.reader()?;
        if let Some(p) = self.grouped_pipeline.get() {
            return Ok(Arc::clone(p));
        }
        // Watched, like every other reader handed to Python: this pipeline
        // is cached on the Experiment and reused across grouped reads, so it
        // outlives any mutation just as the Experiment itself does.
        let p = Arc::new(
            QueryPipeline::from_reader(Box::new(
                crate::open_handle_reader(&self.path).map_err(to_pyerr)?,
            ))
            .map_err(engine_to_pyerr)?,
        );
        let _ = self.grouped_pipeline.set(p);
        Ok(Arc::clone(
            self.grouped_pipeline
                .get()
                .expect("grouped_pipeline populated above"),
        ))
    }

    /// Resolve the optional `modality` kwarg shared by `uns_keys`,
    /// `read_uns` and `read_var` to a `modality_id`. `None` → 0 (global uns). A name
    /// that does not appear in the modality table raises `KeyError`,
    /// which also covers the "named modality on a non-multimodal file"
    /// case (the modality table is empty there).
    fn resolve_uns_modality(&self, modality: Option<&str>) -> PyResult<u8> {
        match modality {
            None => Ok(0),
            Some(name) => self.reader()?.modality_id(name).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("unknown modality '{name}'"))
            }),
        }
    }
}

impl PyExperiment {
    /// Live (non-deleted) row count: physical `n_obs` minus the deletion-vector
    /// popcount. Best-effort — falls back to the physical count when the file
    /// has no deletion vectors or the deletion section can't be read, so it
    /// never panics (the `n_obs` getter and repr must always render).
    fn logical_n_obs_of(&self, reader: &ScxReader) -> u64 {
        let physical = reader.n_obs();
        if !reader.header().has_deletion_vectors() {
            return physical;
        }
        // Decode the deletion section at most once per Experiment (best-effort:
        // a failed read memoizes 0, matching the pre-cache physical fallback).
        let deleted = *self.n_deleted.get_or_init(|| {
            reader
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
    fn n_obs(&self) -> PyResult<u64> {
        Ok(self.logical_n_obs_of(self.reader()?))
    }

    /// Physical (pre-deletion) row count straight from the file header. Equals
    /// [`Self::n_obs`] when the file has no deletion vectors; larger when rows
    /// have been logically deleted via `mark_deleted` (until `compact`).
    #[getter]
    fn n_obs_physical(&self) -> PyResult<u64> {
        Ok(self.reader()?.n_obs())
    }

    /// Number of variables (genes).
    #[getter]
    fn n_vars(&self) -> PyResult<u64> {
        Ok(self.reader()?.n_vars())
    }

    /// `(n_obs, n_vars)` — mirrors `anndata.AnnData.shape`. `n_obs` is the
    /// logical (post-deletion) row count.
    #[getter]
    fn shape(&self) -> PyResult<(u64, u64)> {
        let reader = self.reader()?;
        Ok((self.logical_n_obs_of(reader), reader.n_vars()))
    }

    /// Total number of non-zero entries — **physical**, unlike [`Self::n_obs`].
    ///
    /// Entries in logically deleted rows are still counted: this comes from the
    /// per-shard catalog stats, and excluding them would mean decoding the
    /// matrix. So on a file with deletions this exceeds `to_anndata().X.nnz`;
    /// `compact` makes the two agree.
    #[getter]
    fn nnz(&self) -> PyResult<u64> {
        Ok(self.reader()?.nnz())
    }

    /// Number of CSR shards in the file.
    #[getter]
    fn shard_count(&self) -> PyResult<u32> {
        Ok(self.reader()?.header().n_csr_shards)
    }

    /// Number of `ObsMetadataShard` sections in the catalog. Zero on
    /// legacy single-section files (`obs` is one `ObsMetadata` section);
    /// `>= 1` on Phase 2 / 4 sharded files written by merge, append, or
    /// `from_anndata` when `n_obs > shard_target_rows`.
    #[getter]
    fn obs_metadata_shard_count(&self) -> PyResult<usize> {
        Ok(self.reader()?.obs_metadata_shard_count())
    }

    /// Number of `VarMetadataShard` sections. Mirror of
    /// [`Self::obs_metadata_shard_count`].
    #[getter]
    fn var_metadata_shard_count(&self) -> PyResult<usize> {
        Ok(self.reader()?.var_metadata_shard_count())
    }

    /// Format version (currently 1).
    #[getter]
    fn format_version(&self) -> PyResult<u16> {
        Ok(self.reader()?.header().format_version)
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
    fn codec_id(&self) -> PyResult<u8> {
        Ok(self.reader()?.header().codec_id)
    }

    /// File-header index dtype (`0=u16`, `1=u32`). Used for testing the
    /// SCX → SCX writer's projection-aware index-dtype selection.
    #[getter]
    fn index_dtype(&self) -> PyResult<u8> {
        Ok(self.reader()?.header().index_dtype)
    }

    /// Names of the layers in the file.
    ///
    /// Callable method (`exp.layer_names()`), consistent with the
    /// `obs_keys()` / `var_keys()` / `obsm_keys()` / `varm_keys()` /
    /// `uns_keys()` accessor family (F7).
    fn layer_names(&self) -> PyResult<Vec<String>> {
        Ok(self.reader()?.layer_names())
    }

    /// Column names in `obs` (the cell metadata), excluding the pandas
    /// index column. Pure Arrow IPC footer read — no batch decode.
    /// Raises if the obs section cannot be read (e.g. corrupt file).
    ///
    /// Callable method (e.g. `exp.obs_keys()`) to match AnnData's
    /// `adata.obs_keys()`, not a property.
    fn obs_keys(&self) -> PyResult<Vec<String>> {
        let schema = self
            .reader()?
            .read_obs_schema_physical()
            .map_err(to_pyerr)?;
        Ok(schema_data_columns(Some(schema)))
    }

    /// Column names in `var` (the gene metadata), excluding the pandas
    /// index column. Pure Arrow IPC footer read — no batch decode.
    /// Raises if the var section cannot be read (e.g. corrupt file).
    ///
    /// Callable method (e.g. `exp.var_keys()`) to match AnnData's
    /// `adata.var_keys()`, not a property.
    fn var_keys(&self) -> PyResult<Vec<String>> {
        let schema = self
            .reader()?
            .read_var_schema_physical()
            .map_err(to_pyerr)?;
        Ok(schema_data_columns(Some(schema)))
    }

    /// Keys of the `obsm` cell-embedding mappings. Pure catalog scan.
    /// Callable method (`exp.obsm_keys()`) to match AnnData.
    fn obsm_keys(&self) -> PyResult<Vec<String>> {
        Ok(self.reader()?.list_obsm())
    }

    /// Keys of the `varm` gene-embedding mappings. Pure catalog scan.
    /// Callable method (`exp.varm_keys()`) to match AnnData.
    fn varm_keys(&self) -> PyResult<Vec<String>> {
        Ok(self.reader()?.list_varm())
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
        match self.reader()?.read_uns_for(modality_id) {
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
        convert::uns::read_uns_as_pyobject(py, self.reader()?, modality_id)
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
        let reader = self.reader()?;
        let batch = py
            .detach(|| match columns {
                Some(cols) => {
                    // Retain the pandas index column(s) so the projected frame
                    // keeps its barcode index (parity with unprojected
                    // read_obs). Including the index column also prevents a
                    // pyarrow KeyError when the schema's pandas envelope still
                    // advertises an `index_columns` entry the projection would
                    // otherwise drop (e.g. `scx convert`-produced files).
                    let schema = reader.read_obs_schema_physical()?;
                    let mut proj: Vec<String> = Vec::new();
                    for idx_col in scx_format_io::resolve_index_columns(&schema) {
                        if schema.index_of(&idx_col).is_ok() && !cols.contains(&idx_col) {
                            proj.push(idx_col);
                        }
                    }
                    proj.extend(cols);
                    reader.read_obs_keys(&proj)
                }
                None => reader.read_obs(),
            })
            .map_err(to_pyerr)?;
        let table = convert::record_batch_to_pyarrow(py, &batch)?;
        convert::pyarrow_table_to_pandas(&table)
    }

    /// Read the `var` (gene metadata) table as a pandas DataFrame **without
    /// touching X** — the var-axis mirror of [`read_obs`](Self::read_obs).
    ///
    /// `columns` selects a subset by **physical** column name (matching
    /// `var_keys()`); the pandas index column (gene names) is always retained,
    /// so a projected frame keeps the same index as the unprojected
    /// `read_var()`.
    ///
    /// Unlike `read_obs`, the projection is applied **after** the decode
    /// rather than pushed into the reader. `var` is one section sized by
    /// `n_vars` (a few MB even on an atlas — 5.5 MB for a 61k-gene Census
    /// file), where `obs` scales with `n_obs` and is worth projecting at the
    /// I/O layer. So `columns` here is a convenience, not a memory
    /// optimisation; it does not read less off disk.
    ///
    /// On a multimodal file pass `modality=<name>` to read that modality's
    /// `var`. Omitting it reads the global / single-modality `var`, which on a
    /// multimodal file is not what you usually want — each modality has its
    /// own gene axis.
    #[pyo3(signature = (columns=None, *, modality=None))]
    fn read_var<'py>(
        &self,
        py: Python<'py>,
        columns: Option<Vec<String>>,
        modality: Option<&str>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let modality_id = self.resolve_uns_modality(modality)?;
        // Decode off the GIL; only the pyarrow/pandas conversion needs Python.
        let reader = self.reader()?;
        let batch = py
            .detach(|| reader.read_var_for(modality_id))
            .map_err(to_pyerr)?;
        let batch = match columns {
            None => batch,
            Some(cols) => project_batch_columns(&batch, &cols)?,
        };
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
        let reader = self.reader()?;
        py.detach(|| reader.distinct_obs_values(col, limit, sort))
            .map_err(to_pyerr)
    }

    /// `(codes, categories)` for a single **string/categorical** `obs` column —
    /// the numpy-level accessor for building a global vocabulary or one-hot map.
    ///
    /// `codes` is an `int32` array with one entry per obs row (`-1` for missing,
    /// the `pandas.Categorical.codes` convention); `categories[code]` is the
    /// string value. Equivalent to
    /// `read_obs(columns=[col])[col].cat.codes` / `.cat.categories`, but computed
    /// shard-by-shard in Rust and returned as numpy directly — no Arrow-IPC byte
    /// round-trip, no pyarrow table, no pandas frame. X is never touched.
    ///
    /// Prefer this over `read_obs(columns=[col])` when you want codes: that path
    /// assembles and concatenates the column across shards, whereas this keeps
    /// only the running vocabulary and the output codes.
    ///
    /// Semantics:
    /// - **Category order is first-seen** across shards, not lexicographic. Sort
    ///   `categories` yourself (and remap `codes`) if you need a stable order
    ///   across files.
    /// - **Unreferenced dictionary entries are retained**, matching pandas
    ///   keeping unused levels and matching `distinct_values`' superset rule.
    /// - Both on-disk encodings are accepted, including a file that mixes them
    ///   (`from_anndata` writes dictionary-encoded shards, `append` writes plain
    ///   string shards).
    ///
    /// **Physical row space.** `len(codes) == n_obs_physical`, *not* `n_obs`. On
    /// a file with deletion vectors those differ, and indexing `codes` by a
    /// logical row id addresses the wrong cell — a correctly shaped array of
    /// wrong rows. Check `exp.n_obs == exp.n_obs_physical` before treating the
    /// codes as logical, or filter them yourself. (`read_obs` has the same
    /// contract, for the same reason.)
    ///
    /// Raises `ValueError` for non-string columns and a corrupt-file-class error
    /// for an unknown column name.
    fn obs_categorical<'py>(&self, py: Python<'py>, col: &str) -> PyResult<PyCategorical<'py>> {
        let reader = self.reader()?;
        let (codes, categories) = py
            .detach(|| reader.obs_categorical(col))
            .map_err(to_pyerr)?;
        Ok((PyArray1::from_vec(py, codes), categories))
    }

    /// `obs_categorical` for several columns in **one** shard pass.
    ///
    /// Returns a list of `(codes, categories)` in `cols` order. N columns cost
    /// one projected read per obs shard instead of N — the difference that
    /// matters when a catalog build resolves several covariate columns over a
    /// many-file manifest.
    fn obs_categorical_many<'py>(
        &self,
        py: Python<'py>,
        cols: Vec<String>,
    ) -> PyResult<Vec<PyCategorical<'py>>> {
        let reader = self.reader()?;
        let out = py
            .detach(|| reader.obs_categorical_many(&cols))
            .map_err(to_pyerr)?;
        Ok(out
            .into_iter()
            .map(|(codes, cats)| (PyArray1::from_vec(py, codes), cats))
            .collect())
    }

    /// Codec / shard / format-version internals as a one-line string.
    ///
    /// The AnnData-style `repr` lists the obs/var/obsm/uns keys a scanpy
    /// user expects; the on-disk encoding details live here instead. The
    /// `value_encoding` / `is_integer` / `max_value` tokens are also the
    /// getters of the same names; rendering them reads one 76-byte header per
    /// CSR shard (no decode), so this is O(shards), not O(1).
    fn info(&self, py: Python<'_>) -> PyResult<String> {
        let reader = self.reader()?;
        let h = reader.header();
        let (value_encoding, is_integer) = self.value_encoding_summary(py)?;
        Ok(format!(
            "SCX file: format_version={}, codec_id={}, index_dtype={}, \
             csr_shards={}, nnz={}, has_csc={}, value_encoding={}, is_integer={}, \
             max_value={}, path={}",
            h.format_version,
            h.codec_id,
            h.index_dtype,
            h.n_csr_shards,
            reader.nnz(),
            h.has_csc(),
            value_encoding,
            is_integer,
            reader.catalog().csr_max_value(None),
            self.path.display(),
        ))
    }

    /// The on-disk value encoding of the CSR shards, rendered as `scx info`
    /// prints it: a numpy dtype name (`"uint16"`, `"float32"`, …) when every
    /// shard agrees, `"mixed (uint8, uint16)"` when they differ, `"n/a"` with
    /// no shards. Reads one 76-byte header per shard and decodes nothing. On a
    /// multimodal file this folds every modality's X shards; a layer's
    /// encoding is `adata.layers[name].stored_dtype` on a backed handle.
    #[getter]
    fn value_encoding(&self, py: Python<'_>) -> PyResult<String> {
        Ok(self.value_encoding_summary(py)?.0)
    }

    /// `True` when every CSR shard is integer-encoded (`uint8` / `uint16` /
    /// `uint32`), i.e. the stored values are counts. `False` when any shard is
    /// float-encoded, and for a file with no shards. Same cost as
    /// [`Self::value_encoding`]; no values are decoded.
    #[getter]
    fn is_integer(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.value_encoding_summary(py)?.1)
    }

    /// Largest stored value across the CSR shards, from the per-shard catalog
    /// stats (no decode). Float-encoded shards record no value range, and a
    /// shard without stats contributes nothing, so this is `0` for a float
    /// file — read it together with [`Self::is_integer`]. Physical, like
    /// [`Self::nnz`]: values in logically deleted rows still count.
    #[getter]
    fn max_value(&self) -> PyResult<u32> {
        Ok(self.reader()?.catalog().csr_max_value(None))
    }

    /// `True` when the file has a CSC sidecar (gene-major shards).
    #[getter]
    fn has_csc(&self) -> PyResult<bool> {
        Ok(self.reader()?.header().has_csc())
    }

    /// `True` when the file carries logical deletion vectors — i.e. some
    /// rows are marked deleted and will be dropped (row count shrinks) on
    /// `to_anndata` / `to_h5ad` export.
    #[getter]
    fn has_deletions(&self) -> PyResult<bool> {
        Ok(self.reader()?.header().has_deletion_vectors())
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
        let prov = match self.reader()?.read_provenance() {
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
    /// `modality` scopes the query to one modality of a multimodal file: X
    /// assembly, `select_genes`, and `filter_var` resolve against that
    /// modality's var / `n_vars`, while `filter_obs` always evaluates against
    /// the shared global obs axis. On a multimodal file `modality` is
    /// **required** (omitting it raises `ValueError`); an unknown name raises
    /// `KeyError`. On a single-modality file omit `modality` (the default).
    ///
    /// Example:
    ///     result = pyscx.open("data.scx").query().collect()
    ///     rna = pyscx.open("cite.scx").query(modality="rna") \
    ///               .filter_obs("cell_type == 'T cell'").collect()
    #[pyo3(signature = (modality=None))]
    fn query(&self, modality: Option<&str>) -> PyResult<PyQueryPipeline> {
        // Resolve the modality name → 1-based id (0 = global) using the
        // already-open reader (mirrors `open_backed_csr`). On a multimodal file
        // a modality is required; on a single-modality file it must be omitted.
        let reader = self.reader()?;
        let modality_id: u8 = match modality {
            None => {
                if reader.is_multimodal() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "file is multimodal; pass modality=... (one of {:?})",
                        reader.modality_names()
                    )));
                }
                0
            }
            Some(name) => reader.modality_id(name).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!(
                    "unknown modality '{name}'; available: {:?}",
                    reader.modality_names()
                ))
            })?,
        };
        let pipeline = QueryPipeline::from_reader_for_modality(
            Box::new(crate::open_handle_reader(&self.path).map_err(to_pyerr)?),
            modality_id,
        )
        .map_err(crate::query::engine_to_pyerr)?;
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

        let expected = self.reader()?.n_obs() as usize;
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

        // This handle just mutated its own file, so re-open it rather than
        // leaving it to fail the freshness check on the next read (finding
        // 9.6). Shared with `reload()`.
        self.reopen()?;

        Ok(total)
    }

    /// Re-open the file, picking up anything written since this handle was
    /// opened.
    ///
    /// The way back from a `RuntimeError: '…' changed on disk`. The handle
    /// keeps its identity — anything holding a reference to it sees the new
    /// contents — but every object it handed out earlier (a backed `AnnData`,
    /// a `query()` pipeline) has its own reader and is *not* reloaded by this;
    /// re-derive those from the reloaded handle.
    ///
    /// Never raises for being stale; that is what it is for. It can still
    /// raise if the file is now unreadable or gone.
    ///
    /// Example:
    ///     exp = pyscx.open("atlas.scx")
    ///     pyscx.obs_import("atlas.scx", "calls.csv", key="obs_names")
    ///     exp.reload()
    ///     exp.read_obs()          # now carries the imported columns
    fn reload(&mut self) -> PyResult<()> {
        self.reopen()
    }

    /// Release the file mapping. Idempotent.
    ///
    /// Reads afterwards raise instead of answering. Worth calling before
    /// rewriting the file in place — and required on Windows, where a mapped
    /// file cannot be replaced at all. Dropping the last reference does the
    /// same thing, but on the interpreter's schedule rather than yours.
    ///
    /// Objects this handle handed out (a backed `AnnData`, a `query()`
    /// pipeline) hold their own readers and keep working after it: closing the
    /// handle you opened them from does not close them.
    fn close(&mut self) {
        self.reader = None;
        self.grouped_pipeline = std::sync::OnceLock::new();
        self.n_deleted = std::sync::OnceLock::new();
        self.value_encoding_memo = std::sync::OnceLock::new();
    }

    /// Whether [`close`](Self::close) has been called.
    #[getter]
    fn closed(&self) -> bool {
        self.reader.is_none()
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        self.close();
        // Never swallow an exception raised inside the block.
        false
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
    #[pyo3(signature = (backed=false, cache_shards=4, var_names=None, obs_filter=None, layers=None, preserve_slots=false, modality=None, eager=false, memory_budget=None, obsm=None, preserve_var_order=false, strict_var_names=true, container="csr", data_dtype=None, index_dtype=None, allow_lossy=false))]
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
        container: &str,
        data_dtype: Option<&str>,
        index_dtype: Option<&str>,
        allow_lossy: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        materialize::to_anndata_impl(
            self,
            py,
            backed,
            cache_shards,
            var_names,
            obs_filter,
            layers,
            preserve_slots,
            modality,
            eager,
            memory_budget,
            obsm,
            preserve_var_order,
            strict_var_names,
            container,
            data_dtype,
            index_dtype,
            allow_lossy,
        )
    }

    /// Return a **GPU-resident** AnnData whose `X` is a
    /// `cupyx.scipy.sparse.csr_matrix` decoded onto the device. This is the
    /// cheapest path from SCX-on-disk to a GPU matrix
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
    #[pyo3(signature = (var_names=None, obs_filter=None, layers=None, obsm=None, device="gpu", memory_budget=None, preserve_var_order=false, strict_var_names=true, container="csr", data_dtype=None, index_dtype=None, allow_lossy=false))]
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
        container: &str,
        data_dtype: Option<&str>,
        index_dtype: Option<&str>,
        allow_lossy: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Before the feature gate: a stale handle is stale regardless of how
        // this build was compiled.
        self.reader()?;
        // F3 Phase 1: the device path is f32-native. Accept the kwargs for API
        // parity with `to_anndata`, but reject a non-default plan (host-side
        // narrowing before device upload is a later phase).
        let plan = convert::build_plan(py, container, data_dtype, index_dtype, allow_lossy)?;
        if !plan.is_default_csr_f32() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "to_gpu_anndata does not yet support container / data_dtype / index_dtype \
                 (the device path is f32-native); materialize on the host with \
                 to_anndata(container=..., data_dtype=...) instead",
            ));
        }
        #[cfg(feature = "gpu")]
        {
            gpu_anndata::to_gpu_anndata_impl(
                self,
                py,
                var_names,
                obs_filter,
                layers,
                obsm,
                device,
                memory_budget,
                preserve_var_order,
                strict_var_names,
                &plan,
            )
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
    /// v3 canonical CSR invariant; results are appended with `canonical-csr `
    /// prefixed names. Mirrors `scx validate --deep`. (The decode-sidecar
    /// representation was removed, so there is no sidecar check.)
    ///
    /// Returns a list of (section_name, passed) tuples.
    #[pyo3(signature = (deep=false))]
    fn validate(&self, py: Python<'_>, deep: bool) -> PyResult<Vec<(String, bool)>> {
        let reader = self.reader()?;
        let mut results = reader.validate().map_err(to_pyerr)?;
        if deep {
            // Deep validation re-decodes every shard (CPU-bound, pure Rust) —
            // run it off the GIL so other Python threads aren't blocked.
            py.detach(|| crate::deep_validate_into(reader, &mut results));
        }
        Ok(results)
    }

    /// True if this file is multimodal (Phase B / v2 with
    /// `n_modalities > 0`). Mirrors `header.has_modalities()`.
    #[getter]
    fn is_multimodal(&self) -> PyResult<bool> {
        Ok(self.reader()?.is_multimodal())
    }

    /// Number of registered modalities (0 for v1 files and
    /// single-modality v2 files).
    #[getter]
    fn n_modalities(&self) -> PyResult<u32> {
        Ok(self.reader()?.n_modalities())
    }

    /// Ordered list of modality names (empty for single-modality
    /// files). The position in the list maps 1:1 to the 1-based
    /// modality_id (`names[i] -> modality_id = i + 1`).
    #[getter]
    fn modality_names(&self) -> PyResult<Vec<String>> {
        Ok(self
            .reader()?
            .modality_names()
            .iter()
            .map(|s| s.to_string())
            .collect())
    }

    /// Resolve a modality name to its 1-based `modality_id`.
    /// Returns `None` for unknown names or single-modality files.
    fn modality_id(&self, name: &str) -> PyResult<Option<u8>> {
        Ok(self.reader()?.modality_id(name))
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
        let info = match self.reader()?.modality_info(modality_id) {
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
    /// Eager narrowing (`container` / `data_dtype` / `index_dtype`): each of the
    /// three accepts **either** a scalar (applied to every modality) **or** a
    /// dict keyed by modality name (e.g. `data_dtype={"rna": "uint16", "atac":
    /// "uint8"}`); a dict key naming no modality raises `ValueError`. A modality
    /// with no override keeps the byte-identical zero-copy `f32` CSR path.
    /// `container="dense"` is not yet supported (CSR only). These are rejected
    /// under `backed=True` (backed X stays lazily f32-native, like `to_anndata`).
    #[pyo3(signature = (backed=false, cache_shards=4, container=None, data_dtype=None, index_dtype=None, allow_lossy=false))]
    #[allow(clippy::too_many_arguments)]
    fn to_mudata<'py>(
        &self,
        py: Python<'py>,
        backed: bool,
        cache_shards: usize,
        container: Option<Bound<'py, PyAny>>,
        data_dtype: Option<Bound<'py, PyAny>>,
        index_dtype: Option<Bound<'py, PyAny>>,
        allow_lossy: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if backed {
            // Backed mode decodes lazily per-slice as f32, so a non-default plan
            // can't be honoured (mirrors backed to_anndata). Reject rather than
            // silently ignore the shaping args.
            if container.is_some() || data_dtype.is_some() || index_dtype.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "to_mudata(container=/data_dtype=/index_dtype=) require backed=False; \
                     backed multimodal X is lazily f32-native",
                ));
            }
            crate::mudata::to_mudata_backed(py, &self.path, self.reader()?, cache_shards)
        } else {
            crate::mudata::to_mudata(
                py,
                self.reader()?,
                container.as_ref(),
                data_dtype.as_ref(),
                index_dtype.as_ref(),
                allow_lossy,
            )
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
        materialize::detection_counts_impl(self, py, axis, modality)
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
        materialize::cells_expressing_impl(self, py, gene, modality)
    }

    /// Gather specific rows as a sparse `scipy.sparse.csr_matrix`, in the
    /// requested order.
    ///
    /// The bounded shard-wise gather: each touched shard is decoded once (a
    /// sparse request on a row-group-framed shard decodes only the touched row
    /// groups), and the result is assembled **once** into exact-size buffers in
    /// request order — peak memory is the result plus the shard cache, plus up
    /// to `cache_shards` shards decoding in flight while that cache fills (at
    /// most `2 × cache_shards` decoded shards beside the result), never a
    /// second copy of the result. `rows` is a boolean mask or
    /// any 1-D integer array-like (list, range, ndarray of any integer dtype);
    /// it may contain duplicates and need not be sorted; negative indices wrap
    /// once. Returns raw-local gene indices (no global-vocab remap). On
    /// multimodal files, pass `modality=`.
    ///
    /// `logical=True` (default) indexes the rows `Experiment.n_obs` /
    /// `read_obs()` describe — deletion vectors applied, as
    /// `to_anndata(backed=True).X[rows]` does. `logical=False` indexes the
    /// physical file rows (`n_obs_physical`), deleted cells included. Changed in
    /// 0.17: the method used to address physical rows only, so on a file with
    /// deletion vectors the same ids now select different cells.
    /// `layer=` gathers from that layer's shard family instead of `X`
    /// (`ValueError` if absent; not supported together with a multimodal file).
    ///
    /// `cache_shards` bounds peak decoded-shard memory for this gather (it is
    /// not a speedup knob: within a single call each shard is decoded exactly
    /// once, so the LRU never serves a repeat hit). A fresh reader is opened
    /// per call — intentional, so the method is fork-safe and stateless; this
    /// is the eval / random-access utility, not the training hot path (use
    /// `SparseCellSetDataset` for that).
    ///
    /// Out-of-range row ids raise `IndexError`, as does a boolean mask whose
    /// length is not the row count. This is a drop-in for the backed
    /// `adata.X[rows]` analysis path and the random-access utility an
    /// `IterableDataset` cannot serve.
    #[pyo3(signature = (rows, modality = None, cache_shards = 4, layer = None, logical = true))]
    fn gather_rows_sparse<'py>(
        &self,
        py: Python<'py>,
        rows: &Bound<'py, PyAny>,
        modality: Option<&str>,
        cache_shards: usize,
        layer: Option<&str>,
        logical: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        materialize::gather_rows_sparse_impl(self, py, rows, modality, cache_shards, layer, logical)
    }

    /// Never raises. A repr that throws turns every later traceback into a
    /// second, unrelated error — precisely when the state it would have
    /// described is what you needed to see. So a closed or stale handle
    /// renders as such rather than refusing.
    fn __repr__(&self) -> String {
        let Some(reader) = self.reader.as_ref() else {
            return format!("<Experiment '{}' [closed]>", self.path.display());
        };
        if let Err(e) = reader.check_fresh() {
            return format!(
                "<Experiment '{}' [stale: {}]>",
                self.path.display(),
                stale_repr_detail(&e),
            );
        }
        // Best-effort: the repr must always render, so a failed schema read
        // degrades to an empty key list here (the public `obs_keys` /
        // `var_keys` getters surface the error loudly instead).
        format_anndata_repr(
            "Experiment",
            self.logical_n_obs_of(reader),
            reader.n_vars(),
            &[
                (
                    "obs",
                    schema_data_columns(reader.read_obs_schema_physical().ok()),
                ),
                (
                    "var",
                    schema_data_columns(reader.read_var_schema_physical().ok()),
                ),
                ("uns", self.uns_keys(None).unwrap_or_default()),
                ("obsm", self.obsm_keys().unwrap_or_default()),
                ("varm", self.varm_keys().unwrap_or_default()),
                ("layers", self.layer_names().unwrap_or_default()),
            ],
        )
    }
}

/// `scx info`'s rendering of a distinct, sorted set of shard-header
/// `value_encoding` bytes — the single name when uniform, `mixed (a, b)` when
/// shards differ, `n/a` when there are none — plus whether every one is an
/// integer encoding (`false` for an empty set). Shared by the local and cloud
/// `Experiment` so `info()` stays byte-identical between them.
pub(crate) fn render_value_encodings(distinct: &[u8]) -> (String, bool) {
    use scx_codec::ValueEncoding;
    let name = |b: &u8| ValueEncoding::from_u8(*b).map_or("unknown", |v| v.numpy_name());
    let rendered = match distinct {
        [] => "n/a".to_string(),
        [one] => name(one).to_string(),
        many => format!(
            "mixed ({})",
            many.iter().map(name).collect::<Vec<_>>().join(", ")
        ),
    };
    let is_integer = !distinct.is_empty()
        && distinct
            .iter()
            .all(|b| ValueEncoding::from_u8(*b).is_some_and(|v| v.is_integer()));
    (rendered, is_integer)
}

impl PyExperiment {
    /// [`render_value_encodings`] over this file's CSR shard headers, memoised.
    /// The freshness check is `reader()?`; the first fold runs with the GIL
    /// released.
    fn value_encoding_summary(&self, py: Python<'_>) -> PyResult<(String, bool)> {
        let reader = self.reader()?;
        if let Some(memo) = self.value_encoding_memo.get() {
            return Ok(memo.clone());
        }
        let folded = py
            .detach(|| -> scx_format_io::Result<(String, bool)> {
                let shards = reader
                    .catalog()
                    .shards(scx_format_io::SectionType::CsrShard);
                let distinct = scx_format_io::distinct_sorted_shard_field(reader, &shards, |h| {
                    h.value_encoding
                })?;
                Ok(render_value_encodings(&distinct))
            })
            .map_err(to_pyerr)?;
        Ok(self.value_encoding_memo.get_or_init(|| folded).clone())
    }
}
