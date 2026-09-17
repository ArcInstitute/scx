//! `SparseCellSetDataset` / `SparseCellSetBatchIter` — the native cell-set loader.
//!
//! Split out of `python.rs` by ORG-9.10-2; a pure move.

use std::sync::Arc;

use numpy::PyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::{PyDict, PyTuple};
use scx_format_io::CacheMetrics;

use crate::error::LoaderError;
use crate::plan_engine::IterMetrics;
use crate::sparse_cellset::{SparseCellSetBatch, SparseCellSetLoader, SparseCellSetPlan};

use super::*;

/// `(file_ids, rows, role_tags, set_offsets)` — the cell-set loader's plan.
fn extract_cellset_plan(
    obj: &Bound<'_, PyAny>,
) -> std::result::Result<SparseCellSetPlan, LoaderError> {
    let (file_ids, rows, role_tags, set_offsets) = obj
        .extract::<(Vec<u32>, Vec<u64>, Vec<i32>, Vec<i64>)>()
        .map_err(|e| LoaderError::ConfigError {
            reason: format!(
                "sparse plan extraction failed (expected a tuple \
                 (file_ids:u32[], rows:u64[], role_tags:i32[], set_offsets:i64[])): {e}"
            ),
        })?;
    Ok(SparseCellSetPlan {
        file_ids,
        rows,
        role_tags,
        set_offsets,
    })
}

// ---------------------------------------------------------------------------
// SparseCellSetDataset — native sparse cell-set loader (SCX-DATA-LOADER §4)
// ---------------------------------------------------------------------------

/// Plan-driven native sparse cell-set reader. Each plan item is one batch of
/// cell sets (delimited by `set_offsets`); each yielded dict carries the
/// SCX-DATA-LOADER §4.4 sparse contract. Sibling to `IndexPlanDataset` but
/// emits sparse CSR (not dense pairs) and is multi-file.
#[pyclass]
pub struct SparseCellSetDataset {
    /// `None` once [`SparseCellSetDataset::close`] has run — see
    /// [`IndexPlanDataset::loader`] for why this is an `Option`.
    loader: Option<Arc<SparseCellSetLoader>>,
    default_lookahead: usize,
    /// PID at construction — the shard cache / mmap state is not fork-safe, so
    /// the dataset must be built post-fork in each worker (or `num_workers=0`).
    creation_pid: u32,
}

impl SparseCellSetDataset {
    /// The live loader, or a `RuntimeError` when the dataset has been closed.
    /// Deliberately **not** in the `#[pymethods]` block — a method there would
    /// be exported to Python as `ds.loader`.
    fn loader(&self) -> PyResult<&Arc<SparseCellSetLoader>> {
        self.loader.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "SparseCellSetDataset is closed. close() is terminal on this class: \
                 the tokio runtime is built exactly once so it can never be inherited \
                 across a fork, and so cannot be rebuilt. Construct a new dataset.",
            )
        })
    }

    /// Shared teardown for `close` and `Drop`. **Must be called without the
    /// GIL**: releasing the last reference tears the tokio runtime down, which
    /// blocks on in-flight shard decodes.
    ///
    /// Just a drop — deliberately. An earlier version tried `Arc::into_inner`
    /// here so it could call a bounded shutdown when it was the last owner, and
    /// that is not decidable from this side: the batch iterators, the caller's
    /// `process` closure and (previously) the prefetch tasks all hold
    /// references. The deadline now lives in `BoundedRuntime::drop`, so the
    /// last release bounds itself wherever it happens. See [`crate::runtime`].
    fn shutdown_detached(loader: Option<Arc<SparseCellSetLoader>>) {
        drop(loader);
    }
}

#[pymethods]
impl SparseCellSetDataset {
    /// Args:
    ///     scatter_block_index: Per-dataset escape hatch (default: **False** —
    ///         the opposite of `IndexPlanDataset`) gating the block-index-aware
    ///         gather and its prefetch warm-skip for row-group-framed (v4)
    ///         files. False warms whole shards into the LRU and serves batches
    ///         from cache; True decodes only the touched row-groups per batch.
    ///         Off by default because the cell-set regime is cache-*friendly*
    ///         (sorted data + a reused control pool → a small working set that
    ///         fits the shard cache and is touched most batches). Measured on
    ///         a 50-file Tahoe atlas: steps/s 2.80 → 4.55 and gather 337 ms →
    ///         ~5 ms with it off, at ≈ .h5ad parity.
    ///         **Re-measured after OPT-FORMATIO-1**, with the cache sized to
    ///         the file, as a 2×2 over corpus size × plan locality (sets/s,
    ///         off vs on): tabula 100k/7 shards 899 vs 5.4 random and 812 vs
    ///         6.9 grouped; census_1m/62 shards 11.4 vs 6.9 random and **300
    ///         vs 459 grouped**. So there is no fixed winner — full-shard
    ///         degrades with corpus size while block-index is roughly flat and
    ///         tracks plan locality, and they cross somewhere near 1M cells.
    ///         The default stays False because flipping it costs 117–167× on
    ///         100k-cell corpora, which is the common case; the right answer at
    ///         scale is per-dataset route selection, not a different constant.
    ///         Pass True for a large corpus with local plans, or for a
    ///         genuinely cache-hostile run (working set ≫ cache) where the
    ///         row-group-scoped decode's bounded peak RAM is the memory-safe
    ///         choice. `SCX_SCATTER_BLOCK_INDEX=0` is the process-wide
    ///         reader-layer kill-switch over both settings.
    ///     max_plan_rows: Upper bound on rows per plan. Charges one gathered
    ///         batch against `max_memory_mb` before the shard cache is sized,
    ///         AND refuses a plan wider than it — a cache sized for `N` rows
    ///         while the gather accepts any width is not a bound.
    ///         Default `None` — **uncharged**, and the resolved cache is then
    ///         byte-identical to what it was before this argument existed. This
    ///         class has no `max_plan_size`: plan width is the caller's, so
    ///         only the caller can say what a batch costs, and a guessed
    ///         default would silently shrink the cache on every existing
    ///         dataset. What it bounds is the plan's **row count**, not its
    ///         bytes: the charge uses the manifest's mean density, so a plan of
    ///         denser-than-average rows can still exceed it. `memory_budget()`
    ///         reports the term as `breakdown["batch_buffer_bytes"]`.
    ///     reader_limit: Cap on how many of the manifest's files are open at
    ///         once. Default `None` — open every file and never close one,
    ///         which is what this class has always done and is byte-identical
    ///         in sizing, throughput and gather output. What it bounds is
    ///         **resident memory, not file descriptors**: `ScxReader::open`
    ///         mmaps and closes the descriptor, so an N-file manifest holds N
    ///         mappings and no descriptors — measured, a 5,000-file manifest
    ///         constructs and gathers under a 1024 descriptor limit with the
    ///         process's descriptor count flat. What an open reader does cost
    ///         is ~104 kB resident, over 90% of it the parsed `FullCatalog`;
    ///         at 26k files that is ~2.8-3.2 GB per process, before multiplying by
    ///         DataLoader workers and ranks. The saving is real only because
    ///         an eviction drops that catalog and a reopen re-parses it
    ///         (0.09-20 ms per file), so set this when the manifest is large
    ///         enough for the memory to matter and plans have locality, and
    ///         leave it `None` otherwise. It caps handles the registry is free
    ///         to drop, so a plan leasing more files at once than the limit
    ///         exceeds it rather than blocking; `cache_metrics()["reader_hwm"]`
    ///         reports what actually happened.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        paths,
        cache_shards=None,
        max_memory_mb=None,
        lookahead=None,
        remap_tables=None,
        n_global_genes=None,
        normalize=None,
        log1p=None,
        target_sum=None,
        downsample_target_library_size=None,
        downsample_method=None,
        downsample_seed=None,
        scatter_block_index=None,
        max_plan_rows=None,
        reader_limit=None,
    ))]
    fn new(
        py: Python<'_>,
        paths: Vec<String>,
        cache_shards: Option<usize>,
        max_memory_mb: Option<usize>,
        lookahead: Option<usize>,
        remap_tables: Option<Vec<Vec<i32>>>,
        n_global_genes: Option<usize>,
        normalize: Option<bool>,
        log1p: Option<bool>,
        target_sum: Option<f64>,
        downsample_target_library_size: Option<u64>,
        downsample_method: Option<String>,
        downsample_seed: Option<u64>,
        scatter_block_index: Option<bool>,
        max_plan_rows: Option<usize>,
        reader_limit: Option<usize>,
    ) -> PyResult<Self> {
        if paths.is_empty() {
            return Err(PyRuntimeError::new_err(
                "SparseCellSetDataset requires at least one .scx path",
            ));
        }
        let cache_shards = cache_shards.unwrap_or(128);
        let lookahead = lookahead.unwrap_or(4);
        // `None` is now adaptive (bounded), not `usize::MAX` (unbounded).
        let bytes_budget = max_memory_mb.map(|mb| mb.saturating_mul(1024 * 1024));
        let normalize = normalize.unwrap_or(false);
        let log1p = log1p.unwrap_or(false);
        let target_sum = target_sum.unwrap_or(1e4);
        // Default OFF, unlike `IndexPlanDataset`, and re-measured after the
        // row-group LRU (OPT-FORMATIO-1) rather than inherited: full-shard wins
        // 167× on tabula random and 117× grouped, 1.65× on census_1m random,
        // and LOSES 1.53× on census_1m grouped. The two routes cross near 1M
        // cells — full-shard degrades with corpus size, block-index is flat in
        // it and tracks plan locality — so `false` is the right default for the
        // 100k-cell common case, not a universal answer. The general fix is
        // per-dataset route selection off the per-plan `planned` footprint the
        // engine already computes, not a different constant here.
        let scatter_block_index = scatter_block_index.unwrap_or(false);
        // A zero bound would charge nothing and reject every non-empty plan —
        // the two halves of the contract pointing opposite ways. `ValueError`
        // for the same reason as the downsample checks below: it is a bad
        // *value*, and `IndexPlanLoader` refuses `max_plan_size < 1` likewise.
        if max_plan_rows == Some(0) {
            return Err(PyValueError::new_err(
                "max_plan_rows must be >= 1 (pass None to leave the batch uncharged)",
            ));
        }
        // Same shape: a gather needs at least one open reader, so a cap of zero
        // is a bad value rather than a tighter bound.
        if reader_limit == Some(0) {
            return Err(PyValueError::new_err(
                "reader_limit must be >= 1 (pass None to keep every file open)",
            ));
        }
        // Raised as `ValueError`, not the `loader_err_to_py` default of
        // `RuntimeError`: every argument check on this path is a bad *value*,
        // and a caller catching malformed input would otherwise have to catch
        // two exception types to cover one constructor. Same reasoning, and the
        // same shape, as the `validate_indptr` call in `downsample_counts_csr`.
        // Built before the open below takes ownership of `paths`. A count plus
        // up to two names: actionable without being a wall of paths, and enough
        // text for Python's per-message dedup to make the warning one-shot per
        // dataset. String formatting only — it costs nothing to build eagerly,
        // and the alternative is keeping a clone of every path alive for a
        // warning that usually does not fire.
        let unframed_target = match paths.as_slice() {
            [one] => format!("'{one}'"),
            _ => {
                let shown: Vec<String> = paths.iter().take(2).map(|p| format!("'{p}'")).collect();
                let ellipsis = if paths.len() > 2 { ", …" } else { "" };
                format!("{} files ({}{})", paths.len(), shown.join(", "), ellipsis)
            }
        };

        // Everything from here to the loader is pure Rust over owned data —
        // `paths` and `remap_tables` are already `Vec`s and no `Bound` crosses
        // — and it is not cheap: a stat per path to resolve the downsample
        // config, an mmap and catalog parse per file, then `ScxReader`-wide
        // range validation and the budget tune. Holding the GIL through it
        // stalls every other Python thread for the whole open, which on a
        // many-file manifest is the dominant cost of constructing the dataset.
        //
        // The error is carried out of the closure and turned into a `PyErr`
        // after re-attaching, since building one needs the GIL. Each arm keeps
        // the exception type it had: `ValueError` for a bad downsample value,
        // `RuntimeError` for everything the loader reports — including a failed
        // open, which now reaches us as a `ConfigError` naming the path,
        // because the manifest scan owns the opening and it is the thing that
        // knows how many files may be open at once.
        enum OpenError {
            Downsample(String),
            Loader(crate::error::LoaderError),
        }
        let loader = py
            .detach(move || {
                let downsample = crate::downsample::resolve_downsample_config(
                    &paths,
                    downsample_target_library_size,
                    downsample_method.as_deref(),
                    downsample_seed,
                )
                .map_err(|e| OpenError::Downsample(e.to_string()))?;

                SparseCellSetLoader::open(
                    paths.iter().map(std::path::PathBuf::from).collect(),
                    cache_shards,
                    bytes_budget,
                    lookahead,
                    remap_tables,
                    n_global_genes,
                    normalize,
                    log1p,
                    target_sum,
                    downsample,
                    scatter_block_index,
                    max_plan_rows,
                    reader_limit,
                )
                .map_err(OpenError::Loader)
            })
            .map_err(|e| match e {
                OpenError::Downsample(m) => PyValueError::new_err(m),
                OpenError::Loader(e) => loader_err_to_py(e),
            })?;

        if let Some(v) = loader.cache_sizing() {
            warn_cache_sizing(py, "SparseCellSetDataset", &v)?;
        }
        if should_warn_unframed_scatter(
            scatter_block_index,
            scx_format_io::backed::scatter_block_index_enabled(),
            || loader.any_shard_framed(),
        ) {
            warn_unframed_scatter(py, "SparseCellSetDataset", &unframed_target)?;
        }

        Ok(Self {
            loader: Some(loader),
            default_lookahead: lookahead,
            creation_pid: std::process::id(),
        })
    }

    /// Number of `.scx` files (the `file_id` range).
    #[getter]
    fn n_files(&self) -> PyResult<usize> {
        Ok(self.loader()?.n_files())
    }

    /// CSR column count of emitted batches (max per-file `n_vars`, or the
    /// global vocab size when remap tables were supplied).
    #[getter]
    fn n_cols(&self) -> PyResult<usize> {
        Ok(self.loader()?.n_cols())
    }

    /// True once [`Self::close`] has run. Never raises.
    #[getter]
    fn closed(&self) -> bool {
        self.loader.is_none()
    }

    /// Release the prefetch engine's tokio runtime, bounded by a 5 s deadline,
    /// with the GIL detached. See [`IndexPlanDataset::close`] — same hazard,
    /// same terminal semantics, same best-effort bound when a live
    /// `SparseCellSetBatchIter` still holds a reference.
    fn close(&mut self, py: Python<'_>) {
        let loader = self.loader.take();
        py.detach(|| Self::shutdown_detached(loader));
    }

    /// Drive the loader from a Python iterable of batch plans, each a tuple
    /// `(file_ids, rows, role_tags, set_offsets)`; return an iterator of
    /// sparse batch dicts (the §4.4 contract).
    #[pyo3(signature = (plans, lookahead=None))]
    fn iter_with_plans(
        &self,
        plans: Py<PyAny>,
        lookahead: Option<usize>,
    ) -> PyResult<SparseCellSetBatchIter> {
        if std::process::id() != self.creation_pid {
            return Err(PyRuntimeError::new_err(
                "scx.SparseCellSetDataset requires num_workers=0 (or lazy per-worker \
                 construction). The Rust shard cache and mmap state are not fork-safe.",
            ));
        }
        // Before touching the caller's object: a closed dataset must raise the
        // terminal-state error, not run arbitrary user `__iter__` code first.
        let loader = self.loader()?;
        let lookahead = lookahead.unwrap_or(self.default_lookahead);

        let py_iter: Py<PyAny> = Python::attach(|py| -> PyResult<Py<PyAny>> {
            Ok(plans.bind(py).call_method0("__iter__")?.unbind())
        })?;

        let plan_stream = PyPlanIterator {
            py_iter,
            extract: extract_cellset_plan,
        };
        let inner = Arc::clone(loader).iter_with_plans(plan_stream, lookahead);
        let iter_metrics = inner.iter_metrics();
        Ok(SparseCellSetBatchIter {
            inner: Some(inner),
            iter_metrics,
            cache_metrics: loader.cache_metrics(),
            reader_metrics: loader.reader_metrics(),
            thrash: ThrashSampler::new(
                "SparseCellSetDataset",
                // The *affordable* count, not the requested one: on a large-shard
                // file the byte budget binds first, and a warning that says
                // "exceeds cache_shards=128" while the budget only holds 8 sends
                // the caller to raise a knob that cannot help.
                loader.effective_cache_shards(),
                loader.total_shards(),
                loader.effective_cache_shards() < loader.cache_shards(),
                loader.shard_decoded_bytes(),
            ),
        })
    }

    /// Gather **one** plan synchronously and return its batch dict.
    ///
    /// The same `(file_ids, rows, role_tags, set_offsets)` plan
    /// `iter_with_plans` consumes, and the same §4.4 batch dict it yields, for
    /// a caller that has one plan in hand and no stream to drive:
    ///
    /// ```python
    /// batch = ds.gather(*plan)
    /// ```
    ///
    /// **Admission is decided per call.** The streaming path takes one
    /// row-group verdict per plan across every file and shard the prefetcher
    /// will touch, and compares it against its divided share of the budget
    /// (`budget / (lookahead + 1)`), because a lookahead window's worth of
    /// plans has to coexist. A standalone `gather` has no such window, so it
    /// takes one verdict over the plan against the WHOLE budget. Identical
    /// output either way — the verdict changes what the cache *retains*, not
    /// what is read — but a plan that retains nothing under `iter_with_plans`
    /// may retain here, and the batch is gathered on the calling thread rather
    /// than a prefetched one, so it is not the way to drive an epoch.
    ///
    /// Raises the same errors as the iterator, from the same validation:
    /// `RuntimeError` for a malformed plan or a cross-file set with no
    /// `remap_tables`, `IndexError` for a row past a file's `n_obs`.
    fn gather<'py>(
        &self,
        py: Python<'py>,
        file_ids: Bound<'py, PyAny>,
        rows: Bound<'py, PyAny>,
        role_tags: Bound<'py, PyAny>,
        set_offsets: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if std::process::id() != self.creation_pid {
            return Err(PyRuntimeError::new_err(
                "scx.SparseCellSetDataset requires num_workers=0 (or lazy per-worker \
                 construction). The Rust shard cache and mmap state are not fork-safe.",
            ));
        }
        // Closed-state check before the plan is extracted, as on `iter_with_plans`.
        let loader = Arc::clone(self.loader()?);

        // Repacked into the tuple `extract_cellset_plan` takes rather than
        // extracted per argument, so a malformed plan raises the one message
        // that names the whole expected shape — the same text the iterator
        // raises, which `test_plan_extraction_failure_names_the_expected_tuple`
        // pins.
        let tuple = PyTuple::new(py, [file_ids, rows, role_tags, set_offsets])?;
        let plan = extract_cellset_plan(tuple.as_any()).map_err(loader_err_to_py)?;

        let batch = py
            .detach(move || loader.gather(&plan))
            .map_err(loader_err_to_py)?;

        sparse_cellset_batch_to_dict(py, batch)
    }

    /// Number of distinct `(file_id, shard)` pairs a cell-set plan touches — the
    /// `cache_shards` that would let the whole batch stay resident.
    ///
    /// Pure index arithmetic from the catalogs' shard row ranges: no I/O, no
    /// decode. Takes **one plan**, in the same
    /// `(file_ids, rows, role_tags, set_offsets)` shape `iter_with_plans`
    /// consumes — `role_tags` and `set_offsets` are ignored, since shard
    /// residency depends only on which rows of which files are read — so a
    /// caller can size the cache from the plans it is about to issue without
    /// destructuring them (ORG-9.10-4; it used to take `file_ids` and `rows` as
    /// two separate arrays, which was the same method under a different arity
    /// from the one `IndexPlanDataset` exposes). A plan of the wrong arity — the
    /// pair loader's two-array shape, say — raises `ValueError` from pyo3's
    /// tuple extraction:
    ///
    /// ```python
    /// probe = pyscx.SparseCellSetDataset(paths)
    /// need = max(probe.suggested_cache_shards(p) for p in plans[:64])
    /// ds = pyscx.SparseCellSetDataset(paths, cache_shards=need)
    /// ```
    ///
    /// This is the measurement STATE3's 143 s/batch regime needed: its scattered
    /// perturbation gather touched far more shards than the `cache_shards=16` it
    /// was passing, and raising the cache to 48 was worth ~19×.
    fn suggested_cache_shards<'py>(
        &self,
        py: Python<'py>,
        plan: (Vec<u32>, Vec<u64>, Bound<'py, PyAny>, Bound<'py, PyAny>),
    ) -> PyResult<usize> {
        // Closed-state check before the manual same-length validation below,
        // so a closed dataset reports that rather than a ValueError about its
        // arguments. It cannot precede *all* argument checking: pyo3 converts
        // the tuple parameter before this body runs, so a closed dataset handed
        // a two-tuple still gets the arity `ValueError`.
        let loader = self.loader()?;
        // `role_tags` / `set_offsets` are taken as bare objects, not extracted:
        // the tuple arity is still checked (a 2-tuple is rejected, which is the
        // shape mistake worth catching), but nothing copies two Python
        // sequences the touch count never reads. The documented recipe runs
        // this over 64 plans.
        let (file_ids, rows, _role_tags, _set_offsets) = plan;
        if file_ids.len() != rows.len() {
            return Err(PyValueError::new_err(format!(
                "file_ids and rows must be the same length, got {} and {}",
                file_ids.len(),
                rows.len()
            )));
        }
        Ok(py.detach(|| loader.plan_shard_touch_count(&file_ids, &rows)))
    }

    /// Resolved shard-cache budget:
    ///
    /// ```text
    /// breakdown              - per-component estimate, same six keys as every
    ///                          other class that reports a budget
    /// max_memory_mb          - byte budget in force (adaptive when not passed)
    /// cache_shards           - requested count cap
    /// effective_cache_shards - shards the byte budget holds at average size
    /// shard_decoded_bytes    - average decoded bytes per CSR shard
    /// max_plan_rows          - declared rows-per-plan bound, or None
    /// mean_nnz_per_row       - manifest density the batch charge uses
    /// max_blocking_threads   - cap on simultaneous shard decodes
    /// ```
    ///
    /// `cache_bytes` and `python_overhead_bytes` are the non-zero terms: on this
    /// path the shard cache *is* the budget, with no plan-tuple staging. Pass
    /// `max_plan_rows` and `batch_buffer_bytes` joins them — one gathered CSR
    /// batch that wide at `mean_nnz_per_row`, subtracted before the cache is
    /// sized, so `effective_cache_shards` falls accordingly.
    ///
    /// `budget_exceeded` is `True` when even a one-shard cache does not fit.
    /// Before ORG-9.10-5 `total_bytes` could exceed `max_memory_mb` routinely,
    /// because the tuner did not count the interpreter constant the report
    /// included.
    ///
    /// That is a statement about **the cache this loader sizes**, not a ceiling
    /// on process RSS: the batch's transients are never charged and the batch
    /// itself only when `max_plan_rows` says how wide plans get (this path has
    /// no `max_plan_size`, so plan output size is otherwise caller-controlled
    /// and unbounded), and the shared LRU keeps a single oversize shard rather
    /// than refusing to cache it, so one above-average shard can sit above the
    /// byte cap.
    ///
    /// `max_blocking_threads` is reported beside them because it is the other
    /// thing standing between a wide plan and unbounded transient memory: it
    /// caps how many shard decodes run at once. `lookahead` bounds in-flight
    /// *plans*, not the tasks a plan spawns.
    ///
    /// ORG-9.10-4 renamed `affordable_cache_shards` to `effective_cache_shards`:
    /// it is the same quantity `IndexPlanDataset` reports under that name, and
    /// both are `loader.effective_cache_shards()`.
    fn memory_budget<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let loader = self.loader()?;
        let budget = loader.cache_bytes_budget();
        let per_shard = loader.shard_decoded_bytes();
        dict.set_item("breakdown", loader.budget_breakdown().to_pydict(py)?)?;
        dict.set_item("max_memory_mb", budget / (1024 * 1024))?;
        dict.set_item("cache_shards", loader.cache_shards())?;
        dict.set_item("effective_cache_shards", loader.effective_cache_shards())?;
        dict.set_item("shard_decoded_bytes", per_shard)?;
        dict.set_item("max_plan_rows", loader.max_plan_rows())?;
        dict.set_item("mean_nnz_per_row", loader.mean_nnz_per_row())?;
        dict.set_item("max_blocking_threads", loader.max_blocking_threads())?;
        // Deliberately not folded into `breakdown`: the breakdown is the byte
        // model the shard cache is sized against, and an open reader's cost is
        // not in it. Charging readers there would shrink the cache by a term
        // the tuner has never accounted for; reporting the cap here says what
        // the knob is without pretending it is priced.
        dict.set_item("reader_limit", loader.reader_limit())?;
        dict.set_item("budget_exceeded", loader.budget_exceeded())?;
        Ok(dict)
    }

    /// Snapshot of the readers' shared shard-cache counters, cumulative since
    /// construction (the multi-file sibling of `IndexPlanDataset.cache_metrics`).
    /// Returns a dict with keys: `hits`, `misses` (= shard decodes), `evictions`,
    /// `bytes_inserted`, `duplicate_waiters`, `peak_bytes_in_cache`,
    /// `full_shard_groups`, `block_index_groups`, and the `row_group_*` set
    /// (`hits`, `misses`, `evictions`, `bytes_inserted`, `duplicate_waiters`)
    /// for the decoded row groups a framed `scatter_block_index=True` gather
    /// retains — `hits` … `duplicate_waiters` themselves keep meaning whole
    /// shards. All `int`; atomic, lock-free — sample as often as you like.
    ///
    /// Four further keys describe the **reader registry** rather than the shard
    /// cache: `reader_opens`, `reader_evictions`, `reader_resident` and
    /// `reader_hwm`. They appear only here and on `SparseCellSetBatchIter`,
    /// not on the single-file `IndexPlanDataset`, where they would be
    /// structurally always zero. At the default `reader_limit=None`,
    /// `reader_opens` equals the manifest size and the other three never move
    /// — which is how you check that a dataset is paying nothing for the
    /// bounded path. `reader_hwm` is the one to compare against `reader_limit`:
    /// the limit bounds handles the registry is free to drop, so a plan that
    /// leases more files at once than the limit exceeds it rather than
    /// blocking, and this is where that shows up.
    ///
    /// The last two are the **route** this dataset's gathers actually took:
    /// `block_index_groups > 0` proves the row-group path ran, and
    /// `full_shard_groups > 0` is the warm-into-the-LRU default. Opening an
    /// all-unframed set with `scatter_block_index=True` now warns at
    /// construction (ORG-9.10-4), so these are a confirmation rather than the
    /// only way to find out — but they stay the authority. The warning fires
    /// when *no* file is framed; its **absence** establishes only that at least
    /// one is, never that a gather actually took the route.
    ///
    /// Since ORG-9.10-1 the prefetch half is visible too, through
    /// `SparseCellSetBatchIter.metrics()["prefetch"]`, which since the reader
    /// registry also carries `prefetch_skipped_reader_limit` — plans the
    /// prefetcher declined because they touch more distinct files than
    /// `reader_limit` can hold resident. That is a different
    /// signal, not a second reading of this one: it records what the
    /// *prefetcher* decided, and is all-zero when prefetching is off
    /// (`lookahead=0`) even though the gather still adopts the route. These
    /// counters remain the authority on which route ran.
    fn cache_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let loader = self.loader()?;
        let dict = cache_metrics_to_pydict(py, &loader.cache_metrics())?;
        reader_metrics_into(&dict, &loader.reader_metrics())?;
        Ok(dict)
    }

    /// Never raises, closed or not.
    fn __repr__(&self) -> String {
        match self.loader.as_ref() {
            Some(l) => format!(
                "SparseCellSetDataset(n_files={}, n_cols={})",
                l.n_files(),
                l.n_cols()
            ),
            None => "SparseCellSetDataset(closed)".to_string(),
        }
    }
}

impl Drop for SparseCellSetDataset {
    /// Release the GIL around the runtime teardown on drop — see
    /// [`IndexPlanDataset::drop`].
    fn drop(&mut self) {
        let loader = self.loader.take();
        if unsafe { pyo3::ffi::Py_IsInitialized() } != 0 {
            Python::attach(|py| py.detach(|| Self::shutdown_detached(loader)));
        } else {
            Self::shutdown_detached(loader);
        }
    }
}

/// Python iterator wrapping the boxed engine iterator; converts each
/// `SparseCellSetBatch` into the §4.4 dict.
#[pyclass]
pub struct SparseCellSetBatchIter {
    inner: Option<crate::sparse_cellset::SparseCellSetIter>,
    /// Shared-cache counters, cloned at construction so sampling survives the
    /// inner iterator being dropped on exhaustion.
    cache_metrics: Arc<CacheMetrics>,
    /// Per-iter prefetch counters, cloned at construction for the same
    /// post-drain stability. Only reachable since ORG-9.10-1 folded this arm
    /// onto the same iterator the pair loader uses — before that the engine had
    /// no counters at all, so the sparse path's block-index adoption could only
    /// be inferred from the cache-side `block_index_groups`.
    iter_metrics: Arc<IterMetrics>,
    /// Reader-registry counters, cloned for the same post-drain stability.
    reader_metrics: Arc<crate::reader_registry::ReaderMetrics>,
    /// Samples `cache_metrics` as batches are yielded and warns once on thrash.
    thrash: ThrashSampler,
}

#[pymethods]
impl SparseCellSetBatchIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(inner) = self.inner.as_mut() else {
            return Ok(None);
        };
        // Release the GIL for the prefetch await + gather so the plan-pull
        // worker can call __next__ on the user iterator without contention.
        let next = py.detach(|| inner.next());
        match next {
            Some(Ok(batch)) => {
                let dict = sparse_cellset_batch_to_dict(py, batch)?;
                self.thrash.observe(py, &self.cache_metrics);
                Ok(Some(dict))
            }
            Some(Err(e)) => Err(loader_err_to_py(e)),
            None => {
                // Off the GIL — see `IndexPlanBatchIter::__next__`. After
                // `ds.close()` this can be what releases the last engine
                // reference, and `Drop` cannot cover it once `inner` is taken.
                let inner = self.inner.take();
                py.detach(move || drop(inner));
                Ok(None)
            }
        }
    }

    /// Snapshot of the shared shard-cache counters (same keys as
    /// `SparseCellSetDataset.cache_metrics`). The flat, cache-only half of
    /// [`Self::metrics`], kept because it predates it; safe after exhaustion.
    fn cache_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = cache_metrics_to_pydict(py, &self.cache_metrics)?;
        crate::python::convert::reader_metrics_into(&dict, &self.reader_metrics)?;
        Ok(dict)
    }

    /// Snapshot of cache- and prefetch-side counters as a dict-of-dicts,
    /// identical in shape to `IndexPlanBatchIter.metrics()`:
    ///
    /// ```text
    /// {"cache": {hits, misses, evictions, bytes_inserted, duplicate_waiters,
    ///            peak_bytes_in_cache, full_shard_groups, block_index_groups,
    ///            row_group_hits, row_group_misses, row_group_evictions,
    ///            row_group_bytes_inserted, row_group_duplicate_waiters,
    ///            admitted_group_bytes, rejected_group_bytes, reuse_admissions,
    ///            parallel_group_decodes},
    ///  "prefetch": {prefetch_tasks_spawned,
    ///               prefetch_skipped_cache_hit,
    ///               prefetch_skipped_in_flight,
    ///               prefetch_skipped_block_index,
    ///               prefetch_skipped_reader_limit}}
    /// ```
    ///
    /// `cache` is loader-cumulative (shared with
    /// `SparseCellSetDataset.cache_metrics`); `prefetch` is per-iter and resets
    /// on every `iter_with_plans` call.
    ///
    /// `prefetch_skipped_block_index` counts the **L2 prefetch-time** decision
    /// — shards left undecoded so the gather could take the block-index path.
    /// Against an unframed file it stays 0, which is what makes the silent
    /// no-op visible. It is **not** interchangeable with `cache_metrics`'
    /// `block_index_groups`, which is the route the *gather* took: at
    /// `lookahead=0` no prefetch runs, so every counter here is 0 while
    /// `block_index_groups` is positive. Both handles are cloned at
    /// construction, so this is safe after the iterator has been drained.
    fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let cache = cache_metrics_to_pydict(py, &self.cache_metrics)?;
        crate::python::convert::reader_metrics_into(&cache, &self.reader_metrics)?;
        dict.set_item("cache", cache)?;
        dict.set_item("prefetch", iter_metrics_to_pydict(py, &self.iter_metrics)?)?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("SparseCellSetBatchIter(exhausted={})", self.inner.is_none())
    }
}

impl Drop for SparseCellSetBatchIter {
    /// Release the GIL around dropping the inner iterator — see
    /// [`IndexPlanBatchIter::drop`] for why this iterator, not just the
    /// dataset, has to do it.
    fn drop(&mut self) {
        let inner = self.inner.take();
        if unsafe { pyo3::ffi::Py_IsInitialized() } != 0 {
            Python::attach(|py| py.detach(move || drop(inner)));
        } else {
            drop(inner);
        }
    }
}

/// Convert a `SparseCellSetBatch` into the §4.4 dict: `{indptr, indices, data,
/// shape, cell_indices, file_ids, set_offsets, role_tags}`.
fn sparse_cellset_batch_to_dict<'py>(
    py: Python<'py>,
    batch: SparseCellSetBatch,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("indptr", PyArray1::from_vec(py, batch.indptr))?;
    dict.set_item("indices", PyArray1::from_vec(py, batch.indices))?;
    dict.set_item("data", PyArray1::from_vec(py, batch.data))?;
    dict.set_item("shape", PyTuple::new(py, [batch.shape.0, batch.shape.1])?)?;
    dict.set_item("cell_indices", PyArray1::from_vec(py, batch.cell_indices))?;
    dict.set_item("file_ids", PyArray1::from_vec(py, batch.file_ids))?;
    dict.set_item("set_offsets", PyArray1::from_vec(py, batch.set_offsets))?;
    dict.set_item("role_tags", PyArray1::from_vec(py, batch.role_tags))?;
    Ok(dict)
}
