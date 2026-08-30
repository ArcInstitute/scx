//! `IndexPlanDataset` / `IndexPlanBatchIter` — the paired plan-driven loader.
//!
//! Split out of `python.rs` by ORG-9.10-2; a pure move.

use std::sync::Arc;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use scx_format_io::CacheMetrics;

use crate::error::LoaderError;
use crate::index_plan::{IndexPlanBatch, IndexPlanIter, IndexPlanLoader};
use crate::pipeline::LoaderConfig;
use crate::plan_engine::IterMetrics;

use super::*;

/// Plan-driven paired-batch reader for `(perturbed, control)` ML training.
///
/// Sibling to `TrainingDataset`: instead of streaming shards in catalog order,
/// `IndexPlanDataset` consumes caller-supplied `(pert_idx, ctrl_idx)` plans
/// and returns paired dense batches.
///
/// Surface: `next_batch(plan)` is the synchronous entry point; the streaming
/// `iter_with_plans` API exposes a tokio-driven prefetched iterator.
///
/// # Fork safety
///
/// `IndexPlanDataset` is fork-safe under PyTorch
/// `DataLoader(num_workers > 0, start_method="fork")` **when the dataset is
/// constructed lazily inside the worker's `__iter__`** — same contract as
/// `TrainingDataset`.
///
/// The multi-threaded tokio runtime is **not** built eagerly in
/// `IndexPlanLoader::new()`. It belongs to the `PrefetchEngine` the loader
/// builds on its first `iter_with_plans` call, and the engine builds it lazily
/// in turn — two nested `OnceLock`s, neither touched by `new`, so the first
/// touch is always post-fork in the `DataLoader` worker. So the loader never
/// owns runtime threads at the moment a child is forked, and each worker builds
/// its own runtime fresh; the parent's runtime threads are never inherited. The
/// construct-then-fork case is additionally caught by the PID check in
/// `iter_with_plans` / `next_batch_for_test`.
///
/// **Rayon.** This class *does* reach rayon, just not directly — through
/// `scx-format-io`. `IndexPlanLoader::new` calls `ScxReader::read_obs`, which
/// fans the shard decode out with `par_iter` on any file with sharded obs
/// metadata; and the gather reaches `BackedCsrReader::warm_shards`. Both used to
/// dispatch against rayon's *global* registry, which `fork()` copies as a data
/// structure without its worker threads, so a forked worker parked forever with
/// no error and no batch. (An earlier version of this comment asserted the
/// opposite — that the rayon-after-fork hazard did not apply here.) Both now go
/// through `crate::pool::cpu_pool()`, which is rebuilt whenever the PID changes.
///
/// **Acceptance tests**: in `pyscx/tests/test_fork_safety.py` —
/// `test_fork_index_plan_dataset` (fork + lazy construction + plan iteration),
/// `test_fork_index_plan_sharded_obs` (the `read_obs` path) and
/// `test_fork_index_plan_multi_shard_gather` (the `warm_shards` path).
///
/// Recommended: call `start_method="spawn"` for the same reason it is
/// recommended on `TrainingDataset` — no fork hazards at all.
#[pyclass]
pub struct IndexPlanDataset {
    /// `None` once [`IndexPlanDataset::close`] has run. Every accessor goes
    /// through [`IndexPlanDataset::loader`], which raises on a closed dataset;
    /// `close` needs the `Option` because bounding the runtime teardown
    /// requires *owning* the loader, not merely borrowing it.
    loader: Option<Arc<IndexPlanLoader>>,
    /// PID at construction time — used to detect forking. The shard cache and
    /// mmap state are not fork-safe; consumers must lazily construct the
    /// dataset post-fork in each DataLoader worker.
    creation_pid: u32,
}

impl IndexPlanDataset {
    /// The live loader, or a `RuntimeError` naming the cause when the dataset
    /// has been closed. Deliberately **not** in the `#[pymethods]` block — a
    /// method there would be exported to Python as `ds.loader`.
    fn loader(&self) -> PyResult<&Arc<IndexPlanLoader>> {
        self.loader.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "IndexPlanDataset is closed. close() is terminal on this class: the \
                 tokio runtime is built exactly once so it can never be inherited \
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
    fn shutdown_detached(loader: Option<Arc<IndexPlanLoader>>) {
        drop(loader);
    }
}

#[pymethods]
impl IndexPlanDataset {
    /// Construct a plan-driven paired-batch reader.
    ///
    /// Args:
    ///     path: Path to the .scx file.
    ///     hvg_indices: Gene indices for HVG projection. None = all genes.
    ///         Every index must be < n_vars; an out-of-range index is
    ///         rejected at construction rather than becoming an
    ///         always-zero output column. This class has no modality
    ///         surface — the bound is always the file-wide n_vars.
    ///         Sorted and deduplicated, so batch columns are in ascending
    ///         gene-index order regardless of the order passed and duplicates
    ///         shrink the batch (see `n_output_genes`); a panel that is not
    ///         already ascending-unique gets a UserWarning.
    ///     obs_columns: Obs metadata column names to include in each batch.
    ///     normalize: Apply total-count normalization (default: True).
    ///     log1p: Apply ln(x+1) to each row (default: True). Independent of
    ///         `normalize`; all four (normalize, log1p) combinations are
    ///         honoured, matching `TrainingDataset` semantics.
    ///     target_sum: Normalization target sum (default: 1e4).
    ///     pflog: Apply PFlog (v4) / shifted-log normalization on raw counts
    ///         (Booeshaghi et al.) instead of normalize/log1p (default: False).
    ///         Mutually exclusive with `normalize`/`log1p` (takes precedence when
    ///         True). The centering denominator is over the full transcriptome
    ///         even under `hvg_indices` projection. See `TrainingDataset`.
    ///     pflog_alpha: PFlog NB overdispersion `α` (pseudocount 1/(4α); only
    ///         used when `pflog=True`). None (default) estimates α once at
    ///         construction (single-modality only); a float pins a reference α.
    ///     cache_shards: LRU shard cache count cap (default: 128). Must be
    ///         >= 1. Auto-tuned downward to fit `max_memory_mb`; check the
    ///         resolved value via `effective_cache_shards()`. The cache also
    ///         enforces a byte cap derived from the memory budget — see
    ///         `cache_metrics()` for runtime hit/miss/eviction observability.
    ///     sort_by_shard: Reorder each plan by shard-of-min-row before
    ///         gathering, so the returned rows of X/X_paired are in shard
    ///         locality order (default: True). Disable to preserve the
    ///         caller's input pair order.
    ///     lookahead: Default lookahead used by `iter_with_plans` when the
    ///         caller does not pass an explicit value (default: 4). 0 disables
    ///         shard prefetching. Auto-tuned downward to fit `max_memory_mb`;
    ///         check the resolved value via `effective_lookahead()`.
    ///     max_plan_size: Upper bound on rows-per-batch for the memory budget
    ///         calculation (default: 16384). Sets the ceiling on the dense
    ///         X / X_paired buffers so the loader can refuse pathological
    ///         plans early.
    ///     max_memory_mb: Memory budget in MB (default: 512). On overflow,
    ///         lookahead is reduced first (down to 1), then cache_shards
    ///         (down to 1); construction fails if neither fits.
    ///     scatter_block_index: Per-dataset escape hatch (default: True) gating the
    ///         block-index-aware prefetch skip for row-group-framed (v2) files.
    ///         True lets cold sparse plans decode only the touched row-groups via
    ///         the block index (independent of the Scx1 sidecar); False makes the
    ///         prefetch warm whole framed shards. `SCX_SCATTER_BLOCK_INDEX=0` is
    ///         the process-wide reader-layer kill-switch.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        path,
        hvg_indices=None,
        obs_columns=None,
        normalize=None,
        log1p=None,
        target_sum=None,
        pflog=None,
        pflog_alpha=None,
        cache_shards=None,
        sort_by_shard=None,
        lookahead=None,
        max_plan_size=None,
        max_memory_mb=None,
        scatter_block_index=None,
    ))]
    fn new(
        py: Python<'_>,
        path: &str,
        hvg_indices: Option<Vec<u32>>,
        obs_columns: Option<Vec<String>>,
        normalize: Option<bool>,
        log1p: Option<bool>,
        target_sum: Option<f64>,
        pflog: Option<bool>,
        pflog_alpha: Option<f64>,
        cache_shards: Option<usize>,
        sort_by_shard: Option<bool>,
        lookahead: Option<usize>,
        max_plan_size: Option<usize>,
        max_memory_mb: Option<usize>,
        scatter_block_index: Option<bool>,
    ) -> PyResult<Self> {
        let mut config = LoaderConfig::default();
        if let Some(v) = hvg_indices {
            config.hvg_indices = Some(v);
        }
        if let Some(v) = obs_columns {
            config.obs_columns = v;
        }
        if let Some(v) = normalize {
            config.normalize = v;
        }
        if let Some(v) = log1p {
            config.log1p = v;
        }
        if let Some(v) = target_sum {
            config.target_sum = v;
        }
        if let Some(v) = pflog {
            config.pflog = v;
        }
        if let Some(v) = pflog_alpha {
            config.pflog_alpha = Some(v);
        }
        if let Some(v) = max_memory_mb {
            config.max_memory_mb = v;
        }
        // No explicit budget ⇒ size it from this file's own requested config
        // (bounded, adaptive) instead of the fixed 512 MB default, which against
        // large Pcodec shards drove the auto-tune to `cache_shards = 1` and
        // manufactured cache thrash. `config.max_memory_mb` stays the floor.
        config.auto_memory_budget = max_memory_mb.is_none();

        let cache_shards = cache_shards.unwrap_or(128);
        let sort_by_shard = sort_by_shard.unwrap_or(true);
        let lookahead = lookahead.unwrap_or(4);
        let max_plan_size = max_plan_size.unwrap_or(16384);
        let scatter_block_index = scatter_block_index.unwrap_or(true);

        let mut loader = IndexPlanLoader::new(
            path,
            config,
            cache_shards,
            sort_by_shard,
            lookahead,
            max_plan_size,
        )
        .map_err(loader_err_to_py)?;
        loader.set_scatter_block_index(scatter_block_index);

        if let Some(v) = loader.cache_sizing() {
            warn_cache_sizing(py, "IndexPlanDataset", &v)?;
        }
        if let Some(v) = loader.hvg_panel() {
            warn_hvg_panel(py, "IndexPlanDataset", &v)?;
        }

        if should_warn_unframed_scatter(
            scatter_block_index,
            scx_format_io::backed::scatter_block_index_enabled(),
            || loader.any_shard_framed(),
        ) {
            warn_unframed_scatter(py, "IndexPlanDataset", &format!("'{path}'"))?;
        }

        Ok(Self {
            loader: Some(Arc::new(loader)),
            creation_pid: std::process::id(),
        })
    }

    /// Total number of observations (cells) in the dataset.
    #[getter]
    fn n_obs(&self) -> PyResult<u64> {
        Ok(self.loader()?.n_obs())
    }

    /// Total number of variables (genes) in the dataset.
    #[getter]
    fn n_vars(&self) -> PyResult<u64> {
        Ok(self.loader()?.n_vars())
    }

    /// Number of output genes per batch (HVG count if projection active).
    #[getter]
    fn n_output_genes(&self) -> PyResult<usize> {
        Ok(self.loader()?.n_output_cols())
    }

    /// True once [`Self::close`] has run. Never raises.
    #[getter]
    fn closed(&self) -> bool {
        self.loader.is_none()
    }

    /// Release the loader's tokio runtime, bounded by a 5 s deadline, with the
    /// GIL detached.
    ///
    /// A `#[pyclass]` is dropped with the GIL held, and dropping the runtime
    /// blocks until every already-started `spawn_blocking` returns — those are
    /// `read_shard_cached_arc` decodes that can be hundreds of megabytes of
    /// Pcodec. Without this, `del ds` freezes every other Python thread
    /// (PyTorch CUDA stream callbacks, the logging thread) for that whole
    /// window. `Drop` does the same thing, so `close()` is an explicitness
    /// convenience, not a correctness requirement.
    ///
    /// Idempotent. **Terminal**, unlike `TrainingDataset.close()`: that class
    /// rebuilds its pool and runtime on the next `__iter__`, whereas this one
    /// cannot, because the runtime is built exactly once so that a forked
    /// child can never inherit live tokio threads. Any later call raises
    /// `RuntimeError`; construct a new dataset instead.
    ///
    /// The 5 s bound applies only when this call holds the last reference to
    /// the loader. A still-alive `IndexPlanBatchIter` holds one too; in that
    /// case this releases ours off-GIL and the iterator's own drop — also
    /// off-GIL — finishes the teardown.
    fn close(&mut self, py: Python<'_>) {
        let loader = self.loader.take();
        py.detach(|| Self::shutdown_detached(loader));
    }

    /// Drive the loader from a Python iterable of `(pert_idx, ctrl_idx)`
    /// pair lists; return an iterator of paired-batch dicts.
    ///
    /// Args:
    ///     plans: any iterable yielding `list[tuple[int, int]]` (or any
    ///         sequence of (int, int) pairs).
    ///     lookahead: override the loader's default lookahead. None (the
    ///         default) uses `effective_lookahead()` — the constructor's
    ///         auto-tuned value. 0 disables prefetch entirely (decoded
    ///         synchronously per plan). Larger values trade RAM (~1
    ///         prefetch slot per lookahead) for I/O hiding.
    ///
    /// Returns: an `IndexPlanBatchIter` (Python iterator) whose `__next__`
    /// yields `{"X", "X_paired", "pairs", "obs", "obs_paired"}` dicts.
    /// Plan iteration is lazy: the loader pulls the next plan only when it
    /// is ready to schedule a prefetch for it, and **prefetch depth is
    /// opportunistic** — the loader waits only for the first plan of each
    /// batch, then tops the queue up with whatever the generator has already
    /// produced. A generator that yields plan *i+1* only after inspecting
    /// batch *i* (curriculum / feedback sampling) therefore runs correctly,
    /// merely un-prefetched; it is never required to run `lookahead` plans
    /// ahead of the consumer.
    #[pyo3(signature = (plans, lookahead=None))]
    fn iter_with_plans(
        &self,
        plans: Py<PyAny>,
        lookahead: Option<usize>,
    ) -> PyResult<IndexPlanBatchIter> {
        if std::process::id() != self.creation_pid {
            return Err(PyRuntimeError::new_err(
                "scx.IndexPlanDataset requires num_workers=0 (or lazy per-worker \
                 construction). The Rust shard cache and mmap state are not fork-safe.",
            ));
        }
        let loader = self.loader()?;
        let lookahead = lookahead.unwrap_or_else(|| loader.effective_lookahead());

        // Bind plans → its iter, hold an owned Py<PyAny> Send-safe handle.
        let py_iter: Py<PyAny> = Python::attach(|py| -> PyResult<Py<PyAny>> {
            Ok(plans.bind(py).call_method0("__iter__")?.unbind())
        })?;

        let plan_stream = PyPlanIterator {
            py_iter,
            extract: extract_pair_plan,
        };
        let inner = Arc::clone(loader).iter_with_plans(plan_stream, lookahead);
        let iter_metrics = inner.iter_metrics();
        let cache_metrics = loader.cache_metrics();
        let thrash = ThrashSampler::new(
            "IndexPlanDataset",
            loader.effective_cache_shards(),
            loader.n_shards(),
            // The paired path's auto-tune reduces the count to fit the budget, so
            // a reduction there IS byte-driven.
            loader.effective_cache_shards() < loader.requested_cache_shards(),
            loader.shard_decoded_bytes(),
        );
        Ok(IndexPlanBatchIter {
            inner: Some(inner),
            lookahead,
            iter_metrics,
            cache_metrics,
            thrash,
        })
    }

    /// Effective LRU shard cache size after auto-tuning to fit
    /// `max_memory_mb`. May be less than the user-requested `cache_shards`.
    fn effective_cache_shards(&self) -> PyResult<usize> {
        Ok(self.loader()?.effective_cache_shards())
    }

    /// Number of distinct CSR shards `plan` touches — the `cache_shards` that
    /// would let the whole plan stay resident for one `process_plan` call.
    ///
    /// Pure index arithmetic from the catalog's shard row ranges: no I/O, no
    /// decode, safe to call before iterating. Size the cache from the plans you
    /// will actually issue rather than guessing:
    ///
    /// ```python
    /// probe = pyscx.IndexPlanDataset(path)
    /// need = max(probe.suggested_cache_shards(p) for p in plans[:64])
    /// ds = pyscx.IndexPlanDataset(path, cache_shards=need)
    /// ```
    fn suggested_cache_shards(&self, py: Python<'_>, plan: Vec<(u64, u64)>) -> PyResult<usize> {
        let loader = self.loader()?;
        Ok(py.detach(|| loader.plan_shard_touch_count(&plan)))
    }

    /// Effective default lookahead after auto-tuning to fit `max_memory_mb`.
    /// May be less than the user-requested `lookahead`. Used by
    /// `iter_with_plans` when the caller does not pass an explicit override.
    fn effective_lookahead(&self) -> PyResult<usize> {
        Ok(self.loader()?.effective_lookahead())
    }

    /// Snapshot of the underlying `BackedCsrReader`'s shard-cache counters,
    /// cumulative since dataset construction. Returns a dict with keys:
    ///
    /// ```text
    /// hits                - cache lookups served without decode
    /// misses              - decodes that ran (each shard's first-leader)
    /// evictions           - LRU entries dropped to honour the byte/count cap
    /// bytes_inserted      - cumulative decoded bytes inserted into the cache
    /// duplicate_waiters   - waiters that found a peer leader in flight and
    ///                       skipped redundant decode work
    /// peak_bytes_in_cache - high-water mark of the cache's resident byte
    ///                       budget; useful for sizing `max_memory_mb`
    /// ```
    ///
    /// All values are `int`. Counters are atomic and read with `Relaxed`
    /// ordering. Sample as often as you want — there are no locks involved.
    fn cache_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        cache_metrics_to_pydict(py, &self.loader()?.cache_metrics())
    }

    /// Memory budget diagnostics as a dict:
    ///
    /// ```text
    /// breakdown                - per-component estimate (see below)
    /// max_memory_mb            - user-supplied budget (LoaderConfig)
    /// effective_cache_shards   - post-auto-tune cache count cap
    /// effective_lookahead      - post-auto-tune iter lookahead
    ///
    /// breakdown:
    ///   cache_bytes              - decoded shard cache budget
    ///   batch_buffer_bytes       - dense X / X_paired buffers
    ///   lookahead_overhead_bytes - plan-tuple staging in the iter
    ///   transient_bytes          - per-batch obs Vecs + PairRequest scratch
    ///   python_overhead_bytes    - constant Python/Arrow/numpy overhead
    ///   total_bytes              - sum of the above
    /// ```
    ///
    /// All byte values are `int`. ORG-9.10-4 moved the six components **under**
    /// `breakdown`, where `TrainingDataset` had always reported them, so that
    /// `memory_budget()["breakdown"]["total_bytes"]` reads the same on every
    /// class that reports a budget; the keys beside it stay class-specific.
    /// `MultimodalTrainingDataset` nests one such envelope per modality
    /// (ORG-9.10-5).
    fn memory_budget<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let loader = self.loader()?;
        let dict = PyDict::new(py);
        dict.set_item("breakdown", loader.budget_breakdown().to_pydict(py)?)?;
        dict.set_item("max_memory_mb", loader.max_memory_mb())?;
        dict.set_item("effective_cache_shards", loader.effective_cache_shards())?;
        dict.set_item("effective_lookahead", loader.effective_lookahead())?;
        Ok(dict)
    }

    /// Process one plan and return a paired dense batch dict.
    ///
    /// **Unstable / debug helper.** Superseded by `iter_with_plans`. Retained
    /// so unit tests can drive the loader synchronously without iterator
    /// setup; do not depend on this method from production code — it may be
    /// removed in a future release.
    #[pyo3(name = "_next_batch_for_test")]
    fn next_batch_for_test<'py>(
        &self,
        py: Python<'py>,
        plan: Vec<(u64, u64)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if std::process::id() != self.creation_pid {
            return Err(PyRuntimeError::new_err(
                "scx.IndexPlanDataset requires num_workers=0 (or lazy per-worker \
                 construction). The Rust shard cache and mmap state are not fork-safe.",
            ));
        }

        let loader = Arc::clone(self.loader()?);
        let batch = py
            .detach(move || loader.process_plan(plan))
            .map_err(loader_err_to_py)?;

        index_plan_batch_to_dict(py, batch)
    }

    /// Never raises, closed or not — `repr` is what a debugger and a traceback
    /// call, and neither should fail because the object was closed.
    fn __repr__(&self) -> String {
        match self.loader.as_ref() {
            Some(l) => format!(
                "IndexPlanDataset(n_obs={}, n_vars={}, n_output_genes={})",
                l.n_obs(),
                l.n_vars(),
                l.n_output_cols(),
            ),
            None => "IndexPlanDataset(closed)".to_string(),
        }
    }
}

impl Drop for IndexPlanDataset {
    /// Release the GIL around the runtime teardown on drop.
    ///
    /// Mirrors `TrainingDataset::drop`, including the `Py_IsInitialized` probe
    /// for the case where we are dropped *after* the interpreter has finalized
    /// and there is no GIL to detach from. `py.detach` only wraps pure-Rust
    /// work here — it never calls into Python — so it is safe even
    /// mid-finalization, where the finalizing thread holds the GIL while
    /// destructors run.
    fn drop(&mut self) {
        let loader = self.loader.take();
        if unsafe { pyo3::ffi::Py_IsInitialized() } != 0 {
            Python::attach(|py| py.detach(|| Self::shutdown_detached(loader)));
        } else {
            Self::shutdown_detached(loader);
        }
    }
}

/// `[(pert_row, ctrl_row), ...]` — the pair loader's plan.
fn extract_pair_plan(obj: &Bound<'_, PyAny>) -> std::result::Result<Vec<(u64, u64)>, LoaderError> {
    obj.extract::<Vec<(u64, u64)>>()
        .map_err(|e| LoaderError::ConfigError {
            reason: format!("plan extraction failed: {e}"),
        })
}

/// Python iterator that wraps an [`IndexPlanIter`] and converts each yielded
/// `IndexPlanBatch` into the standard pyscx batch dict.
#[pyclass]
pub struct IndexPlanBatchIter {
    inner: Option<IndexPlanIter>,
    #[allow(dead_code)] // surfaced via __repr__ / future docs
    lookahead: usize,
    /// Snapshot of the iter's prefetch counters, cloned at construction so
    /// the Python `metrics()` accessor stays valid after the iterator is
    /// drained and `inner` is dropped.
    iter_metrics: Arc<IterMetrics>,
    /// Loader-level cache counters, cloned at construction for the same
    /// post-drain stability.
    cache_metrics: Arc<CacheMetrics>,
    /// Samples `cache_metrics` as batches are yielded and warns once on thrash.
    thrash: ThrashSampler,
}

#[pymethods]
impl IndexPlanBatchIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(inner) = self.inner.as_mut() else {
            return Ok(None);
        };
        // Release the GIL for both the prefetch await and the decode work,
        // so the plan-pull worker can call __next__ on the user iterator
        // without contention.
        let next = py.detach(|| inner.next());
        match next {
            Some(Ok(batch)) => {
                let dict = index_plan_batch_to_dict(py, batch)?;
                self.thrash.observe(py, &self.cache_metrics);
                Ok(Some(dict))
            }
            Some(Err(e)) => Err(loader_err_to_py(e)),
            None => {
                // Drop the inner iterator to release the plan-pull thread
                // and tokio runtime references; subsequent next calls return
                // None without re-entering Rust.
                //
                // Off the GIL, and not merely for symmetry with `Drop`: after
                // `ds.close()` the dataset has already released its `Arc`, so
                // exhausting the iterator here can be what releases the *last*
                // one and tears the runtime down. `Drop` cannot cover it —
                // once `inner` is `None` it early-returns.
                let inner = self.inner.take();
                py.detach(move || drop(inner));
                Ok(None)
            }
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "IndexPlanBatchIter(lookahead={}, exhausted={})",
            self.lookahead,
            self.inner.is_none()
        )
    }

    /// Snapshot of cache- and prefetch-side counters as a dict-of-dicts:
    ///
    /// ```text
    /// {"cache": {hits, misses, evictions, bytes_inserted, duplicate_waiters,
    ///            peak_bytes_in_cache},
    ///  "prefetch": {prefetch_tasks_spawned,
    ///               prefetch_skipped_cache_hit,
    ///               prefetch_skipped_in_flight,
    ///               prefetch_skipped_block_index}}
    /// ```
    ///
    /// `cache` reflects loader-cumulative counters (shared with
    /// `IndexPlanDataset.cache_metrics()`). `prefetch` is per-iter — counters
    /// reset across `iter_with_plans` calls. Safe to call after the iterator
    /// is exhausted; the metrics handles are cloned at construction time.
    fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("cache", cache_metrics_to_pydict(py, &self.cache_metrics)?)?;
        dict.set_item("prefetch", iter_metrics_to_pydict(py, &self.iter_metrics)?)?;
        Ok(dict)
    }
}

impl Drop for IndexPlanBatchIter {
    /// Release the GIL around dropping the inner iterator.
    ///
    /// This iterator holds its own `Arc<IndexPlanLoader>`, so in the ordinary
    /// `for b in ds.iter_with_plans(...)` shape it outlives `ds.close()` in the
    /// caller's frame and becomes the *last* reference — which would put the
    /// whole runtime teardown back under the GIL that `close` just took care to
    /// detach. Dropping the inner iterator also aborts in-flight prefetches
    /// (`PlanPrefetchIter::drop`, which `IndexPlanIter` boxes), but an abort
    /// cannot stop a `spawn_blocking` task that has already started.
    fn drop(&mut self) {
        let inner = self.inner.take();
        if unsafe { pyo3::ffi::Py_IsInitialized() } != 0 {
            Python::attach(|py| py.detach(move || drop(inner)));
        } else {
            drop(inner);
        }
    }
}

/// Convert an `IndexPlanBatch` into the Python dict shape:
/// `{"X", "X_paired", "pairs", "obs", "obs_paired"}`.
fn index_plan_batch_to_dict<'py>(
    py: Python<'py>,
    batch: IndexPlanBatch,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);

    let n_pairs = batch.n_pairs();
    let n_cols = batch.x.len().checked_div(n_pairs).unwrap_or(0);

    let x_array = PyArray1::from_vec(py, batch.x)
        .reshape([n_pairs, n_cols])
        .map_err(|e| PyRuntimeError::new_err(format!("failed to reshape X array: {e}")))?;
    dict.set_item("X", x_array)?;

    let xp_array = PyArray1::from_vec(py, batch.x_paired)
        .reshape([n_pairs, n_cols])
        .map_err(|e| PyRuntimeError::new_err(format!("failed to reshape X_paired array: {e}")))?;
    dict.set_item("X_paired", xp_array)?;

    let pairs_list = PyList::empty(py);
    for (p, c) in &batch.pairs {
        let tup = PyTuple::new(py, [*p, *c])
            .map_err(|e| PyRuntimeError::new_err(format!("failed to build pair tuple: {e}")))?;
        pairs_list.append(tup)?;
    }
    dict.set_item("pairs", pairs_list)?;

    dict.set_item("obs", obs_to_pydict(py, batch.obs)?)?;
    dict.set_item("obs_paired", obs_to_pydict(py, batch.obs_paired)?)?;

    Ok(dict)
}
