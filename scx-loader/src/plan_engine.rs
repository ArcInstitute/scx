//! Multi-reader plan-driven prefetch engine.
//!
//! Generalizes the prefetch scaffolding of [`crate::index_plan::IndexPlanLoader`]
//! (plan-pull thread, bounded lookahead queue, lazy tokio runtime, per-plan
//! shard prefetch via `spawn_blocking`) from one `BackedCsrReader` to a
//! `file_id → reader` map whose readers share one [`SharedShardCache`] budget.
//! It is the reusable substrate for the Phase 2 native sparse cell-set loader
//! (SCX-DATA-LOADER §4.3); the pair loader stays on its own copy until the
//! optional Phase 6.1 rewire.
//!
//! The engine is deliberately **concrete**, parameterized by two closures
//! rather than a `PlanGather` trait (CLAUDE.md §2 — the sparse loader is the
//! only consumer):
//!
//! * `rows_of(&plan) -> Vec<(file_id, row)>` — which rows the plan touches, so
//!   the engine can warm their shards across the right readers.
//! * `process(&engine, &plan) -> Result<T>` — the actual gather, run on the
//!   consumer thread once the head plan's shards are warm.
//!
//! Plan order is preserved (no `sort_by_shard` reorder): the sparse gather
//! recovers intra-call shard locality inside `BackedCsrReader::read_rows_with`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};
use std::thread;

use crossbeam_channel::{bounded, Receiver, TryRecvError};
use scx_format_io::{BackedCsrReader, CacheMetrics, ScxReader, SharedShardCache};
use scx_sparse::ScxCsr;
use tokio::runtime::Runtime;
use tokio::task::JoinHandle;

use crate::error::{LoaderError, Result};

type ShardJoin = JoinHandle<scx_format_io::Result<Arc<ScxCsr>>>;

/// A `file_id → reader` map sharing one decoded-shard budget, plus a lazily
/// built tokio runtime for prefetch. Wrap in `Arc` and call
/// [`PrefetchEngine::iter_with_plans`].
pub struct PrefetchEngine {
    /// Readers indexed by `file_id` (the index into this vec). All share one
    /// [`SharedShardCache`] via [`BackedCsrReader::with_shared_cache`].
    readers: Vec<Arc<BackedCsrReader>>,
    /// Lazily built so it is never inherited across a fork — mirrors
    /// `IndexPlanLoader`. Built on the first `iter_with_plans` consumption.
    runtime: OnceLock<crate::runtime::BoundedRuntime>,
    /// Default lookahead depth (overridable per `iter_with_plans` call).
    default_lookahead: usize,
    /// Shared handle to the readers' one `SharedShardCache` counters
    /// (hits / misses / evictions / …). Mirrors `IndexPlanLoader::cache_metrics`;
    /// always populated (a zeroed default when no reader enabled metrics).
    cache_metrics: Arc<CacheMetrics>,
}

impl PrefetchEngine {
    /// Build an engine over `readers` (already sharing one cache). `file_id` is
    /// the reader's index in the slice.
    pub fn new(readers: Vec<Arc<BackedCsrReader>>, default_lookahead: usize) -> Arc<Self> {
        // Read back the shared cache's metrics handle (enabled upstream, e.g. in
        // `from_scx_readers`). All readers share one cache, so any reader's handle
        // is the single aggregate; a zeroed default when metrics were never enabled.
        let cache_metrics = readers
            .iter()
            .find_map(|r| r.metrics().cloned())
            .unwrap_or_else(|| Arc::new(CacheMetrics::default()));
        Arc::new(PrefetchEngine {
            readers,
            runtime: OnceLock::new(),
            default_lookahead,
            cache_metrics,
        })
    }

    /// Convenience: build the shared cache and wrap each `ScxReader` as a
    /// CSR reader sharing it, in slice order (`file_id = index`).
    ///
    /// `scatter_block_index` sets every reader's per-reader block-index gate
    /// ([`BackedCsrReader::set_scatter_block_index`]) before its `Arc` is
    /// shared, which is the only window in which it can be set. Because
    /// [`BackedCsrReader::block_index_eligible`] ANDs that flag, setting it here
    /// gates **both** consumers at once: the L1 gather in `read_rows_with` and
    /// the L2 prefetch warm-skip in [`PlanPrefetchIter`]. `false` means "always
    /// warm the whole shard into the LRU and serve from cache" — the right
    /// choice for a cache-friendly working set, where the block-index path's
    /// `!cache.contains()` predicate would keep a hot shard eligible forever and
    /// re-decode it every batch. The process-wide `SCX_SCATTER_BLOCK_INDEX=0`
    /// kill-switch remains a hard master override over this argument.
    pub fn from_scx_readers(
        scx_readers: Vec<ScxReader>,
        cache_shards: usize,
        bytes_budget: usize,
        default_lookahead: usize,
        scatter_block_index: bool,
    ) -> Arc<Self> {
        let shared = SharedShardCache::new(cache_shards, bytes_budget);
        let readers = scx_readers
            .into_iter()
            .enumerate()
            .map(|(fid, r)| {
                let mut backed =
                    BackedCsrReader::with_shared_cache(r, fid as u32, Arc::clone(&shared));
                // Set before the `Arc`: `BackedCsrReader` exposes no interior
                // mutability for this, and `PrefetchEngine` hands out `&` only.
                backed.set_scatter_block_index(scatter_block_index);
                // One pool for every reader, not one each: `cpu_pool()` is
                // process-wide, so an N-file engine does not spawn N pools.
                // Same fork rationale as `IndexPlanLoader` — see `crate::pool`.
                backed.set_cpu_pool(crate::pool::cpu_pool());
                // Always-on metrics, mirroring `IndexPlanLoader`. `enable_metrics`
                // is idempotent on the shared cache, so doing it per reader installs
                // one aggregate handle that `new` reads back via `metrics()`.
                backed.enable_metrics();
                Arc::new(backed)
            })
            .collect();
        Self::new(readers, default_lookahead)
    }

    /// Number of readers (`file_id` range is `0..n_readers`).
    pub fn n_readers(&self) -> usize {
        self.readers.len()
    }

    /// Borrow the reader for `file_id`.
    pub fn reader(&self, file_id: u32) -> &BackedCsrReader {
        &self.readers[file_id as usize]
    }

    /// Default lookahead depth.
    pub fn default_lookahead(&self) -> usize {
        self.default_lookahead
    }

    /// Shared handle to the readers' one `SharedShardCache` counters. Always
    /// populated; cloning the `Arc` lets callers sample without the cache lock.
    pub fn cache_metrics(&self) -> Arc<CacheMetrics> {
        Arc::clone(&self.cache_metrics)
    }

    /// Lazily build the prefetch runtime (2 blocking-friendly worker threads),
    /// mirroring `IndexPlanLoader::runtime`. Never built at construction, so a
    /// forked child starts with an empty `OnceLock`.
    fn runtime(&self) -> Result<&Runtime> {
        if let Some(rt) = self.runtime.get() {
            return Ok(rt.get());
        }
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("scx-plan-engine")
            .build()
            .map_err(|e| {
                LoaderError::ShutdownError(format!(
                    "failed to create tokio runtime for PrefetchEngine: {e}"
                ))
            })?;
        Ok(self
            .runtime
            .get_or_init(|| {
                crate::runtime::BoundedRuntime::new(rt, crate::pipeline::SHUTDOWN_DEADLINE)
            })
            .get())
    }

    /// Stream `plans` through the engine, pipelining shard prefetch ahead of
    /// `process`. `rows_of` reports the `(file_id, row)`s a plan touches;
    /// `process` performs the gather once those shards are warm. Returns an
    /// iterator of `Result<T>` in plan order. Empty plans are NOT skipped —
    /// `process` is called for every plan (the consumer decides).
    pub fn iter_with_plans<P, T, RowsFn, ProcFn, PlanIter>(
        self: Arc<Self>,
        plans: PlanIter,
        lookahead: usize,
        rows_of: RowsFn,
        process: ProcFn,
    ) -> PlanPrefetchIter<P, T, RowsFn, ProcFn>
    where
        P: Send + 'static,
        PlanIter: Iterator<Item = Result<P>> + Send + 'static,
        RowsFn: Fn(&P) -> Vec<(u32, u64)>,
        ProcFn: Fn(&PrefetchEngine, &P) -> Result<T>,
    {
        let cap = lookahead.max(1);
        let (plan_tx, plan_rx) = bounded(cap);

        let plan_thread = thread::Builder::new()
            .name("scx-plan-engine-pull".to_string())
            .spawn(move || {
                for item in plans {
                    if plan_tx.send(item).is_err() {
                        // Receiver dropped — iter dropped mid-stream.
                        break;
                    }
                }
                // Falling off the loop closes plan_tx, signalling EOS.
            })
            .ok();

        PlanPrefetchIter {
            engine: self,
            plan_rx,
            plan_thread,
            in_flight: VecDeque::with_capacity(cap),
            lookahead,
            plan_stream_done: false,
            plan_stream_error: None,
            rows_of,
            process,
            _phantom: PhantomData,
        }
    }
}

struct InFlight<P> {
    plan: P,
    /// Empty when `lookahead == 0`, the plan touches no rows, or every touched
    /// shard is already cached / in flight.
    prefetches: Vec<ShardJoin>,
}

/// Iterator returned by [`PrefetchEngine::iter_with_plans`].
pub struct PlanPrefetchIter<P, T, RowsFn, ProcFn> {
    engine: Arc<PrefetchEngine>,
    plan_rx: Receiver<Result<P>>,
    /// Detached on drop — the pull worker exits when `plan_rx` drops or the
    /// user iterator ends.
    plan_thread: Option<thread::JoinHandle<()>>,
    in_flight: VecDeque<InFlight<P>>,
    lookahead: usize,
    /// Sticky: once the plan stream closes we stop calling `recv`.
    plan_stream_done: bool,
    /// First plan-stream error, surfaced one-shot after the queue drains.
    plan_stream_error: Option<LoaderError>,
    rows_of: RowsFn,
    process: ProcFn,
    _phantom: PhantomData<fn() -> T>,
}

impl<P, T, RowsFn, ProcFn> PlanPrefetchIter<P, T, RowsFn, ProcFn>
where
    RowsFn: Fn(&P) -> Vec<(u32, u64)>,
{
    /// Refill the in-flight queue up to `lookahead.max(1)` plans, spawning a
    /// shard prefetch per touched shard not already resident or in flight.
    ///
    /// **Prefetch depth is opportunistic, never an obligation on the plan
    /// generator.** Only the first plan is waited for, and only when
    /// `may_block` and the queue is empty — at that point there is nothing to
    /// yield, so blocking is progress. Every later slot is filled with
    /// `try_recv`, so a generator that produces plan *i+1* only after seeing
    /// batch *i* (curriculum / feedback sampling) runs un-prefetched instead of
    /// deadlocking against a queue that will never reach `lookahead`.
    ///
    /// `may_block` is the caller's, not ours: `next` calls this a second time
    /// after popping the head purely to keep the queue warm, and at that point
    /// `in_flight` is empty in exactly the feedback case — so deciding here on
    /// `in_flight.is_empty()` alone would reinstate the deadlock one call
    /// later.
    fn refill(&mut self, may_block: bool) {
        let target = self.lookahead.max(1);
        while self.in_flight.len() < target
            && !self.plan_stream_done
            && self.plan_stream_error.is_none()
        {
            let item = if may_block && self.in_flight.is_empty() {
                match self.plan_rx.recv() {
                    Ok(item) => item,
                    Err(_) => {
                        self.plan_stream_done = true;
                        break;
                    }
                }
            } else {
                match self.plan_rx.try_recv() {
                    Ok(item) => item,
                    // Not ready is not finished. Latching `plan_stream_done`
                    // here would end the epoch the first time a generator is
                    // momentarily slow, silently truncating the data with no
                    // error anywhere.
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        self.plan_stream_done = true;
                        break;
                    }
                }
            };

            match item {
                Ok(plan) => match self.spawn_prefetches(&plan) {
                    Ok(prefetches) => {
                        self.in_flight.push_back(InFlight { plan, prefetches });
                    }
                    Err(e) => {
                        // Runtime construction failed — latch through the
                        // plan-stream-error path so it surfaces after drain.
                        self.plan_stream_error = Some(e);
                        break;
                    }
                },
                Err(e) => {
                    self.plan_stream_error = Some(e);
                    break;
                }
            }
        }
    }

    /// Spawn a `read_shard_cached_arc` prefetch per touched shard, fanned over
    /// the readers the plan's rows hit. Skips shards already cached / in flight
    /// (per reader), so a window touching the same shard queues it once.
    fn spawn_prefetches(&self, plan: &P) -> Result<Vec<ShardJoin>> {
        if self.lookahead == 0 {
            return Ok(Vec::new());
        }
        let rows = (self.rows_of)(plan);
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        // Group rows by file_id so each reader resolves its own shard set.
        let mut by_file: HashMap<u32, Vec<u64>> = HashMap::new();
        for (fid, row) in rows {
            by_file.entry(fid).or_default().push(row);
        }

        let handle = self.engine.runtime()?.handle().clone();
        let mut joins = Vec::new();
        for (fid, rs) in by_file {
            // Prefetch is best-effort: an out-of-range `file_id` from an
            // untrusted plan is skipped here (no panic) and surfaces as a
            // clean error from `SparseCellSetLoader::gather`'s validation.
            let Some(reader) = self.engine.readers.get(fid as usize) else {
                continue;
            };
            // Dedup rows + count unique rows per shard (matches the gather's
            // post-dedup `read_rows_with` grouping). Skip warming a shard that
            // is already cached / in-flight, OR **block-index-eligible** (cold +
            // sparse + framed): leaving it undecoded lets the gather take the
            // group-level block-index path instead of being negated by a
            // full-shard warm (L2). The shared eligibility predicate keeps this in
            // lockstep with the gather's `use_block_index`. Dense/large groups
            // still warm.
            let mut seen: HashSet<u64> = HashSet::with_capacity(rs.len());
            let mut per_shard: HashMap<usize, usize> = HashMap::new();
            for row in rs {
                if seen.insert(row) {
                    if let Some(sidx) = reader.index().shard_for_row(row) {
                        *per_shard.entry(sidx).or_insert(0) += 1;
                    }
                }
            }
            for (sidx, group_len) in per_shard {
                if reader.cache_contains(sidx)
                    || reader.in_flight_contains(sidx)
                    || reader.block_index_eligible(sidx, group_len)
                {
                    continue;
                }
                let reader = Arc::clone(reader);
                joins.push(handle.spawn_blocking(move || reader.read_shard_cached_arc(sidx)));
            }
        }
        Ok(joins)
    }

    /// Block on every prefetch handle for the head plan, surfacing the first
    /// shard read error or join panic.
    fn await_head(&self, prefetches: Vec<ShardJoin>) -> Result<()> {
        let runtime = self.engine.runtime()?;
        for h in prefetches {
            match runtime.block_on(h) {
                Ok(Ok(_arc_shard)) => {
                    // Warm in the shared cache; `process` will hit it.
                }
                Ok(Err(e)) => return Err(LoaderError::FormatError(e)),
                Err(join_err) => {
                    return Err(LoaderError::ShutdownError(format!(
                        "PrefetchEngine prefetch task panicked: {join_err}"
                    )))
                }
            }
        }
        Ok(())
    }
}

impl<P, T, RowsFn, ProcFn> Iterator for PlanPrefetchIter<P, T, RowsFn, ProcFn>
where
    RowsFn: Fn(&P) -> Vec<(u32, u64)>,
    ProcFn: Fn(&PrefetchEngine, &P) -> Result<T>,
{
    type Item = Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        // Nothing has been yielded yet on this call, so waiting for the first
        // plan is the only way to make progress.
        self.refill(true);

        if let Some(head) = self.in_flight.pop_front() {
            // Keep the queue warm during the upcoming process call. Strictly
            // non-blocking: we already hold a plan, and the generator may be
            // waiting on the batch it produces.
            self.refill(false);

            if let Err(e) = self.await_head(head.prefetches) {
                return Some(Err(e));
            }
            return Some((self.process)(&self.engine, &head.plan));
        }

        // Queue empty — surface a deferred plan-stream error one-shot.
        if let Some(e) = self.plan_stream_error.take() {
            self.plan_stream_done = true;
            return Some(Err(e));
        }

        None
    }
}

impl<P, T, RowsFn, ProcFn> Drop for PlanPrefetchIter<P, T, RowsFn, ProcFn> {
    /// Releases the prefetches and the pull worker. The runtime teardown is
    /// *not* done here: this iter is not reliably the last owner — the caller's
    /// `process` closure is a field of this very struct and, on the sparse
    /// path, owns an `Arc<SparseCellSetLoader>` that owns another engine `Arc`.
    /// A `Drop` body runs before its struct's fields, so any `Arc::into_inner`
    /// attempted here fails deterministically. The deadline lives in
    /// `BoundedRuntime::drop` instead — see [`crate::runtime`].
    fn drop(&mut self) {
        // Abort every in-flight shard prefetch, mirroring `IndexPlanLoader::drop`.
        // Without this, up to `lookahead` `spawn_blocking` decodes keep running on
        // the shared runtime after the iterator is gone, holding blocking-pool
        // threads and cache budget. state3 rebuilds the loader per worker per epoch
        // (`IterableDataset::__iter__`), so short-lived iterators are the norm and
        // the leak would accumulate.
        for inflight in self.in_flight.drain(..) {
            for handle in inflight.prefetches {
                handle.abort();
            }
        }
        // Detach the pull worker: dropping `plan_rx` (on struct drop) makes its
        // next `send` fail, so the worker exits on its own. We don't join — a
        // worker parked in `send` would block us. Taking the handle here also
        // marks the field as read.
        let _ = self.plan_thread.take();
    }
}

#[cfg(test)]
#[path = "plan_engine_tests.rs"]
// `pub(crate)` only so `sparse_cellset_tests` can borrow the framed fixture
// writer instead of duplicating it; every test in here stays private to the
// module by default.
pub(crate) mod tests;
