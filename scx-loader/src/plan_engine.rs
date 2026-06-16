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

use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};
use std::thread;

use crossbeam_channel::{bounded, Receiver};
use scx_format_io::{BackedCsrReader, ScxReader, SharedShardCache};
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
    runtime: OnceLock<Runtime>,
    /// Default lookahead depth (overridable per `iter_with_plans` call).
    default_lookahead: usize,
}

impl PrefetchEngine {
    /// Build an engine over `readers` (already sharing one cache). `file_id` is
    /// the reader's index in the slice.
    pub fn new(readers: Vec<Arc<BackedCsrReader>>, default_lookahead: usize) -> Arc<Self> {
        Arc::new(PrefetchEngine {
            readers,
            runtime: OnceLock::new(),
            default_lookahead,
        })
    }

    /// Convenience: build the shared cache and wrap each `ScxReader` as a
    /// CSR reader sharing it, in slice order (`file_id = index`).
    pub fn from_scx_readers(
        scx_readers: Vec<ScxReader>,
        cache_shards: usize,
        bytes_budget: usize,
        default_lookahead: usize,
    ) -> Arc<Self> {
        let shared = SharedShardCache::new(cache_shards, bytes_budget);
        let readers = scx_readers
            .into_iter()
            .enumerate()
            .map(|(fid, r)| {
                Arc::new(BackedCsrReader::with_shared_cache(
                    r,
                    fid as u32,
                    Arc::clone(&shared),
                ))
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

    /// Lazily build the prefetch runtime (2 blocking-friendly worker threads),
    /// mirroring `IndexPlanLoader::runtime`. Never built at construction, so a
    /// forked child starts with an empty `OnceLock`.
    fn runtime(&self) -> Result<&Runtime> {
        if let Some(rt) = self.runtime.get() {
            return Ok(rt);
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
        Ok(self.runtime.get_or_init(|| rt))
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
    fn refill(&mut self) {
        let target = self.lookahead.max(1);
        while self.in_flight.len() < target
            && !self.plan_stream_done
            && self.plan_stream_error.is_none()
        {
            match self.plan_rx.recv() {
                Ok(Ok(plan)) => match self.spawn_prefetches(&plan) {
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
                Ok(Err(e)) => {
                    self.plan_stream_error = Some(e);
                    break;
                }
                Err(_) => {
                    self.plan_stream_done = true;
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
            let reader = &self.engine.readers[fid as usize];
            // shards_for_indices sorts + dedups internally.
            let shards = reader.index().shards_for_indices(&rs);
            for sidx in shards {
                if reader.cache_contains(sidx) || reader.in_flight_contains(sidx) {
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
        self.refill();

        if let Some(head) = self.in_flight.pop_front() {
            // Keep the queue warm during the upcoming process call.
            self.refill();

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
    fn drop(&mut self) {
        // Detach the pull worker: dropping `plan_rx` (on struct drop) makes its
        // next `send` fail, so the worker exits on its own. We don't join — a
        // worker parked in `send` would block us. Taking the handle here also
        // marks the field as read.
        let _ = self.plan_thread.take();
    }
}

#[cfg(test)]
#[path = "plan_engine_tests.rs"]
mod tests;
