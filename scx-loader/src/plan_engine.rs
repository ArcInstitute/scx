//! Multi-reader plan-driven prefetch engine — **the** plan-prefetch iterator.
//!
//! Owns the plan-pull thread, the bounded lookahead queue, the lazy tokio
//! runtime, the per-plan shard prefetch via `spawn_blocking`, and the
//! [`IterMetrics`] counters, over a `file_id → reader` map whose readers share
//! one [`SharedShardCache`] budget.
//!
//! Both loaders run on it (ORG-9.10-1): [`crate::sparse_cellset::SparseCellSetLoader`]
//! multi-file, and [`crate::index_plan::IndexPlanLoader`] with a single reader.
//! The pair loader used to carry a fork of this file — identical `refill`,
//! `await_head` and `Drop`, drifted five ways — and `index_plan::IndexPlanIter`
//! is now a thin adapter over this iterator. **If you are about to add prefetch
//! logic to a caller, it belongs here instead**; re-forking is what this module
//! exists to prevent.
//!
//! The engine is deliberately **concrete**, parameterized by two closures
//! rather than a `PlanGather` trait (CLAUDE.md §2 — two consumers, one shape):
//!
//! * `rows_of(&plan) -> Vec<(file_id, row)>` — which rows the plan touches, so
//!   the engine can warm their shards across the right readers.
//! * `process(&engine, plan) -> Result<T>` — the actual gather, run on the
//!   consumer thread once the head plan's shards are warm. The plan arrives
//!   **by value**: the in-flight queue is its last owner, so a consumer that
//!   needs to consume or reorder it (`IndexPlanLoader::process_plan` sorts in
//!   place) does not pay a defensive clone per batch.
//!
//! Two contracts the engine deliberately does **not** impose, both of which are
//! the caller's to add if it wants them:
//!
//! * **Plan order is preserved** (no `sort_by_shard` reorder). The sparse
//!   gather recovers intra-call shard locality inside
//!   `BackedCsrReader::read_rows_with`; the pair loader sorts inside `process`.
//! * **Empty plans are not skipped** — `process` is called for every plan. The
//!   pair loader, whose spec says an empty plan yields no batch, discards the
//!   zero-row result afterwards in `IndexPlanIter::next`. It must **not** filter
//!   them out of the stream first: the pull worker learns the receiver is gone
//!   only from `plan_tx.send`, and `Filter::next` never reaches it.

use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;

use crossbeam_channel::{bounded, Receiver, TryRecvError};
use scx_format_io::{
    Admit, BackedCsrReader, CacheMetrics, RowGroupKey, ScxReader, SharedShardCache,
};
use tokio::runtime::Runtime;
use tokio::task::JoinHandle;

use crate::budget::profiling_enabled;
use crate::error::{LoaderError, Result};
use crate::reader_registry::ReaderRegistry;

/// tokio's own default `max_blocking_threads`. The engine's cap is clamped to
/// it so the change can only ever tighten concurrency, never widen it.
const TOKIO_DEFAULT_MAX_BLOCKING_THREADS: usize = 512;

/// The blocking-cap arithmetic, as a pure function of its two inputs.
///
/// Split out so it can be pinned by a test that supplies the pool width instead
/// of guessing it. A Python test that re-derived the width with
/// `os.cpu_count()` went red on CI: `cpu_pool` resolves from
/// `num_cpus::get_physical()`, and on a 2-vCPU / 1-physical runner the two
/// disagree — the second environment-dependent version of the same test.
pub(crate) fn clamp_blocking_threads(pool: usize, lookahead: usize) -> usize {
    pool.saturating_add(lookahead)
        .clamp(2, TOKIO_DEFAULT_MAX_BLOCKING_THREADS)
}

/// One prefetch task: a whole-shard warm (`read_shard_cached_arc`) or a
/// row-group warm (`warm_row_groups`). Both are side effects on the shared
/// cache; the decoded value itself is never handed back.
type ShardJoin = JoinHandle<scx_format_io::Result<()>>;

/// A row-group cache key and the decoded bytes it will occupy.
type KeyedBytes = (RowGroupKey, usize);

/// `(planned bytes, byte budget, the keys summed with their sizes)` —
/// [`PrefetchEngine::plan_footprint_keyed`]'s answer.
type PlanFootprint = (usize, usize, Vec<KeyedBytes>);

/// `(prefetch handles, whether the plan fits its budget share, the share it was
/// measured against, its row-group keys with their sizes)` — what one refill's
/// `spawn_prefetches` produces.
type SpawnedPlan = (Vec<ShardJoin>, bool, usize, Vec<KeyedBytes>);

/// A `file_id → reader` map sharing one decoded-shard budget, plus a lazily
/// built tokio runtime for prefetch. Wrap in `Arc` and call
/// [`PrefetchEngine::iter_with_plans`].
pub struct PrefetchEngine {
    /// `file_id → reader`, opened lazily and bounded by `reader_limit` when
    /// the caller sets one. All handles share one [`SharedShardCache`] via
    /// [`BackedCsrReader::with_shared_cache`]. See [`ReaderRegistry`] for why
    /// a handle is leased rather than borrowed, and for the measurement that
    /// says the resource being bounded is resident memory, not descriptors.
    registry: Arc<ReaderRegistry>,
    /// Lazily built so it is never inherited across a fork — mirrors
    /// `IndexPlanLoader`. Built on the first `iter_with_plans` consumption.
    runtime: OnceLock<crate::runtime::BoundedRuntime>,
    /// Default lookahead depth (overridable per `iter_with_plans` call).
    default_lookahead: usize,
    /// Shared handle to the readers' one `SharedShardCache` counters
    /// (hits / misses / evictions / …). Mirrors `IndexPlanLoader::cache_metrics`;
    /// always populated (a zeroed default when no reader enabled metrics).
    cache_metrics: Arc<CacheMetrics>,
    /// Test-only rendezvous that parks every prefetch task in flight. Set
    /// through `&self` (not at construction) because the engine is handed out
    /// as an `Arc` and the pair loader builds it lazily inside `OnceLock`.
    #[cfg(test)]
    prefetch_gate: OnceLock<Arc<PrefetchGate>>,
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
        Self::over_registry(
            ReaderRegistry::from_open(readers),
            default_lookahead,
            cache_metrics,
        )
    }

    /// Build an engine over an already-constructed registry.
    ///
    /// The seam a bounded manifest scan enters through: it has opened at most
    /// `reader_limit` of the files and cannot hand over a `Vec` of all of them.
    pub(crate) fn over_registry(
        registry: Arc<ReaderRegistry>,
        default_lookahead: usize,
        cache_metrics: Arc<CacheMetrics>,
    ) -> Arc<Self> {
        Arc::new(PrefetchEngine {
            registry,
            runtime: OnceLock::new(),
            default_lookahead,
            cache_metrics,
            #[cfg(test)]
            prefetch_gate: OnceLock::new(),
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
        // Through the shared factory, which owns the pre-`Arc` sequence — the
        // block-index gate, the shared pool, metrics — so that a reader the
        // registry reopens later is configured identically to one opened here.
        let readers = scx_readers
            .into_iter()
            .enumerate()
            .map(|(fid, r)| {
                crate::reader_registry::wrap_reader(r, fid as u32, &shared, scatter_block_index)
            })
            .collect();
        Self::new(readers, default_lookahead)
    }

    /// Number of readers (`file_id` range is `0..n_readers`).
    pub fn n_readers(&self) -> usize {
        self.registry.n_files()
    }

    /// The registry, for callers that need a file's metadata without a handle.
    pub(crate) fn registry(&self) -> &Arc<ReaderRegistry> {
        &self.registry
    }

    /// An owned handle for `file_id`, valid until the caller drops it.
    ///
    /// Owned, not borrowed: a bounded registry may close a handle to reclaim
    /// its parsed catalog, and the reads this feeds hold their receiver across
    /// a parallel decode, a single-flight wait, or — in the prefetcher's case —
    /// an unabortable `spawn_blocking` that outlives the iterator. Fallible
    /// because a reopen can fail or land on a replaced file.
    pub fn lease(&self, file_id: u32) -> Result<Arc<BackedCsrReader>> {
        self.registry.lease(file_id)
    }

    /// True if **any** reader has at least one row-group-framed CSR shard, i.e.
    /// the scattered block-index fast path can fire somewhere in this set.
    ///
    /// Any, not all, and that is the whole semantic: one framed file among
    /// twenty means the route is live for that file's rows, so treating the set
    /// as inert would be wrong. Callers use it at open time to warn that
    /// `scatter_block_index=True` bought them nothing —
    /// [`BackedCsrReader::any_shard_framed`] reads shard *headers* only,
    /// memoized per shard, and both this and it short-circuit on the first
    /// framed shard, so the cost is paid in full only by an all-unframed set.
    pub fn any_shard_framed(&self) -> bool {
        self.registry.any_shard_framed()
    }

    /// Deduplicate a plan's `(file, row)` pairs into per-`(file, shard)` buckets.
    ///
    /// The gather passes the same deduplicated set to `read_rows_with`, so each
    /// bucket's length is the `group_len` the block-index decision sees.
    /// An out-of-range `file_id` is skipped rather than panicked on: the
    /// gather's own validation is what reports it, and this runs first.
    pub(crate) fn bucket_plan_rows(
        &self,
        pairs: impl Iterator<Item = (u32, u64)>,
    ) -> HashMap<(u32, usize), Vec<u64>> {
        // Reserved from the iterator's own hint, not left to grow: both callers
        // pass an exact-size iterator, and letting `seen` rehash its way up on
        // every admission decision is the same allocation-and-copy work W1
        // exists to remove. Regression introduced when this walk was extracted
        // from its two copies, each of which did reserve.
        let (lower, upper) = pairs.size_hint();
        let mut seen: HashSet<(u32, u64)> = HashSet::with_capacity(upper.unwrap_or(lower));
        let mut per_shard: HashMap<(u32, usize), Vec<u64>> = HashMap::new();
        for (fid, row) in pairs {
            if !seen.insert((fid, row)) {
                continue;
            }
            // Served from the registry's retained shard index, so bucketing a
            // plan opens nothing. That is what keeps a wide plan's admission
            // sizing from pulling every file it touches into residence before
            // the gather has decided it wants them.
            if let Some(sidx) = self.registry.shard_for_row(fid, row) {
                per_shard.entry((fid, sidx)).or_default().push(row);
            }
        }
        per_shard
    }

    /// `(planned bytes, byte budget)` for a bucketed plan — the ONE definition
    /// of "this plan's footprint in the shared LRU".
    ///
    /// **Review on #535 round 2 (Cursor Agent, codex):** this walk existed
    /// twice, once here and once inline in `spawn_prefetches`, differing only in
    /// what the result is compared against — the whole budget for a synchronous
    /// one-plan caller, `budget / (lookahead + 1)` for a prefetcher sharing it
    /// with a lookahead window. Two copies of the sizing is exactly how the
    /// per-set/union split this method exists to close would come back.
    ///
    /// Counts the row-group bytes of every touched shard, plus the decoded size
    /// of any shard the plan will take **whole** — both kinds share one budget.
    /// Sized from the catalog and the block index; decodes nothing.
    ///
    /// Returning the budget alongside the footprint is what lets a standalone
    /// gather make ONE verdict over the plan instead of letting each set decide
    /// for itself: per-set decisions let a plan whose sets individually fit,
    /// but whose union does not, insert and evict row groups against each
    /// other — exactly the churn the plan-level verdict was introduced to stop
    /// on the iterator path.
    ///
    /// Unlike `bucket_plan_rows`, this needs a real handle per touched file —
    /// the framing memo and the per-shard decoded size are not in the retained
    /// index.
    ///
    /// `held` is what makes that affordable for both callers, which want
    /// opposite things. A synchronous `gather` passes `None`: this leases one
    /// file at a time and drops each, so sizing costs one resident handle
    /// whatever the plan's width. The prefetcher passes the leases it is
    /// already holding, because it needs every touched file anyway — and
    /// without that, a plan touching more files than `reader_limit` would have
    /// this function evict each file as it moved to the next, then watch
    /// `spawn_prefetches` reopen and re-parse every one of them a line later.
    /// **Review on #536 (Antigravity, Cursor Agent):** that thrash was real,
    /// and it was introduced by the fix that made sizing lease one at a time.
    pub(crate) fn plan_footprint(
        &self,
        per_shard: &HashMap<(u32, usize), Vec<u64>>,
        held: Option<&HashMap<u32, Arc<BackedCsrReader>>>,
    ) -> Result<(usize, usize)> {
        let (planned, budget, _) = self.plan_footprint_keyed(per_shard, held, false)?;
        Ok((planned, budget))
    }

    /// [`Self::plan_footprint`], optionally also naming the row-group cache keys
    /// it summed.
    ///
    /// One walk of the block index, two answers. The prefetcher needs both — the
    /// footprint to decide whether the plan fits its share, and the keys to say
    /// *which* groups a partially-admitted plan may retain — and a second walk
    /// to collect the keys is how a sizing decision and an admission decision
    /// come to disagree about a group boundary. Same reasoning as
    /// `BackedCsrReader::planned_row_group_bytes` folding over
    /// `touched_row_groups`.
    ///
    /// `collect_keys == false` allocates nothing extra, which is what the
    /// synchronous `gather` path wants: it has no lookahead window, so no key
    /// can ever be touched by a second plan and the keys would be dead weight.
    pub(crate) fn plan_footprint_keyed(
        &self,
        per_shard: &HashMap<(u32, usize), Vec<u64>>,
        held: Option<&HashMap<u32, Arc<BackedCsrReader>>>,
        collect_keys: bool,
    ) -> Result<PlanFootprint> {
        let mut keys: Vec<KeyedBytes> = Vec::new();
        let mut planned = 0usize;
        let mut budget = usize::MAX;
        // Regrouped per file so that, with `held` absent, each file's lease is
        // dropped before the next is taken and sizing holds **one** handle at a
        // time. Holding them all was the first version, and it defeated
        // `reader_limit` outright: a 64-file plan pinned 64 readers before a
        // single row was read, so nothing was ever evictable and the residency
        // high-water equalled the manifest size. Caught by
        // `residency_is_bounded_by_reader_limit`.
        let mut by_file: HashMap<u32, Vec<(usize, &Vec<u64>)>> = HashMap::new();
        for (&(fid, sidx), shard_rows) in per_shard {
            by_file.entry(fid).or_default().push((sidx, shard_rows));
        }
        for (fid, shards) in by_file {
            let reader = match held.and_then(|m| m.get(&fid)) {
                Some(r) => Arc::clone(r),
                None => self.registry.lease(fid)?,
            };
            for (sidx, shard_rows) in shards {
                if collect_keys {
                    // The keyed walk subsumes the sizing one: `touched_row_groups`
                    // is what `planned_row_group_bytes` folds over, so summing the
                    // bytes here gives the identical number without walking twice.
                    for (g, bytes) in reader.touched_row_groups(sidx, shard_rows) {
                        planned = planned.saturating_add(bytes);
                        keys.push(((fid, sidx, g), bytes));
                    }
                } else {
                    planned =
                        planned.saturating_add(reader.planned_row_group_bytes(sidx, shard_rows));
                }
                if !reader.block_index_eligible(sidx, shard_rows.len()) {
                    planned = planned.saturating_add(reader.shard_decoded_bytes(sidx));
                }
            }
            budget = budget.min(reader.cache_bytes_budget());
        }
        Ok((planned, budget, keys))
    }

    /// Shared handle to the readers' one `SharedShardCache` counters. Always
    /// populated; cloning the `Arc` lets callers sample without the cache lock.
    pub fn cache_metrics(&self) -> Arc<CacheMetrics> {
        Arc::clone(&self.cache_metrics)
    }

    /// Install the test-only prefetch gate (idempotent; the first call wins).
    #[cfg(test)]
    pub(crate) fn set_prefetch_gate(&self, gate: Arc<PrefetchGate>) {
        let _ = self.prefetch_gate.set(gate);
    }

    /// Whether the lazy prefetch runtime has been materialised yet. The
    /// fork-safety contract is that construction alone never builds it, and
    /// this is how an owner one level up (`IndexPlanLoader`) can assert that
    /// without reaching into a private field of another module.
    #[cfg(test)]
    pub(crate) fn runtime_is_built(&self) -> bool {
        self.runtime.get().is_some()
    }

    /// Concurrent shard decodes this engine's blocking pool will run at once.
    ///
    /// `lookahead` bounds in-flight **plans**, not tasks: `spawn_prefetches`
    /// issues one `spawn_blocking` per distinct `(file, shard)` a plan touches,
    /// which is caller-controlled through plan width and reaches 48+ on the
    /// STATE3 shapes `suggested_cache_shards` was written for. Left at tokio's
    /// default the ceiling is **512** simultaneously decoding shards, which for
    /// a census-sized shard is not a bound in any useful sense.
    ///
    /// Sized to the decode pool rather than to `lookahead`: the work is
    /// CPU-bound, so more concurrent decodes than cores buys queueing, not
    /// throughput — while a bound of `lookahead + 1` would cap the ~192-task
    /// case at 5 and serialise exactly the wide plans this loader exists for.
    /// The `+ lookahead` keeps the next plans' warms able to start while the
    /// current one's decodes occupy the pool.
    /// Cap on simultaneously-running shard decodes in the blocking pool.
    ///
    /// **Clamped to tokio's own default of 512 at the top**, so this can only
    /// ever tighten the bound, never loosen it: `cpu_pool`'s size is
    /// `physical.clamp(1, 8)` by default but `SCX_LOADER_CPU_THREADS` is
    /// documented as able to exceed that clamp, and without the ceiling a large
    /// override would permit MORE concurrent blocking tasks than the default
    /// this replaces.
    ///
    /// Sized from `default_lookahead`, the constructor's value — a per-iterator
    /// `lookahead` passed to `iter_with_plans` does not resize a runtime that
    /// is built once and shared.
    pub fn max_blocking_threads(&self) -> usize {
        clamp_blocking_threads(
            crate::pool::cpu_pool().current_num_threads(),
            self.default_lookahead,
        )
    }

    /// Lazily build the prefetch runtime (2 blocking-friendly worker threads),
    /// Never built at construction, so a
    /// forked child starts with an empty `OnceLock`.
    fn runtime(&self) -> Result<&Runtime> {
        if let Some(rt) = self.runtime.get() {
            return Ok(rt.get());
        }
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .max_blocking_threads(self.max_blocking_threads())
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
    ///
    /// `process` takes the plan **by value**. The queue is the plan's last
    /// owner, so handing it over costs nothing, and it lets a consumer that
    /// needs to consume or reorder the plan — `IndexPlanLoader::process_plan`
    /// sorts in place and moves the result into the batch — do so without a
    /// defensive clone on every batch of the training hot path. Its third
    /// argument is the plan's **row-group admission** verdict (see
    /// `spawn_prefetches`): the consumer passes it to
    /// `BackedCsrReader::read_rows_with_admission` for every gather the plan
    /// makes, so the gathers retain only what the plan's verdict admits (the
    /// warm pre-decodes the eligible subset of that) and a plan whose union of
    /// gathers is over budget retains nothing.
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
        ProcFn: Fn(&PrefetchEngine, P, Admit) -> Result<T>,
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
            window: HashMap::new(),
            #[cfg(test)]
            block_until_full: false,
            lookahead,
            plan_stream_done: false,
            plan_stream_error: None,
            rows_of,
            process,
            iter_metrics: Arc::new(IterMetrics::default()),
            _phantom: PhantomData,
        }
    }
}

/// Per-iter prefetch counters. Sampled through
/// [`PlanPrefetchIter::iter_metrics`] and emitted by the Drop-time profile log
/// when `SCX_LOADER_PROFILE=1`.
///
/// All atomics use `Relaxed` ordering — values are statistical and not used for
/// synchronization.
///
/// **With prefetch enabled**, the four counters partition every shard a valid
/// plan touches: each is either spawned as a whole-shard warm or not, for
/// exactly one reason, which is what `tests/test_index_plan.rs`'s conservation
/// law checks. At `lookahead == 0` there is no partition — `spawn_prefetches`
/// returns before any of them, so all four stay zero while the gather still
/// picks a route. These are prefetch-time decisions;
/// `CacheMetrics::block_index_groups` is the route the gather actually took,
/// and the two are not interchangeable.
#[derive(Default, Debug)]
pub struct IterMetrics {
    /// Whole-shard `read_shard_cached_arc` tasks queued onto the runtime's
    /// blocking pool. Row-group warms (see `prefetch_skipped_block_index`) are
    /// not counted here, so the four-way partition of touched shards holds.
    pub prefetch_tasks_spawned: AtomicU64,
    /// Shards whose prefetch was skipped because the LRU already held them.
    pub prefetch_skipped_cache_hit: AtomicU64,
    /// Shards whose prefetch was skipped because a peer leader was already
    /// decoding them in the shared cache's singleflight table.
    pub prefetch_skipped_in_flight: AtomicU64,
    /// Shards whose **whole-shard** prefetch was skipped because the group is
    /// **block-index eligible** (not resident whole + sparse + row-group
    /// framed): the gather decodes only the touched row-groups via the block
    /// index, so warming the whole shard would negate the win (the L2
    /// block-index-aware prefetch skip). The name predates OPT-FORMATIO-1 and
    /// is a contract; since then such a shard is not left cold — when the
    /// whole plan's row groups (every file, every shard) fit
    /// `cache_bytes_budget / (lookahead + 1)`, its touched groups are
    /// pre-decoded into the shard LRU instead (`BackedCsrReader::warm_row_groups`),
    /// and the gather serves them as `CacheMetrics::row_group_hits`. A plan
    /// over its share is neither warmed nor retained by its gathers (the same
    /// verdict reaches them through `process`'s third argument): warming it
    /// would only have evicted the groups before they were read.
    pub prefetch_skipped_block_index: AtomicU64,
    /// Plans whose L2 prefetch was declined outright because they touch more
    /// distinct files than `reader_limit` allows to be resident.
    ///
    /// Zero on every unbounded dataset, which is the default. Non-zero means
    /// the manifest cap and the plan shape are fighting: prefetch cannot hold a
    /// lease per launched file without pinning more than the cap, so it stands
    /// down and the synchronous gather reads one file at a time instead. The
    /// fix is on the caller's side — a wider `reader_limit`, or plans with more
    /// file locality — and this counter is how they find out it is happening.
    pub prefetch_skipped_reader_limit: AtomicU64,
}

/// Test-only rendezvous for holding prefetch tasks in flight.
///
/// `started` must be signalled *before* parking: a test that infers "a task is
/// running" from a sleep passes on a loaded machine even when the closure has
/// regressed to capturing an owner it must not hold, which is precisely the
/// blindness this gate exists to remove.
#[cfg(test)]
pub(crate) struct PrefetchGate {
    started: AtomicU64,
    hold: std::sync::atomic::AtomicBool,
}

#[cfg(test)]
impl PrefetchGate {
    pub(crate) fn new() -> Self {
        Self {
            started: AtomicU64::new(0),
            hold: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// Called from inside a prefetch task: announce, then park.
    fn enter(&self) {
        self.started.fetch_add(1, Ordering::AcqRel);
        while self.hold.load(Ordering::Acquire) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// Block until at least one task has entered, or fail after `timeout`.
    /// Returning `false` means the test could not establish its premise and
    /// must fail rather than assert against an empty runtime.
    pub(crate) fn wait_for_entry(&self, timeout: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        while std::time::Instant::now() < deadline {
            if self.started.load(Ordering::Acquire) > 0 {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        false
    }

    pub(crate) fn release(&self) {
        self.hold.store(false, Ordering::Release);
    }
}

/// The admission verdict for a plan leaving the queue, as a pure function of
/// its own `keys`, whether it fit its budget share, and the window.
///
/// Free-standing so the threshold can be tested without driving an iterator:
/// end to end, whether a peer plan is queued when a verdict is taken depends on
/// how promptly the pull worker was scheduled, and a test asserting the
/// *positive* case that way is flaky by construction (measured at roughly one
/// failure in six under a loaded `cargo test`).
///
/// `count >= 2` is evaluated with the leaving plan's own keys still counted, so
/// it reads as "at least one plan **still queued** also wants this group".
fn verdict_from_window(
    fits_share: bool,
    keys: &[KeyedBytes],
    window: &HashMap<RowGroupKey, u32>,
    share: usize,
) -> Admit {
    if fits_share {
        return Admit::All;
    }
    // The A/B arm: with the reuse policy off, a plan over its share retains
    // nothing, which is what every gather did before W10.
    if !scx_format_io::backed::row_group_admit_reuse_enabled() {
        return Admit::None;
    }
    // Hottest first, ties by key so the verdict is reproducible, and **bounded
    // by the same share the whole-plan rule uses**.
    //
    // ⚠️ The bound is not belt-and-braces; without it the signal degenerates.
    // Measured on tabula_sapiens_100k with a 64-set batch: a plan's random tail
    // touches nearly every row group the file has (≈392 at G=256), so every key
    // reaches a window count of 2 and "admit the reused groups" becomes "admit
    // everything" — the regime the first row-group LRU capture measured at 0 %
    // hits, +350-500 MB and 2-4 % slower. Taking hot keys only while they fit
    // the share keeps the recovery and keeps the bound.
    let mut hot: Vec<&KeyedBytes> = keys
        .iter()
        .filter(|(k, _)| window.get(k).is_some_and(|&c| c >= 2))
        .collect();
    hot.sort_unstable_by(|(ka, _), (kb, _)| {
        let (ca, cb) = (window.get(ka).copied(), window.get(kb).copied());
        cb.cmp(&ca).then(ka.cmp(kb))
    });
    let mut taken: HashSet<RowGroupKey> = HashSet::new();
    let mut bytes = 0usize;
    for (k, b) in hot {
        let next = bytes.saturating_add(*b);
        if next > share {
            // `break`, not `continue`: the list is hottest-first, so a smaller
            // key further down is a colder one, and taking it would trade a
            // more-reused group for a less-reused one to fill the last few
            // bytes.
            break;
        }
        bytes = next;
        taken.insert(*k);
    }
    Admit::groups(taken)
}

struct InFlight<P> {
    plan: P,
    /// Empty when `lookahead == 0`, the plan touches no rows, or every touched
    /// shard is already cached / in flight.
    prefetches: Vec<ShardJoin>,
    /// Whether the plan's whole footprint fit its share of the byte budget
    /// (see `spawn_prefetches`). `true` is a total admission; `false` is where
    /// the reuse signal below decides.
    fits_share: bool,
    /// The `budget / (lookahead + 1)` share `fits_share` was measured against,
    /// and the ceiling a partial verdict is allowed to admit up to.
    share: usize,
    /// Every row-group cache key this plan touches, its contribution to the
    /// iterator's rolling `window`. Carried rather than recomputed on the way
    /// out so the decrement is exactly the increment — a recomputation could
    /// differ if the plan's readers were evicted and reopened in between, and a
    /// key that is incremented once and decremented zero times stays "hot"
    /// forever.
    keys: Vec<KeyedBytes>,
}

/// Iterator returned by [`PrefetchEngine::iter_with_plans`].
pub struct PlanPrefetchIter<P, T, RowsFn, ProcFn> {
    engine: Arc<PrefetchEngine>,
    plan_rx: Receiver<Result<P>>,
    /// Detached on drop — the pull worker exits when `plan_rx` drops or the
    /// user iterator ends.
    plan_thread: Option<thread::JoinHandle<()>>,
    in_flight: VecDeque<InFlight<P>>,
    /// How many of the plans currently in flight touch each row-group key.
    ///
    /// **The reuse signal** (W10 step 2). A plan whose whole footprint fits its
    /// share of the budget is admitted outright, as before. A plan over its
    /// share used to forfeit retention entirely — measured at a 0.000 row-group
    /// hit rate on every multi-shard fixture — and now keeps exactly the groups
    /// a second plan of the window also touches: a control pool, shared
    /// neighbours, a repeated pair member. Its cold tail still decodes and
    /// drops, which is the part an LRU genuinely makes worse.
    ///
    /// Incremented when a plan enters the queue, decremented when it leaves, so
    /// a count of `>= 2` at the moment a plan is consumed means "some other
    /// plan still queued also wants this group".
    window: HashMap<RowGroupKey, u32>,
    /// Test-only rendezvous: make `refill(true)` fill the queue to `lookahead`
    /// instead of returning as soon as it holds one plan.
    ///
    /// The production rule is deliberately opportunistic — "prefetch depth is
    /// never an obligation on the plan generator" — so how many plans are
    /// queued when a verdict is taken depends on scheduling. A test that needs
    /// a *populated* window has to pin that, exactly as `PrefetchGate` pins
    /// "a prefetch task is running" rather than racing a sleep against it.
    #[cfg(test)]
    pub(crate) block_until_full: bool,
    lookahead: usize,
    /// Sticky: once the plan stream closes we stop calling `recv`.
    plan_stream_done: bool,
    /// First plan-stream error, surfaced one-shot after the queue drains.
    plan_stream_error: Option<LoaderError>,
    rows_of: RowsFn,
    process: ProcFn,
    /// Per-iter prefetch counters. Cloning the `Arc` lets a consumer sample
    /// without going through any lock, and keeps the handle valid after this
    /// iterator has been drained and dropped.
    iter_metrics: Arc<IterMetrics>,
    _phantom: PhantomData<fn() -> T>,
}

impl<P, T, RowsFn, ProcFn> PlanPrefetchIter<P, T, RowsFn, ProcFn>
where
    RowsFn: Fn(&P) -> Vec<(u32, u64)>,
{
    /// Cloneable handle to this iter's prefetch counters. Sample at any
    /// time — atomics are `Relaxed`, no locks involved.
    pub fn iter_metrics(&self) -> Arc<IterMetrics> {
        Arc::clone(&self.iter_metrics)
    }

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
            #[cfg(test)]
            let may_block = may_block && (self.block_until_full || self.in_flight.is_empty());
            #[cfg(not(test))]
            let may_block = may_block && self.in_flight.is_empty();
            let item = if may_block {
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
                    Ok((prefetches, fits_share, share, keys)) => {
                        for (k, _) in &keys {
                            *self.window.entry(*k).or_insert(0) += 1;
                        }
                        self.in_flight.push_back(InFlight {
                            plan,
                            prefetches,
                            fits_share,
                            share,
                            keys,
                        });
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
    /// the readers the plan's rows hit, and decide the plan's **row-group
    /// admission** — returned alongside the join handles so `process` applies
    /// the same verdict to every gather of the plan.
    ///
    /// Skips shards already cached / in flight (per reader), so a window
    /// touching the same shard queues it once. A block-index-eligible shard is
    /// never warmed whole; when the plan is admitted it gets a
    /// `warm_row_groups` task instead.
    ///
    /// **One decision per plan, over the shared budget.** Every reader of the
    /// engine draws on one `SharedShardCache`, and a plan's gathers may span
    /// several files (the cell-set loader) and several `read_rows_with` calls
    /// (one per set). So the plan's footprint in that cache — the row-group
    /// bytes of every framed shard it touches, regardless of current residency
    /// or per-bucket density (see the body for why those are not stable), plus
    /// the decoded size of every shard it will take whole — is summed over the
    /// whole plan, every file, every shard, and compared once against
    /// `cache_bytes_budget / (lookahead + 1)`: up to `lookahead` plans are
    /// warming while one is consumed, so that is the share under which a warm
    /// still resides when its gather arrives, and under which the gathers'
    /// own inserts cannot evict a peer plan's warm. A per-file or per-gather
    /// verdict against the full budget let N files (or N sets) each pass while
    /// their union thrashed the LRU — found by review on #528. At
    /// `lookahead == 0` nothing is warmed and the share is the whole budget.
    fn spawn_prefetches(&self, plan: &P) -> Result<SpawnedPlan> {
        let rows = (self.rows_of)(plan);
        if rows.is_empty() {
            return Ok((Vec::new(), true, usize::MAX, Vec::new()));
        }

        // Dedup rows and bucket them per (file, shard). The gather passes the
        // same deduped set to `read_rows_with`, so each bucket's length is the
        // `group_len` its block-index decision sees. Deciding on the same input
        // is what keeps the skip below meaningful as evidence about the gather
        // — though not a guarantee it agrees: a peer sharing the cache can
        // warm a shard in between.
        // Bucketed through the engine's shared helper, so the prefetcher and a
        // synchronous `gather` cannot drift on what a plan's footprint means.
        // Prefetch is best-effort: an out-of-range `file_id` from an untrusted
        // plan is skipped there (no panic) and surfaces as a clean error from
        // the gather's validation.
        let per_shard = self.engine.bucket_plan_rows(rows.into_iter());

        // Plan-level admission, sized exactly from the catalog and the block
        // index (no decode). `planned` is an upper bound on the plan's whole
        // footprint in the shared LRU — both kinds of entry, since they share
        // one budget (review on #528, round 3):
        //
        // * the row-group bytes of every framed shard the plan touches — NOT
        //   filtered by `block_index_eligible`, which is volatile in two ways
        //   the gather can disagree with by the time it runs: its `!contains`
        //   clause looks at whole-shard residency, which a warm can change
        //   before the plan is consumed; and its density window sees this
        //   bucket's row count, while the cell-set loader gathers one set at a
        //   time and each set's smaller count can be sparse where the plan's
        //   union is dense;
        // * the decoded size of every shard the plan will take **whole** (the
        //   non-eligible buckets, warmed by `read_shard_cached_arc` below or
        //   already resident) — a mixed plan whose groups fit the share on
        //   their own but not next to its whole shards would otherwise evict
        //   one with the other on every repeat.
        //
        // Counting a shard the gather then happens to serve the other way only
        // makes the verdict conservative (a lost warm in a mixed regime), never
        // unsafe. Eligibility still decides what L2 task to launch.
        // At `lookahead == 0` there is no prefetch to launch, so the plan needs
        // its admission verdict and nothing else: neither the leases nor the
        // eligibility map below survives the early return. Taking them first
        // spiked residency to the plan's width in order to compute one `bool`,
        // on a configuration documented as disabling prefetch entirely.
        // **Review on #536 (Cursor Agent).**
        // Two ways to reach "no prefetch tasks", both of which still owe the
        // caller a real admission verdict.
        //
        // `lookahead == 0` disables prefetch outright. And a plan touching more
        // distinct files than the registry may keep resident cannot be
        // prefetched either: the prefetcher holds a lease on every file it
        // launches work for — the tasks are unabortable and slice the mapping —
        // so it would pin the plan's whole width, defeat `reader_limit`, and
        // then hand the gather handles the next trim evicts. Measured on #536
        // by codex - gpt-5.6-sol: a 32-file plan at `reader_limit=2,
        // lookahead=4` moved `opens/hwm` from 2/2 to 65/32.
        //
        // **The verdict is still computed, not assumed.** An earlier version
        // returned a flat `false` here on the reasoning that a plan whose
        // readers cannot stay resident cannot get row-group hits either. That
        // conflates two lifetimes: `CacheKey::Group` entries live in the shared
        // byte-budgeted cache and outlive a handle eviction — this module's own
        // docs say so — so a wide plan's few touched groups can fit and hit on
        // the next epoch while its catalogs cannot all stay open. Admission is
        // a statement about the decoded-cache footprint, never about the file
        // count. Flagged independently by codex and Cursor Agent on #536.
        //
        // Sizing here costs one lease per touched file, taken and dropped one
        // at a time, which on a wide plan the gather then repeats. That is the
        // price of a correct verdict; the cheaper answer is a per-shard byte
        // summary retained by the manifest scan, which is a larger change than
        // this round should carry.
        //
        // Unbounded registries (`reader_limit = None`, the default) never take
        // the second branch, so the default path is untouched.
        let declined_for_reader_limit = self.engine.registry().limit().is_some_and(|limit| {
            let touched: HashSet<u32> = per_shard.keys().map(|&(fid, _)| fid).collect();
            touched.len() > limit
        });
        if declined_for_reader_limit {
            self.iter_metrics
                .prefetch_skipped_reader_limit
                .fetch_add(1, Ordering::Relaxed);
        }
        if self.lookahead == 0 || declined_for_reader_limit {
            // No keys on either of these paths, and neither needs them. At
            // `lookahead == 0` the queue holds one plan, so no key can ever
            // reach a count of two; and a plan declined for `reader_limit` is
            // deliberately not leasing its files, which is what naming its keys
            // would cost. Both keep today's all-or-nothing verdict.
            let (planned, budget) = self.engine.plan_footprint(&per_shard, None)?;
            let share = budget / (self.lookahead + 1);
            return Ok((Vec::new(), planned <= share, share, Vec::new()));
        }

        // One lease per touched file for the whole of the rest of this
        // function. Taken up front, and held, for three reasons: the
        // eligibility map and the warm loop below would otherwise lease the
        // same file once per bucket; a handle re-leased between the two could
        // be a *different* `Arc` if a bounded registry evicted and reopened it
        // in between, so the predicate and the task it launches would be
        // reading different objects; and taking them BEFORE the sizing below is
        // what stops a plan wider than `reader_limit` from being sized
        // file-by-file, evicting as it goes, and then reopening every one of
        // them here.
        let mut leased: HashMap<u32, Arc<BackedCsrReader>> = HashMap::new();
        for &(fid, _) in per_shard.keys() {
            if let std::collections::hash_map::Entry::Vacant(e) = leased.entry(fid) {
                e.insert(self.engine.lease(fid)?);
            }
        }
        // Sized against the handles already held, so no file is opened twice
        // for one plan.
        let (planned, budget, keys) =
            self.engine
                .plan_footprint_keyed(&per_shard, Some(&leased), true)?;
        // Recomputed here because the prefetcher needs it per bucket to choose
        // which L2 task to launch; `plan_footprint` consumes the same verdict
        // internally to decide whether to add whole-shard bytes.
        let eligible: HashMap<(u32, usize), bool> = per_shard
            .iter()
            .map(|(&(fid, sidx), rows)| {
                let reader = &leased[&fid];
                ((fid, sidx), reader.block_index_eligible(sidx, rows.len()))
            })
            .collect();
        let share = budget / (self.lookahead + 1);
        let admit_row_groups = planned <= share;

        let handle = self.engine.runtime()?.handle().clone();
        let mut joins = Vec::new();
        for ((fid, sidx), mut rows) in per_shard {
            let reader = &leased[&fid];
            // Attributed one reason at a time, not as a fused `||`: the three
            // skips mean different things to an operator (a warm cache, a peer
            // decode, the L2 block-index adoption) and a fused test can only
            // ever check their sum.
            //
            // The third is the interesting one: not warming a sparse + framed
            // shard *whole* is what lets `read_rows_with` take the group-level
            // block-index path instead of being negated by a full-shard warm.
            // Both sides call the same `block_index_eligible` on the same
            // `group_len`, so neither drifts from the other's *predicate* — but
            // the cache can change under them, so this counter is the
            // prefetch-time decision, not proof of the gather's route. Dense
            // or large groups fall through and warm as before.
            if reader.cache_contains(sidx) {
                self.iter_metrics
                    .prefetch_skipped_cache_hit
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if reader.in_flight_contains(sidx) {
                self.iter_metrics
                    .prefetch_skipped_in_flight
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }
            // Capture the *reader*, never an owner further up: an
            // already-started `spawn_blocking` cannot be aborted, so a task
            // holding the engine or a loader could outlive this iter and
            // release the final reference — dropping the runtime from one of
            // its own threads. See `crate::runtime`.
            let reader = Arc::clone(reader);
            #[cfg(test)]
            let gate = self.engine.prefetch_gate.get().cloned();
            if eligible[&(fid, sidx)] {
                self.iter_metrics
                    .prefetch_skipped_block_index
                    .fetch_add(1, Ordering::Relaxed);
                // Row-group warm (OPT-FORMATIO-1): only for an admitted plan —
                // warming a plan over its share would evict the groups before
                // they are read and cost a second decode.
                if admit_row_groups && planned > 0 {
                    rows.sort_unstable();
                    joins.push(handle.spawn_blocking(move || {
                        #[cfg(test)]
                        if let Some(gate) = gate {
                            gate.enter();
                        }
                        reader.warm_row_groups(sidx, &rows).map(|_| ())
                    }));
                }
                continue;
            }
            self.iter_metrics
                .prefetch_tasks_spawned
                .fetch_add(1, Ordering::Relaxed);
            joins.push(handle.spawn_blocking(move || {
                #[cfg(test)]
                if let Some(gate) = gate {
                    gate.enter();
                }
                reader.read_shard_cached_arc(sidx).map(|_| ())
            }));
        }
        Ok((joins, admit_row_groups, share, keys))
    }

    /// The verdict for a plan leaving the queue, read off the rolling window.
    ///
    /// A plan that fits its share is admitted whole — unchanged from before this
    /// phase, and the reason no existing hit-rate floor can move down: the
    /// change is strictly additive, turning some `None` verdicts into partial
    /// `Groups` ones and never the reverse.
    ///
    /// Called while the leaving plan's own keys are still counted, so the `>= 2`
    /// test reads as "at least one plan still queued also touches this group".
    fn admit_for(&self, fits_share: bool, share: usize, keys: &[KeyedBytes]) -> Admit {
        let verdict = verdict_from_window(fits_share, keys, &self.window, share);
        if verdict.named_keys() > 0 {
            self.engine
                .cache_metrics
                .reuse_admissions
                .fetch_add(1, Ordering::Relaxed);
        }
        verdict
    }

    /// Undo one plan's contribution to the window. Exactly the increment
    /// `refill` made, from the plan's own recorded keys — a key left counted
    /// after its plan is gone would read as hot for the rest of the epoch and
    /// admit a cold tail the phase exists to keep out.
    fn release_keys(&mut self, keys: &[KeyedBytes]) {
        for (k, _) in keys {
            if let std::collections::hash_map::Entry::Occupied(mut e) = self.window.entry(*k) {
                if *e.get() <= 1 {
                    e.remove();
                } else {
                    *e.get_mut() -= 1;
                }
            }
        }
    }

    /// Block on every prefetch handle for the head plan, surfacing the first
    /// shard read error or join panic.
    fn await_head(&self, prefetches: Vec<ShardJoin>) -> Result<()> {
        let runtime = self.engine.runtime()?;
        for h in prefetches {
            match runtime.block_on(h) {
                Ok(Ok(())) => {
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
    ProcFn: Fn(&PrefetchEngine, P, Admit) -> Result<T>,
{
    type Item = Result<T>;

    fn next(&mut self) -> Option<Self::Item> {
        // Nothing has been yielded yet on this call, so waiting for the first
        // plan is the only way to make progress.
        self.refill(true);

        if let Some(InFlight {
            plan,
            prefetches,
            fits_share,
            share,
            keys,
        }) = self.in_flight.pop_front()
        {
            // Keep the queue warm during the upcoming process call. Strictly
            // non-blocking: we already hold a plan, and the generator may be
            // waiting on the batch it produces.
            self.refill(false);

            // AFTER the refill, so the verdict sees every plan the window
            // actually holds — the most information available at the only
            // moment it can be used. Taking it in `spawn_prefetches`, where the
            // pre-change verdict was taken, would ask "do two plans want this
            // group?" of a window that does not yet contain the plans it would
            // have to compare against.
            let admit = self.admit_for(fits_share, share, &keys);
            self.release_keys(&keys);

            if let Err(e) = self.await_head(prefetches) {
                return Some(Err(e));
            }
            return Some((self.process)(&self.engine, plan, admit));
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
        // Abort every in-flight shard prefetch.
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

        // Optional one-shot profile dump. Enabled by `SCX_LOADER_PROFILE=1`
        // — see `crate::budget::profiling_enabled`.
        if profiling_enabled() {
            let cm = &self.engine.cache_metrics;
            let im = &self.iter_metrics;
            let hits = cm.hits.load(Ordering::Relaxed);
            let misses = cm.misses.load(Ordering::Relaxed);
            let evictions = cm.evictions.load(Ordering::Relaxed);
            let bytes_inserted = cm.bytes_inserted.load(Ordering::Relaxed);
            let dup_waiters = cm.duplicate_waiters.load(Ordering::Relaxed);
            let rg_hits = cm.row_group_hits.load(Ordering::Relaxed);
            let rg_misses = cm.row_group_misses.load(Ordering::Relaxed);
            let rg_evictions = cm.row_group_evictions.load(Ordering::Relaxed);
            let rg_bytes_inserted = cm.row_group_bytes_inserted.load(Ordering::Relaxed);
            let spawned = im.prefetch_tasks_spawned.load(Ordering::Relaxed);
            let skip_hit = im.prefetch_skipped_cache_hit.load(Ordering::Relaxed);
            let skip_inflight = im.prefetch_skipped_in_flight.load(Ordering::Relaxed);
            let skip_block_index = im.prefetch_skipped_block_index.load(Ordering::Relaxed);
            eprintln!(
                "scx-loader PlanPrefetchIter cache_metrics: \
                 hits={hits} misses={misses} evictions={evictions} \
                 bytes_inserted={bytes_inserted} duplicate_waiters={dup_waiters} \
                 row_group_hits={rg_hits} row_group_misses={rg_misses} \
                 row_group_evictions={rg_evictions} \
                 row_group_bytes_inserted={rg_bytes_inserted} \
                 prefetch_tasks_spawned={spawned} \
                 prefetch_skipped_cache_hit={skip_hit} \
                 prefetch_skipped_in_flight={skip_inflight} \
                 prefetch_skipped_block_index={skip_block_index}"
            );
        }
    }
}

#[cfg(test)]
#[path = "plan_engine_tests.rs"]
// `pub(crate)` only so `sparse_cellset_tests` can borrow the framed fixture
// writer instead of duplicating it; every test in here stays private to the
// module by default.
pub(crate) mod tests;
