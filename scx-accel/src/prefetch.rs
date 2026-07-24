//! Bounded ordered decode-prefetch + budgeted parallel reductions for the
//! streaming accelerator kernels.
//!
//! The 2.0 CPU stage profiler (`scx_format_io::profile`) showed that the
//! streaming CPU kernels (HVG, normalize, PCA at ≥~1M cells) are **decode-bound**:
//! each `for shard_idx in 0..n_shards { let csr = source.read_shard(idx)?; … }`
//! loop decodes one shard on the calling thread, then reduces it, with no
//! overlap between decode and reduction and no concurrency across shards. This
//! module provides two explicit primitives to close that gap — deliberately
//! **not** a blanket `par_iter` fold/reduce, which would (a) reorder floating
//! summation (not bit-reproducible) and (b) allocate one accumulator per worker
//! (memory × workers, dangerous for `vars²`/`batches×vars` accumulators).
//!
//! 1. [`for_each_shard_ordered`] — a **bounded, ordered decode-prefetch**
//!    pipeline. Up to `depth` shards decode concurrently on the rayon pool; the
//!    consumer closure runs on the **calling thread in strict shard order**.
//!    Because consumption is single-threaded and in shard order, the reduction
//!    sees exactly the sequential accumulation order → **bit-identical** to the
//!    old loop, while decode now overlaps with reduction and runs concurrently
//!    across shards. This is the [`ReductionMode::StableOrder`] default and is
//!    safe for every kernel (row-disjoint scatter and f64 summation alike),
//!    since a `global_row`/`cell_offset` cursor still advances in order.
//! 2. [`reduce_shards_budgeted`] — an **operation-specific parallel reduction**
//!    with one accumulator per worker, merged at the end. Reorders summation, so
//!    it is only reached under [`ReductionMode::ParallelTolerant`] (opt-in). The
//!    worker count is caller-derated by a memory budget so peak stays bounded.
//!
//! [`accumulate_shards`] dispatches between the two by the resolved
//! [`ReductionMode`]; kernels whose reduction depends on the global row offset
//! (batched HVG, pseudobulk) call [`for_each_shard_ordered`] directly (ordered
//! delivery makes the cursor valid) rather than going through the parallel path.
//!
//! The prefetch pipeline mirrors the ingest coordinator
//! (`scx-convert::pipeline::streaming_writer_coordinator_parallel`): a bounded
//! `crossbeam_channel` + a `BTreeMap` reorder buffer keyed by shard index +
//! rolling-window spawn (`in_flight_cap = depth`) + per-worker `catch_unwind`
//! (a worker panic becomes a delivered `Err`, never a hung drain loop).

use std::collections::BTreeMap;
use std::sync::OnceLock;

use crossbeam_channel::bounded;
use scx_format_io::{ColumnShardSource, ShardSource};
use scx_sparse::{ScxCsc, ScxCsr};
use std::sync::Arc;

use crate::error::{AccelError, Result};

/// Default number of shards decoded concurrently / held in flight. Bounds the
/// prefetch pipeline's extra peak memory to `depth` decoded shards (the
/// sequential loop held one). Kept small on purpose; override with
/// `SCX_ACCEL_PREFETCH_DEPTH`.
pub const DEFAULT_PREFETCH_DEPTH: usize = 4;

/// Reduction strategy for the streaming accelerator kernels.
///
/// The default is [`StableOrder`](ReductionMode::StableOrder): bit-reproducible
/// (identical to the pre-2.1 sequential loop) with decode overlapped across
/// shards. [`ParallelTolerant`](ReductionMode::ParallelTolerant) additionally
/// parallelizes the *reduction*, changing floating summation order — results
/// then match only within f64 tolerance. Selected process-wide via the
/// `SCX_ACCEL_REDUCTION_MODE` environment variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionMode {
    /// Ordered decode-prefetch, single-threaded in-order reduction. Bit-exact.
    StableOrder,
    /// Parallel per-worker reduction, merged at the end. Tolerance-only.
    ParallelTolerant,
}

/// Resolve the process-wide reduction mode from `SCX_ACCEL_REDUCTION_MODE`.
///
/// Recognised values (case-insensitive): `stable` / `stable_order` / `ordered`
/// → [`ReductionMode::StableOrder`]; `parallel` / `tolerant` /
/// `parallel_tolerant` → [`ReductionMode::ParallelTolerant`]. Anything else (or
/// unset) → [`ReductionMode::StableOrder`]. Cached on first read.
pub fn reduction_mode() -> ReductionMode {
    static M: OnceLock<ReductionMode> = OnceLock::new();
    *M.get_or_init(|| {
        match std::env::var("SCX_ACCEL_REDUCTION_MODE")
            .ok()
            .map(|s| s.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("parallel" | "tolerant" | "parallel_tolerant") => ReductionMode::ParallelTolerant,
            _ => ReductionMode::StableOrder,
        }
    })
}

/// Resolve the decode-prefetch depth (max shards decoded-but-unconsumed).
///
/// `SCX_ACCEL_PREFETCH_DEPTH` overrides; `0`/`1` disables prefetch (the caller
/// falls back to the sequential loop). Otherwise defaults to
/// [`DEFAULT_PREFETCH_DEPTH`], never exceeding the current rayon pool size (no
/// point queueing more concurrent decodes than there are worker threads).
pub fn prefetch_depth() -> usize {
    static D: OnceLock<usize> = OnceLock::new();
    *D.get_or_init(|| {
        let requested = std::env::var("SCX_ACCEL_PREFETCH_DEPTH")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_PREFETCH_DEPTH);
        let threads = rayon::current_num_threads().max(1);
        requested.min(threads)
    })
}

// Reorder-buffer high-water mark, updated on the calling thread inside the drain
// loop. Thread-local so concurrent test threads don't stomp each other's
// measurement (the drain runs on the calling thread, which is the test thread).
// Used by the head-of-line-stall bound test.
#[cfg(test)]
thread_local! {
    pub(crate) static MAX_REORDER_BUFFER: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Bounded, ordered decode-prefetch over `source`'s shards.
///
/// Decodes up to `depth` shards concurrently on the rayon pool and invokes
/// `consume(shard_idx, csr)` on the **calling thread in strict ascending shard
/// order**. Accumulation performed inside `consume` therefore sees the exact
/// sequential order → bit-reproducible. Uses [`ShardSource::read_shard_arc`], so
/// a caching source (e.g. `BackedCsrReader`) serves/populates its LRU; use
/// [`for_each_shard_ordered_uncached`] for a single-pass op that must not warm
/// the cache.
///
/// Falls back to a plain sequential loop (no channel/threads) when `depth <= 1`,
/// there is at most one shard, the current rayon pool has a single thread, or
/// the caller is itself a rayon worker (see below).
///
/// # Must not be called from within a rayon parallel region
///
/// The drain runs on the calling thread and blocks on the channel while decode
/// tasks run on other pool workers. If the caller is a saturated pool worker
/// (nested `par_iter`/`scope`/`install`), those spawns can't schedule → deadlock.
/// This is why PCA prefetch (which owns inner rayon pools) is deferred. The
/// `current_thread_index().is_some()` fallback defuses it by decoding
/// sequentially when invoked on a worker thread, but callers should still treat
/// "top-level only" as the contract.
///
/// A worker that panics while decoding is caught and delivered as an `Err`, so
/// the drain loop always terminates. If `consume` returns `Err`, iteration
/// stops and the error propagates (dropping the receiver unblocks any parked
/// workers).
pub fn for_each_shard_ordered<S, F>(source: &S, depth: usize, consume: F) -> Result<()>
where
    S: ShardSource + Sync,
    F: FnMut(usize, Arc<ScxCsr>) -> Result<()>,
{
    let n_shards = source.n_shards();
    let read = move |idx: usize| source.read_shard_arc(idx).map_err(AccelError::from);
    for_each_ordered(n_shards, depth, &read, consume)
}

/// Like [`for_each_shard_ordered`] but decodes via [`ShardSource::read_shard`]
/// (uncached) instead of `read_shard_arc`. For **single-pass** ops (e.g.
/// pseudobulk aggregation) where warming a caching source's LRU yields no reuse
/// and would only add cache pressure / evict a co-resident reader's entries
/// (Cursor/Claude/Antigravity review). Same ordering + bounding guarantees.
pub fn for_each_shard_ordered_uncached<S, F>(source: &S, depth: usize, consume: F) -> Result<()>
where
    S: ShardSource + Sync,
    F: FnMut(usize, Arc<ScxCsr>) -> Result<()>,
{
    let n_shards = source.n_shards();
    let read = move |idx: usize| {
        source
            .read_shard(idx)
            .map(Arc::new)
            .map_err(AccelError::from)
    };
    for_each_ordered(n_shards, depth, &read, consume)
}

/// Column-major (`ColumnShardSource`) sibling of [`for_each_shard_ordered`]:
/// bounded ordered decode-prefetch over CSC shards. Same StableOrder /
/// bit-exact guarantee — used by the CSC mean/var kernels.
pub fn for_each_csc_shard_ordered<S, F>(source: &S, depth: usize, consume: F) -> Result<()>
where
    S: ColumnShardSource + Sync,
    F: FnMut(usize, Arc<ScxCsc>) -> Result<()>,
{
    let n_shards = source.n_csc_shards();
    let read = move |idx: usize| {
        source
            .read_csc_shard(idx)
            .map(Arc::new)
            .map_err(AccelError::from)
    };
    for_each_ordered(n_shards, depth, &read, consume)
}

/// Generic bounded ordered decode-prefetch core shared by the CSR and CSC
/// front-ends. Decodes up to `depth` shards concurrently on the rayon pool via
/// `read`, delivering to `consume` on the **calling thread in strict shard
/// order**. See [`for_each_shard_ordered`] for the ordering / fallback / panic
/// semantics.
fn for_each_ordered<T, R, F>(n_shards: usize, depth: usize, read: &R, mut consume: F) -> Result<()>
where
    T: Send + Sync + 'static,
    R: Fn(usize) -> Result<Arc<T>> + Sync,
    F: FnMut(usize, Arc<T>) -> Result<()>,
{
    if n_shards == 0 {
        return Ok(());
    }

    let depth = depth.max(1);
    // Sequential fallback (no channel/threads) when prefetch can't help or would
    // be unsafe: trivial size, a single-thread pool, OR the caller is itself a
    // rayon worker. The last case is the nesting-deadlock guard (Claude review):
    // `in_place_scope` runs the drain on the calling thread, which then blocks on
    // `rx.recv()`; if that thread is a pool worker and the pool is saturated, the
    // decode spawns can never schedule. `current_thread_index().is_some()` is
    // true exactly when we're on a rayon worker → decode sequentially instead.
    if depth <= 1
        || n_shards == 1
        || rayon::current_num_threads() <= 1
        || rayon::current_thread_index().is_some()
    {
        for idx in 0..n_shards {
            let shard = read(idx)?;
            consume(idx, shard)?;
        }
        return Ok(());
    }

    // Cap decoded-but-unconsumed shards at `depth` → extra peak = depth shards.
    let in_flight_cap = depth;
    let (tx, rx) = bounded::<(usize, Result<Arc<T>>)>(in_flight_cap);

    // `move` moves `rx` into the scope so an early `Err` return drops it and
    // unblocks workers parked in `tx.send(...)`. `consume` is moved in too and
    // called on this (the calling) thread — never on a worker — so its `&mut`
    // captures need no synchronisation. `read` is `&R` (Copy) so each spawn
    // captures the shared reference; `R: Sync` makes concurrent calls sound.
    rayon::in_place_scope(move |s| -> Result<()> {
        macro_rules! spawn_shard {
            ($scope:expr, $idx:expr) => {{
                let idx_ = $idx;
                let tx = tx.clone();
                let read = read;
                $scope.spawn(move |_| {
                    // A caught panic becomes a delivered `Err`; a panic that
                    // skipped `tx.send` would leave the drain counter short of
                    // `n_shards` forever (the scope's original `tx` keeps `rx`
                    // open). `AssertUnwindSafe` is sound: the worker only calls
                    // the shared `read` and builds an owned `Arc<T>`.
                    let outcome =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| read(idx_)));
                    let res = match outcome {
                        Ok(r) => r,
                        Err(payload) => Err(AccelError::InvalidInput(format!(
                            "decode-prefetch worker panicked on shard {idx_}: {}",
                            panic_message(payload)
                        ))),
                    };
                    let _ = tx.send((idx_, res));
                });
            }};
        }

        // Prime with up to `depth` tasks, then spawn one more **per ordered
        // consume** (not per receive). This keeps the total live set —
        // in-flight + in-channel + in the reorder buffer — at ≤ `depth` even
        // under a head-of-line stall: if shard 0 is slow, no shard beyond the
        // primed `depth` is ever spawned until shard 0 drains, so the BTreeMap
        // can hold at most `depth - 1` out-of-order shards (Cursor review — the
        // earlier spawn-on-receive let the buffer grow to ~n_shards and broke
        // the RSS bound).
        let mut next_to_spawn = 0usize;
        let prime = in_flight_cap.min(n_shards);
        while next_to_spawn < prime {
            spawn_shard!(s, next_to_spawn);
            next_to_spawn += 1;
        }

        let mut buffer: BTreeMap<usize, Result<Arc<T>>> = BTreeMap::new();
        let mut next_idx = 0usize;
        let mut received = 0usize;
        while received < n_shards {
            let (idx, res) = rx.recv().map_err(|_| {
                AccelError::InvalidInput(
                    "decode-prefetch channel closed before all shards arrived".into(),
                )
            })?;
            received += 1;
            buffer.insert(idx, res);
            #[cfg(test)]
            MAX_REORDER_BUFFER.with(|m| m.set(m.get().max(buffer.len())));
            while let Some(res) = buffer.remove(&next_idx) {
                let shard = res?;
                consume(next_idx, shard)?;
                next_idx += 1;
                // Spawn the next shard only as one drains, bounding the live set.
                if next_to_spawn < n_shards {
                    spawn_shard!(s, next_to_spawn);
                    next_to_spawn += 1;
                }
            }
        }
        Ok(())
    })
}

/// Operation-specific **parallel** shard reduction with one accumulator per
/// worker, merged at the end. Reorders floating summation, so only reachable
/// under [`ReductionMode::ParallelTolerant`].
///
/// `init` builds a fresh zero accumulator, `fold` folds one decoded shard into
/// an accumulator, and `merge` combines two accumulators. `max_workers` bounds
/// concurrency — the caller must derate it by a memory budget (each worker
/// holds one accumulator; see [`crate::mem_budget`]).
pub fn reduce_shards_budgeted<S, T, Init, Fold, Merge>(
    source: &S,
    max_workers: usize,
    init: Init,
    fold: Fold,
    merge: Merge,
) -> Result<T>
where
    S: ShardSource + Sync,
    T: Send,
    Init: Fn() -> T + Sync,
    Fold: Fn(&mut T, usize, &ScxCsr) -> Result<()> + Sync,
    Merge: Fn(T, T) -> T + Sync,
{
    use rayon::prelude::*;

    let n_shards = source.n_shards();
    if n_shards == 0 {
        return Ok(init());
    }

    let workers = max_workers.max(1).min(rayon::current_num_threads().max(1));
    // Segment shards across at most `workers` chunks so at most `workers`
    // accumulators exist concurrently (bounds peak = workers × sizeof(T)).
    let chunk = n_shards.div_ceil(workers).max(1);

    (0..n_shards)
        .into_par_iter()
        .with_min_len(chunk)
        .try_fold(&init, |mut acc: T, idx| -> Result<T> {
            let csr = source.read_shard_arc(idx).map_err(AccelError::from)?;
            fold(&mut acc, idx, &csr)?;
            Ok(acc)
        })
        .try_reduce(&init, |a, b| Ok(merge(a, b)))
}

/// Accumulate over all shards, dispatching on the resolved [`ReductionMode`].
///
/// `StableOrder` (default) runs [`for_each_shard_ordered`] into a single
/// accumulator (bit-exact, ordered, decode-prefetched). `ParallelTolerant` runs
/// [`reduce_shards_budgeted`] (parallel per-worker accumulators, tolerance
/// only). Use this only for reductions **independent of the global row offset**
/// (per-column / per-gene sums); offset-dependent kernels (batched HVG,
/// pseudobulk) call [`for_each_shard_ordered`] directly.
pub fn accumulate_shards<S, T, Init, Fold, Merge>(
    source: &S,
    max_workers: usize,
    init: Init,
    fold: Fold,
    merge: Merge,
) -> Result<T>
where
    S: ShardSource + Sync,
    T: Send,
    Init: Fn() -> T + Sync,
    Fold: Fn(&mut T, usize, &ScxCsr) -> Result<()> + Sync,
    Merge: Fn(T, T) -> T + Sync,
{
    match reduction_mode() {
        ReductionMode::ParallelTolerant => {
            reduce_shards_budgeted(source, max_workers, init, fold, merge)
        }
        ReductionMode::StableOrder => {
            let mut acc = init();
            for_each_shard_ordered(source, prefetch_depth(), |idx, csr| {
                fold(&mut acc, idx, &csr)
            })?;
            Ok(acc)
        }
    }
}

/// Extract a human-readable message from a `catch_unwind` payload. Mirrors the
/// idiom in `scx-convert::pipeline::panic_message`.
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_else(|| "unknown panic payload".to_string())
}

#[cfg(test)]
#[path = "prefetch_tests.rs"]
mod tests;
