//! Bounded ordered decode-prefetch + budgeted parallel reductions for the
//! streaming shard kernels.
//!
//! The 2.0 CPU stage profiler ([`crate::profile`]) showed that the streaming CPU
//! kernels (HVG, normalize, PCA at ≥~1M cells) are **decode-bound**:
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
//! `crossbeam_channel` + a `BTreeMap` reorder buffer keyed by **plan position** +
//! rolling-window spawn (`in_flight_cap = depth`) + per-worker `catch_unwind`
//! (a worker panic becomes a delivered `Err`, never a hung drain loop).
//!
//! # Why this lives in `scx-format-io`
//!
//! It began in `scx-accel` (Phase 2.1) and moved here in Phase 4.2, because two
//! of its three natural consumers sit *below* `scx-accel` in the crate graph and
//! so could not reach it: the backed aggregation kernels in [`crate::backed`]
//! and the GPU staging pipeline in `scx-gpu`. Everything it needs — the
//! [`ShardSource`] / [`ColumnShardSource`] traits it is generic over — is
//! defined here, so the dependency always pointed the wrong way.
//!
//! `scx_accel::prefetch` re-exports this module verbatim, so accelerator call
//! sites and both `SCX_ACCEL_*` environment knobs are unchanged.
//!
//! # Error type
//!
//! The pipeline is generic over the consumer's error enum via [`PrefetchError`]
//! (implemented here for [`ScxError`], in `scx-accel` for `AccelError`, and in
//! `scx-gpu` for `GpuError`). A trait rather than a `From<ScxError>` supertrait
//! so each crate keeps its own message wording for a shard-read failure.

#[cfg(feature = "parallel")]
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::OnceLock;

#[cfg(feature = "parallel")]
use crossbeam_channel::bounded;
use scx_sparse::{ScxCsc, ScxCsr};

use crate::error::ScxError;
use crate::shard_source::{ColumnShardSource, ShardSource};

/// Error types that a decode-prefetch consumer can return.
///
/// The pipeline itself only ever produces two kinds of failure — a shard read
/// that errored, and an internal pipeline fault (a decode worker panicked, or
/// the channel closed before every shard arrived) — so this is the minimal
/// surface a consumer error enum must provide to be usable as the `E` in
/// [`for_each_shard_ordered`] and friends.
pub trait PrefetchError: Send + 'static {
    /// Wrap a [`ShardSource::read_shard`] / [`ShardSource::read_shard_arc`]
    /// failure on `shard_idx`.
    fn from_shard_read(shard_idx: usize, err: ScxError) -> Self;

    /// A pipeline-internal failure: a decode worker panicked, or the channel
    /// closed before all shards were delivered. Neither is reachable through
    /// normal operation; both must surface rather than hang the drain loop.
    fn prefetch_internal(msg: String) -> Self;
}

impl PrefetchError for ScxError {
    fn from_shard_read(_shard_idx: usize, err: ScxError) -> Self {
        // The read errors this crate produces already name the shard
        // (`ShardIndexOutOfBounds`, `ChecksumMismatch { section }`, …), so
        // re-wrapping would only duplicate it.
        err
    }

    fn prefetch_internal(msg: String) -> Self {
        // `ScxError` has no free-form variant; `Io(io::Error::other(..))` is the
        // established spelling for a runtime string error in this crate (see
        // the CSC capability errors in `pyscx::lazy_transform::shard_source`).
        ScxError::Io(std::io::Error::other(msg))
    }
}

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
        requested.min(pool_threads())
    })
}

/// Current rayon pool width, or `1` when the `parallel` feature is off (in which
/// case every front-end takes the sequential path anyway).
#[inline]
fn pool_threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads().max(1)
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// Pure clamp for the decode-prefetch depth: the largest in-flight shard count
/// `≤ requested` whose decoded footprint (`depth × per_shard_bytes`) fits
/// `budget_bytes`, floored at 1.
///
/// Peak prefetch memory is `depth` decoded shards (the sequential loop held
/// one), so this bounds the extra RSS the pipeline can add. Callers that know a
/// per-shard byte estimate (e.g. from catalog `nnz` stats) pass it here; callers
/// without one rely on the small [`DEFAULT_PREFETCH_DEPTH`] cap instead.
pub fn clamp_prefetch_depth(requested: usize, per_shard_bytes: u64, budget_bytes: u64) -> usize {
    let requested = requested.max(1);
    let per = per_shard_bytes.max(1);
    let max_by_budget = (budget_bytes / per).max(1) as usize;
    requested.min(max_by_budget)
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
/// The `current_thread_index().is_some()` fallback defuses it by decoding
/// sequentially when invoked on a worker thread, but callers should still treat
/// "top-level only" as the contract.
///
/// **The safe nesting is the other way round**, which is what let PCA adopt this
/// after being deferred for owning inner pools: a consumer may enter a parallel
/// region — even one on a private pool — from *inside* `consume`, because the
/// live set never exceeds `depth` and the channel capacity **is** `depth`, so a
/// decode worker can always complete its `tx.send` rather than parking while the
/// calling thread is blocked. What is unsafe is calling this function from a
/// worker, not calling into a pool from the consumer.
///
/// Note that a caller who takes this path silently gets the sequential loop.
/// That is a correct outcome, but it is indistinguishable from a correct
/// prefetching one in every result — so a new consumer should carry a test that
/// observes concurrent decodes (see
/// `scx_accel::pca::cpu::tests::covariance_build_decodes_shards_concurrently`),
/// not just one that checks its numbers.
///
/// A worker that panics while decoding is caught and delivered as an `Err`, so
/// the drain loop always terminates. If `consume` returns `Err`, iteration
/// stops and the error propagates (dropping the receiver unblocks any parked
/// workers).
///
/// `S: ?Sized` so a `&dyn ShardSource + Sync` (which is how `scx-gpu`'s staging
/// sources hold their input) can be passed directly.
pub fn for_each_shard_ordered<S, F, E>(source: &S, depth: usize, consume: F) -> Result<(), E>
where
    S: ShardSource + Sync + ?Sized,
    F: FnMut(usize, Arc<ScxCsr>) -> Result<(), E>,
    E: PrefetchError,
{
    let n_shards = source.n_shards();
    let read = move |idx: usize| {
        source
            .read_shard_arc(idx)
            .map_err(|e| E::from_shard_read(idx, e))
    };
    // A source that filters rows inside `read_shard` can say which shards
    // still hold one; the rest would decode only to come back empty.
    match source.visible_shard_indices() {
        Some(indices) => for_each_ordered_selected(&indices, depth, &read, consume),
        None => for_each_ordered(n_shards, depth, &read, consume),
    }
}

/// Like [`for_each_shard_ordered`] but decodes via [`ShardSource::read_shard`]
/// (uncached) instead of `read_shard_arc`. For **single-pass** ops (e.g.
/// pseudobulk aggregation, the backed `col_*`/`row_*` aggregations) where
/// warming a caching source's LRU yields no reuse and would only add cache
/// pressure / evict a co-resident reader's entries. Same ordering + bounding
/// guarantees.
pub fn for_each_shard_ordered_uncached<S, F, E>(
    source: &S,
    depth: usize,
    consume: F,
) -> Result<(), E>
where
    S: ShardSource + Sync + ?Sized,
    F: FnMut(usize, Arc<ScxCsr>) -> Result<(), E>,
    E: PrefetchError,
{
    let n_shards = source.n_shards();
    let read = move |idx: usize| {
        source
            .read_shard(idx)
            .map(Arc::new)
            .map_err(|e| E::from_shard_read(idx, e))
    };
    // A source that filters rows inside `read_shard` can say which shards
    // still hold one; the rest would decode only to come back empty.
    match source.visible_shard_indices() {
        Some(indices) => for_each_ordered_selected(&indices, depth, &read, consume),
        None => for_each_ordered(n_shards, depth, &read, consume),
    }
}

/// [`for_each_shard_ordered_uncached`] restricted to an explicit ascending list
/// of shard indices.
///
/// The GPU CSR staging path builds its plan as a list even when that list is
/// `0..n_shards`, so that it and the CSC path run the *same* driver over the
/// same plan type. A `StagingPlan` variant meaning "all of them" would have
/// been an enum arm only one layout could ever produce — coverage that reads as
/// real and is not.
///
/// `indices` must be strictly ascending (see [`for_each_ordered_selected`]).
///
/// Unlike the two drivers above, this does **not** consult
/// [`ShardSource::visible_shard_indices`]: the caller has already decided which
/// shards to visit, and second-guessing an explicit plan is how a staging path
/// ends up skipping a shard it meant to stage. No in-tree caller both passes a
/// list and overrides that method — the masked aggregation kernels derive their
/// list from the same row set over a source that does not override it, and the
/// GPU staging plans run over sources that do not either. A future source that
/// wants both would have to intersect them itself.
pub fn for_each_shard_ordered_uncached_selected<S, F, E>(
    source: &S,
    indices: &[usize],
    depth: usize,
    consume: F,
) -> Result<(), E>
where
    S: ShardSource + Sync + ?Sized,
    F: FnMut(usize, Arc<ScxCsr>) -> Result<(), E>,
    E: PrefetchError,
{
    let read = move |idx: usize| {
        source
            .read_shard(idx)
            .map(Arc::new)
            .map_err(|e| E::from_shard_read(idx, e))
    };
    for_each_ordered_selected(indices, depth, &read, consume)
}

/// Column-major (`ColumnShardSource`) sibling of [`for_each_shard_ordered`]:
/// bounded ordered decode-prefetch over CSC shards. Same StableOrder /
/// bit-exact guarantee — used by the CSC mean/var kernels.
pub fn for_each_csc_shard_ordered<S, F, E>(source: &S, depth: usize, consume: F) -> Result<(), E>
where
    S: ColumnShardSource + Sync + ?Sized,
    F: FnMut(usize, Arc<ScxCsc>) -> Result<(), E>,
    E: PrefetchError,
{
    let n_shards = source.n_csc_shards();
    let read = move |idx: usize| {
        source
            .read_csc_shard(idx)
            .map(Arc::new)
            .map_err(|e| E::from_shard_read(idx, e))
    };
    for_each_ordered(n_shards, depth, &read, consume)
}

/// [`for_each_csc_shard_ordered`] restricted to an explicit ascending list of
/// shard indices.
///
/// The GPU CSC staging path never sweeps every shard: it iterates the shards
/// whose column range overlaps the gene chunk being processed, which at census
/// scale is one or two out of thirteen. Handing that subset here is what lets
/// it share the bounded, ordered pipeline instead of the hand-rolled
/// `sync_channel(1)` it used before — the difference is a decode depth of 4
/// rather than 1, and a `SCX_GPU_STAGING_MEMORY_BUDGET` that binds.
///
/// `indices` must be strictly ascending (see [`for_each_ordered_selected`]).
/// Duplicates are not rejected but decode the same shard twice, which is never
/// what a caller means.
pub fn for_each_csc_shard_ordered_selected<S, F, E>(
    source: &S,
    indices: &[usize],
    depth: usize,
    consume: F,
) -> Result<(), E>
where
    S: ColumnShardSource + Sync + ?Sized,
    F: FnMut(usize, Arc<ScxCsc>) -> Result<(), E>,
    E: PrefetchError,
{
    let read = move |idx: usize| {
        source
            .read_csc_shard(idx)
            .map(Arc::new)
            .map_err(|e| E::from_shard_read(idx, e))
    };
    for_each_ordered_selected(indices, depth, &read, consume)
}

/// Generic bounded ordered decode-prefetch core shared by the CSR and CSC
/// front-ends. Decodes up to `depth` shards concurrently on the rayon pool via
/// `read`, delivering to `consume` on the **calling thread in strict shard
/// order**. See [`for_each_shard_ordered`] for the ordering / fallback / panic
/// semantics.
fn for_each_ordered<T, R, F, E>(
    n_shards: usize,
    depth: usize,
    read: &R,
    consume: F,
) -> Result<(), E>
where
    T: Send + Sync + 'static,
    R: Fn(usize) -> Result<Arc<T>, E> + Sync,
    F: FnMut(usize, Arc<T>) -> Result<(), E>,
    E: PrefetchError,
{
    for_each_ordered_by(n_shards, |p| p, depth, read, consume)
}

/// [`for_each_ordered`] over an explicit, already-ascending list of shard
/// indices instead of `0..n_shards`.
///
/// The GPU CSC path iterates a **prefiltered subset** — the shards whose column
/// range overlaps a gene chunk — so the dense form cannot serve it. Rather than
/// grow a second copy of the pipeline, the core below is position-indexed and
/// this is the second `idx_of` it is given; the dense form passes the identity
/// and allocates nothing.
///
/// `indices` must be strictly ascending: "strict shard order" is defined by
/// position here, so a caller that shuffles the list gets its own order back,
/// not a sorted one. Every production caller derives it from a catalog scan
/// that is ascending by construction.
fn for_each_ordered_selected<T, R, F, E>(
    indices: &[usize],
    depth: usize,
    read: &R,
    consume: F,
) -> Result<(), E>
where
    T: Send + Sync + 'static,
    R: Fn(usize) -> Result<Arc<T>, E> + Sync,
    F: FnMut(usize, Arc<T>) -> Result<(), E>,
    E: PrefetchError,
{
    for_each_ordered_by(indices.len(), |p| indices[p], depth, read, consume)
}

/// The position-indexed core. `n_items` counts *positions*; `idx_of` maps a
/// position to the shard index handed to `read` and `consume`.
///
/// Splitting position from shard index is what lets one pipeline serve both a
/// dense `0..n` sweep and a prefiltered subset: the reorder buffer and the
/// in-flight window are keyed on position (always `0..n_items`, always
/// contiguous), while the shard index only ever appears at the two boundaries
/// where it means something.
fn for_each_ordered_by<T, R, M, F, E>(
    n_items: usize,
    idx_of: M,
    depth: usize,
    read: &R,
    mut consume: F,
) -> Result<(), E>
where
    T: Send + Sync + 'static,
    R: Fn(usize) -> Result<Arc<T>, E> + Sync,
    M: Fn(usize) -> usize + Sync,
    F: FnMut(usize, Arc<T>) -> Result<(), E>,
    E: PrefetchError,
{
    let n_shards = n_items;
    if n_shards == 0 {
        return Ok(());
    }
    let depth = depth.max(1);

    // Take the prefetch pipeline unless it can't help or would be unsafe:
    // trivial size, a single-thread pool, OR the caller is itself a rayon
    // worker. The last case is the nesting-deadlock guard (Claude review):
    // `in_place_scope` runs the drain on the calling thread, which then blocks
    // on `rx.recv()`; if that thread is a pool worker and the pool is
    // saturated, the decode spawns can never schedule.
    // `current_thread_index().is_some()` is true exactly when we're on a rayon
    // worker → decode sequentially instead.
    //
    // Without the `parallel` feature there is no rayon at all, so the pipeline
    // is compiled out entirely and this always falls through to the sequential
    // loop below.
    #[cfg(feature = "parallel")]
    if depth > 1
        && n_shards > 1
        && rayon::current_num_threads() > 1
        && rayon::current_thread_index().is_none()
    {
        return for_each_ordered_prefetched(n_shards, &idx_of, depth, read, consume);
    }
    #[cfg(not(feature = "parallel"))]
    let _ = depth;

    for pos in 0..n_shards {
        let idx = idx_of(pos);
        let shard = read(idx)?;
        consume(idx, shard)?;
    }
    Ok(())
}

/// The bounded ordered decode-prefetch pipeline itself: a bounded
/// `crossbeam_channel`, a `BTreeMap` reorder buffer keyed by **plan position**, and
/// rolling-window spawn so the live set never exceeds `depth`.
///
/// Split out of [`for_each_ordered`] so the sequential fallback needs no `cfg`
/// and this whole body disappears without the `parallel` feature. Callers must
/// already have ruled out the fallback conditions.
#[cfg(feature = "parallel")]
fn for_each_ordered_prefetched<T, R, M, F, E>(
    n_shards: usize,
    idx_of: &M,
    depth: usize,
    read: &R,
    mut consume: F,
) -> Result<(), E>
where
    T: Send + Sync + 'static,
    R: Fn(usize) -> Result<Arc<T>, E> + Sync,
    M: Fn(usize) -> usize + Sync,
    F: FnMut(usize, Arc<T>) -> Result<(), E>,
    E: PrefetchError,
{
    // Cap decoded-but-unconsumed shards at `depth` → extra peak = depth shards.
    let in_flight_cap = depth;
    let (tx, rx) = bounded::<(usize, Result<Arc<T>, E>)>(in_flight_cap);

    // `move` moves `rx` into the scope so an early `Err` return drops it and
    // unblocks workers parked in `tx.send(...)`. `consume` is moved in too and
    // called on this (the calling) thread — never on a worker — so its `&mut`
    // captures need no synchronisation. `read` is `&R` (Copy) so each spawn
    // captures the shared reference; `R: Sync` makes concurrent calls sound.
    rayon::in_place_scope(move |s| -> Result<(), E> {
        // `$pos` is a *position* in the plan; the shard index it names is
        // `idx_of($pos)`. Everything the pipeline bounds — the channel, the
        // reorder buffer, the spawn window — is keyed on position, so the two
        // may not be interchanged: for a prefiltered plan the shard indices are
        // sparse and would leave `next_idx` waiting forever on a gap.
        macro_rules! spawn_shard {
            ($scope:expr, $pos:expr) => {{
                let pos_ = $pos;
                let idx_ = idx_of(pos_);
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
                        Err(payload) => Err(E::prefetch_internal(format!(
                            "decode-prefetch worker panicked on shard {idx_}: {}",
                            panic_message(payload)
                        ))),
                    };
                    let _ = tx.send((pos_, res));
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

        let mut buffer: BTreeMap<usize, Result<Arc<T>, E>> = BTreeMap::new();
        let mut next_pos = 0usize;
        let mut received = 0usize;
        while received < n_shards {
            let (pos, res) = rx.recv().map_err(|_| {
                E::prefetch_internal(
                    "decode-prefetch channel closed before all shards arrived".into(),
                )
            })?;
            received += 1;
            buffer.insert(pos, res);
            #[cfg(test)]
            MAX_REORDER_BUFFER.with(|m| m.set(m.get().max(buffer.len())));
            while let Some(res) = buffer.remove(&next_pos) {
                let shard = res?;
                consume(idx_of(next_pos), shard)?;
                next_pos += 1;
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

/// Decode-prefetched twin of [`ShardSource::col_means_and_sum_sq`].
///
/// Same single pass, same per-column accumulation order (consumption is
/// single-threaded and in strict shard order), so the result is **bit-identical**
/// to the trait default — which stays in place as this function's oracle. The
/// only difference is that up to `depth` shards decode concurrently instead of
/// one at a time on the calling thread.
///
/// A free function rather than a trait method because the pipeline needs
/// `Self: Sync`, and putting that bound on a *provided* method would make it
/// unavailable through the `&dyn ShardSource` boundaries `scx-gpu` uses. Callers
/// that can name a `Sync` source (CPU PCA today, GPU PCA's identical serial loop
/// in `scx_gpu::gpu_pca` next) opt in here; everyone else keeps the default.
pub fn col_means_and_sum_sq_prefetched<S>(
    source: &S,
    zero_center: bool,
    depth: usize,
) -> Result<(Option<Vec<f64>>, Vec<f64>), ScxError>
where
    S: ShardSource + Sync + ?Sized,
{
    let n_vars = source.n_vars();
    let mut col_sums = vec![0.0f64; n_vars];
    let mut col_sum_sq = vec![0.0f64; n_vars];

    for_each_shard_ordered(source, depth, |_shard_idx, csr| {
        for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
            let v = val as f64;
            col_sums[col as usize] += v;
            col_sum_sq[col as usize] += v * v;
        }
        Ok::<(), ScxError>(())
    })?;

    let means = if zero_center {
        let n = source.n_obs() as f64;
        if n == 0.0 {
            Some(vec![0.0f64; n_vars])
        } else {
            Some(col_sums.iter().map(|s| s / n).collect())
        }
    } else {
        None
    };

    Ok((means, col_sum_sq))
}

/// Operation-specific **parallel** shard reduction with one accumulator per
/// worker, merged at the end. Reorders floating summation, so only reachable
/// under [`ReductionMode::ParallelTolerant`].
///
/// `init` builds a fresh zero accumulator, `fold` folds one decoded shard into
/// an accumulator, and `merge` combines two accumulators. `max_workers` bounds
/// concurrency — the caller must derate it by a memory budget (each worker
/// holds one accumulator; see [`clamp_prefetch_depth`]).
///
/// Without the `parallel` feature this degrades to a single-accumulator
/// sequential fold, which is what `ParallelTolerant` means on a pool of one.
pub fn reduce_shards_budgeted<S, T, Init, Fold, Merge, E>(
    source: &S,
    max_workers: usize,
    init: Init,
    fold: Fold,
    merge: Merge,
) -> Result<T, E>
where
    S: ShardSource + Sync + ?Sized,
    T: Send,
    Init: Fn() -> T + Sync,
    Fold: Fn(&mut T, usize, &ScxCsr) -> Result<(), E> + Sync,
    Merge: Fn(T, T) -> T + Sync,
    E: PrefetchError,
{
    let n_shards = source.n_shards();
    if n_shards == 0 {
        return Ok(init());
    }
    // Honour a row-filtering source's shard plan, like the ordered drivers do.
    // Without this the skip held on the default `StableOrder` mode and was lost
    // under `SCX_ACCEL_REDUCTION_MODE=parallel_tolerant` — measured on HVG
    // `flavor="seurat"`, which reduces through here: 1 decode on a
    // one-of-five-shard row window by default, 5 in tolerant mode.
    let plan: Vec<usize> = match source.visible_shard_indices() {
        Some(indices) => indices,
        None => (0..n_shards).collect(),
    };
    if plan.is_empty() {
        return Ok(init());
    }

    #[cfg(not(feature = "parallel"))]
    {
        let _ = (max_workers, merge);
        let mut acc = init();
        for idx in plan {
            let csr = source
                .read_shard_arc(idx)
                .map_err(|e| E::from_shard_read(idx, e))?;
            fold(&mut acc, idx, &csr)?;
        }
        Ok(acc)
    }

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        let workers = max_workers.max(1).min(rayon::current_num_threads().max(1));
        // Segment shards across at most `workers` chunks so at most `workers`
        // accumulators exist concurrently (bounds peak = workers × sizeof(T)).
        let chunk = plan.len().div_ceil(workers).max(1);

        plan.into_par_iter()
            .with_min_len(chunk)
            .try_fold(&init, |mut acc: T, idx| -> Result<T, E> {
                let csr = source
                    .read_shard_arc(idx)
                    .map_err(|e| E::from_shard_read(idx, e))?;
                fold(&mut acc, idx, &csr)?;
                Ok(acc)
            })
            .try_reduce(&init, |a, b| Ok(merge(a, b)))
    }
}

/// Accumulate over all shards, dispatching on the resolved [`ReductionMode`].
///
/// `StableOrder` (default) runs [`for_each_shard_ordered`] into a single
/// accumulator (bit-exact, ordered, decode-prefetched). `ParallelTolerant` runs
/// [`reduce_shards_budgeted`] (parallel per-worker accumulators, tolerance
/// only). Use this only for reductions **independent of the global row offset**
/// (per-column / per-gene sums); offset-dependent kernels (batched HVG,
/// pseudobulk) call [`for_each_shard_ordered`] directly.
pub fn accumulate_shards<S, T, Init, Fold, Merge, E>(
    source: &S,
    max_workers: usize,
    init: Init,
    fold: Fold,
    merge: Merge,
) -> Result<T, E>
where
    S: ShardSource + Sync + ?Sized,
    T: Send,
    Init: Fn() -> T + Sync,
    Fold: Fn(&mut T, usize, &ScxCsr) -> Result<(), E> + Sync,
    Merge: Fn(T, T) -> T + Sync,
    E: PrefetchError,
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
#[cfg(feature = "parallel")]
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
