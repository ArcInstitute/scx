//! Bounded ordered decode-prefetch + budgeted parallel reductions.
//!
//! **The implementation moved to [`scx_format_io::prefetch`] in Phase 4.2.** It
//! was written here in Phase 2.1, but two of its three natural consumers — the
//! backed aggregation kernels in `scx-format-io` and the GPU staging pipeline in
//! `scx-gpu` — sit *below* this crate in the dependency graph and so could not
//! reach it. Everything the pipeline is generic over ([`ShardSource`],
//! [`ColumnShardSource`]) is defined in `scx-format-io` anyway.
//!
//! This module keeps the accelerator-facing surface intact:
//!
//! * [`AccelError`]'s [`PrefetchError`] impl, which preserves this crate's error
//!   wording (a shard read stays [`AccelError::Scx`]; a pipeline fault stays
//!   [`AccelError::InvalidInput`]).
//! * Thin wrappers that **pin the generic error parameter to [`AccelError`]**.
//!   The relocated pipeline is generic over the consumer's error enum, and a
//!   consume closure ending in a bare `Ok(())` gives inference nothing to work
//!   with. Pinning here means the eight accelerator call sites — and both
//!   `SCX_ACCEL_PREFETCH_DEPTH` / `SCX_ACCEL_REDUCTION_MODE` — are unchanged.

use std::sync::Arc;

pub use scx_format_io::prefetch::{
    clamp_prefetch_depth, prefetch_depth, reduction_mode, PrefetchError, ReductionMode,
    DEFAULT_PREFETCH_DEPTH,
};
use scx_format_io::{ColumnShardSource, ShardSource};
use scx_sparse::{ScxCsc, ScxCsr};

use crate::error::{AccelError, Result};

impl PrefetchError for AccelError {
    /// A shard read failure is an `ScxError`; wrap it in the transparent
    /// [`AccelError::Scx`] arm exactly as the pre-4.2 `.map_err(AccelError::from)`
    /// did, so error text is unchanged.
    fn from_shard_read(_shard_idx: usize, err: scx_format_io::ScxError) -> Self {
        AccelError::Scx(err)
    }

    /// Pipeline-internal faults keep their pre-4.2 `InvalidInput` classification
    /// (a decode-worker panic is not a file-format problem, and the bindings map
    /// `InvalidInput` to a Python `ValueError`).
    fn prefetch_internal(msg: String) -> Self {
        AccelError::InvalidInput(msg)
    }
}

/// Bounded, ordered decode-prefetch over `source`'s shards, erroring as
/// [`AccelError`]. See [`scx_format_io::prefetch::for_each_shard_ordered`] for
/// the ordering, bounding, rayon-nesting and panic semantics.
pub fn for_each_shard_ordered<S, F>(source: &S, depth: usize, consume: F) -> Result<()>
where
    S: ShardSource + Sync + ?Sized,
    F: FnMut(usize, Arc<ScxCsr>) -> Result<()>,
{
    scx_format_io::prefetch::for_each_shard_ordered(source, depth, consume)
}

/// Uncached sibling of [`for_each_shard_ordered`] for single-pass ops. See
/// [`scx_format_io::prefetch::for_each_shard_ordered_uncached`].
pub fn for_each_shard_ordered_uncached<S, F>(source: &S, depth: usize, consume: F) -> Result<()>
where
    S: ShardSource + Sync + ?Sized,
    F: FnMut(usize, Arc<ScxCsr>) -> Result<()>,
{
    scx_format_io::prefetch::for_each_shard_ordered_uncached(source, depth, consume)
}

/// Column-major sibling of [`for_each_shard_ordered`]. See
/// [`scx_format_io::prefetch::for_each_csc_shard_ordered`].
pub fn for_each_csc_shard_ordered<S, F>(source: &S, depth: usize, consume: F) -> Result<()>
where
    S: ColumnShardSource + Sync + ?Sized,
    F: FnMut(usize, Arc<ScxCsc>) -> Result<()>,
{
    scx_format_io::prefetch::for_each_csc_shard_ordered(source, depth, consume)
}

/// Parallel per-worker shard reduction (tolerance-only). See
/// [`scx_format_io::prefetch::reduce_shards_budgeted`].
pub fn reduce_shards_budgeted<S, T, Init, Fold, Merge>(
    source: &S,
    max_workers: usize,
    init: Init,
    fold: Fold,
    merge: Merge,
) -> Result<T>
where
    S: ShardSource + Sync + ?Sized,
    T: Send,
    Init: Fn() -> T + Sync,
    Fold: Fn(&mut T, usize, &ScxCsr) -> Result<()> + Sync,
    Merge: Fn(T, T) -> T + Sync,
{
    scx_format_io::prefetch::reduce_shards_budgeted(source, max_workers, init, fold, merge)
}

/// Accumulate over all shards, dispatching on the resolved [`ReductionMode`].
/// See [`scx_format_io::prefetch::accumulate_shards`].
pub fn accumulate_shards<S, T, Init, Fold, Merge>(
    source: &S,
    max_workers: usize,
    init: Init,
    fold: Fold,
    merge: Merge,
) -> Result<T>
where
    S: ShardSource + Sync + ?Sized,
    T: Send,
    Init: Fn() -> T + Sync,
    Fold: Fn(&mut T, usize, &ScxCsr) -> Result<()> + Sync,
    Merge: Fn(T, T) -> T + Sync,
{
    scx_format_io::prefetch::accumulate_shards(source, max_workers, init, fold, merge)
}
