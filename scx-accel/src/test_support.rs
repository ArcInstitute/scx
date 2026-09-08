//! Shared test doubles for the streaming kernels (`#[cfg(test)]` only).
//!
//! [`GaugedSource`] lives here rather than in one kernel's test module because
//! two families of streaming consumer now need the same instrument: PCA, which
//! owns inner rayon pools and so could mis-nest the decode-prefetch pipeline,
//! and the DE kernels, which adopted it after years of a hand-rolled
//! `for shard_idx in 0..n_shards`. A second copy would have been a second thing
//! to keep honest.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread::ThreadId;

use scx_format_io::ShardSource;
use scx_sparse::ScxCsr;

/// A `ShardSource` that records **which thread** decoded each shard, **which
/// shards** were decoded, and optionally hands the drivers a visible-shard plan.
///
/// The thread record is the anti-trap instrument, and the reason it exists is
/// specific: `for_each_ordered` **silently** falls back to a sequential loop
/// when the caller is a rayon worker, when the pool has one thread, or when the
/// depth is 1. A mis-nested wiring therefore produces no speedup and no error —
/// the change could land, pass every correctness test, and do nothing at all.
///
/// **Thread identity, not observed overlap.** An earlier version asserted a
/// maximum-in-flight count of >= 2, which is a *timing* property: it held on a
/// 24-core box 20 runs out of 20 and failed on a 2-core CI runner, where
/// libtest's own parallelism saturates the pool and the spawned decodes run one
/// at a time. "Did the pipeline engage" is structural — when it does,
/// `read_shard` runs on a rayon worker; when it declines, on the calling thread
/// — so that is what the tests assert. `max_live` is still recorded, but only to
/// make a failure message informative.
///
/// The decoded-index record and the plan are what let a *skip* be asserted:
/// under a row projection a source answers
/// [`visible_shard_indices`](ShardSource::visible_shard_indices) and the drivers
/// are supposed to read nothing else. Without the record, "it skipped" and "it
/// decoded everything and the empty shards contributed nothing" are the same
/// observation.
pub(crate) struct GaugedSource {
    shards: Vec<ScxCsr>,
    n_obs: usize,
    n_vars: usize,
    plan: Option<Vec<usize>>,
    live: AtomicUsize,
    max_live: AtomicUsize,
    decode_threads: Mutex<HashSet<ThreadId>>,
    decoded: Mutex<Vec<usize>>,
}

impl GaugedSource {
    pub(crate) fn new(shards: Vec<ScxCsr>, n_obs: usize, n_vars: usize) -> Self {
        Self {
            shards,
            n_obs,
            n_vars,
            plan: None,
            live: AtomicUsize::new(0),
            max_live: AtomicUsize::new(0),
            decode_threads: Mutex::new(HashSet::new()),
            decoded: Mutex::new(Vec::new()),
        }
    }

    /// Answer `visible_shard_indices` with `plan`, as a row-filtering source
    /// does. `n_obs` is set to the rows those shards actually carry, since a
    /// projected source reports its *visible* row count.
    pub(crate) fn with_plan(mut self, plan: Vec<usize>) -> Self {
        self.n_obs = plan.iter().map(|&i| self.shards[i].n_rows()).sum();
        self.plan = Some(plan);
        self
    }

    /// True when at least one shard decoded somewhere other than `caller`.
    pub(crate) fn decoded_off_thread(&self, caller: ThreadId) -> bool {
        self.decode_threads
            .lock()
            .unwrap()
            .iter()
            .any(|t| *t != caller)
    }

    pub(crate) fn decode_thread_count(&self) -> usize {
        self.decode_threads.lock().unwrap().len()
    }

    pub(crate) fn max_concurrent_decodes(&self) -> usize {
        self.max_live.load(Ordering::SeqCst)
    }

    /// Shard indices decoded so far, ascending and deduplicated. A multi-pass
    /// kernel reads each shard once per pass, so the *set* is the question a
    /// skip test asks; [`Self::decode_count`] is the per-pass one.
    pub(crate) fn decoded_shards(&self) -> Vec<usize> {
        let mut v = self.decoded.lock().unwrap().clone();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Total `read_shard` calls — `passes × visited shards` for a multi-pass
    /// kernel.
    pub(crate) fn decode_count(&self) -> usize {
        self.decoded.lock().unwrap().len()
    }

    pub(crate) fn reset(&self) {
        self.max_live.store(0, Ordering::SeqCst);
        self.decode_threads.lock().unwrap().clear();
        self.decoded.lock().unwrap().clear();
    }
}

/// Assert the pipeline engaged: some shard decoded off the calling thread.
pub(crate) fn assert_prefetch_engaged(src: &GaugedSource, what: &str) {
    let me = std::thread::current().id();
    assert!(
        src.decoded_off_thread(me),
        "{what}: every shard decoded on the calling thread — the prefetch \
         pipeline declined to engage (threads seen: {}, max in flight: {})",
        src.decode_thread_count(),
        src.max_concurrent_decodes()
    );
}

/// Skip rather than fail where the pipeline is *designed* not to engage: a
/// single-thread pool takes the sequential fallback by construction.
pub(crate) fn pool_can_prefetch() -> bool {
    rayon::current_num_threads() > 1
}

impl ShardSource for GaugedSource {
    fn n_shards(&self) -> usize {
        self.shards.len()
    }
    fn n_obs(&self) -> usize {
        self.n_obs
    }
    fn n_vars(&self) -> usize {
        self.n_vars
    }
    fn visible_shard_indices(&self) -> Option<Vec<usize>> {
        self.plan.clone()
    }
    fn read_shard(&self, shard_idx: usize) -> scx_format_io::Result<ScxCsr> {
        self.decode_threads
            .lock()
            .unwrap()
            .insert(std::thread::current().id());
        self.decoded.lock().unwrap().push(shard_idx);
        let now = self.live.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_live.fetch_max(now, Ordering::SeqCst);
        // Kept short: nothing asserts on overlap now, so this only widens the
        // window in which `max_live` can observe some.
        std::thread::sleep(std::time::Duration::from_millis(1));
        let out = self.shards[shard_idx].clone();
        self.live.fetch_sub(1, Ordering::SeqCst);
        Ok(out)
    }
}
