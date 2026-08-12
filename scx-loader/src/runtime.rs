//! A tokio runtime whose teardown is bounded no matter who drops it.
//!
//! # Why this is a newtype and not a `shutdown()` call at each owner
//!
//! The obvious shape — "whoever holds the last `Arc` calls
//! `Runtime::shutdown_timeout`" — requires each owner to *know* it is the last,
//! which in practice means `Arc::into_inner`. That is only sound if you can
//! enumerate every holder of the `Arc`, and twice in this crate we could not:
//!
//! * `PlanPrefetchIter` stores the caller's `process` closure, and
//!   `SparseCellSetLoader::iter_with_plans` builds that closure around an
//!   `Arc<SparseCellSetLoader>` — which holds an `Arc<PrefetchEngine>` of its
//!   own. Rust runs a `Drop::drop` body *before* dropping the struct's fields,
//!   so the closure was still alive at the `Arc::into_inner` call and it failed
//!   **deterministically** on the production path.
//! * `IndexPlanIter`'s prefetch tasks captured the whole loader
//!   (`spawn_blocking(move || loader.backed.read_shard_cached_arc(..))`). An
//!   already-started blocking task cannot be aborted, so it outlived the drop,
//!   `Arc::into_inner` failed, and the task itself released the final reference
//!   — from a runtime thread.
//!
//! Both were invisible to a test that constructed the ownership graph by hand,
//! because the extra owner only exists on the real path. Putting the deadline in
//! the runtime's own `Drop` removes the question: every path that releases the
//! last reference gets a bounded teardown, including ones nobody enumerated.
//!
//! The remaining obligation is narrow and local — a runtime must not be dropped
//! from one of its own threads — and it is discharged by keeping the loader out
//! of the spawned closures, so no task ever holds a reference that could be the
//! last.

use std::time::Duration;

use tokio::runtime::Runtime;

#[cfg(test)]
thread_local! {
    /// Bounded teardowns that ran on *this* thread. Thread-local rather than a
    /// global counter because `cargo test` runs tests in parallel, and the
    /// drops under test happen on the test's own thread.
    pub(crate) static BOUNDED_SHUTDOWNS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// A `Runtime` that shuts down with a deadline when it is dropped.
///
/// A plain `Runtime::drop` waits **indefinitely** for already-started
/// `spawn_blocking` work — for these loaders, a `read_shard_cached_arc` decode
/// that can be hundreds of megabytes of Pcodec.
pub(crate) struct BoundedRuntime {
    rt: Option<Runtime>,
    deadline: Duration,
}

impl BoundedRuntime {
    pub(crate) fn new(rt: Runtime, deadline: Duration) -> Self {
        Self {
            rt: Some(rt),
            deadline,
        }
    }

    /// The runtime. Always `Some` outside `Drop`.
    pub(crate) fn get(&self) -> &Runtime {
        self.rt
            .as_ref()
            .expect("BoundedRuntime is taken only by Drop")
    }
}

impl Drop for BoundedRuntime {
    fn drop(&mut self) {
        if let Some(rt) = self.rt.take() {
            #[cfg(test)]
            BOUNDED_SHUTDOWNS.with(|c| c.set(c.get() + 1));
            rt.shutdown_timeout(self.deadline);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build() -> Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap()
    }

    /// The whole point: a blocking task that overruns the deadline must be
    /// abandoned, not joined. A plain `Runtime::drop` waits for it forever.
    #[test]
    fn drop_abandons_a_task_that_overruns_the_deadline() {
        let bounded = BoundedRuntime::new(build(), Duration::from_millis(200));
        bounded
            .get()
            .handle()
            .spawn_blocking(|| std::thread::sleep(Duration::from_secs(60)));
        // `shutdown_timeout` only waits on *started* tasks.
        std::thread::sleep(Duration::from_millis(100));

        let t0 = std::time::Instant::now();
        drop(bounded);
        let elapsed = t0.elapsed();

        assert!(
            elapsed < Duration::from_secs(10),
            "drop waited {elapsed:?} — it joined the 60s task instead of \
             abandoning it at the deadline"
        );
    }

    #[test]
    fn drop_counts_exactly_one_bounded_shutdown() {
        let before = BOUNDED_SHUTDOWNS.with(|c| c.get());
        drop(BoundedRuntime::new(build(), Duration::from_millis(200)));
        assert_eq!(BOUNDED_SHUTDOWNS.with(|c| c.get()), before + 1);
    }
}
