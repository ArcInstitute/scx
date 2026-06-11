//! Test-only instrumentation for the parallel coordinator.
//!
//! `IN_FLIGHT_NOW` tracks worker tasks currently executing (one
//! entry per running `encode_one_shard_worker`); `IN_FLIGHT_PEAK`
//! is the running maximum across the most recent run. The
//! coordinator acquires `SERIALIZE` for the duration of the
//! parallel scope to ensure exactly one parallel coordinator run
//! is in flight at a time across all tests in the binary, then
//! captures `IN_FLIGHT_PEAK` into the calling thread's
//! `LAST_RUN_PEAK` before releasing the lock. Tests read
//! `LAST_RUN_PEAK` after the streaming call returns; the
//! thread-local pin makes the read race-free without requiring
//! tests themselves to hold the global lock.
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

pub static IN_FLIGHT_NOW: AtomicUsize = AtomicUsize::new(0);
pub static IN_FLIGHT_PEAK: AtomicUsize = AtomicUsize::new(0);
pub static SERIALIZE: Mutex<()> = Mutex::new(());

thread_local! {
    /// Per-thread fault-injection switch read by the parallel
    /// ingest coordinator (`streaming_writer_coordinator_parallel`)
    /// at entry. Workers fail synthetically on the configured shard
    /// index instead of running the real shard worker. Used by the
    /// deadlock regression test to force a mid-stream worker failure
    /// without corrupting the source h5ad on disk.
    ///
    /// Thread-local (not a global atomic) so a concurrently
    /// scheduled non-fault test running on a different thread never
    /// observes another test's fault config. The coordinator copies
    /// the value at entry on its calling thread and propagates it
    /// to the spawned rayon workers; each coordinator invocation
    /// captures its own value.
    ///
    /// Mutate only via [`FailIngestShardGuard`]; the guard restores
    /// the previous value on drop so the hook can't leak across
    /// tests, even on panic mid-flight.
    pub static FAIL_INGEST_SHARD_AT: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Read the current thread's ingest fault-injection setting.
/// Coordinator captures this at entry on its calling thread, then
/// propagates to spawned workers via the closure capture.
pub fn current_ingest_fault_shard() -> Option<usize> {
    FAIL_INGEST_SHARD_AT.with(|c| c.get())
}

/// RAII guard that arms the ingest fault injector for the
/// current thread. Drop restores the previous value — keeps the
/// hook from leaking when the test panics mid-flight. The fault is
/// thread-scoped, so tests that spawn the convert call onto a
/// helper thread must create the guard *on that helper thread*
/// (typically inside the `std::thread::spawn` closure body).
pub struct FailIngestShardGuard {
    prev: Option<usize>,
}

impl FailIngestShardGuard {
    pub fn new(shard_idx: usize) -> Self {
        let prev = FAIL_INGEST_SHARD_AT.with(|c| c.replace(Some(shard_idx)));
        Self { prev }
    }
}

impl Drop for FailIngestShardGuard {
    fn drop(&mut self) {
        let prev = self.prev;
        FAIL_INGEST_SHARD_AT.with(|c| c.set(prev));
    }
}

thread_local! {
    pub static LAST_RUN_PEAK: Cell<usize> = const { Cell::new(0) };
}

pub fn reset_in_flight() {
    IN_FLIGHT_NOW.store(0, Ordering::SeqCst);
    IN_FLIGHT_PEAK.store(0, Ordering::SeqCst);
}

pub fn last_run_peak() -> usize {
    LAST_RUN_PEAK.with(|c| c.get())
}

pub fn set_last_run_peak(v: usize) {
    LAST_RUN_PEAK.with(|c| c.set(v));
}

pub struct InFlightGuard;

impl InFlightGuard {
    pub fn new() -> Self {
        let now = IN_FLIGHT_NOW.fetch_add(1, Ordering::SeqCst) + 1;
        IN_FLIGHT_PEAK.fetch_max(now, Ordering::SeqCst);
        Self
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        IN_FLIGHT_NOW.fetch_sub(1, Ordering::SeqCst);
    }
}
