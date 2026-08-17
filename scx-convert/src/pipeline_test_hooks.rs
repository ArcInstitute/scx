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

/// Running maximum of the coordinator's reorder-buffer occupancy — the
/// `BTreeMap` holding shards that have been *received* but not yet
/// *written*.
///
/// This is the quantity review §11.2 is about, and it is emphatically not
/// [`IN_FLIGHT_PEAK`]: that counter is incremented inside the spawned worker
/// body, so it measures worker bodies executing concurrently, which the rayon
/// pool bounds by `num_threads` no matter what the coordinator does. A test
/// asserting on it cannot observe an unbounded buffer.
pub static BUFFER_LEN_PEAK: AtomicUsize = AtomicUsize::new(0);

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

thread_local! {
    /// Per-thread panic-injection switch for the ingest coordinator.
    /// Unlike [`FAIL_INGEST_SHARD_AT`] (which makes a worker *send* an
    /// `Err`), this makes the worker `panic!` *before* its
    /// `tx.send(...)` — exercising the `catch_unwind` guard that turns
    /// a worker panic into a delivered error rather than a deadlock.
    /// Mutate only via [`PanicIngestShardGuard`].
    pub static PANIC_INGEST_SHARD_AT: Cell<Option<usize>> = const { Cell::new(None) };

    /// Per-thread panic-injection switch for the export coordinator
    /// (`stream_csr_to_group_at`). Same semantics as
    /// [`PANIC_INGEST_SHARD_AT`] for the SCX → h5ad direction. Mutate
    /// only via [`PanicExportShardGuard`].
    pub static PANIC_EXPORT_SHARD_AT: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Read the current thread's ingest panic-injection setting.
pub fn current_ingest_panic_shard() -> Option<usize> {
    PANIC_INGEST_SHARD_AT.with(|c| c.get())
}

/// Read the current thread's export panic-injection setting.
pub fn current_export_panic_shard() -> Option<usize> {
    PANIC_EXPORT_SHARD_AT.with(|c| c.get())
}

/// RAII guard arming the ingest panic injector for the current
/// thread. Drop restores the previous value. Same thread-scoping
/// rules as [`FailIngestShardGuard`]: create it on the thread that
/// invokes the convert.
pub struct PanicIngestShardGuard {
    prev: Option<usize>,
}

impl PanicIngestShardGuard {
    pub fn new(shard_idx: usize) -> Self {
        let prev = PANIC_INGEST_SHARD_AT.with(|c| c.replace(Some(shard_idx)));
        Self { prev }
    }
}

impl Drop for PanicIngestShardGuard {
    fn drop(&mut self) {
        let prev = self.prev;
        PANIC_INGEST_SHARD_AT.with(|c| c.set(prev));
    }
}

/// RAII guard arming the export panic injector for the current
/// thread. Drop restores the previous value.
pub struct PanicExportShardGuard {
    prev: Option<usize>,
}

impl PanicExportShardGuard {
    pub fn new(shard_idx: usize) -> Self {
        let prev = PANIC_EXPORT_SHARD_AT.with(|c| c.replace(Some(shard_idx)));
        Self { prev }
    }
}

impl Drop for PanicExportShardGuard {
    fn drop(&mut self) {
        let prev = self.prev;
        PANIC_EXPORT_SHARD_AT.with(|c| c.set(prev));
    }
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
    /// Per-thread ingest slow-worker switch: `(shard_idx, millis)`. The
    /// worker for that shard sleeps before doing any real work.
    ///
    /// Load-bearing for the reorder-buffer bound test, not a convenience. At
    /// 50 small shards over 4 threads the natural completion skew is well
    /// under the `threads + queue_depth` cap, so a buffer-occupancy assertion
    /// passes against an *unbounded* coordinator too — the same vacuum the
    /// test it replaces had, in a new costume. Delaying shard 0 is what makes
    /// every later shard pile up behind it, which is exactly the production
    /// scenario (`--group-by --reference` makes shard 0 the deliberately
    /// oversized reference shard).
    ///
    /// Mutate only via [`DelayIngestShardGuard`].
    pub static DELAY_INGEST_SHARD_AT: Cell<Option<(usize, u64)>> = const { Cell::new(None) };
}

/// Read the current thread's ingest delay-injection setting.
pub fn current_ingest_delay_shard() -> Option<(usize, u64)> {
    DELAY_INGEST_SHARD_AT.with(|c| c.get())
}

/// RAII guard arming the ingest delay injector for the current thread. Drop
/// restores the previous value. Same thread-scoping rules as
/// [`FailIngestShardGuard`].
pub struct DelayIngestShardGuard {
    prev: Option<(usize, u64)>,
}

impl DelayIngestShardGuard {
    pub fn new(shard_idx: usize, millis: u64) -> Self {
        let prev = DELAY_INGEST_SHARD_AT.with(|c| c.replace(Some((shard_idx, millis))));
        Self { prev }
    }
}

impl Drop for DelayIngestShardGuard {
    fn drop(&mut self) {
        let prev = self.prev;
        DELAY_INGEST_SHARD_AT.with(|c| c.set(prev));
    }
}

thread_local! {
    pub static LAST_RUN_PEAK: Cell<usize> = const { Cell::new(0) };
    pub static LAST_RUN_BUFFER_PEAK: Cell<usize> = const { Cell::new(0) };
}

pub fn reset_in_flight() {
    IN_FLIGHT_NOW.store(0, Ordering::SeqCst);
    IN_FLIGHT_PEAK.store(0, Ordering::SeqCst);
    BUFFER_LEN_PEAK.store(0, Ordering::SeqCst);
}

pub fn last_run_peak() -> usize {
    LAST_RUN_PEAK.with(|c| c.get())
}

pub fn set_last_run_peak(v: usize) {
    LAST_RUN_PEAK.with(|c| c.set(v));
}

/// Fold one observation of the reorder buffer's occupancy into
/// [`BUFFER_LEN_PEAK`]. Called by the coordinator immediately after inserting
/// a received item, which is the moment the buffer is at its largest.
pub fn record_buffer_len(n: usize) {
    BUFFER_LEN_PEAK.fetch_max(n, Ordering::SeqCst);
}

pub fn last_run_buffer_peak() -> usize {
    LAST_RUN_BUFFER_PEAK.with(|c| c.get())
}

pub fn set_last_run_buffer_peak(v: usize) {
    LAST_RUN_BUFFER_PEAK.with(|c| c.set(v));
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
