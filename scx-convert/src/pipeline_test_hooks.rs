//! Test-only fault, panic and delay injection for the two streaming
//! coordinators' worker closures.
//!
//! The counters that used to live here — in-flight workers, reorder-buffer
//! occupancy, the run-serialisation mutex and the per-thread peak pins — moved
//! to [`crate::parallel_drain::hooks`] when the two coordinators were unified:
//! they measure the drain, not either coordinator, and the export direction was
//! never instrumented at all. What is left is per-coordinator, because each
//! switch is read inside that coordinator's own worker closure — the ingest
//! ones before it calls `encode_one_shard_worker`, the export one before
//! `read_shard_payload`.
//!
//! Every switch here is a thread-local rather than a global atomic, so a
//! concurrently scheduled test on another thread never observes another test's
//! configuration. The coordinator copies the value at entry on its calling
//! thread and the worker closure reads the copy.
use std::cell::Cell;

thread_local! {
    /// Per-thread fault-injection switch read by the parallel
    /// ingest coordinator (`streaming_writer_coordinator_parallel`)
    /// at entry. Its worker closure returns a synthetic `Err` for the
    /// configured shard index instead of encoding it. Used by the
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
    /// Unlike [`FAIL_INGEST_SHARD_AT`] (which makes the worker return an
    /// `Err`), this makes it `panic!` — exercising the `catch_unwind` in
    /// [`crate::parallel_drain::ordered_parallel_drain`] that turns a worker
    /// panic into a delivered error rather than a lost send and a hung drain.
    /// Mutate only via [`PanicIngestShardGuard`].
    pub static PANIC_INGEST_SHARD_AT: Cell<Option<usize>> = const { Cell::new(None) };

    /// Per-thread panic-injection switch for the export coordinator
    /// (`stream_csr_to_group_at`). Same semantics as
    /// [`PANIC_INGEST_SHARD_AT`] for the SCX → h5ad direction. Mutate
    /// only via [`PanicExportShardGuard`].
    pub static PANIC_EXPORT_SHARD_AT: Cell<Option<usize>> = const { Cell::new(None) };
}

thread_local! {
    /// Per-thread count of `stream_raw_at` entries on a file that has a
    /// `.raw`.
    ///
    /// Exists because the *only* runtime difference between the streaming and
    /// eager `/raw` writers is which one ran: they emit byte-identical
    /// `/raw/X`, so no assertion on the OUTPUT can tell them apart, and
    /// `ReaderDebugCounts` cannot either from outside — those are
    /// per-`ScxReader` and `write_scx_to_h5ad_streaming` opens its own.
    ///
    /// Deliberately one-sided: only the streaming path is instrumented, so
    /// `h5ad/write.rs` stays byte-for-byte untouched and remains an
    /// independent oracle to diff the streamed output against. A call site
    /// that reverts to the eager writer leaves this at 0.
    pub static RAW_EXPORT_STREAMED: Cell<u32> = const { Cell::new(0) };
}

/// Record that the streaming `/raw` writer ran on this thread.
/// `stream_raw_at` calls this under `#[cfg(test)]`; no production cost.
pub fn note_raw_export_streamed() {
    RAW_EXPORT_STREAMED.with(|c| c.set(c.get().saturating_add(1)));
}

/// Read and reset this thread's streamed-raw counter. Reset-on-read so a
/// test needs no setup call and cannot inherit a sibling's count.
/// `stream_raw_at` runs on the calling thread (only shard decode fans out),
/// so the count lands on the thread that drove the export.
pub fn take_raw_export_streamed() -> u32 {
    RAW_EXPORT_STREAMED.with(|c| c.replace(0))
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
