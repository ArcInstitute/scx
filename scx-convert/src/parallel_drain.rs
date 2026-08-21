//! One ordered parallel drain, shared by the ingest and export streaming
//! coordinators.
//!
//! Both directions have the same shape: fan `n_items` independent units of work
//! across a rayon pool, hand results back over a bounded channel, and apply them
//! **in index order** on the calling thread — the writer side is single-threaded
//! in both (`ScxWriter` on ingest; libhdf5, which holds its own global lock, on
//! export).
//!
//! It was written twice, and the two copies had already diverged on the one line
//! that decides whether the reorder buffer is bounded — ingest spawned its
//! replacement worker per item *received*, export per item *applied*, and only
//! the second bounds anything. This module is that line, once.
//!
//! # Invariants this owns, so a caller cannot re-lose them
//!
//! * **Outstanding work is capped** at `threads + queue_depth`: prime that many,
//!   then spawn exactly one replacement per *applied* item. Spawning per
//!   *received* item instead caps `spawned - received`, which costs nothing —
//!   the buffer holds `received - applied`.
//! * **A worker panic is delivered as an `Err`.** A panic that skipped the send
//!   would strand `received` below `n_items` forever (the scope's own `tx` keeps
//!   `rx` open) and the drain would block for good.
//! * **`rx` is moved into the scope closure**, so an early `return Err(..)` drops
//!   it while unwinding and releases workers parked in `tx.send` on the bounded
//!   channel. Without the `move`, `rx` lives in the parent frame and the scope
//!   can never join them.
//!
//! # Why it is generic over the error, and not gated on `hdf5`
//!
//! `crate::pipeline` — and with it `ConvertError` — is `#[cfg(feature = "hdf5")]`,
//! and so is every test that exercises the two coordinators. Binding the drain to
//! that error would have put its tests in the same place: compiled by CI, run by
//! nothing. Generic over `E` it needs no libhdf5, so
//! `tests::buffer_stays_within_the_window_when_item_zero_stalls` and its
//! siblings run in the ordinary `cargo test --workspace` job.
//!
//! (Plain backticks, not an intra-doc link: the test module is `#[cfg(test)]`,
//! so the link is unresolvable in a normal `cargo doc` run — a warning
//! `cargo clippy --all-targets -- -D warnings` does not lint.)

use std::collections::BTreeMap;

/// A failure of the drain machinery itself — building the pool, or the channel
/// closing before every item arrived — as opposed to a failure of any one item.
///
/// Callers convert it into their own error type; that `From` bound is the only
/// thing [`ordered_parallel_drain`] needs to know about `E`.
pub(crate) struct DrainFailure(pub String);

/// Extract a human-readable message from a `catch_unwind` panic payload.
/// Mirrors the idiom in `pyscx/src/accel/pca.rs`: most panics carry a `String`
/// or `&str`; anything else falls back to a placeholder.
///
/// Moved here verbatim from what is now `pipeline/coordinator.rs`, where both
/// coordinators already
/// shared it — the one piece of this machinery that had been factored out.
pub(crate) fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_else(|| "unknown panic payload".to_string())
}

/// Run `worker(i)` for `i` in `0..n_items` across a rayon pool of `threads`,
/// applying results through `sink` in ascending index order on the calling
/// thread.
///
/// `wrap_panic(i, message)` builds the error for a worker that panicked; it
/// exists because the two callers wrap per-item failures in different envelopes
/// (`ConvertError::ShardRead` with an ingest source name vs. an export label)
/// and flattening that would lose a real difference.
///
/// Returns on the first error from any worker or from `sink`, dropping the
/// receiver so parked workers are released.
pub(crate) fn ordered_parallel_drain<T, E, W, P, S>(
    n_items: usize,
    threads: usize,
    queue_depth: usize,
    thread_name_prefix: &str,
    worker: W,
    wrap_panic: P,
    mut sink: S,
) -> Result<(), E>
where
    T: Send,
    E: From<DrainFailure> + Send,
    W: Fn(usize) -> Result<T, E> + Sync,
    P: Fn(usize, String) -> E + Sync,
    S: FnMut(usize, T) -> Result<(), E>,
{
    use crossbeam_channel::bounded;
    use rayon::ThreadPoolBuilder;

    if n_items == 0 {
        return Ok(());
    }

    let threads = threads.max(1);
    let queue_depth = queue_depth.max(1);
    let outstanding_cap = threads.saturating_add(queue_depth);

    let name_prefix = thread_name_prefix.to_string();
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(move |i| format!("{name_prefix}-{i}"))
        .build()
        .map_err(|e| {
            E::from(DrainFailure(format!(
                "failed to build rayon pool with {threads} threads: {e}"
            )))
        })?;

    let (tx, rx) = bounded::<(usize, Result<T, E>)>(queue_depth);

    // Shadow with references so the `move` closures below copy a `&W` / `&P`
    // instead of moving the values. `W: Sync` is what makes `&W` `Send`.
    let worker = &worker;
    let wrap_panic = &wrap_panic;

    // Serialize drains across the test binary so the counters below are
    // observable race-free, and reset them for this run. No effect in
    // production — the whole block is `#[cfg(test)]`.
    #[cfg(test)]
    let _serial = {
        let lock = hooks::SERIALIZE.lock().unwrap_or_else(|p| p.into_inner());
        hooks::reset();
        lock
    };

    // `move` is load-bearing: see the module docs.
    pool.in_place_scope(move |s| -> Result<(), E> {
        // A macro rather than a closure: `rayon::Scope`'s lifetime is
        // invariant, and a closure taking `&Scope<'_>` would re-borrow the
        // captured references for a shorter lifetime than `'scope`. Inlining
        // the spawn body keeps every borrow on the scope's own lifetime.
        macro_rules! spawn_item {
            ($scope:expr, $idx:expr) => {{
                let idx_ = $idx;
                let tx = tx.clone();
                $scope.spawn(move |_| {
                    #[cfg(test)]
                    let _in_flight = hooks::InFlightGuard::new();
                    let outcome =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| worker(idx_)));
                    let delivered = match outcome {
                        Ok(r) => r,
                        Err(payload) => Err(wrap_panic(idx_, panic_message(payload))),
                    };
                    // Send failure means the drain already returned and dropped
                    // `rx`; the error it returned is the one that matters.
                    let _ = tx.send((idx_, delivered));
                });
            }};
        }

        let mut next_to_spawn: usize = 0;
        let prime = outstanding_cap.min(n_items);
        while next_to_spawn < prime {
            spawn_item!(s, next_to_spawn);
            next_to_spawn += 1;
        }

        let mut buffer: BTreeMap<usize, T> = BTreeMap::new();
        let mut next_idx: usize = 0;
        let mut received: usize = 0;

        while received < n_items {
            let (idx, result) = rx.recv().map_err(|_| {
                E::from(DrainFailure(format!(
                    "parallel worker channel closed before all items arrived \
                     ({thread_name_prefix})"
                )))
            })?;
            received += 1;
            let item = result?;
            buffer.insert(idx, item);
            // Immediately after the insert is when the buffer is largest, and
            // its occupancy is the quantity the window is supposed to bound.
            #[cfg(test)]
            hooks::record_buffer_len(buffer.len());

            while let Some(item) = buffer.remove(&next_idx) {
                sink(next_idx, item)?;
                next_idx += 1;
                // One replacement per *applied* item. This is the bound.
                if next_to_spawn < n_items {
                    spawn_item!(s, next_to_spawn);
                    next_to_spawn += 1;
                }
            }
        }
        Ok(())
    })?;

    // Pin the peaks to this thread before releasing the serialize lock, so a
    // test reading them after the call sees this run's numbers.
    #[cfg(test)]
    {
        use std::sync::atomic::Ordering;
        hooks::set_last_run_peak(hooks::IN_FLIGHT_PEAK.load(Ordering::SeqCst));
        hooks::set_last_run_buffer_peak(hooks::BUFFER_LEN_PEAK.load(Ordering::SeqCst));
        drop(_serial);
    }

    Ok(())
}

/// Test-only instrumentation owned by the drain, because the quantities are
/// the drain's.
///
/// `IN_FLIGHT_PEAK` counts worker bodies executing concurrently — rayon bounds
/// that by the pool size on its own, so it is a liveness signal, not a bound on
/// anything the drain decides. `BUFFER_LEN_PEAK` is the reorder buffer's
/// occupancy, which is the quantity the rolling window exists to bound.
/// Confusing the two is what made the original regression test unable to fail:
/// it asserted `4 <= 6` on a counter rayon pins at the pool size.
#[cfg(test)]
pub(crate) mod hooks {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    pub static IN_FLIGHT_NOW: AtomicUsize = AtomicUsize::new(0);
    pub static IN_FLIGHT_PEAK: AtomicUsize = AtomicUsize::new(0);
    pub static BUFFER_LEN_PEAK: AtomicUsize = AtomicUsize::new(0);
    pub static SERIALIZE: Mutex<()> = Mutex::new(());

    thread_local! {
        pub static LAST_RUN_PEAK: Cell<usize> = const { Cell::new(0) };
        pub static LAST_RUN_BUFFER_PEAK: Cell<usize> = const { Cell::new(0) };
    }

    pub fn reset() {
        IN_FLIGHT_NOW.store(0, Ordering::SeqCst);
        IN_FLIGHT_PEAK.store(0, Ordering::SeqCst);
        BUFFER_LEN_PEAK.store(0, Ordering::SeqCst);
    }

    pub fn record_buffer_len(n: usize) {
        BUFFER_LEN_PEAK.fetch_max(n, Ordering::SeqCst);
    }

    pub fn last_run_peak() -> usize {
        LAST_RUN_PEAK.with(|c| c.get())
    }

    pub fn set_last_run_peak(v: usize) {
        LAST_RUN_PEAK.with(|c| c.set(v));
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
}

#[cfg(test)]
#[path = "parallel_drain_tests.rs"]
mod tests;
