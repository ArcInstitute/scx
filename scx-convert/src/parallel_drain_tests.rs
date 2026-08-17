//! Tests for [`super::ordered_parallel_drain`].
//!
//! None of these need libhdf5. That is deliberate: `crate::pipeline` and every
//! test that drives the two real coordinators are `#[cfg(feature = "hdf5")]`,
//! and until the `Test (hdf5 features)` CI job existed none of those ran
//! anywhere. These run in the ordinary `cargo test --workspace` job, so the
//! drain's three invariants are guarded even if the hdf5 lane is unavailable
//! on some future runner.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::time::Duration;

use super::{hooks, ordered_parallel_drain, DrainFailure};

/// Standalone error type — the drain is generic over `E` precisely so its tests
/// need nothing from the hdf5-gated half of the crate.
#[derive(Debug, PartialEq, Eq)]
enum TestError {
    Drain(String),
    Item(usize),
    Panic(usize, String),
    Sink(usize),
}

impl From<DrainFailure> for TestError {
    fn from(f: DrainFailure) -> Self {
        TestError::Drain(f.0)
    }
}

const THREADS: usize = 4;
const DEPTH: usize = 2;
const CAP: usize = THREADS + DEPTH;

/// The bound, and the reason this module exists.
///
/// Item 0 blocks until every other item has been produced, which is the
/// head-of-line stall that made the ingest coordinator's `BTreeMap` grow toward
/// `n_items`. With one replacement spawned per *applied* item the buffer cannot
/// exceed `threads + queue_depth`.
///
/// Watched red by moving the `spawn_item!` in `ordered_parallel_drain` from the
/// inner apply loop back up to the receive site: peak 6 → 49.
#[test]
fn buffer_stays_within_the_window_when_item_zero_stalls() {
    const N: usize = 50;

    // Item 0 waits on this until the test releases it. Every other item
    // completes immediately, so without the bound they would all pile into the
    // reorder buffer behind item 0.
    let release = Arc::new(Barrier::new(2));
    let produced = Arc::new(AtomicUsize::new(0));

    let release_worker = Arc::clone(&release);
    let produced_worker = Arc::clone(&produced);
    let releaser = std::thread::spawn({
        let produced = Arc::clone(&produced);
        let release = Arc::clone(&release);
        move || {
            // Wait for the priming batch to finish (item 0 is blocked, so that
            // is CAP - 1 of them), then give the drain a further moment. Bounded,
            // nothing more can be produced; unbounded, every remaining item is.
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            while produced.load(Ordering::SeqCst) < CAP - 1 && std::time::Instant::now() < deadline
            {
                std::thread::sleep(Duration::from_millis(5));
            }
            std::thread::sleep(Duration::from_millis(150));
            release.wait();
        }
    });

    let applied = Mutex::new(Vec::<usize>::new());
    let result = ordered_parallel_drain(
        N,
        THREADS,
        DEPTH,
        "drain-test",
        |i| -> Result<usize, TestError> {
            if i == 0 {
                release_worker.wait();
            }
            produced_worker.fetch_add(1, Ordering::SeqCst);
            Ok(i * 10)
        },
        TestError::Panic,
        |idx, value| {
            applied.lock().unwrap().push(idx);
            assert_eq!(value, idx * 10, "item {idx} carried the wrong payload");
            Ok(())
        },
    );
    releaser.join().unwrap();
    assert!(result.is_ok(), "drain failed: {:?}", result.err());

    // Applied in index order, exactly once each.
    let applied = applied.into_inner().unwrap();
    assert_eq!(applied, (0..N).collect::<Vec<_>>());

    let buffer_peak = hooks::last_run_buffer_peak();
    assert!(
        buffer_peak > 1,
        "the stall did not produce any out-of-order arrivals (peak {buffer_peak}); \
         this test would then pass against an unbounded drain too"
    );
    assert!(
        buffer_peak <= CAP,
        "reorder-buffer peak {buffer_peak} exceeds the window {CAP}: a replacement \
         worker is being spawned per received item rather than per applied one"
    );
    assert!(
        hooks::last_run_peak() <= THREADS,
        "more worker bodies executed at once than the pool has threads"
    );
}

/// Run a drain on its own thread and fail — rather than hang — if it does not
/// finish. The drain must be off the test thread: every failure mode these
/// three tests guard against is a *block*, and a watchdog on another thread
/// cannot rescue a test thread that is parked in `rx.recv()` forever. Same
/// shape as the existing ingest deadlock tests in `convert_tests_parallel.rs`.
fn run_off_thread<F>(what: &str, body: F) -> Result<(), TestError>
where
    F: FnOnce() -> Result<(), TestError> + Send + 'static,
{
    let handle = std::thread::spawn(body);
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while !handle.is_finished() {
        assert!(
            std::time::Instant::now() < deadline,
            "ordered_parallel_drain deadlocked on {what}"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    handle.join().expect("drain thread panicked")
}

/// A worker that panics must come back as an `Err` built by `wrap_panic`, not
/// as a lost send. A lost send strands `received` below `n_items` and the drain
/// blocks forever, because the scope's own `tx` keeps the channel open.
///
/// Watched red by deleting the `catch_unwind`: the drain never returns and this
/// fails on the 30 s deadline.
#[test]
fn a_panicking_worker_returns_an_error_instead_of_deadlocking() {
    let outcome = run_off_thread("a panicking worker", || {
        ordered_parallel_drain(
            16,
            THREADS,
            1,
            "drain-panic",
            |i| -> Result<usize, TestError> {
                if i == 3 {
                    panic!("injected panic at item {i}");
                }
                Ok(i)
            },
            TestError::Panic,
            |_, _| Ok(()),
        )
        .map(|_| ())
    });

    match outcome {
        Err(TestError::Panic(3, msg)) => {
            assert!(
                msg.contains("injected panic at item 3"),
                "panic payload lost: {msg}"
            );
        }
        other => panic!("expected the panic wrapped as TestError::Panic(3, _), got {other:?}"),
    }
}

/// A worker returning `Err` propagates, and does so without deadlocking —
/// workers parked in `tx.send` on the bounded channel are released when the
/// early return drops `rx`. That is the `move` on the scope closure; without it
/// `rx` lives in the parent frame and the scope can never join them.
///
/// ⚠️ **The parameters are the test.** The first version used
/// `threads = 4, depth = 1` and item 2 as the failure, and removing `move`
/// from the scope closure did not deadlock it — four instant workers and a
/// drain that returns after two receives never leave anyone parked, so the arm
/// asserted nothing. 16 threads against a depth-1 channel, failing on the item
/// that is received first, does: fifteen workers finish immediately and
/// contend for one channel slot. Verified by removing `move` and watching this
/// hit the 30 s deadline.
#[test]
fn an_item_error_propagates_without_deadlocking() {
    const PARKING_THREADS: usize = 16;
    let outcome = run_off_thread("a worker error", || {
        ordered_parallel_drain(
            64,
            PARKING_THREADS,
            1,
            "drain-err",
            |i| -> Result<usize, TestError> {
                if i == 0 {
                    return Err(TestError::Item(i));
                }
                Ok(i)
            },
            TestError::Panic,
            |_, _| Ok(()),
        )
    });
    assert_eq!(outcome, Err(TestError::Item(0)));
}

/// The sink runs on the calling thread and its error propagates the same way,
/// releasing parked workers on the way out.
#[test]
fn a_sink_error_propagates_without_deadlocking() {
    let outcome = run_off_thread("a sink error", || {
        ordered_parallel_drain(
            64,
            THREADS,
            1,
            "drain-sink-err",
            |i| -> Result<usize, TestError> { Ok(i) },
            TestError::Panic,
            |idx, _| {
                if idx == 5 {
                    return Err(TestError::Sink(idx));
                }
                Ok(())
            },
        )
    });
    assert_eq!(outcome, Err(TestError::Sink(5)));
}

/// Degenerate inputs: no items is a no-op, and a single-threaded, depth-zero
/// request still runs (both are clamped to 1) rather than building a pool with
/// zero threads or a channel with zero capacity.
#[test]
fn empty_and_degenerate_inputs_are_handled() {
    let ran = AtomicUsize::new(0);
    let r: Result<(), TestError> = ordered_parallel_drain(
        0,
        0,
        0,
        "drain-empty",
        |i| -> Result<usize, TestError> {
            ran.fetch_add(1, Ordering::SeqCst);
            Ok(i)
        },
        TestError::Panic,
        |_, _| Ok(()),
    );
    assert!(r.is_ok());
    assert_eq!(ran.load(Ordering::SeqCst), 0, "no items means no workers");

    let applied = Mutex::new(Vec::new());
    let r: Result<(), TestError> = ordered_parallel_drain(
        3,
        0,
        0,
        "drain-degenerate",
        |i| -> Result<usize, TestError> { Ok(i) },
        TestError::Panic,
        |idx, _| {
            applied.lock().unwrap().push(idx);
            Ok(())
        },
    );
    assert!(r.is_ok());
    assert_eq!(applied.into_inner().unwrap(), vec![0, 1, 2]);
}
