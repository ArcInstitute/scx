//! Pure-Rust fork reproducer for the historical `pyscx.TrainingDataset`
//! fork-mode deadlock.
//!
//! The hypothesis under investigation is: does a forked child that
//! constructs and iterates a fresh `TrainingPipeline` deadlock at the
//! Rust layer alone — independent of Python, PyO3, multiprocessing, or
//! PyTorch's DataLoader machinery? If yes, the root cause is in the
//! tokio-runtime / decode-thread / channel lifecycle. If no, the root
//! cause is upstream of `TrainingPipeline` (e.g. PyTorch
//! DataLoader-specific).
//!
//! This test is gated on Linux (`cfg(target_os = "linux")`) since
//! `nix::unistd::fork` semantics are POSIX-with-Linux-specific glibc
//! interactions. Fork hazards in this codebase are pinned to glibc
//! malloc arena state, which is Linux-specific.
//!
//! On `RUST_LOG=trace`, the structured tracing instrumentation added in
//! Phase 1.4 (`pipeline.rs`) prints span entry/exit and intermediate
//! milestones (tokio runtime constructed, I/O stage spawned, decode stage
//! spawned, first batch received). Run with:
//!
//! ```sh
//! RUST_LOG=trace cargo test -p scx-loader --test test_fork_deadlock -- --nocapture
//! ```

#![cfg(target_os = "linux")]

mod common;

use std::time::{Duration, Instant};

use common::write_multi_shard_fixture;
use nix::sys::signal::{kill, Signal};
use nix::sys::wait::{waitpid, WaitPidFlag, WaitStatus};
use nix::unistd::{fork, ForkResult, Pid};
use scx_loader::{LoaderConfig, TrainingPipeline};

/// Fixture sized to span multiple shards so the I/O → Decode pipeline has
/// to actually pump data, but small enough to keep the test fast.
const N_OBS: usize = 200;
const N_VARS: usize = 24;
const N_SHARDS: usize = 4;

/// Hard deadline for the child process. If the deadlock reproduces, the
/// child will hang in `start_epoch` / `next_batch`; we want to surface that
/// as a test failure instead of a CI hang.
const CHILD_TIMEOUT: Duration = Duration::from_secs(20);

/// Small pause between waitpid polls. Keeps the loop CPU-cheap without
/// adding meaningful latency to the success path.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

fn init_tracing_once() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // Off by default — only emit if RUST_LOG is set. `try_init()` is
        // idempotent across the test binary.
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
            )
            .with_target(true)
            .with_thread_ids(true)
            .with_thread_names(true)
            .try_init();
    });
}

/// Drive one full epoch through a freshly constructed `TrainingPipeline`.
/// Returns the number of batches produced. Used inside the forked child.
fn run_one_epoch(path: &std::path::Path) -> usize {
    let config = LoaderConfig {
        batch_size: 32,
        normalize: false,
        log1p: false,
        obs_columns: vec!["cell_id".to_string()],
        max_memory_mb: 128,
        ..Default::default()
    };

    let mut pipeline = TrainingPipeline::new(path, config).expect("TrainingPipeline::new");
    pipeline.start_epoch().expect("start_epoch");

    let mut n_batches = 0;
    while let Some(batch) = pipeline.next_batch().expect("next_batch") {
        n_batches += 1;
        // Touch the data so the optimizer can't elide the read.
        let _ = batch.n_rows();
    }
    n_batches
}

/// Wait for `child` to exit, killing it with SIGKILL after `timeout` and
/// surfacing the timeout as a test failure. Returns the exit status on
/// successful completion.
fn wait_with_timeout(child: Pid, timeout: Duration) -> WaitStatus {
    let deadline = Instant::now() + timeout;
    loop {
        match waitpid(child, Some(WaitPidFlag::WNOHANG)) {
            Ok(WaitStatus::StillAlive) => {
                if Instant::now() >= deadline {
                    eprintln!(
                        "[test_fork_deadlock] child {child} did not exit within {timeout:?} \
                         — sending SIGKILL"
                    );
                    let _ = kill(child, Signal::SIGKILL);
                    // Reap so we don't leave a zombie.
                    let _ = waitpid(child, None);
                    panic!("child process hung (>{timeout:?})");
                }
                std::thread::sleep(POLL_INTERVAL);
            }
            Ok(status) => return status,
            Err(e) => panic!("waitpid failed: {e}"),
        }
    }
}

/// Acceptance: fork a child, construct a fresh `TrainingPipeline`
/// inside it, run one epoch to completion, exit cleanly.
///
/// **Expected outcome on a green build:** child exits with status 0 and
/// produces `ceil(N_OBS / batch_size)` batches.
///
/// **Expected outcome if the Rust layer reproduces the deadlock:**
/// `wait_with_timeout` panics after `CHILD_TIMEOUT` because the child is
/// stuck in `start_epoch` (tokio runtime construction in the child) or
/// `next_batch` (I/O / decode / channel deadlock).
#[test]
fn fork_construct_and_iterate_one_epoch() {
    init_tracing_once();

    let dir = tempfile::tempdir().expect("tempdir");
    let path = write_multi_shard_fixture(&dir.path().join("fixture.scx"), N_OBS, N_VARS, N_SHARDS);

    // SAFETY: the parent does no further work between `fork()` and
    // `waitpid` other than the timeout poll loop; the child branch immediately
    // exits via `_exit`. No async-signal-unsafe code runs in the child after
    // the fork before the sole `run_one_epoch` call.
    match unsafe { fork() }.expect("fork") {
        ForkResult::Child => {
            // Child branch: do all the work in a closure that catches
            // panics so we can exit with a meaningful status code instead
            // of triggering a Rust unwind through fork-inherited state.
            let result = std::panic::catch_unwind(|| run_one_epoch(&path));
            let exit_code = match result {
                Ok(n_batches) => {
                    eprintln!(
                        "[child {}] produced {} batches",
                        std::process::id(),
                        n_batches
                    );
                    if n_batches == 0 {
                        2
                    } else {
                        0
                    }
                }
                Err(_) => {
                    eprintln!("[child {}] panicked in run_one_epoch", std::process::id());
                    1
                }
            };
            // Use `_exit` rather than `std::process::exit` to skip atexit
            // handlers — we don't want fork-inherited Drop impls (e.g. the
            // parent's mmap'd reader, if any) to run from the child.
            unsafe { nix::libc::_exit(exit_code) };
        }
        ForkResult::Parent { child } => {
            let status = wait_with_timeout(child, CHILD_TIMEOUT);
            match status {
                WaitStatus::Exited(_, 0) => { /* success */ }
                WaitStatus::Exited(_, code) => {
                    panic!("child exited non-zero ({code}); see stderr for child diagnostics");
                }
                WaitStatus::Signaled(_, sig, _) => {
                    panic!("child killed by signal {sig:?}");
                }
                other => {
                    panic!("unexpected wait status: {other:?}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ORG-9.10-6 / §X-5 bullet 2 — the fork fixture that reaches the hang path.
// ---------------------------------------------------------------------------

/// Rows in four different shards of the fixture below (50 rows per shard), as
/// `(pert, ctrl)` pairs.
const SPANNING_PLAN: [(u64, u64); 4] = [(0, 70), (10, 140), (20, 190), (30, 80)];

/// Build the `IndexPlanLoader` used on both sides of the fork.
///
/// Every argument here is load-bearing for the premise, not tuning:
/// * `lookahead = 0` — with prefetch on, `IndexPlanIter` warms every touched
///   shard through `tokio::spawn_blocking` first and `warm_shards` then sees no
///   misses at all;
/// * `cache_shards = 8` (`>= 2`) — `warm_shards` short-circuits to a sequential
///   loop at `cache_shards <= 1`;
/// * `set_scatter_block_index(false)` — a block-index-eligible group never
///   enters `full_shards`, so it never reaches `warm_shards` either.
///
/// These are the same three conditions `pyscx/tests/test_fork_safety.py`'s
/// `test_gather_premise_holds_in_parent` documents, transcribed to Rust.
fn open_spanning_loader(path: &std::path::Path) -> scx_loader::IndexPlanLoader {
    let config = LoaderConfig {
        normalize: false,
        log1p: false,
        obs_columns: vec!["cell_id".to_string()],
        max_memory_mb: 1024,
        ..Default::default()
    };
    let mut loader = scx_loader::IndexPlanLoader::new(
        path, config, /*cache_shards=*/ 8, /*sort_by_shard=*/ true,
        /*lookahead=*/ 0, /*max_plan_size=*/ 16384,
    )
    .expect("IndexPlanLoader::new");
    loader.set_scatter_block_index(false);
    loader
}

/// Drive `SPANNING_PLAN` once and report `(full_shard_groups, prefetch_tasks_spawned)`.
///
/// The pair is what makes the premise checkable rather than assumed.
/// `full_shard_groups >= 2` alone is not enough: with prefetch on, the shards
/// are warmed by `spawn_blocking` first, `warm_shards`' `filter_misses` then
/// finds nothing to do and returns before the `par_iter` — yet the gather still
/// records two full-shard groups. Pairing it with `prefetch_tasks_spawned == 0`
/// closes that hole: no shard was pre-warmed, so `warm_shards` is the only
/// thing that can have decoded them, and it saw >= 2 misses.
fn drive_spanning_plan(path: &std::path::Path) -> (u64, u64) {
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    let loader = Arc::new(open_spanning_loader(path));
    let metrics = loader.cache_metrics();
    let mut it = loader.iter_with_plans(std::iter::once(Ok(SPANNING_PLAN.to_vec())), 0);
    let iter_metrics = it.iter_metrics();
    let batch = it.next().expect("one plan, one batch").expect("gather");
    assert!(it.next().is_none(), "one plan, one batch");
    assert_eq!(batch.pairs.len(), SPANNING_PLAN.len());
    drop(it);
    (
        metrics.full_shard_groups.load(Ordering::Relaxed),
        iter_metrics.prefetch_tasks_spawned.load(Ordering::Relaxed),
    )
}

/// Acceptance: a forked child that builds a fresh `IndexPlanLoader` and drives
/// a plan spanning **≥ 2 cold shards** completes, even though the parent has
/// already initialised rayon's *global* registry.
///
/// This is §9.1's hang, and until now nothing could observe it from Rust.
/// `fork_construct_and_iterate_one_epoch` above drives `TrainingPipeline`,
/// which never calls `warm_shards` at all; the Python fork fixture that does
/// was, at the time of the review, a 16-cell single-shard file, so
/// `warm_shards` short-circuited on `misses.len() == 1` and the assertion was
/// unreachable (§X-5, bullet 2).
///
/// The parent half is the guard: it proves, in-process where the counters are
/// readable, that the plan really does drive `warm_shards` down its **parallel**
/// arm. Without it a future fixture change would make the child half vacuous and
/// nothing else would say so.
///
/// **Expected outcome if the fork guard regresses** (i.e. `BackedCsrReader`'s
/// per-reader `cpu_pool` stops being installed, or stops being rebuilt
/// post-fork): the child parks forever in rayon's `LockLatch::wait_and_reset`
/// and `wait_with_timeout` panics on `CHILD_TIMEOUT`.
#[test]
fn fork_index_plan_child_survives_a_plan_spanning_cold_shards() {
    init_tracing_once();

    let dir = tempfile::tempdir().expect("tempdir");
    // 200 rows / 4 shards = 50 rows per shard, written unframed by
    // `write_csr_shard` — so no group is block-index eligible and all four
    // touched shards go through `warm_shards`.
    let path = write_multi_shard_fixture(&dir.path().join("spanning.scx"), 200, 24, 4);

    // Make the parent's global rayon registry live. This is the precondition
    // for the hang: `fork()` copies the registry as a data structure but not
    // its worker threads, so a child dispatching against the inherited copy
    // waits on latches nothing will ever release.
    let primed: usize = rayon::join(|| 1usize, || 2usize).0 + 1;
    assert_eq!(primed, 2, "global rayon pool primed in the parent");

    // Premise, checked in the parent where the counters are readable.
    let (full_shard_groups, prefetched) = drive_spanning_plan(&path);
    assert_eq!(
        prefetched, 0,
        "premise: lookahead must be 0, or the prefetch warms every touched shard \
         and `warm_shards` never reaches its parallel arm"
    );
    assert!(
        full_shard_groups >= 2,
        "premise: the plan must produce >= 2 full-shard groups for warm_shards \
         to dispatch in parallel, got {full_shard_groups}"
    );

    // SAFETY: as in `fork_construct_and_iterate_one_epoch` above — the parent
    // only polls `waitpid`, and the child branch runs one closure and `_exit`s.
    match unsafe { fork() }.expect("fork") {
        ForkResult::Child => {
            let result = std::panic::catch_unwind(|| drive_spanning_plan(&path));
            let exit_code = match result {
                Ok((groups, prefetched)) => {
                    eprintln!(
                        "[child {}] drove the spanning plan; full_shard_groups={groups}, \
                         prefetch_tasks_spawned={prefetched}",
                        std::process::id()
                    );
                    if groups >= 2 && prefetched == 0 {
                        0
                    } else {
                        2
                    }
                }
                Err(_) => {
                    eprintln!(
                        "[child {}] panicked driving the spanning plan",
                        std::process::id()
                    );
                    1
                }
            };
            unsafe { nix::libc::_exit(exit_code) };
        }
        ForkResult::Parent { child } => match wait_with_timeout(child, CHILD_TIMEOUT) {
            WaitStatus::Exited(_, 0) => { /* success */ }
            WaitStatus::Exited(_, 2) => {
                panic!("child took the sequential warm path — the fixture no longer spans shards")
            }
            WaitStatus::Exited(_, code) => {
                panic!("child exited non-zero ({code}); see stderr for child diagnostics")
            }
            WaitStatus::Signaled(_, sig, _) => panic!("child killed by signal {sig:?}"),
            other => panic!("unexpected wait status: {other:?}"),
        },
    }
}
