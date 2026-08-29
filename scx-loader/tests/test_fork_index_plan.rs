//! ORG-9.10-6 / §X-5 bullet 2 — a forked child driving a plan that spans
//! multiple cold shards.
//!
//! **Its own binary, deliberately.** This test primes rayon's *global* pool in
//! the parent, which is the precondition for §9.1's hang. Cargo runs one test
//! binary's `#[test]`s on shared threads, so a sibling that also `fork()`s
//! would inherit those live-registry worker threads from *this* test rather
//! than from its own fixture — turning a pinned property into a cross-test
//! race. `test_decode_pool_determinism.rs` is split from the rest of the suite
//! for the same reason (a process-global it mutates).

#![cfg(target_os = "linux")]

mod common;

use std::time::{Duration, Instant};

use common::write_multi_shard_fixture;
use nix::sys::signal::{kill, Signal};
use nix::sys::wait::{waitpid, WaitPidFlag, WaitStatus};
use nix::unistd::{fork, ForkResult, Pid};
use scx_loader::LoaderConfig;

/// Hard deadline for the child. If the deadlock reproduces, the child hangs in
/// `warm_shards`' `par_iter`; we surface that as a failure, not a CI hang.
const CHILD_TIMEOUT: Duration = Duration::from_secs(20);
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Wait for `child` to exit, killing it with SIGKILL after `timeout`.
fn wait_with_timeout(child: Pid, timeout: Duration) -> WaitStatus {
    let deadline = Instant::now() + timeout;
    loop {
        match waitpid(child, Some(WaitPidFlag::WNOHANG)) {
            Ok(WaitStatus::StillAlive) => {
                if Instant::now() >= deadline {
                    let _ = kill(child, Signal::SIGKILL);
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

/// Rows in four different shards of the fixture below (50 rows per shard), as
/// `(pert, ctrl)` pairs.
const SPANNING_PLAN: [(u64, u64); 4] = [(0, 70), (10, 140), (20, 190), (30, 80)];

/// Drive `SPANNING_PLAN` once through a fresh loader and report
/// `(full_shard_groups, prefetch_tasks_spawned)`.
///
/// Every loader argument below is load-bearing for the premise, not tuning:
/// * `lookahead = 0` — with prefetch on, `IndexPlanIter` warms every touched
///   shard through `tokio::spawn_blocking` first and `warm_shards` then sees no
///   misses at all;
/// * `cache_shards = 8` (`>= 2`) — `warm_shards` short-circuits to a sequential
///   loop at `cache_shards <= 1`;
/// * `set_scatter_block_index(false)` — a block-index-eligible group never
///   enters `full_shards`, so it never reaches `warm_shards` either.
///
/// These are the same conditions `pyscx/tests/test_fork_safety.py`'s
/// `test_gather_premise_holds_in_parent` documents, transcribed to Rust.
///
/// The returned pair is what makes the premise checkable rather than assumed.
/// `full_shard_groups >= 2` alone is not enough: with prefetch on, the shards
/// are warmed by `spawn_blocking`, `warm_shards`' `filter_misses` then finds
/// nothing to do and returns before the `par_iter` — yet the gather still
/// records two full-shard groups. Pairing it with `prefetch_tasks_spawned == 0`
/// closes that hole: no shard was pre-warmed, so `warm_shards` is the only
/// thing that can have decoded them, and it saw >= 2 misses.
fn drive_spanning_plan(path: &std::path::Path) -> (u64, u64) {
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

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
    let loader = Arc::new(loader);

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
            // No printing from the child — see the sibling binary
            // `test_fork_deadlock.rs` for why an `eprintln!` after `fork()` from
            // a multithreaded parent can park until CHILD_TIMEOUT. The exit code
            // distinguishes all three outcomes the parent cares about.
            let result = std::panic::catch_unwind(|| drive_spanning_plan(&path));
            let exit_code = match result {
                Ok((groups, prefetched)) if groups >= 2 && prefetched == 0 => 0,
                Ok(_) => 2,
                Err(_) => 1,
            };
            unsafe { nix::libc::_exit(exit_code) };
        }
        ForkResult::Parent { child } => match wait_with_timeout(child, CHILD_TIMEOUT) {
            WaitStatus::Exited(_, 0) => { /* success */ }
            WaitStatus::Exited(_, 2) => {
                panic!("child took the sequential warm path — the fixture no longer spans shards")
            }
            WaitStatus::Exited(_, 1) => panic!("child panicked driving the spanning plan"),
            WaitStatus::Exited(_, code) => panic!("child exited non-zero ({code})"),
            WaitStatus::Signaled(_, sig, _) => panic!("child killed by signal {sig:?}"),
            other => panic!("unexpected wait status: {other:?}"),
        },
    }
}
