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
    let mut config = LoaderConfig::default();
    config.batch_size = 32;
    config.normalize = false;
    config.log1p = false;
    config.obs_columns = vec!["cell_id".to_string()];
    config.max_memory_mb = 128;

    let mut pipeline = TrainingPipeline::new(path, config).expect("TrainingPipeline::new");
    pipeline.start_epoch().expect("start_epoch");

    let mut n_batches = 0;
    while let Some(batch) = pipeline.next_batch() {
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
                    panic!(
                        "child process hung in TrainingPipeline iteration \
                         (>{timeout:?})"
                    );
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
