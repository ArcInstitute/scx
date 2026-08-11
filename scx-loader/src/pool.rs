//! The loader's private CPU thread pool.
//!
//! # Why this exists
//!
//! Rayon's *global* registry is a process-global static. `fork()` duplicates it
//! as a **data structure** but not its worker threads, so a `par_*` dispatched
//! from a forked child parks forever in `LockLatch::wait_and_reset`: no error,
//! no batch, a training job that simply stops. Under
//! `DataLoader(num_workers > 0, start_method="fork")` the parent has almost
//! always initialised that registry already — `pyscx.from_anndata` alone is
//! enough — so every child inherits the dead one.
//!
//! `TrainingPipeline` avoids this with a per-pipeline pool built inside
//! `start_epoch` (see `decode_stage.rs`'s pool-ownership note). This module is
//! the same idea for everything else in the crate: one pool per process, handed
//! to `BackedCsrReader::set_cpu_pool` and used via `install` at the crate's own
//! parallel dispatch sites.
//!
//! # Why it is keyed on the PID
//!
//! A plain `OnceLock` would reproduce the bug one level down. A parent that
//! touches *any* loader path before forking would fill it, and the child would
//! then inherit a private pool whose worker threads do not exist — identical
//! symptom, new owner. The `python.rs` PID checks catch a *dataset* reused
//! across a fork; they cannot catch a dataset freshly constructed in the child
//! drawing on a stale process-global pool. So the pool is stored with the PID
//! that built it and rebuilt whenever `std::process::id()` disagrees.
//!
//! Dropping the stale inherited pool in the child is safe and does not need a
//! `mem::forget`: `ThreadPool::drop` calls `Registry::terminate`, which sets
//! per-worker latches and returns — it never joins (rayon-core 1.13.0,
//! `registry.rs:594-600`). **Re-check that on a rayon major bump**; if
//! `terminate` ever starts joining, this drop would block forever on threads
//! that no longer exist.

use std::sync::{Arc, Mutex, OnceLock};

/// Hard upper bound on loader rayon worker threads. On many-core hosts the
/// decode work is embarrassingly parallel but memory-bound; more than ~8
/// workers does not pay off and increases the fork-hostile thread count for
/// downstream callers that use spawn-mode multiprocessing.
///
/// Shared with `TrainingPipeline::ensure_decode_pool` so the two cannot drift.
pub const DEFAULT_DECODE_POOL_MAX_THREADS: usize = 8;

/// Environment override for [`resolve_pool_threads`].
pub const CPU_THREADS_ENV: &str = "SCX_LOADER_CPU_THREADS";

/// Worker count for the loader's pool.
///
/// `raw` is the raw `SCX_LOADER_CPU_THREADS` value. Anything that does not
/// parse to a positive integer — unset, empty, `0`, garbage, negative — falls
/// back to the default, deliberately: a mistyped knob should not silently
/// serialise the loader, and `0` has no useful meaning for a pool that must be
/// able to run work.
///
/// Taking the value as an argument rather than reading the environment here
/// keeps the unit tests off the process-global env, which they would otherwise
/// race each other for.
pub fn resolve_pool_threads(raw: Option<&str>) -> usize {
    if let Some(n) = raw.and_then(|v| v.trim().parse::<usize>().ok()) {
        if n > 0 {
            return n;
        }
    }
    num_cpus::get_physical().clamp(1, DEFAULT_DECODE_POOL_MAX_THREADS)
}

/// The cached pool together with the PID that built it.
type PidKeyedPool = Mutex<Option<(u32, Arc<rayon::ThreadPool>)>>;

fn slot() -> &'static PidKeyedPool {
    static SLOT: OnceLock<PidKeyedPool> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

/// The process-wide loader pool, rebuilt whenever `pid` differs from the one
/// that built the cached instance. Split out of [`cpu_pool`] so the fork
/// behaviour is testable without actually forking.
///
/// Panics only if the pool cannot be built at all, which means rayon could not
/// spawn a thread — there is no useful fallback from that, and returning a
/// `Result` here would push a `?` into every call site on the hot path for a
/// condition equivalent to OOM.
fn pool_for_pid(pid: u32) -> Arc<rayon::ThreadPool> {
    let mut guard = slot().lock().unwrap_or_else(|e| e.into_inner());
    if let Some((cached_pid, pool)) = guard.as_ref() {
        if *cached_pid == pid {
            return Arc::clone(pool);
        }
    }
    let n_threads = resolve_pool_threads(std::env::var(CPU_THREADS_ENV).ok().as_deref());
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .thread_name(|i| format!("scx-loader-cpu-{i}"))
            .build()
            .expect("failed to build the scx-loader CPU thread pool"),
    );
    tracing::trace!(pid, n_threads, "scx-loader CPU pool constructed");
    *guard = Some((pid, Arc::clone(&pool)));
    pool
}

/// The loader's CPU pool for *this* process.
///
/// Route every rayon dispatch in this crate — and every call into
/// `scx-format-io` that dispatches rayon underneath — through this, either by
/// handing it to [`scx_format_io::BackedCsrReader::set_cpu_pool`] or by wrapping
/// the call in `cpu_pool().install(...)`. `install` makes this the *current*
/// pool for the calling thread, so nested `par_*` inside a callee dispatches
/// here too, without the callee needing to know about it.
pub fn cpu_pool() -> Arc<rayon::ThreadPool> {
    pool_for_pid(std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every test below mutates the one process-global slot, and cargo runs
    /// them concurrently in a single binary. Without this, a rebuild from one
    /// test lands between another's two `pool_for_pid` calls and fails it for a
    /// reason that has nothing to do with the code under test.
    static SERIALISE: Mutex<()> = Mutex::new(());

    fn exclusive() -> std::sync::MutexGuard<'static, ()> {
        SERIALISE.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn the_same_pid_reuses_one_pool() {
        let _g = exclusive();
        let a = pool_for_pid(4242);
        let b = pool_for_pid(4242);
        assert!(
            Arc::ptr_eq(&a, &b),
            "a repeated pid must reuse the cached pool, not build a second one"
        );
    }

    #[test]
    fn a_changed_pid_rebuilds_the_pool() {
        let _g = exclusive();
        // The fork case: the cached pool belongs to the parent and its worker
        // threads did not survive, so the child must not be handed it back.
        let parent = pool_for_pid(4243);
        let child = pool_for_pid(4244);
        assert!(
            !Arc::ptr_eq(&parent, &child),
            "a changed pid must rebuild — reusing the parent's pool is the \
             original bug with a new owner"
        );
        // And the rebuilt pool is genuinely usable, which is the whole point.
        assert_eq!(child.install(|| 7usize * 6), 42);
    }

    #[test]
    fn cpu_pool_is_stable_within_a_process() {
        let _g = exclusive();
        assert!(Arc::ptr_eq(&cpu_pool(), &cpu_pool()));
    }

    #[test]
    fn resolve_pool_threads_honours_a_positive_override() {
        assert_eq!(resolve_pool_threads(Some("3")), 3);
        assert_eq!(resolve_pool_threads(Some(" 5 ")), 5);
        // Above the clamp on purpose: the clamp is a *default* ceiling, not a
        // cap on an explicit request.
        assert_eq!(resolve_pool_threads(Some("64")), 64);
    }

    #[test]
    fn resolve_pool_threads_falls_back_on_anything_unusable() {
        let default = num_cpus::get_physical().clamp(1, DEFAULT_DECODE_POOL_MAX_THREADS);
        for raw in [
            None,
            Some(""),
            Some("  "),
            Some("0"),
            Some("-1"),
            Some("lots"),
        ] {
            assert_eq!(
                resolve_pool_threads(raw),
                default,
                "{raw:?} must fall back, not serialise the loader"
            );
        }
    }

    #[test]
    fn the_default_is_clamped_and_never_zero() {
        let n = resolve_pool_threads(None);
        assert!(
            (1..=DEFAULT_DECODE_POOL_MAX_THREADS).contains(&n),
            "default {n} outside 1..={DEFAULT_DECODE_POOL_MAX_THREADS}"
        );
    }
}
