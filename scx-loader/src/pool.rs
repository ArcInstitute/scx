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
//! # Why the slot is lock-free and never freed
//!
//! Both of the obvious implementations deadlock in the child, for the same
//! reason the bug exists at all: `fork()` copies mutexes in whatever state they
//! were in, and the threads that would have unlocked them are gone.
//!
//! * **Never drop the inherited pool.** `ThreadPool::drop` → `Registry::terminate`
//!   → `OnceLatch::set_and_tickle_one` → `Sleep::wake_specific_thread`, which does
//!   `sleep_state.is_blocked.lock().unwrap()` (rayon-core 1.13.0,
//!   `sleep/mod.rs:288-291`). A parent worker asleep at fork time leaves that
//!   mutex locked in the child with nobody to release it. "It does not join" is
//!   **not** sufficient grounds to drop it — an earlier version of this comment
//!   said exactly that and was wrong. Entries are therefore leaked deliberately:
//!   one small allocation plus one dead `Arc` per fork generation.
//! * **Never take a lock to read the slot.** A `Mutex` around the slot can itself
//!   be inherited locked, so the child would hang before it ever used the fresh
//!   pool. An `AtomicPtr` to a never-freed entry needs no lock to read, so the
//!   child's first access cannot block on parent state.
//!
//! The cost of the leak is bounded and tiny; the cost of getting it wrong is the
//! exact silent hang this module exists to prevent.

use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Arc;

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

/// A pool together with the PID that built it. Allocated with [`Box::leak`] and
/// **never freed** — see the module docs for why dropping one in a forked child
/// can hang.
struct Entry {
    pid: u32,
    pool: Arc<rayon::ThreadPool>,
}

/// A published [`Entry`] pointer, or null before the first pool is built.
///
/// [`Slot`] rather than a bare static so the tests can drive the PID logic
/// against their own instance. They used to share the production static with a
/// module-local mutex, which does not work: `SparseCellSetLoader::new` and
/// friends now call [`cpu_pool`] too, and those tests never took that mutex, so
/// a real-PID rebuild could land between a test's two calls. Reproduced at 4
/// failures in 40 runs of `RUST_TEST_THREADS=128 cargo test -p scx-loader --lib`
/// before this was split out.
pub(crate) struct Slot(AtomicPtr<Entry>);

impl Slot {
    pub(crate) const fn new() -> Self {
        Slot(AtomicPtr::new(std::ptr::null_mut()))
    }

    /// The pool for `pid`, building and publishing one if the slot is empty or
    /// holds another process's.
    ///
    /// Panics only if rayon cannot spawn a thread at all; there is no useful
    /// fallback from that, and a `Result` would push a `?` into every call site
    /// for a condition equivalent to OOM.
    fn pool_for_pid(&self, pid: u32) -> Arc<rayon::ThreadPool> {
        let current = self.0.load(Ordering::Acquire);
        if !current.is_null() {
            // SAFETY: every pointer ever stored comes from `Box::leak` below and
            // is never freed, so a non-null load is always a live `Entry`.
            let entry = unsafe { &*current };
            if entry.pid == pid {
                return Arc::clone(&entry.pool);
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

        let fresh: *mut Entry = Box::leak(Box::new(Entry {
            pid,
            pool: Arc::clone(&pool),
        }));
        match self
            .0
            .compare_exchange(current, fresh, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => pool,
            Err(winner) => {
                // Someone published first. Our `pool` is this process's own and
                // its threads are alive, so dropping it here is safe — unlike
                // the inherited one, which is why `fresh` is left leaked rather
                // than reclaimed.
                // SAFETY: as above — `winner` came from `Box::leak`.
                let entry = unsafe { &*winner };
                if entry.pid == pid {
                    Arc::clone(&entry.pool)
                } else {
                    pool
                }
            }
        }
    }
}

static SLOT: Slot = Slot::new();

/// The loader's CPU pool for *this* process.
///
/// Route every rayon dispatch in this crate — and every call into
/// `scx-format-io` that dispatches rayon underneath — through this, either by
/// handing it to [`scx_format_io::BackedCsrReader::set_cpu_pool`] or by wrapping
/// the call in `cpu_pool().install(...)`. `install` makes this the *current*
/// pool for the calling thread, so nested `par_*` inside a callee dispatches
/// here too, without the callee needing to know about it.
pub fn cpu_pool() -> Arc<rayon::ThreadPool> {
    SLOT.pool_for_pid(std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each test drives its **own** slot. Sharing the production static — even
    /// under a module-local mutex — is not viable: `SparseCellSetLoader::new`
    /// and the other constructors now call `cpu_pool()` themselves, and those
    /// tests hold no such mutex, so a real-PID rebuild lands between a test's
    /// two calls and fails it for a reason unrelated to the code under test.
    /// That was not hypothetical: 4 of 40 runs of
    /// `RUST_TEST_THREADS=128 cargo test -p scx-loader --lib` failed this way.
    fn fresh_slot() -> Slot {
        Slot::new()
    }

    #[test]
    fn the_same_pid_reuses_one_pool() {
        let slot = fresh_slot();
        let a = slot.pool_for_pid(4242);
        let b = slot.pool_for_pid(4242);
        assert!(
            Arc::ptr_eq(&a, &b),
            "a repeated pid must reuse the cached pool, not build a second one"
        );
    }

    #[test]
    fn a_changed_pid_rebuilds_the_pool() {
        let slot = fresh_slot();
        // The fork case: the cached pool belongs to the parent and its worker
        // threads did not survive, so the child must not be handed it back.
        let parent = slot.pool_for_pid(4243);
        let child = slot.pool_for_pid(4244);
        assert!(
            !Arc::ptr_eq(&parent, &child),
            "a changed pid must rebuild — reusing the parent's pool is the \
             original bug with a new owner"
        );
        // And the rebuilt pool is genuinely usable, which is the whole point.
        assert_eq!(child.install(|| 7usize * 6), 42);
    }

    #[test]
    fn a_rebuild_never_drops_the_previous_pool() {
        // The inherited pool must be leaked, not dropped: `ThreadPool::drop`
        // locks each worker's `is_blocked` mutex, which a forked child can
        // inherit locked with no owner to release it. Holding no `Arc` of our
        // own, the parent pool must still be alive after the rebuild — if the
        // slot had dropped it, `Arc::strong_count` on a resurrected handle
        // could not still see the leaked entry's reference.
        let slot = fresh_slot();
        let parent_weak = Arc::downgrade(&slot.pool_for_pid(4245));
        let _child = slot.pool_for_pid(4246);
        assert!(
            parent_weak.upgrade().is_some(),
            "the superseded pool was dropped — in a forked child that drop can \
             block forever on an inherited-locked worker mutex"
        );
    }

    #[test]
    fn cpu_pool_is_stable_within_a_process() {
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
