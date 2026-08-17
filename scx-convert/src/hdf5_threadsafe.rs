// Runtime check for libhdf5 thread-safety.
//
// The parallel streaming coordinator drives multiple worker threads
// against shared `hdf5::Dataset` handles. libhdf5 serialises
// overlapping reads through its internal global lock when built with
// `--enable-threadsafe` (conda-forge `hdf5=1.12.*=nompi*` ships that
// option). Without it, concurrent reads race on internal state and
// corrupt the output silently.
//
// We probe the library at runtime via `H5is_library_threadsafe`
// (libhdf5 ≥ 1.8.16, available in every release we support) and
// cache the result in a `OnceLock`. Non-thread-safe builds fall
// back to the sequential coordinator with a one-shot warning.

use std::sync::OnceLock;

use crate::warnings::{ConvertWarning, WarningSink};

static IS_THREADSAFE: OnceLock<bool> = OnceLock::new();
static EMITTED_NOT_THREADSAFE: OnceLock<()> = OnceLock::new();

/// Returns `true` iff libhdf5 was built with `--enable-threadsafe`.
///
/// Cached: the FFI probe runs at most once per process.
pub fn hdf5_is_threadsafe() -> bool {
    *IS_THREADSAFE.get_or_init(probe)
}

fn probe() -> bool {
    let mut is_ts: hdf5_sys::h5::hbool_t = 0;
    // SAFETY: pure FFI getter; libhdf5 is statically initialised by
    // the time any `hdf5::File::open` has run on this process, and
    // the call has no preconditions.
    let rc = unsafe { hdf5_sys::h5::H5is_library_threadsafe(&mut is_ts) };
    rc >= 0 && is_ts != 0
}

/// Emit [`ConvertWarning::Hdf5NotThreadsafe`] at most once per
/// process. Subsequent calls are no-ops. The condition is a build-time
/// property of libhdf5 and cannot change within a process, so per-
/// process scope is correct — without this guard the dispatcher would
/// fire the warning on every matrix and every modality-layer of a
/// multimodal h5mu conversion.
pub fn try_emit_not_threadsafe_warning(sink: &mut WarningSink) {
    if EMITTED_NOT_THREADSAFE.set(()).is_ok() {
        sink.emit(ConvertWarning::Hdf5NotThreadsafe);
    }
}

/// Name of the environment variable that turns a non-thread-safe libhdf5
/// from a silent test skip into a hard failure. Set by the `Test (hdf5
/// features)` CI job. Test-only: production never consults it, because a
/// non-thread-safe build is a supported configuration that falls back to
/// the sequential coordinator.
#[cfg(test)]
pub(crate) const REQUIRE_THREADSAFE_ENV: &str = "SCX_REQUIRE_HDF5_THREADSAFE";

/// `true` when the caller has demanded a thread-safe libhdf5 (CI).
///
/// Tests of the parallel coordinators return early on a non-thread-safe
/// build, which is right on a developer laptop and wrong in CI: libtest
/// captures the skip message, so the lane reports green with the
/// coordinator untested.
#[cfg(test)]
pub(crate) fn threadsafe_required() -> bool {
    std::env::var_os(REQUIRE_THREADSAFE_ENV).is_some_and(|v| !v.is_empty() && v != "0")
}

/// Skip-or-fail helper for tests that need the parallel coordinator.
///
/// Returns `true` when the caller should skip. Panics — failing the test —
/// when libhdf5 is not thread-safe *and* [`REQUIRE_THREADSAFE_ENV`] is set,
/// so the CI lane cannot go green on a build where every parallel test
/// quietly returns early.
#[cfg(test)]
pub(crate) fn skip_if_not_threadsafe(test_name: &str) -> bool {
    if hdf5_is_threadsafe() {
        return false;
    }
    assert!(
        !threadsafe_required(),
        "{test_name} needs a libhdf5 built with --enable-threadsafe, and \
         {REQUIRE_THREADSAFE_ENV} is set. This test would otherwise skip \
         silently and take the whole parallel-coordinator surface with it."
    );
    eprintln!("skipping {test_name}: libhdf5 not built thread-safe");
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_a_definite_answer() {
        // We can't assert true/false (depends on the host's libhdf5
        // build), but the call must complete without panicking and
        // must be stable across repeated invocations.
        let a = hdf5_is_threadsafe();
        let b = hdf5_is_threadsafe();
        assert_eq!(a, b);
    }

    /// The lane-level guard. Every test of the parallel ingest / export
    /// coordinators returns early when libhdf5 lacks `--enable-threadsafe`,
    /// and libtest swallows the message — so on such a host the whole
    /// `Test (hdf5 features)` job would report green having exercised none
    /// of it. `docs/development.md`'s claim that the Ubuntu system package
    /// ships the option is a claim about someone else's build, not a
    /// measurement of the runner's. This turns it into one.
    ///
    /// Same shape as `SCX_REQUIRE_GPU`, which exists because 176 `scx-gpu`
    /// tests were passing while doing nothing on a CPU host.
    #[test]
    fn ci_requires_a_threadsafe_libhdf5() {
        if !threadsafe_required() {
            return;
        }
        assert!(
            hdf5_is_threadsafe(),
            "{REQUIRE_THREADSAFE_ENV} is set but H5is_library_threadsafe() \
             reports false: this libhdf5 was built without --enable-threadsafe, \
             so every parallel-coordinator test would skip itself."
        );
    }
}
