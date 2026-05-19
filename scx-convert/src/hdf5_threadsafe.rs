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
}
