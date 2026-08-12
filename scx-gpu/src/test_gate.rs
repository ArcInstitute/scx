//! The single decision point for "should this GPU test run, skip, or fail?".
//!
//! ## Why this exists
//!
//! A GPU test that cannot acquire a device has to do *something*, and the
//! obvious `eprintln! + return` is the wrong something: libtest records the
//! test as **passed** and captures the message, so on a CPU-only host — which
//! is every CI runner and most dev machines — the whole suite reports green
//! while executing none of it. Inverting a plane index in a decode kernel would
//! not have moved a single counter.
//!
//! The fix has two halves and this module is the second:
//!
//! 1. every GPU test carries `#[ignore = "requires a CUDA GPU"]`, so a default
//!    run counts and names it as ignored rather than passed;
//! 2. when the test *is* selected (`--include-ignored`), this module decides
//!    whether the environment can honour that, and under `SCX_REQUIRE_GPU=1`
//!    turns "cannot" into a hard failure instead of a silent early return.
//!
//! The sbatch harness (`benchmarks/scripts/_run_scx_gpu_tests.sh`) sets
//! `SCX_REQUIRE_GPU=1` and passes `--include-ignored`, so on a GPU node a test
//! that quietly declines to run is a red build. `scx-gpu/tests/gpu_test_gating.rs`
//! keeps the two halves in sync.
//!
//! ## Why it is public, and not `#[cfg(test)]`
//!
//! `scx-accel`'s GPU tests need the identical decision, and a `#[cfg(test)]`
//! item is not reachable from another crate. The pre-existing [`crate::test_utils`]
//! module is gated behind this crate's `bench` feature, which
//! `scx-accel --features gpu` does not enable. Before this module there were ten
//! hand-rolled copies of `match GpuDevice::new(0)` across the two crates, three
//! of them buried in helper functions where no test-body-level check could see
//! them; consolidating is the point.

use crate::device::GpuDevice;

/// Set to `1` to turn "no CUDA device" from a skip into a panic.
pub const REQUIRE_GPU_ENV: &str = "SCX_REQUIRE_GPU";

/// Set to `1` to turn "nvcomp not loadable" from a skip into a panic.
pub const REQUIRE_NVCOMP_ENV: &str = "SCX_REQUIRE_NVCOMP";

/// Set to `1` to turn "cuVS not loadable" from a skip into a panic.
pub const REQUIRE_CUVS_ENV: &str = "SCX_REQUIRE_CUVS";

/// Prefix of the line printed when a gate declines to run a test.
///
/// The harness greps for it to report what a GPU-node run did *not* cover, so a
/// library that silently disappears from a node shows up as a listed skip
/// rather than as a test that merely stopped existing.
pub const SKIP_MARKER: &str = "SCX_GPU_TEST_SKIPPED";

fn strict(var: &str) -> bool {
    std::env::var(var).as_deref() == Ok("1")
}

/// Acquire device 0 for a test, or decide the test must skip.
///
/// `None` means the caller should `return` — the gate macros do that, since a
/// function cannot return on its caller's behalf.
///
/// # Panics
///
/// When `SCX_REQUIRE_GPU=1` and no device can be opened. That is the whole
/// point: on a host that is *supposed* to have a GPU, a skip is a failure.
pub fn device_or_skip(what: &str) -> Option<GpuDevice> {
    match GpuDevice::new(0) {
        Ok(dev) => Some(dev),
        Err(e) => {
            assert!(
                !strict(REQUIRE_GPU_ENV),
                "{REQUIRE_GPU_ENV}=1 but no CUDA device is available for {what}: {e}"
            );
            eprintln!("{SKIP_MARKER}: {what} — no CUDA device ({e})");
            None
        }
    }
}

/// Gate on an optional CUDA library (nvcomp, cuVS) that a GPU node may or may
/// not carry. `false` means the caller should `return`.
///
/// Separate from [`device_or_skip`] on purpose: a missing GPU on a GPU node is
/// a broken job, while a missing optional library is a real deployment state
/// that should be *reported*, not asserted away. Each capability gets its own
/// opt-in strict variable so a run that means to cover it can say so.
///
/// # Panics
///
/// When `env_var` is `1` and the capability is unavailable.
pub fn capability_or_skip(cap: &str, env_var: &str, available: bool, what: &str) -> bool {
    if available {
        return true;
    }
    assert!(
        !strict(env_var),
        "{env_var}=1 but {cap} is not available for {what}"
    );
    eprintln!("{SKIP_MARKER}: {what} — {cap} not available");
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The strict variables are opt-in: unset means skip, not panic. Asserted
    /// against the real reader rather than a copy of the comparison.
    #[test]
    fn strict_is_opt_in() {
        // A name no environment sets, so this is the "unset" case.
        assert!(!strict("SCX_REQUIRE_GPU_DEFINITELY_UNSET_2ADB1F"));
    }

    /// An available capability never consults the environment, so it cannot
    /// panic under a strict variable that happens to be set.
    #[test]
    fn available_capability_never_panics() {
        assert!(capability_or_skip(
            "nvcomp",
            REQUIRE_NVCOMP_ENV,
            true,
            "test_gate::available_capability_never_panics"
        ));
    }
}
