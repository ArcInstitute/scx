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
//! `SCX_REQUIRE_GPU=1` — and defaults `SCX_REQUIRE_LARGE_VRAM=1`, the same
//! escalation for [`vram_or_skip`] — and passes `--include-ignored`, so on a GPU
//! node a test that quietly declines to run is a red build.
//! `scx-gpu/tests/gpu_test_gating.rs` keeps the two halves in sync.
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

/// Set to `1` to turn "not enough free VRAM" from a skip into a panic.
pub const REQUIRE_LARGE_VRAM_ENV: &str = "SCX_REQUIRE_LARGE_VRAM";

/// Prefix of the line printed when a gate declines to run a test.
///
/// The harness greps for it to report what a GPU-node run did *not* cover, so a
/// library that silently disappears from a node shows up as a listed skip
/// rather than as a test that merely stopped existing.
pub const SKIP_MARKER: &str = "SCX_GPU_TEST_SKIPPED";

pub use decision::{decline, strict};

/// The strict-escalation decision, in a submodule so [`Strict`] is opaque to
/// the gates.
///
/// The gates below need to know whether a strict variable is set, but a gate
/// that could *inspect* that answer could also act on it — and the whole value
/// of this module is that "no device, and the environment said that is a
/// failure" resolves the same way everywhere. `Strict` is therefore
/// constructed only by [`strict`] and consumed only by [`decline`]; outside
/// this submodule there is no way to get the `bool` back out. A gate that
/// hand-rolls `if env::var(...) == "1"` is a visible edit rather than a
/// one-character drift.
///
/// It also makes the escalation **testable without a GPU and without touching
/// the environment**: [`decision::strict_for_test`] builds the `true` case
/// directly. Before this split the `SCX_REQUIRE_GPU=1` branch was reachable
/// only by running the suite on a CPU host with the variable set — i.e. by
/// hand, in a harness, never in CI — so the one assertion the whole GPU-test
/// contract rests on was itself covered by nothing.
mod decision {
    /// Opaque answer to "is this strict variable set to `1`?".
    #[must_use]
    pub struct Strict(bool);

    /// Read a strict variable. Exactly `"1"` opts in; anything else, including
    /// `"true"` and `"0"`, does not.
    pub fn strict(var: &str) -> Strict {
        Strict(std::env::var(var).as_deref() == Ok("1"))
    }

    /// A gate's precondition is not met: panic under strict, else print the
    /// skip marker for the harness to collect.
    ///
    /// Both messages are closures so the strict-mode text is not built on the
    /// (overwhelmingly common) skip path, and — more usefully — so a test can
    /// prove the non-strict path never builds it at all.
    ///
    /// # Panics
    ///
    /// When `s` was read from a variable set to `1`.
    pub fn decline(s: Strict, reason: impl FnOnce() -> String, skip: impl FnOnce() -> String) {
        if s.0 {
            panic!("{}", reason());
        }
        eprintln!("{}: {}", super::SKIP_MARKER, skip());
    }

    /// Build a `Strict` directly, so the strict branch is reachable from a
    /// CPU-only unit test. Mutating the real variable would not do: it is
    /// process-global, and every other test in the binary shares it.
    #[cfg(test)]
    pub fn strict_for_test(v: bool) -> Strict {
        Strict(v)
    }
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
            decline(
                strict(REQUIRE_GPU_ENV),
                || format!("{REQUIRE_GPU_ENV}=1 but no CUDA device is available for {what}: {e}"),
                || format!("{what} — no CUDA device ({e})"),
            );
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
    decline(
        strict(env_var),
        || format!("{env_var}=1 but {cap} is not available for {what}"),
        || format!("{what} — {cap} not available"),
    );
    false
}

/// Gate on **free VRAM**, for the handful of tests whose whole point is a
/// buffer past a 32-bit index boundary. `false` means the caller should `return`.
///
/// These cannot be shrunk: an 8 GiB allocation is the smallest launch that
/// reaches 2³¹ f32 elements, which is the only size at which the overflow being
/// tested exists at all. A small card is a real deployment state, so it skips
/// and *reports* — but a run that means to cover the case says so with
/// [`REQUIRE_LARGE_VRAM_ENV`] and gets a hard failure instead.
///
/// # Panics
///
/// When `SCX_REQUIRE_LARGE_VRAM=1` and the device has less free VRAM than
/// `need_bytes`, or when free VRAM cannot be queried at all.
pub fn vram_or_skip(dev: &GpuDevice, need_bytes: usize, what: &str) -> bool {
    let free = match dev.free_memory() {
        Ok((free, _total)) => free,
        Err(e) => {
            decline(
                strict(REQUIRE_LARGE_VRAM_ENV),
                || {
                    format!(
                        "{REQUIRE_LARGE_VRAM_ENV}=1 but free VRAM could not be queried \
                         for {what}: {e}"
                    )
                },
                || format!("{what} — free VRAM unknown ({e})"),
            );
            return false;
        }
    };
    if free >= need_bytes {
        return true;
    }
    let (need_mib, free_mib) = (need_bytes / (1 << 20), free / (1 << 20));
    decline(
        strict(REQUIRE_LARGE_VRAM_ENV),
        || {
            format!(
                "{REQUIRE_LARGE_VRAM_ENV}=1 but {what} needs {need_mib} MiB free VRAM \
                 and only {free_mib} MiB is free"
            )
        },
        || format!("{what} — needs {need_mib} MiB free VRAM, {free_mib} MiB available"),
    );
    false
}

#[cfg(test)]
mod tests {
    use super::decision::strict_for_test;
    use super::*;

    /// The strict variables are opt-in: unset means skip, not panic. Asserted
    /// against the real reader rather than a copy of the comparison.
    ///
    /// A name no environment sets, so this is the "unset" case. `decline` must
    /// therefore not panic — and, since the reason closure is `unreachable!`,
    /// it must not even build the strict message.
    #[test]
    fn strict_is_opt_in() {
        decline(
            strict("SCX_REQUIRE_GPU_DEFINITELY_UNSET_2ADB1F"),
            || unreachable!("an unset strict variable must not build a panic message"),
            || "test_gate::strict_is_opt_in".to_string(),
        );
    }

    /// The escalation the entire GPU-test contract rests on: with the variable
    /// set, "cannot run" is a failure, not a skip.
    ///
    /// This is the assertion that had no test. It could previously be reached
    /// only by running the suite on a CPU host with `SCX_REQUIRE_GPU=1` — by
    /// hand or from the sbatch harness, never in CI — so nothing would have
    /// noticed it being weakened, and every GPU test would have gone back to
    /// silently passing while doing nothing.
    #[test]
    #[should_panic(expected = "SCX_REQUIRE_GPU=1 but no CUDA device is available")]
    fn strict_turns_a_skip_into_a_failure() {
        decline(
            strict_for_test(true),
            || format!("{REQUIRE_GPU_ENV}=1 but no CUDA device is available for x"),
            || unreachable!("a strict decline must panic rather than print a skip"),
        );
    }

    /// The same escalation for the optional-library and VRAM gates, which
    /// share `decline` precisely so there is one branch to get right.
    #[test]
    #[should_panic(expected = "SCX_REQUIRE_NVCOMP=1 but nvcomp is not available")]
    fn strict_capability_gate_also_fails() {
        decline(
            strict_for_test(true),
            || format!("{REQUIRE_NVCOMP_ENV}=1 but nvcomp is not available for x"),
            || unreachable!(),
        );
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

    /// An *unavailable* capability under an unset variable skips and returns
    /// `false` — the real gate, not a stand-in for it.
    #[test]
    fn unavailable_capability_skips_when_not_strict() {
        assert!(!capability_or_skip(
            "nvcomp",
            "SCX_REQUIRE_NVCOMP_DEFINITELY_UNSET_2ADB1F",
            false,
            "test_gate::unavailable_capability_skips_when_not_strict"
        ));
    }

    /// `Strict` is opaque outside `decision`, so a gate that reads a strict
    /// variable has nothing it *can* do with the answer except hand it to
    /// `decline`. That closes one shape of drift; this closes the other — a
    /// gate that stops consulting the environment at all and just returns.
    ///
    /// Without it the `decline` tests above would keep passing while the gates
    /// no longer called it, which is exactly the class of green-but-vacuous
    /// test this module exists to eliminate.
    #[test]
    fn every_gate_routes_its_decline_through_decline() {
        let src = include_str!("test_gate.rs");
        for name in ["device_or_skip", "capability_or_skip", "vram_or_skip"] {
            let start = src
                .find(&format!("pub fn {name}("))
                .unwrap_or_else(|| panic!("{name} not found — was it renamed?"));
            // Top-level fns, so the next `\npub fn ` / `\nmod ` / `\n#[cfg(test)]`
            // ends the body without needing to brace-match.
            let rest = &src[start..];
            let end = ["\npub fn ", "\nmod ", "\n#[cfg(test)]"]
                .iter()
                .filter_map(|m| rest[1..].find(m).map(|i| i + 1))
                .min()
                .unwrap_or(rest.len());
            let body = &rest[..end];
            assert!(
                body.contains("decline("),
                "{name} no longer routes through decline() — the \
                 {REQUIRE_GPU_ENV}-style escalation is only tested there"
            );
        }
    }
}
