//! The CUDA-graph capture contract, made enforceable.
//!
//! `gpu_graph.rs` states the rule in prose — "Captured regions MUST NOT
//! allocate device memory or do host syncs — capture mode rejects these APIs"
//! — and two more doc comments repeat it. Nothing checked it. The failure it
//! guards against is quiet in the worst way: an allocation inside a capture
//! region does not raise where it was written. CUDA invalidates the capture,
//! `cuStreamEndCapture` returns no graph, and the one production call site
//! (`scx-accel/src/harmony/gpu.rs`) logs "capture produced no graph; running
//! the kernels directly" and runs on at a multi-× slowdown. The message names
//! no cause, because at that point nothing knows one.
//!
//! This module is the cause, named. A thread-local depth counter is armed by
//! [`crate::gpu_graph::capture_graph`] for exactly the span between
//! `begin_capture` and `end_capture`, and every device allocation and host sync
//! in the crate funnels through [`crate::device::GpuDevice`], which asks
//! [`check`] first. A violation becomes [`GpuError::CaptureViolation`] naming
//! the operation, raised at the line that did it.
//!
//! # Why thread-local
//!
//! `capture_graph` begins capture in
//! `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`, which is precisely the scope CUDA
//! itself applies to the "potentially unsafe API call" rule: calls on *this*
//! thread are restricted, other threads are untouched. A thread-local counter
//! is the same scope, so the guard cannot be tighter or looser than the driver
//! it mirrors.
//!
//! It is a depth counter rather than a flag so a nested capture — none today,
//! but the type permits one — disarms to the right level rather than to zero.
//!
//! # Why this module has no CUDA in it
//!
//! Same reason as [`crate::csr_placement`]: every test that can reach a real
//! capture region is `#[ignore = "requires a CUDA GPU"]`, so a guard whose
//! logic lived next to the driver calls could not be watched fail on a
//! CPU host. Here the arm/disarm/check state machine is plain Rust and its
//! tests run everywhere.

use std::cell::Cell;
use std::marker::PhantomData;

use crate::error::GpuError;

thread_local! {
    /// Nesting depth of active capture regions on this thread. Zero means no
    /// capture is in progress and every device operation is legal.
    static CAPTURE_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// An armed capture region. Created by [`enter`], disarmed on `Drop`.
///
/// Drop is what makes this correct in the presence of the `?` operator and of
/// panics: `capture_graph`'s closure returns `Result`, and an early return from
/// deep inside it must not leave the thread permanently armed — every
/// subsequent allocation on that thread would then be rejected. `Drop` runs on
/// the error path and during unwind, so the disarm cannot be skipped.
///
/// Deliberately `!Send`: the counter belongs to the thread that armed it, and
/// dropping the scope on another thread would decrement a counter that was
/// never incremented.
#[must_use = "the capture region is disarmed when this guard is dropped; \
              binding it to `_` disarms immediately"]
pub(crate) struct CaptureScope {
    /// Makes the type `!Send` / `!Sync`. Carries no data.
    _not_send: PhantomData<*const ()>,
}

impl Drop for CaptureScope {
    fn drop(&mut self) {
        CAPTURE_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// Arm a capture region on this thread until the returned guard is dropped.
///
/// Called by [`crate::gpu_graph::capture_graph`], which owns the only
/// `begin_capture` in the workspace.
pub(crate) fn enter() -> CaptureScope {
    CAPTURE_DEPTH.with(|d| d.set(d.get().saturating_add(1)));
    CaptureScope {
        _not_send: PhantomData,
    }
}

/// Whether this thread is currently inside a capture region.
pub(crate) fn in_capture() -> bool {
    CAPTURE_DEPTH.with(|d| d.get()) > 0
}

/// `Ok(())` outside a capture region; [`GpuError::CaptureViolation`] inside.
///
/// `op` names the operation being refused, in the caller's own words — it is
/// the whole diagnostic value here, so spell it as the method a reader would
/// grep for (`"GpuDevice::alloc_zeros"`), not as a category.
///
/// This returns rather than `debug_assert!`s, against the letter of the task
/// that asked for it, because the crate's own policy is real errors over
/// assertions that vanish in release (`gpu_shard_source.rs`: "returns
/// `InvalidShard` rather than relying on a `debug_assert!`"), because every
/// funnel already returns `Result` so it costs nothing, and because the one
/// caller that can see this error already degrades to direct dispatch with a
/// warning — so a violation becomes a named, logged fallback instead of a
/// silent one.
pub(crate) fn check(op: &str) -> Result<(), GpuError> {
    if in_capture() {
        return Err(GpuError::CaptureViolation(format!(
            "{op} was called inside a CUDA-graph capture region; capture mode \
             rejects device allocations, host synchronization and module loads, \
             and would silently invalidate the graph. Hoist it outside the \
             capture, or pre-grow the buffer before capture begins"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "capture_guard_tests.rs"]
mod tests;
