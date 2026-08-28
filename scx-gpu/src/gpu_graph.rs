//! CUDA Graph capture/replay infrastructure for iteration-heavy GPU loops.
//!
//! Iteration-heavy stages (Harmony k-means, UMAP SGD, GPU DE chunk loops)
//! launch many small kernels per iteration. Per-launch overhead (~5–20 µs
//! each) dominates on small inputs. `cudaStreamBegin/EndCapture` records a
//! stable kernel sequence into a `cudaGraph_t` once; subsequent replays via
//! `cuGraphLaunch` amortize per-launch latency.
//!
//! NOTE: the PCA power loop is **no longer a capture target** — its
//! SpMM-segment capture was removed (`run_resident_power_loop` always reports
//! "no replay"). The `GpuPcaScratch` stable-buffer prerequisite is kept only
//! as a model for the remaining capture sites; do not re-enable PCA capture
//! without first moving its per-iteration QR/eigh/slot-grow allocations
//! outside the capture region.
//!
//! ## Prerequisites
//!
//! - Captured regions MUST NOT allocate device memory or do host syncs —
//!   capture mode rejects these APIs. G2's `GpuPcaScratch` /
//!   `GpuDeChunkScratch` provide stable pre-grown scratch addresses; the
//!   stable-buffer prerequisite is already in place.
//!
//!   This one is **enforced**, not merely stated: [`capture_graph`] arms
//!   [`crate::capture_guard`] for the span of the capture, and every device
//!   allocation, host sync and module load in the crate funnels through
//!   [`crate::device::GpuDevice`], which checks it. A violation returns
//!   [`GpuError::CaptureViolation`] naming the operation — where before, CUDA
//!   invalidated the capture and the only symptom was `end_capture` returning
//!   no graph, several frames away from the cause.
//! - Capture MUST run on a non-NULL stream, and specifically on
//!   `CudaContext::per_thread_stream()`. cudarc's
//!   `CudaContext::default_stream()` returns the NULL stream (`cu_stream
//!   = std::ptr::null_mut()`) which CUDA rejects for capture
//!   (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`).
//!
//!   **Do not capture on `CudaContext::new_stream()`.** It is capturable in
//!   isolation, but creating one flips the cudarc context into multi-stream
//!   mode, which turns on cudarc's automatic cross-stream synchronization
//!   (`is_managing_stream_synchronization()`); the `cuStreamWaitEvent` calls
//!   that inserts invalidate any capture they touch. `per_thread_stream()` is
//!   capturable and does *not* flip that switch. Note this is a property of
//!   the whole context, not of one call site: while any `new_stream()` stream
//!   is alive anywhere in the process — the GPU shard sources and the
//!   ShufDelta decoder each create one — capture can fail. Each call site is
//!   still responsible for any cross-stream synchronization it needs
//!   before/after replay.
//!
//! ## Kill switch
//!
//! Set `SCX_DISABLE_CUDA_GRAPHS=1` to bypass capture at every call site;
//! callers fall back to direct kernel dispatch with no change to results.
//!
//! It is **not** an isolated "graph vs no graph" A/B, because call sites also
//! use it to choose the stream: `harmony/gpu.rs` runs its k-means sub-iter
//! kernels, order upload and sync on the per-thread stream when graphs are
//! enabled and on the device's own stream when they are not, and three sites
//! in `diffexp/gpu.rs` read it purely as a stream selector with no capture
//! involved at all. Read a measurement taken with the switch set accordingly.

use std::sync::{Arc, OnceLock};

use cudarc::driver::safe::{CudaGraph, CudaStream};
use cudarc::driver::sys;

use crate::error::GpuError;

/// Run `build` between `begin_capture` / `end_capture` and return the
/// instantiated graph. The only production capture site is Harmony's
/// k-means sub-loop (`scx-accel/src/harmony/gpu.rs`).
pub fn capture_graph<F>(stream: &Arc<CudaStream>, build: F) -> Result<Option<CudaGraph>, GpuError>
where
    F: FnOnce(&Arc<CudaStream>) -> Result<(), GpuError>,
{
    // ThreadLocal is the recommended mode for library code that wants to
    // capture without disturbing other threads' work on different
    // streams. See CUDA Driver API § cuStreamBeginCapture_v2.
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .map_err(|e| GpuError::CudaError(format!("begin_capture: {e}")))?;

    // Arm the capture contract for exactly the span the driver considers
    // captured. `capture_guard::enter` mirrors THREAD_LOCAL's scope, and its
    // guard disarms on `Drop` — so the `build_result?` below, and any panic
    // inside `build`, cannot leave this thread armed. Scoped to end before
    // `end_capture`, which is itself a legal call.
    let build_result = {
        let _capture_scope = crate::capture_guard::enter();
        build(stream)
    };

    // `end_capture` requires a `CUgraphInstantiate_flags` enum value;
    // the cuda-12.x enum has no "zero flags" variant, and transmuting
    // 0u32 into the enum trips cudarc's runtime enum-validity check.
    // Use `AUTO_FREE_ON_LAUNCH` (value 1) — it requests automatic
    // freeing of memory allocated inside the captured graph, which is
    // a no-op for us because the capture-region contract forbids
    // device allocations inside the closure. The instantiated graph
    // behaves identically to one created with `cudaGraphInstantiate(...,
    // 0)` so long as that invariant holds.
    let flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;

    let end_result = stream.end_capture(flags);

    // If the build closure errored, we've already issued begin_capture
    // and must still call end_capture to drain the stream's capture
    // state. Propagate the build error in preference to end_capture's.
    build_result?;
    end_result.map_err(|e| GpuError::CudaError(format!("end_capture: {e}")))
}

/// Returns `true` unless `SCX_DISABLE_CUDA_GRAPHS=1`. The env var is
/// read once and cached so repeated dispatch-site checks are free.
///
/// In test builds, [`set_cuda_graphs_enabled_override`] takes precedence
/// over the cached env-var read so parity tests can toggle the kill
/// switch in-process.
///
/// Call sites should check this BEFORE invoking [`capture_graph`]; when
/// graphs are disabled they fall back to direct kernel dispatch.
pub fn cuda_graphs_enabled() -> bool {
    if let Some(v) = test_override::current() {
        return v;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("SCX_DISABLE_CUDA_GRAPHS").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        )
    })
}

mod test_override {
    use std::sync::{Mutex, OnceLock};

    static CELL: OnceLock<Mutex<Option<bool>>> = OnceLock::new();

    fn slot() -> &'static Mutex<Option<bool>> {
        CELL.get_or_init(|| Mutex::new(None))
    }

    pub fn current() -> Option<bool> {
        *slot().lock().unwrap()
    }

    pub fn set(value: Option<bool>) -> Option<bool> {
        let mut guard = slot().lock().unwrap();
        let prev = *guard;
        *guard = value;
        prev
    }
}

/// Diagnostic override for [`cuda_graphs_enabled`] — lets parity tests
/// (and any other in-process diagnostic) toggle the kill switch
/// without restarting the process. Returns the previous override value.
///
/// `Some(false)` forces graphs off (mirrors `SCX_DISABLE_CUDA_GRAPHS=1`);
/// `Some(true)` forces graphs on; `None` returns to env-var-controlled
/// behaviour.
///
/// Not gated to `#[cfg(test)]` because downstream crates' integration
/// tests (`scx-accel`, `pyscx`) consume this via the scx-gpu dependency
/// graph — Rust strips `cfg(test)` items at the crate boundary.
/// Calling this from production code is harmless (it only toggles a
/// thread-local override that affects future capture decisions); the
/// expected production setting is the default `None`.
pub fn set_cuda_graphs_enabled_override(enabled: Option<bool>) -> Option<bool> {
    test_override::set(enabled)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stream capture is NOT permitted on cudarc's `default_stream()`
    /// (it returns the NULL stream, which CUDA rejects for capture).
    /// We use `per_thread_stream()` — the CUDA per-thread default
    /// stream — which IS capturable and (unlike `new_stream()`) does
    /// NOT flip the context into multi-stream mode. Multi-stream mode
    /// enables cudarc's automatic cross-stream synchronization
    /// (`is_managing_stream_synchronization()`), and the resulting
    /// `cuStreamWaitEvent` calls invalidate any capture they touch.
    fn capturable_stream(
        dev: &crate::device::GpuDevice,
    ) -> std::sync::Arc<cudarc::driver::safe::CudaStream> {
        dev.context().per_thread_stream()
    }

    /// Smoke test: capturing zero work on an active capture stream still
    /// produces a valid (empty) graph per the CUDA spec. `cuStreamEnd
    /// Capture` returns null only on INVALIDATED captures, not on
    /// empty ones. Confirms the capture/end_capture pair runs without
    /// driver error.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_empty_capture_yields_graph() {
        let dev = require_gpu!();
        let stream = capturable_stream(&dev);
        let graph = capture_graph(&stream, |_s| Ok(())).unwrap();
        let graph = graph.expect("empty capture should produce an empty graph");
        graph
            .launch()
            .expect("empty graph replay should succeed (no work)");
    }

    /// Capturing a `memset_zeros` then replaying it produces the same
    /// device state as direct dispatch. Catches the most basic capture/
    /// replay correctness issue.
    ///
    /// Note: buffers used inside the captured region must be allocated
    /// on the SAME stream that captures, otherwise cudarc's automatic
    /// stream-synchronization tracking inserts a cross-stream
    /// wait-on-event that invalidates the capture (cudarc 0.19's
    /// `device_ptr_mut` checks `is_managing_stream_synchronization`).
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_capture_replay_memset_parity() {
        let dev = require_gpu!();
        let stream = capturable_stream(&dev);

        // Allocate via the capturable stream (not dev.alloc_zeros, which
        // would tie the buffer to the NULL stream and produce a
        // cross-stream wait inside the capture region).
        let mut buf_direct = stream.alloc_zeros::<f32>(1024).unwrap();
        stream.memset_zeros(&mut buf_direct).unwrap();
        stream.synchronize().unwrap();
        let direct = stream.clone_dtoh(&buf_direct).unwrap();
        assert!(direct.iter().all(|&x| x == 0.0));

        let mut buf_graph = stream.alloc_zeros::<f32>(1024).unwrap();
        // Seed with non-zero data so we can prove the captured memset
        // actually ran during replay.
        let one_pattern: Vec<f32> = vec![1.0; 1024];
        stream.memcpy_htod(&one_pattern, &mut buf_graph).unwrap();
        stream.synchronize().unwrap();

        let graph = capture_graph(&stream, |s| {
            s.memset_zeros(&mut buf_graph)
                .map_err(|e| GpuError::CudaError(format!("memset_zeros in capture: {e}")))
        })
        .unwrap();
        let graph = graph.expect("memset capture should produce a graph");
        graph
            .launch()
            .map_err(|e| GpuError::CudaError(format!("graph.launch: {e}")))
            .unwrap();
        stream.synchronize().unwrap();
        let replay = stream.clone_dtoh(&buf_graph).unwrap();
        assert_eq!(direct, replay);
    }

    /// The capture contract, on real hardware: an allocation inside the
    /// capture region is refused **by name**, and the thread is left disarmed.
    ///
    /// This is the arm the CPU tests in `capture_guard_tests.rs` cannot reach —
    /// they exercise the state machine, this proves it is actually wired to
    /// `begin_capture` and to `GpuDevice`. Before the guard, this same code
    /// returned `Ok(None)`: CUDA invalidated the capture, `end_capture` handed
    /// back no graph, and the only symptom was Harmony logging "capture
    /// produced no graph" and running slow for the rest of the call.
    ///
    /// Note what is deliberately *not* refused: `test_capture_replay_memset_
    /// parity` above allocates on the capture stream immediately before
    /// `capture_graph`, which is legal and must stay legal — the contract is
    /// about the region, not about the stream.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn an_allocation_inside_a_capture_region_is_refused_by_name() {
        let dev = require_gpu!();
        let stream = capturable_stream(&dev);
        // Kernels inside a capture run on the capture stream, so the device
        // handle a captured region would hold is this one.
        let dev_pts = dev.with_stream(stream.clone());

        let err = match capture_graph(&stream, |_s| dev_pts.alloc_zeros::<f32>(16).map(|_| ())) {
            Err(e) => e,
            Ok(_) => panic!("an allocation inside the capture region must be refused"),
        };
        assert!(
            matches!(err, GpuError::CaptureViolation(_)),
            "expected CaptureViolation, got {err:?}"
        );
        assert!(
            err.to_string().contains("GpuDevice::alloc_zeros"),
            "the refused operation must be named: {err}"
        );

        // The region is over, so the same call must now succeed. A guard that
        // failed to disarm here would break every later allocation on this
        // thread, which is a far worse failure than the one it prevents.
        dev.alloc_zeros::<f32>(16)
            .expect("the guard must disarm once the capture region ends");
    }

    /// The other half: a host sync inside the region is refused too, and a
    /// legal capture on the same stream still works afterwards. Without the
    /// second half this test would pass against a guard that armed and never
    /// disarmed.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn a_host_sync_inside_a_capture_region_is_refused_and_a_later_capture_still_works() {
        let dev = require_gpu!();
        let stream = capturable_stream(&dev);
        let dev_pts = dev.with_stream(stream.clone());

        let err = match capture_graph(&stream, |_s| dev_pts.synchronize()) {
            Err(e) => e,
            Ok(_) => panic!("a host sync inside the capture region must be refused"),
        };
        assert!(
            err.to_string().contains("GpuDevice::synchronize"),
            "the refused operation must be named: {err}"
        );

        let graph = capture_graph(&stream, |_s| Ok(()))
            .expect("a legal capture after a refused one must still work")
            .expect("an empty capture yields an empty graph");
        graph.launch().expect("empty graph replay");
    }

    /// `cuda_graphs_enabled()` honours the in-process override (used by
    /// parity tests to flip the kill switch without restarting). The
    /// shell-level `SCX_DISABLE_CUDA_GRAPHS=1` path is exercised by the
    /// G10.0 verification block; the OnceLock-cached env read can't be
    /// flipped mid-process.
    #[test]
    fn test_cuda_graphs_enabled_override_round_trip() {
        let prev = set_cuda_graphs_enabled_override(Some(false));
        assert!(!cuda_graphs_enabled(), "override Some(false) should win");

        set_cuda_graphs_enabled_override(Some(true));
        assert!(cuda_graphs_enabled(), "override Some(true) should win");

        // Restore prior state so this test doesn't leak into siblings.
        set_cuda_graphs_enabled_override(prev);
    }
}
