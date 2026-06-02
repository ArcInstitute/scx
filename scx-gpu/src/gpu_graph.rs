//! CUDA Graph capture/replay infrastructure for iteration-heavy GPU loops.
//!
//! Iteration-heavy stages (PCA power, Harmony k-means, UMAP SGD, GPU DE
//! chunk loops) launch many small kernels per iteration. Per-launch
//! overhead (~5–20 µs each) dominates on small inputs. `cudaStreamBegin/
//! EndCapture` records a stable kernel sequence into a `cudaGraph_t` once;
//! subsequent replays via `cuGraphLaunch` amortize per-launch latency.
//!
//! ## Prerequisites
//!
//! - Captured regions MUST NOT allocate device memory or do host syncs —
//!   capture mode rejects these APIs. G2's `GpuPcaScratch` /
//!   `GpuDeChunkScratch` provide stable pre-grown scratch addresses; the
//!   stable-buffer prerequisite is already in place.
//! - Capture MUST run on a non-NULL stream. cudarc's
//!   `CudaContext::default_stream()` returns the NULL stream (`cu_stream
//!   = std::ptr::null_mut()`) which CUDA rejects for capture
//!   (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`). Production call sites
//!   must capture on a stream obtained via
//!   `CudaContext::new_stream()` (an explicit non-blocking stream) or
//!   `CudaContext::per_thread_stream()` (the per-thread default, also
//!   capturable). Each call site is responsible for any cross-stream
//!   synchronization needed before/after replay.
//!
//! ## Kill switch
//!
//! Set `SCX_DISABLE_CUDA_GRAPHS=1` to bypass capture at every call site;
//! callers fall back to direct kernel dispatch with no behavioural change.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use cudarc::driver::safe::{CudaGraph, CudaStream};
use cudarc::driver::sys;

use crate::error::GpuError;

/// Shape signature used to key the graph cache. A captured graph references
/// kernel arguments (buffer pointers + scalar args) baked in at capture
/// time. Whenever any of those would change, the cache must miss and
/// re-capture.
#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
pub enum GraphKey {
    /// Randomized PCA power iteration body (rmatmat → QR → matmat → QR).
    PcaPower {
        n_obs: u32,
        n_vars: u32,
        k: u32,
        n_shards: u32,
    },
    /// Harmony k-means inner sub-iteration (distances → softmax → O/E).
    HarmonyKmeans {
        n_obs: u32,
        n_clusters: u32,
        n_pcs: u32,
        n_batch_levels: u32,
    },
    /// UMAP SGD one-epoch capture. Alpha varies per replay via
    /// `cuGraphExecKernelNodeSetParams_v2`; not part of the key.
    UmapEpoch {
        n_obs: u32,
        n_components: u32,
        n_edges: u64,
    },
    /// GPU DE per-chunk kernel sequence (scatter → sort → searchsort →
    /// tie → pvalues, summed over test groups). One graph per
    /// `(chunk_size_actual, n_ref, n_g_max, n_test_groups, mode)`.
    ///
    /// `mode` distinguishes the three captureable DE flavours, each of
    /// which has a different per-test-group kernel count:
    /// - `0` — `pdex_ref` (scatter+searchsort_u+sort+combined_tie+pvalues
    ///   +stage_u+stage_p).
    /// - `1` — wilcoxon ref-mode (scatter+searchsort_u+sort+combined_tie
    ///   +stage_u+stage_tie).
    /// - `2` — wilcoxon 1-vs-rest (scatter+searchsort_ranksum+stage_u).
    ///
    /// `n_ref` is `n_ref` for pdex_ref / wilcoxon-ref, and `n_pool`
    /// (= n_obs in 1-vs-rest, since the pool is all cells) for the
    /// 1-vs-rest variant; the kernels parameterise on this size the
    /// same way regardless of mode.
    DeChunk {
        chunk_size: u32,
        n_ref: u32,
        n_g_max: u32,
        n_test_groups: u32,
        mode: u8,
    },
}

/// Counters surfaced through `GpuGraphCache::metrics()`.
#[derive(Debug, Default)]
pub struct GpuGraphMetrics {
    captures: AtomicU64,
    replays: AtomicU64,
    capture_failures: AtomicU64,
    capture_overhead_ns: AtomicU64,
}

impl GpuGraphMetrics {
    pub fn captures(&self) -> u64 {
        self.captures.load(Ordering::Relaxed)
    }
    pub fn replays(&self) -> u64 {
        self.replays.load(Ordering::Relaxed)
    }
    pub fn capture_failures(&self) -> u64 {
        self.capture_failures.load(Ordering::Relaxed)
    }
    pub fn capture_overhead_ns(&self) -> u64 {
        self.capture_overhead_ns.load(Ordering::Relaxed)
    }

    /// Replay/(replay+capture) ratio in [0.0, 1.0]. Returns 0.0 if neither
    /// has happened yet.
    pub fn hit_rate(&self) -> f64 {
        let r = self.replays() as f64;
        let c = self.captures() as f64;
        if r + c == 0.0 {
            0.0
        } else {
            r / (r + c)
        }
    }

    pub fn reset(&self) {
        self.captures.store(0, Ordering::Relaxed);
        self.replays.store(0, Ordering::Relaxed);
        self.capture_failures.store(0, Ordering::Relaxed);
        self.capture_overhead_ns.store(0, Ordering::Relaxed);
    }
}

/// Shape-keyed cache of captured `cudaGraph_t`. One cache lives on each
/// `GpuDevice`; cache entries persist across `gpu_randomized_pca` /
/// GPU DE / etc. calls on the same device, so a second
/// run with the same shape signature replays the cached graph rather than
/// recapturing.
///
/// `CudaGraph` is `!Sync` per cudarc's docs. Callers must serialise access
/// (the surrounding `RefCell<GpuGraphCache>` accessor on `GpuDevice` is
/// what enforces this).
pub struct GpuGraphCache {
    entries: HashMap<GraphKey, CudaGraph>,
    metrics: GpuGraphMetrics,
}

impl Default for GpuGraphCache {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuGraphCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            metrics: GpuGraphMetrics::default(),
        }
    }

    pub fn metrics(&self) -> &GpuGraphMetrics {
        &self.metrics
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all cached graphs. Each `CudaGraph::drop` runs
    /// `cuGraphExecDestroy` + `cuGraphDestroy` automatically.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Look up a cached graph by key, or capture a fresh one by running
    /// `build` between `cuStreamBeginCapture` / `cuStreamEndCapture`.
    ///
    /// `build` is the to-be-captured kernel sequence. It MUST run only
    /// kernel launches and async memcpy ops on `stream`. It MUST NOT
    /// allocate device memory, call `dev.synchronize()`, or issue host
    /// syncs — capture mode rejects these.
    ///
    /// Returns a launchable `&CudaGraph` (the caller then calls
    /// `graph.launch()`). On capture failure, returns the underlying
    /// `GpuError` and records the failure in `metrics`.
    ///
    /// When `SCX_DISABLE_CUDA_GRAPHS=1`, this method does not capture —
    /// callers must dispatch the non-graph path via the
    /// [`cuda_graphs_enabled`] check before calling. (We surface the
    /// check at the dispatch site rather than papering over it here, so
    /// failed-capture cases route through the same fallback path as the
    /// kill switch.)
    pub fn get_or_capture<F>(
        &mut self,
        key: GraphKey,
        stream: &Arc<CudaStream>,
        build: F,
    ) -> Result<&CudaGraph, GpuError>
    where
        F: FnOnce(&Arc<CudaStream>) -> Result<(), GpuError>,
    {
        if self.entries.contains_key(&key) {
            self.metrics.replays.fetch_add(1, Ordering::Relaxed);
            return Ok(self.entries.get(&key).unwrap());
        }

        let start = Instant::now();
        let graph = capture_graph(stream, build)?;
        let elapsed = start.elapsed().as_nanos() as u64;
        self.metrics
            .capture_overhead_ns
            .fetch_add(elapsed, Ordering::Relaxed);
        match graph {
            Some(g) => {
                self.metrics.captures.fetch_add(1, Ordering::Relaxed);
                Ok(self.entries.entry(key).or_insert(g))
            }
            None => {
                self.metrics
                    .capture_failures
                    .fetch_add(1, Ordering::Relaxed);
                Err(GpuError::CudaError(
                    "stream capture returned no graph (no work captured?)".into(),
                ))
            }
        }
    }
}

/// Run `build` between `begin_capture` / `end_capture` and return the
/// instantiated graph. Free-standing helper used by [`GpuGraphCache`] and
/// by tests that want to capture a one-shot graph without going through
/// the cache.
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

    let build_result = build(stream);

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
/// Call sites should check this BEFORE invoking
/// [`GpuGraphCache::get_or_capture`]; when graphs are disabled they
/// fall back to direct kernel dispatch.
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

/// Enumerate the kernel nodes inside a captured `CUgraph`.
/// Locate the single kernel-node handle so subsequent
/// replays can update its scalar params via
/// [`exec_kernel_node_set_params`] without recapturing.
///
/// Returns nodes in the order CUDA returns them; for graphs captured
/// from a single `launch_builder().launch(cfg)` call there is exactly
/// one node.
///
/// # Safety
///
/// `graph` must be a valid, non-destroyed `CUgraph` (typically
/// `CudaGraph::cu_graph()`).
pub unsafe fn graph_kernel_nodes(graph: sys::CUgraph) -> Result<Vec<sys::CUgraphNode>, GpuError> {
    // First call with null nodes pointer → driver writes the node
    // count into `num_nodes`. Second call passes a buffer of that
    // size and the driver fills it in. Standard two-pass CUDA
    // enumeration pattern.
    let mut num_nodes: usize = 0;
    let r1 = sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut num_nodes);
    if r1 != sys::CUresult::CUDA_SUCCESS {
        return Err(GpuError::CudaError(format!(
            "cuGraphGetNodes count probe failed: {r1:?}"
        )));
    }
    let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); num_nodes];
    let r2 = sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut num_nodes);
    if r2 != sys::CUresult::CUDA_SUCCESS {
        return Err(GpuError::CudaError(format!(
            "cuGraphGetNodes failed: {r2:?}"
        )));
    }
    nodes.truncate(num_nodes);
    // Filter to kernel nodes (the only type we currently care about).
    let mut kernel_nodes = Vec::with_capacity(nodes.len());
    for node in nodes {
        let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
        let r = sys::cuGraphNodeGetType(node, &mut ty);
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(GpuError::CudaError(format!(
                "cuGraphNodeGetType failed: {r:?}"
            )));
        }
        if matches!(ty, sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL) {
            kernel_nodes.push(node);
        }
    }
    Ok(kernel_nodes)
}

/// Read the current kernel-node params from a captured node. Used by
/// G10.3 (UMAP) to extract the `CUfunction` and grid/block dimensions
/// that the capture baked in, so subsequent
/// [`exec_kernel_node_set_params`] calls can pass them unchanged while
/// updating only the user-controlled `kernelParams` pointer array.
///
/// # Safety
///
/// `node` must be a valid kernel node returned by
/// [`graph_kernel_nodes`].
pub unsafe fn read_kernel_node_params(
    node: sys::CUgraphNode,
) -> Result<sys::CUDA_KERNEL_NODE_PARAMS, GpuError> {
    let mut params: sys::CUDA_KERNEL_NODE_PARAMS = std::mem::zeroed();
    let r = sys::cuGraphKernelNodeGetParams_v2(node, &mut params);
    if r != sys::CUresult::CUDA_SUCCESS {
        return Err(GpuError::CudaError(format!(
            "cuGraphKernelNodeGetParams_v2 failed: {r:?}"
        )));
    }
    Ok(params)
}

/// FFI shim: `cuGraphExecKernelNodeSetParams_v2` — used by G10.3 (UMAP
/// epoch capture) to swap the per-epoch alpha + epoch_i32 scalars
/// between replays without re-capture. cudarc 0.19 exposes the raw FFI
/// at `cudarc::driver::sys::cuGraphExecKernelNodeSetParams_v2` but does
/// not wrap it as safe Rust.
///
/// # Safety
///
/// - `graph_exec` must be a valid, non-destroyed `CUgraphExec`.
/// - `node` must be a kernel node within `graph_exec`'s graph.
/// - `params` must describe the same kernel function and grid/block
///   shape as at capture time; only kernel argument values may change.
/// - The kernel arguments referenced by `params.kernelParams` must
///   remain valid until the next launch using these params completes.
///
/// Returns a `GpuError::CudaError` on driver failure.
pub unsafe fn exec_kernel_node_set_params(
    graph_exec: sys::CUgraphExec,
    node: sys::CUgraphNode,
    params: &sys::CUDA_KERNEL_NODE_PARAMS,
) -> Result<(), GpuError> {
    let r = sys::cuGraphExecKernelNodeSetParams_v2(graph_exec, node, params);
    if r == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(GpuError::CudaError(format!(
            "cuGraphExecKernelNodeSetParams_v2 failed: {r:?}"
        )))
    }
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

    /// GpuGraphCache: capturing once and re-asking for the same key
    /// returns the cached entry (replay counter increments, captures
    /// stays at 1).
    #[test]
    fn test_cache_replays_on_repeat_key() {
        let dev = require_gpu!();
        let stream = capturable_stream(&dev);
        let mut cache = GpuGraphCache::new();

        let mut buf = stream.alloc_zeros::<f32>(64).unwrap();
        let key = GraphKey::PcaPower {
            n_obs: 64,
            n_vars: 16,
            k: 4,
            n_shards: 1,
        };

        // First call: captures.
        let g = cache
            .get_or_capture(key, &stream, |s| {
                s.memset_zeros(&mut buf)
                    .map_err(|e| GpuError::CudaError(format!("memset: {e}")))
            })
            .unwrap();
        g.launch()
            .map_err(|e| GpuError::CudaError(format!("launch: {e}")))
            .unwrap();
        assert_eq!(cache.metrics().captures(), 1);
        assert_eq!(cache.metrics().replays(), 0);

        // Second call with same key: replays from cache. The build
        // closure must NOT run — assert that by panicking if it does.
        let g = cache
            .get_or_capture::<_>(key, &stream, |_s| {
                panic!("build closure must not run on cache hit");
            })
            .unwrap();
        g.launch()
            .map_err(|e| GpuError::CudaError(format!("launch: {e}")))
            .unwrap();
        assert_eq!(cache.metrics().captures(), 1);
        assert_eq!(cache.metrics().replays(), 1);
        assert!(cache.metrics().hit_rate() > 0.0);
    }
}
