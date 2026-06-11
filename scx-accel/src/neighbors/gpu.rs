//! kNN GPU availability probe (behind the `gpu` feature).
//!
//! Device-resident CAGRA kNN runs through the fused PCA→kNN pipeline
//! (`super::super::fused`); this module only exposes the runtime probe.

/// Check whether GPU kNN via cuVS CAGRA is available.
///
/// Returns `true` if the `gpu` feature is enabled, a CUDA GPU is detected,
/// AND the cuVS library (`libcuvs_c.so`) is loadable at runtime.
#[cfg(feature = "gpu")]
pub fn cuvs_available() -> bool {
    scx_gpu::cuvs_available()
}

// NOTE: the standalone host-bounce `build_knn_graph_gpu` entry point was
// removed as dead code. After the rapids-singlecell transition
// no production path called it — the fused PCA→kNN pipeline (`fused.rs`) uses
// the device-resident `scx_gpu::gpu_knn_cagra_device` to keep the embedding on
// the GPU across the handoff, and in-VRAM kNN otherwise routes to
// rapids-singlecell. The host-bounce `scx_gpu::gpu_knn_cagra` it wrapped was
// removed alongside it.
