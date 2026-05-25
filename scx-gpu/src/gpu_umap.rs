//! GPU-accelerated UMAP embedding via native CUDA SGD kernel.
//!
//! Provides [`gpu_umap_native`] which computes a UMAP embedding entirely on GPU,
//! except for spectral initialization and `find_ab_params` which are done on CPU
//! (these are tiny relative to the SGD optimization loop).
//!
//! # Pipeline
//!
//! 1. Compute UMAP curve parameters `(a, b)` on CPU via grid search
//! 2. Build edge list + epoch schedule on CPU
//! 3. Initialize embedding (spectral or random) on CPU
//! 4. Upload embedding, edges, and epoch schedule to GPU (once)
//! 5. For each epoch: launch `umap_sgd_kernel` — kernel filters active edges on-device
//! 6. Download final embedding to host
//!
//! # Non-determinism
//!
//! GPU UMAP results differ from CPU UMAP due to:
//! - `atomicAdd` race conditions (intentional, matches cuML)
//! - f32 precision (CPU uses f64)
//! - Different RNG for negative sampling
//!
//! Validate via embedding quality metrics (Trustworthiness, cluster separation),
//! not exact coordinate matching.

use cudarc::driver::safe::{CudaStream, LaunchConfig};
use cudarc::driver::sys;
use cudarc::driver::{DevicePtr, DevicePtrMut, PushKernelArg};
use std::sync::Arc;

use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::gpu_graph::{
    capture_graph, cuda_graphs_enabled, exec_kernel_node_set_params, graph_kernel_nodes,
    read_kernel_node_params,
};

// ---------------------------------------------------------------------------
// PTX source (compiled by build.rs from kernels/umap_sgd.cu)
// ---------------------------------------------------------------------------

const UMAP_SGD_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/umap_sgd.ptx"));

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// GPU UMAP embedding result.
#[derive(Debug, Clone)]
pub struct GpuUmapResult {
    /// Flat row-major embedding (n_obs × n_components), f32.
    pub embedding: Vec<f32>,
    /// Number of observations.
    pub n_obs: usize,
    /// Number of components (typically 2).
    pub n_components: usize,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute UMAP embedding on GPU via native CUDA SGD kernel.
///
/// # Arguments
///
/// * `dev` — GPU device handle
/// * `conn_indptr` — CSR row pointers for the connectivity matrix (n_obs + 1)
/// * `conn_indices` — CSR column indices for the connectivity matrix
/// * `conn_data` — CSR values (connectivity strengths in [0, 1])
/// * `n_obs` — Number of observations
/// * `n_components` — Output dimensions (default: 2)
/// * `n_epochs` — Number of SGD epochs (default: 200)
/// * `min_dist` — Minimum distance in embedding (default: 0.1)
/// * `spread` — Spread of embedded points (default: 1.0)
/// * `negative_sample_rate` — Negative samples per positive edge (default: 5)
/// * `learning_rate` — Initial learning rate (default: 1.0)
/// * `seed` — Random seed for reproducibility
/// * `init_coords` — Optional initial coordinates (n_obs × n_components, f32, row-major).
///   If None, random initialization is used. (Spectral init should be computed by the
///   caller and passed in — see `scx-accel/src/umap.rs::spectral_init`)
///
/// # Returns
///
/// `GpuUmapResult` with embedding coordinates on host.
#[allow(clippy::too_many_arguments)]
pub fn gpu_umap_native(
    dev: &GpuDevice,
    conn_indptr: &[i64],
    conn_indices: &[i32],
    conn_data: &[f64],
    n_obs: usize,
    n_components: usize,
    n_epochs: usize,
    min_dist: f32,
    spread: f32,
    negative_sample_rate: usize,
    learning_rate: f32,
    seed: u64,
    init_coords: Option<&[f32]>,
) -> Result<GpuUmapResult, GpuError> {
    // --- Validate inputs ---
    if n_obs == 0 {
        return Err(GpuError::ShapeMismatch {
            expected: "n_obs > 0".to_string(),
            got: "0".to_string(),
        });
    }
    if conn_indptr.len() != n_obs + 1 {
        return Err(GpuError::ShapeMismatch {
            expected: format!("conn_indptr length = n_obs + 1 = {}", n_obs + 1),
            got: format!("{}", conn_indptr.len()),
        });
    }
    if n_components == 0 {
        return Err(GpuError::ShapeMismatch {
            expected: "n_components > 0".to_string(),
            got: "0".to_string(),
        });
    }
    if n_epochs == 0 {
        return Err(GpuError::ShapeMismatch {
            expected: "n_epochs > 0".to_string(),
            got: "0".to_string(),
        });
    }

    // --- CPU-side: find a, b parameters ---
    let (a, b) = find_ab_params(spread as f64, min_dist as f64);
    let a = a as f32;
    let b = b as f32;

    // --- CPU-side: build edge list + epoch schedule ---
    let epochs_per_sample = compute_epochs_per_sample(conn_data, n_epochs);

    // Build flat edge list
    let mut head = Vec::with_capacity(conn_data.len());
    let mut tail = Vec::with_capacity(conn_data.len());
    for i in 0..n_obs {
        let start = conn_indptr[i] as usize;
        let end = conn_indptr[i + 1] as usize;
        for &col in &conn_indices[start..end] {
            head.push(i as i32);
            tail.push(col);
        }
    }
    let n_edges = head.len();

    // --- CPU-side: initialize embedding ---
    let embedding_f32: Vec<f32> = if let Some(coords) = init_coords {
        if coords.len() != n_obs * n_components {
            return Err(GpuError::ShapeMismatch {
                expected: format!(
                    "init_coords length = n_obs × n_components = {} × {} = {}",
                    n_obs,
                    n_components,
                    n_obs * n_components
                ),
                got: format!("{}", coords.len()),
            });
        }
        coords.to_vec()
    } else {
        // Random initialization (small Gaussian noise)
        random_init_f32(n_obs, n_components, seed)
    };

    // --- Upload embedding to GPU ---
    let mut d_embedding = dev.htod_copy(&embedding_f32)?;

    // --- Upload edge list + scheduling arrays to GPU (once) ---
    // Edge filtering happens on-device: each thread checks its edge's
    // epoch_of_next_sample and skips inactive edges. This eliminates
    // per-epoch host→device edge list uploads (was 200 × O(n_edges) uploads).
    let d_head = dev.htod_copy(&head)?;
    let d_tail = dev.htod_copy(&tail)?;

    // Convert scheduling arrays to f32 for GPU (f64 precision not needed
    // for epoch scheduling — the comparison is epoch_of_next_sample <= epoch).
    let epochs_per_sample_f32: Vec<f32> = epochs_per_sample.iter().map(|&x| x as f32).collect();
    let epoch_of_next_sample_f32: Vec<f32> = epochs_per_sample
        .clone()
        .iter()
        .map(|&x| x as f32)
        .collect();
    let d_epochs_per_sample = dev.htod_copy(&epochs_per_sample_f32)?;
    let mut d_epoch_of_next_sample = dev.htod_copy(&epoch_of_next_sample_f32)?;

    // --- Load UMAP SGD kernel ---
    let module = dev.load_module_cached(UMAP_SGD_PTX)?;
    let func = module
        .load_function("umap_sgd_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("load umap_sgd_kernel: {e}")))?;

    // --- SGD optimization loop ---
    let n_obs_i32 = n_obs as i32;
    let n_edges_i32 = n_edges as i32;
    let n_components_i32 = n_components as i32;
    let neg_rate_i32 = negative_sample_rate as i32;

    let block_size = 256u32;
    // Validate edge count fits in CUDA grid dimensions.
    // Max grid_dim.x is 2^31-1 (compute capability >= 3.0). With block_size=256,
    // this supports ~549 billion edges — far beyond any real dataset.
    if n_edges > u32::MAX as usize {
        return Err(GpuError::KernelLaunchFailed(format!(
            "UMAP edge count {} exceeds u32::MAX",
            n_edges
        )));
    }
    let grid_size = (n_edges as u32).div_ceil(block_size);
    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // Graph-replay path. The SGD loop is a single
    // `umap_sgd_kernel` per epoch; capturing it once and replaying with
    // per-epoch `alpha` / `epoch_i32` updates via
    // `cuGraphExecKernelNodeSetParams_v2` amortizes the per-launch
    // dispatch cost across all `n_epochs` iterations. Falls back to
    // the per-epoch launch loop on capture failure or
    // `SCX_DISABLE_CUDA_GRAPHS=1`. The fallback path is functionally
    // identical to the pre-G10.3 implementation.
    let graph_outcome = if cuda_graphs_enabled() {
        run_sgd_loop_with_graph(
            dev,
            &func,
            cfg,
            &mut d_embedding,
            &d_head,
            &d_tail,
            &mut d_epoch_of_next_sample,
            &d_epochs_per_sample,
            n_edges_i32,
            n_obs_i32,
            n_components_i32,
            a,
            b,
            learning_rate,
            neg_rate_i32,
            seed,
            n_epochs,
        )
    } else {
        Err(GpuError::CudaError("graphs disabled".into()))
    };

    if graph_outcome.is_err() {
        run_sgd_loop_direct(
            dev,
            &func,
            cfg,
            &mut d_embedding,
            &d_head,
            &d_tail,
            &mut d_epoch_of_next_sample,
            &d_epochs_per_sample,
            n_edges_i32,
            n_obs_i32,
            n_components_i32,
            a,
            b,
            learning_rate,
            neg_rate_i32,
            seed,
            n_epochs,
        )?;
    }

    // --- Download result ---
    dev.synchronize()?;
    let embedding = dev.dtoh_copy(&d_embedding)?;

    Ok(GpuUmapResult {
        embedding,
        n_obs,
        n_components,
    })
}

// ---------------------------------------------------------------------------
// SGD loop driver — graph-capture path and direct fallback
// ---------------------------------------------------------------------------

/// Direct per-epoch kernel-launch path. Identical to the pre-G10.3
/// implementation; used when `SCX_DISABLE_CUDA_GRAPHS=1` or when graph
/// capture fails (returning a `GpuError` to the outer loop, which then
/// dispatches this fallback).
#[allow(clippy::too_many_arguments)]
fn run_sgd_loop_direct(
    dev: &GpuDevice,
    func: &cudarc::driver::safe::CudaFunction,
    cfg: LaunchConfig,
    d_embedding: &mut cudarc::driver::safe::CudaSlice<f32>,
    d_head: &cudarc::driver::safe::CudaSlice<i32>,
    d_tail: &cudarc::driver::safe::CudaSlice<i32>,
    d_epoch_of_next_sample: &mut cudarc::driver::safe::CudaSlice<f32>,
    d_epochs_per_sample: &cudarc::driver::safe::CudaSlice<f32>,
    n_edges_i32: i32,
    n_obs_i32: i32,
    n_components_i32: i32,
    a: f32,
    b: f32,
    learning_rate: f32,
    neg_rate_i32: i32,
    seed: u64,
    n_epochs: usize,
) -> Result<(), GpuError> {
    for epoch in 0..n_epochs {
        let alpha = learning_rate * (1.0 - epoch as f32 / n_epochs as f32);
        let epoch_i32 = epoch as i32;

        let _events = unsafe {
            dev.stream()
                .launch_builder(func)
                .arg(&mut *d_embedding)
                .arg(d_head)
                .arg(d_tail)
                .arg(&mut *d_epoch_of_next_sample)
                .arg(d_epochs_per_sample)
                .arg(&n_edges_i32)
                .arg(&n_obs_i32)
                .arg(&n_components_i32)
                .arg(&a)
                .arg(&b)
                .arg(&alpha)
                .arg(&neg_rate_i32)
                .arg(&seed)
                .arg(&epoch_i32)
                .launch(cfg)
        }
        .map_err(|e| GpuError::KernelLaunchFailed(format!("umap_sgd_kernel: {e}")))?;
    }
    Ok(())
}

/// Graph-capture path. Capture one `umap_sgd_kernel` launch into a
/// `CudaGraph`, then replay it `n_epochs - 1` more times, updating
/// `alpha` and `epoch_i32` per launch via
/// `cuGraphExecKernelNodeSetParams_v2`. Buffer pointers, `n_edges`,
/// `n_obs`, `n_components`, `a`, `b`, `seed`, and `negative_sample_rate`
/// are constant across the schedule and reused unchanged.
///
/// Capture runs on `per_thread_stream()` — the per-thread CUDA default
/// stream, which is capturable and (unlike `new_stream()`) does not
/// flip cudarc into multi-stream-synchronization mode (which would
/// inject cross-stream waits that invalidate the capture; see
/// `gpu_graph` module docs).
///
/// All buffer pointers are read once via the `DevicePtr` traits before
/// capture and stored in stable host-side storage so the `kernelParams`
/// array we pass to `cuGraphExecKernelNodeSetParams_v2` remains valid
/// across replays.
///
/// On any failure (capture invalidated, no kernel node found, FFI
/// error), this returns a `GpuError` and the caller falls back to
/// [`run_sgd_loop_direct`].
#[allow(clippy::too_many_arguments)]
fn run_sgd_loop_with_graph(
    dev: &GpuDevice,
    func: &cudarc::driver::safe::CudaFunction,
    cfg: LaunchConfig,
    d_embedding: &mut cudarc::driver::safe::CudaSlice<f32>,
    d_head: &cudarc::driver::safe::CudaSlice<i32>,
    d_tail: &cudarc::driver::safe::CudaSlice<i32>,
    d_epoch_of_next_sample: &mut cudarc::driver::safe::CudaSlice<f32>,
    d_epochs_per_sample: &cudarc::driver::safe::CudaSlice<f32>,
    n_edges_i32: i32,
    n_obs_i32: i32,
    n_components_i32: i32,
    a: f32,
    b: f32,
    learning_rate: f32,
    neg_rate_i32: i32,
    seed: u64,
    n_epochs: usize,
) -> Result<(), GpuError> {
    // Drain prior NULL-stream work (the htod uploads happen on
    // dev.stream() before we get here). Kernels on per_thread_stream
    // then start from a clean baseline; we don't get cross-stream
    // hazards from uncompleted uploads.
    dev.synchronize()?;

    let pts: Arc<CudaStream> = dev.context().per_thread_stream();

    // First launch: also serves as the capture template. We pass an
    // alpha matching epoch 0 (template values are overwritten before
    // the first REPLAY of the captured graph, but we run epoch 0 here
    // directly as part of capture).
    let alpha_initial = learning_rate;
    let epoch_initial: i32 = 0;

    let graph = capture_graph(&pts, |stream| {
        let _events = unsafe {
            stream
                .launch_builder(func)
                .arg(&mut *d_embedding)
                .arg(d_head)
                .arg(d_tail)
                .arg(&mut *d_epoch_of_next_sample)
                .arg(d_epochs_per_sample)
                .arg(&n_edges_i32)
                .arg(&n_obs_i32)
                .arg(&n_components_i32)
                .arg(&a)
                .arg(&b)
                .arg(&alpha_initial)
                .arg(&neg_rate_i32)
                .arg(&seed)
                .arg(&epoch_initial)
                .launch(cfg)
        }
        .map_err(|e| GpuError::KernelLaunchFailed(format!("umap_sgd_kernel capture: {e}")))?;
        Ok(())
    })?;
    let graph =
        graph.ok_or_else(|| GpuError::CudaError("umap_sgd capture returned no graph".into()))?;

    // Enumerate captured nodes; we expect exactly one kernel node
    // (the single SGD launch). Defensive — bail out otherwise so the
    // caller falls back rather than fingerprinting the wrong node.
    let kernel_nodes = unsafe { graph_kernel_nodes(graph.cu_graph()) }?;
    if kernel_nodes.len() != 1 {
        return Err(GpuError::CudaError(format!(
            "expected 1 captured kernel node, got {}",
            kernel_nodes.len()
        )));
    }
    let node = kernel_nodes[0];

    // Read the captured node's params so we can preserve func + grid
    // + block dims unchanged across our updates. The cu_function
    // handle inside cudarc's `CudaFunction` is `pub(crate)`, so we
    // can't construct it ourselves; reading from the captured node
    // is the canonical way to obtain it.
    let captured_params = unsafe { read_kernel_node_params(node) }?;

    // Build host-side storage for ALL 14 kernel arguments. Pointer
    // stability matters: `cuGraphExecKernelNodeSetParams_v2` reads
    // the values pointed to by `kernelParams` at the time of the call,
    // so these locals only need to outlive the set_params call — but
    // we keep them on this function's stack frame for the whole loop,
    // updating alpha/epoch_i32 in-place each iteration.
    let (d_embedding_ptr, _g0) = d_embedding.device_ptr_mut(&pts);
    let (d_head_ptr, _g1) = d_head.device_ptr(&pts);
    let (d_tail_ptr, _g2) = d_tail.device_ptr(&pts);
    let (d_epoch_of_next_ptr, _g3) = d_epoch_of_next_sample.device_ptr_mut(&pts);
    let (d_epochs_per_sample_ptr, _g4) = d_epochs_per_sample.device_ptr(&pts);

    // Boxed so addresses are heap-stable across stack moves; the
    // kernelParams pointer array below stores pointers into this Box.
    let mut storage = Box::new(UmapKernelArgStorage {
        embedding_ptr: d_embedding_ptr,
        head_ptr: d_head_ptr,
        tail_ptr: d_tail_ptr,
        epoch_of_next_ptr: d_epoch_of_next_ptr,
        epochs_per_sample_ptr: d_epochs_per_sample_ptr,
        n_edges: n_edges_i32,
        n_obs: n_obs_i32,
        n_components: n_components_i32,
        a,
        b,
        alpha: 0.0,
        neg_rate: neg_rate_i32,
        seed,
        epoch: 0,
    });

    let s = storage.as_mut() as *mut UmapKernelArgStorage;
    // SAFETY: each pointer below references a distinct field of
    // `*s`. The Box keeps the storage on the heap at a stable
    // address; we drop `_g*` guards only after the loop finishes,
    // so the device buffers stay marked as "in use" for the
    // duration. The kernelParams array is built once and reused.
    let mut kernel_params: [*mut std::ffi::c_void; 14] = unsafe {
        [
            &raw mut (*s).embedding_ptr as *mut std::ffi::c_void,
            &raw mut (*s).head_ptr as *mut std::ffi::c_void,
            &raw mut (*s).tail_ptr as *mut std::ffi::c_void,
            &raw mut (*s).epoch_of_next_ptr as *mut std::ffi::c_void,
            &raw mut (*s).epochs_per_sample_ptr as *mut std::ffi::c_void,
            &raw mut (*s).n_edges as *mut std::ffi::c_void,
            &raw mut (*s).n_obs as *mut std::ffi::c_void,
            &raw mut (*s).n_components as *mut std::ffi::c_void,
            &raw mut (*s).a as *mut std::ffi::c_void,
            &raw mut (*s).b as *mut std::ffi::c_void,
            &raw mut (*s).alpha as *mut std::ffi::c_void,
            &raw mut (*s).neg_rate as *mut std::ffi::c_void,
            &raw mut (*s).seed as *mut std::ffi::c_void,
            &raw mut (*s).epoch as *mut std::ffi::c_void,
        ]
    };

    let params = sys::CUDA_KERNEL_NODE_PARAMS {
        func: captured_params.func,
        gridDimX: captured_params.gridDimX,
        gridDimY: captured_params.gridDimY,
        gridDimZ: captured_params.gridDimZ,
        blockDimX: captured_params.blockDimX,
        blockDimY: captured_params.blockDimY,
        blockDimZ: captured_params.blockDimZ,
        sharedMemBytes: captured_params.sharedMemBytes,
        kernelParams: kernel_params.as_mut_ptr(),
        extra: std::ptr::null_mut(),
        kern: std::ptr::null_mut(),
        ctx: std::ptr::null_mut(),
    };

    // `begin_capture` RECORDS the launch into the graph but does NOT
    // execute it on the device — capture is a recording mode, not a
    // run. So `launch_builder().launch(cfg)` inside the closure above
    // produced no GPU-side state change. The full SGD schedule runs
    // as `n_epochs` replays of the graph below, with updated alpha +
    // epoch_i32 each iteration to mirror the direct path's
    // `0..n_epochs` loop.
    for epoch in 0..n_epochs {
        storage.alpha = learning_rate * (1.0 - epoch as f32 / n_epochs as f32);
        storage.epoch = epoch as i32;

        // SAFETY: `graph` owns `cu_graph_exec`; `node` is a valid
        // kernel node within it (verified via `graph_kernel_nodes`
        // above); `params` is constructed in this stack frame and
        // its `kernelParams` array references storage that lives
        // as long as `storage` (this stack frame's heap-allocated
        // Box).
        unsafe {
            exec_kernel_node_set_params(graph.cu_graph_exec(), node, &params)?;
        }
        graph
            .launch()
            .map_err(|e| GpuError::CudaError(format!("graph.launch (umap epoch {epoch}): {e}")))?;
    }

    // Drain the side-stream work before returning to the caller, so
    // the subsequent dtoh of d_embedding on dev.stream() sees the
    // final post-replay state.
    pts.synchronize()
        .map_err(|e| GpuError::CudaError(format!("per-thread-stream sync after umap: {e}")))?;
    Ok(())
}

/// Stable storage for the 14 `umap_sgd_kernel` argument values. Pointers
/// into this struct are passed to
/// `cuGraphExecKernelNodeSetParams_v2`; only `alpha` and `epoch` are
/// mutated between replays.
#[repr(C)]
struct UmapKernelArgStorage {
    embedding_ptr: u64,
    head_ptr: u64,
    tail_ptr: u64,
    epoch_of_next_ptr: u64,
    epochs_per_sample_ptr: u64,
    n_edges: i32,
    n_obs: i32,
    n_components: i32,
    a: f32,
    b: f32,
    alpha: f32,
    neg_rate: i32,
    seed: u64,
    epoch: i32,
}

// ---------------------------------------------------------------------------
// CPU-side helpers — delegated to scx_sparse::umap_math
// ---------------------------------------------------------------------------

use scx_sparse::umap_math::{compute_epochs_per_sample, find_ab_params, random_init_f32};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple kNN-like connectivity graph for testing.
    /// Two clusters of 25 points each with high intra-cluster connectivity.
    fn test_graph() -> (Vec<i64>, Vec<i32>, Vec<f64>, usize) {
        let n = 50;
        let k = 5;
        let mut indptr = vec![0i64; n + 1];
        let mut indices = Vec::new();
        let mut data = Vec::new();

        for i in 0..n {
            let cluster_start = if i < 25 { 0 } else { 25 };
            let cluster_end = if i < 25 { 25 } else { 50 };

            let mut neighbors = Vec::new();
            for j in cluster_start..cluster_end {
                if j != i {
                    neighbors.push(j);
                }
                if neighbors.len() == k {
                    break;
                }
            }

            for &j in &neighbors {
                indices.push(j as i32);
                data.push(0.8);
            }
            indptr[i + 1] = indices.len() as i64;
        }

        (indptr, indices, data, n)
    }

    #[test]
    fn test_find_ab_params() {
        let (a, b) = find_ab_params(1.0, 0.1);
        assert!(a > 1.0 && a < 2.5, "a = {a}, expected in [1.0, 2.5]");
        assert!(b > 0.5 && b < 1.2, "b = {b}, expected in [0.5, 1.2]");
    }

    #[test]
    fn test_gpu_umap_basic() {
        let dev = require_gpu!();
        let (indptr, indices, data, n_obs) = test_graph();

        let result = gpu_umap_native(
            &dev, &indptr, &indices, &data, n_obs, 2,   // n_components
            50,  // n_epochs (low for test speed)
            0.1, // min_dist
            1.0, // spread
            5,   // negative_sample_rate
            1.0, // learning_rate
            42,  // seed
            None,
        )
        .unwrap();

        assert_eq!(result.n_obs, 50);
        assert_eq!(result.n_components, 2);
        assert_eq!(result.embedding.len(), 100);

        // All values should be finite
        for &v in &result.embedding {
            assert!(v.is_finite(), "embedding value should be finite, got {v}");
        }
    }

    #[test]
    fn test_gpu_umap_clusters_separated() {
        let dev = require_gpu!();
        let (indptr, indices, data, n_obs) = test_graph();

        let result = gpu_umap_native(
            &dev, &indptr, &indices, &data, n_obs, 2,
            200, // more epochs for better separation
            0.1, 1.0, 5, 1.0, 42, None,
        )
        .unwrap();

        // Compute centroids of two clusters
        let mut c1 = [0.0_f32; 2];
        let mut c2 = [0.0_f32; 2];
        for i in 0..25 {
            c1[0] += result.embedding[i * 2];
            c1[1] += result.embedding[i * 2 + 1];
        }
        for i in 25..50 {
            c2[0] += result.embedding[i * 2];
            c2[1] += result.embedding[i * 2 + 1];
        }
        c1[0] /= 25.0;
        c1[1] /= 25.0;
        c2[0] /= 25.0;
        c2[1] /= 25.0;

        // Clusters should be separated
        let inter_dist = ((c1[0] - c2[0]).powi(2) + (c1[1] - c2[1]).powi(2)).sqrt();

        // Average intra-cluster distance
        let mut intra = 0.0_f32;
        for i in 0..25 {
            let dx = result.embedding[i * 2] - c1[0];
            let dy = result.embedding[i * 2 + 1] - c1[1];
            intra += (dx * dx + dy * dy).sqrt();
        }
        intra /= 25.0;

        assert!(
            inter_dist > intra,
            "inter-cluster distance ({inter_dist:.4}) should exceed intra-cluster distance ({intra:.4})"
        );
    }

    #[test]
    fn test_gpu_umap_custom_init() {
        let dev = require_gpu!();
        let (indptr, indices, data, n_obs) = test_graph();
        let init: Vec<f32> = (0..n_obs * 2).map(|i| (i as f32) * 0.001).collect();

        let result = gpu_umap_native(
            &dev,
            &indptr,
            &indices,
            &data,
            n_obs,
            2,
            50,
            0.1,
            1.0,
            5,
            1.0,
            42,
            Some(&init),
        )
        .unwrap();

        assert_eq!(result.embedding.len(), 100);
        // Should differ from init (SGD moves points)
        let diff: f32 = result
            .embedding
            .iter()
            .zip(init.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>();
        assert!(diff > 0.0, "SGD should have moved points");
    }

    /// G10.3 parity: the graph-replay path produces an embedding
    /// consistent with the irreducible run-to-run jitter of the
    /// direct path — i.e., switching to graph replay does NOT add
    /// observable drift beyond what UMAP's atomicAdd races already
    /// produce.
    ///
    /// Strategy: measure two RMSEs:
    /// - `direct_vs_direct`: same code path twice → atomic-race floor.
    /// - `graph_vs_direct`: graph replay vs direct.
    ///
    /// Assert `graph_vs_direct` is within a small constant factor of
    /// the floor. A bug in the alpha-update mechanism (e.g., never
    /// applying the per-epoch update) would skip the SGD schedule
    /// entirely and produce gross divergence (RMSE jumps by an order
    /// of magnitude); the relative bound catches this without
    /// false-positive on legitimate jitter.
    ///
    /// We use [`set_cuda_graphs_enabled_override`] to flip the kill
    /// switch in-process so both paths run in the same test process
    /// (the `SCX_DISABLE_CUDA_GRAPHS=1` env-var path is cached via
    /// `OnceLock` after the first read and can't be toggled).
    #[test]
    fn test_gpu_umap_graph_vs_direct_parity_seed42() {
        let dev = require_gpu!();
        let (indptr, indices, data, n_obs) = test_graph();

        // Two direct runs → atomic-race floor.
        let prev = crate::gpu_graph::set_cuda_graphs_enabled_override(Some(false));
        let direct_a = gpu_umap_native(
            &dev, &indptr, &indices, &data, n_obs, 2, 50, 0.1, 1.0, 5, 1.0, 42, None,
        )
        .unwrap();
        let direct_b = gpu_umap_native(
            &dev, &indptr, &indices, &data, n_obs, 2, 50, 0.1, 1.0, 5, 1.0, 42, None,
        )
        .unwrap();

        // One graph-replay run.
        crate::gpu_graph::set_cuda_graphs_enabled_override(Some(true));
        let graph = gpu_umap_native(
            &dev, &indptr, &indices, &data, n_obs, 2, 50, 0.1, 1.0, 5, 1.0, 42, None,
        )
        .unwrap();

        crate::gpu_graph::set_cuda_graphs_enabled_override(prev);

        assert_eq!(direct_a.embedding.len(), graph.embedding.len());
        for &v in &graph.embedding {
            assert!(v.is_finite(), "graph embedding NaN/Inf: {v}");
        }

        let rmse = |xs: &[f32], ys: &[f32]| -> f32 {
            let s: f64 = xs
                .iter()
                .zip(ys.iter())
                .map(|(x, y)| (x - y) as f64)
                .map(|d| d * d)
                .sum();
            (s / xs.len() as f64).sqrt() as f32
        };

        let floor = rmse(&direct_a.embedding, &direct_b.embedding);
        let graph_vs_direct = rmse(&graph.embedding, &direct_a.embedding);

        eprintln!("umap parity: direct_vs_direct={floor:.4}, graph_vs_direct={graph_vs_direct:.4}");

        // Graph orchestration should NOT add observable drift beyond
        // the floor. Allow some headroom (2× floor + small additive
        // slack) because the graph path uses per_thread_stream, which
        // has a slightly different physical kernel-launch ordering
        // than the NULL stream and so produces marginally different
        // atomicAdd race outcomes.
        let allowed = (floor * 2.0).max(0.5);
        assert!(
            graph_vs_direct <= allowed,
            "graph_vs_direct RMSE {graph_vs_direct:.4} exceeds \
             2× direct-vs-direct floor {floor:.4} + 0.5 = {allowed:.4}. \
             This suggests the per-epoch alpha/epoch update via \
             cuGraphExecKernelNodeSetParams_v2 is not landing — \
             the graph is replaying with stale captured-time scalars."
        );
    }

    #[test]
    fn test_gpu_umap_error_zero_obs() {
        let dev = require_gpu!();
        let result = gpu_umap_native(
            &dev,
            &[0i64],
            &[],
            &[],
            0,
            2,
            50,
            0.1,
            1.0,
            5,
            1.0,
            42,
            None,
        );
        assert!(result.is_err());
    }
}
