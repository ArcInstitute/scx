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

use cudarc::driver::safe::LaunchConfig;
use cudarc::driver::PushKernelArg;

use crate::device::GpuDevice;
use crate::error::GpuError;

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

    for epoch in 0..n_epochs {
        let alpha = learning_rate * (1.0 - epoch as f32 / n_epochs as f32);
        let epoch_i32 = epoch as i32;

        unsafe {
            dev.stream()
                .launch_builder(&func)
                .arg(&mut d_embedding)
                .arg(&d_head)
                .arg(&d_tail)
                .arg(&mut d_epoch_of_next_sample)
                .arg(&d_epochs_per_sample)
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
