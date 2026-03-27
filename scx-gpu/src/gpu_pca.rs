//! GPU PCA helper operations.
//!
//! Provides the mean-correction kernel used in GPU randomized PCA:
//! `Y = X @ Ω - 1_n × (μ^T @ Ω)`. Rather than materializing the centered
//! matrix `(X - μ)`, the SpMM result Y is corrected in-place by subtracting
//! the precomputed correction vector `mc = μ^T @ Ω` from each row.
//!
//! The full GPU PCA pipeline will be built in Phase 4c Step 2; this module
//! currently provides just the mean-correction building block.

use cudarc::driver::safe::{CudaSlice, LaunchConfig};
use cudarc::driver::PushKernelArg;

use crate::device::GpuDevice;
use crate::error::GpuError;

/// PTX source for the mean-correction kernel, compiled at build time.
const MEAN_CORRECT_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/spmm_mean_correct.ptx"));

/// Subtract a per-column correction vector from each row of a matrix.
///
/// Computes `Y[row, col] -= mc[col]` for all `(row, col)`.
///
/// - `y`: row-major `[n_obs × k]` on GPU, modified in-place
/// - `mc`: correction vector `[k]` on GPU
/// - `n_obs`: number of rows
/// - `k`: number of columns
///
/// The kernel uses 256 threads per block with one thread per element.
/// Total threads = `n_obs × k`.
pub fn mean_correct_gpu(
    dev: &GpuDevice,
    y: &mut CudaSlice<f32>,
    mc: &CudaSlice<f32>,
    n_obs: usize,
    k: usize,
) -> Result<(), GpuError> {
    let total = n_obs * k;
    if total == 0 {
        return Ok(());
    }

    let module = dev.load_module_cached(MEAN_CORRECT_PTX)?;
    let func = module
        .load_function("mean_correct_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("mean_correct_kernel: {e}")))?;

    let n_obs_i32 = n_obs as i32;
    let k_i32 = k as i32;

    let threads: u32 = 256;
    let blocks = (total as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(y)
            .arg(mc)
            .arg(&n_obs_i32)
            .arg(&k_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("mean_correct_kernel: {e}")))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::GpuDevice;

    macro_rules! require_gpu {
        () => {
            match GpuDevice::new(0) {
                Ok(dev) => dev,
                Err(_) => {
                    eprintln!("CUDA not available — skipping GPU test");
                    return;
                }
            }
        };
    }

    #[test]
    fn test_mean_correct() {
        let dev = require_gpu!();

        let n_obs = 4;
        let k = 3;

        // Y (row-major 4×3): each row = [10, 20, 30]
        let y_host: Vec<f32> = vec![
            10.0, 20.0, 30.0, // row 0
            10.0, 20.0, 30.0, // row 1
            10.0, 20.0, 30.0, // row 2
            10.0, 20.0, 30.0, // row 3
        ];
        // mc = [1, 2, 3]
        let mc_host: Vec<f32> = vec![1.0, 2.0, 3.0];

        let mut d_y = dev.htod_copy(&y_host).unwrap();
        let d_mc = dev.htod_copy(&mc_host).unwrap();

        mean_correct_gpu(&dev, &mut d_y, &d_mc, n_obs, k).unwrap();
        dev.synchronize().unwrap();

        let result = dev.dtoh_copy(&d_y).unwrap();

        // Expected: each row = [10-1, 20-2, 30-3] = [9, 18, 27]
        let expected: Vec<f32> = vec![
            9.0, 18.0, 27.0, 9.0, 18.0, 27.0, 9.0, 18.0, 27.0, 9.0, 18.0, 27.0,
        ];

        assert_eq!(result.len(), expected.len());
        for i in 0..result.len() {
            assert!(
                (result[i] - expected[i]).abs() < 1e-5,
                "mean_correct mismatch at {i}: got {}, expected {}",
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_mean_correct_empty() {
        let dev = require_gpu!();

        // Zero-size should not panic
        let mut d_y = dev.alloc_zeros::<f32>(0).unwrap();
        let d_mc = dev.alloc_zeros::<f32>(0).unwrap();
        mean_correct_gpu(&dev, &mut d_y, &d_mc, 0, 0).unwrap();
    }
}
