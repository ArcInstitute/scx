//! GPU fused preprocessing (normalize + log1p) on GPU-resident CSR.
//!
//! Provides GPU-accelerated versions of the CPU preprocessing operations
//! in `scx-engine/src/fused_ops.rs` and `scx-loader/src/normalize.rs`.
//!
//! The kernels operate in-place on [`GpuCsr`] data, modifying the `data`
//! array while leaving `indptr` and `indices` untouched. Each CUDA thread
//! handles one CSR row, computing the row sum, applying normalization, and
//! optionally computing log1p in a single pass over each row's nonzeros.
//!
//! ## Usage
//!
//! ```rust,ignore
//! // Decode a shard on GPU
//! let mut gpu_csr = decode_shard_gpu(&dev, shard_bytes)?;
//!
//! // Apply fused normalize+log1p in-place
//! gpu_normalize_log1p(&dev, &gpu_csr.indptr, &mut gpu_csr.data, gpu_csr.shape.0, 1e4)?;
//! ```
//!
//! ## Numerical differences
//!
//! The GPU path uses f32 arithmetic throughout (row sums, scaling factor,
//! log1p). The CPU path in `scx-engine` accumulates row sums in f64 and
//! computes the scaling factor in f64 before casting back to f32. This
//! produces small rounding differences (~1e-6 relative error). For single-cell
//! RNA-seq data these differences are negligible.

use cudarc::driver::safe::{CudaSlice, LaunchConfig};
use cudarc::driver::PushKernelArg;

use crate::device::GpuDevice;
use crate::error::GpuError;

/// PTX source for the normalize+log1p kernels, compiled at build time.
const NORMALIZE_LOG1P_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/normalize_log1p.ptx"));

/// Fused normalize_total + log1p on a GPU-resident CSR matrix (in-place).
///
/// For each row:
///   `data[i] = log1p(data[i] / row_sum * target_sum)`
///
/// Equivalent to `scx_engine::fused_ops::fused_normalize_log1p` applied to
/// all rows, but executed entirely on GPU.
///
/// # Arguments
///
/// * `dev` — GPU device handle
/// * `indptr` — CSR row pointers `[n_rows + 1]` (i64, on GPU, not modified)
/// * `data` — CSR non-zero values `[nnz]` (f32, on GPU, **modified in-place**)
/// * `n_rows` — number of rows in the CSR matrix
/// * `target_sum` — target sum for normalization (e.g., 1e4)
pub fn gpu_normalize_log1p(
    dev: &GpuDevice,
    indptr: &CudaSlice<i64>,
    data: &mut CudaSlice<f32>,
    n_rows: usize,
    target_sum: f32,
) -> Result<(), GpuError> {
    if n_rows == 0 {
        return Ok(());
    }

    let module = dev.load_module_cached(NORMALIZE_LOG1P_PTX)?;
    let func = module
        .load_function("normalize_log1p_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("normalize_log1p_kernel: {e}")))?;

    let n_rows_i32 = n_rows as i32;
    let threads: u32 = 256;
    let blocks = (n_rows as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(indptr)
            .arg(data)
            .arg(&n_rows_i32)
            .arg(&target_sum)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("normalize_log1p_kernel: {e}")))?;

    Ok(())
}

/// Normalize-only on a GPU-resident CSR matrix (in-place).
///
/// For each row:
///   `data[i] = data[i] / row_sum * target_sum`
///
/// No log1p is applied.
///
/// # Arguments
///
/// * `dev` — GPU device handle
/// * `indptr` — CSR row pointers `[n_rows + 1]` (i64, on GPU, not modified)
/// * `data` — CSR non-zero values `[nnz]` (f32, on GPU, **modified in-place**)
/// * `n_rows` — number of rows in the CSR matrix
/// * `target_sum` — target sum for normalization (e.g., 1e4)
pub fn gpu_normalize(
    dev: &GpuDevice,
    indptr: &CudaSlice<i64>,
    data: &mut CudaSlice<f32>,
    n_rows: usize,
    target_sum: f32,
) -> Result<(), GpuError> {
    if n_rows == 0 {
        return Ok(());
    }

    let module = dev.load_module_cached(NORMALIZE_LOG1P_PTX)?;
    let func = module
        .load_function("normalize_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("normalize_kernel: {e}")))?;

    let n_rows_i32 = n_rows as i32;
    let threads: u32 = 256;
    let blocks = (n_rows as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(indptr)
            .arg(data)
            .arg(&n_rows_i32)
            .arg(&target_sum)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("normalize_kernel: {e}")))?;

    Ok(())
}

/// Log1p-only on a GPU-resident CSR matrix (in-place).
///
/// For each row:
///   `data[i] = log1p(data[i])`
///
/// No normalization is applied.
///
/// # Arguments
///
/// * `dev` — GPU device handle
/// * `indptr` — CSR row pointers `[n_rows + 1]` (i64, on GPU, not modified)
/// * `data` — CSR non-zero values `[nnz]` (f32, on GPU, **modified in-place**)
/// * `n_rows` — number of rows in the CSR matrix
pub fn gpu_log1p(
    dev: &GpuDevice,
    indptr: &CudaSlice<i64>,
    data: &mut CudaSlice<f32>,
    n_rows: usize,
) -> Result<(), GpuError> {
    if n_rows == 0 {
        return Ok(());
    }

    let module = dev.load_module_cached(NORMALIZE_LOG1P_PTX)?;
    let func = module
        .load_function("log1p_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("log1p_kernel: {e}")))?;

    let n_rows_i32 = n_rows as i32;
    let threads: u32 = 256;
    let blocks = (n_rows as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        dev.stream()
            .launch_builder(&func)
            .arg(indptr)
            .arg(data)
            .arg(&n_rows_i32)
            .launch(cfg)
    }
    .map_err(|e| GpuError::KernelLaunchFailed(format!("log1p_kernel: {e}")))?;

    Ok(())
}

/// Dispatch fused operations on a GPU-resident CSR matrix (in-place).
///
/// Selects the optimal kernel based on which operations are requested:
/// - `(Some(target_sum), true)` → fused normalize+log1p
/// - `(Some(target_sum), false)` → normalize only
/// - `(None, true)` → log1p only
/// - `(None, false)` → no-op
///
/// This mirrors `scx_engine::fused_ops::apply_fused_ops` but on GPU.
pub fn gpu_apply_fused_ops(
    dev: &GpuDevice,
    indptr: &CudaSlice<i64>,
    data: &mut CudaSlice<f32>,
    n_rows: usize,
    normalize: Option<f32>,
    log1p: bool,
) -> Result<(), GpuError> {
    match (normalize, log1p) {
        (Some(target_sum), true) => gpu_normalize_log1p(dev, indptr, data, n_rows, target_sum),
        (Some(target_sum), false) => gpu_normalize(dev, indptr, data, n_rows, target_sum),
        (None, true) => gpu_log1p(dev, indptr, data, n_rows),
        (None, false) => Ok(()), // no-op
    }
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

    /// CPU reference: fused normalize+log1p for a CSR matrix.
    /// Matches scx-engine's implementation exactly (f64 intermediates, f32 output).
    fn cpu_fused_normalize_log1p(indptr: &[i64], data: &mut [f32], target_sum: f64) {
        let n_rows = indptr.len() - 1;
        for row in 0..n_rows {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            let row_sum: f64 = data[start..end].iter().map(|&v| v as f64).sum();
            if row_sum > 0.0 {
                let factor = target_sum / row_sum;
                for v in &mut data[start..end] {
                    *v = ((*v as f64 * factor) as f32).ln_1p();
                }
            }
        }
    }

    /// CPU reference: normalize-only.
    fn cpu_normalize(indptr: &[i64], data: &mut [f32], target_sum: f64) {
        let n_rows = indptr.len() - 1;
        for row in 0..n_rows {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            let row_sum: f64 = data[start..end].iter().map(|&v| v as f64).sum();
            if row_sum > 0.0 {
                let factor = target_sum / row_sum;
                for v in &mut data[start..end] {
                    *v = (*v as f64 * factor) as f32;
                }
            }
        }
    }

    /// CPU reference: log1p-only.
    fn cpu_log1p(indptr: &[i64], data: &mut [f32]) {
        let n_rows = indptr.len() - 1;
        for row in 0..n_rows {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            for v in &mut data[start..end] {
                *v = v.ln_1p();
            }
        }
    }

    #[test]
    fn test_gpu_normalize_log1p_matches_cpu() {
        let dev = require_gpu!();

        // 3-row CSR:
        // row 0: [5.0, 10.0] at indices [1, 3]   row_sum = 15
        // row 1: [1.0, 3.0, 7.0] at indices [0, 2, 4]   row_sum = 11
        // row 2: [2.0] at index [2]   row_sum = 2
        let indptr: Vec<i64> = vec![0, 2, 5, 6];
        let data: Vec<f32> = vec![5.0, 10.0, 1.0, 3.0, 7.0, 2.0];
        let target_sum = 10_000.0f32;
        let n_rows = 3;

        // GPU path
        let d_indptr = dev.htod_copy(&indptr).unwrap();
        let mut d_data = dev.htod_copy(&data).unwrap();
        gpu_normalize_log1p(&dev, &d_indptr, &mut d_data, n_rows, target_sum).unwrap();
        dev.synchronize().unwrap();
        let gpu_result = dev.dtoh_copy(&d_data).unwrap();

        // CPU reference
        let mut cpu_data = data.clone();
        cpu_fused_normalize_log1p(&indptr, &mut cpu_data, target_sum as f64);

        // Compare with relative tolerance (GPU f32 vs CPU f64 intermediates)
        assert_eq!(gpu_result.len(), cpu_data.len());
        for i in 0..gpu_result.len() {
            let diff = (gpu_result[i] - cpu_data[i]).abs();
            let denom = cpu_data[i].abs().max(1e-10);
            assert!(
                diff / denom < 1e-5,
                "normalize_log1p mismatch at [{}]: gpu={}, cpu={}, rel_err={}",
                i,
                gpu_result[i],
                cpu_data[i],
                diff / denom
            );
        }
    }

    #[test]
    fn test_gpu_normalize_only() {
        let dev = require_gpu!();

        let indptr: Vec<i64> = vec![0, 3, 5];
        let data: Vec<f32> = vec![1.0, 4.0, 5.0, 2.0, 8.0];
        let target_sum = 1.0f32;
        let n_rows = 2;

        // GPU path
        let d_indptr = dev.htod_copy(&indptr).unwrap();
        let mut d_data = dev.htod_copy(&data).unwrap();
        gpu_normalize(&dev, &d_indptr, &mut d_data, n_rows, target_sum).unwrap();
        dev.synchronize().unwrap();
        let gpu_result = dev.dtoh_copy(&d_data).unwrap();

        // CPU reference
        let mut cpu_data = data.clone();
        cpu_normalize(&indptr, &mut cpu_data, target_sum as f64);

        for i in 0..gpu_result.len() {
            assert!(
                (gpu_result[i] - cpu_data[i]).abs() < 1e-6,
                "normalize mismatch at [{}]: gpu={}, cpu={}",
                i,
                gpu_result[i],
                cpu_data[i]
            );
        }

        // Verify row sums
        let row0_sum: f32 = gpu_result[0..3].iter().sum();
        let row1_sum: f32 = gpu_result[3..5].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-5, "row 0 sum = {row0_sum}");
        assert!((row1_sum - 1.0).abs() < 1e-5, "row 1 sum = {row1_sum}");
    }

    #[test]
    fn test_gpu_log1p_only() {
        let dev = require_gpu!();

        let indptr: Vec<i64> = vec![0, 2, 4];
        let data: Vec<f32> = vec![0.0, 5.0, 10.0, 100.0];
        let n_rows = 2;

        // GPU path
        let d_indptr = dev.htod_copy(&indptr).unwrap();
        let mut d_data = dev.htod_copy(&data).unwrap();
        gpu_log1p(&dev, &d_indptr, &mut d_data, n_rows).unwrap();
        dev.synchronize().unwrap();
        let gpu_result = dev.dtoh_copy(&d_data).unwrap();

        // CPU reference
        let mut cpu_data = data.clone();
        cpu_log1p(&indptr, &mut cpu_data);

        for i in 0..gpu_result.len() {
            assert!(
                (gpu_result[i] - cpu_data[i]).abs() < 1e-6,
                "log1p mismatch at [{}]: gpu={}, cpu={}",
                i,
                gpu_result[i],
                cpu_data[i]
            );
        }
    }

    #[test]
    fn test_gpu_normalize_log1p_empty_row() {
        let dev = require_gpu!();

        // Row 0 is empty, row 1 has data
        let indptr: Vec<i64> = vec![0, 0, 3];
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let target_sum = 1e4f32;
        let n_rows = 2;

        let d_indptr = dev.htod_copy(&indptr).unwrap();
        let mut d_data = dev.htod_copy(&data).unwrap();
        gpu_normalize_log1p(&dev, &d_indptr, &mut d_data, n_rows, target_sum).unwrap();
        dev.synchronize().unwrap();
        let gpu_result = dev.dtoh_copy(&d_data).unwrap();

        // CPU reference
        let mut cpu_data = data.clone();
        cpu_fused_normalize_log1p(&indptr, &mut cpu_data, target_sum as f64);

        for i in 0..gpu_result.len() {
            let diff = (gpu_result[i] - cpu_data[i]).abs();
            assert!(
                diff < 1e-4,
                "empty_row mismatch at [{}]: gpu={}, cpu={}",
                i,
                gpu_result[i],
                cpu_data[i]
            );
        }
    }

    #[test]
    fn test_gpu_normalize_log1p_zero_rows() {
        let dev = require_gpu!();

        // Zero rows should be a no-op
        let indptr: Vec<i64> = vec![0];
        let data: Vec<f32> = vec![];

        let d_indptr = dev.htod_copy(&indptr).unwrap();
        let mut d_data = dev.htod_copy(&data).unwrap();
        gpu_normalize_log1p(&dev, &d_indptr, &mut d_data, 0, 1e4).unwrap();
        // Should not panic
    }

    #[test]
    fn test_gpu_apply_fused_ops_dispatch() {
        let dev = require_gpu!();

        let indptr: Vec<i64> = vec![0, 2, 4];
        let data: Vec<f32> = vec![5.0, 10.0, 3.0, 7.0];
        let n_rows = 2;

        // Test all four dispatch paths

        // 1. Both normalize + log1p
        let d_indptr = dev.htod_copy(&indptr).unwrap();
        let mut d_data = dev.htod_copy(&data).unwrap();
        gpu_apply_fused_ops(&dev, &d_indptr, &mut d_data, n_rows, Some(1e4), true).unwrap();
        dev.synchronize().unwrap();
        let result_both = dev.dtoh_copy(&d_data).unwrap();

        let mut cpu_both = data.clone();
        cpu_fused_normalize_log1p(&indptr, &mut cpu_both, 1e4);
        for i in 0..result_both.len() {
            assert!(
                (result_both[i] - cpu_both[i]).abs() < 1e-4,
                "both mismatch at {i}"
            );
        }

        // 2. Normalize only
        let mut d_data2 = dev.htod_copy(&data).unwrap();
        gpu_apply_fused_ops(&dev, &d_indptr, &mut d_data2, n_rows, Some(1.0), false).unwrap();
        dev.synchronize().unwrap();
        let result_norm = dev.dtoh_copy(&d_data2).unwrap();
        let row0_sum: f32 = result_norm[0..2].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-5);

        // 3. Log1p only
        let mut d_data3 = dev.htod_copy(&data).unwrap();
        gpu_apply_fused_ops(&dev, &d_indptr, &mut d_data3, n_rows, None, true).unwrap();
        dev.synchronize().unwrap();
        let result_log = dev.dtoh_copy(&d_data3).unwrap();
        assert!((result_log[0] - 5.0f32.ln_1p()).abs() < 1e-6);

        // 4. No-op
        let mut d_data4 = dev.htod_copy(&data).unwrap();
        gpu_apply_fused_ops(&dev, &d_indptr, &mut d_data4, n_rows, None, false).unwrap();
        dev.synchronize().unwrap();
        let result_noop = dev.dtoh_copy(&d_data4).unwrap();
        assert_eq!(result_noop, data);
    }

    #[test]
    fn test_gpu_normalize_log1p_large() {
        let dev = require_gpu!();

        // Generate a larger CSR matrix to stress-test the kernel
        let n_rows = 500;
        let nnz_per_row = 10;
        let mut indptr = vec![0i64];
        let mut data = Vec::new();
        let mut state: u64 = 0xCAFE_BABE;

        for _row in 0..n_rows {
            for _col in 0..nnz_per_row {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // UMI-like count values (1-20)
                data.push(((state % 20) + 1) as f32);
            }
            indptr.push(data.len() as i64);
        }

        let target_sum = 1e4f32;

        // GPU
        let d_indptr = dev.htod_copy(&indptr).unwrap();
        let mut d_data = dev.htod_copy(&data).unwrap();
        gpu_normalize_log1p(&dev, &d_indptr, &mut d_data, n_rows, target_sum).unwrap();
        dev.synchronize().unwrap();
        let gpu_result = dev.dtoh_copy(&d_data).unwrap();

        // CPU
        let mut cpu_data = data.clone();
        cpu_fused_normalize_log1p(&indptr, &mut cpu_data, target_sum as f64);

        // Compare
        let mut max_rel_err = 0.0f64;
        for i in 0..gpu_result.len() {
            let diff = (gpu_result[i] as f64 - cpu_data[i] as f64).abs();
            let denom = (cpu_data[i] as f64).abs().max(1e-10);
            let rel = diff / denom;
            if rel > max_rel_err {
                max_rel_err = rel;
            }
        }
        assert!(
            max_rel_err < 1e-5,
            "max relative error = {max_rel_err} (threshold: 1e-5)"
        );
    }
}
