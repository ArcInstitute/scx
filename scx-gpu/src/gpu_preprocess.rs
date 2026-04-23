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

use scx_format::{concatenate_csr, ShardSource};
use scx_sparse::ScxCsr;

use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::shard_pipeline::DoubleBufferedShardLoader;

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

/// Stream a [`ShardSource`] through [`gpu_apply_fused_ops`] and return a
/// single concatenated [`ScxCsr`] on the host.
///
/// This is the eager GPU path used by `pyscx.accel.normalize_total(device="gpu")`
/// and `log1p(device="gpu")`. It uses [`DoubleBufferedShardLoader`] to overlap
/// shard decode with GPU kernel launches. Per shard:
///
/// 1. Clone the shard's data buffer on-device (kernels mutate in-place; the
///    loader hands out shared references).
/// 2. Apply `gpu_apply_fused_ops` to the clone.
/// 3. D→H copy `(indptr, indices, data)` for that shard.
/// 4. Append to a host-side `Vec<ScxCsr>`.
///
/// After the stream completes, the per-shard CSRs are concatenated into a
/// single `ScxCsr` via [`scx_format::concatenate_csr`].
///
/// # Arguments
///
/// * `dev` — GPU device handle.
/// * `source` — shard-wise CSR source; must be `Sync` (required by the loader).
/// * `normalize` — `Some(target_sum)` to apply per-row normalization; `None` to
///   skip normalization.
/// * `log1p` — apply `log1p` elementwise after any normalization.
///
/// When both `normalize` and `log1p` are `None`/`false`, each shard is simply
/// downloaded and concatenated (no kernel launched). Callers can avoid that
/// overhead by not calling this function for the no-op case.
pub fn gpu_preprocess_to_csr(
    dev: &GpuDevice,
    source: &(dyn ShardSource + Sync),
    normalize: Option<f32>,
    log1p: bool,
) -> Result<ScxCsr, GpuError> {
    let n_vars = source.n_vars();
    let n_shards = source.n_shards();

    // Empty-source fast path: return a (0 × n_vars) CSR without spinning up
    // the streaming loader.
    if n_shards == 0 || source.n_obs() == 0 {
        return Ok(ScxCsr::new_unchecked((0, n_vars), vec![0], vec![], vec![]));
    }

    let loader = DoubleBufferedShardLoader::new(dev, source)?;
    let shard_csrs = std::sync::Mutex::new(Vec::<(usize, ScxCsr)>::with_capacity(n_shards));

    loader.for_each_shard(|shard_idx, gpu_csr| {
        let (n_rows, _) = gpu_csr.shape;

        // Apply fused ops in-place on a cloned data buffer. The loader hands
        // out a `&GpuCsr`; we must not mutate the underlying device memory
        // because subsequent passes (if any) would see the post-transform
        // values. Cloning is a small device-to-device copy and keeps the
        // API composable.
        let mut d_data = gpu_csr
            .data
            .try_clone()
            .map_err(|e| GpuError::CudaError(format!("clone shard data: {e}")))?;
        gpu_apply_fused_ops(dev, &gpu_csr.indptr, &mut d_data, n_rows, normalize, log1p)?;

        // Synchronize before D→H copies so the kernel output is visible.
        dev.synchronize()?;

        let indptr = dev.dtoh_copy(&gpu_csr.indptr)?;
        let indices = dev.dtoh_copy(&gpu_csr.indices)?;
        let data = dev.dtoh_copy(&d_data)?;
        let csr = ScxCsr::new_unchecked((n_rows, n_vars), indptr, indices, data);
        shard_csrs.lock().unwrap().push((shard_idx, csr));
        Ok(())
    })?;

    let mut shard_csrs = shard_csrs.into_inner().unwrap();
    shard_csrs.sort_by_key(|&(idx, _)| idx);
    let ordered: Vec<ScxCsr> = shard_csrs.into_iter().map(|(_, csr)| csr).collect();

    concatenate_csr(&ordered, n_vars)
        .map_err(|e| GpuError::InvalidShard(format!("concatenate_csr: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // -------------------------------------------------------------------
    // Task 5.1 — `gpu_preprocess_to_csr` streaming driver tests
    // -------------------------------------------------------------------

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use scx_sparse::ScxCsr;

    struct InMemorySource {
        shards: Vec<ScxCsr>,
        n_obs: usize,
        n_vars: usize,
    }

    impl ShardSource for InMemorySource {
        fn n_shards(&self) -> usize {
            self.shards.len()
        }
        fn n_obs(&self) -> usize {
            self.n_obs
        }
        fn n_vars(&self) -> usize {
            self.n_vars
        }
        fn read_shard(&self, shard_idx: usize) -> scx_format::Result<ScxCsr> {
            Ok(self.shards[shard_idx].clone())
        }
    }

    /// Build a random CSR with ~`density` fraction of nonzeros and strictly
    /// positive f32 values in `[0.5, 20.5)` (UMI-like).
    fn random_pos_csr(n_rows: usize, n_cols: usize, density: f32, seed: u64) -> ScxCsr {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indptr: Vec<i64> = Vec::with_capacity(n_rows + 1);
        let mut indices: Vec<i32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        indptr.push(0);
        for _ in 0..n_rows {
            for c in 0..n_cols {
                if rng.gen_bool(density as f64) {
                    indices.push(c as i32);
                    data.push(rng.gen_range(0.5..20.5));
                }
            }
            indptr.push(indices.len() as i64);
        }
        ScxCsr::new_unchecked((n_rows, n_cols), indptr, indices, data)
    }

    fn split_into_shards(csr: &ScxCsr, n_shards: usize) -> Vec<ScxCsr> {
        let (n_rows, n_cols) = (csr.n_rows(), csr.n_cols());
        let rows_per = n_rows.div_ceil(n_shards);
        let mut out = Vec::new();
        let mut row_start = 0;
        while row_start < n_rows {
            let row_end = (row_start + rows_per).min(n_rows);
            let p0 = csr.indptr[row_start] as usize;
            let p1 = csr.indptr[row_end] as usize;
            let shard_indptr: Vec<i64> = csr.indptr[row_start..=row_end]
                .iter()
                .map(|&p| p - csr.indptr[row_start])
                .collect();
            let shard_indices = csr.indices[p0..p1].to_vec();
            let shard_data = csr.data[p0..p1].to_vec();
            out.push(ScxCsr::new_unchecked(
                (row_end - row_start, n_cols),
                shard_indptr,
                shard_indices,
                shard_data,
            ));
            row_start = row_end;
        }
        out
    }

    #[test]
    fn test_gpu_preprocess_to_csr_normalize_matches_cpu() {
        let dev = require_gpu!();
        let n_rows = 500;
        let n_cols = 100;
        let target_sum = 1e4f32;
        let csr = random_pos_csr(n_rows, n_cols, 0.1, 11);
        let shards = split_into_shards(&csr, 4);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let result =
            gpu_preprocess_to_csr(&dev, &source, Some(target_sum), false).expect("gpu preprocess");
        assert_eq!(result.n_rows(), n_rows);
        assert_eq!(result.n_cols(), n_cols);
        assert_eq!(result.nnz(), csr.nnz());
        assert_eq!(result.indices, csr.indices);
        assert_eq!(result.indptr, csr.indptr);

        // CPU reference: apply normalize on the full concatenated CSR.
        let mut cpu_data = csr.data.clone();
        cpu_normalize(&csr.indptr, &mut cpu_data, target_sum as f64);

        // Elementwise rel-err < 1e-5.
        let mut max_rel_err = 0.0f64;
        for i in 0..cpu_data.len() {
            let diff = (result.data[i] as f64 - cpu_data[i] as f64).abs();
            let denom = (cpu_data[i] as f64).abs().max(1e-10);
            max_rel_err = max_rel_err.max(diff / denom);
        }
        assert!(max_rel_err < 1e-5, "normalize max rel err = {max_rel_err}");

        // Per-row sums equal target_sum (within 1e-3 rel).
        for r in 0..n_rows {
            let p0 = result.indptr[r] as usize;
            let p1 = result.indptr[r + 1] as usize;
            if p0 == p1 {
                continue;
            }
            let row_sum: f32 = result.data[p0..p1].iter().sum();
            let rel = (row_sum - target_sum).abs() / target_sum;
            assert!(rel < 1e-3, "row {r} sum {row_sum} != {target_sum}");
        }
    }

    #[test]
    fn test_gpu_preprocess_to_csr_log1p_matches_cpu() {
        let dev = require_gpu!();
        let n_rows = 500;
        let n_cols = 100;
        let csr = random_pos_csr(n_rows, n_cols, 0.1, 22);
        let shards = split_into_shards(&csr, 4);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let result = gpu_preprocess_to_csr(&dev, &source, None, true).expect("gpu preprocess");
        assert_eq!(result.indices, csr.indices);
        assert_eq!(result.indptr, csr.indptr);

        let mut cpu_data = csr.data.clone();
        cpu_log1p(&csr.indptr, &mut cpu_data);

        for i in 0..cpu_data.len() {
            let diff = (result.data[i] - cpu_data[i]).abs();
            assert!(diff < 1e-6, "log1p mismatch at [{i}]: {diff}");
        }
    }

    #[test]
    fn test_gpu_preprocess_to_csr_fused_matches_cpu() {
        let dev = require_gpu!();
        let n_rows = 500;
        let n_cols = 100;
        let target_sum = 1e4f32;
        let csr = random_pos_csr(n_rows, n_cols, 0.1, 33);
        let shards = split_into_shards(&csr, 4);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let result =
            gpu_preprocess_to_csr(&dev, &source, Some(target_sum), true).expect("gpu preprocess");
        assert_eq!(result.indices, csr.indices);
        assert_eq!(result.indptr, csr.indptr);

        let mut cpu_data = csr.data.clone();
        cpu_fused_normalize_log1p(&csr.indptr, &mut cpu_data, target_sum as f64);

        let mut max_rel_err = 0.0f64;
        for i in 0..cpu_data.len() {
            let diff = (result.data[i] as f64 - cpu_data[i] as f64).abs();
            let denom = (cpu_data[i] as f64).abs().max(1e-10);
            max_rel_err = max_rel_err.max(diff / denom);
        }
        // GPU f32 vs CPU f64 intermediates — same tolerance as the fused kernel test above.
        assert!(
            max_rel_err < 1e-4,
            "fused max rel err = {max_rel_err} (threshold 1e-4)"
        );
    }

    #[test]
    fn test_gpu_preprocess_to_csr_single_shard() {
        // Exercises the single-buffered fallback in DoubleBufferedShardLoader.
        let dev = require_gpu!();
        let n_rows = 100;
        let n_cols = 40;
        let target_sum = 1.0f32;
        let csr = random_pos_csr(n_rows, n_cols, 0.1, 44);
        let source = InMemorySource {
            shards: vec![csr.clone()],
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let result =
            gpu_preprocess_to_csr(&dev, &source, Some(target_sum), false).expect("gpu preprocess");
        assert_eq!(result.n_rows(), n_rows);
        assert_eq!(result.n_cols(), n_cols);
        // Per-row sums should be ~1.0.
        for r in 0..n_rows {
            let p0 = result.indptr[r] as usize;
            let p1 = result.indptr[r + 1] as usize;
            if p0 == p1 {
                continue;
            }
            let row_sum: f32 = result.data[p0..p1].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-4, "row {r} sum = {row_sum}");
        }
    }
}
