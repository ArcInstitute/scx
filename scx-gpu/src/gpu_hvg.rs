//! Streaming GPU HVG (highly variable genes) primitives.
//!
//! Mirrors the CPU streaming functions in `scx_accel::hvg`:
//!
//! * [`gpu_streaming_mean_var`] — per-gene mean and variance (Bessel's correction).
//! * [`gpu_streaming_clip_square_sum`] — clipped sum / clipped-squared sum per gene
//!   used by the seurat_v3 HVG algorithm.
//!
//! Both iterate shards via [`DoubleBufferedShardLoader`] and accumulate per-gene
//! statistics directly on-device in f64 via `atomicAdd` (compute 6.x+ required,
//! which scx-gpu already targets via `compute_70` in `build.rs`).
//!
//! These GPU kernels are accessed via the `scx_accel::*_with_device` dispatch
//! wrappers (see `scx-accel/src/hvg.rs`) when `device = "gpu"`. The f64
//! accumulator path keeps numerical behaviour parity with the CPU fallback —
//! the two implementations agree to ~1e-5 relative error on typical scRNA-seq
//! densities.

use cudarc::driver::safe::{CudaSlice, LaunchConfig};
use cudarc::driver::PushKernelArg;

use scx_format::ShardSource;

use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::shard_pipeline::DoubleBufferedShardLoader;

/// PTX for the HVG atomicAdd kernels. Reuses the same `colmajor_ops` module
/// (shared with PCA helpers) to avoid a second PTX load.
const COLMAJOR_OPS_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/colmajor_ops.ptx"));

/// GPU equivalent of `scx_accel::streaming_mean_var`.
///
/// Streams shards through [`DoubleBufferedShardLoader`], atomically
/// accumulating per-column `Σ x` and `Σ x²` into f64 device buffers, then
/// computes `mean = Σ / n` and `var = (Σ² - n · mean²) / (n - 1)` on the host
/// (O(n_vars) work). Negative variances from numerical noise are clamped to 0.
///
/// Returns a pair `(means, variances)`, both f64 vectors of length `n_vars`.
pub fn gpu_streaming_mean_var(
    dev: &GpuDevice,
    source: &(dyn ShardSource + Sync),
) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
    let n_vars = source.n_vars();
    let n_obs = source.n_obs();

    if n_obs == 0 {
        return Ok((vec![0.0; n_vars], vec![0.0; n_vars]));
    }

    let mut d_col_sum = dev.alloc_zeros::<f64>(n_vars)?;
    let mut d_col_sum_sq = dev.alloc_zeros::<f64>(n_vars)?;

    let module = dev.load_module_cached(COLMAJOR_OPS_PTX)?;
    let func = module
        .load_function("col_sum_sq_nonzeros_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("col_sum_sq_nonzeros: {e}")))?;

    let loader = DoubleBufferedShardLoader::new(dev, source)?;
    loader.for_each_shard(|_idx, gpu_csr| {
        let nnz = gpu_csr.data.len() as i64;
        if nnz == 0 {
            return Ok(());
        }
        let threads: u32 = 256;
        let blocks = ((nnz as u64).div_ceil(threads as u64)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            dev.stream()
                .launch_builder(&func)
                .arg(&gpu_csr.indices)
                .arg(&gpu_csr.data)
                .arg(&nnz)
                .arg(&mut d_col_sum)
                .arg(&mut d_col_sum_sq)
                .launch(cfg)
        }
        .map_err(|e| GpuError::KernelLaunchFailed(format!("col_sum_sq_nonzeros: {e}")))?;
        Ok(())
    })?;

    dev.synchronize()?;
    let col_sum: Vec<f64> = dev.dtoh_copy(&d_col_sum)?;
    let col_sum_sq: Vec<f64> = dev.dtoh_copy(&d_col_sum_sq)?;

    let n = n_obs as f64;
    let denom = (n - 1.0).max(1.0);
    let mut means = vec![0.0f64; n_vars];
    let mut variances = vec![0.0f64; n_vars];
    for j in 0..n_vars {
        let mean = col_sum[j] / n;
        means[j] = mean;
        let var = (col_sum_sq[j] - n * mean * mean) / denom;
        variances[j] = if var < 0.0 { 0.0 } else { var };
    }

    Ok((means, variances))
}

/// GPU equivalent of `scx_accel::streaming_clip_square_sum`.
///
/// For each nonzero `x` at column `c`, accumulates `min(x, clip_val[c])` and
/// `min(x, clip_val[c])²` into per-column f64 buffers via atomicAdd.
///
/// Returns `(batch_count_sum, sq_batch_count_sum)` — both f64 vectors of
/// length `n_vars`.
pub fn gpu_streaming_clip_square_sum(
    dev: &GpuDevice,
    source: &(dyn ShardSource + Sync),
    clip_val: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
    let n_vars = source.n_vars();
    if clip_val.len() != n_vars {
        return Err(GpuError::ShapeMismatch {
            expected: format!("clip_val.len() == n_vars = {n_vars}"),
            got: format!("clip_val.len() = {}", clip_val.len()),
        });
    }

    let mut d_batch_sum = dev.alloc_zeros::<f64>(n_vars)?;
    let mut d_sq_batch_sum = dev.alloc_zeros::<f64>(n_vars)?;
    let d_clip: CudaSlice<f64> = dev.htod_copy(clip_val)?;

    let module = dev.load_module_cached(COLMAJOR_OPS_PTX)?;
    let func = module
        .load_function("col_clip_sq_nonzeros_kernel")
        .map_err(|e| GpuError::KernelLaunchFailed(format!("col_clip_sq_nonzeros: {e}")))?;

    let loader = DoubleBufferedShardLoader::new(dev, source)?;
    loader.for_each_shard(|_idx, gpu_csr| {
        let nnz = gpu_csr.data.len() as i64;
        if nnz == 0 {
            return Ok(());
        }
        let threads: u32 = 256;
        let blocks = ((nnz as u64).div_ceil(threads as u64)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            dev.stream()
                .launch_builder(&func)
                .arg(&gpu_csr.indices)
                .arg(&gpu_csr.data)
                .arg(&nnz)
                .arg(&d_clip)
                .arg(&mut d_batch_sum)
                .arg(&mut d_sq_batch_sum)
                .launch(cfg)
        }
        .map_err(|e| GpuError::KernelLaunchFailed(format!("col_clip_sq_nonzeros: {e}")))?;
        Ok(())
    })?;

    dev.synchronize()?;
    let batch_sum: Vec<f64> = dev.dtoh_copy(&d_batch_sum)?;
    let sq_batch_sum: Vec<f64> = dev.dtoh_copy(&d_sq_batch_sum)?;
    Ok((batch_sum, sq_batch_sum))
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// CPU reference: streaming mean / variance with Bessel's correction.
    fn cpu_mean_var(csr: &ScxCsr) -> (Vec<f64>, Vec<f64>) {
        let (n_obs, n_vars) = (csr.n_rows(), csr.n_cols());
        let mut col_sum = vec![0.0f64; n_vars];
        let mut col_sum_sq = vec![0.0f64; n_vars];
        for (&c, &v) in csr.indices.iter().zip(csr.data.iter()) {
            let c = c as usize;
            let v = v as f64;
            col_sum[c] += v;
            col_sum_sq[c] += v * v;
        }
        let n = n_obs as f64;
        let denom = (n - 1.0).max(1.0);
        let mut means = vec![0.0f64; n_vars];
        let mut variances = vec![0.0f64; n_vars];
        for j in 0..n_vars {
            let m = col_sum[j] / n;
            means[j] = m;
            variances[j] = ((col_sum_sq[j] - n * m * m) / denom).max(0.0);
        }
        (means, variances)
    }

    fn cpu_clip_square_sum(csr: &ScxCsr, clip_val: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n_vars = csr.n_cols();
        let mut bcs = vec![0.0f64; n_vars];
        let mut sbcs = vec![0.0f64; n_vars];
        for (&c, &v) in csr.indices.iter().zip(csr.data.iter()) {
            let c = c as usize;
            let v = (v as f64).min(clip_val[c]);
            bcs[c] += v;
            sbcs[c] += v * v;
        }
        (bcs, sbcs)
    }

    #[test]
    fn test_gpu_streaming_mean_var_matches_cpu() {
        let dev = require_gpu!();
        let n_rows = 500;
        let n_cols = 100;
        let csr = random_pos_csr(n_rows, n_cols, 0.1, 55);
        let shards = split_into_shards(&csr, 4);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };

        let (gpu_means, gpu_vars) =
            gpu_streaming_mean_var(&dev, &source).expect("gpu streaming mean/var");
        let (cpu_means, cpu_vars) = cpu_mean_var(&csr);

        for j in 0..n_cols {
            let dm = (gpu_means[j] - cpu_means[j]).abs();
            let dv = (gpu_vars[j] - cpu_vars[j]).abs();
            let denom_m = cpu_means[j].abs().max(1e-10);
            let denom_v = cpu_vars[j].abs().max(1e-10);
            assert!(
                dm / denom_m < 1e-5,
                "mean mismatch col {j}: gpu={} cpu={} rel={}",
                gpu_means[j],
                cpu_means[j],
                dm / denom_m
            );
            assert!(
                dv / denom_v < 1e-5,
                "var mismatch col {j}: gpu={} cpu={} rel={}",
                gpu_vars[j],
                cpu_vars[j],
                dv / denom_v
            );
        }
    }

    #[test]
    fn test_gpu_streaming_clip_square_sum_matches_cpu() {
        let dev = require_gpu!();
        let n_rows = 500;
        let n_cols = 100;
        let csr = random_pos_csr(n_rows, n_cols, 0.1, 66);
        let shards = split_into_shards(&csr, 4);
        let source = InMemorySource {
            shards,
            n_obs: n_rows,
            n_vars: n_cols,
        };
        // Deterministic clip threshold per column.
        let clip_val: Vec<f64> = (0..n_cols).map(|j| 1.0 + (j as f64) * 0.05).collect();

        let (gpu_bcs, gpu_sbcs) = gpu_streaming_clip_square_sum(&dev, &source, &clip_val)
            .expect("gpu streaming clip sum");
        let (cpu_bcs, cpu_sbcs) = cpu_clip_square_sum(&csr, &clip_val);

        for j in 0..n_cols {
            let d1 = (gpu_bcs[j] - cpu_bcs[j]).abs();
            let d2 = (gpu_sbcs[j] - cpu_sbcs[j]).abs();
            let den1 = cpu_bcs[j].abs().max(1e-10);
            let den2 = cpu_sbcs[j].abs().max(1e-10);
            assert!(
                d1 / den1 < 1e-5,
                "bcs col {j}: gpu={} cpu={}",
                gpu_bcs[j],
                cpu_bcs[j]
            );
            assert!(
                d2 / den2 < 1e-5,
                "sbcs col {j}: gpu={} cpu={}",
                gpu_sbcs[j],
                cpu_sbcs[j]
            );
        }
    }

    #[test]
    fn test_gpu_streaming_mean_var_empty() {
        let dev = require_gpu!();
        let source = InMemorySource {
            shards: vec![ScxCsr::new_unchecked((0, 5), vec![0], vec![], vec![])],
            n_obs: 0,
            n_vars: 5,
        };
        let (means, vars) = gpu_streaming_mean_var(&dev, &source).unwrap();
        assert_eq!(means, vec![0.0; 5]);
        assert_eq!(vars, vec![0.0; 5]);
    }
}
