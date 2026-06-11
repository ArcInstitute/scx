//! CSC streaming mean/var and clipped-square-sum kernels for HVG.
//!
//! Column-major equivalents of [`crate::hvg::streaming_mean_var`] and
//! [`crate::hvg::streaming_clip_square_sum`]. Each kernel iterates CSC
//! shards once, accumulating per-column statistics directly into the
//! output slot — no `O(n_vars)` row-wise scratch.
//!
//! Multi-batch streaming (per-batch mean/var) intentionally stays on
//! CSR: batched lookups index by row, so row-major decode keeps cache
//! locality. See `streaming_mean_var_batched`.

use crate::error::Result;
use crate::hvg::HvgStats;
use scx_format_io::ColumnShardSource;

/// Single-pass streaming mean and variance per column on a CSC source.
///
/// Numerically equivalent to [`crate::streaming_mean_var`] within f64
/// epsilon. Both kernels accumulate `sum` and `sum_sq` in f64 and apply
/// Bessel's correction (ddof=1) at the end.
pub fn streaming_mean_var_csc<S: ColumnShardSource + ?Sized>(source: &S) -> Result<HvgStats> {
    let n_vars = source.n_vars();
    let n_obs = source.n_obs();

    if n_obs == 0 {
        return Ok(HvgStats {
            means: vec![0.0; n_vars],
            variances: vec![0.0; n_vars],
        });
    }

    let mut col_sum = vec![0.0f64; n_vars];
    let mut col_sum_sq = vec![0.0f64; n_vars];

    let n_shards = source.n_csc_shards();
    for shard_idx in 0..n_shards {
        let csc = source.read_csc_shard(shard_idx)?;
        let shard_n_cols = csc.n_cols();
        // Map shard-local column index → global column index using the
        // shard's reported col_range. Shards may not start at 0 (e.g.
        // multi-shard build-csc with a non-zero first shard).
        let (global_col_start, _) = source
            .csc_shard_col_range(shard_idx)
            .unwrap_or((0, shard_n_cols as u32));
        let global_col_start = global_col_start as usize;

        for local_col in 0..shard_n_cols {
            let s = csc.indptr[local_col] as usize;
            let e = csc.indptr[local_col + 1] as usize;
            let global_col = global_col_start + local_col;
            // Bounds check: a corrupted catalog shouldn't OOB-write.
            if global_col >= n_vars {
                continue;
            }
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;
            for &v in &csc.data[s..e] {
                let v = v as f64;
                sum += v;
                sum_sq += v * v;
            }
            col_sum[global_col] += sum;
            col_sum_sq[global_col] += sum_sq;
        }
    }

    let n = n_obs as f64;
    let denom = (n - 1.0).max(1.0);
    let mut means = vec![0.0f64; n_vars];
    let mut variances = vec![0.0f64; n_vars];
    for j in 0..n_vars {
        let mean = col_sum[j] / n;
        means[j] = mean;
        variances[j] = (col_sum_sq[j] - n * mean * mean) / denom;
        if variances[j] < 0.0 {
            variances[j] = 0.0;
        }
    }

    Ok(HvgStats { means, variances })
}

/// Single-pass streaming clipped accumulation per column on a CSC source.
///
/// Equivalent to [`crate::streaming_clip_square_sum`]; `clip_val` length
/// must equal `source.n_vars()`.
pub fn streaming_clip_square_sum_csc<S: ColumnShardSource + ?Sized>(
    source: &S,
    clip_val: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n_vars = source.n_vars();
    debug_assert_eq!(clip_val.len(), n_vars);

    let mut clipped_sum = vec![0.0f64; n_vars];
    let mut clipped_sum_sq = vec![0.0f64; n_vars];

    let n_shards = source.n_csc_shards();
    for shard_idx in 0..n_shards {
        let csc = source.read_csc_shard(shard_idx)?;
        let shard_n_cols = csc.n_cols();
        let (global_col_start, _) = source
            .csc_shard_col_range(shard_idx)
            .unwrap_or((0, shard_n_cols as u32));
        let global_col_start = global_col_start as usize;

        for local_col in 0..shard_n_cols {
            let s = csc.indptr[local_col] as usize;
            let e = csc.indptr[local_col + 1] as usize;
            let global_col = global_col_start + local_col;
            if global_col >= n_vars {
                continue;
            }
            let clip = clip_val[global_col];
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;
            for &v in &csc.data[s..e] {
                let v_clipped = (v as f64).min(clip);
                sum += v_clipped;
                sum_sq += v_clipped * v_clipped;
            }
            clipped_sum[global_col] += sum;
            clipped_sum_sq[global_col] += sum_sq;
        }
    }

    Ok((clipped_sum, clipped_sum_sq))
}

// ---------------------------------------------------------------------------
// Device-dispatched CSC wrappers (CPU default, GPU behind feature = "gpu").
// Mirror the CSR `streaming_*_with_device` wrappers in `crate::hvg`. The GPU
// path runs the one-block-per-column CSC reduce kernel (no `atomicAdd`); it
// requires `S: Sync` because the device source pipelines decode on a scoped
// worker thread. Falls back to the CPU CSC kernel on GPU-init failure.
// ---------------------------------------------------------------------------

/// Device-dispatched [`streaming_mean_var_csc`].
#[cfg(feature = "gpu")]
pub fn streaming_mean_var_csc_with_device<S: ColumnShardSource + Sync>(
    source: &S,
    device: &str,
    device_id: usize,
) -> Result<HvgStats> {
    if device != "gpu" {
        return streaming_mean_var_csc(source);
    }
    let dev = match scx_gpu::GpuDevice::new(device_id) {
        Ok(d) => d,
        Err(_) => return streaming_mean_var_csc(source),
    };
    let (means, variances) = scx_gpu::gpu_streaming_mean_var_csc(&dev, source).map_err(|e| {
        crate::error::AccelError::LinAlg(format!("gpu_streaming_mean_var_csc failed: {e}"))
    })?;
    Ok(HvgStats { means, variances })
}

/// Device-dispatched [`streaming_clip_square_sum_csc`].
#[cfg(feature = "gpu")]
pub fn streaming_clip_square_sum_csc_with_device<S: ColumnShardSource + Sync>(
    source: &S,
    clip_val: &[f64],
    device: &str,
    device_id: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if device != "gpu" {
        return streaming_clip_square_sum_csc(source, clip_val);
    }
    let dev = match scx_gpu::GpuDevice::new(device_id) {
        Ok(d) => d,
        Err(_) => return streaming_clip_square_sum_csc(source, clip_val),
    };
    scx_gpu::gpu_streaming_clip_square_sum_csc(&dev, source, clip_val).map_err(|e| {
        crate::error::AccelError::LinAlg(format!("gpu_streaming_clip_square_sum_csc failed: {e}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csc::test_helpers::{deterministic_dense, write_csr_csc_test_file};
    use crate::hvg::{streaming_clip_square_sum, streaming_mean_var};
    use scx_format_io::{BackedCscReader, BackedCsrReader, ScxReader};
    use tempfile::tempdir;

    /// The `device="cpu"` arm of the GPU dispatch wrappers must be identical
    /// to the plain CPU CSC kernels (no GPU touched). Runs on any host built
    /// with `--features gpu`; the GPU arm is covered by the scx-gpu harness.
    #[cfg(feature = "gpu")]
    #[test]
    fn streaming_csc_with_device_cpu_matches_cpu() {
        let dir = tempdir().unwrap();
        let (n_obs, n_vars) = (10usize, 8usize);
        let dense = deterministic_dense(n_obs, n_vars);
        let path = write_csr_csc_test_file(dir.path(), "dev", n_obs, n_vars, &dense, 3);
        let reader = BackedCscReader::new(ScxReader::open(&path).unwrap(), 0).unwrap();

        let base = streaming_mean_var_csc(&reader).unwrap();
        let dev = streaming_mean_var_csc_with_device(&reader, "cpu", 0).unwrap();
        assert_eq!(base.means, dev.means);
        assert_eq!(base.variances, dev.variances);

        let clip = vec![5.0_f64; n_vars];
        let (bs, bsq) = streaming_clip_square_sum_csc(&reader, &clip).unwrap();
        let (ds, dsq) =
            streaming_clip_square_sum_csc_with_device(&reader, &clip, "cpu", 0).unwrap();
        assert_eq!(bs, ds);
        assert_eq!(bsq, dsq);
    }

    #[test]
    fn streaming_mean_var_csc_matches_csr() {
        let dir = tempdir().unwrap();
        let n_obs = 12usize;
        let n_vars = 10usize;
        let dense = deterministic_dense(n_obs, n_vars);
        let path = write_csr_csc_test_file(dir.path(), "mv", n_obs, n_vars, &dense, 4);

        let csr_reader = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);
        let csr_stats = streaming_mean_var(&csr_reader).unwrap();

        let csc_reader = BackedCscReader::new(ScxReader::open(&path).unwrap(), 0).unwrap();
        let csc_stats = streaming_mean_var_csc(&csc_reader).unwrap();

        for j in 0..n_vars {
            assert!(
                (csr_stats.means[j] - csc_stats.means[j]).abs() < 1e-9,
                "mean[{j}] mismatch: csr={} csc={}",
                csr_stats.means[j],
                csc_stats.means[j]
            );
            assert!(
                (csr_stats.variances[j] - csc_stats.variances[j]).abs() < 1e-7,
                "var[{j}] mismatch: csr={} csc={}",
                csr_stats.variances[j],
                csc_stats.variances[j]
            );
        }
    }

    #[test]
    fn streaming_clip_square_sum_csc_matches_csr() {
        let dir = tempdir().unwrap();
        let n_obs = 8usize;
        let n_vars = 6usize;
        let dense = deterministic_dense(n_obs, n_vars);
        let path = write_csr_csc_test_file(dir.path(), "clip", n_obs, n_vars, &dense, 3);

        let clip = vec![3.0_f64; n_vars];
        let csr_reader = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);
        let (csr_sum, csr_sum_sq) = streaming_clip_square_sum(&csr_reader, &clip).unwrap();

        let csc_reader = BackedCscReader::new(ScxReader::open(&path).unwrap(), 0).unwrap();
        let (csc_sum, csc_sum_sq) = streaming_clip_square_sum_csc(&csc_reader, &clip).unwrap();

        for j in 0..n_vars {
            assert!(
                (csr_sum[j] - csc_sum[j]).abs() < 1e-9,
                "clipped sum[{j}] mismatch"
            );
            assert!(
                (csr_sum_sq[j] - csc_sum_sq[j]).abs() < 1e-7,
                "clipped sum_sq[{j}] mismatch"
            );
        }
    }
}
