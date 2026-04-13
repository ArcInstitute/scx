//! Streaming Highly Variable Genes (HVG) kernels.
//!
//! Provides shard-by-shard streaming computation of per-gene mean/variance
//! and clipped-square-sum statistics for the seurat_v3 HVG algorithm.
//! These operate through the [`ShardSource`] trait, so they work on both
//! raw backed data and lazy-transformed data without materialization.

use crate::error::Result;
use scx_format::ShardSource;

/// Per-gene mean and variance statistics.
#[derive(Debug, Clone)]
pub struct HvgStats {
    /// Per-gene mean expression (length = n_vars).
    pub means: Vec<f64>,
    /// Per-gene variance with Bessel's correction (length = n_vars).
    pub variances: Vec<f64>,
}

/// Single-pass streaming mean and variance per column.
///
/// Accumulates per-column sum and sum-of-squares in f64, then computes:
///   mean = sum / n
///   var  = (sum_sq - n * mean²) / (n - 1)   (Bessel's correction, ddof=1)
///
/// This matches scanpy's `correction=1` parameter in `mean_var()`.
/// Memory: O(n_vars) for two accumulator vectors.
pub fn streaming_mean_var<S: ShardSource>(source: &S) -> Result<HvgStats> {
    let n_vars = source.n_vars();
    let n_obs = source.n_obs();
    let mut col_sum = vec![0.0f64; n_vars];
    let mut col_sum_sq = vec![0.0f64; n_vars];

    for shard_idx in 0..source.n_shards() {
        let csr = source.read_shard(shard_idx)?;
        for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
            let c = col as usize;
            let v = val as f64;
            col_sum[c] += v;
            col_sum_sq[c] += v * v;
        }
    }

    let n = n_obs as f64;
    let denom = (n - 1.0).max(1.0); // avoid division by zero for n <= 1
    let mut means = vec![0.0f64; n_vars];
    let mut variances = vec![0.0f64; n_vars];

    for j in 0..n_vars {
        let mean = col_sum[j] / n;
        means[j] = mean;
        // Var = (sum_sq - n * mean^2) / (n - 1)
        variances[j] = (col_sum_sq[j] - n * mean * mean) / denom;
        // Clamp to zero (numerical noise can produce tiny negatives)
        if variances[j] < 0.0 {
            variances[j] = 0.0;
        }
    }

    Ok(HvgStats { means, variances })
}

/// Single-pass streaming clipped accumulation for seurat_v3 normalized variance.
///
/// For each nonzero value in the matrix, clips it to `min(val, clip_val[col])`,
/// then accumulates both the clipped sum and the clipped-squared sum per column.
///
/// Returns `(batch_counts_sum, squared_batch_counts_sum)` — both `Vec<f64>` of
/// length `n_vars`.
///
/// Memory: O(n_vars).
pub fn streaming_clip_square_sum<S: ShardSource>(
    source: &S,
    clip_val: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n_vars = source.n_vars();
    debug_assert_eq!(clip_val.len(), n_vars);

    let mut batch_counts_sum = vec![0.0f64; n_vars];
    let mut sq_batch_counts_sum = vec![0.0f64; n_vars];

    for shard_idx in 0..source.n_shards() {
        let csr = source.read_shard(shard_idx)?;
        for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
            let c = col as usize;
            let v = (val as f64).min(clip_val[c]);
            batch_counts_sum[c] += v;
            sq_batch_counts_sum[c] += v * v;
        }
    }

    Ok((batch_counts_sum, sq_batch_counts_sum))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_sparse::ScxCsr;

    /// A simple in-memory ShardSource for testing.
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

    fn make_test_source() -> InMemorySource {
        // 4 rows x 3 cols, split into 2 shards of 2 rows each:
        //   [[1, 0, 3],
        //    [0, 2, 0],
        //    [4, 0, 0],
        //    [0, 5, 6]]
        let shard0 = ScxCsr::new_unchecked(
            (2, 3),
            vec![0, 2, 3],       // indptr
            vec![0, 2, 1],       // indices
            vec![1.0, 3.0, 2.0], // data
        );
        let shard1 = ScxCsr::new_unchecked(
            (2, 3),
            vec![0, 1, 3],       // indptr
            vec![0, 1, 2],       // indices
            vec![4.0, 5.0, 6.0], // data
        );
        InMemorySource {
            shards: vec![shard0, shard1],
            n_obs: 4,
            n_vars: 3,
        }
    }

    #[test]
    fn test_streaming_mean_var() {
        let source = make_test_source();
        let stats = streaming_mean_var(&source).unwrap();

        // Column values: col0=[1,0,4,0], col1=[0,2,0,5], col2=[3,0,0,6]
        // Means:  col0=5/4=1.25, col1=7/4=1.75, col2=9/4=2.25
        let expected_means = [1.25, 1.75, 2.25];
        for (got, exp) in stats.means.iter().zip(expected_means.iter()) {
            assert!((got - exp).abs() < 1e-10, "mean: got {got}, expected {exp}");
        }

        // Variances (Bessel, ddof=1):
        //   col0: ((1-1.25)^2 + (0-1.25)^2 + (4-1.25)^2 + (0-1.25)^2) / 3
        //       = (0.0625 + 1.5625 + 7.5625 + 1.5625) / 3 = 10.75 / 3 ≈ 3.5833
        //   col1: ((0-1.75)^2 + (2-1.75)^2 + (0-1.75)^2 + (5-1.75)^2) / 3
        //       = (3.0625 + 0.0625 + 3.0625 + 10.5625) / 3 = 16.75 / 3 ≈ 5.5833
        //   col2: ((3-2.25)^2 + (0-2.25)^2 + (0-2.25)^2 + (6-2.25)^2) / 3
        //       = (0.5625 + 5.0625 + 5.0625 + 14.0625) / 3 = 24.75 / 3 = 8.25
        let expected_vars = [10.75 / 3.0, 16.75 / 3.0, 24.75 / 3.0];
        for (got, exp) in stats.variances.iter().zip(expected_vars.iter()) {
            assert!((got - exp).abs() < 1e-10, "var: got {got}, expected {exp}");
        }
    }

    #[test]
    fn test_streaming_clip_square_sum() {
        let source = make_test_source();
        // clip_val = [2.0, 3.0, 4.0] per column
        let clip_val = [2.0, 3.0, 4.0];

        let (bcs, sbcs) = streaming_clip_square_sum(&source, &clip_val).unwrap();

        // Nonzeros: (col0,1.0), (col2,3.0), (col1,2.0), (col0,4.0), (col1,5.0), (col2,6.0)
        // After clipping:
        //   col0: min(1,2)=1, min(4,2)=2  → sum=3, sq_sum=1+4=5
        //   col1: min(2,3)=2, min(5,3)=3  → sum=5, sq_sum=4+9=13
        //   col2: min(3,4)=3, min(6,4)=4  → sum=7, sq_sum=9+16=25
        assert!((bcs[0] - 3.0).abs() < 1e-10);
        assert!((bcs[1] - 5.0).abs() < 1e-10);
        assert!((bcs[2] - 7.0).abs() < 1e-10);
        assert!((sbcs[0] - 5.0).abs() < 1e-10);
        assert!((sbcs[1] - 13.0).abs() < 1e-10);
        assert!((sbcs[2] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_mean_var_single_obs() {
        // Edge case: 1 row → denom = max(0, 1) = 1
        let shard = ScxCsr::new_unchecked((1, 2), vec![0, 2], vec![0, 1], vec![3.0, 7.0]);
        let source = InMemorySource {
            shards: vec![shard],
            n_obs: 1,
            n_vars: 2,
        };
        let stats = streaming_mean_var(&source).unwrap();
        assert!((stats.means[0] - 3.0).abs() < 1e-10);
        assert!((stats.means[1] - 7.0).abs() < 1e-10);
        // With only 1 obs, var = 0 (numerically: sum_sq - n*mean^2 = 9 - 9 = 0)
        assert!((stats.variances[0]).abs() < 1e-10);
        assert!((stats.variances[1]).abs() < 1e-10);
    }

    #[test]
    fn test_empty_source() {
        let shard = ScxCsr::new_unchecked((0, 3), vec![0], vec![], vec![]);
        let source = InMemorySource {
            shards: vec![shard],
            n_obs: 0,
            n_vars: 3,
        };
        // Should not panic on n=0
        let stats = streaming_mean_var(&source).unwrap();
        assert!(stats.means.iter().all(|&v| v.is_nan() || v == 0.0));
    }
}
