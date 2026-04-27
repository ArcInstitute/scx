//! Shard-by-shard data source abstraction.
//!
//! The [`ShardSource`] trait provides a uniform interface for streaming
//! CSR data shard-by-shard, whether from on-disk backed storage or
//! from lazy-transformed datasets with per-shard transform application.

use scx_sparse::ScxCsr;

use crate::error::Result;

/// A source of CSR shards for streaming computation.
///
/// Implementations provide sequential shard access for algorithms like
/// PCA that process data one shard at a time without materializing the
/// full matrix.
pub trait ShardSource {
    /// Number of shards in this source.
    fn n_shards(&self) -> usize;

    /// Total number of observations (rows) across all shards.
    fn n_obs(&self) -> usize;

    /// Number of variables (columns).
    fn n_vars(&self) -> usize;

    /// Shape as `(n_obs, n_vars)`.
    fn shape(&self) -> (usize, usize) {
        (self.n_obs(), self.n_vars())
    }

    /// Read and decode shard `shard_idx`.
    ///
    /// Implementations may apply transforms (e.g., NormalizeTotal, Log1p)
    /// and/or deletion vector filtering before returning.
    fn read_shard(&self, shard_idx: usize) -> Result<ScxCsr>;

    /// Maximum number of rows across all shards.
    ///
    /// Used by GPU callers to size per-shard scratch buffers up-front
    /// (e.g., `DoubleBufferedShardLoader` pinned slots, covariance-PCA
    /// Gram densification scratch).
    ///
    /// The default implementation reads every shard once — correct but
    /// potentially expensive. Implementors with O(1) access to shard
    /// metadata (e.g., `BackedCsrReader` via its `BackedCsrIndex`) should
    /// override with a cheap version.
    fn max_shard_rows(&self) -> Result<usize> {
        let mut max_rows = 0usize;
        for shard_idx in 0..self.n_shards() {
            let rows = self.read_shard(shard_idx)?.n_rows();
            if rows > max_rows {
                max_rows = rows;
            }
        }
        Ok(max_rows)
    }

    /// Compute column means and per-column sum-of-squares in a single
    /// streaming pass over all shards.
    ///
    /// Returns `(means, col_sum_sq)` where:
    /// - `means`: `Some(Vec<f64>)` of length `n_vars` if `zero_center`, else `None`
    /// - `col_sum_sq`: `Vec<f64>` of length `n_vars` — per-column Σ x²
    fn col_means_and_sum_sq(&self, zero_center: bool) -> Result<(Option<Vec<f64>>, Vec<f64>)> {
        let n_vars = self.n_vars();
        let mut col_sums = vec![0.0f64; n_vars];
        let mut col_sum_sq = vec![0.0f64; n_vars];

        for shard_idx in 0..self.n_shards() {
            let csr = self.read_shard(shard_idx)?;
            for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                let v = val as f64;
                col_sums[col as usize] += v;
                col_sum_sq[col as usize] += v * v;
            }
        }

        let means = if zero_center {
            let n = self.n_obs() as f64;
            if n == 0.0 {
                Some(vec![0.0f64; n_vars])
            } else {
                Some(col_sums.iter().map(|s| s / n).collect())
            }
        } else {
            None
        };

        Ok((means, col_sum_sq))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubSource {
        shards: Vec<ScxCsr>,
        n_obs: usize,
        n_vars: usize,
    }

    impl ShardSource for StubSource {
        fn n_shards(&self) -> usize {
            self.shards.len()
        }
        fn n_obs(&self) -> usize {
            self.n_obs
        }
        fn n_vars(&self) -> usize {
            self.n_vars
        }
        fn read_shard(&self, shard_idx: usize) -> Result<ScxCsr> {
            Ok(self.shards[shard_idx].clone())
        }
    }

    #[test]
    fn max_shard_rows_default_impl() {
        // 3 shards with row counts 2, 5, 3 → max = 5.
        let shards = vec![
            ScxCsr::new_unchecked((2, 3), vec![0, 1, 1], vec![0], vec![1.0]),
            ScxCsr::new_unchecked((5, 3), vec![0, 0, 0, 0, 0, 0], vec![], vec![]),
            ScxCsr::new_unchecked((3, 3), vec![0, 1, 1, 2], vec![2, 0], vec![1.0, 2.0]),
        ];
        let src = StubSource {
            shards,
            n_obs: 10,
            n_vars: 3,
        };
        assert_eq!(src.max_shard_rows().unwrap(), 5);
    }

    #[test]
    fn max_shard_rows_empty() {
        let src = StubSource {
            shards: vec![],
            n_obs: 0,
            n_vars: 3,
        };
        assert_eq!(src.max_shard_rows().unwrap(), 0);
    }
}
