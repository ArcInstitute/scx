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
            Some(col_sums.iter().map(|s| s / n).collect())
        } else {
            None
        };

        Ok((means, col_sum_sq))
    }
}
