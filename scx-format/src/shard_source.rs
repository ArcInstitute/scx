//! Shard-by-shard data source abstractions.
//!
//! Two parallel traits live here — neither extends the other:
//!
//! - [`ShardSource`]: row-major (CSR) streaming. Existing analytical
//!   kernels (PCA, HVG full-pass, projected aggregations) take
//!   `S: ShardSource` and walk shards in row-shard order.
//! - [`ColumnShardSource`]: column-major (CSC) streaming. Kernels that
//!   want column-axis access (DE on a gene subset, projected `col_*`
//!   aggregations on a CSC sidecar) take `S: ColumnShardSource`
//!   directly. Kernels that need both compose `S: ShardSource +
//!   ColumnShardSource`.
//!
//! Capability detection is encoded in the type system: a function that
//! requires CSC takes a `ColumnShardSource` bound, and callers that
//! don't have CSC simply cannot construct it. The runtime
//! "does this dataset have CSC?" question is answered exactly once at
//! the pyscx wrapper layer (`ScxBackedSparseDataset::as_column_source`),
//! never inside the generic kernels.

use std::ops::Range;

use scx_sparse::{ScxCsc, ScxCsr};

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
    /// (e.g., GPU pinned-slot staging, covariance-PCA Gram densification
    /// scratch).
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

/// Column-major shard source — a parallel trait to [`ShardSource`] for
/// CSC (column-major) data access.
///
/// Implementors return [`ScxCsc`] from a per-shard or column-range read.
/// The trait is **not** a sub-trait of [`ShardSource`]: a type that can
/// serve only CSR shards (no CSC sidecar) does not implement
/// `ColumnShardSource`, and a type that has only a CSC sidecar would not
/// implement `ShardSource`. Composing both bounds expresses dual access.
///
/// # Capability detection
///
/// There is no `has_csc()` method on `ShardSource` and no
/// fallible-by-default methods on this trait. Capability is encoded by
/// whether a type implements the trait at all. The pyscx wrapper layer
/// (`ScxBackedSparseDataset::as_column_source`) is the single runtime
/// gate that decides whether a dataset can be cast to
/// `&dyn ColumnShardSource`.
pub trait ColumnShardSource {
    /// Number of CSC shards.
    fn n_csc_shards(&self) -> usize;

    /// Total number of observations (rows). Must match the underlying
    /// CSR view; CSC shards span the full row axis.
    fn n_obs(&self) -> usize;

    /// Number of variables (columns).
    fn n_vars(&self) -> usize;

    /// Shape as `(n_obs, n_vars)`.
    fn shape(&self) -> (usize, usize) {
        (self.n_obs(), self.n_vars())
    }

    /// Read and decode a single CSC shard.
    fn read_csc_shard(&self, shard_idx: usize) -> Result<ScxCsc>;

    /// Read a contiguous column slice across shards.
    ///
    /// Implementors SHOULD skip shards whose `[col_start, col_end)`
    /// does not intersect `col_range`, and `col_slice` partially
    /// overlapping shards post-decode.
    fn read_csc_columns(&self, col_range: Range<u32>) -> Result<ScxCsc>;

    /// Per-shard `[col_start, col_end)` from catalog stats.
    ///
    /// Returns `None` for out-of-range indices. Useful for callers that
    /// want to plan a query (chunk size, skip count) before issuing
    /// `read_csc_columns`.
    fn csc_shard_col_range(&self, shard_idx: usize) -> Option<(u32, u32)>;
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

    /// Stub for the `ColumnShardSource` smoke tests — the real impl
    /// lives in `BackedCscReader` (`backed.rs`) and `LazyShardSource`
    /// (pyscx).
    struct StubColumnSource {
        shards: Vec<ScxCsc>,
        ranges: Vec<(u32, u32)>,
        n_obs: usize,
        n_vars: usize,
    }

    impl ColumnShardSource for StubColumnSource {
        fn n_csc_shards(&self) -> usize {
            self.shards.len()
        }
        fn n_obs(&self) -> usize {
            self.n_obs
        }
        fn n_vars(&self) -> usize {
            self.n_vars
        }
        fn read_csc_shard(&self, shard_idx: usize) -> Result<ScxCsc> {
            Ok(self.shards[shard_idx].clone())
        }
        fn read_csc_columns(&self, col_range: Range<u32>) -> Result<ScxCsc> {
            // Tiny stub: full-range only. Real impls must do
            // shard-skipping and post-decode `col_slice`.
            assert_eq!(col_range, 0..self.n_vars as u32);
            // Concatenate all shards along the column axis using the
            // ScxCsc concatenation logic mirrored from reader.rs.
            let mut indptr = vec![0i64];
            let mut indices = Vec::new();
            let mut data = Vec::new();
            let mut cum_nnz: i64 = 0;
            for shard in &self.shards {
                for i in 1..=shard.n_cols() {
                    indptr.push(shard.indptr[i] + cum_nnz);
                }
                cum_nnz += shard.indptr[shard.n_cols()];
                indices.extend_from_slice(&shard.indices);
                data.extend_from_slice(&shard.data);
            }
            Ok(ScxCsc::new_unchecked(
                (self.n_obs, self.n_vars),
                indptr,
                indices,
                data,
            ))
        }
        fn csc_shard_col_range(&self, shard_idx: usize) -> Option<(u32, u32)> {
            self.ranges.get(shard_idx).copied()
        }
    }

    #[test]
    fn column_shard_source_default_shape() {
        let src = StubColumnSource {
            shards: vec![],
            ranges: vec![],
            n_obs: 5,
            n_vars: 3,
        };
        assert_eq!(src.shape(), (5, 3));
    }

    #[test]
    fn column_shard_source_basic_dispatch() {
        // 3 rows × 2 cols; one shard covering both columns.
        let csc = ScxCsc::new_unchecked((3, 2), vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]);
        let src = StubColumnSource {
            shards: vec![csc.clone()],
            ranges: vec![(0, 2)],
            n_obs: 3,
            n_vars: 2,
        };
        assert_eq!(src.n_csc_shards(), 1);
        assert_eq!(src.csc_shard_col_range(0), Some((0, 2)));
        assert_eq!(src.csc_shard_col_range(99), None);
        let s0 = src.read_csc_shard(0).unwrap();
        assert_eq!(s0.shape, (3, 2));
        let full = src.read_csc_columns(0..2).unwrap();
        assert_eq!(full.shape, (3, 2));
    }

    /// Compile-only check: a function bounded on `ColumnShardSource`
    /// must NOT accept a value typed as `&dyn ShardSource`. We can't
    /// test the negative compile case directly in unit tests, but we
    /// can verify the positive case (the bound applies and we reach
    /// the methods) and rely on rustc's trait machinery for the
    /// negative.
    #[test]
    fn column_shard_source_trait_independent_of_shard_source() {
        fn requires_csc<S: ColumnShardSource>(s: &S) -> usize {
            s.n_csc_shards()
        }
        let csc = ScxCsc::new_unchecked((1, 0), vec![0], vec![], vec![]);
        let src = StubColumnSource {
            shards: vec![csc],
            ranges: vec![(0, 0)],
            n_obs: 1,
            n_vars: 0,
        };
        assert_eq!(requires_csc(&src), 1);
        // The `StubSource` (CSR) deliberately does NOT implement
        // ColumnShardSource — uncommenting the line below should fail
        // at compile time:
        //     requires_csc(&StubSource { shards: vec![], n_obs: 0, n_vars: 0 });
    }
}
