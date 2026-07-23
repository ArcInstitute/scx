//! Column-projecting [`ShardSource`] adapter.
//!
//! Wraps any [`ShardSource`] and presents a reduced column set to the streaming
//! kernels (PCA and friends), applying a sorted-merge column projection to each
//! decoded shard via [`scx_engine::project_csr`]. This is how `mask_var` /
//! `highly_variable` masking reaches the out-of-core PCA paths (backed, lazy,
//! and in-memory single-shard sources alike) **without materializing** the
//! full-width matrix: only the selected columns ever enter the kernel.
//!
//! The projection re-runs per `read_shard(_arc)` call, but `read_shard_arc`
//! delegates to the inner source's cached decode first, so a caching backed
//! reader still decodes each shard once per pass and only the (cheap) column
//! remap repeats.

use std::sync::Arc;

use scx_format_io::shard_source::ShardSource;
use scx_format_io::Result;
use scx_sparse::ScxCsr;

/// A [`ShardSource`] that projects each shard of an inner source down to a
/// fixed set of columns (`cols`, indices into the inner source's var axis).
///
/// `n_vars()` reports the projected column count; every returned shard has its
/// columns remapped to `0..cols.len()` in the order produced by
/// [`scx_engine::project_csr`] (sorted-unique). Callers that need to map a
/// projected column back to the original axis use `cols`.
pub struct ProjectedShardSource<'a, S: ShardSource + ?Sized> {
    inner: &'a S,
    cols: Vec<u32>,
}

impl<'a, S: ShardSource + ?Sized> ProjectedShardSource<'a, S> {
    /// Wrap `inner`, projecting every shard to `cols` (column indices into the
    /// inner var axis). `cols` should be sorted-unique for a stable
    /// projected → original mapping (`project_csr` sorts/dedups defensively).
    pub fn new(inner: &'a S, cols: Vec<u32>) -> Self {
        Self { inner, cols }
    }

    /// The projected → original column indices (length `n_vars()`).
    pub fn cols(&self) -> &[u32] {
        &self.cols
    }
}

impl<S: ShardSource + ?Sized> ShardSource for ProjectedShardSource<'_, S> {
    fn n_shards(&self) -> usize {
        self.inner.n_shards()
    }

    fn n_obs(&self) -> usize {
        self.inner.n_obs()
    }

    fn n_vars(&self) -> usize {
        self.cols.len()
    }

    fn read_shard(&self, shard_idx: usize) -> Result<ScxCsr> {
        let full = self.inner.read_shard(shard_idx)?;
        Ok(scx_engine::project_csr(&full, &self.cols))
    }

    fn read_shard_arc(&self, shard_idx: usize) -> Result<Arc<ScxCsr>> {
        // Decode via the inner source's (possibly cached) path, then project.
        let full = self.inner.read_shard_arc(shard_idx)?;
        Ok(Arc::new(scx_engine::project_csr(&full, &self.cols)))
    }

    fn shard_cache_capacity(&self) -> Option<usize> {
        self.inner.shard_cache_capacity()
    }

    fn max_shard_rows(&self) -> Result<usize> {
        // Column projection does not change row counts.
        self.inner.max_shard_rows()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_format_io::shard_source::SingleShardSource;

    fn csr_3x4() -> ScxCsr {
        // Dense:
        // row0: [1, 0, 2, 0]
        // row1: [0, 3, 0, 4]
        // row2: [5, 0, 0, 6]
        ScxCsr {
            indptr: vec![0, 2, 4, 6],
            indices: vec![0, 2, 1, 3, 0, 3],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape: (3, 4),
        }
    }

    #[test]
    fn projects_columns_and_reports_reduced_n_vars() {
        let csr = csr_3x4();
        let inner = SingleShardSource { csr: &csr };
        // Keep columns {0, 3}.
        let proj = ProjectedShardSource::new(&inner, vec![0, 3]);
        assert_eq!(proj.n_vars(), 2);
        assert_eq!(proj.n_obs(), 3);
        assert_eq!(proj.n_shards(), 1);

        let out = proj.read_shard(0).unwrap();
        assert_eq!(out.shape, (3, 2));
        // row0: col0=1 (kept as local 0), col3 absent → [1, 0]
        // row1: col3=4 → local1 → [0, 4]
        // row2: col0=5, col3=6 → [5, 6]
        let dense = |r: usize| {
            let s = out.indptr[r] as usize;
            let e = out.indptr[r + 1] as usize;
            let mut v = vec![0.0f32; 2];
            for j in s..e {
                v[out.indices[j] as usize] = out.data[j];
            }
            v
        };
        assert_eq!(dense(0), vec![1.0, 0.0]);
        assert_eq!(dense(1), vec![0.0, 4.0]);
        assert_eq!(dense(2), vec![5.0, 6.0]);
    }
}
