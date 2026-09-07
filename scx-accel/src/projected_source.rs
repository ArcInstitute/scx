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

    /// Forwarded: a column projection changes the width of a shard, never
    /// which shards hold a visible row. Without this the wrapper answers the
    /// trait default (`None` = visit every shard) and swallows the inner
    /// source's row-projection skip — which is the composition PCA takes
    /// whenever `mask_var=` is set (`adata.var["highly_variable"]` is
    /// auto-consumed), i.e. the HVG → PCA pipeline on a row subset.
    fn visible_shard_indices(&self) -> Option<Vec<usize>> {
        self.inner.visible_shard_indices()
    }

    fn shard_cache_capacity(&self) -> Option<usize> {
        self.inner.shard_cache_capacity()
    }

    fn max_shard_rows(&self) -> Result<usize> {
        // Column projection does not change row counts.
        self.inner.max_shard_rows()
    }

    fn shard_size_hint(&self) -> Option<scx_format_io::ShardSizeHint> {
        // Forwarded unchanged: rows are untouched by a column projection and
        // `max_nnz` can only shrink, so the inner value stays a valid **upper
        // bound** — which is all the contract promises. Deliberately not scaled
        // by `cols.len() / inner.n_vars()`: nonzeros are not uniform across
        // columns, so a scaled figure could under-bound, and an under-bound is
        // the one thing a consumer may not receive.
        self.inner.shard_size_hint()
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

    /// A source that filters rows tells the prefetch drivers which shards can
    /// still contribute one; a wrapper that does not forward that answers the
    /// trait default (`None` = visit every shard) and silently discards the
    /// plan. A column projection changes a shard's *width*, never which shards
    /// hold a visible row, so forwarding is unconditionally right.
    ///
    /// Pinned here rather than through a decode count: the only in-tree caller
    /// that wraps a row-filtering source is the **GPU** PCA dispatch arm
    /// (`#[cfg(feature = "gpu")]`), so a CPU build cannot observe it end to end.
    #[test]
    fn forwards_the_visible_shard_plan_of_the_inner_source() {
        struct Planned<'a> {
            csr: &'a ScxCsr,
            plan: Option<Vec<usize>>,
        }
        impl ShardSource for Planned<'_> {
            fn n_shards(&self) -> usize {
                3
            }
            fn n_obs(&self) -> usize {
                self.csr.shape.0
            }
            fn n_vars(&self) -> usize {
                self.csr.shape.1
            }
            fn read_shard(&self, _idx: usize) -> scx_format_io::Result<ScxCsr> {
                Ok(self.csr.clone())
            }
            fn visible_shard_indices(&self) -> Option<Vec<usize>> {
                self.plan.clone()
            }
        }

        let csr = csr_3x4();
        let planned = Planned {
            csr: &csr,
            plan: Some(vec![0, 2]),
        };
        let proj = ProjectedShardSource::new(&planned, vec![0, 3]);
        assert_eq!(
            proj.visible_shard_indices(),
            Some(vec![0, 2]),
            "the projection must not swallow the inner source's shard plan"
        );

        // And "no filter" stays "no filter", rather than becoming an empty plan.
        let unfiltered = Planned {
            csr: &csr,
            plan: None,
        };
        let proj = ProjectedShardSource::new(&unfiltered, vec![0, 3]);
        assert_eq!(proj.visible_shard_indices(), None);
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

    /// A source that advertises a hint, so the forward is observable.
    struct HintedSource<'a> {
        csr: &'a ScxCsr,
        hint: Option<scx_format_io::ShardSizeHint>,
    }

    impl ShardSource for HintedSource<'_> {
        fn n_shards(&self) -> usize {
            1
        }
        fn n_obs(&self) -> usize {
            self.csr.n_rows()
        }
        fn n_vars(&self) -> usize {
            self.csr.n_cols()
        }
        fn read_shard(&self, _shard_idx: usize) -> Result<ScxCsr> {
            Ok(self.csr.clone())
        }
        fn shard_size_hint(&self) -> Option<scx_format_io::ShardSizeHint> {
            self.hint
        }
    }

    /// The hint must reach a consumer through the projection, unscaled.
    ///
    /// Scaling `max_nnz` by the kept-column fraction would look tidier and be
    /// wrong: nonzeros are not uniform across columns, so a scaled figure can
    /// *under*-bound, and the contract forbids under-bounding. Over-bounding
    /// only costs a slightly smaller prefetch depth or cache carve.
    #[test]
    fn forwards_the_inner_shard_size_hint_unscaled() {
        let csr = csr_3x4();
        let hint = scx_format_io::ShardSizeHint {
            max_rows: 3,
            max_nnz: 6,
        };
        let inner = HintedSource {
            csr: &csr,
            hint: Some(hint),
        };
        let proj = ProjectedShardSource::new(&inner, vec![0]);
        assert_eq!(proj.shard_size_hint(), Some(hint));

        // And "no cheap estimate" stays "no cheap estimate" — never zero.
        let blind = HintedSource {
            csr: &csr,
            hint: None,
        };
        assert_eq!(
            ProjectedShardSource::new(&blind, vec![0]).shard_size_hint(),
            None
        );
    }
}
