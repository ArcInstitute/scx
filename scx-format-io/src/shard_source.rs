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
use std::sync::Arc;

use scx_sparse::{ScxCsc, ScxCsr};

use crate::error::Result;

/// Cheap upper bounds on the largest shard, from
/// [`ShardSource::shard_size_hint`].
///
/// Both fields are upper bounds — see that method for why exactness is not
/// promised and why the two travel together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardSizeHint {
    /// Upper bound on rows in any one shard.
    pub max_rows: usize,
    /// Upper bound on nonzeros in any one shard.
    pub max_nnz: usize,
}

impl ShardSizeHint {
    /// Bytes one decoded shard of this size occupies as an
    /// [`ScxCsr`](scx_sparse::ScxCsr): `indptr` i64 + `indices` i32 + `data`
    /// f32. Saturating, so a nonsense hint cannot wrap into a small budget.
    pub fn decoded_bytes(&self) -> u64 {
        (self.max_rows as u64 + 1)
            .saturating_mul(8)
            .saturating_add((self.max_nnz as u64).saturating_mul(8))
    }
}

/// Whether a shard spanning `[shard_start, shard_end)` intersects `range`.
///
/// **Half-open on both ends**: a shard ending exactly at `range.start` does not
/// overlap, and one starting exactly at `range.end` does not either.
///
/// A free function because this one boundary condition had four hand-written
/// copies — `FullCatalog::csc_shards_for_col_range`,
/// `BackedCscIndex::shards_for_col_range`, an inline loop in `scx-gpu`'s CSC
/// staging path, and `GpuCscShardSource`'s post-decode filter — which is three
/// more places for an off-by-one to live than the rule deserves. The binary
/// search in `BackedCscIndex` necessarily states it differently; an exhaustive
/// differential test pins it to this.
#[inline]
pub fn col_range_overlaps(shard_start: u32, shard_end: u32, range: &Range<u32>) -> bool {
    // The empty/inverted guard is not redundant with the half-open comparisons.
    // Without it `col_range_overlaps(4, 8, &(5..5))` is `true` (`8 > 5 && 4 < 5`)
    // while `BackedCscIndex::shards_for_col_range` returns nothing for the same
    // input, because it guards `c_lo >= c_hi` up front — so the trait default
    // and the backed override disagreed on an interior empty range, in a
    // function this PR made the single named boundary rule.
    //
    // Latent rather than live (production GPU DE passes non-empty gene chunks),
    // and found by Cursor Agent - Grok 4.6 High and Antigravity - Gemini 3.7
    // Flash independently.
    range.start < range.end && shard_end > range.start && shard_start < range.end
}

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

    /// Read shard `shard_idx`, returning a shared handle that a cached
    /// implementation can hand out without re-decoding.
    ///
    /// Multi-pass kernels (e.g. out-of-core PCA, which makes ~6–7 passes
    /// over every shard) must call this instead of [`read_shard`] so a
    /// caching source decodes each shard once and reuses the `Arc`. The
    /// default wraps [`read_shard`] in a fresh `Arc` (no caching), so
    /// non-caching sources behave exactly as before. `BackedCsrReader`
    /// overrides this to serve from its LRU shard cache.
    ///
    /// [`read_shard`]: ShardSource::read_shard
    fn read_shard_arc(&self, shard_idx: usize) -> Result<Arc<ScxCsr>> {
        Ok(Arc::new(self.read_shard(shard_idx)?))
    }

    /// Capacity (in shards) of this source's decode cache, if it has one.
    ///
    /// `None` means the source does not cache (every `read_shard_arc` call
    /// re-decodes). `Some(cap)` lets multi-pass kernels warn when
    /// `cap < n_shards`, i.e. the cache is too small to hold the working
    /// set and will evict + re-decode each pass — mirroring the DE
    /// streaming path's undersized-cache warning.
    fn shard_cache_capacity(&self) -> Option<usize> {
        None
    }

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

    /// Largest shard's dimensions, when both are available **without
    /// decoding**.
    ///
    /// `None` — the default — means "no cheap estimate", not "zero". A caller
    /// must treat the absence as unknown and fall back to growing buffers on
    /// demand; it must never read `None` as a small number.
    ///
    /// Deliberately **not** split into a row hint and an nnz hint, and
    /// deliberately not modelled on [`max_shard_rows`] (whose default decodes
    /// every shard). The point of a hint is to *avoid* work — pre-sizing GPU
    /// staging buffers so they never grow-and-realloc, and deriving a
    /// per-shard byte estimate for the decode-prefetch depth clamp. A caller
    /// needs both numbers together, and one method returning both means an
    /// implementor cannot supply a cheap nnz while leaving the caller to fall
    /// into the expensive default for rows. Implementors with O(1) catalog
    /// access override this; everyone else returns `None`.
    ///
    /// Both fields are **upper bounds**, not exact counts: a reader over a
    /// column-projected or deletion-filtered view reports the on-disk shard
    /// statistics, which over-count. Over-estimating only costs a slightly
    /// larger buffer or a slightly smaller prefetch depth.
    ///
    /// [`max_shard_rows`]: ShardSource::max_shard_rows
    fn shard_size_hint(&self) -> Option<ShardSizeHint> {
        None
    }

    /// Compute column means and per-column sum-of-squares in a single
    /// streaming pass over all shards.
    ///
    /// Returns `(means, col_sum_sq)` where:
    /// - `means`: `Some(Vec<f64>)` of length `n_vars` if `zero_center`, else `None`
    /// - `col_sum_sq`: `Vec<f64>` of length `n_vars` — per-column Σ x²
    ///
    /// **Decodes one shard at a time on the calling thread.** A caller that can
    /// name a `Sync` source should prefer
    /// [`col_means_and_sum_sq_prefetched`](crate::prefetch::col_means_and_sum_sq_prefetched),
    /// which overlaps decode across shards and is bit-identical to this. This
    /// stays as the fallback for `&dyn ShardSource` callers *and* as that
    /// function's test oracle, so the two must not drift.
    fn col_means_and_sum_sq(&self, zero_center: bool) -> Result<(Option<Vec<f64>>, Vec<f64>)> {
        let n_vars = self.n_vars();
        let mut col_sums = vec![0.0f64; n_vars];
        let mut col_sum_sq = vec![0.0f64; n_vars];

        for shard_idx in 0..self.n_shards() {
            let csr = self.read_shard_arc(shard_idx)?;
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

/// A [`ShardSource`] over a single in-memory CSR matrix (one shard).
///
/// Adapts an already-materialized [`ScxCsr`] to the streaming kernels (PCA,
/// HVG, …) for small or in-memory matrices that don't come from a backed
/// reader. Shared by the `pyscx` and `rscx` bindings, which each previously
/// carried an identical bespoke single-shard adapter.
pub struct SingleShardSource<'a> {
    /// The borrowed in-memory matrix served as shard `0`.
    pub csr: &'a ScxCsr,
}

impl ShardSource for SingleShardSource<'_> {
    fn n_shards(&self) -> usize {
        1
    }
    fn n_obs(&self) -> usize {
        self.csr.n_rows()
    }
    fn n_vars(&self) -> usize {
        self.csr.n_cols()
    }
    fn read_shard(&self, shard_idx: usize) -> Result<ScxCsr> {
        if shard_idx != 0 {
            return Err(crate::error::ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: 1,
            });
        }
        Ok(self.csr.clone())
    }
    fn max_shard_rows(&self) -> Result<usize> {
        Ok(self.csr.n_rows())
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

    /// Column-major twin of [`ShardSource::shard_size_hint`]: cheap upper
    /// bounds on the largest CSC shard, or `None` when the implementor cannot
    /// answer without decoding.
    ///
    /// ⚠️ **`max_rows` counts the major axis, which on this layout is
    /// columns.** `ShardSizeHint` is shared with the row-major side, where the
    /// major axis is rows; the field name follows that side. So a CSC consumer
    /// sizing an `indptr` buffer wants `max_rows + 1` here just as a CSR one
    /// does — the arrays are laid out identically, only the axis they index
    /// differs. `decoded_bytes()` is correct on either layout for the same
    /// reason.
    ///
    /// Exists because the GPU CSC staging path derates its decode-prefetch
    /// depth to a host-RAM budget, and a budget it cannot price is a budget
    /// that silently does nothing.
    fn csc_shard_size_hint(&self) -> Option<ShardSizeHint> {
        None
    }

    /// Every shard index whose `[col_start, col_end)` intersects `col_range`,
    /// **ascending**.
    ///
    /// The default is an O(n_shards) scan over [`csc_shard_col_range`], which
    /// is the same predicate three call sites had each spelled out for
    /// themselves. An implementor holding a sorted column index overrides this
    /// with a binary search; the two then differ only in complexity, not in the
    /// boundary condition — half-open on both ends, so a shard ending exactly
    /// at `col_range.start` does not overlap.
    ///
    /// An empty or inverted `col_range` selects nothing.
    ///
    /// [`csc_shard_col_range`]: ColumnShardSource::csc_shard_col_range
    fn csc_shards_for_col_range(&self, col_range: Range<u32>) -> Vec<usize> {
        if col_range.start >= col_range.end {
            return Vec::new();
        }
        (0..self.n_csc_shards())
            .filter(|&i| {
                self.csc_shard_col_range(i)
                    .is_some_and(|(s, e)| col_range_overlaps(s, e, &col_range))
            })
            .collect()
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
            // ScxCsc concatenation logic mirrored from reader/matrix.rs.
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

    /// Column-major stub whose shard `i` covers `[3i, 3i+3)`, except a short
    /// tail shard — the shape `write_csc_test_file` produces.
    struct StubCscRanges {
        ranges: Vec<(u32, u32)>,
    }

    impl ColumnShardSource for StubCscRanges {
        fn n_csc_shards(&self) -> usize {
            self.ranges.len()
        }
        fn n_obs(&self) -> usize {
            12
        }
        fn n_vars(&self) -> usize {
            self.ranges.last().map(|r| r.1 as usize).unwrap_or(0)
        }
        fn read_csc_shard(&self, _i: usize) -> Result<ScxCsc> {
            unreachable!("range planning does not decode")
        }
        fn read_csc_columns(&self, _r: Range<u32>) -> Result<ScxCsc> {
            unreachable!("range planning does not decode")
        }
        fn csc_shard_col_range(&self, i: usize) -> Option<(u32, u32)> {
            self.ranges.get(i).copied()
        }
    }

    fn stub_csc() -> StubCscRanges {
        StubCscRanges {
            ranges: vec![(0, 3), (3, 6), (6, 9), (9, 10)],
        }
    }

    /// The default `csc_shards_for_col_range` is half-open on **both** ends.
    ///
    /// Hand-computed, not derived from the implementation: a shard ending
    /// exactly at `col_range.start` does not overlap, and one starting exactly
    /// at `col_range.end` does not either. Both are the boundary the GPU
    /// prefilter and `BackedCscIndex`'s binary search must agree with.
    #[test]
    fn csc_shards_for_col_range_default_is_half_open_at_both_ends() {
        let src = stub_csc();
        assert_eq!(src.csc_shards_for_col_range(4..8), vec![1, 2]);
        assert_eq!(src.csc_shards_for_col_range(0..10), vec![0, 1, 2, 3]);
        // [0,3) touches shard 0 only — shard 1 starts exactly at 3.
        assert_eq!(src.csc_shards_for_col_range(0..3), vec![0]);
        // [3,3) is empty; [3,4) is shard 1 alone.
        assert!(src.csc_shards_for_col_range(3..3).is_empty());
        assert_eq!(src.csc_shards_for_col_range(3..4), vec![1]);
        // A range that ends exactly where a shard begins excludes it.
        assert_eq!(src.csc_shards_for_col_range(2..6), vec![0, 1]);
        // Past the end, and inverted. The inverted range is built rather than
        // written as a literal because clippy rejects a reversed literal range
        // outright — which would delete the case, not fix it.
        assert!(src.csc_shards_for_col_range(100..200).is_empty());
        let inverted = std::ops::Range { start: 8, end: 2 };
        assert!(src.csc_shards_for_col_range(inverted).is_empty());
    }

    /// A source that does not override reports no size hint, rather than a
    /// fabricated zero — an under-estimate is what the hint contract forbids.
    #[test]
    fn csc_shard_size_hint_defaults_to_none() {
        assert_eq!(stub_csc().csc_shard_size_hint(), None);
    }
}
