// LazyShardSource — ShardSource / ColumnShardSource for streaming pipelines.
//
// Extracted from the former pyscx/src/lazy_transform.rs (T5.7).

use std::sync::{Arc, OnceLock};

use scx_format_io::{BackedCscReader, BackedCsrReader};
use scx_sparse::{ScxCsc, ScxCsr};

use super::*;

#[cfg(test)]
#[path = "shard_source_tests.rs"]
mod tests;

/// Shard source that applies lazy transforms per-shard.
///
/// Enables streaming PCA (and other shard-by-shard algorithms) on
/// lazy-transformed data without materializing the full matrix.
pub(crate) struct LazyShardSource {
    backed: Arc<BackedCsrReader>,
    /// Optional CSC sidecar reader. Populated when the underlying file
    /// has CSC shards AND the open path requests CSC capability.
    /// `None` ⇒ this `LazyShardSource` cannot serve `ColumnShardSource`
    /// methods (they will return an error).
    backed_csc: Option<Arc<BackedCscReader>>,
    transforms: Vec<Transform>,
    kept_to_global: Option<Arc<Vec<u64>>>,
    col_projection: Option<Arc<Vec<u32>>>,
    /// Inverse of `kept_to_global`, memoised. See [`Self::global_to_live`].
    global_to_live: OnceLock<Vec<i32>>,
    shape_val: (usize, usize),
    /// Serve shard decodes from the reader's decoded-shard LRU instead of
    /// decoding fresh every time. See [`Self::with_cached_reads`].
    cached_reads: bool,
}

impl LazyShardSource {
    /// Create a shard source with an optional pre-existing kept-to-global mapping.
    ///
    /// Pass `None` for `kept_to_global` to signal "all rows kept" — this skips
    /// the deletion-vector filtering path in `read_shard` and avoids allocating
    /// a full identity range vector.
    pub(crate) fn new(
        backed: Arc<BackedCsrReader>,
        transforms: Vec<Transform>,
        kept_to_global: Option<Arc<Vec<u64>>>,
        col_projection: Option<Arc<Vec<u32>>>,
        n_obs: usize,
        n_vars: usize,
    ) -> Self {
        LazyShardSource {
            backed,
            backed_csc: None,
            transforms,
            kept_to_global,
            col_projection,
            global_to_live: OnceLock::new(),
            shape_val: (n_obs, n_vars),
            cached_reads: false,
        }
    }

    /// Serve shard decodes from the wrapped reader's decoded-shard LRU.
    ///
    /// **Multi-pass kernels must opt in.** Out-of-core PCA makes ~6–7 passes
    /// over every shard; without this the source decodes through
    /// `read_shard_uncached`, which also `MADV_DONTNEED`s the shard bytes, so
    /// each pass re-faults *and* re-decodes. Opting in also republishes the
    /// reader's [`shard_cache_capacity`], which is what feeds the
    /// undersized-cache warning and makes `pca(memory_budget=…)` /
    /// `ensure_cache_capacity` mean anything.
    ///
    /// Off by default: single-pass streaming callers (HVG, `score_genes`,
    /// `pflog`) visit each shard once, where the LRU is pure overhead and the
    /// `MADV_DONTNEED` is a win.
    ///
    /// [`shard_cache_capacity`]: scx_format_io::ShardSource::shard_cache_capacity
    pub(crate) fn with_cached_reads(mut self) -> Self {
        self.cached_reads = true;
        self
    }

    /// Decode one shard, honouring [`Self::with_cached_reads`].
    /// Through the `ShardSource` trait rather than the inherent
    /// `read_shard_cached_arc` / `read_shard_uncached` it used to call.
    ///
    /// Behaviour-identical — the trait methods *are* those two — but they carry
    /// `ensure_row_addressable`, so this wrapper cannot walk the shards of an
    /// unscoped reader over overlapping row ranges and hand back one arbitrary
    /// modality. Not reachable from Python today (`to_anndata` refuses on
    /// geometry first, and `open_backed_csr` requires `modality=` when the file
    /// names more than one), but this is exactly the shape that hid
    /// `col_means_and_sum_sq`: an inherent call that the trait's guard never
    /// sees.
    fn decode_shard(&self, shard_idx: usize) -> scx_format_io::Result<Arc<ScxCsr>> {
        use scx_format_io::shard_source::ShardSource;
        if self.cached_reads {
            ShardSource::read_shard_arc(&*self.backed, shard_idx)
        } else {
            Ok(Arc::new(ShardSource::read_shard(&*self.backed, shard_idx)?))
        }
    }

    /// Create a shard source with both CSR and CSC backings.
    ///
    /// Used by callers that want CSC-capable streaming. The CSC reader
    /// must already be constructed (typically by the caller after
    /// inspecting `header.has_csc()`).
    pub(crate) fn new_with_csc(
        backed: Arc<BackedCsrReader>,
        backed_csc: Option<Arc<BackedCscReader>>,
        transforms: Vec<Transform>,
        kept_to_global: Option<Arc<Vec<u64>>>,
        col_projection: Option<Arc<Vec<u32>>>,
        n_obs: usize,
        n_vars: usize,
    ) -> Self {
        LazyShardSource {
            backed,
            backed_csc,
            transforms,
            kept_to_global,
            col_projection,
            global_to_live: OnceLock::new(),
            shape_val: (n_obs, n_vars),
            cached_reads: false,
        }
    }

    /// Returns `true` if this lazy source can serve CSC reads: a CSC sidecar
    /// is present.
    ///
    /// Neither the transform chain nor a row filter is a condition, and both
    /// used to be. Every `Transform` is applied column-major by
    /// `apply_transforms_to_csc`, including the row-indexed `NormalizeTotal` /
    /// `RowScale`, which read their per-row vector at the global row
    /// `ScxCsc::indices` already carries. A row filter is handled by
    /// [`scx_engine::projection::compact_csc_rows_in_place`], which renumbers
    /// a slab's rows onto the live row space before it leaves this reader — so
    /// what a consumer
    /// receives is addressed the way it already assumes: rows are live indices
    /// and `n_obs()` is their count.
    ///
    /// What is left is not a predicate over this source's *state* at all, only
    /// over whether the file brought a sidecar, which is why there is nothing
    /// else to test here.
    ///
    /// Predicate used by both `as_column_source()` implementations —
    /// `ScxLazyTransformedDataset`'s and `ScxBackedSparseDataset`'s, which
    /// routes through this type precisely to get the compaction.
    pub(crate) fn supports_csc(&self) -> bool {
        self.backed_csc.is_some()
    }

    /// The inverse of `kept_to_global`: `global_to_live[g]` is the live index
    /// of global physical row `g`, or `-1` when the row is filtered out.
    ///
    /// The CSC path needs the inverse because a sidecar slab arrives addressed
    /// by global row while every consumer works in live row space. Built once
    /// per source on the first CSC read — 4 B x n_obs_physical, so 4 MB at a
    /// million rows — and never at all for the CSR consumers of
    /// `new_with_csc`, which is why it is a `OnceLock` rather than constructor
    /// work.
    ///
    /// `None` means no compaction is owed: either no row filter is active, or
    /// there is no sidecar to read in the first place.
    ///
    /// **`kept_to_global` must be ascending**, and the compaction depends on it
    /// more sharply than the CSR path does. A monotone map is what preserves
    /// each column's stored row order through the renumbering — which
    /// `scx-gpu`'s `validate_csc` requires and `scx_accel::csc::pseudobulk`
    /// relies on. A descending or shuffled map would come out unsorted and
    /// mislabel rather than fail. Every producer in the tree is ascending
    /// (`axis_align::compose_rows_positional`, `compute_kept_to_global`, and
    /// anndata's `_subset`, which materialises a non-ascending selection rather
    /// than expressing it as a window), so this is an assertion, not a branch.
    fn global_to_live(&self) -> Option<&[i32]> {
        let kept = self.kept_to_global.as_ref()?;
        let backed_csc = self.backed_csc.as_ref()?;
        Some(self.global_to_live.get_or_init(|| {
            debug_assert!(
                kept.windows(2).all(|w| w[0] < w[1]),
                "kept_to_global must be strictly ascending; a non-monotone map \
                 renumbers a column's rows out of order",
            );
            // Sized from the *sidecar's* row count, which is the space its
            // `indices` live in, not from this view's visible count.
            let mut map = vec![-1i32; backed_csc.n_obs()];
            for (live, &g) in kept.iter().enumerate() {
                if let Some(slot) = map.get_mut(g as usize) {
                    *slot = live as i32;
                }
            }
            map
        }))
    }

    /// Whether the **automatic** route should prefer CSC for this source.
    ///
    /// Deliberately separate from [`Self::supports_csc`], which answers whether
    /// CSC *can* be served. Using capability as policy is wrong in one
    /// direction that matters: a CSC column shard spans the whole row axis, so
    /// a narrow row window still decodes every physical cell of the columns it
    /// asks for and throws most of them away, where the CSR path skips the
    /// shards the window empties outright (`visible_shard_indices`). A
    /// `adata[:10_000]` view of a million-cell file is a full-height column
    /// read on the row-major path's terms.
    ///
    /// The discriminator is therefore how much of the file the window actually
    /// spans: CSR's shard skipping is worth something only when the kept rows
    /// leave whole shards empty, and worth nothing when every shard still has
    /// survivors — which is the ordinary `filter_cells` that keeps ~99 % of
    /// cells, where CSC's column locality wins. Half the shards is a coarse cut
    /// between those two regimes rather than a tuned constant, and it is **not
    /// measured**: it is chosen to be obviously right at both ends (an
    /// unfiltered or lightly filtered handle takes CSC, a handful of rows takes
    /// CSR) and its exact placement in between is not something this change
    /// establishes.
    ///
    /// Only `auto` consults this. An explicit `prefer_format="csc"` is served
    /// on any window the reader can compact — the capability is the caller's to
    /// spend.
    pub(crate) fn csc_preferred_for_auto(&self) -> bool {
        if !self.supports_csc() {
            return false;
        }
        let Some(kept) = self.kept_to_global.as_ref() else {
            // No row filter: there is nothing for CSR to prune.
            return true;
        };
        let index = self.backed.index();
        let n_shards = index.n_shards();
        if n_shards == 0 {
            return true;
        }
        index.shards_with_kept_rows(kept).len() * 2 >= n_shards
    }
}

impl scx_format_io::ShardSource for LazyShardSource {
    fn n_shards(&self) -> usize {
        self.backed.index().n_shards()
    }

    fn n_obs(&self) -> usize {
        self.shape_val.0
    }

    fn n_vars(&self) -> usize {
        match &self.col_projection {
            Some(cols) => cols.len(),
            None => self.shape_val.1,
        }
    }

    /// An **upper bound**, not the exact visible maximum.
    ///
    /// Transforms and column projection leave row counts alone, so this
    /// delegates to the wrapped reader's O(1) value — but `kept_to_global`
    /// *shrinks* them, and this does not account for that. Safe because every
    /// consumer sizes scratch/staging buffers with it (GPU pinned slots,
    /// covariance densification), where over-estimating costs memory, not
    /// correctness. Do not treat it as exact.
    fn max_shard_rows(&self) -> scx_format_io::Result<usize> {
        self.backed.max_shard_rows()
    }

    /// Forwarded from the wrapped reader's catalog statistics, for the same
    /// reason and with the same caveat as [`Self::max_shard_rows`]: transforms
    /// and column projection cannot raise either figure and `kept_to_global`
    /// only lowers them, so the on-disk numbers stay valid **upper bounds** —
    /// which is exactly what the contract promises. `None` when the catalog
    /// carries no stats block; a consumer must read that as "unknown", never as
    /// zero.
    fn shard_size_hint(&self) -> Option<scx_format_io::ShardSizeHint> {
        scx_format_io::ShardSource::shard_size_hint(&*self.backed)
    }

    /// The shards a kept-row filter leaves anything in.
    ///
    /// `read_shard_arc` below applies `kept_to_global` *after* decoding, and a
    /// shard none of the kept rows falls in comes back as a 0-row CSR — the
    /// decode, the transform pass and the column projection all paid for
    /// nothing. Answering here lets the `prefetch` drivers skip the read
    /// entirely, which every consumer of `as_shard_source()` gets for free:
    /// PCA (six to seven passes), HVG, `score_genes`, `pflog`, and the DE
    /// kernels — Wilcoxon, pdex and the `pts` counting pass. The DE ones ran
    /// their own `0..n_shards` loop per gene chunk until they adopted the
    /// drivers, so nothing consulted this and they decoded every shard under a
    /// row window; they are also where it pays most, because their shard walk
    /// is inner to the gene-chunk walk (5 decodes to 1 on a one-shard window,
    /// 20 to 4 at four gene chunks).
    ///
    /// Behaviourally identical to today, not merely close: a consumer that
    /// received the empty shard added nothing to its accumulators, and the
    /// ones that track a row cursor advance it by `projected.n_rows()`, which
    /// was zero. `None` when there is no row filter, so the unsubset path is
    /// untouched.
    fn visible_shard_indices(&self) -> Option<Vec<usize>> {
        let kept = self.kept_to_global.as_ref()?;
        // A plan can be empty (nothing kept in any shard), and an empty plan
        // performs no `read_shard` — which is where a watching reader checks
        // that the file has not changed underneath it. Rather than teach this
        // hook to fail, answer `None` when the file is stale: the driver then
        // visits every shard and the first read raises, exactly as it did
        // before there was a plan at all.
        if self.backed.check_fresh().is_err() {
            return None;
        }
        Some(self.backed.index().shards_with_kept_rows(kept))
    }

    fn read_shard(&self, shard_idx: usize) -> scx_format_io::Result<ScxCsr> {
        // `read_shard_arc` owns the pipeline. When nothing else holds the Arc
        // (the uncached path, or any path that derived a fresh CSR) this
        // unwraps for free; only a passthrough hit on the shared LRU copies.
        let arc = self.read_shard_arc(shard_idx)?;
        Ok(Arc::try_unwrap(arc).unwrap_or_else(|shared| (*shared).clone()))
    }

    /// Decode → transforms → column projection → deletion vector.
    ///
    /// Overridden (rather than left to the trait default, which wraps
    /// `read_shard`) so multi-pass kernels reach the reader's decoded-shard
    /// LRU through [`LazyShardSource::with_cached_reads`]. Each stage that
    /// applies produces an owned CSR and the next reads from it; stages that
    /// don't apply are skipped, so a source with no transforms and no view
    /// hands the cached `Arc` straight back with no copy at all.
    fn read_shard_arc(&self, shard_idx: usize) -> scx_format_io::Result<Arc<ScxCsr>> {
        let (s_start, s_end) = self.backed.index().shard_range(shard_idx).ok_or_else(|| {
            scx_format_io::ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: self.backed.index().n_shards(),
            }
        })?;

        let mut decoded = Some(self.decode_shard(shard_idx)?);

        // Transforms mutate in place, so they are the one stage that needs its
        // own buffer up front. `try_unwrap` is what decides whether that costs a
        // copy: on the uncached path the decode just built a refcount-1 `Arc`
        // and this takes ownership for free, while a hit on the shared LRU
        // legitimately clones — mutating the cached shard would corrupt it for
        // every other reader.
        let mut current: Option<ScxCsr> = if self.transforms.is_empty() {
            None
        } else {
            let arc = decoded.take().expect("decoded is Some until moved here");
            let mut owned = Arc::try_unwrap(arc).unwrap_or_else(|shared| (*shared).clone());
            apply_transforms_to_csr(&self.transforms, &mut owned, s_start as usize);
            Some(owned)
        };

        // Column projection: remap column indices into projected space.
        if let Some(cols) = &self.col_projection {
            let projected = {
                let src = current.as_ref().unwrap_or_else(|| {
                    decoded
                        .as_deref()
                        .expect("decoded survives when untransformed")
                });
                scx_engine::projection::project_csr(src, cols)
            };
            current = Some(projected);
        }

        // Deletion vector: keep only this shard's visible rows.
        if let Some(kept) = &self.kept_to_global {
            let filtered = {
                let src = current.as_ref().unwrap_or_else(|| {
                    decoded
                        .as_deref()
                        .expect("decoded survives when untransformed")
                });
                let lo = kept.partition_point(|&r| r < s_start);
                let hi = kept.partition_point(|&r| r < s_end);
                if hi > lo {
                    let local_rows: Vec<usize> = kept[lo..hi]
                        .iter()
                        .map(|&g| (g - s_start) as usize)
                        .collect();
                    extract_local_rows(src, &local_rows)
                } else {
                    // No kept rows in this shard — return empty.
                    let n_projected = self
                        .col_projection
                        .as_ref()
                        .map_or(self.shape_val.1, |c| c.len());
                    ScxCsr::new_unchecked((0, n_projected), vec![0], vec![], vec![])
                }
            };
            current = Some(filtered);
        }

        Ok(match current {
            Some(derived) => Arc::new(derived),
            // Untouched by every stage — hand the decoded `Arc` straight back.
            None => decoded.expect("no stage applied, so decoded was never taken"),
        })
    }

    /// Republish the wrapped reader's LRU capacity, but only when this source
    /// actually reads through it. Reporting `Some(..)` on the uncached path
    /// would tell a multi-pass kernel its working set is cached when every
    /// `read_shard_arc` re-decodes.
    fn shard_cache_capacity(&self) -> Option<usize> {
        if self.cached_reads {
            scx_format_io::ShardSource::shard_cache_capacity(self.backed.as_ref())
        } else {
            None
        }
    }

    // col_means_and_sum_sq: use the default trait impl which iterates
    // read_shard_arc() — transforms and col_projection are applied per-shard,
    // and the cached path is shared.
}

/// Apply lazy transforms in-place on a decoded CSC shard.
///
/// Every arm here mirrors `apply_transforms_to_csr`'s arithmetic **exactly**,
/// including the order and width of each cast, because the two routes must
/// agree bit for bit and not merely within a tolerance. These are element-wise
/// maps with no accumulation, so exact agreement is achievable and anything
/// less would mean one of the two is wrong.
///
/// The row-indexed arms (`NormalizeTotal`, `RowScale`) read their per-row
/// vector at `csc.indices[k]`, which is the nonzero's **global** row. That is
/// the same index CSR reaches as `global_row_offset + row`, so both routes
/// index the same vector the same way.
///
/// That is also why a row filter is applied **after** this function, never
/// before: `row_sums` / `factors` are built at global physical length, so on a
/// slab already renumbered to live rows every lookup would silently return
/// another cell's factor. The callers order the two stages, and
/// `shard_source_tests.rs` pins the order with a test rather than a comment.
///
/// A `NormalizeTotal → Log1p` pair is **fused** on the CSR side, in both
/// `transforms.rs` (slices and shard reads) and `dataset_index.rs` (fancy
/// indexing). On a row whose total is not positive those fused paths skip the
/// normalisation — scanpy's rule — but still apply `ln_1p`, which is what this
/// function does by running the two in sequence. They used to skip `ln_1p`
/// too, which made them disagree with this path on any signed row; that is
/// fixed, and `shard_source_tests.rs` pins the agreement on a cancelling row
/// and a negative-total row.
///
/// Infallible by construction: every `Transform` has a column-major form, so
/// there is no arm left to refuse. The `match` below is exhaustive with no
/// wildcard, and that is the whole guard — a new variant fails to compile here
/// until someone decides what it means column-major. **If that decision is
/// "it cannot be served column-major", refuse it in
/// [`LazyShardSource::supports_csc`] rather than writing an approximate arm
/// here**; the gate is the right place for that and this function is not.
fn apply_transforms_to_csc(transforms: &[Transform], csc: &mut ScxCsc) {
    for transform in transforms {
        match transform {
            Transform::Log1p => {
                for v in &mut csc.data {
                    *v = v.ln_1p();
                }
            }
            Transform::Scale { factor } => {
                for v in &mut csc.data {
                    *v = (*v as f64 * *factor) as f32;
                }
            }
            Transform::NormalizeTotal {
                row_sums,
                target_sum,
            } => {
                // Split the borrow: the value being written and the row index
                // being read live in two fields of the same struct.
                let (data, indices) = (&mut csc.data, &csc.indices);
                debug_assert!(
                    indices
                        .iter()
                        .all(|&r| r >= 0 && (r as usize) < row_sums.len()),
                    "CSC row index outside row_sums ({}); a shard's indices must be \
                     global physical rows",
                    row_sums.len(),
                );
                for (v, &row) in data.iter_mut().zip(indices.iter()) {
                    let sum = row_sums[row as usize];
                    // `sum > 0.0` and *not* an else-branch: CSR leaves a
                    // zero-sum row's values untouched rather than zeroing
                    // them, and a following Log1p then maps 0 -> ln(1) = 0.
                    if sum > 0.0 {
                        *v = (*v as f64 * (*target_sum / sum)) as f32;
                    }
                }
            }
            Transform::RowScale { factors } => {
                let (data, indices) = (&mut csc.data, &csc.indices);
                debug_assert!(
                    indices
                        .iter()
                        .all(|&r| r >= 0 && (r as usize) < factors.len()),
                    "CSC row index outside factors ({}); a shard's indices must be \
                     global physical rows",
                    factors.len(),
                );
                for (v, &row) in data.iter_mut().zip(indices.iter()) {
                    // f32 multiply, matching CSR. NormalizeTotal above goes
                    // through f64 and this one does not; that asymmetry is
                    // CSR's, and copying it is the point.
                    *v *= factors[row as usize] as f32;
                }
            }
        }
    }
}

impl scx_format_io::ColumnShardSource for LazyShardSource {
    fn n_csc_shards(&self) -> usize {
        match &self.backed_csc {
            Some(b) => b.n_shards(),
            None => 0,
        }
    }

    fn n_obs(&self) -> usize {
        self.shape_val.0
    }

    fn n_vars(&self) -> usize {
        match &self.col_projection {
            Some(cols) => cols.len(),
            None => self.shape_val.1,
        }
    }

    /// Decode -> transforms -> row filter -> column projection.
    ///
    /// The row filter runs after the transforms because they index their
    /// per-row vectors at the global row (see `apply_transforms_to_csc`), and
    /// before the projection because compacting first leaves `project_csc`
    /// fewer entries to copy. Both stages are skipped when they do not apply.
    fn read_csc_shard(&self, shard_idx: usize) -> scx_format_io::Result<ScxCsc> {
        let backed = self.backed_csc.as_ref().ok_or_else(|| {
            scx_format_io::ScxError::Io(std::io::Error::other(
                "CSC unavailable: file has no CSC sidecar (open with CSC enabled)",
            ))
        })?;
        // `try_unwrap` for symmetry with the CSR path, and for the honest
        // reason rather than an optimistic one: the decoded-shard LRU normally
        // holds a second strong reference, so this clones exactly as it always
        // did. It takes ownership for free only where the cache is off or the
        // entry was already evicted, and the transforms and the compaction
        // both need an owned slab regardless.
        let arc = backed.read_shard_cached(shard_idx)?;
        let mut csc = Arc::try_unwrap(arc).unwrap_or_else(|shared| (*shared).clone());
        apply_transforms_to_csc(&self.transforms, &mut csc);
        if let Some(map) = self.global_to_live() {
            scx_engine::projection::compact_csc_rows_in_place(&mut csc, map, self.shape_val.0);
        }
        if let Some(ref proj) = self.col_projection {
            // `proj` is sorted/dedup'd GLOBAL column IDs, but `csc` is a
            // shard slab whose own column space is `0..shard_n_cols`.
            // Filter `proj` to entries inside this shard's global range,
            // remap to shard-local, then project.
            let (g_lo, g_hi) =
                scx_format_io::ColumnShardSource::csc_shard_col_range(backed.as_ref(), shard_idx)
                    .ok_or_else(|| {
                    scx_format_io::ScxError::Io(std::io::Error::other(
                        "CSC unavailable: missing shard col range for projection remap",
                    ))
                })?;
            let p_lo = proj.partition_point(|&g| g < g_lo);
            let p_hi = proj.partition_point(|&g| g < g_hi);
            let local: Vec<u32> = proj[p_lo..p_hi].iter().map(|&g| g - g_lo).collect();
            csc = scx_engine::projection::project_csc(&csc, &local);
        }
        Ok(csc)
    }

    fn read_csc_columns(&self, col_range: std::ops::Range<u32>) -> scx_format_io::Result<ScxCsc> {
        let backed = self.backed_csc.as_ref().ok_or_else(|| {
            scx_format_io::ScxError::Io(std::io::Error::other(
                "CSC unavailable: file has no CSC sidecar (open with CSC enabled)",
            ))
        })?;

        // When a column projection is active, the user-facing column
        // axis is the projected one. Translate the projected range into
        // the underlying global range, fetch via the inner reader, and
        // re-project the result to the projected axis.
        let mut csc = match &self.col_projection {
            Some(proj) => {
                let lo = col_range.start as usize;
                let hi = (col_range.end as usize).min(proj.len());
                if lo >= hi {
                    // Empty range — return an empty CSC sized to the
                    // projected n_vars window.
                    return Ok(ScxCsc::new_unchecked(
                        (self.shape_val.0, 0),
                        vec![0],
                        Vec::new(),
                        Vec::new(),
                    ));
                }
                let global_subset = &proj[lo..hi];
                backed.read_csc_columns_subset(global_subset)?
            }
            None => backed.read_csc_columns(col_range)?,
        };

        apply_transforms_to_csc(&self.transforms, &mut csc);
        // Same order as `read_csc_shard`, for the same reason. There is no
        // projection stage here — the column window was translated before the
        // read — so the compaction is the last thing that touches the slab.
        if let Some(map) = self.global_to_live() {
            scx_engine::projection::compact_csc_rows_in_place(&mut csc, map, self.shape_val.0);
        }
        Ok(csc)
    }

    fn csc_shard_col_range(&self, shard_idx: usize) -> Option<(u32, u32)> {
        let backed = self.backed_csc.as_ref()?;
        let (g_lo, g_hi) =
            scx_format_io::ColumnShardSource::csc_shard_col_range(backed.as_ref(), shard_idx)?;
        match &self.col_projection {
            Some(proj) => {
                // Map the inner shard's global range [g_lo, g_hi) onto
                // the projected axis. Consumers iterating shards see
                // ranges in the same axis as `n_vars()` (projected).
                let p_lo = proj.partition_point(|&g| g < g_lo) as u32;
                let p_hi = proj.partition_point(|&g| g < g_hi) as u32;
                Some((p_lo, p_hi))
            }
            None => Some((g_lo, g_hi)),
        }
    }

    /// Forwarded from the sidecar, and load-bearing on GPU.
    ///
    /// `RawGpuCscShardSource::new` pre-sizes its pinned and device buffers
    /// from this and derives its staging prefetch depth from it; a source that
    /// answers `None` gets a grow-on-demand capacity of one and is not bounded
    /// by `SCX_GPU_STAGING_MEMORY_BUDGET` at all. So the CSC-direct GPU route
    /// reaching this type instead of `BackedCscReader` would otherwise be a
    /// staging regression, not just a different code path.
    ///
    /// Describes the **view**, not the file, and that is load-bearing in the
    /// other direction: those allocations are eager, so forwarding the
    /// sidecar's physical hint would have a tiny window reserve the whole
    /// file's widest shard — ~24 B of pinned host memory and 8 B of device
    /// memory per physical nonzero, newly reachable because these handles used
    /// to take the CSR route instead.
    ///
    /// Both terms are sound upper bounds rather than estimates. `max_rows` is
    /// the column count, which under a projection is the widest *projected*
    /// shard. `max_nnz` is bounded by `n_live * that`, since a compacted slab
    /// holds at most one nonzero per (visible row, column) pair — and is
    /// min'd with the inner hint so it can only ever tighten it. With no
    /// window active the product dwarfs the physical bound and this reduces to
    /// forwarding verbatim.
    ///
    /// `csc_shards_for_col_range` is deliberately **not** forwarded:
    /// `BackedCscReader` overrides it with a binary search over on-disk column
    /// ranges, which is the wrong axis once a projection is active. The trait
    /// default is expressed over `csc_shard_col_range` above, which already
    /// reports projected ranges.
    fn csc_shard_size_hint(&self) -> Option<scx_format_io::ShardSizeHint> {
        let inner = scx_format_io::ColumnShardSource::csc_shard_size_hint(
            self.backed_csc.as_ref()?.as_ref(),
        )?;
        let max_cols = if self.col_projection.is_some() {
            (0..scx_format_io::ColumnShardSource::n_csc_shards(self))
                .filter_map(|i| {
                    scx_format_io::ColumnShardSource::csc_shard_col_range(self, i)
                        .map(|(lo, hi)| hi.saturating_sub(lo) as usize)
                })
                .max()
                .unwrap_or(0)
        } else {
            inner.max_rows
        };
        Some(scx_format_io::ShardSizeHint {
            max_rows: max_cols,
            max_nnz: self.shape_val.0.saturating_mul(max_cols).min(inner.max_nnz),
        })
    }
}

/// Extract specific rows from a CSR by local (within-shard) row indices.
fn extract_local_rows(csr: &ScxCsr, local_rows: &[usize]) -> ScxCsr {
    let n_cols = csr.n_cols();
    let mut indptr = Vec::with_capacity(local_rows.len() + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();

    indptr.push(0i64);
    for &local_row in local_rows {
        let s = csr.indptr[local_row] as usize;
        let e = csr.indptr[local_row + 1] as usize;
        indices.extend_from_slice(&csr.indices[s..e]);
        data.extend_from_slice(&csr.data[s..e]);
        indptr.push(indices.len() as i64);
    }

    ScxCsr::new_unchecked((local_rows.len(), n_cols), indptr, indices, data)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Concatenate multiple ScxCsr slices into one.
pub(crate) fn concatenate_csr_vec(slices: &[ScxCsr], n_cols: usize) -> ScxCsr {
    if slices.is_empty() {
        return ScxCsr::new_unchecked((0, n_cols), vec![0], vec![], vec![]);
    }
    if slices.len() == 1 {
        return slices[0].clone();
    }

    let total_rows: usize = slices.iter().map(|s| s.n_rows()).sum();
    let total_nnz: usize = slices.iter().map(|s| s.nnz()).sum();

    let mut indptr = Vec::with_capacity(total_rows + 1);
    let mut indices = Vec::with_capacity(total_nnz);
    let mut data = Vec::with_capacity(total_nnz);

    indptr.push(0i64);
    let mut offset = 0i64;

    for csr in slices {
        for row in 0..csr.n_rows() {
            let s = csr.indptr[row] as usize;
            let e = csr.indptr[row + 1] as usize;
            indices.extend_from_slice(&csr.indices[s..e]);
            data.extend_from_slice(&csr.data[s..e]);
            offset += (e - s) as i64;
            indptr.push(offset);
        }
    }

    ScxCsr::new_unchecked((total_rows, n_cols), indptr, indices, data)
}

/// Extract specific rows from a CSR by global row indices.
pub(crate) fn extract_rows(csr: &ScxCsr, indices: &[u64]) -> ScxCsr {
    let n_rows = indices.len();
    let n_cols = csr.shape.1;
    let mut indptr = Vec::with_capacity(n_rows + 1);
    let mut new_indices = Vec::new();
    let mut new_data = Vec::new();

    indptr.push(0i64);
    for &g_row in indices {
        let r = g_row as usize;
        let s = csr.indptr[r] as usize;
        let e = csr.indptr[r + 1] as usize;
        new_indices.extend_from_slice(&csr.indices[s..e]);
        new_data.extend_from_slice(&csr.data[s..e]);
        indptr.push(new_indices.len() as i64);
    }

    ScxCsr::new_unchecked((n_rows, n_cols), indptr, new_indices, new_data)
}
