//! Deterministic column-block planning for the CPU PCA reductions.
//!
//! Both PCA reductions — the covariance lower triangle and the transpose SpMM —
//! partition their **output** across workers rather than their input rows. Each
//! worker owns a contiguous range of output columns / variable rows, scans every
//! input row, and takes only the nonzeros that land in its range. Nothing is
//! merged afterwards, so the partition never reaches the result: entry `(i, j)`
//! sums rows `0..n_obs` in ascending order for *any* block count, on *any*
//! thread count.
//!
//! That is what makes the split a pure performance knob. Balancing it well
//! matters for speed and not at all for correctness, so [`plan_blocks`] is free
//! to use an exact integer work model computed per shard.
//!
//! # Precondition
//!
//! [`sorted_subrange`] requires each CSR row's column indices to be strictly
//! increasing — the canonical form every SCX writer emits (`is_canonical_csr`
//! gates ingest) and the same precondition `scx_engine::project_csr` already
//! documents for its merge scan. [`rows_strictly_increasing`] is the runtime
//! check callers use to fall back when they cannot assume it.

use std::ops::Range;

use scx_sparse::ScxCsr;

/// Split `0..weights.len()` into exactly `n_blocks` contiguous ranges of
/// approximately equal total weight.
///
/// Ranges are contiguous, non-overlapping, possibly empty, and their union is
/// exactly `0..weights.len()`. When every weight is zero (an empty shard) the
/// split falls back to equal column counts, which is the best available guess
/// and keeps the ranges from collapsing onto the last block.
///
/// Weight is an exact operation count, so the plan is a pure function of the
/// data. It is *not* load-bearing for the result — see the module docs.
pub(crate) fn plan_blocks(weights: &[u64], n_blocks: usize) -> Vec<Range<usize>> {
    let n = weights.len();
    let n_blocks = n_blocks.max(1);
    if n_blocks == 1 {
        return std::iter::once(0..n).collect();
    }

    let total: u128 = weights.iter().map(|&w| u128::from(w)).sum();
    if total == 0 {
        // Nothing to balance: even column counts.
        return (0..n_blocks)
            .map(|b| {
                let lo = n * b / n_blocks;
                let hi = n * (b + 1) / n_blocks;
                lo..hi
            })
            .collect();
    }

    let mut out = Vec::with_capacity(n_blocks);
    let mut start = 0usize;
    let mut cum: u128 = 0;
    for b in 0..n_blocks - 1 {
        let target = total * (b as u128 + 1) / n_blocks as u128;
        let mut end = start;
        while end < n && cum < target {
            cum += u128::from(weights[end]);
            end += 1;
        }
        out.push(start..end);
        start = end;
    }
    out.push(start..n);
    out
}

/// The sub-range `[lo, hi)` of a **strictly increasing** row's index slice whose
/// column indices lie in `[a, b)`.
///
/// Returns positions relative to the slice, not to the CSR's global `indptr`.
#[inline]
pub(crate) fn sorted_subrange(row_indices: &[i32], a: usize, b: usize) -> (usize, usize) {
    let lo = row_indices.partition_point(|&c| (c as usize) < a);
    let hi = row_indices.partition_point(|&c| (c as usize) < b);
    (lo, hi)
}

/// Whether every row's column indices are strictly increasing (sorted, no
/// duplicates) — the canonical form. `O(nnz)`, allocates nothing.
pub(crate) fn rows_strictly_increasing(csr: &ScxCsr) -> bool {
    (0..csr.n_rows()).all(|r| {
        let (s, e) = (csr.indptr[r] as usize, csr.indptr[r + 1] as usize);
        csr.indices[s..e].windows(2).all(|w| w[0] < w[1])
    })
}

/// How many column blocks to plan.
///
/// Bounded by the ambient rayon pool and by the `SCX_ACCEL_NUM_THREADS` policy
/// ceiling. Because the block count cannot change the result, honouring those
/// knobs costs nothing numerically — they bound speed and memory only.
pub(crate) fn block_count(n_cols: usize) -> usize {
    rayon::current_num_threads()
        .min(crate::mem_budget::accel_num_threads().unwrap_or(usize::MAX))
        .max(1)
        .min(n_cols.max(1))
}

/// Carve `buf` into one disjoint mutable slice per block, `stride` elements per
/// column. `buf.len()` must be `stride * <total columns>`.
pub(crate) fn split_by_blocks<'a, T>(
    mut buf: &'a mut [T],
    blocks: &[Range<usize>],
    stride: usize,
) -> Vec<&'a mut [T]> {
    let mut out = Vec::with_capacity(blocks.len());
    let mut consumed = 0usize;
    for block in blocks {
        debug_assert_eq!(block.start, consumed, "blocks must be contiguous");
        let take = block.len() * stride;
        let (head, tail) = buf.split_at_mut(take);
        out.push(head);
        buf = tail;
        consumed = block.end;
    }
    out
}

#[cfg(test)]
#[path = "colblocks_tests.rs"]
mod tests;
