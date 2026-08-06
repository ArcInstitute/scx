//! Group-label partitioning shared by every Wilcoxon / pdex kernel.
//!
//! Callers encode a per-cell group label as an index into `group_names`. Any
//! label `>= n_groups` is the **unlabelled** sentinel: `pyscx` maps a NaN /
//! empty / off-category `obs[groupby]` value to `unique_groups.len()` so that a
//! cell nobody annotated does not silently join group 0.
//!
//! # Why unlabelled cells leave the comparison entirely
//!
//! scanpy's `rank_genes_groups` subsets the matrix to
//! `adata.obs[groupby].isin(groups_order)` before it ranks anything, so an
//! unlabelled cell is in no group, is **not** part of "rest", and is **not** in
//! the rank pool. Reproducing that means one number — `labelled.len()` — has to
//! be the rest denominator *and* the rank-pool size everywhere. Deriving them
//! separately is what made the 1-vs-rest logFC wrong: the rest *numerator*
//! excluded unlabelled cells while the rest *denominator* counted them, so
//! every logFC in every group was inflated by
//! `log2(n_obs − n1) − log2(n_labelled − n1)`.
//!
//! When no cell is unlabelled — the overwhelmingly common case — `labelled` is
//! `0..n_obs`, `pool_pos == group_indices`, and every kernel is bit-identical
//! to a version that never knew about this module.

/// The three views of a group-label array that the DE kernels need.
///
/// Built once per DE call by [`partition_by_group`]; the per-gene loops read
/// it without re-scanning `groups`.
#[derive(Debug, Clone)]
pub struct GroupPartition {
    /// Ascending global cell indices whose label is a real group. The
    /// comparison pool for 1-vs-rest — its length is both the rest denominator
    /// base and the rank-pool size.
    pub labelled: Vec<usize>,
    /// `group_indices[g]` — ascending **global** cell indices in group `g`.
    /// Used by the pairwise (`reference = Some(_)`) arm, which gathers
    /// `group ∪ ref` straight out of the caller's matrix.
    pub group_indices: Vec<Vec<usize>>,
    /// `pool_pos[g]` — ascending positions of group `g`'s cells **within
    /// `labelled`**. Used by the 1-vs-rest arm, whose value/rank buffers are
    /// compacted to the labelled cells.
    pub pool_pos: Vec<Vec<usize>>,
}

impl GroupPartition {
    /// Number of cells carrying a real group label. The rank-pool size and the
    /// base of the rest denominator (`n_labelled - n1`) in 1-vs-rest.
    pub fn n_labelled(&self) -> usize {
        self.labelled.len()
    }

    /// Cells whose label was out of range. Non-zero means the caller handed us
    /// unannotated cells; surfacing the count is the caller's job.
    pub fn n_unlabelled(&self, n_obs: usize) -> usize {
        n_obs - self.labelled.len()
    }

    /// `true` when every cell carries a real label, i.e. the pool is the whole
    /// matrix and `pool_pos[g] == group_indices[g]`.
    pub fn is_total(&self, n_obs: usize) -> bool {
        self.labelled.len() == n_obs
    }
}

/// Partition `groups` (one label per cell) into the labelled pool, the
/// per-group global cell lists, and the per-group pool positions.
///
/// A label `>= n_groups` is unlabelled and appears in none of the three.
/// The filling pass walks `groups` in ascending order, so both index lists come
/// out ascending and the f64 accumulation order of any sum driven by them is
/// deterministic.
///
/// A counting pass runs first so every vector is allocated at its exact size.
/// At atlas scale the growth-doubling alternative transiently holds roughly
/// twice the final footprint across `2·n_groups + 1` vectors, and the extra
/// read over `groups` is far cheaper than that.
pub fn partition_by_group(groups: &[usize], n_groups: usize) -> GroupPartition {
    let mut group_sizes = vec![0usize; n_groups];
    let mut n_labelled = 0usize;
    for &g in groups {
        if g < n_groups {
            group_sizes[g] += 1;
            n_labelled += 1;
        }
    }

    let mut labelled = Vec::with_capacity(n_labelled);
    let mut group_indices: Vec<Vec<usize>> =
        group_sizes.iter().map(|&n| Vec::with_capacity(n)).collect();
    let mut pool_pos: Vec<Vec<usize>> =
        group_sizes.iter().map(|&n| Vec::with_capacity(n)).collect();

    for (i, &g) in groups.iter().enumerate() {
        if g < n_groups {
            group_indices[g].push(i);
            pool_pos[g].push(labelled.len());
            labelled.push(i);
        }
    }
    GroupPartition {
        labelled,
        group_indices,
        pool_pos,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_partition_is_identity() {
        let groups = vec![0, 1, 0, 1, 2];
        let p = partition_by_group(&groups, 3);
        assert_eq!(p.labelled, vec![0, 1, 2, 3, 4]);
        assert_eq!(p.group_indices, p.pool_pos);
        assert!(p.is_total(5));
        assert_eq!(p.n_unlabelled(5), 0);
    }

    #[test]
    fn unlabelled_cells_are_dropped_and_positions_compact() {
        // n_groups = 2, so labels 2 and 7 are both the unlabelled sentinel.
        let groups = vec![0, 2, 1, 7, 0, 1];
        let p = partition_by_group(&groups, 2);
        assert_eq!(p.labelled, vec![0, 2, 4, 5]);
        assert_eq!(p.n_labelled(), 4);
        assert_eq!(p.n_unlabelled(6), 2);
        assert!(!p.is_total(6));
        // Global cell ids.
        assert_eq!(p.group_indices, vec![vec![0, 4], vec![2, 5]]);
        // Positions inside `labelled` = [0, 2, 4, 5].
        assert_eq!(p.pool_pos, vec![vec![0, 2], vec![1, 3]]);
    }

    #[test]
    fn empty_groups_are_kept_as_empty_slots() {
        let groups = vec![0, 0, 0];
        let p = partition_by_group(&groups, 3);
        assert_eq!(p.group_indices.len(), 3);
        assert!(p.group_indices[1].is_empty());
        assert!(p.group_indices[2].is_empty());
    }
}
