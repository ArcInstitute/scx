//! Group-label partitioning shared by every Wilcoxon / pdex kernel.
//!
//! Callers encode a per-cell group label as an index into `group_names`. Any
//! label `>= n_groups` is the **unlabelled** sentinel: `pyscx` maps a NaN /
//! empty / off-category `obs[groupby]` value to `unique_groups.len()` so that a
//! cell nobody annotated does not silently join group 0.
//!
//! # Where an unlabelled cell does and does not appear
//!
//! It is in **no group of its own** — it contributes to no group's sum, count
//! or `pts` column. It **is** in the 1-vs-rest comparison pool: it ranks
//! alongside everyone else and it counts in every group's "rest", numerator and
//! denominator alike. That is scanpy 1.12's rule, which ranks the whole matrix
//! and leaves NaN-labelled cells in `X[~mask_g]`, and since 0.17 it is pyscx's
//! too (X9). `pts` / `pts_rest` (`super::pts`) always followed it.
//!
//! One number — `n_obs` — is therefore the rank-pool size *and* the base of the
//! rest denominator everywhere. Deriving the two separately is what made the
//! 1-vs-rest logFC wrong once before: the rest *numerator* excluded unlabelled
//! cells while the rest *denominator* counted them, so every logFC in every
//! group was off by `log2(n_obs − n1) − log2(n_labelled − n1)`.
//!
//! The pairwise arm (`reference = Some(_)`) is a different comparison and is
//! unaffected: it gathers `group ∪ ref` out of [`GroupPartition::group_indices`]
//! and never had a pool.

/// Per-group cell lists, built once per DE call by [`partition_by_group`] so
/// the per-gene loops do not re-scan `groups`.
///
/// There is no separate "pool" view: the 1-vs-rest pool is every row, so a
/// group's positions *within* the pool are its global cell indices.
#[derive(Debug, Clone)]
pub struct GroupPartition {
    /// `group_indices[g]` — ascending **global** cell indices in group `g`.
    /// Unlabelled cells appear in none of them.
    pub group_indices: Vec<Vec<usize>>,
}

impl GroupPartition {
    /// Cells carrying a real group label. Private: outside this module the
    /// interesting figure is always the complement, and exporting both invites
    /// a caller to use `n_labelled` as a pool size, which it no longer is.
    fn n_labelled(&self) -> usize {
        self.group_indices.iter().map(Vec::len).sum()
    }

    /// Cells whose label was out of range. Non-zero means the caller handed us
    /// unannotated cells; surfacing the count is the caller's job.
    pub fn n_unlabelled(&self, n_obs: usize) -> usize {
        n_obs - self.n_labelled()
    }
}

/// Partition `groups` (one label per cell) into the per-group global cell lists.
///
/// A label `>= n_groups` is unlabelled and appears in none of them. The filling
/// pass walks `groups` in ascending order, so each list comes out ascending and
/// the f64 accumulation order of any sum driven by them is deterministic.
///
/// A counting pass runs first so every vector is allocated at its exact size.
/// At atlas scale the growth-doubling alternative transiently holds roughly
/// twice the final footprint across `n_groups` vectors, and the extra read over
/// `groups` is far cheaper than that.
pub fn partition_by_group(groups: &[usize], n_groups: usize) -> GroupPartition {
    let mut group_sizes = vec![0usize; n_groups];
    for &g in groups {
        if g < n_groups {
            group_sizes[g] += 1;
        }
    }

    let mut group_indices: Vec<Vec<usize>> =
        group_sizes.iter().map(|&n| Vec::with_capacity(n)).collect();

    for (i, &g) in groups.iter().enumerate() {
        if g < n_groups {
            group_indices[g].push(i);
        }
    }
    GroupPartition { group_indices }
}

/// Bucket index for a cell's label, folding every unlabelled spelling onto one
/// slot at `n_groups`.
///
/// Buffers indexed by this must therefore be `n_groups + 1` long. The sentinel
/// slot is written but never read back as a group: what it exists for is to
/// keep an unlabelled cell's value inside the pooled `total`, which is what
/// makes `rest = total − group` describe every other cell.
///
/// `partition_by_group`'s contract admits **any** label `>= n_groups`, not just
/// `n_groups` itself, so this clamps rather than trusting the caller.
#[inline]
pub fn pool_bucket(label: usize, n_groups: usize) -> usize {
    label.min(n_groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_partition_is_identity() {
        let groups = vec![0, 1, 0, 1, 2];
        let p = partition_by_group(&groups, 3);
        assert_eq!(p.group_indices, vec![vec![0, 2], vec![1, 3], vec![4]]);
        assert_eq!(p.n_labelled(), 5);
        assert_eq!(p.n_unlabelled(5), 0);
    }

    #[test]
    fn unlabelled_cells_are_in_no_group() {
        // n_groups = 2, so labels 2 and 7 are both the unlabelled sentinel.
        let groups = vec![0, 2, 1, 7, 0, 1];
        let p = partition_by_group(&groups, 2);
        assert_eq!(p.n_labelled(), 4);
        assert_eq!(p.n_unlabelled(6), 2);
        // Global cell ids; cells 1 and 3 appear nowhere.
        assert_eq!(p.group_indices, vec![vec![0, 4], vec![2, 5]]);
    }

    #[test]
    fn empty_groups_are_kept_as_empty_slots() {
        let groups = vec![0, 0, 0];
        let p = partition_by_group(&groups, 3);
        assert_eq!(p.group_indices.len(), 3);
        assert!(p.group_indices[1].is_empty());
        assert!(p.group_indices[2].is_empty());
    }

    #[test]
    fn pool_bucket_folds_every_sentinel_spelling_onto_one_slot() {
        assert_eq!(pool_bucket(0, 3), 0);
        assert_eq!(pool_bucket(2, 3), 2);
        // Both the canonical sentinel and a larger stray land in the same slot,
        // which is the only reason an `n_groups + 1` buffer is enough.
        assert_eq!(pool_bucket(3, 3), 3);
        assert_eq!(pool_bucket(99, 3), 3);
    }
}
