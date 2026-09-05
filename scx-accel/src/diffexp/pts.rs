//! Per-(group, gene) nonzero counts — the exact integer behind scanpy's `pts`
//! ("fraction of cells in the group expressing the gene").
//!
//! One implementation, fed from whatever the DE ran on — a streamed
//! [`ShardSource`] (backed or lazy), an owned CSR, or an owned dense buffer —
//! so every dispatch route (CSC-direct and the GPU drivers included) reports
//! the same number from the same code. It is a separate pass by design: the
//! Wilcoxon and pdex kernels never learn about it, their signatures stay
//! shared with `rscx` and the criterion benches untouched, and the count is
//! O(nnz) with no ranking, so it costs one more read of the matrix and
//! nothing else.
//!
//! # What counts as "expressing"
//!
//! A value `!= 0.0` on the `f32` the DE saw. That is scanpy's rule
//! (`getnnz(axis=0)` after `eliminate_zeros()` for sparse input,
//! `np.count_nonzero` for dense): an explicit zero stored in a scipy CSR is
//! not counted, a negative value is.
//!
//! # Which cells are "rest"
//!
//! scanpy's `pts_rest[g]` is the nonzero fraction over `X[~mask_g]` — **every
//! other row of the matrix**, cells with no `groupby` label included — and this
//! module reproduces that table exactly. A cell whose label is the unlabelled
//! sentinel (`>= n_groups`, see [`super::groups`]) is therefore in no group's
//! `pts` but in every group's `pts_rest`. (The rank-sum kernels leave such
//! cells out of their pool; that is a pre-existing difference from scanpy 1.12
//! tracked separately, and `pts` follows scanpy's tables, not that pool.)

use scx_format_io::ShardSource;
use scx_sparse::ScxCsr;

use crate::{AccelError, Result};

/// Nonzero counts per group and per gene, plus the whole-matrix total the
/// 1-vs-rest fraction is derived from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupNonzeroCounts {
    /// `counts[g][j]` — cells with label `g` whose value at gene `j` is nonzero.
    pub counts: Vec<Vec<u64>>,
    /// `total[j]` — nonzero cells at gene `j` over **every** row, unlabelled
    /// ones included. Not derivable from `counts` (which see only labelled
    /// rows); it is what makes `pts_rest` scanpy's `X[~mask_g]` fraction.
    pub total: Vec<u64>,
    /// `group_sizes[g]` — cells carrying label `g` (the `pts` denominator).
    pub group_sizes: Vec<usize>,
    /// Rows in the matrix; with `group_sizes[g]` it gives the `pts_rest`
    /// denominator `n_obs - n_g`.
    pub n_obs: usize,
}

/// The two tables scanpy stores: `pts[g][j]` for every group, and
/// `pts_rest[g][j]` only in 1-vs-rest mode (`reference = None`).
#[derive(Debug, Clone, PartialEq)]
pub struct PtsFractions {
    /// `[n_groups][n_vars]`; `NaN` for a group with no cells (0 / 0, as scanpy).
    pub pts: Vec<Vec<f64>>,
    /// `[n_groups][n_vars]` over every other cell of the matrix, `None` when a
    /// reference group was named (scanpy emits `pts_rest` only for
    /// `reference="rest"`).
    pub pts_rest: Option<Vec<Vec<f64>>>,
}

impl GroupNonzeroCounts {
    /// Zeroed accumulators sized from the label vector: `group_sizes` and
    /// `n_obs` are fixed here (they depend only on `groups`), the per-gene
    /// counts fill in through [`add_csr`](Self::add_csr) /
    /// [`add_dense`](Self::add_dense).
    pub fn new(groups: &[usize], n_groups: usize, n_vars: usize) -> Self {
        let mut group_sizes = vec![0usize; n_groups];
        for &g in groups {
            if g < n_groups {
                group_sizes[g] += 1;
            }
        }
        Self {
            counts: vec![vec![0u64; n_vars]; n_groups],
            total: vec![0u64; n_vars],
            group_sizes,
            n_obs: groups.len(),
        }
    }

    /// Number of groups (the label universe, reference included).
    pub fn n_groups(&self) -> usize {
        self.group_sizes.len()
    }

    /// Number of genes.
    pub fn n_vars(&self) -> usize {
        self.total.len()
    }

    /// Fold in a CSR block whose first row is global row `row_offset`.
    ///
    /// `groups` is the full per-cell label vector (one entry per global row).
    /// Errors, rather than panicking, on a block that runs past `groups`, a
    /// column count other than `n_vars`, or a column index out of range — a
    /// malformed shard is data, not a bug in the caller.
    pub fn add_csr(&mut self, csr: &ScxCsr, row_offset: usize, groups: &[usize]) -> Result<()> {
        let n_vars = self.n_vars();
        let n_groups = self.n_groups();
        let n_rows = csr.n_rows();
        if csr.shape.1 != n_vars {
            return Err(AccelError::ShapeError(format!(
                "group nonzero counts: CSR block has {} columns but the count table has n_vars = {}",
                csr.shape.1, n_vars
            )));
        }
        if row_offset + n_rows > groups.len() {
            return Err(AccelError::ShapeError(format!(
                "group nonzero counts: CSR block rows {}..{} run past the {} group labels",
                row_offset,
                row_offset + n_rows,
                groups.len()
            )));
        }
        for row in 0..n_rows {
            let g = groups[row_offset + row];
            let start = csr.indptr[row] as usize;
            let end = csr.indptr[row + 1] as usize;
            for j in start..end {
                if csr.data[j] == 0.0 {
                    continue;
                }
                let col = csr.indices[j] as usize;
                if col >= n_vars {
                    return Err(AccelError::InvalidInput(format!(
                        "group nonzero counts: column index {col} out of range (n_vars = {n_vars}) in CSR row {}",
                        row_offset + row
                    )));
                }
                // Every row is in the total (scanpy's `~mask_g` rest); only a
                // labelled row is in a group.
                self.total[col] += 1;
                if g < n_groups {
                    self.counts[g][col] += 1;
                }
            }
        }
        Ok(())
    }

    /// Fold in a row-major dense block covering global rows `0..n_obs`.
    pub fn add_dense(
        &mut self,
        data: &[f32],
        n_obs: usize,
        n_vars: usize,
        groups: &[usize],
    ) -> Result<()> {
        if n_vars != self.n_vars() {
            return Err(AccelError::ShapeError(format!(
                "group nonzero counts: dense block has {} columns but the count table has n_vars = {}",
                n_vars,
                self.n_vars()
            )));
        }
        if data.len() != n_obs * n_vars {
            return Err(AccelError::InvalidInput(format!(
                "group nonzero counts: data length {} != n_obs {} × n_vars {}",
                data.len(),
                n_obs,
                n_vars
            )));
        }
        if groups.len() != n_obs {
            return Err(AccelError::InvalidInput(format!(
                "group nonzero counts: groups length {} != n_obs {}",
                groups.len(),
                n_obs
            )));
        }
        let n_groups = self.n_groups();
        for (row, &g) in groups.iter().enumerate() {
            let values = &data[row * n_vars..(row + 1) * n_vars];
            for (col, &v) in values.iter().enumerate() {
                if v != 0.0 {
                    self.total[col] += 1;
                    if g < n_groups {
                        self.counts[g][col] += 1;
                    }
                }
            }
        }
        Ok(())
    }

    /// scanpy's two tables from the counts. Plain IEEE division of two exact
    /// integers, so the result is bit-identical to
    /// `getnnz(axis=0) / x_mask.shape[0]` on the same cells: `pts[g]` over the
    /// group's cells, `pts_rest[g]` over `n_obs - n_g` — every other row of the
    /// matrix, as scanpy's `X[~mask_g]`.
    pub fn fractions(&self, reference: Option<usize>) -> PtsFractions {
        let pts: Vec<Vec<f64>> = self
            .counts
            .iter()
            .zip(&self.group_sizes)
            .map(|(row, &n_g)| {
                let denom = n_g as f64;
                row.iter().map(|&c| c as f64 / denom).collect()
            })
            .collect();
        let pts_rest = if reference.is_none() {
            Some(
                self.counts
                    .iter()
                    .zip(&self.group_sizes)
                    .map(|(row, &n_g)| {
                        let denom = (self.n_obs - n_g) as f64;
                        row.iter()
                            .zip(&self.total)
                            .map(|(&c, &total)| (total - c) as f64 / denom)
                            .collect()
                    })
                    .collect(),
            )
        } else {
            None
        };
        PtsFractions { pts, pts_rest }
    }
}

/// Counts over a streamed shard source — the backed or lazy `X` the DE ran on.
///
/// Walks every shard once through [`ShardSource::read_shard_arc`], so a
/// caching source that the DE pass just warmed serves the second read from
/// its LRU. `groups.len()` must be the source's `n_obs` (one label per
/// *visible* row — a subset handle streams its view).
pub fn group_nonzero_counts_streaming<S: ShardSource + ?Sized>(
    source: &S,
    groups: &[usize],
    n_groups: usize,
) -> Result<GroupNonzeroCounts> {
    let n_obs = source.n_obs();
    if groups.len() != n_obs {
        return Err(AccelError::InvalidInput(format!(
            "group nonzero counts: groups length {} != n_obs {}",
            groups.len(),
            n_obs
        )));
    }
    let mut acc = GroupNonzeroCounts::new(groups, n_groups, source.n_vars());
    let mut row_offset = 0usize;
    for shard_idx in 0..source.n_shards() {
        let shard = source.read_shard_arc(shard_idx).map_err(AccelError::Scx)?;
        acc.add_csr(&shard, row_offset, groups)?;
        row_offset += shard.n_rows();
    }
    if row_offset != n_obs {
        return Err(AccelError::ShapeError(format!(
            "group nonzero counts: shards cover {row_offset} rows but the source reports n_obs = {n_obs}"
        )));
    }
    Ok(acc)
}

/// Counts over one in-memory CSR (`groups.len()` must equal its row count).
pub fn group_nonzero_counts_csr(
    csr: &ScxCsr,
    groups: &[usize],
    n_groups: usize,
) -> Result<GroupNonzeroCounts> {
    if groups.len() != csr.n_rows() {
        return Err(AccelError::InvalidInput(format!(
            "group nonzero counts: groups length {} != n_obs {}",
            groups.len(),
            csr.n_rows()
        )));
    }
    let mut acc = GroupNonzeroCounts::new(groups, n_groups, csr.shape.1);
    acc.add_csr(csr, 0, groups)?;
    Ok(acc)
}

/// Counts over a row-major dense `n_obs × n_vars` buffer.
pub fn group_nonzero_counts_dense(
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    groups: &[usize],
    n_groups: usize,
) -> Result<GroupNonzeroCounts> {
    let mut acc = GroupNonzeroCounts::new(groups, n_groups, n_vars);
    acc.add_dense(data, n_obs, n_vars, groups)?;
    Ok(acc)
}

#[cfg(test)]
#[path = "pts_tests.rs"]
mod tests;
