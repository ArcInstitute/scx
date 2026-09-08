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
//! `pts` but in every group's `pts_rest`. Since 0.17 (X9) the rank-sum kernels
//! use that same pool, so `pts_rest` and `pvals` in one result describe one
//! reference population; before then they did not, and this module was the half
//! that already matched scanpy.

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
        let Self { counts, total, .. } = self;
        for row in 0..n_rows {
            let g = groups[row_offset + row];
            // Every row is in the total (scanpy's `~mask_g` rest); only a
            // labelled row is in a group — decided once per row, not per value.
            let mut group_row = (g < n_groups).then(|| &mut counts[g]);
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
                total[col] += 1;
                if let Some(row) = group_row.as_deref_mut() {
                    row[col] += 1;
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
        let Self { counts, total, .. } = self;
        for (row, &g) in groups.iter().enumerate() {
            let mut group_row = (g < n_groups).then(|| &mut counts[g]);
            let values = &data[row * n_vars..(row + 1) * n_vars];
            for (col, &v) in values.iter().enumerate() {
                if v != 0.0 {
                    total[col] += 1;
                    if let Some(row) = group_row.as_deref_mut() {
                        row[col] += 1;
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
/// Walks the source's shards once through the shared decode-prefetch driver
/// ([`crate::prefetch::for_each_shard_ordered`]), which means a row projection
/// skips the shards it empties and the decode overlaps the counting. This is a
/// second decode of the matrix, not a cache hit: the default four-shard LRU
/// holds the *last* shards the DE pass touched, so a fresh sequential scan
/// starting at shard 0 evicts them before it gets there.
///
/// The cached driver, matching the `read_shard_arc` this used to call. On a
/// `LazyShardSource` — the only type pyscx ever passes — the choice is moot
/// either way: its `read_shard` delegates to `read_shard_arc`, and the source's
/// own `cached_reads` flag decides whether the reader's LRU is involved.
///
/// `groups.len()` must be the source's `n_obs` (one label per *visible* row — a
/// subset handle streams its view). `row_offset` is a running count of visible
/// rows, which stays correct under skipping because a skipped shard would have
/// contributed none; the trailing coverage check is what makes that falsifiable.
pub fn group_nonzero_counts_streaming<S: ShardSource + Sync + ?Sized>(
    source: &S,
    groups: &[usize],
    n_groups: usize,
) -> Result<GroupNonzeroCounts> {
    group_nonzero_counts_streaming_with_depth(source, groups, n_groups, None)
}

/// [`group_nonzero_counts_streaming`] with the decode-prefetch depth pinned;
/// see `wilcoxon_rank_sum_streaming_with_depth` for why the knob exists.
pub(crate) fn group_nonzero_counts_streaming_with_depth<S: ShardSource + Sync + ?Sized>(
    source: &S,
    groups: &[usize],
    n_groups: usize,
    depth: Option<usize>,
) -> Result<GroupNonzeroCounts> {
    let n_obs = source.n_obs();
    if groups.len() != n_obs {
        return Err(AccelError::InvalidInput(format!(
            "group nonzero counts: groups length {} != n_obs {}",
            groups.len(),
            n_obs
        )));
    }
    // This pass is not the dense DE workspace, but it is not free either: the
    // accumulator is `(n_groups + 1) x n_vars` u64s, which on a 5 000-target
    // screen across 30 000 genes is ~1.2 GB. Charge it, so the prefetch cannot
    // be granted depth on top of memory already committed.
    let n_vars = source.n_vars();
    let acc_bytes = (n_groups as u64)
        .saturating_add(1)
        .saturating_mul(n_vars as u64)
        .saturating_mul(8);
    let depth = depth.map_or_else(
        || crate::mem_budget::de_prefetch_depth(source.shard_size_hint(), acc_bytes),
        |d| d.max(1),
    );
    let mut acc = GroupNonzeroCounts::new(groups, n_groups, n_vars);
    let mut cursor = scx_format_io::VisibleRowCursor::new(n_obs);
    crate::prefetch::for_each_shard_ordered(source, depth, |shard_idx, shard| {
        let base = cursor
            .advance(shard.n_rows(), shard_idx, "group nonzero counts")
            .map_err(AccelError::ShapeError)?;
        acc.add_csr(&shard, base, groups)?;
        Ok(())
    })?;
    cursor
        .finish("group nonzero counts")
        .map_err(AccelError::ShapeError)?;
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
