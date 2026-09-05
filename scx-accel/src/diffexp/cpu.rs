//! Parallel Wilcoxon rank-sum differential expression.
//!
//! Implements the same algorithm as scanpy's `rank_genes_groups(method="wilcoxon")`:
//! for each gene and each group, compute a Wilcoxon rank-sum (Mann–Whitney U) test
//! comparing that group against the rest (or a specific reference group). Uses a
//! normal approximation with tie correction for the z-score, and Benjamini–Hochberg
//! for multiple testing correction.
//!
//! Parallelised over genes via `rayon`.

use crate::Result;
use rayon::prelude::*;
use scx_format_io::ShardSource;

/// Pseudocount added to group means before log2 fold-change computation.
/// Matches scanpy's value in `_rank_genes_groups.py`.
const LOGFC_PSEUDOCOUNT: f64 = 1e-9;

/// Results from a differential expression analysis.
///
/// Each field is indexed as `[group_idx][gene_rank]`, where genes are
/// sorted by descending signed score (default) or absolute score within each group.
#[derive(Debug, Clone)]
pub struct DiffExpResult {
    /// Group names in the order they appear in the results.
    pub group_names: Vec<String>,
    /// Gene names sorted by score for each group. `[n_groups][n_genes]`
    pub names: Vec<Vec<String>>,
    /// Global var index per gene, parallel to `names`. `[n_groups][n_genes]`.
    /// Carried solely to give the merge/sort a deterministic tiebreak
    /// (ascending var index on equal scores) that is identical across chunk
    /// sizes and backends; not surfaced to Python.
    pub gene_indices: Vec<Vec<usize>>,
    /// Z-scores (signed). `[n_groups][n_genes]`
    pub scores: Vec<Vec<f64>>,
    /// Raw p-values (two-sided). `[n_groups][n_genes]`
    pub pvals: Vec<Vec<f64>>,
    /// BH-adjusted p-values. `[n_groups][n_genes]`
    pub pvals_adj: Vec<Vec<f64>>,
    /// log2 fold-changes (group mean / reference mean). `[n_groups][n_genes]`
    pub logfoldchanges: Vec<Vec<f64>>,
    /// Which execution route produced this result (stamped by the dispatch
    /// entry point; `AccelRoute::Unknown` until then).
    pub exec_info: crate::route::AccelExecutionInfo,
}

impl DiffExpResult {
    /// Keep only the named groups, in the given order — scanpy's
    /// `rank_genes_groups(groups=[...])` as an *output* filter.
    ///
    /// Every tested group's statistics are computed against the same pool
    /// whether or not the caller asked for it (1-vs-rest keeps every other
    /// labelled cell in "rest", pairwise compares against the named reference,
    /// BH is per group), so selecting afterwards is numerically identical to
    /// selecting before, and identical to scanpy — which is the point: a
    /// kernel-level pre-filter would have changed what "rest" means.
    ///
    /// A name that is not in the result (a typo, or the reference group, which
    /// is never a tested group) is an error; the caller decides which of those
    /// to report and which to drop silently.
    pub fn restrict_to_groups(self, order: &[String]) -> Result<DiffExpResult> {
        let mut picks: Vec<usize> = Vec::with_capacity(order.len());
        for name in order {
            let idx = self
                .group_names
                .iter()
                .position(|g| g == name)
                .ok_or_else(|| {
                    crate::AccelError::InvalidInput(format!(
                        "restrict_to_groups: group {name:?} is not among the tested groups {:?}",
                        self.group_names
                    ))
                })?;
            if picks.contains(&idx) {
                return Err(crate::AccelError::InvalidInput(format!(
                    "restrict_to_groups: group {name:?} requested twice"
                )));
            }
            picks.push(idx);
        }
        fn take<T: Clone>(rows: &[Vec<T>], picks: &[usize]) -> Vec<Vec<T>> {
            picks.iter().map(|&i| rows[i].clone()).collect()
        }
        Ok(DiffExpResult {
            group_names: picks.iter().map(|&i| self.group_names[i].clone()).collect(),
            names: take(&self.names, &picks),
            gene_indices: take(&self.gene_indices, &picks),
            scores: take(&self.scores, &picks),
            pvals: take(&self.pvals, &picks),
            pvals_adj: take(&self.pvals_adj, &picks),
            logfoldchanges: take(&self.logfoldchanges, &picks),
            exec_info: self.exec_info,
        })
    }
}

/// Per-gene test result (before sorting/grouping).
#[derive(Debug, Clone)]
struct GeneTestResult {
    gene_idx: usize,   // local index into the (chunk's) `gene_names` slice
    global_idx: usize, // global var index (tiebreak key + emitted `gene_indices`)
    score: f64,        // z-statistic (signed)
    pval: f64,         // two-sided p-value
    logfc: f64,        // log2 fold-change
}

/// Shared DE-ranking comparator used by every ranking site (single-chunk,
/// chunked merge, GPU chunk assembly) so gene order is identical across chunk
/// sizes and backends.
///
/// Sorts by descending score (or `|score|` when `rankby_abs`); ties break by
/// ascending global var index, giving a strict total order. NaN scores sort
/// **last** deterministically (their key is treated as `-inf`).
///
/// The index tiebreak is a determinism choice, not a scanpy parity target:
/// scanpy ranks via `np.argsort(...)[::-1]` (unstable quicksort, then
/// reversed), so its order among equal scores is incidental and not stable.
/// Tied genes carry identical score/pval/logfc, so name-keyed consumers are
/// unaffected — only the order of otherwise-interchangeable rows is pinned.
pub(crate) fn de_rank_cmp(
    a_score: f64,
    a_idx: usize,
    b_score: f64,
    b_idx: usize,
    rankby_abs: bool,
) -> std::cmp::Ordering {
    let key = |s: f64| {
        let k = if rankby_abs { s.abs() } else { s };
        if k.is_nan() {
            f64::NEG_INFINITY // NaN sorts last (descending)
        } else {
            k
        }
    };
    key(b_score)
        .partial_cmp(&key(a_score))
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a_idx.cmp(&b_idx))
}

/// Run Wilcoxon rank-sum DE, 1-vs-rest or vs a specific reference group.
///
/// # Arguments
/// * `data` — Dense column-major matrix, shape `[n_obs × n_vars]` stored as
///   `data[obs * n_vars + var]`.
/// * `n_obs` — Number of observations (cells).
/// * `n_vars` — Number of variables (genes).
/// * `gene_names` — Gene names, length `n_vars`.
/// * `groups` — Group label per cell, length `n_obs`, encoded as indices `0..n_groups`.
///   A label `>= n_groups` marks an **unlabelled** cell (NaN / empty / off-category
///   upstream). Such a cell joins no group's sum or count, but it *is* in the
///   1-vs-rest rank pool and in every group's "rest" — scanpy 1.12's rule, and
///   pyscx's since 0.17. See [`super::groups`].
/// * `group_names` — Unique group names, length `n_groups`.
/// * `reference` — If `Some(idx)`, compare every other group against group `idx`.
///   If `None`, 1-vs-rest.
/// * `gene_index_base` — Global var index of `gene_names[0]`. Chunk drivers pass
///   the chunk start so emitted `gene_indices` (and the tiebreak) are global;
///   single-chunk / dense callers pass `0`.
#[allow(clippy::too_many_arguments)]
pub fn wilcoxon_rank_sum(
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: Option<usize>,
    log_transformed: bool,
    rankby_abs: bool,
    tie_correct: bool,
    gene_index_base: usize,
) -> Result<DiffExpResult> {
    let n_groups = group_names.len();
    if data.len() != n_obs * n_vars {
        return Err(crate::AccelError::InvalidInput(format!(
            "data length {} != n_obs {} × n_vars {}",
            data.len(),
            n_obs,
            n_vars
        )));
    }
    if groups.len() != n_obs {
        return Err(crate::AccelError::InvalidInput(format!(
            "groups length {} != n_obs {}",
            groups.len(),
            n_obs
        )));
    }
    // Finiteness is a contract at the DE accelerator boundary (ACC10).
    ensure_finite_de_input(data)?;

    // Pre-compute the per-group cell lists. A cell whose label is out of range
    // joins none of them, but it is still in the 1-vs-rest pool — see
    // `diffexp::groups`.
    let partition = super::groups::partition_by_group(groups, n_groups);
    let group_indices = &partition.group_indices;

    // Determine which groups to test and what they compare against.
    let test_groups: Vec<usize> = match reference {
        Some(ref_idx) => (0..n_groups).filter(|&g| g != ref_idx).collect(),
        None => (0..n_groups).collect(),
    };

    // Per-gene group sums (for logFC) are computed *inside* the parallel per-var
    // loop below, fused into the value gather each gene already performs — there is
    // no separate serial O(n_obs·n_vars) pre-pass. Each gene's sums are accumulated
    // in ascending cell order (matching `group_indices`, built ascending above), so
    // the f64 accumulation order — and thus the output — is identical to a dedicated
    // pre-pass would produce, just without the serial bottleneck or the n_groups×n_vars
    // allocation.

    let n_test_groups = test_groups.len();

    // --- Pre-rank approach: rank once per gene, then derive per-group statistics ---
    // For 1-vs-rest: rank the labelled pool once per gene (10× fewer sorts).
    // For pairwise: rank (group + ref) cells per test group per gene.
    let gene_group_results: Vec<Vec<(f64, f64, f64)>> = (0..n_vars)
        .into_par_iter()
        .map_init(
            || {
                // Thread-local buffers reused across genes (no per-gene allocation).
                // `group_sum_buf` holds this gene's per-group value sums (1-vs-rest arm).
                (
                    vec![0.0f64; n_obs],
                    Vec::with_capacity(n_obs),
                    Vec::with_capacity(n_obs),
                    // `n_groups + 1`: the last slot is the unlabelled bucket
                    // (`super::groups::pool_bucket`).
                    vec![0.0f64; n_groups + 1],
                )
            },
            |(values_buf, index_buf, ranks_buf, group_sum_buf), var_idx| {
                let mut group_results = Vec::with_capacity(n_test_groups);

                match reference {
                    None => {
                        // 1-vs-rest: rank the whole matrix once, reuse across groups.
                        // The pool is `0..n_obs` — an unlabelled cell is in no group
                        // but it ranks alongside everyone and it counts in every
                        // group's "rest" (scanpy's `X[~mask_g]`; see
                        // `diffexp::groups`), so pool position *is* the global row id.
                        //
                        // Fuse the per-group value sum into the gather (cells walked in
                        // ascending order → per-group sums match the `group_indices`
                        // order). The 1-vs-rest reference sum is then O(1) per group:
                        // `total - group_sum_buf[g]`. `group_sum_buf` is
                        // `n_groups + 1` wide so an unlabelled cell has a slot: its
                        // value must reach `total` (it is in every rest) without
                        // reaching any group.
                        group_sum_buf[..=n_groups].iter_mut().for_each(|s| *s = 0.0);
                        for i in 0..n_obs {
                            let v = data[i * n_vars + var_idx] as f64;
                            values_buf[i] = v;
                            group_sum_buf[super::groups::pool_bucket(groups[i], n_groups)] += v;
                        }
                        let total: f64 = group_sum_buf[..=n_groups].iter().sum();
                        let raw_tc = rank_with_ties(&values_buf[..n_obs], index_buf, ranks_buf);
                        let tc = if tie_correct { raw_tc } else { 0.0 };

                        for &g in &test_groups {
                            let n1 = group_indices[g].len();
                            if n1 == 0 || n1 == n_obs {
                                group_results.push((f64::NAN, 1.0, f64::NAN));
                                continue;
                            }
                            let n2 = n_obs - n1;

                            // Ranks are indexed by pool position, and the pool is
                            // every row, so the group's global cell ids index them
                            // directly.
                            let (score, pval) =
                                wilcoxon_from_ranks(ranks_buf, &group_indices[g], n_obs, tc);

                            let mean_group = group_sum_buf[g] / n1 as f64;
                            let rest_sum = total - group_sum_buf[g];
                            let mean_ref = rest_sum / n2 as f64;
                            let logfc = compute_logfc(mean_group, mean_ref, log_transformed);
                            group_results.push((score, pval, logfc));
                        }
                    }
                    Some(ref_idx) => {
                        // Pairwise: rank only (group + ref) cells per test group.
                        let ref_cells = &group_indices[ref_idx];
                        for &g in &test_groups {
                            let group_cells = &group_indices[g];
                            let n1 = group_cells.len();
                            let n2 = ref_cells.len();
                            if n1 == 0 || n2 == 0 {
                                group_results.push((f64::NAN, 1.0, f64::NAN));
                                continue;
                            }
                            let n_total = n1 + n2;

                            // Gather combined values into buffer.
                            for (i, &cell) in group_cells.iter().enumerate() {
                                values_buf[i] = data[cell * n_vars + var_idx] as f64;
                            }
                            for (i, &cell) in ref_cells.iter().enumerate() {
                                values_buf[n1 + i] = data[cell * n_vars + var_idx] as f64;
                            }

                            let raw_tc =
                                rank_with_ties(&values_buf[..n_total], index_buf, ranks_buf);
                            let tc = if tie_correct { raw_tc } else { 0.0 };

                            // Group cells are at indices 0..n1 in the combined buffer.
                            let group_indices_in_buf: Vec<usize> = (0..n1).collect();
                            let (score, pval) =
                                wilcoxon_from_ranks(ranks_buf, &group_indices_in_buf, n_total, tc);

                            // Group/ref cells are already gathered (ascending cell order)
                            // into values_buf[0..n1] and values_buf[n1..n_total]; sum those
                            // directly rather than from a pre-built group_gene_sums matrix.
                            let mean_group = values_buf[..n1].iter().sum::<f64>() / n1 as f64;
                            let mean_ref = values_buf[n1..n_total].iter().sum::<f64>() / n2 as f64;
                            let logfc = compute_logfc(mean_group, mean_ref, log_transformed);
                            group_results.push((score, pval, logfc));
                        }
                    }
                }
                group_results
            },
        )
        .collect();

    // Transpose: gene_group_results[var][tg_idx] -> per-group sorted gene lists.
    let mut result_names = Vec::with_capacity(n_test_groups);
    let mut result_gene_indices = Vec::with_capacity(n_test_groups);
    let mut result_scores = Vec::with_capacity(n_test_groups);
    let mut result_pvals = Vec::with_capacity(n_test_groups);
    let mut result_pvals_adj = Vec::with_capacity(n_test_groups);
    let mut result_logfc = Vec::with_capacity(n_test_groups);
    let mut result_group_names = Vec::with_capacity(n_test_groups);

    for (tg_idx, &g) in test_groups.iter().enumerate() {
        let mut sorted: Vec<GeneTestResult> = (0..n_vars)
            .map(|var_idx| {
                let (score, pval, logfc) = gene_group_results[var_idx][tg_idx];
                GeneTestResult {
                    gene_idx: var_idx,
                    global_idx: gene_index_base + var_idx,
                    score,
                    pval,
                    logfc,
                }
            })
            .collect();

        sorted
            .sort_by(|a, b| de_rank_cmp(a.score, a.global_idx, b.score, b.global_idx, rankby_abs));

        let names: Vec<String> = sorted
            .iter()
            .map(|r| gene_names[r.gene_idx].clone())
            .collect();
        let gene_indices: Vec<usize> = sorted.iter().map(|r| r.global_idx).collect();
        let scores: Vec<f64> = sorted.iter().map(|r| r.score).collect();
        let pvals: Vec<f64> = sorted.iter().map(|r| r.pval).collect();
        let logfc: Vec<f64> = sorted.iter().map(|r| r.logfc).collect();

        let pvals_adj = benjamini_hochberg(&pvals);

        result_names.push(names);
        result_gene_indices.push(gene_indices);
        result_scores.push(scores);
        result_pvals.push(pvals);
        result_pvals_adj.push(pvals_adj);
        result_logfc.push(logfc);
        result_group_names.push(group_names[g].clone());
    }

    Ok(DiffExpResult {
        group_names: result_group_names,
        names: result_names,
        gene_indices: result_gene_indices,
        scores: result_scores,
        pvals: result_pvals,
        pvals_adj: result_pvals_adj,
        logfoldchanges: result_logfc,
        exec_info: crate::route::AccelExecutionInfo::default(),
    })
}

/// Wilcoxon `(z, two_sided_p)` from a **precomputed** group rank-sum.
///
/// Same math as [`wilcoxon_from_ranks`] (continuity=`false`, the scanpy-parity
/// path) but takes the group rank-sum directly instead of summing a ranks array
/// — the seam the exact sparse-nnz Wilcoxon (§5.3) uses, since it computes the
/// rank-sum analytically from a gene's nonzeros + implicit-zero block rather
/// than materializing an `n_obs` ranks array. `n1` is the test-group cell count,
/// `n_total` the full cell count, `tie_correction` the `Σ(t³−t)` term.
pub(crate) fn wilcoxon_stats_from_rank_sum(
    rank_sum: f64,
    n1: usize,
    n_total: usize,
    tie_correction: f64,
) -> (f64, f64) {
    let n1f = n1 as f64;
    let n2f = (n_total - n1) as f64;
    let n = n_total as f64;
    if n1f == 0.0 || n2f == 0.0 {
        return (0.0, 1.0);
    }
    let u1 = rank_sum - n1f * (n1f + 1.0) / 2.0;
    let mu = n1f * n2f / 2.0;
    let sigma_sq = (n1f * n2f / 12.0) * ((n + 1.0) - tie_correction / (n * (n - 1.0)));
    if sigma_sq <= 0.0 {
        return (0.0, 1.0);
    }
    let sigma = sigma_sq.sqrt();
    let z = (u1 - mu) / sigma;
    let p = 2.0 * normal_sf(z.abs());
    (z, p)
}

/// Compute log2 fold-change between group and reference means.
pub(crate) fn compute_logfc(mean_group: f64, mean_ref: f64, log_transformed: bool) -> f64 {
    if log_transformed {
        let expm1_group = mean_group.exp_m1();
        let expm1_ref = mean_ref.exp_m1();
        ((expm1_group + LOGFC_PSEUDOCOUNT) / (expm1_ref + LOGFC_PSEUDOCOUNT)).log2()
    } else {
        (mean_group + LOGFC_PSEUDOCOUNT).log2() - (mean_ref + LOGFC_PSEUDOCOUNT).log2()
    }
}

/// Reject non-finite input at the DE accelerator boundary (ACC10).
///
/// NaN/Inf cannot be ranked meaningfully — a NaN sorts arbitrarily and poisons
/// the Wilcoxon U statistic and p-value. This is an always-on check (the cost
/// is one pass over a matrix the ranking already streams), replacing the
/// per-gene `debug_assert!` that vanished in release builds. Every dense DE
/// entry point calls it; the sparse/streaming variants inherit it by
/// delegating to the dense kernels per gene chunk.
fn ensure_finite_de_input(data: &[f32]) -> Result<()> {
    if let Some(pos) = data.iter().position(|v| !v.is_finite()) {
        return Err(crate::AccelError::InvalidInput(format!(
            "data contains a non-finite value ({}) at index {pos}; differential expression \
             requires finite input — filter/QC NaN and Inf before ranking",
            data[pos]
        )));
    }
    Ok(())
}

/// Walk the equal-value runs of a sorted sequence, handing each run its 1-based
/// mid-rank, and return the Wilcoxon tie correction `Σ(t³ − t)`.
///
/// **The one definition of mid-ranking and tie correction in this crate**
/// (ORG-7.21-3). Both Wilcoxon rank assignments reduce to this walk and differ
/// only in what they do with a run:
///
/// | caller | `equal` | `offset` | `on_run` |
/// |---|---|---|---|
/// | [`rank_with_ties`] (dense) | compares through a sort permutation | `0` — the run spans a whole column | writes `ranks[idx]` per element |
/// | `csc::wilcoxon::assign_block_ranks` (analytic nnz) | compares `(value, group)` pairs | elements ordered before this block | accumulates into `rank_sum[group]` |
///
/// They were independent copies until this primitive, and that is how the nnz
/// path came to re-derive the 1-vs-rest logFC bug on its own.
///
/// `equal(i, j)` reports whether sorted positions `i` and `j` hold the same
/// value. `offset` is the count of elements ordered *before* this sequence, so
/// the mid-rank of the run at `[i, j)` is `offset + (i + 1 + j) / 2` — computed
/// as `(2·offset + i + 1 + j) / 2` to keep it a single exact division.
///
/// **The GPU arm is deliberately absent.** `scx-gpu`'s ranking lives in a
/// `.cu` kernel and cannot call this; it is covered by the shared *reference
/// values* in `wilcoxon_reference_tests.rs` instead, which is the only kind of
/// agreement a host primitive and a device kernel can have.
pub(crate) fn for_each_tie_run(
    len: usize,
    offset: usize,
    equal: impl Fn(usize, usize) -> bool,
    mut on_run: impl FnMut(f64, std::ops::Range<usize>),
) -> f64 {
    let mut tie_correction = 0.0f64;
    let mut i = 0;
    while i < len {
        let mut j = i + 1;
        while j < len && equal(i, j) {
            j += 1;
        }
        on_run((2 * offset + i + 1 + j) as f64 / 2.0, i..j);
        let t = (j - i) as f64;
        if t > 1.0 {
            tie_correction += t * t * t - t;
        }
        i = j;
    }
    tie_correction
}

/// Rank values with mid-rank tie handling. Returns `(ranks, tie_correction)`.
///
/// `ranks[i]` is the 1-based mid-rank for `values[i]`.
/// `tie_correction` is `Σ (t³ - t)` over tie groups, used in the variance
/// formula for the Wilcoxon test. Shared across all group comparisons for
/// the same gene, since ties are a property of the value distribution.
///
/// # Precondition
///
/// `values` MUST be finite. Finiteness is enforced once, always-on, at the DE
/// accelerator entry boundary (`wilcoxon_rank_sum` rejects non-finite input
/// with `AccelError::InvalidInput`), so this function can assume it. The sort
/// uses [`f64::total_cmp`] rather than `partial_cmp`, giving a deterministic
/// total order even if a NaN somehow reached here — defence in depth, never the
/// primary guard (a stray NaN still yields a meaningless rank, just a stable
/// one) (finding ACC10).
fn rank_with_ties(values: &[f64], index_buf: &mut Vec<usize>, ranks: &mut Vec<f64>) -> f64 {
    let n = values.len();
    index_buf.clear();
    index_buf.extend(0..n);
    index_buf.sort_unstable_by(|&a, &b| values[a].total_cmp(&values[b]));

    ranks.resize(n, 0.0);
    let order: &[usize] = index_buf;
    for_each_tie_run(
        n,
        0,
        |i, j| values[order[j]] == values[order[i]],
        |mid_rank, run| {
            for &idx in &order[run] {
                ranks[idx] = mid_rank;
            }
        },
    )
}

/// Compute Wilcoxon rank-sum z-score and p-value from pre-computed ranks.
///
/// `ranks` contains the 1-based mid-ranks for ALL cells (length n_obs).
/// `group_cells` are the indices of cells in the test group.
/// `n_total` is the total number of cells (n_obs).
/// `tie_correction` is the pre-computed tie correction term from `rank_with_ties`.
fn wilcoxon_from_ranks(
    ranks: &[f64],
    group_cells: &[usize],
    n_total: usize,
    tie_correction: f64,
) -> (f64, f64) {
    // scanpy's `rank_genes_groups` (method="wilcoxon") does NOT apply a
    // continuity correction — keep this path uncorrected.
    let (_u, z, p) = wilcoxon_full_from_ranks(ranks, group_cells, n_total, tie_correction, false);
    (z, p)
}

/// Full Wilcoxon stats from pre-computed ranks: `(u1, z, two_sided_p)`.
///
/// `u1` is the Mann-Whitney U statistic for the test group (the convention
/// matched by `pdex` / `numba_mwu`'s `.statistic`).  `z` is the
/// tie-corrected signed z-score used by the existing scanpy-parity path.
/// All callers within this crate use one or the other; the unified helper
/// avoids recomputing the rank sum twice when both are needed.
///
/// `continuity` selects the two-sided p-value convention. `pdex_ref` matches
/// upstream `pdex` / `numba_mwu` (and scipy's `use_continuity=True` default):
/// subtract 0.5 from `|U − μ|` before standardizing. The scanpy-parity
/// Wilcoxon path passes `false` (scanpy applies no continuity correction).
/// The returned signed `z` is always the *uncorrected* score (used only by the
/// Wilcoxon path); only the p-value reflects `continuity`.
fn wilcoxon_full_from_ranks(
    ranks: &[f64],
    group_cells: &[usize],
    n_total: usize,
    tie_correction: f64,
    continuity: bool,
) -> (f64, f64, f64) {
    let n1 = group_cells.len() as f64;
    let n2 = n_total as f64 - n1;
    let n = n_total as f64;

    if n1 == 0.0 || n2 == 0.0 {
        return (0.0, 0.0, 1.0);
    }

    let rank_sum: f64 = group_cells.iter().map(|&i| ranks[i]).sum();
    let u1 = rank_sum - n1 * (n1 + 1.0) / 2.0;

    let mu = n1 * n2 / 2.0;
    let sigma_sq = (n1 * n2 / 12.0) * ((n + 1.0) - tie_correction / (n * (n - 1.0)));

    if sigma_sq <= 0.0 {
        return (u1, 0.0, 1.0);
    }

    let sigma = sigma_sq.sqrt();
    let z = (u1 - mu) / sigma;
    let z_p = if continuity {
        ((u1 - mu).abs() - 0.5).max(0.0) / sigma
    } else {
        z.abs()
    };
    let p = 2.0 * normal_sf(z_p);
    (u1, z, p)
}

/// Wilcoxon rank-sum (Mann–Whitney U) test with normal approximation and tie correction.
///
/// Returns `(z_score, two_sided_p_value)`.
/// Used only in tests; production code uses `rank_with_ties` + `wilcoxon_from_ranks`.
#[cfg(test)]
fn wilcoxon_test(group: &[f64], rest: &[f64]) -> (f64, f64) {
    let n1 = group.len() as f64;
    let n2 = rest.len() as f64;
    let n = n1 + n2;

    if n1 == 0.0 || n2 == 0.0 {
        return (0.0, 1.0);
    }

    // Combine and rank.
    // Each element: (value, source: 0=group, 1=rest)
    let mut combined: Vec<(f64, u8)> = Vec::with_capacity(group.len() + rest.len());
    for &v in group {
        combined.push((v, 0));
    }
    for &v in rest {
        combined.push((v, 1));
    }

    // Sort by value (stable sort to handle ties consistently).
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Assign mid-ranks and accumulate the group's rank sum, through the SAME
    // walk the two production kernels use (ORG-7.21-3). This was a third,
    // independent copy, spelled `tie_size * tie_size * tie_size - tie_size` --
    // which the CI guard's `t * t * t - t` grep could not see. Test-only today,
    // but it is exactly the spelling a later production clone would use to walk
    // around the guard, so the guard now keys on the shape and this body no
    // longer has one.
    let mut rank_sum_group: f64 = 0.0;
    let tie_correction = for_each_tie_run(
        combined.len(),
        0,
        |i, j| combined[j].0 == combined[i].0,
        |mid_rank, run| {
            for item in &combined[run] {
                if item.1 == 0 {
                    rank_sum_group += mid_rank;
                }
            }
        },
    );

    // U-statistic for group.
    let u1 = rank_sum_group - n1 * (n1 + 1.0) / 2.0;

    // Expected U and variance under H0.
    let mu = n1 * n2 / 2.0;
    let sigma_sq = (n1 * n2 / 12.0) * ((n + 1.0) - tie_correction / (n * (n - 1.0)));

    if sigma_sq <= 0.0 {
        return (0.0, 1.0);
    }

    let sigma = sigma_sq.sqrt();
    let z = (u1 - mu) / sigma;

    // Two-sided p-value using normal approximation.
    let p = 2.0 * normal_cdf(-z.abs());

    (z, p)
}

/// Standard normal survival function: P(Z > z) = 1 − Φ(z).
///
/// Delegates to `libm::erfc`, which is the same IEEE-754-accurate
/// implementation `scipy.stats.norm.sf` uses via the C math library.
/// Accurate to ~1 ULP across the full range and stays finite down to
/// ~1e-300 without needing the Mills-ratio tail the old hand-rolled
/// implementation used for |z| > 8.
///
/// Identity: sf(z) = 0.5 · erfc(z / √2).
pub(crate) fn normal_sf(z: f64) -> f64 {
    0.5 * libm::erfc(z / std::f64::consts::SQRT_2)
}

/// Standard normal CDF: Φ(z) = 1 − sf(z).
#[cfg(test)]
fn normal_cdf(z: f64) -> f64 {
    1.0 - normal_sf(z)
}

/// Benjamini–Hochberg p-value adjustment.
///
/// Takes p-values in their current order and returns adjusted p-values
/// in the same order.
pub fn benjamini_hochberg(pvals: &[f64]) -> Vec<f64> {
    let n = pvals.len();
    if n == 0 {
        return vec![];
    }

    // Create index-sorted array by p-value ascending.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        pvals[a]
            .partial_cmp(&pvals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut adjusted = vec![0.0; n];
    let mut cummin = f64::INFINITY;

    // Process from largest p-value to smallest.
    for (rank_from_end, &orig_idx) in indices.iter().enumerate().rev() {
        let rank = rank_from_end + 1; // 1-indexed rank
        let adj = (pvals[orig_idx] * n as f64 / rank as f64).min(1.0);
        cummin = cummin.min(adj);
        adjusted[orig_idx] = cummin;
    }

    adjusted
}

/// Gene-chunked streaming Wilcoxon rank-sum over a CSR [`ShardSource`].
///
/// Generic over the source rather than taking a `BackedCsrReader`: `groups` is
/// indexed by *visible* cell, so a caller holding a subset SCX handle must pass
/// that handle's view (`as_shard_source()`). Passing the reader underneath it
/// streams every on-disk row and trips the `groups.len() != n_obs` guard below.
/// Multi-pass, so a caching source should opt in (`with_cached_reads()`).
///
/// Instead of materializing the full matrix, processes genes in chunks:
/// 1. For each gene chunk, iterate all shards via `read_shard_arc()`,
///    apply `project_csr()` per shard, scatter into a dense buffer. The
///    cache is populated on the first chunk and reused by every subsequent
///    chunk; sizing `cache_shards >= n_shards` on the `BackedCsrReader`
///    makes the inner loop fully cache-resident after the first pass.
///    A too-small cache evicts the shard the next chunk needs first (the
///    iteration order is linear `0..n_shards`), which defeats the win.
/// 2. Run `wilcoxon_rank_sum()` on the dense buffer for that chunk.
/// 3. Merge all chunk results with global BH correction.
///
/// Peak memory:
///   * O(n_obs × gene_chunk_size) for the dense buffer per chunk
///   * + O(min(cache_shards, n_shards) × decoded-shard-bytes) for the LRU
///       shard cache (≈ 640 MB / shard on Replogle-scale inputs)
#[allow(clippy::too_many_arguments)]
pub fn wilcoxon_rank_sum_streaming<S: ShardSource>(
    source: &S,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: Option<usize>,
    gene_chunk_size: usize,
    log_transformed: bool,
    rankby_abs: bool,
    tie_correct: bool,
) -> Result<DiffExpResult> {
    let n_obs = source.n_obs();
    let n_vars = gene_names.len();

    if groups.len() != n_obs {
        return Err(crate::AccelError::InvalidInput(format!(
            "groups length {} != n_obs {}",
            groups.len(),
            n_obs
        )));
    }
    if gene_chunk_size == 0 {
        return Err(crate::AccelError::InvalidInput(
            "gene_chunk_size must be > 0".to_string(),
        ));
    }
    // `gene_names` describes the *visible* gene axis; the source must present
    // that same axis. Without this a caller that handed over the file instead
    // of a subset handle's view is not caught: the chunk loop slices columns
    // `0..k` and `project_csr` silently drops anything out of range, so the
    // result comes back fully formed and describing the wrong genes. Mirrors
    // the check `pseudobulk_aggregate` already makes.
    if n_vars != source.n_vars() {
        return Err(crate::AccelError::ShapeError(format!(
            "wilcoxon_rank_sum_streaming: gene_names length {} != source.n_vars() {} — the source must \
             be the same gene axis the names describe (pass the handle's view, \
             e.g. `as_shard_source()`, not the raw reader)",
            n_vars,
            source.n_vars(),
        )));
    }

    // Clamp the dense n_obs×chunk f32 workspace to the CPU memory budget.
    let gene_chunk_size = crate::mem_budget::de_gene_chunk_or_err(
        gene_chunk_size,
        n_obs,
        "wilcoxon_rank_sum_streaming",
    )?;

    let n_shards = source.n_shards();

    // Cache-sizing footgun guard. The kernel walks every shard once per
    // gene chunk; if the LRU can't hold all `n_shards` decoded shards
    // simultaneously, the iteration order `0..n_shards` evicts the
    // shard the next chunk re-requests *first*, so the cached path is
    // strictly slower than `read_shard_uncached` (LRU bookkeeping +
    // re-decode). Warn once per call so the caller sees it without
    // spamming per-shard.
    // `None` = a non-caching source (every `read_shard_arc` re-decodes by
    // design); there is no cache to size, so there is nothing to warn about.
    let n_chunks = n_vars.div_ceil(gene_chunk_size);
    if let Some(cache_cap) = source.shard_cache_capacity() {
        if n_chunks > 1 && cache_cap < n_shards {
            log::warn!(
                "wilcoxon_rank_sum_streaming: cache_shards={} < n_shards={} with {} gene chunks — \
                 the cached read path will evict and re-decode every shard on each chunk. \
                 Size the shard cache to >= n_shards for the documented speedup.",
                cache_cap,
                n_shards,
                n_chunks,
            );
        }
    }

    let mut all_chunk_results = Vec::new();

    for chunk_start in (0..n_vars).step_by(gene_chunk_size) {
        let chunk_end = (chunk_start + gene_chunk_size).min(n_vars);
        let chunk_size = chunk_end - chunk_start;
        let col_indices: Vec<u32> = (chunk_start as u32..chunk_end as u32).collect();

        // Build dense buffer (n_obs × chunk_size), zero-initialized.
        let mut dense = vec![0.0f32; n_obs * chunk_size];

        // Stream all shards, project each, scatter into dense buffer.
        //
        // Use the cached API: this kernel makes one pass per gene chunk,
        // so every shard is read O(n_chunks) times. With `cache_shards`
        // sized to hold all shards (the typical Python-side default for
        // full-matrix DE), the second chunk onward is fully cache-resident
        // and the inner loop pays no decompression cost. `&Arc<ScxCsr>`
        // auto-derefs to `&ScxCsr` for `project_csr` — no extra clone.
        let mut global_row = 0usize;
        for shard_idx in 0..n_shards {
            let shard_csr = source
                .read_shard_arc(shard_idx)
                .map_err(crate::AccelError::Scx)?;
            let _r = scx_format_io::reduction_guard();
            let projected = scx_engine::project_csr(&shard_csr, &col_indices);

            for row in 0..projected.n_rows() {
                let start = projected.indptr[row] as usize;
                let end = projected.indptr[row + 1] as usize;
                for j in start..end {
                    let col = projected.indices[j] as usize;
                    dense[(global_row + row) * chunk_size + col] = projected.data[j];
                }
            }
            global_row += projected.n_rows();
        }

        // Run existing wilcoxon_rank_sum on this chunk's dense buffer. The
        // per-gene ranking is the dominant DE compute (O(n_obs·log n_obs)/gene)
        // and runs after the decode/densify above, so it attributes to
        // `reduction` disjointly from `decode`.
        let _rank = scx_format_io::reduction_guard();
        let chunk_genes: Vec<String> = gene_names[chunk_start..chunk_end].to_vec();
        let chunk_result = wilcoxon_rank_sum(
            &dense,
            n_obs,
            chunk_size,
            &chunk_genes,
            groups,
            group_names,
            reference,
            log_transformed,
            rankby_abs,
            tie_correct,
            chunk_start,
        )?;
        all_chunk_results.push(chunk_result);
    }

    // Merge: concatenate per-group gene lists, re-sort, and re-apply BH correction globally.
    merge_diff_exp_results(all_chunk_results, rankby_abs)
}

/// Gene-chunked Wilcoxon rank-sum from an in-memory `ScxCsr`.
///
/// Like `wilcoxon_rank_sum_streaming` but operates on a single in-memory CSR
/// matrix instead of streaming shards from a `BackedCsrReader`. Avoids the
/// O(n_obs × n_vars) dense materialization that `.toarray()` would require.
///
/// Peak memory: O(n_obs × gene_chunk_size) instead of O(n_obs × n_vars).
#[allow(clippy::too_many_arguments)]
pub fn wilcoxon_rank_sum_sparse(
    csr: &scx_sparse::ScxCsr,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: Option<usize>,
    gene_chunk_size: usize,
    log_transformed: bool,
    rankby_abs: bool,
    tie_correct: bool,
) -> Result<DiffExpResult> {
    let n_obs = csr.n_rows();
    let n_vars = gene_names.len();

    if groups.len() != n_obs {
        return Err(crate::AccelError::InvalidInput(format!(
            "groups length {} != n_obs {}",
            groups.len(),
            n_obs
        )));
    }
    if gene_chunk_size == 0 {
        return Err(crate::AccelError::InvalidInput(
            "gene_chunk_size must be > 0".to_string(),
        ));
    }
    if n_vars != csr.n_cols() {
        return Err(crate::AccelError::InvalidInput(format!(
            "gene_names length {} != csr n_cols {}",
            n_vars,
            csr.n_cols()
        )));
    }
    // Clamp the dense n_obs×chunk f32 workspace to the CPU memory budget.
    let gene_chunk_size = crate::mem_budget::de_gene_chunk_or_err(
        gene_chunk_size,
        n_obs,
        "wilcoxon_rank_sum_sparse",
    )?;

    let mut all_chunk_results = Vec::new();

    for chunk_start in (0..n_vars).step_by(gene_chunk_size) {
        let chunk_end = (chunk_start + gene_chunk_size).min(n_vars);
        let chunk_size = chunk_end - chunk_start;
        let col_indices: Vec<u32> = (chunk_start as u32..chunk_end as u32).collect();

        // Project to only the genes in this chunk.
        let projected = scx_engine::project_csr(csr, &col_indices);

        // Scatter projected CSR into a dense buffer (n_obs × chunk_size).
        let mut dense = vec![0.0f32; n_obs * chunk_size];
        for row in 0..projected.n_rows() {
            let start = projected.indptr[row] as usize;
            let end = projected.indptr[row + 1] as usize;
            for j in start..end {
                let col = projected.indices[j] as usize;
                dense[row * chunk_size + col] = projected.data[j];
            }
        }

        let chunk_genes: Vec<String> = gene_names[chunk_start..chunk_end].to_vec();
        let chunk_result = wilcoxon_rank_sum(
            &dense,
            n_obs,
            chunk_size,
            &chunk_genes,
            groups,
            group_names,
            reference,
            log_transformed,
            rankby_abs,
            tie_correct,
            chunk_start,
        )?;
        all_chunk_results.push(chunk_result);
    }

    merge_diff_exp_results(all_chunk_results, rankby_abs)
}

/// Merge per-chunk `DiffExpResult`s into a single result with global BH correction.
///
/// For each group:
/// 1. Concatenate gene names, scores, p-values, and fold-changes from all chunks.
/// 2. Re-sort by descending signed score (or |score| when `rankby_abs`).
/// 3. Apply Benjamini–Hochberg on the globally-sorted raw p-values.
///
/// This is essential because BH correction depends on the total number of tests.
/// Applying BH per-chunk would use `chunk_size` as the denominator instead of
/// `n_vars`, inflating FDR.
pub fn merge_diff_exp_results(
    chunks: Vec<DiffExpResult>,
    rankby_abs: bool,
) -> Result<DiffExpResult> {
    if chunks.is_empty() {
        return Ok(DiffExpResult {
            group_names: vec![],
            names: vec![],
            gene_indices: vec![],
            scores: vec![],
            pvals: vec![],
            pvals_adj: vec![],
            logfoldchanges: vec![],
            exec_info: crate::route::AccelExecutionInfo::default(),
        });
    }
    if chunks.len() == 1 {
        return Ok(chunks.into_iter().next().unwrap());
    }

    // All chunks must have the same group structure.
    let group_names = chunks[0].group_names.clone();
    let n_groups = group_names.len();
    // Carry the route metadata of the first chunk onto the merged result —
    // every chunk shares the same dispatch route.
    let merged_exec_info = chunks[0].exec_info.clone();

    let mut merged_names = Vec::with_capacity(n_groups);
    let mut merged_gene_indices = Vec::with_capacity(n_groups);
    let mut merged_scores = Vec::with_capacity(n_groups);
    let mut merged_pvals = Vec::with_capacity(n_groups);
    let mut merged_pvals_adj = Vec::with_capacity(n_groups);
    let mut merged_logfc = Vec::with_capacity(n_groups);

    for group_name in &group_names {
        // Collect all genes for this group across chunks.
        // Tuple: (name, score, pval, logfc, global_gene_idx).
        let mut gene_entries: Vec<(String, f64, f64, f64, usize)> = Vec::new();

        for chunk in &chunks {
            // Find the position of this group in the chunk's results.
            let chunk_g_pos = chunk.group_names.iter().position(|n| n == group_name);
            let chunk_g_pos = match chunk_g_pos {
                Some(p) => p,
                None => continue, // group not present in this chunk (shouldn't happen)
            };

            let n_genes = chunk.names[chunk_g_pos].len();
            for i in 0..n_genes {
                gene_entries.push((
                    chunk.names[chunk_g_pos][i].clone(),
                    chunk.scores[chunk_g_pos][i],
                    chunk.pvals[chunk_g_pos][i],
                    chunk.logfoldchanges[chunk_g_pos][i],
                    chunk.gene_indices[chunk_g_pos][i],
                ));
            }
        }

        // Sort via the shared comparator: descending score (or |score|), ties
        // broken by ascending global var index, NaN scores last.
        gene_entries.sort_by(|a, b| de_rank_cmp(a.1, a.4, b.1, b.4, rankby_abs));

        let names: Vec<String> = gene_entries.iter().map(|e| e.0.clone()).collect();
        let scores: Vec<f64> = gene_entries.iter().map(|e| e.1).collect();
        let pvals: Vec<f64> = gene_entries.iter().map(|e| e.2).collect();
        let logfc: Vec<f64> = gene_entries.iter().map(|e| e.3).collect();
        let gene_indices: Vec<usize> = gene_entries.iter().map(|e| e.4).collect();

        // Global BH correction across ALL genes.
        let pvals_adj = benjamini_hochberg(&pvals);

        merged_names.push(names);
        merged_gene_indices.push(gene_indices);
        merged_scores.push(scores);
        merged_pvals.push(pvals);
        merged_pvals_adj.push(pvals_adj);
        merged_logfc.push(logfc);
    }

    Ok(DiffExpResult {
        group_names,
        names: merged_names,
        gene_indices: merged_gene_indices,
        scores: merged_scores,
        pvals: merged_pvals,
        pvals_adj: merged_pvals_adj,
        logfoldchanges: merged_logfc,
        exec_info: merged_exec_info,
    })
}

// ---------------------------------------------------------------------------
// pdex `mode="ref"` accelerator
// ---------------------------------------------------------------------------

/// Result of a pdex `mode="ref"` differential expression analysis.
///
/// Mirrors the row-per-(group, feature) frame returned by
/// `pdex.pdex(adata, groupby, mode="ref")`. Inner vectors are indexed
/// `[test_group_idx][gene_idx]`, with `feature_names` in input (var_names)
/// order — there is no per-group sorting (pdex returns input order).
///
/// The reference group is excluded from `group_names`.
#[derive(Debug, Clone)]
pub struct PdexRefResult {
    /// Names of test groups, in the input `group_names` order (reference excluded).
    pub group_names: Vec<String>,
    /// Gene names in input order. Length = `n_vars`.
    pub feature_names: Vec<String>,
    /// `target_mean[group_idx][gene_idx]` in natural (count) space.
    pub target_means: Vec<Vec<f64>>,
    /// `ref_mean[gene_idx]` in natural (count) space.
    pub ref_means: Vec<f64>,
    /// Cell count per test group.
    pub target_memberships: Vec<usize>,
    /// Cell count for the reference group.
    pub ref_membership: usize,
    /// `log2((target_mean + epsilon) / (ref_mean + epsilon))`, matching upstream
    /// pdex: `0/0 -> 0.0`, one-sided zeros `-> ±inf` (only when `epsilon == 0`).
    pub log2_fold_changes: Vec<Vec<f64>>,
    /// `(target_mean - ref_mean) / (ref_mean + epsilon)`, matching upstream pdex:
    /// `0/0 -> 0.0`, `+inf` preserved when `ref_mean + epsilon == 0` and
    /// `target_mean > 0` (only when `epsilon == 0`).
    pub percent_changes: Vec<Vec<f64>>,
    /// Mann-Whitney U statistic for the test group vs the reference.
    pub statistics: Vec<Vec<f64>>,
    /// Two-sided MWU p-values.
    pub p_values: Vec<Vec<f64>>,
    /// Benjamini-Hochberg adjusted p-values per group across genes.
    pub fdrs: Vec<Vec<f64>>,
    /// Per-(test group, gene) **arithmetic** count-space mean, aligned to
    /// `feature_names`. Populated only when CPM filtering is requested (it feeds
    /// the `cpm_filter` keep decision and is never reported); otherwise the
    /// inner vectors are empty.
    pub target_arith_gene_means: Vec<Vec<f64>>,
    /// Per-gene reference **arithmetic** count-space mean, aligned to
    /// `feature_names`. Populated only when CPM filtering is requested.
    pub ref_arith_gene_means: Vec<f64>,
    /// When CPM filtering has been applied, `kept_indices[group_idx]` lists the
    /// indices into `feature_names` / `ref_means` that survived for that test
    /// group, and every per-group result vector (`target_means`,
    /// `log2_fold_changes`, …) has been compacted to that surviving set. `None`
    /// means no filtering was applied and all groups share the full
    /// `feature_names` axis.
    pub kept_indices: Option<Vec<Vec<usize>>>,
    /// Which execution route produced this result (stamped by the dispatch
    /// entry point; `AccelRoute::Unknown` until then).
    pub exec_info: crate::route::AccelExecutionInfo,
}

/// Maps `0/0`-style `NaN` results to `0.0` while preserving `±inf` (one-sided
/// zeros). Matches upstream pdex's `lfc[isnan] = 0.0` / `pc[isnan] = 0.0`
/// semantics for `log2_fold_change` and `percent_change`. Shared by the CPU
/// kernel and the GPU host-side fold-change computation so the two stay
/// bit-identical on genes unexpressed in both groups.
#[inline]
pub(crate) fn nan_to_zero(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x
    }
}

/// Per-cell value transform `f(x)` applied before averaging for pdex pseudobulk.
#[inline]
fn pdex_pre(mode: crate::pseudobulk::GeomMeanMode, x: f64) -> f64 {
    mode.pre(x)
}

/// Per-(group, gene) mean transform `g(y)` applied after dividing by cell count.
#[inline]
fn pdex_post(mode: crate::pseudobulk::GeomMeanMode, y: f64) -> f64 {
    mode.post(y)
}

/// Per-(test group, gene) pdex stats from a dense column buffer with a
/// pre-computed `ref_mean`.
///
/// `values` has length `n_obs`, holding the column for one gene.
/// `group_cells` and `ref_cells` are disjoint cell-index arrays into `values`.
/// Returns `(target_mean, log2_fc, percent_change, u_stat, p_value)`.
///
/// Rank computation runs over only the combined `(group ∪ ref)` cells —
/// matching pdex's `mwu(group_matrix, ref_data)` which does not include
/// other groups in the rank pool.
#[allow(clippy::too_many_arguments)]
fn pdex_gene_target_stats(
    values: &[f64],
    group_cells: &[usize],
    ref_cells: &[usize],
    ref_mean: f64,
    mode: crate::pseudobulk::GeomMeanMode,
    epsilon: f64,
    values_buf: &mut Vec<f64>,
    index_buf: &mut Vec<usize>,
    ranks_buf: &mut Vec<f64>,
    group_buf_indices: &mut Vec<usize>,
) -> (f64, f64, f64, f64, f64) {
    let n1 = group_cells.len();
    let n2 = ref_cells.len();

    let target_mean = if n1 == 0 {
        f64::NAN
    } else {
        let s: f64 = group_cells.iter().map(|&c| pdex_pre(mode, values[c])).sum();
        pdex_post(mode, s / n1 as f64)
    };

    // Match upstream pdex (`_math.py::log2_fold_change` / `percent_change`):
    // both define `0/0 -> 0.0` (a gene unexpressed in both groups reports "no
    // change", not `NaN`) while preserving the legitimate `±inf` of one-sided
    // zeros. With the default `epsilon == 1e-9` the denominators are strictly
    // positive so neither `NaN` nor `±inf` arises; with `epsilon == 0` a
    // reference-undetected gene (`ref_mean == 0`, `target_mean > 0`) is
    // intentionally `±inf`, and `0/0` collapses to `0.0`.
    let log2_fc = nan_to_zero(((target_mean + epsilon) / (ref_mean + epsilon)).log2());
    let percent_change = nan_to_zero((target_mean - ref_mean) / (ref_mean + epsilon));

    if n1 == 0 || n2 == 0 {
        return (target_mean, log2_fc, percent_change, f64::NAN, 1.0);
    }

    // Gather (group, ref) into a contiguous buffer; rank within that pool.
    let n_total = n1 + n2;
    values_buf.clear();
    values_buf.reserve(n_total);
    for &c in group_cells {
        values_buf.push(values[c]);
    }
    for &c in ref_cells {
        values_buf.push(values[c]);
    }

    let tc = rank_with_ties(&values_buf[..n_total], index_buf, ranks_buf);
    group_buf_indices.clear();
    group_buf_indices.extend(0..n1);
    // pdex_ref matches upstream pdex's continuity-corrected two-sided p-value.
    let (u_stat, _z, p) = wilcoxon_full_from_ranks(ranks_buf, group_buf_indices, n_total, tc, true);

    (target_mean, log2_fc, percent_change, u_stat, p)
}

/// pdex `mode="ref"` accelerator over a dense `[n_obs × n_vars]` row-major buffer.
///
/// For each non-reference group `g`, compute per (group, gene):
/// * `target_mean` and `ref_mean` in natural (count) space using `mode`.
/// * `log2((target_mean + epsilon) / (ref_mean + epsilon))`.
/// * `(target_mean - ref_mean) / (ref_mean + epsilon)`.
/// * Mann-Whitney U statistic and two-sided p-value vs the reference cells.
/// * Benjamini-Hochberg FDR across genes (per group).
///
/// Parallelised over genes. Returns rows in input `group_names` and `gene_names`
/// order — the reference group is excluded from output.
///
/// Back-compat shim for callers that do not use `cpm_filter`. Equivalent to
/// `pdex_ref_core(.., compute_cpm = false)`.
#[allow(clippy::too_many_arguments)]
pub fn pdex_ref(
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: usize,
    mode: crate::pseudobulk::GeomMeanMode,
    epsilon: f64,
) -> Result<PdexRefResult> {
    pdex_ref_core(
        data,
        n_obs,
        n_vars,
        gene_names,
        groups,
        group_names,
        reference,
        mode,
        epsilon,
        false,
    )
}

/// pdex `mode="ref"` accelerator core. When `compute_cpm` is true the kernel
/// also accumulates the per-(group, gene) and per-gene reference **arithmetic**
/// count-space means (`target_arith_gene_means` / `ref_arith_gene_means`) that
/// `apply_cpm_filter` consumes for the `cpm_filter` keep decision. This kernel
/// never applies the filter itself (the CPM denominator spans the full gene
/// axis, which a single chunk cannot see) — the caller runs `apply_cpm_filter`
/// after all chunks merge.
#[allow(clippy::too_many_arguments)]
pub fn pdex_ref_core(
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: usize,
    mode: crate::pseudobulk::GeomMeanMode,
    epsilon: f64,
    compute_cpm: bool,
) -> Result<PdexRefResult> {
    if data.len() != n_obs * n_vars {
        return Err(crate::AccelError::InvalidInput(format!(
            "data length {} != n_obs {} × n_vars {}",
            data.len(),
            n_obs,
            n_vars
        )));
    }
    if gene_names.len() != n_vars {
        return Err(crate::AccelError::InvalidInput(format!(
            "gene_names length {} != n_vars {}",
            gene_names.len(),
            n_vars
        )));
    }
    if groups.len() != n_obs {
        return Err(crate::AccelError::InvalidInput(format!(
            "groups length {} != n_obs {}",
            groups.len(),
            n_obs
        )));
    }
    let n_groups = group_names.len();
    if reference >= n_groups {
        return Err(crate::AccelError::InvalidInput(format!(
            "reference index {} out of range (n_groups = {})",
            reference, n_groups
        )));
    }
    if epsilon < 0.0 || !epsilon.is_finite() {
        return Err(crate::AccelError::InvalidInput(format!(
            "epsilon must be non-negative and finite (got {epsilon})"
        )));
    }
    // Finiteness is a contract at the DE accelerator boundary (ACC10).
    ensure_finite_de_input(data)?;

    // Bucket cell indices by group. Cells with `group >= n_groups` are
    // silently dropped (matches pdex's NaN/empty-string handling, which the
    // caller is expected to encode upstream as out-of-range group ids).
    let mut group_indices: Vec<Vec<usize>> = vec![vec![]; n_groups];
    for (i, &g) in groups.iter().enumerate() {
        if g < n_groups {
            group_indices[g].push(i);
        }
    }

    let ref_cells = group_indices[reference].clone();
    let ref_membership = ref_cells.len();
    if ref_membership == 0 {
        return Err(crate::AccelError::InvalidInput(format!(
            "reference group '{}' has zero cells",
            group_names[reference]
        )));
    }

    // Test groups in input order, reference excluded.
    let test_groups: Vec<usize> = (0..n_groups).filter(|&g| g != reference).collect();
    let n_test = test_groups.len();
    let target_memberships: Vec<usize> = test_groups
        .iter()
        .map(|&g| group_indices[g].len())
        .collect();

    // Count-space arithmetic transform used for the CPM keep decision; it is
    // mode-independent (always arithmetic) per pdex's `cpm_bulk`.
    let cpm_mode = mode.arith();

    // Per-gene parallel kernel: compute ref_mean once, then per test-group stats.
    // gene_results[var_idx] =
    //   (ref_mean, Vec<(t_mean, lfc, pct, u, p)>, ref_arith_mean, Vec<t_arith_mean>).
    // The two arithmetic vectors are empty unless `compute_cpm`.
    type PerGroupStats = (f64, f64, f64, f64, f64);
    type GeneResult = (f64, Vec<PerGroupStats>, f64, Vec<f64>);
    let gene_results: Vec<GeneResult> = (0..n_vars)
        .into_par_iter()
        .map_init(
            || {
                (
                    vec![0.0f64; n_obs], // column-value buffer
                    Vec::<f64>::with_capacity(n_obs),
                    Vec::<usize>::with_capacity(n_obs),
                    Vec::<f64>::with_capacity(n_obs),
                    Vec::<usize>::with_capacity(n_obs),
                )
            },
            |(col_buf, values_buf, index_buf, ranks_buf, group_buf_indices), var_idx| {
                for cell in 0..n_obs {
                    col_buf[cell] = data[cell * n_vars + var_idx] as f64;
                }

                // ref_mean is shared across all test groups for this gene.
                let ref_sum: f64 = ref_cells.iter().map(|&c| pdex_pre(mode, col_buf[c])).sum();
                let ref_mean = pdex_post(mode, ref_sum / ref_membership as f64);

                let per_group: Vec<PerGroupStats> = test_groups
                    .iter()
                    .map(|&g| {
                        pdex_gene_target_stats(
                            &col_buf[..n_obs],
                            &group_indices[g],
                            &ref_cells,
                            ref_mean,
                            mode,
                            epsilon,
                            values_buf,
                            index_buf,
                            ranks_buf,
                            group_buf_indices,
                        )
                    })
                    .collect();

                // Arithmetic count-space means for the CPM filter (gated).
                let (ref_arith, target_arith) = if compute_cpm {
                    let ref_arith_sum: f64 =
                        ref_cells.iter().map(|&c| cpm_mode.pre(col_buf[c])).sum();
                    let ref_arith = cpm_mode.post(ref_arith_sum / ref_membership as f64);
                    let target_arith: Vec<f64> = test_groups
                        .iter()
                        .map(|&g| {
                            let cells = &group_indices[g];
                            if cells.is_empty() {
                                f64::NAN
                            } else {
                                let s: f64 = cells.iter().map(|&c| cpm_mode.pre(col_buf[c])).sum();
                                cpm_mode.post(s / cells.len() as f64)
                            }
                        })
                        .collect();
                    (ref_arith, target_arith)
                } else {
                    (0.0, Vec::new())
                };

                (ref_mean, per_group, ref_arith, target_arith)
            },
        )
        .collect();

    // Transpose into per-group flat arrays in gene-input order.
    let mut target_means: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];
    let mut ref_means: Vec<f64> = Vec::with_capacity(n_vars);
    let mut log2_fold_changes: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];
    let mut percent_changes: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];
    let mut statistics: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];
    let mut p_values: Vec<Vec<f64>> = vec![Vec::with_capacity(n_vars); n_test];
    let mut target_arith_gene_means: Vec<Vec<f64>> =
        vec![Vec::with_capacity(if compute_cpm { n_vars } else { 0 }); n_test];
    let mut ref_arith_gene_means: Vec<f64> =
        Vec::with_capacity(if compute_cpm { n_vars } else { 0 });

    for (ref_mean, per_group, ref_arith, target_arith) in gene_results {
        ref_means.push(ref_mean);
        for (tg, s) in per_group.into_iter().enumerate() {
            target_means[tg].push(s.0);
            log2_fold_changes[tg].push(s.1);
            percent_changes[tg].push(s.2);
            statistics[tg].push(s.3);
            p_values[tg].push(s.4);
        }
        if compute_cpm {
            ref_arith_gene_means.push(ref_arith);
            for (tg, m) in target_arith.into_iter().enumerate() {
                target_arith_gene_means[tg].push(m);
            }
        }
    }

    // BH per group across genes. pdex emits p-values clipped to [0, 1] before
    // BH — mirror that to match its FDR output bit-for-bit.
    let fdrs: Vec<Vec<f64>> = p_values
        .iter()
        .map(|pv| {
            let clipped: Vec<f64> = pv.iter().map(|&p| p.clamp(0.0, 1.0)).collect();
            benjamini_hochberg(&clipped)
        })
        .collect();

    // Also clip the reported p_values to [0, 1] (pdex does this too).
    for pv in p_values.iter_mut() {
        for p in pv.iter_mut() {
            *p = p.clamp(0.0, 1.0);
        }
    }

    Ok(PdexRefResult {
        group_names: test_groups
            .iter()
            .map(|&g| group_names[g].clone())
            .collect(),
        feature_names: gene_names.to_vec(),
        target_means,
        ref_means,
        target_memberships,
        ref_membership,
        log2_fold_changes,
        percent_changes,
        statistics,
        p_values,
        fdrs,
        target_arith_gene_means,
        ref_arith_gene_means,
        kept_indices: None,
        exec_info: crate::route::AccelExecutionInfo::default(),
    })
}

/// Gene-chunked pdex `mode="ref"` over an in-memory `ScxCsr` matrix.
///
/// Materializes one gene chunk at a time into a dense `[n_obs × chunk_size]`
/// buffer and calls `pdex_ref` on the chunk. Avoids `O(n_obs × n_vars)`
/// dense expansion while preserving exact-pdex semantics. BH FDR is applied
/// per group across the full gene set after all chunks complete.
#[allow(clippy::too_many_arguments)]
pub fn pdex_ref_sparse(
    csr: &scx_sparse::ScxCsr,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: usize,
    gene_chunk_size: usize,
    mode: crate::pseudobulk::GeomMeanMode,
    epsilon: f64,
    cpm_filter: Option<f64>,
) -> Result<PdexRefResult> {
    let (n_obs, n_vars) = csr.shape;
    if gene_names.len() != n_vars {
        return Err(crate::AccelError::InvalidInput(format!(
            "gene_names length {} != n_vars {}",
            gene_names.len(),
            n_vars
        )));
    }
    if groups.len() != n_obs {
        return Err(crate::AccelError::InvalidInput(format!(
            "groups length {} != n_obs {}",
            groups.len(),
            n_obs
        )));
    }
    if gene_chunk_size == 0 {
        return Err(crate::AccelError::InvalidInput(
            "gene_chunk_size must be > 0".to_string(),
        ));
    }
    // Clamp the dense n_obs×chunk f32 workspace to the CPU memory budget.
    let gene_chunk_size =
        crate::mem_budget::de_gene_chunk_or_err(gene_chunk_size, n_obs, "pdex_ref_sparse")?;

    let mut combined: Option<PdexRefResult> = None;

    for chunk_start in (0..n_vars).step_by(gene_chunk_size) {
        let chunk_end = (chunk_start + gene_chunk_size).min(n_vars);
        let chunk_size = chunk_end - chunk_start;
        let col_indices: Vec<u32> = (chunk_start as u32..chunk_end as u32).collect();

        // Project to the chunk's columns and densify (zero-init).
        let projected = scx_engine::project_csr(csr, &col_indices);
        let mut dense = vec![0.0f32; n_obs * chunk_size];
        for row in 0..projected.n_rows() {
            let s = projected.indptr[row] as usize;
            let e = projected.indptr[row + 1] as usize;
            for j in s..e {
                let col = projected.indices[j] as usize;
                dense[row * chunk_size + col] = projected.data[j];
            }
        }

        let chunk_genes: Vec<String> = gene_names[chunk_start..chunk_end].to_vec();
        let chunk_result = pdex_ref_core(
            &dense,
            n_obs,
            chunk_size,
            &chunk_genes,
            groups,
            group_names,
            reference,
            mode,
            epsilon,
            cpm_filter.is_some(),
        )?;

        combined = Some(match combined.take() {
            None => chunk_result,
            Some(mut acc) => {
                merge_pdex_chunk_into(&mut acc, chunk_result);
                acc
            }
        });
    }

    let mut result = combined.unwrap_or_else(|| empty_pdex_result(group_names, reference));
    finalize_pdex(&mut result, cpm_filter);
    Ok(result)
}

/// Gene-chunked pdex `mode="ref"` streaming over a CSR [`ShardSource`].
///
/// Mirrors [`wilcoxon_rank_sum_streaming`], including the reason it is generic:
/// `groups` / `gene_names` are indexed by *visible* cell and gene, so a caller
/// holding a subset SCX handle must pass that handle's view
/// (`as_shard_source()`), not the reader underneath it. Walks every shard once
/// per gene chunk, so a caching source should opt in (`with_cached_reads()`)
/// and be sized to `>= n_shards`.
#[allow(clippy::too_many_arguments)]
pub fn pdex_ref_streaming<S: ShardSource>(
    source: &S,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: usize,
    gene_chunk_size: usize,
    mode: crate::pseudobulk::GeomMeanMode,
    epsilon: f64,
    cpm_filter: Option<f64>,
) -> Result<PdexRefResult> {
    let n_obs = source.n_obs();
    let n_vars = gene_names.len();
    if groups.len() != n_obs {
        return Err(crate::AccelError::InvalidInput(format!(
            "groups length {} != n_obs {}",
            groups.len(),
            n_obs
        )));
    }
    if gene_chunk_size == 0 {
        return Err(crate::AccelError::InvalidInput(
            "gene_chunk_size must be > 0".to_string(),
        ));
    }
    // `gene_names` describes the *visible* gene axis; the source must present
    // that same axis. Without this a caller that handed over the file instead
    // of a subset handle's view is not caught: the chunk loop slices columns
    // `0..k` and `project_csr` silently drops anything out of range, so the
    // result comes back fully formed and describing the wrong genes. Mirrors
    // the check `pseudobulk_aggregate` already makes. See
    // `wilcoxon_rank_sum_streaming`.
    if n_vars != source.n_vars() {
        return Err(crate::AccelError::ShapeError(format!(
            "pdex_ref_streaming: gene_names length {} != source.n_vars() {} — the source must \
             be the same gene axis the names describe (pass the handle's view, \
             e.g. `as_shard_source()`, not the raw reader)",
            n_vars,
            source.n_vars(),
        )));
    }

    // Clamp the dense n_obs×chunk f32 workspace to the CPU memory budget.
    let gene_chunk_size =
        crate::mem_budget::de_gene_chunk_or_err(gene_chunk_size, n_obs, "pdex_ref_streaming")?;

    let n_shards = source.n_shards();
    // `None` = a non-caching source (every `read_shard_arc` re-decodes by
    // design); there is no cache to size, so there is nothing to warn about.
    let n_chunks = n_vars.div_ceil(gene_chunk_size);
    if let Some(cache_cap) = source.shard_cache_capacity() {
        if n_chunks > 1 && cache_cap < n_shards {
            log::warn!(
                "pdex_ref_streaming: cache_shards={} < n_shards={} with {} gene chunks — \
                 the cached read path will evict and re-decode every shard on each chunk. \
                 Size the shard cache to >= n_shards for the documented speedup.",
                cache_cap,
                n_shards,
                n_chunks,
            );
        }
    }

    let mut combined: Option<PdexRefResult> = None;

    for chunk_start in (0..n_vars).step_by(gene_chunk_size) {
        let chunk_end = (chunk_start + gene_chunk_size).min(n_vars);
        let chunk_size = chunk_end - chunk_start;
        let col_indices: Vec<u32> = (chunk_start as u32..chunk_end as u32).collect();

        let mut dense = vec![0.0f32; n_obs * chunk_size];
        let mut global_row = 0usize;
        for shard_idx in 0..n_shards {
            let shard_csr = source
                .read_shard_arc(shard_idx)
                .map_err(crate::AccelError::Scx)?;
            let _r = scx_format_io::reduction_guard();
            let projected = scx_engine::project_csr(&shard_csr, &col_indices);
            for row in 0..projected.n_rows() {
                let s = projected.indptr[row] as usize;
                let e = projected.indptr[row + 1] as usize;
                for j in s..e {
                    let col = projected.indices[j] as usize;
                    dense[(global_row + row) * chunk_size + col] = projected.data[j];
                }
            }
            global_row += projected.n_rows();
        }

        // MWU compute dominates and runs post-decode → `reduction`, disjoint.
        let _rank = scx_format_io::reduction_guard();
        let chunk_genes: Vec<String> = gene_names[chunk_start..chunk_end].to_vec();
        let chunk_result = pdex_ref_core(
            &dense,
            n_obs,
            chunk_size,
            &chunk_genes,
            groups,
            group_names,
            reference,
            mode,
            epsilon,
            cpm_filter.is_some(),
        )?;

        combined = Some(match combined.take() {
            None => chunk_result,
            Some(mut acc) => {
                merge_pdex_chunk_into(&mut acc, chunk_result);
                acc
            }
        });
    }

    let mut result = combined.unwrap_or_else(|| empty_pdex_result(group_names, reference));
    finalize_pdex(&mut result, cpm_filter);
    Ok(result)
}

pub(crate) fn empty_pdex_result(group_names: &[String], reference: usize) -> PdexRefResult {
    let group_names_out: Vec<String> = (0..group_names.len())
        .filter(|&g| g != reference)
        .map(|g| group_names[g].clone())
        .collect();
    let n_test = group_names_out.len();
    PdexRefResult {
        group_names: group_names_out,
        feature_names: vec![],
        target_means: vec![vec![]; n_test],
        ref_means: vec![],
        target_memberships: vec![0; n_test],
        ref_membership: 0,
        log2_fold_changes: vec![vec![]; n_test],
        percent_changes: vec![vec![]; n_test],
        statistics: vec![vec![]; n_test],
        p_values: vec![vec![]; n_test],
        fdrs: vec![vec![]; n_test],
        target_arith_gene_means: vec![vec![]; n_test],
        ref_arith_gene_means: vec![],
        kept_indices: None,
        exec_info: crate::route::AccelExecutionInfo::default(),
    }
}

pub(crate) fn merge_pdex_chunk_into(acc: &mut PdexRefResult, chunk: PdexRefResult) {
    // The first chunk already initialised the membership counts.
    acc.feature_names.extend(chunk.feature_names);
    acc.ref_means.extend(chunk.ref_means);
    acc.ref_arith_gene_means.extend(chunk.ref_arith_gene_means);
    // Internal pdex chunk-merge invariant (both buffers are produced by this
    // crate with one entry per target group), not a decode/scatter/rank guard
    // on untrusted input.
    // debug-assert-ok: internal invariant, not an untrusted-input boundary.
    debug_assert_eq!(acc.target_means.len(), chunk.target_means.len());
    for tg in 0..acc.target_means.len() {
        acc.target_means[tg].extend(&chunk.target_means[tg]);
        acc.log2_fold_changes[tg].extend(&chunk.log2_fold_changes[tg]);
        acc.percent_changes[tg].extend(&chunk.percent_changes[tg]);
        acc.statistics[tg].extend(&chunk.statistics[tg]);
        acc.p_values[tg].extend(&chunk.p_values[tg]);
        acc.target_arith_gene_means[tg].extend(&chunk.target_arith_gene_means[tg]);
    }
    // Discard chunk.fdrs — we recompute globally after all chunks merge.
}

pub(crate) fn recompute_pdex_fdrs(result: &mut PdexRefResult) {
    result.fdrs = result
        .p_values
        .iter()
        .map(|pv| benjamini_hochberg(pv))
        .collect();
}

/// Finalize a (possibly chunk-merged) pdex result: apply the `cpm_filter` keep
/// mask + survivor-scoped FDR when set, otherwise just recompute FDR over the
/// full gene axis. The single place every route (dense, CSR, streaming, CSC,
/// GPU) converges on after all gene chunks are merged.
pub fn finalize_pdex(result: &mut PdexRefResult, cpm_filter: Option<f64>) {
    match cpm_filter {
        Some(threshold) => apply_cpm_filter(result, threshold),
        None => recompute_pdex_fdrs(result),
    }
}

/// Per-gene pooled counts-per-million from arithmetic count-space gene means:
/// `mean / Σ(means) * 1e6`, with the `pdex` zero-total guard (`denom -> 1.0`).
fn cpm_from_arith_means(means: &[f64]) -> Vec<f64> {
    let total: f64 = means.iter().sum();
    let denom = if total != 0.0 { total } else { 1.0 };
    means.iter().map(|&m| m / denom * 1e6).collect()
}

/// Apply pdex's `cpm_filter` to a fully-merged result: per test group, keep a
/// gene iff `target_cpm > threshold OR ref_cpm > threshold` (strict `>`), drop
/// every other gene from that group's result vectors, and recompute
/// Benjamini-Hochberg FDR over the surviving genes only. The surviving gene
/// indices (into the shared `feature_names` / `ref_means` axis) are recorded in
/// `kept_indices` so the DataFrame builder can recover each survivor's identity.
///
/// Requires `compute_cpm = true` to have populated the arithmetic-mean fields.
pub(crate) fn apply_cpm_filter(result: &mut PdexRefResult, threshold: f64) {
    let n_test = result.target_means.len();
    let ref_cpm = cpm_from_arith_means(&result.ref_arith_gene_means);

    let mut kept_indices: Vec<Vec<usize>> = Vec::with_capacity(n_test);
    for tg in 0..n_test {
        let target_cpm = cpm_from_arith_means(&result.target_arith_gene_means[tg]);
        let kept: Vec<usize> = (0..target_cpm.len())
            .filter(|&gi| target_cpm[gi] > threshold || ref_cpm[gi] > threshold)
            .collect();

        let take = |src: &[f64]| -> Vec<f64> { kept.iter().map(|&i| src[i]).collect() };
        result.target_means[tg] = take(&result.target_means[tg]);
        result.log2_fold_changes[tg] = take(&result.log2_fold_changes[tg]);
        result.percent_changes[tg] = take(&result.percent_changes[tg]);
        result.statistics[tg] = take(&result.statistics[tg]);
        result.p_values[tg] = take(&result.p_values[tg]);
        kept_indices.push(kept);
    }

    result.kept_indices = Some(kept_indices);
    // The arithmetic-mean buffers fed only the keep decision and are never
    // reported; drop them so the returned result doesn't carry full-axis stale
    // data (notably for large gene counts).
    result.target_arith_gene_means = Vec::new();
    result.ref_arith_gene_means = Vec::new();
    // FDR over the survivor universe (matches pdex's post-filter
    // `false_discovery_control`).
    recompute_pdex_fdrs(result);
}

#[cfg(test)]
#[path = "cpu_tests.rs"]
mod tests;
