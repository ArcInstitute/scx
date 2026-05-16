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
    /// Z-scores (signed). `[n_groups][n_genes]`
    pub scores: Vec<Vec<f64>>,
    /// Raw p-values (two-sided). `[n_groups][n_genes]`
    pub pvals: Vec<Vec<f64>>,
    /// BH-adjusted p-values. `[n_groups][n_genes]`
    pub pvals_adj: Vec<Vec<f64>>,
    /// log2 fold-changes (group mean / reference mean). `[n_groups][n_genes]`
    pub logfoldchanges: Vec<Vec<f64>>,
}

/// Per-gene test result (before sorting/grouping).
#[derive(Debug, Clone)]
struct GeneTestResult {
    gene_idx: usize,
    score: f64, // z-statistic (signed)
    pval: f64,  // two-sided p-value
    logfc: f64, // log2 fold-change
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
/// * `group_names` — Unique group names, length `n_groups`.
/// * `reference` — If `Some(idx)`, compare every other group against group `idx`.
///   If `None`, 1-vs-rest.
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

    // Pre-compute cell indices per group.
    let mut group_indices: Vec<Vec<usize>> = vec![vec![]; n_groups];
    for (i, &g) in groups.iter().enumerate() {
        if g < n_groups {
            group_indices[g].push(i);
        }
    }

    // Determine which groups to test and what they compare against.
    let test_groups: Vec<usize> = match reference {
        Some(ref_idx) => (0..n_groups).filter(|&g| g != ref_idx).collect(),
        None => (0..n_groups).collect(),
    };

    // Pre-compute group sums per gene for logFC (avoids redundant gathering).
    // group_gene_sums[g][var] = sum of values for cells in group g at gene var.
    let mut group_gene_sums: Vec<Vec<f64>> = vec![vec![0.0; n_vars]; n_groups];
    for g in 0..n_groups {
        for &cell in &group_indices[g] {
            let base = cell * n_vars;
            for var in 0..n_vars {
                group_gene_sums[g][var] += data[base + var] as f64;
            }
        }
    }

    let n_test_groups = test_groups.len();

    // --- Pre-rank approach: rank once per gene, then derive per-group statistics ---
    // For 1-vs-rest: rank all n_obs values once per gene (10× fewer sorts).
    // For pairwise: rank (group + ref) cells per test group per gene.
    let gene_group_results: Vec<Vec<(f64, f64, f64)>> = (0..n_vars)
        .into_par_iter()
        .map_init(
            || {
                // Thread-local buffers reused across genes (no per-gene allocation).
                (
                    vec![0.0f64; n_obs],
                    Vec::with_capacity(n_obs),
                    Vec::with_capacity(n_obs),
                )
            },
            |(values_buf, index_buf, ranks_buf), var_idx| {
                let mut group_results = Vec::with_capacity(n_test_groups);

                match reference {
                    None => {
                        // 1-vs-rest: rank all n_obs values once, reuse across groups.
                        for i in 0..n_obs {
                            values_buf[i] = data[i * n_vars + var_idx] as f64;
                        }
                        let raw_tc = rank_with_ties(&values_buf[..n_obs], index_buf, ranks_buf);
                        let tc = if tie_correct { raw_tc } else { 0.0 };

                        for &g in &test_groups {
                            let n1 = group_indices[g].len();
                            if n1 == 0 || n1 == n_obs {
                                group_results.push((f64::NAN, 1.0, f64::NAN));
                                continue;
                            }
                            let n2 = n_obs - n1;

                            let (score, pval) =
                                wilcoxon_from_ranks(ranks_buf, &group_indices[g], n_obs, tc);

                            let mean_group = group_gene_sums[g][var_idx] / n1 as f64;
                            let rest_sum: f64 = (0..n_groups)
                                .filter(|&gg| gg != g)
                                .map(|gg| group_gene_sums[gg][var_idx])
                                .sum();
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

                            let mean_group = group_gene_sums[g][var_idx] / n1 as f64;
                            let mean_ref = group_gene_sums[ref_idx][var_idx] / n2 as f64;
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
                    score,
                    pval,
                    logfc,
                }
            })
            .collect();

        sorted.sort_by(|a, b| {
            let a_key = if rankby_abs { a.score.abs() } else { a.score };
            let b_key = if rankby_abs { b.score.abs() } else { b.score };
            b_key
                .partial_cmp(&a_key)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.gene_idx.cmp(&b.gene_idx))
        });

        let names: Vec<String> = sorted
            .iter()
            .map(|r| gene_names[r.gene_idx].clone())
            .collect();
        let scores: Vec<f64> = sorted.iter().map(|r| r.score).collect();
        let pvals: Vec<f64> = sorted.iter().map(|r| r.pval).collect();
        let logfc: Vec<f64> = sorted.iter().map(|r| r.logfc).collect();

        let pvals_adj = benjamini_hochberg(&pvals);

        result_names.push(names);
        result_scores.push(scores);
        result_pvals.push(pvals);
        result_pvals_adj.push(pvals_adj);
        result_logfc.push(logfc);
        result_group_names.push(group_names[g].clone());
    }

    Ok(DiffExpResult {
        group_names: result_group_names,
        names: result_names,
        scores: result_scores,
        pvals: result_pvals,
        pvals_adj: result_pvals_adj,
        logfoldchanges: result_logfc,
    })
}

/// Compute log2 fold-change between group and reference means.
fn compute_logfc(mean_group: f64, mean_ref: f64, log_transformed: bool) -> f64 {
    if log_transformed {
        let expm1_group = mean_group.exp_m1();
        let expm1_ref = mean_ref.exp_m1();
        ((expm1_group + LOGFC_PSEUDOCOUNT) / (expm1_ref + LOGFC_PSEUDOCOUNT)).log2()
    } else {
        (mean_group + LOGFC_PSEUDOCOUNT).log2() - (mean_ref + LOGFC_PSEUDOCOUNT).log2()
    }
}

/// Rank values with mid-rank tie handling. Returns `(ranks, tie_correction)`.
///
/// `ranks[i]` is the 1-based mid-rank for `values[i]`.
/// `tie_correction` is `Σ (t³ - t)` over tie groups, used in the variance
/// formula for the Wilcoxon test. Shared across all group comparisons for
/// the same gene, since ties are a property of the value distribution.
fn rank_with_ties(values: &[f64], index_buf: &mut Vec<usize>, ranks: &mut Vec<f64>) -> f64 {
    // NaN values would be silently ordered as `Equal` by the fallback below,
    // producing a meaningless rank and a garbage p-value downstream. In debug
    // builds, assert that the caller pre-sanitised the input; in release,
    // upstream filters (QC) should have removed NaNs before we get here —
    // if one slips through we still produce a deterministic (if wrong)
    // result rather than panicking.
    debug_assert!(
        !values.iter().any(|v| v.is_nan()),
        "rank_with_ties received NaN input — filter NaNs before ranking"
    );
    let n = values.len();
    index_buf.clear();
    index_buf.extend(0..n);
    index_buf.sort_unstable_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    ranks.resize(n, 0.0);
    let mut tie_correction = 0.0f64;
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && values[index_buf[j]] == values[index_buf[i]] {
            j += 1;
        }
        let mid_rank = (i as f64 + 1.0 + j as f64) / 2.0;
        let tie_size = (j - i) as f64;
        for idx in &index_buf[i..j] {
            ranks[*idx] = mid_rank;
        }
        if tie_size > 1.0 {
            tie_correction += tie_size * tie_size * tie_size - tie_size;
        }
        i = j;
    }
    tie_correction
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
    let (_u, z, p) = wilcoxon_full_from_ranks(ranks, group_cells, n_total, tie_correction);
    (z, p)
}

/// Full Wilcoxon stats from pre-computed ranks: `(u1, z, two_sided_p)`.
///
/// `u1` is the Mann-Whitney U statistic for the test group (the convention
/// matched by `pdex` / `numba_mwu`'s `.statistic`).  `z` is the
/// tie-corrected signed z-score used by the existing scanpy-parity path.
/// All callers within this crate use one or the other; the unified helper
/// avoids recomputing the rank sum twice when both are needed.
fn wilcoxon_full_from_ranks(
    ranks: &[f64],
    group_cells: &[usize],
    n_total: usize,
    tie_correction: f64,
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

    let z = (u1 - mu) / sigma_sq.sqrt();
    let p = 2.0 * normal_sf(z.abs());
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

    // Assign mid-ranks and compute rank sum for group, track tie info.
    let total = combined.len();
    let mut rank_sum_group: f64 = 0.0;
    let mut tie_correction: f64 = 0.0;
    let mut i = 0;

    while i < total {
        // Find extent of this tie group.
        let mut j = i + 1;
        while j < total && combined[j].0 == combined[i].0 {
            j += 1;
        }
        let tie_size = (j - i) as f64;
        // Mid-rank: average of ranks (1-indexed).
        let mid_rank = (i as f64 + 1.0 + j as f64) / 2.0;

        // Add to group rank sum, compute tie correction.
        for item in combined.iter().take(j).skip(i) {
            if item.1 == 0 {
                rank_sum_group += mid_rank;
            }
        }
        if tie_size > 1.0 {
            tie_correction += tie_size * tie_size * tie_size - tie_size;
        }

        i = j;
    }

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
fn normal_sf(z: f64) -> f64 {
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

/// Gene-chunked streaming Wilcoxon rank-sum from `BackedCsrReader`.
///
/// Instead of materializing the full matrix, processes genes in chunks:
/// 1. For each gene chunk, iterate all shards via `read_shard_cached_arc()`,
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
pub fn wilcoxon_rank_sum_streaming(
    reader: &scx_format::backed::BackedCsrReader,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: Option<usize>,
    gene_chunk_size: usize,
    log_transformed: bool,
    rankby_abs: bool,
    tie_correct: bool,
) -> Result<DiffExpResult> {
    let n_obs = reader.n_obs();
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

    let n_shards = reader.index().n_shards();

    // Cache-sizing footgun guard. The kernel walks every shard once per
    // gene chunk; if the LRU can't hold all `n_shards` decoded shards
    // simultaneously, the iteration order `0..n_shards` evicts the
    // shard the next chunk re-requests *first*, so the cached path is
    // strictly slower than `read_shard_uncached` (LRU bookkeeping +
    // re-decode). Warn once per call so the caller sees it without
    // spamming per-shard.
    let cache_cap = reader.cache_capacity();
    let n_chunks = n_vars.div_ceil(gene_chunk_size);
    if n_chunks > 1 && cache_cap < n_shards {
        log::warn!(
            "wilcoxon_rank_sum_streaming: cache_shards={} < n_shards={} with {} gene chunks — \
             the cached read path will evict and re-decode every shard on each chunk. \
             Size the BackedCsrReader cache to >= n_shards for the documented speedup.",
            cache_cap,
            n_shards,
            n_chunks,
        );
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
            let shard_csr = reader
                .read_shard_cached_arc(shard_idx)
                .map_err(crate::AccelError::Scx)?;
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

        // Run existing wilcoxon_rank_sum on this chunk's dense buffer.
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
            scores: vec![],
            pvals: vec![],
            pvals_adj: vec![],
            logfoldchanges: vec![],
        });
    }
    if chunks.len() == 1 {
        return Ok(chunks.into_iter().next().unwrap());
    }

    // All chunks must have the same group structure.
    let group_names = chunks[0].group_names.clone();
    let n_groups = group_names.len();

    let mut merged_names = Vec::with_capacity(n_groups);
    let mut merged_scores = Vec::with_capacity(n_groups);
    let mut merged_pvals = Vec::with_capacity(n_groups);
    let mut merged_pvals_adj = Vec::with_capacity(n_groups);
    let mut merged_logfc = Vec::with_capacity(n_groups);

    for group_name in &group_names {
        // Collect all genes for this group across chunks.
        let mut gene_entries: Vec<(String, f64, f64, f64)> = Vec::new();

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
                ));
            }
        }

        // Sort by signed score descending (default) or |score| descending (rankby_abs).
        gene_entries.sort_by(|a, b| {
            let a_key = if rankby_abs { a.1.abs() } else { a.1 };
            let b_key = if rankby_abs { b.1.abs() } else { b.1 };
            b_key
                .partial_cmp(&a_key)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let names: Vec<String> = gene_entries.iter().map(|e| e.0.clone()).collect();
        let scores: Vec<f64> = gene_entries.iter().map(|e| e.1).collect();
        let pvals: Vec<f64> = gene_entries.iter().map(|e| e.2).collect();
        let logfc: Vec<f64> = gene_entries.iter().map(|e| e.3).collect();

        // Global BH correction across ALL genes.
        let pvals_adj = benjamini_hochberg(&pvals);

        merged_names.push(names);
        merged_scores.push(scores);
        merged_pvals.push(pvals);
        merged_pvals_adj.push(pvals_adj);
        merged_logfc.push(logfc);
    }

    Ok(DiffExpResult {
        group_names,
        names: merged_names,
        scores: merged_scores,
        pvals: merged_pvals,
        pvals_adj: merged_pvals_adj,
        logfoldchanges: merged_logfc,
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
    /// `log2((target_mean + epsilon) / (ref_mean + epsilon))`.
    pub log2_fold_changes: Vec<Vec<f64>>,
    /// `(target_mean - ref_mean) / (ref_mean + epsilon)`.
    pub percent_changes: Vec<Vec<f64>>,
    /// Mann-Whitney U statistic for the test group vs the reference.
    pub statistics: Vec<Vec<f64>>,
    /// Two-sided MWU p-values.
    pub p_values: Vec<Vec<f64>>,
    /// Benjamini-Hochberg adjusted p-values per group across genes.
    pub fdrs: Vec<Vec<f64>>,
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

    let log2_fc = ((target_mean + epsilon) / (ref_mean + epsilon)).log2();
    let percent_change = (target_mean - ref_mean) / (ref_mean + epsilon);

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
    let (u_stat, _z, p) = wilcoxon_full_from_ranks(ranks_buf, group_buf_indices, n_total, tc);

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

    // Per-gene parallel kernel: compute ref_mean once, then per test-group stats.
    // gene_results[var_idx] = (ref_mean, Vec<(t_mean, lfc, pct, u, p)> of len n_test).
    type PerGroupStats = (f64, f64, f64, f64, f64);
    let gene_results: Vec<(f64, Vec<PerGroupStats>)> = (0..n_vars)
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

                (ref_mean, per_group)
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

    for (ref_mean, per_group) in gene_results {
        ref_means.push(ref_mean);
        for (tg, s) in per_group.into_iter().enumerate() {
            target_means[tg].push(s.0);
            log2_fold_changes[tg].push(s.1);
            percent_changes[tg].push(s.2);
            statistics[tg].push(s.3);
            p_values[tg].push(s.4);
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
        let chunk_result = pdex_ref(
            &dense,
            n_obs,
            chunk_size,
            &chunk_genes,
            groups,
            group_names,
            reference,
            mode,
            epsilon,
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
    recompute_pdex_fdrs(&mut result);
    Ok(result)
}

/// Gene-chunked pdex `mode="ref"` streaming from `BackedCsrReader`.
///
/// Mirrors `wilcoxon_rank_sum_streaming`: walks every shard once per gene
/// chunk through the cached shard API. Size the reader's cache to
/// `>= n_shards` to keep the inner loop cache-resident across chunks.
#[allow(clippy::too_many_arguments)]
pub fn pdex_ref_streaming(
    reader: &scx_format::backed::BackedCsrReader,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: usize,
    gene_chunk_size: usize,
    mode: crate::pseudobulk::GeomMeanMode,
    epsilon: f64,
) -> Result<PdexRefResult> {
    let n_obs = reader.n_obs();
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

    let n_shards = reader.index().n_shards();
    let cache_cap = reader.cache_capacity();
    let n_chunks = n_vars.div_ceil(gene_chunk_size);
    if n_chunks > 1 && cache_cap < n_shards {
        log::warn!(
            "pdex_ref_streaming: cache_shards={} < n_shards={} with {} gene chunks — \
             the cached read path will evict and re-decode every shard on each chunk. \
             Size the BackedCsrReader cache to >= n_shards for the documented speedup.",
            cache_cap,
            n_shards,
            n_chunks,
        );
    }

    let mut combined: Option<PdexRefResult> = None;

    for chunk_start in (0..n_vars).step_by(gene_chunk_size) {
        let chunk_end = (chunk_start + gene_chunk_size).min(n_vars);
        let chunk_size = chunk_end - chunk_start;
        let col_indices: Vec<u32> = (chunk_start as u32..chunk_end as u32).collect();

        let mut dense = vec![0.0f32; n_obs * chunk_size];
        let mut global_row = 0usize;
        for shard_idx in 0..n_shards {
            let shard_csr = reader
                .read_shard_cached_arc(shard_idx)
                .map_err(crate::AccelError::Scx)?;
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

        let chunk_genes: Vec<String> = gene_names[chunk_start..chunk_end].to_vec();
        let chunk_result = pdex_ref(
            &dense,
            n_obs,
            chunk_size,
            &chunk_genes,
            groups,
            group_names,
            reference,
            mode,
            epsilon,
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
    recompute_pdex_fdrs(&mut result);
    Ok(result)
}

fn empty_pdex_result(group_names: &[String], reference: usize) -> PdexRefResult {
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
    }
}

fn merge_pdex_chunk_into(acc: &mut PdexRefResult, chunk: PdexRefResult) {
    // The first chunk already initialised the membership counts.
    acc.feature_names.extend(chunk.feature_names);
    acc.ref_means.extend(chunk.ref_means);
    debug_assert_eq!(acc.target_means.len(), chunk.target_means.len());
    for tg in 0..acc.target_means.len() {
        acc.target_means[tg].extend(&chunk.target_means[tg]);
        acc.log2_fold_changes[tg].extend(&chunk.log2_fold_changes[tg]);
        acc.percent_changes[tg].extend(&chunk.percent_changes[tg]);
        acc.statistics[tg].extend(&chunk.statistics[tg]);
        acc.p_values[tg].extend(&chunk.p_values[tg]);
    }
    // Discard chunk.fdrs — we recompute globally after all chunks merge.
}

fn recompute_pdex_fdrs(result: &mut PdexRefResult) {
    result.fdrs = result
        .p_values
        .iter()
        .map(|pv| benjamini_hochberg(pv))
        .collect();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_cdf_symmetry() {
        let cdf_0 = normal_cdf(0.0);
        assert!((cdf_0 - 0.5).abs() < 1e-6, "Φ(0) should be 0.5");

        let cdf_pos = normal_cdf(1.96);
        assert!((cdf_pos - 0.975).abs() < 0.001, "Φ(1.96) ≈ 0.975");

        let cdf_neg = normal_cdf(-1.96);
        assert!((cdf_neg - 0.025).abs() < 0.001, "Φ(-1.96) ≈ 0.025");
    }

    #[test]
    fn test_bh_correction() {
        let pvals = vec![0.01, 0.04, 0.03, 0.005, 0.5];
        let adj = benjamini_hochberg(&pvals);

        // All adjusted ≥ raw
        for (raw, a) in pvals.iter().zip(adj.iter()) {
            assert!(*a >= *raw - 1e-12, "adjusted {a} should be >= raw {raw}");
        }
        // All adjusted ≤ 1
        for a in &adj {
            assert!(*a <= 1.0 + 1e-12, "adjusted {a} should be <= 1.0");
        }
    }

    #[test]
    fn test_bh_monotonicity() {
        // Sorted p-values should yield non-decreasing adjusted p-values.
        let pvals = vec![0.001, 0.01, 0.05, 0.1, 0.5];
        let adj = benjamini_hochberg(&pvals);
        for i in 1..adj.len() {
            assert!(
                adj[i] >= adj[i - 1] - 1e-12,
                "BH adjusted p-values should be monotonically non-decreasing for sorted input"
            );
        }
    }

    #[test]
    fn test_wilcoxon_identical_groups() {
        // Two identical groups should give z ≈ 0, p ≈ 1.
        let group = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rest = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (z, p) = wilcoxon_test(&group, &rest);
        assert!(
            z.abs() < 1e-6,
            "z should be ≈ 0 for identical groups, got {z}"
        );
        assert!(p > 0.9, "p should be ≈ 1 for identical groups, got {p}");
    }

    #[test]
    fn test_wilcoxon_separated_groups() {
        // Fully separated groups should give a very significant p-value.
        let group = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
        let rest = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let (z, p) = wilcoxon_test(&group, &rest);
        assert!(z > 0.0, "z should be positive (group has higher values)");
        assert!(
            p < 0.001,
            "p should be very small for separated groups, got {p}"
        );
    }

    #[test]
    fn test_wilcoxon_rank_sum_basic() {
        // Simple 2-group test: genes with known differential expression.
        let n_obs = 20;
        let n_vars = 3;

        // Gene 0: highly expressed in group 0, low in group 1
        // Gene 1: similar expression in both groups
        // Gene 2: highly expressed in group 1, low in group 0
        let mut data = vec![0.0f32; n_obs * n_vars];
        let groups: Vec<usize> = (0..n_obs).map(|i| if i < 10 { 0 } else { 1 }).collect();

        for i in 0..10 {
            data[i * n_vars + 0] = 10.0 + i as f32; // group 0, gene 0: high
            data[i * n_vars + 1] = 5.0; // group 0, gene 1: medium
            data[i * n_vars + 2] = 1.0; // group 0, gene 2: low
        }
        for i in 10..20 {
            data[i * n_vars + 0] = 1.0; // group 1, gene 0: low
            data[i * n_vars + 1] = 5.0; // group 1, gene 1: medium
            data[i * n_vars + 2] = 10.0 + (i - 10) as f32; // group 1, gene 2: high
        }

        let gene_names: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
        let group_names = vec!["A".to_string(), "B".to_string()];

        let result = wilcoxon_rank_sum(
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            None,
            false,
            true,  // rankby_abs=true to test absolute sort (original test expectation)
            false, // tie_correct=false (match scanpy default)
        )
        .unwrap();

        assert_eq!(result.group_names.len(), 2);
        assert_eq!(result.names[0].len(), n_vars);
        assert_eq!(result.scores[0].len(), n_vars);
        assert_eq!(result.pvals[0].len(), n_vars);
        assert_eq!(result.pvals_adj[0].len(), n_vars);
        assert_eq!(result.logfoldchanges[0].len(), n_vars);

        // For group A: gene_0 and gene_2 should be the top 2 DE genes (sorted by |z|).
        // gene_0 is upregulated in A, gene_2 is downregulated — both have large |z|.
        let top2_a: Vec<&str> = result.names[0][..2].iter().map(|s| s.as_str()).collect();
        assert!(
            top2_a.contains(&"gene_0"),
            "gene_0 should be top-2 DE for group A"
        );
        assert!(
            top2_a.contains(&"gene_2"),
            "gene_2 should be top-2 DE for group A"
        );

        // gene_0 should have positive logFC for group A (upregulated).
        let gene0_idx_a = result.names[0].iter().position(|n| n == "gene_0").unwrap();
        assert!(
            result.logfoldchanges[0][gene0_idx_a] > 0.0,
            "gene_0 should have positive logFC for group A"
        );
        assert!(
            result.scores[0][gene0_idx_a] > 0.0,
            "gene_0 should have positive z for group A"
        );

        // For group B: gene_2 should have positive z and logFC.
        let gene2_idx_b = result.names[1].iter().position(|n| n == "gene_2").unwrap();
        assert!(gene2_idx_b < 2, "gene_2 should be top-2 DE for group B");
        assert!(
            result.scores[1][gene2_idx_b] > 0.0,
            "gene_2 should have positive z for group B"
        );
        assert!(
            result.logfoldchanges[1][gene2_idx_b] > 0.0,
            "gene_2 should have positive logFC for group B"
        );
    }

    #[test]
    fn test_pairwise_reference() {
        let n_obs = 30;
        let n_vars = 2;
        let data = vec![5.0f32; n_obs * n_vars];
        let groups: Vec<usize> = (0..n_obs)
            .map(|i| {
                if i < 10 {
                    0
                } else if i < 20 {
                    1
                } else {
                    2
                }
            })
            .collect();
        let gene_names = vec!["g0".to_string(), "g1".to_string()];
        let group_names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        // With reference = Some(0) (A), only B and C should appear in results.
        let result = wilcoxon_rank_sum(
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            Some(0),
            false,
            false,
            false,
        )
        .unwrap();

        assert_eq!(result.group_names.len(), 2);
        assert!(result.group_names.contains(&"B".to_string()));
        assert!(result.group_names.contains(&"C".to_string()));
        assert!(!result.group_names.contains(&"A".to_string()));
    }

    #[test]
    fn test_logfc_log_transformed() {
        // When data is log1p-transformed, log_transformed=true should apply
        // expm1 before computing the fold-change ratio, matching scanpy.
        let n_obs = 20;
        let n_vars = 2;
        // Group 0: gene 0 has high raw expression (10.0), gene 1 low (1.0)
        // Group 1: gene 0 has low raw expression (1.0), gene 1 high (10.0)
        // We log1p-transform the values before passing to the function.
        let mut data = vec![0.0f32; n_obs * n_vars];
        let groups: Vec<usize> = (0..n_obs).map(|i| if i < 10 { 0 } else { 1 }).collect();

        for i in 0..10 {
            data[i * n_vars + 0] = (10.0f32 + 1.0).ln(); // ln(11) ≈ 2.397
            data[i * n_vars + 1] = (1.0f32 + 1.0).ln(); // ln(2) ≈ 0.693
        }
        for i in 10..20 {
            data[i * n_vars + 0] = (1.0f32 + 1.0).ln(); // ln(2) ≈ 0.693
            data[i * n_vars + 1] = (10.0f32 + 1.0).ln(); // ln(11) ≈ 2.397
        }

        let gene_names = vec!["g0".to_string(), "g1".to_string()];
        let group_names = vec!["A".to_string(), "B".to_string()];

        // With log_transformed=true, logFC should be:
        // log2((expm1(mean_group) + 1e-9) / (expm1(mean_ref) + 1e-9))
        // For group A, gene 0: expm1(ln(11)) = 10.0, expm1(ln(2)) = 1.0
        // → log2((10 + 1e-9) / (1 + 1e-9)) ≈ log2(10) ≈ 3.32
        let result_log = wilcoxon_rank_sum(
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            None,
            true,
            false,
            false,
        )
        .unwrap();

        // With log_transformed=false, logFC operates on the ln-scale values directly.
        let result_raw = wilcoxon_rank_sum(
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            None,
            false,
            false,
            false,
        )
        .unwrap();

        // Find gene g0 in group A.
        let g0_idx_log = result_log.names[0].iter().position(|n| n == "g0").unwrap();
        let g0_idx_raw = result_raw.names[0].iter().position(|n| n == "g0").unwrap();

        let logfc_log = result_log.logfoldchanges[0][g0_idx_log];
        let logfc_raw = result_raw.logfoldchanges[0][g0_idx_raw];

        // The expm1-based logFC should be ≈ log2(10) ≈ 3.32.
        assert!(
            (logfc_log - 10.0f64.log2()).abs() < 0.01,
            "expected logFC ≈ {:.4}, got {:.4}",
            10.0f64.log2(),
            logfc_log
        );

        // The two should differ significantly (raw logFC is on the ln-scale).
        assert!(
            (logfc_log - logfc_raw).abs() > 0.1,
            "log_transformed and raw logFC should differ, got log={logfc_log:.4} raw={logfc_raw:.4}"
        );
    }

    #[test]
    fn test_merge_empty() {
        let merged = merge_diff_exp_results(vec![], false).unwrap();
        assert!(merged.group_names.is_empty());
        assert!(merged.names.is_empty());
    }

    #[test]
    fn test_merge_single_chunk() {
        // Single chunk should pass through unchanged (except BH is unchanged).
        let chunk = DiffExpResult {
            group_names: vec!["A".to_string()],
            names: vec![vec!["g1".to_string(), "g2".to_string()]],
            scores: vec![vec![3.0, 1.0]],
            pvals: vec![vec![0.001, 0.05]],
            pvals_adj: vec![vec![0.002, 0.05]],
            logfoldchanges: vec![vec![2.0, 0.5]],
        };
        let merged = merge_diff_exp_results(vec![chunk.clone()], false).unwrap();
        assert_eq!(merged.group_names, chunk.group_names);
        assert_eq!(merged.names, chunk.names);
        assert_eq!(merged.scores, chunk.scores);
    }

    #[test]
    fn test_merge_two_chunks_resorts() {
        // Two chunks: chunk1 has gene_a (z=1.0), chunk2 has gene_b (z=3.0).
        // After merge, gene_b should come first (higher |z|).
        let chunk1 = DiffExpResult {
            group_names: vec!["G".to_string()],
            names: vec![vec!["gene_a".to_string()]],
            scores: vec![vec![1.0]],
            pvals: vec![vec![0.3]],
            pvals_adj: vec![vec![0.3]],
            logfoldchanges: vec![vec![0.5]],
        };
        let chunk2 = DiffExpResult {
            group_names: vec!["G".to_string()],
            names: vec![vec!["gene_b".to_string()]],
            scores: vec![vec![3.0]],
            pvals: vec![vec![0.001]],
            pvals_adj: vec![vec![0.001]],
            logfoldchanges: vec![vec![2.0]],
        };

        let merged = merge_diff_exp_results(vec![chunk1, chunk2], false).unwrap();
        assert_eq!(merged.group_names, vec!["G"]);
        assert_eq!(merged.names[0].len(), 2);
        // gene_b should be first (signed score 3.0 > 1.0).
        assert_eq!(merged.names[0][0], "gene_b");
        assert_eq!(merged.names[0][1], "gene_a");
        assert_eq!(merged.scores[0][0], 3.0);
        assert_eq!(merged.scores[0][1], 1.0);
    }

    #[test]
    fn test_merge_global_bh() {
        // Two chunks with 1 gene each → merged BH uses n=2, not n=1.
        let chunk1 = DiffExpResult {
            group_names: vec!["G".to_string()],
            names: vec![vec!["gene_a".to_string()]],
            scores: vec![vec![2.0]],
            pvals: vec![vec![0.04]],
            pvals_adj: vec![vec![0.04]], // per-chunk BH with n=1
            logfoldchanges: vec![vec![1.0]],
        };
        let chunk2 = DiffExpResult {
            group_names: vec!["G".to_string()],
            names: vec![vec!["gene_b".to_string()]],
            scores: vec![vec![1.0]],
            pvals: vec![vec![0.03]],
            pvals_adj: vec![vec![0.03]],
            logfoldchanges: vec![vec![0.5]],
        };

        let merged = merge_diff_exp_results(vec![chunk1, chunk2], false).unwrap();
        // With global BH (n=2): sorted p-vals are [0.03, 0.04]
        // Results sorted by signed score: gene_a (z=2) first, gene_b (z=1) second
        // So pvals_adj order follows the score sort.
        // All adjusted should be >= raw and <= 1.
        for (raw, adj) in merged.pvals[0].iter().zip(merged.pvals_adj[0].iter()) {
            assert!(*adj >= *raw - 1e-12);
            assert!(*adj <= 1.0 + 1e-12);
        }
    }

    #[test]
    fn test_wilcoxon_sparse_matches_dense() {
        // Build a small CSR and verify wilcoxon_rank_sum_sparse produces
        // bit-identical results to wilcoxon_rank_sum on equivalent dense data.
        let n_obs = 20;
        let n_vars = 3;

        let mut dense_data = vec![0.0f32; n_obs * n_vars];
        let groups: Vec<usize> = (0..n_obs).map(|i| if i < 10 { 0 } else { 1 }).collect();

        for i in 0..10 {
            dense_data[i * n_vars + 0] = 10.0 + i as f32;
            dense_data[i * n_vars + 1] = 5.0;
            dense_data[i * n_vars + 2] = 1.0;
        }
        for i in 10..20 {
            dense_data[i * n_vars + 0] = 1.0;
            dense_data[i * n_vars + 1] = 5.0;
            dense_data[i * n_vars + 2] = 10.0 + (i - 10) as f32;
        }

        let gene_names: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
        let group_names = vec!["A".to_string(), "B".to_string()];

        // Dense path
        let result_dense = wilcoxon_rank_sum(
            &dense_data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            None,
            false,
            false,
            false,
        )
        .unwrap();

        // Build ScxCsr from the same data
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for r in 0..n_obs {
            for c in 0..n_vars {
                let v = dense_data[r * n_vars + c];
                if v != 0.0 {
                    indices.push(c as i32);
                    data.push(v);
                }
            }
            indptr.push(indices.len() as i64);
        }
        let csr = scx_sparse::ScxCsr::new_unchecked((n_obs, n_vars), indptr, indices, data);

        // Sparse path with chunk_size=2 (forces 2 chunks for 3 genes)
        let result_sparse = wilcoxon_rank_sum_sparse(
            &csr,
            &gene_names,
            &groups,
            &group_names,
            None,
            2,
            false,
            false,
            false,
        )
        .unwrap();

        // Same group structure
        assert_eq!(result_dense.group_names, result_sparse.group_names);

        // Same gene names per group (same ordering)
        for g in 0..result_dense.group_names.len() {
            assert_eq!(result_dense.names[g], result_sparse.names[g]);
            // Scores should match
            for i in 0..result_dense.scores[g].len() {
                assert!(
                    (result_dense.scores[g][i] - result_sparse.scores[g][i]).abs() < 1e-10,
                    "score mismatch at group {} gene {}: {} vs {}",
                    g,
                    i,
                    result_dense.scores[g][i],
                    result_sparse.scores[g][i],
                );
            }
            // Raw p-values should match
            for i in 0..result_dense.pvals[g].len() {
                assert!(
                    (result_dense.pvals[g][i] - result_sparse.pvals[g][i]).abs() < 1e-10,
                    "pval mismatch at group {} gene {}",
                    g,
                    i,
                );
            }
        }
    }

    // ── Multi-shard streaming + cache regression coverage ───────────────────
    // These tests guard the cache-bypass fix at the `read_shard_cached_arc`
    // call site above. They write an SCX file with > 1 CSR shard, then run
    // `wilcoxon_rank_sum_streaming` against a `BackedCsrReader` and compare
    // results to `wilcoxon_rank_sum_sparse` on the same data in memory.
    use arrow::array::{RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use scx_codec::{CodecId, ValueEncoding};
    use scx_format::header::{FileHeader, MAGIC};
    use scx_format::{BackedCsrReader, ScxReader, ScxWriter};
    use std::path::Path;
    use std::sync::Arc as StdArc;

    /// Deterministic dense matrix where ~1/3 of cells are non-zero. The
    /// pattern guarantees nontrivial rank-sum statistics across two groups
    /// because column values vary with both row and column index.
    fn make_dense(n_obs: usize, n_vars: usize) -> Vec<u8> {
        let mut dense = vec![0u8; n_obs * n_vars];
        for r in 0..n_obs {
            for c in 0..n_vars {
                if (r + c) % 3 == 0 {
                    dense[r * n_vars + c] = ((r * 7 + c * 11) % 200 + 1) as u8;
                }
            }
        }
        dense
    }

    /// Write a `.scx` file with `n_shards` CSR shards built from a row
    /// partition of `dense`. The shards split rows evenly (last shard
    /// absorbs the remainder).
    fn write_multi_shard_csr(
        path: &Path,
        n_obs: usize,
        n_vars: usize,
        dense: &[u8],
        n_shards: usize,
    ) -> std::io::Result<()> {
        let header = FileHeader {
            magic: MAGIC,
            format_version: scx_format::CURRENT_FORMAT_VERSION,
            header_length: 256,
            flags: 0,
            n_obs: n_obs as u64,
            n_vars: n_vars as u64,
            nnz: 0,
            n_csr_shards: 0,
            n_csc_shards: 0,
            shard_target_rows: 16384,
            codec_id: 0,
            index_dtype: 0,
            endian: 0,
            reserved_padding: 0,
            root_catalog_offset: 0,
            root_catalog_length: 0,
            full_catalog_offset: 0,
            full_catalog_length: 0,
            manifest_sequence: 1,
            prev_catalog_offset: 0,
            file_checksum: 0,
            front_catalog_offset: 0,
            front_catalog_length: 0,
            n_modalities: 0,
            modality_table_offset: 0,
            modality_table_length: 0,
            reserved: [0u8; 112],
        };
        let mut writer = ScxWriter::new(path, header).unwrap();

        let obs_ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
        let obs_schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
        let obs = RecordBatch::try_new(
            StdArc::new(obs_schema),
            vec![StdArc::new(StringArray::from(
                obs_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap();
        writer.write_obs(&obs).unwrap();

        let var_ids: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
        let var_schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
        let var = RecordBatch::try_new(
            StdArc::new(var_schema),
            vec![StdArc::new(StringArray::from(
                var_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap();
        writer.write_var(&var).unwrap();

        let rows_per_shard = n_obs.div_ceil(n_shards);
        for s in 0..n_shards {
            let row_start = s * rows_per_shard;
            if row_start >= n_obs {
                break;
            }
            let row_end = (row_start + rows_per_shard).min(n_obs);

            let mut indptr: Vec<u64> = vec![0];
            let mut indices = Vec::new();
            let mut values = Vec::new();
            for r in row_start..row_end {
                for c in 0..n_vars {
                    let v = dense[r * n_vars + c];
                    if v != 0 {
                        indices.push(c as u32);
                        values.push(v);
                    }
                }
                indptr.push(indices.len() as u64);
            }
            writer
                .write_csr_shard(
                    &indptr,
                    &indices,
                    &values,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    row_start as u64,
                )
                .unwrap();
        }
        writer.finish().unwrap();
        Ok(())
    }

    /// Build an in-memory `ScxCsr` covering the full dense matrix.
    fn dense_to_full_csr(dense: &[u8], n_obs: usize, n_vars: usize) -> scx_sparse::ScxCsr {
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for r in 0..n_obs {
            for c in 0..n_vars {
                let v = dense[r * n_vars + c];
                if v != 0 {
                    indices.push(c as i32);
                    data.push(v as f32);
                }
            }
            indptr.push(indices.len() as i64);
        }
        scx_sparse::ScxCsr::new_unchecked((n_obs, n_vars), indptr, indices, data)
    }

    fn assert_diffexp_results_match(
        cached: &DiffExpResult,
        reference: &DiffExpResult,
        score_atol: f64,
        pval_atol: f64,
    ) {
        assert_eq!(cached.group_names, reference.group_names);
        for g in 0..cached.group_names.len() {
            assert_eq!(
                cached.names[g], reference.names[g],
                "gene ordering diverges for group {g}"
            );
            for k in 0..cached.scores[g].len() {
                let ds = (cached.scores[g][k] - reference.scores[g][k]).abs();
                assert!(
                    ds < score_atol,
                    "score mismatch at group {g} rank {k}: {} vs {} (Δ={ds})",
                    cached.scores[g][k],
                    reference.scores[g][k]
                );
                let dp = (cached.pvals[g][k] - reference.pvals[g][k]).abs();
                assert!(
                    dp < pval_atol,
                    "pval mismatch at group {g} rank {k}: {} vs {} (Δ={dp})",
                    cached.pvals[g][k],
                    reference.pvals[g][k]
                );
            }
        }
    }

    #[test]
    fn streaming_with_cache_matches_sparse_kernel() {
        // n_obs=120, n_vars=80, n_shards=4, gene_chunk_size=25
        //   → ⌈80/25⌉ = 4 gene chunks, so the outer loop visits every
        //     shard 4× — the exact multi-pass pattern the cache should
        //     short-circuit.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wilcox_cached.scx");
        let n_obs = 120usize;
        let n_vars = 80usize;
        let n_shards = 4usize;
        let gene_chunk_size = 25usize;

        let dense = make_dense(n_obs, n_vars);
        write_multi_shard_csr(&path, n_obs, n_vars, &dense, n_shards).unwrap();

        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("gene_{j}")).collect();
        let group_names = vec!["A".to_string(), "B".to_string()];
        let groups: Vec<usize> = (0..n_obs)
            .map(|i| if i < n_obs / 2 { 0 } else { 1 })
            .collect();

        // Cache sized to cover all shards plus a couple slack slots, matching
        // the documented `cache_shards >= n_shards` recommendation.
        let mut reader = BackedCsrReader::new(ScxReader::open(&path).unwrap(), n_shards + 2);
        let metrics = reader.enable_metrics();
        assert_eq!(reader.index().n_shards(), n_shards);

        let cached_res = wilcoxon_rank_sum_streaming(
            &reader,
            &gene_names,
            &groups,
            &group_names,
            None,
            gene_chunk_size,
            false,
            false,
            false,
        )
        .unwrap();

        // The cache must actually have been consulted: with 4 chunks × 4
        // shards = 16 lookups and 4 unique shards, we expect exactly 4
        // misses (cold population) and 12 hits.
        use std::sync::atomic::Ordering;
        let hits = metrics.hits.load(Ordering::Relaxed);
        let misses = metrics.misses.load(Ordering::Relaxed);
        assert_eq!(
            misses, n_shards as u64,
            "expected one miss per unique shard, got {misses}"
        );
        assert_eq!(
            hits,
            (n_shards * (n_vars.div_ceil(gene_chunk_size) - 1)) as u64,
            "expected cache hits on every chunk after the first"
        );

        let in_mem = dense_to_full_csr(&dense, n_obs, n_vars);
        let reference = wilcoxon_rank_sum_sparse(
            &in_mem,
            &gene_names,
            &groups,
            &group_names,
            None,
            gene_chunk_size,
            false,
            false,
            false,
        )
        .unwrap();

        assert_diffexp_results_match(&cached_res, &reference, 1e-5, 1e-5);
    }

    #[test]
    fn streaming_with_zero_cache_still_correct() {
        // Same dataset; force `cache_shards = 0` so the LRU is `None` and
        // `read_shard_cached_arc` falls through to decode-on-each-call.
        // Result must still equal the sparse kernel.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wilcox_uncached.scx");
        let n_obs = 120usize;
        let n_vars = 80usize;
        let n_shards = 4usize;
        let gene_chunk_size = 25usize;

        let dense = make_dense(n_obs, n_vars);
        write_multi_shard_csr(&path, n_obs, n_vars, &dense, n_shards).unwrap();

        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("gene_{j}")).collect();
        let group_names = vec!["A".to_string(), "B".to_string()];
        let groups: Vec<usize> = (0..n_obs)
            .map(|i| if i < n_obs / 2 { 0 } else { 1 })
            .collect();

        let reader = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);
        assert!(
            !reader.cache_contains(0),
            "cache should be None when cache_shards=0"
        );

        let uncached_res = wilcoxon_rank_sum_streaming(
            &reader,
            &gene_names,
            &groups,
            &group_names,
            None,
            gene_chunk_size,
            false,
            false,
            false,
        )
        .unwrap();

        let in_mem = dense_to_full_csr(&dense, n_obs, n_vars);
        let reference = wilcoxon_rank_sum_sparse(
            &in_mem,
            &gene_names,
            &groups,
            &group_names,
            None,
            gene_chunk_size,
            false,
            false,
            false,
        )
        .unwrap();

        assert_diffexp_results_match(&uncached_res, &reference, 1e-5, 1e-5);
    }

    // ── pdex `mode="ref"` accelerator tests ──────────────────────────────────

    use crate::pseudobulk::GeomMeanMode;

    fn build_pdex_fixture(n_obs: usize, n_vars: usize) -> (Vec<f32>, Vec<usize>, Vec<String>) {
        // Three groups: 0=ref, 1=test_a, 2=test_b. Each gets a third of cells.
        let mut data = vec![0.0f32; n_obs * n_vars];
        let mut groups = vec![0usize; n_obs];
        let third = n_obs / 3;
        for cell in 0..n_obs {
            let g = if cell < third {
                0
            } else if cell < 2 * third {
                1
            } else {
                2
            };
            groups[cell] = g;
            for gene in 0..n_vars {
                // Deterministic, non-trivial pattern: each gene gets a shifted
                // sequence by group, plus a per-cell perturbation.
                let base = (g as f32) * 2.0 + (gene as f32) * 0.5;
                let pert = ((cell + gene * 7) % 11) as f32 * 0.1;
                data[cell * n_vars + gene] = base + pert;
            }
        }
        let group_names = vec!["ref".to_string(), "ta".to_string(), "tb".to_string()];
        (data, groups, group_names)
    }

    #[test]
    fn test_pdex_ref_smoke_dense() {
        let n_obs = 60;
        let n_vars = 4;
        let (data, groups, group_names) = build_pdex_fixture(n_obs, n_vars);
        let gene_names: Vec<String> = (0..n_vars).map(|i| format!("g{i}")).collect();

        let result = pdex_ref(
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            0, // reference = "ref"
            GeomMeanMode::ArithRaw,
            0.0,
        )
        .unwrap();

        // Reference is excluded; two test groups remain.
        assert_eq!(result.group_names, vec!["ta", "tb"]);
        assert_eq!(result.feature_names.len(), n_vars);
        assert_eq!(result.ref_means.len(), n_vars);
        assert_eq!(result.ref_membership, n_obs / 3);
        assert_eq!(result.target_memberships, vec![n_obs / 3, n_obs / 3]);

        // Shapes per test group.
        for tg in 0..2 {
            assert_eq!(result.target_means[tg].len(), n_vars);
            assert_eq!(result.log2_fold_changes[tg].len(), n_vars);
            assert_eq!(result.percent_changes[tg].len(), n_vars);
            assert_eq!(result.statistics[tg].len(), n_vars);
            assert_eq!(result.p_values[tg].len(), n_vars);
            assert_eq!(result.fdrs[tg].len(), n_vars);
        }

        // Sanity: ref/test means should be positive in this fixture, and the
        // test groups have a strictly larger group-shift than the reference,
        // so log2_fc should be positive for every gene.
        for tg in 0..2 {
            for gene in 0..n_vars {
                assert!(
                    result.target_means[tg][gene] > result.ref_means[gene],
                    "test group {} gene {} expected larger mean than ref",
                    tg,
                    gene
                );
                assert!(
                    result.log2_fold_changes[tg][gene] > 0.0,
                    "log2_fc should be positive for tg={} gene={}",
                    tg,
                    gene
                );
                // FDR is BH-adjusted, must be in [0, 1] and >= raw p.
                let p = result.p_values[tg][gene];
                let fdr = result.fdrs[tg][gene];
                assert!((0.0..=1.0).contains(&p));
                assert!((0.0..=1.0).contains(&fdr));
                assert!(fdr + 1e-12 >= p, "FDR must be >= raw p");
            }
        }
    }

    #[test]
    fn test_pdex_ref_four_geom_modes() {
        // Verify that all four GeomMeanMode variants run end-to-end and produce
        // finite means / log2_fcs. Compare ArithRaw and GeomLog1p (the two
        // "no transform per cell" modes) on raw vs log1p inputs.
        let n_obs = 30;
        let n_vars = 3;
        let (mut data, groups, group_names) = build_pdex_fixture(n_obs, n_vars);
        let gene_names: Vec<String> = (0..n_vars).map(|i| format!("g{i}")).collect();

        let modes = [
            GeomMeanMode::ArithRaw,
            GeomMeanMode::ArithLog1pExpand,
            GeomMeanMode::GeomRaw,
            GeomMeanMode::GeomLog1p,
        ];

        for mode in modes {
            let raw_result = pdex_ref(
                &data,
                n_obs,
                n_vars,
                &gene_names,
                &groups,
                &group_names,
                0,
                mode,
                0.0,
            )
            .unwrap();

            for tg in 0..2 {
                for gene in 0..n_vars {
                    assert!(
                        raw_result.target_means[tg][gene].is_finite(),
                        "mode {:?}: target_mean must be finite",
                        mode
                    );
                    assert!(
                        raw_result.ref_means[gene].is_finite(),
                        "mode {:?}: ref_mean must be finite",
                        mode
                    );
                    assert!(
                        raw_result.log2_fold_changes[tg][gene].is_finite(),
                        "mode {:?}: log2_fc must be finite",
                        mode
                    );
                }
            }
        }

        // For GeomLog1p: the input is treated as log1p already, so the natural
        // mean is expm1(arithmetic_mean). Verify by applying log1p to data and
        // checking that GeomRaw on log1p(data) ≈ GeomLog1p on log1p(data).
        for v in data.iter_mut() {
            *v = v.ln_1p();
        }
        let geom_raw = pdex_ref(
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            0,
            GeomMeanMode::GeomRaw,
            0.0,
        )
        .unwrap();
        let geom_log1p = pdex_ref(
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            0,
            GeomMeanMode::GeomLog1p,
            0.0,
        )
        .unwrap();
        for gene in 0..n_vars {
            // GeomRaw on log1p input applies log1p again → expm1(mean(log1p^2(x))).
            // That is NOT equal to GeomLog1p on log1p input (expm1(mean(log1p(x)))).
            // So we only check that both are finite and positive, not that they
            // match. The four-mode parity is enforced by the Python parity test
            // against pdex (pyscx/tests/test_pdex_ref_parity.py).
            assert!(geom_raw.ref_means[gene] >= 0.0);
            assert!(geom_log1p.ref_means[gene] >= 0.0);
        }
    }

    #[test]
    fn test_pdex_ref_epsilon_stabilises_zero_ref() {
        // When ref_mean = 0, epsilon avoids division-by-zero in
        // log2_fold_change and percent_change.
        let n_obs = 12;
        let n_vars = 1;
        let mut data = vec![0.0f32; n_obs * n_vars];
        let groups = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        // Group 0 (ref) is all zeros. Group 1 has positive values.
        for cell in 4..8 {
            data[cell] = 3.0;
        }
        for cell in 8..12 {
            data[cell] = 5.0;
        }
        let group_names = vec!["ref".to_string(), "ta".to_string(), "tb".to_string()];
        let gene_names = vec!["g0".to_string()];

        // epsilon=0 → log2_fc = log2(target / 0) = +inf for non-zero target.
        let r0 = pdex_ref(
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            0,
            GeomMeanMode::ArithRaw,
            0.0,
        )
        .unwrap();
        assert!(
            r0.log2_fold_changes[0][0].is_infinite(),
            "log2_fc with eps=0 and ref_mean=0 should be +inf"
        );

        // epsilon=0.5 → log2_fc is finite.
        let r1 = pdex_ref(
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            0,
            GeomMeanMode::ArithRaw,
            0.5,
        )
        .unwrap();
        for tg in 0..2 {
            assert!(
                r1.log2_fold_changes[tg][0].is_finite(),
                "log2_fc with eps=0.5 should be finite"
            );
        }
    }

    #[test]
    fn test_pdex_ref_sparse_matches_dense() {
        // Build a sparse CSR mirroring the fixture and verify pdex_ref_sparse
        // produces the same outputs as pdex_ref (with gene chunking forcing
        // the merge path).
        let n_obs = 30;
        let n_vars = 5;
        let (dense, groups, group_names) = build_pdex_fixture(n_obs, n_vars);
        let gene_names: Vec<String> = (0..n_vars).map(|i| format!("g{i}")).collect();

        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for r in 0..n_obs {
            for c in 0..n_vars {
                let v = dense[r * n_vars + c];
                if v != 0.0 {
                    indices.push(c as i32);
                    data.push(v);
                }
            }
            indptr.push(indices.len() as i64);
        }
        let csr = scx_sparse::ScxCsr::new_unchecked((n_obs, n_vars), indptr, indices, data);

        let dense_res = pdex_ref(
            &dense,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            0,
            GeomMeanMode::ArithRaw,
            0.0,
        )
        .unwrap();

        // chunk_size=2 → forces three chunks across 5 genes, exercising merge.
        let sparse_res = pdex_ref_sparse(
            &csr,
            &gene_names,
            &groups,
            &group_names,
            0,
            2,
            GeomMeanMode::ArithRaw,
            0.0,
        )
        .unwrap();

        assert_eq!(dense_res.group_names, sparse_res.group_names);
        assert_eq!(dense_res.feature_names, sparse_res.feature_names);
        assert_eq!(dense_res.ref_membership, sparse_res.ref_membership);
        assert_eq!(dense_res.target_memberships, sparse_res.target_memberships);

        for gene in 0..n_vars {
            assert!((dense_res.ref_means[gene] - sparse_res.ref_means[gene]).abs() < 1e-9);
        }
        for tg in 0..dense_res.group_names.len() {
            for gene in 0..n_vars {
                assert!(
                    (dense_res.target_means[tg][gene] - sparse_res.target_means[tg][gene]).abs()
                        < 1e-9
                );
                assert!(
                    (dense_res.log2_fold_changes[tg][gene]
                        - sparse_res.log2_fold_changes[tg][gene])
                        .abs()
                        < 1e-9
                );
                assert!(
                    (dense_res.percent_changes[tg][gene] - sparse_res.percent_changes[tg][gene])
                        .abs()
                        < 1e-9
                );
                assert!(
                    (dense_res.statistics[tg][gene] - sparse_res.statistics[tg][gene]).abs() < 1e-9
                );
                assert!(
                    (dense_res.p_values[tg][gene] - sparse_res.p_values[tg][gene]).abs() < 1e-9
                );
                assert!((dense_res.fdrs[tg][gene] - sparse_res.fdrs[tg][gene]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_pdex_ref_rejects_invalid_inputs() {
        let n_obs = 9;
        let n_vars = 2;
        let (data, mut groups, mut group_names) = build_pdex_fixture(n_obs, n_vars);
        let gene_names: Vec<String> = (0..n_vars).map(|i| format!("g{i}")).collect();

        // Reference index out of range.
        assert!(pdex_ref(
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            99,
            GeomMeanMode::ArithRaw,
            0.0,
        )
        .is_err());

        // Negative epsilon.
        assert!(pdex_ref(
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            0,
            GeomMeanMode::ArithRaw,
            -0.1,
        )
        .is_err());

        // Reference group has zero cells (assign no cell to group 99).
        groups.iter_mut().for_each(|g| {
            if *g == 0 {
                *g = 1;
            }
        });
        group_names.push("empty_ref".to_string());
        assert!(pdex_ref(
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            group_names.len() - 1,
            GeomMeanMode::ArithRaw,
            0.0,
        )
        .is_err());
    }
}
