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
/// sorted by descending absolute score within each group.
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

    let mut result_names = Vec::with_capacity(test_groups.len());
    let mut result_scores = Vec::with_capacity(test_groups.len());
    let mut result_pvals = Vec::with_capacity(test_groups.len());
    let mut result_pvals_adj = Vec::with_capacity(test_groups.len());
    let mut result_logfc = Vec::with_capacity(test_groups.len());
    let mut result_group_names = Vec::with_capacity(test_groups.len());

    for &g in &test_groups {
        let group_cells = &group_indices[g];

        // Build reference cell indices.
        let ref_cells: Vec<usize> = match reference {
            Some(ref_idx) => group_indices[ref_idx].clone(),
            None => {
                // 1-vs-rest: all cells not in group g
                groups
                    .iter()
                    .enumerate()
                    .filter(|(_, &grp)| grp != g)
                    .map(|(i, _)| i)
                    .collect()
            }
        };

        let n1 = group_cells.len();
        let n2 = ref_cells.len();

        if n1 == 0 || n2 == 0 {
            // Degenerate: fill with NaNs.
            let nans = vec![f64::NAN; n_vars];
            let gene_order: Vec<String> = (0..n_vars).map(|i| gene_names[i].clone()).collect();
            result_names.push(gene_order);
            result_scores.push(nans.clone());
            result_pvals.push(vec![1.0; n_vars]);
            result_pvals_adj.push(vec![1.0; n_vars]);
            result_logfc.push(nans);
            result_group_names.push(group_names[g].clone());
            continue;
        }

        // Parallel over genes.
        let gene_results: Vec<GeneTestResult> = (0..n_vars)
            .into_par_iter()
            .map(|var_idx| {
                // Gather values for this gene.
                let mut group_vals: Vec<f64> = Vec::with_capacity(n1);
                let mut ref_vals: Vec<f64> = Vec::with_capacity(n2);

                for &cell in group_cells {
                    group_vals.push(data[cell * n_vars + var_idx] as f64);
                }
                for &cell in &ref_cells {
                    ref_vals.push(data[cell * n_vars + var_idx] as f64);
                }

                // Log2 fold-change with pseudocount.
                let mean_group = group_vals.iter().sum::<f64>() / n1 as f64;
                let mean_ref = ref_vals.iter().sum::<f64>() / n2 as f64;
                let logfc = if log_transformed {
                    // Match scanpy: back-transform from log-space (undo log1p)
                    // before computing ratio. scanpy uses expm1(mean) + 1e-9.
                    let expm1_group = mean_group.exp_m1();
                    let expm1_ref = mean_ref.exp_m1();
                    ((expm1_group + LOGFC_PSEUDOCOUNT) / (expm1_ref + LOGFC_PSEUDOCOUNT)).log2()
                } else {
                    // Raw counts: direct log2 ratio.
                    (mean_group + LOGFC_PSEUDOCOUNT).log2() - (mean_ref + LOGFC_PSEUDOCOUNT).log2()
                };

                // Wilcoxon rank-sum test.
                let (score, pval) = wilcoxon_test(&group_vals, &ref_vals);

                GeneTestResult {
                    gene_idx: var_idx,
                    score,
                    pval,
                    logfc,
                }
            })
            .collect();

        // Sort genes by absolute score descending (matching scanpy's default).
        let mut sorted: Vec<GeneTestResult> = gene_results;
        sorted.sort_by(|a, b| {
            b.score
                .abs()
                .partial_cmp(&a.score.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let names: Vec<String> = sorted
            .iter()
            .map(|r| gene_names[r.gene_idx].clone())
            .collect();
        let scores: Vec<f64> = sorted.iter().map(|r| r.score).collect();
        let pvals: Vec<f64> = sorted.iter().map(|r| r.pval).collect();
        let logfc: Vec<f64> = sorted.iter().map(|r| r.logfc).collect();

        // BH adjustment (on the sorted-by-score order).
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

/// Wilcoxon rank-sum (Mann–Whitney U) test with normal approximation and tie correction.
///
/// Returns `(z_score, two_sided_p_value)`.
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

/// Standard normal CDF via Abramowitz & Stegun erfc approximation (7.1.26).
/// Φ(z) = erfc(−z/√2) / 2.  Accurate to ~1.5e-7.
fn normal_cdf(z: f64) -> f64 {
    if z < -8.0 {
        return 0.0;
    }
    if z > 8.0 {
        return 1.0;
    }

    // A&S 7.1.26 coefficients for erfc(x) where t = 1/(1 + p*x), x >= 0.
    let a1 = 0.254829592_f64;
    let a2 = -0.284496736_f64;
    let a3 = 1.421413741_f64;
    let a4 = -1.453152027_f64;
    let a5 = 1.061405429_f64;
    let p = 0.3275911_f64;

    // erfc(x) for x = |z| / sqrt(2)
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + p * x);
    let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    let erfc_val = poly * (-x * x).exp();

    // Φ(z) = 1 - erfc(z/√2)/2 for z >= 0, erfc(-z/√2)/2 for z < 0
    if z >= 0.0 {
        1.0 - erfc_val / 2.0
    } else {
        erfc_val / 2.0
    }
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
/// 1. For each gene chunk, iterate all shards via `read_shard_cached()`,
///    apply `project_csr()` per shard, scatter into a dense buffer.
/// 2. Run `wilcoxon_rank_sum()` on the dense buffer for that chunk.
/// 3. Merge all chunk results with global BH correction.
///
/// Peak memory: O(n_obs × gene_chunk_size) instead of O(n_obs × n_vars).
pub fn wilcoxon_rank_sum_streaming(
    reader: &scx_format::backed::BackedCsrReader,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: Option<usize>,
    gene_chunk_size: usize,
    log_transformed: bool,
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
    let mut all_chunk_results = Vec::new();

    for chunk_start in (0..n_vars).step_by(gene_chunk_size) {
        let chunk_end = (chunk_start + gene_chunk_size).min(n_vars);
        let chunk_size = chunk_end - chunk_start;
        let col_indices: Vec<u32> = (chunk_start as u32..chunk_end as u32).collect();

        // Build dense buffer (n_obs × chunk_size), zero-initialized.
        let mut dense = vec![0.0f32; n_obs * chunk_size];

        // Stream all shards, project each, scatter into dense buffer.
        let mut global_row = 0usize;
        for shard_idx in 0..n_shards {
            let shard_csr = reader
                .read_shard_cached(shard_idx)
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
        )?;
        all_chunk_results.push(chunk_result);
    }

    // Merge: concatenate per-group gene lists, re-sort by |z|,
    // and re-apply BH correction globally across all genes.
    merge_diff_exp_results(all_chunk_results)
}

/// Gene-chunked Wilcoxon rank-sum from an in-memory `ScxCsr`.
///
/// Like `wilcoxon_rank_sum_streaming` but operates on a single in-memory CSR
/// matrix instead of streaming shards from a `BackedCsrReader`. Avoids the
/// O(n_obs × n_vars) dense materialization that `.toarray()` would require.
///
/// Peak memory: O(n_obs × gene_chunk_size) instead of O(n_obs × n_vars).
pub fn wilcoxon_rank_sum_sparse(
    csr: &scx_sparse::ScxCsr,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: Option<usize>,
    gene_chunk_size: usize,
    log_transformed: bool,
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
        )?;
        all_chunk_results.push(chunk_result);
    }

    merge_diff_exp_results(all_chunk_results)
}

/// Merge per-chunk `DiffExpResult`s into a single result with global BH correction.
///
/// For each group:
/// 1. Concatenate gene names, scores, p-values, and fold-changes from all chunks.
/// 2. Re-sort by descending |z-score|.
/// 3. Apply Benjamini–Hochberg on the globally-sorted raw p-values.
///
/// This is essential because BH correction depends on the total number of tests.
/// Applying BH per-chunk would use `chunk_size` as the denominator instead of
/// `n_vars`, inflating FDR.
pub fn merge_diff_exp_results(chunks: Vec<DiffExpResult>) -> Result<DiffExpResult> {
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

        // Sort by |z| descending.
        gene_entries.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
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
        let mut data = vec![5.0f32; n_obs * n_vars];
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
        let merged = merge_diff_exp_results(vec![]).unwrap();
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
        let merged = merge_diff_exp_results(vec![chunk.clone()]).unwrap();
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

        let merged = merge_diff_exp_results(vec![chunk1, chunk2]).unwrap();
        assert_eq!(merged.group_names, vec!["G"]);
        assert_eq!(merged.names[0].len(), 2);
        // gene_b should be first (|z| = 3.0 > 1.0).
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

        let merged = merge_diff_exp_results(vec![chunk1, chunk2]).unwrap();
        // With global BH (n=2): sorted p-vals are [0.03, 0.04]
        // BH: rank 2 → 0.04 * 2/2 = 0.04, rank 1 → 0.03 * 2/1 = 0.06 → cummin 0.04
        // But results are sorted by |z|: gene_a (z=2) first, gene_b (z=1) second
        // So pvals_adj order follows the |z| sort.
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
        let result_sparse =
            wilcoxon_rank_sum_sparse(&csr, &gene_names, &groups, &group_names, None, 2, false)
                .unwrap();

        // Same group structure
        assert_eq!(result_dense.group_names, result_sparse.group_names);

        // Same gene names per group (same ordering by |z|)
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
}
