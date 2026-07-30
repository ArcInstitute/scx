//! CSC streaming Wilcoxon rank-sum DE.
//!
//! Mirrors [`crate::diffexp::wilcoxon_rank_sum_streaming`] (CSR path)
//! but processes gene chunks as CSC column ranges. For each chunk:
//!   1. `source.read_csc_columns(chunk_start..chunk_end)` materialises
//!      one CSC slab covering the chunk's columns.
//!   2. Scatter the slab's `(row, col, value)` triples into a row-major
//!      `[n_obs × chunk_size]` dense buffer.
//!   3. Call the existing `wilcoxon_rank_sum()` kernel on that buffer.
//!
//! The chunk-merge step (`merge_diff_exp_results`) runs unchanged on
//! per-chunk results, so the numerics are identical to the CSR path.

use std::sync::OnceLock;

use crate::diffexp::cpu::{compute_logfc, de_rank_cmp, wilcoxon_stats_from_rank_sum};
use crate::diffexp::{
    benjamini_hochberg, merge_diff_exp_results, wilcoxon_rank_sum, DiffExpResult,
};
use crate::error::{AccelError, Result};
use scx_format_io::ColumnShardSource;

/// Opt-in gate for the exact sparse-nnz Wilcoxon fast path (§5.3).
///
/// Default **off**: the densify + dense-kernel path stays the measured baseline
/// until a benchmark promotes the nnz path to default. `SCX_ACCEL_WILCOXON_NNZ=1`
/// opts in. The nnz path is *numerically equivalent* (property-tested to ~1e-9)
/// but ranks only the nonzeros plus a synthesized implicit-zero tie-block instead
/// of sorting an `n_obs`-length dense column per gene.
fn nnz_wilcoxon_enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        matches!(
            std::env::var("SCX_ACCEL_WILCOXON_NNZ").ok().as_deref(),
            Some("1" | "true" | "TRUE" | "on")
        )
    })
}

/// Gene-chunked Wilcoxon rank-sum DE driven by a CSC source.
///
/// Returns the same [`DiffExpResult`] as the CSR equivalent. Chunk
/// boundaries are aligned to columns (genes), not rows.
#[allow(clippy::too_many_arguments)]
pub fn wilcoxon_rank_sum_streaming_csc<S: ColumnShardSource + ?Sized>(
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
        return Err(AccelError::InvalidInput(format!(
            "groups length {} != n_obs {}",
            groups.len(),
            n_obs
        )));
    }
    if gene_chunk_size == 0 {
        return Err(AccelError::InvalidInput(
            "gene_chunk_size must be > 0".to_string(),
        ));
    }
    if n_vars != source.n_vars() {
        return Err(AccelError::ShapeError(format!(
            "gene_names length {} != source.n_vars() {}",
            n_vars,
            source.n_vars()
        )));
    }
    // Exact sparse-nnz fast path (§5.3), opt-in and 1-vs-rest only (rankby_abs
    // and an explicit reference keep the densify path). Ranks only the nonzeros +
    // an analytic implicit-zero tie-block — no `n_obs` dense column per gene.
    if reference.is_none() && !rankby_abs && nnz_wilcoxon_enabled() {
        return wilcoxon_rank_sum_nnz_csc(
            source,
            gene_names,
            groups,
            group_names,
            gene_chunk_size,
            log_transformed,
            tie_correct,
        );
    }

    // Clamp the dense n_obs×chunk f32 scatter buffer to the CPU memory budget.
    let gene_chunk_size = crate::mem_budget::de_gene_chunk_or_err(
        gene_chunk_size,
        n_obs,
        "wilcoxon_rank_sum_streaming_csc",
    )?;

    let mut all_chunk_results = Vec::new();
    // Allocate the row-major dense scatter buffer once and reuse across
    // chunks. The trailing chunk may be smaller; we slice down and zero
    // only the active sub-range, avoiding the per-chunk full allocation.
    let max_chunk = gene_chunk_size.min(n_vars);
    let mut dense = vec![0.0f32; n_obs.saturating_mul(max_chunk)];

    for chunk_start in (0..n_vars).step_by(gene_chunk_size) {
        let chunk_end = (chunk_start + gene_chunk_size).min(n_vars);
        let chunk_size = chunk_end - chunk_start;

        // Read this chunk's columns as a single CSC slab.
        let csc = source
            .read_csc_columns(chunk_start as u32..chunk_end as u32)
            .map_err(AccelError::Scx)?;

        // Scatter into row-major dense buffer [n_obs × chunk_size].
        let dense_view = &mut dense[..n_obs * chunk_size];
        dense_view.fill(0.0);
        for local_col in 0..chunk_size {
            let s = csc.indptr[local_col] as usize;
            let e = csc.indptr[local_col + 1] as usize;
            for j in s..e {
                let row = csc.indices[j] as usize;
                if row < n_obs {
                    dense_view[row * chunk_size + local_col] = csc.data[j];
                }
            }
        }

        let chunk_result = wilcoxon_rank_sum(
            dense_view,
            n_obs,
            chunk_size,
            &gene_names[chunk_start..chunk_end],
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

/// Assign 1-based mid-ranks to a value-sorted block (negatives or positives),
/// accumulating each entry's rank into its group's rank-sum and the tie
/// correction `Σ(t³−t)`.
///
/// `block` is sorted ascending by value; `offset` is the count of cells sorted
/// *before* this block (0 for negatives; `n_neg + n_zero` for positives). Global
/// 1-based rank of a tie run at within-block positions `[i, j)` is
/// `offset + (i + 1 + j) / 2`, identical to `rank_with_ties`' mid-rank. Entries
/// whose group is out-of-range (unknown-group cells) still occupy sorted
/// positions (so tie mid-ranks and the correction match the dense ranking) but
/// contribute to no group's rank-sum — they are "rest" for every group.
fn assign_block_ranks(
    block: &[(f64, usize)],
    offset: usize,
    n_groups: usize,
    rank_sum: &mut [f64],
    tie_correction: &mut f64,
) {
    let n = block.len();
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && block[j].0 == block[i].0 {
            j += 1;
        }
        let mid_rank = (2 * offset + i + 1 + j) as f64 / 2.0;
        for entry in &block[i..j] {
            let g = entry.1;
            if g < n_groups {
                rank_sum[g] += mid_rank;
            }
        }
        let t = (j - i) as f64;
        if t > 1.0 {
            *tie_correction += t * t * t - t;
        }
        i = j;
    }
}

/// Per-gene 1-vs-rest Wilcoxon `(score, pval, logfc)` for every group, computed
/// from a single gene's nonzeros (`rows`/`vals`) plus the implicit-zero block —
/// the exact sparse analogue of the dense kernel's per-gene arm. `total_sum`
/// (for logFC's reference mean) **excludes** unknown-group cells, and `n2`
/// **includes** them, matching `wilcoxon_rank_sum`'s 1-vs-rest arm exactly.
#[allow(clippy::too_many_arguments)]
fn gene_stats_nnz(
    rows: &[i32],
    vals: &[f32],
    groups: &[usize],
    group_cell_counts: &[usize],
    n_obs: usize,
    n_groups: usize,
    tie_correct: bool,
    log_transformed: bool,
) -> Vec<(f64, f64, f64)> {
    let mut group_sum = vec![0.0f64; n_groups];
    let mut nonzero_in_g = vec![0usize; n_groups];
    let mut neg: Vec<(f64, usize)> = Vec::new();
    let mut pos: Vec<(f64, usize)> = Vec::new();
    let mut total_sum = 0.0f64;

    for (&row, &v32) in rows.iter().zip(vals.iter()) {
        let row = row as usize;
        if row >= n_obs {
            continue;
        }
        let v = v32 as f64;
        let g = groups[row];
        if g < n_groups {
            // Dense parity: `total` and the per-group sums exclude unknown-group
            // cells; those cells still count toward `n2` via `group_cell_counts`.
            total_sum += v;
            group_sum[g] += v;
            if v != 0.0 {
                nonzero_in_g[g] += 1;
            }
        }
        if v < 0.0 {
            neg.push((v, g));
        } else if v > 0.0 {
            pos.push((v, g));
        }
        // v == 0.0 (explicit stored zero) joins the implicit-zero block.
    }

    let n_neg = neg.len();
    let n_pos = pos.len();
    // Each stored entry is one distinct cell (one entry per (cell,gene) in CSC),
    // and out-of-range rows are dropped above, so nonzeros never exceed n_obs.
    // A corrupt sidecar with duplicate rows in a column would break this.
    debug_assert!(
        n_neg + n_pos <= n_obs,
        "nnz ({}) exceeds n_obs ({n_obs}) — duplicate rows in a CSC column?",
        n_neg + n_pos
    );
    let n_zero_total = n_obs - n_neg - n_pos;
    // Mid-rank of the zero tie-block spanning 1-based ranks [n_neg+1 .. n_neg+n_zero].
    let zero_mid = n_neg as f64 + (n_zero_total as f64 + 1.0) / 2.0;

    let mut rank_sum = vec![0.0f64; n_groups];
    let mut tie_correction = 0.0f64;

    neg.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    assign_block_ranks(&neg, 0, n_groups, &mut rank_sum, &mut tie_correction);

    if n_zero_total > 1 {
        let t = n_zero_total as f64;
        tie_correction += t * t * t - t;
    }

    pos.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    assign_block_ranks(
        &pos,
        n_neg + n_zero_total,
        n_groups,
        &mut rank_sum,
        &mut tie_correction,
    );

    // Zero cells (implicit + explicit) per group all carry the zero mid-rank.
    for g in 0..n_groups {
        let n_zero_cells_g = group_cell_counts[g] - nonzero_in_g[g];
        rank_sum[g] += n_zero_cells_g as f64 * zero_mid;
    }

    let mut out = Vec::with_capacity(n_groups);
    for g in 0..n_groups {
        let n1 = group_cell_counts[g];
        if n1 == 0 || n1 == n_obs {
            out.push((f64::NAN, 1.0, f64::NAN));
            continue;
        }
        let tc = if tie_correct { tie_correction } else { 0.0 };
        let (score, pval) = wilcoxon_stats_from_rank_sum(rank_sum[g], n1, n_obs, tc);
        let mean_group = group_sum[g] / n1 as f64;
        let mean_ref = (total_sum - group_sum[g]) / (n_obs - n1) as f64;
        let logfc = compute_logfc(mean_group, mean_ref, log_transformed);
        out.push((score, pval, logfc));
    }
    out
}

/// Exact sparse-nnz 1-vs-rest Wilcoxon over a CSC source (§5.3).
///
/// Numerically equivalent to `wilcoxon_rank_sum` (1-vs-rest arm) — ranks only
/// each gene's nonzeros plus an analytic implicit-zero tie-block, in
/// `O(nnz·log nnz)` per gene instead of an `O(n_obs·log n_obs)` dense sort.
/// Reads columns in chunks (bounded memory: only the current chunk's CSC slab +
/// the small per-gene result matrix), then does one global sort + BH assembly.
fn wilcoxon_rank_sum_nnz_csc<S: ColumnShardSource + ?Sized>(
    source: &S,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    gene_chunk_size: usize,
    log_transformed: bool,
    tie_correct: bool,
) -> Result<DiffExpResult> {
    let n_obs = source.n_obs();
    let n_vars = gene_names.len();
    let n_groups = group_names.len();

    // Per-group cell counts (unknown-group cells excluded), matching the dense
    // kernel's `group_indices[g].len()`.
    let mut group_cell_counts = vec![0usize; n_groups];
    for &g in groups {
        if g < n_groups {
            group_cell_counts[g] += 1;
        }
    }

    // per_gene[gene][group] = (score, pval, logfc).
    let mut per_gene: Vec<Vec<(f64, f64, f64)>> = vec![Vec::new(); n_vars];

    use rayon::prelude::*;
    for chunk_start in (0..n_vars).step_by(gene_chunk_size) {
        let chunk_end = (chunk_start + gene_chunk_size).min(n_vars);
        let csc = source
            .read_csc_columns(chunk_start as u32..chunk_end as u32)
            .map_err(AccelError::Scx)?;
        // Finiteness contract at the DE boundary, same as the dense kernel.
        crate::finite::ensure_finite_values(&csc.data, "differential expression")?;
        let chunk_size = chunk_end - chunk_start;
        // Genes are independent, so rank them in parallel (matches the dense
        // kernel's rayon-over-genes; keeps the nnz path from looking slow only
        // because it was single-threaded — gemini/Cursor review). Each result is
        // self-contained, so order does not affect the output.
        let chunk_results: Vec<Vec<(f64, f64, f64)>> = (0..chunk_size)
            .into_par_iter()
            .map(|local_col| {
                let s = csc.indptr[local_col] as usize;
                let e = csc.indptr[local_col + 1] as usize;
                gene_stats_nnz(
                    &csc.indices[s..e],
                    &csc.data[s..e],
                    groups,
                    &group_cell_counts,
                    n_obs,
                    n_groups,
                    tie_correct,
                    log_transformed,
                )
            })
            .collect();
        for (local_col, res) in chunk_results.into_iter().enumerate() {
            per_gene[chunk_start + local_col] = res;
        }
    }

    // Assemble per-group sorted gene lists + global BH (ref=None → all groups
    // tested), mirroring `wilcoxon_rank_sum`'s tail with `gene_index_base = 0`.
    let mut result_names = Vec::with_capacity(n_groups);
    let mut result_gene_indices = Vec::with_capacity(n_groups);
    let mut result_scores = Vec::with_capacity(n_groups);
    let mut result_pvals = Vec::with_capacity(n_groups);
    let mut result_pvals_adj = Vec::with_capacity(n_groups);
    let mut result_logfc = Vec::with_capacity(n_groups);

    #[allow(clippy::needless_range_loop)]
    for g in 0..n_groups {
        // (gene_idx, score, pval, logfc); global_idx == gene_idx (base 0).
        let mut sorted: Vec<(usize, f64, f64, f64)> = (0..n_vars)
            .map(|vi| {
                let (s, p, l) = per_gene[vi][g];
                (vi, s, p, l)
            })
            .collect();
        sorted.sort_by(|a, b| de_rank_cmp(a.1, a.0, b.1, b.0, false));

        result_names.push(sorted.iter().map(|r| gene_names[r.0].clone()).collect());
        result_gene_indices.push(sorted.iter().map(|r| r.0).collect::<Vec<_>>());
        let scores: Vec<f64> = sorted.iter().map(|r| r.1).collect();
        let pvals: Vec<f64> = sorted.iter().map(|r| r.2).collect();
        result_logfc.push(sorted.iter().map(|r| r.3).collect::<Vec<_>>());
        result_pvals_adj.push(benjamini_hochberg(&pvals));
        result_scores.push(scores);
        result_pvals.push(pvals);
    }

    Ok(DiffExpResult {
        group_names: group_names.to_vec(),
        names: result_names,
        gene_indices: result_gene_indices,
        scores: result_scores,
        pvals: result_pvals,
        pvals_adj: result_pvals_adj,
        logfoldchanges: result_logfc,
        exec_info: crate::route::AccelExecutionInfo::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csc::test_helpers::{deterministic_dense, write_csr_csc_test_file};
    use crate::diffexp::wilcoxon_rank_sum_streaming;
    use scx_format_io::{BackedCscReader, BackedCsrReader, ScxReader};
    use tempfile::tempdir;

    #[test]
    fn wilcoxon_csc_matches_csr() {
        let dir = tempdir().unwrap();
        let n_obs = 16usize;
        let n_vars = 8usize;
        let dense = deterministic_dense(n_obs, n_vars);
        let path = write_csr_csc_test_file(dir.path(), "wilcox", n_obs, n_vars, &dense, 4);

        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        let group_names = vec!["A".to_string(), "B".to_string()];
        // Half of cells in group 0, half in group 1.
        let groups: Vec<usize> = (0..n_obs)
            .map(|i| if i < n_obs / 2 { 0 } else { 1 })
            .collect();

        let csr_reader = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);
        let csr_res = wilcoxon_rank_sum_streaming(
            &csr_reader,
            &gene_names,
            &groups,
            &group_names,
            None,
            4, // gene_chunk_size
            false,
            false,
            false,
        )
        .unwrap();

        let csc_reader = BackedCscReader::new(ScxReader::open(&path).unwrap(), 0).unwrap();
        let csc_res = wilcoxon_rank_sum_streaming_csc(
            &csc_reader,
            &gene_names,
            &groups,
            &group_names,
            None,
            4,
            false,
            false,
            false,
        )
        .unwrap();

        // Compare per-group sorted score vectors. Names ordering must
        // match because the underlying sort is by score and inputs are
        // identical.
        assert_eq!(csr_res.group_names, csc_res.group_names);
        for g in 0..csr_res.group_names.len() {
            assert_eq!(csr_res.names[g], csc_res.names[g], "group {g} gene order");
            for k in 0..csr_res.scores[g].len() {
                assert!(
                    (csr_res.scores[g][k] - csc_res.scores[g][k]).abs() < 1e-9,
                    "group {g} score[{k}] mismatch"
                );
                assert!(
                    (csr_res.pvals[g][k] - csc_res.pvals[g][k]).abs() < 1e-9,
                    "group {g} pval[{k}] mismatch"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // GPU CSC-direct / CSR-direct v3 Wilcoxon parity (§C.9). All gated on a
    // CUDA device; the binary still links without one. Compared by gene name
    // (not slot index) because `DiffExpResult` is score-sorted and GPU fp32
    // vs CPU f64 can re-order ties.
    // -----------------------------------------------------------------------
    #[cfg(feature = "gpu")]
    use crate::diffexp::DiffExpResult;
    #[cfg(feature = "gpu")]
    use crate::diffexp::{wilcoxon_rank_sum_gpu, GpuDeShardInput};
    #[cfg(feature = "gpu")]
    use crate::route::AccelRoute;
    #[cfg(feature = "gpu")]
    use std::collections::HashMap;

    /// `(group, gene) -> (score, pval, logfc)` lookup, order-independent.
    #[cfg(feature = "gpu")]
    fn de_by_gene(res: &DiffExpResult) -> HashMap<(String, String), (f64, f64, f64)> {
        let mut m = HashMap::new();
        for g in 0..res.group_names.len() {
            for k in 0..res.names[g].len() {
                m.insert(
                    (res.group_names[g].clone(), res.names[g][k].clone()),
                    (res.scores[g][k], res.pvals[g][k], res.logfoldchanges[g][k]),
                );
            }
        }
        m
    }

    /// Assert GPU vs CPU Wilcoxon results agree within fp32-GPU tolerance,
    /// matching by `(group, gene)` and handling NaN (empty groups / undefined
    /// logFC) symmetrically.
    #[cfg(feature = "gpu")]
    fn assert_de_close(cpu: &DiffExpResult, gpu: &DiffExpResult, ctx: &str) {
        assert_eq!(cpu.group_names, gpu.group_names, "{ctx}: group_names");
        let c = de_by_gene(cpu);
        let g = de_by_gene(gpu);
        assert_eq!(c.len(), g.len(), "{ctx}: result entry count");
        for (key, &(sc, pc, lc)) in &c {
            let &(sg, pg, lg) = g
                .get(key)
                .unwrap_or_else(|| panic!("{ctx}: GPU missing {key:?}"));
            // Z-score (signed). NaN-aware.
            if sc.is_finite() && sg.is_finite() {
                let d = (sc - sg).abs();
                assert!(
                    d < 1e-3 || d / sc.abs().max(1e-12) < 1e-3,
                    "{ctx}: score {key:?} cpu={sc} gpu={sg}"
                );
            } else {
                assert_eq!(sc.is_nan(), sg.is_nan(), "{ctx}: score NaN-ness {key:?}");
            }
            // Two-sided p-value.
            let pdiff = (pc - pg).abs();
            assert!(
                pdiff < 1e-6 || pdiff / pc.abs().max(1e-30) < 1e-3,
                "{ctx}: pval {key:?} cpu={pc} gpu={pg}"
            );
            // log2 fold-change. NaN-aware.
            if lc.is_finite() && lg.is_finite() {
                let d = (lc - lg).abs();
                assert!(
                    d < 1e-3 || d / lc.abs().max(1e-12) < 1e-3,
                    "{ctx}: logfc {key:?} cpu={lc} gpu={lg}"
                );
            } else {
                assert_eq!(lc.is_nan(), lg.is_nan(), "{ctx}: logfc NaN-ness {key:?}");
            }
        }
    }

    /// Skip helper: returns true (and prints) when no CUDA device is present.
    #[cfg(feature = "gpu")]
    fn no_gpu() -> bool {
        if scx_gpu::device::GpuDevice::new(0).is_err() {
            eprintln!("CUDA not available — skipping GPU Wilcoxon v3 parity test");
            true
        } else {
            false
        }
    }

    /// Run the CPU CSR streaming baseline for the given fixture/params.
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)]
    fn cpu_baseline(
        path: &std::path::Path,
        gene_names: &[String],
        groups: &[usize],
        group_names: &[String],
        reference: Option<usize>,
        chunk: usize,
    ) -> DiffExpResult {
        let csr_reader = BackedCsrReader::new(ScxReader::open(path).unwrap(), 0);
        wilcoxon_rank_sum_streaming(
            &csr_reader,
            gene_names,
            groups,
            group_names,
            reference,
            chunk,
            false,
            false,
            true,
        )
        .expect("CPU streaming Wilcoxon failed")
    }

    /// Run the GPU backed Wilcoxon path with v3 forced on. `with_csc` selects
    /// the CSC-direct route (sidecar provided) vs the CSR-direct fallback.
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)]
    fn gpu_v3(
        path: &std::path::Path,
        gene_names: &[String],
        groups: &[usize],
        group_names: &[String],
        reference: Option<usize>,
        chunk: usize,
        with_csc: bool,
    ) -> DiffExpResult {
        let csr_reader = BackedCsrReader::new(ScxReader::open(path).unwrap(), 0);
        let csc_reader = if with_csc {
            Some(BackedCscReader::new(ScxReader::open(path).unwrap(), 0).unwrap())
        } else {
            None
        };
        // v3 is the unconditional default since the V1b flip — no override.
        wilcoxon_rank_sum_gpu(
            0,
            GpuDeShardInput::Backed {
                csr: &csr_reader,
                csc: csc_reader.as_ref(),
            },
            gene_names,
            groups,
            group_names,
            reference,
            Some(chunk),
            false,
            false,
            true,
        )
        .expect("GPU v3 Wilcoxon failed")
    }

    #[cfg(feature = "gpu")]
    fn three_group_fixture(
        dir: &std::path::Path,
        name: &str,
        cols_per_csc_shard: usize,
    ) -> (std::path::PathBuf, Vec<String>, Vec<usize>, Vec<String>) {
        let n_obs = 64usize;
        let n_vars = 20usize;
        let dense = deterministic_dense(n_obs, n_vars);
        let path = write_csr_csc_test_file(dir, name, n_obs, n_vars, &dense, cols_per_csc_shard);
        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        let groups: Vec<usize> = (0..n_obs).map(|i| (i * 3) / n_obs).collect();
        let group_names = vec!["ref".to_string(), "ko_a".to_string(), "ko_b".to_string()];
        (path, gene_names, groups, group_names)
    }

    /// (1) CSC-direct vs CPU streaming — 1-vs-rest.
    #[cfg(feature = "gpu")]
    #[test]
    fn wilcoxon_gpu_v3_csc_one_vs_rest() {
        if no_gpu() {
            return;
        }
        let dir = tempdir().unwrap();
        let (path, gn, gr, names) = three_group_fixture(dir.path(), "wil_csc_ovr", 7);
        let cpu = cpu_baseline(&path, &gn, &gr, &names, None, 7);
        let gpu = gpu_v3(&path, &gn, &gr, &names, None, 7, true);
        assert_de_close(&cpu, &gpu, "csc 1-vs-rest");
        assert_eq!(gpu.exec_info.route, AccelRoute::GpuCscV3);
    }

    /// (2) CSC-direct vs CPU streaming — ref-mode.
    #[cfg(feature = "gpu")]
    #[test]
    fn wilcoxon_gpu_v3_csc_ref_mode() {
        if no_gpu() {
            return;
        }
        let dir = tempdir().unwrap();
        let (path, gn, gr, names) = three_group_fixture(dir.path(), "wil_csc_ref", 7);
        let cpu = cpu_baseline(&path, &gn, &gr, &names, Some(0), 7);
        let gpu = gpu_v3(&path, &gn, &gr, &names, Some(0), 7, true);
        assert_de_close(&cpu, &gpu, "csc ref-mode");
        assert_eq!(gpu.exec_info.route, AccelRoute::GpuCscV3);
    }

    /// (3) CSR-direct fallback (no CSC sidecar) vs CPU streaming — 1-vs-rest.
    #[cfg(feature = "gpu")]
    #[test]
    fn wilcoxon_gpu_v3_csr_fallback() {
        if no_gpu() {
            return;
        }
        let dir = tempdir().unwrap();
        let (path, gn, gr, names) = three_group_fixture(dir.path(), "wil_csr_fb", 7);
        let cpu = cpu_baseline(&path, &gn, &gr, &names, None, 7);
        let gpu = gpu_v3(&path, &gn, &gr, &names, None, 7, false);
        assert_de_close(&cpu, &gpu, "csr fallback 1-vs-rest");
        assert_eq!(gpu.exec_info.route, AccelRoute::GpuCsrV3);
    }

    /// (4) Route metadata: CSC sidecar → GpuCscV3; none → GpuCsrV3 (v3 on).
    #[cfg(feature = "gpu")]
    #[test]
    fn wilcoxon_gpu_v3_route_metadata() {
        if no_gpu() {
            return;
        }
        let dir = tempdir().unwrap();
        let (path, gn, gr, names) = three_group_fixture(dir.path(), "wil_route", 7);
        let csc = gpu_v3(&path, &gn, &gr, &names, Some(0), 7, true);
        assert_eq!(
            csc.exec_info.route,
            AccelRoute::GpuCscV3,
            "csc → gpu_csc_v3"
        );
        let csr = gpu_v3(&path, &gn, &gr, &names, Some(0), 7, false);
        assert_eq!(
            csr.exec_info.route,
            AccelRoute::GpuCsrV3,
            "no csc → gpu_csr_v3"
        );
    }

    /// (5) Empty test group — must produce NaN score / p=1 matching CPU.
    #[cfg(feature = "gpu")]
    #[test]
    fn wilcoxon_gpu_v3_csc_empty_group() {
        if no_gpu() {
            return;
        }
        let dir = tempdir().unwrap();
        let (path, gn, gr, _names) = three_group_fixture(dir.path(), "wil_empty", 7);
        // 4th group label with zero members.
        let names = vec![
            "ref".to_string(),
            "ko_a".to_string(),
            "ko_b".to_string(),
            "empty".to_string(),
        ];
        let cpu = cpu_baseline(&path, &gn, &gr, &names, Some(0), 7);
        let gpu = gpu_v3(&path, &gn, &gr, &names, Some(0), 7, true);
        assert_de_close(&cpu, &gpu, "csc empty group");
    }

    /// (6) All-zero gene column — defined U / p=1 for the constant column.
    #[cfg(feature = "gpu")]
    #[test]
    fn wilcoxon_gpu_v3_csc_all_zero_gene() {
        if no_gpu() {
            return;
        }
        let n_obs = 48usize;
        let n_vars = 12usize;
        let mut dense = deterministic_dense(n_obs, n_vars);
        // Force gene 5 to all zeros.
        for r in 0..n_obs {
            dense[r * n_vars + 5] = 0;
        }
        let dir = tempdir().unwrap();
        let path = write_csr_csc_test_file(dir.path(), "wil_zero", n_obs, n_vars, &dense, 5);
        let gn: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        let gr: Vec<usize> = (0..n_obs).map(|i| (i * 2) / n_obs).collect();
        let names = vec!["a".to_string(), "b".to_string()];
        let cpu = cpu_baseline(&path, &gn, &gr, &names, None, 5);
        let gpu = gpu_v3(&path, &gn, &gr, &names, None, 5, true);
        assert_de_close(&cpu, &gpu, "csc all-zero gene");
    }

    /// (7) Tie-spanning gene — constant value across all cells (max ties).
    #[cfg(feature = "gpu")]
    #[test]
    fn wilcoxon_gpu_v3_csc_tie_spanning_gene() {
        if no_gpu() {
            return;
        }
        let n_obs = 48usize;
        let n_vars = 12usize;
        let mut dense = deterministic_dense(n_obs, n_vars);
        // Gene 3 constant (every cell = 5) → fully tied column.
        for r in 0..n_obs {
            dense[r * n_vars + 3] = 5;
        }
        let dir = tempdir().unwrap();
        let path = write_csr_csc_test_file(dir.path(), "wil_tie", n_obs, n_vars, &dense, 5);
        let gn: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        let gr: Vec<usize> = (0..n_obs).map(|i| (i * 2) / n_obs).collect();
        let names = vec!["a".to_string(), "b".to_string()];
        let cpu = cpu_baseline(&path, &gn, &gr, &names, Some(0), 5);
        let gpu = gpu_v3(&path, &gn, &gr, &names, Some(0), 5, true);
        assert_de_close(&cpu, &gpu, "csc tie-spanning gene");
    }

    /// (8) Multi-shard CSC: `cols_per_csc_shard=4`, `n_vars=20` → 5 shards;
    /// `gene_chunk_size=7` spans shard boundaries (exercises range prefilter).
    #[cfg(feature = "gpu")]
    #[test]
    fn wilcoxon_gpu_v3_csc_multi_shard() {
        if no_gpu() {
            return;
        }
        let dir = tempdir().unwrap();
        let (path, gn, gr, names) = three_group_fixture(dir.path(), "wil_multishard", 4);
        let cpu = cpu_baseline(&path, &gn, &gr, &names, None, 7);
        let gpu = gpu_v3(&path, &gn, &gr, &names, None, 7, true);
        assert_de_close(&cpu, &gpu, "csc multi-shard");

        // §B.10 criterion 4: range prefiltering decoded strictly fewer CSC
        // shards than the no-prefilter worst case (n_csc_shards × n_chunks).
        let n_vars = gn.len();
        let n_chunks = n_vars.div_ceil(7); // gene_chunk_size = 7
        let n_csc_shards = n_vars.div_ceil(4); // cols_per_csc_shard = 4
        let decoded = gpu
            .exec_info
            .shards_decoded
            .expect("shards_decoded recorded on v3 CSC route");
        assert!(
            decoded > 0 && decoded < n_csc_shards * n_chunks,
            "expected prefiltered shard count in (0, {}), got {decoded}",
            n_csc_shards * n_chunks
        );
    }

    // (Former test (9) "v3 CSC-direct vs v1 dense-chunk" was removed with the
    // V1b default-flip: v1 is no longer reachable through `plan_de_route`, so it
    // cannot be selected in-process for an A/B comparison. v3-vs-CPU parity
    // across modes/edge-cases is covered by the tests above.)

    // -----------------------------------------------------------------------
    // §5.3 exact sparse-nnz Wilcoxon: property tests vs the dense reference.
    // -----------------------------------------------------------------------

    use proptest::prelude::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use scx_sparse::ScxCsc;

    /// In-memory CSC source for the nnz property test. Stores per-column
    /// `(row, value)` entries verbatim — **including explicit zeros** — so the
    /// fast path's zero handling is exercised, and builds a column-range slab on
    /// demand (local column `i` ↔ global column `start + i`).
    struct InMemCsc {
        cols: Vec<Vec<(i32, f32)>>,
        n_obs: usize,
    }

    impl ColumnShardSource for InMemCsc {
        fn n_csc_shards(&self) -> usize {
            1
        }
        fn n_obs(&self) -> usize {
            self.n_obs
        }
        fn n_vars(&self) -> usize {
            self.cols.len()
        }
        fn read_csc_shard(&self, _i: usize) -> scx_format_io::Result<ScxCsc> {
            self.read_csc_columns(0..self.cols.len() as u32)
        }
        fn read_csc_columns(&self, r: std::ops::Range<u32>) -> scx_format_io::Result<ScxCsc> {
            let (a, b) = (r.start as usize, r.end as usize);
            let mut indptr = vec![0i64];
            let mut indices = Vec::new();
            let mut data = Vec::new();
            for col in &self.cols[a..b] {
                for &(row, v) in col {
                    indices.push(row);
                    data.push(v);
                }
                indptr.push(indices.len() as i64);
            }
            Ok(ScxCsc::new_unchecked(
                (self.n_obs, b - a),
                indptr,
                indices,
                data,
            ))
        }
        fn csc_shard_col_range(&self, _i: usize) -> Option<(u32, u32)> {
            Some((0, self.cols.len() as u32))
        }
    }

    /// Tolerance compare that treats NaN==NaN as equal (empty-group results are
    /// NaN in both paths).
    fn close(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 + 1e-6 * b.abs()
    }

    /// name → (score, pval, pval_adj, logfc) for one group of a result.
    fn group_map(
        res: &DiffExpResult,
        gi: usize,
    ) -> std::collections::HashMap<String, (f64, f64, f64, f64)> {
        let mut m = std::collections::HashMap::new();
        for i in 0..res.names[gi].len() {
            m.insert(
                res.names[gi][i].clone(),
                (
                    res.scores[gi][i],
                    res.pvals[gi][i],
                    res.pvals_adj[gi][i],
                    res.logfoldchanges[gi][i],
                ),
            );
        }
        m
    }

    /// End-to-end integration: the nnz CSC kernel matches the CSR streaming
    /// path on a real backed SCX file (codec round-trip) with an unknown-group
    /// cell class present.
    #[test]
    fn nnz_csc_matches_csr_streaming_on_file() {
        let dir = tempdir().unwrap();
        let (n_obs, n_vars) = (20usize, 10usize);
        let dense = deterministic_dense(n_obs, n_vars);
        let path = write_csr_csc_test_file(dir.path(), "nnzfile", n_obs, n_vars, &dense, 5);

        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        let group_names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        // Cycle 0,1,2,3 → index 3 is the unknown-group sentinel (n_groups == 3).
        let groups: Vec<usize> = (0..n_obs).map(|i| i % 4).collect();

        let csr_reader = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);
        let csr_res = wilcoxon_rank_sum_streaming(
            &csr_reader,
            &gene_names,
            &groups,
            &group_names,
            None,
            4,
            false,
            false,
            true,
        )
        .unwrap();

        let csc_reader = BackedCscReader::new(ScxReader::open(&path).unwrap(), 0).unwrap();
        let nnz_res = wilcoxon_rank_sum_nnz_csc(
            &csc_reader,
            &gene_names,
            &groups,
            &group_names,
            4,
            false,
            true,
        )
        .unwrap();

        assert_eq!(csr_res.group_names, nnz_res.group_names);
        for gi in 0..group_names.len() {
            let dmap = group_map(&csr_res, gi);
            let nmap = group_map(&nnz_res, gi);
            for (name, &(ds, dp, dpa, dl)) in &dmap {
                let &(ns, np, npa, nl) = nmap.get(name).unwrap();
                assert!(close(ds, ns), "score g{gi} {name}: {ds} vs {ns}");
                assert!(close(dp, np), "pval g{gi} {name}: {dp} vs {np}");
                assert!(close(dpa, npa), "padj g{gi} {name}: {dpa} vs {npa}");
                assert!(close(dl, nl), "logfc g{gi} {name}: {dl} vs {nl}");
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// The exact sparse-nnz 1-vs-rest path matches the dense kernel on random
        /// small matrices with negatives, explicit + implicit zeros, unknown-group
        /// cells, and both tie-correction settings — the §5.3 exactness contract.
        #[test]
        fn nnz_matches_dense_wilcoxon(
            seed in any::<u64>(),
            n_obs in 6usize..40,
            n_vars in 2usize..8,
            n_groups in 2usize..4,
            log_transformed in any::<bool>(),
            tie_correct in any::<bool>(),
        ) {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);

            // Group per cell in 0..=n_groups; `n_groups` is the unknown-group
            // sentinel (dense drops it from all groups, keeps it in "rest").
            let groups: Vec<usize> = (0..n_obs).map(|_| rng.gen_range(0..=n_groups)).collect();

            // Dense buffer + verbatim CSC columns. ~55% density; values are small
            // signed ints, occasionally exactly 0 (explicit stored zero).
            let mut dense = vec![0.0f32; n_obs * n_vars];
            let mut cols: Vec<Vec<(i32, f32)>> = vec![Vec::new(); n_vars];
            for (c, col) in cols.iter_mut().enumerate() {
                for row in 0..n_obs {
                    if rng.gen::<f64>() < 0.45 {
                        continue; // implicit zero
                    }
                    let v = rng.gen_range(-3i32..=3) as f32;
                    dense[row * n_vars + c] = v;
                    col.push((row as i32, v)); // may be an explicit zero
                }
            }

            let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
            let group_names: Vec<String> = (0..n_groups).map(|g| format!("grp{g}")).collect();

            let dense_res = wilcoxon_rank_sum(
                &dense, n_obs, n_vars, &gene_names, &groups, &group_names,
                None, log_transformed, false, tie_correct, 0,
            ).unwrap();

            let src = InMemCsc { cols, n_obs };
            let nnz_res = wilcoxon_rank_sum_nnz_csc(
                &src, &gene_names, &groups, &group_names, 3, log_transformed, tie_correct,
            ).unwrap();

            prop_assert_eq!(&dense_res.group_names, &nnz_res.group_names);
            for gi in 0..group_names.len() {
                let dmap = group_map(&dense_res, gi);
                let nmap = group_map(&nnz_res, gi);
                for (name, &(ds, dp, dpa, dl)) in &dmap {
                    let &(ns, np, npa, nl) = nmap.get(name).unwrap();
                    prop_assert!(close(ds, ns), "score g{} {}: {} vs {}", gi, name, ds, ns);
                    prop_assert!(close(dp, np), "pval g{} {}: {} vs {}", gi, name, dp, np);
                    prop_assert!(close(dpa, npa), "padj g{} {}: {} vs {}", gi, name, dpa, npa);
                    prop_assert!(close(dl, nl), "logfc g{} {}: {} vs {}", gi, name, dl, nl);
                }
            }
        }
    }
}
