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

use crate::diffexp::cpu::{
    compute_logfc, de_rank_cmp, for_each_tie_run, wilcoxon_stats_from_rank_sum,
};
use crate::diffexp::{
    benjamini_hochberg, merge_diff_exp_results, wilcoxon_rank_sum, DiffExpResult,
};
use crate::error::{AccelError, Result};
use scx_format_io::ColumnShardSource;

/// Parse of the `SCX_ACCEL_WILCOXON_NNZ` gate, split out from the `OnceLock`
/// read below so the accepted spellings are unit-testable. The process-global
/// cache makes [`nnz_wilcoxon_enabled`] itself untestable from a test binary
/// that has already read it once.
///
/// Only an explicit off spelling turns the kernel off; unset, and any other
/// value, leave the default on.
fn nnz_gate_from_env_str(raw: Option<&str>) -> bool {
    !matches!(raw, Some("0" | "false" | "FALSE" | "off"))
}

/// Gate for the exact sparse-nnz Wilcoxon kernel (§5.3). **On by default**;
/// `SCX_ACCEL_WILCOXON_NNZ=0` falls back to the densify kernel, as a
/// same-build A/B arm and a kill switch.
///
/// It ranks only each gene's nonzeros plus a synthesized implicit-zero
/// tie-block instead of sorting an `n_obs`-length dense column per gene, and
/// is bit-identical to the densify kernel — scores, p-values, adjusted
/// p-values, log fold changes and gene order, pinned at zero tolerance by
/// `nnz_matches_the_densify_kernel_exactly_on_tie_heavy_counts`. It was
/// promoted after measuring 3.1x (tabula_sapiens_100k, 19.6 s -> 6.2 s) and
/// 4.1x (census_1m, 225.9 s -> 55.7 s) over densify, at the same or lower peak
/// RSS; until then it was opt-in with `=1`.
fn nnz_wilcoxon_enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        nnz_gate_from_env_str(std::env::var("SCX_ACCEL_WILCOXON_NNZ").ok().as_deref())
    })
}

/// Whether a CSC Wilcoxon call with these arguments takes the exact sparse-nnz
/// kernel rather than the densify path.
///
/// **The** selection rule, in one place. [`wilcoxon_rank_sum_streaming_csc`]
/// branches on it and the pyscx dispatcher stamps the route from it, so the
/// recorded route cannot disagree with the kernel that ran (review §7.17: the
/// env gate swapped in a structurally different kernel while the stamp said
/// `cpu_csc` either way, which is the "benchmark one route while believing
/// another ran" class the planner exists to close).
///
/// Note the gate alone is not the answer: the nnz kernel is 1-vs-rest only, so
/// `rankby_abs` or an explicit `reference` keeps the densify path whatever the
/// gate says.
pub fn csc_wilcoxon_uses_nnz_kernel(reference: Option<usize>, rankby_abs: bool) -> bool {
    csc_wilcoxon_selects_nnz(nnz_wilcoxon_enabled(), reference, rankby_abs)
}

/// The selection rule itself, with the env gate supplied rather than read.
///
/// Split out purely so it is testable: [`nnz_wilcoxon_enabled`] caches in a
/// `OnceLock`, so a test binary that has resolved it once cannot exercise the
/// other arm — and a test that reimplements the rule locally to get around that
/// pins its own copy instead of this one, which makes it structurally incapable
/// of catching a regression here. Found by Antigravity - Gemini 3.8 Flash.
pub(crate) fn csc_wilcoxon_selects_nnz(
    gate: bool,
    reference: Option<usize>,
    rankby_abs: bool,
) -> bool {
    gate && reference.is_none() && !rankby_abs
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
    // Exact sparse-nnz kernel (§5.3), the default for 1-vs-rest (rankby_abs
    // and an explicit reference keep the densify path). Ranks only the nonzeros +
    // an analytic implicit-zero tie-block — no `n_obs` dense column per gene.
    if csc_wilcoxon_uses_nnz_kernel(reference, rankby_abs) {
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
    wilcoxon_rank_sum_densify_csc(
        source,
        gene_names,
        groups,
        group_names,
        reference,
        gene_chunk_size,
        log_transformed,
        rankby_abs,
        tie_correct,
    )
}

/// The densify kernel: each gene chunk's CSC slab scattered into a row-major
/// dense buffer and ranked by the dense Wilcoxon kernel.
///
/// What [`wilcoxon_rank_sum_streaming_csc`] runs for a `reference=` or
/// `rankby_abs` call, and for 1-vs-rest under `SCX_ACCEL_WILCOXON_NNZ=0`. A
/// function of its own so the exact-parity test can call it without going
/// through the process-global gate. Arguments are validated by the caller.
#[allow(clippy::too_many_arguments)]
fn wilcoxon_rank_sum_densify_csc<S: ColumnShardSource + ?Sized>(
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
/// *before* this block (0 for negatives; `n_neg + n_zero` for positives).
///
/// The walk itself — runs, mid-ranks, `Σ(t³−t)` — is
/// [`crate::diffexp::cpu::for_each_tie_run`], shared with the
/// dense path's `rank_with_ties`. It used to be a second copy of that loop, and
/// the two agreeing was an assertion in a doc comment rather than a fact about
/// the code; all this function supplies now is *where a run's rank goes*.
///
/// An entry's second field is its **pool bucket**
/// (`crate::diffexp::groups::pool_bucket`), not its group: an unlabelled cell is
/// in the pool — it occupies a rank and shifts everyone else's — but its rank
/// mass lands in the sentinel slot, which no group ever reads back.
fn assign_block_ranks(
    block: &[(f64, usize)],
    offset: usize,
    rank_sum: &mut [f64],
    tie_correction: &mut f64,
) {
    *tie_correction += for_each_tie_run(
        block.len(),
        offset,
        |i, j| block[j].0 == block[i].0,
        |mid_rank, run| {
            for entry in &block[run] {
                rank_sum[entry.1] += mid_rank;
            }
        },
    );
}

/// Per-gene 1-vs-rest Wilcoxon `(score, pval, logfc)` for every group, computed
/// from a single gene's nonzeros (`rows`/`vals`) plus the implicit-zero block —
/// the exact sparse analogue of the dense kernel's per-gene arm.
///
/// The comparison pool is every cell (`n_obs`). An unlabelled cell joins no
/// group's sum or count, but its nonzeros do enter the sorted blocks and it does
/// belong to the implicit-zero block, because it is in every group's "rest" —
/// scanpy's `X[~mask_g]`. Same rule as `wilcoxon_rank_sum`'s 1-vs-rest arm; see
/// `crate::diffexp::groups`.
///
/// `group_cell_counts` is `n_groups + 1` long: the last entry is the unlabelled
/// count, needed so the sentinel's share of the implicit-zero block is accounted
/// for rather than silently folded into some group's.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gene_stats_nnz(
    rows: &[i32],
    vals: &[f32],
    groups: &[usize],
    group_cell_counts: &[usize],
    n_obs: usize,
    n_groups: usize,
    tie_correct: bool,
    log_transformed: bool,
) -> Result<Vec<(f64, f64, f64)>> {
    // A runtime check, not a `debug_assert`: the per-bucket loop below indexes
    // `group_cell_counts[n_groups]`, so a short slice is an out-of-bounds panic
    // in exactly the release builds where the assert is compiled out. Same
    // shape as this file's other guards — a named `InvalidInput`, not a crash.
    if group_cell_counts.len() != n_groups + 1 {
        return Err(AccelError::InvalidInput(format!(
            "gene_stats_nnz: group_cell_counts has {} entries, expected {} \
             (one per group plus the unlabelled bucket)",
            group_cell_counts.len(),
            n_groups + 1
        )));
    }
    // `n_groups + 1` wide throughout: the last slot is the unlabelled bucket.
    let mut group_sum = vec![0.0f64; n_groups + 1];
    let mut nonzero_in_g = vec![0usize; n_groups + 1];
    let mut neg: Vec<(f64, usize)> = Vec::new();
    let mut pos: Vec<(f64, usize)> = Vec::new();
    let mut total_sum = 0.0f64;

    for (&row, &v32) in rows.iter().zip(vals.iter()) {
        let row = row as usize;
        if row >= n_obs {
            continue;
        }
        // Pool bucket, not group: an unlabelled cell is in the pool (it ranks,
        // and it is in every group's rest) but in no group.
        let g = crate::diffexp::groups::pool_bucket(groups[row], n_groups);
        let v = v32 as f64;
        total_sum += v;
        group_sum[g] += v;
        if v != 0.0 {
            nonzero_in_g[g] += 1;
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
    // and out-of-range rows are dropped above, so the pooled nonzeros never
    // exceed n_obs. A corrupt sidecar with duplicate
    // rows in a column breaks that, and this used to be a `debug_assert!` —
    // compiled out in exactly the release builds where the subtraction below
    // then wrapped to ~1.8e19 and became a rank-block width. `ScxCsc::new`
    // rejects duplicate row indices, but the decode path builds sidecar shards
    // with `ScxCsc::new_unchecked`, so that check never runs here.
    //
    // `scx-sparse`'s release `overflow-checks` override does not reach this
    // crate, so the guard has to be explicit.
    //
    // Note precisely what it proves: `n_neg + n_pos` counts **nonzero** cells,
    // since explicit stored zeros join the implicit-zero block above. So
    // a column overfull purely with duplicated explicit zeros passes this — the
    // arithmetic is safe either way, but this is not a canonicality verdict.
    let n_zero_total = scx_sparse::implicit_zero_count(n_obs, n_neg + n_pos).map_err(|_| {
        AccelError::InvalidInput(format!(
            "non-canonical CSC sidecar: a gene column holds {} nonzero cells \
             against {n_obs} cells, so at least one cell is stored twice; \
             run `scx validate --deep`",
            n_neg + n_pos
        ))
    })?;
    // Mid-rank of the zero tie-block spanning 1-based ranks [n_neg+1 .. n_neg+n_zero].
    let zero_mid = n_neg as f64 + (n_zero_total as f64 + 1.0) / 2.0;

    let mut rank_sum = vec![0.0f64; n_groups + 1];
    let mut tie_correction = 0.0f64;

    neg.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    assign_block_ranks(&neg, 0, &mut rank_sum, &mut tie_correction);

    if n_zero_total > 1 {
        let t = n_zero_total as f64;
        tie_correction += t * t * t - t;
    }

    pos.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    assign_block_ranks(
        &pos,
        n_neg + n_zero_total,
        &mut rank_sum,
        &mut tie_correction,
    );

    // Zero cells (implicit + explicit) per bucket all carry the zero mid-rank.
    // The unlabelled bucket is walked too — not because anything reads its rank
    // sum, but because its own count has to be checked against the same
    // canonicality rule as a group's.
    //
    // Checked per bucket, not just pooled: the two subtractions have different
    // operands, so a duplicate concentrated in one small group can invert this
    // one while the pooled count above still fits under `n_obs`.
    for g in 0..=n_groups {
        let n_zero_cells_g = scx_sparse::implicit_zero_count(group_cell_counts[g], nonzero_in_g[g])
            .map_err(|_| {
                let what = if g == n_groups {
                    "the unlabelled cells hold".to_string()
                } else {
                    format!("group {g} holds")
                };
                AccelError::InvalidInput(format!(
                    "non-canonical CSC sidecar: {what} {} nonzero cells against \
                     {} cells, so at least one cell is stored twice; \
                     run `scx validate --deep`",
                    nonzero_in_g[g], group_cell_counts[g]
                ))
            })?;
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
    Ok(out)
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

    // Per-bucket cell counts, from the one shared partition the dense and GPU
    // kernels also use. The trailing entry is the unlabelled count — those cells
    // are in the pool (and so in every group's rest) but in no group.
    let partition = crate::diffexp::partition_by_group(groups, n_groups);
    let mut group_cell_counts: Vec<usize> = partition
        .group_indices
        .iter()
        .map(|cells| cells.len())
        .collect();
    group_cell_counts.push(partition.n_unlabelled(n_obs));

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
            .collect::<Result<Vec<_>>>()?;
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

    /// A gene column holding the same cell twice is non-canonical, and both
    /// implicit-zero counts here used to be raw `usize` subtractions guarded
    /// only by a `debug_assert!` — compiled out in exactly the release builds
    /// where they then wrapped to ~1.8e19 and became a rank-block width.
    ///
    /// Two arms, because the two subtractions have different operands and the
    /// second can invert while the first still fits:
    ///   * pooled  — `n_obs - (n_neg + n_pos)`
    ///   * per-bucket — `group_cell_counts[g] - nonzero_in_g[g]`
    ///
    /// `gene_stats_nnz` is called directly: the corruption cannot be written to
    /// a file (writers canonicalize, and the encoder `debug_assert`s it), so a
    /// unit call on the kernel is the only way to exercise it at all.
    // ── Review §7.17 — the nnz kernel has its own recorded route ───────

    #[test]
    fn the_nnz_gate_is_on_unless_explicitly_turned_off() {
        for raw in ["0", "false", "FALSE", "off"] {
            assert!(!nnz_gate_from_env_str(Some(raw)), "{raw} should disable");
        }
        for raw in ["1", "true", "on", "", "False", "no", "0 "] {
            assert!(
                nnz_gate_from_env_str(Some(raw)),
                "{raw:?} should leave the default on"
            );
        }
        assert!(nnz_gate_from_env_str(None), "unset means on");
    }

    /// The selection rule is more than the env gate: the nnz kernel is
    /// 1-vs-rest only. A stamp that read the gate alone would claim
    /// `cpu_csc_nnz` on a `reference=` or `rankby_abs` call that actually ran
    /// the densify path — which is the same class of defect §7.17 reports,
    /// pointing the other way.
    #[test]
    fn the_kernel_predicate_is_more_than_the_env_gate() {
        // The **real** predicate, not a local restatement of it:
        // `csc_wilcoxon_selects_nnz` exists so both arms of the gate are
        // reachable from a process whose `OnceLock` has already resolved.
        // Asserting against a locally-defined copy of the rule would pass
        // whatever the shipped rule did.
        assert!(csc_wilcoxon_selects_nnz(true, None, false));
        assert!(
            !csc_wilcoxon_selects_nnz(true, Some(0), false),
            "an explicit reference opts out"
        );
        assert!(
            !csc_wilcoxon_selects_nnz(true, None, true),
            "rankby_abs opts out"
        );
        assert!(
            !csc_wilcoxon_selects_nnz(false, None, false),
            "the gate is still required"
        );

        // And the exported entry point is that rule with the env gate plugged
        // in — it must not have grown a second condition of its own.
        let gate = nnz_wilcoxon_enabled();
        assert_eq!(
            csc_wilcoxon_uses_nnz_kernel(None, false),
            csc_wilcoxon_selects_nnz(gate, None, false)
        );
        assert!(!csc_wilcoxon_uses_nnz_kernel(Some(0), false));
        assert!(!csc_wilcoxon_uses_nnz_kernel(None, true));
    }

    #[test]
    fn gene_stats_nnz_rejects_a_cell_stored_twice() {
        // 2 cells in one group, none unlabelled. The column stores cell 0
        // twice, so both the pooled count (2 > ... ) and the group count invert.
        let groups = vec![0usize, 0usize];
        let group_cell_counts = vec![2usize, 0usize];
        let err = gene_stats_nnz(
            &[0, 0, 1],
            &[1.0, 2.0, 3.0],
            &groups,
            &group_cell_counts,
            2,
            1,
            true,
            false,
        )
        .expect_err("a cell stored twice must be rejected, not ranked");
        // Assert the *pooled* message specifically. Both guards produce a
        // "non-canonical CSC sidecar" error, so matching only that prefix let
        // this arm pass with the pooled guard sabotaged — the per-group guard
        // was catching it downstream. Whenever the pooled count inverts some
        // bucket's must too (the bucket counts sum to n_obs), so only the
        // message distinguishes which guard fired, and the pooled one still has
        // to exist: its subtraction runs first and would wrap before the
        // per-group loop is ever reached.
        assert!(
            matches!(err, AccelError::InvalidInput(ref m) if m.contains("a gene column holds")),
            "expected the pooled guard to fire first, got: {err:?}"
        );

        // Per-bucket arm: pooled count fits (2 <= 3 cells) but group 0 holds
        // 2 stored entries against its 1 cell. This is the case the pooled
        // check alone would wave through.
        let groups = vec![0usize, 1usize, 1usize];
        let group_cell_counts = vec![1usize, 2usize, 0usize];
        let err = gene_stats_nnz(
            &[0, 0],
            &[1.0, 2.0],
            &groups,
            &group_cell_counts,
            3,
            2,
            true,
            false,
        )
        .expect_err("a per-group overfull count must be rejected too");
        assert!(
            matches!(err, AccelError::InvalidInput(ref m) if m.contains("group 0")),
            "unexpected error: {err:?}"
        );

        // Canonical input on the same shape still works.
        let groups = vec![0usize, 0usize];
        let group_cell_counts = vec![2usize, 0usize];
        assert!(gene_stats_nnz(
            &[0, 1],
            &[1.0, 2.0],
            &groups,
            &group_cell_counts,
            2,
            1,
            true,
            false,
        )
        .is_ok());

        // The unlabelled bucket is guarded on the same rule: one unlabelled
        // cell whose column stores that row twice. The pooled count fits
        // (2 <= 3), so only the per-bucket arm can catch it — and it must name
        // the unlabelled cells rather than a group that is fine.
        let groups = vec![0usize, 0usize, 2usize];
        let group_cell_counts = vec![2usize, 0usize, 1usize];
        let err = gene_stats_nnz(
            &[2, 2],
            &[1.0, 2.0],
            &groups,
            &group_cell_counts,
            3,
            2,
            true,
            false,
        )
        .expect_err("an overfull unlabelled bucket must be rejected too");
        assert!(
            matches!(err, AccelError::InvalidInput(ref m) if m.contains("the unlabelled cells hold")),
            "unexpected error: {err:?}"
        );
    }

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

        // Both CSC kernels: the public driver (the nnz kernel, for 1-vs-rest by
        // default) and the densify kernel it no longer reaches unless the gate
        // is turned off.
        let csc_reader = BackedCscReader::new(ScxReader::open(&path).unwrap(), 0).unwrap();
        let driver = wilcoxon_rank_sum_streaming_csc(
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
        let densify = wilcoxon_rank_sum_densify_csc(
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
        for (kernel, csc_res) in [("driver", &driver), ("densify", &densify)] {
            assert_eq!(csr_res.group_names, csc_res.group_names, "{kernel}");
            for g in 0..csr_res.group_names.len() {
                assert_eq!(
                    csr_res.names[g], csc_res.names[g],
                    "{kernel}: group {g} gene order"
                );
                for k in 0..csr_res.scores[g].len() {
                    assert!(
                        (csr_res.scores[g][k] - csc_res.scores[g][k]).abs() < 1e-9,
                        "{kernel}: group {g} score[{k}] mismatch"
                    );
                    assert!(
                        (csr_res.pvals[g][k] - csc_res.pvals[g][k]).abs() < 1e-9,
                        "{kernel}: group {g} pval[{k}] mismatch"
                    );
                }
            }
        }
    }

    /// Premise check for the GPU §8.3 test, runnable without a device: the
    /// corrupt-sidecar fixture must actually reach a reader **carrying its
    /// corruption**. A writer or codec that sanitised NaN, or rejected the f32
    /// encoding, would leave that test asserting nothing while still passing
    /// its clean arm.
    ///
    /// It also pins the CPU/GPU symmetry the finding is about: the CPU CSC path
    /// already refuses this file.
    #[test]
    fn corrupt_csc_fixture_round_trips_its_corruption() {
        use crate::csc::test_helpers::write_file_with_corrupt_csc;

        let dir = tempdir().unwrap();
        let (n_obs, n_vars) = (16usize, 8usize);
        let dense = deterministic_dense(n_obs, n_vars);

        let clean =
            write_file_with_corrupt_csc(dir.path(), "rt_clean", n_obs, n_vars, &dense, |_, _| {});
        let reader = BackedCscReader::new(ScxReader::open(&clean).unwrap(), 0).unwrap();
        let csc = ColumnShardSource::read_csc_shard(&reader, 0).unwrap();
        assert!(
            csc.data.iter().all(|v| v.is_finite()) && !csc.data.is_empty(),
            "clean fixture must decode to finite values"
        );
        assert!(
            csc.indices.iter().all(|&i| (i as usize) < n_obs),
            "clean fixture rows must be in range"
        );

        let nan =
            write_file_with_corrupt_csc(dir.path(), "rt_nan", n_obs, n_vars, &dense, |_, v| {
                v[3] = f32::NAN;
            });
        let reader = BackedCscReader::new(ScxReader::open(&nan).unwrap(), 0).unwrap();
        let csc = ColumnShardSource::read_csc_shard(&reader, 0).unwrap();
        assert!(csc.data[3].is_nan(), "the NaN must survive the round trip");

        let dup =
            write_file_with_corrupt_csc(dir.path(), "rt_dup", n_obs, n_vars, &dense, |ix, _| {
                ix[3] = ix[2];
            });
        let reader = BackedCscReader::new(ScxReader::open(&dup).unwrap(), 0).unwrap();
        let csc = ColumnShardSource::read_csc_shard(&reader, 0).unwrap();
        assert_eq!(
            csc.indices[3], csc.indices[2],
            "the duplicate row must survive the round trip"
        );

        // An out-of-range row does NOT survive: `check_minor_indices` in the
        // shard decoder bounds CSC row indices against the shard header's
        // `n_minor` (= n_obs). So on any backed file that hazard is already
        // closed one layer below the GPU — the validator's range check and the
        // kernels' `cell` guard are defence in depth for `ColumnShardSource`
        // impls that do not decode through that seam. Pinned here so the layer
        // that actually closes it is named, and a future change that relaxes it
        // shows up as a failure rather than as silence.
        let oob =
            write_file_with_corrupt_csc(dir.path(), "rt_oob", n_obs, n_vars, &dense, |ix, _| {
                ix[3] = n_obs as u32;
            });
        let reader = BackedCscReader::new(ScxReader::open(&oob).unwrap(), 0).unwrap();
        let err = ColumnShardSource::read_csc_shard(&reader, 0)
            .expect_err("the decoder must reject an out-of-range CSC row index");
        assert!(
            matches!(
                err,
                scx_format_io::ScxError::ShardIndexOutOfRange { index, .. }
                    if index == n_obs as u32
            ),
            "expected ShardIndexOutOfRange, got {err:?}"
        );

        // The CPU CSC route rejects the NaN file; §8.3 is that the GPU CSC
        // route did not.
        let reader = BackedCscReader::new(ScxReader::open(&nan).unwrap(), 0).unwrap();
        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        let groups: Vec<usize> = (0..n_obs).map(|i| i % 2).collect();
        let group_names = vec!["a".to_string(), "b".to_string()];
        let err = wilcoxon_rank_sum_streaming_csc(
            &reader,
            &gene_names,
            &groups,
            &group_names,
            None,
            4,
            false,
            false,
            true,
        )
        .expect_err("CPU CSC path must reject a non-finite value");
        assert!(err.to_string().contains("non-finite"), "{err}");
        // The densify kernel, which the driver above no longer reaches by
        // default, refuses it too.
        let err = wilcoxon_rank_sum_densify_csc(
            &reader,
            &gene_names,
            &groups,
            &group_names,
            None,
            4,
            false,
            false,
            true,
        )
        .expect_err("the densify kernel must reject a non-finite value");
        assert!(err.to_string().contains("non-finite"), "{err}");
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
                csc: csc_reader
                    .as_ref()
                    .map(|c| c as &(dyn scx_format_io::ColumnShardSource + Sync)),
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
    #[ignore = "requires a CUDA GPU"]
    fn wilcoxon_gpu_v3_csc_one_vs_rest() {
        require_gpu_or_skip!();
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
    #[ignore = "requires a CUDA GPU"]
    fn wilcoxon_gpu_v3_csc_ref_mode() {
        require_gpu_or_skip!();
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
    #[ignore = "requires a CUDA GPU"]
    fn wilcoxon_gpu_v3_csr_fallback() {
        require_gpu_or_skip!();
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
    #[ignore = "requires a CUDA GPU"]
    fn wilcoxon_gpu_v3_route_metadata() {
        require_gpu_or_skip!();
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
    #[ignore = "requires a CUDA GPU"]
    fn wilcoxon_gpu_v3_csc_empty_group() {
        require_gpu_or_skip!();
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
    #[ignore = "requires a CUDA GPU"]
    fn wilcoxon_gpu_v3_csc_all_zero_gene() {
        require_gpu_or_skip!();
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
    #[ignore = "requires a CUDA GPU"]
    fn wilcoxon_gpu_v3_csc_tie_spanning_gene() {
        require_gpu_or_skip!();
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
    #[ignore = "requires a CUDA GPU"]
    fn wilcoxon_gpu_v3_csc_multi_shard() {
        require_gpu_or_skip!();
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

    /// (10) §8.3 — the CSC-direct route staged shards with **no** validation,
    /// so a malformed sidecar reached the kernels: a NaN sorted above `+INF` in
    /// `block_radix_sort_per_gene_kernel` and silently corrupted U / tie counts
    /// / p-values, and a duplicate `(cell, gene)` raced one `slab` cell with a
    /// nondeterministic winner. The CPU CSC path and `gpu_csr_v3` reject both;
    /// this route was the outlier. Each must now surface as an error, not a
    /// number — driven through the real public entry point over a real file, so
    /// the test covers the route and not just the validator.
    ///
    /// The finding's third defect (an out-of-range row as an unguarded
    /// `cell_to_group[cell]` device read) has no arm here because it cannot be
    /// staged from a backed file at all: the shard decoder rejects it first.
    /// See `corrupt_csc_fixture_round_trips_its_corruption`, which pins that.
    #[cfg(feature = "gpu")]
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn wilcoxon_gpu_v3_csc_rejects_a_malformed_sidecar() {
        require_gpu_or_skip!();
        use crate::csc::test_helpers::write_file_with_corrupt_csc;

        let dir = tempdir().unwrap();
        let n_obs = 64usize;
        let n_vars = 20usize;
        let dense = deterministic_dense(n_obs, n_vars);
        let gn: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        let gr: Vec<usize> = (0..n_obs).map(|i| (i * 3) / n_obs).collect();
        let names = vec!["ref".to_string(), "ko_a".to_string(), "ko_b".to_string()];

        let run = |path: &std::path::Path| {
            let csr_reader = BackedCsrReader::new(ScxReader::open(path).unwrap(), 0);
            let csc_reader = BackedCscReader::new(ScxReader::open(path).unwrap(), 0).unwrap();
            wilcoxon_rank_sum_gpu(
                0,
                GpuDeShardInput::Backed {
                    csr: &csr_reader,
                    csc: Some(&csc_reader as &(dyn scx_format_io::ColumnShardSource + Sync)),
                },
                &gn,
                &gr,
                &names,
                None,
                Some(7),
                false,
                false,
                true,
            )
        };

        // Premise: the same fixture written *without* corruption runs the
        // CSC-direct route to completion, so every rejection below is
        // attributable to the corruption and not to the f32 sidecar encoding.
        let clean =
            write_file_with_corrupt_csc(dir.path(), "csc_clean", n_obs, n_vars, &dense, |_, _| {});
        let ok = run(&clean).expect("clean f32 sidecar must succeed");
        assert_eq!(ok.exec_info.route, AccelRoute::GpuCscV3);

        let nan =
            write_file_with_corrupt_csc(dir.path(), "csc_nan", n_obs, n_vars, &dense, |_, v| {
                v[7] = f32::NAN;
            });
        let dup =
            write_file_with_corrupt_csc(dir.path(), "csc_dup", n_obs, n_vars, &dense, |ix, _| {
                ix[7] = ix[6]; // duplicate row inside the first column
            });

        for (label, needle, path) in [
            ("non-finite", "non-finite", nan),
            ("duplicate row", "unsorted or duplicate", dup),
        ] {
            let err = run(&path).expect_err(&format!("{label} must be rejected"));
            let msg = err.to_string();
            assert!(msg.contains(needle), "{label}: unexpected message: {msg}");
        }
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
    /// path on a real backed SCX file (codec round-trip) with an unlabelled
    /// cell class present.
    #[test]
    fn nnz_csc_matches_csr_streaming_on_file() {
        let dir = tempdir().unwrap();
        let (n_obs, n_vars) = (20usize, 10usize);
        let dense = deterministic_dense(n_obs, n_vars);
        let path = write_csr_csc_test_file(dir.path(), "nnzfile", n_obs, n_vars, &dense, 5);

        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        let group_names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        // Cycle 0,1,2,3 → index 3 is the unlabelled sentinel (n_groups == 3).
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
        /// small matrices with negatives, explicit + implicit zeros, unlabelled
        /// cells, and both tie-correction settings — the §5.3 exactness contract.
        ///
        /// Both arms are also checked against a **third, independent** result:
        /// the dense kernel run with the unlabelled cells collected into one
        /// *extra* group. Agreement between two implementations only proves they
        /// agree — and they did, on the wrong answer, for as long as both left
        /// unlabelled cells out of the pool. The extra-group run is the oracle;
        /// nnz-vs-dense is the parity check.
        ///
        /// `pvals_adj` is compared against the oracle too. An earlier version of
        /// this test skipped it, claiming BH's denominator changes when the
        /// oracle adds a group — it does not: BH is applied **per group**
        /// (`benjamini_hochberg(&pvals)` inside the per-group assembly loop, in
        /// both this file and `diffexp::cpu`), so an extra group cannot move an
        /// existing group's adjusted p-values.
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

            // Group per cell in 0..=n_groups; `n_groups` is the unlabelled
            // sentinel — such a cell belongs to no group but is in the pool, so
            // it ranks and it counts in every group's "rest".
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

            // Oracle: the same matrix, the unlabelled cells promoted to a group
            // of their own. Same pool, same ranks, same rest for every real
            // group — but arrived at without any sentinel handling.
            let mut oracle_group_names = group_names.clone();
            oracle_group_names.push("unlabelled".to_string());
            let oracle_groups: Vec<usize> =
                groups.iter().map(|&g| g.min(n_groups)).collect();
            let oracle_res = wilcoxon_rank_sum(
                &dense, n_obs, n_vars, &gene_names, &oracle_groups, &oracle_group_names,
                None, log_transformed, false, tie_correct, 0,
            ).unwrap();

            prop_assert_eq!(&dense_res.group_names, &nnz_res.group_names);
            for gi in 0..group_names.len() {
                let dmap = group_map(&dense_res, gi);
                let nmap = group_map(&nnz_res, gi);
                let omap = group_map(&oracle_res, gi);
                for (name, &(ds, dp, dpa, dl)) in &dmap {
                    let &(ns, np, npa, nl) = nmap.get(name).unwrap();
                    prop_assert!(close(ds, ns), "score g{} {}: {} vs {}", gi, name, ds, ns);
                    prop_assert!(close(dp, np), "pval g{} {}: {} vs {}", gi, name, dp, np);
                    prop_assert!(close(dpa, npa), "padj g{} {}: {} vs {}", gi, name, dpa, npa);
                    prop_assert!(close(dl, nl), "logfc g{} {}: {} vs {}", gi, name, dl, nl);
                    let &(os, op, opa, ol) = omap.get(name).unwrap();
                    prop_assert!(close(ds, os), "score vs extra-group g{} {}: {} vs {}", gi, name, ds, os);
                    prop_assert!(close(dp, op), "pval vs extra-group g{} {}: {} vs {}", gi, name, dp, op);
                    prop_assert!(close(dpa, opa), "padj vs extra-group g{} {}: {} vs {}", gi, name, dpa, opa);
                    prop_assert!(close(dl, ol), "logfc vs extra-group g{} {}: {} vs {}", gi, name, dl, ol);
                }
            }
        }
    }

    /// The exact-nnz kernel against the densify kernel it would replace, at
    /// **zero** tolerance, on a matrix built to be tie-heavy.
    ///
    /// The property test above allows `1e-9 + 1e-6·|b|`. That bar is too loose
    /// to promote the nnz kernel to the default: a default that moves a p-value
    /// in its last bits reorders genes whose scores tie, and a tie-heavy count
    /// matrix is where that happens. So this pins what the promotion needs —
    /// bit-identical `scores`, `pvals`, `pvals_adj` and `logfoldchanges`, and the
    /// same gene order within every group — through the two public drivers,
    /// chunked the same way, rather than through the per-gene helpers.
    ///
    /// The fixture: counts in `{1, 2, 3}` at ~25% density, so every gene is one
    /// large zero block plus three tie runs; an explicitly stored zero in some
    /// columns; one column with no stored entry and one whose every stored
    /// value is `1`; four groups plus unlabelled cells; and a chunk width that
    /// does not divide `n_vars`.
    #[test]
    fn nnz_matches_the_densify_kernel_exactly_on_tie_heavy_counts() {
        let (n_obs, n_vars, n_groups) = (300usize, 24usize, 4usize);
        let mut rng = ChaCha8Rng::seed_from_u64(0x7_1e5);
        let groups: Vec<usize> = (0..n_obs).map(|_| rng.gen_range(0..=n_groups)).collect();
        let mut cols: Vec<Vec<(i32, f32)>> = vec![Vec::new(); n_vars];
        for (c, col) in cols.iter_mut().enumerate() {
            if c == 3 {
                continue; // no stored entry at all
            }
            for row in 0..n_obs {
                if rng.gen::<f64>() >= 0.25 {
                    if c % 5 == 0 && rng.gen::<f64>() < 0.02 {
                        col.push((row as i32, 0.0)); // explicitly stored zero
                    }
                    continue;
                }
                let v = if c == 7 {
                    1.0
                } else {
                    rng.gen_range(1i32..=3) as f32
                };
                col.push((row as i32, v));
            }
        }
        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        let group_names: Vec<String> = (0..n_groups).map(|g| format!("grp{g}")).collect();
        let src = InMemCsc { cols, n_obs };
        let same = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());

        for log_transformed in [false, true] {
            for tie_correct in [true, false] {
                let dense = wilcoxon_rank_sum_densify_csc(
                    &src,
                    &gene_names,
                    &groups,
                    &group_names,
                    None,
                    5,
                    log_transformed,
                    false,
                    tie_correct,
                )
                .unwrap();
                let nnz = wilcoxon_rank_sum_nnz_csc(
                    &src,
                    &gene_names,
                    &groups,
                    &group_names,
                    5,
                    log_transformed,
                    tie_correct,
                )
                .unwrap();
                let ctx = format!("log_transformed={log_transformed} tie_correct={tie_correct}");
                assert_eq!(dense.group_names, nnz.group_names, "{ctx}");
                for g in 0..n_groups {
                    assert_eq!(dense.names[g], nnz.names[g], "{ctx}: gene order, group {g}");
                    for (field, a, b) in [
                        ("scores", &dense.scores[g], &nnz.scores[g]),
                        ("pvals", &dense.pvals[g], &nnz.pvals[g]),
                        ("pvals_adj", &dense.pvals_adj[g], &nnz.pvals_adj[g]),
                        (
                            "logfoldchanges",
                            &dense.logfoldchanges[g],
                            &nnz.logfoldchanges[g],
                        ),
                    ] {
                        for (k, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
                            assert!(
                                same(x, y),
                                "{ctx}: {field}[{g}][{k}] ({}) densify {x:e} vs nnz {y:e}",
                                dense.names[g][k]
                            );
                        }
                    }
                }
            }
        }
    }

    // --- ORG-7.21-3: the tie-run walk, pinned before it was shared ----------
    //
    // The nnz path's half of the duplication `diffexp::cpu::rank_with_ties`
    // held the other half of. Same walk over equal-value runs, same `Σ(t³−t)`,
    // but offset into a global rank space and accumulating per *group* rather
    // than per element. Pinned here before the two collapsed onto one
    // primitive, so the refactor is provably arithmetic-neutral.
    //
    // Expected values are derived by hand from `(2·offset + i + 1 + j) / 2`,
    // not read off a run of the code. Falsify by perturbing one `mid_rank`.

    /// Leading tie run, singleton, trailing tie run, at both offsets.
    ///
    /// The block sorts as `1,1 | 2 | 3,3` over positions `[0,2) [2,3) [3,5)`.
    /// At `offset = 0` the mid-ranks are `1.5, 3.0, 4.5`; at `offset = 4` they
    /// shift by exactly 4 to `5.5, 7.0, 8.5`. The correction is offset-free:
    /// `(2³−2) + (2³−2) = 12`.
    #[test]
    fn assign_block_ranks_pinned_mid_ranks_and_tie_correction() {
        let block = [(1.0, 0usize), (1.0, 1), (2.0, 0), (3.0, 1), (3.0, 1)];

        let mut rank_sum = vec![0.0f64; 2];
        let mut tc = 0.0f64;
        assign_block_ranks(&block, 0, &mut rank_sum, &mut tc);
        assert_eq!(rank_sum, vec![4.5, 10.5]);
        assert_eq!(tc, 12.0);
        // Ranks 1..=5 sum to 15 whatever the grouping — an independent check
        // on the mid-ranks that does not restate them.
        assert_eq!(rank_sum.iter().sum::<f64>(), 15.0);

        let mut rank_sum = vec![0.0f64; 2];
        let mut tc = 0.0f64;
        assign_block_ranks(&block, 4, &mut rank_sum, &mut tc);
        assert_eq!(rank_sum, vec![12.5, 22.5]);
        assert_eq!(tc, 12.0);
        // Ranks 5..=9 sum to 35.
        assert_eq!(rank_sum.iter().sum::<f64>(), 35.0);
    }

    /// One run spanning the whole block, and an empty block.
    ///
    /// All-equal at `offset = 3`: mid-rank `(2·3 + 0 + 1 + 4)/2 = 5.5`,
    /// correction `4³ − 4 = 60`. An empty block must touch neither output.
    #[test]
    fn assign_block_ranks_pinned_all_equal_and_empty() {
        let block = [(2.5, 0usize), (2.5, 0), (2.5, 1), (2.5, 1)];
        let mut rank_sum = vec![0.0f64; 2];
        let mut tc = 0.0f64;
        assign_block_ranks(&block, 3, &mut rank_sum, &mut tc);
        assert_eq!(rank_sum, vec![11.0, 11.0]);
        assert_eq!(tc, 60.0);

        let mut rank_sum = vec![0.0f64; 2];
        let mut tc = 7.0f64; // pre-loaded: an empty block must not reset it
        assign_block_ranks(&[], 0, &mut rank_sum, &mut tc);
        assert_eq!(rank_sum, vec![0.0, 0.0]);
        assert_eq!(tc, 7.0);
    }
}
