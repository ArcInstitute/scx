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

use crate::diffexp::{merge_diff_exp_results, wilcoxon_rank_sum, DiffExpResult};
use crate::error::{AccelError, Result};
use scx_format::ColumnShardSource;

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
        )?;
        all_chunk_results.push(chunk_result);
    }

    merge_diff_exp_results(all_chunk_results, rankby_abs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csc::test_helpers::{deterministic_dense, write_csr_csc_test_file};
    use crate::diffexp::wilcoxon_rank_sum_streaming;
    use scx_format::{BackedCscReader, BackedCsrReader, ScxReader};
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
}
