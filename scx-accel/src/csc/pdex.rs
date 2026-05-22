//! CSC streaming `pdex_ref` differential expression.
//!
//! Mirrors [`crate::diffexp::pdex_ref_streaming`] (CSR path) but processes
//! gene chunks as CSC column ranges. For each chunk:
//!   1. `source.read_csc_columns(chunk_start..chunk_end)` materialises one
//!      CSC slab covering the chunk's columns.
//!   2. Scatter the slab's `(row, col, value)` triples into a row-major
//!      `[n_obs × chunk_size]` dense buffer.
//!   3. Call the existing `pdex_ref()` kernel on that buffer.
//!
//! The chunk-merge step (`merge_pdex_chunk_into` + `recompute_pdex_fdrs`)
//! runs unchanged on per-chunk results, so the numerics are identical to
//! the CSR path.
//!
//! Same shape as [`crate::csc::wilcoxon::wilcoxon_rank_sum_streaming_csc`];
//! the only difference is the per-chunk kernel call and the merge logic.

use crate::diffexp::{
    empty_pdex_result, merge_pdex_chunk_into, pdex_ref, recompute_pdex_fdrs, PdexRefResult,
};
use crate::error::{AccelError, Result};
use crate::pseudobulk::GeomMeanMode;
use scx_format::ColumnShardSource;

/// Gene-chunked `pdex_ref` driven by a CSC source.
///
/// Returns the same [`PdexRefResult`] as the CSR equivalent
/// (`pdex_ref_streaming`). Chunk boundaries are aligned to columns
/// (genes), not rows.
#[allow(clippy::too_many_arguments)]
pub fn pdex_ref_streaming_csc<S: ColumnShardSource + ?Sized>(
    source: &S,
    gene_names: &[String],
    groups: &[usize],
    group_names: &[String],
    reference: usize,
    gene_chunk_size: usize,
    mode: GeomMeanMode,
    epsilon: f64,
) -> Result<PdexRefResult> {
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

    let mut combined: Option<PdexRefResult> = None;

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

        let chunk_result = pdex_ref(
            dense_view,
            n_obs,
            chunk_size,
            &gene_names[chunk_start..chunk_end],
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csc::test_helpers::{deterministic_dense, write_csr_csc_test_file};
    use crate::diffexp::pdex_ref_streaming;
    use scx_format::{BackedCscReader, BackedCsrReader, ScxReader};
    use tempfile::tempdir;

    /// CSR↔CSC parity: `pdex_ref_streaming` (CSR) and `pdex_ref_streaming_csc`
    /// (CSC) must produce identical `PdexRefResult` on the same fixture.
    #[test]
    fn pdex_ref_csc_matches_csr() {
        let n_obs = 32usize;
        let n_vars = 16usize;
        let cols_per_csc_shard = 8usize;
        let dense = deterministic_dense(n_obs, n_vars);
        let dir = tempdir().unwrap();
        let path = write_csr_csc_test_file(
            dir.path(),
            "pdex_csc_parity",
            n_obs,
            n_vars,
            &dense,
            cols_per_csc_shard,
        );

        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        // 3 groups: ref + 2 KOs, ~equal sizes.
        let groups: Vec<usize> = (0..n_obs).map(|i| (i * 3) / n_obs).collect();
        let group_names = vec!["ref".to_string(), "ko_a".to_string(), "ko_b".to_string()];
        let reference = 0usize;
        let mode = GeomMeanMode::ArithRaw;
        let epsilon = 1e-6;

        let csr_reader = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);
        let csc_reader = BackedCscReader::new(ScxReader::open(&path).unwrap(), 0).unwrap();

        let csr_res = pdex_ref_streaming(
            &csr_reader,
            &gene_names,
            &groups,
            &group_names,
            reference,
            5, // gene_chunk_size — forces multi-chunk on n_vars=16
            mode,
            epsilon,
        )
        .expect("CSR streaming pdex_ref failed");

        let csc_res = pdex_ref_streaming_csc(
            &csc_reader,
            &gene_names,
            &groups,
            &group_names,
            reference,
            5,
            mode,
            epsilon,
        )
        .expect("CSC streaming pdex_ref failed");

        // Structural equality.
        assert_eq!(csr_res.group_names, csc_res.group_names);
        assert_eq!(csr_res.feature_names, csc_res.feature_names);
        assert_eq!(csr_res.ref_membership, csc_res.ref_membership);
        assert_eq!(csr_res.target_memberships, csc_res.target_memberships);

        // Numeric equality: U statistic exact; p / means within fp64 ulps.
        let n_test = csr_res.group_names.len();
        for tg in 0..n_test {
            for var in 0..n_vars {
                let u_csr = csr_res.statistics[tg][var];
                let u_csc = csc_res.statistics[tg][var];
                if u_csr.is_finite() && u_csc.is_finite() {
                    assert!(
                        (u_csr - u_csc).abs() < 1e-9,
                        "U mismatch tg={tg} gene={var}: csr={u_csr}, csc={u_csc}"
                    );
                }
                let p_csr = csr_res.p_values[tg][var];
                let p_csc = csc_res.p_values[tg][var];
                assert!(
                    (p_csr - p_csc).abs() < 1e-12,
                    "p mismatch tg={tg} gene={var}: csr={p_csr}, csc={p_csc}"
                );
                let tm_csr = csr_res.target_means[tg][var];
                let tm_csc = csc_res.target_means[tg][var];
                if tm_csr.is_finite() && tm_csc.is_finite() {
                    assert!(
                        (tm_csr - tm_csc).abs() < 1e-12,
                        "target_mean mismatch tg={tg} gene={var}: csr={tm_csr}, csc={tm_csc}"
                    );
                }
                let r_csr = csr_res.ref_means[var];
                let r_csc = csc_res.ref_means[var];
                if r_csr.is_finite() && r_csc.is_finite() {
                    assert!(
                        (r_csr - r_csc).abs() < 1e-12,
                        "ref_mean mismatch var={var}: csr={r_csr}, csc={r_csc}"
                    );
                }
            }
        }
    }
}
