//! property-style numerical-equivalence tests for CSR vs CSC kernels.
//!
//! Walks a small grid of generated dense reference matrices (varying
//! shape, density, seed, `cols_per_csc_shard`) and verifies that every
//! CSR/CSC kernel pair produces the same numerical result on the same
//! data:
//!
//! - `streaming_mean_var` vs `streaming_mean_var_csc`
//! - `streaming_clip_square_sum` vs `streaming_clip_square_sum_csc`
//! - `wilcoxon_rank_sum_streaming` vs `wilcoxon_rank_sum_streaming_csc`
//! - `pdex_ref_streaming` vs `pdex_ref_streaming_csc`
//! - `pseudobulk_aggregate` vs `pseudobulk_aggregate_csc`
//!
//! Tolerances: f64 sums for u8 inputs are bit-equal; means/vars and
//! clipped sums are within 1e-7 (Bessel-correction division loses a
//! few low bits relative to bit-equality). Wilcoxon scores / pvals
//! are within 1e-9 because the dense buffer scattered by the CSC
//! kernel is byte-for-byte the same as the CSR-streamed dense buffer.
//!
//! Skips the proptest crate intentionally: a deterministic loop over
//! seeded LCG-generated dense matrices is enough for the parity claim
//! and avoids pulling in the proptest framework as a scx-accel
//! dev-dep. The existing codec round-trip proptest in
//! `scx-codec/tests/proptest_roundtrip.rs` covers byte-level CSR/CSC
//! shard encode/decode parity (CSC uses the same `encode_shard` /
//! `decode_shard` codec paths).

#![cfg(test)]

use crate::csc::mean_var::{streaming_clip_square_sum_csc, streaming_mean_var_csc};
use crate::csc::pdex::pdex_ref_streaming_csc;
use crate::csc::test_helpers::write_csr_csc_test_file;
use crate::csc::wilcoxon::wilcoxon_rank_sum_streaming_csc;
use crate::diffexp::{pdex_ref_streaming, wilcoxon_rank_sum_streaming};
use crate::hvg::{streaming_clip_square_sum, streaming_mean_var};
use crate::pseudobulk::{pseudobulk_aggregate, AggregationMethod, GeomMeanMode};
use scx_format::{BackedCscReader, BackedCsrReader, ScxReader};
use tempfile::tempdir;

/// LCG-generated dense reference matrix for parity testing.
///
/// Values are u8 in `[0, 200)`, with sparsity controlled by
/// `density_pct`. Deterministic per `(seed, n_obs, n_vars)` so any
/// failure is reproducible from the printed args.
fn lcg_dense(seed: u64, n_obs: usize, n_vars: usize, density_pct: u8) -> Vec<u8> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut dense = vec![0u8; n_obs * n_vars];
    for r in 0..n_obs {
        for c in 0..n_vars {
            // Update LCG twice: once for the keep/skip decision, once
            // for the value if kept.
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let keep = ((state >> 33) % 100) < density_pct as u64;
            if keep {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                dense[r * n_vars + c] = ((state >> 33) % 200) as u8 + 1;
            }
        }
    }
    dense
}

/// One parity check: build CSR + CSC test file, run all kernel pairs,
/// assert equivalence.
fn check_parity_for(
    seed: u64,
    n_obs: usize,
    n_vars: usize,
    density_pct: u8,
    cols_per_csc_shard: usize,
) {
    let dir = tempdir().unwrap();
    let dense = lcg_dense(seed, n_obs, n_vars, density_pct);
    let path = write_csr_csc_test_file(
        dir.path(),
        &format!("p_{seed}_{n_obs}x{n_vars}_d{density_pct}"),
        n_obs,
        n_vars,
        &dense,
        cols_per_csc_shard,
    );

    // Streaming mean/var parity.
    let csr_reader = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);
    let csr_stats = streaming_mean_var(&csr_reader).unwrap();
    let csc_reader = BackedCscReader::new(ScxReader::open(&path).unwrap(), 0).unwrap();
    let csc_stats = streaming_mean_var_csc(&csc_reader).unwrap();
    for j in 0..n_vars {
        assert!(
            (csr_stats.means[j] - csc_stats.means[j]).abs() < 1e-9,
            "mean[{j}] mismatch: seed={seed} {n_obs}x{n_vars} d={density_pct} cols/shard={cols_per_csc_shard} csr={} csc={}",
            csr_stats.means[j],
            csc_stats.means[j]
        );
        assert!(
            (csr_stats.variances[j] - csc_stats.variances[j]).abs() < 1e-7,
            "var[{j}] mismatch: seed={seed} {n_obs}x{n_vars} d={density_pct}: csr={} csc={}",
            csr_stats.variances[j],
            csc_stats.variances[j]
        );
    }

    // Clipped square sum parity.
    let clip = vec![5.0_f64; n_vars];
    let (csr_sum, csr_sq) = streaming_clip_square_sum(&csr_reader, &clip).unwrap();
    let (csc_sum, csc_sq) = streaming_clip_square_sum_csc(&csc_reader, &clip).unwrap();
    for j in 0..n_vars {
        assert!(
            (csr_sum[j] - csc_sum[j]).abs() < 1e-9,
            "clip_sum[{j}] mismatch: seed={seed} {n_obs}x{n_vars}"
        );
        assert!(
            (csr_sq[j] - csc_sq[j]).abs() < 1e-7,
            "clip_sq[{j}] mismatch: seed={seed} {n_obs}x{n_vars}"
        );
    }

    // Wilcoxon parity (only when there are enough cells in each group).
    if n_obs >= 6 {
        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        let group_names = vec!["A".to_string(), "B".to_string()];
        let groups: Vec<usize> = (0..n_obs)
            .map(|i| if i < n_obs / 2 { 0 } else { 1 })
            .collect();
        let csr_de = wilcoxon_rank_sum_streaming(
            &csr_reader,
            &gene_names,
            &groups,
            &group_names,
            None,
            n_vars.max(1),
            false,
            false,
            false,
        )
        .unwrap();
        let csc_de = wilcoxon_rank_sum_streaming_csc(
            &csc_reader,
            &gene_names,
            &groups,
            &group_names,
            None,
            n_vars.max(1),
            false,
            false,
            false,
        )
        .unwrap();
        assert_eq!(csr_de.group_names, csc_de.group_names);
        for g in 0..csr_de.group_names.len() {
            for k in 0..csr_de.scores[g].len() {
                assert!(
                    (csr_de.scores[g][k] - csc_de.scores[g][k]).abs() < 1e-9,
                    "wilcoxon score[{g}][{k}] mismatch on seed={seed}"
                );
                assert!(
                    (csr_de.pvals[g][k] - csc_de.pvals[g][k]).abs() < 1e-9,
                    "wilcoxon pval[{g}][{k}] mismatch on seed={seed}"
                );
            }
        }
    }

    // pdex_ref parity (only when there are enough cells in each group).
    // Both kernels run the same underlying pdex_ref math on a dense
    // per-chunk buffer that the CSC kernel scatters identically to the
    // CSR stream, so results are bit-equal up to 1e-9.
    if n_obs >= 6 {
        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();
        let group_names = vec!["A".to_string(), "B".to_string()];
        let groups: Vec<usize> = (0..n_obs)
            .map(|i| if i < n_obs / 2 { 0 } else { 1 })
            .collect();
        let reference = 0; // group "A" is the reference
        let csr_pdex = pdex_ref_streaming(
            &csr_reader,
            &gene_names,
            &groups,
            &group_names,
            reference,
            n_vars.max(1),
            GeomMeanMode::ArithRaw,
            1.0,
        )
        .unwrap();
        let csc_pdex = pdex_ref_streaming_csc(
            &csc_reader,
            &gene_names,
            &groups,
            &group_names,
            reference,
            n_vars.max(1),
            GeomMeanMode::ArithRaw,
            1.0,
        )
        .unwrap();
        assert_eq!(csr_pdex.group_names, csc_pdex.group_names);
        for g in 0..csr_pdex.group_names.len() {
            for k in 0..csr_pdex.log2_fold_changes[g].len() {
                assert!(
                    (csr_pdex.log2_fold_changes[g][k] - csc_pdex.log2_fold_changes[g][k]).abs()
                        < 1e-9,
                    "pdex log2fc[{g}][{k}] mismatch on seed={seed} {n_obs}x{n_vars}: csr={} csc={}",
                    csr_pdex.log2_fold_changes[g][k],
                    csc_pdex.log2_fold_changes[g][k]
                );
                assert!(
                    (csr_pdex.statistics[g][k] - csc_pdex.statistics[g][k]).abs() < 1e-9,
                    "pdex statistic[{g}][{k}] mismatch on seed={seed}"
                );
                assert!(
                    (csr_pdex.p_values[g][k] - csc_pdex.p_values[g][k]).abs() < 1e-9,
                    "pdex pval[{g}][{k}] mismatch on seed={seed}"
                );
            }
        }
    }

    // Pseudobulk filtered-gene parity. Pick every-other column as the
    // gene subset so the kernel exercises a non-contiguous run.
    if n_obs >= 4 && n_vars >= 4 {
        let groupby = vec!["batch".to_string()];
        let labels: Vec<String> = (0..n_obs)
            .map(|i| {
                if i < n_obs / 2 {
                    "A".to_string()
                } else {
                    "B".to_string()
                }
            })
            .collect();
        let obs_groups = vec![labels];
        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();

        let csr_full = pseudobulk_aggregate(
            &csr_reader,
            &obs_groups,
            &groupby,
            &gene_names,
            AggregationMethod::Sum,
            0,
        )
        .unwrap();

        let col_subset: Vec<u32> = (0..n_vars as u32).step_by(2).collect();
        let projected_names: Vec<String> = col_subset
            .iter()
            .map(|&c| gene_names[c as usize].clone())
            .collect();
        let cell_to_group: Vec<usize> = (0..n_obs)
            .map(|i| if i < n_obs / 2 { 0 } else { 1 })
            .collect();
        let csc_proj = crate::csc::pseudobulk_aggregate_csc(
            &csc_reader,
            &cell_to_group,
            csr_full.n_groups,
            csr_full.group_labels.clone(),
            &groupby,
            &projected_names,
            &col_subset,
            AggregationMethod::Sum,
            0,
        )
        .unwrap();

        for g in 0..csr_full.n_groups {
            for (out_col, &orig_col) in col_subset.iter().enumerate() {
                let csr_v = csr_full.counts[g * csr_full.n_vars + orig_col as usize];
                let csc_v = csc_proj.counts[g * csc_proj.n_vars + out_col];
                assert!(
                    (csr_v - csc_v).abs() < 1e-9,
                    "pseudobulk[g={g}, col={orig_col}] mismatch on seed={seed}: csr={csr_v} csc={csc_v}"
                );
            }
        }
    }
}

#[test]
fn csr_csc_kernel_parity_grid() {
    // Small grid: shapes × densities × cols_per_csc_shard × seeds.
    // Kept tight so the test runs in <1s; the goal is broad coverage,
    // not exhaustive enumeration.
    let shapes = [(8usize, 6usize), (12, 8), (16, 12), (20, 24)];
    let densities = [10u8, 30, 60, 90];
    let csc_shards = [3usize, 5, 7];
    let seeds = [0xDEADBEEFu64, 0xCAFEBABE, 0x12345678];

    for &(n_obs, n_vars) in &shapes {
        for &density in &densities {
            for &cps in &csc_shards {
                for &seed in &seeds {
                    check_parity_for(seed, n_obs, n_vars, density, cps);
                }
            }
        }
    }
}
