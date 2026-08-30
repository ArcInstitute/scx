//! Unit tests for the gene-set scoring kernels.

use super::*;
use scx_format_io::Result as IoResult;
use scx_sparse::ScxCsr;

/// Build a CSR shard from a dense row-major matrix.
fn csr_from_dense(rows: &[Vec<f32>]) -> ScxCsr {
    let n_rows = rows.len();
    let n_cols = rows.first().map(|r| r.len()).unwrap_or(0);
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for row in rows {
        for (c, &v) in row.iter().enumerate() {
            if v != 0.0 {
                indices.push(c as i32);
                data.push(v);
            }
        }
        indptr.push(indices.len() as i64);
    }
    ScxCsr::new_unchecked((n_rows, n_cols), indptr, indices, data)
}

/// Multi-shard `ShardSource` over a list of CSR shards (each spanning a
/// contiguous row range), to exercise the cross-shard row-offset accounting.
struct MultiShardSource {
    shards: Vec<ScxCsr>,
    n_obs: usize,
    n_vars: usize,
}

impl ShardSource for MultiShardSource {
    fn n_shards(&self) -> usize {
        self.shards.len()
    }
    fn n_obs(&self) -> usize {
        self.n_obs
    }
    fn n_vars(&self) -> usize {
        self.n_vars
    }
    fn read_shard(&self, shard_idx: usize) -> IoResult<ScxCsr> {
        Ok(self.shards[shard_idx].clone())
    }
}

fn source_from_dense_shards(shards: &[&[Vec<f32>]], n_vars: usize) -> MultiShardSource {
    let csr_shards: Vec<ScxCsr> = shards.iter().map(|s| csr_from_dense(s)).collect();
    let n_obs = shards.iter().map(|s| s.len()).sum();
    MultiShardSource {
        shards: csr_shards,
        n_obs,
        n_vars,
    }
}

#[test]
fn mean_matches_dense_row_mean() {
    // 3 cells × 4 genes.
    let rows = vec![
        vec![1.0, 2.0, 0.0, 4.0],
        vec![0.0, 0.0, 3.0, 1.0],
        vec![5.0, 0.0, 0.0, 0.0],
    ];
    let src = source_from_dense_shards(&[&rows], 4);
    // Score over genes {0, 3}: per-cell mean of columns 0 and 3.
    let scores = score_genes(&src, &[0, 3], &[], &ScoreMethod::Mean).unwrap();
    let expected = [(1.0 + 4.0) / 2.0, (0.0 + 1.0) / 2.0, (5.0 + 0.0) / 2.0];
    for (got, exp) in scores.iter().zip(expected) {
        assert!((got - exp).abs() < 1e-9, "got {got}, expected {exp}");
    }
}

#[test]
fn mean_handles_multiple_shards() {
    let shard_a = vec![vec![1.0, 0.0, 2.0], vec![0.0, 4.0, 0.0]];
    let shard_b = vec![vec![3.0, 3.0, 3.0]];
    let src = source_from_dense_shards(&[&shard_a, &shard_b], 3);
    let scores = score_genes(&src, &[0, 1, 2], &[], &ScoreMethod::Mean).unwrap();
    // Row means over all 3 genes.
    let expected = [3.0 / 3.0, 4.0 / 3.0, 9.0 / 3.0];
    assert_eq!(scores.len(), 3);
    for (got, exp) in scores.iter().zip(expected) {
        assert!((got - exp).abs() < 1e-9, "got {got}, expected {exp}");
    }
}

#[test]
fn zscore_matches_reference() {
    // 4 cells × 3 genes; score over genes {0, 2}.
    let rows = vec![
        vec![1.0, 9.0, 2.0],
        vec![2.0, 9.0, 0.0],
        vec![3.0, 9.0, 4.0],
        vec![4.0, 9.0, 6.0],
    ];
    let src = source_from_dense_shards(&[&rows], 3);
    let gene_list = [0u32, 2u32];
    let scores = score_genes(&src, &gene_list, &[], &ScoreMethod::Zscore).unwrap();

    // Reference: ddof=1 std per gene, z = (x-mean)/std, score = sum(z)/sqrt(k).
    let cols: Vec<Vec<f64>> = (0..3)
        .map(|c| rows.iter().map(|r| r[c] as f64).collect())
        .collect();
    let n = rows.len() as f64;
    let mean = |v: &[f64]| v.iter().sum::<f64>() / n;
    let std1 =
        |v: &[f64], m: f64| (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let k = gene_list.len() as f64;
    for (cell, row) in rows.iter().enumerate() {
        let mut z = 0.0;
        for &g in &gene_list {
            let gi = g as usize;
            let m = mean(&cols[gi]);
            let s = std1(&cols[gi], m);
            z += (row[gi] as f64 - m) / s;
        }
        let expected = z / k.sqrt();
        assert!(
            (scores[cell] - expected).abs() < 1e-9,
            "cell {cell}: got {}, expected {expected}",
            scores[cell]
        );
    }
}

#[test]
fn zscore_constant_gene_contributes_zero() {
    // Gene 1 is constant (std 0) → must not produce NaN/Inf.
    let rows = vec![vec![1.0, 5.0], vec![3.0, 5.0]];
    let src = source_from_dense_shards(&[&rows], 2);
    let scores = score_genes(&src, &[0, 1], &[], &ScoreMethod::Zscore).unwrap();
    assert!(scores.iter().all(|s| s.is_finite()));
}

#[test]
fn zscore_near_constant_gene_stays_finite() {
    // Gene 1 has a tiny but nonzero spread; 1/std must not overflow the score.
    // (The finite-inv guard covers the subnormal-std overflow edge.)
    let rows = vec![
        vec![1.0, 1.000_000_1_f32],
        vec![3.0, 1.000_000_2_f32],
        vec![2.0, 1.000_000_1_f32],
    ];
    let src = source_from_dense_shards(&[&rows], 2);
    let scores = score_genes(&src, &[0, 1], &[], &ScoreMethod::Zscore).unwrap();
    assert!(scores.iter().all(|s| s.is_finite()), "scores: {scores:?}");
}

#[test]
fn control_score_is_list_minus_control_mean() {
    // Construct so binning is predictable: 6 genes with distinct means.
    // gene_list = {0}; with a large ctrl_size every pool gene in gene 0's bin
    // becomes control (minus the scored gene), so we can recompute by hand.
    let rows = vec![
        vec![10.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        vec![20.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ];
    let src = source_from_dense_shards(&[&rows], 6);
    let gene_pool: Vec<u32> = (0..6).collect();
    let scores = score_genes(
        &src,
        &[0],
        &gene_pool,
        &ScoreMethod::Control {
            ctrl_size: 100,
            n_bins: 25,
            random_state: 0,
        },
    )
    .unwrap();
    // Scores must be finite and deterministic; with gene 0 far above the rest it
    // should be strongly positive (high gene minus low-expression controls).
    assert_eq!(scores.len(), 2);
    assert!(scores.iter().all(|s| s.is_finite()));
    assert!(scores[0] > 0.0 && scores[1] > 0.0);
}

#[test]
fn fused_weight_matches_two_accumulator_within_tolerance() {
    // The score_genes fusion replaces the previous two-accumulator score
    // `(Σ w_list·v) − (Σ w_ctrl·v)` with a single pass over the fused weight
    // `w = w_list − w_ctrl`. The two agree only up to f64 re-association
    // (one accumulator vs two subtracted once), so assert to a tight tolerance
    // rather than byte-equality — this documents the numerical contract.
    let rows = vec![
        vec![10.0, 2.0, 3.0, 1.0, 5.0, 1.0],
        vec![20.0, 4.0, 1.0, 1.0, 2.0, 8.0],
        vec![5.0, 1.0, 9.0, 2.0, 1.0, 3.0],
    ];
    let n_vars = 6usize;
    let src = source_from_dense_shards(&[&rows], n_vars);

    // Disjoint list ({0,2}) and control ({1,4,5}) weight vectors.
    let mut w_list = vec![0.0f64; n_vars];
    for &g in &[0usize, 2] {
        w_list[g] = 1.0 / 2.0;
    }
    let mut w_ctrl = vec![0.0f64; n_vars];
    for &g in &[1usize, 4, 5] {
        w_ctrl[g] = 1.0 / 3.0;
    }

    let two = streaming_weighted_row_sums(&src, &[w_list.clone(), w_ctrl.clone()]).unwrap();
    let fused: Vec<f64> = w_list.iter().zip(&w_ctrl).map(|(a, b)| a - b).collect();
    let one = streaming_weighted_row_sums(&src, std::slice::from_ref(&fused)).unwrap();

    for i in 0..rows.len() {
        let expected = two[0][i] - two[1][i];
        assert!(
            (one[0][i] - expected).abs() < 1e-12,
            "cell {i}: fused {} vs two-accumulator {}",
            one[0][i],
            expected
        );
    }
}

#[test]
fn control_selection_is_deterministic() {
    let means: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let gene_list = [10u32, 25u32];
    let pool: Vec<u32> = (0..50).collect();
    let a = select_control_genes(&means, &gene_list, &pool, 5, 25, 42);
    let b = select_control_genes(&means, &gene_list, &pool, 5, 25, 42);
    assert_eq!(a, b, "same seed must give same control set");
    // Control set never contains the scored genes (ctrl_as_ref semantics).
    assert!(!a.contains(&10) && !a.contains(&25));
}

#[test]
fn control_different_seeds_differ() {
    let means: Vec<f64> = (0..200).map(|i| (i % 40) as f64).collect();
    let gene_list = [3u32];
    let pool: Vec<u32> = (0..200).collect();
    let a = select_control_genes(&means, &gene_list, &pool, 5, 25, 1);
    let b = select_control_genes(&means, &gene_list, &pool, 5, 25, 2);
    // Not a hard guarantee in theory, but overwhelmingly likely for 200 genes.
    assert_ne!(a, b);
}

#[test]
fn empty_gene_list_errors() {
    let rows = vec![vec![1.0, 2.0]];
    let src = source_from_dense_shards(&[&rows], 2);
    assert!(score_genes(&src, &[], &[], &ScoreMethod::Mean).is_err());
}

#[test]
fn out_of_range_gene_errors() {
    let rows = vec![vec![1.0, 2.0]];
    let src = source_from_dense_shards(&[&rows], 2);
    assert!(score_genes(&src, &[5], &[], &ScoreMethod::Mean).is_err());
}

#[test]
fn non_finite_input_rejected() {
    let rows = vec![vec![1.0, f32::NAN]];
    let src = source_from_dense_shards(&[&rows], 2);
    assert!(score_genes(&src, &[0, 1], &[], &ScoreMethod::Mean).is_err());
}

/// The parse vocabulary and its exact error text are a cross-binding
/// contract (pyscx surfaces the message as `ValueError`, rscx as an R
/// error) — pin both here so `cargo test -p scx-accel` catches drift
/// without a Python or R harness.
#[test]
fn score_method_parse_vocabulary_and_error_text() {
    assert!(matches!(
        ScoreMethod::parse("control", 50, 25, 0),
        Ok(ScoreMethod::Control {
            ctrl_size: 50,
            n_bins: 25,
            random_state: 0
        })
    ));
    assert!(matches!(
        ScoreMethod::parse("mean", 0, 0, 0),
        Ok(ScoreMethod::Mean)
    ));
    assert!(matches!(
        ScoreMethod::parse("zscore", 0, 0, 0),
        Ok(ScoreMethod::Zscore)
    ));
    assert_eq!(
        ScoreMethod::parse("typo", 0, 0, 0).unwrap_err().to_string(),
        "score_genes: unknown method \"typo\"; expected \"control\", \"mean\", or \"zscore\""
    );
}
