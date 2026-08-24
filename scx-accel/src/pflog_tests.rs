//! Unit tests for the PFlog (v4) baseline kernel, α estimator, and
//! baseline-aware PCA.
//!
//! The shared [`reference_dense_v4`] helper is the single source of truth for
//! the transform: the f64 reference is exact, and every kernel assertion (which
//! rides on f32 `delta`) holds to ~`1e-5`. v4 shifts **raw counts** by the
//! matrix-wide pseudocount `1/(4α)`; `delta_ij = log1p(4α·x_ij)`, with no depth
//! division and no per-cell proportion.

use super::*;
use crate::pca::pflog_pca;

/// Build the v4 `delta` source from raw rows: `delta_ij = log1p(4α·x_ij)` at
/// nonzero positions (this is what the lazy `Scale{4α}→Log1p` chain produces).
/// Split across the given shard row-groups.
fn delta_source_from_shards(
    shards: &[&[Vec<f32>]],
    n_vars: usize,
    four_alpha: f64,
) -> MultiShardSource {
    let delta_shards: Vec<Vec<Vec<f32>>> = shards
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&v| {
                            if v != 0.0 {
                                (four_alpha * v as f64).ln_1p() as f32
                            } else {
                                0.0
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    let refs: Vec<&[Vec<f32>]> = delta_shards.iter().map(|s| s.as_slice()).collect();
    raw_source_from_shards(&refs, n_vars)
}

/// v4 reference (exact, f64): `z_ij = log1p(4α·x_ij) − (1/D) Σ_k log1p(4α·x_ik)`.
/// Equivalent to the paper's `log(x+1/(4α)) − mean_k log(x+1/(4α))` (the folded
/// `log(4α)` constant cancels in the centering).
fn reference_dense_v4(rows: &[Vec<f32>], four_alpha: f64) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|row| {
            let logs: Vec<f64> = row
                .iter()
                .map(|&v| (four_alpha * v as f64).ln_1p())
                .collect();
            let mean = logs.iter().sum::<f64>() / logs.len() as f64;
            logs.iter().map(|&l| l - mean).collect()
        })
        .collect()
}

/// Reconstruct exact dense Z from the compact (delta, baseline) form.
fn reconstruct_dense(delta_rows: &[Vec<f32>], baseline: &[f64]) -> Vec<Vec<f64>> {
    delta_rows
        .iter()
        .enumerate()
        .map(|(i, row)| row.iter().map(|&d| d as f64 + baseline[i]).collect())
        .collect()
}

/// Materialize the dense v4 delta rows for a fixture (0 at zeros, since
/// `log1p(4α·0) = 0`), for reconstruction checks.
fn dense_delta_rows(rows: &[Vec<f32>], four_alpha: f64) -> Vec<Vec<f32>> {
    rows.iter()
        .map(|row| {
            row.iter()
                .map(|&v| {
                    if v != 0.0 {
                        (four_alpha * v as f64).ln_1p() as f32
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}

fn fixture() -> Vec<Vec<f32>> {
    vec![
        vec![0.0, 1.0, 3.0, 0.0],
        vec![2.0, 0.0, 0.0, 5.0],
        vec![1.0, 1.0, 1.0, 1.0],
    ]
}

// --- delta source + baseline reconstruct the reference ----------------------

/// The delta source plus baseline must reconstruct the v4 reference exactly.
#[test]
fn dense_equivalence_default_alpha() {
    let rows = fixture();
    let n_vars = 4;
    let four_alpha = 4.0; // α = 1
    let raw = raw_source_from_shards(&[&rows], n_vars);
    let baseline = pflog_baseline_from_raw(&raw, four_alpha).unwrap();

    let delta_rows = dense_delta_rows(&rows, four_alpha);
    let actual = reconstruct_dense(&delta_rows, &baseline);
    let expected = reference_dense_v4(&rows, four_alpha);
    for (a_row, e_row) in actual.iter().zip(expected.iter()) {
        for (&a, &e) in a_row.iter().zip(e_row.iter()) {
            assert!((a - e).abs() <= 1e-5, "got {a}, expected {e}");
        }
    }
}

#[test]
fn dense_equivalence_various_alpha() {
    let rows = vec![vec![0.0, 4.0, 0.0, 2.0], vec![10.0, 0.0, 1.0, 0.0]];
    let n_vars = 4;
    for &four_alpha in &[0.4f64, 2.0, 4.0, 8.0] {
        let raw = raw_source_from_shards(&[&rows], n_vars);
        let baseline = pflog_baseline_from_raw(&raw, four_alpha).unwrap();
        let delta_rows = dense_delta_rows(&rows, four_alpha);
        let actual = reconstruct_dense(&delta_rows, &baseline);
        let expected = reference_dense_v4(&rows, four_alpha);
        for (a_row, e_row) in actual.iter().zip(expected.iter()) {
            for (&a, &e) in a_row.iter().zip(e_row.iter()) {
                assert!(
                    (a - e).abs() <= 1e-5,
                    "four_alpha={four_alpha}: got {a}, expected {e}"
                );
            }
        }
    }
}

/// Exact transform rows sum to zero (the centering property).
#[test]
fn row_sums_are_zero() {
    let rows = fixture();
    let n_vars = 4;
    let four_alpha = 4.0;
    let raw = raw_source_from_shards(&[&rows], n_vars);
    let baseline = pflog_baseline_from_raw(&raw, four_alpha).unwrap();
    let delta_rows = dense_delta_rows(&rows, four_alpha);
    let z = reconstruct_dense(&delta_rows, &baseline);
    for row in &z {
        let s: f64 = row.iter().sum();
        assert!(s.abs() <= 1e-5, "row sum {s} not ~0");
    }
}

/// The centering denominator is `n_vars` (total features), NOT row `nnz`.
/// `[0,1,3,0]` has nnz=2 but D=4; dividing by nnz would give a different value.
#[test]
fn centering_denominator_is_n_vars_not_nnz() {
    let rows = vec![vec![0.0f32, 1.0, 3.0, 0.0]];
    let n_vars = 4;
    let four_alpha = 4.0;
    let raw = raw_source_from_shards(&[&rows], n_vars);
    let baseline = pflog_baseline_from_raw(&raw, four_alpha).unwrap();

    let d1 = (four_alpha * 1.0).ln_1p();
    let d3 = (four_alpha * 3.0).ln_1p();
    let expected = -(d1 + d3) / 4.0; // divide by n_vars = 4
    let wrong = -(d1 + d3) / 2.0; // nnz = 2
    assert!(
        (baseline[0] - expected).abs() <= 1e-9,
        "baseline {} != n_vars-denominator {expected}",
        baseline[0]
    );
    assert!(
        (baseline[0] - wrong).abs() > 1e-6,
        "baseline must NOT use the nnz denominator"
    );
}

/// Baseline is identical whether the rows arrive in one shard or many.
#[test]
fn baseline_invariant_across_shards() {
    let s0 = vec![vec![0.0f32, 1.0, 3.0, 0.0]];
    let s1 = vec![vec![2.0f32, 0.0, 0.0, 5.0], vec![1.0, 1.0, 1.0, 1.0]];
    let n_vars = 4;
    let four_alpha = 4.0;

    let single = {
        let mut all = s0.clone();
        all.extend(s1.clone());
        let src = raw_source_from_shards(&[&all], n_vars);
        pflog_baseline_from_raw(&src, four_alpha).unwrap()
    };
    let multi = {
        let src = raw_source_from_shards(&[&s0, &s1], n_vars);
        pflog_baseline_from_raw(&src, four_alpha).unwrap()
    };
    for (a, b) in single.iter().zip(multi.iter()) {
        assert!((a - b).abs() <= 1e-12, "{a} != {b}");
    }
}

/// `pflog_baseline_from_delta` (sum the delta source) equals the raw kernel —
/// both are exact representations of the same baseline for the same `four_alpha`.
#[test]
fn baseline_from_delta_matches_raw_kernel() {
    let rows = fixture();
    let n_vars = 4;
    let four_alpha = 4.0;
    let raw = raw_source_from_shards(&[&rows], n_vars);
    let from_raw = pflog_baseline_from_raw(&raw, four_alpha).unwrap();

    let delta = delta_source_from_shards(&[&rows], n_vars, four_alpha);
    let from_delta = pflog_baseline_from_delta(&delta).unwrap();

    for (a, b) in from_raw.iter().zip(from_delta.iter()) {
        assert!((a - b).abs() <= 1e-5, "{a} != {b}");
    }
}

/// v4 improvement: an empty (all-zero) cell yields `baseline = 0` and no error
/// (v2 rejected non-positive depth).
#[test]
fn empty_cell_yields_zero_baseline() {
    let rows = vec![vec![0.0f32, 0.0, 0.0, 0.0], vec![1.0, 2.0, 0.0, 3.0]];
    let n_vars = 4;
    let four_alpha = 4.0;
    let raw = raw_source_from_shards(&[&rows], n_vars);
    let baseline = pflog_baseline_from_raw(&raw, four_alpha).unwrap();
    assert!(
        baseline[0].abs() <= 1e-12,
        "empty cell baseline {}",
        baseline[0]
    );
    let expected1 =
        -((four_alpha * 1.0).ln_1p() + (four_alpha * 2.0).ln_1p() + (four_alpha * 3.0).ln_1p())
            / 4.0;
    assert!(
        (baseline[1] - expected1).abs() <= 1e-9,
        "{} != {expected1}",
        baseline[1]
    );
}

// --- validation ------------------------------------------------------------

#[test]
fn rejects_nonpositive_four_alpha() {
    let rows = fixture();
    let raw = raw_source_from_shards(&[&rows], 4);
    assert!(pflog_baseline_from_raw(&raw, 0.0).is_err());
    assert!(pflog_baseline_from_raw(&raw, -1.0).is_err());
}

#[test]
fn rejects_non_finite_four_alpha() {
    let rows = fixture();
    let raw = raw_source_from_shards(&[&rows], 4);
    assert!(pflog_baseline_from_raw(&raw, f64::INFINITY).is_err());
    assert!(pflog_baseline_from_raw(&raw, f64::NAN).is_err());
}

#[test]
fn ensure_finite_rejects_nan_and_inf() {
    // `ensure_finite` is private to the parent module (in scope via `super::*`).
    assert!(ensure_finite(&[1.0, 2.0, 3.0]).is_ok());
    assert!(ensure_finite(&[1.0, f32::NAN, 3.0]).is_err());
    assert!(ensure_finite(&[1.0, f32::INFINITY]).is_err());
    assert!(ensure_finite(&[f32::NEG_INFINITY]).is_err());
}

/// A malformed source whose shard rows exceed the declared `n_obs` must return
/// a clean `ShapeError`, not panic on out-of-bounds indexing.
#[test]
fn baseline_rejects_shard_rows_exceeding_n_obs() {
    let rows = vec![
        vec![1.0f32, 1.0, 1.0, 1.0],
        vec![2.0, 0.0, 1.0, 0.0],
        vec![0.0, 3.0, 0.0, 1.0],
    ];
    let bad = MultiShardSource {
        shards: vec![csr_from_dense(&rows)],
        n_obs: 2, // declares 2 but serves 3
        n_vars: 4,
    };
    let err = pflog_baseline_from_raw(&bad, 4.0);
    assert!(matches!(err, Err(AccelError::ShapeError(_))));
}

// --- PCA: reconstruction matches the dense transform -----------------------

/// Dense matmul: (n × p row-major) @ (p × m row-major) → n × m.
fn matmul(a: &[f64], b: &[f64], n: usize, p: usize, m: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0f64; m]; n];
    for i in 0..n {
        for k in 0..p {
            let aik = a[i * p + k];
            for j in 0..m {
                out[i][j] += aik * b[k * m + j];
            }
        }
    }
    out
}

/// Column-center a dense matrix (subtract per-column mean).
fn column_center(z: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = z.len();
    let m = z[0].len();
    let mut mu = vec![0.0f64; m];
    for row in z {
        for (j, &v) in row.iter().enumerate() {
            mu[j] += v;
        }
    }
    for v in &mut mu {
        *v /= n as f64;
    }
    z.iter()
        .map(|row| row.iter().enumerate().map(|(j, &v)| v - mu[j]).collect())
        .collect()
}

fn pca_fixture() -> Vec<Vec<f32>> {
    vec![
        vec![0.0f32, 1.0, 3.0, 0.0],
        vec![2.0, 0.0, 0.0, 5.0],
        vec![1.0, 1.0, 1.0, 1.0],
        vec![4.0, 2.0, 0.0, 1.0],
        vec![0.0, 0.0, 6.0, 2.0],
        vec![3.0, 3.0, 1.0, 0.0],
    ]
}

/// Full-rank PCA reconstruction `embeddings @ components` must equal the
/// column-centered exact transform (exercises the baseline-aware SpMM incl.
/// μ-includes-baseline).
#[test]
fn pca_zero_centered_reconstructs_centered_transform() {
    let rows = pca_fixture();
    let n_vars = 4;
    let four_alpha = 4.0;
    let raw = raw_source_from_shards(&[&rows], n_vars);
    let baseline = pflog_baseline_from_raw(&raw, four_alpha).unwrap();

    let delta = delta_source_from_shards(&[&rows], n_vars, four_alpha);
    let n_components = 4; // full column rank → exact reconstruction
    let res = pflog_pca(&delta, &baseline, n_components, 0, 4, true, 42).unwrap();

    let recon = matmul(&res.embeddings, &res.components, 6, n_components, n_vars);
    let expected = column_center(&reference_dense_v4(&rows, four_alpha));
    for (r_row, e_row) in recon.iter().zip(expected.iter()) {
        for (&r, &e) in r_row.iter().zip(e_row.iter()) {
            assert!((r - e).abs() <= 1e-4, "centered recon {r} != {e}");
        }
    }
}

/// Uncentered PCA (`zero_center=false`) reconstructs the exact transform Z.
#[test]
fn pca_uncentered_reconstructs_transform() {
    let rows = pca_fixture();
    let n_vars = 4;
    let four_alpha = 4.0;
    let raw = raw_source_from_shards(&[&rows], n_vars);
    let baseline = pflog_baseline_from_raw(&raw, four_alpha).unwrap();

    let delta = delta_source_from_shards(&[&rows], n_vars, four_alpha);
    let n_components = 4;
    let res = pflog_pca(&delta, &baseline, n_components, 0, 4, false, 7).unwrap();

    let recon = matmul(&res.embeddings, &res.components, 6, n_components, n_vars);
    let expected = reference_dense_v4(&rows, four_alpha);
    for (r_row, e_row) in recon.iter().zip(expected.iter()) {
        for (&r, &e) in r_row.iter().zip(e_row.iter()) {
            assert!((r - e).abs() <= 1e-4, "uncentered recon {r} != {e}");
        }
    }
}

/// PCA streamed across multiple shards matches the single-shard result.
#[test]
fn pca_invariant_across_shards() {
    let s0 = vec![vec![0.0f32, 1.0, 3.0, 0.0], vec![2.0, 0.0, 0.0, 5.0]];
    let s1 = vec![
        vec![1.0f32, 1.0, 1.0, 1.0],
        vec![4.0, 2.0, 0.0, 1.0],
        vec![0.0, 0.0, 6.0, 2.0],
        vec![3.0, 3.0, 1.0, 0.0],
    ];
    let mut all = s0.clone();
    all.extend(s1.clone());
    let n_vars = 4;
    let four_alpha = 4.0;

    let raw_all = raw_source_from_shards(&[&all], n_vars);
    let baseline = pflog_baseline_from_raw(&raw_all, four_alpha).unwrap();

    let single = pflog_pca(
        &delta_source_from_shards(&[&all], n_vars, four_alpha),
        &baseline,
        3,
        2,
        4,
        true,
        42,
    )
    .unwrap();
    let multi = pflog_pca(
        &delta_source_from_shards(&[&s0, &s1], n_vars, four_alpha),
        &baseline,
        3,
        2,
        4,
        true,
        42,
    )
    .unwrap();

    for (a, b) in single
        .variance_explained
        .iter()
        .zip(multi.variance_explained.iter())
    {
        assert!((a - b).abs() <= 1e-6, "variance {a} != {b}");
    }
}

/// Zero-centered PCA reconstruction with a non-unit `α`, guarding the
/// `four_alpha` plumbing through baseline + delta + the offset SpMM.
#[test]
fn pca_centered_reconstructs_with_general_alpha() {
    let rows = pca_fixture();
    let n_vars = 4;
    let four_alpha = 2.0; // α = 0.5
    let raw = raw_source_from_shards(&[&rows], n_vars);
    let baseline = pflog_baseline_from_raw(&raw, four_alpha).unwrap();

    let delta = delta_source_from_shards(&[&rows], n_vars, four_alpha);
    let res = pflog_pca(&delta, &baseline, 4, 0, 4, true, 11).unwrap();

    let recon = matmul(&res.embeddings, &res.components, 6, 4, n_vars);
    let expected = column_center(&reference_dense_v4(&rows, four_alpha));
    for (r_row, e_row) in recon.iter().zip(expected.iter()) {
        for (&r, &e) in r_row.iter().zip(e_row.iter()) {
            assert!(
                (r - e).abs() <= 1e-4,
                "general-alpha centered recon {r} != {e}"
            );
        }
    }
}

// --- v4 α estimator --------------------------------------------------------

/// Hand-computed single gene: cells [1, 5] → mean 3, var 8 (n−1 denom),
/// α = (8−3)/9 = 5/9, pseudocount = 1/(4α) = 0.45.
#[test]
fn estimate_alpha_known_single_gene() {
    let rows = vec![vec![1.0f32], vec![5.0f32]];
    let raw = raw_source_from_shards(&[&rows], 1);
    let est = estimate_alpha(&raw, &AlphaOptions::default()).unwrap();
    assert_eq!(est.n_genes_used, 1);
    assert!(!est.fell_back);
    assert!((est.alpha - 5.0 / 9.0).abs() <= 1e-12, "α={}", est.alpha);
    assert!(
        (est.pseudocount - 0.45).abs() <= 1e-12,
        "pc={}",
        est.pseudocount
    );
}

/// Every gene with a mean above `mu_min` enters the pool, including
/// under-dispersed ones (§7.14). Here `g0` has α = 1/9 and `g3` α = 10/9, `g1`
/// is constant so its honest MoM estimate is **negative** (α = (0−4)/16 =
/// −0.25), and `g2` is all-zero so it has no mean to speak of and is the one
/// gene `mu_min` excludes. Pool = {−0.25, 1/9, 10/9} ⇒ `n_genes_used == 3` and
/// the median is 1/9.
///
/// The previous version of this test asserted `n_genes_used == 2` and a median
/// of `(1/9 + 10/9)/2` — it encoded the `var > mean` pre-filter as the
/// specification. Dropping the negative half of the pool before the median is
/// what biased α high on any matrix that has under-dispersed genes, which is
/// every real one.
#[test]
fn estimate_alpha_pools_every_gene_with_a_mean() {
    let rows = vec![
        vec![1.0f32, 4.0, 0.0, 2.0],
        vec![5.0, 4.0, 0.0, 0.0],
        vec![3.0, 4.0, 0.0, 7.0],
    ];
    let raw = raw_source_from_shards(&[&rows], 4);
    let est = estimate_alpha(&raw, &AlphaOptions::default()).unwrap();
    assert_eq!(
        est.n_genes_used, 3,
        "the constant gene has a mean and belongs in the pool"
    );
    assert!(!est.fell_back);
    let expected = 1.0 / 9.0;
    assert!(
        (est.alpha - expected).abs() <= 1e-12,
        "α={} (expected the median of {{-0.25, 1/9, 10/9}})",
        est.alpha
    );
}

/// All-zero matrix: no candidate → fallback to α=0.25 (pseudocount 1.0), no error.
#[test]
fn estimate_alpha_falls_back_on_all_zero() {
    let rows = vec![vec![0.0f32, 0.0, 0.0], vec![0.0, 0.0, 0.0]];
    let raw = raw_source_from_shards(&[&rows], 3);
    let est = estimate_alpha(&raw, &AlphaOptions::default()).unwrap();
    assert!(est.fell_back);
    assert_eq!(est.n_genes_used, 0);
    assert_eq!(est.alpha, 0.25);
    assert!((est.pseudocount - 1.0).abs() <= 1e-12);
}

/// Under-dispersed input (constant columns → var 0 < mean for every gene): the
/// pool is **not** empty — every gene contributes a negative α_g — and the
/// pooled median is negative, so the documented fallback fires.
///
/// `n_genes_used == 3`, not `0`, is the assertion that distinguishes the fixed
/// estimator from the one that dropped under-dispersed genes before pooling.
/// With the pre-filter in place this matrix reached the fallback for the wrong
/// reason ("nothing to pool"), and a matrix with a *few* over-dispersed genes
/// reached no fallback at all and reported those few genes' dispersion as the
/// whole matrix's — see `pflog_reference_tests.rs`'s fixture C.
#[test]
fn estimate_alpha_falls_back_when_the_pooled_median_is_not_positive() {
    let rows = vec![
        vec![2.0f32, 3.0, 4.0],
        vec![2.0, 3.0, 4.0],
        vec![2.0, 3.0, 4.0],
    ];
    let raw = raw_source_from_shards(&[&rows], 3);
    let est = estimate_alpha(&raw, &AlphaOptions::default()).unwrap();
    assert!(est.fell_back);
    assert_eq!(
        est.n_genes_used, 3,
        "all three genes have a mean above mu_min and belong in the pool"
    );
    assert_eq!(est.alpha, 0.25);
}

/// A non-finite count is rejected via `streaming_mean_var`'s input guard.
#[test]
fn estimate_alpha_rejects_non_finite() {
    let rows = vec![vec![1.0f32, f32::NAN, 3.0], vec![2.0, 1.0, 0.0]];
    let raw = raw_source_from_shards(&[&rows], 3);
    assert!(estimate_alpha(&raw, &AlphaOptions::default()).is_err());
}

/// `n_vars == 0` is rejected before any streaming pass.
#[test]
fn estimate_alpha_rejects_zero_n_vars() {
    let src = MultiShardSource {
        shards: vec![],
        n_obs: 0,
        n_vars: 0,
    };
    assert!(matches!(
        estimate_alpha(&src, &AlphaOptions::default()),
        Err(AccelError::InvalidInput(_))
    ));
}
