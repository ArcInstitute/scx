//! Unit tests for the PFlog1pPF baseline kernel and baseline-aware PCA.
//!
//! The shared [`reference_dense`] helper (mirroring the spec's §11.1 Python
//! reference) is the single source of truth: the f64 reference is exact, and
//! every kernel assertion (which rides on f32 `delta`) holds to ~`1e-5`.

use super::*;
use crate::pca::pflog1ppf_pca;
use scx_format_io::{Result as IoResult, ShardSource};
use scx_sparse::ScxCsr;

/// Build a CSR shard from a dense row-major matrix (drops zeros).
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

/// Multi-shard raw-count `ShardSource` over a list of CSR shards.
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

/// Raw-count source split into the given per-shard row groups.
fn raw_source_from_shards(shards: &[&[Vec<f32>]], n_vars: usize) -> MultiShardSource {
    let csr_shards: Vec<ScxCsr> = shards.iter().map(|s| csr_from_dense(s)).collect();
    let n_obs = shards.iter().map(|s| s.len()).sum();
    MultiShardSource {
        shards: csr_shards,
        n_obs,
        n_vars,
    }
}

/// Build the `delta` source from raw rows: `delta_ij = log1p(x_ij / (c·s_i))`
/// at nonzero positions (this is what the lazy `NormalizeTotal{1/c}→Log1p`
/// chain produces). Split across the given shard row-groups.
fn delta_source_from_shards(shards: &[&[Vec<f32>]], n_vars: usize, c: f64) -> MultiShardSource {
    let delta_shards: Vec<Vec<Vec<f32>>> = shards
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|row| {
                    let depth: f64 = row.iter().map(|&v| v as f64).sum();
                    row.iter()
                        .map(|&v| {
                            if v != 0.0 {
                                (v as f64 / (c * depth)).ln_1p() as f32
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

/// Spec §11.1 reference (exact, f64): the single source of truth.
///   z_ij = log(x_ij/s_i + c) − mean_j log(x_ij/s_i + c)
fn reference_dense(rows: &[Vec<f32>], c: f64) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|row| {
            let depth: f64 = row.iter().map(|&v| v as f64).sum();
            let logs: Vec<f64> = row.iter().map(|&v| (v as f64 / depth + c).ln()).collect();
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

fn fixture() -> Vec<Vec<f32>> {
    vec![
        vec![0.0, 1.0, 3.0, 0.0],
        vec![2.0, 0.0, 0.0, 5.0],
        vec![1.0, 1.0, 1.0, 1.0],
    ]
}

// --- delta source matches the reference's delta part -----------------------

/// The delta source plus baseline must reconstruct the spec reference exactly.
#[test]
fn dense_equivalence_default_c() {
    let rows = fixture();
    let n_vars = 4;
    let c = 1.0;
    let raw = raw_source_from_shards(&[&rows], n_vars);
    let depths = pflog1ppf_cell_depths(&raw).unwrap();
    let baseline = pflog1ppf_baseline(&raw, &depths, c).unwrap();

    // delta values for this fixture (same recipe as the lazy chain).
    let delta_rows: Vec<Vec<f32>> = rows
        .iter()
        .map(|row| {
            let depth: f64 = row.iter().map(|&v| v as f64).sum();
            row.iter()
                .map(|&v| {
                    if v != 0.0 {
                        (v as f64 / (c * depth)).ln_1p() as f32
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();

    let actual = reconstruct_dense(&delta_rows, &baseline);
    let expected = reference_dense(&rows, c);
    for (a_row, e_row) in actual.iter().zip(expected.iter()) {
        for (&a, &e) in a_row.iter().zip(e_row.iter()) {
            assert!((a - e).abs() <= 1e-5, "got {a}, expected {e}");
        }
    }
}

#[test]
fn dense_equivalence_general_c() {
    let rows = vec![vec![0.0, 4.0, 0.0, 2.0], vec![10.0, 0.0, 1.0, 0.0]];
    let n_vars = 4;
    for &c in &[0.1f64, 0.5, 1.0, 2.0] {
        let raw = raw_source_from_shards(&[&rows], n_vars);
        let depths = pflog1ppf_cell_depths(&raw).unwrap();
        let baseline = pflog1ppf_baseline(&raw, &depths, c).unwrap();
        let delta_rows: Vec<Vec<f32>> = rows
            .iter()
            .map(|row| {
                let depth: f64 = row.iter().map(|&v| v as f64).sum();
                row.iter()
                    .map(|&v| {
                        if v != 0.0 {
                            (v as f64 / (c * depth)).ln_1p() as f32
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();
        let actual = reconstruct_dense(&delta_rows, &baseline);
        let expected = reference_dense(&rows, c);
        for (a_row, e_row) in actual.iter().zip(expected.iter()) {
            for (&a, &e) in a_row.iter().zip(e_row.iter()) {
                assert!((a - e).abs() <= 1e-5, "c={c}: got {a}, expected {e}");
            }
        }
    }
}

/// Exact transform rows sum to zero (the centering property).
#[test]
fn row_sums_are_zero() {
    let rows = fixture();
    let n_vars = 4;
    let c = 1.0;
    let raw = raw_source_from_shards(&[&rows], n_vars);
    let depths = pflog1ppf_cell_depths(&raw).unwrap();
    let baseline = pflog1ppf_baseline(&raw, &depths, c).unwrap();
    let delta_rows: Vec<Vec<f32>> = rows
        .iter()
        .map(|row| {
            let depth: f64 = row.iter().map(|&v| v as f64).sum();
            row.iter()
                .map(|&v| {
                    if v != 0.0 {
                        (v as f64 / (c * depth)).ln_1p() as f32
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    let z = reconstruct_dense(&delta_rows, &baseline);
    for row in &z {
        let s: f64 = row.iter().sum();
        assert!(s.abs() <= 1e-5, "row sum {s} not ~0");
    }
}

/// Acceptance #9: the centering denominator is `n_vars` (total features), NOT
/// row `nnz`. A dense row [1,1,1,1] (depth 4, all four entries equal) has
/// delta_ij = log1p(0.25) for every j, so baseline = -log1p(0.25). Dividing by
/// nnz (=4) gives the same here, so use a row with zeros to disambiguate:
/// [0,1,3,0] has nnz=2 but D=4.
#[test]
fn centering_denominator_is_n_vars_not_nnz() {
    let rows = vec![vec![0.0f32, 1.0, 3.0, 0.0]];
    let n_vars = 4;
    let c = 1.0;
    let raw = raw_source_from_shards(&[&rows], n_vars);
    let depths = pflog1ppf_cell_depths(&raw).unwrap();
    let baseline = pflog1ppf_baseline(&raw, &depths, c).unwrap();

    let depth = 4.0f64;
    let d1 = (1.0f64 / (c * depth)).ln_1p();
    let d3 = (3.0f64 / (c * depth)).ln_1p();
    // Correct: divide by n_vars = 4.
    let expected = -(d1 + d3) / 4.0;
    // Wrong (nnz=2) would be -(d1+d3)/2.
    let wrong = -(d1 + d3) / 2.0;
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
    let c = 1.0;

    let single = {
        let mut all = s0.clone();
        all.extend(s1.clone());
        let src = raw_source_from_shards(&[&all], n_vars);
        let d = pflog1ppf_cell_depths(&src).unwrap();
        pflog1ppf_baseline(&src, &d, c).unwrap()
    };
    let multi = {
        let src = raw_source_from_shards(&[&s0, &s1], n_vars);
        let d = pflog1ppf_cell_depths(&src).unwrap();
        pflog1ppf_baseline(&src, &d, c).unwrap()
    };
    for (a, b) in single.iter().zip(multi.iter()) {
        assert!((a - b).abs() <= 1e-12, "{a} != {b}");
    }
}

/// `pflog1ppf_baseline_from_delta` (sum the delta source) equals the raw+depths
/// kernel — both are exact representations of the same baseline.
#[test]
fn baseline_from_delta_matches_raw_kernel() {
    let rows = fixture();
    let n_vars = 4;
    let c = 1.0;
    let raw = raw_source_from_shards(&[&rows], n_vars);
    let depths = pflog1ppf_cell_depths(&raw).unwrap();
    let from_raw = pflog1ppf_baseline(&raw, &depths, c).unwrap();

    let delta = delta_source_from_shards(&[&rows], n_vars, c);
    let from_delta = pflog1ppf_baseline_from_delta(&delta).unwrap();

    for (a, b) in from_raw.iter().zip(from_delta.iter()) {
        assert!((a - b).abs() <= 1e-5, "{a} != {b}");
    }
}

// --- validation ------------------------------------------------------------

#[test]
fn rejects_nonpositive_c() {
    let rows = fixture();
    let raw = raw_source_from_shards(&[&rows], 4);
    let depths = pflog1ppf_cell_depths(&raw).unwrap();
    assert!(pflog1ppf_baseline(&raw, &depths, 0.0).is_err());
    assert!(pflog1ppf_baseline(&raw, &depths, -1.0).is_err());
}

#[test]
fn rejects_zero_depth() {
    let rows = vec![vec![0.0f32, 0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0, 1.0]];
    let raw = raw_source_from_shards(&[&rows], 4);
    let depths = pflog1ppf_cell_depths(&raw).unwrap();
    assert!(pflog1ppf_baseline(&raw, &depths, 1.0).is_err());
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

/// Full-rank PCA reconstruction `embeddings @ components` must equal the
/// column-centered exact transform (rotation/sign-invariant check that
/// exercises the baseline-aware SpMM, incl. μ-includes-baseline).
#[test]
fn pca_zero_centered_reconstructs_centered_transform() {
    let rows = vec![
        vec![0.0f32, 1.0, 3.0, 0.0],
        vec![2.0, 0.0, 0.0, 5.0],
        vec![1.0, 1.0, 1.0, 1.0],
        vec![4.0, 2.0, 0.0, 1.0],
        vec![0.0, 0.0, 6.0, 2.0],
        vec![3.0, 3.0, 1.0, 0.0],
    ];
    let n_vars = 4;
    let c = 1.0;
    let raw = raw_source_from_shards(&[&rows], n_vars);
    let depths = pflog1ppf_cell_depths(&raw).unwrap();
    let baseline = pflog1ppf_baseline(&raw, &depths, c).unwrap();

    let delta = delta_source_from_shards(&[&rows], n_vars, c);
    let n_components = 4; // full column rank → exact reconstruction
    let res = pflog1ppf_pca(&delta, &baseline, n_components, 0, 4, true, 42).unwrap();

    let recon = matmul(&res.embeddings, &res.components, 6, n_components, n_vars);
    let expected = column_center(&reference_dense(&rows, c));
    for (r_row, e_row) in recon.iter().zip(expected.iter()) {
        for (&r, &e) in r_row.iter().zip(e_row.iter()) {
            assert!((r - e).abs() <= 1e-4, "centered recon {r} != {e}");
        }
    }
}

/// Uncentered PCA (`zero_center=false`) reconstructs the exact transform Z.
#[test]
fn pca_uncentered_reconstructs_transform() {
    let rows = vec![
        vec![0.0f32, 1.0, 3.0, 0.0],
        vec![2.0, 0.0, 0.0, 5.0],
        vec![1.0, 1.0, 1.0, 1.0],
        vec![4.0, 2.0, 0.0, 1.0],
        vec![0.0, 0.0, 6.0, 2.0],
        vec![3.0, 3.0, 1.0, 0.0],
    ];
    let n_vars = 4;
    let c = 1.0;
    let raw = raw_source_from_shards(&[&rows], n_vars);
    let depths = pflog1ppf_cell_depths(&raw).unwrap();
    let baseline = pflog1ppf_baseline(&raw, &depths, c).unwrap();

    let delta = delta_source_from_shards(&[&rows], n_vars, c);
    let n_components = 4;
    let res = pflog1ppf_pca(&delta, &baseline, n_components, 0, 4, false, 7).unwrap();

    let recon = matmul(&res.embeddings, &res.components, 6, n_components, n_vars);
    let expected = reference_dense(&rows, c);
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
    let c = 1.0;

    let raw_all = raw_source_from_shards(&[&all], n_vars);
    let depths = pflog1ppf_cell_depths(&raw_all).unwrap();
    let baseline = pflog1ppf_baseline(&raw_all, &depths, c).unwrap();

    let single = pflog1ppf_pca(
        &delta_source_from_shards(&[&all], n_vars, c),
        &baseline,
        3,
        2,
        4,
        true,
        42,
    )
    .unwrap();
    let multi = pflog1ppf_pca(
        &delta_source_from_shards(&[&s0, &s1], n_vars, c),
        &baseline,
        3,
        2,
        4,
        true,
        42,
    )
    .unwrap();

    // Variance explained is rotation/sign invariant — compare directly.
    for (a, b) in single
        .variance_explained
        .iter()
        .zip(multi.variance_explained.iter())
    {
        assert!((a - b).abs() <= 1e-6, "variance {a} != {b}");
    }
}

/// Zero-centered PCA reconstruction with a non-unit shift `c ≠ 1`, guarding
/// the `c` plumbing through baseline + delta + the offset SpMM.
#[test]
fn pca_centered_reconstructs_with_general_c() {
    let rows = vec![
        vec![0.0f32, 1.0, 3.0, 0.0],
        vec![2.0, 0.0, 0.0, 5.0],
        vec![1.0, 1.0, 1.0, 1.0],
        vec![4.0, 2.0, 0.0, 1.0],
        vec![0.0, 0.0, 6.0, 2.0],
        vec![3.0, 3.0, 1.0, 0.0],
    ];
    let n_vars = 4;
    let c = 0.5;
    let raw = raw_source_from_shards(&[&rows], n_vars);
    let depths = pflog1ppf_cell_depths(&raw).unwrap();
    let baseline = pflog1ppf_baseline(&raw, &depths, c).unwrap();

    let delta = delta_source_from_shards(&[&rows], n_vars, c);
    let res = pflog1ppf_pca(&delta, &baseline, 4, 0, 4, true, 11).unwrap();

    let recon = matmul(&res.embeddings, &res.components, 6, 4, n_vars);
    let expected = column_center(&reference_dense(&rows, c));
    for (r_row, e_row) in recon.iter().zip(expected.iter()) {
        for (&r, &e) in r_row.iter().zip(e_row.iter()) {
            assert!((r - e).abs() <= 1e-4, "general-c centered recon {r} != {e}");
        }
    }
}

// --- ensure_finite + defensive shape guards --------------------------------

#[test]
fn ensure_finite_rejects_nan_and_inf() {
    // `ensure_finite` is private to the parent module (in scope via `super::*`).
    assert!(ensure_finite(&[1.0, 2.0, 3.0]).is_ok());
    assert!(ensure_finite(&[1.0, f32::NAN, 3.0]).is_err());
    assert!(ensure_finite(&[1.0, f32::INFINITY]).is_err());
    assert!(ensure_finite(&[f32::NEG_INFINITY]).is_err());
}

#[test]
fn rejects_non_finite_c() {
    let rows = fixture();
    let raw = raw_source_from_shards(&[&rows], 4);
    let depths = pflog1ppf_cell_depths(&raw).unwrap();
    assert!(pflog1ppf_baseline(&raw, &depths, f64::INFINITY).is_err());
    assert!(pflog1ppf_baseline(&raw, &depths, f64::NAN).is_err());
}

// --- v4 α estimator --------------------------------------------------------

/// Independent f64 reference for `estimate_alpha`: per-gene MoM
/// `α_g = (var − mean)/mean²` (Bessel-corrected, matching
/// `streaming_mean_var`), filtered by `mean > mu_min && var > mean`, pooled by
/// median. Returns `(median_alpha, n_genes_used)`.
fn reference_alpha(rows: &[Vec<f32>], n_vars: usize, mu_min: f64) -> (f64, usize) {
    let n = rows.len() as f64;
    let denom = (n - 1.0).max(1.0);
    let mut cand: Vec<f64> = Vec::new();
    for g in 0..n_vars {
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for row in rows {
            let v = row[g] as f64;
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / n;
        let var = ((sum_sq - n * mean * mean) / denom).max(0.0);
        if mean > mu_min && var > mean {
            let a = (var - mean) / (mean * mean);
            if a.is_finite() && a > 0.0 {
                cand.push(a);
            }
        }
    }
    let k = cand.len();
    cand.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = if k == 0 {
        f64::NAN
    } else if k % 2 == 1 {
        cand[k / 2]
    } else {
        0.5 * (cand[k / 2 - 1] + cand[k / 2])
    };
    (med, k)
}

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

/// `estimate_alpha` matches the independent f64 reference on a mixed matrix.
#[test]
fn estimate_alpha_matches_reference() {
    let rows = vec![
        vec![0.0f32, 5.0, 2.0, 0.0],
        vec![3.0, 0.0, 8.0, 1.0],
        vec![1.0, 2.0, 0.0, 4.0],
        vec![7.0, 1.0, 3.0, 0.0],
        vec![0.0, 9.0, 1.0, 2.0],
    ];
    let n_vars = 4;
    let opts = AlphaOptions::default();
    let raw = raw_source_from_shards(&[&rows], n_vars);
    let est = estimate_alpha(&raw, &opts).unwrap();
    let (ref_alpha, ref_k) = reference_alpha(&rows, n_vars, opts.mu_min);
    assert_eq!(est.n_genes_used, ref_k);
    assert!(!est.fell_back);
    assert!(
        (est.alpha - ref_alpha).abs() <= 1e-9,
        "{} != {}",
        est.alpha,
        ref_alpha
    );
    assert!((est.pseudocount - 1.0 / (4.0 * ref_alpha)).abs() <= 1e-9);
}

/// Only overdispersed, above-mu_min genes count. Here gene0 (α=1/9) and gene3
/// (α=10/9) qualify; gene1 is constant (var=0) and gene2 is all-zero (mean=0),
/// both excluded → `n_genes_used == 2`, median = mean of the two.
#[test]
fn estimate_alpha_counts_and_pools_mixed_genes() {
    let rows = vec![
        vec![1.0f32, 4.0, 0.0, 2.0],
        vec![5.0, 4.0, 0.0, 0.0],
        vec![3.0, 4.0, 0.0, 7.0],
    ];
    let raw = raw_source_from_shards(&[&rows], 4);
    let est = estimate_alpha(&raw, &AlphaOptions::default()).unwrap();
    assert_eq!(est.n_genes_used, 2);
    assert!(!est.fell_back);
    let expected = 0.5 * (1.0 / 9.0 + 10.0 / 9.0);
    assert!((est.alpha - expected).abs() <= 1e-12, "α={}", est.alpha);
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

/// Under-dispersed input (constant columns → var 0 ≤ mean for every gene):
/// no positive dispersion signal anywhere → fallback fires, no error.
#[test]
fn estimate_alpha_falls_back_when_underdispersed() {
    let rows = vec![
        vec![2.0f32, 3.0, 4.0],
        vec![2.0, 3.0, 4.0],
        vec![2.0, 3.0, 4.0],
    ];
    let raw = raw_source_from_shards(&[&rows], 3);
    let est = estimate_alpha(&raw, &AlphaOptions::default()).unwrap();
    assert!(est.fell_back);
    assert_eq!(est.n_genes_used, 0);
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

/// A malformed source whose shard rows exceed the declared `n_obs` must return
/// a clean `ShapeError`, not panic on out-of-bounds `cell_depths` indexing.
#[test]
fn baseline_rejects_shard_rows_exceeding_n_obs() {
    // Source declares n_obs=2 but serves a 3-row shard.
    let rows = vec![
        vec![1.0f32, 1.0, 1.0, 1.0],
        vec![2.0, 0.0, 1.0, 0.0],
        vec![0.0, 3.0, 0.0, 1.0],
    ];
    let bad = MultiShardSource {
        shards: vec![csr_from_dense(&rows)],
        n_obs: 2,
        n_vars: 4,
    };
    let depths = vec![4.0, 3.0]; // length matches declared n_obs
    let err = pflog1ppf_baseline(&bad, &depths, 1.0);
    assert!(matches!(err, Err(AccelError::ShapeError(_))));
}
