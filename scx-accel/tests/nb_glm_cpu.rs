//! CPU NB-GLM integration tests: fixtures, determinism, error handling (spec §11.2).

use scx_accel::{pseudobulk_nb_glm, AccelError, DispersionMethod, NbGlmContrast, NbGlmOptions};

/// Build a `[n_genes × n_samples]` gene-major count buffer from per-gene rows.
fn gene_major(rows: &[Vec<f64>]) -> (Vec<f64>, usize, usize) {
    let n_genes = rows.len();
    let n_samples = rows[0].len();
    let mut buf = Vec::with_capacity(n_genes * n_samples);
    for r in rows {
        assert_eq!(r.len(), n_samples);
        buf.extend_from_slice(r);
    }
    (buf, n_genes, n_samples)
}

/// Intercept + treatment design for `n` control then `n` treated samples.
fn design_two_condition(n_ctrl: usize, n_trt: usize) -> (Vec<f64>, usize, usize) {
    let n_samples = n_ctrl + n_trt;
    let mut d = Vec::with_capacity(n_samples * 2);
    for _ in 0..n_ctrl {
        d.extend_from_slice(&[1.0, 0.0]);
    }
    for _ in 0..n_trt {
        d.extend_from_slice(&[1.0, 1.0]);
    }
    (d, n_samples, 2)
}

#[test]
fn two_condition_known_sign_and_magnitude() {
    // Gene up in treated (~2×), gene down in treated (~0.5×), gene flat.
    let rows = vec![
        vec![10.0, 12.0, 9.0, 20.0, 22.0, 19.0],  // up: log2fc ≈ +1
        vec![20.0, 18.0, 22.0, 10.0, 9.0, 11.0],  // down: log2fc ≈ -1
        vec![15.0, 14.0, 16.0, 15.0, 14.0, 16.0], // flat: log2fc ≈ 0
    ];
    let (counts, n_genes, n_samples) = gene_major(&rows);
    let (design, ns, nf) = design_two_condition(3, 3);
    assert_eq!(ns, n_samples);

    let sf = vec![1.0; n_samples];
    let res = pseudobulk_nb_glm(
        &counts,
        n_genes,
        n_samples,
        &design,
        nf,
        Some(&sf),
        NbGlmContrast::Coefficient { index: 1 },
        NbGlmOptions::default(),
    )
    .expect("fit");

    assert_eq!(res.log2_fold_change.len(), n_genes);
    // Signs.
    assert!(
        res.log2_fold_change[0] > 0.5,
        "gene 0 up: {}",
        res.log2_fold_change[0]
    );
    assert!(
        res.log2_fold_change[1] < -0.5,
        "gene 1 down: {}",
        res.log2_fold_change[1]
    );
    assert!(
        res.log2_fold_change[2].abs() < 0.3,
        "gene 2 flat: {}",
        res.log2_fold_change[2]
    );
    // Approximate magnitude for the up gene (control ~10.3, treated ~20.3 ⇒ ~+1).
    assert!((res.log2_fold_change[0] - 1.0).abs() < 0.4);
    // No NaN/Inf anywhere in this well-conditioned fit.
    for v in res
        .log2_fold_change
        .iter()
        .chain(res.p_value.iter())
        .chain(res.p_adj.iter())
        .chain(res.dispersion.iter())
    {
        assert!(v.is_finite(), "unexpected non-finite output: {v}");
    }
    // p-values in [0,1]; flat gene least significant.
    for p in &res.p_value {
        assert!((0.0..=1.0).contains(p));
    }
    assert!(res.p_value[2] > res.p_value[0]);
    // Coefficients stored by default.
    assert!(res.beta.is_some());
    assert_eq!(res.beta.as_ref().unwrap().len(), n_genes * nf);
}

#[test]
fn intercept_only_fit() {
    // Single-column design, 5 samples; base means recovered, finite output.
    let rows = vec![
        vec![5.0, 7.0, 6.0, 8.0, 4.0],
        vec![100.0, 110.0, 90.0, 95.0, 105.0],
    ];
    let (counts, n_genes, n_samples) = gene_major(&rows);
    let design = vec![1.0; n_samples]; // intercept only
    let sf = vec![1.0; n_samples];
    let res = pseudobulk_nb_glm(
        &counts,
        n_genes,
        n_samples,
        &design,
        1,
        Some(&sf),
        NbGlmContrast::Coefficient { index: 0 },
        NbGlmOptions::default(),
    )
    .expect("fit");
    assert!((res.base_mean[0] - 6.0).abs() < 1e-9);
    assert!((res.base_mean[1] - 100.0).abs() < 1e-9);
    assert!(res.converged.iter().all(|&c| c));
}

#[test]
fn multi_covariate_with_batch() {
    // Design: intercept + treatment + batch. 8 samples (2 batches × 2 conditions × 2 reps).
    // Rows: [intercept, treatment, batch].
    let mut design = Vec::new();
    let meta = [
        (0.0, 0.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (1.0, 1.0),
    ];
    for (trt, batch) in meta {
        design.extend_from_slice(&[1.0, trt, batch]);
    }
    let n_samples = 8;
    let rows = vec![
        // up in treatment, with a batch shift
        vec![10.0, 11.0, 21.0, 19.0, 13.0, 14.0, 26.0, 24.0],
        vec![30.0, 28.0, 31.0, 29.0, 33.0, 34.0, 32.0, 30.0], // flat
    ];
    let (counts, n_genes, _) = gene_major(&rows);
    // Explicit unit size factors: with only 2 genes, median-ratio/library-size
    // normalization would let gene 0's strong effect leak a composition bias into
    // the flat gene (a small-fixture artifact, not present at real gene counts).
    let sf = vec![1.0; n_samples];
    let res = pseudobulk_nb_glm(
        &counts,
        n_genes,
        n_samples,
        &design,
        3,
        Some(&sf),
        NbGlmContrast::Coefficient { index: 1 },
        NbGlmOptions::default(),
    )
    .expect("fit");
    assert!(
        res.log2_fold_change[0] > 0.4,
        "treatment up: {}",
        res.log2_fold_change[0]
    );
    assert!(res.log2_fold_change[1].abs() < 0.3);
    for v in res.log2_fold_change.iter().chain(res.p_value.iter()) {
        assert!(v.is_finite());
    }
}

#[test]
fn all_zero_gene_handled() {
    let rows = vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // all-zero
        vec![10.0, 12.0, 9.0, 20.0, 22.0, 19.0],
    ];
    let (counts, n_genes, n_samples) = gene_major(&rows);
    let (design, _, nf) = design_two_condition(3, 3);
    let sf = vec![1.0; n_samples];
    let res = pseudobulk_nb_glm(
        &counts,
        n_genes,
        n_samples,
        &design,
        nf,
        Some(&sf),
        NbGlmContrast::Coefficient { index: 1 },
        NbGlmOptions::default(),
    )
    .expect("fit");
    assert_eq!(res.base_mean[0], 0.0);
    assert_eq!(res.log2_fold_change[0], 0.0);
    assert_eq!(res.p_value[0], 1.0);
    assert!(res.dispersion[0].is_nan());
    assert!(res.converged[0]);
    assert_eq!(res.n_iter[0], 0);
    assert_eq!(res.diagnostics.n_all_zero_genes, 1);
}

#[test]
fn high_low_count_mix_no_nan() {
    let rows = vec![
        vec![1.0, 0.0, 2.0, 1.0, 3.0, 0.0],                   // low counts
        vec![5000.0, 5200.0, 4800.0, 9000.0, 9500.0, 8800.0], // high counts, up
    ];
    let (counts, n_genes, n_samples) = gene_major(&rows);
    let (design, _, nf) = design_two_condition(3, 3);
    let sf = vec![1.0; n_samples];
    let res = pseudobulk_nb_glm(
        &counts,
        n_genes,
        n_samples,
        &design,
        nf,
        Some(&sf),
        NbGlmContrast::Coefficient { index: 1 },
        NbGlmOptions::default(),
    )
    .expect("fit");
    for v in res.log2_fold_change.iter().chain(res.p_value.iter()) {
        assert!(v.is_finite(), "non-finite: {v}");
    }
    assert!(res.log2_fold_change[1] > 0.4);
}

#[test]
fn rank_deficient_design_errors() {
    // col2 == col1 ⇒ rank-deficient.
    let n_samples = 6;
    let mut design = Vec::new();
    for s in 0..n_samples {
        let t = if s >= 3 { 1.0 } else { 0.0 };
        design.extend_from_slice(&[1.0, t, t]);
    }
    let rows = vec![vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]];
    let (counts, n_genes, _) = gene_major(&rows);
    let err = pseudobulk_nb_glm(
        &counts,
        n_genes,
        n_samples,
        &design,
        3,
        None,
        NbGlmContrast::Coefficient { index: 1 },
        NbGlmOptions::default(),
    );
    assert!(
        matches!(err, Err(AccelError::InvalidInput(_))),
        "got {err:?}"
    );
}

#[test]
fn too_few_samples_errors() {
    // n_samples (2) == n_features (2).
    let design = vec![1.0, 0.0, 1.0, 1.0];
    let counts = vec![5.0, 8.0];
    let err = pseudobulk_nb_glm(
        &counts,
        1,
        2,
        &design,
        2,
        None,
        NbGlmContrast::Coefficient { index: 1 },
        NbGlmOptions::default(),
    );
    assert!(matches!(err, Err(AccelError::InvalidInput(_))));
}

#[test]
fn very_small_and_large_dispersion_genes() {
    let rows = vec![
        vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0], // ~zero within-group var ⇒ tiny disp
        vec![1.0, 30.0, 5.0, 2.0, 50.0, 8.0],     // huge within-group var ⇒ large disp
    ];
    let (counts, n_genes, n_samples) = gene_major(&rows);
    let (design, _, nf) = design_two_condition(3, 3);
    let sf = vec![1.0; n_samples];
    let res = pseudobulk_nb_glm(
        &counts,
        n_genes,
        n_samples,
        &design,
        nf,
        Some(&sf),
        NbGlmContrast::Coefficient { index: 1 },
        NbGlmOptions::default(),
    )
    .expect("fit");
    assert!(res.dispersion.iter().all(|a| a.is_finite() && *a >= 0.0));
    assert!(
        res.dispersion[1] > res.dispersion[0],
        "gene 1 more dispersed"
    );
}

#[test]
fn determinism_and_convergence_at_scale() {
    // 2000 genes × 6 samples: output must be identical across runs (rayon order
    // independence) and well-conditioned genes converge within max_outer_iters.
    let n_genes = 2000;
    let n_samples = 6;
    let (design, _, nf) = design_two_condition(3, 3);
    let sf = vec![1.0; n_samples];
    let mut counts = vec![0.0; n_genes * n_samples];
    for g in 0..n_genes {
        // deterministic pseudo-random counts, treatment-up for even genes
        let base = 5.0 + (g % 50) as f64;
        let up = if g % 2 == 0 { 2.0 } else { 1.0 };
        let pattern = [0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
        for s in 0..n_samples {
            let trt = if s >= 3 { up } else { 1.0 };
            let jitter = 1.0 + 0.1 * pattern[s];
            counts[g * n_samples + s] = (base * trt * jitter).round().max(0.0);
        }
    }
    let run = || {
        pseudobulk_nb_glm(
            &counts,
            n_genes,
            n_samples,
            &design,
            nf,
            Some(&sf),
            NbGlmContrast::Coefficient { index: 1 },
            NbGlmOptions::default(),
        )
        .expect("fit")
    };
    let a = run();
    let b = run();
    assert_eq!(a.log2_fold_change, b.log2_fold_change);
    assert_eq!(a.p_value, b.p_value);
    assert_eq!(a.dispersion, b.dispersion);
    // Convergence: the vast majority of well-conditioned genes converge.
    let n_conv = a.converged.iter().filter(|&&c| c).count();
    assert!(
        n_conv >= n_genes - n_genes / 100,
        "most genes converge: {n_conv}/{n_genes}"
    );
    let opts = NbGlmOptions::default();
    assert!(a
        .n_iter
        .iter()
        .all(|&n| n <= opts.max_outer_iters as u32 + 1));
}

#[test]
#[ignore = "perf sanity (spec §15); run with --release --ignored"]
fn perf_30k_genes_200_samples_5_features() {
    let n_genes = 30_000;
    let n_samples = 200;
    let n_features = 5;
    // Design: intercept + treatment + 3 numeric covariates.
    let mut design = vec![0.0; n_samples * n_features];
    for s in 0..n_samples {
        let base = s * n_features;
        design[base] = 1.0;
        design[base + 1] = if s >= n_samples / 2 { 1.0 } else { 0.0 };
        design[base + 2] = (s % 7) as f64 / 7.0;
        design[base + 3] = (s % 3) as f64 / 3.0;
        design[base + 4] = ((s * 13) % 5) as f64 / 5.0;
    }
    let mut counts = vec![0.0; n_genes * n_samples];
    for g in 0..n_genes {
        let base = 5.0 + (g % 100) as f64;
        for s in 0..n_samples {
            let trt = if s >= n_samples / 2 && g % 2 == 0 {
                1.8
            } else {
                1.0
            };
            counts[g * n_samples + s] = (base * trt * (1.0 + 0.1 * ((g + s) % 5) as f64)).round();
        }
    }
    let t0 = std::time::Instant::now();
    let res = pseudobulk_nb_glm(
        &counts,
        n_genes,
        n_samples,
        &design,
        n_features,
        None,
        NbGlmContrast::Coefficient { index: 1 },
        NbGlmOptions::default(),
    )
    .expect("fit");
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "nb_glm 30k×200×5: {elapsed:.2}s on {} threads",
        res.diagnostics.rayon_threads_used.unwrap_or(0)
    );
    // Generous bound so the test never flakes on a busy node; the spec target is
    // "well under 10 s on 8 threads" — read the printed timing for the real number.
    assert!(elapsed < 60.0, "fit took {elapsed:.2}s");
}

#[test]
fn moments_only_method_runs() {
    let rows = vec![vec![10.0, 12.0, 9.0, 20.0, 22.0, 19.0]];
    let (counts, n_genes, n_samples) = gene_major(&rows);
    let (design, _, nf) = design_two_condition(3, 3);
    let sf = vec![1.0; n_samples];
    let opts = NbGlmOptions {
        dispersion: DispersionMethod::Moments,
        ..Default::default()
    };
    let res = pseudobulk_nb_glm(
        &counts,
        n_genes,
        n_samples,
        &design,
        nf,
        Some(&sf),
        NbGlmContrast::Coefficient { index: 1 },
        opts,
    )
    .expect("fit");
    // Moments method: final dispersion equals the moments MLE (no shrinkage).
    assert_eq!(res.dispersion[0], res.dispersion_mle[0]);
    assert!(res.diagnostics.dispersion_trend.is_none());
    assert!(res.log2_fold_change[0] > 0.5);
}

// ---------------------------------------------------------------------------
// Phase 5 stress tests (spec §16): robustness on adversarial inputs. These
// assert the fitter never panics and never leaks NaN/Inf into fitted output
// (degrading conservatively instead) — not specific numeric values.
// ---------------------------------------------------------------------------

/// A near-rank-deficient (highly collinear, not exactly singular) design must
/// stay stable: either a finite fit (the ridge keeps `XᵀWX` invertible) or a
/// conservative `pvalue=1, stat=0, lfcSE=inf` — never a panic or a NaN log2FC.
#[test]
fn near_rank_deficient_design_is_stable() {
    // 6 samples × 3 features: col 2 ≈ col 1 + tiny perturbation (condition number
    // ~1e6) — collinear but not identical, so QR rank passes but the information
    // matrix is ill-conditioned.
    let n_samples = 6;
    let n_features = 3;
    let mut design = Vec::with_capacity(n_samples * n_features);
    for s in 0..n_samples {
        let t = if s >= 3 { 1.0 } else { 0.0 };
        let eps = (s as f64) * 1e-6; // breaks exact collinearity
        design.extend_from_slice(&[1.0, t, t + eps]);
    }
    let rows = vec![
        vec![10.0, 12.0, 9.0, 20.0, 22.0, 19.0],
        vec![5.0, 6.0, 4.0, 5.0, 7.0, 6.0],
    ];
    let (counts, n_genes, _) = gene_major(&rows);
    let sf = vec![1.0; n_samples];
    let res = pseudobulk_nb_glm(
        &counts,
        n_genes,
        n_samples,
        &design,
        n_features,
        Some(&sf),
        NbGlmContrast::Coefficient { index: 1 },
        NbGlmOptions::default(),
    );
    // Validation may accept (near-singular passes the rank check) — if so, every
    // fitted gene must have a finite log2FC and a p-value in [0, 1].
    if let Ok(res) = res {
        for (g, &lfc) in res.log2_fold_change.iter().enumerate() {
            assert!(lfc.is_finite(), "gene {g} log2FC must be finite, got {lfc}");
            let p = res.p_value[g];
            assert!((0.0..=1.0).contains(&p), "gene {g} p={p} out of range");
        }
    }
    // If it errors instead, that is also acceptable (a typed AccelError) — the
    // only unacceptable outcome is a panic, which would fail the test above.
}

/// The replicate floor: exactly 2 pseudobulk samples per condition (n_samples =
/// n_features + 2) must fit with finite output and the correct effect sign.
#[test]
fn minimum_two_replicates_per_condition() {
    let (design, n_samples, nf) = design_two_condition(2, 2); // 4 samples, 2 features
    let rows = vec![
        vec![10.0, 11.0, 21.0, 19.0], // up in treated
        vec![20.0, 22.0, 9.0, 11.0],  // down in treated
        vec![15.0, 14.0, 15.0, 16.0], // flat
    ];
    let (counts, n_genes, _) = gene_major(&rows);
    let sf = vec![1.0; n_samples];
    let res = pseudobulk_nb_glm(
        &counts,
        n_genes,
        n_samples,
        &design,
        nf,
        Some(&sf),
        NbGlmContrast::Coefficient { index: 1 },
        NbGlmOptions::default(),
    )
    .expect("2 replicates per condition should fit");
    for v in res.log2_fold_change.iter().chain(res.p_value.iter()) {
        assert!(v.is_finite());
    }
    assert!(
        res.log2_fold_change[0] > 0.3,
        "gene 0 up: {}",
        res.log2_fold_change[0]
    );
    assert!(
        res.log2_fold_change[1] < -0.3,
        "gene 1 down: {}",
        res.log2_fold_change[1]
    );
}

/// Heavily imbalanced condition group sizes (8 control vs 2 treated) must fit
/// with finite output, correct sign, and be deterministic across runs.
#[test]
fn imbalanced_group_sizes() {
    let (design, n_samples, nf) = design_two_condition(8, 2);
    let rows = vec![
        // 8 control then 2 treated; gene up ~2x in treated.
        vec![10.0, 11.0, 9.0, 12.0, 10.0, 8.0, 11.0, 9.0, 21.0, 19.0],
        vec![15.0, 14.0, 16.0, 15.0, 14.0, 16.0, 15.0, 15.0, 15.0, 14.0], // flat
    ];
    let (counts, n_genes, _) = gene_major(&rows);
    let sf = vec![1.0; n_samples];
    let run = || {
        pseudobulk_nb_glm(
            &counts,
            n_genes,
            n_samples,
            &design,
            nf,
            Some(&sf),
            NbGlmContrast::Coefficient { index: 1 },
            NbGlmOptions::default(),
        )
        .expect("fit")
    };
    let a = run();
    let b = run();
    assert_eq!(a.log2_fold_change, b.log2_fold_change);
    assert_eq!(a.p_value, b.p_value);
    for v in a.log2_fold_change.iter().chain(a.p_value.iter()) {
        assert!(v.is_finite());
    }
    assert!(
        a.log2_fold_change[0] > 0.4,
        "gene 0 up: {}",
        a.log2_fold_change[0]
    );
    assert!(
        a.log2_fold_change[1].abs() < 0.3,
        "gene 1 flat: {}",
        a.log2_fold_change[1]
    );
}

/// Genes spanning a huge count dynamic range (~1 to ~1e6) in a single fit must
/// produce no NaN/Inf and finite, non-negative dispersions.
#[test]
fn extreme_count_dynamic_range() {
    let (design, n_samples, nf) = design_two_condition(3, 3);
    let rows = vec![
        vec![1.0, 0.0, 2.0, 1.0, 2.0, 1.0],            // ~unit counts
        vec![100.0, 110.0, 90.0, 200.0, 210.0, 195.0], // mid, up
        vec![
            1_000_000.0,
            1_050_000.0,
            980_000.0,
            990_000.0,
            1_010_000.0,
            1_001_000.0,
        ], // huge, flat
    ];
    let (counts, n_genes, _) = gene_major(&rows);
    let sf = vec![1.0; n_samples];
    let res = pseudobulk_nb_glm(
        &counts,
        n_genes,
        n_samples,
        &design,
        nf,
        Some(&sf),
        NbGlmContrast::Coefficient { index: 1 },
        NbGlmOptions::default(),
    )
    .expect("fit");
    for v in res
        .log2_fold_change
        .iter()
        .chain(res.p_value.iter())
        .chain(res.standard_error.iter())
    {
        assert!(!v.is_nan(), "no NaN in fitted output");
    }
    for &a in &res.dispersion {
        assert!(
            a.is_finite() && a >= 0.0,
            "dispersion {a} must be finite & non-negative"
        );
    }
    assert!(
        res.log2_fold_change[1] > 0.4,
        "mid gene up: {}",
        res.log2_fold_change[1]
    );
}
