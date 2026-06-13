//! Validation, size-factor, and helper-wiring tests for the NB-GLM module
//! (spec §11.1–11.2, §11.5 partial).

use super::*;
use crate::error::AccelError;

// A simple two-condition design with 4 samples: intercept + treatment indicator.
// Row-major [n_samples × n_features].
fn design_2cond() -> (Vec<f64>, usize, usize) {
    let n_samples = 4;
    let n_features = 2;
    let design = vec![
        1.0, 0.0, // sample 0: control
        1.0, 0.0, // sample 1: control
        1.0, 1.0, // sample 2: treated
        1.0, 1.0, // sample 3: treated
    ];
    (design, n_samples, n_features)
}

#[test]
fn validate_accepts_well_formed_inputs() {
    let (design, n_samples, n_features) = design_2cond();
    let n_genes = 3;
    let counts = vec![
        5.0, 6.0, 9.0, 11.0, // gene 0
        0.0, 1.0, 2.0, 3.0, // gene 1
        4.0, 4.0, 4.0, 4.0, // gene 2
    ];
    let sf = vec![1.0, 1.1, 0.9, 1.0];
    assert!(validate_inputs(&counts, n_genes, n_samples, &design, n_features, Some(&sf)).is_ok());
    // size_factors=None is allowed (computed later).
    assert!(validate_inputs(&counts, n_genes, n_samples, &design, n_features, None).is_ok());
}

#[test]
fn validate_rejects_shape_mismatches() {
    let (design, n_samples, n_features) = design_2cond();
    let counts = vec![1.0; 3 * 4]; // valid: 3 genes × 4 samples
                                   // wrong counts length (11 != 3*4)
    let bad_counts = vec![1.0; 11];
    assert!(matches!(
        validate_inputs(&bad_counts, 3, n_samples, &design, n_features, None),
        Err(AccelError::ShapeError(_))
    ));
    // wrong design length
    let bad_design = vec![1.0; 7];
    assert!(matches!(
        validate_inputs(&counts, 3, n_samples, &bad_design, n_features, None),
        Err(AccelError::ShapeError(_))
    ));
    // wrong size_factors length
    let sf = vec![1.0, 1.0];
    assert!(matches!(
        validate_inputs(&counts, 3, n_samples, &design, n_features, Some(&sf)),
        Err(AccelError::ShapeError(_))
    ));
}

#[test]
fn validate_rejects_bad_values() {
    let (design, n_samples, n_features) = design_2cond();
    // negative count
    let mut counts = vec![1.0; 4 * 3];
    counts[5] = -1.0;
    assert!(matches!(
        validate_inputs(&counts, 3, n_samples, &design, n_features, None),
        Err(AccelError::InvalidInput(_))
    ));
    // non-finite count
    counts[5] = f64::NAN;
    assert!(matches!(
        validate_inputs(&counts, 3, n_samples, &design, n_features, None),
        Err(AccelError::InvalidInput(_))
    ));
    // non-positive size factor
    let good = vec![1.0; 4 * 3];
    let sf = vec![1.0, 0.0, 1.0, 1.0];
    assert!(matches!(
        validate_inputs(&good, 3, n_samples, &design, n_features, Some(&sf)),
        Err(AccelError::InvalidInput(_))
    ));
}

#[test]
fn validate_rejects_too_few_samples() {
    // n_samples == n_features → no residual df.
    let design = vec![1.0, 0.0, 1.0, 1.0]; // 2 samples × 2 features
    let counts = vec![1.0, 2.0, 3.0, 4.0]; // 2 genes × 2 samples
    assert!(matches!(
        validate_inputs(&counts, 2, 2, &design, 2, None),
        Err(AccelError::InvalidInput(_))
    ));
}

#[test]
fn validate_rejects_rank_deficient_design() {
    // 4 samples × 3 features where col 2 = col 1 (collinear).
    let n_samples = 4;
    let n_features = 3;
    let design = vec![
        1.0, 0.0, 0.0, //
        1.0, 0.0, 0.0, //
        1.0, 1.0, 1.0, //
        1.0, 1.0, 1.0, //
    ];
    let counts = vec![1.0; n_samples * 2];
    let err = validate_inputs(&counts, 2, n_samples, &design, n_features, None);
    assert!(
        matches!(err, Err(AccelError::InvalidInput(_))),
        "got {err:?}"
    );
}

#[test]
fn validate_rejects_empty_dims() {
    assert!(matches!(
        validate_inputs(&[], 0, 4, &[], 2, None),
        Err(AccelError::InvalidInput(_))
    ));
}

#[test]
fn contrast_validation() {
    // coefficient index in / out of range
    assert!(validate_contrast(&NbGlmContrast::Coefficient { index: 1 }, 2).is_ok());
    assert!(matches!(
        validate_contrast(&NbGlmContrast::Coefficient { index: 5 }, 2),
        Err(AccelError::InvalidInput(_))
    ));
    // vector length match / mismatch / non-finite
    assert!(validate_contrast(
        &NbGlmContrast::Vector {
            weights: vec![0.0, 1.0]
        },
        2
    )
    .is_ok());
    assert!(matches!(
        validate_contrast(&NbGlmContrast::Vector { weights: vec![1.0] }, 2),
        Err(AccelError::InvalidInput(_))
    ));
    assert!(matches!(
        validate_contrast(
            &NbGlmContrast::Vector {
                weights: vec![1.0, f64::INFINITY]
            },
            2
        ),
        Err(AccelError::InvalidInput(_))
    ));
}

#[test]
fn median_ratio_recovers_known_per_sample_scaling() {
    // Build a base profile, then scale each sample's counts by a known factor.
    // Median-ratio should recover those factors up to the geometric-mean-1
    // normalization.
    let n_genes = 50;
    let n_samples = 4;
    let scale = [2.0_f64, 1.0, 0.5, 1.5];
    // Per-gene base expression (deterministic, all strictly positive).
    let mut counts = vec![0.0; n_genes * n_samples];
    for g in 0..n_genes {
        let base = 10.0 + (g as f64);
        for s in 0..n_samples {
            counts[g * n_samples + s] = (base * scale[s]).round().max(1.0);
        }
    }
    let factors = median_ratio_size_factors(&counts, n_genes, n_samples);

    // Geometric mean of the result is 1.
    let log_mean: f64 = factors.iter().map(|f| f.ln()).sum::<f64>() / n_samples as f64;
    assert!((log_mean.exp() - 1.0).abs() < 1e-9, "geo-mean should be 1");

    // Ratios between recovered factors track the ratios of the planted scaling.
    let scale_log_mean: f64 = scale.iter().map(|f| f.ln()).sum::<f64>() / n_samples as f64;
    let scale_geo = scale_log_mean.exp();
    for s in 0..n_samples {
        let expected = scale[s] / scale_geo;
        assert!(
            (factors[s] - expected).abs() < 0.05,
            "sample {s}: factor {} vs expected {expected}",
            factors[s]
        );
    }
}

#[test]
fn median_ratio_falls_back_on_too_few_reference_genes() {
    // Every gene has a zero somewhere → no valid geometric means → library size.
    let n_genes = 3;
    let n_samples = 4;
    let counts = vec![
        0.0, 10.0, 20.0, 30.0, // gene 0 (zero in sample 0)
        5.0, 0.0, 15.0, 25.0, // gene 1 (zero in sample 1)
        2.0, 4.0, 0.0, 8.0, // gene 2 (zero in sample 2)
    ];
    let factors = median_ratio_size_factors(&counts, n_genes, n_samples);
    assert_eq!(factors.len(), n_samples);
    assert!(factors.iter().all(|f| f.is_finite() && *f > 0.0));
    let log_mean: f64 = factors.iter().map(|f| f.ln()).sum::<f64>() / n_samples as f64;
    assert!((log_mean.exp() - 1.0).abs() < 1e-9);
}

#[test]
fn diffexp_helpers_are_wired() {
    // normal_sf is now pub(crate) and reachable from this module; BH too.
    let sf0 = crate::diffexp::normal_sf(0.0);
    assert!((sf0 - 0.5).abs() < 1e-12);
    let sf196 = crate::diffexp::normal_sf(1.96);
    assert!((sf196 - 0.024_997_895_148_220_4).abs() < 1e-9);
    let adj = crate::diffexp::benjamini_hochberg(&[0.01, 0.02, 0.5]);
    assert_eq!(adj.len(), 3);
    assert!(adj.iter().all(|p| (0.0..=1.0).contains(p)));
}

#[test]
fn options_default_has_no_adam_fields_and_sane_values() {
    let o = NbGlmOptions::default();
    assert_eq!(o.dispersion, DispersionMethod::CoxReidShrunk);
    assert!(o.fit_dispersion_trend && o.shrink_dispersion);
    assert_eq!(o.eta_min, -30.0);
    assert_eq!(o.eta_max, 30.0);
    assert!(o.min_disp > 0.0 && o.max_disp > o.min_disp);
}
