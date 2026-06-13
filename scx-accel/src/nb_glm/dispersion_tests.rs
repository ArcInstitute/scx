//! Unit tests for the dispersion estimator, trend fit, and shrinkage (spec §11.1).

use super::*;
use crate::nb_glm::NbGlmOptions;

fn close(a: f64, b: f64, rel: f64, abs: f64) {
    let diff = (a - b).abs();
    let tol = abs.max(rel * a.abs().max(b.abs()));
    assert!(diff <= tol, "expected {a} ≈ {b} (|Δ|={diff} > tol={tol})");
}

// Intercept + treatment, 6 samples.
fn fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>, usize, usize) {
    let design = vec![
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0, //
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
    ];
    // Genuinely over-dispersed within each condition group so the Cox–Reid
    // optimum sits clearly in the interior (not pinned near min_disp).
    let mu = vec![10.0, 10.0, 10.0, 15.0, 15.0, 15.0];
    let y = vec![3.0, 10.0, 20.0, 6.0, 24.0, 12.0];
    (design, mu, y, 6, 2)
}

#[test]
fn cox_reid_mle_gradient_zero_and_matches_grid() {
    let (design, mu, y, ns, nf) = fixture();
    let opts = NbGlmOptions::default();
    let fit = fit_dispersion(&y, &mu, &design, ns, nf, 0.5, None, &opts);
    assert!(!fit.at_low && !fit.at_high, "optimum should be interior");

    // Gradient ≈ 0 at the returned alpha.
    let g = cox_reid_grad(&y, &mu, &design, ns, nf, opts.beta_ridge, fit.alpha, None);
    assert!(g.abs() < 1e-4, "gradient at MLE should be ~0, got {g}");

    // Matches a log-spaced grid-search argmax of the CR objective.
    let n = 400;
    let (lo, hi) = (opts.min_disp.ln(), opts.max_disp.ln());
    let mut best_t = lo;
    let mut best_v = f64::NEG_INFINITY;
    for i in 0..=n {
        let t = lo + (hi - lo) * (i as f64 / n as f64);
        let v = cox_reid_value(&y, &mu, &design, ns, nf, opts.beta_ridge, t.exp(), None);
        if v > best_v {
            best_v = v;
            best_t = t;
        }
    }
    close(fit.alpha.ln(), best_t, 0.0, 0.1); // grid resolution ~ (hi-lo)/n
}

#[test]
fn trend_fit_recovers_known_coefficients() {
    // alpha = a0 + a1/mu_bar with a0=0.1, a1=2.0 over a spread of base means.
    let (a0, a1) = (0.1, 2.0);
    let n = 200;
    let base_mean: Vec<f64> = (0..n).map(|i| 5.0 + (i as f64) * 2.5).collect();
    let alpha_mle: Vec<f64> = base_mean.iter().map(|&m| a0 + a1 / m).collect();
    let valid = vec![true; n];
    let opts = NbGlmOptions::default();
    let trend = fit_dispersion_trend(&base_mean, &alpha_mle, &valid, &opts).expect("trend");
    close(trend.a0, a0, 1e-3, 1e-4);
    close(trend.a1, a1, 1e-3, 1e-4);
}

#[test]
fn shrinkage_pulls_toward_target_and_respects_prior_strength() {
    let (design, mu, y, ns, nf) = fixture();
    let opts = NbGlmOptions::default();
    let mle = fit_dispersion(&y, &mu, &design, ns, nf, 0.5, None, &opts);

    // Target deliberately offset from the MLE.
    let target = (mle.alpha * 0.25).max(opts.min_disp);
    let log_target = target.ln();

    // Strong prior (tiny variance) pulls the estimate toward the target.
    let tight = DispPrior {
        log_alpha_trend: log_target,
        sigma_lr2: 1e-3,
    };
    let shr_tight = fit_dispersion(&y, &mu, &design, ns, nf, mle.alpha, Some(&tight), &opts);
    // Weak prior (huge variance) leaves it ~at the MLE.
    let weak = DispPrior {
        log_alpha_trend: log_target,
        sigma_lr2: 1e6,
    };
    let shr_weak = fit_dispersion(&y, &mu, &design, ns, nf, mle.alpha, Some(&weak), &opts);

    let d_tight = (shr_tight.alpha.ln() - log_target).abs();
    let d_mle = (mle.alpha.ln() - log_target).abs();
    assert!(
        d_tight < d_mle,
        "strong prior should move toward target: d_tight={d_tight} d_mle={d_mle}"
    );
    close(shr_weak.alpha, mle.alpha, 1e-2, 1e-3);
}

#[test]
fn estimate_prior_var_has_floor() {
    // Identical residuals ⇒ MAD = 0 ⇒ falls back to the minimum prior var.
    let log_targets = vec![0.0; 10];
    let alpha_mle = vec![1.0; 10]; // ln(1) - 0 = 0 residual
    let valid = vec![true; 10];
    // m=6, p=2 ⇒ trigamma((m−p)/2)=trigamma(2)>0; with MAD=0 the prior var floors
    // at MIN_PRIOR_VAR rather than going negative.
    let pv = estimate_prior_var(&log_targets, &alpha_mle, &valid, 6, 2);
    assert!(pv >= MIN_PRIOR_VAR - 1e-12);
}

#[test]
fn moments_dispersion_clamped_positive() {
    let opts = NbGlmOptions::default();
    let counts = vec![5.0, 8.0, 3.0, 12.0, 6.0];
    let sf = vec![1.0; 5];
    let a = moments_dispersion(&counts, &sf, &opts);
    assert!(a >= opts.min_disp && a <= opts.max_disp);
}
