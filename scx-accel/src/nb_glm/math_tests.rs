//! Unit + finite-difference tests for the NB-GLM math primitives (spec §11.1).

use super::*;

/// Assert `a ≈ b` with a combined relative/absolute tolerance.
fn close(a: f64, b: f64, rel: f64, abs: f64) {
    let diff = (a - b).abs();
    let tol = abs.max(rel * a.abs().max(b.abs()));
    assert!(diff <= tol, "expected {a} ≈ {b} (|Δ|={diff} > tol={tol})");
}

#[test]
fn nb_loglik_matches_hand_computed_nb() {
    // mu=2, alpha=0.5 ⇒ theta=2, scipy nbinom(n=2, p=0.5).logpmf(3) = ln(0.125).
    let ll = nb_loglik(3.0, 2.0, 0.5);
    close(ll, (0.125_f64).ln(), 0.0, 1e-12);
}

#[test]
fn nb_loglik_poisson_limit() {
    // alpha → 0 ⇒ Poisson(mu=2).logpmf(3) = 3·ln2 - 2 - ln(3!).
    let ll = nb_loglik(3.0, 2.0, 1e-15);
    let expect = 3.0 * 2.0_f64.ln() - 2.0 - libm::lgamma(4.0);
    close(ll, expect, 0.0, 1e-12);
}

#[test]
fn nb_loglik_zero_count_is_finite() {
    // y=0 must not produce log(0)·0 = NaN even as mu → min.
    let ll = nb_loglik(0.0, 1e-10, 0.5);
    assert!(ll.is_finite(), "ll should be finite for y=0, got {ll}");
}

#[test]
fn digamma_known_value_and_finite_difference() {
    // ψ(1) = -γ (Euler–Mascheroni).
    close(digamma(1.0), -0.577_215_664_901_532_9, 0.0, 1e-9);
    // ψ(x) = d/dx lgamma(x): central difference at a few points.
    for &x in &[0.4_f64, 1.7, 5.0, 50.0, 1e4] {
        let h = 1e-5 * x;
        let fd = (libm::lgamma(x + h) - libm::lgamma(x - h)) / (2.0 * h);
        close(digamma(x), fd, 1e-5, 1e-7);
    }
}

#[test]
fn irls_weight_is_expected_fisher_information_in_eta() {
    // The NB working weight W = mu/(1+αμ) is the expected Fisher information of
    // the log-likelihood w.r.t. eta (= log μ). At y = μ the *observed*
    // information equals the expected one, so -d²ℓ/dη² (central FD) ≈ W.
    let alpha = 0.7;
    for &eta0 in &[-0.5_f64, 0.3, 1.2] {
        let mu0 = eta0.exp();
        let y = mu0; // y = mu ⇒ observed info == expected info
        let f = |eta: f64| nb_loglik(y, eta.exp(), alpha);
        let h = 1e-3;
        let d2 = (f(eta0 + h) - 2.0 * f(eta0) + f(eta0 - h)) / (h * h);
        let w = nb_irls_weight(mu0, alpha);
        close(-d2, w, 1e-3, 1e-6);
    }
}

#[test]
fn nb_working_response_formula() {
    // z = eta - log s + (y - mu)/mu.
    let (eta, log_s, y, mu) = (0.5, 0.2, 7.0, 4.0);
    close(
        nb_working_response(eta, log_s, y, mu),
        eta - log_s + (y - mu) / mu,
        0.0,
        1e-12,
    );
}

/// A fixed two-sample, slightly over-dispersed fixture whose dispersion gradient
/// is clearly non-zero (so relative tolerances are meaningful).
fn disp_fixture() -> (Vec<f64>, Vec<f64>, f64) {
    let y = vec![5.0, 12.0, 3.0, 20.0];
    let mu = vec![6.0, 10.0, 4.0, 18.0];
    let alpha = 0.4;
    (y, mu, alpha)
}

#[test]
fn nb_loglik_dlogalpha_matches_finite_difference() {
    let (y, mu, alpha) = disp_fixture();
    let analytic = nb_loglik_sum_dlogalpha(&y, &mu, alpha);
    let la = alpha.ln();
    let h = 1e-5;
    let fd = (nb_loglik_sum(&y, &mu, (la + h).exp()) - nb_loglik_sum(&y, &mu, (la - h).exp()))
        / (2.0 * h);
    assert!(analytic.abs() > 1e-3, "gradient should be non-trivial");
    close(analytic, fd, 1e-4, 1e-6);
}

#[test]
fn cox_reid_gradient_matches_finite_difference() {
    let (y, mu, alpha) = disp_fixture();
    // Synthetic smooth log-det term g(α) = c0 + c1·ln(1+α) standing in for the
    // IRLS log det(XᵀWX) the Phase-2 machinery supplies. g'(α) = c1/(1+α), so
    // dg/d(log α) = α·g'(α) = c1·α/(1+α).
    let (c0, c1) = (1.3, 0.9);
    let log_det = |a: f64| c0 + c1 * a.ln_1p();
    let dlogdet_dlogalpha = c1 * alpha / (1.0 + alpha);

    let analytic = cox_reid_dispersion_gradient(&y, &mu, alpha, dlogdet_dlogalpha);
    let la = alpha.ln();
    let h = 1e-5;
    let obj = |a: f64| cox_reid_objective(&y, &mu, a, log_det(a));
    let fd = (obj((la + h).exp()) - obj((la - h).exp())) / (2.0 * h);
    close(analytic, fd, 1e-4, 1e-6);
}

#[test]
fn cox_reid_objective_decomposition() {
    let (y, mu, alpha) = disp_fixture();
    let ld = 2.5;
    close(
        cox_reid_objective(&y, &mu, alpha, ld),
        nb_loglik_sum(&y, &mu, alpha) - 0.5 * ld,
        0.0,
        1e-12,
    );
}

#[test]
fn trigamma_known_value_and_finite_difference() {
    // ψ′(1) = π²/6.
    close(
        trigamma(1.0),
        std::f64::consts::PI * std::f64::consts::PI / 6.0,
        0.0,
        1e-9,
    );
    // ψ′(x) = d/dx ψ(x): central difference at a few points.
    for &x in &[0.4_f64, 1.7, 5.0, 50.0, 1e3] {
        let h = 1e-5 * x;
        let fd = (digamma(x + h) - digamma(x - h)) / (2.0 * h);
        close(trigamma(x), fd, 1e-5, 1e-9);
    }
}

#[test]
fn incomplete_beta_known_values() {
    // Symmetric: I_{0.5}(a, a) = 0.5.
    close(incomplete_beta(2.0, 2.0, 0.5), 0.5, 0.0, 1e-12);
    close(incomplete_beta(5.0, 5.0, 0.5), 0.5, 0.0, 1e-12);
    // I_x(1, 1) = x (uniform CDF).
    close(incomplete_beta(1.0, 1.0, 0.3), 0.3, 0.0, 1e-12);
    // Boundaries.
    assert_eq!(incomplete_beta(2.0, 3.0, 0.0), 0.0);
    assert_eq!(incomplete_beta(2.0, 3.0, 1.0), 1.0);
}

#[test]
fn f_quantile_matches_r_qf() {
    // Reference values from R `qf(0.99, d1, d2)`.
    let cases = [
        (1.0_f64, 10.0_f64, 10.044_29_f64),
        (2.0, 10.0, 7.559_43),
        (2.0, 4.0, 18.000_00),
        (3.0, 5.0, 12.059_92),
        (5.0, 30.0, 3.699_64),
    ];
    for &(d1, d2, q) in &cases {
        let got = f_quantile(0.99, d1, d2);
        close(got, q, 1e-3, 1e-3);
        // Round-trip: the CDF at the quantile recovers p.
        close(f_cdf(got, d1, d2), 0.99, 0.0, 1e-6);
    }
}

#[test]
fn lowess_recovers_linear_exactly() {
    // Local-linear LOWESS returns the line exactly on linear data, any span.
    let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
    let yhat = lowess(&x, &y, 0.3, 3);
    for i in 0..x.len() {
        close(yhat[i], y[i], 0.0, 1e-9);
    }
}

#[test]
fn lowess_downweights_outlier() {
    // A mildly-noisy linear trend with one large spike. The noise keeps the
    // median absolute residual positive (so robustness engages — exactly like R,
    // which disables robustness when >half the points fit perfectly), and the
    // robustness iterations must then pull the smoothed value at the spike
    // substantially back toward the underlying line vs the non-robust (iter=0) fit.
    let x: Vec<f64> = (0..21).map(|i| i as f64).collect();
    let mut y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| xi + if i % 2 == 0 { 0.3 } else { -0.3 })
        .collect();
    y[10] = 100.0; // outlier at x=10 (true value ~10)
    let yhat0 = lowess(&x, &y, 0.3, 0); // no robustness
    let yhat3 = lowess(&x, &y, 0.3, 3); // 3 robustness iterations
    assert!(
        yhat3[10] < yhat0[10] - 5.0,
        "robustness should downweight the outlier: iter0={} iter3={}",
        yhat0[10],
        yhat3[10]
    );
    assert!(
        yhat3[10] < 30.0,
        "smoothed spike should be well below the raw value 100, got {}",
        yhat3[10]
    );
    // Endpoints (no outlier nearby) stay close to the line.
    close(yhat3[0], 0.0, 0.0, 2.0);
    close(yhat3[20], 20.0, 0.0, 2.0);
}

#[test]
fn lowess_flat_input_stays_flat() {
    // A constant numRej-like curve must smooth to (near-)constant so the
    // independent-filtering 1-SE rule keeps the smallest cutoff (no-op).
    let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let y = vec![42.0_f64; 50];
    let yhat = lowess(&x, &y, 0.2, 3);
    for &v in &yhat {
        close(v, 42.0, 0.0, 1e-9);
    }
}

#[test]
fn clamp_helpers() {
    assert_eq!(clamp_eta(100.0, -30.0, 30.0), 30.0);
    assert_eq!(clamp_eta(-100.0, -30.0, 30.0), -30.0);
    assert_eq!(clamp_eta(1.0, -30.0, 30.0), 1.0);
    assert_eq!(floor_mu(1e-20, 1e-10), 1e-10);
    assert_eq!(floor_mu(5.0, 1e-10), 5.0);
}
