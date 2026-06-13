//! Unit tests for the per-gene IRLS mean fit (spec §11.1).

use super::*;
use crate::nb_glm::math::nb_loglik_sum;
use crate::nb_glm::NbGlmOptions;

fn close(a: f64, b: f64, rel: f64, abs: f64) {
    let diff = (a - b).abs();
    let tol = abs.max(rel * a.abs().max(b.abs()));
    assert!(diff <= tol, "expected {a} ≈ {b} (|Δ|={diff} > tol={tol})");
}

// Intercept + treatment design, 6 samples (3 control, 3 treated), row-major.
fn design_2cond() -> (Vec<f64>, usize, usize) {
    let design = vec![
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0, // controls
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // treated
    ];
    (design, 6, 2)
}

#[test]
fn poisson_limit_recovers_coefficients() {
    // beta_true = [ln 10, ln 2] ⇒ control mu=10, treated mu=20. y = mu exactly
    // sits on the model, so IRLS must recover beta_true (alpha → 0, offset 0).
    let (design, ns, nf) = design_2cond();
    let beta_true = [10.0_f64.ln(), 2.0_f64.ln()];
    let y = vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0];
    let log_sf = vec![0.0; ns];
    let beta0 = vec![0.0, 0.0];
    let opts = NbGlmOptions::default();
    let fit = fit_gene_irls(&y, &design, &log_sf, ns, nf, 1e-10, &beta0, &opts);
    assert!(fit.converged, "should converge");
    close(fit.beta[0], beta_true[0], 1e-4, 1e-5);
    close(fit.beta[1], beta_true[1], 1e-4, 1e-5);
}

#[test]
fn intercept_only_recovers_log_mean() {
    // 1-feature (intercept) design ⇒ MLE mu = mean(y), beta0 = ln(mean).
    let ns = 5;
    let nf = 1;
    let design = vec![1.0; ns];
    let y = vec![3.0, 7.0, 5.0, 11.0, 9.0];
    let log_sf = vec![0.0; ns];
    let mean = y.iter().sum::<f64>() / ns as f64;
    let opts = NbGlmOptions::default();
    let fit = fit_gene_irls(&y, &design, &log_sf, ns, nf, 0.3, &[0.0], &opts);
    assert!(fit.converged);
    close(fit.beta[0], mean.ln(), 1e-5, 1e-6);
    for &m in &fit.mu {
        close(m, mean, 1e-5, 1e-6);
    }
}

#[test]
fn fisher_matches_finite_difference_hessian() {
    // At y = mu the observed information equals the expected info XᵀWX (= fisher),
    // so the FD Hessian of the NLL in beta matches `fisher` (ridge is negligible).
    let (design, ns, nf) = design_2cond();
    let y = vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0];
    let log_sf = vec![0.0; ns];
    let alpha = 0.3;
    let opts = NbGlmOptions::default();
    let fit = fit_gene_irls(&y, &design, &log_sf, ns, nf, alpha, &[0.0, 0.0], &opts);
    let beta = fit.beta.clone();

    // NLL as a function of beta (smooth: no clamp/floor near the optimum).
    let nll = |b: &[f64]| {
        let mu: Vec<f64> = (0..ns)
            .map(|s| {
                let mut eta = log_sf[s];
                for k in 0..nf {
                    eta += design[s * nf + k] * b[k];
                }
                eta.exp()
            })
            .collect();
        -nb_loglik_sum(&y, &mu, alpha)
    };
    let h = 1e-4;
    let perturb = |j: usize, dj: f64, k: usize, dk: f64| {
        let mut b = beta.clone();
        b[j] += dj;
        b[k] += dk;
        nll(&b)
    };
    for j in 0..nf {
        for k in 0..nf {
            let hjk = if j == k {
                (perturb(j, h, j, 0.0) - 2.0 * nll(&beta) + perturb(j, -h, j, 0.0)) / (h * h)
            } else {
                (perturb(j, h, k, h) - perturb(j, h, k, -h) - perturb(j, -h, k, h)
                    + perturb(j, -h, k, -h))
                    / (4.0 * h * h)
            };
            close(fit.fisher[j * nf + k], hjk, 1e-2, 1e-3);
        }
    }
}

#[test]
fn solve_spd_solves_known_system() {
    // [[4,1],[1,3]] x = [1,2] ⇒ x = [1/11, 7/11].
    let a = vec![4.0, 1.0, 1.0, 3.0];
    let b = vec![1.0, 2.0];
    let x = solve_spd(&a, &b, 2).unwrap();
    close(x[0], 1.0 / 11.0, 1e-12, 1e-12);
    close(x[1], 7.0 / 11.0, 1e-12, 1e-12);
}
