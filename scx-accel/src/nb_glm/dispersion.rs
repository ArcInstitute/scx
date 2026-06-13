//! NB dispersion estimation: Cox–Reid adjusted MLE, parametric trend fit, and
//! empirical-Bayes shrinkage (spec §7.3, §7.5–7.6).
//!
//! The per-gene MLE maximizes the Cox–Reid adjusted profile log-likelihood (which
//! removes the small-sample bias plain MLE has when `n_samples ≈ n_features`). The
//! cross-gene trend fit + log-normal shrinkage is what stabilizes low-replicate
//! genes. The acceptance bar is DESeq2 ranking/sign parity, not exact numerics
//! (spec §2), so the trend GLM and prior-variance estimator are faithful-but-
//! approximate versions of the DESeq2 internals.

use faer::linalg::solvers::DenseSolveCore;
use faer::{MatRef, Side};

use super::irls::{assemble_xtwx, solve_spd};
use super::math::{cox_reid_dispersion_gradient, cox_reid_objective, nb_irls_weight};
use super::types::{DispersionTrend, NbGlmOptions};

/// Minimum empirical-Bayes prior variance on `log(alpha)` (DESeq2 uses 0.25).
const MIN_PRIOR_VAR: f64 = 0.25;

/// Log-normal shrinkage prior for one gene's dispersion (spec §7.6).
#[derive(Debug, Clone, Copy)]
pub(crate) struct DispPrior {
    /// Prior mean of `log(alpha)` — the trend value (or global median) for the gene.
    pub log_alpha_trend: f64,
    /// Prior variance of `log(alpha)`.
    pub sigma_lr2: f64,
}

/// Result of a single-gene dispersion fit.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DispFit {
    pub alpha: f64,
    pub at_low: bool,
    pub at_high: bool,
}

/// Method-of-moments dispersion init from normalized counts (spec §7.3), clamped
/// to `[min_disp, max_disp]`.
pub(crate) fn moments_dispersion(
    counts_row: &[f64],
    size_factors: &[f64],
    opts: &NbGlmOptions,
) -> f64 {
    let n = counts_row.len() as f64;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for (s, &c) in counts_row.iter().enumerate() {
        let yn = c / size_factors[s];
        sum += yn;
        sum_sq += yn * yn;
    }
    let mean = sum / n;
    if mean <= 0.0 {
        return opts.min_disp;
    }
    let var = (sum_sq / n - mean * mean).max(0.0);
    let alpha0 = (var - mean) / (mean * mean);
    alpha0.clamp(opts.min_disp, opts.max_disp)
}

/// `log det(M)` and `M⁻¹` (row-major `n²`) for `M = XᵀW X + ridge·I`. Cholesky
/// first, partial-pivot LU fallback. `None` if the matrices are non-finite.
fn m_stats(
    design: &[f64],
    w: &[f64],
    n_samples: usize,
    n_features: usize,
    ridge: f64,
) -> Option<(f64, Vec<f64>)> {
    let xtwx = assemble_xtwx(design, w, n_samples, n_features, ridge);
    let a = MatRef::from_row_major_slice(&xtwx, n_features, n_features);
    let (log_det, inv) = match a.llt(Side::Lower) {
        Ok(llt) => {
            let l = llt.L();
            let mut ld = 0.0;
            for i in 0..n_features {
                ld += l[(i, i)].abs().ln();
            }
            (2.0 * ld, llt.inverse())
        }
        Err(_) => {
            let lu = a.partial_piv_lu();
            let u = lu.U();
            let mut ld = 0.0;
            for i in 0..n_features {
                ld += u[(i, i)].abs().ln();
            }
            (ld, lu.inverse())
        }
    };
    if !log_det.is_finite() {
        return None;
    }
    let mut m_inv = vec![0.0_f64; n_features * n_features];
    for r in 0..n_features {
        for c in 0..n_features {
            m_inv[r * n_features + c] = inv[(r, c)];
        }
    }
    Some((log_det, m_inv))
}

/// NB working weights `W = mu/(1+alpha*mu)` for all samples (dispersion holds `mu`
/// fixed and varies `alpha`).
fn weights(mu: &[f64], alpha: f64) -> Vec<f64> {
    mu.iter().map(|&m| nb_irls_weight(m, alpha)).collect()
}

/// Cox–Reid adjusted profile log-likelihood at `alpha` (with optional MAP prior),
/// holding `mu` fixed (spec §7.5–7.6). The optimizer ascends [`cox_reid_grad`]
/// directly; this value form is the objective the dispersion tests grid-search.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn cox_reid_value(
    y: &[f64],
    mu: &[f64],
    design: &[f64],
    n_samples: usize,
    n_features: usize,
    ridge: f64,
    alpha: f64,
    prior: Option<&DispPrior>,
) -> f64 {
    let w = weights(mu, alpha);
    let log_det = match m_stats(design, &w, n_samples, n_features, ridge) {
        Some((ld, _)) => ld,
        None => return f64::NEG_INFINITY,
    };
    let mut v = cox_reid_objective(y, mu, alpha, log_det);
    if let Some(p) = prior {
        let d = alpha.ln() - p.log_alpha_trend;
        v -= d * d / (2.0 * p.sigma_lr2);
    }
    v
}

/// Derivative of [`cox_reid_value`] with respect to `t = log(alpha)`.
///
/// `dCR/dt = ∂(Σ logNB)/∂t − ½·α·Σ_s W_s²·h_s` where `h_s = x_sᵀ M⁻¹ x_s`
/// (since `dW/dα = −W²` and `d log det M/dα = tr(M⁻¹ dM/dα)`; ridge is constant).
#[allow(clippy::too_many_arguments)]
pub(crate) fn cox_reid_grad(
    y: &[f64],
    mu: &[f64],
    design: &[f64],
    n_samples: usize,
    n_features: usize,
    ridge: f64,
    alpha: f64,
    prior: Option<&DispPrior>,
) -> f64 {
    let w = weights(mu, alpha);
    let m_inv = match m_stats(design, &w, n_samples, n_features, ridge) {
        Some((_, inv)) => inv,
        None => return 0.0,
    };
    // dlogdet/dalpha = Σ_s (−W_s²) · h_s ; dlogdet/dt = α · that.
    let mut dlogdet_dalpha = 0.0;
    for s in 0..n_samples {
        let row = &design[s * n_features..(s + 1) * n_features];
        let mut h = 0.0;
        for j in 0..n_features {
            let mut mij_xj = 0.0;
            for k in 0..n_features {
                mij_xj += m_inv[j * n_features + k] * row[k];
            }
            h += row[j] * mij_xj;
        }
        dlogdet_dalpha += -(w[s] * w[s]) * h;
    }
    let dlogdet_dt = alpha * dlogdet_dalpha;
    let mut g = cox_reid_dispersion_gradient(y, mu, alpha, dlogdet_dt);
    if let Some(p) = prior {
        g -= (alpha.ln() - p.log_alpha_trend) / p.sigma_lr2;
    }
    g
}

/// Maximize the Cox–Reid (optionally MAP-penalized) objective over `alpha` by a
/// safeguarded Newton step on `g(t)=0`, `t=log(alpha)`, with bisection fallback
/// (spec §7.5). Bracketed to `[log min_disp, log max_disp]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_dispersion(
    y: &[f64],
    mu: &[f64],
    design: &[f64],
    n_samples: usize,
    n_features: usize,
    alpha_init: f64,
    prior: Option<&DispPrior>,
    opts: &NbGlmOptions,
) -> DispFit {
    let ridge = opts.beta_ridge;
    let t_lo = opts.min_disp.ln();
    let t_hi = opts.max_disp.ln();
    let g = |t: f64| cox_reid_grad(y, mu, design, n_samples, n_features, ridge, t.exp(), prior);

    // Bracket the root `g(t)=0` (g decreasing through the CR optimum) by geometric
    // expansion in `t = log(alpha)` from the seed `alpha_init`. This keeps the
    // bracket tight even though `[min_disp, max_disp]` spans ~23 in log-space —
    // regula-falsi over the full span is too stiff to converge in a few evals.
    let t0 = alpha_init.clamp(opts.min_disp, opts.max_disp).ln();
    let g0 = g(t0);
    if !g0.is_finite() {
        return DispFit {
            alpha: alpha_init.clamp(opts.min_disp, opts.max_disp),
            at_low: false,
            at_high: false,
        };
    }
    let (mut a, mut ga, mut b, mut gb);
    if g0 >= 0.0 {
        // Optimum at larger alpha: expand the upper end until g turns negative.
        a = t0;
        ga = g0;
        let mut step = 0.5;
        let mut tb = (t0 + step).min(t_hi);
        loop {
            let gtb = g(tb);
            if !gtb.is_finite() || gtb <= 0.0 {
                b = tb;
                gb = gtb;
                break;
            }
            if tb >= t_hi {
                return DispFit {
                    alpha: opts.max_disp,
                    at_low: false,
                    at_high: true,
                };
            }
            a = tb;
            ga = gtb;
            step *= 2.0;
            tb = (tb + step).min(t_hi);
        }
    } else {
        // Optimum at smaller alpha: expand the lower end until g turns positive.
        b = t0;
        gb = g0;
        let mut step = 0.5;
        let mut ta = (t0 - step).max(t_lo);
        loop {
            let gta = g(ta);
            if !gta.is_finite() || gta >= 0.0 {
                a = ta;
                ga = gta;
                break;
            }
            if ta <= t_lo {
                return DispFit {
                    alpha: opts.min_disp,
                    at_low: true,
                    at_high: false,
                };
            }
            b = ta;
            gb = gta;
            step *= 2.0;
            ta = (ta - step).max(t_lo);
        }
    }

    // Illinois (modified regula-falsi) on the sign-bracket {(a,ga), (b,gb)} with
    // ga·gb ≤ 0: one gradient eval per iteration, guaranteed bracketing,
    // superlinear convergence — the safeguarded 1-D root finder the spec asks for
    // (§7.5), without the extra gradient evals a numeric-derivative Newton costs.
    // Canonical form: `b` is always the most recent point; halve the retained
    // endpoint's value when it stagnates.
    let mut t = b;
    for _ in 0..opts.disp_newton_iters {
        let mut c = (a * gb - b * ga) / (gb - ga);
        if !c.is_finite() || c <= a.min(b) || c >= a.max(b) {
            c = 0.5 * (a + b); // bisection safeguard
        }
        let gc = g(c);
        t = c;
        if gc.abs() < 1e-8 || (a - b).abs() < 1e-10 {
            break;
        }
        if gc * gb < 0.0 {
            // root in (b, c): discard a, shift the bracket.
            a = b;
            ga = gb;
        } else {
            // same sign as gb ⇒ a retained; halve its value to un-stick it.
            ga *= 0.5;
        }
        b = c;
        gb = gc;
    }
    let alpha = t.exp().clamp(opts.min_disp, opts.max_disp);
    DispFit {
        alpha,
        at_low: alpha <= opts.min_disp * (1.0 + 1e-9),
        at_high: alpha >= opts.max_disp * (1.0 - 1e-9),
    }
}

/// Fit the parametric mean→dispersion trend `alpha_trend(mu_bar) = a0 + a1/mu_bar`
/// via a gamma-family GLM (identity link, weight `1/fit²`) over valid genes
/// (spec §7.6). Returns `None` if too few usable genes.
pub(crate) fn fit_dispersion_trend(
    base_mean: &[f64],
    alpha_mle: &[f64],
    valid: &[bool],
    opts: &NbGlmOptions,
) -> Option<DispersionTrend> {
    // Usable genes: valid fit, positive base mean and dispersion.
    let idx: Vec<usize> = (0..alpha_mle.len())
        .filter(|&g| {
            valid[g] && base_mean[g] > 0.0 && alpha_mle[g].is_finite() && alpha_mle[g] > 0.0
        })
        .collect();
    if idx.len() < 3 {
        return None;
    }

    // Design rows [1, 1/mu_bar]; response alpha_mle. Identity-link gamma IRLS:
    // working response is just the response, weights 1/fit². Iterate a few times.
    let inv_mu: Vec<f64> = idx.iter().map(|&g| 1.0 / base_mean[g]).collect();
    let resp: Vec<f64> = idx.iter().map(|&g| alpha_mle[g]).collect();

    // Init: a0 = median dispersion, a1 = 0.
    let mut sorted = resp.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let mut a0 = sorted[sorted.len() / 2].max(opts.min_disp);
    let mut a1 = 0.0_f64;

    for _ in 0..10 {
        // Weighted 2×2 normal equations for [a0, a1] with weight 1/fit².
        let mut xtwx = [0.0_f64; 4]; // row-major 2×2
        let mut xtwz = [0.0_f64; 2];
        for i in 0..idx.len() {
            let x0 = 1.0;
            let x1 = inv_mu[i];
            let fit = (a0 + a1 * x1).max(opts.min_disp);
            let wgt = 1.0 / (fit * fit);
            let z = resp[i]; // identity link ⇒ working response == response
            xtwx[0] += wgt * x0 * x0;
            xtwx[1] += wgt * x0 * x1;
            xtwx[3] += wgt * x1 * x1;
            xtwz[0] += wgt * x0 * z;
            xtwz[1] += wgt * x1 * z;
        }
        xtwx[2] = xtwx[1];
        match solve_spd(&xtwx, &xtwz, 2) {
            Some(sol) => {
                let (na0, na1) = (sol[0], sol[1]);
                if !na0.is_finite() || !na1.is_finite() {
                    break;
                }
                let converged = (na0 - a0).abs() <= 1e-8 * (a0.abs() + 1e-8)
                    && (na1 - a1).abs() <= 1e-8 * (a1.abs() + 1e-8);
                a0 = na0;
                a1 = na1;
                if converged {
                    break;
                }
            }
            None => break,
        }
    }

    if !a0.is_finite() || !a1.is_finite() {
        return None;
    }
    Some(DispersionTrend { a0, a1 })
}

/// Robust empirical-Bayes prior variance of `log(alpha)` about a per-gene target
/// (trend value or global median), via the MAD of log-residuals (spec §7.6).
pub(crate) fn estimate_prior_var(log_targets: &[f64], alpha_mle: &[f64], valid: &[bool]) -> f64 {
    let mut resid: Vec<f64> = (0..alpha_mle.len())
        .filter(|&g| valid[g] && alpha_mle[g].is_finite() && alpha_mle[g] > 0.0)
        .map(|g| alpha_mle[g].ln() - log_targets[g])
        .collect();
    if resid.len() < 3 {
        return MIN_PRIOR_VAR;
    }
    resid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = resid[resid.len() / 2];
    let mut abs_dev: Vec<f64> = resid.iter().map(|r| (r - med).abs()).collect();
    abs_dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = abs_dev[abs_dev.len() / 2];
    let sigma = 1.4826 * mad;
    (sigma * sigma).max(MIN_PRIOR_VAR)
}

#[cfg(test)]
#[path = "dispersion_tests.rs"]
mod tests;
