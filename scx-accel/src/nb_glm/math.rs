//! Pure numerical primitives for the NB-GLM fit.
//!
//! These are scalar / per-gene-vector functions with no rayon and no fitting
//! loop — the Phase-2 IRLS (`irls.rs`) and dispersion (`dispersion.rs`) code
//! consumes them. Keeping them pure makes the log-likelihood, the analytic IRLS
//! working weights, and the Cox–Reid objective independently finite-difference
//! testable (spec §11.1).
//!
//! Conventions: `y` is an observed count, `mu` the fitted mean, `alpha` the NB
//! dispersion (`theta = 1/alpha`), all `f64`. NB2 parameterization per spec §3.1.

/// `theta` below which we treat the NB as its Poisson limit (`alpha → 0`).
const POISSON_ALPHA_EPS: f64 = 1e-12;

/// Clamp the linear predictor `eta` into `[eta_min, eta_max]` (spec §7.9).
#[inline]
pub fn clamp_eta(eta: f64, eta_min: f64, eta_max: f64) -> f64 {
    eta.clamp(eta_min, eta_max)
}

/// Floor a fitted mean at `min_mu` so NB working weights stay finite (spec §7.9).
#[inline]
pub fn floor_mu(mu: f64, min_mu: f64) -> f64 {
    mu.max(min_mu)
}

/// NB2 log-likelihood of a single observation (spec §3.1), in the numerically
/// stable `log1p(alpha*mu)` form (spec §7.9). Falls back to the Poisson limit as
/// `alpha → 0`.
pub fn nb_loglik(y: f64, mu: f64, alpha: f64) -> f64 {
    debug_assert!(mu > 0.0, "mu must be positive (floor it before calling)");
    if alpha <= POISSON_ALPHA_EPS {
        // Poisson limit: y*log(mu) - mu - lgamma(y+1).
        let mut ll = -mu - libm::lgamma(y + 1.0);
        if y > 0.0 {
            ll += y * mu.ln();
        }
        return ll;
    }
    let theta = 1.0 / alpha;
    let log1p_am = (alpha * mu).ln_1p(); // log(1 + alpha*mu)
                                         // theta*(log theta - log(theta+mu)) = -theta * log1p(alpha*mu).
    let mut ll =
        libm::lgamma(y + theta) - libm::lgamma(theta) - libm::lgamma(y + 1.0) - theta * log1p_am;
    if y > 0.0 {
        // y*(log mu - log(theta+mu)) = y*(log(alpha*mu) - log1p(alpha*mu)).
        ll += y * ((alpha * mu).ln() - log1p_am);
    }
    ll
}

/// Sum of [`nb_loglik`] over all samples of one gene.
pub fn nb_loglik_sum(y: &[f64], mu: &[f64], alpha: f64) -> f64 {
    debug_assert_eq!(y.len(), mu.len());
    y.iter()
        .zip(mu.iter())
        .map(|(&yi, &mui)| nb_loglik(yi, mui, alpha))
        .sum()
}

/// NB IRLS working weight `W = mu / (1 + alpha*mu)` (spec §7.4). This is the
/// diagonal of the working-weight matrix used to form `XᵀWX`.
#[inline]
pub fn nb_irls_weight(mu: f64, alpha: f64) -> f64 {
    mu / (1.0 + alpha * mu)
}

/// IRLS working response `z = eta - log(s) + (y - mu)/mu` (spec §7.4), where
/// `log_s = log(size_factor)` is the offset.
#[inline]
pub fn nb_working_response(eta: f64, log_s: f64, y: f64, mu: f64) -> f64 {
    eta - log_s + (y - mu) / mu
}

/// Digamma (ψ) function, `d/dx lgamma(x)`, for `x > 0`. Recurrence up to `x ≥ 10`
/// then the standard asymptotic series; accurate (~1e-11) across the full `theta`
/// range (`alpha ∈ [1e-8, 100]` ⇒ `theta ∈ [0.01, 1e8]`). Not in `libm`.
pub fn digamma(mut x: f64) -> f64 {
    debug_assert!(x > 0.0, "digamma defined here for x > 0");
    let mut result = 0.0;
    while x < 10.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln() - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
}

/// Trigamma (ψ′) function, `d²/dx² lgamma(x)` = `dψ/dx`, for `x > 0`. Recurrence
/// `ψ′(x) = ψ′(x+1) + 1/x²` up to `x ≥ 6`, then the standard asymptotic series.
/// Mirrors [`digamma`]; needed for the DESeq2 dispersion-prior variance term
/// `trigamma((m−p)/2)` (the expected sampling variance of a log-dispersion MLE).
pub fn trigamma(mut x: f64) -> f64 {
    debug_assert!(x > 0.0, "trigamma defined here for x > 0");
    let mut result = 0.0;
    while x < 10.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    // ψ′(x) ~ 1/x + 1/(2x²) + 1/(6x³) − 1/(30x⁵) + 1/(42x⁷) − …
    result + inv + 0.5 * inv2 + inv * inv2 * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 / 42.0))
}

/// Continued-fraction tail for the regularized incomplete beta (Lentz's method;
/// Numerical Recipes §6.4). Converges for `x < (a+1)/(a+b+2)`.
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 300;
    const EPS: f64 = 1e-14;
    const FPMIN: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let m = m as f64;
        let m2 = 2.0 * m;
        // even step
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        // odd step
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularized incomplete beta `I_x(a, b) = B(x; a, b) / B(a, b)`, for
/// `0 ≤ x ≤ 1`, `a, b > 0`. This is the CDF building block for the F distribution.
pub fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Prefactor x^a (1-x)^b / (a·B(a,b)) in log space for stability.
    let ln_beta = libm::lgamma(a + b) - libm::lgamma(a) - libm::lgamma(b);
    let bt = (ln_beta + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// CDF of the F distribution with `(d1, d2)` degrees of freedom at `f ≥ 0`:
/// `P(F ≤ f) = I_{x}(d1/2, d2/2)` with `x = d1·f / (d1·f + d2)`.
pub fn f_cdf(f: f64, d1: f64, d2: f64) -> f64 {
    if f <= 0.0 {
        return 0.0;
    }
    let x = d1 * f / (d1 * f + d2);
    incomplete_beta(d1 / 2.0, d2 / 2.0, x)
}

/// Quantile (inverse CDF) of the F distribution: returns `f` with
/// `f_cdf(f, d1, d2) = p`, by bracketing then bisection on the monotone CDF.
/// Used for the DESeq2 Cook's-distance cutoff `qf(0.99, p, m−p)`.
pub fn f_quantile(p: f64, d1: f64, d2: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Bracket: expand the upper bound until the CDF clears p.
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    while f_cdf(hi, d1, d2) < p {
        hi *= 2.0;
        if hi > 1e12 {
            return hi;
        }
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f_cdf(mid, d1, d2) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-10 * (hi + 1.0) {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// Locally-weighted scatterplot smoothing (LOWESS), local-linear with tricube
/// neighborhood weights and bisquare robustness iterations (Cleveland 1979).
///
/// `x` must be sorted ascending. `f` is the smoother span (fraction of points in
/// each local fit, clamped to `[2, n]` points); `iter` is the number of
/// robustness iterations (DESeq2's independent filtering uses `f = 1/5`,
/// `iter = 3`). Returns the smoothed value at each `x[i]`. On exactly-linear data
/// it reproduces the line regardless of the weights.
pub fn lowess(x: &[f64], y: &[f64], f: f64, iter: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![y[0]];
    }
    let r = ((f * n as f64).ceil() as usize).clamp(2, n);
    let mut yhat = vec![0.0_f64; n];
    let mut rw = vec![1.0_f64; n]; // robustness weights

    // One local weighted-linear fit at x[i]. `use_rw` toggles the bisquare
    // robustness weights; the tricube neighborhood weights always apply.
    let fit_at = |i: usize, rw: &[f64], use_rw: bool| -> f64 {
        // Contiguous window of `r` points minimizing the max distance to x[i]
        // (valid because x is sorted): slide right while the next point is
        // closer to x[i] than the current leftmost.
        let mut lo = 0usize;
        let mut hi = r - 1;
        while hi < n - 1 && (x[i] - x[lo]) > (x[hi + 1] - x[i]) {
            lo += 1;
            hi += 1;
        }
        let h = (x[i] - x[lo]).max(x[hi] - x[i]); // bandwidth
        let (mut sw, mut swx, mut swy, mut swxx, mut swxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for j in lo..=hi {
            let dist = (x[j] - x[i]).abs();
            let tw = if h > 0.0 {
                let u = dist / h;
                if u >= 1.0 {
                    0.0
                } else {
                    let t = 1.0 - u * u * u;
                    t * t * t
                }
            } else {
                1.0
            };
            let w = if use_rw { tw * rw[j] } else { tw };
            sw += w;
            swx += w * x[j];
            swy += w * y[j];
            swxx += w * x[j] * x[j];
            swxy += w * x[j] * y[j];
        }
        let denom = sw * swxx - swx * swx;
        if sw > 0.0 && denom.abs() > 1e-12 * (sw * swxx).abs().max(1.0) {
            let b = (sw * swxy - swx * swy) / denom;
            let a = (swy - b * swx) / sw;
            a + b * x[i]
        } else if sw > 0.0 {
            swy / sw // degenerate (collinear x in window): weighted mean
        } else {
            f64::NAN // all robustness weights zero — caller refits without rw
        }
    };

    for it in 0..=iter {
        for (i, yhat_i) in yhat.iter_mut().enumerate() {
            let v = fit_at(i, &rw, true);
            // If robustness zeroed the whole neighborhood, fall back to the
            // non-robust (tricube-only) local fit rather than leaving the point
            // pinned to a raw outlier value.
            *yhat_i = if v.is_finite() {
                v
            } else {
                fit_at(i, &rw, false)
            };
        }
        if it == iter {
            break;
        }
        // Bisquare robustness weights from the median absolute residual.
        let resid: Vec<f64> = (0..n).map(|i| (y[i] - yhat[i]).abs()).collect();
        let mut sorted = resid.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cut = 6.0 * sorted[n / 2];
        for i in 0..n {
            rw[i] = if cut <= 0.0 {
                1.0
            } else {
                let u = resid[i] / cut;
                if u >= 1.0 {
                    0.0
                } else {
                    let t = 1.0 - u * u;
                    t * t
                }
            };
        }
    }
    yhat
}

/// Analytic derivative of [`nb_loglik_sum`] with respect to `log(alpha)`.
///
/// Using `theta = 1/alpha`, `d/d(log alpha) = -theta · dL/dtheta`, and per
/// observation
/// `dL/dtheta = ψ(y+theta) - ψ(theta) + log(theta) - log(theta+mu) + 1
///             - theta/(theta+mu) - y/(theta+mu)`.
///
/// This is the quantity the Phase-2 dispersion Newton step ascends; the
/// finite-difference test validates it against [`nb_loglik_sum`].
pub fn nb_loglik_sum_dlogalpha(y: &[f64], mu: &[f64], alpha: f64) -> f64 {
    debug_assert_eq!(y.len(), mu.len());
    let theta = 1.0 / alpha;
    let psi_theta = digamma(theta);
    let log_theta = theta.ln();
    let mut grad_theta = 0.0;
    for (&yi, &mui) in y.iter().zip(mu.iter()) {
        let tpm = theta + mui;
        grad_theta +=
            digamma(yi + theta) - psi_theta + log_theta - tpm.ln() + 1.0 - theta / tpm - yi / tpm;
    }
    -theta * grad_theta
}

/// Cox–Reid adjusted profile log-likelihood (spec §7.5):
/// `CR(alpha) = Σ_s logNB(y_s | mu_s, alpha) - 0.5·log det(XᵀW(alpha)X)`.
///
/// The `log_det_xtwx` term (`log det(XᵀWX)` at this `alpha`) is supplied by the
/// caller — assembling `XᵀWX` is the IRLS machinery's job (Phase 2), so this
/// stays a pure function of the count/mean vectors plus that scalar.
pub fn cox_reid_objective(y: &[f64], mu: &[f64], alpha: f64, log_det_xtwx: f64) -> f64 {
    nb_loglik_sum(y, mu, alpha) - 0.5 * log_det_xtwx
}

/// Derivative of [`cox_reid_objective`] with respect to `log(alpha)`.
///
/// `dCR/d(log alpha) = d(Σ logNB)/d(log alpha) - 0.5·d(log det XᵀWX)/d(log alpha)`.
/// The log-det derivative `dlogdet_dlogalpha` is supplied by the caller (Phase-2
/// IRLS machinery has the design and the `(XᵀWX)^{-1}`); pass `0.0` to recover
/// the plain profile-likelihood derivative.
pub fn cox_reid_dispersion_gradient(
    y: &[f64],
    mu: &[f64],
    alpha: f64,
    dlogdet_dlogalpha: f64,
) -> f64 {
    nb_loglik_sum_dlogalpha(y, mu, alpha) - 0.5 * dlogdet_dlogalpha
}

#[cfg(test)]
#[path = "math_tests.rs"]
mod tests;
