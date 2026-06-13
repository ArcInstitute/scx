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
