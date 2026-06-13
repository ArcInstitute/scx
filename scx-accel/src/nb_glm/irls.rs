//! Per-gene IRLS / Fisher-scoring mean fit (spec §7.4, §8.1).
//!
//! Given a fixed dispersion `alpha`, fit the log-link NB mean coefficients `beta`
//! by iteratively reweighted least squares with NB working weights
//! `W = mu/(1+alpha*mu)`. The final `XᵀWX + ridge·I` is the expected Fisher
//! information — computed once here and reused by the dispersion log-det
//! (`dispersion.rs`) and the Wald covariance (`wald.rs`).

use faer::linalg::solvers::Solve;
use faer::{Mat, MatRef, Side};

use super::math::{clamp_eta, floor_mu, nb_irls_weight, nb_loglik_sum, nb_working_response};
use super::types::NbGlmOptions;

/// Outcome of a single-gene IRLS fit at a fixed dispersion.
#[derive(Debug, Clone)]
pub(crate) struct GeneFit {
    /// Mean-model coefficients, length `n_features` (natural-log scale).
    pub beta: Vec<f64>,
    /// Fitted means `mu`, length `n_samples`.
    pub mu: Vec<f64>,
    /// Final deviance `-2·logL` at the fitted `beta`.
    pub deviance: f64,
    /// `XᵀWX + ridge·I`, `n_features²` row-major (symmetric). The Fisher
    /// information used for Wald SEs and the Cox–Reid log-det.
    pub fisher: Vec<f64>,
    /// Whether the deviance converged within `max_irls_iters`.
    pub converged: bool,
}

/// Assemble `XᵀWX + ridge·I` (`n_features²`, row-major, symmetric) from a diagonal
/// weight vector `w` (length `n_samples`). Shared by the IRLS Fisher matrix and the
/// dispersion log-det so both use the identical `M(α)`.
pub(crate) fn assemble_xtwx(
    design: &[f64],
    w: &[f64],
    n_samples: usize,
    n_features: usize,
    ridge: f64,
) -> Vec<f64> {
    let mut xtwx = vec![0.0_f64; n_features * n_features];
    for s in 0..n_samples {
        let row = &design[s * n_features..(s + 1) * n_features];
        let ws = w[s];
        for j in 0..n_features {
            let wxj = ws * row[j];
            // upper triangle (incl. diagonal)
            for k in j..n_features {
                xtwx[j * n_features + k] += wxj * row[k];
            }
        }
    }
    // mirror upper → lower, add ridge on the diagonal
    for j in 0..n_features {
        for k in (j + 1)..n_features {
            xtwx[k * n_features + j] = xtwx[j * n_features + k];
        }
        xtwx[j * n_features + j] += ridge;
    }
    xtwx
}

/// Solve the symmetric system `a · x = b` (`a` is `n²` row-major, `b` length `n`).
/// Tries Cholesky (SPD) first, falls back to partial-pivot LU on a non-PD matrix.
/// Returns `None` if the solution is non-finite.
pub(crate) fn solve_spd(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let a_view = MatRef::from_row_major_slice(a, n, n);
    let mut rhs = Mat::<f64>::zeros(n, 1);
    for (j, &bj) in b.iter().enumerate() {
        rhs[(j, 0)] = bj;
    }
    let sol = match a_view.llt(Side::Lower) {
        Ok(llt) => llt.solve(rhs.as_ref()),
        Err(_) => a_view.partial_piv_lu().solve(rhs.as_ref()),
    };
    let out: Vec<f64> = (0..n).map(|j| sol[(j, 0)]).collect();
    if out.iter().all(|v| v.is_finite()) {
        Some(out)
    } else {
        None
    }
}

/// Compute `eta`, `mu`, working weight `w`, and working response `z` for the
/// current `beta` into caller-provided scratch buffers (each length `n_samples`).
#[allow(clippy::too_many_arguments)]
fn working_arrays(
    counts_row: &[f64],
    design: &[f64],
    log_size_factors: &[f64],
    beta: &[f64],
    n_samples: usize,
    n_features: usize,
    alpha: f64,
    opts: &NbGlmOptions,
    mu: &mut [f64],
    w: &mut [f64],
    z: &mut [f64],
) {
    for s in 0..n_samples {
        let row = &design[s * n_features..(s + 1) * n_features];
        let mut eta = log_size_factors[s];
        for k in 0..n_features {
            eta += row[k] * beta[k];
        }
        let eta_c = clamp_eta(eta, opts.eta_min, opts.eta_max);
        let mu_s = floor_mu(eta_c.exp(), opts.min_mu);
        mu[s] = mu_s;
        w[s] = nb_irls_weight(mu_s, alpha);
        z[s] = nb_working_response(eta_c, log_size_factors[s], counts_row[s], mu_s);
    }
}

/// Fit `beta` for one gene at a fixed dispersion `alpha` (spec §7.4).
///
/// `design` is `[n_samples × n_features]` row-major; `beta_init` (length
/// `n_features`) seeds the iteration (warm-started across outer passes by the
/// orchestrator). Convergence is on the relative change in deviance.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_gene_irls(
    counts_row: &[f64],
    design: &[f64],
    log_size_factors: &[f64],
    n_samples: usize,
    n_features: usize,
    alpha: f64,
    beta_init: &[f64],
    opts: &NbGlmOptions,
) -> GeneFit {
    let mut beta = beta_init.to_vec();
    let mut mu = vec![0.0_f64; n_samples];
    let mut w = vec![0.0_f64; n_samples];
    let mut z = vec![0.0_f64; n_samples];

    let mut prev_dev = f64::INFINITY;
    let mut converged = false;

    for iter in 0..opts.max_irls_iters {
        working_arrays(
            counts_row,
            design,
            log_size_factors,
            &beta,
            n_samples,
            n_features,
            alpha,
            opts,
            &mut mu,
            &mut w,
            &mut z,
        );
        let dev = -2.0 * nb_loglik_sum(counts_row, &mu, alpha);
        if iter > 0 && (prev_dev - dev).abs() <= opts.irls_tol * (dev.abs() + 1e-8) {
            converged = true;
            break;
        }
        prev_dev = dev;

        // Build XᵀWX (+ridge) and XᵀWz, then solve for the next beta.
        let xtwx = assemble_xtwx(design, &w, n_samples, n_features, opts.beta_ridge);
        let mut xtwz = vec![0.0_f64; n_features];
        for s in 0..n_samples {
            let row = &design[s * n_features..(s + 1) * n_features];
            let wz = w[s] * z[s];
            for (j, &xj) in row.iter().enumerate() {
                xtwz[j] += wz * xj;
            }
        }
        match solve_spd(&xtwx, &xtwz, n_features) {
            Some(next) => beta = next,
            None => break, // non-finite solve → stop, mark non-converged
        }
    }

    // Final consistent pass so mu / fisher match the returned beta regardless of
    // how the loop exited (convergence, iteration cap, or a failed solve).
    working_arrays(
        counts_row,
        design,
        log_size_factors,
        &beta,
        n_samples,
        n_features,
        alpha,
        opts,
        &mut mu,
        &mut w,
        &mut z,
    );
    let fisher = assemble_xtwx(design, &w, n_samples, n_features, opts.beta_ridge);
    let deviance = -2.0 * nb_loglik_sum(counts_row, &mu, alpha);

    GeneFit {
        beta,
        mu,
        deviance,
        fisher,
        converged,
    }
}

#[cfg(test)]
#[path = "irls_tests.rs"]
mod tests;
