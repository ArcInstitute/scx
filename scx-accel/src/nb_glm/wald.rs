//! Wald inference for a contrast (spec §7.7) and the BH wrapper (§7.8).
//!
//! The contrast covariance is `cov(beta) = (XᵀWX + ridge)⁻¹` from the final IRLS
//! Fisher information. Ill-conditioned / non-PD information after the ridge yields
//! conservative output (`p=1`, `stat=0`, `se=inf`) rather than a spurious call.

use faer::linalg::solvers::DenseSolveCore;
use faer::{MatRef, Side};

use super::types::NbGlmContrast;
use crate::diffexp::normal_sf;

/// Per-gene Wald output for a contrast.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WaldOut {
    pub log2_fold_change: f64,
    pub standard_error: f64,
    pub wald_stat: f64,
    pub p_value: f64,
}

/// Build the contrast weight vector `c` (length `n_features`) from an
/// [`NbGlmContrast`]. The caller has already validated it (`validate_contrast`).
pub(crate) fn contrast_vector(contrast: &NbGlmContrast, n_features: usize) -> Vec<f64> {
    match contrast {
        NbGlmContrast::Coefficient { index } => {
            let mut c = vec![0.0_f64; n_features];
            if *index < n_features {
                c[*index] = 1.0;
            }
            c
        }
        NbGlmContrast::Vector { weights } => weights.clone(),
    }
}

/// Invert the Fisher information (`n²` row-major, symmetric PD). Cholesky first,
/// partial-pivot LU fallback. Returns the inverse as `n²` row-major, or `None`.
fn invert(fisher: &[f64], n: usize) -> Option<Vec<f64>> {
    let a = MatRef::from_row_major_slice(fisher, n, n);
    let inv = match a.llt(Side::Lower) {
        Ok(llt) => llt.inverse(),
        Err(_) => a.partial_piv_lu().inverse(),
    };
    let mut out = vec![0.0_f64; n * n];
    for r in 0..n {
        for c in 0..n {
            out[r * n + c] = inv[(r, c)];
        }
    }
    if out.iter().all(|v| v.is_finite()) {
        Some(out)
    } else {
        None
    }
}

/// Compute the Wald statistics for contrast `c` on a single gene (spec §7.7).
/// `log2_fold_change = (c·beta)/ln2` is always reported; the SE/stat/p degrade
/// conservatively when the covariance is unusable.
///
/// Also returns the contrast covariance `cov(beta) = (XᵀWX + ridge)⁻¹` (row-major
/// `n_features²`) when the Fisher information inverts, or `None` when it does not.
/// The Cook's-distance pass reuses this inverse (`mod.rs`), avoiding a second
/// per-gene matrix inversion.
pub(crate) fn wald_stat(
    beta: &[f64],
    fisher: &[f64],
    n_features: usize,
    c: &[f64],
) -> (WaldOut, Option<Vec<f64>>) {
    let effect: f64 = c.iter().zip(beta.iter()).map(|(&ci, &bi)| ci * bi).sum();
    let log2_fold_change = effect / std::f64::consts::LN_2;

    let conservative = WaldOut {
        log2_fold_change,
        standard_error: f64::INFINITY,
        wald_stat: 0.0,
        p_value: 1.0,
    };

    let cov = match invert(fisher, n_features) {
        Some(cov) => cov,
        None => return (conservative, None),
    };

    // var = cᵀ cov c
    let mut var = 0.0;
    for j in 0..n_features {
        let mut cov_c_j = 0.0;
        for k in 0..n_features {
            cov_c_j += cov[j * n_features + k] * c[k];
        }
        var += c[j] * cov_c_j;
    }

    if !(var.is_finite() && var > 0.0) {
        // The covariance is still valid for leverage even if this contrast's
        // variance degenerates, so return it for the Cook's pass.
        return (conservative, Some(cov));
    }

    let se = var.sqrt();
    let stat = effect / se;
    let p = (2.0 * normal_sf(stat.abs())).clamp(0.0, 1.0);
    (
        WaldOut {
            log2_fold_change,
            standard_error: se,
            wald_stat: stat,
            p_value: p,
        },
        Some(cov),
    )
}

#[cfg(test)]
#[path = "wald_tests.rs"]
mod tests;
