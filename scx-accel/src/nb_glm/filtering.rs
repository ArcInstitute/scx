//! DESeq2 `results()`-stage gene filtering (spec §19, v2): Cook's-distance
//! outlier detection and independent filtering on the base mean.
//!
//! Cook's distance flags genes with a single high-leverage, high-residual sample
//! (their `p_value`/`p_adj` become NaN). Independent filtering removes low-base-mean
//! genes from the multiple-testing burden (their `p_adj` becomes NaN), choosing the
//! cutoff that maximizes rejections via the genefilter lowess + 1-SE rule. Both are
//! default-on to match DESeq2; both degrade to no-ops when they cannot apply.

use super::math::{f_quantile, lowess, nb_irls_weight};
use crate::diffexp::benjamini_hochberg;

/// DESeq2 Cook's-distance cutoff `qf(0.99, n_features, residual_df)` — the 0.99
/// quantile of the F distribution.
pub(crate) fn cooks_cutoff(n_features: usize, residual_df: usize) -> f64 {
    f_quantile(0.99, n_features as f64, residual_df as f64)
}

/// Maximum Cook's distance over samples for one gene (DESeq2 `calculateCooksDistance`):
/// `D = max_s (pearson_s² / p) · (h_s / (1 − h_s)²)`, where the leverage
/// `h_s = w_s · (x_sᵀ M⁻¹ x_s)`, `w_s = mu_s/(1 + α·mu_s)` is the NB working weight,
/// `M⁻¹ = cov` is the (reused) Fisher inverse, and the Pearson residual is
/// `(y_s − mu_s)/sqrt(mu_s + α·mu_s²)`.
///
/// `cov` is row-major `n_features²`. Returns 0.0 if no finite contribution (e.g.
/// `n_samples == 0`); a sample at leverage `h_s ≥ 1` yields `+∞` (always an outlier).
pub(crate) fn cooks_distance(
    cov: &[f64],
    mu: &[f64],
    design: &[f64],
    counts_row: &[f64],
    alpha: f64,
    n_samples: usize,
    n_features: usize,
) -> f64 {
    let p = n_features as f64;
    let mut max_d = 0.0_f64;
    for s in 0..n_samples {
        let mu_s = mu[s];
        let var = mu_s + alpha * mu_s * mu_s;
        if !var.is_finite() || var <= 0.0 {
            continue;
        }
        let w_s = nb_irls_weight(mu_s, alpha);
        // h_s = w_s · x_sᵀ M⁻¹ x_s
        let row = &design[s * n_features..(s + 1) * n_features];
        let mut q = 0.0;
        for j in 0..n_features {
            let mut mij_xj = 0.0;
            for k in 0..n_features {
                mij_xj += cov[j * n_features + k] * row[k];
            }
            q += row[j] * mij_xj;
        }
        let h = w_s * q;
        if !h.is_finite() {
            continue;
        }
        // h ≥ 1 — or 1−h underflowing to 0 in f64 — is maximal leverage, a
        // definite outlier; flag it rather than letting `1/(1−h)²` skip it.
        let one_minus_h = 1.0 - h;
        if one_minus_h <= 0.0 {
            return f64::INFINITY;
        }
        let resid = counts_row[s] - mu_s;
        let pearson2 = resid * resid / var;
        let d = (pearson2 / p) * (h / (one_minus_h * one_minus_h));
        // An overflowed (non-finite) Cook's distance is also an extreme outlier.
        if !d.is_finite() {
            return f64::INFINITY;
        }
        if d > max_d {
            max_d = d;
        }
    }
    max_d
}

/// Benjamini–Hochberg over a vector that may contain NaN p-values: NaN entries are
/// excluded from the test (and stay NaN in the output), and the BH denominator is
/// the count of finite p-values. Unlike [`crate::diffexp::benjamini_hochberg`],
/// this is correct in the presence of NaN (DESeq2 / `p.adjust(na.rm=TRUE)` behavior).
pub(crate) fn benjamini_hochberg_masked(pvals: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; pvals.len()];
    let finite: Vec<usize> = (0..pvals.len()).filter(|&i| pvals[i].is_finite()).collect();
    if finite.is_empty() {
        return out;
    }
    let sub: Vec<f64> = finite.iter().map(|&i| pvals[i]).collect();
    let adj = benjamini_hochberg(&sub);
    for (slot, &i) in finite.iter().enumerate() {
        out[i] = adj[slot];
    }
    out
}

/// Type-7 (R default) quantile of an ascending-sorted slice at `q ∈ [0, 1]`.
/// `q` is clamped to `[0, 1]` so an out-of-range value can't index out of bounds.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let q = q.clamp(0.0, 1.0);
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let h = (n as f64 - 1.0) * q;
    let lo = h.floor() as usize;
    let frac = h - lo as f64;
    if lo + 1 < n {
        sorted[lo] + frac * (sorted[lo + 1] - sorted[lo])
    } else {
        sorted[n - 1]
    }
}

/// Number of theta points in the independent-filtering base-mean grid (DESeq2 uses 50).
const N_THETA: usize = 50;

/// DESeq2-style independent filtering on the base mean (genefilter algorithm).
///
/// Over a grid of base-mean quantile cutoffs, count the BH rejections at level
/// `alpha` among retained genes; smooth `numRej ~ theta` with `lowess(f=1/5)`; pick
/// the first theta whose smoothed value is within one RMSE of the maximum (the
/// "1-SE rule", which favors the smallest cutoff). Genes below the chosen base-mean
/// cutoff get `p_adj = NaN`; BH is recomputed over the retained set. Genes with a
/// NaN `p_value` (e.g. Cook's outliers) stay NaN throughout.
///
/// Matches genefilter on the 1-SE rule: the RMSE is over `numRej > 0` points only,
/// and `max(numRej) <= 10` short-circuits to no filtering.
///
/// Returns `(p_adj, base_mean_threshold, n_filtered)`. Degrades to plain masked BH
/// (no filtering) when too few genes are eligible or no cutoff improves on retaining
/// everything.
pub(crate) fn independent_filter(
    base_mean: &[f64],
    p_value: &[f64],
    alpha: f64,
) -> (Vec<f64>, Option<f64>, usize) {
    let n = p_value.len();
    let finite: Vec<usize> = (0..n).filter(|&i| p_value[i].is_finite()).collect();
    // Need enough genes for filtering to be meaningful.
    if finite.len() < 10 {
        return (benjamini_hochberg_masked(p_value), None, 0);
    }

    let mut bm_sorted: Vec<f64> = finite.iter().map(|&i| base_mean[i]).collect();
    bm_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Rejection count at each base-mean cutoff (BH recomputed over retained genes).
    let count_rejections = |cutoff: f64| -> usize {
        let retained: Vec<f64> = finite
            .iter()
            .filter(|&&i| base_mean[i] >= cutoff)
            .map(|&i| p_value[i])
            .collect();
        if retained.is_empty() {
            return 0;
        }
        benjamini_hochberg(&retained)
            .iter()
            .filter(|&&a| a < alpha)
            .count()
    };

    let thetas: Vec<f64> = (0..N_THETA)
        .map(|k| 0.95 * k as f64 / (N_THETA as f64 - 1.0))
        .collect();
    let num_rej: Vec<f64> = thetas
        .iter()
        .map(|&t| count_rejections(quantile_sorted(&bm_sorted, t)) as f64)
        .collect();

    // No-op guard: if no cutoff strictly beats retaining everything (theta=0),
    // filtering cannot help — keep all genes (matches DESeq2's all-finite padj
    // when the filter is uninformative, e.g. near-uniform base means).
    let base_rej = num_rej[0];
    if num_rej.iter().all(|&v| v <= base_rej) {
        return (benjamini_hochberg_masked(p_value), None, 0);
    }

    // genefilter short-circuit: with very few rejections the optimum is noise —
    // keep all genes (DESeq2 `if (max(numRej) <= 10) j <- 1`).
    let max_rej = num_rej.iter().copied().fold(0.0_f64, f64::max);
    if max_rej <= 10.0 {
        return (benjamini_hochberg_masked(p_value), None, 0);
    }

    // Smooth and apply the 1-SE rule: first theta within one RMSE of the max.
    // The RMSE is taken over the points with positive rejections only (genefilter:
    // `residual <- numRej[numRej>0] - lo.fit$y[numRej>0]`), so the zero-rejection
    // tail doesn't deflate it.
    let smoothed = lowess(&thetas, &num_rej, 0.2, 3);
    let max_s = smoothed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (mut sse, mut cnt) = (0.0_f64, 0usize);
    for (k, &y) in num_rej.iter().enumerate() {
        if y > 0.0 {
            let r = y - smoothed[k];
            sse += r * r;
            cnt += 1;
        }
    }
    let rmse = if cnt > 0 {
        (sse / cnt as f64).sqrt()
    } else {
        0.0
    };
    let thresh = max_s - rmse;
    let j = smoothed.iter().position(|&v| v >= thresh).unwrap_or(0);
    let chosen_theta = thetas[j];

    if chosen_theta <= 0.0 {
        return (benjamini_hochberg_masked(p_value), None, 0);
    }

    let cutoff = quantile_sorted(&bm_sorted, chosen_theta);
    let retained_idx: Vec<usize> = finite
        .iter()
        .copied()
        .filter(|&i| base_mean[i] >= cutoff)
        .collect();
    let retained_p: Vec<f64> = retained_idx.iter().map(|&i| p_value[i]).collect();
    let retained_adj = benjamini_hochberg(&retained_p);
    let mut p_adj = vec![f64::NAN; n];
    for (slot, &i) in retained_idx.iter().enumerate() {
        p_adj[i] = retained_adj[slot];
    }
    let n_filtered = finite.len() - retained_idx.len();
    (p_adj, Some(cutoff), n_filtered)
}

#[cfg(test)]
#[path = "filtering_tests.rs"]
mod tests;
