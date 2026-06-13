//! Pseudobulk negative-binomial GLM (NB-GLM) for differential expression.
//!
//! A Rust-native, **CPU-only**, `f64` end-to-end DESeq2-*core* fitter (IRLS /
//! Fisher scoring for the mean, Cox–Reid adjusted dispersion with a parametric
//! trend + empirical-Bayes shrinkage, Wald inference). See `GPU-NB-GLM-SPEC.md`.
//! There is no GPU path in v1 — a future GPU extension (spec §13) is additive.
//!
//! Layout: [`types`] (public surface), [`math`] (pure numerical primitives),
//! [`validate`] (input/contrast checks), [`size_factors`] (median-ratio factors),
//! [`irls`] (per-gene mean fit), [`dispersion`] (Cox–Reid MLE + trend + shrinkage),
//! [`wald`] (contrast inference). [`pseudobulk_nb_glm`] orchestrates them with
//! rayon gene-parallelism and deterministic output.

pub mod math;
pub mod types;

mod dispersion;
mod filtering;
mod irls;
mod size_factors;
mod validate;
mod wald;

pub use types::{
    DispersionMethod, DispersionTrend, NbGlmContrast, NbGlmDiagnostics, NbGlmOptions, NbGlmResult,
};

use rayon::prelude::*;

use crate::error::Result;
use dispersion::{
    estimate_prior_var, fit_dispersion, fit_dispersion_trend, moments_dispersion, DispPrior,
};
use size_factors::median_ratio_size_factors;
use validate::{validate_contrast, validate_inputs};

/// Init floor for the intercept coefficient (`ln(max(base_mean, floor))`, §7.3).
const MEAN_FLOOR: f64 = 1e-4;

/// Per-gene fitted state threaded through the orchestration pipeline.
#[derive(Clone)]
struct GeneState {
    beta: Vec<f64>,
    mu: Vec<f64>,
    fisher: Vec<f64>,
    alpha: f64,
    base_mean: f64,
    converged: bool,
    n_iter: u32,
    at_low: bool,
    at_high: bool,
    all_zero: bool,
}

impl GeneState {
    fn all_zero(n_features: usize, n_samples: usize, base_mean: f64) -> Self {
        GeneState {
            beta: vec![0.0; n_features],
            mu: vec![0.0; n_samples],
            fisher: vec![0.0; n_features * n_features],
            alpha: f64::NAN,
            base_mean,
            converged: true,
            n_iter: 0,
            at_low: false,
            at_high: false,
            all_zero: true,
        }
    }
}

/// Fit a pseudobulk negative-binomial GLM and test a contrast (spec §4.1, §8.2).
///
/// * `counts_gene_major` — `[n_genes × n_samples]` row-major, non-negative finite.
/// * `design_row_major` — `[n_samples × n_features]` row-major, full column rank.
/// * `size_factors` — `None` ⇒ DESeq2 median-ratio factors (§7.1).
///
/// Runs gene-parallel IRLS + Cox–Reid dispersion (with trend fit + empirical-Bayes
/// shrinkage for `DispersionMethod::CoxReidShrunk`), then Wald inference + BH on the
/// calling thread. Output is independent of rayon scheduling order.
// The 8-argument signature is the public API mandated by spec §4.1.
#[allow(clippy::too_many_arguments)]
pub fn pseudobulk_nb_glm(
    counts_gene_major: &[f64],
    n_genes: usize,
    n_samples: usize,
    design_row_major: &[f64],
    n_features: usize,
    size_factors: Option<&[f64]>,
    contrast: NbGlmContrast,
    options: NbGlmOptions,
) -> Result<NbGlmResult> {
    let start = std::time::Instant::now();

    validate_inputs(
        counts_gene_major,
        n_genes,
        n_samples,
        design_row_major,
        n_features,
        size_factors,
    )?;
    validate_contrast(&contrast, n_features)?;

    let sf: Vec<f64> = match size_factors {
        Some(s) => s.to_vec(),
        None => median_ratio_size_factors(counts_gene_major, n_genes, n_samples),
    };
    let log_sf: Vec<f64> = sf.iter().map(|v| v.ln()).collect();
    let c = wald::contrast_vector(&contrast, n_features);

    let row_of = |g: usize| &counts_gene_major[g * n_samples..(g + 1) * n_samples];
    let base_mean_of = |row: &[f64]| -> f64 {
        row.iter().zip(sf.iter()).map(|(&y, &s)| y / s).sum::<f64>() / n_samples as f64
    };

    // --- Per-gene MLE pass (gene-parallel): IRLS ↔ Cox–Reid dispersion. ---
    let mle: Vec<GeneState> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let row = row_of(g);
            let base_mean = base_mean_of(row);
            if row.iter().all(|&y| y == 0.0) {
                return GeneState::all_zero(n_features, n_samples, base_mean);
            }
            let mut beta_init = vec![0.0_f64; n_features];
            beta_init[0] = base_mean.max(MEAN_FLOOR).ln();
            let mut alpha = moments_dispersion(row, &sf, &options);
            let mut fit = irls::fit_gene_irls(
                row,
                design_row_major,
                &log_sf,
                n_samples,
                n_features,
                alpha,
                &beta_init,
                &options,
            );
            let mut n_outer = 1u32;
            let mut at_low = false;
            let mut at_high = false;
            // Moments (single pass) is trivially "outer-converged"; the Cox–Reid
            // path must reach the tolerance break to count as converged.
            let mut outer_converged = true;
            if options.dispersion != DispersionMethod::Moments {
                outer_converged = false;
                for _ in 0..options.max_outer_iters {
                    let df = fit_dispersion(
                        row,
                        &fit.mu,
                        design_row_major,
                        n_samples,
                        n_features,
                        alpha,
                        None,
                        &options,
                    );
                    let prev_alpha = alpha;
                    let prev_dev = fit.deviance;
                    alpha = df.alpha;
                    at_low = df.at_low;
                    at_high = df.at_high;
                    fit = irls::fit_gene_irls(
                        row,
                        design_row_major,
                        &log_sf,
                        n_samples,
                        n_features,
                        alpha,
                        &fit.beta,
                        &options,
                    );
                    n_outer += 1;
                    // The outer loop alternates dispersion (α) and mean (β); the
                    // inner IRLS already drives β to `irls_tol` for the current α,
                    // so the alternation has converged once α stops moving. (The
                    // deviance moves *with* α, so an extra rel-deviance test would
                    // just over-tighten the criterion.)
                    let _ = prev_dev;
                    // atol+rtol form: the `+ 1.0` floor means a near-Poisson
                    // dispersion (alpha ≈ 0, where alpha barely affects the
                    // variance) converges on an *absolute* change rather than
                    // chasing relative precision against a denominator ~1e-8.
                    let rel_a = (alpha - prev_alpha).abs() / (prev_alpha.abs() + 1.0);
                    if rel_a < options.outer_tol {
                        outer_converged = true;
                        break;
                    }
                }
            }
            GeneState {
                beta: fit.beta,
                mu: fit.mu,
                fisher: fit.fisher,
                alpha,
                base_mean,
                // Report converged only if both the inner IRLS and the outer
                // mean↔dispersion alternation reached tolerance.
                converged: fit.converged && outer_converged,
                n_iter: n_outer,
                at_low,
                at_high,
                all_zero: false,
            }
        })
        .collect();

    let alpha_mle: Vec<f64> = mle.iter().map(|m| m.alpha).collect();
    let base_means: Vec<f64> = mle.iter().map(|m| m.base_mean).collect();
    let valid: Vec<bool> = mle
        .iter()
        .map(|m| !m.all_zero && m.alpha.is_finite())
        .collect();

    // --- Cross-gene dispersion trend + empirical-Bayes shrinkage (CoxReidShrunk). ---
    let mut dispersion_trend = None;
    let mut dispersion_prior_var = None;
    let final_states: Vec<GeneState> = if options.dispersion == DispersionMethod::CoxReidShrunk
        && options.shrink_dispersion
    {
        let trend = if options.fit_dispersion_trend {
            fit_dispersion_trend(&base_means, &alpha_mle, &valid, &options)
        } else {
            None
        };
        // Per-gene log target: trend value, else global median of valid MLEs.
        let log_targets: Vec<f64> = match &trend {
            Some(t) => base_means
                .iter()
                .map(|&mu| t.eval(mu).max(options.min_disp).ln())
                .collect(),
            None => {
                let mut valid_alphas: Vec<f64> = (0..n_genes)
                    .filter(|&g| valid[g] && alpha_mle[g] > 0.0)
                    .map(|g| alpha_mle[g])
                    .collect();
                let median = if valid_alphas.is_empty() {
                    options.min_disp
                } else {
                    valid_alphas
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    valid_alphas[valid_alphas.len() / 2]
                };
                vec![median.max(options.min_disp).ln(); n_genes]
            }
        };
        let prior_var = estimate_prior_var(&log_targets, &alpha_mle, &valid, n_samples, n_features);
        dispersion_trend = trend;
        dispersion_prior_var = Some(prior_var);

        (0..n_genes)
            .into_par_iter()
            .map(|g| {
                if mle[g].all_zero {
                    return mle[g].clone();
                }
                let row = row_of(g);
                let prior = DispPrior {
                    log_alpha_trend: log_targets[g],
                    sigma_lr2: prior_var,
                };
                let df = fit_dispersion(
                    row,
                    &mle[g].mu,
                    design_row_major,
                    n_samples,
                    n_features,
                    mle[g].alpha,
                    Some(&prior),
                    &options,
                );
                // Refit beta once at the shrunken dispersion (§7.6 step 4).
                let fit = irls::fit_gene_irls(
                    row,
                    design_row_major,
                    &log_sf,
                    n_samples,
                    n_features,
                    df.alpha,
                    &mle[g].beta,
                    &options,
                );
                GeneState {
                    beta: fit.beta,
                    mu: fit.mu,
                    fisher: fit.fisher,
                    alpha: df.alpha,
                    base_mean: mle[g].base_mean,
                    converged: mle[g].converged && fit.converged,
                    n_iter: mle[g].n_iter,
                    at_low: df.at_low,
                    at_high: df.at_high,
                    all_zero: false,
                }
            })
            .collect()
    } else {
        mle.clone()
    };

    // --- Wald inference + Cook's distance (gene-parallel). ---
    // Cook's reuses the contrast covariance `wald_stat` already inverts (the
    // ridged Fisher inverse), so it costs only the per-sample leverage forms.
    let wald_out: Vec<(wald::WaldOut, f64)> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let st = &final_states[g];
            if st.all_zero {
                // Conservative, matching the non-converged Wald path (spec §7.7).
                let w = wald::WaldOut {
                    log2_fold_change: 0.0,
                    standard_error: f64::INFINITY,
                    wald_stat: 0.0,
                    p_value: 1.0,
                };
                return (w, f64::NAN);
            }
            if !options.compute_wald {
                let effect: f64 = c.iter().zip(st.beta.iter()).map(|(&ci, &bi)| ci * bi).sum();
                let w = wald::WaldOut {
                    log2_fold_change: effect / std::f64::consts::LN_2,
                    standard_error: f64::NAN,
                    wald_stat: f64::NAN,
                    p_value: f64::NAN,
                };
                return (w, f64::NAN);
            }
            let (w, cov) = wald::wald_stat(&st.beta, &st.fisher, n_features, &c);
            let cooks = match &cov {
                Some(cov) => filtering::cooks_distance(
                    cov,
                    &st.mu,
                    design_row_major,
                    row_of(g),
                    st.alpha,
                    n_samples,
                    n_features,
                ),
                None => f64::NAN,
            };
            (w, cooks)
        })
        .collect();

    let cooks: Vec<f64> = wald_out.iter().map(|(_, c)| *c).collect();
    let p_value_raw: Vec<f64> = wald_out.iter().map(|(w, _)| w.p_value).collect();

    // --- Cook's-distance outlier filtering (DESeq2 `results()` default). ---
    // Genes whose max Cook's distance exceeds the cutoff have their p-value (and
    // hence p_adj) set to NaN. Applied only when the residual df supports it
    // (DESeq2 requires m − p ≥ 3); log2FC / SE / stat are still reported.
    let residual_df = n_samples.saturating_sub(n_features);
    let cooks_cut = if options.compute_wald && options.cooks_filtering && residual_df >= 3 {
        Some(
            options
                .cooks_cutoff
                .unwrap_or_else(|| filtering::cooks_cutoff(n_features, residual_df)),
        )
    } else {
        None
    };
    let mut p_value = p_value_raw;
    let mut n_cooks_outliers = 0usize;
    if let Some(cut) = cooks_cut {
        for g in 0..n_genes {
            if !final_states[g].all_zero && cooks[g].is_finite() && cooks[g] > cut {
                p_value[g] = f64::NAN;
                n_cooks_outliers += 1;
            }
        }
    }

    // --- Multiple-testing correction: independent filtering (default) or masked BH. ---
    let mut n_independent_filtered = 0usize;
    let mut independent_filter_threshold = None;
    let p_adj = if !(options.compute_bh && options.compute_wald) {
        vec![f64::NAN; n_genes]
    } else if options.independent_filtering {
        let (adj, thr, n_filtered) =
            filtering::independent_filter(&base_means, &p_value, options.independent_filter_alpha);
        n_independent_filtered = n_filtered;
        independent_filter_threshold = thr;
        adj
    } else {
        filtering::benjamini_hochberg_masked(&p_value)
    };

    // --- Assemble result + diagnostics. ---
    let beta = if options.store_coefficients {
        let mut flat = vec![0.0_f64; n_genes * n_features];
        for (g, st) in final_states.iter().enumerate() {
            flat[g * n_features..(g + 1) * n_features].copy_from_slice(&st.beta);
        }
        Some(flat)
    } else {
        None
    };
    let fitted_means = if options.store_fitted_means {
        let mut flat = vec![0.0_f64; n_genes * n_samples];
        for (g, st) in final_states.iter().enumerate() {
            flat[g * n_samples..(g + 1) * n_samples].copy_from_slice(&st.mu);
        }
        Some(flat)
    } else {
        None
    };

    let dispersion: Vec<f64> = final_states.iter().map(|s| s.alpha).collect();
    let log2_fold_change: Vec<f64> = wald_out.iter().map(|(w, _)| w.log2_fold_change).collect();
    let standard_error: Vec<f64> = wald_out.iter().map(|(w, _)| w.standard_error).collect();
    let wald_stat: Vec<f64> = wald_out.iter().map(|(w, _)| w.wald_stat).collect();
    let converged: Vec<bool> = final_states.iter().map(|s| s.converged).collect();
    let n_iter: Vec<u32> = final_states.iter().map(|s| s.n_iter).collect();

    let n_all_zero_genes = final_states.iter().filter(|s| s.all_zero).count();
    let n_boundary_dispersion_low = final_states
        .iter()
        .filter(|s| !s.all_zero && s.at_low)
        .count();
    let n_boundary_dispersion_high = final_states
        .iter()
        .filter(|s| !s.all_zero && s.at_high)
        .count();
    let n_nonconverged = final_states
        .iter()
        .filter(|s| !s.all_zero && !s.converged)
        .count();

    let diagnostics = NbGlmDiagnostics {
        n_rank_deficient_design_columns: 0, // rank-deficient designs are rejected in validation
        n_all_zero_genes,
        n_boundary_dispersion_low,
        n_boundary_dispersion_high,
        n_nonconverged,
        n_cooks_outliers,
        n_independent_filtered,
        independent_filter_threshold,
        cooks_cutoff: cooks_cut,
        dispersion_trend,
        dispersion_prior_var,
        elapsed_fit_ms: Some(start.elapsed().as_secs_f64() * 1e3),
        rayon_threads_used: Some(rayon::current_num_threads()),
    };

    Ok(NbGlmResult {
        n_genes,
        n_samples,
        n_features,
        beta,
        dispersion,
        dispersion_mle: alpha_mle,
        log2_fold_change,
        standard_error,
        wald_stat,
        p_value,
        p_adj,
        cooks,
        base_mean: base_means,
        converged,
        n_iter,
        fitted_means,
        diagnostics,
    })
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
