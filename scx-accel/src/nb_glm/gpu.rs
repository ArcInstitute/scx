//! GPU orchestrator for the pseudobulk NB-GLM (Stage A).
//!
//! Mirrors [`super::pseudobulk_nb_glm`] but replaces the two gene-parallel fit
//! phases (per-gene MLE; shrinkage refit) with GPU kernel launches via
//! `scx_gpu::gpu_nb_glm_fit`. Every cross-gene step (dispersion trend, prior
//! variance) and the post-fit tail (Wald, Cook's, filtering, BH, assembly) reuse
//! the host code unchanged — see [`super::assemble_result`]. The output is the
//! same [`NbGlmResult`] the CPU path produces, within the cross-validation
//! tolerances in `DE-GPU-ACC.md` §10.
//!
//! The fit is `f64` end-to-end (kernel + host), so GPU↔CPU agreement is bounded
//! only by the ported special functions and the root-finder's convergence path,
//! not by precision.

#![cfg(feature = "gpu")]

use scx_gpu::{
    gpu_nb_glm_fit, GpuDevice, GpuNbGlmFit, GpuNbGlmOpts, GpuNbGlmPass, GPU_NB_GLM_METHOD_CR_MLE,
    GPU_NB_GLM_METHOD_CR_SHRUNK, GPU_NB_GLM_METHOD_MOMENTS, GPU_NB_GLM_NSUB_MAX, GPU_NB_GLM_PMAX,
};

use super::dispersion::{estimate_prior_var, fit_dispersion_trend};
use super::size_factors::median_ratio_size_factors;
use super::validate::{validate_contrast, validate_inputs};
use super::{assemble_result, wald, GeneState};
use crate::error::{AccelError, Result};
use crate::nb_glm::{DispersionMethod, NbGlmContrast, NbGlmOptions, NbGlmResult};

/// Map `DispersionMethod` → the kernel's integer method tag.
fn method_tag(m: DispersionMethod) -> i32 {
    match m {
        DispersionMethod::Moments => GPU_NB_GLM_METHOD_MOMENTS,
        DispersionMethod::CoxReidMle => GPU_NB_GLM_METHOD_CR_MLE,
        DispersionMethod::CoxReidShrunk => GPU_NB_GLM_METHOD_CR_SHRUNK,
    }
}

fn gpu_opts(options: &NbGlmOptions) -> GpuNbGlmOpts {
    GpuNbGlmOpts {
        method: method_tag(options.dispersion),
        max_irls_iters: options.max_irls_iters as i32,
        irls_tol: options.irls_tol,
        min_mu: options.min_mu,
        eta_min: options.eta_min,
        eta_max: options.eta_max,
        beta_ridge: options.beta_ridge,
        min_disp: options.min_disp,
        max_disp: options.max_disp,
        disp_newton_iters: options.disp_newton_iters as i32,
        max_outer_iters: options.max_outer_iters as i32,
        outer_tol: options.outer_tol,
    }
}

/// Reconstruct one gene's [`GeneState`] from a flat GPU fit. `all_zero` is taken
/// from the counts row (canonical, matching the CPU `GeneState::all_zero`).
fn state_from_fit(
    fit: &GpuNbGlmFit,
    g: usize,
    n_samples: usize,
    n_features: usize,
    base_mean: f64,
    all_zero: bool,
) -> GeneState {
    if all_zero {
        return GeneState::all_zero(n_features, n_samples, base_mean);
    }
    GeneState {
        beta: fit.beta[g * n_features..(g + 1) * n_features].to_vec(),
        mu: fit.mu[g * n_samples..(g + 1) * n_samples].to_vec(),
        fisher: fit.fisher[g * n_features * n_features..(g + 1) * n_features * n_features].to_vec(),
        alpha: fit.alpha[g],
        base_mean,
        converged: fit.converged[g] == 1,
        n_iter: fit.n_iter[g],
        at_low: fit.at_low[g] == 1,
        at_high: fit.at_high[g] == 1,
        all_zero: false,
    }
}

/// GPU pseudobulk NB-GLM fit + contrast test. Signature mirrors
/// [`super::pseudobulk_nb_glm`] with a leading [`GpuDevice`] handle.
///
/// Requires `n_features ≤ GPU_NB_GLM_PMAX` and `n_samples ≤ GPU_NB_GLM_NSUB_MAX`;
/// the route planner gates these so larger inputs never reach here, but they are
/// also checked defensively.
#[allow(clippy::too_many_arguments)]
pub fn gpu_pseudobulk_nb_glm(
    dev: &GpuDevice,
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

    if n_features > GPU_NB_GLM_PMAX || n_samples > GPU_NB_GLM_NSUB_MAX {
        return Err(AccelError::InvalidInput(format!(
            "GPU NB-GLM supports n_features ≤ {GPU_NB_GLM_PMAX} and n_samples ≤ \
             {GPU_NB_GLM_NSUB_MAX}; got n_features={n_features}, n_samples={n_samples} \
             (route to device=\"cpu\")"
        )));
    }

    let sf: Vec<f64> = match size_factors {
        Some(s) => s.to_vec(),
        None => median_ratio_size_factors(counts_gene_major, n_genes, n_samples),
    };
    let log_sf: Vec<f64> = sf.iter().map(|v| v.ln()).collect();
    let c = wald::contrast_vector(&contrast, n_features);

    let base_means: Vec<f64> = (0..n_genes)
        .map(|g| {
            let row = &counts_gene_major[g * n_samples..(g + 1) * n_samples];
            row.iter().zip(sf.iter()).map(|(&y, &s)| y / s).sum::<f64>() / n_samples as f64
        })
        .collect();
    let all_zero: Vec<bool> = (0..n_genes)
        .map(|g| {
            counts_gene_major[g * n_samples..(g + 1) * n_samples]
                .iter()
                .all(|&y| y == 0.0)
        })
        .collect();

    let gopts = gpu_opts(&options);

    // --- GPU pass 1: per-gene MLE (IRLS ↔ Cox–Reid). ---
    let mle = gpu_nb_glm_fit(
        dev,
        counts_gene_major,
        design_row_major,
        &log_sf,
        &sf,
        n_genes,
        n_samples,
        n_features,
        gopts,
        GpuNbGlmPass::Mle {
            base_mean: &base_means,
        },
    )
    .map_err(|e| AccelError::LinAlg(format!("GPU NB-GLM MLE fit: {e}")))?;

    let alpha_mle: Vec<f64> = mle.alpha.clone();
    let mle_states: Vec<GeneState> = (0..n_genes)
        .map(|g| state_from_fit(&mle, g, n_samples, n_features, base_means[g], all_zero[g]))
        .collect();

    // --- Cross-gene trend + empirical-Bayes shrinkage (host) + GPU refit. ---
    let mut dispersion_trend = None;
    let mut dispersion_prior_var = None;
    let final_states: Vec<GeneState> = if options.dispersion == DispersionMethod::CoxReidShrunk
        && options.shrink_dispersion
    {
        let valid: Vec<bool> = (0..n_genes)
            .map(|g| !all_zero[g] && alpha_mle[g].is_finite())
            .collect();
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

        // GPU pass 2: shrink dispersion against the prior + refit β.
        let shr = gpu_nb_glm_fit(
            dev,
            counts_gene_major,
            design_row_major,
            &log_sf,
            &sf,
            n_genes,
            n_samples,
            n_features,
            gopts,
            GpuNbGlmPass::Shrink {
                beta_in: &mle.beta,
                alpha_in: &mle.alpha,
                prior_log_target: &log_targets,
                prior_var,
            },
        )
        .map_err(|e| AccelError::LinAlg(format!("GPU NB-GLM shrink fit: {e}")))?;

        (0..n_genes)
            .map(|g| {
                if all_zero[g] {
                    return mle_states[g].clone();
                }
                let mut st = state_from_fit(&shr, g, n_samples, n_features, base_means[g], false);
                // converged = MLE ∧ refit; n_iter carries the MLE outer count
                // (matches the CPU phase-2b assembly in mod.rs).
                st.converged = mle_states[g].converged && st.converged;
                st.n_iter = mle_states[g].n_iter;
                st
            })
            .collect()
    } else {
        mle_states
    };

    Ok(assemble_result(
        counts_gene_major,
        n_genes,
        n_samples,
        design_row_major,
        n_features,
        &c,
        &options,
        &final_states,
        alpha_mle,
        base_means,
        dispersion_trend,
        dispersion_prior_var,
        start,
    ))
}

#[cfg(test)]
#[path = "gpu_tests.rs"]
mod tests;
