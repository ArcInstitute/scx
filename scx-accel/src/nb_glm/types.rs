//! Public types for the pseudobulk negative-binomial GLM (NB-GLM).
//!
//! These mirror the type surface in `GPU-NB-GLM-SPEC.md` §4.1–4.2. The fitter is
//! CPU-only and `f64` end-to-end (there is no GPU path in v1), so none of these
//! types carry a backend/device dimension — a future GPU extension is additive
//! (spec §13) and would introduce its own enum arm rather than change these.

/// Tunable knobs for the NB-GLM fit. Constructed via [`Default`] in the common
/// case; every field has a documented default below.
///
/// Note the deliberate absence of Adam hyperparameters (learning rate, `beta1`,
/// `beta2`, gradient clipping, …) — v1 fits with IRLS / Fisher scoring for the
/// mean and a safeguarded Newton step for the dispersion (spec §7), so there is
/// nothing to tune by hand.
#[derive(Debug, Clone)]
pub struct NbGlmOptions {
    // --- IRLS (mean fit, spec §7.4) ---
    /// Maximum IRLS iterations per inner mean fit (converges in ~5–15).
    pub max_irls_iters: usize,
    /// Relative-change-in-deviance convergence tolerance for IRLS.
    pub irls_tol: f64,
    /// Floor applied to `mu` after `exp` to keep weights finite.
    pub min_mu: f64,
    /// Lower clamp on the linear predictor `eta`.
    pub eta_min: f64,
    /// Upper clamp on the linear predictor `eta`.
    pub eta_max: f64,
    /// Ridge added to the diagonal of `XᵀWX` before the solve.
    pub beta_ridge: f64,

    // --- Dispersion (spec §7.5–7.6) ---
    /// Which dispersion estimator to use.
    pub dispersion: DispersionMethod,
    /// Lower clamp on the NB dispersion `alpha`.
    pub min_disp: f64,
    /// Upper clamp on the NB dispersion `alpha`.
    pub max_disp: f64,
    /// Iteration cap for the safeguarded 1-D dispersion Newton/Brent step.
    pub disp_newton_iters: usize,
    /// Fit a parametric mean→dispersion trend across genes.
    pub fit_dispersion_trend: bool,
    /// Apply empirical-Bayes dispersion shrinkage toward the trend.
    pub shrink_dispersion: bool,

    // --- Outer loop (spec §3.2) ---
    /// Maximum outer (mean ↔ dispersion) alternations per gene.
    pub max_outer_iters: usize,
    /// Relative convergence tolerance for the outer mean↔dispersion alternation
    /// (on the dispersion `alpha`). `1e-4` (dispersion to ~4 significant figures)
    /// is ample — LFC and Wald p-values are insensitive to `alpha` far below
    /// this — and is reliably reachable within `max_outer_iters` for a coupled
    /// fixed point, unlike a needlessly tight `1e-6`.
    pub outer_tol: f64,

    // --- Inference / outputs ---
    /// Compute Wald statistics, SEs, and p-values.
    pub compute_wald: bool,
    /// Compute Benjamini–Hochberg adjusted p-values.
    pub compute_bh: bool,
    /// Retain the fitted `beta` coefficients in the result.
    pub store_coefficients: bool,
    /// Retain the fitted means `mu` in the result.
    pub store_fitted_means: bool,
}

impl Default for NbGlmOptions {
    fn default() -> Self {
        Self {
            max_irls_iters: 100,
            irls_tol: 1e-8,
            min_mu: 1e-10,
            eta_min: -30.0,
            eta_max: 30.0,
            beta_ridge: 1e-6,

            dispersion: DispersionMethod::CoxReidShrunk,
            min_disp: 1e-8,
            max_disp: 100.0,
            disp_newton_iters: 25,
            fit_dispersion_trend: true,
            shrink_dispersion: true,

            max_outer_iters: 10,
            outer_tol: 1e-4,

            compute_wald: true,
            compute_bh: true,
            store_coefficients: true,
            store_fitted_means: false,
        }
    }
}

/// Dispersion estimation strategy (spec §4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispersionMethod {
    /// Method-of-moments only (fast, noisy at low n; for testing / large-n).
    Moments,
    /// Per-gene Cox–Reid adjusted profile-likelihood MLE (no shrinkage).
    CoxReidMle,
    /// `CoxReidMle` + parametric trend fit + empirical-Bayes shrinkage (default).
    CoxReidShrunk,
}

/// The hypothesis tested by the Wald step (spec §4.2).
#[derive(Debug, Clone)]
pub enum NbGlmContrast {
    /// Test a single coefficient (e.g. the treatment column) against zero.
    Coefficient { index: usize },
    /// Arbitrary contrast vector `c`; tests `c·beta = 0`.
    Vector { weights: Vec<f64> },
}

/// Fitted parametric mean→dispersion trend `alpha_trend(mu) = a0 + a1 / mu`
/// (DESeq2-style, spec §7.6).
#[derive(Debug, Clone, Copy)]
pub struct DispersionTrend {
    /// Asymptotic dispersion (intercept term).
    pub a0: f64,
    /// `1/mu` coefficient.
    pub a1: f64,
}

impl DispersionTrend {
    /// Evaluate the trend at a base mean `mu_bar`.
    pub fn eval(&self, mu_bar: f64) -> f64 {
        self.a0 + self.a1 / mu_bar
    }
}

/// Per-fit diagnostics (spec §4.2). Carries the cross-gene trend/shrinkage state
/// and boundary/convergence counts; NB-GLM-specific numbers live here rather than
/// in the `adata.uns["scx_accel"]` route dict (spec §9).
#[derive(Debug, Clone)]
pub struct NbGlmDiagnostics {
    /// Design columns dropped/flagged as rank-deficient.
    pub n_rank_deficient_design_columns: usize,
    /// Genes with all-zero counts (skipped fit).
    pub n_all_zero_genes: usize,
    /// Genes whose dispersion hit the lower clamp.
    pub n_boundary_dispersion_low: usize,
    /// Genes whose dispersion hit the upper clamp.
    pub n_boundary_dispersion_high: usize,
    /// Genes that did not converge within `max_outer_iters`.
    pub n_nonconverged: usize,
    /// Fitted mean→dispersion trend, if `fit_dispersion_trend`.
    pub dispersion_trend: Option<DispersionTrend>,
    /// Empirical-Bayes prior variance (log scale), if shrinkage ran.
    pub dispersion_prior_var: Option<f64>,
    /// Wall-clock spent in the fit, if timed.
    pub elapsed_fit_ms: Option<f64>,
    /// Effective rayon thread count used, if recorded.
    pub rayon_threads_used: Option<usize>,
}

/// The result of a pseudobulk NB-GLM fit (spec §4.2). Vectors are length
/// `n_genes` unless noted; `beta` and `fitted_means` are gene-major
/// `[n_genes, n_features]` / `[n_genes, n_samples]` row-major when present.
#[derive(Debug, Clone)]
pub struct NbGlmResult {
    pub n_genes: usize,
    pub n_samples: usize,
    pub n_features: usize,
    /// `[n_genes, n_features]` mean-model coefficients if `store_coefficients`.
    pub beta: Option<Vec<f64>>,
    /// Shrunken NB dispersion `alpha` per gene.
    pub dispersion: Vec<f64>,
    /// Pre-shrinkage MLE dispersion `alpha` per gene.
    pub dispersion_mle: Vec<f64>,
    /// Contrast effect on log2 scale (`effect / ln 2`).
    pub log2_fold_change: Vec<f64>,
    /// Wald standard error for the contrast (natural-log scale).
    pub standard_error: Vec<f64>,
    /// Wald statistic (`effect / SE`).
    pub wald_stat: Vec<f64>,
    /// Two-sided Wald p-values.
    pub p_value: Vec<f64>,
    /// Benjamini–Hochberg adjusted p-values.
    pub p_adj: Vec<f64>,
    /// Normalized mean per gene (base mean).
    pub base_mean: Vec<f64>,
    /// Per-gene convergence flags.
    pub converged: Vec<bool>,
    /// Outer iterations used per gene.
    pub n_iter: Vec<u32>,
    /// `[n_genes, n_samples]` fitted means if `store_fitted_means`.
    pub fitted_means: Option<Vec<f64>>,
    pub diagnostics: NbGlmDiagnostics,
}
