//! Unit tests for the dispersion estimator, trend fit, and shrinkage (spec §11.1).

use super::*;
use crate::nb_glm::NbGlmOptions;

fn close(a: f64, b: f64, rel: f64, abs: f64) {
    let diff = (a - b).abs();
    let tol = abs.max(rel * a.abs().max(b.abs()));
    assert!(diff <= tol, "expected {a} ≈ {b} (|Δ|={diff} > tol={tol})");
}

// Intercept + treatment, 6 samples.
fn fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>, usize, usize) {
    let design = vec![
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0, //
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
    ];
    // Genuinely over-dispersed within each condition group so the Cox–Reid
    // optimum sits clearly in the interior (not pinned near min_disp).
    let mu = vec![10.0, 10.0, 10.0, 15.0, 15.0, 15.0];
    let y = vec![3.0, 10.0, 20.0, 6.0, 24.0, 12.0];
    (design, mu, y, 6, 2)
}

#[test]
fn cox_reid_mle_gradient_zero_and_matches_grid() {
    let (design, mu, y, ns, nf) = fixture();
    let opts = NbGlmOptions::default();
    let fit = fit_dispersion(&y, &mu, &design, ns, nf, 0.5, None, &opts);
    assert!(!fit.at_low && !fit.at_high, "optimum should be interior");

    // Gradient ≈ 0 at the returned alpha.
    let g = cox_reid_grad(&y, &mu, &design, ns, nf, opts.beta_ridge, fit.alpha, None);
    assert!(g.abs() < 1e-4, "gradient at MLE should be ~0, got {g}");

    // Matches a log-spaced grid-search argmax of the CR objective.
    let n = 400;
    let (lo, hi) = (opts.min_disp.ln(), opts.max_disp.ln());
    let mut best_t = lo;
    let mut best_v = f64::NEG_INFINITY;
    for i in 0..=n {
        let t = lo + (hi - lo) * (i as f64 / n as f64);
        let v = cox_reid_value(&y, &mu, &design, ns, nf, opts.beta_ridge, t.exp(), None);
        if v > best_v {
            best_v = v;
            best_t = t;
        }
    }
    close(fit.alpha.ln(), best_t, 0.0, 0.1); // grid resolution ~ (hi-lo)/n
}

#[test]
fn trend_fit_recovers_known_coefficients() {
    // alpha = a0 + a1/mu_bar with a0=0.1, a1=2.0 over a spread of base means.
    let (a0, a1) = (0.1, 2.0);
    let n = 200;
    let base_mean: Vec<f64> = (0..n).map(|i| 5.0 + (i as f64) * 2.5).collect();
    let alpha_mle: Vec<f64> = base_mean.iter().map(|&m| a0 + a1 / m).collect();
    let valid = vec![true; n];
    let opts = NbGlmOptions::default();
    let trend = fit_dispersion_trend(&base_mean, &alpha_mle, &valid, &opts).expect("trend");
    close(trend.a0, a0, 1e-3, 1e-4);
    close(trend.a1, a1, 1e-3, 1e-4);
}

#[test]
fn shrinkage_pulls_toward_target_and_respects_prior_strength() {
    let (design, mu, y, ns, nf) = fixture();
    let opts = NbGlmOptions::default();
    let mle = fit_dispersion(&y, &mu, &design, ns, nf, 0.5, None, &opts);

    // Target deliberately offset from the MLE.
    let target = (mle.alpha * 0.25).max(opts.min_disp);
    let log_target = target.ln();

    // Strong prior (tiny variance) pulls the estimate toward the target.
    let tight = DispPrior {
        log_alpha_trend: log_target,
        sigma_lr2: 1e-3,
    };
    let shr_tight = fit_dispersion(&y, &mu, &design, ns, nf, mle.alpha, Some(&tight), &opts);
    // Weak prior (huge variance) leaves it ~at the MLE.
    let weak = DispPrior {
        log_alpha_trend: log_target,
        sigma_lr2: 1e6,
    };
    let shr_weak = fit_dispersion(&y, &mu, &design, ns, nf, mle.alpha, Some(&weak), &opts);

    let d_tight = (shr_tight.alpha.ln() - log_target).abs();
    let d_mle = (mle.alpha.ln() - log_target).abs();
    assert!(
        d_tight < d_mle,
        "strong prior should move toward target: d_tight={d_tight} d_mle={d_mle}"
    );
    close(shr_weak.alpha, mle.alpha, 1e-2, 1e-3);
}

#[test]
fn trend_fit_trims_dispersion_outliers() {
    // Known trend alpha = a0 + a1/mu_bar for most genes, plus a few gross
    // dispersion outliers (1000× the trend). The DESeq2-style trim loop must drop
    // them so the recovered coefficients stay close to truth.
    let (a0, a1) = (0.1, 2.0);
    let n = 200;
    let base_mean: Vec<f64> = (0..n).map(|i| 5.0 + (i as f64) * 2.5).collect();
    let mut alpha_mle: Vec<f64> = base_mean.iter().map(|&m| a0 + a1 / m).collect();
    for &g in &[10usize, 50, 120, 175] {
        alpha_mle[g] = (a0 + a1 / base_mean[g]) * 1000.0; // ≫ 15× ⇒ must be trimmed
    }
    let valid = vec![true; n];
    let opts = NbGlmOptions::default();
    let trend = fit_dispersion_trend(&base_mean, &alpha_mle, &valid, &opts).expect("trend");
    close(trend.a0, a0, 5e-2, 1e-2);
    close(trend.a1, a1, 5e-2, 1e-2);
}

#[test]
fn estimate_prior_var_subtracts_sampling_variance() {
    // Non-zero scatter in the log-residuals: more of the expected sampling variance
    // `trigamma((m−p)/2)` is subtracted at small residual df than at large df, so
    // the prior variance is strictly smaller there (and both stay above the floor).
    let n = 60;
    let log_targets = vec![0.0; n];
    let alpha_mle: Vec<f64> = (0..n).map(|i| ((i as f64 - 30.0) * 0.1).exp()).collect();
    let valid = vec![true; n];
    let min_disp = NbGlmOptions::default().min_disp;
    // trigamma(2)≈0.645
    let pv_small = estimate_prior_var(&log_targets, &alpha_mle, &valid, 6, 2, min_disp).prior_var;
    // trigamma(99)≈0.005
    let pv_large = estimate_prior_var(&log_targets, &alpha_mle, &valid, 200, 2, min_disp).prior_var;
    assert!(
        pv_large > pv_small,
        "smaller residual df subtracts more sampling variance: small={pv_small} large={pv_large}"
    );
    assert!(
        pv_small > MIN_PRIOR_VAR,
        "the scatter should clear the floor so the subtraction is observable: {pv_small}"
    );
}

#[test]
fn estimate_prior_var_has_floor() {
    // Identical residuals ⇒ MAD = 0 ⇒ falls back to the minimum prior var.
    let log_targets = vec![0.0; 10];
    let alpha_mle = vec![1.0; 10]; // ln(1) - 0 = 0 residual
    let valid = vec![true; 10];
    // m=6, p=2 ⇒ trigamma((m−p)/2)=trigamma(2)>0; with MAD=0 the prior var floors
    // at MIN_PRIOR_VAR rather than going negative.
    let min_disp = NbGlmOptions::default().min_disp;
    let pv = estimate_prior_var(&log_targets, &alpha_mle, &valid, 6, 2, min_disp).prior_var;
    assert!(pv >= MIN_PRIOR_VAR - 1e-12);
}

#[test]
fn moments_dispersion_clamped_positive() {
    let opts = NbGlmOptions::default();
    let counts = vec![5.0, 8.0, 3.0, 12.0, 6.0];
    let sf = vec![1.0; 5];
    let a = moments_dispersion(&counts, &sf, &opts);
    assert!(a >= opts.min_disp && a <= opts.max_disp);
}

// ── Review §7.15 follow-up — the median convention is a classification boundary ──

#[test]
fn median_of_sorted_averages_the_two_middles_at_even_length() {
    // Odd: the middle element, same as the upper-median rule this replaced.
    assert_eq!(median_of_sorted(&[1.0, 2.0, 3.0]), 2.0);
    assert_eq!(median_of_sorted(&[5.0]), 5.0);
    // Even: the *average*, which the upper-median rule got wrong.
    assert_eq!(median_of_sorted(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    assert_eq!(median_of_sorted(&[1.0, 4.0]), 2.5);
    // Degenerate.
    assert_eq!(median_of_sorted(&[]), 0.0);
}

/// The even-length boundary case, end to end through `estimate_prior_var`.
///
/// Four usable genes with log-residuals `[0.3304, 0.3456, 0.8216, 5.6631]`
/// about a zero target. Measured against pydeseq2 0.5.4's
/// `mean_absolute_deviation` (`np.median(|x - np.median(x)|) / norm.ppf(0.75)`):
///
/// | rule | scaled MAD | `2·√(squared_logres)` | flags |
/// |---|---|---|---|
/// | `values[len/2]` (upper median, pre-fix) | 0.72825 | 1.45651 | 5.6631 only |
/// | `np.median` (pydeseq2, and now SCX) | 0.36413 | 0.72825 | 0.8216 **and** 5.6631 |
///
/// So 0.8216 sits *between* the two thresholds: it is the gene the old
/// convention silently kept shrinking. An odd-length fixture cannot show this,
/// and neither can the 24-gene pydeseq2 reference — there both conventions agree
/// on zero outliers, which is why that test alone was not enough.
#[test]
fn an_even_length_residual_set_classifies_the_way_pydeseq2_does() {
    let log_targets = vec![0.0; 4];
    // exp(residual), since the residual is `ln(alpha_mle) - log_target`.
    let resid = [0.3304_f64, 0.3456, 0.8216, 5.6631];
    let alpha_mle: Vec<f64> = resid.iter().map(|r| r.exp()).collect();
    let valid = vec![true; 4];

    // n_samples <= n_features suppresses the trigamma subtraction, so
    // `squared_logres` is observable directly rather than through the floor.
    let fit = estimate_prior_var(&log_targets, &alpha_mle, &valid, 2, 2, 1e-8);
    let scaled_mad = fit.squared_logres.sqrt();
    assert!(
        (scaled_mad - 0.364_127_104_864_975_8).abs() < 1e-9,
        "scaled MAD {scaled_mad} != pydeseq2's 0.3641271048649758 \
         (the upper-median rule gives 0.7282531199999999)"
    );

    let mask = dispersion_outlier_mask(&log_targets, &alpha_mle, &valid, fit.squared_logres, 2.0);
    assert_eq!(
        mask,
        vec![false, false, true, true],
        "pydeseq2 flags both 0.8216 and 5.6631; the upper-median rule flagged only 5.6631"
    );
}

/// `above_min_disp` runs on every shrink, not only when the carve-out is on.
///
/// Pinned because the docs now say so out loud, and because the original wording
/// said the opposite: `disp_outlier_sd=None` was described as an escape hatch
/// back to pre-0.20 numbers, which it is not — the knob gates the carve-out and
/// nothing else, while this filter moves `prior_var` for *every* gene on a panel
/// with a clamped tail. Found by Cursor Agent - Grok 4.6 High.
#[test]
fn the_residual_filter_excludes_min_disp_pinned_genes() {
    const MIN_DISP: f64 = 1e-8;
    // Six genes with modest scatter about the target, plus four pinned at the
    // lower clamp. The pinned four sit ~18 log units below and would dominate
    // any MAD that included them.
    let spread = [-0.4_f64, -0.2, -0.05, 0.05, 0.2, 0.4];
    let mut alpha_mle: Vec<f64> = spread.iter().map(|r| r.exp()).collect();
    alpha_mle.extend(std::iter::repeat_n(MIN_DISP, 4));
    let n = alpha_mle.len();
    let log_targets = vec![0.0; n];
    let valid = vec![true; n];

    let filtered = estimate_prior_var(&log_targets, &alpha_mle, &valid, 2, 2, MIN_DISP);

    // Premise: the pinned genes really are far enough out to matter. Recompute
    // the same statistic over *all* genes and check the two disagree sharply.
    let unfiltered = estimate_prior_var(&log_targets, &alpha_mle, &valid, 2, 2, 0.0);
    // Measured on this fixture: 0.08792 filtered vs 0.79132 unfiltered, a 9.0x
    // inflation from four pinned genes out of ten. The bar is set from that,
    // not guessed — a first draft asserting 10x failed on the real 9.0x.
    assert!(
        unfiltered.squared_logres > 5.0 * filtered.squared_logres,
        "premise: including the clamped tail should inflate the MAD; \
         filtered={} unfiltered={}",
        filtered.squared_logres,
        unfiltered.squared_logres
    );

    // And the filtered statistic is the one computed from the six real genes.
    let only_real: Vec<f64> = spread.iter().map(|r| r.exp()).collect();
    let expected = estimate_prior_var(&[0.0; 6], &only_real, &[true; 6], 2, 2, MIN_DISP);
    assert_eq!(
        filtered.squared_logres.to_bits(),
        expected.squared_logres.to_bits(),
        "the filter must leave exactly the above-floor genes: {} vs {}",
        filtered.squared_logres,
        expected.squared_logres
    );

    // The outlier threshold is what that inflation would have broken. It scales
    // as √(squared_logres), so a 9.0x inflation moves the bar 3.0x further out
    // — which is the sense in which the filter is load-bearing for the gate and
    // not merely for the prior width.
    let thresh_filtered = 2.0 * filtered.squared_logres.sqrt();
    let thresh_unfiltered = 2.0 * unfiltered.squared_logres.sqrt();
    assert!(
        thresh_unfiltered > 2.0 * thresh_filtered,
        "the clamped tail should push the outlier threshold far out: \
         {thresh_filtered} -> {thresh_unfiltered}"
    );
}

/// `squared_logres` on a small pool is the **raw** MAD², not the prior floor.
///
/// The two fields are not interchangeable: `prior_var` is floored at
/// `MIN_PRIOR_VAR = 0.25` by design, while `squared_logres` is what the outlier
/// threshold takes the square root of. Substituting the floor turns a threshold
/// pydeseq2 computes as `2·√0 = 0` into `2·√0.25 = 1`, so a gene pydeseq2
/// exempts from shrinkage gets shrunk instead. pydeseq2 0.5.4 runs
/// `mean_absolute_deviation` on one or two residuals like any other pool.
/// Reachable after gene projection or when `above_min_disp` leaves few genes.
/// Found by codex - gpt-5.6-sol.
#[test]
fn a_small_residual_pool_reports_its_real_mad_not_the_prior_floor() {
    const MIN_DISP: f64 = 1e-8;
    // One usable gene: residual 0.2 about a zero target, so MAD = 0.
    let fit_one = estimate_prior_var(&[0.0], &[0.2_f64.exp()], &[true], 2, 2, MIN_DISP);
    assert_eq!(
        fit_one.squared_logres, 0.0,
        "a single residual has zero deviation; pydeseq2 reports 0, not the floor"
    );
    assert_eq!(
        fit_one.prior_var, MIN_PRIOR_VAR,
        "the *prior* is still floored — that part was always right"
    );
    // And the classifier follows: with a zero threshold the gene is an outlier,
    // which is pydeseq2's answer. Under the old `len < 3` branch the threshold
    // was `2·√0.25 = 1` and 0.2 was shrunk.
    let mask = dispersion_outlier_mask(
        &[0.0],
        &[0.2_f64.exp()],
        &[true],
        fit_one.squared_logres,
        2.0,
    );
    assert_eq!(mask, vec![true]);

    // Two genes: a real, non-degenerate MAD.
    let resid = [0.1_f64, 0.5];
    let alphas: Vec<f64> = resid.iter().map(|r| r.exp()).collect();
    let fit_two = estimate_prior_var(&[0.0, 0.0], &alphas, &[true, true], 2, 2, MIN_DISP);
    let expected = (1.4826_f64 * 0.2).powi(2); // scaled MAD of [0.1, 0.5] about 0.3
    assert!(
        (fit_two.squared_logres - expected).abs() < 1e-3,
        "two-gene MAD² {} should be ≈ {expected}",
        fit_two.squared_logres
    );

    // Empty stays explicit, and still reports 0 rather than the floor.
    let fit_none = estimate_prior_var(&[], &[], &[], 2, 2, MIN_DISP);
    assert_eq!(fit_none.squared_logres, 0.0);
    assert_eq!(fit_none.prior_var, MIN_PRIOR_VAR);
}

/// A non-finite residual never reaches the sort.
///
/// `partial_cmp` returns `None` on a `NaN` and the comparator maps that to
/// `Equal`, so the sort silently accepts it and the median is whatever the
/// unstable ordering left in the middle.
#[test]
fn non_finite_residuals_are_dropped_before_the_median() {
    // A `-inf` log target (what `min_disp = 0` with a zero trend would produce)
    // makes gene 1's residual `+inf`; gene 2's `NaN`.
    let log_targets = vec![0.0, f64::NEG_INFINITY, f64::NEG_INFINITY];
    let alpha_mle = vec![0.2_f64.exp(), 1.0, 0.0];
    let valid = vec![true; 3];
    let fit = estimate_prior_var(&log_targets, &alpha_mle, &valid, 2, 2, 1e-8);
    assert!(
        fit.squared_logres.is_finite() && fit.prior_var.is_finite(),
        "a non-finite residual must not reach the statistic: {fit:?}"
    );
    // Only gene 0 survives the filter, so this is the single-residual case.
    assert_eq!(fit.squared_logres, 0.0);
}

/// The no-trend shrinkage target uses the averaging median.
///
/// The MAD moved onto numpy's median for pydeseq2 parity; this fallback kept
/// `values[len / 2]` in two copied blocks, so a run with
/// `fit_dispersion_trend = false` shrank toward a different centre than the
/// prior width was calibrated against. No test could reach it while the
/// computation was inline in `mod.rs` and `gpu.rs`, which is why it is a
/// function now. Flagged by Cursor Agent - Grok 4.6 High and
/// Antigravity - Gemini 3.8 Flash.
#[test]
fn the_no_trend_shrinkage_target_uses_the_averaging_median() {
    let opts = NbGlmOptions::default();
    // Even-length pool — the only case where the two conventions differ.
    let alpha_mle = vec![0.1_f64, 0.2, 0.4, 0.8];
    let valid = vec![true; 4];
    let base_means = vec![100.0; 4];

    let targets = shrinkage_log_targets(None, &base_means, &alpha_mle, &valid, &opts);

    let averaging = 0.3_f64; // (0.2 + 0.4) / 2
    let upper = 0.4_f64; // values[len / 2]
    assert!(
        (averaging - upper).abs() > 1e-9,
        "premise: the two conventions must differ on this pool"
    );
    assert!(
        targets.iter().all(|&t| (t - averaging.ln()).abs() < 1e-12),
        "target should be ln({averaging}) = {}, got {:?}",
        averaging.ln(),
        &targets[..2]
    );

    // Odd length: the two rules agree, and the target is the middle element.
    let odd = vec![0.1_f64, 0.2, 0.4];
    let t_odd = shrinkage_log_targets(None, &base_means[..3], &odd, &valid[..3], &opts);
    assert!(t_odd.iter().all(|&t| (t - 0.2_f64.ln()).abs() < 1e-12));

    // The `min_disp` clamp still applies below the floor.
    let tiny = vec![1e-30_f64, 1e-30];
    let t_tiny = shrinkage_log_targets(None, &base_means[..2], &tiny, &valid[..2], &opts);
    assert!(t_tiny
        .iter()
        .all(|&t| (t - opts.min_disp.ln()).abs() < 1e-12));

    // With a trend, the target is the trend value and the median is unused.
    let trend = DispersionTrend { a0: 0.05, a1: 4.0 };
    let t_trend = shrinkage_log_targets(Some(&trend), &base_means, &alpha_mle, &valid, &opts);
    assert!((t_trend[0] - trend.eval(100.0).ln()).abs() < 1e-12);
}
