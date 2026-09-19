//! Shared UMAP math helpers.
//!
//! Pure-math functions used by both `scx-accel` (CPU UMAP) and `scx-gpu`
//! (GPU UMAP). Lives in `scx-sparse` because both crates already depend on it
//! and these functions have no format/GPU/engine dependencies.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::Uniform;

// ---------------------------------------------------------------------------
// find_ab_params
// ---------------------------------------------------------------------------

/// Find the `(a, b)` parameters for the UMAP membership-strength curve.
///
/// The curve is: `1 / (1 + a · d^{2b})`.
/// Fit to the piecewise target:
///   `f(d) = 1` if `d ≤ min_dist`, else `exp(-(d - min_dist) / spread)`.
///
/// Gauss-Newton least-squares on the residual `f(x; a, b) - y`, seeded at
/// `(a, b) = (1.93, 0.79)` — the class-attribute placeholders used by
/// umap-learn's `UMAP` constructor before `curve_fit` runs (not the fitted
/// result, which is ≈ (1.577, 0.895) for `spread=1.0, min_dist=0.1`). The
/// seed is close enough to the global basin that GN converges there in
/// ≲ 20 iterations. Backtracking line search accepts a step only if the
/// residual sum of squares decreases. Target precision is ~1e-4 of scipy
/// `curve_fit` / umap-learn.
pub fn find_ab_params(spread: f64, min_dist: f64) -> (f64, f64) {
    // Match umap-learn's sampling exactly: `np.linspace(0, 3*spread, 300)`
    // with a strict `x < min_dist` branch. Including x=0 anchors the curve
    // at pred=1; the Jacobian contribution at x=0 is skipped below to
    // avoid ln(0) blowing up `db`.
    let n_points = 300;
    let x_max = 3.0 * spread;
    let xs: Vec<f64> = (0..n_points)
        .map(|i| x_max * i as f64 / (n_points - 1) as f64)
        .collect();
    let ys: Vec<f64> = xs
        .iter()
        .map(|&x| {
            if x < min_dist {
                1.0
            } else {
                (-(x - min_dist) / spread).exp()
            }
        })
        .collect();

    let sse = |a: f64, b: f64| -> f64 {
        xs.iter()
            .zip(ys.iter())
            .map(|(&x, &y)| {
                let pred = 1.0 / (1.0 + a * x.powf(2.0 * b));
                let r = pred - y;
                r * r
            })
            .sum::<f64>()
    };

    // Seed from umap-learn's class-default `_a`/`_b` placeholders. These
    // are NOT the `curve_fit` result; they sit near the global-optimum
    // basin (fitted ≈ 1.577, 0.895 at spread=1, min_dist=0.1) and
    // Gauss-Newton reliably converges from here within ≲ 20 iterations.
    let mut a = 1.93_f64;
    let mut b = 0.79_f64;
    let mut err = sse(a, b);

    const MAX_ITER: usize = 64;
    const STEP_TOL: f64 = 1e-10;
    const GRAD_TOL: f64 = 1e-14;

    for _ in 0..MAX_ITER {
        // Accumulate J^T J (2×2, symmetric) and J^T r (2×1).
        let mut jtj_aa = 0.0;
        let mut jtj_ab = 0.0;
        let mut jtj_bb = 0.0;
        let mut jtr_a = 0.0;
        let mut jtr_b = 0.0;

        for (&x, &y) in xs.iter().zip(ys.iter()) {
            if x <= 0.0 {
                // Residual is 0 here (pred=1=y when b>0, xb=0); Jacobian
                // has a 0·ln(0) factor that is 0 by L'Hôpital but NaN in
                // floating point. Skip.
                continue;
            }
            let xb = x.powf(2.0 * b);
            let denom = 1.0 + a * xb;
            let pred = 1.0 / denom;
            let r = pred - y;
            let common = -xb / (denom * denom);
            let da = common;
            // d/db (a · x^(2b)) = 2 a · x^(2b) · ln(x)
            let db = common * a * 2.0 * x.ln();

            jtj_aa += da * da;
            jtj_ab += da * db;
            jtj_bb += db * db;
            jtr_a += da * r;
            jtr_b += db * r;
        }

        if jtr_a.abs() < GRAD_TOL && jtr_b.abs() < GRAD_TOL {
            break;
        }

        // Solve (J^T J) · δ = -J^T r using the closed-form 2×2 inverse.
        let det = jtj_aa * jtj_bb - jtj_ab * jtj_ab;
        if det.abs() < 1e-20 {
            break;
        }
        let inv = 1.0 / det;
        let delta_a = inv * (jtj_bb * (-jtr_a) - jtj_ab * (-jtr_b));
        let delta_b = inv * (-jtj_ab * (-jtr_a) + jtj_aa * (-jtr_b));

        // Backtracking line search: halve the step until SSE decreases or
        // the step shrinks below tolerance.
        let mut alpha = 1.0_f64;
        let mut accepted = false;
        for _ in 0..20 {
            let trial_a = (a + alpha * delta_a).max(1e-4);
            let trial_b = (b + alpha * delta_b).max(1e-4);
            let trial_err = sse(trial_a, trial_b);
            if trial_err < err {
                a = trial_a;
                b = trial_b;
                err = trial_err;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            break;
        }
        if (alpha * delta_a).abs() < STEP_TOL && (alpha * delta_b).abs() < STEP_TOL {
            break;
        }
    }

    (a, b)
}

// ---------------------------------------------------------------------------
// compute_epochs_per_sample
// ---------------------------------------------------------------------------

/// Compute per-edge sampling schedule for UMAP SGD.
///
/// Higher-weight edges are sampled more frequently. The max-weight edge
/// is sampled every epoch; lower-weight edges less often.
pub fn compute_epochs_per_sample(weights: &[f64], n_epochs: usize) -> Vec<f64> {
    let max_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_weight <= 0.0 {
        return vec![n_epochs as f64 + 1.0; weights.len()];
    }

    weights
        .iter()
        .map(|&w| {
            if w <= 0.0 {
                n_epochs as f64 + 1.0 // never sample
            } else {
                n_epochs as f64 / (w / max_weight * n_epochs as f64).max(1.0)
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// random_init
// ---------------------------------------------------------------------------

/// Random initialization for UMAP: uniform over `[-10, 10]` (f64).
///
/// Matches umap-learn, whose `init="random"` draws
/// `random_state.uniform(low=-10.0, high=10.0, size=(n, d))` — the same ±10
/// span the spectral path is expanded to.
///
/// This used to be `N(0, 1e-4) × 10`, an effective std of ~1e-3: about 10⁴ too
/// small (review §7.11, the same defect as the spectral scaling). At that scale
/// the layout began far below the SGD's `clip_val = 4.0` and the optimizer had to
/// spend its early epochs merely inflating the cloud.
pub fn random_init_f64(n_obs: usize, n_components: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let uniform = Uniform::new(-10.0_f64, 10.0);
    (0..n_obs * n_components)
        .map(|_| rng.sample(uniform))
        .collect()
}

/// Random initialization for UMAP: uniform over `[-10, 10]` (f32).
///
/// Same distribution as [`random_init_f64`] but in single precision for GPU use.
pub fn random_init_f32(n_obs: usize, n_components: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let uniform = Uniform::new(-10.0_f32, 10.0);
    (0..n_obs * n_components)
        .map(|_| rng.sample(uniform))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_ab_params_default() {
        let (a, b) = find_ab_params(1.0, 0.1);
        // umap-learn's scipy.optimize.curve_fit gives a≈1.929, b≈0.7915.
        assert!(
            a > 1.0 && a < 2.5,
            "a = {a}, expected in [1.0, 2.5] (umap-learn: ~1.93)"
        );
        assert!(
            b > 0.5 && b < 1.2,
            "b = {b}, expected in [0.5, 1.2] (umap-learn: ~0.79)"
        );
    }

    #[test]
    fn test_find_ab_params_various() {
        let cases = [(1.0, 0.1), (1.0, 0.25), (1.0, 0.5), (0.5, 0.1), (2.0, 0.1)];
        for (spread, min_dist) in cases {
            let (a, b) = find_ab_params(spread, min_dist);
            assert!(
                a > 0.2 && a < 9.8,
                "a={a} out of safe interior range for spread={spread}, min_dist={min_dist}"
            );
            assert!(
                b > 0.2 && b < 3.9,
                "b={b} out of safe interior range for spread={spread}, min_dist={min_dist}"
            );
            let pred_at_zero = 1.0 / (1.0 + a * (0.001_f64).powf(2.0 * b));
            assert!(
                pred_at_zero > 0.99,
                "pred(0) = {pred_at_zero} for spread={spread}, min_dist={min_dist}"
            );
        }
    }

    #[test]
    fn test_find_ab_params_extreme() {
        let (a, b) = find_ab_params(0.01, 0.001);
        assert!(a.is_finite(), "a should be finite, got {a}");
        assert!(b.is_finite(), "b should be finite, got {b}");
        assert!(a > 0.0, "a should be positive, got {a}");
        assert!(b > 0.0, "b should be positive, got {b}");
    }

    #[test]
    fn test_epochs_per_sample() {
        let weights = vec![1.0, 0.5, 0.25, 0.1];
        let schedule = compute_epochs_per_sample(&weights, 200);
        assert!((schedule[0] - 1.0).abs() < 1e-6);
        assert!(schedule[1] > schedule[0]);
        assert!(schedule[2] > schedule[1]);
        assert!(schedule[3] > schedule[2]);
    }

    #[test]
    fn test_epochs_per_sample_zero_weights() {
        let weights = vec![0.0, 0.0];
        let schedule = compute_epochs_per_sample(&weights, 100);
        assert_eq!(schedule, vec![101.0, 101.0]);
    }

    #[test]
    fn test_random_init_f64_deterministic() {
        let init1 = random_init_f64(10, 2, 42);
        let init2 = random_init_f64(10, 2, 42);
        assert_eq!(init1, init2, "same seed should give same init");
    }

    #[test]
    fn test_random_init_f32_deterministic() {
        let init1 = random_init_f32(10, 2, 42);
        let init2 = random_init_f32(10, 2, 42);
        assert_eq!(init1, init2, "same seed should give same init");
    }

    #[test]
    fn test_random_init_f64_size() {
        let init = random_init_f64(100, 3, 0);
        assert_eq!(init.len(), 300);
    }

    #[test]
    fn test_random_init_f32_size() {
        let init = random_init_f32(100, 3, 0);
        assert_eq!(init.len(), 300);
    }
}
