//! Shared UMAP math helpers.
//!
//! Pure-math functions used by both `scx-accel` (CPU UMAP) and `scx-gpu`
//! (GPU UMAP). Lives in `scx-sparse` because both crates already depend on it
//! and these functions have no format/GPU/engine dependencies.

use rand::prelude::*;
use rand_distr::Normal;

// ---------------------------------------------------------------------------
// find_ab_params
// ---------------------------------------------------------------------------

/// Find the `(a, b)` parameters for the UMAP membership-strength curve.
///
/// The curve is: `1 / (1 + a · d^{2b})`.
/// Fit to the piecewise target:
///   `f(d) = 1` if `d ≤ min_dist`, else `exp(-(d - min_dist) / spread)`.
///
/// Uses a coarse grid search followed by local refinement, replicating
/// umap-learn's `scipy.optimize.curve_fit` result closely enough for
/// SGD embedding quality.
pub fn find_ab_params(spread: f64, min_dist: f64) -> (f64, f64) {
    // Grid boundary constants for the coarse search.
    // Default UMAP params (spread=1.0, min_dist=0.1) produce a≈1.93, b≈0.79,
    // well within these ranges. Unusual param combos could hit the edges.
    const A_STEP: f64 = 0.1;
    const A_STEPS: usize = 100;
    const B_STEP: f64 = 0.1;
    const B_STEPS: usize = 40;
    // Derived bounds (A: 0.1..10.0, B: 0.1..4.0)
    let a_start = A_STEP;
    let b_start = B_STEP;
    let a_end = A_STEP * A_STEPS as f64;
    let b_end = B_STEP * B_STEPS as f64;

    let n_points = 300;
    let x_max = 3.0 * spread;
    let xs: Vec<f64> = (0..n_points)
        .map(|i| (i as f64 + 0.5) / n_points as f64 * x_max)
        .collect();
    let ys: Vec<f64> = xs
        .iter()
        .map(|&x| {
            if x <= min_dist {
                1.0
            } else {
                (-(x - min_dist) / spread).exp()
            }
        })
        .collect();

    // Least-squares error for a candidate (a, b) pair.
    let compute_error = |a: f64, b: f64| -> f64 {
        xs.iter()
            .zip(ys.iter())
            .map(|(&x, &y)| {
                let pred = 1.0 / (1.0 + a * x.powf(2.0 * b));
                (pred - y) * (pred - y)
            })
            .sum()
    };

    let mut best_a = 1.0_f64;
    let mut best_b = 1.0_f64;
    let mut best_err = f64::MAX;

    // Coarse grid
    for a_idx in 1..=A_STEPS {
        let a = a_idx as f64 * A_STEP;
        for b_idx in 1..=B_STEPS {
            let b = b_idx as f64 * B_STEP;
            let err = compute_error(a, b);
            if err < best_err {
                best_err = err;
                best_a = a;
                best_b = b;
            }
        }
    }

    // Fine refinement around best
    let refine_range = 0.1;
    let refine_steps = 20;
    let a_lo = (best_a - refine_range).max(0.01);
    let a_hi = best_a + refine_range;
    let b_lo = (best_b - refine_range).max(0.01);
    let b_hi = best_b + refine_range;

    for a_idx in 0..=refine_steps {
        let a = a_lo + (a_hi - a_lo) * a_idx as f64 / refine_steps as f64;
        for b_idx in 0..=refine_steps {
            let b = b_lo + (b_hi - b_lo) * b_idx as f64 / refine_steps as f64;
            let err = compute_error(a, b);
            if err < best_err {
                best_err = err;
                best_a = a;
                best_b = b;
            }
        }
    }

    // Boundary detection: warn if the optimum is near a grid edge,
    // which indicates the true optimum may lie outside the search range.
    if (best_a - a_start).abs() < A_STEP || (a_end - best_a).abs() < A_STEP {
        eprintln!(
            "scx WARN: UMAP find_ab_params: optimal `a` ({:.4}) is near search boundary \
             [{}, {}] for spread={}, min_dist={}. Results may be inaccurate.",
            best_a, a_start, a_end, spread, min_dist
        );
    }
    if (best_b - b_start).abs() < B_STEP || (b_end - best_b).abs() < B_STEP {
        eprintln!(
            "scx WARN: UMAP find_ab_params: optimal `b` ({:.4}) is near search boundary \
             [{}, {}] for spread={}, min_dist={}. Results may be inaccurate.",
            best_b, b_start, b_end, spread, min_dist
        );
    }

    (best_a, best_b)
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

/// Random initialization for UMAP: small Gaussian noise (f64).
///
/// Produces an `n_obs × n_components` embedding with effective std dev ≈ 1e-3.
pub fn random_init_f64(n_obs: usize, n_components: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f64, 1e-4).unwrap();
    (0..n_obs * n_components)
        .map(|_| rng.sample(normal) * 10.0)
        .collect()
}

/// Random initialization for UMAP: small Gaussian noise (f32).
///
/// Produces an `n_obs × n_components` embedding with effective std dev ≈ 1e-3.
/// Same distribution as [`random_init_f64`] but in single precision for GPU use.
pub fn random_init_f32(n_obs: usize, n_components: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f32, 1e-4).unwrap();
    (0..n_obs * n_components)
        .map(|_| rng.sample(normal) * 10.0)
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
