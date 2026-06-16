//! Unit tests for Cook's distance, masked BH, and independent filtering (spec §19).

use super::*;

fn close(a: f64, b: f64, rel: f64, abs: f64) {
    let diff = (a - b).abs();
    let tol = abs.max(rel * a.abs().max(b.abs()));
    assert!(diff <= tol, "expected {a} ≈ {b} (|Δ|={diff} > tol={tol})");
}

#[test]
fn cooks_distance_hand_computed_intercept_only() {
    // Intercept-only design (n_features=1): M = ΣW = 15 (W=5 each), cov = 1/15.
    // Leverage h_s = W·(1·cov·1) = 5/15 = 1/3. Sample 2 is the outlier:
    // pearson² = (30−10)²/(10+0.1·100) = 400/20 = 20, so
    // D = (20/1)·((1/3)/(2/3)²) = 20·(3/4) = 15.
    let cov = vec![1.0 / 15.0];
    let mu = vec![10.0, 10.0, 10.0];
    let design = vec![1.0, 1.0, 1.0];
    let counts = vec![10.0, 10.0, 30.0];
    let d = cooks_distance(&cov, &mu, &design, &counts, 0.1, 3, 1);
    close(d, 15.0, 1e-12, 1e-12);
}

#[test]
fn cooks_distance_zero_for_perfect_fit() {
    // y == mu everywhere ⇒ zero residuals ⇒ Cook's distance 0.
    let cov = vec![1.0 / 15.0];
    let mu = vec![10.0, 10.0, 10.0];
    let design = vec![1.0, 1.0, 1.0];
    let counts = vec![10.0, 10.0, 10.0];
    let d = cooks_distance(&cov, &mu, &design, &counts, 0.1, 3, 1);
    assert_eq!(d, 0.0);
}

#[test]
fn cooks_cutoff_is_f_quantile() {
    // qf(0.99, 2, 10) ≈ 7.5594.
    close(cooks_cutoff(2, 10), 7.559_43, 1e-3, 1e-3);
}

#[test]
fn benjamini_hochberg_masked_skips_nan() {
    // Finite p = [0.01, 0.04, 0.5] (n=3): BH ⇒ [0.03, 0.06, 0.5]; NaN passes through.
    let pvals = vec![0.01, f64::NAN, 0.04, 0.5];
    let adj = benjamini_hochberg_masked(&pvals);
    close(adj[0], 0.03, 1e-12, 1e-12);
    assert!(adj[1].is_nan(), "NaN p-value must stay NaN");
    close(adj[2], 0.06, 1e-12, 1e-12);
    close(adj[3], 0.5, 1e-12, 1e-12);
}

#[test]
fn benjamini_hochberg_masked_all_nan() {
    let adj = benjamini_hochberg_masked(&[f64::NAN, f64::NAN]);
    assert!(adj.iter().all(|v| v.is_nan()));
}

#[test]
fn independent_filter_no_op_on_uniform_base_mean() {
    // Near-uniform base mean ⇒ filtering cannot improve rejections ⇒ no-op:
    // every gene keeps a finite p_adj (this is what protects the parity gate).
    let n = 60;
    let base_mean = vec![2500.0_f64; n];
    // A spread of p-values (some significant) on a constant base mean.
    let p_value: Vec<f64> = (0..n).map(|i| if i < 20 { 0.001 } else { 0.6 }).collect();
    let (p_adj, threshold, n_filtered) = independent_filter(&base_mean, &p_value, 0.1);
    assert!(threshold.is_none(), "uniform base mean should not filter");
    assert_eq!(n_filtered, 0);
    assert!(
        p_adj.iter().all(|v| v.is_finite()),
        "no gene should be NaN'd by a no-op filter"
    );
}

#[test]
fn independent_filter_fires_when_signal_tracks_base_mean() {
    // Low-base-mean genes are null (p=0.8); high-base-mean genes are borderline
    // (p=0.04). At full n the borderline genes miss the 0.05 cutoff; removing the
    // low-base-mean nulls lets them clear it ⇒ filtering strictly increases
    // rejections, so it fires and NaNs the low-base-mean genes' p_adj.
    let n = 50;
    let base_mean: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let p_value: Vec<f64> = (0..n).map(|i| if i < 25 { 0.8 } else { 0.04 }).collect();
    let (p_adj, threshold, n_filtered) = independent_filter(&base_mean, &p_value, 0.05);
    assert!(threshold.is_some(), "filtering should fire");
    assert!(
        n_filtered > 0,
        "some low-base-mean genes should be filtered"
    );
    assert!(
        p_adj[0].is_nan(),
        "the lowest-base-mean gene should be filtered (NaN p_adj)"
    );
    assert!(
        p_adj[n - 1].is_finite(),
        "the highest-base-mean gene should be retained"
    );
}

#[test]
fn independent_filter_too_few_genes_is_no_op() {
    let base_mean = vec![1.0, 2.0, 3.0];
    let p_value = vec![0.01, 0.2, 0.5];
    let (p_adj, threshold, n_filtered) = independent_filter(&base_mean, &p_value, 0.1);
    assert!(threshold.is_none());
    assert_eq!(n_filtered, 0);
    assert!(p_adj.iter().all(|v| v.is_finite()));
}

#[test]
fn independent_filter_is_thread_count_invariant() {
    // The 50-cutoff BH sweep is rayon-parallel; assert the output is bit-identical
    // regardless of pool size (order-preserving `collect`, integer rejection
    // counts, deterministic downstream selection). Locks the "byte-identical"
    // claim against a future order-dependent reduction.
    let n = 3000;
    let base_mean: Vec<f64> = (0..n)
        .map(|i| 1.0 + (i as f64 * 0.013).sin().abs() * 800.0)
        .collect();
    // A signal: low p-values concentrated at high base mean → filtering helps.
    let p_value: Vec<f64> = (0..n)
        .map(|i| {
            if base_mean[i] > 400.0 {
                ((i * 7 + 1) % 50) as f64 / 5000.0
            } else {
                0.2 + ((i * 13 + 3) % 800) as f64 / 1000.0
            }
        })
        .collect();
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| independent_filter(&base_mean, &p_value, 0.1))
    };
    let (adj1, thr1, nf1) = run(1);
    let (adj8, thr8, nf8) = run(8);
    assert_eq!(nf1, nf8, "n_filtered differs by thread count");
    assert_eq!(
        thr1.map(f64::to_bits),
        thr8.map(f64::to_bits),
        "threshold differs by thread count"
    );
    assert_eq!(adj1.len(), adj8.len());
    for (i, (a, b)) in adj1.iter().zip(&adj8).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "p_adj[{i}] differs by thread count"
        );
    }
}
