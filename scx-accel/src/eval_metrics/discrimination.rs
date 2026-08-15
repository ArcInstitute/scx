//! Discrimination score for perturbation prediction evaluation.
//!
//! Computes how well each predicted perturbation effect ranks among all real
//! effects by pairwise distance. For each perturbation, the predicted effect
//! vector is compared to all real effect vectors using L1/L2/cosine distance;
//! the rank of the correct perturbation is normalized to `[1/P, 1.0]`,
//! matching the cell-eval reference (`score = 1 - rank / P`).
//!
//! Key features:
//! - Per-perturbation gene exclusion (`exclude_target_gene`): removes the
//!   column for each gene whose name matches the perturbation name, preventing
//!   trivially high scores from knockdown-gene dominance.
//! - Parallelized across perturbations with rayon.
//! - Uses the shared distance kernels from `distances.rs`.

use super::distances::point_distance;
use super::DistanceMetric;
use rayon::prelude::*;

/// Result of discrimination score computation.
#[derive(Debug, Clone)]
pub struct DiscriminationResult {
    /// Per-perturbation normalized rank scores.
    ///
    /// Range is `[1/P, 1.0]` where P = number of perturbations:
    /// - 1.0 = rank 0 (predicted effect is closest to correct real effect)
    /// - 1/P = rank P−1 (predicted effect is furthest from correct real effect)
    ///
    /// This matches cell-eval's convention: `score = 1 - rank / P`.
    pub scores: Vec<f64>,
    /// Perturbation names in order (excluding control).
    pub pert_names: Vec<String>,
}

/// Compute discrimination scores for all perturbations.
///
/// # Algorithm
///
/// 1. Input: `real_effects[P, G]` and `pred_effects[P, G]` — perturbation
///    effects (pseudobulk means minus control), already computed.
/// 2. For each perturbation `p`:
///    - If `exclude_target_gene` and `gene_names` is provided, build a mask
///      excluding the gene column whose name matches `pert_names[p]`.
///    - Compute distances from `pred_effects[p]` to all `real_effects[i]`
///      using the masked gene set.
///    - Sort distances ascendingly, find rank of `p` in sorted order.
///    - `score[p] = 1 - rank / P`
///
/// # Arguments
/// * `real_effects` — `[n_perts × n_genes]` row-major: perturbation effects
///   from real data (means_real[p] - means_real[ctrl]).
/// * `pred_effects` — `[n_perts × n_genes]` row-major: perturbation effects
///   from predicted data.
/// * `n_perts` — Number of perturbations (rows).
/// * `n_genes` — Number of genes (columns).
/// * `pert_names` — Name of each perturbation (length = n_perts).
/// * `gene_names` — Name of each gene (length = n_genes). Required if
///   `exclude_target_gene` is true.
/// * `metric` — Distance metric (L1, L2/Euclidean, Cosine).
/// * `exclude_target_gene` — If true, exclude the column for each gene whose
///   name matches the perturbation name when computing that perturbation's
///   distances.
///
/// # Returns
/// `DiscriminationResult` with per-perturbation scores.
#[allow(clippy::too_many_arguments)]
pub fn compute_discrimination_score(
    real_effects: &[f64],
    pred_effects: &[f64],
    n_perts: usize,
    n_genes: usize,
    pert_names: &[String],
    gene_names: Option<&[String]>,
    metric: DistanceMetric,
    exclude_target_gene: bool,
) -> crate::Result<DiscriminationResult> {
    // ── Validation ───────────────────────────────────────────────────
    if real_effects.len() != n_perts * n_genes {
        return Err(crate::AccelError::InvalidInput(format!(
            "real_effects length {} doesn't match n_perts={} × n_genes={}",
            real_effects.len(),
            n_perts,
            n_genes
        )));
    }
    if pred_effects.len() != n_perts * n_genes {
        return Err(crate::AccelError::InvalidInput(format!(
            "pred_effects length {} doesn't match n_perts={} × n_genes={}",
            pred_effects.len(),
            n_perts,
            n_genes
        )));
    }
    if pert_names.len() != n_perts {
        return Err(crate::AccelError::InvalidInput(format!(
            "pert_names length {} != n_perts {}",
            pert_names.len(),
            n_perts
        )));
    }
    if exclude_target_gene {
        if let Some(gn) = gene_names {
            if gn.len() != n_genes {
                return Err(crate::AccelError::InvalidInput(format!(
                    "gene_names length {} != n_genes {}",
                    gn.len(),
                    n_genes
                )));
            }
        } else {
            return Err(crate::AccelError::InvalidInput(
                "gene_names required when exclude_target_gene=true".to_string(),
            ));
        }
    }
    if n_perts == 0 {
        return Err(crate::AccelError::InvalidInput(
            "no perturbations provided".to_string(),
        ));
    }
    if n_genes == 0 {
        return Err(crate::AccelError::InvalidInput("n_genes is 0".to_string()));
    }

    // Pseudobulk means with all-zero groups produce NaN (0/0); NaN distances
    // silently yield rank 0 (since `NaN < x` is false), which would mask the
    // upstream bug as an artificially perfect score. Reject in release too.
    if !real_effects.iter().all(|v| v.is_finite()) {
        return Err(crate::AccelError::InvalidInput(
            "real_effects contains non-finite values (NaN/Inf); \
             likely caused by an empty perturbation group in pseudobulk input"
                .to_string(),
        ));
    }
    if !pred_effects.iter().all(|v| v.is_finite()) {
        return Err(crate::AccelError::InvalidInput(
            "pred_effects contains non-finite values (NaN/Inf); \
             likely caused by an empty perturbation group in pseudobulk input"
                .to_string(),
        ));
    }

    // ── Pre-build gene name → column index map (if needed) ──────────
    let gene_idx_map: Option<std::collections::HashMap<&str, usize>> = if exclude_target_gene {
        gene_names.map(|gn| {
            gn.iter()
                .enumerate()
                .map(|(i, g)| (g.as_str(), i))
                .collect()
        })
    } else {
        None
    };

    // ── Compute scores in parallel ──────────────────────────────────
    let scores: Vec<f64> = (0..n_perts)
        .into_par_iter()
        .map(|p| {
            // Determine which gene column to exclude (if any) for this perturbation.
            let exclude_col: Option<usize> = if exclude_target_gene {
                gene_idx_map
                    .as_ref()
                    .and_then(|m| m.get(pert_names[p].as_str()).copied())
            } else {
                None
            };

            // Compute distance from pred_effects[p] to each real_effects[i].
            let pred_row = &pred_effects[p * n_genes..(p + 1) * n_genes];
            let mut distances: Vec<f64> = Vec::with_capacity(n_perts);

            for i in 0..n_perts {
                let real_row = &real_effects[i * n_genes..(i + 1) * n_genes];
                let d = masked_distance(pred_row, real_row, n_genes, exclude_col, metric);
                distances.push(d);
            }

            // Find rank of the correct perturbation (index p) in sorted order.
            // Rank = number of perturbations with distance strictly less than
            // the distance to the correct one.
            let correct_dist = distances[p];
            let rank = distances.iter().filter(|&&d| d < correct_dist).count();

            // Normalize: 1.0 = rank 0 (best), 1/P = rank P-1 (worst).
            // Matches cell-eval convention: score = 1 - rank / P.
            1.0 - (rank as f64) / (n_perts as f64)
        })
        .collect();

    Ok(DiscriminationResult {
        scores,
        pert_names: pert_names.to_vec(),
    })
}

/// Compute distance between two row vectors with an optional column exclusion.
///
/// When `exclude_col` is `Some(j)`, gene column `j` is skipped. The distance
/// is computed inline via iterators — zero heap allocation per call. This is
/// important because this function is called O(P²) times in the inner loop.
#[inline]
fn masked_distance(
    a: &[f64],
    b: &[f64],
    n_genes: usize,
    exclude_col: Option<usize>,
    metric: DistanceMetric,
) -> f64 {
    match exclude_col {
        None => point_distance(a, b, n_genes, metric),
        Some(excl) => {
            let n_masked = n_genes - 1;
            if n_masked == 0 {
                return 0.0;
            }
            // Compute distance inline without allocating masked vectors.
            let filtered = || (0..n_genes).filter(move |&k| k != excl);
            match metric {
                DistanceMetric::L1 => filtered().map(|k| (a[k] - b[k]).abs()).sum(),
                DistanceMetric::Euclidean => {
                    let sum_sq: f64 = filtered()
                        .map(|k| {
                            let d = a[k] - b[k];
                            d * d
                        })
                        .sum();
                    sum_sq.sqrt()
                }
                DistanceMetric::Cosine => {
                    let (mut dot, mut norm_a, mut norm_b) = (0.0f64, 0.0f64, 0.0f64);
                    for k in filtered() {
                        dot += a[k] * b[k];
                        norm_a += a[k] * a[k];
                        norm_b += b[k] * b[k];
                    }
                    let denom = norm_a.sqrt() * norm_b.sqrt();
                    if denom < 1e-30 {
                        0.0
                    } else {
                        1.0 - dot / denom
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create simple effects matrices for testing.
    fn make_test_effects() -> (Vec<f64>, Vec<f64>, Vec<String>, Vec<String>) {
        // 3 perturbations × 4 genes
        // Real effects: clearly distinct vectors
        let real_effects = vec![
            1.0, 0.0, 0.0, 0.0, // pert_0: gene_0 up
            0.0, 1.0, 0.0, 0.0, // pert_1: gene_1 up
            0.0, 0.0, 1.0, 0.0, // pert_2: gene_2 up
        ];
        // Pred effects: identical to real (perfect prediction)
        let pred_effects = real_effects.clone();

        let pert_names = vec![
            "gene_0".to_string(),
            "gene_1".to_string(),
            "gene_2".to_string(),
        ];
        let gene_names = vec![
            "gene_0".to_string(),
            "gene_1".to_string(),
            "gene_2".to_string(),
            "gene_3".to_string(),
        ];

        (real_effects, pred_effects, pert_names, gene_names)
    }

    #[test]
    fn test_perfect_prediction_l2() {
        let (real, pred, perts, genes) = make_test_effects();
        let result = compute_discrimination_score(
            &real,
            &pred,
            3,
            4,
            &perts,
            Some(&genes),
            DistanceMetric::Euclidean,
            false,
        )
        .unwrap();

        assert_eq!(result.pert_names.len(), 3);
        // Perfect prediction: every pred matches its real effect exactly,
        // distance=0 should give rank 0, so score = 1.0.
        for (i, &score) in result.scores.iter().enumerate() {
            assert!(
                (score - 1.0).abs() < 1e-12,
                "pert_{i} should have score 1.0, got {score}"
            );
        }
    }

    #[test]
    fn test_perfect_prediction_l1() {
        let (real, pred, perts, genes) = make_test_effects();
        let result = compute_discrimination_score(
            &real,
            &pred,
            3,
            4,
            &perts,
            Some(&genes),
            DistanceMetric::L1,
            false,
        )
        .unwrap();

        for &score in &result.scores {
            assert!(
                (score - 1.0).abs() < 1e-12,
                "perfect pred L1 score should be 1.0, got {score}"
            );
        }
    }

    #[test]
    fn test_perfect_prediction_cosine() {
        let (real, pred, perts, genes) = make_test_effects();
        let result = compute_discrimination_score(
            &real,
            &pred,
            3,
            4,
            &perts,
            Some(&genes),
            DistanceMetric::Cosine,
            false,
        )
        .unwrap();

        for &score in &result.scores {
            assert!(
                (score - 1.0).abs() < 1e-12,
                "perfect pred cosine score should be 1.0, got {score}"
            );
        }
    }

    #[test]
    fn test_worst_prediction() {
        // Pred effects are maximally wrong: each pred matches the wrong real.
        // pert_0's pred looks like pert_2's real, etc.
        let real_effects = vec![
            1.0, 0.0, 0.0, // pert_0
            0.0, 1.0, 0.0, // pert_1
            0.0, 0.0, 1.0, // pert_2
        ];
        let pred_effects = vec![
            0.0, 0.0, 1.0, // pert_0 pred → looks like pert_2
            1.0, 0.0, 0.0, // pert_1 pred → looks like pert_0
            0.0, 1.0, 0.0, // pert_2 pred → looks like pert_1
        ];
        let perts = vec!["p0".to_string(), "p1".to_string(), "p2".to_string()];

        let result = compute_discrimination_score(
            &real_effects,
            &pred_effects,
            3,
            3,
            &perts,
            None,
            DistanceMetric::Euclidean,
            false,
        )
        .unwrap();

        // Each prediction's closest real match is NOT the correct one.
        // With 3 perts and all wrong, the rank should be > 0 for each.
        for &score in &result.scores {
            assert!(
                score < 1.0,
                "wrong pred should have score < 1.0, got {score}"
            );
        }
    }

    #[test]
    fn test_random_prediction_average_score() {
        // With many random perturbations, average discrimination score
        // should be approximately 0.5 (random ranking).
        let n_perts = 100;
        let n_genes = 10;
        let mut rng_seed = 42u64;

        // Simple LCG for deterministic pseudo-random without pulling in rand
        let mut next_rand = || -> f64 {
            rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng_seed >> 33) as f64 / (1u64 << 31) as f64
        };

        let real_effects: Vec<f64> = (0..n_perts * n_genes).map(|_| next_rand()).collect();
        let pred_effects: Vec<f64> = (0..n_perts * n_genes).map(|_| next_rand()).collect();
        let perts: Vec<String> = (0..n_perts).map(|i| format!("p{i}")).collect();

        let result = compute_discrimination_score(
            &real_effects,
            &pred_effects,
            n_perts,
            n_genes,
            &perts,
            None,
            DistanceMetric::Euclidean,
            false,
        )
        .unwrap();

        let avg_score = result.scores.iter().sum::<f64>() / n_perts as f64;
        // Random ranking → expected average ≈ 0.5 (with some variance)
        assert!(
            (avg_score - 0.5).abs() < 0.15,
            "random prediction average score should be ~0.5, got {avg_score}"
        );
    }

    #[test]
    fn test_exclude_target_gene() {
        // 2 perturbations × 3 genes, where perturbation names match gene names.
        // gene_0 is strongly affected by pert "gene_0", creating a trivial match.
        let real_effects = vec![
            10.0, 0.1, 0.2, // gene_0: gene_0 column dominates
            0.1, 10.0, 0.2, // gene_1: gene_1 column dominates
        ];
        // Pred: same structure but with gene columns swapped for non-target genes
        let pred_effects = vec![
            10.0, 0.2, 0.1, // gene_0: gene_0 column still dominates
            0.2, 10.0, 0.1, // gene_1: gene_1 column still dominates
        ];
        let perts = vec!["gene_0".to_string(), "gene_1".to_string()];
        let genes = vec![
            "gene_0".to_string(),
            "gene_1".to_string(),
            "gene_2".to_string(),
        ];

        // Without exclusion: should be easy to discriminate (score 1.0)
        let without = compute_discrimination_score(
            &real_effects,
            &pred_effects,
            2,
            3,
            &perts,
            Some(&genes),
            DistanceMetric::L1,
            false,
        )
        .unwrap();

        // With exclusion: removes the dominant column, discrimination may be harder
        let with_excl = compute_discrimination_score(
            &real_effects,
            &pred_effects,
            2,
            3,
            &perts,
            Some(&genes),
            DistanceMetric::L1,
            true,
        )
        .unwrap();

        // Both should produce valid scores
        for &s in &without.scores {
            assert!((0.0..=1.0).contains(&s), "score out of range: {s}");
        }
        for &s in &with_excl.scores {
            assert!((0.0..=1.0).contains(&s), "score out of range: {s}");
        }
    }

    #[test]
    fn test_exclude_target_gene_missing_name() {
        // Perturbation name doesn't match any gene → no exclusion applied
        let real_effects = vec![1.0, 0.0, 0.0, 1.0];
        let pred_effects = vec![1.0, 0.0, 0.0, 1.0];
        let perts = vec!["unknown_pert_0".to_string(), "unknown_pert_1".to_string()];
        let genes = vec!["gene_A".to_string(), "gene_B".to_string()];

        let result = compute_discrimination_score(
            &real_effects,
            &pred_effects,
            2,
            2,
            &perts,
            Some(&genes),
            DistanceMetric::L1,
            true, // exclude_target_gene=true, but names don't match
        )
        .unwrap();

        // Should still work — just no columns excluded
        assert_eq!(result.scores.len(), 2);
    }

    #[test]
    fn test_single_perturbation() {
        // Only one perturbation: rank is always 0, score is always 1.0
        let real = vec![1.0, 2.0, 3.0];
        let pred = vec![4.0, 5.0, 6.0]; // different, but only 1 pert so rank=0
        let perts = vec!["p0".to_string()];

        let result = compute_discrimination_score(
            &real,
            &pred,
            1,
            3,
            &perts,
            None,
            DistanceMetric::Euclidean,
            false,
        )
        .unwrap();

        assert_eq!(result.scores.len(), 1);
        assert!(
            (result.scores[0] - 1.0).abs() < 1e-12,
            "single pert always gets score 1.0"
        );
    }

    #[test]
    fn test_validation_errors() {
        let perts = vec!["a".to_string(), "b".to_string()];
        let genes = ["g0".to_string(), "g1".to_string()];

        // Wrong real_effects length
        assert!(compute_discrimination_score(
            &[0.0; 3], // should be 4
            &[0.0; 4],
            2,
            2,
            &perts,
            None,
            DistanceMetric::L1,
            false,
        )
        .is_err());

        // Wrong pred_effects length
        assert!(compute_discrimination_score(
            &[0.0; 4],
            &[0.0; 3], // should be 4
            2,
            2,
            &perts,
            None,
            DistanceMetric::L1,
            false,
        )
        .is_err());

        // exclude_target_gene=true but no gene_names
        assert!(compute_discrimination_score(
            &[0.0; 4],
            &[0.0; 4],
            2,
            2,
            &perts,
            None, // missing!
            DistanceMetric::L1,
            true,
        )
        .is_err());

        // gene_names length mismatch
        assert!(compute_discrimination_score(
            &[0.0; 4],
            &[0.0; 4],
            2,
            2,
            &perts,
            Some(&genes[..1]), // length 1, need 2
            DistanceMetric::L1,
            true,
        )
        .is_err());

        // n_perts == 0
        assert!(
            compute_discrimination_score(&[], &[], 0, 2, &[], None, DistanceMetric::L1, false,)
                .is_err()
        );
    }

    #[test]
    fn test_matches_cell_eval_algorithm() {
        // Reproduce the cell-eval discrimination_score algorithm step by step
        // on a small example and verify Rust matches.
        //
        // 3 perturbations × 3 genes:
        //   real_effects = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        //   pred_effects = [[0.9, 0.1, 0], [0.1, 1.8, 0.1], [0.1, 0.1, 2.7]]
        //
        // Using L1 metric, no gene exclusion:
        // For pert_0: distances from pred[0]=[0.9,0.1,0] to each real row:
        //   to real[0]=[1,0,0]: |0.9-1|+|0.1-0|+|0-0| = 0.1+0.1+0 = 0.2
        //   to real[1]=[0,2,0]: |0.9-0|+|0.1-2|+|0-0| = 0.9+1.9+0 = 2.8
        //   to real[2]=[0,0,3]: |0.9-0|+|0.1-0|+|0-3| = 0.9+0.1+3 = 4.0
        //   sorted: [0.2, 2.8, 4.0], correct=0, rank=0
        //   score = 1 - 0/3 = 1.0
        //
        // For pert_1: distances from pred[1]=[0.1,1.8,0.1] to each real row:
        //   to real[0]=[1,0,0]: |0.1-1|+|1.8-0|+|0.1-0| = 0.9+1.8+0.1 = 2.8
        //   to real[1]=[0,2,0]: |0.1-0|+|1.8-2|+|0.1-0| = 0.1+0.2+0.1 = 0.4
        //   to real[2]=[0,0,3]: |0.1-0|+|1.8-0|+|0.1-3| = 0.1+1.8+2.9 = 4.8
        //   sorted: [0.4, 2.8, 4.8], correct=1, rank=0
        //   score = 1 - 0/3 = 1.0
        //
        // For pert_2: distances from pred[2]=[0.1,0.1,2.7] to each real row:
        //   to real[0]=[1,0,0]: |0.1-1|+|0.1-0|+|2.7-0| = 0.9+0.1+2.7 = 3.7
        //   to real[1]=[0,2,0]: |0.1-0|+|0.1-2|+|2.7-0| = 0.1+1.9+2.7 = 4.7
        //   to real[2]=[0,0,3]: |0.1-0|+|0.1-0|+|2.7-3| = 0.1+0.1+0.3 = 0.5
        //   sorted: [0.5, 3.7, 4.7], correct=2, rank=0
        //   score = 1 - 0/3 = 1.0

        let real_effects = vec![
            1.0, 0.0, 0.0, // pert_0
            0.0, 2.0, 0.0, // pert_1
            0.0, 0.0, 3.0, // pert_2
        ];
        let pred_effects = vec![
            0.9, 0.1, 0.0, // pert_0 pred
            0.1, 1.8, 0.1, // pert_1 pred
            0.1, 0.1, 2.7, // pert_2 pred
        ];
        let perts = vec!["p0".to_string(), "p1".to_string(), "p2".to_string()];

        let result = compute_discrimination_score(
            &real_effects,
            &pred_effects,
            3,
            3,
            &perts,
            None,
            DistanceMetric::L1,
            false,
        )
        .unwrap();

        // All scores should be 1.0 (predictions are closest to correct real)
        for (i, &score) in result.scores.iter().enumerate() {
            assert!(
                (score - 1.0).abs() < 1e-12,
                "pert_{i} expected score 1.0, got {score}"
            );
        }
    }

    #[test]
    fn test_rejects_non_finite_effects() {
        let perts = vec!["p0".to_string(), "p1".to_string()];

        let mut real_with_nan = vec![1.0, 2.0, 3.0, 4.0];
        real_with_nan[2] = f64::NAN;
        let err = compute_discrimination_score(
            &real_with_nan,
            &[0.0; 4],
            2,
            2,
            &perts,
            None,
            DistanceMetric::Euclidean,
            false,
        )
        .expect_err("NaN in real_effects must error");
        assert!(
            matches!(err, crate::AccelError::InvalidInput(ref m) if m.contains("real_effects"))
        );

        let mut pred_with_inf = vec![1.0, 2.0, 3.0, 4.0];
        pred_with_inf[0] = f64::INFINITY;
        let err = compute_discrimination_score(
            &[0.0; 4],
            &pred_with_inf,
            2,
            2,
            &perts,
            None,
            DistanceMetric::Euclidean,
            false,
        )
        .expect_err("Inf in pred_effects must error");
        assert!(
            matches!(err, crate::AccelError::InvalidInput(ref m) if m.contains("pred_effects"))
        );
    }

    #[test]
    fn test_parallel_stress() {
        // Stress test with many perturbations to exercise rayon par_iter.
        let n_perts = 200;
        let n_genes = 50;

        // Each perturbation has a unique "signature" gene.
        let mut real_effects = vec![0.0; n_perts * n_genes];
        let mut pred_effects = vec![0.0; n_perts * n_genes];
        for p in 0..n_perts {
            // Real: spike in gene p % n_genes
            let sig_gene = p % n_genes;
            real_effects[p * n_genes + sig_gene] = (p + 1) as f64;
            // Pred: same spike with small noise
            pred_effects[p * n_genes + sig_gene] = (p + 1) as f64 + 0.01;
        }

        let perts: Vec<String> = (0..n_perts).map(|i| format!("p{i}")).collect();

        let result = compute_discrimination_score(
            &real_effects,
            &pred_effects,
            n_perts,
            n_genes,
            &perts,
            None,
            DistanceMetric::Euclidean,
            false,
        )
        .unwrap();

        assert_eq!(result.scores.len(), n_perts);

        // With unique signatures plus tiny noise, most should get score 1.0.
        // 200 perts × 50 genes: 4 perts share each signature gene (p % n_genes),
        // causing distance ties that reduce some scores below 1.0.
        // Check at least 80% get score 1.0.
        let perfect_count = result
            .scores
            .iter()
            .filter(|&&s| (s - 1.0).abs() < 1e-12)
            .count();
        assert!(
            perfect_count >= n_perts * 4 / 5,
            "expected most perts to score 1.0, got {perfect_count}/{n_perts}"
        );
    }
}
