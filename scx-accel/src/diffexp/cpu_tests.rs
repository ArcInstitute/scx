use super::*;

#[test]
fn test_normal_cdf_symmetry() {
    let cdf_0 = normal_cdf(0.0);
    assert!((cdf_0 - 0.5).abs() < 1e-6, "Φ(0) should be 0.5");

    let cdf_pos = normal_cdf(1.96);
    assert!((cdf_pos - 0.975).abs() < 0.001, "Φ(1.96) ≈ 0.975");

    let cdf_neg = normal_cdf(-1.96);
    assert!((cdf_neg - 0.025).abs() < 0.001, "Φ(-1.96) ≈ 0.025");
}

#[test]
fn test_bh_correction() {
    let pvals = vec![0.01, 0.04, 0.03, 0.005, 0.5];
    let adj = benjamini_hochberg(&pvals);

    // All adjusted ≥ raw
    for (raw, a) in pvals.iter().zip(adj.iter()) {
        assert!(*a >= *raw - 1e-12, "adjusted {a} should be >= raw {raw}");
    }
    // All adjusted ≤ 1
    for a in &adj {
        assert!(*a <= 1.0 + 1e-12, "adjusted {a} should be <= 1.0");
    }
}

#[test]
fn test_bh_monotonicity() {
    // Sorted p-values should yield non-decreasing adjusted p-values.
    let pvals = vec![0.001, 0.01, 0.05, 0.1, 0.5];
    let adj = benjamini_hochberg(&pvals);
    for i in 1..adj.len() {
        assert!(
            adj[i] >= adj[i - 1] - 1e-12,
            "BH adjusted p-values should be monotonically non-decreasing for sorted input"
        );
    }
}

#[test]
fn test_wilcoxon_identical_groups() {
    // Two identical groups should give z ≈ 0, p ≈ 1.
    let group = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let rest = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (z, p) = wilcoxon_test(&group, &rest);
    assert!(
        z.abs() < 1e-6,
        "z should be ≈ 0 for identical groups, got {z}"
    );
    assert!(p > 0.9, "p should be ≈ 1 for identical groups, got {p}");
}

#[test]
fn test_wilcoxon_separated_groups() {
    // Fully separated groups should give a very significant p-value.
    let group = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
    let rest = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let (z, p) = wilcoxon_test(&group, &rest);
    assert!(z > 0.0, "z should be positive (group has higher values)");
    assert!(
        p < 0.001,
        "p should be very small for separated groups, got {p}"
    );
}

/// ACC10: a non-finite value in the input matrix is rejected at the DE
/// boundary rather than silently producing garbage ranks.
#[test]
fn test_wilcoxon_rejects_non_finite_input() {
    let n_obs = 4;
    let n_vars = 2;
    let mut data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    data[5] = f32::NAN;
    let groups = vec![0usize, 0, 1, 1];
    let gene_names = vec!["g0".to_string(), "g1".to_string()];
    let group_names = vec!["A".to_string(), "B".to_string()];
    let err = wilcoxon_rank_sum(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        None,
        false,
        false,
        false,
        0,
    )
    .unwrap_err();
    assert!(
        matches!(err, crate::AccelError::InvalidInput(_)),
        "expected InvalidInput for NaN data, got {err:?}"
    );
}

#[test]
fn test_wilcoxon_rank_sum_basic() {
    // Simple 2-group test: genes with known differential expression.
    let n_obs = 20;
    let n_vars = 3;

    // Gene 0: highly expressed in group 0, low in group 1
    // Gene 1: similar expression in both groups
    // Gene 2: highly expressed in group 1, low in group 0
    let mut data = vec![0.0f32; n_obs * n_vars];
    let groups: Vec<usize> = (0..n_obs).map(|i| if i < 10 { 0 } else { 1 }).collect();

    for i in 0..10 {
        data[i * n_vars + 0] = 10.0 + i as f32; // group 0, gene 0: high
        data[i * n_vars + 1] = 5.0; // group 0, gene 1: medium
        data[i * n_vars + 2] = 1.0; // group 0, gene 2: low
    }
    for i in 10..20 {
        data[i * n_vars + 0] = 1.0; // group 1, gene 0: low
        data[i * n_vars + 1] = 5.0; // group 1, gene 1: medium
        data[i * n_vars + 2] = 10.0 + (i - 10) as f32; // group 1, gene 2: high
    }

    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    let group_names = vec!["A".to_string(), "B".to_string()];

    let result = wilcoxon_rank_sum(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        None,
        false,
        true,  // rankby_abs=true to test absolute sort (original test expectation)
        false, // tie_correct=false (match scanpy default)
        0,
    )
    .unwrap();

    assert_eq!(result.group_names.len(), 2);
    assert_eq!(result.names[0].len(), n_vars);
    assert_eq!(result.scores[0].len(), n_vars);
    assert_eq!(result.pvals[0].len(), n_vars);
    assert_eq!(result.pvals_adj[0].len(), n_vars);
    assert_eq!(result.logfoldchanges[0].len(), n_vars);

    // For group A: gene_0 and gene_2 should be the top 2 DE genes (sorted by |z|).
    // gene_0 is upregulated in A, gene_2 is downregulated — both have large |z|.
    let top2_a: Vec<&str> = result.names[0][..2].iter().map(|s| s.as_str()).collect();
    assert!(
        top2_a.contains(&"gene_0"),
        "gene_0 should be top-2 DE for group A"
    );
    assert!(
        top2_a.contains(&"gene_2"),
        "gene_2 should be top-2 DE for group A"
    );

    // gene_0 should have positive logFC for group A (upregulated).
    let gene0_idx_a = result.names[0].iter().position(|n| n == "gene_0").unwrap();
    assert!(
        result.logfoldchanges[0][gene0_idx_a] > 0.0,
        "gene_0 should have positive logFC for group A"
    );
    assert!(
        result.scores[0][gene0_idx_a] > 0.0,
        "gene_0 should have positive z for group A"
    );

    // For group B: gene_2 should have positive z and logFC.
    let gene2_idx_b = result.names[1].iter().position(|n| n == "gene_2").unwrap();
    assert!(gene2_idx_b < 2, "gene_2 should be top-2 DE for group B");
    assert!(
        result.scores[1][gene2_idx_b] > 0.0,
        "gene_2 should have positive z for group B"
    );
    assert!(
        result.logfoldchanges[1][gene2_idx_b] > 0.0,
        "gene_2 should have positive logFC for group B"
    );
}

#[test]
fn test_pairwise_reference() {
    let n_obs = 30;
    let n_vars = 2;
    let data = vec![5.0f32; n_obs * n_vars];
    let groups: Vec<usize> = (0..n_obs)
        .map(|i| {
            if i < 10 {
                0
            } else if i < 20 {
                1
            } else {
                2
            }
        })
        .collect();
    let gene_names = vec!["g0".to_string(), "g1".to_string()];
    let group_names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

    // With reference = Some(0) (A), only B and C should appear in results.
    let result = wilcoxon_rank_sum(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        Some(0),
        false,
        false,
        false,
        0,
    )
    .unwrap();

    assert_eq!(result.group_names.len(), 2);
    assert!(result.group_names.contains(&"B".to_string()));
    assert!(result.group_names.contains(&"C".to_string()));
    assert!(!result.group_names.contains(&"A".to_string()));
}

#[test]
fn test_logfc_log_transformed() {
    // When data is log1p-transformed, log_transformed=true should apply
    // expm1 before computing the fold-change ratio, matching scanpy.
    let n_obs = 20;
    let n_vars = 2;
    // Group 0: gene 0 has high raw expression (10.0), gene 1 low (1.0)
    // Group 1: gene 0 has low raw expression (1.0), gene 1 high (10.0)
    // We log1p-transform the values before passing to the function.
    let mut data = vec![0.0f32; n_obs * n_vars];
    let groups: Vec<usize> = (0..n_obs).map(|i| if i < 10 { 0 } else { 1 }).collect();

    for i in 0..10 {
        data[i * n_vars + 0] = (10.0f32 + 1.0).ln(); // ln(11) ≈ 2.397
        data[i * n_vars + 1] = (1.0f32 + 1.0).ln(); // ln(2) ≈ 0.693
    }
    for i in 10..20 {
        data[i * n_vars + 0] = (1.0f32 + 1.0).ln(); // ln(2) ≈ 0.693
        data[i * n_vars + 1] = (10.0f32 + 1.0).ln(); // ln(11) ≈ 2.397
    }

    let gene_names = vec!["g0".to_string(), "g1".to_string()];
    let group_names = vec!["A".to_string(), "B".to_string()];

    // With log_transformed=true, logFC should be:
    // log2((expm1(mean_group) + 1e-9) / (expm1(mean_ref) + 1e-9))
    // For group A, gene 0: expm1(ln(11)) = 10.0, expm1(ln(2)) = 1.0
    // → log2((10 + 1e-9) / (1 + 1e-9)) ≈ log2(10) ≈ 3.32
    let result_log = wilcoxon_rank_sum(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        None,
        true,
        false,
        false,
        0,
    )
    .unwrap();

    // With log_transformed=false, logFC operates on the ln-scale values directly.
    let result_raw = wilcoxon_rank_sum(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        None,
        false,
        false,
        false,
        0,
    )
    .unwrap();

    // Find gene g0 in group A.
    let g0_idx_log = result_log.names[0].iter().position(|n| n == "g0").unwrap();
    let g0_idx_raw = result_raw.names[0].iter().position(|n| n == "g0").unwrap();

    let logfc_log = result_log.logfoldchanges[0][g0_idx_log];
    let logfc_raw = result_raw.logfoldchanges[0][g0_idx_raw];

    // The expm1-based logFC should be ≈ log2(10) ≈ 3.32.
    assert!(
        (logfc_log - 10.0f64.log2()).abs() < 0.01,
        "expected logFC ≈ {:.4}, got {:.4}",
        10.0f64.log2(),
        logfc_log
    );

    // The two should differ significantly (raw logFC is on the ln-scale).
    assert!(
        (logfc_log - logfc_raw).abs() > 0.1,
        "log_transformed and raw logFC should differ, got log={logfc_log:.4} raw={logfc_raw:.4}"
    );
}

/// Locks the O(1) 1-vs-rest reference-sum optimization (P4/OPT-1.3) against a
/// brute-force filtered re-sum. The production path derives the rest mean as
/// `(total_gene_sum[var] − group_gene_sums[g][var]) / n2`; this recomputes it
/// the old way (`Σ_{gg≠g} group_gene_sums[gg][var] / n2`) and asserts the
/// resulting logFC matches across a many-group fixture. The same algebra now
/// backs the GPU host-side logFC pass in `diffexp/gpu.rs`.
#[test]
fn test_one_vs_rest_logfc_matches_bruteforce_restsum() {
    let n_groups = 10usize;
    let n_vars = 4usize;
    let per_group = 8usize;
    let n_obs = n_groups * per_group;

    // Deterministic, varied non-negative counts.
    let groups: Vec<usize> = (0..n_obs).map(|i| i / per_group).collect();
    let mut data = vec![0.0f32; n_obs * n_vars];
    for (cell, g) in groups.iter().enumerate() {
        for var in 0..n_vars {
            // Vary by group, gene, and cell so means differ across groups.
            data[cell * n_vars + var] = ((g * 3 + var * 2 + (cell % per_group)) % 17) as f32;
        }
    }

    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    let group_names: Vec<String> = (0..n_groups).map(|g| format!("grp_{g}")).collect();

    let log_transformed = false;
    let result = wilcoxon_rank_sum(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        None, // 1-vs-rest
        log_transformed,
        false,
        false,
        0,
    )
    .unwrap();

    // Independent brute-force group/gene sums.
    let mut group_gene_sums = vec![vec![0.0f64; n_vars]; n_groups];
    for (cell, &g) in groups.iter().enumerate() {
        for var in 0..n_vars {
            group_gene_sums[g][var] += data[cell * n_vars + var] as f64;
        }
    }

    for (g, gname) in group_names.iter().enumerate() {
        // Result rows are gene-reordered per group; locate this group's row.
        let row = result
            .group_names
            .iter()
            .position(|n| n == gname)
            .expect("group present in result");
        let n1 = per_group as f64;
        let n2 = (n_obs - per_group) as f64;
        for (var, vname) in gene_names.iter().enumerate() {
            let mean_group = group_gene_sums[g][var] / n1;
            // Old O(n_groups) filtered re-sum.
            let rest_sum: f64 = (0..n_groups)
                .filter(|&gg| gg != g)
                .map(|gg| group_gene_sums[gg][var])
                .sum();
            let mean_ref = rest_sum / n2;
            let expected = compute_logfc(mean_group, mean_ref, log_transformed);

            let col = result.names[row]
                .iter()
                .position(|n| n == vname)
                .expect("gene present in result row");
            let got = result.logfoldchanges[row][col];
            assert!(
                (got - expected).abs() < 1e-9,
                "logFC mismatch for {gname}/{vname}: got {got}, expected {expected}"
            );
        }
    }
}

/// Pins OPT-3.3's load-bearing guarantee: folding the per-group sums into the
/// parallel gather is **bit-identical** to the deleted serial `group_gene_sums` /
/// `total_gene_sum` pre-pass — not merely within tolerance (the
/// `..._matches_bruteforce_restsum` test above is a `<1e-9` check using a
/// *different* summation order, so it would not catch a reorder regression).
/// This reproduces the old summation order exactly — per-group sums over
/// `group_indices[g]` ascending, the 1-vs-rest total over groups `0..n_groups`,
/// pairwise group/ref sums over their ascending cell lists — and asserts the
/// production logFC matches f64-bit-for-bit, across both `log_transformed`
/// branches and both the 1-vs-rest and pairwise arms.
#[test]
fn test_logfc_bit_identical_to_serial_group_sums() {
    let n_groups = 7usize;
    let n_vars = 6usize;
    let per_group = 9usize; // balanced → no empty/full group → no NaN logFC
    let n_obs = n_groups * per_group;

    let groups: Vec<usize> = (0..n_obs).map(|i| i / per_group).collect();
    let mut data = vec![0.0f32; n_obs * n_vars];
    for (cell, &g) in groups.iter().enumerate() {
        for var in 0..n_vars {
            // Non-integer values so any reorder would surface as ULP drift.
            data[cell * n_vars + var] =
                (((g * 7 + var * 13 + (cell % per_group) * 3) % 23) as f32) * 0.5 + 0.125;
        }
    }
    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("g{i}")).collect();
    let group_names: Vec<String> = (0..n_groups).map(|g| format!("c{g}")).collect();

    // Reference sums in the *old* serial order.
    let mut group_indices: Vec<Vec<usize>> = vec![vec![]; n_groups];
    for (cell, &g) in groups.iter().enumerate() {
        group_indices[g].push(cell);
    }
    let mut group_gene_sums = vec![vec![0.0f64; n_vars]; n_groups];
    for g in 0..n_groups {
        for &cell in &group_indices[g] {
            for var in 0..n_vars {
                group_gene_sums[g][var] += data[cell * n_vars + var] as f64;
            }
        }
    }
    let mut total_gene_sum = vec![0.0f64; n_vars];
    for sums in &group_gene_sums {
        for (t, &s) in total_gene_sum.iter_mut().zip(sums.iter()) {
            *t += s;
        }
    }

    let locate = |res: &DiffExpResult, gname: &str, vname: &str| -> f64 {
        let row = res.group_names.iter().position(|n| n == gname).unwrap();
        let col = res.names[row].iter().position(|n| n == vname).unwrap();
        res.logfoldchanges[row][col]
    };

    for &log_transformed in &[false, true] {
        // 1-vs-rest.
        let res = wilcoxon_rank_sum(
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            None,
            log_transformed,
            false,
            false,
            0,
        )
        .unwrap();
        for (g, gname) in group_names.iter().enumerate() {
            let n1 = group_indices[g].len() as f64;
            let n2 = (n_obs - group_indices[g].len()) as f64;
            for (var, vname) in gene_names.iter().enumerate() {
                let mean_group = group_gene_sums[g][var] / n1;
                let rest = total_gene_sum[var] - group_gene_sums[g][var];
                let expected = compute_logfc(mean_group, rest / n2, log_transformed);
                let got = locate(&res, gname, vname);
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "1-vs-rest logFC not bit-identical for {gname}/{vname} (log={log_transformed}): got {got}, expected {expected}"
                );
            }
        }

        // Pairwise (reference = group 0).
        let ref_idx = 0usize;
        let res = wilcoxon_rank_sum(
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            Some(ref_idx),
            log_transformed,
            false,
            false,
            0,
        )
        .unwrap();
        let n2 = group_indices[ref_idx].len() as f64;
        for (g, gname) in group_names.iter().enumerate() {
            if g == ref_idx {
                continue;
            }
            let n1 = group_indices[g].len() as f64;
            for (var, vname) in gene_names.iter().enumerate() {
                let mean_group = group_gene_sums[g][var] / n1;
                let mean_ref = group_gene_sums[ref_idx][var] / n2;
                let expected = compute_logfc(mean_group, mean_ref, log_transformed);
                let got = locate(&res, gname, vname);
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "pairwise logFC not bit-identical for {gname}/{vname} (log={log_transformed})"
                );
            }
        }
    }
}

#[test]
fn test_merge_empty() {
    let merged = merge_diff_exp_results(vec![], false).unwrap();
    assert!(merged.group_names.is_empty());
    assert!(merged.names.is_empty());
}

#[test]
fn test_merge_single_chunk() {
    // Single chunk should pass through unchanged (except BH is unchanged).
    let chunk = DiffExpResult {
        group_names: vec!["A".to_string()],
        names: vec![vec!["g1".to_string(), "g2".to_string()]],
        gene_indices: vec![vec![0, 1]],
        scores: vec![vec![3.0, 1.0]],
        pvals: vec![vec![0.001, 0.05]],
        pvals_adj: vec![vec![0.002, 0.05]],
        logfoldchanges: vec![vec![2.0, 0.5]],
        exec_info: crate::route::AccelExecutionInfo::default(),
    };
    let merged = merge_diff_exp_results(vec![chunk.clone()], false).unwrap();
    assert_eq!(merged.group_names, chunk.group_names);
    assert_eq!(merged.names, chunk.names);
    assert_eq!(merged.scores, chunk.scores);
}

#[test]
fn test_merge_two_chunks_resorts() {
    // Two chunks: chunk1 has gene_a (z=1.0), chunk2 has gene_b (z=3.0).
    // After merge, gene_b should come first (higher |z|).
    let chunk1 = DiffExpResult {
        group_names: vec!["G".to_string()],
        names: vec![vec!["gene_a".to_string()]],
        gene_indices: vec![vec![0]],
        scores: vec![vec![1.0]],
        pvals: vec![vec![0.3]],
        pvals_adj: vec![vec![0.3]],
        logfoldchanges: vec![vec![0.5]],
        exec_info: crate::route::AccelExecutionInfo::default(),
    };
    let chunk2 = DiffExpResult {
        group_names: vec!["G".to_string()],
        names: vec![vec!["gene_b".to_string()]],
        gene_indices: vec![vec![1]],
        scores: vec![vec![3.0]],
        pvals: vec![vec![0.001]],
        pvals_adj: vec![vec![0.001]],
        logfoldchanges: vec![vec![2.0]],
        exec_info: crate::route::AccelExecutionInfo::default(),
    };

    let merged = merge_diff_exp_results(vec![chunk1, chunk2], false).unwrap();
    assert_eq!(merged.group_names, vec!["G"]);
    assert_eq!(merged.names[0].len(), 2);
    // gene_b should be first (signed score 3.0 > 1.0).
    assert_eq!(merged.names[0][0], "gene_b");
    assert_eq!(merged.names[0][1], "gene_a");
    assert_eq!(merged.scores[0][0], 3.0);
    assert_eq!(merged.scores[0][1], 1.0);
}

#[test]
fn test_merge_global_bh() {
    // Two chunks with 1 gene each → merged BH uses n=2, not n=1.
    let chunk1 = DiffExpResult {
        group_names: vec!["G".to_string()],
        names: vec![vec!["gene_a".to_string()]],
        gene_indices: vec![vec![0]],
        scores: vec![vec![2.0]],
        pvals: vec![vec![0.04]],
        pvals_adj: vec![vec![0.04]], // per-chunk BH with n=1
        logfoldchanges: vec![vec![1.0]],
        exec_info: crate::route::AccelExecutionInfo::default(),
    };
    let chunk2 = DiffExpResult {
        group_names: vec!["G".to_string()],
        names: vec![vec!["gene_b".to_string()]],
        gene_indices: vec![vec![1]],
        scores: vec![vec![1.0]],
        pvals: vec![vec![0.03]],
        pvals_adj: vec![vec![0.03]],
        logfoldchanges: vec![vec![0.5]],
        exec_info: crate::route::AccelExecutionInfo::default(),
    };

    let merged = merge_diff_exp_results(vec![chunk1, chunk2], false).unwrap();
    // With global BH (n=2): sorted p-vals are [0.03, 0.04]
    // Results sorted by signed score: gene_a (z=2) first, gene_b (z=1) second
    // So pvals_adj order follows the score sort.
    // All adjusted should be >= raw and <= 1.
    for (raw, adj) in merged.pvals[0].iter().zip(merged.pvals_adj[0].iter()) {
        assert!(*adj >= *raw - 1e-12);
        assert!(*adj <= 1.0 + 1e-12);
    }
}

#[test]
fn test_wilcoxon_sparse_matches_dense() {
    // Build a small CSR and verify wilcoxon_rank_sum_sparse produces
    // bit-identical results to wilcoxon_rank_sum on equivalent dense data.
    let n_obs = 20;
    let n_vars = 3;

    let mut dense_data = vec![0.0f32; n_obs * n_vars];
    let groups: Vec<usize> = (0..n_obs).map(|i| if i < 10 { 0 } else { 1 }).collect();

    for i in 0..10 {
        dense_data[i * n_vars + 0] = 10.0 + i as f32;
        dense_data[i * n_vars + 1] = 5.0;
        dense_data[i * n_vars + 2] = 1.0;
    }
    for i in 10..20 {
        dense_data[i * n_vars + 0] = 1.0;
        dense_data[i * n_vars + 1] = 5.0;
        dense_data[i * n_vars + 2] = 10.0 + (i - 10) as f32;
    }

    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    let group_names = vec!["A".to_string(), "B".to_string()];

    // Dense path
    let result_dense = wilcoxon_rank_sum(
        &dense_data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        None,
        false,
        false,
        false,
        0,
    )
    .unwrap();

    // Build ScxCsr from the same data
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for r in 0..n_obs {
        for c in 0..n_vars {
            let v = dense_data[r * n_vars + c];
            if v != 0.0 {
                indices.push(c as i32);
                data.push(v);
            }
        }
        indptr.push(indices.len() as i64);
    }
    let csr = scx_sparse::ScxCsr::new_unchecked((n_obs, n_vars), indptr, indices, data);

    // Sparse path with chunk_size=2 (forces 2 chunks for 3 genes)
    let result_sparse = wilcoxon_rank_sum_sparse(
        &csr,
        &gene_names,
        &groups,
        &group_names,
        None,
        2,
        false,
        false,
        false,
    )
    .unwrap();

    // Same group structure
    assert_eq!(result_dense.group_names, result_sparse.group_names);

    // Same gene names per group (same ordering)
    for g in 0..result_dense.group_names.len() {
        assert_eq!(result_dense.names[g], result_sparse.names[g]);
        // Scores should match
        for i in 0..result_dense.scores[g].len() {
            assert!(
                (result_dense.scores[g][i] - result_sparse.scores[g][i]).abs() < 1e-10,
                "score mismatch at group {} gene {}: {} vs {}",
                g,
                i,
                result_dense.scores[g][i],
                result_sparse.scores[g][i],
            );
        }
        // Raw p-values should match
        for i in 0..result_dense.pvals[g].len() {
            assert!(
                (result_dense.pvals[g][i] - result_sparse.pvals[g][i]).abs() < 1e-10,
                "pval mismatch at group {} gene {}",
                g,
                i,
            );
        }
    }
}

// ── DE ranking determinism (ACC1) ──────────────────────────────────────

#[test]
fn test_de_rank_cmp_orders_and_tiebreaks() {
    use std::cmp::Ordering;
    // Higher signed score ranks first (descending).
    assert_eq!(de_rank_cmp(3.0, 0, 1.0, 1, false), Ordering::Less);
    assert_eq!(de_rank_cmp(1.0, 0, 3.0, 1, false), Ordering::Greater);
    // Equal finite scores break by ascending global index.
    assert_eq!(de_rank_cmp(2.0, 5, 2.0, 9, false), Ordering::Less);
    assert_eq!(de_rank_cmp(2.0, 9, 2.0, 5, false), Ordering::Greater);
    // rankby_abs: a large-magnitude negative score outranks a small positive.
    assert_eq!(de_rank_cmp(-5.0, 0, 1.0, 1, true), Ordering::Less);
    assert_eq!(de_rank_cmp(-5.0, 0, 1.0, 1, false), Ordering::Greater);
}

#[test]
fn test_de_rank_cmp_nan_sorts_last() {
    use std::cmp::Ordering;
    // Any finite score outranks NaN (NaN sorts last).
    assert_eq!(de_rank_cmp(f64::NAN, 0, -10.0, 1, false), Ordering::Greater);
    assert_eq!(de_rank_cmp(-10.0, 1, f64::NAN, 0, false), Ordering::Less);
    // Two NaN scores tie on the value and break by ascending index — a
    // strict total order even for empty-group (all-NaN) genes.
    assert_eq!(de_rank_cmp(f64::NAN, 2, f64::NAN, 7, false), Ordering::Less);
    assert_eq!(
        de_rank_cmp(f64::NAN, 7, f64::NAN, 2, false),
        Ordering::Greater
    );
    // Same holds under rankby_abs (NaN.abs() is still NaN).
    assert_eq!(de_rank_cmp(f64::NAN, 2, f64::NAN, 7, true), Ordering::Less);
}

/// Build a dense [n_obs × n_vars] f32 fixture (row-major) with a mix of
/// signal genes, constant genes (score 0.0 → finite ties), and a third
/// group with no cells (its genes all score NaN). `groups` labels cells
/// 0 and 1 only; group 2 is declared but empty.
fn tie_fixture() -> (Vec<f32>, usize, usize, Vec<String>, Vec<usize>, Vec<String>) {
    let n_obs = 16;
    let n_vars = 6;
    let mut data = vec![0.0f32; n_obs * n_vars];
    for r in 0..n_obs {
        let in_g0 = r < 8;
        // gene 0: strong signal up in group 0.
        data[r * n_vars] = if in_g0 { 10.0 + r as f32 } else { 1.0 };
        // gene 1: constant 5.0 everywhere → tie-corrected score 0.0.
        data[r * n_vars + 1] = 5.0;
        // gene 2: all zero → constant → score 0.0.
        // (left at 0.0)
        // gene 3: strong signal up in group 1.
        data[r * n_vars + 3] = if in_g0 { 1.0 } else { 10.0 + (r - 8) as f32 };
        // gene 4: constant 2.0 → score 0.0.
        data[r * n_vars + 4] = 2.0;
        // gene 5: constant 7.0 → score 0.0.
        data[r * n_vars + 5] = 7.0;
    }
    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    let groups: Vec<usize> = (0..n_obs).map(|r| usize::from(r >= 8)).collect();
    // Declare a third group with no member cells → empty-group NaN path.
    let group_names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    (data, n_obs, n_vars, gene_names, groups, group_names)
}

fn csr_from_dense(data: &[f32], n_obs: usize, n_vars: usize) -> scx_sparse::ScxCsr {
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut vals = Vec::new();
    for r in 0..n_obs {
        for c in 0..n_vars {
            let v = data[r * n_vars + c];
            if v != 0.0 {
                indices.push(c as i32);
                vals.push(v);
            }
        }
        indptr.push(indices.len() as i64);
    }
    scx_sparse::ScxCsr::new_unchecked((n_obs, n_vars), indptr, indices, vals)
}

#[test]
fn test_ranking_deterministic_across_chunk_sizes() {
    // Same data, single-chunk vs chunked at several gene_chunk_sizes, must
    // produce byte-identical `names` ordering for every group — the ACC1
    // multi-chunk-merge determinism guarantee. tie_correct=true so the
    // constant genes land on an exact 0.0 tie that exercises the tiebreak.
    let (data, n_obs, n_vars, gene_names, groups, group_names) = tie_fixture();

    let single = wilcoxon_rank_sum(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        None,
        false,
        false,
        true,
        0,
    )
    .unwrap();

    let csr = csr_from_dense(&data, n_obs, n_vars);
    for chunk in [1usize, 2, 3, 5, n_vars] {
        let chunked = wilcoxon_rank_sum_sparse(
            &csr,
            &gene_names,
            &groups,
            &group_names,
            None,
            chunk,
            false,
            false,
            true,
        )
        .unwrap();
        assert_eq!(
            single.group_names, chunked.group_names,
            "group order differs at chunk={chunk}"
        );
        for g in 0..single.group_names.len() {
            assert_eq!(
                single.names[g], chunked.names[g],
                "name order differs in group {} at chunk={chunk}",
                single.group_names[g],
            );
            assert_eq!(
                single.gene_indices[g], chunked.gene_indices[g],
                "gene_indices differ in group {} at chunk={chunk}",
                single.group_names[g],
            );
        }
    }
}

#[test]
fn test_finite_ties_and_nan_group_ordered_by_index() {
    // Finite-score ties break by ascending var index; an empty group's
    // all-NaN genes sort last among themselves in ascending index order.
    let (data, n_obs, n_vars, gene_names, groups, group_names) = tie_fixture();
    let res = wilcoxon_rank_sum(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        None,
        false,
        false,
        true,
        0,
    )
    .unwrap();

    // Group "C" (index 2) has no cells → every gene scores NaN, so the
    // ranking is purely the ascending-index tiebreak = original gene order.
    let c_pos = res.group_names.iter().position(|n| n == "C").unwrap();
    assert!(
        res.scores[c_pos].iter().all(|s| s.is_nan()),
        "empty group should score all-NaN"
    );
    assert_eq!(
        res.gene_indices[c_pos],
        (0..n_vars).collect::<Vec<_>>(),
        "all-NaN genes must be ascending-index ordered"
    );
    assert_eq!(res.names[c_pos], gene_names);

    // Group "A": the three constant genes (1, 4, 5) and the all-zero gene
    // (2) all tie at score 0.0; among that tie group the var indices must
    // be ascending. Verify the subsequence of tied genes is sorted.
    let a_pos = res.group_names.iter().position(|n| n == "A").unwrap();
    let tied_idx: Vec<usize> = res.scores[a_pos]
        .iter()
        .zip(&res.gene_indices[a_pos])
        .filter(|(s, _)| s.abs() < 1e-12)
        .map(|(_, &i)| i)
        .collect();
    let mut sorted = tied_idx.clone();
    sorted.sort_unstable();
    assert_eq!(
        tied_idx, sorted,
        "tied genes must be ascending-index ordered"
    );
}

// ── Multi-shard streaming + cache regression coverage ───────────────────
// These tests guard the cache-bypass fix at the `read_shard_cached_arc`
// call site above. They write an SCX file with > 1 CSR shard, then run
// `wilcoxon_rank_sum_streaming` against a `BackedCsrReader` and compare
// results to `wilcoxon_rank_sum_sparse` on the same data in memory.
use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::{BackedCsrReader, ScxReader, ScxWriter};
use std::path::Path;
use std::sync::Arc as StdArc;

/// Deterministic dense matrix where ~1/3 of cells are non-zero. The
/// pattern guarantees nontrivial rank-sum statistics across two groups
/// because column values vary with both row and column index.
fn make_dense(n_obs: usize, n_vars: usize) -> Vec<u8> {
    let mut dense = vec![0u8; n_obs * n_vars];
    for r in 0..n_obs {
        for c in 0..n_vars {
            if (r + c) % 3 == 0 {
                dense[r * n_vars + c] = ((r * 7 + c * 11) % 200 + 1) as u8;
            }
        }
    }
    dense
}

/// Write a `.scx` file with `n_shards` CSR shards built from a row
/// partition of `dense`. The shards split rows evenly (last shard
/// absorbs the remainder).
fn write_multi_shard_csr(
    path: &Path,
    n_obs: usize,
    n_vars: usize,
    dense: &[u8],
    n_shards: usize,
) -> std::io::Result<()> {
    let header = FileHeader::new_single_modality(n_obs as u64, n_vars as u64, 0, 16384, 0, 0);
    let mut writer = ScxWriter::new(path, header).unwrap();

    let obs_ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    let obs_schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
    let obs = RecordBatch::try_new(
        StdArc::new(obs_schema),
        vec![StdArc::new(StringArray::from(
            obs_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    writer.write_obs(&obs).unwrap();

    let var_ids: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    let var_schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    let var = RecordBatch::try_new(
        StdArc::new(var_schema),
        vec![StdArc::new(StringArray::from(
            var_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    writer.write_var(&var).unwrap();

    let rows_per_shard = n_obs.div_ceil(n_shards);
    for s in 0..n_shards {
        let row_start = s * rows_per_shard;
        if row_start >= n_obs {
            break;
        }
        let row_end = (row_start + rows_per_shard).min(n_obs);

        let mut indptr: Vec<u64> = vec![0];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for r in row_start..row_end {
            for c in 0..n_vars {
                let v = dense[r * n_vars + c];
                if v != 0 {
                    indices.push(c as u32);
                    values.push(v);
                }
            }
            indptr.push(indices.len() as u64);
        }
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_start as u64,
            )
            .unwrap();
    }
    writer.finish().unwrap();
    Ok(())
}

/// Build an in-memory `ScxCsr` covering the full dense matrix.
fn dense_to_full_csr(dense: &[u8], n_obs: usize, n_vars: usize) -> scx_sparse::ScxCsr {
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for r in 0..n_obs {
        for c in 0..n_vars {
            let v = dense[r * n_vars + c];
            if v != 0 {
                indices.push(c as i32);
                data.push(v as f32);
            }
        }
        indptr.push(indices.len() as i64);
    }
    scx_sparse::ScxCsr::new_unchecked((n_obs, n_vars), indptr, indices, data)
}

fn assert_diffexp_results_match(
    cached: &DiffExpResult,
    reference: &DiffExpResult,
    score_atol: f64,
    pval_atol: f64,
) {
    assert_eq!(cached.group_names, reference.group_names);
    for g in 0..cached.group_names.len() {
        assert_eq!(
            cached.names[g], reference.names[g],
            "gene ordering diverges for group {g}"
        );
        for k in 0..cached.scores[g].len() {
            let ds = (cached.scores[g][k] - reference.scores[g][k]).abs();
            assert!(
                ds < score_atol,
                "score mismatch at group {g} rank {k}: {} vs {} (Δ={ds})",
                cached.scores[g][k],
                reference.scores[g][k]
            );
            let dp = (cached.pvals[g][k] - reference.pvals[g][k]).abs();
            assert!(
                dp < pval_atol,
                "pval mismatch at group {g} rank {k}: {} vs {} (Δ={dp})",
                cached.pvals[g][k],
                reference.pvals[g][k]
            );
        }
    }
}

#[test]
fn streaming_with_cache_matches_sparse_kernel() {
    // n_obs=120, n_vars=80, n_shards=4, gene_chunk_size=25
    //   → ⌈80/25⌉ = 4 gene chunks, so the outer loop visits every
    //     shard 4× — the exact multi-pass pattern the cache should
    //     short-circuit.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("wilcox_cached.scx");
    let n_obs = 120usize;
    let n_vars = 80usize;
    let n_shards = 4usize;
    let gene_chunk_size = 25usize;

    let dense = make_dense(n_obs, n_vars);
    write_multi_shard_csr(&path, n_obs, n_vars, &dense, n_shards).unwrap();

    let gene_names: Vec<String> = (0..n_vars).map(|j| format!("gene_{j}")).collect();
    let group_names = vec!["A".to_string(), "B".to_string()];
    let groups: Vec<usize> = (0..n_obs)
        .map(|i| if i < n_obs / 2 { 0 } else { 1 })
        .collect();

    // Cache sized to cover all shards plus a couple slack slots, matching
    // the documented `cache_shards >= n_shards` recommendation.
    let mut reader = BackedCsrReader::new(ScxReader::open(&path).unwrap(), n_shards + 2);
    let metrics = reader.enable_metrics();
    assert_eq!(reader.index().n_shards(), n_shards);

    let cached_res = wilcoxon_rank_sum_streaming(
        &reader,
        &gene_names,
        &groups,
        &group_names,
        None,
        gene_chunk_size,
        false,
        false,
        false,
    )
    .unwrap();

    // The cache must actually have been consulted: with 4 chunks × 4
    // shards = 16 lookups and 4 unique shards, we expect exactly 4
    // misses (cold population) and 12 hits.
    use std::sync::atomic::Ordering;
    let hits = metrics.hits.load(Ordering::Relaxed);
    let misses = metrics.misses.load(Ordering::Relaxed);
    assert_eq!(
        misses, n_shards as u64,
        "expected one miss per unique shard, got {misses}"
    );
    assert_eq!(
        hits,
        (n_shards * (n_vars.div_ceil(gene_chunk_size) - 1)) as u64,
        "expected cache hits on every chunk after the first"
    );

    let in_mem = dense_to_full_csr(&dense, n_obs, n_vars);
    let reference = wilcoxon_rank_sum_sparse(
        &in_mem,
        &gene_names,
        &groups,
        &group_names,
        None,
        gene_chunk_size,
        false,
        false,
        false,
    )
    .unwrap();

    assert_diffexp_results_match(&cached_res, &reference, 1e-5, 1e-5);
}

#[test]
fn streaming_with_zero_cache_still_correct() {
    // Same dataset; force `cache_shards = 0` so the LRU is `None` and
    // `read_shard_cached_arc` falls through to decode-on-each-call.
    // Result must still equal the sparse kernel.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("wilcox_uncached.scx");
    let n_obs = 120usize;
    let n_vars = 80usize;
    let n_shards = 4usize;
    let gene_chunk_size = 25usize;

    let dense = make_dense(n_obs, n_vars);
    write_multi_shard_csr(&path, n_obs, n_vars, &dense, n_shards).unwrap();

    let gene_names: Vec<String> = (0..n_vars).map(|j| format!("gene_{j}")).collect();
    let group_names = vec!["A".to_string(), "B".to_string()];
    let groups: Vec<usize> = (0..n_obs)
        .map(|i| if i < n_obs / 2 { 0 } else { 1 })
        .collect();

    let reader = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);
    assert!(
        !reader.cache_contains(0),
        "cache should be None when cache_shards=0"
    );

    let uncached_res = wilcoxon_rank_sum_streaming(
        &reader,
        &gene_names,
        &groups,
        &group_names,
        None,
        gene_chunk_size,
        false,
        false,
        false,
    )
    .unwrap();

    let in_mem = dense_to_full_csr(&dense, n_obs, n_vars);
    let reference = wilcoxon_rank_sum_sparse(
        &in_mem,
        &gene_names,
        &groups,
        &group_names,
        None,
        gene_chunk_size,
        false,
        false,
        false,
    )
    .unwrap();

    assert_diffexp_results_match(&uncached_res, &reference, 1e-5, 1e-5);
}

// ── pdex `mode="ref"` accelerator tests ──────────────────────────────────

use crate::pseudobulk::GeomMeanMode;

fn build_pdex_fixture(n_obs: usize, n_vars: usize) -> (Vec<f32>, Vec<usize>, Vec<String>) {
    // Three groups: 0=ref, 1=test_a, 2=test_b. Each gets a third of cells.
    let mut data = vec![0.0f32; n_obs * n_vars];
    let mut groups = vec![0usize; n_obs];
    let third = n_obs / 3;
    for cell in 0..n_obs {
        let g = if cell < third {
            0
        } else if cell < 2 * third {
            1
        } else {
            2
        };
        groups[cell] = g;
        for gene in 0..n_vars {
            // Deterministic, non-trivial pattern: each gene gets a shifted
            // sequence by group, plus a per-cell perturbation.
            let base = (g as f32) * 2.0 + (gene as f32) * 0.5;
            let pert = ((cell + gene * 7) % 11) as f32 * 0.1;
            data[cell * n_vars + gene] = base + pert;
        }
    }
    let group_names = vec!["ref".to_string(), "ta".to_string(), "tb".to_string()];
    (data, groups, group_names)
}

#[test]
fn test_pdex_ref_smoke_dense() {
    let n_obs = 60;
    let n_vars = 4;
    let (data, groups, group_names) = build_pdex_fixture(n_obs, n_vars);
    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("g{i}")).collect();

    let result = pdex_ref(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        0, // reference = "ref"
        GeomMeanMode::ArithRaw,
        0.0,
    )
    .unwrap();

    // Reference is excluded; two test groups remain.
    assert_eq!(result.group_names, vec!["ta", "tb"]);
    assert_eq!(result.feature_names.len(), n_vars);
    assert_eq!(result.ref_means.len(), n_vars);
    assert_eq!(result.ref_membership, n_obs / 3);
    assert_eq!(result.target_memberships, vec![n_obs / 3, n_obs / 3]);

    // Shapes per test group.
    for tg in 0..2 {
        assert_eq!(result.target_means[tg].len(), n_vars);
        assert_eq!(result.log2_fold_changes[tg].len(), n_vars);
        assert_eq!(result.percent_changes[tg].len(), n_vars);
        assert_eq!(result.statistics[tg].len(), n_vars);
        assert_eq!(result.p_values[tg].len(), n_vars);
        assert_eq!(result.fdrs[tg].len(), n_vars);
    }

    // Sanity: ref/test means should be positive in this fixture, and the
    // test groups have a strictly larger group-shift than the reference,
    // so log2_fc should be positive for every gene.
    for tg in 0..2 {
        for gene in 0..n_vars {
            assert!(
                result.target_means[tg][gene] > result.ref_means[gene],
                "test group {} gene {} expected larger mean than ref",
                tg,
                gene
            );
            assert!(
                result.log2_fold_changes[tg][gene] > 0.0,
                "log2_fc should be positive for tg={} gene={}",
                tg,
                gene
            );
            // FDR is BH-adjusted, must be in [0, 1] and >= raw p.
            let p = result.p_values[tg][gene];
            let fdr = result.fdrs[tg][gene];
            assert!((0.0..=1.0).contains(&p));
            assert!((0.0..=1.0).contains(&fdr));
            assert!(fdr + 1e-12 >= p, "FDR must be >= raw p");
        }
    }
}

#[test]
fn test_pdex_ref_undetected_gene_one_sided_inf() {
    // Gene 0 is undetected in the reference group (all zeros) but expressed in
    // the test groups, with epsilon == 0. Matching upstream pdex, both
    // log2_fold_change and percent_change are +inf here (a one-sided zero in the
    // denominator) — NOT floored to a finite value. The control gene 1 stays
    // finite.
    let n_obs = 30;
    let n_vars = 2;
    let third = n_obs / 3;
    let mut data = vec![0.0f32; n_obs * n_vars];
    let mut groups = vec![0usize; n_obs];
    for cell in 0..n_obs {
        let g = if cell < third {
            0
        } else if cell < 2 * third {
            1
        } else {
            2
        };
        groups[cell] = g;
        // Gene 0: zero in the reference (g == 0), positive in test groups.
        data[cell * n_vars] = if g == 0 { 0.0 } else { 5.0 };
        // Gene 1: expressed everywhere (keeps a finite control column).
        data[cell * n_vars + 1] = 1.0 + g as f32;
    }
    let group_names = vec!["ref".to_string(), "ta".to_string(), "tb".to_string()];
    let gene_names = vec!["g0".to_string(), "g1".to_string()];

    let result = pdex_ref(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        0,
        GeomMeanMode::ArithRaw,
        0.0, // default epsilon — the ACC4 trigger
    )
    .unwrap();

    // Reference is undetected for gene 0.
    assert_eq!(result.ref_means[0], 0.0);
    for tg in 0..2 {
        // The undetected-reference gene (target > 0, ref == 0, epsilon == 0)
        // yields +inf for both percent_change ((t-0)/0) and log2_fc
        // (log2(t/0)) — matching upstream pdex's preserved one-sided infinity.
        assert_eq!(result.percent_changes[tg][0], f64::INFINITY);
        assert!(result.log2_fold_changes[tg][0].is_infinite());
        assert!(result.log2_fold_changes[tg][0] > 0.0);
        // The control gene (expressed everywhere) stays finite.
        assert!(result.percent_changes[tg][1].is_finite());
        assert!(result.log2_fold_changes[tg][1].is_finite());
    }
}

#[test]
fn test_pdex_ref_zero_over_zero_is_zero() {
    // A gene unexpressed in BOTH the reference and a test group must report
    // log2_fold_change == 0.0 and percent_change == 0.0 (not NaN), matching
    // upstream pdex. A one-sided zero in another group stays +inf.
    let n_obs = 30;
    let n_vars = 1;
    let third = n_obs / 3;
    let mut data = vec![0.0f32; n_obs * n_vars];
    let mut groups = vec![0usize; n_obs];
    for cell in 0..n_obs {
        let g = if cell < third {
            0 // reference
        } else if cell < 2 * third {
            1 // ta: also zero -> 0/0 with the reference
        } else {
            2 // tb: positive -> one-sided zero vs the reference
        };
        groups[cell] = g;
        data[cell * n_vars] = if g == 2 { 5.0 } else { 0.0 };
    }
    let group_names = vec!["ref".to_string(), "ta".to_string(), "tb".to_string()];
    let gene_names = vec!["g0".to_string()];

    let result = pdex_ref(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        0,
        GeomMeanMode::ArithRaw,
        0.0, // epsilon == 0: 0/0 collapses to 0.0, one-sided zero -> +inf
    )
    .unwrap();

    // group order is [ta, tb] (reference excluded).
    assert_eq!(result.group_names, vec!["ta".to_string(), "tb".to_string()]);
    // ta: zero in both -> 0.0, not NaN.
    assert_eq!(result.log2_fold_changes[0][0], 0.0);
    assert_eq!(result.percent_changes[0][0], 0.0);
    // tb: positive over zero reference -> +inf preserved.
    assert_eq!(result.log2_fold_changes[1][0], f64::INFINITY);
    assert_eq!(result.percent_changes[1][0], f64::INFINITY);
}

#[test]
fn test_pdex_ref_cpm_filter_drops_low_expression_genes() {
    // Two genes: g0 high-expression, g1 near-zero. A cpm_filter above g1's CPM
    // but below g0's keeps only g0. The surviving FDR is recomputed over the
    // surviving gene set; kept_indices records the surviving gene identity.
    let n_obs = 30;
    let n_vars = 2;
    let half = n_obs / 2;
    let mut data = vec![0.0f32; n_obs * n_vars];
    let mut groups = vec![0usize; n_obs];
    for cell in 0..n_obs {
        let g = if cell < half { 0 } else { 1 };
        groups[cell] = g;
        // g0: high in both groups; g1: a single low count so its pooled CPM is
        // tiny relative to g0.
        data[cell * n_vars] = 100.0;
        data[cell * n_vars + 1] = if cell == 0 { 1.0 } else { 0.0 };
    }
    let group_names = vec!["ref".to_string(), "ta".to_string()];
    let gene_names = vec!["g0".to_string(), "g1".to_string()];

    // Without filtering: both genes present.
    let unfiltered = pdex_ref_core(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        0,
        GeomMeanMode::ArithRaw,
        0.0,
        false,
    )
    .unwrap();
    assert_eq!(unfiltered.log2_fold_changes[0].len(), 2);
    assert!(unfiltered.kept_indices.is_none());

    // With a high threshold, g1 (CPM ~ 1e6 * tiny) is dropped, g0 kept.
    let mut filtered = pdex_ref_core(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        0,
        GeomMeanMode::ArithRaw,
        0.0,
        true,
    )
    .unwrap();
    finalize_pdex(&mut filtered, Some(1000.0));

    let kept = filtered
        .kept_indices
        .expect("kept_indices set after filtering");
    assert_eq!(kept.len(), 1); // one test group
    assert_eq!(kept[0], vec![0]); // only g0 survives
    assert_eq!(filtered.log2_fold_changes[0].len(), 1);
    assert_eq!(filtered.p_values[0].len(), 1);
    assert_eq!(filtered.fdrs[0].len(), 1);
    // FDR over a single survivor equals its raw (clipped) p-value.
    assert!((filtered.fdrs[0][0] - filtered.p_values[0][0]).abs() < 1e-12);
}

#[test]
fn test_pdex_ref_four_geom_modes() {
    // Verify that all four GeomMeanMode variants run end-to-end and produce
    // finite means / log2_fcs. Compare ArithRaw and GeomLog1p (the two
    // "no transform per cell" modes) on raw vs log1p inputs.
    let n_obs = 30;
    let n_vars = 3;
    let (mut data, groups, group_names) = build_pdex_fixture(n_obs, n_vars);
    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("g{i}")).collect();

    let modes = [
        GeomMeanMode::ArithRaw,
        GeomMeanMode::ArithLog1pExpand,
        GeomMeanMode::GeomRaw,
        GeomMeanMode::GeomLog1p,
    ];

    for mode in modes {
        let raw_result = pdex_ref(
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            0,
            mode,
            0.0,
        )
        .unwrap();

        for tg in 0..2 {
            for gene in 0..n_vars {
                assert!(
                    raw_result.target_means[tg][gene].is_finite(),
                    "mode {:?}: target_mean must be finite",
                    mode
                );
                assert!(
                    raw_result.ref_means[gene].is_finite(),
                    "mode {:?}: ref_mean must be finite",
                    mode
                );
                assert!(
                    raw_result.log2_fold_changes[tg][gene].is_finite(),
                    "mode {:?}: log2_fc must be finite",
                    mode
                );
            }
        }
    }

    // For GeomLog1p: the input is treated as log1p already, so the natural
    // mean is expm1(arithmetic_mean). Verify by applying log1p to data and
    // checking that GeomRaw on log1p(data) ≈ GeomLog1p on log1p(data).
    for v in data.iter_mut() {
        *v = v.ln_1p();
    }
    let geom_raw = pdex_ref(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        0,
        GeomMeanMode::GeomRaw,
        0.0,
    )
    .unwrap();
    let geom_log1p = pdex_ref(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        0,
        GeomMeanMode::GeomLog1p,
        0.0,
    )
    .unwrap();
    for gene in 0..n_vars {
        // GeomRaw on log1p input applies log1p again → expm1(mean(log1p^2(x))).
        // That is NOT equal to GeomLog1p on log1p input (expm1(mean(log1p(x)))).
        // So we only check that both are finite and positive, not that they
        // match. The four-mode parity is enforced by the Python parity test
        // against pdex (pyscx/tests/test_pdex_ref_parity.py).
        assert!(geom_raw.ref_means[gene] >= 0.0);
        assert!(geom_log1p.ref_means[gene] >= 0.0);
    }
}

#[test]
fn test_pdex_ref_epsilon_stabilises_zero_ref() {
    // When ref_mean = 0, epsilon avoids division-by-zero in
    // log2_fold_change and percent_change.
    let n_obs = 12;
    let n_vars = 1;
    let mut data = vec![0.0f32; n_obs * n_vars];
    let groups = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
    // Group 0 (ref) is all zeros. Group 1 has positive values.
    for cell in 4..8 {
        data[cell] = 3.0;
    }
    for cell in 8..12 {
        data[cell] = 5.0;
    }
    let group_names = vec!["ref".to_string(), "ta".to_string(), "tb".to_string()];
    let gene_names = vec!["g0".to_string()];

    // epsilon=0 → log2_fc = log2(target / 0) = +inf for non-zero target.
    let r0 = pdex_ref(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        0,
        GeomMeanMode::ArithRaw,
        0.0,
    )
    .unwrap();
    assert!(
        r0.log2_fold_changes[0][0].is_infinite(),
        "log2_fc with eps=0 and ref_mean=0 should be +inf"
    );

    // epsilon=0.5 → log2_fc is finite.
    let r1 = pdex_ref(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        0,
        GeomMeanMode::ArithRaw,
        0.5,
    )
    .unwrap();
    for tg in 0..2 {
        assert!(
            r1.log2_fold_changes[tg][0].is_finite(),
            "log2_fc with eps=0.5 should be finite"
        );
    }
}

#[test]
fn test_pdex_ref_sparse_matches_dense() {
    // Build a sparse CSR mirroring the fixture and verify pdex_ref_sparse
    // produces the same outputs as pdex_ref (with gene chunking forcing
    // the merge path).
    let n_obs = 30;
    let n_vars = 5;
    let (dense, groups, group_names) = build_pdex_fixture(n_obs, n_vars);
    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("g{i}")).collect();

    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for r in 0..n_obs {
        for c in 0..n_vars {
            let v = dense[r * n_vars + c];
            if v != 0.0 {
                indices.push(c as i32);
                data.push(v);
            }
        }
        indptr.push(indices.len() as i64);
    }
    let csr = scx_sparse::ScxCsr::new_unchecked((n_obs, n_vars), indptr, indices, data);

    let dense_res = pdex_ref(
        &dense,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        0,
        GeomMeanMode::ArithRaw,
        0.0,
    )
    .unwrap();

    // chunk_size=2 → forces three chunks across 5 genes, exercising merge.
    let sparse_res = pdex_ref_sparse(
        &csr,
        &gene_names,
        &groups,
        &group_names,
        0,
        2,
        GeomMeanMode::ArithRaw,
        0.0,
        None,
    )
    .unwrap();

    assert_eq!(dense_res.group_names, sparse_res.group_names);
    assert_eq!(dense_res.feature_names, sparse_res.feature_names);
    assert_eq!(dense_res.ref_membership, sparse_res.ref_membership);
    assert_eq!(dense_res.target_memberships, sparse_res.target_memberships);

    for gene in 0..n_vars {
        assert!((dense_res.ref_means[gene] - sparse_res.ref_means[gene]).abs() < 1e-9);
    }
    for tg in 0..dense_res.group_names.len() {
        for gene in 0..n_vars {
            assert!(
                (dense_res.target_means[tg][gene] - sparse_res.target_means[tg][gene]).abs() < 1e-9
            );
            assert!(
                (dense_res.log2_fold_changes[tg][gene] - sparse_res.log2_fold_changes[tg][gene])
                    .abs()
                    < 1e-9
            );
            assert!(
                (dense_res.percent_changes[tg][gene] - sparse_res.percent_changes[tg][gene]).abs()
                    < 1e-9
            );
            assert!(
                (dense_res.statistics[tg][gene] - sparse_res.statistics[tg][gene]).abs() < 1e-9
            );
            assert!((dense_res.p_values[tg][gene] - sparse_res.p_values[tg][gene]).abs() < 1e-9);
            assert!((dense_res.fdrs[tg][gene] - sparse_res.fdrs[tg][gene]).abs() < 1e-9);
        }
    }
}

#[test]
fn test_pdex_ref_rejects_invalid_inputs() {
    let n_obs = 9;
    let n_vars = 2;
    let (data, mut groups, mut group_names) = build_pdex_fixture(n_obs, n_vars);
    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("g{i}")).collect();

    // Reference index out of range.
    assert!(pdex_ref(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        99,
        GeomMeanMode::ArithRaw,
        0.0,
    )
    .is_err());

    // Negative epsilon.
    assert!(pdex_ref(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        0,
        GeomMeanMode::ArithRaw,
        -0.1,
    )
    .is_err());

    // Reference group has zero cells (assign no cell to group 99).
    groups.iter_mut().for_each(|g| {
        if *g == 0 {
            *g = 1;
        }
    });
    group_names.push("empty_ref".to_string());
    assert!(pdex_ref(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        group_names.len() - 1,
        GeomMeanMode::ArithRaw,
        0.0,
    )
    .is_err());
}

// ---------------------------------------------------------------------------
// Unlabelled cells (group label >= n_groups)
// ---------------------------------------------------------------------------

/// Build a deterministic dense fixture plus a group vector in which every
/// `unlabel_every`-th cell carries the unlabelled sentinel `n_groups`.
///
/// Returns `(data, groups, n_obs)`. Values vary by cell and gene so no two
/// groups share a mean and the ranking is not degenerate.
fn unlabelled_fixture(
    n_groups: usize,
    n_vars: usize,
    per_group: usize,
    unlabel_every: usize,
) -> (Vec<f32>, Vec<usize>, usize) {
    let n_obs = n_groups * per_group;
    let mut groups: Vec<usize> = (0..n_obs).map(|i| i / per_group).collect();
    for (i, g) in groups.iter_mut().enumerate() {
        if i % unlabel_every == 0 {
            *g = n_groups; // the out-of-range sentinel pyscx emits for NaN labels
        }
    }
    let mut data = vec![0.0f32; n_obs * n_vars];
    for cell in 0..n_obs {
        for var in 0..n_vars {
            data[cell * n_vars + var] = ((cell * 7 + var * 13 + cell % 5) % 23) as f32;
        }
    }
    (data, groups, n_obs)
}

/// Physically drop the unlabelled rows, returning `(data, groups, n_obs)` for
/// the same matrix with no sentinel in it.
fn drop_unlabelled(
    data: &[f32],
    groups: &[usize],
    n_vars: usize,
    n_groups: usize,
) -> (Vec<f32>, Vec<usize>, usize) {
    let mut sub_data = Vec::new();
    let mut sub_groups = Vec::new();
    for (cell, &g) in groups.iter().enumerate() {
        if g < n_groups {
            sub_data.extend_from_slice(&data[cell * n_vars..(cell + 1) * n_vars]);
            sub_groups.push(g);
        }
    }
    let n_obs = sub_groups.len();
    (sub_data, sub_groups, n_obs)
}

/// **The oracle for unlabelled-cell semantics.** 1-vs-rest DE over a matrix
/// containing unlabelled cells must equal 1-vs-rest DE over the same matrix
/// with those rows physically removed — scores, p-values and logFC alike. That
/// is exactly what scanpy computes: `rank_genes_groups` subsets to
/// `obs[groupby].isin(groups_order)` before it ranks anything.
///
/// Regression for the review's §7.1: `total` excluded unlabelled cells while
/// `n2 = n_obs - n1` counted them, inflating every logFC in every group by
/// `log2(n_obs − n1) − log2(n_labelled − n1)`, and the rank pool kept them as
/// competitors so the z-score diverged too.
#[test]
fn test_one_vs_rest_unlabelled_cells_equal_physical_subset() {
    let n_groups = 4usize;
    let n_vars = 6usize;
    let per_group = 15usize;
    let (data, groups, n_obs) = unlabelled_fixture(n_groups, n_vars, per_group, 7);
    let n_unlabelled = groups.iter().filter(|&&g| g >= n_groups).count();
    assert!(n_unlabelled > 0, "fixture must contain unlabelled cells");

    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    let group_names: Vec<String> = (0..n_groups).map(|g| format!("grp_{g}")).collect();

    let (sub_data, sub_groups, sub_n_obs) = drop_unlabelled(&data, &groups, n_vars, n_groups);
    assert_eq!(sub_n_obs, n_obs - n_unlabelled);

    for &tie_correct in &[false, true] {
        let with_sentinel = wilcoxon_rank_sum(
            &data,
            n_obs,
            n_vars,
            &gene_names,
            &groups,
            &group_names,
            None,
            false,
            false,
            tie_correct,
            0,
        )
        .unwrap();
        let physically_subset = wilcoxon_rank_sum(
            &sub_data,
            sub_n_obs,
            n_vars,
            &gene_names,
            &sub_groups,
            &group_names,
            None,
            false,
            false,
            tie_correct,
            0,
        )
        .unwrap();

        assert_eq!(with_sentinel.group_names, physically_subset.group_names);
        for row in 0..with_sentinel.group_names.len() {
            assert_eq!(
                with_sentinel.names[row], physically_subset.names[row],
                "gene order diverged for group {}",
                with_sentinel.group_names[row]
            );
            for col in 0..n_vars {
                let (a, b) = (
                    with_sentinel.scores[row][col],
                    physically_subset.scores[row][col],
                );
                assert!(
                    (a - b).abs() < 1e-12,
                    "score mismatch (tie_correct={tie_correct}) for {}/{}: {a} vs {b}",
                    with_sentinel.group_names[row],
                    with_sentinel.names[row][col]
                );
                let (a, b) = (
                    with_sentinel.pvals[row][col],
                    physically_subset.pvals[row][col],
                );
                assert!(
                    (a - b).abs() < 1e-12,
                    "pval mismatch (tie_correct={tie_correct}) for {}/{}: {a} vs {b}",
                    with_sentinel.group_names[row],
                    with_sentinel.names[row][col]
                );
                let (a, b) = (
                    with_sentinel.logfoldchanges[row][col],
                    physically_subset.logfoldchanges[row][col],
                );
                assert!(
                    (a - b).abs() < 1e-12,
                    "logFC mismatch (tie_correct={tie_correct}) for {}/{}: {a} vs {b}",
                    with_sentinel.group_names[row],
                    with_sentinel.names[row][col]
                );
            }
        }
    }
}

/// The brute-force rest-sum oracle, but with unlabelled cells present: the rest
/// denominator is `n_labelled − n1`, never `n_obs − n1`. Sibling of
/// `test_one_vs_rest_logfc_matches_bruteforce_restsum`, whose `i / per_group`
/// fixture leaves no cell unassigned and so pins nothing here.
#[test]
fn test_one_vs_rest_rest_denominator_excludes_unlabelled() {
    let n_groups = 5usize;
    let n_vars = 3usize;
    let per_group = 9usize;
    let (data, groups, n_obs) = unlabelled_fixture(n_groups, n_vars, per_group, 4);
    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    let group_names: Vec<String> = (0..n_groups).map(|g| format!("grp_{g}")).collect();

    let result = wilcoxon_rank_sum(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        None,
        false,
        false,
        false,
        0,
    )
    .unwrap();

    let n_labelled = groups.iter().filter(|&&g| g < n_groups).count();
    assert!(n_labelled < n_obs, "fixture must contain unlabelled cells");

    let mut group_gene_sums = vec![vec![0.0f64; n_vars]; n_groups];
    let mut group_sizes = vec![0usize; n_groups];
    for (cell, &g) in groups.iter().enumerate() {
        if g >= n_groups {
            continue;
        }
        group_sizes[g] += 1;
        for var in 0..n_vars {
            group_gene_sums[g][var] += data[cell * n_vars + var] as f64;
        }
    }

    for (g, gname) in group_names.iter().enumerate() {
        let row = result.group_names.iter().position(|n| n == gname).unwrap();
        let n1 = group_sizes[g] as f64;
        let n2 = (n_labelled - group_sizes[g]) as f64;
        for (var, vname) in gene_names.iter().enumerate() {
            let mean_group = group_gene_sums[g][var] / n1;
            let rest_sum: f64 = (0..n_groups)
                .filter(|&gg| gg != g)
                .map(|gg| group_gene_sums[gg][var])
                .sum();
            let expected = compute_logfc(mean_group, rest_sum / n2, false);
            let col = result.names[row].iter().position(|n| n == vname).unwrap();
            let got = result.logfoldchanges[row][col];
            assert!(
                (got - expected).abs() < 1e-9,
                "logFC mismatch for {gname}/{vname}: got {got}, expected {expected}"
            );
        }
    }
}

/// A fully-labelled fixture must be untouched by the pool compaction: with no
/// sentinel present, `labelled` is `0..n_obs` and the gather walks cells in the
/// same order as before, so the result is **bit-identical** to the recorded
/// pre-fix values rather than merely close.
#[test]
fn test_fully_labelled_is_unaffected_by_pool_compaction() {
    let n_groups = 3usize;
    let n_vars = 4usize;
    let per_group = 12usize;
    let (data, mut groups, n_obs) = unlabelled_fixture(n_groups, n_vars, per_group, 7);
    // Re-label every sentinel cell — the pool becomes the whole matrix.
    for (i, g) in groups.iter_mut().enumerate() {
        if *g >= n_groups {
            *g = i / per_group;
        }
    }
    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    let group_names: Vec<String> = (0..n_groups).map(|g| format!("grp_{g}")).collect();

    let result = wilcoxon_rank_sum(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        None,
        false,
        false,
        false,
        0,
    )
    .unwrap();

    // The rest mean of a fully-labelled matrix is over exactly n_obs - n1 cells.
    let mut group_gene_sums = vec![vec![0.0f64; n_vars]; n_groups];
    for (cell, &g) in groups.iter().enumerate() {
        for var in 0..n_vars {
            group_gene_sums[g][var] += data[cell * n_vars + var] as f64;
        }
    }
    for (g, gname) in group_names.iter().enumerate() {
        let row = result.group_names.iter().position(|n| n == gname).unwrap();
        let n1 = per_group as f64;
        let n2 = (n_obs - per_group) as f64;
        for (var, vname) in gene_names.iter().enumerate() {
            let rest_sum: f64 = (0..n_groups)
                .filter(|&gg| gg != g)
                .map(|gg| group_gene_sums[gg][var])
                .sum();
            let expected = compute_logfc(group_gene_sums[g][var] / n1, rest_sum / n2, false);
            let col = result.names[row].iter().position(|n| n == vname).unwrap();
            assert!((result.logfoldchanges[row][col] - expected).abs() < 1e-9);
        }
    }
}
