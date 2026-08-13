#[cfg(feature = "gpu")]
use super::gpu::harmony_integrate_gpu;
use super::*;

// --- helpers ----------------------------------------------------------

/// Random f32 row-major embeddings (N x d).
fn random_embeddings(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n * d).map(|_| rng.gen::<f32>() - 0.5).collect()
}

/// Two-Gaussian-cluster embeddings with a batch variable that shifts one
/// of the clusters; used to verify that a Harmony iteration reduces the
/// objective.
fn batched_gaussian(n_per: usize, d: usize, seed: u64) -> (Vec<f32>, Vec<u32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut emb = Vec::with_capacity(4 * n_per * d);
    let mut batch = Vec::with_capacity(4 * n_per);
    // Cluster A, batch 0
    for _ in 0..n_per {
        for j in 0..d {
            let mean = if j == 0 { 2.0 } else { 0.0 };
            emb.push(mean + 0.3 * (rng.gen::<f32>() - 0.5));
        }
        batch.push(0);
    }
    // Cluster A, batch 1 (shifted)
    for _ in 0..n_per {
        for j in 0..d {
            let mean = if j == 0 {
                2.0
            } else if j == 1 {
                1.0
            } else {
                0.0
            };
            emb.push(mean + 0.3 * (rng.gen::<f32>() - 0.5));
        }
        batch.push(1);
    }
    // Cluster B, batch 0
    for _ in 0..n_per {
        for j in 0..d {
            let mean = if j == 0 { -2.0 } else { 0.0 };
            emb.push(mean + 0.3 * (rng.gen::<f32>() - 0.5));
        }
        batch.push(0);
    }
    // Cluster B, batch 1 (shifted)
    for _ in 0..n_per {
        for j in 0..d {
            let mean = if j == 0 {
                -2.0
            } else if j == 1 {
                1.0
            } else {
                0.0
            };
            emb.push(mean + 0.3 * (rng.gen::<f32>() - 0.5));
        }
        batch.push(1);
    }
    (emb, batch)
}

fn build_state(emb: &[f32], n: usize, d: usize, labels: Vec<u32>, n_levels: usize) -> HarmonyState {
    let cov = BatchCovariate {
        labels,
        n_levels,
        name: None,
    };
    let config = HarmonyConfig {
        n_clusters: Some(4),
        random_state: 42,
        ..Default::default()
    };
    HarmonyState::new(emb, n, d, &[cov], &config).unwrap()
}

// --- tests ------------------------------------------------------------

#[test]
fn test_kmeans_pp_distinct() {
    let n = 200;
    let d = 5;
    let k = 8;
    let emb = random_embeddings(n, d, 1);
    // Transpose + normalize, mimicking HarmonyState::new prelude.
    let mut z = vec![0f64; d * n];
    for i in 0..n {
        for j in 0..d {
            z[j + i * d] = emb[i * d + j] as f64;
        }
    }
    l2_normalize_columns(&mut z, d, n);
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let y = kmeans_plus_plus(&z, d, n, k, &mut rng);
    // All K centroids should be distinct (d-dim vectors).
    for a in 0..k {
        for b in (a + 1)..k {
            let sa = &y[a * d..(a + 1) * d];
            let sb = &y[b * d..(b + 1) * d];
            let eq = sa.iter().zip(sb).all(|(x, y)| (x - y).abs() < 1e-12);
            assert!(!eq, "centroids {} and {} are identical", a, b);
        }
    }
}

#[test]
fn test_kmeans_config_convergence_reachable() {
    // Regression guard for the pre-1.4 k-means convergence deadlock: the check
    // needs >= 2*window_size objectives, so the default config MUST allow at
    // least that many sub-iterations (4 < 2*3 made it dead code).
    let cfg = HarmonyConfig::default();
    assert!(
        cfg.max_iter_kmeans >= 2 * cfg.window_size,
        "max_iter_kmeans ({}) < 2*window_size ({}) — convergence check can never fire",
        cfg.max_iter_kmeans,
        2 * cfg.window_size
    );
}

#[test]
fn test_kmeans_pp_spreads_across_clusters() {
    // Four tight, well-separated clusters, each a pure unit axis in 4-D.
    // Correct D²-to-ALL k-means++ must place its k seeds on k distinct axes
    // (a candidate coinciding with any chosen centroid has min-distance 0 and
    // is skipped). The pre-1.4 last-centroid-only seeding measured distance to
    // the most-recent centroid alone, so it could re-select an already-covered
    // axis (via a different cell index) and collapse two seeds onto it.
    let d = 4;
    let k = 4;
    let per = 25;
    let n = k * per;
    let mut emb = vec![0f32; n * d];
    for c in 0..k {
        for p in 0..per {
            emb[(c * per + p) * d + c] = 1.0; // pure axis c
        }
    }
    // Transpose + normalize like HarmonyState::new.
    let mut z = vec![0f64; d * n];
    for i in 0..n {
        for j in 0..d {
            z[j + i * d] = emb[i * d + j] as f64;
        }
    }
    l2_normalize_columns(&mut z, d, n);
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let y = kmeans_plus_plus(&z, d, n, k, &mut rng);
    // Every pair of chosen centroids must be near-orthogonal (squared chord
    // distance ≈ 2). Collapsed same-axis seeds would have distance ≈ 0.
    for a in 0..k {
        for b in (a + 1)..k {
            let sa = &y[a * d..(a + 1) * d];
            let sb = &y[b * d..(b + 1) * d];
            let dot: f64 = sa.iter().zip(sb).map(|(x, y)| x * y).sum();
            let dsq = 2.0 * (1.0 - dot);
            assert!(
                dsq > 1.0,
                "seeds {a},{b} too close (dsq={dsq}); k-means++ failed to spread across clusters"
            );
        }
    }
}

#[test]
fn test_soft_assignments_sum_to_one() {
    let n = 150;
    let d = 4;
    let emb = random_embeddings(n, d, 2);
    let labels = (0..n as u32).map(|i| i % 3).collect::<Vec<_>>();
    let s = build_state(&emb, n, d, labels, 3);
    let k = s.k;
    for i in 0..n {
        let mut sum = 0f64;
        for ku in 0..k {
            // R is stored f32 (memory optimisation); promote on read.
            sum += s.r[ku * n + i] as f64;
        }
        // Tolerance widened from 1e-9 to 1e-5 to account for rounding
        // when the f64 softmax output is cast to f32 storage. K rounded
        // f32 values sum to within ~K · eps_f32 (~3e-7 at K=3) of 1.0.
        assert!((sum - 1.0).abs() < 1e-5, "column {} sum={}", i, sum);
    }
}

#[test]
fn test_o_e_consistency() {
    let n = 120;
    let d = 4;
    let emb = random_embeddings(n, d, 3);
    let labels = (0..n as u32).map(|i| i % 4).collect::<Vec<_>>();
    let s = build_state(&emb, n, d, labels, 4);
    let k = s.k;
    let b = s.layout.b;

    // For single covariate: sum_b O[k,b] == sum_i R[k,i]. Both
    // accumulators promote f32 R to f64 on read; equality is exact
    // because compute_o_e iterates cells in the same order (i = 0..n
    // filtered by membership) as this test's sum.
    for ku in 0..k {
        let mut row_sum = 0f64;
        for i in 0..n {
            row_sum += s.r[ku * n + i] as f64;
        }
        let mut o_sum = 0f64;
        for gb in 0..b {
            o_sum += s.o[ku * b + gb];
        }
        assert!(
            (row_sum - o_sum).abs() < 1e-8,
            "cluster {}: row_sum={} o_sum={}",
            ku,
            row_sum,
            o_sum
        );
    }
    // E[k,b] ≈ pr_b[b] * row_sum(R).
    for ku in 0..k {
        let mut row_sum = 0f64;
        for i in 0..n {
            row_sum += s.r[ku * n + i] as f64;
        }
        for gb in 0..b {
            let expected = s.pr_b[gb] * row_sum;
            let got = s.e[ku * b + gb];
            assert!(
                (expected - got).abs() < 1e-8,
                "E[{},{}] expected {} got {}",
                ku,
                gb,
                expected,
                got
            );
        }
    }
}

#[test]
fn test_arrowhead_inverse_matches_lu() {
    // Build a random arrowhead matrix.
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let size = 6; // B' = 5
    let mut mat = vec![0f64; size * size];
    let a: f64 = 5.0 + rng.gen::<f64>();
    mat[0] = a;
    for j in 1..size {
        let c: f64 = rng.gen::<f64>() + 0.1;
        let d: f64 = rng.gen::<f64>() + 2.0;
        mat[j] = c; // row 0
        mat[j * size] = c; // col 0
        mat[j * size + j] = d; // diagonal
    }
    // Ensure symmetric positive-definiteness-ish by bumping diagonal.
    // (Not strictly required for LU.)

    let inv_ah = arrowhead_inverse(&mat, size).unwrap();
    let inv_lu = full_matrix_inverse(&mat, size).unwrap();
    for r in 0..size {
        for c in 0..size {
            let d = inv_ah[r * size + c] - inv_lu[r * size + c];
            assert!(
                d.abs() < 1e-8,
                "mismatch at ({},{}): ah={} lu={}",
                r,
                c,
                inv_ah[r * size + c],
                inv_lu[r * size + c]
            );
        }
    }
}

#[test]
fn test_arrowhead_inverse_rejects_near_zero_diagonal() {
    // Arrowhead matrix whose trailing diagonal block contains a
    // near-zero entry (simulating an empty batch-level/cluster combo
    // under alpha=0). The guard should surface this as
    // NumericalInstability rather than produce a 1/0 inverse.
    let size = 4;
    let mut mat = vec![0f64; size * size];
    mat[0] = 2.0;
    for j in 1..size {
        mat[j] = 0.5; // row 0
        mat[j * size] = 0.5; // col 0
        mat[j * size + j] = 1.0; // diagonal
    }
    // Zero out one trailing diagonal entry.
    mat[2 * size + 2] = 0.0;

    match arrowhead_inverse(&mat, size) {
        Err(AccelError::NumericalInstability(msg)) => {
            assert!(
                msg.contains("arrowhead_inverse"),
                "unexpected error message: {msg}"
            );
        }
        Ok(_) => panic!("expected NumericalInstability, got Ok"),
        Err(other) => panic!("expected NumericalInstability, got {other:?}"),
    }
}

#[test]
fn test_convergence_detection() {
    // Asymptotic decreasing objective: obj → 10 from above, so the
    // relative decrease shrinks over time and eventually crosses epsilon.
    let objs: Vec<f64> = (0..30).map(|i| 10.0 + 1.0 / (i as f64 + 1.0)).collect();
    let mut fired_at = None;
    for i in 1..objs.len() {
        if check_convergence_harmony(&objs[..=i], 1e-2) {
            fired_at = Some(i);
            break;
        }
    }
    assert!(fired_at.is_some(), "Harmony convergence never fired");

    // Increasing objective must NOT fire Harmony convergence (signed).
    let rising: Vec<f64> = (0..10).map(|i| 10.0 + i as f64).collect();
    assert!(!check_convergence_harmony(&rising, 1e-2));

    // k-means (window=3): needs ≥6 samples; an asymptotic sequence
    // eventually has window sums nearly equal, triggering convergence.
    assert!(check_convergence_kmeans(&objs, 3, 1e-3));
    assert!(!check_convergence_kmeans(&objs[..5], 3, 1e-3));
}

#[test]
fn test_batch_pruning() {
    // Construct a state with one tiny batch that should be excluded.
    let n = 100;
    let d = 4;
    let emb = random_embeddings(n, d, 5);
    // 3 levels: level 0 has 1 cell (tiny), levels 1 and 2 split the rest.
    let mut labels = vec![0u32; n];
    labels[0] = 0;
    for (i, lab) in labels.iter_mut().enumerate().take(n).skip(1) {
        *lab = if i % 2 == 0 { 1 } else { 2 };
    }
    let cov = BatchCovariate {
        labels,
        n_levels: 3,
        name: None,
    };
    let config = HarmonyConfig {
        n_clusters: Some(4),
        batch_prop_cutoff: 0.01, // tiny-batch level has avg_R ≈ 1/N which may be > cutoff
        ..Default::default()
    };
    let state = HarmonyState::new(&emb, n, d, &[cov], &config).unwrap();

    // With n_levels=3 and level 0 having 1 cell, the pruning may or may
    // not drop level 0 depending on avg_R. What we can guarantee: the
    // function returns a sensible (kept, active_cov) pair.
    let (kept, active) = prune_batches_for_cluster(&state, 0);
    assert!(active <= 1);
    if active == 1 {
        assert!(kept.len() >= 2); // covariate survival rule
    }

    // Explicit 2-level case where one level fails cutoff.
    let n2 = 100;
    let mut labels2 = vec![1u32; n2];
    labels2[0] = 0; // single cell in level 0
    let cov2 = BatchCovariate {
        labels: labels2,
        n_levels: 2,
        name: None,
    };
    let config2 = HarmonyConfig {
        n_clusters: Some(3),
        batch_prop_cutoff: 0.5, // force level 0 to fail
        ..Default::default()
    };
    let emb2 = random_embeddings(n2, d, 6);
    let state2 = HarmonyState::new(&emb2, n2, d, &[cov2], &config2).unwrap();
    let (kept2, active2) = prune_batches_for_cluster(&state2, 0);
    // Only 1 level survives, covariate drops out.
    assert_eq!(active2, 0);
    assert!(kept2.is_empty());
}

#[test]
fn test_dynamic_lambda() {
    let n = 80;
    let d = 4;
    let emb = random_embeddings(n, d, 7);
    let labels = (0..n as u32).map(|i| i % 3).collect::<Vec<_>>();
    let s = build_state(&emb, n, d, labels, 3);
    let kept: Vec<usize> = (0..s.layout.b).collect();
    let lam = build_dynamic_lambda(&s, 0, &kept);
    assert_eq!(lam.len(), kept.len() + 1);
    assert_eq!(lam[0], 0.0);
    for (j, &gb) in kept.iter().enumerate() {
        let expected = s.config.alpha * s.e[gb];
        assert!((lam[j + 1] - expected).abs() < 1e-12);
    }
}

#[test]
fn test_single_iteration_decreases_objective() {
    let n_per = 80;
    let d = 5;
    let (emb, labels) = batched_gaussian(n_per, d, 13);
    let n = emb.len() / d;
    let cov = BatchCovariate {
        labels,
        n_levels: 2,
        name: None,
    };
    let config = HarmonyConfig {
        n_clusters: Some(4),
        max_iter: 1,
        max_iter_kmeans: 4,
        random_state: 13,
        ..Default::default()
    };
    let mut state = HarmonyState::new(&emb, n, d, &[cov], &config).unwrap();
    // Record objective before any update and after one k-means sub-loop.
    let obj0 = state.compute_objective();
    state.update_r();
    let obj1 = state.compute_objective();
    assert!(
        obj1 <= obj0 + 1e-6,
        "objective did not decrease: obj0={} obj1={}",
        obj0,
        obj1
    );
}

#[test]
fn test_determinism_same_seed() {
    let n = 200;
    let d = 5;
    let emb = random_embeddings(n, d, 21);
    let labels: Vec<u32> = (0..n as u32).map(|i| i % 3).collect();
    let cov = BatchCovariate {
        labels: labels.clone(),
        n_levels: 3,
        name: None,
    };
    let config = HarmonyConfig {
        n_clusters: Some(5),
        max_iter: 2,
        random_state: 99,
        ..Default::default()
    };
    let r1 = harmony_integrate(&emb, n, d, std::slice::from_ref(&cov), &config).unwrap();
    let r2 = harmony_integrate(&emb, n, d, std::slice::from_ref(&cov), &config).unwrap();
    assert_eq!(r1.z_corrected.len(), r2.z_corrected.len());
    for (a, b) in r1.z_corrected.iter().zip(r2.z_corrected.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "z_corrected diverged");
    }
}

#[test]
fn test_multi_covariate_runs() {
    let n = 300;
    let d = 6;
    let emb = random_embeddings(n, d, 33);
    let labels_a: Vec<u32> = (0..n as u32).map(|i| i % 3).collect();
    let labels_b: Vec<u32> = (0..n as u32).map(|i| (i / 3) % 4).collect();
    let cov_a = BatchCovariate {
        labels: labels_a,
        n_levels: 3,
        name: Some("donor".into()),
    };
    let cov_b = BatchCovariate {
        labels: labels_b,
        n_levels: 4,
        name: Some("tech".into()),
    };
    let config = HarmonyConfig {
        n_clusters: Some(6),
        max_iter: 2,
        random_state: 55,
        ..Default::default()
    };
    let result = harmony_integrate(&emb, n, d, &[cov_a, cov_b], &config).unwrap();
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_pcs, d);
    assert_eq!(result.z_corrected.len(), n * d);
    assert!(result.z_corrected.iter().all(|v| v.is_finite()));
}

// ── GPU tests ─────────────────────────────────────────────────────
// `#[ignore]`d and gated, so a machine without a CUDA driver reports them as
// ignored rather than as passes that did nothing. The GPU harness re-selects
// them with `--include-ignored` under `SCX_REQUIRE_GPU=1`.

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_gpu_harmony_shape_matches_cpu() {
    require_gpu_or_skip!();
    let n = 200;
    let d = 6;
    let emb = random_embeddings(n, d, 100);
    let labels: Vec<u32> = (0..n as u32).map(|i| i % 3).collect();
    let cov = BatchCovariate {
        labels,
        n_levels: 3,
        name: None,
    };
    let config = HarmonyConfig {
        n_clusters: Some(5),
        max_iter: 2,
        random_state: 7,
        ..Default::default()
    };
    let result = harmony_integrate_gpu(0, &emb, n, d, std::slice::from_ref(&cov), &config).unwrap();
    assert_eq!(result.n_obs, n);
    assert_eq!(result.n_pcs, d);
    assert_eq!(result.z_corrected.len(), n * d);
    assert!(result.z_corrected.iter().all(|v| v.is_finite()));
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_gpu_vs_cpu_per_pc_correlation() {
    require_gpu_or_skip!();
    let n_per = 80;
    let d = 5;
    let (emb, labels) = batched_gaussian(n_per, d, 123);
    let n = emb.len() / d;
    let cov = BatchCovariate {
        labels,
        n_levels: 2,
        name: None,
    };
    let config = HarmonyConfig {
        n_clusters: Some(4),
        max_iter: 3,
        random_state: 11,
        ..Default::default()
    };
    let cpu = harmony_integrate(&emb, n, d, std::slice::from_ref(&cov), &config).unwrap();
    let gpu = harmony_integrate_gpu(0, &emb, n, d, std::slice::from_ref(&cov), &config).unwrap();

    // Compare per-PC Pearson correlation (f32 rounding → expect ~0.99+).
    for pc in 0..d {
        let mut x: Vec<f64> = Vec::with_capacity(n);
        let mut y: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            x.push(cpu.z_corrected[i * d + pc]);
            y.push(gpu.z_corrected[i * d + pc]);
        }
        let mx: f64 = x.iter().sum::<f64>() / n as f64;
        let my: f64 = y.iter().sum::<f64>() / n as f64;
        let mut num = 0f64;
        let mut dx = 0f64;
        let mut dy = 0f64;
        for i in 0..n {
            let a = x[i] - mx;
            let b = y[i] - my;
            num += a * b;
            dx += a * a;
            dy += b * b;
        }
        let r = num / (dx.sqrt() * dy.sqrt() + 1e-30);
        // Either strong correlation OR both PCs are near-constant (dx or dy ~ 0).
        if dx > 1e-8 && dy > 1e-8 {
            assert!(r > 0.95, "PC {pc}: r={r}");
        }
    }
}

/// G10.2: graph-capture path matches the direct per-sub-iter
/// dispatch path to within the per-PC Pearson-r tolerance the
/// pre-existing `test_gpu_vs_cpu_per_pc_correlation` uses as the
/// GPU correctness contract.
///
/// Harmony's GPU path is already non-bit-exact across runs (the
/// docstring on `harmony_integrate_gpu` calls out f32 rounding +
/// atomic-ordering nondeterminism in the O/E updates), so a
/// fixed-tolerance coordinate-wise check would be brittle. Using
/// the same Pearson-r ≥ 0.95 contract as the CPU↔GPU test gives
/// us a noise-aware bound that catches "graph replay produced a
/// fundamentally different embedding" while accepting legitimate
/// jitter.
///
/// We flip the kill switch in-process via
/// `set_cuda_graphs_enabled_override` so both paths run in the
/// same test process.
#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_gpu_harmony_graph_vs_direct_parity() {
    require_gpu_or_skip!();
    let n_per = 80;
    let d = 5;
    let (emb, labels) = batched_gaussian(n_per, d, 123);
    let n = emb.len() / d;
    let cov = BatchCovariate {
        labels,
        n_levels: 2,
        name: None,
    };
    let config = HarmonyConfig {
        n_clusters: Some(4),
        max_iter: 3,
        random_state: 11,
        ..Default::default()
    };

    let prev = scx_gpu::set_cuda_graphs_enabled_override(Some(false));
    let direct = harmony_integrate_gpu(0, &emb, n, d, std::slice::from_ref(&cov), &config).unwrap();

    scx_gpu::set_cuda_graphs_enabled_override(Some(true));
    let graph = harmony_integrate_gpu(0, &emb, n, d, std::slice::from_ref(&cov), &config).unwrap();

    scx_gpu::set_cuda_graphs_enabled_override(prev);

    assert_eq!(direct.z_corrected.len(), graph.z_corrected.len());
    assert!(graph.z_corrected.iter().all(|v| v.is_finite()));

    // Per-PC Pearson r between graph and direct paths.
    for pc in 0..d {
        let mut x: Vec<f64> = Vec::with_capacity(n);
        let mut y: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            x.push(direct.z_corrected[i * d + pc]);
            y.push(graph.z_corrected[i * d + pc]);
        }
        let mx: f64 = x.iter().sum::<f64>() / n as f64;
        let my: f64 = y.iter().sum::<f64>() / n as f64;
        let mut num = 0f64;
        let mut dx = 0f64;
        let mut dy = 0f64;
        for i in 0..n {
            let a = x[i] - mx;
            let b = y[i] - my;
            num += a * b;
            dx += a * a;
            dy += b * b;
        }
        let r = num / (dx.sqrt() * dy.sqrt() + 1e-30);
        if dx > 1e-8 && dy > 1e-8 {
            assert!(
                r > 0.95,
                "graph vs direct PC {pc}: r={r:.4} (expected > 0.95). \
                     Suggests sub-iter capture/replay diverges from the \
                     direct kernel sequence — likely a stream-binding or \
                     buffer-pointer bug, NOT atomic-race jitter (which \
                     stays within the GPU correctness contract)."
            );
        }
    }
}

/// Regression test for the hoisted `sub_graph` lifecycle: across
/// a multi-outer-iter Harmony run, the k-means sub-iter capture
/// must fire exactly ONCE (not once per outer iter). Pre-G10.2
/// hoist behaviour was `max_iter` captures; post-hoist should be
/// exactly 1 (warm-up runs first, capture fires on the second
/// sub-iter, every other sub-iter across all later outer iters
/// replays).
#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_gpu_harmony_captures_once_across_outer_iters() {
    require_gpu_or_skip!();
    let n_per = 80;
    let d = 5;
    let (emb, labels) = batched_gaussian(n_per, d, 123);
    let n = emb.len() / d;
    let cov = BatchCovariate {
        labels,
        n_levels: 2,
        name: None,
    };
    // max_iter ≥ 2 is the load-bearing parameter: a single outer
    // iter wouldn't distinguish hoisted vs per-iter capture.
    let config = HarmonyConfig {
        n_clusters: Some(4),
        max_iter: 3,
        random_state: 11,
        ..Default::default()
    };

    let prev = scx_gpu::set_cuda_graphs_enabled_override(Some(true));
    super::gpu::HARMONY_CAPTURE_ATTEMPTS.store(0, std::sync::atomic::Ordering::Relaxed);
    let _ = harmony_integrate_gpu(0, &emb, n, d, std::slice::from_ref(&cov), &config).unwrap();
    let captures = super::gpu::HARMONY_CAPTURE_ATTEMPTS.load(std::sync::atomic::Ordering::Relaxed);
    scx_gpu::set_cuda_graphs_enabled_override(prev);

    assert_eq!(
        captures, 1,
        "Harmony k-means sub-iter capture fired {captures} times across \
             {} outer iters; expected exactly 1. A count equal to max_iter \
             means `sub_graph` is still being re-declared inside the outer \
             loop instead of hoisted above it.",
        config.max_iter
    );
}

#[cfg(feature = "gpu")]
#[test]
fn test_gpu_memory_estimate_reasonable() {
    use scx_gpu::gpu_harmony_memory_bytes;
    let bytes = gpu_harmony_memory_bytes(10_000, 30, 50, 3, 1);
    // Order-of-magnitude: ~few MB, well under 1 GB.
    assert!(bytes > 1_000_000);
    assert!(bytes < 1_000_000_000);
}

/// The refusal message is the entire remedy the caller gets — `device="auto"`
/// does not re-run on CPU — so pin that it actually names the shortfall and the
/// way out, not just that it is non-empty. Runs on a CPU host under
/// `--features gpu`: the message builder is pure arithmetic and formatting.
#[cfg(feature = "gpu")]
#[test]
fn test_gpu_vram_message_names_the_shortfall_and_the_remedy() {
    // 8M cells × 50 PCs × 100 clusters against 6.1 GB free of 79.1 GB.
    let msg = super::gpu::harmony_vram_message(
        0,
        8_000_000,
        50,
        100,
        4,
        1,
        6_100_000_000,
        79_100_000_000,
    );
    // The three numbers a user needs to act on.
    assert!(
        msg.contains("8000000 cells × 50 PCs × 100 clusters"),
        "{msg}"
    );
    // Pins the figure `docs/gpu-setup.md` quotes for these exact inputs. Not
    // redundant with the shape assertion: the doc example already drifted once
    // (14.2 GB, a number I estimated rather than rendered) and moved again when
    // the bound was tightened (11.2 → 11.3 → 11.4). Tightening it further *should*
    // fail here, so the doc is updated alongside instead of going stale.
    assert!(
        msg.contains("≥11.4 GB"),
        "docs/gpu-setup.md quotes ≥11.4 GB for these inputs — update both together. Got: {msg}"
    );
    assert!(msg.contains("6.1 GB of 79.1 GB"), "{msg}");
    assert!(msg.contains("GPU 0"), "{msg}");
    // `≥`, not `=`: the estimate excludes transient scratch.
    assert!(msg.contains("≥"), "{msg}");
    // And what to do about it. Without this the message is a diagnosis with no
    // prescription, which is what the raw cudarc error already was.
    assert!(msg.contains(r#"device="cpu""#), "{msg}");
    assert!(msg.contains("n_clusters"), "{msg}");
    // Does not restate the op name — pyscx prefixes it, and the note form
    // appends this to an allocation error that already names its buffer.
    assert!(!msg.contains("harmony_integrate"), "{msg}");
}

/// The VRAM pre-flight (§8.10) must not refuse a run that fits.
///
/// It is the one way this change could regress a working setup: a refusal is a
/// hard error with no CPU fallback behind it, so an estimate that over-counted
/// would turn ordinary Harmony calls into failures. A run this small on any
/// real card must sail through — and it exercises the *live* probe against
/// actual `cuMemGetInfo`, not the arithmetic, which `gpu_harmony_fits`'
/// own unit tests already cover on a CPU host.
#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_gpu_preflight_admits_a_run_that_fits() {
    require_gpu_or_skip!();
    let n = 500;
    let d = 8;
    let emb = random_embeddings(n, d, 11);
    let labels: Vec<u32> = (0..n as u32).map(|i| i % 4).collect();
    let cov = BatchCovariate {
        labels,
        n_levels: 4,
        name: None,
    };
    let config = HarmonyConfig {
        n_clusters: Some(6),
        max_iter: 2,
        random_state: 3,
        ..Default::default()
    };
    let result = harmony_integrate_gpu(0, &emb, n, d, std::slice::from_ref(&cov), &config)
        .expect("the VRAM pre-flight must not refuse a run this size on a real GPU");
    assert_eq!(result.z_corrected.len(), n * d);
    assert!(result.z_corrected.iter().all(|v| v.is_finite()));
}
