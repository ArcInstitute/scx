use super::*;
use crate::diffexp::{pdex_ref, wilcoxon_rank_sum};

// ---- Synthetic dimension-bound guards (pure, no GPU required) ----

#[test]
fn test_validate_gpu_de_dims_rejects_oversized_n_obs() {
    // n_obs above i32::MAX cannot be expressed as 32-bit GPU cell indices.
    let too_many = i32::MAX as usize + 1;
    let err = validate_gpu_de_dims(too_many, 10).unwrap_err();
    assert!(matches!(err, AccelError::InvalidInput(_)), "got {err:?}");
    // A normal case returns the checked product.
    assert_eq!(validate_gpu_de_dims(1_000, 50).unwrap(), 50_000);
}

#[test]
fn test_validate_gpu_de_dims_rejects_product_overflow() {
    // n_obs within i32 but n_obs × n_vars overflows usize.
    let n_obs = i32::MAX as usize;
    let n_vars = usize::MAX / 2;
    let err = validate_gpu_de_dims(n_obs, n_vars).unwrap_err();
    assert!(matches!(err, AccelError::InvalidInput(_)), "got {err:?}");
}

#[test]
fn test_checked_offset_i32_boundary() {
    assert_eq!(
        checked_offset_i32(i32::MAX as usize, "ctx").unwrap(),
        i32::MAX
    );
    let err = checked_offset_i32(i32::MAX as usize + 1, "ctx").unwrap_err();
    assert!(matches!(err, AccelError::InvalidInput(_)), "got {err:?}");
}

#[test]
fn test_validate_pdex_inputs_enforces_dim_guard_without_buffer() {
    // No dense buffer (sparse/streaming) still rejects oversized n_obs.
    let n_obs = i32::MAX as usize + 1;
    let err = validate_pdex_inputs(None, n_obs, 10, 10, n_obs, 3, 0, 1e-6).unwrap_err();
    assert!(matches!(err, AccelError::InvalidInput(_)), "got {err:?}");
    // A dense buffer whose length disagrees with n_obs × n_vars still errors.
    let err = validate_pdex_inputs(Some(99), 10, 10, 10, 10, 3, 0, 1e-6).unwrap_err();
    assert!(matches!(err, AccelError::InvalidInput(_)), "got {err:?}");
    // Consistent small dims pass.
    validate_pdex_inputs(Some(100), 10, 10, 10, 10, 3, 0, 1e-6).unwrap();
}

#[test]
fn test_validate_wilcoxon_inputs_enforces_dim_guard_without_buffer() {
    let n_obs = i32::MAX as usize + 1;
    let err = validate_wilcoxon_inputs(None, n_obs, 10, 10, n_obs).unwrap_err();
    assert!(matches!(err, AccelError::InvalidInput(_)), "got {err:?}");
    let err = validate_wilcoxon_inputs(Some(99), 10, 10, 10, 10).unwrap_err();
    assert!(matches!(err, AccelError::InvalidInput(_)), "got {err:?}");
    validate_wilcoxon_inputs(Some(100), 10, 10, 10, 10).unwrap();
}

/// `(data, n_obs, n_vars, gene_names, groups, group_names, reference)`.
type Fixture = (
    Vec<f32>,
    usize,
    usize,
    Vec<String>,
    Vec<usize>,
    Vec<String>,
    usize,
);

/// Deterministic small fixture: 60 cells × 8 genes, 3 groups (ref +
/// 2 KOs) with engineered fold changes. Mirrors the shape of
/// `pyscx/tests/test_pdex_ref_parity.py::_make_adata`.
fn make_fixture() -> Fixture {
    let n_obs = 60usize;
    let n_vars = 8usize;
    // group 0: ref (20 cells), group 1: KO_A (20 cells), group 2: KO_B (20 cells).
    let groups: Vec<usize> = (0..n_obs).map(|i| i / 20).collect();
    let group_names = vec![
        "non-targeting".to_string(),
        "KO_A".to_string(),
        "KO_B".to_string(),
    ];
    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();

    let mut data = vec![0.0f32; n_obs * n_vars];
    let mut state: u64 = 0xDEADBEEFCAFE;
    let mut next_uniform = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // [0, 1)
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };

    for cell in 0..n_obs {
        let g = groups[cell];
        for gene in 0..n_vars {
            // Base rate ~3 per (cell, gene), perturbed for groups 1/2 on
            // a couple of genes.
            let base = 3.0_f64;
            let mut lambda = base;
            if g == 1 && gene == 2 {
                lambda = base * 2.5;
            }
            if g == 2 && gene == 5 {
                lambda = base * 2.0;
            }
            // Poisson-ish: approximate by sampling N=10 Bernoulli with
            // p = lambda/10 — produces integer counts in [0, 10].
            let p = (lambda / 10.0).clamp(0.0, 1.0);
            let mut k = 0u32;
            for _ in 0..10 {
                if next_uniform() < p {
                    k += 1;
                }
            }
            data[cell * n_vars + gene] = k as f32;
        }
    }
    (data, n_obs, n_vars, gene_names, groups, group_names, 0)
}

/// Per-(group × gene) U statistic must match CPU exactly; p-value must
/// match within `< 1e-9 abs / 1e-6 rel`, target/ref/log2fc within
/// `< 1e-4 rel` (mirrors `test_pdex_ref_parity.py` tolerance).
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_pdex_ref_gpu_dense_matches_cpu() {
    require_gpu_or_skip!();

    let (data, n_obs, n_vars, gene_names, groups, group_names, reference) = make_fixture();
    let mode = crate::pseudobulk::GeomMeanMode::ArithRaw;
    let epsilon = 1e-6;

    let cpu = pdex_ref(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        reference,
        mode,
        epsilon,
    )
    .expect("CPU pdex_ref failed");

    let gpu = pdex_ref_gpu_dense(
        0,
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        reference,
        mode,
        epsilon,
        None,
    )
    .expect("GPU pdex_ref_dense failed");

    assert_eq!(cpu.group_names, gpu.group_names);
    assert_eq!(cpu.feature_names, gpu.feature_names);
    assert_eq!(cpu.ref_membership, gpu.ref_membership);
    assert_eq!(cpu.target_memberships, gpu.target_memberships);
    for tg in 0..cpu.group_names.len() {
        for var in 0..n_vars {
            let u_cpu = cpu.statistics[tg][var];
            let u_gpu = gpu.statistics[tg][var];
            if u_cpu.is_finite() && u_gpu.is_finite() {
                assert!(
                    (u_cpu - u_gpu).abs() < 1e-6,
                    "U mismatch tg={tg} gene={var}: cpu={u_cpu}, gpu={u_gpu}"
                );
            }

            let p_cpu = cpu.p_values[tg][var];
            let p_gpu = gpu.p_values[tg][var];
            assert!(
                (p_cpu - p_gpu).abs() < 1e-9
                    || (p_cpu - p_gpu).abs() / p_cpu.abs().max(1e-12) < 1e-6,
                "p-value mismatch tg={tg} gene={var}: cpu={p_cpu}, gpu={p_gpu}"
            );

            let tm_cpu = cpu.target_means[tg][var];
            let tm_gpu = gpu.target_means[tg][var];
            assert!(
                (tm_cpu - tm_gpu).abs() < 1e-4
                    || (tm_cpu - tm_gpu).abs() / tm_cpu.abs().max(1e-9) < 1e-4,
                "target_mean mismatch tg={tg} gene={var}: cpu={tm_cpu}, gpu={tm_gpu}"
            );

            let l_cpu = cpu.log2_fold_changes[tg][var];
            let l_gpu = gpu.log2_fold_changes[tg][var];
            if l_cpu.is_finite() && l_gpu.is_finite() {
                assert!(
                    (l_cpu - l_gpu).abs() < 1e-4
                        || (l_cpu - l_gpu).abs() / l_cpu.abs().max(1e-9) < 1e-4,
                    "log2_fold_change mismatch tg={tg} gene={var}: cpu={l_cpu}, gpu={l_gpu}"
                );
            }
        }
        for var in 0..n_vars {
            let r_cpu = cpu.ref_means[var];
            let r_gpu = gpu.ref_means[var];
            assert!(
                (r_cpu - r_gpu).abs() < 1e-4
                    || (r_cpu - r_gpu).abs() / r_cpu.abs().max(1e-9) < 1e-4,
                "ref_mean mismatch gene={var}: cpu={r_cpu}, gpu={r_gpu}"
            );
        }
    }
}

/// G10.4 parity: graph-captured per-chunk path produces identical
/// results to the direct per-chunk path under the same fixture.
/// Unlike UMAP (where atomicAdd races create irreducible run-to-run
/// jitter), pdex_ref's kernels are deterministic given fixed input
/// — sort + searchsorted + tie-correct + pvalues — so the two
/// paths should agree to fp32 tolerance bit-for-bit on U / p /
/// log2_fc / means. Any divergence implies the graph-replay path
/// is feeding stale buffer pointers or missing a kernel.
///
/// Uses `set_cuda_graphs_enabled_override` to flip the kill switch
/// in-process so both branches run in the same test invocation
/// (the `SCX_DISABLE_CUDA_GRAPHS=1` env var is `OnceLock`-cached
/// at process start and can't be re-read).
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_pdex_ref_gpu_graph_vs_direct_parity() {
    require_gpu_or_skip!();

    let (data, n_obs, n_vars, gene_names, groups, group_names, reference) = make_fixture();
    let mode = crate::pseudobulk::GeomMeanMode::ArithRaw;
    let epsilon = 1e-6;

    let prev = scx_gpu::set_cuda_graphs_enabled_override(Some(false));
    let direct = pdex_ref_gpu_dense(
        0,
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        reference,
        mode,
        epsilon,
        None,
    )
    .expect("direct pdex_ref_gpu_dense failed");

    scx_gpu::set_cuda_graphs_enabled_override(Some(true));
    let graph = pdex_ref_gpu_dense(
        0,
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        reference,
        mode,
        epsilon,
        None,
    )
    .expect("graph pdex_ref_gpu_dense failed");

    scx_gpu::set_cuda_graphs_enabled_override(prev);

    assert_eq!(direct.group_names, graph.group_names);
    assert_eq!(direct.feature_names, graph.feature_names);

    // The graph path should produce bit-for-bit identical outputs
    // to the direct path (same kernels, same inputs, deterministic
    // sort/searchsort/pvalues — no atomics in this DE family).
    // A modest tolerance accommodates kernel-launch reordering
    // between per_thread_stream and NULL stream, but anything
    // beyond fp32 rounding suggests a real bug.
    for tg in 0..direct.group_names.len() {
        for var in 0..n_vars {
            let u_d = direct.statistics[tg][var];
            let u_g = graph.statistics[tg][var];
            if u_d.is_finite() && u_g.is_finite() {
                assert!(
                    (u_d - u_g).abs() < 1e-6,
                    "U mismatch tg={tg} gene={var}: direct={u_d}, graph={u_g}"
                );
            }
            let p_d = direct.p_values[tg][var];
            let p_g = graph.p_values[tg][var];
            assert!(
                (p_d - p_g).abs() < 1e-9 || (p_d - p_g).abs() / p_d.abs().max(1e-12) < 1e-6,
                "p-value mismatch tg={tg} gene={var}: direct={p_d}, graph={p_g}"
            );
            let tm_d = direct.target_means[tg][var];
            let tm_g = graph.target_means[tg][var];
            assert!(
                (tm_d - tm_g).abs() < 1e-6 || (tm_d - tm_g).abs() / tm_d.abs().max(1e-9) < 1e-6,
                "target_mean mismatch tg={tg} gene={var}: direct={tm_d}, graph={tm_g}"
            );
        }
        for var in 0..n_vars {
            let r_d = direct.ref_means[var];
            let r_g = graph.ref_means[var];
            assert!(
                (r_d - r_g).abs() < 1e-6 || (r_d - r_g).abs() / r_d.abs().max(1e-9) < 1e-6,
                "ref_mean mismatch gene={var}: direct={r_d}, graph={r_g}"
            );
        }
    }
}

/// Wilcoxon 1-vs-rest: per-(group × gene) U and p must match CPU within
/// tolerance. Test compares raw p (not BH) since the per-group sort
/// ordering can differ on ties.
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_wilcoxon_gpu_dense_matches_cpu_one_vs_rest() {
    require_gpu_or_skip!();

    let (data, n_obs, n_vars, gene_names, groups, group_names, _reference) = make_fixture();

    let cpu = wilcoxon_rank_sum(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        None,  // 1-vs-rest
        false, // not log-transformed
        false, // rankby_abs
        true,  // tie_correct
        0,     // gene_index_base
    )
    .expect("CPU wilcoxon failed");

    let gpu = wilcoxon_rank_sum_gpu_dense(
        0,
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
    )
    .expect("GPU wilcoxon failed");

    // Convert per-group sorted lists into hashmap (gene_name → (score, pval))
    // for stable comparison regardless of tie-broken sort order.
    use std::collections::HashMap;
    let group_to_map = |res: &DiffExpResult, g: usize| -> HashMap<String, (f64, f64)> {
        res.names[g]
            .iter()
            .zip(res.scores[g].iter().zip(res.pvals[g].iter()))
            .map(|(n, (&s, &p))| (n.clone(), (s, p)))
            .collect()
    };

    for g in 0..cpu.group_names.len() {
        let cpu_map = group_to_map(&cpu, g);
        let gpu_map = group_to_map(&gpu, g);
        for gene in &gene_names {
            let (s_cpu, p_cpu) = cpu_map.get(gene).copied().unwrap_or((f64::NAN, 1.0));
            let (s_gpu, p_gpu) = gpu_map.get(gene).copied().unwrap_or((f64::NAN, 1.0));
            if s_cpu.is_finite() && s_gpu.is_finite() {
                assert!(
                    (s_cpu - s_gpu).abs() < 1e-6,
                    "score mismatch group={} gene={}: cpu={}, gpu={}",
                    cpu.group_names[g],
                    gene,
                    s_cpu,
                    s_gpu
                );
            }
            assert!(
                (p_cpu - p_gpu).abs() < 1e-9
                    || (p_cpu - p_gpu).abs() / p_cpu.abs().max(1e-12) < 1e-6,
                "pval mismatch group={} gene={}: cpu={}, gpu={}",
                cpu.group_names[g],
                gene,
                p_cpu,
                p_gpu
            );
        }
    }
}

/// G10.5 parity (1-vs-rest): graph-captured wilcoxon path
/// produces identical results to the direct path. Wilcoxon's
/// captureable kernels (scatter / block_sort / tie / searchsorted /
/// ranksum) are deterministic given fixed input — atomicAdd lives
/// only in `gpu_de_pseudobulk_all_groups`, which runs OUTSIDE the
/// captured region — so we can assert fp32-tight tolerance on U /
/// p / score, same shape as
/// `test_pdex_ref_gpu_graph_vs_direct_parity`.
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_wilcoxon_gpu_one_vs_rest_graph_vs_direct_parity() {
    require_gpu_or_skip!();

    let (data, n_obs, n_vars, gene_names, groups, group_names, _reference) = make_fixture();

    let prev = scx_gpu::set_cuda_graphs_enabled_override(Some(false));
    let direct = wilcoxon_rank_sum_gpu_dense(
        0,
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        None,  // 1-vs-rest
        false, // not log-transformed
        false, // rankby_abs
        true,  // tie_correct
    )
    .expect("direct wilcoxon 1-vs-rest failed");

    scx_gpu::set_cuda_graphs_enabled_override(Some(true));
    let graph = wilcoxon_rank_sum_gpu_dense(
        0,
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
    )
    .expect("graph wilcoxon 1-vs-rest failed");

    scx_gpu::set_cuda_graphs_enabled_override(prev);

    assert_eq!(direct.group_names, graph.group_names);

    use std::collections::HashMap;
    let group_to_map = |res: &DiffExpResult, g: usize| -> HashMap<String, (f64, f64)> {
        res.names[g]
            .iter()
            .zip(res.scores[g].iter().zip(res.pvals[g].iter()))
            .map(|(n, (&s, &p))| (n.clone(), (s, p)))
            .collect()
    };

    for g in 0..direct.group_names.len() {
        let d_map = group_to_map(&direct, g);
        let g_map = group_to_map(&graph, g);
        for gene in &gene_names {
            let (s_d, p_d) = d_map.get(gene).copied().unwrap_or((f64::NAN, 1.0));
            let (s_g, p_g) = g_map.get(gene).copied().unwrap_or((f64::NAN, 1.0));
            if s_d.is_finite() && s_g.is_finite() {
                assert!(
                    (s_d - s_g).abs() < 1e-6,
                    "score mismatch group={} gene={}: direct={}, graph={}",
                    direct.group_names[g],
                    gene,
                    s_d,
                    s_g
                );
            }
            assert!(
                (p_d - p_g).abs() < 1e-9 || (p_d - p_g).abs() / p_d.abs().max(1e-12) < 1e-6,
                "pval mismatch group={} gene={}: direct={}, graph={}",
                direct.group_names[g],
                gene,
                p_d,
                p_g
            );
        }
    }
}

/// G10.5 parity (ref-mode): graph-captured wilcoxon path matches
/// direct in ref-mode. Same kernel determinism contract as the
/// 1-vs-rest test above, but exercises ref-mode (`mode=1`) and the
/// per-tg combined-tie + tie_per_group staging path.
///
/// The shared `make_fixture` provides a reference group via its
/// last return value; passing it as `reference: Some(...)` routes
/// the GPU driver into the ref-mode capture path.
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_wilcoxon_gpu_ref_mode_graph_vs_direct_parity() {
    require_gpu_or_skip!();

    let (data, n_obs, n_vars, gene_names, groups, group_names, reference) = make_fixture();
    // `reference` is a `usize` (group index) — make_fixture always
    // supplies one for pdex_ref. For wilcoxon we wrap it in `Some`
    // to drive the ref-mode capture path (mode = 1).

    let prev = scx_gpu::set_cuda_graphs_enabled_override(Some(false));
    let direct = wilcoxon_rank_sum_gpu_dense(
        0,
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        Some(reference),
        false,
        false,
        true,
    )
    .expect("direct wilcoxon ref-mode failed");

    scx_gpu::set_cuda_graphs_enabled_override(Some(true));
    let graph = wilcoxon_rank_sum_gpu_dense(
        0,
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        Some(reference),
        false,
        false,
        true,
    )
    .expect("graph wilcoxon ref-mode failed");

    scx_gpu::set_cuda_graphs_enabled_override(prev);

    assert_eq!(direct.group_names, graph.group_names);

    use std::collections::HashMap;
    let group_to_map = |res: &DiffExpResult, g: usize| -> HashMap<String, (f64, f64)> {
        res.names[g]
            .iter()
            .zip(res.scores[g].iter().zip(res.pvals[g].iter()))
            .map(|(n, (&s, &p))| (n.clone(), (s, p)))
            .collect()
    };

    for g in 0..direct.group_names.len() {
        let d_map = group_to_map(&direct, g);
        let g_map = group_to_map(&graph, g);
        for gene in &gene_names {
            let (s_d, p_d) = d_map.get(gene).copied().unwrap_or((f64::NAN, 1.0));
            let (s_g, p_g) = g_map.get(gene).copied().unwrap_or((f64::NAN, 1.0));
            if s_d.is_finite() && s_g.is_finite() {
                assert!(
                    (s_d - s_g).abs() < 1e-6,
                    "ref-mode score mismatch group={} gene={}: direct={}, graph={}",
                    direct.group_names[g],
                    gene,
                    s_d,
                    s_g
                );
            }
            assert!(
                (p_d - p_g).abs() < 1e-9 || (p_d - p_g).abs() / p_d.abs().max(1e-12) < 1e-6,
                "ref-mode pval mismatch group={} gene={}: direct={}, graph={}",
                direct.group_names[g],
                gene,
                p_d,
                p_g
            );
        }
    }
}

/// Multi-chunk regression: a sparse CSR with `n_vars > gene_chunk_size`
/// must produce the same per-(group × gene) statistics as the dense
/// path. This catches the buf-zeroing bug that the 90×15 dense fixture
/// in `test_pdex_ref_gpu_dense_matches_cpu` is too small to surface:
/// without a per-chunk `buf.fill(0.0)`, non-zero entries from chunk
/// N-1 leak into the zero positions of chunk N and corrupt every U.
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_pdex_ref_gpu_sparse_multi_chunk_matches_dense() {
    require_gpu_or_skip!();

    // 80 cells × 200 genes — chunk_size = 64 forces 4 chunks. Keep
    // the matrix sparse-on-purpose (mostly zeros) so the leak would
    // manifest as non-zero leakage into zero positions.
    let n_obs = 80usize;
    let n_vars = 200usize;
    let groups: Vec<usize> = (0..n_obs).map(|i| i / 40).collect(); // 2 groups of 40
    let group_names = vec!["ref".to_string(), "test".to_string()];
    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("g_{i}")).collect();

    // Deterministic sparse fixture: ~10% density, integer counts in [0, 5].
    let mut data = vec![0.0f32; n_obs * n_vars];
    let mut state: u64 = 0xBEEFCAFEBABE;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for cell in 0..n_obs {
        for gene in 0..n_vars {
            if (next() % 10) == 0 {
                data[cell * n_vars + gene] = (next() % 6) as f32;
            }
        }
    }

    // Build an ScxCsr from the dense matrix.
    let mut indptr: Vec<i64> = Vec::with_capacity(n_obs + 1);
    let mut indices: Vec<i32> = Vec::new();
    let mut sparse_data: Vec<f32> = Vec::new();
    indptr.push(0);
    for cell in 0..n_obs {
        for gene in 0..n_vars {
            let v = data[cell * n_vars + gene];
            if v != 0.0 {
                indices.push(gene as i32);
                sparse_data.push(v);
            }
        }
        indptr.push(indices.len() as i64);
    }
    let csr = scx_sparse::ScxCsr::new_unchecked((n_obs, n_vars), indptr, indices, sparse_data);

    let mode = GeomMeanMode::ArithRaw;
    let epsilon = 1e-6;

    let dense_result = pdex_ref_gpu_dense(
        0,
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        0,
        mode,
        epsilon,
        None,
    )
    .expect("dense GPU pdex_ref failed");

    let sparse_result = pdex_ref_gpu(
        0,
        GpuDeShardInput::Csr(&csr),
        &gene_names,
        &groups,
        &group_names,
        0,
        Some(64),
        mode,
        epsilon,
        None,
    )
    .expect("sparse GPU pdex_ref failed");

    assert_eq!(dense_result.group_names, sparse_result.group_names);
    for tg in 0..dense_result.group_names.len() {
        for var in 0..n_vars {
            let u_d = dense_result.statistics[tg][var];
            let u_s = sparse_result.statistics[tg][var];
            if u_d.is_finite() && u_s.is_finite() {
                assert!(
                    (u_d - u_s).abs() < 1e-6,
                    "multi-chunk U mismatch tg={tg} gene={var}: dense={u_d}, sparse={u_s}"
                );
            }
            let p_d = dense_result.p_values[tg][var];
            let p_s = sparse_result.p_values[tg][var];
            assert!(
                (p_d - p_s).abs() < 1e-9 || (p_d - p_s).abs() / p_d.abs().max(1e-12) < 1e-6,
                "multi-chunk p mismatch tg={tg} gene={var}: dense={p_d}, sparse={p_s}"
            );
        }
    }
}

/// G1.5 regression: `pdex_ref` GPU vs CPU on a fixture large enough to
/// force the tiled merge-sort path. At `n_obs = 12_000` with a 50/50
/// split, `n_ref ≈ 6_000` (fast path) but the test group cells used to
/// rank against ref also exceed 8192 when chained with ref — the
/// combined-tie sort + group sort step actually only sees ≤ n_g cells,
/// so this hits the fast path on ref. To genuinely exercise the
/// multi-tile path inside `gpu_de_block_sort`, we use a fixture where
/// the reference group itself is above 8192 cells (12K total, with
/// 9000 reference cells and 3000 test cells).
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_pdex_ref_gpu_multi_tile_matches_cpu() {
    require_gpu_or_skip!();

    let n_obs = 12_000usize;
    let n_vars = 6usize;
    // Reference = first 9000 cells (above 8192 → multi-tile sort on ref).
    // Test groups = next 1500 + last 1500.
    let groups: Vec<usize> = (0..n_obs)
        .map(|i| {
            if i < 9000 {
                0
            } else if i < 10_500 {
                1
            } else {
                2
            }
        })
        .collect();
    let group_names = vec![
        "non-targeting".to_string(),
        "KO_A".to_string(),
        "KO_B".to_string(),
    ];
    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();

    // Deterministic Poisson-ish counts; perturb gene 2 in KO_A, gene 4 in KO_B.
    let mut data = vec![0.0f32; n_obs * n_vars];
    let mut state: u64 = 0x1357_2468_ACE0_BDF1;
    let mut next_uniform = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for cell in 0..n_obs {
        let g = groups[cell];
        for gene in 0..n_vars {
            let lambda: f64 = if g == 1 && gene == 2 {
                7.5
            } else if g == 2 && gene == 4 {
                6.0
            } else {
                3.0
            };
            let p = (lambda / 10.0).clamp(0.0, 1.0);
            let mut k = 0u32;
            for _ in 0..10 {
                if next_uniform() < p {
                    k += 1;
                }
            }
            data[cell * n_vars + gene] = k as f32;
        }
    }

    let mode = crate::pseudobulk::GeomMeanMode::ArithRaw;
    let epsilon = 1e-6;

    let cpu = pdex_ref(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        0,
        mode,
        epsilon,
    )
    .expect("CPU pdex_ref failed");

    let gpu = pdex_ref_gpu_dense(
        0,
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        0,
        mode,
        epsilon,
        None,
    )
    .expect("GPU pdex_ref_dense failed at n_obs=12000 (n_ref=9000 > 8192)");

    assert_eq!(cpu.group_names, gpu.group_names);
    assert_eq!(cpu.ref_membership, gpu.ref_membership);
    for tg in 0..cpu.group_names.len() {
        for var in 0..n_vars {
            let u_cpu = cpu.statistics[tg][var];
            let u_gpu = gpu.statistics[tg][var];
            if u_cpu.is_finite() && u_gpu.is_finite() {
                // U statistic is integer-valued; allow a tiny float epsilon
                // for the host-side accumulator's f64 rounding.
                assert!(
                    (u_cpu - u_gpu).abs() < 1e-6,
                    "multi-tile U mismatch tg={tg} gene={var}: cpu={u_cpu}, gpu={u_gpu}"
                );
            }
            let p_cpu = cpu.p_values[tg][var];
            let p_gpu = gpu.p_values[tg][var];
            assert!(
                (p_cpu - p_gpu).abs() < 1e-9
                    || (p_cpu - p_gpu).abs() / p_cpu.abs().max(1e-12) < 1e-6,
                "multi-tile p mismatch tg={tg} gene={var}: cpu={p_cpu}, gpu={p_gpu}"
            );
        }
    }
}

/// §8.2 regression: ref-mode Wilcoxon with a reference **smaller** than the
/// largest test group.
///
/// `wilcoxon_chunk_gpu_sequence_v3` sorts twice per chunk — the pool, then (in
/// ref-mode only) each test group's slab. The driver used to size `slab_aux`
/// from `pool_len` alone, so a small reference paired with a test group over
/// `GPU_DE_BLOCK_SORT_CAPACITY` sent the group sort down the multi-tile path
/// with a buffer built for the pool, and `gpu_de_block_sort` rejected it. The
/// realistic trigger is ordinary: `reference="B cell"` against a much larger
/// T-cell group.
///
/// The fixture is deliberately lopsided rather than merely large. `n_vars = 4`
/// pins `chunk_size` to 4 (`resolve_chunk_size` mins against `n_vars`), so the
/// old sizing yields `next_pow2(4 × 512) = 2_048` against the group sort's
/// `4 × 9_000 = 36_000` — a clean `ShapeMismatch`, not a near miss that the
/// power-of-two rounding could absorb. 9_000 also clears the 8_192 single-tile
/// capacity, so the multi-tile path genuinely runs.
///
/// 1-vs-rest cannot reach this: its pool is every labelled cell, which is
/// `≥ n_g_max` by construction. Hence ref-mode here, and the sibling
/// `test_wilcoxon_gpu_dense_matches_cpu_one_vs_rest` for the other mode.
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_wilcoxon_gpu_ref_mode_small_ref_large_group_matches_cpu() {
    require_gpu_or_skip!();

    let n_ref = 512usize;
    let n_test_cells = 9_000usize;
    let n_obs = n_ref + n_test_cells;
    let n_vars = 4usize;

    let groups: Vec<usize> = (0..n_obs).map(|i| usize::from(i >= n_ref)).collect();
    let group_names = vec!["B_cell".to_string(), "T_cell".to_string()];
    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();

    // Deterministic counts (same LCG idiom as the pdex multi-tile fixture);
    // gene 1 is up in the test group, gene 3 down, the rest flat.
    let mut data = vec![0.0f32; n_obs * n_vars];
    let mut state: u64 = 0x2468_ACE0_1357_9BDF;
    let mut next_uniform = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for cell in 0..n_obs {
        let is_test = groups[cell] == 1;
        for gene in 0..n_vars {
            let lambda: f64 = match (is_test, gene) {
                (true, 1) => 7.0,
                (true, 3) => 1.5,
                _ => 3.5,
            };
            let p = (lambda / 10.0).clamp(0.0, 1.0);
            let mut k = 0u32;
            for _ in 0..10 {
                if next_uniform() < p {
                    k += 1;
                }
            }
            data[cell * n_vars + gene] = k as f32;
        }
    }

    let cpu = wilcoxon_rank_sum(
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        Some(0), // reference = the SMALL group
        false,   // not log-transformed
        false,   // rankby_abs
        true,    // tie_correct
        0,       // gene_index_base
    )
    .expect("CPU wilcoxon (ref-mode) failed");

    let gpu = wilcoxon_rank_sum_gpu_dense(
        0,
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        Some(0),
        false,
        false,
        true,
    )
    .expect(
        "GPU wilcoxon ref-mode failed with n_ref=512 < n_g=9000 — \
         slab_aux sized from the pool instead of the largest sort (§8.2)",
    );

    assert_eq!(cpu.group_names, gpu.group_names);
    // Map by gene name: the per-group output is score-sorted, and ties can
    // order differently between host and device.
    use std::collections::HashMap;
    let group_to_map = |res: &DiffExpResult, g: usize| -> HashMap<String, (f64, f64)> {
        res.names[g]
            .iter()
            .zip(res.scores[g].iter().zip(res.pvals[g].iter()))
            .map(|(n, (&s, &p))| (n.clone(), (s, p)))
            .collect()
    };
    for g in 0..cpu.group_names.len() {
        let cpu_map = group_to_map(&cpu, g);
        let gpu_map = group_to_map(&gpu, g);
        for gene in &gene_names {
            let (s_cpu, p_cpu) = cpu_map.get(gene).copied().unwrap_or((f64::NAN, 1.0));
            let (s_gpu, p_gpu) = gpu_map.get(gene).copied().unwrap_or((f64::NAN, 1.0));
            assert!(
                s_cpu.is_finite() && s_gpu.is_finite(),
                "non-finite score group={} gene={gene}: cpu={s_cpu}, gpu={s_gpu}",
                cpu.group_names[g],
            );
            assert!(
                (s_cpu - s_gpu).abs() < 1e-6,
                "score mismatch group={} gene={gene}: cpu={s_cpu}, gpu={s_gpu}",
                cpu.group_names[g],
            );
            assert!(
                (p_cpu - p_gpu).abs() < 1e-9
                    || (p_cpu - p_gpu).abs() / p_cpu.abs().max(1e-12) < 1e-6,
                "pval mismatch group={} gene={gene}: cpu={p_cpu}, gpu={p_gpu}",
                cpu.group_names[g],
            );
        }
    }
}

// ---------------------------------------------------------------------
// G1.7 — dedicated edge-case probes.
//
// Each test builds a tiny synthetic fixture targeting one explicit edge
// case the original G1 spec called out, then asserts CPU↔GPU parity via
// `pdex_ref_gpu_dense` against the CPU `pdex_ref`. Earlier coverage was
// transitive (via Poisson-sampled parity); G1.7 adds standalone probes
// so each edge case has a named failure mode.
// ---------------------------------------------------------------------

/// Shared assertion: CPU↔GPU exact U + tolerance-based p / means. Mirrors
/// the comparison block in `test_pdex_ref_gpu_dense_matches_cpu`.
fn assert_pdex_parity(cpu: &PdexRefResult, gpu: &PdexRefResult, label: &str) {
    assert_eq!(cpu.group_names, gpu.group_names, "{label}: group_names");
    assert_eq!(
        cpu.feature_names, gpu.feature_names,
        "{label}: feature_names"
    );
    assert_eq!(
        cpu.ref_membership, gpu.ref_membership,
        "{label}: ref_membership"
    );
    assert_eq!(
        cpu.target_memberships, gpu.target_memberships,
        "{label}: target_memberships"
    );
    let n_vars = cpu.feature_names.len();
    for tg in 0..cpu.group_names.len() {
        for var in 0..n_vars {
            let u_cpu = cpu.statistics[tg][var];
            let u_gpu = gpu.statistics[tg][var];
            if u_cpu.is_finite() && u_gpu.is_finite() {
                assert!(
                    (u_cpu - u_gpu).abs() < 1e-6,
                    "{label}: U mismatch tg={tg} gene={var}: cpu={u_cpu}, gpu={u_gpu}"
                );
            } else {
                assert_eq!(
                    u_cpu.is_finite(),
                    u_gpu.is_finite(),
                    "{label}: U finite-mask mismatch tg={tg} gene={var}"
                );
            }
            let p_cpu = cpu.p_values[tg][var];
            let p_gpu = gpu.p_values[tg][var];
            assert!(
                (p_cpu - p_gpu).abs() < 1e-9
                    || (p_cpu - p_gpu).abs() / p_cpu.abs().max(1e-12) < 1e-6,
                "{label}: p mismatch tg={tg} gene={var}: cpu={p_cpu}, gpu={p_gpu}"
            );
            let tm_cpu = cpu.target_means[tg][var];
            let tm_gpu = gpu.target_means[tg][var];
            if tm_cpu.is_finite() && tm_gpu.is_finite() {
                assert!(
                    (tm_cpu - tm_gpu).abs() < 1e-9
                        || (tm_cpu - tm_gpu).abs() / tm_cpu.abs().max(1e-9) < 1e-6,
                    "{label}: target_mean mismatch tg={tg} gene={var}: cpu={tm_cpu}, gpu={tm_gpu}"
                );
            }
            let l_cpu = cpu.log2_fold_changes[tg][var];
            let l_gpu = gpu.log2_fold_changes[tg][var];
            if l_cpu.is_finite() && l_gpu.is_finite() {
                assert!(
                    (l_cpu - l_gpu).abs() < 1e-4
                        || (l_cpu - l_gpu).abs() / l_cpu.abs().max(1e-9) < 1e-4,
                    "{label}: log2_fc mismatch tg={tg} gene={var}: cpu={l_cpu}, gpu={l_gpu}"
                );
            }
        }
    }
}

fn run_pdex_pair(
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
    groups: &[usize],
    group_names: &[String],
    reference: usize,
) -> (PdexRefResult, PdexRefResult) {
    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("g_{i}")).collect();
    let mode = crate::pseudobulk::GeomMeanMode::ArithRaw;
    let epsilon = 1e-6;
    let cpu = pdex_ref(
        data,
        n_obs,
        n_vars,
        &gene_names,
        groups,
        group_names,
        reference,
        mode,
        epsilon,
    )
    .expect("CPU pdex_ref failed");
    let gpu = pdex_ref_gpu_dense(
        0,
        data,
        n_obs,
        n_vars,
        &gene_names,
        groups,
        group_names,
        reference,
        mode,
        epsilon,
        None,
    )
    .expect("GPU pdex_ref_dense failed");
    (cpu, gpu)
}

/// Edge case 1 — test group with a single cell.
/// 11 cells × 3 genes; reference = 10 cells, test_A = {cell 10}.
/// Hand-crafted counts: gene 0 puts the singleton above the entire ref
/// distribution; gene 1 puts it below; gene 2 puts it inside.
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_pdex_ref_gpu_group_of_one_cell() {
    require_gpu_or_skip!();
    let n_obs = 11usize;
    let n_vars = 3usize;
    let mut data = vec![0.0f32; n_obs * n_vars];
    // gene 0: ref counts 0..9 (cells 0..10), singleton = 100 (above all).
    // gene 1: ref counts 10..19, singleton = 0 (below all).
    // gene 2: ref counts 1,2,2,3,3,3,4,4,5,5; singleton = 3 (mid-range with ties).
    let ref_g0: [f32; 10] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let ref_g1: [f32; 10] = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
    let ref_g2: [f32; 10] = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0];
    for cell in 0..10 {
        data[cell * n_vars] = ref_g0[cell];
        data[cell * n_vars + 1] = ref_g1[cell];
        data[cell * n_vars + 2] = ref_g2[cell];
    }
    data[10 * n_vars] = 100.0;
    data[10 * n_vars + 1] = 0.0;
    data[10 * n_vars + 2] = 3.0;
    let groups: Vec<usize> = (0..11).map(|i| if i < 10 { 0 } else { 1 }).collect();
    let group_names = vec!["ref".to_string(), "test".to_string()];

    let (cpu, gpu) = run_pdex_pair(&data, n_obs, n_vars, &groups, &group_names, 0);
    // Sanity-check structure before parity assert: 1 test group, 1-cell membership.
    assert_eq!(gpu.target_memberships, vec![1usize]);
    assert_eq!(gpu.ref_membership, 10);
    // All p_values must be finite (no NaN from divide-by-zero sigma at n_g=1).
    for var in 0..n_vars {
        assert!(
            gpu.p_values[0][var].is_finite(),
            "group-of-one p-value must be finite at gene {var}, got {}",
            gpu.p_values[0][var]
        );
    }
    assert_pdex_parity(&cpu, &gpu, "group_of_one_cell");
}

/// Edge case 2 — gene with zero counts in every cell.
/// 20 cells × 4 genes, gene 2 is all-zero. ref={0..9}, test_A={10..19}.
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_pdex_ref_gpu_all_zero_gene() {
    require_gpu_or_skip!();
    let n_obs = 20usize;
    let n_vars = 4usize;
    let mut data = vec![0.0f32; n_obs * n_vars];
    // Non-zero genes: deterministic spread.
    for cell in 0..n_obs {
        data[cell * n_vars] = (cell as f32) % 5.0;
        data[cell * n_vars + 1] = (cell as f32) * 0.5;
        // gene 2 left as 0.0
        data[cell * n_vars + 3] = if cell < 10 { 1.0 } else { 3.0 };
    }
    let groups: Vec<usize> = (0..n_obs).map(|i| if i < 10 { 0 } else { 1 }).collect();
    let group_names = vec!["ref".to_string(), "test".to_string()];

    let (cpu, gpu) = run_pdex_pair(&data, n_obs, n_vars, &groups, &group_names, 0);
    // For gene 2 (all-zero), both means = 0 and the CPU reports U = n1·n2/2,
    // p = 1 (everything tied). Verify GPU matches:
    assert_eq!(gpu.target_means[0][2], 0.0);
    assert_eq!(gpu.ref_means[2], 0.0);
    assert!(
        (gpu.p_values[0][2] - 1.0).abs() < 1e-9,
        "all-zero gene must give p = 1, got {}",
        gpu.p_values[0][2]
    );
    assert_pdex_parity(&cpu, &gpu, "all_zero_gene");
}

/// Edge case 3 — reference smaller than test group (n_ref=3, n_test=30).
/// Exercises the asymmetric `n1 / n2` path in the variance formula.
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_pdex_ref_gpu_ref_smaller_than_test_group() {
    require_gpu_or_skip!();
    let n_obs = 33usize;
    let n_vars = 4usize;
    let mut data = vec![0.0f32; n_obs * n_vars];
    // Deterministic-ish values; doesn't matter much, just want spread.
    let mut state: u64 = 0xCAFEBABE;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) % 20) as f32
    };
    for v in data.iter_mut() {
        *v = next();
    }
    // Reference = first 3 cells; test group A = remaining 30.
    let groups: Vec<usize> = (0..n_obs).map(|i| if i < 3 { 0 } else { 1 }).collect();
    let group_names = vec!["ref".to_string(), "test".to_string()];

    let (cpu, gpu) = run_pdex_pair(&data, n_obs, n_vars, &groups, &group_names, 0);
    assert_eq!(gpu.ref_membership, 3);
    assert_eq!(gpu.target_memberships, vec![30usize]);
    assert_pdex_parity(&cpu, &gpu, "ref_smaller_than_test_group");
}

/// Edge case 4 — test group has identical values for one gene
/// (zero-variance group). Exercises the combined-tie-term path on a
/// pathologically tie-heavy input.
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_pdex_ref_gpu_all_equal_values_in_group() {
    require_gpu_or_skip!();
    let n_obs = 20usize;
    let n_vars = 3usize;
    let mut data = vec![0.0f32; n_obs * n_vars];
    // Gene 0: normal spread.
    // Gene 1: ref has spread; test group A is constant 5.0 (zero-variance).
    // Gene 2: both groups have spread but several values tie with each other.
    for cell in 0..n_obs {
        data[cell * n_vars] = ((cell as f32) % 7.0) + 0.5;
        data[cell * n_vars + 1] = if cell < 10 {
            (cell as f32) % 11.0 // ref: spread including 5.0 a few times
        } else {
            5.0 // test_A: constant
        };
        data[cell * n_vars + 2] = if cell % 3 == 0 { 2.0 } else { 4.0 };
    }
    let groups: Vec<usize> = (0..n_obs).map(|i| if i < 10 { 0 } else { 1 }).collect();
    let group_names = vec!["ref".to_string(), "test".to_string()];

    let (cpu, gpu) = run_pdex_pair(&data, n_obs, n_vars, &groups, &group_names, 0);
    // Sanity-check the test group for gene 1 truly is constant.
    for cell in 10..n_obs {
        assert_eq!(data[cell * n_vars + 1], 5.0);
    }
    assert_pdex_parity(&cpu, &gpu, "all_equal_values_in_group");
}

/// The lazy arm of `pdex_ref_gpu` / `wilcoxon_rank_sum_gpu`
/// matches the dense reference on the same fixture. We feed the same `ScxCsr`
/// through both paths: dense via `pdex_ref_gpu_dense` (which densifies to CSR
/// and routes through v3 — it has not gone through the old host-upload
/// primitive since that path was retired), and lazy via
/// `pdex_ref_gpu(GpuDeShardInput::Lazy(&InMemoryCsrShardSource(&csr)))`
/// (new device-resident
/// scatter). The two paths must agree bit-for-bit on the U statistic
/// (integer-valued) and within the documented p-value / FDR
/// tolerance.
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_gpu_lazy_entry_points_match_dense_reference() {
    require_gpu_or_skip!();

    let n_obs = 60usize;
    let n_vars = 150usize;
    let groups: Vec<usize> = (0..n_obs).map(|i| i / 30).collect(); // 2 groups of 30
    let group_names = vec!["ref".to_string(), "test".to_string()];
    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("g_{i}")).collect();

    // Reproducible sparse fixture (~12% density, integer counts ≤ 5).
    let mut data = vec![0.0f32; n_obs * n_vars];
    let mut state: u64 = 0xDEAD_BEEF_CAFE_F00D;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for cell in 0..n_obs {
        for gene in 0..n_vars {
            if (next() % 9) == 0 {
                data[cell * n_vars + gene] = (next() % 6) as f32;
            }
        }
    }

    // Build the matching ScxCsr.
    let mut indptr: Vec<i64> = Vec::with_capacity(n_obs + 1);
    let mut indices: Vec<i32> = Vec::new();
    let mut sparse_data: Vec<f32> = Vec::new();
    indptr.push(0);
    for cell in 0..n_obs {
        for gene in 0..n_vars {
            let v = data[cell * n_vars + gene];
            if v != 0.0 {
                indices.push(gene as i32);
                sparse_data.push(v);
            }
        }
        indptr.push(indices.len() as i64);
    }
    let csr = scx_sparse::ScxCsr::new_unchecked((n_obs, n_vars), indptr, indices, sparse_data);
    let in_mem = scx_gpu::InMemoryCsrShardSource::new(&csr);

    // ----- pdex_ref -----
    let mode = GeomMeanMode::ArithRaw;
    let epsilon = 1e-6;
    let dense_res = pdex_ref_gpu_dense(
        0,
        &data,
        n_obs,
        n_vars,
        &gene_names,
        &groups,
        &group_names,
        0,
        mode,
        epsilon,
        None,
    )
    .expect("dense pdex_ref failed");
    let lazy_res = pdex_ref_gpu(
        0,
        GpuDeShardInput::Lazy(&in_mem),
        &gene_names,
        &groups,
        &group_names,
        0,
        Some(50), // 3 chunks
        mode,
        epsilon,
        None,
    )
    .expect("lazy pdex_ref failed");

    assert_eq!(dense_res.group_names, lazy_res.group_names);
    for tg in 0..dense_res.group_names.len() {
        for var in 0..n_vars {
            let u_d = dense_res.statistics[tg][var];
            let u_l = lazy_res.statistics[tg][var];
            if u_d.is_finite() && u_l.is_finite() {
                assert!(
                    (u_d - u_l).abs() < 1e-6,
                    "pdex U mismatch tg={tg} gene={var}: dense={u_d}, lazy={u_l}"
                );
            }
            let p_d = dense_res.p_values[tg][var];
            let p_l = lazy_res.p_values[tg][var];
            assert!(
                (p_d - p_l).abs() < 1e-9 || (p_d - p_l).abs() / p_d.abs().max(1e-12) < 1e-6,
                "pdex p mismatch tg={tg} gene={var}: dense={p_d}, lazy={p_l}"
            );
        }
    }

    // ----- wilcoxon_rank_sum (1-vs-rest) -----
    let dense_w = wilcoxon_rank_sum_gpu_dense(
        0,
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
    )
    .expect("dense wilcoxon failed");
    let lazy_w = wilcoxon_rank_sum_gpu(
        0,
        GpuDeShardInput::Lazy(&in_mem),
        &gene_names,
        &groups,
        &group_names,
        None,
        Some(50),
        false,
        false,
        true,
    )
    .expect("lazy wilcoxon failed");

    assert_eq!(dense_w.group_names.len(), lazy_w.group_names.len());
    // wilcoxon_rank_sum sorts within each group; merging restores
    // gene-name ordering. Compare per-(group, gene) by name lookup.
    for g in 0..dense_w.group_names.len() {
        let mut d_map: std::collections::HashMap<String, (f64, f64)> =
            std::collections::HashMap::new();
        for (i, name) in dense_w.names[g].iter().enumerate() {
            d_map.insert(name.clone(), (dense_w.scores[g][i], dense_w.pvals[g][i]));
        }
        for (i, name) in lazy_w.names[g].iter().enumerate() {
            let (d_score, d_pval) = d_map[name];
            let l_score = lazy_w.scores[g][i];
            let l_pval = lazy_w.pvals[g][i];
            if d_score.is_finite() && l_score.is_finite() {
                assert!(
                    (d_score - l_score).abs() < 1e-6,
                    "wilcoxon z mismatch group={} gene={}: dense={}, lazy={}",
                    dense_w.group_names[g],
                    name,
                    d_score,
                    l_score
                );
            }
            assert!(
                (d_pval - l_pval).abs() < 1e-9
                    || (d_pval - l_pval).abs() / d_pval.abs().max(1e-12) < 1e-6,
                "wilcoxon p mismatch group={} gene={}: dense={}, lazy={}",
                dense_w.group_names[g],
                name,
                d_pval,
                l_pval
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Unlabelled cells on the GPU 1-vs-rest path
// ---------------------------------------------------------------------------

/// **The oracle for unlabelled-cell semantics on the GPU**, mirroring
/// `diffexp::cpu_tests::test_one_vs_rest_unlabelled_cells_equal_physical_subset`.
///
/// GPU 1-vs-rest over a matrix containing unlabelled cells (group label
/// `>= n_groups`) must equal GPU 1-vs-rest over the same matrix with those rows
/// physically removed. The v3 driver used to build its rank pool from
/// `0..n_obs` and derive `rest_n = n_obs - n_g` while the pseudobulk sums it
/// subtracted covered labelled groups only — the same numerator/denominator
/// mismatch the CPU kernels had, replicated deliberately for parity.
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_wilcoxon_gpu_one_vs_rest_unlabelled_equals_physical_subset() {
    require_gpu_or_skip!();

    let n_groups = 3usize;
    let n_vars = 6usize;
    let per_group = 20usize;
    let n_obs = n_groups * per_group;

    let mut groups: Vec<usize> = (0..n_obs).map(|i| i / per_group).collect();
    for (i, g) in groups.iter_mut().enumerate() {
        if i % 6 == 0 {
            *g = n_groups; // unlabelled sentinel
        }
    }
    let mut data = vec![0.0f32; n_obs * n_vars];
    for cell in 0..n_obs {
        for var in 0..n_vars {
            data[cell * n_vars + var] = ((cell * 7 + var * 13 + cell % 5) % 23) as f32;
        }
    }

    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    let group_names: Vec<String> = (0..n_groups).map(|g| format!("grp_{g}")).collect();

    let mut sub_data = Vec::new();
    let mut sub_groups = Vec::new();
    for (cell, &g) in groups.iter().enumerate() {
        if g < n_groups {
            sub_data.extend_from_slice(&data[cell * n_vars..(cell + 1) * n_vars]);
            sub_groups.push(g);
        }
    }
    let sub_n_obs = sub_groups.len();
    assert!(sub_n_obs < n_obs, "fixture must contain unlabelled cells");

    let with_sentinel = wilcoxon_rank_sum_gpu_dense(
        0,
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
    )
    .expect("GPU wilcoxon (with unlabelled cells) failed");

    let physically_subset = wilcoxon_rank_sum_gpu_dense(
        0,
        &sub_data,
        sub_n_obs,
        n_vars,
        &gene_names,
        &sub_groups,
        &group_names,
        None,
        false,
        false,
        true,
    )
    .expect("GPU wilcoxon (physical subset) failed");

    use std::collections::HashMap;
    let to_map = |res: &DiffExpResult, g: usize| -> HashMap<String, (f64, f64, f64)> {
        res.names[g]
            .iter()
            .enumerate()
            .map(|(i, n)| {
                (
                    n.clone(),
                    (res.scores[g][i], res.pvals[g][i], res.logfoldchanges[g][i]),
                )
            })
            .collect()
    };

    for (g, group_name) in group_names.iter().enumerate() {
        let a = to_map(&with_sentinel, g);
        let b = to_map(&physically_subset, g);
        for gene in &gene_names {
            let (sa, pa, la) = a[gene];
            let (sb, pb, lb) = b[gene];
            assert!(
                (sa - sb).abs() < 1e-6,
                "score mismatch group={} gene={gene}: sentinel={sa}, subset={sb}",
                group_name
            );
            assert!(
                (pa - pb).abs() < 1e-9 || (pa - pb).abs() / pa.abs().max(1e-12) < 1e-6,
                "pval mismatch group={} gene={gene}: sentinel={pa}, subset={pb}",
                group_name
            );
            assert!(
                (la - lb).abs() < 1e-6,
                "logFC mismatch group={} gene={gene}: sentinel={la}, subset={lb}",
                group_name
            );
        }
    }
}

/// The GPU 1-vs-rest arm must also agree with the *CPU* kernel when unlabelled
/// cells are present — the cross-device parity the CPU-only fix would break.
#[test]
#[ignore = "requires a CUDA GPU"]
fn test_wilcoxon_gpu_unlabelled_matches_cpu() {
    require_gpu_or_skip!();

    let n_groups = 3usize;
    let n_vars = 5usize;
    let per_group = 18usize;
    let n_obs = n_groups * per_group;
    let mut groups: Vec<usize> = (0..n_obs).map(|i| i / per_group).collect();
    for (i, g) in groups.iter_mut().enumerate() {
        if i % 5 == 0 {
            *g = n_groups;
        }
    }
    let mut data = vec![0.0f32; n_obs * n_vars];
    for cell in 0..n_obs {
        for var in 0..n_vars {
            data[cell * n_vars + var] = ((cell * 3 + var * 11) % 19) as f32;
        }
    }
    let gene_names: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    let group_names: Vec<String> = (0..n_groups).map(|g| format!("grp_{g}")).collect();

    let cpu = wilcoxon_rank_sum(
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
    .expect("CPU wilcoxon failed");
    let gpu = wilcoxon_rank_sum_gpu_dense(
        0,
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
    )
    .expect("GPU wilcoxon failed");

    use std::collections::HashMap;
    let to_map = |res: &DiffExpResult, g: usize| -> HashMap<String, (f64, f64)> {
        res.names[g]
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), (res.scores[g][i], res.logfoldchanges[g][i])))
            .collect()
    };
    for (g, group_name) in group_names.iter().enumerate() {
        let c = to_map(&cpu, g);
        let d = to_map(&gpu, g);
        for gene in &gene_names {
            let (sc, lc) = c[gene];
            let (sg, lg) = d[gene];
            assert!(
                (sc - sg).abs() < 1e-6,
                "score mismatch group={} gene={gene}: cpu={sc}, gpu={sg}",
                group_name
            );
            assert!(
                (lc - lg).abs() < 1e-6,
                "logFC mismatch group={} gene={gene}: cpu={lc}, gpu={lg}",
                group_name
            );
        }
    }
}

/// The GPU arm against the **same external reference values** the two host
/// kernels use (ORG-7.21-3).
///
/// The third implementation of the rank-sum, and the one that cannot share a
/// line of code with the other two: it is a `.cu` kernel, so the tie-run walk
/// `for_each_tie_run` unified on the host is a separate CUDA implementation
/// here by necessity. Reference values are therefore the *only* agreement
/// available between them — which is exactly why they had to exist before this
/// arm could be checked against anything but a sibling SCX kernel.
///
/// Same table (`scx_testkit::de_reference`), same fixture, same two
/// conventions. In particular the kernel is handed all 12 rows including the
/// two unlabelled ones while every expected value was computed on the 10
/// labelled rows, so the unlabelled-cell contract is asserted here against
/// scipy rather than against `wilcoxon_rank_sum_gpu_dense`'s own physical-subset
/// run (which `test_wilcoxon_gpu_one_vs_rest_unlabelled_equals_physical_subset`
/// above still covers, and which cannot see a bug both arms share).
#[test]
fn test_wilcoxon_gpu_matches_the_external_reference_values() {
    require_gpu_or_skip!();
    use crate::diffexp::wilcoxon_reference_values as r;

    let data = r::dense_x();
    let gene_names: Vec<String> = (0..r::N_VARS).map(|j| format!("g{j}")).collect();
    let group_names: Vec<String> = (0..r::N_GROUPS).map(|g| format!("grp{g}")).collect();

    for tie_correct in [true, false] {
        let result = wilcoxon_rank_sum_gpu_dense(
            0,
            &data,
            r::N_OBS,
            r::N_VARS,
            &gene_names,
            &r::FIXTURE_GROUPS,
            &group_names,
            None,  // 1-vs-rest
            false, // log_transformed
            false, // rankby_abs
            tie_correct,
        )
        .expect("GPU wilcoxon on the reference fixture");

        for (grp, names_in_group) in result.names.iter().enumerate() {
            for (gene, want_name) in gene_names.iter().enumerate() {
                // Keyed by name, not position: three genes tie at score 0 and
                // their relative order in a sorted result is not a contract.
                let col = names_in_group
                    .iter()
                    .position(|n| n == want_name)
                    .unwrap_or_else(|| panic!("gene {want_name} missing from group {grp}"));
                let (z, p) = (result.scores[grp][col], result.pvals[grp][col]);

                let (want_z, want_p, atol_z, atol_p) = if tie_correct {
                    (
                        r::SCIPY_Z_TIE_CORRECTED[gene][grp],
                        r::SCIPY_P_TIE_CORRECTED[gene][grp],
                        // Not abs=0 like the host arms: the CUDA kernel
                        // accumulates rank sums in a different order, so this
                        // is an f64-reduction-order bound, not a convention
                        // difference. It is still four orders tighter than the
                        // gap between the two tie conventions (7.6e-02), so it
                        // cannot launder one into the other.
                        1e-9,
                        1e-12,
                    )
                } else {
                    (
                        r::SCANPY_Z_UNCORRECTED[gene][grp],
                        r::SCANPY_P_UNCORRECTED[gene][grp],
                        r::Z_UNCORRECTED_ATOL,
                        r::P_UNCORRECTED_ATOL,
                    )
                };
                assert!(
                    (z - want_z).abs() <= atol_z,
                    "gpu gene {want_name} group {grp} tie_correct={tie_correct}: \
                     z {z} vs reference {want_z}"
                );
                assert!(
                    (p - want_p).abs() <= atol_p,
                    "gpu gene {want_name} group {grp} tie_correct={tie_correct}: \
                     p {p} vs reference {want_p}"
                );
            }
        }
    }
}
