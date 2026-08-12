//! CPU↔GPU parity tests for the GPU NB-GLM orchestrator.
//!
//! No external reference exists (pyDESeq2 OOMs in the comprehensive correctness
//! bench), so the CPU `f64` fitter is the reference. These assert
//! per-quantity *relative* tolerances (not bit-equality): a ported special-
//! function set + a safeguarded Illinois root-find diverge in convergence path
//! from the CPU implementation.
//!
//! They are `#[ignore]`d and gated, so on a host with no CUDA device (worker
//! and login nodes) they are reported as ignored rather than as passes that did
//! nothing. The GPU harness (`slurm_scx_gpu_tests.sh`) re-selects them with
//! `--include-ignored` under `SCX_REQUIRE_GPU=1` and runs them on an H100.

use super::gpu_pseudobulk_nb_glm;
use crate::nb_glm::{pseudobulk_nb_glm, DispersionMethod, NbGlmContrast, NbGlmOptions};
use scx_gpu::GpuDevice;

/// Deterministic synthetic Perturb-seq-shaped pseudobulk: `n_genes` genes,
/// `n_sub` samples split into reference/target halves, design `[1, is_target]`.
fn synth(n_genes: usize, n_sub: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    assert!(n_sub % 2 == 0);
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut next = || {
        // xorshift64*
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        (state.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut counts = vec![0.0_f64; n_genes * n_sub];
    for g in 0..n_genes {
        // Base expression and a per-gene effect; some genes flat, some up/down.
        let base = 5.0 + 60.0 * next();
        let lfc = (next() - 0.5) * 2.0; // log fold change in [-1, 1]
        for s in 0..n_sub {
            let is_target = if s >= n_sub / 2 { 1.0 } else { 0.0 };
            let mu = base * (lfc * is_target).exp();
            // Poisson-ish jitter via the uniform stream (deterministic).
            let noise = 0.7 + 0.6 * next();
            counts[g * n_sub + s] = (mu * noise).round().max(0.0);
        }
    }
    // design [n_sub × 2] row-major: [1, is_target]
    let mut design = vec![0.0_f64; n_sub * 2];
    for s in 0..n_sub {
        design[s * 2] = 1.0;
        design[s * 2 + 1] = if s >= n_sub / 2 { 1.0 } else { 0.0 };
    }
    (counts, design)
}

/// Spearman correlation between two slices (rank concordance check).
fn spearman(a: &[f64], b: &[f64]) -> f64 {
    fn ranks(v: &[f64]) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(std::cmp::Ordering::Equal));
        let mut r = vec![0.0_f64; v.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
                j += 1;
            }
            let avg = (i + j) as f64 / 2.0 + 1.0;
            for k in i..=j {
                r[idx[k]] = avg;
            }
            i = j + 1;
        }
        r
    }
    let ra = ranks(a);
    let rb = ranks(b);
    let n = a.len() as f64;
    let ma = ra.iter().sum::<f64>() / n;
    let mb = rb.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..a.len() {
        cov += (ra[i] - ma) * (rb[i] - mb);
        va += (ra[i] - ma).powi(2);
        vb += (rb[i] - mb).powi(2);
    }
    if va == 0.0 || vb == 0.0 {
        return 1.0;
    }
    cov / (va.sqrt() * vb.sqrt())
}

fn run_parity(method: DispersionMethod, n_sub: usize) {
    // The gate lives in each `#[test]` caller, so by here a device is
    // guaranteed and a failure to open one is a real failure, not a skip.
    let dev = GpuDevice::new(0).expect("device gate passed but opening device 0 failed");
    let n_genes = 400;
    let (counts, design) = synth(n_genes, n_sub, 42);
    let options = NbGlmOptions {
        dispersion: method,
        ..Default::default()
    };
    let contrast = NbGlmContrast::Coefficient { index: 1 };

    let cpu = pseudobulk_nb_glm(
        &counts,
        n_genes,
        n_sub,
        &design,
        2,
        None,
        contrast.clone(),
        options.clone(),
    )
    .expect("cpu fit");
    let gpu = gpu_pseudobulk_nb_glm(
        &dev, &counts, n_genes, n_sub, &design, 2, None, contrast, options,
    )
    .expect("gpu fit");

    // Rank concordance — the headline agreement metric.
    let s_lfc = spearman(&cpu.log2_fold_change, &gpu.log2_fold_change);
    assert!(
        s_lfc >= 0.999,
        "log2fc Spearman {s_lfc} < 0.999 (method {method:?}, n_sub {n_sub})"
    );

    // Per-quantity relative tolerances (log2fc tight; dispersion looser).
    let mut max_rel_lfc = 0.0_f64;
    let mut max_rel_disp = 0.0_f64;
    for g in 0..n_genes {
        let (cl, gl) = (cpu.log2_fold_change[g], gpu.log2_fold_change[g]);
        if cl.is_finite() && gl.is_finite() {
            max_rel_lfc = max_rel_lfc.max((cl - gl).abs() / (cl.abs() + 1e-6));
        }
        let (cd, gd) = (cpu.dispersion[g], gpu.dispersion[g]);
        if cd.is_finite() && gd.is_finite() && cd > 1e-6 {
            max_rel_disp = max_rel_disp.max((cd - gd).abs() / (cd.abs() + 1e-6));
        }
        // p-values: compare where both are assessable.
        let (cp, gp) = (cpu.p_value[g], gpu.p_value[g]);
        if cp.is_finite() && gp.is_finite() {
            assert!(
                (cp - gp).abs() <= 1e-4 + 1e-3 * cp.abs(),
                "p_value mismatch g={g}: cpu={cp} gpu={gp}"
            );
        }
    }
    // 2e-3: nvcc --use_fast_math transcendentals + the Illinois root-find's GPU
    // convergence path set a ~1e-4 worst-case floor; the Spearman ≥ 0.999 above
    // is the real ranking guard.
    assert!(
        max_rel_lfc <= 2e-3,
        "max rel log2fc {max_rel_lfc} > 2e-3 (method {method:?})"
    );
    // Dispersion is genuinely ill-conditioned where the root-find is near-flat;
    // a looser bar per §10.
    assert!(
        max_rel_disp <= 5e-2,
        "max rel dispersion {max_rel_disp} > 5e-2 (method {method:?})"
    );
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn gpu_cpu_parity_cox_reid_shrunk_small_nsub() {
    require_gpu_or_skip!();
    run_parity(DispersionMethod::CoxReidShrunk, 6);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn gpu_cpu_parity_cox_reid_mle() {
    require_gpu_or_skip!();
    run_parity(DispersionMethod::CoxReidMle, 8);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn gpu_cpu_parity_moments() {
    require_gpu_or_skip!();
    run_parity(DispersionMethod::Moments, 8);
}
