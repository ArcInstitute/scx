//! Criterion micro-benchmark for the pseudobulk NB-GLM fitter (spec §16, Phase 5).
//!
//! Profiles `scx_accel::pseudobulk_nb_glm` across a small `(n_genes, n_samples,
//! n_features)` grid. Each grid point is benchmarked twice — `Moments` (no
//! Cox–Reid loop) vs `CoxReidShrunk` (the default: per-gene Cox–Reid MLE +
//! trend + EB shrinkage) — so the wall-clock delta isolates the cost of the
//! dispersion sub-problem. Throughput is reported in genes/sec.
//!
//! Run: `cargo bench -p scx-accel --bench nb_glm`
//! Quick smoke: append `-- --quick` (or `--sample-size 10 --measurement-time 2`).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use scx_accel::{pseudobulk_nb_glm, DispersionMethod, NbGlmContrast, NbGlmOptions};

/// Build deterministic synthetic pseudobulk input:
/// gene-major counts `[n_genes × n_samples]` + a row-major design
/// `[n_samples × n_features]` (col 0 intercept, col 1 treatment 0/0…/1/1…, the
/// rest pseudo-random covariates). Half the genes carry a ~2× treatment effect.
fn build_input(n_genes: usize, n_samples: usize, n_features: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let half = n_samples / 2;

    // Design: [1, treatment, covariate_2..covariate_{k-1}].
    let mut design = vec![0.0_f64; n_samples * n_features];
    for s in 0..n_samples {
        let base = s * n_features;
        design[base] = 1.0;
        if n_features > 1 {
            design[base + 1] = if s >= half { 1.0 } else { 0.0 };
        }
        for k in 2..n_features {
            design[base + k] = rng.gen_range(0.0..1.0);
        }
    }

    // Counts: per-gene base level, ~2× in treated samples for even genes, with
    // mild multiplicative noise. Integer-valued f64.
    let mut counts = vec![0.0_f64; n_genes * n_samples];
    for g in 0..n_genes {
        let level = 5.0 + (g % 100) as f64;
        let up = if g % 2 == 0 { 2.0 } else { 1.0 };
        for s in 0..n_samples {
            let trt = if s >= half { up } else { 1.0 };
            let noise = rng.gen_range(0.8..1.2);
            counts[g * n_samples + s] = (level * trt * noise).round().max(0.0);
        }
    }
    (counts, design)
}

fn bench_nb_glm(c: &mut Criterion) {
    // (label, n_genes, n_samples, n_features)
    let grids = [
        ("10kg_50s_3f", 10_000usize, 50usize, 3usize),
        ("30kg_200s_5f", 30_000usize, 200usize, 5usize),
        ("5kg_1000s_10f", 5_000usize, 1_000usize, 10usize),
    ];

    let mut group = c.benchmark_group("pseudobulk_nb_glm");
    group.sample_size(10);

    for (label, n_genes, n_samples, n_features) in grids {
        let (counts, design) = build_input(n_genes, n_samples, n_features);
        group.throughput(Throughput::Elements(n_genes as u64));

        // Default path: Cox–Reid MLE + trend + EB shrinkage.
        group.bench_with_input(
            BenchmarkId::new("cox_reid_shrunk", label),
            &(&counts, &design),
            |b, (counts, design)| {
                b.iter(|| {
                    pseudobulk_nb_glm(
                        counts,
                        n_genes,
                        n_samples,
                        design,
                        n_features,
                        None,
                        NbGlmContrast::Coefficient { index: 1 },
                        NbGlmOptions::default(),
                    )
                    .expect("fit")
                });
            },
        );

        // Moments-only: skips the Cox–Reid dispersion loop. The delta vs the
        // default isolates the dispersion sub-problem's cost.
        let moments_opts = NbGlmOptions {
            dispersion: DispersionMethod::Moments,
            ..Default::default()
        };
        group.bench_with_input(
            BenchmarkId::new("moments", label),
            &(&counts, &design, &moments_opts),
            |b, (counts, design, opts)| {
                b.iter(|| {
                    pseudobulk_nb_glm(
                        counts,
                        n_genes,
                        n_samples,
                        design,
                        n_features,
                        None,
                        NbGlmContrast::Coefficient { index: 1 },
                        (*opts).clone(),
                    )
                    .expect("fit")
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_nb_glm);
criterion_main!(benches);
