//! Microbenchmark for OPT-3.3: the DE `group_gene_sums` region of
//! `scx_accel::diffexp::cpu::wilcoxon_rank_sum`.
//!
//! ```text
//! cargo bench -p scx-accel --bench de_group_sums
//! ```
//!
//! This isolates *exactly* the delta of OPT-3.3 — nothing else in
//! `wilcoxon_rank_sum` changed. Both arms perform the identical per-var parallel
//! gather + an identical O(n_obs) consumer of `values_buf` (a stand-in for the
//! sort/Wilcoxon work that is byte-identical between the two versions and would
//! only add equal noise to both arms). The arms differ only in how the per-group
//! gene sums are produced:
//!
//! - `before`: the old **serial** `O(n_obs·n_vars)` `group_gene_sums` pass plus
//!   the serial `O(n_groups·n_vars)` `total_gene_sum` pass, run before the
//!   parallel per-var loop (which then only gathers and reads the precomputed
//!   sums) — the code OPT-3.3 deleted.
//! - `after`: no pre-pass; per-group sums are accumulated *inside* the gather
//!   each var already performs, and the 1-vs-rest total is a per-var reduction
//!   over `group_sum_buf` — the code OPT-3.3 introduced.
//!
//! Reported difference = (serial pass removed) − (in-gather add + per-var total
//! reduction added). 1-vs-rest is benched (the common DE case): `test_groups`
//! is all groups.

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rayon::prelude::*;

struct Fixture {
    data: Vec<f32>,
    n_obs: usize,
    n_vars: usize,
    n_groups: usize,
    groups: Vec<usize>,
    group_indices: Vec<Vec<usize>>,
    test_groups: Vec<usize>,
}

fn make_fixture(n_obs: usize, n_vars: usize, n_groups: usize) -> Fixture {
    // Deterministic, varied non-negative counts; balanced ascending groups
    // (cell i → group i % n_groups), so group_indices[g] is ascending — matching
    // the production `group_indices` build and OPT-3.3's bit-identity assumption.
    //
    // Every cell is labelled here, which is why the `n2 = n_obs - n1` below is
    // numerically fine. **Do not copy that expression into a kernel**: the
    // production rest denominator is `n_labelled - n1`, and conflating the two
    // is exactly the defect `diffexp::groups` exists to prevent. This file
    // measures the cost of the summation shape, not its semantics.
    let mut data = vec![0.0f32; n_obs * n_vars];
    for cell in 0..n_obs {
        let g = cell % n_groups;
        let base = cell * n_vars;
        for var in 0..n_vars {
            data[base + var] = ((g * 3 + var * 2 + (cell % 31)) % 17) as f32;
        }
    }
    let groups: Vec<usize> = (0..n_obs).map(|i| i % n_groups).collect();
    let mut group_indices: Vec<Vec<usize>> = vec![vec![]; n_groups];
    for (i, &g) in groups.iter().enumerate() {
        group_indices[g].push(i);
    }
    let test_groups: Vec<usize> = (0..n_groups).collect(); // 1-vs-rest
    Fixture {
        data,
        n_obs,
        n_vars,
        n_groups,
        groups,
        group_indices,
        test_groups,
    }
}

/// "before" — serial group_gene_sums + total_gene_sum pre-pass, then a parallel
/// per-var loop that only gathers and reads the precomputed sums.
fn before(f: &Fixture) -> f64 {
    let Fixture {
        data,
        n_obs,
        n_vars,
        n_groups,
        group_indices,
        test_groups,
        ..
    } = f;
    let (n_obs, n_vars, n_groups) = (*n_obs, *n_vars, *n_groups);

    // Serial pre-passes (deleted by OPT-3.3).
    let mut group_gene_sums: Vec<Vec<f64>> = vec![vec![0.0; n_vars]; n_groups];
    for g in 0..n_groups {
        for &cell in &group_indices[g] {
            let base = cell * n_vars;
            for var in 0..n_vars {
                group_gene_sums[g][var] += data[base + var] as f64;
            }
        }
    }
    let mut total_gene_sum = vec![0.0f64; n_vars];
    for sums in &group_gene_sums {
        for (t, &s) in total_gene_sum.iter_mut().zip(sums.iter()) {
            *t += s;
        }
    }

    (0..n_vars)
        .into_par_iter()
        .map_init(
            || vec![0.0f64; n_obs],
            |values_buf, var_idx| {
                for i in 0..n_obs {
                    values_buf[i] = data[i * n_vars + var_idx] as f64;
                }
                // Shared O(n_obs) consumer (stand-in for the identical sort/Wilcoxon work).
                let mut local = values_buf.iter().sum::<f64>();
                for &g in test_groups {
                    let n1 = group_indices[g].len();
                    let n2 = n_obs - n1;
                    let mean_group = group_gene_sums[g][var_idx] / n1 as f64;
                    let rest = total_gene_sum[var_idx] - group_gene_sums[g][var_idx];
                    local += mean_group - rest / n2 as f64;
                }
                local
            },
        )
        .sum()
}

/// "after" — no pre-pass; per-group sums fused into the gather, total reduced per var.
fn after(f: &Fixture) -> f64 {
    let Fixture {
        data,
        n_obs,
        n_vars,
        n_groups,
        groups,
        group_indices,
        test_groups,
        ..
    } = f;
    let (n_obs, n_vars, n_groups) = (*n_obs, *n_vars, *n_groups);

    (0..n_vars)
        .into_par_iter()
        .map_init(
            || (vec![0.0f64; n_obs], vec![0.0f64; n_groups]),
            |(values_buf, group_sum_buf), var_idx| {
                group_sum_buf[..n_groups].iter_mut().for_each(|s| *s = 0.0);
                for i in 0..n_obs {
                    let v = data[i * n_vars + var_idx] as f64;
                    values_buf[i] = v;
                    let g = groups[i];
                    if g < n_groups {
                        group_sum_buf[g] += v;
                    }
                }
                let total: f64 = group_sum_buf[..n_groups].iter().sum();
                // Shared O(n_obs) consumer (identical to `before`).
                let mut local = values_buf.iter().sum::<f64>();
                for &g in test_groups {
                    let n1 = group_indices[g].len();
                    let n2 = n_obs - n1;
                    let mean_group = group_sum_buf[g] / n1 as f64;
                    let rest = total - group_sum_buf[g];
                    local += mean_group - rest / n2 as f64;
                }
                local
            },
        )
        .sum()
}

fn bench(c: &mut Criterion) {
    // (label, n_obs, n_vars, n_groups)
    let shapes = [
        ("pbmc3k_like", 2700usize, 1838usize, 8usize),
        ("mid_50k", 50_000, 2_000, 20),
        ("wide_groups_20k_g100", 20_000, 2_000, 100),
    ];

    let mut group = c.benchmark_group("de_group_sums");
    group.sample_size(40);
    group.measurement_time(Duration::from_secs(8));
    group.warm_up_time(Duration::from_secs(2));

    for (label, n_obs, n_vars, n_groups) in shapes {
        let f = make_fixture(n_obs, n_vars, n_groups);
        // Sanity: both arms produce equal results on this fixture.
        let (b, a) = (before(&f), after(&f));
        assert!(
            (a - b).abs() <= 1e-6 * b.abs().max(1.0),
            "{label}: before={b} after={a} diverged"
        );

        group.throughput(Throughput::Elements((n_obs * n_vars) as u64));
        group.bench_with_input(BenchmarkId::new("before", label), &f, |bn, f| {
            bn.iter(|| black_box(before(black_box(f))))
        });
        group.bench_with_input(BenchmarkId::new("after", label), &f, |bn, f| {
            bn.iter(|| black_box(after(black_box(f))))
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
