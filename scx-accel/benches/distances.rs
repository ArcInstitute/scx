//! Criterion microbenchmarks for `scx_accel::eval_metrics::distances`.
//!
//! Run all distance-kernel benches:
//! ```text
//! cargo bench -p scx-accel --bench distances
//! ```
//!
//! Filter by group/id (criterion regex):
//! ```text
//! cargo bench -p scx-accel --bench distances -- 'cross/euclidean'
//! cargo bench -p scx-accel --bench distances -- 'self/cosine'
//! cargo bench -p scx-accel --bench distances -- 'f32'
//! ```
//!
//! Parameter grid per the Phase 4 spec
//! (`SCX-EVAL-METRIC-IMPROVE.md`):
//!
//! - `(n_a, n_b) ∈ {(500,500), (2000,2000), (5000,2000), (1000,10000)}`
//! - `n_dims ∈ {2_000, 18_000}`
//! - `metric ∈ {Euclidean, L1, Cosine}`
//! - `backend ∈ {Scalar, Gemm}` (Gemm skipped for L1 — kernel returns
//!   `AccelError::InvalidInput`)
//! - `dtype ∈ {f32, f64}`
//!
//! Two top-level groups:
//! - `mean_pairwise_distance` (cross-distance — `(n_a, n_b)`)
//! - `mean_pairwise_distance_self` (square self-distance — `(n, n)`)
//!
//! The cross grid is the slow one (e.g. 5000×2000×18000 in f64 with the
//! scalar path is multi-second). The default criterion sample size of 100
//! would push that into ~minutes; the bench therefore lowers
//! `sample_size`/`measurement_time` for the heavy shapes — see
//! `bench_cross` / `bench_self` below.

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use scx_accel::eval_metrics::distances::{
    mean_pairwise_distance, mean_pairwise_distance_self, DistanceBackend, PairwiseFloat,
};
use scx_accel::DistanceMetric;

// ── Synthetic data generation ───────────────────────────────────────────

/// Deterministic LCG sequence in `[-1, 1]`. Keeps allocation costs out of
/// the timed region without depending on `rand` and its features.
fn synthetic_matrix<F: PairwiseFloat>(n_rows: usize, n_dims: usize, seed: u64) -> Vec<F> {
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut next = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 32) as f64 / u32::MAX as f64) * 2.0 - 1.0
    };
    let mut buf = Vec::with_capacity(n_rows * n_dims);
    for _ in 0..n_rows * n_dims {
        buf.push(F::from_f64(next()));
    }
    buf
}

// ── Configuration knobs ─────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct CrossShape {
    n_a: usize,
    n_b: usize,
    label: &'static str,
}

const CROSS_SHAPES: &[CrossShape] = &[
    CrossShape {
        n_a: 500,
        n_b: 500,
        label: "500x500",
    },
    CrossShape {
        n_a: 2000,
        n_b: 2000,
        label: "2000x2000",
    },
    CrossShape {
        n_a: 5000,
        n_b: 2000,
        label: "5000x2000",
    },
    CrossShape {
        n_a: 1000,
        n_b: 10000,
        label: "1000x10000",
    },
];

const SELF_SIZES: &[(usize, &str)] = &[(500, "500"), (2000, "2000"), (5000, "5000")];

const DIMS: &[(usize, &str)] = &[(2_000, "d2k"), (18_000, "d18k")];

const METRICS: &[(DistanceMetric, &str)] = &[
    (DistanceMetric::Euclidean, "euclidean"),
    (DistanceMetric::L1, "l1"),
    (DistanceMetric::Cosine, "cosine"),
];

const BACKENDS: &[(DistanceBackend, &str)] = &[
    (DistanceBackend::Scalar, "scalar"),
    (DistanceBackend::Gemm, "gemm"),
];

// ── Sampling tuning ─────────────────────────────────────────────────────

/// Pick a criterion sample budget that completes in seconds, not minutes,
/// for the heavy cross / self shapes. Scalar paths at 5000×2000×18000 in
/// f64 measure multi-second per iter; gemm typically ≥ 50× faster, so
/// the scalar branch needs lower sample counts and longer measurement
/// windows to stay under a few minutes total per id.
fn budget(n_pairs: usize, n_dims: usize, scalar: bool) -> (usize, Duration, Duration) {
    let work = n_pairs as u128 * n_dims as u128;
    let (sample_size, warmup_ms, measure_s) = if scalar {
        // Scalar throughput is roughly 1 ns/op on a modern CPU. Cap the
        // total wall budget at ~30 s per id by scaling samples down as
        // work grows.
        if work > 1_000_000_000 {
            (10, 100, 30)
        } else if work > 100_000_000 {
            (20, 100, 15)
        } else {
            (40, 200, 8)
        }
    } else {
        // Gemm path: faer's matmul + parallel row-norm expansion lands at
        // ~50–100 GF/s on AVX2, so 1e10 work ≈ 100 ms/iter. Standard
        // criterion sample sizes are fine.
        if work > 10_000_000_000 {
            (30, 200, 15)
        } else if work > 1_000_000_000 {
            (50, 500, 10)
        } else {
            (60, 500, 5)
        }
    };
    (
        sample_size,
        Duration::from_millis(warmup_ms),
        Duration::from_secs(measure_s),
    )
}

// ── Cross-distance bench ────────────────────────────────────────────────

fn bench_cross_for_dtype<F: PairwiseFloat>(c: &mut Criterion, dtype_label: &str) {
    let mut group = c.benchmark_group(format!("mean_pairwise_distance/{dtype_label}"));
    group.throughput(Throughput::Elements(1));

    for &CrossShape { n_a, n_b, label } in CROSS_SHAPES {
        for &(n_dims, dlabel) in DIMS {
            // Allocate once per shape; criterion's iter() reuses the
            // borrowed inputs so timed code only includes the kernel.
            let a: Vec<F> = synthetic_matrix(n_a, n_dims, 0xA1A2_A3A4);
            let b: Vec<F> = synthetic_matrix(n_b, n_dims, 0xB1B2_B3B4);

            for &(metric, mlabel) in METRICS {
                for &(backend, blabel) in BACKENDS {
                    if matches!(metric, DistanceMetric::L1)
                        && matches!(backend, DistanceBackend::Gemm)
                    {
                        continue; // gemm + L1 returns InvalidInput
                    }
                    let id = format!("{label}/{dlabel}/{mlabel}/{blabel}");
                    let scalar_path = matches!(backend, DistanceBackend::Scalar);
                    let (samples, warmup, measure) = budget(n_a * n_b, n_dims, scalar_path);
                    group
                        .sample_size(samples)
                        .warm_up_time(warmup)
                        .measurement_time(measure);

                    group.bench_function(BenchmarkId::from_parameter(id), |bencher| {
                        bencher.iter(|| {
                            black_box(
                                mean_pairwise_distance::<F>(
                                    black_box(&a),
                                    black_box(&b),
                                    n_a,
                                    n_b,
                                    n_dims,
                                    metric,
                                    backend,
                                )
                                .unwrap(),
                            )
                        });
                    });
                }
            }
        }
    }

    group.finish();
}

fn bench_cross(c: &mut Criterion) {
    bench_cross_for_dtype::<f32>(c, "f32");
    bench_cross_for_dtype::<f64>(c, "f64");
}

// ── Self-distance bench ─────────────────────────────────────────────────

fn bench_self_for_dtype<F: PairwiseFloat>(c: &mut Criterion, dtype_label: &str) {
    let mut group = c.benchmark_group(format!("mean_pairwise_distance_self/{dtype_label}"));
    group.throughput(Throughput::Elements(1));

    for &(n, label) in SELF_SIZES {
        for &(n_dims, dlabel) in DIMS {
            let a: Vec<F> = synthetic_matrix(n, n_dims, 0xC1C2_C3C4);

            for &(metric, mlabel) in METRICS {
                for &(backend, blabel) in BACKENDS {
                    if matches!(metric, DistanceMetric::L1)
                        && matches!(backend, DistanceBackend::Gemm)
                    {
                        continue;
                    }
                    let id = format!("{label}/{dlabel}/{mlabel}/{blabel}");
                    let scalar_path = matches!(backend, DistanceBackend::Scalar);
                    // Self kernel is upper-triangle: ~n²/2 pairs
                    let (samples, warmup, measure) = budget((n * n) / 2, n_dims, scalar_path);
                    group
                        .sample_size(samples)
                        .warm_up_time(warmup)
                        .measurement_time(measure);

                    group.bench_function(BenchmarkId::from_parameter(id), |bencher| {
                        bencher.iter(|| {
                            black_box(
                                mean_pairwise_distance_self::<F>(
                                    black_box(&a),
                                    n,
                                    n_dims,
                                    metric,
                                    backend,
                                )
                                .unwrap(),
                            )
                        });
                    });
                }
            }
        }
    }

    group.finish();
}

fn bench_self(c: &mut Criterion) {
    bench_self_for_dtype::<f32>(c, "f32");
    bench_self_for_dtype::<f64>(c, "f64");
}

criterion_group!(distances, bench_cross, bench_self);
criterion_main!(distances);
