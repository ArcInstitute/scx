//! Microbenchmark for OPT-ACCEL-4: the pseudobulk CSR scatter.
//!
//! ```text
//! cargo bench -p scx-accel --bench pseudobulk
//! ```
//!
//! Isolates exactly the delta of OPT-ACCEL-4 — the per-nonzero
//! `counts[g · n_vars + col] += v` scatter that used to run serially on the
//! calling thread and now partitions its output across the rayon pool. Both
//! arms build the same group mapping and apply the same mean divide; they
//! differ only in the scatter:
//!
//! - `before`: the serial loop OPT-ACCEL-4 replaced, replicated inline (one
//!   dependent load-add-store per nonzero, cells ascending).
//! - `after`: the shipped kernel through the public entry points —
//!   [`pseudobulk_aggregate_inmemory`] for one matrix, [`pseudobulk_aggregate`]
//!   over an in-memory multi-shard source for the streaming path (where the
//!   `before` arm runs the same serial loop inside the same ordered
//!   decode-prefetch driver, so decode overlap is equal on both arms).
//!
//! The two arms are asserted **bit-identical** on every fixture before timing
//! — the kernel's contract, not a tolerance — on values chosen so that a
//! reordering would show (a log-uniform f32 spread; integer counts sum exactly
//! in any order and would hide one).
//!
//! Grid: 200k cells × 2 000 genes at ~5 % density (20M nonzeros, 100 per row —
//! an HVG-subset matrix) with 2, 64 and 2 048 groups, and 20k cells × 20 000
//! genes at ~10 % (40M nonzeros, 2 000 per row — a full transcriptome) with 2
//! and 64 groups. Two groups is the `pseudobulk_dex` case–control shape: on
//! short rows the kernel keeps the serial walk (no partition reads fewer cache
//! lines than it does), on long rows it takes the column blocks. Throughput is
//! nonzeros per second.

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use scx_accel::{
    build_group_mapping, pseudobulk_aggregate, pseudobulk_aggregate_inmemory, AggregationMethod,
};
use scx_format_io::shard_source::ShardSource;
use scx_sparse::ScxCsr;

/// Minimal in-memory multi-shard CSR source (same shape as `covariance_pca.rs`).
struct InMemoryShards {
    shards: Vec<ScxCsr>,
    n_obs: usize,
    n_vars: usize,
}

impl ShardSource for InMemoryShards {
    fn n_shards(&self) -> usize {
        self.shards.len()
    }
    fn n_obs(&self) -> usize {
        self.n_obs
    }
    fn n_vars(&self) -> usize {
        self.n_vars
    }
    fn read_shard(&self, shard_idx: usize) -> scx_format_io::Result<ScxCsr> {
        Ok(self.shards[shard_idx].clone())
    }
}

struct Fixture {
    csr: ScxCsr,
    shards: InMemoryShards,
    obs_groups: Vec<Vec<String>>,
    groupby: Vec<String>,
    genes: Vec<String>,
    nnz: usize,
}

/// Deterministic sparse fixture: each row keeps ~`density` of the columns in
/// ascending order; values are log-uniform on `[1e-4, 1e4]` so f64 sums are
/// order-sensitive. Cells are labelled `cell i → group i % n_groups`, which puts
/// every group in every shard — the shape a group partition likes best, and
/// still not enough to make it competitive at two groups.
fn make_fixture(
    n_obs: usize,
    n_vars: usize,
    density: f64,
    n_groups: usize,
    shard_rows: usize,
) -> Fixture {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let mut indptr = Vec::with_capacity(n_obs + 1);
    indptr.push(0i64);
    let mut indices: Vec<i32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    for _ in 0..n_obs {
        for c in 0..n_vars {
            if rng.gen::<f64>() < density {
                indices.push(c as i32);
                let v: f64 = rng.gen_range(-4.0..4.0);
                data.push(10f64.powf(v) as f32);
            }
        }
        indptr.push(indices.len() as i64);
    }
    let nnz = indices.len();
    let csr = ScxCsr::new_unchecked((n_obs, n_vars), indptr, indices, data);

    let mut shards = Vec::new();
    let mut lo = 0usize;
    while lo < n_obs {
        let hi = (lo + shard_rows).min(n_obs);
        let (s, e) = (csr.indptr[lo] as usize, csr.indptr[hi] as usize);
        shards.push(ScxCsr::new_unchecked(
            (hi - lo, n_vars),
            csr.indptr[lo..=hi].iter().map(|&p| p - s as i64).collect(),
            csr.indices[s..e].to_vec(),
            csr.data[s..e].to_vec(),
        ));
        lo = hi;
    }

    Fixture {
        shards: InMemoryShards {
            shards,
            n_obs,
            n_vars,
        },
        csr,
        obs_groups: vec![(0..n_obs)
            .map(|i| format!("g{:05}", i % n_groups))
            .collect()],
        groupby: vec!["group".to_string()],
        genes: (0..n_vars).map(|j| format!("gene{j}")).collect(),
        nnz,
    }
}

/// The serial scatter OPT-ACCEL-4 replaced, plus the serial mean divide.
fn serial_scatter(
    csr: &ScxCsr,
    cell_to_group: &[usize],
    n_groups: usize,
    want_mean: bool,
) -> Vec<f64> {
    let n_vars = csr.shape.1;
    let mut counts = vec![0.0f64; n_groups * n_vars];
    let mut cell_counts = vec![0usize; n_groups];
    for &g in cell_to_group {
        cell_counts[g] += 1;
    }
    for (row, &g) in cell_to_group.iter().enumerate() {
        let (s, e) = (csr.indptr[row] as usize, csr.indptr[row + 1] as usize);
        for j in s..e {
            counts[g * n_vars + csr.indices[j] as usize] += csr.data[j] as f64;
        }
    }
    if want_mean {
        for g in 0..n_groups {
            if cell_counts[g] > 0 {
                let cc = cell_counts[g] as f64;
                for v in &mut counts[g * n_vars..(g + 1) * n_vars] {
                    *v /= cc;
                }
            }
        }
    }
    counts
}

/// "before", in-memory: group mapping + serial scatter.
fn before_inmemory(f: &Fixture) -> Vec<f64> {
    let (ctg, labels) = build_group_mapping(&f.obs_groups, f.csr.shape.0);
    serial_scatter(&f.csr, &ctg, labels.len(), true)
}

/// "after", in-memory: the shipped kernel.
fn after_inmemory(f: &Fixture) -> Vec<f64> {
    pseudobulk_aggregate_inmemory(
        &f.csr,
        &f.obs_groups,
        &f.groupby,
        &f.genes,
        AggregationMethod::Mean,
        0,
    )
    .unwrap()
    .counts
}

/// "before", streaming: the same ordered decode-prefetch driver the shipped
/// path uses, with the serial scatter in the consume closure.
fn before_streaming(f: &Fixture) -> Vec<f64> {
    let n_obs = f.shards.n_obs();
    let n_vars = f.shards.n_vars();
    let (ctg, labels) = build_group_mapping(&f.obs_groups, n_obs);
    let n_groups = labels.len();
    let mut counts = vec![0.0f64; n_groups * n_vars];
    let mut cell_counts = vec![0usize; n_groups];
    for &g in &ctg {
        cell_counts[g] += 1;
    }
    let mut global_row = 0usize;
    scx_accel::prefetch::for_each_shard_ordered_uncached(
        &f.shards,
        scx_accel::prefetch::prefetch_depth(),
        |_idx, shard| {
            for row in 0..shard.n_rows() {
                let g = ctg[global_row + row];
                let (s, e) = (shard.indptr[row] as usize, shard.indptr[row + 1] as usize);
                for j in s..e {
                    counts[g * n_vars + shard.indices[j] as usize] += shard.data[j] as f64;
                }
            }
            global_row += shard.n_rows();
            Ok(())
        },
    )
    .unwrap();
    for g in 0..n_groups {
        if cell_counts[g] > 0 {
            let cc = cell_counts[g] as f64;
            for v in &mut counts[g * n_vars..(g + 1) * n_vars] {
                *v /= cc;
            }
        }
    }
    counts
}

/// "after", streaming: the shipped path over the same shards.
fn after_streaming(f: &Fixture) -> Vec<f64> {
    pseudobulk_aggregate(
        &f.shards,
        &f.obs_groups,
        &f.groupby,
        &f.genes,
        AggregationMethod::Mean,
        0,
    )
    .unwrap()
    .counts
}

fn assert_bits_eq(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}[{i}]: {x} != {y}");
    }
}

fn bench(c: &mut Criterion) {
    // (label, n_obs, n_vars, density, groups): the short-row shape is an
    // HVG-subset matrix (100 nonzeros per row), where two groups stay on the
    // serial walk by design; the long-row shape is a full-transcriptome one
    // (2 000 nonzeros per row), where two groups take the column blocks.
    let shapes: [(&str, usize, usize, f64, &[usize]); 2] = [
        (
            "short_rows_200k_x_2k",
            200_000,
            2_000,
            0.05,
            &[2, 64, 2_048],
        ),
        ("long_rows_20k_x_20k", 20_000, 20_000, 0.10, &[2, 64]),
    ];
    let shard_rows = 4_096usize;

    let mut group = c.benchmark_group("pseudobulk_scatter");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(2));

    for (shape, n_obs, n_vars, density, group_counts) in shapes {
        for &n_groups in group_counts {
            let f = make_fixture(n_obs, n_vars, density, n_groups, shard_rows);
            let label = format!("{shape}/g{n_groups}");

            // The contract, checked before anything is timed.
            let want = before_inmemory(&f);
            assert_bits_eq(&after_inmemory(&f), &want, &format!("{label}: in-memory"));
            assert_bits_eq(
                &before_streaming(&f),
                &want,
                &format!("{label}: serial streaming"),
            );
            assert_bits_eq(&after_streaming(&f), &want, &format!("{label}: streaming"));

            group.throughput(Throughput::Elements(f.nnz as u64));
            group.bench_with_input(BenchmarkId::new("before/inmemory", &label), &f, |b, f| {
                b.iter(|| black_box(before_inmemory(black_box(f))))
            });
            group.bench_with_input(BenchmarkId::new("after/inmemory", &label), &f, |b, f| {
                b.iter(|| black_box(after_inmemory(black_box(f))))
            });
            group.bench_with_input(BenchmarkId::new("before/streaming", &label), &f, |b, f| {
                b.iter(|| black_box(before_streaming(black_box(f))))
            });
            group.bench_with_input(BenchmarkId::new("after/streaming", &label), &f, |b, f| {
                b.iter(|| black_box(after_streaming(black_box(f))))
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
