//! Criterion microbenchmark for the streaming covariance-PCA build (OPT-1.5 / P6).
//!
//! Exercises the public [`scx_accel::covariance_pca`] streaming path — the
//! out-of-core covariance accumulation that OPT-1.5 switched from the serial
//! `sparse_outer_product_accumulate` to the parallel `_par` accumulator (and
//! dropped the eager upper-triangle mirror).
//!
//! Run:
//! ```text
//! cargo bench -p scx-accel --bench covariance_pca
//! ```
//!
//! To capture a before/after delta for the OPT-1.5 change, run this on the
//! patched tree, `git stash`, run again on the baseline, and compare. The
//! benchmark is deterministic (fixed seed) so the only variable is the
//! accumulator implementation.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use scx_accel::covariance_pca;
use scx_format_io::shard_source::ShardSource;
use scx_sparse::ScxCsr;

/// Minimal in-memory multi-shard CSR source for benchmarking.
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

/// Build a synthetic HVG-like source: `n_shards` shards of `rows_per_shard`
/// rows each, `n_vars` columns, ~`density` nonzeros per row (small positive
/// integer counts, sorted column indices — canonical CSR).
fn build_source(
    n_shards: usize,
    rows_per_shard: usize,
    n_vars: usize,
    density: f64,
) -> InMemoryShards {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let nnz_per_row = ((n_vars as f64) * density).round() as usize;
    let nnz_per_row = nnz_per_row.max(1).min(n_vars);

    let mut shards = Vec::with_capacity(n_shards);
    for _ in 0..n_shards {
        let mut indptr = Vec::with_capacity(rows_per_shard + 1);
        let mut indices: Vec<i32> = Vec::with_capacity(rows_per_shard * nnz_per_row);
        let mut data: Vec<f32> = Vec::with_capacity(rows_per_shard * nnz_per_row);
        indptr.push(0i64);
        for _ in 0..rows_per_shard {
            // Sample `nnz_per_row` distinct columns in O(nnz_per_row), then sort
            // (canonical CSR). `rand::seq::index::sample` avoids the O(nnz²)
            // rejection loop a `contains`-check would incur.
            let mut cols: Vec<usize> =
                rand::seq::index::sample(&mut rng, n_vars, nnz_per_row).into_vec();
            cols.sort_unstable();
            for &c in &cols {
                indices.push(c as i32);
                data.push(rng.gen_range(1..=20) as f32);
            }
            indptr.push(indices.len() as i64);
        }
        shards.push(
            ScxCsr::new((rows_per_shard, n_vars), indptr, indices, data)
                .expect("valid synthetic CSR"),
        );
    }

    InMemoryShards {
        shards,
        n_obs: n_shards * rows_per_shard,
        n_vars,
    }
}

fn bench_covariance_pca(c: &mut Criterion) {
    // (n_shards, rows_per_shard, n_vars, density) — HVG-selected regime where
    // covariance PCA is the chosen route (n_vars <= COVARIANCE_PCA_THRESHOLD).
    let grids = [
        ("8x25k_2000v_5pct", 8usize, 25_000usize, 2_000usize, 0.05f64),
        ("8x25k_5000v_3pct", 8usize, 25_000usize, 5_000usize, 0.03f64),
    ];

    let mut group = c.benchmark_group("covariance_pca_streaming");
    group.sample_size(10);

    for (label, n_shards, rows_per_shard, n_vars, density) in grids {
        let src = build_source(n_shards, rows_per_shard, n_vars, density);
        let total_nnz: usize = src.shards.iter().map(|s| s.indices.len()).sum();
        group.throughput(Throughput::Elements(total_nnz as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &src, |b, src| {
            b.iter(|| covariance_pca(src, 50, true).expect("pca ok"));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_covariance_pca);
criterion_main!(benches);
