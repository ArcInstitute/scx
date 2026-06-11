//! Criterion microbenchmark for OPT-1.6 / P7: the in-memory PCA variance pass.
//!
//! OPT-1.6 changed [`scx_accel::randomized_pca_inmemory`] to compute column sums
//! and sum-of-squares in a single fused pass (`ScxCsr::col_sums_and_sum_sq`) and
//! derive total variance from `col_sum_sq` (`total_variance_from_col_sq`),
//! deleting the prior `compute_total_variance_inmemory` (a full nnz pass plus a
//! `col_nnz()` pass) and the separate `col_sums()` mean pass.
//!
//! Run:
//! ```text
//! cargo bench -p scx-accel --bench randomized_pca_inmemory
//! ```
//!
//! This isolates the *variance bookkeeping* rather than the whole PCA: the ~6
//! streaming-SpMM passes (O(nnz·k)) dominate total PCA wall-clock and drown out
//! the O(nnz) variance passes, so a full-PCA before/after cannot resolve the
//! effect. Both arms below are measured in one build — `old_*` replicates the
//! deleted multi-pass logic inline, `new_*` calls the shipped fused path — so
//! there is no rebuild/contention noise between them.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use scx_sparse::total_variance_from_col_sq;
use scx_sparse::ScxCsr;

/// Build a synthetic HVG-like CSR: `n_obs × n_vars`, ~`density` nonzeros per row
/// (small positive integer counts, sorted column indices — canonical CSR).
fn build_csr(n_obs: usize, n_vars: usize, density: f64) -> ScxCsr {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let nnz_per_row = (((n_vars as f64) * density).round() as usize).clamp(1, n_vars);

    let mut indptr = Vec::with_capacity(n_obs + 1);
    let mut indices: Vec<i32> = Vec::with_capacity(n_obs * nnz_per_row);
    let mut data: Vec<f32> = Vec::with_capacity(n_obs * nnz_per_row);
    indptr.push(0i64);
    for _ in 0..n_obs {
        let mut cols: Vec<usize> =
            rand::seq::index::sample(&mut rng, n_vars, nnz_per_row).into_vec();
        cols.sort_unstable();
        for &c in &cols {
            indices.push(c as i32);
            data.push(rng.gen_range(1..=20) as f32);
        }
        indptr.push(indices.len() as i64);
    }
    ScxCsr::new((n_obs, n_vars), indptr, indices, data).expect("valid synthetic CSR")
}

/// OLD path (pre-OPT-1.6): `col_sums` mean pass + the deleted
/// `compute_total_variance_inmemory` (centered two-pass over nnz + `col_nnz`).
fn old_variance_bookkeeping(csr: &ScxCsr) -> (Vec<f64>, f64) {
    let n_obs = csr.n_rows();
    let n_vars = csr.n_cols();
    let sums = csr.col_sums();
    let means: Vec<f64> = sums.iter().map(|&s| s / n_obs as f64).collect();

    // compute_total_variance_inmemory(csr, Some(&means)):
    let mut total = 0.0f64;
    for r in 0..n_obs {
        let start = csr.indptr[r] as usize;
        let end = csr.indptr[r + 1] as usize;
        for j in start..end {
            let v = csr.data[j] as f64 - means[csr.indices[j] as usize];
            total += v * v;
        }
    }
    let col_nnz = csr.col_nnz();
    for c in 0..n_vars {
        let n_zeros = n_obs.saturating_sub(col_nnz[c] as usize);
        total += n_zeros as f64 * means[c] * means[c];
    }
    let total_var = total / (n_obs as f64 - 1.0).max(1.0);
    (means, total_var)
}

/// NEW path (OPT-1.6): one fused `col_sums_and_sum_sq` pass + closed-form
/// `total_variance_from_col_sq`.
fn new_variance_bookkeeping(csr: &ScxCsr) -> (Vec<f64>, f64) {
    let n_obs = csr.n_rows();
    let (sums, sum_sq) = csr.col_sums_and_sum_sq();
    let means: Vec<f64> = sums.iter().map(|&s| s / n_obs as f64).collect();
    let total_var = total_variance_from_col_sq(&sum_sq, Some(&means), n_obs);
    (means, total_var)
}

fn bench_variance_bookkeeping(c: &mut Criterion) {
    // (label, n_obs, n_vars, density) — HVG-selected regimes.
    let grids = [
        ("50k_x_5000v_3pct", 50_000usize, 5_000usize, 0.03f64),
        ("100k_x_2000v_5pct", 100_000usize, 2_000usize, 0.05f64),
    ];

    let mut group = c.benchmark_group("pca_inmemory_variance_pass");

    for (label, n_obs, n_vars, density) in grids {
        let csr = build_csr(n_obs, n_vars, density);
        // Sanity: the two paths agree on total variance (parity) before timing.
        let (_, old_tv) = old_variance_bookkeeping(&csr);
        let (_, new_tv) = new_variance_bookkeeping(&csr);
        assert!(
            (old_tv - new_tv).abs() <= 1e-6 * old_tv.abs().max(1.0),
            "{label}: total-variance parity broken: old={old_tv}, new={new_tv}"
        );

        group.throughput(Throughput::Elements(csr.indices.len() as u64));
        group.bench_with_input(BenchmarkId::new("old_multipass", label), &csr, |b, csr| {
            b.iter(|| black_box(old_variance_bookkeeping(black_box(csr))));
        });
        group.bench_with_input(BenchmarkId::new("new_fused", label), &csr, |b, csr| {
            b.iter(|| black_box(new_variance_bookkeeping(black_box(csr))));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_variance_bookkeeping);
criterion_main!(benches);
