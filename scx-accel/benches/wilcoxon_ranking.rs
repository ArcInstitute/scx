//! Criterion microbenchmark for the dense Wilcoxon rank-sum hot path.
//!
//! ```text
//! cargo bench -p scx-accel --bench wilcoxon_ranking
//! ```
//!
//! **Why this exists.** `diffexp::cpu::rank_with_ties` runs once per gene inside
//! the dense DE loop, and ORG-7.21-3 moved its tie-run walk behind a shared
//! `for_each_tie_run(len, offset, equal, on_run)` primitive taking two closures.
//! The two closures are expected to inline to exactly what the hand-rolled loop
//! compiled to — but "expected to inline" is not a measurement, and
//! `scx-accel/benches/` had no DE bench at all. This is the local before/after,
//! so a hot-path refactor is not left waiting on a benchmark-gate run.
//!
//! The grid varies **tie density**, because that is what the walk is sensitive
//! to: the `while j < len && equal(i, j)` inner loop is the whole cost of a run,
//! so a column of distinct values (every run length 1) and a column of counts
//! (long runs, the real single-cell case) exercise different paths.
//!
//! - `n_obs = 20_000`, `n_vars = 40`, 4 groups, 1-vs-rest
//! - `distinct` — every value unique; every tie run has length 1
//! - `counts` — small integer counts (~12 distinct values); long runs
//! - `sparse` — 90 % exact zeros; one enormous run per column plus a tail
//! - `constant` — one run spanning the column; the degenerate maximum
//!
//! Each is run at `tie_correct ∈ {false, true}`; the flag changes only the
//! variance term, not the walk, so a gap between the two arms means the
//! correction accumulation — not the ranking — is what moved.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use scx_accel::diffexp::wilcoxon_rank_sum;

const N_OBS: usize = 20_000;
const N_VARS: usize = 40;
const N_GROUPS: usize = 4;

/// Deterministic LCG — no `rand` dependency drift between bench runs.
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// The four tie regimes, row-major `[n_obs × n_vars]`.
fn matrix(kind: &str) -> Vec<f32> {
    let mut rng = Lcg(0x5CA1AB1E);
    let n = N_OBS * N_VARS;
    match kind {
        // Every value distinct: the walk never extends a run past length 1.
        "distinct" => (0..n).map(|i| i as f32 * 1.000_001).collect(),
        // ~12 distinct values: long runs, the realistic raw-count shape.
        "counts" => (0..n).map(|_| (rng.next_u32() % 12) as f32).collect(),
        // 90 % zeros: one huge run per column, plus a short nonzero tail.
        "sparse" => (0..n)
            .map(|_| {
                let r = rng.next_u32() % 10;
                if r == 0 {
                    (rng.next_u32() % 30 + 1) as f32
                } else {
                    0.0
                }
            })
            .collect(),
        // One run spanning the whole column: the degenerate maximum.
        "constant" => vec![3.0; n],
        other => panic!("unknown matrix kind {other}"),
    }
}

fn bench_wilcoxon(c: &mut Criterion) {
    let gene_names: Vec<String> = (0..N_VARS).map(|j| format!("g{j}")).collect();
    let group_names: Vec<String> = (0..N_GROUPS).map(|g| format!("grp{g}")).collect();
    let groups: Vec<usize> = (0..N_OBS).map(|i| i % N_GROUPS).collect();

    let mut group = c.benchmark_group("wilcoxon_rank_sum");
    for kind in ["distinct", "counts", "sparse", "constant"] {
        let data = matrix(kind);
        for tie_correct in [false, true] {
            group.bench_with_input(
                BenchmarkId::new(kind, if tie_correct { "tc" } else { "no_tc" }),
                &tie_correct,
                |b, &tc| {
                    b.iter(|| {
                        wilcoxon_rank_sum(
                            &data,
                            N_OBS,
                            N_VARS,
                            &gene_names,
                            &groups,
                            &group_names,
                            None,
                            false,
                            false,
                            tc,
                            0,
                        )
                        .expect("wilcoxon_rank_sum")
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_wilcoxon);
criterion_main!(benches);
