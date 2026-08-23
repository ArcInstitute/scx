//! Golden per-column mean/variance values for every `scx-accel` finalize site.
//!
//! Phase 7a unifies **nine** copies of the `(Σx² − n·mean²)/(n−1)` finalize —
//! four here in [`super::super::cpu`], one in `crate::csc::mean_var`, two in
//! [`super::gpu`]'s batched device wrapper, and two in `scx-gpu`'s `gpu_hvg` —
//! onto `scx_sparse::finalize_column_moments`. (An earlier version of this
//! comment said seven, written before the last two were found; the CI guard and
//! `scx_sparse::moments`' module docs are the authority.) These
//! tests exist to make that adoption *provably* behaviour-preserving rather than
//! preserving-by-intent, which is the whole reason the Organization series
//! requires the safety net to land before the refactor.
//!
//! Why **bit** equality and not a tolerance: the refactor moves arithmetic
//! between crates, and a tolerance of even 1e-12 would hide exactly the class of
//! change that matters — a reordered accumulation, a `mean * mean` folded into a
//! single `mul_add`, or a different division order. `f64::to_bits` catches all
//! three. The one site that cannot be pinned this way is
//! [`super::super::cpu::streaming_mean_var_expm1`]: `f64::exp_m1` is a libm
//! call, not IEEE-exact across platforms and versions, so that arm asserts a
//! tight relative tolerance instead and says so.
//!
//! The fixture's four columns are the cases the review's §7.6/§7.14 findings
//! name, so the pinned numbers are not arbitrary:
//!
//! | col | shape | what it pins |
//! |---|---|---|
//! | 0 | all implicit zeros | `mean = var = 0`, no division by a zero count |
//! | 1 | one nonzero in 9 rows | the sparse-majority case |
//! | 2 | eight 200s and one 201 | `Σx² − n·mean²` cancellation: the subtraction loses ~11 of f64's 16 digits |
//! | 3 | mixed small ints with holes | the ordinary case |

use super::*;
use scx_sparse::ScxCsr;

/// 9 rows × 4 cols, row-major, delivered as three 3-row shards.
const GOLDEN_DENSE: [u8; 36] = [
    0, 0, 200, 1, //
    0, 0, 200, 2, //
    0, 0, 200, 3, //
    0, 7, 200, 0, //
    0, 0, 201, 4, //
    0, 0, 200, 5, //
    0, 0, 200, 0, //
    0, 0, 200, 6, //
    0, 0, 200, 7, //
];

/// Cell → batch for the batched arm. The trailing `-1` is deliberate: the
/// batched kernel skips `b < 0`, so batch 2 holds two cells and the **global**
/// stats it derives cover 8 cells, not 9. That asymmetry with
/// [`streaming_mean_var`] is existing documented behaviour, and pinning it is
/// the point — a refactor that "helpfully" folded the unassigned cell back in
/// would change published HVG numbers.
const GOLDEN_BATCH: [i32; 9] = [0, 0, 0, 1, 1, 1, 2, 2, -1];

fn shards_from_dense(dense: &[u8], n_obs: usize, n_vars: usize, rows_per: usize) -> Vec<ScxCsr> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < n_obs {
        let end = (start + rows_per).min(n_obs);
        let mut indptr: Vec<i64> = vec![0];
        let mut indices: Vec<i32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        for r in start..end {
            for c in 0..n_vars {
                let v = dense[r * n_vars + c];
                if v != 0 {
                    indices.push(c as i32);
                    data.push(v as f32);
                }
            }
            indptr.push(indices.len() as i64);
        }
        out.push(ScxCsr::new_unchecked(
            (end - start, n_vars),
            indptr,
            indices,
            data,
        ));
        start = end;
    }
    out
}

fn golden_source() -> InMemorySource {
    InMemorySource {
        shards: shards_from_dense(&GOLDEN_DENSE, 9, 4, 3),
        n_obs: 9,
        n_vars: 4,
    }
}

/// Small-magnitude fixture for the `expm1` arm: `expm1(200)` is ~7.2e86 and its
/// square ~5e173, which stresses nothing useful and makes the numbers unreadable.
fn small_source() -> InMemorySource {
    let dense: [u8; 12] = [
        1, 0, 3, //
        0, 2, 0, //
        2, 0, 1, //
        0, 3, 0, //
    ];
    InMemorySource {
        shards: shards_from_dense(&dense, 4, 3, 2),
        n_obs: 4,
        n_vars: 3,
    }
}

/// Assert every element of `got` is bit-identical to `want`.
///
/// Bit equality, not a tolerance: see this module's header. A failure prints
/// both the bits and the decimal so a genuine arithmetic change is legible.
fn assert_bits(got: &[f64], want: &[u64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w,
            "{what}[{i}]: got 0x{:016x} ({g}), want 0x{w:016x} ({})",
            g.to_bits(),
            f64::from_bits(w)
        );
    }
}

/// `streaming_mean_var` — the CSR raw-moments finalize (`hvg/cpu.rs`, site 1 of 9).
///
/// Column 2 is the cancellation witness. Its exact variance is `1/9`
/// (`0.111111111111...`); the closed form returns `0.11111111110949423` —
/// **eleven** correct digits out of sixteen. That is not a defect being pinned,
/// it is the documented `Σx² − n·mean²` behaviour, and it is the reason Phase 7a
/// makes the primitive *report* precision loss instead of absorbing it. A
/// refactor that changed the loss (either way) has to change this number.
#[test]
fn streaming_mean_var_golden() {
    let stats = streaming_mean_var(&golden_source()).unwrap();
    assert_bits(
        &stats.means,
        &[
            0x0000000000000000, // 0            — all-zero column
            0x3fe8e38e38e38e39, // 0.7777…      — 7/9
            0x4069038e38e38e39, // 200.1111…
            0x4008e38e38e38e39, // 3.1111…      — 28/9
        ],
        "means",
    );
    assert_bits(
        &stats.variances,
        &[
            0x0000000000000000, // 0
            0x4015c71c71c71c72, // 5.444444444444445
            0x3fbc71c71c700000, // 0.11111111110949423 — exact answer is 1/9
            0x401a71c71c71c71c, // 6.611111111111111
        ],
        "variances",
    );
}

/// `streaming_mean_var_batched` — the per-batch and global finalizes
/// (`hvg/cpu.rs`, sites 3 and 4 of 9).
///
/// These two are the sites whose clamp is `.max(0.0)` rather than
/// `if v < 0.0 { 0.0 }`. For every value here the two are bit-identical, which
/// is exactly why the divergence needs its own unit test on the primitive
/// (see `scx_sparse::moments`) rather than an end-to-end one: it is reachable
/// only through a NaN, and a NaN cannot reach here past `ensure_finite_hvg_data`.
#[test]
fn streaming_mean_var_batched_golden() {
    let stats = streaming_mean_var_batched(&golden_source(), &GOLDEN_BATCH, 3).unwrap();

    // Batch 2 holds two cells, not three: cell 8 carries `-1` and is skipped.
    assert_eq!(stats.batch_counts, vec![3, 3, 2], "batch_counts");

    let want_means: [[u64; 4]; 3] = [
        [0, 0, 0x4069000000000000, 0x4000000000000000],
        [
            0,
            0x4002aaaaaaaaaaab,
            0x40690aaaaaaaaaab,
            0x4008000000000000,
        ],
        [0, 0, 0x4069000000000000, 0x4008000000000000],
    ];
    let want_vars: [[u64; 4]; 3] = [
        [0, 0, 0, 0x3ff0000000000000],
        [
            0,
            0x4030555555555555,
            0x3fd5555555540000,
            0x401c000000000000,
        ],
        [0, 0, 0, 0x4032000000000000],
    ];
    for (b, batch) in stats.per_batch.iter().enumerate() {
        assert_bits(&batch.means, &want_means[b], &format!("batch{b}.means"));
        assert_bits(
            &batch.variances,
            &want_vars[b],
            &format!("batch{b}.variances"),
        );
    }

    // The global arm is derived from the per-batch raw moments, so it covers the
    // **8** batched cells — not all 9. Pinned deliberately: a refactor that
    // folded the unassigned cell back in would silently change published HVG
    // numbers, and the divergence from `streaming_mean_var` above is the tell.
    assert_bits(
        &stats.global.means,
        &[
            0,
            0x3fec000000000000,
            0x4069040000000000,
            0x4005000000000000,
        ],
        "global.means",
    );
    assert_bits(
        &stats.global.variances,
        &[
            0,
            0x4018800000000000,
            0x3fc0000000000000,
            0x4014800000000000,
        ],
        "global.variances",
    );
}

/// `streaming_mean_var_expm1` — the seurat count-space finalize
/// (`hvg/cpu.rs`, site 2 of 9).
///
/// The **only** arm here that is not bit-pinned. `f64::exp_m1` is a libm call
/// and is not IEEE-exact across platforms or libm versions, so pinning its bits
/// would make this test fail on a different machine for a reason that has
/// nothing to do with the refactor. A 1e-14 relative bar is tight enough to
/// catch a changed finalize and loose enough to survive a libm difference.
#[test]
fn streaming_mean_var_expm1_golden() {
    let stats = streaming_mean_var_expm1(&small_source(), 1.0).unwrap();
    let want_means = [2.0268344818474238, 6.36864825552958, 5.200954687911678];
    let want_vars = [9.11343273669088, 80.94634502366765, 86.3368311418435];
    for (i, (&g, &w)) in stats.means.iter().zip(want_means.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-14 * w.abs(),
            "expm1 mean[{i}]: got {g}, want {w}"
        );
    }
    for (i, (&g, &w)) in stats.variances.iter().zip(want_vars.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-14 * w.abs(),
            "expm1 var[{i}]: got {g}, want {w}"
        );
    }
}

/// `streaming_mean_var_csc` — the CSC finalize (`csc/mean_var.rs`, site 5 of 9).
///
/// Pinned separately from the CSR arm rather than asserted equal to it. The two
/// accumulate in different orders (CSR walks a shard's rows, CSC walks a
/// column's entries contiguously), so bit equality between them is not a
/// property either kernel promises — `csc::parity_test` asserts them within
/// 1e-9, and that is the right bar *there*. What this test pins is that the CSC
/// side's own arithmetic does not move.
#[test]
fn streaming_mean_var_csc_golden() {
    use crate::csc::mean_var::streaming_mean_var_csc;
    use crate::csc::test_helpers::write_csr_csc_test_file;
    use scx_format_io::{BackedCscReader, BackedCsrReader, ScxReader};

    let dir = tempfile::tempdir().unwrap();
    let path = write_csr_csc_test_file(dir.path(), "moments_golden", 9, 4, &GOLDEN_DENSE, 4);

    let csc = BackedCscReader::new(ScxReader::open(&path).unwrap(), 0).unwrap();
    let csc_stats = streaming_mean_var_csc(&csc).unwrap();
    assert_bits(
        &csc_stats.means,
        &[
            0x0000000000000000,
            0x3fe8e38e38e38e39,
            0x4069038e38e38e39,
            0x4008e38e38e38e39,
        ],
        "csc.means",
    );
    assert_bits(
        &csc_stats.variances,
        &[
            0x0000000000000000,
            0x4015c71c71c71c72,
            0x3fbc71c71c700000,
            0x401a71c71c71c71c,
        ],
        "csc.variances",
    );

    // And the cross-format bar the parity suite uses, on this fixture.
    let csr = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);
    let csr_stats = streaming_mean_var(&csr).unwrap();
    for j in 0..4 {
        assert!(
            (csr_stats.means[j] - csc_stats.means[j]).abs() < 1e-9,
            "mean[{j}]: csr={} csc={}",
            csr_stats.means[j],
            csc_stats.means[j]
        );
        assert!(
            (csr_stats.variances[j] - csc_stats.variances[j]).abs() < 1e-9,
            "var[{j}]: csr={} csc={}",
            csr_stats.variances[j],
            csc_stats.variances[j]
        );
    }
}
