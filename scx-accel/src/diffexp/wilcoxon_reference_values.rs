//! Pinned external reference values for the Wilcoxon rank-sum DE family
//! (ORG-7.21-3, ORG-7.21-4).
//!
//! # Why an external reference at all
//!
//! SCX has **three** Wilcoxon rank-sum implementations: the dense CSR kernel
//! ([`crate::diffexp::cpu`]), the analytic sparse-nnz one
//! (`crate::csc::wilcoxon::gene_stats_nnz`) and a CUDA kernel driven by
//! [`crate::diffexp::gpu`]. Two of them now share one tie-run walk
//! ([`for_each_tie_run`](crate::diffexp::cpu::for_each_tie_run)); the third is a
//! `.cu` file and structurally cannot. So the only agreement all three can have
//! is *agreement on a number computed by something else* — and until this
//! module, no such number existed. The nnz path's correctness rested entirely
//! on a proptest against the dense kernel, which makes a dense-kernel bug
//! invisible, and the review established that the nnz path had already
//! re-derived one of the dense path's bugs independently.
//!
//! One table, three consumers, all inside this crate — including the GPU arm,
//! whose test lives in `diffexp/gpu_tests.rs` because `scx-accel` is where the
//! GPU DE orchestration lives. Nothing in `scx-gpu` reads it, so it needs no
//! cross-crate home.
//!
//! # Regenerating
//!
//! ```text
//! .venv/bin/python benchmarks/scripts/generate_de_parity_references.py
//! ```
//!
//! That script owns the fixture *and* the expected values, so they cannot
//! drift. Do not hand-edit a constant here: if a number needs to change, the
//! reference changed, and the script is what says so.
//!
//! # The two conventions, and why there are two tables
//!
//! `scipy.stats.mannwhitneyu(..., method="asymptotic")` **always** applies the
//! `Σ(t³−t)` tie correction. SCX's `tie_correct` defaults to `false`, matching
//! scanpy, which applies none. On this fixture the two answers differ by
//! **7.647e-02** in `z` — a different answer, not rounding. So:
//!
//! | arm | oracle | tolerance | why |
//! |---|---|---|---|
//! | `tie_correct = true` | scipy 1.17.1 | `abs = 0` on `z` | both f64 end to end |
//! | `tie_correct = false` | scanpy 1.12 | `1e-6` on `z`, `1e-12` on `p` | scanpy stores `scores` as **float32** and `pvals` as float64 — verified against the recarray dtypes, not assumed |
//!
//! Pinning the uncorrected arm against scipy would be wrong rather than loose.

/// Cells in the fixture, including the two unlabelled ones.
pub const N_OBS: usize = 12;
/// Genes in the fixture.
pub const N_VARS: usize = 6;
/// Real groups. `FIXTURE_GROUPS[i] == N_GROUPS` is the unlabelled sentinel.
pub const N_GROUPS: usize = 2;
/// Cells that take part in the comparison — `N_OBS` minus the unlabelled ones.
pub const N_LABELLED: usize = 10;

/// The fixture, row-major `[N_OBS × N_VARS]`.
///
/// Every value is exactly representable in `f32`, so the `f32 → f64` promotion
/// inside the kernels is lossless and an `f64` reference is comparable at
/// `abs = 0`.
///
/// Rows 10 and 11 are unlabelled and carry deliberately extreme values: if any
/// kernel ever let an unlabelled cell into the rank pool or the rest
/// denominator, no tolerance would hide it.
///
/// The six genes each pin a path some arm treats differently:
///
/// | gene | shape | what it catches |
/// |---|---|---|
/// | `g0` | distinct, separating | the clean signal — no ties at all |
/// | `g1` | constant everywhere | a total tie; `σ²` collapses to exactly 0 |
/// | `g2` | all zeros | degenerate, and no stored nonzeros for the nnz arm |
/// | `g3` | negatives + stored zeros | the nnz arm's `neg` block, and stored zeros that must join the *implicit*-zero block rather than `pos` |
/// | `g4` | small counts | partial ties — the realistic single-cell shape |
/// | `g5` | nonzero **only** in unlabelled rows | the leak canary: to the labelled pool this gene is identical to `g2`, so any answer that differs from `g2`'s means unlabelled cells reached the pool |
pub const FIXTURE_X: [[f32; N_VARS]; N_OBS] = [
    [1.0, 3.0, 0.0, -2.0, 0.0, 0.0],
    [2.0, 3.0, 0.0, -1.0, 1.0, 0.0],
    [3.0, 3.0, 0.0, 0.0, 1.0, 0.0],
    [4.0, 3.0, 0.0, 0.0, 2.0, 0.0],
    [11.0, 3.0, 0.0, -1.0, 1.0, 0.0],
    [12.0, 3.0, 0.0, 0.0, 2.0, 0.0],
    [13.0, 3.0, 0.0, 2.0, 2.0, 0.0],
    [14.0, 3.0, 0.0, 3.0, 3.0, 0.0],
    [5.0, 3.0, 0.0, 1.0, 2.0, 0.0],
    [15.0, 3.0, 0.0, 4.0, 3.0, 0.0],
    [100.0, 3.0, 0.0, -50.0, 9.0, 7.0],
    [200.0, 3.0, 0.0, 50.0, 9.0, 8.0],
];

/// Group of each cell. `2` (`== N_GROUPS`) is the unlabelled sentinel.
pub const FIXTURE_GROUPS: [usize; N_OBS] = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 2, 2];

/// scipy `z` per `[gene][group]`, 1-vs-rest, `tie_correct = true`.
///
/// Computed over the **labelled subset only** — scipy has no notion of a cell
/// outside the pool, so the reference is `mannwhitneyu(group_values,
/// rest_values)` on the physically subsetted matrix. A kernel handed the full
/// `N_OBS` matrix plus the sentinel must reproduce it exactly; that equality
/// *is* the unlabelled-cell contract, referenced to scipy rather than to a
/// sibling implementation of the same idea.
///
/// `g1`, `g2` and `g5` are fully tied over the labelled pool, so
/// `σ² = (n₁n₂/12)·((n+1) − Σ(t³−t)/(n(n−1)))` is exactly 0. scipy warns and
/// yields `nan` there; every SCX kernel returns the documented `(0.0, 1.0)`.
/// Those three cells pin SCX's contract, not scipy's — the generator says so
/// at the point it writes them.
pub const SCIPY_Z_TIE_CORRECTED: [[f64; N_GROUPS]; N_VARS] = [
    [-2.6111648393354674, 2.6111648393354674],
    [0.0, 0.0],
    [0.0, 0.0],
    [-1.4849242404917498, 1.4849242404917498],
    [-1.6431676725154984, 1.6431676725154984],
    [0.0, 0.0],
];

/// scipy two-sided `p` per `[gene][group]`, `tie_correct = true`.
pub const SCIPY_P_TIE_CORRECTED: [[f64; N_GROUPS]; N_VARS] = [
    [0.009023438818080326, 0.009023438818080326],
    [1.0, 1.0],
    [1.0, 1.0],
    [0.13756389390990328, 0.13756389390990328],
    [0.10034824646229074, 0.10034824646229074],
    [1.0, 1.0],
];

/// scanpy `scores` per `[gene][group]`, `tie_correct = false` (the default).
///
/// **float32 in scanpy's recarray**, hence [`Z_UNCORRECTED_ATOL`] rather than an
/// exact match. `g0` shows it: scipy says `-2.6111648393354674`, scanpy says
/// `-2.6111648082733154`, and `g0` has no ties at all — so the gap is purely
/// scanpy's storage width, not a difference in convention.
pub const SCANPY_Z_UNCORRECTED: [[f64; N_GROUPS]; N_VARS] = [
    [-2.6111648082733154, 2.6111648082733154],
    [0.0, 0.0],
    [0.0, 0.0],
    [-1.4622522592544556, 1.4622522592544556],
    [-1.5666989088058472, 1.5666989088058472],
    [0.0, 0.0],
];

/// scanpy `pvals` per `[gene][group]`, `tie_correct = false`.
///
/// float64 in scanpy's recarray — computed from an f64 z, not from the f32
/// `scores` above — so this one is pinned tightly.
pub const SCANPY_P_UNCORRECTED: [[f64; N_GROUPS]; N_VARS] = [
    [0.009023438818080326, 0.009023438818080326],
    [1.0, 1.0],
    [1.0, 1.0],
    [0.14367208180696023, 0.14367208180696023],
    [0.11718508719813801, 0.11718508719813801],
    [1.0, 1.0],
];

/// Tolerance for [`SCANPY_Z_UNCORRECTED`]: scanpy's f32 storage, one decimal
/// order of headroom over the observed 3.1e-08.
pub const Z_UNCORRECTED_ATOL: f64 = 1e-6;

/// Tolerance for [`SCANPY_P_UNCORRECTED`] — f64 on both sides.
pub const P_UNCORRECTED_ATOL: f64 = 1e-12;

/// The fixture flattened row-major, for kernels that take a dense `&[f32]`.
pub fn dense_x() -> Vec<f32> {
    FIXTURE_X.iter().flat_map(|r| r.iter().copied()).collect()
}

/// One gene's column as `(row_indices, values)` in CSC order.
///
/// **Stored zeros are kept.** `g3` has exact zeros at rows 2, 3 and 5, and the
/// nnz kernel must route them into the *implicit*-zero tie block rather than
/// the `pos` block — a distinction a column built by dropping zeros could not
/// express, and therefore could not test.
pub fn csc_column(gene: usize) -> (Vec<i32>, Vec<f32>) {
    let mut rows = Vec::new();
    let mut vals = Vec::new();
    for (row, cells) in FIXTURE_X.iter().enumerate() {
        // g2 is genuinely all-zero: it has no stored entries at all, which is
        // the "no nonzeros whatsoever" arm of the nnz kernel.
        if gene == 2 {
            continue;
        }
        let v = cells[gene];
        if v != 0.0 || gene == 3 {
            rows.push(row as i32);
            vals.push(v);
        }
    }
    (rows, vals)
}

/// Cells per real group, indexed by group. Excludes the unlabelled cells.
pub fn group_cell_counts() -> Vec<usize> {
    let mut counts = vec![0usize; N_GROUPS];
    for &g in FIXTURE_GROUPS.iter() {
        if g < N_GROUPS {
            counts[g] += 1;
        }
    }
    counts
}
