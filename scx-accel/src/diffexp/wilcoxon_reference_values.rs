//! Pinned external reference values for the Wilcoxon rank-sum DE family
//! (ORG-7.21-3, ORG-7.21-4).
//!
//! # Why an external reference at all
//!
//! SCX has **three** Wilcoxon rank-sum implementations: the dense CSR kernel
//! ([`crate::diffexp::cpu`]), the analytic sparse-nnz one
//! (`crate::csc::wilcoxon::gene_stats_nnz`) and a CUDA kernel driven by
//! [`crate::diffexp::gpu`]. Two of them now share one tie-run walk
//! ([`crate::diffexp::cpu::for_each_tie_run`]); the third is a `.cu` file and
//! structurally cannot. So the only agreement all three can have is *agreement
//! on a number computed by something else* — and until this module, no such
//! number existed. The nnz path's correctness rested entirely on a proptest
//! against the dense kernel, which makes a dense-kernel bug invisible, and the
//! review established that the nnz path had already re-derived one of the dense
//! path's bugs independently.
//!
//! # Every value here was PRODUCED by another implementation, not derived here
//!
//! | table | source |
//! |---|---|
//! | [`SCIPY_P_TIE_CORRECTED`] | `scipy.stats.mannwhitneyu(...).pvalue` |
//! | [`SCANPY_Z_TIE_CORRECTED`] / [`SCANPY_P_TIE_CORRECTED`] | `scanpy.tl.rank_genes_groups(tie_correct=True)` |
//! | [`SCANPY_Z_UNCORRECTED`] / [`SCANPY_P_UNCORRECTED`] | the same, `tie_correct=False` |
//!
//! **The first version of this module was self-referential and that was the
//! defect, not the numbers.** It took only `U` from scipy and rebuilt `z` and
//! `p` with `sigma_sq = (n1·n2/12)·((n+1) − tc/(n(n−1)))` — the same transform
//! as [`crate::diffexp::cpu::wilcoxon_stats_from_rank_sum`]. The values were
//! bit-identical either way (measured: `max |Δp| = 0.0` against
//! `mannwhitneyu(...).pvalue`), so nothing about the constants changed when this
//! was fixed. What changed is that a bug in the `Σ(t³−t)` term copied into the
//! generator would previously have matched SCX exactly — and the uncorrected arm
//! could not have caught it, because that arm runs with `tc = 0`. That is the
//! "two implementations of one semantics agreeing" failure ORG-7.21-3 exists to
//! retire, relocated into the oracle. Do not reintroduce a computed reference.
//!
//! # Two conventions, and what scanpy actually does
//!
//! `scipy.stats.mannwhitneyu(..., method="asymptotic")` **always** applies the
//! `Σ(t³−t)` tie correction. `scanpy.tl.rank_genes_groups` takes a
//! `tie_correct` parameter that **defaults to `False`**, and SCX's default
//! matches scanpy's — so scanpy *can* produce the corrected convention and
//! simply does not by default. On this fixture the two conventions differ by
//! **4.364e-01** in `z`, both measured from scanpy: a difference in definition,
//! not in precision, so each arm needs its own reference.
//!
//! [`SCIPY_P_TIE_CORRECTED`] and [`SCANPY_P_TIE_CORRECTED`] agree at
//! **exactly 0.0** — two independent implementations of the tie term. That
//! agreement is asserted in `wilcoxon_reference_tests.rs`, because it is what
//! makes either of them an oracle for SCX rather than a second opinion on the
//! same arithmetic.
//!
//! | field | bar | why |
//! |---|---|---|
//! | `z`, either arm | [`Z_ATOL`] = 1e-6 | scanpy stores `scores` as **float32** in its recarray |
//! | `p`, corrected | [`P_CORRECTED_ATOL`] = 1e-15 | scipy is f64; only `normal_sf` vs `norm.sf` separates them |
//! | `p`, uncorrected | [`P_UNCORRECTED_ATOL`] = 1e-12 | scanpy's `pvals` are float64 |
//!
//! The dtypes were checked against the recarray, not assumed.
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

/// Cells in the fixture, including the two unlabelled ones.
pub const N_OBS: usize = 12;
/// Genes in the fixture.
pub const N_VARS: usize = 6;
/// Real groups. `FIXTURE_GROUPS[i] == N_GROUPS` is the unlabelled sentinel.
pub const N_GROUPS: usize = 2;

/// Tolerance for every `z` table: scanpy's f32 storage, one decimal order over
/// the observed 1.07e-07.
pub const Z_ATOL: f64 = 1e-6;
/// Tolerance for [`SCIPY_P_TIE_CORRECTED`] — f64 on both sides.
pub const P_CORRECTED_ATOL: f64 = 1e-15;
/// Tolerance for [`SCANPY_P_UNCORRECTED`] — f64 on both sides.
pub const P_UNCORRECTED_ATOL: f64 = 1e-12;

/// The fixture, row-major `[N_OBS × N_VARS]`.
///
/// Every value is exactly representable in `f32`, so the `f32 → f64` promotion
/// inside the kernels is lossless.
///
/// Rows 10 and 11 are unlabelled and carry deliberately extreme values. An
/// unlabelled cell is in no group but **is** in the rank pool and in every
/// group's "rest" (scanpy's rule, SCX's since 0.17 / X9), so a kernel that
/// dropped them from either could not hide behind a tolerance.
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
/// | `g5` | nonzero **only** in unlabelled rows | the inclusion canary: over the labelled cells alone it is identical to the all-zero `g2`, so an answer that *matches* `g2`'s means unlabelled cells never reached the pool |
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

/// scipy's own two-sided p per `[gene][group]`, 1-vs-rest, `tie_correct = true`.
///
/// `mannwhitneyu(...).pvalue` verbatim — scipy applies its own tie correction
/// internally, which is what makes this an independent check on SCX's
/// `Σ(t³−t)/(n(n−1))` term rather than a restatement of it.
///
/// Computed over **every** cell: each group is compared with every row outside
/// it, unlabelled rows included, which is what scanpy's `X[~mask_g]` means and
/// what scipy expresses natively as the `g != grp` side of the split. So the
/// unlabelled-cell contract is referenced to scipy rather than to a sibling
/// implementation of the same idea. (Before X9 these were computed on the
/// labelled subset, which pinned the divergent rule.)
///
/// `g1` and `g2` are fully tied over the pool, so
/// `σ² = (n₁n₂/12)·((n+1) − Σ(t³−t)/(n(n−1)))` is exactly 0. scipy warns and
/// yields `nan` there; every SCX kernel returns the documented `(0.0, 1.0)`.
/// Those cells pin SCX's contract, not scipy's — the generator says so at the
/// point it writes them. `g5` is **not** among them any more: its two nonzeros
/// sit in the unlabelled rows, which are now in the pool.
pub const SCIPY_P_TIE_CORRECTED: [[f64; N_GROUPS]; N_VARS] = [
    [0.004483250402085382, 0.22322514530675575],
    [1.0, 1.0],
    [1.0, 1.0],
    [0.25143684995587434, 0.25143684995587434],
    [0.03668277440246522, 0.6760528085643454],
    [0.21189355203127813, 0.21189355203127813],
];

/// scanpy `scores` per `[gene][group]` with `tie_correct=True`.
///
/// The corrected arm's `z` reference. scipy exposes no z, and reconstructing one
/// from `U` with SCX's own variance formula is what made the first version of
/// this module self-referential — so the z oracle is scanpy, which implements
/// the tie term independently. **float32** in the recarray, hence [`Z_ATOL`].
pub const SCANPY_Z_TIE_CORRECTED: [[f64; N_GROUPS]; N_VARS] = [
    [-2.8419928550720215, 1.2179969549179077],
    [0.0, 0.0],
    [0.0, 0.0],
    [-1.1468663215637207, 1.1468663215637207],
    [-2.0892772674560547, 0.417855441570282],
    [-1.2483755350112915, -1.2483755350112915],
];

/// scanpy `pvals` per `[gene][group]` with `tie_correct=True`.
///
/// Agrees with [`SCIPY_P_TIE_CORRECTED`] at **exactly 0.0** — two independent
/// implementations of the tie term. `wilcoxon_reference_tests.rs` asserts that,
/// because it is what makes either of them an oracle for SCX rather than a
/// second opinion on the same arithmetic.
pub const SCANPY_P_TIE_CORRECTED: [[f64; N_GROUPS]; N_VARS] = [
    [0.004483250402085382, 0.22322514530675575],
    [1.0, 1.0],
    [1.0, 1.0],
    [0.25143684995587434, 0.25143684995587434],
    [0.03668277440246522, 0.6760528085643454],
    [0.21189355203127813, 0.21189355203127813],
];

/// scanpy `scores` per `[gene][group]` with `tie_correct=False` — the default.
///
/// float32, like the corrected arm. `g0` shows the storage width plainly: it has
/// no ties at all, so both conventions must agree exactly on it, and the two
/// scanpy tables do.
pub const SCANPY_Z_UNCORRECTED: [[f64; N_GROUPS]; N_VARS] = [
    [-2.8419928550720215, 1.2179969549179077],
    [0.0, 0.0],
    [0.0, 0.0],
    [-1.1367970705032349, 1.1367970705032349],
    [-2.0299949645996094, 0.40599897503852844],
    [-0.8119979500770569, -0.8119979500770569],
];

/// scanpy `pvals` per `[gene][group]` with `tie_correct=False`.
///
/// float64 in the recarray — computed from an f64 z, not from the f32 `scores`
/// above — so this one is pinned tightly.
pub const SCANPY_P_UNCORRECTED: [[f64; N_GROUPS]; N_VARS] = [
    [0.004483250402085382, 0.22322514530675575],
    [1.0, 1.0],
    [1.0, 1.0],
    [0.2556231075464126, 0.2556231075464126],
    [0.042357062026854894, 0.6847433561373875],
    [0.41679281184762706, 0.41679281184762706],
];

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

/// Cells per pool bucket: `N_GROUPS` real groups followed by the unlabelled
/// count, the shape `gene_stats_nnz` takes.
pub fn group_cell_counts() -> Vec<usize> {
    let mut counts = vec![0usize; N_GROUPS + 1];
    for &g in FIXTURE_GROUPS.iter() {
        counts[g.min(N_GROUPS)] += 1;
    }
    counts
}
