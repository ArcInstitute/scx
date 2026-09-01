//! The external oracle for the Wilcoxon rank-sum family (ORG-7.21-3).
//!
//! Two of SCX's three rank-sum implementations are checked here against
//! reference values computed by scipy and scanpy, held in
//! [`super::wilcoxon_reference_values`]. The third — `scx_gpu::gpu_diffexp` — checks
//! itself against the same table from its own crate; see
//! `scx-gpu/src/gpu_diffexp_tests.rs`.
//!
//! # What was missing
//!
//! Before this, the analytic sparse-nnz kernel's only correctness evidence was
//! a proptest against the dense kernel. That is the right *shape* of test and
//! it cannot see a dense-kernel bug — and the review established that the nnz
//! path had already re-derived one of the dense path's bugs independently. An
//! agreement between two implementations of one idea is not an oracle. A number
//! neither of them produced is.
//!
//! # The unlabelled-cell contract, referenced rather than asserted
//!
//! Every expected value was computed on the **labelled subset** of the fixture
//! (10 of 12 rows), because scipy has no notion of a cell outside the
//! comparison pool. The kernels here are handed the **full 12 rows** plus the
//! unlabelled sentinel. So "adding two unlabelled cells changes nothing" is not
//! a separate assertion bolted on afterwards — it is the only way these
//! assertions can pass at all, and it is stated against scipy instead of
//! against another SCX kernel. That closes ORG-7.21-5's `unknown-group cells`
//! item, which until now was three SCX arms agreeing with each other.
//!
//! Gene `g5` is the sharpest form of it: nonzero *only* in the two unlabelled
//! rows. Over the labelled pool it is identical to the all-zero gene `g2`, so
//! any answer that differs between the two proves a leak.

use super::wilcoxon_reference_values as r;

use crate::csc::wilcoxon::gene_stats_nnz;
use crate::diffexp::wilcoxon_rank_sum;

fn gene_names() -> Vec<String> {
    (0..r::N_VARS).map(|j| format!("g{j}")).collect()
}

fn group_names() -> Vec<String> {
    (0..r::N_GROUPS).map(|g| format!("grp{g}")).collect()
}

/// `(gene, group) -> (score, pval)` out of a `DiffExpResult`, keyed by **name**.
///
/// Keyed by name and not by position, deliberately: `DiffExpResult` sorts each
/// group's genes by score, and this fixture has three genes tied at score 0, so
/// their relative order is not a contract. Indexing positionally would pin the
/// sort, not the statistic.
fn by_name(result: &crate::diffexp::DiffExpResult) -> Vec<Vec<(f64, f64)>> {
    let names = gene_names();
    (0..r::N_GROUPS)
        .map(|g| {
            names
                .iter()
                .map(|want| {
                    let col = result.names[g]
                        .iter()
                        .position(|n| n == want)
                        .unwrap_or_else(|| panic!("gene {want} missing from group {g}"));
                    (result.scores[g][col], result.pvals[g][col])
                })
                .collect()
        })
        .collect()
}

fn dense_result(tie_correct: bool) -> Vec<Vec<(f64, f64)>> {
    let data = r::dense_x();
    let result = wilcoxon_rank_sum(
        &data,
        r::N_OBS,
        r::N_VARS,
        &gene_names(),
        &r::FIXTURE_GROUPS,
        &group_names(),
        None,  // 1-vs-rest
        false, // log_transformed
        false, // rankby_abs
        tie_correct,
        0,
    )
    .expect("dense wilcoxon on the reference fixture");
    by_name(&result)
}

/// Assert a `[group][gene]` result against a `[gene][group]` reference table.
///
/// `atol_z == 0.0` means exactly that — `assert_eq!`, not "a tolerance so tight
/// it may as well be". The two dense arms differ only in which table and which
/// tolerances they pass, so they share this body; a per-arm loop is how the two
/// would drift into asserting different things about the same result.
fn assert_matches_reference(
    oracle: &str,
    got: &[Vec<(f64, f64)>],
    want_z: &[[f64; r::N_GROUPS]; r::N_VARS],
    want_p: &[[f64; r::N_GROUPS]; r::N_VARS],
    atol_z: f64,
    atol_p: f64,
) {
    for (grp, per_gene) in got.iter().enumerate() {
        for (gene, &(z, p)) in per_gene.iter().enumerate() {
            let (wz, wp) = (want_z[gene][grp], want_p[gene][grp]);
            if atol_z == 0.0 {
                assert_eq!(z, wz, "{oracle}: gene g{gene} group {grp}: z");
            } else {
                assert!(
                    (z - wz).abs() <= atol_z,
                    "{oracle}: gene g{gene} group {grp}: z {z} vs {wz}"
                );
            }
            assert!(
                (p - wp).abs() <= atol_p,
                "{oracle}: gene g{gene} group {grp}: p {p} vs {wp}"
            );
        }
    }
}

/// The dense CSR kernel with `tie_correct = true`, against **two independent
/// references**: scipy's own p-value and scanpy's tie-corrected scores.
///
/// Neither is derived here. The first version of this test asserted `z` at
/// `abs = 0` against a `z` this repo's own generator had *rebuilt* from scipy's
/// `U` using the same variance formula as
/// [`wilcoxon_stats_from_rank_sum`](crate::diffexp::cpu::wilcoxon_stats_from_rank_sum)
/// — so a bug in the `Σ(t³−t)` term would have matched, and the uncorrected arm
/// could not have caught it because it runs with `tc = 0`. The constants did not
/// change when that was fixed (`max |Δp| = 0.0` against
/// `mannwhitneyu(...).pvalue`); their provenance did.
#[test]
fn dense_wilcoxon_matches_the_tie_corrected_references() {
    assert_matches_reference(
        "scanpy(tie_correct=True) z / scipy p",
        &dense_result(true),
        &r::SCANPY_Z_TIE_CORRECTED,
        &r::SCIPY_P_TIE_CORRECTED,
        r::Z_ATOL,
        r::P_CORRECTED_ATOL,
    );
}

/// scipy and scanpy agree on the tie-corrected p-value at **exactly 0.0**.
///
/// The premise that makes either of them an oracle. Two libraries implementing
/// `Σ(t³−t)/(n(n−1))` independently and landing on the same double is evidence
/// about the *formula*; SCX matching one of them is then evidence about SCX. Drop
/// this and the corrected arm rests on a single external implementation again,
/// which is one better than resting on SCX's own but not two.
#[test]
fn the_two_libraries_agree_on_the_tie_corrected_p_value() {
    for (gene, (sp, sc)) in r::SCIPY_P_TIE_CORRECTED
        .iter()
        .zip(r::SCANPY_P_TIE_CORRECTED.iter())
        .enumerate()
    {
        for (grp, (&a, &b)) in sp.iter().zip(sc.iter()).enumerate() {
            assert_eq!(
                a, b,
                "gene g{gene} group {grp}: scipy p {a} vs scanpy(tie_correct=True) p {b} — \
                 the two references for the corrected convention have diverged, so neither \
                 is confirming the other's tie term any more"
            );
        }
    }
}

/// The dense CSR kernel against **scanpy**, `tie_correct = false` — the default.
///
/// A separate reference on purpose. scipy always applies the `Σ(t³−t)`
/// correction, so it is not an oracle for this arm at all: on this fixture the
/// two conventions differ by 7.6e-02 in `z`, which is a different answer rather
/// than a looser one. scanpy is the implementation of *this* convention, and its
/// `scores` are float32 in the recarray, which is where
/// [`Z_ATOL`](r::Z_ATOL) comes from.
#[test]
fn dense_wilcoxon_matches_scanpy_without_tie_correction() {
    assert_matches_reference(
        "scanpy(tie_correct=False)",
        &dense_result(false),
        &r::SCANPY_Z_UNCORRECTED,
        &r::SCANPY_P_UNCORRECTED,
        r::Z_ATOL,
        r::P_UNCORRECTED_ATOL,
    );
}

/// The two conventions are not each other's tolerance.
///
/// The premise the two tests above rest on. Without it, someone could widen
/// [`Z_ATOL`](r::Z_ATOL) to `0.1`, point both arms at one table, and every
/// assertion would still pass while the `tie_correct` flag had stopped meaning
/// anything.
///
/// Both sides of this comparison come from **scanpy**, which takes a
/// `tie_correct` parameter (default `False`). So this measures a difference in
/// *convention*, not one between libraries — which is the honest framing: scanpy
/// can produce either answer, and SCX's default matches scanpy's default.
#[test]
fn the_two_tie_conventions_are_genuinely_different_answers() {
    let worst = r::SCANPY_Z_TIE_CORRECTED
        .iter()
        .zip(r::SCANPY_Z_UNCORRECTED.iter())
        .flat_map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()))
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e3 * r::Z_ATOL,
        "the corrected and uncorrected references differ by only {worst:.3e}, which is \
         within {:.0e} of the z tolerance — the fixture has stopped distinguishing the \
         two conventions and both arms would pass against either table",
        r::Z_ATOL
    );
}

/// The analytic sparse-nnz kernel against the same reference values.
///
/// Called directly rather than through `wilcoxon_rank_sum_streaming_csc`,
/// because that entry point is behind `SCX_ACCEL_WILCOXON_NNZ` (default off) —
/// an env-gated test would silently stop testing the moment the default moved.
///
/// `gene_stats_nnz` takes one gene's stored nonzeros, so the columns come from
/// [`csc_column`](r::csc_column), which **keeps stored zeros**: `g3` has exact
/// zeros at rows 2, 3 and 5 and they must join the implicit-zero tie block, not
/// the positive block. `g2` has no stored entries at all — the empty-column arm.
#[test]
fn nnz_wilcoxon_matches_the_same_reference_values() {
    let counts = r::group_cell_counts();
    for gene in 0..r::N_VARS {
        let (rows, vals) = r::csc_column(gene);
        for &tie_correct in &[true, false] {
            let stats = gene_stats_nnz(
                &rows,
                &vals,
                &r::FIXTURE_GROUPS,
                &counts,
                r::N_OBS,
                r::N_LABELLED,
                r::N_GROUPS,
                tie_correct,
                false, // log_transformed
            )
            .expect("nnz wilcoxon on the reference fixture");

            for (grp, &(z, p, _logfc)) in stats.iter().enumerate() {
                let (want_z, want_p, atol_p) = if tie_correct {
                    (
                        r::SCANPY_Z_TIE_CORRECTED[gene][grp],
                        r::SCIPY_P_TIE_CORRECTED[gene][grp],
                        r::P_CORRECTED_ATOL,
                    )
                } else {
                    (
                        r::SCANPY_Z_UNCORRECTED[gene][grp],
                        r::SCANPY_P_UNCORRECTED[gene][grp],
                        r::P_UNCORRECTED_ATOL,
                    )
                };
                let atol_z = r::Z_ATOL;
                assert!(
                    (z - want_z).abs() <= atol_z,
                    "nnz gene g{gene} group {grp} tie_correct={tie_correct}: z {z} vs {want_z}"
                );
                assert!(
                    (p - want_p).abs() <= atol_p,
                    "nnz gene g{gene} group {grp} tie_correct={tie_correct}: p {p} vs {want_p}"
                );
            }
        }
    }
}

/// The leak canary, stated as its own assertion so a failure names the cause.
///
/// `g5` is nonzero only in the two unlabelled rows; `g2` is all zeros. Over the
/// labelled pool they are the same column, so every kernel must return the same
/// answer for both. If an unlabelled cell reaches the rank pool or the rest
/// denominator, `g5` diverges from `g2` and this fires — where the two tests
/// above would only report "gene g5 does not match its reference", which is the
/// same failure without the diagnosis.
#[test]
fn a_gene_nonzero_only_in_unlabelled_cells_is_indistinguishable_from_an_empty_one() {
    for tie_correct in [true, false] {
        let got = dense_result(tie_correct);
        for (grp, per_gene) in got.iter().enumerate() {
            assert_eq!(
                per_gene[2], per_gene[5],
                "group {grp} tie_correct={tie_correct}: g5 (nonzero only in unlabelled rows) \
                 differs from g2 (all zeros) — an unlabelled cell reached the comparison pool"
            );
        }
    }
}
