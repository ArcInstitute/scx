//! Golden values for [`super::binned_dispersion_norm`], pinned from the
//! **pandas reference** it replaces — pyscx's `_hvg_helpers.binned_dispersion_norm`
//! (itself the verbatim port of scanpy's seurat-flavor
//! `_get_disp_stats` / `_postprocess_dispersions_seurat`), executed under
//! pandas 2.x / numpy 2.x in the repo `.venv` on 2026-08-29. House rule
//! (`pflog_reference_values.rs`): pin from the external reference, never from
//! what SCX returns. The end-to-end scanpy oracle lives in
//! `pyscx/tests/test_hvg.py::test_column_parity_with_scanpy` (allclose 1e-5
//! against `sc.pp.highly_variable_genes(flavor="seurat")`).
//!
//! Comparisons are at 1e-12 rather than bit-exact: pandas' groupby mean/std
//! carry Kahan compensation, so the last ulp of a bin statistic is an
//! implementation detail, not part of the contract. The singleton-bin `1.0`
//! IS exact (`d / d`), and asserted as such.

// The goldens are verbatim pandas outputs; several happen to be the closest
// f64 to 1/√2 (a two-value bin z-scores to exactly ±1/√2), and clippy wants
// them spelled `FRAC_1_SQRT_2`. They stay literal: a pinned golden must read
// as the reference produced it, and its neighbours (…74, …76, …79) differ in
// the last ulp precisely because they are measurements, not the constant.
#![allow(clippy::approx_constant)]

use super::binned_dispersion_norm;

const NAN: f64 = f64::NAN;

fn assert_close(got: &[f64], want: &[f64], case: &str) {
    assert_eq!(got.len(), want.len(), "{case}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if w.is_nan() {
            assert!(g.is_nan(), "{case}: gene {i} should be NaN, got {g}");
            continue;
        }
        let tol = 1e-12 * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol,
            "{case}: gene {i}: got {g}, want {w} (diff {})",
            (g - w).abs()
        );
    }
}

#[test]
fn two_bins_two_genes_each() {
    let got = binned_dispersion_norm(&[0.0, 0.1, 1.0, 1.1], &[1.0, 2.0, 3.0, 5.0], 2);
    let want = [
        -0.7071067811865475,
        0.7071067811865475,
        -0.7071067811865475,
        0.7071067811865475,
    ];
    assert_close(&got, &want, "two bins");
}

/// The scanpy quirk: a one-gene bin gets normalized dispersion EXACTLY 1.0
/// via `dev = avg; avg = 0` — not 0, which the tempting `dev = 1` would give.
#[test]
fn singleton_bin_is_exactly_one() {
    let got = binned_dispersion_norm(&[0.0, 0.05, 2.0], &[1.0, 3.0, 7.0], 2);
    let want = [-0.7071067811865475, 0.7071067811865475, 1.0];
    assert_close(&got, &want, "singleton bin");
    assert_eq!(
        got[2], 1.0,
        "the singleton must be exactly 1.0, bit for bit"
    );
}

/// NaN dispersions stay NaN in the output AND are skipped by the bin stats —
/// the neighbours' z-scores are computed as if the NaN gene were absent.
#[test]
fn nan_dispersion_is_preserved_and_skipped() {
    let got = binned_dispersion_norm(&[0.0, 0.1, 0.2, 1.0, 1.1], &[1.0, NAN, 2.0, 4.0, 8.0], 2);
    let want = [
        -0.7071067811865475,
        NAN,
        0.7071067811865475,
        -0.7071067811865475,
        0.7071067811865475,
    ];
    assert_close(&got, &want, "nan dispersion");
}

/// `mn == mx`: pandas widens the zero-width range by 0.1% on both ends before
/// binning (0.001 absolute when the value is 0). All genes land in one bin.
#[test]
fn all_equal_means_widen_the_range() {
    let got = binned_dispersion_norm(&[0.5, 0.5, 0.5], &[1.0, 2.0, 4.0], 3);
    let want = [-0.8728715609439697, -0.2182178902359925, 1.0910894511799618];
    assert_close(&got, &want, "all-equal means");

    let got = binned_dispersion_norm(&[0.0, 0.0], &[1.0, 3.0], 2);
    let want = [-0.7071067811865475, 0.7071067811865475];
    assert_close(&got, &want, "all-zero means");
}

/// Right-closed intervals: the gene exactly on the interior edge (0.5 with
/// edges at [−0.001, 0.5, 1.0]) lands in the bin to its LEFT, grouping with
/// {0.0, 0.25} — a left-closed cut would put it with {0.75, 1.0} and change
/// every value in both bins.
#[test]
fn gene_on_an_interior_edge_lands_left() {
    let got = binned_dispersion_norm(&[0.0, 0.5, 1.0, 0.25, 0.75], &[1.0, 2.0, 3.0, 4.0, 5.0], 2);
    let want = [
        -0.8728715609439697,
        -0.2182178902359925,
        -0.7071067811865475,
        1.0910894511799618,
        0.7071067811865475,
    ];
    assert_close(&got, &want, "interior edge");
}

#[test]
fn one_gene_is_a_singleton_bin() {
    let got = binned_dispersion_norm(&[0.7], &[2.5], 20);
    assert_close(&got, &[1.0], "one gene");
}

/// More bins than distinct values leaves most bins unused — the pandas
/// `.map()`-not-`.loc[]` robustness case (unused categories must not panic
/// or perturb the observed bins' stats).
#[test]
fn unused_bins_do_not_disturb_observed_ones() {
    let got = binned_dispersion_norm(&[0.0, 0.0, 1.0, 1.0], &[1.0, 2.0, 3.0, 5.0], 10);
    let want = [
        -0.7071067811865475,
        0.7071067811865475,
        -0.7071067811865475,
        0.7071067811865475,
    ];
    assert_close(&got, &want, "unused bins");
}

/// Thirty genes over the scanpy-default 20 bins: a mix of 1- and 2-gene bins,
/// exercising edge placement on accumulated-float means (0.15000000000000002
/// and friends) exactly as pandas binned them.
#[test]
fn thirty_genes_twenty_bins_golden() {
    let log_means: Vec<f64> = (0..30).map(|i| 0.05 * i as f64).collect();
    let log_dispersions: Vec<f64> = (0..30)
        .map(|i| ((i * 37) % 11) as f64 + 0.1 * i as f64)
        .collect();
    let got = binned_dispersion_norm(&log_means, &log_dispersions, 20);
    let want = [
        -0.7071067811865475,
        0.7071067811865475,
        1.0,
        -0.7071067811865475,
        0.7071067811865476,
        1.0,
        -0.7071067811865476,
        0.7071067811865475,
        1.0,
        -0.7071067811865475,
        0.7071067811865474,
        1.0,
        -0.7071067811865472,
        0.7071067811865476,
        1.0,
        1.0,
        0.7071067811865474,
        -0.7071067811865476,
        1.0,
        0.7071067811865479,
        -0.7071067811865475,
        1.0,
        -0.7071067811865472,
        0.7071067811865476,
        1.0,
        -0.7071067811865475,
        0.7071067811865475,
        1.0,
        -0.7071067811865475,
        0.7071067811865475,
    ];
    assert_close(&got, &want, "thirty genes");
}

/// Degenerate inputs answer rather than erroring (the pandas reference raises
/// on an empty cut; no SCX caller can reach that, and all-NaN per gene is the
/// honest answer for the others).
#[test]
fn degenerate_inputs_yield_nan_not_panic() {
    assert!(binned_dispersion_norm(&[], &[], 20).is_empty());
    let got = binned_dispersion_norm(&[0.1, 0.2], &[1.0, 2.0], 0);
    assert!(got.iter().all(|v| v.is_nan()), "n_bins == 0 → all NaN");
    let got = binned_dispersion_norm(&[NAN, NAN], &[1.0, 2.0], 5);
    assert!(got.iter().all(|v| v.is_nan()), "all-NaN means → all NaN");
}

/// A gene whose mean is NaN gets no bin and a NaN output, without perturbing
/// the genes that did bin.
#[test]
fn nan_mean_gene_is_excluded_from_binning() {
    // With the NaN gene out of nanmin/nanmax, the two real genes span the
    // whole range and land in one singleton bin each → exactly 1.0.
    let got = binned_dispersion_norm(&[0.0, NAN, 0.1], &[1.0, 100.0, 2.0], 2);
    let want = [1.0, NAN, 1.0];
    assert_close(&got, &want, "nan mean");
}
