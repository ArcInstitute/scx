//! Every documented cell-eval / arc-bench parity claim, asserted with no Python
//! installed (ORG-7.21-4).
//!
//! Provenance, the bars and the two deliberately-unpinned claims live in
//! [`super::cell_eval_reference_values`]. What matters here is the coverage
//! statement: before this module, `pyscx/tests/test_cell_eval_parity.py` was the
//! only thing comparing these metrics to their references, and it skips
//! everywhere `cell_eval` is absent — which is CI and all six conda envs.

use super::cell_eval_reference_values as r;
use super::distances::DistanceBackend;
use super::{
    compute_bulk_metrics, compute_control_baseline, compute_discrimination_score,
    compute_energy_distance, compute_knockdown_efficiency, compute_log_deviation, BulkMetric,
    DistanceMetric,
};

/// Assert a per-perturbation vector against its pinned table, keyed by position
/// in [`r::CE_SCORED_NAMES`] — and check the names line up first, because a
/// kernel that reordered its output would otherwise be compared against the
/// wrong perturbation's value.
fn assert_by_name(what: &str, names: &[String], got: &[f64], want: &[f64], atol: f64) {
    assert_by_name_rel(what, names, got, want, atol, 0.0)
}

/// As [`assert_by_name`], with numpy's `atol + rtol·|want|` bar — the one
/// `assert_allclose` actually applies, and the one cell-eval's f32 storage
/// requires for values above ~1.
fn assert_by_name_rel(
    what: &str,
    names: &[String],
    got: &[f64],
    want: &[f64],
    atol: f64,
    rtol: f64,
) {
    assert_eq!(
        names,
        r::owned(&r::CE_SCORED_NAMES),
        "{what}: perturbation order differs from cell-eval's"
    );
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let bar = atol + rtol * w.abs();
        assert!(
            (g - w).abs() <= bar,
            "{what}[{}] = {g} vs cell-eval {w} (|Δ| = {:.3e}, bar {bar:.3e})",
            r::CE_SCORED_NAMES[i],
            (g - w).abs()
        );
    }
}

#[test]
fn bulk_metrics_match_cell_eval() {
    let names = r::owned(&r::CE_ROW_NAMES);
    let res = compute_bulk_metrics(
        &r::means_real(),
        &r::means_pred(),
        r::CE_CTRL_IDX,
        r::CE_N_ROWS,
        r::CE_N_GENES,
        &names,
        &[
            BulkMetric::PearsonDelta,
            BulkMetric::Mse,
            BulkMetric::Mae,
            BulkMetric::MseDelta,
            BulkMetric::MaeDelta,
        ],
    )
    .unwrap();
    for (key, want) in [
        ("pearson_delta", &r::CE_PEARSON_DELTA[..]),
        ("mse", &r::CE_MSE[..]),
        ("mae", &r::CE_MAE[..]),
        ("mse_delta", &r::CE_MSE_DELTA[..]),
        ("mae_delta", &r::CE_MAE_DELTA[..]),
    ] {
        let got = res
            .metrics
            .get(key)
            .unwrap_or_else(|| panic!("{key} missing from the result"));
        assert_by_name_rel(key, &res.pert_names, got, want, r::CE_ATOL, r::CE_RTOL);
    }
}

/// Discrimination scores are `1 − rank/P` over an integer rank, so equality is
/// exact: a mismatch means the *ranking* differs, which no tolerance should
/// absorb. All six documented configurations are covered — three metrics ×
/// `exclude_target_gene` on and off — because the target-gene mask is the part
/// `discrimination.rs` had a private second copy of.
#[test]
fn discrimination_scores_match_cell_eval_exactly() {
    let names = r::owned(&r::CE_SCORED_NAMES);
    let genes = r::owned(&r::CE_GENE_NAMES);
    let (real, pred) = (r::effects_real(), r::effects_pred());
    for (metric, excl, want) in [
        (DistanceMetric::L1, true, &r::CE_DISCRIM_L1_EXCL[..]),
        (DistanceMetric::L1, false, &r::CE_DISCRIM_L1_ALL[..]),
        (DistanceMetric::Euclidean, true, &r::CE_DISCRIM_L2_EXCL[..]),
        (DistanceMetric::Euclidean, false, &r::CE_DISCRIM_L2_ALL[..]),
        (DistanceMetric::Cosine, true, &r::CE_DISCRIM_COSINE_EXCL[..]),
        (DistanceMetric::Cosine, false, &r::CE_DISCRIM_COSINE_ALL[..]),
    ] {
        let res = compute_discrimination_score(
            &real,
            &pred,
            r::CE_N_ROWS - 1,
            r::CE_N_GENES,
            &names,
            Some(&genes),
            metric,
            excl,
        )
        .unwrap();
        assert_by_name(
            &format!("discrimination {metric:?} exclude_target_gene={excl}"),
            &res.pert_names,
            &res.scores,
            want,
            0.0,
        );
    }
}

/// The e-distance correlation, at the tolerance
/// `docs/scanpy/accel-perturbation-metrics.md` states for it.
///
/// Both backends are checked. `Gemm` is what `dtype="f32"` selects by default and
/// `Scalar` is the arm the docs name as the reference shape; running only one of
/// them would leave the other free to diverge from cell-eval while the pair
/// agreed with each other.
#[test]
fn the_edistance_correlation_matches_cell_eval() {
    let names = r::owned(&r::CE_SCORED_NAMES);
    let groups = r::cell_groups();
    let scored: Vec<u32> = (0..r::CE_N_ROWS as u32)
        .filter(|g| *g != r::CE_CTRL_IDX as u32)
        .collect();
    for backend in [DistanceBackend::Gemm, DistanceBackend::Scalar] {
        let res = compute_energy_distance(
            &r::cells_real(),
            &r::cells_pred(),
            &groups,
            &groups,
            r::CE_CTRL_IDX as u32,
            &names,
            &scored,
            r::CE_N_GENES,
            DistanceMetric::Euclidean,
            backend,
        )
        .unwrap();
        assert!(
            (res.correlation - r::CE_EDISTANCE_CORR).abs() <= r::CE_EDISTANCE_ATOL,
            "{backend:?}: correlation {} vs cell-eval {} (|Δ| = {:.3e}, bar {:.0e})",
            res.correlation,
            r::CE_EDISTANCE_CORR,
            (res.correlation - r::CE_EDISTANCE_CORR).abs(),
            r::CE_EDISTANCE_ATOL
        );
    }
}

/// arc-bench's control baseline: the per-gene mean over control cells.
#[test]
fn the_control_baseline_matches_arc_bench() {
    let (indptr, indices, data) = r::cells_real_csr();
    let labels = r::owned(&r::CE_CELL_LABELS);
    let got = compute_control_baseline(
        &indptr,
        &indices,
        &data,
        &labels,
        r::CE_ROW_NAMES[r::CE_CTRL_IDX],
        r::CE_N_GENES,
    )
    .unwrap();
    for (g, (got_v, want_v)) in got.iter().zip(r::CE_KD_BASELINE.iter()).enumerate() {
        assert!(
            (got_v - want_v).abs() <= r::CE_ATOL + r::CE_RTOL * want_v.abs(),
            "baseline[{g}] = {got_v} vs arc-bench {want_v}"
        );
    }
}

/// Knockdown efficiency and log deviation, including **which cells have no
/// value**.
///
/// The NaN mask is asserted before the values and separately: arc-bench reports
/// NaN for control cells and for cells whose perturbation names no gene in the
/// matrix, and a kernel that filled those with `0.0` would otherwise pass a
/// value comparison over the cells it did fill.
#[test]
fn knockdown_and_log_deviation_match_arc_bench() {
    let (indptr, indices, data) = r::cells_real_csr();
    let labels = r::owned(&r::CE_CELL_LABELS);
    let genes = r::owned(&r::CE_GENE_NAMES);
    let ctrl = r::CE_ROW_NAMES[r::CE_CTRL_IDX];
    let baseline: Vec<f64> = r::CE_KD_BASELINE.to_vec();

    let kd = compute_knockdown_efficiency(
        &indptr,
        &indices,
        &data,
        &labels,
        ctrl,
        &genes,
        &baseline,
        r::CE_KD_EPS,
    )
    .unwrap();
    let baseline_log: Vec<f64> = baseline.iter().map(|v| v.ln_1p()).collect();
    let fc = compute_log_deviation(
        &indptr,
        &indices,
        &data,
        &labels,
        ctrl,
        &genes,
        &baseline_log,
        true,
    )
    .unwrap();

    for (name, got) in [("knockdown_efficiency", &kd), ("log_deviation", &fc)] {
        let mask: Vec<bool> = got.iter().map(|v| v.is_nan()).collect();
        assert_eq!(
            mask,
            r::CE_KD_IS_NAN.to_vec(),
            "{name}: the set of cells with no value differs from arc-bench's"
        );
    }
    assert!(
        r::CE_KD_IS_NAN.iter().any(|b| *b) && r::CE_KD_IS_NAN.iter().any(|b| !*b),
        "the fixture must have both scored and unscored cells, or the NaN-mask \
         assertion is satisfied by an all-or-nothing answer"
    );

    for (name, got, want) in [
        ("knockdown_efficiency", &kd, &r::CE_KD_EFFICIENCY[..]),
        ("log_deviation", &fc, &r::CE_KD_LOG_DEVIATION[..]),
    ] {
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            if r::CE_KD_IS_NAN[i] {
                continue;
            }
            assert!(
                (*g as f64 - w).abs() <= r::CE_ATOL + r::CE_RTOL * w.abs(),
                "{name}[{i}] = {g} vs arc-bench {w} (|Δ| = {:.3e}, bar {:.3e})",
                (*g as f64 - w).abs(),
                r::CE_ATOL + r::CE_RTOL * w.abs()
            );
        }
    }
}
