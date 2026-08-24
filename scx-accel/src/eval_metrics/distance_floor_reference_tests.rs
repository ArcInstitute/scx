//! The gemm distance expansion on large-norm f32 input (§7.12, ORG-7.21-5).
//!
//! Two kernels share the expansion and therefore share the defect: the exact-kNN
//! path in `crate::neighbors::cpu` and the blocked pairwise kernel in
//! [`super::distances`]. Both are asserted here against the same fixture, so a
//! fix applied to one and not the other fails rather than passes quietly — the
//! shape ORG-7.21-5 asked for after `discrimination.rs` was found carrying a
//! private second copy of three metrics.
//!
//! Provenance, the measured collapse, and why neighbour *order* is not pinned
//! live in [`super::distance_floor_reference_values`].

use super::distance_floor_reference_values as r;
use super::distances::{mean_pairwise_distance_self, DistanceBackend};
use super::DistanceMetric;

/// Row 0's three nearest, as `(row, distance)`.
fn row0_neighbours() -> Vec<(usize, f64)> {
    let x = r::floor_x();
    let (idx, dist) =
        crate::neighbors::cpu::build_knn_exact(&x, r::FLOOR_N_ROWS, r::FLOOR_N_DIMS, 3);
    (0..3).map(|j| (idx[j], dist[j])).collect()
}

/// The kNN path must report the planted separations, not zero.
///
/// Before the fix the f32 expansion returned `d² = 0.0` for both planted pairs,
/// so two rows an order of magnitude apart in distance came back identical — and
/// identical to a genuine duplicate. No tolerance is needed on the planted
/// values themselves: `2⁻⁵` and `2⁻⁴` are exact in f64.
#[test]
fn exact_knn_resolves_two_near_duplicates_that_are_not_the_same_distance() {
    let got = row0_neighbours();
    let mut nearest: Vec<usize> = got.iter().take(2).map(|(i, _)| *i).collect();
    nearest.sort_unstable();
    assert_eq!(
        nearest,
        r::FLOOR_ROW0_NEAREST.to_vec(),
        "row 0's two nearest are {nearest:?}, scipy says {:?}",
        r::FLOOR_ROW0_NEAREST
    );

    for (row, want) in r::FLOOR_PLANTED {
        let (_, got_d) = got
            .iter()
            .find(|(i, _)| *i == row)
            .copied()
            .unwrap_or_else(|| panic!("row {row} is not among row 0's three nearest: {got:?}"));
        let rel = (got_d - want).abs() / want;
        assert!(
            rel <= r::FLOOR_DIST_RTOL,
            "row 0 -> row {row}: got d = {got_d:.6e}, exact is {want:.6e} \
             (relative {rel:.3e}, bar {:.3e}). The f32 gemm expansion returned \
             exactly 0.0 here before §7.12 was fixed.",
            r::FLOOR_DIST_RTOL
        );
    }
}

/// The two planted pairs must not report the *same* distance.
///
/// The premise for the test above, and the sharper statement of the bug: a
/// tolerance assertion could be satisfied by a kernel that had merely got less
/// wrong, whereas `d(0,1) == d(0,2)` is what the collapsed expansion actually
/// produced — both exactly zero.
#[test]
fn the_two_planted_distances_are_distinguishable() {
    let got = row0_neighbours();
    let pick = |row: usize| {
        got.iter()
            .find(|(i, _)| *i == row)
            .map(|(_, d)| *d)
            .unwrap_or_else(|| panic!("row {row} not in row 0's neighbours: {got:?}"))
    };
    let (r1, d1) = r::FLOOR_PLANTED[0];
    let (r2, d2) = r::FLOOR_PLANTED[1];
    let (g1, g2) = (pick(r1), pick(r2));
    assert!(
        g1 < g2,
        "the planted pairs came back as d({r1})={g1:.6e} and d({r2})={g2:.6e}; \
         by construction they are {d1:.6e} and {d2:.6e}"
    );
    assert!(
        (g2 - g1) > 0.5 * (d2 - d1),
        "the gap between the two planted distances collapsed to {:.6e}, against \
         {:.6e} by construction",
        g2 - g1,
        d2 - d1
    );
}

/// The blocked pairwise kernel on the same fixture, against scipy's mean.
///
/// The convention pinned is the one this kernel documents —
/// `sum(d(i,j) for i<j) * 2 / (n*n)`, sklearn's
/// `pairwise_distances(a, a).mean()` with the self diagonal forced to 0. A first
/// version of the reference averaged the off-diagonal entries instead and
/// disagreed by exactly `552/576`.
///
/// Over all 24 rows the two collapsed pairs are 2 of 576 and move the mean far
/// too little for this assertion to catch §7.12 — reverting the fix leaves it
/// green. `the_cluster_mean_collapses_without_the_floor` below is the arm that
/// covers this kernel; this one covers the convention and the magnitude.
#[test]
fn the_blocked_pairwise_kernel_matches_scipy_on_large_norm_f32() {
    let x = r::floor_x();
    let got = mean_pairwise_distance_self(
        &x,
        r::FLOOR_N_ROWS,
        r::FLOOR_N_DIMS,
        DistanceMetric::Euclidean,
        DistanceBackend::Gemm,
    )
    .unwrap();
    let rel = (got - r::FLOOR_MEAN_PAIRWISE).abs() / r::FLOOR_MEAN_PAIRWISE;
    assert!(
        rel <= r::FLOOR_DIST_RTOL,
        "mean pairwise distance {got:.10e} vs scipy {:.10e} (relative {rel:.3e}, \
         bar {:.3e})",
        r::FLOOR_MEAN_PAIRWISE,
        r::FLOOR_DIST_RTOL
    );
}

/// `docs/scanpy.md`'s claim, tested as it is written: **f32 + gemm** against
/// **f64 + scalar**, on input large-norm enough to break it.
///
/// The floor is keyed to the accumulator's precision, so the scalar f64 arm
/// never reaches the fallback at all — the two agree because the gemm arm
/// recomputes the pairs it cannot resolve, not because a tolerance was widened
/// until they did.
#[test]
fn f32_gemm_matches_f64_scalar_on_large_norm_input() {
    let x32 = r::floor_x();
    let x64: Vec<f64> = x32.iter().map(|&v| v as f64).collect();
    let gemm_f32 = mean_pairwise_distance_self(
        &x32,
        r::FLOOR_N_ROWS,
        r::FLOOR_N_DIMS,
        DistanceMetric::Euclidean,
        DistanceBackend::Gemm,
    )
    .unwrap();
    let scalar_f64 = mean_pairwise_distance_self(
        &x64,
        r::FLOOR_N_ROWS,
        r::FLOOR_N_DIMS,
        DistanceMetric::Euclidean,
        DistanceBackend::Scalar,
    )
    .unwrap();
    let rel = (gemm_f32 - scalar_f64).abs() / scalar_f64;
    assert!(
        rel <= r::FLOOR_DIST_RTOL,
        "f32+gemm {gemm_f32:.10e} vs f64+scalar {scalar_f64:.10e} \
         (relative {rel:.3e}, bar {:.3e})",
        r::FLOOR_DIST_RTOL
    );
}

/// The arm that actually catches §7.12 in **this** kernel.
///
/// Over the whole fixture the defect is 2 pairs in 576 and invisible to a mean;
/// over the planted cluster alone every pair is one the f32 expansion loses, so
/// the mean goes from [`r::FLOOR_CLUSTER_MEAN_PAIRWISE`] to exactly 0. Reverting
/// the fix in `eval_metrics/distances.rs` reds this and nothing else in the
/// file, which is why it exists: the falsification of the first version of these
/// tests passed on the reverted kernel, and that gap was the finding.
#[test]
fn the_cluster_mean_collapses_without_the_floor() {
    let x = r::floor_cluster_x();
    let got = mean_pairwise_distance_self(
        &x,
        r::FLOOR_CLUSTER_N_ROWS,
        r::FLOOR_N_DIMS,
        DistanceMetric::Euclidean,
        DistanceBackend::Gemm,
    )
    .unwrap();
    let rel = (got - r::FLOOR_CLUSTER_MEAN_PAIRWISE).abs() / r::FLOOR_CLUSTER_MEAN_PAIRWISE;
    assert!(
        rel <= r::FLOOR_DIST_RTOL,
        "the cluster's mean pairwise distance came back {got:.10e} against \
         {:.10e} (relative {rel:.3e}, bar {:.3e}). The f32 gemm expansion \
         returns 0.0 for every pair here.",
        r::FLOOR_CLUSTER_MEAN_PAIRWISE,
        r::FLOOR_DIST_RTOL
    );
}
