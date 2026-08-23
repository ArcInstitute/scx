//! The three `discrimination_score` divergences from cell-eval (review §7.13),
//! and the fixtures that make each one visible.
//!
//! `docs/scanpy.md` pins this metric at `exact (abs=0)` against cell-eval. It was
//! not exact in three ways, and none of them was detectable by the existing
//! parity fixture (`_make_cell_eval_adata`, 400×20 continuous random): it has no
//! ties, no duplicated gene names and no zero-norm rows, so all three passed
//! vacuously. These are the fixtures ORG-7.21-5 names.
//!
//! The reference is `cell_eval/metrics/_anndata.py::discrimination_score`
//! (read at `/home/nickyoungblut/dev/python/cell-eval`, v0.2.x):
//!
//! ```python
//! include_mask   = np.flatnonzero(data.genes != p)          # ALL matching columns
//! distances      = skm.pairwise_distances(real[:, m], pred[p, m], metric=metric)
//! sorted_indices = np.argsort(distances)
//! rank           = np.flatnonzero(sorted_indices == p_index)[0]   # argsort POSITION
//! norm_ranks[p]  = 1 - rank / data.perts.size
//! ```
//!
//! One caveat worth stating rather than glossing: `np.argsort`'s default kind is
//! `quicksort`, which is **not** a stable sort, so the reference's own tie
//! behaviour is formally implementation-defined. We match the stable reading —
//! ties break by index, which is what `kind="stable"` gives and what numpy
//! produces on the small already-tied arrays this metric sees. That is a choice,
//! not a derivation, and it is the only interpretation under which the reference's
//! score is reproducible at all.
//!
//! Distances are symmetric in all three metrics, so computing `d(pred, real)`
//! where the reference computes `d(real, pred)` is not a fourth divergence.

use super::*;

/// Ties: rank must be the argsort *position*, not the count of strictly-smaller
/// distances.
///
/// Fixture: every real effect is the same vector, so a prediction equal to it is
/// equidistant (0.0) from all `P` of them. cell-eval puts the correct
/// perturbation at argsort position `p`; counting strictly-smaller distances puts
/// every perturbation at rank 0 and scores them all a perfect 1.0.
///
/// With `P = 3` and `p = 2` the review's worked example is score `1/3`; ours was
/// `1.0`. A metric that reports a perfect score for a model that cannot tell three
/// perturbations apart is the failure mode.
#[test]
fn ties_rank_by_index_like_cell_eval_argsort() {
    let n_perts = 3usize;
    let n_genes = 4usize;
    // Three identical real effects → every distance is a tie.
    let real: Vec<f64> = std::iter::repeat_n([1.0, 2.0, 3.0, 4.0], n_perts)
        .flatten()
        .collect();
    let pred = real.clone();
    let perts: Vec<String> = (0..n_perts).map(|i| format!("p{i}")).collect();

    let got = compute_discrimination_score(
        &real,
        &pred,
        n_perts,
        n_genes,
        &perts,
        None,
        DistanceMetric::Euclidean,
        false,
    )
    .unwrap();

    // `1 - p/P` for each p: the stable-argsort position of p among P equal keys.
    let want = [1.0, 1.0 - 1.0 / 3.0, 1.0 - 2.0 / 3.0];
    for (p, (&g, &w)) in got.scores.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-12,
            "pert {p}: got {g}, cell-eval gives {w} (all scores {:?})",
            got.scores
        );
    }
}

/// Duplicate `var_names`: `exclude_target_gene` must drop **every** column whose
/// name matches the perturbation, matching `np.flatnonzero(genes != p)`.
///
/// A `HashMap<&str, usize>` keeps only the last index, so one of the duplicated
/// target columns survives — and the surviving copy restores exactly the trivial
/// self-match the flag exists to remove. Duplicate gene symbols are routine in
/// 10x data, so this is not a synthetic worry.
///
/// Fixture: gene `"g0"` appears twice. Perturbation `"g0"`'s real effect is a
/// large spike in *both* copies and nothing elsewhere; the other perturbations
/// differ from it only outside the duplicated columns. With both `g0` columns
/// excluded, `p0`'s prediction is identical to every real effect on the remaining
/// genes, so it ties at index 0 and scores 1.0 for the wrong reason — and with
/// only one excluded, the surviving spike makes it 1.0 for the *trivial* reason.
/// The observable difference is the excluded width, so the test asserts on that
/// directly via a distance that must ignore both columns.
#[test]
fn duplicate_var_names_exclude_every_matching_column() {
    let n_perts = 2usize;
    let n_genes = 3usize;
    // Columns: g0, g0 (duplicate), g1.
    let gene_names = vec!["g0".to_string(), "g0".to_string(), "g1".to_string()];
    let perts = vec!["g0".to_string(), "g1".to_string()];

    // Real: p0 spikes both g0 columns; p1 spikes g1.
    let real = vec![
        10.0, 10.0, 0.0, // p0
        0.0, 0.0, 10.0, // p1
    ];
    // Pred for p0 spikes only the FIRST g0 column, and matches p1 on g1.
    // If both g0 columns are excluded, pred[p0] restricted to {g1} is [10.0],
    // which equals real[p1] restricted to {g1} and NOT real[p0] ([0.0]) — so p0
    // ranks second and scores 0.5. If only the second g0 column is excluded, the
    // surviving first column (10.0 vs 10.0 for p0, 10.0 vs 0.0 for p1) pulls p0
    // to rank 0 and scores 1.0.
    let pred = vec![
        10.0, 0.0, 10.0, // p0
        0.0, 0.0, 10.0, // p1
    ];

    let got = compute_discrimination_score(
        &real,
        &pred,
        n_perts,
        n_genes,
        &perts,
        Some(&gene_names),
        DistanceMetric::Euclidean,
        true,
    )
    .unwrap();

    assert!(
        (got.scores[0] - 0.5).abs() < 1e-12,
        "pert g0 scored {} — expected 0.5, which requires BOTH duplicated g0 \\
         columns to be excluded. 1.0 means one survived and restored the \\
         self-match that exclude_target_gene exists to remove. Scores {:?}",
        got.scores[0],
        got.scores
    );
}

/// A zero-norm effect vector under a masked cosine distance must score `1.0`
/// (maximally distant), not `0.0` (identical).
///
/// `distances::cosine_distance` returns `1.0` for a zero norm and clamps to
/// `[0, 2]`; sklearn's `cosine_distances` — which is what the reference calls —
/// does the same. `discrimination.rs`'s masked copy returned `0.0` and applied no
/// clamp. An all-zero effect vector therefore beat every genuine distance, which
/// inverts the ranking rather than perturbing it.
///
/// Getting this fixture wrong the first time is worth recording, because the
/// wrong version passed: the exclusion is keyed to the perturbation `p` whose
/// score is being computed and applied to *every* row of the comparison, not
/// row-by-row. So making `real[p1]` nonzero only in `p1`'s own target column does
/// nothing — under `p0`'s mask that column is still there. The zero-norm row has
/// to be nonzero only in the column `p0` excludes.
///
/// It also has to *steal rank* from the correct match, or the score is 1.0 either
/// way: if `p0`'s own real effect is already at distance 0, a bogus 0.0 elsewhere
/// changes nothing. So `pred[p0]` is deliberately a poor-but-nonzero match for
/// `real[p0]` (cosine ≈ 0.293), and `real[p1]` is the zero-norm row. Under the
/// old semantics its 0.0 undercuts 0.293 and pushes the correct perturbation to
/// rank 1 (score 0.5); under the correct semantics its 1.0 loses and the correct
/// perturbation keeps rank 0 (score 1.0).
///
/// Independent of the tie fix by construction: 0.293 and 1.0 are not equal, so no
/// tie-breaking rule is in play here.
#[test]
fn masked_cosine_treats_a_zero_norm_vector_as_maximally_distant() {
    let n_perts = 2usize;
    let n_genes = 3usize;
    let gene_names = vec!["g0".to_string(), "g1".to_string(), "g2".to_string()];
    let perts = vec!["g0".to_string(), "g1".to_string()];

    // p0 excludes column 0. Masked columns are {1, 2}.
    let real = vec![
        0.0, 1.0, 0.0, // p0 -> masked [1, 0]
        9.0, 0.0, 0.0, // p1 -> masked [0, 0]  <- zero norm under p0's mask
    ];
    let pred = vec![
        0.0, 1.0, 1.0, // p0 -> masked [1, 1]; cosine to [1, 0] is 1 - 1/sqrt(2)
        9.0, 0.0, 0.0, // p1
    ];

    let got = compute_discrimination_score(
        &real,
        &pred,
        n_perts,
        n_genes,
        &perts,
        Some(&gene_names),
        DistanceMetric::Cosine,
        true,
    )
    .unwrap();

    // Premise: the correct match must NOT be at distance 0, or this fixture
    // cannot distinguish the two semantics at all (which is how the first
    // version of it passed against the broken code).
    // Column 0 is p0's target gene, so it is the one excluded.
    let keep = [false, true, true];
    let correct_dist = super::super::distances::point_distance_masked(
        &pred[0..3],
        &real[0..3],
        n_genes,
        Some(&keep),
        DistanceMetric::Cosine,
    );
    assert!(
        correct_dist > 1e-6,
        "premise broken: p0's correct match is at distance {correct_dist}, so a \
         spurious 0.0 elsewhere cannot steal rank 0 and the test is vacuous"
    );

    assert!(
        (got.scores[0] - 1.0).abs() < 1e-12,
        "pert g0 scored {} — expected 1.0. A zero-norm masked vector must be \
         distance 1.0 (maximally distant), not 0.0, or it undercuts every genuine \
         distance and steals rank 0. Scores {:?}",
        got.scores[0],
        got.scores
    );
}

// A fourth test lived here, asserting that `discrimination.rs`'s masked kernel
// agreed with `distances::point_distance` on the zero-norm and clamp conventions.
// It is gone rather than kept green: now that there is exactly one kernel, that
// assertion is a tautology, and a tautology in a test file is worse than nothing
// — it reads as coverage. The conventions themselves are stated once, where they
// live, in `distances::tests::cosine_zero_norm_and_range_hold_on_both_entry_points`.
//
// What that test was really guarding — that this module does not grow a second
// kernel again — is not a runtime property, so it is a CI guard (ORG-7.21-5)
// rather than a `#[test]`.
