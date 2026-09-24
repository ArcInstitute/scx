//! The three `discrimination_score` divergences from cell-eval (review §7.13),
//! and the fixtures that make each one visible.
//!
//! `docs/scanpy/accel-perturbation-metrics.md` pins this metric at
//! `exact (abs=0)` against cell-eval. It was
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
//! One caveat, and it is sharper than the first draft of this comment said:
//! `np.argsort`'s default kind is `quicksort`, which is **not** a stable sort, so
//! the reference's own tie behaviour is implementation-defined. SCX matches the
//! stable reading. That agrees with the reference on a **total** tie — measured,
//! not assumed — and on a mixed tie may or may not: numpy 2.4.4 returns reverse
//! index order within each tied block for `[3,3,1,1]` but plain index order for
//! `[1,1,2]`, so mixed-tie parity is unpredictable rather than reliably wrong.
//! The stable reading is kept anyway,
//! because it is deterministic and reproducible across versions and languages
//! where the reference's is neither; see
//! [`mixed_ties_pin_scx_stable_semantics_not_cell_eval_parity`], which pins that
//! as SCX's contract rather than a parity claim.
//!
//! Distances are symmetric in all three metrics, so computing `d(pred, real)`
//! where the reference computes `d(real, pred)` is not a fourth divergence.

use super::*;

/// A **total** tie: rank must be the argsort *position*, not the count of
/// strictly-smaller distances.
///
/// Scoped to a total tie on purpose: it is the shape on which the stable and
/// unstable readings of `np.argsort` have been *measured* to agree, so it is the
/// only tie shape on which exact cell-eval parity is even a meaningful claim.
///
/// "Measured" rather than "guaranteed", deliberately. An unstable sort owes no
/// contract about equal keys, all-equal included, so this is an empirical
/// compatibility point that a future numpy could move. That is an argument for
/// pinning it against the installed reference — which the Python
/// `TestDiscriminationTieParity` does — not for claiming it must hold. The mixed-tie case, where they
/// diverge and SCX deliberately keeps the stable reading, is
/// [`mixed_ties_pin_scx_stable_semantics_not_cell_eval_parity`].
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
fn total_ties_rank_by_index_matching_cell_eval() {
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

/// A **mixed** tie — some distances equal, some not — pins SCX's own
/// deterministic semantics, which is where they stop matching cell-eval.
///
/// This is the case the total-tie fixture above cannot see. `np.argsort`'s default
/// `quicksort` preserves index order on a *fully* tied array, so a total tie
/// always agrees with the stable reading. A mixed tie *may* diverge — it is not
/// guaranteed either way, which is the whole problem: `[1, 1, 2]` happens to
/// agree on numpy 2.4.4 while `[3, 3, 1, 1]` does not, and nothing about the data
/// predicts which. Where it diverges the order is not even random-looking — it is
/// *reverse* index order within each tied block:
///
/// ```text
/// np.argsort([3, 3, 1, 1])                 -> [3, 2, 1, 0]
/// np.argsort([3, 3, 1, 1], kind="stable")  -> [2, 3, 0, 1]   <- mixed: differs
/// np.argsort([1, 1, 2])                    -> [0, 1, 2]      <- mixed: agrees
/// np.argsort([5, 5, 5])                    -> [0, 1, 2]      <- total: always agrees
/// ```
///
/// So for distances `[3, 3, 1, 1]` the two readings give different scores:
///
/// | pert | SCX (stable) | cell-eval on numpy 2.4.4 (quicksort) |
/// |------|--------------|--------------------------------------|
/// | p0   | 0.50         | 0.25                                 |
/// | p1   | 0.25         | 0.50                                 |
/// | p2   | 1.00         | 0.75                                 |
/// | p3   | 0.75         | 1.00                                 |
///
/// **SCX deliberately keeps the stable reading**, and this test is that contract
/// rather than a parity claim. Chasing the reference here would mean
/// reimplementing NumPy's introsort — pivot choices, array-size thresholds, dtype
/// dispatch — and would then break on any NumPy release that touched it. A
/// deterministic, documented tie rule is worth more than bit-parity with an
/// unspecified one, and it is reproducible across versions, platforms and
/// languages, which the reference's is not.
///
/// What *is* a bug, and is fixed alongside this test, is having claimed exact
/// cell-eval parity on ties in `docs/scanpy/accel-perturbation-metrics.md`.
/// Ties agree with the reference
/// only when the tie is total.
#[test]
fn mixed_ties_pin_scx_stable_semantics_not_cell_eval_parity() {
    let n_perts = 4usize;
    let n_genes = 2usize;
    // Every prediction is the zero vector, so each perturbation sees the same
    // L1 distance vector [3, 3, 1, 1] — two tied blocks, not one.
    let real = vec![
        3.0, 0.0, // p0
        -3.0, 0.0, // p1
        1.0, 0.0, // p2
        -1.0, 0.0, // p3
    ];
    let pred = vec![0.0; n_perts * n_genes];
    let perts: Vec<String> = (0..n_perts).map(|i| format!("p{i}")).collect();

    let got = compute_discrimination_score(
        &real,
        &pred,
        n_perts,
        n_genes,
        &perts,
        None,
        DistanceMetric::L1,
        false,
    )
    .unwrap();

    // Premise: the fixture must produce a MIXED tie. A total tie would make this
    // a duplicate of the test above and would agree with the reference, hiding
    // exactly the divergence being pinned.
    let d: Vec<f64> = (0..n_perts)
        .map(|i| (0..n_genes).map(|g| (real[i * n_genes + g]).abs()).sum())
        .collect();
    let distinct = {
        let mut v = d.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.dedup();
        v.len()
    };
    assert_eq!(
        distinct, 2,
        "premise broken: distances {d:?} have {distinct} distinct values, need \
         exactly 2 so there are two tied blocks rather than one"
    );

    // The stable-argsort ranks: position of p among distances sorted ascending,
    // ties by ascending index.
    let want = [0.5, 0.25, 1.0, 0.75];
    for (p, (&g, &w)) in got.scores.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-12,
            "pert {p}: got {g}, SCX's documented stable semantics give {w} \
             (all scores {:?}). Note cell-eval on numpy 2.4.4 gives \
             [0.25, 0.5, 0.75, 1.0] here — that divergence is documented, not a \
             target.",
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

/// Excluding **every** gene column must be an error, not a distance of zero.
///
/// The boundary the duplicate-name fixture cannot reach: it always leaves an
/// unrelated `g1` column standing. When every column is named after the
/// perturbation, the keep mask is all-false and the kernels answer anyway — L1
/// and L2 reduce an empty iterator to `0.0`, cosine returns `1.0` from its
/// zero-denominator branch — so every perturbation ties and scores a meaningless
/// `1.0`. Measured before the fix: `{g0: 1.0, g1: 1.0}`.
///
/// The reference does not answer: cell-eval hands its empty
/// `np.flatnonzero(genes != p)` to sklearn, which raises `Found array with 0
/// feature(s) (shape=(2, 0))`. Verified against the installed package.
///
/// Reachable rather than hypothetical — a single-gene panel whose gene is the
/// perturbation target, or a matrix whose `var_names` are all one symbol.
#[test]
fn excluding_every_column_is_an_error_not_a_perfect_score() {
    let n_perts = 2usize;
    let n_genes = 2usize;
    // BOTH columns are named "g0", and "g0" is a perturbation.
    let gene_names = vec!["g0".to_string(), "g0".to_string()];
    let perts = vec!["g0".to_string(), "g1".to_string()];
    let real = vec![7.0, 3.0, 1.0, 5.0];
    let pred = real.clone();

    let err = compute_discrimination_score(
        &real,
        &pred,
        n_perts,
        n_genes,
        &perts,
        Some(&gene_names),
        DistanceMetric::L1,
        true,
    )
    .expect_err(
        "excluding every column must be rejected — before the fix this returned \
         a perfect 1.0 for every perturbation",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("every gene column") && msg.contains("g0"),
        "error should name the condition and the perturbation, got: {msg}"
    );

    // The complement: one surviving column is fine, so the guard is keyed to
    // "nothing left" and not merely to "a duplicate exists".
    let ok_genes = vec!["g0".to_string(), "g0".to_string(), "g1".to_string()];
    let ok_real = vec![7.0, 3.0, 2.0, 1.0, 5.0, 4.0];
    compute_discrimination_score(
        &ok_real,
        &ok_real.clone(),
        2,
        3,
        &perts,
        Some(&ok_genes),
        DistanceMetric::L1,
        true,
    )
    .expect("one surviving column must still be accepted");
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
