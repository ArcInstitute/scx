//! `estimate_alpha` against a generating `α` (§7.14, ORG-7.21-4).
//!
//! # What was missing
//!
//! Every previous test of this estimator compared it to a re-implementation of
//! its own definition — including the `var > mean` pre-filter that was the bug.
//! Nothing simulated counts from a known dispersion and asked whether the
//! estimator found it. These fixtures do; their provenance and the reason the
//! bars are shaped the way they are live in
//! [`super::pflog_reference_values`].
//!
//! # Three arms, and only one of them is a tolerance
//!
//! * **A** — the pool must not be truncated, stated as two exact facts.
//! * **B** — the accuracy claim, and the accept side: the fix must not be
//!   "always return something smaller".
//! * **C** — the mechanism, with no tolerance at all: on a matrix that is mostly
//!   under-dispersed, the old estimator reported four genes' dispersion as the
//!   whole matrix's and called it success.

use super::pflog_reference_values as r;
use super::{estimate_alpha, AlphaOptions};
use scx_format_io::ShardSource;

/// Fixture A must actually contain a truncation for its arms to mean anything.
///
/// A compile-time assertion rather than a `#[test]`: both sides are pinned
/// constants, so there is nothing to run — clippy says so too
/// (`assertions_on_constants`). If a regeneration ever emits a fixture the old
/// filter would not have truncated, this stops the build instead of failing a
/// test that reads as a numerics regression.
const _: () = assert!(r::A_TRUNCATED_SURVIVORS < r::A_N_GENES_POOLED);

/// The generator emits the `mu_min` it filtered with; the estimator compiles
/// against its own default. If they diverge, every pinned count below describes
/// a different pool than the one the tests assert over — and nothing else here
/// would fail.
#[test]
fn the_fixtures_were_generated_with_the_mu_min_the_estimator_uses() {
    assert_eq!(
        AlphaOptions::default().mu_min,
        r::MU_MIN,
        "the fixtures were generated against mu_min={}, the estimator defaults to {} \
         — regenerate rather than adjusting the constant here",
        r::MU_MIN,
        AlphaOptions::default().mu_min
    );
}

/// Every fixture's shape must match the dimensions pinned beside it.
///
/// The count matrices are literals, so an edit that adds or removes a row would
/// otherwise change what every arm below measures without any of them saying so.
#[test]
fn every_fixture_has_the_shape_pinned_beside_it() {
    for (name, src, n_cells, n_genes) in [
        ("A", r::source_a(), r::A_N_CELLS, r::A_N_GENES),
        ("B", r::source_b(), r::B_N_CELLS, r::B_N_GENES),
        ("C", r::source_c(), r::C_N_CELLS, r::C_N_GENES),
    ] {
        assert_eq!(src.n_obs(), n_cells, "fixture {name}: cell count");
        assert_eq!(src.n_vars(), n_genes, "fixture {name}: gene count");
        assert!(
            src.n_shards() > 1,
            "fixture {name}: single shard, so the \
             per-shard moment merge in streaming_mean_var goes untested"
        );
    }
}

// --- fixture A: the pool holds every gene with a mean ------------------------

/// The exact arm. `n_genes_used` is the count of genes that entered the median,
/// and on this fixture that must be **every** gene with a mean above `mu_min`.
/// The pre-fix estimator kept [`r::A_TRUNCATED_SURVIVORS`] of them.
///
/// This assertion needs no tolerance and cannot pass by sampling luck, which is
/// what makes it the primary guard against the truncation returning.
#[test]
fn the_pool_holds_every_gene_with_a_mean() {
    let est = estimate_alpha(&r::source_a(), &AlphaOptions::default()).unwrap();
    assert!(
        !est.fell_back,
        "fixture A is meant to yield an estimate, not the fallback"
    );
    assert_eq!(
        est.n_genes_used,
        r::A_N_GENES_POOLED,
        "the pool dropped {} gene(s); the pre-fix filter kept only {} of {} and \
         that is the bias this fixture exists to catch",
        r::A_N_GENES_POOLED - est.n_genes_used,
        r::A_TRUNCATED_SURVIVORS,
        r::A_N_GENES_POOLED
    );
}

/// The second exact arm: the estimate must be at least [`r::A_SEPARATION`]×
/// closer to the generating `α` than the pre-fix answer on the same matrix.
///
/// Deterministic for this fixture — both quantities are properties of one pinned
/// count matrix — so unlike the bar below it says something the sampling spread
/// cannot wash out.
#[test]
fn the_estimate_is_closer_to_the_generating_alpha_than_the_truncated_one() {
    let est = estimate_alpha(&r::source_a(), &AlphaOptions::default()).unwrap();
    let ours = (est.alpha - r::A_ALPHA_TRUE).abs();
    let theirs = (r::A_TRUNCATED_ALPHA - r::A_ALPHA_TRUE).abs();
    assert!(
        ours * r::A_SEPARATION < theirs,
        "α={} is {:.2}x closer to α_true={} than the truncated {} — under the \
         {}x this fixture is pinned for",
        est.alpha,
        theirs / ours,
        r::A_ALPHA_TRUE,
        r::A_TRUNCATED_ALPHA,
        r::A_SEPARATION
    );
}

/// The gross-breakage bound. Deliberately loose: see the next test for why it
/// is not the thing that catches the bug.
#[test]
fn the_estimate_is_within_its_sampling_bar_of_the_generating_alpha() {
    let est = estimate_alpha(&r::source_a(), &AlphaOptions::default()).unwrap();
    let rel = (est.alpha - r::A_ALPHA_TRUE).abs() / r::A_ALPHA_TRUE;
    assert!(
        rel <= r::A_REL_BAR,
        "α={} is {rel:.4} off α_true={} (bar {})",
        est.alpha,
        r::A_ALPHA_TRUE,
        r::A_REL_BAR
    );
}

/// The premise for the two exact arms: **this bar accepts the truncated
/// answer.**
///
/// It is asserted rather than left as a comment because the natural
/// simplification of this file is "the tolerance test covers it, drop the exact
/// ones" — and that is false. At 50 cells the estimator's own 99th-percentile
/// spread is wider than the bias being detected, so no tolerance on a fixture
/// of this size can tell the two estimators apart. If a future fixture *does*
/// separate them by tolerance, this test fails and should be deleted along with
/// the note in [`super::pflog_reference_values`] that explains it.
#[test]
fn the_sampling_bar_cannot_separate_the_two_estimators() {
    let rel_truncated = (r::A_TRUNCATED_ALPHA - r::A_ALPHA_TRUE).abs() / r::A_ALPHA_TRUE;
    assert!(
        rel_truncated <= r::A_REL_BAR,
        "the bar {} now REJECTS the truncated answer (rel {rel_truncated:.4}), so a \
         tolerance test would catch the bug on its own — re-read the exact arms \
         above and the measurement note in pflog_reference_values.rs",
        r::A_REL_BAR
    );
}

// --- fixture B: the accuracy claim, and the accept side ----------------------

/// Strong overdispersion, where 50-odd cells do determine `α`: the estimate lands
/// within [`r::B_REL_BAR`] of the generating value.
///
/// This is also the accept side. Removing a filter that discarded genes could
/// have been "make every answer smaller"; here **no** gene was discarded even
/// before the fix, so the estimate must be unchanged and still right. A fix that
/// biased `α` low would pass fixture A's arms and fail this one.
#[test]
fn strong_overdispersion_recovers_the_generating_alpha() {
    let est = estimate_alpha(&r::source_b(), &AlphaOptions::default()).unwrap();
    assert!(!est.fell_back);
    assert_eq!(est.n_genes_used, r::B_N_GENES_POOLED);
    let rel = (est.alpha - r::B_ALPHA_TRUE).abs() / r::B_ALPHA_TRUE;
    assert!(
        rel <= r::B_REL_BAR,
        "α={} is {rel:.4} off α_true={} (bar {})",
        est.alpha,
        r::B_ALPHA_TRUE,
        r::B_REL_BAR
    );
    assert!(
        est.pseudocount > 0.0 && est.pseudocount.is_finite(),
        "pseudocount {} is not usable",
        est.pseudocount
    );
}

// --- fixture C: the mechanism, with no tolerance -----------------------------

/// [`r::C_N_UNDERDISPERSED`] of [`r::C_N_GENES_POOLED`] genes are binomial, so
/// `Var < mean` for each of them by construction rather than by luck; the rest
/// are genuinely over-dispersed. The pre-fix estimator kept exactly the
/// over-dispersed ones and reported their median, [`r::C_TRUNCATED_ALPHA`], as
/// the dispersion of the whole matrix — with `fell_back = false`.
///
/// The fixed estimator pools all of them, finds the median non-positive, and
/// takes the documented fallback. No tolerance is involved in that verdict.
#[test]
fn an_underdispersed_matrix_falls_back_instead_of_reporting_a_handful_of_genes() {
    let opts = AlphaOptions::default();
    let est = estimate_alpha(&r::source_c(), &opts).unwrap();
    assert!(
        est.fell_back,
        "α={} was returned as a confident estimate; on a matrix where {} of {} \
         genes are under-dispersed the honest answer is the fallback",
        est.alpha,
        r::C_N_UNDERDISPERSED,
        r::C_N_GENES_POOLED
    );
    assert_eq!(
        est.alpha, opts.fallback_alpha,
        "the fallback must report `fallback_alpha`, not a clamped estimate"
    );
    assert_eq!(
        est.n_genes_used,
        r::C_N_GENES_POOLED,
        "the fallback fired with {} gene(s) pooled; it must fire on the pooled \
         median over all {} of them, not on an empty pool — that distinction is \
         what `uns[\"pflog\"]` reports",
        est.n_genes_used,
        r::C_N_GENES_POOLED
    );
    assert!(
        (r::C_TRUNCATED_ALPHA - est.alpha).abs() > 0.1,
        "the pre-fix answer {} is no longer distinguishable from the fallback {}",
        r::C_TRUNCATED_ALPHA,
        est.alpha
    );
    // And why the pre-fix answer was dangerous rather than merely wrong: it is
    // very nearly the over-dispersed block's *true* dispersion, so it looked like
    // a real measurement of the matrix while describing four genes of it.
    let rel = (r::C_TRUNCATED_ALPHA - r::C_ALPHA_OVER).abs() / r::C_ALPHA_OVER;
    assert!(
        rel < 0.05,
        "the truncated answer {} is no longer close to the over-dispersed block's \
         α_true {} (relative {rel:.3e}); the fixture no longer shows a plausible \
         wrong answer",
        r::C_TRUNCATED_ALPHA,
        r::C_ALPHA_OVER
    );
}

/// Fixture C's shape, stated over the pinned constants rather than recomputed:
/// the genes the old filter kept and the genes it discarded must account for the
/// whole pool. Recomputing the moments here would put a fourth copy of the
/// estimator in the file that deletes the other two.
#[test]
fn fixture_c_is_all_underdispersed_genes_plus_the_survivors() {
    assert_eq!(
        r::C_TRUNCATED_SURVIVORS + r::C_N_UNDERDISPERSED,
        r::C_N_GENES_POOLED,
        "{} survivors + {} under-dispersed != {} pooled; the fixture is not the \
         clean split its arms assume",
        r::C_TRUNCATED_SURVIVORS,
        r::C_N_UNDERDISPERSED,
        r::C_N_GENES_POOLED
    );
}
