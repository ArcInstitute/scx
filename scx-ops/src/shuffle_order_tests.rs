//! Unit tests for the seeded permutation producer.

use super::*;
use std::collections::HashSet;

#[test]
fn permutation_is_a_permutation() {
    for n in [0usize, 1, 2, 7, 1000] {
        let perm = seeded_permutation(n, 42);
        assert_eq!(perm.len(), n, "n={n}");
        let set: HashSet<u64> = perm.iter().copied().collect();
        assert_eq!(set.len(), n, "n={n}: values must be distinct");
        assert!(
            perm.iter().all(|&v| (v as usize) < n),
            "n={n}: values must be in range"
        );
    }
}

#[test]
fn same_seed_is_deterministic() {
    let a = seeded_permutation(500, 1234);
    let b = seeded_permutation(500, 1234);
    assert_eq!(a, b);
}

#[test]
fn different_seeds_differ() {
    // The anti-tautology half of the determinism test: without this, replacing
    // the RNG with a constant would leave `same_seed_is_deterministic` green.
    let a = seeded_permutation(500, 1234);
    let b = seeded_permutation(500, 1235);
    assert_ne!(a, b);
}

#[test]
fn permutation_is_not_the_identity() {
    // 1/1000! chance of a false failure; the real thing this catches is a
    // shuffle that silently became a no-op.
    let perm = seeded_permutation(1000, 42);
    let identity: Vec<u64> = (0..1000).collect();
    assert_ne!(perm, identity);
}

#[test]
fn adjacent_seeds_do_not_alias() {
    // The collision the crate's pre-existing *additive* `(seed, epoch)`
    // convention produces: with `seed + epoch`, `(s+1, e)` and `(s, e+1)` are
    // the same stream. Shuffle has one component today, but the mixer is the
    // convention any future component must be added under — so pin that
    // neighbouring seeds are decorrelated rather than merely different.
    let base = seeded_permutation(2000, 100);
    for delta in 1..=8u64 {
        let other = seeded_permutation(2000, 100 + delta);
        let agree = base.iter().zip(&other).filter(|(a, b)| a == b).count();
        // Two independent uniform permutations of 2000 elements agree in ~1
        // position on average; 20 is far beyond noise but far below the
        // thousands that a correlated stream would produce.
        assert!(
            agree < 20,
            "seed 100 vs {}: {agree} positions agree — streams look correlated",
            100 + delta
        );
    }
}

#[test]
fn fixed_seed_yields_the_pinned_permutation() {
    // A literal pin, not a property. Two things it catches that nothing else
    // does: a `rand` / `rand_chacha` bump silently changing `SliceRandom::
    // shuffle` or `seed_from_u64`, and a change to `SHUFFLE_DOMAIN_TAG`. Both
    // would relayout every shuffled file at a given seed while every
    // property-style test above stayed green.
    //
    // Regenerate ONLY with a deliberate decision to break reproducibility of
    // already-materialised corpora, and say so in the commit message.
    assert_eq!(
        seeded_permutation(16, 42),
        vec![0, 4, 2, 12, 3, 13, 11, 5, 15, 8, 6, 7, 14, 9, 10, 1]
    );
}

#[test]
fn domain_tag_separates_from_a_bare_seed() {
    // The tag exists so `shuffle(seed=42)` is uncorrelated with a bare
    // `ChaCha8Rng::seed_from_u64(42)` permutation — and 42 is the default on
    // both this surface and the training loader's, so that pairing is the
    // likely one. The bare derivation below *was* the loader's at epoch 0; the
    // loader has since moved to a tagged, chained seed of its own
    // (`scx-loader/src/seed.rs`), so this now pins the weaker, and sufficient,
    // property: this module's output is not a plain seed-42 shuffle.
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let mut bare = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let mut loader_style: Vec<u64> = (0..64).collect();
    loader_style.shuffle(&mut bare);

    assert_ne!(seeded_permutation(64, 42), loader_style);
}
