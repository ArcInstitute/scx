//! Seeded global row permutation for `scx sort --shuffle` (data-load Phase 1D).
//!
//! `scx sort` is already a complete external-memory obs-axis reorder, and it
//! reduces on `new_pos`: pass 0 computes an order array and *every* downstream
//! consumer reads only that array, never the sort keys. This module is the third
//! pass-0 producer — a seeded random permutation instead of `stable_argsort` —
//! so the whole emission, alignment and bounded-memory machinery is reused
//! unchanged.
//!
//! # Why a global pre-shuffle exists
//!
//! `TrainingDataset`'s randomization is two-level and both levels are bounded by
//! *physical layout*: level 1 permutes shard order, level 2 Fisher–Yates
//! shuffles within a shard group (`scx-loader/src/shuffle.rs`). On a file whose
//! rows arrived clustered — by donor, plate, or cell type — batch composition is
//! capped by `shard_group_size`, and widening it costs memory linearly.
//! Permuting the rows *once, on disk* moves that cost off the training loop.
//!
//! # Keying: mixed, not added
//!
//! The ChaCha8 seed is `splitmix64(splitmix64(user_seed) ^ SHUFFLE_DOMAIN_TAG)`.
//!
//! Two deliberate choices:
//!
//! - **Mix, don't add.** Every pre-existing `scx-loader` seed site combines
//!   components *additively* (`shuffle.rs`, `decode_stage.rs`), which collides:
//!   `(seed + 1, epoch)` aliases `(seed, epoch + 1)`. `scx-loader`'s
//!   `downsample.rs` established the chained-SplitMix64 convention for anything
//!   with more than one component; this mirrors it. (It cannot *import* it —
//!   that lives in `scx-loader`, which `scx-ops` does not depend on.)
//! - **Domain-separate.** Without the tag, `shuffle(seed=42)` and the training
//!   loader's `(seed=42, epoch=0)` shard permutation would be driven by the
//!   same ChaCha8 stream. `42` is the default on both surfaces, so that pairing
//!   is the *likely* one, not a corner case. The correlation would be a
//!   statistical wart rather than a bug, but the tag costs one xor.
//!
//! SplitMix64 is a **mixer, not a KDF** — it is here to decorrelate nearby
//! seeds, not to resist an adversary. Nothing about this permutation is secret.

/// Domain separator for the shuffle seed, so a shuffled file's row permutation
/// is uncorrelated with the training loader's shard permutation at the same
/// numeric seed. Arbitrary but fixed: changing it changes every shuffled
/// file's layout at a given seed, which would silently break reproducibility of
/// already-materialised corpora. Pinned by
/// `fixed_seed_yields_the_pinned_permutation`.
const SHUFFLE_DOMAIN_TAG: u64 = 0x5343_5853_4855_4646; // b"SCXSHUFF"

/// SplitMix64 finalizer. Mixer, not a KDF — see the module docs.
#[inline]
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Derive the RNG seed for a shuffle from the user-supplied seed.
#[inline]
fn permutation_seed(seed: u64) -> u64 {
    splitmix64(splitmix64(seed) ^ SHUFFLE_DOMAIN_TAG)
}

/// Seeded uniform permutation of `0..n`, as output-row → input-row.
///
/// The result is the shuffle-mode analogue of `stable_argsort`'s return value:
/// `perm[new_position] = old_local_index`. `n` is the **live** (deletion
/// filtered) row count, which is why the same seed produces a different
/// permutation on a file with deletions than on one without — the permutation
/// is over the rows that survive, and there is no stable identity to key on
/// that would make it otherwise.
///
/// Memory is `n * 8` bytes, identical to the `Vec<u64>` `stable_argsort`
/// already allocates on this path, so shuffle mode does not change the
/// engine's memory profile.
pub fn seeded_permutation(n: usize, seed: u64) -> Vec<u64> {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(permutation_seed(seed));
    let mut perm: Vec<u64> = (0..n as u64).collect();
    perm.shuffle(&mut rng);
    perm
}

#[cfg(test)]
#[path = "shuffle_order_tests.rs"]
mod tests;
