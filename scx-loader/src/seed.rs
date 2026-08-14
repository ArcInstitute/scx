//! One way to compose a seed out of several components.
//!
//! # Why chaining and not addition
//!
//! The obvious composition — `seed + component * PHI` — collides, because
//! addition is commutative and associative over the whole key space: `(seed, e)`
//! and `(seed + PHI, e - 1)` land on **the same** stream. A seed sweep over
//! `s, s + PHI, s + 2·PHI` at fixed `e` therefore silently repeats orderings
//! from neighbouring epochs, and an additive domain tag has the same problem
//! one level up: with `seed + TAG` as the tag, two differently-tagged streams
//! coincide whenever their seeds differ by `TAG`.
//!
//! Chaining each component through a finalizer removes both: no additive
//! relationship between inputs survives the mixer, so distinct key tuples land
//! on independent-looking streams.
//!
//! `downsample.rs` introduced this convention for its four-component row key
//! and its doc comment named the two additive sites as the defect; this module
//! is that convention made shared, with the two sites converted. `scx-ops`'
//! `shuffle_order.rs` writes the same mixer out a third time because it cannot
//! depend on this crate — deliberate duplication, noted there.
//!
//! **This is a mixer, not a KDF.** The goal is that distinct key tuples land on
//! independent-looking streams; there is no secrecy or preimage-resistance
//! claim, and nothing here should be reused where one is needed.

/// Domain tag for the Level-1 shard-order shuffle (`shuffle.rs`).
pub(crate) const SHARD_SHUFFLE_TAG: u64 = 0x5343_5853_4852_4431; // b"SCXSHRD1"

/// Domain tag for the Level-2 within-group row shuffle (`decode_stage.rs`).
pub(crate) const ROW_SHUFFLE_TAG: u64 = 0x5343_5852_4F57_5F32; // b"SCXROW_2"

/// SplitMix64 finalizer. Used to *chain* key components rather than add them.
#[inline]
pub(crate) fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Derive a per-epoch stream seed from `(seed, domain tag, epoch)`.
///
/// The tag keeps the two shuffle levels on independent streams at the same
/// `(seed, epoch)`; chaining keeps `(seed, epoch)` pairs from aliasing each
/// other.
#[inline]
pub(crate) fn epoch_stream_seed(seed: u64, domain_tag: u64, epoch: u64) -> u64 {
    let h = splitmix64(seed);
    splitmix64(splitmix64(h ^ domain_tag) ^ epoch)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PHI, the multiplier the two additive sites used. `(s, e)` and
    /// `(s + PHI, e - 1)` produced *identical* streams under that scheme; this
    /// is the collision the module exists to remove.
    const PHI: u64 = 0x9E37_79B9_7F4A_7C15;

    /// Anti-tautology: demonstrate the collision in the old derivation, so the
    /// assertion below is known to be testing something.
    #[test]
    fn the_old_additive_scheme_really_did_collide() {
        let old = |seed: u64, epoch: u64| seed.wrapping_add(epoch.wrapping_mul(PHI));
        assert_eq!(old(42, 7), old(42u64.wrapping_add(PHI), 6));
    }

    #[test]
    fn chaining_separates_seed_from_epoch() {
        let a = epoch_stream_seed(42, SHARD_SHUFFLE_TAG, 7);
        let b = epoch_stream_seed(42u64.wrapping_add(PHI), SHARD_SHUFFLE_TAG, 6);
        assert_ne!(a, b, "(s, e) must not alias (s + PHI, e - 1)");
    }

    /// The old row-shuffle tag was `seed + 0xDEADBEEF`, so the row stream at
    /// `s` was byte-identical to the shard stream at `s + 0xDEADBEEF`.
    #[test]
    fn the_old_additive_domain_tag_really_did_collide() {
        const DEADBEEF: u64 = 0xDEADBEEF;
        let old_shard = |seed: u64, epoch: u64| seed.wrapping_add(epoch.wrapping_mul(PHI));
        let old_row = |seed: u64, epoch: u64| {
            seed.wrapping_add(DEADBEEF)
                .wrapping_add(epoch.wrapping_mul(PHI))
        };
        assert_eq!(old_row(42, 3), old_shard(42u64.wrapping_add(DEADBEEF), 3));
    }

    #[test]
    fn the_two_shuffle_levels_are_domain_separated() {
        const DEADBEEF: u64 = 0xDEADBEEF;
        assert_ne!(
            epoch_stream_seed(42, ROW_SHUFFLE_TAG, 3),
            epoch_stream_seed(42, SHARD_SHUFFLE_TAG, 3),
            "the two levels must not share a stream at the same (seed, epoch)"
        );
        assert_ne!(
            epoch_stream_seed(42, ROW_SHUFFLE_TAG, 3),
            epoch_stream_seed(42u64.wrapping_add(DEADBEEF), SHARD_SHUFFLE_TAG, 3),
            "nor at the offset the old additive tag made them coincide at"
        );
    }

    #[test]
    fn distinct_epochs_give_distinct_streams() {
        let seeds: Vec<u64> = (0..8)
            .map(|e| epoch_stream_seed(42, SHARD_SHUFFLE_TAG, e))
            .collect();
        let mut uniq = seeds.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), seeds.len(), "epochs collided: {seeds:?}");
    }

    #[test]
    fn the_derivation_is_deterministic() {
        assert_eq!(
            epoch_stream_seed(7, SHARD_SHUFFLE_TAG, 3),
            epoch_stream_seed(7, SHARD_SHUFFLE_TAG, 3)
        );
    }
}
