//! Quasi-random shard and row shuffling for training data randomization.
//!
//! Implements [SPEC.md §8.3](../SPEC.md#83-quasi-random-shard-shuffle):
//! two-level shuffle that provides training randomization without random I/O.
//!
//! - **Level 1 (shard order)**: Permute shard indices each epoch, then group
//!   into contiguous shard groups for sequential disk I/O.
//! - **Level 2 (row shuffle)**: Fisher-Yates shuffle on pooled cell indices
//!   within each shard group.

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Shuffles shard indices into randomized shard groups each epoch.
///
/// Uses a deterministic RNG seeded from `(seed, epoch)` so that the same
/// seed and epoch always produce the identical shard ordering.
#[derive(Debug)]
pub struct ShardShuffler {
    n_shards: usize,
    shard_group_size: usize,
    rng_seed: u64,
    epoch: u64,
}

impl ShardShuffler {
    /// Create a new shard shuffler.
    ///
    /// # Arguments
    /// - `n_shards`: Total number of CSR shards in the file.
    /// - `shard_group_size`: Number of shards per I/O group (must be >= 1).
    /// - `seed`: RNG seed for reproducibility.
    pub fn new(n_shards: usize, shard_group_size: usize, seed: u64) -> crate::error::Result<Self> {
        if shard_group_size == 0 {
            return Err(crate::error::LoaderError::ConfigError {
                reason: "shard_group_size must be >= 1".to_string(),
            });
        }
        Ok(ShardShuffler {
            n_shards,
            shard_group_size,
            rng_seed: seed,
            epoch: 0,
        })
    }

    /// Generate shuffled shard groups for this epoch.
    ///
    /// **Level 1 (shard order)**: Create a seeded RNG from `(seed, epoch)`.
    /// Randomly permute shard indices `[0..n_shards)`. Group into groups of
    /// `shard_group_size`. The last group may be smaller.
    ///
    /// Increments `self.epoch` after generating groups.
    pub fn shuffle_epoch(&mut self) -> Vec<Vec<usize>> {
        // Derive a unique RNG from (seed, epoch) for reproducibility
        let combined_seed = self.rng_seed.wrapping_add(self.epoch.wrapping_mul(0x9E3779B97F4A7C15));
        let mut rng = ChaCha8Rng::seed_from_u64(combined_seed);

        // Permute shard indices
        let mut indices: Vec<usize> = (0..self.n_shards).collect();
        indices.shuffle(&mut rng);

        // Group into chunks of shard_group_size
        let groups: Vec<Vec<usize>> = indices
            .chunks(self.shard_group_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        self.epoch += 1;
        groups
    }

    /// Current epoch counter (incremented after each `shuffle_epoch` call).
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

/// Row-level shuffle utilities.
pub struct RowShuffler;

impl RowShuffler {
    /// **Level 2 (row shuffle)**: Fisher-Yates shuffle on an array of global
    /// cell indices. Used after pooling rows from a shard group.
    pub fn shuffle_rows(cell_indices: &mut [u64], rng: &mut impl rand::Rng) {
        cell_indices.shuffle(rng);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_different_epochs_different_orderings() {
        let mut shuffler = ShardShuffler::new(20, 4, 42).unwrap();
        let epoch0 = shuffler.shuffle_epoch();
        let epoch1 = shuffler.shuffle_epoch();

        // Flatten to compare orderings
        let flat0: Vec<usize> = epoch0.into_iter().flatten().collect();
        let flat1: Vec<usize> = epoch1.into_iter().flatten().collect();

        assert_ne!(flat0, flat1, "different epochs should produce different orderings");
    }

    #[test]
    fn test_same_seed_same_epoch_identical() {
        let mut shuffler1 = ShardShuffler::new(20, 4, 42).unwrap();
        let mut shuffler2 = ShardShuffler::new(20, 4, 42).unwrap();

        let groups1 = shuffler1.shuffle_epoch();
        let groups2 = shuffler2.shuffle_epoch();

        assert_eq!(groups1, groups2, "same seed + same epoch must be identical");
    }

    #[test]
    fn test_all_shard_indices_present() {
        let n_shards = 17;
        let mut shuffler = ShardShuffler::new(n_shards, 5, 123).unwrap();

        for _ in 0..3 {
            let groups = shuffler.shuffle_epoch();
            let flat: Vec<usize> = groups.into_iter().flatten().collect();

            assert_eq!(flat.len(), n_shards, "all shards should appear");
            let unique: HashSet<usize> = flat.iter().copied().collect();
            assert_eq!(unique.len(), n_shards, "no duplicates");
            for i in 0..n_shards {
                assert!(unique.contains(&i), "shard index {i} missing");
            }
        }
    }

    #[test]
    fn test_group_sizes_correct() {
        // 17 shards, group_size=5 → 4 groups: [5, 5, 5, 2]
        let mut shuffler = ShardShuffler::new(17, 5, 99).unwrap();
        let groups = shuffler.shuffle_epoch();

        assert_eq!(groups.len(), 4);
        assert_eq!(groups[0].len(), 5);
        assert_eq!(groups[1].len(), 5);
        assert_eq!(groups[2].len(), 5);
        assert_eq!(groups[3].len(), 2); // last group is smaller

        // Exact divisible case: 20 shards, group_size=5 → 4 groups all size 5
        let mut shuffler2 = ShardShuffler::new(20, 5, 99).unwrap();
        let groups2 = shuffler2.shuffle_epoch();
        assert_eq!(groups2.len(), 4);
        for g in &groups2 {
            assert_eq!(g.len(), 5);
        }
    }

    #[test]
    fn test_single_shard() {
        let mut shuffler = ShardShuffler::new(1, 8, 42).unwrap();
        let groups = shuffler.shuffle_epoch();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0], vec![0]);
    }

    #[test]
    fn test_group_size_larger_than_n_shards() {
        // group_size > n_shards → single group
        let mut shuffler = ShardShuffler::new(3, 10, 42).unwrap();
        let groups = shuffler.shuffle_epoch();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3);
    }

    #[test]
    fn test_epoch_counter_increments() {
        let mut shuffler = ShardShuffler::new(10, 4, 42).unwrap();
        assert_eq!(shuffler.epoch(), 0);
        shuffler.shuffle_epoch();
        assert_eq!(shuffler.epoch(), 1);
        shuffler.shuffle_epoch();
        assert_eq!(shuffler.epoch(), 2);
    }

    #[test]
    fn test_row_shuffle_is_permutation() {
        let mut indices: Vec<u64> = (0..100).collect();
        let original: Vec<u64> = indices.clone();

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        RowShuffler::shuffle_rows(&mut indices, &mut rng);

        // Should be a permutation: same elements, different order
        assert_ne!(indices, original, "shuffle should change order");
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, original, "all indices must be present after shuffle");
    }

    #[test]
    fn test_row_shuffle_reproducible() {
        let mut indices1: Vec<u64> = (0..50).collect();
        let mut indices2: Vec<u64> = (0..50).collect();

        let mut rng1 = ChaCha8Rng::seed_from_u64(99);
        let mut rng2 = ChaCha8Rng::seed_from_u64(99);

        RowShuffler::shuffle_rows(&mut indices1, &mut rng1);
        RowShuffler::shuffle_rows(&mut indices2, &mut rng2);

        assert_eq!(indices1, indices2, "same seed must produce same shuffle");
    }

    #[test]
    fn test_row_shuffle_empty() {
        let mut indices: Vec<u64> = vec![];
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        RowShuffler::shuffle_rows(&mut indices, &mut rng);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_zero_shard_group_size_returns_error() {
        let result = ShardShuffler::new(10, 0, 42);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("shard_group_size"), "expected 'shard_group_size' in: {msg}");
    }
}
