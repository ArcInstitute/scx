use crate::error::{LoaderError, Result};

/// Configuration for the training data loader pipeline.
///
/// See [SPEC.md §8.8](../SPEC.md#88-throughput-projections) for the memory
/// budget model and [Phase2.md §5.2](../Phase2.md#step-5-scx-loader) for
/// the full config description.
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Mini-batch size (default: 1024).
    pub batch_size: usize,
    /// Number of shards read per I/O group (default: 8).
    /// Sequential I/O within each group for disk efficiency.
    pub shard_group_size: usize,
    /// Ring buffer depth — number of pre-built batches to buffer (default: 4).
    pub prefetch_batches: usize,
    /// Gene indices for HVG projection. None = use all genes.
    pub hvg_indices: Option<Vec<u32>>,
    /// Obs metadata column names to include in each batch.
    pub obs_columns: Vec<String>,
    /// Apply total-count normalization (default: true).
    pub normalize: bool,
    /// Apply log1p transformation (default: true).
    pub log1p: bool,
    /// Normalization target sum (default: 1e4).
    pub target_sum: f64,
    /// RNG seed for reproducibility.
    pub seed: u64,
    /// Memory budget in MB (default: 512).
    /// Pipeline auto-tunes shard_group_size and prefetch_batches to fit.
    pub max_memory_mb: usize,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        LoaderConfig {
            batch_size: 1024,
            shard_group_size: 8,
            prefetch_batches: 4,
            hvg_indices: None,
            obs_columns: Vec::new(),
            normalize: true,
            log1p: true,
            target_sum: 1e4,
            seed: 42,
            max_memory_mb: 512,
        }
    }
}

impl LoaderConfig {
    /// Validate the configuration, returning `ConfigError` for invalid settings.
    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(LoaderError::ConfigError {
                reason: "batch_size must be > 0".to_string(),
            });
        }
        if self.shard_group_size == 0 {
            return Err(LoaderError::ConfigError {
                reason: "shard_group_size must be > 0".to_string(),
            });
        }
        if self.prefetch_batches == 0 {
            return Err(LoaderError::ConfigError {
                reason: "prefetch_batches must be > 0".to_string(),
            });
        }
        if self.target_sum <= 0.0 {
            return Err(LoaderError::ConfigError {
                reason: "target_sum must be > 0.0".to_string(),
            });
        }
        if self.max_memory_mb < 64 {
            return Err(LoaderError::ConfigError {
                reason: "max_memory_mb must be >= 64 (minimum viable budget)".to_string(),
            });
        }
        Ok(())
    }
}

/// Result of memory budget computation. Contains the effective parameters
/// after auto-tuning to fit within `max_memory_mb`.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Effective shard_group_size (may be reduced to fit budget).
    pub shard_group_size: usize,
    /// Effective prefetch_batches (may be reduced to fit budget).
    pub prefetch_batches: usize,
    /// Estimated total memory in bytes.
    pub estimated_bytes: usize,
}

/// Compute the memory budget for the training pipeline.
///
/// Implements the memory model from [SPEC.md §8.8]:
/// ```text
/// n_output_genes     = hvg_indices.len() if present, else n_vars
/// shard_group_buffer = shard_group_size × ~30 MB (compressed shard estimate)
/// decoded_row_buffer = shard_group_size × shard_target_rows × avg_nnz × 6 bytes
/// pinned_batch_ring  = prefetch_batches × batch_size × n_output_genes × 4 bytes
/// overhead           = ~10 MB (indexes, metadata, obs RecordBatch)
/// ```
///
/// If total exceeds `max_memory_mb`, reduces `prefetch_batches` first (to
/// minimum 2), then `shard_group_size` (to minimum 1).
pub fn compute_memory_budget(
    config: &LoaderConfig,
    n_vars: u64,
    shard_target_rows: u32,
    avg_nnz_per_cell: f64,
) -> MemoryBudget {
    let n_output_genes = match &config.hvg_indices {
        Some(hvg) => hvg.len(),
        None => n_vars as usize,
    };

    let max_bytes = config.max_memory_mb * 1024 * 1024;

    let mut shard_group_size = config.shard_group_size;
    let mut prefetch_batches = config.prefetch_batches;

    loop {
        let estimated = estimate_memory(
            shard_group_size,
            prefetch_batches,
            config.batch_size,
            n_output_genes,
            shard_target_rows as usize,
            avg_nnz_per_cell,
        );

        if estimated <= max_bytes {
            return MemoryBudget {
                shard_group_size,
                prefetch_batches,
                estimated_bytes: estimated,
            };
        }

        // Reduce prefetch_batches first (to minimum 2)
        if prefetch_batches > 2 {
            prefetch_batches -= 1;
            continue;
        }

        // Then reduce shard_group_size (to minimum 1)
        if shard_group_size > 1 {
            shard_group_size -= 1;
            continue;
        }

        // Both at minimums — return best-effort estimate
        return MemoryBudget {
            shard_group_size,
            prefetch_batches,
            estimated_bytes: estimated,
        };
    }
}

/// Estimate total memory usage for given parameters.
fn estimate_memory(
    shard_group_size: usize,
    prefetch_batches: usize,
    batch_size: usize,
    n_output_genes: usize,
    shard_target_rows: usize,
    avg_nnz_per_cell: f64,
) -> usize {
    const COMPRESSED_SHARD_ESTIMATE: usize = 30 * 1024 * 1024; // ~30 MB
    const OVERHEAD: usize = 10 * 1024 * 1024; // ~10 MB
    const BYTES_PER_NNZ: usize = 6; // indptr contrib + index + value

    let shard_group_buffer = shard_group_size * COMPRESSED_SHARD_ESTIMATE;
    let decoded_row_buffer =
        shard_group_size * shard_target_rows * (avg_nnz_per_cell as usize) * BYTES_PER_NNZ;
    let pinned_batch_ring = prefetch_batches * batch_size * n_output_genes * 4; // f32

    shard_group_buffer + decoded_row_buffer + pinned_batch_ring + OVERHEAD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validates() {
        let config = LoaderConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_zero_batch_size_fails() {
        let config = LoaderConfig {
            batch_size: 0,
            ..LoaderConfig::default()
        };
        let err = config.validate().unwrap_err();
        match err {
            LoaderError::ConfigError { reason } => {
                assert!(reason.contains("batch_size"), "unexpected reason: {reason}");
            }
            _ => panic!("expected ConfigError, got: {err:?}"),
        }
    }

    #[test]
    fn test_zero_shard_group_size_fails() {
        let config = LoaderConfig {
            shard_group_size: 0,
            ..LoaderConfig::default()
        };
        let err = config.validate().unwrap_err();
        match err {
            LoaderError::ConfigError { reason } => {
                assert!(
                    reason.contains("shard_group_size"),
                    "unexpected reason: {reason}"
                );
            }
            _ => panic!("expected ConfigError, got: {err:?}"),
        }
    }

    #[test]
    fn test_zero_prefetch_batches_fails() {
        let config = LoaderConfig {
            prefetch_batches: 0,
            ..LoaderConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_zero_target_sum_fails() {
        let config = LoaderConfig {
            target_sum: 0.0,
            ..LoaderConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_negative_target_sum_fails() {
        let config = LoaderConfig {
            target_sum: -1.0,
            ..LoaderConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_small_memory_budget_fails() {
        let config = LoaderConfig {
            max_memory_mb: 32,
            ..LoaderConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_minimum_memory_budget_passes() {
        let config = LoaderConfig {
            max_memory_mb: 64,
            ..LoaderConfig::default()
        };
        assert!(config.validate().is_ok());
    }

    // --- Memory budget tests ---

    #[test]
    fn test_memory_budget_2k_hvg_within_512mb() {
        // Default config with 2K HVG genes: should be ~330 MB (within 512 MB budget)
        let config = LoaderConfig {
            hvg_indices: Some((0..2000).collect()),
            ..LoaderConfig::default()
        };
        let budget = compute_memory_budget(
            &config, 30_000, // n_vars (full gene count)
            16_384, // shard_target_rows
            10.0,   // avg_nnz_per_cell
        );
        let budget_mb = budget.estimated_bytes / (1024 * 1024);
        assert!(
            budget_mb <= 512,
            "2K HVG budget {budget_mb} MB should be <= 512 MB"
        );
        // With HVG, should not need to reduce parameters
        assert_eq!(budget.shard_group_size, 8);
        assert_eq!(budget.prefetch_batches, 4);
    }

    #[test]
    fn test_memory_budget_30k_genes_auto_tuned() {
        // Default config with all 30K genes (no HVG): larger, may need tuning
        let config = LoaderConfig::default();
        let budget = compute_memory_budget(
            &config, 30_000, // n_vars
            16_384, // shard_target_rows
            10.0,   // avg_nnz_per_cell
        );
        // Budget should still return valid parameters
        assert!(budget.shard_group_size >= 1);
        assert!(budget.prefetch_batches >= 2);
    }

    #[test]
    fn test_memory_budget_128mb_reduced() {
        // Config with max_memory_mb=128: should reduce parameters
        let config = LoaderConfig {
            max_memory_mb: 128,
            ..LoaderConfig::default()
        };
        let budget = compute_memory_budget(
            &config, 30_000, // n_vars
            16_384, // shard_target_rows
            10.0,   // avg_nnz_per_cell
        );
        // At least one parameter should be reduced from default
        assert!(
            budget.shard_group_size < 8 || budget.prefetch_batches < 4,
            "128 MB budget should reduce at least one parameter: \
             shard_group_size={}, prefetch_batches={}",
            budget.shard_group_size,
            budget.prefetch_batches
        );
    }

    #[test]
    fn test_memory_budget_64mb_both_minimums() {
        // Very small budget (64 MB): both should be at minimums
        let config = LoaderConfig {
            max_memory_mb: 64,
            ..LoaderConfig::default()
        };
        let budget = compute_memory_budget(
            &config, 30_000, // n_vars
            16_384, // shard_target_rows
            10.0,   // avg_nnz_per_cell
        );
        // Both should be at their minimums
        assert_eq!(
            budget.shard_group_size, 1,
            "shard_group_size should be at minimum 1"
        );
        assert_eq!(
            budget.prefetch_batches, 2,
            "prefetch_batches should be at minimum 2"
        );
    }
}
