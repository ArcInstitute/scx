//! Plan-driven paired-batch reads from an `.scx` file.
//!
//! Sibling to the sequential `TrainingPipeline`. The consumer supplies a stream
//! of `Vec<(u64, u64)>` plans (perturbed, control row index pairs); this module
//! gathers the rows via `BackedCsrReader::read_row_indices`, projects + normalizes
//! them through the existing `HvgProjection` / `fused_normalize_log1p_dense`
//! primitives, and yields paired dense `IndexPlanBatch` values.
//!
//! See `PER-CELL-CONTROL-PAIRING.md` at the workspace root for the full design.
//!
//! Phase 1 — single-threaded core. The iterator surface, lookahead prefetch,
//! shard-sorted plan iteration, and vectorised pair scatter land in later phases.

use std::collections::HashMap;
use std::path::Path;

use arrow::record_batch::RecordBatch;

use scx_format::{BackedCsrReader, ScxReader};

use crate::batch::ObsColumn;
use crate::decode_stage::extract_obs_columns;
use crate::error::{LoaderError, Result};
use crate::normalize::fused_normalize_log1p_dense;
use crate::pipeline::LoaderConfig;
use crate::projection::{scatter_row_full, HvgProjection};

/// One paired batch produced by `IndexPlanLoader`.
///
/// `x` and `x_paired` are row-major dense `[B * n_output_cols]` buffers in plan
/// order; `pairs[i]` is the `(pert_idx, ctrl_idx)` whose expression occupies
/// row `i` of both `x` and `x_paired`.
pub struct IndexPlanBatch {
    /// Perturbed-side dense expression `[n_pairs * n_output_cols]`.
    pub x: Vec<f32>,
    /// Control-side dense expression `[n_pairs * n_output_cols]`.
    pub x_paired: Vec<f32>,
    /// `(pert_idx, ctrl_idx)` pairs, in the order rows appear in `x` / `x_paired`.
    pub pairs: Vec<(u64, u64)>,
    /// Obs columns gathered for the perturbed side.
    pub obs: HashMap<String, ObsColumn>,
    /// Obs columns gathered for the control side.
    pub obs_paired: HashMap<String, ObsColumn>,
}

impl IndexPlanBatch {
    pub fn n_pairs(&self) -> usize {
        self.pairs.len()
    }
}

/// Plan-driven paired-batch reader.
///
/// Phase 1 surface: construct via `new`, drive via `process_plan`. The iterator
/// API (`iter_with_plans`) lands in Phase 4.
pub struct IndexPlanLoader {
    backed: BackedCsrReader,
    obs_metadata: RecordBatch,
    config: LoaderConfig,
    hvg_projection: Option<HvgProjection>,
    n_output_cols: usize,
}

impl IndexPlanLoader {
    /// Open an SCX file for plan-driven reads.
    ///
    /// Validates HVG indices and obs columns at construction so misconfiguration
    /// surfaces before the first `process_plan` call.
    ///
    /// `cache_shards` sizes the LRU shard cache inside `BackedCsrReader`; the
    /// shard cache hit rate is the dominant performance lever for plan-driven
    /// access patterns. Must be `>= 1`.
    ///
    /// Sequential-pipeline-only fields on `LoaderConfig` (`batch_size`,
    /// `shard_group_size`, `prefetch_batches`, `seed`) are silently ignored on
    /// this path.
    pub fn new(
        path: impl AsRef<Path>,
        config: LoaderConfig,
        cache_shards: usize,
    ) -> Result<Self> {
        if cache_shards < 1 {
            return Err(LoaderError::ConfigError {
                reason: "cache_shards must be >= 1".to_string(),
            });
        }

        let reader = ScxReader::open(path.as_ref())?;
        let n_obs = reader.n_obs();
        let n_vars = reader.n_vars();

        // Borrow obs / sizes off ScxReader before BackedCsrReader::new takes ownership.
        let obs_metadata = reader.read_obs()?;

        // Validate obs columns up front — fail at construction, not on first batch.
        if !config.obs_columns.is_empty() && n_obs > 0 {
            extract_obs_columns(&obs_metadata, &[0u64], &config.obs_columns)?;
        }

        // HvgProjection::new does not validate against n_vars — do it here so
        // out-of-range indices fail at construction rather than panicking later
        // in scatter_row.
        let hvg_projection = match &config.hvg_indices {
            Some(idxs) => {
                let n_vars_u32: u32 = u32::try_from(n_vars).map_err(|_| {
                    LoaderError::ConfigError {
                        reason: format!(
                            "n_vars={n_vars} exceeds u32::MAX; HVG indices use u32"
                        ),
                    }
                })?;
                if let Some(&bad) = idxs.iter().find(|&&i| i >= n_vars_u32) {
                    return Err(LoaderError::ConfigError {
                        reason: format!(
                            "HVG index {bad} is out of range (n_vars={n_vars})"
                        ),
                    });
                }
                Some(HvgProjection::new(idxs.clone()))
            }
            None => None,
        };

        let n_output_cols = hvg_projection
            .as_ref()
            .map(|p| p.n_output_cols())
            .unwrap_or(n_vars as usize);

        let backed = BackedCsrReader::new(reader, cache_shards);

        Ok(Self {
            backed,
            obs_metadata,
            config,
            hvg_projection,
            n_output_cols,
        })
    }

    pub fn n_obs(&self) -> u64 {
        self.backed.n_obs() as u64
    }

    pub fn n_vars(&self) -> u64 {
        self.backed.n_vars() as u64
    }

    pub fn n_output_cols(&self) -> usize {
        self.n_output_cols
    }

    /// Process a single plan: validate, gather both sides, project, normalize,
    /// extract obs.
    ///
    /// Empty plans yield a zero-row `IndexPlanBatch`. Out-of-range indices
    /// short-circuit before any I/O. Duplicate row indices in the plan are
    /// passed through to `read_row_indices`, which returns duplicate output
    /// rows in the same order — consumer is responsible for deduplication if
    /// it matters.
    pub fn process_plan(&self, plan: Vec<(u64, u64)>) -> Result<IndexPlanBatch> {
        let n_obs = self.n_obs();

        for &(p, c) in &plan {
            if p >= n_obs {
                return Err(LoaderError::IndexOutOfRange {
                    idx: p,
                    n_obs: n_obs as usize,
                });
            }
            if c >= n_obs {
                return Err(LoaderError::IndexOutOfRange {
                    idx: c,
                    n_obs: n_obs as usize,
                });
            }
        }

        let n_pairs = plan.len();
        let n_cols = self.n_output_cols;

        if n_pairs == 0 {
            return Ok(IndexPlanBatch {
                x: Vec::new(),
                x_paired: Vec::new(),
                pairs: Vec::new(),
                obs: HashMap::new(),
                obs_paired: HashMap::new(),
            });
        }

        // One read_row_indices call per side. The reader sorts indices for
        // shard locality internally, decodes via the LRU shard cache, then
        // restores caller-supplied order in the returned ScxCsr.
        let pert_indices: Vec<u64> = plan.iter().map(|(p, _)| *p).collect();
        let ctrl_indices: Vec<u64> = plan.iter().map(|(_, c)| *c).collect();
        let pert_csr = self.backed.read_row_indices(&pert_indices)?;
        let ctrl_csr = self.backed.read_row_indices(&ctrl_indices)?;

        let mut x = vec![0f32; n_pairs * n_cols];
        let mut x_paired = vec![0f32; n_pairs * n_cols];

        // Direct indptr/indices/data slicing — ScxCsr::row_slice would allocate
        // a fresh i64 indptr per row.
        for i in 0..n_pairs {
            let pi_lo = pert_csr.indptr[i] as usize;
            let pi_hi = pert_csr.indptr[i + 1] as usize;
            let ci_lo = ctrl_csr.indptr[i] as usize;
            let ci_hi = ctrl_csr.indptr[i + 1] as usize;

            let p_out = &mut x[i * n_cols..][..n_cols];
            let c_out = &mut x_paired[i * n_cols..][..n_cols];

            match &self.hvg_projection {
                Some(hvg) => {
                    hvg.scatter_row(
                        &pert_csr.indices[pi_lo..pi_hi],
                        &pert_csr.data[pi_lo..pi_hi],
                        p_out,
                    );
                    hvg.scatter_row(
                        &ctrl_csr.indices[ci_lo..ci_hi],
                        &ctrl_csr.data[ci_lo..ci_hi],
                        c_out,
                    );
                }
                None => {
                    scatter_row_full(
                        &pert_csr.indices[pi_lo..pi_hi],
                        &pert_csr.data[pi_lo..pi_hi],
                        p_out,
                    )?;
                    scatter_row_full(
                        &ctrl_csr.indices[ci_lo..ci_hi],
                        &ctrl_csr.data[ci_lo..ci_hi],
                        c_out,
                    )?;
                }
            }

            if self.config.normalize {
                fused_normalize_log1p_dense(p_out, self.config.target_sum);
                fused_normalize_log1p_dense(c_out, self.config.target_sum);
            }
        }

        let obs = extract_obs_columns(&self.obs_metadata, &pert_indices, &self.config.obs_columns)?;
        let obs_paired =
            extract_obs_columns(&self.obs_metadata, &ctrl_indices, &self.config.obs_columns)?;

        Ok(IndexPlanBatch {
            x,
            x_paired,
            pairs: plan,
            obs,
            obs_paired,
        })
    }
}
