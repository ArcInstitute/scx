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
    /// If true (default), `process_plan` reorders the plan by
    /// `min(shard_of(p), shard_of(c))` before gathering, so the returned
    /// rows of `x` / `x_paired` and `pairs` are coherent in shard-locality
    /// order. Consumers that need strict input-order outputs can pass
    /// `sort_by_shard=False`.
    sort_by_shard: bool,
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
        sort_by_shard: bool,
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
            sort_by_shard,
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

    /// Whether `process_plan` reorders pairs by `min(shard_of(p), shard_of(c))`
    /// before gathering.
    pub fn sort_by_shard(&self) -> bool {
        self.sort_by_shard
    }

    /// O(log n_shards) lookup of the shard containing `row`. Returns `None`
    /// for rows outside every shard's range (should not happen for valid
    /// `row < n_obs` on a well-formed file).
    fn shard_of(&self, row: u64) -> Option<usize> {
        self.backed.index().shard_for_row(row)
    }

    /// Process a single plan: validate, gather both sides, project, normalize,
    /// extract obs.
    ///
    /// Empty plans yield a zero-row `IndexPlanBatch`. Out-of-range indices
    /// short-circuit before any I/O. Duplicate row indices in the plan are
    /// passed through to `read_row_indices`, which returns duplicate output
    /// rows in the same order — consumer is responsible for deduplication if
    /// it matters.
    ///
    /// If `sort_by_shard` is enabled (default), the plan is reordered by
    /// `min(shard_of(p), shard_of(c))` before gathering. The returned
    /// `pairs` field reflects the post-sort order: row `i` of `x` / `x_paired`
    /// always corresponds to `pairs[i]`, regardless of the input order.
    pub fn process_plan(&self, mut plan: Vec<(u64, u64)>) -> Result<IndexPlanBatch> {
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

        // Phase 2: reorder the plan by shard-of-min-row so consecutive pairs
        // land on contiguous shards. read_row_indices already sorts internally
        // for *gather* locality; this sort is for *plan-level coherence* — so
        // the consumer's `pairs` / `x` / `x_paired` arrays are aligned in the
        // post-sort order.
        //
        // Stable sort preserves the original plan order for ties (same shard).
        // `shard_of` is `None` only for malformed files; treat that as
        // `usize::MAX` so problematic pairs sink to the end without bailing.
        if self.sort_by_shard {
            plan.sort_by_key(|&(p, c)| {
                let sp = self.shard_of(p).unwrap_or(usize::MAX);
                let sc = self.shard_of(c).unwrap_or(usize::MAX);
                sp.min(sc)
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc as StdArc;

    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use scx_codec::{CodecId, ValueEncoding};
    use scx_format::header::{FileHeader, MAGIC};
    use scx_format::writer::ScxWriter;

    /// Build a minimal multi-shard `.scx` file. Each row `r` has a single
    /// non-zero at column `r % n_vars` with value `((r + 1) & 0xFF) as u8`,
    /// so `(row, col, value)` is recoverable from the row index alone.
    fn write_multi_shard_fixture(
        path: &std::path::Path,
        n_obs: usize,
        n_vars: usize,
        n_shards: usize,
    ) -> std::path::PathBuf {
        assert!(n_obs % n_shards == 0, "n_obs must divide n_shards in this fixture");
        let rows_per_shard = n_obs / n_shards;

        let header = FileHeader {
            magic: MAGIC,
            format_version: 1,
            header_length: 256,
            flags: 0,
            n_obs: n_obs as u64,
            n_vars: n_vars as u64,
            nnz: n_obs as u64,
            n_csr_shards: 0,
            n_csc_shards: 0,
            shard_target_rows: rows_per_shard as u32,
            codec_id: 0,
            index_dtype: 0,
            endian: 0,
            reserved_padding: 0,
            root_catalog_offset: 0,
            root_catalog_length: 0,
            full_catalog_offset: 0,
            full_catalog_length: 0,
            manifest_sequence: 1,
            prev_catalog_offset: 0,
            file_checksum: 0,
            front_catalog_offset: 0,
            front_catalog_length: 0,
            reserved: [0u8; 132],
        };
        let mut writer = ScxWriter::new(path, header).unwrap();

        let obs_schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
        let cell_ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
        let obs = arrow::record_batch::RecordBatch::try_new(
            StdArc::new(obs_schema),
            vec![StdArc::new(StringArray::from(
                cell_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap();
        writer.write_obs(&obs).unwrap();

        let var_schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
        let gene_ids: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
        let var = arrow::record_batch::RecordBatch::try_new(
            StdArc::new(var_schema),
            vec![StdArc::new(StringArray::from(
                gene_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap();
        writer.write_var(&var).unwrap();

        for s in 0..n_shards {
            let row_start = s * rows_per_shard;
            let mut indptr = vec![0u64];
            let mut indices = Vec::new();
            let mut values = Vec::new();
            for local in 0..rows_per_shard {
                let row = row_start + local;
                let col = (row % n_vars) as u32;
                let val = ((row + 1) & 0xFF) as u8;
                indices.push(col);
                values.push(val);
                indptr.push(*indptr.last().unwrap() + 1);
            }
            writer
                .write_csr_shard(
                    &indptr,
                    &indices,
                    &values,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    row_start as u64,
                )
                .unwrap();
        }

        writer.finish().unwrap();
        path.to_path_buf()
    }

    fn open_loader(
        path: &std::path::Path,
        sort_by_shard: bool,
    ) -> IndexPlanLoader {
        let mut config = LoaderConfig::default();
        config.normalize = false;
        config.log1p = false;
        config.obs_columns = vec!["cell_id".to_string()];
        IndexPlanLoader::new(path, config, /*cache_shards*/ 4, sort_by_shard).unwrap()
    }

    /// Phase 2.4: post-sort invariant — `pairs[i]` aligns with `x[i]` and
    /// `x_paired[i]`, and `pairs` is monotonically non-decreasing in
    /// `min(shard_of(p), shard_of(c))` after the sort.
    #[test]
    fn sort_by_shard_aligns_pairs_with_rows_and_is_monotone() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
        let loader = open_loader(&path, /*sort_by_shard*/ true);

        // Plan deliberately scrambled across shards.
        let plan: Vec<(u64, u64)> = vec![
            (15, 0),  // shards (3, 0) -> min 0
            (4, 5),   // shards (1, 1) -> min 1
            (8, 9),   // shards (2, 2) -> min 2
            (3, 12),  // shards (0, 3) -> min 0
            (10, 11), // shards (2, 2) -> min 2
            (1, 2),   // shards (0, 0) -> min 0
        ];
        let batch = loader.process_plan(plan.clone()).unwrap();
        let n_cols = loader.n_output_cols();

        // Pairs must align with the rows of x / x_paired: each row encodes
        // (row_idx % n_vars, (row_idx + 1) & 0xFF) in our fixture.
        for i in 0..batch.pairs.len() {
            let (p, c) = batch.pairs[i];
            let p_out = &batch.x[i * n_cols..][..n_cols];
            let c_out = &batch.x_paired[i * n_cols..][..n_cols];

            let p_col = (p as usize) % n_cols;
            let c_col = (c as usize) % n_cols;
            assert_eq!(p_out[p_col], ((p as usize + 1) & 0xFF) as f32);
            assert_eq!(c_out[c_col], ((c as usize + 1) & 0xFF) as f32);
        }

        // Post-sort key is non-decreasing.
        let keys: Vec<usize> = batch
            .pairs
            .iter()
            .map(|&(p, c)| {
                let sp = loader.shard_of(p).unwrap();
                let sc = loader.shard_of(c).unwrap();
                sp.min(sc)
            })
            .collect();
        for w in keys.windows(2) {
            assert!(w[0] <= w[1], "post-sort plan must be non-decreasing in min-shard");
        }
    }

    /// Phase 2.4: parity invariant — for the same input plan, sorted vs
    /// unsorted runs produce the same `(pair, x_row, x_paired_row)` *set*
    /// (just permuted). HVG-on and HVG-off both verified.
    #[test]
    fn sort_by_shard_is_a_pure_permutation_of_unsorted_output() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);

        let plan: Vec<(u64, u64)> = vec![
            (15, 0),
            (4, 5),
            (8, 9),
            (3, 12),
            (10, 11),
            (1, 2),
        ];

        let unsorted = open_loader(&path, false).process_plan(plan.clone()).unwrap();
        let sorted = open_loader(&path, true).process_plan(plan.clone()).unwrap();

        // Unsorted preserves input plan order (sanity).
        assert_eq!(unsorted.pairs, plan);

        // Build (pair, row, paired_row) tuples and compare as multisets.
        let n_cols = unsorted.x.len() / unsorted.pairs.len();
        let triples = |b: &IndexPlanBatch| -> Vec<((u64, u64), Vec<u32>, Vec<u32>)> {
            (0..b.pairs.len())
                .map(|i| {
                    let r: Vec<u32> =
                        b.x[i * n_cols..][..n_cols].iter().map(|v| v.to_bits()).collect();
                    let pr: Vec<u32> = b.x_paired[i * n_cols..][..n_cols]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect();
                    (b.pairs[i], r, pr)
                })
                .collect()
        };
        let mut unsorted_triples = triples(&unsorted);
        let mut sorted_triples = triples(&sorted);
        unsorted_triples.sort_by_key(|t| t.0);
        sorted_triples.sort_by_key(|t| t.0);
        assert_eq!(unsorted_triples, sorted_triples);
    }

    /// Empty plan + sort_by_shard=true must short-circuit cleanly.
    #[test]
    fn sort_by_shard_handles_empty_plan() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 8, 4, 2);
        let loader = open_loader(&path, true);
        let batch = loader.process_plan(Vec::new()).unwrap();
        assert!(batch.x.is_empty());
        assert!(batch.x_paired.is_empty());
        assert!(batch.pairs.is_empty());
    }
}
