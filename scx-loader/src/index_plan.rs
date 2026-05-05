//! Plan-driven paired-batch reads from an `.scx` file.
//!
//! Sibling to the sequential `TrainingPipeline`. The consumer supplies a stream
//! of `Vec<(u64, u64)>` plans (perturbed, control row index pairs); this module
//! gathers both sides through one shard-local dense pass, projects + normalizes
//! them through the existing `HvgProjection` and the shared
//! `apply_dense_transforms` dispatcher in `normalize.rs` (the same dispatcher
//! `TrainingDataset` uses), and yields paired dense `IndexPlanBatch` values.
//!
//!
//! Surface: synchronous [`IndexPlanLoader::process_plan`] for one-shot batch
//! gathering, plus an async [`IndexPlanLoader::iter_with_plans`] iterator that
//! pipelines per-plan shard prefetch (via tokio `spawn_blocking`) ahead of the
//! consumer. Plans are reordered by `min(shard_of(p), shard_of(c))` for
//! locality when `sort_by_shard=true` (the default). Perturbed and control
//! requests are grouped together by shard, so each touched cached shard is
//! walked once and rows scatter directly into the final dense buffers without
//! materialising intermediate `ScxCsr` values.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use arrow::record_batch::RecordBatch;
use crossbeam_channel::{bounded, Receiver};
use scx_format::{BackedCsrReader, ScxReader};
use scx_sparse::ScxCsr;
use tokio::runtime::Runtime;
use tokio::task::JoinHandle;

use crate::batch::ObsColumn;
use crate::budget::{profiling_enabled, BudgetBreakdown, PYTHON_OVERHEAD_BYTES};
use crate::decode_stage::extract_obs_columns;
use crate::error::{LoaderError, Result};
use crate::normalize::apply_dense_transforms;
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PairSide {
    Perturbed,
    Control,
}

#[derive(Clone, Copy, Debug)]
struct PairRequest {
    row: u64,
    pair_idx: usize,
    side: PairSide,
}

struct PairedDenseGather {
    x: Vec<f32>,
    x_paired: Vec<f32>,
    pert_indices: Vec<u64>,
    ctrl_indices: Vec<u64>,
}

/// Plan-driven paired-batch reader.
///
/// Two surfaces share `process_plan` underneath: the synchronous
/// [`Self::process_plan`] for one-shot use, and [`Self::iter_with_plans`] —
/// an async iterator that pipelines per-plan shard prefetch (via tokio
/// `spawn_blocking`) ahead of the consumer.
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
    /// Tokio runtime used by [`IndexPlanIter`] to spawn shard prefetches via
    /// `spawn_blocking`. `read_shard_cached_arc` is synchronous and CPU/IO
    /// bound, so it must run on the blocking pool — bare `tokio::spawn` does
    /// not accept it. 2 worker threads matches `TrainingPipeline`.
    runtime: Runtime,
    /// Effective LRU shard cache size after auto-tuning to fit
    /// `max_memory_mb`. May be less than the user-requested `cache_shards`.
    effective_cache_shards: usize,
    /// Effective default lookahead after auto-tuning. `iter_with_plans`
    /// uses this when the caller passes `lookahead=None`. May be less than
    /// the user-requested `lookahead`.
    effective_lookahead: usize,
    /// Hard ceiling on rows-per-batch. The constructor sizes the dense
    /// `x` / `x_paired` budget against this value, and `process_plan`
    /// rejects plans larger than it so a misbehaving consumer can't
    /// silently exceed `max_memory_mb`.
    max_plan_size: usize,
    /// Shared handle to the underlying `BackedCsrReader`'s cache counters.
    /// Always populated — `BackedCsrReader::enable_metrics` is called in
    /// [`Self::new`] so the iter's profile log and the consumer-side
    /// snapshot accessor have a stable handle.
    cache_metrics: Arc<scx_format::CacheMetrics>,
    /// Per-component memory breakdown produced by the auto-tune at
    /// construction. Surfaced through `IndexPlanDataset.memory_budget()` for
    /// production sizing.
    budget_breakdown: BudgetBreakdown,
}

impl IndexPlanLoader {
    /// Open an SCX file for plan-driven reads.
    ///
    /// Validates HVG indices and obs columns at construction so misconfiguration
    /// surfaces before the first `process_plan` call.
    ///
    /// `cache_shards` sizes the LRU shard cache inside `BackedCsrReader`; the
    /// shard cache hit rate is the dominant performance lever for plan-driven
    /// access patterns. Must be `>= 1`. The value passed here is the *target*;
    /// auto-tuning may reduce it to fit `config.max_memory_mb` (visible via
    /// [`Self::effective_cache_shards`]).
    ///
    /// `lookahead` is the *target* default lookahead used by `iter_with_plans`
    /// when the caller passes `lookahead=None`; auto-tuning may reduce it to
    /// fit `config.max_memory_mb` (visible via [`Self::effective_lookahead`]).
    /// `lookahead=0` disables the prefetch path entirely.
    ///
    /// `max_plan_size` is the hard ceiling on rows-per-batch. It is both
    /// used as the upper bound for the memory-budget calculation at
    /// construction and enforced at every `process_plan` / `iter_with_plans`
    /// call — plans longer than this are rejected with a `ConfigError` so a
    /// misbehaving consumer cannot silently exceed `max_memory_mb`.
    /// Default 16384.
    ///
    /// Sequential-pipeline-only fields on `LoaderConfig` (`batch_size`,
    /// `shard_group_size`, `prefetch_batches`, `seed`) are silently ignored on
    /// this path.
    pub fn new(
        path: impl AsRef<Path>,
        config: LoaderConfig,
        cache_shards: usize,
        sort_by_shard: bool,
        lookahead: usize,
        max_plan_size: usize,
    ) -> Result<Self> {
        if cache_shards < 1 {
            return Err(LoaderError::ConfigError {
                reason: "cache_shards must be >= 1".to_string(),
            });
        }
        if max_plan_size < 1 {
            return Err(LoaderError::ConfigError {
                reason: "max_plan_size must be >= 1".to_string(),
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
                let n_vars_u32: u32 =
                    u32::try_from(n_vars).map_err(|_| LoaderError::ConfigError {
                        reason: format!("n_vars={n_vars} exceeds u32::MAX; HVG indices use u32"),
                    })?;
                if let Some(&bad) = idxs.iter().find(|&&i| i >= n_vars_u32) {
                    return Err(LoaderError::ConfigError {
                        reason: format!("HVG index {bad} is out of range (n_vars={n_vars})"),
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

        // ----- Memory budget auto-tune -----------------------------------
        //
        // Per-component model (rendered uniformly via `BudgetBreakdown` —
        // see `crate::budget`):
        //   shard_decoded_bytes = (avg_nnz_per_shard × 8) + (avg_rows_per_shard × 8)
        //                                    ^^^ i32 indices (4) + f32 data (4)
        //                                                                ^^^ i64 indptr (8)
        //   cache_bytes        = effective_cache_shards × shard_decoded_bytes
        //   batch_buffer_bytes = 2 × max_plan_size × n_output_cols × 4
        //                            (paired x / x_paired)
        //   lookahead_overhead = effective_lookahead × max_plan_size × 16
        //                                                          ^^^ (u64, u64) plan tuple
        //   transient_bytes    = obs_extraction_bytes + pair_request_buffer_bytes
        //                            (per-batch, co-resident with batch_buffer)
        //   python_overhead    = PYTHON_OVERHEAD_BYTES (50 MB constant)
        //
        // Mmap'd file is in the kernel page cache (evicted under pressure)
        // and intentionally NOT counted against the budget.
        let (avg_nnz_per_shard, avg_rows_per_shard) = {
            let csr_shards = reader.catalog().shards_sorted();
            let n_csr_shards = csr_shards.len();
            if n_csr_shards == 0 {
                (0u64, 0u64)
            } else {
                let total_nnz: u64 = csr_shards
                    .iter()
                    .map(|e| e.stats.as_ref().map(|s| s.nnz).unwrap_or(0))
                    .sum();
                let total_rows: u64 = csr_shards
                    .iter()
                    .map(|e| {
                        e.stats
                            .as_ref()
                            .map(|s| s.row_end - s.row_start)
                            .unwrap_or(0)
                    })
                    .sum();
                (
                    total_nnz / n_csr_shards as u64,
                    total_rows / n_csr_shards as u64,
                )
            }
        };
        let shard_decoded_bytes =
            (avg_nnz_per_shard.saturating_mul(8) + avg_rows_per_shard.saturating_mul(8)) as usize;

        let batch_buffer_bytes = 2usize
            .saturating_mul(max_plan_size)
            .saturating_mul(n_output_cols)
            .saturating_mul(4);
        const PLAN_TUPLE_BYTES: usize = 16; // sizeof((u64, u64))
                                            // `gather_pairs_dense` request scratch holds 2 × max_plan_size
                                            // `PairRequest`s. Use `size_of` so this term tracks struct churn
                                            // automatically instead of drifting against a hand-derived constant.
        const PAIR_REQUEST_BYTES: usize = std::mem::size_of::<PairRequest>();
        // `extract_obs_columns` allocates one `Vec` per configured obs
        // column per side (pert + ctrl). `ObsColumn` variants store `i64`
        // (8 B), `f64` (8 B), or `Categorical(Vec<u32>, …)` (4 B/cell —
        // the per-batch label dictionary is shared, not per-cell, and
        // intentionally excluded from this term). `8` is the worst-case
        // per-cell width across the supported variants.
        const OBS_CELL_BYTES: usize = 8;

        let transient_bytes = {
            let obs_bytes = 2usize
                .saturating_mul(max_plan_size)
                .saturating_mul(config.obs_columns.len())
                .saturating_mul(OBS_CELL_BYTES);
            let request_bytes = 2usize
                .saturating_mul(max_plan_size)
                .saturating_mul(PAIR_REQUEST_BYTES);
            obs_bytes.saturating_add(request_bytes)
        };

        let budget_bytes = config
            .max_memory_mb
            .saturating_mul(1024)
            .saturating_mul(1024);

        let mut effective_cache_shards = cache_shards;
        let mut effective_lookahead = lookahead;

        let breakdown = |cache: usize, la: usize| -> BudgetBreakdown {
            BudgetBreakdown::new(
                cache.saturating_mul(shard_decoded_bytes),
                batch_buffer_bytes,
                la.saturating_mul(max_plan_size)
                    .saturating_mul(PLAN_TUPLE_BYTES),
                transient_bytes,
                PYTHON_OVERHEAD_BYTES,
            )
        };

        // Reduce lookahead first (down to 1 — we want the iterator path to
        // remain functional even under tight memory; an explicit
        // `lookahead == 0` is preserved through the loop because the
        // `> 1` guard never decrements it). Then reduce cache_shards down
        // to 1.
        while breakdown(effective_cache_shards, effective_lookahead).total_bytes > budget_bytes {
            if effective_lookahead > 1 {
                effective_lookahead -= 1;
            } else if effective_cache_shards > 1 {
                effective_cache_shards -= 1;
            } else {
                let b = breakdown(effective_cache_shards, effective_lookahead);
                return Err(LoaderError::ConfigError {
                    reason: format!(
                        "max_memory_mb={} is below the floor for this file: \
                         estimated {} MB at cache_shards=1, lookahead=1 \
                         (shard_decoded={} KB, batch_buffer={} MB, transient={} KB, \
                         py_overhead=50 MB). Increase max_memory_mb.",
                        config.max_memory_mb,
                        b.total_bytes / (1024 * 1024),
                        shard_decoded_bytes / 1024,
                        batch_buffer_bytes / (1024 * 1024),
                        transient_bytes / 1024,
                    ),
                });
            }
        }
        let budget_breakdown = breakdown(effective_cache_shards, effective_lookahead);

        // Tighten the LRU's byte cap to match the auto-tune model, so the
        // cache can't overshoot `max_memory_mb` when actual shard sizes
        // diverge from the average used in `shard_decoded_bytes`. The count
        // cap stays in place too — both are enforced.
        let cache_bytes_budget = effective_cache_shards.saturating_mul(shard_decoded_bytes);
        let mut backed = BackedCsrReader::new_with_byte_budget(
            reader,
            effective_cache_shards,
            cache_bytes_budget,
        );
        // Always-on metrics on this surface — the iter's profile log and
        // the per-iter snapshot accessor read from this handle.
        let cache_metrics = backed.enable_metrics();

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("scx-index-plan")
            .build()
            .map_err(|e| {
                LoaderError::ShutdownError(format!(
                    "failed to create tokio runtime for IndexPlanLoader: {e}"
                ))
            })?;

        Ok(Self {
            backed,
            obs_metadata,
            config,
            hvg_projection,
            n_output_cols,
            sort_by_shard,
            runtime,
            effective_cache_shards,
            effective_lookahead,
            max_plan_size,
            cache_metrics,
            budget_breakdown,
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

    /// Shared handle to the underlying `BackedCsrReader`'s cache counters
    /// (hits / misses / evictions / bytes_inserted / duplicate_waiters).
    /// Always populated; cloning the `Arc` lets callers sample without
    /// touching the cache lock.
    pub fn cache_metrics(&self) -> Arc<scx_format::CacheMetrics> {
        Arc::clone(&self.cache_metrics)
    }

    /// Effective LRU shard cache size after auto-tuning to fit
    /// `config.max_memory_mb`. May be less than the user-requested value.
    pub fn effective_cache_shards(&self) -> usize {
        self.effective_cache_shards
    }

    /// Effective default lookahead after auto-tuning. Used by
    /// `iter_with_plans` when the caller passes `lookahead=None`.
    pub fn effective_lookahead(&self) -> usize {
        self.effective_lookahead
    }

    /// Per-component memory breakdown produced by the auto-tune at
    /// construction. Surfaces the cache / batch / lookahead / transient /
    /// python-overhead split that drove the effective `cache_shards` and
    /// `lookahead` reductions, for production sizing and benchmark
    /// validation.
    pub fn budget_breakdown(&self) -> BudgetBreakdown {
        self.budget_breakdown
    }

    /// User-facing memory budget in MB (the `max_memory_mb` passed to
    /// `LoaderConfig`). Convenience accessor — paired with
    /// [`Self::budget_breakdown`] for sizing diagnostics.
    pub fn max_memory_mb(&self) -> usize {
        self.config.max_memory_mb
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
    /// short-circuit before any I/O. Duplicate row indices in the plan produce
    /// duplicate output rows in the same order as the returned `pairs` field.
    ///
    /// If `sort_by_shard` is enabled (default), the plan is reordered by
    /// `min(shard_of(p), shard_of(c))` before gathering. The returned
    /// `pairs` field reflects the post-sort order: row `i` of `x` / `x_paired`
    /// always corresponds to `pairs[i]`, regardless of the input order.
    pub fn process_plan(&self, mut plan: Vec<(u64, u64)>) -> Result<IndexPlanBatch> {
        if plan.len() > self.max_plan_size {
            return Err(LoaderError::ConfigError {
                reason: format!(
                    "plan size {} exceeds max_plan_size {} (raise max_plan_size at \
                     construction or split the plan)",
                    plan.len(),
                    self.max_plan_size,
                ),
            });
        }

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

        if n_pairs == 0 {
            return Ok(IndexPlanBatch {
                x: Vec::new(),
                x_paired: Vec::new(),
                pairs: Vec::new(),
                obs: HashMap::new(),
                obs_paired: HashMap::new(),
            });
        }

        // Reorder the plan by shard-of-min-row so consecutive pairs land on
        // contiguous shards. The fused gather sorts row requests internally
        // for shard locality; this sort is for plan-level coherence, so the
        // consumer's `pairs` / `x` / `x_paired` arrays are aligned in the
        // post-sort order.
        //
        // Stable sort preserves the original plan order for ties (same shard).
        // `shard_of` is `None` only for malformed files; treat that as
        // `usize::MAX` so problematic pairs sink to the end without bailing.
        // Use `sort_by_cached_key` so each pair's two binary-search lookups
        // run exactly once — `sort_by_key` may re-invoke the closure during
        // merges, doubling the lookup cost for larger plans.
        if self.sort_by_shard {
            plan.sort_by_cached_key(|&(p, c)| {
                let sp = self.shard_of(p).unwrap_or(usize::MAX);
                let sc = self.shard_of(c).unwrap_or(usize::MAX);
                sp.min(sc)
            });
        }

        let gathered = self.gather_pairs_dense(&plan)?;

        let obs = extract_obs_columns(
            &self.obs_metadata,
            &gathered.pert_indices,
            &self.config.obs_columns,
        )?;
        let obs_paired = extract_obs_columns(
            &self.obs_metadata,
            &gathered.ctrl_indices,
            &self.config.obs_columns,
        )?;

        Ok(IndexPlanBatch {
            x: gathered.x,
            x_paired: gathered.x_paired,
            pairs: plan,
            obs,
            obs_paired,
        })
    }

    /// Gather both sides of a paired plan in one shard-local pass.
    ///
    /// The request list contains perturbed and control rows together, sorted
    /// by row so all requests for a shard are contiguous. Repeated row indices
    /// reuse the same CSR row slice and scatter once per destination
    /// occurrence, preserving duplicate-pair semantics without repeating shard
    /// lookup or row-pointer work.
    fn gather_pairs_dense(&self, plan: &[(u64, u64)]) -> Result<PairedDenseGather> {
        let n_pairs = plan.len();
        let n_cols = self.n_output_cols;

        let mut x = vec![0f32; n_pairs * n_cols];
        let mut x_paired = vec![0f32; n_pairs * n_cols];
        let mut pert_indices = Vec::with_capacity(n_pairs);
        let mut ctrl_indices = Vec::with_capacity(n_pairs);
        let mut requests = Vec::with_capacity(n_pairs * 2);

        for (pair_idx, &(pert, ctrl)) in plan.iter().enumerate() {
            pert_indices.push(pert);
            ctrl_indices.push(ctrl);
            requests.push(PairRequest {
                row: pert,
                pair_idx,
                side: PairSide::Perturbed,
            });
            requests.push(PairRequest {
                row: ctrl,
                pair_idx,
                side: PairSide::Control,
            });
        }

        requests.sort_by_key(|r| r.row);

        let mut start = 0;
        while start < requests.len() {
            let row = requests[start].row;
            let shard_idx = self.backed.index().shard_for_row(row).ok_or_else(|| {
                scx_format::ScxError::Io(std::io::Error::other(format!(
                    "row index {row} is not covered by any shard (n_obs={})",
                    self.n_obs()
                )))
            })?;
            let (s_start, s_end) = self.backed.index().shard_range(shard_idx).ok_or(
                scx_format::ScxError::ShardIndexOutOfBounds {
                    index: shard_idx,
                    count: self.backed.index().n_shards(),
                },
            )?;

            let end = start + requests[start..].partition_point(|r| r.row < s_end);
            let shard = self.backed.read_shard_cached_arc(shard_idx)?;

            let mut row_start = start;
            while row_start < end {
                let row = requests[row_start].row;
                let row_end =
                    row_start + requests[row_start..end].partition_point(|r| r.row == row);
                let local = (row - s_start) as usize;
                let lo = *shard
                    .indptr
                    .get(local)
                    .ok_or(scx_format::ScxError::InconsistentCsr)?
                    as usize;
                let hi = *shard
                    .indptr
                    .get(local + 1)
                    .ok_or(scx_format::ScxError::InconsistentCsr)?
                    as usize;
                if hi < lo || hi > shard.indices.len() || hi > shard.data.len() {
                    return Err(scx_format::ScxError::InconsistentCsr.into());
                }
                let idx = &shard.indices[lo..hi];
                let data = &shard.data[lo..hi];

                for &request in &requests[row_start..row_end] {
                    self.scatter_pair_request(request, idx, data, n_cols, &mut x, &mut x_paired)?;
                }

                row_start = row_end;
            }

            start = end;
        }

        for i in 0..n_pairs {
            let p_out = &mut x[i * n_cols..][..n_cols];
            apply_dense_transforms(
                p_out,
                self.config.normalize,
                self.config.log1p,
                self.config.target_sum,
            );
            let c_out = &mut x_paired[i * n_cols..][..n_cols];
            apply_dense_transforms(
                c_out,
                self.config.normalize,
                self.config.log1p,
                self.config.target_sum,
            );
        }

        Ok(PairedDenseGather {
            x,
            x_paired,
            pert_indices,
            ctrl_indices,
        })
    }

    fn scatter_pair_request(
        &self,
        request: PairRequest,
        idx: &[i32],
        data: &[f32],
        n_cols: usize,
        x: &mut [f32],
        x_paired: &mut [f32],
    ) -> Result<()> {
        let out = match request.side {
            PairSide::Perturbed => &mut x[request.pair_idx * n_cols..][..n_cols],
            PairSide::Control => &mut x_paired[request.pair_idx * n_cols..][..n_cols],
        };
        match self.hvg_projection.as_ref() {
            Some(hvg) => hvg.scatter_row(idx, data, out),
            None => scatter_row_full(idx, data, out)?,
        }
        Ok(())
    }

    /// Drive the loader from a plan stream, hiding shard-prefetch latency
    /// behind upcoming-batch compute.
    ///
    /// `plans` is any `Iterator<Item = Result<Vec<(u64, u64)>, LoaderError>>` —
    /// the iterator is consumed lazily. `lookahead` controls how many
    /// upcoming plans get shard-prefetched concurrently with the current
    /// batch's decode (0 disables; default per Python API is 4).
    ///
    /// The loader's tokio runtime spawns one `spawn_blocking` task per shard
    /// referenced by an upcoming plan that is not already resident in the
    /// LRU. Decode of the head plan blocks on its prefetch handles before
    /// calling `process_plan`.
    pub fn iter_with_plans<I>(self: Arc<Self>, plans: I, lookahead: usize) -> IndexPlanIter
    where
        I: Iterator<Item = std::result::Result<Vec<(u64, u64)>, LoaderError>> + Send + 'static,
    {
        IndexPlanIter::new(self, plans, lookahead)
    }
}

/// Per-plan in-flight state: the plan itself plus one `spawn_blocking` join
/// handle per shard scheduled for prefetch (cached / in-flight shards are
/// filtered out by the iter pre-check before this Vec is built).
type ShardJoin = JoinHandle<scx_format::Result<Arc<ScxCsr>>>;

struct InFlight {
    plan: Vec<(u64, u64)>,
    /// Empty when `lookahead == 0`, when the plan is empty, or when every
    /// shard the plan touches is already cached or being decoded by a peer
    /// (the iter pre-check skips spawning in those cases — see
    /// `IndexPlanIter::spawn_prefetches`).
    prefetches: Vec<ShardJoin>,
}

/// Per-iter prefetch counters. Sampled by `IndexPlanIter::iter_metrics` and
/// emitted by the Drop-time profile log when `SCX_LOADER_PROFILE=1`.
///
/// All atomics use `Relaxed` ordering — values are statistical and not used
/// for synchronization.
#[derive(Default, Debug)]
pub struct IterMetrics {
    /// `tokio::spawn_blocking` tasks queued onto the runtime's blocking pool.
    pub prefetch_tasks_spawned: AtomicU64,
    /// Shards whose prefetch was skipped because the LRU already held them.
    pub prefetch_skipped_cache_hit: AtomicU64,
    /// Shards whose prefetch was skipped because a peer leader was already
    /// decoding them in `BackedCsrReader`'s singleflight table.
    pub prefetch_skipped_in_flight: AtomicU64,
}

/// Iterator returned by [`IndexPlanLoader::iter_with_plans`].
///
/// Drives the plan-pull thread, the per-plan shard prefetches, and the
/// per-batch decode. `next` blocks on the head plan's prefetch handles,
/// then delegates to [`IndexPlanLoader::process_plan`].
pub struct IndexPlanIter {
    loader: Arc<IndexPlanLoader>,
    plan_rx: Receiver<std::result::Result<Vec<(u64, u64)>, LoaderError>>,
    /// Pull worker that owns the user-supplied `plans` iterator. Detached on
    /// drop — the worker exits naturally when `plan_rx` is dropped (next
    /// `send` fails) or when the user iterator returns `None`.
    plan_thread: Option<thread::JoinHandle<()>>,
    in_flight: VecDeque<InFlight>,
    lookahead: usize,
    /// Sticky flag: once the plan stream is closed (StopIteration / disconnect)
    /// we stop calling `recv` so the iter drains the queue and finishes.
    plan_stream_done: bool,
    /// Latched error: the first Err yielded by the plan iterator. We stash
    /// rather than return immediately so the queue of already-pulled plans
    /// drains gracefully; next() surfaces the error one-shot after the
    /// queue empties, then sets `plan_stream_done`.
    plan_stream_error: Option<LoaderError>,
    /// Per-iter prefetch counters. Cloning the `Arc` lets a consumer sample
    /// without going through any lock.
    iter_metrics: Arc<IterMetrics>,
}

impl IndexPlanIter {
    fn new<I>(loader: Arc<IndexPlanLoader>, plans: I, lookahead: usize) -> Self
    where
        I: Iterator<Item = std::result::Result<Vec<(u64, u64)>, LoaderError>> + Send + 'static,
    {
        let cap = lookahead.max(1);
        let (plan_tx, plan_rx) = bounded(cap);

        let plan_thread = thread::Builder::new()
            .name("scx-index-plan-pull".to_string())
            .spawn(move || {
                for item in plans {
                    if plan_tx.send(item).is_err() {
                        // Receiver dropped — iter was dropped mid-stream.
                        break;
                    }
                }
                // Falling off the loop closes plan_tx, signalling EOS.
            })
            .ok();

        Self {
            loader,
            plan_rx,
            plan_thread,
            in_flight: VecDeque::with_capacity(cap),
            lookahead,
            plan_stream_done: false,
            plan_stream_error: None,
            iter_metrics: Arc::new(IterMetrics::default()),
        }
    }

    /// Cloneable handle to this iter's prefetch counters. Sample at any
    /// time — atomics are `Relaxed`, no locks involved.
    pub fn iter_metrics(&self) -> Arc<IterMetrics> {
        Arc::clone(&self.iter_metrics)
    }

    /// Refill the in-flight queue up to `lookahead.max(1)` plans, spawning a
    /// shard prefetch per shard referenced by each plan that is not already
    /// resident in the LRU.
    ///
    /// On a plan-stream error, latches the error in `plan_stream_error` and
    /// stops; the iterator drains `in_flight` first and surfaces the error
    /// one-shot after the queue empties. Plain end-of-stream sets
    /// `plan_stream_done` instead.
    fn refill(&mut self) {
        let target = self.lookahead.max(1);
        while self.in_flight.len() < target
            && !self.plan_stream_done
            && self.plan_stream_error.is_none()
        {
            match self.plan_rx.recv() {
                Ok(Ok(plan)) => {
                    let prefetches = self.spawn_prefetches(&plan);
                    self.in_flight.push_back(InFlight { plan, prefetches });
                }
                Ok(Err(e)) => {
                    self.plan_stream_error = Some(e);
                    break;
                }
                Err(_) => {
                    // plan_tx dropped → end of stream.
                    self.plan_stream_done = true;
                    break;
                }
            }
        }
    }

    fn spawn_prefetches(&self, plan: &[(u64, u64)]) -> Vec<ShardJoin> {
        if self.lookahead == 0 || plan.is_empty() {
            return Vec::new();
        }

        let mut all_rows: Vec<u64> = Vec::with_capacity(plan.len() * 2);
        for &(p, c) in plan {
            all_rows.push(p);
            all_rows.push(c);
        }
        // shards_for_indices internally sorts + dedups, so the returned
        // shard set has no duplicates we'd waste prefetches on.
        let shards = self.loader.backed.index().shards_for_indices(&all_rows);

        // Skip shards that are already cached or whose decode is already in
        // flight via the BackedCsrReader singleflight table. Without this
        // filter, a window of N plans touching shard S queues up to N
        // `spawn_blocking` tasks for S — the singleflight short-circuits
        // the redundant decode but the per-task tokio overhead and the
        // associated `runtime.block_on` round-trips are still paid in
        // `await_head`. Filtering here keeps the queue tight.
        let handle = self.loader.runtime.handle().clone();
        shards
            .into_iter()
            .filter(|&sidx| {
                if self.loader.backed.cache_contains(sidx) {
                    self.iter_metrics
                        .prefetch_skipped_cache_hit
                        .fetch_add(1, Ordering::Relaxed);
                    return false;
                }
                if self.loader.backed.in_flight_contains(sidx) {
                    self.iter_metrics
                        .prefetch_skipped_in_flight
                        .fetch_add(1, Ordering::Relaxed);
                    return false;
                }
                true
            })
            .map(|sidx| {
                self.iter_metrics
                    .prefetch_tasks_spawned
                    .fetch_add(1, Ordering::Relaxed);
                let loader = Arc::clone(&self.loader);
                handle.spawn_blocking(move || loader.backed.read_shard_cached_arc(sidx))
            })
            .collect()
    }

    /// Block on every prefetch handle for the head plan. Surfaces the first
    /// shard read error or join panic.
    fn await_head(&self, prefetches: Vec<ShardJoin>) -> std::result::Result<(), LoaderError> {
        for h in prefetches {
            match self.loader.runtime.block_on(h) {
                Ok(Ok(_arc_shard)) => {
                    // Shard is now warm in the LRU cache; subsequent
                    // process_plan calls will hit it through the dense gather.
                }
                Ok(Err(e)) => return Err(LoaderError::FormatError(e)),
                Err(join_err) => {
                    return Err(LoaderError::ShutdownError(format!(
                        "IndexPlanIter prefetch task panicked: {join_err}"
                    )))
                }
            }
        }
        Ok(())
    }
}

impl Iterator for IndexPlanIter {
    type Item = Result<IndexPlanBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        // Loop so empty plans are silently skipped (spec: "Plan list is empty
        // → yield no batch for that plan; continue to the next").
        loop {
            self.refill();

            if let Some(head) = self.in_flight.pop_front() {
                // Refill again so the queue stays warm during the upcoming
                // process_plan call. Errors latched here drain through later.
                self.refill();

                if head.plan.is_empty() {
                    // Skip empty plan; loop to pull the next.
                    continue;
                }

                if let Err(e) = self.await_head(head.prefetches) {
                    return Some(Err(e));
                }
                return Some(self.loader.process_plan(head.plan));
            }

            // Queue empty — surface a deferred plan-stream error one-shot,
            // then mark stream done so subsequent next() calls return None.
            if let Some(e) = self.plan_stream_error.take() {
                self.plan_stream_done = true;
                return Some(Err(e));
            }

            return None;
        }
    }
}

impl Drop for IndexPlanIter {
    /// Best-effort shutdown. The loader's tokio runtime is owned by the
    /// `Arc<IndexPlanLoader>` (which the dataset, not this iter, holds), so
    /// dropping the iter does not tear down the runtime — we only need to
    /// release the prefetch tasks and the plan-pull thread.
    fn drop(&mut self) {
        // Abort any outstanding prefetch handles so the runtime threads
        // stop blocking on shards we no longer need.
        for in_flight in self.in_flight.drain(..) {
            for h in in_flight.prefetches {
                h.abort();
            }
        }

        // Drain plan_rx so the plan-pull thread's next send fails fast and
        // the thread exits. We don't `join` — the user iterator could be a
        // slow Python generator and we don't want to block on it.
        while self.plan_rx.try_recv().is_ok() {}

        // Detach the plan thread; it will exit on its next send-fail.
        let _ = self.plan_thread.take();

        // Optional one-shot profile dump. Enabled by `SCX_LOADER_PROFILE=1`
        // — see `crate::budget::profiling_enabled`.
        if profiling_enabled() {
            let cm = &self.loader.cache_metrics;
            let im = &self.iter_metrics;
            let hits = cm.hits.load(Ordering::Relaxed);
            let misses = cm.misses.load(Ordering::Relaxed);
            let evictions = cm.evictions.load(Ordering::Relaxed);
            let bytes_inserted = cm.bytes_inserted.load(Ordering::Relaxed);
            let dup_waiters = cm.duplicate_waiters.load(Ordering::Relaxed);
            let spawned = im.prefetch_tasks_spawned.load(Ordering::Relaxed);
            let skip_hit = im.prefetch_skipped_cache_hit.load(Ordering::Relaxed);
            let skip_inflight = im.prefetch_skipped_in_flight.load(Ordering::Relaxed);
            eprintln!(
                "scx-loader IndexPlanIter cache_metrics: \
                 hits={hits} misses={misses} evictions={evictions} \
                 bytes_inserted={bytes_inserted} duplicate_waiters={dup_waiters} \
                 prefetch_tasks_spawned={spawned} \
                 prefetch_skipped_cache_hit={skip_hit} \
                 prefetch_skipped_in_flight={skip_inflight}"
            );
        }
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
        assert!(
            n_obs % n_shards == 0,
            "n_obs must divide n_shards in this fixture"
        );
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

    fn open_loader(path: &std::path::Path, sort_by_shard: bool) -> IndexPlanLoader {
        let mut config = LoaderConfig::default();
        config.normalize = false;
        config.log1p = false;
        config.obs_columns = vec!["cell_id".to_string()];
        IndexPlanLoader::new(
            path,
            config,
            /*cache_shards*/ 4,
            sort_by_shard,
            /*lookahead*/ 4,
            /*max_plan_size*/ 16384,
        )
        .unwrap()
    }

    fn open_loader_hvg(
        path: &std::path::Path,
        sort_by_shard: bool,
        hvg_indices: Vec<u32>,
    ) -> IndexPlanLoader {
        let mut config = LoaderConfig::default();
        config.normalize = false;
        config.log1p = false;
        config.obs_columns = vec!["cell_id".to_string()];
        config.hvg_indices = Some(hvg_indices);
        IndexPlanLoader::new(
            path,
            config,
            /*cache_shards*/ 4,
            sort_by_shard,
            /*lookahead*/ 4,
            /*max_plan_size*/ 16384,
        )
        .unwrap()
    }

    fn assert_full_fixture_row(row: u64, dense: &[f32], n_vars: usize) {
        let col = (row as usize) % n_vars;
        let val = ((row as usize + 1) & 0xFF) as f32;
        assert_eq!(dense[col], val, "row {row} expected value at col {col}");
        assert_eq!(
            dense.iter().filter(|&&v| v != 0.0).count(),
            1,
            "row {row} should have one nonzero"
        );
    }

    fn assert_hvg_fixture_row(row: u64, dense: &[f32], hvg_indices: &[u32], n_vars: usize) {
        let col = ((row as usize) % n_vars) as u32;
        let expected_pos = hvg_indices.iter().position(|&h| h == col);
        match expected_pos {
            Some(pos) => {
                assert_eq!(
                    dense[pos],
                    ((row as usize + 1) & 0xFF) as f32,
                    "row {row} expected HVG col {col} at projected pos {pos}"
                );
                assert_eq!(
                    dense.iter().filter(|&&v| v != 0.0).count(),
                    1,
                    "row {row} should have one projected nonzero"
                );
            }
            None => assert!(
                dense.iter().all(|&v| v == 0.0),
                "row {row} should project to an all-zero HVG row"
            ),
        }
    }

    fn categorical_strings(col: &ObsColumn) -> Vec<String> {
        match col {
            ObsColumn::Categorical(codes, categories) => codes
                .iter()
                .map(|&code| categories[code as usize].clone())
                .collect(),
            other => panic!("expected categorical obs column, got {other:?}"),
        }
    }

    #[test]
    fn fused_paired_gather_preserves_duplicate_rows_and_same_row_pairs() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
        let loader = open_loader(&path, false);

        let plan: Vec<(u64, u64)> = vec![(1, 5), (2, 5), (5, 5), (9, 1), (1, 9)];
        let batch = loader.process_plan(plan.clone()).unwrap();
        let n_cols = loader.n_output_cols();

        assert_eq!(batch.pairs, plan);
        for (i, &(pert, ctrl)) in batch.pairs.iter().enumerate() {
            let p_row = &batch.x[i * n_cols..][..n_cols];
            let c_row = &batch.x_paired[i * n_cols..][..n_cols];
            assert_full_fixture_row(pert, p_row, n_cols);
            assert_full_fixture_row(ctrl, c_row, n_cols);
            if pert == ctrl {
                let p_bits: Vec<u32> = p_row.iter().map(|v| v.to_bits()).collect();
                let c_bits: Vec<u32> = c_row.iter().map(|v| v.to_bits()).collect();
                assert_eq!(p_bits, c_bits, "same-row pair should produce equal rows");
            }
        }
    }

    #[test]
    fn fused_paired_gather_handles_hvg_projection_and_empty_projected_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
        let hvg = vec![1u32, 5u32];
        let loader = open_loader_hvg(&path, false, hvg.clone());

        let plan: Vec<(u64, u64)> = vec![(5, 1), (9, 5), (2, 10), (13, 5)];
        let batch = loader.process_plan(plan.clone()).unwrap();
        let n_cols = loader.n_output_cols();
        assert_eq!(n_cols, hvg.len());

        for (i, &(pert, ctrl)) in batch.pairs.iter().enumerate() {
            let p_row = &batch.x[i * n_cols..][..n_cols];
            let c_row = &batch.x_paired[i * n_cols..][..n_cols];
            assert_hvg_fixture_row(pert, p_row, &hvg, 8);
            assert_hvg_fixture_row(ctrl, c_row, &hvg, 8);
        }
    }

    #[test]
    fn fused_paired_gather_keeps_obs_aligned_after_shard_sort() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
        let loader = open_loader(&path, true);

        let plan: Vec<(u64, u64)> = vec![(15, 0), (4, 5), (8, 9), (3, 12), (10, 11), (1, 2)];
        let batch = loader.process_plan(plan).unwrap();
        let obs = categorical_strings(batch.obs.get("cell_id").unwrap());
        let obs_paired = categorical_strings(batch.obs_paired.get("cell_id").unwrap());

        assert_eq!(obs.len(), batch.pairs.len());
        assert_eq!(obs_paired.len(), batch.pairs.len());
        for (i, &(pert, ctrl)) in batch.pairs.iter().enumerate() {
            assert_eq!(obs[i], format!("cell_{pert}"));
            assert_eq!(obs_paired[i], format!("cell_{ctrl}"));
        }
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
            assert!(
                w[0] <= w[1],
                "post-sort plan must be non-decreasing in min-shard"
            );
        }
    }

    /// Phase 2.4: parity invariant — for the same input plan, sorted vs
    /// unsorted runs produce the same `(pair, x_row, x_paired_row)` *set*
    /// (just permuted). HVG-on and HVG-off both verified.
    #[test]
    fn sort_by_shard_is_a_pure_permutation_of_unsorted_output() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);

        let plan: Vec<(u64, u64)> = vec![(15, 0), (4, 5), (8, 9), (3, 12), (10, 11), (1, 2)];

        let unsorted = open_loader(&path, false)
            .process_plan(plan.clone())
            .unwrap();
        let sorted = open_loader(&path, true).process_plan(plan.clone()).unwrap();

        // Unsorted preserves input plan order (sanity).
        assert_eq!(unsorted.pairs, plan);

        // Build (pair, row, paired_row) tuples and compare as multisets.
        let n_cols = unsorted.x.len() / unsorted.pairs.len();
        let triples = |b: &IndexPlanBatch| -> Vec<((u64, u64), Vec<u32>, Vec<u32>)> {
            (0..b.pairs.len())
                .map(|i| {
                    let r: Vec<u32> = b.x[i * n_cols..][..n_cols]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect();
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

    // ---------------------------------------------------------------------
    // Phase 4 — iter_with_plans
    // ---------------------------------------------------------------------

    /// Helper: collect all plans from a Vec into the iterator-of-Result form
    /// that `iter_with_plans` expects.
    fn into_plan_iter(
        plans: Vec<Vec<(u64, u64)>>,
    ) -> impl Iterator<Item = std::result::Result<Vec<(u64, u64)>, LoaderError>> + Send + 'static
    {
        plans.into_iter().map(Ok)
    }

    fn open_loader_arc(path: &std::path::Path) -> Arc<IndexPlanLoader> {
        let mut config = LoaderConfig::default();
        config.normalize = false;
        config.log1p = false;
        config.obs_columns = vec!["cell_id".to_string()];
        Arc::new(
            IndexPlanLoader::new(
                path, config, /*cache_shards*/ 4, /*sort_by_shard*/ true,
                /*lookahead*/ 4, /*max_plan_size*/ 16384,
            )
            .unwrap(),
        )
    }

    /// Iterator yields one batch per plan, batches are correctly aligned.
    #[test]
    fn iter_with_plans_yields_one_batch_per_plan() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
        let loader = open_loader_arc(&path);

        let plans = vec![
            vec![(0u64, 1u64), (2, 3)],
            vec![(4, 5)],
            vec![(8, 12), (10, 15)],
        ];
        let it = loader.iter_with_plans(into_plan_iter(plans.clone()), 4);
        let batches: Vec<_> = it.map(|r| r.unwrap()).collect();

        assert_eq!(batches.len(), 3);
        // Each batch has the expected number of pairs (post-sort, but the
        // batch contents are a permutation of the plan).
        for (i, b) in batches.iter().enumerate() {
            assert_eq!(b.pairs.len(), plans[i].len());
            // Build sorted multisets for set equality.
            let mut got = b.pairs.clone();
            got.sort();
            let mut want = plans[i].clone();
            want.sort();
            assert_eq!(got, want);
        }
    }

    /// Lookahead = 0 vs lookahead = 4 must produce identical outputs (up to
    /// the existing sort_by_shard semantics).
    #[test]
    fn iter_with_plans_lookahead_zero_vs_four_parity() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
        let plans = vec![
            vec![(15u64, 0u64), (4, 5), (8, 9)],
            vec![(3, 12), (10, 11), (1, 2)],
        ];

        let it_zero = open_loader_arc(&path).iter_with_plans(into_plan_iter(plans.clone()), 0);
        let it_four = open_loader_arc(&path).iter_with_plans(into_plan_iter(plans.clone()), 4);

        let zero: Vec<_> = it_zero.map(|r| r.unwrap()).collect();
        let four: Vec<_> = it_four.map(|r| r.unwrap()).collect();

        assert_eq!(zero.len(), four.len());
        for (a, b) in zero.iter().zip(four.iter()) {
            assert_eq!(
                a.pairs, b.pairs,
                "pair order should match between lookahead=0 and 4"
            );
            assert_eq!(a.x, b.x);
            assert_eq!(a.x_paired, b.x_paired);
        }
    }

    /// Errors injected into the plan stream propagate as Err items in order.
    #[test]
    fn iter_with_plans_propagates_plan_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 8, 4, 2);
        let loader = open_loader_arc(&path);

        let plans: Vec<std::result::Result<Vec<(u64, u64)>, LoaderError>> = vec![
            Ok(vec![(0, 1)]),
            Err(LoaderError::ChannelError("synthetic".into())),
            Ok(vec![(2, 3)]),
        ];
        let mut it = loader.iter_with_plans(plans.into_iter(), 2);

        // First a successful batch, then the error, then iteration stops
        // (sticky `plan_stream_error`).
        assert!(matches!(it.next(), Some(Ok(_))));
        let second = it.next().expect("second item");
        match second {
            Err(LoaderError::ChannelError(s)) => assert!(s.contains("synthetic")),
            Err(other) => panic!("expected ChannelError, got {other}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
        assert!(
            it.next().is_none(),
            "iteration must stop after a plan-stream error"
        );
    }

    /// Out-of-range row in a plan yields a Result::Err(IndexOutOfRange) on
    /// that batch and stops iteration. (Validation is per-batch, not in the
    /// pull thread.)
    #[test]
    fn iter_with_plans_propagates_decode_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 8, 4, 2);
        let loader = open_loader_arc(&path);

        let plans = vec![vec![(0u64, 1u64)], vec![(99, 99)]];
        let it = loader.iter_with_plans(into_plan_iter(plans), 2);
        let mut results = it;

        let first = results.next().expect("first batch");
        assert!(first.is_ok());
        let second = results.next().expect("second batch");
        match second {
            Err(LoaderError::IndexOutOfRange { idx, .. }) => assert_eq!(idx, 99),
            Err(other) => panic!("expected IndexOutOfRange, got LoaderError: {other}"),
            Ok(_) => panic!("expected IndexOutOfRange, got Ok"),
        }
    }

    /// Drop mid-iteration must not deadlock. Build an iterator with a long
    /// plan stream, take 1 batch, then drop.
    #[test]
    fn iter_with_plans_drop_mid_iteration() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
        let loader = open_loader_arc(&path);

        let plans: Vec<_> = (0..1000).map(|_| vec![(0u64, 1u64)]).collect();
        let mut it = loader.iter_with_plans(into_plan_iter(plans), 4);
        let _first = it.next().unwrap().unwrap();
        drop(it); // must not hang
    }

    // ---------------------------------------------------------------------
    // Phase 5 — memory budget + auto-tuning
    // ---------------------------------------------------------------------

    /// Build a multi-shard fixture with a tunable nnz_per_row, so the
    /// memory-budget tests can dial in the relative weight of the LRU cache.
    fn write_dense_fixture(
        path: &std::path::Path,
        n_obs: usize,
        n_vars: usize,
        n_shards: usize,
        nnz_per_row: usize,
    ) -> std::path::PathBuf {
        assert!(n_obs % n_shards == 0);
        assert!(nnz_per_row <= n_vars);
        let rows_per_shard = n_obs / n_shards;
        let total_nnz = (n_obs * nnz_per_row) as u64;

        let header = FileHeader {
            magic: MAGIC,
            format_version: 1,
            header_length: 256,
            flags: 0,
            n_obs: n_obs as u64,
            n_vars: n_vars as u64,
            nnz: total_nnz,
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
                for k in 0..nnz_per_row {
                    let col = ((row + k * 7919) % n_vars) as u32;
                    indices.push(col);
                    values.push(((row + k + 1) & 0xFF) as u8);
                }
                indptr.push(*indptr.last().unwrap() + nnz_per_row as u64);
            }
            // Indices must be sorted within each row for the CSR format;
            // sort each row's slice.
            for local in 0..rows_per_shard {
                let lo = indptr[local] as usize;
                let hi = indptr[local + 1] as usize;
                let mut pairs: Vec<(u32, u8)> = (lo..hi).map(|j| (indices[j], values[j])).collect();
                pairs.sort_by_key(|&(c, _)| c);
                pairs.dedup_by_key(|&mut (c, _)| c);
                let new_lo = lo;
                for (j, (c, v)) in pairs.iter().enumerate() {
                    indices[new_lo + j] = *c;
                    values[new_lo + j] = *v;
                }
                // dedup may shorten — fix the indptr accordingly by rebuilding
                // (rare; skip for simplicity if no dups).
                let _ = (new_lo,);
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

    /// Generous memory budget — both effective values match the requested.
    #[test]
    fn budget_generous_no_autotune() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 32, 8, 4);
        let mut config = LoaderConfig::default();
        config.max_memory_mb = 4096; // way over budget needed for this tiny file
        let loader = IndexPlanLoader::new(
            &path, config, /*cache_shards*/ 8, /*sort_by_shard*/ true,
            /*lookahead*/ 4, /*max_plan_size*/ 1024,
        )
        .unwrap();
        assert_eq!(loader.effective_cache_shards(), 8);
        assert_eq!(loader.effective_lookahead(), 4);
    }

    /// Tight budget — auto-tune kicks in, lookahead reduced first.
    /// Sized so the lookahead overhead dominates the over-budget margin
    /// (max_plan_size=65536 → 1 MB per lookahead unit).
    #[test]
    fn budget_tight_reduces_lookahead_first() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 256, 8, 4);

        let mut config = LoaderConfig::default();
        // Budget components at requested settings (post-issue-#6 model):
        //   python       = 50 MB
        //   batch buffer = 2 × 65536 × 8 × 4         ≈ 4 MB
        //   lookahead    = 8 × 65536 × 16            ≈ 8 MB
        //   transient    = 2 × 65536 × 24 (no obs)   ≈ 3 MB  (PairRequest)
        //   shard cache  = ~negligible (sparse fixture)
        // Total ≈ 65 MB. Floor at lookahead=1: 50+4+1+3 ≈ 58 MB.
        // A 60 MB budget forces lookahead reduction without floor-failure.
        config.max_memory_mb = 60;
        let loader = IndexPlanLoader::new(
            &path, config, /*cache_shards*/ 8, /*sort_by_shard*/ true,
            /*lookahead*/ 8, /*max_plan_size*/ 65536,
        )
        .unwrap();
        assert!(
            loader.effective_lookahead() < 8,
            "lookahead should be reduced under tight budget; got {}",
            loader.effective_lookahead()
        );
        assert!(
            loader.effective_lookahead() >= 1,
            "lookahead floor is 1; got {}",
            loader.effective_lookahead()
        );
        // Cache shards should NOT have been touched yet.
        assert_eq!(loader.effective_cache_shards(), 8);
    }

    /// Even tighter budget — lookahead at the floor (1), cache_shards reduced
    /// further. Verifies the "reduce cache_shards next" branch using a dense
    /// fixture so the LRU shard cache has meaningful weight.
    #[test]
    fn budget_very_tight_reduces_cache_shards() {
        let dir = tempfile::tempdir().unwrap();
        // 1024 rows × 64 vars × 8 shards × 32 nnz/row.
        // shard_decoded ≈ (32 × 128 × 8) + (128 × 8) ≈ 33 KB per shard.
        // 16 cache shards ≈ 528 KB.
        //
        // Actually for a meaningful cache contribution we need much higher
        // density. Bump nnz_per_row.
        let path = write_dense_fixture(&dir.path().join("f.scx"), 1024, 4096, 8, 2048);
        // shard_decoded ≈ (2048 × 128 × 8) + (128 × 8) ≈ 2.1 MB per shard.
        // 16 cache shards ≈ 33 MB.
        //
        // Budget at requested settings:
        //   python       = 50 MB
        //   batch buffer = 2 × 1024 × 4096 × 4 ≈ 32 MB
        //   lookahead    = 4 × 1024 × 16       ≈ 64 KB
        //   shard cache  = 16 × 2.1 MB         ≈ 33 MB
        // Total ≈ 115 MB. Budget 96 forces lookahead → 1, then cache_shards.

        let mut config = LoaderConfig::default();
        config.max_memory_mb = 96;
        let loader = IndexPlanLoader::new(
            &path, config, /*cache_shards*/ 16, /*sort_by_shard*/ true,
            /*lookahead*/ 4, /*max_plan_size*/ 1024,
        )
        .unwrap();
        assert_eq!(
            loader.effective_lookahead(),
            1,
            "lookahead should be at the floor (got {})",
            loader.effective_lookahead()
        );
        assert!(
            loader.effective_cache_shards() < 16,
            "cache_shards should also be reduced; got {}",
            loader.effective_cache_shards()
        );
        assert!(
            loader.effective_cache_shards() >= 1,
            "cache_shards floor is 1; got {}",
            loader.effective_cache_shards()
        );
    }

    /// Budget below the floor — construction must fail with a clear ConfigError.
    #[test]
    fn budget_below_floor_refuses_construction() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 4096, 4096, 4);

        let mut config = LoaderConfig::default();
        config.max_memory_mb = 40; // below the 50 MB python overhead alone
        let result = IndexPlanLoader::new(
            &path, config, /*cache_shards*/ 4, /*sort_by_shard*/ true,
            /*lookahead*/ 2, /*max_plan_size*/ 4096,
        );
        let err = match result {
            Ok(_) => panic!("expected ConfigError, got Ok"),
            Err(e) => e,
        };
        match err {
            LoaderError::ConfigError { reason } => {
                assert!(
                    reason.contains("max_memory_mb=40"),
                    "error should mention requested budget: {reason}"
                );
                assert!(
                    reason.contains("Increase max_memory_mb"),
                    "error should suggest the fix: {reason}"
                );
            }
            other => panic!("expected ConfigError, got {other}"),
        }
    }

    /// Caller explicitly chooses lookahead=0 — honored when it fits the
    /// budget (no prefetch path activated).
    #[test]
    fn budget_lookahead_zero_honored_when_fits() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 32, 8, 4);
        let mut config = LoaderConfig::default();
        config.max_memory_mb = 256;
        let loader = IndexPlanLoader::new(
            &path, config, /*cache_shards*/ 4, /*sort_by_shard*/ true,
            /*lookahead*/ 0, /*max_plan_size*/ 1024,
        )
        .unwrap();
        assert_eq!(loader.effective_lookahead(), 0);
    }

    /// `process_plan` must reject plans larger than `max_plan_size` so a
    /// misbehaving consumer cannot silently exceed the memory budget.
    /// Plans at or below the ceiling are accepted as before.
    #[test]
    fn process_plan_rejects_oversize_plan() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 32, 8, 4);
        let mut config = LoaderConfig::default();
        config.normalize = false;
        config.log1p = false;
        config.obs_columns = vec!["cell_id".to_string()];

        let loader = IndexPlanLoader::new(
            &path, config, /*cache_shards*/ 4, /*sort_by_shard*/ false,
            /*lookahead*/ 1, /*max_plan_size*/ 4,
        )
        .unwrap();

        // At-the-ceiling plan succeeds.
        let ok_plan = vec![(0u64, 1u64), (2, 3), (4, 5), (6, 7)];
        let batch = loader.process_plan(ok_plan).unwrap();
        assert_eq!(batch.pairs.len(), 4);

        // Over-the-ceiling plan rejects with a ConfigError naming both numbers.
        let big_plan: Vec<(u64, u64)> = (0..5u64).map(|i| (i, (i + 1) % 32)).collect();
        match loader.process_plan(big_plan) {
            Err(LoaderError::ConfigError { reason }) => {
                assert!(reason.contains("plan size 5"), "got: {reason}");
                assert!(reason.contains("max_plan_size 4"), "got: {reason}");
            }
            Err(other) => panic!("expected ConfigError, got {other}"),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }
}
