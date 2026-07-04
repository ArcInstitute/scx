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

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;

use arrow::record_batch::RecordBatch;
use crossbeam_channel::{bounded, Receiver};
use scx_format_io::{BackedCsrReader, ScxReader};
use scx_sparse::ScxCsr;
use tokio::runtime::Runtime;
use tokio::task::JoinHandle;

use crate::batch::ObsColumn;
use crate::budget::{profiling_enabled, BudgetBreakdown, PYTHON_OVERHEAD_BYTES};
use crate::decode_stage::{build_category_dicts, extract_obs_columns, CategoryDict};
use crate::error::{LoaderError, Result};
use crate::normalize::apply_dense_transforms;
use crate::pipeline::LoaderConfig;
use crate::projection::{pflog1ppf_row_full, scatter_row_full, HvgProjection};

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
///
/// # Fork safety
///
/// The tokio runtime is built lazily on first use (see [`Self::runtime`]),
/// not in [`Self::new`]. This matches the lazy-init pattern in
/// `TrainingPipeline` and is the contract that lets a parent process
/// construct an `IndexPlanLoader` and then have a PyTorch `DataLoader`
/// fork worker processes: the runtime threads never exist in the parent
/// at fork time, so the child does not inherit a wedged thread pool.
pub struct IndexPlanLoader {
    backed: BackedCsrReader,
    obs_metadata: RecordBatch,
    /// Stable global category dictionaries (one per categorical obs column),
    /// computed once at construction. Built from the same full obs table as
    /// `TrainingPipeline`, so categorical codes match across the two paths.
    cat_dicts: HashMap<String, CategoryDict>,
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
    ///
    /// Lazily built by [`Self::runtime`] on first use so the parent process
    /// never holds tokio I/O threads that would be inherited across `fork(2)`.
    runtime: OnceLock<Runtime>,
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
    cache_metrics: Arc<scx_format_io::CacheMetrics>,
    /// Per-component memory breakdown produced by the auto-tune at
    /// construction. Surfaced through `IndexPlanDataset.memory_budget()` for
    /// production sizing.
    budget_breakdown: BudgetBreakdown,
    /// Per-dataset escape hatch gating the **L2 sidecar-aware prefetch skip**
    /// only (default `true`). When `false`, the prefetch warms cold shards as
    /// before, so the gather falls back to full-shard decode — legacy
    /// behaviour. Because it gates *only* the prefetch skip, it has no effect at
    /// `lookahead == 0`: with no prefetch, the L1 gather still reaches the
    /// sidecar regardless of this flag. L1 (`gather_pairs_dense` →
    /// `read_rows_with`) is unconditional; the process-wide
    /// `SCX_SCATTER_SIDECAR=0` env switch disables the sidecar at the reader
    /// layer entirely.
    scatter_sidecar: bool,
    /// Per-dataset escape hatch for the codec-agnostic row-group block-index
    /// path (default `true`). Independent of `scatter_sidecar` — a framed file
    /// adopts the block-index path even with the Scx1 sidecar off. Unlike
    /// `scatter_sidecar` (whose loader flag gates only the prefetch skip),
    /// `set_scatter_block_index` propagates this to the backed reader, so
    /// `False` disables **both** the L2 prefetch skip **and** L1
    /// (`read_rows_with`) block-index adoption — a clean off-switch that
    /// full-shard-decodes. The process-wide reader default still comes from
    /// `SCX_SCATTER_BLOCK_INDEX`.
    scatter_block_index: bool,
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
        scatter_sidecar: bool,
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

        // Stable global category dictionaries, built once over the full obs
        // table so codes are identical across pulls and match TrainingPipeline.
        // Also validates that requested columns exist.
        let cat_dicts = build_category_dicts(&obs_metadata, &config.obs_columns)?;

        // Validate obs column dtypes up front — fail at construction, not on
        // first batch (the builder validates presence + string/dict layout;
        // this also catches unsupported numeric/other dtypes via a row probe).
        if !config.obs_columns.is_empty() && n_obs > 0 {
            extract_obs_columns(&obs_metadata, &[0u64], &config.obs_columns, &cat_dicts)?;
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
                                            // `gather_pairs_dense` deduplicates the 2 × max_plan_size (pert+ctrl)
                                            // rows into `unique_rows: Vec<u64>`, a `row_to_requests:
                                            // Vec<Vec<PairRequest>>` fan-out map, and a `row_to_pos:
                                            // HashMap<u64, usize>`. Worst case (all rows distinct) holds
                                            // 2 × max_plan_size entries across these structures. Use `size_of`
                                            // for each so the term tracks struct churn automatically instead of
                                            // drifting against a hand-derived constant. (HashMap load-factor
                                            // slack is ignored — it's a small, transient term dominated by the
                                            // shard cache and dense batch buffers.)
        const GATHER_SCRATCH_BYTES_PER_ROW: usize = std::mem::size_of::<u64>()           // unique_rows entry
            + std::mem::size_of::<Vec<PairRequest>>()                          // row_to_requests outer Vec header
            + std::mem::size_of::<PairRequest>()                               // one PairRequest (inner, all-distinct)
            + std::mem::size_of::<(u64, usize)>(); // row_to_pos entry
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
                .saturating_mul(GATHER_SCRATCH_BYTES_PER_ROW);
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

        // The tokio runtime is built lazily on first use (see `Self::runtime`)
        // to keep `new` fork-safe — a parent process can construct an
        // `IndexPlanLoader` and then have a `DataLoader` fork worker
        // processes without inheriting a wedged thread pool.
        Ok(Self {
            backed,
            obs_metadata,
            cat_dicts,
            config,
            hvg_projection,
            n_output_cols,
            sort_by_shard,
            runtime: OnceLock::new(),
            effective_cache_shards,
            effective_lookahead,
            max_plan_size,
            cache_metrics,
            budget_breakdown,
            scatter_sidecar,
            // Default on; the Python layer overrides via
            // `set_scatter_block_index` when the caller passes the kwarg.
            scatter_block_index: true,
        })
    }

    /// Override the block-index adoption gate (default `true`). Called by the
    /// Python constructor to honor its `scatter_block_index` kwarg; kept as a
    /// post-construction setter so the many-arg `new` signature (and its test
    /// callers) stays unchanged. Propagates to the backed reader so the off-switch
    /// disables **both** the L1 gather adoption and the L2 prefetch skip — a
    /// `scatter_block_index=False` file full-shard-decodes, no leak via L1. See
    /// [`Self::scatter_block_index`].
    pub fn set_scatter_block_index(&mut self, enabled: bool) {
        self.scatter_block_index = enabled;
        self.backed.set_scatter_block_index(enabled);
    }

    /// Return the tokio runtime, building it on first call.
    ///
    /// Lazy construction is the fork-safety contract: the parent process
    /// must not own a multi-threaded tokio runtime at the moment a child
    /// is forked, so `IndexPlanLoader::new` does not build one. The first
    /// call here (always from an `IndexPlanIter`, always post-fork in the
    /// `DataLoader` worker) materializes the runtime; later calls return
    /// the same instance via `OnceLock`.
    ///
    /// Note on the build-then-`get_or_init` pattern: `OnceLock::get_or_try_init`
    /// (which would let us build *inside* the init closure with fallibility)
    /// is still unstable as of Rust 1.94. The stable build-then-set pattern
    /// below is race-tolerant — at most one constructed runtime wins
    /// `get_or_init` and the rest are dropped — at the cost of one extra
    /// runtime construction in the rare concurrent first-touch case.
    fn runtime(&self) -> Result<&Runtime> {
        if let Some(rt) = self.runtime.get() {
            return Ok(rt);
        }
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("scx-index-plan")
            .build()
            .map_err(|e| {
                LoaderError::ShutdownError(format!(
                    "failed to create tokio runtime for IndexPlanLoader: {e}"
                ))
            })?;
        Ok(self.runtime.get_or_init(|| rt))
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
    pub fn cache_metrics(&self) -> Arc<scx_format_io::CacheMetrics> {
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

    /// Whether the L2 sidecar-aware prefetch skip is enabled for this dataset
    /// (default `true`). Gates only the prefetch skip in `IndexPlanIter`; L1 is
    /// unconditional. See [`Self::scatter_sidecar`] field docs.
    pub fn scatter_sidecar(&self) -> bool {
        self.scatter_sidecar
    }

    /// Whether the L2 block-index-aware prefetch skip is enabled for this dataset
    /// (default `true`). Gates only the prefetch skip in `IndexPlanIter`; L1
    /// block-index adoption is governed by the reader's own env-derived gate. See
    /// [`Self::scatter_block_index`] field docs.
    pub fn scatter_block_index(&self) -> bool {
        self.scatter_block_index
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
            &self.cat_dicts,
        )?;
        let obs_paired = extract_obs_columns(
            &self.obs_metadata,
            &gathered.ctrl_indices,
            &self.config.obs_columns,
            &self.cat_dicts,
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

        // Deduplicate the (pert, ctrl) rows into a single set of distinct rows
        // plus a fan-out map (unique-row → every PairRequest referencing it).
        // The gather then runs through `BackedCsrReader::read_rows_with`, which
        // sorts internally and — per shard — decodes either O(rows) via the
        // scx1 decode sidecar (cold, sparse-per-shard groups) or the full shard
        // (cached / dense / no sidecar). Output is byte-identical to the old
        // full-shard-decode-and-slice path because each PairRequest still
        // carries its `pair_idx` and writes the same output slot; only the
        // decode strategy changes (L1 sidecar gather).
        let mut row_to_pos: HashMap<u64, usize> = HashMap::with_capacity(n_pairs * 2);
        let mut unique_rows: Vec<u64> = Vec::with_capacity(n_pairs * 2);
        let mut row_to_requests: Vec<Vec<PairRequest>> = Vec::with_capacity(n_pairs * 2);
        for (pair_idx, &(pert, ctrl)) in plan.iter().enumerate() {
            pert_indices.push(pert);
            ctrl_indices.push(ctrl);
            for (row, side) in [(pert, PairSide::Perturbed), (ctrl, PairSide::Control)] {
                let pos = *row_to_pos.entry(row).or_insert_with(|| {
                    unique_rows.push(row);
                    row_to_requests.push(Vec::new());
                    unique_rows.len() - 1
                });
                row_to_requests[pos].push(PairRequest { pair_idx, side });
            }
        }

        // `scatter_pair_request` returns the loader's `Result` (LoaderError),
        // but `read_rows_with`'s closure must return `scx_format_io::Result`
        // (ScxError). Stash any scatter error in a slot and abort iteration with
        // a sentinel ScxError, then surface the original LoaderError afterward —
        // preserving the precise error variant rather than stringifying it.
        let mut scatter_err: Option<LoaderError> = None;
        let res = self
            .backed
            .read_rows_with(&unique_rows, |orig_pos, idx, data| {
                for &request in &row_to_requests[orig_pos] {
                    if let Err(e) =
                        self.scatter_pair_request(request, idx, data, n_cols, &mut x, &mut x_paired)
                    {
                        // Stash the real LoaderError and abort iteration with a
                        // sentinel ScxError (the closure must return ScxError).
                        scatter_err = Some(e);
                        return Err(scx_format_io::ScxError::InconsistentCsr);
                    }
                }
                Ok(())
            });
        // Invariant: the closure returns Err ONLY after setting `scatter_err`,
        // so a Some here is always the original scatter error — check it before
        // `res` so the precise LoaderError wins over the sentinel.
        if let Some(e) = scatter_err {
            return Err(e);
        }
        res?;

        // PFlog1pPF is applied at scatter time (it needs the full pre-projection
        // row for depth/baseline — see `scatter_pair_request`), so the
        // post-scatter normalize/log1p dispatch is skipped in that mode.
        if !self.config.pflog1ppf {
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
        if self.config.pflog1ppf {
            // Depth/D over the FULL transcriptome (`idx`/`data` is the full row);
            // each output slot is written by exactly one request, so the scatter
            // fully produces the final PFlog1pPF row (delta + baseline).
            let n_vars_full = self.backed.n_vars();
            let c = self.config.pflog1ppf_c;
            match self.hvg_projection.as_ref() {
                Some(hvg) => hvg.scatter_pflog1ppf_row(idx, data, c, n_vars_full, out),
                None => pflog1ppf_row_full(idx, data, c, n_vars_full, out)?,
            }
        } else {
            match self.hvg_projection.as_ref() {
                Some(hvg) => hvg.scatter_row(idx, data, out),
                None => scatter_row_full(idx, data, out)?,
            }
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
type ShardJoin = JoinHandle<scx_format_io::Result<Arc<ScxCsr>>>;

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
    /// Shards whose prefetch was skipped because the group is **sidecar-eligible**
    /// (cold + sparse): the dense gather's `read_rows_with` decodes the touched
    /// rows O(rows) via the scx1 decode sidecar, so warming the whole shard would
    /// negate the win (the L2 sidecar-aware prefetch skip).
    pub prefetch_skipped_sidecar: AtomicU64,
    /// Shards whose prefetch was skipped because the group is **block-index
    /// eligible** (cold + sparse + row-group framed): the dense gather decodes
    /// only the touched row-groups via the block index, so warming the whole
    /// shard would negate the win. Independent of the Scx1-sidecar decision (the
    /// L2 block-index-aware prefetch skip).
    pub prefetch_skipped_block_index: AtomicU64,
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
                Ok(Ok(plan)) => match self.spawn_prefetches(&plan) {
                    Ok(prefetches) => {
                        self.in_flight.push_back(InFlight { plan, prefetches });
                    }
                    Err(e) => {
                        // Tokio runtime construction failed (rare — only on
                        // OS thread-creation exhaustion). Latch the error
                        // through the existing plan-stream-error path so it
                        // surfaces after the in-flight queue drains.
                        self.plan_stream_error = Some(e);
                        break;
                    }
                },
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

    fn spawn_prefetches(&self, plan: &[(u64, u64)]) -> Result<Vec<ShardJoin>> {
        if self.lookahead == 0 || plan.is_empty() {
            return Ok(Vec::new());
        }

        // Deduplicate the (pert, ctrl) rows and count unique rows per shard. The
        // gather (`gather_pairs_dense`) passes the SAME deduped set to
        // `read_rows_with`, so this per-shard `group_len` matches what the
        // gather's sidecar decision sees — the prefetch skip and the gather
        // choice must agree (else the adoption the metric proves diverges).
        let index = self.loader.backed.index();
        let mut seen: HashSet<u64> = HashSet::with_capacity(plan.len() * 2);
        let mut per_shard: HashMap<usize, usize> = HashMap::new();
        for &(p, c) in plan {
            for row in [p, c] {
                if seen.insert(row) {
                    if let Some(sidx) = index.shard_for_row(row) {
                        *per_shard.entry(sidx).or_insert(0) += 1;
                    }
                }
            }
        }

        // Warm only shards NOT served by the sidecar. Skips:
        //  - already cached / in-flight (the singleflight already covers them);
        //  - **sidecar-eligible** cold sparse groups — leaving them undecoded is
        //    what lets `read_rows_with` take the O(rows) sidecar path (L2). The
        //    skip predicate is the shared `sidecar_eligible`, so it can never
        //    drift from the gather's `use_sidecar`.
        // Dense/large groups (and sidecar-less shards) still prefetch and warm
        // the cache as before.
        let handle = self.loader.runtime()?.handle().clone();
        Ok(per_shard
            .into_iter()
            .filter(|&(sidx, group_len)| {
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
                if self.loader.scatter_sidecar()
                    && self.loader.backed.sidecar_eligible(sidx, group_len)
                {
                    self.iter_metrics
                        .prefetch_skipped_sidecar
                        .fetch_add(1, Ordering::Relaxed);
                    return false;
                }
                // Block-index-eligible framed shards: leave them undecoded so
                // `read_rows_with` takes the group-level block-index path. Gated
                // by the loader's `scatter_block_index` flag and independent of
                // the Scx1-sidecar decision, so a framed file adopts the path
                // even with the sidecar disabled.
                if self.loader.scatter_block_index()
                    && self.loader.backed.block_index_eligible(sidx, group_len)
                {
                    self.iter_metrics
                        .prefetch_skipped_block_index
                        .fetch_add(1, Ordering::Relaxed);
                    return false;
                }
                true
            })
            .map(|(sidx, _group_len)| {
                self.iter_metrics
                    .prefetch_tasks_spawned
                    .fetch_add(1, Ordering::Relaxed);
                let loader = Arc::clone(&self.loader);
                handle.spawn_blocking(move || loader.backed.read_shard_cached_arc(sidx))
            })
            .collect())
    }

    /// Block on every prefetch handle for the head plan. Surfaces the first
    /// shard read error or join panic.
    fn await_head(&self, prefetches: Vec<ShardJoin>) -> std::result::Result<(), LoaderError> {
        let runtime = self.loader.runtime()?;
        for h in prefetches {
            match runtime.block_on(h) {
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
            let skip_sidecar = im.prefetch_skipped_sidecar.load(Ordering::Relaxed);
            let skip_block_index = im.prefetch_skipped_block_index.load(Ordering::Relaxed);
            eprintln!(
                "scx-loader IndexPlanIter cache_metrics: \
                 hits={hits} misses={misses} evictions={evictions} \
                 bytes_inserted={bytes_inserted} duplicate_waiters={dup_waiters} \
                 prefetch_tasks_spawned={spawned} \
                 prefetch_skipped_cache_hit={skip_hit} \
                 prefetch_skipped_in_flight={skip_inflight} \
                 prefetch_skipped_sidecar={skip_sidecar} \
                 prefetch_skipped_block_index={skip_block_index}"
            );
        }
    }
}

#[cfg(test)]
#[path = "index_plan_tests.rs"]
mod tests;
