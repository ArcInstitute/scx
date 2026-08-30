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

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use arrow::record_batch::RecordBatch;
use scx_format_io::{BackedCsrReader, ScxReader};

use crate::batch::ObsColumn;
use crate::budget::{BudgetBreakdown, BudgetModel, PYTHON_OVERHEAD_BYTES};
use crate::decode_stage::{build_category_dicts, extract_obs_columns, CategoryDict};
use crate::error::{LoaderError, Result};
use crate::normalize::apply_dense_transforms;
use crate::pipeline::LoaderConfig;
use crate::plan_engine::{IterMetrics, PrefetchEngine};
use crate::projection::{pflog_row_full, scatter_row_full, HvgProjection};

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
/// The tokio runtime is built lazily on first use — it belongs to the
/// [`PrefetchEngine`], which this loader itself only builds on the first
/// [`Self::iter_with_plans`] call — never in [`Self::new`]. This matches the lazy-init pattern in
/// `TrainingPipeline` and is the contract that lets a parent process
/// construct an `IndexPlanLoader` and then have a PyTorch `DataLoader`
/// fork worker processes: the runtime threads never exist in the parent
/// at fork time, so the child does not inherit a wedged thread pool.
pub struct IndexPlanLoader {
    /// `Arc` so prefetch tasks can capture the reader alone — see
    /// [`crate::runtime`] for why a task must never hold the loader.
    backed: Arc<BackedCsrReader>,
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
    /// The one-reader [`PrefetchEngine`] this loader's iterator runs on. It
    /// owns the plan-pull thread, the lookahead queue, the prefetch runtime and
    /// the `IterMetrics` counters — everything this module used to fork into
    /// its own copy (ORG-9.10-1).
    ///
    /// Built on the **first** [`Self::iter_with_plans`] call, not in `new`, for
    /// two reasons. It preserves the fork-safety contract: the engine's tokio
    /// runtime is itself lazy, so a parent that constructs a loader and then
    /// forks `DataLoader` workers never holds inherited I/O threads. And it
    /// keeps [`Self::set_scatter_block_index`] working — that setter reaches
    /// the reader through `Arc::get_mut`, which an engine built in `new` would
    /// defeat by holding a second `Arc` to it.
    engine: OnceLock<Arc<PrefetchEngine>>,
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
    /// `Some` iff the auto-tune shrank the shard cache below the requested
    /// `cache_shards`. Read once by `IndexPlanDataset::new` to emit the
    /// caller-facing `UserWarning`; see [`crate::budget::assess_cache_sizing`].
    cache_sizing: Option<crate::budget::CacheSizingVerdict>,
    /// The `cache_shards` the caller asked for, pre-auto-tune. Compared against
    /// `effective_cache_shards` to tell a byte-driven reduction from none.
    requested_cache_shards: usize,
    /// Average decoded bytes per CSR shard — lets a diagnostic name a concrete
    /// `max_memory_mb` instead of saying "raise it".
    shard_decoded_bytes: usize,
    /// Per-dataset escape hatch for the codec-agnostic row-group block-index
    /// path (default `true`). `set_scatter_block_index` propagates this to the
    /// backed reader, so `False` disables **both** the L2 prefetch skip **and**
    /// L1 (`read_rows_with`) block-index adoption — a clean off-switch that
    /// full-shard-decodes. The process-wide reader default still comes from
    /// `SCX_SCATTER_BLOCK_INDEX`.
    scatter_block_index: bool,
}

/// Knobs the plan-driven auto-tune reduces, in reduction order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexPlanParams {
    pub cache_shards: usize,
    pub lookahead: usize,
}

/// The `IndexPlanLoader` arm of [`crate::budget::BudgetModel`].
///
/// Holds only numbers, so its monotonicity is testable with no file and no
/// fixture (`crate::budget::assert_monotone_reduction_chain`).
///
/// `lookahead` gives before `cache_shards` so the iterator stays functional
/// under tight memory, and both floor at 1 rather than 0 for the same reason.
/// An explicit `lookahead == 0` (prefetch disabled) is preserved: `reduce`
/// never decrements below 1, so it cannot walk a deliberate 0 upward or
/// downward.
pub(crate) struct IndexPlanBudgetModel {
    shard_decoded_bytes: usize,
    batch_buffer_bytes: usize,
    transient_bytes: usize,
    max_plan_size: usize,
}

impl IndexPlanBudgetModel {
    /// `sizeof((u64, u64))` — one plan tuple staged per lookahead slot.
    const PLAN_TUPLE_BYTES: usize = 16;
}

impl BudgetModel for IndexPlanBudgetModel {
    type Params = IndexPlanParams;

    fn estimate(&self, p: IndexPlanParams) -> BudgetBreakdown {
        BudgetBreakdown::new(
            p.cache_shards.saturating_mul(self.shard_decoded_bytes),
            self.batch_buffer_bytes,
            p.lookahead
                .saturating_mul(self.max_plan_size)
                .saturating_mul(Self::PLAN_TUPLE_BYTES),
            self.transient_bytes,
            PYTHON_OVERHEAD_BYTES,
        )
    }

    fn reduce(&self, p: IndexPlanParams) -> Option<IndexPlanParams> {
        if p.lookahead > 1 {
            Some(IndexPlanParams {
                lookahead: p.lookahead - 1,
                ..p
            })
        } else if p.cache_shards > 1 {
            Some(IndexPlanParams {
                cache_shards: p.cache_shards - 1,
                ..p
            })
        } else {
            None
        }
    }
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
        mut config: LoaderConfig,
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
        // This loader has no modality surface, so it reads the flattened shard
        // list. On a multimodal file that list claims every obs row once per
        // modality and `BackedCsrReader`'s row index keeps an arbitrary one —
        // answering from a modality the caller never chose. Refuse instead.
        crate::pipeline::ensure_csr_ranges_are_readable(&reader, None, "IndexPlanLoader")?;
        let n_obs = reader.n_obs();
        let n_vars = reader.n_vars();
        // Captured before `reader` is consumed by `BackedCsrReader::new` below;
        // gates the pflog α auto-estimate (a whole-file source would pool
        // modalities on a multimodal file).
        let n_modalities = reader.n_modalities();

        // Borrow obs / sizes off ScxReader before BackedCsrReader::new takes ownership.
        //
        // Wrapped in `install` because `read_obs` is not the plain section read
        // it looks like: on a file with sharded obs metadata it fans the shard
        // decode out with `par_iter` (`ScxReader::read_sharded_layout_by_prefix`),
        // and this constructor runs inside the forked `DataLoader` worker, where
        // rayon's inherited global pool has no live threads. Every atlas-scale
        // file has sharded obs, so this — not the gather — is the first thing a
        // forked worker hangs on. `install` makes the loader's pool *current*
        // for this thread, so the nested `par_iter` lands there without
        // `ScxReader` needing to know anything about pools.
        let obs_metadata = crate::pool::cpu_pool().install(|| reader.read_obs())?;

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

        // `HvgProjection::new` range-checks against n_vars, so an out-of-range
        // index fails at construction rather than becoming a silently all-zero
        // output column downstream.
        let hvg_projection = match &config.hvg_indices {
            Some(idxs) => Some(HvgProjection::new(idxs.clone(), n_vars)?),
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

        let model = IndexPlanBudgetModel {
            shard_decoded_bytes,
            batch_buffer_bytes,
            transient_bytes,
            max_plan_size,
        };
        let requested = IndexPlanParams {
            cache_shards,
            lookahead,
        };

        // `auto_memory_budget` (set by the Python binding when the caller passed
        // no `max_memory_mb`) resolves the budget from the *requested* config's
        // own need rather than a fixed 512 MB, so a file with large shards is not
        // silently shrunk to `cache_shards = 1` to fit a default that was never
        // chosen for it. Same policy and same arithmetic as the sequential path.
        // `config.max_memory_mb` is rewritten to the resolved value so
        // `max_memory_mb()` / `memory_budget()` report what is actually in force.
        if config.auto_memory_budget {
            config.max_memory_mb = crate::budget::adaptive_budget_mb(
                model.estimate(requested).total_bytes,
                config.max_memory_mb,
            );
        }
        let budget_bytes = config
            .max_memory_mb
            .saturating_mul(1024)
            .saturating_mul(1024);

        // Shrinking `cache_shards` is *reported* (see `cache_sizing` below and
        // the warning in `python.rs`) rather than silent: against the ~470 MB
        // Pcodec shards STATE3 hit, the historical hard 512 MB default drove
        // the descent to `cache_shards = 1` and manufactured the 143 s/batch
        // thrash regime with no signal to the caller. The descent's *behaviour*
        // is deliberately unchanged — it may still reach 1 — because refusing
        // would turn configurations that work today into hard errors.
        //
        // Exhaustion, on the other hand, *is* refused here, and that stays a
        // per-loader decision rather than something the shared driver imposes:
        // a plan-driven loader that cannot hold one shard has nothing useful to
        // do, whereas the sequential path flags it and continues.
        let tuned = crate::budget::tune(&model, requested, budget_bytes);
        if tuned.exhausted {
            return Err(LoaderError::ConfigError {
                reason: format!(
                    "max_memory_mb={} is below the floor for this file: \
                     estimated {} MB at cache_shards=1, lookahead=1 \
                     (shard_decoded={} KB, batch_buffer={} MB, transient={} KB, \
                     py_overhead=50 MB). Increase max_memory_mb.",
                    config.max_memory_mb,
                    tuned.breakdown.total_bytes / (1024 * 1024),
                    shard_decoded_bytes / 1024,
                    batch_buffer_bytes / (1024 * 1024),
                    transient_bytes / 1024,
                ),
            });
        }
        let effective_cache_shards = tuned.params.cache_shards;
        let effective_lookahead = tuned.params.lookahead;
        let budget_breakdown = tuned.breakdown;

        // Construction-time verdict for the caller-facing warning. `None` when
        // the requested cache survived the auto-tune, which is the common case.
        // The non-cache term is passed so the suggested `max_memory_mb` covers
        // the batch buffers and Python overhead too — advice that only sized the
        // cache would still not fit.
        let cache_sizing = crate::budget::assess_cache_sizing(
            cache_shards,
            effective_cache_shards,
            shard_decoded_bytes,
            config.max_memory_mb,
            budget_breakdown
                .total_bytes
                .saturating_sub(budget_breakdown.cache_bytes),
            // An adaptive budget's reduction is by design; only an explicit
            // `max_memory_mb` that conflicts with the requested cache is the
            // caller's problem to resolve.
            !config.auto_memory_budget,
        );

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
        // Keep `warm_shards` off rayon's global registry: this loader is
        // routinely constructed inside a forked `DataLoader` worker, where the
        // inherited global pool has no live worker threads and any dispatch to
        // it hangs forever. See `crate::pool`.
        backed.set_cpu_pool(crate::pool::cpu_pool());
        // Always-on metrics on this surface — the iter's profile log and
        // the per-iter snapshot accessor read from this handle.
        let cache_metrics = backed.enable_metrics();

        // v4 PFlog: resolve α once (mirrors TrainingPipeline::new). A pinned α
        // is validated; `None` ⇒ estimate over the raw CSR shards via the
        // just-built `backed` reader (single-modality only). After this,
        // `config.pflog_alpha` is `Some` whenever `config.pflog`.
        if config.pflog {
            match config.pflog_alpha {
                Some(a) if a <= 0.0 || a.is_nan() || a.is_infinite() => {
                    return Err(LoaderError::ConfigError {
                        reason: "pflog_alpha must be positive and finite".to_string(),
                    });
                }
                Some(_) => {}
                None => {
                    if config.modality_id.is_some() || n_modalities > 1 {
                        return Err(LoaderError::ConfigError {
                            reason: "pflog_alpha must be set explicitly for a multimodal or \
                                     modality-scoped loader (auto-estimation would pool \
                                     modalities); estimate it once via pyscx.accel.pflog and \
                                     pass pflog_alpha"
                                .to_string(),
                        });
                    }
                    // On the loader's pool for the same reason as `read_obs`
                    // above: `estimate_alpha` walks shards through
                    // `scx_format_io::prefetch`, whose `rayon::in_place_scope`
                    // cannot tell an inherited-and-dead global registry from a
                    // live one. Inside `install` the prefetch takes its
                    // already-on-a-worker sequential path — the right trade for
                    // a one-time construction-path estimate.
                    let est = crate::pool::cpu_pool()
                        .install(|| {
                            scx_accel::estimate_alpha(&backed, &scx_accel::AlphaOptions::default())
                        })
                        .map_err(|e| LoaderError::ConfigError {
                            reason: format!("pflog α estimation failed: {e}"),
                        })?;
                    log::info!(
                        "pflog: estimated α={:.6} (pseudocount={:.6}, n_genes_used={}, fell_back={})",
                        est.alpha,
                        est.pseudocount,
                        est.n_genes_used,
                        est.fell_back
                    );
                    config.pflog_alpha = Some(est.alpha);
                }
            }
        }

        // The prefetch engine — and with it the tokio runtime — is built
        // lazily on first use (see the `engine` field) to keep `new` fork-safe:
        // a parent process can construct an `IndexPlanLoader` and then have a
        // `DataLoader` fork worker processes without inheriting a wedged
        // thread pool.
        Ok(Self {
            backed: Arc::new(backed),
            obs_metadata,
            cat_dicts,
            config,
            hvg_projection,
            n_output_cols,
            sort_by_shard,
            engine: OnceLock::new(),
            effective_cache_shards,
            effective_lookahead,
            max_plan_size,
            cache_metrics,
            budget_breakdown,
            cache_sizing,
            requested_cache_shards: cache_shards,
            shard_decoded_bytes,
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
        // Called only during construction, before any clone escapes, so the
        // `Arc` is still uniquely owned.
        Arc::get_mut(&mut self.backed)
            .expect("set_scatter_block_index runs before the reader Arc is shared")
            .set_scatter_block_index(enabled);
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

    /// Whether block-index adoption is enabled for this dataset (default
    /// `true`). Reports the value [`Self::set_scatter_block_index`] pushed into
    /// the backed reader, which is where the decision actually lives:
    /// `BackedCsrReader::block_index_eligible` ANDs it, so it gates **both** the
    /// L1 gather adoption and the L2 prefetch warm-skip. The prefetch path does
    /// not consult this getter — it asks the reader.
    pub fn scatter_block_index(&self) -> bool {
        self.scatter_block_index
    }

    /// True if the opened file has at least one row-group-framed CSR shard.
    /// The scattered block-index fast path can only fire on framed shards; an
    /// all-unframed (legacy v1) file full-shard-decodes every gather no matter
    /// the `scatter_block_index` setting. The Python constructor uses this to
    /// warn when a caller requests the fast path on an unframed file — reframe
    /// with `scx optimize --row-group-rows 256 <file>`.
    pub fn any_shard_framed(&self) -> bool {
        self.backed.any_shard_framed()
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

    /// `Some` iff the memory budget forced the shard cache below the requested
    /// `cache_shards` at construction. Consumed by the Python constructor to
    /// warn; `None` is the common case.
    pub fn cache_sizing(&self) -> Option<crate::budget::CacheSizingVerdict> {
        self.cache_sizing
    }

    /// Verdict on whether building the HVG projection changed the panel the
    /// caller passed — `None` when it was already ascending and unique.
    /// Mirrors `TrainingPipeline::hvg_panel`; see
    /// [`crate::projection::assess_hvg_panel`].
    pub fn hvg_panel(&self) -> Option<crate::projection::HvgPanelVerdict> {
        self.config
            .hvg_indices
            .as_deref()
            .and_then(crate::projection::assess_hvg_panel)
    }

    /// The `cache_shards` the caller requested, before the budget auto-tune.
    pub fn requested_cache_shards(&self) -> usize {
        self.requested_cache_shards
    }

    /// Average decoded bytes per CSR shard, as used by the budget model.
    pub fn shard_decoded_bytes(&self) -> usize {
        self.shard_decoded_bytes
    }

    /// Number of CSR shards in the file — the cap on distinct cache entries.
    pub fn n_shards(&self) -> usize {
        self.backed.index().n_shards()
    }

    /// O(log n_shards) lookup of the shard containing `row`. Returns `None`
    /// for rows outside every shard's range (should not happen for valid
    /// `row < n_obs` on a well-formed file).
    pub fn shard_of(&self, row: u64) -> Option<usize> {
        self.backed.index().shard_for_row(row)
    }

    /// Number of distinct CSR shards a `(pert, ctrl)` plan touches — i.e. the
    /// `cache_shards` that would let the whole plan stay resident for the
    /// duration of one `process_plan` call.
    ///
    /// Pure index arithmetic via [`scx_format_io::backed::BackedCsrIndex::shards_for_indices`]
    /// (which sorts + dedups internally); no I/O and no decode, so it is safe to
    /// call on a plan before deciding how to size the cache. Deliberately *not*
    /// routed through the prefetch path, which computes the same set but is
    /// skipped entirely when `lookahead == 0`.
    pub fn plan_shard_touch_count(&self, plan: &[(u64, u64)]) -> usize {
        let mut rows: Vec<u64> = Vec::with_capacity(plan.len() * 2);
        for &(p, c) in plan {
            rows.push(p);
            rows.push(c);
        }
        self.backed.index().shards_for_indices(&rows).len()
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
        // Full pre-projection depth per output slot (see `scatter_pair_request`).
        // Used as the normalize denominator so an HVG-projected panel normalizes
        // by full transcriptome depth, not the panel-local sum.
        let mut depth_x = vec![0f64; n_pairs];
        let mut depth_x_paired = vec![0f64; n_pairs];
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
                    if let Err(e) = self.scatter_pair_request(
                        request,
                        idx,
                        data,
                        n_cols,
                        &mut x,
                        &mut x_paired,
                        &mut depth_x,
                        &mut depth_x_paired,
                    ) {
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

        // PFlog is applied at scatter time (it needs the full pre-projection
        // row for the centering baseline — see `scatter_pair_request`), so the
        // post-scatter normalize/log1p dispatch is skipped in that mode.
        if !self.config.pflog {
            for i in 0..n_pairs {
                let p_out = &mut x[i * n_cols..][..n_cols];
                apply_dense_transforms(
                    p_out,
                    self.config.normalize,
                    self.config.log1p,
                    self.config.target_sum,
                    depth_x[i],
                );
                let c_out = &mut x_paired[i * n_cols..][..n_cols];
                apply_dense_transforms(
                    c_out,
                    self.config.normalize,
                    self.config.log1p,
                    self.config.target_sum,
                    depth_x_paired[i],
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

    #[allow(clippy::too_many_arguments)]
    fn scatter_pair_request(
        &self,
        request: PairRequest,
        idx: &[i32],
        data: &[f32],
        n_cols: usize,
        x: &mut [f32],
        x_paired: &mut [f32],
        depth_x: &mut [f64],
        depth_x_paired: &mut [f64],
    ) -> Result<()> {
        let out = match request.side {
            PairSide::Perturbed => &mut x[request.pair_idx * n_cols..][..n_cols],
            PairSide::Control => &mut x_paired[request.pair_idx * n_cols..][..n_cols],
        };
        if self.config.pflog {
            // Centering D over the FULL transcriptome (`idx`/`data` is the full
            // row); each output slot is written by exactly one request, so the
            // scatter fully produces the final PFlog row (delta + baseline).
            // v4 has no depth; `four_alpha = 4α`.
            let n_vars_full = self.backed.n_vars();
            let four_alpha = 4.0
                * self
                    .config
                    .pflog_alpha
                    .expect("pflog_alpha resolved at construction");
            match self.hvg_projection.as_ref() {
                Some(hvg) => hvg.scatter_pflog_row(idx, data, four_alpha, n_vars_full, out),
                None => pflog_row_full(idx, data, four_alpha, n_vars_full, out)?,
            }
        } else {
            match self.hvg_projection.as_ref() {
                Some(hvg) => hvg.scatter_row(idx, data, out),
                None => scatter_row_full(idx, data, out)?,
            }
            // Record the cell's FULL pre-projection depth (`data` is the full
            // row's stored nonzeros) for the post-scatter normalize dispatch.
            // With an HVG projection `out` holds only the panel genes, so its
            // own sum would be a panel-local depth that diverges from scanpy's
            // normalize-then-subset semantics. Only needed when normalizing —
            // the depth vectors stay 0.0 (and are ignored) otherwise.
            if self.config.normalize {
                // Clipped when a log follows — see `normalize::transform_depth`.
                let depth: f64 = crate::normalize::transform_depth(data, self.config.log1p);
                match request.side {
                    PairSide::Perturbed => depth_x[request.pair_idx] = depth,
                    PairSide::Control => depth_x_paired[request.pair_idx] = depth,
                }
            }
        }
        Ok(())
    }

    /// The prefetch engine, built on first use. One reader (`file_id = 0`),
    /// wrapping the reader this loader already configured — `set_cpu_pool` and
    /// `enable_metrics` ran in `new`, so this deliberately does **not** go
    /// through `PrefetchEngine::from_scx_readers`, which would build a second
    /// shard cache and discard both.
    fn engine(&self) -> &Arc<PrefetchEngine> {
        self.engine.get_or_init(|| {
            PrefetchEngine::new(vec![Arc::clone(&self.backed)], self.effective_lookahead)
        })
    }

    /// Drive the loader from a plan stream, hiding shard-prefetch latency
    /// behind upcoming-batch compute.
    ///
    /// `plans` is any `Iterator<Item = Result<Vec<(u64, u64)>, LoaderError>>` —
    /// the iterator is consumed lazily. `lookahead` controls how many
    /// upcoming plans get shard-prefetched concurrently with the current
    /// batch's decode (0 disables; default per Python API is 4).
    ///
    /// One `spawn_blocking` task is queued per shard referenced by an upcoming
    /// plan that is not already resident in the LRU, being decoded by a peer,
    /// or block-index eligible. Decode of the head plan blocks on its prefetch
    /// handles before calling [`Self::process_plan`].
    ///
    /// **Empty plans yield no batch** (SCX-DATA-LOADER: "Plan list is empty →
    /// yield no batch for that plan; continue to the next"). That is this
    /// loader's contract, not the engine's — [`PrefetchEngine::iter_with_plans`]
    /// deliberately calls `process` for every plan — so the skip lives in
    /// [`IndexPlanIter::next`], which discards the zero-row batch `process_plan`
    /// returns for an empty plan.
    ///
    /// It must **not** be a `filter` on the plan stream, however tempting: the
    /// pull worker observes cancellation only through `plan_tx.send`, and
    /// `Filter::next` discards items before reaching it. An all-empty stream
    /// would then never learn the receiver was dropped and would spin forever
    /// (measured: ~300M pulls in the 10s after drop), and even a finite empty
    /// prefix would be pulled with no backpressure, since it never occupies a
    /// channel slot. Pinned by
    /// `dropping_the_iter_stops_an_endless_empty_plan_generator`.
    pub fn iter_with_plans<I>(self: Arc<Self>, plans: I, lookahead: usize) -> IndexPlanIter
    where
        I: Iterator<Item = std::result::Result<Vec<(u64, u64)>, LoaderError>> + Send + 'static,
    {
        let engine = Arc::clone(self.engine());
        let loader = Arc::clone(&self);
        let iter = engine.iter_with_plans(
            plans,
            lookahead,
            // Single file, so every row is `file_id = 0`. Both sides of a pair
            // are reported: the gather passes the same deduped row set to
            // `read_rows_with`, so the engine's per-shard `group_len` matches
            // what the gather's block-index decision sees.
            |plan: &Vec<(u64, u64)>| {
                // `with_capacity` + push rather than `flat_map(..).collect()`:
                // the length is exactly `2 * plan.len()`, but `FlatMap`'s
                // `size_hint` lower bound is not, so `collect` grows the Vec on
                // the training hot path.
                let mut rows = Vec::with_capacity(plan.len() * 2);
                for &(p, c) in plan {
                    rows.push((0u32, p));
                    rows.push((0u32, c));
                }
                rows
            },
            // The plan arrives by value, so `process_plan` keeps its in-place
            // `sort_by_shard` and moves the sorted plan straight into the
            // batch's `pairs` — no defensive clone per batch.
            move |_engine: &PrefetchEngine, plan: Vec<(u64, u64)>| loader.process_plan(plan),
        );
        // Taken before boxing, which erases the inherent method.
        let iter_metrics = iter.iter_metrics();
        IndexPlanIter {
            inner: Box::new(iter),
            iter_metrics,
        }
    }
}

/// Iterator returned by [`IndexPlanLoader::iter_with_plans`].
///
/// **An adapter, not an implementation.** The plan-pull thread, the bounded
/// lookahead queue, the per-plan shard prefetch, the prefetch counters and the
/// bounded teardown all live in [`crate::plan_engine`]; this type exists only
/// because `PlanPrefetchIter`'s closure type parameters are unnameable, and
/// `IndexPlanBatchIter` has to store the iterator in a `#[pyclass]` field while
/// still reading its counters.
///
/// Until ORG-9.10-1 this was a 340-line fork of `PlanPrefetchIter` that had
/// already drifted five ways. If you find yourself adding logic here, it
/// belongs in the engine instead — that is the whole point of the fold.
pub struct IndexPlanIter {
    /// `Sync` as well as `Send` because `IndexPlanBatchIter` stores this in a
    /// `#[pyclass]` field, and pyo3 requires the whole struct to be `Sync`.
    inner: Box<dyn Iterator<Item = Result<IndexPlanBatch>> + Send + Sync>,
    iter_metrics: Arc<IterMetrics>,
}

impl IndexPlanIter {
    /// Cloneable handle to this iter's prefetch counters. Sample at any
    /// time — atomics are `Relaxed`, no locks involved. Stays valid after the
    /// iterator is drained and dropped.
    pub fn iter_metrics(&self) -> Arc<IterMetrics> {
        Arc::clone(&self.iter_metrics)
    }
}

impl Iterator for IndexPlanIter {
    type Item = Result<IndexPlanBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        // Skip the zero-row batches an empty plan produces — the loader's
        // contract, which the engine deliberately does not implement. This is
        // exact rather than a heuristic: `process_plan` moves the (sorted) plan
        // into `pairs`, so `pairs.is_empty()` holds precisely when the plan was
        // empty. Errors pass through untouched and in order.
        loop {
            match self.inner.next()? {
                Ok(batch) if batch.pairs.is_empty() => continue,
                item => return Some(item),
            }
        }
    }
}

#[cfg(test)]
#[path = "index_plan_tests.rs"]
mod tests;
