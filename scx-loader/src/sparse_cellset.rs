//! Native sparse cell-set loader.
//!
//! Consumes state3's role-tagged, multi-file cell-set plans and gathers them as
//! sparse CSR through the [`crate::plan_engine::PrefetchEngine`], emitting the
//! SCX-DATA-LOADER §4.4 batch contract. Each plan item is **one batch** of cell
//! sets, delimited by `set_offsets`; each set's rows are gathered via one
//! `read_rows_with` per reader (single-file fast path) into per-set CSR builders.
//!
//! Output is **raw-local CSR by default**; an **optional** per-file `local→global`
//! remap (off by default) emits global-vocab CSR for callers that want it (§4.3
//! item 3). With the remap on, this path owns the whole of the consumer's
//! `finalize_csr_row` equivalent: remap, `-1` drop, sort, coalesce, the
//! non-negativity clip, and — when configured — a seeded count downsample
//! ([`crate::downsample`]). Only the *raw-local* configuration still leaves
//! remap/coalesce to the caller.

use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;
use scx_format_io::{CacheMetrics, ScxReader};

use crate::error::{LoaderError, Result};
use crate::plan_engine::{IterMetrics, PrefetchEngine};
use crate::sparse_cellset_collate::{collate_cell, CellIn, CellOut, CollateConfig, PreprocessMode};

/// One batch of cell sets to gather. Rows are flat across all sets in the
/// batch; `set_offsets` (length `n_sets + 1`) delimits each set's row range.
/// On the common path every row of a set shares a `file_id`.
#[derive(Clone, Debug)]
pub struct SparseCellSetPlan {
    pub file_ids: Vec<u32>,
    pub rows: Vec<u64>,
    pub role_tags: Vec<i32>,
    pub set_offsets: Vec<i64>,
}

/// Gathered sparse batch — the §4.4 contract. One output row per plan row, in
/// plan order; `set_offsets` carries over from the plan.
#[derive(Clone, Debug)]
pub struct SparseCellSetBatch {
    pub indptr: Vec<i64>,
    pub indices: Vec<i32>,
    pub data: Vec<f32>,
    pub shape: (usize, usize),
    pub cell_indices: Vec<u64>,
    pub file_ids: Vec<u32>,
    pub set_offsets: Vec<i64>,
    pub role_tags: Vec<i32>,
}

/// Run-constant collation scalars.
#[derive(Clone, Copy, Debug)]
pub struct CollateScalars {
    pub k_enc: usize,
    pub mode: PreprocessMode,
    pub target_sum: f64,
    /// PFlog (v4) NB overdispersion `α`; required when `mode == PflogRaw`.
    pub pflog_alpha: Option<f64>,
    pub n_genes_total: i64,
    pub lib_size_redef: bool,
}

/// Collated batch — stacked dense tensors, row-major `[total_rows, K]`. Wraps
/// directly into state3's `ScRNABatch` (no Python `task.specify`/`fill_one`).
#[derive(Clone, Debug)]
pub struct CollatedCellSetBatch {
    pub encoder_gene_ids: Vec<i64>, // [n_rows * k_enc]
    pub encoder_counts: Vec<f32>,   // [n_rows * k_enc]
    pub encoder_mask: Vec<u8>,      // [n_rows * k_enc]
    pub encoder_pad_mask: Vec<u8>,  // [n_rows * k_enc]
    pub target_counts: Vec<f32>,    // [n_rows * k_dec]
    pub library_size: Vec<f32>,     // [n_rows]
    pub cell_indices: Vec<u64>,
    pub file_ids: Vec<u32>,
    pub set_offsets: Vec<i64>,
    pub role_tags: Vec<i32>,
    pub n_rows: usize,
    pub k_enc: usize,
    pub k_dec: usize,
}

/// Drives the prefetch engine to gather sparse cell-set batches.
pub struct SparseCellSetLoader {
    engine: Arc<PrefetchEngine>,
    /// Optional per-`file_id` `local→global` table (`-1` = gene absent). When
    /// set, gathered indices are remapped into the global vocab; otherwise
    /// indices are raw-local (the default — state3 remaps in Python).
    remap: Option<Vec<Vec<i32>>>,
    normalize: bool,
    log1p: bool,
    target_sum: f64,
    /// CSR column count: max per-file `n_vars` (raw-local) or global vocab size
    /// (remapped). Raw-local indices are file-local — consumers disambiguate via
    /// `file_ids`; `n_cols` is a nominal upper bound.
    n_cols: usize,
    /// Resolved cache byte budget (bytes) actually handed to the shared cache.
    /// Reported through `SparseCellSetDataset.memory_budget()`.
    cache_bytes_budget: usize,
    /// Requested shard-cache count cap.
    cache_shards: usize,
    /// `min(cache_shards, budget / avg_shard)` — the count that is actually
    /// resident-capable, i.e. **the binding constraint**. Surfaced through
    /// [`Self::effective_cache_shards`] and reported as `effective_cache_shards`,
    /// the name the paired loader uses for the same quantity; the field keeps
    /// the more descriptive spelling. `cache_shards` alone is
    /// misleading on a large-shard file where the byte budget binds first, which
    /// is exactly the STATE3 regime this loader targets.
    affordable_cache_shards: usize,
    /// Average decoded bytes per CSR shard across every file, the unit the
    /// budget model counts in.
    shard_decoded_bytes: usize,
    /// `estimate(effective_cache_shards)` from the shared auto-tune — the
    /// number the loader reports *and* the one it enforced.
    budget_breakdown: crate::budget::BudgetBreakdown,
    /// Byte cap actually handed to the shared shard cache. Equals
    /// `budget_breakdown.cache_bytes` whenever the shard size is known, and the
    /// raw budget when it is not — see [`Self::enforced_cache_bytes`].
    enforced_cache_bytes: usize,
    /// The auto-tune ran out of knobs before fitting: even a one-shard cache
    /// exceeds the budget. Reported rather than refused, matching
    /// `MemoryBudget::budget_exceeded` on the sequential path
    /// (`IndexPlanLoader` is the one class that refuses instead).
    budget_exceeded: bool,
    /// `Some` iff `cache_bytes_budget` cannot hold `cache_shards` average
    /// shards. Consumed by the Python constructor to warn; see
    /// [`crate::budget::assess_cache_sizing`].
    cache_sizing: Option<crate::budget::CacheSizingVerdict>,
    /// Optional seeded count-downsample applied per row in [`Self::transform_row`],
    /// i.e. **before** the batch leaves Rust. Placement is load-bearing, not
    /// convenience: consumers sample the decoder query from the gathered counts
    /// (`counts > 0`) and then hand the *same* arrays to the collate kernel, so a
    /// downsample applied later would draw the query from pre-downsample expressed
    /// genes while the numerics used post-downsample counts — a silent divergence
    /// from the Python reference, which downsamples inside `finalize_csr_row`
    /// before task specification.
    downsample: Option<crate::downsample::DownsampleConfig>,
}

/// Average decoded bytes per CSR shard across `readers`, from catalog stats
/// only (no decode). Same per-shard model as `IndexPlanLoader`'s auto-tune and
/// as `scx_format_io`'s `SizeHint for ScxCsr`: `nnz × 8` (i32 indices + f32
/// data) + `rows × 8` (i64 indptr).
fn avg_shard_decoded_bytes(readers: &[ScxReader]) -> usize {
    let mut total_nnz = 0u64;
    let mut total_rows = 0u64;
    // Only shards that CONTRIBUTED to the totals may count toward the divisor.
    // Counting stat-less shards in the denominator averages their 0 bytes into
    // the result, *under*-estimating the per-shard size — which then
    // *over*-estimates how many shards the byte budget affords and makes the
    // sizing diagnostic under-warn on exactly the files whose catalogs are
    // incomplete. Flagged independently by all three round-2 reviewers.
    let mut n_counted = 0u64;
    for r in readers {
        for e in r.catalog().shards_sorted() {
            if let Some(s) = e.stats.as_ref() {
                total_nnz += s.nnz;
                total_rows += s.row_end - s.row_start;
                n_counted += 1;
            }
        }
    }
    // No shard carried stats ⇒ the size is genuinely unknown. Returning 0 is the
    // signal `SparseCellSetLoader::new` reads as "the byte cap tells us nothing",
    // falling back to the count cap rather than to a fabricated average.
    total_nnz
        .saturating_mul(8)
        .saturating_add(total_rows.saturating_mul(8))
        .checked_div(n_counted)
        .unwrap_or(0) as usize
}

use crate::budget::BudgetModel;

/// Resolve the cache count in **closed form**, and return it in the same
/// [`crate::budget::Tuned`] shape [`crate::budget::tune`] would.
///
/// This model is affine in its single knob — `n × shard_decoded_bytes +
/// PYTHON_OVERHEAD_BYTES` — so the largest `n` that fits is a division, not a
/// search. Running the shared descent here would step down one shard at a time
/// from a **caller-controlled** `cache_shards`: measured at ~0.73 ns/step, that
/// is 0.7 s of pure spinning at `cache_shards=1e9` and does not terminate in
/// any useful time at `usize::MAX`. The pre-ORG-9.10-5 code was O(1) arithmetic
/// and this restores that. (Raised in review — codex, round 2.)
///
/// It must agree with the driver exactly, and
/// `closed_form_agrees_with_the_shared_driver` pins that across a budget sweep;
/// the model keeps its [`crate::budget::BudgetModel`] impl so the two are
/// comparable and so the monotonicity harness still covers it.
fn resolve_sparse_cache_shards(
    model: &SparseCellSetBudgetModel,
    requested: SparseCellSetParams,
    budget_bytes: usize,
) -> crate::budget::Tuned<SparseCellSetParams> {
    let non_cache = model
        .estimate(SparseCellSetParams { cache_shards: 0 })
        .total_bytes;
    let affordable = budget_bytes
        .saturating_sub(non_cache)
        .checked_div(model.shard_decoded_bytes)
        .unwrap_or(requested.cache_shards);
    // The descent floors at one shard and never rises above the request — and
    // an explicit request of 0 stays 0, because `reduce` only ever *reduces*.
    let ceiling = requested.cache_shards;
    let cache_shards = affordable.clamp(ceiling.min(1), ceiling);
    let params = SparseCellSetParams { cache_shards };
    let breakdown = model.estimate(params);
    crate::budget::Tuned {
        exhausted: !breakdown.fits_within_bytes(budget_bytes),
        params,
        breakdown,
    }
}

/// The byte cap actually handed to the shared shard cache.
///
/// With a known per-shard size this is the tuned cache, the same tightening
/// `IndexPlanLoader` performs so the cache cannot overshoot the budget when
/// real shard sizes diverge from the average.
///
/// With an **unknown** one it must be the raw budget. `avg_shard_decoded_bytes`
/// returns 0 when no shard in any catalog carries stats, which means "the byte
/// cap tells us nothing" — and the model's cache term is then `n × 0 = 0`, so
/// tightening to it would cap the cache at **zero bytes**: "we cannot size the
/// cache" silently becoming "no cache", and strictly worse than the budget this
/// path passed before ORG-9.10-5. `affordable_cache_shards` already falls back
/// to the count cap in exactly this case; this is the byte half of the same
/// rule.
///
/// Pure, and separated from `new` deliberately: no writer emits a stats-less
/// CSR entry, so the branch is unreachable from any file a test can build. The
/// decision is asserted directly instead of through a fixture that cannot exist.
fn resolve_enforced_cache_bytes(
    shard_decoded_bytes: usize,
    model_cache_bytes: usize,
    raw_budget_bytes: usize,
) -> usize {
    if shard_decoded_bytes == 0 {
        raw_budget_bytes
    } else {
        model_cache_bytes
    }
}

/// The one knob the sparse auto-tune reduces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SparseCellSetParams {
    pub(crate) cache_shards: usize,
}

/// The `SparseCellSetLoader` arm of [`crate::budget::BudgetModel`].
///
/// Only two terms are non-zero. **What that does and does not buy is worth
/// being exact about**, because ORG-9.10-5 made the budget cover the
/// interpreter constant and it would be easy to over-read that as a hard
/// ceiling on the process:
///
/// * **Charged**: the decoded shard cache, sized and enforced from this model,
///   plus the interpreter/numpy/Arrow constant every path pays.
/// * **Not charged**: the gathered batch itself and its transients. Unlike
///   `IndexPlanLoader`, this path has no `max_plan_size`, so a plan's output
///   size is caller-controlled and unbounded — there is nothing fixed to cost.
/// * **Not a hard cap**: `WeightedLruCache::put_with_budget` deliberately keeps
///   a single entry that exceeds the byte budget on its own (refusing would
///   defeat the cache for any outsized shard), so one above-average shard can
///   sit above the cap. The model tunes on the *average* shard size.
///
/// So `budget_exceeded == false` means "the cache this loader sizes fits the
/// budget", not "the process will stay under `max_memory_mb`".
///
/// The floor is 1, not 0, for the reason `IndexPlanLoader` refuses below 1: a
/// cache that can hold nothing re-decodes every shard on every batch, and the
/// engine's own `cache_shards.max(2)` would override a 0 anyway. Exhaustion
/// here is reported (a below-floor `CacheSizingVerdict`), not refused — an
/// absurd budget must keep warning rather than start raising.
pub(crate) struct SparseCellSetBudgetModel {
    shard_decoded_bytes: usize,
}

impl crate::budget::BudgetModel for SparseCellSetBudgetModel {
    type Params = SparseCellSetParams;

    fn estimate(&self, p: SparseCellSetParams) -> crate::budget::BudgetBreakdown {
        crate::budget::BudgetBreakdown::new(
            p.cache_shards.saturating_mul(self.shard_decoded_bytes),
            0,
            0,
            0,
            crate::budget::PYTHON_OVERHEAD_BYTES,
        )
    }

    fn reduce(&self, p: SparseCellSetParams) -> Option<SparseCellSetParams> {
        (p.cache_shards > 1).then(|| SparseCellSetParams {
            cache_shards: p.cache_shards - 1,
        })
    }
}

impl SparseCellSetLoader {
    /// Build a loader over `scx_readers` (one per `file_id`, in slice order),
    /// sharing one decoded-shard budget. `remap`/`n_global_genes` enable global
    /// remap (`n_global_genes` falls back to the tables' max+1 when `None`).
    ///
    /// `bytes_budget` is the shared shard cache's byte cap. `None` resolves it
    /// **adaptively** — the requested cache's own need with headroom, clamped to
    /// `[512 MB, ADAPTIVE_BUDGET_CAP_MB]` via
    /// [`crate::budget::adaptive_budget_mb`] — rather than the historical
    /// `usize::MAX`, which left peak RSS unbounded (STATE3 observed ~23 GB).
    /// This matches how `TrainingDataset` and `IndexPlanDataset` resolve a
    /// `None` budget, so all three loader classes share one policy.
    ///
    /// `scatter_block_index` is a pure pass-through to
    /// [`PrefetchEngine::from_scx_readers`] — deliberately not stored, because
    /// the flag's only consumer is the reader it is set on, and a second copy
    /// here could disagree with it. `SparseCellSetDataset` *defaults* it to
    /// `false` and passes the caller's kwarg; see that constructor for why the
    /// cell-set regime wants the full-shard warm+cache path by default.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scx_readers: Vec<ScxReader>,
        cache_shards: usize,
        bytes_budget: Option<usize>,
        lookahead: usize,
        remap: Option<Vec<Vec<i32>>>,
        n_global_genes: Option<usize>,
        normalize: bool,
        log1p: bool,
        target_sum: f64,
        downsample: Option<crate::downsample::DownsampleConfig>,
        scatter_block_index: bool,
    ) -> Result<Arc<Self>> {
        // Same reason as `IndexPlanLoader`: this loader resolves cells by global
        // obs row through `BackedCsrReader`, so a multimodal file's flattened
        // shard list would answer from an arbitrary modality.
        for (file_id, reader) in scx_readers.iter().enumerate() {
            crate::pipeline::ensure_csr_ranges_are_readable(
                reader,
                None,
                &format!("SparseCellSetLoader (file {file_id})"),
            )?;
        }
        if let Some(cfg) = &downsample {
            cfg.validate()?;
            // An empty table means "no stable per-file identity", which keys on
            // `(seed, method, row)` alone. That is fine for one file and ambiguous
            // for several — two files' row 5 would share a draw. Refuse it rather
            // than fall back to `file_id`, which would be construction-order
            // keying: a reordered manifest or a debugging subset silently redrawing
            // every cell while producing perfectly plausible output.
            if cfg.file_identities.is_empty() && scx_readers.len() > 1 {
                return Err(LoaderError::ConfigError {
                    reason: format!(
                        "downsample over {} files requires file_identities (one stable \
                         per-file id, e.g. scx_loader::downsample::file_identity(path)); \
                         without them two files' row N would share a draw",
                        scx_readers.len()
                    ),
                });
            }
            if !cfg.file_identities.is_empty() && cfg.file_identities.len() != scx_readers.len() {
                return Err(LoaderError::ConfigError {
                    reason: format!(
                        "downsample file_identities has {} entries but there are {} files",
                        cfg.file_identities.len(),
                        scx_readers.len()
                    ),
                });
            }
        }
        if let Some(tables) = &remap {
            if tables.len() != scx_readers.len() {
                return Err(LoaderError::ConfigError {
                    reason: format!(
                        "remap has {} tables but there are {} files",
                        tables.len(),
                        scx_readers.len()
                    ),
                });
            }
        }
        let n_cols = match (&remap, n_global_genes) {
            (Some(_), Some(g)) => g,
            (Some(tables), None) => tables
                .iter()
                .flat_map(|t| t.iter().copied())
                .filter(|&g| g >= 0)
                .max()
                .map(|g| g as usize + 1)
                .unwrap_or(0),
            (None, _) => scx_readers
                .iter()
                .map(|r| r.n_vars() as usize)
                .max()
                .unwrap_or(0),
        };
        // Resolve the cache byte budget before the readers are moved into the
        // engine: the adaptive path needs their catalog stats.
        let shard_decoded_bytes = avg_shard_decoded_bytes(&scx_readers);
        let model = SparseCellSetBudgetModel {
            shard_decoded_bytes,
        };
        let requested = SparseCellSetParams { cache_shards };
        let cache_bytes_budget = match bytes_budget {
            Some(b) => b,
            None => crate::budget::adaptive_budget_mb(
                model.estimate(requested).total_bytes,
                crate::pipeline::LoaderConfig::default().max_memory_mb,
            )
            .saturating_mul(1024 * 1024),
        };
        // An unknown shard size (no catalog stats anywhere) means "the byte cap
        // tells us nothing", which is the count cap — never a fabricated
        // average and never a descent driven by the interpreter constant alone.
        let tuned = if shard_decoded_bytes == 0 {
            crate::budget::Tuned {
                params: requested,
                breakdown: model.estimate(requested),
                exhausted: false,
            }
        } else {
            resolve_sparse_cache_shards(&model, requested, cache_bytes_budget)
        };
        let affordable_cache_shards = tuned.params.cache_shards;
        let budget_breakdown = tuned.breakdown;
        let enforced_cache_bytes = resolve_enforced_cache_bytes(
            shard_decoded_bytes,
            budget_breakdown.cache_bytes,
            cache_bytes_budget,
        );
        // ORG-9.10-5: the non-cache term is the interpreter constant, not 0.
        // Passing 0 here is what let the reported `total_bytes` exceed the
        // budget the tuner had just checked — the tuner disagreeing with the
        // report, which `IndexPlanLoader` never did.
        let cache_sizing = crate::budget::assess_cache_sizing(
            cache_shards,
            affordable_cache_shards,
            shard_decoded_bytes,
            cache_bytes_budget / (1024 * 1024),
            budget_breakdown
                .total_bytes
                .saturating_sub(budget_breakdown.cache_bytes),
            bytes_budget.is_some(),
        );

        // Hand the engine the *tuned* caps, both of them — the same tightening
        // `IndexPlanLoader` performs, so the cache cannot overshoot the budget
        // when actual shard sizes diverge from the average. Before ORG-9.10-5
        // this path handed over the raw request and the raw byte budget, so the
        // numbers it reported described a cache it was not enforcing.
        let engine = PrefetchEngine::from_scx_readers(
            scx_readers,
            affordable_cache_shards,
            enforced_cache_bytes,
            lookahead,
            scatter_block_index,
        );
        Ok(Arc::new(SparseCellSetLoader {
            engine,
            remap,
            normalize,
            log1p,
            target_sum,
            n_cols,
            cache_bytes_budget,
            cache_shards,
            affordable_cache_shards,
            shard_decoded_bytes,
            budget_breakdown,
            enforced_cache_bytes,
            budget_exceeded: tuned.exhausted,
            cache_sizing,
            downsample,
        }))
    }

    /// Resolved shared-cache byte budget. With an explicit `max_memory_mb` this
    /// is that value; otherwise the adaptive resolution.
    pub fn cache_bytes_budget(&self) -> usize {
        self.cache_bytes_budget
    }

    /// Requested shard-cache count cap. See [`Self::effective_cache_shards`] for
    /// the value that actually binds.
    pub fn cache_shards(&self) -> usize {
        self.cache_shards
    }

    /// Shard-cache entries actually affordable: `min(cache_shards, budget /
    /// avg_shard_bytes)`.
    ///
    /// The count cap and the byte cap are enforced independently, and on a
    /// large-shard file the **byte** cap binds first — a 4 GB budget over ~470 MB
    /// shards holds ~8 entries however high `cache_shards` is set. Diagnostics
    /// must be phrased against this, not against the request, or they name a knob
    /// that cannot fix the problem (`IndexPlanLoader::effective_cache_shards` is
    /// the same idea on the paired path).
    pub fn effective_cache_shards(&self) -> usize {
        self.affordable_cache_shards
    }

    /// Average decoded bytes per CSR shard across all files.
    pub fn shard_decoded_bytes(&self) -> usize {
        self.shard_decoded_bytes
    }

    /// Per-component memory estimate, in the one shape every class that reports
    /// a budget uses (ORG-9.10-4).
    ///
    /// Only two terms are non-zero, and that is the model, not an omission: the
    /// shard cache **is** this loader's budget — there is no batch buffer,
    /// no plan-tuple staging and no per-batch obs scratch on the gather path —
    /// plus the interpreter/numpy/Arrow constant every path pays.
    ///
    /// Since ORG-9.10-5 the interpreter constant is **budgeted**, not merely
    /// reported: the auto-tune subtracts it before sizing the cache, so
    /// `total_bytes` fits `cache_bytes_budget` whenever the tune was not
    /// exhausted — the invariant `IndexPlanLoader` always held and this path
    /// did not.
    pub fn budget_breakdown(&self) -> crate::budget::BudgetBreakdown {
        self.budget_breakdown
    }

    /// Byte cap the shared shard cache is actually running under.
    ///
    /// Normally `budget_breakdown().cache_bytes`. On a file whose catalog
    /// carries no shard stats the per-shard size is unknown, the model's cache
    /// term collapses to zero, and this falls back to the resolved budget —
    /// otherwise "we cannot size the cache" would silently mean "no cache".
    pub fn enforced_cache_bytes(&self) -> usize {
        self.enforced_cache_bytes
    }

    /// `true` when even a one-shard cache exceeds the budget, so
    /// [`Self::budget_breakdown`] is over budget by construction.
    ///
    /// The sparse counterpart of `MemoryBudget::budget_exceeded`. Reported, not
    /// refused: an absurd `max_memory_mb` must keep warning rather than start
    /// raising, which is the one place this path's exhaustion policy differs
    /// from `IndexPlanLoader`'s (and it is a deliberate difference, see
    /// `crate::budget::Tuned::exhausted`).
    pub fn budget_exceeded(&self) -> bool {
        self.budget_exceeded
    }

    /// Total CSR shards across every file — the cap on distinct entries in the
    /// shared cache, which is keyed `(file_id, shard)`.
    pub fn total_shards(&self) -> usize {
        (0..self.engine.n_readers())
            .map(|fid| self.engine.reader(fid as u32).index().n_shards())
            .sum()
    }

    /// `Some` iff the byte budget cannot hold `cache_shards` average shards.
    pub fn cache_sizing(&self) -> Option<crate::budget::CacheSizingVerdict> {
        self.cache_sizing
    }

    /// Number of distinct `(file_id, shard)` pairs a cell-set plan touches —
    /// i.e. the `cache_shards` that would let the whole batch stay resident.
    ///
    /// Pure index arithmetic (`shards_for_indices` per file); no I/O, no decode.
    /// Rows are grouped by `file_id` first because shard indices are per-file and
    /// would otherwise collide across files.
    pub fn plan_shard_touch_count(&self, file_ids: &[u32], rows: &[u64]) -> usize {
        let mut by_file: HashMap<u32, Vec<u64>> = HashMap::new();
        for (&f, &r) in file_ids.iter().zip(rows.iter()) {
            by_file.entry(f).or_default().push(r);
        }
        by_file
            .into_iter()
            .filter(|(f, _)| (*f as usize) < self.engine.n_readers())
            .map(|(f, rs)| self.engine.reader(f).index().shards_for_indices(&rs).len())
            .sum()
    }

    /// Number of readers (`file_id` range).
    pub fn n_files(&self) -> usize {
        self.engine.n_readers()
    }

    /// True if any file in the set has a row-group-framed CSR shard.
    ///
    /// The multi-file sibling of [`crate::IndexPlanLoader::any_shard_framed`],
    /// and the same consumer: the Python constructor warns when the caller
    /// asked for `scatter_block_index=True` against a set where the route can
    /// never fire.
    pub fn any_shard_framed(&self) -> bool {
        self.engine.any_shard_framed()
    }

    /// Shared handle to the readers' one `SharedShardCache` counters
    /// (hits / misses / evictions / …), cumulative since construction.
    pub fn cache_metrics(&self) -> Arc<CacheMetrics> {
        self.engine.cache_metrics()
    }

    /// CSR column count of emitted batches.
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Stream `plans` (each one batch) into gathered §4.4 batches, pipelining
    /// shard prefetch via the engine.
    pub fn iter_with_plans<I>(self: Arc<Self>, plans: I, lookahead: usize) -> SparseCellSetIter
    where
        I: Iterator<Item = Result<SparseCellSetPlan>> + Send + 'static,
    {
        let engine = Arc::clone(&self.engine);
        let loader = Arc::clone(&self);
        let iter = engine.iter_with_plans(
            plans,
            lookahead,
            |plan: &SparseCellSetPlan| {
                plan.file_ids
                    .iter()
                    .copied()
                    .zip(plan.rows.iter().copied())
                    .collect()
            },
            move |eng: &PrefetchEngine, plan: SparseCellSetPlan| loader.gather(eng, &plan),
        );
        // Taken before boxing: `Box<dyn Iterator>` erases the inherent method,
        // and the counters are per-iter (they reset every `iter_with_plans`
        // call), so the handle cannot come from the loader instead.
        let iter_metrics = iter.iter_metrics();
        SparseCellSetIter {
            inner: Box::new(iter),
            iter_metrics,
        }
    }

    /// Gather one batch of cell sets into the §4.4 contract. The `process`
    /// callback for the engine (shards already warmed).
    pub fn gather(
        &self,
        engine: &PrefetchEngine,
        plan: &SparseCellSetPlan,
    ) -> Result<SparseCellSetBatch> {
        let total_rows = plan.rows.len();
        if plan.file_ids.len() != total_rows || plan.role_tags.len() != total_rows {
            return Err(LoaderError::ConfigError {
                reason: "plan file_ids/rows/role_tags length mismatch".into(),
            });
        }

        // Plans arrive straight from (untrusted) Python. Validate structure up
        // front so a malformed plan returns a clean error instead of panicking
        // on an unchecked slice/index deep in the gather (mirrors the sibling
        // `IndexPlanLoader`, which validates row indices before reading).
        let n_files = self.n_files();
        for &fid in &plan.file_ids {
            if fid as usize >= n_files {
                return Err(LoaderError::ConfigError {
                    reason: format!("plan file_id {fid} out of range (n_files={n_files})"),
                });
            }
        }
        // `set_offsets` must be non-decreasing and bounded by `total_rows`, so
        // every `[lo..hi]` slice below is in range.
        let mut prev: i64 = 0;
        for (k, &off) in plan.set_offsets.iter().enumerate() {
            if off < 0 || off as usize > total_rows {
                return Err(LoaderError::ConfigError {
                    reason: format!("plan set_offsets[{k}]={off} out of range [0, {total_rows}]"),
                });
            }
            if k > 0 && off < prev {
                return Err(LoaderError::ConfigError {
                    reason: format!(
                        "plan set_offsets must be non-decreasing (set_offsets[{k}]={off} < {prev})"
                    ),
                });
            }
            prev = off;
        }
        // Row indices must be in range for their file, surfaced as `IndexError`
        // (consistent with `Experiment.gather_rows_sparse` and the pair loader).
        for (&fid, &row) in plan.file_ids.iter().zip(plan.rows.iter()) {
            let n_obs = engine.reader(fid).n_obs();
            if row as usize >= n_obs {
                return Err(LoaderError::IndexOutOfRange { idx: row, n_obs });
            }
        }

        let n_sets = plan.set_offsets.len().saturating_sub(1);

        let mut indptr: Vec<i64> = Vec::with_capacity(total_rows + 1);
        indptr.push(0);
        let mut indices: Vec<i32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        let mut cell_indices: Vec<u64> = Vec::with_capacity(total_rows);
        let mut out_file_ids: Vec<u32> = Vec::with_capacity(total_rows);
        let mut role_tags: Vec<i32> = Vec::with_capacity(total_rows);

        for s in 0..n_sets {
            let lo = plan.set_offsets[s] as usize;
            let hi = plan.set_offsets[s + 1] as usize;
            if hi <= lo {
                continue; // empty set — boundary only, no rows
            }
            let set_fids = &plan.file_ids[lo..hi];
            let set_rows = &plan.rows[lo..hi];
            let n = hi - lo;

            // Per-row gathered (and transformed) CSR, placed in set order.
            let mut per_row: Vec<Option<(Vec<i32>, Vec<f32>)>> = (0..n).map(|_| None).collect();

            let uniform = set_fids.iter().all(|&f| f == set_fids[0]);
            if uniform {
                // --- single-file fast path (the only current configuration) ---
                let fid = set_fids[0];
                let reader = engine.reader(fid);
                reader
                    .read_rows_with(set_rows, |orig_pos, idx, dat| {
                        // `orig_pos` is the position in `set_rows`, not the row id
                        // — the scatter fires in shard-grouped order. The row id is
                        // what keys the downsample RNG, so read it back through the
                        // request array.
                        per_row[orig_pos] =
                            Some(self.transform_row(fid, set_rows[orig_pos], idx, dat));
                        Ok(())
                    })
                    .map_err(LoaderError::FormatError)?;
            } else {
                // --- cross-file insurance path (requires global remap) ---
                if self.remap.is_none() {
                    return Err(LoaderError::ConfigError {
                        reason: "cross-file cell set requires global-vocab remap tables \
                                 (raw-local indices from different files are not comparable)"
                            .into(),
                    });
                }
                let mut by_file: HashMap<u32, Vec<(usize, u64)>> = HashMap::new();
                for (j, (&f, &r)) in set_fids.iter().zip(set_rows.iter()).enumerate() {
                    by_file.entry(f).or_default().push((j, r));
                }
                for (f, items) in by_file {
                    let reader = engine.reader(f);
                    let rs: Vec<u64> = items.iter().map(|&(_, r)| r).collect();
                    reader
                        .read_rows_with(&rs, |orig_pos, idx, dat| {
                            let (within_set_pos, src_row) = items[orig_pos];
                            per_row[within_set_pos] =
                                Some(self.transform_row(f, src_row, idx, dat));
                            Ok(())
                        })
                        .map_err(LoaderError::FormatError)?;
                }
            }

            for (j, row) in per_row.into_iter().enumerate() {
                let (ridx, rdat) = row.ok_or_else(|| {
                    LoaderError::ChannelError(format!(
                        "row {} (set {s}) was not scattered",
                        set_rows[j]
                    ))
                })?;
                indices.extend_from_slice(&ridx);
                data.extend_from_slice(&rdat);
                indptr.push(indices.len() as i64);
                cell_indices.push(set_rows[j]);
                out_file_ids.push(set_fids[j]);
                role_tags.push(plan.role_tags[lo + j]);
            }
        }

        Ok(SparseCellSetBatch {
            indptr,
            indices,
            data,
            shape: (cell_indices.len(), self.n_cols),
            cell_indices,
            file_ids: out_file_ids,
            set_offsets: plan.set_offsets.clone(),
            role_tags,
        })
    }

    /// Apply the optional remap, the non-negativity clip, the optional seeded
    /// downsample, and the value-only transforms to one gathered row.
    ///
    /// Stage order mirrors the Python reference's `finalize_csr_row`
    /// (`state3/src/state3/data/dataset.py:107-121`) exactly:
    /// remap → coalesce → **clip** → **downsample** → prune, then transforms.
    ///
    /// The clip lands **after** coalescing, not inside `remap_row`, because that is
    /// where the reference puts it and the two disagree: a `+5` and a `−3` mapping
    /// to the same global gene must coalesce to `2`, not to `5`.
    ///
    /// The clip runs unconditionally; the zero-prune does **not**. The reference
    /// prunes only inside its downsample branch, so with downsampling off a clipped
    /// `−1 → 0` stays an explicit zero and nnz is unchanged. Callers that count nnz
    /// therefore see the same structure with and without the clip.
    fn transform_row(&self, fid: u32, row: u64, idx: &[i32], dat: &[f32]) -> (Vec<i32>, Vec<f32>) {
        let (mut out_idx, mut out_dat) = match &self.remap {
            Some(tables) => remap_row(idx, dat, &tables[fid as usize]),
            None => (idx.to_vec(), dat.to_vec()),
        };
        crate::downsample::clip_negatives(&mut out_dat);
        if let Some(cfg) = &self.downsample {
            crate::downsample::downsample_row(
                &mut out_idx,
                &mut out_dat,
                cfg,
                cfg.identity_for(fid),
                row,
            );
        }
        if self.normalize || self.log1p {
            apply_sparse_transforms(&mut out_dat, self.normalize, self.log1p, self.target_sum);
        }
        (out_idx, out_dat)
    }
}

/// Run the per-cell collation kernel over an already-gathered, **global-vocab**
/// CSR batch (each row sorted-unique, as `remap_row` / state3's `finalize_csr_row`
/// produce). Pure compute — rayon over rows, no I/O. The query gene ids and
/// per-cell masks are supplied by the caller (Python's RNG side).
///
/// This is the entry the state3 "3A hybrid" actually uses: Python gathers (it
/// must, to sample the query from each set's expressed genes), then collates the
/// gathered CSR here. `set_offsets` / `cell_indices` / `file_ids` / `role_tags`
/// pass through unchanged. `enc_mask_positions` empty ⇒ no encoder query masking.
#[allow(clippy::too_many_arguments)]
pub fn collate_gathered(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    set_offsets: &[i64],
    cell_indices: Vec<u64>,
    file_ids: Vec<u32>,
    role_tags: Vec<i32>,
    k_dec: usize,
    query_gene_ids: &[i32],
    enc_mask_positions: &[u8],
    hide_readout: &[u8],
    n_measured: &[u32],
    scalars: &CollateScalars,
) -> Result<CollatedCellSetBatch> {
    // Fail loud (not `expect`-panic in the per-cell kernel) for a direct Rust
    // caller that selects the v4 PFlog mode without a pinned α — there is no
    // dataset here to estimate from.
    if scalars.mode == PreprocessMode::PflogRaw {
        match scalars.pflog_alpha {
            None => {
                return Err(LoaderError::ConfigError {
                    reason: "PflogRaw collate mode requires pflog_alpha (no dataset to estimate \
                             from here)"
                        .to_string(),
                });
            }
            Some(a) if a <= 0.0 || a.is_nan() || a.is_infinite() => {
                return Err(LoaderError::ConfigError {
                    reason: format!("pflog_alpha must be positive and finite, got {a}"),
                });
            }
            _ => {}
        }
    }
    let n_rows = cell_indices.len();
    let k_enc = scalars.k_enc;
    let n_sets = set_offsets.len().saturating_sub(1);

    let want = |what: &str, got: usize, exp: usize| {
        Err(LoaderError::ConfigError {
            reason: format!("collate_gathered: {what} len {got} != {exp}"),
        })
    };
    if indptr.len() != n_rows + 1 {
        return want("indptr", indptr.len(), n_rows + 1);
    }
    // Shape alone is not enough: the rayon loop below slices `indices[lo..hi]`
    // straight from these entries, so a non-monotonic or negative `indptr` panics
    // rather than erroring. Pre-existing gap on this entry point, closed here
    // because the check is shared with `downsample_counts_csr`.
    validate_indptr(indptr, data.len())?;
    if query_gene_ids.len() != n_sets * k_dec {
        return want("query_gene_ids", query_gene_ids.len(), n_sets * k_dec);
    }
    if hide_readout.len() != n_rows {
        return want("hide_readout", hide_readout.len(), n_rows);
    }
    if n_measured.len() != n_sets {
        return want("n_measured", n_measured.len(), n_sets);
    }
    let has_mask = !enc_mask_positions.is_empty();
    if has_mask && enc_mask_positions.len() != n_rows * k_dec {
        return want(
            "enc_mask_positions",
            enc_mask_positions.len(),
            n_rows * k_dec,
        );
    }

    // Row → set index, for per-set query / n_measured lookup.
    let mut row_set = vec![0usize; n_rows];
    for s in 0..n_sets {
        let lo = set_offsets[s] as usize;
        let hi = set_offsets[s + 1] as usize;
        for r in row_set.iter_mut().take(hi).skip(lo) {
            *r = s;
        }
    }

    let mut enc_ids = vec![0i64; n_rows * k_enc];
    let mut enc_counts = vec![0f32; n_rows * k_enc];
    let mut enc_mask = vec![0u8; n_rows * k_enc];
    let mut enc_pad = vec![0u8; n_rows * k_enc];
    let mut target = vec![0f32; n_rows * k_dec];
    let mut library = vec![0f32; n_rows];

    // On the loader's pool, never rayon's global registry. This kernel is
    // exposed as `pyscx.collate_cellset_gathered`, so a forked DataLoader
    // worker can call it with no dataset in hand and therefore no PID check in
    // front of it — and a global-pool dispatch from a forked child hangs
    // forever. See `crate::pool`.
    crate::pool::cpu_pool().install(|| {
        enc_ids
            .par_chunks_mut(k_enc)
            .zip(enc_counts.par_chunks_mut(k_enc))
            .zip(enc_mask.par_chunks_mut(k_enc))
            .zip(enc_pad.par_chunks_mut(k_enc))
            .zip(target.par_chunks_mut(k_dec))
            .zip(library.par_iter_mut())
            .enumerate()
            .for_each(|(r, (((((eid, ecnt), emask), epad), tgt), libslot))| {
                let s = row_set[r];
                let lo = indptr[r] as usize;
                let hi = indptr[r + 1] as usize;
                let cin = CellIn {
                    gene_ids: &indices[lo..hi],
                    raw: &data[lo..hi],
                    query: &query_gene_ids[s * k_dec..(s + 1) * k_dec],
                    enc_mask_positions: if has_mask {
                        Some(&enc_mask_positions[r * k_dec..(r + 1) * k_dec])
                    } else {
                        None
                    },
                    hide_readout: hide_readout[r] != 0,
                };
                let cfg = CollateConfig {
                    k_enc,
                    mode: scalars.mode,
                    target_sum: scalars.target_sum,
                    n_measured: n_measured[s] as usize,
                    pflog_alpha: scalars.pflog_alpha,
                    n_genes_total: scalars.n_genes_total,
                    lib_size_redef: scalars.lib_size_redef,
                };
                let mut out = CellOut {
                    enc_ids: eid,
                    enc_counts: ecnt,
                    enc_mask: emask,
                    enc_pad: epad,
                    target: tgt,
                };
                *libslot = collate_cell(&cin, &cfg, &mut out);
            });
    });

    Ok(CollatedCellSetBatch {
        encoder_gene_ids: enc_ids,
        encoder_counts: enc_counts,
        encoder_mask: enc_mask,
        encoder_pad_mask: enc_pad,
        target_counts: target,
        library_size: library,
        cell_indices,
        file_ids,
        set_offsets: set_offsets.to_vec(),
        role_tags,
        n_rows,
        k_enc,
        k_dec,
    })
}

/// Validate that `indptr` is a well-formed CSR row-pointer array over `nnz`
/// non-zeros: non-negative, non-decreasing, starting at 0 and ending at `nnz`.
///
/// `indptr.last() == nnz` alone is necessary but **not** sufficient, and the gap
/// is a panic rather than a wrong answer: `[0, 3, 2]` over 2 non-zeros passes a
/// last-element check, then slices `indices[0..3]` out of bounds. Entries are also
/// cast `i64 -> usize`, so a negative wraps to an enormous index and panics the
/// same way. Callers on the Python boundary hand us arbitrary numpy arrays, and
/// the crate's convention is to return an error on malformed input rather than
/// panic across FFI.
pub(crate) fn validate_indptr(indptr: &[i64], nnz: usize) -> Result<()> {
    let bad = |reason: String| Err(LoaderError::ConfigError { reason });
    match indptr.first() {
        None => return bad("indptr is empty (expected at least one entry)".into()),
        Some(&f) if f != 0 => {
            return bad(format!("indptr[0] must be 0, got {f}"));
        }
        _ => {}
    }
    for (r, w) in indptr.windows(2).enumerate() {
        if w[1] < w[0] {
            return bad(format!(
                "indptr must be non-decreasing: indptr[{}]={} < indptr[{r}]={}",
                r + 1,
                w[1],
                w[0]
            ));
        }
    }
    let last = *indptr.last().unwrap_or(&0);
    if last < 0 || last as usize != nnz {
        return bad(format!(
            "indptr's last entry {last} must equal the non-zero count {nnz}"
        ));
    }
    Ok(())
}

/// Map local gene ids to global via `local_to_global` (`-1` = drop), then sort
/// by global id and coalesce duplicates by summing — matching state3's
/// `local_to_global` + `_coalesce_gene_counts` (`dataset.py:381-391`), so the
/// output CSR row stays canonical (sorted, unique).
fn remap_row(indices: &[i32], data: &[f32], local_to_global: &[i32]) -> (Vec<i32>, Vec<f32>) {
    let mut pairs: Vec<(i32, f32)> = Vec::with_capacity(indices.len());
    for (&col, &val) in indices.iter().zip(data.iter()) {
        let g = if col >= 0 {
            local_to_global.get(col as usize).copied().unwrap_or(-1)
        } else {
            -1
        };
        if g >= 0 {
            pairs.push((g, val));
        }
    }
    pairs.sort_by_key(|&(g, _)| g);
    let mut out_idx: Vec<i32> = Vec::with_capacity(pairs.len());
    let mut out_dat: Vec<f32> = Vec::with_capacity(pairs.len());
    for (g, v) in pairs {
        if out_idx.last() == Some(&g) {
            *out_dat.last_mut().unwrap() += v;
        } else {
            out_idx.push(g);
            out_dat.push(v);
        }
    }
    (out_idx, out_dat)
}

/// Value-only, zero-preserving sparse transforms on a row's `data`, delegating
/// to the canonical dense helpers so the result is **bit-identical** to the
/// dense path on the stored nonzeros (SCX-DATA-LOADER §0). The dense transforms
/// are zero-preserving (`0 * factor = 0`; `ln(1+0) = 0`) and `normalize` derives
/// its scale from the row sum — which, over a sparse row's nonzeros, equals the
/// full row sum since the absent zeros contribute nothing. `normalize` rounds
/// `(v as f64 * factor) as f32` per element (not an f32 scale multiply), exactly
/// as [`crate::normalize::normalize_dense_row`]; `log1p` **is**
/// [`crate::normalize::log1p_dense_row`], which computes `(v.max(0.0) + 1.0).ln()`
/// — not `f32::ln_1p`, as an earlier version of this comment claimed. Its clip
/// is a no-op here: `transform_row` has already run
/// [`crate::downsample::clip_negatives`] over the same buffer.
fn apply_sparse_transforms(data: &mut [f32], normalize: bool, log1p: bool, target_sum: f64) {
    if normalize {
        crate::normalize::normalize_dense_row(data, target_sum);
    }
    if log1p {
        crate::normalize::log1p_dense_row(data);
    }
}

/// Iterator returned by [`SparseCellSetLoader::iter_with_plans`].
///
/// **An adapter, not an implementation** — the sibling of
/// [`crate::index_plan::IndexPlanIter`], and for the same two reasons. The
/// plan-pull thread, the lookahead queue, the per-plan shard prefetch and the
/// counters all live in [`crate::plan_engine`]; this exists because
/// `PlanPrefetchIter`'s closure type parameters are unnameable (the `process`
/// closure captures an `Arc<SparseCellSetLoader>`), and because boxing to
/// `dyn Iterator` would erase the inherent `iter_metrics()` that
/// `SparseCellSetBatchIter.metrics()` needs.
///
/// Being a named `Iterator` rather than a `(Box<dyn Iterator>, Arc<IterMetrics>)`
/// tuple keeps `for batch in loader.iter_with_plans(..)` working for Rust
/// callers, which the tuple form broke.
pub struct SparseCellSetIter {
    inner: Box<dyn Iterator<Item = Result<SparseCellSetBatch>> + Send + Sync>,
    iter_metrics: Arc<IterMetrics>,
}

impl SparseCellSetIter {
    /// Cloneable handle to this iter's prefetch counters. Sample at any time —
    /// atomics are `Relaxed`, no locks. Stays valid after the iterator is
    /// drained and dropped.
    pub fn iter_metrics(&self) -> Arc<IterMetrics> {
        Arc::clone(&self.iter_metrics)
    }
}

impl Iterator for SparseCellSetIter {
    type Item = Result<SparseCellSetBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

#[cfg(test)]
#[path = "sparse_cellset_tests.rs"]
mod tests;
