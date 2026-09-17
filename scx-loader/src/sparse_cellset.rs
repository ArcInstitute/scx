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
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;
use scx_format_io::freshness::FileIdentity;
use scx_format_io::{
    Admit, BackedCsrIndex, BackedCsrReader, CacheMetrics, ScxReader, SharedShardCache,
};

use crate::error::{LoaderError, Result};
use crate::plan_engine::{IterMetrics, PrefetchEngine};
use crate::reader_registry::FileSlot;
use crate::sparse_cellset_collate::{
    collate_cell, CellIn, CellOut, CollateConfig, PreprocessMode, RowMask, SetQueryIndex,
};
use crate::tokenize::Scratch;

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
    /// `1` where a `target_counts` slot is padding rather than a real query
    /// position — all zero unless per-row queries are in use and some row's
    /// query is shorter than `k_dec`. Without it a padded `0.0` is
    /// indistinguishable from a real zero target.
    pub target_pad_mask: Vec<u8>, // [n_rows * k_dec]
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
    /// Caller-declared cap on simultaneously-resident readers, or `None` for
    /// "open everything" (the default, and what every caller got before this
    /// existed). Reported through `SparseCellSetDataset.memory_budget()`.
    reader_limit: Option<usize>,
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
    /// Caller-declared upper bound on rows per plan, or `None` (uncharged).
    max_plan_rows: Option<usize>,
    /// Manifest-wide mean non-zeros per row — the density the batch charge and
    /// its `memory_budget()` report are computed at.
    mean_nnz_per_row: f64,
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
///
/// Also returns the manifest's mean non-zeros per row, which the same walk
/// already accumulates and used to discard. That is the per-row size a gathered
/// batch is charged at (see [`SparseCellSetBudgetModel`]); deriving it here
/// rather than in a second catalog walk keeps one definition of which shards
/// count, and the divisor rule below applies to both.
#[derive(Default)]
struct ShardStatsAccum {
    total_nnz: u64,
    total_rows: u64,
    // Only shards that CONTRIBUTED to the totals may count toward the divisor.
    // Counting stat-less shards in the denominator averages their 0 bytes into
    // the result, *under*-estimating the per-shard size — which then
    // *over*-estimates how many shards the byte budget affords and makes the
    // sizing diagnostic under-warn on exactly the files whose catalogs are
    // incomplete. Flagged independently by all three round-2 reviewers.
    n_counted: u64,
}

impl ShardStatsAccum {
    /// Fold one file in. Per file rather than over a slice because the manifest
    /// scan may hold only `reader_limit` readers at once and has no slice of
    /// all of them to hand at the end.
    fn add(&mut self, reader: &ScxReader) {
        for e in reader.catalog().shards_sorted() {
            if let Some(s) = e.stats.as_ref() {
                self.total_nnz += s.nnz;
                self.total_rows += s.row_end - s.row_start;
                self.n_counted += 1;
            }
        }
    }

    fn finish(&self) -> (usize, f64) {
        // No shard carried stats ⇒ the size is genuinely unknown. Returning 0
        // is the signal the constructor reads as "the byte cap tells us
        // nothing", falling back to the count cap rather than to a fabricated
        // average.
        let avg_bytes = self
            .total_nnz
            .saturating_mul(8)
            .saturating_add(self.total_rows.saturating_mul(8))
            .checked_div(self.n_counted)
            .unwrap_or(0) as usize;
        // Rows, not shards, in this divisor: the mean is per row of the
        // manifest, so shards of unequal height must not be weighted equally.
        // 0.0 when no shard carried stats, matching `avg_bytes`'s "genuinely
        // unknown" signal.
        let mean_nnz_per_row = if self.total_rows == 0 {
            0.0
        } else {
            self.total_nnz as f64 / self.total_rows as f64
        };
        (avg_bytes, mean_nnz_per_row)
    }
}

#[cfg(test)]
fn avg_shard_decoded_bytes(readers: &[ScxReader]) -> (usize, f64) {
    let mut acc = ShardStatsAccum::default();
    for r in readers {
        acc.add(r);
    }
    acc.finish()
}

/// Capacity to pre-size a gathered batch's `indices` / `data` to.
///
/// `rows x mean_nnz_per_row`, biased up by an eighth. Shared with the tests so
/// they assert against the policy rather than restating its arithmetic, which
/// is what lets them keep proving "no reallocation happened" when the bias
/// changes.
/// Process-wide switch for the **multi-set batch executor** (W11).
///
/// Default `plan`: one occurrence table over the whole batch, one read per file
/// over its deduplicated rows, each unique `(file, row)` transformed once.
/// `SCX_CELLSET_EXECUTOR=set` restores the pre-W11 per-set walk — one read per
/// set, a per-set `Vec<Option<(Vec<i32>, Vec<f32>)>>`, and an
/// `extend_from_slice` copy of every row — which is the same-build A/B arm for
/// the capture, exactly as `SCX_ROW_GROUP_ADMIT=plan` is for the admission
/// policy.
///
/// Anything other than `set` (case-insensitive) is `plan`, including an unset
/// variable and a typo: a misspelled arm must not silently restore the old
/// executor in a capture that then reports it as the default.
pub(crate) fn whole_plan_executor() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !std::env::var("SCX_CELLSET_EXECUTOR").is_ok_and(|v| v.eq_ignore_ascii_case("set"))
    })
}

pub(crate) fn presize_nnz(rows: usize, mean_nnz_per_row: f64) -> usize {
    let est = (rows as f64 * mean_nnz_per_row).ceil() as usize;
    est.saturating_add(est / 8)
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
/// * **Charged only if declared**: the gathered batch. Unlike `IndexPlanLoader`
///   this path has no `max_plan_size` — a plan's output size is caller-controlled
///   and unbounded, so there is nothing fixed to cost until the caller says how
///   wide its plans get. `max_plan_rows` is that declaration, and it is opt-in
///   on purpose: charging a guessed default would shrink the cache on every
///   existing caller, and on a file where the adaptive cap already binds
///   (census_500k affords 22 of the 31 shards it wants) that is throughput
///   traded away for an estimate nobody asked for.
/// * **Charged with it**: the batch executor's unique-row read, on the
///   configurations that cannot avoid it — a remap, a downsample, or a manifest
///   of more than one file; see `SparseCellSetBudgetModel::transient_bytes`.
/// * **Not charged**: the batch's per-row transients, and nothing at all when
///   `max_plan_rows` is `None`.
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
    /// One gathered batch at the caller's declared `max_plan_rows`, or 0.
    batch_bytes: usize,
    /// Bytes the batch executor's unique-row read holds **beside** the batch.
    ///
    /// W11 reads a plan's deduplicated rows into one exactly-sized `ScxCsr` and
    /// then writes the batch from it, so for the span of a gather two buffers
    /// are live. Not always, though: a single-file plan with no repeated rows
    /// and no length-changing transform is handed the read's own buffers as the
    /// batch, and holds one.
    ///
    /// Which of the two a gather takes is decided per *plan* — a repeated row
    /// forces the general path on any configuration — so the model charges on
    /// the three facts it knows at construction: a remap or a downsample makes
    /// every plan take it, and so does a manifest of more than one file. On a
    /// single-file raw-local loader the term is 0 and a plan that repeats a row
    /// pays an uncharged transient, which is a **floor rather than a bound** and
    /// is said here rather than left to be discovered. The charged figure is one
    /// batch, which is its ceiling: the unique rows of a plan are at most its
    /// rows, and are fewer exactly when the duplicates that force this path
    /// exist.
    transient_bytes: usize,
}

impl SparseCellSetBudgetModel {
    /// Bytes one gathered batch of `rows` occupies, at `mean_nnz_per_row`.
    ///
    /// The §4.4 batch is CSR, so this is the `csr_component_bytes` shape —
    /// `nnz × 8` for the i32 indices plus the f32 data, and `(rows + 1) × 8` for
    /// the i64 indptr — not `IndexPlanLoader`'s dense `2 × rows × n_cols × 4`.
    /// The per-row obs and id arrays are left out: they are two orders of
    /// magnitude smaller than the CSR at any realistic density, and a term that
    /// small would lend the estimate a precision it does not have.
    fn batch_bytes_for(max_plan_rows: usize, mean_nnz_per_row: f64) -> usize {
        // `presize_nnz`, not the raw mean. It was chosen because the gather
        // allocated the biased figure and charging the unbiased one would have
        // under-reported the batch by exactly the eighth the bias adds. W11
        // allocates exactly, so the eighth is now headroom rather than a model
        // of the allocation — kept, because a manifest-wide mean is an estimate
        // and the errors are not symmetric: under-reporting a batch is what
        // makes `max_memory_mb` mean nothing.
        let nnz = presize_nnz(max_plan_rows, mean_nnz_per_row);
        nnz.saturating_mul(8)
            .saturating_add(max_plan_rows.saturating_add(1).saturating_mul(8))
    }
}

impl crate::budget::BudgetModel for SparseCellSetBudgetModel {
    type Params = SparseCellSetParams;

    fn estimate(&self, p: SparseCellSetParams) -> crate::budget::BudgetBreakdown {
        crate::budget::BudgetBreakdown::new(
            p.cache_shards.saturating_mul(self.shard_decoded_bytes),
            // The existing `batch_buffer_bytes` slot, rather than a seventh key:
            // every loader class reports the same six, and the breakdown's shape
            // is pinned by `test_every_budget_carries_the_same_breakdown`.
            self.batch_bytes,
            0,
            self.transient_bytes,
            crate::budget::PYTHON_OVERHEAD_BYTES,
        )
    }

    fn reduce(&self, p: SparseCellSetParams) -> Option<SparseCellSetParams> {
        (p.cache_shards > 1).then(|| SparseCellSetParams {
            cache_shards: p.cache_shards - 1,
        })
    }
}

/// Make a manifest entry independent of the process working directory.
///
/// A reopen happens at an arbitrary later time, so a relative entry would be
/// resolved against whatever cwd the process has *then* — a bounded-mode
/// regression against `main`, where every mmap was established in the
/// constructor and held, so cwd could not matter. Reproduced on #536 by
/// codex - gpt-5.6-sol and confirmed by Cursor Agent and Antigravity.
///
/// Hand-rolled rather than `std::path::absolute`, which is Rust 1.79 and would
/// have silently broken the >= 1.78 floor that `README.md` and
/// `docs/development.md` document and no `rust-version` key enforces — so no CI
/// leg would have caught it. **Caught on #536 by codex.**
///
/// Deliberately not `canonicalize`: this removes the cwd dependence without
/// resolving symlinks, which would change which file a repointed link names.
/// A failure is an error, never a silent fallback to the relative spelling —
/// that spelling is the bug.
fn absolute_manifest_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    let cwd = std::env::current_dir().map_err(|e| LoaderError::ConfigError {
        reason: format!(
            "cannot resolve the relative manifest entry {} — the process working \
             directory is unreadable ({e}); pass absolute paths",
            path.display()
        ),
    })?;
    Ok(cwd.join(path))
}

/// One pass over the manifest, holding at most `reader_limit` files open.
///
/// Everything the constructor needs from a file — the CSR-range validation, the
/// column count, the catalog's shard stats, the row count and the shard index —
/// comes from its catalog, so there is no way to learn it without opening the
/// file. What a bounded scan changes is not *whether* each file is opened but
/// how many are open **at once**, which is the term that costs ~104 kB of
/// resident memory apiece (see [`crate::reader_registry`]).
struct ManifestScan {
    files: Vec<FileSlot>,
    /// The handles the scan kept: all of them at `reader_limit = None`, the
    /// first `limit` otherwise. First rather than most-recent because a scan
    /// has no access pattern to learn from yet, and the alternative — keeping
    /// the *last* `limit` — would evict exactly the files a plan starting at
    /// `file_id` 0 asks for next.
    retained: Vec<(u32, ScxReader)>,
    n_vars_max: usize,
    stats: ShardStatsAccum,
    /// Folded here, while every file is open anyway, so nothing has to reopen
    /// the manifest later to answer it. `None` when the caller did not ask for
    /// the block-index route, which is its only consumer.
    any_framed: Option<bool>,
}

impl ManifestScan {
    /// Scan readers the caller has already opened, keeping every one.
    ///
    /// The path behind `SparseCellSetLoader::new`, which exists for callers
    /// that hold `ScxReader`s rather than paths. No slot carries an index or an
    /// identity: nothing here can ever be evicted, so nothing can be reopened.
    fn from_open_readers(readers: Vec<ScxReader>, want_framing: bool) -> Result<Self> {
        let mut scan = Self {
            files: Vec::with_capacity(readers.len()),
            retained: Vec::with_capacity(readers.len()),
            n_vars_max: 0,
            stats: ShardStatsAccum::default(),
            any_framed: want_framing.then_some(false),
        };
        for (fid, reader) in readers.into_iter().enumerate() {
            scan.absorb(
                fid as u32, reader, /* retain */ true, /* reopenable */ false,
            )?;
        }
        Ok(scan)
    }

    /// Open each path in turn, keeping at most `limit` of the handles.
    fn from_paths(paths: &[PathBuf], limit: Option<usize>, want_framing: bool) -> Result<Self> {
        let mut scan = Self {
            files: Vec::with_capacity(paths.len()),
            retained: Vec::with_capacity(limit.unwrap_or(paths.len()).min(paths.len())),
            n_vars_max: 0,
            stats: ShardStatsAccum::default(),
            any_framed: want_framing.then_some(false),
        };
        for (fid, path) in paths.iter().enumerate() {
            // Resolved BEFORE the open, and the resolved path is what gets
            // opened. Doing it afterwards left a window in which a concurrent
            // `set_current_dir` mapped one directory's file and recorded the
            // other's as the reopen target. **Review on #536 (codex).**
            let path = absolute_manifest_path(path)?;
            let reader = ScxReader::open(&path).map_err(|e| LoaderError::ConfigError {
                reason: format!("failed to open {}: {e}", path.display()),
            })?;
            let retain = limit.is_none_or(|k| scan.retained.len() < k);
            scan.absorb(fid as u32, reader, retain, /* reopenable */ true)?;
        }
        Ok(scan)
    }

    fn absorb(
        &mut self,
        file_id: u32,
        reader: ScxReader,
        retain: bool,
        reopenable: bool,
    ) -> Result<()> {
        // Same reason as `IndexPlanLoader`: this loader resolves cells by
        // global obs row through `BackedCsrReader`, so a multimodal file's
        // flattened shard list would answer from an arbitrary modality. Checked
        // during the scan, while the file is open, so a manifest with a bad
        // file still fails at construction rather than at first gather.
        crate::pipeline::ensure_csr_ranges_are_readable(
            &reader,
            None,
            &format!("SparseCellSetLoader (file {file_id})"),
        )?;
        self.n_vars_max = self.n_vars_max.max(reader.n_vars() as usize);
        self.stats.add(&reader);
        // Folded while the file is open. Answering it later meant reopening
        // every slot the scan had just closed, on a bounded manifest, to
        // produce a constructor warning. **Review on #536.**
        if self.any_framed == Some(false) && reader.any_csr_shard_framed() {
            self.any_framed = Some(true);
        }
        self.files.push(FileSlot {
            // Already absolute: `from_paths` resolved it before opening, and
            // `from_open_readers` is handed readers whose paths the caller
            // chose. See `absolute_manifest_path`.
            path: reader.path().to_path_buf(),
            n_obs: reader.n_obs(),
            index: BackedCsrIndex::from_catalog(reader.catalog()),
            // Stamped whenever the registry could reopen at all — which is
            // every path-built loader, not only a limited one. `open(paths, …,
            // None)` never evicts and so never reopens today, but it holds the
            // recipe that would, and a slot without an identity is a hole
            // waiting for whoever next changes when eviction runs. Free: both
            // halves come from the reader already in hand.
            identity: reopenable.then(|| FileIdentity::of(&reader)),
        });
        if retain {
            self.retained.push((file_id, reader));
        }
        Ok(())
    }

    fn n_files(&self) -> usize {
        self.files.len()
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
        max_plan_rows: Option<usize>,
    ) -> Result<Arc<Self>> {
        Self::from_scan(
            ManifestScan::from_open_readers(scx_readers, scatter_block_index)?,
            cache_shards,
            bytes_budget,
            lookahead,
            remap,
            n_global_genes,
            normalize,
            log1p,
            target_sum,
            downsample,
            scatter_block_index,
            max_plan_rows,
            None,
        )
    }

    /// Build a loader over a manifest of **paths**, opening at most
    /// `reader_limit` of them at a time and keeping at most that many resident.
    ///
    /// `reader_limit = None` is the default and opens everything, exactly as
    /// [`SparseCellSetLoader::new`] does — identical sizing, identical gather
    /// output, and nothing ever reopened. A `Some(k)` trades resident memory
    /// for reopens: each open reader costs ~104 kB, over 90 % of it the parsed
    /// `FullCatalog`, which at a 26 k-file manifest is ~2.8-3.2 GB per process.
    /// A reopen re-parses that catalog, which is what makes the saving real —
    /// see [`crate::reader_registry`].
    ///
    /// Every other argument means what it does on `new`.
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        paths: Vec<PathBuf>,
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
        max_plan_rows: Option<usize>,
        reader_limit: Option<usize>,
    ) -> Result<Arc<Self>> {
        if reader_limit == Some(0) {
            return Err(LoaderError::ConfigError {
                reason: "reader_limit must be >= 1 (a gather needs at least one open reader); \
                         pass None to keep every file open"
                    .into(),
            });
        }
        Self::from_scan(
            ManifestScan::from_paths(&paths, reader_limit, scatter_block_index)?,
            cache_shards,
            bytes_budget,
            lookahead,
            remap,
            n_global_genes,
            normalize,
            log1p,
            target_sum,
            downsample,
            scatter_block_index,
            max_plan_rows,
            reader_limit,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_scan(
        scan: ManifestScan,
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
        max_plan_rows: Option<usize>,
        reader_limit: Option<usize>,
    ) -> Result<Arc<Self>> {
        let n_files = scan.n_files();
        if let Some(cfg) = &downsample {
            cfg.validate()?;
            // An empty table means "no stable per-file identity", which keys on
            // `(seed, method, row)` alone. That is fine for one file and ambiguous
            // for several — two files' row 5 would share a draw. Refuse it rather
            // than fall back to `file_id`, which would be construction-order
            // keying: a reordered manifest or a debugging subset silently redrawing
            // every cell while producing perfectly plausible output.
            if cfg.file_identities.is_empty() && n_files > 1 {
                return Err(LoaderError::ConfigError {
                    reason: format!(
                        "downsample over {} files requires file_identities (one stable \
                         per-file id, e.g. scx_loader::downsample::file_identity(path)); \
                         without them two files' row N would share a draw",
                        n_files
                    ),
                });
            }
            if !cfg.file_identities.is_empty() && cfg.file_identities.len() != n_files {
                return Err(LoaderError::ConfigError {
                    reason: format!(
                        "downsample file_identities has {} entries but there are {} files",
                        cfg.file_identities.len(),
                        n_files
                    ),
                });
            }
        }
        if let Some(tables) = &remap {
            if tables.len() != n_files {
                return Err(LoaderError::ConfigError {
                    reason: format!(
                        "remap has {} tables but there are {} files",
                        tables.len(),
                        n_files
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
            (None, _) => scan.n_vars_max,
        };
        // Both from the scan's incremental fold over each file's catalog: the
        // adaptive budget path needs these, and a bounded scan has no slice of
        // all the readers left to walk.
        let (shard_decoded_bytes, mean_nnz_per_row) = scan.stats.finish();
        // `None` ⇒ 0 ⇒ the breakdown and the resolved cache are byte-identical
        // to what this loader produced before the term existed. Only a caller
        // who declares how wide its plans get pays for one.
        let batch_bytes = max_plan_rows
            .map(|rows| SparseCellSetBudgetModel::batch_bytes_for(rows, mean_nnz_per_row))
            .unwrap_or(0);
        let model = SparseCellSetBudgetModel {
            shard_decoded_bytes,
            batch_bytes,
            // Charged only where a gather cannot take the fast path — see
            // the field's doc.
            transient_bytes: if remap.is_some() || downsample.is_some() || n_files > 1 {
                batch_bytes
            } else {
                0
            },
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
        let shared = SharedShardCache::new(affordable_cache_shards, enforced_cache_bytes);
        let retained: Vec<(u32, Arc<BackedCsrReader>)> = scan
            .retained
            .into_iter()
            .map(|(fid, r)| {
                (
                    fid,
                    crate::reader_registry::wrap_reader(r, fid, &shared, scatter_block_index),
                )
            })
            .collect();
        // Read back the one aggregate handle `wrap_reader`'s `enable_metrics`
        // installed on the shared cache, as `PrefetchEngine::new` does. A
        // zeroed default only when the manifest retained nothing, which the
        // `reader_limit >= 1` check above makes unreachable for a non-empty
        // manifest.
        let cache_metrics = retained
            .iter()
            .find_map(|(_, r)| r.metrics().cloned())
            .unwrap_or_else(|| Arc::new(CacheMetrics::default()));
        let registry = crate::reader_registry::ReaderRegistry::from_scan(
            scan.files,
            retained,
            reader_limit,
            shared,
            scatter_block_index,
            scan.any_framed,
        );
        let engine = PrefetchEngine::over_registry(registry, lookahead, cache_metrics);
        Ok(Arc::new(SparseCellSetLoader {
            engine,
            reader_limit,
            remap,
            normalize,
            log1p,
            target_sum,
            n_cols,
            cache_bytes_budget,
            cache_shards,
            affordable_cache_shards,
            max_plan_rows,
            mean_nnz_per_row,
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

    /// Caller-declared rows-per-plan bound, or `None` if the batch is uncharged.
    pub fn max_plan_rows(&self) -> Option<usize> {
        self.max_plan_rows
    }

    /// Manifest-wide mean non-zeros per row (0.0 when no shard carries stats).
    pub fn mean_nnz_per_row(&self) -> f64 {
        self.mean_nnz_per_row
    }

    /// Cap on simultaneously-running shard decodes in the prefetch engine.
    pub fn max_blocking_threads(&self) -> usize {
        self.engine.max_blocking_threads()
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
    /// Two terms are non-zero by default, and that is the model, not an
    /// omission: the shard cache **is** this loader's budget — no plan-tuple
    /// staging and no per-batch obs scratch on the gather path — plus the
    /// interpreter/numpy/Arrow constant every path pays. A caller that declares
    /// `max_plan_rows` adds a third, `batch_buffer_bytes`: one gathered CSR
    /// batch at that width and the manifest's mean density.
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
            .map(|fid| self.engine.registry().n_shards(fid as u32))
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
            .map(|(f, rs)| self.engine.registry().shards_touched(f, &rs))
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

    /// Cap on simultaneously-resident readers, or `None` for "open everything".
    pub fn reader_limit(&self) -> Option<usize> {
        self.reader_limit
    }

    /// Registry counters (opens / evictions / residency), cumulative since
    /// construction. At `reader_limit = None` `opens` equals the manifest size
    /// and the other two never move — which is the shape of the assertion that
    /// the default path really does nothing new.
    pub fn reader_metrics(&self) -> Arc<crate::reader_registry::ReaderMetrics> {
        self.engine.registry().metrics()
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
        // Width is refused in the STREAM, so an over-wide plan never reaches
        // `spawn_prefetches` — which would otherwise size it and await real
        // shard decodes before the gather rejected it.
        let gate = Arc::clone(&self);
        let plans =
            plans.map(move |p| p.and_then(|plan| gate.check_plan_width(&plan).map(|()| plan)));
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
            move |eng: &PrefetchEngine, plan: SparseCellSetPlan, admit_row_groups: Admit| {
                loader.gather_admitting(eng, &plan, Some(admit_row_groups))
            },
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

    /// Refuse a plan wider than the caller's declared `max_plan_rows`.
    ///
    /// **Review on #535 round 2 (codex).** Called BEFORE any admission sizing
    /// or prefetch I/O on both public routes. The first version checked inside
    /// `gather_admitting`, which meant an over-wide plan had already allocated
    /// its dedup set and — on the iterator path — spawned and awaited shard
    /// decodes before the refusal it was promised. A limit that only takes
    /// effect after the work it was meant to prevent is not a limit.
    fn check_plan_width(&self, plan: &SparseCellSetPlan) -> Result<()> {
        if let Some(limit) = self.max_plan_rows {
            let total_rows = plan.rows.len();
            if total_rows > limit {
                return Err(LoaderError::ConfigError {
                    reason: format!(
                        "plan has {total_rows} rows, exceeding max_plan_rows {limit} \
                         (raise max_plan_rows at construction, or split the plan; \
                         the shard cache was sized against {limit})"
                    ),
                });
            }
        }
        Ok(())
    }

    /// Gather one batch of cell sets into the §4.4 contract, synchronously.
    ///
    /// Takes **one** row-group admission verdict over the whole plan, sized
    /// against the entire byte budget, and carries it into every set's read —
    /// the same shape the engine-driven path ([`Self::iter_with_plans`]) uses,
    /// differing only in that it has no lookahead window to divide the budget
    /// by. Deciding per set instead would let a plan whose sets each fit, but
    /// whose union does not, evict its own row groups set by set.
    pub fn gather(&self, plan: &SparseCellSetPlan) -> Result<SparseCellSetBatch> {
        self.check_plan_width(plan)?;
        // One verdict, inline: the prefetcher makes the same two calls and
        // compares against its divided share, so the only thing that differed
        // was the comparison — a named wrapper around it added a hop and no
        // decision.
        let per_shard = self
            .engine
            .bucket_plan_rows(plan.file_ids.iter().copied().zip(plan.rows.iter().copied()));
        let (planned, budget) = self.engine.plan_footprint(&per_shard, None)?;
        self.gather_admitting(&self.engine, plan, Some(Admit::from(planned <= budget)))
    }

    /// [`Self::gather`] with the row-group admission decided by the caller.
    /// A plan is gathered one `read_rows_with` call per set (and per file on
    /// the cross-file path), so only the caller sees the plan's whole working
    /// set: the prefetch engine sums it over every file and shard and passes
    /// its verdict in, so sets that fit one at a time but not together do not
    /// churn the LRU. The `process` callback for the engine (shards already
    /// warmed).
    pub(crate) fn gather_admitting(
        &self,
        engine: &PrefetchEngine,
        plan: &SparseCellSetPlan,
        admit_row_groups: Option<Admit>,
    ) -> Result<SparseCellSetBatch> {
        self.validate_plan(engine, plan)?;
        if whole_plan_executor() {
            self.gather_whole_plan(engine, plan, admit_row_groups)
        } else {
            self.gather_per_set(engine, plan, admit_row_groups)
        }
    }

    /// Positions of `plan` the gather emits: `[set_offsets[0], set_offsets[n])`.
    ///
    /// `set_offsets` is validated non-decreasing and in range but is **not**
    /// required to start at 0 or to end at `rows.len()`, so a plan can name rows
    /// no set covers. The per-set walk never reads those — it iterates set
    /// spans — and the whole-plan executor must not either, or a plan with a
    /// partial cover would read rows the other arm does not and move the
    /// admission footprint with it.
    fn emitted_range(plan: &SparseCellSetPlan) -> (usize, usize) {
        let n_sets = plan.set_offsets.len().saturating_sub(1);
        if n_sets == 0 {
            return (0, 0);
        }
        (
            plan.set_offsets[0] as usize,
            plan.set_offsets[n_sets] as usize,
        )
    }

    /// A set whose rows span more than one file needs a global vocabulary:
    /// raw-local indices from different files are not comparable.
    ///
    /// Checked over every set **before** any read, where the per-set walk
    /// discovers it partway through and has already done the earlier sets' I/O.
    /// Same error, same message; only the wasted work differs.
    fn check_cross_file_sets(&self, plan: &SparseCellSetPlan) -> Result<()> {
        if self.remap.is_some() {
            return Ok(());
        }
        let n_sets = plan.set_offsets.len().saturating_sub(1);
        for s in 0..n_sets {
            let lo = plan.set_offsets[s] as usize;
            let hi = plan.set_offsets[s + 1] as usize;
            if hi <= lo {
                continue;
            }
            let f0 = plan.file_ids[lo];
            if plan.file_ids[lo..hi].iter().any(|&f| f != f0) {
                return Err(LoaderError::ConfigError {
                    reason: "cross-file cell set requires global-vocab remap tables \
                                 (raw-local indices from different files are not comparable)"
                        .into(),
                });
            }
        }
        Ok(())
    }

    /// W11 — the multi-set batch executor.
    ///
    /// One occurrence table over the whole plan, **one read per file** over that
    /// file's deduplicated rows, each unique `(file, row)` transformed once, and
    /// the output allocated exactly once from the measured lengths. Against the
    /// per-set walk this removes, per plan: `n_sets - n_files` reads, two `Vec`
    /// allocations per row, one full copy of the batch, and every duplicate
    /// occurrence's decode and transform.
    ///
    /// **Dedup cannot change the output.** [`Self::transform_row`] is a pure
    /// function of `(file_id, row, indices, data)` — the remap table is indexed
    /// by `file_id` and the downsample draw is keyed on the file's *content*
    /// identity and the *physical* row ([`crate::seed::row_seed`]), never on
    /// batch position — so two occurrences of one `(file, row)` are bit-identical
    /// and one may be memcpy'd from the other.
    fn gather_whole_plan(
        &self,
        engine: &PrefetchEngine,
        plan: &SparseCellSetPlan,
        admit_row_groups: Option<Admit>,
    ) -> Result<SparseCellSetBatch> {
        self.check_cross_file_sets(plan)?;
        let (emit_lo, emit_hi) = Self::emitted_range(plan);
        let occ = Occurrences::build(plan, emit_lo, emit_hi);
        let n_out = emit_hi - emit_lo;

        // --- one read per file, over its unique rows, exact spans -----------
        //
        // `read_row_indices_with_admission` carries the plan's verdict and does
        // the indptr-only prescan itself, so the rows land in one exactly-sized
        // allocation with no estimate. Widening from per-set to per-plan is also
        // what lets the chunked parallel group decode overlap groups across
        // shards rather than within one set's worth.
        let mut per_file: Vec<scx_sparse::ScxCsr> = Vec::with_capacity(occ.by_file.len());
        for (fid, slots) in &occ.by_file {
            let rows: Vec<u64> = slots.iter().map(|&s| occ.slots[s as usize].1).collect();
            let reader = engine.lease(*fid)?;
            per_file.push(
                reader
                    .read_row_indices_with_admission(&rows, admit_row_groups.as_ref())
                    .map_err(LoaderError::FormatError)?,
            );
        }

        // --- fast path: the read IS the batch --------------------------------
        //
        // One file, no duplicate occurrences, and no stage that could shrink a
        // row: the rows are already in plan order at their final lengths, so
        // the `ScxCsr` is moved out whole and the gather's whole output is the
        // one allocation `read_row_indices_with_admission` already made. Taken
        // BEFORE the transform bookkeeping below, which would otherwise cost
        // four `Vec`s per plan this path does not need.
        let no_shrink = self.remap.is_none() && self.downsample.is_none();
        if no_shrink && occ.identity && per_file.len() == 1 {
            let mut csr = per_file.pop().expect("one bucket");
            // Elementwise over the whole batch: `clip_negatives` is per value,
            // so a row-by-row walk would be the same work in more passes.
            crate::downsample::clip_negatives(&mut csr.data);
            if self.normalize || self.log1p {
                // Per row, because `normalize` divides by that row's library
                // size. In place, into the spans the indptr already names.
                let bounds: Vec<(usize, usize)> = (0..n_out)
                    .map(|l| (csr.indptr[l] as usize, csr.indptr[l + 1] as usize))
                    .collect();
                let rows = spans_mut(&mut csr.data, &bounds);
                crate::pool::cpu_pool().install(|| {
                    rows.into_par_iter().for_each(|d| {
                        apply_sparse_transforms(d, self.normalize, self.log1p, self.target_sum)
                    })
                });
            }
            return Ok(SparseCellSetBatch {
                indptr: csr.indptr,
                indices: csr.indices,
                data: csr.data,
                shape: (n_out, self.n_cols),
                cell_indices: plan.rows[emit_lo..emit_hi].to_vec(),
                file_ids: plan.file_ids[emit_lo..emit_hi].to_vec(),
                set_offsets: plan.set_offsets.clone(),
                role_tags: plan.role_tags[emit_lo..emit_hi].to_vec(),
            });
        }

        // --- transform each unique row once, in place ------------------------
        //
        // Every stage either shrinks a row or preserves it (`remap_row` drops
        // `-1` sentinels and coalesces; `downsample_row` truncates; the clip and
        // the value transforms are elementwise), so a row's output always fits
        // its own raw span and no second buffer is needed to hold it.
        let mut lens: Vec<u32> = vec![0; occ.slots.len()];
        for (bucket, (fid, slots)) in occ.by_file.iter().enumerate() {
            let csr = &mut per_file[bucket];
            let bounds: Vec<(usize, usize)> = (0..slots.len())
                .map(|l| (csr.indptr[l] as usize, csr.indptr[l + 1] as usize))
                .collect();
            let idx_spans = spans_mut(&mut csr.indices, &bounds);
            let dat_spans = spans_mut(&mut csr.data, &bounds);
            let fid = *fid;
            // `cpu_pool().install`, never rayon's global registry: this runs on
            // the consumer thread, which in a forked DataLoader worker has no
            // usable global pool (see `crate::pool`).
            let out: Vec<u32> = crate::pool::cpu_pool().install(|| {
                idx_spans
                    .into_par_iter()
                    .zip(dat_spans.into_par_iter())
                    .enumerate()
                    .map_init(TransformScratch::default, |sc, (l, (ix, dx))| {
                        let row = occ.slots[slots[l] as usize].1;
                        if no_shrink {
                            // Nothing can move, so the row is transformed
                            // where it already lies — no scratch, no copy.
                            crate::downsample::clip_negatives(dx);
                            if self.normalize || self.log1p {
                                apply_sparse_transforms(
                                    dx,
                                    self.normalize,
                                    self.log1p,
                                    self.target_sum,
                                );
                            }
                            ix.len() as u32
                        } else {
                            self.transform_row_into(fid, row, ix, dx, sc);
                            let n = sc.idx.len();
                            ix[..n].copy_from_slice(&sc.idx);
                            dx[..n].copy_from_slice(&sc.dat);
                            n as u32
                        }
                    })
                    .collect()
            });
            for (l, &n) in out.iter().enumerate() {
                lens[slots[l] as usize] = n;
            }
        }

        // --- assemble in plan order ------------------------------------------
        let mut indptr: Vec<i64> = Vec::with_capacity(n_out + 1);
        indptr.push(0);
        let mut acc: i64 = 0;
        for p in 0..n_out {
            acc += i64::from(lens[occ.slot_of_pos[p] as usize]);
            indptr.push(acc);
        }
        let nnz = acc as usize;
        let mut indices: Vec<i32> = vec![0; nnz];
        let mut data: Vec<f32> = vec![0.0; nnz];
        let bounds: Vec<(usize, usize)> = (0..n_out)
            .map(|p| (indptr[p] as usize, indptr[p + 1] as usize))
            .collect();
        let out_idx = spans_mut(&mut indices, &bounds);
        let out_dat = spans_mut(&mut data, &bounds);
        let occ_ref = &occ;
        let per_file_ref = &per_file;
        crate::pool::cpu_pool().install(|| {
            out_idx
                .into_par_iter()
                .zip(out_dat.into_par_iter())
                .enumerate()
                .for_each(|(p, (oi, od))| {
                    let slot = occ_ref.slot_of_pos[p] as usize;
                    let (bucket, local) = occ_ref.slot_loc[slot];
                    let csr = &per_file_ref[bucket as usize];
                    let lo = csr.indptr[local as usize] as usize;
                    let n = oi.len();
                    oi.copy_from_slice(&csr.indices[lo..lo + n]);
                    od.copy_from_slice(&csr.data[lo..lo + n]);
                })
        });

        Ok(SparseCellSetBatch {
            indptr,
            indices,
            data,
            shape: (n_out, self.n_cols),
            cell_indices: plan.rows[emit_lo..emit_hi].to_vec(),
            file_ids: plan.file_ids[emit_lo..emit_hi].to_vec(),
            set_offsets: plan.set_offsets.clone(),
            role_tags: plan.role_tags[emit_lo..emit_hi].to_vec(),
        })
    }

    /// Structural validation of a plan, which arrives straight from (untrusted)
    /// Python on both public routes. Split out of the gather so every executor
    /// below runs against a plan already known to be well formed, and so a
    /// malformed plan is refused before any read rather than panicking on an
    /// unchecked slice deep inside one.
    fn validate_plan(&self, engine: &PrefetchEngine, plan: &SparseCellSetPlan) -> Result<()> {
        let total_rows = plan.rows.len();
        // Width is checked by `check_plan_width` on both public routes, before
        // any admission sizing or prefetch I/O — not here, where the work it
        // refuses has already happened.
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
            // From the registry's slot, not from a handle: the `file_id` range
            // check above has already run, and validating a plan must not be
            // what pulls every file it names into residence.
            let n_obs = engine.registry().n_obs(fid).unwrap_or(0) as usize;
            if row as usize >= n_obs {
                return Err(LoaderError::IndexOutOfRange { idx: row, n_obs });
            }
        }
        Ok(())
    }

    /// The per-set walk: one [`scx_format_io::BackedCsrReader::read_rows_with_admission`]
    /// call per set (and per file within a cross-file set), a per-set
    /// `Vec<Option<(Vec<i32>, Vec<f32>)>>` holding the gathered rows, and an
    /// `extend_from_slice` copy of every row into the batch.
    fn gather_per_set(
        &self,
        engine: &PrefetchEngine,
        plan: &SparseCellSetPlan,
        admit_row_groups: Option<Admit>,
    ) -> Result<SparseCellSetBatch> {
        let total_rows = plan.rows.len();

        let n_sets = plan.set_offsets.len().saturating_sub(1);

        let mut indptr: Vec<i64> = Vec::with_capacity(total_rows + 1);
        indptr.push(0);
        // Pre-size the two nnz-sized outputs so the append loop below does not
        // grow them geometrically from empty, which costs an allocation plus a
        // copy of everything written so far at each doubling — priced as
        // allocations, not memcpys.
        //
        // **An estimate from catalog stats, deliberately, not an exact count.**
        // `with_capacity` is a hint: undershooting costs a reallocation the
        // caller would have paid anyway, overshooting costs transient bytes.
        // Nothing here needs a bound, so nothing here should pay for one — and
        // the exact version did pay. Summing the plan's true row lengths means
        // decoding each touched shard's indptr, once per plan, and a two-build
        // A/B measured that at ~9 % of `gather_grouped_s512` on tabula (medians
        // 89.4 → 80.8 sets/s in one arm ordering and 77.7 → 84.6 in the other,
        // both agreeing) against a 1.85–2.07× win on pbmc3k, whose single-shard
        // batches made the decode free. The estimate keeps the win and drops
        // the I/O: `mean_nnz_per_row` comes from the catalog walk the budget
        // model already does at construction, and reads nothing.
        //
        // **Biased upward by an eighth, and that is the whole point.** The two
        // errors are not symmetric: overshooting wastes transient bytes, while
        // undershooting by any margin at all forces a reallocation at FULL
        // size — the single most expensive one, copying the whole batch. A
        // mean-exact estimate lands just under the truth about half the time,
        // and an A/B caught that costing the entire win: on pbmc3k the plan's
        // rows average 850.5 non-zeros against the manifest's 847.0, so the
        // unbiased estimate came 0.4 % short, took that one full-size copy, and
        // measured *slower* than growing from empty (3949 vs 4799 sets/s).
        //
        // An eighth is a shift, covers ordinary sampling variation in row
        // density, and costs ~0.9 MB on a 1024-row pbmc3k batch. A plan that
        // deliberately selects the densest rows can still exceed it and pay the
        // one reallocation; that is the rare case, and it is the case the
        // estimate cannot serve without reading the data.
        //
        // 0 when no shard carried stats, which is the same "size unknown"
        // signal the byte budget falls back on — capacity 0 is exactly the
        // pre-change behaviour, so an unknowable file is no worse off.
        let planned_nnz = presize_nnz(total_rows, self.mean_nnz_per_row);
        let mut indices: Vec<i32> = Vec::with_capacity(planned_nnz);
        let mut data: Vec<f32> = Vec::with_capacity(planned_nnz);
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
                let reader = engine.lease(fid)?;
                reader
                    .read_rows_with_admission(
                        set_rows,
                        admit_row_groups.as_ref(),
                        |orig_pos, idx, dat| {
                            // `orig_pos` is the position in `set_rows`, not the row id
                            // — the scatter fires in shard-grouped order. The row id is
                            // what keys the downsample RNG, so read it back through the
                            // request array.
                            per_row[orig_pos] =
                                Some(self.transform_row(fid, set_rows[orig_pos], idx, dat));
                            Ok(())
                        },
                    )
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
                    let reader = engine.lease(f)?;
                    let rs: Vec<u64> = items.iter().map(|&(_, r)| r).collect();
                    reader
                        .read_rows_with_admission(
                            &rs,
                            admit_row_groups.as_ref(),
                            |orig_pos, idx, dat| {
                                let (within_set_pos, src_row) = items[orig_pos];
                                per_row[within_set_pos] =
                                    Some(self.transform_row(f, src_row, idx, dat));
                                Ok(())
                            },
                        )
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
        let mut sc = TransformScratch::default();
        self.transform_row_into(fid, row, idx, dat, &mut sc);
        (sc.idx, sc.dat)
    }

    /// [`Self::transform_row`] writing into a caller-owned [`TransformScratch`],
    /// so the batch executor pays its allocations once per rayon worker rather
    /// than once per row. One implementation of the stage order, so the owned
    /// and the reused form cannot drift.
    fn transform_row_into(
        &self,
        fid: u32,
        row: u64,
        idx: &[i32],
        dat: &[f32],
        sc: &mut TransformScratch,
    ) {
        match &self.remap {
            Some(tables) => remap_row_into(
                idx,
                dat,
                &tables[fid as usize],
                &mut sc.pairs,
                &mut sc.idx,
                &mut sc.dat,
            ),
            None => {
                sc.idx.clear();
                sc.idx.extend_from_slice(idx);
                sc.dat.clear();
                sc.dat.extend_from_slice(dat);
            }
        }
        crate::downsample::clip_negatives(&mut sc.dat);
        if let Some(cfg) = &self.downsample {
            crate::downsample::downsample_row(
                &mut sc.idx,
                &mut sc.dat,
                cfg,
                cfg.identity_for(fid),
                row,
            );
        }
        if self.normalize || self.log1p {
            apply_sparse_transforms(&mut sc.dat, self.normalize, self.log1p, self.target_sum);
        }
    }
}

/// Where one row's decoder query and its mask live, resolved once for both
/// addressings so the parallel loop has no branch.
struct RowQuery {
    /// Span of this row's query ids in the flat `query_gene_ids`.
    q_lo: usize,
    q_hi: usize,
    /// Start of this row's mask bits. The mask is `k_dec`-strided per row on the
    /// per-set path and parallel to the ragged query on the per-row path, so it
    /// needs its own offset rather than reusing `q_lo`.
    m_lo: usize,
    /// Index into the sorted-panel table. Rows of one set share a panel on the
    /// per-set path; every row has its own on the per-row path.
    panel: usize,
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
    query_offsets: Option<&[i64]>,
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
    // Three panics this entry could still reach across the FFI, all reproduced
    // against the built extension before being closed here. `validate_indptr`
    // below closed the non-monotonic case; these are its siblings.
    if k_enc == 0 || k_dec == 0 {
        // `par_chunks_mut(0)` panics with "chunk_size must not be zero".
        return Err(LoaderError::ConfigError {
            reason: format!(
                "collate_gathered: k_enc and k_dec must be >= 1, got {k_enc} and {k_dec}"
            ),
        });
    }
    if indices.len() != data.len() {
        // A short `indices` with a well-formed `indptr` over `data` slices out
        // of bounds in the row loop.
        return want("indices", indices.len(), data.len());
    }
    // The twin of `pyscx.tokenize.top_k`'s guard. `n_genes_total` fixes the two
    // sentinels, so a non-positive value emits nonsense token ids on any row the
    // per-row check never sees — an empty row at `-5` emitted `[-5, -4]`.
    if scalars.n_genes_total < 1 {
        return Err(LoaderError::ConfigError {
            reason: format!(
                "collate_gathered: n_genes_total must be >= 1, got {}",
                scalars.n_genes_total
            ),
        });
    }
    if indptr.len() != n_rows + 1 {
        return want("indptr", indptr.len(), n_rows + 1);
    }
    // Shape alone is not enough: the rayon loop below slices `indices[lo..hi]`
    // straight from these entries, so a non-monotonic or negative `indptr` panics
    // rather than erroring. Pre-existing gap on this entry point, closed here
    // because the check is shared with `downsample_counts_csr`.
    validate_indptr(indptr, data.len())?;
    if hide_readout.len() != n_rows {
        return want("hide_readout", hide_readout.len(), n_rows);
    }
    // `set_offsets` is the other prefix array on this entry and was the only one
    // never validated. `validate_indptr` is exactly the right check for it: it
    // must start at 0, be non-decreasing, and END at `n_rows`. Without the last
    // clause `set_offsets = [0, 1]` over two rows was ACCEPTED and row 1 kept
    // `row_set = 0` — a silently wrong set assignment rather than a panic, which
    // is the worse failure. (With `n_sets = 0` the same hole reached an
    // out-of-bounds index on `n_measured`; that path is now unreachable because
    // it needs `k_dec == 0`, which is rejected above, but the array is validated
    // rather than left resting on that coincidence.)
    // `validate_indptr` is exactly this check — starts at 0, non-decreasing,
    // ends at the declared total — so it is called rather than reimplemented,
    // the way `query_offsets` below already does. The explanation of what the
    // last clause buys lives in the mapped message.
    validate_indptr(set_offsets, n_rows).map_err(|_| LoaderError::ConfigError {
        reason: format!(
            "collate_gathered: set_offsets must start at 0, be non-decreasing, and end at n_rows {n_rows}; rows past the last set would silently keep set 0"
        ),
    })?;
    // These two pass through untouched into the batch, so a wrong length yields
    // a batch whose arrays disagree with `n_rows`.
    if file_ids.len() != n_rows {
        return want("file_ids", file_ids.len(), n_rows);
    }
    if role_tags.len() != n_rows {
        return want("role_tags", role_tags.len(), n_rows);
    }
    let has_mask = !enc_mask_positions.is_empty();

    // Row → set index, for per-set query / n_measured lookup.
    let mut row_set = vec![0usize; n_rows];
    for s in 0..n_sets {
        let lo = set_offsets[s] as usize;
        let hi = set_offsets[s + 1] as usize;
        for r in row_set.iter_mut().take(hi).skip(lo) {
            *r = s;
        }
    }

    // Resolve each row's query span, mask span, panel and `n_measured` once,
    // so the parallel loop below is identical on both addressings.
    //
    // The choice is made on `query_offsets.is_some()` and never on a length:
    // `n_sets == n_rows` whenever every set is a singleton, which is the common
    // shape for R4 consumers, so a length-sniffing dispatch would pick the
    // wrong reading exactly where it matters most.
    let (rows, measured_of_row, n_panels) = match query_offsets {
        None => {
            if query_gene_ids.len() != n_sets * k_dec {
                return want("query_gene_ids", query_gene_ids.len(), n_sets * k_dec);
            }
            if n_measured.len() != n_sets {
                return want("n_measured", n_measured.len(), n_sets);
            }
            if has_mask && enc_mask_positions.len() != n_rows * k_dec {
                return want(
                    "enc_mask_positions",
                    enc_mask_positions.len(),
                    n_rows * k_dec,
                );
            }
            let rows: Vec<RowQuery> = (0..n_rows)
                .map(|r| {
                    let s = row_set[r];
                    RowQuery {
                        q_lo: s * k_dec,
                        q_hi: (s + 1) * k_dec,
                        m_lo: r * k_dec,
                        panel: s,
                    }
                })
                .collect();
            let measured: Vec<u32> = (0..n_rows).map(|r| n_measured[row_set[r]]).collect();
            (rows, measured, n_sets)
        }
        Some(off) => {
            if off.len() != n_rows + 1 {
                return want("query_offsets", off.len(), n_rows + 1);
            }
            // Same reason as `indptr`: the loop slices straight from these, so a
            // non-monotonic or negative offset would panic rather than error.
            validate_indptr(off, query_gene_ids.len()).map_err(|e| LoaderError::ConfigError {
                reason: format!("collate_gathered: query_offsets {e}"),
            })?;
            let widest = (0..n_rows)
                .map(|r| (off[r + 1] - off[r]) as usize)
                .max()
                .unwrap_or(0);
            if widest > k_dec {
                return Err(LoaderError::ConfigError {
                    reason: format!(
                        "collate_gathered: k_dec {k_dec} is narrower than the widest per-row query ({widest}); k_dec is the padded output width"
                    ),
                });
            }
            if n_measured.len() != n_rows {
                return want("n_measured", n_measured.len(), n_rows);
            }
            // Per-row masks are parallel to the ragged query, not `k_dec`-strided.
            if has_mask && enc_mask_positions.len() != query_gene_ids.len() {
                return want(
                    "enc_mask_positions",
                    enc_mask_positions.len(),
                    query_gene_ids.len(),
                );
            }
            let rows: Vec<RowQuery> = (0..n_rows)
                .map(|r| RowQuery {
                    q_lo: off[r] as usize,
                    q_hi: off[r + 1] as usize,
                    m_lo: off[r] as usize,
                    panel: r,
                })
                .collect();
            (rows, n_measured.to_vec(), n_rows)
        }
    };

    // `pflog_raw` divides by `n_measured`, so a zero yields a non-finite centre
    // and writes -inf into every encoder slot of that row. `transform::pflog_raw`
    // documented that "the collator validates it upstream" and the collator did
    // not — checked: a v2-shaped call with `n_measured = [0]` returned
    // `encoder_counts [-inf, -inf]`. Contract v3 makes the array per-row, so one
    // bad entry poisons one cell rather than a set, which is easier to pass by
    // accident on a ragged or empty query.
    if scalars.mode == PreprocessMode::PflogRaw {
        if let Some(i) = measured_of_row.iter().position(|&m| m == 0) {
            return Err(LoaderError::ConfigError {
                reason: format!(
                    "collate_gathered: PflogRaw requires n_measured >= 1; row {i} has 0, which would centre by a zero denominator"
                ),
            });
        }
    }

    // One sorted panel per distinct query, built before the row loop because on
    // the per-set path the panel is shared by the set's rows while the mask bits
    // are per-row. Skipped entirely on the perturbation path (no mask ⇒ nothing
    // to look up), so that path pays neither the sort nor the allocation.
    //
    // Per-row queries cost one sort per row instead of one per set — the price
    // of the addressing, not a regression in the per-set path.
    let query_indices: Vec<SetQueryIndex> = if has_mask {
        let mut by_panel: Vec<Option<SetQueryIndex>> = (0..n_panels).map(|_| None).collect();
        for row in &rows {
            by_panel[row.panel]
                .get_or_insert_with(|| SetQueryIndex::new(&query_gene_ids[row.q_lo..row.q_hi]));
        }
        by_panel
            .into_iter()
            .map(|p| p.unwrap_or_else(|| SetQueryIndex::new(&[])))
            .collect()
    } else {
        Vec::new()
    };

    let mut enc_ids = vec![0i64; n_rows * k_enc];
    let mut enc_counts = vec![0f32; n_rows * k_enc];
    let mut enc_mask = vec![0u8; n_rows * k_enc];
    let mut enc_pad = vec![0u8; n_rows * k_enc];
    let mut target = vec![0f32; n_rows * k_dec];
    let mut target_pad = vec![0u8; n_rows * k_dec];
    let mut library = vec![0f32; n_rows];

    // On the loader's pool, never rayon's global registry. This kernel is
    // exposed as `pyscx.collate_cellset_gathered`, so a forked DataLoader
    // worker can call it with no dataset in hand and therefore no PID check in
    // front of it — and a global-pool dispatch from a forked child hangs
    // forever. See `crate::pool`.
    // `map_init` rather than `for_each_init` so the per-row invariant check
    // below can fail the call: the row is already in cache at that point, so the
    // check costs no extra pass over the data.
    let row_check: std::result::Result<(), String> = crate::pool::cpu_pool().install(|| {
        enc_ids
            .par_chunks_mut(k_enc)
            .zip(enc_counts.par_chunks_mut(k_enc))
            .zip(enc_mask.par_chunks_mut(k_enc))
            .zip(enc_pad.par_chunks_mut(k_enc))
            .zip(target.par_chunks_mut(k_dec))
            .zip(target_pad.par_chunks_mut(k_dec))
            .zip(library.par_iter_mut())
            .enumerate()
            // `for_each_init`, not `for_each`: the kernel's per-row buffers are
            // created once per rayon worker instead of three `Vec`s per cell.
            .map_init(
                Scratch::new,
                |scratch, (r, ((((((eid, ecnt), emask), epad), tgt), tpad), libslot))| {
                    let q = &rows[r];
                    let qlen = q.q_hi - q.q_lo;
                    let lo = indptr[r] as usize;
                    let hi = indptr[r + 1] as usize;
                    // Same invariant `pyscx.tokenize` enforces, and for the same
                    // reason: `collate_cell`'s target gather and its
                    // `lib_size_redef` sum both exact-match binary-search
                    // `gene_ids`, so an unsorted row silently answers 0.0 for a
                    // gene the row actually carries. Reproduced on the built
                    // extension: a row `{0: 5, 2: 7, 1: 9}` queried with
                    // `[0, 1, 2]` returned `[5.0, 0.0, 0.0]` instead of
                    // `[5.0, 9.0, 7.0]` — corrupted targets, no error. The crop
                    // also needs ids below `n_genes_total`, which IS the
                    // GENE_MASK token.
                    crate::tokenize::check_row_ids(
                        &indices[lo..hi],
                        Some(scalars.n_genes_total as usize),
                        "collate_gathered",
                    )?;
                    let cin = CellIn {
                        gene_ids: &indices[lo..hi],
                        raw: &data[lo..hi],
                        query: &query_gene_ids[q.q_lo..q.q_hi],
                        mask: has_mask.then(|| RowMask {
                            positions: &enc_mask_positions[q.m_lo..q.m_lo + qlen],
                            index: &query_indices[q.panel],
                        }),
                        hide_readout: hide_readout[r] != 0,
                    };
                    let cfg = CollateConfig {
                        k_enc,
                        mode: scalars.mode,
                        target_sum: scalars.target_sum,
                        n_measured: measured_of_row[r] as usize,
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
                    *libslot = collate_cell(&cin, &cfg, scratch, &mut out);
                    // Slots past this row's query are padding. `collate_cell`
                    // writes only `query.len()` targets, so the tail keeps the
                    // 0.0 it was allocated with — indistinguishable from a real
                    // zero target without this mask.
                    for slot in tpad.iter_mut().skip(qlen) {
                        *slot = 1;
                    }
                    Ok(())
                },
            )
            .collect()
    });
    row_check.map_err(|reason| LoaderError::ConfigError { reason })?;

    Ok(CollatedCellSetBatch {
        encoder_gene_ids: enc_ids,
        encoder_counts: enc_counts,
        encoder_mask: enc_mask,
        encoder_pad_mask: enc_pad,
        target_counts: target,
        target_pad_mask: target_pad,
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

/// Where every output position's row comes from, over the whole batch.
///
/// Built once per plan, over the **emitted** positions only. The point of it is
/// that a `(file, row)` the plan names more than once — a control pool repeated
/// in every set, a shared neighbour, a group sampled with replacement — is read
/// and transformed once and memcpy'd into each of its positions.
struct Occurrences {
    /// Unique `(file_id, row)` in first-occurrence order.
    slots: Vec<(u32, u64)>,
    /// Emitted position (relative to `emit_lo`) -> slot index.
    slot_of_pos: Vec<u32>,
    /// `(file_id, its slot indices)`, files in first-appearance order and each
    /// file's slots in first-occurrence order — which is the row order its one
    /// `read_row_indices_with_admission` call is made in, and therefore the row
    /// order of the `ScxCsr` that comes back.
    by_file: Vec<(u32, Vec<u32>)>,
    /// Slot -> `(bucket in by_file, local row in that bucket's csr)`.
    slot_loc: Vec<(u32, u32)>,
    /// `slot_of_pos[i] == i` for every `i` — i.e. the plan names no row twice.
    identity: bool,
}

impl Occurrences {
    fn build(plan: &SparseCellSetPlan, emit_lo: usize, emit_hi: usize) -> Self {
        let n = emit_hi.saturating_sub(emit_lo);
        let mut index: HashMap<(u32, u64), u32> = HashMap::with_capacity(n);
        let mut slots: Vec<(u32, u64)> = Vec::with_capacity(n);
        let mut slot_of_pos: Vec<u32> = Vec::with_capacity(n);
        let mut by_file: Vec<(u32, Vec<u32>)> = Vec::new();
        let mut file_bucket: HashMap<u32, usize> = HashMap::new();
        let mut slot_loc: Vec<(u32, u32)> = Vec::with_capacity(n);
        let mut identity = true;

        for p in emit_lo..emit_hi {
            let key = (plan.file_ids[p], plan.rows[p]);
            let slot = match index.get(&key) {
                Some(&s) => s,
                None => {
                    let s = slots.len() as u32;
                    index.insert(key, s);
                    slots.push(key);
                    let bucket = *file_bucket.entry(key.0).or_insert_with(|| {
                        by_file.push((key.0, Vec::new()));
                        by_file.len() - 1
                    });
                    let local = by_file[bucket].1.len() as u32;
                    by_file[bucket].1.push(s);
                    slot_loc.push((bucket as u32, local));
                    s
                }
            };
            identity &= slot as usize == slot_of_pos.len();
            slot_of_pos.push(slot);
        }

        Occurrences {
            slots,
            slot_of_pos,
            by_file,
            slot_loc,
            identity,
        }
    }
}

/// Split `buf` into the disjoint spans `bounds` names, for a parallel scatter.
///
/// `bounds` must be ascending and non-overlapping, which is what a prefix sum
/// over row lengths produces — that is the whole reason the spans can be written
/// in parallel without synchronisation. Asserted in debug rather than merely
/// assumed; in release an overlap makes `split_at_mut` panic on the negative
/// stride, so it cannot silently alias.
fn spans_mut<'a, T>(buf: &'a mut [T], bounds: &[(usize, usize)]) -> Vec<&'a mut [T]> {
    let mut out = Vec::with_capacity(bounds.len());
    let mut rest = buf;
    let mut base = 0usize;
    for &(lo, hi) in bounds {
        debug_assert!(
            lo >= base && hi >= lo,
            "spans must be ascending and non-overlapping: ({lo}, {hi}) after {base}"
        );
        let (_, tail) = rest.split_at_mut(lo - base);
        let (span, tail) = tail.split_at_mut(hi - lo);
        out.push(span);
        rest = tail;
        base = hi;
    }
    out
}

/// Per-rayon-worker buffers for [`SparseCellSetLoader::transform_row_into`],
/// created once per worker by `map_init` instead of three `Vec`s per row.
#[derive(Default)]
struct TransformScratch {
    idx: Vec<i32>,
    dat: Vec<f32>,
    pairs: Vec<(i32, f32)>,
}

/// Map local gene ids to global via `local_to_global` (`-1` = drop), then sort
/// by global id and coalesce duplicates by summing — matching state3's
/// `local_to_global` + `_coalesce_gene_counts` (`dataset.py:381-391`), so the
/// output CSR row stays canonical (sorted, unique).
/// Writes into caller-owned buffers rather than returning them, so the batch
/// executor pays three allocations per rayon worker instead of three per row.
/// `sort_by_key` is stable, and the sum order within an equal-id run is what
/// makes the coalesced f32 reproducible.
fn remap_row_into(
    indices: &[i32],
    data: &[f32],
    local_to_global: &[i32],
    pairs: &mut Vec<(i32, f32)>,
    out_idx: &mut Vec<i32>,
    out_dat: &mut Vec<f32>,
) {
    pairs.clear();
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
    out_idx.clear();
    out_dat.clear();
    for &(g, v) in pairs.iter() {
        if out_idx.last() == Some(&g) {
            *out_dat.last_mut().unwrap() += v;
        } else {
            out_idx.push(g);
            out_dat.push(v);
        }
    }
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
