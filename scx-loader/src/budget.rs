//! Memory-budget breakdowns shared by the sequential and plan-driven loaders.
//!
//! Both [`crate::pipeline::TrainingPipeline`] and
//! [`crate::index_plan::IndexPlanLoader`] auto-tune their config to fit a
//! caller-supplied `max_memory_mb`. The two paths have different shapes
//! but decompose into the same handful of components: cache, batch buffer,
//! per-path lookahead/prefetch overhead, transient temporaries, and
//! constant Python overhead.
//!
//! [`BudgetBreakdown`] is the shared component type every path produces at
//! construction time. Since ORG-9.10-5 the auto-tune *algorithm* is shared too:
//! [`tune`] drives the descent for all three loaders and each module supplies
//! only its own knobs and reduction order through [`BudgetModel`]. What stays
//! per-module is the **exhaustion policy** — see [`Tuned::exhausted`].
//!
//! This module also owns the two **cache-sizing verdicts** the plan-driven
//! loaders warn from — [`assess_cache_sizing`] (construction time, from the
//! budget model) and [`assess_cache_thrash`] (runtime, from observed
//! [`CacheMetrics`]). Both are deliberately pure so they can be unit-tested
//! without a Python interpreter or a fixture file; only the `warnings.warn`
//! call lives in `python.rs`.

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;
use scx_format_io::CacheMetrics;
use std::sync::atomic::Ordering;

/// Constant Python/Arrow/numpy/threads overhead estimate (bytes). Both
/// budget models add this to their per-component sum.
pub const PYTHON_OVERHEAD_BYTES: usize = 50 * 1024 * 1024;

/// Lower bound on the shard cache that a budget auto-tune may reach
/// **silently**.
///
/// Below this, a gather batch whose rows land in more than a handful of shards
/// re-decodes shards it just evicted — the 143 s/batch regime STATE3 hit
/// (fixed by raising a consumer-side `cache_shards=16`, not by any SCX change).
/// The auto-tune is still allowed to go lower, because refusing would turn
/// configurations that work today into hard errors; it just may not do so
/// without telling the caller. See [`assess_cache_sizing`].
pub const MIN_CACHE_SHARDS: usize = 8;

/// Minimum observed `misses` before [`assess_cache_thrash`] will return a
/// verdict.
///
/// Every run begins with a cold cache, where misses are *expected* and carry no
/// information about the working set. Sampling too early reports the warm-up as
/// pathology.
const THRASH_MIN_MISSES: u64 = 64;

/// Miss rate above which the cache is not absorbing repeat access.
const THRASH_MISS_RATE: f64 = 0.5;

/// Evictions-per-miss above which misses are *displacing live entries* rather
/// than filling empty slots. This is the term that separates thrash from a
/// merely cold cache, and it is why the predicate needs both.
const THRASH_EVICTIONS_PER_MISS: f64 = 0.5;

/// Adaptive budget (MB) for a plan-driven loader whose caller passed
/// `max_memory_mb=None`.
///
/// Mirrors the sequential path's `pipeline::adaptive_budget_mb` arithmetic —
/// the requested configuration's own need, rounded up to whole MB with ~12 %
/// headroom, clamped to `[floor_mb, ADAPTIVE_BUDGET_CAP_MB]` — so all three
/// loader classes resolve a `None` budget the same way instead of each
/// inventing a policy. Being clamped *up* to `floor_mb` means this never
/// tightens a small file below the old default; being clamped down to the cap
/// means a genuinely huge request still falls through to the auto-tune (and its
/// warning) rather than reserving unbounded RSS.
pub fn adaptive_budget_mb(requested_need_bytes: usize, floor_mb: usize) -> usize {
    let need_mb = requested_need_bytes.div_ceil(1024 * 1024);
    let with_headroom = need_mb.saturating_add(need_mb / 8);
    let cap = crate::pipeline::ADAPTIVE_BUDGET_CAP_MB;
    // `clamp` panics if min > max, so let an unusually high floor win.
    with_headroom.clamp(floor_mb, cap.max(floor_mb))
}

/// Construction-time verdict: the memory budget could not hold the shard cache
/// the caller asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheSizingVerdict {
    /// What the caller asked for (explicitly or by default).
    pub requested_cache_shards: usize,
    /// What the budget actually affords.
    pub effective_cache_shards: usize,
    /// Average decoded bytes per shard, the model's per-shard unit.
    pub shard_decoded_bytes: usize,
    /// The budget that forced the reduction.
    pub budget_mb: usize,
    /// `max_memory_mb` that would have held `requested_cache_shards`.
    pub budget_mb_for_requested: usize,
    /// `effective_cache_shards < MIN_CACHE_SHARDS` — thrash is likely, not
    /// merely possible.
    pub below_floor: bool,
}

/// Returns a verdict iff the budget forced the shard cache below what was
/// requested **and** the caller needs to know. `None` means silence.
///
/// `budget_was_explicit` is what keeps this from crying wolf. Under an
/// *adaptive* budget the cap is deliberate — it is the whole point of bounding
/// RSS — and a file with few large shards will routinely be "reduced" from the
/// default 128 to a couple of dozen while running perfectly well (a real
/// example: 194 MB shards ⇒ 21 shards fit the 4 GB cap, at 0.7 GB observed RSS).
/// Warning there would fire on healthy default configurations, which is how a
/// diagnostic gets filtered and stops working. So under an adaptive budget the
/// verdict is withheld unless the affordable count falls below
/// [`MIN_CACHE_SHARDS`], where thrash becomes likely rather than hypothetical.
/// An **explicit** `max_memory_mb` that conflicts with an explicit
/// `cache_shards` is always reported: the caller asked for two things that don't
/// fit and only they can decide which one gives.
///
/// Pure: takes the already-computed model terms, so the auto-tune loops stay in
/// their own modules and this stays unit-testable.
pub fn assess_cache_sizing(
    requested_cache_shards: usize,
    effective_cache_shards: usize,
    shard_decoded_bytes: usize,
    budget_mb: usize,
    non_cache_bytes: usize,
    budget_was_explicit: bool,
) -> Option<CacheSizingVerdict> {
    if effective_cache_shards >= requested_cache_shards {
        return None;
    }
    let below_floor = effective_cache_shards < MIN_CACHE_SHARDS;
    if !budget_was_explicit && !below_floor {
        return None;
    }
    let need = requested_cache_shards
        .saturating_mul(shard_decoded_bytes)
        .saturating_add(non_cache_bytes);
    Some(CacheSizingVerdict {
        requested_cache_shards,
        effective_cache_shards,
        shard_decoded_bytes,
        budget_mb,
        budget_mb_for_requested: need.div_ceil(1024 * 1024),
        below_floor,
    })
}

/// Runtime verdict: the observed cache counters are consistent with a working
/// set larger than the cache.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThrashVerdict {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// `misses / (hits + misses)`.
    pub miss_rate: f64,
    /// `evictions / misses` — the term that distinguishes thrash from a cold
    /// cache.
    pub evictions_per_miss: f64,
    /// The cache size in effect.
    pub cache_shards: usize,
}

/// Returns a verdict iff `m` looks like working-set overflow, `None` otherwise.
///
/// Requires **three** conditions, and the third is the load-bearing one:
///
/// 1. `misses >= THRASH_MIN_MISSES` — past warm-up, so there is signal at all.
/// 2. `miss_rate > THRASH_MISS_RATE` — the cache is not absorbing repeat access.
/// 3. `evictions_per_miss > THRASH_EVICTIONS_PER_MISS` — each miss is
///    *displacing a live entry*.
///
/// Without (3) this would fire on any cold sequential scan, where every read is
/// a legitimate first touch: high miss rate, no evictions, no pathology. That
/// distinction is the whole point of the diagnostic, so it is asserted by
/// `cold_scan_is_not_thrash` rather than left to reviewer inspection.
pub fn assess_cache_thrash(m: &CacheMetrics, cache_shards: usize) -> Option<ThrashVerdict> {
    let hits = m.hits.load(Ordering::Relaxed);
    let misses = m.misses.load(Ordering::Relaxed);
    let evictions = m.evictions.load(Ordering::Relaxed);

    if misses < THRASH_MIN_MISSES {
        return None;
    }
    let total = hits.saturating_add(misses);
    if total == 0 {
        return None;
    }
    let miss_rate = misses as f64 / total as f64;
    let evictions_per_miss = evictions as f64 / misses as f64;
    if miss_rate <= THRASH_MISS_RATE || evictions_per_miss <= THRASH_EVICTIONS_PER_MISS {
        return None;
    }
    Some(ThrashVerdict {
        hits,
        misses,
        evictions,
        miss_rate,
        evictions_per_miss,
        cache_shards,
    })
}

/// Per-component memory breakdown produced by both budget paths so that
/// callers (Python `memory_budget()` accessors, the benchmark harness, the
/// `SCX_LOADER_PROFILE` Drop dump) can render a consistent "where the bytes
/// go" report.
///
/// Fields are intentionally additive — `total_bytes` is the saturating sum
/// of the others. None of them count mmap-resident pages: the kernel page
/// cache is treated as evictable under pressure on both paths and explicitly
/// excluded from the auto-tune budget. (The sequential path keeps its
/// `mmap_bytes` term separate on the existing
/// [`crate::pipeline::MemoryBudget`] surface for backwards compatibility.)
#[derive(Debug, Clone, Copy, Default)]
pub struct BudgetBreakdown {
    /// Decoded shard cache budget (bytes). Sequential:
    /// `(shard_group_size + 1) × decoded_shard_bytes`. Plan-driven:
    /// `effective_cache_shards × shard_decoded_bytes`.
    pub cache_bytes: usize,
    /// Dense batch buffer(s) carried in flight. Sequential:
    /// `(prefetch_batches.max(2) + 1) × batch_size × n_output_genes × 4`.
    /// Plan-driven: `2 × max_plan_size × n_output_cols × 4` (paired
    /// `x` / `x_paired`).
    pub batch_buffer_bytes: usize,
    /// Per-path lookahead/prefetch staging overhead (e.g. plan-tuple Vecs in
    /// the index-plan iterator). Sequential paths report `0` here.
    pub lookahead_overhead_bytes: usize,
    /// Short-lived temporaries that are co-resident with the batch buffer
    /// during a `process_plan` call: per-batch obs `Vec`s allocated by
    /// `extract_obs_columns`, the `PairRequest` sort scratch in
    /// `gather_pairs_dense`, etc. Empty (`0`) on the sequential path.
    pub transient_bytes: usize,
    /// Constant Python interpreter / numpy / Arrow / thread-stack overhead.
    /// Equal to [`PYTHON_OVERHEAD_BYTES`] on both paths.
    pub python_overhead_bytes: usize,
    /// Saturating sum of every other field.
    pub total_bytes: usize,
}

impl BudgetBreakdown {
    /// Construct a breakdown from its component parts. `total_bytes` is the
    /// saturating sum of the inputs — callers don't pass it explicitly.
    pub fn new(
        cache_bytes: usize,
        batch_buffer_bytes: usize,
        lookahead_overhead_bytes: usize,
        transient_bytes: usize,
        python_overhead_bytes: usize,
    ) -> Self {
        let total_bytes = cache_bytes
            .saturating_add(batch_buffer_bytes)
            .saturating_add(lookahead_overhead_bytes)
            .saturating_add(transient_bytes)
            .saturating_add(python_overhead_bytes);
        BudgetBreakdown {
            cache_bytes,
            batch_buffer_bytes,
            lookahead_overhead_bytes,
            transient_bytes,
            python_overhead_bytes,
            total_bytes,
        }
    }

    /// Returns `true` iff `total_bytes <= max_memory_mb × 1 MiB`.
    pub fn fits_within(&self, max_memory_mb: usize) -> bool {
        self.fits_within_bytes(max_memory_mb.saturating_mul(1024 * 1024))
    }

    /// Byte-granularity form of [`Self::fits_within`], and the predicate
    /// [`tune`] terminates on.
    ///
    /// Bytes rather than MB is not a stylistic choice: `SparseCellSetLoader`
    /// takes its budget in bytes and its tests exercise budgets far below
    /// 1 MiB, which whole-MB rounding would collapse to zero.
    pub fn fits_within_bytes(&self, budget_bytes: usize) -> bool {
        self.total_bytes <= budget_bytes
    }

    /// Render as a Python dict with keys
    /// `{cache_bytes, batch_buffer_bytes, lookahead_overhead_bytes,
    ///   transient_bytes, python_overhead_bytes, total_bytes}`. All values
    /// are `int`.
    #[cfg(feature = "python")]
    pub fn to_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("cache_bytes", self.cache_bytes)?;
        dict.set_item("batch_buffer_bytes", self.batch_buffer_bytes)?;
        dict.set_item("lookahead_overhead_bytes", self.lookahead_overhead_bytes)?;
        dict.set_item("transient_bytes", self.transient_bytes)?;
        dict.set_item("python_overhead_bytes", self.python_overhead_bytes)?;
        dict.set_item("total_bytes", self.total_bytes)?;
        Ok(dict)
    }
}

/// Returns `true` if `SCX_LOADER_PROFILE` is set to `"1"` or `"true"`.
///
/// The sole reader of the variable. Every profiling call site — `pipeline.rs`,
/// `decode_stage.rs`, `io_stage.rs`, `index_plan.rs` — reads it once at stage
/// scope through this function and passes the resulting `bool` down, including
/// into `io_stage`'s `spawn_blocking` closures.
pub(crate) fn profiling_enabled() -> bool {
    std::env::var("SCX_LOADER_PROFILE")
        .map(|v| v == "1" || v == "true")
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// The shared auto-tune driver (ORG-9.10-5)
// ---------------------------------------------------------------------------

/// What a loader configuration costs, and how to make it cheaper.
///
/// The three loader classes disagree about *which* knobs they tune, in what
/// order, and what to do when the knobs run out — those are real differences
/// and stay with the implementations. What they had also forked was the loop
/// itself, the comparison against the budget, and whether the model was
/// monotone at all; [`tune`] owns all three.
///
/// **Contract: `estimate` must be monotone under `reduce`.** A step that raises
/// the estimate would make the descent non-terminating in spirit (it can only
/// ever bottom out at the floor) and, worse, breaks
/// `MultimodalTrainingDataset`'s pin, which relies on "a smaller config always
/// fits within the same budget" to force every modality onto the cross-modality
/// minimum. [`tune`] `debug_assert`s the property on every step and
/// [`assert_monotone_reduction_chain`] checks the whole chain in tests, so it is
/// a checked property rather than the unstated assumption it used to be.
pub(crate) trait BudgetModel {
    /// The knobs this model tunes.
    type Params: Copy + std::fmt::Debug;

    /// Per-component cost of `params`. Must not count mmap-resident pages —
    /// every path treats the kernel page cache as evictable and excludes it
    /// (see [`BudgetBreakdown`]).
    fn estimate(&self, params: Self::Params) -> BudgetBreakdown;

    /// The next-cheaper knob setting in this model's reduction order, or `None`
    /// at the floor. Must not raise [`Self::estimate`].
    fn reduce(&self, params: Self::Params) -> Option<Self::Params>;
}

/// The outcome of an auto-tune.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Tuned<P> {
    /// The knob setting that fits, or the floor when `exhausted`.
    pub(crate) params: P,
    /// `estimate(params)` — what the caller should report.
    pub(crate) breakdown: BudgetBreakdown,
    /// `true` when the model ran out of knobs before fitting. **What to do
    /// about it is the caller's decision, deliberately**: the sequential path
    /// flags it and continues (and raises a `UserWarning`), `IndexPlanLoader`
    /// refuses construction, and `SparseCellSetLoader` bottoms out at its
    /// one-shard floor and warns. Collapsing those into one policy would change
    /// three user-visible behaviours to no end.
    pub(crate) exhausted: bool,
}

/// Shrink `requested` along the model's reduction order until it fits
/// `budget_bytes`, or until the model runs out of knobs.
pub(crate) fn tune<M: BudgetModel>(
    model: &M,
    requested: M::Params,
    budget_bytes: usize,
) -> Tuned<M::Params> {
    let mut params = requested;
    let mut breakdown = model.estimate(params);
    loop {
        if breakdown.fits_within_bytes(budget_bytes) {
            return Tuned {
                params,
                breakdown,
                exhausted: false,
            };
        }
        let Some(next) = model.reduce(params) else {
            return Tuned {
                params,
                breakdown,
                exhausted: true,
            };
        };
        let next_breakdown = model.estimate(next);
        debug_assert!(
            next_breakdown.total_bytes <= breakdown.total_bytes,
            "BudgetModel is not monotone: {params:?} -> {next:?} raised the estimate \
             from {} to {} bytes",
            breakdown.total_bytes,
            next_breakdown.total_bytes,
        );
        params = next;
        breakdown = next_breakdown;
    }
}

/// Walk `model`'s entire reduction chain from `requested` and assert the
/// estimate never rises.
///
/// The shared harness behind every model's monotonicity test. Models are plain
/// value types, so this runs with no fixture, no file and no interpreter.
#[cfg(test)]
pub(crate) fn assert_monotone_reduction_chain<M: BudgetModel>(model: &M, requested: M::Params) {
    let mut params = requested;
    let mut prev = model.estimate(params);
    let mut steps = 0usize;
    while let Some(next) = model.reduce(params) {
        let cur = model.estimate(next);
        assert!(
            cur.total_bytes <= prev.total_bytes,
            "not monotone: {params:?} ({} bytes) -> {next:?} ({} bytes)",
            prev.total_bytes,
            cur.total_bytes,
        );
        params = next;
        prev = cur;
        steps += 1;
        assert!(steps < 5_000_000, "reduction chain did not terminate");
    }
    assert!(
        steps > 0,
        "the chain had no steps from {requested:?}, so this proves nothing"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breakdown_total_is_saturating_sum() {
        let b = BudgetBreakdown::new(100, 200, 30, 40, 50);
        assert_eq!(b.total_bytes, 100 + 200 + 30 + 40 + 50);
    }

    #[test]
    fn breakdown_fits_within_threshold() {
        let b = BudgetBreakdown::new(0, 0, 0, 0, 1024 * 1024); // 1 MiB
        assert!(b.fits_within(1));
        assert!(b.fits_within(2));
    }

    #[test]
    fn breakdown_does_not_fit_above_threshold() {
        let b = BudgetBreakdown::new(0, 0, 0, 0, 2 * 1024 * 1024); // 2 MiB
        assert!(!b.fits_within(1));
    }

    #[test]
    fn breakdown_handles_overflow_safely() {
        let b = BudgetBreakdown::new(usize::MAX, 1, 0, 0, 0);
        assert_eq!(b.total_bytes, usize::MAX);
    }

    // ---- adaptive_budget_mb ------------------------------------------------

    #[test]
    fn adaptive_budget_never_drops_below_the_floor() {
        // A tiny need must not tighten the budget below the historical default.
        assert_eq!(adaptive_budget_mb(1024, 512), 512);
    }

    #[test]
    fn adaptive_budget_raises_to_fit_a_large_request() {
        // 1000 MB need + 12.5% headroom = 1125, above the 512 floor, below cap.
        let mb = adaptive_budget_mb(1000 * 1024 * 1024, 512);
        assert_eq!(mb, 1125);
    }

    #[test]
    fn adaptive_budget_clamps_to_the_cap() {
        let mb = adaptive_budget_mb(usize::MAX / 2, 512);
        assert_eq!(mb, crate::pipeline::ADAPTIVE_BUDGET_CAP_MB);
    }

    #[test]
    fn adaptive_budget_floor_above_cap_does_not_panic() {
        // `clamp` panics when min > max; the floor must win instead.
        let floor = crate::pipeline::ADAPTIVE_BUDGET_CAP_MB + 4096;
        assert_eq!(adaptive_budget_mb(1024, floor), floor);
    }

    // ---- assess_cache_sizing ---------------------------------------------

    #[test]
    fn sizing_silent_when_request_is_honoured() {
        assert!(assess_cache_sizing(128, 128, 1024, 512, 0, true).is_none());
        // Defensive: an effective value *above* the request is still silence.
        assert!(assess_cache_sizing(64, 128, 1024, 512, 0, true).is_none());
    }

    #[test]
    fn sizing_silent_under_an_adaptive_budget_above_the_floor() {
        // The false positive a preflight caught on real data: 194 MB shards mean
        // the adaptive 4 GB cap affords 21 of 128, on a run whose observed RSS was
        // 0.7 GB. That is the cap working, not a problem — warning there would
        // fire on healthy default configurations.
        assert!(
            assess_cache_sizing(128, 21, 194 * 1024 * 1024, 4096, 0, false).is_none(),
            "an adaptive budget's own cap must not warn while above the floor"
        );
        // The same numbers WITH an explicit budget are worth reporting: the
        // caller asked for two things that don't fit.
        assert!(assess_cache_sizing(128, 21, 194 * 1024 * 1024, 4096, 0, true).is_some());
    }

    #[test]
    fn sizing_reports_below_floor_even_under_an_adaptive_budget() {
        // Below MIN_CACHE_SHARDS thrash is likely rather than hypothetical, so
        // the adaptive suppression lifts.
        let v = assess_cache_sizing(128, 4, 1024 * 1024, 8, 0, false)
            .expect("below the floor must be reported even when adaptive");
        assert!(v.below_floor);
    }

    #[test]
    fn sizing_reports_shrink_above_the_floor_without_flagging_the_floor() {
        let v = assess_cache_sizing(128, 32, 1024 * 1024, 64, 0, true)
            .expect("shrink must be reported");
        assert_eq!(v.requested_cache_shards, 128);
        assert_eq!(v.effective_cache_shards, 32);
        assert!(
            !v.below_floor,
            "32 is above MIN_CACHE_SHARDS; the escalated wording must not fire"
        );
        // 128 shards × 1 MiB = 128 MB would have held the request.
        assert_eq!(v.budget_mb_for_requested, 128);
    }

    #[test]
    fn sizing_flags_below_floor() {
        let v =
            assess_cache_sizing(128, 4, 1024 * 1024, 8, 0, true).expect("shrink must be reported");
        assert!(v.below_floor, "4 < MIN_CACHE_SHARDS must escalate");
    }

    #[test]
    fn sizing_counts_non_cache_terms_in_the_suggested_budget() {
        // The suggested budget must cover the batch buffers and Python overhead
        // too, or following the advice still would not fit.
        let v = assess_cache_sizing(16, 4, 1024 * 1024, 8, 100 * 1024 * 1024, true).unwrap();
        assert_eq!(v.budget_mb_for_requested, 16 + 100);
    }

    // ---- assess_cache_thrash --------------------------------------------

    fn metrics(hits: u64, misses: u64, evictions: u64) -> CacheMetrics {
        let m = CacheMetrics::default();
        m.hits.store(hits, Ordering::Relaxed);
        m.misses.store(misses, Ordering::Relaxed);
        m.evictions.store(evictions, Ordering::Relaxed);
        m
    }

    #[test]
    fn thrash_detected_when_misses_displace_live_entries() {
        let v = assess_cache_thrash(&metrics(10, 1000, 900), 16).expect("thrash must be detected");
        assert!(v.miss_rate > 0.9);
        assert!(v.evictions_per_miss > 0.8);
        assert_eq!(v.cache_shards, 16);
    }

    #[test]
    fn cold_scan_is_not_thrash() {
        // The load-bearing negative case. A cold sequential scan has a ~100%
        // miss rate and *no* evictions: every read is a legitimate first touch.
        // A predicate keyed on miss rate alone would fire here and cry wolf on
        // every well-configured run.
        assert!(
            assess_cache_thrash(&metrics(0, 1000, 0), 128).is_none(),
            "high miss rate with no evictions is a cold cache, not thrash"
        );
        // Still not thrash when a few evictions trickle in near the end.
        assert!(assess_cache_thrash(&metrics(0, 1000, 100), 128).is_none());
    }

    #[test]
    fn warm_cache_is_not_thrash() {
        // Evictions can exceed misses on a healthy warm cache (a big entry
        // displaces several small ones), so the eviction term alone is not
        // sufficient either — the low miss rate must veto.
        assert!(assess_cache_thrash(&metrics(10_000, 200, 400), 128).is_none());
    }

    #[test]
    fn thrash_silent_below_the_warmup_floor() {
        // Same ratios as the positive case, too few samples to trust.
        assert!(assess_cache_thrash(&metrics(0, 63, 60), 4).is_none());
        assert!(assess_cache_thrash(&metrics(0, 64, 60), 4).is_some());
    }

    #[test]
    fn thrash_silent_on_zeroed_metrics() {
        // `PrefetchEngine::new` installs a zeroed handle when no reader enabled
        // metrics; that must read as "no signal", never as "no thrash proven".
        assert!(assess_cache_thrash(&CacheMetrics::default(), 128).is_none());
    }
}
