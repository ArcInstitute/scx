//! The `UserWarning` layer and the cache-thrash sampler.
//!
//! Split out of `python.rs` by ORG-9.10-2; a pure move.

use std::sync::atomic::{AtomicBool, Ordering};

use scx_format_io::CacheMetrics;

use super::*;
// Named explicitly rather than arriving through a parent glob: this is the
// one cross-sibling dependency in the warning layer, and it should be visible
// at the top of the file that has it.
use super::multimodal::{min_total_mb_clearing_the_floor, MIN_MODALITY_BUDGET_MB};

/// Should the unframed-scatter preflight warn — and, just as importantly, should
/// the framing scan run at all?
///
/// `any_framed` is a closure, and the conjunct order is the whole point.
/// `BackedCsrReader::any_shard_framed` reads a shard header per shard, and on an
/// all-unframed file — the one case that cannot short-circuit — it reads every
/// one. With the caller's kwarg off, or the process-global
/// `SCX_SCATTER_BLOCK_INDEX` switch off, there is no warning to emit *and*
/// reframing could not enable the route either, so that scan is pure cost.
///
/// Extracted rather than spelled out at the two constructors so a test can
/// assert the closure was never called. From Python it is not observable: the
/// switch is memoized in a `OnceLock`, so a subprocess can only show that no
/// warning was emitted — which was already true when the check sat *after* the
/// scan (found by codex in review round 2, against a test of mine that could
/// not fail).
pub(super) fn should_warn_unframed_scatter(
    requested: bool,
    globally_enabled: bool,
    any_framed: impl FnOnce() -> bool,
) -> bool {
    requested && globally_enabled && !any_framed()
}

/// Preflight `UserWarning` for `scatter_block_index=True` against a file (or a
/// set of files) where no CSR shard is row-group framed.
///
/// The block-index fast path can only fire on framed shards, and only when the
/// process-global `SCX_SCATTER_BLOCK_INDEX` switch is on. If the caller asked
/// for it, the switch is enabled, and the data is an all-unframed legacy
/// layout, then every batch full-shard-decodes with no other signal — warn and
/// point at the reframe command. When the switch is off, reframing cannot
/// enable the path either, so there is nothing to warn about.
///
/// `target` describes what was opened — a quoted path on the single-file
/// classes, a count plus a sample on the multi-file one — and carries the
/// deduplication: Python's default filter keys on (message, category, module,
/// lineno), so embedding the paths makes this one-shot per dataset rather than
/// per process.
///
/// Callers reach this through [`should_warn_unframed_scatter`], which is what
/// keeps the process-global switch ahead of the framing scan. The switch is
/// re-checked here as a live backstop — a `OnceLock` read — so a future caller
/// that forgets the precheck emits nothing rather than warning while the route
/// is globally disabled. It is a backstop, not the gate: it cannot un-spend the
/// scan a caller has already paid for.
///
/// Same house style as [`warn_cache_sizing`]: warn and continue, never refuse.
pub(super) fn warn_unframed_scatter(py: Python<'_>, dataset: &str, target: &str) -> PyResult<()> {
    if !scx_format_io::backed::scatter_block_index_enabled() {
        return Ok(());
    }
    let warnings = crate::pyimport::import_module(py, "warnings")?;
    let user_warning = crate::pyimport::import_module(py, "builtins")?.getattr("UserWarning")?;
    let msg = format!(
        "{dataset} opened {target} with scatter_block_index=True, but no \
         CSR shard is row-group framed (unframed legacy file). Scattered reads \
         will full-shard-decode every batch — the block-index fast path cannot \
         fire. Reframe with `scx optimize --row-group-rows 256 <file>`, or pass \
         scatter_block_index=False to silence this warning."
    );
    warnings.call_method1("warn", (msg, user_warning))?;
    Ok(())
}

/// Emit the construction-time `UserWarning` for a shard cache the memory budget
/// could not afford.
///
/// Follows the house style of the `scatter_block_index` preflight above: warn and
/// continue (never refuse), name the observed numbers, name the knob, and name
/// the exact value that would fix it. Deduping is left to CPython's
/// `__warningregistry__`, which is correct here because the message text is
/// fixed per `(dataset, requested, effective, budget)` — unlike the runtime
/// thrash warning, whose counters vary every call and which therefore needs its
/// own latch.
pub(super) fn warn_cache_sizing(
    py: Python<'_>,
    dataset: &str,
    v: &crate::budget::CacheSizingVerdict,
) -> PyResult<()> {
    let warnings = crate::pyimport::import_module(py, "warnings")?;
    let user_warning = crate::pyimport::import_module(py, "builtins")?.getattr("UserWarning")?;
    let consequence = if v.below_floor {
        format!(
            " That is below the {} shards a gather batch typically touches, so \
             every batch will re-decode shards it just evicted (the pathology \
             behind STATE3's 143 s/batch).",
            crate::budget::MIN_CACHE_SHARDS
        )
    } else {
        String::new()
    };
    let msg = format!(
        "{dataset}: max_memory_mb={} affords only {} of the {} requested \
         cache_shards (avg decoded shard = {} KB).{consequence} Pass \
         max_memory_mb>={} to hold the requested cache, or lower cache_shards \
         to {} to make the reduction explicit.",
        v.budget_mb,
        v.effective_cache_shards,
        v.requested_cache_shards,
        v.shard_decoded_bytes / 1024,
        v.budget_mb_for_requested,
        v.effective_cache_shards,
    );
    warnings.call_method1("warn", (msg, user_warning))?;
    Ok(())
}

/// Emit the construction-time `UserWarning` for per-modality budget floors that
/// sum to more than the caller asked for.
///
/// ORG-9.10-5. `MultimodalTrainingDataset` divides `max_memory_mb` across
/// modalities in proportion to their nnz, then raises each share to the floor
/// `LoaderConfig::validate` requires — so a request that cannot be divided
/// without starving someone is silently rounded *up*, by up to
/// `MIN_MODALITY_BUDGET_MB` per modality. The split is not the thing to change
/// (a share below the floor fails construction outright); being silent about it
/// was.
pub(super) fn warn_modality_budget_floors(
    py: Python<'_>,
    requested_mb: usize,
    effective_mb: usize,
    per_modality_nnz: &[u64],
) -> PyResult<()> {
    let warnings = crate::pyimport::import_module(py, "warnings")?;
    let user_warning = crate::pyimport::import_module(py, "builtins")?.getattr("UserWarning")?;
    let n_modalities = per_modality_nnz.len();
    // Quote the fixed point, not the current effective sum: the floor is
    // re-applied after the new split, so echoing `effective_mb` back would
    // advise a value that fires this same warning again.
    let advice = match min_total_mb_clearing_the_floor(per_modality_nnz) {
        Some(mb) => {
            format!("Pass max_memory_mb>={mb} for a split that clears the floor on its own.")
        }
        None => "A modality with no recorded nnz gets a zero proportional share at \
             every budget, so raising max_memory_mb cannot lift it off the floor."
            .to_string(),
    };
    let msg = format!(
        "MultimodalTrainingDataset: max_memory_mb={requested_mb} cannot be split across \
         {n_modalities} modalities without a share falling below the \
         {MIN_MODALITY_BUDGET_MB} MB per-modality floor, so the effective total is \
         {effective_mb} MB. Peak RSS is budgeted against that number, not the one you \
         passed; see memory_budget()['effective_total_mb']. {advice}"
    );
    warnings.call_method1("warn", (msg, user_warning))?;
    Ok(())
}

/// Emit the construction-time `UserWarning` for a budget the auto-tune could
/// not meet even at its minimums.
///
/// ORG-9.10-5. This fact was only ever a `log::warn!`, which a notebook or a
/// training script does not show, while the two plan-driven classes have always
/// raised a `UserWarning` for the equivalent (`warn_cache_sizing`) — so the one
/// class that *silently exceeds the budget it was given* was the quiet one.
///
/// Deliberately **not** extended to `shuffle_quality_degraded`: that is a
/// shuffle-entropy quality note rather than a memory verdict, it fires on
/// ordinary tight-budget runs, and a warning users learn to filter stops
/// working. It stays a `log::warn!`.
pub(super) fn warn_budget_exceeded(
    py: Python<'_>,
    dataset: &str,
    max_memory_mb: usize,
    budget: &crate::pipeline::MemoryBudget,
) -> PyResult<()> {
    let warnings = crate::pyimport::import_module(py, "warnings")?;
    let user_warning = crate::pyimport::import_module(py, "builtins")?.getattr("UserWarning")?;
    let msg = format!(
        "{dataset}: max_memory_mb={} cannot hold this file even at the auto-tune's \
         minimums (batch_size={}, shard_group_size={}, prefetch_batches={}); the \
         estimate is {} MB. Peak RSS will exceed the budget. Pass \
         max_memory_mb>={} to fit, or set hvg_indices to shrink the batch.",
        max_memory_mb,
        budget.batch_size,
        budget.shard_group_size,
        budget.prefetch_batches,
        budget.estimated_bytes.div_ceil(1024 * 1024),
        budget.estimated_bytes.div_ceil(1024 * 1024),
    );
    warnings.call_method1("warn", (msg, user_warning))?;
    Ok(())
}

/// Emit the construction-time `UserWarning` for an `hvg_indices` panel that the
/// projection had to canonicalise.
///
/// Same house style as [`warn_cache_sizing`]: warn and continue, name the
/// observed numbers, say what the caller actually gets. Silent for a panel that
/// is already ascending and unique — which the documented recipe,
/// `np.where(adata.var["highly_variable"])[0]`, always is — so this fires only
/// when the caller's own column→gene mapping has genuinely diverged from the
/// batch's.
pub(super) fn warn_hvg_panel(
    py: Python<'_>,
    dataset: &str,
    v: &crate::projection::HvgPanelVerdict,
) -> PyResult<()> {
    let warnings = crate::pyimport::import_module(py, "warnings")?;
    let user_warning = crate::pyimport::import_module(py, "builtins")?.getattr("UserWarning")?;
    let mut parts: Vec<String> = Vec::new();
    if v.was_reordered {
        parts.push(
            "sorted into ascending gene-index order, so batch columns are NOT in the \
             order you passed"
                .to_string(),
        );
    }
    if v.unique_len != v.requested_len {
        parts.push(format!(
            "deduplicated from {} to {} entries, so the batch is {} columns wide",
            v.requested_len, v.unique_len, v.unique_len
        ));
    }
    let msg = format!(
        "{dataset}: hvg_indices was {}. Index your gene names with \
         `np.unique(hvg_indices)` to recover the column order the batches actually use.",
        parts.join(" and "),
    );
    warnings.call_method1("warn", (msg, user_warning))?;
    Ok(())
}

/// Samples cache counters while an iterator drains and warns once if they look
/// like working-set overflow.
///
/// Lives on the iterator rather than the dataset because thrash is a property of
/// the *plans being consumed*, not of the file: the same dataset can be
/// well-sized for a consecutive-obs manifest and badly sized for a scattered
/// perturbation gather (STATE3 measured 19× between those two on one file).
pub(super) struct ThrashSampler {
    dataset: &'static str,
    cache_shards: usize,
    /// Total distinct cache entries the file(s) can ever hold — summed over
    /// files, since the shared cache is keyed `(file_id, shard)`. Caps the
    /// suggested size: under thrash the same shard is re-requested many times
    /// per batch, so a requests-per-batch estimate *overcounts* the working set
    /// (measured: 174 suggested on a 31-shard file). Caching more entries than
    /// exist is meaningless, so this is both a correct bound and a tight one.
    total_shards: usize,
    /// `true` when the **byte** budget, not the count cap, is what limits
    /// residency. Decides which knob the advice leads with: raising
    /// `cache_shards` cannot help a byte-bound cache, so leading with it there is
    /// a false primary diagnosis (round-2 review, Cursor P3).
    byte_bound: bool,
    /// Average decoded bytes per shard, so the advice can name a concrete
    /// `max_memory_mb` rather than telling the caller to "raise" it.
    shard_decoded_bytes: usize,
    batches: u64,
    /// One-shot: the message embeds live counters, so every call would be a
    /// distinct message text and CPython's per-text dedupe would not suppress
    /// it. Mirrors the `AtomicBool` latch in `pyscx/src/accel/rapids.rs`.
    warned: AtomicBool,
}

/// Batch interval between thrash checks.
///
/// **Must be small enough to fire inside a short run.** A first version used 32
/// and never sampled at all on the benchmark's own scattered-gather workload,
/// which runs **30** batches (`cellset_gather::_n_batches_for` floors at
/// `_MIN_N_BATCHES = 30` for census-scale files) — the diagnostic was silently
/// dead on exactly the case it was built for. 8 gives three checks in a 30-batch
/// run while still costing only three relaxed atomic loads per check, and
/// premature verdicts are suppressed by `budget::THRASH_MIN_MISSES` rather than
/// by the cadence: a scattered batch issues tens of shard reads, so 8 batches is
/// already hundreds of samples.
const THRASH_SAMPLE_EVERY: u64 = 8;

impl ThrashSampler {
    pub(super) fn new(
        dataset: &'static str,
        cache_shards: usize,
        total_shards: usize,
        byte_bound: bool,
        shard_decoded_bytes: usize,
    ) -> Self {
        ThrashSampler {
            dataset,
            cache_shards,
            total_shards,
            byte_bound,
            shard_decoded_bytes,
            batches: 0,
            warned: AtomicBool::new(false),
        }
    }

    /// Call once per yielded batch. Cheap until the sampling interval is hit,
    /// and a no-op once it has warned.
    pub(super) fn observe(&mut self, py: Python<'_>, m: &CacheMetrics) {
        self.batches += 1;
        if !self.batches.is_multiple_of(THRASH_SAMPLE_EVERY) || self.warned.load(Ordering::Relaxed)
        {
            return;
        }
        let Some(v) = crate::budget::assess_cache_thrash(m, self.cache_shards) else {
            return;
        };
        // Average shard *requests* per batch, capped at the number of shards
        // that exist. The raw ratio overcounts badly under thrash (the same
        // shard is re-requested after each eviction), and suggesting more
        // entries than the file has is meaningless — so the cap is what makes
        // this actionable rather than merely derived.
        let per_batch = ((v.hits + v.misses).div_ceil(self.batches.max(1)) as usize)
            .min(self.total_shards.max(1))
            .max(self.cache_shards + 1);
        // A warning must never break iteration: if emitting it raises (e.g. the
        // caller turned UserWarning into an error via `simplefilter`), that is
        // the caller's chosen behaviour for warnings, and propagating it from
        // `__next__` would corrupt an otherwise healthy training loop. Record
        // that we warned either way.
        let _ = warn_cache_thrash(
            py,
            self.dataset,
            &v,
            per_batch,
            self.byte_bound,
            self.shard_decoded_bytes,
            &self.warned,
        );
    }
}

/// Emit the runtime `UserWarning` for observed cache thrash. Latches so it fires
/// at most once per iterator.
fn warn_cache_thrash(
    py: Python<'_>,
    dataset: &str,
    v: &crate::budget::ThrashVerdict,
    suggested_cache_shards: usize,
    byte_bound: bool,
    shard_decoded_bytes: usize,
    latch: &AtomicBool,
) -> PyResult<()> {
    if latch.swap(true, Ordering::Relaxed) {
        return Ok(());
    }
    let warnings = crate::pyimport::import_module(py, "warnings")?;
    let user_warning = crate::pyimport::import_module(py, "builtins")?.getattr("UserWarning")?;
    // Lead with the knob that can actually fix it. When the byte budget is the
    // limiter, `cache_shards` is already at or above what is resident-capable and
    // raising it changes nothing — advising it first is a false diagnosis even
    // though the sentence is technically hedged.
    let fix = if byte_bound {
        let need_mb = suggested_cache_shards
            .saturating_mul(shard_decoded_bytes)
            .div_ceil(1024 * 1024)
            .max(1);
        format!(
            "The BYTE budget is the limiter here, not the count: raise \
             max_memory_mb to >={need_mb} (enough for ~{suggested_cache_shards} \
             shards of ~{} KB). Raising cache_shards alone cannot help",
            shard_decoded_bytes / 1024
        )
    } else {
        format!(
            "Try cache_shards>={suggested_cache_shards} (≈ the shards one batch \
             touches, estimated from this run; `suggested_cache_shards(plan)` \
             gives the exact count for a given plan), or raise max_memory_mb"
        )
    };
    let msg = format!(
        "{dataset}: shard-cache thrash detected — {:.0}% of {} shard reads missed \
         and {:.2} entries were evicted per miss, so the working set exceeds the \
         {}-entry cache. Throughput is likely dominated by re-decoding shards \
         that were just evicted. {fix}. Suppress with \
         warnings.filterwarnings('ignore', message='.*shard-cache thrash.*').",
        v.miss_rate * 100.0,
        v.hits + v.misses,
        v.evictions_per_miss,
        v.cache_shards,
    );
    warnings.call_method1("warn", (msg, user_warning))?;
    Ok(())
}

// These compile only under `--features python`, which a bare
// `cargo test -p scx-loader` does not enable — that run is 339 tests, not 350.
// CI's `cargo test --workspace --exclude rscx` *does* run them: pyscx is a
// workspace member depending on `scx-loader` with `features = ["python"]`, and
// resolver-v2 unification turns the feature on for this crate's test target
// too. Verified by name in that job's output. If pyscx ever stops being built
// alongside, this module goes silently unrun.
//
// The gap was 16 before Phase 9f and is 11 now: `layout_check_tests` moved to
// `pipeline.rs` with the pure function it tests, so those five no longer need
// the feature. The 11 that remain genuinely exercise binding code.
#[cfg(test)]
mod preflight_decision_tests {
    use super::should_warn_unframed_scatter;
    use std::cell::Cell;

    /// The framing scan must not run when the process-global switch is off.
    ///
    /// This is the property the Python-side subprocess test cannot establish.
    /// Before review round 2 the switch was checked *inside* the warning helper,
    /// i.e. after the scan: no warning was emitted either way, so the subprocess
    /// test passed on the broken ordering too. Asserting the closure went
    /// uncalled is what actually pins it.
    #[test]
    fn the_framing_scan_does_not_run_when_the_global_switch_is_off() {
        let scanned = Cell::new(false);
        let warn = should_warn_unframed_scatter(true, false, || {
            scanned.set(true);
            false
        });
        assert!(!warn, "the global switch off means nothing to warn about");
        assert!(
            !scanned.get(),
            "any_shard_framed() reads a shard header per shard; with the route \
             globally disabled that scan buys nothing and must be skipped"
        );
    }

    /// ...and not when the caller never asked for the route either.
    #[test]
    fn the_framing_scan_does_not_run_when_the_caller_did_not_ask() {
        let scanned = Cell::new(false);
        let warn = should_warn_unframed_scatter(false, true, || {
            scanned.set(true);
            false
        });
        assert!(!warn);
        assert!(!scanned.get());
    }

    /// The positive arm: both gates open, nothing framed → scan runs, warn.
    /// Without this the two negatives above are satisfied by a function that
    /// always returns false and never calls the closure at all.
    #[test]
    fn an_unframed_file_the_caller_asked_about_warns_after_scanning() {
        let scanned = Cell::new(false);
        let warn = should_warn_unframed_scatter(true, true, || {
            scanned.set(true);
            false
        });
        assert!(warn);
        assert!(scanned.get(), "the scan is what decides this case");
    }

    /// And a framed file is silent — the `any`-across-readers answer is what
    /// suppresses the warning, not a second gate.
    #[test]
    fn a_framed_file_is_silent() {
        assert!(!should_warn_unframed_scatter(true, true, || true));
    }
}
