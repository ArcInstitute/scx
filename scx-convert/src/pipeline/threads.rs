//! Reader-thread and queue-depth derating.
//!
//! These are the *consumers* of `crate::budget`'s allocation table, not more
//! budget arithmetic: `crate::budget` decides what fraction of a budget one
//! in-flight shard may claim, and this module decides how many shards that
//! allows in flight at once. Review 11.5 was the two disagreeing.

use super::error::ConvertError;
use crate::warnings::{ConvertWarning, WarningSink};

/// Resolve a `reader_threads` option to a concrete
/// worker count. `None` (auto) reads `RAYON_NUM_THREADS` if set, else
/// falls back to [`std::thread::available_parallelism`].
pub(crate) fn resolve_reader_threads(reader_threads: Option<usize>) -> usize {
    if let Some(n) = reader_threads {
        return n.max(1);
    }
    if let Ok(s) = std::env::var("RAYON_NUM_THREADS") {
        if let Ok(n) = s.parse::<usize>() {
            return n.max(1);
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Apply `memory_budget` to the requested `(reader_threads,
/// writer_queue_depth)` so the peak outstanding shards stay within the
/// budget. Returns `(granted_threads, granted_depth)`.
///
/// Peak outstanding shards in the parallel coordinators is
/// `granted_threads + granted_depth` (workers in-flight + reorder
/// buffer + bounded channel), each holding roughly `per_shard_bytes`.
/// The constraint is therefore `(granted_threads + granted_depth) ×
/// per_shard_bytes ≤ budget`. The derate prefers shrinking
/// `granted_depth` over `granted_threads` so the dispatcher stays on
/// the parallel route under tight budgets — collapsing
/// `granted_threads` to 1 would route through the sequential
/// coordinator and lose parallelism entirely. Both have a floor of 1
/// (a queue depth of zero would starve the writer).
///
/// Shared by the streaming **ingest** coordinator
/// (`run_streaming_writer_coordinator`) and the streaming **export**
/// coordinator (`h5ad::stream_write`); `shard_noun` / `remedy`
/// customise the refusal error for each caller.
/// Refuse (T4.7 — no silent cap) when a single shard's working set cannot fit
/// `memory_budget`. `per_shard_bytes == 0` (empty / stats-less shard) and an
/// unset budget are both no-ops. Shared by the parallel derate
/// ([`derate_threads_and_depth`]) and the sequential grouped-ranges dispatch
/// (M2) so every route rejects an oversized shard with the same message.
pub(crate) fn ensure_shard_fits_budget(
    memory_budget: Option<u64>,
    per_shard_bytes: u64,
    shard_noun: &str,
    remedy: &str,
) -> Result<(), ConvertError> {
    if let Some(budget) = memory_budget {
        if per_shard_bytes > 0 && per_shard_bytes > budget {
            return Err(ConvertError::Other(format!(
                "single {shard_noun} requires \u{2248} {per_shard_bytes} bytes \
                 but memory_budget is {budget}; {remedy}"
            )));
        }
    }
    Ok(())
}

pub(crate) fn derate_threads_and_depth(
    memory_budget: Option<u64>,
    per_shard_bytes: u64,
    requested_threads: usize,
    requested_depth: usize,
    shard_noun: &str,
    remedy: &str,
    sink: &mut WarningSink,
) -> Result<(usize, usize), ConvertError> {
    let Some(budget) = memory_budget else {
        return Ok((requested_threads, requested_depth));
    };
    if per_shard_bytes == 0 {
        // Empty shards (no rows / no stats) — nothing to cap.
        return Ok((requested_threads, requested_depth));
    }
    // Refuse when even a single shard cannot fit the budget (T4.7 — no silent
    // cap). Shared with the sequential grouped path (M2) so both routes reject
    // an oversized shard identically.
    ensure_shard_fits_budget(memory_budget, per_shard_bytes, shard_noun, remedy)?;
    // peak outstanding = threads + depth. Solve for the largest
    // outstanding ≤ budget / per_shard_bytes.
    let outstanding_max = (budget / per_shard_bytes).max(1) as usize;
    let requested_outstanding = requested_threads.saturating_add(requested_depth);
    if requested_outstanding <= outstanding_max {
        return Ok((requested_threads, requested_depth));
    }
    // Preserve parallelism: shrink depth first (floor 1), then shrink
    // threads only if necessary (floor 1). Reserving one slot for depth
    // and giving the rest to threads keeps `granted_threads > 1`
    // whenever `outstanding_max >= 3` (at `outstanding_max == 2`,
    // `granted_threads == 1` routes to the sequential coordinator), so the
    // dispatcher stays on the parallel route under tight budgets instead of
    // falling back to sequential.
    let granted_threads = outstanding_max
        .saturating_sub(1)
        .min(requested_threads)
        .max(1);
    let granted_depth = outstanding_max
        .saturating_sub(granted_threads)
        .min(requested_depth)
        .max(1);
    sink.emit(ConvertWarning::ReaderThreadsDerated {
        requested: requested_threads,
        granted: granted_threads,
        reason: format!(
            "memory_budget {budget} caps outstanding shards to {outstanding_max} \
             (per shard \u{2248} {per_shard_bytes} bytes); writer_queue_depth granted = \
             {granted_depth}"
        ),
    });
    Ok((granted_threads, granted_depth))
}
