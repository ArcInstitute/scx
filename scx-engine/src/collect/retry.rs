//! Resilient parallel shard map.
//!
//! Generic infrastructure — nothing here knows what a query is. It exists
//! because a single shard GET exhausting its per-request retry budget must not
//! abort an atlas-scale query via `?`-propagation.

use rayon::prelude::*;

use crate::error::{EngineError, Result};

/// Is `e` worth a second attempt? Transient I/O failures are — on the cloud
/// path every `CloudError` (timeout, download-failed, rate-limited) is erased
/// to [`EngineError::IoError`] at the `CloudSectionReader` boundary, and
/// `Generic` covers other transient surfaces. Deterministic decode failures
/// (`FormatError` / `ArrowError` / `CsrError` / `SchemaError`) are NOT: a retry
/// would just re-fail and double the work on genuinely corrupt input.
fn is_retryable_engine_err(e: &EngineError) -> bool {
    matches!(e, EngineError::IoError(_) | EngineError::Generic(_))
}

/// Run `f` over `items` in parallel, tolerating transient per-item failures.
///
/// Unlike `items.par_iter().map(f).collect::<Result<Vec<_>>>()`, pass 1 does
/// NOT short-circuit on the first `Err`: it runs `f` on every item via rayon
/// and keeps each result paired with its input index. If everything succeeded,
/// the results are returned in input order.
///
/// If some items failed:
/// - A non-retryable failure ([`is_retryable_engine_err`] == false, i.e. a
///   deterministic decode error) is returned immediately — this preserves the
///   prior fast-fail on corrupt input.
/// - Otherwise the failed items are retried once, **sequentially**. Lower
///   concurrency on the retry pass gives a transient/congestion window time to
///   clear, and each call still gets a fresh per-request retry budget from the
///   cloud `RetryingStore`. The call fails only if an item is still unrecovered
///   after the retry pass, and the error then names the offending indices
///   rather than surfacing just the first error.
///
/// This keeps a single shard GET exhausting its per-request retry budget from
/// aborting a whole atlas-scale query via `?`-propagation. `f` may be invoked
/// up to twice per item, so it must be idempotent.
pub(crate) fn par_map_with_shard_retry<I, T, F>(items: &[I], f: F) -> Result<Vec<T>>
where
    I: Sync,
    T: Send,
    F: Fn(&I) -> Result<T> + Sync,
{
    // Pass 1: parallel, order-preserving (slice `par_iter` is indexed), keep
    // every Result so a single failure doesn't discard the other shards' work.
    let pass1: Vec<Result<T>> = items.par_iter().map(&f).collect();

    let mut slots: Vec<Option<T>> = Vec::with_capacity(items.len());
    let mut failed: Vec<usize> = Vec::new();
    let mut first_nonretryable: Option<EngineError> = None;
    for (i, r) in pass1.into_iter().enumerate() {
        match r {
            Ok(v) => slots.push(Some(v)),
            Err(e) => {
                if !is_retryable_engine_err(&e) && first_nonretryable.is_none() {
                    first_nonretryable = Some(e);
                }
                slots.push(None);
                failed.push(i);
            }
        }
    }

    // A deterministic decode failure is not worth a second attempt — fail fast.
    if let Some(e) = first_nonretryable {
        return Err(e);
    }

    if !failed.is_empty() {
        let mut last_err: Option<EngineError> = None;
        let mut unrecovered: Vec<usize> = Vec::new();
        for &i in &failed {
            match f(&items[i]) {
                Ok(v) => slots[i] = Some(v),
                Err(e) => {
                    last_err = Some(e);
                    unrecovered.push(i);
                }
            }
        }
        if !unrecovered.is_empty() {
            return Err(EngineError::Generic(format!(
                "shard read failed after retry for {} of {} item(s) (indices {:?}): {}",
                unrecovered.len(),
                items.len(),
                unrecovered,
                last_err
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "unknown error".to_string()),
            )));
        }
    }

    Ok(slots
        .into_iter()
        .map(|s| s.expect("every slot filled by pass 1 success or retry recovery"))
        .collect())
}
