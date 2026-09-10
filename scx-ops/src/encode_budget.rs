//! What one shard costs while it is being encoded, and how many may be in
//! flight at once.
//!
//! Every rewrite op that encodes shards off the calling thread needs the same
//! two answers, and before this module `scx sort` was the only one that had
//! them — privately. The cost model here is the crate's single declaration of
//! "bytes per nonzero for a shard in flight", the way
//! `scx-convert/src/budget.rs` is that crate's.
//!
//! **This is an estimate, not a bound.** The same honesty applies here as in
//! that table, where every allocation row stays `enforced: false`: the figure
//! prices the buffers a shard's encode is known to hold, and a peak can exceed
//! it — allocator behaviour, a codec's internal workspace, and the shard being
//! gathered while others encode are all outside it. What a `--memory-budget`
//! buys is that the in-flight count stops scaling with the host's core count.
//! The floor is one shard: a single shard's encode is irreducible, so the
//! concurrency clamps to 1 rather than refusing the op.

use crate::sort_engine::GROUP_BYTES_PER_NNZ;

/// Bytes one in-flight shard costs per nonzero, across its **whole** phase:
/// the gather buffers ([`GROUP_BYTES_PER_NNZ`]), the encoder's own copy of the
/// values, and the framed encode's live buffers.
///
/// `1 + 1 + 4`, the same derivation `scx-convert/src/budget.rs` carries for its
/// **ingest** derate — export holds no encode buffers and is sized by its own
/// decode-phase model there. `encode_shard_framed` holds every row group's
/// encoded bytes alongside the streams assembled from them (2x), and
/// `encode_shard_adaptive` runs two candidates under `rayon::join` (x2), each
/// bounded by its input because every codec here is a compressor or a 1:1 copy.
///
/// **Declared here rather than shared** with that table: `scx-convert` depends
/// on `scx-ops`, so the dependency cannot run the other way, and every constant
/// in `budget.rs` is `pub(crate)`. This is the same deliberate duplication as
/// `GROUP_BYTES_PER_NNZ` (8) beside `budget::PAYLOAD_BYTES_PER_NNZ` (8) — one
/// "i32 + f32" model, declared once per crate. Change one and change the other.
pub(crate) const ENCODE_PHASE_MULTIPLE: u64 = 6;

/// [`ENCODE_PHASE_MULTIPLE`] in bytes per nonzero — 48 at the current model,
/// matching `scx-convert`'s per-worker sparse-ingest figure for the same reason
/// (the same phase, priced the same way).
pub(crate) fn encode_phase_bytes(nnz: u64) -> u64 {
    nnz.saturating_mul(ENCODE_PHASE_MULTIPLE)
        .saturating_mul(GROUP_BYTES_PER_NNZ)
}

/// Bytes of in-flight encode phase allowed when the caller sets no
/// `--memory-budget`.
///
/// **Not a tuning knob — a blast-radius limit.** Encoding N shards at once
/// costs N times one shard's phase, so a default of "fill the pool" would make
/// an op's peak RSS scale with the host's core count. Measured, that is not
/// hypothetical: a census_1m shard carries ~8.2M nonzeros, which
/// [`encode_phase_bytes`] prices at ~394 MB, so sixteen in flight would add
/// ~6.3 GB to an op whose serial peak was ~3.3 GB. A file whose shards are
/// small (pbmc3k, smartseq2) still fills the pool, because the cap is in bytes
/// rather than shards.
///
/// 1 GiB, chosen the way `sort`'s 256 MiB `DEFAULT_GROUP_WRITE_BLOCK_BYTES`
/// was: large enough to be useful on ordinary files, small enough that the
/// default peak increase is a constant a reader can hold in their head. A
/// caller who wants more concurrency on deep shards raises `--memory-budget`;
/// the wall/peak trade is theirs to make, and the per-shard figure above is
/// what lets them compute it.
pub(crate) const DEFAULT_IN_FLIGHT_BYTES: u64 = 1024 * 1024 * 1024;

/// Resolve the caller's `--memory-budget` into an in-flight byte allowance.
///
/// `None` is the default on every surface, and it means
/// [`DEFAULT_IN_FLIGHT_BYTES`] rather than "unbounded" — see that constant for
/// why. An explicit budget is used as given, `0` included (which pins the
/// concurrency to one shard).
pub(crate) fn resolve_in_flight_budget(memory_budget: Option<u64>) -> u64 {
    memory_budget.unwrap_or(DEFAULT_IN_FLIGHT_BYTES)
}

/// Group consecutive shards into chunks that may be encoded concurrently.
///
/// Returns one length per chunk, in order, covering `nnz` exactly. A chunk is
/// closed when adding the next shard would either exceed `threads` (there is no
/// point holding more shards than can be worked on) or push the chunk's whole
/// live phase past `in_flight_bytes`. A chunk always holds at least one shard,
/// even one whose own phase exceeds the allowance — see the module note on the
/// floor.
///
/// `in_flight_bytes` is a resolved number, never an `Option`, so the default
/// and the caller's budget travel the same path and are testable against each
/// other; [`resolve_in_flight_budget`] maps one to the other.
///
/// Taking the *whole* list rather than answering one shard at a time is what
/// lets this be tested against a real distribution of shard sizes, including
/// the skewed one a permutation or a deletion filter produces, without a file.
pub(crate) fn plan_encode_chunks(nnz: &[u64], threads: usize, in_flight_bytes: u64) -> Vec<usize> {
    let threads = threads.max(1);
    let mut chunks = Vec::new();
    let mut i = 0usize;
    while i < nnz.len() {
        // The first shard is always admitted, so a chunk is never empty and the
        // loop always advances.
        let mut len = 1usize;
        let mut bytes = encode_phase_bytes(nnz[i]);
        while i + len < nnz.len() && len < threads {
            let next = encode_phase_bytes(nnz[i + len]);
            if bytes.saturating_add(next) > in_flight_bytes {
                break;
            }
            bytes = bytes.saturating_add(next);
            len += 1;
        }
        chunks.push(len);
        i += len;
    }
    chunks
}

#[cfg(test)]
#[path = "encode_budget_tests.rs"]
mod tests;
