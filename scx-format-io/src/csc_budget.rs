// What a `--memory-limit` buys the CSC sidecar builder.
//
// The shares and cost models for ingest live in `scx_convert::budget`, which
// owns the allocation table. These four live here instead, at the bottom of the
// stack, because the code that claims against them does:
// `csc_sidecar::write_csc_sidecar` and `ScxWriter::finish` are in this crate,
// and `scx-ops`' `run_build_csc` is above it but below `scx-convert`.
//
// They are *declared* in the allocation table rather than restated there —
// `scx-convert`'s rows cite these constants — so `enforced: true` is a claim
// about the same number production reads, and the per-phase sum test checks the
// number that is actually used. `scx-ops/src/encode_budget.rs` is the
// alternative and says so itself: a deliberate second copy of the model,
// kept in sync by hand.

use crate::mem::Share;

/// `u32` row + `f32` value: one nonzero staged in a bucket, and one nonzero in
/// an emitted shard's arrays.
pub const CSC_PAYLOAD_BYTES_PER_NNZ: u64 = 8;

/// Column-block buckets plus their block slack, held across the whole push
/// phase.
///
/// This is the one share the builder **enforces**: it spills against exactly
/// this figure, and the realized bound is it plus
/// `2 * n_buckets * block_capacity` of block slack (`block_capacity` is
/// `block_bytes` plus one maximal row block; the factor of two is because
/// sealing sweeps every bucket a pushed row touched before the spill loop
/// runs). `CscBuilderConfig` declares it and
/// `staged_bytes_never_exceed_the_declared_bound` asserts it.
///
/// Note this share is **live during the emit phase too** — see
/// [`CSC_EMIT_SHARE`].
pub const CSC_BUILD_BUCKET_SHARE: Share = Share::new(1, 2);

/// One source CSR shard decoded in flight at a time. (The `build-csc` path
/// also re-encoded it until it became an in-place append.)
pub const CSC_BUILD_INPUT_SHARE: Share = Share::new(1, 4);

/// The emitted shards' arrays, their raw-value buffer and the encoder's
/// streams.
///
/// A quarter, not the whole budget, and the reason is a correction: the emit
/// does **not** start with the buckets gone. `CscEmitter` owns every bucket
/// that has not been drained yet, and `build_bucket` materialises *all* the
/// shards one bucket owns (`shards_per_bucket`, three at the census layout)
/// before the first is yielded. So up to `CSC_BUILD_BUCKET_SHARE` of
/// un-spilled buckets is concurrent with this, which is why the emit is a
/// phase that *sums* with the bucket share rather than one that succeeds it.
///
/// What is no longer concurrent is the source shard: the last `push_shard`
/// has returned and its decode buffers are dropped before `finish()`.
pub const CSC_EMIT_SHARE: Share = Share::new(1, 4);

/// Bytes one decoded CSR shard of `nnz` nonzeros over `n_rows` rows occupies.
///
/// `indices` + `data` at [`CSC_PAYLOAD_BYTES_PER_NNZ`], plus an `i64` indptr
/// entry per row. It does **not** charge the decoder's scratch on the way to
/// those arrays; see the allocation table's row, which says so.
pub fn decoded_csr_shard_bytes(nnz: u64, n_rows: u64) -> u64 {
    nnz.saturating_mul(CSC_PAYLOAD_BYTES_PER_NNZ)
        .saturating_add(n_rows.saturating_add(1).saturating_mul(8))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The push phase holds the buckets and one in-flight source shard at the
    /// same time, so their shares must fit together. The emit phase is
    /// separate because the *source shard* is gone by then — but the buckets
    /// are NOT, so the allocation table sums this bucket share into the emit
    /// phase as well.
    #[test]
    fn the_two_push_phase_shares_fit_together() {
        let budget = 1 << 30;
        let claimed = CSC_BUILD_BUCKET_SHARE.of(budget) + CSC_BUILD_INPUT_SHARE.of(budget);
        assert!(claimed <= budget, "{claimed} > {budget}");
    }

    /// `min_budget_for` is the refusal predicate *and* the "raise it to at
    /// least N" message; a budget it reports as sufficient must actually be.
    #[test]
    fn min_budget_for_admits_the_shard_it_names() {
        for nnz in [1u64, 7, 1_000, 1_400_000_000] {
            let unit = decoded_csr_shard_bytes(nnz, 1_000_000);
            let need = CSC_BUILD_INPUT_SHARE.min_budget_for(unit);
            assert!(
                CSC_BUILD_INPUT_SHARE.of(need) >= unit,
                "nnz={nnz}: a {need}-byte budget yields {} for a {unit}-byte shard",
                CSC_BUILD_INPUT_SHARE.of(need),
            );
        }
    }
}
