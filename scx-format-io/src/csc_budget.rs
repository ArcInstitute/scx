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
/// this figure, and the realized bound is it plus `n_buckets * block_bytes`.
pub const CSC_BUILD_BUCKET_SHARE: Share = Share::new(1, 2);

/// One source CSR shard decoded, and on the `build-csc` path re-encoded, in
/// flight at a time.
pub const CSC_BUILD_INPUT_SHARE: Share = Share::new(1, 4);

/// The emitted shard's arrays, its raw-value buffer and the encoder's streams.
///
/// The whole budget rather than a share: the emit runs after the last push, so
/// the buckets are gone by the time a shard is materialised.
pub const CSC_EMIT_SHARE: Share = Share::new(1, 1);

/// Bytes one decoded CSR shard of `nnz` nonzeros over `n_rows` rows occupies.
///
/// `indices` + `data` at [`CSC_PAYLOAD_BYTES_PER_NNZ`], plus an `i64` indptr
/// entry per row. It does **not** charge the re-encode that `run_build_csc`
/// performs beside it; see the allocation table's row, which says so and points
/// at the `SparseIngest` row for the terms.
pub fn decoded_csr_shard_bytes(nnz: u64, n_rows: u64) -> u64 {
    nnz.saturating_mul(CSC_PAYLOAD_BYTES_PER_NNZ)
        .saturating_add(n_rows.saturating_add(1).saturating_mul(8))
}

/// Bytes one emitted CSC shard of `nnz` nonzeros over `n_cols` columns
/// occupies before the encoder runs.
pub fn emitted_shard_bytes(nnz: u64, n_cols: u64) -> u64 {
    nnz.saturating_mul(CSC_PAYLOAD_BYTES_PER_NNZ)
        .saturating_add(n_cols.saturating_add(1).saturating_mul(8))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The push phase holds the buckets and one in-flight source shard at the
    /// same time, so their shares must fit together. (The emit phase is not
    /// concurrent with either — that is why it is a separate phase in the
    /// allocation table rather than a third row summed with these.)
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
