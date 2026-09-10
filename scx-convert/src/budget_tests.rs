//! Tests for [`super`] — the memory-budget allocation table.
//!
//! Ungated, like the module they cover, so they run in the ordinary
//! `cargo test --workspace` job rather than only in the hdf5 lane.

use super::*;

/// The invariant the table exists to make checkable: within a phase, the
/// reservations held at once cannot claim more than the whole budget.
///
/// Per **phase**, not globally. The CSC external transpose claims half the
/// budget for a column chunk and a quarter for bucket records, but those are
/// pass 1 and pass 2 — never live together — so a single global sum would be
/// asserting something the design never promised, and would have to be
/// weakened until it could not fail.
///
/// ⚠️ **Watched red by** setting `SHARD_BUDGET_SHARE` to `Share::new(1, 2)` —
/// the share the pre-fix code effectively took, since it divided the budget by
/// 4 to size a slab and then charged 2x that slab against the full budget. With
/// multiplicity 4 that reads `1/2 x 4 = 2 > 1`, and this test is what says so.
#[test]
fn allocation_table_shares_sum_to_at_most_one_per_phase() {
    // Exhaustive by construction. The first version of this test hard-coded
    // five variants; `Phase::CscSidecar` was added in the same commit and left
    // off the list, so the phase was declared and never checked — the invariant
    // silently stopped covering it. The `match` below turns that into a compile
    // error: a new variant does not build until it is listed here.
    fn all_phases() -> Vec<Phase> {
        let every = |p: Phase| -> Phase {
            match p {
                Phase::DenseIngest
                | Phase::SparseIngest
                | Phase::Export
                | Phase::CscExternalColumnScan
                | Phase::CscExternalBucketDrain
                | Phase::CscSidecar => p,
            }
        };
        [
            Phase::DenseIngest,
            Phase::SparseIngest,
            Phase::Export,
            Phase::CscExternalColumnScan,
            Phase::CscExternalBucketDrain,
            Phase::CscSidecar,
        ]
        .into_iter()
        .map(every)
        .collect()
    }
    let phases = all_phases();
    for phase in phases {
        // Exact rational sum over a common denominator — no floats, so a
        // design where three phases each take a third sums to exactly one.
        let rows: Vec<&Reservation> = ALLOCATION_TABLE
            .iter()
            .filter(|r| r.phase == phase)
            .collect();
        assert!(
            !rows.is_empty(),
            "{phase:?} has no declared reservation; either it claims nothing \
             (then drop the variant) or the table is incomplete"
        );
        let den: u64 = rows.iter().map(|r| r.share.denominator()).product();
        let claimed: u64 = rows
            .iter()
            .map(|r| den / r.share.denominator() * r.share.numerator() * r.multiplicity)
            .sum();
        assert!(
            claimed <= den,
            "{phase:?} claims {claimed}/{den} of the memory budget, which is \
             more than there is. Reservations held in one phase are \
             concurrent by definition, so this is an over-subscription, not a \
             rounding artefact: {:?}",
            rows.iter()
                .map(|r| format!("{} ({}) at {}", r.name, r.share_str(), r.site))
                .collect::<Vec<_>>()
        );
    }
}

/// A reservation the code does not enforce is allowed, but it must say so —
/// otherwise the table reads as coverage it does not have.
#[test]
fn unenforced_reservations_are_declared_not_silent() {
    let unenforced: Vec<&str> = ALLOCATION_TABLE
        .iter()
        .filter(|r| !r.enforced)
        .map(|r| r.name)
        .collect();
    // Seven, and the count is pinned so an eighth cannot arrive unannounced:
    //
    //   * the two §11.4 CSC bucket rows — bucket count and bucket record
    //     buffer are sized from the *mean* nnz/row, so a right-skewed depth
    //     distribution overshoots them;
    //   * the CSC sidecar row — the budget sizes the transpose chunk, while the
    //     writer's full-length index/value copies and the encoder's streams are
    //     live alongside it (and `build_csc.rs` additionally retains every
    //     source shard). It controls shard sizing, not a ceiling;
    //   * the CSC column-chunk row — the scan always reads the first column
    //     whole before testing the budget, so one wide column exceeds the
    //     share, and the floor exceeds it for budgets under 128 bytes.
    //
    //   * the two **ingest** rows (sparse and dense) — now sized from the
    //     whole worker phase rather than the reader stage, 48 B/nnz against
    //     the 16 they used to claim, but `encoded <= payload` is an estimate
    //     and not a codec guarantee. The sparse row enumerates exactly what
    //     it does not bound: frame expansion, intra-codec planes, the encoded
    //     indptr, the bitmap, and readers on the density guess.
    //
    //   * the **export** row — no encode term (that path decodes) and sized
    //     from the catalog's exact nnz, but `filter_shard` holds two indptrs
    //     on its no-filter path and doubling-grown output buffers alongside
    //     the originals on its masked one.
    //
    // **Still seven.** An intermediate revision of this PR flipped all three
    // per-shard rows to `enforced: true` and then, on review, flipped two back
    // and finally the third. What this PR changes is the *estimate* — 3x
    // better on sparse ingest, 3.7x on dense, and no longer blind to the
    // encoder — not any row's flag. The share did not move either, which is
    // why `allocation_table_shares_sum_to_at_most_one_per_phase` is
    // unaffected. Widening a cost model is not the same as proving a ceiling,
    // and this count is what keeps the two from being confused.
    assert_eq!(
        unenforced.len(),
        7,
        "expected all four CSC rows plus the three per-shard rows to be \
         unenforced, got {unenforced:?}. Adding an unenforced row without \
         updating this count lets a known gap enter the table unannounced; \
         removing one means a gap actually closed and this test should say so."
    );
}

#[test]
fn share_of_and_min_budget_are_inverses() {
    for (num, den) in [(1u64, 4u64), (1, 2), (1, 8), (3, 4), (1, 1)] {
        let s = Share::new(num, den);
        for unit in [1u64, 7, 1000, 4096, 1 << 20] {
            let min = s.min_budget_for(unit);
            assert!(
                s.of(min) >= unit,
                "Share({num}/{den}).min_budget_for({unit}) = {min}, but that \
                 budget's share is {} — the refusal predicate and the \
                 advertised minimum would disagree, which is the drift that \
                 put a `budget / row_bytes == 0` guard behind a message \
                 promising `4 x row_bytes`",
                s.of(min)
            );
        }
    }
}

#[test]
fn shard_share_permits_more_than_one_outstanding_shard() {
    // The whole of §11.5 in one line: at max_concurrent() == 1 the derate can
    // only ever grant a single outstanding shard, and one outstanding shard
    // *is* the sequential coordinator.
    assert!(
        SHARD_BUDGET_SHARE.max_concurrent() >= 2,
        "SHARD_BUDGET_SHARE lets only {} shard(s) be outstanding, so every \
         budgeted convert routes to the sequential coordinator (§11.5)",
        SHARD_BUDGET_SHARE.max_concurrent()
    );
}

/// The dense peak does not depend on the source dtype width, for every dtype
/// the reader supports. This is the fact the pre-fix `/4` got wrong by keying
/// its reserve to `sizeof(source_dtype)`.
#[test]
fn dense_peak_is_independent_of_source_width() {
    for dtype_bytes in [1u64, 2, 4, 8] {
        assert_eq!(
            dense_peak_bytes_per_elem(dtype_bytes),
            DENSE_SPARSIFY_BYTES_PER_ELEM,
            "dtype width {dtype_bytes} changed the dense peak; the resident \
             slab is f32 whatever the source was, and the sparsified output \
             does not depend on the source width at all"
        );
    }
    // A hypothetical wider source would be read-bound, and the `max` says so.
    assert_eq!(dense_peak_bytes_per_elem(16), 20);
}

/// One dense slab, sized by the table, stays inside the share it was given —
/// for every dtype, at every budget. This is the property both §11.5 and the
/// narrow-dtype overshoot violate, expressed without an HDF5 fixture.
#[test]
fn dense_slab_never_exceeds_its_share() {
    for dtype_bytes in [1u64, 2, 4, 8] {
        for n_vars in [1u64, 17, 1000, 60_000] {
            for budget in [1u64 << 16, 1 << 20, 1 << 24, 1 << 30] {
                let Ok(rows) = dense_max_slab_rows(budget, n_vars, dtype_bytes) else {
                    // Refused: then the budget must genuinely be below the
                    // advertised minimum, or the refusal is spurious.
                    assert!(
                        budget < dense_min_budget(n_vars, dtype_bytes),
                        "refused a budget of {budget} for n_vars={n_vars} \
                         dtype_bytes={dtype_bytes}, but the advertised minimum \
                         is {}",
                        dense_min_budget(n_vars, dtype_bytes)
                    );
                    continue;
                };
                let held = dense_slab_bytes(rows, n_vars, dtype_bytes);
                assert!(
                    held <= SHARD_BUDGET_SHARE.of(budget),
                    "one dense slab of {rows} rows x {n_vars} vars \
                     (dtype_bytes={dtype_bytes}) holds {held} B, over its \
                     {} B share of a {budget} B budget",
                    SHARD_BUDGET_SHARE.of(budget)
                );
                // And the cap is not needlessly tight: one more row would not fit.
                assert!(
                    dense_slab_bytes(rows + 1, n_vars, dtype_bytes) > SHARD_BUDGET_SHARE.of(budget),
                    "the slab cap left a whole unused row — a cap that is \
                     merely safe rather than tight would let this test pass \
                     while throughput quietly halved"
                );
            }
        }
    }
}

#[test]
fn shard_working_set_counts_the_whole_worker_phase_and_the_indptr() {
    // 50 nnz, 100 rows. Spelled as the derivation rather than as a total, so
    // a change to either multiple has to be made here deliberately instead of
    // being pasted out of a failure message:
    //
    //   payload           50 x 8            = 400
    //   encoder value copy 50 x 8 x 1       = 400
    //   framed encode     50 x 8 x 4        = 1600
    //   indptr            101 x 8           = 808
    //                                        ----
    //                                        3208
    let payload = 50 * PAYLOAD_BYTES_PER_NNZ;
    assert_eq!(
        shard_working_set_bytes(50, 100),
        payload
            + payload * ENCODER_VALUE_COPY_MULTIPLE
            + payload * ENCODE_TRANSIENT_MULTIPLE
            + 101 * INDPTR_BYTES_PER_ROW
    );
    assert_eq!(shard_working_set_bytes(50, 100), 3208);
    // Never zero: a zero would make the derate treat the budget as unset.
    assert_eq!(shard_working_set_bytes(0, 0), 8);
}

/// The encode transient is actually charged, and charged *concurrently* — the
/// whole-phase figure is strictly more than the payload plus one copy of it.
///
/// Watched red by setting `ENCODE_TRANSIENT_MULTIPLE` to 0: the first
/// assertion then reports 16 B/nnz, which is the pre-PR-42 model and the state
/// this item exists to leave.
#[test]
fn the_worker_phase_charges_the_framed_encode() {
    assert_eq!(WORKER_PHASE_BYTES_PER_NNZ, 48);
    // Export is on a **decode** path and must not carry the encode term:
    // `stream_write.rs` has no encode call at all. 50 x 8 x 2 + 101 x 8.
    assert_eq!(
        shard_decode_working_set_bytes(50, 100),
        50 * PAYLOAD_BYTES_PER_NNZ * (1 + DECODE_SCRATCH_MULTIPLE) + 101 * INDPTR_BYTES_PER_ROW
    );
    assert_eq!(shard_decode_working_set_bytes(50, 100), 1608);
    assert!(
        shard_decode_working_set_bytes(50, 100) < shard_working_set_bytes(50, 100),
        "an export shard must be cheaper than an ingest shard, or the export \
         derate is paying for an encoder it never runs"
    );
    // And the density-estimate path asks the table for the same figure.
    assert_eq!(
        estimated_worker_bytes(100, 1_000, PARALLEL_DENSITY_DEFAULT_DEN),
        100 * 1_000 * WORKER_PHASE_BYTES_PER_NNZ / PARALLEL_DENSITY_DEFAULT_DEN
    );
}
