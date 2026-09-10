//! Tests for the in-flight encode cost model.

use super::*;

/// The budget's whole job: the shards in one chunk must fit it. Swept over
/// budgets that bracket the per-shard cost so the clamp actually binds — at a
/// budget far above one shard's phase the `threads` cap is what closes a chunk
/// and every arm would pass whatever the arithmetic said.
#[test]
fn a_chunk_of_shards_fits_the_budget() {
    let per_shard_nnz = 1_000_000u64;
    let one_shard = encode_phase_bytes(per_shard_nnz); // 48 MB at the current model
    for &budget in &[
        one_shard / 2,     // below one shard: must still admit exactly one
        one_shard,         // exactly one
        one_shard * 2 + 1, // two
        one_shard * 100,   // more than `threads`, so `threads` binds
    ] {
        for &threads in &[1usize, 4, 16] {
            let nnz = vec![per_shard_nnz; 40];
            let chunks = plan_encode_chunks(&nnz, threads, budget);
            assert_eq!(
                chunks.iter().sum::<usize>(),
                nnz.len(),
                "chunks must cover every shard (budget {budget}, threads {threads})"
            );
            for (i, &len) in chunks.iter().enumerate() {
                assert!(len >= 1, "chunk {i} is empty");
                assert!(
                    len <= threads,
                    "chunk {i} holds {len} shards on {threads} threads"
                );
                let bytes = encode_phase_bytes(per_shard_nnz * len as u64);
                assert!(
                    len == 1 || bytes <= budget,
                    "chunk {i}: {len} shards x {one_shard} B = {bytes} B exceeds budget {budget}"
                );
            }
        }
    }
}

/// A single shard whose own phase exceeds the budget is admitted alone rather
/// than refused. One shard's encode is irreducible, and the ops this feeds have
/// no smaller unit to fall back to — unlike `scx-convert`'s ingest derate,
/// which hard-errors because it can tell the caller to lower
/// `shard_target_rows`.
#[test]
fn one_oversized_shard_is_admitted_alone_not_refused() {
    let huge = 1_000_000_000u64;
    let chunks = plan_encode_chunks(&[huge, huge, huge], 16, 1024);
    assert_eq!(chunks, vec![1, 1, 1]);
}

/// With bytes effectively unlimited, `threads` is the only thing closing a
/// chunk. Also pins that the default is a real number rather than "unbounded":
/// a default of `u64::MAX` would make an op's peak scale with the host's core
/// count, which is exactly what `DEFAULT_IN_FLIGHT_BYTES` exists to stop.
#[test]
fn chunks_are_thread_sized_when_bytes_are_not_the_limit() {
    assert_eq!(resolve_in_flight_budget(None), DEFAULT_IN_FLIGHT_BYTES);
    assert_eq!(resolve_in_flight_budget(Some(123)), 123);
    assert_eq!(resolve_in_flight_budget(Some(0)), 0);
    let nnz = vec![10u64; 10];
    assert_eq!(plan_encode_chunks(&nnz, 4, u64::MAX), vec![4, 4, 2]);
    assert_eq!(plan_encode_chunks(&nnz, 1, u64::MAX), vec![1; 10]);
    assert_eq!(plan_encode_chunks(&nnz, 32, u64::MAX), vec![10]);
}

/// A skewed distribution is the realistic one — a deletion filter or a
/// permutation concentrates nonzeros — and it is where a planner that assumed a
/// uniform shard size would overshoot. Chunks must be sized by the shards they
/// actually hold.
#[test]
fn a_skewed_distribution_is_chunked_by_its_own_shards() {
    let one = encode_phase_bytes(100);
    // Two small shards fit together; the fat one goes alone.
    let nnz = vec![100u64, 100, 10_000, 100, 100];
    let chunks = plan_encode_chunks(&nnz, 16, one * 2);
    assert_eq!(chunks.iter().sum::<usize>(), nnz.len());
    let mut at = 0usize;
    for &len in &chunks {
        let sum: u64 = nnz[at..at + len].iter().sum();
        assert!(
            len == 1 || encode_phase_bytes(sum) <= one * 2,
            "chunk of {len} starting at {at} exceeds the budget"
        );
        at += len;
    }
    assert!(
        chunks.contains(&1),
        "the 10_000-nnz shard must be chunked alone: {chunks:?}"
    );
}

/// Degenerate inputs a real catalog can produce: no shards at all, and shards
/// with no nonzeros (a zero-row or all-deleted shard).
#[test]
fn empty_and_zero_nnz_inputs_are_handled() {
    assert!(plan_encode_chunks(&[], 8, 1 << 20).is_empty());
    // Zero-cost shards must not stall the loop on a zero-length chunk, and they
    // fit *any* budget including zero: adding one adds nothing, so `threads` is
    // what closes the chunk. A budget of zero is therefore not "no
    // concurrency" — it is "nothing that costs anything", which for shards with
    // real nonzeros does come out as one per chunk (asserted above).
    assert_eq!(plan_encode_chunks(&[0, 0, 0], 2, 0), vec![2, 1]);
    assert_eq!(plan_encode_chunks(&[0, 0, 0], 2, u64::MAX), vec![2, 1]);
    assert_eq!(
        plan_encode_chunks(&[1_000_000, 1_000_000], 8, 0),
        vec![1, 1]
    );
}
