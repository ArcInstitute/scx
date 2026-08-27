//! CPU tests for the CSR length and placement invariants.
//!
//! Every one of these runs on a host with no CUDA. That is the point of
//! [`super`] being CUDA-free: the six decode loops these checks guard are
//! reachable only from `#[ignore = "requires a CUDA GPU"]` tests, so before
//! this file the arithmetic could not be watched fail.
//!
//! Each rejection test below was written against a copy of the function with
//! *its* clause deleted and confirmed to fail there first; a check test that
//! passes against a check that isn't there is worse than no test.

use super::*;

/// The error text, for tests that assert on wording rather than on the variant.
fn msg(e: GpuError) -> String {
    match e {
        GpuError::InvalidShard(s) => s,
        other => panic!("expected InvalidShard, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// check_placement
// ---------------------------------------------------------------------------

/// A placement descriptor, so the tests read as `at(base, len)` rather than as
/// four positional arguments two of which are interchangeable `usize`s.
fn at(base: usize, len: usize) -> Placement<'static> {
    Placement {
        base,
        len,
        op: "unit",
        index: 0,
    }
}

/// The shape every one of the six loops produces: consecutive units whose bases
/// are the running nnz prefix sum and whose last one ends exactly at the
/// combined length. Accept-side coverage — a check that rejects nothing is
/// caught by the rejection tests, a check that rejects *everything* is caught
/// only here.
#[test]
fn an_exact_tiling_of_the_combined_buffer_is_accepted() {
    let group_nnz = [7usize, 0, 13, 1, 40];
    let total: usize = group_nnz.iter().sum();
    let mut base = 0usize;
    for (gi, &len) in group_nnz.iter().enumerate() {
        let p = Placement {
            base,
            len,
            op: "group",
            index: gi,
        };
        check_placement(p, len, len, total)
            .unwrap_or_else(|e| panic!("group {gi} rejected: {}", msg(e)));
        base += len;
    }
    assert_eq!(base, total, "the fixture must tile the buffer exactly");
}

/// A unit ending exactly on the last element is in range. The off-by-one that
/// makes this fail is the same one that makes a real final group unplaceable,
/// so it is asserted rather than left to the tiling test above.
#[test]
fn a_group_ending_exactly_at_the_end_is_accepted() {
    check_placement(at(90, 10), 10, 10, 100).unwrap();
}

/// The bounds arm — the only invariant of the four that is a *panic* today
/// rather than an error. `CudaSlice::slice_mut` is `try_slice_mut(..).unwrap()`.
#[test]
fn a_placement_past_the_end_of_the_combined_buffer_is_rejected() {
    let p = Placement {
        base: 95,
        len: 10,
        op: "overrunning group",
        index: 4,
    };
    let e = msg(check_placement(p, 10, 10, 100).unwrap_err());
    assert!(e.contains("overrunning group 4"), "{e}");
    assert!(
        e.contains("100"),
        "the message must name the buffer it overran: {e}"
    );
}

/// One past the end, which is the weakest case the bounds arm has to reject —
/// a `>=` written where `>` belongs passes the test above and fails here.
#[test]
fn a_placement_one_element_past_the_end_is_rejected() {
    assert!(check_placement(at(91, 10), 10, 10, 100).is_err());
    // ... and the neighbouring accept still holds, so the arm is not simply
    // rejecting everything near the boundary.
    check_placement(at(90, 10), 10, 10, 100).unwrap();
}

/// `base + len` wraps in release Rust, which would turn an out-of-range
/// placement into an in-range one. Both operands descend from unauthenticated
/// header fields.
#[test]
fn a_placement_offset_that_overflows_usize_is_rejected() {
    let e = msg(check_placement(at(usize::MAX - 4, 10), 10, 10, 100).unwrap_err());
    assert!(e.contains("overflows usize"), "{e}");
}

/// A decode that produced fewer indices than the block index declared. This is
/// the arm with real content on the framed Scx1 path, where `forbp_decode_gpu`
/// sizes its output from the bitstream's own per-row nnz varints.
#[test]
fn an_indices_decode_shorter_than_declared_is_rejected() {
    let p = Placement {
        base: 0,
        len: 10,
        op: "framed Scx1 group",
        index: 3,
    };
    let e = msg(check_placement(p, 9, 10, 100).unwrap_err());
    assert!(e.contains("framed Scx1 group 3"), "{e}");
    assert!(e.contains("9") && e.contains("10"), "{e}");
}

/// The other direction. A `<` written where `!=` belongs passes the test above.
#[test]
fn an_indices_decode_longer_than_declared_is_rejected() {
    assert!(check_placement(at(0, 10), 11, 10, 100).is_err());
}

/// The values half, both directions. `indices` and `data` are checked
/// separately so the message says which buffer disagreed.
#[test]
fn a_values_decode_of_the_wrong_length_is_rejected_in_both_directions() {
    let short = msg(check_placement(at(0, 10), 10, 9, 100).unwrap_err());
    assert!(
        short.contains("values"),
        "the message must name the buffer: {short}"
    );
    assert!(check_placement(at(0, 10), 10, 11, 100).is_err());
}

/// An empty unit is placeable anywhere inside the buffer — including at its
/// very end, which is where the offset of a trailing zero-nnz row-group lands.
/// The framed loops skip `len == 0` before calling `place`; `gpu_csr_assemble`
/// deliberately does not, so an empty shard's declared length still goes
/// through `check_placement`. Both must be fine, which is what this pins.
#[test]
fn an_empty_group_is_accepted_up_to_and_including_the_end() {
    check_placement(at(0, 0), 0, 0, 100).unwrap();
    check_placement(at(100, 0), 0, 0, 100).unwrap();
    check_placement(at(0, 0), 0, 0, 0).unwrap();
    // Past the end is still past the end.
    assert!(check_placement(at(101, 0), 0, 0, 100).is_err());
}

// ---------------------------------------------------------------------------
// check_csr_lengths
// ---------------------------------------------------------------------------

#[test]
fn a_well_formed_csr_triple_is_accepted() {
    check_csr_lengths(5, 17, 17, 4, "shard 0").unwrap();
}

/// The invariant `pyscx`'s device-CSR handoff assumed and nothing established:
/// it took `nnz = indices.len()` and handed both pointers to a `cupyx` CSR.
#[test]
fn a_csr_whose_two_nnz_arrays_disagree_is_rejected() {
    let e = msg(check_csr_lengths(5, 17, 16, 4, "handoff").unwrap_err());
    assert!(e.contains("handoff"), "{e}");
    assert!(e.contains("17") && e.contains("16"), "{e}");
    // Both directions: a `<` where `!=` belongs passes the case above.
    assert!(check_csr_lengths(5, 16, 17, 4, "handoff").is_err());
}

#[test]
fn an_indptr_of_the_wrong_length_is_rejected_in_both_directions() {
    let short = msg(check_csr_lengths(4, 17, 17, 4, "short indptr").unwrap_err());
    assert!(
        short.contains("5"),
        "the message must name the expected length: {short}"
    );
    assert!(check_csr_lengths(6, 17, 17, 4, "long indptr").is_err());
}

/// A zero-row matrix still carries the leading `0`. `n_rows + 1` computed as
/// `n_rows` would accept an empty indptr here and be caught nowhere else.
#[test]
fn an_empty_matrix_needs_a_one_element_indptr() {
    check_csr_lengths(1, 0, 0, 0, "empty").unwrap();
    assert!(check_csr_lengths(0, 0, 0, 0, "empty with no indptr").is_err());
}

/// `n_rows + 1` wraps in release Rust, so an unauthenticated `usize::MAX` row
/// count would make the *expected* indptr length zero and accept an empty
/// indptr for a matrix that cannot exist. Debug builds would panic instead —
/// a failure mode that differs by profile is worse than either one alone.
///
/// No current caller can reach it (`n_major` is a `u32`, and a host `shape[0]`
/// that large could never have allocated the buffers being checked), so this
/// pins the arm rather than reproducing a live bug. Found by **codex**.
#[test]
fn a_row_count_whose_successor_overflows_is_rejected() {
    let e = msg(check_csr_lengths(0, 0, 0, usize::MAX, "hostile shape").unwrap_err());
    assert!(e.contains("overflows usize"), "{e}");
    assert!(e.contains("hostile shape"), "{e}");
    // The neighbouring value still behaves normally rather than being swept up.
    assert!(check_csr_lengths(0, 0, 0, usize::MAX - 1, "large but representable").is_err());
}

// ---------------------------------------------------------------------------
// check_coverage
// ---------------------------------------------------------------------------

/// `(base, len)` spans, so the tests read as a picture of the buffer.
fn spans(v: &[(usize, usize)]) -> Vec<(usize, usize)> {
    v.to_vec()
}

#[test]
fn an_assembly_that_tiles_the_matrix_is_accepted() {
    let mut i = spans(&[(0, 10), (10, 20), (30, 10)]);
    let mut v = spans(&[(0, 10), (10, 20), (30, 10)]);
    check_coverage(&mut i, &mut v, 40, "framed Scx1 shard").unwrap();

    let mut e0 = spans(&[]);
    let mut e1 = spans(&[]);
    check_coverage(&mut e0, &mut e1, 0, "empty shard").unwrap();
}

/// The pipelined path places groups **out of order** by design — each worker
/// writes its own disjoint region — so arrival order must not matter.
#[test]
fn an_out_of_order_tiling_is_accepted() {
    let mut i = spans(&[(30, 10), (0, 10), (10, 20)]);
    let mut v = spans(&[(10, 20), (30, 10), (0, 10)]);
    check_coverage(&mut i, &mut v, 40, "pipelined shard").unwrap();
}

/// **The case a running sum accepts.** Lengths add to exactly `nnz`, every span
/// is individually in range so `check_placement` passes each one, and yet
/// `[8, 10)` is never written and `[4, 6)` is written twice. Found by
/// **Cursor Agent**.
#[test]
fn an_overlap_that_leaves_a_hole_is_rejected_even_though_the_lengths_sum() {
    let mut i = spans(&[(0, 6), (4, 4)]);
    let mut v = spans(&[(0, 6), (4, 4)]);
    assert_eq!(
        i.iter().map(|s| s.1).sum::<usize>(),
        10,
        "fixture premise: the lengths must sum to nnz, or this tests nothing"
    );
    let e = msg(check_coverage(&mut i, &mut v, 10, "overlapping shard").unwrap_err());
    assert!(e.contains("overlapping"), "{e}");
    assert!(
        e.contains("indices"),
        "the message must name which half: {e}"
    );
}

/// A decode that skipped a unit. The buffers are `alloc_zeros`, so the gap
/// reads back as structural zeros in a result of exactly the right shape.
#[test]
fn an_assembly_missing_a_unit_is_rejected() {
    let mut i = spans(&[(0, 10), (10, 23)]);
    let mut v = spans(&[(0, 10), (10, 23)]);
    let e = msg(check_coverage(&mut i, &mut v, 40, "framed Scx1 shard").unwrap_err());
    assert!(e.contains("framed Scx1 shard"), "{e}");
    assert!(
        e.contains("indices"),
        "the message must name which half: {e}"
    );
    assert!(e.contains("33") && e.contains("40"), "{e}");
}

/// A gap in the middle, which the trailing-total check alone cannot see: the
/// spans here sum to less than `nnz` *and* skip `[10, 15)`, and the walk names
/// the gap rather than only the shortfall.
#[test]
fn a_gap_in_the_middle_is_rejected_as_a_gap() {
    let mut i = spans(&[(0, 10), (15, 25)]);
    let mut v = spans(&[(0, 10), (15, 25)]);
    let e = msg(check_coverage(&mut i, &mut v, 40, "gapped shard").unwrap_err());
    assert!(e.contains("leaving a gap"), "{e}");
}

/// The values half alone, which is the case one shared span list would miss:
/// the cross-shard nvcomp path fills indices and values in separate passes over
/// separate chunks, so a dropped values chunk leaves indices fully tiled.
#[test]
fn an_assembly_whose_values_pass_dropped_a_chunk_is_rejected() {
    let mut i = spans(&[(0, 40)]);
    let mut v = spans(&[(0, 31)]);
    let e = msg(check_coverage(&mut i, &mut v, 40, "nvcomp cross-shard batch").unwrap_err());
    assert!(
        e.contains("values"),
        "the message must name the values half: {e}"
    );
    assert!(e.contains("31"), "{e}");
}

/// Placing a unit twice at the same base. A sum would call this over-placement;
/// the walk calls it what it is.
#[test]
fn a_doubly_placed_unit_is_rejected() {
    let mut i = spans(&[(0, 20), (0, 20)]);
    let mut v = spans(&[(0, 20), (20, 20)]);
    assert!(check_coverage(&mut i, &mut v, 40, "double-placed indices").is_err());
}

/// Zero-length units are placeable and must not disturb the tiling. The framed
/// loops skip `len == 0` before calling `place`; `gpu_csr_assemble` calls it
/// unconditionally so an empty shard's declared length is still checked. Both
/// are correct, which is exactly what makes the skipping an optimisation rather
/// than a requirement.
#[test]
fn zero_length_units_do_not_disturb_the_tiling() {
    let mut i = spans(&[(0, 10), (10, 0), (10, 30)]);
    let mut v = spans(&[(0, 10), (10, 0), (10, 30)]);
    check_coverage(&mut i, &mut v, 40, "shard with an empty unit").unwrap();
}

/// Spans that tile contiguously from zero but run *past* `nnz`. The walk itself
/// cannot catch this — every step continues from the cursor — so it is the
/// trailing `cursor != nnz` that has to, and a `<` written there accepts it.
///
/// `check_placement` bounds each span to the buffer, so no current caller can
/// produce this; the arm is pinned because `check_coverage` is a standalone
/// function that must not depend on its caller having run the other check.
#[test]
fn a_tiling_that_overshoots_the_matrix_is_rejected() {
    let mut i = spans(&[(0, 10), (10, 10)]);
    let mut v = spans(&[(0, 10), (10, 10)]);
    let e = msg(check_coverage(&mut i, &mut v, 15, "overshooting shard").unwrap_err());
    assert!(e.contains("20") && e.contains("15"), "{e}");
}

/// The running cursor must not wrap. `check_coverage` is a standalone validator
/// over untrusted `(base, len)` pairs — the overshoot test above says so
/// explicitly — so it cannot lean on `check_placement` having bounded them
/// first. Debug would panic and release would wrap to zero and *accept*, which
/// is the profile-dependent split this series removed from `n_rows + 1`.
/// Found by **codex**.
#[test]
fn a_span_walk_that_overflows_the_cursor_is_rejected() {
    let mut i = spans(&[(0, usize::MAX), (usize::MAX, 1)]);
    let mut v = spans(&[(0, usize::MAX), (usize::MAX, 1)]);
    let e = msg(check_coverage(&mut i, &mut v, 0, "hostile spans").unwrap_err());
    assert!(e.contains("overflowing usize"), "{e}");
}
