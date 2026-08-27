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
