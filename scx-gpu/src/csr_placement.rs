//! The length invariants every GPU CSR rests on, in one place.
//!
//! Six decode loops build a combined device CSR the same way: allocate
//! `nnz`-sized `indices`/`data` buffers, then copy each row-group's (or, in
//! `gpu_csr_assemble`, each shard's) decoded buffers into
//! `combined[base .. base + len]`. Every invariant that placement rests on had
//! to be re-established — or forgotten — once per copy, and the one that binds
//! the finished matrix together was forgotten everywhere: nothing checked that
//! a `GpuCsr`'s two `nnz`-sized arrays were the same length.
//!
//! This module holds the arithmetic half of those invariants. It is
//! deliberately **CUDA-free**: it names no device type, calls nothing in
//! `cudarc`, and therefore runs in `cargo test` on any host. That matters here
//! more than usual — 20 of `shard_decode.rs`'s 22 tests and all 3 of
//! `gpu_csr_assemble.rs`'s are `#[ignore = "requires a CUDA GPU"]`, so before
//! this module none of this arithmetic had a reachable test on a CPU host.
//!
//! The placement half of the seam lands on top of these in the next commit of
//! this series; this one establishes the CSR-level invariant and the
//! constructor that enforces it.

use crate::error::GpuError;

/// Reject a CSR triple whose three buffers do not describe the same matrix.
///
/// Two invariants, both structural: the two `nnz`-sized arrays must agree with
/// each other, and `indptr` must carry one entry per row plus the terminator.
///
/// The first is not hypothetical. `GpuCsr`'s fields are public and it was built
/// by struct literal in a dozen places, none of which compared `indices.len()`
/// against `data.len()`; downstream, `pyscx`'s device-CSR handoff took
/// `nnz = indices.len()` and handed both pointers to a `cupyx` CSR, which then
/// read past the end of whichever buffer was shorter. Enforcing it at
/// construction (see [`GpuCsr::new`](crate::shard_decode::GpuCsr::new)) is what
/// makes `GpuCsr::nnz()` a fact rather than a guess.
pub(crate) fn check_csr_lengths(
    indptr_len: usize,
    indices_len: usize,
    data_len: usize,
    n_rows: usize,
    what: &str,
) -> Result<(), GpuError> {
    if indices_len != data_len {
        return Err(GpuError::InvalidShard(format!(
            "{what}: CSR indices has {indices_len} elements but data has {data_len} — \
             the two nnz-sized arrays must agree"
        )));
    }
    if indptr_len != n_rows + 1 {
        return Err(GpuError::InvalidShard(format!(
            "{what}: CSR indptr has {indptr_len} elements but {} were expected for {n_rows} rows",
            n_rows + 1
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "csr_placement_tests.rs"]
mod tests;
