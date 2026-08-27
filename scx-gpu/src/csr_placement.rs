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
//! Three checks, each with a distinct job. [`check_csr_lengths`] binds a
//! finished CSR's three buffers together — the invariant `GpuCsr::new`
//! enforces. [`check_placement`] gates one unit's copy into the combined
//! buffers. [`check_coverage`] establishes at the end that the units placed
//! actually covered the matrix, rather than leaving a gap that reads back as
//! `alloc_zeros`'s structural zeros.
//!
//! The device layer over them is [`CombinedCsr`](crate::combined_csr::CombinedCsr).

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
    // `n_rows + 1` rather than a bare `+`: `n_rows` is unauthenticated in every
    // caller, and in release Rust `usize::MAX + 1` wraps to 0 — which would make
    // an *empty* indptr the expected length for a `usize::MAX`-row matrix and
    // let the check pass on a shape that cannot exist. Debug builds would panic
    // instead, so the failure mode differs by profile, which is worse.
    let expected = n_rows.checked_add(1).ok_or_else(|| {
        GpuError::InvalidShard(format!(
            "{what}: CSR row count {n_rows} + 1 overflows usize"
        ))
    })?;
    if indptr_len != expected {
        return Err(GpuError::InvalidShard(format!(
            "{what}: CSR indptr has {indptr_len} elements but {expected} were expected for \
             {n_rows} rows"
        )));
    }
    Ok(())
}

/// Where one decoded unit goes, and what to call it if it does not fit.
///
/// `base` and `len` are two adjacent `usize`s that a positional call could
/// transpose silently — `place(dev, len, base, ..)` type-checks and writes the
/// right number of elements to the wrong place. Naming them is the same fix
/// `RowSegment` applied to the PCA segment functions, for the same reason.
///
/// `op` names the kind of unit and `index` which one, so an error is prefixed
/// `framed Scx1 group 3:` — [`Placement::label`] is `"{op} {index}"`, and which
/// half failed is named by the rest of `check_placement`'s message rather than
/// by the label. Both are cheap to carry and are only turned into a `String`
/// when something is actually wrong.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Placement<'a> {
    /// Offset into the combined buffers, in elements.
    pub(crate) base: usize,
    /// Number of elements this unit contributes, as declared by the block index
    /// or catalog — not as decoded. `check_placement` compares the two.
    pub(crate) len: usize,
    /// The kind of unit, e.g. `"framed Scx1 group"` or `"shard"`.
    pub(crate) op: &'a str,
    /// Which unit, in whatever order the caller walks them.
    pub(crate) index: usize,
}

impl Placement<'_> {
    /// The message prefix, built only when something is wrong.
    fn label(&self) -> String {
        format!("{} {}", self.op, self.index)
    }
}

/// Reject a unit whose decoded buffers cannot be placed at `at.base`.
///
/// `at.len` is what the block index (or the catalog) says this unit holds;
/// `indices_len` / `data_len` are what its decode actually produced;
/// `combined_nnz` is the length of the combined buffers being filled.
///
/// The overflow arm is not decoration: `base` and `len` both descend from
/// unauthenticated shard-header and block-index fields, and `base + len` in
/// release Rust wraps rather than panicking, which would turn an out-of-range
/// placement into an in-range one.
pub(crate) fn check_placement(
    at: Placement<'_>,
    indices_len: usize,
    data_len: usize,
    combined_nnz: usize,
) -> Result<(), GpuError> {
    let Placement { base, len, .. } = at;
    if indices_len != len {
        return Err(GpuError::InvalidShard(format!(
            "{}: device decode produced {indices_len} indices but {len} were declared",
            at.label()
        )));
    }
    if data_len != len {
        return Err(GpuError::InvalidShard(format!(
            "{}: device decode produced {data_len} values but {len} were declared",
            at.label()
        )));
    }
    let end = base.checked_add(len).ok_or_else(|| {
        GpuError::InvalidShard(format!(
            "{}: placement offset {base} + {len} elements overflows usize",
            at.label()
        ))
    })?;
    if end > combined_nnz {
        return Err(GpuError::InvalidShard(format!(
            "{}: placement at offset {base} spans {len} elements, past the end of the \
             {combined_nnz}-element combined buffer",
            at.label()
        )));
    }
    Ok(())
}

/// Reject an assembly whose placed units do not exactly tile the matrix.
///
/// A decode that skipped a unit returns a full-shaped CSR whose gap holds the
/// zeros `alloc_zeros` left there — the right shape, the wrong matrix, and
/// nothing downstream able to tell the difference. This is the same failure the
/// streaming PCA drive's row-coverage check closed, one layer down.
///
/// **This checks a tiling, not a sum.** The first version added each unit's
/// length and demanded the total equal `nnz`, which accepts an assembly that
/// overlaps in one place and leaves a hole in another as long as the lengths
/// happen to add up — `[0, 6)` then `[4, 8)` against `nnz = 10` sums to 10 with
/// `[8, 10)` never written. Both spans are individually in range, so
/// [`check_placement`] cannot see it either. Sorting by base and walking closes
/// exactly that.
///
/// The two halves are tallied separately because the cross-shard nvcomp path
/// fills `indices` and `data` in different passes over different chunks: a
/// dropped values chunk leaves indices fully tiled, so one shared span list
/// would accept it.
///
/// The spans are sorted in place rather than copied — the pipelined path places
/// groups **out of order** by design (each worker writes its own disjoint
/// region), so this cannot assume arrival order and must not care about it.
///
/// The running cursor uses `checked_add` for the same reason
/// [`check_csr_lengths`] does: this is a standalone validator over untrusted
/// `(base, len)` pairs, and it must not depend on its caller having run
/// [`check_placement`] first. `[(0, usize::MAX), (usize::MAX, 1)]` would
/// otherwise panic in debug and wrap to zero — and therefore *accept* — in
/// release, which is exactly the profile-dependent failure this series removed
/// from the row-count arithmetic.
pub(crate) fn check_coverage(
    indices_spans: &mut [(usize, usize)],
    values_spans: &mut [(usize, usize)],
    nnz: usize,
    what: &str,
) -> Result<(), GpuError> {
    for (spans, half) in [(indices_spans, "indices"), (values_spans, "values")] {
        spans.sort_unstable();
        let mut cursor = 0usize;
        for &(base, len) in spans.iter() {
            if base != cursor {
                let how = if base > cursor {
                    "leaving a gap"
                } else {
                    "overlapping"
                };
                return Err(GpuError::InvalidShard(format!(
                    "{what}: {half} placement at {base} does not continue from {cursor} \
                     ({how}) — the placed units do not tile the matrix, so the gap would \
                     read back as zeros"
                )));
            }
            cursor = cursor.checked_add(len).ok_or_else(|| {
                GpuError::InvalidShard(format!(
                    "{what}: {half} placement at {base} spans {len} elements, overflowing usize"
                ))
            })?;
        }
        if cursor != nnz {
            return Err(GpuError::InvalidShard(format!(
                "{what}: placed {cursor} of {nnz} {half} — the decoded units do not cover \
                 the matrix, so the gap would read back as zeros"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "csr_placement_tests.rs"]
mod tests;
