//! The shared bounds-guard family.
//!
//! Split out of `dispatch.rs` (ORG-3.7-2). Pure move.
//!
//! These are the checks every decode path is supposed to apply. Keeping them
//! in one module is the point: the review's §3.5 finding is that they drifted
//! because each decoder re-derived its own.

use crate::codec_id::CodecError;
use std::ops::Range;

/// Bounds-checked slice of a sub-stream by a resolved byte range.
pub(crate) fn slice_span<'a>(
    bytes: &'a [u8],
    range: &Range<usize>,
    which: &str,
) -> Result<&'a [u8], CodecError> {
    bytes.get(range.clone()).ok_or_else(|| {
        CodecError::MalformedInput(format!(
            "row-group {which} range {}..{} out of bounds (stream len {})",
            range.start,
            range.end,
            bytes.len()
        ))
    })
}

/// Convert Vec<u64> to Vec<i64> via zero-copy reinterpretation.
/// CSR indptr values are always non-negative and well below i64::MAX,
/// so the bit patterns are identical. Uses bytemuck for safe transmute.
pub(crate) fn u64_vec_to_i64(data: Vec<u64>) -> Result<Vec<i64>, CodecError> {
    if let Some(&bad) = data.iter().find(|&&v| v > i64::MAX as u64) {
        return Err(CodecError::MalformedInput(format!(
            "indptr value {bad} exceeds i64::MAX (corrupt or hostile input)"
        )));
    }
    Ok(bytemuck::cast_vec::<u64, i64>(data))
}

/// Convert Vec<u32> to Vec<i32> via zero-copy reinterpretation, rejecting any
/// index at or above `bound`.
///
/// # Why the bound rides along here
///
/// This scan already existed, to keep a `> i32::MAX` value from reinterpreting
/// to a *negative* `i32`. In the good case it walks every element and returns
/// `None`, so the caller's minor-axis bound check costs nothing extra when it
/// rides on the same pass — only the comparand changes. Measured as a standalone
/// pass instead, the same check cost **+5.1–6.4%** of per-shard decode
/// (194.9M nnz, memory-bandwidth-bound at 10.0 GB/s, so not optimisable in
/// place). Folding it in is what makes it free on the hot path.
///
/// `bound` is the shard's `n_minor`, clamped by the caller to at most
/// `i32::MAX as u32 + 1` so the original sign guarantee still holds. A caller
/// with no bound to enforce (a decode that is not addressing a column axis)
/// passes exactly that clamp value and gets the pre-existing behaviour.
pub(crate) fn u32_vec_to_i32_bounded(data: Vec<u32>, bound: u32) -> Result<Vec<i32>, CodecError> {
    // Clamp internally rather than trusting the caller. A `bound` above the sign
    // limit would otherwise *widen* the check past what this function guaranteed
    // before it took a bound at all, letting a caller disable the sign guard by
    // accident. `clamp_index_bound` already does this for callers that use it;
    // doing it here too means no caller can get it wrong.
    let bound = bound.min(NO_INDEX_BOUND);
    if let Some((position, &bad)) = data.iter().enumerate().find(|&(_, &v)| v >= bound) {
        // Classify by whether a real column bound was declared, NOT by whether
        // the offending value also happens to exceed `i32::MAX`.
        //
        // Getting this backwards reintroduced the exact defect the typed
        // `IndexOutOfRange` variant exists to remove: with a declared bound, a
        // value ≥ 2^31 took the sign branch, so the scipy path reported
        // `MalformedInput` → `ScxError::Codec` → `RuntimeError` while the native
        // path reported `ShardIndexOutOfRange` → `CorruptFile` → `ValueError`
        // for the same payload. The Python exception type depended on whether
        // the caller asked for `f32` or a narrowed dtype.
        //
        // With a bound present, an out-of-range index is an out-of-range index
        // at any magnitude. The legacy sign-only message survives for
        // `NO_INDEX_BOUND`, where there is no column axis to be out of.
        //
        // A declared width at or above 2^31 clamps to the same sentinel, so it
        // takes the sign branch too. That is accurate rather than a collision:
        // an `i32` CSR cannot represent such a column at all, so "exceeds
        // i32::MAX" is the actual reason for the rejection, and naming the
        // declared width instead would describe a bound that is not what
        // stopped it. The native (`u32`) path legitimately accepts the same
        // index, because it has no sign hazard to begin with — the two domains
        // differ there because the *representations* differ, not because the
        // classification is inconsistent.
        return Err(if bound == NO_INDEX_BOUND {
            CodecError::MalformedInput(format!(
                "column index {bad} exceeds i32::MAX (corrupt or hostile input)"
            ))
        } else {
            CodecError::IndexOutOfRange {
                index: bad,
                position,
                bound,
            }
        });
    }
    Ok(bytemuck::cast_vec::<u32, i32>(data))
}

/// The bound that reproduces the pre-existing behaviour: reject only what would
/// reinterpret to a negative `i32`. Callers that know the shard's `n_minor` pass
/// [`clamp_index_bound`] instead.
pub const NO_INDEX_BOUND: u32 = i32::MAX as u32 + 1;

/// Clamp a shard's `n_minor` into a usable index bound.
///
/// `n_minor == 0` means the shard header does not *declare* a column axis — old
/// writers stamped the file-level `n_vars`, which is `0` on a multimodal file
/// because the real count is per-modality. Two multimodal conformance fixtures
/// carry `n_minor = 0` on shards with hundreds of nonzeros, so treating it as a
/// bound would reject valid files. Anything above `i32::MAX` is clamped down to
/// preserve the sign guarantee.
pub fn clamp_index_bound(n_minor: u32) -> u32 {
    if n_minor == 0 {
        NO_INDEX_BOUND
    } else {
        n_minor.min(NO_INDEX_BOUND)
    }
}

/// Fail loud if a zstd-decompressed sub-stream is not exactly the expected
/// length before an in-place undelta indexes it (a truncated frame would
/// otherwise mis-align the per-plane cumulative sum).
pub(crate) fn expect_exact_len(got: usize, expected: usize, which: &str) -> Result<(), CodecError> {
    if got != expected {
        return Err(CodecError::MalformedInput(format!(
            "shufdelta {which} decompressed length {got} != expected {expected}"
        )));
    }
    Ok(())
}

/// Checked `count * width` for decode allocation caps / expected byte lengths.
/// A hostile shard header can carry an `nnz`/`n_rows` that overflows `usize`
/// when scaled by a byte width; return a `MalformedInput` error instead of
/// panicking (debug) or wrapping to a bogus cap (release) (F-e).
pub(crate) fn checked_len(count: usize, width: usize, what: &str) -> Result<usize, CodecError> {
    count.checked_mul(width).ok_or_else(|| {
        CodecError::MalformedInput(format!("{what} length {count} * {width} overflows usize"))
    })
}

/// Checked byte cap for an indptr sub-stream: `(n_rows + 1) * 8`. Guards the
/// `+ 1` as well so a `usize::MAX` `n_rows` can't wrap to 0 before the multiply
/// (`n_rows` comes from a u32 header field today, but keep it panic-free) (F-e).
pub(crate) fn indptr_byte_cap(n_rows: usize) -> Result<usize, CodecError> {
    n_rows
        .checked_add(1)
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| {
            CodecError::MalformedInput(format!(
                "indptr length (n_rows={n_rows} + 1) * 8 overflows usize"
            ))
        })
}

/// Plausibility bound for a decode allocation (F-f): the number of output
/// elements a Scx1 primitive can produce is physically bounded by the number
/// of input *bits*, since the theoretical minimum is 1 bit per element (a
/// Golomb/Rice code with `k ≥ ceil(log2(max))` encodes zero in 1 bit). Reject
/// a header that declares more elements than `input_len * 8`, so a hostile
/// `nnz`/`n_rows` can't drive a `Vec::with_capacity` into an eager multi-GiB
/// allocation (or a 32-bit `capacity overflow` panic) from a few bytes of
/// compressed input. This bound is deliberately loose — it can never reject
/// valid data — and complements the `checked_len`/`indptr_byte_cap` overflow
/// guards, which catch a different failure mode (`usize` overflow).
///
/// Uses `checked_mul` (not `saturating_mul`): on a 32-bit target an
/// `input_len ≥ 512 MiB` would saturate `input_len * 8` to `usize::MAX`, and a
/// `declared == usize::MAX` would then slip past a saturating comparison. When
/// `input_len * 8` overflows `usize` the true bound exceeds any representable
/// `declared`, so the input is trivially plausible and we accept it.
pub(crate) fn bound_capacity(
    declared: usize,
    input_len: usize,
    what: &str,
) -> Result<usize, CodecError> {
    if let Some(max_elements) = input_len.checked_mul(8) {
        if declared > max_elements {
            return Err(CodecError::MalformedInput(format!(
                "{what}: declared {declared} elements but input is only {input_len} bytes \
                 (max {max_elements} elements at 1 bit/element)"
            )));
        }
    }
    Ok(declared)
}

/// Post-decode structural check: assert the three decoded arrays actually are
/// the CSR the shard header declared.
///
/// Every other guard in this module is a *pre*-decode plausibility bound on
/// byte lengths. This one runs after, and it is needed because a shard's three
/// sub-streams are decoded independently: `delta_golomb_decode` and
/// `rice_decode` return exactly the count the caller asked for, while
/// `forbp_decode_with_hint` used to return whatever its own per-row nnz varints
/// said. So a corrupt Scx1 shard could decode to `indptr = [0, 6]` with three
/// indices and six values — a structurally invalid CSR, returned as `Ok`.
///
/// Downstream that is not benign. `ScxCsr::new_unchecked` restates these
/// invariants as a caller obligation and only `debug_assert`s them, so in a
/// release build `scx_sparse::transpose::csr_to_csc` walks
/// `indptr[row]..indptr[row + 1]` and indexes past the end of `indices` —
/// a panic on malformed input, which the reader convention forbids.
///
/// Mirrors invariants 1–4 and 6 of `ScxCsr::new_unchecked`. Invariant 5 (every
/// index below the minor-axis extent) is enforced separately, by
/// `u32_vec_to_i32_bounded` here and `check_minor_indices` in `scx-format-io`,
/// because it needs `n_minor`, which the codec layer is not given.
///
/// `values_len` is an element count, not a byte count.
pub(crate) fn check_decoded_shape(
    indptr: &[u64],
    indices_len: usize,
    values_len: usize,
    n_rows: usize,
    nnz: usize,
) -> Result<(), CodecError> {
    check_indptr_shape(indptr, n_rows, Some(nnz))?;
    if indices_len != nnz {
        return Err(CodecError::MalformedInput(format!(
            "decoded CSR has {indices_len} indices != declared nnz {nnz}"
        )));
    }
    if values_len != nnz {
        return Err(CodecError::MalformedInput(format!(
            "decoded CSR has {values_len} values != declared nnz {nnz}"
        )));
    }
    Ok(())
}

/// Validate the indptr half of the CSR shape invariant: `len == n_rows + 1`,
/// starts at 0, monotone non-decreasing, and (when the caller knows it) ends at
/// the declared `nnz`.
///
/// Generic over the integer width because the decode families hand it different
/// types — the whole-shard decoders carry `u64` straight off the sub-stream,
/// while the indptr-only paths have already widened to the `i64` scipy layout.
/// One implementation rather than two on purpose: the hand-rolled first/last
/// check that used to live in [`decode_row_group_indptr_only`] is exactly how
/// the *monotonicity* half went missing on every GPU path that builds its
/// `GpuCsr` indptr from it.
///
/// Monotonicity is not redundant with the endpoint checks. An interior entry can
/// exceed `nnz` while the last one is honest — `[0, 5, 2]` for `nnz = 2` — and a
/// consumer walking `indptr[row]..indptr[row + 1]` then reads past the end of
/// `indices` one row early. Nor is a zero start implied by the codec:
/// `delta_golomb_decode` reads its first value as a raw LE `u64`, so an Scx1
/// indptr may begin anywhere even though its deltas are non-negative.
///
/// `nnz` is `None` for callers that decode the indptr alone and have no declared
/// non-zero count to compare against.
pub fn check_indptr_shape<T: Copy + Into<i128>>(
    indptr: &[T],
    n_rows: usize,
    nnz: Option<usize>,
) -> Result<(), CodecError> {
    let expected_indptr_len = n_rows
        .checked_add(1)
        .ok_or_else(|| CodecError::MalformedInput(format!("CSR n_rows+1 overflow: {n_rows}")))?;
    if indptr.len() != expected_indptr_len {
        return Err(CodecError::MalformedInput(format!(
            "decoded CSR indptr length {} != n_rows + 1 ({expected_indptr_len})",
            indptr.len()
        )));
    }
    let first: i128 = match indptr.first() {
        Some(&v) => v.into(),
        None => {
            return Err(CodecError::MalformedInput(
                "decoded CSR indptr is empty".into(),
            ))
        }
    };
    if first != 0 {
        return Err(CodecError::MalformedInput(format!(
            "decoded CSR indptr must start at 0, got {first}"
        )));
    }
    let mut prev = first;
    for (row, &raw) in indptr.iter().enumerate().skip(1) {
        let cur: i128 = raw.into();
        if cur < prev {
            return Err(CodecError::MalformedInput(format!(
                "decoded CSR indptr not monotone at row {}: {prev} > {cur}",
                row - 1
            )));
        }
        prev = cur;
    }
    if let Some(nnz) = nnz {
        if prev != nnz as i128 {
            return Err(CodecError::MalformedInput(format!(
                "decoded CSR indptr ends at {prev} != declared nnz {nnz}"
            )));
        }
    }
    Ok(())
}
