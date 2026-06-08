// Per-shard CSR validation and normalisation helpers shared by the
// in-memory pyscx converter (`pyscx::anndata`) and the streaming
// converter (`scx_convert::h5ad_stream`). Centralising them here
// keeps the canonical pre-encode normalisation identical across
// both paths.

use crate::csr::CsrError;

const INSERTION_THRESHOLD: usize = 32;

/// The storage orientation a sparse matrix's on-disk arrays imply,
/// as classified by [`validate_sparse_layout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseLayout {
    /// indptr length matches `n_obs + 1` (and not `n_vars + 1`): row-major.
    Csr,
    /// indptr length matches `n_vars + 1` (and not `n_obs + 1`): column-major.
    Csc,
    /// The orientation can't be decided from the indptr length alone:
    /// either the matrix is square (`n_obs == n_vars`, so both lengths
    /// fit) or the indptr length fits neither (corrupt / unexpected
    /// layout). The caller must consult the `encoding-type` attribute
    /// or a documented default and/or reject the input.
    Ambiguous,
}

/// Classify a sparse matrix's storage orientation from its declared
/// `shape = (n_obs, n_vars)`, on-disk `indptr` length, and (optionally)
/// the largest index value present in `indices`.
///
/// An h5ad group missing its `encoding-type` attribute carries no
/// orientation marker, yet a CSC group has `indptr` + `indices` just
/// like CSR. Misclassifying a CSC matrix as CSR panics (the
/// `n_vars + 1`-length indptr indexes out of bounds) or silently
/// transposes the data (review finding C1). This is the single rule
/// shared by h5ad format detection and the eager/streaming X readers
/// so all three agree.
///
/// When `max_index` is provided (the readers know it; detection does
/// not scan for it), an orientation that would place an index past its
/// implied minor dimension is downgraded to [`SparseLayout::Ambiguous`]
/// rather than committed to — a CSR's column indices must be `< n_vars`
/// and a CSC's row indices `< n_obs`.
pub fn validate_sparse_layout(
    shape: (usize, usize),
    indptr_len: usize,
    max_index: Option<i64>,
) -> SparseLayout {
    let (n_obs, n_vars) = shape;
    let fits_csr = indptr_len == n_obs.saturating_add(1);
    let fits_csc = indptr_len == n_vars.saturating_add(1);
    match (fits_csr, fits_csc) {
        (true, false) => {
            if max_index.is_some_and(|m| m >= n_vars as i64) {
                SparseLayout::Ambiguous
            } else {
                SparseLayout::Csr
            }
        }
        (false, true) => {
            if max_index.is_some_and(|m| m >= n_obs as i64) {
                SparseLayout::Ambiguous
            } else {
                SparseLayout::Csc
            }
        }
        // Both fit (square) or neither fits (corrupt): undecidable here.
        _ => SparseLayout::Ambiguous,
    }
}

/// Validate the shape invariants of a CSR triplet:
/// - `indptr[0] >= 0`,
/// - indptr monotonically non-decreasing,
/// - every entry of `indices` falls in `[0, n_vars)`.
///
/// Does not require `indptr[0] == 0` — callers may pass a shard-local
/// slice whose first value reflects the cumulative nnz at the shard's
/// row offset.
pub fn validate_csr_arrays(indptr: &[i64], indices: &[i32], n_vars: u64) -> Result<(), CsrError> {
    if let Some(&first) = indptr.first() {
        if first < 0 {
            return Err(CsrError::IndptrNegative {
                value: first,
                position: 0,
            });
        }
    }
    for i in 1..indptr.len() {
        if indptr[i] < indptr[i - 1] {
            return Err(CsrError::IndptrNotMonotonic { index: i });
        }
    }
    for (position, &idx) in indices.iter().enumerate() {
        if idx < 0 || (idx as u64) >= n_vars {
            return Err(CsrError::IndexOutOfRange {
                index: idx,
                n_cols: n_vars as usize,
                position,
            });
        }
    }
    Ok(())
}

/// Compute the `[start, end)` nnz range a CSR shard occupies, from
/// its shard-local indptr slice (`indptr[row_start..=row_end]`, length
/// `n_rows + 1`). Used to slice the `indices` / `data` arrays (in
/// memory or from disk) before rebasing.
///
/// Guards the non-negative base and `end >= start` monotonicity at the
/// endpoints so a reversed range can't silently produce an
/// underflowed slice (the bug behind review finding C5). Also guards
/// the upper bound: `end` must not exceed `backing_len` (the length of
/// the `indices` / `data` array the returned range slices into), so a
/// corrupt indptr whose last value overruns the backing array returns
/// a clean [`CsrError::NnzOutOfRange`] instead of panicking at the
/// slice. In-memory callers pass `indices.len().min(data.len())`;
/// streaming callers pass the on-disk dataset length. Full per-element
/// validation happens in [`rebase_csr_shard`].
pub fn shard_nnz_bounds(
    indptr_slice: &[i64],
    backing_len: usize,
) -> Result<(usize, usize), CsrError> {
    let base = indptr_slice.first().copied().unwrap_or(0);
    let end = indptr_slice.last().copied().unwrap_or(0);
    if base < 0 {
        return Err(CsrError::IndptrNegative {
            value: base,
            position: 0,
        });
    }
    if end < base {
        return Err(CsrError::IndptrNotMonotonic {
            index: indptr_slice.len().saturating_sub(1),
        });
    }
    let end_usize = end as usize;
    if end_usize > backing_len {
        return Err(CsrError::NnzOutOfRange {
            nnz_end: end_usize,
            backing_len,
        });
    }
    Ok((base as usize, end_usize))
}

/// Rebase a shard-local CSR slice into a standalone shard: validate
/// (monotonic indptr + column bound `indices < n_vars` via
/// [`validate_csr_arrays`]), rebase the indptr so it starts at 0, and
/// cast widths (indptr → `u64`, indices → `u32`).
///
/// `indptr_slice` is `indptr[row_start..=row_end]` (length `n_rows +
/// 1`, values still relative to the file-wide nnz origin);
/// `shard_indices` is the `i32` indices already sliced to this shard's
/// nnz range (see [`shard_nnz_bounds`]). The caller slices `data`
/// itself with the same range, because some callers read `data` lazily
/// from disk and never hold the whole array.
///
/// Single source of truth for the per-shard rebase performed by every
/// ingest path (the two eager `pipeline.rs` sites, the two streaming
/// `h5ad_stream.rs` readers, and the materialized-CSC
/// `csc_stream.rs` reader). Folding them here closed review findings
/// C5 (missing monotonicity guard) and C6 (the eager sites previously
/// skipped the column-bound check).
pub fn rebase_csr_shard(
    indptr_slice: &[i64],
    shard_indices: &[i32],
    n_vars: u64,
) -> Result<(Vec<u64>, Vec<u32>), CsrError> {
    validate_csr_arrays(indptr_slice, shard_indices, n_vars)?;
    // `validate_csr_arrays` guarantees `indptr_slice[0] >= 0` and
    // monotonic non-decreasing, so `base` is non-negative and every
    // `v - base` is non-negative — the `as u64` cast is lossless.
    let base = indptr_slice.first().copied().unwrap_or(0);
    let indptr: Vec<u64> = indptr_slice.iter().map(|&v| (v - base) as u64).collect();
    // Indices are validated into `[0, n_vars)`, so `as u32` is lossless.
    let indices: Vec<u32> = shard_indices.iter().map(|&v| v as u32).collect();
    Ok((indptr, indices))
}

/// Sort each row's `(indices, values)` pairs by column index, in
/// lock-step. Small rows (≤ [`INSERTION_THRESHOLD`] nnz) use an
/// in-place insertion sort that allocates nothing; longer rows
/// build a temporary `Vec<(u32, f32)>` and `sort_by_key`.
///
/// Idempotent on already-sorted input.
pub fn sort_csr_rows_in_place(indptr: &[u64], indices: &mut [u32], values: &mut [f32]) {
    let n_rows = indptr.len().saturating_sub(1);
    for row in 0..n_rows {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        if end <= start + 1 {
            continue;
        }
        let row_idx = &mut indices[start..end];
        let row_val = &mut values[start..end];
        if row_idx.len() <= INSERTION_THRESHOLD {
            for i in 1..row_idx.len() {
                let mut j = i;
                while j > 0 && row_idx[j - 1] > row_idx[j] {
                    row_idx.swap(j - 1, j);
                    row_val.swap(j - 1, j);
                    j -= 1;
                }
            }
        } else {
            let mut pairs: Vec<(u32, f32)> = row_idx
                .iter()
                .zip(row_val.iter())
                .map(|(&i, &v)| (i, v))
                .collect();
            pairs.sort_by_key(|p| p.0);
            for (k, (i, v)) in pairs.into_iter().enumerate() {
                row_idx[k] = i;
                row_val[k] = v;
            }
        }
    }
}

/// Compact `indices`/`values` in place, dropping entries where the
/// value is exactly `0.0`, and rewriting `indptr` so it reflects the
/// new cumulative nnz per row.
///
/// Single forward pass: each iteration reads the original
/// `indptr[row + 1]` (saved into `next_start` before any write
/// touches that slot) and then overwrites it with the new cumulative
/// nnz. `indptr[0]` must be 0 — the streaming caller always rebases
/// shard-local indptrs to start at 0 before invoking this.
///
/// `indices` and `values` need `&mut Vec` (not `&mut [_]`) because
/// compaction shortens them via `Vec::truncate`. `indptr` doesn't
/// change length, so it could be a slice — kept as `&mut Vec` for
/// caller symmetry.
#[allow(clippy::ptr_arg)]
pub fn drop_explicit_zeros_inplace(
    indptr: &mut Vec<u64>,
    indices: &mut Vec<u32>,
    values: &mut Vec<f32>,
) {
    debug_assert!(
        !indptr.is_empty() && indptr[0] == 0,
        "indptr must be non-empty and start at 0"
    );

    if values.iter().all(|&v| v != 0.0) {
        return;
    }

    let n_rows = indptr.len() - 1;
    let mut write: u64 = 0;
    let mut next_start: u64 = indptr[0];
    for row in 0..n_rows {
        let start = next_start as usize;
        let end = indptr[row + 1] as usize;
        next_start = indptr[row + 1];
        for k in start..end {
            if values[k] != 0.0 {
                indices[write as usize] = indices[k];
                values[write as usize] = values[k];
                write += 1;
            }
        }
        indptr[row + 1] = write;
    }
    indices.truncate(write as usize);
    values.truncate(write as usize);
}

/// Sum the values of entries that share the same `(row, col)`
/// coordinate (the MatrixMarket / scipy `sum_duplicates()` rule).
/// `records` must already be sorted by `(row, col)`. Returns the
/// number of duplicate pairs that were merged (`0` means no
/// duplicates). Resulting `0.0` values are left in place — callers
/// drop them downstream (`drop_explicit_zeros_inplace` for CSR
/// arrays, the MTX/CSC builders' own zero-skip).
///
/// Single source of truth for the "sum duplicates" rule across the
/// untrusted COO ingest paths (the MTX reader and the external-memory
/// CSC→CSR transposer), closing the duplicated implementations behind
/// review finding CLI1. Generic over the coordinate types so both
/// `(usize, usize, f32)` (MTX) and `(u64, u32, f32)` (CSC transpose)
/// callers share it.
pub fn coalesce_sorted_coo<R, C>(records: &mut Vec<(R, C, f32)>) -> u64
where
    R: Copy + PartialEq,
    C: Copy + PartialEq,
{
    if records.is_empty() {
        return 0;
    }
    let mut dup_count: u64 = 0;
    let mut write = 0usize;
    for read in 1..records.len() {
        let (r, c, v) = records[read];
        let (pr, pc, pv) = records[write];
        if r == pr && c == pc {
            // Merge into the previous slot.
            records[write] = (pr, pc, pv + v);
            dup_count += 1;
        } else {
            write += 1;
            records[write] = (r, c, v);
        }
    }
    records.truncate(write + 1);
    dup_count
}

/// Sum adjacent entries that share a column index within each row.
/// Rows must already be sorted by column (see
/// [`sort_csr_rows_in_place`]); `indptr[0]` must be 0. Rewrites
/// `indptr` and compacts `indices` / `values` in a single forward
/// pass. Resulting `0.0` values are left for
/// [`drop_explicit_zeros_inplace`] to remove.
#[allow(clippy::ptr_arg)]
fn dedup_sum_sorted_rows(indptr: &mut [u64], indices: &mut Vec<u32>, values: &mut Vec<f32>) {
    debug_assert!(
        !indptr.is_empty() && indptr[0] == 0,
        "indptr must be non-empty and start at 0"
    );
    let n_rows = indptr.len().saturating_sub(1);
    let mut write: usize = 0;
    let mut next_start: usize = 0;
    for row in 0..n_rows {
        let start = next_start;
        let end = indptr[row + 1] as usize;
        next_start = end;
        let mut k = start;
        while k < end {
            let col = indices[k];
            let mut sum = values[k];
            let mut j = k + 1;
            while j < end && indices[j] == col {
                sum += values[j];
                j += 1;
            }
            indices[write] = col;
            values[write] = sum;
            write += 1;
            k = j;
        }
        indptr[row + 1] = write as u64;
    }
    indices.truncate(write);
    values.truncate(write);
}

/// Returns `true` iff [`canonicalize_csr`] would leave this CSR
/// unchanged: every row's column indices are strictly increasing
/// (already sorted **and** free of duplicate `(row, col)` coordinates —
/// duplicates manifest as equal adjacent indices) and no stored value
/// is an explicit zero. Read-only, `O(nnz)`, allocates nothing.
/// `indptr[0]` is assumed to be 0.
///
/// Lets hot ingest paths skip the canonicalize copy/sort when the input
/// (e.g. a scipy CSR with `has_canonical_format == True`) is already
/// canonical — the overwhelmingly common case.
pub fn is_canonical_csr(indptr: &[u64], indices: &[u32], values: &[f32]) -> bool {
    let n_rows = indptr.len().saturating_sub(1);
    for row in 0..n_rows {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        if end < start || end > indices.len() || end > values.len() {
            return false;
        }
        let row_idx = &indices[start..end];
        for w in row_idx.windows(2) {
            if w[1] <= w[0] {
                return false;
            }
        }
        if values[start..end].contains(&0.0) {
            return false;
        }
    }
    true
}

/// Canonicalize a CSR matrix in place for *untrusted* ingest sources
/// (MatrixMarket per review finding CLI1; the messy-h5ad CSC→CSR
/// transpose): sort each row by column index, sum entries that share a
/// `(row, col)` coordinate, and drop the resulting explicit zeros.
/// `indptr[0]` must be 0.
///
/// Idempotent on already-canonical input (and short-circuits via
/// [`is_canonical_csr`] in that case, skipping the per-row sort), but
/// **do not** call it on SCX→SCX shard rewrites — it would needlessly
/// re-scan canonical shards and regress streaming-write throughput. Hot
/// per-shard rebases use [`rebase_csr_shard`] (validation only, no
/// sort/dedup) instead.
#[allow(clippy::ptr_arg)]
pub fn canonicalize_csr(indptr: &mut Vec<u64>, indices: &mut Vec<u32>, values: &mut Vec<f32>) {
    if is_canonical_csr(indptr, indices, values) {
        return;
    }
    sort_csr_rows_in_place(indptr, indices, values);
    dedup_sum_sorted_rows(indptr, indices, values);
    drop_explicit_zeros_inplace(indptr, indices, values);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_csr_arrays_ok() {
        let indptr = [0i64, 2, 5, 5, 7];
        let indices = [0i32, 3, 1, 2, 4, 0, 2];
        let n_vars: u64 = 5;
        validate_csr_arrays(&indptr, &indices, n_vars).unwrap();
    }

    #[test]
    fn validate_csr_arrays_allows_nonzero_indptr_start() {
        // Shard-local slice starting at 17 (cumulative nnz at the
        // shard's row offset) — must validate.
        let indptr = [17i64, 19, 22];
        let indices = [0i32, 1, 2, 3, 4];
        validate_csr_arrays(&indptr, &indices, 5).unwrap();
    }

    #[test]
    fn validate_csr_arrays_rejects_negative_first() {
        let indptr = [-1i64, 0, 2];
        let indices = [0i32, 1];
        let err = validate_csr_arrays(&indptr, &indices, 4).unwrap_err();
        match err {
            CsrError::IndptrNegative { value, position } => {
                assert_eq!(value, -1);
                assert_eq!(position, 0);
            }
            other => panic!("expected IndptrNegative, got {other:?}"),
        }
    }

    #[test]
    fn validate_csr_arrays_rejects_non_monotonic() {
        let indptr = [0i64, 3, 2];
        let indices = [0i32, 1, 2];
        let err = validate_csr_arrays(&indptr, &indices, 4).unwrap_err();
        match err {
            CsrError::IndptrNotMonotonic { index } => assert_eq!(index, 2),
            other => panic!("expected IndptrNotMonotonic, got {other:?}"),
        }
    }

    #[test]
    fn validate_csr_arrays_rejects_oob_index() {
        let indptr = [0i64, 2];
        let indices = [0i32, 7];
        let err = validate_csr_arrays(&indptr, &indices, 5).unwrap_err();
        match err {
            CsrError::IndexOutOfRange {
                index,
                n_cols,
                position,
            } => {
                assert_eq!(index, 7);
                assert_eq!(n_cols, 5);
                assert_eq!(position, 1);
            }
            other => panic!("expected IndexOutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn validate_csr_arrays_rejects_negative_index() {
        let indptr = [0i64, 1];
        let indices = [-3i32];
        let err = validate_csr_arrays(&indptr, &indices, 5).unwrap_err();
        match err {
            CsrError::IndexOutOfRange { index, .. } => assert_eq!(index, -3),
            other => panic!("expected IndexOutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn sort_csr_rows_in_place_short_row() {
        // Single row, 3 nnz, descending — insertion sort path.
        let indptr = [0u64, 3];
        let mut indices = vec![2u32, 0, 1];
        let mut values = vec![20.0f32, 0.0, 10.0];
        sort_csr_rows_in_place(&indptr, &mut indices, &mut values);
        assert_eq!(indices, vec![0u32, 1, 2]);
        assert_eq!(values, vec![0.0f32, 10.0, 20.0]);
    }

    #[test]
    fn sort_csr_rows_in_place_long_row() {
        // 64 nnz reversed — exercises the pairwise sort path.
        let n: usize = 64;
        let indptr = [0u64, n as u64];
        let mut indices: Vec<u32> = (0..n as u32).rev().collect();
        let mut values: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        sort_csr_rows_in_place(&indptr, &mut indices, &mut values);
        let expected_idx: Vec<u32> = (0..n as u32).collect();
        let expected_val: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        assert_eq!(indices, expected_idx);
        assert_eq!(values, expected_val);
    }

    #[test]
    fn sort_csr_rows_idempotent() {
        let indptr = [0u64, 4];
        let original_indices = vec![0u32, 1, 2, 3];
        let original_values = vec![10.0f32, 20.0, 30.0, 40.0];
        let mut indices = original_indices.clone();
        let mut values = original_values.clone();
        sort_csr_rows_in_place(&indptr, &mut indices, &mut values);
        assert_eq!(indices, original_indices);
        assert_eq!(values, original_values);
    }

    #[test]
    fn sort_csr_rows_multiple_rows() {
        // Two rows, both descending; verifies per-row scoping.
        let indptr = [0u64, 3, 6];
        let mut indices = vec![2u32, 1, 0, 4, 2, 3];
        let mut values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        sort_csr_rows_in_place(&indptr, &mut indices, &mut values);
        assert_eq!(indices, vec![0u32, 1, 2, 2, 3, 4]);
        assert_eq!(values, vec![3.0f32, 2.0, 1.0, 5.0, 6.0, 4.0]);
    }

    #[test]
    fn drop_explicit_zeros_inplace_basic() {
        let mut indptr = vec![0u64, 3, 5];
        let mut indices = vec![0u32, 1, 2, 1, 3];
        let mut values = vec![1.0f32, 0.0, 2.0, 0.0, 3.0];
        drop_explicit_zeros_inplace(&mut indptr, &mut indices, &mut values);
        assert_eq!(indptr, vec![0u64, 2, 3]);
        assert_eq!(indices, vec![0u32, 2, 3]);
        assert_eq!(values, vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn drop_explicit_zeros_inplace_no_zeros() {
        let mut indptr = vec![0u64, 2, 3];
        let original_indices = vec![0u32, 1, 2];
        let original_values = vec![1.0f32, 2.0, 3.0];
        let mut indices = original_indices.clone();
        let mut values = original_values.clone();
        drop_explicit_zeros_inplace(&mut indptr, &mut indices, &mut values);
        assert_eq!(indptr, vec![0u64, 2, 3]);
        assert_eq!(indices, original_indices);
        assert_eq!(values, original_values);
    }

    #[test]
    fn drop_explicit_zeros_inplace_all_zeros() {
        let mut indptr = vec![0u64, 2, 4];
        let mut indices = vec![0u32, 1, 2, 3];
        let mut values = vec![0.0f32, 0.0, 0.0, 0.0];
        drop_explicit_zeros_inplace(&mut indptr, &mut indices, &mut values);
        assert_eq!(indptr, vec![0u64, 0, 0]);
        assert!(indices.is_empty());
        assert!(values.is_empty());
    }

    /// Composition: a shard with unsorted indices + explicit zeros,
    /// normalised by `sort_csr_rows_in_place` then
    /// `drop_explicit_zeros_inplace`, produces the canonical form a
    /// scipy-style pipeline (`sorted_indices()` + drop-zeros) would
    /// produce.
    #[test]
    fn canonical_form_composition_matches_reference() {
        // Two rows. Row 0: indices [3, 0, 2] with values [0, 5, 0].
        // Row 1: indices [4, 1, 4, 0] with values [7, 0, 8, 9] (note
        // duplicate column 4 is allowed pre-canonicalisation —
        // we don't dedup here; only sort+filter).
        let indptr = vec![0u64, 3, 7];
        let indices = vec![3u32, 0, 2, 4, 1, 4, 0];
        let values = vec![0.0f32, 5.0, 0.0, 7.0, 0.0, 8.0, 9.0];

        // --- Reference: hand-rolled sort + zero-drop. ---
        let mut ref_indptr = vec![0u64];
        let mut ref_indices: Vec<u32> = Vec::new();
        let mut ref_values: Vec<f32> = Vec::new();
        for row in 0..indptr.len() - 1 {
            let s = indptr[row] as usize;
            let e = indptr[row + 1] as usize;
            let mut pairs: Vec<(u32, f32)> = indices[s..e]
                .iter()
                .zip(values[s..e].iter())
                .map(|(&i, &v)| (i, v))
                .collect();
            pairs.sort_by_key(|p| p.0);
            for (i, v) in pairs {
                if v != 0.0 {
                    ref_indices.push(i);
                    ref_values.push(v);
                }
            }
            ref_indptr.push(ref_indices.len() as u64);
        }

        // --- Helpers under test. ---
        let mut hot_indptr = indptr.clone();
        let mut hot_indices = indices.clone();
        let mut hot_values = values.clone();
        sort_csr_rows_in_place(&hot_indptr, &mut hot_indices, &mut hot_values);
        drop_explicit_zeros_inplace(&mut hot_indptr, &mut hot_indices, &mut hot_values);

        assert_eq!(hot_indptr, ref_indptr);
        assert_eq!(hot_indices, ref_indices);
        assert_eq!(hot_values, ref_values);
    }

    // ---- shard_nnz_bounds (C5) ----

    #[test]
    fn shard_nnz_bounds_basic() {
        // Shard-local slice with a non-zero base. `backing_len` >= end.
        assert_eq!(shard_nnz_bounds(&[17i64, 19, 22], 22).unwrap(), (17, 22));
    }

    #[test]
    fn shard_nnz_bounds_rejects_reversed_range() {
        // C5: a reversed endpoint range must error, not underflow-wrap
        // into a giant slice.
        let err = shard_nnz_bounds(&[22i64, 19, 17], 22).unwrap_err();
        assert!(matches!(err, CsrError::IndptrNotMonotonic { .. }));
    }

    #[test]
    fn shard_nnz_bounds_rejects_negative_base() {
        let err = shard_nnz_bounds(&[-1i64, 3], 3).unwrap_err();
        assert!(matches!(err, CsrError::IndptrNegative { .. }));
    }

    #[test]
    fn shard_nnz_bounds_rejects_overrun() {
        // A corrupt indptr whose last value overruns the backing array
        // must error cleanly rather than panic at the downstream slice.
        let err = shard_nnz_bounds(&[0i64, 5], 3).unwrap_err();
        assert!(matches!(
            err,
            CsrError::NnzOutOfRange {
                nnz_end: 5,
                backing_len: 3
            }
        ));
    }

    // ---- rebase_csr_shard (C6) ----

    #[test]
    fn rebase_csr_shard_rebases_and_casts() {
        // Shard-local indptr starting at 17; indices already sliced.
        let indptr_slice = [17i64, 19, 22];
        let shard_indices = [0i32, 3, 1, 2, 4];
        let (indptr, indices) = rebase_csr_shard(&indptr_slice, &shard_indices, 5).unwrap();
        assert_eq!(indptr, vec![0u64, 2, 5]);
        assert_eq!(indices, vec![0u32, 3, 1, 2, 4]);
    }

    #[test]
    fn rebase_csr_shard_rejects_out_of_range_column() {
        // C6: the column-bound check the eager sites previously skipped.
        let indptr_slice = [0i64, 2];
        let shard_indices = [0i32, 9];
        let err = rebase_csr_shard(&indptr_slice, &shard_indices, 5).unwrap_err();
        assert!(matches!(err, CsrError::IndexOutOfRange { index: 9, .. }));
    }

    #[test]
    fn rebase_csr_shard_rejects_non_monotonic() {
        let indptr_slice = [0i64, 3, 2];
        let shard_indices = [0i32, 1, 2];
        let err = rebase_csr_shard(&indptr_slice, &shard_indices, 5).unwrap_err();
        assert!(matches!(err, CsrError::IndptrNotMonotonic { .. }));
    }

    // ---- coalesce_sorted_coo (CLI1) ----

    #[test]
    fn coalesce_sorted_coo_no_duplicates() {
        let mut v = vec![(0u64, 0u32, 1.0f32), (0, 1, 2.0), (1, 0, 3.0)];
        assert_eq!(coalesce_sorted_coo(&mut v), 0);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn coalesce_sorted_coo_sums_duplicates() {
        let mut v = vec![(0u64, 0u32, 1.0f32), (0, 0, 2.5), (0, 0, 0.5), (1, 0, 3.0)];
        assert_eq!(coalesce_sorted_coo(&mut v), 2);
        assert_eq!(v, vec![(0, 0, 4.0), (1, 0, 3.0)]);
    }

    #[test]
    fn coalesce_sorted_coo_generic_usize_coords() {
        // The MTX reader's `(usize, usize, f32)` coordinate type.
        let mut v = vec![(0usize, 2usize, 1.0f32), (0, 2, 4.0), (3, 1, 2.0)];
        assert_eq!(coalesce_sorted_coo(&mut v), 1);
        assert_eq!(v, vec![(0, 2, 5.0), (3, 1, 2.0)]);
    }

    // ---- canonicalize_csr ----

    #[test]
    fn canonicalize_csr_sorts_dedups_and_drops_zeros() {
        // Row 0: unsorted with a duplicate column 2 (3.0 + (-3.0) = 0 → dropped)
        //        and an explicit zero at column 1.
        // Row 1: sorted, one duplicate column 0 (1.0 + 2.0 = 3.0).
        let mut indptr = vec![0u64, 4, 7];
        let mut indices = vec![2u32, 0, 1, 2, 0, 0, 4];
        let mut values = vec![3.0f32, 5.0, 0.0, -3.0, 1.0, 2.0, 7.0];
        canonicalize_csr(&mut indptr, &mut indices, &mut values);
        // Row 0: col 0 = 5.0 (col 1 zero dropped, col 2 summed to 0 dropped).
        // Row 1: col 0 = 3.0, col 4 = 7.0.
        assert_eq!(indptr, vec![0u64, 1, 3]);
        assert_eq!(indices, vec![0u32, 0, 4]);
        assert_eq!(values, vec![5.0f32, 3.0, 7.0]);
    }

    #[test]
    fn canonicalize_csr_idempotent_on_canonical_input() {
        let mut indptr = vec![0u64, 2, 3];
        let mut indices = vec![0u32, 2, 1];
        let mut values = vec![1.0f32, 2.0, 3.0];
        canonicalize_csr(&mut indptr, &mut indices, &mut values);
        assert_eq!(indptr, vec![0u64, 2, 3]);
        assert_eq!(indices, vec![0u32, 2, 1]);
        assert_eq!(values, vec![1.0f32, 2.0, 3.0]);
    }

    // ---- is_canonical_csr ----

    #[test]
    fn is_canonical_csr_accepts_canonical() {
        let indptr = vec![0u64, 2, 3];
        let indices = vec![0u32, 2, 1];
        let values = vec![1.0f32, 2.0, 3.0];
        assert!(is_canonical_csr(&indptr, &indices, &values));
    }

    #[test]
    fn is_canonical_csr_rejects_unsorted_dup_and_zero() {
        // Unsorted within a row.
        assert!(!is_canonical_csr(&[0, 2], &[2, 0], &[1.0, 2.0]));
        // Duplicate column (equal adjacent indices).
        assert!(!is_canonical_csr(&[0, 2], &[1, 1], &[1.0, 2.0]));
        // Explicit zero value.
        assert!(!is_canonical_csr(&[0, 2], &[0, 1], &[1.0, 0.0]));
    }

    // ---- validate_sparse_layout (C1) ----

    #[test]
    fn validate_sparse_layout_non_square() {
        // shape (100 obs, 50 vars): CSR indptr len 101, CSC indptr len 51.
        assert_eq!(
            validate_sparse_layout((100, 50), 101, None),
            SparseLayout::Csr
        );
        assert_eq!(
            validate_sparse_layout((100, 50), 51, None),
            SparseLayout::Csc
        );
    }

    #[test]
    fn validate_sparse_layout_square_is_ambiguous() {
        assert_eq!(
            validate_sparse_layout((64, 64), 65, None),
            SparseLayout::Ambiguous
        );
    }

    #[test]
    fn validate_sparse_layout_corrupt_length_is_ambiguous() {
        // Fits neither n_obs+1 nor n_vars+1.
        assert_eq!(
            validate_sparse_layout((100, 50), 7, None),
            SparseLayout::Ambiguous
        );
    }

    #[test]
    fn validate_sparse_layout_max_index_downgrades_impossible_csr() {
        // Length says CSR, but a column index >= n_vars can't be CSR.
        assert_eq!(
            validate_sparse_layout((100, 50), 101, Some(50)),
            SparseLayout::Ambiguous
        );
        // In-range max index keeps the CSR classification.
        assert_eq!(
            validate_sparse_layout((100, 50), 101, Some(49)),
            SparseLayout::Csr
        );
    }
}
