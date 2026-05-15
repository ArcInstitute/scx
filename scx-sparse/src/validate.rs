// Per-shard CSR validation and normalisation helpers.
//
// These three functions are shared between the in-memory pyscx
// converter (`pyscx::anndata`) and the streaming converter
// (`scx_convert::h5ad_stream`). Phase 2 of STREAMING-CONVERSION.md
// hoisted them out of their private homes so both paths apply the
// same canonical pre-encode normalisation.

use crate::csr::CsrError;

const INSERTION_THRESHOLD: usize = 32;

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
}
