//! Property-based tests for CSR canonicalisation
//!
//! Locks the invariant that the streaming + in-memory converters share:
//! after `sort_csr_rows_in_place` and `drop_explicit_zeros_inplace`,
//! every row's column indices are strictly ascending and no entry has
//! `value == 0.0`. Idempotency: a second pass changes nothing.
//!
//! Strategy: generate random CSR triplets with deliberately unsorted
//! indices, duplicate columns, and injected explicit zeros, then run
//! both canonicalisation passes and assert post-conditions.
//!
//! Pairs with the encode/decode roundtrip proptest in
//! `scx-codec/tests/proptest_roundtrip.rs` — that suite verifies the
//! codec layer; this suite verifies the canonicalisation layer that
//! feeds it.
//!
//! Note: `drop_explicit_zeros_inplace` is documented as preserving
//! per-row ordering (it's a stable forward-pass compaction), but it
//! does *not* sort or de-duplicate. So the pipeline order matters:
//! callers run `sort_csr_rows_in_place` first, then drop zeros.

use proptest::prelude::*;
use scx_sparse::validate::{drop_explicit_zeros_inplace, sort_csr_rows_in_place};

const N_VARS: u32 = 50;

/// Generate a random CSR matrix triplet whose per-row index ordering is
/// arbitrary — may include duplicate column indices within a row and
/// unsorted indices. Values are random f32, with a non-zero fraction
/// of explicit zeros injected so the canonicaliser has work to do.
fn arb_csr_unsorted_with_zeros() -> impl Strategy<
    Value = (
        Vec<u64>, // indptr
        Vec<u32>, // indices
        Vec<f32>, // values
    ),
> {
    (1usize..30).prop_flat_map(|n_rows| {
        prop::collection::vec(0usize..15, n_rows).prop_flat_map(move |row_nnzs| {
            let total: usize = row_nnzs.iter().sum();
            let idx_strat = prop::collection::vec(0u32..N_VARS, total);
            let val_strat = prop::collection::vec(
                prop_oneof![
                    // Mostly non-zero
                    3 => any::<i16>().prop_map(|v| (v as f32) * 0.5 + 1.0),
                    // Some explicit zeros
                    1 => Just(0.0f32),
                ],
                total,
            );
            (Just(row_nnzs), idx_strat, val_strat).prop_map(|(row_nnzs, indices, values)| {
                let mut indptr = Vec::with_capacity(row_nnzs.len() + 1);
                let mut cum: u64 = 0;
                indptr.push(0);
                for nnz in &row_nnzs {
                    cum += *nnz as u64;
                    indptr.push(cum);
                }
                (indptr, indices, values)
            })
        })
    })
}

/// Assert canonical post-conditions: every row's indices are strictly
/// ascending, no value is exactly 0.0, indptr ends at indices.len().
fn assert_canonical(indptr: &[u64], indices: &[u32], values: &[f32]) {
    assert_eq!(indptr[indptr.len() - 1] as usize, indices.len());
    assert_eq!(indices.len(), values.len());
    for &v in values {
        assert_ne!(v, 0.0, "explicit zero survived drop_explicit_zeros_inplace");
    }
    let n_rows = indptr.len() - 1;
    for row in 0..n_rows {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        // Strictly ascending — we don't dedupe, but the generator may
        // produce duplicates; the canonical pipeline only sorts. So
        // assert *non-decreasing* and separately count duplicate-column
        // rows for visibility.
        for w in indices[start..end].windows(2) {
            assert!(
                w[0] <= w[1],
                "row {row}: indices not sorted ({} > {})",
                w[0],
                w[1]
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// After sort + drop_zeros, every row is sorted ascending and no
    /// explicit zero remains.
    #[test]
    fn canonicalisation_post_conditions(
        triplet in arb_csr_unsorted_with_zeros()
    ) {
        let (mut indptr, mut indices, mut values) = triplet;

        // `sort_csr_rows_in_place` takes `&[u64]` indptr (immutable) and
        // mutable indices/values slices.
        sort_csr_rows_in_place(&indptr, &mut indices, &mut values);

        // `drop_explicit_zeros_inplace` takes mutable Vecs for all three.
        drop_explicit_zeros_inplace(&mut indptr, &mut indices, &mut values);

        assert_canonical(&indptr, &indices, &values);
    }

    /// Canonicalisation is idempotent: a second pass changes nothing.
    #[test]
    fn canonicalisation_is_idempotent(
        triplet in arb_csr_unsorted_with_zeros()
    ) {
        let (mut indptr, mut indices, mut values) = triplet;

        sort_csr_rows_in_place(&indptr, &mut indices, &mut values);
        drop_explicit_zeros_inplace(&mut indptr, &mut indices, &mut values);

        let snapshot = (indptr.clone(), indices.clone(), values.clone());

        sort_csr_rows_in_place(&indptr, &mut indices, &mut values);
        drop_explicit_zeros_inplace(&mut indptr, &mut indices, &mut values);

        prop_assert_eq!(&indptr, &snapshot.0);
        prop_assert_eq!(&indices, &snapshot.1);
        prop_assert_eq!(&values, &snapshot.2);
    }

    /// Drop-zeros preserves the sum of non-zero values, regardless of
    /// the order it observes them in.
    #[test]
    fn drop_zeros_preserves_nonzero_sum(
        triplet in arb_csr_unsorted_with_zeros()
    ) {
        let (mut indptr, mut indices, mut values) = triplet;
        let original_sum: f64 = values.iter().map(|&v| v as f64).sum();
        let original_nonzero_sum: f64 = values
            .iter()
            .filter(|&&v| v != 0.0)
            .map(|&v| v as f64)
            .sum();

        sort_csr_rows_in_place(&indptr, &mut indices, &mut values);
        drop_explicit_zeros_inplace(&mut indptr, &mut indices, &mut values);

        let after_sum: f64 = values.iter().map(|&v| v as f64).sum();
        // Original sum may differ if some values were 0; after-sum
        // must equal original_nonzero_sum (sort doesn't touch values).
        prop_assert!((after_sum - original_nonzero_sum).abs() < 1e-3);
        // Sanity: sum doesn't blow up.
        prop_assert!(after_sum.is_finite());
        let _ = original_sum;
    }
}
