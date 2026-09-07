//! `GroupNonzeroCounts` — dense, CSR and streamed inputs agree; the counting
//! rule is scanpy's (`!= 0`, explicit zeros ignored, negatives counted; an
//! unlabelled cell is in no group but in every group's rest); the fractions
//! divide by the right pools.

use super::*;
use crate::hvg::cpu::InMemorySource;

const OOR: usize = 3;

/// 6 cells × 4 genes; labels `[0, 0, 1, 1, 2, unlabelled]`.
///
/// Row 5 (unlabelled) is dense with nonzeros so a leak into a group count
/// shows — and so its presence in every `pts_rest` is visible. Gene 3 is zero
/// in every labelled cell, so its `pts` is 0 everywhere while its `pts_rest`
/// is the unlabelled row alone.
fn dense_fixture() -> (Vec<f32>, Vec<usize>) {
    #[rustfmt::skip]
    let data: Vec<f32> = vec![
        1.0, 0.0, 2.0, 0.0,   // g0
        0.0, 0.0, 3.0, 0.0,   // g0
        -1.0, 5.0, 0.0, 0.0,  // g1  (negative counts as expressing)
        0.0, 0.0, 0.0, 0.0,   // g1
        4.0, 4.0, 4.0, 0.0,   // g2
        9.0, 9.0, 9.0, 9.0,   // unlabelled: in no group, in every group's rest
    ];
    (data, vec![0, 0, 1, 1, 2, OOR])
}

/// CSR of the fixture without stored zeros.
fn csr_fixture() -> ScxCsr {
    let (data, _) = dense_fixture();
    dense_to_csr(&data, 6, 4)
}

fn dense_to_csr(data: &[f32], n_obs: usize, n_vars: usize) -> ScxCsr {
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..n_obs {
        for col in 0..n_vars {
            let v = data[row * n_vars + col];
            if v != 0.0 {
                indices.push(col as i32);
                values.push(v);
            }
        }
        indptr.push(indices.len() as i64);
    }
    ScxCsr::new((n_obs, n_vars), indptr, indices, values).unwrap()
}

fn expected_counts() -> GroupNonzeroCounts {
    GroupNonzeroCounts {
        counts: vec![
            vec![1, 0, 2, 0], // g0: rows 0, 1
            vec![1, 1, 0, 0], // g1: rows 2, 3
            vec![1, 1, 1, 0], // g2: row 4
        ],
        // Every row, the unlabelled one included (scanpy's `~mask_g`).
        total: vec![4, 3, 4, 1],
        group_sizes: vec![2, 2, 1],
        n_obs: 6,
    }
}

#[test]
fn dense_counts_match_the_hand_count_and_skip_unlabelled_rows() {
    let (data, groups) = dense_fixture();
    let got = group_nonzero_counts_dense(&data, 6, 4, &groups, 3).unwrap();
    assert_eq!(got, expected_counts());
}

#[test]
fn csr_counts_equal_dense_counts() {
    let (data, groups) = dense_fixture();
    let dense = group_nonzero_counts_dense(&data, 6, 4, &groups, 3).unwrap();
    let csr = group_nonzero_counts_csr(&csr_fixture(), &groups, 3).unwrap();
    assert_eq!(csr, dense);
}

#[test]
fn streamed_counts_over_two_shards_equal_the_single_csr() {
    let (data, groups) = dense_fixture();
    let whole = group_nonzero_counts_csr(&csr_fixture(), &groups, 3).unwrap();
    let source = InMemorySource {
        shards: vec![
            dense_to_csr(&data[..3 * 4], 3, 4),
            dense_to_csr(&data[3 * 4..], 3, 4),
        ],
        n_obs: 6,
        n_vars: 4,
    };
    let streamed = group_nonzero_counts_streaming(&source, &groups, 3).unwrap();
    assert_eq!(streamed, whole);
}

#[test]
fn an_explicitly_stored_zero_is_not_expressing() {
    // scanpy runs `eliminate_zeros()` before `getnnz`; a scipy CSR handed to
    // pyscx may still carry stored zeros, and they must not count.
    let csr = ScxCsr::new((2, 2), vec![0, 2, 3], vec![0, 1, 0], vec![0.0, 7.0, 0.0]).unwrap();
    let got = group_nonzero_counts_csr(&csr, &[0, 0], 1).unwrap();
    assert_eq!(got.counts, vec![vec![0, 1]]);
    assert_eq!(got.total, vec![0, 1]);
}

#[test]
fn fractions_divide_by_group_size_and_every_other_cell() {
    let counts = expected_counts();
    let f = counts.fractions(None);
    // g0 has 2 cells: gene 0 in 1 of 2, gene 2 in 2 of 2.
    assert_eq!(f.pts[0], vec![0.5, 0.0, 1.0, 0.0]);
    // g2 has 1 cell.
    assert_eq!(f.pts[2], vec![1.0, 1.0, 1.0, 0.0]);
    // rest of g0 = the other 4 rows (2, 3, 4 and the unlabelled 5): gene 0 in
    // 3 of them, gene 1 in 3, gene 2 in 2, gene 3 in 1 (row 5 only) — scanpy's
    // `X[~mask_g]`, so the unlabelled row is in numerator and denominator.
    let rest = f.pts_rest.as_ref().unwrap();
    assert_eq!(rest[0], vec![3.0 / 4.0, 3.0 / 4.0, 2.0 / 4.0, 1.0 / 4.0]);
    // rest of g2 = the other 5 rows.
    assert_eq!(rest[2], vec![3.0 / 5.0, 2.0 / 5.0, 3.0 / 5.0, 1.0 / 5.0]);
}

#[test]
fn a_named_reference_yields_no_pts_rest_but_keeps_the_reference_column() {
    let counts = expected_counts();
    let f = counts.fractions(Some(1));
    assert!(f.pts_rest.is_none());
    // The reference group's own fraction is still reported (scanpy keeps the
    // reference as a `pts` column).
    assert_eq!(f.pts.len(), 3);
    assert_eq!(f.pts[1], vec![0.5, 0.5, 0.0, 0.0]);
}

#[test]
fn an_empty_group_is_nan_not_a_panic() {
    // Label 1 exists in the universe but no cell carries it: 0 / 0.
    let data = vec![1.0f32, 0.0, 2.0, 3.0];
    let got = group_nonzero_counts_dense(&data, 2, 2, &[0, 0], 2).unwrap();
    let f = got.fractions(None);
    assert!(f.pts[1].iter().all(|v| v.is_nan()));
    let rest = f.pts_rest.as_ref().unwrap();
    // Its rest is every other row, here both: gene 0 nonzero in 2 of 2, gene 1
    // in 1 of 2.
    assert_eq!(rest[1], vec![1.0, 0.5]);
    // A group that IS every row has an empty rest: NaN as well.
    assert!(rest[0].iter().all(|v| v.is_nan()));
}

#[test]
fn length_and_shape_mismatches_are_errors() {
    let (data, groups) = dense_fixture();
    assert!(group_nonzero_counts_dense(&data, 6, 4, &groups[..5], 3).is_err());
    assert!(group_nonzero_counts_dense(&data[..20], 6, 4, &groups, 3).is_err());
    assert!(group_nonzero_counts_csr(&csr_fixture(), &groups[..5], 3).is_err());
    // A block whose column count disagrees with the table.
    let mut acc = GroupNonzeroCounts::new(&groups, 3, 5);
    assert!(acc.add_csr(&csr_fixture(), 0, &groups).is_err());
    // A block that runs past the label vector.
    let mut acc = GroupNonzeroCounts::new(&groups, 3, 4);
    assert!(acc.add_csr(&csr_fixture(), 1, &groups).is_err());
    // A column index out of range inside the block.
    let bad = ScxCsr::new_unchecked((1, 4), vec![0, 1], vec![7], vec![1.0]);
    let mut acc = GroupNonzeroCounts::new(&[0], 1, 4);
    assert!(acc.add_csr(&bad, 0, &[0]).is_err());
}

// ── The `pts` pass under a row projection ───────────────────────────────
//
// `group_nonzero_counts_streaming` was the third hand-rolled
// `for shard_idx in 0..n_shards` in this module, and it is the one whose row
// cursor really indexes per-cell data (`groups[row_offset + row]`). Going
// through the shared driver means a shard the projection empties is not read;
// the cursor stays a running count of visible rows, because a skipped shard
// would have advanced it by zero.

/// A `GaugedSource` shard: `rows × 4` with a nonzero wherever `(global + c)`
/// is not a multiple of 3, so no two shards carry the same counts.
fn gauged_pts_shards() -> Vec<ScxCsr> {
    (0..4usize)
        .map(|s| {
            let mut data = vec![0.0f32; 3 * 4];
            for r in 0..3usize {
                let global = s * 3 + r;
                for c in 0..4 {
                    if !(global + c).is_multiple_of(3) {
                        data[r * 4 + c] = ((global * 5 + c) % 9 + 1) as f32;
                    }
                }
            }
            dense_to_csr(&data, 3, 4)
        })
        .collect()
}

#[test]
fn the_pts_pass_skips_the_shards_a_row_projection_empties() {
    use crate::test_support::GaugedSource;

    let plan = vec![0usize, 3];
    let src = GaugedSource::new(gauged_pts_shards(), 12, 4).with_plan(plan.clone());
    let groups: Vec<usize> = (0..src.n_obs()).map(|i| i % 2).collect();

    let streamed = group_nonzero_counts_streaming(&src, &groups, 2).unwrap();
    assert_eq!(src.decoded_shards(), plan, "read a shard the plan excluded");
    assert_eq!(src.decode_count(), plan.len(), "single pass, one read each");

    // And the counts are those rows' counts — the skip changed nothing. The
    // oracle is the CSR kernel on the same rows gathered into one matrix.
    let shards = gauged_pts_shards();
    let mut dense = Vec::new();
    for &i in &plan {
        let s = &shards[i];
        for r in 0..s.n_rows() {
            let mut row = vec![0.0f32; 4];
            for j in s.indptr[r] as usize..s.indptr[r + 1] as usize {
                row[s.indices[j] as usize] = s.data[j];
            }
            dense.extend_from_slice(&row);
        }
    }
    let oracle = group_nonzero_counts_csr(&dense_to_csr(&dense, 6, 4), &groups, 2).unwrap();
    assert_eq!(streamed, oracle);
}

#[test]
fn without_a_projection_the_pts_pass_reads_every_shard() {
    use crate::test_support::GaugedSource;

    let src = GaugedSource::new(gauged_pts_shards(), 12, 4);
    let groups: Vec<usize> = (0..12).map(|i| i % 2).collect();
    group_nonzero_counts_streaming(&src, &groups, 2).unwrap();
    assert_eq!(src.decoded_shards(), vec![0, 1, 2, 3]);
}
