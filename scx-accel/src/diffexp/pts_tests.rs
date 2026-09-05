//! `GroupNonzeroCounts` — dense, CSR and streamed inputs agree; the counting
//! rule is scanpy's (`!= 0`, explicit zeros ignored, negatives counted,
//! unlabelled cells in nothing); the fractions divide by the right pools.

use super::*;
use crate::hvg::cpu::InMemorySource;

const OOR: usize = 3;

/// 6 cells × 4 genes; labels `[0, 0, 1, 1, 2, unlabelled]`.
///
/// Row 5 (unlabelled) is dense with nonzeros so any leak into a count shows.
/// Gene 3 is zero in every labelled cell so its `pts` is 0 everywhere and its
/// `pts_rest` denominator still counts the cells.
fn dense_fixture() -> (Vec<f32>, Vec<usize>) {
    #[rustfmt::skip]
    let data: Vec<f32> = vec![
        1.0, 0.0, 2.0, 0.0,   // g0
        0.0, 0.0, 3.0, 0.0,   // g0
        -1.0, 5.0, 0.0, 0.0,  // g1  (negative counts as expressing)
        0.0, 0.0, 0.0, 0.0,   // g1
        4.0, 4.0, 4.0, 0.0,   // g2
        9.0, 9.0, 9.0, 9.0,   // unlabelled: must not count anywhere
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
        labelled_total: vec![3, 2, 3, 0],
        group_sizes: vec![2, 2, 1],
        n_labelled: 5,
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
    assert_eq!(got.labelled_total, vec![0, 1]);
}

#[test]
fn fractions_divide_by_group_size_and_labelled_rest() {
    let counts = expected_counts();
    let f = counts.fractions(None);
    // g0 has 2 cells: gene 0 in 1 of 2, gene 2 in 2 of 2.
    assert_eq!(f.pts[0], vec![0.5, 0.0, 1.0, 0.0]);
    // g2 has 1 cell.
    assert_eq!(f.pts[2], vec![1.0, 1.0, 1.0, 0.0]);
    // rest of g0 = 3 labelled cells (rows 2, 3, 4): gene 0 in 2 of them,
    // gene 1 in 2, gene 2 in 1, gene 3 in 0. The unlabelled row 5 (all
    // nonzero) is in neither numerator nor denominator.
    let rest = f.pts_rest.as_ref().unwrap();
    assert_eq!(rest[0], vec![2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 0.0]);
    // rest of g2 = 4 labelled cells.
    assert_eq!(rest[2], vec![2.0 / 4.0, 1.0 / 4.0, 2.0 / 4.0, 0.0]);
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
    // Its rest is every labelled cell: gene 0 nonzero in 2 of 2, gene 1 in 1 of 2.
    assert_eq!(rest[1], vec![1.0, 0.5]);
    // A group that IS every labelled cell has an empty rest: NaN as well.
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
