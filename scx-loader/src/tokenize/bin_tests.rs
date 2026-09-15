use super::*;

fn run(vals: &[f32], n_bins: usize, tie: BinTie) -> Vec<i64> {
    run_at(vals, n_bins, tie, 0)
}

fn run_at(vals: &[f32], n_bins: usize, tie: BinTie, row: u64) -> Vec<i64> {
    let mut out = vec![-9i64; vals.len()];
    bin_values(
        vals,
        BinEdges::PerCellQuantile,
        n_bins,
        tie,
        row,
        &mut Vec::new(),
        &mut out,
    )
    .unwrap();
    out
}

fn edges_of(vals: &[f32], n_bins: usize) -> Vec<f64> {
    let mut sorted: Vec<f64> = vals
        .iter()
        .filter(|v| **v > 0.0)
        .map(|&v| v as f64)
        .collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Through the SAME helper the kernel uses, so a change to the interpolation
    // cannot pass here and fail there.
    let m = sorted.len();
    push_quantile_edges(&mut sorted, m, n_bins - 1);
    sorted.split_off(m)
}

// Reference vectors produced by numpy 2.4.4:
//   nz = a[a.nonzero()]; bins = np.quantile(nz, np.linspace(0, 1, n_bins - 1))
//   np.digitize(nz, bins) / np.digitize(nz, bins, right=True)
// See `benchmarks/scripts/gen_tokenize_goldens.py` for the committed goldens;
// these three cases are inlined so the kernel's own unit tests do not depend on
// a fixture file.

#[test]
fn quantile_edges_match_numpy_linear_interpolation() {
    assert_eq!(
        edges_of(&[1.0, 2.0, 5.0, 11.0], 6),
        vec![1.0, 1.75, 3.5, 6.5, 11.0]
    );
    assert_eq!(
        edges_of(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], 5),
        vec![1.0, 2.333333333333333, 4.666666666666666, 9.0]
    );
}

#[test]
fn edges_widen_f32_inputs_to_f64_the_way_numpy_does() {
    // np.quantile on a float32 array RETURNS float64 (0.3f32 is exactly the
    // 0.30000001 numpy printed, which is why the expected edges are unchanged), so the interpolation
    // happens on the widened values. Computing in f32 gives different edges.
    let got = edges_of(&[0.1, 0.2, 0.3, 0.7], 6);
    assert_eq!(
        got,
        vec![
            0.10000000149011612,
            0.1750000026077032,
            0.2500000074505806,
            0.4000000059604645,
            0.699999988079071
        ]
    );
    // Anti-tautology: the same interpolation carried out in f32 disagrees, so
    // the assertion above is pinning the width, not just the arithmetic.
    let in_f32 = 0.1f32 + 0.5f32 * (0.2f32 - 0.1f32);
    assert_ne!(in_f32 as f64, got[1]);
}

#[test]
fn left_and_right_digitize_match_numpy() {
    let vals = [1.0f32, 2.0, 5.0, 11.0];
    assert_eq!(run(&vals, 6, BinTie::Left), vec![1, 2, 3, 5]);
    assert_eq!(run(&vals, 6, BinTie::Right), vec![0, 2, 3, 4]);

    let vals = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    assert_eq!(run(&vals, 5, BinTie::Left), vec![2, 1, 2, 1, 3, 4, 1, 3]);
    assert_eq!(run(&vals, 5, BinTie::Right), vec![2, 0, 2, 0, 3, 3, 1, 3]);
}

#[test]
fn a_row_whose_nonzeros_are_all_equal_collapses_every_edge() {
    // The reference pathology, pinned rather than hidden: every quantile edge
    // becomes the same value, so left = n_bins - 1 and right = 0 and scGPT
    // assigns a uniformly random bin over the whole range.
    let vals = [4.0f32, 4.0, 4.0];
    assert_eq!(edges_of(&vals, 6), vec![4.0; 5]);
    assert_eq!(run(&vals, 6, BinTie::Left), vec![5, 5, 5]);
    assert_eq!(run(&vals, 6, BinTie::Right), vec![0, 0, 0]);
}

#[test]
fn the_seeded_randomisation_stays_within_the_two_deterministic_bounds() {
    // The strongest claim available about the reference: every draw it can
    // produce — and every draw this kernel produces — lies in [right, left].
    // `ceil` makes the reachable set [right + 1, left] whenever the bounds
    // differ, which is why the reference never emits bin 0 for a non-zero value.
    let vals = [4.0f32, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0];
    let left = run(&vals, 6, BinTie::Left);
    let right = run(&vals, 6, BinTie::Right);
    let mut seen = std::collections::BTreeSet::new();
    for row in 0..200u64 {
        let got = run_at(
            &vals,
            6,
            BinTie::SeededUniform {
                seed: 7,
                file_identity: 11,
            },
            row,
        );
        for i in 0..vals.len() {
            assert!(
                got[i] >= right[i] && got[i] <= left[i],
                "row {row} position {i}: {} outside [{}, {}]",
                got[i],
                right[i],
                left[i]
            );
            seen.insert(got[i]);
        }
    }
    // And the bracket is not vacuous: the draws actually spread over it.
    assert_eq!(
        seen.into_iter().collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5],
        "the seeded tiebreak should reach every bin the reference can"
    );
}

#[test]
fn a_value_strictly_inside_a_bin_has_no_tiebreak_to_make() {
    // left == right there, so all three tie rules agree — the randomisation only
    // ever fires on an edge.
    let vals = [1.0f32, 2.0, 5.0, 11.0];
    let l = run(&vals, 6, BinTie::Left);
    let seeded = run_at(
        &vals,
        6,
        BinTie::SeededUniform {
            seed: 1,
            file_identity: 2,
        },
        3,
    );
    // Position 1 (value 2.0) sits strictly between edges 1.75 and 3.5.
    assert_eq!(l[1], seeded[1]);
}

#[test]
fn zeros_stay_in_bin_zero() {
    let out = run(&[0.0, 5.0, 0.0, 1.0], 5, BinTie::Left);
    assert_eq!(out[0], 0);
    assert_eq!(out[2], 0);
    assert!(out[1] > 0 && out[3] > 0);
}

#[test]
fn negatives_and_nan_clip_to_zero_and_land_in_bin_zero() {
    // Declared divergence: scGPT's `nonzero()` would have binned a negative.
    let out = run(&[-3.0, f32::NAN, 5.0, 1.0], 5, BinTie::Left);
    assert_eq!(out[0], 0);
    assert_eq!(out[1], 0);
}

#[test]
fn an_all_zero_row_is_all_bin_zero() {
    assert_eq!(run(&[0.0, 0.0, 0.0], 6, BinTie::Left), vec![0, 0, 0]);
}

#[test]
fn an_empty_row_is_accepted() {
    assert_eq!(run(&[], 6, BinTie::Left), Vec::<i64>::new());
}

#[test]
fn a_single_nonzero_value_is_the_degenerate_case_too() {
    assert_eq!(run(&[7.0], 6, BinTie::Left), vec![5]);
    assert_eq!(run(&[7.0], 6, BinTie::Right), vec![0]);
}

#[test]
fn seeded_draws_are_reproducible_and_keyed_on_the_row() {
    let vals = [4.0f32; 16];
    let k = BinTie::SeededUniform {
        seed: 42,
        file_identity: 99,
    };
    assert_eq!(run_at(&vals, 8, k, 5), run_at(&vals, 8, k, 5));
    assert_ne!(run_at(&vals, 8, k, 5), run_at(&vals, 8, k, 6));
    assert_ne!(
        run_at(&vals, 8, k, 5),
        run_at(
            &vals,
            8,
            BinTie::SeededUniform {
                seed: 42,
                file_identity: 100
            },
            5
        )
    );
    assert_ne!(
        run_at(&vals, 8, k, 5),
        run_at(
            &vals,
            8,
            BinTie::SeededUniform {
                seed: 43,
                file_identity: 99
            },
            5
        )
    );
}

#[test]
fn fixed_edges_are_used_as_given() {
    let mut out = vec![0i64; 4];
    bin_values(
        &[0.5, 1.5, 2.5, 3.5],
        BinEdges::Fixed(&[1.0, 2.0, 3.0]),
        4,
        BinTie::Left,
        0,
        &mut Vec::new(),
        &mut out,
    )
    .unwrap();
    assert_eq!(out, vec![0, 1, 2, 3]);
}

#[test]
fn fixed_edges_must_be_finite_and_non_decreasing() {
    let mut out = vec![0i64; 1];
    for bad in [vec![2.0, 1.0], vec![1.0, f64::NAN]] {
        assert!(
            bin_values(
                &[1.0],
                BinEdges::Fixed(&bad),
                3,
                BinTie::Left,
                0,
                &mut Vec::new(),
                &mut out
            )
            .is_err(),
            "{bad:?} was accepted"
        );
    }
    // Edge count must match `n_bins`, or the emitted bins and the declared
    // range disagree: `edges=[1,2,3]` with `n_bins=3` emitted bin 3.
    assert!(bin_values(
        &[1.0],
        BinEdges::Fixed(&[1.0, 2.0, 3.0]),
        3,
        BinTie::Left,
        0,
        &mut Vec::new(),
        &mut out
    )
    .is_err());
    // ...and it is checked BEFORE the all-zero early return, or a batch with
    // nothing to bin accepts any edges at all.
    let mut zeros = vec![0i64; 2];
    assert!(bin_values(
        &[0.0, 0.0],
        BinEdges::Fixed(&[3.0, 1.0]),
        3,
        BinTie::Left,
        0,
        &mut Vec::new(),
        &mut zeros
    )
    .is_err());
}

#[test]
fn shape_and_parameter_errors_are_returned_not_panicked() {
    let mut short = vec![0i64; 1];
    assert!(bin_values(
        &[1.0, 2.0],
        BinEdges::PerCellQuantile,
        5,
        BinTie::Left,
        0,
        &mut Vec::new(),
        &mut short
    )
    .is_err());
    let mut out = vec![0i64; 2];
    for n_bins in [0, 1, 2] {
        assert!(
            bin_values(
                &[1.0, 2.0],
                BinEdges::PerCellQuantile,
                n_bins,
                BinTie::Left,
                0,
                &mut Vec::new(),
                &mut out
            )
            .is_err(),
            "n_bins={n_bins} was accepted"
        );
    }
}

#[test]
fn reusing_one_buffer_across_rows_is_not_observable() {
    let rows: [&[f32]; 3] = [&[1.0, 2.0, 5.0, 11.0], &[3.0, 1.0], &[]];
    let mut shared = Vec::new();
    for vals in rows {
        let mut fresh = vec![0i64; vals.len()];
        let mut reused = vec![0i64; vals.len()];
        bin_values(
            vals,
            BinEdges::PerCellQuantile,
            6,
            BinTie::Left,
            0,
            &mut Vec::new(),
            &mut fresh,
        )
        .unwrap();
        bin_values(
            vals,
            BinEdges::PerCellQuantile,
            6,
            BinTie::Left,
            0,
            &mut shared,
            &mut reused,
        )
        .unwrap();
        assert_eq!(fresh, reused);
    }
}
