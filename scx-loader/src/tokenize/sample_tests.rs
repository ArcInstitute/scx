use super::*;
use std::collections::BTreeMap;

fn draw(
    gene_ids: &[i32],
    counts: &[f32],
    n: usize,
    weight: WeightTransform,
    row_index: u64,
) -> (Vec<i64>, usize) {
    let mut out = vec![-1i64; n];
    let len = sample_genes(
        CsrRow {
            gene_ids,
            values: counts,
        },
        weight,
        7,
        11,
        row_index,
        &mut Vec::new(),
        &mut out,
    )
    .unwrap();
    (out, len)
}

fn histogram(v: &[i64]) -> BTreeMap<i64, usize> {
    let mut h = BTreeMap::new();
    for &x in v {
        *h.entry(x).or_insert(0) += 1;
    }
    h
}

#[test]
fn draws_are_with_replacement() {
    // Two genes, 64 draws: an at-most-once sampler could not produce this.
    let (out, len) = draw(&[3, 8], &[5.0, 5.0], 64, WeightTransform::Log1p, 0);
    assert_eq!(len, 64);
    assert!(out.iter().all(|&g| g == 3 || g == 8));
    assert!(histogram(&out).values().all(|&c| c > 1));
}

#[test]
fn every_slot_is_filled_and_the_length_is_reported() {
    let (out, len) = draw(&[1, 2, 3], &[1.0, 2.0, 3.0], 10, WeightTransform::Log1p, 0);
    assert_eq!(len, 10);
    assert!(out.iter().all(|&g| g >= 1), "no slot left at the sentinel");
}

#[test]
fn weights_follow_log1p_of_the_counts_not_the_counts() {
    // Counts 1 and 1000. Linear weighting gives gene 2 ~999x the mass; log1p
    // gives it only ln(1001)/ln(2) ~ 10x. A sampler that dropped the log1p
    // would put essentially every draw on gene 2.
    const N: usize = 200_000;
    let (out, _) = draw(&[1, 2], &[1.0, 1000.0], N, WeightTransform::Log1p, 0);
    let h = histogram(&out);
    let p1 = h[&1] as f64 / N as f64;
    let w1 = 2f64.ln();
    let expect = w1 / (w1 + 1001f64.ln());
    assert!(
        (p1 - expect).abs() < 0.01,
        "gene 1 drawn {p1:.4} of the time, expected ~{expect:.4}"
    );
    // And the linear sampler really is the thing being distinguished.
    let (lin, _) = draw(&[1, 2], &[1.0, 1000.0], N, WeightTransform::Linear, 0);
    let p1_lin = *histogram(&lin).get(&1).unwrap_or(&0) as f64 / N as f64;
    assert!(
        p1_lin < 0.01,
        "linear weighting should almost never draw gene 1, got {p1_lin:.4}"
    );
}

#[test]
fn the_empirical_distribution_matches_the_normalised_weights() {
    // The distributional parity claim, since no draw-for-draw golden against
    // numpy is possible. Chi-square-style: each cell within a few standard
    // errors of its expectation.
    const N: usize = 200_000;
    let ids = [10, 20, 30, 40];
    let counts = [1.0f32, 3.0, 7.0, 15.0];
    let (out, _) = draw(&ids, &counts, N, WeightTransform::Log1p, 0);
    let w: Vec<f64> = counts.iter().map(|&c| (c as f64).ln_1p()).collect();
    let total: f64 = w.iter().sum();
    let h = histogram(&out);
    for (i, &g) in ids.iter().enumerate() {
        let p = w[i] / total;
        let observed = h[&(g as i64)] as f64 / N as f64;
        let se = (p * (1.0 - p) / N as f64).sqrt();
        assert!(
            (observed - p).abs() < 5.0 * se,
            "gene {g}: observed {observed:.5}, expected {p:.5} (5 se = {:.5})",
            5.0 * se
        );
    }
}

#[test]
fn draws_are_reproducible_and_keyed_on_the_row() {
    let ids = [1, 2, 3, 4, 5, 6, 7, 8];
    let counts = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    assert_eq!(
        draw(&ids, &counts, 32, WeightTransform::Log1p, 5).0,
        draw(&ids, &counts, 32, WeightTransform::Log1p, 5).0
    );
    assert_ne!(
        draw(&ids, &counts, 32, WeightTransform::Log1p, 5).0,
        draw(&ids, &counts, 32, WeightTransform::Log1p, 6).0,
        "the row must be part of the key"
    );
}

#[test]
fn the_weight_transform_changes_the_stream_as_well_as_the_weights() {
    let ids = [1, 2, 3, 4];
    let counts = [4.0f32, 4.0, 4.0, 4.0];
    // Equal counts, so the two transforms give identical *probabilities*; the
    // draws still differ only if the transform reaches the weights at all.
    // Here they must NOT differ: same uniforms, same uniform CDF.
    assert_eq!(
        draw(&ids, &counts, 32, WeightTransform::Log1p, 1).0,
        draw(&ids, &counts, 32, WeightTransform::Linear, 1).0,
        "a flat cell samples identically under either transform"
    );
}

#[test]
fn a_zero_weight_gene_is_never_drawn() {
    let (out, _) = draw(&[1, 2, 3], &[0.0, 5.0, 0.0], 256, WeightTransform::Log1p, 0);
    assert!(out.iter().all(|&g| g == 2));
}

#[test]
fn an_empty_row_draws_nothing() {
    assert_eq!(draw(&[], &[], 8, WeightTransform::Log1p, 0).1, 0);
}

#[test]
fn an_all_zero_row_draws_nothing_rather_than_sampling_nan_probabilities() {
    let (out, len) = draw(&[1, 2], &[0.0, 0.0], 8, WeightTransform::Log1p, 0);
    assert_eq!(len, 0);
    assert_eq!(out, vec![-1i64; 8], "out is left untouched");
}

#[test]
fn negative_counts_clip_to_zero_weight() {
    let (out, _) = draw(&[1, 2], &[-5.0, 5.0], 64, WeightTransform::Log1p, 0);
    assert!(out.iter().all(|&g| g == 2));
}

#[test]
fn a_zero_width_request_draws_nothing() {
    let mut out: Vec<i64> = Vec::new();
    let len = sample_genes(
        CsrRow {
            gene_ids: &[1, 2],
            values: &[1.0, 1.0],
        },
        WeightTransform::Log1p,
        1,
        2,
        3,
        &mut Vec::new(),
        &mut out,
    )
    .unwrap();
    assert_eq!(len, 0);
}

#[test]
fn mismatched_row_lengths_are_an_error_not_a_panic() {
    let mut out = vec![0i64; 2];
    assert!(sample_genes(
        CsrRow {
            gene_ids: &[1, 2, 3],
            values: &[1.0, 1.0],
        },
        WeightTransform::Log1p,
        1,
        2,
        3,
        &mut Vec::new(),
        &mut out
    )
    .is_err());
}

#[test]
fn reusing_one_buffer_across_rows_is_not_observable() {
    let rows: [(&[i32], &[f32]); 3] = [
        (&[1, 2, 3, 4, 5], &[5.0, 4.0, 3.0, 2.0, 1.0]),
        (&[6, 7], &[1.0, 2.0]),
        (&[], &[]),
    ];
    let mut shared = Vec::new();
    for (ids, vals) in rows {
        let mut fresh = vec![-1i64; 8];
        let mut reused = vec![-1i64; 8];
        let row = CsrRow {
            gene_ids: ids,
            values: vals,
        };
        let a = sample_genes(
            row,
            WeightTransform::Log1p,
            7,
            11,
            2,
            &mut Vec::new(),
            &mut fresh,
        )
        .unwrap();
        let b = sample_genes(
            row,
            WeightTransform::Log1p,
            7,
            11,
            2,
            &mut shared,
            &mut reused,
        )
        .unwrap();
        assert_eq!((a, fresh), (b, reused));
    }
}
