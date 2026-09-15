use super::*;

fn norm(stat: &[f32]) -> PerGeneNorm {
    PerGeneNorm::new(stat.to_vec().into(), "test-v1").unwrap()
}

fn rank(gene_ids: &[i32], counts: &[f32], n: &PerGeneNorm, l_max: usize) -> (Vec<i64>, usize) {
    let mut out = vec![-1i64; l_max];
    let len = rank_tokens(
        CsrRow {
            gene_ids,
            values: counts,
        },
        n,
        1e4,
        &mut Vec::new(),
        &mut Vec::new(),
        &mut out,
    )
    .unwrap();
    (out, len)
}

#[test]
fn ranks_by_normalised_value_descending() {
    let n = norm(&[1.0; 6]);
    let (out, len) = rank(&[1, 3, 5], &[2.0, 9.0, 4.0], &n, 3);
    assert_eq!(len, 3);
    assert_eq!(out, vec![3, 5, 1]);
}

#[test]
fn the_per_gene_statistic_changes_the_order() {
    // Gene 1 has the smaller count but a far smaller corpus median, so it is the
    // more enriched gene and must rank first. With the divide removed, gene 3
    // wins on raw count and this fails — which is the point: the divide is the
    // whole difference between "rank by expression" and "rank by enrichment".
    let n = norm(&[1.0, 0.01, 1.0, 10.0]);
    let (out, _) = rank(&[1, 3], &[2.0, 9.0], &n, 2);
    assert_eq!(out, vec![1, 3]);
}

#[test]
fn ties_break_by_gene_id_ascending() {
    // The declared rule. Geneformer's own order here is quicksort-arbitrary, so
    // this pins SCX's choice, not agreement with the reference.
    let n = norm(&[1.0; 10]);
    let (out, _) = rank(&[7, 2, 9, 4], &[3.0, 3.0, 3.0, 3.0], &n, 4);
    assert_eq!(out, vec![2, 4, 7, 9]);
}

#[test]
fn equal_normalised_values_tie_even_when_the_counts_differ() {
    // count/stat equal for both: 2/1 and 8/4. The tiebreak must fire on the
    // NORMALISED value, not on the raw count.
    let n = norm(&[1.0, 1.0, 1.0, 1.0, 4.0]);
    let (out, _) = rank(&[4, 1], &[8.0, 2.0], &n, 2);
    assert_eq!(out, vec![1, 4]);
}

#[test]
fn truncates_to_the_output_buffer_and_reports_the_length() {
    let n = norm(&[1.0; 6]);
    let (out, len) = rank(&[1, 2, 3, 4], &[4.0, 3.0, 2.0, 1.0], &n, 2);
    assert_eq!(len, 2);
    assert_eq!(out, vec![1, 2]);
}

#[test]
fn slots_beyond_the_reported_length_are_left_untouched() {
    // No PAD fill: this kernel reports a length and the consumer owns padding.
    let n = norm(&[1.0; 6]);
    let (out, len) = rank(&[1, 2], &[4.0, 3.0], &n, 5);
    assert_eq!(len, 2);
    assert_eq!(out, vec![1, 2, -1, -1, -1]);
}

#[test]
fn non_positive_counts_are_dropped_before_ranking() {
    let n = norm(&[1.0; 6]);
    let (out, len) = rank(&[1, 2, 3], &[0.0, 5.0, -2.0], &n, 3);
    assert_eq!(len, 1);
    assert_eq!(out[0], 2);
}

#[test]
fn an_empty_row_ranks_nothing() {
    let n = norm(&[1.0; 6]);
    assert_eq!(rank(&[], &[], &n, 3).1, 0);
}

#[test]
fn an_all_zero_row_ranks_nothing_rather_than_dividing_by_zero() {
    // The reference divides by `n_counts` and would produce NaN here.
    let n = norm(&[1.0; 6]);
    assert_eq!(rank(&[1, 2], &[0.0, 0.0], &n, 3).1, 0);
}

#[test]
fn per_cell_depth_scaling_does_not_change_the_order() {
    // Stated so nobody claims the depth divide is pinned by a rank test: it is a
    // monotone scaling of the whole row, so it cannot reorder anything (up to
    // f32 rounding of the normalised values, which these inputs stay clear of).
    // The step is kept because `target_sum` and the divide are part of the
    // reference's declared identity, not because this kernel's output needs it.
    let n = norm(&[1.0, 0.5, 2.0, 1.0]);
    let (a, _) = rank(&[1, 2, 3], &[2.0, 8.0, 4.0], &n, 3);
    let (b, _) = rank(&[1, 2, 3], &[200.0, 800.0, 400.0], &n, 3);
    assert_eq!(a, b);
}

#[test]
fn a_gene_id_outside_the_vocabulary_is_an_error_not_a_panic() {
    let n = norm(&[1.0; 3]);
    let mut out = vec![0i64; 2];
    let err = rank_tokens(
        CsrRow {
            gene_ids: &[1, 7],
            values: &[1.0, 1.0],
        },
        &n,
        1e4,
        &mut Vec::new(),
        &mut Vec::new(),
        &mut out,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("outside the normalisation vocabulary"),
        "unexpected error: {err}"
    );
}

#[test]
fn a_statistics_vector_with_a_non_positive_or_non_finite_entry_is_refused() {
    for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
        let err = PerGeneNorm::new(vec![1.0, bad, 1.0].into(), "v").unwrap_err();
        assert!(
            err.to_string().contains("finite and strictly positive"),
            "{bad} was accepted: {err}"
        );
    }
}

#[test]
fn an_empty_statistics_vector_is_refused() {
    assert!(PerGeneNorm::new(Vec::new().into(), "v").is_err());
}

#[test]
fn identity_covers_both_the_statistics_and_the_vocabulary_version() {
    let a = norm(&[1.0, 2.0, 3.0]);
    let b = norm(&[1.0, 2.0, 3.0]);
    let c = norm(&[1.0, 2.0, 3.5]);
    let d = PerGeneNorm::new(vec![1.0, 2.0, 3.0].into(), "test-v2").unwrap();
    assert_eq!(a.identity(), b.identity(), "same inputs, same identity");
    assert_ne!(a.identity(), c.identity(), "statistics ignored");
    assert_ne!(a.identity(), d.identity(), "vocabulary version ignored");
    assert_eq!(a.vocabulary_version(), "test-v1");
    assert_eq!(a.len(), 3);
}

#[test]
fn reusing_scratch_across_rows_is_not_observable() {
    let n = norm(&[1.0; 8]);
    let rows: [(&[i32], &[f32]); 3] = [
        (&[1, 2, 3, 4, 5], &[5.0, 4.0, 3.0, 2.0, 1.0]),
        (&[6, 7], &[1.0, 2.0]),
        (&[], &[]),
    ];
    let (mut order, mut values) = (Vec::new(), Vec::new());
    for (ids, vals) in rows {
        let mut fresh = vec![-1i64; 4];
        let mut reused = vec![-1i64; 4];
        let a = rank_tokens(
            CsrRow {
                gene_ids: ids,
                values: vals,
            },
            &n,
            1e4,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut fresh,
        )
        .unwrap();
        let b = rank_tokens(
            CsrRow {
                gene_ids: ids,
                values: vals,
            },
            &n,
            1e4,
            &mut order,
            &mut values,
            &mut reused,
        )
        .unwrap();
        assert_eq!((a, fresh), (b, reused));
    }
}
