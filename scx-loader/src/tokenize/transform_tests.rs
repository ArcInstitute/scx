use super::*;

fn run(f: impl Fn(&[f32], &mut [f32]), src: &[f32]) -> Vec<f32> {
    let mut dst = vec![0.0; src.len()];
    f(src, &mut dst);
    dst
}

fn approx(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-6 + 1e-5 * w.abs(),
            "position {i}: {g} vs {w}"
        );
    }
}

#[test]
fn library_size_clips_negatives_and_ignores_nan() {
    assert_eq!(library_size(&[1.0, 2.0, 3.0]), 6.0);
    assert_eq!(library_size(&[1.0, -5.0, 3.0]), 4.0);
    assert_eq!(library_size(&[1.0, f32::NAN, 3.0]), 4.0);
    assert_eq!(library_size(&[]), 0.0);
}

#[test]
fn library_size_is_exact_for_integer_counts_below_two_to_the_24() {
    // The claim the module doc makes about order-independence rests on this.
    let counts: Vec<f32> = (1..=2000).map(|i| i as f32).collect();
    assert_eq!(library_size(&counts), 2_001_000.0);
}

#[test]
fn pass_through_clips_negatives_and_nan_to_zero() {
    assert_eq!(
        run(pass_through, &[3.0, -1.0, 0.0, f32::NAN]),
        vec![3.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn log1p_raw_is_shifted_log_on_clipped_counts() {
    approx(
        &run(log1p_raw, &[0.0, 1.0, 3.0, -4.0]),
        &[0.0, 2f32.ln(), 4f32.ln(), 0.0],
    );
}

#[test]
fn normalize_log1p_scales_to_target_sum() {
    // lib = 4, target_sum = 8 ⇒ factor 2 ⇒ ln1p(2c).
    approx(
        &run(|s, d| normalize_log1p(s, d, 8.0, 4.0), &[1.0, 3.0]),
        &[3f32.ln(), 7f32.ln()],
    );
}

#[test]
fn normalize_log1p_leaves_values_unscaled_when_the_library_is_empty() {
    // Not a divide by zero, and not all-zero output: an empty cell keeps its
    // (clipped) values and takes the plain shifted log.
    approx(
        &run(|s, d| normalize_log1p(s, d, 1e4, 0.0), &[1.0, 3.0]),
        &[2f32.ln(), 4f32.ln()],
    );
}

#[test]
fn pflog_raw_centres_by_the_declared_panel_size() {
    // alpha = 0.25 ⇒ 4a = 1 ⇒ terms are ln1p(c); centre = sum / n_measured.
    let src = [0.0f32, 1.0, 3.0];
    let terms = [0.0f64, 2f64.ln(), 4f64.ln()];
    let centre = (terms.iter().sum::<f64>() / 3.0) as f32;
    approx(
        &run(|s, d| pflog_raw(s, d, 0.25, 3), &src),
        &terms.map(|t| t as f32 - centre),
    );
}

#[test]
fn pflog_raw_centre_denominator_is_n_measured_not_the_row_length() {
    // The two differ whenever the row is sparser than the measured panel, which
    // is always: `n_measured` is the panel, the row is its non-zeros.
    let src = [1.0f32, 3.0];
    let by_panel = run(|s, d| pflog_raw(s, d, 0.25, 100), &src);
    let by_row = run(|s, d| pflog_raw(s, d, 0.25, src.len()), &src);
    assert_ne!(by_panel, by_row);
}

#[test]
fn pflog_raw_on_an_all_zero_row_stays_all_zero() {
    // Every term 0 ⇒ centre 0 ⇒ output 0. There is no per-cell depth term that
    // could turn an empty cell into a non-zero offset.
    assert_eq!(
        run(|s, d| pflog_raw(s, d, 0.25, 8), &[0.0, 0.0]),
        vec![0.0, 0.0]
    );
}

#[test]
fn measured_mask_marks_panel_positions_the_row_carries() {
    let mut out = vec![9u8; 4];
    measured_mask(&[2, 5, 9], &[5, 1, 9, 2], &mut out);
    assert_eq!(out, vec![1, 0, 1, 1]);
}

#[test]
fn measured_mask_answers_each_repeated_panel_position_independently() {
    let mut out = vec![0u8; 3];
    measured_mask(&[7], &[7, 7, 8], &mut out);
    assert_eq!(out, vec![1, 1, 0]);
}

#[test]
fn measured_mask_is_measured_not_nonzero() {
    // Gene 5 is carried with a zero value: present in the panel sense, even
    // though nothing about its count distinguishes it from an absent gene.
    let mut out = vec![0u8; 2];
    measured_mask(&[5, 6], &[5, 4], &mut out);
    assert_eq!(out, vec![1, 0]);
}

#[test]
fn every_transform_handles_an_empty_row() {
    for f in [
        &pass_through as &dyn Fn(&[f32], &mut [f32]),
        &log1p_raw,
        &|s: &[f32], d: &mut [f32]| normalize_log1p(s, d, 1e4, 0.0),
    ] {
        assert!(run(f, &[]).is_empty());
    }
    // pflog on an empty row: the centre is 0/n, which is finite for n > 0.
    assert!(run(|s, d| pflog_raw(s, d, 0.25, 8), &[]).is_empty());
}
