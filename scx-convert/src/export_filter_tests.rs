//! Unit tests for the `min_counts` pre-pass.
//!
//! The load-bearing property here is the coordinate system: the returned mask
//! must be **global-length** (header `n_obs`, counting deleted rows) so it can
//! be ANDed with the deletion-vector mask. A visible-space vector would be
//! silently off-by-N on any file with deletions.

use super::*;
use crate::convert_tests_common::{create_test_h5ad, h5ad_to_scx, IngestOptions, WarningSink};

/// Row sums as the exporter itself would compute them, straight from the
/// backed reader — an independent oracle for the mask.
fn oracle_row_sums(scx_path: &Path) -> Vec<f64> {
    let reader = ScxReader::open(scx_path).unwrap();
    BackedCsrReader::new(reader, 2).row_sums().unwrap()
}

fn build_fixture(dir: &std::path::Path, n_obs: usize, n_vars: usize) -> std::path::PathBuf {
    let h5ad = dir.join("in.h5ad");
    let scx = dir.join("in.scx");
    create_test_h5ad(&h5ad, n_obs, n_vars, "csr", false);
    h5ad_to_scx(
        &h5ad,
        &scx,
        &IngestOptions::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    scx
}

#[test]
fn mask_matches_row_sums_with_inclusive_boundary() {
    let dir = tempfile::tempdir().unwrap();
    let scx = build_fixture(dir.path(), 20, 15);
    let sums = oracle_row_sums(&scx);

    // Pick an actually-present row sum so the `>=` boundary is exercised
    // rather than assumed.
    let threshold = sums[3];
    let mask = min_counts_obs_mask(&scx, 0, threshold).unwrap();

    assert_eq!(mask.len(), sums.len());
    for (i, (&keep, &sum)) in mask.iter().zip(sums.iter()).enumerate() {
        assert_eq!(
            keep,
            sum >= threshold,
            "row {i}: sum {sum}, thr {threshold}"
        );
    }
    assert!(mask[3], "a row exactly at the threshold must be kept (>=)");
}

#[test]
fn zero_threshold_keeps_every_row() {
    let dir = tempfile::tempdir().unwrap();
    let scx = build_fixture(dir.path(), 12, 8);
    let mask = min_counts_obs_mask(&scx, 0, 0.0).unwrap();
    assert_eq!(mask.len(), 12);
    assert!(mask.iter().all(|&b| b));
}

/// The whole feature rests on this: the mask indexes the physical obs row
/// space, so it stays full-length even when rows are logically deleted.
#[test]
fn mask_is_global_length_on_a_file_with_deletions() {
    let dir = tempfile::tempdir().unwrap();
    let scx = build_fixture(dir.path(), 20, 15);

    scx_ops::mark_deleted(&scx, &[0, 1, 2, 3, 4]).unwrap();

    let reader = ScxReader::open(&scx).unwrap();
    assert_eq!(reader.n_obs(), 20, "header n_obs is unchanged by a delete");
    assert_eq!(
        reader.deletion_keep_mask().unwrap().unwrap().len(),
        20,
        "the DV mask is global-length"
    );

    let mask = min_counts_obs_mask(&scx, 0, 1.0).unwrap();
    assert_eq!(
        mask.len(),
        20,
        "min_counts mask must be global-length so it ANDs with the DV mask"
    );
}

#[test]
fn negative_or_nan_threshold_errors() {
    let dir = tempfile::tempdir().unwrap();
    let scx = build_fixture(dir.path(), 6, 4);
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        let err = min_counts_obs_mask(&scx, 0, bad).unwrap_err();
        assert!(
            err.to_string().contains("min_counts"),
            "unhelpful error for {bad}: {err}"
        );
    }
}
