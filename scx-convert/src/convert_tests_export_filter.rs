//! Integration tests for the caller-supplied export row filter
//! (`ConvertOptions::export_obs_keep_mask` / `export_min_counts`).
//!
//! The invariants worth pinning, in rough order of how badly a regression
//! would hurt:
//!
//! 1. A caller mask never resurrects a logically deleted row — the two masks
//!    intersect, they do not substitute.
//! 2. Every obs-axis section (`/X`, `obs`, `obsm`, `/layers/*`) agrees on the
//!    kept row count. A disagreement produces an h5ad anndata refuses to open.
//! 3. With no filter set, the output is unchanged — including the absence of
//!    a `/uns/scx_export` group.

use super::convert_tests_common::*;
use super::h5ad::stream_write::EXPORT_PROVENANCE_KEY;
use super::pipeline::scx_to_h5ad_streaming;
use std::sync::Arc;

fn build_scx(dir: &Path, n_obs: usize, n_vars: usize, extras: bool) -> std::path::PathBuf {
    let h5ad = dir.join("in.h5ad");
    let scx = dir.join("in.scx");
    create_test_h5ad(&h5ad, n_obs, n_vars, "csr", extras);
    h5ad_to_scx(
        &h5ad,
        &scx,
        &ConvertOptions::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    scx
}

fn export(scx: &Path, out: &Path, opts: &ConvertOptions) -> Result<(), ConvertError> {
    scx_to_h5ad_streaming(scx, out, opts, &mut WarningSink::log())
}

fn mask_opts(mask: Vec<bool>) -> ConvertOptions {
    ConvertOptions {
        export_obs_keep_mask: Some(Arc::from(mask)),
        ..Default::default()
    }
}

/// Rows in `/X`, read from the pre-allocated indptr rather than the shape
/// attribute so a mismatch between the two would still be caught.
fn x_rows(file: &hdf5::File) -> usize {
    let indptr: Vec<i64> = file
        .dataset("X/indptr")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    indptr.len() - 1
}

fn x_shape_attr(file: &hdf5::File) -> Vec<i64> {
    file.group("X")
        .unwrap()
        .attr("shape")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec()
}

fn dataset_len(file: &hdf5::File, path: &str) -> usize {
    file.dataset(path).unwrap().shape()[0]
}

fn x_values(file: &hdf5::File) -> Vec<f32> {
    file.dataset("X/data").unwrap().read_1d().unwrap().to_vec()
}

// ---------------------------------------------------------------------------
// Mask semantics
// ---------------------------------------------------------------------------

#[test]
fn mask_filters_x_obs_obsm_and_layers_consistently() {
    let dir = tempfile::tempdir().unwrap();
    let (n_obs, n_vars) = (20usize, 15usize);
    let scx = build_scx(dir.path(), n_obs, n_vars, true);
    let out = dir.path().join("out.h5ad");

    // Keep even rows.
    let mask: Vec<bool> = (0..n_obs).map(|i| i % 2 == 0).collect();
    let kept = mask.iter().filter(|&&b| b).count();
    export(&scx, &out, &mask_opts(mask)).unwrap();

    let f = hdf5::File::open(&out).unwrap();
    assert_eq!(x_rows(&f), kept, "/X row count");
    assert_eq!(x_shape_attr(&f), vec![kept as i64, n_vars as i64]);
    assert_eq!(dataset_len(&f, "obs/_index"), kept, "obs row count");
    assert_eq!(
        f.dataset("obsm/X_pca").unwrap().shape()[0],
        kept,
        "obsm rows must track /X or anndata refuses to open the file"
    );
    assert_eq!(
        dataset_len(&f, "layers/raw/indptr"),
        kept + 1,
        "layer row count"
    );
}

/// The exported values must be the source's even rows, not merely the right
/// *count* of rows — a row-count-only assertion passes on an off-by-one gather.
#[test]
fn mask_selects_the_right_rows_not_just_the_right_count() {
    let dir = tempfile::tempdir().unwrap();
    let (n_obs, n_vars) = (12usize, 10usize);
    let scx = build_scx(dir.path(), n_obs, n_vars, false);
    let unfiltered = dir.path().join("all.h5ad");
    let filtered = dir.path().join("even.h5ad");

    export(&scx, &unfiltered, &ConvertOptions::default()).unwrap();

    let mask: Vec<bool> = (0..n_obs).map(|i| i % 2 == 0).collect();
    export(&scx, &filtered, &mask_opts(mask.clone())).unwrap();

    let all = hdf5::File::open(&unfiltered).unwrap();
    let all_indptr: Vec<i64> = all.dataset("X/indptr").unwrap().read_1d().unwrap().to_vec();
    let all_data = x_values(&all);

    let mut expected = Vec::new();
    for (row, keep) in mask.iter().enumerate() {
        if *keep {
            let (s, e) = (all_indptr[row] as usize, all_indptr[row + 1] as usize);
            expected.extend_from_slice(&all_data[s..e]);
        }
    }

    let got = hdf5::File::open(&filtered).unwrap();
    assert_eq!(x_values(&got), expected);
}

/// The core safety property: the deletion vector wins. A caller mask marking a
/// deleted row `true` must not bring it back.
#[test]
fn mask_intersects_deletion_vector_and_never_resurrects() {
    let dir = tempfile::tempdir().unwrap();
    let (n_obs, n_vars) = (20usize, 15usize);
    let scx = build_scx(dir.path(), n_obs, n_vars, false);

    // Baseline captured before the delete, so the expected values below come
    // from the same file rather than a re-generated one.
    let all_out = dir.path().join("all.h5ad");
    export(&scx, &all_out, &ConvertOptions::default()).unwrap();

    scx_ops::mark_deleted(&scx, &[0, 1]).unwrap();

    // All-true except row 2 — rows 0 and 1 are `true` here but deleted.
    let mut mask = vec![true; n_obs];
    mask[2] = false;

    let out = dir.path().join("out.h5ad");
    export(&scx, &out, &mask_opts(mask)).unwrap();

    let f = hdf5::File::open(&out).unwrap();
    assert_eq!(
        x_rows(&f),
        n_obs - 3,
        "expected 2 deleted + 1 masked-out rows dropped"
    );

    let all = hdf5::File::open(&all_out).unwrap();
    let all_indptr: Vec<i64> = all.dataset("X/indptr").unwrap().read_1d().unwrap().to_vec();
    let all_data = x_values(&all);

    let mut expected = Vec::new();
    for row in 3..n_obs {
        let (s, e) = (all_indptr[row] as usize, all_indptr[row + 1] as usize);
        expected.extend_from_slice(&all_data[s..e]);
    }
    assert_eq!(x_values(&f), expected);
}

#[test]
fn wrong_length_mask_errors_in_both_directions() {
    let dir = tempfile::tempdir().unwrap();
    let n_obs = 20usize;
    let scx = build_scx(dir.path(), n_obs, 15, false);
    let out = dir.path().join("out.h5ad");

    for len in [n_obs - 1, n_obs + 1] {
        let err = export(&scx, &out, &mask_opts(vec![true; len])).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(&len.to_string()), "{msg}");
        assert!(msg.contains(&n_obs.to_string()), "{msg}");
        assert!(msg.contains("global"), "message must name the space: {msg}");
    }
}

#[test]
fn all_false_mask_errors_rather_than_writing_an_empty_h5ad() {
    let dir = tempfile::tempdir().unwrap();
    let scx = build_scx(dir.path(), 10, 8, false);
    let out = dir.path().join("out.h5ad");
    let err = export(&scx, &out, &mask_opts(vec![false; 10])).unwrap_err();
    assert!(err.to_string().contains("keeps zero"), "{err}");
}

/// A deletion vector covering every row keeps its historical (no-error)
/// behaviour — the zero-row guard is scoped to *caller* filters.
#[test]
fn all_deleted_without_a_caller_filter_still_exports() {
    let dir = tempfile::tempdir().unwrap();
    let n_obs = 6usize;
    let scx = build_scx(dir.path(), n_obs, 4, false);
    let all: Vec<u64> = (0..n_obs as u64).collect();
    scx_ops::mark_deleted(&scx, &all).unwrap();

    let out = dir.path().join("out.h5ad");
    export(&scx, &out, &ConvertOptions::default()).unwrap();
    assert_eq!(x_rows(&hdf5::File::open(&out).unwrap()), 0);
}

// ---------------------------------------------------------------------------
// min_counts
// ---------------------------------------------------------------------------

#[test]
fn min_counts_matches_an_equivalent_explicit_mask() {
    let dir = tempfile::tempdir().unwrap();
    let (n_obs, n_vars) = (20usize, 15usize);
    let scx = build_scx(dir.path(), n_obs, n_vars, false);

    let threshold = 100.0;
    let expected_mask = crate::min_counts_obs_mask(&scx, 0, threshold).unwrap();
    let kept = expected_mask.iter().filter(|&&b| b).count();
    assert!(kept > 0 && kept < n_obs, "threshold must actually bite");

    let by_counts = dir.path().join("counts.h5ad");
    export(
        &scx,
        &by_counts,
        &ConvertOptions {
            export_min_counts: Some(threshold),
            ..Default::default()
        },
    )
    .unwrap();

    let by_mask = dir.path().join("mask.h5ad");
    export(&scx, &by_mask, &mask_opts(expected_mask)).unwrap();

    let a = hdf5::File::open(&by_counts).unwrap();
    let b = hdf5::File::open(&by_mask).unwrap();
    assert_eq!(x_rows(&a), kept);
    assert_eq!(x_values(&a), x_values(&b));
}

#[test]
fn min_counts_and_mask_intersect() {
    let dir = tempfile::tempdir().unwrap();
    let (n_obs, n_vars) = (20usize, 15usize);
    let scx = build_scx(dir.path(), n_obs, n_vars, false);

    let threshold = 100.0;
    let counts_mask = crate::min_counts_obs_mask(&scx, 0, threshold).unwrap();
    let user_mask: Vec<bool> = (0..n_obs).map(|i| i % 2 == 0).collect();
    let expected = counts_mask
        .iter()
        .zip(user_mask.iter())
        .filter(|(a, b)| **a && **b)
        .count();

    let out = dir.path().join("out.h5ad");
    export(
        &scx,
        &out,
        &ConvertOptions {
            export_obs_keep_mask: Some(Arc::from(user_mask)),
            export_min_counts: Some(threshold),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(x_rows(&hdf5::File::open(&out).unwrap()), expected);
}

// ---------------------------------------------------------------------------
// Parallel / sequential equivalence
// ---------------------------------------------------------------------------

/// The masked path runs through `stream_csr_into_prealloc_parallel`'s reorder
/// buffer when threads > 1; it must produce byte-identical output.
#[test]
fn masked_export_parallel_equals_sequential() {
    let dir = tempfile::tempdir().unwrap();
    let (n_obs, n_vars) = (40usize, 12usize);
    let scx = build_scx(dir.path(), n_obs, n_vars, true);
    let mask: Vec<bool> = (0..n_obs).map(|i| i % 3 != 0).collect();

    let seq_out = dir.path().join("seq.h5ad");
    export(
        &scx,
        &seq_out,
        &ConvertOptions {
            reader_threads: Some(1),
            ..mask_opts(mask.clone())
        },
    )
    .unwrap();

    let par_out = dir.path().join("par.h5ad");
    export(
        &scx,
        &par_out,
        &ConvertOptions {
            reader_threads: Some(4),
            ..mask_opts(mask)
        },
    )
    .unwrap();

    let seq = hdf5::File::open(&seq_out).unwrap();
    let par = hdf5::File::open(&par_out).unwrap();
    for ds in ["X/indptr", "X/indices", "X/data"] {
        let a: Vec<f64> = seq
            .dataset(ds)
            .unwrap()
            .read_1d::<f64>()
            .map(|v| v.to_vec())
            .unwrap_or_default();
        let b: Vec<f64> = par
            .dataset(ds)
            .unwrap()
            .read_1d::<f64>()
            .map(|v| v.to_vec())
            .unwrap_or_default();
        assert_eq!(a, b, "{ds} diverged between 1 and 4 reader threads");
    }
}

// ---------------------------------------------------------------------------
// Direction guards
// ---------------------------------------------------------------------------

#[test]
fn export_row_filter_is_rejected_on_import_directions() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    let scx = dir.path().join("out.scx");
    create_test_h5ad(&h5ad, 10, 8, "csr", false);

    let opts = ConvertOptions {
        export_min_counts: Some(5.0),
        ..Default::default()
    };
    let err = h5ad_to_scx(&h5ad, &scx, &opts, &mut WarningSink::log()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("export_min_counts"), "{msg}");
    assert!(msg.contains("h5ad_to_scx"), "{msg}");

    let err = h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("export_min_counts"));
}

// ---------------------------------------------------------------------------
// uns provenance
// ---------------------------------------------------------------------------

#[test]
fn unfiltered_export_writes_no_scx_export_key() {
    let dir = tempfile::tempdir().unwrap();
    let scx = build_scx(dir.path(), 10, 8, true);
    let out = dir.path().join("out.h5ad");
    export(&scx, &out, &ConvertOptions::default()).unwrap();

    let f = hdf5::File::open(&out).unwrap();
    let uns = f.group("uns").unwrap();
    assert!(
        uns.member_names()
            .unwrap()
            .iter()
            .all(|n| n != EXPORT_PROVENANCE_KEY),
        "an unfiltered export must be unchanged"
    );
}

/// A deletion-vector-only export must also stay clean — the note is gated on
/// caller filters, not on the mask merely existing.
#[test]
fn deletion_only_export_writes_no_scx_export_key() {
    let dir = tempfile::tempdir().unwrap();
    let scx = build_scx(dir.path(), 10, 8, true);
    scx_ops::mark_deleted(&scx, &[0]).unwrap();
    let out = dir.path().join("out.h5ad");
    export(&scx, &out, &ConvertOptions::default()).unwrap();

    let f = hdf5::File::open(&out).unwrap();
    assert!(f
        .group("uns")
        .unwrap()
        .member_names()
        .unwrap()
        .iter()
        .all(|n| n != EXPORT_PROVENANCE_KEY));
}

#[test]
fn filtered_export_records_provenance_alongside_existing_uns() {
    let dir = tempfile::tempdir().unwrap();
    let scx = build_scx(dir.path(), 20, 15, true);
    let out = dir.path().join("out.h5ad");
    export(
        &scx,
        &out,
        &ConvertOptions {
            export_min_counts: Some(100.0),
            ..Default::default()
        },
    )
    .unwrap();

    let f = hdf5::File::open(&out).unwrap();
    let note = f.group(&format!("uns/{EXPORT_PROVENANCE_KEY}")).unwrap();
    let members = note.member_names().unwrap();
    for key in [
        "n_obs_source",
        "n_obs_written",
        "deletion_filtered",
        "filter",
        "min_counts",
    ] {
        assert!(
            members.contains(&key.to_string()),
            "missing {key}: {members:?}"
        );
    }

    let written: i64 = note
        .dataset("n_obs_written")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(written as usize, x_rows(&f), "note must match the output");

    // The source's own uns keys survive.
    let uns = f.group("uns").unwrap().member_names().unwrap();
    assert!(uns.contains(&"species".to_string()));
    assert!(uns.contains(&"version".to_string()));
}

/// A source `uns` that already defines `scx_export` is never clobbered.
#[test]
fn existing_scx_export_uns_key_is_preserved() {
    let dir = tempfile::tempdir().unwrap();
    let scx = build_scx(dir.path(), 20, 15, false);

    scx_ops::set_uns(
        &scx,
        &serde_json::json!({ EXPORT_PROVENANCE_KEY: "user-owned" }),
    )
    .unwrap();

    let mut categories: Vec<&'static str> = Vec::new();
    let mut sink = WarningSink::with_handler(move |_w| {});
    let out = dir.path().join("out.h5ad");
    scx_to_h5ad_streaming(
        &scx,
        &out,
        &ConvertOptions {
            export_min_counts: Some(100.0),
            ..Default::default()
        },
        &mut sink,
    )
    .unwrap();
    categories.extend(sink.counts().keys().copied());
    assert!(
        categories.contains(&"skipped_uns_key"),
        "a collision must warn, not overwrite: {categories:?}"
    );

    let f = hdf5::File::open(&out).unwrap();
    let value: hdf5::types::VarLenUnicode = f
        .dataset(&format!("uns/{EXPORT_PROVENANCE_KEY}"))
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(value.as_str(), "user-owned");
}
