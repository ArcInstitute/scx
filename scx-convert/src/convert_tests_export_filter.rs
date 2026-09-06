//! Integration tests for the caller-supplied export row filter
//! (`IngestOptions::export_obs_keep_mask` / `export_min_counts`).
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
        &IngestOptions::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    scx
}

fn export(scx: &Path, out: &Path, opts: &ExportOptions) -> Result<(), ConvertError> {
    scx_to_h5ad_streaming(scx, out, opts, &mut WarningSink::log())
}

fn mask_opts(mask: Vec<bool>) -> ExportOptions {
    ExportOptions {
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

    export(&scx, &unfiltered, &ExportOptions::default()).unwrap();

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
    export(&scx, &all_out, &ExportOptions::default()).unwrap();

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
    export(&scx, &out, &ExportOptions::default()).unwrap();
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
        &ExportOptions {
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
        &ExportOptions {
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
        &ExportOptions {
            reader_threads: Some(1),
            ..mask_opts(mask.clone())
        },
    )
    .unwrap();

    let par_out = dir.path().join("par.h5ad");
    export(
        &scx,
        &par_out,
        &ExportOptions {
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

/// A modality-scoped export is reachable with `--min-counts` (the direction is
/// still `scx_to_h5ad`), so it must honour the same "records what it dropped"
/// contract as the single-modality path — that path writes no `/uns` of its
/// own, so the note has to be added explicitly.
#[test]
fn modality_scoped_export_records_provenance_and_filters() {
    use crate::h5mu::write::scx_modality_to_h5ad_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("in.h5mu");
    let scx = dir.path().join("in.scx");
    create_test_h5mu(&h5mu, 20, 8, 4);
    crate::h5mu::pipeline::h5mu_to_scx(
        &h5mu,
        &scx,
        &IngestOptions::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let n_obs = ScxReader::open(&scx).unwrap().n_obs() as usize;
    let mask: Vec<bool> = (0..n_obs).map(|i| i % 2 == 0).collect();
    let kept = mask.iter().filter(|&&b| b).count();

    let out = dir.path().join("rna.h5ad");
    scx_modality_to_h5ad_streaming(&scx, &out, "rna", &mask_opts(mask), &mut WarningSink::log())
        .unwrap();

    let f = hdf5::File::open(&out).unwrap();
    assert_eq!(x_rows(&f), kept, "modality X must be filtered");
    assert_eq!(dataset_len(&f, "obs/_index"), kept, "shared obs too");

    let note = f.group(&format!("uns/{EXPORT_PROVENANCE_KEY}")).unwrap();
    let written: i64 = note
        .dataset("n_obs_written")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(written as usize, kept);
}

/// And an unfiltered modality export must stay byte-identical — no new /uns.
#[test]
fn unfiltered_modality_export_writes_no_provenance() {
    use crate::h5mu::write::scx_modality_to_h5ad_streaming;

    let dir = tempfile::tempdir().unwrap();
    let h5mu = dir.path().join("in.h5mu");
    let scx = dir.path().join("in.scx");
    create_test_h5mu(&h5mu, 10, 6, 3);
    crate::h5mu::pipeline::h5mu_to_scx(
        &h5mu,
        &scx,
        &IngestOptions::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let out = dir.path().join("rna.h5ad");
    scx_modality_to_h5ad_streaming(
        &scx,
        &out,
        "rna",
        &ExportOptions::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let f = hdf5::File::open(&out).unwrap();
    assert!(
        f.group("uns").is_err()
            || !f
                .group("uns")
                .unwrap()
                .member_names()
                .unwrap()
                .contains(&EXPORT_PROVENANCE_KEY.to_string())
    );
}

// ---------------------------------------------------------------------------
// Direction guards
// ---------------------------------------------------------------------------
//
// `export_row_filter_is_rejected_on_import_directions` lived here. It asserted
// that `h5ad_to_scx` / `h5ad_to_scx_streaming` returned an error naming
// `export_min_counts` when handed one.
//
// ORG-11.16-3 retired it by making the state it tested unrepresentable:
// `export_min_counts` is on `ExportOptions`, the ingest entry points take
// `IngestOptions`, and the combination no longer type-checks. The test did not
// lose coverage -- it changed instrument, from a runtime string to a compile
// error, and the compile-fail doctest pair on `ExportOptions` is where that
// claim is now asserted.

// ---------------------------------------------------------------------------
// uns provenance
// ---------------------------------------------------------------------------

#[test]
fn unfiltered_export_writes_no_scx_export_key() {
    let dir = tempfile::tempdir().unwrap();
    let scx = build_scx(dir.path(), 10, 8, true);
    let out = dir.path().join("out.h5ad");
    export(&scx, &out, &ExportOptions::default()).unwrap();

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
    export(&scx, &out, &ExportOptions::default()).unwrap();

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
        &ExportOptions {
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
        &ExportOptions {
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

// ---------------------------------------------------------------------------
// `/raw` under a row filter (OPT-CONVERT-1)
//
// Invariant 2 of this file's header — "every obs-axis section agrees on the
// kept row count" — covers `/raw` too: raw shares the obs axis. It was
// untested here because raw was exported eagerly, by a different function,
// with its own row-filter implementation (`write.rs::filter_csr_rows`).
// ---------------------------------------------------------------------------

/// `build_scx` with an `adata.raw` on a WIDER gene axis, spanning several
/// raw shards.
///
/// `add_raw_group`'s canonical CSR is deliberately not returned: every
/// assertion below compares against an unfiltered export of the same file, so
/// that the expectation is not a second derivation of the fixture generator.
fn build_scx_with_raw(
    dir: &Path,
    n_obs: usize,
    n_vars: usize,
    raw_n_vars: usize,
) -> std::path::PathBuf {
    let h5ad = dir.join("raw_in.h5ad");
    let scx = dir.join("raw_in.scx");
    // `extras = true` also writes a LAYER named "raw". Deliberate: it proves
    // the `RawCsrShard` and `LayerCsrShard` families do not cross-contaminate
    // now that both reach the same streaming driver.
    create_test_h5ad(&h5ad, n_obs, n_vars, "csr", true);
    add_raw_group(&h5ad, n_obs, raw_n_vars);
    let opts = IngestOptions {
        shard_target_rows: 4,
        ..IngestOptions::default()
    };
    h5ad_to_scx(&h5ad, &scx, &opts, &mut WarningSink::log()).unwrap();
    scx
}

fn raw_values(file: &hdf5::File) -> Vec<f32> {
    file.dataset("raw/X/data")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec()
}

fn raw_indptr(file: &hdf5::File) -> Vec<i64> {
    file.dataset("raw/X/indptr")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec()
}

/// A caller mask filters `/raw/X` to the same rows as `/X` and `obs`, and
/// the surviving VALUES are the masked rows of the unfiltered export.
///
/// Values, not just counts, on purpose. The pre-allocation that sizes
/// `/raw/X/{indices,data}` comes from a pre-scan over the raw shards' own
/// indptrs; a pre-scan that read `/X`'s instead would still produce the right
/// row count. On this fixture raw is SPARSER than X (1–2 nnz/row against
/// 2–3), so the mis-sized datasets would be over-allocated and the failure
/// would be a garbage tail — silent unless the values are checked.
#[test]
fn mask_filters_raw_consistently_with_x_and_obs() {
    let dir = tempfile::tempdir().unwrap();
    let (n_obs, n_vars, raw_n_vars) = (20usize, 15usize, 23usize);
    let scx = build_scx_with_raw(dir.path(), n_obs, n_vars, raw_n_vars);

    // Unfiltered baseline from the same file, so the expectation is not a
    // re-derivation of the fixture generator.
    let all_out = dir.path().join("all.h5ad");
    export(&scx, &all_out, &ExportOptions::default()).unwrap();
    let all = hdf5::File::open(&all_out).unwrap();
    let all_ip = raw_indptr(&all);
    let all_data = raw_values(&all);

    // ODD rows, not even. `create_test_h5ad` gives row r `2 + (r % 2)`
    // nonzeros and `add_raw_group` gives it 2 (1 where its two columns
    // collide), so an EVEN-row mask makes X's kept nnz and raw's coincide
    // exactly — and a pre-scan that sized `/raw/X` from `/X` would allocate
    // the right length by accident. The premise is asserted below rather
    // than left to the fixture generators to keep agreeing.
    let mask: Vec<bool> = (0..n_obs).map(|r| r % 2 == 1).collect();
    let kept = mask.iter().filter(|&&b| b).count();

    let out = dir.path().join("out.h5ad");
    export(&scx, &out, &mask_opts(mask.clone())).unwrap();
    let f = hdf5::File::open(&out).unwrap();

    assert_eq!(x_rows(&f), kept, "/X row count");
    assert_eq!(raw_indptr(&f).len() - 1, kept, "/raw/X row count");
    assert_eq!(dataset_len(&f, "obs/_index"), kept, "obs row count");

    let raw_shape: Vec<i64> = f
        .group("raw/X")
        .unwrap()
        .attr("shape")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(raw_shape, vec![kept as i64, raw_n_vars as i64]);

    let mut expected = Vec::new();
    for (row, &keep) in mask.iter().enumerate() {
        if keep {
            let (s, e) = (all_ip[row] as usize, all_ip[row + 1] as usize);
            expected.extend_from_slice(&all_data[s..e]);
        }
    }
    assert_eq!(raw_values(&f), expected, "/raw/X kept values");
    assert_eq!(
        dataset_len(&f, "raw/X/indices"),
        expected.len(),
        "indices must be allocated to the kept nnz, not over-allocated"
    );
    assert_eq!(dataset_len(&f, "raw/X/data"), expected.len());

    // The premise that makes the two assertions above discriminating: if
    // `/X` and `/raw/X` keep the same number of nonzeros, a pre-scan that
    // read the wrong section family would still allocate the right length,
    // and this test would pass against that bug. Watched red with the
    // pre-scan pointed at `/X` — it fails only because these differ.
    assert_ne!(
        dataset_len(&f, "X/data"),
        dataset_len(&f, "raw/X/data"),
        "fixture must keep a DIFFERENT nnz for X and raw, else the \
         allocation assertions above are vacuous"
    );
}

/// A deletion vector filters `/raw` identically on the streaming and eager
/// exporters. The eager path applies it through `write.rs::filter_csr_rows`,
/// an independent implementation — which is what makes this a differential
/// rather than a restatement of the streaming code.
#[test]
fn deletion_vector_filters_raw_like_the_eager_exporter() {
    let dir = tempfile::tempdir().unwrap();
    let (n_obs, n_vars, raw_n_vars) = (20usize, 15usize, 23usize);
    let scx = build_scx_with_raw(dir.path(), n_obs, n_vars, raw_n_vars);

    scx_ops::mark_deleted(&scx, &[0, 1, 7]).unwrap();

    let streamed_out = dir.path().join("streamed.h5ad");
    export(&scx, &streamed_out, &ExportOptions::default()).unwrap();
    let eager_out = dir.path().join("eager.h5ad");
    crate::pipeline::scx_to_h5ad(&scx, &eager_out, &mut WarningSink::log()).unwrap();

    let s = hdf5::File::open(&streamed_out).unwrap();
    let e = hdf5::File::open(&eager_out).unwrap();
    assert_eq!(raw_indptr(&s), raw_indptr(&e), "raw indptr");
    assert_eq!(raw_values(&s), raw_values(&e), "raw data");
    assert_eq!(
        dataset_len(&s, "raw/X/indices"),
        dataset_len(&e, "raw/X/indices"),
        "raw indices length"
    );
    assert_eq!(raw_indptr(&s).len() - 1, n_obs - 3, "deleted rows dropped");
    assert_eq!(x_rows(&s), n_obs - 3, "/X agrees with /raw");
}

/// Deleting every row leaves an empty-but-well-formed `/raw/X`: a
/// single-element indptr and zero-length indices/data, matching what the
/// eager writer produces. Sibling of
/// `all_deleted_without_a_caller_filter_still_exports` for the raw section.
#[test]
fn all_deleted_export_writes_an_empty_raw() {
    let dir = tempfile::tempdir().unwrap();
    let (n_obs, n_vars, raw_n_vars) = (8usize, 10usize, 17usize);
    let scx = build_scx_with_raw(dir.path(), n_obs, n_vars, raw_n_vars);

    let all: Vec<u64> = (0..n_obs as u64).collect();
    scx_ops::mark_deleted(&scx, &all).unwrap();

    let out = dir.path().join("empty.h5ad");
    export(&scx, &out, &ExportOptions::default()).unwrap();
    let f = hdf5::File::open(&out).unwrap();

    assert_eq!(raw_indptr(&f), vec![0i64], "empty raw indptr");
    assert_eq!(dataset_len(&f, "raw/X/indices"), 0);
    assert_eq!(dataset_len(&f, "raw/X/data"), 0);
    let shape: Vec<i64> = f
        .group("raw/X")
        .unwrap()
        .attr("shape")
        .unwrap()
        .read_1d()
        .unwrap()
        .to_vec();
    assert_eq!(shape, vec![0, raw_n_vars as i64]);
}
