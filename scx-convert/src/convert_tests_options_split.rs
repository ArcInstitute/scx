//! Output identity across the ingest/export options split (ORG-11.16-3).
//!
//! The split moves 33 fields between types, renames one, deletes another, and
//! rewrites five construction sites plus ~200 test call sites. Its whole claim
//! is that **not one output byte changes**, and the failure mode that claim has
//! is quiet: five of the construction sites end in `..Default::default()` or
//! `..opts.clone()`, so an initializer lost in the churn still compiles and
//! silently reverts that field to its default.
//!
//! Per-section digests are what catch that. `Strictness::Content` compares each
//! section's identity, size and bytes but not its offset — offsets shift
//! whenever anything earlier in the file changes size, including the excluded
//! `Provenance` section, so pinning them would make the comparison depend on
//! exactly the thing it is trying to ignore. `DEFAULT_EXCLUDED` drops
//! `Provenance`, which stamps `SystemTime::now()`; a whole-file BLAKE3 would be
//! useless here for that reason.

use super::convert_tests_common::*;
use scx_testkit::digest::{assert_matches_golden, Strictness};

fn golden(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/goldens")
        .join(format!("{name}.json"))
}

/// Eager h5ad ingest at default options.
#[test]
fn eager_h5ad_ingest_output_is_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    create_test_h5ad(&h5ad, 64, 13, "csr", true);
    let scx = dir.path().join("out.scx");
    super::h5ad_to_scx(
        &h5ad,
        &scx,
        &ConvertOptions::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    assert_matches_golden(&scx, &golden("eager_h5ad_default"), Strictness::Content).unwrap();
}

/// Streaming h5ad ingest with **non-default** options on the fields most
/// likely to be dropped by a bad split.
///
/// Deliberately not `Default::default()`: a defaulted field here changes the
/// *section set* (no predicate index, one obs section instead of shards), not
/// merely section bytes, so the digest names what went missing instead of
/// reporting an opaque byte difference.
#[test]
fn streaming_h5ad_ingest_output_is_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    create_test_h5ad(&h5ad, 64, 13, "csr", true);
    let scx = dir.path().join("out.scx");
    let opts = ConvertOptions {
        shard_target_rows: 4,
        reader_threads: Some(1),
        writer_queue_depth: 2,
        index_obs: vec!["n_counts".to_string()],
        obs_shard_policy: scx_format_io::ObsShardPolicy::Always,
        ..ConvertOptions::default()
    };
    super::h5ad_to_scx_streaming(
        &h5ad,
        &scx,
        &opts,
        &StreamingOverrides::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    assert_matches_golden(
        &scx,
        &golden("streaming_h5ad_nondefault"),
        Strictness::Content,
    )
    .unwrap();
}

/// The export arm's oracle: SCX -> h5ad -> SCX, digesting the second SCX.
///
/// `digest_file` reads SCX only, and libhdf5 output is not reliably
/// byte-reproducible (heap free-lists), so re-ingesting is the honest way to
/// hash an h5ad through this harness.
///
/// ⚠️ Known blind spot, stated rather than left to be discovered: a change in
/// the exported h5ad that re-ingest normalises away — attribute ordering, say —
/// is invisible here. The value-level half is covered by
/// `convert_tests_export_filter`'s `read_h5ad_x_triplet` assertions.
#[test]
fn export_round_trip_output_is_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let h5ad = dir.path().join("in.h5ad");
    create_test_h5ad(&h5ad, 64, 13, "csr", true);
    let scx = dir.path().join("mid.scx");
    super::h5ad_to_scx(
        &h5ad,
        &scx,
        &ConvertOptions::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let exported = dir.path().join("exported.h5ad");
    super::scx_to_h5ad_streaming(
        &scx,
        &exported,
        &ConvertOptions {
            tool: "scx".into(),
            ..ConvertOptions::default()
        },
        &mut WarningSink::log(),
    )
    .unwrap();

    let back = dir.path().join("back.scx");
    super::h5ad_to_scx(
        &exported,
        &back,
        &ConvertOptions::default(),
        &mut WarningSink::log(),
    )
    .unwrap();
    assert_matches_golden(&back, &golden("export_round_trip"), Strictness::Content).unwrap();
}
