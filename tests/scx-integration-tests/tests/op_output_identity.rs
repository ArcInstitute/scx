//! The committed byte-identity A/B matrix: what does each rewriting op write?
//!
//! Seven ops, one manifest of per-section digests, three ways to use it:
//!
//! * **default** — assert against the checked-in golden. A refactor that
//!   changes any op's output fails `cargo test`, in CI, without anybody having
//!   to remember to run an A/B.
//! * **`SCX_TESTKIT_AB_DUMP=<path>`** — write the manifest and assert nothing.
//! * **`SCX_TESTKIT_AB_BASE=<path>`** — assert the manifest equals the one at
//!   `<path>`.
//!
//! The last two are the cross-tree A/B. Run the dump in a worktree at the base
//! commit, then the compare in the worktree carrying the change:
//!
//! ```text
//! git worktree add /tmp/scx-base <base-sha>
//! (cd /tmp/scx-base && SCX_TESTKIT_AB_DUMP=/tmp/base.json \
//!    cargo test -p scx-integration-tests --test op_output_identity)
//! SCX_TESTKIT_AB_BASE=/tmp/base.json \
//!    cargo test -p scx-integration-tests --test op_output_identity
//! ```
//!
//! That is the harness the organization series' Phase 5a/5b behaviour-identity
//! claims were measured with — as a scratch test hand-copied into two
//! worktrees and then deleted. It is committed here so the next such claim is
//! reproducible instead of anecdotal.
//!
//! **`SCX_TESTKIT_BLESS=1` rewrites the golden.** A blessed golden is a claim
//! that the output change was intended; the diff names the op and the section,
//! so review it rather than blessing reflexively.
//!
//! ## What this cannot see
//!
//! `FileHeader::file_checksum` is deliberately outside the digest (it covers
//! the `Provenance` section, which is itself excluded), so a change to what
//! `file_checksum` *means* passes here untouched and needs its own oracle. So
//! does the 4096-byte root catalog at offset 256, which has no production
//! readers. See `scx_testkit::ab`'s module docs for the full list.
//!
//! ## Relationship to `testkit_against_real_ops.rs`
//!
//! That file asserts the weaker, prior property — that provenance's clock does
//! not leak into an op's digest, i.e. that the harness is *usable* on that op.
//! It compares two runs of the same code. This file compares across commits.

mod common;

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use common::{appendable_rows, fixture_all_families, fixture_all_families_without_raw};
use scx_codec::ValueEncoding;
use scx_testkit::ab::{assert_manifests_eq, resolve_against_env, OpDigestManifest};
use scx_testkit::digest::Strictness;
use scx_testkit::fixtures::mixed_codec_file;

/// Every op PR-01 names, in the order the manifest reports them.
const EXPECTED_OPS: &[&str] = &[
    "append",
    "build_csc",
    "build_csc_indexed",
    "compact",
    "compact_indexed",
    "delete",
    "merge",
    "optimize",
    "sort",
];

/// Force a predicate index on one obs column.
///
/// `index_auto_threshold: 0` is the tree-wide "do nothing unless forced"
/// sentinel; `forced_columns` is what actually makes the pass run.
fn index_obs_on(column: &str) -> scx_engine::ConversionPredicateIndexOptions {
    scx_engine::ConversionPredicateIndexOptions {
        index_obs: vec![column.to_string()],
        index_var: Vec::new(),
        index_preset: None,
        index_auto_threshold: 0,
    }
}

fn golden() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/op_output_identity.json")
}

/// Run every op once under `dir` and digest each output.
///
/// Deliberately built from scratch on each call rather than cached: two calls
/// must be able to straddle a wall-clock second, which is what
/// `the_matrix_does_not_depend_on_the_wall_clock` needs.
fn build_manifest(dir: &Path) -> OpDigestManifest {
    let mut m = OpDigestManifest::new();
    let src = fixture_all_families(dir, "src.scx");

    // --- copy-out ops over the rich fixture ------------------------------
    let out = dir.join("compact.scx");
    scx_ops::compact(&src, &out).unwrap();
    m.record("compact", &out, Strictness::Content).unwrap();

    let out = dir.join("sort.scx");
    let sort_opts = scx_ops::SortOptions {
        by: vec!["cell_type".to_string()],
        ..Default::default()
    };
    scx_ops::sort_engine::sort(&src, &out, &sort_opts).unwrap();
    m.record("sort", &out, Strictness::Content).unwrap();

    let out = dir.join("optimize.scx");
    scx_ops::optimize(&src, &out, None, scx_format_io::ObsShardPolicy::Off).unwrap();
    m.record("optimize", &out, Strictness::Content).unwrap();

    // `merge` validates var identity and requires matching layers, so both
    // inputs are the same fixture shape rather than two arbitrary files.
    let y = fixture_all_families(dir, "merge_y.scx");
    let out = dir.join("merge.scx");
    scx_ops::merge(&[&src, &y], &out).unwrap();
    m.record("merge", &out, Strictness::Content).unwrap();

    // --- build-csc, over the three-codec fixture -------------------------
    //
    // Not the all-families fixture. `fixture_all_families` carries no CSC
    // sidecar (nothing but build-csc writes one) and build-csc strips
    // varm/obsp/varp/raw/bitmaps/group-index on the way, so most of that
    // fixture's richness never reaches the output. What build-csc's diff
    // actually touches is its per-shard CSR re-emit loop, and
    // `mixed_codec_file` is the only fixture in the tree that exercises all
    // three encoder paths (unframed integer, row-group-framed integer, float).
    let csc_src = mixed_codec_file(&dir.join("csc_src.scx")).unwrap();
    let out = dir.join("build_csc.scx");
    scx_ops::run_build_csc(&csc_src, &out, "1G", false, 1024, None).unwrap();
    m.record("build_csc", &out, Strictness::Content).unwrap();

    // --- the same two ops over an input carrying per-shard `column_stats` ---
    //
    // Necessary, not belt-and-braces. `StatsDigest.column_stats` is in the
    // digest because a file whose shard bytes are byte-identical can still
    // carry different Level-1 pruning statistics — the §6.1 shape, where stale
    // stats make a query silently return nothing. But **no fixture in this
    // matrix produces them**: `write_csr_shard` computes shard stats without
    // column stats, and neither `fixture_all_families` nor `mixed_codec_file`
    // goes through an index pass. Without these two arms the `column_stats`
    // half of the digest is dead weight here, and build-csc's
    // `carry_csr_shard_column_stats_from` — the exact call PR-07's diff sits
    // next to — could be deleted with the golden unmoved.
    let indexed = dir.join("indexed.scx");
    scx_ops::compact_with_options(
        &src,
        &indexed,
        &scx_ops::CompactOptions {
            index_options: index_obs_on("cell_type"),
            ..Default::default()
        },
    )
    .unwrap();
    m.record("compact_indexed", &indexed, Strictness::Content)
        .unwrap();

    let out = dir.join("build_csc_indexed.scx");
    scx_ops::run_build_csc(&indexed, &out, "1G", false, 1024, None).unwrap();
    m.record("build_csc_indexed", &out, Strictness::Content)
        .unwrap();

    // --- in-place ops, each on its own copy ------------------------------
    //
    // Digested on the mutated target, not on a separate output: that IS the
    // artifact for an in-place op.
    // `append` refuses a file carrying `adata.raw` (`append.rs:592`,
    // `OpsError::RawUnsupported`) because it extends X's obs axis and not
    // raw's. Its input is therefore the all-families fixture minus that one
    // family — every other family is still present and still digested.
    let target = fixture_all_families_without_raw(dir, "append.scx");
    let (new_obs, indptr, indices, values) = appendable_rows(5);
    scx_ops::append(
        &target,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        &scx_ops::AppendOptions {
            // Smaller than the 5 appended rows on purpose, so the append
            // produces more than one shard and the digest covers the
            // shard-boundary arithmetic rather than a single-shard special
            // case.
            shard_target_rows: NonZeroU32::new(2).unwrap(),
            ..Default::default()
        },
    )
    .unwrap();
    m.record("append", &target, Strictness::Content).unwrap();

    let target = dir.join("delete.scx");
    std::fs::copy(&src, &target).unwrap();
    // The fixture already carries a deletion vector (rows 2 and 5), so this
    // exercises the merge-into-existing path, not the create path.
    scx_ops::mark_deleted(&target, &[1, 6]).unwrap();
    m.record("delete", &target, Strictness::Content).unwrap();

    m
}

/// The premise: the matrix really does cover the seven ops PR-01 names.
///
/// Without it, dropping an op from `build_manifest` would only be caught by a
/// golden re-bless — which is exactly the moment somebody is least likely to
/// notice coverage shrinking.
#[test]
fn the_matrix_covers_every_op_the_plan_names() {
    let dir = tempfile::tempdir().unwrap();
    let m = build_manifest(dir.path());
    let labels: Vec<&str> = m.labels().collect();
    assert_eq!(labels, EXPECTED_OPS);
}

/// The digests must not move between two runs of the same code a second apart.
///
/// This is what makes a checked-in golden viable at all: every one of these
/// ops stamps `SystemTime::now()` into its provenance section, and the golden
/// would be a coin flip on the wall clock if the exclusion did not hold.
#[test]
fn the_matrix_does_not_depend_on_the_wall_clock() {
    let a_dir = tempfile::tempdir().unwrap();
    let a = build_manifest(a_dir.path());

    scx_testkit::ab::wait_for_next_second();

    let b_dir = tempfile::tempdir().unwrap();
    let b = build_manifest(b_dir.path());

    // Premise: the two runs really did stamp different provenance, so the
    // agreement below is a statement about the exclusion and not about two
    // identical files.
    scx_testkit::ab::assert_provenance_actually_differs(
        &a_dir.path().join("compact.scx"),
        &b_dir.path().join("compact.scx"),
    );

    assert_manifests_eq(&a, &b);
}

/// The claim itself: no op's output has changed.
///
/// Default mode asserts against the golden; `SCX_TESTKIT_AB_DUMP` /
/// `SCX_TESTKIT_AB_BASE` redirect it to the cross-tree A/B. See the module
/// docs.
#[test]
fn every_op_still_writes_what_it_wrote() {
    let dir = tempfile::tempdir().unwrap();
    let m = build_manifest(dir.path());
    resolve_against_env(&m, &golden()).unwrap();
}
