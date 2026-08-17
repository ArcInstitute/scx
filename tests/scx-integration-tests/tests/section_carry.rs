//! Cross-op invariant: **what each rewrite op does with each section family.**
//!
//! Sibling of `deletion_survival.rs`, and the same shape of test for the same
//! reason. That file pins one family (deletion vectors) across every op, because
//! one missing invariant was found independently in four crates. This file pins
//! *all* of them, because the deletion-vector defect was never really about
//! deletion vectors — it was about "which sections survive this op" being
//! encoded four times in four incompatible shapes, with nothing connecting them.
//!
//! `scx_ops::carry` is now the one place that answers the question, and it is
//! checked at run time by `carry::audit`. But an audit can only catch an op that
//! contradicts the table; it cannot catch a table that is wrong about every op
//! in the same direction. That is what these tests are for: they run the real
//! ops over a fixture carrying **every** family and assert the surviving set
//! against the declaration, independently.
//!
//! ## Several assertions here pin known bugs
//!
//! `merge` drops obsp/varp, and `build-csc` drops varm/obsp/varp/`.raw`/
//! bitmaps/the group index. Those are §6.3 and §6.4 of the 2026-08-05 review,
//! both open Majors. They are asserted **as drops** because that is what the
//! code does today, and the next PR in this series flips them — at which point
//! these assertions move to the other list and the diff says exactly what
//! changed for whom.
//!
//! A test that encodes a known bug has to say so in the test. `append`'s
//! categorical tests are the cautionary case: they assert `Utf8` because that is
//! what `append` produces, and nothing in them records that `Dictionary` is what
//! it *should* produce, so they read as a specification for the bug.

mod common;

use std::collections::BTreeSet;
use std::path::Path;

use scx_format_io::reader::ScxReader;
use scx_format_io::ObsShardPolicy;
use scx_ops::carry::{family, policy, Carry, RewriteOp, SectionFamily};

use common::{
    fixture_all_families, fixture_all_families_without_varm, fixture_multimodal_per_modality_obsp,
    fixture_with_csr_obsp,
};

/// The families a file actually carries, read off its catalog.
fn families_of(path: &Path) -> BTreeSet<SectionFamily> {
    let reader = ScxReader::open(path).unwrap();
    reader
        .catalog()
        .entries
        .iter()
        .map(|e| family(e.section_type))
        .collect()
}

/// Assert an op's output against the table, both directions at once.
///
/// Checking only the survivors would pass a run that also carried something the
/// table says it drops, and checking only the drops would pass one that lost
/// everything.
///
/// `Rebuilt` and `Conditional` entries assert nothing and fall through — that is
/// their whole definition, and there is deliberately no hand-maintained exempt
/// list here to drift out of step with the table.
fn assert_matches_table(op: RewriteOp, input: &Path, output: &Path) {
    let before = families_of(input);
    let after = families_of(output);

    let mut problems = Vec::new();
    for &f in &before {
        let present = after.contains(&f);
        match policy(op, f) {
            Carry::Verbatim | Carry::RowFiltered | Carry::Remapped if !present => problems.push(
                format!("{}: table says carried, output does not have it", f.label()),
            ),
            Carry::Dropped { .. } if present => {
                problems.push(format!("{}: table says dropped, output has it", f.label()))
            }
            _ => {}
        }
    }
    assert!(
        problems.is_empty(),
        "{} disagrees with scx_ops::carry:\n  {}\ninput families:  {:?}\noutput families: {:?}",
        op.label(),
        problems.join("\n  "),
        before,
        after
    );
}

/// The fixture must actually carry everything, or every test below is vacuous.
///
/// This is the premise assertion. Without it, a fixture that quietly stopped
/// writing `varp` would turn "build-csc drops varp" into a statement about
/// nothing, and the whole file would still be green.
#[test]
fn the_fixture_carries_every_family_an_op_can_decide_about() {
    let dir = tempfile::tempdir().unwrap();
    let got = families_of(&fixture_all_families(dir.path(), "all.scx"));

    // Not present, and correctly so: `XCsc` and `LayerCsc` need `build-csc`
    // (covered separately, below), `ModalityTable` needs a multimodal file, and
    // `Unwritten` has no writer in this workspace at all.
    let cannot_be_here = [
        SectionFamily::XCsc,
        SectionFamily::LayerCsc,
        SectionFamily::ModalityTable,
        SectionFamily::Unwritten,
        // Needs `n_vars > n_obs` (an obsp graph is obs x obs, and the writer
        // takes every shard's minor extent from the header's `n_vars`), so it
        // cannot live in this fixture's shape. `csr_backed_obsp_is_its_own_family`
        // covers it with dimensions that permit it.
        SectionFamily::ObspCsr,
    ];
    let want: BTreeSet<_> = SectionFamily::ALL
        .iter()
        .copied()
        .filter(|f| !cannot_be_here.contains(f))
        .collect();

    assert_eq!(
        got,
        want,
        "the all-families fixture is incomplete; missing: {:?}",
        want.difference(&got).collect::<Vec<_>>()
    );
}

#[test]
fn compact_matches_the_table() {
    let dir = tempfile::tempdir().unwrap();
    let input = fixture_all_families(dir.path(), "in.scx");
    let output = dir.path().join("compacted.scx");
    scx_ops::compact(&input, &output).unwrap();
    assert_matches_table(RewriteOp::Compact, &input, &output);
}

#[test]
fn optimize_matches_the_table() {
    let dir = tempfile::tempdir().unwrap();
    let input = fixture_all_families(dir.path(), "in.scx");
    let output = dir.path().join("optimized.scx");
    scx_ops::optimize::optimize(&input, &output, None, ObsShardPolicy::Off).unwrap();
    assert_matches_table(RewriteOp::Optimize, &input, &output);
}

#[test]
fn sort_matches_the_table() {
    let dir = tempfile::tempdir().unwrap();
    let input = fixture_all_families(dir.path(), "in.scx");
    let output = dir.path().join("sorted.scx");
    let opts = scx_ops::SortOptions {
        by: vec!["cell_type".to_string()],
        ..Default::default()
    };
    scx_ops::sort_engine::sort(&input, &output, &opts).unwrap();
    assert_matches_table(RewriteOp::Sort, &input, &output);
}

/// The other arm of `sort`'s `Conditional` bitmap entry.
///
/// `sort_matches_the_table` runs with the default `BitmapPolicy::Off`, under
/// which sort drops the sidecar — so on its own it would leave "conditional"
/// meaning "we never checked". With `Always`, sort rebuilds the bitmaps against
/// the permuted rows and the family comes back.
///
/// This is the difference between a table entry that documents a decision and
/// one that documents an absence of testing.
#[test]
fn sort_rebuilds_bitmaps_when_asked_to() {
    let dir = tempfile::tempdir().unwrap();
    let input = fixture_all_families(dir.path(), "in.scx");
    let output = dir.path().join("sorted_bitmap.scx");
    let opts = scx_ops::SortOptions {
        by: vec!["cell_type".to_string()],
        bitmap: scx_format_io::BitmapPolicy::Always,
        ..Default::default()
    };
    scx_ops::sort_engine::sort(&input, &output, &opts).unwrap();
    assert_matches_table(RewriteOp::Sort, &input, &output);

    assert!(
        families_of(&output).contains(&SectionFamily::Bitmap),
        "sort --bitmap always must re-emit the detection bitmaps"
    );
    assert!(
        matches!(
            policy(RewriteOp::Sort, SectionFamily::Bitmap),
            Carry::Conditional { .. }
        ),
        "both arms are exercised, so the entry must stay Conditional"
    );
}

#[test]
fn merge_matches_the_table() {
    let dir = tempfile::tempdir().unwrap();
    let a = fixture_all_families(dir.path(), "a.scx");
    let b = fixture_all_families(dir.path(), "b.scx");
    let output = dir.path().join("merged.scx");
    scx_ops::merge::merge(&[&a, &b], &output).unwrap();
    assert_matches_table(RewriteOp::Merge, &a, &output);
}

#[test]
fn build_csc_matches_the_table() {
    let dir = tempfile::tempdir().unwrap();
    let input = fixture_all_families(dir.path(), "in.scx");
    let output = dir.path().join("with_csc.scx");
    scx_ops::run_build_csc(&input, &output, "1G", false, 1024, None).unwrap();
    assert_matches_table(RewriteOp::BuildCsc, &input, &output);

    // The op's whole purpose, and the one family it *adds*.
    assert!(families_of(&output).contains(&SectionFamily::XCsc));
}

/// The four cells of the table that are open Majors, asserted as the losses
/// they are — from the user's side, not the catalog's.
///
/// Deliberately spelled out rather than folded into `assert_matches_table`:
/// when Phase 5b fixes them, this test is what has to be inverted, and a
/// reviewer of that PR should be able to read what changes without
/// reconstructing it from a table walk.
#[test]
fn known_open_drops_are_still_dropping() {
    let dir = tempfile::tempdir().unwrap();
    let input = fixture_all_families(dir.path(), "in.scx");

    // §6.4 — merge loses obsp and varp, silently. `compact` remaps obsp through
    // its keep-mask and `sort` through its permutation, so a merged kNN graph
    // vanishes where a compacted or sorted one survives.
    let merged = dir.path().join("merged.scx");
    let b = fixture_all_families(dir.path(), "b.scx");
    scx_ops::merge::merge(&[&input, &b], &merged).unwrap();
    let after_merge = families_of(&merged);
    assert!(!after_merge.contains(&SectionFamily::Obsp), "§6.4 fixed?");
    assert!(!after_merge.contains(&SectionFamily::Varp), "§6.4 fixed?");

    // §6.3 — build-csc's carry allowlist is the narrowest of any op here, and
    // its output renames over the input with no prior catalog, so `scx
    // rollback` cannot recover any of this.
    let with_csc = dir.path().join("with_csc.scx");
    scx_ops::run_build_csc(&input, &with_csc, "1G", false, 1024, None).unwrap();
    let after_csc = families_of(&with_csc);
    for f in [
        SectionFamily::Varm,
        SectionFamily::Obsp,
        SectionFamily::Varp,
        SectionFamily::Raw,
        SectionFamily::Bitmap,
        SectionFamily::GroupIndex,
    ] {
        assert!(
            !after_csc.contains(&f),
            "§6.3 is fixed for {} — invert this test and the carry table entry",
            f.label()
        );
    }

    // And the ops that get it right, so the contrast is pinned too: losing
    // these would be a regression that "build-csc drops things" would hide.
    let optimized = dir.path().join("optimized.scx");
    scx_ops::optimize::optimize(&input, &optimized, None, ObsShardPolicy::Off).unwrap();
    let after_opt = families_of(&optimized);
    for f in [
        SectionFamily::Varm,
        SectionFamily::Obsp,
        SectionFamily::Varp,
        SectionFamily::Bitmap,
        SectionFamily::GroupIndex,
    ] {
        assert!(
            after_opt.contains(&f),
            "optimize carries {} and must keep doing so",
            f.label()
        );
    }
}

/// A **CSR-backed** obsp graph is a different family from a COO one, and the ops
/// treat them differently: `optimize` re-encodes `ObspCsrShard` in its shard
/// loop, `compact` and `sort` never read it.
///
/// Folding the two into one family — which is what the first version of the
/// table did — is wrong in both directions, and this is the input that shows it:
/// a CSR-only file would have produced a graphless compact output and then
/// failed the audit on unchanged code, while a file carrying one graph of each
/// kind would have kept the family "present" and let the CSR loss through
/// unremarked.
#[test]
fn csr_backed_obsp_is_its_own_family() {
    let dir = tempfile::tempdir().unwrap();
    let input = fixture_with_csr_obsp(dir.path(), "csr_obsp.scx");
    assert!(
        families_of(&input).contains(&SectionFamily::ObspCsr),
        "premise: the fixture must carry a CSR-backed obsp"
    );
    assert!(
        !families_of(&input).contains(&SectionFamily::Obsp),
        "premise: and no COO obsp, or this cannot distinguish the two"
    );

    // compact must SUCCEED (it did before the audit existed) and must be
    // recorded as dropping it.
    let compacted = dir.path().join("compacted.scx");
    scx_ops::compact(&input, &compacted).unwrap();
    assert_matches_table(RewriteOp::Compact, &input, &compacted);
    assert!(!families_of(&compacted).contains(&SectionFamily::ObspCsr));

    // optimize is the one op that keeps it.
    let optimized = dir.path().join("optimized.scx");
    scx_ops::optimize::optimize(&input, &optimized, None, ObsShardPolicy::Off).unwrap();
    assert!(
        families_of(&optimized).contains(&SectionFamily::ObspCsr),
        "optimize re-encodes ObspCsrShard in its shard loop and must keep doing so"
    );
}

/// `merge` takes `varm` from **input 0 only**, so a key present just in a later
/// input is not carried — and the multimodal path omits global varm entirely.
///
/// Declaring that cell `Verbatim` made this merge a hard error, after the output
/// had already been written, on code that had always worked. The audit was
/// right and the table was wrong; this is the case that says so.
///
/// `merge_matches_the_table` cannot see it: both its inputs are copies of the
/// same fixture, so input 0 always has varm.
#[test]
fn merge_succeeds_when_only_a_later_input_has_varm() {
    let dir = tempfile::tempdir().unwrap();
    let with_varm = fixture_all_families(dir.path(), "with_varm.scx");
    let without_varm = fixture_all_families_without_varm(dir.path(), "no_varm.scx");
    assert!(!families_of(&without_varm).contains(&SectionFamily::Varm));

    let out = dir.path().join("merged.scx");
    scx_ops::merge::merge(&[&without_varm, &with_varm], &out)
        .expect("merge must not fail because input 0 lacks varm");

    assert!(
        !families_of(&out).contains(&SectionFamily::Varm),
        "and it really is dropped — merge sources varm from input 0 only, which \
         is why the cell is Conditional rather than Verbatim"
    );
}

/// A pairwise graph scoped to a modality is a different decision from a
/// file-level one, and `compact` treats them differently: it remaps the global
/// graph through its keep-mask and drops the per-modality graphs outright,
/// because the format has no per-modality pairwise reader.
///
/// A file-wide "obsp is remapped" cell made this compact — which had always
/// worked, warning as it dropped — into a hard failure. The scope split closes
/// that direction *and* the mirror one: a file with both a global and a
/// per-modality graph no longer has the global copy vouch for the lost one.
#[test]
fn per_modality_pairwise_is_a_different_decision_from_global() {
    let dir = tempfile::tempdir().unwrap();
    let input = fixture_multimodal_per_modality_obsp(dir.path(), "mm_obsp.scx");
    assert!(
        families_of(&input).contains(&SectionFamily::Obsp),
        "premise: the fixture must carry an obsp graph at all"
    );

    let out = dir.path().join("compacted.scx");
    scx_ops::compact(&input, &out).expect("compact must not fail on per-modality-only obsp");

    assert!(
        !families_of(&out).contains(&SectionFamily::Obsp),
        "and it really is dropped — which is why the per-modality cell is a \
         declared drop rather than the global cell's Remapped"
    );
}
