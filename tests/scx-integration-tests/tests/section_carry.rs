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
//! ## What used to be pinned here as a known bug
//!
//! Until Phase 5b this file asserted, in `known_open_drops_are_still_dropping`,
//! that `merge` loses obsp/varp and `build-csc` loses varm/obsp/varp/`.raw`/
//! bitmaps/the group index — §6.4 and §6.3 of the 2026-08-05 review, both open
//! Majors, both asserted **as drops** because that is what the code did.
//!
//! Those assertions are now inverted, and the test that inverts them
//! (`merge_carries_the_graph_it_used_to_drop`,
//! `build_csc_carries_what_optimize_carries`) is deliberately spelled out from
//! the user's side rather than folded into the table walk, so a reader can see
//! what changed without reconstructing it from `scx_ops::carry`.
//!
//! A test that encodes a known bug has to say so in the test — that is why the
//! inversion was mechanical. `append`'s categorical tests are the cautionary
//! case: they assert `Utf8` because that is what `append` produces, and nothing
//! in them records that `Dictionary` is what it *should* produce, so they read
//! as a specification for the bug rather than a note against it.

mod common;

use std::collections::BTreeSet;
use std::path::Path;

use scx_format_io::reader::ScxReader;
use scx_format_io::ObsShardPolicy;
use scx_ops::carry::{family, policy, Carry, RewriteOp, SectionFamily};

use common::{
    fixture_all_families, fixture_all_families_obsp, fixture_all_families_with_extra_obsm,
    fixture_all_families_without_varm, fixture_multimodal_both_obsp_scopes,
    fixture_multimodal_per_modality_obsp, fixture_with_colliding_obsm_keys,
    fixture_with_colliding_obsp_keys, fixture_with_csr_obsp, fixture_with_explicit_zero_in_x,
    fixture_with_sharded_obsm,
};

/// The `(row, col)` endpoints of a COO pairwise batch, widened to `i64`.
///
/// The coordinate width is `Int32` or `Int64` depending on how the section was
/// written and how wide the axis is, and a remap picks the output width from the
/// *new* dimension — so a test that assumed one width would pass or fail for
/// reasons unrelated to what it is checking.
fn coo_endpoints(batch: &arrow::array::RecordBatch) -> (Vec<i64>, Vec<i64>) {
    use arrow::array::{Array, Int32Array, Int64Array};
    let col = |name: &str| -> Vec<i64> {
        let a = batch.column_by_name(name).unwrap();
        match a.data_type() {
            arrow::datatypes::DataType::Int32 => a
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .iter()
                .map(|v| v.unwrap() as i64)
                .collect(),
            arrow::datatypes::DataType::Int64 => a
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
                .map(|v| v.unwrap())
                .collect(),
            other => panic!("unexpected COO coordinate type {other:?}"),
        }
    };
    (col("row"), col("col"))
}

/// Every `(gene, global row)` pair the file's detection bitmaps claim.
fn bitmap_detections(path: &Path) -> BTreeSet<(u32, u64)> {
    let reader = ScxReader::open(path).unwrap();
    let mut out = BTreeSet::new();
    let n = reader.catalog().bitmap_shards_for_modality(0).len();
    for idx in 0..n {
        let shard = reader.read_bitmap_shard(idx).unwrap();
        for (&gene, rows) in &shard.genes {
            for local in rows.iter() {
                out.insert((gene, shard.row_start + local as u64));
            }
        }
    }
    out
}

/// Every `(gene, global row)` pair the file's X shards actually store.
fn x_detections(path: &Path) -> BTreeSet<(u32, u64)> {
    let reader = ScxReader::open(path).unwrap();
    let mut out = BTreeSet::new();
    for entry in reader.catalog().csr_shards_sorted() {
        let row_start = entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
        let (indptr, indices, _) = reader.read_shard_from_entry(entry).unwrap();
        for row in 0..indptr.len() - 1 {
            for &gene in &indices[indptr[row] as usize..indptr[row + 1] as usize] {
                out.insert((gene as u32, row_start + row as u64));
            }
        }
    }
    out
}

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
    scx_ops::run_build_csc(&input, &output, "1G", false, 1024, None, None).unwrap();
    assert_matches_table(RewriteOp::BuildCsc, &input, &output);

    // The op's whole purpose, and the one family it *adds*.
    assert!(families_of(&output).contains(&SectionFamily::XCsc));
}

/// §6.4, from the user's side: a merged file keeps its cell–cell graph.
///
/// This is the inversion of `known_open_drops_are_still_dropping`. Before
/// Phase 5b `merge` had no obsp or varp writer at all — only a comment reading
/// "obsp dropped (as plain merge does)" — so merging per-sample files after
/// computing kNN silently produced an atlas with no graph, while `compact`
/// remapped one through its keep-mask and `sort` through its permutation.
///
/// The `data` values are checked, not just the family: obsp is remapped by each
/// input's global row offset, and an off-by-one offset produces a graph that is
/// present, well-formed, and wrong.
#[test]
fn merge_carries_the_graph_it_used_to_drop() {
    let dir = tempfile::tempdir().unwrap();
    let a = fixture_all_families(dir.path(), "a.scx");
    let b = fixture_all_families(dir.path(), "b.scx");
    let merged = dir.path().join("merged.scx");
    scx_ops::merge::merge(&[&a, &b], &merged).unwrap();

    let after = families_of(&merged);
    assert!(
        after.contains(&SectionFamily::Obsp),
        "merge must carry obsp (§6.4)"
    );
    assert!(
        after.contains(&SectionFamily::Varp),
        "merge must carry varp (§6.4)"
    );

    // Each input contributes its own edges, rebased into the concatenated row
    // space. Both inputs are the same fixture, so the merged graph is the input
    // graph twice over — once at offset 0, once at the offset of input 1.
    let reader = ScxReader::open(&merged).unwrap();
    let obsp = reader.read_all_obsp().unwrap();
    let graph = obsp
        .get("connectivities")
        .expect("the merged file must still name the graph 'connectivities'");
    let (rows, cols) = coo_endpoints(graph);
    assert_eq!(
        rows.len(),
        2 * common::N_OBS,
        "both inputs' edges must survive, not just input 0's"
    );

    // `merge` *carries* deletion vectors rather than applying them (`compact`
    // is the op that materialises a delete), so every physical row survives and
    // input 1's offset is exactly `N_OBS`. Asserted exactly rather than as
    // "some offset > 0": an off-by-one rebase produces a graph that is present,
    // well-formed, and wrong, and only the exact value catches it.
    assert_eq!(reader.header().n_obs, 2 * common::N_OBS as u64);
    let mut by_block = [0usize, 0usize];
    for (&r, &c) in rows.iter().zip(cols.iter()) {
        assert!(
            (r as u64) < reader.header().n_obs && (c as u64) < reader.header().n_obs,
            "every remapped endpoint must land inside the merged obs axis \
             (got row={r}, col={c}, n_obs={})",
            reader.header().n_obs
        );
        by_block[(r as usize) / common::N_OBS] += 1;
    }
    assert_eq!(
        by_block,
        [common::N_OBS, common::N_OBS],
        "each input must contribute its edges to its own half of the merged axis"
    );
    // And the second block really is offset, rather than a second copy of the
    // first: input 1's rows all start at N_OBS.
    assert_eq!(
        rows.iter().filter(|&&r| r >= common::N_OBS as i64).count(),
        common::N_OBS
    );
}

/// §6.3, from the user's side: `build-csc` no longer loses what `optimize`
/// keeps.
///
/// The other half of the inversion. While `build-csc --in-place` renamed a wholly
/// new file over the target carrying no prior catalog, `scx rollback` could not
/// recover any of this — which is what made the narrowest carry allowlist in the
/// crate the most dangerous one. It is an append now and carries everything by
/// construction; this stays as the end-to-end pin.
#[test]
fn build_csc_carries_what_optimize_carries() {
    let dir = tempfile::tempdir().unwrap();
    let input = fixture_all_families(dir.path(), "in.scx");

    let with_csc = dir.path().join("with_csc.scx");
    scx_ops::run_build_csc(&input, &with_csc, "1G", false, 1024, None, None).unwrap();
    let after_csc = families_of(&with_csc);

    let optimized = dir.path().join("optimized.scx");
    scx_ops::optimize::optimize(&input, &optimized, None, ObsShardPolicy::Off).unwrap();
    let after_opt = families_of(&optimized);

    for f in [
        SectionFamily::Varm,
        SectionFamily::Obsp,
        SectionFamily::Varp,
        SectionFamily::Raw,
        SectionFamily::Bitmap,
        SectionFamily::GroupIndex,
    ] {
        assert!(
            after_csc.contains(&f),
            "build-csc must carry {} (§6.3)",
            f.label()
        );
    }

    // `.raw` is the one family in that list `optimize` still does not carry, so
    // the two sets are compared with it excluded rather than asserted equal —
    // saying so here keeps the next reader from "fixing" the difference.
    let opt_should_carry = [
        SectionFamily::Varm,
        SectionFamily::Obsp,
        SectionFamily::Varp,
        SectionFamily::Bitmap,
        SectionFamily::GroupIndex,
    ];
    for f in opt_should_carry {
        assert!(
            after_opt.contains(&f),
            "optimize carries {} and must keep doing so",
            f.label()
        );
    }

    // Presence in the catalog is not the property. A section copied verbatim
    // into a file whose header or stats no longer describe it is present and
    // useless, and `.raw` is the case where that was a live question: nothing
    // sets a raw column count on the output, so it has to come back from the
    // shards' own stats.
    let src = ScxReader::open(&input).unwrap();
    let out = ScxReader::open(&with_csc).unwrap();
    assert_eq!(
        out.raw_n_vars(),
        src.raw_n_vars(),
        "raw's own var extent must survive the copy, not just its sections"
    );
    assert_eq!(
        out.read_raw_var().unwrap().num_rows(),
        src.read_raw_var().unwrap().num_rows()
    );
    assert_eq!(
        out.read_all_varm().unwrap().keys().collect::<BTreeSet<_>>(),
        src.read_all_varm().unwrap().keys().collect::<BTreeSet<_>>()
    );
    assert_eq!(
        out.read_all_obsp().unwrap().keys().collect::<BTreeSet<_>>(),
        src.read_all_obsp().unwrap().keys().collect::<BTreeSet<_>>()
    );
    assert_eq!(
        out.read_all_varp().unwrap().keys().collect::<BTreeSet<_>>(),
        src.read_all_varp().unwrap().keys().collect::<BTreeSet<_>>()
    );
    // build-csc does not canonicalise, so the bitmaps describe an unchanged
    // matrix and must agree with it exactly.
    assert_eq!(bitmap_detections(&with_csc), x_detections(&with_csc));
}

/// Merge takes its obsm key set from **input 0 only**, so a key that only a
/// later input has is never considered at all.
///
/// Before Phase 5b this merge succeeded and produced a file with no `X_umap`,
/// with nothing said. A *layer* in the same position has always been a hard
/// `OpsError::LayerMissing` naming the file index; this is that parity.
#[test]
fn merge_rejects_a_key_only_a_later_input_has() {
    let dir = tempfile::tempdir().unwrap();
    let plain = fixture_all_families(dir.path(), "plain.scx");
    let extra = fixture_all_families_with_extra_obsm(dir.path(), "extra.scx", "X_umap");

    let out = dir.path().join("merged.scx");
    let err = scx_ops::merge::merge(&[&plain, &extra], &out)
        .expect_err("a key only input 1 carries must not be silently invisible");
    let msg = err.to_string();
    assert!(
        msg.contains("X_umap") && msg.contains("obsm"),
        "the error must name the axis and the key, got: {msg}"
    );
    assert!(
        msg.contains('0'),
        "and the input that lacks it, as LayerMissing does, got: {msg}"
    );
}

/// The mirror case: input 0 has the key and a later input does not.
///
/// This is the one the review describes — "merging 100 per-sample files where
/// one lacks `X_umap` silently yields an atlas with no UMAP". It was a
/// `continue 'next_key` with no diagnostic.
#[test]
fn merge_rejects_a_key_a_later_input_lacks() {
    let dir = tempfile::tempdir().unwrap();
    let extra = fixture_all_families_with_extra_obsm(dir.path(), "extra.scx", "X_umap");
    let plain = fixture_all_families(dir.path(), "plain.scx");

    let out = dir.path().join("merged.scx");
    let err = scx_ops::merge::merge(&[&extra, &plain], &out)
        .expect_err("a key input 1 lacks must not be dropped without a word");
    let msg = err.to_string();
    assert!(
        msg.contains("X_umap") && msg.contains("obsm"),
        "the error must name the axis and the key, got: {msg}"
    );
    assert!(msg.contains('1'), "and the input that lacks it, got: {msg}");
}

/// The same key-set rule applies to `obsp`, and it has to be pinned separately.
///
/// `merge_rejects_a_key_*` above cover `obsm`, which goes through
/// `merge_global_dense_mapping_sharded`; `obsp` goes through `merge_obsp`,
/// which is a different function with its own copy of the presence check. A
/// regression in one would not show up in the other's tests, and this is the
/// exact `[has_knn, no_knn]` case the choice was argued over.
///
/// Requested by Cursor Agent - Grok 4.6 High.
#[test]
fn merge_rejects_an_obsp_key_a_later_input_lacks() {
    let dir = tempfile::tempdir().unwrap();
    let knn = fixture_all_families_obsp(dir.path(), "knn.scx", "connectivities", false);
    let other = fixture_all_families_obsp(dir.path(), "other.scx", "distances", false);

    let out = dir.path().join("merged.scx");
    let err = scx_ops::merge::merge(&[&knn, &other], &out)
        .expect_err("a graph input 1 lacks must not be dropped without a word");
    let msg = err.to_string();
    assert!(
        msg.contains("obsp") && msg.contains("connectivities"),
        "the error must name the axis and the key, got: {msg}"
    );
}

/// A graph whose `data` column disagrees across inputs must be refused **before**
/// anything is written, not discovered on read.
///
/// `remap_obsp_coo_to_dim` deliberately preserves each input's `data` dtype and
/// nullability (it rebuilds only `row`/`col`, at a width chosen from the merged
/// dimension). So a `Float32` graph and a `Float64` graph — each perfectly
/// readable alone — become shards under one key that `read_all_obsp`'s
/// `concat_batches` cannot assemble. Measured, with the guard removed:
/// `"It is not possible to concatenate arrays of different data types (Float32,
/// Float64)"` — the merge succeeded and produced a file whose graph could not
/// be read back.
///
/// The obsm path has had this guard since Phase 3b; the obsp carry shipped with
/// the missing-key half of that parity and none of the schema half.
///
/// Found independently by codex - gpt-5.6-sol and Cursor Agent - Grok 4.6 High.
#[test]
fn merge_rejects_an_obsp_graph_whose_value_dtype_disagrees() {
    let dir = tempfile::tempdir().unwrap();
    let f32_side = fixture_all_families_obsp(dir.path(), "f32.scx", "connectivities", false);
    let f64_side = fixture_all_families_obsp(dir.path(), "f64.scx", "connectivities", true);

    // Premise: each input is individually readable, so the failure below is
    // about the *combination* and not about a broken fixture.
    for p in [&f32_side, &f64_side] {
        assert!(ScxReader::open(p)
            .unwrap()
            .read_all_obsp()
            .unwrap()
            .contains_key("connectivities"));
    }

    let out = dir.path().join("merged.scx");
    let err = scx_ops::merge::merge(&[&f32_side, &f64_side], &out)
        .expect_err("a dtype-mismatched graph must be refused before writing");
    let msg = err.to_string();
    assert!(
        msg.contains("obsp") && msg.contains("connectivities"),
        "the error must name the axis and the key, got: {msg}"
    );
    assert!(
        msg.contains("Float32") && msg.contains("Float64"),
        "and both dtypes, so the user knows which side to cast: {msg}"
    );
}

/// Merging files that agree on their obsm keys stays a merge.
///
/// The accept side of the two guards above. Without it, "reject a mismatched
/// key set" is indistinguishable from "reject every obsm", and the tests above
/// would still pass if merge simply stopped carrying obsm at all.
#[test]
fn merge_accepts_a_key_every_input_has() {
    let dir = tempfile::tempdir().unwrap();
    let a = fixture_all_families_with_extra_obsm(dir.path(), "a.scx", "X_umap");
    let b = fixture_all_families_with_extra_obsm(dir.path(), "b.scx", "X_umap");

    let out = dir.path().join("merged.scx");
    scx_ops::merge::merge(&[&a, &b], &out).expect("identical key sets must merge");

    let reader = ScxReader::open(&out).unwrap();
    let keys: BTreeSet<String> = reader.read_all_obsm().unwrap().into_keys().collect();
    assert!(
        keys.contains("X_umap") && keys.contains("X_pca"),
        "both keys must survive, got: {keys:?}"
    );
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

/// The same scope split, for `merge` — the trap Phase 5b walks into.
///
/// Giving `merge` a **global** obsp carry makes the per-modality case inherit
/// it, and `merge_multimodal` has no per-modality pairwise reader either. So a
/// multimodal merge of a file whose only graph is modality-scoped would write
/// its whole output and then hard-fail the audit, on a merge that had always
/// worked. That is the exact failure the scope split was added for in 5a, and
/// the exact one a new carry re-opens if it forgets the override.
#[test]
fn merge_drops_per_modality_pairwise() {
    let dir = tempfile::tempdir().unwrap();
    let a = fixture_multimodal_per_modality_obsp(dir.path(), "a.scx");
    let b = fixture_multimodal_per_modality_obsp(dir.path(), "b.scx");
    assert!(
        families_of(&a).contains(&SectionFamily::Obsp),
        "premise: the fixture must carry a modality-scoped obsp"
    );

    let out = dir.path().join("merged.scx");
    scx_ops::merge::merge(&[&a, &b], &out).expect("merge must not fail on per-modality-only obsp");

    assert!(
        !families_of(&out).contains(&SectionFamily::Obsp),
        "and it really is dropped — merge has no per-modality pairwise reader, \
         so the per-modality cell is a declared drop, not the global Remapped"
    );
}

/// Both scopes at once, which is the only input that can tell the two apart.
///
/// `merge_drops_per_modality_pairwise` above uses a fixture whose *only* graph
/// is modality-scoped, so it cannot distinguish "the per-modality drop is
/// correct" from "merge carries no obsp on a multimodal file at all" — and
/// nothing else exercised the multimodal path's *global* obsp carry, which
/// `merge_obsp_only` exists for.
///
/// It is also the case where a scope-blind audit is most wrong: the surviving
/// global copy keeps the family "present" and vouches for the lost
/// modality-scoped one, so the loss passes unremarked. That is the second half
/// of the finding Phase 5a's review round 2 made.
#[test]
fn merge_keeps_the_global_graph_and_drops_the_modality_scoped_one() {
    let dir = tempfile::tempdir().unwrap();
    let a = fixture_multimodal_both_obsp_scopes(dir.path(), "a.scx");
    let b = fixture_multimodal_both_obsp_scopes(dir.path(), "b.scx");

    let scoped_keys = |p: &Path| -> (BTreeSet<String>, BTreeSet<String>) {
        let reader = ScxReader::open(p).unwrap();
        let mut global = BTreeSet::new();
        let mut per_modality = BTreeSet::new();
        for e in &reader.catalog().entries {
            if family(e.section_type) != SectionFamily::Obsp {
                continue;
            }
            let target = if e.modality_id == 0 {
                &mut global
            } else {
                &mut per_modality
            };
            target.insert(e.name.clone());
        }
        (global, per_modality)
    };

    let (in_global, in_scoped) = scoped_keys(&a);
    assert_eq!(in_global.len(), 1, "premise: one file-scope graph");
    assert_eq!(in_scoped.len(), 1, "premise: and one modality-scoped graph");

    let out = dir.path().join("merged.scx");
    scx_ops::merge::merge(&[&a, &b], &out).unwrap();

    let (out_global, out_scoped) = scoped_keys(&out);
    assert!(
        !out_global.is_empty(),
        "the file-scope graph must survive a multimodal merge — obs is shared \
         across modalities, so it is well defined and rebased like everything \
         else on that axis"
    );
    assert!(
        out_scoped.is_empty(),
        "and the modality-scoped one must not: there is no per-modality pairwise \
         reader to round-trip it through (got {out_scoped:?})"
    );
}

/// A row-sharded `obsm` must come out of `build-csc` still sharded.
///
/// Not part of §6.3, and not something this PR set out to change. The old
/// `copy_obsm` went `read_all_obsm()` → `write_obsm(name, batch)`, and
/// `read_all_obsm` *assembles* shards — so `build-csc` on an atlas with sharded
/// `obsm` silently rebuilt it as one legacy `ObsmEmbedding` section. That is the
/// `obsm` twin of review §4.5, which was about `obs` and was fixed by
/// `copy_obs_var_preserving_layout`. Moving `obsm` onto the same verbatim copy
/// as the rest of the dense mappings fixes it; this pins the fix, since nothing
/// else in the suite would notice it regressing.
#[test]
fn build_csc_keeps_obsm_sharded() {
    use scx_format_io::SectionType;

    let dir = tempfile::tempdir().unwrap();
    let input = fixture_with_sharded_obsm(dir.path(), "sharded_obsm.scx");
    let count = |p: &Path, ty: SectionType| {
        ScxReader::open(p)
            .unwrap()
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == ty)
            .count()
    };
    assert_eq!(
        count(&input, SectionType::ObsmEmbeddingShard),
        2,
        "premise: the fixture's obsm must actually be sharded"
    );

    let output = dir.path().join("with_csc.scx");
    scx_ops::run_build_csc(&input, &output, "1G", false, 1024, None, None).unwrap();

    assert_eq!(
        count(&output, SectionType::ObsmEmbeddingShard),
        2,
        "build-csc must preserve the sharded obsm layout, not collapse it"
    );
    assert_eq!(
        count(&output, SectionType::ObsmEmbedding),
        0,
        "and must not have rebuilt it as one legacy section"
    );
    // The values still round-trip through the assembling reader.
    assert_eq!(
        ScxReader::open(&output)
            .unwrap()
            .read_obsm("X_pca")
            .unwrap()
            .num_rows(),
        common::N_OBS
    );
}

/// A detection bitmap must describe the matrix it ships with.
///
/// `optimize` copied bitmaps verbatim on the strength of a comment claiming
/// `canonicalize_csr` leaves the per-row expressed-gene set unchanged. It does
/// not: `canonicalize_csr` calls `drop_explicit_zeros_inplace`, and
/// `BitmapShard::build_from_csr` keys off the stored index regardless of its
/// value. So an input holding an explicit zero came out of `scx optimize` with a
/// bitmap claiming a gene that the output's own X no longer stores, and
/// `Experiment.detection_counts` over-reported it with nothing to notice by.
///
/// Asserted as agreement with the output's X rather than as a family drop,
/// because agreement is the property; whether the fix drops the sidecar or
/// rebuilds it is the table's business, not this test's.
#[test]
fn optimize_bitmaps_describe_the_matrix_they_ship_with() {
    let dir = tempfile::tempdir().unwrap();
    let input = fixture_with_explicit_zero_in_x(dir.path(), "zeroed.scx");

    // Premise: the input is the interesting shape — a bitmap that over-reports
    // relative to a canonical reading of its own X. Without this the test would
    // pass on a fixture that never had an explicit zero.
    assert!(
        bitmap_detections(&input).len() > x_canonical_detections(&input).len(),
        "premise: the fixture's bitmap must claim a gene canonicalisation removes"
    );

    let out = dir.path().join("optimized.scx");
    scx_ops::optimize::optimize(&input, &out, None, ObsShardPolicy::Off).unwrap();

    if families_of(&out).contains(&SectionFamily::Bitmap) {
        assert_eq!(
            bitmap_detections(&out),
            x_detections(&out),
            "a carried bitmap must agree with the X it was carried alongside"
        );
    }
}

/// The accept side: a file canonicalisation does not touch keeps its bitmaps.
///
/// Without this, "drop the bitmap when canonicalisation rewrote a shard" is
/// indistinguishable from "drop the bitmap", and the guard above would still
/// pass if `optimize` simply stopped carrying detection sidecars at all — which
/// would be a silent regression against §6.13 and against `optimize`'s whole
/// reason for carrying more than any other op.
#[test]
fn optimize_keeps_bitmaps_when_canonicalisation_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let input = fixture_all_families(dir.path(), "clean.scx");
    assert_eq!(
        bitmap_detections(&input),
        x_detections(&input),
        "premise: this fixture's X is already canonical, so nothing is dropped"
    );

    let out = dir.path().join("optimized.scx");
    scx_ops::optimize::optimize(&input, &out, None, ObsShardPolicy::Off).unwrap();

    assert!(
        families_of(&out).contains(&SectionFamily::Bitmap),
        "optimize must still carry bitmaps through a rewrite that changes nothing"
    );
    assert_eq!(bitmap_detections(&out), x_detections(&out));
}

/// What the file's X would detect **after** canonicalisation — i.e. ignoring
/// stored zeros. Only used for the premise assertion above, where the point is
/// that the on-disk bitmap and this disagree.
fn x_canonical_detections(path: &Path) -> BTreeSet<(u32, u64)> {
    let reader = ScxReader::open(path).unwrap();
    let mut out = BTreeSet::new();
    for entry in reader.catalog().csr_shards_sorted() {
        let row_start = entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
        let (indptr, indices, values) = reader.read_shard_from_entry(entry).unwrap();
        for row in 0..indptr.len() - 1 {
            for i in indptr[row] as usize..indptr[row + 1] as usize {
                if values[i] != 0.0 {
                    out.insert((indices[i] as u32, row_start + row as u64));
                }
            }
        }
    }
    out
}

/// A key whose name is a prefix of another key's shard naming must not swallow
/// that other key's shards.
///
/// `pairwise_entries` matched on `name.starts_with("obsp/<key>_shard_")` alone,
/// so looking up `g` also collected `obsp/g_shard_x_shard_0` — a shard of the
/// separate key `g_shard_x`. Both landed under `g` with overlapping row covers,
/// and the merged graph failed on read. `read_sharded_layout_by_prefix` has
/// always parsed the suffix as a `u32`; this now matches it.
///
/// Found by codex - gpt-5.6-sol.
#[test]
fn merge_does_not_conflate_keys_that_share_a_shard_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let a = fixture_with_colliding_obsp_keys(dir.path(), "a.scx");
    let b = fixture_with_colliding_obsp_keys(dir.path(), "b.scx");

    // Premise: both keys really are present and independently readable.
    let src = ScxReader::open(&a).unwrap().read_all_obsp().unwrap();
    assert!(
        src.contains_key("g") && src.contains_key("g_shard_x"),
        "premise: {:?}",
        src.keys()
    );

    let out = dir.path().join("merged.scx");
    scx_ops::merge::merge(&[&a, &b], &out).unwrap();

    let merged = ScxReader::open(&out).unwrap().read_all_obsp().unwrap();
    assert!(
        merged.contains_key("g") && merged.contains_key("g_shard_x"),
        "both keys must survive as themselves, got {:?}",
        merged.keys()
    );
    // Two inputs x one shard each. If `g` had swallowed `g_shard_x`'s shard it
    // would carry twice that many edges — and the row cover would not assemble.
    assert_eq!(merged["g"].num_rows(), 2 * common::N_OBS);
    assert_eq!(merged["g_shard_x"].num_rows(), 2 * common::N_OBS);
}

/// The obsm half of the shard-prefix collision, and the legacy key the union
/// used to drop.
///
/// `merge_pairwise::pairwise_entries` was fixed in round 1; the obsm/varm
/// helper resolves its shards through `dense_mapping_shards_sorted`, which
/// settles membership with `starts_with` and therefore had the identical bug —
/// looking up `g` also collects `obsm/g_shard_x_shard_0`. Merge would succeed
/// and `read_obsm` would then reject the oversized cover.
///
/// The same fixture carries `legacy_shard_name`, a legacy single section whose
/// key contains `_shard_`. The key scan excluded those, so it was dropped
/// silently — which contradicted this PR's own claim that keys are the union
/// across inputs.
///
/// Found by codex - gpt-5.6-sol.
#[test]
fn merge_does_not_conflate_obsm_keys_that_share_a_shard_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let a = fixture_with_colliding_obsm_keys(dir.path(), "a.scx");
    let b = fixture_with_colliding_obsm_keys(dir.path(), "b.scx");

    let src = ScxReader::open(&a).unwrap().read_all_obsm().unwrap();
    for k in ["g", "g_shard_x", "legacy_shard_name"] {
        assert!(
            src.contains_key(k),
            "premise: input must carry {k}, got {:?}",
            src.keys()
        );
    }

    let out = dir.path().join("merged.scx");
    scx_ops::merge::merge(&[&a, &b], &out).unwrap();

    let merged = ScxReader::open(&out).unwrap().read_all_obsm().unwrap();
    for k in ["g", "g_shard_x", "legacy_shard_name"] {
        assert!(
            merged.contains_key(k),
            "{k} must survive, got {:?}",
            merged.keys()
        );
        // Two inputs. If `g` had swallowed `g_shard_x`'s shards it would carry
        // twice this many rows — and the cover would not assemble at all.
        assert_eq!(
            merged[k].num_rows(),
            2 * common::N_OBS,
            "{k} must have exactly both inputs' rows"
        );
    }
}
