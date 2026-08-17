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

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Float32Array, Int32Array, Int64Array, RecordBatch, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::reader::ScxReader;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ObsShardPolicy;
use scx_ops::carry::{family, policy, Carry, RewriteOp, SectionFamily};

const N_OBS: usize = 8;
const N_VARS: usize = 6;
/// Two X shards, so `compact` and `sort` have a row layout to actually change.
/// With one shard several ops are trivially identity and the test proves less.
const SHARD_ROWS: usize = 4;
const RAW_N_VARS: usize = 9;

/// Every family the format can carry, in one file.
///
/// Built by hand rather than by running an op, so it does not inherit any op's
/// idea of what a file contains — which is the thing under test.
fn fixture_all_families(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    let mut writer = ScxWriter::new(
        &path,
        FileHeader::new_single_modality(N_OBS as u64, N_VARS as u64, 0, SHARD_ROWS as u32, 0, 0),
    )
    .unwrap();

    // --- obs / var -------------------------------------------------------
    let obs = obs_batch();
    writer.write_obs(&obs).unwrap();
    let var = var_batch(N_VARS, "gene");
    writer.write_var(&var).unwrap();

    // --- X, in two shards ------------------------------------------------
    let mut shard_ranges: Vec<(u64, u64)> = Vec::new();
    for (shard_idx, row_start) in (0..N_OBS).step_by(SHARD_ROWS).enumerate() {
        let (indptr, indices, values) = csr_rows(row_start, SHARD_ROWS, N_VARS, 1);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_start as u64,
            )
            .unwrap();
        shard_ranges.push((row_start as u64, (row_start + SHARD_ROWS) as u64));

        // --- detection bitmaps, keyed to this shard's local rows ---------
        let shard = scx_format_io::bitmap::BitmapShard::build_from_csr(
            row_start as u64,
            SHARD_ROWS as u32,
            N_VARS as u32,
            &indptr,
            &indices,
        );
        writer.write_bitmap_shard(&shard).unwrap();
        let _ = shard_idx;
    }

    // --- a layer ---------------------------------------------------------
    let (l_indptr, l_indices, l_values) = csr_rows(0, N_OBS, N_VARS, 3);
    writer
        .write_layer_csr_shard(
            &l_indptr,
            &l_indices,
            &l_values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
            "spliced",
            0,
        )
        .unwrap();

    // --- obsm / varm -----------------------------------------------------
    writer.write_obsm("X_pca", &dense_embedding(N_OBS)).unwrap();
    writer.write_varm("PCs", &dense_embedding(N_VARS)).unwrap();

    // --- obsp / varp -----------------------------------------------------
    writer
        .write_obsp_shard_coo(
            "connectivities",
            0,
            0,
            N_OBS as u64,
            N_OBS as u64,
            &coo_batch(N_OBS),
        )
        .unwrap();
    writer.write_varp("gene_corr", &coo_i32(N_VARS)).unwrap();

    // --- uns -------------------------------------------------------------
    writer
        .write_uns(&serde_json::json!({ "carry_fixture": true }))
        .unwrap();

    // --- adata.raw (its own, wider var axis) -----------------------------
    writer.set_raw_n_vars(RAW_N_VARS as u64);
    let (r_indptr, r_indices, r_values) = csr_rows(0, N_OBS, RAW_N_VARS, 7);
    writer
        .write_raw_csr_shard(
            &r_indptr,
            &r_indices,
            &r_values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer.write_raw_var(&var_batch(RAW_N_VARS, "raw")).unwrap();

    // --- the grouped-sort sidecar ----------------------------------------
    writer
        .write_group_index(
            serde_json::json!({
                "group_by": "cell_type",
                "reference_shard": 0,
                "reference_labels": ["T cell"],
                "records": [],
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap();

    // --- predicate indexes ------------------------------------------------
    let opts = scx_engine::PredicateIndexBuildOptions {
        forced_columns: vec!["cell_type".to_string()],
        preset_columns: Vec::new(),
        auto_threshold: 0,
        high_cardinality_threshold: 100_000,
    };
    let (mut outcomes, mut cols) = (Vec::new(), Vec::new());
    let obs_bytes = scx_engine::build_obs_predicate_index_bytes(
        &obs,
        &shard_ranges,
        &opts,
        &mut outcomes,
        &mut cols,
    )
    .unwrap()
    .expect("the fixture's cell_type column must produce an obs predicate index");
    writer.write_obs_predicate_index(&obs_bytes).unwrap();

    let var_opts = scx_engine::PredicateIndexBuildOptions {
        forced_columns: vec!["gene_kind".to_string()],
        preset_columns: Vec::new(),
        auto_threshold: 0,
        high_cardinality_threshold: 100_000,
    };
    let (mut v_outcomes, mut v_cols) = (Vec::new(), Vec::new());
    let var_bytes = scx_engine::build_var_predicate_index_bytes(
        &var,
        &[(0, N_VARS as u64)],
        &var_opts,
        &mut v_outcomes,
        &mut v_cols,
    )
    .unwrap()
    .expect("the fixture's gene_kind column must produce a var predicate index");
    writer.write_var_predicate_index(&var_bytes).unwrap();

    writer.finish().unwrap();

    // --- deletion vectors, and the CSC sidecar ----------------------------
    // Both via their real ops: a hand-written deletion vector would not be
    // exercising the same section the ops read, and `build-csc` is the only
    // thing that writes a CSC sidecar.
    scx_ops::mark_deleted(&path, &[2, 5]).unwrap();
    path
}

fn obs_batch() -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("cell_type", DataType::Utf8, true),
            Field::new("n_counts", DataType::UInt32, false),
        ])),
        vec![
            Arc::new(StringArray::from(
                (0..N_OBS).map(|i| format!("cell_{i}")).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                (0..N_OBS)
                    .map(|i| if i % 2 == 0 { "T cell" } else { "B cell" })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(UInt32Array::from(
                (0..N_OBS).map(|i| (100 + i) as u32).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

fn var_batch(n: usize, prefix: &str) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("gene_id", DataType::Utf8, false),
            Field::new("gene_kind", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(StringArray::from(
                (0..n).map(|i| format!("{prefix}_{i}")).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                (0..n)
                    .map(|i| if i % 3 == 0 { "mito" } else { "nuclear" })
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// Two nonzeros per row, deterministic, with `salt` separating X from the layer
/// and from raw so a mixed-up carry shows as wrong values rather than as a pass.
fn csr_rows(
    row_start: usize,
    n_rows: usize,
    n_cols: usize,
    salt: usize,
) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr = vec![0u64];
    let (mut indices, mut values) = (Vec::new(), Vec::new());
    for r in 0..n_rows {
        let row = row_start + r;
        let (c0, c1) = ((row * 2) % n_cols, (row * 2 + 1) % n_cols);
        let (lo, hi) = if c0 <= c1 { (c0, c1) } else { (c1, c0) };
        indices.push(lo as u32);
        indices.push(hi as u32);
        values.push(((row + salt) % 255 + 1) as u8);
        values.push(((row + salt + 1) % 255 + 1) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    (indptr, indices, values)
}

fn dense_embedding(n_rows: usize) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("0", DataType::Float32, false),
            Field::new("1", DataType::Float32, false),
        ])),
        vec![
            Arc::new(Float32Array::from(
                (0..n_rows).map(|i| i as f32).collect::<Vec<_>>(),
            )),
            Arc::new(Float32Array::from(
                (0..n_rows).map(|i| -(i as f32)).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// obs×obs COO in the Int64 form `write_obsp_shard_coo` takes.
fn coo_batch(n: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("row", DataType::Int64, false),
            Field::new("col", DataType::Int64, false),
            Field::new("data", DataType::Float32, false),
        ],
        HashMap::from([
            ("n_rows".to_string(), n.to_string()),
            ("n_cols".to_string(), n.to_string()),
        ]),
    ));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>())),
            Arc::new(Int64Array::from(
                (0..n as i64)
                    .map(|r| (r + 1) % n as i64)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float32Array::from(
                (0..n).map(|i| (i + 1) as f32).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// var×var COO in the Int32 form `write_varp` documents.
fn coo_i32(n: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("row", DataType::Int32, false),
            Field::new("col", DataType::Int32, false),
            Field::new("data", DataType::Float32, false),
        ],
        HashMap::from([
            ("n_rows".to_string(), n.to_string()),
            ("n_cols".to_string(), n.to_string()),
        ]),
    ));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from((0..n as i32).collect::<Vec<_>>())),
            Arc::new(Int32Array::from(
                (0..n as i32)
                    .map(|r| (r + 1) % n as i32)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float32Array::from(
                (0..n).map(|i| (i + 1) as f32).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
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
