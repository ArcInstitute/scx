//! The committed byte-identity A/B matrix: what does each rewriting op write?
//!
//! Sixteen arms over nine ops, one manifest of per-section digests, three ways
//! to use it:
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
//! ⚠️ **The base worktree must already contain this harness.** Cargo has no
//! `op_output_identity` target at a commit that predates it, so the recipe
//! above cannot establish a baseline against an arbitrary merge base — it works
//! from the first commit that carries the harness onward. To A/B against an
//! older commit, cherry-pick **the whole harness commit** onto it
//! (`git cherry-pick 8210f236`) and say in the PR body that you did. Not "this
//! file plus `scx_testkit::ab`": that pair does not build, because this file
//! also needs `common::{appendable_rows, fixture_all_families_without_raw}`
//! and `scx-testkit`'s `pub mod ab;`.
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
//! Read this before citing a green run as "the rewrite did not change bytes".
//! It is true of these seventeen arms and of nothing else.
//!
//! **Regions of the file.** `FileHeader::file_checksum` is deliberately outside
//! the digest (it covers the `Provenance` section, which is itself excluded),
//! so a change to what `file_checksum` *means* passes here untouched and needs
//! its own oracle. So does the 4096-byte root catalog at offset 256, which has
//! no production readers. See `scx_testkit::ab`'s module docs.
//!
//! **Ops not in the matrix.** Every one of these is a provenance-stamping write
//! path that a rewrite refactor can break, and none is pinned here:
//!
//! * **`scx upgrade`** — a full `ScxWriter` rewrite with its own
//!   `RewriteOp::Upgrade` carry-policy row (`scx-ops/src/carry.rs`). Absent for
//!   a structural reason, not an oversight: `run_upgrade` lives in `scx-cli`,
//!   which is a **bin-only crate** with no `lib.rs`, so this crate cannot call
//!   it. Adding an arm means either spawning the `scx` binary or moving the op
//!   into a library crate; both are larger than this file.
//! * **The multimodal branch of `compact` and `merge`.** Both ops are in the
//!   matrix, but `compact_multimodal` / `merge_multimodal` are private arms
//!   `compact` / `merge` dispatch into on a multimodal input, and no fixture
//!   here is multimodal — so those arms run in neither.
//! * **`scx subset`**, **`scx-convert`** in both directions, and
//!   **`pyscx.from_anndata`**.
//! * **The other in-place obs writers, and the streamed obs path.**
//!   `attach_external_obs` is in the matrix over a categorical-obs fixture, but
//!   that fixture's obs is one legacy section, so the arm runs the
//!   *materialising* rewrite (`write_obs_shards_from_whole`); the per-shard
//!   streamed rewrite is pinned only by `scx-ops`'s own tests.
//!   `modify_metadata`, `attach_external_layer` (cellbender) and the
//!   `obs_import` / `doublet_import` producers share that obs writer and are
//!   not digested at all.
//!
//! **Framing, by exactly one arm.** `optimize_framed` is the only arm whose
//! output goes through `scx_format_io::encode_shard_framed`; every other
//! fixture here is unframed, so the fifteen other arms say nothing about the
//! row-group layout. Do not "simplify" its fixture — the row count and the
//! `decode_target` are what make it cover anything, and its own premise
//! assertions explain why.
//!
//! **Layout.** Every arm runs at `Strictness::Content`, which ignores section
//! offsets. A change that only moves sections — a byte-passthrough or an
//! in-place claim — is invisible here; that is what `Strictness::Layout` is for
//! and no arm uses it.
//!
//! ## Relationship to `testkit_against_real_ops.rs`
//!
//! That file asserts the weaker, prior property — that provenance's clock does
//! not leak into an op's digest, i.e. that the harness is *usable* on that op.
//! It compares two runs of the same code. This file compares across commits.

mod common;

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use common::{
    appendable_rows, appendable_rows_categorical, fixture_all_families,
    fixture_all_families_with_categorical_obs, fixture_all_families_without_raw,
    fixture_all_families_without_raw_with_categorical_obs, fixture_with_csr_obsp,
};
use scx_codec::ValueEncoding;
use scx_testkit::ab::{assert_manifests_eq, resolve_against_env, OpDigestManifest};
use scx_testkit::digest::Strictness;
use scx_testkit::fixtures::{
    csc_multi_shard_file, mixed_codec_file, mixed_codec_file_with, FixtureOpts,
};

/// Every arm, in the order the manifest reports them (`labels()` walks a
/// `BTreeMap`, so this constant is asserted **sorted**): the nine labels over
/// the seven ops PR-01 names, with `compact` and `build_csc` doubled over an
/// index-carrying input, plus `attach_obs` (the in-place obs attach, added with
/// the categorical-fidelity change), `attach_var` (its var-axis twin, added
/// with the var attach) so both in-place attaches are pinned from here on,
/// `append_categorical` and `merge_categorical` (the two rewrite ops that write
/// obs the caller can hand them a categorical in, added when they stopped
/// decoding those to plain strings), `sort_categorical_spilled` (the obs
/// spill-scatter write, which is a *different writer* from the `sort` arm's
/// in-memory one), and `optimize_framed` and `optimize_csr_obsp` — seventeen in
/// all.
/// `optimize_framed` is the **only** arm whose output goes through the
/// row-group-framed encoder, and `optimize_csr_obsp` the **only** one whose
/// output carries an `ObspCsrShard`; see their comments in `build_manifest`
/// before changing either fixture.
const EXPECTED_OPS: &[&str] = &[
    "append",
    "append_categorical",
    "attach_obs",
    "attach_var",
    "build_csc",
    "build_csc_indexed",
    "build_csc_multi_shard",
    "compact",
    "compact_indexed",
    "delete",
    "merge",
    "merge_categorical",
    "optimize",
    "optimize_csr_obsp",
    "optimize_framed",
    "sort",
    "sort_categorical_spilled",
];

/// Force a predicate index on one obs column.
///
/// `index_auto_threshold: 0` is the tree-wide "do nothing unless forced"
/// sentinel; the non-empty `index_obs` is what actually makes the pass run.
/// (`forced_columns` is the corresponding field one layer down, on
/// `PredicateIndexBuildOptions` — not this struct.)
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

    // `sort`'s obs **spill-scatter** writer, which the arm above never reaches:
    // that one runs on a plain-`Utf8` obs with no `--memory-budget`, so it takes
    // the in-memory `take` path and has no dictionary to get wrong. This one
    // engages the bounded path (sharded obs + a budget) over a categorical, and
    // pins the bytes it emits for a declared-but-unused level (`"NK cell"`), the
    // declared order, the `scx.categorical.ordered` stamp and the one vocabulary
    // every output shard must share. The spill path used to rebuild all of that
    // from each shard's own rows.
    //
    // Two steps because the spill only engages on an input whose obs is already
    // sharded, and the fixtures write obs as a single section. X is pinned to
    // `InMemory` on the second sort so the arm isolates the obs writer;
    // `strategy_differential_identical` covers the external X path.
    let cat_src = fixture_all_families_with_categorical_obs(dir, "sort_cat_src.scx");
    let sharded = dir.join("sort_cat_sharded.scx");
    let spill_opts = scx_ops::SortOptions {
        by: vec!["cell_type".to_string()],
        shard_target_rows: 2,
        ..Default::default()
    };
    scx_ops::sort_engine::sort(&cat_src, &sharded, &spill_opts).unwrap();
    // Budget derived from the data, not from the machine, so the arm is
    // reproducible: three rows' worth of obs forces multiple spill partitions
    // while still clearing the "one output shard must fit" guard.
    let s0 = scx_format_io::ScxReader::open(&sharded)
        .unwrap()
        .read_obs_shard(0)
        .unwrap();
    let obs_bytes: usize = s0.columns().iter().map(|c| c.get_array_memory_size()).sum();
    let bytes_per_row = (obs_bytes / s0.num_rows().max(1)) as u64;
    let out = dir.join("sort_categorical_spilled.scx");
    let summary = scx_ops::sort_engine::sort_with_strategy(
        &sharded,
        &out,
        &scx_ops::SortOptions {
            memory_budget: Some(bytes_per_row * 3),
            ..spill_opts.clone()
        },
        Some(scx_ops::SortStrategy::InMemory),
    )
    .unwrap();
    assert!(
        summary.obs_spilled && summary.obs_partitions >= 2,
        "the sort_categorical_spilled arm must actually spill obs \
         (spilled={}, partitions={})",
        summary.obs_spilled,
        summary.obs_partitions,
    );
    m.record("sort_categorical_spilled", &out, Strictness::Content)
        .unwrap();

    let out = dir.join("optimize.scx");
    scx_ops::optimize(&src, &out, None, scx_format_io::ObsShardPolicy::Off).unwrap();
    m.record("optimize", &out, Strictness::Content).unwrap();

    // `merge` validates var identity and requires matching layers, so both
    // inputs are the same fixture shape rather than two arbitrary files.
    let y = fixture_all_families(dir, "merge_y.scx");
    let out = dir.join("merge.scx");
    scx_ops::merge(&[&src, &y], &out).unwrap();
    m.record("merge", &out, Strictness::Content).unwrap();

    // The same merge over a fixture whose `cell_type` is a categorical. The
    // arm above cannot see this change at all: `fixture_all_families`' obs is
    // plain `Utf8`, so merge's obs path took its no-dictionary route either
    // way. What this digest pins is that a merged output's obs shards carry
    // the dictionary, its declared levels (`"NK cell"` is declared and unused)
    // and the `scx.categorical.ordered` stamp — the bytes merge used to get
    // wrong by casting every dictionary column to plain strings.
    let cat_x = fixture_all_families_with_categorical_obs(dir, "merge_cat_x.scx");
    let cat_y = fixture_all_families_with_categorical_obs(dir, "merge_cat_y.scx");
    let out = dir.join("merge_categorical.scx");
    scx_ops::merge(&[&cat_x, &cat_y], &out).unwrap();
    m.record("merge_categorical", &out, Strictness::Content)
        .unwrap();

    // --- build-csc, over the three-codec fixture -------------------------
    //
    // Not the all-families fixture, and no longer for the reason this comment
    // used to give: it said build-csc "strips varm/obsp/varp/raw/bitmaps/
    // group-index on the way", which was true before Phase 5b and is false now
    // — those six are all `Carry::Verbatim` (`scx_ops::carry::build_csc`), and
    // `section_carry.rs::build_csc_carries_what_optimize_carries` pins it.
    //
    // The reason that survives is what build-csc's diff actually touches: its
    // per-shard CSR re-emit loop. `mixed_codec_file` is the only fixture in the
    // tree that exercises all three encoder paths (unframed integer,
    // row-group-framed integer, float). `build_csc_indexed` below covers the
    // rich families — all but `.raw`, the bitmaps and the group index, which
    // its input loses to the `compact` it is laundered through, not to
    // build-csc.
    let csc_src = mixed_codec_file(&dir.join("csc_src.scx")).unwrap();
    let out = dir.join("build_csc.scx");
    scx_ops::run_build_csc(&csc_src, &out, "1G", false, 1024, None, None).unwrap();
    m.record("build_csc", &out, Strictness::Content).unwrap();

    // --- build-csc emitting SEVERAL CSC shards ----------------------------
    //
    // The arm above emits exactly **one** CSC shard in one column chunk (1024
    // columns per shard against 40, a 1 GiB budget against 24 rows), so it
    // pins nothing about `col_start` stamping, chunk boundaries, or the
    // concatenation between them. Every byte of multi-shard CSC emission was
    // unpinned until this arm.
    //
    // `cols_per_shard = 3` over 7 columns gives `[0,3) [3,6) [6,7)` — three
    // shards with a **short last one**, which is the boundary an off-by-one in
    // the final chunk lands on. The fixture's irregular geometry (5/1/11-row
    // shards, per-row nnz from 0 to 7, empty columns at both ends and in the
    // middle) is documented on `csc_multi_shard_file`; each irregularity buys
    // one specific failure that uniform geometry hides.
    //
    // The budget is generous on purpose, so the boundaries come from
    // `cols_per_shard` alone and this arm is byte-identical across the
    // streaming-builder change rather than needing a bless.
    let csc_multi = csc_multi_shard_file(&dir.join("csc_multi_src.scx")).unwrap();
    let out = dir.join("build_csc_multi_shard.scx");
    scx_ops::run_build_csc(&csc_multi, &out, "1G", false, 3, None, None).unwrap();
    m.record("build_csc_multi_shard", &out, Strictness::Content)
        .unwrap();

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
    scx_ops::run_build_csc(&indexed, &out, "1G", false, 1024, None, None).unwrap();
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

    // The append twin of `merge_categorical`, and for the same reason: the arm
    // above runs over a plain-`Utf8` obs and so is blind to whether append
    // still decodes categoricals. Both sides of the append are dictionaries
    // here — the target's existing obs and the rows being added — which is what
    // makes the digest cover the *write* rather than the reader's
    // dictionary/plain reconciliation.
    let target = fixture_all_families_without_raw_with_categorical_obs(dir, "append_cat.scx");
    let (new_obs, indptr, indices, values) = appendable_rows_categorical(5);
    scx_ops::append(
        &target,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        &scx_ops::AppendOptions {
            shard_target_rows: NonZeroU32::new(2).unwrap(),
            ..Default::default()
        },
    )
    .unwrap();
    m.record("append_categorical", &target, Strictness::Content)
        .unwrap();

    let target = dir.join("delete.scx");
    std::fs::copy(&src, &target).unwrap();
    // The fixture already carries a deletion vector (rows 2 and 5), so this
    // exercises the merge-into-existing path, not the create path.
    scx_ops::mark_deleted(&target, &[1, 6]).unwrap();
    m.record("delete", &target, Strictness::Content).unwrap();

    // `attach_external_obs` over a fixture whose `cell_type` is a categorical,
    // attaching one plain and one categorical column keyed on `cell_id`. What
    // this digest pins: the target's dictionary column and its `ordered` stamp
    // written through unchanged, and the attached categorical landing as a
    // dictionary — the bytes the in-place writers used to get wrong by casting
    // every dictionary column to plain strings.
    let target = fixture_all_families_with_categorical_obs(dir, "attach_obs.scx");
    scx_ops::attach_external_obs(&target, &attach_obs_payload(), &attach_obs_options()).unwrap();
    m.record("attach_obs", &target, Strictness::Content)
        .unwrap();

    // `attach_external_var` over the same fixture, attaching one plain and one
    // categorical column keyed on `gene_id`. What this digest pins beyond the
    // op's own tests: that a var attach leaves every *other* section's bytes
    // alone — obs (categorical stamp included), X, the layer, the CSC sidecar,
    // `.raw`, `obsm`/`varm`/`obsp`/`varp`, the bitmap and the deletion vector.
    // The manifest is the only place that is checked across all of them at once.
    let target = fixture_all_families_with_categorical_obs(dir, "attach_var.scx");
    scx_ops::attach_external_var(&target, &attach_var_payload(), &attach_var_options()).unwrap();
    m.record("attach_var", &target, Strictness::Content)
        .unwrap();

    // --- the only arm that reaches the row-group-framed encoder ------------
    //
    // Every arm above writes **unframed** shards, and not by accident:
    // `fixture_all_families` never calls `ScxWriter::set_framing`, so
    // `framing_for_rewrite` sees an unframed input and hands the writer
    // `None`; and `mixed_codec_file`'s one framed shard is re-encoded
    // unframed by `run_build_csc(.., None)` above. So before this arm,
    // `scx_format_io::encode_shard_framed` — the function every framed write
    // in the workspace funnels through — was pinned by nothing here.
    //
    // Two properties of the fixture are load-bearing, not incidental:
    //
    // * `n_obs = 1545` gives three shards of 515 rows, so at G = 256 each is
    //   **three row groups with a 3-row tail**. A single-group shard would
    //   pin the framed layout no better than an unframed one — the same trap
    //   `mixed_codec_file` calls out for its own framed shard, and the reason
    //   the default `n_obs = 24` cannot be used here (515 > 256 is the whole
    //   point). A rotation of the groups, a reversal, and an off-by-one on
    //   the short tail are all visible in this digest and in none other.
    // * `decode_target: Some(Auto)` makes the adaptive dual-encode fire on
    //   the two integer shards, so the arm covers the candidate-selection
    //   path as well as the group layout.
    //
    // `optimize_with_framing` rather than `optimize`: framing is an explicit
    // argument there, so the arm cannot be silently un-framed by a change to
    // how another op infers `output_framed` from its input's header.
    let framed_src = mixed_codec_file_with(
        &dir.join("framed_src.scx"),
        &FixtureOpts {
            n_obs: 1545,
            ..Default::default()
        },
    )
    .unwrap();
    let out = dir.join("optimize_framed.scx");
    scx_ops::optimize_with_framing(
        &framed_src,
        &out,
        None,
        scx_format_io::ObsShardPolicy::Off,
        Some(scx_format_io::FramingConfig {
            row_group_rows: 256,
            target_nnz: None,
            trial: false,
            decode_target: Some(scx_format_io::codec_select::DecodeTarget::Auto),
        }),
    )
    .unwrap();
    assert_output_shards_are_multi_group(&out);
    m.record("optimize_framed", &out, Strictness::Content)
        .unwrap();

    // --- the only arm whose output carries a CSR-backed obsp graph --------
    //
    // `optimize` is the one op that keeps an `ObspCsrShard` (`carry.rs` drops
    // it on compact / merge / sort), and `fixture_with_csr_obsp` is the one
    // fixture that writes one — so without this arm the section's on-disk
    // bytes are pinned by nothing at all. That is how OPT-FORMATIO-4 survived
    // a format-level refactor: an obsp shard's minor extent was stamped from
    // the gene axis instead of the cell axis, and all twelve other arms
    // stayed green.
    //
    // The fixture has more cells than genes on purpose, which is what makes
    // the two axes distinguishable in these bytes.
    let obsp_src = fixture_with_csr_obsp(dir, "csr_obsp_src.scx");
    let out = dir.join("optimize_csr_obsp.scx");
    scx_ops::optimize(&obsp_src, &out, None, scx_format_io::ObsShardPolicy::Off).unwrap();
    assert_output_carries_an_obsp_csr_graph(&out);
    m.record("optimize_csr_obsp", &out, Strictness::Content)
        .unwrap();

    m
}

/// The premise `optimize_csr_obsp` rests on: the output really carries a
/// CSR-backed obsp graph, and its minor extent really is the obs axis.
///
/// Asserted rather than assumed for the same reason as
/// `assert_output_shards_are_multi_group`: if `optimize` ever stopped carrying
/// the section, or the fixture stopped writing one, the arm would stay green
/// while covering nothing. The extent check is the narrower claim a digest
/// cannot make — it records that the bytes did not move, not that they were
/// right to begin with.
fn assert_output_carries_an_obsp_csr_graph(path: &Path) {
    let reader = scx_format_io::ScxReader::open(path).unwrap();
    let entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == scx_format_io::section::SectionType::ObspCsrShard)
        .collect();
    assert!(
        !entries.is_empty(),
        "{}: no ObspCsrShard survived optimize — this arm covers nothing",
        path.display()
    );
    let n_obs = reader.header().n_obs;
    assert_ne!(
        n_obs,
        reader.header().n_vars,
        "{}: the two axes must differ, or stamping the wrong one is invisible",
        path.display()
    );
    for entry in entries {
        let header = reader.read_shard_header(entry).unwrap();
        assert_eq!(
            header.n_minor as u64,
            n_obs,
            "{}: {} is obs x obs, so its minor extent is n_obs",
            path.display(),
            entry.name
        );
    }
}

/// The premise `optimize_framed` rests on: its output's CSR shards really are
/// row-group-framed and really do span more than one group.
///
/// Asserted rather than assumed because both halves are silent when they
/// break. A future change to the fixture's `n_obs`, or to how `optimize`
/// decides to frame, would leave the arm green while it quietly stopped
/// covering the group layout — which is the only thing it is there for.
fn assert_output_shards_are_multi_group(path: &Path) {
    let reader = scx_format_io::ScxReader::open(path).unwrap();
    let shards = reader.catalog().csr_shards_sorted();
    assert!(!shards.is_empty(), "no CSR shards in {}", path.display());
    for entry in shards {
        let header = reader.read_shard_header(entry).unwrap();
        assert!(
            header.shard_format_version > scx_format_io::shard::DEFAULT_WRITE_SHARD_FORMAT_VERSION,
            "{}: shard {} is unframed (v{}), so it carries no block index",
            path.display(),
            entry.name,
            header.shard_format_version
        );
        assert!(
            header.n_major as usize > 256,
            "{}: shard {} has {} rows, which is one row group at G=256 — \
             the arm no longer covers the group layout",
            path.display(),
            entry.name,
            header.n_major
        );
    }
}

/// Four of the fixture's genes, in reverse order (a key join, not a positional
/// one), with a float score and an ordered categorical whose declared order is
/// neither alphabetical nor first-appearance and whose third level no row uses.
fn attach_var_payload() -> scx_ops::ExternalVarData {
    use arrow::array::{
        Array, DictionaryArray, Float32Array, Int32Array, RecordBatch, StringArray,
    };
    use arrow::datatypes::{Field, Int32Type, Schema};
    use std::collections::HashMap;
    use std::sync::Arc;

    let row_keys: Vec<String> = (0..4).rev().map(|i| format!("gene_{i}")).collect();
    let score = Float32Array::from((0..4).map(|i| i as f32 * 0.5).collect::<Vec<_>>());
    let class = DictionaryArray::<Int32Type>::try_new(
        Int32Array::from(vec![1, 0, 1, 0]),
        Arc::new(StringArray::from(vec![
            "promoter",
            "enhancer",
            "intergenic",
        ])),
    )
    .unwrap();
    let mut md = HashMap::new();
    md.insert(
        scx_format_io::CATEGORICAL_ORDERED_KEY.to_string(),
        "true".to_string(),
    );
    let schema = Schema::new(vec![
        Field::new("peak_score", score.data_type().clone(), true),
        Field::new("peak_class", class.data_type().clone(), true).with_metadata(md),
    ]);
    scx_ops::ExternalVarData {
        row_keys,
        row_annotations: RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(score), Arc::new(class)],
        )
        .unwrap(),
        uns: serde_json::Map::new(),
        source_checksum: None,
        source_name: Some("peaks.csv".to_string()),
    }
}

fn attach_var_options() -> scx_ops::AttachVarOptions {
    scx_ops::AttachVarOptions {
        join_key: scx_ops::AxisJoinKey::Column("gene_id".to_string()),
        status_column: Some("peak_status".to_string()),
        ..Default::default()
    }
}

/// Six of the fixture's eight cells, in reverse order (a key join, not a
/// positional one), with a float score and an ordered categorical call whose
/// declared order is neither alphabetical nor first-appearance and whose third
/// level no row uses.
fn attach_obs_payload() -> scx_ops::ExternalObsData {
    use arrow::array::{
        Array, DictionaryArray, Float32Array, Int32Array, RecordBatch, StringArray,
    };
    use arrow::datatypes::{Field, Int32Type, Schema};
    use std::collections::HashMap;
    use std::sync::Arc;

    let row_keys: Vec<String> = (0..6).rev().map(|i| format!("cell_{i}")).collect();
    let score = Float32Array::from((0..6).map(|i| i as f32 * 0.25).collect::<Vec<_>>());
    let call = DictionaryArray::<Int32Type>::try_new(
        Int32Array::from(vec![1, 0, 1, 0, 1, 0]),
        Arc::new(StringArray::from(vec!["doublet", "singlet", "unsure"])),
    )
    .unwrap();
    let mut md = HashMap::new();
    md.insert(
        scx_format_io::CATEGORICAL_ORDERED_KEY.to_string(),
        "true".to_string(),
    );
    let schema = Schema::new(vec![
        Field::new("dbl_score", score.data_type().clone(), true),
        Field::new("dbl_call", call.data_type().clone(), true).with_metadata(md),
    ]);
    scx_ops::ExternalObsData {
        row_keys,
        row_annotations: RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(score), Arc::new(call)],
        )
        .unwrap(),
        row_embeddings: Vec::new(),
        uns: serde_json::Map::new(),
        source_checksum: None,
        source_name: Some("calls.csv".to_string()),
    }
}

fn attach_obs_options() -> scx_ops::AttachObsOptions {
    scx_ops::AttachObsOptions {
        join_key: scx_ops::ObsJoinKey::Column("cell_id".to_string()),
        status_column: Some("dbl_status".to_string()),
        ..Default::default()
    }
}

/// The premise: the matrix really does cover every arm it is meant to — the
/// nine over the seven ops PR-01 names, with `compact` and `build_csc` doubled
/// over an index-carrying input, plus the `attach_obs` and `attach_var` arms.
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
