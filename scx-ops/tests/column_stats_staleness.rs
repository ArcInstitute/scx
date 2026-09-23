//! Per-shard catalog `column_stats` must never outlive the obs values they
//! describe.
//!
//! The catalog carries a `MinMax` / `CategoryBitset` per indexed obs column per
//! CSR shard, and `scx_engine::pushdown` prunes a shard from those stats
//! **without consulting the predicate index at all** — it gates only on
//! `!stats.column_stats.is_empty()`. So an in-place op that replaces obs values
//! and leaves the stats behind does not merely lose pushdown: it makes the query
//! engine exclude shards whose rows now match, and the caller gets a short row
//! set with no error and no warning.
//!
//! Every assertion here is against ground truth computed from the obs frame the
//! op wrote, never against a second query path — two paths agreeing proves
//! agreement, not correctness, and the stale stats are consulted by both.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{BooleanArray, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_engine::{ConversionPredicateIndexOptions, QueryPipeline};
use scx_format_io::catalog::ColumnStat;
use scx_format_io::header::FileHeader;
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;
use scx_ops::{modify_metadata, MetadataPatch};
use tempfile::TempDir;

/// The review's fixture: 400 cells over 4 CSR shards of 100.
const N_OBS: usize = 400;
const N_VARS: usize = 8;
const SHARD_ROWS: usize = 100;

/// `n_counts` at row `i`, before the rescale: 100…499.
fn n_counts_before(i: usize) -> i64 {
    100 + i as i64
}

/// …and after: 1000…4990. Ten times larger, so every shard's stale `max`
/// (≤ 499) sits below a predicate that the new values clear easily.
fn n_counts_after(i: usize) -> i64 {
    n_counts_before(i) * 10
}

/// `passed_qc` is `Boolean`, which the predicate-index builder rejects as an
/// unsupported dtype — the lever the `obs_bytes == None` case below pulls.
fn obs_frame(value_at: fn(usize) -> i64) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("n_counts", DataType::Int64, false),
        Field::new("passed_qc", DataType::Boolean, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                (0..N_OBS).map(|i| format!("cell_{i}")).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                (0..N_OBS).map(value_at).collect::<Vec<_>>(),
            )),
            Arc::new(BooleanArray::from(
                (0..N_OBS).map(|i| i % 3 != 0).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

fn var_frame() -> RecordBatch {
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            (0..N_VARS).map(|i| format!("g{i}")).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

/// One nonzero per row so the shards are non-degenerate; the values are
/// irrelevant to pushdown, which reads obs stats only.
fn shard_csr(n_rows: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let indptr: Vec<u64> = (0..=n_rows as u64).collect();
    let indices: Vec<u32> = (0..n_rows).map(|r| (r % N_VARS) as u32).collect();
    let values: Vec<u8> = (0..n_rows).map(|r| ((r % 255) + 1) as u8).collect();
    (indptr, indices, values)
}

/// A file with four CSR shards, an obs predicate index over `n_counts`, and —
/// crucially — the per-shard `column_stats` that index implies. Without the
/// `apply_obs_shard_column_stats` call the fixture would be blind to this whole
/// class of bug, which is exactly why the pre-existing index fixtures in
/// `external_obs_tests.rs` never caught it.
/// Like [`write_indexed_fixture`] but **without** the per-shard `column_stats`:
/// the index section is written and `apply_obs_shard_column_stats` is not called.
///
/// That is exactly the on-disk shape `modify_metadata` leaves when it falls back
/// to building the index over obs-shard ranges — it writes the index and then
/// clears the stats, because the index's shard space is not the CSR one.
fn write_indexed_fixture_without_stats(dir: &TempDir, name: &str) -> PathBuf {
    let path = dir.path().join(name);
    let header =
        FileHeader::new_single_modality(N_OBS as u64, N_VARS as u64, 0, SHARD_ROWS as u32, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    let obs = obs_frame(n_counts_before);
    writer.write_obs(&obs).unwrap();
    writer.write_var(&var_frame()).unwrap();

    let mut ranges: Vec<(u64, u64)> = Vec::new();
    for start in (0..N_OBS).step_by(SHARD_ROWS) {
        let (indptr, indices, values) = shard_csr(SHARD_ROWS);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                start as u64,
            )
            .unwrap();
        ranges.push((start as u64, (start + SHARD_ROWS) as u64));
    }

    let opts = scx_engine::PredicateIndexBuildOptions {
        forced_columns: vec!["n_counts".to_string()],
        preset_columns: Vec::new(),
        auto_threshold: 1000,
        high_cardinality_threshold: 100_000,
    };
    let mut outcomes = Vec::new();
    let mut named = Vec::new();
    let bytes = scx_engine::build_obs_predicate_index_bytes(
        &obs,
        &ranges,
        &opts,
        &mut outcomes,
        &mut named,
    )
    .unwrap()
    .expect("n_counts must be indexable");
    writer.write_obs_predicate_index(&bytes).unwrap();
    // Deliberately NOT `apply_obs_shard_column_stats`.

    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1_710_000_000,
            action: "convert".to_string(),
            tool: "column_stats_staleness fixture (no stats)".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();
    writer.finish().unwrap();
    path
}

fn write_indexed_fixture(dir: &TempDir, name: &str) -> PathBuf {
    let path = dir.path().join(name);
    let header =
        FileHeader::new_single_modality(N_OBS as u64, N_VARS as u64, 0, SHARD_ROWS as u32, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    let obs = obs_frame(n_counts_before);
    writer.write_obs(&obs).unwrap();
    writer.write_var(&var_frame()).unwrap();

    let mut ranges: Vec<(u64, u64)> = Vec::new();
    for start in (0..N_OBS).step_by(SHARD_ROWS) {
        let (indptr, indices, values) = shard_csr(SHARD_ROWS);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                start as u64,
            )
            .unwrap();
        ranges.push((start as u64, (start + SHARD_ROWS) as u64));
    }
    assert_eq!(ranges.len(), N_OBS / SHARD_ROWS);

    let opts = scx_engine::PredicateIndexBuildOptions {
        forced_columns: vec!["n_counts".to_string()],
        preset_columns: Vec::new(),
        auto_threshold: 1000,
        high_cardinality_threshold: 100_000,
    };
    let mut outcomes = Vec::new();
    let mut named = Vec::new();
    let bytes = scx_engine::build_obs_predicate_index_bytes(
        &obs,
        &ranges,
        &opts,
        &mut outcomes,
        &mut named,
    )
    .unwrap()
    .expect("n_counts must be indexable");
    assert_eq!(named, vec!["n_counts".to_string()]);
    writer.write_obs_predicate_index(&bytes).unwrap();
    scx_engine::apply_obs_shard_column_stats(&mut writer, &bytes, ranges.len()).unwrap();

    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1_710_000_000,
            action: "convert".to_string(),
            tool: "column_stats_staleness fixture".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();
    writer.finish().unwrap();
    path
}

/// Every `ColumnStat` on every modality-0 CSR shard entry, flattened.
fn csr_column_stats(path: &Path) -> Vec<ColumnStat> {
    let reader = ScxReader::open(path).unwrap();
    reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CsrShard && e.modality_id == 0)
        .filter_map(|e| e.stats.as_ref())
        .flat_map(|s| s.column_stats.clone())
        .collect()
}

/// `n_indexed_columns` must equal `column_stats.len()` on every entry —
/// `ShardStats::read_from` loops exactly `n_indexed_columns` times over the
/// stats payload, so a drifted pair is a decode bug, not a cosmetic one.
fn assert_stats_counts_agree(path: &Path) {
    let reader = ScxReader::open(path).unwrap();
    for e in reader.catalog().entries.iter() {
        if let Some(s) = e.stats.as_ref() {
            assert_eq!(
                s.n_indexed_columns as usize,
                s.column_stats.len(),
                "entry '{}' has n_indexed_columns={} but {} column_stats",
                e.name,
                s.n_indexed_columns,
                s.column_stats.len()
            );
        }
    }
}

/// `count()` rather than `collect()`: it reports the Level-1 skip count
/// alongside the match count and never decodes X, so an assertion can name both
/// the wrong answer and the pruning decision that produced it.
fn query(path: &Path, filter: &str) -> scx_engine::CountResult {
    QueryPipeline::open(path)
        .unwrap()
        .filter_obs(filter)
        .unwrap()
        .count()
        .unwrap()
}

/// Ground truth for `n_counts > threshold`, straight off the frame the op
/// wrote — no query engine involved.
fn expected_gt(value_at: fn(usize) -> i64, threshold: i64) -> usize {
    (0..N_OBS).filter(|&i| value_at(i) > threshold).count()
}

fn replace_obs(path: &Path, index: ConversionPredicateIndexOptions) {
    modify_metadata(
        path,
        &MetadataPatch {
            obs: Some(obs_frame(n_counts_after)),
            index,
            ..Default::default()
        },
    )
    .unwrap();
}

fn no_index_rebuild() -> ConversionPredicateIndexOptions {
    ConversionPredicateIndexOptions::default()
}

// ---------------------------------------------------------------------------
// The finding
// ---------------------------------------------------------------------------

/// The review's measurement, reproduced end to end.
///
/// Before: `n_counts > 1500` correctly matches nothing (the column tops out at
/// 499). After a `modify_metadata` that scales the column ten-fold it must match
/// 349 rows. It returned **0** — every shard excluded by a `MinMax` recorded
/// against values that no longer exist anywhere in the file.
#[test]
fn obs_replacement_does_not_leave_stale_column_stats_pruning_matching_shards() {
    let dir = TempDir::new().unwrap();
    let path = write_indexed_fixture(&dir, "atlas.scx");

    // Precondition: the fixture really does carry stats, and they really do
    // prune — otherwise the test could pass on a file that never had pushdown.
    assert!(
        !csr_column_stats(&path).is_empty(),
        "fixture must carry per-shard column stats"
    );
    let before = query(&path, "n_counts > 1500");
    assert_eq!(expected_gt(n_counts_before, 1500), 0);
    assert_eq!(
        before.matched_rows, 0,
        "before the replacement the empty result is the correct one"
    );
    assert_eq!(
        before.skipped_shards, before.total_shards,
        "the fixture's stats must be live: all 4 shards pruned on a predicate \
         no pre-replacement value can satisfy"
    );

    replace_obs(&path, no_index_rebuild());

    let want = expected_gt(n_counts_after, 1500);
    assert_eq!(want, 349, "guard on the fixture's own arithmetic");
    let after = query(&path, "n_counts > 1500");
    assert_eq!(
        after.matched_rows, want,
        "rows silently pruned by column stats describing the pre-replacement values \
         (skipped {} of {} shards)",
        after.skipped_shards, after.total_shards
    );

    // And the mechanism, not just the symptom: nothing stale is left behind.
    //
    // "Not stale" no longer means "absent". The file was indexed on `n_counts`,
    // so the replacement carries that index forward and re-derives the bounds —
    // which is the stronger outcome, since Level-1 pruning survives the edit
    // instead of being switched off. What must not survive is a bound describing
    // the values that were replaced.
    let stats = csr_column_stats(&path);
    assert!(
        !stats.is_empty(),
        "the carried-forward index must re-derive the stats, not leave the file unprunable"
    );
    for stat in &stats {
        match stat {
            ColumnStat::MinMax { min, max, .. } => assert!(
                *min >= n_counts_after(0) as f64 && *max <= n_counts_after(N_OBS - 1) as f64,
                "MinMax [{min}, {max}] does not describe the post-replacement values \
                 [{}, {}] — a stale bound survived",
                n_counts_after(0),
                n_counts_after(N_OBS - 1),
            ),
            other => panic!("unexpected stat for an Int64 column: {other:?}"),
        }
    }
    assert_stats_counts_agree(&path);
}

/// The carry-forward itself, from the section's own direction: an obs
/// replacement that names no `index_*` must leave the file indexed on the
/// columns it was already indexed on.
///
/// `pyscx.doublet_consensus` is exactly this call — a wholesale obs replacement
/// with no index kwargs — so before the carry it silently reverted
/// `query().filter_obs(...)` to a full obs scan on the last step of the
/// documented doublet workflow.
#[test]
fn an_obs_replacement_carries_the_existing_index_forward() {
    let dir = TempDir::new().unwrap();
    let path = write_indexed_fixture(&dir, "atlas.scx");
    assert!(has_obs_index(&path), "fixture must start with an index");

    let summary = modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(obs_frame(n_counts_after)),
            index: no_index_rebuild(),
            ..Default::default()
        },
    )
    .unwrap();

    assert!(summary.obs_carried_forward);
    assert!(!summary.obs_predicate_index_dropped);
    assert!(summary.obs_columns_not_carried.is_empty());
    assert_eq!(
        summary
            .index
            .result
            .as_ref()
            .map(|r| r.obs_indexed_columns.clone()),
        Some(vec!["n_counts".to_string()]),
    );
    assert!(
        has_obs_index(&path),
        "the file was indexed on n_counts and must still be"
    );
}

/// The carry can fail, and when it does the stats must go with it.
///
/// Replacing `n_counts` with a `Boolean` column of the same name leaves the
/// carried column present but unindexable, so the builder emits no bytes — the
/// `obs_bytes == None` route, reached here without any `index_*` request at all.
/// This is the branch the carry-forward *added*, and the one that would
/// otherwise leave the pre-replacement bounds standing.
#[test]
fn a_carry_that_indexes_nothing_still_clears_the_stats() {
    let dir = TempDir::new().unwrap();
    let path = write_indexed_fixture(&dir, "atlas.scx");

    // Same column name, unindexable dtype.
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("n_counts", DataType::Boolean, false),
    ]);
    let obs = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                (0..N_OBS).map(|i| format!("cell_{i}")).collect::<Vec<_>>(),
            )),
            Arc::new(BooleanArray::from(
                (0..N_OBS).map(|i| i % 2 == 0).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();

    let summary = modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(obs),
            index: no_index_rebuild(),
            ..Default::default()
        },
    )
    .unwrap();

    assert!(summary.obs_predicate_index_dropped);
    assert_eq!(
        summary.obs_columns_not_carried,
        vec!["n_counts".to_string()]
    );
    assert!(
        !summary.obs_carried_forward,
        "a carry that indexed nothing carried nothing forward — reporting this true \
         beside obs_predicate_index_dropped on one provenance entry is how a consumer \
         grepping for the carry misreads a dropped index as a kept one"
    );
    assert!(!has_obs_index(&path));
    assert!(
        csr_column_stats(&path).is_empty(),
        "a carry that indexed nothing must not leave the pre-replacement bounds standing"
    );
    assert_stats_counts_agree(&path);
}

/// An explicit request stays authoritative, and the narrowing is reported.
///
/// The builder is handed only what the caller named, so it emits no outcome for
/// a column the file indexed and the request omits — `obs_columns_not_carried`
/// is the only channel that can name it, and without it the loss is silent.
#[test]
fn an_explicit_request_reports_the_index_columns_it_drops() {
    let dir = TempDir::new().unwrap();
    let path = write_indexed_fixture(&dir, "atlas.scx");

    // The fixture indexes `n_counts`; ask for `cell_id` instead.
    let summary = modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(obs_frame(n_counts_after)),
            index: ConversionPredicateIndexOptions {
                index_obs: vec!["cell_id".to_string()],
                index_var: vec![],
                index_preset: None,
                index_auto_threshold: 0,
            },
            ..Default::default()
        },
    )
    .unwrap();

    assert!(
        !summary.obs_carried_forward,
        "the caller named their own columns"
    );
    assert_eq!(
        summary.obs_columns_not_carried,
        vec!["n_counts".to_string()]
    );
    // Rows stay correct either way — the point is that pushdown on `n_counts`
    // is gone and the caller was told.
    assert_eq!(
        query(&path, "n_counts > 1500").matched_rows,
        expected_gt(n_counts_after, 1500)
    );
}

/// The two report channels overlap, and a front end rendering both raw warns
/// twice about one column.
///
/// On the carry path every column that could not be carried also has a
/// `PresetSkipped` outcome naming it; on the explicit-request path the builder
/// never sees the old index and emits nothing. `*_not_carried_unreported` is the
/// difference, and it lives in `scx-ops` so pyscx and the CLI cannot drift —
/// the CLI double-warned the moment it started rendering outcomes at all.
#[test]
fn the_two_report_channels_do_not_both_name_the_same_column() {
    let dir = TempDir::new().unwrap();

    // Carry path: `n_counts` comes back as an unindexable dtype, so the builder
    // emits a `PresetSkipped` for it AND it lands in `obs_columns_not_carried`.
    let path = write_indexed_fixture(&dir, "carry.scx");
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("n_counts", DataType::Boolean, false),
    ]);
    let obs = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                (0..N_OBS).map(|i| format!("cell_{i}")).collect::<Vec<_>>(),
            )),
            Arc::new(BooleanArray::from(
                (0..N_OBS).map(|i| i % 2 == 0).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    let carried = modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(obs),
            index: no_index_rebuild(),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        carried.obs_columns_not_carried,
        vec!["n_counts".to_string()]
    );
    assert!(
        carried.obs_not_carried_unreported().is_empty(),
        "the builder already emitted a PresetSkipped naming this column"
    );

    // Explicit path: the builder is handed only what the caller named, so it
    // says nothing about `n_counts` and this channel is the only one that can.
    let path = write_indexed_fixture(&dir, "explicit.scx");
    let explicit = modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(obs_frame(n_counts_after)),
            index: ConversionPredicateIndexOptions {
                index_obs: vec!["cell_id".to_string()],
                index_var: vec![],
                index_preset: None,
                index_auto_threshold: 0,
            },
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        explicit.obs_not_carried_unreported(),
        vec!["n_counts".to_string()],
        "nothing else names it, so filtering it out would silence the loss"
    );
}

/// Naming an index column on ONE axis must not change the other axis's policy.
///
/// `user_wants_index` is a whole-patch question — true if any index knob is set
/// — so reading it per axis let `index_var=[…]` take the obs axis off
/// carry-forward and onto auto-detect. Under the "omit `index_*` to carry"
/// contract that is a footgun: the caller said nothing about obs.
#[test]
fn an_index_request_on_one_axis_leaves_the_other_axis_carrying() {
    let dir = TempDir::new().unwrap();
    let path = write_indexed_fixture(&dir, "atlas.scx");

    let summary = modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(obs_frame(n_counts_after)),
            var: Some(var_frame()),
            index: ConversionPredicateIndexOptions {
                index_obs: vec![],
                index_var: vec!["gene_id".to_string()],
                index_preset: None,
                index_auto_threshold: 0,
            },
            ..Default::default()
        },
    )
    .unwrap();

    assert!(
        summary.obs_carried_forward,
        "naming only index_var must leave obs on carry-forward"
    );
    assert!(summary.obs_columns_not_carried.is_empty());
    assert_eq!(
        summary
            .index
            .result
            .as_ref()
            .map(|r| r.obs_indexed_columns.clone()),
        Some(vec!["n_counts".to_string()]),
        "obs must be indexed on the file's own column, not auto-detected"
    );
    assert!(has_obs_index(&path));
    // …and the var axis took the caller's explicit list, so it is not a carry.
    assert!(!summary.var_carried_forward);
}

fn has_obs_index(path: &Path) -> bool {
    ScxReader::open(path)
        .unwrap()
        .catalog()
        .entries
        .iter()
        .any(|e| e.section_type == SectionType::ObsPredicateIndex)
}

/// `Gt` is not the only arm that prunes from a stale bound: `Eq` and `In` skip a
/// shard whose `[min, max]` excludes the value (`pushdown.rs:129-142`, `:251+`).
/// A single-row lookup by exact count therefore vanishes too.
///
/// The mirror cases are deliberately absent, because they are not bugs: `Lt`/`Le`
/// prune on the stale `min` (100…400), which lies *below* the rescaled values, so
/// shards are over-**admitted** and the row evaluator still returns the correct
/// answer. Stale stats only produce wrong rows in the pruning direction — a test
/// asserting otherwise passes identically before and after the fix, which is how
/// the first draft of this one was caught.
#[test]
fn stale_column_stats_also_break_exact_and_set_membership_lookups() {
    let dir = TempDir::new().unwrap();
    let path = write_indexed_fixture(&dir, "atlas.scx");
    replace_obs(&path, no_index_rebuild());

    // n_counts_after(150) == 2500 — one cell, in shard 1.
    let want_eq = (0..N_OBS).filter(|&i| n_counts_after(i) == 2500).count();
    assert_eq!(want_eq, 1);
    assert_eq!(query(&path, "n_counts == 2500").matched_rows, want_eq);

    let want_in = (0..N_OBS)
        .filter(|&i| matches!(n_counts_after(i), 2500 | 4200))
        .count();
    assert_eq!(want_in, 2);
    assert_eq!(
        query(&path, "n_counts in [2500, 4200]").matched_rows,
        want_in
    );
}

// ---------------------------------------------------------------------------
// The other two ways `modify_metadata` reaches the stale state
// ---------------------------------------------------------------------------

/// A rebuild was requested, but the only requested column has an unsupported
/// dtype, so the builder produces no index bytes and there is nothing to derive
/// stats from. The old stats must still go — this is the path a fix that only
/// looks at `rebuild_obs_index` would miss.
#[test]
fn obs_replacement_with_an_unindexable_rebuild_request_still_clears() {
    let dir = TempDir::new().unwrap();
    let path = write_indexed_fixture(&dir, "atlas.scx");

    modify_metadata(
        &path,
        &MetadataPatch {
            obs: Some(obs_frame(n_counts_after)),
            index: ConversionPredicateIndexOptions {
                index_obs: vec!["passed_qc".to_string()],
                index_var: vec![],
                index_preset: None,
                index_auto_threshold: 0,
            },
            ..Default::default()
        },
    )
    .unwrap();

    assert!(
        csr_column_stats(&path).is_empty(),
        "an index rebuild that produced nothing must not leave the old stats standing"
    );
    assert_eq!(
        query(&path, "n_counts > 1500").matched_rows,
        expected_gt(n_counts_after, 1500)
    );
    assert_stats_counts_agree(&path);
}

/// The third route, and the least obvious: a rebuild that *does* produce index
/// bytes, but whose CSR ranges do not tile `[0, n_obs)`.
///
/// `modify_metadata` then finishes the index over the **obs-shard** ranges
/// instead (`use_csr == false`) and deliberately skips the derive, because
/// `derive_shard_column_stats` would be addressing a different shard space. So a
/// fresh `ObsPredicateIndex` is written and `per_shard_obs_stats` stays `None` —
/// a file that looks freshly indexed while still carrying the previous obs's
/// bounds. Only the `None` arm's clear catches this one.
///
/// The fixture under-covers on purpose (3 shards of 100 for `n_obs = 400`),
/// which is the shape the production comment describes as "some shards missing
/// stats on older files".
#[test]
fn obs_replacement_clears_when_csr_ranges_do_not_cover_the_obs_axis() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("undercovered.scx");
    let header =
        FileHeader::new_single_modality(N_OBS as u64, N_VARS as u64, 0, SHARD_ROWS as u32, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    let obs = obs_frame(n_counts_before);
    writer.write_obs(&obs).unwrap();
    writer.write_var(&var_frame()).unwrap();

    // Three shards for four shards' worth of obs rows.
    let mut ranges: Vec<(u64, u64)> = Vec::new();
    for start in (0..N_OBS - SHARD_ROWS).step_by(SHARD_ROWS) {
        let (indptr, indices, values) = shard_csr(SHARD_ROWS);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                start as u64,
            )
            .unwrap();
        ranges.push((start as u64, (start + SHARD_ROWS) as u64));
    }
    assert_eq!(ranges.len(), 3, "deliberately under-covering n_obs = 400");

    let opts = scx_engine::PredicateIndexBuildOptions {
        forced_columns: vec!["n_counts".to_string()],
        preset_columns: Vec::new(),
        auto_threshold: 1000,
        high_cardinality_threshold: 100_000,
    };
    let bytes = scx_engine::build_obs_predicate_index_bytes(
        &obs,
        &ranges,
        &opts,
        &mut Vec::new(),
        &mut Vec::new(),
    )
    .unwrap()
    .expect("n_counts must be indexable");
    writer.write_obs_predicate_index(&bytes).unwrap();
    scx_engine::apply_obs_shard_column_stats(&mut writer, &bytes, ranges.len()).unwrap();
    writer.finish().unwrap();

    assert!(!csr_column_stats(&path).is_empty());

    // Rebuild requested and satisfiable — but the CSR ranges cover 300 of 400
    // rows, so the derive is skipped and `per_shard_obs_stats` comes back `None`.
    replace_obs(
        &path,
        ConversionPredicateIndexOptions {
            index_obs: vec!["n_counts".to_string()],
            index_var: vec![],
            index_preset: None,
            index_auto_threshold: 1000,
        },
    );

    assert!(
        ScxReader::open(&path)
            .unwrap()
            .catalog()
            .entries
            .iter()
            .any(|e| e.section_type == SectionType::ObsPredicateIndex),
        "a fresh index IS written on this path — which is what makes the stale \
         stats so easy to miss"
    );
    assert!(
        csr_column_stats(&path).is_empty(),
        "index written over obs-shard ranges, stats not re-derived — the old \
         bounds must not survive"
    );
    assert_stats_counts_agree(&path);
}

/// A rebuild that *does* produce an index re-derives the stats, so pushdown
/// comes back rather than being disabled forever. Without this the fix would be
/// a permanent performance regression dressed up as a correctness fix.
#[test]
fn obs_replacement_with_an_index_rebuild_restores_pruning() {
    let dir = TempDir::new().unwrap();
    let path = write_indexed_fixture(&dir, "atlas.scx");

    replace_obs(
        &path,
        ConversionPredicateIndexOptions {
            index_obs: vec!["n_counts".to_string()],
            index_var: vec![],
            index_preset: None,
            index_auto_threshold: 1000,
        },
    );

    let stats = csr_column_stats(&path);
    assert!(
        !stats.is_empty(),
        "a requested rebuild must re-derive the stats, not merely clear them"
    );
    // The re-derived bounds describe the NEW values.
    let maxes: Vec<f64> = stats
        .iter()
        .filter_map(|cs| match cs {
            ColumnStat::MinMax { max, .. } => Some(*max),
            _ => None,
        })
        .collect();
    assert!(
        maxes.iter().any(|&m| m > 1500.0),
        "re-derived MinMax still describes the pre-replacement range: {maxes:?}"
    );

    // Shards hold 1000–1990 / 2000–2990 / 3000–3990 / 4000–4990, so a `> 3000`
    // predicate must prune exactly the first two — pruning restored, and
    // restored *correctly*.
    let r = query(&path, "n_counts > 3000");
    assert_eq!(r.matched_rows, expected_gt(n_counts_after, 3000));
    assert_eq!(
        (r.skipped_shards, r.total_shards),
        (2, 4),
        "re-derived stats must prune the two shards that cannot match"
    );
    assert_eq!(
        query(&path, "n_counts > 1500").matched_rows,
        expected_gt(n_counts_after, 1500)
    );
    assert_stats_counts_agree(&path);
}

// ---------------------------------------------------------------------------
// Ops that must NOT lose their stats
// ---------------------------------------------------------------------------

/// `optimize` re-emits every shard through the writer, and until Phase 5c it
/// never re-derived the stats — so it silently **lost** them, costing Level-1
/// pruning and nothing else. This test used to assert that loss: it read
/// `csr_column_stats(&out).is_empty()` and `skipped_shards == 0`, which is a
/// specification for the bug rather than for the behaviour anyone wanted. It
/// now asserts the carry, and the two failure modes it exists to separate are
/// still separated — losing a stat costs pruning, keeping a *stale* one returns
/// the wrong rows, and only the second is a correctness bug.
///
/// `optimize` declares `ObsPredicateIndex => Carry::Verbatim`; the audit checks
/// the section is present and cannot see whether the statistics its pruning
/// needs came along. `build-csc` had the identical defect —
/// `scx-cli/tests/cli_ops_integration.rs::build_csc_carries_predicate_index_and_pushdown`.
#[test]
fn optimize_carries_the_column_stats_and_the_pruning() {
    let dir = TempDir::new().unwrap();
    let path = write_indexed_fixture(&dir, "atlas.scx");
    let before_stats = csr_column_stats(&path);
    assert!(
        !before_stats.is_empty(),
        "fixture precondition: the input must carry per-shard column stats"
    );
    let before = query(&path, "n_counts > 300");

    let out = dir.path().join("optimized.scx");
    scx_ops::optimize(&path, &out, None, scx_format_io::ObsShardPolicy::Auto).unwrap();

    assert_eq!(
        csr_column_stats(&out),
        before_stats,
        "optimize preserves row order and CSR shard boundaries, so the stats it \
         carries from the input must equal the input's"
    );
    let after = query(&out, "n_counts > 300");
    assert_eq!(after.matched_rows, expected_gt(n_counts_before, 300));
    assert_eq!(
        after.skipped_shards, before.skipped_shards,
        "carrying the index bytes without the stats leaves Level-1 pruning off"
    );
    assert_stats_counts_agree(&out);
}

/// A patch that never touches obs has no reason to cost the file its pushdown.
/// `column_stats` are obs-derived exclusively, so a `uns`-only replace must
/// leave them exactly as they were.
#[test]
fn uns_only_patch_keeps_the_column_stats() {
    let dir = TempDir::new().unwrap();
    let path = write_indexed_fixture(&dir, "atlas.scx");
    let before = csr_column_stats(&path);
    assert!(!before.is_empty());

    modify_metadata(
        &path,
        &MetadataPatch {
            uns: Some(serde_json::json!({"state": "v1"})),
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(
        csr_column_stats(&path),
        before,
        "a uns-only patch must not disable Level-1 pruning"
    );
    assert_stats_counts_agree(&path);
}

/// An index whose keying a rewrite cannot verify must not have statistics
/// fabricated for it.
///
/// `optimize` / `upgrade` re-encode every CSR shard (as `build-csc` did before it
/// became an in-place append) and must restore the
/// per-shard `column_stats` Level-1 pruning reads. The first implementation
/// re-derived them from the carried index bytes, gated on
/// `max(shard_id) + 1 == n_csr_shards`. **That equality is not proof the index is
/// keyed to the CSR partition**: `modify_metadata` falls back to building the
/// index over *obs*-shard ranges when the CSR shards do not tile `[0, n_obs)`,
/// and an obs partition can have the same shard *count* as the CSR one with
/// different *boundaries* — CSR `[0,100) [100,200) [200,300)` against obs
/// `[0,134) [134,268) [268,400)`. Derivation would then attach shard 0's obs
/// statistics to CSR rows they do not describe, and Level-1 consumes
/// `column_stats` *before* residual evaluation — so a fabricated "value absent"
/// prunes a shard holding real matches and the query returns an incomplete row
/// set. Reported by review (codex - gpt-5.6-sol) on PR #451.
///
/// The fix carries the input's stats instead of re-deriving, which makes the
/// inference unnecessary: a file in that state has had its stats cleared, so
/// there is nothing to carry and nothing to mis-attribute. This asserts the end
/// state — no stats in, no stats out, rows unchanged — rather than the
/// arithmetic, so it stays valid if the implementation changes again.
#[test]
fn a_rewrite_does_not_fabricate_stats_for_an_unverifiable_index() {
    let dir = TempDir::new().unwrap();

    // Control half: stats present on the input must survive the rewrite. Without
    // it this test would pass equally well if rewrites simply never carried
    // stats at all.
    let with_stats = write_indexed_fixture(&dir, "atlas.scx");
    let before = csr_column_stats(&with_stats);
    assert!(!before.is_empty(), "fixture precondition: stats present");
    let out = dir.path().join("optimized.scx");
    scx_ops::optimize(&with_stats, &out, None, scx_format_io::ObsShardPolicy::Auto).unwrap();
    assert_eq!(
        csr_column_stats(&out),
        before,
        "control: a rewrite must carry the stats a CSR-keyed index left behind"
    );

    // The case that matters: index section present, stats absent.
    let no_stats = write_indexed_fixture_without_stats(&dir, "no_stats.scx");
    assert!(
        csr_column_stats(&no_stats).is_empty(),
        "setup: no per-shard stats"
    );
    assert!(
        ScxReader::open(&no_stats)
            .unwrap()
            .read_obs_predicate_index_bytes()
            .unwrap()
            .is_some(),
        "setup: the index section must be present, or this proves nothing"
    );

    for (label, out) in [
        ("optimize", dir.path().join("opt_no_stats.scx")),
        ("build-csc", dir.path().join("csc_no_stats.scx")),
    ] {
        if label == "optimize" {
            scx_ops::optimize(&no_stats, &out, None, scx_format_io::ObsShardPolicy::Auto).unwrap();
        } else {
            scx_ops::run_build_csc(&no_stats, &out, Some("64M"), false, 0, None, None).unwrap();
        }
        assert!(
            csr_column_stats(&out).is_empty(),
            "{label} must not derive Level-1 statistics from an index whose keying it \
             cannot verify — fabricated bounds prune shards holding real matches"
        );
        // Losing pruning costs speed, never rows.
        let r = query(&out, "n_counts > 300");
        assert_eq!(r.matched_rows, expected_gt(n_counts_before, 300), "{label}");
        assert_eq!(r.skipped_shards, 0, "{label}: no stats, nothing to prune");
    }
}
