use super::*;

use std::sync::Arc as StdArc;

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;

use crate::error::LoaderError;

/// Minimal multi-shard `.scx`: row `r` has one non-zero at column `r % n_vars`
/// with value `((r + 1) & 0xFF)`, so `(col, value)` is recoverable from the
/// row index. Mirrors `index_plan_tests::write_multi_shard_fixture`.
fn write_multi_shard_fixture(path: &std::path::Path, n_obs: usize, n_vars: usize, n_shards: usize) {
    assert!(n_obs.is_multiple_of(n_shards), "n_obs must divide n_shards");
    let rows_per_shard = n_obs / n_shards;
    let header = FileHeader::new_single_modality(
        n_obs as u64,
        n_vars as u64,
        n_obs as u64,
        rows_per_shard as u32,
        0,
        0,
    );
    let mut writer = ScxWriter::new(path, header).unwrap();

    let obs_schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
    let cell_ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    let obs = arrow::record_batch::RecordBatch::try_new(
        StdArc::new(obs_schema),
        vec![StdArc::new(StringArray::from(
            cell_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    writer.write_obs(&obs).unwrap();

    let var_schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    let gene_ids: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    let var = arrow::record_batch::RecordBatch::try_new(
        StdArc::new(var_schema),
        vec![StdArc::new(StringArray::from(
            gene_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    writer.write_var(&var).unwrap();

    for s in 0..n_shards {
        let row_start = s * rows_per_shard;
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for local in 0..rows_per_shard {
            let row = row_start + local;
            indices.push((row % n_vars) as u32);
            values.push(((row + 1) & 0xFF) as u8);
            indptr.push(*indptr.last().unwrap() + 1);
        }
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
    }
    writer.finish().unwrap();
}

/// A two-file engine: each file 32 rows × 8 vars × 4 shards, sharing one cache.
fn two_file_engine(dir: &std::path::Path, cache_shards: usize) -> Arc<PrefetchEngine> {
    let p0 = dir.join("f0.scx");
    let p1 = dir.join("f1.scx");
    write_multi_shard_fixture(&p0, 32, 8, 4);
    write_multi_shard_fixture(&p1, 32, 8, 4);
    let readers = vec![ScxReader::open(&p0).unwrap(), ScxReader::open(&p1).unwrap()];
    PrefetchEngine::from_scx_readers(
        readers,
        cache_shards,
        usize::MAX,
        /*lookahead*/ 4,
        // These fixtures are unframed, so the gate is inert here either way;
        // `false` states the intent (warm + cache) rather than relying on that.
        /*scatter_block_index*/
        false,
    )
}

/// Plan = list of `(file_id, row)`; `rows_of` just clones it.
type Plan = Vec<(u32, u64)>;

fn rows_of(plan: &Plan) -> Vec<(u32, u64)> {
    plan.clone()
}

/// Gather the single non-zero `(col, value)` of each plan row through the
/// engine's readers — exercises the real `read_rows_with` path.
///
/// Takes the plan **by value**, matching `ProcFn` since ORG-9.10-1: the queue
/// is the plan's last owner, so a consumer that needs to consume or reorder it
/// (the pair loader sorts in place) does not have to clone per batch.
fn gather(engine: &PrefetchEngine, plan: Plan, admit_row_groups: bool) -> Result<Vec<(i32, f32)>> {
    let mut out = Vec::with_capacity(plan.len());
    for &(fid, row) in &plan {
        let mut got: Option<(i32, f32)> = None;
        engine
            .reader(fid)
            .read_rows_with_admission(&[row], Some(admit_row_groups), |_pos, idx, data| {
                got = idx.first().copied().zip(data.first().copied());
                Ok(())
            })
            .map_err(LoaderError::FormatError)?;
        out.push(got.expect("row has one non-zero"));
    }
    Ok(out)
}

fn into_iter(plans: Vec<Plan>) -> impl Iterator<Item = Result<Plan>> + Send + 'static {
    plans.into_iter().map(Ok)
}

/// Expected `(col, value)` for fixture row `r` (n_vars = 8).
fn expected(row: u64) -> (i32, f32) {
    ((row % 8) as i32, ((row + 1) & 0xFF) as f32)
}

#[test]
fn engine_gathers_multi_file_plans_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let engine = two_file_engine(dir.path(), 8);
    let plans = vec![vec![(0u32, 5u64), (1, 30), (0, 0)], vec![(1, 7), (1, 31)]];
    let out: Vec<_> = Arc::clone(&engine)
        .iter_with_plans(into_iter(plans.clone()), 4, rows_of, gather)
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(out.len(), 2);
    for (batch, plan) in out.iter().zip(plans.iter()) {
        let want: Vec<(i32, f32)> = plan.iter().map(|&(_, r)| expected(r)).collect();
        assert_eq!(batch, &want);
    }
}

#[test]
fn engine_lookahead_zero_vs_four_parity() {
    let dir = tempfile::tempdir().unwrap();
    let plans = vec![
        vec![(0u32, 31u64), (1, 0), (0, 5)],
        vec![(1, 12), (0, 20), (1, 30)],
    ];
    let run = |lookahead: usize| -> Vec<Vec<(i32, f32)>> {
        let engine = two_file_engine(dir.path(), 8);
        engine
            .iter_with_plans(into_iter(plans.clone()), lookahead, rows_of, gather)
            .map(|r| r.unwrap())
            .collect()
    };
    assert_eq!(run(0), run(4));
}

#[test]
fn engine_propagates_plan_errors() {
    let dir = tempfile::tempdir().unwrap();
    let engine = two_file_engine(dir.path(), 8);
    let plans: Vec<Result<Plan>> = vec![
        Ok(vec![(0u32, 1u64)]),
        Err(LoaderError::ChannelError("synthetic".into())),
        Ok(vec![(1, 2)]),
    ];
    let mut it = engine.iter_with_plans(plans.into_iter(), 2, rows_of, gather);
    assert!(matches!(it.next(), Some(Ok(_))));
    match it.next().expect("second item") {
        Err(LoaderError::ChannelError(s)) => assert!(s.contains("synthetic")),
        other => panic!("expected ChannelError, got {other:?}"),
    }
    assert!(it.next().is_none(), "iteration stops after a plan error");
}

#[test]
fn engine_propagates_gather_errors() {
    let dir = tempfile::tempdir().unwrap();
    let engine = two_file_engine(dir.path(), 8);
    // Row 999 is out of range (files have 32 rows) → read_rows_with errors.
    let plans = vec![vec![(0u32, 999u64)]];
    let mut it = engine.iter_with_plans(into_iter(plans), 4, rows_of, gather);
    assert!(matches!(it.next(), Some(Err(_))));
}

#[test]
fn engine_drop_mid_iteration_is_safe() {
    let dir = tempfile::tempdir().unwrap();
    let engine = two_file_engine(dir.path(), 8);
    let plans = vec![vec![(0u32, 1u64)], vec![(1, 2)], vec![(0, 3)], vec![(1, 4)]];
    // lookahead 3 leaves several in-flight prefetch handles queued when dropped;
    // Drop aborts them (M2) and detaches the pull worker — no deadlock, no leak.
    let mut it = engine.iter_with_plans(into_iter(plans), 3, rows_of, gather);
    let _ = it.next();
    drop(it);
}

#[test]
fn engine_warms_shards_across_both_readers() {
    let dir = tempfile::tempdir().unwrap();
    // cache large enough to hold every touched shard across both files.
    let engine = two_file_engine(dir.path(), 16);
    // file 0 rows in shard 0 and shard 3; file 1 row in shard 1.
    let plans = vec![vec![(0u32, 0u64), (0, 31), (1, 8)]];
    let out: Vec<_> = Arc::clone(&engine)
        .iter_with_plans(into_iter(plans), 4, rows_of, gather)
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(out.len(), 1);
    // Prefetch warmed the touched shards in BOTH readers (shared cache).
    assert!(engine.reader(0).cache_contains(0));
    assert!(engine.reader(0).cache_contains(3));
    assert!(engine.reader(1).cache_contains(1));
}

// --- lazy runtime / fork-safety (Phase 1.2) -------------------------------

#[test]
fn engine_runtime_not_built_at_construction() {
    let dir = tempfile::tempdir().unwrap();
    let engine = two_file_engine(dir.path(), 8);
    assert!(
        engine.runtime.get().is_none(),
        "runtime must not be built until first iteration (fork-safety)"
    );
}

#[test]
fn engine_runtime_built_after_iter_consumed() {
    let dir = tempfile::tempdir().unwrap();
    let engine = two_file_engine(dir.path(), 8);
    assert!(engine.runtime.get().is_none());
    let plans = vec![vec![(0u32, 1u64), (1, 2)]];
    let mut it = Arc::clone(&engine).iter_with_plans(into_iter(plans), 2, rows_of, gather);
    let _ = it.next();
    assert!(
        engine.runtime.get().is_some(),
        "first consumption materializes the runtime via OnceLock"
    );
}

#[test]
fn engine_runtime_idempotent_under_concurrent_init() {
    let dir = tempfile::tempdir().unwrap();
    let engine = two_file_engine(dir.path(), 8);
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let e = Arc::clone(&engine);
            std::thread::spawn(move || e.runtime().unwrap().handle().id())
        })
        .collect();
    let ids: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert!(
        ids.windows(2).all(|w| w[0] == w[1]),
        "all threads must observe the same lazily-built runtime"
    );
}

// ---------------------------------------------------------------------
// §9.4 — prefetch depth is opportunistic, never an obligation on the plan
// generator. `PlanPrefetchIter::refill` had the same blocking-`recv` loop as
// `IndexPlanIter::refill`, and deadlocked the same way against a generator
// that produces plan i+1 only after seeing batch i.
// ---------------------------------------------------------------------

/// Run `f` on a worker thread; panic rather than hang the suite if it has not
/// finished within `secs`.
fn with_deadline<T: Send + 'static>(
    secs: u64,
    label: &'static str,
    f: impl FnOnce() -> T + Send + 'static,
) -> T {
    let (tx, rx) = bounded(1);
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    match rx.recv_timeout(std::time::Duration::from_secs(secs)) {
        Ok(v) => v,
        Err(_) => panic!("{label}: no result within {secs}s — the iterator deadlocked"),
    }
}

/// A plan generator that will not produce plan `i + 1` until the consumer has
/// acknowledged batch `i`.
struct FeedbackPlans {
    plans: std::vec::IntoIter<Plan>,
    ack: Receiver<()>,
    first: bool,
}

impl Iterator for FeedbackPlans {
    type Item = Result<Plan>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.first {
            self.ack.recv().ok()?;
        }
        self.first = false;
        self.plans.next().map(Ok)
    }
}

#[test]
fn engine_feedback_generator_yields_every_batch() {
    let plans: Vec<Plan> = vec![
        vec![(0u32, 5u64)],
        vec![(1u32, 30u64)],
        vec![(0u32, 12u64)],
        vec![(1u32, 7u64)],
    ];
    let want = plans.len();

    let got = with_deadline(30, "engine feedback plan generator", move || {
        let dir = tempfile::tempdir().unwrap();
        let engine = two_file_engine(dir.path(), 8);
        let (ack_tx, ack_rx) = bounded(0);
        let it = engine.iter_with_plans(
            FeedbackPlans {
                plans: plans.into_iter(),
                ack: ack_rx,
                first: true,
            },
            /*lookahead*/ 4,
            rows_of,
            gather,
        );

        let mut count = 0usize;
        for batch in it {
            batch.expect("batch must gather");
            count += 1;
            let _ = ack_tx.send(());
        }
        count
    });

    assert_eq!(
        got, want,
        "every plan must produce a batch when the generator waits on the previous one"
    );
}

/// Red for the *wrong* fix: mapping `TryRecvError::Empty` onto
/// `plan_stream_done` truncates the epoch to one batch, silently.
#[test]
fn engine_slow_generator_does_not_end_the_epoch() {
    let plans: Vec<Plan> = (0..5u64).map(|i| vec![(0u32, i * 3)]).collect();
    let want = plans.len();

    let got = with_deadline(30, "engine slow plan generator", move || {
        let dir = tempfile::tempdir().unwrap();
        let engine = two_file_engine(dir.path(), 8);
        let slow = plans.into_iter().map(|p| {
            std::thread::sleep(std::time::Duration::from_millis(20));
            Ok(p)
        });
        let mut count = 0usize;
        for batch in engine.iter_with_plans(slow, /*lookahead*/ 4, rows_of, gather) {
            batch.expect("batch must gather");
            count += 1;
        }
        count
    });

    assert_eq!(got, want, "a slow generator must not truncate the epoch");
}

/// Over-fix guard: with the generator ahead, the queue still fills to depth.
#[test]
fn engine_prefetch_depth_survives_when_the_generator_keeps_up() {
    const LOOKAHEAD: usize = 4;
    let dir = tempfile::tempdir().unwrap();
    let engine = two_file_engine(dir.path(), 8);

    let plans: Vec<Plan> = (0..LOOKAHEAD as u64).map(|i| vec![(0u32, i * 3)]).collect();

    // Exactly `LOOKAHEAD` plans into a `bounded(LOOKAHEAD)` channel, so the
    // pull thread buffers all of them and runs to exhaustion without the
    // consumer — which is what makes the `try_recv` arms race-free here.
    let (drained_tx, drained_rx) = bounded(1);
    let gen = into_iter(plans).chain(std::iter::from_fn(move || -> Option<Result<Plan>> {
        let _ = drained_tx.send(());
        None
    }));

    let mut it = engine.iter_with_plans(gen, LOOKAHEAD, rows_of, gather);
    drained_rx
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("plan generator must drain into the bounded channel");

    it.next()
        .expect("first batch")
        .expect("first batch must gather");

    assert_eq!(
        it.in_flight.len(),
        LOOKAHEAD - 1,
        "the queue must still be prefetched to depth when the generator is ahead"
    );
}

// ---------------------------------------------------------------------
// Pre-refactor pins for ORG-9.10-1 (fold `IndexPlanIter` into this engine).
//
// Each test here states a contract that differs between the two forked
// iterators, so the fold has to satisfy both arms deliberately instead of
// inheriting whichever one it happened to be written against. The mirror-image
// halves live in `index_plan_tests.rs`.
// ---------------------------------------------------------------------

/// Row-group-**framed** single-file fixture (v4 file / v2 shards, multi-entry
/// `BlockIndex`), the shape `BackedCsrReader::block_index_eligible` needs.
///
/// `write_csr_shard` — what `write_multi_shard_fixture` above uses — only emits
/// the unframed layout, so a framed fixture has to go through
/// `encode_one_shard(..., Some(FramingConfig { .. }))` + `write_preencoded_shard`.
/// Same recipe as `scx-format-io/src/backed_tests.rs::write_framed_file`.
///
/// Row `r` has one non-zero at column `r % n_vars` with value `(r % 250 + 1)`
/// (never 0, so the fixture stays valid for codecs that reject a zero value).
/// Fixture shape, fixed rather than parameterised: `framed_expected` below is
/// the gather oracle and hard-codes the same `N_VARS`, so a caller varying the
/// shape independently would silently corrupt it.
pub(crate) const FRAMED_N_OBS: usize = 256;
pub(crate) const FRAMED_N_VARS: usize = 8;
const FRAMED_N_SHARDS: usize = 4;
const FRAMED_ROW_GROUP_ROWS: u32 = 16;

pub(crate) fn write_framed_fixture(path: &std::path::Path) {
    let (n_obs, n_vars, n_shards, row_group_rows) = (
        FRAMED_N_OBS,
        FRAMED_N_VARS,
        FRAMED_N_SHARDS,
        FRAMED_ROW_GROUP_ROWS,
    );
    let rows_per_shard = n_obs / n_shards;

    let mut header = FileHeader::new_single_modality(
        n_obs as u64,
        n_vars as u64,
        n_obs as u64,
        rows_per_shard as u32,
        0,
        0,
    );
    header.format_version = scx_format_io::header::CURRENT_FORMAT_VERSION; // v4 (framed)
    let mut writer = ScxWriter::new(path, header).unwrap();

    let obs_schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
    let cell_ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    let obs = arrow::record_batch::RecordBatch::try_new(
        StdArc::new(obs_schema),
        vec![StdArc::new(StringArray::from(
            cell_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    writer.write_obs(&obs).unwrap();

    let var_schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    let gene_ids: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    let var = arrow::record_batch::RecordBatch::try_new(
        StdArc::new(var_schema),
        vec![StdArc::new(StringArray::from(
            gene_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    writer.write_var(&var).unwrap();

    for s in 0..n_shards {
        let row_start = s * rows_per_shard;
        let mut indptr = vec![0u64];
        let mut indices: Vec<u32> = Vec::new();
        let mut values: Vec<f32> = Vec::new();
        for local in 0..rows_per_shard {
            let row = row_start + local;
            indices.push((row % n_vars) as u32);
            values.push(framed_expected(row as u64).1);
            indptr.push(*indptr.last().unwrap() + 1);
        }
        let mut enc_opts = scx_format_io::EncodeShardOptions::new(
            format!("X_shard_{s}"),
            scx_format_io::section::SectionType::CsrShard,
            n_vars as u64,
            row_start as u64,
            0,
        );
        enc_opts.explicit_codec = Some(CodecId::None);
        enc_opts.framing = Some(scx_format_io::FramingConfig {
            row_group_rows,
            ..Default::default()
        });
        let pre = scx_format_io::encode_one_shard(&indptr, &indices, &values, &enc_opts).unwrap();
        writer.write_preencoded_shard(pre).unwrap();
    }
    writer.finish().unwrap();
}

/// Expected `(col, value)` for a `write_framed_fixture` row.
pub(crate) fn framed_expected(row: u64) -> (i32, f32) {
    (
        (row % FRAMED_N_VARS as u64) as i32,
        ((row % 250) + 1) as f32,
    )
}

/// A one-file engine over a framed fixture, with the reader's per-reader
/// block-index gate set explicitly.
///
/// This used to hand-roll `PrefetchEngine::new` because `from_scx_readers`
/// exposed no knob for the gate — ORG-9.10-1 drift (a) seen from the other
/// side. It has one now (9b), so this is the production constructor with the
/// gate passed through, which is what makes these tests cover the real path.
fn framed_engine(dir: &std::path::Path, scatter_block_index: bool) -> Arc<PrefetchEngine> {
    // `usize::MAX` is a count-only cache: it retains no row groups (see
    // `ShardCache::caches_groups`), so these engines exercise the block-index
    // route without the row-group LRU. The row-group tests below pass a
    // finite budget.
    framed_engine_with_budget(dir, scatter_block_index, usize::MAX)
}

fn framed_engine_with_budget(
    dir: &std::path::Path,
    scatter_block_index: bool,
    bytes_budget: usize,
) -> Arc<PrefetchEngine> {
    let path = dir.join("framed.scx");
    write_framed_fixture(&path);
    PrefetchEngine::from_scx_readers(
        vec![ScxReader::open(&path).unwrap()],
        // Large enough to hold every shard the plans touch, so the reader gate
        // is the only thing that decides whether they warm.
        /*cache_shards*/
        8,
        bytes_budget,
        /*default_lookahead*/ 4,
        scatter_block_index,
    )
}

/// Bytes one decoded row group of the framed fixture occupies in the LRU:
/// `FRAMED_ROW_GROUP_ROWS` rows at one non-zero each — `(16 + 1) × 8 + 16 × 4 +
/// 16 × 4`. The row-group warm tests size their budgets from it.
pub(crate) const FRAMED_GROUP_BYTES: usize =
    (FRAMED_ROW_GROUP_ROWS as usize + 1) * 8 + FRAMED_ROW_GROUP_ROWS as usize * 8;

/// **Pin (ORG-9.10-1, drift (a)/L2).** The engine's prefetch must NOT warm a
/// shard whose group is block-index-eligible — leaving it undecoded is what
/// lets the gather take the O(rows) row-group path instead of having the win
/// negated by an eager full-shard warm.
///
/// Nothing tested this branch: `plan_engine.rs`'s
/// `reader.block_index_eligible(sidx, group_len)` arm could be deleted and the
/// whole suite stayed green. When this pin was written the engine had no
/// `IterMetrics` (drift (b)), so `block_index_groups` was the only available
/// proxy — the gather takes the block-index path **only** if the shard is still
/// cold when it runs, so a prefetch that warmed it shows up as
/// `full_shard_groups` instead.
///
/// Since the fold this arm *does* have counters, so the assertion below reads
/// `prefetch_skipped_block_index` directly as well. Both are kept deliberately:
/// the counter is the prefetch-time decision, `block_index_groups` is the route
/// the gather took, and asserting only one of them would let the other drift.
#[test]
fn engine_does_not_warm_a_block_index_eligible_shard() {
    let dir = tempfile::tempdir().unwrap();
    let engine = framed_engine(dir.path(), /*scatter_block_index*/ true);

    // One row in each of shards 0..3 (64 rows per shard). group_len = 1, so
    // `1 * ROW_RANGE_WINDOW_DIVISOR < 64` — cost-eligible on every shard.
    let plan: Plan = vec![(0u32, 5u64), (0, 70), (0, 140), (0, 200)];
    let it = Arc::clone(&engine).iter_with_plans(into_iter(vec![plan.clone()]), 4, rows_of, gather);
    let iter_metrics = it.iter_metrics();
    let out: Vec<_> = it.map(|r| r.unwrap()).collect();

    let want: Vec<(i32, f32)> = plan.iter().map(|&(_, r)| framed_expected(r)).collect();
    assert_eq!(out, vec![want], "the gather must still be correct");

    let m = engine.cache_metrics();
    use std::sync::atomic::Ordering as AtomicOrdering;
    // The prefetch-side half of the claim, asserted directly since ORG-9.10-1
    // gave this arm `IterMetrics`. Before that this test could only reach for
    // `CacheMetrics::block_index_groups`, which is the *gather's* route — so a
    // prefetch that warmed the shard for some unrelated reason, leaving the
    // gather to take the block-index path anyway, would have read identically.
    assert_eq!(
        iter_metrics
            .prefetch_skipped_block_index
            .load(AtomicOrdering::Relaxed),
        4,
        "all four touched shards must be skipped *as block-index eligible*, not          merely left unwarmed for some other reason (spawned={}, cache_hit={},          in_flight={})",
        iter_metrics
            .prefetch_tasks_spawned
            .load(AtomicOrdering::Relaxed),
        iter_metrics
            .prefetch_skipped_cache_hit
            .load(AtomicOrdering::Relaxed),
        iter_metrics
            .prefetch_skipped_in_flight
            .load(AtomicOrdering::Relaxed),
    );
    assert!(
        m.block_index_groups.load(AtomicOrdering::Relaxed) > 0,
        "premise + claim: every group is framed, cold and sparse, so the gather \
         must take the block-index path — it can only do that if the prefetch \
         left the shard un-warmed (block_index={}, full_shard={})",
        m.block_index_groups.load(AtomicOrdering::Relaxed),
        m.full_shard_groups.load(AtomicOrdering::Relaxed),
    );
    assert_eq!(
        m.full_shard_groups.load(AtomicOrdering::Relaxed),
        0,
        "a warmed shard would have been served full-shard instead"
    );
    for sidx in 0..4 {
        assert!(
            !engine.reader(0).cache_contains(sidx),
            "shard {sidx} must still be cold: the block-index path decodes row \
             groups without inserting the whole shard into the LRU"
        );
    }
}

/// The negative arm of the pin above: with the reader's gate off, the same plan
/// must warm every touched shard and be served full-shard. Together the two
/// fix the branch in place — one of them fails whichever way the predicate is
/// broken.
#[test]
fn engine_warms_the_same_shards_when_the_reader_gate_is_off() {
    let dir = tempfile::tempdir().unwrap();
    let engine = framed_engine(dir.path(), /*scatter_block_index*/ false);

    let plan: Plan = vec![(0u32, 5u64), (0, 70), (0, 140), (0, 200)];
    let out: Vec<_> = Arc::clone(&engine)
        .iter_with_plans(into_iter(vec![plan.clone()]), 4, rows_of, gather)
        .map(|r| r.unwrap())
        .collect();

    let want: Vec<(i32, f32)> = plan.iter().map(|&(_, r)| framed_expected(r)).collect();
    assert_eq!(out, vec![want]);

    let m = engine.cache_metrics();
    use std::sync::atomic::Ordering as AtomicOrdering;
    assert_eq!(
        m.block_index_groups.load(AtomicOrdering::Relaxed),
        0,
        "the per-reader gate is a complete off-switch: no L1 adoption either"
    );
    assert!(m.full_shard_groups.load(AtomicOrdering::Relaxed) > 0);
    for sidx in 0..4 {
        assert!(
            engine.reader(0).cache_contains(sidx),
            "shard {sidx} must be warm with the gate off"
        );
    }
}

/// **Pin (9b).** `from_scx_readers` must apply the gate to **every** reader,
/// not just the first.
///
/// The two tests above are one-file, so they stay green if the setter is
/// applied only to `fid == 0` — and the one production caller,
/// `SparseCellSetLoader`, is *multi*-file by design (that is the whole point of
/// the cell-set loader). A gate that reached reader 0 only would leave every
/// other file on the opposite path, and the aggregate counters would still show
/// both routes exercised, which reads as success.
///
/// Asserted per reader through `cache_contains`, because `cache_metrics()` is a
/// single aggregate over the shared cache and cannot attribute a group to a
/// file.
#[test]
fn from_scx_readers_applies_the_block_index_gate_to_every_reader() {
    for gate in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let p0 = dir.path().join("f0.scx");
        let p1 = dir.path().join("f1.scx");
        write_framed_fixture(&p0);
        write_framed_fixture(&p1);
        let engine = PrefetchEngine::from_scx_readers(
            vec![ScxReader::open(&p0).unwrap(), ScxReader::open(&p1).unwrap()],
            /*cache_shards*/ 16,
            usize::MAX,
            /*default_lookahead*/ 4,
            gate,
        );

        // One row in each of shards 0..3 of *both* files.
        let plan: Plan = vec![
            (0u32, 5u64),
            (0, 70),
            (0, 140),
            (0, 200),
            (1, 5),
            (1, 70),
            (1, 140),
            (1, 200),
        ];
        let out: Vec<_> = Arc::clone(&engine)
            .iter_with_plans(into_iter(vec![plan.clone()]), 4, rows_of, gather)
            .map(|r| r.unwrap())
            .collect();
        let want: Vec<(i32, f32)> = plan.iter().map(|&(_, r)| framed_expected(r)).collect();
        assert_eq!(
            out,
            vec![want],
            "gate={gate}: the gather must still be correct"
        );

        for fid in 0..2u32 {
            for sidx in 0..4 {
                assert_eq!(
                    engine.reader(fid).cache_contains(sidx),
                    !gate,
                    "gate={gate}: file {fid} shard {sidx} — with the gate off every \
                     touched shard warms into the LRU; with it on none of them do. A \
                     gate applied to reader 0 only fails here on file 1."
                );
            }
        }
    }
}

/// **Pre-refactor pin (ORG-9.10-1).** The engine calls `process` for **every**
/// plan, empty ones included — the documented opposite of `IndexPlanIter`,
/// which skips them (`index_plan_tests::iter_skips_empty_plans_mid_stream`).
///
/// The fold has to keep both behaviours, so this is the pin that stops it being
/// resolved silently in whichever direction the merged `next` happens to take.
#[test]
fn engine_calls_process_for_every_plan_including_empty() {
    let dir = tempfile::tempdir().unwrap();
    let engine = two_file_engine(dir.path(), 8);

    let plans: Vec<Plan> = vec![vec![], vec![(0u32, 5u64)], vec![], vec![]];
    let out: Vec<_> = engine
        .iter_with_plans(into_iter(plans), 2, rows_of, gather)
        .map(|r| r.unwrap())
        .collect();

    assert_eq!(
        out.len(),
        4,
        "empty plans are NOT skipped on this arm — `process` runs for each and \
         the consumer decides"
    );
    assert_eq!(out[0], Vec::new());
    assert_eq!(out[1], vec![expected(5)]);
    assert_eq!(out[2], Vec::new());
    assert_eq!(out[3], Vec::new());
}

/// **Pre-refactor pin (ORG-9.10-1, drift (c)).** Dropping the iterator
/// mid-stream releases the plan-pull worker promptly.
///
/// Drift (c) was that the pair arm drained `plan_rx` in `Drop` and this one did
/// not, while both satisfied this test — dropping the `plan_rx` field wakes a
/// parked sender on its own. That is what this pin records: the two were
/// **observably identical here**, so the fold's question was whether the drain
/// earned its place, not how to port it.
///
/// It did not. The drain was carried over briefly and then removed, because it
/// is *not* neutral in a way this test cannot see: it advances the user's plan
/// generator during teardown. See
/// `drop_does_not_keep_pulling_from_an_endless_plan_generator`, which measures
/// exactly that (0 extra pulls without the drain, `lookahead` with it).
#[test]
fn engine_pull_worker_exits_promptly_after_drop() {
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

    let dir = tempfile::tempdir().unwrap();
    let engine = two_file_engine(dir.path(), 8);

    // Owned by the test, not `static` — see the sibling pin in
    // `index_plan_tests::pull_worker_exits_promptly_after_iter_drop`.
    let exited = Arc::new(AtomicBool::new(false));

    struct ExitFlagPlans {
        remaining: usize,
        exited: Arc<AtomicBool>,
    }
    impl Iterator for ExitFlagPlans {
        type Item = Result<Plan>;
        fn next(&mut self) -> Option<Self::Item> {
            if self.remaining == 0 {
                return None;
            }
            self.remaining -= 1;
            Some(Ok(vec![(0u32, 1u64)]))
        }
    }
    impl Drop for ExitFlagPlans {
        fn drop(&mut self) {
            self.exited.store(true, AtomicOrdering::Release);
        }
    }

    let mut it = engine.iter_with_plans(
        ExitFlagPlans {
            remaining: 100_000,
            exited: Arc::clone(&exited),
        },
        4,
        rows_of,
        gather,
    );
    let _first = it.next().expect("first batch").expect("gather");
    assert!(
        !exited.load(AtomicOrdering::Acquire),
        "premise: the worker must still be alive while the iterator is"
    );
    drop(it);

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        if exited.load(AtomicOrdering::Acquire) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    panic!("plan-pull worker still alive 5 s after the iterator dropped");
}

// ---------------------------------------------------------------------
// §9.3 — bounded, GIL-free teardown.
// ---------------------------------------------------------------------

/// **The limit of the prefetch counters**, pinned so the docs cannot overstate
/// them again.
///
/// `spawn_prefetches` returns early when `lookahead == 0`, so every
/// `IterMetrics` counter stays zero — while the *gather* still takes the
/// block-index route and increments `CacheMetrics::block_index_groups`. The two
/// therefore measure different things: `prefetch_skipped_block_index` is an L2
/// prefetch-time decision that only exists when prefetching is enabled, and
/// `block_index_groups` is the route the gather actually took.
///
/// This was documented the wrong way round — as "a disagreement between them is
/// a bug" — which would have made `lookahead=0` read as a defect.
#[test]
fn lookahead_zero_leaves_prefetch_counters_zero_while_the_gather_still_adopts() {
    let dir = tempfile::tempdir().unwrap();
    let engine = framed_engine(dir.path(), /*scatter_block_index*/ true);

    let plan: Plan = vec![(0u32, 5u64), (0, 70), (0, 140), (0, 200)];
    let it = Arc::clone(&engine).iter_with_plans(into_iter(vec![plan.clone()]), 0, rows_of, gather);
    let iter_metrics = it.iter_metrics();
    let out: Vec<_> = it.map(|r| r.unwrap()).collect();

    let want: Vec<(i32, f32)> = plan.iter().map(|&(_, r)| framed_expected(r)).collect();
    assert_eq!(out, vec![want], "the gather must still be correct");

    use std::sync::atomic::Ordering as AtomicOrdering;
    for (name, v) in [
        (
            "prefetch_tasks_spawned",
            iter_metrics
                .prefetch_tasks_spawned
                .load(AtomicOrdering::Relaxed),
        ),
        (
            "prefetch_skipped_cache_hit",
            iter_metrics
                .prefetch_skipped_cache_hit
                .load(AtomicOrdering::Relaxed),
        ),
        (
            "prefetch_skipped_in_flight",
            iter_metrics
                .prefetch_skipped_in_flight
                .load(AtomicOrdering::Relaxed),
        ),
        (
            "prefetch_skipped_block_index",
            iter_metrics
                .prefetch_skipped_block_index
                .load(AtomicOrdering::Relaxed),
        ),
    ] {
        assert_eq!(v, 0, "{name} must be 0 at lookahead=0 — no prefetch ran");
    }

    let m = engine.cache_metrics();
    assert!(
        m.block_index_groups.load(AtomicOrdering::Relaxed) > 0,
        "the gather still adopts the block-index route with no prefetch at all, \
         which is precisely why a zero prefetch counter is not evidence that it \
         did not (block_index={}, full_shard={})",
        m.block_index_groups.load(AtomicOrdering::Relaxed),
        m.full_shard_groups.load(AtomicOrdering::Relaxed),
    );
}

/// **Dropping an iterator must not keep pulling from the user's plan
/// generator.**
///
/// The `plan_rx` drain this PR briefly carried over from the pair arm —
/// `while self.plan_rx.try_recv().is_ok() {}` inside `Drop` — was justified as
/// "bounded by the channel capacity". It is not: the receiver is still
/// *connected* while the loop runs, so every successful `try_recv` frees a slot,
/// the parked pull worker wakes, calls `plans.next()` again and refills it. A
/// large, infinite or side-effecting producer can therefore be advanced far
/// past the lookahead during destruction, and the `BoundedRuntime` deadline
/// does not apply — it is reached only *after* this `Drop` body returns.
///
/// Dropping `plan_rx` with the struct's fields disconnects the channel and
/// wakes a blocked sender on its own, which is what the engine did before the
/// fold. So the drain bought nothing and cost an unbounded pull.
///
/// The existing `engine_pull_worker_exits_promptly_after_drop` cannot see this:
/// its producer is finite (100k) and it only observes that the worker eventually
/// exits *after* `drop` has already returned.
#[test]
fn drop_does_not_keep_pulling_from_an_endless_plan_generator() {
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    let dir = tempfile::tempdir().unwrap();
    let engine = two_file_engine(dir.path(), 8);

    /// Never ends, counts every pull, and announces its own release.
    struct EndlessPlans {
        pulled: Arc<AtomicU64>,
        exited: Arc<std::sync::atomic::AtomicBool>,
    }
    impl Iterator for EndlessPlans {
        type Item = Result<Plan>;
        fn next(&mut self) -> Option<Self::Item> {
            self.pulled.fetch_add(1, AtomicOrdering::AcqRel);
            Some(Ok(vec![(0u32, 1u64)]))
        }
    }
    impl Drop for EndlessPlans {
        fn drop(&mut self) {
            self.exited
                .store(true, std::sync::atomic::Ordering::Release);
        }
    }

    const LOOKAHEAD: usize = 4;
    let pulled = Arc::new(AtomicU64::new(0));
    let exited = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut it = engine.iter_with_plans(
        EndlessPlans {
            pulled: Arc::clone(&pulled),
            exited: Arc::clone(&exited),
        },
        LOOKAHEAD,
        rows_of,
        gather,
    );
    it.next().expect("first batch").expect("gather");

    // Premise, by channel occupancy rather than by pull count: `before` must be
    // sampled at quiescence, and a count threshold cannot see quiescence. The
    // pipeline absorbs anywhere from 6 to 10 pulls before the worker parks
    // (1 handed to `process` + 0..=LOOKAHEAD retained in `in_flight` +
    // LOOKAHEAD in the channel + 1 in the worker's hands), so any fixed count
    // below that range can be satisfied mid-fill on a loaded runner — and the
    // remaining ordinary fill pulls then land after `before` and get miscounted
    // as teardown pulls (observed on 2-core CI: before=8, after=10, extra=2).
    //
    // `plan_rx.is_full()` is the quiescence predicate: once the channel is at
    // capacity the only agent that can free a slot is the consumer, and this
    // thread never calls `next()` again. At most ONE later pull is possible —
    // the worker taking its next item before parking in `send` — which is
    // exactly the `extra <= 1` slack asserted below.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !it.plan_rx.is_full() {
        assert!(
            std::time::Instant::now() < deadline,
            "worker never filled the channel (pulled={}) — premise failed",
            pulled.load(AtomicOrdering::Acquire)
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let before = pulled.load(AtomicOrdering::Acquire);

    let t0 = std::time::Instant::now();
    drop(it);
    let elapsed = t0.elapsed();

    // Conclusion, by completion signal: the producer's `Drop` fires only when
    // the worker's `for item in plans` loop ends. Measured: 0 extra pulls on
    // 8/8 runs without the drain, exactly `LOOKAHEAD` (4) with it — the drain
    // frees one slot per `try_recv` and the worker refills each one. Allow 1
    // for the worker being mid-`next` when the channel disconnects.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !exited.load(AtomicOrdering::Acquire) {
        assert!(
            std::time::Instant::now() < deadline,
            "the pull worker was still running 10s after drop (pulled={})",
            pulled.load(AtomicOrdering::Acquire)
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let after = pulled.load(AtomicOrdering::Acquire);
    let extra = after - before;

    assert!(
        extra <= 1,
        "dropping the iterator pulled {extra} more plans from an endless \
         generator (before={before}, after={after}, lookahead={LOOKAHEAD}). \
         Drop must disconnect and let the worker fail its next `send`, not free \
         slots for it to refill — on the ML path that generator is Python."
    );
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "drop took {elapsed:?}"
    );
}

/// **OPT-FORMATIO-1, the L2 half.** A block-index-eligible shard is still
/// never warmed *whole* (`prefetch_skipped_block_index` counts it, the
/// whole-shard LRU stays cold — the pin above is untouched), but when the
/// plan's row groups fit `budget / (lookahead + 1)` the prefetcher pre-decodes
/// them into the shard LRU, and the gather serves every one as a
/// `row_group_hit`. The same single plan at `lookahead = 0` (no prefetch at
/// all) decodes the same groups on the gather instead — one plan, not two,
/// because a second identical plan would hit through L1 regardless of what
/// the prefetcher did.
#[test]
fn engine_warms_row_groups_for_an_eligible_plan() {
    use std::sync::atomic::Ordering as AtomicOrdering;
    // One row in each of shards 0..3 → one group per shard, 4 groups.
    let plan: Plan = vec![(0u32, 5u64), (0, 70), (0, 140), (0, 200)];
    let planned = 4 * FRAMED_GROUP_BYTES;
    // Fits its share with lookahead 4: budget / 5 ≥ planned.
    let budget = 8 * planned;

    let dir = tempfile::tempdir().unwrap();
    let engine = framed_engine_with_budget(dir.path(), true, budget);
    assert_eq!(
        engine.reader(0).cache_bytes_budget(),
        budget,
        "premise: finite budget"
    );
    let it = Arc::clone(&engine).iter_with_plans(into_iter(vec![plan.clone()]), 4, rows_of, gather);
    let iter_metrics = it.iter_metrics();
    let out: Vec<_> = it.map(|r| r.unwrap()).collect();
    let want: Vec<(i32, f32)> = plan.iter().map(|&(_, r)| framed_expected(r)).collect();
    assert_eq!(out, vec![want.clone()], "the gather must still be correct");

    let m = engine.cache_metrics();
    assert_eq!(
        iter_metrics
            .prefetch_skipped_block_index
            .load(AtomicOrdering::Relaxed),
        4,
        "still counted as block-index eligible — the counter is a contract"
    );
    assert_eq!(
        iter_metrics
            .prefetch_tasks_spawned
            .load(AtomicOrdering::Relaxed),
        0,
        "a row-group warm is not a whole-shard spawn (the four-way partition holds)"
    );
    for sidx in 0..4 {
        assert!(
            !engine.reader(0).cache_contains(sidx),
            "shard {sidx} must not be resident whole"
        );
    }
    assert_eq!(m.full_shard_groups.load(AtomicOrdering::Relaxed), 0);
    assert!(m.block_index_groups.load(AtomicOrdering::Relaxed) > 0);
    assert_eq!(
        m.row_group_misses.load(AtomicOrdering::Relaxed),
        4,
        "the prefetcher decoded each touched group once"
    );
    assert_eq!(
        m.row_group_hits.load(AtomicOrdering::Relaxed),
        4,
        "the gather found every group already resident"
    );
    assert_eq!(m.row_group_evictions.load(AtomicOrdering::Relaxed), 0);

    // lookahead = 0: no prefetch ran, so the gather itself decodes the groups.
    let dir0 = tempfile::tempdir().unwrap();
    let engine0 = framed_engine_with_budget(dir0.path(), true, budget);
    let out0: Vec<_> = Arc::clone(&engine0)
        .iter_with_plans(into_iter(vec![plan.clone()]), 0, rows_of, gather)
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(out0, vec![want]);
    let m0 = engine0.cache_metrics();
    assert_eq!(m0.row_group_misses.load(AtomicOrdering::Relaxed), 4);
    assert_eq!(
        m0.row_group_hits.load(AtomicOrdering::Relaxed),
        0,
        "nothing pre-decoded them: a hit here would mean a warm ran at lookahead 0"
    );
}

/// The negative arm: a plan whose row groups do **not** fit their share of the
/// budget is neither warmed nor retained. The verdict is one per plan and
/// reaches the gather through `process`'s third argument, so the L1 gather
/// bypasses the LRU too — the budget is chosen so the whole plan fits the
/// *cache* (an L1-only rule against the full budget would admit it) but not
/// `budget / (lookahead + 1)`. Falsifiers: a warm that ran anyway shows up as
/// hits; a gather that admitted on its own shows up as inserted bytes.
#[test]
fn engine_bypasses_row_groups_over_their_budget_share() {
    use std::sync::atomic::Ordering as AtomicOrdering;
    let plan: Plan = vec![(0u32, 5u64), (0, 70), (0, 140), (0, 200)];
    let planned = 4 * FRAMED_GROUP_BYTES;
    // 2 × planned holds every group, but / 5 is under it.
    let budget = 2 * planned;
    assert!(
        budget / 5 < planned && budget >= planned,
        "premise: fits the cache, not the share"
    );

    let dir = tempfile::tempdir().unwrap();
    let engine = framed_engine_with_budget(dir.path(), true, budget);
    let it = Arc::clone(&engine).iter_with_plans(into_iter(vec![plan.clone()]), 4, rows_of, gather);
    let iter_metrics = it.iter_metrics();
    let out: Vec<_> = it.map(|r| r.unwrap()).collect();
    let want: Vec<(i32, f32)> = plan.iter().map(|&(_, r)| framed_expected(r)).collect();
    assert_eq!(out, vec![want]);

    let m = engine.cache_metrics();
    assert_eq!(
        iter_metrics
            .prefetch_skipped_block_index
            .load(AtomicOrdering::Relaxed),
        4
    );
    assert_eq!(
        m.row_group_misses.load(AtomicOrdering::Relaxed),
        4,
        "the gather decoded them"
    );
    assert_eq!(
        m.row_group_hits.load(AtomicOrdering::Relaxed),
        0,
        "no warm ran: the plan's groups exceed budget / (lookahead + 1)"
    );
    assert_eq!(
        m.row_group_bytes_inserted.load(AtomicOrdering::Relaxed),
        0,
        "and the gather did not retain them either — one verdict per plan"
    );
    assert_eq!(engine.reader(0).cache_bytes_used(), 0);
}

/// **Review on #528 (Antigravity, codex).** The admission verdict is over the
/// whole plan, not per file: two files each fit `budget / (lookahead + 1)` on
/// their own, their union does not, and the shared cache sees the union. A
/// per-file verdict warmed and retained both halves and let them evict each
/// other. Falsifier: `row_group_bytes_inserted > 0` (or any hit) means a
/// per-file decision admitted.
#[test]
fn engine_admits_row_groups_per_plan_not_per_file() {
    use std::sync::atomic::Ordering as AtomicOrdering;
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("f0.scx");
    let p1 = dir.path().join("f1.scx");
    write_framed_fixture(&p0);
    write_framed_fixture(&p1);
    // One row in each of shards 0..3 of BOTH files: 4 groups per file.
    let per_file = 4 * FRAMED_GROUP_BYTES;
    // share ∈ [per_file, 2 × per_file): each file fits, the union does not.
    let budget = 5 * (per_file + per_file / 2);
    assert!(
        budget / 5 >= per_file && budget / 5 < 2 * per_file,
        "premise"
    );
    let engine = PrefetchEngine::from_scx_readers(
        vec![ScxReader::open(&p0).unwrap(), ScxReader::open(&p1).unwrap()],
        8,
        budget,
        4,
        true,
    );
    let plan: Plan = vec![
        (0u32, 5u64),
        (0, 70),
        (0, 140),
        (0, 200),
        (1, 5),
        (1, 70),
        (1, 140),
        (1, 200),
    ];
    let it = Arc::clone(&engine).iter_with_plans(into_iter(vec![plan.clone()]), 4, rows_of, gather);
    let iter_metrics = it.iter_metrics();
    let out: Vec<_> = it.map(|r| r.unwrap()).collect();
    let want: Vec<(i32, f32)> = plan.iter().map(|&(_, r)| framed_expected(r)).collect();
    assert_eq!(out, vec![want]);

    let m = engine.cache_metrics();
    assert_eq!(
        iter_metrics
            .prefetch_skipped_block_index
            .load(AtomicOrdering::Relaxed),
        8,
        "all eight (file, shard) buckets are block-index eligible"
    );
    assert_eq!(m.row_group_misses.load(AtomicOrdering::Relaxed), 8);
    assert_eq!(
        m.row_group_hits.load(AtomicOrdering::Relaxed)
            + m.row_group_bytes_inserted.load(AtomicOrdering::Relaxed),
        0,
        "the union is over the share: nothing warmed, nothing retained"
    );

    // Control: the same plan on one file alone fits, is warmed, and hits.
    let dir1 = tempfile::tempdir().unwrap();
    let engine1 = framed_engine_with_budget(dir1.path(), true, budget);
    let half: Plan = vec![(0u32, 5u64), (0, 70), (0, 140), (0, 200)];
    let want1: Vec<(i32, f32)> = half.iter().map(|&(_, r)| framed_expected(r)).collect();
    let out1: Vec<_> = Arc::clone(&engine1)
        .iter_with_plans(into_iter(vec![half]), 4, rows_of, gather)
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(out1, vec![want1]);
    let m1 = engine1.cache_metrics();
    assert_eq!(
        m1.row_group_hits.load(AtomicOrdering::Relaxed),
        4,
        "control: one file fits and hits"
    );
}
