// Paired-gather correctness → owning tests:
//   pair-dedup + plan-order (reused ctrl, pert==ctrl)
//        ......................................... fused_paired_gather_preserves_duplicate_rows_and_same_row_pairs

use super::*;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc as StdArc;

// Named explicitly since ORG-9.10-1: the iterator machinery these tests drive
// now lives in `plan_engine`, so `index_plan` no longer re-exports the channel
// and atomic types through `use super::*`.
use crossbeam_channel::{bounded, Receiver};

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;

/// Build a minimal multi-shard `.scx` file. Each row `r` has a single
/// non-zero at column `r % n_vars` with value `((r + 1) & 0xFF) as u8`,
/// so `(row, col, value)` is recoverable from the row index alone.
fn write_multi_shard_fixture(
    path: &std::path::Path,
    n_obs: usize,
    n_vars: usize,
    n_shards: usize,
) -> std::path::PathBuf {
    assert!(
        n_obs.is_multiple_of(n_shards),
        "n_obs must divide n_shards in this fixture"
    );
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
            let col = (row % n_vars) as u32;
            let val = ((row + 1) & 0xFF) as u8;
            indices.push(col);
            values.push(val);
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
    path.to_path_buf()
}

fn open_loader(path: &std::path::Path, sort_by_shard: bool) -> IndexPlanLoader {
    let config = LoaderConfig {
        normalize: false,
        log1p: false,
        obs_columns: vec!["cell_id".to_string()],
        ..Default::default()
    };
    IndexPlanLoader::new(
        path,
        config,
        /*cache_shards*/ 4,
        sort_by_shard,
        /*lookahead*/ 4,
        /*max_plan_size*/ 16384,
    )
    .unwrap()
}

fn open_loader_hvg(
    path: &std::path::Path,
    sort_by_shard: bool,
    hvg_indices: Vec<u32>,
) -> IndexPlanLoader {
    let config = LoaderConfig {
        normalize: false,
        log1p: false,
        obs_columns: vec!["cell_id".to_string()],
        hvg_indices: Some(hvg_indices),
        ..Default::default()
    };
    IndexPlanLoader::new(
        path,
        config,
        /*cache_shards*/ 4,
        sort_by_shard,
        /*lookahead*/ 4,
        /*max_plan_size*/ 16384,
    )
    .unwrap()
}

fn assert_full_fixture_row(row: u64, dense: &[f32], n_vars: usize) {
    let col = (row as usize) % n_vars;
    let val = ((row as usize + 1) & 0xFF) as f32;
    assert_eq!(dense[col], val, "row {row} expected value at col {col}");
    assert_eq!(
        dense.iter().filter(|&&v| v != 0.0).count(),
        1,
        "row {row} should have one nonzero"
    );
}

fn assert_hvg_fixture_row(row: u64, dense: &[f32], hvg_indices: &[u32], n_vars: usize) {
    let col = ((row as usize) % n_vars) as u32;
    let expected_pos = hvg_indices.iter().position(|&h| h == col);
    match expected_pos {
        Some(pos) => {
            assert_eq!(
                dense[pos],
                ((row as usize + 1) & 0xFF) as f32,
                "row {row} expected HVG col {col} at projected pos {pos}"
            );
            assert_eq!(
                dense.iter().filter(|&&v| v != 0.0).count(),
                1,
                "row {row} should have one projected nonzero"
            );
        }
        None => assert!(
            dense.iter().all(|&v| v == 0.0),
            "row {row} should project to an all-zero HVG row"
        ),
    }
}

fn categorical_strings(col: &ObsColumn) -> Vec<String> {
    match col {
        ObsColumn::Categorical(codes, categories) => codes
            .iter()
            .map(|&code| categories[code as usize].clone())
            .collect(),
        other => panic!("expected categorical obs column, got {other:?}"),
    }
}

#[test]
fn fused_paired_gather_preserves_duplicate_rows_and_same_row_pairs() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
    let loader = open_loader(&path, false);

    let plan: Vec<(u64, u64)> = vec![(1, 5), (2, 5), (5, 5), (9, 1), (1, 9)];
    let batch = loader.process_plan(plan.clone()).unwrap();
    let n_cols = loader.n_output_cols();

    assert_eq!(batch.pairs, plan);
    for (i, &(pert, ctrl)) in batch.pairs.iter().enumerate() {
        let p_row = &batch.x[i * n_cols..][..n_cols];
        let c_row = &batch.x_paired[i * n_cols..][..n_cols];
        assert_full_fixture_row(pert, p_row, n_cols);
        assert_full_fixture_row(ctrl, c_row, n_cols);
        if pert == ctrl {
            let p_bits: Vec<u32> = p_row.iter().map(|v| v.to_bits()).collect();
            let c_bits: Vec<u32> = c_row.iter().map(|v| v.to_bits()).collect();
            assert_eq!(p_bits, c_bits, "same-row pair should produce equal rows");
        }
    }
}

#[test]
fn fused_paired_gather_handles_hvg_projection_and_empty_projected_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
    let hvg = vec![1u32, 5u32];
    let loader = open_loader_hvg(&path, false, hvg.clone());

    let plan: Vec<(u64, u64)> = vec![(5, 1), (9, 5), (2, 10), (13, 5)];
    let batch = loader.process_plan(plan.clone()).unwrap();
    let n_cols = loader.n_output_cols();
    assert_eq!(n_cols, hvg.len());

    for (i, &(pert, ctrl)) in batch.pairs.iter().enumerate() {
        let p_row = &batch.x[i * n_cols..][..n_cols];
        let c_row = &batch.x_paired[i * n_cols..][..n_cols];
        assert_hvg_fixture_row(pert, p_row, &hvg, 8);
        assert_hvg_fixture_row(ctrl, c_row, &hvg, 8);
    }
}

#[test]
fn fused_paired_gather_keeps_obs_aligned_after_shard_sort() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
    let loader = open_loader(&path, true);

    let plan: Vec<(u64, u64)> = vec![(15, 0), (4, 5), (8, 9), (3, 12), (10, 11), (1, 2)];
    let batch = loader.process_plan(plan).unwrap();
    let obs = categorical_strings(batch.obs.get("cell_id").unwrap());
    let obs_paired = categorical_strings(batch.obs_paired.get("cell_id").unwrap());

    assert_eq!(obs.len(), batch.pairs.len());
    assert_eq!(obs_paired.len(), batch.pairs.len());
    for (i, &(pert, ctrl)) in batch.pairs.iter().enumerate() {
        assert_eq!(obs[i], format!("cell_{pert}"));
        assert_eq!(obs_paired[i], format!("cell_{ctrl}"));
    }
}

/// Phase 2.4: post-sort invariant — `pairs[i]` aligns with `x[i]` and
/// `x_paired[i]`, and `pairs` is monotonically non-decreasing in
/// `min(shard_of(p), shard_of(c))` after the sort.
#[test]
fn sort_by_shard_aligns_pairs_with_rows_and_is_monotone() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
    let loader = open_loader(&path, /*sort_by_shard*/ true);

    // Plan deliberately scrambled across shards.
    let plan: Vec<(u64, u64)> = vec![
        (15, 0),  // shards (3, 0) -> min 0
        (4, 5),   // shards (1, 1) -> min 1
        (8, 9),   // shards (2, 2) -> min 2
        (3, 12),  // shards (0, 3) -> min 0
        (10, 11), // shards (2, 2) -> min 2
        (1, 2),   // shards (0, 0) -> min 0
    ];
    let batch = loader.process_plan(plan.clone()).unwrap();
    let n_cols = loader.n_output_cols();

    // Pairs must align with the rows of x / x_paired: each row encodes
    // (row_idx % n_vars, (row_idx + 1) & 0xFF) in our fixture.
    for i in 0..batch.pairs.len() {
        let (p, c) = batch.pairs[i];
        let p_out = &batch.x[i * n_cols..][..n_cols];
        let c_out = &batch.x_paired[i * n_cols..][..n_cols];

        let p_col = (p as usize) % n_cols;
        let c_col = (c as usize) % n_cols;
        assert_eq!(p_out[p_col], ((p as usize + 1) & 0xFF) as f32);
        assert_eq!(c_out[c_col], ((c as usize + 1) & 0xFF) as f32);
    }

    // Post-sort key is non-decreasing.
    let keys: Vec<usize> = batch
        .pairs
        .iter()
        .map(|&(p, c)| {
            let sp = loader.shard_of(p).unwrap();
            let sc = loader.shard_of(c).unwrap();
            sp.min(sc)
        })
        .collect();
    for w in keys.windows(2) {
        assert!(
            w[0] <= w[1],
            "post-sort plan must be non-decreasing in min-shard"
        );
    }
}

/// Phase 2.4: parity invariant — for the same input plan, sorted vs
/// unsorted runs produce the same `(pair, x_row, x_paired_row)` *set*
/// (just permuted). HVG-on and HVG-off both verified.
#[test]
fn sort_by_shard_is_a_pure_permutation_of_unsorted_output() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);

    let plan: Vec<(u64, u64)> = vec![(15, 0), (4, 5), (8, 9), (3, 12), (10, 11), (1, 2)];

    let unsorted = open_loader(&path, false)
        .process_plan(plan.clone())
        .unwrap();
    let sorted = open_loader(&path, true).process_plan(plan.clone()).unwrap();

    // Unsorted preserves input plan order (sanity).
    assert_eq!(unsorted.pairs, plan);

    // Build (pair, row, paired_row) tuples and compare as multisets.
    let n_cols = unsorted.x.len() / unsorted.pairs.len();
    /// `(pair, row bits, paired-row bits)` — compared as a multiset.
    type Triples = Vec<((u64, u64), Vec<u32>, Vec<u32>)>;
    let triples = |b: &IndexPlanBatch| -> Triples {
        (0..b.pairs.len())
            .map(|i| {
                let r: Vec<u32> = b.x[i * n_cols..][..n_cols]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                let pr: Vec<u32> = b.x_paired[i * n_cols..][..n_cols]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                (b.pairs[i], r, pr)
            })
            .collect()
    };
    let mut unsorted_triples = triples(&unsorted);
    let mut sorted_triples = triples(&sorted);
    unsorted_triples.sort_by_key(|t| t.0);
    sorted_triples.sort_by_key(|t| t.0);
    assert_eq!(unsorted_triples, sorted_triples);
}

/// Empty plan + sort_by_shard=true must short-circuit cleanly.
#[test]
fn sort_by_shard_handles_empty_plan() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 8, 4, 2);
    let loader = open_loader(&path, true);
    let batch = loader.process_plan(Vec::new()).unwrap();
    assert!(batch.x.is_empty());
    assert!(batch.x_paired.is_empty());
    assert!(batch.pairs.is_empty());
}

// ---------------------------------------------------------------------
// Phase 4 — iter_with_plans
// ---------------------------------------------------------------------

/// Helper: collect all plans from a Vec into the iterator-of-Result form
/// that `iter_with_plans` expects.
fn into_plan_iter(
    plans: Vec<Vec<(u64, u64)>>,
) -> impl Iterator<Item = std::result::Result<Vec<(u64, u64)>, LoaderError>> + Send + 'static {
    plans.into_iter().map(Ok)
}

fn open_loader_arc(path: &std::path::Path) -> Arc<IndexPlanLoader> {
    let config = LoaderConfig {
        normalize: false,
        log1p: false,
        obs_columns: vec!["cell_id".to_string()],
        ..Default::default()
    };
    Arc::new(
        IndexPlanLoader::new(
            path, config, /*cache_shards*/ 4, /*sort_by_shard*/ true,
            /*lookahead*/ 4, /*max_plan_size*/ 16384,
        )
        .unwrap(),
    )
}

/// Iterator yields one batch per plan, batches are correctly aligned.
#[test]
fn iter_with_plans_yields_one_batch_per_plan() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
    let loader = open_loader_arc(&path);

    let plans = vec![
        vec![(0u64, 1u64), (2, 3)],
        vec![(4, 5)],
        vec![(8, 12), (10, 15)],
    ];
    let it = loader.iter_with_plans(into_plan_iter(plans.clone()), 4);
    let batches: Vec<_> = it.map(|r| r.unwrap()).collect();

    assert_eq!(batches.len(), 3);
    // Each batch has the expected number of pairs (post-sort, but the
    // batch contents are a permutation of the plan).
    for (i, b) in batches.iter().enumerate() {
        assert_eq!(b.pairs.len(), plans[i].len());
        // Build sorted multisets for set equality.
        let mut got = b.pairs.clone();
        got.sort();
        let mut want = plans[i].clone();
        want.sort();
        assert_eq!(got, want);
    }
}

/// Lookahead = 0 vs lookahead = 4 must produce identical outputs (up to
/// the existing sort_by_shard semantics).
#[test]
fn iter_with_plans_lookahead_zero_vs_four_parity() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
    let plans = vec![
        vec![(15u64, 0u64), (4, 5), (8, 9)],
        vec![(3, 12), (10, 11), (1, 2)],
    ];

    let it_zero = open_loader_arc(&path).iter_with_plans(into_plan_iter(plans.clone()), 0);
    let it_four = open_loader_arc(&path).iter_with_plans(into_plan_iter(plans.clone()), 4);

    let zero: Vec<_> = it_zero.map(|r| r.unwrap()).collect();
    let four: Vec<_> = it_four.map(|r| r.unwrap()).collect();

    assert_eq!(zero.len(), four.len());
    for (a, b) in zero.iter().zip(four.iter()) {
        assert_eq!(
            a.pairs, b.pairs,
            "pair order should match between lookahead=0 and 4"
        );
        assert_eq!(a.x, b.x);
        assert_eq!(a.x_paired, b.x_paired);
    }
}

/// Errors injected into the plan stream propagate as Err items in order.
#[test]
fn iter_with_plans_propagates_plan_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 8, 4, 2);
    let loader = open_loader_arc(&path);

    let plans: Vec<std::result::Result<Vec<(u64, u64)>, LoaderError>> = vec![
        Ok(vec![(0, 1)]),
        Err(LoaderError::ChannelError("synthetic".into())),
        Ok(vec![(2, 3)]),
    ];
    let mut it = loader.iter_with_plans(plans.into_iter(), 2);

    // First a successful batch, then the error, then iteration stops
    // (sticky `plan_stream_error`).
    assert!(matches!(it.next(), Some(Ok(_))));
    let second = it.next().expect("second item");
    match second {
        Err(LoaderError::ChannelError(s)) => assert!(s.contains("synthetic")),
        Err(other) => panic!("expected ChannelError, got {other}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
    assert!(
        it.next().is_none(),
        "iteration must stop after a plan-stream error"
    );
}

/// Out-of-range row in a plan yields a Result::Err(IndexOutOfRange) on
/// that batch and stops iteration. (Validation is per-batch, not in the
/// pull thread.)
#[test]
fn iter_with_plans_propagates_decode_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 8, 4, 2);
    let loader = open_loader_arc(&path);

    let plans = vec![vec![(0u64, 1u64)], vec![(99, 99)]];
    let it = loader.iter_with_plans(into_plan_iter(plans), 2);
    let mut results = it;

    let first = results.next().expect("first batch");
    assert!(first.is_ok());
    let second = results.next().expect("second batch");
    match second {
        Err(LoaderError::IndexOutOfRange { idx, .. }) => assert_eq!(idx, 99),
        Err(other) => panic!("expected IndexOutOfRange, got LoaderError: {other}"),
        Ok(_) => panic!("expected IndexOutOfRange, got Ok"),
    }
}

/// Drop mid-iteration must not deadlock. Build an iterator with a long
/// plan stream, take 1 batch, then drop.
#[test]
fn iter_with_plans_drop_mid_iteration() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
    let loader = open_loader_arc(&path);

    let plans: Vec<_> = (0..1000).map(|_| vec![(0u64, 1u64)]).collect();
    let mut it = loader.iter_with_plans(into_plan_iter(plans), 4);
    let _first = it.next().unwrap().unwrap();
    drop(it); // must not hang
}

/// **Pre-refactor pin (ORG-9.10-1).** `IndexPlanIter::next` loops so an empty
/// plan yields no batch and iteration continues — the SCX-DATA-LOADER spec's
/// "plan list is empty → yield no batch for that plan; continue to the next".
///
/// `PlanPrefetchIter`, the engine this iterator is now an adapter over, does
/// the **opposite** by design (`plan_engine.rs`: "Empty plans are NOT skipped —
/// `process` is called for every plan"). That divergence was a fourth drift the
/// review's list of three did not name, and it was the constraint the fold had
/// to satisfy: it is met in `IndexPlanIter::next`, which discards the zero-row
/// batch `process_plan` returns for an empty plan, leaving the engine's contract
/// alone. Not by filtering the plan *stream* — that was tried, and it put the
/// skip upstream of the pull worker's only cancellation point (see
/// `dropping_the_iter_stops_an_endless_empty_plan_generator`).
/// Before the fold it was pinned only from Python
/// (`pyscx/tests/test_index_plan_dataset.py::test_empty_plan_inside_stream_is_skipped`),
/// so a Rust-side fold would have gone green here and red only in the Python
/// suite. See `plan_engine_tests::engine_calls_process_for_every_plan_including_empty`
/// for the mirror-image pin on the other arm.
#[test]
fn iter_skips_empty_plans_mid_stream() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
    let loader = open_loader_arc(&path);

    // Empty plans at the head, in the middle, at the tail, and back to back:
    // the skip is a `continue` inside `next`, so consecutive empties must all
    // be consumed within one `next` call rather than each ending the stream.
    let plans = vec![
        vec![],
        vec![(0u64, 1u64)],
        vec![],
        vec![],
        vec![(2u64, 3u64)],
        vec![],
    ];
    let batches: Vec<_> = loader
        .iter_with_plans(into_plan_iter(plans), 2)
        .map(|r| r.unwrap())
        .collect();

    assert_eq!(
        batches.len(),
        2,
        "four empty plans must yield no batches; got {} batches",
        batches.len()
    );
    assert_eq!(batches[0].pairs, vec![(0, 1)]);
    assert_eq!(batches[1].pairs, vec![(2, 3)]);
}

/// **Pre-refactor pin (ORG-9.10-1, drift (c)).** Dropping the iterator
/// mid-stream must release the plan-pull worker *promptly*, not eventually.
///
/// **Be precise about which contract this pins**, because the obvious reading is
/// wrong: it is *not* a pin on `IndexPlanIter::drop`'s
/// `while self.plan_rx.try_recv().is_ok() {}` drain. The observation starts only
/// after `drop(it)` has returned, and by then field destruction has closed
/// `plan_rx` regardless — so deleting the drain leaves this green. What it does
/// pin is that the receiver is *released at all*: it fails if something retains
/// it past the iterator's lifetime, which is how it was watched red (a leaked
/// `plan_rx` clone in `Drop`).
///
/// That distinction is the point rather than a caveat. Drift (c) was that this
/// arm drained and `PlanPrefetchIter` did not, and both reached the same
/// observable outcome here — so the fold's real question was whether the drain
/// bought anything at all. It did not, and it was not neutral either: it
/// advances the user's plan generator during teardown, which
/// `plan_engine_tests::drop_does_not_keep_pulling_from_an_endless_plan_generator`
/// measures and which this test cannot see. The drain is gone; both arms now
/// rely on field destruction to wake the sender.
///
/// The generator blocks in `send` once the bounded channel fills, so a worker
/// that stays parked keeps `exited` false and the test fails on its deadline
/// rather than hanging CI. `plan_engine_tests::engine_pull_worker_exits_promptly_after_drop`
/// pins the same receiver-release contract on the other arm.
#[test]
fn pull_worker_exits_promptly_after_iter_drop() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};

    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
    let loader = open_loader_arc(&path);

    // Owned by the test, not `static`: cargo runs this module's tests on shared
    // threads within one process, so process-global counters would couple this
    // test to any future sibling that reuses them.
    let exited = Arc::new(AtomicBool::new(false));
    let produced = Arc::new(AtomicUsize::new(0));

    /// Sets `exited` when the pull worker drops it — which happens when the
    /// worker's `for item in plans` loop ends, i.e. when its `send` fails.
    struct ExitFlagPlans {
        remaining: usize,
        exited: Arc<AtomicBool>,
        produced: Arc<AtomicUsize>,
    }
    impl Iterator for ExitFlagPlans {
        type Item = std::result::Result<Vec<(u64, u64)>, LoaderError>;
        fn next(&mut self) -> Option<Self::Item> {
            if self.remaining == 0 {
                return None;
            }
            self.remaining -= 1;
            self.produced.fetch_add(1, AtomicOrdering::Relaxed);
            Some(Ok(vec![(0u64, 1u64)]))
        }
    }
    impl Drop for ExitFlagPlans {
        fn drop(&mut self) {
            self.exited.store(true, AtomicOrdering::Release);
        }
    }

    let mut it = loader.iter_with_plans(
        ExitFlagPlans {
            remaining: 100_000,
            exited: Arc::clone(&exited),
            produced: Arc::clone(&produced),
        },
        4,
    );
    let _first = it.next().unwrap().unwrap();
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
    panic!(
        "plan-pull worker still alive 5 s after the iterator dropped \
         (produced {} plans)",
        produced.load(AtomicOrdering::Relaxed)
    );
}

// ---------------------------------------------------------------------
// Phase 5 — memory budget + auto-tuning
// ---------------------------------------------------------------------

/// Build a multi-shard fixture with a tunable nnz_per_row, so the
/// memory-budget tests can dial in the relative weight of the LRU cache.
fn write_dense_fixture(
    path: &std::path::Path,
    n_obs: usize,
    n_vars: usize,
    n_shards: usize,
    nnz_per_row: usize,
) -> std::path::PathBuf {
    assert!(n_obs.is_multiple_of(n_shards));
    assert!(nnz_per_row <= n_vars);
    let rows_per_shard = n_obs / n_shards;
    let total_nnz = (n_obs * nnz_per_row) as u64;

    let header = FileHeader::new_single_modality(
        n_obs as u64,
        n_vars as u64,
        total_nnz,
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
            for k in 0..nnz_per_row {
                let col = ((row + k * 7919) % n_vars) as u32;
                indices.push(col);
                values.push(((row + k + 1) & 0xFF) as u8);
            }
            indptr.push(*indptr.last().unwrap() + nnz_per_row as u64);
        }
        // Indices must be sorted within each row for the CSR format;
        // sort each row's slice.
        for local in 0..rows_per_shard {
            let lo = indptr[local] as usize;
            let hi = indptr[local + 1] as usize;
            let mut pairs: Vec<(u32, u8)> = (lo..hi).map(|j| (indices[j], values[j])).collect();
            pairs.sort_by_key(|&(c, _)| c);
            pairs.dedup_by_key(|&mut (c, _)| c);
            let new_lo = lo;
            for (j, (c, v)) in pairs.iter().enumerate() {
                indices[new_lo + j] = *c;
                values[new_lo + j] = *v;
            }
            // dedup may shorten — fix the indptr accordingly by rebuilding
            // (rare; skip for simplicity if no dups).
            let _ = (new_lo,);
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
    path.to_path_buf()
}

/// Generous memory budget — both effective values match the requested.
#[test]
fn budget_generous_no_autotune() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 32, 8, 4);
    // Way over the budget needed for this tiny file.
    let config = LoaderConfig {
        max_memory_mb: 4096,
        ..Default::default()
    };
    let loader = IndexPlanLoader::new(
        &path, config, /*cache_shards*/ 8, /*sort_by_shard*/ true, /*lookahead*/ 4,
        /*max_plan_size*/ 1024,
    )
    .unwrap();
    assert_eq!(loader.effective_cache_shards(), 8);
    assert_eq!(loader.effective_lookahead(), 4);
}

/// Tight budget — auto-tune kicks in, lookahead reduced first.
/// Sized so the lookahead overhead dominates the over-budget margin
/// (max_plan_size=65536 → 1 MB per lookahead unit).
#[test]
fn budget_tight_reduces_lookahead_first() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 256, 8, 4);

    // Budget components at requested settings (post-L1-dedup-gather model):
    //   python       = 50 MB
    //   batch buffer = 2 × 65536 × 8 × 4         ≈ 4 MB
    //   lookahead    = 8 × 65536 × 16            ≈ 8 MB
    //   transient    = 2 × 65536 × 64 (no obs)   ≈ 8 MB  (dedup scratch:
    //                  unique_rows + row_to_requests + row_to_pos, per-row)
    //   shard cache  = ~negligible (sparse fixture)
    // Total ≈ 70 MB. Floor at lookahead=1: 50+4+1+8 ≈ 63 MB.
    // A 66 MB budget forces lookahead reduction without floor-failure.
    let config = LoaderConfig {
        max_memory_mb: 66,
        ..Default::default()
    };
    let loader = IndexPlanLoader::new(
        &path, config, /*cache_shards*/ 8, /*sort_by_shard*/ true, /*lookahead*/ 8,
        /*max_plan_size*/ 65536,
    )
    .unwrap();
    assert!(
        loader.effective_lookahead() < 8,
        "lookahead should be reduced under tight budget; got {}",
        loader.effective_lookahead()
    );
    assert!(
        loader.effective_lookahead() >= 1,
        "lookahead floor is 1; got {}",
        loader.effective_lookahead()
    );
    // Cache shards should NOT have been touched yet.
    assert_eq!(loader.effective_cache_shards(), 8);
}

/// Even tighter budget — lookahead at the floor (1), cache_shards reduced
/// further. Verifies the "reduce cache_shards next" branch using a dense
/// fixture so the LRU shard cache has meaningful weight.
#[test]
fn budget_very_tight_reduces_cache_shards() {
    let dir = tempfile::tempdir().unwrap();
    // 1024 rows × 64 vars × 8 shards × 32 nnz/row.
    // shard_decoded ≈ (32 × 128 × 8) + (128 × 8) ≈ 33 KB per shard.
    // 16 cache shards ≈ 528 KB.
    //
    // Actually for a meaningful cache contribution we need much higher
    // density. Bump nnz_per_row.
    let path = write_dense_fixture(&dir.path().join("f.scx"), 1024, 4096, 8, 2048);
    // shard_decoded ≈ (2048 × 128 × 8) + (128 × 8) ≈ 2.1 MB per shard.
    // 16 cache shards ≈ 33 MB.
    //
    // Budget at requested settings:
    //   python       = 50 MB
    //   batch buffer = 2 × 1024 × 4096 × 4 ≈ 32 MB
    //   lookahead    = 4 × 1024 × 16       ≈ 64 KB
    //   shard cache  = 16 × 2.1 MB         ≈ 33 MB
    // Total ≈ 115 MB. Budget 96 forces lookahead → 1, then cache_shards.

    let config = LoaderConfig {
        max_memory_mb: 96,
        ..Default::default()
    };
    let loader = IndexPlanLoader::new(
        &path, config, /*cache_shards*/ 16, /*sort_by_shard*/ true, /*lookahead*/ 4,
        /*max_plan_size*/ 1024,
    )
    .unwrap();
    assert_eq!(
        loader.effective_lookahead(),
        1,
        "lookahead should be at the floor (got {})",
        loader.effective_lookahead()
    );
    assert!(
        loader.effective_cache_shards() < 16,
        "cache_shards should also be reduced; got {}",
        loader.effective_cache_shards()
    );
    assert!(
        loader.effective_cache_shards() >= 1,
        "cache_shards floor is 1; got {}",
        loader.effective_cache_shards()
    );
}

/// Budget below the floor — construction must fail with a clear ConfigError.
#[test]
fn budget_below_floor_refuses_construction() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 4096, 4096, 4);

    // Below the 50 MB python overhead alone.
    let config = LoaderConfig {
        max_memory_mb: 40,
        ..Default::default()
    };
    let result = IndexPlanLoader::new(
        &path, config, /*cache_shards*/ 4, /*sort_by_shard*/ true, /*lookahead*/ 2,
        /*max_plan_size*/ 4096,
    );
    let err = match result {
        Ok(_) => panic!("expected ConfigError, got Ok"),
        Err(e) => e,
    };
    match err {
        LoaderError::ConfigError { reason } => {
            assert!(
                reason.contains("max_memory_mb=40"),
                "error should mention requested budget: {reason}"
            );
            assert!(
                reason.contains("Increase max_memory_mb"),
                "error should suggest the fix: {reason}"
            );
        }
        other => panic!("expected ConfigError, got {other}"),
    }
}

/// A cache reduction must be *reported*, not silent. This is the signal whose
/// absence made STATE3's 143 s/batch a bisection exercise.
#[test]
fn budget_shrink_records_a_cache_sizing_verdict() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_dense_fixture(&dir.path().join("f.scx"), 1024, 4096, 8, 2048);
    let config = LoaderConfig {
        max_memory_mb: 96,
        ..Default::default()
    };
    let loader = IndexPlanLoader::new(
        &path, config, /*cache_shards*/ 16, /*sort_by_shard*/ true, /*lookahead*/ 4,
        /*max_plan_size*/ 1024,
    )
    .unwrap();

    let v = loader
        .cache_sizing()
        .expect("a cache_shards reduction must produce a verdict");
    assert_eq!(v.requested_cache_shards, 16);
    assert_eq!(v.effective_cache_shards, loader.effective_cache_shards());
    assert!(v.effective_cache_shards < 16);
    assert_eq!(v.budget_mb, 96);
    // The suggested budget must actually hold the request — including the
    // non-cache terms, or following the advice still would not fit.
    assert!(
        v.budget_mb_for_requested > 96,
        "suggested budget {} must exceed the one that failed",
        v.budget_mb_for_requested
    );
}

/// The anti-tautology partner: a generous budget must stay silent. Without this
/// a verdict that always fired would pass the test above.
#[test]
fn budget_generous_records_no_cache_sizing_verdict() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 32, 8, 4);
    let config = LoaderConfig {
        max_memory_mb: 4096,
        ..Default::default()
    };
    let loader = IndexPlanLoader::new(
        &path, config, /*cache_shards*/ 8, /*sort_by_shard*/ true, /*lookahead*/ 4,
        /*max_plan_size*/ 1024,
    )
    .unwrap();
    assert!(
        loader.cache_sizing().is_none(),
        "the requested cache survived; there is nothing to warn about"
    );
}

/// `auto_memory_budget` must prevent the shrink that the fixed 512 MB default
/// causes on the same file — the defect that manufactured the thrash regime.
#[test]
fn auto_memory_budget_preserves_the_requested_cache() {
    let dir = tempfile::tempdir().unwrap();
    // Keep the fixture cheap and make the *cache* the deciding term by
    // requesting many shards rather than writing enormous ones: 64 rows/shard ×
    // 512 nnz/row ⇒ shard_decoded ≈ 257 KB, so 2048 cache shards ≈ 538 MB while
    // the batch buffer is only 2 × 1024 × 512 × 4 = 4 MB. Total ≈ 592 MB, which
    // straddles the historical 512 MB default — the shape of STATE3's file,
    // where the cache, not the batch buffer, was what did not fit.
    let path = write_dense_fixture(&dir.path().join("f.scx"), 256, 512, 4, 512);
    const CACHE_SHARDS: usize = 2048;
    const MAX_PLAN: usize = 1024;

    // Arm A: the historical hard default.
    let fixed = LoaderConfig {
        max_memory_mb: 512,
        ..Default::default()
    };
    let fixed_loader = IndexPlanLoader::new(&path, fixed, CACHE_SHARDS, true, 4, MAX_PLAN).unwrap();

    // Arm B: same file, same request, adaptive budget.
    let auto = LoaderConfig {
        max_memory_mb: 512, // now the *floor*, not the ceiling
        auto_memory_budget: true,
        ..Default::default()
    };
    let auto_loader = IndexPlanLoader::new(&path, auto, CACHE_SHARDS, true, 4, MAX_PLAN).unwrap();

    assert!(
        fixed_loader.effective_cache_shards() < CACHE_SHARDS,
        "premise: the fixed 512 MB default must shrink this file's cache \
         (got {}) — otherwise this test proves nothing",
        fixed_loader.effective_cache_shards()
    );
    assert_eq!(
        auto_loader.effective_cache_shards(),
        CACHE_SHARDS,
        "the adaptive budget must honour the requested cache"
    );
    assert!(auto_loader.cache_sizing().is_none());
    assert!(
        auto_loader.max_memory_mb() > 512,
        "the resolved budget must be reported, not the floor: got {}",
        auto_loader.max_memory_mb()
    );
}

/// The adaptive budget is clamped, not unbounded: a request the cap cannot hold
/// still gets tuned down — but *silently*, because the cap is the point.
#[test]
fn auto_memory_budget_is_capped_not_unbounded() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_dense_fixture(&dir.path().join("f.scx"), 256, 512, 4, 512);
    let config = LoaderConfig {
        auto_memory_budget: true,
        ..Default::default()
    };
    // 65536 × ~257 KB ≈ 17 GB, far past ADAPTIVE_BUDGET_CAP_MB.
    let loader = IndexPlanLoader::new(&path, config, 65536, true, 4, 1024).unwrap();
    assert_eq!(
        loader.max_memory_mb(),
        crate::pipeline::ADAPTIVE_BUDGET_CAP_MB,
        "adaptive budget must clamp to the cap"
    );
    assert!(
        loader.effective_cache_shards() < 65536,
        "the cap must still tune the cache down"
    );
    // Deliberately silent: the reduction is the cap doing its job, and the
    // surviving cache is far above MIN_CACHE_SHARDS. Warning here would fire on
    // healthy default configurations — a real preflight on 194 MB shards showed
    // exactly that (128 → 21 affordable, 0.7 GB observed RSS).
    assert!(
        loader.cache_sizing().is_none(),
        "an adaptive cap's reduction must not warn while above the floor; got {:?}",
        loader.cache_sizing()
    );
}

// The remaining branch — an *adaptive* budget landing below MIN_CACHE_SHARDS —
// is covered at the predicate level by
// `budget::tests::sizing_reports_below_floor_even_under_an_adaptive_budget`, not
// here: reaching it through a real file needs an average shard above
// `ADAPTIVE_BUDGET_CAP_MB / MIN_CACHE_SHARDS` = 512 MB, which is impractical as a
// unit fixture (and rare in the wild). The reachable below-floor path is an
// explicit tiny budget, covered by
// `sparse_cellset_tests::very_tight_budget_flags_below_floor`.

/// `plan_shard_touch_count` must count distinct shards with no I/O, and must
/// dedup both within and across the pert/ctrl sides of the plan.
#[test]
fn plan_shard_touch_count_counts_distinct_shards() {
    let dir = tempfile::tempdir().unwrap();
    // 32 rows over 4 shards ⇒ 8 rows per shard: shard 0 = rows 0..8, etc.
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 32, 8, 4);
    let loader = open_loader(&path, /*sort_by_shard*/ true);

    // Both rows in shard 0.
    assert_eq!(loader.plan_shard_touch_count(&[(0, 7)]), 1);
    // Rows in shards 0 and 3.
    assert_eq!(loader.plan_shard_touch_count(&[(0, 31)]), 2);
    // Duplicates across pairs collapse to the same two shards.
    assert_eq!(
        loader.plan_shard_touch_count(&[(0, 31), (1, 30), (2, 29)]),
        2
    );
    // One row per shard ⇒ every shard.
    assert_eq!(
        loader.plan_shard_touch_count(&[(0, 8), (16, 24)]),
        4,
        "a plan spanning all four shards must report 4"
    );
    // Empty plan touches nothing.
    assert_eq!(loader.plan_shard_touch_count(&[]), 0);
}

/// Caller explicitly chooses lookahead=0 — honored when it fits the
/// budget (no prefetch path activated).
#[test]
fn budget_lookahead_zero_honored_when_fits() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 32, 8, 4);
    let config = LoaderConfig {
        max_memory_mb: 256,
        ..Default::default()
    };
    let loader = IndexPlanLoader::new(
        &path, config, /*cache_shards*/ 4, /*sort_by_shard*/ true, /*lookahead*/ 0,
        /*max_plan_size*/ 1024,
    )
    .unwrap();
    assert_eq!(loader.effective_lookahead(), 0);
}

// ---- ORG-9.10-5 pre-refactor pins ------------------------------------

/// The **model** is monotone: shrinking a knob may not raise the estimate.
///
/// Read under a budget generous enough that no tuning happens, so
/// `budget_breakdown()` is `estimate(requested)` verbatim — the only way to
/// observe the model itself before `ORG-9.10-5` extracts it behind
/// `BudgetModel::estimate`. This is the property the shared `tune()` driver
/// will `debug_assert` on every reduction step, and the one §9.6's multimodal
/// guard has always rested on without asserting.
///
/// Note what this does *not* follow from: "a tighter budget never raises an
/// effective knob" holds even for a non-monotone model, because the descent
/// simply runs longer. Checking the descent's output cannot catch this.
#[test]
fn the_budget_model_is_monotone_in_every_tuned_knob() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_dense_fixture(&dir.path().join("f.scx"), 1024, 4096, 8, 2048);

    // 4096 MB is far above the ~115 MB this fixture needs at the requested
    // config, so nothing is tuned and the breakdown is the raw estimate.
    let estimate = |cache: usize, lookahead: usize| {
        let loader = IndexPlanLoader::new(
            &path,
            LoaderConfig {
                max_memory_mb: 4096,
                ..Default::default()
            },
            cache,
            /*sort_by_shard*/ true,
            lookahead,
            /*max_plan_size*/ 1024,
        )
        .unwrap();
        assert_eq!(
            (
                loader.effective_cache_shards(),
                loader.effective_lookahead()
            ),
            (cache, lookahead),
            "premise: the budget must be generous enough that nothing is tuned"
        );
        loader.budget_breakdown().total_bytes
    };

    for cache in (1..16usize).rev() {
        assert!(
            estimate(cache, 4) <= estimate(cache + 1, 4),
            "cache_shards {cache} estimates more than {}",
            cache + 1
        );
    }
    for la in (1..8usize).rev() {
        assert!(
            estimate(8, la) <= estimate(8, la + 1),
            "lookahead {la} estimates more than {}",
            la + 1
        );
    }
    assert!(
        estimate(1, 1) <= estimate(16, 8),
        "the joint floor estimates more than the requested config"
    );
}

/// Tightening the budget may never *raise* an effective knob — a property of
/// the **descent**, not of the model (see
/// `the_budget_model_is_monotone_in_every_tuned_knob` for why the two are
/// independent). Pinned because `ORG-9.10-5`'s shared driver replaces the
/// hand-written loop that produces it.
#[test]
fn a_tighter_budget_never_raises_an_effective_knob() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_dense_fixture(&dir.path().join("f.scx"), 1024, 4096, 8, 2048);

    let tune = |mb: usize| {
        IndexPlanLoader::new(
            &path,
            LoaderConfig {
                max_memory_mb: mb,
                ..Default::default()
            },
            /*cache_shards*/ 16,
            /*sort_by_shard*/ true,
            /*lookahead*/ 4,
            /*max_plan_size*/ 1024,
        )
        .ok()
        .map(|l| (l.effective_lookahead(), l.effective_cache_shards()))
    };

    let mut prev: Option<(usize, usize)> = None;
    let mut saw_reduction = false;
    // Descending, so each step is "the budget got tighter".
    for mb in (88..=256).rev().step_by(4) {
        let Some(cur) = tune(mb) else { continue };
        if let Some(p) = prev {
            assert!(
                cur.0 <= p.0 && cur.1 <= p.1,
                "budget {mb} MB tuned to (lookahead={}, cache={}) which exceeds the \
                 looser budget's (lookahead={}, cache={})",
                cur.0,
                cur.1,
                p.0,
                p.1
            );
        }
        if cur.0 < 4 || cur.1 < 16 {
            saw_reduction = true;
        }
        prev = Some(cur);
    }
    assert!(
        saw_reduction,
        "no budget in the sweep engaged the auto-tune, so this proves nothing"
    );
}

/// A loader that constructed must fit the breakdown it reports.
///
/// `BudgetBreakdown::fits_within` exists and no loop uses it as its
/// terminator. This is the Rust-side twin of
/// `test_index_plan_dataset.py::test_memory_budget_matches_max_memory_mb`, and
/// the shape `SparseCellSetLoader` does *not* satisfy today (ORG-9.10-5).
#[test]
fn a_constructed_loader_fits_the_breakdown_it_reports() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_dense_fixture(&dir.path().join("f.scx"), 1024, 4096, 8, 2048);
    let mut saw_tuned = false;
    for mb in (88..=256).step_by(4) {
        let Ok(loader) = IndexPlanLoader::new(
            &path,
            LoaderConfig {
                max_memory_mb: mb,
                ..Default::default()
            },
            /*cache_shards*/ 16,
            /*sort_by_shard*/ true,
            /*lookahead*/ 4,
            /*max_plan_size*/ 1024,
        ) else {
            continue;
        };
        assert!(
            loader
                .budget_breakdown()
                .fits_within(loader.max_memory_mb()),
            "budget {mb} MB constructed but its breakdown totals {} bytes",
            loader.budget_breakdown().total_bytes
        );
        if loader.effective_lookahead() < 4 || loader.effective_cache_shards() < 16 {
            saw_tuned = true;
        }
    }
    assert!(
        saw_tuned,
        "no budget in the sweep engaged the auto-tune, so this proves nothing"
    );
}

/// `process_plan` must reject plans larger than `max_plan_size` so a
/// misbehaving consumer cannot silently exceed the memory budget.
/// Plans at or below the ceiling are accepted as before.
#[test]
fn process_plan_rejects_oversize_plan() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 32, 8, 4);
    let config = LoaderConfig {
        normalize: false,
        log1p: false,
        obs_columns: vec!["cell_id".to_string()],
        ..Default::default()
    };

    let loader = IndexPlanLoader::new(
        &path, config, /*cache_shards*/ 4, /*sort_by_shard*/ false, /*lookahead*/ 1,
        /*max_plan_size*/ 4,
    )
    .unwrap();

    // At-the-ceiling plan succeeds.
    let ok_plan = vec![(0u64, 1u64), (2, 3), (4, 5), (6, 7)];
    let batch = loader.process_plan(ok_plan).unwrap();
    assert_eq!(batch.pairs.len(), 4);

    // Over-the-ceiling plan rejects with a ConfigError naming both numbers.
    let big_plan: Vec<(u64, u64)> = (0..5u64).map(|i| (i, (i + 1) % 32)).collect();
    match loader.process_plan(big_plan) {
        Err(LoaderError::ConfigError { reason }) => {
            assert!(reason.contains("plan size 5"), "got: {reason}");
            assert!(reason.contains("max_plan_size 4"), "got: {reason}");
        }
        Err(other) => panic!("expected ConfigError, got {other}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

// ─── Lazy runtime / fork safety (Patch 11 § P1 #16) ──────────────
//
// The loader defers building its `PrefetchEngine` — and with it the tokio
// runtime — to the first `iter_with_plans` call. Constructing a loader must
// not build a runtime: it would otherwise be inherited by `DataLoader` worker
// processes after fork, which is §9.1.
//
// Since ORG-9.10-1 the runtime belongs to the engine, so the probe is
// `PrefetchEngine::runtime_is_built()` rather than a field of this module.
// `lazy_runtime_idempotent_under_concurrent_init` moved with it, as
// `plan_engine_tests::engine_runtime_idempotent_under_concurrent_init` — after
// the fold there is exactly one `OnceLock` and one test for it.

/// Whether the loader has materialised its engine *and* that engine's runtime.
/// Two states are distinct and both matter: no engine at all (what `new` must
/// leave behind) and an engine whose runtime is still unbuilt.
fn runtime_is_built(loader: &IndexPlanLoader) -> bool {
    loader
        .engine
        .get()
        .is_some_and(|engine| engine.runtime_is_built())
}

#[test]
fn lazy_runtime_not_built_at_construction() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
    let loader = open_loader(&path, false);
    assert!(
        loader.engine.get().is_none(),
        "IndexPlanLoader::new must not eagerly build the prefetch engine — \
             its tokio runtime would defeat the DataLoader fork-safety contract"
    );
    assert!(!runtime_is_built(&loader));
}

#[test]
fn lazy_runtime_built_after_iter_with_plans_consumed() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
    let loader = Arc::new(open_loader(&path, false));

    // Sanity-check the precondition.
    assert!(!runtime_is_built(&loader));

    let plans = vec![Ok(vec![(0u64, 1u64), (2, 3)])].into_iter();
    let iter = Arc::clone(&loader).iter_with_plans(plans, /*lookahead*/ 2);
    // Consume one batch — this drives `refill` → `spawn_prefetches`
    // → the engine's lazy `runtime()` accessor.
    let mut iter = iter;
    let _ = iter.next();
    assert!(
        runtime_is_built(&loader),
        "first iter consumption should materialize the runtime via OnceLock"
    );
}

// ---------------------------------------------------------------------
// §9.4 — prefetch depth is opportunistic, never an obligation on the plan
// generator.
//
// `refill` used to loop on a *blocking* `recv` until `in_flight` reached
// `lookahead.max(1)` (default 4). A generator that produces plan i+1 only
// after seeing batch i — curriculum / feedback sampling, a normal shape for
// perturbation training — deadlocked permanently: the generator waited for
// batch i, the consumer waited for plan i+4, and nothing timed either out.
// ---------------------------------------------------------------------

/// Run `f` on a worker thread; panic rather than hang the suite if it has not
/// finished within `secs`. Without this, the red for a `refill` deadlock is a
/// `cargo test` run that never returns and has to be killed by hand — which
/// reads as infrastructure trouble rather than as a failing test.
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
    plans: std::vec::IntoIter<Vec<(u64, u64)>>,
    ack: Receiver<()>,
    first: bool,
}

impl Iterator for FeedbackPlans {
    type Item = std::result::Result<Vec<(u64, u64)>, LoaderError>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.first {
            // Park until the consumer has seen the previous batch. A sender
            // that has gone away means the consumer stopped early.
            self.ack.recv().ok()?;
        }
        self.first = false;
        self.plans.next().map(Ok)
    }
}

/// The §9.4 red: a feedback generator must still see every batch.
#[test]
fn feedback_generator_yields_every_batch() {
    let plans = vec![
        vec![(0u64, 1u64)],
        vec![(2u64, 3u64)],
        vec![(4u64, 5u64)],
        vec![(6u64, 7u64)],
    ];
    let want = plans.len();

    let got = with_deadline(30, "feedback plan generator", move || {
        let dir = tempfile::tempdir().unwrap();
        let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
        let loader = open_loader_arc(&path);

        // Rendezvous: the ack is delivered only once the generator asks for it,
        // so the two sides really are lock-stepped.
        let (ack_tx, ack_rx) = bounded(0);
        let it = loader.iter_with_plans(
            FeedbackPlans {
                plans: plans.into_iter(),
                ack: ack_rx,
                first: true,
            },
            /*lookahead*/ 4,
        );

        let mut count = 0usize;
        for batch in it {
            batch.expect("batch must decode");
            count += 1;
            // What a feedback sampler does after scoring the batch.
            let _ = ack_tx.send(());
        }
        count
    });

    assert_eq!(
        got, want,
        "every plan must produce a batch when the generator waits on the previous one"
    );
}

/// A generator that is merely slow must not end the epoch.
///
/// This is the red for the *wrong* fix rather than for the old code: swapping
/// the blocking `recv` for a `try_recv` whose `Empty` arm sets
/// `plan_stream_done` truncates the epoch to a single batch, silently and with
/// no error anywhere.
#[test]
fn a_slow_generator_does_not_end_the_epoch() {
    let plans: Vec<Vec<(u64, u64)>> = (0..5).map(|i| vec![(i * 2, i * 2 + 1)]).collect();
    let want = plans.len();

    let got = with_deadline(30, "slow plan generator", move || {
        let dir = tempfile::tempdir().unwrap();
        let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
        let loader = open_loader_arc(&path);

        let slow = plans.into_iter().map(|p| {
            std::thread::sleep(std::time::Duration::from_millis(20));
            Ok(p)
        });
        let mut count = 0usize;
        for batch in loader.iter_with_plans(slow, /*lookahead*/ 4) {
            batch.expect("batch must decode");
            count += 1;
        }
        count
    });

    assert_eq!(got, want, "a slow generator must not truncate the epoch");
}

// The over-fix guard for §9.4 — "when the generator keeps up, the queue still
// fills to `lookahead`" — is now
// `plan_engine_tests::engine_prefetch_depth_survives_when_the_generator_keeps_up`.
// It reads `PlanPrefetchIter::in_flight`, and since ORG-9.10-1 that queue is
// the only one: `IndexPlanIter` is an adapter with no queue of its own, so a
// copy here could only re-assert what the engine's test already asserts about
// the same code. Deleted rather than weakened to a proxy.

// ---------------------------------------------------------------------
// §9.3 — bounded, GIL-free teardown.
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// §9.3 — the teardown deadline holds regardless of who releases the last
// reference.
// ---------------------------------------------------------------------

/// A prefetch task must not capture the loader.
///
/// It used to: `spawn_blocking(move || loader.backed.read_shard_cached_arc(..))`.
/// An already-started blocking task cannot be aborted, so such a task could
/// outlive the iterator and release the *final* loader reference — dropping the
/// tokio runtime from one of that runtime's own threads. Capturing only the
/// `Arc<BackedCsrReader>` means no task can ever be the last owner.
///
/// The gate is load-bearing, and so is its *start signal*. A first version
/// counted references after `next()` returned and **passed with the loader
/// captured again** — the tasks had already finished. A second version parked
/// them but inferred "a task is running" from a 300 ms sleep, which fails the
/// same way on a loaded machine: no task started, nothing held, count clean.
/// The gate now announces entry before parking, and the test fails outright if
/// no task announces.
///
/// Since ORG-9.10-1 the spawning code lives in `PlanPrefetchIter` and has no
/// path to the loader at all, so the loader count alone can no longer go wrong
/// — the reachable owner is now the `Arc<PrefetchEngine>`, and a task holding
/// *that* reproduces the identical hazard one level down. Both counts are
/// therefore asserted, and the engine count is the one with teeth.
#[test]
fn a_prefetch_task_does_not_capture_the_loader() {
    let dir = tempfile::tempdir().unwrap();
    // Unframed: a framed file routes the scattered gather through the block
    // index, and `spawn_prefetches` then skips warming — no task, nothing to
    // observe.
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 32, 8, 4);

    let config = LoaderConfig {
        normalize: false,
        log1p: false,
        obs_columns: vec!["cell_id".to_string()],
        ..Default::default()
    };
    let mut raw = IndexPlanLoader::new(&path, config, 4, true, 4, 16384).unwrap();
    // Order matters: `set_scatter_block_index` reaches the reader through
    // `Arc::get_mut`, and `engine()` clones that `Arc` into the engine. Install
    // the gate on the engine itself — there is exactly one `OnceLock` for it,
    // rather than a loader-side stash copied across at engine construction.
    raw.set_scatter_block_index(false);
    let gate = Arc::new(crate::plan_engine::PrefetchGate::new());
    raw.engine().set_prefetch_gate(Arc::clone(&gate));
    let loader = Arc::new(raw);

    let mut it = Arc::clone(&loader).iter_with_plans(
        into_plan_iter(vec![
            vec![(0u64, 1u64)],
            vec![(8, 12)],
            vec![(16, 20)],
            vec![(24, 28)],
        ]),
        4,
    );
    assert_eq!(
        Arc::strong_count(&loader),
        2,
        "unexpected extra owner at start"
    );

    // Spawn the queue's prefetches on a background thread: the head plan's are
    // awaited, and the gate is holding them, so `next()` would block here.
    let handle = std::thread::spawn(move || {
        let b = it.next();
        (it, b)
    });

    // Wait for a task to actually announce itself. A sleep here would let the
    // test pass on a loaded machine with the bug present: no task started, so
    // no task is holding anything, so the count looks clean.
    let entered = gate.wait_for_entry(std::time::Duration::from_secs(30));
    let held = Arc::strong_count(&loader);
    // Two owners: the loader's own `OnceLock` and the live iterator. A third
    // means a `spawn_blocking` task captured it.
    let held_engine = loader.engine.get().map(Arc::strong_count).unwrap_or(0);
    gate.release();

    let (it, batch) = handle.join().expect("worker must not panic");
    batch.expect("a batch").expect("must decode");
    drop(it);

    assert!(
        entered,
        "no prefetch task started within 30s — the premise never held, so the \
         ownership assertion below would have been vacuous"
    );
    assert_eq!(
        held, 2,
        "a prefetch task in flight is holding an Arc<IndexPlanLoader> \
         (strong_count={held}); such a task can outlive the iterator and drop \
         the runtime from a runtime thread"
    );
    assert_eq!(
        held_engine, 2,
        "a prefetch task in flight is holding an Arc<PrefetchEngine> \
         (strong_count={held_engine}); the engine owns the runtime, so an \
         unabortable blocking task that holds it can release the last reference \
         from one of that runtime's own threads — see `crate::runtime`"
    );
}

/// Whoever releases the last reference gets the bounded teardown — including
/// the iterator, which is the last owner after the advertised
/// `ds.close(); list(it)` ordering.
#[test]
fn the_iterator_as_last_owner_still_gets_a_bounded_teardown() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);

    let before = crate::runtime::BOUNDED_SHUTDOWNS.with(|c| c.get());

    let loader = open_loader_arc(&path);
    let mut it = Arc::clone(&loader)
        .iter_with_plans(into_plan_iter(vec![vec![(0u64, 1u64)], vec![(4, 9)]]), 4);
    it.next().expect("first batch").expect("must decode"); // forces the runtime

    drop(loader); // the `ds.close()` half — the iter is now the only owner
    for b in it.by_ref() {
        b.expect("must decode");
    }
    drop(it);

    assert_eq!(
        crate::runtime::BOUNDED_SHUTDOWNS.with(|c| c.get()),
        before + 1,
        "the iterator released the last reference, so the runtime must have \
         gone down through BoundedRuntime::drop"
    );
}

// ---------------------------------------------------------------------
// `prefetch_skipped_in_flight` — the one `IterMetrics` counter with no pin
// ---------------------------------------------------------------------

/// **Pre-refactor pin (ORG-9.10-1, drift (b)).** A shard that a *peer* is
/// already decoding must be skipped by the prefetch and attributed to
/// `prefetch_skipped_in_flight` — not spawned again, and not charged to one of
/// the other three counters.
///
/// This is the counter 9a left unpinned, and the reason is structural: the
/// in-flight table is published by the singleflight leader inside
/// `SharedShardCache::get_or_decode` and torn down the instant the decode
/// returns, so from `scx-loader` there was no way to observe the window without
/// racing a sleep against a decode. `test_index_plan.rs`'s conservation law
/// covers the *sum*, so a bug that mis-attributed an in-flight skip as a
/// cache-hit skip would leave every existing test green.
///
/// The premise is established by a predicate, never by timing:
/// `scx-format-io`'s `test-hooks` `set_decode_barrier` parks the leader at the
/// top of `decode_shard` — after the in-flight publish, before the decode — and
/// the test waits on `in_flight_contains`, failing loudly if it never becomes
/// true rather than asserting against an empty table.
///
/// `next()` runs on a worker because it cannot complete while the barrier is
/// held: after the prefetch skip, `process_plan`'s own gather joins the
/// singleflight as a *follower* and blocks on the same leader.
#[test]
fn prefetch_skips_and_counts_a_shard_a_peer_is_already_decoding() {
    const HELD_SHARD: usize = 2;

    let dir = tempfile::tempdir().unwrap();
    // 32 rows / 4 shards → 8 rows each; shard 2 is rows 16..24.
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 32, 8, 4);

    let config = LoaderConfig {
        normalize: false,
        log1p: false,
        obs_columns: vec!["cell_id".to_string()],
        ..Default::default()
    };
    let mut raw = IndexPlanLoader::new(&path, config, 8, true, 4, 16384).unwrap();
    // Unframed fixture, but pin the gate off anyway so a future framed default
    // cannot turn this into a block-index skip and quietly retarget the test.
    raw.set_scatter_block_index(false);

    let release = StdArc::new(std::sync::atomic::AtomicBool::new(false));
    let entered = StdArc::new(AtomicU64::new(0));
    {
        let release = StdArc::clone(&release);
        let entered = StdArc::clone(&entered);
        Arc::get_mut(&mut raw.backed)
            .expect("reader is unshared before the loader is wrapped")
            .set_decode_barrier(Arc::new(move |sidx: usize| {
                if sidx != HELD_SHARD {
                    return;
                }
                entered.fetch_add(1, Ordering::AcqRel);
                while !release.load(Ordering::Acquire) {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }));
    }
    let loader = Arc::new(raw);

    // The peer: becomes singleflight leader for HELD_SHARD and parks.
    let peer_reader = Arc::clone(&loader.backed);
    let peer = std::thread::spawn(move || peer_reader.read_shard_cached_arc(HELD_SHARD));

    // Premise, by predicate: the shard really is in the in-flight table.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while !loader.backed.in_flight_contains(HELD_SHARD) {
        assert!(
            std::time::Instant::now() < deadline,
            "the peer never published shard {HELD_SHARD} in the in-flight table \
             ({} barrier entries) — the premise failed, so the assertion below \
             would have been vacuous",
            entered.load(Ordering::Acquire)
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    // Both rows live in HELD_SHARD, so it is the only shard the plan touches.
    let mut it = Arc::clone(&loader).iter_with_plans(into_plan_iter(vec![vec![(16u64, 17u64)]]), 1);
    let metrics = it.iter_metrics();
    let consumer = std::thread::spawn(move || {
        let b = it.next();
        (it, b)
    });

    // `spawn_prefetches` has run once the counter moves; releasing before that
    // would let the leader finish and reopen the race this test exists to close.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while metrics.prefetch_skipped_in_flight.load(Ordering::Relaxed) == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "prefetch_skipped_in_flight never incremented for a shard that was \
             demonstrably in flight: spawned={} cache_hit={} block_index={}",
            metrics.prefetch_tasks_spawned.load(Ordering::Relaxed),
            metrics.prefetch_skipped_cache_hit.load(Ordering::Relaxed),
            metrics.prefetch_skipped_block_index.load(Ordering::Relaxed),
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    assert_eq!(
        metrics.prefetch_tasks_spawned.load(Ordering::Relaxed),
        0,
        "the only shard the plan touches was already in flight, so nothing \
         should have been spawned"
    );
    assert_eq!(
        metrics.prefetch_skipped_cache_hit.load(Ordering::Relaxed),
        0,
        "the shard was in flight, not cached — mis-attributing the skip is \
         exactly what the conservation law cannot see"
    );

    release.store(true, Ordering::Release);
    peer.join().expect("peer thread").expect("peer decode");
    let (it, batch) = consumer.join().expect("consumer thread");
    batch.expect("a batch").expect("must decode");
    drop(it);

    assert_eq!(
        metrics.prefetch_skipped_in_flight.load(Ordering::Relaxed),
        1,
        "exactly one shard was skipped as in-flight"
    );
}

/// **Dropping the iterator must stop the plan generator even when every plan is
/// empty.**
///
/// The empty-plan skip is this loader's contract, not the engine's, so the fold
/// first expressed it as `plans.filter(..)` on the stream handed to the engine.
/// That put the filter *upstream* of the only place cancellation is observed:
///
/// ```ignore
/// for item in plans {
///     if plan_tx.send(item).is_err() { break; }   // the sole cancellation point
/// }
/// ```
///
/// `Filter::next` discards non-matching items internally and never returns to
/// that loop body, so a stream of empty plans never reaches `send` and never
/// learns the receiver is gone. An endless empty stream spins the detached
/// worker forever — on the Python path, calling `PyPlanIterator` long after the
/// batch iterator was dropped — and a long finite empty prefix is pulled with no
/// backpressure at all, since it never occupies a channel slot.
///
/// Skipping inside `next` instead keeps every plan going through the bounded
/// channel, which is what the pre-fold loop did.
///
/// # Both halves are rendezvous, not sleeps
///
/// An earlier version of this test dropped immediately and treated "the pull
/// count did not move for 50 ms" as proof the worker had stopped. That reading
/// is also what a worker that has *not started yet* produces, so on a slow
/// scheduler the test would declare success, set `stop`, and let a late-starting
/// broken worker exit quietly — passing on exactly the bug it exists to catch.
///
/// So: wait for `pulled >= 1` before dropping, which proves the worker really is
/// pulling; and after the drop wait on `exited`, set by the producer's own
/// `Drop`. That fires only when the worker's `for item in plans` loop ends, i.e.
/// when `send` observed the disconnect — a real completion signal rather than an
/// inference from silence.
#[test]
fn dropping_the_iter_stops_an_endless_empty_plan_generator() {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};

    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
    let loader = open_loader_arc(&path);

    struct EndlessEmptyPlans {
        pulled: Arc<AtomicU64>,
        stop: Arc<AtomicBool>,
        exited: Arc<AtomicBool>,
    }
    impl Iterator for EndlessEmptyPlans {
        type Item = std::result::Result<Vec<(u64, u64)>, LoaderError>;
        fn next(&mut self) -> Option<Self::Item> {
            // The escape hatch exists so a *failing* run cleans up instead of
            // leaking a hot spin loop for the rest of the suite.
            if self.stop.load(AtomicOrdering::Acquire) {
                return None;
            }
            self.pulled.fetch_add(1, AtomicOrdering::AcqRel);
            Some(Ok(Vec::new()))
        }
    }
    impl Drop for EndlessEmptyPlans {
        fn drop(&mut self) {
            self.exited.store(true, AtomicOrdering::Release);
        }
    }

    let pulled = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let exited = Arc::new(AtomicBool::new(false));
    let it = loader.iter_with_plans(
        EndlessEmptyPlans {
            pulled: Arc::clone(&pulled),
            stop: Arc::clone(&stop),
            exited: Arc::clone(&exited),
        },
        /*lookahead*/ 4,
    );

    // Premise: the worker is actually pulling. Without this the drop below can
    // race ahead of the thread ever starting, and "quiet" would prove nothing.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while pulled.load(AtomicOrdering::Acquire) == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "the pull worker never pulled a single plan — premise failed, so the \
             assertion below would have been vacuous"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    // Deliberately not consumed: with every plan empty there is no batch to
    // wait for, and `next` would block. The worker is already running.
    drop(it);

    // Conclusion: the producer was dropped, which happens only when the worker's
    // `for item in plans` loop ended — i.e. `send` saw the disconnect.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut released = false;
    while std::time::Instant::now() < deadline {
        if exited.load(AtomicOrdering::Acquire) {
            released = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    // Set only after the verdict is decided, so a broken worker cannot be
    // rescued by the escape hatch before we have looked.
    stop.store(true, AtomicOrdering::Release);

    assert!(
        released,
        "10s after the iterator was dropped the plan generator had still not \
         been released ({} pulls and counting) — dropping `plan_rx` cannot \
         cancel a worker that never reaches `send`",
        pulled.load(AtomicOrdering::Acquire)
    );
}

/// Empty plans must not reorder or swallow a plan-stream error.
///
/// `iter_with_plans_propagates_plan_errors` puts non-empty plans on both sides
/// of the error, so it never exercises the interaction the empty-plan skip
/// actually introduces: `IndexPlanIter::next` now `continue`s past zero-row
/// batches, and the engine latches a plan-stream error and surfaces it one-shot
/// *after* the already-queued plans drain. Skipping is only safe if `Err` never
/// matches the `continue` arm and the latch is not consumed by the loop.
///
/// Found while checking round 4's own claim rather than reported by a reviewer;
/// the property held on inspection, but nothing pinned it.
#[test]
fn empty_plans_before_an_error_do_not_reorder_or_swallow_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 8, 4, 2);
    let loader = open_loader_arc(&path);

    // Empty plans immediately either side of the error, and a trailing
    // non-empty plan that must never be reached (the error is sticky).
    let plans: Vec<std::result::Result<Vec<(u64, u64)>, LoaderError>> = vec![
        Ok(vec![]),
        Ok(vec![(0, 1)]),
        Ok(vec![]),
        Ok(vec![]),
        Err(LoaderError::ChannelError("synthetic".into())),
        Ok(vec![(2, 3)]),
    ];
    let mut it = loader.iter_with_plans(plans.into_iter(), 2);

    // The leading empty is skipped, so the first item is the real batch.
    let first = it.next().expect("a first item").expect("must decode");
    assert_eq!(
        first.pairs,
        vec![(0, 1)],
        "leading empty plan must be skipped"
    );

    // Two more empties are skipped *within* this call, and the latched error
    // surfaces next rather than being consumed by the skip loop.
    match it.next().expect("a second item") {
        Err(LoaderError::ChannelError(s)) => assert!(s.contains("synthetic")),
        Err(other) => panic!("expected the ChannelError, got {other}"),
        Ok(b) => panic!(
            "empty plans between the batch and the error were not skipped, or \
             the error was reordered behind one: got a batch with {} pairs",
            b.pairs.len()
        ),
    }

    assert!(
        it.next().is_none(),
        "the plan-stream error is sticky — nothing after it may be yielded"
    );
}
