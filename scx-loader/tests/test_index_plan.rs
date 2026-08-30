//! Integration tests for `scx_loader::IndexPlanLoader`.
//!
//! Covers:
//! - `process_plan` shape correctness (HVG on/off).
//! - Normalize + log1p semantics.
//! - Empty plan + out-of-range index error paths.
//! - sort_by_shard: pairs/rows alignment + same-plan-different-order parity.
//! - HVG-projected paired scatter end-to-end via process_plan.

mod common;

use common::{
    open_loader, open_loader_hvg, open_loader_normalized, open_loader_with_flags,
    write_multi_shard_fixture,
};
use scx_loader::LoaderError;

const N_OBS: usize = 100;
const N_VARS: usize = 50;
const N_SHARDS: usize = 5;

fn fixture(dir: &tempfile::TempDir) -> std::path::PathBuf {
    write_multi_shard_fixture(&dir.path().join("fixture.scx"), N_OBS, N_VARS, N_SHARDS)
}

/// Spec: "process_plan on a 100-cell test fixture, plan [(0,1),(2,3),(4,5)],
/// output shapes [3, n_cols] and [3, n_cols]."
#[test]
fn process_plan_basic_shape() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);
    let loader = open_loader(&path, /*sort_by_shard=*/ false);

    let plan = vec![(0u64, 1u64), (2, 3), (4, 5)];
    let batch = loader.process_plan(plan.clone()).unwrap();

    assert_eq!(batch.pairs.len(), 3);
    assert_eq!(batch.x.len(), 3 * loader.n_output_cols());
    assert_eq!(batch.x_paired.len(), 3 * loader.n_output_cols());
    assert_eq!(batch.pairs, plan); // sort_by_shard off → input order preserved
    assert_eq!(loader.n_output_cols(), N_VARS);
    // obs is gathered for both sides
    assert!(batch.obs.contains_key("cell_id"));
    assert!(batch.obs_paired.contains_key("cell_id"));
}

/// Spec: "HVG projection on/off: when hvg_indices=Some([0,5,10]), output
/// cols == 3; when None, output cols == n_vars."
#[test]
fn process_plan_hvg_projection_shapes() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    let loader_none = open_loader(&path, false);
    assert_eq!(loader_none.n_output_cols(), N_VARS);
    let b_none = loader_none.process_plan(vec![(0, 1)]).unwrap();
    assert_eq!(b_none.x.len(), N_VARS);

    let loader_hvg = open_loader_hvg(&path, vec![0, 5, 10], false);
    assert_eq!(loader_hvg.n_output_cols(), 3);
    let b_hvg = loader_hvg.process_plan(vec![(0, 1)]).unwrap();
    assert_eq!(b_hvg.x.len(), 3);
    assert_eq!(b_hvg.x_paired.len(), 3);
}

/// Spec: "Normalize+log1p on/off: row sums match target_sum after normalize;
/// log1p applied if flag set."
///
/// In our fixture each row has exactly one nonzero, so post-normalize that
/// nonzero equals `target_sum` (the row sum), and post-log1p it equals
/// `ln(1 + target_sum)`.
#[test]
fn process_plan_normalize_log1p_semantics() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);
    let target_sum = 1.0e4_f64;

    let loader = open_loader_normalized(&path, /*sort_by_shard=*/ false, target_sum);
    let batch = loader.process_plan(vec![(7u64, 8u64)]).unwrap();

    // Row 7's single nonzero is at col 7%50=7 with raw value 8.
    // After normalize: 8 × (1e4 / 8) = 1e4. After log1p: ln(1 + 1e4).
    let row = &batch.x[..N_VARS];
    let expected = (1.0f64 + target_sum).ln() as f32;
    let nonzero_count = row.iter().filter(|&&v| v != 0.0).count();
    assert_eq!(nonzero_count, 1, "exactly one nonzero per row");
    let nonzero = row[7 % N_VARS];
    assert!(
        (nonzero - expected).abs() < 1e-3,
        "log1p(target_sum) ≈ {expected}, got {nonzero}"
    );
}

/// Spec: `(normalize=true, log1p=false)` — row sum equals `target_sum`,
/// no log1p applied. Regression test for the bug where `IndexPlanLoader`
/// always called `fused_normalize_log1p_dense` whenever `normalize=true`,
/// silently applying log1p that the caller did not request.
#[test]
fn process_plan_normalize_only_no_log1p() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);
    let target_sum = 1.0e4_f64;

    let loader = open_loader_with_flags(
        &path, /*normalize=*/ true, /*log1p=*/ false, target_sum,
        /*sort_by_shard=*/ false,
    );
    // Pert row 7 -> nonzero at col 7, raw value 8.
    // Ctrl row 8 -> nonzero at col 8, raw value 9.
    let batch = loader.process_plan(vec![(7u64, 8u64)]).unwrap();

    let p_row = &batch.x[..N_VARS];
    let c_row = &batch.x_paired[..N_VARS];

    // Both pert and ctrl rows should sum to target_sum (normalize only).
    let p_sum: f64 = p_row.iter().map(|&v| v as f64).sum();
    let c_sum: f64 = c_row.iter().map(|&v| v as f64).sum();
    assert!(
        (p_sum - target_sum).abs() < 1e-2,
        "pert row sum {p_sum} != target {target_sum}"
    );
    assert!(
        (c_sum - target_sum).abs() < 1e-2,
        "ctrl row sum {c_sum} != target {target_sum}"
    );

    // Single nonzero per row, equal to target_sum (no log1p).
    assert_eq!(p_row.iter().filter(|&&v| v != 0.0).count(), 1);
    assert_eq!(c_row.iter().filter(|&&v| v != 0.0).count(), 1);
    assert!((p_row[7 % N_VARS] as f64 - target_sum).abs() < 1e-2);
    assert!((c_row[8 % N_VARS] as f64 - target_sum).abs() < 1e-2);
}

/// Spec: `(normalize=false, log1p=true)` — `ln(1 + raw)` per element, no
/// scaling. Regression test for the bug where `IndexPlanLoader` skipped
/// the transform entirely when `normalize=false`, returning raw rows even
/// though the caller requested log1p (the state-scx ST default).
#[test]
fn process_plan_log1p_only_no_normalize() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    let loader = open_loader_with_flags(
        &path, /*normalize=*/ false, /*log1p=*/ true, /*target_sum=*/ 0.0,
        /*sort_by_shard=*/ false,
    );
    // Pert row 7 -> col 7, raw value 8 -> ln(9).
    // Ctrl row 8 -> col 8, raw value 9 -> ln(10).
    let batch = loader.process_plan(vec![(7u64, 8u64)]).unwrap();

    let p_row = &batch.x[..N_VARS];
    let c_row = &batch.x_paired[..N_VARS];

    let p_expected = ((8.0_f32) + 1.0).ln();
    let c_expected = ((9.0_f32) + 1.0).ln();

    assert!(
        (p_row[7 % N_VARS] - p_expected).abs() < 1e-5,
        "pert log1p mismatch: got {} expected {}",
        p_row[7 % N_VARS],
        p_expected
    );
    assert!(
        (c_row[8 % N_VARS] - c_expected).abs() < 1e-5,
        "ctrl log1p mismatch: got {} expected {}",
        c_row[8 % N_VARS],
        c_expected
    );
    // Other columns should be ln(0+1) = 0 — but we wrote zeros into the
    // dense buffer, so the log1p applies element-wise and yields exactly 0.
    for (i, &v) in p_row.iter().enumerate() {
        if i != 7 % N_VARS {
            assert!(v.abs() < 1e-7);
        }
    }
}

/// Spec: `(normalize=false, log1p=false)` — pass-through, raw values.
#[test]
fn process_plan_no_normalize_no_log1p_returns_raw() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    let loader = open_loader(&path, /*sort_by_shard=*/ false);
    let batch = loader.process_plan(vec![(7u64, 8u64)]).unwrap();

    let p_row = &batch.x[..N_VARS];
    let c_row = &batch.x_paired[..N_VARS];

    assert_eq!(p_row[7 % N_VARS], 8.0);
    assert_eq!(c_row[8 % N_VARS], 9.0);
    assert_eq!(p_row.iter().filter(|&&v| v != 0.0).count(), 1);
    assert_eq!(c_row.iter().filter(|&&v| v != 0.0).count(), 1);
}

/// Spec: "Empty plan: yields a zero-row batch."
#[test]
fn process_plan_empty_yields_zero_row_batch() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);
    let loader = open_loader(&path, false);

    let batch = loader.process_plan(Vec::new()).unwrap();
    assert!(batch.x.is_empty());
    assert!(batch.x_paired.is_empty());
    assert!(batch.pairs.is_empty());
    assert!(batch.obs.is_empty());
    assert!(batch.obs_paired.is_empty());
}

/// Spec: "Out-of-range index: returns IndexError-shaped error."
#[test]
fn process_plan_out_of_range_returns_index_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);
    let loader = open_loader(&path, false);

    // pert side OOR
    let r = loader.process_plan(vec![(N_OBS as u64 + 5, 0)]);
    match r {
        Err(LoaderError::IndexOutOfRange { idx, n_obs }) => {
            assert_eq!(idx, N_OBS as u64 + 5);
            assert_eq!(n_obs, N_OBS);
        }
        Err(other) => panic!("expected IndexOutOfRange, got {other}"),
        Ok(_) => panic!("expected error, got Ok"),
    }

    // ctrl side OOR
    let r = loader.process_plan(vec![(0, N_OBS as u64 + 1)]);
    assert!(matches!(r, Err(LoaderError::IndexOutOfRange { .. })));
}

/// Spec: "Sort-by-shard preserves pair semantics: the post-sort pairs field,
/// when iterated in returned order, equals row order in x / x_paired."
///
/// Our fixture's row `r` has a single nonzero at col `r % n_vars` with value
/// `((r+1) & 0xFF)`. We pin the alignment by checking each row's reconstructed
/// (col, value) against `pairs[i]`.
#[test]
fn sort_by_shard_pairs_align_with_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);
    let loader = open_loader(&path, /*sort_by_shard=*/ true);
    let n_cols = loader.n_output_cols();

    // Deliberately scrambled across shards.
    let plan: Vec<(u64, u64)> = vec![
        (95, 0),
        (40, 50),
        (80, 90),
        (3, 12),
        (60, 11),
        (1, 2),
        (25, 35),
    ];
    let batch = loader.process_plan(plan).unwrap();

    for i in 0..batch.pairs.len() {
        let (p, c) = batch.pairs[i];
        let p_row = &batch.x[i * n_cols..(i + 1) * n_cols];
        let c_row = &batch.x_paired[i * n_cols..(i + 1) * n_cols];

        let p_col = (p as usize) % N_VARS;
        let c_col = (c as usize) % N_VARS;
        assert_eq!(p_row[p_col], ((p as usize + 1) & 0xFF) as f32);
        assert_eq!(c_row[c_col], ((c as usize + 1) & 0xFF) as f32);
        // Single nonzero per row.
        assert_eq!(p_row.iter().filter(|&&v| v != 0.0).count(), 1);
        assert_eq!(c_row.iter().filter(|&&v| v != 0.0).count(), 1);
    }
}

/// Spec: "same-plan-different-order parity invariant (output equality up to
/// row permutation when sort_by_shard is on vs off)."
#[test]
fn sort_by_shard_is_pure_permutation() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    let plan: Vec<(u64, u64)> = vec![(95, 0), (40, 50), (80, 90), (3, 12), (60, 11), (1, 2)];

    let loader_unsorted = open_loader(&path, false);
    let loader_sorted = open_loader(&path, true);

    let b_un = loader_unsorted.process_plan(plan.clone()).unwrap();
    let b_so = loader_sorted.process_plan(plan.clone()).unwrap();

    let n_cols = loader_unsorted.n_output_cols();
    /// `(pair, row bits, paired-row bits)` — compared as a multiset.
    type Triples = Vec<((u64, u64), Vec<u32>, Vec<u32>)>;
    let triples = |b: &scx_loader::IndexPlanBatch| -> Triples {
        (0..b.pairs.len())
            .map(|i| {
                let r: Vec<u32> = b.x[i * n_cols..(i + 1) * n_cols]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                let pr: Vec<u32> = b.x_paired[i * n_cols..(i + 1) * n_cols]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                (b.pairs[i], r, pr)
            })
            .collect()
    };
    let mut un = triples(&b_un);
    let mut so = triples(&b_so);
    un.sort_by_key(|t| t.0);
    so.sort_by_key(|t| t.0);
    assert_eq!(un, so);
}

/// HVG-projected paired scatter via `process_plan` reproduces the expected
/// `(col, value)` tuple per row on both sides of the pair, exercising the
/// `read_rows_with` + `HvgProjection::scatter_row` dense-gather path.
#[test]
fn hvg_paired_scatter_via_process_plan() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);
    // Hits HVG positions {0, 1, 2} via cols {7, 17, 27} for rows 7, 17, 27.
    let hvg = vec![7u32, 17, 27];
    let loader = open_loader_hvg(&path, hvg.clone(), false);

    let plan = vec![(7u64, 17u64), (27, 7), (17, 27)];
    let batch = loader.process_plan(plan.clone()).unwrap();

    // For row 7 → col 7, value 8 → HVG position 0.
    let n_cols = 3;
    for (i, &(p, c)) in plan.iter().enumerate() {
        let p_row = &batch.x[i * n_cols..(i + 1) * n_cols];
        let c_row = &batch.x_paired[i * n_cols..(i + 1) * n_cols];

        let p_pos = hvg
            .iter()
            .position(|&g| g as usize == (p as usize) % N_VARS);
        let c_pos = hvg
            .iter()
            .position(|&g| g as usize == (c as usize) % N_VARS);

        match p_pos {
            Some(pos) => {
                assert_eq!(p_row[pos], ((p as usize + 1) & 0xFF) as f32);
                assert_eq!(p_row.iter().filter(|&&v| v != 0.0).count(), 1);
            }
            None => assert!(p_row.iter().all(|&v| v == 0.0)),
        }
        match c_pos {
            Some(pos) => {
                assert_eq!(c_row[pos], ((c as usize + 1) & 0xFF) as f32);
                assert_eq!(c_row.iter().filter(|&&v| v != 0.0).count(), 1);
            }
            None => assert!(c_row.iter().all(|&v| v == 0.0)),
        }
    }
}

/// Constructor rejects `cache_shards < 1`.
#[test]
fn ctor_rejects_zero_cache_shards() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    let config = scx_loader::LoaderConfig {
        max_memory_mb: 1024,
        ..Default::default()
    };
    let r = scx_loader::IndexPlanLoader::new(&path, config, 0, true, 4, 1024);
    assert!(matches!(r, Err(LoaderError::ConfigError { .. })));
}

/// Constructor surfaces missing-obs-column as the typed `ObsColumnNotFound`
/// variant carrying the requested name and the available columns. Python
/// wrapper maps this to KeyError; that's covered in the Python integration
/// tests.
#[test]
fn ctor_rejects_missing_obs_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    let config = scx_loader::LoaderConfig {
        obs_columns: vec!["nonexistent_column".to_string()],
        max_memory_mb: 1024,
        ..Default::default()
    };
    let r = scx_loader::IndexPlanLoader::new(&path, config, 4, true, 4, 1024);
    match r {
        Err(LoaderError::ObsColumnNotFound { name, available }) => {
            assert_eq!(name, "nonexistent_column");
            assert!(
                available.iter().any(|c| c == "cell_id"),
                "available list should include the fixture's cell_id column; got {available:?}"
            );
        }
        Err(other) => panic!("expected ObsColumnNotFound, got {other}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

/// Constructor rejects HVG indices >= n_vars.
#[test]
fn ctor_rejects_hvg_out_of_range() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    let config = scx_loader::LoaderConfig {
        hvg_indices: Some(vec![0, 5, N_VARS as u32]), // last one is OOR
        max_memory_mb: 1024,
        ..Default::default()
    };
    let r = scx_loader::IndexPlanLoader::new(&path, config, 4, true, 4, 1024);
    match r {
        Err(LoaderError::ConfigError { reason }) => {
            assert!(reason.contains("HVG index"));
            assert!(reason.contains("out of range"));
        }
        Err(other) => panic!("expected ConfigError, got {other}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

// ---------------------------------------------------------------------------
// Singleflight + byte-budgeted cache + counters
// ---------------------------------------------------------------------------

/// N threads racing to decode the same shard go through the singleflight:
/// only one becomes the leader and decodes; every other thread either waits
/// on the leader's Condvar (`duplicate_waiters++`) or finds the populated
/// cache after the leader inserts (`hits++`).
///
/// The strongest invariant — and the load-bearing one — is `misses == 1`:
/// no matter how the N threads interleave with the leader's decode/insert,
/// the underlying `read_csr_shard` must run exactly once. The
/// `hits + duplicate_waiters` accounting confirms every non-leader thread
/// saw a deduplicated result.
#[test]
fn singleflight_dedupes_concurrent_decode() {
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::thread;

    const N_THREADS: usize = 16;

    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    let reader = scx_format_io::ScxReader::open(&path).unwrap();
    let mut backed = scx_format_io::BackedCsrReader::new(reader, /*cache_shards=*/ 4);
    let metrics = backed.enable_metrics();
    let backed = Arc::new(backed);

    let barrier = Arc::new(Barrier::new(N_THREADS));
    let mut handles = Vec::new();
    for _ in 0..N_THREADS {
        let b = Arc::clone(&backed);
        let bar = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            bar.wait();
            b.read_shard_cached_arc(0).unwrap()
        }));
    }
    let arcs: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // Every thread observes the same decoded Arc (cached after leader inserts).
    let leader = &arcs[0];
    for (i, a) in arcs.iter().enumerate().skip(1) {
        assert!(
            Arc::ptr_eq(leader, a),
            "thread {i} observed a different Arc than the leader — not deduplicated"
        );
    }

    let misses = metrics.misses.load(Ordering::Relaxed);
    let hits = metrics.hits.load(Ordering::Relaxed);

    // Load-bearing: only the leader actually decoded.
    assert_eq!(misses, 1, "exactly one decode (leader) should run");
    // Every non-leader thread eventually returns from the cache (either it
    // arrived after the leader inserted and got a hit on the first check,
    // or it was a waiter and got a hit on the post-wake re-check). Either
    // way the hits counter records `N_THREADS - 1`.
    assert_eq!(
        hits,
        (N_THREADS as u64) - 1,
        "every non-leader thread should observe a cache hit \
         (hits={hits}, threads={N_THREADS})"
    );
    // `duplicate_waiters` is intentionally not asserted on: it is
    // timing-dependent (peers can pure-hit the cache if they arrive after
    // the leader's insert) and the deterministic invariants above already
    // prove dedup.
}

/// `WeightedLruCache` evicts oldest entries when the byte cap is exceeded.
/// We size the budget so only one shard fits at a time; reading 4 shards
/// in order should produce 3 evictions and leave only the last in cache.
#[test]
fn cache_byte_budget_evicts_when_exceeded() {
    use std::sync::atomic::Ordering;

    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    // Decoded shard size for this fixture: 1 nonzero per row, 20 rows per
    // shard (N_OBS=100 / N_SHARDS=5 = 20). So per-shard bytes ≈
    //   indptr.len()=21 × 8 = 168
    //   indices.len()=20 × 4 = 80
    //   data.len()=20 × 4 = 80
    //   ≈ 328 bytes.
    // Pick a budget that admits just one entry at a time so eviction is
    // forced on every insert past the first.
    let per_shard_bytes_approx: usize = 21 * 8 + 20 * 4 + 20 * 4;
    let budget = per_shard_bytes_approx; // 1 shard fits

    let reader = scx_format_io::ScxReader::open(&path).unwrap();
    let mut backed = scx_format_io::BackedCsrReader::new_with_byte_budget(
        reader, /*cache_shards=*/ 10, // count cap loose; byte cap binds
        budget,
    );
    let metrics = backed.enable_metrics();

    for sidx in 0..N_SHARDS {
        backed.read_shard_cached_arc(sidx).unwrap();
    }

    let evictions = metrics.evictions.load(Ordering::Relaxed);
    let misses = metrics.misses.load(Ordering::Relaxed);
    assert_eq!(misses as usize, N_SHARDS, "every read should miss + decode");
    assert!(
        evictions as usize >= N_SHARDS - 1,
        "byte budget should evict ≥ N-1 entries (got evictions={evictions})"
    );
    // Last shard should be present, second-to-last should be evicted.
    assert!(
        backed.cache_contains(N_SHARDS - 1),
        "last-read shard should be the surviving entry"
    );
    assert!(
        !backed.cache_contains(0),
        "first-read shard should have been evicted"
    );
}

/// `IndexPlanIter::spawn_prefetches` skips shards already in the cache.
///
/// Strategy: run plan1 to completion through one iter (this populates the
/// LRU via `process_plan`'s read_rows_with), then start a SECOND iter on
/// the same `Arc<IndexPlanLoader>` for plan2 — which references the same
/// shards. The second iter's `spawn_prefetches` finds every touched shard
/// already cached and skips spawning, recording
/// `prefetch_skipped_cache_hit > 0` and `prefetch_tasks_spawned == 0`.
#[test]
fn iter_skips_prefetch_when_cached() {
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);
    let loader = Arc::new(open_loader(&path, /*sort_by_shard=*/ true));

    let plan: Vec<(u64, u64)> = (0u64..16).map(|i| (i, i + 1)).collect();

    // Iter 1: warm the cache by running plan1 through the loader.
    {
        let mut iter1 = Arc::clone(&loader)
            .iter_with_plans(std::iter::once(Ok(plan.clone())), /*lookahead=*/ 1);
        let _ = iter1.next().unwrap().unwrap();
        assert!(iter1.next().is_none(), "iter1 should be drained");
    }

    // Iter 2: re-run the same plan. Every shard is now cached, so the
    // pre-check filter should skip every prefetch spawn.
    let mut iter2 = Arc::clone(&loader)
        .iter_with_plans(std::iter::once(Ok(plan.clone())), /*lookahead=*/ 1);
    let im = iter2.iter_metrics();
    let _ = iter2.next().unwrap().unwrap();
    assert!(iter2.next().is_none(), "iter2 should be drained");

    let spawned = im.prefetch_tasks_spawned.load(Ordering::Relaxed);
    let skipped_hit = im.prefetch_skipped_cache_hit.load(Ordering::Relaxed);
    assert_eq!(
        spawned, 0,
        "iter2 should not spawn any prefetch tasks (every shard cached)"
    );
    assert!(
        skipped_hit > 0,
        "iter2 should record skipped_cache_hit > 0 (got skipped_hit={skipped_hit})"
    );
}
/// **Pre-refactor pin (ORG-9.10-1, drift (b)).** Every shard the plan touches
/// is accounted for exactly once across the four `IterMetrics` counters:
///
/// ```text
/// spawned + skipped_cache_hit + skipped_in_flight + skipped_block_index
///     == distinct shards touched by the plan
/// ```
///
/// `spawn_prefetches` is one `filter().map()` over a per-shard map, so the law
/// is structural — which is exactly why it is the right thing to pin before
/// `IndexPlanIter` is folded into `PlanPrefetchIter`. That arm has **no**
/// counters at all today, so "port the counters across" is otherwise an
/// unchecked claim: dropping one increment during the move would leave every
/// existing assertion green (`iter_skips_prefetch_when_cached` above reads two
/// of the four; nothing reads the other two).
///
/// Two arms of the law are exercised: a cold cache, where every shard is
/// spawned, and a warm one, where every shard is skipped as a cache hit. Be
/// precise about what that does **not** cover — the conservation law holds
/// whatever the split, so a zero term is not evidence about its increment path:
///
/// * `prefetch_skipped_block_index` stays zero here (the fixture is unframed).
///   Its increment is pinned only from Python, by
///   `pyscx/tests/test_index_plan_dataset.py::test_scatter_block_index_flag_gates_prefetch_skip`.
///   `plan_engine_tests::engine_does_not_warm_a_block_index_eligible_shard` is
///   *not* a second pin for it: that test asserts `CacheMetrics`'
///   `block_index_groups`, and the engine arm has no `IterMetrics` at all —
///   which is drift (b) itself.
/// * `prefetch_skipped_in_flight` is never shown to increment anywhere. Closing
///   that needs a concurrent peer decode mid-`spawn_prefetches`; it is a real
///   remaining gap, not something this test covers.
#[test]
fn iter_prefetch_counters_account_for_every_touched_shard() {
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);
    // `cache_shards = 8` (the shared helpers pin 4) so all five touched shards
    // stay resident between the two arms — under a 4-shard LRU the warm arm
    // re-spawns the two evicted shards, which is a cache property, not the
    // accounting property under test. The law holds either way; the per-counter
    // assertions below would not.
    let loader = Arc::new(
        scx_loader::IndexPlanLoader::new(
            &path,
            scx_loader::LoaderConfig {
                normalize: false,
                log1p: false,
                obs_columns: vec!["cell_id".to_string()],
                max_memory_mb: 1024,
                ..Default::default()
            },
            /*cache_shards=*/ 8,
            /*sort_by_shard=*/ true,
            /*lookahead=*/ 4,
            /*max_plan_size=*/ 16384,
        )
        .unwrap(),
    );

    // Two rows in each of the five 20-row shards → 5 distinct shards, 10 rows.
    let plan: Vec<(u64, u64)> = vec![(0, 1), (25, 30), (45, 50), (65, 70), (85, 90)];
    const TOUCHED_SHARDS: u64 = 5;

    let drain = |loader: &Arc<scx_loader::IndexPlanLoader>| -> [u64; 4] {
        let mut it = Arc::clone(loader)
            .iter_with_plans(std::iter::once(Ok(plan.clone())), /*lookahead=*/ 1);
        let im = it.iter_metrics();
        let _ = it.next().expect("one batch").expect("gather");
        assert!(it.next().is_none(), "iter should be drained");
        [
            im.prefetch_tasks_spawned.load(Ordering::Relaxed),
            im.prefetch_skipped_cache_hit.load(Ordering::Relaxed),
            im.prefetch_skipped_in_flight.load(Ordering::Relaxed),
            im.prefetch_skipped_block_index.load(Ordering::Relaxed),
        ]
    };

    let cold = drain(&loader);
    assert_eq!(
        cold.iter().sum::<u64>(),
        TOUCHED_SHARDS,
        "cold: counters must account for all {TOUCHED_SHARDS} touched shards, got {cold:?}"
    );
    assert_eq!(
        cold[0], TOUCHED_SHARDS,
        "cold: every touched shard must be spawned, got {cold:?}"
    );

    let warm = drain(&loader);
    assert_eq!(
        warm.iter().sum::<u64>(),
        TOUCHED_SHARDS,
        "warm: counters must account for all {TOUCHED_SHARDS} touched shards, got {warm:?}"
    );
    assert_eq!(
        warm[1], TOUCHED_SHARDS,
        "warm: every touched shard must be skipped as a cache hit, got {warm:?}"
    );
}

/// **Pre-refactor pin (ORG-9.10-1).** `sort_by_shard = true` emits pairs in
/// `min(shard_of(p), shard_of(c))` order, **stably** — ties keep the caller's
/// order.
///
/// `process_plan` takes `Vec<(u64,u64)>` **by value** and does this sort in
/// place, then moves the sorted plan into the batch's `pairs`.
///
/// Written as a pre-refactor pin on the assumption that ORG-9.10-1 would have
/// to re-sign it — `PlanPrefetchIter`'s `ProcFn` took `&P`, and the engine's
/// module docs say it preserves plan order ("no `sort_by_shard` reorder"). The
/// fold went the other way instead: `ProcFn` now takes the plan **by value**,
/// because the in-flight queue is its last owner, so `process_plan` is
/// untouched and there is no per-batch clone. The pin still earns its place —
/// it is the only test here that can see the sort stop happening.
///
/// The two existing sort tests cannot: `sort_by_shard_pairs_align_with_rows`
/// only checks `pairs[i]` against row `i`, and `sort_by_shard_is_pure_permutation`
/// sorts both sides before comparing. Both stay green if the sort silently stops
/// happening.
#[test]
fn sort_by_shard_emits_pairs_in_shard_order_stably() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    // 100 obs / 5 shards → 20 rows each. Sort keys, in input order:
    //   (95, 0)  -> min(shard 4, shard 0) = 0
    //   (40, 50) -> min(shard 2, shard 2) = 2
    //   (80, 90) -> min(shard 4, shard 4) = 4
    //   (3, 12)  -> min(shard 0, shard 0) = 0
    //   (60, 11) -> min(shard 3, shard 0) = 0
    //   (1, 2)   -> min(shard 0, shard 0) = 0
    let plan: Vec<(u64, u64)> = vec![(95, 0), (40, 50), (80, 90), (3, 12), (60, 11), (1, 2)];

    let sorted = open_loader(&path, /*sort_by_shard=*/ true)
        .process_plan(plan.clone())
        .unwrap();
    assert_eq!(
        sorted.pairs,
        vec![(95, 0), (3, 12), (60, 11), (1, 2), (40, 50), (80, 90)],
        "keys 0,0,0,0,2,4 — and the four key-0 pairs keep their input order \
         (the sort is stable)"
    );

    let unsorted = open_loader(&path, /*sort_by_shard=*/ false)
        .process_plan(plan.clone())
        .unwrap();
    assert_eq!(
        unsorted.pairs, plan,
        "with the flag off, caller order survives"
    );
}

// ---------------------------------------------------------------------------
// BudgetBreakdown + peak_bytes_in_cache
// ---------------------------------------------------------------------------

/// `BudgetBreakdown.total_bytes` is the saturating sum of every component
/// field. The auto-tune in `IndexPlanLoader::new` produces a breakdown
/// after possibly reducing `effective_lookahead` / `effective_cache_shards`,
/// so the constructor's stored breakdown should be self-consistent.
#[test]
fn budget_breakdown_components_sum_to_total() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);
    let loader = open_loader(&path, /*sort_by_shard=*/ true);
    let b = loader.budget_breakdown();
    let sum = b
        .cache_bytes
        .saturating_add(b.batch_buffer_bytes)
        .saturating_add(b.lookahead_overhead_bytes)
        .saturating_add(b.transient_bytes)
        .saturating_add(b.python_overhead_bytes);
    assert_eq!(
        sum,
        b.total_bytes,
        "components ({} + {} + {} + {} + {}) must equal total_bytes ({})",
        b.cache_bytes,
        b.batch_buffer_bytes,
        b.lookahead_overhead_bytes,
        b.transient_bytes,
        b.python_overhead_bytes,
        b.total_bytes,
    );
}

/// `transient_bytes` should account for both per-batch obs Vecs (one per
/// configured obs column per side) and the `PairRequest` sort scratch
/// (~24 B per pair × 2). Lower-bound the term against an explicit formula
/// so a future refactor that removes the obs accounting fails this test.
#[test]
fn budget_breakdown_includes_transient_terms() {
    use scx_loader::{IndexPlanLoader, LoaderConfig};

    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    // Single obs column (fixture only has `cell_id`), max_plan_size 8192 →
    // expect transient ≥
    //   2 × 8192 × 1 × 8        = 128 KiB  (obs Vec, both sides)
    //   + 2 × 8192 × 24         = 384 KiB  (PairRequest scratch)
    //   = 512 KiB.
    // Also verify the no-obs case has request scratch only (smaller bound).
    let n_obs_cols: usize = 1;
    let max_plan_size = 8192usize;
    let config = LoaderConfig {
        normalize: false,
        log1p: false,
        obs_columns: vec!["cell_id".to_string()],
        max_memory_mb: 1024,
        ..Default::default()
    };
    let loader = IndexPlanLoader::new(
        &path,
        config,
        /*cache_shards*/ 4,
        /*sort_by_shard*/ true,
        /*lookahead*/ 4,
        max_plan_size,
    )
    .unwrap();
    let lower_bound = (2 * max_plan_size * n_obs_cols * 8) + (2 * max_plan_size * 24);
    let transient = loader.budget_breakdown().transient_bytes;
    assert!(
        transient >= lower_bound,
        "transient_bytes ({transient}) should be >= obs+request lower bound ({lower_bound})"
    );

    // No-obs case: should still account for the PairRequest scratch but
    // not for any obs Vecs. Budget should drop by exactly the obs term.
    let cfg2 = LoaderConfig {
        normalize: false,
        log1p: false,
        obs_columns: vec![],
        max_memory_mb: 1024,
        ..Default::default()
    };
    let loader2 = IndexPlanLoader::new(
        &path,
        cfg2,
        /*cache_shards*/ 4,
        /*sort_by_shard*/ true,
        /*lookahead*/ 4,
        max_plan_size,
    )
    .unwrap();
    let transient2 = loader2.budget_breakdown().transient_bytes;
    let request_only = 2 * max_plan_size * 24;
    assert!(
        transient2 >= request_only,
        "no-obs transient_bytes ({transient2}) should still cover request scratch ({request_only})"
    );
    assert!(
        transient2 < transient,
        "removing obs columns should shrink transient_bytes \
         (with-obs={transient}, no-obs={transient2})"
    );
}

/// `CacheMetrics.peak_bytes_in_cache` is a `fetch_max`-updated high-water
/// gauge. Read N shards in order; the gauge should advance and end at a
/// value that's both > 0 and consistent with the cumulative
/// `bytes_inserted` minus what's been evicted.
#[test]
fn peak_bytes_in_cache_records_high_water() {
    use std::sync::atomic::Ordering;

    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    let reader = scx_format_io::ScxReader::open(&path).unwrap();
    let mut backed = scx_format_io::BackedCsrReader::new(reader, /*cache_shards=*/ 16);
    let metrics = backed.enable_metrics();

    // Touch every shard so the cache fills up (N_SHARDS = 5 well below the
    // count cap of 16, so no eviction).
    for sidx in 0..N_SHARDS {
        backed.read_shard_cached_arc(sidx).unwrap();
    }

    let peak = metrics.peak_bytes_in_cache.load(Ordering::Relaxed);
    let inserted = metrics.bytes_inserted.load(Ordering::Relaxed);
    let evictions = metrics.evictions.load(Ordering::Relaxed);

    assert!(peak > 0, "peak_bytes_in_cache should advance past 0");
    // No eviction at this size, so peak == inserted at end.
    assert_eq!(
        evictions, 0,
        "no eviction expected with cache_shards=16 > N_SHARDS={N_SHARDS}"
    );
    assert_eq!(
        peak, inserted,
        "without eviction, peak should equal cumulative bytes_inserted"
    );
}
