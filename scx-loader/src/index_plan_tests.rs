use super::*;

use std::sync::Arc as StdArc;

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
        n_obs % n_shards == 0,
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
    let mut config = LoaderConfig::default();
    config.normalize = false;
    config.log1p = false;
    config.obs_columns = vec!["cell_id".to_string()];
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
    let mut config = LoaderConfig::default();
    config.normalize = false;
    config.log1p = false;
    config.obs_columns = vec!["cell_id".to_string()];
    config.hvg_indices = Some(hvg_indices);
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
    let triples = |b: &IndexPlanBatch| -> Vec<((u64, u64), Vec<u32>, Vec<u32>)> {
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
    let mut config = LoaderConfig::default();
    config.normalize = false;
    config.log1p = false;
    config.obs_columns = vec!["cell_id".to_string()];
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

/// With the sidecar-aware prefetch, a sparse plan
/// driven at `lookahead=4` must STILL reach the O(rows) sidecar path — the
/// prefetch skips warming sidecar-eligible cold shards instead of negating the
/// gather. Pre-L2 this read `sidecar_groups=0`. Output must stay
/// byte-identical to `lookahead=0`.
#[test]
fn iter_with_plans_lookahead_four_still_reaches_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let scx1 = write_dense_scx1_fixture(&dir.path().join("scx1.scx"), 256, 8, 256, CodecId::Scx1);
    assert_eq!(count_sidecars(&scx1), 8, "fixture must carry sidecars");

    // Sparse-per-shard plans: ≤3 unique rows per 32-row shard → eligible.
    let plans = vec![
        vec![(3u64, 200u64), (40, 12), (70, 5)],
        vec![(130, 5), (250, 200), (12, 3)],
    ];

    // Wide var axis (sidecar-friendly deltas) → small max_plan_size + generous
    // budget, as in the Phase-1 sidecar test.
    let mk = || {
        let mut config = LoaderConfig::default();
        config.normalize = false;
        config.log1p = false;
        config.obs_columns = vec!["cell_id".to_string()];
        config.max_memory_mb = 4096;
        Arc::new(
            IndexPlanLoader::new(
                &scx1, config, /*cache_shards*/ 8, /*sort_by_shard*/ false,
                /*lookahead*/ 4, /*max_plan_size*/ 256,
            )
            .unwrap(),
        )
    };
    let loader0 = mk();
    let loader4 = mk();
    let zero: Vec<_> = Arc::clone(&loader0)
        .iter_with_plans(into_plan_iter(plans.clone()), 0)
        .map(|r| r.unwrap())
        .collect();
    let four: Vec<_> = Arc::clone(&loader4)
        .iter_with_plans(into_plan_iter(plans.clone()), 4)
        .map(|r| r.unwrap())
        .collect();

    // Hard gate: lookahead=4 still adopts the sidecar (prefetch skipped eligible
    // shards rather than warming them).
    let sg = loader4
        .cache_metrics()
        .sidecar_groups
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        sg > 0,
        "lookahead=4 must still reach the sidecar after L2 (sidecar-aware prefetch); got {sg}"
    );

    // Byte-identical output vs lookahead=0.
    assert_eq!(zero.len(), four.len());
    for (a, b) in zero.iter().zip(four.iter()) {
        assert_eq!(a.pairs, b.pairs, "pair order must match lookahead 0 vs 4");
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
    assert!(n_obs % n_shards == 0);
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

/// Build a dense multi-shard `.scx` fixture with **strictly increasing,
/// collision-free** columns per row (mirrors the proven
/// `scx-format-io backed_tests::write_sidecar_scx1_file` recipe), written with
/// the given `codec`. With `CodecId::Scx1` + dense integer rows the writer
/// emits a per-row decode sidecar per shard (within the 25% overhead budget);
/// with `CodecId::None` it emits none. Two calls with identical params produce
/// byte-identical logical data, enabling a sidecar-vs-full-shard parity check.
fn write_dense_scx1_fixture(
    path: &std::path::Path,
    n_obs: usize,
    n_shards: usize,
    nnz_per_row: usize,
    codec: CodecId,
) -> std::path::PathBuf {
    assert!(n_obs % n_shards == 0);
    let rows_per_shard = n_obs / n_shards;
    // Strictly increasing cols with gaps up to 250 (like the proven
    // backed_tests recipe) → max col < nnz_per_row * 251. Large-ish deltas keep
    // the Scx1-encoded shard from compressing so far that the per-row sidecar
    // exceeds the 25% overhead budget and gets skipped.
    let n_vars = nnz_per_row * 251 + 16;
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
            let mut col = 0u32;
            for k in 0..nnz_per_row {
                col += 1 + ((row * 13 + k * 7) % 250) as u32; // strictly increasing
                indices.push(col);
                values.push(1u8 + ((row + k) % 5) as u8); // nonzero 1..=5
            }
            indptr.push(*indptr.last().unwrap() + nnz_per_row as u64);
        }
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                codec,
                ValueEncoding::Uint8,
                row_start as u64,
            )
            .unwrap();
    }
    writer.finish().unwrap();
    path.to_path_buf()
}

/// Count `DecodeMetadataShard` (per-row scx1 decode sidecar) sections in a file.
fn count_sidecars(path: &std::path::Path) -> usize {
    let reader = ScxReader::open(path).unwrap();
    reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == scx_format_io::SectionType::DecodeMetadataShard)
        .count()
}

/// L1 (STATE-TX-SIDECAR.md): a cache-cold, sparse-per-shard plan gathered
/// through `gather_pairs_dense` must reach the O(rows) scx1 **decode sidecar**
/// (`read_rows_with` → `scatter_group_via_sidecar`), bumping
/// `CacheMetrics.sidecar_groups`. And the sidecar gather must be **byte-identical**
/// to a full-shard-decode gather of the same logical data (a `CodecId::None`
/// copy, which carries no sidecar so always takes the full-shard path).
#[test]
fn gather_pairs_dense_reaches_sidecar_and_matches_full_decode() {
    let dir = tempfile::tempdir().unwrap();
    // Dense integer rows keep the per-row sidecar overhead well under the 25%
    // budget, so every Scx1 shard carries a sidecar. 8 shards × 32 rows = 256.
    let scx1 = write_dense_scx1_fixture(&dir.path().join("scx1.scx"), 256, 8, 256, CodecId::Scx1);
    let none = write_dense_scx1_fixture(&dir.path().join("none.scx"), 256, 8, 256, CodecId::None);
    assert_eq!(
        count_sidecars(&scx1),
        8,
        "every Scx1 shard must carry a decode sidecar (else the test is vacuous)"
    );
    assert_eq!(count_sidecars(&none), 0, "CodecId::None carries no sidecar");

    // Sparse-per-shard plan: a few scattered rows across distinct shards so
    // `group_len * 4 < shard_rows (32)` holds → sidecar-eligible. Include a
    // reused control row and a `pert == ctrl` pair to exercise the fan-out map.
    let plan: Vec<(u64, u64)> = vec![(3, 200), (40, 200), (70, 12), (130, 130), (250, 5)];

    // Fresh (cache-cold) loaders, no prefetch on the synchronous `process_plan`
    // path. Construct inline: the sidecar-friendly fixture has a wide var axis
    // (large deltas keep the sidecar under budget), so use a small max_plan_size
    // + generous memory budget to keep the dense batch buffer in range.
    let mk = |p: &std::path::Path| {
        let mut config = LoaderConfig::default();
        config.normalize = false;
        config.log1p = false;
        config.obs_columns = vec!["cell_id".to_string()];
        config.max_memory_mb = 4096;
        IndexPlanLoader::new(
            p, config, /*cache_shards*/ 8, /*sort_by_shard*/ false, /*lookahead*/ 0,
            /*max_plan_size*/ 256,
        )
        .unwrap()
    };
    let scx1_loader = mk(&scx1);
    let none_loader = mk(&none);

    let scx1_batch = scx1_loader.process_plan(plan.clone()).unwrap();
    let none_batch = none_loader.process_plan(plan.clone()).unwrap();

    let m = scx1_loader.cache_metrics();
    assert!(
        m.sidecar_groups.load(std::sync::atomic::Ordering::Relaxed) > 0,
        "cold sparse gather on the Scx1 fixture must reach the sidecar path \
         (sidecar_groups > 0); got {}",
        m.sidecar_groups.load(std::sync::atomic::Ordering::Relaxed)
    );
    let m_none = none_loader.cache_metrics();
    assert_eq!(
        m_none
            .sidecar_groups
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "sidecar-less fixture must take the full-shard path only"
    );

    // Byte-identical output: sidecar decode vs full-shard decode of the same data.
    assert_eq!(scx1_batch.pairs, none_batch.pairs);
    let p_bits = |b: &[f32]| b.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        p_bits(&scx1_batch.x),
        p_bits(&none_batch.x),
        "perturbed-side gather must be byte-identical across sidecar/full-shard"
    );
    assert_eq!(
        p_bits(&scx1_batch.x_paired),
        p_bits(&none_batch.x_paired),
        "control-side gather must be byte-identical across sidecar/full-shard"
    );
}

/// Generous memory budget — both effective values match the requested.
#[test]
fn budget_generous_no_autotune() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 32, 8, 4);
    let mut config = LoaderConfig::default();
    config.max_memory_mb = 4096; // way over budget needed for this tiny file
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

    let mut config = LoaderConfig::default();
    // Budget components at requested settings (post-L1-dedup-gather model):
    //   python       = 50 MB
    //   batch buffer = 2 × 65536 × 8 × 4         ≈ 4 MB
    //   lookahead    = 8 × 65536 × 16            ≈ 8 MB
    //   transient    = 2 × 65536 × 64 (no obs)   ≈ 8 MB  (dedup scratch:
    //                  unique_rows + row_to_requests + row_to_pos, per-row)
    //   shard cache  = ~negligible (sparse fixture)
    // Total ≈ 70 MB. Floor at lookahead=1: 50+4+1+8 ≈ 63 MB.
    // A 66 MB budget forces lookahead reduction without floor-failure.
    config.max_memory_mb = 66;
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

    let mut config = LoaderConfig::default();
    config.max_memory_mb = 96;
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

    let mut config = LoaderConfig::default();
    config.max_memory_mb = 40; // below the 50 MB python overhead alone
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

/// Caller explicitly chooses lookahead=0 — honored when it fits the
/// budget (no prefetch path activated).
#[test]
fn budget_lookahead_zero_honored_when_fits() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 32, 8, 4);
    let mut config = LoaderConfig::default();
    config.max_memory_mb = 256;
    let loader = IndexPlanLoader::new(
        &path, config, /*cache_shards*/ 4, /*sort_by_shard*/ true, /*lookahead*/ 0,
        /*max_plan_size*/ 1024,
    )
    .unwrap();
    assert_eq!(loader.effective_lookahead(), 0);
}

/// `process_plan` must reject plans larger than `max_plan_size` so a
/// misbehaving consumer cannot silently exceed the memory budget.
/// Plans at or below the ceiling are accepted as before.
#[test]
fn process_plan_rejects_oversize_plan() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 32, 8, 4);
    let mut config = LoaderConfig::default();
    config.normalize = false;
    config.log1p = false;
    config.obs_columns = vec!["cell_id".to_string()];

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
// The loader holds a `OnceLock<Runtime>` and defers tokio runtime
// construction to the first `iter_with_plans` call. The contract is
// checked here by inspecting `loader.runtime.get()` before and after
// touching the iter; constructing the loader must not build a
// runtime, since that runtime would otherwise be inherited by
// `DataLoader` worker processes after fork.

#[test]
fn lazy_runtime_not_built_at_construction() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
    let loader = open_loader(&path, false);
    assert!(
        loader.runtime.get().is_none(),
        "IndexPlanLoader::new must not eagerly build the tokio runtime — \
             that would defeat the DataLoader fork-safety contract"
    );
}

#[test]
fn lazy_runtime_built_after_iter_with_plans_consumed() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
    let loader = Arc::new(open_loader(&path, false));

    // Sanity-check the precondition.
    assert!(loader.runtime.get().is_none());

    let plans = vec![Ok(vec![(0u64, 1u64), (2, 3)])].into_iter();
    let iter = Arc::clone(&loader).iter_with_plans(plans, /*lookahead*/ 2);
    // Consume one batch — this drives `refill` → `spawn_prefetches`
    // → the lazy `runtime()` accessor.
    let mut iter = iter;
    let _ = iter.next();
    assert!(
        loader.runtime.get().is_some(),
        "first iter consumption should materialize the runtime via OnceLock"
    );
}

#[test]
fn lazy_runtime_idempotent_under_concurrent_init() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("f.scx"), 16, 8, 4);
    let loader = Arc::new(open_loader(&path, false));

    // Spawn several threads that each grab the runtime through the
    // private accessor. Compare pointer addresses to confirm all
    // observers see the same Runtime instance.
    let mut handles = Vec::new();
    for _ in 0..4 {
        let l = Arc::clone(&loader);
        handles.push(std::thread::spawn(move || {
            let rt = l.runtime().expect("runtime build must succeed");
            rt as *const Runtime as usize
        }));
    }
    let addrs: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let first = addrs[0];
    for a in &addrs[1..] {
        assert_eq!(
            *a, first,
            "all threads must observe the same Runtime instance"
        );
    }
}
