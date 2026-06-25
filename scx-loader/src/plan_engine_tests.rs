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
    assert!(n_obs % n_shards == 0, "n_obs must divide n_shards");
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

/// Single-file dense Scx1 fixture with strictly-increasing, collision-free
/// columns (gaps up to 250) so the writer emits a per-row decode sidecar per
/// shard. Mirrors `index_plan_tests::write_dense_scx1_fixture`.
fn write_dense_scx1_fixture(
    path: &std::path::Path,
    n_obs: usize,
    n_shards: usize,
    nnz_per_row: usize,
) {
    assert!(n_obs % n_shards == 0);
    let rows_per_shard = n_obs / n_shards;
    let n_vars = nnz_per_row * 251 + 16;
    let header = FileHeader::new_single_modality(
        n_obs as u64,
        n_vars as u64,
        (n_obs * nnz_per_row) as u64,
        rows_per_shard as u32,
        0,
        0,
    );
    let mut writer = ScxWriter::new(path, header).unwrap();

    let obs_schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
    let cell_ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    writer
        .write_obs(
            &arrow::record_batch::RecordBatch::try_new(
                StdArc::new(obs_schema),
                vec![StdArc::new(StringArray::from(
                    cell_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                ))],
            )
            .unwrap(),
        )
        .unwrap();
    let var_schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    let gene_ids: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    writer
        .write_var(
            &arrow::record_batch::RecordBatch::try_new(
                StdArc::new(var_schema),
                vec![StdArc::new(StringArray::from(
                    gene_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                ))],
            )
            .unwrap(),
        )
        .unwrap();

    for s in 0..n_shards {
        let row_start = s * rows_per_shard;
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for local in 0..rows_per_shard {
            let row = row_start + local;
            let mut col = 0u32;
            for k in 0..nnz_per_row {
                col += 1 + ((row * 13 + k * 7) % 250) as u32;
                indices.push(col);
                values.push(1u8 + ((row + k) % 5) as u8);
            }
            indptr.push(*indptr.last().unwrap() + nnz_per_row as u64);
        }
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::Scx1,
                ValueEncoding::Uint8,
                row_start as u64,
            )
            .unwrap();
    }
    writer.finish().unwrap();
}

fn count_sidecars(path: &std::path::Path) -> usize {
    ScxReader::open(path)
        .unwrap()
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == scx_format_io::SectionType::DecodeMetadataShard)
        .count()
}

/// Gather full CSR rows (not just the first nonzero) — for codec-agnostic
/// self-parity checks across lookahead settings.
fn gather_full(engine: &PrefetchEngine, plan: &Plan) -> Result<Vec<(Vec<i32>, Vec<f32>)>> {
    let mut out = Vec::with_capacity(plan.len());
    for &(fid, row) in plan {
        let mut got = (Vec::new(), Vec::new());
        engine
            .reader(fid)
            .read_rows_with(&[row], |_pos, idx, data| {
                got = (idx.to_vec(), data.to_vec());
                Ok(())
            })
            .map_err(LoaderError::FormatError)?;
        out.push(got);
    }
    Ok(out)
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
        /*scatter_sidecar*/ true,
    )
}

/// Plan = list of `(file_id, row)`; `rows_of` just clones it.
type Plan = Vec<(u32, u64)>;

fn rows_of(plan: &Plan) -> Vec<(u32, u64)> {
    plan.clone()
}

/// Gather the single non-zero `(col, value)` of each plan row through the
/// engine's readers — exercises the real `read_rows_with` path.
fn gather(engine: &PrefetchEngine, plan: &Plan) -> Result<Vec<(i32, f32)>> {
    let mut out = Vec::with_capacity(plan.len());
    for &(fid, row) in plan {
        let mut got: Option<(i32, f32)> = None;
        engine
            .reader(fid)
            .read_rows_with(&[row], |_pos, idx, data| {
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

/// L2 sidecar-aware prefetch on the sparse cell-set path: with the sidecar-aware
/// prefetch, a sparse plan driven at `lookahead=4` must STILL reach the O(rows)
/// sidecar path — the prefetch skips warming sidecar-eligible cold shards. Hard
/// gate on `sidecar_groups > 0`; output byte-identical to `lookahead=0`.
#[test]
fn engine_lookahead_four_still_reaches_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scx1.scx");
    write_dense_scx1_fixture(&path, 128, 4, 256); // 4 shards × 32 rows
    assert_eq!(count_sidecars(&path), 4, "fixture must carry sidecars");

    // Sparse-per-shard plan: ≤3 unique rows per 32-row shard → eligible.
    let plans = vec![
        vec![(0u32, 3u64), (0, 40), (0, 70)],
        vec![(0, 100), (0, 12), (0, 5)],
    ];

    let run = |lookahead: usize| -> (Vec<Vec<(Vec<i32>, Vec<f32>)>>, u64) {
        let reader = ScxReader::open(&path).unwrap();
        let engine = PrefetchEngine::from_scx_readers(
            vec![reader],
            /*cache_shards*/ 8,
            usize::MAX,
            lookahead,
            /*scatter_sidecar*/ true,
        );
        let out: Vec<_> = Arc::clone(&engine)
            .iter_with_plans(into_iter(plans.clone()), lookahead, rows_of, gather_full)
            .map(|r| r.unwrap())
            .collect();
        let sg = engine
            .cache_metrics()
            .sidecar_groups
            .load(std::sync::atomic::Ordering::Relaxed);
        (out, sg)
    };

    let (zero, _) = run(0);
    let (four, sg_four) = run(4);
    assert!(
        sg_four > 0,
        "lookahead=4 must still reach the sidecar after L2 (sidecar-aware prefetch); got {sg_four}"
    );
    assert_eq!(
        zero, four,
        "output must be byte-identical across lookahead 0 vs 4"
    );
}

/// Per-reader `scatter_sidecar = false` (the sparse cell-set default, SCX-CACHE-
/// SHARDS.md Phase 1): the SAME sparse plan that reaches the sidecar when the
/// gate is on must instead take the full-shard cached path and warm reused
/// shards into the LRU — `sidecar_groups == 0`, `full_shard_groups > 0`, and a
/// shard touched by two plans yields a cache `hit`. Output stays byte-identical
/// to the sidecar-on run (path choice never changes results).
#[test]
fn engine_scatter_sidecar_off_warms_cache_instead_of_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scx1.scx");
    write_dense_scx1_fixture(&path, 128, 4, 256); // 4 shards × 32 rows
    assert_eq!(count_sidecars(&path), 4, "fixture must carry sidecars");

    // Plan 1 touches shards {0,1,2}; plan 2 touches {3,0} — shard 0 is reused.
    let plans = vec![
        vec![(0u32, 3u64), (0, 40), (0, 70)],
        vec![(0, 100), (0, 12), (0, 5)],
    ];

    let build = |scatter_sidecar: bool| -> Arc<PrefetchEngine> {
        PrefetchEngine::from_scx_readers(
            vec![ScxReader::open(&path).unwrap()],
            /*cache_shards*/ 8,
            usize::MAX,
            /*lookahead*/ 4,
            scatter_sidecar,
        )
    };
    let collect = |engine: &Arc<PrefetchEngine>| -> Vec<Vec<(Vec<i32>, Vec<f32>)>> {
        Arc::clone(engine)
            .iter_with_plans(into_iter(plans.clone()), 4, rows_of, gather_full)
            .map(|r| r.unwrap())
            .collect()
    };
    let load = |c: &std::sync::atomic::AtomicU64| c.load(std::sync::atomic::Ordering::Relaxed);

    let off = build(false);
    let off_out = collect(&off);
    let m = off.cache_metrics();
    assert_eq!(
        load(&m.sidecar_groups),
        0,
        "scatter_sidecar=false must never take the sidecar path"
    );
    assert!(
        load(&m.full_shard_groups) > 0,
        "scatter_sidecar=false must use the full-shard cached path"
    );
    assert!(
        load(&m.hits) > 0,
        "the shard reused across both plans must produce a cache hit (warmed LRU)"
    );

    // Path choice never changes results: identical output with the sidecar on.
    let on_out = collect(&build(true));
    assert_eq!(
        off_out, on_out,
        "output must be byte-identical regardless of the scatter_sidecar gate"
    );
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
