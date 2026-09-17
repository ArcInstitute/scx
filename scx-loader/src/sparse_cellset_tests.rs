use super::*;

use std::sync::Arc as StdArc;

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;
use scx_format_io::{BackedCsrReader, ScxReader};

/// Minimal multi-shard `.scx`: row `r` has one non-zero at column `r % n_vars`
/// with value `((r + 1) & 0xFF)`. Mirrors the plan_engine / index_plan fixture.
fn write_fixture(path: &std::path::Path, n_obs: usize, n_vars: usize, n_shards: usize) {
    assert!(n_obs.is_multiple_of(n_shards));
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

/// Deterministic `(col, value)` for fixture row `r`.
fn expected_cell(row: u64, n_vars: usize) -> (i32, f32) {
    ((row as usize % n_vars) as i32, ((row + 1) & 0xFF) as f32)
}

/// Per output row `j` of a batch: its `(indices, data)` slice.
fn batch_row(batch: &SparseCellSetBatch, j: usize) -> (&[i32], &[f32]) {
    let lo = batch.indptr[j] as usize;
    let hi = batch.indptr[j + 1] as usize;
    (&batch.indices[lo..hi], &batch.data[lo..hi])
}

fn open(path: &std::path::Path) -> ScxReader {
    ScxReader::open(path).unwrap()
}

/// **Review on #528 (codex).** A cell-set plan is gathered one
/// `read_rows_with` call per set, so a per-gather admission rule sees one set
/// at a time: two sets that each fit `budget / (lookahead + 1)` but whose union
/// does not would both be retained and evict each other — the insert/evict/
/// zero-hit scan the admission rule exists to bypass. The verdict is taken once
/// per plan by the engine and carried into every set's gather. Falsifier:
/// `row_group_bytes_inserted > 0` means a per-set decision admitted.
#[test]
fn cellset_plan_admission_is_per_plan_not_per_set() {
    use crate::plan_engine::tests::{framed_expected, write_framed_fixture, FRAMED_GROUP_BYTES};
    use std::sync::atomic::Ordering as AtomicOrdering;

    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("f0.scx");
    write_framed_fixture(&p0);
    // Set A: rows 5/70/140/200 → groups 0/4/8/12; set B: rows 20/85/155/215 →
    // groups 1/5/9/13. Each set is 4 groups; the union is 8.
    let per_set = 4 * FRAMED_GROUP_BYTES;
    let budget = 5 * (per_set + per_set / 2);
    assert!(budget / 5 >= per_set && budget / 5 < 2 * per_set, "premise");
    let loader = SparseCellSetLoader::new(
        vec![open(&p0)],
        /*cache_shards*/ 16,
        Some(budget),
        /*lookahead*/ 4,
        /*remap*/ None,
        /*n_global_genes*/ None,
        false,
        false,
        0.0,
        /*downsample*/ None,
        /*scatter_block_index*/ true,
        /*max_plan_rows*/ None,
    )
    .unwrap();
    let rows: Vec<u64> = vec![5, 70, 140, 200, 20, 85, 155, 215];
    let plan = SparseCellSetPlan {
        file_ids: vec![0; 8],
        rows: rows.clone(),
        role_tags: vec![0, 0, 0, 0, 1, 1, 1, 1],
        set_offsets: vec![0, 4, 8],
    };
    let batches: Vec<_> = StdArc::clone(&loader)
        .iter_with_plans(vec![Ok(plan.clone())].into_iter(), 4)
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(batches.len(), 1);
    for (j, &row) in rows.iter().enumerate() {
        let (idx, dat) = batch_row(&batches[0], j);
        let (ecol, eval) = framed_expected(row);
        assert_eq!((idx, dat), (&[ecol][..], &[eval][..]), "row {j}");
    }
    let m = loader.cache_metrics();
    assert!(
        m.block_index_groups.load(AtomicOrdering::Relaxed) > 0,
        "premise: the framed gather took the row-group route"
    );
    assert_eq!(m.row_group_misses.load(AtomicOrdering::Relaxed), 8);
    assert_eq!(
        m.row_group_hits.load(AtomicOrdering::Relaxed)
            + m.row_group_bytes_inserted.load(AtomicOrdering::Relaxed),
        0,
        "two sets that fit one at a time but not together must retain nothing"
    );
}

/// **Review on #528 round 2 (codex), re-pinned for W11.** The plan's admission
/// sum must not key on the plan-level density window. Sixteen rows of one
/// 64-row shard across four sets of four: dense as a plan, sparse per set. Four
/// groups exceed the budget, so the plan must be refused; a density-filtered
/// sum (0 bytes) would admit it and the gathers would insert. Falsifier: any
/// `row_group_bytes_inserted`, under **either** executor.
///
/// ⚠️ The two executors take **different routes** on this shape, and that is
/// the finding rather than a defect. The per-set walk gathers four rows at a
/// time, so the shard reads as sparse and every set takes the row-group path —
/// which is the mismatch this test was written to expose, since the engine's
/// sum had already bucketed the same rows as dense. The W11 batch executor
/// reads the plan's rows in one call, so its bucketing **is** the sum's: the
/// shard reads as dense and the gather takes the full-shard path. The hole is
/// closed by construction there rather than by the sum's behaviour, so both
/// arms are asserted — the `set` arm keeps the original guard alive for
/// `SCX_CELLSET_EXECUTOR=set`.
///
/// `lookahead = 0` on purpose: with prefetch on, the dense plan-level bucket
/// makes the prefetcher warm the shard **whole**, and the gathers then slice
/// the resident shard (`full_shard_groups`) — the per-set hole only opens when
/// nothing warmed the shard first, which is exactly the no-prefetch path.
#[test]
fn cellset_plan_admission_ignores_plan_level_density() {
    use crate::plan_engine::tests::{framed_expected, write_framed_fixture, FRAMED_GROUP_BYTES};
    use std::sync::atomic::Ordering as AtomicOrdering;

    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("f0.scx");
    write_framed_fixture(&p0);
    // 16 rows in shard 0 (64 rows, 4 groups of 16): 16 × 4 ≥ 64 is dense at
    // plan level; each set of 4 rows is sparse (4 × 4 < 64).
    let rows: Vec<u64> = (0..16u64).map(|i| i * 4).collect();
    // At lookahead 0 the share is the whole budget: three and a half groups.
    let budget = 3 * FRAMED_GROUP_BYTES + FRAMED_GROUP_BYTES / 2;
    assert!(
        budget < 4 * FRAMED_GROUP_BYTES,
        "premise: four groups exceed the budget"
    );
    let loader = SparseCellSetLoader::new(
        vec![open(&p0)],
        /*cache_shards*/ 16,
        Some(budget),
        /*lookahead*/ 4,
        /*remap*/ None,
        /*n_global_genes*/ None,
        false,
        false,
        0.0,
        /*downsample*/ None,
        /*scatter_block_index*/ true,
        /*max_plan_rows*/ None,
    )
    .unwrap();
    assert!(
        !loader
            .engine
            .lease(0)
            .unwrap()
            .block_index_eligible(0, rows.len()),
        "premise: dense at plan level"
    );
    assert!(
        loader.engine.lease(0).unwrap().block_index_eligible(0, 4),
        "premise: sparse per set"
    );
    let plan = SparseCellSetPlan {
        file_ids: vec![0; 16],
        rows: rows.clone(),
        role_tags: vec![0; 16],
        set_offsets: vec![0, 4, 8, 12, 16],
    };
    let batches: Vec<_> = StdArc::clone(&loader)
        .iter_with_plans(vec![Ok(plan.clone())].into_iter(), /*lookahead*/ 0)
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(batches.len(), 1);
    for (j, &row) in rows.iter().enumerate() {
        let (idx, dat) = batch_row(&batches[0], j);
        let (ecol, eval) = framed_expected(row);
        assert_eq!((idx, dat), (&[ecol][..], &[eval][..]), "row {j}");
    }
    let m = loader.cache_metrics();
    assert_eq!(
        m.block_index_groups.load(AtomicOrdering::Relaxed),
        0,
        "the batch executor reads the plan in one call, so the shard is dense for \
         the gather too and the full-shard route is taken"
    );
    assert_eq!(m.full_shard_groups.load(AtomicOrdering::Relaxed), 1);
    assert_eq!(
        m.row_group_bytes_inserted.load(AtomicOrdering::Relaxed),
        0,
        "the plan is over budget and retains nothing"
    );
    assert_eq!(m.row_group_hits.load(AtomicOrdering::Relaxed), 0);

    // --- the `SCX_CELLSET_EXECUTOR=set` arm, on its own cold cache ----------
    //
    // A second loader, because the first arm left the shard resident and
    // `block_index_eligible` tests `!cached`. The verdict is computed the way
    // `gather` computes it, so the two arms are admitted identically and only
    // the executor differs.
    let per_set_loader = SparseCellSetLoader::new(
        vec![open(&p0)],
        16,
        Some(budget),
        4,
        None,
        None,
        false,
        false,
        0.0,
        None,
        true,
        None,
    )
    .unwrap();
    let buckets = per_set_loader
        .engine
        .bucket_plan_rows(plan.file_ids.iter().copied().zip(plan.rows.iter().copied()));
    let (planned, share) = per_set_loader
        .engine
        .plan_footprint(&buckets, None)
        .unwrap();
    assert!(planned > share, "premise: the plan is over its share");
    let per_set = per_set_loader
        .gather_per_set(
            &per_set_loader.engine,
            &plan,
            Some(crate::sparse_cellset::Admit::from(false)),
        )
        .unwrap();
    assert_eq!(
        per_set.indptr, batches[0].indptr,
        "same bytes, either executor"
    );
    assert_eq!(per_set.indices, batches[0].indices);
    assert_eq!(per_set.data, batches[0].data);
    let pm = per_set_loader.cache_metrics();
    assert_eq!(
        pm.block_index_groups.load(AtomicOrdering::Relaxed),
        4,
        "the per-set walk gathers four rows at a time, so the shard reads as sparse"
    );
    assert_eq!(
        pm.row_group_bytes_inserted.load(AtomicOrdering::Relaxed),
        0,
        "four sets, sparse each, dense together: the plan is over budget and retains nothing"
    );
    assert_eq!(pm.row_group_hits.load(AtomicOrdering::Relaxed), 0);
}

/// **Pin (9b).** `SparseCellSetLoader::new` must pass `scatter_block_index`
/// through to the engine rather than hard-coding either value.
///
/// `plan_engine_tests::from_scx_readers_applies_the_block_index_gate_to_every_reader`
/// pins the engine end; nothing pinned this hop, so a `new` that dropped the
/// argument and passed a literal would have stayed green. Multi-file on purpose
/// — that is the shape the cell-set loader is for, and it is the shape in which
/// a partially-applied gate hides.
///
/// The fixture must be **framed**: `block_index_eligible` requires
/// `shard_is_framed`, so on this module's default `write_fixture` output both
/// arms take the full-shard path and the test would pass vacuously. Hence the
/// borrowed `write_framed_fixture` (256 rows × 4 shards, row groups of 16).
#[test]
fn loader_threads_the_block_index_gate_into_the_engine() {
    use crate::plan_engine::tests::{framed_expected, write_framed_fixture};

    for gate in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let p0 = dir.path().join("f0.scx");
        let p1 = dir.path().join("f1.scx");
        write_framed_fixture(&p0);
        write_framed_fixture(&p1);

        let loader = SparseCellSetLoader::new(
            vec![open(&p0), open(&p1)],
            /*cache_shards*/ 16,
            None,
            /*lookahead*/ 4,
            /*remap*/ None,
            /*n_global_genes*/ None,
            false,
            false,
            0.0,
            /*downsample*/ None,
            gate,
            /*max_plan_rows*/ None,
        )
        .unwrap();

        // One row in each of shards 0..3 (64 rows/shard) of both files, so every
        // touched group is cost-eligible (`group_len * 4 < 64`).
        let rows = [5u64, 70, 140, 200];
        let plan = SparseCellSetPlan {
            file_ids: vec![0, 0, 0, 0, 1, 1, 1, 1],
            rows: rows.iter().chain(rows.iter()).copied().collect(),
            role_tags: vec![0, 0, 0, 0, 1, 1, 1, 1],
            set_offsets: vec![0, 4, 8],
        };
        let batches: Vec<_> = StdArc::clone(&loader)
            .iter_with_plans(vec![Ok(plan.clone())].into_iter(), 4)
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(batches.len(), 1);
        for (j, &row) in plan.rows.iter().enumerate() {
            let (idx, dat) = batch_row(&batches[0], j);
            let (ecol, eval) = framed_expected(row);
            assert_eq!(
                (idx, dat),
                (&[ecol][..], &[eval][..]),
                "gate={gate} row {j}"
            );
        }

        let m = loader.cache_metrics();
        use std::sync::atomic::Ordering as AtomicOrdering;
        let block_index = m.block_index_groups.load(AtomicOrdering::Relaxed);
        let full_shard = m.full_shard_groups.load(AtomicOrdering::Relaxed);
        if gate {
            assert!(
                block_index > 0 && full_shard == 0,
                "gate=true must reach the row-group path \
                 (block_index={block_index}, full_shard={full_shard})"
            );
        } else {
            assert!(
                full_shard > 0 && block_index == 0,
                "gate=false must warm whole shards into the LRU \
                 (block_index={block_index}, full_shard={full_shard})"
            );
        }
    }
}

#[test]
fn gather_single_file_sets_matches_reference_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("f0.scx");
    let p1 = dir.path().join("f1.scx");
    write_fixture(&p0, 32, 8, 4);
    write_fixture(&p1, 32, 8, 4);

    let loader = SparseCellSetLoader::new(
        vec![open(&p0), open(&p1)],
        /*cache_shards*/ 8,
        None,
        /*lookahead*/ 4,
        /*remap*/ None,
        /*n_global_genes*/ None,
        false,
        false,
        0.0,
        /*downsample*/ None,
        /*scatter_block_index*/ false,
        /*max_plan_rows*/ None,
    )
    .unwrap();

    // Two single-file sets: scattered (file 0), duplicate (file 1).
    let plan = SparseCellSetPlan {
        file_ids: vec![0, 0, 0, 1, 1, 1],
        rows: vec![5, 3, 0, 7, 7, 31],
        role_tags: vec![0, 0, 0, 1, 1, 1],
        set_offsets: vec![0, 3, 6],
    };
    let batches: Vec<_> = loader
        .iter_with_plans(vec![Ok(plan.clone())].into_iter(), 4)
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(batches.len(), 1);
    let b = &batches[0];

    assert_eq!(b.shape, (6, 8));
    assert_eq!(b.set_offsets, vec![0, 3, 6]);
    assert_eq!(b.cell_indices, plan.rows);
    assert_eq!(b.file_ids, plan.file_ids);
    assert_eq!(b.role_tags, plan.role_tags);
    assert_eq!(b.indptr.len(), 7);

    // Each output row matches the deterministic fixture value, in plan order.
    for (j, (&fid, &row)) in plan.file_ids.iter().zip(plan.rows.iter()).enumerate() {
        let (idx, dat) = batch_row(b, j);
        let (ecol, eval) = expected_cell(row, 8);
        assert_eq!(idx, &[ecol], "row {j} (file {fid} row {row}) indices");
        assert_eq!(dat, &[eval], "row {j} value");
    }

    // Cross-check against read_row_indices for the file-1 duplicate set.
    let r1 = BackedCsrReader::new(open(&p1), 0);
    let ref_csr = r1.read_row_indices(&[7, 7, 31]).unwrap();
    for k in 0..3 {
        let (idx, dat) = batch_row(b, 3 + k);
        let lo = ref_csr.indptr[k] as usize;
        let hi = ref_csr.indptr[k + 1] as usize;
        assert_eq!(idx, &ref_csr.indices[lo..hi]);
        assert_eq!(dat, &ref_csr.data[lo..hi]);
    }
}

#[test]
fn empty_set_keeps_boundary_without_rows() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("f0.scx");
    write_fixture(&p0, 16, 8, 2);
    let loader = SparseCellSetLoader::new(
        vec![open(&p0)],
        4,
        None,
        2,
        None,
        None,
        false,
        false,
        0.0,
        None,
        /*scatter_block_index*/ false,
        /*max_plan_rows*/ None,
    )
    .unwrap();
    // set 0: two rows; set 1: empty; set 2: one row.
    let plan = SparseCellSetPlan {
        file_ids: vec![0, 0, 0],
        rows: vec![1, 2, 9],
        role_tags: vec![0, 0, 2],
        set_offsets: vec![0, 2, 2, 3],
    };
    let b = loader
        .iter_with_plans(vec![Ok(plan)].into_iter(), 2)
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(b.set_offsets, vec![0, 2, 2, 3]);
    assert_eq!(b.cell_indices, vec![1, 2, 9]);
    assert_eq!(b.shape.0, 3);
}

#[test]
fn remap_row_maps_drops_sentinels_and_coalesces() {
    // cols 0→10, 1→(-1 drop), 2→20, 3→10 (duplicate of col0's global).
    let table = vec![10, -1, 20, 10];
    let (mut pairs, mut idx, mut dat) = (Vec::new(), Vec::new(), Vec::new());
    remap_row_into(
        &[0, 1, 2, 3],
        &[1.0, 9.0, 2.0, 4.0],
        &table,
        &mut pairs,
        &mut idx,
        &mut dat,
    );
    // global 10 gets col0 (1.0) + col3 (4.0) = 5.0; col1 dropped; global 20 = 2.0.
    assert_eq!(idx, vec![10, 20]);
    assert_eq!(dat, vec![5.0, 2.0]);
}

#[test]
fn gather_cross_file_set_concatenates_in_global_space() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("f0.scx");
    let p1 = dir.path().join("f1.scx");
    write_fixture(&p0, 32, 8, 4);
    write_fixture(&p1, 32, 8, 4);

    // file 0: local g → global g (0..8); file 1: local g → global 100+g.
    let remap = vec![
        (0..8).collect::<Vec<i32>>(),
        (0..8).map(|g| 100 + g).collect(),
    ];
    let loader = SparseCellSetLoader::new(
        vec![open(&p0), open(&p1)],
        8,
        None,
        4,
        Some(remap),
        /*n_global_genes*/ Some(108),
        false,
        false,
        0.0,
        /*downsample*/ None,
        /*scatter_block_index*/ false,
        /*max_plan_rows*/ None,
    )
    .unwrap();

    // One cross-file set: file 0 row 5 (col5) + file 1 row 7 (col7).
    let plan = SparseCellSetPlan {
        file_ids: vec![0, 1],
        rows: vec![5, 7],
        role_tags: vec![0, 0],
        set_offsets: vec![0, 2],
    };
    let b = loader
        .iter_with_plans(vec![Ok(plan)].into_iter(), 4)
        .next()
        .unwrap()
        .unwrap();
    assert_eq!(b.shape, (2, 108));
    // row 0: file0 col5 → global 5; row 1: file1 col7 → global 107.
    assert_eq!(batch_row(&b, 0).0, &[5]);
    assert_eq!(batch_row(&b, 1).0, &[107]);
    assert_eq!(b.file_ids, vec![0, 1]);
}

#[test]
fn sparse_transforms_match_dense_reference() {
    let mut sparse_data = vec![2.0f32, 4.0, 4.0];
    apply_sparse_transforms(&mut sparse_data, true, true, 20.0);

    // Dense reference: a row with the same nonzeros at arbitrary positions.
    let mut dense = vec![0.0f32; 8];
    dense[1] = 2.0;
    dense[4] = 4.0;
    dense[6] = 4.0;
    crate::normalize::normalize_dense_row(&mut dense, 20.0);
    crate::normalize::log1p_dense_row(&mut dense);
    let dense_nonzero = vec![dense[1], dense[4], dense[6]];

    // Delegating to the canonical dense helpers makes this bit-identical, not
    // just within tolerance (SCX-DATA-LOADER §0).
    assert_eq!(sparse_data, dense_nonzero);
}

/// Build a single-file loader over a small fixture for malformed-plan tests.
fn malformed_plan_loader(dir: &std::path::Path) -> StdArc<SparseCellSetLoader> {
    let p0 = dir.join("f0.scx");
    write_fixture(&p0, 16, 8, 2);
    SparseCellSetLoader::new(
        vec![open(&p0)],
        4,
        None,
        2,
        None,
        None,
        false,
        false,
        0.0,
        None,
        /*scatter_block_index*/ false,
        /*max_plan_rows*/ None,
    )
    .unwrap()
}

/// Run one plan and return the first batch's `Result`.
fn run_one(
    loader: StdArc<SparseCellSetLoader>,
    plan: SparseCellSetPlan,
) -> Result<SparseCellSetBatch> {
    loader
        .iter_with_plans(vec![Ok(plan)].into_iter(), 2)
        .next()
        .unwrap()
}

#[test]
fn collate_gathered_emits_stacked_tensors_matching_kernel() {
    use crate::sparse_cellset_collate::PreprocessMode;
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("f0.scx");
    write_fixture(&p0, 32, 8, 4); // row r: col r%8, val (r+1)&0xFF
                                  // Identity remap (local g → global g) so gather emits global CSR.
    let loader = SparseCellSetLoader::new(
        vec![open(&p0)],
        8,
        None,
        4,
        Some(vec![(0..8).collect::<Vec<i32>>()]),
        Some(8),
        false,
        false,
        0.0,
        /*downsample*/ None,
        /*scatter_block_index*/ false,
        /*max_plan_rows*/ None,
    )
    .unwrap();

    // Gather one set, two cells: row 2 (col2,val3), row 5 (col5,val6) → global CSR,
    // then collate it (the state3 flow: Python gathers, then collate_gathered).
    let g = loader
        .iter_with_plans(
            vec![Ok(SparseCellSetPlan {
                file_ids: vec![0, 0],
                rows: vec![2, 5],
                role_tags: vec![0, 0],
                set_offsets: vec![0, 2],
            })]
            .into_iter(),
            4,
        )
        .next()
        .unwrap()
        .unwrap();

    let scalars = CollateScalars {
        k_enc: 4,
        mode: PreprocessMode::Log1pRaw,
        target_sum: 1e4,
        pflog_alpha: None,
        n_genes_total: 8,
        lib_size_redef: false,
    };
    let b = collate_gathered(
        &g.indptr,
        &g.indices,
        &g.data,
        &g.set_offsets,
        g.cell_indices,
        g.file_ids,
        g.role_tags,
        /*k_dec*/ 4,
        &[2, 5, 7, 0], // query (shared across the set)
        None,          // query_offsets: per-set addressing (contract v2 shape)
        &[],           // enc_mask_positions: pert-style (no encoder masking)
        &[0, 0],       // hide_readout
        &[8],          // n_measured
        &scalars,
    )
    .unwrap();

    assert_eq!(b.n_rows, 2);
    assert_eq!((b.k_enc, b.k_dec), (4, 4));
    assert_eq!(b.encoder_gene_ids.len(), 2 * 4);
    assert_eq!(b.target_counts.len(), 2 * 4);
    // cell 0 (row2): gene2 val3. encoder slot0 = gene2, rest PAD(=9).
    assert_eq!(&b.encoder_gene_ids[0..4], &[2, 9, 9, 9]);
    assert!((b.encoder_counts[0] - 3.0f32.ln_1p()).abs() < 1e-6);
    // target gather over query [2,5,7,0]: cell0 has gene2=3 → [3,0,0,0].
    assert_eq!(&b.target_counts[0..4], &[3.0, 0.0, 0.0, 0.0]);
    assert_eq!(b.library_size[0], 3.0);
    // cell 1 (row5): gene5 val6 → query position 1.
    assert_eq!(&b.encoder_gene_ids[4..8], &[5, 9, 9, 9]);
    assert_eq!(&b.target_counts[4..8], &[0.0, 6.0, 0.0, 0.0]);
    assert_eq!(b.library_size[1], 6.0);
}

/// Two rows of ONE set, one shared query panel, different withheld genes.
///
/// The per-set `query` and the per-row `enc_mask_positions` are sliced on
/// different axes (`sparse_cellset.rs`'s rayon row loop takes `query` at
/// `s * k_dec` and the mask at `r * k_dec`), which is exactly why the withheld
/// set may not be hoisted out of the row loop: hoisting applies row 0's bits to
/// every row of the set. Nothing covered that — the one pre-existing
/// `collate_gathered` test passes an empty mask, and every masking test in
/// `sparse_cellset_collate_tests.rs` drives a single cell through
/// `collate_cell`, where a one-row set cannot tell the two axes apart.
///
/// Built from synthetic CSR rather than a real gather because the shared
/// fixture writes one non-zero per row, and a one-gene cell has no top-K to
/// withhold from.
#[test]
fn collate_gathered_applies_each_rows_own_mask_within_a_set() {
    use crate::sparse_cellset_collate::PreprocessMode;

    // Two identical cells: genes [1,3,5] with counts [3,2,1] (already rank
    // order, so `order` is [1,3,5] and `take` is the whole row at k_enc=3).
    let indptr = [0i64, 3, 6];
    let indices = [1i32, 3, 5, 1, 3, 5];
    let data = [3.0f32, 2.0, 1.0, 3.0, 2.0, 1.0];
    let set_offsets = [0i64, 2]; // ONE set spanning both rows

    let scalars = CollateScalars {
        k_enc: 3,
        // PassThrough so encoder counts are the raw values and compare exactly.
        mode: PreprocessMode::PassThrough,
        target_sum: 1e4,
        pflog_alpha: None,
        n_genes_total: 8, // ⇒ GENE_MASK = 8, PAD = 9
        lib_size_redef: false,
    };
    let b = collate_gathered(
        &indptr,
        &indices,
        &data,
        &set_offsets,
        vec![0u64, 1],
        vec![0u32, 0],
        vec![0i32, 0],
        /*k_dec*/ 3,
        /*query, per SET*/ &[1, 3, 5],
        /*query_offsets*/ None,
        // Per ROW, k_dec each: row 0 withholds gene 1, row 1 withholds gene 5.
        /*enc_mask_positions*/
        &[1, 0, 0, 0, 0, 1],
        &[0, 0],
        &[3],
        &scalars,
    )
    .unwrap();

    // Row 0 drops gene 1, survivors compact left.
    assert_eq!(&b.encoder_gene_ids[0..3], &[3, 5, 9]);
    assert_eq!(&b.encoder_counts[0..3], &[2.0, 1.0, 0.0]);
    // Row 1 drops gene 5 — NOT gene 1. This is the assertion a hoist breaks.
    assert_eq!(&b.encoder_gene_ids[3..6], &[1, 3, 9]);
    assert_eq!(&b.encoder_counts[3..6], &[3.0, 2.0, 0.0]);
    // Neither row is all-masked, so no GENE_MASK token and no set expr mask.
    assert_eq!(&b.encoder_mask[0..6], &[0, 0, 0, 0, 0, 0]);
    assert_eq!(&b.encoder_pad_mask[0..6], &[0, 0, 1, 0, 0, 1]);
}

#[test]
fn malformed_plan_file_id_out_of_range_returns_error_not_panic() {
    let dir = tempfile::tempdir().unwrap();
    let loader = malformed_plan_loader(dir.path());
    // file_id 9 but only 1 file.
    let plan = SparseCellSetPlan {
        file_ids: vec![0, 9],
        rows: vec![1, 2],
        role_tags: vec![0, 0],
        set_offsets: vec![0, 2],
    };
    assert!(matches!(
        run_one(loader, plan),
        Err(LoaderError::ConfigError { .. })
    ));
}

#[test]
fn malformed_plan_set_offsets_out_of_bounds_returns_error_not_panic() {
    let dir = tempfile::tempdir().unwrap();
    let loader = malformed_plan_loader(dir.path());
    // set_offsets[1]=5 > total_rows=2 would panic the slice without validation.
    let plan = SparseCellSetPlan {
        file_ids: vec![0, 0],
        rows: vec![1, 2],
        role_tags: vec![0, 0],
        set_offsets: vec![0, 5],
    };
    assert!(matches!(
        run_one(loader, plan),
        Err(LoaderError::ConfigError { .. })
    ));
}

#[test]
fn malformed_plan_set_offsets_non_monotonic_returns_error_not_panic() {
    let dir = tempfile::tempdir().unwrap();
    let loader = malformed_plan_loader(dir.path());
    let plan = SparseCellSetPlan {
        file_ids: vec![0, 0, 0],
        rows: vec![1, 2, 3],
        role_tags: vec![0, 0, 0],
        set_offsets: vec![0, 2, 1], // decreasing
    };
    assert!(matches!(
        run_one(loader, plan),
        Err(LoaderError::ConfigError { .. })
    ));
}

#[test]
fn malformed_plan_row_out_of_range_raises_index_out_of_range() {
    let dir = tempfile::tempdir().unwrap();
    let loader = malformed_plan_loader(dir.path());
    // Fixture has 16 rows; row 99 is out of range → IndexError on the PyO3 side.
    let plan = SparseCellSetPlan {
        file_ids: vec![0, 0],
        rows: vec![1, 99],
        role_tags: vec![0, 0],
        set_offsets: vec![0, 2],
    };
    assert!(matches!(
        run_one(loader, plan),
        Err(LoaderError::IndexOutOfRange { idx: 99, n_obs: 16 })
    ));
}

// ---------------------------------------------------------------------
// 1A — cache sizing / budget policy on the sparse path
// ---------------------------------------------------------------------

/// Build a two-file loader with an explicit byte budget.
fn budget_loader(
    dir: &std::path::Path,
    cache_shards: usize,
    bytes_budget: Option<usize>,
) -> StdArc<SparseCellSetLoader> {
    let p0 = dir.join("b0.scx");
    let p1 = dir.join("b1.scx");
    write_fixture(&p0, 32, 8, 4);
    write_fixture(&p1, 32, 8, 4);
    SparseCellSetLoader::new(
        vec![open(&p0), open(&p1)],
        cache_shards,
        bytes_budget,
        4,
        None,
        None,
        false,
        false,
        0.0,
        /*downsample*/ None,
        /*scatter_block_index*/ false,
        /*max_plan_rows*/ None,
    )
    .unwrap()
}

/// `bytes_budget=None` must resolve to a *bounded* adaptive budget, not the
/// historical `usize::MAX`. The unbounded default is what let STATE3 reach
/// ~23 GB RSS with no ceiling anywhere in the stack.
#[test]
fn none_budget_resolves_to_a_bounded_adaptive_value() {
    let dir = tempfile::tempdir().unwrap();
    let loader = budget_loader(dir.path(), 128, None);
    let budget = loader.cache_bytes_budget();
    assert!(budget < usize::MAX, "the budget must be bounded");
    // Small fixture ⇒ need is far below the floor, so the floor wins.
    assert_eq!(
        budget,
        crate::pipeline::LoaderConfig::default().max_memory_mb * 1024 * 1024,
        "a tiny file must keep the floor, never be tightened below it"
    );
    assert!(
        loader.cache_sizing().is_none(),
        "the floor comfortably holds 128 tiny shards; nothing to warn about"
    );
}

/// An explicit budget too small for the requested cache must be reported.
///
/// **Rewritten by ORG-9.10-5.** It used to run on a 1 KB budget over 128 B
/// shards and assert exactly 8 affordable shards — arithmetic that only worked
/// while the tuner ignored the 50 MB interpreter constant it reported. A 1 KB
/// *process* budget affords no cache at all, so the case now runs on a budget
/// that is small relative to the request but not absurd in absolute terms,
/// which is the regime the diagnostic is actually for.
#[test]
fn explicit_tight_budget_reports_a_sizing_verdict() {
    let dir = tempfile::tempdir().unwrap();
    // 32 KB shards; 64 MiB budget − 50 MiB interpreter = 448 shards affordable.
    let loader = sized_budget_loader(dir.path(), 4096, Some(64 * 1024 * 1024));
    let v = loader
        .cache_sizing()
        .expect("a budget that cannot hold the request must be reported");
    assert_eq!(v.requested_cache_shards, 4096);
    assert_eq!(v.effective_cache_shards, 448);
    assert!(!v.below_floor, "448 is far above MIN_CACHE_SHARDS");
    assert_eq!(v.shard_decoded_bytes, 32_768);
    assert!(
        loader
            .budget_breakdown()
            .fits_within_bytes(64 * 1024 * 1024),
        "the reported total must fit the budget the tuner checked"
    );
}

/// Below MIN_CACHE_SHARDS the verdict escalates, so the warning can say the
/// stronger thing.
#[test]
fn very_tight_budget_flags_below_floor() {
    let dir = tempfile::tempdir().unwrap();
    // 50 MiB (the interpreter constant) + room for exactly 4 of the 32 KB shards.
    let budget = 50 * 1024 * 1024 + 4 * 32_768;
    let loader = sized_budget_loader(dir.path(), 4096, Some(budget));
    let v = loader.cache_sizing().expect("must be reported");
    assert_eq!(v.effective_cache_shards, 4);
    assert!(v.below_floor);
}

/// An explicit budget that comfortably holds the request stays silent — the
/// anti-tautology partner for the two tests above.
#[test]
fn generous_explicit_budget_is_silent() {
    let dir = tempfile::tempdir().unwrap();
    let loader = budget_loader(dir.path(), 8, Some(64 * 1024 * 1024));
    assert!(loader.cache_sizing().is_none());
}

/// `plan_shard_touch_count` must group by `file_id` before counting, since shard
/// indices are per-file: the same shard index in two files is two entries in the
/// shared cache, which is keyed `(file_id, shard)`.
#[test]
fn plan_shard_touch_count_is_per_file() {
    let dir = tempfile::tempdir().unwrap();
    let loader = budget_loader(dir.path(), 8, None);
    // 32 rows over 4 shards ⇒ 8 rows/shard.

    // One shard, one file.
    assert_eq!(loader.plan_shard_touch_count(&[0, 0], &[0, 7]), 1);
    // Shard 0 of file 0 and shard 0 of file 1 are distinct cache entries.
    assert_eq!(
        loader.plan_shard_touch_count(&[0, 1], &[0, 0]),
        2,
        "same shard index in different files must count twice"
    );
    // All four shards of one file.
    assert_eq!(
        loader.plan_shard_touch_count(&[0, 0, 0, 0], &[0, 8, 16, 24]),
        4
    );
    // Both files, all shards.
    assert_eq!(
        loader.plan_shard_touch_count(&[0, 0, 0, 0, 1, 1, 1, 1], &[0, 8, 16, 24, 0, 8, 16, 24]),
        8
    );
    // Duplicate rows collapse.
    assert_eq!(loader.plan_shard_touch_count(&[0, 0, 0], &[3, 3, 3]), 1);
    assert_eq!(loader.plan_shard_touch_count(&[], &[]), 0);
}

/// An out-of-range `file_id` must not panic — the helper is called on
/// caller-supplied plans before any validation has run.
#[test]
fn plan_shard_touch_count_ignores_unknown_file_ids() {
    let dir = tempfile::tempdir().unwrap();
    let loader = budget_loader(dir.path(), 8, None);
    assert_eq!(loader.plan_shard_touch_count(&[9], &[0]), 0);
    assert_eq!(loader.plan_shard_touch_count(&[0, 9], &[0, 0]), 1);
}

/// The thrash diagnostic must be phrased against the **binding** constraint.
///
/// Round-1 review (Cursor, P2): the sparse path sampled with the *requested*
/// `cache_shards` while `IndexPlanDataset` sampled with the post-auto-tune value.
/// On the regime this loader targets (~470 MB shards, adaptive 4 GB budget) the
/// byte cap binds long before the count does, so a warning naming `cache_shards`
/// sends the caller to raise a knob that cannot help.
#[test]
fn effective_cache_shards_reports_the_byte_cap_when_it_binds() {
    let dir = tempfile::tempdir().unwrap();
    // 32 KB shards; a 64 MiB budget affords 448 after the interpreter constant,
    // so a 4096-count request is bound by BYTES, not by the count.
    let loader = sized_budget_loader(dir.path(), 4096, Some(64 * 1024 * 1024));
    assert_eq!(loader.cache_shards(), 4096, "the request is unchanged");
    assert_eq!(
        loader.effective_cache_shards(),
        448,
        "the byte cap is what actually binds"
    );
    assert!(
        loader.effective_cache_shards() < loader.cache_shards(),
        "premise: this fixture must be byte-bound, else the test proves nothing"
    );
}

/// With a generous budget the count cap binds and the two agree.
#[test]
fn effective_cache_shards_equals_request_when_the_count_binds() {
    let dir = tempfile::tempdir().unwrap();
    let loader = budget_loader(dir.path(), 8, Some(64 * 1024 * 1024));
    assert_eq!(loader.effective_cache_shards(), loader.cache_shards());
}

/// A zero byte budget bottoms the cache out at **one** shard, not zero.
///
/// **Inverted by ORG-9.10-5**, deliberately: the shared descent floors at 1 for
/// the same reason `IndexPlanLoader` refuses below 1 — a cache that can hold
/// nothing re-decodes every shard of every batch, and `BackedCsrReader` applies
/// its own `cache_shards.max(2)` to a 0 anyway, so reporting 0 described a
/// cache that never existed. The budget is still hopeless and still says so:
/// the verdict is `below_floor`, which is what the warning escalates on.
///
/// Round-2 review (Cursor, P2) caught that this test was previously named
/// `..._falls_back_to_the_count_when_shard_size_is_unknown` while asserting
/// `shard_decoded_bytes == 128` — i.e. the known-size case, the opposite of its
/// name. The unknown-size fallback is covered by
/// `avg_shard_decoded_bytes_returns_zero_when_no_shard_has_stats` +
/// `effective_cache_shards_falls_back_to_the_count_when_size_is_unknown` below.
#[test]
fn effective_cache_shards_floors_at_one_under_a_zero_budget() {
    let dir = tempfile::tempdir().unwrap();
    let loader = budget_loader(dir.path(), 8, Some(0));
    assert_eq!(
        loader.shard_decoded_bytes(),
        128,
        "premise: the shard size is KNOWN here"
    );
    assert_eq!(loader.effective_cache_shards(), 1);
    assert!(
        loader
            .cache_sizing()
            .expect("a zero budget must be reported")
            .below_floor,
        "a hopeless budget must still escalate, not be silently floored"
    );
}

/// Stat-less shards must not be averaged into the per-shard size.
///
/// Round-2 review, flagged independently by all three reviewers: counting a
/// shard with no catalog stats in the divisor averages its 0 bytes in, so the
/// per-shard estimate comes out low, the affordable count comes out high, and the
/// sizing diagnostic under-warns on precisely the files with incomplete catalogs.
#[test]
fn avg_shard_decoded_bytes_ignores_stat_less_shards_in_the_divisor() {
    // `write_fixture` emits stats for every shard, so build the arithmetic
    // directly against the helper's contract: N shards with stats summing to S
    // bytes must average S/N, independent of how many stat-less shards exist.
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("s0.scx");
    write_fixture(&p0, 32, 8, 4);
    let one = avg_shard_decoded_bytes(&[open(&p0)]);
    assert_eq!(one.0, 128, "8 rows x 1 nnz => (8*8 + 8*8)/1 per shard");

    // Two identical files: twice the shards, twice the totals, same average.
    let p1 = dir.path().join("s1.scx");
    write_fixture(&p1, 32, 8, 4);
    let two = avg_shard_decoded_bytes(&[open(&p0), open(&p1)]);
    assert_eq!(
        two.0, one.0,
        "the average must be scale-invariant; a drift here means the divisor \
         and the numerator are counting different shard sets"
    );

    // The mean density rides the same walk and the same divisor rule, but is
    // per ROW: `write_fixture` gives every row exactly one non-zero, so shards
    // of unequal height cannot pull it away from 1.0.
    assert_eq!(one.1, 1.0, "8 rows x 1 nnz => 1 nnz/row");
    assert_eq!(two.1, one.1, "density must be scale-invariant too");
}

/// No shard carries stats ⇒ size unknown ⇒ the byte cap says nothing, so the
/// count cap is what binds (never a fabricated average, never 0 entries).
#[test]
fn effective_cache_shards_falls_back_to_the_count_when_size_is_unknown() {
    // An empty reader set is the degenerate "no shards carry stats" case the
    // helper must survive; `SparseCellSetLoader::new` rejects zero files, so the
    // helper is exercised directly.
    assert_eq!(avg_shard_decoded_bytes(&[]), (0, 0.0));
}

/// The **byte** half of that fallback, which the test above never covered.
///
/// ORG-9.10-5 began handing the engine the tuned cache bytes instead of the raw
/// budget. On a stats-less file the model's cache term is `n × 0 = 0`, so the
/// tightening would cap the shared cache at zero bytes — "we cannot size the
/// cache" turned into "no cache", strictly worse than the behaviour it
/// replaced. No writer emits a stats-less CSR entry, so this is asserted on the
/// decision rather than through a file that cannot be built.
#[test]
fn an_unknown_shard_size_keeps_the_raw_byte_budget() {
    const RAW: usize = 512 * 1024 * 1024;

    // Known size: the tuned cache binds, as on the paired loader.
    assert_eq!(
        resolve_enforced_cache_bytes(32_768, 14_680_064, RAW),
        14_680_064
    );

    // Unknown size: the model says 0 bytes, which must NOT reach the cache.
    assert_eq!(resolve_enforced_cache_bytes(0, 0, RAW), RAW);
    assert_ne!(
        resolve_enforced_cache_bytes(0, 0, RAW),
        0,
        "a stats-less file must not end up with a zero-byte shard cache"
    );
}

// ---- ORG-9.10-5 pre-refactor pins ------------------------------------

/// Two files whose shards are large enough that the cache term dominates the
/// 50 MB interpreter constant — the regime where the sparse budget model's
/// answer actually differs from the paired loader's.
fn sized_budget_loader(
    dir: &std::path::Path,
    cache_shards: usize,
    bytes_budget: Option<usize>,
) -> StdArc<SparseCellSetLoader> {
    let p0 = dir.join("s0.scx");
    let p1 = dir.join("s1.scx");
    // 2048 rows/shard x 16 B/row = 32 KB per shard.
    write_fixture(&p0, 8192, 64, 4);
    write_fixture(&p1, 8192, 64, 4);
    SparseCellSetLoader::new(
        vec![open(&p0), open(&p1)],
        cache_shards,
        bytes_budget,
        4,
        None,
        None,
        false,
        false,
        0.0,
        /*downsample*/ None,
        /*scatter_block_index*/ false,
        /*max_plan_rows*/ None,
    )
    .unwrap()
}

/// A loader built with an explicit `max_plan_rows` charges one gathered batch.
fn batch_charged_loader(
    dir: &std::path::Path,
    cache_shards: usize,
    bytes_budget: Option<usize>,
    max_plan_rows: Option<usize>,
) -> StdArc<SparseCellSetLoader> {
    let p0 = dir.join("b0.scx");
    write_fixture(&p0, 8192, 64, 4);
    SparseCellSetLoader::new(
        vec![open(&p0)],
        cache_shards,
        bytes_budget,
        4,
        None,
        None,
        false,
        false,
        0.0,
        /*downsample*/ None,
        /*scatter_block_index*/ false,
        max_plan_rows,
    )
    .unwrap()
}

/// `max_plan_rows` is what turns the batch charge on; without it, nothing moves.
///
/// The default has to be byte-identical rather than merely "small": this class
/// has no `max_plan_size`, so any default would be a guess about plan width, and
/// on a file where the byte budget already binds a guess spends cache — the
/// lever `cache_shards` 16 -> 31 measured at 2,486x on census_500k — to buy an
/// estimate the caller never asked for.
#[test]
fn the_batch_is_uncharged_until_max_plan_rows_is_declared() {
    let dir = tempfile::tempdir().unwrap();
    let budget = Some(50 * 1024 * 1024 + 64 * 32_768);

    let plain = batch_charged_loader(dir.path(), 64, budget, None);
    assert_eq!(
        plain.budget_breakdown().batch_buffer_bytes,
        0,
        "an undeclared plan width must cost nothing"
    );
    assert_eq!(plain.max_plan_rows(), None);

    // `write_fixture` gives one non-zero per row, so a 4096-row plan is
    // 4096 nnz x 8 B + 4097 x 8 B of indptr.
    let declared = batch_charged_loader(dir.path(), 64, budget, Some(4096));
    assert_eq!(declared.mean_nnz_per_row(), 1.0);
    // Against the pre-size POLICY, not the raw mean: the gather allocates
    // `presize_nnz`, so a charge computed from the unbiased estimate would
    // under-report the batch by the eighth the bias adds. Restating the
    // arithmetic here instead would let the two drift apart silently, which is
    // the same trap the capacity assertions fell into.
    assert_eq!(
        declared.budget_breakdown().batch_buffer_bytes,
        crate::sparse_cellset::presize_nnz(4096, 1.0) * 8 + 4097 * 8
    );
    assert_eq!(declared.max_plan_rows(), Some(4096));

    // Same budget, same shards requested: the charge comes out of the cache.
    assert!(
        declared.effective_cache_shards() < plain.effective_cache_shards(),
        "charging a batch must shrink the affordable cache: {} vs {}",
        declared.effective_cache_shards(),
        plain.effective_cache_shards()
    );

    // ...and monotonically so, which is what makes the term a bound rather
    // than a flag.
    let wider = batch_charged_loader(dir.path(), 64, budget, Some(16384));
    assert!(
        wider.effective_cache_shards() < declared.effective_cache_shards(),
        "a wider plan must cost more cache: {} vs {}",
        wider.effective_cache_shards(),
        declared.effective_cache_shards()
    );
    assert!(
        wider.budget_breakdown().total_bytes <= wider.cache_bytes_budget()
            || wider.budget_exceeded(),
        "the charged breakdown must still fit the budget it was tuned against"
    );
}

/// `max_plan_rows` is an upper bound, so it must actually refuse.
///
/// It sizes the shard cache down as though plans were at most that wide. A
/// value that only ever shrinks the budget while the gather accepts any plan is
/// not a bound — the cache would be sized for 4096 rows and the batch allocated
/// for a million. `IndexPlanLoader` refuses past `max_plan_size` for exactly
/// this reason; this mirrors it.
#[test]
fn a_plan_wider_than_max_plan_rows_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let loader = batch_charged_loader(dir.path(), 64, None, Some(4));

    let plan = |n: usize| SparseCellSetPlan {
        file_ids: vec![0; n],
        rows: (0..n as u64).collect(),
        role_tags: vec![0; n],
        set_offsets: vec![0, n as i64],
    };

    // At the bound: accepted.
    assert!(run_one(StdArc::clone(&loader), plan(4)).is_ok());
    // Past it: refused, and the message names the knob and the sizing.
    let err = run_one(StdArc::clone(&loader), plan(5)).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("max_plan_rows"), "{msg}");
    assert!(msg.contains('5') && msg.contains('4'), "{msg}");

    // An undeclared bound refuses nothing — the default must stay inert.
    let unbounded = batch_charged_loader(dir.path(), 64, None, None);
    assert!(run_one(unbounded, plan(64)).is_ok());

    // The ITERATOR route refuses too, and refuses in the stream: the plan must
    // not reach the prefetcher, which would size it and await shard decodes
    // before the gather rejected it. Review on #535 round 2 (codex).
    let streamed = batch_charged_loader(dir.path(), 64, None, Some(4));
    let mut it = StdArc::clone(&streamed).iter_with_plans(vec![Ok(plan(5))].into_iter(), 4);
    let first = it
        .next()
        .expect("the stream must yield the refusal, not end");
    let msg = first.unwrap_err().to_string();
    assert!(msg.contains("max_plan_rows"), "{msg}");
    // The error alone would also be produced by a LATE refusal, after the
    // prefetcher had sized the plan and awaited decodes — which is the thing
    // the fix is about. Zero cache activity is what distinguishes refusing in
    // the stream from refusing at the end of it. Review on #535 round 3
    // (Cursor Agent).
    drop(it);
    use std::sync::atomic::Ordering as AtomicOrdering;
    let m = streamed.cache_metrics();
    assert_eq!(
        m.misses.load(AtomicOrdering::Relaxed) + m.row_group_misses.load(AtomicOrdering::Relaxed),
        0,
        "an over-wide plan must be refused before the prefetcher touches a shard"
    );
}

/// A loader whose budget was not exceeded must fit the breakdown it reports.
///
/// ⚠️ **Red before `ORG-9.10-5`, by design.** `SparseCellSetLoader::new` sizes
/// the cache by plain division and passes `non_cache_bytes: 0` to
/// `assess_cache_sizing`, so the 50 MB constant it *reports* is not one it
/// budgets for. `IndexPlanLoader` cannot violate this
/// (`a_constructed_loader_fits_the_breakdown_it_reports`) and neither can the
/// sequential path
/// (`pipeline::tests::a_budget_that_was_not_exceeded_fits_the_breakdown_it_reports`).
///
/// The implication is guarded on `budget_exceeded` because exhaustion is not an
/// error here: a budget below the interpreter constant alone bottoms out at the
/// one-shard floor and warns, rather than refusing construction the way the
/// paired loader does.
#[test]
fn an_unexceeded_sparse_budget_fits_the_breakdown_it_reports() {
    let dir = tempfile::tempdir().unwrap();
    let mut saw_nonempty = false;
    for mb in [1usize, 8, 32, 64, 96, 128, 192] {
        let loader = sized_budget_loader(dir.path(), 4096, Some(mb * 1024 * 1024));
        if loader.budget_exceeded() {
            continue;
        }
        saw_nonempty = true;
        assert!(
            loader.budget_breakdown().fits_within(mb),
            "budget {mb} MB was not exceeded but its breakdown totals {} bytes over \
             {} cache shards",
            loader.budget_breakdown().total_bytes,
            loader.effective_cache_shards(),
        );
    }
    assert!(
        saw_nonempty,
        "every budget in the sweep was exceeded, so this proves nothing"
    );
}

/// The closed form must return exactly what the shared driver would.
///
/// `SparseCellSetLoader` resolves its one knob by division rather than by
/// walking `tune`'s descent, because the knob is caller-controlled and the
/// descent is O(cache_shards) — 0.7 s at `cache_shards=1e9`, unbounded at
/// `usize::MAX` (review, round 2). That is only safe while the two agree, so
/// this sweeps budgets across the interesting range and compares them
/// element-for-element. It also covers the boundaries the division gets wrong
/// most easily: an exact fit, a zero budget, and an explicit `cache_shards=0`,
/// which must stay 0 rather than being floored up to 1.
#[test]
fn closed_form_agrees_with_the_shared_driver() {
    const SHARD: usize = 32_768;
    let model = SparseCellSetBudgetModel {
        shard_decoded_bytes: SHARD,
        batch_bytes: 0,
        transient_bytes: 0,
    };
    let py = crate::budget::PYTHON_OVERHEAD_BYTES;
    let budgets = [
        0,
        1,
        py - 1,
        py,
        py + 1,
        py + SHARD - 1,
        py + SHARD,
        py + 4 * SHARD,
        py + 448 * SHARD,
        512 * 1024 * 1024,
        4096 * 1024 * 1024,
    ];
    for requested in [0usize, 1, 8, 128, 4096] {
        let req = SparseCellSetParams {
            cache_shards: requested,
        };
        for &budget in &budgets {
            let closed = resolve_sparse_cache_shards(&model, req, budget);
            let driven = crate::budget::tune(&model, req, budget);
            assert_eq!(
                (closed.params.cache_shards, closed.exhausted),
                (driven.params.cache_shards, driven.exhausted),
                "closed form and driver disagree at requested={requested}, budget={budget}"
            );
            assert_eq!(closed.breakdown.total_bytes, driven.breakdown.total_bytes);
        }
    }
}

/// The whole reduction chain, through the shared harness. Fixture-free.
#[test]
fn the_sparse_reduction_chain_is_monotone() {
    for &shard_decoded_bytes in &[128usize, 32_768, 470 * 1024 * 1024, 0] {
        crate::budget::assert_monotone_reduction_chain(
            &SparseCellSetBudgetModel {
                shard_decoded_bytes,
                batch_bytes: 0,
                transient_bytes: 0,
            },
            SparseCellSetParams { cache_shards: 128 },
        );
    }
}

/// The **model** is monotone: fewer cache shards may not estimate more.
///
/// Read under a budget generous enough that nothing is tuned, so
/// `budget_breakdown()` is the raw estimate. The sparse arm of the property
/// the shared `tune()` driver will assert on every reduction step.
#[test]
fn the_sparse_budget_model_is_monotone_in_cache_shards() {
    let dir = tempfile::tempdir().unwrap();
    let estimate = |cache: usize| {
        // 4 GB over a fixture needing ~50 MB: nothing is tuned.
        let loader = sized_budget_loader(dir.path(), cache, Some(4096 * 1024 * 1024));
        assert_eq!(
            loader.effective_cache_shards(),
            cache,
            "premise: the budget must be generous enough that nothing is tuned"
        );
        loader.budget_breakdown().total_bytes
    };
    for cache in (1..32usize).rev() {
        assert!(
            estimate(cache) <= estimate(cache + 1),
            "cache_shards {cache} estimates more than {}",
            cache + 1
        );
    }
    assert!(estimate(0) <= estimate(32));
}

// ==========================================================================
// Gather-stage clip + seeded downsample (Phase 1B)
//
// These sit at the gather level rather than in `downsample_tests.rs` because
// what they assert is *placement*: that the clip and the draw happen before the
// batch leaves Rust, and that the draw is keyed to the row's own identity rather
// than to its position in the manifest or the batch.
// ==========================================================================

/// Multi-nonzero float fixture: row `r` has `nnz` entries at columns `0..nnz`
/// with values large enough that downsampling to a small target actually bites.
/// `negative_at` optionally forces one entry negative so the clip has something
/// to do (the `Uint8` fixture above cannot represent one).
fn write_float_fixture(
    path: &std::path::Path,
    n_obs: usize,
    n_vars: usize,
    n_shards: usize,
    nnz: usize,
    negative_at: Option<(usize, usize)>,
) {
    assert!(n_obs.is_multiple_of(n_shards));
    assert!(nnz <= n_vars);
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
        let mut indices: Vec<u32> = Vec::new();
        let mut bytes: Vec<u8> = Vec::new();
        for local in 0..rows_per_shard {
            let row = row_start + local;
            for c in 0..nnz {
                indices.push(c as u32);
                let mut v = (50 + (row * 7 + c * 3) % 50) as f32;
                if negative_at == Some((row, c)) {
                    v = -v;
                }
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            indptr.push(*indptr.last().unwrap() + nnz as u64);
        }
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &bytes,
                CodecId::None,
                ValueEncoding::Float32,
                row_start as u64,
            )
            .unwrap();
    }
    writer.finish().unwrap();
}

/// Gather one single-file set of `rows` from a loader.
fn gather_rows(loader: Arc<SparseCellSetLoader>, fid: u32, rows: &[u64]) -> SparseCellSetBatch {
    let n = rows.len();
    let plan = SparseCellSetPlan {
        file_ids: vec![fid; n],
        rows: rows.to_vec(),
        role_tags: vec![0; n],
        set_offsets: vec![0, n as i64],
    };
    loader
        .iter_with_plans(vec![Ok(plan)].into_iter(), 2)
        .next()
        .unwrap()
        .unwrap()
}

fn ds_cfg(
    target: u64,
    method: crate::downsample::DownsampleMethod,
    seed: u64,
    identities: Vec<u64>,
) -> crate::downsample::DownsampleConfig {
    crate::downsample::DownsampleConfig {
        target_library_size: target,
        method,
        seed,
        file_identities: identities,
    }
}

#[test]
fn gather_clips_negatives_in_the_emitted_csr() {
    // Before 1B a negative stored count travelled straight through the gather to
    // the consumer, while the collate kernel clipped it lazily per read — so the
    // two disagreed about what the row contained.
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("neg.scx");
    write_float_fixture(&p0, 8, 4, 2, 3, Some((5, 1)));
    let loader = SparseCellSetLoader::new(
        vec![open(&p0)],
        4,
        None,
        2,
        None,
        None,
        false,
        false,
        0.0,
        None,
        /*scatter_block_index*/ false,
        /*max_plan_rows*/ None,
    )
    .unwrap();

    let b = gather_rows(loader.clone(), 0, &[5]);
    let (idx, dat) = batch_row(&b, 0);
    assert!(dat.iter().all(|&v| v >= 0.0), "negative leaked: {dat:?}");
    // No downsample ⇒ no prune, so the clipped entry survives as an explicit
    // zero and nnz is unchanged. Callers that count nnz must see this.
    assert_eq!(idx.len(), 3, "clip must not change nnz: {idx:?}");
    assert_eq!(dat[1], 0.0, "the clipped entry should be an explicit zero");
}

#[test]
fn gather_clip_runs_after_coalescing_not_before() {
    // Two local columns mapping to one global gene, one positive and one negative.
    // Clipping before the coalesce would give 60; the reference clips after, which
    // gives 60 - 57 = 3. Getting this backwards is silent and only shows up as a
    // small systematic count inflation.
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("dup.scx");
    // Row 0, cols 0..2: values 50, 53, 56 by construction; force col 1 negative.
    write_float_fixture(&p0, 4, 4, 1, 3, Some((0, 1)));
    // locals 0 and 1 both -> global 0; local 2 -> global 1.
    let remap = vec![vec![0i32, 0, 1, -1]];
    let loader = SparseCellSetLoader::new(
        vec![open(&p0)],
        4,
        None,
        2,
        Some(remap),
        Some(2),
        false,
        false,
        0.0,
        None,
        /*scatter_block_index*/ false,
        /*max_plan_rows*/ None,
    )
    .unwrap();

    let b = gather_rows(loader.clone(), 0, &[0]);
    let (idx, dat) = batch_row(&b, 0);
    assert_eq!(idx, &[0, 1]);
    // 50 + (-53) = -3 -> clipped to 0. Clipping first would have produced 50.
    assert_eq!(
        dat[0], 0.0,
        "clip appears to run before the coalesce: {dat:?}"
    );
    assert_eq!(dat[1], 56.0);
}

#[test]
fn gather_downsamples_to_the_target_with_multinomial() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("d0.scx");
    write_float_fixture(&p0, 16, 8, 2, 6, None);
    let ident = crate::downsample::file_identity(p0.to_str().unwrap());
    let loader = SparseCellSetLoader::new(
        vec![open(&p0)],
        4,
        None,
        2,
        None,
        None,
        false,
        false,
        0.0,
        Some(ds_cfg(
            40,
            crate::downsample::DownsampleMethod::Multinomial,
            7,
            vec![ident],
        )),
        /*scatter_block_index*/ false,
        /*max_plan_rows*/ None,
    )
    .unwrap();

    let b = gather_rows(loader.clone(), 0, &[0, 1, 2, 3, 4]);
    for j in 0..5 {
        let (_, dat) = batch_row(&b, j);
        let lib: f32 = dat.iter().sum();
        assert_eq!(lib, 40.0, "row {j} missed the target: {dat:?}");
    }
}

#[test]
fn gather_without_downsample_leaves_counts_untouched() {
    // Anti-tautology for the test above: the fixture rows are well above the
    // target, so if the downsample were a no-op the assertion there would be
    // asserting nothing.
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("d0.scx");
    write_float_fixture(&p0, 16, 8, 2, 6, None);
    let loader = SparseCellSetLoader::new(
        vec![open(&p0)],
        4,
        None,
        2,
        None,
        None,
        false,
        false,
        0.0,
        None,
        /*scatter_block_index*/ false,
        /*max_plan_rows*/ None,
    )
    .unwrap();
    let (_, dat) = {
        let b = gather_rows(loader.clone(), 0, &[0]);
        let (i, d) = batch_row(&b, 0);
        (i.to_vec(), d.to_vec())
    };
    let lib: f32 = dat.iter().sum();
    assert!(
        lib > 40.0,
        "fixture too small to detect a downsample: {lib}"
    );
}

#[test]
fn gather_downsample_is_reproducible_across_loader_instances() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("d0.scx");
    write_float_fixture(&p0, 16, 8, 2, 6, None);
    let ident = crate::downsample::file_identity(p0.to_str().unwrap());

    let build = || {
        SparseCellSetLoader::new(
            vec![open(&p0)],
            4,
            None,
            2,
            None,
            None,
            false,
            false,
            0.0,
            Some(ds_cfg(
                30,
                crate::downsample::DownsampleMethod::Binomial,
                11,
                vec![ident],
            )),
            /*scatter_block_index*/ false,
            /*max_plan_rows*/ None,
        )
        .unwrap()
    };

    let a = gather_rows(build(), 0, &[3, 7, 11]);
    let b = gather_rows(build(), 0, &[3, 7, 11]);
    assert_eq!(a.data, b.data);
    assert_eq!(a.indices, b.indices);
}

#[test]
fn gather_downsample_is_invariant_to_row_order_within_a_plan() {
    // `read_rows_with` scatters in shard-grouped order, so the callback sees rows
    // in a different order than the plan requests them. The draw must follow the
    // row id, not the arrival order.
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("d0.scx");
    write_float_fixture(&p0, 16, 8, 4, 6, None);
    let ident = crate::downsample::file_identity(p0.to_str().unwrap());
    let loader = SparseCellSetLoader::new(
        vec![open(&p0)],
        8,
        None,
        2,
        None,
        None,
        false,
        false,
        0.0,
        Some(ds_cfg(
            25,
            crate::downsample::DownsampleMethod::Multinomial,
            3,
            vec![ident],
        )),
        /*scatter_block_index*/ false,
        /*max_plan_rows*/ None,
    )
    .unwrap();

    let forward = gather_rows(loader.clone(), 0, &[1, 5, 9, 13]);
    let reversed = gather_rows(loader.clone(), 0, &[13, 9, 5, 1]);
    for (j, r) in [1usize, 5, 9, 13].iter().enumerate() {
        let (_, fwd) = batch_row(&forward, j);
        // Same row, opposite position in the batch.
        let (_, rev) = batch_row(&reversed, 3 - j);
        assert_eq!(fwd, rev, "row {r} drew differently by batch position");
    }
}

#[test]
fn gather_downsample_is_invariant_to_manifest_order() {
    // THE test that justifies keying on the resolved path rather than on
    // `file_id`. `file_id` is loader-construction order, so a reordered manifest —
    // or a debugging subset — would silently redraw every cell under a `file_id`
    // key while producing perfectly plausible output.
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    write_float_fixture(&p0, 16, 8, 2, 6, None);
    write_float_fixture(&p1, 16, 8, 2, 5, None);
    let i0 = crate::downsample::file_identity(p0.to_str().unwrap());
    let i1 = crate::downsample::file_identity(p1.to_str().unwrap());

    let build = |paths: [&std::path::Path; 2], idents: Vec<u64>| {
        SparseCellSetLoader::new(
            vec![open(paths[0]), open(paths[1])],
            8,
            None,
            2,
            None,
            None,
            false,
            false,
            0.0,
            Some(ds_cfg(
                30,
                crate::downsample::DownsampleMethod::Multinomial,
                5,
                idents,
            )),
            /*scatter_block_index*/ false,
            /*max_plan_rows*/ None,
        )
        .unwrap()
    };

    // a.scx is file_id 0 here...
    let ab = gather_rows(build([&p0, &p1], vec![i0, i1]), 0, &[2, 6]);
    // ...and file_id 1 here. Same cells, same draw.
    let ba = gather_rows(build([&p1, &p0], vec![i1, i0]), 1, &[2, 6]);
    assert_eq!(
        ab.data, ba.data,
        "the draw moved when the manifest was reordered"
    );

    // And the two *files* must still differ from each other, or the identity is
    // being ignored altogether.
    let b_rows = gather_rows(build([&p0, &p1], vec![i0, i1]), 1, &[2, 6]);
    assert_ne!(
        ab.data, b_rows.data,
        "both files drew identically — file identity is not reaching the key"
    );
}

#[test]
fn gather_rejects_an_invalid_downsample_config() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("d0.scx");
    write_float_fixture(&p0, 8, 4, 2, 3, None);

    let mk = |cfg: crate::downsample::DownsampleConfig| {
        SparseCellSetLoader::new(
            vec![open(&p0)],
            4,
            None,
            2,
            None,
            None,
            false,
            false,
            0.0,
            Some(cfg),
            /*scatter_block_index*/ false,
            /*max_plan_rows*/ None,
        )
    };

    // Zero target cannot sample.
    assert!(mk(ds_cfg(
        0,
        crate::downsample::DownsampleMethod::Binomial,
        1,
        vec![7]
    ))
    .is_err());

    // An identity table that does not cover the files is a silent
    // wrong-cell-keyed bug waiting to happen, so it is rejected rather than
    // padded.
    let err = mk(ds_cfg(
        10,
        crate::downsample::DownsampleMethod::Binomial,
        1,
        vec![7, 8, 9],
    ))
    .err()
    .expect("mismatched identities must be rejected")
    .to_string();
    assert!(err.contains("file_identities"), "unhelpful: {err}");

    // A single file with no identities is fine — it keys on (seed, method, row).
    assert!(mk(ds_cfg(
        10,
        crate::downsample::DownsampleMethod::Binomial,
        1,
        vec![]
    ))
    .is_ok());
}

#[test]
fn gather_rejects_an_empty_identity_table_across_multiple_files() {
    // Without identities, two files' row N would share a draw. The tempting
    // fallback — key on `file_id` — is construction-order keying, the very scheme
    // `gather_downsample_is_invariant_to_manifest_order` exists to rule out. So
    // this is refused at construction rather than silently keyed.
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    write_float_fixture(&p0, 8, 4, 2, 3, None);
    write_float_fixture(&p1, 8, 4, 2, 3, None);

    let err = SparseCellSetLoader::new(
        vec![open(&p0), open(&p1)],
        4,
        None,
        2,
        None,
        None,
        false,
        false,
        0.0,
        Some(ds_cfg(
            10,
            crate::downsample::DownsampleMethod::Multinomial,
            1,
            vec![],
        )),
        /*scatter_block_index*/ false,
        /*max_plan_rows*/ None,
    )
    .err()
    .expect("multi-file downsample without identities must be rejected")
    .to_string();
    assert!(
        err.contains("file_identities"),
        "message should name the missing input: {err}"
    );
    assert!(
        err.contains("share a draw"),
        "message should say why it matters: {err}"
    );
}

// ==========================================================================
// CSR `indptr` validation
//
// The gap three reviewers converged on: `indptr.last() == nnz` is necessary but
// not sufficient, and the failure mode is a panic (an out-of-bounds slice, or a
// negative entry wrapping through `as usize`) rather than a wrong answer. Both
// entry points take arbitrary caller-supplied arrays.
// ==========================================================================

#[test]
fn validate_indptr_accepts_well_formed_input() {
    assert!(validate_indptr(&[0, 2, 5], 5).is_ok());
    assert!(validate_indptr(&[0], 0).is_ok());
    // Empty rows in the middle are legal CSR.
    assert!(validate_indptr(&[0, 3, 3, 3, 4], 4).is_ok());
}

#[test]
fn validate_indptr_rejects_a_non_monotonic_array_whose_last_entry_looks_right() {
    // The exact case reviewers named: `last == nnz` passes a naive check, then
    // `indices[0..3]` panics on a 2-element slice.
    let err = validate_indptr(&[0, 3, 2], 2).unwrap_err().to_string();
    assert!(err.contains("non-decreasing"), "unhelpful: {err}");
}

#[test]
fn validate_indptr_rejects_negatives_and_a_bad_start_and_a_bad_total() {
    // A negative would wrap to an enormous index through `as usize`.
    assert!(validate_indptr(&[0, -1, 2], 2).is_err());
    // A non-zero start silently drops a prefix rather than erroring.
    let err = validate_indptr(&[1, 3], 3).unwrap_err().to_string();
    assert!(err.contains("indptr[0]"), "unhelpful: {err}");
    // Totals that disagree with the data length.
    assert!(validate_indptr(&[0, 2], 5).is_err());
    assert!(validate_indptr(&[], 0).is_err());
}

#[test]
fn collate_gathered_errors_rather_than_panicking_on_a_bad_indptr() {
    // Before this check the call below panicked inside the rayon loop. An error
    // is the crate convention for malformed input, and a panic here would cross
    // the FFI boundary.
    let scalars = CollateScalars {
        k_enc: 2,
        mode: PreprocessMode::PassThrough,
        target_sum: 1e4,
        pflog_alpha: None,
        n_genes_total: 100,
        lib_size_redef: false,
    };
    let err = collate_gathered(
        &[0, 3, 2], // non-monotonic, last == nnz
        &[1, 2],
        &[1.0, 2.0],
        &[0, 2],
        vec![0, 1],
        vec![0, 0],
        vec![0, 0],
        1,
        &[1],
        None,
        &[],
        &[0, 0],
        &[2],
        &scalars,
    )
    .expect_err("a non-monotonic indptr must be an error, not a panic")
    .to_string();
    assert!(err.contains("non-decreasing"), "unhelpful: {err}");
}

// ---------------------------------------------------------------------
// §9.3 — teardown through the *production* ownership graph.
//
// The engine `Arc` has more owners than are visible from any one of them:
// `SparseCellSetLoader` holds one, `PlanPrefetchIter` holds one, and the
// `process` closure the iter stores holds an `Arc<SparseCellSetLoader>` that
// holds a third. A round-1 attempt put the deadline behind `Arc::into_inner`
// in `PlanPrefetchIter::drop`, which — because a `Drop` body runs before its
// struct's own fields — could never succeed here. The test that shipped with it
// used a capture-free `gather` and so passed against the broken path.
//
// These go through `SparseCellSetLoader::iter_with_plans`, which is what builds
// the loader-capturing closure.
// ---------------------------------------------------------------------

#[test]
fn teardown_through_the_real_iter_ownership_graph_is_bounded() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("f0.scx");
    write_fixture(&p0, 32, 8, 4);

    let before = crate::runtime::BOUNDED_SHUTDOWNS.with(|c| c.get());

    let loader = SparseCellSetLoader::new(
        vec![open(&p0)],
        /*cache_shards*/ 4,
        /*bytes_budget*/ None,
        /*lookahead*/ 4,
        /*remap*/ None,
        /*n_global_genes*/ None,
        /*normalize*/ false,
        /*log1p*/ false,
        /*target_sum*/ 1e4,
        /*downsample*/ None,
        /*scatter_block_index*/ false,
        /*max_plan_rows*/ None,
    )
    .unwrap();

    let plans = vec![
        SparseCellSetPlan {
            file_ids: vec![0, 0],
            rows: vec![0, 9],
            role_tags: vec![0, 0],
            set_offsets: vec![0, 2],
        },
        SparseCellSetPlan {
            file_ids: vec![0, 0],
            rows: vec![17, 26],
            role_tags: vec![0, 0],
            set_offsets: vec![0, 2],
        },
    ];
    // The production constructor: `process` closes over an
    // `Arc<SparseCellSetLoader>`, which owns another engine Arc.
    let mut it = Arc::clone(&loader).iter_with_plans(plans.into_iter().map(Ok), 4);
    it.next().expect("first batch").expect("must gather"); // forces the runtime

    // The `ds.close()` half: the dataset's loader reference goes away while the
    // iterator (and the closure inside it) are still alive.
    drop(loader);
    for b in it.by_ref() {
        b.expect("must gather");
    }
    drop(it);

    assert_eq!(
        crate::runtime::BOUNDED_SHUTDOWNS.with(|c| c.get()),
        before + 1,
        "the engine outlived every explicit owner, so its runtime must still \
         have gone down through BoundedRuntime::drop"
    );
}

/// Like [`write_fixture`] but with **ragged** rows: row `r` carries
/// `(r % 4) + 1` non-zeros, so the manifest mean is 2.5 and no single row has
/// it. Exists so the pre-sizing estimate can be tested where it is genuinely an
/// estimate — `write_fixture`'s uniform one-non-zero-per-row makes the mean
/// exact for every plan, which cannot distinguish an estimate from a count.
fn write_ragged_fixture(path: &std::path::Path, n_obs: usize, n_vars: usize, n_shards: usize) {
    assert!(n_obs.is_multiple_of(n_shards));
    let rows_per_shard = n_obs / n_shards;
    let header = FileHeader::new_single_modality(
        n_obs as u64,
        n_vars as u64,
        // Header nnz must match what the shards actually store, or the catalog
        // stats the estimate reads disagree with the payload.
        (0..n_obs).map(|r| (r % 4) as u64 + 1).sum::<u64>(),
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
            let nnz = (row % 4) + 1;
            for k in 0..nnz {
                // Ascending and unique within the row, as CSR requires.
                indices.push(((row + k) % n_vars) as u32);
                values.push(((row + k + 1) & 0xFF) as u8);
            }
            indptr.push(*indptr.last().unwrap() + nnz as u64);
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

/// W11 supersedes W1's estimate: the gather allocates **exactly**, on a fixture
/// built to make an estimate wrong in both directions.
///
/// ⚠️ This test used to assert the opposite, and said why: the capacity came
/// from the manifest's mean density, so a densest-rows plan under-shot and a
/// sparsest-rows plan over-shot. Its own closing comment asked to be told if
/// that ever changed — "pinned so a silent return to an exact indptr prescan,
/// which would make this case free, is visible" — and W11 is that return. The
/// batch executor reads the plan's rows through
/// `BackedCsrReader::read_row_indices_with_admission`, which prescans each
/// touched shard's indptr and carves the output exactly, so there is no
/// estimate left to be wrong. `presize_nnz` survives as the **budget** charge
/// only (see `the_batch_is_uncharged_until_max_plan_rows_is_declared`), where a
/// per-plan figure is not available at construction time.
///
/// The prescan is the ~9 % of `gather_grouped_s512` W1 measured and removed. It
/// bought a capacity hint then; it buys the exact spans, the removal of every
/// per-row `Vec` and the removal of one whole copy of the batch now. That trade
/// is what the phase-6 capture measures.
#[test]
fn gather_allocates_exactly_on_a_non_uniform_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ragged.scx");
    // Rows 0..64, row r has (r % 4) + 1 non-zeros ⇒ mean 2.5.
    write_ragged_fixture(&path, 64, 16, 2);

    let gather = |rows: Vec<u64>| {
        let loader = SparseCellSetLoader::new(
            vec![open(&path)],
            8,
            None,
            4,
            None,
            None,
            false,
            false,
            0.0,
            None,
            false,
            None,
        )
        .unwrap();
        let n = rows.len();
        let plan = SparseCellSetPlan {
            file_ids: vec![0; n],
            rows,
            role_tags: vec![0; n],
            set_offsets: vec![0, n as i64],
        };
        loader
            .iter_with_plans(vec![Ok(plan)].into_iter(), 4)
            .map(|r| r.unwrap())
            .next()
            .unwrap()
    };

    // Densest rows only (r % 4 == 3 ⇒ 4 nnz each): the mean under-shoots.
    let dense: Vec<u64> = (0..64).filter(|r| r % 4 == 3).collect();
    let n_dense = dense.len();
    let b = gather(dense);
    assert_eq!(b.indices.len(), n_dense * 4, "4 nnz per selected row");
    // The case the estimate could not serve without reading the data: the
    // biased mean still falls short of a densest-rows plan, and the exact
    // allocation covers it with no reallocation at all.
    assert!(
        crate::sparse_cellset::presize_nnz(n_dense, 2.5) < b.indices.len(),
        "premise: a densest-rows plan exceeds the biased estimate ({} vs {})",
        crate::sparse_cellset::presize_nnz(n_dense, 2.5),
        b.indices.len()
    );
    assert_eq!(
        b.indices.capacity(),
        b.indices.len(),
        "exact, not estimated"
    );
    assert_eq!(b.data.capacity(), b.data.len(), "exact, not estimated");

    // Sparsest rows only (r % 4 == 0 ⇒ 1 nnz each): the mean over-shoots.
    let sparse: Vec<u64> = (0..64).filter(|r| r % 4 == 0).collect();
    let n_sparse = sparse.len();
    let b = gather(sparse);
    assert_eq!(b.indices.len(), n_sparse, "1 nnz per selected row");
    assert!(
        crate::sparse_cellset::presize_nnz(n_sparse, 2.5) > b.indices.len(),
        "premise: the mean over-estimates a sparsest-rows plan ({} vs {})",
        crate::sparse_cellset::presize_nnz(n_sparse, 2.5),
        b.indices.len()
    );
    assert_eq!(
        b.indices.capacity(),
        b.indices.len(),
        "exact, not estimated"
    );
    assert_eq!(b.data.capacity(), b.data.len(), "exact, not estimated");

    // Whole file: the only shape the mean could ever get right, and the exact
    // allocation is still tighter than the biased estimate.
    let b = gather((0..64).collect());
    assert_eq!(b.indices.len(), 160, "sum of (r % 4) + 1 over 64 rows");
    assert_eq!(b.indices.capacity(), 160, "exact, not estimated");
    assert!(
        crate::sparse_cellset::presize_nnz(64, 2.5) > b.indices.capacity(),
        "premise: the biased estimate would have over-allocated even here"
    );
}

/// W11: the raw-local fast path allocates `indices`/`data` **once, exactly** —
/// there is no append loop left to reallocate.
///
/// ⚠️ Supersedes W1's assertion, which was `capacity() == presize_nnz(rows,
/// mean)` — the capacity the estimate asked for. A single-file plan with no
/// duplicate rows and no length-changing transform is now served by moving the
/// `ScxCsr` that `read_row_indices_with_admission` returns straight out as the
/// batch, so `capacity() == len()` is the whole claim: one allocation for the
/// gather, sized by the indptr prescan rather than guessed from the manifest
/// mean.
#[test]
fn gather_allocates_indices_and_data_exactly_on_the_raw_local_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("presize.scx");
    write_fixture(&path, 640, 32, 4);

    let loader = SparseCellSetLoader::new(
        vec![open(&path)],
        /*cache_shards*/ 8,
        None,
        /*lookahead*/ 4,
        /*remap*/ None,
        /*n_global_genes*/ None,
        false,
        false,
        0.0,
        /*downsample*/ None,
        /*scatter_block_index*/ false,
        /*max_plan_rows*/ None,
    )
    .unwrap();

    // Every row of the fixture carries exactly one non-zero, so the expected
    // total is the row count — known independently of the code under test.
    let rows: Vec<u64> = (0..640).collect();
    let plan = SparseCellSetPlan {
        file_ids: vec![0; 640],
        rows,
        role_tags: vec![0; 640],
        set_offsets: vec![0, 320, 640],
    };
    let batches: Vec<_> = loader
        .iter_with_plans(vec![Ok(plan)].into_iter(), 4)
        .map(|r| r.unwrap())
        .collect();
    let b = &batches[0];

    assert_eq!(b.indices.len(), 640, "one nnz per row in this fixture");
    assert_eq!(
        b.indices.capacity(),
        640,
        "indices should be exactly sized, not estimated (len {})",
        b.indices.len()
    );
    assert_eq!(
        b.data.capacity(),
        640,
        "data should be exactly sized, not estimated (len {})",
        b.data.len()
    );
    assert_eq!(b.indptr.len(), 641);
    // `presize_nnz` is no longer what sizes the gather; it survives only as the
    // budget charge, where no per-plan figure exists at construction time. The
    // estimate would have over-allocated this plan by an eighth.
    assert!(crate::sparse_cellset::presize_nnz(640, 1.0) > b.indices.capacity());
}

/// The pre-size must be a *bound*, never a truncation: `capacity >= len` on a
/// path where the transform drops entries (remap sentinels).
#[test]
fn gather_presize_is_an_upper_bound_when_remap_drops_entries() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("presize_remap.scx");
    write_fixture(&path, 64, 8, 2);

    // Map column 0 to the `-1` drop sentinel, every other column to itself.
    let mut table: Vec<i32> = (0..8).collect();
    table[0] = -1;
    let loader = SparseCellSetLoader::new(
        vec![open(&path)],
        /*cache_shards*/ 4,
        None,
        /*lookahead*/ 2,
        /*remap*/ Some(vec![table]),
        /*n_global_genes*/ Some(8),
        false,
        false,
        0.0,
        /*downsample*/ None,
        /*scatter_block_index*/ false,
        /*max_plan_rows*/ None,
    )
    .unwrap();

    let rows: Vec<u64> = (0..64).collect();
    let plan = SparseCellSetPlan {
        file_ids: vec![0; 64],
        rows,
        role_tags: vec![0; 64],
        set_offsets: vec![0, 64],
    };
    let batches: Vec<_> = loader
        .iter_with_plans(vec![Ok(plan)].into_iter(), 2)
        .map(|r| r.unwrap())
        .collect();
    let b = &batches[0];

    assert!(
        b.indices.capacity() >= b.indices.len(),
        "pre-size under-allocated: capacity {} < len {}",
        b.indices.capacity(),
        b.indices.len()
    );
    // Rows whose single non-zero sat at column 0 dropped out entirely.
    let dropped = (0..64u64).filter(|r| r % 8 == 0).count();
    assert_eq!(b.indices.len(), 64 - dropped);
}

// --------------------------------------------------------------------------
// Contract v3 — per-row query addressing
// --------------------------------------------------------------------------

fn v3_scalars(mode: PreprocessMode) -> CollateScalars {
    CollateScalars {
        k_enc: 4,
        mode,
        target_sum: 1e4,
        pflog_alpha: Some(0.25),
        n_genes_total: 8,
        lib_size_redef: false,
    }
}

/// Two rows in one set, three expressed genes each.
fn v3_fixture() -> (Vec<i64>, Vec<i32>, Vec<f32>, Vec<i64>) {
    (
        vec![0, 3, 6],
        vec![1, 3, 5, 1, 3, 5],
        vec![4.0, 2.0, 6.0, 1.0, 9.0, 3.0],
        vec![0, 2],
    )
}

#[allow(clippy::too_many_arguments)]
fn v3_collate(
    k_dec: usize,
    query: &[i32],
    offsets: Option<&[i64]>,
    mask: &[u8],
    n_measured: &[u32],
    scalars: &CollateScalars,
) -> CollatedCellSetBatch {
    let (indptr, indices, data, set_offsets) = v3_fixture();
    collate_gathered(
        &indptr,
        &indices,
        &data,
        &set_offsets,
        vec![0u64, 1],
        vec![0u32, 0],
        vec![0i32, 0],
        k_dec,
        query,
        offsets,
        mask,
        &[0, 0],
        n_measured,
        scalars,
    )
    .unwrap()
}

fn assert_same_batch(a: &CollatedCellSetBatch, b: &CollatedCellSetBatch) {
    assert_eq!(a.encoder_gene_ids, b.encoder_gene_ids);
    assert_eq!(a.encoder_counts, b.encoder_counts);
    assert_eq!(a.encoder_mask, b.encoder_mask);
    assert_eq!(a.encoder_pad_mask, b.encoder_pad_mask);
    assert_eq!(a.target_counts, b.target_counts);
    assert_eq!(a.target_pad_mask, b.target_pad_mask);
    assert_eq!(a.library_size, b.library_size);
    assert_eq!((a.n_rows, a.k_enc, a.k_dec), (b.n_rows, b.k_enc, b.k_dec));
}

#[test]
fn per_row_queries_reproduce_the_per_set_call_when_the_rows_share_a_query() {
    // The compatibility claim: v2's addressing is a special case of v3's, so a
    // caller that duplicates the set's panel per row gets byte-identical output.
    let sc = v3_scalars(PreprocessMode::PassThrough);
    let mask = [1u8, 0, 0, 0, 0, 1];
    let per_set = v3_collate(3, &[1, 3, 5], None, &mask, &[3], &sc);
    let per_row = v3_collate(
        3,
        &[1, 3, 5, 1, 3, 5],
        Some(&[0, 3, 6]),
        &mask,
        &[3, 3],
        &sc,
    );
    assert_same_batch(&per_set, &per_row);
}

#[test]
fn per_row_queries_let_two_rows_of_one_set_query_different_genes() {
    // The capability v3 exists for. Row 0 asks for gene 1, row 1 for gene 3; a
    // per-set call cannot express this at all.
    let sc = v3_scalars(PreprocessMode::PassThrough);
    let b = v3_collate(1, &[1, 3], Some(&[0, 1, 2]), &[], &[3, 3], &sc);
    assert_eq!(b.target_counts, vec![4.0, 9.0]);
    assert_eq!(b.target_pad_mask, vec![0, 0]);
}

#[test]
fn per_row_queries_pad_short_rows_and_mark_the_padding() {
    // Row 0 queries two genes, row 1 one. `k_dec` is the padded output width, and
    // without the mask the padded 0.0 is indistinguishable from a real zero.
    let sc = v3_scalars(PreprocessMode::PassThrough);
    let b = v3_collate(2, &[1, 3, 5], Some(&[0, 2, 3]), &[], &[3, 3], &sc);
    assert_eq!(b.k_dec, 2);
    assert_eq!(b.target_counts, vec![4.0, 2.0, 3.0, 0.0]);
    assert_eq!(b.target_pad_mask, vec![0, 0, 0, 1]);
}

#[test]
fn a_real_zero_target_is_distinguishable_from_padding() {
    // Gene 7 is not in either row, so its target is a genuine 0.0 with pad 0 —
    // the pair the mask exists to separate.
    let sc = v3_scalars(PreprocessMode::PassThrough);
    let b = v3_collate(2, &[7, 1, 3], Some(&[0, 2, 3]), &[], &[3, 3], &sc);
    assert_eq!(b.target_counts, vec![0.0, 4.0, 9.0, 0.0]);
    assert_eq!(b.target_pad_mask, vec![0, 0, 0, 1]);
}

#[test]
fn target_pad_mask_is_all_zero_on_the_per_set_path() {
    let sc = v3_scalars(PreprocessMode::PassThrough);
    let b = v3_collate(3, &[1, 3, 5], None, &[], &[3], &sc);
    assert_eq!(b.target_pad_mask, vec![0u8; 2 * 3]);
}

#[test]
fn per_row_masks_are_parallel_to_the_ragged_query_not_k_dec_strided() {
    // Row 0's query is [1, 3] and row 1's is [5]; the mask has three entries, one
    // per query position, and withholds gene 3 from row 0 only.
    let sc = v3_scalars(PreprocessMode::PassThrough);
    let b = v3_collate(2, &[1, 3, 5], Some(&[0, 2, 3]), &[0, 1, 0], &[3, 3], &sc);
    // Row 0: genes 5, 1 survive (3 withheld) in count-desc order 6.0, 4.0.
    assert_eq!(&b.encoder_gene_ids[0..2], &[5, 1]);
    // Row 1: nothing withheld, all three genes present, 9.0 > 3.0 > 1.0.
    assert_eq!(&b.encoder_gene_ids[4..7], &[3, 5, 1]);
}

#[test]
fn n_measured_is_read_per_row_when_query_offsets_are_supplied() {
    // Discriminating because `n_measured` is the PFlog centring denominator:
    // two rows of one set with different panel sizes must centre differently,
    // which a per-set read cannot produce.
    let sc = v3_scalars(PreprocessMode::PflogRaw);
    let same = v3_collate(1, &[1, 1], Some(&[0, 1, 2]), &[], &[3, 3], &sc);
    let differ = v3_collate(1, &[1, 1], Some(&[0, 1, 2]), &[], &[3, 100], &sc);
    assert_eq!(same.encoder_counts[0], differ.encoder_counts[0]);
    assert_ne!(
        same.encoder_counts[sc.k_enc], differ.encoder_counts[sc.k_enc],
        "row 1's centring must follow row 1's n_measured"
    );
}

#[test]
fn a_batch_of_singleton_sets_still_reads_the_per_set_addressing() {
    // `n_sets == n_rows` here, so a dispatch that sniffed lengths instead of
    // checking `query_offsets.is_some()` would read the per-set panel as a
    // ragged one and silently give every row the first set's query.
    let sc = v3_scalars(PreprocessMode::PassThrough);
    let (indptr, indices, data, _) = v3_fixture();
    let b = collate_gathered(
        &indptr,
        &indices,
        &data,
        &[0, 1, 2], // two singleton sets
        vec![0u64, 1],
        vec![0u32, 0],
        vec![0i32, 0],
        2,
        &[1, 3, 5, 7], // per-SET: [1, 3] for set 0, [5, 7] for set 1
        None,
        &[],
        &[0, 0],
        &[3, 3],
        &sc,
    )
    .unwrap();
    assert_eq!(b.target_counts, vec![4.0, 2.0, 3.0, 0.0]);
}

#[test]
fn malformed_query_offsets_are_errors_not_panics() {
    let sc = v3_scalars(PreprocessMode::PassThrough);
    let (indptr, indices, data, set_offsets) = v3_fixture();
    let call = |k_dec: usize, query: &[i32], off: &[i64], nm: &[u32]| {
        collate_gathered(
            &indptr,
            &indices,
            &data,
            &set_offsets,
            vec![0u64, 1],
            vec![0u32, 0],
            vec![0i32, 0],
            k_dec,
            query,
            Some(off),
            &[],
            &[0, 0],
            nm,
            &sc,
        )
    };
    // Wrong length.
    assert!(call(3, &[1, 3, 5], &[0, 3], &[3, 3]).is_err());
    // Non-monotonic, last == len.
    assert!(call(3, &[1, 3, 5], &[0, 5, 3], &[3, 3]).is_err());
    // Does not end at the flat query length.
    assert!(call(3, &[1, 3, 5], &[0, 1, 2], &[3, 3]).is_err());
    // k_dec narrower than the widest row query.
    assert!(call(1, &[1, 3, 5], &[0, 2, 3], &[3, 3]).is_err());
    // n_measured still per-set.
    assert!(call(3, &[1, 3, 5], &[0, 2, 3], &[3]).is_err());
}

#[test]
fn each_row_consults_its_own_query_panel() {
    // Both rows of the set flag their single query position, but they query
    // DIFFERENT genes — so a lookup that reused the set's first panel would
    // withhold gene 1 from both instead of gene 1 from row 0 and gene 3 from
    // row 1. The weaker version of this test (row 1 flagging nothing) passes
    // with the panel indexed by set, which is why it is written this way.
    let sc = v3_scalars(PreprocessMode::PassThrough);
    let b = v3_collate(1, &[1, 3], Some(&[0, 1, 2]), &[1, 1], &[3, 3], &sc);
    // Row 0 (4.0, 2.0, 6.0 on genes 1, 3, 5) loses gene 1.
    assert_eq!(&b.encoder_gene_ids[0..2], &[5, 3]);
    // Row 1 (1.0, 9.0, 3.0 on genes 1, 3, 5) loses gene 3 — NOT gene 1.
    assert_eq!(&b.encoder_gene_ids[4..6], &[5, 1]);
}

#[test]
fn collate_gathered_rejects_the_three_shapes_that_used_to_panic() {
    // All three were reproduced against the built extension as
    // `pyo3_runtime.PanicException` or as silently non-finite output before
    // being closed; `validate_indptr` had already covered the fourth.
    let sc = v3_scalars(PreprocessMode::PassThrough);
    let (indptr, indices, data, set_offsets) = v3_fixture();
    let call = |k_enc: usize, k_dec: usize, ix: &[i32], dt: &[f32], nm: &[u32], mode| {
        let mut scalars = sc;
        scalars.k_enc = k_enc;
        scalars.mode = mode;
        collate_gathered(
            &indptr,
            ix,
            dt,
            &set_offsets,
            vec![0u64, 1],
            vec![0u32, 0],
            vec![0i32, 0],
            k_dec,
            &[1],
            None,
            &[],
            &[0, 0],
            nm,
            &scalars,
        )
    };
    // k_enc == 0 => `par_chunks_mut(0)` panics with "chunk_size must not be zero".
    assert!(call(0, 1, &indices, &data, &[3], PreprocessMode::PassThrough).is_err());
    // k_dec == 0 => same, on the target chunks.
    assert!(call(4, 0, &indices, &data, &[3], PreprocessMode::PassThrough).is_err());
    // A short `indices` against a well-formed `indptr` over `data` slices OOB.
    assert!(call(
        4,
        1,
        &indices[..3],
        &data,
        &[3],
        PreprocessMode::PassThrough
    )
    .is_err());
    // PflogRaw with n_measured == 0 centres by a zero denominator and wrote
    // -inf into every encoder slot rather than erroring.
    let mut pflog = sc;
    pflog.mode = PreprocessMode::PflogRaw;
    let err = collate_gathered(
        &indptr,
        &indices,
        &data,
        &set_offsets,
        vec![0u64, 1],
        vec![0u32, 0],
        vec![0i32, 0],
        1,
        &[1],
        None,
        &[],
        &[0, 0],
        &[0],
        &pflog,
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("n_measured >= 1"),
        "unexpected error: {err}"
    );
    // And the well-formed call still works, so the guards are not vacuous.
    assert!(call(4, 1, &indices, &data, &[3], PreprocessMode::PassThrough).is_ok());
}

#[test]
fn collate_gathered_validates_set_offsets_like_any_other_prefix_array() {
    // `set_offsets` was the one prefix array on this entry that nothing checked.
    // `[0, 1]` over two rows was ACCEPTED and left row 1 with `row_set = 0` — a
    // silently wrong set assignment, which is worse than the panic the same hole
    // reached at `n_sets = 0`.
    let sc = v3_scalars(PreprocessMode::PassThrough);
    let (indptr, indices, data, _) = v3_fixture();
    let call = |set_offsets: &[i64], n_measured: &[u32]| {
        collate_gathered(
            &indptr,
            &indices,
            &data,
            set_offsets,
            vec![0u64, 1],
            vec![0u32, 0],
            vec![0i32, 0],
            1,
            &[1, 1],
            None,
            &[],
            &[0, 0],
            n_measured,
            &sc,
        )
    };
    // Does not reach the last row.
    assert!(call(&[0, 1], &[3]).is_err());
    // Does not start at 0.
    assert!(call(&[1, 2], &[3]).is_err());
    // Non-monotonic.
    assert!(call(&[0, 2, 1], &[3, 3]).is_err());
    // Anti-vacuity: the well-formed two-singleton-set case is accepted.
    assert!(call(&[0, 1, 2], &[3, 3]).is_ok());
}

#[test]
fn collate_gathered_refuses_an_unsorted_row_instead_of_gathering_zeros() {
    // `collate_cell` exact-match binary-searches `gene_ids` for its target
    // gather. On the built extension a row `{0: 5, 2: 7, 1: 9}` queried with
    // `[0, 1, 2]` returned targets `[5.0, 0.0, 0.0]` instead of
    // `[5.0, 9.0, 7.0]` — corrupted training batches, no error.
    let sc = v3_scalars(PreprocessMode::PassThrough);
    let err = collate_gathered(
        &[0, 3],
        &[0, 2, 1],
        &[5.0, 7.0, 9.0],
        &[0, 1],
        vec![0u64],
        vec![0u32],
        vec![0i32],
        3,
        &[0, 1, 2],
        None,
        &[],
        &[0],
        &[8],
        &sc,
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("strictly ascending"),
        "unexpected error: {err}"
    );
}

#[test]
fn collate_gathered_refuses_a_gene_id_that_would_collide_with_gene_mask() {
    // `n_genes_total` IS the GENE_MASK token, so an id at or above it emits a
    // real gene indistinguishable from the sentinel.
    let sc = v3_scalars(PreprocessMode::PassThrough); // n_genes_total = 8
    let err = collate_gathered(
        &[0, 2],
        &[3, 8],
        &[5.0, 7.0],
        &[0, 1],
        vec![0u64],
        vec![0u32],
        vec![0i32],
        1,
        &[3],
        None,
        &[],
        &[0],
        &[8],
        &sc,
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("outside the vocabulary"),
        "unexpected error: {err}"
    );
}

// ---------------------------------------------------------------------------
// W11 — the multi-set batch executor
// ---------------------------------------------------------------------------

/// Every field of a batch, for a field-for-field comparison of the two
/// executors.
///
/// Destructured rather than listed: `SparseCellSetBatch` has no `PartialEq`,
/// and a field added later is a compile error here instead of a field that
/// silently stops being compared. (`bounded_and_unbounded_gathers_are_byte_identical`
/// in `tests/test_reader_registry.rs` omitted `set_offsets` and `role_tags` for
/// exactly that reason.)
#[allow(clippy::type_complexity)]
fn batch_fields(
    b: &SparseCellSetBatch,
) -> (
    &[i64],
    &[i32],
    &[f32],
    (usize, usize),
    &[u64],
    &[u32],
    &[i64],
    &[i32],
) {
    let SparseCellSetBatch {
        indptr,
        indices,
        data,
        shape,
        cell_indices,
        file_ids,
        set_offsets,
        role_tags,
    } = b;
    (
        indptr,
        indices,
        data,
        *shape,
        cell_indices,
        file_ids,
        set_offsets,
        role_tags,
    )
}

/// Run one plan through both executors, each on its own cold loader, and assert
/// every field agrees. The verdict is held at `Admit::All` so the only variable
/// is the executor.
fn assert_executors_agree(
    make: &dyn Fn() -> StdArc<SparseCellSetLoader>,
    plan: &SparseCellSetPlan,
    label: &str,
) -> SparseCellSetBatch {
    let a = make();
    let whole = a
        .gather_whole_plan(&a.engine, plan, Some(Admit::All))
        .unwrap_or_else(|e| panic!("{label}: whole-plan executor: {e}"));
    let b = make();
    let per_set = b
        .gather_per_set(&b.engine, plan, Some(Admit::All))
        .unwrap_or_else(|e| panic!("{label}: per-set walk: {e}"));
    assert_eq!(batch_fields(&whole), batch_fields(&per_set), "{label}");
    whole
}

/// A loader factory, so each executor arm gets its own cold cache.
type MakeLoader<'a> = Box<dyn Fn() -> StdArc<SparseCellSetLoader> + 'a>;

fn ragged_loader(
    path: &std::path::Path,
    remap: Option<Vec<Vec<i32>>>,
    n_global: Option<usize>,
    normalize: bool,
    log1p: bool,
    downsample: Option<crate::downsample::DownsampleConfig>,
) -> StdArc<SparseCellSetLoader> {
    SparseCellSetLoader::new(
        vec![open(path)],
        /*cache_shards*/ 8,
        None,
        /*lookahead*/ 4,
        remap,
        n_global,
        normalize,
        log1p,
        /*target_sum*/ 1e4,
        downsample,
        /*scatter_block_index*/ false,
        /*max_plan_rows*/ None,
    )
    .unwrap()
}

fn plan_of(rows: Vec<u64>, set_offsets: Vec<i64>) -> SparseCellSetPlan {
    let n = rows.len();
    SparseCellSetPlan {
        file_ids: vec![0; n],
        // Distinct per position, so a batch that emits the right rows in the
        // wrong order still fails.
        role_tags: (0..n as i32).collect(),
        rows,
        set_offsets,
    }
}

/// The parity table: every plan shape the two executors must agree on, under
/// every transform configuration.
///
/// The fixture is **ragged** (row `r` carries `(r % 4) + 1` non-zeros) on
/// purpose — a one-nnz-per-row fixture cannot tell a correct span from an
/// off-by-one one, because every span is the same width.
#[test]
fn whole_plan_gather_matches_the_per_set_walk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("parity.scx");
    write_ragged_fixture(&path, 64, 16, 2);

    // Local col 0 -> the `-1` drop sentinel; cols 1 and 2 collide on one global
    // gene, so the remap arm exercises the drop AND the coalesce.
    let mut table: Vec<i32> = (0..16).collect();
    table[0] = -1;
    table[2] = 1;

    let configs: Vec<(&str, MakeLoader<'_>)> = vec![
        (
            "raw-local",
            Box::new(|| ragged_loader(&path, None, None, false, false, None)),
        ),
        (
            "remap (drops + coalesce)",
            Box::new(|| {
                ragged_loader(
                    &path,
                    Some(vec![table.clone()]),
                    Some(16),
                    false,
                    false,
                    None,
                )
            }),
        ),
        (
            "normalize + log1p",
            Box::new(|| ragged_loader(&path, None, None, true, true, None)),
        ),
        (
            "downsample",
            Box::new(|| {
                ragged_loader(
                    &path,
                    None,
                    None,
                    false,
                    false,
                    Some(ds_cfg(
                        3,
                        crate::downsample::DownsampleMethod::Multinomial,
                        11,
                        vec![777],
                    )),
                )
            }),
        ),
    ];

    let shapes: Vec<(&str, SparseCellSetPlan)> = vec![
        (
            "distinct rows, four sets",
            plan_of((0..16).collect(), vec![0, 4, 8, 12, 16]),
        ),
        (
            "adjacent duplicate inside one set",
            plan_of(vec![5, 3, 0, 7, 7, 31], vec![0, 3, 6]),
        ),
        (
            "non-adjacent duplicate inside one set",
            plan_of(vec![7, 3, 7, 0, 31, 7], vec![0, 3, 6]),
        ),
        (
            "the same row in two different sets",
            plan_of(vec![9, 2, 40, 9, 17, 40], vec![0, 3, 6]),
        ),
        (
            "a shared block in every set (the STATE3 shape)",
            plan_of(
                vec![1, 2, 3, 20, 1, 2, 3, 21, 1, 2, 3, 22],
                vec![0, 4, 8, 12],
            ),
        ),
        (
            "empty set in the middle",
            plan_of(vec![4, 5, 6], vec![0, 2, 2, 3]),
        ),
        (
            "empty sets at both ends",
            plan_of(vec![4, 5, 6], vec![0, 0, 3, 3]),
        ),
        (
            "set_offsets covering only part of the plan",
            plan_of(vec![10, 11, 12, 13, 14], vec![1, 3]),
        ),
        ("no sets at all", plan_of(vec![1, 2, 3], vec![0])),
        // `set_offsets` is not required to be non-empty either, and
        // `validate_plan`'s loop accepts it, so both executors must.
        ("no set_offsets at all", plan_of(vec![1, 2, 3], vec![])),
        ("one empty set", plan_of(vec![1, 2, 3], vec![0, 0])),
        (
            "cross-shard, descending, with repeats",
            plan_of(vec![63, 1, 63, 32, 31, 1], vec![0, 6]),
        ),
        (
            "a zero-nnz row after the remap drop",
            // Rows whose only column is local 0 map to nothing under `table`.
            plan_of(vec![0, 16, 32, 48, 1], vec![0, 5]),
        ),
    ];

    for (cfg_name, make) in &configs {
        for (shape_name, plan) in &shapes {
            assert_executors_agree(make.as_ref(), plan, &format!("{cfg_name} / {shape_name}"));
        }
    }
}

/// A `(file, row)` named by two different sets of one plan comes back at both
/// positions, with the same bytes, from **one** shard decode.
///
/// No Rust test covered a cross-set duplicate before W11: every multi-set plan
/// literal in this file used disjoint `(file, row)` pairs across its sets, and
/// the one duplicate that existed (`[5, 3, 0, 7, 7, 31]`) was adjacent inside a
/// single set.
///
/// ⚠️ It says nothing about **row** deduplication, because on this
/// configuration there is none: a raw-local single-file plan takes the direct
/// path, where a repeat costs one more memcpy from the resident shard and the
/// batch is the read. `duplicate_occurrences_draw_the_same_downsample` and
/// `the_occurrence_table_dedups_across_sets_and_within_them` cover the
/// configuration that does deduplicate.
#[test]
fn a_row_in_two_sets_comes_back_at_both_positions() {
    use std::sync::atomic::Ordering as AtomicOrdering;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dup_across_sets.scx");
    write_ragged_fixture(&path, 64, 16, 2);

    // Row 9 in both sets; every other row distinct. Eight positions, seven
    // unique rows.
    let plan = plan_of(vec![9, 2, 40, 5, 9, 17, 40, 33], vec![0, 4, 8]);

    let loader = ragged_loader(&path, None, None, false, false, None);
    let b = loader
        .gather_whole_plan(&loader.engine, &plan, Some(Admit::All))
        .unwrap();

    // Both occurrences are present, in plan order, with the same bytes.
    assert_eq!(b.cell_indices, plan.rows);
    assert_eq!(batch_row(&b, 0), batch_row(&b, 4), "row 9 twice");
    assert_eq!(batch_row(&b, 2), batch_row(&b, 6), "row 40 twice");
    for (j, &row) in plan.rows.iter().enumerate() {
        let nnz = (row as usize % 4) + 1;
        assert_eq!(batch_row(&b, j).0.len(), nnz, "position {j} (row {row})");
    }

    // And the plan issued exactly one read, over the seven unique rows: one
    // whole-shard miss per shard, not one per set.
    let m = loader.cache_metrics();
    assert_eq!(
        m.misses.load(AtomicOrdering::Relaxed),
        2,
        "two shards, decoded once each for the whole plan"
    );
}

/// Duplicate occurrences must draw the **same** downsample.
///
/// They do because `row_seed` keys on the file's content identity and the
/// physical row, never on batch position — which is what makes it safe to
/// transform a row once and memcpy it into each of its positions. Guarded from
/// the other side by `gather_downsample_is_invariant_to_row_order_within_a_plan`.
#[test]
fn duplicate_occurrences_draw_the_same_downsample() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dup_downsample.scx");
    write_ragged_fixture(&path, 64, 16, 2);
    let loader = ragged_loader(
        &path,
        None,
        None,
        false,
        false,
        Some(ds_cfg(
            2,
            crate::downsample::DownsampleMethod::Multinomial,
            99,
            vec![4242],
        )),
    );
    // Row 3 carries four non-zeros, so a target of 2 really does resample it.
    let plan = plan_of(vec![3, 11, 3, 27, 3], vec![0, 2, 5]);
    let b = loader
        .gather_whole_plan(&loader.engine, &plan, Some(Admit::All))
        .unwrap();
    // Against the per-set walk, which transforms every occurrence separately:
    // if the draw were keyed on anything but the physical row, the two
    // executors would disagree even though all three copies still matched.
    let ref_loader = ragged_loader(
        &path,
        None,
        None,
        false,
        false,
        Some(ds_cfg(
            2,
            crate::downsample::DownsampleMethod::Multinomial,
            99,
            vec![4242],
        )),
    );
    let per_set = ref_loader
        .gather_per_set(&ref_loader.engine, &plan, Some(Admit::All))
        .unwrap();
    assert_eq!(batch_fields(&b), batch_fields(&per_set));
    assert_eq!(batch_row(&b, 0), batch_row(&b, 2));
    assert_eq!(batch_row(&b, 0), batch_row(&b, 4));
    let (_, dat) = batch_row(&b, 0);
    assert_eq!(
        dat.iter().sum::<f32>(),
        2.0,
        "premise: the row really was downsampled"
    );
}

/// An empty set must not shift the fields that are indexed by **plan** position
/// rather than by set-local position.
///
/// `empty_set_keeps_boundary_without_rows` asserts `set_offsets`,
/// `cell_indices` and the row count — but not `role_tags` or `file_ids`, and
/// `role_tags` is the one field the per-set walk reads as `plan.role_tags[lo + j]`.
#[test]
fn an_empty_set_does_not_shift_role_tags_or_file_ids() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty_set.scx");
    write_ragged_fixture(&path, 64, 16, 2);
    let loader = ragged_loader(&path, None, None, false, false, None);

    let plan = SparseCellSetPlan {
        file_ids: vec![0; 5],
        rows: vec![4, 5, 6, 7, 8],
        role_tags: vec![10, 11, 12, 13, 14],
        // Two empty sets, one of them between two populated ones.
        set_offsets: vec![0, 2, 2, 4, 4, 5],
    };
    let b = loader
        .gather_whole_plan(&loader.engine, &plan, Some(Admit::All))
        .unwrap();
    assert_eq!(b.role_tags, vec![10, 11, 12, 13, 14]);
    assert_eq!(b.file_ids, vec![0; 5]);
    assert_eq!(b.cell_indices, vec![4, 5, 6, 7, 8]);
    assert_eq!(b.set_offsets, plan.set_offsets);
}

/// A cross-file plan with **several** rows per file keeps plan order.
///
/// `gather_cross_file_set_concatenates_in_global_space` is the weakest instance
/// the shape allows: one row per file, and only `indices` asserted. Both
/// executors group a plan's rows by file, so a bug that emitted the per-file
/// concatenation instead of the plan order needs more than one row per file to
/// show.
#[test]
fn a_cross_file_plan_with_many_rows_per_file_keeps_plan_order() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("cf0.scx");
    let p1 = dir.path().join("cf1.scx");
    write_ragged_fixture(&p0, 32, 8, 2);
    write_ragged_fixture(&p1, 32, 8, 2);

    // Identity on file 0; file 1's genes shifted into a disjoint global block,
    // so a row attributed to the wrong file is visible in the indices.
    let t0: Vec<i32> = (0..8).collect();
    let t1: Vec<i32> = (0..8).map(|g| g + 8).collect();
    let loader = SparseCellSetLoader::new(
        vec![open(&p0), open(&p1)],
        8,
        None,
        4,
        Some(vec![t0, t1]),
        Some(16),
        false,
        false,
        0.0,
        None,
        false,
        None,
    )
    .unwrap();

    // Interleaved files, several rows each, one row repeated across files.
    let plan = SparseCellSetPlan {
        file_ids: vec![0, 1, 0, 1, 1, 0, 1, 0],
        rows: vec![3, 3, 11, 20, 3, 27, 11, 3],
        role_tags: (0..8).collect(),
        set_offsets: vec![0, 4, 8],
    };

    let a = loader.clone();
    let whole = a
        .gather_whole_plan(&a.engine, &plan, Some(Admit::All))
        .unwrap();
    let b = SparseCellSetLoader::new(
        vec![open(&p0), open(&p1)],
        8,
        None,
        4,
        Some(vec![(0..8).collect(), (0..8).map(|g| g + 8).collect()]),
        Some(16),
        false,
        false,
        0.0,
        None,
        false,
        None,
    )
    .unwrap();
    let per_set = b
        .gather_per_set(&b.engine, &plan, Some(Admit::All))
        .unwrap();
    assert_eq!(batch_fields(&whole), batch_fields(&per_set));

    // Every position's genes must sit in its own file's global block.
    for (j, &fid) in plan.file_ids.iter().enumerate() {
        let (idx, _) = batch_row(&whole, j);
        let in_block = idx.iter().all(|&g| if fid == 0 { g < 8 } else { g >= 8 });
        assert!(in_block, "position {j} (file {fid}) got genes {idx:?}");
    }
    assert_eq!(whole.file_ids, plan.file_ids);
    assert_eq!(whole.cell_indices, plan.rows);
}

/// `set_offsets` need not cover the whole plan, and the rows it does not cover
/// must not be **read** — not merely not emitted.
///
/// Reading them would move the plan's footprint, and with it the admission
/// verdict and every counter the capture reads, while leaving the output
/// identical. So the claim is asserted on the shard misses, not on the batch.
#[test]
fn rows_no_set_covers_are_not_read() {
    use std::sync::atomic::Ordering as AtomicOrdering;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("partial_cover.scx");
    // Four shards of 16 rows.
    write_ragged_fixture(&path, 64, 16, 4);
    let loader = ragged_loader(&path, None, None, false, false, None);

    // Covered: positions 1..3, rows 1 and 17 (shards 0 and 1). Uncovered: rows
    // 33 and 49, which are the only rows naming shards 2 and 3.
    let plan = plan_of(vec![33, 1, 17, 49], vec![1, 3]);
    let b = loader
        .gather_whole_plan(&loader.engine, &plan, Some(Admit::All))
        .unwrap();
    assert_eq!(b.cell_indices, vec![1, 17]);
    assert_eq!(b.shape.0, 2);

    let m = loader.cache_metrics();
    assert_eq!(
        m.misses.load(AtomicOrdering::Relaxed),
        2,
        "only the two shards the emitted rows name should have been decoded"
    );
}

/// The occurrence table itself: dedup across sets **and** within one.
///
/// Asserted directly rather than through a gather, because dedup is an
/// optimisation and not a behaviour — an executor that deduplicated nothing
/// still produces a byte-identical batch, so no output assertion anywhere can
/// see it. (Confirmed by mutation: disabling the dedup outright leaves all 525
/// other tests green.)
#[test]
fn the_occurrence_table_dedups_across_sets_and_within_them() {
    let plan = SparseCellSetPlan {
        file_ids: vec![0, 1, 0, 0, 1, 0],
        //            ^  ^  ^  ^  ^  ^
        // (0,9) twice — positions 0 and 3, in different sets; (1,9) is a
        // DIFFERENT pair and must not collide with it; (0,4) twice inside set 1.
        rows: vec![9, 9, 4, 9, 2, 4],
        role_tags: vec![0; 6],
        set_offsets: vec![0, 3, 6],
    };
    let occ = Occurrences::build(&plan, 0, 6);
    assert_eq!(occ.slots, vec![(0, 9), (1, 9), (0, 4), (1, 2)]);
    assert_eq!(occ.slot_of_pos, vec![0, 1, 2, 0, 3, 2]);
    assert_eq!(occ.by_file, vec![(0, vec![0, 2]), (1, vec![1, 3])]);
    // slot -> (bucket, local row in that bucket's read)
    assert_eq!(occ.slot_loc, vec![(0, 0), (1, 0), (0, 1), (1, 1)]);

    // A plan with no repeats maps every position to its own slot.
    let clean = SparseCellSetPlan {
        file_ids: vec![0, 0, 0],
        rows: vec![7, 1, 4],
        role_tags: vec![0; 3],
        set_offsets: vec![0, 3],
    };
    let occ = Occurrences::build(&clean, 0, 3);
    assert_eq!(occ.slot_of_pos, vec![0, 1, 2]);

    // And the table covers the EMITTED range only.
    let occ = Occurrences::build(&plan, 1, 4);
    assert_eq!(occ.slots, vec![(1, 9), (0, 4), (0, 9)]);
    assert_eq!(occ.slot_of_pos, vec![0, 1, 2]);
}

/// W11: the batch executor's unique-row read is charged **only** on the
/// configurations that must take the general path.
///
/// A single-file raw-local loader is handed the read's own buffers as the
/// batch, so it holds one; a remap, a downsample or a manifest of more than one
/// file forces the assemble-from-a-second-buffer path on every plan, so those
/// hold two. The charge is one batch either way, which is the ceiling: a plan's
/// unique rows are at most its rows.
#[test]
fn the_unique_row_read_is_charged_only_where_it_can_happen() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("t0.scx");
    let p1 = dir.path().join("t1.scx");
    write_fixture(&p0, 8192, 64, 4);
    write_fixture(&p1, 8192, 64, 4);

    let build = |paths: Vec<&std::path::Path>,
                 remap: Option<Vec<Vec<i32>>>,
                 downsample: Option<crate::downsample::DownsampleConfig>,
                 max_plan_rows: Option<usize>| {
        let n = paths.len();
        SparseCellSetLoader::new(
            paths.into_iter().map(open).collect(),
            8,
            None,
            4,
            remap,
            if n > 1 { Some(64) } else { None },
            false,
            false,
            0.0,
            downsample,
            false,
            max_plan_rows,
        )
        .unwrap()
    };

    let bd = |l: &StdArc<SparseCellSetLoader>| l.budget_breakdown();

    // The fast configuration: one file, raw-local. One buffer, so no transient.
    let fast = build(vec![&p0], None, None, Some(4096));
    assert!(
        bd(&fast).batch_buffer_bytes > 0,
        "premise: the batch is charged"
    );
    assert_eq!(bd(&fast).transient_bytes, 0);

    // Each of the three facts that forces the general path, on its own.
    let two_files = build(
        vec![&p0, &p1],
        Some(vec![(0..64).collect(), (0..64).collect()]),
        None,
        Some(4096),
    );
    assert_eq!(
        bd(&two_files).transient_bytes,
        bd(&two_files).batch_buffer_bytes,
        "a multi-file manifest assembles from a second buffer"
    );
    let remapped = build(vec![&p0], Some(vec![(0..64).collect()]), None, Some(4096));
    assert_eq!(
        bd(&remapped).transient_bytes,
        bd(&remapped).batch_buffer_bytes
    );
    let downsampled = build(
        vec![&p0],
        None,
        Some(ds_cfg(
            100,
            crate::downsample::DownsampleMethod::Multinomial,
            1,
            vec![7],
        )),
        Some(4096),
    );
    assert_eq!(
        bd(&downsampled).transient_bytes,
        bd(&downsampled).batch_buffer_bytes
    );

    // And nothing is charged at all without `max_plan_rows`, on either shape —
    // the byte-identity promise the term was added under.
    let uncharged = build(vec![&p0], Some(vec![(0..64).collect()]), None, None);
    assert_eq!(bd(&uncharged).batch_buffer_bytes, 0);
    assert_eq!(bd(&uncharged).transient_bytes, 0);
}

/// Both executors refuse a cross-file **set** without remap tables, with the
/// same message.
///
/// They refuse at different moments and that is deliberate: the per-set walk
/// discovers it inside the set loop, having already read every earlier set, and
/// the batch executor checks every set before any read. Same error, same words;
/// only the wasted I/O differs. Nothing compared the two, and the message is
/// what `test_gather_raises_the_same_errors_as_the_iterator` asserts on the
/// Python side.
#[test]
fn both_executors_refuse_a_cross_file_set_identically() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("x0.scx");
    let p1 = dir.path().join("x1.scx");
    write_ragged_fixture(&p0, 32, 8, 2);
    write_ragged_fixture(&p1, 32, 8, 2);
    let mk = || {
        SparseCellSetLoader::new(
            vec![open(&p0), open(&p1)],
            8,
            None,
            4,
            /*remap*/ None,
            None,
            false,
            false,
            0.0,
            None,
            false,
            None,
        )
        .unwrap()
    };
    // Set 0 is single-file and would be gathered before the per-set walk ever
    // looks at set 1 — which is the moment the two executors differ.
    let plan = SparseCellSetPlan {
        file_ids: vec![0, 0, 0, 1],
        rows: vec![1, 2, 3, 4],
        role_tags: vec![0; 4],
        set_offsets: vec![0, 2, 4],
    };
    let a = mk();
    let whole = a
        .gather_whole_plan(&a.engine, &plan, Some(Admit::All))
        .unwrap_err()
        .to_string();
    let b = mk();
    let per_set = b
        .gather_per_set(&b.engine, &plan, Some(Admit::All))
        .unwrap_err()
        .to_string();
    assert_eq!(whole, per_set);
    assert!(whole.contains("cross-file cell set"), "{whole}");

    // And the batch executor refused before touching a shard, where the
    // per-set walk had already decoded set 0's.
    use std::sync::atomic::Ordering as AtomicOrdering;
    assert_eq!(a.cache_metrics().misses.load(AtomicOrdering::Relaxed), 0);
    assert!(b.cache_metrics().misses.load(AtomicOrdering::Relaxed) > 0);
}

/// The routing rule: which plans have to be assembled from a separate read.
///
/// This is the phase's one real configuration decision and it was made by
/// measurement, so it is pinned directly rather than inferred from a timing.
/// Deduplicating a raw-local single-file plan forces the assembly — a second
/// batch-sized allocation and a second full copy — to save one memcpy per
/// repeated row from an already-resident shard, and the A/B priced that at
/// 0.85x on `gather_random`.
#[test]
fn only_a_multi_file_or_length_changing_plan_is_assembled() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("r0.scx");
    let p1 = dir.path().join("r1.scx");
    write_ragged_fixture(&p0, 32, 8, 2);
    write_ragged_fixture(&p1, 32, 8, 2);

    let one_file = SparseCellSetPlan {
        file_ids: vec![0; 4],
        // Repeats on purpose: a repeat is NOT a reason to assemble.
        rows: vec![3, 7, 3, 7],
        role_tags: vec![0; 4],
        set_offsets: vec![0, 2, 4],
    };
    let two_files = SparseCellSetPlan {
        file_ids: vec![0, 0, 1, 1],
        rows: vec![3, 7, 3, 7],
        role_tags: vec![0; 4],
        set_offsets: vec![0, 2, 4],
    };

    let raw = SparseCellSetLoader::new(
        vec![open(&p0), open(&p1)],
        8,
        None,
        4,
        None,
        None,
        false,
        false,
        0.0,
        None,
        false,
        None,
    )
    .unwrap();
    assert!(
        !raw.plan_needs_assembly(&one_file, 0, 4),
        "raw-local, one file"
    );
    assert!(raw.plan_needs_assembly(&two_files, 0, 4), "two files");

    // `normalize` / `log1p` are elementwise, so they do NOT force it.
    let scaled = SparseCellSetLoader::new(
        vec![open(&p0), open(&p1)],
        8,
        None,
        4,
        None,
        None,
        /*normalize*/ true,
        /*log1p*/ true,
        1e4,
        None,
        false,
        None,
    )
    .unwrap();
    assert!(
        !scaled.plan_needs_assembly(&one_file, 0, 4),
        "value-only transforms"
    );

    // A remap drops and coalesces; a downsample truncates. Both can shrink a
    // row, so neither can be written where it landed.
    let remapped = SparseCellSetLoader::new(
        vec![open(&p0), open(&p1)],
        8,
        None,
        4,
        Some(vec![(0..8).collect(), (0..8).collect()]),
        Some(8),
        false,
        false,
        0.0,
        None,
        false,
        None,
    )
    .unwrap();
    assert!(remapped.plan_needs_assembly(&one_file, 0, 4), "remap");

    let downsampled = SparseCellSetLoader::new(
        vec![open(&p0), open(&p1)],
        8,
        None,
        4,
        None,
        None,
        false,
        false,
        0.0,
        Some(ds_cfg(
            2,
            crate::downsample::DownsampleMethod::Multinomial,
            5,
            vec![1, 2],
        )),
        false,
        None,
    )
    .unwrap();
    assert!(
        downsampled.plan_needs_assembly(&one_file, 0, 4),
        "downsample"
    );

    // And the rule the budget charges on is the loader-level form of the same
    // one: it must not charge the configuration that never assembles.
    assert_eq!(raw.budget_breakdown().transient_bytes, 0);
}
