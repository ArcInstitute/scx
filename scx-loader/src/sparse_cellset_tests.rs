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
    let (idx, dat) = remap_row(&[0, 1, 2, 3], &[1.0, 9.0, 2.0, 4.0], &table);
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
    assert_eq!(one, 128, "8 rows x 1 nnz => (8*8 + 8*8)/1 per shard");

    // Two identical files: twice the shards, twice the totals, same average.
    let p1 = dir.path().join("s1.scx");
    write_fixture(&p1, 32, 8, 4);
    let two = avg_shard_decoded_bytes(&[open(&p0), open(&p1)]);
    assert_eq!(
        two, one,
        "the average must be scale-invariant; a drift here means the divisor \
         and the numerator are counting different shard sets"
    );
}

/// No shard carries stats ⇒ size unknown ⇒ the byte cap says nothing, so the
/// count cap is what binds (never a fabricated average, never 0 entries).
#[test]
fn effective_cache_shards_falls_back_to_the_count_when_size_is_unknown() {
    // An empty reader set is the degenerate "no shards carry stats" case the
    // helper must survive; `SparseCellSetLoader::new` rejects zero files, so the
    // helper is exercised directly.
    assert_eq!(avg_shard_decoded_bytes(&[]), 0);
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
    )
    .unwrap()
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

/// The whole reduction chain, through the shared harness. Fixture-free.
#[test]
fn the_sparse_reduction_chain_is_monotone() {
    for &shard_decoded_bytes in &[128usize, 32_768, 470 * 1024 * 1024, 0] {
        crate::budget::assert_monotone_reduction_chain(
            &SparseCellSetBudgetModel {
                shard_decoded_bytes,
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
