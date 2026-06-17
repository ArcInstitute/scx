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
    assert!(n_obs % n_shards == 0);
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
        usize::MAX,
        /*lookahead*/ 4,
        /*remap*/ None,
        /*n_global_genes*/ None,
        false,
        false,
        0.0,
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
        usize::MAX,
        2,
        None,
        None,
        false,
        false,
        0.0,
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
        usize::MAX,
        4,
        Some(remap),
        /*n_global_genes*/ Some(108),
        false,
        false,
        0.0,
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
        usize::MAX,
        2,
        None,
        None,
        false,
        false,
        0.0,
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
fn gather_collated_emits_stacked_tensors_matching_kernel() {
    use crate::sparse_cellset_collate::PreprocessMode;
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("f0.scx");
    write_fixture(&p0, 32, 8, 4); // row r: col r%8, val (r+1)&0xFF
                                  // Identity remap (local g → global g) so gene ids are global.
    let loader = SparseCellSetLoader::new(
        vec![open(&p0)],
        8,
        usize::MAX,
        4,
        Some(vec![(0..8).collect::<Vec<i32>>()]),
        Some(8),
        false,
        false,
        0.0,
    )
    .unwrap();

    // One set, two cells: row 2 (col2,val3), row 5 (col5,val6).
    let k_dec = 4;
    let plan = CollatedCellSetPlan {
        base: SparseCellSetPlan {
            file_ids: vec![0, 0],
            rows: vec![2, 5],
            role_tags: vec![0, 0],
            set_offsets: vec![0, 2],
        },
        k_dec,
        query_gene_ids: vec![2, 5, 7, 0], // shared across the set
        enc_mask_positions: Vec::new(),   // pert-style: no encoder masking
        hide_readout: vec![0, 0],
        n_measured: vec![8],
    };
    let scalars = CollateScalars {
        k_enc: 4,
        mode: PreprocessMode::Log1pRaw,
        target_sum: 1e4,
        n_genes_total: 8,
        lib_size_redef: false,
    };
    let b = loader
        .iter_with_plans_collated(vec![Ok(plan)].into_iter(), 4, scalars)
        .next()
        .unwrap()
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
fn gather_collated_requires_remap() {
    let dir = tempfile::tempdir().unwrap();
    let loader = malformed_plan_loader(dir.path()); // built without remap
    let plan = CollatedCellSetPlan {
        base: SparseCellSetPlan {
            file_ids: vec![0],
            rows: vec![1],
            role_tags: vec![0],
            set_offsets: vec![0, 1],
        },
        k_dec: 2,
        query_gene_ids: vec![0, 1],
        enc_mask_positions: Vec::new(),
        hide_readout: vec![0],
        n_measured: vec![8],
    };
    let scalars = CollateScalars {
        k_enc: 2,
        mode: crate::sparse_cellset_collate::PreprocessMode::Log1pRaw,
        target_sum: 1e4,
        n_genes_total: 8,
        lib_size_redef: false,
    };
    let got = loader
        .iter_with_plans_collated(vec![Ok(plan)].into_iter(), 2, scalars)
        .next()
        .unwrap();
    assert!(matches!(got, Err(LoaderError::ConfigError { .. })));
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
