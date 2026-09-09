use super::*;
use crate::pipeline::QueryPipeline;
use arrow::array::{Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;
use std::sync::Arc;

fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, n_vars, nnz, 16384, 0, 0)
}

fn sample_obs(n: usize) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, true),
    ]);
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let types: Vec<&str> = (0..n)
        .map(|i| match i % 3 {
            0 => "T cell",
            1 => "B cell",
            _ => "NK cell",
        })
        .collect();
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(types)),
        ],
    )
    .unwrap()
}

fn sample_var(n: usize) -> RecordBatch {
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn write_test_file(dir: &tempfile::TempDir, n_obs: usize, n_vars: usize) -> std::path::PathBuf {
    let path = dir.path().join("test.scx");
    let header = sample_header(n_obs as u64, n_vars as u64, (n_obs * 2) as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..n_obs {
        let col0 = (row * 2) % n_vars;
        let col1 = (row * 2 + 1) % n_vars;
        let (c0, c1) = if col0 < col1 {
            (col0, col1)
        } else {
            (col1, col0)
        };
        indices.push(c0 as u32);
        indices.push(c1 as u32);
        values.push(((row + 1) % 256) as u8);
        values.push(((row + 2) % 256) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }

    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer.finish().unwrap();
    path
}

// -----------------------------------------------------------------------
// A three-CSR-shard fixture with irregular geometry.
//
// Row counts [3, 1, 5]: a one-row shard in the middle makes an off-by-one in a
// running offset visible, which uniform shards cannot. Every other CSR fixture
// in this file is either a single shard (so it cannot see the merge at all) or
// four shards of three rows each (so a swapped pair looks like a shift). The
// `keep_me` obs column is "no" for exactly the middle shard's one row, so an
// `Eq` filter can empty a shard *between* two live ones — the case where the
// merge's running offset has to skip a gap, and "no" again for row 5, which is
// interior to the last shard — so that shard is *partially* kept and the
// copying slow path runs. Without an interior drop every live shard's mask is
// all-true, and a fast path that ignored its mask entirely would pass.
// -----------------------------------------------------------------------

const IRREG_SHARD_ROWS: [usize; 3] = [3, 1, 5];
const IRREG_N_VARS: usize = 6;
const IRREG_N_OBS: usize = 9;

/// Row `row`'s two stored entries: `(ascending columns, values)`.
///
/// Values are `10*(row+1)` and `10*(row+1)+1`, so every row's payload is
/// distinct from every other row's and a reversed or duplicated shard shows up
/// as a value mismatch rather than only a shape one.
fn irreg_row(row: usize) -> ([u32; 2], [u8; 2]) {
    let a = (row % IRREG_N_VARS) as u32;
    let b = ((row + 2) % IRREG_N_VARS) as u32;
    let (c0, c1) = if a < b { (a, b) } else { (b, a) };
    let (v0, v1) = if a < b {
        ((10 * (row + 1)) as u8, (10 * (row + 1) + 1) as u8)
    } else {
        ((10 * (row + 1) + 1) as u8, (10 * (row + 1)) as u8)
    };
    ([c0, c1], [v0, v1])
}

/// Dense image of the fixture's rows `rows`, columns `cols` (in that order).
fn irreg_dense(rows: &[usize], cols: &[u32]) -> Vec<Vec<f32>> {
    rows.iter()
        .map(|&row| {
            let (row_cols, vals) = irreg_row(row);
            cols.iter()
                .map(|&c| {
                    row_cols
                        .iter()
                        .position(|&rc| rc == c)
                        .map(|k| vals[k] as f32)
                        .unwrap_or(0.0)
                })
                .collect()
        })
        .collect()
}

/// Dense image of a collected `X`.
fn dense_of(x: &scx_sparse::ScxCsr) -> Vec<Vec<f32>> {
    let n_cols = x.shape.1;
    (0..x.shape.0)
        .map(|r| {
            let mut out = vec![0.0f32; n_cols];
            for k in x.indptr[r] as usize..x.indptr[r + 1] as usize {
                out[x.indices[k] as usize] = x.data[k];
            }
            out
        })
        .collect()
}

fn irreg_obs() -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("keep_me", DataType::Utf8, false),
    ]);
    let ids: Vec<String> = (0..IRREG_N_OBS).map(|i| format!("cell_{i}")).collect();
    // Row 3 is the middle shard's only row; row 5 is interior to the last one.
    let keep: Vec<&str> = (0..IRREG_N_OBS)
        .map(|i| if i == 3 || i == 5 { "no" } else { "yes" })
        .collect();
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(keep)),
        ],
    )
    .unwrap()
}

fn write_irregular_multishard_file(dir: &tempfile::TempDir) -> std::path::PathBuf {
    let path = dir.path().join("irregular_shards.scx");
    let header = sample_header(
        IRREG_N_OBS as u64,
        IRREG_N_VARS as u64,
        (IRREG_N_OBS * 2) as u64,
    );
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&irreg_obs()).unwrap();
    writer.write_var(&sample_var(IRREG_N_VARS)).unwrap();

    let mut row_start = 0usize;
    for &n_rows in IRREG_SHARD_ROWS.iter() {
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for row in row_start..row_start + n_rows {
            let (cols, vals) = irreg_row(row);
            indices.extend_from_slice(&cols);
            values.extend_from_slice(&vals);
            indptr.push(indptr.last().unwrap() + 2);
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
        row_start += n_rows;
    }
    writer.finish().unwrap();
    path
}

/// The assembled matrix survives an irregular shard layout, a shard emptied in
/// the middle, a projection and a limit.
///
/// This is the merge's test. It assembles by taking the first decoded shard's
/// buffers as the accumulator and appending the rest into them, so the running
/// offset now starts from the accumulator's own last entry rather than from
/// zero — an arithmetic change that a uniform-shard fixture, or one where the
/// first shard is always shard 0, cannot distinguish from the original.
#[test]
fn collect_assembles_shards_of_unequal_size_in_row_order() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_irregular_multishard_file(&dir);
    let all_cols: Vec<u32> = (0..IRREG_N_VARS as u32).collect();

    // Premise: three shards of the declared unequal sizes, and no two rows
    // carry the same payload.
    let probe = QueryPipeline::open(&path).unwrap().collect().unwrap();
    assert_eq!(probe.total_shards, 3, "fixture must be three CSR shards");
    assert_ne!(
        IRREG_SHARD_ROWS[0], IRREG_SHARD_ROWS[1],
        "equal shard sizes would hide an offset error"
    );
    let rows_all: Vec<usize> = (0..IRREG_N_OBS).collect();
    assert_eq!(dense_of(&probe.x), irreg_dense(&rows_all, &all_cols));
    assert_eq!(probe.matched_rows, IRREG_N_OBS);

    // A filter that empties the *middle* shard and partially keeps the last
    // one: the merge has to carry its running offset across a shard that
    // contributes nothing, and the row filter has to actually filter.
    let last_shard_start = IRREG_SHARD_ROWS[0] + IRREG_SHARD_ROWS[1];
    assert!(
        last_shard_start < 5 && 5 < IRREG_N_OBS,
        "row 5 must be interior to the last shard, or no shard is partially \
         kept and a filter that ignored its mask would still pass"
    );
    let kept = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("keep_me == 'yes'")
        .unwrap()
        .collect()
        .unwrap();
    let rows_kept: Vec<usize> = (0..IRREG_N_OBS).filter(|&r| r != 3 && r != 5).collect();
    assert_eq!(rows_kept.len(), 7);
    assert_eq!(dense_of(&kept.x), irreg_dense(&rows_kept, &all_cols));

    // A projection, requested out of ascending order so the output-column
    // reorder runs too.
    let projected = QueryPipeline::open(&path)
        .unwrap()
        .select_genes(vec![4, 0])
        .collect()
        .unwrap();
    assert_eq!(
        dense_of(&projected.x),
        irreg_dense(&rows_all, &[4, 0]),
        "projected columns must follow the requested order across all shards"
    );

    // A limit that lands exactly on the second shard's only row, so the
    // decoded prefix is two shards of unequal length.
    let limited = QueryPipeline::open(&path)
        .unwrap()
        .limit(4)
        .collect()
        .unwrap();
    assert_eq!(
        dense_of(&limited.x),
        irreg_dense(&[0, 1, 2, 3], &all_cols),
        "the limit prefix spans shard 0 (3 rows) and shard 1 (1 row)"
    );
}

// -----------------------------------------------------------------------
// F3 Tests: filter_csr_rows
// -----------------------------------------------------------------------

#[test]
fn filter_alternating_rows() {
    // 4-row CSR: keep rows 0 and 2
    let indptr = vec![0i64, 2, 5, 7, 10];
    let indices = vec![0i32, 1, 0, 1, 2, 1, 3, 0, 2, 3];
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let keep = vec![true, false, true, false];

    let (new_ip, new_idx, new_data) = filter_csr_rows(&indptr, &indices, &data, &keep).unwrap();
    assert_eq!(new_ip, vec![0, 2, 4]);
    assert_eq!(new_idx, vec![0, 1, 1, 3]);
    assert_eq!(new_data, vec![1.0, 2.0, 6.0, 7.0]);
}

#[test]
fn filter_keep_all() {
    let indptr = vec![0i64, 2, 5];
    let indices = vec![0i32, 1, 0, 1, 2];
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let keep = vec![true, true];

    let (new_ip, new_idx, new_data) = filter_csr_rows(&indptr, &indices, &data, &keep).unwrap();
    assert_eq!(new_ip, vec![0, 2, 5]);
    assert_eq!(new_idx, indices);
    assert_eq!(new_data, data);
}

#[test]
fn filter_keep_none() {
    let indptr = vec![0i64, 2, 5];
    let indices = vec![0i32, 1, 0, 1, 2];
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let keep = vec![false, false];

    let (new_ip, new_idx, new_data) = filter_csr_rows(&indptr, &indices, &data, &keep).unwrap();
    assert_eq!(new_ip, vec![0]);
    assert!(new_idx.is_empty());
    assert!(new_data.is_empty());
}

/// An all-true mask must hand the decoded buffers back, not an equal copy of
/// them.
///
/// Pointer identity is the assertion because value equality cannot tell the two
/// apart, and "same values" is not the claim — `filter_keep_all` above already
/// pins that for the borrowing form. What this pins is that an unfiltered query
/// stops paying ~8 B/nnz per shard to rebuild what it just decoded.
#[test]
fn filter_owned_keeps_all_rows_by_moving_them() {
    let indptr = vec![0i64, 2, 5];
    let indices = vec![0i32, 1, 0, 1, 2];
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let keep = vec![true, true];

    let (ip_ptr, idx_ptr, data_ptr) = (indptr.as_ptr(), indices.as_ptr(), data.as_ptr());
    let (new_ip, new_idx, new_data) =
        crate::collect::filter_csr_rows_owned(indptr, indices, data, &keep).unwrap();

    assert_eq!(new_ip.as_ptr(), ip_ptr, "indptr was copied, not moved");
    assert_eq!(new_idx.as_ptr(), idx_ptr, "indices were copied, not moved");
    assert_eq!(new_data.as_ptr(), data_ptr, "data was copied, not moved");
    // And the contract the move has to satisfy to be an identity at all.
    assert_eq!(new_ip, vec![0, 2, 5]);
    assert_eq!(new_idx, vec![0, 1, 0, 1, 2]);
    assert_eq!(new_data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

/// The owned filter still rejects a mask that does not cover the shard — in the
/// direction that an `all()` test would wave through.
///
/// A mask of every `true` but the wrong *length* is the trap: `all()` over a
/// short mask is vacuously true, so a fast path that tested it before the length
/// check would return the shard's every row while the obs half carried only the
/// rows the mask described. The check has to come first, and only the short
/// direction proves it did.
#[test]
fn filter_owned_rejects_a_short_all_true_mask_rather_than_moving_it() {
    let indptr = vec![0i64, 2, 5, 7];
    let indices = vec![0i32, 1, 0, 1, 2, 1, 3];
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    let short = vec![true, true];
    let err = crate::collect::filter_csr_rows_owned(
        indptr.clone(),
        indices.clone(),
        data.clone(),
        &short,
    )
    .expect_err("a 2-entry mask must not be accepted for a 3-row shard")
    .to_string();
    assert!(err.contains("keep mask covers 2 rows"), "unexpected: {err}");

    let long = vec![true, true, true, true];
    assert!(
        crate::collect::filter_csr_rows_owned(indptr, indices, data, &long).is_err(),
        "a 4-entry mask must not be accepted for a 3-row shard either"
    );
}

/// A filtered shard's buffers are reserved once, for exactly what it keeps.
///
/// The capacities are compared against what `Vec::with_capacity(n)` *itself*
/// returns for the same `n`, not against `n`. `with_capacity` promises only "at
/// least", so asserting `capacity() == len()` would be pinning undocumented
/// allocator behaviour and could fail on a conforming stdlib that over-allocates
/// — a red test with no regression behind it. Comparing against the reference
/// value states the actual claim ("this buffer was reserved up front for its
/// final size") and stays true however much `with_capacity` rounds up, while
/// still failing on the growth-from-empty it replaced: a doubling `Vec` lands on
/// 8 for a length of 5 or 7, which is not what `with_capacity(5)` or
/// `with_capacity(7)` gives.
#[test]
fn filter_sizes_its_buffers_to_the_kept_rows() {
    // 7 rows / 11 nnz, of which 4 rows and 7 non-zeros are kept: neither 5
    // (`indptr`) nor 7 is a power of two, so growth-from-empty cannot land on
    // either by coincidence — it reaches 8 for both, as does the old
    // `mask_len + 1` indptr reservation (8 = 7 + 1).
    let indptr = vec![0i64, 2, 3, 5, 6, 8, 9, 11];
    let indices = vec![0i32, 1, 2, 0, 3, 1, 2, 3, 0, 1, 2];
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    let keep = vec![true, true, true, false, true, false, false];

    let (new_ip, new_idx, new_data) = filter_csr_rows(&indptr, &indices, &data, &keep).unwrap();

    assert_eq!(new_ip.len(), 5, "4 kept rows + the leading 0");
    assert_eq!(new_idx.len(), 7, "2 + 1 + 2 + 2 kept non-zeros");
    assert_eq!(
        new_ip.capacity(),
        Vec::<i64>::with_capacity(5).capacity(),
        "indptr was not reserved for its 4 kept rows + 1"
    );
    assert_eq!(
        new_idx.capacity(),
        Vec::<i32>::with_capacity(7).capacity(),
        "indices were not reserved for their 7 kept non-zeros"
    );
    assert_eq!(
        new_data.capacity(),
        Vec::<f32>::with_capacity(7).capacity(),
        "data was not reserved for its 7 kept non-zeros"
    );
    // The values are still the kept rows', not just the right count of them.
    assert_eq!(new_ip, vec![0, 2, 3, 5, 7]);
    assert_eq!(new_idx, vec![0, 1, 2, 0, 3, 2, 3]);
    assert_eq!(new_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0]);
}

/// A keep mask that disagrees with the CSR row count is a corrupt or
/// truncated file, not a request to guess.
///
/// This used to be a `debug_assert!` followed by `min(keep_mask.len(),
/// n_rows)`, so in release — the builds users run — the *shorter* case
/// silently dropped every row past the end of the mask and returned a
/// truncated query result. That direction never panicked, which is exactly
/// why it needed a test rather than an assertion: a wrong answer looks like
/// an answer.
#[test]
fn filter_rejects_a_mask_that_disagrees_with_the_csr() {
    let indptr = vec![0i64, 2, 5, 7];
    let indices = vec![0i32, 1, 0, 1, 2, 1, 3];
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    // Shorter than the CSR: the silent-truncation direction. Pre-fix this
    // returned Ok with 1 row where 3 were present.
    let short = vec![true];
    let err = filter_csr_rows(&indptr, &indices, &data, &short)
        .expect_err("a short mask must error, not truncate");
    let msg = err.to_string();
    assert!(
        msg.contains('1') && msg.contains('3'),
        "must report both counts: {msg}"
    );

    // Longer than the CSR: would have indexed past the end of indptr.
    let long = vec![true, true, true, true, true];
    assert!(filter_csr_rows(&indptr, &indices, &data, &long).is_err());

    // Control: an exactly-matching mask still filters.
    let ok = vec![true, false, true];
    let (ip, _, _) = filter_csr_rows(&indptr, &indices, &data, &ok).unwrap();
    assert_eq!(ip.len(), 3, "two kept rows");
}

// -----------------------------------------------------------------------
// F2 Tests: Pipeline execution via QueryPipeline::collect()
// -----------------------------------------------------------------------

#[test]
fn collect_no_predicates_returns_all() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, 12, 5);
    let result = QueryPipeline::open(&path).unwrap().collect().unwrap();
    assert_eq!(result.x.n_rows(), 12);
    assert_eq!(result.x.n_cols(), 5);
    assert_eq!(result.obs.num_rows(), 12);
    assert_eq!(result.var.num_rows(), 5);
    assert_eq!(result.total_shards, 1);
}

#[test]
fn collect_filter_obs_returns_subset() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, 12, 5);
    let result = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'T cell'")
        .unwrap()
        .collect()
        .unwrap();
    // cells 0, 3, 6, 9 are "T cell" (i % 3 == 0)
    assert_eq!(result.x.n_rows(), 4);
    assert_eq!(result.obs.num_rows(), 4);
    // Check that filtered obs matches
    let cell_types = result
        .obs
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for i in 0..cell_types.len() {
        assert_eq!(cell_types.value(i), "T cell");
    }
}

#[test]
fn collect_gene_projection() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, 12, 10);
    let result = QueryPipeline::open(&path)
        .unwrap()
        .select_genes(vec![0, 3, 7])
        .collect()
        .unwrap();
    assert_eq!(result.x.n_cols(), 3);
    assert_eq!(result.var.num_rows(), 3);
    // Check var metadata has correct gene IDs
    let gene_ids = result
        .var
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(gene_ids.value(0), "gene_0");
    assert_eq!(gene_ids.value(1), "gene_3");
    assert_eq!(gene_ids.value(2), "gene_7");
}

#[test]
fn collect_gene_projection_preserves_requested_order() {
    // F4: select_genes must return columns in the caller's requested order,
    // not ascending index order.
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, 12, 10);

    let requested = vec![7u32, 3, 0];
    let proj = QueryPipeline::open(&path)
        .unwrap()
        .select_genes(requested.clone())
        .collect()
        .unwrap();
    assert_eq!(proj.x.n_cols(), 3);
    assert_eq!(proj.var.num_rows(), 3);

    // var rows follow the requested order.
    let gene_ids = proj
        .var
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(gene_ids.value(0), "gene_7");
    assert_eq!(gene_ids.value(1), "gene_3");
    assert_eq!(gene_ids.value(2), "gene_0");

    // X columns match: projected column j equals full column requested[j].
    let full = QueryPipeline::open(&path).unwrap().collect().unwrap();
    let dense_full = full.x.to_dense().unwrap(); // row-major, 12×10
    let dense_proj = proj.x.to_dense().unwrap(); // row-major, 12×3
    for r in 0..12 {
        for (j, &g) in requested.iter().enumerate() {
            assert_eq!(
                dense_proj[r * 3 + j],
                dense_full[r * 10 + g as usize],
                "row {r} col {j} (gene {g})"
            );
        }
    }
}

#[test]
fn collect_gene_projection_ascending_unchanged() {
    // An already-ascending-unique request takes the no-reorder fast path:
    // output columns are exactly the requested genes, in order, matching the
    // corresponding columns of the full (unprojected) matrix.
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, 12, 10);
    let requested = vec![0u32, 3, 7];
    let proj = QueryPipeline::open(&path)
        .unwrap()
        .select_genes(requested.clone())
        .collect()
        .unwrap();
    assert_eq!(proj.x.n_cols(), 3);

    let full = QueryPipeline::open(&path).unwrap().collect().unwrap();
    let dense_full = full.x.to_dense().unwrap(); // 12×10
    let dense_proj = proj.x.to_dense().unwrap(); // 12×3
    for r in 0..12 {
        for (j, &g) in requested.iter().enumerate() {
            assert_eq!(dense_proj[r * 3 + j], dense_full[r * 10 + g as usize]);
        }
    }
}

#[test]
fn collect_gene_projection_order_with_var_predicate() {
    // PR #242 review: select_genes order must survive intersection with a
    // var predicate that drops one of the requested genes.
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, 12, 10);
    let result = QueryPipeline::open(&path)
        .unwrap()
        .select_genes(vec![7, 3, 0]) // non-ascending
        .filter_var("gene_id != 'gene_3'")
        .unwrap()
        .collect()
        .unwrap();
    // gene_3 removed; the survivors keep the requested order [7, 0].
    assert_eq!(result.x.n_cols(), 2);
    let gene_ids = result
        .var
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(gene_ids.value(0), "gene_7");
    assert_eq!(gene_ids.value(1), "gene_0");
}

#[test]
fn collect_with_normalize_log1p() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, 6, 5);
    let result = QueryPipeline::open(&path)
        .unwrap()
        .with_normalize(1e4)
        .with_log1p()
        .collect()
        .unwrap();
    assert_eq!(result.x.n_rows(), 6);
    // Values should be transformed (no longer raw integers)
    // Each row had 2 non-zero values, after normalize+log1p they should be ln(v/sum*1e4 + 1)
    for row in 0..result.x.n_rows() {
        let start = result.x.indptr[row] as usize;
        let end = result.x.indptr[row + 1] as usize;
        for i in start..end {
            assert!(
                result.x.data[i] > 0.0,
                "fused ops should produce positive values"
            );
            assert!(
                result.x.data[i] < 20.0,
                "ln(10001) ≈ 9.21, values should be reasonable"
            );
        }
    }
}

#[test]
fn collect_empty_result() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, 12, 5);
    // Filter for a cell_type that doesn't exist — but the column exists
    // so the predicate is valid. All cells are T cell, B cell, or NK cell.
    // Use cell_id which is unique to force empty result.
    let result = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_id == 'nonexistent'")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(result.x.n_rows(), 0);
    assert_eq!(result.obs.num_rows(), 0);
}

#[test]
fn collect_with_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, 12, 5);
    let result = QueryPipeline::open(&path)
        .unwrap()
        .limit(3)
        .collect()
        .unwrap();
    assert_eq!(result.x.n_rows(), 3);
    assert_eq!(result.obs.num_rows(), 3);
    // matched_rows must reflect the
    // pre-limit Level-2 match count (12 rows match the no-predicate
    // pipeline), not the post-limit returned count.
    assert_eq!(result.matched_rows, 12);
}

#[test]
fn collect_limit_exceeds_total() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, 6, 5);
    let result = QueryPipeline::open(&path)
        .unwrap()
        .limit(100)
        .collect()
        .unwrap();
    assert_eq!(result.x.n_rows(), 6);
    assert_eq!(result.obs.num_rows(), 6);
    // No truncation occurred — matched_rows must equal returned rows.
    assert_eq!(result.matched_rows, result.x.n_rows());
}

#[test]
fn count_matches_collect_matched_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, 12, 5);
    // No predicate: every cell matches.
    let c = QueryPipeline::open(&path).unwrap().count().unwrap();
    assert_eq!(c.matched_rows, 12);
    let result = QueryPipeline::open(&path).unwrap().collect().unwrap();
    assert_eq!(c.matched_rows, result.matched_rows);
    assert_eq!(c.total_shards, result.total_shards);
    assert_eq!(c.skipped_shards, result.skipped_shards);
    assert_eq!(c.candidate_shard_rows, result.candidate_shard_rows);
}

#[test]
fn collect_reports_shard_value_max() {
    // `write_test_file` writes Uint8 counts (row+1)%256 / (row+2)%256 over
    // 12 rows, so the on-disk max is 13 (row 11). collect() must surface it
    // via QueryResult::max_value for the F4 decode-loss guard.
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, 12, 5);
    let result = QueryPipeline::open(&path).unwrap().collect().unwrap();
    assert_eq!(result.max_value, 13);
}

#[test]
fn count_with_predicate_matches_collect() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, 12, 5);
    // count() and collect().matched_rows must agree for the same predicate,
    // even though count() never decodes X.
    let pipeline = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'T cell'")
        .unwrap();
    let c = pipeline.count().unwrap();
    let result = pipeline.collect().unwrap();
    assert_eq!(c.matched_rows, result.matched_rows);
    assert_eq!(c.matched_rows, result.x.n_rows());
}

#[test]
fn count_ignores_limit() {
    // CLI6: count() reports the true match count regardless of `limit`.
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, 12, 5);
    let no_limit = QueryPipeline::open(&path).unwrap().count().unwrap();
    let with_limit = QueryPipeline::open(&path)
        .unwrap()
        .limit(3)
        .count()
        .unwrap();
    assert_eq!(no_limit.matched_rows, 12);
    assert_eq!(
        with_limit.matched_rows, no_limit.matched_rows,
        "limit must not change the counted match total"
    );
}

#[test]
fn count_empty_result() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, 12, 5);
    let c = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_id == 'nonexistent'")
        .unwrap()
        .count()
        .unwrap();
    assert_eq!(c.matched_rows, 0);
}

#[test]
fn collect_multiple_filter_obs_and_semantics() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, 12, 5);
    // cell_type == 'T cell' AND cell_id == 'cell_0' → only cell_0
    let result = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'T cell'")
        .unwrap()
        .filter_obs("cell_id == 'cell_0'")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(result.x.n_rows(), 1);
    assert_eq!(result.obs.num_rows(), 1);
    let cell_ids = result
        .obs
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(cell_ids.value(0), "cell_0");
}

#[test]
fn collect_obs_row_count_matches_x() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, 12, 5);
    let result = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'B cell'")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(result.obs.num_rows(), result.x.n_rows());
}

#[test]
fn collect_var_row_count_matches_x_cols() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, 12, 10);
    let result = QueryPipeline::open(&path)
        .unwrap()
        .select_genes(vec![1, 5, 9])
        .collect()
        .unwrap();
    assert_eq!(result.var.num_rows(), result.x.n_cols());
}

#[test]
fn collect_full_pipeline() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, 12, 10);
    let result = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'T cell'")
        .unwrap()
        .select_genes(vec![0, 2, 4, 6, 8])
        .with_normalize(1e4)
        .with_log1p()
        .collect()
        .unwrap();
    // T cell: indices 0, 3, 6, 9 → 4 cells
    assert_eq!(result.x.n_rows(), 4);
    assert_eq!(result.x.n_cols(), 5); // 5 projected genes
    assert_eq!(result.obs.num_rows(), 4);
    assert_eq!(result.var.num_rows(), 5);
}

// -----------------------------------------------------------------------
// FILTER-OBS-OOM follow-ups: full-scan limit-pushdown (#1) and no-match
// short-circuit (#2). Both run `plan_and_mask` + `materialize` against a
// *borrowed* pipeline so the per-shard X-decode counter
// (`read_shard_from_entry`) can be inspected after materialisation.
// -----------------------------------------------------------------------

const MS_SHARDS: usize = 4;
const MS_ROWS_PER_SHARD: usize = 3;

/// 4 CSR + 4 obs shards of 3 rows each (12 rows). `cell_type` is indexed
/// and cycles T/B/NK within every shard, so `'T cell'` appears in EVERY
/// shard (0 % catalog skip → the low-selectivity full-scan path), while a
/// value like `'Z'` is absent from the predicate index entirely.
fn write_indexed_multishard_file(dir: &tempfile::TempDir) -> std::path::PathBuf {
    let n_obs = (MS_SHARDS * MS_ROWS_PER_SHARD) as u64;
    let n_vars = 4usize;
    let path = dir.path().join("indexed_multishard.scx");

    let ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    let types: Vec<&str> = (0..n_obs as usize)
        .map(|i| match i % MS_ROWS_PER_SHARD {
            0 => "T cell",
            1 => "B cell",
            _ => "NK cell",
        })
        .collect();
    let obs_schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, false),
    ]);
    let obs = RecordBatch::try_new(
        Arc::new(obs_schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(types)),
        ],
    )
    .unwrap();
    let var = sample_var(n_vars);

    let mut header = sample_header(n_obs, n_vars as u64, n_obs * 2);
    header.shard_target_rows = MS_ROWS_PER_SHARD as u32;
    let mut writer = ScxWriter::new(&path, header).unwrap();

    // Sharded obs aligned with the CSR shards.
    for s in 0..MS_SHARDS {
        let rs = (s * MS_ROWS_PER_SHARD) as u64;
        let slice = obs.slice(rs as usize, MS_ROWS_PER_SHARD);
        writer
            .write_obs_shard(s as u32, rs, MS_ROWS_PER_SHARD as u64, n_obs, &slice)
            .unwrap();
    }
    writer.write_var(&var).unwrap();

    // CSR shards: 1 nnz/row.
    let mut csr_row_ranges = Vec::new();
    for s in 0..MS_SHARDS {
        let rs = (s * MS_ROWS_PER_SHARD) as u64;
        let indptr: Vec<u64> = (0..=MS_ROWS_PER_SHARD as u64).collect();
        let indices: Vec<u32> = (0..MS_ROWS_PER_SHARD as u32)
            .map(|i| i % n_vars as u32)
            .collect();
        let values: Vec<u8> = vec![1u8; MS_ROWS_PER_SHARD];
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                rs,
            )
            .unwrap();
        csr_row_ranges.push((rs, rs + MS_ROWS_PER_SHARD as u64));
    }

    let opts = crate::ConversionPredicateIndexOptions {
        index_obs: vec!["cell_type".to_string()],
        index_var: Vec::new(),
        index_preset: None,
        index_auto_threshold: 1000,
    };
    crate::build_and_write_conversion_predicate_indexes(
        &mut writer,
        &obs,
        &var,
        &csr_row_ranges,
        n_vars,
        &opts,
    )
    .unwrap();
    writer.finish().unwrap();
    path
}

// Only the `debug_assertions`-gated assertions reference this, so it is
// dead code in release builds.
#[cfg(debug_assertions)]
fn x_decode_count(pipeline: &QueryPipeline) -> u64 {
    pipeline
        .reader()
        .as_any()
        .downcast_ref::<scx_format_io::reader::ScxReader>()
        .expect("local reader")
        .debug_counts()
        .read_shard_from_entry
        .load(std::sync::atomic::Ordering::Relaxed)
}

/// Item #1: a low-selectivity predicate matches a row in every shard
/// (0 % skip), so without limit-pushdown `materialize` would decode all 4
/// X shards. With `.limit(1)` the first shard alone fills the budget, so
/// only ONE X shard is decoded.
#[test]
fn limit_pushdown_bounds_full_scan_x_decode() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_indexed_multishard_file(&dir);

    let pipeline = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'T cell'")
        .unwrap()
        .limit(1);
    let pm = plan_and_mask(&pipeline).unwrap();
    // No catalog skip: 'T cell' is present in every shard.
    assert_eq!(pm.skipped_shards, 0);
    assert_eq!(pm.matched_rows, MS_SHARDS); // one T cell per shard
    let r = materialize(&pipeline, pm).unwrap();

    assert_eq!(r.x.n_rows(), 1);
    assert_eq!(r.obs.num_rows(), 1);
    assert_eq!(r.matched_rows, MS_SHARDS, "pre-limit Level-2 count");
    // The X-decode counter is only incremented under `debug_assertions`
    // (compiled away in release), so assert it only when present.
    #[cfg(debug_assertions)]
    assert_eq!(
        x_decode_count(&pipeline),
        1,
        "limit(1) must decode only the first X shard, not all {MS_SHARDS}"
    );
}

/// The over-fix guard, end to end: a file written by the normal conversion
/// path must still be recognised as carrying a complete vocabulary, and a
/// filter naming a value the file does not contain must still short-circuit
/// without reading a single obs shard.
///
/// Gating the short-circuit on completeness is only worth doing if
/// completeness is the ordinary case. If this goes red, the guard has
/// turned every miss on every file into a full obs scan.
#[test]
fn a_converted_file_carries_a_complete_vocabulary_and_still_short_circuits() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_indexed_multishard_file(&dir);

    let pipeline = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'Z'")
        .unwrap();
    let sorted_shards = scan_shards(pipeline.reader().catalog(), pipeline.csr_shard_positions());
    let plan = build_plan(&pipeline, &sorted_shards).unwrap();
    let dict = plan
        .category_dicts
        .get(&scx_format_io::column_name_hash("cell_type"))
        .expect("cell_type is indexed");
    assert!(
        dict.complete,
        "every CSR shard of a freshly converted file carries the column's \
         CategoryBitset, so its vocabulary is the complete value set"
    );

    let pm = plan_and_mask(&pipeline).unwrap();
    assert_eq!(
        pm.skipped_shards, MS_SHARDS,
        "'Z' is absent from a complete vocabulary → prune every shard"
    );
    assert_eq!(pm.matched_rows, 0);
}

/// The catalog-derived signal itself.
///
/// `derive_shard_column_stats` emits a `CategoryBitset` for *every* shard of
/// an indexed categorical column — an all-zero one where the column has no
/// values in that shard — so a shard without one was never seen by the
/// build that produced the vocabulary. That is what `append` without
/// `--index-obs` leaves behind.
#[test]
fn a_shard_without_the_columns_bitset_makes_the_vocabulary_incomplete() {
    use scx_format_io::catalog::{ColumnStat, FullCatalogEntry, ShardStats};
    use scx_format_io::section::SectionType;

    let hash = scx_format_io::column_name_hash("cell_type");
    let shard = |column_stats: Vec<ColumnStat>| FullCatalogEntry {
        name: "X_shard".to_string(),
        offset: 4352,
        length: 10,
        section_type: SectionType::CsrShard,
        checksum: [0; 32],
        modality_id: 0,
        stats: Some(ShardStats {
            row_start: 0,
            row_end: 10,
            col_start: 0,
            col_end: 0,
            nnz: 1,
            value_min: 1,
            value_max: 1,
            value_sum: 1,
            n_indexed_columns: column_stats.len() as u8,
            column_stats,
        }),
    };
    let bitset = || ColumnStat::CategoryBitset {
        column_name_hash: hash,
        bitset: vec![0b0000_0001],
    };

    let indexed = shard(vec![bitset()]);
    let appended = shard(Vec::new());
    // A shard carrying stats for some *other* indexed column is just as
    // uncovered for this one.
    let other_column = shard(vec![ColumnStat::CategoryBitset {
        column_name_hash: scx_format_io::column_name_hash("tissue"),
        bitset: vec![0b0000_0001],
    }]);
    // `ColumnStat::column_name_hash` answers for `MinMax` too, so a numeric
    // stat under this column's hash satisfies a hash-only test — and would
    // let a numeric column license a categorical vocabulary.
    let numeric_stat = shard(vec![ColumnStat::MinMax {
        column_name_hash: hash,
        min: 0.0,
        max: 1.0,
    }]);
    // Sized for a nine-value vocabulary: a different index build.
    let wrong_len = shard(vec![ColumnStat::CategoryBitset {
        column_name_hash: hash,
        bitset: vec![0b0000_0001, 0b0000_0000],
    }]);

    // One value → one byte, which is what `bitset()` carries.
    let complete = |shards: &[&FullCatalogEntry], n_values: usize| {
        collect_bitset_coverage(shards)
            .get(&hash)
            .is_some_and(|c| c.covers(shards.len(), n_values))
    };

    assert!(complete(&[&indexed, &indexed], 1));
    assert!(
        !complete(&[&indexed, &appended], 1),
        "the appended shard was never seen by the index build"
    );
    assert!(!complete(&[&other_column], 1));
    assert!(
        !complete(&[&numeric_stat], 1),
        "a MinMax stat is not evidence that a categorical vocabulary is complete"
    );
    assert!(
        !complete(&[&wrong_len], 1),
        "a bitset sized for another vocabulary means the stats and the index \
         section came from different builds"
    );
    assert!(
        !complete(&[&indexed, &wrong_len], 1),
        "one disagreeing shard is enough — bit i no longer means entry i"
    );
    assert!(
        !complete(&[&indexed], 9),
        "nine values need two bytes; this shard carries one"
    );
    assert!(
        !complete(&[], 1),
        "with nothing to check against, a coverage claim is vacuous"
    );

    // Counting stat *records* rather than distinct shards lets one shard's
    // surplus pay for another shard's absence. Two bitsets under the hash on
    // shard A and none on shard B still totals two — and shard B, which
    // nothing covers, is what the vocabulary would then be claiming to
    // describe. `&[&indexed, &indexed]` above does not catch this: it
    // repeats an entry *reference*, which is two shards each carrying one.
    let doubled = shard(vec![bitset(), bitset()]);
    assert!(
        !complete(&[&doubled, &appended], 1),
        "a duplicate bitset in one shard must not stand in for a shard that \
         carries none"
    );
    assert!(
        !complete(&[&doubled, &indexed], 1),
        "two bitsets for one column in a single shard is a malformed \
         catalog, not extra evidence"
    );
}

// -----------------------------------------------------------------------
// A multimodal fixture with TWO CSR shards per modality, written interleaved.
//
// Every other multimodal fixture in the tree writes one shard per modality at
// `row_start 0` (`write_csr_shard_for(id, 0, shard)` — verified across
// scx-cli, scx-cloud, scx-convert, scx-format-io and the integration crate), so
// none of them can tell a per-modality shard list from the flattened
// all-modality one: with one shard each, position 0 is position 0 either way.
// Here each modality has shards at `row_start` 0 and 4 and the four are written
// rna, adt, rna, adt — so a modality's positions are non-adjacent catalog
// entries, and the flattened list carries duplicate `row_start`s.
//
// It carries no obs predicate index, and that is a measurement rather than an
// omission: `build_and_write_conversion_predicate_indexes` refuses this file
// with `ColumnStatsShardCountMismatch { got: 2, expected: 0 }`, because the
// derived per-shard column stats have nowhere to attach on a per-modality CSR
// list. So the modality-scoped predicate below takes the legacy full-decode obs
// path — which is what `build_plan` forces for `modality_id != 0` anyway, by
// nulling `obs_predicate_index` (the index's `shard_id` space is the flattened
// order and would resolve rows against another modality's shard).
// -----------------------------------------------------------------------

const MM_SHARD_ROWS: [usize; 2] = [4, 3];
const MM_N_OBS: usize = 7;
const MM_RNA_VARS: usize = 5;
const MM_ADT_VARS: usize = 3;

/// rna row `r`: gene `r % 5`, value `r + 1`.
fn mm_rna_cell(r: usize) -> (u32, u8) {
    ((r % MM_RNA_VARS) as u32, (r + 1) as u8)
}

/// adt row `r`: gene `r % 3`, value `100 + r` — a different column *and* a
/// different value from rna's, so reading the wrong modality is a value
/// mismatch, not only a width one.
fn mm_adt_cell(r: usize) -> (u32, u8) {
    ((r % MM_ADT_VARS) as u32, (100 + r) as u8)
}

fn mm_dense(rows: &[usize], n_vars: usize, cell: impl Fn(usize) -> (u32, u8)) -> Vec<Vec<f32>> {
    rows.iter()
        .map(|&r| {
            let (col, val) = cell(r);
            let mut out = vec![0.0f32; n_vars];
            out[col as usize] = val as f32;
            out
        })
        .collect()
}

fn write_multimodal_multishard_file(dir: &tempfile::TempDir) -> (std::path::PathBuf, u8, u8) {
    use scx_format_io::modality::ModalityType;
    let path = dir.path().join("mm_multishard.scx");
    let obs = sample_obs(MM_N_OBS);
    // header n_vars is the file-wide max across modalities.
    let mut header = sample_header(MM_N_OBS as u64, MM_RNA_VARS as u64, 0);
    header.shard_target_rows = MM_SHARD_ROWS[0] as u32;
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&obs).unwrap();

    let rna_id = writer
        .add_modality(
            "rna",
            ModalityType::Rna,
            CodecId::None,
            ValueEncoding::Uint8,
            false,
        )
        .unwrap();
    let adt_id = writer
        .add_modality(
            "adt",
            ModalityType::Protein,
            CodecId::None,
            ValueEncoding::Uint8,
            false,
        )
        .unwrap();
    writer
        .write_var_for(rna_id, &sample_var(MM_RNA_VARS))
        .unwrap();
    writer
        .write_var_for(adt_id, &sample_var(MM_ADT_VARS))
        .unwrap();
    writer
        .set_modality_n_vars(rna_id, MM_RNA_VARS as u64)
        .unwrap();
    writer
        .set_modality_n_vars(adt_id, MM_ADT_VARS as u64)
        .unwrap();

    // Interleaved: rna@0, adt@0, rna@4, adt@4.
    let mut row_start = 0usize;
    let mut csr_row_ranges = Vec::new();
    for &n_rows in MM_SHARD_ROWS.iter() {
        for (mid, n_vars, cell) in [
            (
                rna_id,
                MM_RNA_VARS,
                &mm_rna_cell as &dyn Fn(usize) -> (u32, u8),
            ),
            (
                adt_id,
                MM_ADT_VARS,
                &mm_adt_cell as &dyn Fn(usize) -> (u32, u8),
            ),
        ] {
            let _ = n_vars;
            let mut indptr = vec![0u64];
            let mut indices = Vec::new();
            let mut values = Vec::new();
            for row in row_start..row_start + n_rows {
                let (col, val) = cell(row);
                indices.push(col);
                values.push(val);
                indptr.push(indptr.last().unwrap() + 1);
            }
            let shard = scx_format_io::ShardBuffers::new(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
            );
            writer
                .write_csr_shard_for(mid, row_start as u64, shard)
                .unwrap();
        }
        csr_row_ranges.push((row_start as u64, (row_start + n_rows) as u64));
        row_start += n_rows;
    }

    // Recorded so the shape stays visible even though nothing consumes it: the
    // per-modality row ranges an index *would* have been built over.
    debug_assert_eq!(csr_row_ranges.len(), MM_SHARD_ROWS.len());
    writer.finish().unwrap();
    (path, rna_id, adt_id)
}

/// A modality-scoped query reads *that* modality's shards, on a file where the
/// per-modality list and the flattened list disagree about every position.
///
/// The shard list is now derived once per pipeline and threaded to the pruner,
/// the masker and both materialize paths, which is only safe if the list is the
/// queried modality's. With one shard per modality — every other multimodal
/// fixture in the tree — modality 1's shard is at position 0 of both lists, so
/// the wrong basis is undetectable. Two interleaved shards each make it a value
/// mismatch.
#[test]
fn a_modality_scoped_collect_reads_that_modalitys_shards_not_position_zeros() {
    let dir = tempfile::tempdir().unwrap();
    let (path, rna_id, adt_id) = write_multimodal_multishard_file(&dir);

    // -- Premises, read back off the written catalog --
    let reader = scx_format_io::ScxReader::open(&path).unwrap();
    let catalog = reader.catalog();
    let rna_pos = catalog.csr_shard_indices(Some(rna_id));
    let adt_pos = catalog.csr_shard_indices(Some(adt_id));
    let flat = catalog.csr_shard_indices(None);
    assert_eq!(rna_pos.len(), 2, "rna must have two CSR shards");
    assert_eq!(adt_pos.len(), 2, "adt must have two CSR shards");
    assert_eq!(flat.len(), 4);
    assert_ne!(
        rna_pos[1] - rna_pos[0],
        1,
        "the two modalities' shards must be interleaved in the catalog, or a \
         flattened list would agree with the per-modality one"
    );
    assert!(
        catalog.has_overlapping_csr_ranges(),
        "each modality independently tiles [0, n_obs), so the flattened list \
         must overlap — that is what makes positional reuse across modalities \
         wrong"
    );
    drop(reader);

    let rows_all: Vec<usize> = (0..MM_N_OBS).collect();

    let rna = QueryPipeline::open_for_modality(&path, rna_id)
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(rna.x.shape, (MM_N_OBS, MM_RNA_VARS));
    assert_eq!(rna.total_shards, 2, "modality-scoped shard count");
    assert_eq!(
        dense_of(&rna.x),
        mm_dense(&rows_all, MM_RNA_VARS, mm_rna_cell)
    );

    let adt = QueryPipeline::open_for_modality(&path, adt_id)
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(adt.x.shape, (MM_N_OBS, MM_ADT_VARS));
    assert_eq!(adt.total_shards, 2);
    assert_eq!(
        dense_of(&adt.x),
        mm_dense(&rows_all, MM_ADT_VARS, mm_adt_cell)
    );

    // A predicate on the shared global obs axis, scoped to one modality: the
    // rows come from the obs half, the values from this modality's shards.
    let t_cells: Vec<usize> = (0..MM_N_OBS).filter(|r| r % 3 == 0).collect();
    assert_eq!(
        t_cells,
        vec![0, 3, 6],
        "sample_obs assigns cell_type by r % 3"
    );
    for (mid, n_vars, cell) in [
        (
            rna_id,
            MM_RNA_VARS,
            &mm_rna_cell as &dyn Fn(usize) -> (u32, u8),
        ),
        (
            adt_id,
            MM_ADT_VARS,
            &mm_adt_cell as &dyn Fn(usize) -> (u32, u8),
        ),
    ] {
        let filtered = QueryPipeline::open_for_modality(&path, mid)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(filtered.matched_rows, t_cells.len());
        assert_eq!(
            dense_of(&filtered.x),
            mm_dense(&t_cells, n_vars, cell),
            "modality {mid} returned another modality's values"
        );
    }
}

/// Proof that *which* modality's shards you scan is load-bearing.
///
/// `csr_shards_for_modality(0)` filters to modality 0 — it is **not** the
/// flattened all-modality list, though the test that appears to prove it
/// (`csr_shards_for_modality_0_matches_shards_sorted_single_modality`) is a
/// single-modality fixture. On a multimodal catalog the two lists give
/// opposite verdicts, which is what this pins.
///
/// ⚠️ **It does not guard `build_plan`'s call site.** It scans the modality
/// itself rather than going through `build_plan`, so reverting that call to
/// a hardcoded `0` would leave this green. Closing that needs a multimodal
/// fixture file with a predicate index; what this test buys is that the
/// argument matters at all, so the revert would be a behaviour change
/// rather than a no-op.
#[test]
fn completeness_follows_the_queried_modality_not_modality_zero() {
    use scx_format_io::catalog::{ColumnStat, FullCatalog, FullCatalogEntry, ShardStats};
    use scx_format_io::section::SectionType;

    let hash = scx_format_io::column_name_hash("cell_type");
    let shard = |modality_id: u8, row_start: u64, column_stats: Vec<ColumnStat>| FullCatalogEntry {
        name: format!("X_shard_m{modality_id}_{row_start}"),
        offset: 4352 + row_start,
        length: 10,
        section_type: SectionType::CsrShard,
        checksum: [0; 32],
        modality_id,
        stats: Some(ShardStats {
            row_start,
            row_end: row_start + 10,
            col_start: 0,
            col_end: 0,
            nnz: 1,
            value_min: 1,
            value_max: 1,
            value_sum: 1,
            n_indexed_columns: column_stats.len() as u8,
            column_stats,
        }),
    };
    let bitset = vec![ColumnStat::CategoryBitset {
        column_name_hash: hash,
        bitset: vec![0b0000_0001],
    }];

    // Modality 0 is fully indexed; modality 1's shards carry no stats.
    let catalog = FullCatalog {
        catalog_version: scx_format_io::CURRENT_CATALOG_VERSION,
        manifest_sequence: 1,
        prev_catalog_offset: 0,
        n_obs: 20,
        entries: vec![
            shard(0, 0, bitset.clone()),
            shard(0, 10, bitset.clone()),
            shard(1, 0, Vec::new()),
            shard(1, 10, Vec::new()),
        ],
        data_generation: 0,
        csc_build_generation: 0,
    };

    let covers = |modality_id: u8| {
        let shards = scan_shards(&catalog, &catalog.csr_shard_indices(Some(modality_id)));
        collect_bitset_coverage(&shards)
            .get(&hash)
            .is_some_and(|c| c.covers(shards.len(), 1))
    };

    assert!(covers(0), "modality 0's shards are all indexed");
    assert!(
        !covers(1),
        "a modality-1 query must judge completeness over modality 1's \
         shards — reading modality 0's would license pruning shards nothing \
         was checked against"
    );
}

/// Item #1 control: with no limit the full-scan path decodes every
/// candidate shard and the pre-limit count equals the assembled rows.
#[test]
fn no_limit_full_scan_decodes_all_shards() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_indexed_multishard_file(&dir);

    let pipeline = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'T cell'")
        .unwrap();
    let pm = plan_and_mask(&pipeline).unwrap();
    let r = materialize(&pipeline, pm).unwrap();

    assert_eq!(r.x.n_rows(), MS_SHARDS);
    // Debug-only counter (see `limit_pushdown_bounds_full_scan_x_decode`).
    #[cfg(debug_assertions)]
    assert_eq!(
        x_decode_count(&pipeline),
        MS_SHARDS as u64,
        "unlimited query decodes every candidate shard"
    );
}

/// Item #2: an equality whose value is absent from the (indexed) column's
/// global dictionary must skip every shard and touch NO obs/X metadata.
#[test]
fn no_match_indexed_value_short_circuits_without_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_indexed_multishard_file(&dir);

    let pipeline = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'Z'")
        .unwrap();
    let pm = plan_and_mask(&pipeline).unwrap();
    assert_eq!(pm.total_shards, MS_SHARDS);
    assert_eq!(
        pm.skipped_shards, MS_SHARDS,
        "an absent indexed value skips every shard"
    );
    assert_eq!(pm.matched_rows, 0);
    let r = materialize(&pipeline, pm).unwrap();
    assert_eq!(r.x.n_rows(), 0);
    assert_eq!(r.obs.num_rows(), 0);
    assert_eq!(r.skipped_shards, MS_SHARDS);

    // Read counters are only incremented under `debug_assertions` (compiled
    // away in release), so inspect them only when present.
    #[cfg(debug_assertions)]
    {
        let reader = pipeline
            .reader()
            .as_any()
            .downcast_ref::<scx_format_io::reader::ScxReader>()
            .unwrap();
        let counts = reader.debug_counts();
        use std::sync::atomic::Ordering::Relaxed;
        assert_eq!(
            counts.read_obs.load(Relaxed),
            0,
            "short-circuit must not materialise the full obs table"
        );
        // The full obs scan is avoided entirely. `materialize` reads at most
        // one obs shard as the 0-row schema template for the empty result (a
        // bounded, MB-scale read) — never the whole file. `count()` skips
        // materialise and reads zero (asserted in the integration test).
        assert!(
            counts.read_obs_shard.load(Relaxed) < MS_SHARDS as u64,
            "short-circuit must not scan all obs shards"
        );
        assert_eq!(
            counts.read_shard_from_entry.load(Relaxed),
            0,
            "short-circuit must not decode any X shard"
        );
    }
}
