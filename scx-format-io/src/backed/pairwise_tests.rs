//! Tests for [`BackedPairwiseReader`].
//!
//! Every claim here has a named mutation in the phase-4 plan; the mutation was
//! applied and seen to fail exactly its own test before the test was kept.

use super::*;
use crate::writer::ScxWriter;
use arrow::array::{Float32Array, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_format::FileHeader;
use std::collections::HashMap;
use tempfile::TempDir;

fn header(n_obs: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, 4, 0, 16384, 0, 0)
}

fn obs(n: usize) -> arrow::array::RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    arrow::array::RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "cell_id",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn var(n: usize) -> arrow::array::RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
    arrow::array::RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "gene_id",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn coo_meta(n_rows: usize, n_cols: usize) -> HashMap<String, String> {
    HashMap::from([
        ("n_rows".to_string(), n_rows.to_string()),
        ("n_cols".to_string(), n_cols.to_string()),
    ])
}

/// A COO batch in the Int32 (v1) coordinate form.
fn coo_i32(triples: &[(i64, i64, f32)], n_rows: usize, n_cols: usize) -> arrow::array::RecordBatch {
    let schema = Schema::new(vec![
        Field::new("row", DataType::Int32, false),
        Field::new("col", DataType::Int32, false),
        Field::new("data", DataType::Float32, false),
    ])
    .with_metadata(coo_meta(n_rows, n_cols));
    arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(
                triples.iter().map(|t| t.0 as i32).collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                triples.iter().map(|t| t.1 as i32).collect::<Vec<_>>(),
            )),
            Arc::new(Float32Array::from(
                triples.iter().map(|t| t.2).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// The same triples in the Int64 (v2 wide-axis) form, with Float64 values.
fn coo_i64_f64(
    triples: &[(i64, i64, f32)],
    n_rows: usize,
    n_cols: usize,
) -> arrow::array::RecordBatch {
    let schema = Schema::new(vec![
        Field::new("row", DataType::Int64, false),
        Field::new("col", DataType::Int64, false),
        Field::new("data", DataType::Float64, false),
    ])
    .with_metadata(coo_meta(n_rows, n_cols));
    arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int64Array::from(
                triples.iter().map(|t| t.0).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                triples.iter().map(|t| t.1).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                triples.iter().map(|t| t.2 as f64).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// A deterministic ring graph: row `r` is connected to `r+1` and `r+2` (mod n),
/// weight `r * 10 + hop`. Distinct per edge, so a dropped or misrouted triple is
/// visible in the value, not only in the position.
fn ring_triples(n: usize) -> Vec<(i64, i64, f32)> {
    let mut out = Vec::new();
    for r in 0..n {
        for hop in 1..=2usize {
            let c = (r + hop) % n;
            out.push((r as i64, c as i64, (r * 10 + hop) as f32));
        }
    }
    out
}

/// Dense reference built straight from the triples, bypassing the reader.
fn dense_ref(triples: &[(i64, i64, f32)], n: usize) -> Vec<Vec<f32>> {
    let mut m = vec![vec![0.0f32; n]; n];
    for &(r, c, v) in triples {
        m[r as usize][c as usize] = v;
    }
    m
}

fn assert_matches_ref(rows: &PairwiseRows, reference: &[Vec<f32>]) {
    for i in 0..rows.n_rows as usize {
        let (cols, vals) = rows.row(i);
        assert!(
            cols.windows(2).all(|w| w[0] < w[1]),
            "row {i} columns are not strictly ascending: {cols:?}"
        );
        let global = rows.row_start as usize + i;
        let mut expected: Vec<(i64, f32)> = reference[global]
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0.0)
            .map(|(c, &v)| (c as i64, v))
            .collect();
        expected.sort_by_key(|&(c, _)| c);
        let got: Vec<(i64, f32)> = cols.iter().copied().zip(vals.iter().copied()).collect();
        assert_eq!(got, expected, "row {global}");
    }
}

/// Write `n_shards` obsp shards for `connectivities`. `shuffle_within_shard`
/// reverses each shard's triple order, which the format permits and
/// `scx-convert`'s override path actually produces.
fn write_sharded(
    dir: &TempDir,
    n: usize,
    n_shards: usize,
    triples: &[(i64, i64, f32)],
    shuffle_within_shard: bool,
    wide: bool,
) -> std::path::PathBuf {
    let path = dir.path().join("obsp_sharded.scx");
    let mut w = ScxWriter::new(&path, header(n as u64)).unwrap();
    w.write_obs(&obs(n)).unwrap();
    w.write_var(&var(4)).unwrap();
    let rows_per_shard = n.div_ceil(n_shards);
    for s in 0..n_shards {
        let start = s * rows_per_shard;
        let shard_rows = rows_per_shard.min(n.saturating_sub(start));
        let mut in_shard: Vec<(i64, i64, f32)> = triples
            .iter()
            .copied()
            .filter(|t| (t.0 as usize) >= start && (t.0 as usize) < start + shard_rows)
            .collect();
        if shuffle_within_shard {
            in_shard.reverse();
        }
        let batch = if wide {
            coo_i64_f64(&in_shard, n, n)
        } else {
            coo_i32(&in_shard, n, n)
        };
        w.write_obsp_shard_coo(
            "connectivities",
            s as u32,
            start as u64,
            shard_rows as u64,
            n as u64,
            &batch,
        )
        .unwrap();
    }
    w.finish().unwrap();
    path
}

/// Write the graph as one unsharded `ObspEmbedding` — the form `scx sort`
/// re-emits.
fn write_legacy(dir: &TempDir, n: usize, triples: &[(i64, i64, f32)]) -> std::path::PathBuf {
    let path = dir.path().join("obsp_legacy.scx");
    let mut w = ScxWriter::new(&path, header(n as u64)).unwrap();
    w.write_obs(&obs(n)).unwrap();
    w.write_var(&var(4)).unwrap();
    w.write_obsp("connectivities", &coo_i32(triples, n, n))
        .unwrap();
    w.finish().unwrap();
    path
}

fn open(path: &std::path::Path) -> BackedPairwiseReader {
    BackedPairwiseReader::new_obsp(ScxReader::open(path).unwrap(), "connectivities").unwrap()
}

// -----------------------------------------------------------------------
// Row-range equality
// -----------------------------------------------------------------------

#[test]
fn row_range_equals_the_whole_matrix_slice() {
    let dir = tempfile::tempdir().unwrap();
    let n = 24;
    let triples = ring_triples(n);
    let reference = dense_ref(&triples, n);
    let path = write_sharded(&dir, n, 4, &triples, false, false);
    let backed = open(&path);
    assert_eq!(backed.n_rows(), n as u64);
    assert_eq!(backed.n_cols(), n as u64);

    // Every range, including the ones that start and end mid-shard.
    for start in 0..=n as u64 {
        for end in start..=n as u64 {
            let rows = backed.read_rows_range(start, end).unwrap();
            assert_eq!(rows.n_rows, end - start);
            assert_eq!(rows.row_start, start);
            assert_eq!(rows.indptr.len(), (end - start) as usize + 1);
            assert_eq!(rows.indptr[0], 0);
            assert_eq!(*rows.indptr.last().unwrap() as usize, rows.nnz());
            assert_matches_ref(&rows, &reference);
        }
    }
}

#[test]
fn a_range_spanning_a_shard_boundary_reads_both_shards() {
    // Replaces the plan doc's "multi-block obsp shard (> 65,535 rows)" case,
    // which describes `ObspCsrShard`'s u16 `BlockIndex` — a different section
    // type that no read API materialises. Section 22 has no BlockIndex; its
    // boundary is the shard cover, and this is that boundary.
    let dir = tempfile::tempdir().unwrap();
    let n = 20;
    let triples = ring_triples(n);
    let reference = dense_ref(&triples, n);
    let path = write_sharded(&dir, n, 4, &triples, false, false); // 5 rows/shard
    let backed = open(&path);
    assert_eq!(backed.shard_count(), 4);

    // [3, 8) straddles shards 0 and 1; [3, 13) straddles three.
    for (start, end, want_shards) in [(3u64, 8u64, 2usize), (3, 13, 3), (4, 6, 2)] {
        assert_eq!(
            backed.shards_overlapping(start, end).len(),
            want_shards,
            "range [{start}, {end})"
        );
        let rows = backed.read_rows_range(start, end).unwrap();
        assert!(rows.nnz() > 0, "range [{start}, {end}) read nothing");
        assert_matches_ref(&rows, &reference);
    }
}

#[test]
fn unsorted_triples_within_a_shard_still_produce_ascending_rows() {
    let dir = tempfile::tempdir().unwrap();
    let n = 16;
    let triples = ring_triples(n);
    let reference = dense_ref(&triples, n);
    // `shuffle_within_shard` reverses each shard's triples: rows descend and
    // each row's columns arrive in the wrong order.
    let path = write_sharded(&dir, n, 2, &triples, true, false);
    let rows = open(&path).read_rows_range(0, n as u64).unwrap();
    assert_matches_ref(&rows, &reference);
}

#[test]
fn a_legacy_unsharded_obsp_reads_its_true_row_count() {
    // `batch.num_rows()` on this section is nnz (32), not n_obs (16). The
    // layout resolver must take `n_rows` off the schema metadata instead.
    let dir = tempfile::tempdir().unwrap();
    let n = 16;
    let triples = ring_triples(n);
    assert_eq!(triples.len(), 2 * n, "the fixture must have nnz != n_rows");
    let path = write_legacy(&dir, n, &triples);
    let backed = open(&path);
    assert!(backed.is_legacy_single_section());
    assert_eq!(backed.n_rows(), n as u64);
    assert_eq!(backed.shard_count(), 1);
    let rows = backed.read_rows_range(0, n as u64).unwrap();
    assert_matches_ref(&rows, &dense_ref(&triples, n));
    // And the range clamp is against n_obs, not nnz.
    assert!(backed.read_rows_range(0, n as u64 + 1).is_err());
}

#[test]
fn int64_coordinates_and_float64_values_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let n = 12;
    let triples = ring_triples(n);
    let path = write_sharded(&dir, n, 3, &triples, false, true);
    let rows = open(&path).read_rows_range(0, n as u64).unwrap();
    assert_matches_ref(&rows, &dense_ref(&triples, n));
}

// -----------------------------------------------------------------------
// Edges of the range contract
// -----------------------------------------------------------------------

#[test]
fn an_empty_range_is_an_empty_csr_not_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let triples = ring_triples(10);
    let path = write_sharded(&dir, 10, 2, &triples, false, false);
    let backed = open(&path);
    for (s, e) in [(0u64, 0u64), (5, 5), (10, 10)] {
        let rows = backed.read_rows_range(s, e).unwrap();
        assert_eq!(rows.n_rows, 0);
        assert_eq!(rows.indptr, vec![0]);
        assert_eq!(rows.nnz(), 0);
        assert_eq!(rows.n_cols, 10);
    }
}

#[test]
fn an_invalid_range_is_refused_rather_than_answered_empty() {
    let dir = tempfile::tempdir().unwrap();
    let triples = ring_triples(10);
    let path = write_sharded(&dir, 10, 2, &triples, false, false);
    let backed = open(&path);
    // Past the end, inverted, and a `start` past the end with an in-range
    // `end`. The last two used to fall through to the empty-range arm and come
    // back `Ok` with a `row_start` naming no row, which a caller looping over
    // blocks cannot tell from a genuinely empty graph.
    for (start, stop) in [(8u64, 11u64), (7, 3), (20, 5), (11, 11), (0, 11)] {
        let err = match backed.read_rows_range(start, stop) {
            Ok(rows) => panic!(
                "range [{start}, {stop}) was accepted, returning row_start={} n_rows={}",
                rows.row_start, rows.n_rows
            ),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("invalid row range"),
            "[{start}, {stop}): {err}"
        );
    }
    // The boundary that IS legal: an empty range at the very end.
    let rows = backed.read_rows_range(10, 10).unwrap();
    assert_eq!(rows.n_rows, 0);
}

#[test]
fn a_shard_carrying_a_triple_outside_its_stamped_span_is_refused() {
    // Silently filtering it makes one misfiled triple invisible twice: skipped
    // in this shard's range, and never looked for in the shard that covers its
    // row. Two wrong answers and no error.
    let dir = tempfile::tempdir().unwrap();
    let n = 12;
    let path = dir.path().join("misfiled.scx");
    let mut w = ScxWriter::new(&path, header(n as u64)).unwrap();
    w.write_obs(&obs(n)).unwrap();
    w.write_var(&var(4)).unwrap();
    // Shard 0 covers rows [0, 6) but is handed a triple at row 9.
    w.write_obsp_shard_coo(
        "connectivities",
        0,
        0,
        6,
        n as u64,
        &coo_i32(&[(1, 2, 1.0), (9, 3, 1.0)], n, n),
    )
    .unwrap();
    w.write_obsp_shard_coo("connectivities", 1, 6, 6, n as u64, &coo_i32(&[], n, n))
        .unwrap();
    w.finish().unwrap();
    let err = open(&path).read_rows_range(0, 6).unwrap_err().to_string();
    assert!(err.contains("stamped [0, 6)"), "{err}");
}

#[test]
fn a_nullable_coo_column_is_refused_rather_than_read_as_garbage() {
    use arrow::array::{Float32Array, Int32Array};
    let dir = tempfile::tempdir().unwrap();
    let n = 4;
    let path = dir.path().join("nulls.scx");
    let mut w = ScxWriter::new(&path, header(n as u64)).unwrap();
    w.write_obs(&obs(n)).unwrap();
    w.write_var(&var(4)).unwrap();
    let schema = Schema::new(vec![
        Field::new("row", DataType::Int32, true),
        Field::new("col", DataType::Int32, true),
        Field::new("data", DataType::Float32, true),
    ])
    .with_metadata(coo_meta(n, n));
    let batch = arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(0), None])),
            Arc::new(Int32Array::from(vec![Some(1), Some(2)])),
            Arc::new(Float32Array::from(vec![Some(1.0), Some(2.0)])),
        ],
    )
    .unwrap();
    w.write_obsp("connectivities", &batch).unwrap();
    w.finish().unwrap();
    let err = open(&path).read_rows_range(0, 4).unwrap_err().to_string();
    assert!(err.contains("null entries"), "{err}");
}

#[test]
fn a_later_shard_with_the_wrong_column_count_errors_rather_than_panicking() {
    // This used to reach `batch.column(2)` — which arrow PANICS on out of
    // bounds — because the layout resolver read shard 0's schema and no other.
    // It is now refused at OPEN, by the resolver's cross-shard schema check,
    // which is earlier and better: nothing decodes at all. The per-shard
    // `num_columns() != 3` guard in `decode_shard` stays as defence in depth —
    // it still covers the legacy single-section path, which has no later shard
    // to compare against.
    let dir = tempfile::tempdir().unwrap();
    let n = 8;
    let path = dir.path().join("ragged.scx");
    let mut w = ScxWriter::new(&path, header(n as u64)).unwrap();
    w.write_obs(&obs(n)).unwrap();
    w.write_var(&var(4)).unwrap();
    w.write_obsp_shard_coo(
        "connectivities",
        0,
        0,
        4,
        n as u64,
        &coo_i32(&[(0, 1, 1.0)], n, n),
    )
    .unwrap();
    let two_col = Schema::new(vec![
        Field::new("row", DataType::Int32, false),
        Field::new("col", DataType::Int32, false),
    ])
    .with_metadata(coo_meta(n, n));
    let batch = arrow::array::RecordBatch::try_new(
        Arc::new(two_col),
        vec![
            Arc::new(Int32Array::from(vec![5])),
            Arc::new(Int32Array::from(vec![6])),
        ],
    )
    .unwrap();
    w.write_obsp_shard_coo("connectivities", 1, 4, 4, n as u64, &batch)
        .unwrap();
    w.finish().unwrap();
    let err =
        match BackedPairwiseReader::new_obsp(ScxReader::open(&path).unwrap(), "connectivities") {
            Ok(_) => panic!("a ragged shard schema was accepted"),
            Err(e) => e.to_string(),
        };
    assert!(err.contains("column schema differs from shard 0"), "{err}");
}

#[test]
fn shards_that_disagree_about_the_matrix_extent_are_refused() {
    // The column extent is taken from shard 0 and used for the whole mapping,
    // so a later shard that disagrees would be read against the wrong axis —
    // and the only symptom would be edges silently rejected as out of range,
    // or accepted when they should not be.
    let dir = tempfile::tempdir().unwrap();
    let n = 8;
    let path = dir.path().join("extent.scx");
    let mut w = ScxWriter::new(&path, header(n as u64)).unwrap();
    w.write_obs(&obs(n)).unwrap();
    w.write_var(&var(4)).unwrap();
    w.write_obsp_shard_coo(
        "connectivities",
        0,
        0,
        4,
        n as u64,
        &coo_i32(&[(0, 1, 1.0)], n, n),
    )
    .unwrap();
    // Same three columns, but the second shard claims a different n_cols.
    w.write_obsp_shard_coo(
        "connectivities",
        1,
        4,
        4,
        n as u64,
        &coo_i32(&[(4, 5, 1.0)], n, n + 3),
    )
    .unwrap();
    w.finish().unwrap();
    let err =
        match BackedPairwiseReader::new_obsp(ScxReader::open(&path).unwrap(), "connectivities") {
            Ok(_) => panic!("shards disagreeing about n_cols were accepted"),
            Err(e) => e.to_string(),
        };
    assert!(err.contains("declares n_cols"), "{err}");
}

#[test]
fn a_non_square_obsp_is_refused() {
    // `obsp` is obs x obs by definition and the plan builders treat a column
    // index as a row id. A non-square one would hand the gather rows that do
    // not exist, so it is refused here rather than at gather time.
    let dir = tempfile::tempdir().unwrap();
    let n = 6;
    let path = dir.path().join("oblong.scx");
    let mut w = ScxWriter::new(&path, header(n as u64)).unwrap();
    w.write_obs(&obs(n)).unwrap();
    w.write_var(&var(4)).unwrap();
    w.write_obsp("connectivities", &coo_i32(&[(0, 1, 1.0)], n, n + 5))
        .unwrap();
    w.finish().unwrap();
    let err =
        match BackedPairwiseReader::new_obsp(ScxReader::open(&path).unwrap(), "connectivities") {
            Ok(_) => panic!("a non-square obsp was accepted"),
            Err(e) => e.to_string(),
        };
    assert!(err.contains("square"), "{err}");
}

#[test]
fn a_three_column_batch_with_the_wrong_field_names_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let n = 4;
    let path = dir.path().join("misnamed.scx");
    let mut w = ScxWriter::new(&path, header(n as u64)).unwrap();
    w.write_obs(&obs(n)).unwrap();
    w.write_var(&var(4)).unwrap();
    let schema = Schema::new(vec![
        Field::new("i", DataType::Int32, false),
        Field::new("j", DataType::Int32, false),
        Field::new("v", DataType::Float32, false),
    ])
    .with_metadata(coo_meta(n, n));
    let batch = arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(vec![0])),
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Float32Array::from(vec![1.0f32])),
        ],
    )
    .unwrap();
    w.write_obsp("connectivities", &batch).unwrap();
    w.finish().unwrap();
    let err =
        match BackedPairwiseReader::new_obsp(ScxReader::open(&path).unwrap(), "connectivities") {
            Ok(_) => panic!("a batch with the wrong field names was read as COO"),
            Err(e) => e.to_string(),
        };
    assert!(err.contains("row/col/data"), "{err}");
}

#[test]
fn an_isolated_row_has_degree_zero() {
    let dir = tempfile::tempdir().unwrap();
    let n = 8;
    // Drop every edge out of row 3.
    let triples: Vec<_> = ring_triples(n).into_iter().filter(|t| t.0 != 3).collect();
    let path = write_sharded(&dir, n, 2, &triples, false, false);
    let rows = open(&path).read_rows_range(0, n as u64).unwrap();
    assert_eq!(rows.indptr[3], rows.indptr[4], "row 3 must be empty");
    assert_matches_ref(&rows, &dense_ref(&triples, n));
}

#[test]
fn a_shard_carrying_no_in_range_triples_still_reads() {
    let dir = tempfile::tempdir().unwrap();
    let n = 12;
    // Only rows 0-3 have edges, so shards 1 and 2 are empty of triples but
    // still cover rows.
    let triples: Vec<_> = ring_triples(n).into_iter().filter(|t| t.0 < 4).collect();
    let path = write_sharded(&dir, n, 3, &triples, false, false);
    let rows = open(&path).read_rows_range(4, 12).unwrap();
    assert_eq!(rows.nnz(), 0);
    assert_eq!(rows.n_rows, 8);
    assert_eq!(rows.indptr, vec![0; 9]);
}

// -----------------------------------------------------------------------
// Retention and refusals
// -----------------------------------------------------------------------

#[test]
fn consecutive_ranges_inside_one_shard_hit_the_memo() {
    let dir = tempfile::tempdir().unwrap();
    let n = 40;
    let triples = ring_triples(n);
    let path = write_sharded(&dir, n, 4, &triples, false, false); // 10 rows/shard
    let backed = open(&path);
    for start in (0..10).step_by(2) {
        backed.read_rows_range(start, start + 2).unwrap();
    }
    let (hits, misses) = backed.memo_metrics();
    assert_eq!(misses, 1, "one shard should have been decoded once");
    assert_eq!(hits, 4, "the other four reads should have been memo hits");
}

#[test]
fn a_non_coo_section_under_obsp_is_refused_rather_than_misread() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.scx");
    let mut w = ScxWriter::new(&path, header(4)).unwrap();
    w.write_obs(&obs(4)).unwrap();
    w.write_var(&var(4)).unwrap();
    // Two columns, not three — a dense batch parked under `obsp/`.
    let schema = Schema::new(vec![
        Field::new("a", DataType::Float32, false),
        Field::new("b", DataType::Float32, false),
    ])
    .with_metadata(coo_meta(4, 4));
    let batch = arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float32Array::from(vec![0.0f32; 4])),
            Arc::new(Float32Array::from(vec![0.0f32; 4])),
        ],
    )
    .unwrap();
    w.write_obsp("connectivities", &batch).unwrap();
    w.finish().unwrap();

    let err =
        match BackedPairwiseReader::new_obsp(ScxReader::open(&path).unwrap(), "connectivities") {
            Ok(_) => panic!("a 2-column section under obsp/ was accepted as COO"),
            Err(e) => e.to_string(),
        };
    assert!(err.contains("row/col/data"), "{err}");
}

#[test]
fn a_column_index_past_the_matrix_extent_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let n = 6;
    let path = dir.path().join("oob.scx");
    let mut w = ScxWriter::new(&path, header(n as u64)).unwrap();
    w.write_obs(&obs(n)).unwrap();
    w.write_var(&var(4)).unwrap();
    w.write_obsp("connectivities", &coo_i32(&[(0, 99, 1.0)], n, n))
        .unwrap();
    w.finish().unwrap();
    let err = open(&path).read_rows_range(0, 1).unwrap_err().to_string();
    assert!(err.contains("out of range"), "{err}");
}

#[test]
fn an_obsp_that_is_square_but_not_on_the_file_s_obs_axis_is_refused() {
    // Squareness alone lets a 4x4 graph open on an 8-cell file, and nothing
    // downstream says so: the plan builder emits four centres, a physical
    // `read_obsp_rows` reports four columns against an `n_obs_physical` of
    // eight, and the only symptom is half the cells quietly missing.
    let dir = tempfile::tempdir().unwrap();
    let n_obs = 8usize;
    let path = dir.path().join("wrong_axis.scx");
    let mut w = ScxWriter::new(&path, header(n_obs as u64)).unwrap();
    w.write_obs(&obs(n_obs)).unwrap();
    w.write_var(&var(4)).unwrap();
    // A perfectly square 4x4 graph on an 8-cell file.
    w.write_obsp(
        "connectivities",
        &coo_i32(&[(0, 1, 1.0), (1, 0, 1.0)], 4, 4),
    )
    .unwrap();
    w.finish().unwrap();
    let err =
        match BackedPairwiseReader::new_obsp(ScxReader::open(&path).unwrap(), "connectivities") {
            Ok(r) => panic!(
                "a {}x{} obsp opened on an {n_obs}-cell file",
                r.n_rows(),
                r.n_cols()
            ),
            Err(e) => e.to_string(),
        };
    assert!(err.contains("file's own obs axis"), "{err}");
    assert!(err.contains("8 observations"), "{err}");
}
