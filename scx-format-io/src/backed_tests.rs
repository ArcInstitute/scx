use super::*;
use crate::header::FileHeader;
use crate::writer::ScxWriter;
use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_sparse::concatenate_csr;
use std::sync::Arc;
use tempfile::TempDir;

// --- Test helpers (same as reader.rs test helpers) ---

fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, n_vars, nnz, 16384, 0, 0)
}

fn sample_obs(n: usize) -> arrow::array::RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
    arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn sample_var(n: usize) -> arrow::array::RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn sample_shard_data(n_rows: usize, n_vars: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();

    for row in 0..n_rows {
        let col0 = (row * 2) % n_vars;
        let col1 = (row * 2 + 1) % n_vars;
        indices.push(col0 as u32);
        indices.push(col1 as u32);
        values.push(((row + 1) % 256) as u8);
        values.push(((row + 2) % 256) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    (indptr, indices, values)
}

/// Write a test file with the specified number of shards and return
/// a `BackedCsrReader` along with the reference full CSR.
fn write_test_file_and_open(
    dir: &TempDir,
    n_obs: usize,
    n_vars: usize,
    n_shards: usize,
    cache_shards: usize,
) -> (BackedCsrReader, ScxCsr) {
    let path = dir.path().join("test.scx");
    let total_nnz = n_obs * 2;
    let header = sample_header(n_obs as u64, n_vars as u64, total_nnz as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    let rows_per_shard = n_obs / n_shards;
    for s in 0..n_shards {
        let shard_rows = if s == n_shards - 1 {
            n_obs - rows_per_shard * s
        } else {
            rows_per_shard
        };
        let (indptr, indices, values) = sample_shard_data(shard_rows, n_vars);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                (s * rows_per_shard) as u64,
            )
            .unwrap();
    }

    writer.finish().unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let full_csr = {
        let r2 = ScxReader::open(&path).unwrap();
        r2.read_all_csr_shards().unwrap()
    };
    let backed = BackedCsrReader::new(reader, cache_shards);
    (backed, full_csr)
}

/// Build a 2-shard Scx1 file with dense rows (so the encoder emits decode
/// sidecars within the overhead budget) and return the path + the raw CSR
/// triplet (f32 values) for reference slicing.
fn write_sidecar_scx1_file(
    dir: &TempDir,
    rows_per_shard: usize,
    nnz_per_row: usize,
) -> (std::path::PathBuf, Vec<u64>, Vec<u32>, Vec<f32>) {
    let path = dir.path().join("sidecar.scx");
    let n_obs = rows_per_shard * 2;
    let n_vars = 200_000usize;
    let mut header = sample_header(n_obs as u64, n_vars as u64, 0);
    header.index_dtype = 1; // u32 indices

    let mut indptr = vec![0u64];
    let mut indices: Vec<u32> = Vec::new();
    let mut values_u8: Vec<u8> = Vec::new();
    for r in 0..n_obs {
        let mut col = 0u32;
        for k in 0..nnz_per_row {
            col += 1 + ((r * 13 + k * 7) % 250) as u32; // gaps → frame_bits ~8
            indices.push(col);
            values_u8.push(1u8 + ((r + k) % 5) as u8);
        }
        indptr.push(indices.len() as u64);
    }
    let values_f32: Vec<f32> = values_u8.iter().map(|&v| v as f32).collect();

    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
    for s in 0..2 {
        let r0 = s * rows_per_shard;
        let r1 = r0 + rows_per_shard;
        let lo = indptr[r0] as usize;
        let local_indptr: Vec<u64> = indptr[r0..=r1].iter().map(|&p| p - indptr[r0]).collect();
        let hi = indptr[r1] as usize;
        writer
            .write_csr_shard(
                &local_indptr,
                &indices[lo..hi],
                &values_u8[lo..hi],
                CodecId::Scx1,
                ValueEncoding::Uint8,
                r0 as u64,
            )
            .unwrap();
    }
    writer.finish().unwrap();
    (path, indptr, indices, values_f32)
}

/// Reference: extract rows `[a, b)` from a raw CSR triplet, rebasing indptr.
fn slice_raw(
    indptr: &[u64],
    indices: &[u32],
    values: &[f32],
    a: usize,
    b: usize,
) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
    let lo = indptr[a] as usize;
    let hi = indptr[b] as usize;
    (
        indptr[a..=b]
            .iter()
            .map(|&p| (p - indptr[a]) as i64)
            .collect(),
        indices[lo..hi].iter().map(|&v| v as i32).collect(),
        values[lo..hi].to_vec(),
    )
}

#[test]
fn read_rows_sidecar_row_range_matches_full_decode() {
    let dir = TempDir::new().unwrap();
    let rows_per_shard = 32usize;
    let (path, indptr, indices, values) = write_sidecar_scx1_file(&dir, rows_per_shard, 256);

    // Confirm decode sidecars were actually emitted (else the row-range path
    // is never taken and the test would be vacuous).
    let probe = ScxReader::open(&path).unwrap();
    let n_sidecars = probe
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == crate::section::SectionType::DecodeMetadataShard)
        .count();
    assert_eq!(n_sidecars, 2, "both Scx1 shards must carry decode sidecars");

    let backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);

    // Small windows (< shard_rows/4 = 8) of cold shards take the row-range
    // path; assert byte-identical to the reference slice. Cover a window in
    // shard 0, a window in shard 1, and a single row.
    let windows = [(2usize, 6usize), (40, 44), (10, 11), (33, 37)];
    for (a, b) in windows {
        let got = backed.read_rows(a as u64, b as u64).unwrap();
        let (eip, eix, ev) = slice_raw(&indptr, &indices, &values, a, b);
        assert_eq!(got.indptr, eip, "indptr [{a},{b})");
        assert_eq!(got.indices, eix, "indices [{a},{b})");
        assert_eq!(got.data, ev, "data [{a},{b})");
    }

    // The results above are correct via *either* path, so assert the
    // row-range fast path was actually exercised (one decode per small
    // window) — otherwise a silently-disabled fast path would pass too.
    #[cfg(debug_assertions)]
    {
        use std::sync::atomic::Ordering;
        assert_eq!(
            backed
                .reader
                .debug_counts()
                .decode_scx1_row_range
                .load(Ordering::Relaxed),
            windows.len() as u64,
            "each small window must take the sidecar row-range path",
        );
    }

    // A large window (whole file) takes the full-decode path and must also
    // match the reference.
    let full = backed.read_rows(0, (rows_per_shard * 2) as u64).unwrap();
    let (eip, eix, ev) = slice_raw(&indptr, &indices, &values, 0, rows_per_shard * 2);
    assert_eq!(full.indptr, eip);
    assert_eq!(full.indices, eix);
    assert_eq!(full.data, ev);
}

/// G2: the *scattered* gather (`read_rows_with`, the path SparseCellSetDataset
/// uses) must produce byte-identical rows whether it decodes via the Scx1
/// sidecar (O(rows)) or the full-shard fallback, and must actually take the
/// sidecar path for sparse cold groups.
#[test]
fn read_rows_with_sidecar_matches_full_decode() {
    let dir = TempDir::new().unwrap();
    let rows_per_shard = 32usize;
    let (path, indptr, indices, values) = write_sidecar_scx1_file(&dir, rows_per_shard, 256);

    // Gather scattered rows into a dense buffer keyed by request position.
    fn gather(backed: &BackedCsrReader, rows: &[u64]) -> Vec<(Vec<i32>, Vec<f32>)> {
        let mut out: Vec<(Vec<i32>, Vec<f32>)> = vec![Default::default(); rows.len()];
        backed
            .read_rows_with(rows, |i, idx, data| {
                out[i] = (idx.to_vec(), data.to_vec());
                Ok(())
            })
            .unwrap();
        out
    }

    let assert_matches = |out: &[(Vec<i32>, Vec<f32>)], rows: &[u64]| {
        for (i, &row) in rows.iter().enumerate() {
            let (_ip, eix, ev) =
                slice_raw(&indptr, &indices, &values, row as usize, row as usize + 1);
            assert_eq!(out[i].0, eix, "indices row {row}");
            assert_eq!(out[i].1, ev, "data row {row}");
        }
    };

    // Sparse group across both shards with consecutive runs ([5,6], [40,41])
    // and singletons ([2],[63]) + a duplicate request (5). Per-shard group
    // size (≤4) << shard_rows/4 = 8 ⇒ sidecar path. Runs: shard0 {2},{5,6};
    // shard1 {8,9},{31} ⇒ 4 row-range decodes.
    let backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    let sparse_rows = [2u64, 5, 6, 40, 41, 63, 5];
    let out = gather(&backed, &sparse_rows);
    assert_matches(&out, &sparse_rows);
    #[cfg(debug_assertions)]
    {
        use std::sync::atomic::Ordering;
        assert_eq!(
            backed
                .reader
                .debug_counts()
                .decode_scx1_row_range
                .load(Ordering::Relaxed),
            4,
            "sparse scattered group must take the sidecar row-range path (4 runs)",
        );
    }

    // Dense group (≥8 rows/shard ⇒ k*4 ≥ shard_rows) takes the full-shard
    // fallback; a fresh reader keeps the counter clean. Same byte-identical rows.
    let backed2 = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    let dense_rows: Vec<u64> = (0..10).chain(32..42).collect();
    let out2 = gather(&backed2, &dense_rows);
    assert_matches(&out2, &dense_rows);
    #[cfg(debug_assertions)]
    {
        use std::sync::atomic::Ordering;
        assert_eq!(
            backed2
                .reader
                .debug_counts()
                .decode_scx1_row_range
                .load(Ordering::Relaxed),
            0,
            "dense group must take the full-shard fallback, not the sidecar path",
        );
    }
}

/// Phase 0 adoption counter: `read_rows_with` must bump
/// `CacheMetrics::sidecar_groups` for sparse cold groups it serves via the
/// sidecar, and `CacheMetrics::full_shard_groups` for groups it serves via a
/// full-shard decode. `sidecar_adoption_rate = sidecar / (sidecar + full)` is
/// the primary success signal for the IndexPlan sidecar gather work (the
/// wall-clock-invisible §6.2 trap). NOTE: the `SCX_SCATTER_SIDECAR=0` arm is
/// not asserted here — `scatter_sidecar_enabled()` caches the env via
/// `OnceLock`, so it cannot be toggled within one test process.
#[test]
fn read_rows_with_bumps_sidecar_adoption_counters() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let rows_per_shard = 32usize;
    let (path, _indptr, _indices, _values) = write_sidecar_scx1_file(&dir, rows_per_shard, 256);

    fn gather(backed: &BackedCsrReader, rows: &[u64]) {
        backed
            .read_rows_with(rows, |_i, _idx, _data| Ok(()))
            .unwrap();
    }

    // Sparse cold group across both shards (≤4 rows/shard << shard_rows/4 = 8)
    // ⇒ both shard groups take the sidecar path.
    let mut backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    let m = backed.enable_metrics();
    gather(&backed, &[2u64, 5, 6, 40, 41]);
    assert_eq!(
        m.sidecar_groups.load(Ordering::Relaxed),
        2,
        "both sparse cold shard groups must be served via the sidecar",
    );
    assert_eq!(
        m.full_shard_groups.load(Ordering::Relaxed),
        0,
        "no full-shard decode for a sparse cold gather",
    );

    // Dense group (≥8 rows/shard ⇒ k*4 ≥ shard_rows) ⇒ both shard groups take
    // the full-shard fallback. Fresh reader keeps the counters clean.
    let mut backed2 = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    let m2 = backed2.enable_metrics();
    let dense_rows: Vec<u64> = (0..10).chain(32..42).collect();
    gather(&backed2, &dense_rows);
    assert_eq!(
        m2.sidecar_groups.load(Ordering::Relaxed),
        0,
        "dense group must not take the sidecar path",
    );
    assert_eq!(
        m2.full_shard_groups.load(Ordering::Relaxed),
        2,
        "both dense shard groups must be served via full-shard decode",
    );
}

/// Lever S: the per-shard Scx1 sidecar metadata is
/// parsed once and reused across batches. Repeatedly gathering sparse groups
/// from the same shard takes the sidecar path each time (so the rows are still
/// correct) yet parses the metadata only once *per distinct shard* — proof the
/// cache eliminates the per-batch O(shard-rows) BLAKE3 + clone reparse.
#[test]
fn sidecar_metadata_cached_across_batches() {
    let dir = TempDir::new().unwrap();
    let rows_per_shard = 32usize;
    let (path, indptr, indices, values) = write_sidecar_scx1_file(&dir, rows_per_shard, 256);

    fn gather(backed: &BackedCsrReader, rows: &[u64]) -> Vec<(Vec<i32>, Vec<f32>)> {
        let mut out: Vec<(Vec<i32>, Vec<f32>)> = vec![Default::default(); rows.len()];
        backed
            .read_rows_with(rows, |i, idx, data| {
                out[i] = (idx.to_vec(), data.to_vec());
                Ok(())
            })
            .unwrap();
        out
    }
    let assert_matches = |out: &[(Vec<i32>, Vec<f32>)], rows: &[u64]| {
        for (i, &row) in rows.iter().enumerate() {
            let (_ip, eix, ev) =
                slice_raw(&indptr, &indices, &values, row as usize, row as usize + 1);
            assert_eq!(out[i].0, eix, "indices row {row}");
            assert_eq!(out[i].1, ev, "data row {row}");
        }
    };

    // Three "batches", all sparse sidecar groups (group*4 < shard_rows=32):
    // b1, b2 touch shard 0; b3 touches shard 1. One reader (cache persists).
    let backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    let batches: [&[u64]; 3] = [&[2, 5, 6], &[3, 7], &[40, 41]];
    for b in batches {
        let out = gather(&backed, b);
        assert_matches(&out, b);
    }
    #[cfg(debug_assertions)]
    {
        use std::sync::atomic::Ordering;
        let dc = backed.reader.debug_counts();
        assert!(
            dc.decode_scx1_row_range.load(Ordering::Relaxed) >= 3,
            "all three sparse batches must take the sidecar row-range path",
        );
        assert_eq!(
            dc.sidecar_meta_parse.load(Ordering::Relaxed),
            2,
            "metadata parsed once per distinct shard (shard 0 reused across b1/b2), not per batch",
        );
    }
}

#[test]
fn shard_source_read_shard_arc_serves_cache_hits() {
    use crate::shard_source::ShardSource;

    let dir = TempDir::new().unwrap();
    // 4 shards, cache big enough to hold all of them.
    let (backed, _full) = write_test_file_and_open(&dir, 64, 100, 4, 4);

    assert_eq!(backed.shard_cache_capacity(), Some(4));
    assert_eq!(ShardSource::n_shards(&backed), 4);

    // First read warms the cache; the second returns the *same* Arc (no
    // re-decode) — the mechanism multi-pass PCA relies on.
    let first = backed.read_shard_arc(1).unwrap();
    assert!(backed.cache_contains(1));
    let second = backed.read_shard_arc(1).unwrap();
    assert!(
        Arc::ptr_eq(&first, &second),
        "cached read_shard_arc must return the same Arc on a hit"
    );
}

#[test]
fn ensure_cache_capacity_grows_count_cap() {
    use crate::shard_source::ShardSource;

    let dir = TempDir::new().unwrap();
    // 4 shards, opened with the small default-style count cap of 1.
    let (backed, _full) = write_test_file_and_open(&dir, 64, 100, 4, 1);
    assert_eq!(backed.cache_capacity(), 1);
    assert_eq!(ShardSource::n_shards(&backed), 4);

    // Reserve for all 4 shards within a generous byte budget → cap grows,
    // and the undersized signal clears.
    let achieved = backed.ensure_cache_capacity(4, 1 << 30);
    assert_eq!(achieved, 4);
    assert_eq!(backed.cache_capacity(), 4);
    assert_eq!(backed.shard_cache_capacity(), Some(4));

    // All 4 shards now stay resident across reads (no eviction).
    for i in 0..4 {
        let _ = backed.read_shard_arc(i).unwrap();
    }
    for i in 0..4 {
        assert!(backed.cache_contains(i), "shard {i} should be cached");
    }
}

#[test]
fn ensure_cache_capacity_noop_without_cache() {
    let dir = TempDir::new().unwrap();
    // cache_shards = 0 → no cache installed.
    let (backed, _full) = write_test_file_and_open(&dir, 64, 100, 4, 0);
    assert_eq!(backed.cache_capacity(), 0);
    assert_eq!(backed.ensure_cache_capacity(4, 1 << 30), 0);
}

#[test]
fn shard_source_cache_capacity_none_for_uncached() {
    // A non-caching ShardSource reports no cache capacity, so multi-pass
    // kernels skip the undersized-cache warning.
    use crate::shard_source::ShardSource;

    struct Uncached;
    impl ShardSource for Uncached {
        fn n_shards(&self) -> usize {
            1
        }
        fn n_obs(&self) -> usize {
            1
        }
        fn n_vars(&self) -> usize {
            1
        }
        fn read_shard(&self, _idx: usize) -> Result<ScxCsr> {
            Ok(ScxCsr::new((1, 1), vec![0i64, 0], Vec::<i32>::new(), Vec::<f32>::new()).unwrap())
        }
    }
    assert_eq!(Uncached.shard_cache_capacity(), None);
}

#[test]
fn read_rows_small_window_falls_back_without_sidecar() {
    // The None-codec helper emits no sidecars; a small-window read must
    // still return correct rows via the full-decode fallback.
    let dir = TempDir::new().unwrap();
    let (backed, full) = write_test_file_and_open(&dir, 64, 100, 2, 4);
    let got = backed.read_rows(2, 6).unwrap();
    let expect = full.row_slice(2, 6).unwrap();
    assert_eq!(got.indptr, expect.indptr);
    assert_eq!(got.indices, expect.indices);
    assert_eq!(got.data, expect.data);
}

// -----------------------------------------------------------------------
// BackedDenseReader (obsm row-gather) tests
// -----------------------------------------------------------------------

/// Build a dense obsm batch for global rows `start..start+n` with
/// `d` columns; value[r][c] = (r*d + c) as f32 (globally unique).
fn obsm_batch(start: usize, n: usize, d: usize) -> arrow::array::RecordBatch {
    use arrow::array::{ArrayRef, Float32Array};
    let fields: Vec<Field> = (0..d)
        .map(|c| Field::new(c.to_string(), DataType::Float32, false))
        .collect();
    let cols: Vec<ArrayRef> = (0..d)
        .map(|c| {
            let vals: Vec<f32> = (start..start + n).map(|r| (r * d + c) as f32).collect();
            Arc::new(Float32Array::from(vals)) as ArrayRef
        })
        .collect();
    arrow::array::RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap()
}

fn obsm_cell(batch: &arrow::array::RecordBatch, row: usize, col: usize) -> f32 {
    batch
        .column(col)
        .as_any()
        .downcast_ref::<arrow::array::Float32Array>()
        .unwrap()
        .value(row)
}

/// Write a file with `n_shards` obsm shards for key `X_emb`
/// (`d`-dimensional) and return its path.
fn write_obsm_file(dir: &TempDir, n_obs: usize, n_shards: usize, d: usize) -> std::path::PathBuf {
    let path = dir.path().join("obsm.scx");
    let header = sample_header(n_obs as u64, 4, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(4)).unwrap();
    let rows_per_shard = n_obs / n_shards;
    for s in 0..n_shards {
        let start = s * rows_per_shard;
        let shard_rows = if s == n_shards - 1 {
            n_obs - start
        } else {
            rows_per_shard
        };
        let batch = obsm_batch(start, shard_rows, d);
        writer
            .write_obsm_shard(
                "X_emb",
                s as u32,
                start as u64,
                shard_rows as u64,
                n_obs as u64,
                &batch,
            )
            .unwrap();
    }
    writer.finish().unwrap();
    path
}

#[test]
fn test_backed_dense_shape_and_dtype() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_obsm_file(&dir, 100, 4, 6);
    let reader = ScxReader::open(&path).unwrap();
    let backed = BackedDenseReader::new_obsm(reader, "X_emb", 2).unwrap();
    assert_eq!(backed.shape(), (100, 6));
    assert_eq!(backed.dtype(), &DataType::Float32);
}

#[test]
fn test_backed_dense_row_gather_matches_full() {
    let dir = tempfile::tempdir().unwrap();
    let n_obs = 100;
    let d = 6;
    let path = write_obsm_file(&dir, n_obs, 4, d);

    let full = ScxReader::open(&path).unwrap().read_obsm("X_emb").unwrap();
    let reader = ScxReader::open(&path).unwrap();
    let backed = BackedDenseReader::new_obsm(reader, "X_emb", 2).unwrap();

    // Scattered, cross-shard, out-of-order, with the last row.
    let idx: Vec<u64> = vec![5, 0, 99, 47, 23, 24, 1];
    let got = backed.read_row_indices(&idx).unwrap();
    assert_eq!(got.num_rows(), idx.len());
    assert_eq!(got.num_columns(), d);
    for (out_row, &g) in idx.iter().enumerate() {
        for c in 0..d {
            assert_eq!(
                obsm_cell(&got, out_row, c),
                obsm_cell(&full, g as usize, c),
                "row {g} col {c}"
            );
        }
    }
}

#[test]
fn test_backed_dense_duplicate_indices() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_obsm_file(&dir, 50, 2, 3);
    let full = ScxReader::open(&path).unwrap().read_obsm("X_emb").unwrap();
    let reader = ScxReader::open(&path).unwrap();
    let backed = BackedDenseReader::new_obsm(reader, "X_emb", 4).unwrap();
    let idx: Vec<u64> = vec![10, 10, 3, 10];
    let got = backed.read_row_indices(&idx).unwrap();
    assert_eq!(got.num_rows(), 4);
    for (out_row, &g) in idx.iter().enumerate() {
        assert_eq!(obsm_cell(&got, out_row, 0), obsm_cell(&full, g as usize, 0));
    }
}

#[test]
fn test_backed_dense_range_and_all() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_obsm_file(&dir, 60, 3, 4);
    let full = ScxReader::open(&path).unwrap().read_obsm("X_emb").unwrap();
    let reader = ScxReader::open(&path).unwrap();
    let backed = BackedDenseReader::new_obsm(reader, "X_emb", 2).unwrap();

    let rng = backed.read_rows_range(20, 25).unwrap();
    assert_eq!(rng.num_rows(), 5);
    for (out_row, g) in (20..25).enumerate() {
        assert_eq!(obsm_cell(&rng, out_row, 1), obsm_cell(&full, g, 1));
    }

    let all = backed.read_all().unwrap();
    assert_eq!(all.num_rows(), 60);
}

#[test]
fn test_backed_dense_out_of_range_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_obsm_file(&dir, 30, 2, 2);
    let reader = ScxReader::open(&path).unwrap();
    let backed = BackedDenseReader::new_obsm(reader, "X_emb", 2).unwrap();
    // Row 30 is out of range (valid rows are 0..30).
    assert!(backed.read_row_indices(&[5, 30]).is_err());
}

#[test]
fn test_backed_dense_legacy_single_section() {
    // Legacy single-section obsm (write_obsm) is treated as one shard.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy.scx");
    let n_obs = 40;
    let d = 5;
    let header = sample_header(n_obs as u64, 4, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(4)).unwrap();
    writer
        .write_obsm("X_emb", &obsm_batch(0, n_obs, d))
        .unwrap();
    writer.finish().unwrap();

    let full = ScxReader::open(&path).unwrap().read_obsm("X_emb").unwrap();
    let reader = ScxReader::open(&path).unwrap();
    let backed = BackedDenseReader::new_obsm(reader, "X_emb", 0).unwrap();
    assert_eq!(backed.shape(), (n_obs, d));
    let idx: Vec<u64> = vec![39, 0, 17];
    let got = backed.read_row_indices(&idx).unwrap();
    for (out_row, &g) in idx.iter().enumerate() {
        for c in 0..d {
            assert_eq!(obsm_cell(&got, out_row, c), obsm_cell(&full, g as usize, c));
        }
    }
}

// -----------------------------------------------------------------------
// BackedCsrIndex tests
// -----------------------------------------------------------------------

#[test]
fn test_index_from_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
    assert_eq!(backed.index().n_shards(), 4);
}

#[test]
fn test_shards_for_range_single_shard() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
    // 4 shards of 3 rows each: [0,3), [3,6), [6,9), [9,12)
    let shards = backed.index().shards_for_range(0, 3);
    assert_eq!(shards, vec![0]);
}

#[test]
fn test_shards_for_range_two_shards() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
    // Spans shard 0 [0,3) and shard 1 [3,6)
    let shards = backed.index().shards_for_range(1, 5);
    assert_eq!(shards, vec![0, 1]);
}

#[test]
fn test_shards_for_range_all_shards() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
    let shards = backed.index().shards_for_range(0, 12);
    assert_eq!(shards, vec![0, 1, 2, 3]);
}

#[test]
fn test_shards_for_range_empty() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
    let shards = backed.index().shards_for_range(5, 5);
    assert!(shards.is_empty());
}

#[test]
fn test_shards_for_range_at_boundary() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
    // Exact shard boundary: [3,6) should be shard 1 only
    let shards = backed.index().shards_for_range(3, 6);
    assert_eq!(shards, vec![1]);
}

#[test]
fn test_shards_for_indices_scattered() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
    // Row 0 in shard 0, row 5 in shard 1, row 11 in shard 3
    let shards = backed.index().shards_for_indices(&[0, 5, 11]);
    assert_eq!(shards, vec![0, 1, 3]);
}

#[test]
fn test_shards_for_indices_all_one_shard() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
    let shards = backed.index().shards_for_indices(&[0, 1, 2]);
    assert_eq!(shards, vec![0]);
}

#[test]
fn test_shards_for_indices_duplicates() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
    let shards = backed.index().shards_for_indices(&[0, 0, 1, 1]);
    assert_eq!(shards, vec![0]);
}

#[test]
fn test_shards_for_indices_unsorted() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
    let shards = backed.index().shards_for_indices(&[11, 0, 5]);
    assert_eq!(shards, vec![0, 1, 3]);
}

// -----------------------------------------------------------------------
// BackedCsrReader::read_rows tests
// -----------------------------------------------------------------------

#[test]
fn test_read_rows_full_range() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

    let result = backed.read_rows(0, 12).unwrap();
    assert_eq!(result.shape, full.shape);
    assert_eq!(result.indptr, full.indptr);
    assert_eq!(result.indices, full.indices);
    assert_eq!(result.data, full.data);
}

#[test]
fn test_read_rows_single_row() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

    for row in 0..12 {
        let backed_row = backed.read_rows(row, row + 1).unwrap();
        let full_row = full.row_slice(row as usize, row as usize + 1).unwrap();
        assert_eq!(
            backed_row.indptr, full_row.indptr,
            "indptr mismatch at row {row}"
        );
        assert_eq!(
            backed_row.indices, full_row.indices,
            "indices mismatch at row {row}"
        );
        assert_eq!(backed_row.data, full_row.data, "data mismatch at row {row}");
    }
}

#[test]
fn test_read_rows_cross_shard_boundary() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

    // Spans shards 0 and 1 (rows [2,5))
    let result = backed.read_rows(2, 5).unwrap();
    let expected = full.row_slice(2, 5).unwrap();
    assert_eq!(result.shape, expected.shape);
    assert_eq!(result.indptr, expected.indptr);
    assert_eq!(result.indices, expected.indices);
    assert_eq!(result.data, expected.data);
}

/// Locks the post-unification X-path dispatch: `read_shard_uncached(i)`
/// (which now always routes through `shard_entry → read_shard_from_entry`)
/// must produce per-shard `(indptr, indices, data)` triples that are
/// byte-equal to `ScxReader::read_csr_shard(i)` on a single-modality file.
///
/// Pre-fix the X branch took the direct `reader.read_csr_shard(shard_idx)`
/// path; post-fix it goes via `x_sorted_entries`. The two paths should
/// resolve to the same shard at the same catalog offset on non-multimodal
/// files (multimodal-only callers rely on the new path for correctness),
/// so this test pins the invariant.
#[test]
fn test_read_shard_uncached_byte_equal_to_reader_read_csr_shard() {
    let dir = tempfile::tempdir().unwrap();
    let n_obs = 24;
    let n_vars = 10;
    let n_shards = 4;
    let (backed, _) = write_test_file_and_open(&dir, n_obs, n_vars, n_shards, 0);

    // Reopen the file via `ScxReader` to call `read_csr_shard` directly.
    // We compare against this canonical per-shard decoder.
    let path = dir.path().join("test.scx");
    let reader = ScxReader::open(&path).unwrap();

    assert_eq!(backed.shard_count(), n_shards);
    for i in 0..n_shards {
        let backed_shard = backed.read_shard_uncached(i).unwrap();
        let (indptr, indices, data) = reader.read_csr_shard(i).unwrap();
        assert_eq!(backed_shard.indptr, indptr, "indptr mismatch at shard {i}");
        assert_eq!(
            backed_shard.indices, indices,
            "indices mismatch at shard {i}"
        );
        assert_eq!(backed_shard.data, data, "data mismatch at shard {i}");
    }
}

#[test]
fn test_read_rows_every_contiguous_range() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

    // Test every possible contiguous range
    for start in 0..12u64 {
        for end in (start + 1)..=12 {
            let result = backed.read_rows(start, end).unwrap();
            let expected = full.row_slice(start as usize, end as usize).unwrap();
            assert_eq!(
                result.indptr, expected.indptr,
                "indptr mismatch at [{start}, {end})"
            );
            assert_eq!(
                result.indices, expected.indices,
                "indices mismatch at [{start}, {end})"
            );
            assert_eq!(
                result.data, expected.data,
                "data mismatch at [{start}, {end})"
            );
        }
    }
}

#[test]
fn test_read_rows_empty_range() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 4);

    let result = backed.read_rows(5, 5).unwrap();
    assert_eq!(result.n_rows(), 0);
    assert_eq!(result.nnz(), 0);
}

// -----------------------------------------------------------------------
// BackedCsrReader::read_row_indices tests
// -----------------------------------------------------------------------

#[test]
fn test_read_row_indices_scattered() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

    let indices = [0u64, 5, 11];
    let result = backed.read_row_indices(&indices).unwrap();
    assert_eq!(result.n_rows(), 3);

    // Each row should match the corresponding row from the full CSR
    for (i, &row) in indices.iter().enumerate() {
        let expected = full.row_slice(row as usize, row as usize + 1).unwrap();
        let actual = result.row_slice(i, i + 1).unwrap();
        assert_eq!(actual.indices, expected.indices, "row {row} indices");
        assert_eq!(actual.data, expected.data, "row {row} data");
    }
}

#[test]
fn test_read_row_indices_empty() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 4);

    let result = backed.read_row_indices(&[]).unwrap();
    assert_eq!(result.n_rows(), 0);
}

// -----------------------------------------------------------------------
// BackedCsrReader::read_rows_with tests
// -----------------------------------------------------------------------

/// Helper: gather rows via `read_rows_with` into per-request `(indices,
/// data)` clones in caller order.
fn gather_with(backed: &BackedCsrReader, rows: &[u64]) -> Vec<(Vec<i32>, Vec<f32>)> {
    let mut out: Vec<(Vec<i32>, Vec<f32>)> = vec![Default::default(); rows.len()];
    backed
        .read_rows_with(rows, |i, idx, data| {
            out[i] = (idx.to_vec(), data.to_vec());
            Ok(())
        })
        .unwrap();
    out
}

#[test]
fn test_read_rows_with_matches_read_row_indices_scattered() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

    let indices = [0u64, 5, 11];
    let dense = gather_with(&backed, &indices);

    for (i, &row) in indices.iter().enumerate() {
        let expected = full.row_slice(row as usize, row as usize + 1).unwrap();
        assert_eq!(dense[i].0, expected.indices, "row {row} indices");
        assert_eq!(dense[i].1, expected.data, "row {row} data");
    }
}

#[test]
fn test_read_rows_with_empty_no_scatter_calls() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 4);

    let mut count = 0;
    backed
        .read_rows_with(&[], |_, _, _| {
            count += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn test_read_rows_with_duplicates_call_scatter_per_occurrence() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

    // Same row twice; scatter must fire twice with identical content.
    let indices = [3u64, 3];
    let dense = gather_with(&backed, &indices);

    let expected = full.row_slice(3, 4).unwrap();
    assert_eq!(dense[0].0, expected.indices);
    assert_eq!(dense[0].1, expected.data);
    assert_eq!(dense[1].0, expected.indices);
    assert_eq!(dense[1].1, expected.data);
}

#[test]
fn test_read_rows_with_unsorted_caller_order() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

    // Mixed shard order — the public scatter callback receives `i`
    // matching the input position, regardless of internal sort.
    let indices = [11u64, 0, 7, 4, 3];
    let dense = gather_with(&backed, &indices);

    for (i, &row) in indices.iter().enumerate() {
        let expected = full.row_slice(row as usize, row as usize + 1).unwrap();
        assert_eq!(dense[i].0, expected.indices, "i={i} row={row} indices");
        assert_eq!(dense[i].1, expected.data, "i={i} row={row} data");
    }
}

#[test]
fn test_read_rows_with_out_of_range_errors() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 4);

    let result = backed.read_rows_with(&[5, 99], |_, _, _| Ok(()));
    assert!(result.is_err(), "OOR row should surface as error");
}

// -----------------------------------------------------------------------
// LRU cache behavior tests
// -----------------------------------------------------------------------

#[test]
fn test_cache_disabled() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 0);

    // Should still work without cache
    let result = backed.read_rows(0, 12).unwrap();
    assert_eq!(result.indptr, full.indptr);
}

#[test]
fn test_cache_hit() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 4);

    // First read (cold)
    let r1 = backed.read_rows(0, 3).unwrap();
    // Second read (cached)
    let r2 = backed.read_rows(0, 3).unwrap();
    assert_eq!(r1.indptr, r2.indptr);
    assert_eq!(r1.indices, r2.indices);
    assert_eq!(r1.data, r2.data);
}

#[test]
fn test_cache_eviction() {
    let dir = tempfile::tempdir().unwrap();
    // Cache can hold 2 shards, file has 4
    let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 2);

    // Access shards 0, 1, 2 — shard 0 should be evicted
    let _ = backed.read_rows(0, 3).unwrap(); // shard 0
    let _ = backed.read_rows(3, 6).unwrap(); // shard 1
    let _ = backed.read_rows(6, 9).unwrap(); // shard 2 (evicts shard 0)

    // Re-read shard 0 — should still produce correct results
    let result = backed.read_rows(0, 3).unwrap();
    let expected = full.row_slice(0, 3).unwrap();
    assert_eq!(result.indptr, expected.indptr);
    assert_eq!(result.indices, expected.indices);
    assert_eq!(result.data, expected.data);
}

#[test]
fn cache_metrics_count_eviction_and_peak() {
    // Regression for the count-cap eviction undercount (CACHE-CULM-BUG): a new
    // key that evicts the LRU is returned by `push` (not `put`, which yields
    // `None`), so `evictions` must increment and `peak_bytes_in_cache` must be
    // the resident high-water mark — NOT the cumulative `bytes_inserted`.
    let dir = tempfile::tempdir().unwrap();
    // 4 shards, cache holds 2 → touching all 4 forces 2 count-cap evictions.
    let (mut backed, _full) = write_test_file_and_open(&dir, 12, 10, 4, 2);
    let m = backed.enable_metrics();
    for (a, b) in [(0u64, 3u64), (3, 6), (6, 9), (9, 12)] {
        let _ = backed.read_rows(a, b).unwrap();
    }
    use std::sync::atomic::Ordering;
    assert_eq!(m.misses.load(Ordering::Relaxed), 4, "4 cold shards decoded");
    assert_eq!(
        m.evictions.load(Ordering::Relaxed),
        2,
        "4 inserts into a cap-2 cache ⇒ 2 count-cap evictions",
    );
    let inserted = m.bytes_inserted.load(Ordering::Relaxed);
    let peak = m.peak_bytes_in_cache.load(Ordering::Relaxed);
    assert!(inserted > 0 && peak > 0);
    assert!(
        peak < inserted,
        "peak (resident high-water, ~2 shards) must be < cumulative bytes_inserted \
         (~4 shards); got peak={peak} inserted={inserted}",
    );
}

// -----------------------------------------------------------------------
// warm_shards / parallel cold-shard decode tests
// -----------------------------------------------------------------------

/// Build a reader on the same on-disk file as `seq_backed` but with
/// `cache_shards = 1` (forces the sequential body inside `warm_shards`
/// even when the `parallel` feature is on). Used to verify that the
/// parallel decode path produces byte-identical output to the
/// sequential one.
fn open_with_cache_shards(dir: &TempDir, cache_shards: usize) -> BackedCsrReader {
    let path = dir.path().join("test.scx");
    let reader = ScxReader::open(&path).unwrap();
    BackedCsrReader::new(reader, cache_shards)
}

#[test]
fn test_warm_shards_parity_parallel_vs_sequential() {
    // 8 shards, scattered indices that touch ≥ 3 of them. Compare
    // parallel-decode output against a sequential control.
    let dir = tempfile::tempdir().unwrap();
    let (par, _full) = write_test_file_and_open(&dir, 32, 10, 8, 8);
    let seq = open_with_cache_shards(&dir, 1);

    let indices: [u64; 7] = [0, 5, 11, 18, 24, 27, 31];
    let par_out = par.read_row_indices(&indices).unwrap();
    let seq_out = seq.read_row_indices(&indices).unwrap();

    assert_eq!(par_out.indptr, seq_out.indptr);
    assert_eq!(par_out.indices, seq_out.indices);
    assert_eq!(par_out.data, seq_out.data);

    // Same comparison for read_rows (range) and read_rows_with (gather).
    let par_range = par.read_rows(2, 30).unwrap();
    let seq_range = seq.read_rows(2, 30).unwrap();
    assert_eq!(par_range.indptr, seq_range.indptr);
    assert_eq!(par_range.indices, seq_range.indices);
    assert_eq!(par_range.data, seq_range.data);

    let par_gather = gather_with(&par, &indices);
    let seq_gather = gather_with(&seq, &indices);
    assert_eq!(par_gather, seq_gather);
}

#[test]
fn test_warm_shards_no_misses_is_noop() {
    // Pre-warm every touched shard, then call read_row_indices on the
    // same set. The metrics `misses` counter should stay at the
    // pre-warm value because warm_shards filters cache hits up front.
    let dir = tempfile::tempdir().unwrap();
    let (mut backed, _) = write_test_file_and_open(&dir, 32, 10, 8, 8);
    let metrics = backed.enable_metrics();

    // Pre-warm shards 0, 1, 3 directly.
    for s in [0usize, 1, 3] {
        backed.read_shard_cached_arc(s).unwrap();
    }
    let baseline_misses = metrics.misses.load(Ordering::Relaxed);
    assert_eq!(baseline_misses, 3, "pre-warm should miss exactly 3 times");

    // Indices 0, 5, 13 land on shards 0, 1, 3 respectively.
    let _ = backed.read_row_indices(&[0u64, 5, 13]).unwrap();

    let after_misses = metrics.misses.load(Ordering::Relaxed);
    assert_eq!(
        after_misses, baseline_misses,
        "fully cached read should not register any new miss",
    );
}

#[test]
fn test_warm_shards_concurrent_dedup_via_singleflight() {
    // N threads each call read_row_indices for the same set of cold
    // shards. The singleflight table must dedupe so total misses ==
    // n_unique_shards regardless of N.
    use std::sync::Arc;
    use std::thread;

    let dir = tempfile::tempdir().unwrap();
    let (mut backed, _) = write_test_file_and_open(&dir, 64, 10, 8, 8);
    let metrics = backed.enable_metrics();
    let backed = Arc::new(backed);

    // Indices touching 4 unique shards: 0, 1, 5, 7.
    let indices: Vec<u64> = vec![0, 12, 41, 60];

    let n_threads = 8;
    let mut handles = Vec::with_capacity(n_threads);
    for _ in 0..n_threads {
        let b = Arc::clone(&backed);
        let ix = indices.clone();
        handles.push(thread::spawn(move || {
            b.read_row_indices(&ix).unwrap();
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    let n_unique_shards: u64 = 4;
    let misses = metrics.misses.load(Ordering::Relaxed);
    assert!(
        misses <= n_unique_shards,
        "expected ≤ {n_unique_shards} misses across {n_threads} concurrent readers, \
             got {misses} — singleflight broke under parallel warm_shards",
    );
    // And at least one — every shard had to be decoded once.
    assert!(misses >= 1, "expected ≥ 1 miss");

    // Singleflight slots must be empty after all threads finish.
    for s in [0usize, 1, 5, 7] {
        assert!(
            !backed.in_flight_contains(s),
            "in_flight slot for shard {s} should be cleared after join",
        );
    }
}

#[test]
fn test_warm_shards_propagates_oor_error() {
    // Calling warm_shards directly with an out-of-range shard_idx
    // should surface a ShardIndexOutOfBounds error from the underlying
    // read path, and must not strand the singleflight table.
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 32, 10, 4, 4);

    // Shard 99 doesn't exist — file has 4 shards.
    let result = backed.warm_shards(&[0usize, 1, 99]);
    assert!(result.is_err(), "OOR shard idx must error");

    // Singleflight cleanup: every leader's LeaderGuard::Drop must have
    // removed its slot regardless of error.
    for s in 0..4usize {
        assert!(
            !backed.in_flight_contains(s),
            "in_flight slot for shard {s} should be cleared after error",
        );
    }
    assert!(!backed.in_flight_contains(99));
}

#[test]
fn test_warm_shards_no_cache_is_noop() {
    // cache_shards = 0 disables the cache entirely — warm_shards must
    // be a no-op (the per-shard read path will decode without caching).
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);

    backed.warm_shards(&[0usize, 1, 2, 3]).unwrap();
    // And the multi-shard reader path still works.
    let result = backed.read_rows(0, 12).unwrap();
    assert_eq!(result.shape.0, 12);
}

#[test]
fn test_warm_shards_no_deadlock_with_concurrent_direct_decoders() {
    // Regression for the AB/BA lock-order inversion between
    // `warm_shards` and `read_shard_cached_arc`'s miss path. We spawn
    // two pools against the same `BackedCsrReader`:
    //   - "warmers": call `read_rows`, which funnels through `warm_shards`
    //   - "direct":  call `read_shard_cached_arc` directly
    // Both target overlapping cold shards, so the two lock-acquisition
    // paths run concurrently. Pre-fix this deadlocks probabilistically;
    // post-fix every thread joins. If the deadlock returns, the test
    // hangs and CI's job timeout fails the run.
    //
    // `cache_shards = 8` < n_shards = 16 also exercises the P2 truncation
    // branch (warm caps at cache_shards instead of decoding the full
    // miss list and thrashing the LRU).
    use std::sync::Arc;
    use std::thread;

    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 128, 10, 16, 8);
    let backed = Arc::new(backed);

    let n_iters = 64;
    let n_warmers = 4;
    let n_direct = 4;
    let target_shards: Vec<usize> = (0..16).collect();

    let mut handles = Vec::with_capacity(n_warmers + n_direct);
    for _ in 0..n_warmers {
        let b = Arc::clone(&backed);
        handles.push(thread::spawn(move || {
            for _ in 0..n_iters {
                let _ = b.read_rows(0, 128).unwrap();
            }
        }));
    }
    for _ in 0..n_direct {
        let b = Arc::clone(&backed);
        let shards = target_shards.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..n_iters {
                for &s in &shards {
                    let _ = b.read_shard_cached_arc(s).unwrap();
                }
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

// -----------------------------------------------------------------------
// CSR concatenation tests
// -----------------------------------------------------------------------

#[test]
fn test_concatenate_empty() {
    let result = concatenate_csr(&[], 5).unwrap();
    assert_eq!(result.n_rows(), 0);
    assert_eq!(result.nnz(), 0);
    assert_eq!(result.shape.1, 5);
}

#[test]
fn test_concatenate_single() {
    let csr = ScxCsr::new_unchecked((2, 5), vec![0, 2, 3], vec![1, 3, 2], vec![5.0, 10.0, 2.0]);
    let result = concatenate_csr(std::slice::from_ref(&csr), 5).unwrap();
    assert_eq!(result.indptr, csr.indptr);
    assert_eq!(result.indices, csr.indices);
    assert_eq!(result.data, csr.data);
}

#[test]
fn test_concatenate_two() {
    let csr1 = ScxCsr::new_unchecked((1, 5), vec![0, 2], vec![1, 3], vec![5.0, 10.0]);

    let csr2 = ScxCsr::new_unchecked((1, 5), vec![0, 1], vec![2], vec![7.0]);

    let result = concatenate_csr(&[csr1, csr2], 5).unwrap();
    assert_eq!(result.shape, (2, 5));
    assert_eq!(result.indptr, vec![0, 2, 3]);
    assert_eq!(result.indices, vec![1, 3, 2]);
    assert_eq!(result.data, vec![5.0, 10.0, 7.0]);
}

#[test]
fn test_concatenate_empty_plus_nonempty() {
    let empty = ScxCsr::new_unchecked((0, 5), vec![0], vec![], vec![]);
    let nonempty = ScxCsr::new_unchecked((1, 5), vec![0, 2], vec![1, 3], vec![5.0, 10.0]);

    let result = concatenate_csr(&[empty, nonempty], 5).unwrap();
    assert_eq!(result.shape, (1, 5));
    assert_eq!(result.indptr, vec![0, 2]);
    assert_eq!(result.indices, vec![1, 3]);
    assert_eq!(result.data, vec![5.0, 10.0]);
}

#[test]
fn test_shape_accessors() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 4);
    assert_eq!(backed.shape(), (12, 10));
    assert_eq!(backed.n_obs(), 12);
    assert_eq!(backed.n_vars(), 10);
}

// -----------------------------------------------------------------------
// BackedCscReader tests
// -----------------------------------------------------------------------

/// Build a CSC arrays for a column range from a row-major dense
/// reference. Returns `(indptr_u64, indices_u32, values_u8)`.
fn csc_arrays_for_range(
    dense: &[u8],
    n_rows: usize,
    n_cols: usize,
    col_start: usize,
    col_end: usize,
) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr: Vec<u64> = vec![0];
    let mut indices: Vec<u32> = Vec::new();
    let mut values: Vec<u8> = Vec::new();
    for col in col_start..col_end {
        for row in 0..n_rows {
            let v = dense[row * n_cols + col];
            if v != 0 {
                indices.push(row as u32);
                values.push(v);
            }
        }
        indptr.push(indices.len() as u64);
    }
    (indptr, indices, values)
}

/// Write a CSR + 4-shard-CSC test file with `cols_per_csc_shard`
/// columns per CSC shard. Returns the path and the dense reference
/// matrix (row-major u8).
fn write_csc_test_file(
    dir: &TempDir,
    n_obs: usize,
    n_vars: usize,
    cols_per_csc_shard: usize,
) -> (std::path::PathBuf, Vec<u8>) {
    let path = dir.path().join("with_csc.scx");
    let header = sample_header(n_obs as u64, n_vars as u64, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    // Build a deterministic row-major dense matrix; pick a nnz
    // pattern that distributes values across all columns.
    let mut dense = vec![0u8; n_obs * n_vars];
    for r in 0..n_obs {
        for c in 0..n_vars {
            if (r + c) % 3 == 0 {
                dense[r * n_vars + c] = ((r * 7 + c * 11) % 200 + 1) as u8;
            }
        }
    }

    // CSR shard built from the dense matrix.
    let mut indptr_csr = vec![0u64];
    let mut indices_csr = Vec::new();
    let mut values_csr = Vec::new();
    for r in 0..n_obs {
        for c in 0..n_vars {
            let v = dense[r * n_vars + c];
            if v != 0 {
                indices_csr.push(c as u32);
                values_csr.push(v);
            }
        }
        indptr_csr.push(indices_csr.len() as u64);
    }
    writer
        .write_csr_shard(
            &indptr_csr,
            &indices_csr,
            &values_csr,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();

    // CSC sidecar split into shards by column.
    let mut col_start = 0usize;
    while col_start < n_vars {
        let col_end = (col_start + cols_per_csc_shard).min(n_vars);
        let (ip, ix, vb) = csc_arrays_for_range(&dense, n_obs, n_vars, col_start, col_end);
        writer
            .write_csc_shard(
                &ip,
                &ix,
                &vb,
                CodecId::None,
                ValueEncoding::Uint8,
                col_start as u64,
            )
            .unwrap();
        col_start = col_end;
    }
    writer.finish().unwrap();
    (path, dense)
}

/// A freshly-written CSC file stamps matching generations
/// (`data_generation == csc_build_generation`), so the backed CSC
/// reader opens it without a staleness error.
#[test]
fn freshly_written_csc_has_matching_generations() {
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = write_csc_test_file(&dir, 12, 10, 3);
    let reader = ScxReader::open(&path).unwrap();
    // Writer default data_generation = 1; `write_csc_shard` stamps
    // csc_build_generation = 1 via the `write_shard_inner` chokepoint.
    assert_eq!(reader.catalog().data_generation, 1);
    assert_eq!(reader.catalog().csc_build_generation, 1);
    assert!(BackedCscReader::new(reader, 0).is_ok());
}

/// Unit-test the freshness guard directly: matched generations pass;
/// a build generation behind the data generation is rejected; an
/// empty sidecar list and legacy (0/0) files are never rejected.
#[test]
fn csc_freshness_guard_logic() {
    let csc_entry = FullCatalogEntry {
        name: "X_csc_shard_0".to_string(),
        offset: 4352,
        length: 100,
        section_type: SectionType::CscShard,
        checksum: [0u8; 32],
        modality_id: 0,
        stats: None,
    };
    let mut cat = FullCatalog {
        catalog_version: 4,
        manifest_sequence: 0,
        prev_catalog_offset: 0,
        n_obs: 10,
        entries: vec![csc_entry.clone()],
        data_generation: 3,
        csc_build_generation: 3,
    };

    // Fresh: matched generations with a sidecar present.
    assert!(check_csc_sidecar_fresh(&cat, std::slice::from_ref(&csc_entry)).is_ok());

    // Stale: the sidecar was built one generation behind the data.
    cat.csc_build_generation = 2;
    let err = check_csc_sidecar_fresh(&cat, std::slice::from_ref(&csc_entry)).unwrap_err();
    assert!(matches!(
        err,
        ScxError::StaleCscSidecar {
            built_generation: 2,
            data_generation: 3,
        }
    ));

    // No sidecar entries → nothing to validate even on a mismatch.
    assert!(check_csc_sidecar_fresh(&cat, &[]).is_ok());

    // Legacy file (both counters default to 0) → treated as fresh.
    cat.data_generation = 0;
    cat.csc_build_generation = 0;
    assert!(check_csc_sidecar_fresh(&cat, &[csc_entry]).is_ok());
}

#[test]
fn backed_csc_index_basic() {
    // 12 rows × 10 cols, 3 cols per CSC shard → 4 shards: [0,3), [3,6), [6,9), [9,10).
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = write_csc_test_file(&dir, 12, 10, 3);
    let reader = ScxReader::open(&path).unwrap();
    let csc = BackedCscReader::new(reader, 0).unwrap();

    assert_eq!(csc.n_shards(), 4);
    assert_eq!(csc.n_obs(), 12);
    assert_eq!(csc.n_vars(), 10);

    // Per-shard ranges via the index.
    let idx = csc.index();
    assert_eq!(idx.shard_col_range(0), Some((0, 3)));
    assert_eq!(idx.shard_col_range(1), Some((3, 6)));
    assert_eq!(idx.shard_col_range(2), Some((6, 9)));
    assert_eq!(idx.shard_col_range(3), Some((9, 10)));
    assert_eq!(idx.shard_col_range(99), None);

    // shards_for_col_range: cover shards 1+2 only.
    assert_eq!(idx.shards_for_col_range(4, 8), vec![1, 2]);
    assert_eq!(idx.shards_for_col_range(0, 10), vec![0, 1, 2, 3]);
    // Half-open boundary exclusion.
    assert_eq!(idx.shards_for_col_range(0, 3), vec![0]);
    // Past the end / empty / inverted.
    assert!(idx.shards_for_col_range(100, 200).is_empty());
    assert!(idx.shards_for_col_range(5, 5).is_empty());

    // shard_for_col single lookups.
    assert_eq!(idx.shard_for_col(0), Some(0));
    assert_eq!(idx.shard_for_col(2), Some(0));
    assert_eq!(idx.shard_for_col(3), Some(1));
    assert_eq!(idx.shard_for_col(9), Some(3));
    assert_eq!(idx.shard_for_col(10), None);
}

#[test]
fn backed_csc_read_csc_columns_correctness() {
    // 8 rows × 12 cols, 4 cols per CSC shard → 3 shards.
    let dir = tempfile::tempdir().unwrap();
    let (path, dense) = write_csc_test_file(&dir, 8, 12, 4);
    let reader = ScxReader::open(&path).unwrap();
    let csc = BackedCscReader::new(reader, 4).unwrap();

    // Helper: dense slice as f32 for column range [c_lo, c_hi).
    let dense_slice = |c_lo: usize, c_hi: usize| -> Vec<f32> {
        let cols = c_hi - c_lo;
        let mut out = vec![0f32; 8 * cols];
        for r in 0..8 {
            for (oc, sc) in (c_lo..c_hi).enumerate() {
                out[r * cols + oc] = dense[r * 12 + sc] as f32;
            }
        }
        out
    };

    for &(c_lo, c_hi) in &[(0u32, 12u32), (1, 5), (5, 11), (4, 8), (0, 0)] {
        let got = csc.read_csc_columns(c_lo..c_hi).unwrap();
        assert_eq!(got.shape, (8, (c_hi - c_lo) as usize));
        assert_eq!(
            got.to_dense().unwrap(),
            dense_slice(c_lo as usize, c_hi as usize),
            "mismatch on cols [{c_lo}..{c_hi})"
        );
    }
}

#[test]
fn backed_csc_read_csc_columns_skip_count_metric() {
    // 8 rows × 12 cols, 4 cols per CSC shard → 3 shards.
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = write_csc_test_file(&dir, 8, 12, 4);
    let reader = ScxReader::open(&path).unwrap();
    let mut csc = BackedCscReader::new(reader, 4).unwrap();
    let metrics = csc.enable_metrics();

    // Query that overlaps shards 0 and 1 (cols [2..6) crosses the
    // 0..4 / 4..8 boundary). Shard 2 is skipped.
    let _ = csc.read_csc_columns(2..6).unwrap();
    let misses_after = metrics.misses.load(Ordering::Relaxed);
    assert_eq!(misses_after, 2, "exactly 2 shard decodes expected");
    assert_eq!(metrics.hits.load(Ordering::Relaxed), 0);

    // Reissue same range — both shards now in cache → 0 new misses.
    let _ = csc.read_csc_columns(2..6).unwrap();
    assert_eq!(
        metrics.misses.load(Ordering::Relaxed),
        misses_after,
        "no new decodes; both shards served from cache"
    );
    assert_eq!(metrics.hits.load(Ordering::Relaxed), 2);
}

#[test]
fn backed_csc_read_csc_columns_subset() {
    let dir = tempfile::tempdir().unwrap();
    let (path, dense) = write_csc_test_file(&dir, 8, 12, 4);
    let reader = ScxReader::open(&path).unwrap();
    let csc = BackedCscReader::new(reader, 4).unwrap();

    // Non-contiguous subset.
    let cols = [0u32, 1, 5, 6, 11];
    let got = csc.read_csc_columns_subset(&cols).unwrap();
    assert_eq!(got.shape, (8, cols.len()));
    let got_dense = got.to_dense().unwrap();
    for (oc, &sc) in cols.iter().enumerate() {
        for r in 0..8 {
            assert_eq!(
                got_dense[r * cols.len() + oc],
                dense[r * 12 + sc as usize] as f32,
                "subset mismatch at row {r} col {sc}"
            );
        }
    }
}

#[test]
fn backed_csc_column_shard_source_trait_dispatch() {
    // Confirm the trait impl delegates correctly.
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = write_csc_test_file(&dir, 8, 12, 4);
    let reader = ScxReader::open(&path).unwrap();
    let csc = BackedCscReader::new(reader, 0).unwrap();
    let trait_obj: &dyn crate::ColumnShardSource = &csc;
    assert_eq!(trait_obj.n_csc_shards(), 3);
    assert_eq!(trait_obj.n_obs(), 8);
    assert_eq!(trait_obj.n_vars(), 12);
    assert_eq!(trait_obj.shape(), (8, 12));
    assert_eq!(trait_obj.csc_shard_col_range(0), Some((0, 4)));
    assert_eq!(trait_obj.csc_shard_col_range(2), Some((8, 12)));
    assert_eq!(trait_obj.csc_shard_col_range(99), None);
    let s0 = trait_obj.read_csc_shard(0).unwrap();
    assert_eq!(s0.n_cols(), 4);
}

/// Phase 5b regression: when bitmaps are missing on a multimodal
/// file, the CSR fallback in `gene_detection_counts` /
/// `cells_expressing_gene` must read only the requested modality's
/// shards. Pre-fix this called `read_all()` (global) and folded in
/// rows from every modality.
#[cfg(feature = "deletion-vectors")]
#[test]
fn multimodal_fallback_does_not_mix_modalities() {
    use crate::modality::ModalityType;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multimodal.scx");
    let n_obs: u64 = 4;
    // Both modalities use the same n_vars (4) but distribute their
    // nonzeros differently, so a global merge would visibly inflate
    // the per-gene counts.
    let n_vars: u64 = 4;
    let header = sample_header(n_obs, n_vars, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs as usize)).unwrap();
    let rna_id = writer
        .add_modality(
            "rna",
            ModalityType::Rna,
            CodecId::None,
            ValueEncoding::Uint8,
            false,
        )
        .unwrap();
    let atac_id = writer
        .add_modality(
            "atac",
            ModalityType::Atac,
            CodecId::None,
            ValueEncoding::Uint8,
            false,
        )
        .unwrap();
    writer.write_var_for(rna_id, &sample_var(4)).unwrap();
    writer.write_var_for(atac_id, &sample_var(4)).unwrap();
    writer.set_modality_n_vars(rna_id, 4).unwrap();
    writer.set_modality_n_vars(atac_id, 4).unwrap();

    // RNA: cells 0..4 all express only gene 0.
    let rna_indptr: Vec<u64> = vec![0, 1, 2, 3, 4];
    let rna_indices: Vec<u32> = vec![0, 0, 0, 0];
    let rna_values: Vec<u8> = vec![1, 1, 1, 1];
    writer
        .write_csr_shard_for(
            rna_id,
            &rna_indptr,
            &rna_indices,
            &rna_values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    // ATAC: cells 0..4 all express only gene 3.
    let atac_indptr: Vec<u64> = vec![0, 1, 2, 3, 4];
    let atac_indices: Vec<u32> = vec![3, 3, 3, 3];
    let atac_values: Vec<u8> = vec![1, 1, 1, 1];
    writer
        .write_csr_shard_for(
            atac_id,
            &atac_indptr,
            &atac_indices,
            &atac_values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer.finish().unwrap();

    // No bitmap shards were written → both methods hit the
    // CSR fallback path. Without the modality-scoped read, the
    // global path folds in ATAC's gene 3 rows when asked about
    // RNA (and vice versa).
    let rna_reader = ScxReader::open(&path).unwrap();
    let rna_backed = BackedCsrReader::for_modality(rna_reader, rna_id, 0);
    assert!(
        !rna_backed.has_full_bitmap_coverage(),
        "no bitmaps were written; expected fallback path"
    );
    let rna_counts = rna_backed.gene_detection_counts().unwrap();
    assert_eq!(
        rna_counts,
        vec![4, 0, 0, 0],
        "RNA-only counts must not include ATAC's gene 3"
    );

    let atac_reader = ScxReader::open(&path).unwrap();
    let atac_backed = BackedCsrReader::for_modality(atac_reader, atac_id, 0);
    let atac_counts = atac_backed.gene_detection_counts().unwrap();
    assert_eq!(
        atac_counts,
        vec![0, 0, 0, 4],
        "ATAC-only counts must not include RNA's gene 0"
    );

    // cells_expressing_gene must follow the same scoping. RNA's
    // gene 3 has no hits; pre-fix this would return ATAC's rows.
    let rna_reader = ScxReader::open(&path).unwrap();
    let rna_backed = BackedCsrReader::for_modality(rna_reader, rna_id, 0);
    let rna_gene3 = rna_backed.cells_expressing_gene(3).unwrap();
    assert!(
        rna_gene3.is_empty(),
        "RNA modality has no cells expressing gene 3; got {rna_gene3:?}"
    );
}

/// Write a fixture `.scx` at an explicit path (peer of `write_test_file_and_open`
/// that returns nothing — the caller opens its own reader(s)).
fn write_fixture_at(path: &std::path::Path, n_obs: usize, n_vars: usize, n_shards: usize) {
    let total_nnz = n_obs * 2;
    let header = sample_header(n_obs as u64, n_vars as u64, total_nnz as u64);
    let mut writer = ScxWriter::new(path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
    let rows_per_shard = n_obs / n_shards;
    for s in 0..n_shards {
        let shard_rows = if s == n_shards - 1 {
            n_obs - rows_per_shard * s
        } else {
            rows_per_shard
        };
        let (indptr, indices, values) = sample_shard_data(shard_rows, n_vars);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                (s * rows_per_shard) as u64,
            )
            .unwrap();
    }
    writer.finish().unwrap();
}

/// Two readers backing distinct files share ONE `SharedShardCache` under a
/// single count cap: keys are namespaced by `file_id`, and the budget is global
/// across readers (Phase 1, SCX-DATA-LOADER §4.1).
#[test]
fn shared_shard_cache_spans_readers_under_one_budget() {
    let dir = TempDir::new().unwrap();
    let p0 = dir.path().join("f0.scx");
    let p1 = dir.path().join("f1.scx");
    write_fixture_at(&p0, 64, 100, 4);
    write_fixture_at(&p1, 64, 100, 4);

    // One shared cache, count cap = 1 across BOTH readers.
    let shared = SharedShardCache::new(1, usize::MAX);
    let r0 =
        BackedCsrReader::with_shared_cache(ScxReader::open(&p0).unwrap(), 0, Arc::clone(&shared));
    let r1 =
        BackedCsrReader::with_shared_cache(ScxReader::open(&p1).unwrap(), 1, Arc::clone(&shared));

    // Warm (file 0, shard 0); keys are namespaced, so file 1 shard 0 is distinct.
    let _ = r0.read_shard_cached_arc(0).unwrap();
    assert!(r0.cache_contains(0));
    assert!(!r1.cache_contains(0));

    // Reading (file 1, shard 0) under a cap of 1 evicts (file 0, shard 0):
    // proves the budget is shared, not per-reader.
    let _ = r1.read_shard_cached_arc(0).unwrap();
    assert!(r1.cache_contains(0));
    assert!(
        !r0.cache_contains(0),
        "shared cap=1 must evict the other reader's shard"
    );

    // The count cap is the one shared cap, seen identically through either reader.
    assert_eq!(r0.cache_capacity(), 1);
    assert_eq!(r1.cache_capacity(), 1);
}
