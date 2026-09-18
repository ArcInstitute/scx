use super::cache::WeightedLruCache;
use super::csc::check_csc_sidecar_fresh;
use super::*;
use crate::encoder::{encode_one_shard, EncodeShardOptions};
use crate::header::FileHeader;
use crate::section::SectionType;
use crate::writer::{ScxWriter, ShardBuffers};
use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_sparse::concatenate_csr;
use std::sync::Arc;
use tempfile::TempDir;

// --- Test helpers (same as reader_tests.rs helpers) ---

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

/// Write a **row-group-framed** `CodecId::None` file (F5 Phase 1: v4 file /
/// v2 shards, multi-entry `BlockIndex`) with `n_shards` shards each framed at
/// `row_group_rows` rows per group. Returns the path + the reference full CSR
/// (read back through the framed decode path). Uses `encode_one_shard(...,
/// Some(row_group_rows))` + `write_preencoded_shard` since the plain
/// `write_csr_shard` writer method only emits the unframed layout.
fn write_framed_file(
    dir: &TempDir,
    n_obs: usize,
    n_vars: usize,
    n_shards: usize,
    row_group_rows: u32,
    codec: CodecId,
) -> (std::path::PathBuf, ScxCsr) {
    let path = dir.path().join("framed.scx");
    let total_nnz = n_obs * 2;
    let mut header = sample_header(n_obs as u64, n_vars as u64, total_nnz as u64);
    header.format_version = crate::header::CURRENT_FORMAT_VERSION; // v4 (framed)
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
        let (indptr, indices, values_u8) = sample_shard_data(shard_rows, n_vars);
        let values_f32: Vec<f32> = values_u8.iter().map(|&v| v as f32).collect();
        let mut opts = EncodeShardOptions::new(
            format!("X_shard_{s}"),
            SectionType::CsrShard,
            n_vars as u64,
            (s * rows_per_shard) as u64,
            0,
        );
        opts.explicit_codec = Some(codec);
        opts.framing = Some(crate::encoder::FramingConfig {
            row_group_rows,
            target_nnz: None,
            trial: false,
            decode_target: None,
        });
        let pre = encode_one_shard(&indptr, &indices, &values_f32, &opts).unwrap();
        writer.write_preencoded_shard(pre).unwrap();
    }
    writer.finish().unwrap();

    let full_csr = ScxReader::open(&path)
        .unwrap()
        .read_all_csr_shards()
        .unwrap();
    (path, full_csr)
}

/// F5 Phase 1: scattered `read_rows_with` over a **row-group-framed** None file
/// (no Scx1 sidecar) must decode only touched groups via the block index and
/// return rows byte-identical to a full decode — for unsorted, duplicate, and
/// far-apart requests.
#[test]
fn read_rows_with_block_index_matches_full_decode() {
    let dir = TempDir::new().unwrap();
    // 2 shards × 32 rows, groups of 4 rows.
    let (path, full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::None);

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
            let lo = full.indptr[row as usize] as usize;
            let hi = full.indptr[row as usize + 1] as usize;
            assert_eq!(out[i].0, full.indices[lo..hi], "indices row {row}");
            assert_eq!(out[i].1, full.data[lo..hi], "data row {row}");
        }
    };

    let backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    // Sparse, unsorted, far-apart, with a duplicate (5) and a cross-group run.
    let sparse_rows = [2u64, 5, 6, 40, 41, 63, 5];
    let out = gather(&backed, &sparse_rows);
    assert_matches(&out, &sparse_rows);

    // Dense group also correct (full-shard fallback path over a framed shard).
    let backed2 = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    let dense_rows: Vec<u64> = (0..10).chain(32..42).collect();
    let out2 = gather(&backed2, &dense_rows);
    assert_matches(&out2, &dense_rows);
}

/// F5 Phase 2: framed scattered reads must be byte-identical to a full decode for
/// **every** codec (None + ShufDeltaZstd + Zstd/Lz4/Pcodec), and a sparse cold
/// gather over a compressed framed shard (no Scx1 sidecar) must take the
/// block-index path. Proves `encode_shard_framed` + generic `decode_row_group`
/// end-to-end through the writer and backed reader.
#[test]
fn read_rows_with_block_index_all_codecs() {
    use std::sync::atomic::Ordering;
    let sparse_rows = [2u64, 5, 6, 40, 41, 63, 5];
    for codec in [
        CodecId::None,
        CodecId::ShufDeltaZstd,
        CodecId::Zstd,
        CodecId::Lz4Shuffle,
        CodecId::Pcodec,
    ] {
        let dir = TempDir::new().unwrap();
        let (path, full) = write_framed_file(&dir, 64, 100, 2, 4, codec);

        let mut backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
        let m = backed.enable_metrics();
        let mut out: Vec<(Vec<i32>, Vec<f32>)> = vec![Default::default(); sparse_rows.len()];
        backed
            .read_rows_with(&sparse_rows, |i, idx, data| {
                out[i] = (idx.to_vec(), data.to_vec());
                Ok(())
            })
            .unwrap();
        for (i, &row) in sparse_rows.iter().enumerate() {
            let lo = full.indptr[row as usize] as usize;
            let hi = full.indptr[row as usize + 1] as usize;
            assert_eq!(
                out[i].0,
                full.indices[lo..hi],
                "{codec:?} indices row {row}"
            );
            assert_eq!(out[i].1, full.data[lo..hi], "{codec:?} data row {row}");
        }
        // Both sparse cold shard groups served via the codec-agnostic block index.
        assert_eq!(
            m.block_index_groups.load(Ordering::Relaxed),
            2,
            "{codec:?}: sparse cold gather must take the block-index path",
        );

        // OPT-FORMATIO-1: the same gather again is served from the row-group
        // LRU — every codec's decoded groups round-trip through the cache
        // byte-identically, and nothing is decoded a second time.
        let decoded_groups = m.row_group_misses.load(Ordering::Relaxed);
        assert!(
            decoded_groups > 0,
            "{codec:?}: premise — groups were retained"
        );
        let mut again: Vec<(Vec<i32>, Vec<f32>)> = vec![Default::default(); sparse_rows.len()];
        backed
            .read_rows_with(&sparse_rows, |i, idx, data| {
                again[i] = (idx.to_vec(), data.to_vec());
                Ok(())
            })
            .unwrap();
        assert_eq!(
            again, out,
            "{codec:?}: cached groups must reproduce the decode"
        );
        assert_eq!(
            m.row_group_misses.load(Ordering::Relaxed),
            decoded_groups,
            "{codec:?}: warm gather must decode nothing"
        );
        assert_eq!(
            m.row_group_hits.load(Ordering::Relaxed),
            decoded_groups,
            "{codec:?}: warm gather must hit every group"
        );
        assert_eq!(m.block_index_groups.load(Ordering::Relaxed), 4);
    }
}

/// F5 Phase 1 adoption counter: a sparse cold gather over a framed None file
/// bumps `CacheMetrics::block_index_groups` (not `sidecar_groups`); a dense
/// gather falls back to `full_shard_groups`. Symmetric with the sidecar
/// adoption test.
#[test]
fn read_rows_with_bumps_block_index_counters() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let (path, _full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::None);

    fn gather(backed: &BackedCsrReader, rows: &[u64]) {
        backed
            .read_rows_with(rows, |_i, _idx, _data| Ok(()))
            .unwrap();
    }

    // Sparse cold group across both shards (≤4 rows/shard << shard_rows/4 = 8)
    // ⇒ both shard groups take the block-index path (no Scx1 sidecar on None).
    let mut backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    let m = backed.enable_metrics();
    gather(&backed, &[2u64, 5, 6, 40, 41]);
    assert_eq!(
        m.block_index_groups.load(Ordering::Relaxed),
        2,
        "both sparse cold shard groups must be served via the block index",
    );
    assert_eq!(
        m.full_shard_groups.load(Ordering::Relaxed),
        0,
        "no full-shard decode for a sparse cold framed gather",
    );

    // Dense group ⇒ full-shard fallback; fresh reader keeps counters clean.
    let mut backed2 = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    let m2 = backed2.enable_metrics();
    let dense_rows: Vec<u64> = (0..10).chain(32..42).collect();
    gather(&backed2, &dense_rows);
    assert_eq!(m2.block_index_groups.load(Ordering::Relaxed), 0);
    assert_eq!(
        m2.full_shard_groups.load(Ordering::Relaxed),
        2,
        "both dense shard groups must fall back to full-shard decode",
    );
}

/// `any_shard_framed` reports whether the fast scattered
/// path can fire — true on a framed file, false on a legacy unframed one. The
/// Python `IndexPlanDataset` constructor uses it to warn when a caller requests
/// `scatter_block_index=True` on an unframed file that can only full-shard-decode.
#[test]
fn any_shard_framed_detects_framing() {
    let dir = TempDir::new().unwrap();

    // Framed file (v4 / shard v2, multi-entry block index) ⇒ true.
    let (framed_path, _full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::None);
    let framed = BackedCsrReader::new(ScxReader::open(&framed_path).unwrap(), 4);
    assert!(
        framed.any_shard_framed(),
        "a row-group-framed file must report at least one framed shard",
    );

    // Legacy unframed file (plain write_csr_shard ⇒ shard v1) ⇒ false.
    let (unframed, _full2) = write_test_file_and_open(&dir, 64, 100, 2, 4);
    assert!(
        !unframed.any_shard_framed(),
        "an all-unframed legacy file must report no framed shards",
    );
}

/// A framed shard's scattered gather routes through the
/// codec-agnostic block index. `block_index_eligible` gates the group-level
/// row-group path — the loader-adoption goal for framed training files.
#[test]
fn read_rows_with_block_index_framed_shard() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let (path, full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::ShufDeltaZstd);

    // scatter_block_index defaults on (env), so the framed gather routes through
    // the block index rather than pre-warming + full-shard decoding.
    let cache = SharedShardCache::new(4, usize::MAX);
    let mut backed = BackedCsrReader::with_shared_cache(ScxReader::open(&path).unwrap(), 0, cache);
    let m = backed.enable_metrics();

    let sparse_rows = [2u64, 5, 6, 40, 41];
    let mut out: Vec<(Vec<i32>, Vec<f32>)> = vec![Default::default(); sparse_rows.len()];
    backed
        .read_rows_with(&sparse_rows, |i, idx, data| {
            out[i] = (idx.to_vec(), data.to_vec());
            Ok(())
        })
        .unwrap();
    for (i, &row) in sparse_rows.iter().enumerate() {
        let lo = full.indptr[row as usize] as usize;
        let hi = full.indptr[row as usize + 1] as usize;
        assert_eq!(out[i].0, full.indices[lo..hi], "indices row {row}");
        assert_eq!(out[i].1, full.data[lo..hi], "data row {row}");
    }
    assert_eq!(
        m.block_index_groups.load(Ordering::Relaxed),
        2,
        "framed gather must take the block-index path",
    );
    assert_eq!(
        m.full_shard_groups.load(Ordering::Relaxed),
        0,
        "no full-shard decode for a sparse cold framed gather",
    );
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
// The dense shard cache
// -----------------------------------------------------------------------
//
// The dense LRU had no test of any kind — not eviction, not the byte budget,
// not the singleflight around it — despite carrying its own transcription of
// all three. These pin the behaviour that has to survive being folded into the
// shared cache.

/// Bytes one decoded 25-row × 6-col f32 shard occupies by the cache's own
/// measure. Taken from the reader rather than recomputed, so the fixture cannot
/// drift from `RecordBatch::get_array_memory_size()`.
fn dense_shard_bytes(dir: &TempDir) -> usize {
    let path = write_obsm_file(dir, 100, 4, 6);
    let reader = ScxReader::open(&path).unwrap();
    let mut backed = BackedDenseReader::new_obsm(reader, "X_emb", 4).unwrap();
    let metrics = backed.enable_metrics();
    backed.read_row_indices(&[0]).unwrap();
    metrics.bytes_inserted.load(Ordering::Relaxed) as usize
}

#[test]
fn dense_cache_evicts_to_fit_the_byte_budget() {
    let probe_dir = tempfile::tempdir().unwrap();
    let one_shard = dense_shard_bytes(&probe_dir);
    assert!(one_shard > 0, "probe measured a zero-byte shard");

    let dir = tempfile::tempdir().unwrap();
    let path = write_obsm_file(&dir, 100, 4, 6);
    let reader = ScxReader::open(&path).unwrap();
    // Count cap of 4 (every shard fits) but a byte budget that holds two, so
    // any eviction here is the byte cap's doing and not the count cap's.
    let mut backed =
        BackedDenseReader::new_obsm_with_byte_budget(reader, "X_emb", 4, one_shard * 2).unwrap();
    let metrics = backed.enable_metrics();

    // One row from each of the four shards, in order.
    backed.read_row_indices(&[0, 25, 50, 75]).unwrap();

    assert!(
        metrics.evictions.load(Ordering::Relaxed) >= 2,
        "four shards into a two-shard byte budget must evict at least twice, got {}",
        metrics.evictions.load(Ordering::Relaxed)
    );
    assert!(
        metrics.peak_bytes_in_cache.load(Ordering::Relaxed) as usize <= one_shard * 2,
        "peak {} exceeded the {} byte budget",
        metrics.peak_bytes_in_cache.load(Ordering::Relaxed),
        one_shard * 2
    );
    // The first shard must be gone; the last must be resident.
    assert!(
        !backed.cache_contains(0),
        "shard 0 should have been evicted"
    );
    assert!(backed.cache_contains(3), "shard 3 was just inserted");
}

#[test]
fn dense_cache_admits_a_shard_larger_than_the_whole_budget() {
    // Documented behaviour: an entry that cannot fit on its own is still
    // inserted after everything else is evicted. Refusing to cache it would
    // defeat the cache for any outsized shard — and, worse, silently, since
    // every read would then look like a miss with no eviction to explain it.
    let dir = tempfile::tempdir().unwrap();
    let path = write_obsm_file(&dir, 100, 4, 6);
    let reader = ScxReader::open(&path).unwrap();
    let mut backed = BackedDenseReader::new_obsm_with_byte_budget(reader, "X_emb", 4, 1).unwrap();
    let metrics = backed.enable_metrics();

    backed.read_row_indices(&[0]).unwrap();

    assert!(
        backed.cache_contains(0),
        "an oversized shard must still be cached, not silently dropped"
    );
    // And it is served from the cache on the next read, not re-decoded.
    // Asserted on `misses`, not `hits`: one `read_row_indices` warms and then
    // gathers, so it takes more than one hit off the cache per call and the
    // hit count is an artefact of the gather, not of the caching.
    backed.read_row_indices(&[1]).unwrap();
    assert_eq!(
        metrics.misses.load(Ordering::Relaxed),
        1,
        "the shard must be decoded exactly once across both reads"
    );
    assert!(metrics.hits.load(Ordering::Relaxed) > 0);
}

#[test]
fn dense_reader_dedups_concurrent_decodes_of_one_shard() {
    // The dense singleflight, asserted the same way the CSR one is: with
    // singleflight, total misses can never exceed the number of distinct
    // shards touched, however many threads race. Without it, each thread
    // decodes its own copy.
    use std::thread;

    let dir = tempfile::tempdir().unwrap();
    let path = write_obsm_file(&dir, 100, 4, 6);
    let reader = ScxReader::open(&path).unwrap();
    let mut backed = BackedDenseReader::new_obsm(reader, "X_emb", 4).unwrap();
    let metrics = backed.enable_metrics();
    let backed = Arc::new(backed);

    // Rows 0 and 50 live in shards 0 and 2 — two distinct shards.
    let n_threads = 8;
    let handles: Vec<_> = (0..n_threads)
        .map(|_| {
            let b = Arc::clone(&backed);
            thread::spawn(move || {
                b.read_row_indices(&[0, 50]).unwrap();
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }

    let misses = metrics.misses.load(Ordering::Relaxed);
    assert!(
        misses <= 2,
        "expected ≤ 2 misses across {n_threads} concurrent readers of 2 shards, got {misses} \
         — the dense singleflight is not deduplicating"
    );
    assert!(misses >= 1, "every shard had to be decoded once");
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

// -----------------------------------------------------------------------
// shards_with_kept_rows — the predicate the masked kernels skip on
// -----------------------------------------------------------------------
//
// Pure index arithmetic, so it is pinned exhaustively here rather than
// inferred from the kernels' output: a kernel that visits a shard it did not
// need still produces the right numbers, so bit-identity alone cannot say
// whether the skip list is correct.

#[test]
fn kept_rows_confined_to_one_shard_select_only_that_shard() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
    // 4 shards of 3 rows each: [0,3), [3,6), [6,9), [9,12).
    assert_eq!(backed.index().shards_with_kept_rows(&[6, 7, 8]), vec![2]);
    assert_eq!(backed.index().shards_with_kept_rows(&[0]), vec![0]);
    assert_eq!(backed.index().shards_with_kept_rows(&[11]), vec![3]);
}

#[test]
fn kept_rows_straddling_a_boundary_select_both_shards() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
    // Half-open ranges: row 3 belongs to shard 1, not shard 0.
    assert_eq!(backed.index().shards_with_kept_rows(&[2, 3]), vec![0, 1]);
    assert_eq!(backed.index().shards_with_kept_rows(&[3]), vec![1]);
}

#[test]
fn a_gap_in_kept_rows_skips_the_shards_it_spans() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
    // Shards 1 and 2 contribute nothing — the case the change exists for.
    assert_eq!(backed.index().shards_with_kept_rows(&[1, 10]), vec![0, 3]);
}

#[test]
fn every_row_kept_selects_every_shard() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
    let all: Vec<u64> = (0..12).collect();
    assert_eq!(
        backed.index().shards_with_kept_rows(&all),
        vec![0, 1, 2, 3],
        "the unsubset case must not skip anything"
    );
}

#[test]
fn no_kept_rows_selects_no_shard() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
    assert!(backed.index().shards_with_kept_rows(&[]).is_empty());
    // Out of range on both sides, not just empty.
    assert!(backed.index().shards_with_kept_rows(&[99]).is_empty());
}

#[test]
fn shards_with_kept_rows_agrees_with_shards_for_indices_when_sorted() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
    // The pre-existing sort-then-scan helper is the oracle for the ascending
    // input both are contracted for. They may not diverge: the kernels walk
    // `kept_rows` with the same `partition_point` pair this uses.
    for kept in [
        vec![0u64],
        vec![1, 10],
        vec![2, 3],
        vec![4, 5, 6, 7],
        (0..12).collect::<Vec<u64>>(),
    ] {
        assert_eq!(
            backed.index().shards_with_kept_rows(&kept),
            backed.index().shards_for_indices(&kept),
            "kept={kept:?}"
        );
    }
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
// BackedCsrReader::read_row_indices — bounded gather (PR C / REC-1)
// -----------------------------------------------------------------------

/// `read_row_indices` shares `read_rows_with`'s out-of-range contract: an
/// error, not a silently shorter result. (Before PR C it dropped the row via
/// `shards_for_indices`, so a caller asking for 5 rows could get 4 back.)
#[test]
fn test_read_row_indices_out_of_range_errors() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 4);

    let err = backed
        .read_row_indices(&[5, 99])
        .expect_err("an out-of-range row must error, not be dropped");
    assert!(
        err.to_string().contains("99"),
        "the error must name the offending row: {err}"
    );
}

/// Request order is the output order; every occurrence of a duplicate is its
/// own output row (the loader's cellset gather pins the same on `[7, 7, 31]`).
#[test]
fn test_read_row_indices_duplicates_and_unsorted() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

    let rows = [11u64, 0, 7, 7, 4, 3];
    let out = backed.read_row_indices(&rows).unwrap();
    assert_eq!(out.n_rows(), rows.len());
    assert_eq!(out.shape.1, full.shape.1);
    for (i, &row) in rows.iter().enumerate() {
        let expected = full.row_slice(row as usize, row as usize + 1).unwrap();
        let actual = out.row_slice(i, i + 1).unwrap();
        assert_eq!(actual.indices, expected.indices, "i={i} row={row} indices");
        assert_eq!(actual.data, expected.data, "i={i} row={row} data");
    }
    // The indptr is exact — no over-allocation left behind.
    assert_eq!(out.indices.len(), out.nnz());
    assert_eq!(out.indices.capacity(), out.indices.len());
    assert_eq!(out.data.capacity(), out.data.len());
}

/// On a framed file the gather must be byte-identical to a full decode for
/// every codec — sparse groups go through the block index, dense ones through
/// the full-shard path, and the two must agree with `read_all`.
#[test]
fn read_row_indices_matches_full_decode_on_framed_file_all_codecs() {
    let sparse_rows = [2u64, 5, 6, 40, 41, 63, 5];
    let dense_rows: Vec<u64> = (0..10).chain(32..42).collect();
    for codec in [
        CodecId::None,
        CodecId::ShufDeltaZstd,
        CodecId::Zstd,
        CodecId::Lz4Shuffle,
        CodecId::Pcodec,
    ] {
        let dir = TempDir::new().unwrap();
        let (path, full) = write_framed_file(&dir, 64, 100, 2, 4, codec);
        for rows in [&sparse_rows[..], &dense_rows[..]] {
            let backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
            let out = backed.read_row_indices(rows).unwrap();
            assert_eq!(out.n_rows(), rows.len(), "{codec:?}");
            for (i, &row) in rows.iter().enumerate() {
                let lo = full.indptr[row as usize] as usize;
                let hi = full.indptr[row as usize + 1] as usize;
                let olo = out.indptr[i] as usize;
                let ohi = out.indptr[i + 1] as usize;
                assert_eq!(
                    &out.indices[olo..ohi],
                    &full.indices[lo..hi],
                    "{codec:?} indices row {row}"
                );
                assert_eq!(
                    &out.data[olo..ohi],
                    &full.data[lo..hi],
                    "{codec:?} data row {row}"
                );
            }
        }
    }
}

/// A sparse cold gather over a framed shard takes the block-index path — the
/// touched row groups are decoded, the shard is **not** full-decoded into the
/// LRU. This is the caching-policy change PR C makes to `read_row_indices`
/// (it used to full-decode and cache unconditionally).
#[test]
fn read_row_indices_takes_the_block_index_path() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let (path, _full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::None);

    let mut backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    let m = backed.enable_metrics();
    backed.read_row_indices(&[2u64, 5, 6, 40, 41]).unwrap();
    assert_eq!(
        m.block_index_groups.load(Ordering::Relaxed),
        2,
        "both sparse cold shard groups must be served via the block index",
    );
    assert_eq!(m.full_shard_groups.load(Ordering::Relaxed), 0);
    assert_eq!(
        m.misses.load(Ordering::Relaxed),
        0,
        "a block-index gather must not full-decode a shard into the LRU",
    );
}

/// The indptr-only prescan indexes the decoded indptr with catalog-derived
/// local rows, so a shard whose header under-reports its row count must be
/// rejected with `InvalidCatalog` (the same guard `decode_shard` applies) —
/// never an index-out-of-bounds panic.
#[test]
fn read_row_indices_rejects_a_shard_whose_header_underreports_rows() {
    let dir = TempDir::new().unwrap();
    let path = write_shard_shrunk_in_header(&dir, 8, 10, 4);
    let backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);

    let err = backed
        .read_row_indices(&[0, 7])
        .expect_err("a short shard must error, not panic");
    assert!(
        matches!(err, ScxError::InvalidCatalog(_)),
        "expected InvalidCatalog, got {err:?}"
    );

    // `read_rows` shares the prescan.
    let err = backed
        .read_rows(0, 8)
        .expect_err("read_rows must reject the same shard");
    assert!(
        matches!(err, ScxError::InvalidCatalog(_)),
        "expected InvalidCatalog, got {err:?}"
    );
}

/// A bulk `read_rows` — more full-path shards than the LRU holds — copies the
/// shards that are already resident out of the cache and decodes the rest
/// *uncached*, keeping the LRU's entries as they were: no evictions, no new
/// entries (the copied residents are promoted, as any hit is), and a later
/// small read of the resident range still hits. Before
/// PR C the single up-front warm (truncated to `cache_shards`) evicted the
/// resident shards and the gather loop decoded them again.
#[test]
fn read_rows_bulk_serves_resident_shards_from_the_cache_and_leaves_it_alone() {
    use std::sync::atomic::Ordering;
    let dir = tempfile::tempdir().unwrap();
    // 8 shards × 8 rows, a 4-slot cache.
    let (mut backed, full) = write_test_file_and_open(&dir, 64, 10, 8, 4);
    let m = backed.enable_metrics();

    // Warm shards 2 and 3 (rows 16..32) the way an earlier `X[a:b]` would.
    let pre = backed.read_rows(16, 32).unwrap();
    assert_eq!(pre.n_rows(), 16);
    assert_eq!(m.misses.load(Ordering::Relaxed), 2);
    let hits_before = m.hits.load(Ordering::Relaxed);

    let out = backed.read_rows(0, 64).unwrap();
    assert_eq!(out.indptr, full.indptr);
    assert_eq!(out.indices, full.indices);
    assert_eq!(out.data, full.data);
    assert_eq!(
        m.misses.load(Ordering::Relaxed),
        2,
        "a bulk read must not push shards through the LRU",
    );
    assert_eq!(m.evictions.load(Ordering::Relaxed), 0);
    assert_eq!(
        m.hits.load(Ordering::Relaxed) - hits_before,
        2,
        "the two resident shards are served from the cache",
    );
    // Single allocation: the buffers are exactly the result, no headroom.
    assert_eq!(out.indices.capacity(), out.indices.len());
    assert_eq!(out.data.capacity(), out.data.len());

    // The cache still holds exactly what it held before the bulk read.
    let again = backed.read_rows(16, 32).unwrap();
    assert_eq!(again.indices, pre.indices);
    assert_eq!(m.misses.load(Ordering::Relaxed), 2);
}

/// `read_rows` refuses a range past `n_obs` instead of sizing its output from
/// it: the pre-sized assembly would leave the uncovered tail of `indptr` at
/// zero — a non-monotone CSR — and `end = u64::MAX` would overflow the
/// allocation. (The pre-PR-C concatenate path returned a shorter matrix; the
/// Python and R bindings clamp or reject before the call, so only a direct
/// Rust caller can reach this.)
#[test]
fn read_rows_rejects_a_range_past_n_obs() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);
    let n_obs = backed.n_obs() as u64;

    for (start, end) in [(n_obs - 2, n_obs + 10), (n_obs, n_obs + 1), (0, u64::MAX)] {
        let err = backed
            .read_rows(start, end)
            .expect_err("a range past n_obs must error, not emit a malformed CSR");
        assert!(
            err.to_string().contains("out of range"),
            "{start}..{end}: {err}"
        );
    }
    // The boundary itself is fine.
    let tail = backed.read_rows(n_obs - 2, n_obs).unwrap();
    assert_eq!(tail.indices, full.row_slice(10, 12).unwrap().indices);
    // `start >= end` is still the empty matrix, wherever it sits.
    assert_eq!(backed.read_rows(n_obs + 5, n_obs + 5).unwrap().n_rows(), 0);
}

/// A catalog whose CSR shards leave a gap in the row axis must make `read_rows`
/// refuse the range, not return a shorter matrix (a range wholly inside the gap
/// used to come back as `(0, n_vars)`) or a non-monotone `indptr` (a range
/// straddling it).
#[test]
fn read_rows_rejects_a_catalog_gap() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gap.scx");
    let n_vars = 10usize;
    // Header says 12 rows; shards cover 0..4 and 8..12 — rows 4..8 are nobody's.
    let header = sample_header(12, n_vars as u64, 16);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(12)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
    for row_start in [0u64, 8] {
        let (indptr, indices, values) = sample_shard_data(4, n_vars);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_start,
            )
            .unwrap();
    }
    writer.finish().unwrap();
    let backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);

    for (start, end) in [(0u64, 12u64), (4, 8), (2, 10), (6, 9)] {
        let err = backed
            .read_rows(start, end)
            .expect_err("a range touching the gap must error");
        assert!(
            matches!(err, ScxError::InvalidCatalog(_)),
            "{start}..{end}: expected InvalidCatalog, got {err:?}"
        );
    }
    // Ranges inside a covered shard still read.
    assert_eq!(backed.read_rows(0, 4).unwrap().n_rows(), 4);
    assert_eq!(backed.read_rows(9, 12).unwrap().n_rows(), 3);
}

/// A gap and an overlap of equal size sum to the requested row count, so a
/// length-only tiling check accepts them; the positional check does not.
/// Shards `0..4`, `6..10`, `8..12` on a 12-row file: rows 4..6 are nobody's and
/// rows 8..10 are two shards'.
#[test]
fn read_rows_rejects_an_equal_gap_and_overlap() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gap_overlap.scx");
    let n_vars = 10usize;
    let header = sample_header(12, n_vars as u64, 24);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(12)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
    for row_start in [0u64, 6, 8] {
        let (indptr, indices, values) = sample_shard_data(4, n_vars);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_start,
            )
            .unwrap();
    }
    writer.finish().unwrap();
    let backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);

    // Window lengths 4 + 4 + 4 == 12 requested rows — only position tells.
    let err = backed
        .read_rows(0, 12)
        .expect_err("an equal gap and overlap must not pass as a tiling");
    assert!(
        matches!(err, ScxError::InvalidCatalog(_)),
        "expected InvalidCatalog, got {err:?}"
    );
    assert!(
        err.to_string().contains("previous shard ended at 4"),
        "{err}"
    );
    // A range inside one shard, or across the overlap-free prefix, still reads.
    assert_eq!(backed.read_rows(0, 4).unwrap().n_rows(), 4);
    assert_eq!(backed.read_rows(1, 3).unwrap().n_rows(), 2);
}

/// `BackedCsrReader::new` on a multimodal file indexes every modality's shards
/// (each tiles `[0, n_obs)` on its own), so a range read would copy two shards'
/// windows into the same output rows. The positional tiling check refuses it;
/// the scoped `for_modality` reader reads the range.
#[test]
fn read_rows_rejects_overlapping_modalities_on_an_unscoped_reader() {
    use crate::modality::ModalityType;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("mm.scx");
    let n_obs: u64 = 4;
    let header = sample_header(n_obs, 4, 0);
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
    let indptr: Vec<u64> = vec![0, 1, 2, 3, 4];
    let values: Vec<u8> = vec![1, 1, 1, 1];
    let rna_indices: Vec<u32> = vec![0, 0, 0, 0];
    let atac_indices: Vec<u32> = vec![3, 3, 3, 3];
    writer
        .write_csr_shard_for(
            rna_id,
            0,
            ShardBuffers::new(
                &indptr,
                &rna_indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
            ),
        )
        .unwrap();
    writer
        .write_csr_shard_for(
            atac_id,
            0,
            ShardBuffers::new(
                &indptr,
                &atac_indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
            ),
        )
        .unwrap();
    writer.finish().unwrap();

    let unscoped = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    let err = unscoped
        .read_rows(0, n_obs)
        .expect_err("two modalities' shards overlap every output row");
    assert!(
        matches!(err, ScxError::InvalidCatalog(_)),
        "expected InvalidCatalog, got {err:?}"
    );
    assert!(err.to_string().contains("for_modality"), "{err}");

    let scoped = BackedCsrReader::for_modality(ScxReader::open(&path).unwrap(), atac_id, 4);
    let out = scoped.read_rows(0, n_obs).unwrap();
    assert_eq!(out.n_rows(), 4);
    assert_eq!(out.indices, vec![3, 3, 3, 3]);
}

/// A range that fits the LRU still goes through it (warm + copy), so the
/// sequential chunk iterator's re-reads keep hitting.
#[test]
fn read_rows_cache_sized_range_still_populates_the_cache() {
    use std::sync::atomic::Ordering;
    let dir = tempfile::tempdir().unwrap();
    let (mut backed, full) = write_test_file_and_open(&dir, 64, 10, 8, 4);
    let m = backed.enable_metrics();

    // 4 full shards == cache_shards ⇒ cached path.
    let out = backed.read_rows(0, 32).unwrap();
    assert_eq!(out.indices, full.row_slice(0, 32).unwrap().indices);
    assert_eq!(m.misses.load(Ordering::Relaxed), 4);
    // The warm decoded the four shards; the copy loop then took each from the
    // cache (four hits). A second read is all hits, no decode.
    let hits_after_first = m.hits.load(Ordering::Relaxed);
    let again = backed.read_rows(0, 32).unwrap();
    assert_eq!(again.indices, out.indices);
    assert_eq!(
        m.misses.load(Ordering::Relaxed),
        4,
        "second read is all hits"
    );
    assert_eq!(m.hits.load(Ordering::Relaxed) - hits_after_first, 4);
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

// -----------------------------------------------------------------------
// The CSC shard cache
// -----------------------------------------------------------------------
//
// The CSC cache had no eviction, capacity or put/get test of any kind. These
// pin the count-cap behaviour that survived the unification (`new` still opens
// count-only) and the byte accounting and singleflight that are new to it.

#[test]
fn csc_cache_evicts_at_the_count_cap() {
    let dir = tempfile::tempdir().unwrap();
    // 12 vars / 3 per shard = 4 CSC shards, cache capped at 2.
    let (path, _) = write_csc_test_file(&dir, 12, 12, 3);
    let reader = ScxReader::open(&path).unwrap();
    let mut backed = BackedCscReader::new(reader, 2).unwrap();
    let metrics = backed.enable_metrics();
    assert_eq!(backed.n_shards(), 4);

    for s in [0usize, 1, 2] {
        backed.read_shard_cached(s).unwrap();
    }
    assert_eq!(
        metrics.misses.load(Ordering::Relaxed),
        3,
        "three cold shards, three misses"
    );
    assert_eq!(
        metrics.evictions.load(Ordering::Relaxed),
        1,
        "the third insert into a 2-entry cache evicts shard 0"
    );

    // Shards 1 and 2 are the residents.
    backed.read_shard_cached(2).unwrap();
    backed.read_shard_cached(1).unwrap();
    assert_eq!(metrics.hits.load(Ordering::Relaxed), 2);
    assert_eq!(
        metrics.misses.load(Ordering::Relaxed),
        3,
        "resident shards must not re-decode"
    );

    // Shard 0 was evicted, so it decodes again — and displaces another.
    backed.read_shard_cached(0).unwrap();
    assert_eq!(
        metrics.misses.load(Ordering::Relaxed),
        4,
        "shard 0 was evicted and must miss"
    );
    assert_eq!(metrics.evictions.load(Ordering::Relaxed), 2);
}

#[test]
fn csc_cache_counts_an_eviction_but_not_a_replacement() {
    // `put_with_budget` uses `LruCache::push`, not `put`, precisely because
    // `put` returns `None` when a new key evicts the LRU — so a count-cap
    // eviction would go uncounted. Nothing checked that, and the distinction is
    // invisible until someone reads the eviction counter and believes it.
    //
    // Driven at the CSC instantiation of the shared cache: `usize::MAX` bytes
    // so only the count cap can evict, which is what this is about.
    let metrics = Arc::new(CacheMetrics::default());
    let mut cache: WeightedLruCache<usize, ScxCsc> = WeightedLruCache::new(2, usize::MAX);
    cache.metrics = Some(Arc::clone(&metrics));
    let empty = || {
        Arc::new(ScxCsc::new_unchecked(
            (4, 1),
            vec![0, 0],
            Vec::new(),
            Vec::new(),
        ))
    };

    cache.put_with_budget(0, empty());
    cache.put_with_budget(0, empty());
    assert_eq!(
        metrics.evictions.load(Ordering::Relaxed),
        0,
        "re-inserting the same key is a replacement, not an eviction"
    );

    cache.put_with_budget(1, empty());
    cache.put_with_budget(2, empty());
    assert_eq!(
        metrics.evictions.load(Ordering::Relaxed),
        1,
        "the third distinct key must evict one and be counted"
    );
    assert!(cache.get(&0).is_none(), "key 0 is the LRU and must be gone");
    assert!(cache.get(&2).is_some());
}

#[test]
fn csc_metrics_now_report_bytes_like_the_other_two_caches() {
    // **Changed behaviour.** This test previously asserted these were zero, and
    // the hand-rolled CSC cache said they were "intentionally left at zero".
    // Sharing the
    // cache means CSC accounts for bytes the way CSR and dense always have, so
    // anything sampling `CacheMetrics` off a `BackedCscReader` sees real
    // numbers where it saw zeros. `CacheMetrics`'s fields are unchanged — it is
    // pyscx's FFI shape — only which caches populate them.
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = write_csc_test_file(&dir, 12, 12, 3);
    let reader = ScxReader::open(&path).unwrap();
    let mut backed = BackedCscReader::new(reader, 4).unwrap();
    let metrics = backed.enable_metrics();

    for s in 0..backed.n_shards() {
        backed.read_shard_cached(s).unwrap();
    }

    assert!(metrics.misses.load(Ordering::Relaxed) > 0, "shards decoded");
    assert!(
        metrics.bytes_inserted.load(Ordering::Relaxed) > 0,
        "CSC now measures what it caches"
    );
    assert!(
        metrics.peak_bytes_in_cache.load(Ordering::Relaxed) > 0,
        "and reports the high-water mark"
    );
    assert_eq!(
        metrics.duplicate_waiters.load(Ordering::Relaxed),
        0,
        "nothing raced in this test, so nobody waited"
    );
}

#[test]
fn shard_cache_decodes_one_contended_key_exactly_once() {
    // The deterministic counterpart to the two reader-level dedup tests below.
    // Those spawn threads and bound `misses` by the number of distinct shards,
    // which is the same guarantee the CSR test has always asserted — but it is
    // only *evidence* of singleflight when the threads actually contend. A
    // scheduler that runs one thread to completion first would populate the
    // cache, turn every peer into an ordinary LRU hit, and satisfy the bound
    // with the singleflight removed. That was measured not to happen here (8
    // threads over 2 shards gave 11–15 misses with `in_flight` disabled), but
    // "did not happen on this machine" is not determinism.
    //
    // So drive `ShardCache` directly and make the contention a property of the
    // test rather than of the scheduler: whoever wins leadership holds its
    // decode open until every follower has registered as a waiter, so no peer
    // can race past on a cache hit. The deadline is what keeps a *broken*
    // singleflight a bounded failure rather than a hang — with every thread a
    // leader, `duplicate_waiters` never rises and each spins to the deadline,
    // then `decodes` is 8 and the assert fires.
    use std::sync::atomic::AtomicUsize;
    use std::thread;
    use std::time::{Duration, Instant};

    const FOLLOWERS: u64 = 7;
    let cache: Arc<ShardCache<usize, ScxCsc>> = ShardCache::new(4, usize::MAX);
    let metrics = cache.enable_metrics();
    let decodes = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..=FOLLOWERS)
        .map(|_| {
            let c = Arc::clone(&cache);
            let d = Arc::clone(&decodes);
            let m = Arc::clone(&metrics);
            thread::spawn(move || {
                c.get_or_decode(0usize, || {
                    d.fetch_add(1, Ordering::Relaxed);
                    let deadline = Instant::now() + Duration::from_secs(10);
                    while m.duplicate_waiters.load(Ordering::Relaxed) < FOLLOWERS
                        && Instant::now() < deadline
                    {
                        std::thread::yield_now();
                    }
                    Ok(Arc::new(ScxCsc::new_unchecked(
                        (4, 1),
                        vec![0, 0],
                        Vec::new(),
                        Vec::new(),
                    )))
                })
                .unwrap();
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(
        decodes.load(Ordering::Relaxed),
        1,
        "{} threads contending on one key must produce exactly one decode",
        FOLLOWERS + 1
    );
    assert_eq!(
        metrics.duplicate_waiters.load(Ordering::Relaxed),
        FOLLOWERS,
        "every non-leader must have waited on the leader's Condvar"
    );
    assert_eq!(metrics.misses.load(Ordering::Relaxed), 1);
    assert_eq!(
        metrics.hits.load(Ordering::Relaxed),
        FOLLOWERS,
        "each waiter is served from the cache the leader filled"
    );
}

#[test]
fn dense_reader_enable_metrics_is_idempotent() {
    // **Changed behaviour**, and the one the PR description originally denied:
    // this used to install a fresh `CacheMetrics` per call, so a second call
    // reset the counters and orphaned the first caller's handle. It now returns
    // the same accumulating handle, matching `BackedCsrReader`.
    let dir = tempfile::tempdir().unwrap();
    let path = write_obsm_file(&dir, 100, 4, 6);
    let reader = ScxReader::open(&path).unwrap();
    let mut backed = BackedDenseReader::new_obsm(reader, "X_emb", 4).unwrap();

    let first = backed.enable_metrics();
    backed.read_row_indices(&[0]).unwrap();
    let after_one = first.misses.load(Ordering::Relaxed);
    assert!(after_one > 0, "a cold read must miss");

    let second = backed.enable_metrics();
    assert!(
        Arc::ptr_eq(&first, &second),
        "re-enabling must hand back the same counters, not a fresh set"
    );
    assert_eq!(
        second.misses.load(Ordering::Relaxed),
        after_one,
        "counters accumulate for the life of the reader; they are not reset"
    );
}

#[test]
fn csr_reader_enable_metrics_is_idempotent() {
    // The CSR reader has behaved this way since the shared cache existed — its
    // rustdoc just said the opposite ("subsequent calls rebind to a fresh
    // metrics handle") until round 2 caught the contradiction with the CSC and
    // dense docs. Pinned here so the corrected sentence is enforced rather than
    // asserted; this is the method `IndexPlanLoader` and `plan_engine` call.
    let dir = tempfile::tempdir().unwrap();
    let (mut backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 4);

    let first = backed.enable_metrics();
    backed.read_rows(0, 3).unwrap();
    let after_one = first.misses.load(Ordering::Relaxed);
    assert!(after_one > 0, "a cold read must miss");

    let second = backed.enable_metrics();
    assert!(
        Arc::ptr_eq(&first, &second),
        "re-enabling must hand back the same counters, not a fresh set"
    );
    assert_eq!(
        second.misses.load(Ordering::Relaxed),
        after_one,
        "counters accumulate for the life of the reader; they are not reset"
    );
}

#[test]
fn csc_reader_enable_metrics_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = write_csc_test_file(&dir, 12, 12, 3);
    let reader = ScxReader::open(&path).unwrap();
    let mut backed = BackedCscReader::new(reader, 4).unwrap();

    let first = backed.enable_metrics();
    backed.read_shard_cached(0).unwrap();
    let after_one = first.misses.load(Ordering::Relaxed);
    assert!(after_one > 0);

    let second = backed.enable_metrics();
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(second.misses.load(Ordering::Relaxed), after_one);
}

#[test]
fn csc_reader_dedups_concurrent_decodes_of_one_shard() {
    // **New behaviour.** The hand-rolled CSC cache had no singleflight at all:
    // N threads that all missed on the same cold shard each decoded their own
    // copy. Sharing the cache means one decodes and the rest wait, so total
    // misses are bounded by the number of distinct shards — the same guarantee
    // the CSR and dense readers have had.
    use std::thread;

    let dir = tempfile::tempdir().unwrap();
    // 24 vars / 6 per shard = 4 CSC shards; touch two of them.
    let (path, _) = write_csc_test_file(&dir, 64, 24, 6);
    let reader = ScxReader::open(&path).unwrap();
    let mut backed = BackedCscReader::new(reader, 4).unwrap();
    let metrics = backed.enable_metrics();
    assert_eq!(backed.n_shards(), 4);
    let backed = Arc::new(backed);

    let n_threads = 8;
    let handles: Vec<_> = (0..n_threads)
        .map(|_| {
            let b = Arc::clone(&backed);
            thread::spawn(move || {
                b.read_shard_cached(0).unwrap();
                b.read_shard_cached(2).unwrap();
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }

    let misses = metrics.misses.load(Ordering::Relaxed);
    assert!(
        misses <= 2,
        "expected ≤ 2 misses across {n_threads} concurrent readers of 2 shards, got {misses} \
         — the CSC singleflight is not deduplicating"
    );
    assert!(misses >= 1, "every shard had to be decoded once");
}

#[test]
fn a_csc_reader_can_now_be_opened_under_a_byte_budget() {
    // The capability CSC could not express before: `BackedCscReader::new` is
    // count-only (`usize::MAX`), and `with_byte_budget` bounds it in bytes.
    // Both are the same cache now, so this is a constructor, not a new path.
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = write_csc_test_file(&dir, 12, 12, 3);

    let probe = ScxReader::open(&path).unwrap();
    let mut probe_reader = BackedCscReader::new(probe, 4).unwrap();
    let probe_metrics = probe_reader.enable_metrics();
    probe_reader.read_shard_cached(0).unwrap();
    let one_shard = probe_metrics.bytes_inserted.load(Ordering::Relaxed) as usize;
    assert!(one_shard > 0);

    // Count cap of 4 (all four fit) but room for two, so any eviction is the
    // byte cap's doing.
    let reader = ScxReader::open(&path).unwrap();
    let mut backed = BackedCscReader::with_byte_budget(reader, 4, one_shard * 2).unwrap();
    let metrics = backed.enable_metrics();
    for s in 0..backed.n_shards() {
        backed.read_shard_cached(s).unwrap();
    }

    assert!(
        metrics.evictions.load(Ordering::Relaxed) >= 2,
        "four shards into a two-shard byte budget must evict, got {}",
        metrics.evictions.load(Ordering::Relaxed)
    );
    assert!(
        metrics.peak_bytes_in_cache.load(Ordering::Relaxed) as usize <= one_shard * 2,
        "peak exceeded the byte budget"
    );
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

/// `BackedCscReader`'s binary-search override and the trait's linear default
/// must select the **same** shards for every column range, including the
/// degenerate ones.
///
/// The two are separate implementations of one predicate — the whole point of
/// putting the predicate on the trait is that the GPU staging path stops
/// carrying a third copy. So the property to pin is not "the override returns
/// something sensible" but "the override and the default cannot disagree", and
/// the range space here is small enough to check exhaustively rather than
/// sampled.
///
/// The reference below is written out in the test rather than reached through
/// the trait, because `BackedCscReader` overrides the default — calling
/// `csc_shards_for_col_range` on it can only ever exercise the override.
#[test]
fn backed_csc_shards_for_col_range_matches_the_trait_default_predicate() {
    let dir = tempfile::tempdir().unwrap();
    // 8 rows x 12 cols, 4 cols per shard -> 3 shards: [0,4), [4,8), [8,12).
    let (path, _) = write_csc_test_file(&dir, 8, 12, 4);
    let reader = ScxReader::open(&path).unwrap();
    let csc = BackedCscReader::new(reader, 0).unwrap();
    let src: &dyn crate::ColumnShardSource = &csc;

    let ranges: Vec<(u32, u32)> = (0..src.n_csc_shards())
        .map(|i| src.csc_shard_col_range(i).unwrap())
        .collect();
    // The reference IS `col_range_overlaps`, deliberately.
    //
    // This closure used to restate the rule by hand, complete with its own
    // `lo >= hi` guard — which meant the test pinned the binary search to a
    // *third* copy of the predicate rather than to the helper, while
    // `col_range_overlaps`'s docstring claimed the opposite. It also hid a real
    // disagreement: the helper had no empty-range guard, so the two answered
    // differently on an interior empty range and this test could not see it
    // (Cursor Agent - Grok 4.6 High).
    //
    // The hand-computed anchors below are what keep this from passing by both
    // sides being wrong together.
    let reference = |lo: u32, hi: u32| -> Vec<usize> {
        (0..ranges.len())
            .filter(|&i| crate::col_range_overlaps(ranges[i].0, ranges[i].1, &(lo..hi)))
            .collect()
    };

    // Exhaustive over a window that runs past n_vars on both ends.
    for lo in 0..=14u32 {
        for hi in 0..=14u32 {
            assert_eq!(
                src.csc_shards_for_col_range(lo..hi),
                reference(lo, hi),
                "override and default disagree on [{lo}, {hi})"
            );
        }
    }

    // Hand-computed anchors, so the exhaustive loop above cannot pass by both
    // sides being wrong in the same direction.
    assert_eq!(src.csc_shards_for_col_range(0..4), vec![0]);
    assert_eq!(src.csc_shards_for_col_range(3..5), vec![0, 1]);
    assert_eq!(src.csc_shards_for_col_range(4..8), vec![1]);
    assert_eq!(src.csc_shards_for_col_range(0..12), vec![0, 1, 2]);
    assert!(src.csc_shards_for_col_range(12..14).is_empty());
    assert!(src.csc_shards_for_col_range(5..5).is_empty());

    // The empty-range case, stated against the helper directly: an interior
    // empty range overlaps nothing, even though both half-open comparisons
    // taken alone would say otherwise (`8 > 5 && 4 < 5`).
    //
    // Built from bindings rather than literals so clippy's
    // `reversed_empty_ranges` does not reject the very inputs under test.
    let empty = |at: u32| at..at;
    assert!(!crate::col_range_overlaps(4, 8, &empty(5)));
    assert!(!crate::col_range_overlaps(0, 12, &empty(6)));
    // ...and an inverted range likewise.
    let inverted = {
        let (lo, hi) = (9u32, 3u32);
        lo..hi
    };
    assert!(!crate::col_range_overlaps(0, 12, &inverted));
    // Sanity: the same shard against a one-column range that does overlap.
    assert!(crate::col_range_overlaps(4, 8, &(5..6)));
}

/// The CSC size hint is a true upper bound on every shard, and tight on a file
/// we wrote (catalog stats are exact there).
///
/// `max_rows` is the major axis, which on this layout is **columns** — asserting
/// it against `n_cols()` rather than `n_rows()` is the point, since the field
/// name comes from the row-major side and reading it as rows would size the
/// `indptr` buffer against the wrong axis.
#[test]
fn backed_csc_shard_size_hint_bounds_and_matches_every_decoded_shard() {
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = write_csc_test_file(&dir, 8, 12, 4);
    let reader = ScxReader::open(&path).unwrap();
    let csc = BackedCscReader::new(reader, 0).unwrap();
    let src: &dyn crate::ColumnShardSource = &csc;

    let hint = src
        .csc_shard_size_hint()
        .expect("catalog carries CSC stats");

    let mut observed_cols = 0usize;
    let mut observed_nnz = 0usize;
    for i in 0..src.n_csc_shards() {
        let shard = src.read_csc_shard(i).unwrap();
        assert!(
            shard.n_cols() <= hint.max_rows,
            "shard {i} has {} columns, above the hint's major-axis bound {}",
            shard.n_cols(),
            hint.max_rows
        );
        assert!(
            shard.data.len() <= hint.max_nnz,
            "shard {i} has {} nonzeros, above the hint's bound {}",
            shard.data.len(),
            hint.max_nnz
        );
        observed_cols = observed_cols.max(shard.n_cols());
        observed_nnz = observed_nnz.max(shard.data.len());
    }
    assert_eq!(
        hint.max_rows, observed_cols,
        "hint is looser than it needs to be on the major axis"
    );
    assert_eq!(
        hint.max_nnz, observed_nnz,
        "hint is looser than it needs to be on nnz"
    );
    assert!(hint.decoded_bytes() > 0);
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
    let rna_shard = ShardBuffers::new(
        &rna_indptr,
        &rna_indices,
        &rna_values,
        CodecId::None,
        ValueEncoding::Uint8,
    );
    writer.write_csr_shard_for(rna_id, 0, rna_shard).unwrap();
    // ATAC: cells 0..4 all express only gene 3.
    let atac_indptr: Vec<u64> = vec![0, 1, 2, 3, 4];
    let atac_indices: Vec<u32> = vec![3, 3, 3, 3];
    let atac_values: Vec<u8> = vec![1, 1, 1, 1];
    let atac_shard = ShardBuffers::new(
        &atac_indptr,
        &atac_indices,
        &atac_values,
        CodecId::None,
        ValueEncoding::Uint8,
    );
    writer.write_csr_shard_for(atac_id, 0, atac_shard).unwrap();
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

/// Characterization: what `BackedCsrReader::new` — the *unscoped* constructor —
/// does on a multimodal file. Nothing pinned this before, and the answer is
/// surprising enough to be worth a test rather than a reading of the source.
///
/// `::new` builds its index from `CatalogView::csr_shards_sorted()`, which
/// filters on `section_type == CsrShard` with **no `modality_id` predicate**.
/// On a two-modality file that means one index over *both* modalities' shards,
/// whose row ranges overlap (each modality independently tiles `[0, n_obs)`),
/// while `n_vars` comes from the file header — the max across modalities, not
/// any one modality's. `for_modality` is the scoped constructor and is what
/// every production caller uses (`pyscx/src/experiment.rs`,
/// `pyscx/src/backed/multimodal.rs`, `pyscx/src/convert/multimodal.rs`,
/// `scx-convert/src/export_filter.rs`).
///
/// This test asserts today's behaviour, deliberately. It is not an endorsement
/// — it is the tripwire that makes a change to it visible, and the safety net
/// for the `backed.rs` split in Phase 3b.
#[test]
fn backed_csr_reader_new_on_a_multimodal_file_folds_every_modality() {
    use crate::modality::ModalityType;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multimodal_unscoped.scx");

    // The two modalities have DIFFERENT column counts, so "which n_vars did
    // the reader take?" has an observable answer.
    let n_obs: u64 = 4;
    const RNA_VARS: u64 = 3;
    const ATAC_VARS: u64 = 5;
    let header = sample_header(n_obs, ATAC_VARS, 0);
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
    writer
        .write_var_for(rna_id, &sample_var(RNA_VARS as usize))
        .unwrap();
    writer
        .write_var_for(atac_id, &sample_var(ATAC_VARS as usize))
        .unwrap();
    writer.set_modality_n_vars(rna_id, RNA_VARS).unwrap();
    writer.set_modality_n_vars(atac_id, ATAC_VARS).unwrap();

    // RNA values 11..=14 at column 0; ATAC values 21..=24 at column 4. The
    // disjoint value ranges make "whose rows came back?" answerable.
    let rna_shard = ShardBuffers::new(
        &[0, 1, 2, 3, 4],
        &[0, 0, 0, 0],
        &[11, 12, 13, 14],
        CodecId::None,
        ValueEncoding::Uint8,
    );
    writer.write_csr_shard_for(rna_id, 0, rna_shard).unwrap();
    let atac_shard = ShardBuffers::new(
        &[0, 1, 2, 3, 4],
        &[4, 4, 4, 4],
        &[21, 22, 23, 24],
        CodecId::None,
        ValueEncoding::Uint8,
    );
    writer.write_csr_shard_for(atac_id, 0, atac_shard).unwrap();
    writer.finish().unwrap();

    let unscoped = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);

    // 1. The index spans BOTH modalities: two shards, and their row ranges
    //    overlap because each modality tiles [0, n_obs) independently.
    assert_eq!(
        unscoped.index().n_shards(),
        2,
        "::new indexes every CsrShard regardless of modality"
    );
    assert_eq!(unscoped.index().shard_range(0), Some((0, 4)));
    assert_eq!(
        unscoped.index().shard_range(1),
        Some((0, 4)),
        "the two modalities' row ranges overlap — this is the shape of the trap"
    );

    // 2. `shape` is header-derived, so it reports 4 rows while the index
    //    covers 8 shard-rows. The pair disagreeing is the finding.
    assert_eq!(
        unscoped.shape(),
        (4, ATAC_VARS as usize),
        "n_obs from the header; n_vars is the file-wide max, not RNA's 3"
    );

    // 3. `read_all` concatenates every shard, so it returns 8 rows for a
    //    4-cell file, with both modalities' values folded together.
    let all = unscoped.read_all().unwrap();
    assert_eq!(
        all.shape,
        (8, ATAC_VARS as usize),
        "read_all folds both modalities: 4 RNA rows + 4 ATAC rows"
    );
    assert_eq!(
        all.data,
        vec![11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 23.0, 24.0]
    );
    assert_eq!(all.indices, vec![0, 0, 0, 0, 4, 4, 4, 4]);

    // 4. The scoped constructor is the one that answers per modality.
    let rna = BackedCsrReader::for_modality(ScxReader::open(&path).unwrap(), rna_id, 0);
    assert_eq!(rna.shape(), (4, RNA_VARS as usize));
    let rna_rows = rna.read_rows(0, 4).unwrap();
    assert_eq!(rna_rows.data, vec![11.0, 12.0, 13.0, 14.0]);
    assert_eq!(rna_rows.shape, (4, RNA_VARS as usize));
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

// ---------------------------------------------------------------------------
// Phase 4.2 — decode-prefetch bit-identity
//
// The aggregation kernels above stream shards through
// `prefetch::for_each_shard_ordered_uncached` instead of a plain `for` loop.
// The pipeline decodes up to `depth` shards concurrently but *consumes* them on
// the calling thread in strict shard order, so every f64 accumulation and every
// `Vec::extend` concatenation must see the exact sequence the old loop did.
//
// "Must" is the claim these tests discharge, against a hand-written sequential
// reference over `read_shard_uncached` — not against another prefetched kernel.
// ---------------------------------------------------------------------------

/// Multi-shard **float** fixture, built so that shard order is *observable* in
/// the f64 accumulators.
///
/// Two properties, both load-bearing:
///
/// 1. **Float, not integer.** Integer values sum exactly at every grouping, so
///    a bit-identity assertion over them proves nothing.
/// 2. **Shard 0 holds `+1e16`, shard 1 holds `-1e16`, the rest hold small
///    inexact fractions.** This is the property that took two attempts to get
///    right. The kernels reduce *per-shard partial sums*, not raw values, so
///    what has to be order-sensitive is the merge of five numbers — and the
///    obvious "cycle through three magnitudes" fixture is completely blind to
///    it (measured: 0 of the 119 non-identity permutations of a 5-shard visit
///    order changed any column sum). With a cancelling pair the small terms are
///    swallowed or retained depending on *when* they are added, and 108 of
///    those 119 orders now give a different f64 result.
///    [`fixture_is_sensitive_to_shard_order`] pins the property so the
///    bit-identity tests below cannot quietly become vacuous.
fn write_float_file_and_open(
    dir: &TempDir,
    n_obs: usize,
    n_vars: usize,
    n_shards: usize,
) -> BackedCsrReader {
    let path = dir.path().join("float.scx");
    let nnz_per_row = 3usize;
    let header = sample_header(n_obs as u64, n_vars as u64, (n_obs * nnz_per_row) as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    let rows_per_shard = n_obs / n_shards;
    for s in 0..n_shards {
        let row_start = s * rows_per_shard;
        let shard_rows = if s == n_shards - 1 {
            n_obs - row_start
        } else {
            rows_per_shard
        };
        let mut indptr = vec![0u64];
        let mut indices: Vec<u32> = Vec::new();
        let mut values: Vec<f32> = Vec::new();
        for r in 0..shard_rows {
            let g = row_start + r;
            // Three strictly-increasing columns per row, values alternating
            // between huge, tiny and awkward-fraction so summation order is
            // observable in the low mantissa bits.
            let cols = [
                (g % n_vars) as u32,
                ((g + 3) % n_vars) as u32,
                ((g + 7) % n_vars) as u32,
            ];
            let mut cols: Vec<u32> = cols.to_vec();
            cols.sort_unstable();
            cols.dedup();
            for (k, &c) in cols.iter().enumerate() {
                indices.push(c);
                values.push(match s {
                    0 => 1.0e16f32,
                    1 => -1.0e16f32,
                    _ => [1.0f32 / 3.0, 2.0 / 7.0, 11.0 / 13.0][(g + k) % 3],
                });
            }
            indptr.push(indices.len() as u64);
        }
        let value_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &value_bytes,
                CodecId::None,
                ValueEncoding::Float32,
                row_start as u64,
            )
            .unwrap();
    }
    writer.finish().unwrap();
    BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0)
}

/// Guard against a vacuous pass: the prefetch pipeline only engages with a
/// multi-thread rayon pool, `depth > 1` and more than one shard. If any of
/// those is false every kernel silently takes the same sequential path as the
/// reference and these tests prove nothing.
///
/// This is also why the two tests that call it are `parallel`-gated rather than
/// left to run: without the feature `prefetch::pool_threads()` is 1, so
/// `prefetch_depth()` is 1, and the guard fires — correctly. Both sides of the
/// comparison would be the sequential loop.
#[cfg(feature = "parallel")]
fn assert_prefetch_engages(backed: &BackedCsrReader) {
    assert!(
        crate::prefetch::prefetch_depth() > 1,
        "SCX_ACCEL_PREFETCH_DEPTH<=1 in this process — the prefetch path is \
         disabled and the bit-identity assertions below are vacuous"
    );
    #[cfg(feature = "parallel")]
    assert!(
        rayon::current_num_threads() > 1,
        "single-thread rayon pool — the prefetch path is disabled and the \
         bit-identity assertions below are vacuous"
    );
    assert!(
        backed.index().n_shards() > 1,
        "single-shard fixture — the prefetch path is disabled"
    );
}

#[cfg(feature = "parallel")]
fn assert_bits_eq(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length differs");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{label}[{i}]: {g} is not bit-identical to the sequential {w}"
        );
    }
}

/// The premise check for the two bit-identity tests: on this fixture, visiting
/// shards in a different order *does* change the f64 column sums. Without it a
/// reordering regression would slip through silently and both tests would still
/// be green — the failure mode that made the previous fixture worthless.
#[test]
fn fixture_is_sensitive_to_shard_order() {
    let dir = tempfile::tempdir().unwrap();
    let (n_obs, n_vars, n_shards) = (120usize, 11usize, 5usize);
    let backed = write_float_file_and_open(&dir, n_obs, n_vars, n_shards);

    let sum_in = |order: &[usize]| -> Vec<f64> {
        let mut sums = vec![0.0f64; n_vars];
        for &idx in order {
            let csr = backed.read_shard_uncached(idx).unwrap();
            for (s, p) in sums.iter_mut().zip(csr.col_sums().iter()) {
                *s += p;
            }
        }
        sums
    };
    let forward: Vec<usize> = (0..n_shards).collect();
    let reversed: Vec<usize> = (0..n_shards).rev().collect();
    let a = sum_in(&forward);
    let b = sum_in(&reversed);
    assert!(
        a.iter().zip(&b).any(|(x, y)| x.to_bits() != y.to_bits()),
        "fixture is order-insensitive: reversing the shard visit order left \
         every column sum bit-identical, so the bit-identity tests below \
         cannot detect a reordering regression"
    );
}

#[cfg(feature = "parallel")]
#[test]
fn prefetched_aggregations_are_bit_identical_to_a_sequential_loop() {
    let dir = tempfile::tempdir().unwrap();
    let (n_obs, n_vars, n_shards) = (120usize, 11usize, 5usize);
    let backed = write_float_file_and_open(&dir, n_obs, n_vars, n_shards);
    assert_prefetch_engages(&backed);

    // --- Sequential reference: the pre-4.2 loop shape, spelled out once. ---
    let mut ref_row_sums: Vec<f64> = Vec::new();
    let mut ref_row_nnz: Vec<i64> = Vec::new();
    let mut ref_row_sq: Vec<f64> = Vec::new();
    let mut ref_row_var: Vec<f64> = Vec::new();
    let mut ref_row_max: Vec<f64> = Vec::new();
    let mut ref_row_min: Vec<f64> = Vec::new();
    let mut ref_col_sums = vec![0.0f64; n_vars];
    let mut ref_col_nnz = vec![0u32; n_vars];
    let mut ref_col_max = vec![f64::NEG_INFINITY; n_vars];
    let mut ref_col_min = vec![f64::INFINITY; n_vars];
    for idx in 0..n_shards {
        let csr = backed.read_shard_uncached(idx).unwrap();
        ref_row_sums.extend(csr.row_sums());
        ref_row_nnz.extend(csr.row_nnz());
        ref_row_sq.extend(csr.row_sum_of_squares());
        ref_row_var.extend(csr.row_var().unwrap());
        ref_row_max.extend(csr.row_max().unwrap());
        ref_row_min.extend(csr.row_min().unwrap());
        for (s, p) in ref_col_sums.iter_mut().zip(csr.col_sums().iter()) {
            *s += p;
        }
        for (c, p) in ref_col_nnz.iter_mut().zip(csr.col_nnz().iter()) {
            *c = c.saturating_add(*p);
        }
        for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
            let c = col as usize;
            ref_col_max[c] = ref_col_max[c].max(val as f64);
            ref_col_min[c] = ref_col_min[c].min(val as f64);
        }
    }
    // Implicit-zero corrections, matching `col_max` / `col_min`.
    let total_nnz: Vec<usize> = ref_col_nnz.iter().map(|&c| c as usize).collect();
    for c in 0..n_vars {
        if total_nnz[c] < n_obs {
            ref_col_max[c] = if ref_col_max[c] == f64::NEG_INFINITY {
                0.0
            } else {
                ref_col_max[c].max(0.0)
            };
            ref_col_min[c] = if ref_col_min[c] == f64::INFINITY {
                0.0
            } else {
                ref_col_min[c].min(0.0)
            };
        }
    }

    // --- Prefetched kernels ---
    assert_bits_eq("row_sums", &backed.row_sums().unwrap(), &ref_row_sums);
    assert_bits_eq(
        "row_sum_of_squares",
        &backed.row_sum_of_squares().unwrap(),
        &ref_row_sq,
    );
    assert_bits_eq("row_var", &backed.row_var().unwrap(), &ref_row_var);
    assert_bits_eq("row_max", &backed.row_max().unwrap(), &ref_row_max);
    assert_bits_eq("row_min", &backed.row_min().unwrap(), &ref_row_min);
    assert_bits_eq("col_sums", &backed.col_sums().unwrap(), &ref_col_sums);
    assert_bits_eq("col_max", &backed.col_max().unwrap(), &ref_col_max);
    assert_bits_eq("col_min", &backed.col_min().unwrap(), &ref_col_min);
    assert_eq!(backed.row_nnz().unwrap(), ref_row_nnz, "row_nnz");
    assert_eq!(backed.col_nnz().unwrap(), ref_col_nnz, "col_nnz");

    // The fused twins must agree bit-for-bit with their unfused counterparts,
    // which is what makes them a safe substitution (4.1's contract, re-checked
    // now that both sides stream through the prefetch pipeline).
    let (nnz, sums) = backed.row_nnz_and_sums().unwrap();
    assert_eq!(nnz, ref_row_nnz, "row_nnz_and_sums: nnz");
    assert_bits_eq("row_nnz_and_sums: sums", &sums, &ref_row_sums);
    let (csums, cnnz) = backed.col_sums_and_nnz().unwrap();
    assert_bits_eq("col_sums_and_nnz: sums", &csums, &ref_col_sums);
    assert_eq!(cnnz, ref_col_nnz, "col_sums_and_nnz: nnz");

    // `col_var` is two prefetched passes (means, then squared deviations).
    let means: Vec<f64> = ref_col_sums.iter().map(|&s| s / n_obs as f64).collect();
    let mut ref_sq_dev = vec![0.0f64; n_vars];
    for idx in 0..n_shards {
        let csr = backed.read_shard_uncached(idx).unwrap();
        for (s, p) in ref_sq_dev
            .iter_mut()
            .zip(csr.col_var_partial(&means).iter())
        {
            *s += p;
        }
    }
    let ref_col_var: Vec<f64> = (0..n_vars)
        .map(|c| {
            let n_zeros = n_obs - total_nnz[c];
            (ref_sq_dev[c] + n_zeros as f64 * means[c] * means[c]) / n_obs as f64
        })
        .collect();
    assert_bits_eq("col_var", &backed.col_var().unwrap(), &ref_col_var);
}

/// The oracle for one kept set: every masked kernel against a sequential
/// re-derivation from the raw shards.
///
/// Parametrized on `kept` because the two call sites exercise different code.
/// A set spanning every shard checks the arithmetic; a set that leaves whole
/// shards empty additionally checks that skipping those shards changes
/// nothing — which is only a real check while a *spanning* set is asserted
/// too, since a kernel that wrongly skipped a shard holding kept rows would
/// pass the second case on its own.
#[cfg(feature = "parallel")]
fn assert_masked_aggregations_bit_identical(
    backed: &BackedCsrReader,
    n_vars: usize,
    n_shards: usize,
    kept: &[u64],
) {
    let n_kept = kept.len();

    let mut ref_sums = vec![0.0f64; n_vars];
    let mut ref_counts = vec![0u32; n_vars];
    let mut ref_max = vec![f64::NEG_INFINITY; n_vars];
    let mut ref_min = vec![f64::INFINITY; n_vars];
    let mut ref_nnz = vec![0usize; n_vars];
    for idx in 0..n_shards {
        let csr = backed.read_shard_uncached(idx).unwrap();
        let (s_start, s_end) = backed.index().shard_range(idx).unwrap();
        let lo = kept.partition_point(|&r| r < s_start);
        let hi = kept.partition_point(|&r| r < s_end);
        for &g in &kept[lo..hi] {
            let local = (g - s_start) as usize;
            for j in csr.indptr[local] as usize..csr.indptr[local + 1] as usize {
                let c = csr.indices[j] as usize;
                let v = csr.data[j] as f64;
                ref_sums[c] += v;
                ref_counts[c] += 1;
                ref_nnz[c] += 1;
                ref_max[c] = ref_max[c].max(v);
                ref_min[c] = ref_min[c].min(v);
            }
        }
    }
    for c in 0..n_vars {
        if ref_nnz[c] < n_kept {
            ref_max[c] = if ref_max[c] == f64::NEG_INFINITY {
                0.0
            } else {
                ref_max[c].max(0.0)
            };
            ref_min[c] = if ref_min[c] == f64::INFINITY {
                0.0
            } else {
                ref_min[c].min(0.0)
            };
        }
    }

    assert_bits_eq(
        "col_sums_masked",
        &backed.col_sums_masked(kept).unwrap(),
        &ref_sums,
    );
    assert_bits_eq(
        "col_max_masked",
        &backed.col_max_masked(kept).unwrap(),
        &ref_max,
    );
    assert_bits_eq(
        "col_min_masked",
        &backed.col_min_masked(kept).unwrap(),
        &ref_min,
    );
    let nnz_as_f64: Vec<f64> = ref_counts.iter().map(|&c| c as f64).collect();
    assert_bits_eq(
        "col_nnz_masked",
        &backed.col_nnz_masked(kept).unwrap(),
        &nnz_as_f64,
    );

    let (fs, fc) = backed.col_sums_and_nnz_masked(kept).unwrap();
    assert_bits_eq("col_sums_and_nnz_masked: sums", &fs, &ref_sums);
    assert_eq!(fc, ref_counts, "col_sums_and_nnz_masked: nnz");

    // col_var_masked: two prefetched passes over the kept rows.
    let means: Vec<f64> = ref_sums.iter().map(|&s| s / n_kept as f64).collect();
    let mut sq_dev = vec![0.0f64; n_vars];
    for idx in 0..n_shards {
        let csr = backed.read_shard_uncached(idx).unwrap();
        let (s_start, s_end) = backed.index().shard_range(idx).unwrap();
        let lo = kept.partition_point(|&r| r < s_start);
        let hi = kept.partition_point(|&r| r < s_end);
        for &g in &kept[lo..hi] {
            let local = (g - s_start) as usize;
            for j in csr.indptr[local] as usize..csr.indptr[local + 1] as usize {
                let c = csr.indices[j] as usize;
                let d = csr.data[j] as f64 - means[c];
                sq_dev[c] += d * d;
            }
        }
    }
    let ref_var: Vec<f64> = (0..n_vars)
        .map(|c| {
            let n_zeros = n_kept - ref_nnz[c];
            (sq_dev[c] + n_zeros as f64 * means[c] * means[c]) / n_kept as f64
        })
        .collect();
    assert_bits_eq(
        "col_var_masked",
        &backed.col_var_masked(kept).unwrap(),
        &ref_var,
    );
}

#[cfg(feature = "parallel")]
#[test]
fn prefetched_masked_aggregations_are_bit_identical_to_a_sequential_loop() {
    let dir = tempfile::tempdir().unwrap();
    let (n_obs, n_vars, n_shards) = (120usize, 11usize, 5usize);
    let backed = write_float_file_and_open(&dir, n_obs, n_vars, n_shards);
    assert_prefetch_engages(&backed);

    // Strictly ascending kept set spanning every shard — the masked kernels
    // `partition_point` it per shard, so it must stay sorted.
    let spanning: Vec<u64> = (0..n_obs as u64).filter(|r| r % 3 != 1).collect();
    assert_eq!(
        backed.index().shards_with_kept_rows(&spanning).len(),
        n_shards,
        "premise: this set reaches every shard, so nothing is skipped"
    );
    assert_masked_aggregations_bit_identical(&backed, n_vars, n_shards, &spanning);

    // 5 shards of 24 rows: keep only the first and the last, so shards 1, 2
    // and 3 hold no visible row and are never decoded. The set above cannot
    // exercise that — it was written to span every shard on purpose.
    let sparse_shards: Vec<u64> = (0..24u64).chain(96..120u64).collect();
    assert_eq!(
        backed.index().shards_with_kept_rows(&sparse_shards),
        vec![0, 4],
        "premise: three of five shards contribute nothing"
    );
    assert_masked_aggregations_bit_identical(&backed, n_vars, n_shards, &sparse_shards);
}

/// `shard_size_hint` reads catalog statistics — no decode — and reports upper
/// bounds that actually bound the shards.
///
/// The GPU staging path pre-sizes its pinned + device buffers from this and
/// derates its decode-prefetch depth by `decoded_bytes()`, so an *under*-bound
/// would silently reintroduce the grow-and-realloc it exists to remove.
#[test]
fn shard_size_hint_bounds_every_shard() {
    let dir = tempfile::tempdir().unwrap();
    // 12 rows over 4 shards: 3, 3, 3, 3 rows; sample_shard_data puts 2 nnz per
    // row, so every shard has 6 nnz.
    let (backed, _full) = write_test_file_and_open(&dir, 12, 8, 4, 2);

    let hint = crate::ShardSource::shard_size_hint(&backed).expect("catalog carries shard stats");

    let mut observed_max_rows = 0usize;
    let mut observed_max_nnz = 0usize;
    for s in 0..crate::ShardSource::n_shards(&backed) {
        let csr = backed.read_shard_uncached(s).unwrap();
        observed_max_rows = observed_max_rows.max(csr.n_rows());
        observed_max_nnz = observed_max_nnz.max(csr.data.len());
    }
    assert!(observed_max_nnz > 0, "premise: the fixture has nonzeros");

    assert!(
        hint.max_rows >= observed_max_rows,
        "row hint {} under-bounds the largest shard ({observed_max_rows} rows)",
        hint.max_rows
    );
    assert!(
        hint.max_nnz >= observed_max_nnz,
        "nnz hint {} under-bounds the largest shard ({observed_max_nnz} nnz)",
        hint.max_nnz
    );
    // On an unprojected, undeleted reader the bounds are exact.
    assert_eq!(hint.max_rows, observed_max_rows);
    assert_eq!(hint.max_nnz, observed_max_nnz);

    // The byte estimate the depth clamp divides by: indptr i64 + indices i32 +
    // data f32.
    assert_eq!(
        hint.decoded_bytes(),
        (hint.max_rows as u64 + 1) * 8 + hint.max_nnz as u64 * 8
    );
}

/// A source with no cheap hint declines rather than reporting zero — a caller
/// must be able to tell "unknown" from "empty", since it sizes buffers from it.
#[test]
fn shard_size_hint_defaults_to_none() {
    let csr = ScxCsr::new_unchecked((2, 3), vec![0i64, 1, 2], vec![0i32, 2], vec![1.0f32, 2.0]);
    let src = crate::shard_source::SingleShardSource { csr: &csr };
    assert!(
        crate::ShardSource::shard_size_hint(&src).is_none(),
        "the trait default must be None, not Some(zero)"
    );
}

// ---------------------------------------------------------------------------
// catalog_int_value_max
// ---------------------------------------------------------------------------

/// Integer-encoded shards: the catalog carries the exact maximum, and reading
/// it costs no decode.
///
/// This is the value the `pdex_ref` log1p probe applies its `< 30` heuristic
/// to, so a wrong answer here silently changes which `GeomMeanMode` a backed
/// DE run picks.
#[test]
fn catalog_int_value_max_reports_exact_max_for_integer_shards() {
    let dir = tempfile::tempdir().unwrap();
    let (backed, full) = write_test_file_and_open(&dir, 40, 8, 4, 4);

    let expected = full.data.iter().fold(0.0f32, |a, &b| a.max(b)).round() as u32;
    assert!(expected > 0, "fixture must contain nonzero values");
    assert_eq!(backed.catalog_int_value_max(), Some(expected));
}

/// Float-encoded shards record `value_max = 0` by design, so the catalog can
/// prove nothing about a nonempty float matrix — `None` means *unknown*, and
/// callers must not read it as "the max is 0".
#[test]
fn catalog_int_value_max_is_none_for_float_shards() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("float.scx");
    let (n_obs, n_vars) = (12usize, 5usize);
    let header = sample_header(n_obs as u64, n_vars as u64, (n_obs * 2) as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    // Two fractional values per row — well above the heuristic's threshold, so
    // a caller that mistook `None` for "max is 0" would answer "log1p" here.
    let mut indptr: Vec<u64> = vec![0];
    let mut indices: Vec<u32> = Vec::new();
    let mut values: Vec<u8> = Vec::new();
    for row in 0..n_obs {
        indices.push((row % n_vars) as u32);
        indices.push(((row + 1) % n_vars) as u32);
        values.extend_from_slice(&(100.5f32).to_le_bytes());
        values.extend_from_slice(&(200.25f32).to_le_bytes());
        indptr.push(indptr.last().unwrap() + 2);
    }
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Float32,
            0,
        )
        .unwrap();
    writer.finish().unwrap();

    let backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 2);
    assert_eq!(
        backed.catalog_int_value_max(),
        None,
        "float shards record no value range; None must mean unknown, not zero"
    );
}

/// An empty matrix is the one case where `value_max == 0` is not ambiguous —
/// `nnz == 0` proves it, so the method answers instead of refusing.
#[test]
fn catalog_int_value_max_reports_zero_for_an_empty_matrix() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.scx");
    let (n_obs, n_vars) = (6usize, 3usize);
    let header = sample_header(n_obs as u64, n_vars as u64, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
    writer
        .write_csr_shard(
            &vec![0u64; n_obs + 1],
            &[],
            &[],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer.finish().unwrap();

    let backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 2);
    assert_eq!(backed.total_nnz().unwrap(), 0);
    assert_eq!(backed.catalog_int_value_max(), Some(0));
}

// ---------------------------------------------------------------------------
// A shard that decodes fewer rows than the catalog claims
// ---------------------------------------------------------------------------

/// Write a single-shard file, then rewrite the shard header in place so it
/// declares fewer rows than the catalog entry does.
///
/// The two counts live in different places and neither authenticates the other:
/// `BackedCsrIndex` reads `(row_start, row_end)` from the catalog stats, which
/// the catalog's BLAKE3 covers; `decode_shard` derives the row count from the
/// decoded indptr, whose length is governed by the shard header — inside the
/// section payload, which the fast read path deliberately does not re-hash
/// (`read_shard_from_entry` says so in its own doc). Truncation, a partial
/// write, or a hostile file separates them.
///
/// `n_major` and `indptr_length` are patched together because the `None` codec
/// requires `indptr_bytes.len() == (n_major + 1) * 8` — so the shard stays
/// internally consistent and decodes cleanly to `new_rows`. That is the point:
/// nothing about the shard is detectably wrong until it is compared to the
/// catalog.
fn write_shard_shrunk_in_header(
    dir: &TempDir,
    n_obs: usize,
    n_vars: usize,
    new_rows: u32,
) -> std::path::PathBuf {
    let path = dir.path().join("shrunk.scx");
    let (indptr, indices, values) = sample_shard_data(n_obs, n_vars);
    let header = sample_header(n_obs as u64, n_vars as u64, *indptr.last().unwrap());
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
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

    // Locate the shard section, then patch its header in place.
    let offset = {
        let reader = ScxReader::open(&path).unwrap();
        let entry = reader
            .catalog()
            .entries
            .iter()
            .find(|e| e.section_type == SectionType::CsrShard)
            .expect("one CSR shard");
        entry.offset as usize
    };

    let mut bytes = std::fs::read(&path).unwrap();
    // ShardHeader field byte offsets, per `ShardHeader::write_to`:
    //   n_major 12, nnz 20, indptr_length 40, indices_length 48,
    //   values_length 56.
    //
    // Every one of these is patched so the shrunk shard stays *internally*
    // coherent — `sample_shard_data` gives 2 nnz per row, indices are u16
    // (n_vars ≤ 65535) and values u8. A shard that were internally inconsistent
    // would be caught by the codec's own length checks and would test those
    // instead of the catalog comparison this exists for.
    let new_nnz = (new_rows as u64) * 2;
    bytes[offset + 12..offset + 16].copy_from_slice(&new_rows.to_le_bytes());
    bytes[offset + 20..offset + 28].copy_from_slice(&new_nnz.to_le_bytes());
    bytes[offset + 40..offset + 44].copy_from_slice(&((new_rows + 1) * 8).to_le_bytes());
    bytes[offset + 48..offset + 52].copy_from_slice(&((new_nnz as u32) * 2).to_le_bytes());
    bytes[offset + 56..offset + 60].copy_from_slice(&(new_nnz as u32).to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    path
}

/// The catalog says this shard covers 8 rows; it decodes to 4.
///
/// `read_rows_with`'s full-shard fallback computes each row's position as
/// `row - shard_row_start` from the **catalog** and then indexes the *decoded*
/// indptr with it — an index-out-of-bounds panic inside the reader, where the
/// convention requires an error.
///
/// The check goes in `decode_shard` rather than at that call site, because the
/// same decoded `ScxCsr` is handed to `read_rows_with`, `read_shard_cached*`
/// and the `ShardSource` impl; all three are asserted here so the guard is
/// pinned to the decode, not to the one caller the panic surfaced through.
#[test]
fn shard_decoding_fewer_rows_than_the_catalog_claims_errors() {
    let dir = TempDir::new().unwrap();
    let path = write_shard_shrunk_in_header(&dir, 8, 10, 4);
    let backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);

    let err = backed
        .read_rows_with(&[0, 7], |_, _, _| Ok(()))
        .expect_err("the full-shard fallback must error, not panic");
    assert!(
        matches!(err, ScxError::InvalidCatalog(_)),
        "expected InvalidCatalog, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains('8') && msg.contains('4'),
        "the error must report both counts so the file can be diagnosed: {msg}"
    );

    // Same guard, reached through the shard cache directly.
    assert!(
        backed.read_shard_cached_arc(0).is_err(),
        "the cache must not hand out a short shard either"
    );

    // `read_shard_uncached` is a second, independent decode entry point — it
    // builds its own `ScxCsr` and never touches `decode_shard`. It is public and
    // streaming callers use it, so a guard placed only on the cached path would
    // leave this one panicking.
    assert!(
        backed.read_shard_uncached(0).is_err(),
        "the uncached decode path needs the same guard"
    );

    // `read_rows` was already guarded — by a *different* check, downstream,
    // which rejects the row slice rather than the shard. Asserting only through
    // it would have looked like proof while testing nothing: pin the error type
    // so a future refactor cannot quietly route this path back to the panic.
    let err = backed.read_rows(0, 8).unwrap_err();
    assert!(
        matches!(err, ScxError::InvalidCatalog(_)),
        "read_rows should now fail at the decode, not at the row slice: {err:?}"
    );
}

/// Control: an untouched file with the same shape still reads. Without this a
/// guard that rejected every shard would pass the test above.
#[test]
fn a_shard_whose_rows_match_the_catalog_still_reads() {
    let dir = TempDir::new().unwrap();
    let (backed, full) = write_test_file_and_open(&dir, 8, 10, 1, 4);
    let got = backed.read_rows(0, 8).unwrap();
    assert_eq!(got.shape.0, 8);
    assert_eq!(full.shape.0, 8);
}

/// The block-index row-run path is the **third** decode seam: it calls
/// `scx_codec::decode_row_group` directly and never passes through
/// `decode_shard_regions_scipy`, so it needs the minor-axis bound check by hand.
///
/// Without it, a *partial* read of a framed shard would be the one remaining way
/// to get an unvalidated column index out of the reader — the two whole-shard
/// seams would look fully guarded while a scattered `read_rows_with` over the
/// same file handed the bad index straight to the caller.
///
/// The shard is written legitimately and then its header's `n_minor` is narrowed
/// in place, so the payload decodes cleanly and only a comparison against
/// `n_minor` can catch it.
#[test]
fn the_block_index_row_run_path_rejects_an_out_of_range_index() {
    let dir = TempDir::new().unwrap();
    let (path, _full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::None);

    // Narrow every CSR shard header's n_minor from 100 to 8; `sample_shard_data`
    // puts columns up to 99 in there, so most rows now carry out-of-range indices.
    let offsets: Vec<usize> = {
        let reader = ScxReader::open(&path).unwrap();
        reader
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CsrShard)
            .map(|e| e.offset as usize)
            .collect()
    };
    assert_eq!(offsets.len(), 2, "fixture should have two shards");
    let mut bytes = std::fs::read(&path).unwrap();
    for off in offsets {
        // n_minor is at byte 16 of the shard header (`ShardHeader::write_to`).
        bytes[off + 16..off + 20].copy_from_slice(&8u32.to_le_bytes());
    }
    std::fs::write(&path, &bytes).unwrap();

    let backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);

    // A scattered read is what selects the block-index path (a dense request
    // would fall back to a full-shard decode, which the other seam guards).
    let err = backed
        .read_rows_with(&[1, 9, 30], |_, _, _| Ok(()))
        .expect_err("a scattered framed read must reject the out-of-range index");
    assert!(
        matches!(err, ScxError::ShardIndexOutOfRange { .. })
            || matches!(err, ScxError::InvalidCatalog(_)),
        "expected an index/catalog error, got {err:?}"
    );
}

/// A shard header that *widens* its own `n_minor` past the catalog's width.
///
/// This is the move the seam's original bound could not stop. Indices were
/// checked against the payload's own `n_minor`, so raising it smuggled an index
/// through that is still out of range for the matrix the catalog describes —
/// and the default (non-dense) read then handed `scipy.sparse.csr_matrix` a
/// structurally invalid matrix, whose `.toarray()` misplaces the value into
/// another row. Guarding only the densify sites left that open, because the
/// corruption happened on scipy's side of the boundary.
///
/// The catalog is the authority: its BLAKE3 covers catalog bytes, while the
/// shard payload is not covered at all. Requiring the two to agree removes the
/// move entirely — the payload cannot widen itself.
#[test]
fn a_shard_header_that_widens_n_minor_past_the_catalog_is_rejected() {
    let dir = TempDir::new().unwrap();
    let n_vars = 10usize;
    let path = dir.path().join("widened.scx");
    let (indptr, indices, values) = sample_shard_data(6, n_vars);
    let header = sample_header(6, n_vars as u64, *indptr.last().unwrap());
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(6)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
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

    // Sanity: it reads before we touch it.
    ScxReader::open(&path)
        .unwrap()
        .read_all_csr_shards()
        .expect("fixture must be readable before the mutation");

    // Widen n_minor (shard-header byte 16) from 10 to 13, leaving the catalog's
    // authenticated col_end at 10.
    let offset = {
        let reader = ScxReader::open(&path).unwrap();
        reader
            .catalog()
            .entries
            .iter()
            .find(|e| e.section_type == SectionType::CsrShard)
            .expect("one CSR shard")
            .offset as usize
    };
    let mut bytes = std::fs::read(&path).unwrap();
    assert_eq!(
        u32::from_le_bytes(bytes[offset + 16..offset + 20].try_into().unwrap()),
        n_vars as u32,
    );
    bytes[offset + 16..offset + 20].copy_from_slice(&13u32.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    let err = ScxReader::open(&path)
        .unwrap()
        .read_all_csr_shards()
        .expect_err("a header/catalog width disagreement must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("13") && msg.contains("10"),
        "must report both widths so the file can be diagnosed: {msg}"
    );

    // The eager path alone is not proof. `BackedCsrReader` passes a *transient*
    // entry with `stats: None`, so the catalog-stats reconcile no-ops there and
    // this was the surface that stayed open after the first attempt: backed
    // AnnData, lazy transforms and every `ShardSource` accelerator read through
    // it. Measured before the fix: `read_shard_cached_arc` returned
    // `Ok(shape=(6, 10))` carrying an index of 12.
    let backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    for (what, r) in [
        (
            "read_shard_cached_arc",
            backed.read_shard_cached_arc(0).map(|_| ()),
        ),
        (
            "read_shard_uncached",
            backed.read_shard_uncached(0).map(|_| ()),
        ),
        (
            "read_rows_with",
            backed.read_rows_with(&[0, 3], |_, _, _| Ok(())),
        ),
    ] {
        assert!(
            r.is_err(),
            "{what}: the backed path must reject a widened header too"
        );
    }
}

/// The framed **block-index** path is a third seam: it decodes row-groups
/// directly and bounds them by `header.n_minor` alone, so a widened payload
/// could still get an out-of-range index out through a *scattered* read while
/// the whole-shard paths rejected the same file.
#[test]
fn the_block_index_path_also_rejects_a_widened_header() {
    let dir = TempDir::new().unwrap();
    let (path, _full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::None);

    let offsets: Vec<usize> = {
        let reader = ScxReader::open(&path).unwrap();
        reader
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CsrShard)
            .map(|e| e.offset as usize)
            .collect()
    };
    let mut bytes = std::fs::read(&path).unwrap();
    for off in &offsets {
        // Widen n_minor (shard-header byte 16) from 100 to 128.
        bytes[off + 16..off + 20].copy_from_slice(&128u32.to_le_bytes());
    }
    std::fs::write(&path, &bytes).unwrap();

    let backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    // A scattered request is what selects the block-index path; a dense one
    // falls back to a full-shard decode, which a different guard covers.
    assert!(
        backed
            .read_rows_with(&[1, 9, 30], |_, _, _| Ok(()))
            .is_err(),
        "a scattered framed read must reject a widened header"
    );
}

// -----------------------------------------------------------------------
// set_cpu_pool / warm_shards pool routing
// -----------------------------------------------------------------------
//
// `warm_shards` used to dispatch unconditionally against rayon's *global*
// registry. `fork()` duplicates that registry as a data structure but not its
// worker threads, so a `par_*` from a forked child parks forever in
// `LockLatch::wait_and_reset` — which is how the ML loader hung under
// `DataLoader(num_workers>0, start_method="fork")`. `set_cpu_pool` lets a
// caller that may be forked hand in a pool built *after* the fork.
//
// Observing *which* pool ran the decode needs a tag that unrelated,
// concurrently-running tests in this binary cannot forge. Each test below
// stamps its own nonce into a thread-local on its pool's workers via
// `spawn_handler`; `warm_one_shard` records the nonce it sees. A decode on the
// global pool (or on any other test's pool) carries nonce 0 and is ignored, so
// there is nothing to serialise between tests.
//
// Every item below is `parallel`-only: `set_cpu_pool` and `warm_one_shard` are
// both `#[cfg(feature = "parallel")]`, and `rayon::ThreadPool` does not exist
// without it. Un-gated, this block is why `cargo test -p scx-format-io
// --no-default-features` had never compiled.

#[cfg(feature = "parallel")]
use std::cell::Cell;
#[cfg(feature = "parallel")]
use std::collections::BTreeSet;
#[cfg(feature = "parallel")]
use std::sync::Mutex;

#[cfg(feature = "parallel")]
thread_local! {
    /// Nonce of the test pool owning this worker thread; 0 on any other thread.
    static POOL_NONCE: Cell<u64> = const { Cell::new(0) };
}

#[cfg(feature = "parallel")]
static WARM_NONCES: Mutex<BTreeSet<u64>> = Mutex::new(BTreeSet::new());

thread_local! {
    /// How many times `shard_indptr` decoded **on this thread**.
    ///
    /// Thread-local rather than a global counter because libtest runs tests in
    /// parallel on their own threads, and the prescan this pins runs on the
    /// calling thread. A shared counter would make the assertion depend on what
    /// else happened to be running.
    static INDPTR_DECODES: Cell<usize> = const { Cell::new(0) };
}

/// Called from `BackedCsrReader::shard_indptr`, unconditionally in the
/// production body — see `note_indptr_decode` at the end of `backed/csr.rs` for
/// why it cannot be a `#[cfg(test)]` at the call site.
pub(super) fn note_indptr_decode() {
    INDPTR_DECODES.with(|c| c.set(c.get() + 1));
}

/// This thread's `shard_indptr` decode count.
fn indptr_decodes() -> usize {
    INDPTR_DECODES.with(|c| c.get())
}

/// Called from `BackedCsrReader::warm_one_shard` on whatever thread rayon chose.
#[cfg(feature = "parallel")]
pub(super) fn note_warm_thread() {
    let nonce = POOL_NONCE.with(|n| n.get());
    if nonce != 0 {
        WARM_NONCES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(nonce);
    }
}

#[cfg(feature = "parallel")]
fn warmed_on(nonce: u64) -> bool {
    WARM_NONCES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .contains(&nonce)
}

/// A pool whose workers stamp `nonce` into `POOL_NONCE`.
#[cfg(feature = "parallel")]
fn tagged_pool(nonce: u64) -> Arc<rayon::ThreadPool> {
    Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .spawn_handler(move |thread| {
                std::thread::spawn(move || {
                    POOL_NONCE.with(|n| n.set(nonce));
                    thread.run();
                });
                Ok(())
            })
            .build()
            .unwrap(),
    )
}

/// Rows spread across all four shards of the fixture, so `warm_shards` gets
/// more than one miss — below that it short-circuits to a sequential loop and
/// never dispatches at all (which is exactly why the old fork test, a single
/// 16-cell shard, passed vacuously).
#[cfg(feature = "parallel")]
const ACROSS_SHARDS: [u64; 4] = [0, 4, 8, 11];

#[cfg(feature = "parallel")]
#[test]
fn warm_shards_runs_on_the_injected_pool() {
    let dir = tempfile::tempdir().unwrap();
    let (mut backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);
    let nonce = 0x5C0001_u64;
    backed.set_cpu_pool(tagged_pool(nonce));

    let out = gather_with(&backed, &ACROSS_SHARDS);

    assert!(
        warmed_on(nonce),
        "warm_shards dispatched somewhere other than the injected pool — under \
         fork that is the global registry, whose workers do not exist"
    );
    // The routing must not change the answer.
    for (i, &row) in ACROSS_SHARDS.iter().enumerate() {
        let expected = full.row_slice(row as usize, row as usize + 1).unwrap();
        assert_eq!(out[i].0, expected.indices, "row {row} indices");
        assert_eq!(out[i].1, expected.data, "row {row} data");
    }
}

#[cfg(feature = "parallel")]
#[test]
fn a_reader_without_a_pool_keeps_using_the_global_one() {
    // Over-fix guard: `scx-accel`/`scx-ops`/`scx-engine`/`scx-cli` never set a
    // pool and must keep composing with their own global-pool parallel regions.
    // Build a tagged pool and deliberately do NOT inject it.
    let dir = tempfile::tempdir().unwrap();
    let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);
    let nonce = 0x5C0002_u64;
    let _unused = tagged_pool(nonce);

    let out = gather_with(&backed, &ACROSS_SHARDS);

    assert!(!warmed_on(nonce), "an un-injected pool must never be used");
    for (i, &row) in ACROSS_SHARDS.iter().enumerate() {
        let expected = full.row_slice(row as usize, row as usize + 1).unwrap();
        assert_eq!(out[i].0, expected.indices, "row {row} indices");
        assert_eq!(out[i].1, expected.data, "row {row} data");
    }
}

// ---------------------------------------------------------------------------
// stored_value_encoding / cache_shards (REC-7, PR D)
// ---------------------------------------------------------------------------

/// Write a `CodecId::None` file whose X shards carry the given value encodings
/// one shard each (4 rows per shard), plus an optional layer with its own
/// per-shard encodings. Values are the u8 sample data re-encoded to the
/// requested width, so every encoding holds them exactly.
fn write_encoded_file(
    dir: &TempDir,
    name: &str,
    x_encodings: &[ValueEncoding],
    layer: Option<(&str, &[ValueEncoding])>,
) -> std::path::PathBuf {
    let n_vars = 8usize;
    let rows_per_shard = 4usize;
    let n_obs = rows_per_shard * x_encodings.len().max(1);
    let path = dir.path().join(name);
    let header = sample_header(n_obs as u64, n_vars as u64, (n_obs * 2) as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
    for (s, &enc) in x_encodings.iter().enumerate() {
        let (indptr, indices, values_u8) = sample_shard_data(rows_per_shard, n_vars);
        let values_f32: Vec<f32> = values_u8.iter().map(|&v| v as f32).collect();
        let bytes = enc.encode_f32_batch(&values_f32).unwrap();
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &bytes,
                CodecId::None,
                enc,
                (s * rows_per_shard) as u64,
            )
            .unwrap();
    }
    if let Some((layer_name, encs)) = layer {
        for (s, &enc) in encs.iter().enumerate() {
            let (indptr, indices, values_u8) = sample_shard_data(rows_per_shard, n_vars);
            let values_f32: Vec<f32> = values_u8.iter().map(|&v| v as f32).collect();
            let bytes = enc.encode_f32_batch(&values_f32).unwrap();
            let shard = ShardBuffers::new(&indptr, &indices, &bytes, CodecId::None, enc);
            writer
                .write_layer_csr_shard(layer_name, s as u32, (s * rows_per_shard) as u64, shard)
                .unwrap();
        }
    }
    writer.finish().unwrap();
    path
}

#[test]
fn stored_value_encoding_uniform_family_reports_itself() {
    let dir = TempDir::new().unwrap();
    let path = write_encoded_file(
        &dir,
        "u16.scx",
        &[ValueEncoding::Uint16, ValueEncoding::Uint16],
        None,
    );
    let backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    assert_eq!(
        backed.stored_value_encoding().unwrap(),
        Some(ValueEncoding::Uint16)
    );
    // Second call is served from the memo and agrees.
    assert_eq!(
        backed.stored_value_encoding().unwrap(),
        Some(ValueEncoding::Uint16)
    );
    // A uniform float16 family is float16, not the write-side "any float ⇒ f32".
    let path = write_encoded_file(&dir, "f16.scx", &[ValueEncoding::Float16], None);
    let backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    assert_eq!(
        backed.stored_value_encoding().unwrap(),
        Some(ValueEncoding::Float16)
    );
}

#[test]
fn stored_value_encoding_mixed_family_reports_the_widest() {
    let dir = TempDir::new().unwrap();
    let path = write_encoded_file(
        &dir,
        "mixed_int.scx",
        &[
            ValueEncoding::Uint8,
            ValueEncoding::Uint16,
            ValueEncoding::Uint8,
        ],
        None,
    );
    let backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    assert_eq!(
        backed.stored_value_encoding().unwrap(),
        Some(ValueEncoding::Uint16)
    );

    let path = write_encoded_file(
        &dir,
        "mixed_float.scx",
        &[ValueEncoding::Uint8, ValueEncoding::Float16],
        None,
    );
    let backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    assert_eq!(
        backed.stored_value_encoding().unwrap(),
        Some(ValueEncoding::Float32)
    );
}

#[test]
fn stored_value_encoding_layer_reader_reports_the_layers_family() {
    let dir = TempDir::new().unwrap();
    let path = write_encoded_file(
        &dir,
        "layer.scx",
        &[ValueEncoding::Uint8, ValueEncoding::Uint8],
        Some(("norm", &[ValueEncoding::Float32, ValueEncoding::Float32])),
    );
    let x = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    assert_eq!(
        x.stored_value_encoding().unwrap(),
        Some(ValueEncoding::Uint8)
    );
    let layer = BackedCsrReader::new_for_layer(ScxReader::open(&path).unwrap(), "norm", 4);
    assert_eq!(
        layer.stored_value_encoding().unwrap(),
        Some(ValueEncoding::Float32),
        "a layer reader must fold the layer's shards, not X's"
    );
}

#[test]
fn stored_value_encoding_for_modality_reader_is_scoped() {
    use crate::modality::ModalityType;
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("mm_enc.scx");
    let n_obs: u64 = 4;
    let header = sample_header(n_obs, 4, 0);
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
            ValueEncoding::Uint32,
            false,
        )
        .unwrap();
    writer.write_var_for(rna_id, &sample_var(4)).unwrap();
    writer.write_var_for(atac_id, &sample_var(4)).unwrap();
    writer.set_modality_n_vars(rna_id, 4).unwrap();
    writer.set_modality_n_vars(atac_id, 4).unwrap();
    let indptr: Vec<u64> = vec![0, 1, 2, 3, 4];
    let indices: Vec<u32> = vec![0, 1, 2, 3];
    let values_u8: Vec<u8> = vec![1, 2, 3, 4];
    let values_u32 = ValueEncoding::Uint32
        .encode_f32_batch(&[1.0, 2.0, 3.0, 4.0])
        .unwrap();
    writer
        .write_csr_shard_for(
            rna_id,
            0,
            ShardBuffers::new(
                &indptr,
                &indices,
                &values_u8,
                CodecId::None,
                ValueEncoding::Uint8,
            ),
        )
        .unwrap();
    writer
        .write_csr_shard_for(
            atac_id,
            0,
            ShardBuffers::new(
                &indptr,
                &indices,
                &values_u32,
                CodecId::None,
                ValueEncoding::Uint32,
            ),
        )
        .unwrap();
    writer.finish().unwrap();

    let rna = BackedCsrReader::for_modality(ScxReader::open(&path).unwrap(), rna_id, 4);
    let atac = BackedCsrReader::for_modality(ScxReader::open(&path).unwrap(), atac_id, 4);
    let all = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    assert_eq!(
        rna.stored_value_encoding().unwrap(),
        Some(ValueEncoding::Uint8)
    );
    assert_eq!(
        atac.stored_value_encoding().unwrap(),
        Some(ValueEncoding::Uint32)
    );
    // The unscoped reader folds every modality's shards.
    assert_eq!(
        all.stored_value_encoding().unwrap(),
        Some(ValueEncoding::Uint32)
    );
}

#[test]
fn cache_shards_reports_the_requested_count_including_zero() {
    let dir = TempDir::new().unwrap();
    let (backed, _) = write_test_file_and_open(&dir, 20, 8, 4, 0);
    assert_eq!(
        backed.cache_shards(),
        0,
        "0 is 'no cache', not clamped to 1"
    );
    let dir2 = TempDir::new().unwrap();
    let (backed, _) = write_test_file_and_open(&dir2, 20, 8, 4, 4);
    assert_eq!(backed.cache_shards(), 4);
}

#[test]
fn stored_value_encoding_refuses_on_a_watched_reader_once_the_file_changed() {
    let dir = TempDir::new().unwrap();
    let path = write_encoded_file(&dir, "watched.scx", &[ValueEncoding::Uint8], None);
    let backed = BackedCsrReader::new(ScxReader::open(&path).unwrap().watching().unwrap(), 4);
    // Warm the memo.
    assert_eq!(
        backed.stored_value_encoding().unwrap(),
        Some(ValueEncoding::Uint8)
    );
    // Replace the file (new inode, as a copy-out op does) with a wider family.
    let newer = write_encoded_file(&dir, "newer.scx", &[ValueEncoding::Uint16], None);
    std::fs::rename(&newer, &path).unwrap();
    let err = backed
        .stored_value_encoding()
        .expect_err("a warm memo must not answer for a file that changed underneath");
    assert!(
        matches!(err, ScxError::FileChangedOnDisk { .. }),
        "expected FileChangedOnDisk, got {err:?}"
    );
    // A fresh reader sees the new family.
    let fresh = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 4);
    assert_eq!(
        fresh.stored_value_encoding().unwrap(),
        Some(ValueEncoding::Uint16)
    );
}

// ---------------------------------------------------------------------------
// OPT-FORMATIO-1 — the row-group LRU
//
// A scattered gather over a framed file takes the block-index path, which
// before this series decoded the touched row groups per call and dropped them:
// the whole-shard LRU never populated (its `!contains` clause is part of the
// eligibility), so every batch re-decoded the same groups. The groups now live
// in the same LRU under `CacheKey::Group`, bounded by the byte budget alone,
// and report through the `row_group_*` counters so `hits` / `misses` keep
// meaning "whole shard". Fixture geometry (`write_framed_file(.., 64, n_vars,
// 2, 4, ..)`): 2 shards × 32 rows, 8 groups of 4 rows per shard, 2 nnz/row —
// a group is `5 × 8 + 8 × 4 + 8 × 4 = 104` bytes, a shard `33 × 8 + 64 × 8 =
// 776`.
// ---------------------------------------------------------------------------

const RG_GROUP_BYTES: usize = 5 * 8 + 8 * 4 + 8 * 4;
const RG_SHARD_BYTES: usize = 33 * 8 + 64 * 4 + 64 * 4;

/// Distinct `(shard, group)` pairs `rows` touch in the fixture above — the
/// oracle for `row_group_misses` on a cold cache.
fn rg_distinct_groups(rows: &[u64]) -> u64 {
    let mut keys: Vec<(usize, usize)> = rows
        .iter()
        .map(|&r| ((r / 32) as usize, ((r % 32) / 4) as usize))
        .collect();
    keys.sort_unstable();
    keys.dedup();
    keys.len() as u64
}

fn rg_gather(backed: &BackedCsrReader, rows: &[u64]) -> Vec<(Vec<i32>, Vec<f32>)> {
    let mut out: Vec<(Vec<i32>, Vec<f32>)> = vec![Default::default(); rows.len()];
    backed
        .read_rows_with(rows, |i, idx, data| {
            out[i] = (idx.to_vec(), data.to_vec());
            Ok(())
        })
        .unwrap();
    out
}

fn rg_assert_matches_full(out: &[(Vec<i32>, Vec<f32>)], rows: &[u64], full: &ScxCsr, ctx: &str) {
    for (i, &row) in rows.iter().enumerate() {
        let lo = full.indptr[row as usize] as usize;
        let hi = full.indptr[row as usize + 1] as usize;
        assert_eq!(out[i].0, full.indices[lo..hi], "{ctx}: indices row {row}");
        assert_eq!(out[i].1, full.data[lo..hi], "{ctx}: data row {row}");
    }
}

/// The headline: the second identical gather decodes nothing. Every touched
/// group is a `row_group_hit`, `row_group_misses` does not move, the output is
/// byte-identical to the first pass and to a full decode, the whole-shard
/// counters stay at zero (the row-group path never touches that half), and
/// `block_index_groups` still counts the cache-served groups as block-index
/// route — the `read_scattered` gate floors depend on that.
#[test]
fn second_gather_is_served_from_the_row_group_lru() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let (path, full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::ShufDeltaZstd);
    let mut backed =
        BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 4, 1 << 20);
    let m = backed.enable_metrics();

    // shard 0: rows 2, 5, 6 → groups {0, 1}; shard 1: 40, 41, 63 → {2, 7}.
    let rows = [2u64, 5, 6, 40, 41, 63, 5];
    let distinct = rg_distinct_groups(&rows);
    assert_eq!(distinct, 4, "fixture premise");

    let first = rg_gather(&backed, &rows);
    rg_assert_matches_full(&first, &rows, &full, "pass 1");
    assert_eq!(
        m.row_group_misses.load(Ordering::Relaxed),
        distinct,
        "cold: one decode per group"
    );
    assert_eq!(m.row_group_hits.load(Ordering::Relaxed), 0);
    assert_eq!(m.block_index_groups.load(Ordering::Relaxed), 2);
    assert_eq!(
        backed.cache_bytes_used(),
        distinct as usize * RG_GROUP_BYTES
    );

    let admitted_after_cold = m.admitted_group_bytes.load(Ordering::Relaxed);
    let parallel_after_cold = m.parallel_group_decodes.load(Ordering::Relaxed);
    assert_eq!(
        admitted_after_cold as usize,
        distinct as usize * RG_GROUP_BYTES,
        "charged once per GROUP lookup — the unit `row_group_hits` counts too — \
         not once per requested row: one `row_group` consultation serves every \
         row of its run, so the seven rows here are four lookups"
    );

    let second = rg_gather(&backed, &rows);
    assert_eq!(
        second, first,
        "the cached groups must reproduce the decoded ones exactly"
    );

    // Two properties of the resident fast path, neither of which the suite could
    // see before review on #540 asked for them.
    //
    // (a) A hit is still a lookup the verdict decided. While the accounting
    //     lived inside `row_group`, which the fast path bypasses, a warm gather
    //     charged NEITHER counter and the pair silently stopped describing the
    //     lookups it documents.
    assert_eq!(
        m.admitted_group_bytes.load(Ordering::Relaxed),
        2 * admitted_after_cold,
        "warm: the same four group lookups are charged again"
    );
    assert_eq!(
        m.rejected_group_bytes.load(Ordering::Relaxed),
        0,
        "nothing was refused: this gather fits its budget"
    );
    // (b) A cache hit must never enter rayon. Routing hits through the pool cost
    //     a measured regression on the 0.99-hit-rate `index_plan` path (p50
    //     27.96 -> 29.17 ms, 1 of 12 rounds, p = 0.006), and without this
    //     assertion a change that reintroduced it would go green.
    assert_eq!(
        m.parallel_group_decodes.load(Ordering::Relaxed),
        parallel_after_cold,
        "warm: every group was resident, so nothing was dispatched to the pool"
    );
    assert_eq!(
        m.row_group_misses.load(Ordering::Relaxed),
        distinct,
        "warm: nothing decoded again"
    );
    assert_eq!(
        m.row_group_hits.load(Ordering::Relaxed),
        distinct,
        "warm: every group served from the LRU"
    );
    assert_eq!(
        m.block_index_groups.load(Ordering::Relaxed),
        4,
        "a cache-served group is still the block-index route"
    );
    assert_eq!(m.full_shard_groups.load(Ordering::Relaxed), 0);
    assert_eq!(
        m.hits.load(Ordering::Relaxed) + m.misses.load(Ordering::Relaxed),
        0,
        "the row-group path never touches the whole-shard counters"
    );
    assert_eq!(m.row_group_evictions.load(Ordering::Relaxed), 0);
    assert_eq!(
        m.row_group_bytes_inserted.load(Ordering::Relaxed),
        (distinct as usize * RG_GROUP_BYTES) as u64
    );
    assert!(
        !backed.cache_contains(0) && !backed.cache_contains(1),
        "no whole shard was inserted"
    );
}

/// Two readers share one cache; the key carries `file_id`, so file 1's row
/// group `(shard 1, group 2)` is not file 0's. The files differ in `n_vars`,
/// which changes the column indices of every row ≥ 30 — a key without
/// `file_id` would hand file 1 file 0's groups and fail the identity check.
#[test]
fn row_group_lru_is_namespaced_by_file_id() {
    use std::sync::atomic::Ordering;
    let d0 = TempDir::new().unwrap();
    let d1 = TempDir::new().unwrap();
    let (p0, full0) = write_framed_file(&d0, 64, 100, 2, 4, CodecId::None);
    let (p1, full1) = write_framed_file(&d1, 64, 60, 2, 4, CodecId::None);
    let rows = [40u64, 41, 63];
    {
        // `sample_shard_data` indexes `(local_row * 2) % n_vars`, so the two
        // files agree until a shard-local row reaches 30; row 63 (shard 1,
        // local 31) is where the group `(1, 7)` genuinely differs.
        let lo = full0.indptr[63] as usize;
        assert_ne!(
            full0.indices[lo..lo + 2],
            full1.indices[lo..lo + 2],
            "fixture premise: the two files differ at the same (shard, group)"
        );
    }

    let shared = SharedShardCache::new(4, 1 << 20);
    let r0 =
        BackedCsrReader::with_shared_cache(ScxReader::open(&p0).unwrap(), 0, Arc::clone(&shared));
    let r1 =
        BackedCsrReader::with_shared_cache(ScxReader::open(&p1).unwrap(), 1, Arc::clone(&shared));
    let m = shared.enable_metrics();

    rg_assert_matches_full(&rg_gather(&r0, &rows), &rows, &full0, "file 0");
    rg_assert_matches_full(&rg_gather(&r1, &rows), &rows, &full1, "file 1");
    assert_eq!(
        m.row_group_misses.load(Ordering::Relaxed),
        2 * rg_distinct_groups(&rows),
        "each file decodes its own groups"
    );
    assert_eq!(
        m.row_group_hits.load(Ordering::Relaxed),
        0,
        "no cross-file hit"
    );
    // And each reader is warm for its own file.
    rg_assert_matches_full(&rg_gather(&r0, &rows), &rows, &full0, "file 0 warm");
    assert_eq!(
        m.row_group_hits.load(Ordering::Relaxed),
        rg_distinct_groups(&rows)
    );
}

/// Row groups are bounded by the byte budget: two gathers that each fit a
/// two-and-a-half-group budget but together exceed it evict the older groups
/// to admit the newer, the output stays exact, and the resident bytes never
/// exceed the budget. (An entry larger than the whole budget would still be
/// admitted — `put_with_budget`'s contract — so the budget here is
/// deliberately more than one group.)
#[test]
fn row_group_lru_evicts_to_fit_the_budget() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let (path, full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::Zstd);
    let budget = 2 * RG_GROUP_BYTES + RG_GROUP_BYTES / 2;
    let mut backed =
        BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 4, budget);
    let m = backed.enable_metrics();

    // Shard 0 groups {0, 1}, then shard 1 groups {2, 7}: each gather fits, the
    // union does not.
    let a = [2u64, 5, 6];
    let b = [40u64, 41, 63];
    rg_assert_matches_full(&rg_gather(&backed, &a), &a, &full, "gather A");
    assert_eq!(m.row_group_misses.load(Ordering::Relaxed), 2);
    assert_eq!(m.row_group_evictions.load(Ordering::Relaxed), 0);
    rg_assert_matches_full(&rg_gather(&backed, &b), &b, &full, "gather B");
    assert_eq!(m.row_group_misses.load(Ordering::Relaxed), 4);
    assert!(
        m.row_group_evictions.load(Ordering::Relaxed) >= 1,
        "B's two groups on top of A's two through a 2.5-group budget must evict (evictions={})",
        m.row_group_evictions.load(Ordering::Relaxed)
    );
    assert!(backed.cache_bytes_used() <= budget);
    assert!(m.peak_bytes_in_cache.load(Ordering::Relaxed) as usize <= budget);

    // Still exact once the cache is churning, and B (the newer) is what stayed.
    rg_assert_matches_full(&rg_gather(&backed, &b), &b, &full, "B again");
    assert_eq!(
        m.row_group_hits.load(Ordering::Relaxed),
        2,
        "B's groups survived A's eviction"
    );
    rg_assert_matches_full(&rg_gather(&backed, &a), &a, &full, "A again");
    assert!(backed.cache_bytes_used() <= budget);
}

/// Admission: a gather whose row groups do not all fit the budget retains
/// **nothing** — every group decodes and is dropped, no eviction, no resident
/// bytes — instead of churning the LRU for zero hits (a scan larger than the
/// cache). The very next gather that does fit is admitted as usual.
#[test]
fn over_budget_gather_bypasses_the_row_group_lru() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let (path, full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::Zstd);
    let budget = 3 * RG_GROUP_BYTES;
    let mut backed =
        BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 4, budget);
    let m = backed.enable_metrics();

    // Four groups through a three-group budget: bypass.
    let big = [2u64, 5, 6, 40, 41, 63];
    rg_assert_matches_full(
        &rg_gather(&backed, &big),
        &big,
        &full,
        "over budget, pass 1",
    );
    rg_assert_matches_full(
        &rg_gather(&backed, &big),
        &big,
        &full,
        "over budget, pass 2",
    );
    assert_eq!(
        m.row_group_misses.load(Ordering::Relaxed),
        8,
        "decoded twice — nothing retained"
    );
    assert_eq!(m.row_group_hits.load(Ordering::Relaxed), 0);
    assert_eq!(
        m.row_group_bytes_inserted.load(Ordering::Relaxed),
        0,
        "not admitted"
    );
    assert_eq!(
        m.row_group_evictions.load(Ordering::Relaxed),
        0,
        "and so nothing to evict"
    );
    assert_eq!(backed.cache_bytes_used(), 0);
    assert_eq!(
        m.block_index_groups.load(Ordering::Relaxed),
        4,
        "the route is unchanged"
    );

    // Two groups through the same budget: admitted, and the repeat hits.
    let small = [40u64, 41, 63];
    rg_assert_matches_full(&rg_gather(&backed, &small), &small, &full, "fits, pass 1");
    assert_eq!(
        m.row_group_bytes_inserted.load(Ordering::Relaxed) as usize,
        2 * RG_GROUP_BYTES
    );
    rg_assert_matches_full(&rg_gather(&backed, &small), &small, &full, "fits, pass 2");
    assert_eq!(m.row_group_hits.load(Ordering::Relaxed), 2);
    assert_eq!(backed.cache_bytes_used(), 2 * RG_GROUP_BYTES);
}

/// `cache_shards` caps **whole shards** only. With a cap of one, resident row
/// groups survive a whole-shard insert, and the second whole shard evicts the
/// first (the LRU *shard*), not the groups.
#[test]
fn shard_count_cap_applies_to_whole_shards_only() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let (path, _full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::None);
    let mut backed =
        BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 1, 1 << 20);
    let m = backed.enable_metrics();

    let rows = [2u64, 5, 6, 40, 41, 63];
    let _ = rg_gather(&backed, &rows);
    assert_eq!(m.row_group_misses.load(Ordering::Relaxed), 4);
    assert_eq!(backed.cache_bytes_used(), 4 * RG_GROUP_BYTES);

    let _ = backed.read_shard_cached_arc(0).unwrap();
    assert!(backed.cache_contains(0));
    assert_eq!(
        m.row_group_evictions.load(Ordering::Relaxed),
        0,
        "groups survive a shard insert"
    );
    assert_eq!(m.evictions.load(Ordering::Relaxed), 0);
    assert_eq!(
        backed.cache_bytes_used(),
        4 * RG_GROUP_BYTES + RG_SHARD_BYTES
    );

    let _ = backed.read_shard_cached_arc(1).unwrap();
    assert!(backed.cache_contains(1));
    assert!(
        !backed.cache_contains(0),
        "cap 1: the older whole shard goes"
    );
    assert_eq!(m.evictions.load(Ordering::Relaxed), 1);
    assert_eq!(
        m.row_group_evictions.load(Ordering::Relaxed),
        0,
        "the count cap must skip over the (older) row groups"
    );
    assert_eq!(
        backed.cache_bytes_used(),
        4 * RG_GROUP_BYTES + RG_SHARD_BYTES
    );
    assert_eq!(backed.cache_capacity(), 1);
}

/// A count-only reader (`BackedCsrReader::new`) does not get an unbounded
/// row-group cache: its `cache_shards` is converted into the bytes that many
/// of its largest shards would take, which bounds groups and never binds
/// before the count cap for whole shards. A cache built externally with
/// `usize::MAX` stays count-only and retains no groups at all.
#[test]
fn count_only_reader_bounds_groups_at_cache_shards_worth_of_bytes() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let (path, full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::None);
    let rows = [2u64, 5, 6, 40, 41, 63];

    let mut backed = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 2);
    assert_eq!(backed.cache_bytes_budget(), 2 * RG_SHARD_BYTES);
    let m = backed.enable_metrics();
    rg_assert_matches_full(&rg_gather(&backed, &rows), &rows, &full, "derived budget");
    assert_eq!(
        m.row_group_misses.load(Ordering::Relaxed),
        4,
        "groups are retained"
    );
    // Whole-shard behaviour is unchanged: two shards fit the count cap and
    // the derived bytes alike.
    let _ = backed.read_shard_cached_arc(0).unwrap();
    let _ = backed.read_shard_cached_arc(1).unwrap();
    assert!(backed.cache_contains(0) && backed.cache_contains(1));
    assert_eq!(m.evictions.load(Ordering::Relaxed), 0);

    let shared = SharedShardCache::new(4, usize::MAX);
    let mut counted =
        BackedCsrReader::with_shared_cache(ScxReader::open(&path).unwrap(), 0, shared);
    assert_eq!(counted.cache_bytes_budget(), usize::MAX);
    let m2 = counted.enable_metrics();
    rg_assert_matches_full(&rg_gather(&counted, &rows), &rows, &full, "count-only");
    rg_assert_matches_full(
        &rg_gather(&counted, &rows),
        &rows,
        &full,
        "count-only again",
    );
    assert_eq!(
        m2.row_group_misses.load(Ordering::Relaxed) + m2.row_group_hits.load(Ordering::Relaxed),
        0,
        "a count-only cache retains no row groups"
    );
    assert_eq!(
        m2.block_index_groups.load(Ordering::Relaxed),
        4,
        "the route is unchanged"
    );
    assert_eq!(counted.cache_bytes_used(), 0);
}

/// Both off-switches — the per-reader gate and `cache_shards = 0` — leave the
/// block-index route in place but retain nothing and move no `row_group_*`
/// counter; the output is unchanged. This is the arm the same-build A/B
/// capture runs (`SCX_ROW_GROUP_CACHE=0`).
#[test]
fn row_group_cache_off_decodes_uncached() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let (path, full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::ShufDeltaZstd);
    let rows = [2u64, 5, 6, 40, 41, 63];

    let mut gated =
        BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 4, 1 << 20);
    gated.set_row_group_cache(false);
    let m = gated.enable_metrics();
    rg_assert_matches_full(&rg_gather(&gated, &rows), &rows, &full, "gate off, pass 1");
    rg_assert_matches_full(&rg_gather(&gated, &rows), &rows, &full, "gate off, pass 2");
    assert_eq!(m.row_group_misses.load(Ordering::Relaxed), 0);
    assert_eq!(m.row_group_hits.load(Ordering::Relaxed), 0);
    assert_eq!(m.row_group_bytes_inserted.load(Ordering::Relaxed), 0);
    assert_eq!(
        m.block_index_groups.load(Ordering::Relaxed),
        4,
        "route unchanged"
    );
    assert_eq!(gated.cache_bytes_used(), 0);

    let mut uncached =
        BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 0, 1 << 20);
    let m = uncached.enable_metrics();
    rg_assert_matches_full(&rg_gather(&uncached, &rows), &rows, &full, "no cache");
    rg_assert_matches_full(
        &rg_gather(&uncached, &rows),
        &rows,
        &full,
        "no cache, again",
    );
    assert_eq!(
        m.row_group_misses.load(Ordering::Relaxed) + m.row_group_hits.load(Ordering::Relaxed),
        0
    );
    assert_eq!(m.block_index_groups.load(Ordering::Relaxed), 4);
}

/// The framing layout (header scalars, sub-stream ranges, resolved block index)
/// is resolved once per shard and shared by every later read — the same `Arc`
/// comes back, not a re-parse. An unframed file resolves to `None` per shard.
#[test]
fn framed_layout_is_resolved_once_per_shard() {
    let dir = TempDir::new().unwrap();
    let (path, _full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::None);
    let backed = BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 4, 1 << 20);

    let first = backed.framed_layout(0).expect("framed shard has a layout");
    assert_eq!(first.spans.len(), 8, "32 rows / G=4");
    assert_eq!(first.n_major, 32);
    let again = backed.framed_layout(0).unwrap();
    assert!(Arc::ptr_eq(&first, &again), "memoized, not re-resolved");

    let rows = [2u64, 5, 6, 40, 41, 63];
    let _ = rg_gather(&backed, &rows);
    let _ = rg_gather(&backed, &rows);
    let after = backed.framed_layout(0).unwrap();
    assert!(Arc::ptr_eq(&first, &after), "the gathers re-used the memo");
    assert!(backed.framed_layout(1).is_some());
    assert!(
        backed.framed_layout(2).is_none(),
        "out of range: no shard, no layout"
    );

    let (unframed, _) = write_test_file_and_open(&dir, 64, 100, 4, 4);
    for s in 0..4 {
        assert!(unframed.framed_layout(s).is_none(), "unframed shard {s}");
    }
    assert_eq!(unframed.planned_row_group_bytes(0, &[1, 2]), 0);
    assert_eq!(unframed.warm_row_groups(0, &[1, 2]).unwrap(), 0);
}

/// `read_rows(start, end)`'s narrow-window path goes through the same row-group
/// LRU: the second identical window is all hits and byte-identical.
#[test]
fn read_rows_window_path_hits_the_row_group_lru() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let (path, full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::ShufDeltaZstd);
    let mut backed =
        BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 4, 1 << 20);
    let m = backed.enable_metrics();

    // Rows 2..5 of shard 0: groups 0 (rows 0-3) and 1 (row 4); 3 × 4 < 32 so
    // the window takes the row-range (block-index) plan.
    let first = backed.read_rows(2, 5).unwrap();
    let want = full.row_slice(2, 5).unwrap();
    assert_eq!(first.indptr, want.indptr);
    assert_eq!(first.indices, want.indices);
    assert_eq!(first.data, want.data);
    assert_eq!(m.row_group_misses.load(Ordering::Relaxed), 2);
    assert!(
        !backed.cache_contains(0),
        "a narrow window never decodes the whole shard"
    );

    let second = backed.read_rows(2, 5).unwrap();
    assert_eq!(second.indices, first.indices);
    assert_eq!(second.data, first.data);
    assert_eq!(m.row_group_misses.load(Ordering::Relaxed), 2);
    assert_eq!(m.row_group_hits.load(Ordering::Relaxed), 2);
}

/// The L2 prefetcher sizes a warm from the block index before decoding:
/// `planned_row_group_bytes` must equal what the LRU then charges, and
/// `warm_row_groups` must leave the gather with nothing to decode.
#[test]
fn planned_row_group_bytes_matches_decoded_size_and_warm_feeds_the_gather() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let (path, full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::Zstd);
    let mut backed =
        BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 4, 1 << 20);
    let m = backed.enable_metrics();

    // Shard 1, rows 40, 41, 63 → groups 2 and 7. Unsorted on purpose.
    let rows = [63u64, 40, 41];
    let planned = backed.planned_row_group_bytes(1, &rows);
    assert_eq!(planned, 2 * RG_GROUP_BYTES);
    assert_eq!(
        m.row_group_misses.load(Ordering::Relaxed),
        0,
        "planning decodes nothing"
    );

    assert_eq!(backed.warm_row_groups(1, &rows).unwrap(), 2);
    assert_eq!(m.row_group_misses.load(Ordering::Relaxed), 2);
    assert_eq!(
        m.row_group_bytes_inserted.load(Ordering::Relaxed) as usize,
        planned
    );
    assert_eq!(backed.cache_bytes_used(), planned);

    rg_assert_matches_full(&rg_gather(&backed, &rows), &rows, &full, "after warm");
    assert_eq!(
        m.row_group_misses.load(Ordering::Relaxed),
        2,
        "the gather decoded nothing"
    );
    assert_eq!(m.row_group_hits.load(Ordering::Relaxed), 2);
    assert_eq!(m.block_index_groups.load(Ordering::Relaxed), 1);

    // A row outside the shard is an error, as on the gather.
    assert!(backed.warm_row_groups(1, &[5]).is_err());
    assert!(backed.warm_row_groups(0, &[64]).is_err());
}

/// `touched_row_groups` names the groups `planned_row_group_bytes` sizes, and
/// both are one walk of the block index.
///
/// **The expectation is derived from the fixture's geometry, not from the
/// subject.** Asking `planned_row_group_bytes` what to expect would make this
/// test unable to see the two of them agreeing on a wrong answer — which is
/// precisely the risk created by expressing one in terms of the other.
///
/// Fixture: 64 rows over 2 shards of 32, row groups of 4. Shard 1 covers rows
/// 32..64, so rows 40, 41, 63 are shard-local 8, 9, 31 and fall in groups
/// 8/4 = 2, 9/4 = 2 and 31/4 = 7.
#[test]
fn touched_row_groups_names_the_groups_planned_bytes_sizes() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let (path, _full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::Zstd);
    let mut backed =
        BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 4, 1 << 20);
    let m = backed.enable_metrics();

    // Unsorted and duplicated on purpose: the contract is ascending and
    // deduplicated output whatever the caller passes.
    let rows = [63u64, 40, 41, 40, 63];
    let keys = backed.touched_row_groups(1, &rows);
    assert_eq!(
        keys.iter().map(|&(g, _)| g).collect::<Vec<_>>(),
        vec![2usize, 7],
        "ascending, deduplicated group indices"
    );
    for &(g, bytes) in &keys {
        assert_eq!(bytes, RG_GROUP_BYTES, "group {g} charges one group's bytes");
    }

    // The fold is the sizing, and the sizing is 2 groups — NOT 5, which is what
    // a lost `dedup()` would report for these five rows.
    assert_eq!(
        backed.planned_row_group_bytes(1, &rows),
        2 * RG_GROUP_BYTES,
        "planned bytes == the fold over the named keys"
    );

    // Neither call decodes anything or touches the LRU.
    assert_eq!(m.row_group_misses.load(Ordering::Relaxed), 0);
    assert_eq!(m.row_group_hits.load(Ordering::Relaxed), 0);
    assert_eq!(backed.cache_bytes_used(), 0);

    // Degenerate inputs name nothing rather than panicking.
    assert!(backed.touched_row_groups(1, &[]).is_empty());
    assert!(backed.touched_row_groups(99, &rows).is_empty());
    assert_eq!(backed.planned_row_group_bytes(99, &rows), 0);

    // And the route gate applies to the key list exactly as it does to the
    // sizing — a caller must not be able to name keys no gather would retain.
    let mut off =
        BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 4, 1 << 20);
    off.set_scatter_block_index(false);
    assert!(off.touched_row_groups(1, &rows).is_empty());
    assert_eq!(off.planned_row_group_bytes(1, &rows), 0);
}

/// The chunked, gather-wide parallel group decode is byte-identical to the
/// serial walk at every pool width, including the width-1 case that takes the
/// serial path outright.
///
/// The gather decodes its row groups a chunk at a time, one chunk per pool
/// width, so the pool width changes how the work is batched and must change
/// nothing else. `parallel_group_decodes` is asserted alongside so the test can
/// tell "the widths agree" from "no width ever ran in parallel" — three widths
/// agreeing about a path none of them took would read as coverage.
///
/// ⚠️ `#[cfg(feature = "parallel")]`: `set_cpu_pool` and the `rayon` dependency
/// only exist under that feature, and `--no-default-features` compiles this
/// file. Un-gated it broke the `Feature matrix (clippy, no-hdf5 legs)` CI leg —
/// the same regression this file already records ~400 lines above, reintroduced.
#[test]
#[cfg(feature = "parallel")]
fn parallel_group_decode_matches_the_serial_walk_at_every_pool_width() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let (path, full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::Zstd);
    // Seven rows per shard, one per row group. Seven is the widest request the
    // block-index route accepts here — `block_index_eligible` needs
    // `len * ROW_RANGE_WINDOW_DIVISOR < shard_rows`, i.e. `len * 4 < 32` — and
    // the first version of this test asked for all 64 rows, which fails that
    // window and took the whole-shard route on both shards, so there was no
    // group decode to parallelise at all. The premise assertion below is what
    // caught it.
    let rows: Vec<u64> = (0..7u64)
        .map(|i| i * 4)
        .chain((0..7).map(|i| 32 + i * 4))
        .collect();

    let mut reference: Option<Vec<(Vec<i32>, Vec<f32>)>> = None;
    for width in [1usize, 2, 3, 8] {
        let mut backed =
            BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 4, 1 << 20);
        backed.set_cpu_pool(std::sync::Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(width)
                .build()
                .unwrap(),
        ));
        let m = backed.enable_metrics();

        let out = rg_gather(&backed, &rows);
        rg_assert_matches_full(&out, &rows, &full, &format!("width {width}"));
        match &reference {
            None => reference = Some(out),
            Some(want) => assert_eq!(&out, want, "width {width} disagrees with width 1"),
        }

        let parallel = m.parallel_group_decodes.load(Ordering::Relaxed);
        if width == 1 {
            assert_eq!(
                parallel, 0,
                "a width-1 pool chunks one group at a time and takes the serial path"
            );
        } else {
            assert!(
                parallel > 0,
                "width {width} must actually have decoded a chunk in parallel; \
                 without this the agreement above is between four serial runs"
            );
        }
    }
}

/// A gather's own whole shards count against the budget beside its row groups,
/// because they share one and `read_shard_cached_arc` inserts them whether or
/// not row groups are admitted.
///
/// This is the half of the phase-5 accounting item that is real. The per-gather
/// rule used to sum only the `use_block_index` groups, while the plan-level rule
/// (`PrefetchEngine::plan_footprint`) has summed both since review on #528 — so
/// a mixed gather could admit row groups it was about to evict with its own
/// whole-shard decode.
///
/// Fixture: 64 rows, 2 shards of 32, row groups of 4. Eight rows of shard 0 is
/// a quarter of the shard, which fails `block_index_eligible`'s
/// `len * 4 < shard_rows` window and takes the whole-shard path (776 B); two
/// rows of shard 1 land in one row group (104 B). Budget 400 B sits between the
/// two sums, so the two rules disagree on exactly this gather.
///
/// Mutation: restore `groups.iter().filter(|g| g.use_block_index)` and this
/// reddens.
#[test]
fn the_per_gather_rule_counts_its_own_whole_shards() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let (path, full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::None);
    let budget = 400usize;
    assert!(
        RG_GROUP_BYTES < budget && budget < RG_SHARD_BYTES + RG_GROUP_BYTES,
        "premise: the budget separates the group-only sum from the mixed sum \
         ({RG_GROUP_BYTES} < {budget} < {} )",
        RG_SHARD_BYTES + RG_GROUP_BYTES
    );
    let mut backed =
        BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 4, budget);
    let m = backed.enable_metrics();

    // Eight rows of shard 0 (whole-shard route) + two rows of shard 1 that share
    // one row group (block-index route).
    let mixed = [0u64, 1, 2, 3, 4, 5, 6, 7, 40, 41];
    rg_assert_matches_full(&rg_gather(&backed, &mixed), &mixed, &full, "mixed");

    // Premise: the gather really did split across the two routes. Without this
    // the assertion below would pass on a gather that never took either.
    assert_eq!(
        m.full_shard_groups.load(Ordering::Relaxed),
        1,
        "premise: shard 0 took the whole-shard route"
    );
    assert_eq!(
        m.block_index_groups.load(Ordering::Relaxed),
        1,
        "premise: shard 1 took the block-index route"
    );

    assert_eq!(
        m.row_group_bytes_inserted.load(Ordering::Relaxed),
        0,
        "the row group is NOT retained: 776 B of whole shard + 104 B of group is \
         over the 400 B budget, so retaining it would only have been evicted by \
         the shard decode on the next repeat"
    );
    assert_eq!(
        m.row_group_misses.load(Ordering::Relaxed),
        1,
        "it still decoded — admission governs retention, never correctness"
    );
}

/// **The free-bytes falsification.** The phase-5 plan text prescribed comparing
/// a gather's planned row-group bytes against `bytes_budget() - bytes_used()`.
/// This test states what that would mean, so the prescription is measured
/// rather than adopted: a cache sitting *at* its budget — the steady state of
/// any correctly sized cache — would have zero free bytes, and every subsequent
/// gather would be refused admission forever, however small.
///
/// The LRU already handles the arithmetic the prescription was reaching for:
/// `evict_bytes_for` makes room by evicting, so a resident entry is displaceable
/// and is not a claim on the budget. What a gather must fit is its OWN
/// footprint, which is what the rule compares.
///
/// Mutation: change `gather_row_groups_fit_budget`'s comparison to
/// `bytes_budget().saturating_sub(bytes_used())` and this reddens.
#[test]
fn a_full_cache_does_not_refuse_a_small_gather() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let (path, full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::None);
    // Exactly three groups fit.
    let budget = 3 * RG_GROUP_BYTES;
    let mut backed =
        BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 4, budget);
    let m = backed.enable_metrics();

    // Fill the cache to its budget with three groups of shard 0.
    let fill = [0u64, 4, 8];
    rg_assert_matches_full(&rg_gather(&backed, &fill), &fill, &full, "fill");
    assert_eq!(
        backed.cache_bytes_used(),
        budget,
        "premise: the cache is exactly full"
    );
    let inserted_after_fill = m.row_group_bytes_inserted.load(Ordering::Relaxed);
    assert_eq!(inserted_after_fill as usize, 3 * RG_GROUP_BYTES);

    // A one-group gather elsewhere in the file. Its own footprint is a twelfth
    // of the budget; there are zero free bytes.
    let small = [40u64];
    rg_assert_matches_full(&rg_gather(&backed, &small), &small, &full, "small");
    assert!(
        m.row_group_bytes_inserted.load(Ordering::Relaxed) > inserted_after_fill,
        "a gather that fits the budget is admitted even against a full cache; \
         got {} inserted bytes, unchanged from the fill",
        m.row_group_bytes_inserted.load(Ordering::Relaxed)
    );
    assert!(
        m.row_group_evictions.load(Ordering::Relaxed) > 0,
        "and it made room by evicting, which is the LRU doing its job"
    );
}

/// A non-admitted gather still serves resident groups as hits — admission only
/// stops *misses* from being inserted — and its misses decode uncached with no
/// singleflight slot (the leader of a slot that inserts nothing would make
/// every waiter re-decode in turn; review on #528).
#[test]
fn bypassed_gather_still_hits_resident_groups() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let (path, full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::None);
    let budget = 3 * RG_GROUP_BYTES;
    let mut backed =
        BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 4, budget);
    let m = backed.enable_metrics();

    // Two groups admitted (they fit).
    let small = [40u64, 41, 63];
    rg_assert_matches_full(&rg_gather(&backed, &small), &small, &full, "fits");
    assert_eq!(
        m.row_group_bytes_inserted.load(Ordering::Relaxed) as usize,
        2 * RG_GROUP_BYTES
    );

    // Four groups, over budget: the two resident ones are hits, the other two
    // decode and drop; nothing is inserted or evicted.
    let big = [2u64, 5, 6, 40, 41, 63];
    rg_assert_matches_full(&rg_gather(&backed, &big), &big, &full, "over budget");
    assert_eq!(
        m.row_group_hits.load(Ordering::Relaxed),
        2,
        "resident groups served"
    );
    assert_eq!(
        m.row_group_misses.load(Ordering::Relaxed),
        4,
        "2 admitted + 2 uncached"
    );
    assert_eq!(
        m.row_group_bytes_inserted.load(Ordering::Relaxed) as usize,
        2 * RG_GROUP_BYTES
    );
    assert_eq!(m.row_group_evictions.load(Ordering::Relaxed), 0);
    assert_eq!(backed.cache_bytes_used(), 2 * RG_GROUP_BYTES);
}

/// `read_rows(start, end)`'s window path applies the same admission: a window
/// whose groups do not fit the budget decodes and drops rather than evicting
/// everything to retain them (review on #528: row groups are fixed-height, not
/// fixed-nnz, so "a quarter of the rows" is not "a quarter of the bytes").
#[test]
fn read_rows_window_over_budget_is_not_retained() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let (path, full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::None);
    // Budget holds one group; rows 2..5 touch two.
    let budget = RG_GROUP_BYTES + RG_GROUP_BYTES / 2;
    let mut backed =
        BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 4, budget);
    let m = backed.enable_metrics();

    let got = backed.read_rows(2, 5).unwrap();
    let want = full.row_slice(2, 5).unwrap();
    assert_eq!(got.indices, want.indices);
    assert_eq!(got.data, want.data);
    assert_eq!(m.row_group_misses.load(Ordering::Relaxed), 2);
    assert_eq!(
        m.row_group_bytes_inserted.load(Ordering::Relaxed),
        0,
        "not retained"
    );
    assert_eq!(backed.cache_bytes_used(), 0);

    // A one-group window fits and is retained.
    let _ = backed.read_rows(0, 3).unwrap();
    assert_eq!(
        m.row_group_bytes_inserted.load(Ordering::Relaxed) as usize,
        RG_GROUP_BYTES
    );
}

/// `read_rows` takes one admission verdict over every row-range window of the
/// read (review on #528 round 2): two edge windows that each fit the budget
/// but not together would otherwise evict each other on every repeat of the
/// same read.
#[test]
fn read_rows_edge_windows_are_admitted_together() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let (path, full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::None);
    // Budget holds one group and a half; rows 30..34 are one group in each
    // shard (shard 0 rows 30-31 → group 7, shard 1 rows 0-1 → group 0).
    let budget = RG_GROUP_BYTES + RG_GROUP_BYTES / 2;
    let mut backed =
        BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 4, budget);
    let m = backed.enable_metrics();

    let got = backed.read_rows(30, 34).unwrap();
    let want = full.row_slice(30, 34).unwrap();
    assert_eq!(got.indices, want.indices);
    assert_eq!(got.data, want.data);
    assert_eq!(
        m.row_group_misses.load(Ordering::Relaxed),
        2,
        "two windows, two groups"
    );
    assert_eq!(
        m.row_group_bytes_inserted.load(Ordering::Relaxed),
        0,
        "each window fits alone, the pair does not: neither is retained"
    );
    assert_eq!(m.row_group_evictions.load(Ordering::Relaxed), 0);
    assert_eq!(backed.cache_bytes_used(), 0);

    // A single-window read of one of them fits and is retained.
    let _ = backed.read_rows(30, 32).unwrap();
    assert_eq!(
        m.row_group_bytes_inserted.load(Ordering::Relaxed) as usize,
        RG_GROUP_BYTES
    );
}

/// Sizing a plan's row groups must not resolve framing layouts when the route
/// is statically off (review on #528 round 3): the cell-set loader defaults
/// `scatter_block_index` off, and its default path gained no new per-shard
/// header/block-index work from the admission sum.
#[test]
fn planned_row_group_bytes_is_zero_with_the_route_off() {
    let dir = TempDir::new().unwrap();
    let (path, _full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::None);
    let mut backed =
        BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 4, 1 << 20);
    assert!(
        backed.planned_row_group_bytes(1, &[40, 41, 63]) > 0,
        "premise: route on"
    );

    let mut off =
        BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 4, 1 << 20);
    off.set_scatter_block_index(false);
    assert_eq!(off.planned_row_group_bytes(1, &[40, 41, 63]), 0);
    assert_eq!(
        off.warm_row_groups(1, &[40, 41, 63]).unwrap(),
        0,
        "nothing to warm into either"
    );
    // Whole-shard sizing is independent of the route.
    assert_eq!(backed.shard_decoded_bytes(0), RG_SHARD_BYTES);
    let _ = &mut backed;
}

/// The documented scatter-order contract: on a **mixed** request every
/// full-shard fallback fires before every block-index row group, so a *later*
/// row can be handed to the callback before an *earlier* one.
///
/// All three reviewers on #540 noted the split was documented and not asserted.
/// It matters because the pre-split docs promised shard-grouped, sorted-by-row
/// order, and the only guarantee now is the `orig_pos` argument — so this test
/// pins both halves: the order really is per-pass, and `orig_pos` really does
/// still address the request.
///
/// Fixture: 256 rows, 4 shards of 64, groups of 16. Row 2 alone in shard 0 is
/// a sparse request and takes the block-index route; sixteen consecutive rows of
/// shard 1 are a quarter of it, fail `block_index_eligible`'s window and take
/// the whole-shard route. Row 2 is the LOWEST row requested, so any sorted-by-row
/// contract would emit it first.
#[test]
fn a_mixed_request_scatters_full_shard_groups_before_block_index_groups() {
    let dir = TempDir::new().unwrap();
    let (path, full) = write_framed_file(&dir, 256, 100, 4, 16, CodecId::None);
    let backed = BackedCsrReader::new_with_byte_budget(ScxReader::open(&path).unwrap(), 4, 1 << 20);

    let mut rows: Vec<u64> = vec![2];
    rows.extend(64..80u64);
    assert_eq!(
        rows[0], 2,
        "premise: the block-index row is the lowest row requested"
    );
    assert!(
        !backed.block_index_eligible(1, 16),
        "premise: sixteen of shard 1's sixty-four rows take the whole-shard route"
    );
    assert!(
        backed.block_index_eligible(0, 1),
        "premise: one row of shard 0 takes the block-index route"
    );

    let mut order: Vec<u64> = Vec::new();
    backed
        .read_rows_with(&rows, |orig_pos, idx, data| {
            // `orig_pos` addresses the request — the one guarantee — so the row
            // it names is read back through the request array, exactly as every
            // in-tree consumer does.
            let row = rows[orig_pos];
            let (lo, hi) = (
                full.indptr[row as usize] as usize,
                full.indptr[row as usize + 1] as usize,
            );
            assert_eq!(idx, &full.indices[lo..hi], "row {row} indices");
            assert_eq!(data, &full.data[lo..hi], "row {row} data");
            order.push(row);
            Ok(())
        })
        .unwrap();

    assert_eq!(order.len(), rows.len(), "every requested row was scattered");
    assert_eq!(
        *order.last().unwrap(),
        2,
        "the block-index row fires LAST despite being the lowest row requested: \
         pass 1 serves the whole-shard fallback, pass 2 the row groups. If this \
         ever reads 2 first, the order has become sorted-by-row again and the \
         docs on `read_rows_with_admission` are the thing to change."
    );
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_ne!(
        order, sorted,
        "premise: the fixture actually exercises out-of-row-order scatter"
    );
}

/// `read_row_indices` can carry a caller's plan-wide admission verdict.
///
/// W11's cell-set executor reads a whole plan's **deduplicated** row list
/// through this method, for the exact indptr prescan and the disjoint output
/// spans it already has. Without this variant the caller would have to choose
/// between that prescan and the plan-wide verdict its prefetch engine already
/// took — `read_row_indices` passed a literal `None` and decided for itself.
#[test]
fn read_row_indices_carries_the_callers_admission_verdict() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    // 2 shards x 32 rows, 8 row groups of 4 per shard. Two rows in shard 0 is
    // 2 * 4 < 32, so the request takes the block-index route.
    let (path, full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::None);
    let rows = [0u64, 5];
    let budget = RG_GROUP_BYTES * 8;

    let open = |p: &std::path::Path| {
        BackedCsrReader::new_with_byte_budget(ScxReader::open(p).unwrap(), 4, budget)
    };

    // Self-deciding (`None`): the two groups fit the budget, so both are kept.
    let mut a = open(&path);
    let ma = a.enable_metrics();
    let got = a.read_row_indices(&rows).unwrap();
    assert_eq!(
        ma.row_group_bytes_inserted.load(Ordering::Relaxed) as usize,
        RG_GROUP_BYTES * 2,
        "premise: left to itself this read retains both groups"
    );

    // The bytes are right, and are the reference's.
    for (out, &r) in rows.iter().enumerate() {
        let want = full.row_slice(r as usize, r as usize + 1).unwrap();
        let lo = got.indptr[out] as usize;
        let hi = got.indptr[out + 1] as usize;
        assert_eq!(&got.indices[lo..hi], &want.indices[..], "row {r}");
        assert_eq!(&got.data[lo..hi], &want.data[..], "row {r}");
    }

    // `Admit::None` over the same read: identical output, nothing retained.
    let mut b = open(&path);
    let mb = b.enable_metrics();
    let refused = b
        .read_row_indices_with_admission(&rows, Some(&Admit::None))
        .unwrap();
    assert_eq!(refused.indptr, got.indptr);
    assert_eq!(refused.indices, got.indices);
    assert_eq!(refused.data, got.data);
    assert_eq!(
        mb.row_group_bytes_inserted.load(Ordering::Relaxed),
        0,
        "the caller's verdict refused every group"
    );
    assert_eq!(b.cache_bytes_used(), 0);

    // `Admit::Groups` keeps exactly the named key. Row 0 is group 0 of shard 0,
    // row 5 is group 1; naming only the first retains one group's bytes.
    let mut c = open(&path);
    let mc = c.enable_metrics();
    let keys: std::collections::HashSet<RowGroupKey> =
        [(0u32, 0usize, 0usize)].into_iter().collect();
    let partial = c
        .read_row_indices_with_admission(&rows, Some(&Admit::groups(keys)))
        .unwrap();
    assert_eq!(partial.indices, got.indices);
    assert_eq!(partial.data, got.data);
    assert_eq!(
        mc.row_group_bytes_inserted.load(Ordering::Relaxed) as usize,
        RG_GROUP_BYTES,
        "one named key, one group's bytes"
    );
}

/// The sibling guard, and it was missing: `read_rows_with_admission` must
/// honour the caller's verdict too.
///
/// ⚠️ Found while mutation-testing the `read_row_indices` split above. Passing
/// a literal `None` in place of `admit_row_groups` here — i.e. ignoring every
/// caller's verdict outright — left **all 475** of this crate's tests green;
/// only `scx-loader` reddened (10 `plan_engine` / `sparse_cellset` tests). The
/// crate that owns the API could not see its own contract break, so this is
/// the accept-side test for it rather than a restatement of the loader's.
#[test]
fn read_rows_with_admission_honours_the_callers_verdict() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    let (path, _full) = write_framed_file(&dir, 64, 100, 2, 4, CodecId::None);
    let rows = [0u64, 5];
    let budget = RG_GROUP_BYTES * 8;
    let open = |p: &std::path::Path| {
        BackedCsrReader::new_with_byte_budget(ScxReader::open(p).unwrap(), 4, budget)
    };

    let collect = |backed: &BackedCsrReader, admit: Option<&Admit>| {
        let mut out: Vec<(usize, Vec<i32>, Vec<f32>)> = Vec::new();
        backed
            .read_rows_with_admission(&rows, admit, |pos, idx, val| {
                out.push((pos, idx.to_vec(), val.to_vec()));
                Ok(())
            })
            .unwrap();
        out.sort_by_key(|&(p, _, _)| p);
        out
    };

    let mut a = open(&path);
    let ma = a.enable_metrics();
    let want = collect(&a, None);
    assert_eq!(
        ma.row_group_bytes_inserted.load(Ordering::Relaxed) as usize,
        RG_GROUP_BYTES * 2,
        "premise: left to itself this read retains both groups"
    );

    let mut b = open(&path);
    let mb = b.enable_metrics();
    assert_eq!(collect(&b, Some(&Admit::None)), want, "same bytes");
    assert_eq!(
        mb.row_group_bytes_inserted.load(Ordering::Relaxed),
        0,
        "the caller's verdict refused every group"
    );
    assert_eq!(b.cache_bytes_used(), 0);
}

/// The indptr prescan reads a **resident** shard's indptr rather than decoding
/// it again, and does so without counting a read that did not happen.
///
/// `shard_indptr` decodes the stream afresh every call by design — it must not
/// touch the LRU, or a prescan between planning and warming could change what
/// `block_index_eligible` admits. That is right when the shard is cold and pure
/// waste when it is not, and `read_row_indices` paid it on every call: measured
/// at 2.73 -> 0.55 ms per gather on a 100k-row synthetic with every shard warm,
/// i.e. seven re-decodes of a 16,384-entry indptr were 80 % of the call.
///
/// The observable is the **hit count**: the peek must add none. A `get_cached`
/// in its place would count one per shard group and inflate the
/// `shard_cache_hit_rate` the loader reports with lookups no read performed —
/// which is the mutation this is watched failing against.
#[test]
fn the_indptr_prescan_reads_a_resident_shard_without_counting_a_hit() {
    use std::sync::atomic::Ordering;
    let dir = TempDir::new().unwrap();
    // Unframed and multi-shard, so every request group takes the full-shard
    // path and the prescan is the only other thing touching the cache.
    let (mut backed, full) = write_test_file_and_open(&dir, 64, 40, 4, 8);
    let m = backed.enable_metrics();
    let rows = [1u64, 17, 33, 49];
    // libtest may reuse a thread, so this is a delta and not an absolute.
    let decodes0 = indptr_decodes();

    // Cold: the prescan takes the decode branch, and the answer is right.
    let cold = backed.read_row_indices(&rows).unwrap();
    for (out, &r) in rows.iter().enumerate() {
        let want = full.row_slice(r as usize, r as usize + 1).unwrap();
        let (lo, hi) = (cold.indptr[out] as usize, cold.indptr[out + 1] as usize);
        assert_eq!(&cold.indices[lo..hi], &want.indices[..], "cold row {r}");
        assert_eq!(&cold.data[lo..hi], &want.data[..], "cold row {r}");
    }
    let hits_after_cold = m.hits.load(Ordering::Relaxed);
    let misses_after_cold = m.misses.load(Ordering::Relaxed);
    assert_eq!(
        misses_after_cold, 4,
        "premise: four shards, decoded once each"
    );
    assert_eq!(
        indptr_decodes() - decodes0,
        4,
        "premise: cold, the prescan decodes one indptr per shard"
    );

    // Warm: the prescan peeks instead, so the only hits are the scatter's own —
    // one `read_shard_cached_arc` per shard request group.
    let warm = backed.read_row_indices(&rows).unwrap();
    assert_eq!(warm.indptr, cold.indptr);
    assert_eq!(warm.indices, cold.indices);
    assert_eq!(warm.data, cold.data);
    assert_eq!(
        m.misses.load(Ordering::Relaxed),
        misses_after_cold,
        "nothing should have been decoded twice"
    );
    assert_eq!(
        m.hits.load(Ordering::Relaxed) - hits_after_cold,
        4,
        "one hit per shard request group — the prescan's peek must add none"
    );
    // The claim the hit count cannot make: no second decode happened at all.
    // `shard_indptr` is invisible to the cache metrics by design, so it is
    // counted directly, in test builds only.
    assert_eq!(
        indptr_decodes() - decodes0,
        4,
        "the warm gather must not have decoded a single indptr"
    );
}
