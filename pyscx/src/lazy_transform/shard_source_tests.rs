//! `LazyShardSource` view + caching behaviour.
//!
//! The Python suite (`pyscx/tests/test_pca_axis_view.py`) covers *correctness*
//! of the view end to end. What it cannot see is the second half of the
//! contract: that a multi-pass kernel served by this source still reads
//! through the reader's decoded-shard LRU.
//!
//! That distinction has teeth. `read_shard_uncached` also `MADV_DONTNEED`s the
//! shard bytes, so a source that quietly stops caching makes every one of
//! out-of-core PCA's ~6–7 passes re-fault *and* re-decode — a large, silent
//! slowdown with no wrong answer to catch it. There is no Python-observable
//! signal for it either: `pca(memory_budget=…)` calls `ensure_cache_capacity`,
//! which raises the count cap toward `n_shards`, so the kernel's
//! undersized-cache warning cannot be provoked from that side.

use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::{BackedCsrReader, FileHeader, ScxReader, ScxWriter, ShardSource};
use tempfile::TempDir;

use super::{LazyShardSource, Transform};

const N_OBS: usize = 12;
const N_VARS: usize = 6;
const N_SHARDS: usize = 3;
const ROWS_PER_SHARD: usize = N_OBS / N_SHARDS;

/// Two nnz per row: columns `row % N_VARS` and `(row + 1) % N_VARS`, with
/// values that identify the row, so a mis-selected row is visible in the data.
fn shard_data(first_row: usize, n_rows: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for r in 0..n_rows {
        let global = first_row + r;
        let (c0, c1) = (global % N_VARS, (global + 1) % N_VARS);
        let (lo, hi) = if c0 < c1 { (c0, c1) } else { (c1, c0) };
        indices.push(lo as u32);
        indices.push(hi as u32);
        values.push((global + 1) as u8);
        values.push((global + 1) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    (indptr, indices, values)
}

fn string_batch(name: &str, prefix: &str, n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("{prefix}{i}")).collect();
    let schema = Schema::new(vec![Field::new(name, DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(String::as_str).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

/// A three-shard file. Row-window behaviour is structurally invisible on a
/// single shard, so every fixture here is multi-shard on purpose.
fn multishard_reader(dir: &TempDir, cache_shards: usize) -> Arc<BackedCsrReader> {
    let path = dir.path().join("view.scx");
    let header = FileHeader::new_single_modality(
        N_OBS as u64,
        N_VARS as u64,
        (N_OBS * 2) as u64,
        16384,
        0,
        0,
    );
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer
        .write_obs(&string_batch("cell_id", "c", N_OBS))
        .unwrap();
    writer
        .write_var(&string_batch("gene_id", "g", N_VARS))
        .unwrap();
    for s in 0..N_SHARDS {
        let first = s * ROWS_PER_SHARD;
        let (indptr, indices, values) = shard_data(first, ROWS_PER_SHARD);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                first as u64,
            )
            .unwrap();
    }
    writer.finish().unwrap();
    Arc::new(BackedCsrReader::new(
        ScxReader::open(&path).unwrap(),
        cache_shards,
    ))
}

fn full_source(reader: &Arc<BackedCsrReader>) -> LazyShardSource {
    LazyShardSource::new(Arc::clone(reader), Vec::new(), None, None, N_OBS, N_VARS)
}

// ---------------------------------------------------------------------------
// Caching
// ---------------------------------------------------------------------------

#[test]
fn cached_reads_are_opt_in() {
    let dir = TempDir::new().unwrap();
    let reader = multishard_reader(&dir, N_SHARDS);

    // Off by default: single-pass callers (HVG, score_genes, pflog) keep the
    // uncached read, where the LRU is overhead and MADV_DONTNEED is a win.
    assert_eq!(full_source(&reader).shard_cache_capacity(), None);

    // Opting in republishes the reader's capacity, which is what lets a
    // multi-pass kernel warn about an undersized working set.
    let cached = full_source(&reader).with_cached_reads();
    assert_eq!(cached.shard_cache_capacity(), Some(reader.cache_capacity()));
}

#[test]
fn cached_reads_serve_repeat_passes_from_the_lru() {
    let dir = TempDir::new().unwrap();
    let reader = multishard_reader(&dir, N_SHARDS);
    let source = full_source(&reader).with_cached_reads();

    // A passthrough source hands back the cached `Arc` itself, so a second
    // pass over the same shard is pointer-identical to the first. That is the
    // property out-of-core PCA depends on; an uncached source cannot have it.
    let first = source.read_shard_arc(1).unwrap();
    let second = source.read_shard_arc(1).unwrap();
    assert!(
        Arc::ptr_eq(&first, &second),
        "cached passthrough must reuse the decoded shard, not re-decode it"
    );

    let uncached = full_source(&reader);
    let a = uncached.read_shard_arc(1).unwrap();
    let b = uncached.read_shard_arc(1).unwrap();
    assert!(!Arc::ptr_eq(&a, &b));
    assert_eq!(a.indices, b.indices, "…but still decodes the same bytes");
}

#[test]
fn cached_and_uncached_reads_agree() {
    let dir = TempDir::new().unwrap();
    let reader = multishard_reader(&dir, N_SHARDS);
    for shard in 0..N_SHARDS {
        let plain = full_source(&reader).read_shard(shard).unwrap();
        let cached = full_source(&reader)
            .with_cached_reads()
            .read_shard(shard)
            .unwrap();
        assert_eq!(plain.shape, cached.shape);
        assert_eq!(plain.indptr, cached.indptr);
        assert_eq!(plain.indices, cached.indices);
        assert_eq!(plain.data, cached.data);
    }
}

#[test]
fn read_shard_does_not_alias_the_cache() {
    // `read_shard` returns an owned CSR. On the cached passthrough path the
    // `Arc` is shared, so it has to clone — mutating the result must not
    // corrupt the LRU entry every later pass will read.
    let dir = TempDir::new().unwrap();
    let reader = multishard_reader(&dir, N_SHARDS);
    let source = full_source(&reader).with_cached_reads();

    let cached = source.read_shard_arc(0).unwrap();
    let mut owned = source.read_shard(0).unwrap();
    owned.data[0] = 222.0;
    assert_ne!(owned.data[0], cached.data[0]);
    assert_eq!(source.read_shard_arc(0).unwrap().data, cached.data);
}

// ---------------------------------------------------------------------------
// The view: kept_to_global and col_projection
// ---------------------------------------------------------------------------

#[test]
fn kept_to_global_selects_rows_within_each_shard() {
    let dir = TempDir::new().unwrap();
    let reader = multishard_reader(&dir, N_SHARDS);
    // One row from each shard, plus one that leaves shard 2 with a single row.
    let kept: Vec<u64> = vec![1, 5, 9, 10];
    let source = LazyShardSource::new(
        Arc::clone(&reader),
        Vec::new(),
        Some(Arc::new(kept.clone())),
        None,
        kept.len(),
        N_VARS,
    )
    .with_cached_reads();

    assert_eq!(ShardSource::n_obs(&source), kept.len());

    let mut seen = Vec::new();
    for shard in 0..N_SHARDS {
        let csr = source.read_shard(shard).unwrap();
        for r in 0..csr.n_rows() {
            // Values encode `global_row + 1` (see `shard_data`).
            seen.push(csr.data[csr.indptr[r] as usize] as u64 - 1);
        }
    }
    assert_eq!(seen, kept, "rows must be the kept globals, in order");
}

#[test]
fn a_shard_with_no_kept_rows_yields_an_empty_projected_shard() {
    let dir = TempDir::new().unwrap();
    let reader = multishard_reader(&dir, N_SHARDS);
    let cols: Vec<u32> = vec![0, 2];
    let kept: Vec<u64> = vec![0, 1]; // shard 0 only
    let source = LazyShardSource::new(
        Arc::clone(&reader),
        Vec::new(),
        Some(Arc::new(kept)),
        Some(Arc::new(cols.clone())),
        2,
        cols.len(),
    );

    let empty = source.read_shard(1).unwrap();
    assert_eq!(empty.n_rows(), 0);
    assert_eq!(
        empty.n_cols(),
        cols.len(),
        "an empty shard must still be sized to the projected width"
    );
}

#[test]
fn col_projection_narrows_the_reported_var_axis() {
    let dir = TempDir::new().unwrap();
    let reader = multishard_reader(&dir, N_SHARDS);
    let cols: Vec<u32> = vec![1, 3, 4];
    let source = LazyShardSource::new(
        Arc::clone(&reader),
        Vec::new(),
        None,
        Some(Arc::new(cols.clone())),
        N_OBS,
        cols.len(),
    )
    .with_cached_reads();

    assert_eq!(ShardSource::n_vars(&source), cols.len());
    for shard in 0..N_SHARDS {
        let csr = source.read_shard(shard).unwrap();
        assert_eq!(csr.n_cols(), cols.len());
        assert!(csr.indices.iter().all(|&c| (c as usize) < cols.len()));
    }
}

#[test]
fn projection_does_not_write_through_to_the_cached_shard() {
    // The derive stages read straight out of the cached `Arc` to skip a copy.
    // If one ever mutated in place instead, the LRU entry would be poisoned
    // for every other consumer of the same reader — including an unprojected
    // source. Pin that they stay independent.
    let dir = TempDir::new().unwrap();
    let reader = multishard_reader(&dir, N_SHARDS);
    let cols: Vec<u32> = vec![0, 1];

    let projected = LazyShardSource::new(
        Arc::clone(&reader),
        Vec::new(),
        None,
        Some(Arc::new(cols.clone())),
        N_OBS,
        cols.len(),
    )
    .with_cached_reads();
    let _ = projected.read_shard(0).unwrap();

    let plain = full_source(&reader)
        .with_cached_reads()
        .read_shard(0)
        .unwrap();
    assert_eq!(plain.n_cols(), N_VARS);
    let expected = full_source(&reader).read_shard(0).unwrap();
    assert_eq!(plain.indices, expected.indices);
    assert_eq!(plain.data, expected.data);
}

// ---------------------------------------------------------------------------
// The transform stage, and caching *through* a view
// ---------------------------------------------------------------------------

#[test]
fn transforms_compose_with_projection_and_row_filter() {
    // Every other test builds sources with `Vec::new()` transforms, so the
    // `current = Some(owned)` hand-off from the transform stage into projection
    // and then row filtering is otherwise unexercised.
    let dir = TempDir::new().unwrap();
    let reader = multishard_reader(&dir, N_SHARDS);
    let cols: Vec<u32> = vec![1, 2, 3];
    let kept: Vec<u64> = vec![1, 5, 9];

    let plain = LazyShardSource::new(
        Arc::clone(&reader),
        Vec::new(),
        Some(Arc::new(kept.clone())),
        Some(Arc::new(cols.clone())),
        kept.len(),
        cols.len(),
    );
    let logged = LazyShardSource::new(
        Arc::clone(&reader),
        vec![Transform::Log1p],
        Some(Arc::new(kept.clone())),
        Some(Arc::new(cols.clone())),
        kept.len(),
        cols.len(),
    );

    for shard in 0..N_SHARDS {
        let a = plain.read_shard(shard).unwrap();
        let b = logged.read_shard(shard).unwrap();
        // The transform runs, and the projection + row filter still apply on
        // top of its output rather than on the raw shard.
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.indptr, b.indptr);
        assert_eq!(a.indices, b.indices);
        for (raw, t) in a.data.iter().zip(b.data.iter()) {
            assert!(
                (t - raw.ln_1p()).abs() < 1e-6,
                "log1p not applied before projection/filter: {raw} -> {t}"
            );
        }
    }
}

/// Phase 4.2 made the transform stage take ownership of the decoded shard via
/// `Arc::try_unwrap` instead of always cloning it. That is free on the uncached
/// path — but it must still clone on a **cache hit**, because mutating the
/// shard the LRU is holding would poison it for every other reader of the same
/// file. `try_unwrap` fails there (the cache holds a strong ref), which is what
/// makes the optimisation safe; this pins that it stays that way.
#[test]
fn transforms_do_not_write_through_to_the_cached_shard() {
    let dir = TempDir::new().unwrap();
    let reader = multishard_reader(&dir, N_SHARDS);

    // Warm the LRU with the untransformed shard, and keep a copy of the truth.
    let expected = full_source(&reader)
        .with_cached_reads()
        .read_shard(0)
        .unwrap();

    // A transformed source over the same reader now takes a cache hit.
    let logged = LazyShardSource::new(
        Arc::clone(&reader),
        vec![Transform::Log1p],
        None,
        None,
        N_OBS,
        N_VARS,
    )
    .with_cached_reads();
    let transformed = logged.read_shard(0).unwrap();
    assert!(
        transformed
            .data
            .iter()
            .zip(expected.data.iter())
            .any(|(t, e)| (t - e).abs() > 1e-6),
        "fixture has no values log1p actually changes — the write-through \
         check below cannot fail"
    );

    // Re-read through a plain source: the cached shard must still be raw.
    let after = full_source(&reader)
        .with_cached_reads()
        .read_shard(0)
        .unwrap();
    assert_eq!(
        after.data, expected.data,
        "the transform mutated the LRU's shard in place — every other consumer \
         of this reader would now see log1p'd values"
    );
}

#[test]
fn cached_reads_survive_a_view() {
    // The regression this guards: dropping the LRU specifically for the
    // *subset* multi-pass case. A view derives a fresh CSR per read, so
    // `Arc::ptr_eq` cannot show reuse — but the decode underneath must still
    // come from the cache, which `shard_cache_capacity()` reports and the
    // singleflight/LRU makes observable through repeated identical output.
    let dir = TempDir::new().unwrap();
    let reader = multishard_reader(&dir, N_SHARDS);
    let kept: Vec<u64> = vec![1, 5, 9, 10];
    let build = || {
        LazyShardSource::new(
            Arc::clone(&reader),
            Vec::new(),
            Some(Arc::new(kept.clone())),
            None,
            kept.len(),
            N_VARS,
        )
        .with_cached_reads()
    };

    let source = build();
    assert_eq!(
        source.shard_cache_capacity(),
        Some(reader.cache_capacity()),
        "a view-bearing cached source must still publish the reader's LRU"
    );

    // Repeated passes over the same shard agree exactly — the derived CSR is
    // rebuilt, the decode is not.
    let first = source.read_shard_arc(1).unwrap();
    let second = source.read_shard_arc(1).unwrap();
    assert_eq!(first.indptr, second.indptr);
    assert_eq!(first.indices, second.indices);
    assert_eq!(first.data, second.data);

    // And the view is still applied — not silently bypassed by the cache.
    let uncached = LazyShardSource::new(
        Arc::clone(&reader),
        Vec::new(),
        Some(Arc::new(kept.clone())),
        None,
        kept.len(),
        N_VARS,
    );
    assert_eq!(
        source.read_shard(1).unwrap().n_rows(),
        uncached.read_shard(1).unwrap().n_rows()
    );
}
