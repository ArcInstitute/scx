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

// ---------------------------------------------------------------------------
// Size hints
// ---------------------------------------------------------------------------

/// `shard_size_hint` must survive the view, and must stay an upper bound.
///
/// PCA reads it to decide how much of the user's `memory_budget` to carve out
/// for decode-prefetch. If it stopped reaching the caller the carve would
/// silently become zero — no error, no wrong answer, just a peak-RSS ceiling
/// that no longer means what `pca(memory_budget=…)` documents.
#[test]
fn shard_size_hint_survives_transforms_projection_and_row_filter() {
    let dir = TempDir::new().unwrap();
    let reader = multishard_reader(&dir, N_SHARDS);
    let from_reader = ShardSource::shard_size_hint(&*reader).expect("catalog carries shard stats");

    let plain = full_source(&reader);
    assert_eq!(plain.shard_size_hint(), Some(from_reader));

    // Every stage that narrows the view can only *lower* the true figures, so
    // the on-disk numbers stay valid upper bounds rather than becoming stale.
    let narrowed = LazyShardSource::new(
        Arc::clone(&reader),
        vec![Transform::Log1p],
        Some(Arc::new((0..N_OBS as u64).step_by(2).collect())),
        Some(Arc::new(vec![0u32, 2])),
        N_OBS.div_ceil(2),
        2,
    );
    let hint = narrowed.shard_size_hint().expect("hint survives the view");
    assert_eq!(hint, from_reader);

    for shard in 0..N_SHARDS {
        let csr = narrowed.read_shard(shard).unwrap();
        assert!(
            csr.n_rows() <= hint.max_rows && csr.indices.len() <= hint.max_nnz,
            "shard {shard} ({} rows, {} nnz) exceeds the hint {hint:?} — an \
             under-bound is the one thing the contract forbids",
            csr.n_rows(),
            csr.indices.len()
        );
    }
}

// ---------------------------------------------------------------------------
// CSR / CSC transform parity
//
// `NormalizeTotal` and `RowScale` used to disqualify a lazy chain from CSC
// dispatch, on the grounds that they are not "column-local". They are not —
// but column-locality is the wrong test for a CSC reader, which knows each
// nonzero's global row because `ScxCsc::indices` *is* that row. Both
// transforms carry their per-row vector with them, so the CSC path needs a
// lookup, not a pass.
//
// What that buys, measured on `census_1m` (1M x 61,497, 50 groups, one
// `log1p` chain so the old gate admitted it): Wilcoxon DE 1,270 s / 7,319 MB
// on `cpu_csr` against 226 s / 5,652 MB on `cpu_csc`. The standard analyst
// chain is `normalize_total -> log1p`, which the old gate refused, so that
// 5.6x was unreachable from the path essentially everyone takes.
//
// The tests below are about the half that matters more than the speed: the
// two routes must agree **bit for bit**. They can, because every transform
// here is an element-wise map and none of them accumulates.
// ---------------------------------------------------------------------------

use scx_sparse::{ScxCsc, ScxCsr};

/// A small matrix chosen so each arm of the comparison can actually fail.
///
/// Every value/​sum pair here is adversarial on purpose, because a fixture of
/// round numbers makes this whole file vacuous: with `sum = 10` the factor
/// `1e4 / sum` is exactly representable, so widening or narrowing the
/// intermediate changes nothing and a mutant passes. The first draft of these
/// tests had exactly that hole — two of three mutations survived it.
///
/// - **Row 0** sums to 3, so `1e4 / 3` is not representable in f32, and its
///   value 7 is one of the pairs where an f64 intermediate (23333.334) and an
///   f32 one (23333.332) land on different floats.
/// - **Row 1** carries 4097 against the same sum, a second such pair at a
///   different magnitude.
/// - **Row 2** is structurally empty and **row 3** is stored zeros: two
///   distinct spellings of the zero-sum row whose normalisation CSR skips.
/// - **Row 4** holds 16,777,219, past f32's 2²⁴ exact-integer range.
/// - **Rows 6 and 7** are signed, and they are the rows this fixture was
///   missing. `from_anndata` accepts them, and they are the only way to reach
///   the `sum > 0.0` guard's *false* branch with something other than zeros:
///   row 6 cancels to a sum of exactly 0 and row 7 sums to −1. Without them
///   the guard is only ever taken on all-zero rows, where every candidate
///   behaviour agrees and the branch is untestable — which is how the fused
///   CSR path came to skip `ln_1p` there while every other route applied it.
fn parity_matrix() -> (ScxCsr, ScxCsc, Vec<f64>) {
    // row -> [(col, value)]
    let rows: Vec<Vec<(usize, f32)>> = vec![
        vec![(0, 1.0), (2, 2.0)],                    // sum 3 -> factor 3333.333...
        vec![(0, 4097.0), (1, 1.0), (2, 1.0)],       // sum 4099
        vec![],                                      // structurally empty
        vec![(0, 0.0), (1, 0.0)],                    // stored zeros -> zero-sum row
        vec![(0, 1.0), (1, 2.0), (2, 16_777_219.0)], // past f32's 2^24
        vec![(2, 7.0)],                              // sum 7
        vec![(0, 0.5), (1, -0.5)],                   // cancels: sum exactly 0
        vec![(0, -2.0), (1, 1.0)],                   // negative sum (-1)
    ];
    let n_rows = rows.len();
    let n_cols = 3;

    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for r in &rows {
        for &(c, v) in r {
            indices.push(c as i32);
            data.push(v);
        }
        indptr.push(data.len() as i64);
    }
    let csr = ScxCsr {
        shape: (n_rows, n_cols),
        indptr,
        indices,
        data,
    };

    // The same matrix column-major. Built independently of the CSR rather
    // than transposed from it, so a bug in one construction cannot hide in
    // both.
    let mut c_indptr = vec![0i64];
    let mut c_indices = Vec::new();
    let mut c_data = Vec::new();
    for c in 0..n_cols {
        for (r, row) in rows.iter().enumerate() {
            for &(cc, v) in row {
                if cc == c {
                    c_indices.push(r as i32);
                    c_data.push(v);
                }
            }
        }
        c_indptr.push(c_data.len() as i64);
    }
    let csc = ScxCsc {
        shape: (n_rows, n_cols),
        indptr: c_indptr,
        indices: c_indices,
        data: c_data,
    };

    let row_sums: Vec<f64> = rows
        .iter()
        .map(|r| r.iter().map(|&(_, v)| v as f64).sum())
        .collect();
    (csr, csc, row_sums)
}

/// Gather `(row, col) -> value` from each layout so the two can be compared
/// without depending on either's storage order.
fn csr_cells(csr: &ScxCsr) -> Vec<((usize, usize), f32)> {
    let mut out = Vec::new();
    for row in 0..csr.shape.0 {
        for k in csr.indptr[row] as usize..csr.indptr[row + 1] as usize {
            out.push(((row, csr.indices[k] as usize), csr.data[k]));
        }
    }
    out.sort_by_key(|&(rc, _)| rc);
    out
}

fn csc_cells(csc: &ScxCsc) -> Vec<((usize, usize), f32)> {
    let mut out = Vec::new();
    for col in 0..csc.shape.1 {
        for k in csc.indptr[col] as usize..csc.indptr[col + 1] as usize {
            out.push(((csc.indices[k] as usize, col), csc.data[k]));
        }
    }
    out.sort_by_key(|&(rc, _)| rc);
    out
}

fn assert_bit_identical(chain: &[Transform], label: &str) {
    let (mut csr, mut csc, _) = parity_matrix();
    super::super::transforms::apply_transforms_to_csr(chain, &mut csr, 0);
    super::apply_transforms_to_csc(chain, &mut csc);

    let a = csr_cells(&csr);
    let b = csc_cells(&csc);
    assert_eq!(a.len(), b.len(), "{label}: nnz differs");
    for ((rc_a, va), (rc_b, vb)) in a.iter().zip(b.iter()) {
        assert_eq!(rc_a, rc_b, "{label}: cell coordinates diverged");
        // `to_bits`, not `==`: this contract is exact agreement, and `==`
        // would also call two different NaNs unequal and two zeros of
        // opposite sign equal.
        assert_eq!(
            va.to_bits(),
            vb.to_bits(),
            "{label}: cell {rc_a:?} differs — CSR {va} vs CSC {vb}",
        );
    }
}

fn normalize(row_sums: &[f64]) -> Transform {
    Transform::NormalizeTotal {
        row_sums: Arc::new(row_sums.to_vec()),
        target_sum: 1e4,
    }
}

#[test]
fn normalize_total_is_bit_identical_across_layouts() {
    let (_, _, sums) = parity_matrix();
    assert_bit_identical(&[normalize(&sums)], "normalize_total");
}

#[test]
fn the_standard_normalize_then_log1p_chain_is_bit_identical() {
    // The chain `pipeline_ooc_constrained` runs, and the one the old gate
    // refused. CSR takes a *fused* path for this exact pair
    // (`((v * factor) as f32).ln_1p()`); CSC applies the two in sequence.
    // Agreement is not a coincidence — the fused form rounds to f32 between
    // the multiply and the log, which is what sequencing does too.
    let (_, _, sums) = parity_matrix();
    assert_bit_identical(&[normalize(&sums), Transform::Log1p], "normalize+log1p");
}

#[test]
fn row_scale_is_bit_identical_across_layouts() {
    // Not 0.5 / 0.75 / 1.0: those are f32-exact, so `factors[row] as f32`
    // and an f64 multiply agree and the test cannot see the difference.
    let factors: Vec<f64> = vec![1.0 / 3.0, 0.1, 1.0 / 7.0, 1.1, 0.7, 1.0 / 3.0, -0.25, 3.5];
    assert_bit_identical(
        &[Transform::RowScale {
            factors: Arc::new(factors),
        }],
        "row_scale",
    );
}

#[test]
fn a_mixed_chain_is_bit_identical_across_layouts() {
    let (_, _, sums) = parity_matrix();
    let factors: Vec<f64> = vec![1.0 / 3.0, 0.1, 1.0 / 7.0, 1.1, 0.7, 1.0 / 3.0, -0.25, 3.5];
    assert_bit_identical(
        &[
            normalize(&sums),
            Transform::Log1p,
            Transform::Scale { factor: 2.5 },
            Transform::RowScale {
                factors: Arc::new(factors),
            },
        ],
        "mixed",
    );
}

#[test]
fn a_zero_sum_row_is_left_alone_on_both_paths() {
    // Premise assertion for the tests above: without a non-positive-sum row in
    // the fixture, `sum > 0.0` is never false and the branch that skips
    // normalisation is never taken, so the parity tests would pass without
    // exercising it. Row 2 is structurally empty, row 3 stored zeros, and rows
    // 6 and 7 reach the same branch carrying **non-zero** values, which the
    // all-zero rows cannot do.
    let (_, _, sums) = parity_matrix();
    assert_eq!(sums[2], 0.0, "row 2 should be structurally empty");
    assert_eq!(sums[3], 0.0, "row 3 should be a stored-zero row");
    assert_eq!(sums[6], 0.0, "row 6 should cancel to exactly zero");
    assert!(sums[7] < 0.0, "row 7 should have a negative total");

    let (mut csr, mut csc, _) = parity_matrix();
    let chain = [normalize(&sums)];
    super::super::transforms::apply_transforms_to_csr(&chain, &mut csr, 0);
    super::apply_transforms_to_csc(&chain, &mut csc);
    for ((r, _), v) in csr_cells(&csr) {
        if r == 3 {
            assert_eq!(v, 0.0, "CSR changed a zero-sum row's values");
        }
    }
    for ((r, _), v) in csc_cells(&csc) {
        if r == 3 {
            assert_eq!(v, 0.0, "CSC changed a zero-sum row's values");
        }
    }
}

#[test]
fn the_fused_csr_prefix_agrees_with_the_unfused_one_on_a_non_positive_row() {
    // `apply_transforms_to_csr` special-cases a leading `NormalizeTotal →
    // Log1p` pair. On a row whose total is not positive, `NormalizeTotal` is
    // skipped but `Log1p` must still apply — the fused branch used to do
    // neither, so it silently disagreed with its own general path (and with
    // `apply_transforms_to_csc` and scanpy) on any signed row. Interposing an
    // identity `Scale` defeats the fusion, which is what makes the two paths
    // comparable on identical input.
    //
    // This covers only the `transforms.rs` copy of the fusion. The second
    // copy, `dataset_index.rs::apply_transforms_per_row`, is reached through
    // `__getitem__` rather than a shard source, so it is pinned end to end
    // instead, by `test_a_signed_row_reads_the_same_through_a_slice_and_a_fancy_index`.
    let (_, _, sums) = parity_matrix();
    let (mut fused, _, _) = parity_matrix();
    let (mut unfused, _, _) = parity_matrix();

    super::super::transforms::apply_transforms_to_csr(
        &[normalize(&sums), Transform::Log1p],
        &mut fused,
        0,
    );
    super::super::transforms::apply_transforms_to_csr(
        &[
            normalize(&sums),
            Transform::Scale { factor: 1.0 },
            Transform::Log1p,
        ],
        &mut unfused,
        0,
    );

    let (a, b) = (csr_cells(&fused), csr_cells(&unfused));
    assert_eq!(a.len(), b.len());
    for ((rc_a, va), (rc_b, vb)) in a.iter().zip(b.iter()) {
        assert_eq!(rc_a, rc_b);
        assert_eq!(
            va.to_bits(),
            vb.to_bits(),
            "cell {rc_a:?}: fused {va} vs unfused {vb}",
        );
    }
    // Premise: the signed rows really did take the non-positive branch, so the
    // equality above is not trivially true of a fixture that never reaches it.
    let row7: Vec<f32> = csr_cells(&fused)
        .into_iter()
        .filter(|&((r, _), _)| r == 7)
        .map(|(_, v)| v)
        .collect();
    assert!(
        row7.iter().any(|v| v.is_nan()),
        "row 7 holds -2.0, whose ln_1p is NaN; got {row7:?} — the branch was not taken",
    );
}

#[test]
fn the_csc_gate_refuses_without_a_sidecar_whatever_the_chain() {
    // `supports_csc` makes two decisions, and a transform chain is no longer
    // one of them: every variant is served column-major, so the chain the old
    // gate refused (`normalize_total → log1p`) cannot be the reason a source
    // is rejected. The exhaustive `match` in `apply_transforms_to_csc` is what
    // forces a future variant to be considered rather than admitted silently.
    //
    // Only the no-sidecar half is reachable here — `new_with_csc` needs a real
    // `BackedCscReader`, which these fixtures do not write. The sidecar-present
    // and row-filter halves are covered end to end from Python
    // (`test_csc_dispatch_lazy.py`), where a real CSC file exists.
    let dir = TempDir::new().unwrap();
    let reader = multishard_reader(&dir, 0);
    let (_, _, sums) = parity_matrix();

    for chain in [
        vec![],
        vec![Transform::Log1p],
        vec![normalize(&sums), Transform::Log1p],
    ] {
        let src = LazyShardSource::new_with_csc(
            Arc::clone(&reader),
            None,
            chain.clone(),
            None,
            None,
            N_OBS,
            N_VARS,
        );
        assert!(
            !src.supports_csc(),
            "no sidecar must refuse, chain len {}",
            chain.len()
        );
    }
}
