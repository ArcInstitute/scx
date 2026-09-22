// Criterion benchmark for the chunked CSR→CSC transpose (P5 / OPT-1.4).
//
// Run: cargo bench -p scx-sparse --bench transpose_bench
//
// Before/after isolation, mirroring `scx-codec/benches/codec_bench.rs`
// (which keeps the pre-optimization `OldBitWriter` inline next to the new
// `BitWriter`): `transpose_chunk_sort` is the OLD sort-based body retained as a
// baseline; `transpose_chunk_scatter` is the NEW O(nnz) counting scatter that
// now lives in `transpose.rs::transpose_column_chunk`. Both operate on the same
// synthetic shards through the public `ScxCsr` API, so the only variable is the
// algorithm (sort vs. counting scatter).

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use scx_sparse::{
    streaming_csr_to_csc_iter_with_cap, CscBuilder, CscBuilderConfig, MemSpillStore, ScxCsr,
};

/// OLD: collect (local_col, row, value) tuples then stable-sort by (col, row).
fn transpose_chunk_sort(
    shards: &[ScxCsr],
    col_start: usize,
    col_end: usize,
) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
    let chunk_n_cols = col_end - col_start;

    let mut entries: Vec<(usize, i32, f32)> = Vec::new();
    let mut row_offset: usize = 0;
    for shard in shards {
        let shard_n_rows = shard.n_rows();
        for row in 0..shard_n_rows {
            let start = shard.indptr[row] as usize;
            let end = shard.indptr[row + 1] as usize;
            for j in start..end {
                let col = shard.indices[j] as usize;
                if col >= col_start && col < col_end {
                    let global_row = (row_offset + row) as i32;
                    entries.push((col - col_start, global_row, shard.data[j]));
                }
            }
        }
        row_offset += shard_n_rows;
    }

    entries.sort_by_key(|&(col, row, _)| (col, row));

    let mut indptr = vec![0i64; chunk_n_cols + 1];
    let mut indices = Vec::with_capacity(entries.len());
    let mut data = Vec::with_capacity(entries.len());
    for &(local_col, _, _) in &entries {
        indptr[local_col + 1] += 1;
    }
    for i in 1..=chunk_n_cols {
        indptr[i] += indptr[i - 1];
    }
    for &(_, row, val) in &entries {
        indices.push(row);
        data.push(val);
    }
    (indptr, indices, data)
}

/// NEW: O(nnz) counting scatter (count → prefix-sum → scatter). Mirrors the
/// current `transpose_column_chunk` body.
fn transpose_chunk_scatter(
    shards: &[ScxCsr],
    col_start: usize,
    col_end: usize,
) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
    let chunk_n_cols = col_end - col_start;

    let mut col_counts = vec![0usize; chunk_n_cols];
    for shard in shards {
        for &col in &shard.indices {
            let col = col as usize;
            if col >= col_start && col < col_end {
                col_counts[col - col_start] += 1;
            }
        }
    }

    let mut indptr = Vec::with_capacity(chunk_n_cols + 1);
    indptr.push(0i64);
    let mut cumsum = 0i64;
    for &count in &col_counts {
        cumsum += count as i64;
        indptr.push(cumsum);
    }
    let nnz = cumsum as usize;

    let mut indices = vec![0i32; nnz];
    let mut data = vec![0.0f32; nnz];
    let mut cursor = vec![0usize; chunk_n_cols];
    let mut row_offset: usize = 0;
    for shard in shards {
        let shard_n_rows = shard.n_rows();
        for row in 0..shard_n_rows {
            let start = shard.indptr[row] as usize;
            let end = shard.indptr[row + 1] as usize;
            for j in start..end {
                let col = shard.indices[j] as usize;
                if col >= col_start && col < col_end {
                    let local_col = col - col_start;
                    let dest = indptr[local_col] as usize + cursor[local_col];
                    indices[dest] = (row_offset + row) as i32;
                    data[dest] = shard.data[j];
                    cursor[local_col] += 1;
                }
            }
        }
        row_offset += shard_n_rows;
    }
    (indptr, indices, data)
}

/// Build `n_shards` row-shards of a synthetic `n_rows × n_cols` CSR with roughly
/// `nnz_per_row` sorted-unique columns per row (deterministic via a seeded RNG).
fn build_shards(n_rows: usize, n_cols: usize, nnz_per_row: usize, n_shards: usize) -> Vec<ScxCsr> {
    let mut rng = ChaCha8Rng::seed_from_u64(0xC5C0_0114);
    let rows_per_shard = n_rows.div_ceil(n_shards);
    let mut shards = Vec::with_capacity(n_shards);

    for s in 0..n_shards {
        let row_lo = s * rows_per_shard;
        let row_hi = ((s + 1) * rows_per_shard).min(n_rows);
        let shard_rows = row_hi.saturating_sub(row_lo);

        let mut indptr = Vec::with_capacity(shard_rows + 1);
        indptr.push(0i64);
        let mut indices: Vec<i32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();

        for _ in 0..shard_rows {
            // Draw sorted-unique column indices for this row.
            let mut cols: Vec<u32> = (0..nnz_per_row)
                .map(|_| rng.gen_range(0..n_cols as u32))
                .collect();
            cols.sort_unstable();
            cols.dedup();
            for &c in &cols {
                indices.push(c as i32);
                data.push(rng.gen_range(1.0..100.0));
            }
            indptr.push(indices.len() as i64);
        }

        shards.push(ScxCsr::new((shard_rows, n_cols), indptr, indices, data).unwrap());
    }
    shards
}

fn bench_transpose(c: &mut Criterion) {
    // ~50k × 5k, ~3 nnz/row → ~150k nnz, split across 4 row-shards.
    let n_rows = 50_000;
    let n_cols = 5_000;
    let shards = build_shards(n_rows, n_cols, 3, 4);
    let nnz: usize = shards.iter().map(|s| s.nnz()).sum();

    let mut group = c.benchmark_group("transpose_column_chunk_50k_x_5k");
    group.throughput(Throughput::Elements(nnz as u64));

    group.bench_function("old_sort", |b| {
        b.iter(|| transpose_chunk_sort(black_box(&shards), 0, n_cols));
    });
    group.bench_function("new_scatter", |b| {
        b.iter(|| transpose_chunk_scatter(black_box(&shards), 0, n_cols));
    });

    group.finish();
}

/// The whole build, both ways, **swept over the chunk count**.
///
/// `bench_transpose` above times one column chunk, which is the wrong unit for
/// the cost that dominates: the chunked transpose rescans every nonzero of
/// every shard twice *per chunk*, so its total is `2 * nnz * n_chunks` while
/// the builder's is `2 * nnz` plus one encode/parse of a re-partitioned copy.
/// A single-chunk benchmark cannot see that and reports the two as near-equal.
///
/// One point cannot carry the claim either — any speedup divided by any chunk
/// count yields some ratio. Sweeping `cols_per_shard` holds nnz fixed and
/// varies only `n_chunks`, so the chunked arm's wall must rise roughly
/// linearly in it while the builder's stays flat. That is the falsifiable
/// form: if the builder's arm also rises, the bucketing is not O(nnz).
fn bench_full_build(c: &mut Criterion) {
    let n_rows = 20_000;
    let n_cols = 8_000;
    let shards = build_shards(n_rows, n_cols, 20, 8);
    let nnz: usize = shards.iter().map(|s| s.nnz()).sum();

    for cols_per_shard in [800usize, 200, 50] {
        let n_chunks = n_cols.div_ceil(cols_per_shard);
        let mut group = c.benchmark_group(format!("full_csc_build_20k_x_8k/{n_chunks}_chunks"));
        group.throughput(Throughput::Elements(nnz as u64));
        group.sample_size(10);

        group.bench_function("chunked_transpose", |b| {
            b.iter(|| {
                let mut it = streaming_csr_to_csc_iter_with_cap(
                    black_box(&shards),
                    n_rows,
                    n_cols,
                    usize::MAX,
                    cols_per_shard,
                )
                .unwrap();
                let mut total = 0usize;
                for chunk in &mut it {
                    total += chunk.unwrap().indices.len();
                }
                total
            });
        });

        group.bench_function("csc_builder", |b| {
            b.iter(|| {
                let cfg = CscBuilderConfig {
                    cols_per_shard,
                    memory_bytes: usize::MAX,
                    spill_after_bytes: usize::MAX,
                    ..CscBuilderConfig::default()
                };
                let mut builder =
                    CscBuilder::new(n_rows, n_cols, cfg, Box::new(MemSpillStore::new())).unwrap();
                let mut row_start = 0u64;
                for s in black_box(&shards) {
                    builder.push_shard(row_start, s).unwrap();
                    row_start += s.n_rows() as u64;
                }
                let mut em = builder.finish().unwrap();
                let (mut ip, mut ix, mut dt) = (Vec::new(), Vec::new(), Vec::new());
                let mut total = 0usize;
                while em
                    .next_shard_into(&mut ip, &mut ix, &mut dt)
                    .unwrap()
                    .is_some()
                {
                    total += ix.len();
                }
                total
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_transpose, bench_full_build);
criterion_main!(benches);
