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
use scx_sparse::ScxCsr;

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

criterion_group!(benches, bench_transpose);
criterion_main!(benches);
