// Criterion benchmarks for scx-codec decode performance.
//
// Run: cargo bench -p scx-codec

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use scx_codec::forbp::{forbp_decode, forbp_encode};
use scx_codec::rice::{rice_decode, rice_encode, B_VAL};
use scx_codec::{decode_shard, encode_shard, CodecId, ValueEncoding};

/// Generate realistic CSR data for benchmarking.
/// Returns (indptr, indices, values_u8, n_rows, nnz).
fn generate_shard(n_rows: usize, avg_nnz_per_row: usize, n_vars: u32) -> (Vec<u64>, Vec<u32>, Vec<u8>, usize, usize) {
    let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let mut next = || -> u64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let mut indptr = Vec::with_capacity(n_rows + 1);
    let mut indices = Vec::new();
    let mut values = Vec::new();
    indptr.push(0u64);

    for _ in 0..n_rows {
        // Vary nnz around average (50% to 150%)
        let nnz = ((avg_nnz_per_row as u64 / 2) + next() % (avg_nnz_per_row as u64 + 1)) as usize;
        let nnz = nnz.min(n_vars as usize);

        // Generate sorted unique column indices
        let mut row_indices: Vec<u32> = Vec::with_capacity(nnz);
        let mut prev = 0u32;
        for _ in 0..nnz {
            let gap = 1 + (next() % ((n_vars as u64 / nnz.max(1) as u64).max(1)));
            prev = prev.saturating_add(gap as u32).min(n_vars - 1);
            row_indices.push(prev);
        }
        row_indices.sort_unstable();
        row_indices.dedup();

        let actual_nnz = row_indices.len();
        indptr.push(indptr.last().unwrap() + actual_nnz as u64);
        indices.extend_from_slice(&row_indices);

        // UMI-like values (mostly 1-5, occasional outlier)
        for _ in 0..actual_nnz {
            let v = match next() % 100 {
                0..=60 => 1u8,
                61..=80 => 2,
                81..=90 => 3,
                91..=95 => (4 + next() % 4) as u8,
                _ => (8 + next() % 20) as u8,
            };
            values.push(v);
        }
    }

    let nnz = *indptr.last().unwrap() as usize;
    (indptr, indices, values, n_rows, nnz)
}

fn bench_forbp_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("forbp_decode");

    for &(n_rows, avg_nnz, label) in &[
        (128, 200, "128r_200nnz"),
        (2048, 500, "2048r_500nnz"),
        (16384, 2000, "16384r_2000nnz"),
    ] {
        let (indptr, indices, _, n_rows_actual, _) = generate_shard(n_rows, avg_nnz, 30000);
        let row_lengths: Vec<usize> = indptr.windows(2).map(|w| (w[1] - w[0]) as usize).collect();
        let encoded = forbp_encode(&indices, &row_lengths, true);

        group.bench_with_input(BenchmarkId::new("u16", label), &encoded, |b, enc| {
            b.iter(|| {
                forbp_decode(black_box(enc), black_box(n_rows_actual), black_box(true)).unwrap()
            })
        });
    }

    group.finish();
}

fn bench_rice_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("rice_decode");

    for &(n_values, label) in &[
        (1000, "1K"),
        (100_000, "100K"),
        (1_000_000, "1M"),
    ] {
        // Generate UMI-like values (1-based)
        let mut state: u64 = 0xCAFE_BABE_1234_5678;
        let values: Vec<u32> = (0..n_values)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                match state % 100 {
                    0..=60 => 1,
                    61..=80 => 2,
                    81..=90 => 3,
                    91..=95 => 4 + (state % 4) as u32,
                    _ => 8 + (state % 20) as u32,
                }
            })
            .collect();
        let encoded = rice_encode(&values, B_VAL);

        group.bench_with_input(BenchmarkId::new("umi", label), &(encoded.clone(), n_values), |b, (enc, n)| {
            b.iter(|| {
                rice_decode(black_box(enc), black_box(*n), black_box(B_VAL)).unwrap()
            })
        });
    }

    group.finish();
}

fn bench_decode_shard(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_shard");

    for &(n_rows, avg_nnz, label) in &[
        (2048, 500, "2048r_500nnz"),
        (16384, 2000, "16384r_2000nnz"),
    ] {
        let (indptr, indices, values, n_rows_actual, nnz) = generate_shard(n_rows, avg_nnz, 30000);
        let index_dtype_u16 = true;

        let encoded = encode_shard(
            &indptr,
            &indices,
            &values,
            CodecId::Scx1,
            ValueEncoding::Uint8,
            index_dtype_u16,
        )
        .unwrap();

        group.bench_with_input(BenchmarkId::new("scx1", label), &encoded, |b, enc| {
            b.iter(|| {
                decode_shard(
                    black_box(enc),
                    black_box(CodecId::Scx1),
                    black_box(ValueEncoding::Uint8),
                    black_box(n_rows_actual),
                    black_box(nnz),
                    black_box(index_dtype_u16),
                )
                .unwrap()
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_forbp_decode, bench_rice_decode, bench_decode_shard);
criterion_main!(benches);
