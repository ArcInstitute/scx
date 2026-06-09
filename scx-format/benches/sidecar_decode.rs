// Criterion benchmark for the Task 4.3 decode-sidecar fast path.
//
// Run: cargo bench -p scx-format --bench sidecar_decode
//
// Compares, on one Scx1 shard:
//   * `full_sequential`   — `decode_shard_ref` (no sidecar; the baseline).
//   * `window_random`     — `decode_scx1_row_range` over a 256-row mid-shard
//                           window (the O(window) random-access win).
//   * `full_parallel`     — `decode_scx1_parallel` across cores (the multi-core
//                           full-decode win).
//
// `full_sequential` vs `window_random` quantifies random access; `full_parallel`
// vs `full_sequential` quantifies parallel speedup. Pure CPU — no GPU needed.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use scx_codec::{
    decode_scx1_row_range, decode_shard_ref, encode_shard, CodecId, EncodedShardRef, ValueEncoding,
};
use scx_format::decode_scx1_parallel;

/// Build one Scx1 shard (+ its encoder-emitted sidecar metadata) shaped like
/// real scRNA-seq: `n_rows` rows averaging `avg_nnz` non-zeros (≥128 ⇒ BitPacker4x).
fn build_shard(
    n_rows: usize,
    avg_nnz: usize,
    n_cols: u32,
) -> (
    Vec<u8>,
    Vec<u8>,
    Vec<u8>,
    scx_codec::Scx1DecodeMetadata,
    usize,
    usize,
) {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || -> u64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut indptr = vec![0u64];
    let mut indices: Vec<u32> = Vec::new();
    let mut values: Vec<u8> = Vec::new();
    for _ in 0..n_rows {
        let target = ((avg_nnz / 2) + (next() as usize % (avg_nnz + 1))).min(n_cols as usize);
        let mut row: Vec<u32> = Vec::with_capacity(target);
        let mut prev = 0u32;
        for _ in 0..target {
            let gap = 1 + (next() % ((n_cols as u64 / target.max(1) as u64).max(1)));
            prev = prev.saturating_add(gap as u32).min(n_cols - 1);
            row.push(prev);
        }
        row.sort_unstable();
        row.dedup(); // strictly increasing, unique
        for &idx in &row {
            indices.push(idx);
            let v = (1 + (next() % 97)) as u16; // non-zero (Scx1 Rice requires ≥1)
            values.extend_from_slice(&v.to_le_bytes());
        }
        indptr.push(indices.len() as u64);
    }
    let nnz = indices.len();
    let venc = ValueEncoding::Uint16;
    let encoded = encode_shard(&indptr, &indices, &values, CodecId::Scx1, venc, false).unwrap();
    let meta = encoded.scx1_decode.clone().expect("Scx1 emits metadata");
    (
        encoded.indptr_bytes,
        encoded.indices_bytes,
        encoded.values_bytes,
        meta,
        n_rows,
        nnz,
    )
}

fn bench_sidecar_decode(c: &mut Criterion) {
    let n_rows = 16_384usize;
    let n_cols = 20_000u32;
    let (ib, xb, vb, meta, _nr, nnz) = build_shard(n_rows, 256, n_cols);
    let venc = ValueEncoding::Uint16;
    let r = EncodedShardRef {
        indptr_bytes: &ib,
        indices_bytes: &xb,
        values_bytes: &vb,
    };

    let mut group = c.benchmark_group("sidecar_decode");
    group.throughput(Throughput::Elements(nnz as u64));

    group.bench_function("full_sequential", |b| {
        b.iter(|| {
            black_box(decode_shard_ref(&r, CodecId::Scx1, venc, n_rows, nnz, false).unwrap());
        })
    });

    // 256-row window in the middle of the shard.
    let win = 256usize;
    let win_start = n_rows / 2;
    group.bench_function("window_random_256rows", |b| {
        b.iter(|| {
            black_box(decode_scx1_row_range(&r, venc, &meta, win_start, win).unwrap());
        })
    });

    group.bench_function("full_parallel", |b| {
        b.iter(|| {
            black_box(decode_scx1_parallel(&r, venc, &meta, 1024).unwrap());
        })
    });

    group.finish();
}

criterion_group!(benches, bench_sidecar_decode);
criterion_main!(benches);
