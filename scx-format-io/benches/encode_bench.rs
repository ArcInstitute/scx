//! Shard-encode benchmarks: what the intra-shard parallelism is worth.
//!
//! Run it twice to get the speedup **and** the thread count it came from —
//! rayon reads `RAYON_NUM_THREADS` once, at first pool use, so a single
//! process cannot sweep widths:
//!
//! ```text
//! RAYON_NUM_THREADS=1 cargo bench -p scx-format-io -- --save-baseline serial
//! cargo bench -p scx-format-io -- --baseline serial
//! ```
//!
//! The four arms are the encoder's four distinct shapes, not four sizes:
//!
//! * `unframed` — one `encode_shard` for the whole shard. The control: no
//!   framing means no row groups to spread, so this arm must not move.
//! * `framed_fast` — row-group-framed at the default G, single-encode
//!   (`decode_target: None`, the `fast` profile). Isolates the row-group axis.
//! * `framed_auto` — framed *and* dual-encoded against `ShufDeltaZstd`, which
//!   is what `codec="auto"` does to every integer shard. Both axes at once,
//!   and the only arm a default write actually takes.
//! * `framed_auto_float` — the float path. `ShufDeltaZstd` never competes for
//!   float, so this is the row-group axis under `Pcodec`, and it is here to
//!   show that the `auto` gain above is not all coming from the second
//!   candidate.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use scx_codec::value_encoding::values_to_raw_bytes;
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::codec_select::DecodeTarget;
use scx_format_io::encoder::{encode_shard_adaptive, FramingConfig, DEFAULT_ROW_GROUP_ROWS};

/// A census-like integer shard: `n_rows` rows of `nnz` sorted unique columns,
/// counts in `1..=max_count`. Deliberately the same generator shape as
/// `encoder.rs`'s `gen_int_shard`, so a number here is comparable with the
/// unit tests' fixtures.
fn gen_shard(
    n_rows: usize,
    nnz: usize,
    n_cols: u32,
    max_count: u32,
) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let mut indptr = Vec::with_capacity(n_rows + 1);
    let mut indices = Vec::with_capacity(n_rows * nnz);
    let mut values = Vec::with_capacity(n_rows * nnz);
    indptr.push(0u64);
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let stride = (n_cols as usize / nnz.max(1)).max(1);
    for row in 0..n_rows {
        let mut cols: Vec<u32> = (0..nnz)
            .map(|k| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let jitter = (state >> 33) as usize % stride.max(1);
                ((k * stride + jitter + row) % n_cols as usize) as u32
            })
            .collect();
        cols.sort_unstable();
        cols.dedup();
        for (k, col) in cols.iter().enumerate() {
            indices.push(*col);
            values.push(((row + k) as u32 % max_count + 1) as f32);
        }
        indptr.push(indices.len() as u64);
    }
    (indptr, indices, values)
}

fn framing(decode_target: Option<DecodeTarget>) -> FramingConfig {
    FramingConfig {
        row_group_rows: DEFAULT_ROW_GROUP_ROWS,
        target_nnz: None,
        trial: false,
        decode_target,
    }
}

fn bench_encode(c: &mut Criterion) {
    // 4096 rows at the default G = 256 is 16 row groups — the shape a
    // `shard_target_rows = 16384` write reaches with 64.
    let (indptr, indices, values) = gen_shard(4096, 60, 20_000, 255);
    let nnz = *indptr.last().unwrap();

    let mut group = c.benchmark_group("encode_shard_adaptive");
    group.throughput(Throughput::Elements(nnz));

    let int_enc = ValueEncoding::Uint8;
    let int_bytes = values_to_raw_bytes(&values, int_enc).expect("int values");
    let float_enc = ValueEncoding::Float32;
    let float_bytes = values_to_raw_bytes(&values, float_enc).expect("float values");

    group.bench_function("unframed", |b| {
        b.iter(|| {
            encode_shard_adaptive(
                &indptr,
                &indices,
                &int_bytes,
                CodecId::Zstd,
                int_enc,
                false,
                None,
            )
            .expect("encode")
        })
    });
    group.bench_function("framed_fast", |b| {
        b.iter(|| {
            encode_shard_adaptive(
                &indptr,
                &indices,
                &int_bytes,
                CodecId::Zstd,
                int_enc,
                false,
                Some(framing(None)),
            )
            .expect("encode")
        })
    });
    group.bench_function("framed_auto", |b| {
        b.iter(|| {
            encode_shard_adaptive(
                &indptr,
                &indices,
                &int_bytes,
                CodecId::Zstd,
                int_enc,
                false,
                Some(framing(Some(DecodeTarget::Auto))),
            )
            .expect("encode")
        })
    });
    group.bench_function("framed_auto_float", |b| {
        b.iter(|| {
            encode_shard_adaptive(
                &indptr,
                &indices,
                &float_bytes,
                CodecId::Pcodec,
                float_enc,
                false,
                Some(framing(Some(DecodeTarget::Auto))),
            )
            .expect("encode")
        })
    });
    group.finish();
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
