// Criterion benchmarks for scx-codec decode performance.
//
// Run: cargo bench -p scx-codec

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scx_codec::bitstream::BitWriter;
use scx_codec::forbp::{forbp_decode, forbp_encode};
use scx_codec::rice::{rice_decode, rice_encode, B_VAL};
use scx_codec::{decode_shard, encode_shard, CodecId, ValueEncoding};

// ---------------------------------------------------------------------------
// P3 / OPT-2.1 before/after: u64-buffered BitWriter vs the old per-bit writer.
//
// `OldBitWriter` is the pre-OPT-2.1 implementation (write a bit at a time);
// `BitWriter` is the current u64-buffered one. Both are driven through the
// `Bw` trait by the identical `rice_pack` routine so the only variable is the
// bit-packing strategy — this isolates the encode-path win the rest of the
// suite (decode-only) does not measure.
// ---------------------------------------------------------------------------

/// The original per-bit `BitWriter` (LSB-first, one bit per push), retained
/// here only as the encode-path baseline.
struct OldBitWriter {
    buffer: Vec<u8>,
    current_byte: u8,
    bit_pos: u8,
}

impl OldBitWriter {
    fn new() -> Self {
        Self {
            buffer: Vec::new(),
            current_byte: 0,
            bit_pos: 0,
        }
    }
    #[inline]
    fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current_byte |= 1 << self.bit_pos;
        }
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.buffer.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 0;
        }
    }
}

trait Bw {
    fn bw_new() -> Self;
    fn bw_write_bits(&mut self, value: u64, n_bits: u8);
    fn bw_write_unary(&mut self, q: u64);
    fn bw_pad_to_byte(&mut self);
    fn bw_flush(self) -> Vec<u8>;
}

impl Bw for OldBitWriter {
    fn bw_new() -> Self {
        OldBitWriter::new()
    }
    #[inline]
    fn bw_write_bits(&mut self, value: u64, n_bits: u8) {
        for i in 0..n_bits {
            self.write_bit((value >> i) & 1 != 0);
        }
    }
    #[inline]
    fn bw_write_unary(&mut self, q: u64) {
        for _ in 0..q {
            self.write_bit(true);
        }
        self.write_bit(false);
    }
    fn bw_pad_to_byte(&mut self) {
        if self.bit_pos > 0 {
            self.buffer.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 0;
        }
    }
    fn bw_flush(mut self) -> Vec<u8> {
        self.bw_pad_to_byte();
        self.buffer
    }
}

impl Bw for BitWriter {
    fn bw_new() -> Self {
        BitWriter::new()
    }
    #[inline]
    fn bw_write_bits(&mut self, value: u64, n_bits: u8) {
        self.write_bits(value, n_bits);
    }
    #[inline]
    fn bw_write_unary(&mut self, q: u64) {
        self.write_unary(q);
    }
    fn bw_pad_to_byte(&mut self) {
        self.pad_to_byte();
    }
    fn bw_flush(self) -> Vec<u8> {
        self.flush()
    }
}

/// Pack `shifted` values as blocked Rice codewords (unary quotient + `k`
/// remainder bits, byte-padded per block) using writer `W`. Mirrors the inner
/// loop of `rice_encode`, with a fixed `k` per block to keep both writers on
/// the exact same op stream.
fn rice_pack<W: Bw>(shifted: &[u32], k: u8, block_size: usize) -> Vec<u8> {
    let mut w = W::bw_new();
    for chunk in shifted.chunks(block_size) {
        w.bw_write_bits(k as u64, 8);
        for &s in chunk {
            let q = (s >> k) as u64;
            let r = (s & ((1u32 << k).wrapping_sub(1))) as u64;
            w.bw_write_unary(q);
            if k > 0 {
                w.bw_write_bits(r, k);
            }
        }
        w.bw_pad_to_byte();
    }
    w.bw_flush()
}

/// Generate UMI-like shifted (>= 0) values for the writer micro-bench.
fn gen_shifted(n: usize) -> Vec<u32> {
    let mut state: u64 = 0xCAFE_BABE_1234_5678;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            match state % 100 {
                0..=60 => 0,
                61..=80 => 1,
                81..=90 => 2,
                91..=95 => 3 + (state % 4) as u32,
                _ => 7 + (state % 20) as u32,
            }
        })
        .collect()
}

fn bench_bitwriter_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitwriter_encode");
    // k = 1 is representative of Scx1-routed (median <= 8) count data.
    let k = 1u8;
    for &(n, label) in &[(100_000usize, "100K"), (1_000_000, "1M")] {
        let shifted = gen_shifted(n);
        // Sanity: both writers must agree byte-for-byte.
        assert_eq!(
            rice_pack::<OldBitWriter>(&shifted, k, B_VAL),
            rice_pack::<BitWriter>(&shifted, k, B_VAL),
            "old/new BitWriter output diverged"
        );
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("old_per_bit", label), &shifted, |b, s| {
            b.iter(|| black_box(rice_pack::<OldBitWriter>(black_box(s), k, B_VAL)))
        });
        group.bench_with_input(BenchmarkId::new("new_u64buf", label), &shifted, |b, s| {
            b.iter(|| black_box(rice_pack::<BitWriter>(black_box(s), k, B_VAL)))
        });
    }
    group.finish();
}

fn bench_rice_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("rice_encode");
    for &(n_values, label) in &[(100_000usize, "100K"), (1_000_000, "1M")] {
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
        group.throughput(Throughput::Elements(n_values as u64));
        group.bench_with_input(BenchmarkId::new("umi", label), &values, |b, v| {
            b.iter(|| rice_encode(black_box(v), black_box(B_VAL)).unwrap())
        });
    }
    group.finish();
}

fn bench_encode_shard(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_shard");
    for &(n_rows, avg_nnz, label) in &[(2048, 500, "2048r_500nnz"), (16384, 2000, "16384r_2000nnz")]
    {
        let (indptr, indices, values, _, nnz) = generate_shard(n_rows, avg_nnz, 30000);
        group.throughput(Throughput::Elements(nnz as u64));
        group.bench_with_input(
            BenchmarkId::new("scx1", label),
            &(indptr, indices, values),
            |b, (ip, idx, val)| {
                b.iter(|| {
                    encode_shard(
                        black_box(ip),
                        black_box(idx),
                        black_box(val),
                        CodecId::Scx1,
                        ValueEncoding::Uint8,
                        true,
                    )
                    .unwrap()
                })
            },
        );
    }
    group.finish();
}

/// Generate realistic CSR data for benchmarking.
/// Returns (indptr, indices, values_u8, n_rows, nnz).
fn generate_shard(
    n_rows: usize,
    avg_nnz_per_row: usize,
    n_vars: u32,
) -> (Vec<u64>, Vec<u32>, Vec<u8>, usize, usize) {
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
        let encoded = forbp_encode(&indices, &row_lengths, true).unwrap();

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

    for &(n_values, label) in &[(1000, "1K"), (100_000, "100K"), (1_000_000, "1M")] {
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
        let encoded = rice_encode(&values, B_VAL).unwrap();

        group.bench_with_input(
            BenchmarkId::new("umi", label),
            &(encoded.clone(), n_values),
            |b, (enc, n)| {
                b.iter(|| rice_decode(black_box(enc), black_box(*n), black_box(B_VAL)).unwrap())
            },
        );
    }

    group.finish();
}

fn bench_decode_shard(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_shard");

    for &(n_rows, avg_nnz, label) in &[(2048, 500, "2048r_500nnz"), (16384, 2000, "16384r_2000nnz")]
    {
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

criterion_group!(
    benches,
    bench_forbp_decode,
    bench_rice_decode,
    bench_decode_shard,
    bench_bitwriter_encode,
    bench_rice_encode,
    bench_encode_shard
);
criterion_main!(benches);
