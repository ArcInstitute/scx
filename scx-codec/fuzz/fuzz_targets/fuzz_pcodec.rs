#![no_main]
use libfuzzer_sys::fuzz_target;
use scx_codec::{decode_shard_ref, CodecId, EncodedShardRef, ValueEncoding};

// Fuzz the public Pcodec decode path. The integer arms zstd-decompress the
// values; the float arms go through pcodec, which sizes its output from the
// stream rather than the declared shape unless bounded. Every call must return
// a `Result` — malformed input yields `Err`, never a panic or an unbounded
// allocation.
fuzz_target!(|data: &[u8]| {
    let n = data.len();
    let a = n / 3;
    let b = 2 * n / 3;
    let shard = EncodedShardRef {
        indptr_bytes: &data[..a],
        indices_bytes: &data[a..b],
        values_bytes: &data[b..],
    };

    for venc in [
        ValueEncoding::Uint8,
        ValueEncoding::Uint16,
        ValueEncoding::Uint32,
        ValueEncoding::Float32,
        ValueEncoding::Float16,
    ] {
        for idx16 in [true, false] {
            for &(n_rows, nnz) in &[(0usize, 0usize), (1, 4), (10, 100)] {
                let _ = decode_shard_ref(&shard, CodecId::Pcodec, venc, n_rows, nnz, idx16);
            }
        }
    }
});
