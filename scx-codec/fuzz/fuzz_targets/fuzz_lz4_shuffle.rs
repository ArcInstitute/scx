#![no_main]
use libfuzzer_sys::fuzz_target;
use scx_codec::{decode_indptr_only, decode_shard_ref, CodecId, EncodedShardRef, ValueEncoding};

// Fuzz the public Lz4Shuffle decode path (LZ4 frame decompress → byte-unshuffle,
// with the bounded-alloc / exact-length guards). Arbitrary bytes are split into
// three sub-streams (indptr / indices / values) and decoded under several
// value-encoding, index-dtype, and shape combinations. Every call must return a
// `Result` — malformed input yields `Err`, never a panic or an unbounded
// allocation. This codec is production-selected: `select_codec_for_modality`
// picks it for non-binary integer ATAC counts.
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
                let _ = decode_shard_ref(&shard, CodecId::Lz4Shuffle, venc, n_rows, nnz, idx16);
            }
        }
    }

    // The indptr-only seam carries its own copy of the cap.
    for &n_rows in &[0usize, 1, 10, 1_000_000] {
        let _ = decode_indptr_only(data, CodecId::Lz4Shuffle, n_rows);
    }
});
