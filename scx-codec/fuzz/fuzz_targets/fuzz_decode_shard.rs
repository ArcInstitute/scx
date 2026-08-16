#![no_main]
use libfuzzer_sys::fuzz_target;
use scx_codec::{
    decode_indptr_only, decode_row_group, decode_row_group_indptr_only, decode_shard_native,
    decode_shard_ref, decode_shard_scipy, CodecId, EncodedShardRef, RowGroupSpan, ValueEncoding,
    NO_INDEX_BOUND,
};

// Composite-level fuzzing of the shard decode dispatch.
//
// The five original targets all sat at the *primitive* level (bitstream, rice,
// forbp, delta-golomb), but the structurally-invalid-CSR class lives one level
// up, at the seam that knows `n_rows` and `nnz` and applies the shared guards.
// This target drives that seam across **every** codec rather than one.
//
// It also covers the two entry points that deliberately bypass
// `decode_shard_ref` for Scx1 — `decode_shard_scipy` and `decode_shard_native`
// re-implement the guard prologue to skip a u32 -> bytes -> f32 round trip, so
// they are exactly the paths where a guard can go missing unnoticed.
//
// The first input byte selects the codec so libFuzzer can steer coverage across
// all six from one corpus; the rest is split into three sub-streams. Every call
// must return a `Result` — malformed input yields `Err`, never a panic, and
// never an allocation unbounded by the declared shape.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let codec = match data[0] % 6 {
        0 => CodecId::None,
        1 => CodecId::Scx1,
        2 => CodecId::Zstd,
        3 => CodecId::Lz4Shuffle,
        4 => CodecId::Pcodec,
        _ => CodecId::ShufDeltaZstd,
    };
    let body = &data[1..];
    let n = body.len();
    let a = n / 3;
    let b = 2 * n / 3;
    let shard = EncodedShardRef {
        indptr_bytes: &body[..a],
        indices_bytes: &body[a..b],
        values_bytes: &body[b..],
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
                let _ = decode_shard_ref(&shard, codec, venc, n_rows, nnz, idx16);
                // The Scx1 short-circuits live in these two, not in the above.
                //
                // Both index bounds, because they classify differently and that
                // split was itself a defect once: `NO_INDEX_BOUND` is the
                // sentinel "reject only what reinterprets negative", while a
                // real declared width yields `IndexOutOfRange`. Anything at or
                // above the sentinel clamps back to it, so a large value would
                // silently exercise only the first branch.
                for bound in [NO_INDEX_BOUND, 1000] {
                    let _ = decode_shard_scipy(&shard, codec, venc, n_rows, nnz, idx16, bound);
                }
                let _ = decode_shard_native(&shard, codec, venc, n_rows, nnz, idx16);
            }
        }
    }

    for &n_rows in &[0usize, 1, 10, 1_000_000] {
        let _ = decode_indptr_only(body, codec, n_rows);
    }

    // The row-group seam slices three frames out of the sub-streams and funnels
    // them back through `decode_shard_ref`, relabelling errors with the group.
    //
    // The span endpoints are taken from the input *unclamped*, so libFuzzer can
    // drive them past the stream length or invert them and exercise
    // `slice_span`'s rejection. An earlier version derived them from `len/3`
    // and `2*len/3`, which is always ordered and always in bounds — it reached
    // the entry point but could never fuzz the parser boundary it claimed to.
    let g = |i: usize| body.get(i).copied().unwrap_or(0) as usize;
    let span = RowGroupSpan {
        row_start: 0,
        n_rows: (g(0) as u16) % 8,
        nnz: (g(1) as u32) % 64,
        indptr: g(2)..g(3),
        indices: g(4)..g(5),
        values: g(6)..g(7),
    };
    for venc in [ValueEncoding::Uint8, ValueEncoding::Uint32, ValueEncoding::Float32] {
        for idx16 in [true, false] {
            let _ = decode_row_group(codec, &span, body, body, body, venc, idx16);
        }
    }
    let _ = decode_row_group_indptr_only(codec, &span, body);
});
