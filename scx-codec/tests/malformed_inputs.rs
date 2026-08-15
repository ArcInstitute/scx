//! Conformance tests for the runtime-check replacements of the former
//! `debug_assert!` corruption guards in `scx-codec` (P0 #4 of the 2026-05-11
//! code review). Each test hand-crafts a malformed input that would have
//! silently misbehaved in release builds before the patch and asserts a
//! `MalformedInput` error is now returned.
//!
//! Also covers the Float16 stride conformance test from P0 #3.

use scx_codec::bitstream::{BitReader, BitWriter};
use scx_codec::delta_golomb::{delta_golomb_decode, delta_golomb_encode};
use scx_codec::dispatch::{
    decode_indptr_only, decode_row_group, decode_row_group_indptr_only, decode_shard_native,
    decode_shard_ref, decode_shard_scipy, CodecError, CodecId, EncodedShardRef, RowGroupSpan,
};
use scx_codec::forbp::{forbp_decode_with_hint, forbp_encode};
use scx_codec::rice::{rice_decode, rice_encode, B_VAL};
use scx_codec::shuffle::{byte_shuffle, byte_unshuffle};
use scx_codec::value_encoding::values_to_raw_bytes;
use scx_codec::ValueEncoding;

// ---------------------------------------------------------------------------
// Peak-allocation tracking, for the two tests that assert a decode does not
// size a buffer from an untrusted header field.
//
// Those decodes fail either way and with the same message, so an `is_err()` or
// message assertion cannot tell a bounded implementation from an unbounded one
// — the only observable difference is how much memory it claims on the way to
// the error. Counters are thread-local and libtest gives each test its own
// thread, so a concurrent test cannot pollute the measurement.
// ---------------------------------------------------------------------------

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static LIVE: Cell<usize> = const { Cell::new(0) };
    static PEAK: Cell<usize> = const { Cell::new(0) };
}

struct PeakTracking;

unsafe impl GlobalAlloc for PeakTracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            // `try_with`: TLS may be torn down late in the thread's life, and an
            // allocator must not panic there.
            let _ = LIVE.try_with(|live| {
                let now = live.get() + layout.size();
                live.set(now);
                let _ = PEAK.try_with(|peak| {
                    if now > peak.get() {
                        peak.set(now);
                    }
                });
            });
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _ = LIVE.try_with(|live| live.set(live.get().saturating_sub(layout.size())));
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOC: PeakTracking = PeakTracking;

/// Run `f`, returning its value and the peak bytes live on this thread during it.
fn peak_alloc_bytes<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE.with(|l| l.get());
    PEAK.with(|p| p.set(base));
    let out = f();
    let peak = PEAK.with(|p| p.get());
    (out, peak.saturating_sub(base))
}

// ---------------------------------------------------------------------------
// P0 #3: Float16 stride conformance
// ---------------------------------------------------------------------------

#[test]
fn float16_emits_2_byte_le_per_element() {
    let data = [0.0_f32, 1.5, -2.25, 100.0, 1e-3];
    let bytes = values_to_raw_bytes(&data, ValueEncoding::Float16).unwrap();

    // 2 bytes per element, not 4.
    assert_eq!(bytes.len(), data.len() * 2);

    // Each 2-byte LE pair is a valid f16 bit pattern produced by from_f32.
    for (i, chunk) in bytes.chunks_exact(2).enumerate() {
        let expected = half::f16::from_f32(data[i]);
        let got = half::f16::from_le_bytes([chunk[0], chunk[1]]);
        assert_eq!(got.to_bits(), expected.to_bits());
    }
}

// ---------------------------------------------------------------------------
// P0 #4: shuffle.rs — ragged length rejection
// ---------------------------------------------------------------------------

#[test]
fn byte_shuffle_rejects_ragged_input() {
    // 7 bytes is not divisible by element_width 4.
    let input: Vec<u8> = vec![0; 7];
    assert!(matches!(
        byte_shuffle(&input, 4),
        Err(CodecError::MalformedInput(_))
    ));
}

#[test]
fn byte_unshuffle_rejects_ragged_input() {
    // Decoder-facing direction — most safety-critical.
    let input: Vec<u8> = vec![0; 5];
    assert!(matches!(
        byte_unshuffle(&input, 2),
        Err(CodecError::MalformedInput(_))
    ));
}

// ---------------------------------------------------------------------------
// P0 #4: rice.rs — non-zero block-header high nibble rejection
// ---------------------------------------------------------------------------

#[test]
fn rice_rejects_nonzero_high_nibble() {
    // Hand-craft a single byte where the high nibble is 0x1 (reserved
    // must be zero per spec). Low nibble k=0 is fine.
    let bad_header = vec![0x10u8];
    match rice_decode(&bad_header, 1, B_VAL) {
        Err(CodecError::MalformedInput(msg)) => {
            assert!(msg.contains("high nibble"), "got: {msg}");
        }
        other => panic!("expected MalformedInput, got {other:?}"),
    }
}

#[test]
fn rice_accepts_zero_high_nibble() {
    // Sanity: build a valid 1-value Rice stream with k=0 and confirm it
    // decodes — to make sure the new check didn't break the happy path.
    let mut bw = BitWriter::new();
    // Header byte: k = 0 (low nibble 0, high nibble 0).
    bw.write_bits(0, 8);
    // Encode value 1 (zero-shifted), q=0 → unary(0) = "0".
    bw.write_unary(0);
    let _ = BitReader::new(&[]); // ensure trait usable
    let bytes = bw.flush();

    let decoded = rice_decode(&bytes, 1, B_VAL).expect("valid stream should decode");
    assert_eq!(decoded, vec![1]);
}

// ---------------------------------------------------------------------------
// P0 #4: forbp.rs — unsorted indices rejection
// ---------------------------------------------------------------------------

#[test]
fn forbp_rejects_unsorted_row_indices() {
    // Single row with descending indices — would wrap on u32 subtraction
    // in release before this patch.
    let indices = vec![3u32, 1, 5];
    let row_lengths = vec![3usize];

    match forbp_encode(&indices, &row_lengths, false) {
        Err(CodecError::MalformedInput(msg)) => {
            assert!(msg.contains("sorted"), "got: {msg}");
        }
        other => panic!("expected MalformedInput, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// P0 #4: delta_golomb.rs — non-monotone indptr rejection
// ---------------------------------------------------------------------------

#[test]
fn delta_golomb_rejects_non_monotone_indptr() {
    let indptr = vec![0u64, 5, 3, 10];
    match delta_golomb_encode(&indptr) {
        Err(CodecError::MalformedInput(msg)) => {
            assert!(msg.contains("monotone"), "got: {msg}");
        }
        other => panic!("expected MalformedInput, got {other:?}"),
    }
}

#[test]
fn delta_golomb_accepts_equal_indptr_windows() {
    // Equal consecutive values mean zero-delta rows — should be allowed.
    let indptr = vec![0u64, 5, 5, 10];
    let encoded = delta_golomb_encode(&indptr).expect("equal deltas are valid");
    assert!(!encoded.is_empty());
}

// ---------------------------------------------------------------------------
// F-f: hostile-capacity rejection — a shard header declaring far more elements
// than the compressed sub-stream could physically produce must return `Err`
// (not eagerly allocate GBs / `capacity overflow`-panic) before any allocation.
// ---------------------------------------------------------------------------

#[test]
fn test_decode_hostile_capacity_rice() {
    // 10 bytes can encode at most 80 elements (1 bit/element); 1M is rejected.
    match rice_decode(&[0u8; 10], 1_000_000, B_VAL) {
        Err(CodecError::MalformedInput(msg)) => {
            assert!(msg.contains("rice values"), "got: {msg}");
        }
        other => panic!("expected MalformedInput, got {other:?}"),
    }
}

#[test]
fn test_decode_hostile_capacity_delta_golomb() {
    // Returns the primitive's own `BitStreamError` (not a CodecError); the
    // Scx1 decode path produces the `MalformedInput` message upstream.
    assert!(delta_golomb_decode(&[0u8; 10], 1_000_000).is_err());
}

#[test]
fn test_decode_hostile_capacity_forbp() {
    // nnz_hint = 1M against a 10-byte input → rejected before allocation.
    assert!(forbp_decode_with_hint(&[0u8; 10], 1, 1_000_000, false).is_err());
    // n_rows = 1M against a 10-byte input → rejected by the tighter
    // `n_rows > data.len()` bound (each row writes ≥1 varint byte).
    assert!(forbp_decode_with_hint(&[0u8; 10], 1_000_000, 0, false).is_err());
}

#[test]
fn test_decode_hostile_capacity_shard_ref_scx1() {
    // A crafted Scx1 shard header with nnz = 2^32 - 1 and a tiny payload must
    // return MalformedInput without requesting a multi-GiB allocation.
    let encoded = EncodedShardRef {
        indptr_bytes: &[0u8; 16],
        indices_bytes: &[0u8; 8],
        values_bytes: &[0u8; 8],
    };
    let err = decode_shard_ref(
        &encoded,
        CodecId::Scx1,
        ValueEncoding::Uint32,
        1,                 // n_rows
        u32::MAX as usize, // nnz = 2^32 - 1
        false,
    )
    .unwrap_err();
    assert!(
        matches!(err, CodecError::MalformedInput(_)),
        "expected MalformedInput, got {err:?}"
    );
}

#[test]
fn test_decode_capacity_boundary_passes() {
    // n_values exactly at the bound (input_len * 8) must NOT be rejected by the
    // capacity guard. Decode may still fail on actual stream content, but the
    // error must not be the `bound_capacity` "declared … elements" message.
    let data = [0u8; 4];
    let n = data.len() * 8; // 32 — exactly at the bound
    match rice_decode(&data, n, B_VAL) {
        Ok(_) => {}
        Err(CodecError::MalformedInput(msg)) => {
            assert!(
                !msg.contains("max") || !msg.contains("at 1 bit/element"),
                "boundary must not trip the capacity guard, got: {msg}"
            );
        }
        Err(_) => {} // any other downstream failure is fine
    }
}

// ---------------------------------------------------------------------------
// Bounded decompression: Lz4Shuffle and Pcodec
//
// Every other codec caps how many bytes a sub-stream may decompress to before
// it decompresses (`zstd_decode_bounded`, keyed on `indptr_byte_cap` /
// `checked_len`). `Lz4Shuffle` did not: `lz4_frame_decompress` was a plain
// `read_to_end` into an unbounded `Vec`, so a small shard could force an
// arbitrarily large allocation, and the declared-length check only ran
// afterwards — on a buffer that had already been materialised.
//
// This is production-reachable: `select_codec_for_modality` picks
// `Lz4Shuffle` for non-binary integer ATAC counts, so every Multiome /
// TEA-seq ATAC modality writes shards that decode through this path.
//
// The assertions key on the error *message*, not `is_err()`: an unbounded
// decode also errors, just one full allocation too late, so `is_err()` passes
// against the broken code.
// ---------------------------------------------------------------------------

/// Stream-compress `n_bytes` of zeros into an LZ4 frame without ever holding
/// `n_bytes` in memory. Returns a small frame that decompresses to `n_bytes`.
fn lz4_bomb(n_bytes: usize) -> Vec<u8> {
    use std::io::Write;
    const CHUNK: usize = 64 * 1024;
    let zeros = [0u8; CHUNK];
    let mut enc = lz4_flex::frame::FrameEncoder::new(Vec::new());
    let mut written = 0;
    while written < n_bytes {
        let n = CHUNK.min(n_bytes - written);
        enc.write_all(&zeros[..n]).unwrap();
        written += n;
    }
    enc.finish().unwrap()
}

/// 64 MiB of zeros, which LZ4 stores in a few hundred KiB at most. Large
/// enough that decoding it unbounded is unmistakable, small enough that the
/// test stays fast if the guard regresses.
const BOMB_BYTES: usize = 64 * 1024 * 1024;

#[test]
fn test_lz4_indptr_bomb_is_capped_before_decompression() {
    let bomb = lz4_bomb(BOMB_BYTES);
    assert!(
        bomb.len() < 1 << 20,
        "bomb should be small on the wire, got {} bytes",
        bomb.len()
    );

    // `indptr` is the first sub-stream decompressed. A 2-row shard bounds it
    // to 24 bytes, so 64 MiB must be refused, not decompressed and measured.
    // The other two streams are valid frames so that this test can only fail
    // on the indptr cap — a garbage sibling stream would mask it.
    let indices_raw: Vec<u8> = [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
    let indices = lz4_roundtrip_compress(&byte_shuffle(&indices_raw, 4).unwrap());
    let values = lz4_roundtrip_compress(&byte_shuffle(&[0u8; 8], 4).unwrap());
    let encoded = EncodedShardRef {
        indptr_bytes: &bomb,
        indices_bytes: &indices,
        values_bytes: &values,
    };
    let err = decode_shard_ref(
        &encoded,
        CodecId::Lz4Shuffle,
        ValueEncoding::Uint32,
        2, // n_rows
        2, // nnz
        false,
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("exceeds limit"),
        "expected the decompression cap to fire before allocating, got: {msg}"
    );
}

#[test]
fn test_lz4_values_bomb_is_capped_before_decompression() {
    let bomb = lz4_bomb(BOMB_BYTES);
    // A valid 2-row/2-nnz indptr and indices, so the bomb in `values` is what
    // the decoder trips on rather than an earlier sub-stream.
    let indptr_raw: Vec<u8> = [0u64, 1, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
    let indices_raw: Vec<u8> = [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
    let indptr = lz4_roundtrip_compress(&byte_shuffle(&indptr_raw, 8).unwrap());
    let indices = lz4_roundtrip_compress(&byte_shuffle(&indices_raw, 4).unwrap());

    let encoded = EncodedShardRef {
        indptr_bytes: &indptr,
        indices_bytes: &indices,
        values_bytes: &bomb,
    };
    let err = decode_shard_ref(
        &encoded,
        CodecId::Lz4Shuffle,
        ValueEncoding::Uint32,
        2, // n_rows
        2, // nnz → values bounded to 8 bytes
        false,
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("exceeds limit"),
        "expected the decompression cap to fire before allocating, got: {msg}"
    );
}

/// Compress with the same LZ4 frame settings the encoder uses.
fn lz4_roundtrip_compress(data: &[u8]) -> Vec<u8> {
    use std::io::Write;
    let mut enc = lz4_flex::frame::FrameEncoder::new(Vec::new());
    enc.write_all(data).unwrap();
    enc.finish().unwrap()
}

#[test]
fn test_lz4_hostile_shape_errors_instead_of_panicking() {
    // `n_rows` chosen so `(n_rows + 1) * 8` overflows `usize`. The five sibling
    // codecs reach `indptr_byte_cap`, whose checked arithmetic returns an
    // error; Lz4Shuffle skipped it and hit the unchecked `count * 8` inside
    // `le_bytes_to_u64`. `scx-codec` sets `overflow-checks = true` in release,
    // so that was a panic on malformed input in every profile — which the
    // reader convention forbids.
    let encoded = EncodedShardRef {
        indptr_bytes: &lz4_roundtrip_compress(&[0u8; 32]),
        indices_bytes: &lz4_roundtrip_compress(&[0u8; 8]),
        values_bytes: &lz4_roundtrip_compress(&[0u8; 8]),
    };
    let err = decode_shard_ref(
        &encoded,
        CodecId::Lz4Shuffle,
        ValueEncoding::Uint32,
        usize::MAX / 4, // n_rows
        2,
        false,
    )
    .unwrap_err();
    assert!(
        matches!(err, CodecError::MalformedInput(_)),
        "expected MalformedInput, got {err:?}"
    );
}

#[test]
fn test_lz4_indptr_only_bomb_is_capped() {
    // `decode_indptr_only`'s Lz4Shuffle arm carried the same hole as the full
    // decoder and is reached independently (indptr-only reads).
    let bomb = lz4_bomb(BOMB_BYTES);
    let err = decode_indptr_only(&bomb, CodecId::Lz4Shuffle, 2).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("exceeds limit"),
        "expected the decompression cap to fire before allocating, got: {msg}"
    );
}

#[test]
fn test_pcodec_float_values_are_bounded_before_decompression() {
    // Pcodec's float arms called `pco::standalone::simple_decompress` with no
    // bound and checked the length only afterwards. Build a genuine pcodec
    // stream of many floats, then declare a tiny `nnz`.
    const N: usize = 1 << 20; // 1Mi floats → 4 MiB decoded
    let floats = vec![0.0f32; N];
    let values = pco::standalone::simple_compress(&floats, &pco::ChunkConfig::default()).unwrap();

    let indptr_raw: Vec<u8> = [0u64, 1, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
    let indices_raw: Vec<u8> = [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
    let indptr = zstd::encode_all(indptr_raw.as_slice(), 3).unwrap();
    let indices = zstd::encode_all(indices_raw.as_slice(), 3).unwrap();

    let encoded = EncodedShardRef {
        indptr_bytes: &indptr,
        indices_bytes: &indices,
        values_bytes: &values,
    };
    let err = decode_shard_ref(
        &encoded,
        CodecId::Pcodec,
        ValueEncoding::Float32,
        2, // n_rows
        2, // nnz → 2 floats declared, 1Mi present
        false,
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("exceeds limit"),
        "expected the element cap to fire before decompressing, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// The CSR shape invariant at the decode boundary
//
// A decoded shard is three arrays that must agree, and they are produced by
// three independent sub-stream decoders: `delta_golomb_decode` and
// `rice_decode` return exactly the count the *caller* asks for, while FOR-BP
// returns whatever its own per-row nnz varints say. Nothing used to compare
// them, so a corrupt Scx1 shard decoded to a structurally invalid CSR whose
// `indices` were shorter than `indptr.last()` — and `ScxCsr::new_unchecked`
// only `debug_asserts` the difference, so in release the consumers index past
// the end of `indices` instead of erroring.
// ---------------------------------------------------------------------------

/// Build an Scx1 shard whose header declares `nnz = 6` for one row while the
/// FOR-BP index stream encodes only 3 indices. Every sub-stream is produced by
/// the real encoder — the shard is malformed only in that the three disagree.
fn scx1_shard_with_short_index_stream() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let indptr_bytes = delta_golomb_encode(&[0, 6]).unwrap();
    let indices_bytes = forbp_encode(&[1, 2, 3], &[3], false).unwrap();
    let values_bytes = rice_encode(&[1, 1, 1, 1, 1, 1], B_VAL).unwrap();
    (indptr_bytes, indices_bytes, values_bytes)
}

#[test]
fn forbp_hint_shorter_than_declared_nnz_is_rejected() {
    // The primitive where the divergence originates. `nnz_hint` used to be
    // advisory ("for pre-allocation"), so a stream declaring 3 indices decoded
    // happily against a caller that asked for 6. Every production caller passes
    // the exact nnz, so the hint is a contract.
    let encoded = forbp_encode(&[1, 2, 3], &[3], false).unwrap();
    assert!(forbp_decode_with_hint(&encoded, 1, 3, false).is_ok());
    assert!(
        forbp_decode_with_hint(&encoded, 1, 6, false).is_err(),
        "3-index stream accepted against a declared nnz of 6"
    );
}

/// Build an Scx1 shard that is self-consistent everywhere the *primitives* can
/// see — the FOR-BP stream really does hold 3 indices and Rice really does hold
/// 3 values, both matching the declared `nnz = 3` — but whose indptr claims the
/// single row ends at 2. Only a check that compares the arrays *to each other*
/// catches it, so this fixture pins `check_decoded_shape` specifically, where
/// [`scx1_shard_with_short_index_stream`] is caught one layer down by FOR-BP's
/// own contract.
fn scx1_shard_with_disagreeing_indptr() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let indptr_bytes = delta_golomb_encode(&[0, 2]).unwrap();
    let indices_bytes = forbp_encode(&[1, 2, 3], &[3], false).unwrap();
    let values_bytes = rice_encode(&[1, 1, 1], B_VAL).unwrap();
    (indptr_bytes, indices_bytes, values_bytes)
}

#[test]
fn scx1_shard_shorter_than_declared_nnz_is_rejected() {
    // The review's reproducer verbatim: pre-fix, `decode_shard_scipy` returned
    // `Ok(([0, 6], [1, 2, 3], [1.0; 6]))` — an indptr claiming six non-zeros
    // over an `indices` of three. Caught now by FOR-BP's declared-nnz contract,
    // so the error kind is `BitStream`, not `MalformedInput`; the shard-level
    // `check_decoded_shape` is the second line of defence behind it (pinned by
    // `scx1_shard_indptr_disagreeing_with_nnz_is_rejected`).
    let (indptr_bytes, indices_bytes, values_bytes) = scx1_shard_with_short_index_stream();
    let encoded = EncodedShardRef {
        indptr_bytes: &indptr_bytes,
        indices_bytes: &indices_bytes,
        values_bytes: &values_bytes,
    };

    assert!(
        decode_shard_scipy(
            &encoded,
            CodecId::Scx1,
            ValueEncoding::Uint32,
            1,
            6,
            false,
            100
        )
        .is_err(),
        "decode_shard_scipy accepted a 3-index stream against a declared nnz of 6"
    );
    assert!(
        decode_shard_native(&encoded, CodecId::Scx1, ValueEncoding::Uint32, 1, 6, false).is_err(),
        "decode_shard_native accepted a 3-index stream against a declared nnz of 6"
    );
    assert!(
        decode_shard_ref(&encoded, CodecId::Scx1, ValueEncoding::Uint32, 1, 6, false).is_err(),
        "decode_shard_ref accepted a 3-index stream against a declared nnz of 6"
    );
}

#[test]
fn scx1_shard_indptr_disagreeing_with_nnz_is_rejected() {
    // Pins the shard-level shape gate on all three entry points.
    // `decode_shard_scipy` / `_native` short-circuit Scx1 and never reach
    // `decode_shard_ref`, so each carries its own call and each needs its own
    // assertion — the review's reproducer was on the scipy path.
    let (indptr_bytes, indices_bytes, values_bytes) = scx1_shard_with_disagreeing_indptr();
    let encoded = EncodedShardRef {
        indptr_bytes: &indptr_bytes,
        indices_bytes: &indices_bytes,
        values_bytes: &values_bytes,
    };

    let err = decode_shard_scipy(
        &encoded,
        CodecId::Scx1,
        ValueEncoding::Uint32,
        1,
        3,
        false,
        100,
    )
    .unwrap_err();
    assert!(
        matches!(&err, CodecError::MalformedInput(msg) if msg.contains("indptr ends at")),
        "expected an indptr/nnz MalformedInput, got {err:?}"
    );

    // `ShardValuesNative` is not `Debug`, so match rather than `unwrap_err`.
    match decode_shard_native(&encoded, CodecId::Scx1, ValueEncoding::Uint32, 1, 3, false) {
        Err(CodecError::MalformedInput(msg)) => assert!(msg.contains("indptr ends at"), "{msg}"),
        Err(other) => panic!("expected MalformedInput, got {other:?}"),
        Ok((indptr, indices, _)) => panic!(
            "accepted a malformed shard: indptr={indptr:?}, indices.len()={}",
            indices.len()
        ),
    }

    let err =
        decode_shard_ref(&encoded, CodecId::Scx1, ValueEncoding::Uint32, 1, 3, false).unwrap_err();
    assert!(
        matches!(&err, CodecError::MalformedInput(msg) if msg.contains("indptr ends at")),
        "expected an indptr/nnz MalformedInput, got {err:?}"
    );
}

#[test]
fn row_group_shape_mismatch_is_rejected() {
    // The framed (v2 shard) path. `decode_row_group` used to hand-roll exactly
    // this check (`indptr[0] == 0`, `indptr.last() == nnz`) and nothing else;
    // it now inherits the full gate from `decode_shard_ref`, and must still
    // name the group in the message.
    let (indptr_bytes, indices_bytes, values_bytes) = scx1_shard_with_disagreeing_indptr();
    let span = RowGroupSpan {
        row_start: 7,
        n_rows: 1,
        nnz: 3,
        indptr: 0..indptr_bytes.len(),
        indices: 0..indices_bytes.len(),
        values: 0..values_bytes.len(),
    };
    let err = decode_row_group(
        CodecId::Scx1,
        &span,
        &indptr_bytes,
        &indices_bytes,
        &values_bytes,
        ValueEncoding::Uint32,
        false,
    )
    .unwrap_err();
    match &err {
        CodecError::MalformedInput(msg) => {
            assert!(msg.contains("row-group at row 7"), "{msg}");
            assert!(msg.contains("indptr ends at"), "{msg}");
        }
        other => panic!("expected MalformedInput, got {other:?}"),
    }
}

#[test]
fn none_shard_indptr_last_not_nnz_is_rejected() {
    // The indptr half of the same invariant. Under `CodecId::None` the indptr
    // is raw LE bytes, so it can say anything: here it claims the single row
    // ends at 5 while the header declares nnz = 2. `csr_to_csc` walks
    // `indptr[row]..indptr[row + 1]` and would index past `indices`.
    let indptr: Vec<u8> = [0u64, 5].iter().flat_map(|v| v.to_le_bytes()).collect();
    let indices: Vec<u8> = [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
    let values: Vec<u8> = [7u32, 9].iter().flat_map(|v| v.to_le_bytes()).collect();
    let encoded = EncodedShardRef {
        indptr_bytes: &indptr,
        indices_bytes: &indices,
        values_bytes: &values,
    };
    let err =
        decode_shard_ref(&encoded, CodecId::None, ValueEncoding::Uint32, 1, 2, false).unwrap_err();
    assert!(
        matches!(err, CodecError::MalformedInput(_)),
        "expected MalformedInput, got {err:?}"
    );
}

#[test]
fn none_shard_non_monotone_indptr_is_rejected() {
    // `indptr.last() == nnz` alone is not enough: an interior entry can still
    // exceed `indices.len()` while the last one is honest, which is the same
    // out-of-bounds walk one row earlier.
    let indptr: Vec<u8> = [0u64, 5, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
    let indices: Vec<u8> = [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
    let values: Vec<u8> = [7u32, 9].iter().flat_map(|v| v.to_le_bytes()).collect();
    let encoded = EncodedShardRef {
        indptr_bytes: &indptr,
        indices_bytes: &indices,
        values_bytes: &values,
    };
    let err =
        decode_shard_ref(&encoded, CodecId::None, ValueEncoding::Uint32, 2, 2, false).unwrap_err();
    assert!(
        matches!(err, CodecError::MalformedInput(_)),
        "expected MalformedInput, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// The indptr-only decode seam
//
// `decode_indptr_only` / `decode_row_group_indptr_only` are what every
// direct-to-device GPU decoder builds its `GpuCsr` indptr from — they never
// reach `decode_shard_ref`, so they never saw `check_decoded_shape`. Both
// checked at most `first == 0` and `last == nnz`, which is not enough: an
// interior entry can exceed nnz while the last one is honest, and a consumer
// walking `indptr[row]..indptr[row + 1]` then reads past the end of `indices`
// one row early. That is the same defect `none_shard_non_monotone_indptr_is_rejected`
// pins on the whole-shard path.
// ---------------------------------------------------------------------------

#[test]
fn indptr_only_rejects_non_monotone() {
    // `[0, 5, 2]` over 2 rows: ends at 2, starts at 0, right length — every
    // endpoint check passes and row 0 still addresses [0, 5) of a 2-element
    // index buffer.
    let raw: Vec<u8> = [0u64, 5, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
    let err = decode_indptr_only(&raw, CodecId::None, 2).unwrap_err();
    assert!(
        matches!(&err, CodecError::MalformedInput(msg) if msg.contains("not monotone")),
        "expected a monotonicity MalformedInput, got {err:?}"
    );
}

#[test]
fn indptr_only_rejects_non_zero_start() {
    // Scx1's Delta-Golomb encodes non-negative deltas, so its output is always
    // monotone — but it reads the *first* value as a raw LE u64, so the start is
    // unconstrained by the codec and has to be checked.
    let raw: Vec<u8> = [3u64, 5].iter().flat_map(|v| v.to_le_bytes()).collect();
    let err = decode_indptr_only(&raw, CodecId::None, 1).unwrap_err();
    assert!(
        matches!(&err, CodecError::MalformedInput(msg) if msg.contains("must start at 0")),
        "expected a zero-start MalformedInput, got {err:?}"
    );
}

#[test]
fn indptr_only_accepts_a_well_formed_indptr() {
    // Over-rejection guard: the shape gate must not reject valid data.
    let raw: Vec<u8> = [0u64, 2, 2, 7]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(
        decode_indptr_only(&raw, CodecId::None, 3).unwrap(),
        vec![0i64, 2, 2, 7]
    );
}

#[test]
fn row_group_indptr_only_rejects_non_monotone() {
    // The seam every framed GPU assembler goes through: `prescan_framed_group_indptr`
    // calls this once per group and concatenates the results into the combined
    // indptr it uploads.
    let raw: Vec<u8> = [0u64, 5, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
    let span = RowGroupSpan {
        row_start: 4,
        n_rows: 2,
        nnz: 2,
        indptr: 0..raw.len(),
        indices: 0..0,
        values: 0..0,
    };
    let err = decode_row_group_indptr_only(CodecId::None, &span, &raw).unwrap_err();
    match &err {
        CodecError::MalformedInput(msg) => assert!(msg.contains("not monotone"), "{msg}"),
        other => panic!("expected MalformedInput, got {other:?}"),
    }
}

#[test]
fn row_group_names_the_group_for_a_short_forbp_stream() {
    // The corruption this PR targets fails inside `forbp_decode_with_hint`,
    // whose error carries no message and arrives as `CodecError::BitStream`.
    // Relabelling only `MalformedInput` left the primary case anonymous.
    let (indptr_bytes, indices_bytes, values_bytes) = scx1_shard_with_short_index_stream();
    let span = RowGroupSpan {
        row_start: 9,
        n_rows: 1,
        nnz: 6,
        indptr: 0..indptr_bytes.len(),
        indices: 0..indices_bytes.len(),
        values: 0..values_bytes.len(),
    };
    let err = decode_row_group(
        CodecId::Scx1,
        &span,
        &indptr_bytes,
        &indices_bytes,
        &values_bytes,
        ValueEncoding::Uint32,
        false,
    )
    .unwrap_err();
    assert!(
        format!("{err}").contains("row-group at row 9"),
        "error does not name the failing group: {err}"
    );
}

// ---------------------------------------------------------------------------
// Regression: the Pcodec float arms must decode what this crate encodes.
//
// The bounded-decode work above briefly guarded `pcodec_decompress_bounded`
// with `bound_capacity`, whose "at least 1 bit per output value" floor is a
// property of the Scx1 Golomb/Rice codes and **not** of pcodec, which is an
// entropy coder. Low-entropy float payloads — a constant layer, or raw counts
// stored as `f32`, which is the standard AnnData layout — compress below that
// floor, so the guard rejected shards `encode_shard` had just produced.
//
// These are round trips through the public API, at nnz large enough to leave
// the region where pcodec's per-chunk overhead keeps the ratio above 1
// bit/value. The pre-existing dispatch proptests cannot see this: `arb_csr`
// caps nnz near 1000, which stays above the floor.
// ---------------------------------------------------------------------------

/// Build a CSR shard: `n_rows` rows of `per_row` entries, values from `f`.
fn pcodec_shard(
    n_rows: usize,
    per_row: usize,
    encoding: ValueEncoding,
    f: impl Fn(usize) -> f32,
) -> (Vec<u64>, Vec<u32>, Vec<u8>, usize) {
    let nnz = n_rows * per_row;
    let indptr: Vec<u64> = (0..=n_rows).map(|r| (r * per_row) as u64).collect();
    let indices: Vec<u32> = (0..nnz).map(|i| (i % per_row) as u32).collect();
    let floats: Vec<f32> = (0..nnz).map(&f).collect();
    let values = values_to_raw_bytes(&floats, encoding).unwrap();
    (indptr, indices, values, nnz)
}

fn assert_pcodec_roundtrips(label: &str, encoding: ValueEncoding, f: impl Fn(usize) -> f32) {
    // 500 rows × 200 = 100_000 nnz — a realistic shard size, and far enough
    // past pcodec's chunk overhead that low-entropy data lands under 1 bit/value.
    let (indptr, indices, values, nnz) = pcodec_shard(500, 200, encoding, f);
    let encoded = scx_codec::dispatch::encode_shard(
        &indptr,
        &indices,
        &values,
        CodecId::Pcodec,
        encoding,
        false,
    )
    .unwrap();

    // Premise: this payload really is compressed below the 1-bit-per-value
    // floor. If it were not, the test would pass for the wrong reason.
    let bits_per_value = (encoded.values_bytes.len() as f64 * 8.0) / nnz as f64;
    assert!(
        bits_per_value < 1.0,
        "{label}: premise failed — {bits_per_value:.3} bits/value is above the \
         1-bit floor, so this case would not exercise the regression"
    );

    let decoded = scx_codec::dispatch::decode_shard(
        &encoded,
        CodecId::Pcodec,
        encoding,
        500,
        nnz,
        false,
    )
    .unwrap_or_else(|e| {
        panic!("{label}: encode→decode round trip failed at {bits_per_value:.3} bits/value: {e}")
    });
    assert_eq!(decoded.0.len(), 501, "{label}: indptr length");
    assert_eq!(decoded.1.len(), nnz, "{label}: indices length");
    assert_eq!(
        decoded.2.len(),
        nnz * encoding.byte_width(),
        "{label}: values byte length"
    );
    assert_eq!(decoded.2, values, "{label}: values must round-trip exactly");
}

#[test]
fn test_pcodec_low_entropy_float_roundtrip() {
    // A constant layer: pcodec reaches ~0 bits/value.
    assert_pcodec_roundtrips("constant f32", ValueEncoding::Float32, |_| 1.0);

    // Raw counts stored as f32 — mostly 1.0, some 2.0, few 3/4. This is what
    // an AnnData `X` of raw scRNA-seq counts looks like, and it measures
    // ~0.96 bits/value: under the floor, and not a degenerate input.
    assert_pcodec_roundtrips("raw counts as f32", ValueEncoding::Float32, |i| {
        match (i * 2_654_435_761) % 100 {
            0..=79 => 1.0,
            80..=93 => 2.0,
            94..=97 => 3.0,
            _ => 4.0,
        }
    });
}

#[test]
fn test_pcodec_low_entropy_float16_roundtrip() {
    // The Float16 arm decodes through the same pcodec helper (as f32, then
    // narrowed), so it carried the same false rejection.
    assert_pcodec_roundtrips("constant f16", ValueEncoding::Float16, |_| 1.0);
}

/// The inverse of `test_pcodec_float_values_are_bounded_before_decompression`:
/// a *large declared* `nnz` against a *tiny* pcodec stream.
///
/// Sizing the float destination from the header made this a second allocation
/// as large as the indices bomb that let it through. `le_bytes_to_indices`
/// wants an exact `nnz * width` bytes, and Zstd expands a few KiB of zero
/// indices into exactly that — so "the indices stream corroborates `nnz`" is
/// not a bound, it is the same untrusted number arriving by a longer route.
///
/// The decode fails either way, with the same message, so the assertion is on
/// **allocation**: the values arm must not claim another `nnz * 4`. The indices
/// buffer itself is pre-existing and shared with every other codec, so it is
/// measured as the baseline rather than treated as a defect here.
#[test]
fn test_pcodec_large_declared_nnz_does_not_preallocate_from_the_header() {
    const NNZ: usize = 16 * 1024 * 1024;
    const FLOAT_DST_BYTES: usize = NNZ * 4;

    let indptr_raw: Vec<u8> = [0u64, NNZ as u64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let indptr = zstd::encode_all(indptr_raw.as_slice(), 3).unwrap();
    let indices = zstd::encode_all(vec![0u8; NNZ * 4].as_slice(), 3).unwrap();
    assert!(
        indices.len() < 64 * 1024,
        "premise: the indices bomb must be tiny, got {} B",
        indices.len()
    );

    // A genuine, tiny pcodec stream carrying two values.
    let values =
        pco::standalone::simple_compress(&[1.0f32, 2.0], &pco::ChunkConfig::default()).unwrap();

    // Baseline: the same shard decoded as an *integer* Pcodec shard. Integer
    // values take the Zstd arm, so this measures everything except the float
    // destination — indices bomb, growth transients and all.
    let (_, baseline) = peak_alloc_bytes(|| {
        let encoded = EncodedShardRef {
            indptr_bytes: &indptr,
            indices_bytes: &indices,
            values_bytes: &values,
        };
        decode_shard_ref(
            &encoded,
            CodecId::Pcodec,
            ValueEncoding::Uint32,
            1,
            NNZ,
            false,
        )
    });

    let (res, peak) = peak_alloc_bytes(|| {
        let encoded = EncodedShardRef {
            indptr_bytes: &indptr,
            indices_bytes: &indices,
            values_bytes: &values,
        };
        decode_shard_ref(
            &encoded,
            CodecId::Pcodec,
            ValueEncoding::Float32,
            1,
            NNZ,
            false,
        )
    });

    let err = res.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("pcodec value count 2") && msg.contains(&NNZ.to_string()),
        "expected the count mismatch after a stream-sized decode, got: {msg}"
    );

    // Half the float destination is a wide margin: a header-sized `vec![0f32;
    // nnz]` adds a full FLOAT_DST_BYTES over the baseline, while growing from
    // the stream adds the fixed slab (256 KiB) plus two decoded values.
    let budget = baseline + FLOAT_DST_BYTES / 2;
    assert!(
        peak <= budget,
        "the float arm allocated from the untrusted header: peak {peak} B vs \
         baseline {baseline} B (+{} B). A stream-sized decode should add well \
         under {} B, not approach the {FLOAT_DST_BYTES} B destination the \
         header declared.",
        peak.saturating_sub(baseline),
        FLOAT_DST_BYTES / 2
    );
}
