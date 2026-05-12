//! Conformance tests for the runtime-check replacements of the former
//! `debug_assert!` corruption guards in `scx-codec` (P0 #4 of the 2026-05-11
//! code review). Each test hand-crafts a malformed input that would have
//! silently misbehaved in release builds before the patch and asserts a
//! `MalformedInput` error is now returned.
//!
//! Also covers the Float16 stride conformance test from P0 #3.

use scx_codec::bitstream::{BitReader, BitWriter};
use scx_codec::delta_golomb::delta_golomb_encode;
use scx_codec::dispatch::CodecError;
use scx_codec::forbp::forbp_encode;
use scx_codec::rice::{rice_decode, B_VAL};
use scx_codec::shuffle::{byte_shuffle, byte_unshuffle};
use scx_codec::value_encoding::values_to_raw_bytes;
use scx_codec::ValueEncoding;

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
