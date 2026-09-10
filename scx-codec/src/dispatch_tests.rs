//! Unit tests for [`super`] (`dispatch.rs`).
//!
//! Extracted verbatim from the inline `mod tests` per
//! `docs/conventions.md` § Test Organization. Kept as a `#[path]` sibling
//! rather than moved to `tests/` so it retains white-box access to the
//! crate-private codec bodies and guard helpers via `use super::*`.
//!
//! The imports below `use super::*` are the only lines not carried over
//! verbatim: `dispatch.rs` used to import them for its own body, so the inline
//! module inherited them, and the ORG-3.7-2 split moved both owners elsewhere.

use std::io::Cursor;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use super::*;
use crate::codecs::shufdelta::SHUFDELTA_ZSTD_LEVEL;

/// F8: a value buffer whose length is not a multiple of the element width
/// is rejected, not silently truncated to drop the partial element.
#[test]
fn raw_bytes_to_u32_rejects_ragged_input() {
    // 3 bytes is not a multiple of 2 (Uint16) or 4 (Uint32).
    assert!(matches!(
        raw_bytes_to_u32(&[1, 2, 3], ValueEncoding::Uint16),
        Err(CodecError::MalformedInput(_))
    ));
    assert!(matches!(
        raw_bytes_to_u32(&[1, 2, 3], ValueEncoding::Uint32),
        Err(CodecError::MalformedInput(_))
    ));
    // Exact multiples decode fine.
    assert_eq!(
        raw_bytes_to_u32(&[1, 0, 2, 0], ValueEncoding::Uint16).unwrap(),
        vec![1, 2]
    );
    assert_eq!(
        raw_bytes_to_u32(&[5, 0, 0, 0], ValueEncoding::Uint32).unwrap(),
        vec![5]
    );
}

/// An on-disk `u32::MAX` decodes to f32 as exactly 2³² (the conversion
/// rounds up), and the rewrite paths re-encode that decoded f32 under the
/// input's own `Uint32` encoding. So this arm must accept 2³² and saturate
/// it back to `u32::MAX` — a bound that rejects it aborts compact / merge /
/// sort / build_csc on format-valid archives. On the detect path, fresh
/// out-of-range values are kept away from this arm by
/// `detect_value_encoding` rather than by this check — but callers that
/// pass an explicit encoding bypass that, and still saturate; see the
/// comment on the arm itself.
#[test]
fn encode_f32_uint32_preserves_decoded_u32_max() {
    let decoded_max = u32::MAX as f32;
    assert_eq!(decoded_max, (1u128 << 32) as f32, "u32::MAX as f32 IS 2^32");

    let mut buf = Vec::new();
    ValueEncoding::Uint32
        .encode_f32(&mut buf, decoded_max)
        .unwrap();
    assert_eq!(
        buf,
        u32::MAX.to_le_bytes(),
        "u32::MAX must survive re-encode"
    );

    // The largest f32 strictly below 2³² (2³² - 2⁸) is exact either way.
    buf.clear();
    ValueEncoding::Uint32
        .encode_f32(&mut buf, 4_294_967_040.0f32)
        .unwrap();
    assert_eq!(buf, 4_294_967_040u32.to_le_bytes());

    // Genuinely out of range, and NaN, are still refused.
    assert!(ValueEncoding::Uint32
        .encode_f32(&mut Vec::new(), 8_589_934_592.0f32)
        .is_err());
    assert!(ValueEncoding::Uint32.encode_f32_batch(&[f32::NAN]).is_err());
}

/// Every value the batch path writes must be the byte the per-value path would
/// have written, for every encoding — the batch is a speed change and nothing
/// else. The slice deliberately carries each encoding's edge values (`0`, the
/// inclusive upper bound, the largest exact f32 below it) plus a fractional and
/// a subnormal, because a width-specialised cast is where a truncation-vs-round
/// or a saturation difference would show up.
#[test]
fn encode_f32_into_is_byte_identical_to_the_per_value_path() {
    let cases: &[(ValueEncoding, &[f32])] = &[
        (
            ValueEncoding::Uint8,
            &[0.0, 1.0, 7.9, 254.0, 255.0, f32::MIN_POSITIVE],
        ),
        (
            ValueEncoding::Uint16,
            &[0.0, 1.0, 255.5, 65534.0, 65535.0, f32::MIN_POSITIVE],
        ),
        (
            ValueEncoding::Uint32,
            // 2³² is the inclusive bound (a decoded on-disk `u32::MAX`), and
            // 2³² - 2⁸ is the largest exact f32 below it.
            &[0.0, 1.0, 4_294_967_040.0, (1u64 << 32) as f32],
        ),
        (
            ValueEncoding::Float32,
            &[0.0, -1.5, 3.25, f32::MIN_POSITIVE, f32::MAX, f32::NAN],
        ),
        (
            // f16 saturates to inf above 65504 by design, and that is part of
            // the contract the batch path must reproduce rather than reject.
            ValueEncoding::Float16,
            &[0.0, -1.5, 3.25, 65504.0, 70000.0, f32::NAN],
        ),
    ];
    for &(enc, data) in cases {
        let mut want = Vec::new();
        for &v in data {
            enc.encode_f32(&mut want, v)
                .unwrap_or_else(|e| panic!("{enc:?} rejected {v} on the per-value path: {e}"));
        }
        let got = enc.encode_f32_batch(data).unwrap();
        assert_eq!(got, want, "{enc:?} batch bytes differ from per-value bytes");
        assert_eq!(
            got.len(),
            data.len() * enc.byte_width(),
            "{enc:?} wrote the wrong number of bytes"
        );

        // Appending into a non-empty buffer must not disturb what is already
        // there: this is how the rewrite ops accumulate a shard, row by row.
        let mut acc = vec![0xAAu8; 3];
        enc.encode_f32_into(&mut acc, data).unwrap();
        assert_eq!(&acc[..3], &[0xAA; 3], "{enc:?} clobbered the prefix");
        assert_eq!(&acc[3..], &want[..], "{enc:?} appended the wrong bytes");
    }
}

/// An out-of-range value anywhere in the slice must be reported, and must leave
/// the caller's buffer exactly as it was found. Position matters: the fast path
/// tests the whole slice before writing anything, so a bug that wrote the good
/// prefix first would only show up with the bad value in the middle or at the
/// end.
#[test]
fn encode_f32_into_rejects_out_of_range_and_leaves_the_buffer_untouched() {
    let bad: &[(ValueEncoding, f32)] = &[
        (ValueEncoding::Uint8, 256.0),
        (ValueEncoding::Uint8, -1.0),
        (ValueEncoding::Uint8, f32::NAN),
        (ValueEncoding::Uint16, 65536.0),
        (ValueEncoding::Uint16, f32::NAN),
        (ValueEncoding::Uint32, 8_589_934_592.0),
        (ValueEncoding::Uint32, f32::NAN),
    ];
    for &(enc, offender) in bad {
        for pos in 0..3 {
            let mut data = vec![1.0f32, 2.0, 3.0];
            data[pos] = offender;
            let mut acc = vec![0x5Au8; 5];
            let err = enc
                .encode_f32_into(&mut acc, &data)
                .expect_err(&format!("{enc:?} accepted {offender} at index {pos}"));
            assert!(
                format!("{err}").contains("out of range"),
                "{enc:?} error does not name the problem: {err}"
            );
            assert_eq!(
                acc,
                vec![0x5Au8; 5],
                "{enc:?} left bytes in the buffer after rejecting {offender} at index {pos}"
            );
        }
    }
}

/// Build a small CSR matrix for testing.
/// 3 rows, varying nnz:
///   row 0: cols [1, 3]       vals [5, 10]
///   row 1: cols [0, 2, 4]    vals [1, 3, 7]
///   row 2: cols [2]          vals [2]
fn make_test_csr(value_encoding: ValueEncoding) -> (Vec<u64>, Vec<u32>, Vec<u8>, usize, usize) {
    let indptr: Vec<u64> = vec![0, 2, 5, 6];
    let indices: Vec<u32> = vec![1, 3, 0, 2, 4, 2];
    let values_u32: Vec<u32> = vec![5, 10, 1, 3, 7, 2];
    let n_rows = 3;
    let nnz = 6;

    let values_bytes = match value_encoding {
        ValueEncoding::Uint8 => values_u32.iter().map(|&v| v as u8).collect::<Vec<u8>>(),
        ValueEncoding::Uint16 => {
            let mut buf = Vec::new();
            for &v in &values_u32 {
                buf.write_u16::<LittleEndian>(v as u16).unwrap();
            }
            buf
        }
        ValueEncoding::Uint32 => {
            let mut buf = Vec::new();
            for &v in &values_u32 {
                buf.write_u32::<LittleEndian>(v).unwrap();
            }
            buf
        }
        ValueEncoding::Float32 => {
            let mut buf = Vec::new();
            for &v in &values_u32 {
                buf.write_f32::<LittleEndian>(v as f32).unwrap();
            }
            buf
        }
        ValueEncoding::Float16 => {
            // For testing purposes, just use 2 bytes per value
            let mut buf = Vec::new();
            for &v in &values_u32 {
                buf.write_u16::<LittleEndian>(v as u16).unwrap();
            }
            buf
        }
    };

    (indptr, indices, values_bytes, n_rows, nnz)
}

/// Task 6.7: Round-trip through each CodecId × integer ValueEncoding.
#[test]
fn test_roundtrip_all_integer_codecs() {
    let codecs = [
        CodecId::None,
        CodecId::Scx1,
        CodecId::Zstd,
        CodecId::Lz4Shuffle,
        CodecId::ShufDeltaZstd,
    ];
    let encodings = [
        ValueEncoding::Uint8,
        ValueEncoding::Uint16,
        ValueEncoding::Uint32,
    ];

    for &codec in &codecs {
        for &enc in &encodings {
            for &u16_idx in &[true, false] {
                let (indptr, indices, values, n_rows, nnz) = make_test_csr(enc);

                let encoded =
                    encode_shard(&indptr, &indices, &values, codec, enc, u16_idx).unwrap();

                let (dec_indptr, dec_indices, dec_values) =
                    decode_shard(&encoded, codec, enc, n_rows, nnz, u16_idx).unwrap();

                assert_eq!(
                    indptr, dec_indptr,
                    "indptr mismatch: codec={codec:?} enc={enc:?} u16={u16_idx}"
                );
                assert_eq!(
                    indices, dec_indices,
                    "indices mismatch: codec={codec:?} enc={enc:?} u16={u16_idx}"
                );
                assert_eq!(
                    values, dec_values,
                    "values mismatch: codec={codec:?} enc={enc:?} u16={u16_idx}"
                );
            }
        }
    }
}

/// ShufDeltaZstd round-trips float values (zstd-only value path, #142).
#[test]
fn test_shufdelta_float32_roundtrip() {
    for &u16_idx in &[true, false] {
        let (indptr, indices, values, n_rows, nnz) = make_test_csr(ValueEncoding::Float32);
        let encoded = encode_shard(
            &indptr,
            &indices,
            &values,
            CodecId::ShufDeltaZstd,
            ValueEncoding::Float32,
            u16_idx,
        )
        .unwrap();
        let (dec_ip, dec_ix, dec_v) = decode_shard(
            &encoded,
            CodecId::ShufDeltaZstd,
            ValueEncoding::Float32,
            n_rows,
            nnz,
            u16_idx,
        )
        .unwrap();
        assert_eq!(indptr, dec_ip);
        assert_eq!(indices, dec_ix);
        assert_eq!(values, dec_v);
    }
}

/// ShufDeltaZstd `decode_indptr_only` matches the full decode's indptr.
#[test]
fn test_shufdelta_decode_indptr_only() {
    let (indptr, indices, values, n_rows, _nnz) = make_test_csr(ValueEncoding::Uint32);
    let encoded = encode_shard(
        &indptr,
        &indices,
        &values,
        CodecId::ShufDeltaZstd,
        ValueEncoding::Uint32,
        false,
    )
    .unwrap();
    let ip = decode_indptr_only(&encoded.indptr_bytes, CodecId::ShufDeltaZstd, n_rows).unwrap();
    let expected: Vec<i64> = indptr.iter().map(|&v| v as i64).collect();
    assert_eq!(ip, expected);
}

/// A truncated ShufDeltaZstd sub-frame is rejected (bounded-alloc / exact-len
/// guard) rather than panicking in the in-place undelta.
#[test]
fn test_shufdelta_rejects_truncated_frame() {
    let (indptr, indices, values, n_rows, nnz) = make_test_csr(ValueEncoding::Uint32);
    let encoded = encode_shard(
        &indptr,
        &indices,
        &values,
        CodecId::ShufDeltaZstd,
        ValueEncoding::Uint32,
        false,
    )
    .unwrap();
    // Corrupt the indices frame: a valid but too-short zstd frame (empty payload).
    let short_indices = zstd::encode_all(&b""[..], SHUFDELTA_ZSTD_LEVEL).unwrap();
    let bad = EncodedShardRef {
        indptr_bytes: &encoded.indptr_bytes,
        indices_bytes: &short_indices,
        values_bytes: &encoded.values_bytes,
    };
    let res = decode_shard_ref(
        &bad,
        CodecId::ShufDeltaZstd,
        ValueEncoding::Uint32,
        n_rows,
        nnz,
        false,
    );
    assert!(matches!(res, Err(CodecError::MalformedInput(_))));
}

/// Frame a small CSR into `None` row-groups (local-rebased indptr, contiguous
/// indices/values) and return the three sub-streams + spans. Mirrors the
/// writer-side `frame_none_shard` layout so `decode_row_group` can be tested
/// in isolation.
#[allow(clippy::type_complexity)]
fn frame_none_for_test(
    indptr: &[u64],
    indices: &[u32],
    values_u32: &[u32],
    group_rows: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<RowGroupSpan>) {
    let n_rows = indptr.len() - 1;
    let w_i = 2usize; // u16 indices in this fixture
    let w_v = 4usize; // u32 values
    let indices_bytes = indices_to_le_bytes(indices, true).unwrap();
    let mut values_bytes = Vec::new();
    for &v in values_u32 {
        values_bytes.write_u32::<LittleEndian>(v).unwrap();
    }
    let mut indptr_stream = Vec::new();
    let mut spans = Vec::new();
    let mut r0 = 0usize;
    while r0 < n_rows {
        let r1 = (r0 + group_rows).min(n_rows);
        let base = indptr[r0];
        let ip_off = indptr_stream.len();
        for &ip in &indptr[r0..=r1] {
            indptr_stream.write_u64::<LittleEndian>(ip - base).unwrap();
        }
        let nnz_in_block = (indptr[r1] - base) as u32;
        spans.push(RowGroupSpan {
            row_start: r0 as u32,
            n_rows: (r1 - r0) as u16,
            nnz: nnz_in_block,
            indptr: ip_off..indptr_stream.len(),
            indices: (indptr[r0] as usize * w_i)..(indptr[r1] as usize * w_i),
            values: (indptr[r0] as usize * w_v)..(indptr[r1] as usize * w_v),
        });
        r0 = r1;
    }
    (indptr_stream, indices_bytes, values_bytes, spans)
}

#[test]
fn test_decode_row_group_none_parity_and_independence() {
    let indptr: Vec<u64> = vec![0, 2, 5, 6, 8];
    let indices: Vec<u32> = vec![1, 3, 0, 2, 4, 2, 0, 5];
    let values_u32: Vec<u32> = vec![5, 10, 1, 3, 7, 2, 9, 4];
    let (mut ip_stream, ix_stream, vv_stream, spans) =
        frame_none_for_test(&indptr, &indices, &values_u32, 2);
    assert_eq!(spans.len(), 2);

    // Group 0: rows 0..2 → local indptr [0,2,5], indices [1,3,0,2,4], vals [5,10,1,3,7]
    let (g0_ip, g0_ix, g0_v) = decode_row_group(
        CodecId::None,
        &spans[0],
        &ip_stream,
        &ix_stream,
        &vv_stream,
        ValueEncoding::Uint32,
        true,
    )
    .unwrap();
    assert_eq!(g0_ip, vec![0, 2, 5]);
    assert_eq!(g0_ix, vec![1, 3, 0, 2, 4]);
    assert_eq!(
        raw_bytes_to_u32(&g0_v, ValueEncoding::Uint32).unwrap(),
        vec![5, 10, 1, 3, 7]
    );

    // Group 1: rows 2..4 → local indptr [0,1,3], indices [2,0,5], vals [2,9,4]
    let (g1_ip, g1_ix, g1_v) = decode_row_group(
        CodecId::None,
        &spans[1],
        &ip_stream,
        &ix_stream,
        &vv_stream,
        ValueEncoding::Uint32,
        true,
    )
    .unwrap();
    assert_eq!(g1_ip, vec![0, 1, 3]);
    assert_eq!(g1_ix, vec![2, 0, 5]);
    assert_eq!(
        raw_bytes_to_u32(&g1_v, ValueEncoding::Uint32).unwrap(),
        vec![2, 9, 4]
    );

    // Cross-group independence: corrupt group 0's indptr bytes; group 1 still decodes.
    for b in ip_stream[spans[0].indptr.clone()].iter_mut() {
        *b = 0xFF;
    }
    let (g1_ip2, g1_ix2, _) = decode_row_group(
        CodecId::None,
        &spans[1],
        &ip_stream,
        &ix_stream,
        &vv_stream,
        ValueEncoding::Uint32,
        true,
    )
    .unwrap();
    assert_eq!(g1_ip2, vec![0, 1, 3]);
    assert_eq!(g1_ix2, vec![2, 0, 5]);
}

#[test]
fn test_decode_row_group_rejects_oob_span() {
    let span = RowGroupSpan {
        row_start: 0,
        n_rows: 1,
        nnz: 1,
        indptr: 0..16,
        indices: 0..2,
        values: 0..4,
    };
    // Empty streams → range out of bounds → MalformedInput, not a panic.
    let res = decode_row_group(
        CodecId::None,
        &span,
        &[],
        &[],
        &[],
        ValueEncoding::Uint32,
        true,
    );
    assert!(matches!(res, Err(CodecError::MalformedInput(_))));
}

/// Frame a CSR into row-groups the way the writer's `encode_shard_framed`
/// does, but at the codec level: each group is an independent
/// `encode_shard(codec, ..)` over its **local-rebased** indptr, and the
/// three sub-streams are concatenated. Returns (indptr, indices, values,
/// spans) — exactly the layout `decode_row_group` consumes.
#[allow(clippy::type_complexity)]
fn frame_codec_for_test(
    codec: CodecId,
    indptr: &[u64],
    indices: &[u32],
    values_bytes: &[u8],
    venc: ValueEncoding,
    idx16: bool,
    group_rows: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<RowGroupSpan>) {
    let n_rows = indptr.len() - 1;
    let w_v = venc.byte_width();
    let (mut ip_stream, mut ix_stream, mut vv_stream) = (Vec::new(), Vec::new(), Vec::new());
    let mut spans = Vec::new();
    let mut r0 = 0usize;
    while r0 < n_rows {
        let r1 = (r0 + group_rows).min(n_rows);
        let base = indptr[r0];
        let local_indptr: Vec<u64> = (r0..=r1).map(|r| indptr[r] - base).collect();
        let g_indices = &indices[indptr[r0] as usize..indptr[r1] as usize];
        let g_values = &values_bytes[indptr[r0] as usize * w_v..indptr[r1] as usize * w_v];
        let enc = encode_shard(&local_indptr, g_indices, g_values, codec, venc, idx16).unwrap();
        let (ip_off, ix_off, vv_off) = (ip_stream.len(), ix_stream.len(), vv_stream.len());
        ip_stream.extend_from_slice(&enc.indptr_bytes);
        ix_stream.extend_from_slice(&enc.indices_bytes);
        vv_stream.extend_from_slice(&enc.values_bytes);
        spans.push(RowGroupSpan {
            row_start: r0 as u32,
            n_rows: (r1 - r0) as u16,
            nnz: (indptr[r1] - base) as u32,
            indptr: ip_off..ip_stream.len(),
            indices: ix_off..ix_stream.len(),
            values: vv_off..vv_stream.len(),
        });
        r0 = r1;
    }
    (ip_stream, ix_stream, vv_stream, spans)
}

/// Per-group parity + cross-group independence for the **compressed** framed
/// codecs (the `None` case is covered by
/// `test_decode_row_group_none_parity_and_independence`). Each group decodes
/// to its local CSR byte-identically to the source, and corrupting one
/// group's compressed frame leaves the others decodable — the property that
/// makes framed random access safe.
#[test]
fn test_decode_row_group_compressed_parity_and_independence() {
    let indptr: Vec<u64> = vec![0, 2, 5, 6, 8];
    let indices: Vec<u32> = vec![1, 3, 0, 2, 4, 2, 0, 5];
    let values_u32: Vec<u32> = vec![5, 10, 1, 3, 7, 2, 9, 4];
    let mut values_bytes = Vec::new();
    for &v in &values_u32 {
        values_bytes.write_u32::<LittleEndian>(v).unwrap();
    }
    for codec in [CodecId::ShufDeltaZstd, CodecId::Zstd, CodecId::Lz4Shuffle] {
        let (ip, ix, mut vv, spans) = frame_codec_for_test(
            codec,
            &indptr,
            &indices,
            &values_bytes,
            ValueEncoding::Uint32,
            true,
            2,
        );
        assert_eq!(spans.len(), 2, "codec {codec:?}");

        // Group 1 decodes to its local CSR: rows 2..4 → indptr [0,1,3].
        let decode_g1 = |vv: &[u8]| {
            decode_row_group(codec, &spans[1], &ip, &ix, vv, ValueEncoding::Uint32, true).unwrap()
        };
        let (g1_ip, g1_ix, g1_v) = decode_g1(&vv);
        assert_eq!(g1_ip, vec![0, 1, 3], "codec {codec:?}");
        assert_eq!(g1_ix, vec![2, 0, 5], "codec {codec:?}");
        assert_eq!(
            raw_bytes_to_u32(&g1_v, ValueEncoding::Uint32).unwrap(),
            vec![2, 9, 4],
            "codec {codec:?}"
        );

        // Cross-group independence: shred group 0's values frame; group 1
        // still decodes (it only reads its own byte ranges).
        for b in vv[spans[0].values.clone()].iter_mut() {
            *b = 0xFF;
        }
        let (g1_ip2, g1_ix2, _) = decode_g1(&vv);
        assert_eq!(g1_ip2, vec![0, 1, 3], "codec {codec:?} after corruption");
        assert_eq!(g1_ix2, vec![2, 0, 5], "codec {codec:?} after corruption");
    }
}

/// The truncated-frame guard also covers the `indptr` and `values`
/// sub-frames (the existing `test_shufdelta_rejects_truncated_frame` only
/// exercises `indices`).
#[test]
fn test_shufdelta_rejects_truncated_indptr_or_values_frame() {
    let (indptr, indices, values, n_rows, nnz) = make_test_csr(ValueEncoding::Uint32);
    let encoded = encode_shard(
        &indptr,
        &indices,
        &values,
        CodecId::ShufDeltaZstd,
        ValueEncoding::Uint32,
        false,
    )
    .unwrap();
    let short = zstd::encode_all(&b""[..], SHUFDELTA_ZSTD_LEVEL).unwrap();

    // Truncated indptr frame.
    let bad_indptr = EncodedShardRef {
        indptr_bytes: &short,
        indices_bytes: &encoded.indices_bytes,
        values_bytes: &encoded.values_bytes,
    };
    assert!(matches!(
        decode_shard_ref(
            &bad_indptr,
            CodecId::ShufDeltaZstd,
            ValueEncoding::Uint32,
            n_rows,
            nnz,
            false,
        ),
        Err(CodecError::MalformedInput(_))
    ));

    // Truncated values frame.
    let bad_values = EncodedShardRef {
        indptr_bytes: &encoded.indptr_bytes,
        indices_bytes: &encoded.indices_bytes,
        values_bytes: &short,
    };
    assert!(matches!(
        decode_shard_ref(
            &bad_values,
            CodecId::ShufDeltaZstd,
            ValueEncoding::Uint32,
            n_rows,
            nnz,
            false,
        ),
        Err(CodecError::MalformedInput(_))
    ));
}

/// Task 6.8: None codec produces raw LE bytes.
#[test]
fn test_none_produces_raw_bytes() {
    let (indptr, indices, values, _n_rows, _nnz) = make_test_csr(ValueEncoding::Uint32);
    let encoded = encode_shard(
        &indptr,
        &indices,
        &values,
        CodecId::None,
        ValueEncoding::Uint32,
        false,
    )
    .unwrap();

    // indptr: 4 u64 values = 32 bytes
    assert_eq!(encoded.indptr_bytes.len(), 4 * 8);
    // First u64 should be 0
    let mut cursor = Cursor::new(&encoded.indptr_bytes);
    assert_eq!(cursor.read_u64::<LittleEndian>().unwrap(), 0);
    assert_eq!(cursor.read_u64::<LittleEndian>().unwrap(), 2);
    assert_eq!(cursor.read_u64::<LittleEndian>().unwrap(), 5);
    assert_eq!(cursor.read_u64::<LittleEndian>().unwrap(), 6);

    // indices: 6 u32 values = 24 bytes (index_dtype_u16=false)
    assert_eq!(encoded.indices_bytes.len(), 6 * 4);
    let mut cursor = Cursor::new(&encoded.indices_bytes);
    assert_eq!(cursor.read_u32::<LittleEndian>().unwrap(), 1);
    assert_eq!(cursor.read_u32::<LittleEndian>().unwrap(), 3);

    // values: pass-through
    assert_eq!(encoded.values_bytes, values);
}

/// Task 6.9: Scx1 + Float32 returns error.
#[test]
fn test_scx1_float32_error() {
    let (indptr, indices, values, _n_rows, _nnz) = make_test_csr(ValueEncoding::Float32);
    let result = encode_shard(
        &indptr,
        &indices,
        &values,
        CodecId::Scx1,
        ValueEncoding::Float32,
        false,
    );
    assert!(matches!(result, Err(CodecError::FloatWithScx1)));

    // Also test decode path
    let encoded = EncodedShard {
        indptr_bytes: vec![],
        indices_bytes: vec![],
        values_bytes: vec![],
    };
    let result = decode_shard(&encoded, CodecId::Scx1, ValueEncoding::Float32, 3, 6, false);
    assert!(matches!(result, Err(CodecError::FloatWithScx1)));
}

/// Task 6.10: Zstd + Float32 round-trips correctly.
#[test]
fn test_zstd_float32_roundtrip() {
    let (indptr, indices, values, n_rows, nnz) = make_test_csr(ValueEncoding::Float32);
    let encoded = encode_shard(
        &indptr,
        &indices,
        &values,
        CodecId::Zstd,
        ValueEncoding::Float32,
        false,
    )
    .unwrap();
    let (dec_indptr, dec_indices, dec_values) = decode_shard(
        &encoded,
        CodecId::Zstd,
        ValueEncoding::Float32,
        n_rows,
        nnz,
        false,
    )
    .unwrap();

    assert_eq!(indptr, dec_indptr);
    assert_eq!(indices, dec_indices);
    assert_eq!(values, dec_values);
}

/// LZ4Shuffle + Float32 round-trips correctly.
#[test]
fn test_lz4_shuffle_float32_roundtrip() {
    let (indptr, indices, values, n_rows, nnz) = make_test_csr(ValueEncoding::Float32);
    let encoded = encode_shard(
        &indptr,
        &indices,
        &values,
        CodecId::Lz4Shuffle,
        ValueEncoding::Float32,
        false,
    )
    .unwrap();
    let (dec_indptr, dec_indices, dec_values) = decode_shard(
        &encoded,
        CodecId::Lz4Shuffle,
        ValueEncoding::Float32,
        n_rows,
        nnz,
        false,
    )
    .unwrap();

    assert_eq!(indptr, dec_indptr);
    assert_eq!(indices, dec_indices);
    assert_eq!(values, dec_values);
}

/// LZ4Shuffle + Float16 round-trips correctly.
#[test]
fn test_lz4_shuffle_float16_roundtrip() {
    let (indptr, indices, values, n_rows, nnz) = make_test_csr(ValueEncoding::Float16);
    let encoded = encode_shard(
        &indptr,
        &indices,
        &values,
        CodecId::Lz4Shuffle,
        ValueEncoding::Float16,
        false,
    )
    .unwrap();
    let (dec_indptr, dec_indices, dec_values) = decode_shard(
        &encoded,
        CodecId::Lz4Shuffle,
        ValueEncoding::Float16,
        n_rows,
        nnz,
        false,
    )
    .unwrap();

    assert_eq!(indptr, dec_indptr);
    assert_eq!(indices, dec_indices);
    assert_eq!(values, dec_values);
}

/// Test None with u16 indices.
#[test]
fn test_none_u16_indices() {
    let (indptr, indices, values, _n_rows, _nnz) = make_test_csr(ValueEncoding::Uint16);
    let encoded = encode_shard(
        &indptr,
        &indices,
        &values,
        CodecId::None,
        ValueEncoding::Uint16,
        true,
    )
    .unwrap();

    // indices: 6 u16 values = 12 bytes
    assert_eq!(encoded.indices_bytes.len(), 6 * 2);
}

/// Test CodecId and ValueEncoding from_u8 helpers.
#[test]
fn test_from_u8_helpers() {
    assert_eq!(CodecId::from_u8(0), Some(CodecId::None));
    assert_eq!(CodecId::from_u8(1), Some(CodecId::Scx1));
    assert_eq!(CodecId::from_u8(2), Some(CodecId::Zstd));
    assert_eq!(CodecId::from_u8(3), Some(CodecId::Lz4Shuffle));
    assert_eq!(CodecId::from_u8(4), Some(CodecId::Pcodec));
    assert_eq!(CodecId::from_u8(5), Some(CodecId::ShufDeltaZstd));
    assert_eq!(CodecId::from_u8(6), None);

    assert_eq!(ValueEncoding::from_u8(0), Some(ValueEncoding::Uint8));
    assert_eq!(ValueEncoding::from_u8(4), Some(ValueEncoding::Float16));
    assert_eq!(ValueEncoding::from_u8(5), None);

    assert_eq!(ValueEncoding::Uint8.byte_width(), 1);
    assert_eq!(ValueEncoding::Uint16.byte_width(), 2);
    assert_eq!(ValueEncoding::Float32.byte_width(), 4);
    assert!(ValueEncoding::Uint32.is_integer());
    assert!(!ValueEncoding::Float32.is_integer());
}

#[test]
fn test_u32_to_raw_bytes_rejects_overflow() {
    // u8 overflow
    let data = vec![256u32];
    assert!(u32_to_raw_bytes(&data, ValueEncoding::Uint8).is_err());

    // u16 overflow
    let data = vec![65536u32];
    assert!(u32_to_raw_bytes(&data, ValueEncoding::Uint16).is_err());

    // u32 should accept any value
    let data = vec![u32::MAX];
    assert!(u32_to_raw_bytes(&data, ValueEncoding::Uint32).is_ok());
}

#[test]
fn test_indices_to_le_bytes_rejects_overflow() {
    // u16 overflow with index_dtype_u16=true
    let indices = vec![70000u32];
    assert!(indices_to_le_bytes(&indices, true).is_err());

    // Same index with u32 mode should succeed
    assert!(indices_to_le_bytes(&indices, false).is_ok());
}

#[test]
fn test_u64_to_i64_cast_valid() {
    let data = vec![0u64, 100, i64::MAX as u64];
    let result = u64_vec_to_i64(data).unwrap();
    assert_eq!(result, vec![0i64, 100, i64::MAX]);
}

#[test]
fn test_u64_to_i64_rejects_overflow() {
    let data = vec![0u64, 100, u64::MAX];
    match u64_vec_to_i64(data) {
        Err(CodecError::MalformedInput(msg)) => {
            assert!(msg.contains("exceeds i64::MAX"), "got: {msg}");
        }
        other => panic!("expected MalformedInput, got {other:?}"),
    }
}

#[test]
fn test_u32_to_i32_cast_valid() {
    let data = vec![0u32, 100, i32::MAX as u32];
    let result = u32_vec_to_i32_bounded(data, NO_INDEX_BOUND).unwrap();
    assert_eq!(result, vec![0i32, 100, i32::MAX]);
}

#[test]
fn test_u32_to_i32_rejects_overflow() {
    let data = vec![0u32, 100, u32::MAX];
    match u32_vec_to_i32_bounded(data, NO_INDEX_BOUND) {
        Err(CodecError::MalformedInput(msg)) => {
            assert!(msg.contains("exceeds i32::MAX"), "got: {msg}");
        }
        other => panic!("expected MalformedInput, got {other:?}"),
    }
}

/// The bound rides on the same scan as the sign guard, so it must produce a
/// *different* message: "exceeds i32::MAX" would be a lie about an index of
/// 9 in an 8-column shard, and would send whoever reads it looking for an
/// overflow that is not there.
#[test]
fn test_u32_to_i32_rejects_out_of_range_index_with_its_own_message() {
    let data = vec![0u32, 3, 9];
    match u32_vec_to_i32_bounded(data, 8) {
        Err(e @ CodecError::IndexOutOfRange { .. }) => {
            let CodecError::IndexOutOfRange {
                index,
                position,
                bound,
            } = e
            else {
                unreachable!()
            };
            assert_eq!((index, position, bound), (9, 2, 8));
            assert!(
                !e.to_string().contains("i32::MAX"),
                "an in-i32-range index is not an overflow: {e}"
            );
        }
        other => panic!("expected IndexOutOfRange, got {other:?}"),
    }
}

/// `clamp_index_bound` maps the "undeclared" sentinel to the sign-only
/// bound. Old writers stamped the file-level `n_vars` into every shard
/// header, which is 0 on a multimodal file, so treating 0 as a real bound
/// rejects valid files (two multimodal conformance fixtures, specifically).
#[test]
fn test_clamp_index_bound() {
    assert_eq!(clamp_index_bound(0), NO_INDEX_BOUND, "0 means undeclared");
    assert_eq!(clamp_index_bound(8), 8);
    assert_eq!(
        clamp_index_bound(u32::MAX),
        NO_INDEX_BOUND,
        "must not widen past the sign guarantee"
    );
}

#[test]
fn test_values_raw_to_f32_uint8() {
    let raw = vec![0u8, 1, 127, 255];
    let result = values_raw_to_f32(&raw, ValueEncoding::Uint8);
    assert_eq!(result, vec![0.0f32, 1.0, 127.0, 255.0]);
}

#[test]
fn test_values_raw_to_f32_float32_le() {
    let vals = [1.0f32, -2.5, 0.0, f32::MAX];
    let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let result = values_raw_to_f32(&raw, ValueEncoding::Float32);
    assert_eq!(result, vals.to_vec());
}

#[test]
fn test_zstd_decode_bounded_rejects_oversized() {
    // Compress data that's larger than we'll allow
    let raw_data = vec![0u8; 1000];
    let compressed = zstd::encode_all(raw_data.as_slice(), 3).unwrap();

    // Allow only 100 bytes decompressed — should fail
    let result = zstd_decode_bounded(&compressed, 100);
    assert!(result.is_err());

    // Allow 1000 bytes — should succeed
    let result = zstd_decode_bounded(&compressed, 1000);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1000);
}

#[test]
fn test_codec_id_parse_cli() {
    assert_eq!(CodecId::parse_cli("auto").unwrap(), None);
    assert_eq!(CodecId::parse_cli("none").unwrap(), Some(CodecId::None));
    assert_eq!(CodecId::parse_cli("scx1").unwrap(), Some(CodecId::Scx1));
    assert_eq!(CodecId::parse_cli("zstd").unwrap(), Some(CodecId::Zstd));
    assert_eq!(
        CodecId::parse_cli("lz4").unwrap(),
        Some(CodecId::Lz4Shuffle)
    );
    assert_eq!(CodecId::parse_cli("pcodec").unwrap(), Some(CodecId::Pcodec));
    assert!(CodecId::parse_cli("gzip").is_err());
}

#[test]
fn test_codec_id_display_name() {
    assert_eq!(CodecId::None.display_name(), "none");
    assert_eq!(CodecId::Scx1.display_name(), "scx1");
    assert_eq!(CodecId::Zstd.display_name(), "zstd");
    assert_eq!(CodecId::Lz4Shuffle.display_name(), "lz4+shuffle");
    assert_eq!(CodecId::Pcodec.display_name(), "pcodec");
    // Every explicit CLI codec parses back to a value whose display name
    // is stable (round-trip guard so the two maps can't drift).
    for s in ["none", "scx1", "zstd", "pcodec"] {
        let c = CodecId::parse_cli(s).unwrap().unwrap();
        assert_eq!(c.display_name(), s);
    }
}

/// F-e: a hostile shard header carrying an `nnz` that overflows `usize`
/// when scaled by the value/index byte width must return a `MalformedInput`
/// error, not panic (debug) or wrap to a bogus allocation cap (release).
/// Covers `None` (whose `le_bytes_to_indices` computes `nnz * elem`
/// internally), `Zstd` (whose caps are computed up front), and `Scx1`
/// (whose `checked_len` guards were added in F-f).
#[test]
fn decode_rejects_nnz_length_overflow() {
    let encoded = EncodedShardRef {
        indptr_bytes: &[0u8; 16],
        indices_bytes: &[0u8; 4],
        values_bytes: &[0u8; 4],
    };
    for codec in [CodecId::None, CodecId::Zstd, CodecId::Scx1] {
        let err = decode_shard_ref(
            &encoded,
            codec,
            ValueEncoding::Uint32,
            1,          // n_rows
            usize::MAX, // nnz — nnz * width overflows usize
            false,
        )
        .unwrap_err();
        assert!(
            matches!(err, CodecError::MalformedInput(ref m) if m.contains("overflows usize")),
            "codec {codec:?}: expected MalformedInput overflow error, got {err:?}"
        );
    }
}
