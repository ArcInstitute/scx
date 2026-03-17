//! Conformance test vectors for SCX codecs (Task 17.6).
//!
//! These tests encode known inputs and assert the exact output bytes,
//! serving as the authoritative reference for codec implementations.

use scx_codec::bitstream::{BitReader, BitWriter};
use scx_codec::delta_golomb::{delta_golomb_decode, delta_golomb_encode};
use scx_codec::forbp::{forbp_decode, forbp_encode};
use scx_codec::rice::{rice_decode, rice_encode, B_VAL};
use scx_codec::{decode_shard, encode_shard, CodecId, ValueEncoding};

// =========================================================================
// Rice codec reference vectors
// =========================================================================

#[test]
fn rice_ref_all_ones_4() {
    // [1,1,1,1] → shifted [0,0,0,0], median=0, k=0
    // Header byte: 0x00 (k=0)
    // Each value: unary(0) = single 0-bit → 4 zero bits
    // Body: 0000 padded to byte = 0x00
    let values = vec![1u32; 4];
    let encoded = rice_encode(&values, B_VAL);
    assert_eq!(encoded, vec![0x00, 0x00]);
    let decoded = rice_decode(&encoded, 4, B_VAL).unwrap();
    assert_eq!(decoded, values);
}

#[test]
fn rice_ref_one_two() {
    // [1, 2] → shifted [0, 1], median=0, k=0
    // Header: 0x00
    // val 0: shifted=0, q=0, unary(0) = 0-bit
    // val 1: shifted=1, q=1, unary(1) = 1,0  (LSB-first: bits are 1 then 0)
    // Bits in LSB-first order: 0 | 1 0 = 010 → padded = 0b00000010 = 0x02
    let values = vec![1u32, 2];
    let encoded = rice_encode(&values, B_VAL);
    assert_eq!(encoded, vec![0x00, 0x02]);
    let decoded = rice_decode(&encoded, 2, B_VAL).unwrap();
    assert_eq!(decoded, values);
}

#[test]
fn rice_ref_larger_k() {
    // [4, 5, 6, 7] → shifted [3, 4, 5, 6], median=floor_median([3,4,5,6])=4
    // k = max(0, floor(log2(0.6931 * 4))) = max(0, floor(log2(2.7724))) = max(0, 1) = 1
    // Header byte: 0x01 (k=1)
    //
    // val 3: q=3>>1=1, r=3&1=1. unary(1)=1,0. bits(1,k=1)=1.  → 1 0 1
    // val 4: q=4>>1=2, r=4&1=0. unary(2)=1,1,0. bits(0,k=1)=0. → 1 1 0 0
    // val 5: q=5>>1=2, r=5&1=1. unary(2)=1,1,0. bits(1,k=1)=1. → 1 1 0 1
    // val 6: q=6>>1=3, r=6&1=0. unary(3)=1,1,1,0. bits(0,k=1)=0. → 1 1 1 0 0
    //
    // All bits LSB-first after header byte:
    // 1 0 1 | 1 1 0 0 | 1 1 0 1 | 1 1 1 0 0
    // = 19 bits
    // Byte 0 (header): 0x01
    // Byte 1: bits 0-7: 1 0 1 1 1 0 0 1 = LSB-first = 0b10011101 = 0x9D
    // Byte 2: bits 8-15: 1 0 1 1 1 1 0 0 = LSB-first = 0b00111101 = 0x3D
    // Byte 3: bits 16-18: 0 + padding = 0b00000000 = 0x00
    let values = vec![4u32, 5, 6, 7];
    let encoded = rice_encode(&values, B_VAL);
    // 19 bits → 3 bytes (header + 2 body bytes with padding in last byte)
    assert_eq!(encoded, vec![0x01, 0x9D, 0x3D]);
    let decoded = rice_decode(&encoded, 4, B_VAL).unwrap();
    assert_eq!(decoded, values);
}

// =========================================================================
// FOR-BP codec reference vectors
// =========================================================================

#[test]
fn forbp_ref_single_row_u16() {
    // Single row with indices [0, 5, 10]
    // Block header: block_nnz=3 (u32 LE), n_rows=1 (u16 LE)
    // Varint nnz: 3 (single byte 0x03)
    // frame_min=0 (u16 LE = 0x00, 0x00), frame_bits=bits_needed(max_delta)
    // deltas: [0, 5, 5], max_delta=5, frame_bits=3
    // Bit-packed deltas (3 bits each, LSB-first): 000 101 101 = 9 bits → 2 bytes
    let indices = vec![0u32, 5, 10];
    let row_lengths = vec![3usize];
    let encoded = forbp_encode(&indices, &row_lengths, true);

    // Verify round-trip
    let (dec_idx, dec_rl) = forbp_decode(&encoded, 1, true).unwrap();
    assert_eq!(dec_idx, indices);
    assert_eq!(dec_rl, row_lengths);

    // Verify exact header bytes
    assert_eq!(encoded[0..4], [0x03, 0x00, 0x00, 0x00]); // block_nnz = 3
    assert_eq!(encoded[4..6], [0x01, 0x00]); // n_rows = 1
    assert_eq!(encoded[6], 0x03); // varint nnz = 3
    assert_eq!(encoded[7..9], [0x00, 0x00]); // frame_min = 0 (u16)
    assert_eq!(encoded[9], 3); // frame_bits = 3
}

#[test]
fn forbp_ref_empty_row() {
    // One empty row followed by one row with data
    let indices = vec![42u32];
    let row_lengths = vec![0usize, 1];
    let encoded = forbp_encode(&indices, &row_lengths, true);
    let (dec_idx, dec_rl) = forbp_decode(&encoded, 2, true).unwrap();
    assert_eq!(dec_idx, indices);
    assert_eq!(dec_rl, row_lengths);

    // block_nnz = 1
    assert_eq!(encoded[0..4], [0x01, 0x00, 0x00, 0x00]);
    // n_rows = 2
    assert_eq!(encoded[4..6], [0x02, 0x00]);
    // varint for row 0: 0 (empty)
    assert_eq!(encoded[6], 0x00);
    // varint for row 1: 1
    assert_eq!(encoded[7], 0x01);
    // frame_min = 42 (u16 LE)
    assert_eq!(encoded[8..10], [42, 0x00]);
    // frame_bits = 0 (single index, delta=[0], max_delta=0)
    assert_eq!(encoded[10], 0);
}

// =========================================================================
// Delta-Golomb codec reference vectors
// =========================================================================

#[test]
fn delta_golomb_ref_basic() {
    // indptr = [0, 150, 280, 500]
    // deltas = [150, 130, 220], median=150, k=compute_k(150)
    // k = max(0, floor(log2(0.6931 * 150))) = max(0, floor(log2(103.965))) = floor(6.70) = 6
    let indptr = vec![0u64, 150, 280, 500];
    let encoded = delta_golomb_encode(&indptr);

    // First 8 bytes: raw LE u64 = 0
    assert_eq!(encoded[0..8], [0, 0, 0, 0, 0, 0, 0, 0]);
    // Byte 8: k value
    assert_eq!(encoded[8], 6);

    let decoded = delta_golomb_decode(&encoded, 4).unwrap();
    assert_eq!(decoded, indptr);
}

#[test]
fn delta_golomb_ref_single_entry() {
    // indptr = [42] → just raw u64, no deltas
    let indptr = vec![42u64];
    let encoded = delta_golomb_encode(&indptr);
    assert_eq!(encoded.len(), 8);
    assert_eq!(encoded, [42, 0, 0, 0, 0, 0, 0, 0]);
    let decoded = delta_golomb_decode(&encoded, 1).unwrap();
    assert_eq!(decoded, indptr);
}

#[test]
fn delta_golomb_ref_zero_deltas() {
    // indptr = [0, 0, 0] → deltas = [0, 0], median=0, k=0
    // Each delta: unary(0)=0-bit, no remainder
    let indptr = vec![0u64, 0, 0];
    let encoded = delta_golomb_encode(&indptr);
    assert_eq!(encoded[0..8], [0, 0, 0, 0, 0, 0, 0, 0]); // first value
    assert_eq!(encoded[8], 0); // k=0
                               // Two unary(0) = two 0-bits → padded to 1 byte = 0x00
    assert_eq!(encoded[9], 0x00);
    let decoded = delta_golomb_decode(&encoded, 3).unwrap();
    assert_eq!(decoded, indptr);
}

// =========================================================================
// Full shard reference vector (CodecId::None)
// =========================================================================

#[test]
fn shard_ref_none_u8_u16() {
    // 3 rows × 5 cols, u16 indices, u8 values
    // row 0: col 1 = 5, col 3 = 10
    // row 1: col 0 = 1, col 2 = 3, col 4 = 7
    // row 2: col 2 = 2
    let indptr: Vec<u64> = vec![0, 2, 5, 6];
    let indices: Vec<u32> = vec![1, 3, 0, 2, 4, 2];
    let values_raw: Vec<u8> = vec![5, 10, 1, 3, 7, 2]; // u8 encoding

    let encoded = encode_shard(
        &indptr,
        &indices,
        &values_raw,
        CodecId::None,
        ValueEncoding::Uint8,
        true, // u16 indices
    )
    .unwrap();

    // indptr: 4 × u64 LE = 32 bytes
    assert_eq!(encoded.indptr_bytes.len(), 32);
    assert_eq!(&encoded.indptr_bytes[0..8], &[0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(&encoded.indptr_bytes[8..16], &[2, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(&encoded.indptr_bytes[16..24], &[5, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(&encoded.indptr_bytes[24..32], &[6, 0, 0, 0, 0, 0, 0, 0]);

    // indices: 6 × u16 LE = 12 bytes
    assert_eq!(encoded.indices_bytes.len(), 12);
    assert_eq!(
        encoded.indices_bytes,
        vec![1, 0, 3, 0, 0, 0, 2, 0, 4, 0, 2, 0]
    );

    // values: 6 × u8 = 6 bytes (pass-through)
    assert_eq!(encoded.values_bytes, values_raw);

    // Decode round-trip
    let (dec_indptr, dec_indices, dec_values) =
        decode_shard(&encoded, CodecId::None, ValueEncoding::Uint8, 3, 6, true).unwrap();
    assert_eq!(dec_indptr, indptr);
    assert_eq!(dec_indices, indices);
    assert_eq!(dec_values, values_raw);
}

// =========================================================================
// Full shard reference vector (CodecId::Scx1)
// =========================================================================

#[test]
fn shard_ref_scx1_roundtrip() {
    // Verify Scx1 codec round-trips exactly for the same 3×5 matrix
    let indptr: Vec<u64> = vec![0, 2, 5, 6];
    let indices: Vec<u32> = vec![1, 3, 0, 2, 4, 2];
    let values_raw: Vec<u8> = vec![5, 10, 1, 3, 7, 2]; // u8 encoding

    let encoded = encode_shard(
        &indptr,
        &indices,
        &values_raw,
        CodecId::Scx1,
        ValueEncoding::Uint8,
        true,
    )
    .unwrap();

    // Verify encoded sizes are smaller or comparable to raw
    // (for this tiny matrix they may not be smaller, but should round-trip)
    let (dec_indptr, dec_indices, dec_values) =
        decode_shard(&encoded, CodecId::Scx1, ValueEncoding::Uint8, 3, 6, true).unwrap();
    assert_eq!(dec_indptr, indptr);
    assert_eq!(dec_indices, indices);
    assert_eq!(dec_values, values_raw);
}

// =========================================================================
// Bitstream reference vectors
// =========================================================================

#[test]
fn bitstream_ref_lsb_first() {
    // Write value 0b1101 as 4 bits LSB-first
    // Expected byte: bit0=1, bit1=0, bit2=1, bit3=1, rest=0 → 0b00001101 = 0x0D
    let mut writer = BitWriter::new();
    writer.write_bits(0b1101, 4);
    let data = writer.flush();
    assert_eq!(data, vec![0x0D]);

    let mut reader = BitReader::new(&data);
    assert_eq!(reader.read_bits(4).unwrap(), 0b1101);
}

#[test]
fn bitstream_ref_unary_three() {
    // unary(3) = three 1-bits followed by one 0-bit: 1110
    // LSB-first in byte: bit0=1, bit1=1, bit2=1, bit3=0 → 0b00000111 = 0x07
    let mut writer = BitWriter::new();
    writer.write_unary(3);
    let data = writer.flush();
    assert_eq!(data, vec![0x07]);

    let mut reader = BitReader::new(&data);
    assert_eq!(reader.read_unary().unwrap(), 3);
}
