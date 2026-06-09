//! Shared test utilities for scx-gpu tests and benchmarks.

use scx_codec::{encode_shard, CodecId, ValueEncoding};
use scx_format::shard::{ShardHeader, SHARD_HEADER_SIZE};

/// Build a complete shard byte buffer from raw CSR arrays.
///
/// Encodes with the specified codec, builds a ShardHeader with correct
/// offsets, and serializes header + encoded data into a single buffer.
pub fn build_test_shard(
    indptr: &[u64],
    indices: &[u32],
    values_raw: &[u8],
    codec_id: CodecId,
    value_encoding: ValueEncoding,
    n_cols: u32,
) -> Vec<u8> {
    build_test_shard_with_metadata(
        indptr,
        indices,
        values_raw,
        codec_id,
        value_encoding,
        n_cols,
    )
    .0
}

/// Like [`build_test_shard`], but also returns the encoder-emitted Scx1 decode
/// metadata ([`scx_codec::Scx1DecodeMetadata`] from `EncodedShard::scx1_decode`)
/// so GPU sidecar-driven decode paths can be tested for byte-parity against the
/// CPU-prescan path. The metadata is `None` for non-Scx1 codecs.
pub fn build_test_shard_with_metadata(
    indptr: &[u64],
    indices: &[u32],
    values_raw: &[u8],
    codec_id: CodecId,
    value_encoding: ValueEncoding,
    n_cols: u32,
) -> (Vec<u8>, Option<scx_codec::Scx1DecodeMetadata>) {
    let n_rows = (indptr.len() - 1) as u32;
    let nnz = *indptr.last().unwrap();
    let index_dtype_u16 = n_cols <= 65535;

    let encoded = encode_shard(
        indptr,
        indices,
        values_raw,
        codec_id,
        value_encoding,
        index_dtype_u16,
    )
    .expect("encode_shard failed");

    let indptr_rel_offset = SHARD_HEADER_SIZE as u32;
    let indices_rel_offset = indptr_rel_offset + encoded.indptr_bytes.len() as u32;
    let values_rel_offset = indices_rel_offset + encoded.indices_bytes.len() as u32;

    let header = ShardHeader {
        magic: *b"SCXS",
        shard_format_version: 1,
        shard_type: 0,
        codec_id: codec_id as u8,
        value_encoding: value_encoding as u8,
        index_dtype: if index_dtype_u16 { 0 } else { 1 },
        reserved_flags: [0; 3],
        n_major: n_rows,
        n_minor: n_cols,
        nnz,
        global_offset: 0,
        indptr_rel_offset,
        indptr_length: encoded.indptr_bytes.len() as u32,
        indices_rel_offset,
        indices_length: encoded.indices_bytes.len() as u32,
        values_rel_offset,
        values_length: encoded.values_bytes.len() as u32,
        block_index_rel_offset: 0,
        block_index_length: 0,
        checksum: [0; 8],
    };

    let mut buf = Vec::new();
    header.write_to(&mut buf).expect("write header");
    buf.extend_from_slice(&encoded.indptr_bytes);
    buf.extend_from_slice(&encoded.indices_bytes);
    buf.extend_from_slice(&encoded.values_bytes);
    (buf, encoded.scx1_decode)
}
