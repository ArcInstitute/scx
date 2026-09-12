//! Shared test utilities for scx-gpu tests and benchmarks.

use scx_codec::{encode_shard, CodecId, ValueEncoding};
use scx_format_io::shard::{ShardHeader, SHARD_HEADER_SIZE};

/// Build a complete (unframed, shard v1) shard byte buffer from raw CSR arrays.
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
    buf
}

/// Build a deterministic CSR with the given per-row nnz counts.
///
/// Columns are strictly increasing within each row (FOR-BP requires sorted
/// indices). Returns `(indptr, indices, values_u16)`.
pub fn build_dense_csr(row_nnzs: &[usize], n_cols: u32) -> (Vec<u64>, Vec<u32>, Vec<u16>) {
    let mut indptr = vec![0u64];
    let mut indices: Vec<u32> = Vec::new();
    let mut values_u16: Vec<u16> = Vec::new();
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;
    for &nnz_row in row_nnzs {
        let mut col = 0u32;
        for _ in 0..nnz_row {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            col += (state % 4 + 1) as u32; // strictly increasing
            assert!(col < n_cols, "test column overflow — widen n_cols");
            indices.push(col);
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            values_u16.push((state % 7 + 1) as u16);
        }
        indptr.push(indices.len() as u64);
    }
    (indptr, indices, values_u16)
}

/// Assemble a full shard-section byte buffer (header + payload + block index)
/// from an [`scx_format_io::encode_one_shard`] result, mirroring the on-disk
/// layout the writer produces — a **framed (shard v2)** shard.
///
/// Asserts the framing actually engaged, so a caller cannot vacuously test the
/// framed decode path against an unframed shard.
pub fn build_framed_test_shard(
    indptr: &[u64],
    indices: &[u32],
    values_f32: &[f32],
    explicit_codec: CodecId,
    n_cols: u32,
    row_group_rows: u32,
) -> Vec<u8> {
    use scx_format_io::section::SectionType;
    let index_dtype = if n_cols <= 65535 { 0u8 } else { 1u8 };
    let framing = scx_format_io::FramingConfig {
        row_group_rows,
        target_nnz: None,
        trial: false,
        decode_target: None,
    };
    let mut enc_opts = scx_format_io::EncodeShardOptions::new(
        "X_shard_0".to_string(),
        SectionType::CsrShard,
        n_cols as u64,
        0,
        index_dtype,
    );
    enc_opts.explicit_codec = Some(explicit_codec);
    enc_opts.framing = Some(framing);
    let section = scx_format_io::encode_one_shard(indptr, indices, values_f32, &enc_opts)
        .expect("encode_one_shard (framed) failed");
    assert_eq!(
        section.shard_format_version(),
        2,
        "expected framed (v2) shard for codec {explicit_codec:?}"
    );
    let mut buf = section.header_buf.clone();
    buf.extend_from_slice(&section.encoded.indptr_bytes);
    buf.extend_from_slice(&section.encoded.indices_bytes);
    buf.extend_from_slice(&section.encoded.values_bytes);
    buf.extend_from_slice(&section.block_index_bytes);
    buf
}
