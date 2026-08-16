//! Split out of `dispatch.rs` (ORG-3.7-2). Pure move.

use crate::codec_id::{CodecError, DecodedShard, EncodedShard, EncodedShardRef, ValueEncoding};
use crate::shard_codec::{DecodeBounds, ShardCodec, ShardShape};

/// `CodecId::Zstd` — each sub-stream zstd-compressed, no pre-filter.
pub struct ZstdCodec;

impl ShardCodec for ZstdCodec {
    fn encode(
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        value_encoding: ValueEncoding,
        index_dtype_u16: bool,
    ) -> Result<EncodedShard, CodecError> {
        encode_zstd(indptr, indices, values, value_encoding, index_dtype_u16)
    }

    fn decode(
        encoded: &EncodedShardRef,
        shape: ShardShape,
        value_encoding: ValueEncoding,
        index_dtype_u16: bool,
        bounds: &DecodeBounds,
    ) -> Result<DecodedShard, CodecError> {
        decode_zstd_ref(encoded, shape, value_encoding, index_dtype_u16, bounds)
    }

    fn decode_indptr_only(
        indptr_bytes: &[u8],
        n_rows: usize,
        indptr_max: usize,
    ) -> Result<Vec<u64>, CodecError> {
        let raw = zstd_decode_bounded(indptr_bytes, indptr_max)?;
        le_bytes_to_u64(&raw, n_rows + 1)
    }
}

use crate::raw::{
    indices_to_le_bytes, le_bytes_to_indices, le_bytes_to_u64, u64_slice_to_le_bytes,
};

fn encode_zstd(
    indptr: &[u64],
    indices: &[u32],
    values: &[u8],
    _value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<EncodedShard, CodecError> {
    let indptr_raw = u64_slice_to_le_bytes(indptr);
    let indices_raw = indices_to_le_bytes(indices, index_dtype_u16)?;

    let indptr_bytes = zstd::encode_all(indptr_raw.as_slice(), 3)?;
    let indices_bytes = zstd::encode_all(indices_raw.as_slice(), 3)?;
    let values_bytes = zstd::encode_all(values, 3)?;

    Ok(EncodedShard {
        indptr_bytes,
        indices_bytes,
        values_bytes,
    })
}

/// Decompress Zstd data with an upper bound on decompressed size.
///
/// Public so the GPU decode path (`scx-gpu`) can zstd-decompress a single
/// ShufDeltaZstd sub-stream frame to its intermediate "still shuffled+delta'd"
/// plane bytes and upload those to the device, where the undelta/unshuffle/
/// convert transforms run as kernels. Re-exported as
/// [`crate::zstd_decompress_bounded`]. `max_bytes` is the exact expected
/// decompressed length; callers should treat a shorter result as malformed
/// input (see the `expect_exact_len` checks in the framed decoders).
pub fn zstd_decode_bounded(data: &[u8], max_bytes: usize) -> Result<Vec<u8>, CodecError> {
    use std::io::Read;
    let decoder = zstd::Decoder::new(data)?;
    // Cap initial allocation to avoid huge alloc from untrusted max_bytes
    let mut output = Vec::with_capacity(max_bytes.min(1 << 20));
    // `saturating_add` for the same reason as `lz4_frame_decompress`: unreachable
    // today, but a `+ 1` here would be a panic rather than an error under this
    // crate's release `overflow-checks`.
    let mut limited = decoder.take((max_bytes as u64).saturating_add(1));
    limited.read_to_end(&mut output)?;
    if output.len() > max_bytes {
        return Err(CodecError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "decompressed size {} exceeds limit {}",
                output.len(),
                max_bytes
            ),
        )));
    }
    Ok(output)
}

fn decode_zstd_ref(
    encoded: &EncodedShardRef,
    shape: ShardShape,
    _value_encoding: ValueEncoding,
    index_dtype_u16: bool,
    bounds: &DecodeBounds,
) -> Result<DecodedShard, CodecError> {
    let (n_rows, nnz) = (shape.n_rows, shape.nnz);
    let DecodeBounds {
        indptr_max,
        indices_max,
        values_max,
    } = *bounds;

    let indptr_raw = zstd_decode_bounded(encoded.indptr_bytes, indptr_max)?;
    let indices_raw = zstd_decode_bounded(encoded.indices_bytes, indices_max)?;
    let values_raw = zstd_decode_bounded(encoded.values_bytes, values_max)?;

    let indptr = le_bytes_to_u64(&indptr_raw, n_rows + 1)?;
    let indices = le_bytes_to_indices(&indices_raw, nnz, index_dtype_u16)?;

    let expected_len = values_max;
    if values_raw.len() != expected_len {
        return Err(CodecError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "decompressed values byte length {} != expected {}",
                values_raw.len(),
                expected_len
            ),
        )));
    }

    Ok((indptr, indices, values_raw))
}
