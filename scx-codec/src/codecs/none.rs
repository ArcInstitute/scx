//! Split out of `dispatch.rs` (ORG-3.7-2). Pure move.

use crate::codec_id::{CodecError, DecodedShard, EncodedShard, EncodedShardRef, ValueEncoding};
use crate::raw::{
    indices_to_le_bytes, le_bytes_to_indices, le_bytes_to_u64, u64_slice_to_le_bytes,
};
use crate::shard_codec::{DecodeBounds, ShardCodec, ShardShape};

/// `CodecId::None` — no compression; the sub-streams are raw little-endian.
pub struct NoneCodec;

impl ShardCodec for NoneCodec {
    fn encode(
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        _value_encoding: ValueEncoding,
        index_dtype_u16: bool,
    ) -> Result<EncodedShard, CodecError> {
        encode_none(indptr, indices, values, index_dtype_u16)
    }

    fn decode(
        encoded: &EncodedShardRef,
        shape: ShardShape,
        _value_encoding: ValueEncoding,
        index_dtype_u16: bool,
        bounds: &DecodeBounds,
    ) -> Result<DecodedShard, CodecError> {
        decode_none_ref(encoded, shape, index_dtype_u16, bounds)
    }

    fn decode_indptr_only(
        indptr_bytes: &[u8],
        n_rows: usize,
        _indptr_max: usize,
    ) -> Result<Vec<u64>, CodecError> {
        // Uncompressed: `le_bytes_to_u64` already requires the exact length,
        // so the cap is implied rather than applied separately.
        le_bytes_to_u64(indptr_bytes, n_rows + 1)
    }
}

pub(crate) fn encode_none(
    indptr: &[u64],
    indices: &[u32],
    values: &[u8],
    index_dtype_u16: bool,
) -> Result<EncodedShard, CodecError> {
    let indptr_bytes = u64_slice_to_le_bytes(indptr);
    let indices_bytes = indices_to_le_bytes(indices, index_dtype_u16)?;
    Ok(EncodedShard {
        indptr_bytes,
        indices_bytes,
        values_bytes: values.to_vec(),
    })
}

fn decode_none_ref(
    encoded: &EncodedShardRef,
    shape: ShardShape,
    index_dtype_u16: bool,
    bounds: &DecodeBounds,
) -> Result<DecodedShard, CodecError> {
    // `bounds` was derived by the driver, which is also what guards the
    // `nnz * elem` multiplication `le_bytes_to_indices` performs internally
    // and the `n_rows + 1` below — both would otherwise be reachable with a
    // hostile header (F-e).
    let indptr = le_bytes_to_u64(encoded.indptr_bytes, shape.n_rows + 1)?;
    let indices = le_bytes_to_indices(encoded.indices_bytes, shape.nnz, index_dtype_u16)?;
    let expected_len = bounds.values_max;
    if encoded.values_bytes.len() != expected_len {
        return Err(CodecError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "values byte length {} != expected {}",
                encoded.values_bytes.len(),
                expected_len
            ),
        )));
    }
    Ok((indptr, indices, encoded.values_bytes.to_vec()))
}
