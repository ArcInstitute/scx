//! Split out of `dispatch.rs` (ORG-3.7-2). Pure move.

use crate::codec_id::{CodecError, DecodedShard, EncodedShard, EncodedShardRef, ValueEncoding};
use crate::guards::checked_len;
use crate::raw::{
    indices_to_le_bytes, le_bytes_to_indices, le_bytes_to_u64, u64_slice_to_le_bytes,
};

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

pub(crate) fn decode_none_ref(
    encoded: &EncodedShardRef,
    n_rows: usize,
    nnz: usize,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<DecodedShard, CodecError> {
    // Guard the indices length too: le_bytes_to_indices computes `nnz * elem`
    // internally, so a hostile `nnz` must be checked before the call (the
    // compressed paths gate this via their up-front indices_max) (F-e).
    checked_len(nnz, if index_dtype_u16 { 2 } else { 4 }, "indices")?;
    let indptr = le_bytes_to_u64(encoded.indptr_bytes, n_rows + 1)?;
    let indices = le_bytes_to_indices(encoded.indices_bytes, nnz, index_dtype_u16)?;
    let expected_len = checked_len(nnz, value_encoding.byte_width(), "values")?;
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

// ---------------------------------------------------------------------------
// CodecId::Scx1
// ---------------------------------------------------------------------------
