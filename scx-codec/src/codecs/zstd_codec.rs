//! Split out of `dispatch.rs` (ORG-3.7-2). Pure move.

use crate::codec_id::{CodecError, DecodedShard, EncodedShard, EncodedShardRef, ValueEncoding};
use crate::guards::{checked_len, indptr_byte_cap};
use crate::raw::{
    indices_to_le_bytes, le_bytes_to_indices, le_bytes_to_u64, u64_slice_to_le_bytes,
};

pub(crate) fn encode_zstd(
    indptr: &[u64],
    indices: &[u32],
    values: &[u8],
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

pub(crate) fn decode_zstd_ref(
    encoded: &EncodedShardRef,
    n_rows: usize,
    nnz: usize,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<DecodedShard, CodecError> {
    let indptr_max = indptr_byte_cap(n_rows)?;
    let indices_max = checked_len(nnz, if index_dtype_u16 { 2 } else { 4 }, "indices")?;
    let values_max = checked_len(nnz, value_encoding.byte_width(), "values")?;

    let indptr_raw = zstd_decode_bounded(encoded.indptr_bytes, indptr_max)?;
    let indices_raw = zstd_decode_bounded(encoded.indices_bytes, indices_max)?;
    let values_raw = zstd_decode_bounded(encoded.values_bytes, values_max)?;

    let indptr = le_bytes_to_u64(&indptr_raw, n_rows + 1)?;
    let indices = le_bytes_to_indices(&indices_raw, nnz, index_dtype_u16)?;

    let expected_len = checked_len(nnz, value_encoding.byte_width(), "values")?;
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
