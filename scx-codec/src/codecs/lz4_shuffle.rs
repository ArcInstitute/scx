//! Split out of `dispatch.rs` (ORG-3.7-2). Pure move.

use crate::codec_id::{CodecError, DecodedShard, EncodedShard, EncodedShardRef, ValueEncoding};

use crate::shard_codec::{DecodeBounds, ShardCodec, ShardShape};

/// `CodecId::Lz4Shuffle` — byte-shuffle pre-filter, then LZ4 frame.
pub struct Lz4ShuffleCodec;

impl ShardCodec for Lz4ShuffleCodec {
    fn encode(
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        value_encoding: ValueEncoding,
        index_dtype_u16: bool,
    ) -> Result<EncodedShard, CodecError> {
        encode_lz4_shuffle(indptr, indices, values, value_encoding, index_dtype_u16)
    }

    fn decode(
        encoded: &EncodedShardRef,
        shape: ShardShape,
        value_encoding: ValueEncoding,
        index_dtype_u16: bool,
        bounds: &DecodeBounds,
    ) -> Result<DecodedShard, CodecError> {
        decode_lz4_shuffle_ref(encoded, shape, value_encoding, index_dtype_u16, bounds)
    }

    fn decode_indptr_only(
        indptr_bytes: &[u8],
        n_rows: usize,
        indptr_max: usize,
    ) -> Result<Vec<u64>, CodecError> {
        let shuffled = lz4_frame_decompress(indptr_bytes, indptr_max)?;
        let raw = byte_unshuffle(&shuffled, 8)?;
        le_bytes_to_u64(&raw, n_rows + 1)
    }
}

use crate::raw::{
    indices_to_le_bytes, le_bytes_to_indices, le_bytes_to_u64, u64_slice_to_le_bytes,
};
use crate::shuffle::{byte_shuffle, byte_unshuffle};

pub(crate) fn lz4_frame_compress(data: &[u8]) -> Result<Vec<u8>, CodecError> {
    use std::io::Write;
    let mut encoder = lz4_flex::frame::FrameEncoder::new(Vec::new());
    encoder.write_all(data)?;
    let buf = encoder
        .finish()
        .map_err(|e| CodecError::Io(std::io::Error::other(e)))?;
    Ok(buf)
}

/// Decompress an LZ4 frame, refusing to produce more than `max_bytes`.
///
/// `max_bytes` is the exact expected decompressed length, derived from the
/// shard's declared shape (`indptr_byte_cap` / `checked_len`) exactly as the
/// Zstd paths derive theirs. The cap is applied *during* decompression via
/// `Read::take`, not checked afterwards: an LZ4 frame compresses runs at a
/// ratio well past 100:1, so a small shard could otherwise force an
/// arbitrarily large allocation before anything noticed the length was wrong.
/// Mirrors [`zstd_decode_bounded`].
pub(crate) fn lz4_frame_decompress(data: &[u8], max_bytes: usize) -> Result<Vec<u8>, CodecError> {
    use std::io::Read;
    let decoder = lz4_flex::frame::FrameDecoder::new(data);
    let mut out = Vec::with_capacity(max_bytes.min(1 << 20));
    // `saturating_add`: no current caller can reach `max_bytes == usize::MAX`
    // (the indices arm's `checked_len(nnz, >= 2)` errors first), but this crate
    // sets `overflow-checks = true` in release, so a plain `+ 1` would make a
    // reader panic rather than error if a future caller ever did. Keep the
    // property local to the guard instead of resting it on call order.
    let mut limited = decoder.take((max_bytes as u64).saturating_add(1));
    limited.read_to_end(&mut out)?;
    if out.len() > max_bytes {
        return Err(CodecError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "decompressed size {} exceeds limit {}",
                out.len(),
                max_bytes
            ),
        )));
    }
    Ok(out)
}

fn encode_lz4_shuffle(
    indptr: &[u64],
    indices: &[u32],
    values: &[u8],
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<EncodedShard, CodecError> {
    let indptr_raw = u64_slice_to_le_bytes(indptr);
    let indices_raw = indices_to_le_bytes(indices, index_dtype_u16)?;

    // Byte-shuffle then LZ4 frame compress each array
    let indptr_shuffled = byte_shuffle(&indptr_raw, 8)?; // u64 = 8 bytes
    let index_width = if index_dtype_u16 { 2 } else { 4 };
    let indices_shuffled = byte_shuffle(&indices_raw, index_width)?;
    let values_shuffled = byte_shuffle(values, value_encoding.byte_width())?;

    let indptr_bytes = lz4_frame_compress(&indptr_shuffled)?;
    let indices_bytes = lz4_frame_compress(&indices_shuffled)?;
    let values_bytes = lz4_frame_compress(&values_shuffled)?;

    Ok(EncodedShard {
        indptr_bytes,
        indices_bytes,
        values_bytes,
    })
}

fn decode_lz4_shuffle_ref(
    encoded: &EncodedShardRef,
    shape: ShardShape,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
    bounds: &DecodeBounds,
) -> Result<DecodedShard, CodecError> {
    let (n_rows, nnz) = (shape.n_rows, shape.nnz);
    // Caps come from the driver's `DecodeBounds::derive` — this codec used to
    // derive nothing at all, which is how it reached `le_bytes_to_u64`'s
    // unchecked `count * 8` and panicked in release (#436).
    let index_width = if index_dtype_u16 { 2 } else { 4 };
    let DecodeBounds {
        indptr_max,
        indices_max,
        values_max,
    } = *bounds;

    // LZ4 frame decompress then byte-unshuffle each array
    let indptr_shuffled = lz4_frame_decompress(encoded.indptr_bytes, indptr_max)?;
    let indices_shuffled = lz4_frame_decompress(encoded.indices_bytes, indices_max)?;
    let values_shuffled = lz4_frame_decompress(encoded.values_bytes, values_max)?;

    let indptr_raw = byte_unshuffle(&indptr_shuffled, 8)?;
    let indices_raw = byte_unshuffle(&indices_shuffled, index_width)?;
    let values_raw = byte_unshuffle(&values_shuffled, value_encoding.byte_width())?;

    let indptr = le_bytes_to_u64(&indptr_raw, n_rows + 1)?;
    let indices = le_bytes_to_indices(&indices_raw, nnz, index_dtype_u16)?;

    // `values_max` is the exact expected length, so a *short* stream is what
    // remains to catch here — the cap above already refused a long one.
    if values_raw.len() != values_max {
        return Err(CodecError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "decompressed values byte length {} != expected {}",
                values_raw.len(),
                values_max
            ),
        )));
    }

    Ok((indptr, indices, values_raw))
}
