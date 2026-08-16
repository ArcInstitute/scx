//! `CodecId::ShufDeltaZstd` — byte-filter codec. Per sub-stream:
//!
//!   indices / indptr : byte-shuffle -> byte-delta -> zstd
//!   integer values   : byte-shuffle -> zstd          (no delta)
//!   float values     : zstd only    (no shuffle, no delta)
//!
//! Monolithic per shard. Random-row / grouped reads decode the whole shard; the
//! row-group-framed form (F5-b) rides the BlockIndex substrate and is out of
//! scope here.
//!
//! Split out of `dispatch.rs` (ORG-3.7-2).

use crate::byte_delta::{byte_delta_planes, byte_undelta_planes};
use crate::codec_id::{CodecError, DecodedShard, EncodedShard, EncodedShardRef, ValueEncoding};
use crate::codecs::zstd_codec::zstd_decode_bounded;
use crate::guards::expect_exact_len;
use crate::shard_codec::{DecodeBounds, ShardCodec, ShardShape};

/// `CodecId::ShufDeltaZstd` — byte-shuffle + byte-delta pre-filter, then zstd.
pub struct ShufDeltaZstdCodec;

impl ShardCodec for ShufDeltaZstdCodec {
    fn encode(
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        value_encoding: ValueEncoding,
        index_dtype_u16: bool,
    ) -> Result<EncodedShard, CodecError> {
        encode_shufdelta_zstd(indptr, indices, values, value_encoding, index_dtype_u16)
    }

    fn decode(
        encoded: &EncodedShardRef,
        shape: ShardShape,
        value_encoding: ValueEncoding,
        index_dtype_u16: bool,
        bounds: &DecodeBounds,
    ) -> Result<DecodedShard, CodecError> {
        decode_shufdelta_zstd_ref(encoded, shape, value_encoding, index_dtype_u16, bounds)
    }

    fn decode_indptr_only(
        indptr_bytes: &[u8],
        n_rows: usize,
        indptr_max: usize,
    ) -> Result<Vec<u64>, CodecError> {
        let mut planes = zstd_decode_bounded(indptr_bytes, indptr_max)?;
        expect_exact_len(planes.len(), indptr_max, "indptr")?;
        byte_undelta_planes(&mut planes, 8, n_rows + 1);
        let raw = byte_unshuffle(&planes, 8)?;
        le_bytes_to_u64(&raw, n_rows + 1)
    }
}

use crate::raw::{
    indices_to_le_bytes, le_bytes_to_indices, le_bytes_to_u64, u64_slice_to_le_bytes,
};
use crate::shuffle::{byte_shuffle, byte_unshuffle};

/// zstd level for the ShufDeltaZstd codec. Matches the plain `Zstd` codec's
/// level so a shufdelta-vs-zstd size comparison isolates the shuffle+delta
/// pre-filter rather than the compression level.
pub(crate) const SHUFDELTA_ZSTD_LEVEL: i32 = 3;

fn encode_shufdelta_zstd(
    indptr: &[u64],
    indices: &[u32],
    values: &[u8],
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<EncodedShard, CodecError> {
    let indptr_raw = u64_slice_to_le_bytes(indptr);
    let indices_raw = indices_to_le_bytes(indices, index_dtype_u16)?;

    // indptr: byte-shuffle(8) -> byte-delta -> zstd
    let mut indptr_planes = byte_shuffle(&indptr_raw, 8)?;
    byte_delta_planes(&mut indptr_planes, 8, indptr.len());
    let indptr_bytes = zstd::encode_all(indptr_planes.as_slice(), SHUFDELTA_ZSTD_LEVEL)?;

    // indices: byte-shuffle(width) -> byte-delta -> zstd
    let index_width = if index_dtype_u16 { 2 } else { 4 };
    let mut indices_planes = byte_shuffle(&indices_raw, index_width)?;
    byte_delta_planes(&mut indices_planes, index_width, indices.len());
    let indices_bytes = zstd::encode_all(indices_planes.as_slice(), SHUFDELTA_ZSTD_LEVEL)?;

    // values: integer -> byte-shuffle(width) -> zstd (no delta); float -> zstd only
    let values_bytes = if value_encoding.is_integer() {
        let shuffled = byte_shuffle(values, value_encoding.byte_width())?;
        zstd::encode_all(shuffled.as_slice(), SHUFDELTA_ZSTD_LEVEL)?
    } else {
        zstd::encode_all(values, SHUFDELTA_ZSTD_LEVEL)?
    };

    Ok(EncodedShard {
        indptr_bytes,
        indices_bytes,
        values_bytes,
    })
}

fn decode_shufdelta_zstd_ref(
    encoded: &EncodedShardRef,
    shape: ShardShape,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
    bounds: &DecodeBounds,
) -> Result<DecodedShard, CodecError> {
    let (n_rows, nnz) = (shape.n_rows, shape.nnz);
    let index_width = if index_dtype_u16 { 2 } else { 4 };
    let DecodeBounds {
        indptr_max,
        indices_max,
        values_max,
    } = *bounds;

    // indptr: zstd -> byte-undelta -> byte-unshuffle
    let mut indptr_planes = zstd_decode_bounded(encoded.indptr_bytes, indptr_max)?;
    expect_exact_len(indptr_planes.len(), indptr_max, "indptr")?;
    byte_undelta_planes(&mut indptr_planes, 8, n_rows + 1);
    let indptr_raw = byte_unshuffle(&indptr_planes, 8)?;

    // indices: zstd -> byte-undelta -> byte-unshuffle
    let mut indices_planes = zstd_decode_bounded(encoded.indices_bytes, indices_max)?;
    expect_exact_len(indices_planes.len(), indices_max, "indices")?;
    byte_undelta_planes(&mut indices_planes, index_width, nnz);
    let indices_raw = byte_unshuffle(&indices_planes, index_width)?;

    // values: integer -> zstd -> byte-unshuffle; float -> zstd only
    let values_raw = if value_encoding.is_integer() {
        let planes = zstd_decode_bounded(encoded.values_bytes, values_max)?;
        expect_exact_len(planes.len(), values_max, "values")?;
        byte_unshuffle(&planes, value_encoding.byte_width())?
    } else {
        zstd_decode_bounded(encoded.values_bytes, values_max)?
    };

    let indptr = le_bytes_to_u64(&indptr_raw, n_rows + 1)?;
    let indices = le_bytes_to_indices(&indices_raw, nnz, index_dtype_u16)?;

    let expected_len = values_max;
    if values_raw.len() != expected_len {
        return Err(CodecError::MalformedInput(format!(
            "shufdelta values byte length {} != expected {}",
            values_raw.len(),
            expected_len
        )));
    }

    Ok((indptr, indices, values_raw))
}
