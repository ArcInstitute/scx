// Codec ID dispatch + zstd fallback (docs/codec.md (Codec IDs))

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Cursor;

use crate::bitstream::BitStreamError;
use crate::delta_golomb::{delta_golomb_decode, delta_golomb_encode};
use crate::forbp::{forbp_decode_with_hint, forbp_encode};
use crate::rice::{rice_decode, rice_encode, B_VAL};
use crate::shuffle::{byte_shuffle, byte_unshuffle};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Identifies the compression codec used for a shard (docs/format.md (Arrow IPC)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecId {
    /// Raw little-endian arrays, no compression.
    None = 0,
    /// Domain-specific: Delta-Golomb (indptr) + FOR-BP (indices) + Rice (values).
    /// Integer value encodings only.
    Scx1 = 1,
    /// Zstd compression per array. Works with any value encoding.
    Zstd = 2,
    /// LZ4 frame compression with byte-shuffle pre-filter.
    /// Matches Zarr/Blosc compression style. Works with any value encoding.
    Lz4Shuffle = 3,
    /// Pcodec (pco) lossless numerical compression.
    /// Optimal for float layers; uses Zstd for indptr/indices.
    Pcodec = 4,
}

impl CodecId {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::Scx1),
            2 => Some(Self::Zstd),
            3 => Some(Self::Lz4Shuffle),
            4 => Some(Self::Pcodec),
            _ => None,
        }
    }
}

/// Value encoding for the data array in a CSR shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueEncoding {
    Uint8 = 0,
    Uint16 = 1,
    Uint32 = 2,
    Float32 = 3,
    Float16 = 4,
}

impl ValueEncoding {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Uint16),
            2 => Some(Self::Uint32),
            3 => Some(Self::Float32),
            4 => Some(Self::Float16),
            _ => None,
        }
    }

    /// Number of bytes per value element.
    pub fn byte_width(&self) -> usize {
        match self {
            Self::Uint8 => 1,
            Self::Uint16 | Self::Float16 => 2,
            Self::Uint32 | Self::Float32 => 4,
        }
    }

    /// Returns `true` for integer encodings that can use the Scx1 codec.
    pub fn is_integer(&self) -> bool {
        matches!(self, Self::Uint8 | Self::Uint16 | Self::Uint32)
    }

    /// Encode a single f32 value to raw LE bytes, with range checking.
    ///
    /// This is the inverse of `values_raw_to_f32` for one element.
    pub fn encode_f32(&self, buf: &mut Vec<u8>, value: f32) -> Result<(), CodecError> {
        match self {
            Self::Uint8 => {
                if !(0.0..=255.0).contains(&value) {
                    return Err(CodecError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("value {value} out of range for uint8 (0..255)"),
                    )));
                }
                buf.push(value as u8);
            }
            Self::Uint16 => {
                if !(0.0..=65535.0).contains(&value) {
                    return Err(CodecError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("value {value} out of range for uint16 (0..65535)"),
                    )));
                }
                buf.extend_from_slice(&(value as u16).to_le_bytes());
            }
            Self::Uint32 => {
                if !(0.0..=u32::MAX as f32).contains(&value) {
                    return Err(CodecError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("value {value} out of range for uint32"),
                    )));
                }
                buf.extend_from_slice(&(value as u32).to_le_bytes());
            }
            Self::Float32 => buf.extend_from_slice(&value.to_le_bytes()),
            Self::Float16 => {
                buf.extend_from_slice(&half::f16::from_f32(value).to_le_bytes());
            }
        }
        Ok(())
    }

    /// Batch-encode a slice of f32 values to raw LE bytes.
    ///
    /// This is the inverse of `values_raw_to_f32`.
    pub fn encode_f32_batch(&self, data: &[f32]) -> Result<Vec<u8>, CodecError> {
        let mut bytes = Vec::with_capacity(data.len() * self.byte_width());
        for &v in data {
            self.encode_f32(&mut bytes, v)?;
        }
        Ok(bytes)
    }
}

/// The encoded byte arrays for a single CSR shard (owned).
#[derive(Debug)]
pub struct EncodedShard {
    pub indptr_bytes: Vec<u8>,
    pub indices_bytes: Vec<u8>,
    pub values_bytes: Vec<u8>,
}

/// Borrowed reference to encoded shard byte arrays (zero-copy from mmap).
#[derive(Debug)]
pub struct EncodedShardRef<'a> {
    pub indptr_bytes: &'a [u8],
    pub indices_bytes: &'a [u8],
    pub values_bytes: &'a [u8],
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error("bitstream error: {0}")]
    BitStream(#[from] BitStreamError),

    #[error("unsupported codec id: {0}")]
    UnsupportedCodec(u8),

    #[error("unsupported value encoding: {0}")]
    UnsupportedValueEncoding(u8),

    #[error("Scx1 codec does not support float value encodings")]
    FloatWithScx1,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Decoded shard: `(indptr, indices, values_raw_bytes)`.
pub type DecodedShard = (Vec<u64>, Vec<u32>, Vec<u8>);

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Encode a CSR shard's three arrays using the specified codec.
///
/// - `indptr`: the indptr array (length = n_rows + 1).
/// - `indices`: the column indices (length = nnz), stored as u32.
/// - `values`: raw little-endian bytes of the value array (length = nnz × value_encoding.byte_width()).
/// - `index_dtype_u16`: if true, indices fit in u16 (n_vars <= 65535).
pub fn encode_shard(
    indptr: &[u64],
    indices: &[u32],
    values: &[u8],
    codec_id: CodecId,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<EncodedShard, CodecError> {
    match codec_id {
        CodecId::None => encode_none(indptr, indices, values, index_dtype_u16),
        CodecId::Scx1 => encode_scx1(indptr, indices, values, value_encoding, index_dtype_u16),
        CodecId::Zstd => encode_zstd(indptr, indices, values, index_dtype_u16),
        CodecId::Lz4Shuffle => {
            encode_lz4_shuffle(indptr, indices, values, value_encoding, index_dtype_u16)
        }
        CodecId::Pcodec => encode_pcodec(indptr, indices, values, value_encoding, index_dtype_u16),
    }
}

/// Decode an `EncodedShard` back to `(indptr, indices, values_bytes)`.
///
/// - `n_rows`: number of rows (indptr has n_rows + 1 entries).
/// - `nnz`: number of non-zero values.
pub fn decode_shard(
    encoded: &EncodedShard,
    codec_id: CodecId,
    value_encoding: ValueEncoding,
    n_rows: usize,
    nnz: usize,
    index_dtype_u16: bool,
) -> Result<DecodedShard, CodecError> {
    let r = EncodedShardRef {
        indptr_bytes: &encoded.indptr_bytes,
        indices_bytes: &encoded.indices_bytes,
        values_bytes: &encoded.values_bytes,
    };
    decode_shard_ref(&r, codec_id, value_encoding, n_rows, nnz, index_dtype_u16)
}

/// Decode an `EncodedShardRef` (borrowed) back to `(indptr, indices, values_bytes)`.
///
/// Zero-copy variant that avoids cloning mmap slices into owned Vecs.
pub fn decode_shard_ref(
    encoded: &EncodedShardRef,
    codec_id: CodecId,
    value_encoding: ValueEncoding,
    n_rows: usize,
    nnz: usize,
    index_dtype_u16: bool,
) -> Result<DecodedShard, CodecError> {
    match codec_id {
        CodecId::None => decode_none_ref(encoded, n_rows, nnz, value_encoding, index_dtype_u16),
        CodecId::Scx1 => decode_scx1_ref(encoded, value_encoding, n_rows, nnz, index_dtype_u16),
        CodecId::Zstd => decode_zstd_ref(encoded, n_rows, nnz, value_encoding, index_dtype_u16),
        CodecId::Lz4Shuffle => {
            decode_lz4_shuffle_ref(encoded, n_rows, nnz, value_encoding, index_dtype_u16)
        }
        CodecId::Pcodec => decode_pcodec_ref(encoded, n_rows, nnz, value_encoding, index_dtype_u16),
    }
}

/// Scipy-compatible decoded shard: `(indptr_i64, indices_i32, data_f32)`.
///
/// Eliminates intermediate type conversions by producing the final scipy
/// types directly from the codec decoders.
pub type ScipyShard = (Vec<i64>, Vec<i32>, Vec<f32>);

/// Decode an `EncodedShardRef` directly to scipy-compatible types.
///
/// Returns `(Vec<i64>, Vec<i32>, Vec<f32>)` without intermediate raw byte
/// conversions, saving 3 allocations per shard compared to `decode_shard_ref`
/// + manual type conversion.
pub fn decode_shard_scipy(
    encoded: &EncodedShardRef,
    codec_id: CodecId,
    value_encoding: ValueEncoding,
    n_rows: usize,
    nnz: usize,
    index_dtype_u16: bool,
) -> Result<ScipyShard, CodecError> {
    // For Scx1, we can avoid the u32→raw_bytes→f32 chain for values
    if codec_id == CodecId::Scx1 {
        if !value_encoding.is_integer() {
            return Err(CodecError::FloatWithScx1);
        }
        // indptr: delta_golomb → Vec<u64> → Vec<i64>
        let indptr_u64 = delta_golomb_decode(encoded.indptr_bytes, n_rows + 1)?;
        let indptr = u64_vec_to_i64(indptr_u64)?;

        // indices: forbp → Vec<u32> → Vec<i32>
        let (indices_u32, _) =
            forbp_decode_with_hint(encoded.indices_bytes, n_rows, nnz, index_dtype_u16)?;
        let indices = u32_vec_to_i32(indices_u32)?;

        // values: rice → Vec<u32> → Vec<f32> directly (skip raw bytes intermediate)
        let values_u32 = rice_decode(encoded.values_bytes, nnz, B_VAL)?;
        let data: Vec<f32> = values_u32.into_iter().map(|v| v as f32).collect();

        return Ok((indptr, indices, data));
    }

    // For None, Zstd, and Lz4Shuffle: decode to raw types, then convert
    let (indptr_u64, indices_u32, values_raw) = decode_shard_ref(
        encoded,
        codec_id,
        value_encoding,
        n_rows,
        nnz,
        index_dtype_u16,
    )?;
    let indptr = u64_vec_to_i64(indptr_u64)?;
    let indices = u32_vec_to_i32(indices_u32)?;
    let data = values_raw_to_f32(&values_raw, value_encoding);
    Ok((indptr, indices, data))
}

/// Convert Vec<u64> to Vec<i64> via zero-copy reinterpretation.
/// CSR indptr values are always non-negative and well below i64::MAX,
/// so the bit patterns are identical. Uses bytemuck for safe transmute.
fn u64_vec_to_i64(data: Vec<u64>) -> Result<Vec<i64>, CodecError> {
    debug_assert!(
        data.iter().all(|&v| v <= i64::MAX as u64),
        "indptr value exceeds i64::MAX"
    );
    Ok(bytemuck::cast_vec::<u64, i64>(data))
}

/// Convert Vec<u32> to Vec<i32> via zero-copy reinterpretation.
/// Column indices are always non-negative and below n_vars (well within i32 range),
/// so the bit patterns are identical. Uses bytemuck for safe transmute.
fn u32_vec_to_i32(data: Vec<u32>) -> Result<Vec<i32>, CodecError> {
    debug_assert!(
        data.iter().all(|&v| v <= i32::MAX as u32),
        "index value exceeds i32::MAX"
    );
    Ok(bytemuck::cast_vec::<u32, i32>(data))
}

/// Convert raw LE value bytes to f32 according to ValueEncoding.
fn values_raw_to_f32(raw: &[u8], encoding: ValueEncoding) -> Vec<f32> {
    match encoding {
        ValueEncoding::Uint8 => {
            let mut out = Vec::with_capacity(raw.len());
            out.extend(raw.iter().map(|&b| b as f32));
            out
        }
        ValueEncoding::Uint16 => raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as f32)
            .collect(),
        ValueEncoding::Uint32 => raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        ValueEncoding::Float32 => {
            // `bytemuck::cast_slice::<u8, f32>` panics if the source bytes
            // aren't 4-byte aligned. Mmap'd payloads are usually aligned, but
            // we can't rely on it — decompressed buffers from Zstd/LZ4 land at
            // whatever alignment the allocator picked. Branch on alignment +
            // length; fall back to a scalar byteswap-free decode otherwise.
            #[cfg(target_endian = "little")]
            {
                if (raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>())
                    && raw.len().is_multiple_of(std::mem::size_of::<f32>())
                {
                    bytemuck::cast_slice::<u8, f32>(raw).to_vec()
                } else {
                    raw.chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                }
            }
            #[cfg(not(target_endian = "little"))]
            {
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
        }
        ValueEncoding::Float16 => {
            // `half::slice::HalfFloatSliceExt::convert_to_f32_slice` uses a
            // vectorized path when the input is aligned. Same alignment
            // guard as Float32 above.
            #[cfg(target_endian = "little")]
            {
                if (raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<half::f16>())
                    && raw.len().is_multiple_of(std::mem::size_of::<half::f16>())
                {
                    use half::slice::HalfFloatSliceExt;
                    // SAFETY: alignment + length checked immediately above,
                    // `half::f16` is `#[repr(transparent)]` over `u16`, so any
                    // aligned 2-byte little-endian group is a valid `f16` bit
                    // pattern.
                    let src: &[half::f16] = unsafe {
                        std::slice::from_raw_parts(
                            raw.as_ptr() as *const half::f16,
                            raw.len() / std::mem::size_of::<half::f16>(),
                        )
                    };
                    let mut out = vec![0.0f32; src.len()];
                    src.convert_to_f32_slice(&mut out);
                    out
                } else {
                    raw.chunks_exact(2)
                        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                        .collect()
                }
            }
            #[cfg(not(target_endian = "little"))]
            {
                raw.chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CodecId::None
// ---------------------------------------------------------------------------

fn encode_none(
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
    n_rows: usize,
    nnz: usize,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<DecodedShard, CodecError> {
    let indptr = le_bytes_to_u64(encoded.indptr_bytes, n_rows + 1)?;
    let indices = le_bytes_to_indices(encoded.indices_bytes, nnz, index_dtype_u16)?;
    let expected_len = nnz * value_encoding.byte_width();
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

fn encode_scx1(
    indptr: &[u64],
    indices: &[u32],
    values: &[u8],
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<EncodedShard, CodecError> {
    if !value_encoding.is_integer() {
        return Err(CodecError::FloatWithScx1);
    }

    // indptr → Delta-Golomb
    let indptr_bytes = delta_golomb_encode(indptr);

    // indices → FOR-BP (needs row_lengths from indptr)
    let row_lengths: Vec<usize> = indptr.windows(2).map(|w| (w[1] - w[0]) as usize).collect();
    let indices_bytes = forbp_encode(indices, &row_lengths, index_dtype_u16)?;

    // values → reinterpret to u32, then Rice encode
    let values_u32 = raw_bytes_to_u32(values, value_encoding);
    let values_bytes = rice_encode(&values_u32, B_VAL)?;

    Ok(EncodedShard {
        indptr_bytes,
        indices_bytes,
        values_bytes,
    })
}

fn decode_scx1_ref(
    encoded: &EncodedShardRef,
    value_encoding: ValueEncoding,
    n_rows: usize,
    nnz: usize,
    index_dtype_u16: bool,
) -> Result<DecodedShard, CodecError> {
    if !value_encoding.is_integer() {
        return Err(CodecError::FloatWithScx1);
    }

    // indptr ← Delta-Golomb
    let indptr = delta_golomb_decode(encoded.indptr_bytes, n_rows + 1)?;

    // indices ← FOR-BP (with nnz hint for pre-allocation)
    let (indices, _row_lengths) =
        forbp_decode_with_hint(encoded.indices_bytes, n_rows, nnz, index_dtype_u16)?;

    // values ← Rice decode, then convert u32 back to raw bytes
    let values_u32 = rice_decode(encoded.values_bytes, nnz, B_VAL)?;
    let values_bytes = u32_to_raw_bytes(&values_u32, value_encoding)?;

    Ok((indptr, indices, values_bytes))
}

// ---------------------------------------------------------------------------
// CodecId::Zstd
// ---------------------------------------------------------------------------

fn encode_zstd(
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
fn zstd_decode_bounded(data: &[u8], max_bytes: usize) -> Result<Vec<u8>, CodecError> {
    use std::io::Read;
    let decoder = zstd::Decoder::new(data)?;
    // Cap initial allocation to avoid huge alloc from untrusted max_bytes
    let mut output = Vec::with_capacity(max_bytes.min(1 << 20));
    let mut limited = decoder.take(max_bytes as u64 + 1);
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
    n_rows: usize,
    nnz: usize,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<DecodedShard, CodecError> {
    let indptr_max = (n_rows + 1) * 8;
    let indices_max = nnz * (if index_dtype_u16 { 2 } else { 4 });
    let values_max = nnz * value_encoding.byte_width();

    let indptr_raw = zstd_decode_bounded(encoded.indptr_bytes, indptr_max)?;
    let indices_raw = zstd_decode_bounded(encoded.indices_bytes, indices_max)?;
    let values_raw = zstd_decode_bounded(encoded.values_bytes, values_max)?;

    let indptr = le_bytes_to_u64(&indptr_raw, n_rows + 1)?;
    let indices = le_bytes_to_indices(&indices_raw, nnz, index_dtype_u16)?;

    let expected_len = nnz * value_encoding.byte_width();
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

// ---------------------------------------------------------------------------
// CodecId::Lz4Shuffle
// ---------------------------------------------------------------------------

fn lz4_frame_compress(data: &[u8]) -> Result<Vec<u8>, CodecError> {
    use std::io::Write;
    let mut encoder = lz4_flex::frame::FrameEncoder::new(Vec::new());
    encoder.write_all(data)?;
    let buf = encoder
        .finish()
        .map_err(|e| CodecError::Io(std::io::Error::other(e)))?;
    Ok(buf)
}

fn lz4_frame_decompress(data: &[u8]) -> Result<Vec<u8>, CodecError> {
    use std::io::Read;
    let mut decoder = lz4_flex::frame::FrameDecoder::new(data);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out)?;
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
    let indptr_shuffled = byte_shuffle(&indptr_raw, 8); // u64 = 8 bytes
    let index_width = if index_dtype_u16 { 2 } else { 4 };
    let indices_shuffled = byte_shuffle(&indices_raw, index_width);
    let values_shuffled = byte_shuffle(values, value_encoding.byte_width());

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
    n_rows: usize,
    nnz: usize,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<DecodedShard, CodecError> {
    // LZ4 frame decompress then byte-unshuffle each array
    let indptr_shuffled = lz4_frame_decompress(encoded.indptr_bytes)?;
    let indices_shuffled = lz4_frame_decompress(encoded.indices_bytes)?;
    let values_shuffled = lz4_frame_decompress(encoded.values_bytes)?;

    let indptr_raw = byte_unshuffle(&indptr_shuffled, 8);
    let index_width = if index_dtype_u16 { 2 } else { 4 };
    let indices_raw = byte_unshuffle(&indices_shuffled, index_width);
    let values_raw = byte_unshuffle(&values_shuffled, value_encoding.byte_width());

    let indptr = le_bytes_to_u64(&indptr_raw, n_rows + 1)?;
    let indices = le_bytes_to_indices(&indices_raw, nnz, index_dtype_u16)?;

    let expected_len = nnz * value_encoding.byte_width();
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

// ---------------------------------------------------------------------------
// CodecId::Pcodec
// ---------------------------------------------------------------------------

fn encode_pcodec(
    indptr: &[u64],
    indices: &[u32],
    values: &[u8],
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<EncodedShard, CodecError> {
    let indptr_raw = u64_slice_to_le_bytes(indptr);
    let indices_raw = indices_to_le_bytes(indices, index_dtype_u16)?;

    // indptr and indices: Zstd (already well-compressed by generic codecs)
    let indptr_bytes = zstd::encode_all(indptr_raw.as_slice(), 3)?;
    let indices_bytes = zstd::encode_all(indices_raw.as_slice(), 3)?;

    // values: Pcodec for float encodings, Zstd for integer encodings
    let values_bytes = match value_encoding {
        ValueEncoding::Float32 => {
            let floats: Vec<f32> = values
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            pco::standalone::simple_compress(&floats, &pco::ChunkConfig::default())
                .map_err(|e| CodecError::Io(std::io::Error::other(e.to_string())))?
        }
        ValueEncoding::Float16 => {
            // Widen f16 to f32, then compress as f32
            let floats: Vec<f32> = values
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
            pco::standalone::simple_compress(&floats, &pco::ChunkConfig::default())
                .map_err(|e| CodecError::Io(std::io::Error::other(e.to_string())))?
        }
        _ => {
            // Integer encodings: Zstd (Pcodec advantage is on floats)
            zstd::encode_all(values, 3)?
        }
    };

    Ok(EncodedShard {
        indptr_bytes,
        indices_bytes,
        values_bytes,
    })
}

fn decode_pcodec_ref(
    encoded: &EncodedShardRef,
    n_rows: usize,
    nnz: usize,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<DecodedShard, CodecError> {
    // indptr and indices: Zstd decompress
    let indptr_max = (n_rows + 1) * 8;
    let indices_max = nnz * (if index_dtype_u16 { 2 } else { 4 });

    let indptr_raw = zstd_decode_bounded(encoded.indptr_bytes, indptr_max)?;
    let indices_raw = zstd_decode_bounded(encoded.indices_bytes, indices_max)?;

    let indptr = le_bytes_to_u64(&indptr_raw, n_rows + 1)?;
    let indices = le_bytes_to_indices(&indices_raw, nnz, index_dtype_u16)?;

    // values: Pcodec for float encodings, Zstd for integer encodings
    let values_raw = match value_encoding {
        ValueEncoding::Float32 => {
            let floats: Vec<f32> = pco::standalone::simple_decompress(encoded.values_bytes)
                .map_err(|e| CodecError::Io(std::io::Error::other(e.to_string())))?;
            let mut buf = Vec::with_capacity(floats.len() * 4);
            for &f in &floats {
                buf.extend_from_slice(&f.to_le_bytes());
            }
            buf
        }
        ValueEncoding::Float16 => {
            // Decompress as f32, narrow back to f16
            let floats: Vec<f32> = pco::standalone::simple_decompress(encoded.values_bytes)
                .map_err(|e| CodecError::Io(std::io::Error::other(e.to_string())))?;
            let mut buf = Vec::with_capacity(floats.len() * 2);
            for &f in &floats {
                buf.extend_from_slice(&half::f16::from_f32(f).to_le_bytes());
            }
            buf
        }
        _ => {
            // Integer encodings: Zstd decompress
            let values_max = nnz * value_encoding.byte_width();
            zstd_decode_bounded(encoded.values_bytes, values_max)?
        }
    };

    let expected_len = nnz * value_encoding.byte_width();
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

// ---------------------------------------------------------------------------
// Helpers: serialization
// ---------------------------------------------------------------------------

fn u64_slice_to_le_bytes(data: &[u64]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(data.len() * 8);
    for &v in data {
        buf.write_u64::<LittleEndian>(v).unwrap();
    }
    buf
}

fn le_bytes_to_u64(data: &[u8], count: usize) -> Result<Vec<u64>, CodecError> {
    if data.len() != count * 8 {
        return Err(CodecError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "indptr byte length {} != expected {}",
                data.len(),
                count * 8
            ),
        )));
    }
    let mut cursor = Cursor::new(data);
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        result.push(cursor.read_u64::<LittleEndian>()?);
    }
    Ok(result)
}

fn indices_to_le_bytes(indices: &[u32], index_dtype_u16: bool) -> Result<Vec<u8>, CodecError> {
    if index_dtype_u16 {
        let mut buf = Vec::with_capacity(indices.len() * 2);
        for &v in indices {
            if v > u16::MAX as u32 {
                return Err(CodecError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("index {} exceeds u16 range", v),
                )));
            }
            buf.write_u16::<LittleEndian>(v as u16).unwrap();
        }
        Ok(buf)
    } else {
        let mut buf = Vec::with_capacity(indices.len() * 4);
        for &v in indices {
            buf.write_u32::<LittleEndian>(v).unwrap();
        }
        Ok(buf)
    }
}

fn le_bytes_to_indices(
    data: &[u8],
    count: usize,
    index_dtype_u16: bool,
) -> Result<Vec<u32>, CodecError> {
    let elem_size = if index_dtype_u16 { 2 } else { 4 };
    if data.len() != count * elem_size {
        return Err(CodecError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "indices byte length {} != expected {}",
                data.len(),
                count * elem_size
            ),
        )));
    }
    let mut cursor = Cursor::new(data);
    let mut result = Vec::with_capacity(count);
    if index_dtype_u16 {
        for _ in 0..count {
            result.push(cursor.read_u16::<LittleEndian>()? as u32);
        }
    } else {
        for _ in 0..count {
            result.push(cursor.read_u32::<LittleEndian>()?);
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Helpers: value type conversion for Rice codec
// ---------------------------------------------------------------------------

/// Reinterpret raw LE value bytes as `Vec<u32>` according to `ValueEncoding`.
fn raw_bytes_to_u32(data: &[u8], encoding: ValueEncoding) -> Vec<u32> {
    match encoding {
        ValueEncoding::Uint8 => data.iter().map(|&b| b as u32).collect(),
        ValueEncoding::Uint16 => {
            let mut cursor = Cursor::new(data);
            let mut out = Vec::with_capacity(data.len() / 2);
            while let Ok(v) = cursor.read_u16::<LittleEndian>() {
                out.push(v as u32);
            }
            out
        }
        ValueEncoding::Uint32 => {
            let mut cursor = Cursor::new(data);
            let mut out = Vec::with_capacity(data.len() / 4);
            while let Ok(v) = cursor.read_u32::<LittleEndian>() {
                out.push(v);
            }
            out
        }
        ValueEncoding::Float32 | ValueEncoding::Float16 => {
            unreachable!("raw_bytes_to_u32 called with float encoding")
        }
    }
}

/// Convert `Vec<u32>` back to raw LE bytes according to `ValueEncoding`.
fn u32_to_raw_bytes(data: &[u32], encoding: ValueEncoding) -> Result<Vec<u8>, CodecError> {
    match encoding {
        ValueEncoding::Uint8 => {
            let mut out = Vec::with_capacity(data.len());
            for &v in data {
                if v > u8::MAX as u32 {
                    return Err(CodecError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("decoded value {} exceeds u8 range", v),
                    )));
                }
                out.push(v as u8);
            }
            Ok(out)
        }
        ValueEncoding::Uint16 => {
            let mut buf = Vec::with_capacity(data.len() * 2);
            for &v in data {
                if v > u16::MAX as u32 {
                    return Err(CodecError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("decoded value {} exceeds u16 range", v),
                    )));
                }
                buf.write_u16::<LittleEndian>(v as u16).unwrap();
            }
            Ok(buf)
        }
        ValueEncoding::Uint32 => {
            let mut buf = Vec::with_capacity(data.len() * 4);
            for &v in data {
                buf.write_u32::<LittleEndian>(v).unwrap();
            }
            Ok(buf)
        }
        ValueEncoding::Float32 | ValueEncoding::Float16 => {
            unreachable!("u32_to_raw_bytes called with float encoding")
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(CodecId::from_u8(5), None);

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
    #[should_panic(expected = "indptr value exceeds i64::MAX")]
    fn test_u64_to_i64_debug_assert_overflow() {
        let data = vec![0u64, 100, u64::MAX];
        let _ = u64_vec_to_i64(data);
    }

    #[test]
    fn test_u32_to_i32_cast_valid() {
        let data = vec![0u32, 100, i32::MAX as u32];
        let result = u32_vec_to_i32(data).unwrap();
        assert_eq!(result, vec![0i32, 100, i32::MAX]);
    }

    #[test]
    #[should_panic(expected = "index value exceeds i32::MAX")]
    fn test_u32_to_i32_debug_assert_overflow() {
        let data = vec![0u32, 100, u32::MAX];
        let _ = u32_vec_to_i32(data);
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
}
