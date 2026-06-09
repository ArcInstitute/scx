// Codec ID dispatch + zstd fallback (docs/codec.md (Codec IDs))

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Cursor;

use crate::bitstream::BitStreamError;
use crate::delta_golomb::{delta_golomb_decode, delta_golomb_encode};
use crate::forbp::{
    forbp_decode_with_hint, forbp_decode_with_metadata, forbp_encode_with_metadata,
    ForBpRowMetadata,
};
use crate::rice::{
    rice_decode, rice_decode_with_metadata, rice_encode_with_metadata, RiceBlockMetadata, B_VAL,
};
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

    /// Parse a CLI/codec selection string. `"auto"` → `None` (the writer
    /// auto-selects per shard); every other accepted token maps to an
    /// explicit codec. This is the single source of truth for the CLI codec
    /// vocabulary — all command-line entry points delegate here so the
    /// accepted set can't drift between subcommands.
    pub fn parse_cli(s: &str) -> Result<Option<CodecId>, String> {
        match s {
            "auto" => Ok(None),
            "none" => Ok(Some(CodecId::None)),
            "scx1" => Ok(Some(CodecId::Scx1)),
            "zstd" => Ok(Some(CodecId::Zstd)),
            "lz4" => Ok(Some(CodecId::Lz4Shuffle)),
            "pcodec" => Ok(Some(CodecId::Pcodec)),
            other => Err(format!(
                "unknown codec: '{other}'. Use auto, none, scx1, zstd, lz4, or pcodec."
            )),
        }
    }

    /// Human-readable codec name for display (e.g. `scx info`). The single
    /// source of truth for codec id → name rendering.
    pub fn display_name(&self) -> &'static str {
        match self {
            CodecId::None => "none",
            CodecId::Scx1 => "scx1",
            CodecId::Zstd => "zstd",
            CodecId::Lz4Shuffle => "lz4+shuffle",
            CodecId::Pcodec => "pcodec",
        }
    }
}

/// Caller-supplied codec choice for write operations that emit shards.
///
/// `Auto` defers per-shard codec selection to the writer (data- and
/// modality-driven). `Explicit(c)` forces every emitted shard to use
/// codec `c`. Distinguishing these avoids overloading `CodecId::None`
/// as a sentinel for "auto-select" (`None` is a real codec: no
/// compression).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecSelection {
    Auto,
    Explicit(CodecId),
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

/// The Scx1 decode metadata produced as a byproduct of actual encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scx1DecodeMetadata {
    pub rows: Vec<ForBpRowMetadata>,
    pub rice_blocks: Vec<RiceBlockMetadata>,
}

/// The encoded byte arrays for a single CSR shard (owned).
#[derive(Debug)]
pub struct EncodedShard {
    pub indptr_bytes: Vec<u8>,
    pub indices_bytes: Vec<u8>,
    pub values_bytes: Vec<u8>,
    /// Present only for Scx1 integer shards. This is emitted by the encoder
    /// that wrote the bitstreams, so decode sidecars do not re-derive codec
    /// internals in a second crate.
    pub scx1_decode: Option<Scx1DecodeMetadata>,
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

    #[error("malformed codec input: {0}")]
    MalformedInput(String),
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

/// Convert a raw [`DecodedShard`] (`u64` indptr / `u32` indices / raw value
/// bytes) into the scipy-compatible `(i64, i32, f32)` triple, matching
/// [`decode_shard_scipy`]'s conversions. Lets callers that decode via the
/// metadata offsets ([`decode_scx1_row_range`] / parallel decode) produce the
/// same scipy types the sequential reader path returns.
pub fn decoded_shard_to_scipy(
    decoded: DecodedShard,
    value_encoding: ValueEncoding,
) -> Result<ScipyShard, CodecError> {
    let (indptr_u64, indices_u32, values_raw) = decoded;
    let indptr = u64_vec_to_i64(indptr_u64)?;
    let indices = u32_vec_to_i32(indices_u32)?;
    let data = values_raw_to_f32(&values_raw, value_encoding);
    Ok((indptr, indices, data))
}

/// Decode **only** the indptr region of a shard, skipping indices/data.
///
/// Used by callers that need just the row-pointer array — e.g. the
/// streaming SCX → h5ad export's `precompute_total_nnz` when a
/// deletion vector is active and only nnz-per-row counts matter.
/// Mirrors the indptr sub-path of [`decode_shard_scipy`] but does no
/// work on `indices_bytes` or `values_bytes`.
pub fn decode_indptr_only(
    indptr_bytes: &[u8],
    codec_id: CodecId,
    n_rows: usize,
) -> Result<Vec<i64>, CodecError> {
    let indptr_u64: Vec<u64> = match codec_id {
        CodecId::None => le_bytes_to_u64(indptr_bytes, n_rows + 1)?,
        CodecId::Scx1 => delta_golomb_decode(indptr_bytes, n_rows + 1)?,
        CodecId::Zstd | CodecId::Pcodec => {
            let raw = zstd_decode_bounded(indptr_bytes, (n_rows + 1) * 8)?;
            le_bytes_to_u64(&raw, n_rows + 1)?
        }
        CodecId::Lz4Shuffle => {
            let shuffled = lz4_frame_decompress(indptr_bytes)?;
            let raw = byte_unshuffle(&shuffled, 8)?;
            le_bytes_to_u64(&raw, n_rows + 1)?
        }
    };
    u64_vec_to_i64(indptr_u64)
}

/// Convert Vec<u64> to Vec<i64> via zero-copy reinterpretation.
/// CSR indptr values are always non-negative and well below i64::MAX,
/// so the bit patterns are identical. Uses bytemuck for safe transmute.
fn u64_vec_to_i64(data: Vec<u64>) -> Result<Vec<i64>, CodecError> {
    if let Some(&bad) = data.iter().find(|&&v| v > i64::MAX as u64) {
        return Err(CodecError::MalformedInput(format!(
            "indptr value {bad} exceeds i64::MAX (corrupt or hostile input)"
        )));
    }
    Ok(bytemuck::cast_vec::<u64, i64>(data))
}

/// Convert Vec<u32> to Vec<i32> via zero-copy reinterpretation.
/// Column indices are always non-negative and below n_vars (well within i32 range),
/// so the bit patterns are identical. Uses bytemuck for safe transmute.
fn u32_vec_to_i32(data: Vec<u32>) -> Result<Vec<i32>, CodecError> {
    if let Some(&bad) = data.iter().find(|&&v| v > i32::MAX as u32) {
        return Err(CodecError::MalformedInput(format!(
            "column index {bad} exceeds i32::MAX (corrupt or hostile input)"
        )));
    }
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
        scx1_decode: None,
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
    let indptr_bytes = delta_golomb_encode(indptr)?;

    // indices → FOR-BP (needs row_lengths from indptr)
    let row_lengths: Vec<usize> = indptr.windows(2).map(|w| (w[1] - w[0]) as usize).collect();
    let indices_encoded = forbp_encode_with_metadata(indices, &row_lengths, index_dtype_u16)?;

    // values → reinterpret to u32, then Rice encode
    let values_u32 = raw_bytes_to_u32(values, value_encoding)?;
    let values_encoded = rice_encode_with_metadata(&values_u32, B_VAL)?;

    Ok(EncodedShard {
        indptr_bytes,
        indices_bytes: indices_encoded.bytes,
        values_bytes: values_encoded.bytes,
        scx1_decode: Some(Scx1DecodeMetadata {
            rows: indices_encoded.rows,
            rice_blocks: values_encoded.blocks,
        }),
    })
}

/// Decode a Scx1 shard through encoder-produced metadata offsets.
///
/// The returned arrays should match [`decode_shard_ref`] for the same shard.
/// Unlike the normal decoder, this reconstructs `indptr` from row metadata and
/// seeks directly to per-row/per-block offset positions for indices and values.
pub fn decode_scx1_with_metadata(
    encoded: &EncodedShardRef,
    value_encoding: ValueEncoding,
    n_rows: usize,
    nnz: usize,
    metadata: &Scx1DecodeMetadata,
) -> Result<DecodedShard, CodecError> {
    if !value_encoding.is_integer() {
        return Err(CodecError::FloatWithScx1);
    }
    if metadata.rows.len() != n_rows {
        return Err(CodecError::MalformedInput(format!(
            "Scx1 metadata row count {} != n_rows {n_rows}",
            metadata.rows.len()
        )));
    }

    let mut indptr = Vec::with_capacity(n_rows + 1);
    indptr.push(0);
    let mut covered = 0u64;
    for (row_idx, row) in metadata.rows.iter().enumerate() {
        if row.value_start != covered {
            return Err(CodecError::MalformedInput(format!(
                "Scx1 metadata row {row_idx} value_start {} != expected {covered}",
                row.value_start
            )));
        }
        covered = covered
            .checked_add(row.nnz as u64)
            .ok_or_else(|| CodecError::MalformedInput("Scx1 metadata nnz overflows u64".into()))?;
        indptr.push(covered);
    }
    if covered != nnz as u64 {
        return Err(CodecError::MalformedInput(format!(
            "Scx1 metadata covers {covered} values, expected {nnz}"
        )));
    }

    let indices = forbp_decode_with_metadata(encoded.indices_bytes, &metadata.rows)?;
    if indices.len() != nnz {
        return Err(CodecError::MalformedInput(format!(
            "Scx1 metadata decoded {} indices, expected {nnz}",
            indices.len()
        )));
    }

    let values_u32 = rice_decode_with_metadata(encoded.values_bytes, &metadata.rice_blocks)?;
    if values_u32.len() != nnz {
        return Err(CodecError::MalformedInput(format!(
            "Scx1 metadata decoded {} values, expected {nnz}",
            values_u32.len()
        )));
    }
    let values_bytes = u32_to_raw_bytes(&values_u32, value_encoding)?;

    Ok((indptr, indices, values_bytes))
}

/// Random-access decode of a contiguous **row range** `[row_start, row_start +
/// n_rows)` of an Scx1 shard, using the sidecar offsets to seek directly.
///
/// The result is byte-identical to the corresponding slice of
/// [`decode_scx1_with_metadata`] / [`decode_shard_ref`], but the work is
/// O(window) rather than O(shard): only the requested rows' FOR-BP deltas and
/// the Rice blocks covering their value range are touched. Returns a shard-local
/// `(indptr, indices, values_bytes)` whose `indptr` starts at 0.
///
/// FOR-BP rows are seeked by absolute `indices_bit_offset` (the row sub-slice is
/// passed verbatim). The Rice blocks are keyed by value ordinal (256/block), so
/// the covering block span is rebased to a 0-based `value_start` (absolute
/// `bit_offset` untouched) to satisfy `rice_decode_with_metadata`'s continuity
/// check, then the partial head/tail of the 256-value blocks is trimmed.
pub fn decode_scx1_row_range(
    encoded: &EncodedShardRef,
    value_encoding: ValueEncoding,
    metadata: &Scx1DecodeMetadata,
    row_start: usize,
    n_rows: usize,
) -> Result<DecodedShard, CodecError> {
    if !value_encoding.is_integer() {
        return Err(CodecError::FloatWithScx1);
    }
    let row_end = row_start
        .checked_add(n_rows)
        .ok_or_else(|| CodecError::MalformedInput("Scx1 row range end overflows usize".into()))?;
    if row_end > metadata.rows.len() {
        return Err(CodecError::MalformedInput(format!(
            "Scx1 row range [{row_start}, {row_end}) exceeds shard rows {}",
            metadata.rows.len()
        )));
    }
    let row_slice = &metadata.rows[row_start..row_end];

    // Shard-local indptr (starts at 0) + window nnz.
    let mut indptr = Vec::with_capacity(n_rows + 1);
    indptr.push(0u64);
    let mut window_nnz = 0u64;
    for row in row_slice {
        window_nnz = window_nnz
            .checked_add(row.nnz as u64)
            .ok_or_else(|| CodecError::MalformedInput("Scx1 row-range nnz overflows u64".into()))?;
        indptr.push(window_nnz);
    }
    let window_nnz = window_nnz as usize;

    // Indices: the row sub-slice decodes directly (absolute bit offsets).
    let indices = forbp_decode_with_metadata(encoded.indices_bytes, row_slice)?;
    if indices.len() != window_nnz {
        return Err(CodecError::MalformedInput(format!(
            "Scx1 row-range decoded {} indices, expected {window_nnz}",
            indices.len()
        )));
    }

    // Values: map the window's value ordinal range to the covering Rice blocks.
    let values_u32 = if window_nnz == 0 {
        Vec::new()
    } else {
        let v0 = row_slice[0].value_start;
        let v1 = v0 + window_nnz as u64;
        let blocks = &metadata.rice_blocks;
        // First block covering v0 (largest start <= v0) and exclusive end (first
        // block whose start is >= v1). Blocks are sorted by value_start.
        let b0 = blocks
            .partition_point(|b| b.value_start <= v0)
            .saturating_sub(1);
        let b1 = blocks.partition_point(|b| b.value_start < v1);
        let covered = &blocks[b0..b1];
        let block_base = covered[0].value_start;
        let rebased: Vec<RiceBlockMetadata> = covered
            .iter()
            .map(|b| RiceBlockMetadata {
                value_start: b.value_start - block_base,
                n_values: b.n_values,
                bit_offset: b.bit_offset,
                k: b.k,
            })
            .collect();
        let decoded = rice_decode_with_metadata(encoded.values_bytes, &rebased)?;
        let head = (v0 - block_base) as usize;
        if head + window_nnz > decoded.len() {
            return Err(CodecError::MalformedInput(format!(
                "Scx1 row-range value window [{head}, {}) exceeds covered block decode {}",
                head + window_nnz,
                decoded.len()
            )));
        }
        decoded[head..head + window_nnz].to_vec()
    };
    let values_bytes = u32_to_raw_bytes(&values_u32, value_encoding)?;

    Ok((indptr, indices, values_bytes))
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
        scx1_decode: None,
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
        scx1_decode: None,
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

    let indptr_raw = byte_unshuffle(&indptr_shuffled, 8)?;
    let index_width = if index_dtype_u16 { 2 } else { 4 };
    let indices_raw = byte_unshuffle(&indices_shuffled, index_width)?;
    let values_raw = byte_unshuffle(&values_shuffled, value_encoding.byte_width())?;

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
        scx1_decode: None,
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
///
/// `data` is a writer-side buffer sized `n_values × width`, so a ragged tail
/// (length not a multiple of the element width) is an invariant violation, not
/// expected input. Use `chunks_exact` and reject the remainder with
/// [`CodecError::MalformedInput`] rather than silently dropping the partial
/// element the way a `while let Ok(read_…)` loop did (finding F8 — the
/// always-on form of the raggedness guard).
fn raw_bytes_to_u32(data: &[u8], encoding: ValueEncoding) -> Result<Vec<u32>, CodecError> {
    match encoding {
        ValueEncoding::Uint8 => Ok(data.iter().map(|&b| b as u32).collect()),
        ValueEncoding::Uint16 => {
            let chunks = data.chunks_exact(2);
            if !chunks.remainder().is_empty() {
                return Err(CodecError::MalformedInput(format!(
                    "Uint16 value buffer length {} is not a multiple of 2",
                    data.len()
                )));
            }
            Ok(chunks
                .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
                .collect())
        }
        ValueEncoding::Uint32 => {
            let chunks = data.chunks_exact(4);
            if !chunks.remainder().is_empty() {
                return Err(CodecError::MalformedInput(format!(
                    "Uint32 value buffer length {} is not a multiple of 4",
                    data.len()
                )));
            }
            Ok(chunks
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
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

    /// F8: a value buffer whose length is not a multiple of the element width
    /// is rejected, not silently truncated to drop the partial element.
    #[test]
    fn raw_bytes_to_u32_rejects_ragged_input() {
        // 3 bytes is not a multiple of 2 (Uint16) or 4 (Uint32).
        assert!(matches!(
            raw_bytes_to_u32(&[1, 2, 3], ValueEncoding::Uint16),
            Err(CodecError::MalformedInput(_))
        ));
        assert!(matches!(
            raw_bytes_to_u32(&[1, 2, 3], ValueEncoding::Uint32),
            Err(CodecError::MalformedInput(_))
        ));
        // Exact multiples decode fine.
        assert_eq!(
            raw_bytes_to_u32(&[1, 0, 2, 0], ValueEncoding::Uint16).unwrap(),
            vec![1, 2]
        );
        assert_eq!(
            raw_bytes_to_u32(&[5, 0, 0, 0], ValueEncoding::Uint32).unwrap(),
            vec![5]
        );
    }

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
            scx1_decode: None,
        };
        let result = decode_shard(&encoded, CodecId::Scx1, ValueEncoding::Float32, 3, 6, false);
        assert!(matches!(result, Err(CodecError::FloatWithScx1)));
    }

    /// The encoder-emitted Scx1 decode metadata must reproduce the canonical
    /// decode when fed to `decode_scx1_with_metadata` (parity), and a corrupted
    /// offset must NOT reproduce it (proving the open-verify parity net catches
    /// a bad sidecar — either an error or a divergent decode).
    #[test]
    fn decode_scx1_with_metadata_parity_and_detects_corruption() {
        use crate::value_encoding::values_to_raw_bytes;

        // 3 rows, strictly-increasing indices, non-zero integer values.
        let indptr: Vec<u64> = vec![0, 2, 2, 5];
        let indices: Vec<u32> = vec![0, 3, 1, 4, 9];
        let vals_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (n_rows, nnz) = (3usize, 5usize);
        let value_encoding = ValueEncoding::Uint16;
        let values = values_to_raw_bytes(&vals_f32, value_encoding).unwrap();

        let encoded = encode_shard(
            &indptr,
            &indices,
            &values,
            CodecId::Scx1,
            value_encoding,
            false,
        )
        .unwrap();
        let meta = encoded
            .scx1_decode
            .clone()
            .expect("Scx1 encode must emit decode metadata");
        let r = EncodedShardRef {
            indptr_bytes: &encoded.indptr_bytes,
            indices_bytes: &encoded.indices_bytes,
            values_bytes: &encoded.values_bytes,
        };
        let canonical =
            decode_shard_ref(&r, CodecId::Scx1, value_encoding, n_rows, nnz, false).unwrap();

        // Parity: decode-via-offsets == canonical decode.
        let via = decode_scx1_with_metadata(&r, value_encoding, n_rows, nnz, &meta).unwrap();
        assert_eq!(via, canonical, "decode-via-metadata must match canonical");

        // Corrupt a Rice value-block bit offset → must error or diverge.
        let mut bad_rice = meta.clone();
        bad_rice.rice_blocks[0].bit_offset = bad_rice.rice_blocks[0].bit_offset.wrapping_add(5);
        let got = decode_scx1_with_metadata(&r, value_encoding, n_rows, nnz, &bad_rice);
        assert!(
            got.map_or(true, |d| d != canonical),
            "corrupted Rice block offset must not reproduce the canonical decode"
        );

        // Corrupt a row's index bit offset → must error or diverge.
        let mut bad_idx = meta.clone();
        if let Some(row) = bad_idx.rows.iter_mut().find(|row| row.nnz > 0) {
            row.indices_bit_offset = row.indices_bit_offset.wrapping_add(7);
        }
        let got = decode_scx1_with_metadata(&r, value_encoding, n_rows, nnz, &bad_idx);
        assert!(
            got.map_or(true, |d| d != canonical),
            "corrupted index bit offset must not reproduce the canonical decode"
        );
    }

    /// `decode_scx1_row_range` over any window must be byte-identical to the
    /// corresponding slice of the canonical full decode — across Rice-block
    /// (256) and FOR-BP SIMD (≥128 nnz) boundaries, leading empty rows, the
    /// last partial block, single-row, and empty windows.
    #[test]
    fn decode_scx1_row_range_matches_canonical_slice() {
        use crate::value_encoding::values_to_raw_bytes;

        // Row nnz pattern mixing: empties, tiny rows, FOR-BP SIMD rows (≥128),
        // and a total nnz spanning several 256-value Rice blocks.
        let row_nnz: Vec<usize> = vec![
            0, 1, 5, 130, 0, 200, 7, 256, 257, 3, 0, 140, 511, 1, 64, 300, 0, 2, 128, 90,
        ];
        let n_cols = 4000u32;
        let mut indptr = vec![0u64];
        let mut indices: Vec<u32> = Vec::new();
        let mut vals_f32: Vec<f32> = Vec::new();
        for &m in &row_nnz {
            for j in 0..m {
                indices.push(j as u32); // strictly increasing, < n_cols
                vals_f32.push((1 + (j % 97)) as f32); // non-zero (Scx1 Rice requires ≥1)
            }
            indptr.push(indices.len() as u64);
        }
        let n_rows = row_nnz.len();
        let nnz = indices.len();
        let venc = ValueEncoding::Uint16;
        let bw = venc.byte_width();
        let values = values_to_raw_bytes(&vals_f32, venc).unwrap();

        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Scx1, venc, false).unwrap();
        let meta = encoded.scx1_decode.clone().expect("Scx1 emits metadata");
        let r = EncodedShardRef {
            indptr_bytes: &encoded.indptr_bytes,
            indices_bytes: &encoded.indices_bytes,
            values_bytes: &encoded.values_bytes,
        };
        let (full_indptr, full_indices, full_values) =
            decode_shard_ref(&r, CodecId::Scx1, venc, n_rows, nnz, false).unwrap();

        // Exhaustively check every contiguous window [s, s+k].
        for s in 0..=n_rows {
            for k in 0..=(n_rows - s) {
                let (w_indptr, w_indices, w_values) =
                    decode_scx1_row_range(&r, venc, &meta, s, k).unwrap();

                // indptr: local, starts at 0, equals the full indptr slice rebased.
                let base = full_indptr[s];
                let expect_indptr: Vec<u64> =
                    full_indptr[s..=s + k].iter().map(|&p| p - base).collect();
                assert_eq!(w_indptr, expect_indptr, "indptr window s={s} k={k}");

                let lo = full_indptr[s] as usize;
                let hi = full_indptr[s + k] as usize;
                assert_eq!(
                    w_indices,
                    full_indices[lo..hi],
                    "indices window s={s} k={k}"
                );
                assert_eq!(
                    w_values,
                    full_values[lo * bw..hi * bw],
                    "values window s={s} k={k}"
                );
            }
        }

        // Full-shard window equals the whole canonical decode.
        let (fi, fx, fv) = decode_scx1_row_range(&r, venc, &meta, 0, n_rows).unwrap();
        assert_eq!((fi, fx, fv), (full_indptr, full_indices, full_values));

        // Out-of-range is rejected, not a panic.
        assert!(decode_scx1_row_range(&r, venc, &meta, n_rows, 1).is_err());
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
    fn test_u64_to_i64_rejects_overflow() {
        let data = vec![0u64, 100, u64::MAX];
        match u64_vec_to_i64(data) {
            Err(CodecError::MalformedInput(msg)) => {
                assert!(msg.contains("exceeds i64::MAX"), "got: {msg}");
            }
            other => panic!("expected MalformedInput, got {other:?}"),
        }
    }

    #[test]
    fn test_u32_to_i32_cast_valid() {
        let data = vec![0u32, 100, i32::MAX as u32];
        let result = u32_vec_to_i32(data).unwrap();
        assert_eq!(result, vec![0i32, 100, i32::MAX]);
    }

    #[test]
    fn test_u32_to_i32_rejects_overflow() {
        let data = vec![0u32, 100, u32::MAX];
        match u32_vec_to_i32(data) {
            Err(CodecError::MalformedInput(msg)) => {
                assert!(msg.contains("exceeds i32::MAX"), "got: {msg}");
            }
            other => panic!("expected MalformedInput, got {other:?}"),
        }
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

    #[test]
    fn test_codec_id_parse_cli() {
        assert_eq!(CodecId::parse_cli("auto").unwrap(), None);
        assert_eq!(CodecId::parse_cli("none").unwrap(), Some(CodecId::None));
        assert_eq!(CodecId::parse_cli("scx1").unwrap(), Some(CodecId::Scx1));
        assert_eq!(CodecId::parse_cli("zstd").unwrap(), Some(CodecId::Zstd));
        assert_eq!(
            CodecId::parse_cli("lz4").unwrap(),
            Some(CodecId::Lz4Shuffle)
        );
        assert_eq!(CodecId::parse_cli("pcodec").unwrap(), Some(CodecId::Pcodec));
        assert!(CodecId::parse_cli("gzip").is_err());
    }

    #[test]
    fn test_codec_id_display_name() {
        assert_eq!(CodecId::None.display_name(), "none");
        assert_eq!(CodecId::Scx1.display_name(), "scx1");
        assert_eq!(CodecId::Zstd.display_name(), "zstd");
        assert_eq!(CodecId::Lz4Shuffle.display_name(), "lz4+shuffle");
        assert_eq!(CodecId::Pcodec.display_name(), "pcodec");
        // Every explicit CLI codec parses back to a value whose display name
        // is stable (round-trip guard so the two maps can't drift).
        for s in ["none", "scx1", "zstd", "pcodec"] {
            let c = CodecId::parse_cli(s).unwrap().unwrap();
            assert_eq!(c.display_name(), s);
        }
    }
}
