// Codec ID dispatch + zstd fallback (docs/codec.md (Codec IDs))

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Cursor;
use std::ops::Range;

use crate::bitstream::BitStreamError;
use crate::byte_delta::{byte_delta_planes, byte_undelta_planes};
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
    /// Byte-shuffle + byte-delta (indices/indptr only) + zstd. Ported from
    /// Monolithic per shard (F5 Phase 0 measurement spike — not row-group framed).
    ShufDeltaZstd = 5,
}

impl CodecId {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::Scx1),
            2 => Some(Self::Zstd),
            3 => Some(Self::Lz4Shuffle),
            4 => Some(Self::Pcodec),
            5 => Some(Self::ShufDeltaZstd),
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
            "shufdelta" => Ok(Some(CodecId::ShufDeltaZstd)),
            other => Err(format!(
                "unknown codec: '{other}'. Use auto, none, scx1, zstd, lz4, pcodec, or shufdelta."
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
            CodecId::ShufDeltaZstd => "shufdelta",
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
                // Inclusive at 2³², and deliberately so. `u32::MAX as f32` IS
                // 2³² — the conversion rounds up — so this one f32 value has
                // two provenances the encoder cannot tell apart: a genuine
                // out-of-range 2³², or the f32 image of an on-disk `u32::MAX`
                // that a rewrite path (compact / merge / sort / build_csc)
                // just decoded and is handing straight back. Rejecting it
                // would fail those ops on format-valid archives, so accept and
                // let `as u32` saturate — which is the *correct* answer for
                // the decode-seam provenance.
                //
                // On the *detect* path, fresh out-of-range data is caught
                // upstream by `detect_value_encoding`, which sends anything
                // above `u32::MAX` to `Float32` so this arm is never selected
                // for it. `attach_external_layer`, which re-derives each
                // shard's encoding after canonicalizing, applies that same
                // fresh-data rule and so never selects this arm either.
                //
                // `scx_ops::encoding_for_canonicalized` is the deliberate
                // exception and arrives here **on purpose**: its values may
                // have come off disk, where an f32 of exactly 2³² is the image
                // of a stored `u32::MAX`, so it keeps `Uint32` precisely to get
                // the saturation above — which restores the original value.
                // Do not "fix" it to divert at 2³²; this arm is what makes a
                // rewrite of such an archive lossless. Strictly above 2³² it
                // does divert, since no `u32` decodes there.
                //
                // What remains: a caller passing an explicit encoding bypasses
                // every detector and still saturates.
                //
                // Separately, this arm cannot see the loss that happens
                // *before* it. The rewrite paths decode integer shards to
                // `f32`, so any on-disk `u32` above 2²⁴ is already rounded by
                // the time it arrives; 2³² saturation is the tail of that, not
                // its own bug. Closing it needs the rewrite paths to carry
                // native `u32` instead of round-tripping through `f32`, which
                // is a separate change.
                //
                // `contains` (not `<=`) so NaN is rejected rather than written
                // as 0.
                const UINT32_BOUND: f32 = (1u128 << 32) as f32;
                if !(0.0..=UINT32_BOUND).contains(&value) {
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

    #[error("malformed codec input: {0}")]
    MalformedInput(String),

    /// A decoded minor-axis index is at or past the caller-supplied bound.
    ///
    /// Typed rather than folded into [`CodecError::MalformedInput`] so the
    /// reader can map it onto `ScxError::ShardIndexOutOfRange` and have it
    /// classify as `CorruptFile`. Without a distinct variant the scipy path
    /// (where this bound rides on the existing `i32::MAX` scan) and the native
    /// path (which runs its own pass) would report the same corruption as two
    /// different error classes — `RuntimeError` on one and `ValueError` on the
    /// other, from the same broken file.
    #[error(
        "column index {index} at position {position} is out of range for n_minor \
         {bound} (corrupt or truncated shard payload)"
    )]
    IndexOutOfRange {
        index: u32,
        position: usize,
        bound: u32,
    },
}

/// Decoded shard: `(indptr, indices, values_raw_bytes)`.
pub type DecodedShard = (Vec<u64>, Vec<u32>, Vec<u8>);

/// A resolved, validated row-group within a framed (v4/shard-v2) shard.
///
/// Produced by `scx_format::resolve_block_index` from the on-disk `BlockIndex`;
/// consumed by [`decode_row_group`]. The three `Range<usize>` are byte ranges
/// **into each sub-stream** (indptr / indices / values), inferred from the
/// per-entry offsets (`[offset[g], offset[g+1])`, last = stream length). This is
/// the codec-agnostic random-access unit underneath F5-b: a group decodes
/// independently to a *local* CSR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowGroupSpan {
    /// First (shard-local) row covered by this group.
    pub row_start: u32,
    /// Number of rows in this group.
    pub n_rows: u16,
    /// Non-zeros in this group (== decoded indptr's last value).
    pub nnz: u32,
    /// Byte range of this group's frame within the indptr sub-stream.
    pub indptr: Range<usize>,
    /// Byte range of this group's frame within the indices sub-stream.
    pub indices: Range<usize>,
    /// Byte range of this group's frame within the values sub-stream.
    pub values: Range<usize>,
}

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
        CodecId::ShufDeltaZstd => {
            encode_shufdelta_zstd(indptr, indices, values, value_encoding, index_dtype_u16)
        }
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
    let decoded = match codec_id {
        CodecId::None => decode_none_ref(encoded, n_rows, nnz, value_encoding, index_dtype_u16),
        CodecId::Scx1 => decode_scx1_ref(encoded, value_encoding, n_rows, nnz, index_dtype_u16),
        CodecId::Zstd => decode_zstd_ref(encoded, n_rows, nnz, value_encoding, index_dtype_u16),
        CodecId::Lz4Shuffle => {
            decode_lz4_shuffle_ref(encoded, n_rows, nnz, value_encoding, index_dtype_u16)
        }
        CodecId::Pcodec => decode_pcodec_ref(encoded, n_rows, nnz, value_encoding, index_dtype_u16),
        CodecId::ShufDeltaZstd => {
            decode_shufdelta_zstd_ref(encoded, n_rows, nnz, value_encoding, index_dtype_u16)
        }
    }?;
    // Single structural gate for every codec, and for every row group, since
    // `decode_row_group` funnels through here. Values are still raw bytes at
    // this point, so they get their own byte-exact check — dividing by the
    // element width would round a ragged length down to a passing element
    // count. Several arms already check this; those become belt-and-braces.
    let (indptr, indices, values) = &decoded;
    let expected_value_bytes = checked_len(nnz, value_encoding.byte_width(), "values")?;
    if values.len() != expected_value_bytes {
        return Err(CodecError::MalformedInput(format!(
            "shard decoded {} value bytes != declared nnz {nnz} * {} bytes/element",
            values.len(),
            value_encoding.byte_width()
        )));
    }
    check_decoded_shape(indptr, indices.len(), nnz, n_rows, nnz)?;
    Ok(decoded)
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
    index_bound: u32,
) -> Result<ScipyShard, CodecError> {
    // For Scx1, we can avoid the u32→raw_bytes→f32 chain for values
    if codec_id == CodecId::Scx1 {
        if !value_encoding.is_integer() {
            return Err(CodecError::FloatWithScx1);
        }

        // Same L1 overflow + L2 plausibility guards as `decode_scx1_ref`, so
        // every Scx1 decode entry point rejects hostile headers up front (F-f).
        indptr_byte_cap(n_rows)?;
        checked_len(nnz, if index_dtype_u16 { 2 } else { 4 }, "scx1 indices")?;
        checked_len(nnz, value_encoding.byte_width(), "scx1 values")?;
        let n_rows_p1 = n_rows.checked_add(1).ok_or_else(|| {
            CodecError::MalformedInput(format!("scx1 n_rows+1 overflow: {n_rows}"))
        })?;
        bound_capacity(n_rows_p1, encoded.indptr_bytes.len(), "scx1 indptr")?;
        bound_capacity(nnz, encoded.indices_bytes.len(), "scx1 indices")?;
        bound_capacity(nnz, encoded.values_bytes.len(), "scx1 values")?;

        // indptr: delta_golomb → Vec<u64> → Vec<i64>
        let indptr_u64 = delta_golomb_decode(encoded.indptr_bytes, n_rows_p1)?;

        // indices: forbp → Vec<u32> → Vec<i32>
        let (indices_u32, _) =
            forbp_decode_with_hint(encoded.indices_bytes, n_rows, nnz, index_dtype_u16)?;

        // values: rice → Vec<u32> → Vec<f32> directly (skip raw bytes intermediate)
        let values_u32 = rice_decode(encoded.values_bytes, nnz, B_VAL)?;

        // This path short-circuits Scx1 and never reaches `decode_shard_ref`,
        // so it needs its own structural gate. Check before the conversions —
        // `u64_vec_to_i64` consumes the indptr.
        check_decoded_shape(
            &indptr_u64,
            indices_u32.len(),
            values_u32.len(),
            n_rows,
            nnz,
        )?;

        let indptr = u64_vec_to_i64(indptr_u64)?;
        let indices = u32_vec_to_i32_bounded(indices_u32, index_bound)?;
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
    let indices = u32_vec_to_i32_bounded(indices_u32, index_bound)?;
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
    index_bound: u32,
) -> Result<ScipyShard, CodecError> {
    let (indptr_u64, indices_u32, values_raw) = decoded;
    let indptr = u64_vec_to_i64(indptr_u64)?;
    let indices = u32_vec_to_i32_bounded(indices_u32, index_bound)?;
    let data = values_raw_to_f32(&values_raw, value_encoding);
    Ok((indptr, indices, data))
}

/// A shard's decoded values kept at their **native** width: integer-encoded
/// shards stay `u32` (never rounded through `f32`), float-encoded shards stay
/// `f32`. This is the value carrier for the in-assembly narrow read path — the
/// caller casts each variant into the target dtype via `scx_codec`'s
/// `checked_cast_u32_into` / `checked_cast_f32_into`, so an integer count above
/// 2²⁴ narrows to an exact integer dtype losslessly.
pub enum ShardValuesNative {
    /// Integer-encoded shard values (`Uint8`/`Uint16`/`Uint32`), widened to `u32`.
    U32(Vec<u32>),
    /// Float-encoded shard values (`Float32`/`Float16`), decoded to `f32`.
    F32(Vec<f32>),
}

/// A shard decoded to native types: `i64` indptr, `u32` indices, and
/// [`ShardValuesNative`] values.
pub type NativeShard = (Vec<i64>, Vec<u32>, ShardValuesNative);

/// Decode an `EncodedShardRef` to native types (the in-assembly narrow twin of
/// [`decode_shard_scipy`]).
///
/// Unlike the scipy path, integer values are kept as `u32` (not cast to `f32`)
/// and indices are kept as `u32` (not cast to `i32`), so the caller can narrow
/// directly to the requested dtype through the fail-loud native cast gate. Only
/// float-encoded shards produce `f32` values (there is no lossless integer form
/// for them).
pub fn decode_shard_native(
    encoded: &EncodedShardRef,
    codec_id: CodecId,
    value_encoding: ValueEncoding,
    n_rows: usize,
    nnz: usize,
    index_dtype_u16: bool,
) -> Result<NativeShard, CodecError> {
    // Scx1: keep the Rice-decoded u32 values and forbp u32 indices as-is.
    if codec_id == CodecId::Scx1 {
        if !value_encoding.is_integer() {
            return Err(CodecError::FloatWithScx1);
        }

        // Same L1 overflow + L2 plausibility guards as `decode_shard_scipy`.
        indptr_byte_cap(n_rows)?;
        checked_len(nnz, if index_dtype_u16 { 2 } else { 4 }, "scx1 indices")?;
        checked_len(nnz, value_encoding.byte_width(), "scx1 values")?;
        let n_rows_p1 = n_rows.checked_add(1).ok_or_else(|| {
            CodecError::MalformedInput(format!("scx1 n_rows+1 overflow: {n_rows}"))
        })?;
        bound_capacity(n_rows_p1, encoded.indptr_bytes.len(), "scx1 indptr")?;
        bound_capacity(nnz, encoded.indices_bytes.len(), "scx1 indices")?;
        bound_capacity(nnz, encoded.values_bytes.len(), "scx1 values")?;

        let indptr_u64 = delta_golomb_decode(encoded.indptr_bytes, n_rows_p1)?;
        let (indices, _) =
            forbp_decode_with_hint(encoded.indices_bytes, n_rows, nnz, index_dtype_u16)?;
        let values_u32 = rice_decode(encoded.values_bytes, nnz, B_VAL)?;

        // Own structural gate: like the scipy twin, this arm short-circuits
        // Scx1 and never reaches `decode_shard_ref`.
        check_decoded_shape(&indptr_u64, indices.len(), values_u32.len(), n_rows, nnz)?;

        let indptr = u64_vec_to_i64(indptr_u64)?;
        return Ok((indptr, indices, ShardValuesNative::U32(values_u32)));
    }

    // None / Zstd / Lz4Shuffle / ShufDeltaZstd / Pcodec: decode to raw types,
    // then widen integer bytes to u32 (or decode float bytes to f32).
    let (indptr_u64, indices, values_raw) = decode_shard_ref(
        encoded,
        codec_id,
        value_encoding,
        n_rows,
        nnz,
        index_dtype_u16,
    )?;
    let indptr = u64_vec_to_i64(indptr_u64)?;
    let values = decoded_values_to_native(&values_raw, value_encoding)?;
    Ok((indptr, indices, values))
}

/// Convert a raw [`DecodedShard`] into the native `(i64, u32, ShardValuesNative)`
/// triple (the in-assembly narrow twin of [`decoded_shard_to_scipy`]), used by
/// the framed per-row-group decode path.
pub fn decoded_shard_to_native(
    decoded: DecodedShard,
    value_encoding: ValueEncoding,
) -> Result<NativeShard, CodecError> {
    let (indptr_u64, indices, values_raw) = decoded;
    let indptr = u64_vec_to_i64(indptr_u64)?;
    let values = decoded_values_to_native(&values_raw, value_encoding)?;
    Ok((indptr, indices, values))
}

/// Raw value bytes → native values per encoding: integer → widened `u32`,
/// float → `f32`.
fn decoded_values_to_native(
    values_raw: &[u8],
    value_encoding: ValueEncoding,
) -> Result<ShardValuesNative, CodecError> {
    if value_encoding.is_integer() {
        Ok(ShardValuesNative::U32(raw_bytes_to_u32(
            values_raw,
            value_encoding,
        )?))
    } else {
        Ok(ShardValuesNative::F32(values_raw_to_f32(
            values_raw,
            value_encoding,
        )))
    }
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
    // Guard the `+ 1` so a `usize::MAX` `n_rows` can't wrap before it reaches
    // the sub-stream decoders (parity with the other Scx1 entry points — F-f).
    let n_rows_p1 = n_rows
        .checked_add(1)
        .ok_or_else(|| CodecError::MalformedInput(format!("n_rows+1 overflow: {n_rows}")))?;
    let indptr_u64: Vec<u64> = match codec_id {
        CodecId::None => le_bytes_to_u64(indptr_bytes, n_rows_p1)?,
        CodecId::Scx1 => delta_golomb_decode(indptr_bytes, n_rows_p1)?,
        CodecId::Zstd | CodecId::Pcodec => {
            let raw = zstd_decode_bounded(indptr_bytes, indptr_byte_cap(n_rows)?)?;
            le_bytes_to_u64(&raw, n_rows_p1)?
        }
        CodecId::Lz4Shuffle => {
            let shuffled = lz4_frame_decompress(indptr_bytes, indptr_byte_cap(n_rows)?)?;
            let raw = byte_unshuffle(&shuffled, 8)?;
            le_bytes_to_u64(&raw, n_rows_p1)?
        }
        CodecId::ShufDeltaZstd => {
            let indptr_max = indptr_byte_cap(n_rows)?;
            let mut planes = zstd_decode_bounded(indptr_bytes, indptr_max)?;
            expect_exact_len(planes.len(), indptr_max, "indptr")?;
            byte_undelta_planes(&mut planes, 8, n_rows_p1);
            let raw = byte_unshuffle(&planes, 8)?;
            le_bytes_to_u64(&raw, n_rows_p1)?
        }
    };
    let indptr = u64_vec_to_i64(indptr_u64)?;
    // Every caller decodes a shard-local or group-local indptr, so a zero start
    // and monotonicity both hold for any honest stream. Checked here rather than
    // per caller because this is the only seam the direct-to-device GPU decoders
    // pass through — they never reach `check_decoded_shape`, and a `GpuCsr` whose
    // indptr addresses past its own `indices` is handed straight to cuSPARSE /
    // `cupyx.sparse.csr_matrix`, which walk it exactly as `csr_to_csc` does.
    // `nnz` is not known here; the whole-shard callers check it themselves.
    check_indptr_shape(&indptr, n_rows, None)?;
    Ok(indptr)
}

/// Decode a single row-group of a framed (v4/shard-v2) shard to a **local** CSR.
///
/// Codec-agnostic random-access primitive. The
/// three `*_bytes` slices are the shard's *whole* sub-streams; `span` carries the
/// byte ranges of this group's frame within each. Returns a group-local
/// [`DecodedShard`]: `indptr.len() == n_rows+1`, `indptr[0] == 0`,
/// `indptr.last() == nnz`, `indices.len() == values.len()/width == nnz`.
///
/// Works for every codec (None / ShufDeltaZstd / Zstd / Lz4Shuffle / Pcodec /
/// Scx1): a group is a standalone encoded sub-shard, so this delegates to the
/// ordinary [`decode_shard_ref`] over the group's three byte frames.
pub fn decode_row_group(
    codec_id: CodecId,
    span: &RowGroupSpan,
    indptr_bytes: &[u8],
    indices_bytes: &[u8],
    values_bytes: &[u8],
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<DecodedShard, CodecError> {
    let ip = slice_span(indptr_bytes, &span.indptr, "indptr")?;
    let ix = slice_span(indices_bytes, &span.indices, "indices")?;
    let vv = slice_span(values_bytes, &span.values, "values")?;
    let n_rows = span.n_rows as usize;
    let nnz = span.nnz as usize;

    // A row-group is a standalone encoded sub-shard with a group-local indptr, so
    // decode is just the ordinary per-shard decoder over the group's three byte
    // frames — codec-agnostic (None / ShufDeltaZstd / Zstd / Lz4Shuffle / Pcodec /
    // Scx1) with no per-codec code here.
    let enc = EncodedShardRef {
        indptr_bytes: ip,
        indices_bytes: ix,
        values_bytes: vv,
    };
    // `decode_shard_ref` applies the full shape gate, which subsumes the
    // framed wire invariant this used to check by hand ("each group decodes to
    // a local CSR": `indptr[0] == 0`, `indptr.last() == nnz`) and adds the
    // index/value lengths it did not check. Re-label the message so a framed
    // shard still says which group failed.
    //
    // `BitStream` is relabelled too, not just `MalformedInput`. The corruption
    // this gate exists for — a FOR-BP stream shorter than the declared nnz —
    // fails inside `forbp_decode_with_hint`, whose error carries no message and
    // so arrives as `BitStream`. Prefixing only `MalformedInput` left exactly
    // the primary case anonymous. `IndexOutOfRange` is deliberately *not*
    // folded in: it is a typed variant that `ScxError` maps to
    // `ShardIndexOutOfRange` / `CorruptFile`, and flattening it here would
    // downgrade a corrupt-file report to a generic codec error.
    decode_shard_ref(&enc, codec_id, value_encoding, n_rows, nnz, index_dtype_u16).map_err(|e| {
        match e {
            CodecError::MalformedInput(_) | CodecError::BitStream(_) => {
                CodecError::MalformedInput(format!("row-group at row {}: {e}", span.row_start))
            }
            other => other,
        }
    })
}

/// Decode **only** a row-group's local indptr (F-b). Mirrors [`decode_row_group`]
/// but slices and decodes just the indptr sub-stream frame — the indices/values
/// frames are never touched — for callers that need per-row offsets without the
/// data (e.g. assembling a shard's global indptr). Returns the group-local i64
/// indptr (`len == n_rows+1`, `[0] == 0`, `last == nnz`). Codec-agnostic.
pub fn decode_row_group_indptr_only(
    codec_id: CodecId,
    span: &RowGroupSpan,
    indptr_bytes: &[u8],
) -> Result<Vec<i64>, CodecError> {
    let ip = slice_span(indptr_bytes, &span.indptr, "indptr")?;
    let indptr = decode_indptr_only(ip, codec_id, span.n_rows as usize)?;
    // `decode_indptr_only` has already checked length, zero start and
    // monotonicity; this adds the group's declared nnz, which only the span
    // knows. Was a hand-rolled `first`/`last` pair that omitted monotonicity —
    // and since every framed GPU assembler builds its combined indptr out of
    // this function, that omission was the whole hole.
    check_indptr_shape(&indptr, span.n_rows as usize, Some(span.nnz as usize)).map_err(|e| {
        CodecError::MalformedInput(format!("row-group at row {}: {e}", span.row_start))
    })?;
    Ok(indptr)
}

/// Bounds-checked slice of a sub-stream by a resolved byte range.
fn slice_span<'a>(
    bytes: &'a [u8],
    range: &Range<usize>,
    which: &str,
) -> Result<&'a [u8], CodecError> {
    bytes.get(range.clone()).ok_or_else(|| {
        CodecError::MalformedInput(format!(
            "row-group {which} range {}..{} out of bounds (stream len {})",
            range.start,
            range.end,
            bytes.len()
        ))
    })
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

/// Convert Vec<u32> to Vec<i32> via zero-copy reinterpretation, rejecting any
/// index at or above `bound`.
///
/// # Why the bound rides along here
///
/// This scan already existed, to keep a `> i32::MAX` value from reinterpreting
/// to a *negative* `i32`. In the good case it walks every element and returns
/// `None`, so the caller's minor-axis bound check costs nothing extra when it
/// rides on the same pass — only the comparand changes. Measured as a standalone
/// pass instead, the same check cost **+5.1–6.4%** of per-shard decode
/// (194.9M nnz, memory-bandwidth-bound at 10.0 GB/s, so not optimisable in
/// place). Folding it in is what makes it free on the hot path.
///
/// `bound` is the shard's `n_minor`, clamped by the caller to at most
/// `i32::MAX as u32 + 1` so the original sign guarantee still holds. A caller
/// with no bound to enforce (a decode that is not addressing a column axis)
/// passes exactly that clamp value and gets the pre-existing behaviour.
fn u32_vec_to_i32_bounded(data: Vec<u32>, bound: u32) -> Result<Vec<i32>, CodecError> {
    // Clamp internally rather than trusting the caller. A `bound` above the sign
    // limit would otherwise *widen* the check past what this function guaranteed
    // before it took a bound at all, letting a caller disable the sign guard by
    // accident. `clamp_index_bound` already does this for callers that use it;
    // doing it here too means no caller can get it wrong.
    let bound = bound.min(NO_INDEX_BOUND);
    if let Some((position, &bad)) = data.iter().enumerate().find(|&(_, &v)| v >= bound) {
        // Classify by whether a real column bound was declared, NOT by whether
        // the offending value also happens to exceed `i32::MAX`.
        //
        // Getting this backwards reintroduced the exact defect the typed
        // `IndexOutOfRange` variant exists to remove: with a declared bound, a
        // value ≥ 2^31 took the sign branch, so the scipy path reported
        // `MalformedInput` → `ScxError::Codec` → `RuntimeError` while the native
        // path reported `ShardIndexOutOfRange` → `CorruptFile` → `ValueError`
        // for the same payload. The Python exception type depended on whether
        // the caller asked for `f32` or a narrowed dtype.
        //
        // With a bound present, an out-of-range index is an out-of-range index
        // at any magnitude. The legacy sign-only message survives for
        // `NO_INDEX_BOUND`, where there is no column axis to be out of.
        //
        // A declared width at or above 2^31 clamps to the same sentinel, so it
        // takes the sign branch too. That is accurate rather than a collision:
        // an `i32` CSR cannot represent such a column at all, so "exceeds
        // i32::MAX" is the actual reason for the rejection, and naming the
        // declared width instead would describe a bound that is not what
        // stopped it. The native (`u32`) path legitimately accepts the same
        // index, because it has no sign hazard to begin with — the two domains
        // differ there because the *representations* differ, not because the
        // classification is inconsistent.
        return Err(if bound == NO_INDEX_BOUND {
            CodecError::MalformedInput(format!(
                "column index {bad} exceeds i32::MAX (corrupt or hostile input)"
            ))
        } else {
            CodecError::IndexOutOfRange {
                index: bad,
                position,
                bound,
            }
        });
    }
    Ok(bytemuck::cast_vec::<u32, i32>(data))
}

/// The bound that reproduces the pre-existing behaviour: reject only what would
/// reinterpret to a negative `i32`. Callers that know the shard's `n_minor` pass
/// [`clamp_index_bound`] instead.
pub const NO_INDEX_BOUND: u32 = i32::MAX as u32 + 1;

/// Clamp a shard's `n_minor` into a usable index bound.
///
/// `n_minor == 0` means the shard header does not *declare* a column axis — old
/// writers stamped the file-level `n_vars`, which is `0` on a multimodal file
/// because the real count is per-modality. Two multimodal conformance fixtures
/// carry `n_minor = 0` on shards with hundreds of nonzeros, so treating it as a
/// bound would reject valid files. Anything above `i32::MAX` is clamped down to
/// preserve the sign guarantee.
pub fn clamp_index_bound(n_minor: u32) -> u32 {
    if n_minor == 0 {
        NO_INDEX_BOUND
    } else {
        n_minor.min(NO_INDEX_BOUND)
    }
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
    let indices_bytes = forbp_encode(indices, &row_lengths, index_dtype_u16)?;

    // values → reinterpret to u32, then Rice encode
    let values_u32 = raw_bytes_to_u32(values, value_encoding)?;
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

    // L1: reject headers whose byte-length computations overflow `usize`
    // (parity with every other codec path — F-e).
    indptr_byte_cap(n_rows)?;
    let idx_width = if index_dtype_u16 { 2 } else { 4 };
    checked_len(nnz, idx_width, "scx1 indices")?;
    checked_len(nnz, value_encoding.byte_width(), "scx1 values")?;

    // L2: reject headers declaring more elements than the compressed
    // sub-streams could physically produce (F-f), before any allocation.
    let n_rows_p1 = n_rows
        .checked_add(1)
        .ok_or_else(|| CodecError::MalformedInput(format!("scx1 n_rows+1 overflow: {n_rows}")))?;
    bound_capacity(n_rows_p1, encoded.indptr_bytes.len(), "scx1 indptr")?;
    bound_capacity(nnz, encoded.indices_bytes.len(), "scx1 indices")?;
    bound_capacity(nnz, encoded.values_bytes.len(), "scx1 values")?;

    // indptr ← Delta-Golomb
    let indptr = delta_golomb_decode(encoded.indptr_bytes, n_rows_p1)?;

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

/// Decompress an LZ4 frame, refusing to produce more than `max_bytes`.
///
/// `max_bytes` is the exact expected decompressed length, derived from the
/// shard's declared shape (`indptr_byte_cap` / `checked_len`) exactly as the
/// Zstd paths derive theirs. The cap is applied *during* decompression via
/// `Read::take`, not checked afterwards: an LZ4 frame compresses runs at a
/// ratio well past 100:1, so a small shard could otherwise force an
/// arbitrarily large allocation before anything noticed the length was wrong.
/// Mirrors [`zstd_decode_bounded`].
fn lz4_frame_decompress(data: &[u8], max_bytes: usize) -> Result<Vec<u8>, CodecError> {
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
    n_rows: usize,
    nnz: usize,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<DecodedShard, CodecError> {
    // Bound every sub-stream from the declared shape before decompressing, in
    // the same order and by the same helpers as `decode_zstd_ref`. Deriving the
    // caps up front is also what keeps the unchecked `count * width`
    // multiplications inside `le_bytes_to_u64` / `le_bytes_to_indices` out of
    // reach of a hostile `n_rows` / `nnz`: `indptr_byte_cap` and `checked_len`
    // do those multiplications in checked arithmetic and return an error, where
    // this codec previously reached them directly and panicked under the
    // crate's `overflow-checks = true`.
    let index_width = if index_dtype_u16 { 2 } else { 4 };
    let indptr_max = indptr_byte_cap(n_rows)?;
    let indices_max = checked_len(nnz, index_width, "indices")?;
    let values_max = checked_len(nnz, value_encoding.byte_width(), "values")?;

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

// ---------------------------------------------------------------------------
// CodecId::ShufDeltaZstd
//
// byte-filter codec: per sub-stream,
//   indices / indptr : byte-shuffle -> byte-delta -> zstd
//   integer values   : byte-shuffle -> zstd          (no delta)
//   float values     : zstd only    (no shuffle, no delta)
// Monolithic per shard. Random-row / grouped
// reads decode the whole shard; the row-group-framed form (F5-b) rides the
// BlockIndex substrate and is out of scope here.
// ---------------------------------------------------------------------------

/// zstd level for the ShufDeltaZstd codec. Matches the plain `Zstd` codec's
/// level so a shufdelta-vs-zstd size comparison isolates the shuffle+delta
/// pre-filter rather than the compression level.
const SHUFDELTA_ZSTD_LEVEL: i32 = 3;

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
    n_rows: usize,
    nnz: usize,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<DecodedShard, CodecError> {
    let index_width = if index_dtype_u16 { 2 } else { 4 };
    let indptr_max = indptr_byte_cap(n_rows)?;
    let indices_max = checked_len(nnz, index_width, "indices")?;
    let values_max = checked_len(nnz, value_encoding.byte_width(), "values")?;

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

    let expected_len = checked_len(nnz, value_encoding.byte_width(), "values")?;
    if values_raw.len() != expected_len {
        return Err(CodecError::MalformedInput(format!(
            "shufdelta values byte length {} != expected {}",
            values_raw.len(),
            expected_len
        )));
    }

    Ok((indptr, indices, values_raw))
}

/// Fail loud if a zstd-decompressed sub-stream is not exactly the expected
/// length before an in-place undelta indexes it (a truncated frame would
/// otherwise mis-align the per-plane cumulative sum).
fn expect_exact_len(got: usize, expected: usize, which: &str) -> Result<(), CodecError> {
    if got != expected {
        return Err(CodecError::MalformedInput(format!(
            "shufdelta {which} decompressed length {got} != expected {expected}"
        )));
    }
    Ok(())
}

/// Checked `count * width` for decode allocation caps / expected byte lengths.
/// A hostile shard header can carry an `nnz`/`n_rows` that overflows `usize`
/// when scaled by a byte width; return a `MalformedInput` error instead of
/// panicking (debug) or wrapping to a bogus cap (release) (F-e).
fn checked_len(count: usize, width: usize, what: &str) -> Result<usize, CodecError> {
    count.checked_mul(width).ok_or_else(|| {
        CodecError::MalformedInput(format!("{what} length {count} * {width} overflows usize"))
    })
}

/// Checked byte cap for an indptr sub-stream: `(n_rows + 1) * 8`. Guards the
/// `+ 1` as well so a `usize::MAX` `n_rows` can't wrap to 0 before the multiply
/// (`n_rows` comes from a u32 header field today, but keep it panic-free) (F-e).
fn indptr_byte_cap(n_rows: usize) -> Result<usize, CodecError> {
    n_rows
        .checked_add(1)
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| {
            CodecError::MalformedInput(format!(
                "indptr length (n_rows={n_rows} + 1) * 8 overflows usize"
            ))
        })
}

/// Plausibility bound for a decode allocation (F-f): the number of output
/// elements a Scx1 primitive can produce is physically bounded by the number
/// of input *bits*, since the theoretical minimum is 1 bit per element (a
/// Golomb/Rice code with `k ≥ ceil(log2(max))` encodes zero in 1 bit). Reject
/// a header that declares more elements than `input_len * 8`, so a hostile
/// `nnz`/`n_rows` can't drive a `Vec::with_capacity` into an eager multi-GiB
/// allocation (or a 32-bit `capacity overflow` panic) from a few bytes of
/// compressed input. This bound is deliberately loose — it can never reject
/// valid data — and complements the `checked_len`/`indptr_byte_cap` overflow
/// guards, which catch a different failure mode (`usize` overflow).
///
/// Uses `checked_mul` (not `saturating_mul`): on a 32-bit target an
/// `input_len ≥ 512 MiB` would saturate `input_len * 8` to `usize::MAX`, and a
/// `declared == usize::MAX` would then slip past a saturating comparison. When
/// `input_len * 8` overflows `usize` the true bound exceeds any representable
/// `declared`, so the input is trivially plausible and we accept it.
pub(crate) fn bound_capacity(
    declared: usize,
    input_len: usize,
    what: &str,
) -> Result<usize, CodecError> {
    if let Some(max_elements) = input_len.checked_mul(8) {
        if declared > max_elements {
            return Err(CodecError::MalformedInput(format!(
                "{what}: declared {declared} elements but input is only {input_len} bytes \
                 (max {max_elements} elements at 1 bit/element)"
            )));
        }
    }
    Ok(declared)
}

/// Post-decode structural check: assert the three decoded arrays actually are
/// the CSR the shard header declared.
///
/// Every other guard in this module is a *pre*-decode plausibility bound on
/// byte lengths. This one runs after, and it is needed because a shard's three
/// sub-streams are decoded independently: `delta_golomb_decode` and
/// `rice_decode` return exactly the count the caller asked for, while
/// `forbp_decode_with_hint` used to return whatever its own per-row nnz varints
/// said. So a corrupt Scx1 shard could decode to `indptr = [0, 6]` with three
/// indices and six values — a structurally invalid CSR, returned as `Ok`.
///
/// Downstream that is not benign. `ScxCsr::new_unchecked` restates these
/// invariants as a caller obligation and only `debug_assert`s them, so in a
/// release build `scx_sparse::transpose::csr_to_csc` walks
/// `indptr[row]..indptr[row + 1]` and indexes past the end of `indices` —
/// a panic on malformed input, which the reader convention forbids.
///
/// Mirrors invariants 1–4 and 6 of `ScxCsr::new_unchecked`. Invariant 5 (every
/// index below the minor-axis extent) is enforced separately, by
/// `u32_vec_to_i32_bounded` here and `check_minor_indices` in `scx-format-io`,
/// because it needs `n_minor`, which the codec layer is not given.
///
/// `values_len` is an element count, not a byte count.
fn check_decoded_shape(
    indptr: &[u64],
    indices_len: usize,
    values_len: usize,
    n_rows: usize,
    nnz: usize,
) -> Result<(), CodecError> {
    check_indptr_shape(indptr, n_rows, Some(nnz))?;
    if indices_len != nnz {
        return Err(CodecError::MalformedInput(format!(
            "decoded CSR has {indices_len} indices != declared nnz {nnz}"
        )));
    }
    if values_len != nnz {
        return Err(CodecError::MalformedInput(format!(
            "decoded CSR has {values_len} values != declared nnz {nnz}"
        )));
    }
    Ok(())
}

/// Validate the indptr half of the CSR shape invariant: `len == n_rows + 1`,
/// starts at 0, monotone non-decreasing, and (when the caller knows it) ends at
/// the declared `nnz`.
///
/// Generic over the integer width because the decode families hand it different
/// types — the whole-shard decoders carry `u64` straight off the sub-stream,
/// while the indptr-only paths have already widened to the `i64` scipy layout.
/// One implementation rather than two on purpose: the hand-rolled first/last
/// check that used to live in [`decode_row_group_indptr_only`] is exactly how
/// the *monotonicity* half went missing on every GPU path that builds its
/// `GpuCsr` indptr from it.
///
/// Monotonicity is not redundant with the endpoint checks. An interior entry can
/// exceed `nnz` while the last one is honest — `[0, 5, 2]` for `nnz = 2` — and a
/// consumer walking `indptr[row]..indptr[row + 1]` then reads past the end of
/// `indices` one row early. Nor is a zero start implied by the codec:
/// `delta_golomb_decode` reads its first value as a raw LE `u64`, so an Scx1
/// indptr may begin anywhere even though its deltas are non-negative.
///
/// `nnz` is `None` for callers that decode the indptr alone and have no declared
/// non-zero count to compare against.
pub fn check_indptr_shape<T: Copy + Into<i128>>(
    indptr: &[T],
    n_rows: usize,
    nnz: Option<usize>,
) -> Result<(), CodecError> {
    let expected_indptr_len = n_rows
        .checked_add(1)
        .ok_or_else(|| CodecError::MalformedInput(format!("CSR n_rows+1 overflow: {n_rows}")))?;
    if indptr.len() != expected_indptr_len {
        return Err(CodecError::MalformedInput(format!(
            "decoded CSR indptr length {} != n_rows + 1 ({expected_indptr_len})",
            indptr.len()
        )));
    }
    let first: i128 = match indptr.first() {
        Some(&v) => v.into(),
        None => {
            return Err(CodecError::MalformedInput(
                "decoded CSR indptr is empty".into(),
            ))
        }
    };
    if first != 0 {
        return Err(CodecError::MalformedInput(format!(
            "decoded CSR indptr must start at 0, got {first}"
        )));
    }
    let mut prev = first;
    for (row, &raw) in indptr.iter().enumerate().skip(1) {
        let cur: i128 = raw.into();
        if cur < prev {
            return Err(CodecError::MalformedInput(format!(
                "decoded CSR indptr not monotone at row {}: {prev} > {cur}",
                row - 1
            )));
        }
        prev = cur;
    }
    if let Some(nnz) = nnz {
        if prev != nnz as i128 {
            return Err(CodecError::MalformedInput(format!(
                "decoded CSR indptr ends at {prev} != declared nnz {nnz}"
            )));
        }
    }
    Ok(())
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

/// Decompress exactly `n_values` f32s from a pcodec stream.
///
/// Two opposite hostile shapes meet here, and a guard for one is not a guard
/// for the other:
///
/// - **Stream larger than declared.** `pco::standalone::simple_decompress`
///   sizes its output from the *stream*, so a shard declaring two values could
///   decompress a million and the length check downstream would fire one full
///   allocation too late. Bounded below by refusing to append past `n_values`.
/// - **Declared larger than stream.** Sizing the destination from `n_values`
///   instead inverts the problem — see the `n_values` note below. Bounded by
///   growing `out` only as pco actually produces numbers.
///
/// **Do not add a compression-ratio plausibility bound to this function.**
/// [`bound_capacity`] is sound only for the Scx1 primitives, whose Golomb/Rice
/// codes spend at least one input bit per output value. pcodec is an entropy
/// coder with no such floor: a constant run costs ~0 bits/value, and ordinary
/// single-cell payloads sit under it too — raw counts stored as `f32` (the
/// standard AnnData layout, ~80% ones) measure ≈0.96 bits/value. Applying the
/// 1-bit floor here rejected shards this crate's *own encoder* had just
/// produced, at every realistic shard size. See
/// `test_pcodec_low_entropy_float_roundtrip`.
///
/// **`n_values` is not trustworthy, and must not size the destination.** It is
/// a shard-header field. It is tempting to argue it has already been
/// corroborated, because `decode_pcodec_ref` decodes the indices sub-stream
/// first and `le_bytes_to_indices` demands an *exact* `n_values * index_width`
/// bytes — but those are *decompressed* bytes, and Zstd will expand a ~2 KiB
/// run of zero indices into exactly the `n_values * width` the check wants. A
/// `vec![0f32; n_values]` sized from the header therefore hands a few kilobytes
/// of input a second allocation of the same magnitude as the indices bomb that
/// let it through: measured at 165 MB of peak RSS from a 2.5 KiB shard before
/// this loop replaced it.
///
/// So the destination grows with numbers pco has *actually produced*, capped at
/// `n_values`, which is the same shape `zstd_decode_bounded` uses — eager
/// capacity clamped, `read_to_end` growing against a `take` limiter. `read`
/// wants a `dst` whose length is a multiple of 256 (or ≥ the chunk remainder),
/// hence the fixed slab.
fn pcodec_decompress_bounded(data: &[u8], n_values: usize) -> Result<Vec<f32>, CodecError> {
    use pco::standalone::{DecompressorItem, FileDecompressor};

    // 65536 = 256 × 256, satisfying `ChunkDecompressor::read`'s stride rule.
    const SLAB: usize = 1 << 16;

    let pco_err = |e: pco::errors::PcoError| CodecError::Io(std::io::Error::other(e.to_string()));
    let too_many = || {
        CodecError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("decompressed pcodec values exceeds limit {n_values}"),
        ))
    };

    // Overflow guard only — `n_values` still bounds the *total*, it just never
    // gets to be an up-front allocation.
    checked_len(n_values, std::mem::size_of::<f32>(), "pcodec values")?;

    let mut out: Vec<f32> = Vec::with_capacity(n_values.min(SLAB));
    // Round up to `read`'s 256 stride, but never overshoot a small shard: a
    // fixed 256 KiB slab made a 5k-value shard decode ~1.5× slower than the
    // eager path it replaces. `checked_len` above keeps `next_multiple_of`
    // clear of overflow. Both branches of `read`'s contract stay satisfied —
    // this is a multiple of 256, and for an honest stream it is also ≥ the
    // chunk remainder.
    let slab_len = SLAB.min(n_values.next_multiple_of(256).max(256));
    let mut slab = vec![0f32; slab_len];
    let (fd, mut src) = FileDecompressor::new(data).map_err(pco_err)?;

    loop {
        match fd.chunk_decompressor::<f32, _>(src).map_err(pco_err)? {
            DecompressorItem::EndOfData(_) => break,
            DecompressorItem::Chunk(mut cd) => {
                loop {
                    let progress = cd.read(&mut slab).map_err(pco_err)?;
                    // Refuse *before* appending, so an over-long stream can
                    // never grow `out` past the declared shape.
                    if out.len() + progress.n_processed > n_values {
                        return Err(too_many());
                    }
                    out.extend_from_slice(&slab[..progress.n_processed]);
                    if progress.finished {
                        break;
                    }
                    if progress.n_processed == 0 {
                        // Defensive: a chunk that reports neither progress nor
                        // completion would otherwise spin forever.
                        return Err(CodecError::MalformedInput(
                            "pcodec chunk made no progress".to_string(),
                        ));
                    }
                }
                src = cd.into_src();
            }
        }
    }

    if out.len() != n_values {
        return Err(CodecError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "decompressed pcodec value count {} != expected {n_values}",
                out.len()
            ),
        )));
    }
    Ok(out)
}

fn decode_pcodec_ref(
    encoded: &EncodedShardRef,
    n_rows: usize,
    nnz: usize,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<DecodedShard, CodecError> {
    // indptr and indices: Zstd decompress
    let indptr_max = indptr_byte_cap(n_rows)?;
    let indices_max = checked_len(nnz, if index_dtype_u16 { 2 } else { 4 }, "indices")?;

    let indptr_raw = zstd_decode_bounded(encoded.indptr_bytes, indptr_max)?;
    let indices_raw = zstd_decode_bounded(encoded.indices_bytes, indices_max)?;

    let indptr = le_bytes_to_u64(&indptr_raw, n_rows + 1)?;
    let indices = le_bytes_to_indices(&indices_raw, nnz, index_dtype_u16)?;

    // values: Pcodec for float encodings, Zstd for integer encodings
    let values_raw = match value_encoding {
        ValueEncoding::Float32 => {
            let floats = pcodec_decompress_bounded(encoded.values_bytes, nnz)?;
            let mut buf = Vec::with_capacity(floats.len() * 4);
            for &f in &floats {
                buf.extend_from_slice(&f.to_le_bytes());
            }
            buf
        }
        ValueEncoding::Float16 => {
            // Decompress as f32, narrow back to f16
            let floats = pcodec_decompress_bounded(encoded.values_bytes, nnz)?;
            let mut buf = Vec::with_capacity(floats.len() * 2);
            for &f in &floats {
                buf.extend_from_slice(&half::f16::from_f32(f).to_le_bytes());
            }
            buf
        }
        _ => {
            // Integer encodings: Zstd decompress
            let values_max = checked_len(nnz, value_encoding.byte_width(), "values")?;
            zstd_decode_bounded(encoded.values_bytes, values_max)?
        }
    };

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

    /// An on-disk `u32::MAX` decodes to f32 as exactly 2³² (the conversion
    /// rounds up), and the rewrite paths re-encode that decoded f32 under the
    /// input's own `Uint32` encoding. So this arm must accept 2³² and saturate
    /// it back to `u32::MAX` — a bound that rejects it aborts compact / merge /
    /// sort / build_csc on format-valid archives. On the detect path, fresh
    /// out-of-range values are kept away from this arm by
    /// `detect_value_encoding` rather than by this check — but callers that
    /// pass an explicit encoding bypass that, and still saturate; see the
    /// comment on the arm itself.
    #[test]
    fn encode_f32_uint32_preserves_decoded_u32_max() {
        let decoded_max = u32::MAX as f32;
        assert_eq!(decoded_max, (1u128 << 32) as f32, "u32::MAX as f32 IS 2^32");

        let mut buf = Vec::new();
        ValueEncoding::Uint32
            .encode_f32(&mut buf, decoded_max)
            .unwrap();
        assert_eq!(
            buf,
            u32::MAX.to_le_bytes(),
            "u32::MAX must survive re-encode"
        );

        // The largest f32 strictly below 2³² (2³² - 2⁸) is exact either way.
        buf.clear();
        ValueEncoding::Uint32
            .encode_f32(&mut buf, 4_294_967_040.0f32)
            .unwrap();
        assert_eq!(buf, 4_294_967_040u32.to_le_bytes());

        // Genuinely out of range, and NaN, are still refused.
        assert!(ValueEncoding::Uint32
            .encode_f32(&mut Vec::new(), 8_589_934_592.0f32)
            .is_err());
        assert!(ValueEncoding::Uint32.encode_f32_batch(&[f32::NAN]).is_err());
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
            CodecId::ShufDeltaZstd,
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

    /// ShufDeltaZstd round-trips float values (zstd-only value path, #142).
    #[test]
    fn test_shufdelta_float32_roundtrip() {
        for &u16_idx in &[true, false] {
            let (indptr, indices, values, n_rows, nnz) = make_test_csr(ValueEncoding::Float32);
            let encoded = encode_shard(
                &indptr,
                &indices,
                &values,
                CodecId::ShufDeltaZstd,
                ValueEncoding::Float32,
                u16_idx,
            )
            .unwrap();
            let (dec_ip, dec_ix, dec_v) = decode_shard(
                &encoded,
                CodecId::ShufDeltaZstd,
                ValueEncoding::Float32,
                n_rows,
                nnz,
                u16_idx,
            )
            .unwrap();
            assert_eq!(indptr, dec_ip);
            assert_eq!(indices, dec_ix);
            assert_eq!(values, dec_v);
        }
    }

    /// ShufDeltaZstd `decode_indptr_only` matches the full decode's indptr.
    #[test]
    fn test_shufdelta_decode_indptr_only() {
        let (indptr, indices, values, n_rows, _nnz) = make_test_csr(ValueEncoding::Uint32);
        let encoded = encode_shard(
            &indptr,
            &indices,
            &values,
            CodecId::ShufDeltaZstd,
            ValueEncoding::Uint32,
            false,
        )
        .unwrap();
        let ip = decode_indptr_only(&encoded.indptr_bytes, CodecId::ShufDeltaZstd, n_rows).unwrap();
        let expected: Vec<i64> = indptr.iter().map(|&v| v as i64).collect();
        assert_eq!(ip, expected);
    }

    /// A truncated ShufDeltaZstd sub-frame is rejected (bounded-alloc / exact-len
    /// guard) rather than panicking in the in-place undelta.
    #[test]
    fn test_shufdelta_rejects_truncated_frame() {
        let (indptr, indices, values, n_rows, nnz) = make_test_csr(ValueEncoding::Uint32);
        let encoded = encode_shard(
            &indptr,
            &indices,
            &values,
            CodecId::ShufDeltaZstd,
            ValueEncoding::Uint32,
            false,
        )
        .unwrap();
        // Corrupt the indices frame: a valid but too-short zstd frame (empty payload).
        let short_indices = zstd::encode_all(&b""[..], SHUFDELTA_ZSTD_LEVEL).unwrap();
        let bad = EncodedShardRef {
            indptr_bytes: &encoded.indptr_bytes,
            indices_bytes: &short_indices,
            values_bytes: &encoded.values_bytes,
        };
        let res = decode_shard_ref(
            &bad,
            CodecId::ShufDeltaZstd,
            ValueEncoding::Uint32,
            n_rows,
            nnz,
            false,
        );
        assert!(matches!(res, Err(CodecError::MalformedInput(_))));
    }

    /// Frame a small CSR into `None` row-groups (local-rebased indptr, contiguous
    /// indices/values) and return the three sub-streams + spans. Mirrors the
    /// writer-side `frame_none_shard` layout so `decode_row_group` can be tested
    /// in isolation.
    #[allow(clippy::type_complexity)]
    fn frame_none_for_test(
        indptr: &[u64],
        indices: &[u32],
        values_u32: &[u32],
        group_rows: usize,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<RowGroupSpan>) {
        let n_rows = indptr.len() - 1;
        let w_i = 2usize; // u16 indices in this fixture
        let w_v = 4usize; // u32 values
        let indices_bytes = indices_to_le_bytes(indices, true).unwrap();
        let mut values_bytes = Vec::new();
        for &v in values_u32 {
            values_bytes.write_u32::<LittleEndian>(v).unwrap();
        }
        let mut indptr_stream = Vec::new();
        let mut spans = Vec::new();
        let mut r0 = 0usize;
        while r0 < n_rows {
            let r1 = (r0 + group_rows).min(n_rows);
            let base = indptr[r0];
            let ip_off = indptr_stream.len();
            for &ip in &indptr[r0..=r1] {
                indptr_stream.write_u64::<LittleEndian>(ip - base).unwrap();
            }
            let nnz_in_block = (indptr[r1] - base) as u32;
            spans.push(RowGroupSpan {
                row_start: r0 as u32,
                n_rows: (r1 - r0) as u16,
                nnz: nnz_in_block,
                indptr: ip_off..indptr_stream.len(),
                indices: (indptr[r0] as usize * w_i)..(indptr[r1] as usize * w_i),
                values: (indptr[r0] as usize * w_v)..(indptr[r1] as usize * w_v),
            });
            r0 = r1;
        }
        (indptr_stream, indices_bytes, values_bytes, spans)
    }

    #[test]
    fn test_decode_row_group_none_parity_and_independence() {
        let indptr: Vec<u64> = vec![0, 2, 5, 6, 8];
        let indices: Vec<u32> = vec![1, 3, 0, 2, 4, 2, 0, 5];
        let values_u32: Vec<u32> = vec![5, 10, 1, 3, 7, 2, 9, 4];
        let (mut ip_stream, ix_stream, vv_stream, spans) =
            frame_none_for_test(&indptr, &indices, &values_u32, 2);
        assert_eq!(spans.len(), 2);

        // Group 0: rows 0..2 → local indptr [0,2,5], indices [1,3,0,2,4], vals [5,10,1,3,7]
        let (g0_ip, g0_ix, g0_v) = decode_row_group(
            CodecId::None,
            &spans[0],
            &ip_stream,
            &ix_stream,
            &vv_stream,
            ValueEncoding::Uint32,
            true,
        )
        .unwrap();
        assert_eq!(g0_ip, vec![0, 2, 5]);
        assert_eq!(g0_ix, vec![1, 3, 0, 2, 4]);
        assert_eq!(
            raw_bytes_to_u32(&g0_v, ValueEncoding::Uint32).unwrap(),
            vec![5, 10, 1, 3, 7]
        );

        // Group 1: rows 2..4 → local indptr [0,1,3], indices [2,0,5], vals [2,9,4]
        let (g1_ip, g1_ix, g1_v) = decode_row_group(
            CodecId::None,
            &spans[1],
            &ip_stream,
            &ix_stream,
            &vv_stream,
            ValueEncoding::Uint32,
            true,
        )
        .unwrap();
        assert_eq!(g1_ip, vec![0, 1, 3]);
        assert_eq!(g1_ix, vec![2, 0, 5]);
        assert_eq!(
            raw_bytes_to_u32(&g1_v, ValueEncoding::Uint32).unwrap(),
            vec![2, 9, 4]
        );

        // Cross-group independence: corrupt group 0's indptr bytes; group 1 still decodes.
        for b in ip_stream[spans[0].indptr.clone()].iter_mut() {
            *b = 0xFF;
        }
        let (g1_ip2, g1_ix2, _) = decode_row_group(
            CodecId::None,
            &spans[1],
            &ip_stream,
            &ix_stream,
            &vv_stream,
            ValueEncoding::Uint32,
            true,
        )
        .unwrap();
        assert_eq!(g1_ip2, vec![0, 1, 3]);
        assert_eq!(g1_ix2, vec![2, 0, 5]);
    }

    #[test]
    fn test_decode_row_group_rejects_oob_span() {
        let span = RowGroupSpan {
            row_start: 0,
            n_rows: 1,
            nnz: 1,
            indptr: 0..16,
            indices: 0..2,
            values: 0..4,
        };
        // Empty streams → range out of bounds → MalformedInput, not a panic.
        let res = decode_row_group(
            CodecId::None,
            &span,
            &[],
            &[],
            &[],
            ValueEncoding::Uint32,
            true,
        );
        assert!(matches!(res, Err(CodecError::MalformedInput(_))));
    }

    /// Frame a CSR into row-groups the way the writer's `encode_shard_framed`
    /// does, but at the codec level: each group is an independent
    /// `encode_shard(codec, ..)` over its **local-rebased** indptr, and the
    /// three sub-streams are concatenated. Returns (indptr, indices, values,
    /// spans) — exactly the layout `decode_row_group` consumes.
    #[allow(clippy::type_complexity)]
    fn frame_codec_for_test(
        codec: CodecId,
        indptr: &[u64],
        indices: &[u32],
        values_bytes: &[u8],
        venc: ValueEncoding,
        idx16: bool,
        group_rows: usize,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<RowGroupSpan>) {
        let n_rows = indptr.len() - 1;
        let w_v = venc.byte_width();
        let (mut ip_stream, mut ix_stream, mut vv_stream) = (Vec::new(), Vec::new(), Vec::new());
        let mut spans = Vec::new();
        let mut r0 = 0usize;
        while r0 < n_rows {
            let r1 = (r0 + group_rows).min(n_rows);
            let base = indptr[r0];
            let local_indptr: Vec<u64> = (r0..=r1).map(|r| indptr[r] - base).collect();
            let g_indices = &indices[indptr[r0] as usize..indptr[r1] as usize];
            let g_values = &values_bytes[indptr[r0] as usize * w_v..indptr[r1] as usize * w_v];
            let enc = encode_shard(&local_indptr, g_indices, g_values, codec, venc, idx16).unwrap();
            let (ip_off, ix_off, vv_off) = (ip_stream.len(), ix_stream.len(), vv_stream.len());
            ip_stream.extend_from_slice(&enc.indptr_bytes);
            ix_stream.extend_from_slice(&enc.indices_bytes);
            vv_stream.extend_from_slice(&enc.values_bytes);
            spans.push(RowGroupSpan {
                row_start: r0 as u32,
                n_rows: (r1 - r0) as u16,
                nnz: (indptr[r1] - base) as u32,
                indptr: ip_off..ip_stream.len(),
                indices: ix_off..ix_stream.len(),
                values: vv_off..vv_stream.len(),
            });
            r0 = r1;
        }
        (ip_stream, ix_stream, vv_stream, spans)
    }

    /// Per-group parity + cross-group independence for the **compressed** framed
    /// codecs (the `None` case is covered by
    /// `test_decode_row_group_none_parity_and_independence`). Each group decodes
    /// to its local CSR byte-identically to the source, and corrupting one
    /// group's compressed frame leaves the others decodable — the property that
    /// makes framed random access safe.
    #[test]
    fn test_decode_row_group_compressed_parity_and_independence() {
        let indptr: Vec<u64> = vec![0, 2, 5, 6, 8];
        let indices: Vec<u32> = vec![1, 3, 0, 2, 4, 2, 0, 5];
        let values_u32: Vec<u32> = vec![5, 10, 1, 3, 7, 2, 9, 4];
        let mut values_bytes = Vec::new();
        for &v in &values_u32 {
            values_bytes.write_u32::<LittleEndian>(v).unwrap();
        }
        for codec in [CodecId::ShufDeltaZstd, CodecId::Zstd, CodecId::Lz4Shuffle] {
            let (ip, ix, mut vv, spans) = frame_codec_for_test(
                codec,
                &indptr,
                &indices,
                &values_bytes,
                ValueEncoding::Uint32,
                true,
                2,
            );
            assert_eq!(spans.len(), 2, "codec {codec:?}");

            // Group 1 decodes to its local CSR: rows 2..4 → indptr [0,1,3].
            let decode_g1 = |vv: &[u8]| {
                decode_row_group(codec, &spans[1], &ip, &ix, vv, ValueEncoding::Uint32, true)
                    .unwrap()
            };
            let (g1_ip, g1_ix, g1_v) = decode_g1(&vv);
            assert_eq!(g1_ip, vec![0, 1, 3], "codec {codec:?}");
            assert_eq!(g1_ix, vec![2, 0, 5], "codec {codec:?}");
            assert_eq!(
                raw_bytes_to_u32(&g1_v, ValueEncoding::Uint32).unwrap(),
                vec![2, 9, 4],
                "codec {codec:?}"
            );

            // Cross-group independence: shred group 0's values frame; group 1
            // still decodes (it only reads its own byte ranges).
            for b in vv[spans[0].values.clone()].iter_mut() {
                *b = 0xFF;
            }
            let (g1_ip2, g1_ix2, _) = decode_g1(&vv);
            assert_eq!(g1_ip2, vec![0, 1, 3], "codec {codec:?} after corruption");
            assert_eq!(g1_ix2, vec![2, 0, 5], "codec {codec:?} after corruption");
        }
    }

    /// The truncated-frame guard also covers the `indptr` and `values`
    /// sub-frames (the existing `test_shufdelta_rejects_truncated_frame` only
    /// exercises `indices`).
    #[test]
    fn test_shufdelta_rejects_truncated_indptr_or_values_frame() {
        let (indptr, indices, values, n_rows, nnz) = make_test_csr(ValueEncoding::Uint32);
        let encoded = encode_shard(
            &indptr,
            &indices,
            &values,
            CodecId::ShufDeltaZstd,
            ValueEncoding::Uint32,
            false,
        )
        .unwrap();
        let short = zstd::encode_all(&b""[..], SHUFDELTA_ZSTD_LEVEL).unwrap();

        // Truncated indptr frame.
        let bad_indptr = EncodedShardRef {
            indptr_bytes: &short,
            indices_bytes: &encoded.indices_bytes,
            values_bytes: &encoded.values_bytes,
        };
        assert!(matches!(
            decode_shard_ref(
                &bad_indptr,
                CodecId::ShufDeltaZstd,
                ValueEncoding::Uint32,
                n_rows,
                nnz,
                false,
            ),
            Err(CodecError::MalformedInput(_))
        ));

        // Truncated values frame.
        let bad_values = EncodedShardRef {
            indptr_bytes: &encoded.indptr_bytes,
            indices_bytes: &encoded.indices_bytes,
            values_bytes: &short,
        };
        assert!(matches!(
            decode_shard_ref(
                &bad_values,
                CodecId::ShufDeltaZstd,
                ValueEncoding::Uint32,
                n_rows,
                nnz,
                false,
            ),
            Err(CodecError::MalformedInput(_))
        ));
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
        assert_eq!(CodecId::from_u8(5), Some(CodecId::ShufDeltaZstd));
        assert_eq!(CodecId::from_u8(6), None);

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
        let result = u32_vec_to_i32_bounded(data, NO_INDEX_BOUND).unwrap();
        assert_eq!(result, vec![0i32, 100, i32::MAX]);
    }

    #[test]
    fn test_u32_to_i32_rejects_overflow() {
        let data = vec![0u32, 100, u32::MAX];
        match u32_vec_to_i32_bounded(data, NO_INDEX_BOUND) {
            Err(CodecError::MalformedInput(msg)) => {
                assert!(msg.contains("exceeds i32::MAX"), "got: {msg}");
            }
            other => panic!("expected MalformedInput, got {other:?}"),
        }
    }

    /// The bound rides on the same scan as the sign guard, so it must produce a
    /// *different* message: "exceeds i32::MAX" would be a lie about an index of
    /// 9 in an 8-column shard, and would send whoever reads it looking for an
    /// overflow that is not there.
    #[test]
    fn test_u32_to_i32_rejects_out_of_range_index_with_its_own_message() {
        let data = vec![0u32, 3, 9];
        match u32_vec_to_i32_bounded(data, 8) {
            Err(e @ CodecError::IndexOutOfRange { .. }) => {
                let CodecError::IndexOutOfRange {
                    index,
                    position,
                    bound,
                } = e
                else {
                    unreachable!()
                };
                assert_eq!((index, position, bound), (9, 2, 8));
                assert!(
                    !e.to_string().contains("i32::MAX"),
                    "an in-i32-range index is not an overflow: {e}"
                );
            }
            other => panic!("expected IndexOutOfRange, got {other:?}"),
        }
    }

    /// `clamp_index_bound` maps the "undeclared" sentinel to the sign-only
    /// bound. Old writers stamped the file-level `n_vars` into every shard
    /// header, which is 0 on a multimodal file, so treating 0 as a real bound
    /// rejects valid files (two multimodal conformance fixtures, specifically).
    #[test]
    fn test_clamp_index_bound() {
        assert_eq!(clamp_index_bound(0), NO_INDEX_BOUND, "0 means undeclared");
        assert_eq!(clamp_index_bound(8), 8);
        assert_eq!(
            clamp_index_bound(u32::MAX),
            NO_INDEX_BOUND,
            "must not widen past the sign guarantee"
        );
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

    /// F-e: a hostile shard header carrying an `nnz` that overflows `usize`
    /// when scaled by the value/index byte width must return a `MalformedInput`
    /// error, not panic (debug) or wrap to a bogus allocation cap (release).
    /// Covers `None` (whose `le_bytes_to_indices` computes `nnz * elem`
    /// internally), `Zstd` (whose caps are computed up front), and `Scx1`
    /// (whose `checked_len` guards were added in F-f).
    #[test]
    fn decode_rejects_nnz_length_overflow() {
        let encoded = EncodedShardRef {
            indptr_bytes: &[0u8; 16],
            indices_bytes: &[0u8; 4],
            values_bytes: &[0u8; 4],
        };
        for codec in [CodecId::None, CodecId::Zstd, CodecId::Scx1] {
            let err = decode_shard_ref(
                &encoded,
                codec,
                ValueEncoding::Uint32,
                1,          // n_rows
                usize::MAX, // nnz — nnz * width overflows usize
                false,
            )
            .unwrap_err();
            assert!(
                matches!(err, CodecError::MalformedInput(ref m) if m.contains("overflows usize")),
                "codec {codec:?}: expected MalformedInput overflow error, got {err:?}"
            );
        }
    }
}
