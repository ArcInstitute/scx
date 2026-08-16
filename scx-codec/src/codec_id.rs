//! Wire-format type system: codec ids, value encodings, the shard
//! containers and the crate error type.
//!
//! Split out of `dispatch.rs` (ORG-3.7-2). Pure move.

use std::ops::Range;

use crate::bitstream::BitStreamError;

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
/// consumed by [`crate::decode_row_group`]. The three `Range<usize>` are byte ranges
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

/// Scipy-compatible decoded shard: `(indptr_i64, indices_i32, data_f32)`.
///
/// Eliminates intermediate type conversions by producing the final scipy
/// types directly from the codec decoders.
pub type ScipyShard = (Vec<i64>, Vec<i32>, Vec<f32>);

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
