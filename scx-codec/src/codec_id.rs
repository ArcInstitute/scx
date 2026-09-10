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

/// The upper bound `Uint32` accepts, **inclusive at 2³²**.
///
/// Declared once because two things must agree on it: the per-value range check
/// in [`ValueEncoding::encode_f32`] and the slice-wide one in
/// [`ValueEncoding::encode_f32_into`]. The reason it is 2³² rather than
/// `u32::MAX` is on `encode_f32`'s `Uint32` arm and is load-bearing — do not
/// tighten it here without reading that comment.
const UINT32_BOUND_F32: f32 = (1u128 << 32) as f32;

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

    /// The numpy dtype name of this encoding — `"uint8"`, `"uint16"`,
    /// `"uint32"`, `"float32"` or `"float16"`. Every variant is a valid
    /// `numpy.dtype(...)` argument, so this is the one rendering shared by
    /// `scx info`, pyscx's `stored_dtype` / `Experiment.value_encoding` and the
    /// ops summaries (`value_encoding_name` in `scx-cli` and a
    /// `format!("{:?}").to_lowercase()` in pyscx used to carry their own copies).
    pub fn numpy_name(&self) -> &'static str {
        match self {
            Self::Uint8 => "uint8",
            Self::Uint16 => "uint16",
            Self::Uint32 => "uint32",
            Self::Float32 => "float32",
            Self::Float16 => "float16",
        }
    }

    /// The one encoding that describes a whole shard family, for **reporting**.
    ///
    /// `None` when there are no shards. A uniform family reports its own
    /// encoding (`Float16` stays `Float16`); a mixed one reports the widest:
    /// any float present ⇒ `Float32`, otherwise the widest integer.
    ///
    /// This is deliberately not `scx_ops`' `widest_value_encoding`, which picks
    /// an encoding to *write* into and therefore widens every float source to
    /// `Float32` even when all inputs are `Float16` — the conservative answer
    /// for a writer, and the wrong one for a reader describing what is on disk.
    pub fn widest(encs: &[ValueEncoding]) -> Option<ValueEncoding> {
        let first = *encs.first()?;
        if encs.iter().all(|&e| e == first) {
            return Some(first);
        }
        if encs.iter().any(|e| !e.is_integer()) {
            return Some(Self::Float32);
        }
        encs.iter().copied().max_by_key(|e| e.byte_width())
    }

    /// Encode a single f32 value to raw LE bytes, with range checking.
    ///
    /// This is the inverse of `values_raw_to_f32` for one element.
    pub fn encode_f32(&self, buf: &mut Vec<u8>, value: f32) -> Result<(), CodecError> {
        match self {
            Self::Uint8 => {
                if !(0.0..=255.0).contains(&value) {
                    return Err(CodecError::ValueOutOfRange {
                        value,
                        encoding: "Uint8",
                        max: u8::MAX as f64,
                    });
                }
                buf.push(value as u8);
            }
            Self::Uint16 => {
                if !(0.0..=65535.0).contains(&value) {
                    return Err(CodecError::ValueOutOfRange {
                        value,
                        encoding: "Uint16",
                        max: u16::MAX as f64,
                    });
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
                if !(0.0..=UINT32_BOUND_F32).contains(&value) {
                    return Err(CodecError::ValueOutOfRange {
                        value,
                        encoding: "Uint32",
                        max: u32::MAX as f64,
                    });
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
        self.encode_f32_into(&mut bytes, data)?;
        Ok(bytes)
    }

    /// [`Self::encode_f32_batch`] appending into a caller-owned buffer.
    ///
    /// Exists for the rewrite ops' per-shard accumulators, which append row by
    /// row into one buffer and must not allocate per row, and it is where the
    /// work actually happens — `encode_f32_batch` is a thin wrapper.
    ///
    /// **Why this is not a loop over [`Self::encode_f32`].** It was, and that
    /// showed up: measured on `scx optimize` at census_1m,
    /// `encode_f32`/`encode_f32_batch` accounted for **3.96 % of all cycles** —
    /// and because they run on the serial critical path (every writer converts
    /// values on the calling thread), that is roughly **10 % of the op's wall**,
    /// against a parallel encode that PR-42 already spread across the pool. The
    /// per-value shape costs a `match` on the encoding, a scalar range test and
    /// a capacity check for every nonzero. Here the `match` is hoisted, the
    /// range test is one pass a compiler can vectorise, and the capacity is
    /// reserved once.
    ///
    /// Measured (`cargo bench -p scx-codec -- value_encode`, 100K values,
    /// per-value -> batch): `Uint8` 314 -> 152 us (**2.07x**), `Uint16` 285 ->
    /// 159 us (**1.79x**), `Uint32` 336 -> 197 us (**1.71x**), `Float32` 257 ->
    /// 9.2 us (**27.8x** — little-endian f32 is an identity transform and
    /// `extend` turns it into a bulk copy). The integer widths land well short
    /// of the 5-10x the OPT plan projected; the residual is the saturating
    /// float-to-int cast, costed in the comment on the write below.
    ///
    /// **A rejected slice leaves `buf` untouched.** The range pass runs before
    /// anything is written, and reports the *first* offending value in
    /// [`CodecError::ValueOutOfRange`], so the caller's accumulator cannot gain
    /// half a row behind an `Err` and no rollback is needed. NaN counts as
    /// out of range, which is what stops `as uN` writing it as a silent `0`.
    pub fn encode_f32_into(&self, buf: &mut Vec<u8>, data: &[f32]) -> Result<(), CodecError> {
        // One pass to find the first unrepresentable value, and if there is one
        // the error is built from it directly -- `buf` is never touched, so
        // there is nothing to roll back. An earlier revision replayed the whole
        // slice through `encode_f32` to rediscover the offender and truncated
        // the partial writes; once `CodecError::ValueOutOfRange` began carrying
        // the value that replay had nothing left to discover.
        //
        // The bounds MUST match `encode_f32`'s arm for arm -- this is the same
        // range rule, applied to a slice -- which is why `Uint32`'s lives in a
        // shared constant. `find`, not `all`, so the value survives; `contains`,
        // not a comparison pair, so NaN is rejected rather than written as a
        // silent `0` by `value as uN`.
        let offender = match self {
            Self::Uint8 => data
                .iter()
                .find(|v| !(0.0..=255.0).contains(*v))
                .map(|&v| (v, "Uint8", u8::MAX as f64)),
            Self::Uint16 => data
                .iter()
                .find(|v| !(0.0..=65535.0).contains(*v))
                .map(|&v| (v, "Uint16", u16::MAX as f64)),
            Self::Uint32 => data
                .iter()
                .find(|v| !(0.0..=UINT32_BOUND_F32).contains(*v))
                .map(|&v| (v, "Uint32", u32::MAX as f64)),
            // Every f32 is representable as itself, and `f16::from_f32`
            // saturates to +/-inf by design rather than failing.
            Self::Float32 | Self::Float16 => None,
        };
        if let Some((value, encoding, max)) = offender {
            return Err(CodecError::ValueOutOfRange {
                value,
                encoding,
                max,
            });
        }
        buf.reserve(data.len() * self.byte_width());
        // The remaining cost is the saturating `as` cast, not the write. Two
        // alternatives were measured and rejected, at 100K values:
        //   * writing through a `resize`d slice instead of `extend`: noise on
        //     the integer widths (u8 152.8 -> 151.5 us) and **89 % worse** on
        //     `Float32` (9.3 -> 17.7 us), because the `resize` memset costs
        //     more than the bulk copy `extend` gets for an identity transform.
        //   * `f32::to_int_unchecked` after the range pass: u8 152.8 -> 98.8 us
        //     (a further 1.55x, 4.05x against the per-value path). Declined.
        //     It buys ~0.8 s of a 33 s census_1m compact — ~2.4 % — in exchange
        //     for an `unsafe` whose precondition lives twenty lines up in
        //     another statement: any later edit that reorders, short-circuits or
        //     loosens `in_range` turns a rejected value into UB, and the value
        //     it turns into garbage is NaN, which is precisely what that pass
        //     exists to catch. Revisit only with the check fused into the loop.
        match self {
            Self::Uint8 => buf.extend(data.iter().map(|&v| v as u8)),
            Self::Uint16 => buf.extend(data.iter().flat_map(|&v| (v as u16).to_le_bytes())),
            // Saturating `as` is the documented behaviour for a decoded
            // `u32::MAX` (an f32 of exactly 2³²) and is why the bound above is
            // inclusive. See `encode_f32`'s `Uint32` arm.
            Self::Uint32 => buf.extend(data.iter().flat_map(|&v| (v as u32).to_le_bytes())),
            Self::Float32 => buf.extend(data.iter().flat_map(|&v| v.to_le_bytes())),
            Self::Float16 => buf.extend(
                data.iter()
                    .flat_map(|&v| half::f16::from_f32(v).to_le_bytes()),
            ),
        }
        Ok(())
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

    /// An `f32` value cannot be represented by the target integer encoding —
    /// out of range, or NaN.
    ///
    /// Typed rather than an `Io(InvalidData)` string so a caller can lift it
    /// into its own error without parsing a message. `scx-ops` does exactly
    /// that (`OpsError::ValueOutOfRange`, which pyscx maps and which names the
    /// offending value), and before this variant existed it had to re-run the
    /// whole per-value loop just to rediscover which value was bad.
    /// Classification is unchanged: like the `Io` form it had before, this
    /// falls through `ScxError`'s catch-all to `ScxError::Codec`.
    #[error("f32 value {value} out of range for {encoding} encoding (max {max})")]
    ValueOutOfRange {
        value: f32,
        encoding: &'static str,
        max: f64,
    },

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

#[cfg(test)]
mod value_encoding_tests {
    use super::ValueEncoding;

    #[test]
    fn numpy_name_is_a_valid_numpy_dtype_for_every_variant() {
        let all = [
            ValueEncoding::Uint8,
            ValueEncoding::Uint16,
            ValueEncoding::Uint32,
            ValueEncoding::Float32,
            ValueEncoding::Float16,
        ];
        let names: Vec<&str> = all.iter().map(|e| e.numpy_name()).collect();
        assert_eq!(names, ["uint8", "uint16", "uint32", "float32", "float16"]);
        // Round-trips through the byte code, so the two tables cannot drift.
        for e in all {
            assert_eq!(ValueEncoding::from_u8(e as u8), Some(e));
        }
    }

    #[test]
    fn widest_reports_uniform_families_as_themselves() {
        assert_eq!(ValueEncoding::widest(&[]), None);
        assert_eq!(
            ValueEncoding::widest(&[ValueEncoding::Uint16, ValueEncoding::Uint16]),
            Some(ValueEncoding::Uint16)
        );
        // A uniform half-precision family is reported as such — the write-side
        // fold would say Float32 here, which is the distinction this test pins.
        assert_eq!(
            ValueEncoding::widest(&[ValueEncoding::Float16, ValueEncoding::Float16]),
            Some(ValueEncoding::Float16)
        );
    }

    #[test]
    fn widest_mixed_family_is_the_widest_integer_or_float32() {
        assert_eq!(
            ValueEncoding::widest(&[
                ValueEncoding::Uint8,
                ValueEncoding::Uint32,
                ValueEncoding::Uint16
            ]),
            Some(ValueEncoding::Uint32)
        );
        assert_eq!(
            ValueEncoding::widest(&[ValueEncoding::Uint8, ValueEncoding::Float16]),
            Some(ValueEncoding::Float32)
        );
        assert_eq!(
            ValueEncoding::widest(&[ValueEncoding::Float32, ValueEncoding::Float16]),
            Some(ValueEncoding::Float32)
        );
    }
}
