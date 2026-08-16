//! The `ShardCodec` seam: one shape for all six codecs, and one place where the
//! shape-derived allocation caps are computed.
//!
//! ORG-3.7-3. Review §3.5's finding is that the bounds-guard family drifted —
//! `decode_none_ref` was missing a cap its siblings applied, and (found while
//! implementing this series, #436) `decode_lz4_shuffle_ref` was missing the
//! whole family on a codec the writer auto-selects for ATAC counts. The cause
//! is structural: every decoder re-derived its own caps from `n_rows` / `nnz`,
//! so "forgot one" was a thing a decoder could silently be.
//!
//! [`DecodeBounds::derive`] is now the only place those three byte caps are
//! computed. A codec receives them; it cannot invent its own.
//!
//! # What this deliberately does *not* centralise
//!
//! `guards::bound_capacity` (crate-private) stays inside `scx1`. It rejects a header
//! declaring more elements than the input has *bits*, which is sound for
//! Golomb/Rice — the unary quotient spends at least one bit per value — and
//! **false for every entropy coder**: a constant `f32` layer reaches 0.0006
//! bits/value through pcodec. PR #436 applied it to pcodec and the crate's own
//! encoder began producing shards its own decoder refused, for the most common
//! real payload there is (raw counts stored as `f32`). Hoisting it into
//! `DecodeBounds` would have made that one bug into six.
//!
//! The rule the split encodes: **shape-derived byte caps are uniform and belong
//! to the driver; a bound that asserts something about a codec's compression
//! belongs to that codec.**

use crate::codec_id::{CodecError, DecodedShard, EncodedShard, EncodedShardRef, ValueEncoding};
use crate::guards::{checked_len, indptr_byte_cap};

/// The declared shape of a shard, as the caller read it from the header.
///
/// Bundled because `n_rows` and `nnz` are both `usize` and were passed
/// positionally to six decoders, one of which (`codecs::scx1`) took
/// them in a different order than the other five — a swap no type could catch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardShape {
    /// Rows in this shard; the indptr decodes to `n_rows + 1` entries.
    pub n_rows: usize,
    /// Declared non-zeros; the length of both `indices` and the value array.
    pub nnz: usize,
}

/// Maximum decoded byte length of each sub-stream, from the declared shape.
///
/// Every value here is an *exact* expected length, not a slack bound, so a
/// decoder may both cap decompression at it and compare against it afterwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeBounds {
    /// `(n_rows + 1) * 8`.
    pub indptr_max: usize,
    /// `nnz * (2 or 4)`, per `index_dtype_u16`.
    pub indices_max: usize,
    /// `nnz * value_encoding.byte_width()`.
    pub values_max: usize,
}

impl DecodeBounds {
    /// The single derivation site. Each product is checked, so a hostile
    /// `n_rows` / `nnz` is rejected here rather than overflowing inside a
    /// codec — which is what made the LZ4 path panic in release before #436.
    pub fn derive(
        shape: ShardShape,
        value_encoding: ValueEncoding,
        index_dtype_u16: bool,
    ) -> Result<Self, CodecError> {
        Ok(Self {
            indptr_max: indptr_byte_cap(shape.n_rows)?,
            indices_max: checked_len(shape.nnz, if index_dtype_u16 { 2 } else { 4 }, "indices")?,
            values_max: checked_len(shape.nnz, value_encoding.byte_width(), "values")?,
        })
    }
}

/// One shard codec.
///
/// Implementors are unit types in [`crate::codecs`]; the driver in
/// [`crate::dispatch`] selects between them. Associated functions rather than
/// methods because a codec is stateless — there is nothing to construct.
pub trait ShardCodec {
    /// Whether this codec can represent `value_encoding` at all.
    ///
    /// Only `Scx1` overrides: its value stream is Rice-coded, which is defined
    /// over integers. The driver checks this before dispatching, so the
    /// rejection is uniform rather than each decoder's first statement.
    fn supports(_value_encoding: ValueEncoding) -> bool {
        true
    }

    fn encode(
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
        value_encoding: ValueEncoding,
        index_dtype_u16: bool,
    ) -> Result<EncodedShard, CodecError>;

    /// Decode a whole shard. `bounds` is derived once by the driver from
    /// `shape`; implementors must not re-derive it.
    fn decode(
        encoded: &EncodedShardRef,
        shape: ShardShape,
        value_encoding: ValueEncoding,
        index_dtype_u16: bool,
        bounds: &DecodeBounds,
    ) -> Result<DecodedShard, CodecError>;

    /// Decode only the indptr sub-stream, for callers that need row offsets
    /// without materialising indices or values.
    fn decode_indptr_only(
        indptr_bytes: &[u8],
        n_rows: usize,
        indptr_max: usize,
    ) -> Result<Vec<u64>, CodecError>;
}
