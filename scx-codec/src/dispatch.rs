// Codec ID dispatch: selects a codec implementation and applies the guards
// that every decode path shares (docs/codec.md (Codec IDs)).

use crate::codecs::lz4_shuffle::Lz4ShuffleCodec;
use crate::codecs::none::NoneCodec;
use crate::codecs::pcodec::PcodecCodec;
use crate::codecs::scx1::Scx1Codec;
use crate::codecs::shufdelta::ShufDeltaZstdCodec;
use crate::codecs::zstd_codec::ZstdCodec;
use crate::delta_golomb::delta_golomb_decode;
use crate::forbp::forbp_decode_with_hint;
use crate::guards::*;
use crate::raw::*;
use crate::rice::{rice_decode, B_VAL};
use crate::shard_codec::{DecodeBounds, ShardCodec, ShardShape};

// Re-exported so `scx_codec::dispatch::{CodecId, ValueEncoding, ...}` keeps
// resolving after the split: several in-crate modules and three `scx-format-io`
// integration tests import through that path, and `dispatch_tests.rs` reaches
// all of it through `use super::*`.
pub use crate::codec_id::*;
// These four were `pub` items of the public `dispatch` module before the split,
// so `scx_codec::dispatch::{..}` was a supported path for each. The glob `use`
// above restores the types; these restore the functions and the constant, which
// an out-of-tree caller would otherwise hit E0603 on.
pub use crate::codecs::zstd_codec::zstd_decode_bounded;
pub use crate::guards::{check_indptr_shape, clamp_index_bound, NO_INDEX_BOUND};

/// Encode a CSR shard's three arrays using the specified codec.
///
/// - `indptr`: the indptr array (length = n_rows + 1).
/// - `indices`: the column indices (length = nnz), stored as u32.
/// - `values`: raw little-endian bytes of the value array (length = nnz × value_encoding.byte_width()).
/// - `index_dtype_u16`: if true, every index fits in u16 — i.e. the largest
///   index `n_vars - 1` is `<= u16::MAX`, so up to 65_536 columns.
pub fn encode_shard(
    indptr: &[u64],
    indices: &[u32],
    values: &[u8],
    codec_id: CodecId,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
) -> Result<EncodedShard, CodecError> {
    match codec_id {
        CodecId::None => {
            NoneCodec::encode(indptr, indices, values, value_encoding, index_dtype_u16)
        }
        CodecId::Scx1 => {
            Scx1Codec::encode(indptr, indices, values, value_encoding, index_dtype_u16)
        }
        CodecId::Zstd => {
            ZstdCodec::encode(indptr, indices, values, value_encoding, index_dtype_u16)
        }
        CodecId::Lz4Shuffle => {
            Lz4ShuffleCodec::encode(indptr, indices, values, value_encoding, index_dtype_u16)
        }
        CodecId::Pcodec => {
            PcodecCodec::encode(indptr, indices, values, value_encoding, index_dtype_u16)
        }
        CodecId::ShufDeltaZstd => {
            ShufDeltaZstdCodec::encode(indptr, indices, values, value_encoding, index_dtype_u16)
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

/// Dispatch one codec, rejecting an encoding it cannot represent first.
///
/// Centralising `supports` is why `CodecError::FloatWithScx1` is now raised in
/// one place rather than being each decoder's first statement.
fn decode_via<C: ShardCodec>(
    encoded: &EncodedShardRef,
    shape: ShardShape,
    value_encoding: ValueEncoding,
    index_dtype_u16: bool,
    bounds: &DecodeBounds,
) -> Result<DecodedShard, CodecError> {
    if !C::supports(value_encoding) {
        return Err(CodecError::FloatWithScx1);
    }
    C::decode(encoded, shape, value_encoding, index_dtype_u16, bounds)
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
    // One shape, one derivation of the caps, then dispatch. A codec receives
    // `bounds`; it has no way to derive its own, which is what made "this
    // decoder forgot a guard" representable before (§3.5, and #436's LZ4 hole).
    let shape = ShardShape { n_rows, nnz };
    let bounds = DecodeBounds::derive(shape, value_encoding, index_dtype_u16)?;
    let decoded = match codec_id {
        CodecId::None => {
            decode_via::<NoneCodec>(encoded, shape, value_encoding, index_dtype_u16, &bounds)
        }
        CodecId::Scx1 => {
            decode_via::<Scx1Codec>(encoded, shape, value_encoding, index_dtype_u16, &bounds)
        }
        CodecId::Zstd => {
            decode_via::<ZstdCodec>(encoded, shape, value_encoding, index_dtype_u16, &bounds)
        }
        CodecId::Lz4Shuffle => {
            decode_via::<Lz4ShuffleCodec>(encoded, shape, value_encoding, index_dtype_u16, &bounds)
        }
        CodecId::Pcodec => {
            decode_via::<PcodecCodec>(encoded, shape, value_encoding, index_dtype_u16, &bounds)
        }
        CodecId::ShufDeltaZstd => decode_via::<ShufDeltaZstdCodec>(
            encoded,
            shape,
            value_encoding,
            index_dtype_u16,
            &bounds,
        ),
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
/// [`decode_shard_scipy`]'s conversions. Lets callers that decode a shard
/// themselves (e.g. the parallel decode path) produce the same scipy types the
/// sequential reader path returns.
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
    // `indptr_byte_cap` performs the `(n_rows + 1) * 8` in checked arithmetic,
    // so computing it first is also what stops a `usize::MAX` `n_rows` from
    // wrapping inside a codec's own `n_rows + 1` (F-f).
    let indptr_max = indptr_byte_cap(n_rows)?;
    let indptr_u64: Vec<u64> = match codec_id {
        CodecId::None => NoneCodec::decode_indptr_only(indptr_bytes, n_rows, indptr_max)?,
        CodecId::Scx1 => Scx1Codec::decode_indptr_only(indptr_bytes, n_rows, indptr_max)?,
        CodecId::Zstd => ZstdCodec::decode_indptr_only(indptr_bytes, n_rows, indptr_max)?,
        CodecId::Pcodec => PcodecCodec::decode_indptr_only(indptr_bytes, n_rows, indptr_max)?,
        CodecId::Lz4Shuffle => {
            Lz4ShuffleCodec::decode_indptr_only(indptr_bytes, n_rows, indptr_max)?
        }
        CodecId::ShufDeltaZstd => {
            ShufDeltaZstdCodec::decode_indptr_only(indptr_bytes, n_rows, indptr_max)?
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

#[cfg(test)]
#[path = "dispatch_tests.rs"]
mod tests;
