//! Split out of `dispatch.rs` (ORG-3.7-2). Pure move.

use crate::codec_id::{CodecError, DecodedShard, EncodedShard, EncodedShardRef, ValueEncoding};
use crate::delta_golomb::{delta_golomb_decode, delta_golomb_encode};
use crate::forbp::{forbp_decode_with_hint, forbp_encode};
use crate::guards::{bound_capacity, checked_len, indptr_byte_cap};
use crate::raw::{raw_bytes_to_u32, u32_to_raw_bytes};
use crate::rice::{rice_decode, rice_encode, B_VAL};

pub(crate) fn encode_scx1(
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

pub(crate) fn decode_scx1_ref(
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
