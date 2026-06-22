// Integer detection, value encoding selection, and raw byte conversion.
//
// The detect/encode primitives live in `scx_codec::value_encoding`. This
// module adds the codec auto-selection policy on top.

use scx_codec::{CodecError, CodecId, ValueEncoding};
use scx_format_io::{select_codec, select_codec_for_modality, ModalityType};

pub use scx_codec::value_encoding::{
    detect_value_encoding as detect_value_encoding_only, values_to_raw_bytes,
};

/// CSR `index_dtype` code for a matrix with `n_vars` columns:
/// `0` = u16 indices (`n_vars ≤ u16::MAX`), `1` = u32 indices.
///
/// Single source of truth for the `if n_vars <= 65535 { 0 } else { 1 }`
/// boundary that every CSR / layer writer in the convert pipeline needs.
pub(crate) fn index_dtype_for(n_vars: u64) -> u8 {
    if n_vars <= u16::MAX as u64 {
        0
    } else {
        1
    }
}

/// Detect the best value encoding and auto-select codec for the data.
/// When `explicit_codec` is Some, uses that codec (with Scx1→Zstd fallback for floats).
/// When None, auto-selects based on value distribution. Single-modality
/// callers behave as if every shard is RNA (matches v1 / pre-Phase-E
/// output bit-for-bit).
pub fn detect_value_encoding(
    data: &[f32],
    explicit_codec: Option<CodecId>,
) -> Result<(ValueEncoding, CodecId), CodecError> {
    let encoding = detect_value_encoding_only(data);
    let raw_bytes = values_to_raw_bytes(data, encoding)?;

    let codec = match explicit_codec {
        Some(codec_id) => {
            if codec_id == CodecId::Scx1 && !encoding.is_integer() {
                CodecId::Zstd
            } else {
                codec_id
            }
        }
        None => select_codec(&raw_bytes, encoding),
    };

    Ok((encoding, codec))
}

/// Modality-aware variant of [`detect_value_encoding`]. Used by the
/// h5mu pipeline and any other writer that knows the biological
/// modality type of the data being encoded.
pub fn detect_value_encoding_for_modality(
    data: &[f32],
    explicit_codec: Option<CodecId>,
    modality_type: ModalityType,
) -> Result<(ValueEncoding, CodecId), CodecError> {
    let encoding = detect_value_encoding_only(data);
    let raw_bytes = values_to_raw_bytes(data, encoding)?;

    let codec = match explicit_codec {
        Some(codec_id) => {
            if codec_id == CodecId::Scx1 && !encoding.is_integer() {
                CodecId::Zstd
            } else {
                codec_id
            }
        }
        None => select_codec_for_modality(&raw_bytes, encoding, modality_type),
    };

    Ok((encoding, codec))
}
