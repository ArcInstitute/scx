// Integer detection, value encoding selection, and raw byte conversion.
//
// The detect/encode primitives live in `scx_codec::value_encoding`. This
// module adds the codec auto-selection policy on top.

use scx_codec::{CodecId, ValueEncoding};
use scx_format::{select_codec, select_codec_for_modality, ModalityType};

pub use scx_codec::value_encoding::{
    detect_value_encoding as detect_value_encoding_only, values_to_raw_bytes,
};

/// Detect the best value encoding and auto-select codec for the data.
/// When `explicit_codec` is Some, uses that codec (with Scx1→Zstd fallback for floats).
/// When None, auto-selects based on value distribution. Single-modality
/// callers behave as if every shard is RNA (matches v1 / pre-Phase-E
/// output bit-for-bit).
pub fn detect_value_encoding(
    data: &[f32],
    explicit_codec: Option<CodecId>,
) -> (ValueEncoding, CodecId) {
    let encoding = detect_value_encoding_only(data);
    let raw_bytes = values_to_raw_bytes(data, encoding);

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

    (encoding, codec)
}

/// Modality-aware variant of [`detect_value_encoding`]. Used by the
/// h5mu pipeline and any other writer that knows the biological
/// modality type of the data being encoded.
pub fn detect_value_encoding_for_modality(
    data: &[f32],
    explicit_codec: Option<CodecId>,
    modality_type: ModalityType,
) -> (ValueEncoding, CodecId) {
    let encoding = detect_value_encoding_only(data);
    let raw_bytes = values_to_raw_bytes(data, encoding);

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

    (encoding, codec)
}
