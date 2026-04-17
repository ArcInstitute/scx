// Integer detection, value encoding selection, and raw byte conversion.
//
// The detect/encode primitives live in `scx_codec::value_encoding`. This
// module adds the codec auto-selection policy on top.

use scx_codec::{CodecId, ValueEncoding};
use scx_format::select_codec;

pub use scx_codec::value_encoding::{
    detect_value_encoding as detect_value_encoding_only, values_to_raw_bytes,
};

/// Detect the best value encoding and auto-select codec for the data.
/// When `explicit_codec` is Some, uses that codec (with Scx1→Zstd fallback for floats).
/// When None, auto-selects based on value distribution.
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
