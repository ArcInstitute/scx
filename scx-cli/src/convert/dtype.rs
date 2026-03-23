// Integer detection, value encoding selection, and raw byte conversion

use scx_codec::{CodecId, ValueEncoding};
use scx_format::select_codec;

/// Check if all values in the data array are non-negative integers.
pub fn is_integer_data(data: &[f32]) -> bool {
    data.iter()
        .all(|&v| v.is_finite() && v >= 0.0 && v == v.floor())
}

/// Detect the best value encoding for the data.
pub fn detect_value_encoding_only(data: &[f32]) -> ValueEncoding {
    if !is_integer_data(data) {
        return ValueEncoding::Float32;
    }

    // Compare as f64 to avoid precision loss for values > 2^24 and
    // saturation for values > u32::MAX (finding 8.1).
    let max_val: f64 = data.iter().map(|&v| v as f64).fold(0.0f64, f64::max);

    if max_val <= 255.0 {
        ValueEncoding::Uint8
    } else if max_val <= 65535.0 {
        ValueEncoding::Uint16
    } else {
        ValueEncoding::Uint32
    }
}

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

/// Convert f32 data to raw LE bytes according to the value encoding.
pub fn values_to_raw_bytes(data: &[f32], encoding: ValueEncoding) -> Vec<u8> {
    match encoding {
        ValueEncoding::Uint8 => data.iter().map(|&v| v as u8).collect(),
        ValueEncoding::Uint16 => {
            let mut buf = Vec::with_capacity(data.len() * 2);
            for &v in data {
                buf.extend_from_slice(&(v as u16).to_le_bytes());
            }
            buf
        }
        ValueEncoding::Uint32 => {
            let mut buf = Vec::with_capacity(data.len() * 4);
            for &v in data {
                buf.extend_from_slice(&(v as u32).to_le_bytes());
            }
            buf
        }
        ValueEncoding::Float32 => {
            let mut buf = Vec::with_capacity(data.len() * 4);
            for &v in data {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            buf
        }
        ValueEncoding::Float16 => {
            unimplemented!("Float16 encoding not supported in Phase 1")
        }
    }
}
