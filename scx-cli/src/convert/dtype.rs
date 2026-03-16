// Integer detection, value encoding selection, and raw byte conversion

use scx_codec::{CodecId, ValueEncoding};

/// Check if all values in the data array are non-negative integers.
pub fn is_integer_data(data: &[f32]) -> bool {
    data.iter()
        .all(|&v| v.is_finite() && v >= 0.0 && v == v.floor())
}

/// Detect the best value encoding and codec for the data.
pub fn detect_value_encoding(data: &[f32]) -> (ValueEncoding, CodecId) {
    if !is_integer_data(data) {
        return (ValueEncoding::Float32, CodecId::Zstd);
    }

    // Find max value
    let max_val = data.iter().map(|&v| v as u32).max().unwrap_or(0);

    let encoding = if max_val <= 255 {
        ValueEncoding::Uint8
    } else if max_val <= 65535 {
        ValueEncoding::Uint16
    } else {
        ValueEncoding::Uint32
    };

    (encoding, CodecId::Scx1)
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
