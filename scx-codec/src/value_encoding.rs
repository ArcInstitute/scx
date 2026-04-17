//! Canonical `detect_value_encoding` and `values_to_raw_bytes` implementations.
//!
//! Until this module existed, three crates (`pyscx/anndata.rs`,
//! `scx-cli/convert/dtype.rs`, `scx-mtx/convert.rs`) each carried their own
//! structurally-identical copy of the detect + encode logic. Per finding
//! H11, all three now delegate here. Future tweaks to integer detection or
//! float-fallback policy happen in one place.
//!
//! Conventions:
//! - Integer detection uses `v.is_finite() && v >= 0.0 && v == v.floor()`.
//!   `is_finite()` rejects Infinity/NaN so they fall through to `Float32`
//!   (original pyscx finding 9.4).
//! - Max-value comparison is done in `f64` so values beyond `2^24` (f32's
//!   contiguous integer range) don't saturate (finding 9.1).
//! - `Float16` falls back to `Float32` bytes with no panic — the canonical
//!   rule is that callers should have re-encoded before reaching an
//!   f32-only path. This matches `scx-ops::compact` and tolerates the
//!   existing `pyscx::anndata::encode_values` preconditions. A
//!   `log::warn!` fires at most once per process when the fallback is
//!   taken, preserving the visibility the prior `scx-cli` `eprintln!`
//!   warning offered before this module absorbed the three call sites.

use std::sync::Once;

use byteorder::{LittleEndian, WriteBytesExt};

use crate::dispatch::ValueEncoding;

static FLOAT16_FALLBACK_WARNED: Once = Once::new();

/// Return true if every element is a non-negative finite integer-valued f32.
#[inline]
pub fn is_integer_data(data: &[f32]) -> bool {
    data.iter()
        .all(|&v| v.is_finite() && v >= 0.0 && v == v.floor())
}

/// Pick the narrowest integer encoding that holds every value, or fall back
/// to `Float32` if any value isn't a finite non-negative integer.
pub fn detect_value_encoding(data: &[f32]) -> ValueEncoding {
    if !is_integer_data(data) {
        return ValueEncoding::Float32;
    }
    // Compare as f64 so values > 2^24 don't saturate under f32.
    let max_val: f64 = data.iter().map(|&v| v as f64).fold(0.0f64, f64::max);
    if max_val <= u8::MAX as f64 {
        ValueEncoding::Uint8
    } else if max_val <= u16::MAX as f64 {
        ValueEncoding::Uint16
    } else {
        ValueEncoding::Uint32
    }
}

/// Serialize f32 values to LE raw bytes according to `encoding`.
///
/// `Float16` falls back to `Float32` bytes — see module docs for rationale.
pub fn values_to_raw_bytes(data: &[f32], encoding: ValueEncoding) -> Vec<u8> {
    match encoding {
        ValueEncoding::Uint8 => data.iter().map(|&v| v as u8).collect(),
        ValueEncoding::Uint16 => {
            let mut buf = Vec::with_capacity(data.len() * 2);
            for &v in data {
                buf.write_u16::<LittleEndian>(v as u16).unwrap();
            }
            buf
        }
        ValueEncoding::Uint32 => {
            let mut buf = Vec::with_capacity(data.len() * 4);
            for &v in data {
                buf.write_u32::<LittleEndian>(v as u32).unwrap();
            }
            buf
        }
        ValueEncoding::Float32 | ValueEncoding::Float16 => {
            if matches!(encoding, ValueEncoding::Float16) {
                FLOAT16_FALLBACK_WARNED.call_once(|| {
                    log::warn!(
                        "Float16 value encoding is not implemented; serializing as Float32 bytes. \
                         Re-encode upstream to avoid this fallback."
                    );
                });
            }
            let mut buf = Vec::with_capacity(data.len() * 4);
            for &v in data {
                buf.write_f32::<LittleEndian>(v).unwrap();
            }
            buf
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_uint8_for_small_ints() {
        assert_eq!(
            detect_value_encoding(&[0.0, 1.0, 255.0]),
            ValueEncoding::Uint8
        );
    }

    #[test]
    fn detect_uint16_at_boundary() {
        assert_eq!(
            detect_value_encoding(&[256.0, 65535.0]),
            ValueEncoding::Uint16
        );
    }

    #[test]
    fn detect_uint32_above_u16() {
        assert_eq!(
            detect_value_encoding(&[65536.0, 1e9]),
            ValueEncoding::Uint32
        );
    }

    #[test]
    fn detect_float32_for_fractional() {
        assert_eq!(detect_value_encoding(&[1.5, 2.0]), ValueEncoding::Float32);
    }

    #[test]
    fn detect_float32_for_nan() {
        assert_eq!(
            detect_value_encoding(&[1.0, f32::NAN]),
            ValueEncoding::Float32
        );
    }

    #[test]
    fn detect_float32_for_negative() {
        assert_eq!(detect_value_encoding(&[1.0, -1.0]), ValueEncoding::Float32);
    }

    #[test]
    fn detect_float32_for_infinity() {
        assert_eq!(
            detect_value_encoding(&[f32::INFINITY, 1.0]),
            ValueEncoding::Float32
        );
    }

    #[test]
    fn round_trip_uint8() {
        let data = [0.0_f32, 1.0, 128.0, 255.0];
        let bytes = values_to_raw_bytes(&data, ValueEncoding::Uint8);
        let decoded: Vec<f32> = bytes.iter().map(|&b| b as f32).collect();
        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn round_trip_uint16() {
        let data = [0.0_f32, 256.0, 65535.0];
        let bytes = values_to_raw_bytes(&data, ValueEncoding::Uint16);
        let decoded: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as f32)
            .collect();
        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn round_trip_float32() {
        let data = [0.0_f32, 1.5, -2.75, std::f32::consts::PI];
        let bytes = values_to_raw_bytes(&data, ValueEncoding::Float32);
        let decoded: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn float16_falls_back_to_float32_bytes() {
        let data = [1.5_f32, 2.25];
        let f32_bytes = values_to_raw_bytes(&data, ValueEncoding::Float32);
        let f16_bytes = values_to_raw_bytes(&data, ValueEncoding::Float16);
        assert_eq!(f32_bytes, f16_bytes);
    }
}
